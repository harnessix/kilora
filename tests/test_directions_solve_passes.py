# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import copy
from dataclasses import replace

import pytest
import torch
from test_directions_contrast import Block, Model, _samples, _set_change, _settings
from test_directions_layout import _model

from kilora.library import contrast


@torch.no_grad()
def _reference_solve(
    model,
    samples,
    pairs,
    *,
    settings=None,
    mounts=None,
    model_sha,
    license,
    partial=None,
    progress=None,
    target_source=None,
    refit_round=0,
):

    settings = settings or contrast.Settings()
    settings.validate()
    pairs = [p if isinstance(p, contrast.PromptPair) else contrast.PromptPair(*p) for p in pairs]
    if not pairs or not samples:
        raise ValueError("at least one pair and trajectory sample are required")
    requested = list(target_source.mounts) if mounts is None and target_source else mounts
    selected = contrast.resolve_mounts(model, requested)
    if target_source is not None and set(selected) - target_source.mounts.keys():
        raise ValueError("effect mounts must be routed mounts of the source file")
    excluded = {}
    if any(p.target != p.student for p in pairs):
        excluded = {
            n: contrast._EXCLUSION for n in selected if n.endswith(contrast._CAPTION_MOUNTS)
        }
    fitted = [n for n in selected if n not in excluded]
    if not fitted:
        raise ValueError("no mounts remain after caption exclusions")
    metadata = contrast.json.loads(
        contrast.json.dumps(
            {
                **contrast.asdict(settings),
                "pairs": [contrast.asdict(p) for p in pairs],
                "mounts": selected,
                "excluded_mounts": excluded,
                "model_sha": model_sha,
                "license": license,
                "family": "anima",
                "mode": "joint",
                "sample_count": len(samples),
                "license_notice": contrast.get_preset("anima").notice,
            }
        )
    )
    if settings.refit_rounds:
        metadata["refit_round"] = refit_round
    if target_source is not None:
        metadata.update(
            {
                "target_source": "lora",
                "source": target_source.metadata,
                "license": target_source.license,
                "prompts": [{"target": p.target, "student": p.student} for p in pairs],
            }
        )
    result, start = contrast.ContrastResult({}, {**metadata, "fit": {}, "block_gaps": {}}), 0
    if partial is not None and contrast.Path(partial).exists():
        result, start = contrast._load_partial(partial, metadata)
    result.metadata["exact_fits"] = sum(m.exact_fit for m in result.mounts.values())
    modules = dict(model.named_modules())
    indices = sorted({int(n.split(".")[1]) for n in fitted})
    if start < 0 or start > len(indices):
        raise ValueError("partial file has an invalid next block")
    completed = {n for n in fitted if int(n.split(".")[1]) in indices[:start]}
    if set(result.mounts) != completed:
        raise ValueError("partial file does not contain exactly the completed blocks")
    layout = None
    if settings.layout_source == "plain":
        from kilora.library.layout import PlainLayout

        layout = PlainLayout(model)
    changes = contrast._Changes(model)
    target_changes = contrast._Changes(model)

    @contrast.contextmanager
    def student_chain(enabled):
        with changes.active(enabled), target_changes.active(not enabled):
            yield

    training = {module: module.training for module in model.modules()}
    model.eval()
    try:
        if layout is not None:
            layout.install()
        if target_source is not None:
            for name, solution in target_source.mounts.items():
                target_changes.add(name, solution)
        for name, solution in result.mounts.items():
            changes.add(name, solution)
        for position in range(start, len(indices)):
            index = indices[position]
            prefix = f"transformer_blocks.{index}"
            block = modules[prefix]
            captured = []
            device = next(model.parameters()).device
            for sample in samples:
                target_call = contrast._map_tensors(sample.target, lambda t, d=device: t.to(d))
                student_call = contrast._map_tensors(sample.student, lambda t, d=device: t.to(d))
                if layout is None:
                    with student_chain(False):
                        target = contrast._capture(model, block, target_call)
                    with student_chain(True):
                        student = contrast._capture(model, block, student_call)
                else:
                    with student_chain(True), layout.active("record"):
                        student = contrast._capture(model, block, student_call)
                    with student_chain(False), layout.active("replay"):
                        target = contrast._capture(model, block, target_call)
                captured.append((target, student))
            for group in contrast._GROUPS:
                names = [f"{prefix}.{suffix}" for suffix in group]
                names = [n for n in names if n in fitted]
                if not names:
                    continue
                statistics = {}
                for target, student in captured:
                    if layout is None:
                        with student_chain(False):
                            desired = contrast._observe(block, target, names, modules)
                        with student_chain(True):
                            actual = contrast._observe(block, student, names, modules)
                    else:
                        with student_chain(True), layout.active("record"):
                            actual = contrast._observe(block, student, names, modules)
                        with student_chain(False), layout.active("replay"):
                            desired = contrast._observe(block, target, names, modules)
                    for name in names:
                        x, y = actual[name]
                        target_y = desired[name][1]
                        if y.shape != target_y.shape:
                            raise ValueError(f"{name}: target and student token shapes differ")
                        x = x.reshape(-1, x.shape[-1])
                        difference = (target_y - y).reshape(-1, y.shape[-1])
                        with contrast._matmul_allow_tf32(settings.stats_precision == "tf32"):
                            c, s = x.T @ x, difference.T @ x
                        if name in statistics:
                            statistics[name][0].add_(c)
                            statistics[name][1].add_(s)
                        else:
                            statistics[name] = (c, s)
                for name in names:
                    solution = contrast.fit_mount(
                        *statistics.pop(name),
                        settings.rank,
                        ridge=settings.ridge,
                        fit_dtype=settings.fit_dtype,
                    )
                    result.mounts[name] = solution
                    result.metadata["exact_fits"] += int(solution.exact_fit)
                    result.metadata["fit"][name] = {
                        "ridge": solution.ridge,
                        "fit_share": solution.fit_share,
                    }
                    if solution.exact_fit:
                        result.metadata["fit"][name]["exact_fit"] = True
                    if solution.svd_fallback:
                        result.metadata["fit"][name]["svd_fallback"] = True
                        result.metadata["svd_fallbacks"] = sum(
                            int(record.get("svd_fallback", False))
                            for record in result.metadata["fit"].values()
                        )
                    changes.add(name, solution)
            numerator = denominator = 0.0
            for target, student in captured:
                if layout is None:
                    with student_chain(False):
                        wanted = contrast._tensor_output(contrast._run_block(block, target)).float()
                    with student_chain(True):
                        got = contrast._tensor_output(contrast._run_block(block, student)).float()
                else:
                    with student_chain(True), layout.active("record"):
                        got = contrast._tensor_output(contrast._run_block(block, student)).float()
                    with student_chain(False), layout.active("replay"):
                        wanted = contrast._tensor_output(contrast._run_block(block, target)).float()
                numerator += float((wanted - got).square().sum())
                denominator += float(wanted.square().sum())
            gap = contrast.math.sqrt(numerator / max(denominator, 1e-30))
            result.metadata["block_gaps"][str(index)] = gap
            if partial is not None:
                contrast._save_partial(partial, result, position + 1)
            if progress is not None:
                progress(f"Block {index}: relative gap {gap:.6g}")
    finally:
        changes.close()
        target_changes.close()
        if layout is not None:
            layout.close()
        for module, was_training in training.items():
            module.training = was_training
    return result


class CountingModel(Model):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0

    def forward(self, *args, **kwargs):
        self.forward_calls += 1
        return super().forward(*args, **kwargs)


def _run(solver, model, samples, **kwargs):
    return solver(
        model,
        samples,
        [("same", "same")],
        model_sha="toy",
        license="Apache-2.0",
        settings=replace(kwargs.pop("settings", _settings()), replay_batch=1),
        **kwargs,
    )


def _equal(actual, expected):
    assert actual.metadata == expected.metadata
    assert actual.mounts.keys() == expected.mounts.keys()
    for name in actual.mounts:
        assert torch.allclose(
            actual.mounts[name].delta(),
            expected.mounts[name].delta(),
            atol=1e-6,
            rtol=0,
        )


@pytest.mark.parametrize("plain", [False, True])
@pytest.mark.parametrize("effect", [False, True])
def test_matches_previous_solver(plain, effect):
    model = _model() if plain else Model()
    delta = _set_change(model)
    _set_change(model, block=1)
    source = None
    if effect:
        source = contrast.EffectSource(
            {
                f"transformer_blocks.{i}.attn1.to_q": contrast.MountSolution(
                    torch.eye(4),
                    delta.T,
                    0,
                    1,
                )
                for i in range(2)
            },
            {"file": "toy"},
            "Apache-2.0",
        )
    settings = replace(_settings(), layout_source="plain" if plain else "target")
    samples = _samples(count=2)
    expected = _run(
        _reference_solve,
        copy.deepcopy(model),
        samples,
        settings=settings,
        target_source=source,
    )
    before = model(**samples[0].student)[0].detach().clone()
    actual = _run(contrast.solve, model, samples, settings=settings, target_source=source)
    _equal(actual, expected)
    assert torch.equal(before, model(**samples[0].student)[0])


@pytest.mark.parametrize("sparse", [False, True])
def test_capture_count_and_intermediate_blocks(sparse, capsys):
    model = CountingModel()
    model.transformer_blocks.append(Block(4))
    _set_change(model)
    samples = _samples(count=3)
    mounts = (
        None
        if not sparse
        else [
            "transformer_blocks.0.attn1.to_q",
            "transformer_blocks.2.attn1.to_q",
        ]
    )
    expected = _run(_reference_solve, copy.deepcopy(model), samples, mounts=mounts)
    actual = _run(contrast.solve, model, samples, mounts=mounts)
    assert model.forward_calls == 2 * len(samples)
    assert "Model passes: 6" in capsys.readouterr().out
    _equal(actual, expected)


@pytest.mark.parametrize("plain", [False, True])
def test_resume_recomputes_prefix_once(tmp_path, plain):
    model = _model() if plain else CountingModel()
    _set_change(model)
    samples = _samples(count=2)
    settings = replace(_settings(), layout_source="plain" if plain else "target")
    partial = tmp_path / "solve.partial.safetensors"

    def interrupt(message):
        raise InterruptedError(message)

    with pytest.raises(InterruptedError):
        _run(contrast.solve, model, samples, settings=settings, partial=partial, progress=interrupt)
    if not plain:
        model.forward_calls = 0
    resumed = _run(contrast.solve, model, samples, settings=settings, partial=partial)
    if not plain:
        assert model.forward_calls == 2 * len(samples)
    full = _run(contrast.solve, model, samples, settings=settings)
    _equal(resumed, full)
    for name in full.mounts:
        assert torch.equal(resumed.mounts[name].down, full.mounts[name].down)
        assert torch.equal(resumed.mounts[name].up, full.mounts[name].up)
