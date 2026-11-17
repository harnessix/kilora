# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kilora.compute import sampling_progress
from kilora.errors import KiLoRAError
from kilora.families import block_list
from kilora.recording import (
    MAP_RIDGE,
    BlockFrames,
    BlockMaps,
    BlockStreamRecorder,
    PathRecorder,
    clamp_errors,
    hierarchy_errors,
    noise_weight,
    observing_blocks,
    prediction_counts,
    prediction_note,
    record_hierarchy,
)

SIGMAS = (4.0, 3.0, 2.0, 1.0, 0.5, 0.25)
WIDTH, RANK, LEVELS = 6, 3, 4


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(WIDTH, WIDTH)

    def forward(self, x):
        return torch.tanh(self.linear(x))


class Blocks(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.blocks = nn.ModuleList(Block() for _ in range(LEVELS))
        with torch.no_grad():
            for block in self.blocks:
                linear = block.linear
                linear.weight.copy_(torch.randn(WIDTH, WIDTH, generator=gen) / WIDTH**0.5)
                linear.bias.copy_(torch.randn(WIDTH, generator=gen) * 0.1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def _frames(seed=1):
    gen = torch.Generator().manual_seed(seed)
    outputs = {
        f"blocks.{level}.{name}": torch.randn(2, WIDTH, generator=gen)
        for level in range(LEVELS)
        for name in ("attn.out", "mlp.out")
    }
    return BlockFrames(outputs, k=RANK)


def _generate(model, recorder=None, seed=2):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 5, WIDTH, generator=gen)
    outs = []
    with torch.no_grad():
        for i, sigma in enumerate(SIGMAS):
            xs = x * (1 + sigma) + 0.1 * i
            with sampling_progress(i / (len(SIGMAS) - 1)):
                if recorder is None:
                    outs.append(model(xs))
                    continue
                recorder.set_sigma(sigma)
                with observing_blocks(list(enumerate(model.blocks)), recorder):
                    outs.append(model(xs))
    return outs, x


def _recorded(seed=2):
    model = Blocks()
    frames = _frames()
    recorder = BlockStreamRecorder(frames)
    _generate(model, recorder, seed)
    return model, frames, recorder.finish_reads()


def _hand_map(reads, level):
    low = torch.stack([r.reads[level] for r in reads], 1).double()
    high = torch.stack([r.reads[level + 1] for r in reads], 1).double()
    auto = high @ high.t()
    ridge = MAP_RIDGE * auto.diagonal().mean()
    return low @ high.t() @ torch.linalg.inv(auto + ridge * torch.eye(auto.shape[0]))


def test_reads_are_the_frame_of_the_pooled_block_output():
    model, frames, reads = _recorded()
    assert [r.step for r in reads] == list(range(1, len(SIGMAS) + 1))
    assert [r.sigma for r in reads] == list(SIGMAS)
    frame = frames.frame("blocks", 2, WIDTH)
    assert frame.shape == (RANK, WIDTH)
    assert torch.allclose(frame @ frame.t(), torch.eye(RANK, dtype=torch.float64), atol=1e-10)
    _, x = _generate(model)
    h = x * (1 + SIGMAS[0])
    with torch.no_grad():
        for block in model.blocks[:3]:
            h = block(h)
    expected = frame @ h.double().reshape(-1, WIDTH).mean(0)
    assert torch.allclose(reads[0].reads[2], expected, atol=1e-10)


def test_maps_equal_the_ridge_solution_by_hand():
    _, _, reads = _recorded()
    maps = BlockMaps()
    maps.update(reads)
    assert sorted(maps.sums) == [0, 1, 2]
    for level in range(LEVELS - 1):
        assert torch.allclose(maps.map(level), _hand_map(reads, level), atol=1e-9)


def test_maps_average_over_generations():
    _, _, first = _recorded(seed=2)
    _, _, second = _recorded(seed=3)
    maps = BlockMaps(decay=0.9)
    maps.update(first)
    one = {k: v.clone() for k, v in maps.sums[1].items()}
    maps.update(second)
    fresh = BlockMaps()
    fresh.update(second)
    for name, value in maps.sums[1].items():
        assert torch.allclose(value, 0.9 * one[name] + 0.1 * fresh.sums[1][name])


def test_hierarchical_error_and_precision():
    _, _, reads = _recorded()
    maps = BlockMaps()
    maps.update(reads)
    errors = hierarchy_errors(maps, reads)
    assert sorted(errors) == ["blocks.0", "blocks.1", "blocks.2"]
    for level in range(LEVELS - 1):
        g = _hand_map(reads, level)
        residual = sum(
            float((r.reads[level] - g @ r.reads[level + 1]).pow(2).sum()) for r in reads
        ) / len(reads)
        assert maps.residual(level) == pytest.approx(residual, rel=1e-6)
        for entry, read in zip(errors[f"blocks.{level}"], reads, strict=True):
            expected = read.reads[level] - g @ read.reads[level + 1]
            assert torch.allclose(entry.error, expected, atol=1e-9)
            pi = noise_weight(read.sigma) / residual
            assert float(entry.precision) == pytest.approx(pi, rel=1e-5)


def test_without_maps_no_hierarchical_error_until_one_generation_ran():
    _, _, reads = _recorded()
    maps = BlockMaps.from_tensors({})
    assert not maps.sums
    stats = SimpleNamespace(block_reads=reads)
    record_hierarchy(maps, stats)
    assert stats.hier_errors == {} and stats.clamp_errors == {}
    assert sorted(maps.sums) == [0, 1, 2]
    record_hierarchy(maps, stats)
    assert stats.hier_errors == {}
    again = SimpleNamespace(block_reads=reads)
    record_hierarchy(maps, again)
    assert again.hier_errors


def test_maps_survive_a_save_and_load():
    from kilora.live_imports import ImportHistory

    _, _, reads = _recorded()
    store = ImportHistory([])
    store.block_maps.update(reads)
    branch = store.fork()
    store.checkout(branch)
    tensors = store.state_tensors()
    assert any(k.startswith(BlockMaps.PREFIX) for k in tensors)
    loaded = ImportHistory([])
    loaded.load_tensors(tensors)
    for level in range(LEVELS - 1):
        assert torch.equal(loaded.block_maps.map(level), store.block_maps.map(level))
    old = {k: v for k, v in tensors.items() if not k.startswith(BlockMaps.PREFIX)}
    loaded.load_tensors(old)
    assert not loaded.block_maps.sums
    assert loaded.forest.current == branch
    loaded.checkout("main")
    for level in range(LEVELS - 1):
        assert torch.equal(loaded.block_maps.map(level), store.block_maps.map(level))
    from kilora.live import _fixed_part

    assert set(_fixed_part(tensors)) == set(_fixed_part(old))


def test_clamped_top_and_its_downward_step(monkeypatch):
    _, _, reads = _recorded()
    maps = BlockMaps()
    maps.update(reads)
    gen = torch.Generator().manual_seed(9)
    targets = {
        r.sigma: {level: torch.randn(RANK, generator=gen).double() for level in (2, 3)}
        for r in reads
    }
    errors = clamp_errors(maps, reads, targets, rho=0.7, top=2)
    assert sorted(errors) == [f"blocks.{level}" for level in range(LEVELS)]
    for index, read in enumerate(reads):
        star, w = targets[read.sigma], noise_weight(read.sigma)
        top = errors["blocks.3"][index]
        assert torch.allclose(top.error, star[3] - read.reads[3].double())
        assert float(top.precision) == pytest.approx(w)
        second = errors["blocks.2"][index]
        assert torch.allclose(second.error, star[2] - read.reads[2].double())
        assert float(second.precision) == pytest.approx(w * 0.7)
        expected = maps.map(1) @ star[2]
        down = errors["blocks.1"][index]
        assert torch.allclose(down.predicted, expected, atol=1e-10)
        assert torch.allclose(down.error, expected - read.reads[1].double(), atol=1e-10)
        assert float(down.precision) == pytest.approx(w * 0.7**2)
        lowest = errors["blocks.0"][index]
        assert torch.allclose(lowest.predicted, maps.map(0) @ expected, atol=1e-10)
        assert float(lowest.precision) == pytest.approx(w * 0.7**3)
    counts = prediction_counts({}, {}, {}, errors)
    assert counts["eps_clamp_steps"] == len(reads) and counts["eps_clamp_nodes"] == LEVELS
    assert counts["eps_steps"] == len(reads) and counts["eps_nodes"] == LEVELS
    monkeypatch.delenv("KILORA_PREDICTION_DEBUG", raising=False)
    note = prediction_note({}, {}, {}, errors, 0.5)
    assert "eps_" not in note and "target_forward_s=0.50" in note
    monkeypatch.setenv("KILORA_PREDICTION_DEBUG", "1")
    note = prediction_note({}, {}, {}, errors, 0.5)
    assert f"eps_clamp_nodes={LEVELS}" in note and "target_forward_s=0.50" in note


def test_target_forward_is_cached():
    from kilora.comfyui import live_write

    model = Blocks()
    frames = _frames()
    calls = []

    def apply_model(x, t, **cond):
        calls.append(float(t[0]))
        return model(x)

    handle = SimpleNamespace(
        session=SimpleNamespace(record_prediction=True),
        block_modules=list(enumerate(model.blocks)),
        apply_model=apply_model,
        conditioning={},
        seed=7,
        timesteps=[0.9, 0.5, 0.2],
        process_in=None,
        sampling=None,
        patcher=None,
    )
    blocks = BlockStreamRecorder(frames)
    latent = torch.randn(1, 5, WIDTH, generator=torch.Generator().manual_seed(4))
    first, _ = live_write.target_reads(handle, blocks, latent)
    assert calls == pytest.approx([0.9, 0.5, 0.2], rel=1e-6)
    assert set(first) == {0.9, 0.5, 0.2}
    cache = handle.session.__dict__["_target_reads"]
    assert {key[1] for key in cache} == {0.9, 0.5, 0.2}
    assert all(set(reads) == set(range(LEVELS)) for reads in first.values())
    record = SimpleNamespace(extra={})
    live_write._attach_targets(handle, blocks, latent, record)
    assert len(calls) == 3
    assert record.extra["target_reads"].keys() == first.keys()
    for sigma, reads in first.items():
        for level, value in reads.items():
            assert torch.equal(record.extra["target_reads"][sigma][level], value)
    handle.seed = 8
    live_write.target_reads(handle, blocks, latent)
    assert len(calls) == 6
    noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(7))
    x_t = 0.5 * noise + 0.5 * latent
    with torch.no_grad():
        h = x_t
        for block in model.blocks:
            h = block(h)
    expected = frames.frame("blocks", 3, WIDTH) @ h.double().reshape(-1, WIDTH).mean(0)
    assert torch.allclose(first[0.5][3], expected, atol=1e-6)
    handle.session.record_prediction = False
    off = SimpleNamespace(extra={})
    live_write._attach_targets(handle, blocks, latent, off)
    assert off.extra == {}


def test_family_block_list():
    attr, modules = block_list("anima", Blocks())
    assert attr == "blocks" and [i for i, _ in modules] == list(range(LEVELS))
    with pytest.raises(KiLoRAError, match="block list"):
        block_list("flux", Blocks())


def test_chunk_recorder_outputs_equal_with_block_reads():
    model = Blocks()
    expected, _ = _generate(model)
    recorder = PathRecorder(
        {"mount": torch.eye(3)}, {"mount": torch.eye(3)}, blocks=BlockStreamRecorder(_frames())
    )
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(2, 5, WIDTH, generator=gen)
    actual = []
    with torch.no_grad():
        for i, sigma in enumerate(SIGMAS):
            recorder.set_sigma(sigma)
            with (
                sampling_progress(i / (len(SIGMAS) - 1)),
                recorder.recording(),
                observing_blocks(list(enumerate(model.blocks)), recorder.blocks),
            ):
                actual.append(model(x * (1 + sigma) + 0.1 * i))
    for a, b in zip(actual, expected, strict=True):
        assert torch.equal(a, b)
    stats = recorder.finish()
    assert len(stats.block_reads) == len(SIGMAS)
    assert all(not m._forward_hooks for m in model.blocks)


@pytest.fixture
def imported_session(tmp_path, monkeypatch):
    from test_import_modes import LAYERS, Toy, _lora

    from kilora.live import SESSIONS
    from kilora.live_imports import imported_session_for
    from kilora.multi_import import import_loras

    SESSIONS.clear()
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    model = Toy().eval()
    paths = [tmp_path / f"{i}.safetensors" for i in range(2)]
    for i, path in enumerate(paths):
        _lora(model, path, rank=4, seed=i, alpha=4.0, layers=LAYERS)
    sources = [
        dict(
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            path=str(p),
            strength=1.0,
            axis="both",
        )
        for p in paths
    ]
    routed = import_loras([str(p) for p in paths], model).adapter_set
    shapes = {t: list(e.down.shape) for t, e in routed.experts.items()}
    session = imported_session_for(routed, sources, "toy", shapes, tmp_path / "saves")
    yield session, model
    SESSIONS.clear()


def test_imported_block_reads_leave_outputs_and_state_equal(imported_session):
    from test_import_modes import DIM, Toy

    from kilora import compose
    from kilora.live_imports import ImportEditor, import_source_frames

    session, model = imported_session
    session.align_for_moves()
    weights = {}
    for target in session.state:
        try:
            weights[target] = model.get_submodule(target).weight
        except AttributeError:
            continue
    import_source_frames(session, weights)
    before = {t: {k: v.clone() for k, v in parts.items()} for t, parts in session.state.items()}
    base = Toy().eval()
    adapter = compose(base, session.sets())
    gen = torch.Generator().manual_seed(5)
    xs = [torch.randn(1, 3, DIM, generator=gen) for _ in SIGMAS]
    with torch.no_grad():
        expected = [base(x) for x in xs]
        recorder = ImportEditor(session).recorder(session)
        assert recorder.blocks is not None
        actual = []
        for sigma, x in zip(SIGMAS, xs, strict=True):
            recorder.set_sigma(sigma)
            with (
                recorder.recording(),
                observing_blocks(list(enumerate(base.blocks)), recorder.blocks),
            ):
                actual.append(base(x))
    adapter.remove()
    for a, b in zip(actual, expected, strict=True):
        assert torch.equal(a, b)
    for target, parts in before.items():
        for key, value in parts.items():
            assert torch.equal(session.state[target][key], value)
    stats = recorder.finish()
    assert len(stats.block_reads) == len(SIGMAS)
    assert all(read.reads for read in stats.block_reads)
    assert set(stats.block_reads[0].reads) == {0}
    assert all(bool(torch.isfinite(read.reads[0]).all()) for read in stats.block_reads)
    assert stats.cross_errors
    session.record_prediction = False
    assert ImportEditor(session).recorder(session) is None
