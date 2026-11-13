# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import copy
import errno
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from test_comfyui import FakeBaseModel, FakePatcher
from test_directions_contrast import Block, Linear, Model, _samples, _settings
from torch import nn
from torch.nn import functional as F

from kilora.comfyui import nodes as N
from kilora.comfyui import prompt_solve as P
from kilora.comfyui.trigger_words import trigger_words_of
from kilora.library import contrast
from kilora.lora_import import import_lora
from kilora.presets import anima_name, pack_license

H3_SET = {
    "subjects": ["a tree beside a lake", "a small house on a hill"],
    "concepts": [
        {"label": "ink", "style": "thin ink lines", "cue": "ink"},
        {"label": "wash", "style": "soft watercolor", "cue": "wash"},
    ],
}


def prompt_set(data=None):
    return P.PromptSet(contrast.prompt_set_from_data(data or H3_SET))


def test_prompt_set_fields_json_shared_subjects_and_pairs(tmp_path):
    node = P.KiLoRAPromptSet()
    fields, report = node.build("a tree\n\na house\n", "thin lines", "ink", "line")
    assert fields.groups["line"] == [
        contrast.PromptPair("a tree, thin lines", "a tree, ink", "line", "ink"),
        contrast.PromptPair("a house, thin lines", "a house, ink", "line", "ink"),
    ]
    assert "line: 2 pairs" in report
    shared, _ = node.build("ignored", "", "", "", json=json.dumps(H3_SET))
    assert len(shared.groups["ink"]) == len(shared.groups["wash"]) == 2
    local = copy.deepcopy(H3_SET)
    local["concepts"][1]["subjects"] = ["a cat"]
    current, _ = node.build("", "", "", "", json=json.dumps(local))
    assert len(current.groups["wash"]) == 1
    explicit = {"pairs": [asdict(fields.groups["line"][0])], "concepts": "ignored"}
    direct, _ = node.build("", "", "", "", json=json.dumps(explicit))
    assert direct.groups["line"] == fields.groups["line"][:1]
    chained, _ = node.build("a bird", "oil paint", "oil", "oil", prompt_set=shared)
    assert list(chained.groups) == ["ink", "wash", "oil"]
    assert list(shared.groups) == ["ink", "wash"]
    path = tmp_path / "set.json"
    path.write_text(json.dumps(local))
    assert contrast.read_prompt_set(path) == current.groups


@pytest.mark.parametrize(
    "data, message",
    [
        ({}, "nonempty concepts"),
        ({"concepts": [None]}, "each concept"),
        ({"concepts": [{"label": "../bad", "style": "", "subjects": ["tree"]}]}, "labels"),
        ({"concepts": [{"label": "a", "style": "", "subjects": []}]}, "subjects"),
        ({"pairs": []}, "nonempty list"),
        ({"pairs": [{}]}, "labels"),
        ({"pairs": [{"concept": "a", "target": "", "student": "x", "cue": "a"}]}, "nonempty"),
    ],
)
def test_bad_prompt_sets_raise(data, message):
    with pytest.raises(ValueError, match=message):
        contrast.prompt_set_from_data(data)


def test_bad_json_and_duplicate_chain_raise():
    with pytest.raises(ValueError, match="JSON"):
        P.KiLoRAPromptSet().build("", "", "", "", json="{")
    with pytest.raises(ValueError, match="duplicate concept label"):
        P.KiLoRAPromptSet().build("tree", "ink", "ink", "ink", prompt_set=prompt_set())


class HostBlock(nn.Module):
    def __init__(self):
        super().__init__()
        source = Block(4)
        self.self_attn, self.cross_attn = nn.Module(), nn.Module()
        for host, original in ((self.self_attn, source.attn1), (self.cross_attn, source.attn2)):
            host.q_proj, host.k_proj, host.v_proj = original.to_q, original.to_k, original.to_v
            host.output_proj = original.to_out[0]
        self.mlp = nn.Module()
        self.mlp.layer1, self.mlp.layer2 = source.ff.net[0].proj, source.ff.net[2]

    def forward(self, hidden_states, encoder_hidden_states, target=False):
        for module in self.modules():
            if isinstance(module, (Linear, CastLinear)):
                module.target = target
        x = hidden_states
        for attention, context in ((self.self_attn, x), (self.cross_attn, encoder_hidden_states)):
            context = x if attention is self.self_attn else context
            q, k, v = attention.q_proj(x), attention.k_proj(context), attention.v_proj(context)
            if k.shape != q.shape:
                k = k.mean(-2, keepdim=True).expand_as(q)
                v = v.mean(-2, keepdim=True).expand_as(q)
            x = attention.output_proj((q + k + v) / 3)
        return self.mlp.layer2(self.mlp.layer1(x))


class HostModel(nn.Module):
    kilora_family = "anima"

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([HostBlock(), HostBlock()])

    def forward(self, hidden_states, encoder_hidden_states=None, target=False):
        x = hidden_states
        if encoder_hidden_states is None:
            encoder_hidden_states = x
        for block in self.blocks:
            x = block(x, encoder_hidden_states=encoder_hidden_states, target=target)
        return (x,)


class CastLinear(nn.Module):
    in_features = out_features = 4

    def __init__(self):
        super().__init__()
        self.register_buffer("packed", torch.eye(4, dtype=torch.int8))
        self.register_buffer("target_delta", torch.zeros(4, 4))
        self.target = False
        self.casts = 0

    @property
    def weight(self):
        raise AssertionError("The solve must use the layer's dequantizing forward")

    def forward(self, x):
        self.casts += 1
        result = F.linear(x, self.packed.to(device=x.device, dtype=x.dtype))
        return result + F.linear(x, self.target_delta) if self.target else result


def host_patcher():
    model = HostModel()
    model.blocks[0].self_attn.q_proj.target_delta[0, 1] = 0.3
    patcher = FakePatcher(FakeBaseModel(model))
    patcher.load_device = torch.device("cpu")
    return patcher


@pytest.fixture
def world(tmp_path, monkeypatch):
    root, temporary = tmp_path / "loras", tmp_path / "temp"
    root.mkdir()
    temporary.mkdir()
    monkeypatch.setattr(N.FOLDERS, "first_dir", lambda folder: str(root))
    monkeypatch.setattr(N.FOLDERS, "path", lambda folder, name: str(root / name))
    counts = SimpleNamespace(loads=0, cache=0, trajectories=0, encoded=[], bars=[], load_options=[])
    mm = ModuleType("comfy.model_management")

    def load(models, memory_required=0, force_full_load=False):
        counts.loads += 1
        counts.load_options.append((memory_required, force_full_load))

    def clear():
        counts.cache += 1

    mm.get_torch_device = lambda: torch.device("cpu")
    mm.get_free_memory = lambda device: 1024**4
    mm.load_models_gpu = load
    mm.soft_empty_cache = clear
    mm.throw_exception_if_processing_interrupted = lambda: None
    utils = ModuleType("comfy.utils")

    class Progress:
        def __init__(self, total):
            self.total, self.done = total, 0
            counts.bars.append(self)

        def update(self, count):
            self.done += count

    utils.ProgressBar = Progress
    comfy = ModuleType("comfy")
    comfy.model_management, comfy.utils = mm, utils
    for name, module in (("comfy", comfy), ("comfy.model_management", mm), ("comfy.utils", utils)):
        monkeypatch.setitem(sys.modules, name, module)
    folders = ModuleType("folder_paths")
    folders.get_temp_directory = lambda: str(temporary)
    monkeypatch.setitem(sys.modules, "folder_paths", folders)

    class Clip:
        def tokenize(self, text):
            counts.encoded.append(text)
            return text

        def encode_from_tokens_scheduled(self, tokens):
            return [[torch.zeros(1, 2, 4), {"text": tokens}]]

    def trajectories(patcher, captions, pairs, settings, progress):
        counts.trajectories += 1
        progress.update(len(pairs) * len(settings.seeds))
        return _samples(count=2)

    monkeypatch.setattr(P, "trajectory_samples", trajectories)
    return SimpleNamespace(
        model=host_patcher(),
        clip=Clip(),
        root=root,
        temp=temporary,
        counts=counts,
        mm=mm,
        kwargs={"name": "solved", "rank": 2, "size": 16, "steps": 2, "seeds": "1"},
    )


def run(world, **kwargs):
    arguments = {
        "model": world.model,
        "clip": world.clip,
        "prompt_set": prompt_set(),
        **world.kwargs,
        **kwargs,
    }
    return P.KiLoRASolveFromPrompts().solve(**arguments)["result"]


@pytest.fixture
def malloc_prefetch(monkeypatch):
    prefetch = ModuleType("comfy.model_prefetch")
    state = SimpleNamespace(recording=False, begins=0, ends=0, enabled=[])

    def enabled(device):
        return True

    def begin(device):
        active = prefetch.malloc_graph_enabled(device)
        state.enabled.append(active)
        if active:
            assert not state.recording
            state.recording = True
            state.begins += 1

    def end():
        if state.recording:
            state.recording = False
            state.ends += 1

    prefetch.malloc_graph_enabled = enabled
    prefetch.malloc_graph_begin = begin
    prefetch.malloc_graph_end = end
    prefetch.malloc_graph_recording = lambda: state.recording
    monkeypatch.setitem(sys.modules, "comfy.model_prefetch", prefetch)
    return SimpleNamespace(module=prefetch, state=state, original=enabled)


def test_solve_disables_malloc_graph_for_model_capture(world, monkeypatch, malloc_prefetch):
    model = world.model.model.diffusion_model
    forward = model.forward
    prefetch = malloc_prefetch.module

    def compiled_forward(*args, **kwargs):
        prefetch.malloc_graph_begin("cpu")
        output = forward(*args, **kwargs)
        prefetch.malloc_graph_end()
        return output

    monkeypatch.setattr(model, "forward", compiled_forward)
    run(world)
    assert malloc_prefetch.state.enabled
    assert not any(malloc_prefetch.state.enabled)
    assert malloc_prefetch.state.begins == malloc_prefetch.state.ends == 0
    assert prefetch.malloc_graph_enabled is malloc_prefetch.original


@pytest.mark.parametrize("error", [RuntimeError("failed"), KeyboardInterrupt()])
@pytest.mark.parametrize("phase", ["capture", "observe"])
def test_solve_restores_malloc_graph_on_failure(world, monkeypatch, malloc_prefetch, phase, error):
    model = world.model.model.diffusion_model
    module = model if phase == "capture" else model.blocks[0]

    original = module.forward

    def fail(*args, **kwargs):
        if malloc_prefetch.module.malloc_graph_enabled("cpu"):
            return original(*args, **kwargs)
        raise error

    monkeypatch.setattr(module, "forward", fail)
    with pytest.raises(type(error) if isinstance(error, KeyboardInterrupt) else ValueError):
        run(world)
    assert malloc_prefetch.module.malloc_graph_enabled is malloc_prefetch.original
    assert all(
        not module._forward_hooks and not module._forward_pre_hooks for module in model.modules()
    )


def test_observe_disables_malloc_graph(malloc_prefetch):
    prefetch = malloc_prefetch.module

    class CompiledBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            prefetch.malloc_graph_begin(x.device)
            output = self.linear(x)
            prefetch.malloc_graph_end()
            return output

    block = CompiledBlock()
    observed = contrast._observe(
        block, ((torch.ones(1, 4),), {}), ["linear"], dict(block.named_modules())
    )
    assert set(observed) == {"linear"}
    assert malloc_prefetch.state.enabled == [False]
    assert malloc_prefetch.state.begins == malloc_prefetch.state.ends == 0
    assert prefetch.malloc_graph_enabled is malloc_prefetch.original


@pytest.mark.parametrize("recording", [False, True])
@pytest.mark.parametrize("state_api", [False, True])
def test_capture_ends_existing_malloc_graph_only_with_state_api(
    monkeypatch, malloc_prefetch, recording, state_api
):
    prefetch = malloc_prefetch.module
    malloc_prefetch.state.recording = recording
    if not state_api:
        monkeypatch.delattr(prefetch, "malloc_graph_recording")
    model = Model()
    contrast._capture(model, model.transformer_blocks[0], _samples(1)[0].target)
    assert malloc_prefetch.state.ends == int(recording and state_api)
    assert prefetch.malloc_graph_enabled is malloc_prefetch.original


def test_capture_restores_malloc_graph_when_cleanup_raises(monkeypatch, malloc_prefetch):
    prefetch = malloc_prefetch.module
    malloc_prefetch.state.recording = True
    calls = []

    def fail():
        calls.append(True)
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(prefetch, "malloc_graph_end", fail)
    model = Model()
    with pytest.raises(RuntimeError, match="cleanup failed"):
        contrast._capture(model, model.transformer_blocks[0], _samples(1)[0].target)
    assert calls == [True]
    assert prefetch.malloc_graph_enabled is malloc_prefetch.original
    assert not model.transformer_blocks[0]._forward_pre_hooks


@pytest.mark.parametrize("missing", ["module", "enabled"])
def test_solve_without_malloc_graph_api(world, monkeypatch, malloc_prefetch, missing):
    if missing == "module":
        monkeypatch.delitem(sys.modules, "comfy.model_prefetch")
    else:
        monkeypatch.delattr(malloc_prefetch.module, "malloc_graph_enabled")
    paths, _, _ = run(world)
    assert paths


@pytest.mark.parametrize("error", [None, RuntimeError("failed"), KeyboardInterrupt()])
def test_trajectory_capture_disables_and_restores_malloc_graph(
    world, monkeypatch, malloc_prefetch, error
):
    samplers = ModuleType("comfy.samplers")
    helpers = ModuleType("comfy.sampler_helpers")
    monkeypatch.setattr(sys.modules["comfy"], "samplers", samplers, raising=False)
    monkeypatch.setitem(sys.modules, "comfy.samplers", samplers)
    monkeypatch.setitem(sys.modules, "comfy.sampler_helpers", helpers)
    helpers.convert_cond = lambda conditioning: conditioning
    samplers.encode_model_conds = lambda extra, cond, *args: cond
    world.model.model.extra_conds = lambda **kwargs: {}
    prefetch = malloc_prefetch.module
    sample = _samples(1)[0].target

    def calc(model, *args):
        prefetch.malloc_graph_begin("cpu")
        if error is not None:
            raise error
        output = model.diffusion_model(**sample)
        prefetch.malloc_graph_end()
        return output

    samplers.calc_cond_batch = calc
    if error is None:
        captured = P._capture_call(world.model, [], torch.zeros(1), torch.zeros(1), {})
        assert torch.equal(captured["hidden_states"], sample["hidden_states"])
    else:
        with pytest.raises(type(error)):
            P._capture_call(world.model, [], torch.zeros(1), torch.zeros(1), {})
    assert malloc_prefetch.state.enabled == [False]
    assert malloc_prefetch.state.begins == malloc_prefetch.state.ends == 0
    assert prefetch.malloc_graph_enabled is malloc_prefetch.original
    assert not world.model.model.diffusion_model._forward_pre_hooks


@pytest.mark.parametrize("mode, count", [("single", 2), ("joint", 1)])
def test_files_plain_lora_metadata_and_chain(world, mode, count):
    paths, chain, report = run(world, mode=mode)
    assert len(paths.splitlines()) == len(chain) == count
    assert world.counts.loads == world.counts.cache == 1
    assert "Estimated GPU memory" in report
    assert "Model loaded fully for the solve" in report
    assert report.count("Times: trajectories ") == count
    assert report.count("mounts on the exact path)") == count
    assert world.counts.load_options[0][1] is True
    for path in paths.splitlines():
        assert Path(path).parent == world.root / "kilora-solved"
        assert Path(path).exists()
        assert chain[paths.splitlines().index(path)].strength == 0.7
        with safe_open(path, framework="pt") as file:
            header = file.metadata()
        metadata = json.loads(header["kilora_directions"])
        assert metadata["model_sha"] == P._model_sha(world.model)
        assert metadata["license"] == pack_license("anima")[0]
        assert metadata["concept"] in ("ink", "wash", "joint")
        assert header["kilora_cue"]
        assert set(trigger_words_of(path)) == {pair["cue"] for pair in metadata["pairs"]}
        adapter = P.SolveAdapter(world.model)
        imported = import_lora(path, patching_tree(adapter), family="anima", experts=1)
        reference = contrast.solve(
            world.model.model.diffusion_model,
            _samples(count=2),
            [contrast.PromptPair(**pair) for pair in metadata["pairs"]],
            settings=contrast.Settings(
                **{
                    **{k: v for k, v in metadata["prompt_solve"]["settings"].items()},
                    "seeds": (1,),
                    "sample_steps": (0, 1),
                }
            ),
            model_sha=metadata["model_sha"],
            license=metadata["license"],
            adapter=adapter,
        )
        weights = load_file(path)
        for name, solution in reference.mounts.items():
            stem = "lora_unet_" + anima_name(name).replace(".", "_")
            assert torch.equal(
                weights[stem + ".lora_up.weight"] @ weights[stem + ".lora_down.weight"],
                solution.delta(),
            )
        assert set(imported.adapter_set.experts) == set(reference.mounts)
    assert not list(world.temp.iterdir())


def patching_tree(adapter):
    from kilora.comfyui.patching import shape_model

    return shape_model(adapter.shapes).to_empty(device="cpu")


def test_matching_cache_skips_solve_and_encoding(world, monkeypatch):
    initial = run(world)
    counts = len(world.counts.encoded)
    monkeypatch.setattr(contrast, "solve", lambda *a, **kw: pytest.fail("cached solve ran"))
    cached = run(world)
    assert cached[0] == initial[0]
    assert cached[2].count("Reused") == 2
    assert world.counts.loads == 1
    assert len(world.counts.encoded) == counts


@pytest.mark.parametrize("mode", ["single", "joint"])
def test_cache_conflict_uses_hash_and_reuses_it(world, monkeypatch, mode):
    original, _, _ = run(world, mode=mode)
    before = {path: Path(path).read_bytes() for path in original.splitlines()}
    paths, _, report = run(world, mode=mode, rank=3)
    for old, new in zip(original.splitlines(), paths.splitlines(), strict=True):
        record = solve_record(new)
        digest = hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:8]
        assert Path(new) == Path(old).with_stem(f"{Path(old).stem}-{digest}")
        assert f"Kept {old} (other solve inputs); wrote {new}" in report
        assert Path(old).read_bytes() == before[old]
    counts = (world.counts.loads, world.counts.trajectories, len(world.counts.encoded))
    monkeypatch.setattr(contrast, "solve", lambda *a, **kw: pytest.fail("cached solve ran"))
    cached, _, report = run(world, mode=mode, rank=3)
    assert cached == paths
    assert report.count("Reused") == len(paths.splitlines())
    assert counts == (world.counts.loads, world.counts.trajectories, len(world.counts.encoded))


def test_cache_conflict_and_overwrite(world):
    original, _, _ = run(world)
    paths, _, report = run(world, rank=3, overwrite=True)
    assert paths == original
    assert "Kept" not in report
    assert all(solve_record(path)["settings"]["rank"] == 3 for path in paths.splitlines())
    assert not list(world.root.rglob("solved-ink-*.safetensors"))
    assert world.counts.loads == 2


def test_record_without_ridge_gets_a_new_filename(world):
    original, _, _ = run(world, mode="joint")
    with safe_open(original, framework="pt") as file:
        metadata = file.metadata()
    directions = json.loads(metadata["kilora_directions"])
    del directions["prompt_solve"]["settings"]["ridge"]
    metadata["kilora_directions"] = json.dumps(directions)
    save_file(load_file(original), original, metadata=metadata)
    before = Path(original).read_bytes()
    paths, _, report = run(world, mode="joint")
    assert paths != original
    assert Path(original).read_bytes() == before
    assert "ridge" in solve_record(paths)["settings"]
    assert f"Kept {original} (other solve inputs); wrote {paths}" in report


@pytest.mark.parametrize("existing", [False, True])
def test_batch_same_label_different_pairs_uses_final_paths(world, existing):
    if existing:
        run(world)
    first, second = copy.deepcopy(H3_SET), copy.deepcopy(H3_SET)
    first["subjects"], second["subjects"] = ["a bird"], ["a cat"]
    paths, _, _ = run(
        world,
        prompt_set=[prompt_set(first), prompt_set(second)],
        name=["solved", "solved"],
    )
    paths = [Path(path) for path in paths.splitlines()]
    assert len(set(paths)) == 4
    for index, path in enumerate(paths):
        label = "ink" if index % 2 == 0 else "wash"
        digest = hashlib.sha256(
            json.dumps(solve_record(path), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:8]
        assert path.stem == f"solved-{label}-{digest}"
    assert "a bird" in solve_record(paths[0])["pairs"][0]["target"]
    assert "a cat" in solve_record(paths[2])["pairs"][0]["target"]


@pytest.mark.parametrize("hashed", [False, True])
def test_batch_equal_final_paths_still_collide(world, hashed):
    if hashed:
        run(world)
    with pytest.raises(ValueError, match="Batch output names collide"):
        run(world, prompt_set=[prompt_set(), prompt_set()], name=["solved", "solved"], rank=3)


def test_hash_clash_still_raises(world, monkeypatch):
    original, _, _ = run(world)
    monkeypatch.setattr(
        P, "hashlib", SimpleNamespace(sha256=lambda *a: SimpleNamespace(hexdigest=lambda: "0" * 64))
    )
    for path in original.splitlines():
        path = Path(path)
        path.with_stem(f"{path.stem}-00000000").write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="different solve inputs.*enable overwrite"):
        run(world, rank=3)
    assert world.counts.loads == 1


def test_batch_load_once_list_inputs_and_chain(world):
    incoming = N.patching.ModChain()
    paths, chain, _ = run(
        world,
        model=[world.model],
        clip=[world.clip],
        prompt_set=[prompt_set(), prompt_set()],
        name=["first", "second"],
        mode=["single"],
        rank=[2],
        steps=[2],
        size=[16],
        cfg=[4.0],
        seeds=["1"],
        negative_prompt=[""],
        overwrite=[False],
        sets=[incoming],
    )
    assert len(paths.splitlines()) == len(chain) == 4
    assert world.counts.loads == 1
    assert {Path(path).stem for path in paths.splitlines()} == {
        "first-ink",
        "first-wash",
        "second-ink",
        "second-wash",
    }


def test_interruption_cleans_hooks_and_publishes_no_file(world):
    checks = []

    def interrupted():
        checks.append(True)
        if len(checks) == 3:
            raise InterruptedError("cancelled")

    world.mm.throw_exception_if_processing_interrupted = interrupted
    with pytest.raises(InterruptedError, match="Prompt set 1.*cancelled"):
        run(world)
    assert not list(world.root.rglob("*.safetensors"))
    assert not list(world.temp.iterdir())
    assert all(
        not module._forward_hooks and not module._forward_pre_hooks
        for module in world.model.model.diffusion_model.modules()
    )


def test_second_set_failure_keeps_finished_files(world, monkeypatch):
    original = P.trajectory_samples

    def fail(patcher, captions, pairs, settings, progress):
        if world.counts.trajectories == 2:
            raise RuntimeError("second set failed")
        return original(patcher, captions, pairs, settings, progress)

    monkeypatch.setattr(P, "trajectory_samples", fail)
    with pytest.raises(ValueError, match="Prompt set 2.*second set failed"):
        run(world, prompt_set=[prompt_set(), prompt_set()])
    assert len(list(world.root.rglob("*.safetensors"))) == 2


def test_unsupported_family_names_supported_families(world):
    del world.model.load_device
    world.model.model.diffusion_model.kilora_family = "flux"
    with pytest.raises(
        ValueError, match="single-stream family flux yet:.*Supported families: anima"
    ):
        run(world)
    assert world.counts.loads == 0


@pytest.mark.parametrize("registered", [False, True])
def test_family_without_spec_names_supported_families(world, monkeypatch, registered):
    from kilora import families

    family = "test-no-spec-prompt-solve"
    if registered:
        monkeypatch.setitem(families.FAMILIES, family, None)
    else:
        monkeypatch.delitem(families.FAMILIES, family, raising=False)
    monkeypatch.setattr(P.patching, "detect_family", lambda patcher: SimpleNamespace(family=family))
    with pytest.raises(ValueError) as error:
        P.SolveAdapter(world.model)
    assert str(error.value) == (
        f"Prompt solve does not support single-stream family {family} yet: "
        "its text and image tokens share one sequence. Supported families: anima."
    )
    assert world.counts.loads == 0


def test_packed_2d_and_dequantizing_forward_without_weight_reads():
    patcher = host_patcher()
    cast = CastLinear()
    cast.target_delta[0, 1] = 0.3
    patcher.model.diffusion_model.blocks[0].self_attn.q_proj = cast
    adapter = P.SolveAdapter(patcher)
    samples = _samples(count=2)
    for sample in samples:
        for call in (sample.target, sample.student):
            call["hidden_states"] = call["hidden_states"].squeeze(0)
            call["encoder_hidden_states"] = call["encoder_hidden_states"].squeeze(0)
    result = contrast.solve(
        patcher.model.diffusion_model,
        samples,
        [("same", "same")],
        settings=_settings(),
        model_sha="toy",
        license="test",
        adapter=adapter,
    )
    recovered = result.mounts["transformer_blocks.0.attn1.to_q"].delta()
    torch.testing.assert_close(recovered, cast.target_delta, atol=2e-4, rtol=2e-3)
    assert cast.casts > 0
    assert all(
        not module._forward_hooks and not module._forward_pre_hooks
        for module in patcher.model.diffusion_model.modules()
    )


def test_sets_output_applies_on_toy_model(world):
    _, chain, _ = run(world)
    x = _samples(count=1)[0].student["hidden_states"]
    before = world.model.run(x)[0].detach().clone()
    patched, report, _ = N.KiLoRAApply().apply(world.model, chain, 1, record_history=False)
    assert patched is not world.model
    assert report
    after = patched.run(x)[0]
    assert torch.isfinite(after).all()
    assert not torch.equal(before, after)
    assert torch.equal(before, world.model.run(x)[0])


def test_unused_nodes_preserve_existing_model_output():
    model = Model()
    sample = _samples(count=1)[0]
    before = model(**sample.student)[0].clone()
    P.KiLoRAPromptSet().build("tree", "ink", "ink", "ink")
    assert torch.equal(before, model(**sample.student)[0])


@pytest.mark.parametrize("cfg", [1.0, 4.0])
@pytest.mark.parametrize("custom_sampling", [False, True])
@pytest.mark.parametrize("chunks", [(0, 1), (1, 0), (0,)])
@pytest.mark.parametrize("packed", [False, True])
def test_comfy_sampling_bridge_records_target_cfg_inputs(
    world, monkeypatch, custom_sampling, chunks, packed, cfg
):
    samplers = ModuleType("comfy.samplers")
    helpers = ModuleType("comfy.sampler_helpers")
    sample_module = ModuleType("comfy.sample")
    comfy = sys.modules["comfy"]
    comfy.samplers, comfy.sample = samplers, sample_module
    helpers.convert_cond = lambda conditioning: conditioning
    samplers.encode_model_conds = lambda extra, cond, *args: cond
    noise_seeds = []

    def prepare_noise(latent, seed):
        noise_seeds.append(seed)
        return torch.randn(latent.shape, generator=torch.Generator().manual_seed(seed))

    sample_module.prepare_noise = prepare_noise
    sampler_name, scheduler_name = "er_sde", "simple"
    if custom_sampling:
        sampler_name, scheduler_name = "euler", "normal"
        spec = P.families.get_family("anima")
        monkeypatch.setitem(
            P.families.FAMILIES,
            "anima",
            replace(spec, solve_sampling={"sampler": sampler_name, "scheduler": scheduler_name}),
        )
    schedules = []

    def calculate_sigmas(sampling, schedule, steps):
        schedules.append(schedule)
        return torch.arange(steps, -1, -1)

    samplers.calculate_sigmas = calculate_sigmas
    sampler_names = []

    def sampler_object(name):
        sampler_names.append(name)
        return object()

    samplers.sampler_object = sampler_object
    world.model.model.extra_conds = lambda **kwargs: {}
    world.model.model.latent_format = SimpleNamespace(
        latent_channels=4,
        spacial_downscale_ratio=8,
        latent_dimensions=2,
    )
    world.model.model.model_sampling = object()
    world.model.pre_run = lambda: None
    world.model.cleanup = lambda: None

    captures = []

    def calc(model, conds, x, timestep, options):
        captures.append(x.shape[0])
        text = conds[0][0][1]["text"]
        hidden = x.reshape(x.shape[0], -1, 4)
        if packed:
            hidden = hidden.flatten(0, 1)
        return model.diffusion_model(
            hidden_states=hidden,
            encoder_hidden_states=torch.ones_like(hidden) * len(text),
            target=text == "target",
        )

    samplers.calc_cond_batch = calc

    sampler_calls = []
    truncate = False

    class Guider:
        def __init__(self, patcher):
            self.model_patcher = patcher

        def set_conds(self, positive, negative):
            self.original_conds = {"positive": positive, "negative": negative}

        def set_cfg(self, value):
            assert value == cfg

        def inner_sample(self, noise, latent, device, sampler, sigmas, *args, latent_shapes=None):
            sampler_calls.append((noise.shape[0], args[-1], latent_shapes))
            assert latent.shape == noise.shape
            wrapper = self.model_options["model_function_wrapper"]
            for sigma in sigmas[:-2] if truncate else sigmas[:-1]:
                conditional = noise + sigma
                inputs = torch.cat(
                    [conditional if chunk == 0 else conditional + 1000 for chunk in chunks]
                )
                arguments = {
                    "input": inputs,
                    "timestep": (
                        sigma.reshape(1) if len(chunks) == 1 else sigma.expand(inputs.shape[0])
                    ),
                    "c": {},
                    "cond_or_uncond": list(chunks),
                }
                wrapper(
                    lambda *a, **kw: noise,
                    {
                        **arguments,
                        "input": conditional + 1000,
                        "timestep": sigma.expand(noise.shape[0]),
                        "cond_or_uncond": [1],
                    },
                )
                wrapper(lambda *a, **kw: noise, arguments)
                wrapper(lambda *a, **kw: noise, arguments)

    samplers.CFGGuider = Guider
    for name, module in (
        ("comfy.samplers", samplers),
        ("comfy.sample", sample_module),
        ("comfy.sampler_helpers", helpers),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    settings = replace(_settings(), seeds=(1, 7), cfg=cfg)
    pairs = [
        contrast.PromptPair("target", "student", "ink", "ink"),
        contrast.PromptPair("another target", "student", "ink", "ink"),
    ]
    captions = {
        text: P._encode(world.clip, text) for text in ("target", "another target", "student", "")
    }
    progress = sys.modules["comfy.utils"].ProgressBar(4)
    samples = REAL_TRAJECTORIES(world.model, captions, pairs, settings, progress)
    assert len(samples) == 8
    assert sampler_names == [sampler_name] * 2
    assert schedules == [scheduler_name] * 2
    assert noise_seeds == [1, 7, 1, 7]
    assert sampler_calls == [(2, 1, [(2, 4, 2, 2)])] * 2
    assert captures == ([2, 2, 1, 1, 1, 1] if packed else [2, 2]) * 4
    assert samples[0].target["target"] is True
    assert samples[0].student["target"] is False
    assert progress.done == 4
    expected = []
    for pair in pairs:
        for seed in settings.seeds:
            expected.extend(
                REAL_TRAJECTORIES(
                    world.model,
                    captions,
                    [pair],
                    replace(settings, seeds=(seed,)),
                    sys.modules["comfy.utils"].ProgressBar(1),
                )
            )
    for actual, single in zip(samples, expected, strict=True):
        torch.testing.assert_close(actual.target, single.target)
        torch.testing.assert_close(actual.student, single.student)
        torch.testing.assert_close(actual.target["hidden_states"], actual.student["hidden_states"])
    truncate = True
    with pytest.raises(ValueError, match="Sampler returned fewer trajectory steps than requested"):
        REAL_TRAJECTORIES(world.model, captions, pairs, settings, progress)
    assert not world.model.model.diffusion_model._forward_pre_hooks


REAL_TRAJECTORIES = P.trajectory_samples


def test_generic_family_uses_registered_adapter_without_node_changes(world, monkeypatch):
    from kilora import families
    from kilora.bindings import BINDINGS
    from kilora.family_spec import FamilySpec, TokenLayout
    from kilora.presets import PRESETS

    family = "test-packed-prompt-solve"
    monkeypatch.setitem(families.FAMILIES, family, None)
    monkeypatch.setitem(BINDINGS, family, None)
    monkeypatch.setitem(PRESETS, family, None)
    monkeypatch.setitem(families.BLOCK_LISTS, family, "blocks")
    families.register(
        FamilySpec(
            family=family,
            include=("blocks.*.*.*",),
            layouts=(
                ("*.cross_attn.k_proj", TokenLayout("text1d")),
                ("*.cross_attn.v_proj", TokenLayout("text1d")),
                ("*", TokenLayout("image2d")),
            ),
        )
    )
    world.model.model.diffusion_model.kilora_family = family
    paths, _, _ = run(world, mode="joint")
    weights = load_file(paths)
    assert "diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight" in weights
    adapter = P.SolveAdapter(world.model)
    imported = import_lora(
        paths,
        patching_tree(adapter),
        family=family,
        experts=1,
        full_model=world.model.model.diffusion_model,
    )
    assert imported.adapter_set.experts
    assert all(
        "cross_attn.k_proj" not in name and "cross_attn.v_proj" not in name
        for name in imported.adapter_set.experts
    )


@pytest.mark.parametrize("with_hook", [False, True])
@pytest.mark.parametrize("shared_layout", [False, True])
def test_single_stream_family_with_block_list_is_refused(
    world, monkeypatch, with_hook, shared_layout
):
    from kilora import families
    from kilora.bindings import BINDINGS
    from kilora.family_spec import FamilySpec, TokenLayout
    from kilora.presets import PRESETS

    family = "test-single-stream-prompt-solve"
    monkeypatch.setitem(families.FAMILIES, family, None)
    monkeypatch.setitem(BINDINGS, family, None)
    monkeypatch.setitem(PRESETS, family, None)
    monkeypatch.setitem(families.BLOCK_LISTS, family, "blocks")
    layouts = ()
    if shared_layout:
        layouts = (
            ("*.cross_attn.k_proj", TokenLayout("text1d")),
            ("*", TokenLayout("image2d", text="before")),
        )
    families.register(FamilySpec(family=family, include=("blocks.*.*.*",), layouts=layouts))
    if with_hook:

        def image_token_rows(self, tokens, latent_hw):
            raise AssertionError("Prompt solve must not use the image-row hook yet")

        monkeypatch.setattr(FamilySpec, "image_token_rows", image_token_rows, raising=False)
    world.model.model.diffusion_model.kilora_family = family
    del world.model.load_device
    with pytest.raises(ValueError) as error:
        run(world)
    assert str(error.value) == (
        f"Prompt solve does not support single-stream family {family} yet: "
        "its text and image tokens share one sequence. Supported families: anima."
    )
    assert world.counts.loads == world.counts.trajectories == 0
    assert not world.counts.encoded
    assert not list(world.root.rglob("*.safetensors"))
    assert all(
        not module._forward_hooks and not module._forward_pre_hooks
        for module in world.model.model.diffusion_model.modules()
    )


def test_atomic_publication_does_not_replace_an_existing_file(tmp_path):
    source, destination = tmp_path / "complete", tmp_path / "existing"
    source.write_bytes(b"new")
    destination.write_bytes(b"old")
    with pytest.raises(FileExistsError):
        P._publish(source, destination)
    assert destination.read_bytes() == b"old"
    P._publish(source, destination, overwrite=True)
    assert destination.read_bytes() == b"new"


@pytest.mark.parametrize("device", [None, "missing"])
def test_adapter_falls_back_to_comfy_device(world, device):
    if device == "missing":
        del world.model.load_device
    else:
        world.model.load_device = device
    assert P.SolveAdapter(world.model).device == torch.device("cpu")


def test_model_hash_cache_tracks_model_and_patch_state(world, monkeypatch):
    model = world.model.model.diffusion_model
    original = model.state_dict
    reads = []

    def state_dict():
        reads.append(True)
        return original()

    monkeypatch.setattr(model, "state_dict", state_dict)
    initial = P._model_sha(world.model)
    assert P._model_sha(world.model) == initial
    assert len(reads) == 1
    world.model.patches_uuid = "changed"
    assert P._model_sha(world.model) == initial
    assert len(reads) == 2
    world.model.patches = {"weight": object()}
    with pytest.raises(ValueError, match="without applied"):
        P._model_sha(world.model)
    world.model.patches = {}
    monkeypatch.setattr(P.patching, "applied", lambda patcher: {"mods": True})
    with pytest.raises(ValueError, match="without applied"):
        P._model_sha(world.model)


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("exists", [False, True])
def test_publication_across_filesystems(tmp_path, monkeypatch, overwrite, exists):
    source, destination = tmp_path / "complete", tmp_path / "published"
    source.write_bytes(b"new")
    if exists:
        destination.write_bytes(b"old")
    operation = "replace" if overwrite else "link"
    original = getattr(P.os, operation)
    calls = []

    def cross_device(src, dst):
        calls.append(src)
        if len(calls) == 1:
            raise OSError(errno.EXDEV, "different filesystems")
        return original(src, dst)

    monkeypatch.setattr(P.os, operation, cross_device)
    if exists and not overwrite:
        with pytest.raises(FileExistsError):
            P._publish(source, destination, overwrite=overwrite)
        assert destination.read_bytes() == b"old"
        assert source.read_bytes() == b"new"
    else:
        P._publish(source, destination, overwrite=overwrite)
        assert destination.read_bytes() == b"new"
        assert not source.exists()
    assert len(calls) == 2
    assert not list(tmp_path.glob(".*.partial-*"))


@pytest.mark.parametrize("overwrite", [False, True])
def test_cross_filesystem_copy_error_removes_staged_file(tmp_path, monkeypatch, overwrite):
    source, destination = tmp_path / "complete", tmp_path / "published"
    source.write_bytes(b"new")

    def cross_device(*args):
        raise OSError(errno.EXDEV, "different filesystems")

    def fail_copy(input_file, output):
        output.write(b"partial")
        raise OSError("copy failed")

    monkeypatch.setattr(P.os, "replace" if overwrite else "link", cross_device)
    monkeypatch.setattr(P.shutil, "copyfileobj", fail_copy)
    with pytest.raises(OSError, match="copy failed"):
        P._publish(source, destination, overwrite=overwrite)
    assert not destination.exists()
    assert source.read_bytes() == b"new"
    assert not list(tmp_path.glob(".*.partial-*"))


def test_adapter_groups_separate_projections_with_shared_input():
    patcher = host_patcher()
    adapter = P.SolveAdapter(patcher)
    x = torch.randn(1, 3, 4)
    inputs = ((x,), {"encoder_hidden_states": x.clone(), "target": False})
    groups = adapter.groups(0, inputs)
    qkv = [f"transformer_blocks.0.attn1.to_{name}" for name in ("q", "k", "v")]
    assert groups[0] == qkv
    assert sorted(name for group in groups for name in group) == sorted(
        name for name in adapter.mounts if adapter.owner(name) == 0
    )


class PackedSolveBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn = nn.Module()
        self.cross_attn.k_proj = Linear(4)
        self.cross_attn.k_proj.target_delta[0, 1] = 0.3

    def forward(self, hidden_states, encoder_hidden_states=None, target=False):
        self.cross_attn.k_proj.target = target
        return self.cross_attn.k_proj(hidden_states)


@pytest.fixture
def packed_solve(world, monkeypatch):
    from kilora import families
    from kilora.bindings import BINDINGS
    from kilora.family_spec import FamilySpec, TokenLayout
    from kilora.presets import PRESETS

    family = "test-image-rows-solve"
    for registry in (families.FAMILIES, BINDINGS, PRESETS):
        monkeypatch.setitem(registry, family, None)
    monkeypatch.setitem(families.BLOCK_LISTS, family, "blocks")
    families.register(
        FamilySpec(
            family=family,
            include=("blocks.*.cross_attn.k_proj",),
            image_token_target="blocks.0.cross_attn.k_proj",
            layouts=(("*", TokenLayout("image2d", patch=(1, 1), text="before")),),
        )
    )
    model = world.model.model.diffusion_model
    model.kilora_family = family
    model.blocks = nn.ModuleList([PackedSolveBlock()])
    return model, P.SolveAdapter(world.model, latent_hw=(2, 2))


def packed_sample(batched, target_count=4, student_count=4):
    image = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10

    def inputs(text_count, image_count, target):
        x = torch.cat((torch.full((text_count, 4), -3.0), image[:image_count]))
        return {"hidden_states": x.unsqueeze(0) if batched else x, "target": target}

    return image, contrast.Sample(inputs(3, target_count, True), inputs(1, student_count, False))


@pytest.mark.parametrize("batched", [False, True])
def test_packed_solve_matches_image_only(packed_solve, batched):
    model, adapter = packed_solve
    image, sample = packed_sample(batched)
    for values in (sample.target, sample.student):
        x = values["hidden_states"]
        selected = x[..., adapter.rows(x.shape[-2]), :]
        torch.testing.assert_close(selected, image.unsqueeze(0) if batched else image)
    packed = contrast.solve(
        model,
        [sample],
        [("long caption", "short")],
        settings=_settings(),
        adapter=adapter,
        model_sha="toy",
        license="Apache-2.0",
    )
    assert not packed.metadata["excluded_mounts"]
    image = image.unsqueeze(0) if batched else image
    plain = contrast.Sample(
        {"hidden_states": image, "target": True},
        {"hidden_states": image, "target": False},
    )
    reference = contrast.solve(
        model,
        [plain],
        [("long caption", "short")],
        settings=_settings(),
        adapter=adapter,
        model_sha="toy",
        license="Apache-2.0",
    )
    for name, solution in packed.mounts.items():
        expected = reference.mounts[name]
        torch.testing.assert_close(solution.up.T @ solution.down, expected.up.T @ expected.down)


def test_packed_solve_invalid_layout(packed_solve, monkeypatch):
    from kilora.family_spec import FamilySpec

    _, adapter = packed_solve
    monkeypatch.setattr(FamilySpec, "image_token_rows", lambda self, tokens, latent_hw: None)
    with pytest.raises(ValueError, match="test-image-rows-solve.*2 tokens"):
        adapter.rows(2)


def test_packed_solve_row_count_mismatch(packed_solve, monkeypatch):
    model, adapter = packed_solve
    _, sample = packed_sample(False)
    monkeypatch.setattr(adapter, "rows", lambda tokens: slice(3 if tokens == 7 else 0, None))
    with pytest.raises(ValueError, match="image row counts differ"):
        contrast.solve(
            model,
            [sample],
            [("long caption", "short")],
            settings=_settings(),
            adapter=adapter,
            model_sha="toy",
            license="Apache-2.0",
        )


@pytest.mark.parametrize("with_image_target", [False, True])
def test_cross_attention_rows_preserve_results(world, monkeypatch, with_image_target):
    from kilora import families
    from kilora.family_spec import FamilySpec

    if with_image_target:
        monkeypatch.setitem(
            families.FAMILIES,
            "anima",
            replace(families.FAMILIES["anima"], image_token_target="blocks.0.self_attn.q_proj"),
        )

    def unexpected_rows(self, tokens, latent_hw):
        raise AssertionError("Cross-attention solving must not select image rows")

    monkeypatch.setattr(FamilySpec, "image_token_rows", unexpected_rows)
    model = world.model.model.diffusion_model
    adapter = P.SolveAdapter(world.model)
    assert not adapter.select_image_rows
    assert adapter.rows(12) is None
    samples = _samples()
    baseline = contrast.solve(
        model,
        samples,
        [("target", "student")],
        settings=_settings(),
        adapter=adapter,
        model_sha="toy",
        license="Apache-2.0",
    )
    monkeypatch.setattr(adapter, "rows", None)
    reference = contrast.solve(
        model,
        samples,
        [("target", "student")],
        settings=_settings(),
        adapter=adapter,
        model_sha="toy",
        license="Apache-2.0",
    )
    assert baseline.metadata == reference.metadata
    assert baseline.mounts.keys() == reference.mounts.keys()
    for name, solution in baseline.mounts.items():
        assert torch.equal(solution.down, reference.mounts[name].down)
        assert torch.equal(solution.up, reference.mounts[name].up)


@pytest.fixture
def sampling_family(world, monkeypatch):
    from kilora import families
    from kilora.bindings import BINDINGS
    from kilora.family_spec import FamilySpec, TokenLayout
    from kilora.presets import PRESETS

    family = "test-sampling-prompt-solve"
    for registry in (families.FAMILIES, BINDINGS, PRESETS):
        monkeypatch.setitem(registry, family, None)
    monkeypatch.setitem(families.BLOCK_LISTS, family, "blocks")
    spec = FamilySpec(
        family=family,
        include=("blocks.*.*.*",),
        layouts=(
            ("*.cross_attn.k_proj", TokenLayout("text1d")),
            ("*.cross_attn.v_proj", TokenLayout("text1d")),
            ("*", TokenLayout("image2d")),
        ),
        solve_sampling={"steps": 6, "cfg": 2.0, "size": 1024},
    )
    families.register(spec)
    world.model.model.diffusion_model.kilora_family = family
    return spec


def solve_record(path):
    with safe_open(path, framework="pt") as file:
        return json.loads(file.metadata()["kilora_directions"])["prompt_solve"]


def test_auto_family_sampling_report_and_cache(world, sampling_family):
    paths, _, report = run(world, mode="joint", steps=0, cfg=-1, size=0)
    record = solve_record(paths)
    assert record["settings"]["steps"] == 6
    assert record["settings"]["cfg"] == 2.0
    assert record["settings"]["size"] == 1024
    assert record["settings"]["sample_steps"] == [0, 2, 5]
    for name, value in sampling_family.solve_sampling.items():
        assert f"{name} {value} (family {sampling_family.family})" in report
    assert "sampler er_sde (default)" in report
    assert "scheduler simple (default)" in report
    before = Path(paths).read_bytes()
    _, _, report = run(world, mode="joint", steps=6, cfg=2.0, size=1024)
    assert "Reused" in report
    assert world.counts.loads == 1
    assert Path(paths).read_bytes() == before


def test_anima_auto_reuses_explicit_default_cache(world):
    paths, _, _ = run(world, mode="joint", steps=15, cfg=1.0, size=768)
    assert solve_record(paths)["settings"]["sample_steps"] == [0, 7, 14]
    before = Path(paths).read_bytes()
    _, _, report = run(world, mode="joint", steps=0, cfg=-1, size=0)
    assert "Reused" in report
    assert world.counts.loads == 1
    assert Path(paths).read_bytes() == before
    for text in ("steps 15 (default)", "cfg 1.0 (default)", "size 768 (default)"):
        assert text in report


@pytest.mark.parametrize("steps, sampled", [(1, [0]), (2, [0, 1]), (3, [0, 1, 2]), (6, [0, 2, 5])])
def test_explicit_sampling_overrides_family_and_deduplicates(
    world, sampling_family, steps, sampled
):
    paths, _, report = run(world, mode="joint", steps=steps, cfg=0.0, size=16)
    settings = solve_record(paths)["settings"]
    assert settings["steps"] == steps
    assert settings["cfg"] == 0.0
    assert settings["size"] == 16
    assert settings["sample_steps"] == sampled
    assert f"steps {steps} (explicit)" in report
    assert "cfg 0.0 (explicit)" in report
    assert "size 16 (explicit)" in report


def test_unset_family_sampling_uses_solver_defaults():
    from kilora.family_spec import FamilySpec

    values, report = P._solve_sampling(FamilySpec(family="test-unset-sampling"))
    assert values == {
        "steps": 15,
        "cfg": 1.0,
        "size": 768,
        "sampler": "er_sde",
        "scheduler": "simple",
    }
    assert report.count("(default)") == 5


def test_mixed_auto_and_explicit_sampling(sampling_family):
    values, report = P._solve_sampling(sampling_family, cfg=0.0)
    assert (values["steps"], values["cfg"], values["size"]) == (6, 0.0, 1024)
    assert "cfg 0.0 (explicit)" in report


@pytest.mark.parametrize("cfg", [1.0, 4.0])
def test_anima_auto_and_explicit_new_solves_are_identical(world, monkeypatch, cfg):
    explicit, _, _ = run(world, name="explicit", mode="joint", steps=15, cfg=cfg, size=768)
    if cfg == 4.0:
        constructor = contrast.Settings

        def previous_settings(**kwargs):
            return constructor(**{"cfg": 4.0, **kwargs})

        monkeypatch.setattr(contrast, "Settings", previous_settings)
    auto, _, _ = run(world, name="auto", mode="joint", steps=0, cfg=-1, size=0)
    expected, actual = load_file(explicit), load_file(auto)
    assert expected.keys() == actual.keys()
    for key in expected:
        assert torch.equal(expected[key], actual[key])
    with safe_open(explicit, framework="pt") as file:
        metadata = file.metadata()
    with safe_open(auto, framework="pt") as file:
        assert file.metadata() == metadata
    assert world.counts.loads == 2


def test_node_replay_batches_match_single_sample_file(world, monkeypatch, malloc_prefetch):
    paths, _, _ = run(world, name="batched", mode="joint")
    constructor = contrast.Settings

    def single_settings(**kwargs):
        return constructor(**{**kwargs, "replay_batch": 1})

    monkeypatch.setattr(contrast, "Settings", single_settings)
    single_paths, _, _ = run(world, name="single", mode="joint")
    batched, single = load_file(paths), load_file(single_paths)
    assert batched.keys() == single.keys()
    for name in batched:
        if name.endswith(".lora_down.weight"):
            up = name.replace(".lora_down.weight", ".lora_up.weight")
            torch.testing.assert_close(
                batched[up] @ batched[name], single[up] @ single[name], atol=2e-6, rtol=2e-5
            )
        elif name.endswith(".alpha"):
            assert torch.equal(batched[name], single[name])
    records = []
    for path in (paths, single_paths):
        with safe_open(path, framework="pt") as file:
            records.append(json.loads(file.metadata()["kilora_directions"]))
    assert records[0]["replay_batch"] == 8
    assert records[1]["replay_batch"] == 1
    assert records[0]["block_gaps"] == pytest.approx(records[1]["block_gaps"], abs=2e-6)
    assert malloc_prefetch.module.malloc_graph_enabled is malloc_prefetch.original
    assert all(
        not module._forward_hooks and not module._forward_pre_hooks
        for module in world.model.model.diffusion_model.modules()
    )


def test_adapter_groups_accept_batched_calls(world):
    adapter = P.SolveAdapter(world.model)
    x = torch.randn(8, 3, 4)
    groups = adapter.groups(0, ((x,), {"encoder_hidden_states": x.clone(), "target": False}))
    assert groups[0] == [f"transformer_blocks.0.attn1.to_{name}" for name in ("q", "k", "v")]
    assert adapter.rows(3) is None


@pytest.mark.parametrize("fits", [False, True])
def test_solve_load_mode_depends_on_free_memory(world, monkeypatch, fits):
    monkeypatch.setattr(world.mm, "get_free_memory", lambda device: 1000 if fits else 999)
    monkeypatch.setattr(contrast, "memory_estimate", lambda *args, **kwargs: 1000)
    _, _, report = run(world, mode="joint")
    assert world.counts.load_options == [(1000, fits)]
    expected = (
        "Model loaded fully for the solve"
        if fits
        else "Model loaded in parts (not enough free GPU memory)"
    )
    assert expected in report


def test_solve_preserves_load_call_without_full_load_option(world, monkeypatch):
    options = []

    def load(models, memory_required=0):
        options.append(memory_required)

    monkeypatch.setattr(world.mm, "load_models_gpu", load)
    _, _, report = run(world, mode="joint")
    assert len(options) == 1
    assert "Model load mode unchanged (full-load option unavailable)" in report


def test_solve_timings_do_not_change_cache_record(world):
    paths, _, report = run(world, mode="joint")
    path = Path(paths)
    before = path.read_bytes()
    record = solve_record(path)
    assert "fit_seconds" not in record
    assert "exact_fits" not in record
    with safe_open(path, framework="pt") as file:
        metadata = json.loads(file.metadata()["kilora_directions"])
    assert "fit_seconds" not in metadata
    assert 0 <= metadata["exact_fits"] <= len(metadata["fit"])
    assert "Times: trajectories " in report
    loads, trajectories = world.counts.loads, world.counts.trajectories
    reused, _, report = run(world, mode="joint")
    assert reused == paths
    assert path.read_bytes() == before
    assert world.counts.loads == loads
    assert world.counts.trajectories == trajectories
    assert f"Reused {path}" in report
    assert "Times:" not in report


@pytest.mark.parametrize("mode", ["joint", "single"])
def test_repeat_node_solves_write_equal_bytes_and_report_times(world, monkeypatch, mode):
    clock = [0.0]
    step = [1.0]

    def tick():
        clock[0] += step[0]
        return clock[0]

    monkeypatch.setattr(contrast.time, "perf_counter", tick)
    paths, _, first_report = run(world, mode=mode)
    before = {path: Path(path).read_bytes() for path in paths.splitlines()}
    trajectories = world.counts.trajectories
    step[0] = 2.0
    repeated, _, second_report = run(world, mode=mode, overwrite=True)
    assert repeated == paths
    assert world.counts.trajectories == 2 * trajectories
    assert before == {path: Path(path).read_bytes() for path in repeated.splitlines()}
    for report in (first_report, second_report):
        assert report.count("Times: trajectories ") == trajectories
        assert "block replays " in report
        assert "fits " in report
        assert "mounts on the exact path)" in report


def test_coupling_options_off_keep_the_main_record(world):
    paths, _, _ = run(world)
    for path in paths.splitlines():
        record = solve_record(path)
        assert not {"overlap_penalty", "deflate", "concept_order", "set_pairs"} & set(record)
        with safe_open(path, framework="pt") as file:
            metadata = json.loads(file.metadata()["kilora_directions"])
        assert "overlap_penalty" not in metadata and "deflate" not in metadata
    before = {path: Path(path).read_bytes() for path in paths.splitlines()}
    again, _, report = run(world, overlap_penalty=0.0, deflate=False)
    assert again == paths and report.count("Reused") == 2
    assert all(Path(path).read_bytes() == data for path, data in before.items())


def test_coupling_options_reach_the_solve_and_the_files(world, monkeypatch):
    seen = []
    original = contrast.solve

    def recording(*args, **kwargs):
        seen.append(kwargs.get("coupling"))
        return original(*args, **kwargs)

    monkeypatch.setattr(contrast, "solve", recording)
    paths, _, report = run(world, overlap_penalty=0.5, deflate=True)
    assert len(seen) == 4 and world.counts.trajectories == 4
    assert [c.overlap_round for c in seen] == [0, 0, 1, 1]
    assert [len(c.earlier) for c in seen] == [0, 1, 0, 1]
    assert [c.overlap_penalty for c in seen] == [0.0, 0.0, 0.5, 0.5]
    assert report.count("Times: trajectories ") == 4
    for path, label in zip(paths.splitlines(), ("ink", "wash"), strict=True):
        record = solve_record(path)
        assert record["mode"] == "single"
        assert record["overlap_penalty"] == 0.5 and record["deflate"] is True
        assert record["concept_order"] == ["ink", "wash"]
        assert len(record["set_pairs"]) == 4
        with safe_open(path, framework="pt") as file:
            metadata = json.loads(file.metadata()["kilora_directions"])
        assert metadata["concept"] == label
        assert metadata["overlap_penalty"] == 0.5 and metadata["overlap_round"] == 1
        assert metadata["deflate"] is True
    seen.clear()
    cached, _, report = run(world, overlap_penalty=0.5, deflate=True)
    assert cached == paths and report.count("Reused") == 2 and not seen


def test_coupled_set_solves_again_when_one_file_is_missing(world):
    paths, _, _ = run(world, deflate=True)
    first, second = paths.splitlines()
    Path(second).unlink()
    again, _, report = run(world, deflate=True)
    assert again == paths and "Reused" not in report
    assert Path(first).exists() and Path(second).exists()


def test_joint_mode_ignores_the_coupling_options(world):
    plain, _, _ = run(world, mode="joint")
    coupled, _, report = run(world, mode="joint", overlap_penalty=0.5, deflate=True)
    assert coupled == plain and report.count("Reused") == 1


@pytest.mark.parametrize("value", [-1.0, float("nan")])
def test_node_rejects_invalid_overlap_penalty(world, value):
    with pytest.raises(ValueError, match="overlap_penalty"):
        run(world, overlap_penalty=value)


def test_solved_concept_resolves_as_a_mention(world, monkeypatch, tmp_path):
    from test_mention_prompt import FakeClip

    monkeypatch.setenv("KILORA_LIBRARY_DIR", str(tmp_path / "library"))
    paths, _, _ = run(world)
    listed = sorted(p.relative_to(world.root).as_posix() for p in world.root.rglob("*"))
    monkeypatch.setattr(N.FOLDERS, "list", lambda folder, extension=None: listed)
    clip = FakeClip()
    _, chain, report = N.KiLoRAMentionPrompt().encode(clip, "@wash:0.5 a cat")
    entries = list(chain)
    assert [(e.path, e.strength) for e in entries] == [(paths.splitlines()[1], 0.5)]
    assert clip.texts == ["wash, a cat"]
    assert "solved-wash" in report


def test_default_adapter_scale_keeps_the_unscaled_change():
    layer = nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(3, 4))
    solution = contrast.MountSolution(
        down=torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        up=torch.tensor([[0.2, 0.0, -0.1], [0.0, 0.3, 0.0]]),
        ridge=0.1,
        fit_share=1.0,
    )
    adapter = SimpleNamespace(device=torch.device("cpu"), locations={"layer": (layer, None)})
    changes = P._AdapterChanges(adapter)
    assert changes.scale == 1.0
    changes.add("layer", solution)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    with changes.active(True):
        hooked = layer(x)
    changes.close()
    plain = layer(x)
    change = F.linear(F.linear(x.float(), solution.down), solution.up.T)
    torch.testing.assert_close(hooked, plain + change.to(plain.dtype))
