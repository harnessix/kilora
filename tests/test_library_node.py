# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from safetensors.torch import save_file
from test_bypass import Patcher
from test_comfyui import FakeBaseModel, folders
from test_library_probe import Flux

from kilora.comfyui import patching
from kilora.comfyui.library_node import KiLoRALoRALibrary
from kilora.comfyui.nodes import COMFY_DEPTH_STEPS


class SamplingPatcher(Patcher):
    def __init__(self, model, object_patches=None, model_options=None, injections=None):
        super().__init__(model, object_patches, model_options, injections)
        self.wrappers = {}

    def clone(self):
        return SamplingPatcher(self.model, self.object_patches, self.model_options, self.injections)

    def add_wrapper_with_key(self, kind, key, wrapper):
        self.wrappers[kind, key] = wrapper


@pytest.fixture(params=[False, True])
def routing_loop(request):
    from kilora import compute

    compute.use_depth_loop(request.param)
    yield
    compute.use_depth_loop(False)


def test_library_selection_matches_joint_and_cleans_up(
    tmp_path,
    monkeypatch,
    folders,
    routing_loop,
):
    monkeypatch.setenv("KILORA_LIBRARY_DIR", str(tmp_path / "index"))
    monkeypatch.setattr("kilora.library.index._family", lambda keys: "flux")
    folder = tmp_path / "external-library"
    folder.mkdir()
    torch.manual_seed(42)
    for i in range(2):
        save_file(
            {
                "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.randn(2, 4),
                "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.randn(4, 2),
            },
            str(folder / f"style{i}.safetensors"),
        )

    class SpatialFlux(Flux):
        def forward(self, x, timestep, context):
            batch, channels, height, width = x.shape
            values = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
            values = super().forward(values, timestep, context)
            return values.reshape(batch, height, width, channels).permute(0, 3, 1, 2)

    model = SpatialFlux()
    for block in model.double_blocks:
        block.img_attn.qkv = torch.nn.Linear(4, 12)
        block.img_attn.proj = torch.nn.Linear(4, 4)
    model.kilora_family = "flux"
    for block in model.double_blocks:

        def forward(x, context, block=block):
            x = x + block.img_attn.qkv(x).reshape(*x.shape[:-1], 3, 4).sum(dim=-2) + context
            return x + block.img_mlp(x)

        block.forward = forward
    patcher = SamplingPatcher(FakeBaseModel(model))
    patcher.load_device = torch.device("cpu")
    args = {
        "input": torch.randn(1, 3, 4, 4),
        "timestep": torch.ones(1),
        "c": {"context": torch.randn(1, 16, 4)},
    }
    from kilora import library

    calls = []
    original_select = library.select

    def counted(*args, **kwargs):
        calls.append(1)
        return original_select(*args, **kwargs)

    monkeypatch.setattr(library, "select", counted)
    samples = []
    original_apply = patching.apply

    def capture_apply(*apply_args, **apply_kwargs):
        observer = apply_kwargs.get("share_observer")
        if observer is not None:

            def capture(shares, args):
                samples.append((float(args["timestep"][0]), shares.detach().clone().mean(0)))
                observer(shares, args)

            apply_kwargs["share_observer"] = capture
        return original_apply(*apply_args, **apply_kwargs)

    monkeypatch.setattr(patching, "apply", capture_apply)
    node = KiLoRALoRALibrary()
    installed, report, recipe = node.apply(patcher, str(folder), "style", count=5, min_relevance=0)
    assert "selection runs at the first step" in report
    from pathlib import Path

    assert not Path(recipe).exists()
    wrapper = installed.model_options["model_function_wrapper"]
    outer = installed.wrappers["outer_sample", "kilora_library_probe"]
    original = model.double_blocks[0].img_attn.qkv.forward
    outputs = []

    def run():
        outputs.append(wrapper(model, args))
        outputs.append(wrapper(model, args))
        wrapper(model, {**args, "timestep": torch.full((1,), 0.5)})
        assert model.double_blocks[0].img_attn.qkv.forward != original

    outer(run)
    import hashlib

    sorted_paths = sorted(
        folder.glob("*.safetensors"), key=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    )
    assert len(samples) == 3
    means = torch.stack(
        [
            torch.stack([v for t, v in samples if t == step]).mean(0)
            for step in {t for t, _ in samples}
        ]
    ).mean(0)
    for path, mean in zip(sorted_paths, means, strict=True):
        line = next(line for line in node.last_report.splitlines() if line.startswith(path.name))
        assert float(line.split("mean routing weight ")[1]) == pytest.approx(float(mean), abs=5e-5)
    assert model.double_blocks[0].img_attn.qkv.forward == original
    assert torch.equal(outputs[0], outputs[1])
    assert len(calls) == 1
    from safetensors import safe_open

    from kilora.comfyui.nodes import KiLoRALoadMod

    assert Path(recipe).exists()
    with safe_open(recipe, framework="pt") as contents:
        assert list(contents.keys()) == []
    loaded, _ = KiLoRALoadMod().load(patcher, Path(recipe).name, 1.0)
    loaded.inject()
    try:
        replay = loaded.model_options["model_function_wrapper"](model, args)
    finally:
        loaded.eject()
    assert torch.equal(outputs[0], replay)
    outer(lambda: wrapper(model, args))
    assert len(calls) == 2
    assert model.double_blocks[0].img_attn.qkv.forward == original
    assert "style0.safetensors" in node.last_report and "style1.safetensors" in node.last_report
    lines = [line for line in node.last_report.splitlines() if line.startswith("style")]
    assert len(lines) == 2
    weights = [float(line.split("mean routing weight ")[1]) for line in lines]
    assert all(0 <= weight <= 1 for weight in weights)
    assert sum(weights) == pytest.approx(1, abs=1e-4)
    paths = sorted(
        folder.glob("*.safetensors"), key=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    )
    chain = [patching.ModEntry("lora", str(p), strength=0.8, axis="creative") for p in paths]
    plain, _ = patching.apply(
        patcher,
        chain,
        None,
        joint=True,
        bypass="on",
        freeze=0.4,
        axis_alpha="auto",
        depth_steps=COMFY_DEPTH_STEPS,
    )
    plain.inject()
    try:
        expected = plain.model_options["model_function_wrapper"](model, args)
    finally:
        plain.eject()
    assert torch.equal(outputs[0], expected)


def test_loader_tracks_model_device(tmp_path, monkeypatch, folders):
    from kilora import library

    monkeypatch.setenv("KILORA_LIBRARY_DIR", str(tmp_path / "index"))
    devices = []

    def loader(paths, *, device):
        devices.append((paths, device))
        return object()

    monkeypatch.setattr(library, "LibraryLoader", loader)
    model = Flux()
    model.kilora_family = "flux"
    patcher = SamplingPatcher(FakeBaseModel(model))
    first, second = object(), object()
    patcher.load_device = first
    node = KiLoRALoRALibrary()
    node.apply(patcher, str(tmp_path), "")
    initial = node.loader
    node.apply(patcher, str(tmp_path), "")
    assert node.loader is initial
    patcher.load_device = second
    node.apply(patcher, str(tmp_path), "")
    assert node.loader is not initial
    assert [device for _, device in devices] == [first, second]


def test_loader_uses_comfy_device_when_patcher_has_none(tmp_path, monkeypatch, folders):
    import sys
    from types import ModuleType

    from kilora import library

    monkeypatch.setenv("KILORA_LIBRARY_DIR", str(tmp_path / "index"))
    comfy = ModuleType("comfy")
    management = ModuleType("comfy.model_management")
    device = object()
    management.get_torch_device = lambda: device
    comfy.model_management = management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    seen = []
    monkeypatch.setattr(library, "LibraryLoader", lambda paths, *, device: seen.append(device))
    model = Flux()
    model.kilora_family = "flux"
    KiLoRALoRALibrary().apply(SamplingPatcher(FakeBaseModel(model)), str(tmp_path), "")
    assert seen == [device]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_loader_keeps_set_on_device_and_matches_cpu(tmp_path, monkeypatch, folders):
    from kilora import adaptive_gate

    monkeypatch.setattr(adaptive_gate, "triton", None)
    monkeypatch.setenv("KILORA_LIBRARY_DIR", str(tmp_path / "index"))
    monkeypatch.setattr("kilora.library.index._family", lambda keys: "flux")
    folder = tmp_path / "styles"
    folder.mkdir()
    torch.manual_seed(42)
    save_file(
        {
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.randn(2, 4),
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.randn(4, 2),
        },
        str(folder / "style.safetensors"),
    )
    model = Flux()
    model.kilora_family = "flux"
    for block in model.double_blocks:
        block.img_attn.qkv = torch.nn.Linear(4, 12)
        block.img_attn.proj = torch.nn.Linear(4, 4)

        def forward(x, context, block=block):
            x = x + block.img_attn.qkv(x).reshape(*x.shape[:-1], 3, 4).sum(dim=-2) + context
            return x + block.img_mlp(x)

        block.forward = forward
    model.cuda()
    patcher = SamplingPatcher(FakeBaseModel(model))
    args = {
        "input": torch.randn(1, 5, 3, device="cuda"),
        "timestep": torch.ones(1, device="cuda"),
        "c": {"context": torch.randn(1, 5, 4, device="cuda")},
    }
    node = KiLoRALoRALibrary()
    outputs = []
    for device in (torch.device("cpu"), torch.device("cuda")):
        patcher.load_device = device
        installed, _, _ = node.apply(patcher, str(folder), "style", min_relevance=0)
        wrapper = installed.model_options["model_function_wrapper"]
        outer = installed.wrappers["outer_sample", "kilora_library_probe"]
        outputs.append(outer(lambda wrapper=wrapper: wrapper(model, args)))
        assert node.loader.routed.bytes > 0
        (_, choice), _ = next(iter(node.loader.routed._entries.values()))
        expert = next(iter(choice.adapter_sets[0].experts.values()))
        assert expert.down.device.type == device.type
        assert ("_library_transfer" in choice.adapter_sets[0].extra) == (device.type == "cpu")
    assert torch.equal(*outputs)


def test_empty_library_runs_base(tmp_path, monkeypatch, folders):
    monkeypatch.setenv("KILORA_LIBRARY_DIR", str(tmp_path / "index"))
    model = Flux()
    model.kilora_family = "flux"
    patcher = SamplingPatcher(FakeBaseModel(model))
    node = KiLoRALoRALibrary()
    installed, _, _ = node.apply(patcher, str(tmp_path / "empty"), "nothing")
    args = {
        "input": torch.randn(1, 5, 3),
        "timestep": torch.ones(1),
        "c": {"context": torch.randn(1, 5, 4)},
    }
    wrapper = installed.model_options["model_function_wrapper"]
    outer = installed.wrappers["outer_sample", "kilora_library_probe"]
    expected = model(args["input"], args["timestep"], **args["c"])
    for _ in range(2):
        assert torch.equal(outer(lambda: wrapper(model, args)), expected)
    assert "No files" in node.last_report
    assert "mean routing weight" not in node.last_report


def test_unrelated_library_and_sampling_failure_leave_model_unchanged(
    tmp_path,
    monkeypatch,
    folders,
):
    monkeypatch.setenv("KILORA_LIBRARY_DIR", str(tmp_path / "index"))
    folder = folders["loras"]
    save_file({"unrelated.lora_down.weight": torch.ones(1, 4)}, str(folder / "other.safetensors"))
    model = Flux()
    model.kilora_family = "flux"
    patcher = SamplingPatcher(FakeBaseModel(model))
    node = KiLoRALoRALibrary()
    installed, _, _ = node.apply(patcher, str(folder), "anything", min_relevance=0)
    args = {
        "input": torch.randn(1, 5, 3),
        "timestep": torch.ones(1),
        "c": {"context": torch.randn(1, 5, 4)},
    }
    wrapper = installed.model_options["model_function_wrapper"]
    outer = installed.wrappers["outer_sample", "kilora_library_probe"]
    expected = model(args["input"], args["timestep"], **args["c"])
    assert torch.equal(outer(lambda: wrapper(model, args)), expected)
    assert "No files" in node.last_report


def test_choice_ejects_hooks_after_failure():
    import pytest

    from kilora.comfyui.bypass import ForwardHook, injection
    from kilora.comfyui.library_node import _Choice
    from kilora.library.probe import StepZeroProbe

    model = Flux()
    patcher = SamplingPatcher(FakeBaseModel(model))
    hook = ForwardHook("diffusion_model.double_blocks.0.img_attn.qkv", lambda x, y: y + 1)
    patcher.set_injections("kilora", [injection([hook])])
    choice = _Choice(patcher)
    probe = StepZeroProbe(model, model, model.double_blocks, {}, lambda _: choice)
    args = {
        "input": torch.randn(1, 5, 3),
        "timestep": torch.ones(1),
        "c": {"context": torch.randn(1, 5, 4)},
    }
    original = model.double_blocks[0].img_attn.qkv.forward

    def fail():
        probe.wrap()(model, args)
        assert hook.injected
        raise RuntimeError("sampling failed")

    with pytest.raises(RuntimeError, match="sampling failed"):
        probe.outer_sample(fail)
    assert not hook.injected
    assert model.double_blocks[0].img_attn.qkv.forward == original


def test_library_masks_match_region_chain(tmp_path, monkeypatch, folders, routing_loop):
    import hashlib

    from kilora import library
    from kilora.comfyui.nodes import KiLoRARegion

    monkeypatch.setenv("KILORA_LIBRARY_DIR", str(tmp_path / "index"))
    monkeypatch.setattr("kilora.library.index._family", lambda keys: "flux")
    folder = tmp_path / "styles"
    folder.mkdir()
    torch.manual_seed(88)
    for i in range(2):
        save_file(
            {
                "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.eye(4)[i : i + 1],
                "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.ones(4, 1),
            },
            str(folder / f"style{i}.safetensors"),
        )
    paths = sorted(
        folder.glob("*.safetensors"), key=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    )
    masks = [torch.zeros(1, 8, 8), torch.zeros(1, 8, 8)]
    masks[0][:, :, :4] = 1
    masks[1][:, :, 4:] = 1
    seen = []
    original_select = library.select

    def select(index, prompt, tokens, family, count, **kwargs):
        regions = kwargs["regions"]
        assert set(regions) == {"a", "b"}
        assert regions["a"].shape == (4, 4)
        assert torch.equal(regions["a"], masks[0][0, ::2, ::2])
        assert torch.equal(regions["b"], masks[1][0, ::2, ::2])
        retrieval_inputs = index.retrieval_inputs

        def planted_inputs(family, clip=None):
            inputs = dict(retrieval_inputs(family, clip))
            group = "0-attention-in-4"
            inputs["groups"] = {group: 4}
            inputs["summaries"] = {}
            inputs["directions"] = {
                hashlib.sha256(path.read_bytes()).hexdigest(): {group: torch.eye(4)[i : i + 1]}
                for i, path in enumerate(
                    [folder / "style0.safetensors", folder / "style1.safetensors"]
                )
            }
            return inputs

        monkeypatch.setattr(index, "retrieval_inputs", planted_inputs)
        result = original_select(index, prompt, tokens, family, count, **kwargs)
        seen.append({item.path.rsplit("/", 1)[-1]: item.regions for item in result})
        return result

    monkeypatch.setattr(library, "select", select)

    class SpatialFlux(Flux):
        def forward(self, x, timestep, context):
            batch, channels, height, width = x.shape
            values = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
            values = super().forward(values, timestep, context)
            return values.reshape(batch, height, width, channels).permute(0, 3, 1, 2)

    model = SpatialFlux()
    model.kilora_family = "flux"
    with torch.no_grad():
        model.img_in.weight.zero_()
        model.img_in.weight[:3, :3].copy_(torch.eye(3))
        model.img_in.bias.zero_()
    for block in model.double_blocks:
        block.img_attn.qkv = torch.nn.Linear(4, 12)
        block.img_attn.proj = torch.nn.Linear(4, 4)

        def forward(x, context, block=block):
            x = x + block.img_attn.qkv(x).reshape(*x.shape[:-1], 3, 4).sum(dim=-2) + context
            return x + block.img_mlp(x)

        block.forward = forward
    patcher = SamplingPatcher(FakeBaseModel(model))
    args = {
        "input": torch.randn(1, 3, 4, 4),
        "timestep": torch.ones(1),
        "c": {"context": torch.randn(1, 16, 4)},
    }
    args["input"].zero_()
    args["input"][:, 0, :, :2] = 1
    args["input"][:, 1, :, 2:] = 1
    args["timestep"].zero_()
    node = KiLoRALoRALibrary()
    installed, _, _ = node.apply(
        patcher, str(folder), "", count=1, min_relevance=0, mask_a=masks[0], mask_b=masks[1]
    )
    wrapper = installed.model_options["model_function_wrapper"]
    outer = installed.wrappers["outer_sample", "kilora_library_probe"]
    actual = outer(lambda: wrapper(model, args))
    assert seen == [{"style0.safetensors": ("a",), "style1.safetensors": ("b",)}]
    assert node.last_report.count("region was not saved") == 2

    chain = patching.ModChain(())
    for path in paths:
        mask = masks[int(path.stem[-1])]
        entry = patching.ModEntry("lora", str(path), strength=0.8, axis="creative")
        region_chain = KiLoRARegion().region(patching.ModChain((entry,)), mask)[0]
        chain = chain.add(*region_chain)
    plain, _ = patching.apply(
        patcher,
        chain,
        None,
        joint=True,
        bypass="on",
        freeze=0.4,
        axis_alpha="auto",
        depth_steps=COMFY_DEPTH_STEPS,
    )
    plain.inject()
    try:
        expected = plain.model_options["model_function_wrapper"](model, args)
    finally:
        plain.eject()
    assert torch.equal(actual, expected)


@pytest.fixture
def library_sampling(tmp_path, monkeypatch, folders):
    monkeypatch.setenv("KILORA_LIBRARY_DIR", str(tmp_path / "index"))
    monkeypatch.setattr("kilora.library.index._family", lambda keys: "flux")
    folder = tmp_path / "styles"
    folder.mkdir()
    torch.manual_seed(172)
    for i in range(2):
        save_file(
            {
                "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.randn(2, 4),
                "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.randn(4, 2),
            },
            str(folder / f"style{i}.safetensors"),
        )
    model = Flux()
    model.kilora_family = "flux"
    for block in model.double_blocks:
        block.img_attn.qkv = torch.nn.Linear(4, 12)
        block.img_attn.proj = torch.nn.Linear(4, 4)

        def forward(x, context, block=block):
            x = x + block.img_attn.qkv(x).reshape(*x.shape[:-1], 3, 4).sum(dim=-2) + context
            return x + block.img_mlp(x)

        block.forward = forward
    patcher = SamplingPatcher(FakeBaseModel(model))
    args = {
        "input": torch.randn(1, 5, 3),
        "timestep": torch.ones(1),
        "c": {"context": torch.randn(1, 5, 4)},
    }

    def sample(installed):
        wrapper = installed.model_options["model_function_wrapper"]
        outer = installed.wrappers["outer_sample", "kilora_library_probe"]
        return outer(lambda: wrapper(model, args))

    return folder, patcher, sample


def test_library_reuses_synced_hashes_and_reports_choices(
    library_sampling, monkeypatch, routing_loop
):
    import re

    from kilora import references

    folder, patcher, sample = library_sampling
    calls = []
    original = references._sha256

    def counted(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(references, "_sha256", counted)
    node = KiLoRALoRALibrary()
    installed, _, _ = node.apply(patcher, str(folder), "style", count=5, min_relevance=0)
    first = sample(installed)
    assert set(calls) == set(folder.glob("*.safetensors"))
    for path in folder.glob("*.safetensors"):
        assert re.search(
            rf"{re.escape(path.name)}: score -?\d+\.\d{{4}}, strength 0\.8", node.last_report
        )
    assert re.search(r"Read \d+\.\d{3}s; import \d+\.\d{3}s; key setup", node.last_report)
    calls.clear()
    from pathlib import Path

    opened = []
    original_open = Path.open

    def counted_open(path, *args, **kwargs):
        if path.parent == folder and path.suffix == ".safetensors":
            opened.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    installed, _, _ = node.apply(patcher, str(folder), "style", count=5, min_relevance=0)
    assert torch.equal(first, sample(installed))
    assert calls == []
    assert opened == []


def test_library_skips_file_changed_after_sync(library_sampling, monkeypatch):
    import os

    from kilora import references

    folder, patcher, sample = library_sampling
    node = KiLoRALoRALibrary()
    installed, _, _ = node.apply(patcher, str(folder), "style", count=5, min_relevance=0)
    path = folder / "style0.safetensors"
    before = path.stat()
    save_file(
        {
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(2, 4),
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
        },
        str(path),
    )
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    calls = []
    original = references._sha256

    def counted(file):
        calls.append(file)
        return original(file)

    monkeypatch.setattr(references, "_sha256", counted)
    sample(installed)
    assert calls.count(path.resolve()) == 1
    assert "style0.safetensors: skipped" in node.last_report
    assert "style1.safetensors: score" in node.last_report


def test_library_folder_order_does_not_change_output(library_sampling, tmp_path, routing_loop):
    import shutil

    folder, patcher, sample = library_sampling
    other = tmp_path / "reversed"
    other.mkdir()
    for i, path in enumerate(sorted(folder.glob("*.safetensors"), reverse=True)):
        shutil.copyfile(path, other / f"copy{i}.safetensors")
    node = KiLoRALoRALibrary()
    installed, _, _ = node.apply(patcher, str(folder), "", count=5, min_relevance=0)
    first = sample(installed)
    installed, _, _ = node.apply(patcher, str(other), "", count=5, min_relevance=0)
    assert torch.equal(first, sample(installed))
