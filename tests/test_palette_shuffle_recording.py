# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from collections import Counter
from unittest.mock import patch

import pytest
import torch
from test_families import MLP, H, _ab

from kilora import compute, fused

SHARE_SAMPLES = 4
REFUSALS = [1, 1, 1, 0, 0, 0]
SCALAR_READS = 2
RECORDING = "routes or gates are being recorded"
STEPS = 6
FREEZE = 0.4
SIGMAS = [float(STEPS - i) for i in range(STEPS)] + [0.0]


def _setup(tmp_path):
    from safetensors.torch import save_file
    from test_comfyui import FakeBaseModel, FakePatcher
    from test_families import _tree

    from kilora.comfyui import patching

    tmp_path.mkdir(parents=True, exist_ok=True)
    layers = {f"blocks.0.self_attn.{p}": (H, H) for p in ("q_proj", "k_proj", "v_proj")}
    layers["blocks.0.self_attn.output_proj"] = (H, H)
    layers["blocks.0.mlp.layer1"] = (H, MLP)
    layers["blocks.0.mlp.layer2"] = (MLP, H)
    layers["blocks.0.adaln_modulation_cross_attn.1"] = (H, 3 * H)
    layers["llm_adapter.proj"] = (H, H)
    model = _tree("Anima", layers)
    gen = torch.Generator().manual_seed(811)
    paths = []
    for i, strength in enumerate((0.6, 0.85, 1.0)):
        state = {}
        for path, dim in [
            ("blocks_0_self_attn_q_proj", H),
            ("blocks_0_self_attn_v_proj", H),
            ("blocks_0_adaln_modulation_cross_attn_1", 3 * H),
            ("llm_adapter_proj", H),
        ]:
            a, b = _ab(gen, H, dim)
            state[f"lora_unet_{path}.lora_down.weight"] = a.contiguous()
            state[f"lora_unet_{path}.lora_up.weight"] = (b * strength).contiguous()
        path = tmp_path / f"pick{i}.safetensors"
        save_file(state, str(path))
        paths.append(str(path))
    chain = patching.ModChain(tuple(patching.ModEntry("lora", p, axis="creative") for p in paths))
    patched, _ = patching.apply(
        FakePatcher(FakeBaseModel(model)), chain, 1, joint=True, freeze=FREEZE
    )
    layers = [
        patched.object_patches[f"diffusion_model.blocks.0.self_attn.{p}"]
        for p in ("q_proj", "v_proj")
    ]
    mounts = [
        patched.object_patches[f"diffusion_model.{p}"]
        for p in ("blocks.0.adaln_modulation_cross_attn.1", "llm_adapter.proj")
        if f"diffusion_model.{p}" in patched.object_patches
    ]
    return patched.model_options["model_function_wrapper"], layers, mounts


def _generate(tmp_path, monkeypatch, *, fused_check: bool, sync_counts=None):
    from kilora.call_values import current_mount_share
    from kilora.comfyui import patching

    monkeypatch.setattr(fused, "_NOTED", set())
    reasons: list[str] = []
    recorders: list[bool] = []
    if fused_check:
        pytest.importorskip("triton")
        monkeypatch.setenv("KILORA_FUSED", "1")
        monkeypatch.setattr(fused, "HAVE_TRITON", True)
        monkeypatch.setattr(fused, "device_ok", lambda t: True)
        monkeypatch.setattr(fused, "fits", lambda *a, **k: True)
        original_note = fused._note

        def note(reason):
            reasons.append(reason)
            original_note(reason)

        monkeypatch.setattr(fused, "_note", note)
        original_ok = fused._fused_ok

        def checked(y0, sets, depth=None, td=None):
            recorders.append(bool(compute._SET_SHARES or compute._SHARES or compute._ROUTES))
            if original_ok(y0, sets, depth, td):
                reasons.append("fused")
            return False

        monkeypatch.setattr(fused, "_fused_ok", checked)
    else:
        monkeypatch.setenv("KILORA_FUSED", "0")
    wrapper, layers, mounts = _setup(tmp_path)
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(2, 4, H, generator=gen)
    m = torch.randn(2, H, generator=gen)
    per_step = []
    sample_sigmas = torch.tensor(SIGMAS)

    def apply_model(inp, timestep, **kwargs):
        shares = [current_mount_share(patching.JOINT_SET, i, 3) for i in range(3)]
        outs = [layer(x) for layer in layers for _ in range(3)]
        outs += [mount(m) for mount in mounts]
        return outs, shares

    with torch.no_grad():
        for step in range(STEPS):
            reasons.clear()
            recorders.clear()
            args = {
                "input": torch.zeros(2, 4, 4, 4),
                "timestep": torch.tensor([SIGMAS[step]]),
                "c": {"transformer_options": {"sample_sigmas": sample_sigmas}},
            }
            if sync_counts is not None and step > 0:
                sync_counts.append([])
                originals = {
                    n: getattr(torch.Tensor, n)
                    for n in ("item", "__bool__", "__float__", "__int__", "tolist")
                }

                def wrap(name, sink=sync_counts[-1], originals=originals):
                    def counted(tensor, *a, **k):
                        sink.append(name)
                        return originals[name](tensor, *a, **k)

                    return counted

                with patch.multiple(torch.Tensor, **{n: wrap(n) for n in originals}):
                    result = wrapper(apply_model, args)
            else:
                result = wrapper(apply_model, args)
            per_step.append((Counter(reasons), sum(recorders), result))
    return per_step


def _progress(step):
    return step / (STEPS - 1)


def test_palette_shuffle_records_only_for_the_share_before_the_freeze(tmp_path, monkeypatch):
    steps = _generate(tmp_path, monkeypatch, fused_check=True)
    assert len(steps) == len(REFUSALS)
    for step, (reasons, recording, _) in enumerate(steps):
        print(f"step {step} progress {_progress(step):.2f}: {dict(reasons)}, recorders {recording}")
        assert reasons[RECORDING] == REFUSALS[step], (step, reasons)
        assert reasons[RECORDING] <= SHARE_SAMPLES
        if _progress(step) > FREEZE:
            assert recording == 0, (step, recording)


def test_palette_shuffle_warm_steps_after_the_freeze_read_few_host_scalars(tmp_path, monkeypatch):
    syncs: list[list[str]] = []
    _generate(tmp_path, monkeypatch, fused_check=False, sync_counts=syncs)
    for step, counts in enumerate(syncs, start=1):
        scalars = [n for n in counts if n != "tolist"]
        print(f"step {step} host scalar reads: {len(scalars)} {Counter(counts)}")
        if _progress(step - 1) >= FREEZE:
            assert len(scalars) <= SCALAR_READS, (step, counts)
            assert "tolist" not in counts, (step, counts)


@pytest.mark.parametrize("fused_check", [False, True])
def test_palette_shuffle_outputs_and_shares_repeat_bit_identically(
    tmp_path, monkeypatch, fused_check
):
    first = _generate(tmp_path / "a", monkeypatch, fused_check=fused_check)
    second = _generate(tmp_path / "b", monkeypatch, fused_check=fused_check)
    for (_, _, (outs_a, shares_a)), (_, _, (outs_b, shares_b)) in zip(first, second, strict=True):
        for a, b in zip(outs_a, outs_b, strict=True):
            assert torch.equal(a, b)
        for a, b in zip(shares_a, shares_b, strict=True):
            assert torch.equal(torch.as_tensor(a), torch.as_tensor(b))


def test_palette_shuffle_fused_check_keeps_eager_values(tmp_path, monkeypatch):
    eager = _generate(tmp_path / "a", monkeypatch, fused_check=False)
    checked = _generate(tmp_path / "b", monkeypatch, fused_check=True)
    for (_, _, (outs_a, shares_a)), (_, _, (outs_b, shares_b)) in zip(eager, checked, strict=True):
        for a, b in zip(outs_a, outs_b, strict=True):
            assert torch.equal(a, b)
        for a, b in zip(shares_a, shares_b, strict=True):
            assert torch.equal(torch.as_tensor(a), torch.as_tensor(b))
