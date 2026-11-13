# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from test_comfyui import FakeBaseModel, FakePatcher, denoiser, folders
from test_comfyui_progress import _karras

from kilora.comfyui import nodes as N

LAYERS = {
    "person": ("layers_0_attention_to_q", "layers_0_attention_to_k", "layers_1_attention_to_q"),
    "ink": ("layers_0_attention_to_q",),
    "wash": ("layers_0_attention_to_q", "layers_1_attention_to_v"),
    "line": ("layers_1_attention_to_q", "layers_0_attention_to_k"),
    "prop": ("layers_0_attention_to_q",),
}


def _save(dirs, name, seed):
    from safetensors.torch import save_file

    gen = torch.Generator().manual_seed(seed)
    tensors = {}
    for layer in LAYERS[name]:
        tensors[f"lora_unet_{layer}.lora_down.weight"] = torch.randn(2, 16, generator=gen)
        tensors[f"lora_unet_{layer}.lora_up.weight"] = torch.randn(16, 2, generator=gen) * 0.3
    save_file(tensors, str(dirs["loras"] / f"{name}.safetensors"))


def _run(patched, steps=8):
    wrapper = patched.model_options["model_function_wrapper"]
    sigmas = _karras(steps)
    x = torch.randn(2, 6, 4, generator=torch.Generator().manual_seed(3))
    out = []

    def apply_model(x, t, **c):
        return patched.run(x)

    for sigma in sigmas[:-1]:
        timestep = torch.full((2,), float(sigma))
        c = {
            "transformer_options": {
                "sample_sigmas": sigmas,
                "sigmas": timestep[:1],
                "cond_or_uncond": [0, 1],
            }
        }
        out.append(
            wrapper(
                apply_model, {"input": x, "timestep": timestep, "c": c, "cond_or_uncond": [0, 1]}
            )
        )
    return out


@pytest.mark.parametrize("ip_first", [True, False])
def test_one_ip_and_three_creative_loras_sample(denoiser, folders, ip_first):
    names = [("person", "ip"), ("ink", "creative"), ("wash", "creative"), ("line", "creative")]
    if not ip_first:
        names = names[1:] + names[:1]
    chain = None
    for i, (name, axis) in enumerate(names):
        _save(folders, name, 40 + i)
        (chain,) = N.KiLoRAImportLoRA().add(name + ".safetensors", 0.8, sets=chain, axis=axis)
    patched, report, _ = N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(denoiser)), chain, 3)
    out = _run(patched)
    assert all(torch.isfinite(y).all() for y in out)


def test_state_keys_find_the_probes_of_each_axis(denoiser, folders, monkeypatch):
    from kilora.comfyui import state_keys

    monkeypatch.setenv("KILORA_IMPORT_SCORE", "keys")

    seen = []
    real = state_keys.keys_from_records

    def spy(record, positions, *args, **kwargs):
        seen.append(
            {
                n: [e["queries"][positions[n]] for e in record[n] if e.get("queries") is not None]
                for n in positions
            }
        )
        return real(record, positions, *args, **kwargs)

    monkeypatch.setattr(state_keys, "keys_from_records", spy)
    names = [("person", "ip"), ("prop", "ip"), ("ink", "creative"), ("wash", "creative")]
    chain = None
    for i, (name, axis) in enumerate(names):
        _save(folders, name, 60 + i)
        (chain,) = N.KiLoRAImportLoRA().add(name + ".safetensors", 0.8, sets=chain, axis=axis)
    patched, _, _ = N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(denoiser)), chain, 3)
    _run(patched, steps=3)
    assert len(seen) == 2
    for found in seen:
        assert found and all(qs and all(q is not None for q in qs) for qs in found.values())


@pytest.mark.parametrize("ip_depth", ["1", "chain"])
def test_ip_lora_runs_at_its_strength_times_the_file_scale(
    denoiser,
    folders,
    monkeypatch,
    ip_depth,
):
    from safetensors.torch import save_file

    monkeypatch.setenv("KILORA_IP_DEPTH", ip_depth)
    gen = torch.Generator().manual_seed(7)
    down, up = torch.randn(2, 16, generator=gen), torch.randn(16, 2, generator=gen) * 0.3
    save_file(
        {
            "lora_unet_layers_0_attention_to_q.lora_down.weight": down,
            "lora_unet_layers_0_attention_to_q.lora_up.weight": up,
            "lora_unet_layers_0_attention_to_q.alpha": torch.tensor(1.0),
        },
        str(folders["loras"] / "person.safetensors"),
    )
    (chain,) = N.KiLoRAImportLoRA().add("person.safetensors", 1.0, axis="ip")
    patched, report, _ = N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(denoiser)), chain, 3)
    assert "file alpha/rank 0.5" in report
    base = denoiser.get_submodule("layers.0.attention.to_q")
    h = torch.randn(1, 5, 16)
    with torch.no_grad():
        change = patched.object_patches["diffusion_model.layers.0.attention.to_q"](h) - base(h)
    if ip_depth == "1":
        want = 0.5 * (h @ down.t()) @ up.t()
    else:
        from kilora import compute

        (eager_ip,) = N.KiLoRAImportLoRA().add("person.safetensors", 1.0, axis="ip")
        other, _, _ = N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(denoiser)), eager_ip, 3)
        with torch.no_grad(), compute.eager():
            want = other.object_patches["diffusion_model.layers.0.attention.to_q"](h) - base(h)
    torch.testing.assert_close(change, want, rtol=1e-5, atol=1e-6)


def test_auto_alpha_report_names_what_is_missing(denoiser, folders):
    _save(folders, "person", 80)
    (chain,) = N.KiLoRAImportLoRA().add("person.safetensors", 1.0, axis="ip")
    _, report, _ = N.KiLoRAApply().apply(
        FakePatcher(FakeBaseModel(denoiser)), chain, 3, axis_alpha="auto"
    )
    assert "no CLIP reaches KiLoRA Apply" in report
    assert "Anima only" in report


def test_trigger_position_messages():
    from kilora.attention_maps import trigger_positions

    ids = {"t5xxl_ids": torch.tensor([3, 9, 4514, 122, 12840, 6, 1])}
    found, message = trigger_positions("anima", ids, {"bangboo.safetensors": [4514, 122, 12840]})
    assert found == {"bangboo.safetensors": (2, 3, 4)} and message is None
    _, message = trigger_positions("anima", {}, {"a": [5]})
    assert "no t5xxl_ids" in message
    _, message = trigger_positions("anima", ids, {"/x/tataru.safetensors": [77]})
    assert "tataru.safetensors is not in the prompt" in message
