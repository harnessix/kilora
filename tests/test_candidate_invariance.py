# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from safetensors.torch import save_file
from test_comfyui import FakeBaseModel, FakePatcher, denoiser, folders
from test_import_modes import DIM, LAYERS, Toy, _lora

from kilora import compose, compute, key_cache
from kilora.comfyui import nodes as N
from kilora.multi_import import import_loras
from kilora.testing import TinyDenoiser

TOL = dict(rtol=1e-5, atol=1e-6)


@pytest.fixture(autouse=True)
def own_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "key-cache"))


def _three(tmp_path):
    model = Toy().eval()
    paths = []
    for i, (seed, layers) in enumerate([(11, LAYERS), (29, LAYERS), (5, LAYERS[:2])]):
        p = tmp_path / f"l{i}.safetensors"
        _lora(model, p, rank=3, seed=seed, alpha=3.0, layers=layers)
        paths.append(str(p))
    return model, paths


def _x():
    return torch.randn(1, 24, DIM, generator=torch.Generator().manual_seed(4))


def _out(model, paths, order, x, depth=3):
    m = Toy().eval()
    adapter = compose(m, [import_loras([paths[i] for i in order], model).adapter_set], depth=depth)
    with torch.no_grad():
        y = m(x)
        y = m(x)
    pairs = zip(adapter.targets, adapter.sites, strict=True)
    keys = {t: s.tensors["s0_keys_in"].detach().clone() for t, s in pairs}
    adapter.remove()
    return y, keys


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("score", ["keys", "read_share"])
@pytest.mark.parametrize("depth", [1, 3])
def test_shuffled_loras_give_the_same_output_and_keys(tmp_path, monkeypatch, fast, score, depth):
    monkeypatch.setattr(compute, "LOWRANK_ENABLED", fast)
    monkeypatch.setenv("KILORA_IMPORT_SCORE", score)
    model, paths = _three(tmp_path)
    x = _x()
    y, keys = _out(model, paths, [0, 1, 2], x, depth)
    order = [2, 0, 1]
    y2, keys2 = _out(model, paths, order, x, depth)
    print(f"PERM score={score} depth={depth} fast={fast}: {float((y2 - y).abs().max()):.2e}")
    torch.testing.assert_close(y2, y, **TOL)
    for t in keys:
        a = compute.orthonormal_keys(keys[t])[order]
        b = compute.orthonormal_keys(keys2[t])
        proj = torch.einsum("emr,enr->emn", a, b)
        assert torch.allclose(
            proj.pow(2).sum((-1, -2)), torch.full((3,), float(a.shape[1])), atol=1e-4
        )
    torch.testing.assert_close(_out(model, paths, [0, 1, 2], x, depth)[0], y, **TOL)
    torch.testing.assert_close(_out(model, paths, order, x, depth)[0], y, **TOL)


def test_shuffled_chain_gives_the_same_output_in_comfyui(folders):
    den = TinyDenoiser(seed=0).eval()
    den.kilora_family = "kilora-tiny-1"
    names = []
    for i, seed in enumerate((3, 8, 13)):
        gen = torch.Generator().manual_seed(seed)
        stem = "lora_unet_layers_0_attention_to_q"
        sd = {
            f"{stem}.lora_down.weight": torch.randn(2, 16, generator=gen),
            f"{stem}.lora_up.weight": torch.randn(16, 2, generator=gen),
            f"{stem}.alpha": torch.tensor(1.0),
        }
        save_file(sd, str(folders["loras"] / f"c{i}.safetensors"))
        names.append(f"c{i}.safetensors")
    x = torch.randn(1, 6, 4, generator=torch.Generator().manual_seed(2))

    def run(order):
        chain = None
        for i in order:
            (chain,) = N.KiLoRAImportLoRA().add(names[i], 0.8, sets=chain)
        patched, _, _ = N.KiLoRAApply().apply(FakePatcher(FakeBaseModel(den)), chain, 3)
        return patched.run(x)

    torch.testing.assert_close(run([2, 0, 1]), run([0, 1, 2]), **TOL)


def _write(path, a, b, layer="blocks.0.mlp.layer1"):
    stem = "lora_unet_" + layer.replace(".", "_")
    save_file(
        {
            f"{stem}.lora_down.weight": a.contiguous(),
            f"{stem}.lora_up.weight": b,
            f"{stem}.alpha": torch.tensor(float(a.shape[0])),
        },
        str(path),
    )


@pytest.mark.parametrize("score", ["keys", "read_share"])
def test_an_unrelated_lora_barely_moves_the_others(tmp_path, monkeypatch, score):
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "1")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", score)
    g = torch.Generator().manual_seed(6)
    basis, _ = torch.linalg.qr(torch.randn(DIM, DIM, generator=g))
    inside, outside = basis[:, :8], basis[:, 8:]
    layer_input = (torch.randn(1, 48, 8, generator=g) @ inside.t()) * 2
    model = Toy().eval()
    q = model.blocks[0].self_attn.q_proj
    x = torch.linalg.solve(
        torch.eye(DIM) + q.weight, (layer_input - q.bias).transpose(-1, -2)
    ).transpose(-1, -2)
    paths = []
    for i in range(2):
        a = torch.randn(3, DIM, generator=g) * 0.3
        _write(tmp_path / f"r{i}.safetensors", a, torch.randn(20, 3, generator=g) * 0.3)
        paths.append(str(tmp_path / f"r{i}.safetensors"))
    unrelated = tmp_path / "unrelated.safetensors"
    unrelated_down = (torch.randn(3, 4, generator=g) @ outside.t()) * 0.3
    assert (layer_input @ unrelated_down.t()).abs().max() < 1e-5
    _write(unrelated, unrelated_down, torch.randn(20, 3, generator=g) * 0.3)

    def change(files):
        m = Toy().eval()
        seen = {}
        q = m.blocks[0].mlp.layer1
        layer_input = x + m.blocks[0].self_attn.q_proj(x)
        base = q(layer_input).detach()
        adapter = compose(m, [import_loras(files, model).adapter_set], depth=3)
        hook = q.register_forward_hook(lambda mod, inp, out: seen.__setitem__("y", out.detach()))
        with torch.no_grad():
            m(x)
            m(x)
        hook.remove()
        adapter.remove()
        return seen["y"] - base

    two = change(paths)
    three = change([*paths, str(unrelated)])
    drift = float((three - two).norm() / two.norm())
    print(f"DRIFT score={score}: {drift:.4f}")
    assert two.norm() > 0 and drift < DRIFT_BOUND[score]


DRIFT_BOUND = {"keys": 0.35, "read_share": 0.16}


@pytest.mark.parametrize("score", ["keys", "read_share"])
def test_unread_expert_cannot_add_to_the_chain(monkeypatch, score):
    import dataclasses

    from test_share_transition import planted

    monkeypatch.setenv("KILORA_FUSED", "0")
    monkeypatch.setenv("KILORA_IMPORT_FIXED_READ", "0")
    monkeypatch.setenv("KILORA_IMPORT_SCORE", score)
    monkeypatch.setenv("KILORA_MASKS", "expect")
    s = dataclasses.replace(
        planted(),
        router=torch.eye(3),
        keys_in=torch.eye(3).unsqueeze(1),
        down=torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        up=torch.eye(3),
        routing_select="top1",
        carry="drop",
    )
    x = torch.tensor([[[1.0, 0.2], [0.2, 1.0]]])
    base = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    with torch.no_grad():
        actual = compute.run_layer(x, base, [s], 7)
        changed = dataclasses.replace(s, up=s.up.clone())
        changed.up[2] *= 100
        expected = compute.run_layer(x, base, [changed], 7)
    assert torch.equal(actual, expected)
    assert torch.equal(actual[..., 2], base[..., 2])
