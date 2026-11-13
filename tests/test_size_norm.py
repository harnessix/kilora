# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import math

import pytest
import torch
from test_multi_import import LAYERS, ToyAnima, lora_weights, to_keys, write

from kilora import compose
from kilora.lora_import import import_lora
from kilora.multi_import import _SIZES, import_loras, lora_file_size, lycoris_delta
from kilora.size_norm import S_REF, dora_delta, layer_delta_sq, low_rank_sq, size_scale

Q = "blocks.0.self_attn.q_proj"
QK = "lora_unet_blocks_0_self_attn_q_proj"


@pytest.fixture
def model():
    return ToyAnima().eval()


def _sets(result):
    return {t: (ex.down.clone(), ex.up.clone()) for t, ex in result.adapter_set.experts.items()}


def _delta(result, target, e=0):
    ex = result.adapter_set.experts[target]
    return ex.up[e, 0].t().double() @ ex.down[e, 0].double()


def test_s_ref_is_the_library_median():
    table = [
        0.0026,
        0.0137,
        0.0094,
        0.0095,
        0.0103,
        0.0115,
        0.0109,
        0.0180,
        0.0200,
        0.0177,
        0.0274,
        0.0365,
        0.1061,
        0.0679,
        0.0589,
        0.0641,
    ]
    values = sorted(table)
    assert len(values) == 16
    assert pytest.approx((values[7] + values[8]) / 2) == S_REF


def test_trace_form_equals_the_built_low_rank_change():
    g = torch.Generator().manual_seed(0)
    down, up = torch.randn(4, 20, generator=g), torch.randn(12, 4, generator=g)
    want = float(((0.7 * up.double() @ down.double()) ** 2).sum())
    assert low_rank_sq(down, up, 0.7) == pytest.approx(want, rel=1e-10)
    tensors = {"down": down, "up": up}
    assert layer_delta_sq(tensors, 0.7) == pytest.approx(want, rel=1e-10)


def test_lokr_size_equals_the_built_change():
    g = torch.Generator().manual_seed(1)
    tensors = {
        "lokr_w1": torch.randn(4, 4, generator=g),
        "lokr_w2_a": torch.randn(4, 2, generator=g),
        "lokr_w2_b": torch.randn(2, 4, generator=g),
        "alpha": torch.tensor(1.0),
    }
    built = lycoris_delta(tensors)[0].double()
    assert layer_delta_sq(tensors) == pytest.approx(float((built**2).sum()), rel=1e-5)


def test_loha_size_equals_the_built_change():
    g = torch.Generator().manual_seed(2)
    shapes = {"w1_a": (16, 2), "w1_b": (2, 16), "w2_a": (16, 2), "w2_b": (2, 16)}
    tensors = {f"hada_{n}": torch.randn(*s, generator=g) for n, s in shapes.items()}
    tensors["alpha"] = torch.tensor(2.0)
    built = lycoris_delta(tensors)[0].double()
    assert layer_delta_sq(tensors) == pytest.approx(float((built**2).sum()), rel=1e-5)


def test_dora_size_uses_the_real_base_weight():
    g = torch.Generator().manual_seed(3)
    w0 = torch.randn(12, 20, generator=g)
    down, up = torch.randn(4, 20, generator=g) * 0.1, torch.randn(12, 4, generator=g) * 0.1
    m = torch.rand(1, 20, generator=g) + 0.5
    calc = w0.double() + 0.5 * (up.double() @ down.double())
    built = m.double() * calc / calc.norm(dim=0, keepdim=True) - w0.double()
    tensors = {"down": down, "up": up, "dora_scale": m}
    got = layer_delta_sq(tensors, 0.5, w0)
    assert got == pytest.approx(float((built**2).sum()), rel=1e-8)
    torch.testing.assert_close(dora_delta(w0, 0.5 * up @ down, m), built)
    assert layer_delta_sq(tensors, 0.5, None) is None


def test_file_size_is_the_relative_frobenius_norm(model, tmp_path):
    w = lora_weights(model, rank=4)
    path = write(tmp_path, "a", to_keys(w, "kohya", alpha=2.0))
    num = sum(float(((0.5 * b.double() @ a.double()) ** 2).sum()) for a, b in w.values())
    den = sum(float((model.get_submodule(n).weight.double() ** 2).sum()) for n in w)
    assert lora_file_size(path, model) == pytest.approx(math.sqrt(num / den), rel=1e-6)


def test_file_size_is_kept_for_an_import_without_base_weights(model, tmp_path):
    w = lora_weights(model, rank=4, seed=5)
    path = write(tmp_path, "kept", to_keys(w, "kohya", alpha=4.0))
    meta = copy.deepcopy(model).to("meta")
    _SIZES.clear()
    assert lora_file_size(path, meta) is None
    size = lora_file_size(path, model)
    assert lora_file_size(path, meta) == size
    assert size_scale(None) == 1.0


def test_off_is_bit_identical_over_several_calls(model, tmp_path):
    a = write(tmp_path, "a", to_keys(lora_weights(model, rank=4, seed=1), "kohya", alpha=2.0))
    b = write(tmp_path, "b", to_keys(lora_weights(model, rank=3, seed=2), "peft"))
    for files, mix in (([a], "stack"), ([a, b], None)):
        plain = import_loras(files, model)
        off = import_loras(files, model, size_norm="off")
        assert plain.report() == off.report()
        for target, (down, up) in _sets(plain).items():
            assert torch.equal(down, off.adapter_set.experts[target].down)
            assert torch.equal(up, off.adapter_set.experts[target].up)
        outs = []
        for result in (plain, off):
            m = copy.deepcopy(model)
            kw = {"mix": mix} if mix else {}
            compose(m, [result.adapter_set], **kw)
            gen = torch.Generator().manual_seed(4)
            x, ctx = torch.randn(2, 5, 16, generator=gen), torch.randn(2, 3, 8, generator=gen)
            with torch.no_grad():
                outs.append([m(x, ctx) for _ in range(3)])
        for one, two in zip(*outs, strict=True):
            assert torch.equal(one, two)


def test_off_plain_import_is_bit_identical(model, tmp_path):
    path = write(tmp_path, "p", to_keys(lora_weights(model, rank=4), "kohya", alpha=2.0))
    plain = import_lora(path, model, experts=1, weight=0.7)
    off = import_lora(path, model, experts=1, weight=0.7, size_norm="off")
    assert plain.adapter_set.strength == off.adapter_set.strength
    assert plain.report() == off.report()
    for target, ex in plain.adapter_set.experts.items():
        assert torch.equal(ex.up, off.adapter_set.experts[target].up)


def test_the_scale_is_one_number_per_file(model, tmp_path):
    w = lora_weights(model, rank=4, seed=7)
    path = write(tmp_path, "one", to_keys(w, "kohya", alpha=2.0))
    off = import_loras([path], model, size_norm="off")
    on = import_loras([path], model, size_norm="weight")
    c = on.sources[0]["size_scale"]
    assert c == pytest.approx(S_REF / lora_file_size(path, model))
    a = min(c, 1.0)
    assert on.sources[0]["weight"] == pytest.approx(a)
    for target in LAYERS:
        torch.testing.assert_close(_delta(on, target), a * _delta(off, target))
    assert "S_f" in on.report() and "c_f" in on.report()
    assert "S_f" not in off.report()


def test_two_files_of_different_size_are_shrunk_to_at_most_the_unit_size(model, tmp_path):
    small_keys = to_keys(lora_weights(model, rank=4, seed=1), "kohya", alpha=0.5)
    large_keys = to_keys(lora_weights(model, rank=2, seed=2), "kohya", alpha=8.0)
    small, large = write(tmp_path, "small", small_keys), write(tmp_path, "large", large_keys)
    s_small, s_large = lora_file_size(small, model), lora_file_size(large, model)
    assert s_large > 4 * s_small
    assert s_small * size_scale(s_small) == pytest.approx(s_large * size_scale(s_large))
    on = import_loras([small, large], model, size_norm="weight")
    den = sum(float((model.get_submodule(n).weight.double() ** 2).sum()) for n in LAYERS)
    for e, size in ((0, s_small), (1, s_large)):
        num = sum(float((_delta(on, t, e) ** 2).sum()) for t in LAYERS)
        assert math.sqrt(num / den) == pytest.approx(min(size, S_REF), rel=1e-4)


def _scaled_file(model, tmp_path, name, times):
    w = {n: (d, u * times) for n, (d, u) in lora_weights(model, rank=4, seed=5).items()}
    return write(tmp_path, name, to_keys(w, "kohya", alpha=4.0))


def test_import_shrinks_a_strong_file_and_keeps_a_weak_file(model, tmp_path):
    unit = _scaled_file(model, tmp_path, "unit", 1.0)
    s_unit = lora_file_size(unit, model)
    weak = _scaled_file(model, tmp_path, "weak", 0.2 * S_REF / s_unit)
    strong = _scaled_file(model, tmp_path, "strong", 5.0 * S_REF / s_unit)
    c_weak = size_scale(lora_file_size(weak, model))
    c_strong = size_scale(lora_file_size(strong, model))
    assert c_weak > 1.0 > c_strong
    off = import_loras([weak, strong], model, size_norm="off")
    on = import_loras([weak, strong], model, size_norm="weight")
    assert on.sources[0]["size_scale"] == pytest.approx(c_weak)
    assert on.sources[0]["weight"] == pytest.approx(1.0)
    assert on.sources[1]["size_scale"] == pytest.approx(c_strong)
    assert on.sources[1]["weight"] == pytest.approx(c_strong)
    for target in LAYERS:
        torch.testing.assert_close(_delta(on, target, 0), _delta(off, target, 0))
        torch.testing.assert_close(_delta(on, target, 1), c_strong * _delta(off, target, 1))
    assert "shrink" in on.report() and "applied scale s" in on.report()


def test_plain_import_scales_the_strength(model, tmp_path):
    path = write(tmp_path, "s", to_keys(lora_weights(model, rank=4), "kohya", alpha=2.0))
    off = import_lora(path, model, experts=1, weight=0.7)
    on = import_lora(path, model, experts=1, weight=0.7, size_norm="weight")
    c = size_scale(lora_file_size(path, model))
    assert on.adapter_set.strength == pytest.approx(off.adapter_set.strength * min(c, 1.0))
    assert "S_f" in on.report()


def test_plain_import_shrinks_a_strong_file_and_keeps_a_weak_file(model, tmp_path):
    unit = _scaled_file(model, tmp_path, "unit", 1.0)
    s_unit = lora_file_size(unit, model)
    weak = _scaled_file(model, tmp_path, "weak", 0.2 * S_REF / s_unit)
    strong = _scaled_file(model, tmp_path, "strong", 5.0 * S_REF / s_unit)
    c_weak = size_scale(lora_file_size(weak, model))
    c_strong = size_scale(lora_file_size(strong, model))
    assert c_weak > 1.0 > c_strong
    for path, c, want in ((weak, c_weak, 1.0), (strong, c_strong, c_strong)):
        off = import_lora(path, model, experts=1, weight=0.7)
        on = import_lora(path, model, experts=1, weight=0.7, size_norm="weight")
        assert on.adapter_set.strength == pytest.approx(off.adapter_set.strength * want)
        source = on.adapter_set.imported_from
        assert source["size_scale"] == pytest.approx(c)
        assert source["applied_scale"] == pytest.approx(want)
        assert source["size_rule"] == "shrink"
        assert "applied scale s" in on.report() and "c_f" in on.report()


def test_unknown_size_norm_stops(model, tmp_path):
    path = write(tmp_path, "u", to_keys(lora_weights(model), "kohya"))
    with pytest.raises(ValueError):
        import_loras([path], model, size_norm="output")


def test_import_lora_node_keeps_main_entries_when_off(monkeypatch):
    from kilora.comfyui import nodes as N

    monkeypatch.setattr(N, "_file", lambda folder, name, kind: f"/x/{name}")
    off = N.KiLoRAImportLoRA().add("a.safetensors", 0.7)[0]
    assert type(off[0]) is N.ModEntry
    on = N.KiLoRAImportLoRA().add("a.safetensors", 0.7, size_norm="weight")[0]
    assert isinstance(on[0], N.SizedLoRAEntry) and on[0].size_norm == "weight"
    assert on[0].strength == off[0].strength
