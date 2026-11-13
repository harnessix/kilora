# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import pytest
import torch
from safetensors.torch import save_file
from test_multi_import import ToyAnima, lora_weights, to_keys, write

from kilora import key_cache
from kilora import multi_import as MI
from kilora import size_norm as SN
from kilora.comfyui import nodes as N


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    folder = tmp_path / "key-cache"
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(folder))
    monkeypatch.delenv("KILORA_SIZE_CACHE", raising=False)
    MI._STORED_SIZES.clear()
    MI._SIZES.clear()
    SN._BASE_NORMS.clear()
    yield folder
    MI._STORED_SIZES.clear()
    SN._BASE_NORMS.clear()


@pytest.fixture
def model():
    return ToyAnima().eval()


def _weights(model):
    return {n: m.weight for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}


def _base(model):
    weights = _weights(model)
    norms = SN.base_norms(weights)
    return norms, SN.base_identity(norms, {t: tuple(w.shape) for t, w in weights.items()})


@pytest.fixture
def counted(monkeypatch):
    calls = []
    real = MI._lora_file_size

    def wrapped(*a, **k):
        calls.append(a[0])
        return real(*a, **k)

    monkeypatch.setattr(MI, "_lora_file_size", wrapped)
    return calls


@pytest.mark.parametrize("budget", [0.0, 1.0, 2.0, 10.0])
@pytest.mark.parametrize("c", [0.17, 1.0, 6.9])
def test_the_scale_shrinks_only_for_every_radial_budget(c, budget):
    s, named = SN.applied_scale(c, budget)
    assert s == pytest.approx(min(c, 1.0)) and named == "shrink"
    assert SN.applied_scale(c) == (s, named)


def test_the_report_line_names_size_scale_applied_scale_and_rule():
    line = SN.size_note("anrower.safetensors", 0.0026, 6.9, 1.0, "shrink")
    for part in ("S_f 0.0026", "c_f 6.9", "s 1", "shrink", "anrower.safetensors"):
        assert part in line
    unknown = SN.size_note("x", None, 1.0, 1.0, "shrink")
    assert "S_f unknown" in unknown and "shrink" in unknown


def test_size_normed_chain_applies_the_rule(monkeypatch, model):
    P = N.patching
    weights = {t: w.detach() for t, w in _weights(model).items()}

    class Binding:
        family = "anima"

    class Full:
        diffusion_model = model

    class Patcher:
        model = Full()

    monkeypatch.setattr(P, "detect_family", lambda m: Binding())
    monkeypatch.setattr(P, "diffusers_shapes", lambda m, b: {})
    monkeypatch.setattr(P, "shape_model", lambda shapes: model)
    monkeypatch.setattr(P, "host_weights", lambda m, shapes: weights)
    sizes = {"/x/weak.safetensors": SN.S_REF / 6.9, "/x/strong.safetensors": SN.S_REF / 0.28}
    monkeypatch.setattr(MI, "lora_file_size", lambda path, *a, **k: sizes[path])
    chain = [
        P.ModEntry(kind="lora", path="/x/weak.safetensors", strength=0.7),
        P.ModEntry(kind="lora", path="/x/strong.safetensors", strength=0.7),
        P.ModEntry(kind="mod", path="/x/m.kilora", strength=0.5),
    ]
    for budget in (0.0, 2.0, 8.0):
        out, lines = N.size_normed_chain(Patcher(), chain, True, radial_budget=budget)
        got = [e.strength for e in out]
        assert got[0] == pytest.approx(0.7)
        assert got[1] == pytest.approx(0.7 * 0.28)
        assert got[2] == 0.5
        assert "shrink" in lines[0] and "shrink" in lines[1]
        assert not any("budget cap" in x or "(full)" in x for x in lines)
        assert all("S_f" in x and "c_f" in x and "applied scale s" in x for x in lines)


def test_a_second_call_computes_no_size(model, tmp_path, counted, cache):
    path = write(tmp_path, "a", to_keys(lora_weights(model, rank=4), "kohya", alpha=2.0))
    norms, base_id = _base(model)
    first = MI.lora_file_size(path, model, base_norms=norms, base_id=base_id)
    assert first is not None and len(counted) == 1
    assert MI.lora_file_size(path, model, base_norms=norms, base_id=base_id) == first
    assert len(counted) == 1
    assert any(name.endswith(".json") for name in os.listdir(cache / "size"))


def test_a_new_process_reads_the_size_from_disk(model, tmp_path, counted):
    path = write(tmp_path, "a", to_keys(lora_weights(model, rank=4), "kohya", alpha=2.0))
    norms, base_id = _base(model)
    first = MI.lora_file_size(path, model, base_norms=norms, base_id=base_id)
    MI._STORED_SIZES.clear()
    MI._SIZES.clear()
    SN._BASE_NORMS.clear()
    norms, again = _base(model)
    assert again == base_id
    assert MI.lora_file_size(path, model, base_norms=norms, base_id=base_id) == first
    assert len(counted) == 1


def test_a_changed_file_or_another_base_misses(model, tmp_path, counted):
    path = write(tmp_path, "a", to_keys(lora_weights(model, rank=4), "kohya", alpha=2.0))
    norms, base_id = _base(model)
    first = MI.lora_file_size(path, model, base_norms=norms, base_id=base_id)
    other = ToyAnima(seed=3).eval()
    other_norms, other_id = _base(other)
    assert other_id != base_id
    second = MI.lora_file_size(path, other, base_norms=other_norms, base_id=other_id)
    assert len(counted) == 2 and second != first
    write(tmp_path, "a", to_keys(lora_weights(model, rank=4, seed=9), "kohya", alpha=2.0))
    key_cache._FILE_HASHES.clear()
    changed = MI.lora_file_size(path, model, base_norms=norms, base_id=base_id)
    assert len(counted) == 3 and changed != first


def test_base_norms_give_the_same_size_as_the_weights(model, tmp_path):
    path = write(tmp_path, "a", to_keys(lora_weights(model, rank=4), "kohya", alpha=2.0))
    norms, _ = _base(model)
    plain = MI.lora_file_size(path, model)
    assert MI.lora_file_size(path, model, base_norms=norms) == pytest.approx(plain, rel=1e-5)


def test_base_norms_are_computed_once_per_base(model, monkeypatch):
    calls = []
    real = SN.weight_sq
    monkeypatch.setattr(SN, "weight_sq", lambda w: calls.append(1) or real(w))
    weights = _weights(model)
    SN.base_norms(weights)
    n = len(calls)
    assert n == len(weights)
    SN.base_norms(weights)
    assert len(calls) == n
    with torch.no_grad():
        next(iter(weights.values())).mul_(2.0)
    SN.base_norms(weights)
    assert len(calls) == 2 * n


def test_the_cache_can_be_switched_off(model, tmp_path, counted, monkeypatch, cache):
    monkeypatch.setenv("KILORA_SIZE_CACHE", "0")
    path = write(tmp_path, "a", to_keys(lora_weights(model, rank=4), "kohya", alpha=2.0))
    norms, base_id = _base(model)
    MI.lora_file_size(path, model, base_norms=norms, base_id=base_id)
    assert not (cache / "size").exists()


def test_the_worker_stores_the_size_from_the_base_file(model, tmp_path, counted):
    from kilora.prepare import base_file_norms, prepare_sizes

    base = tmp_path / "anima_base.safetensors"
    save_file(
        {f"model.diffusion_model.{k}": v.contiguous() for k, v in model.state_dict().items()},
        str(base),
    )
    path = write(tmp_path, "a", to_keys(lora_weights(model, rank=4), "kohya", alpha=2.0))
    sizes = prepare_sizes([path], str(base), family="anima")
    assert sizes[path] is not None and sizes[path] > 0.0
    assert len(counted) == 1
    family, norms, shapes, _ = base_file_norms(str(base), family="anima")
    assert family == "anima" and norms
    stored = MI.cached_lora_size(
        key_cache.file_sha256(path),
        SN.base_identity(norms, shapes),
        MI.lora_size_variant(path, "anima"),
    )
    assert stored == sizes[path]
    prepare_sizes([path], str(base), family="anima")
    assert len(counted) == 1
    assert MI.lora_file_size(path, model) == pytest.approx(stored, rel=1e-4)
    assert len(counted) == 1


def test_the_worker_lists_the_named_base_files(tmp_path, monkeypatch):
    from kilora.comfyui import startup_prepare as SP

    base = tmp_path / "b.safetensors"
    base.write_bytes(b"")
    monkeypatch.setenv("KILORA_SIZE_BASE", os.pathsep.join([str(base), str(tmp_path / "no")]))
    assert SP.base_model_files() == [str(base)]


def test_the_base_identity_ignores_target_names_but_not_the_weights(model):
    norms, base_id = SN.base_key(_weights(model))
    shapes = {t: tuple(w.shape) for t, w in _weights(model).items()}
    renamed = {f"t{i}": v for i, v in enumerate(reversed(list(norms.values())))}
    renamed_shapes = {f"t{i}": s for i, s in enumerate(reversed(list(shapes.values())))}
    assert SN.base_identity(renamed, renamed_shapes) == base_id
    assert SN.base_identity({t: float(v) for t, v in norms.items()}, shapes) == base_id
    _, other = SN.base_key(_weights(ToyAnima(seed=3).eval()))
    assert other != base_id


def _inference_model():
    with torch.inference_mode():
        m = ToyAnima().eval()
    assert all(w.is_inference() for w in _weights(m).values())
    return m


def test_the_norm_cache_key_of_an_inference_weight_has_no_version(model):
    inference = _inference_model()
    for w in _weights(inference).values():
        key = SN.weight_identity(w)
        assert key == (w.data_ptr(), None, tuple(w.shape), str(w.dtype), str(w.device))
    for w in _weights(model).values():
        assert SN.weight_identity(w)[1] == w._version


def test_inference_weights_give_the_same_size(model, tmp_path):
    path = write(tmp_path, "a", to_keys(lora_weights(model, rank=4), "kohya", alpha=2.0))
    inference = _inference_model()
    plain = MI.lora_file_size(path, model)
    MI._STORED_SIZES.clear()
    MI._SIZES.clear()
    SN._BASE_NORMS.clear()
    got = MI.lora_file_size(path, inference)
    assert got is not None and got == pytest.approx(plain, rel=1e-6)
    norms, base_id = _base(inference)
    assert base_id == _base(model)[1]
    assert MI.lora_file_size(path, inference, base_norms=norms, base_id=base_id) == (
        pytest.approx(plain, rel=1e-6)
    )


def test_a_second_call_on_inference_weights_reads_the_norm_cache(monkeypatch):
    inference = _inference_model()
    calls = []
    real = SN.weight_sq
    monkeypatch.setattr(SN, "weight_sq", lambda w: calls.append(1) or real(w))
    weights = _weights(inference)
    first = SN.base_norms(weights)
    n = len(calls)
    assert n == len(weights)
    assert SN.base_norms(weights) == first
    assert len(calls) == n


def test_size_normed_chain_works_on_inference_weights(monkeypatch, model, tmp_path):
    P = N.patching
    inference = _inference_model()
    weights = _weights(inference)
    path = write(tmp_path, "s", to_keys(lora_weights(model, rank=4), "kohya", alpha=2.0))

    class Binding:
        family = "anima"

    class Full:
        diffusion_model = inference

    class Patcher:
        model = Full()

    monkeypatch.setattr(P, "detect_family", lambda m: Binding())
    monkeypatch.setattr(P, "diffusers_shapes", lambda m, b: {})
    monkeypatch.setattr(P, "shape_model", lambda shapes: inference)
    monkeypatch.setattr(P, "host_weights", lambda m, shapes: weights)
    chain = [P.ModEntry(kind="lora", path=path, strength=0.7)]
    out, lines = N.size_normed_chain(Patcher(), chain, True)
    c = SN.size_scale(MI.lora_file_size(path, model))
    assert out[0].strength == pytest.approx(0.7 * min(c, 1.0), rel=1e-5)
    assert len(lines) == 1 and "S_f" in lines[0]
    calls = []
    real = SN.weight_sq
    monkeypatch.setattr(SN, "weight_sq", lambda w: calls.append(1) or real(w))
    again, _ = N.size_normed_chain(Patcher(), chain, True)
    assert again[0].strength == pytest.approx(out[0].strength)
    assert calls == []
