# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from kilora_editor_chunk.build import AnimaCollector, DepthMix, build, collect
from test_chunk_build import _tiny_anima

from kilora import compute
from kilora.editor_packs import read_pack
from kilora.testing import TINY_FAMILY, make_live_mod

TARGETS = ("transformer_blocks.*.attn1.to_out.0", "transformer_blocks.*.attn2.to_q")


class _Pipe:
    def __init__(self, model):
        import diffusers

        self.scheduler = diffusers.FlowMatchEulerDiscreteScheduler(shift=3.0)
        self.transformer = model


def _encode(text):
    return torch.randn(5, 12, generator=torch.Generator().manual_seed(len(text)))


def _collector(tmp_path, path="mod", mix=None, mod=None, targets=TARGETS):
    model = _tiny_anima()
    if mod is None:
        mod = tmp_path / "live.kmod"
        make_live_mod(mod, model=model, targets=targets)
    return AnimaCollector(
        _Pipe(model), str(mod), size=64, steps=2, guidance=4.0, encode=_encode, path=path, mix=mix
    )


def _one_expert_mod(path, model):
    from kilora.adapter import Settings, attach
    from kilora.config import KiLoRAConfig
    from kilora.live import LiveSpec
    from kilora.live_sets import LivePreset, make_live_set
    from kilora.modfile import save_mod

    config = KiLoRAConfig(
        target_modules=list(TARGETS), num_experts=1, routes=2, rank=2, router_dim=8
    )
    adapter = attach(model, config, name="main")
    main = adapter.to_sets()[0]
    adapter.remove()
    preset = LivePreset("one", 1, 1, ("attn1.to_out.0", "attn2.to_q"))
    live, layout = make_live_set(main, preset, TINY_FAMILY)
    for ex in live.experts.values():
        ex.up.zero_()
    spec = LiveSpec(
        layout=layout, updater={"id": "any"}, live_set=live.name, speed=0.5, speed_max=1.0
    )
    save_mod(
        path,
        [live],
        Settings(depth=2, softmax_scope="per_set", routing_select="top1"),
        id="test-one",
        base_model={"family": TINY_FAMILY},
        license="CC0-1.0",
        live=spec.to_manifest(),
    )


def _close(a, b):
    for t in a:
        torch.testing.assert_close(a[t], b[t], rtol=1e-4, atol=1e-7)


def test_depth_one_matches_the_hook_path(tmp_path):
    model = _tiny_anima()
    mod = tmp_path / "one.kmod"
    _one_expert_mod(mod, model)
    pipe = _Pipe(model)
    new = AnimaCollector(pipe, str(mod), size=64, steps=2, encode=_encode, path="mod")
    lat, _ = new.generate("a cat", 3)
    got = new.grad_mod(lat, "a cat", 5, 1)
    new.path = "hook"
    want = new.grad(lat, "a cat", 5)
    assert float(sum(g.abs().sum() for g in want.values())) > 0
    _close(got, want)


def test_one_step_replay_at_step_one_is_the_full_chain_at_depth_one(tmp_path):
    t = _collector(tmp_path)
    lat, _ = t.generate("a cat", 3)
    _close(t.grad_mod(lat, None, 7, 1, 1), t.grad_mod(lat, None, 7, 1))


def test_one_step_replay_differs_from_the_full_chain_after_step_one(tmp_path, monkeypatch):
    monkeypatch.setattr(compute, "read_dropout", lambda z, *a, **k: torch.ones_like(z))
    t = _collector(tmp_path, targets=("transformer_blocks.*.ff.net.2",))
    lat, _ = t.generate("a cat", 3)
    full = t.grad_mod(lat, None, 7, 2)
    flat = t.grad_mod(lat, None, 7, 2, 2)
    assert all(float(flat[h].abs().sum()) > 0 for h in flat)
    assert any(not torch.allclose(full[h], flat[h]) for h in full)


def test_dropout_masks_match_between_the_full_loop_and_the_replay(tmp_path, monkeypatch):
    seen: dict[str, list] = {"full": [], "flat": []}
    real = compute.read_dropout
    run = ["full"]

    def spy(z, base, per_expert, target, k, columns=None):
        m = real(z, base, per_expert, target, k, columns)
        seen[run[0]].append((target, k, m.detach().clone()))
        return m

    monkeypatch.setattr(compute, "read_dropout", spy)
    t = _collector(tmp_path, targets=("transformer_blocks.*.ff.net.2",))
    lat, _ = t.generate("a cat", 3)
    seen["full"].clear()
    t.grad_mod(lat, None, 11, 4)
    run[0] = "flat"
    t.grad_mod(lat, None, 11, 4, 3)
    first = seen["full"][0][0].split("#")[0]
    full = {(tg, k): m for tg, k, m in seen["full"] if k < 3 and tg.startswith(first)}
    flat = {(tg, k): m for tg, k, m in seen["flat"] if tg.startswith(first)}
    assert flat and set(flat) == set(full)
    for key, m in flat.items():
        assert torch.equal(m, full[key]), key


def test_depth_mix_draws():
    mix = DepthMix(full=0.5, depth_max=7)
    draws = [mix.draw(s) for s in range(400)]
    assert draws == [mix.draw(s) for s in range(400)]
    assert all(1 <= d <= 7 and (k is None or 1 <= k <= d) for d, k in draws)
    share = sum(k is None for _, k in draws) / len(draws)
    assert 0.4 < share < 0.6
    assert {d for d, _ in draws} == set(range(1, 8))


def test_packs_record_the_collection(tmp_path):
    t = _collector(tmp_path, mix=DepthMix(full=0.5, depth_max=3))
    assert t.collection["path"] == "mod" and t.collection["read_dropout"] is True
    prompts = {"_common": ["a cat", "a tree"], "ink": ["a cat, ink", "a dog, ink"]}
    files = collect(t, prompts, tmp_path / "events", draws=2, log=lambda _: None)
    r = build(files, t.names, t.shapes, tmp_path / "packs", "anima", common=4, family=2)
    m, _ = read_pack(r.packs["common"])
    record = m.raw["training"]["collection"]
    assert record == {
        "path": "mod",
        "mod": "test-live",
        "read_dropout": True,
        "depth_mix": {"full": 0.5, "one_step": 0.5, "depth": [1, 3]},
    }
    old = torch.load(files[0], weights_only=False)
    old.pop("collection")
    torch.save(old, files[0])
    with pytest.raises(ValueError, match="different ways"):
        build(files, t.names, t.shapes, tmp_path / "packs2", "anima", common=4, family=2)


def test_hook_path_is_unchanged(tmp_path):
    t = _collector(tmp_path, path="hook")
    assert t.collection is None and t.mix is None
    with pytest.raises(ValueError, match="'hook' or 'mod'"):
        _collector(tmp_path, path="chain")
