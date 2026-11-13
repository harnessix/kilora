# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import math
import time

import pytest
import torch

from kilora import KiLoRAConfig, attach
from kilora.compute import orthonormal_keys
from kilora.fast_keys import PromptCache, fast_state_keys, write_keys
from kilora.subspace_keys import subspace_distance
from kilora.testing import TinyDenoiser

CHANNELS = 4
TOKENS = 12


def _encoder(calls):
    def encode(prompt):
        calls.append(prompt)
        seed = sum(map(ord, prompt))
        n = 3 + len(prompt) % 3
        return torch.randn(n, CHANNELS, generator=torch.Generator().manual_seed(seed))

    return encode


def _setup(m_in=2, m_out=3):
    model = TinyDenoiser(channels=CHANNELS, width=16, num_layers=2, seed=0).eval()
    adapter = attach(
        model,
        KiLoRAConfig(
            target_modules=["layers.*.feed_forward.w2"],
            num_experts=4,
            routes=1,
            router_dim=8,
            m_in=m_in,
            m_out=m_out,
            depth=1,
            train_depth=None,
        ),
    )
    with torch.no_grad():
        for site in adapter.sites:
            site.tensors["s0_up"].normal_(0, 0.3, generator=torch.Generator().manual_seed(1))
    calls = []

    def run(latents, sigmas, text):
        calls.append(latents.shape[0])
        noisy = latents * sigmas.view(-1, 1, 1)
        return model(torch.cat([noisy, text], dim=1))

    return model, adapter, run, calls


PROMPTS = ["a red fox", "a castle at night", "portrait, oil paint"]


def test_one_calibration_call_and_weight_derived_state_keys():
    _, adapter, run, calls = _setup()
    before = adapter.to_sets()[0]
    keys = fast_state_keys(
        adapter,
        run,
        PromptCache(_encoder([])),
        PROMPTS,
        latent_shape=(TOKENS, CHANNELS),
        m_in=2,
        m_out=3,
    )
    assert calls == [len(PROMPTS) * 2]
    assert adapter.current_depth == 1
    assert set(keys) == set(adapter.targets)
    for target, k in keys.items():
        expert = before.experts[target]
        for part in ("keys_in", "keys_out", "rows_in", "rows_out"):
            torch.testing.assert_close(k[part], getattr(expert, part), rtol=0, atol=0)
        assert k["keys_in"].shape[-1] == expert.down.shape[-1]
        assert k["keys_out"].shape[-1] == expert.up.shape[-1]


def test_encoder_runs_once_per_distinct_prompt_across_calls():
    _, adapter, run, calls = _setup()
    encoded = []
    cache = PromptCache(_encoder(encoded))
    fast_state_keys(adapter, run, cache, PROMPTS[:2], latent_shape=(TOKENS, CHANNELS))
    fast_state_keys(adapter, run, cache, PROMPTS[1:] + PROMPTS[:1], latent_shape=(TOKENS, CHANNELS))
    assert sorted(encoded) == sorted(PROMPTS)
    assert len(cache) == 3
    assert calls == [4, 6]


def test_same_seed_gives_the_same_keys():
    _, adapter, run, _ = _setup()
    cache = PromptCache(_encoder([]))
    a = fast_state_keys(adapter, run, cache, PROMPTS, latent_shape=(TOKENS, CHANNELS))
    b = fast_state_keys(adapter, run, cache, PROMPTS, latent_shape=(TOKENS, CHANNELS))
    for t in a:
        assert torch.equal(a[t]["keys_in"], b[t]["keys_in"])


def test_write_keys_sets_the_source_and_keeps_the_lora_weights():
    _, adapter, run, _ = _setup()
    cache = PromptCache(_encoder([]))
    keys = fast_state_keys(
        adapter, run, cache, PROMPTS, latent_shape=(TOKENS, CHANNELS), m_in=2, m_out=3
    )
    old = adapter.to_sets()[0]
    new = write_keys(old, keys)
    assert new.key_source == "states" and old.key_source == "random"
    assert new is not old
    for t, ex in new.experts.items():
        assert torch.equal(ex.down, old.experts[t].down)
        assert torch.equal(ex.up, old.experts[t].up)
        torch.testing.assert_close(ex.keys_in, keys[t]["keys_in"])
        torch.testing.assert_close(ex.keys_out, keys[t]["keys_out"])
        assert torch.equal(ex.rows_in, keys[t]["rows_in"])
    assert torch.equal(new.experts[t].keys_in, old.experts[t].keys_in)
    with pytest.raises(ValueError, match="does not have"):
        write_keys(old, {"nope": keys[t]})


def test_the_pass_takes_under_three_seconds_on_cpu():
    _, adapter, run, _ = _setup()
    cache = PromptCache(_encoder([]))
    prompts = [f"prompt number {i}" for i in range(8)]
    for p in prompts:
        cache.get(p)
    start = time.perf_counter()
    fast_state_keys(adapter, run, cache, prompts, latent_shape=(64, CHANNELS))
    assert time.perf_counter() - start < 3.0


def test_cli_writes_a_loadable_kmod(tmp_path, monkeypatch, capsys):
    pytest.importorskip("diffusers")
    pytest.importorskip("peft")
    from PIL import Image
    from test_train import pipeline

    import kilora.train as T
    from kilora import load_mod

    folder = tmp_path / "images"
    folder.mkdir()
    gen = torch.Generator().manual_seed(0)
    for i in range(3):
        pixels = (torch.rand(40, 48, 3, generator=gen) * 255).to(torch.uint8).numpy()
        Image.fromarray(pixels).save(folder / f"img{i}.png")
        (folder / f"img{i}.txt").write_text(f"caption {i}")
    common = [
        "--model",
        "tiny",
        "--resolution",
        "32",
        "--precision",
        "fp32",
        "--device",
        "cpu",
        "--num-experts",
        "4",
        "--router-dim",
        "8",
        "--m-in",
        "2",
        "--m-out",
        "3",
        "--license",
        "CC0-1.0",
        "--static",
    ]
    first = tmp_path / "first.kmod"
    T.run(
        T.build_parser().parse_args(
            [*common, "--data", str(folder), "--output", str(first), "--steps", "2"]
        ),
        pipe=pipeline(),
    )
    monkeypatch.setattr(T, "FAST_RESOLUTION", 32)
    out = tmp_path / "keyed.kmod"
    args = [
        *common,
        "--build-keys",
        str(out),
        "--mod",
        str(first),
        "--source",
        "states",
        "--fast",
        "--prompts",
        "a red fox",
        "a castle",
    ]
    T.run(T.build_parser().parse_args(args), pipe=pipeline())
    assert "Forward pass:" in capsys.readouterr().err
    old, new = load_mod(first), load_mod(out)
    assert new.sets[0].key_source == "states"
    assert new.manifest["license"] == "CC0-1.0"
    for t, ex in new.sets[0].experts.items():
        assert torch.equal(ex.down, old.sets[0].experts[t].down)
        torch.testing.assert_close(ex.keys_in, old.sets[0].experts[t].keys_in)
        torch.testing.assert_close(ex.keys_out, old.sets[0].experts[t].keys_out)
    args[args.index(str(out))] = str(first)
    T.run(T.build_parser().parse_args(args), pipe=pipeline())
    assert load_mod(first).sets[0].key_source == "states"


def test_low_rank_fast_path_and_plain_loop_give_the_same_keys(monkeypatch):
    from kilora import compute as C

    out = []
    for fast in (True, False):
        monkeypatch.setattr(C, "LOWRANK_ENABLED", fast)
        _, adapter, run, _ = _setup()
        cache = PromptCache(_encoder([]))
        out.append(fast_state_keys(adapter, run, cache, PROMPTS, latent_shape=(TOKENS, CHANNELS)))
    for t in out[0]:
        for part in ("keys_in", "keys_out"):
            for e in range(4):
                assert subspace_distance(out[0][t][part][e], out[1][t][part][e]) < 1e-3


WIDTH = 16
TARGETS = ["layers.0.feed_forward.w1", "layers.1.feed_forward.w1"]


def _input_set(
    name="imported",
    experts=3,
    rank=2,
    router_dim=8,
    seed=5,
    targets=TARGETS,
    output_width=2 * WIDTH,
):
    from kilora.sets import AdapterSet, KeyedExperts, make_router

    gen = torch.Generator().manual_seed(seed)
    parts = {
        t: KeyedExperts(
            keys_in=orthonormal_keys(torch.randn(experts, 2, router_dim, generator=gen)),
            keys_out=orthonormal_keys(torch.randn(experts, 2, router_dim, generator=gen)),
            down=torch.randn(experts, 1, rank, WIDTH, generator=gen) * 0.3,
            up=torch.randn(experts, 1, rank, output_width, generator=gen) * 0.3,
        )
        for t in targets
    }
    return AdapterSet(
        name=name,
        part_type=("lora",),
        experts=parts,
        routers={output_width: make_router(router_dim, output_width, seed=11)},
        form="input",
        imported_from={"file": "by-hand"},
    )


def _model_and_run():
    model = TinyDenoiser(channels=CHANNELS, width=WIDTH, num_layers=2, seed=0).eval()

    def run(latents, sigmas, text):
        return model(torch.cat([latents * sigmas.view(-1, 1, 1), text], dim=1))

    return model, run


def _check_moved(keys, start, m):
    for target, k in keys.items():
        experts = k["keys_in"].shape[0]
        torch.testing.assert_close(
            k["keys_in"] @ k["keys_in"].transpose(-1, -2),
            torch.eye(m).expand(experts, m, m),
            atol=1e-5,
            rtol=0,
        )
        before = orthonormal_keys(start[target])
        moved = max(subspace_distance(k["keys_in"][e], before[e]) for e in range(experts))
        assert moved > 0.05, f"{target}: keys_in did not move"


def test_input_form_set_gets_input_and_output_keys(caplog):
    from kilora import compose

    model, run = _model_and_run()
    imported = _input_set()
    adapter = compose(model, [imported], depth=2)
    start = {t: ex.keys_in.clone() for t, ex in imported.experts.items()}
    keys = fast_state_keys(
        adapter,
        run,
        PromptCache(_encoder([])),
        PROMPTS,
        latent_shape=(TOKENS, CHANNELS),
        m_in=2,
        m_out=3,
    )
    assert set(keys) == set(TARGETS)
    _check_moved(keys, start, 2)
    for k in keys.values():
        assert k["keys_out"].shape[1] == 3 and bool((k["rows_out"] == 3).all())
    adapter.remove()
    model, run = _model_and_run()
    adapter = compose(model, [imported], depth=1)
    with caplog.at_level("INFO", logger="kilora.subspace_keys"):
        keys = fast_state_keys(
            adapter,
            run,
            PromptCache(_encoder([])),
            PROMPTS,
            latent_shape=(TOKENS, CHANNELS),
            m_in=2,
            m_out=3,
            depth=1,
        )
    for k in keys.values():
        assert k["keys_out"].shape[1] == 3
        torch.testing.assert_close(k["keys_out"][:, :2], k["keys_in"], atol=1e-6, rtol=0)
    assert "no output-key data" in caplog.text


def test_state_and_input_sets_composed_together_both_get_keys():
    from kilora import compose

    model, run = _model_and_run()
    state = attach(
        model,
        KiLoRAConfig(
            target_modules=["layers.*.feed_forward.w1"],
            num_experts=4,
            routes=1,
            router_dim=8,
            m_in=2,
            m_out=3,
            depth=1,
            train_depth=None,
        ),
    ).to_sets()[0]
    gen = torch.Generator().manual_seed(1)
    for ex in state.experts.values():
        ex.up = torch.randn(ex.up.shape, generator=gen) * 0.3
    model, run = _model_and_run()
    imported = _input_set()
    for order in ((state, imported), (imported, state)):
        adapter = compose(model, list(order), depth=2)
        cache = PromptCache(_encoder([]))
        for i, s in enumerate(order):
            start = {t: ex.keys_in.clone() for t, ex in s.experts.items()}
            keys = fast_state_keys(
                adapter,
                run,
                cache,
                PROMPTS,
                latent_shape=(TOKENS, CHANNELS),
                m_in=2,
                m_out=3,
                set_index=i,
            )
            assert set(keys) == set(TARGETS)
            if s.form == "input":
                _check_moved(keys, start, 2)
                assert all(k["keys_out"].shape[1] == 3 for k in keys.values())
            else:
                for target, k in keys.items():
                    torch.testing.assert_close(k["keys_in"], start[target])
                    assert k["keys_out"].shape[-1] == s.experts[target].up.shape[-1]
        adapter.remove()


def test_probes_do_not_change_the_layer_output():
    from kilora import compose
    from kilora.compute import key_probes, run_layer

    model, _ = _model_and_run()
    state = attach(
        model,
        KiLoRAConfig(
            target_modules=["layers.*.feed_forward.w1"],
            num_experts=4,
            routes=1,
            router_dim=8,
            depth=1,
            train_depth=None,
        ),
    ).to_sets()[0]
    gen = torch.Generator().manual_seed(2)
    for ex in state.experts.values():
        ex.up = torch.randn(ex.up.shape, generator=gen) * 0.3
    model, _ = _model_and_run()
    adapter = compose(model, [state, _input_set()], depth=2)
    index = 0
    sets = adapter._layer_sets(index, adapter.sites[index], torch.device("cpu"))
    assert {s.form for s in sets} == {"state", "input"}
    x = torch.randn(2, TOKENS, WIDTH, generator=gen)
    y0 = torch.randn(2, TOKENS, 2 * WIDTH, generator=gen)
    with torch.no_grad():
        plain = run_layer(x, y0, sets, 2, target=adapter.targets[index])
        with key_probes() as record:
            probed = run_layer(x, y0, sets, 2, target=adapter.targets[index])
    torch.testing.assert_close(probed, plain, atol=1e-5, rtol=1e-5)
    entries = record[adapter.targets[index]]
    steps = [e for e in entries if e["k"] != "output"]
    assert [e["k"] for e in steps] == [1, 2]
    assert all(e["queries"][0] is not None and e["queries"][1] is not None for e in steps)
    assert steps[0]["prev"] is None and all(g is not None for g in steps[1]["prev"])


@pytest.mark.parametrize("imported_first", [False, True])
def test_mixed_attention_probes_keep_set_positions(imported_first):
    from kilora import compose
    from kilora.compute import key_probes, run_layer

    targets = ["layers.0.attention.to_out.0", "layers.1.attention.to_out.0"]
    model, run = _model_and_run()
    state_adapter = attach(
        model,
        KiLoRAConfig(
            target_modules=["layers.*.attention.to_out.0"],
            num_experts=4,
            routes=1,
            router_dim=8,
            depth=2,
            train_depth=None,
        ),
    )
    state = state_adapter.to_sets()[0]
    state_adapter.remove()
    imported = _input_set(targets=targets, output_width=WIDTH)
    order = [imported, state] if imported_first else [state, imported]
    adapter = compose(model, order, depth=2)
    for index in range(2):
        keys = fast_state_keys(
            adapter,
            run,
            PromptCache(_encoder([])),
            PROMPTS,
            latent_shape=(TOKENS, CHANNELS),
            set_index=index,
        )
        assert set(keys) == set(targets)
    sets = adapter._layer_sets(0, adapter.sites[0], torch.device("cpu"))
    x = torch.randn(2, TOKENS, WIDTH, requires_grad=True)
    y = torch.randn_like(x, requires_grad=True)
    with key_probes() as record:
        output = run_layer(x, y, sets, 2, target=targets[0])
        output.sum().backward()
    entries = record[targets[0]]
    plain_pos, state_pos = (0, 1) if imported_first else (1, 0)
    steps = [e for e in entries if e["k"] != "output"]
    assert [e["k"] for e in steps] == [1, 2]
    assert all(
        e["queries"][plain_pos] is not None and e["queries"][state_pos] is not None for e in steps
    )
    outputs = [e for e in entries if e["k"] == "output"]
    assert any(
        e["grads"][state_pos] is not None and e["grads"][plain_pos] is not None for e in outputs
    )
    adapter.remove()


@pytest.mark.parametrize("scope", ["token", "tensor"])
def test_calibrated_temperature_gives_the_target_top_weight(scope):
    from kilora.compute import norm_energy, orthonormal_keys, sparsemax
    from kilora.fast_keys import TARGET_TOP, calibrate_log_tau

    gen = torch.Generator().manual_seed(0)
    queries = {"t": torch.randn(4, 30, 8, generator=gen)}
    keys = {"t": (torch.randn(6, 1, 8, generator=gen), None)}
    log_tau = calibrate_log_tau(queries, keys, scope=scope)
    scores = norm_energy(queries["t"], orthonormal_keys(keys["t"][0]), None)
    if scope == "tensor":
        scores = scores.mean(-2)
    top = float(sparsemax(math.exp(log_tau) * scores).max(-1).values.mean())
    flat = float(sparsemax(scores).max(-1).values.mean())
    assert abs(top - TARGET_TOP) < 0.01
    if scope == "tensor":
        assert flat < TARGET_TOP
    else:
        assert log_tau < calibrate_log_tau(queries, keys, scope="tensor")
