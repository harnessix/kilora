# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from kilora import compute, ir
from kilora.ir.driver import run_layer_ir


def sets():
    imported = compute.LayerSet(
        form="input",
        import_score="derived",
        key="imported",
        carry="drop",
        router=torch.zeros(1, 2),
        keys_in=torch.zeros(2, 1, 1),
        down=torch.tensor([[0.2, 0.0, 0.0], [0.0, 0.3, 0.0]]),
        up=torch.tensor([[0.2, 0.1], [-0.1, 0.2]]),
        gate_mode="none",
    )
    state = compute.LayerSet(
        key="state",
        carry="drop",
        router=torch.zeros(1, 2),
        keys_in=torch.zeros(2, 1, 1),
        keys_out=torch.ones(2, 1, 1),
        down=torch.tensor([[0.0, 0.2], [0.2, 0.2]]),
        up=torch.tensor([[0.1, 0.3], [0.2, -0.1]]),
        gate_mode="none",
    )
    return [imported, state]


def inputs():
    return (
        torch.tensor([[[1.0, 0.4, 0.2], [0.2, 1.0, 0.8]]]),
        torch.tensor([[[0.8, 0.2], [0.2, 0.9]]]),
    )


@pytest.mark.parametrize("depth", [1, 3])
def test_four_experts_share_one_selection_on_their_read_states(depth, monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", "expect")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    layers = sets()
    x, y0 = inputs()
    scored, shares = [], []
    original = compute.derived_logits

    def score(s, value):
        logits = original(s, value)
        scored.append((s.form, value.clone(), logits.clone()))
        return logits

    monkeypatch.setattr(compute, "derived_logits", score)
    with (
        torch.no_grad(),
        compute.eager(),
        compute.recording_shares(lambda _, gates: shares.append([g.clone() for g in gates])),
    ):
        actual = compute.run_layer(x, y0, layers, depth, target="layer")
    assert len(scored) == 2 * depth
    carried = y0
    coefficients = [None, None]
    alive = [None, None]
    refs = [None, None]
    base = [F.linear(x, layers[0].down), F.linear(y0, layers[1].down)]
    for step in range(depth):
        pair = scored[step * 2 : step * 2 + 2]
        assert pair[0][0] == "input" and pair[1][0] == "state"
        assert torch.equal(pair[0][1], x)
        torch.testing.assert_close(pair[1][1], carried)
        torch.testing.assert_close(pair[1][2], original(layers[1], carried))
        expected = compute.sparsemax(torch.cat([p[2] for p in pair], -1))
        torch.testing.assert_close(torch.cat(shares[step], -1), expected)
        torch.testing.assert_close(expected.sum(-1), torch.ones_like(expected[..., 0]))
        output = y0
        next_carried = y0
        for i, layer in enumerate(layers):
            read = F.linear(x if i == 0 else carried, layer.down)
            part = read * shares[step][i]
            if alive[i] is not None:
                part = part * alive[i]
            metric = compute._step_metric(layer, y0.device)
            if coefficients[i] is not None and layer.form == "input":
                settings = compute._import_depth_settings(layer)
                z = compute.depth_split_step(coefficients[i], part, settings, metric, refs[i])
            else:
                z = part if coefficients[i] is None else coefficients[i] + part
            size = compute._metric_size(z, metric)
            if step == 0:
                refs[i] = size
            else:
                ratio = compute.IMPORT_DEPTH_BUDGET * refs[i] / size.clamp_min(1e-30)
                z = z * ratio.clamp(max=1)[..., None]
            coefficients[i] = z
            output = output + z @ layer.up
            mask = compute.read_dropout(z, base[i], layer.per_expert, "layer", step + 1)
            alive[i] = mask if alive[i] is None else alive[i] * mask
            next_carried = next_carried + (mask * z) @ layer.up
        carried = next_carried
    torch.testing.assert_close(actual, output)


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("zero", [0, 1])
def test_zero_set_matches_remaining_set(depth, zero, monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", "expect")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    layers = sets()
    layers[zero] = replace(
        layers[zero],
        down=torch.zeros_like(layers[zero].down),
        up=torch.zeros_like(layers[zero].up),
    )
    x, y0 = inputs()
    with torch.no_grad(), compute.eager():
        actual = compute.run_layer(x, y0, layers, depth)
        expected = compute.run_layer(x, y0, [layers[1 - zero]], depth)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("scope", ["token", "tensor"])
@pytest.mark.parametrize("selection", ["sparsemax", "top1"])
@pytest.mark.parametrize("masks", ["sample", "expect"])
def test_joint_eager_fused_and_ir_agree(depth, scope, selection, masks, monkeypatch):
    pytest.importorskip("triton")
    monkeypatch.setenv("KILORA_MASKS", masks)
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    layers = [replace(s, routing_scope=scope, routing_select=selection) for s in sets()]
    x, y0 = inputs()
    from kilora import fused

    with torch.no_grad(), compute.dropout_key(7, 3):
        with compute.eager():
            expected = compute.run_layer(x, y0, layers, depth, target="layer")
        actual = fused.run_fused(y0, layers, depth, target="layer", x=x)
        with ir.interpret_all():
            interpreted = run_layer_ir(x, y0, layers, depth, target="layer")
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(interpreted, expected)


def test_state_default_uses_down_and_ignores_learned_router(monkeypatch):
    monkeypatch.delenv("KILORA_IMPORT_SCORE", raising=False)
    x, y0 = inputs()
    state = sets()[1]
    changed = replace(
        state,
        router=torch.full_like(state.router, 1000),
        keys_in=torch.full_like(state.keys_in, 23),
        keys_out=-state.keys_out,
        routing_down=torch.zeros_like(state.down),
        routing_up=state.up,
    )
    assert compute.import_score(state) == "derived"
    with torch.no_grad(), compute.eager():
        first = compute.run_layer(x, y0, [state], 1)
        second = compute.run_layer(x, y0, [changed], 1)
    assert torch.equal(first, second)
    weights = compute.sparsemax(compute.derived_logits(state, y0))
    torch.testing.assert_close(first, y0 + (F.linear(y0, state.down) * weights) @ state.up)


@pytest.mark.parametrize("mode", ["keys", "read_share"])
def test_legacy_import_scores_join_derived_state_selection(mode):
    layers = [replace(s, import_score=mode) for s in sets()]
    x, y0 = inputs()
    with torch.no_grad():
        gates = compute._mixed_route(
            y0,
            layers,
            [compute._set_keys(s) for s in layers],
            None,
            False,
            0.0,
            [None, None],
            True,
            x=x,
        )
    total = torch.cat(gates, -1).sum(-1)
    torch.testing.assert_close(total, torch.ones_like(total))
    assert compute.import_score(layers[1]) == "derived"


@pytest.fixture(scope="module")
def prior_compute():
    from pathlib import Path
    from types import SimpleNamespace

    namespace = dict(vars(compute))
    source = Path(__file__).with_name("data").joinpath("import_depth_reference.txt").read_text()
    exec(compile(source, "import_depth_reference.py", "exec"), namespace)
    loop = namespace["_run_input"]

    def run(x, y0, layers, depth, target=""):
        if depth <= 0:
            return y0
        return loop(x, y0, layers, list(range(len(layers))), depth, [None] * len(layers), target)

    return SimpleNamespace(run_layer=run, dropout_key=compute.dropout_key, eager=compute.eager)


@pytest.mark.parametrize("depth", [0, 0.5, *range(1, 17), 2.5])
@pytest.mark.parametrize("backend", ["eager", "fused"])
@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("target", ["layer", "layers.0.attention.to_out.0"])
def test_import_only_is_bit_exact_to_starting_branch(
    depth, backend, count, target, prior_compute, monkeypatch
):
    if backend == "fused":
        pytest.importorskip("triton")
    monkeypatch.setenv("KILORA_MASKS", "expect")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    monkeypatch.setenv("KILORA_FUSED", "1")
    x, y0 = inputs()
    first = sets()[0]
    layers = [first] if count == 1 else [first, replace(first, key="other", down=first.down * 0.7)]
    with torch.no_grad(), compute.dropout_key(7, 3), prior_compute.dropout_key(7, 3):
        if backend == "eager":
            with compute.eager(), prior_compute.eager():
                expected = prior_compute.run_layer(x, y0, layers, depth, target=target)
                actual = compute.run_layer(x, y0, layers, depth, target=target)
        else:
            expected = prior_compute.run_layer(x, y0, layers, depth, target=target)
            actual = compute.run_layer(x, y0, layers, depth, target=target)
    assert torch.equal(actual, expected)


def test_state_without_experts_does_not_join_imported_route(monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", "expect")
    imported, state = sets()
    empty = replace(state, down=None, up=None)
    x, y0 = inputs()
    with torch.no_grad(), compute.eager():
        expected = compute.run_layer(x, y0, [imported], 3)
        actual = compute.run_layer(x, y0, [imported, empty], 3)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("scope", ["token", "tensor"])
def test_file_balance_adds_share_units_and_set_prior(scope, monkeypatch):
    x, y0 = inputs()
    imported, state = sets()
    imported = replace(
        imported,
        down=imported.down[:1],
        up=imported.up[:1],
        log_prior=0.7,
        routing_scope=scope,
        tau=0.5,
    )
    state = replace(
        state,
        down=state.down.repeat(2, 1),
        up=state.up.repeat(2, 1),
        log_prior=0.2,
        routing_scope=scope,
        tau=3.0,
    )
    layers = [imported, state]
    for on in ("0", "1"):
        monkeypatch.setenv("KILORA_FILE_BALANCE", on)
        with torch.no_grad():
            logits = [compute.derived_logits(imported, x), compute.derived_logits(state, y0)]
            if on == "1":
                logits = [lg + 1 / (2 * lg.shape[-1]) for lg in logits]
            expected = compute._gates_from_logits(logits, layers, True, 0.0)
            actual = compute._mixed_route(
                y0, layers, None, None, True, 0.0, [None, None], True, x=x
            )
        for a, b in zip(actual, expected, strict=True):
            assert torch.equal(a, b)


def test_equal_scores_give_equal_file_shares(monkeypatch):
    x = torch.ones(1, 2, 2)
    first = compute.LayerSet(
        router=torch.eye(2),
        keys_in=torch.zeros(1, 1, 2),
        form="input",
        import_score="derived",
        down=torch.ones(1, 2),
        up=torch.ones(1, 2),
        gate_mode="none",
    )
    mod = replace(
        first,
        form="state",
        down=first.down.repeat(4, 1),
        up=first.up.repeat(4, 1),
        keys_in=torch.zeros(4, 1, 2),
    )
    with torch.no_grad():
        monkeypatch.setenv("KILORA_FILE_BALANCE", "0")
        off = compute._mixed_route(x, [first, mod], None, None, True, 0.0, [None, None], True, x=x)
        monkeypatch.setenv("KILORA_FILE_BALANCE", "1")
        on = compute._mixed_route(x, [first, mod], None, None, True, 0.0, [None, None], True, x=x)
    torch.testing.assert_close(off[0], torch.full_like(off[0], 0.2))
    torch.testing.assert_close(off[1].sum(-1), torch.full_like(off[1][..., 0], 0.8))
    assert torch.equal(on[0], torch.full_like(on[0], 0.5))
    assert torch.equal(on[1], torch.full_like(on[1], 0.125))
    assert torch.equal(on[1].sum(-1), torch.full_like(on[1][..., 0], 0.5))


@pytest.mark.parametrize("depth", [1, 3, 2.5])
@pytest.mark.parametrize("state_only", [False, True])
def test_balance_switch_preserves_single_set_and_single_expert_imports(
    depth, state_only, monkeypatch
):
    x, y0 = inputs()
    imported, state = sets()
    imported = replace(imported, down=imported.down[:1], up=imported.up[:1])
    layers = [state] if state_only else [imported, replace(imported, key="second")]
    outputs = []
    monkeypatch.setenv("KILORA_MASKS", "expect")
    with torch.no_grad(), compute.dropout_key(11, 2):
        for on in ("0", "1"):
            monkeypatch.setenv("KILORA_FILE_BALANCE", on)
            with compute.eager():
                outputs.append(compute.run_layer(x, y0, layers, depth))
    assert torch.equal(*outputs)


@pytest.mark.parametrize("mode", ["keys", "read_share", "derived"])
def test_saved_settings_and_environment_cannot_restore_state_learned_keys(mode, monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_SCORE", mode)
    x, y0 = inputs()
    state = replace(sets()[1], import_score=mode)
    changed = replace(
        state,
        router=torch.zeros_like(state.router),
        keys_in=torch.randn_like(state.keys_in),
        keys_out=None,
    )
    assert compute.import_score(state) == "derived"
    with torch.no_grad(), compute.eager(), compute.dropout_key(11, 2):
        expected = compute.run_layer(x, y0, [state], 3)
        actual = compute.run_layer(x, y0, [changed], 3)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("backend", ["loop", "lowrank", "path", "fused"])
def test_direct_backends_follow_the_derived_state_contract(backend, monkeypatch):
    if backend == "fused":
        pytest.importorskip("triton")
    from kilora import fused

    monkeypatch.setenv("KILORA_MASKS", "expect")
    x, y0 = inputs()
    state = sets()[1]
    if backend == "path":
        state = replace(state, iteration="path")
    with torch.no_grad(), compute.dropout_key(13, 2), compute.eager():
        expected = compute.run_layer(x, y0, [state], 3, target="layer")
        if backend == "loop":
            actual = compute._run_loop(y0, [state], 3, False, 0, None, None, "layer", x=x)
        elif backend == "lowrank":
            actual = compute._run_lowrank(y0, [state], 3, False, target="layer", x=x)
        elif backend == "path":
            actual = compute._run_path(y0, [state], 3, False, 0, None, "layer", x=x)
        else:
            actual = fused.run_fused(y0, [state], 3, target="layer", x=x)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("summary", ["keys", "read_map"])
def test_state_candidates_ignore_legacy_router_and_row_counts(summary, monkeypatch):
    monkeypatch.setenv("KILORA_MASKS", "expect")
    x, y0 = inputs()
    state = replace(sets()[1], candidates=1, candidate_summary=summary)
    changed = replace(
        state,
        router=torch.full_like(state.router, 1000),
        keys_in=torch.zeros_like(state.keys_in),
        rows_in=torch.tensor([0, 100]),
        rows_out=torch.tensor([100, 0]),
    )
    with torch.no_grad(), compute.eager(), compute.dropout_key(13, 2):
        expected = compute.run_layer(x, y0, [state], 3)
        actual = compute.run_layer(x, y0, [changed], 3)
    assert torch.equal(actual, expected)


def test_state_only_pools_preserve_shared_and_per_set_selection(monkeypatch):
    monkeypatch.setenv("KILORA_FILE_BALANCE", "1")
    x, y0 = inputs()
    first = sets()[1]
    second = replace(first, key="second", down=first.down.flip(0))
    with torch.no_grad():
        separate = compute._mixed_route(
            y0, [first, second], None, None, False, 0.0, [None, None], True, x=x
        )
        joint = compute._mixed_route(
            y0, [first, second], None, None, True, 0.0, [None, None], True, x=x
        )
    for gate in separate:
        torch.testing.assert_close(gate.sum(-1), torch.ones_like(gate[..., 0]))
    total = torch.cat(joint, -1).sum(-1)
    torch.testing.assert_close(total, torch.ones_like(total))


@pytest.mark.parametrize("depth", [1, 3])
def test_joint_path_starts_exclude_region_masked_imports(depth, monkeypatch):
    from kilora.path_walks import _derived_walks

    monkeypatch.setenv("KILORA_MASKS", "expect")
    monkeypatch.setenv("KILORA_DERIVED_MODE", "keys")
    imported, state = [replace(s, iteration="path", path_width=4) for s in sets()]
    imported = replace(imported, expert_mask=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
    x, y0 = inputs()
    with torch.no_grad(), compute.eager():
        records, _ = _derived_walks(
            y0, [imported, state], depth, 0.0, None, "region", [None, None], x, 4
        )
    assert len(records) == 4
    for record in records:
        gates = record["gates"][0]
        assert gates[0, 0, 1] == 0
        assert gates[0, 1, 0] == 0
