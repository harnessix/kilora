# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import hashlib

import pytest
import torch
from test_bypass import Patcher
from test_comfyui import FakeBaseModel
from test_families import _model
from test_import_modes import LAYERS, Toy, _lora
from test_working_copies import routed_chain

from kilora.comfyui import patching
from kilora.live import SESSIONS
from kilora.live_imports import (
    ImportEditor,
    import_block_recorder,
    import_frames,
    import_source_frames,
    imported_session_for,
)
from kilora.multi_import import import_loras
from kilora.recording import (
    BlockFrames,
    CrossStepRecorder,
    block_index,
    cross_step_error,
    noise_weight,
)
from kilora.sections import DenseChange, KronChange, LowRankChange

SIGMAS = (3.0, 2.0, 0.5)


@pytest.fixture
def imported(tmp_path, monkeypatch):
    SESSIONS.clear()
    monkeypatch.setenv("KILORA_IMPORT_SCORE", "derived")
    model = Toy().eval()
    paths = [tmp_path / f"{i}.safetensors" for i in range(2)]
    for i, path in enumerate(paths):
        _lora(model, path, rank=4, seed=i, alpha=4.0, layers=LAYERS)
    sources = [
        dict(
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            path=str(p),
            strength=1.0,
            axis="both",
        )
        for p in paths
    ]
    routed = import_loras([str(p) for p in paths], model).adapter_set
    shapes = {t: list(e.down.shape) for t, e in routed.experts.items()}
    session = imported_session_for(routed, sources, "toy", shapes, tmp_path / "saves")
    yield session, model
    SESSIONS.clear()


def _weights(session, model):
    out = {}
    for target in session.state:
        try:
            out[target] = model.get_submodule(target).weight
        except AttributeError:
            continue
    return out


def _ready(session, model):
    session.align_for_moves()
    import_source_frames(session, _weights(session, model))
    return import_frames(session)


def _stored_edit(pieces, device):
    if not pieces:
        return None
    ups, downs = [], []
    for up_e, down_e, rank, strength, core in pieces:
        b = up_e.detach().double().T
        mix = strength.double().reshape(()) * (
            core.double() if core is not None else torch.eye(int(rank), dtype=torch.float64)
        )
        ups.append(b @ mix.to(b.device))
        downs.append(down_e.detach().double())
    up = torch.cat(ups, 1)
    down = torch.cat(downs, 0)
    return up.detach().to(device, torch.float64), down.detach().to(device, torch.float64)


def _reference_j(frame, device):
    p_in = frame.p_in.to(device, torch.float64)
    p_out = frame.p_out.to(device, torch.float64)
    if frame.base is not None:
        jhat = frame.base.to(device, torch.float64)
    elif frame.weight is None:
        jhat = p_out @ p_in.t()
    else:
        weight = frame.weight.detach()
        jhat = p_out @ (weight @ frame.p_in.to(weight.device, weight.dtype).t()).to(device).double()
    stored = _stored_edit(frame.edit, device)
    if stored is None and frame.up is not None and frame.down is not None:
        stored = (
            frame.up.detach().to(device, torch.float64),
            frame.down.detach().to(device, torch.float64),
        )
    if stored is not None:
        up, down = stored
        jhat = jhat + (p_out @ up) @ (down @ p_in.t())
    return jhat


def _activations(frames, seed):
    gen = torch.Generator().manual_seed(seed)
    inputs, outputs = {}, {}
    for target, group in frames.items():
        frame = group[0]
        inputs[target] = [torch.randn(2, 3, frame.p_in.shape[1], generator=gen) for _ in SIGMAS]
        outputs[target] = [torch.randn(2, 3, frame.p_out.shape[1], generator=gen) for _ in SIGMAS]
    return inputs, outputs


def _mean_row(value):
    flat = value.detach().reshape(-1, value.shape[-1]).double()
    ones = torch.ones(flat.shape[0], 1, dtype=flat.dtype, device=flat.device)
    weight = ones.sum(0).clamp_min(1e-12).unsqueeze(-1)
    return ((ones.t() @ flat) / weight)[0]


def _reference_errors(frame, inputs, outputs):
    device = torch.device("cpu")
    jhat = _reference_j(frame, device)
    last = None
    found = []
    pairs = zip(SIGMAS, inputs, outputs, strict=True)
    for sigma, incoming, outgoing in pairs:
        z = _mean_row(incoming) @ frame.p_in.to(device, torch.float64).t()
        y = _mean_row(outgoing) @ frame.p_out.to(device, torch.float64).t()
        if last is not None and last[0] != sigma and last[1].shape == z.shape:
            predicted, error, change = cross_step_error(
                z, y, last[1], last[2], jhat, last[0], sigma
            )
            precision = (noise_weight(sigma) * change).float()
            found.append((z, y, predicted, error, precision))
        last = (sigma, z, y)
    return found


def _record(frames, inputs, outputs):
    recorder = CrossStepRecorder(frames)
    with recorder.recording():
        for index, sigma in enumerate(SIGMAS):
            recorder.set_sigma(sigma)
            for target in frames:
                recorder.observe_output(
                    target, inputs[target][index], inputs[target][index], outputs[target][index]
                )
    device = torch.device("cpu")
    built = {}
    for target, group in frames.items():
        for node, frame in enumerate(group):
            got = recorder._j((target, node), device)
            assert torch.equal(got, _reference_j(frame, device))
            built[(target, node)] = got
    errors = recorder.finish_errors()
    for target, group in frames.items():
        for node, frame in enumerate(group):
            expected = _reference_errors(frame, inputs[target], outputs[target])
            actual = [entry for entry in errors[target] if entry.node == node]
            assert len(actual) == len(expected)
            for entry, (z, y, predicted, error, precision) in zip(actual, expected, strict=True):
                assert torch.equal(entry.z, z)
                assert torch.equal(entry.y, y)
                assert torch.equal(entry.predicted, predicted)
                assert torch.equal(entry.error, error)
                assert torch.equal(entry.precision, precision)
    return built, errors


def _reference_responses(session, state):
    sources = import_source_frames(session)
    experts = session._source.experts
    by_target = {}
    for move in session.moves():
        if move.target not in sources or move.expert in experts[move.target].exact:
            continue
        by_target.setdefault(move.target, []).append(move)
    values, rows = {}, {}
    for target, moves in by_target.items():
        ex = experts[target]
        parts = state[target]
        columns = []
        for move in moves:
            rank = parts.get(f"core_{move.expert}")
            rank = rank.shape[0] if rank is not None else ex.down.shape[2]
            left = ex.up[move.expert, 0, :rank].detach().double().T.cpu()
            right = ex.down[move.expert, 0, :rank].detach().double().cpu()
            core = parts.get(f"core_{move.expert}")
            core = torch.eye(rank, dtype=torch.float64) if core is None else core.double()
            scale = parts[f"strength_{move.expert}"].double().reshape(())
            mix = scale * move.direction.double() if move.part == "core" else core
            columns.append((left, right, mix))

        def respond(x, columns=columns):
            return torch.stack(
                [((x @ right.t()) @ mix.t()) @ left.t() for left, right, mix in columns], -1
            )

        for node, (_, p_in, p_out, _) in enumerate(sources[target]):
            pinv = torch.linalg.pinv(p_in.double())

            def frame_response(z, pinv=pinv, respond=respond):
                return respond(z.double() @ pinv.t())

            probe = torch.ones(p_out.shape[0], dtype=torch.float64)
            values[(target, node)] = frame_response(probe)
            rows[(target, node)] = p_out
    return values, rows


def test_two_generations_match_the_stored_edit(imported):
    session, model = imported
    frames = _ready(session, model)
    assert frames
    inputs, outputs = _activations(frames, 5)
    first_j, first_errors = _record(frames, inputs, outputs)
    first_inputs = _reference_responses(session, session.state)
    setup = ImportEditor(session).prediction_setup(session.state)
    assert setup is not None
    _, responses, p_out, _ = setup
    for key, (_, fn) in responses.items():
        probe = torch.ones(p_out[key].shape[0], dtype=torch.float64)
        assert torch.equal(fn(probe), first_inputs[0][key])
        assert torch.equal(p_out[key], first_inputs[1][key])

    target = next(
        name
        for name, parts in session.state.items()
        if any(key.startswith("core_") for key in parts)
    )
    expert = next(key for key in session.state[target] if key.startswith("core_")).removeprefix(
        "core_"
    )
    strength = f"strength_{expert}"
    session.state[target][strength] = session.state[target][strength] + 0.05

    again = import_frames(session)
    second_j, second_errors = _record(again, inputs, outputs)
    assert any(not torch.equal(first_j[key], second_j[key]) for key in first_j)
    assert any(
        not torch.equal(a.error, b.error)
        for target_name, entries in first_errors.items()
        for a, b in zip(entries, second_errors[target_name], strict=True)
    )
    second_inputs = _reference_responses(session, session.state)
    setup = ImportEditor(session).prediction_setup(session.state)
    _, responses, p_out, _ = setup
    changed = False
    for key, (_, fn) in responses.items():
        probe = torch.ones(p_out[key].shape[0], dtype=torch.float64)
        assert torch.equal(fn(probe), second_inputs[0][key])
        if key[0] == target:
            assert not torch.equal(first_inputs[0][key], second_inputs[0][key])
            changed = True
    assert changed


def _walk(obj, seen, found):
    identity = id(obj)
    if identity in seen:
        return
    seen.add(identity)
    if isinstance(obj, torch.Tensor):
        found.append(obj)
        return
    if isinstance(obj, dict):
        for value in obj.values():
            _walk(value, seen, found)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            _walk(value, seen, found)
        return
    values = getattr(obj, "__dict__", None)
    if isinstance(values, dict):
        for value in values.values():
            _walk(value, seen, found)


def test_import_frames_keep_no_full_width_float64_lora(imported):
    session, model = imported
    frames = _ready(session, model)
    allowed = set()
    for group in frames.values():
        for frame in group:
            assert frame.up is None and frame.down is None
            for up_e, down_e, *_rest in frame.edit:
                assert up_e.dtype != torch.float64
                assert down_e.dtype != torch.float64
            for tensor in (frame.p_in, frame.p_out, frame.base):
                if isinstance(tensor, torch.Tensor) and tensor.numel():
                    allowed.add(tensor.untyped_storage().data_ptr())
    found = []
    _walk(frames, set(), found)
    for tensor in found:
        if tensor.dtype != torch.float64 or tensor.device.type != "cpu" or not tensor.numel():
            continue
        assert tensor.untyped_storage().data_ptr() in allowed


def test_block_frames_match_the_concatenated_rows(imported):
    session, model = imported
    _ready(session, model)
    recorder = import_block_recorder(session)
    assert recorder is not None
    listed = recorder.frames
    catted = BlockFrames({t: torch.cat(list(rows), 0) for t, rows in listed.outputs.items()})
    for target, rows in listed.outputs.items():
        assert isinstance(rows, tuple)
        assert rows
        if len(rows) > 1:
            stacked = torch.cat(list(rows), 0)
            held = []
            _walk(listed, set(), held)
            same = [
                tensor
                for tensor in held
                if tensor.shape == stacked.shape and torch.equal(tensor, stacked)
            ]
            assert not same
        index = block_index(target, "blocks")
        assert index is not None
        width = rows[0].shape[-1]
        got = listed.frame("blocks", index, width)
        ref = catted.frame("blocks", index, width)
        assert got is not None and ref is not None
        assert torch.equal(got, ref)


def test_unit_strength_cast_keeps_source_storage():
    down = torch.randn(2, 4)
    up = torch.randn(3, 2)
    change = LowRankChange(down.clone(), up.clone(), 1.0)
    x = torch.randn(1, 4)
    y = torch.zeros(1, 3)
    change.add(x, y)
    stored_down, stored_up = change._cast[(x.device, x.dtype)]
    assert stored_down.data_ptr() == change.down.data_ptr()
    assert stored_up.data_ptr() == change.up.data_ptr()

    dense = DenseChange(torch.randn(3, 4), 1.0)
    dense.add(x, y)
    assert dense._cast[(x.device, x.dtype)].data_ptr() == dense.delta.data_ptr()

    kron = KronChange(torch.randn(2, 2), torch.randn(2, 2), 1.0)
    kron_x = torch.randn(1, 4)
    kron.add(kron_x, torch.zeros(1, 4))
    stored_w1, stored_w2 = kron._cast[(kron_x.device, kron_x.dtype)]
    assert stored_w1.data_ptr() == kron.w1.data_ptr()
    assert stored_w2.data_ptr() == kron.w2.data_ptr()

    scaled = LowRankChange(down.clone(), up.clone(), 0.5)
    scaled.add(x, y)
    _, scaled_up = scaled._cast[(x.device, x.dtype)]
    assert scaled_up.data_ptr() != scaled.up.data_ptr()
    assert torch.equal(scaled_up, scaled.up * 0.5)

    wide = x.double()
    change.add(wide, y.double())
    cast_down, _ = change._cast[(wide.device, wide.dtype)]
    assert cast_down.dtype == torch.float64
    assert cast_down.data_ptr() != change.down.data_ptr()


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def test_routed_applies_share_readonly_storage_and_keep_writes(tmp_path):
    chain = [
        dataclasses.replace(e, axis=patching.resolve_axis(e)[0]) for e in routed_chain(tmp_path)
    ]
    base = Patcher(FakeBaseModel(_model("flux")))
    binding = patching.detect_family(base)
    model = base.model.diffusion_model
    patching._ROUTED_CACHE.clear()
    SESSIONS.clear()
    try:
        first, _ = patching.joint_choice(
            base, chain, binding, model, open_session=True, saves_dir=tmp_path
        )
        second, _ = patching.joint_choice(
            base, chain, binding, model, open_session=True, saves_dir=tmp_path
        )
        assert len(patching._ROUTED_CACHE) == 1
        cached = next(iter(patching._ROUTED_CACHE.values()))
        target = next(iter(first.adapter_sets[0].experts))
        left = first.adapter_sets[0].experts[target]
        right = second.adapter_sets[0].experts[target]
        original = cached.adapter_set.experts[target]
        assert left.keys_in.data_ptr() == right.keys_in.data_ptr() == original.keys_in.data_ptr()
        assert left.keys_out.data_ptr() == original.keys_out.data_ptr()
        width = next(iter(first.adapter_sets[0].routers))
        assert (
            first.adapter_sets[0].routers[width].data_ptr()
            == second.adapter_sets[0].routers[width].data_ptr()
            == cached.adapter_set.routers[width].data_ptr()
        )
        assert left.down.data_ptr() != right.down.data_ptr()
        assert left.down.data_ptr() != original.down.data_ptr()
        assert left.up.data_ptr() != original.up.data_ptr()
        if first.sections:
            shared = _first_tensor(first.sections[0].data)
            other = _first_tensor(second.sections[0].data)
            kept = _first_tensor(cached.sections[0].data)
            if shared is not None:
                assert other is not None and kept is not None
                assert shared.data_ptr() == other.data_ptr() == kept.data_ptr()

        session = first.session
        assert session is second.session
        owned = session._source.experts[target]
        before_right = right.down.clone()
        before_cache = original.down.clone()
        before_up = original.up.clone()
        before_owned = owned.down.clone()
        left.down.add_(1)
        assert torch.equal(right.down, before_right)
        assert torch.equal(original.down, before_cache)
        assert torch.equal(original.up, before_up)
        assert torch.equal(owned.down, before_owned)
        assert not torch.equal(left.down, original.down)
        strength = next(key for key in session.state[target] if key.startswith("strength_"))
        session.state[target][strength] = session.state[target][strength] + 0.05
        assert torch.equal(right.down, before_right)
        assert torch.equal(original.down, before_cache)
    finally:
        SESSIONS.clear()
        patching._ROUTED_CACHE.clear()
