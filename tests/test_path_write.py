# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from kilora_editor_delta import (
    default_v0,
    delta_write,
    load_path_editor,
    recorder_for,
    save_path_editor,
)
from live_helpers import build_live_mod
from test_comfyui import FakeBaseModel, FakePatcher, denoiser, folders

from kilora.comfyui import nodes as N
from kilora.comfyui import patching
from kilora.editors import WriteRule, register_rule
from kilora.live import SESSIONS
from kilora.recording import PathRecorder


@pytest.fixture(autouse=True)
def fresh_sessions():
    SESSIONS.clear()
    yield
    SESSIONS.clear()


def _setup(denoiser, folders, tmp_path):
    build_live_mod(denoiser, folders["kilora"] / "ink.kmod")
    base = FakePatcher(FakeBaseModel(denoiser))
    (chain,) = N.KiLoRASelect().select("ink.kmod", 1.0, 1.0)
    patched, _, _ = N.KiLoRAApply().apply(base, chain, 1)
    session = patching.sessions_of(patched)[0]
    session.learning = True
    cols = {t: int(p["down"].numel() // p["down"].shape[-1]) for t, p in session.state.items()}
    outs = {t: p["up"].shape[-1] for t, p in session.state.items()}
    targets = list(session.state)
    gen = torch.Generator().manual_seed(11)
    path = tmp_path / "path.kedit.safetensors"
    save_path_editor(
        path,
        id="path-test",
        version="1.0.0",
        base_family=session.mod.manifest["base_model"]["family"],
        v0=default_v0(targets, cols, seed=5),
        beta={t: 0.5 for t in targets},
        lam={t: 0.3 for t in targets},
        h_bar={t: torch.randn(outs[t], 4, generator=gen) * 0.1 for t in targets},
    )
    editor = load_path_editor(path)
    xs = [torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(i)) for i in range(3)]
    return base, chain, patched, session, editor, xs


class _Collect(PathRecorder):
    def observe(self, target, y):
        if target in self.a:
            self.sums.setdefault(target, []).append(y.detach().reshape(-1, y.shape[-1]).double())


def test_streamed_statistics_equal_offline(denoiser, folders, tmp_path):
    base, chain, patched, session, editor, xs = _setup(denoiser, folders, tmp_path)
    rec = recorder_for(session, editor)
    with rec.recording():
        for x in xs:
            patched.run(x)
    stats = rec.finish()
    raw = _Collect(a=rec.a, v0=rec.v0)
    with raw.recording():
        for x in xs:
            patched.run(x)
    assert set(stats.c_kk) == set(raw.sums) and stats.c_kk
    for t, ys in raw.sums.items():
        y = torch.cat(ys)
        c = y @ (rec.v0[t].double() @ rec.a[t].double()).t()
        k = c / torch.sqrt((c * c).sum(1).mean())
        c_kk = k.t() @ k / k.shape[0]
        c_vk = y.t() @ k / k.shape[0]
        torch.testing.assert_close(stats.c_kk[t], c_kk.float(), rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(stats.c_vk[t], c_vk.float(), rtol=1e-5, atol=1e-7)
        assert abs(float(torch.trace(stats.c_kk[t])) - 1.0) < 1e-5


def test_outputs_are_identical_with_recorder_on_and_off(denoiser, folders, tmp_path):
    base, chain, patched, session, editor, xs = _setup(denoiser, folders, tmp_path)
    off = [patched.run(x) for x in xs]
    rec = recorder_for(session, editor)
    with rec.recording():
        on = [patched.run(x) for x in xs]
    assert rec.sums
    assert all(torch.equal(a, b) for a, b in zip(off, on, strict=True))


def test_interrupted_sampling_writes_nothing(denoiser, folders, tmp_path):
    base, chain, patched, session, editor, xs = _setup(denoiser, folders, tmp_path)
    before = {t: p["up"].clone() for t, p in session.state.items()}
    rec = recorder_for(session, editor)
    with pytest.raises(KeyboardInterrupt), rec.recording():
        patched.run(xs[0])
        raise KeyboardInterrupt
    report = delta_write(session, editor, rec.finish())
    assert "Nothing was recorded" in report
    assert session.commits == []
    assert all(torch.equal(before[t], p["up"]) for t, p in session.state.items())


def test_write_commits_and_undo_restores_b(denoiser, folders, tmp_path):
    base, chain, patched, session, editor, xs = _setup(denoiser, folders, tmp_path)
    before = {t: p["up"].clone() for t, p in session.state.items()}
    rec = recorder_for(session, editor)
    with rec.recording():
        for x in xs:
            patched.run(x)
    report = delta_write(session, editor, rec.finish())
    assert "1 commits" in report and len(session.commits) == 1
    for t, p in session.state.items():
        d = (p["up"] - before[t]).reshape(-1, p["up"].shape[-1]).t()
        assert d.abs().max() > 0
        v0 = editor.v0[t]
        outside = d - (d @ v0.t()) @ v0
        assert outside.abs().max() < 1e-5
    rewritten, _, _ = N.KiLoRAApply().apply(base, chain, 1)
    assert not torch.equal(rewritten.run(xs[0]), patched.run(xs[0]))
    session.undo()
    assert all(torch.equal(before[t], p["up"]) for t, p in session.state.items())
    restored, _, _ = N.KiLoRAApply().apply(base, chain, 1)
    assert torch.equal(restored.run(xs[0]), patched.run(xs[0]))


def test_learning_switch_off_records_and_writes_nothing(denoiser, folders, tmp_path):
    base, chain, patched, session, editor, xs = _setup(denoiser, folders, tmp_path)
    session.learning = False
    rec = recorder_for(session, editor)
    with rec.recording():
        for x in xs:
            patched.run(x)
    assert rec.sums == {}
    report = delta_write(session, editor, rec.finish())
    assert "off" in report and session.commits == []


def test_save_and_reload_after_write(denoiser, folders, tmp_path):
    base, chain, patched, session, editor, xs = _setup(denoiser, folders, tmp_path)
    rec = recorder_for(session, editor)
    with rec.recording():
        patched.run(xs[0])
    delta_write(session, editor, rec.finish())
    written = {t: p["up"].clone() for t, p in session.state.items()}
    name = session.save_point("after-write")
    session.reset()
    session.load(name)
    assert all(torch.equal(written[t], p["up"]) for t, p in session.state.items())


class WrapPatcher(FakePatcher):
    def clone(self):
        return WrapPatcher(self.model, self.object_patches, self.model_options)

    def set_model_unet_function_wrapper(self, fn):
        self.model_options["model_function_wrapper"] = fn

    def sample(self, x, steps=3, guidance=True, stop_at=None):
        fn = self.model_options.get("model_function_wrapper")
        outs = []
        for i in range(steps):
            if stop_at is not None and i == stop_at:
                raise KeyboardInterrupt
            batch = torch.cat([x, -x]) if guidance else x
            cou = [0, 1] if guidance else [0]

            def apply_model(inp, t, **c):
                return self.run(inp)

            t = torch.tensor([steps - i])
            args = {"input": batch, "timestep": t, "c": {}, "cond_or_uncond": cou}
            outs.append(fn(apply_model, args) if fn else apply_model(batch, args["timestep"]))
        return outs


def test_comfyui_record_and_write_nodes(denoiser, folders, tmp_path):
    base, chain, patched, session, editor, xs = _setup(denoiser, folders, tmp_path)
    import shutil

    shutil.copy(tmp_path / "path.kedit.safetensors", folders["kilora"] / "path.kedit.safetensors")
    wp = WrapPatcher(patched.model, patched.object_patches, patched.model_options)
    model, handle = N.KiLoRALiveRecord().record(wp, "path.kedit.safetensors")
    plain = wp.sample(xs[0])
    recorded = model.sample(xs[0])
    assert all(torch.equal(a, b) for a, b in zip(plain, recorded, strict=True))
    assert {s.n for s in handle.recorder.sums.values()} == {15}
    samples, report = N.KiLoRALiveWrite().write(handle, {"samples": xs[0]})["result"]
    assert "1 commits" in report and samples["samples"] is xs[0]
    _, report = N.KiLoRALiveWrite().write(handle, {"samples": xs[0]})["result"]
    assert "Nothing was recorded" in report and len(session.commits) == 1


def test_comfyui_interrupted_sampling_writes_nothing(denoiser, folders, tmp_path):
    base, chain, patched, session, editor, xs = _setup(denoiser, folders, tmp_path)
    import shutil

    shutil.copy(tmp_path / "path.kedit.safetensors", folders["kilora"] / "path.kedit.safetensors")
    wp = WrapPatcher(patched.model, patched.object_patches, patched.model_options)
    model, handle = N.KiLoRALiveRecord().record(wp, "path.kedit.safetensors")
    with pytest.raises(KeyboardInterrupt):
        model.sample(xs[0], stop_at=2)
    model, handle = N.KiLoRALiveRecord().record(wp, "path.kedit.safetensors")
    _, report = N.KiLoRALiveWrite().write(handle, {"samples": xs[0]})["result"]
    assert "Nothing was recorded" in report and session.commits == []


class _SpyRule(WriteRule):
    def __init__(self, path, write=True):
        self.records = []
        self.do_write = write

    def next_state(self, state, record):
        self.records.append(record)
        if not self.do_write:
            return None
        return {t: {**parts, "up": parts["up"] * 0.5} for t, parts in state.items()}


def _spy_file(folders, name):
    import json

    from safetensors.torch import save_file

    path = folders["kilora"] / name
    save_file(
        {"x": torch.zeros(1)},
        str(path),
        metadata={"kilora.manifest": json.dumps({"format": "kilora-editor", "kind": "spy-rule"})},
    )
    return name


def _spy_setup(denoiser, folders, tmp_path, write=True):
    base, chain, patched, session, editor, xs = _setup(denoiser, folders, tmp_path)
    spies = []

    def factory(path):
        spies.append(_SpyRule(path, write))
        return spies[-1]

    register_rule("spy-rule", factory)
    name = _spy_file(folders, "spy.kedit.safetensors")
    wp = WrapPatcher(patched.model, patched.object_patches, patched.model_options)
    model, handle = N.KiLoRALiveRecord().record(wp, name)
    return session, model, handle, spies, xs


def test_plugged_rule_gets_latent_conditioning_and_timesteps(denoiser, folders, tmp_path):
    session, model, handle, spies, xs = _spy_setup(denoiser, folders, tmp_path)
    before = {t: p["up"].clone() for t, p in session.state.items()}
    assert handle.recorder is None
    model.sample(xs[0], steps=3)
    final = {"samples": torch.randn(1, 5, 4)}
    _, report = N.KiLoRALiveWrite().write(handle, final)["result"]
    rec = spies[0].records[0]
    assert torch.equal(rec.latent, final["samples"])
    assert rec.timesteps == [3.0, 2.0, 1.0]
    assert isinstance(rec.conditioning, dict) and rec.stats is None
    assert "1 commits" in report
    assert all(torch.equal(p["up"], before[t] * 0.5) for t, p in session.state.items())
    session.undo()
    assert all(torch.equal(p["up"], before[t]) for t, p in session.state.items())


def test_plugged_rule_without_write_commits_nothing(denoiser, folders, tmp_path):
    session, model, handle, spies, xs = _spy_setup(denoiser, folders, tmp_path, write=False)
    model.sample(xs[0], steps=2)
    _, report = N.KiLoRALiveWrite().write(handle, {"samples": xs[0]})["result"]
    assert "no write" in report and session.commits == []


def test_plugged_rule_with_learning_off_is_not_called(denoiser, folders, tmp_path):
    session, model, handle, spies, xs = _spy_setup(denoiser, folders, tmp_path)
    session.learning = False
    model.sample(xs[0], steps=2)
    _, report = N.KiLoRALiveWrite().write(handle, {"samples": xs[0]})["result"]
    assert "off" in report and spies[0].records == [] and session.commits == []
