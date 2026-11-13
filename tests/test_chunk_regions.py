# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch
from test_chunk_live import _record, _ups, setup
from test_regions import _planted_probes

from kilora.editors import commit_write, editor_for
from kilora.regions import Region
from kilora.subspace_keys import subspace_distance


def _region_write(session, path, target, name):
    editor = editor_for(path)
    editor.check(session)
    t0 = editor.targets[0]
    d = int(editor.routes[t0][1].shape[-1])
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(9)))
    m = min(4, d // 2)
    signal, other = q[:, :m].t(), q[:, m : 2 * m].t()
    probes = _planted_probes(signal, other, d=d)
    probes = {t0: probes["layer.attn"]}
    mask = torch.zeros(1, 1, 6, 8)
    mask[..., :4] = 1.0
    record = _record(session, target)
    record.extra["region"] = Region(mask, name=name, caption=name, probes=probes)
    return commit_write(session, editor, record), editor, signal, t0


def test_region_expert_survives_save_load_undo(setup):
    session, path, target = setup
    before = _ups(session)
    report, editor, signal, t0 = _region_write(session, path, target, "red scarf")
    assert "1 commits" in report
    book = session.editor_state.regions
    (entry,) = book.experts
    assert entry.name == "red scarf"
    m = signal.shape[0]
    assert subspace_distance(entry.keys[t0]["keys_in"][:m], signal) < 0.2
    for t, up in _ups(session).items():
        changed = (up - before[t]).reshape(-1, up.shape[-1]).abs().sum(-1) > 0
        cols = editor.col_expert[t]
        assert not bool(changed[cols != entry.index].any()), t
    name = session.save_point("region")
    _region_write(session, path, target, "blue hat")
    assert [e.name for e in session.editor_state.regions.experts] == ["red scarf", "blue hat"]
    session.undo()
    assert [e.name for e in session.editor_state.regions.experts] == ["red scarf"]
    _region_write(session, path, target, "green cup")
    session.load(name)
    assert [e.name for e in session.editor_state.regions.experts] == ["red scarf", "green cup"]
    session.editor_state.checkout("branch-1")
    (back,) = session.editor_state.regions.experts
    assert (back.name, back.caption, back.index, back.new) == (
        entry.name,
        entry.caption,
        entry.index,
        entry.new,
    )
    assert torch.equal(back.keys[t0]["keys_in"], entry.keys[t0]["keys_in"])
    assert session.editor_state.regions.find("Red Scarf") is back
