# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import re

import torch

from kilora.live_imports import NOTE_LARGEST, ImportEditor, Move, TermMove

TARGET = "diffusion_model.layers.12.attention.qkv"


def _editor(count):
    editor = object.__new__(ImportEditor)
    editor._note_moves = {}
    for n in range(count):
        target = f"{TARGET}.{n}"
        expert = n % 2
        kind = n % 5
        if kind < 3:
            move = Move(f"{target}:{expert}:core:{kind}:0", target, expert, "core", torch.ones(1))
        elif kind == 3:
            move = Move(f"{target}:{expert}:strength", target, expert, "strength", torch.ones(1))
        else:
            move = TermMove(f"{target}:term:o{n}:i{n}", target, f"o{n}", f"i{n}", None, None)
        step = 0.0 if n % 7 == 0 else (-1) ** n * (n + 1) * 1e-4
        editor._note_moves[move.id] = (move, step)
    entries = (f"{key}={step:.4g}" for key, (_, step) in editor._note_moves.items())
    editor._full_moves = " Moves: " + ", ".join(entries) + "."
    return editor


def _expected_counts(count):
    counts = {"core": [0, 0], "strength": [0, 0], "pack term": [0, 0]}
    for n in range(count):
        label = ("core", "core", "core", "strength", "pack term")[n % 5]
        counts[label][0] += 1
        counts[label][1] += n % 7 != 0
    return counts


def test_summary_counts_and_largest(monkeypatch):
    monkeypatch.delenv("KILORA_IMPORT_NOTE_FULL", raising=False)
    count = 1000
    note = _editor(count)._moves_note()
    counts = _expected_counts(count)
    measured = sum(c[0] for c in counts.values())
    written = sum(c[1] for c in counts.values())
    assert note.startswith(f" Moves: {measured} measured, {written} written (")
    for label, (m, w) in counts.items():
        assert f"{label} {m}/{w}" in note
    largest = note.split(" Largest: ", 1)[1].rstrip(".").split(", ")
    assert len(largest) == NOTE_LARGEST
    sizes = [abs(float(entry.rsplit(" = ", 1)[1])) for entry in largest]
    assert sizes == sorted(sizes, reverse=True)
    expected = [(n + 1) * 1e-4 for n in (999, 998, 997, 996, 995)]
    assert sizes == [float(f"{s:.4g}") for s in expected]
    assert largest[0] == f"{TARGET}.999 pack term = -0.1"
    assert largest[1] == f"{TARGET}.998 LoRA 0 strength = 0.0999"
    assert largest[2] == f"{TARGET}.997 LoRA 1 core 2:0 = -0.0998"
    assert len(note) < 600
    assert "=" not in note.split(" Largest: ")[0]


def test_summary_without_written_moves(monkeypatch):
    monkeypatch.delenv("KILORA_IMPORT_NOTE_FULL", raising=False)
    editor = _editor(1)
    note = editor._moves_note()
    assert note == " Moves: 1 measured, 0 written (core 1/0)."
    editor._note_moves = {}
    assert editor._moves_note() == ""


def test_full_list_behind_environment(monkeypatch):
    monkeypatch.setenv("KILORA_IMPORT_NOTE_FULL", "1")
    editor = _editor(1000)
    note = editor._moves_note()
    assert note == editor._full_moves
    assert len(re.findall(r"=", note)) == 1000
    assert "Largest" not in note
