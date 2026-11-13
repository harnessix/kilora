# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import random

from kilora.palette import (
    PaletteEntry,
    Resolved,
    ShuffleState,
    load_shuffle_state,
    reset_shuffle_state,
    reward_shuffle,
    save_shuffle_state,
    shuffle,
)

KEY = hashlib.sha256(b"model").hexdigest()


def _entries(vectors):
    entries = [Resolved(PaletteEntry("lora", str(i)), str(i)) for i in range(len(vectors))]
    embeddings = {str(i): (KEY, vector) for i, vector in enumerate(vectors) if vector is not None}
    return entries, embeddings


def test_embedding_state_round_trip_fallback_and_reset(tmp_path):
    entries, embeddings = _entries([[1.0, 0.0], None])
    entries.append(Resolved(PaletteEntry("lora", "fixed", axis="functional"), "fixed"))
    state = ShuffleState()
    pick = shuffle(
        entries, 2, (2, 2), (0, 1), mode="learn_embedding", state=state, embeddings=embeddings
    )
    assert pick[-1] == entries[-1]
    reward_shuffle(state, pick, (0, 1), "keep", embeddings)
    assert len(state.counts) == 2 and KEY in state.embedding
    path = tmp_path / "a.kpalette.json"
    save_shuffle_state(path, state)
    data = json.loads((tmp_path / "a.kpalette.state.json").read_text())
    assert set(data) == {"embedding", *state.counts}

    def check_private(value):
        if isinstance(value, dict):
            for key, child in value.items():
                assert key in {"embedding", "B", "f"} or (
                    len(key) == 64 and all(c in "0123456789abcdef" for c in key)
                )
                check_private(child)
        elif isinstance(value, list):
            for child in value:
                check_private(child)
        else:
            assert type(value) in (int, float)

    check_private(data)
    assert load_shuffle_state(path) == state
    assert pick == shuffle(
        entries, 2, (2, 2), (0, 1), mode="learn_embedding", embeddings=embeddings
    )
    assert shuffle(
        entries, 9, (1, 1), (0, 1), mode="learn_embedding", state=state, embeddings=embeddings
    ) == shuffle(
        entries,
        9,
        (1, 1),
        (0, 1),
        mode="learn_embedding",
        state=load_shuffle_state(path),
        embeddings=embeddings,
    )
    reset_shuffle_state(path)
    assert load_shuffle_state(path) == ShuffleState()


def test_embedding_feedback_uses_one_observation_for_the_whole_pick():
    entries, embeddings = _entries([[1.0, 0.0], [0.0, 1.0]])
    state = ShuffleState()
    reward_shuffle(state, entries, (0, 1), "keep", embeddings)
    posterior = state.embedding[KEY]
    assert posterior["B"][0][1] == 1.0
    assert posterior["f"][:2] == [1.0, 1.0]
    assert all(counts[0] == [1, 0] for counts in state.counts.values())


def test_embedding_bandit_learns_best_subset():
    entries, embeddings = _entries([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    state = ShuffleState()
    best = {"0", "1"}
    noise = [True] * 6 + [False] * 24
    random.Random(21).shuffle(noise)
    for seed in range(30):
        pick = shuffle(
            entries,
            seed,
            (2, 2),
            (0, 1),
            mode="learn_embedding",
            state=state,
            embeddings=embeddings,
        )
        keep = ({item.path for item in pick} == best) != noise[seed]
        reward_shuffle(state, pick, (0, 1), "keep" if keep else "discard", embeddings)
    successes = sum(
        {
            item.path
            for item in shuffle(
                entries,
                seed,
                (2, 2),
                (0, 1),
                mode="learn_embedding",
                state=state,
                embeddings=embeddings,
            )
        }
        == best
        for seed in range(100, 300)
    )
    assert successes >= 140


def test_embedding_generalizes_to_unseen_entry():
    entries, embeddings = _entries([[1, 0], [0.99, 0.01], [-1, 0], [0, 1]])
    embedding_state, beta_state = ShuffleState(), ShuffleState()
    for i in range(8):
        chosen = [Resolved(entries[0].entry, entries[0].path)] if i % 2 == 0 else [entries[2]]
        reward = "keep" if i % 2 == 0 else "discard"
        reward_shuffle(embedding_state, chosen, (0, 1), reward, embeddings)
        reward_shuffle(beta_state, chosen, (0, 1), reward)
    candidates = [entries[1], entries[3]]

    def rate(mode, state):
        return sum(
            shuffle(
                candidates, seed, (1, 1), (0, 1), mode=mode, state=state, embeddings=embeddings
            )[0].path
            == "1"
            for seed in range(300)
        )

    assert rate("learn_embedding", embedding_state) > 180
    assert 120 < rate("learn", beta_state) < 180


def test_node_passes_indexed_embeddings_and_persists_feedback(tmp_path, monkeypatch):
    import torch

    from kilora import palette
    from kilora.comfyui import nodes
    from kilora.library import LibraryIndex

    folder = tmp_path / "loras"
    folder.mkdir()
    lora = folder / "a.safetensors"
    lora.write_bytes(b"local example")
    pal_dir = tmp_path / "palettes"
    pal_dir.mkdir()
    monkeypatch.setattr(palette, "PALETTE_DIR", str(pal_dir))
    monkeypatch.setattr(nodes.FOLDERS, "path", lambda kind, name: str(folder / name))
    monkeypatch.setattr(nodes.FOLDERS, "dirs", lambda kind: [str(folder)])
    monkeypatch.setattr(nodes.FOLDERS, "list", lambda *args: [])
    model = tmp_path / "model"
    model.mkdir()
    index_path = tmp_path / "index"
    index_path.mkdir()
    for path in (
        model / "config.json",
        model / "model.safetensors",
        index_path / "basis.json",
        index_path / "basis.safetensors",
    ):
        path.write_bytes(path.name.encode())

    class Index:
        path = index_path
        paths = {str(lora): "abc"}
        records = {"abc": {"family": "flux"}}

        def sync(self, paths, reuse_text=False):
            assert paths == [str(folder)] and reuse_text

        def embedding_inputs(self, family, model_path):
            assert family == "flux" and model_path == str(model)
            return {"abc": torch.tensor([1.0, 0.0])}

    monkeypatch.setattr(LibraryIndex, "open", lambda: Index())
    palette.save_palette(
        pal_dir / "a.kpalette.json", palette.Palette("a", [palette.PaletteEntry("lora", lora.name)])
    )
    _, _, _, report = nodes.KiLoRAPalette().load(
        "a.kpalette.json",
        shuffle=True,
        pick_min=1,
        pick_max=1,
        mode="learn_embedding",
        embedding_model_path=str(model),
        reward="keep",
    )
    assert "feedback recorded" in report
    state = palette.load_shuffle_state(pal_dir / "a.kpalette.json")
    assert len(state.counts) == 1 and len(state.embedding) == 1
    assert "learn_embedding" in nodes.KiLoRAPalette.INPUT_TYPES()["optional"]["mode"][0]


def test_old_modes_and_missing_vectors_keep_their_picks():
    entries, _ = _entries([[1, 0], [0, 1], None, None])
    state = ShuffleState()
    for seed in range(20):
        random_pick = shuffle(entries, seed, (2, 3), (0.2, 0.9))
        assert random_pick == shuffle(
            entries,
            seed,
            (2, 3),
            (0.2, 0.9),
            mode="random",
            state=state,
            embeddings={"0": (KEY, [1, 0])},
        )
        beta_pick = shuffle(entries, seed, (2, 3), (0.2, 0.9), mode="learn", state=state)
        assert beta_pick == shuffle(
            entries, seed, (2, 3), (0.2, 0.9), mode="learn_embedding", state=state, embeddings={}
        )
