# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kilora.library import StreamingCodebook, retrieve


def _library():
    records = {
        "a": {"family": "flux", "axis": "creative", "tags": ["portrait"]},
        "b": {"family": "flux", "axis": "creative", "tags": ["landscape"]},
        "c": {"family": "flux", "axis": "functional", "tags": ["detail"]},
        "d": {"family": "other", "axis": "creative", "tags": ["portrait"]},
    }
    summaries = {
        "a": torch.tensor([1.0, 0.0]),
        "b": torch.tensor([0.0, 1.0]),
        "c": torch.tensor([1.0, 0.0]),
        "d": torch.tensor([1.0, 0.0]),
    }
    texts = {key: value.clone() for key, value in summaries.items()}
    book = StreamingCodebook(seed=11)
    book.add_many((key, summaries[key]) for key in ("a", "b", "c"))
    book.fit_alignment({key: texts[key] for key in ("a", "b", "c")})
    return records, summaries, texts, book


def _select(*, query=None, probe=None, **kwargs):
    return retrieve(
        *_library(),
        text_query=torch.tensor([1.0, 0.0]) if query is None else query,
        probe={"group": torch.tensor([[1.0, 0.0]])} if probe is None else probe,
        family="flux",
        k=kwargs.pop("k", 2),
        **kwargs,
    )


def test_best_and_deterministic_selection():
    first = _select(seed=42)
    assert first[0].digest == "a"
    assert first[0].score == pytest.approx(1.0)
    assert first == _select(seed=42)


def test_all_eligible_and_tags():
    assert {x.digest for x in _select(k=10, min_relevance=0)} == {"a", "b"}
    assert {x.digest for x in _select(k=10, min_relevance=0, include_tags="DETAIL")} == {
        "a",
        "b",
        "c",
    }
    assert [x.digest for x in _select(k=10, min_relevance=0, exclude_tags="Portrait")] == ["b"]


def test_all_eligible_with_negative_score():
    results = _select(
        k=10, min_relevance=0, query=torch.tensor([-1.0, 0.0]), probe={"group": torch.zeros(1, 2)}
    )
    assert {x.digest for x in results} == {"a", "b"}
    assert next(x.score for x in results if x.digest == "a") < 0


def test_threshold_and_zero_probe():
    assert [x.digest for x in _select(min_relevance=0.8)] == ["a"]
    assert _select(query=torch.zeros(2), probe={"group": torch.zeros(1, 2)}) == []


def test_ties_and_unclustered_entries():
    records, summaries, texts, book = _library()
    records["aa"] = dict(records["a"])
    summaries["aa"] = summaries["a"]
    texts["aa"] = texts["a"]
    results = retrieve(
        records,
        summaries,
        texts,
        book,
        text_query=texts["a"],
        probe={"group": torch.tensor([[1.0, 0.0]])},
        family="flux",
        k=2,
    )
    assert [x.digest for x in results] == ["a", "aa"]


def test_cluster_shortlist_and_grouped_probe():
    records = {f"{i:02}": {"family": "flux", "tags": []} for i in range(20)}
    summaries = {key: torch.tensor([1.0, 0.0, 0.0, 1.0]) for key in records}
    summaries["00"] = torch.tensor([0.0, 1.0, 1.0, 0.0])
    texts = {key: value.clone() for key, value in summaries.items()}
    book = StreamingCodebook(seed=2)
    book.add_many(summaries.items())
    book.fit_alignment(texts)
    results = retrieve(
        records,
        summaries,
        texts,
        book,
        text_query=texts["00"],
        probe={"first": torch.tensor([[0.0, 2.0]]), "second": torch.tensor([[3.0, 0.0]])},
        family="flux",
        k=1,
    )
    assert results[0].digest == "00"
    assert results[0].score == pytest.approx(1.0)


def test_probe_matches_groups_by_name_and_uses_the_whole_subspace():
    from kilora.library.retrieve import _content_share

    groups = {"0-attention-in-2": 2, "1-attention-in-2": 2, "0-mlp-in-3": 3}
    probe = {"0-attention-in-2": torch.tensor([[1.0, 1.0]]), "0-mlp-in-3": torch.eye(3)[:1]}
    summary = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    assert abs(_content_share(summary, probe, groups) - (1.0 + 1.0 + 1.0) / (2 + 2 + 1)) < 1e-6
    full = {g: torch.eye(w) for g, w in groups.items()}
    assert abs(_content_share(summary, probe, groups, full) - 1.0) < 1e-6
    with pytest.raises(ValueError, match="family groups"):
        _content_share(torch.zeros(5), probe, groups)
