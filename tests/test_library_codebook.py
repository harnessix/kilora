# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch

from kilora.library import FamilyCodebooks, StreamingCodebook


def planted():
    generator = torch.Generator().manual_seed(47)
    centers = torch.randn(8, 24, generator=generator)
    vectors = centers[torch.arange(400) % 8] + 0.12 * torch.randn(400, 24, generator=generator)
    return [(f"{i:064x}", vector) for i, vector in enumerate(vectors)]


def test_balanced_clusters_and_streaming_order():
    entries = planted()
    batch = StreamingCodebook(seed=12)
    batch.add_many(reversed(entries))
    stream = StreamingCodebook(seed=12)
    for key, vector in entries:
        stream.add(key, vector)
    assert batch.assignments == stream.assignments
    assert batch.sizes == stream.sizes
    assert all(torch.equal(batch.centroids[i], stream.centroids[i]) for i in batch.centroids)
    assert max(batch.sizes.values()) < 2 * len(entries) / len(batch.sizes)
    assert abs(len(batch.sizes) - math.ceil(math.sqrt(len(entries)))) <= 1


def test_remove_and_readd_restores_assignment_and_centroids():
    entries = planted()[:100]
    book = StreamingCodebook(seed=5)
    book.add_many(entries)
    before = book.assignments
    centroids = book.centroids
    key, vector = entries[41]
    book.remove(key)
    assert key not in book.assignments
    book.add(key, vector)
    assert book.assignments == before
    assert all(torch.equal(centroids[i], book.centroids[i]) for i in centroids)


def test_cluster_count_and_removal_to_empty():
    book = StreamingCodebook()
    entries = planted()[:80]
    for n, (key, vector) in enumerate(entries, 1):
        book.add(key, vector)
        assert abs(len(book.sizes) - math.ceil(math.sqrt(n))) <= 1
    for n, (key, _) in zip(range(79, -1, -1), entries, strict=False):
        book.remove(key)
        assert abs(len(book.sizes) - math.ceil(math.sqrt(n))) <= 1
    assert not book.sizes


def test_alignment_families_and_validation():
    books = FamilyCodebooks()
    one = books.for_family("one")
    two = books.for_family("two")
    one.add_many([("a", torch.tensor([1.0, 2.0])), ("b", torch.tensor([3.0, 4.0]))])
    two.add("a", torch.tensor([7.0]))
    matrix = one.fit_alignment({"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0, 1.0])})
    for row, key in enumerate(("a", "b")):
        torch.testing.assert_close(matrix[row], one.centroids[one.assignments[key]] / 1.001)
    assert one.alignment is not None
    one.remove("a")
    assert one.alignment is None
    assert two.sizes == {0: 1}
    with pytest.raises(ValueError):
        one.add("c", torch.tensor([float("nan"), 0.0]))
    with pytest.raises(ValueError):
        one.add_many([("c", torch.ones(2)), ("d", torch.ones(3))])
    assert "c" not in one.assignments


def test_family_codebook_round_trip_is_exact():
    torch.manual_seed(32)
    books = FamilyCodebooks(seed=4)
    values = {f"{i:064x}": torch.randn(12) for i in range(30)}
    book = books.for_family("demo")
    book.add_many(values.items())
    state, tensors = books.dump()
    tensors.update({f"{key}/summary": value for key, value in values.items()})
    restored = FamilyCodebooks.load(state, tensors, seed=4).for_family("demo")
    assert restored.assignments == book.assignments
    assert restored._next_id == book._next_id
    assert all(torch.equal(v, restored.centroids[k]) for k, v in book.centroids.items())


def test_five_hundred_adds_are_fast():
    import time

    torch.manual_seed(21)
    values = torch.randn(500, 512)
    book = StreamingCodebook()
    start = time.monotonic()
    for i, value in enumerate(values):
        book.add(f"{i:064x}", value)
    assert time.monotonic() - start < 5
