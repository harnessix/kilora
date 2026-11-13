# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import fnmatch
from unittest.mock import patch

import pytest
import torch
from kilora_editor_chunk import ChunkEditor
from kilora_editor_chunk.build import ZImageCollector, build, collect
from test_chunk_build import StandIn

from kilora.editor_packs import read_pack
from kilora.lora_import import _resolver
from kilora.presets import get_preset


def test_host_collects_resolver_targets_and_input_basis(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            attention = torch.nn.Module()
            attention.to_q = torch.nn.Linear(5, 7)
            attention.to_k = torch.nn.Linear(5, 7)
            block = torch.nn.Module()
            block.attn1 = attention
            self.transformer_blocks = torch.nn.ModuleList([block])

        def forward(self, x):
            attention = self.transformer_blocks[0].attn1
            return attention.to_q(x) + attention.to_k(x)

    model = Model().requires_grad_(False)

    class Pipe:
        transformer = model

    class TinyCollector(ZImageCollector):
        def _embed(self, text):
            return text

        def generate(self, prompt, seed):
            self.keys, self.counts = {}, {}
            x = torch.randn(12, 5, generator=torch.Generator().manual_seed(seed))
            original_cpu = torch.Tensor.cpu

            def guarded_cpu(tensor, *args, **kwargs):
                assert all(tensor is not v for v in self.covariance_sums.values())
                return original_cpu(tensor, *args, **kwargs)

            with torch.no_grad(), patch.object(torch.Tensor, "cpu", guarded_cpu):
                self.model(x)
                assert not self._covariance_seen
                self.model(x)
                assert not self._covariance_seen
            keys = {}
            self.keys = None
            return x, keys

        def grad(self, latent, prompt, seed):
            self.du = {n: torch.zeros(self.shapes[n], requires_grad=True) for n in self.names}
            loss = self.model(latent).square().mean()
            grads = torch.autograd.grad(loss, [self.du[n] for n in self.names])
            self.du = None
            return dict(zip(self.names, grads, strict=True))

    collector = TinyCollector(
        Pipe(), None, model=model, host_family="anima", hooks="*attn1.to_*", columns=2
    )
    modules = dict(model.named_modules())
    linear = {n: m for n, m in modules.items() if isinstance(m, torch.nn.Linear)}
    preset = get_preset("anima")
    resolve = _resolver("diffusers", list(modules), preset.aliases(list(linear)), strict=False)
    expected = [
        n
        for n in linear
        if fnmatch.fnmatchcase(n, "*attn1.to_*")
        and any(fnmatch.fnmatchcase(n, p) for p in preset.full_target_modules)
        and n in resolve(n)
    ]
    assert collector.names == expected and expected
    events = collect(
        collector, {"_common": ["a cat"]}, tmp_path / "events", draws=1, log=lambda _: None
    )
    event = torch.load(events[0], weights_only=False)
    assert set(event) == {"family", "prompt", "seed", "draws", "collection", "g_empty", "g_full"}
    x = torch.randn(12, 5, generator=torch.Generator().manual_seed(1000))
    group = collector.covariance_index[expected[0]]
    assert torch.allclose(collector.covariance_sums[group], 2 * (x.T @ x))
    assert collector.covariance_counts[group] == 24
    assert group == collector.covariance_index[expected[1]]
    old_event = dict(event, keys={n: torch.zeros(5, 5, dtype=torch.float64) for n in expected})
    old_file = tmp_path / "old.pt"
    torch.save(old_event, old_file)
    assert events[0].stat().st_size < old_file.stat().st_size
    result = build(events, collector.names, collector.shapes, tmp_path / "packs", "anima", common=2)
    editor = ChunkEditor.load_packs([result.packs["common"]])
    assert editor.names == expected
    for name in expected:
        assert editor.shapes[name] == [linear[name].out_features, 2]
        assert torch.equal(editor.chunks[0].input_basis[name], collector.proj[name].T)
        basis = editor.chunks[0].input_basis[name]
        values = editor.chunks[0].input_eigenvalues[name]
        data = torch.load(tmp_path / "events" / "covariance.pt", weights_only=False)
        group = data["index"][name]
        cov = data["sums"][group] / data["counts"][group]
        assert basis.shape == (linear[name].in_features, 2)
        assert torch.allclose(basis.T @ basis, torch.eye(2), atol=1e-5)
        assert torch.allclose(cov @ basis, basis * values, atol=1e-5, rtol=1e-4)
        assert torch.allclose(
            values, torch.linalg.eigvalsh(cov.double())[-2:].flip(0).float(), atol=1e-5, rtol=1e-4
        )


def test_legacy_pack_tensors_and_load(tmp_path):
    collector = StandIn()
    events = collect(
        collector, {"_common": ["one", "two"]}, tmp_path / "events", log=lambda _: None
    )
    first = build(events, collector.names, collector.shapes, tmp_path / "first", "tiny", common=2)
    second = build(events, collector.names, collector.shapes, tmp_path / "second", "tiny", common=2)
    _, a = read_pack(first.packs["common"])
    _, b = read_pack(second.packs["common"])
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not any(k.startswith("input_") for k in a)
    loaded = ChunkEditor.load_packs([first.packs["common"]])
    assert loaded.chunks[0].input_basis == {}
    assert loaded.chunks[0].input_eigenvalues == {}


def test_covariance_splits_and_deduplicates(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            block = torch.nn.Module()
            block.attn1 = torch.nn.Module()
            block.attn1.to_q = torch.nn.Linear(256, 4)
            block.attn1.to_k = torch.nn.Linear(256, 4)
            block.ff = torch.nn.Module()
            block.ff.net = torch.nn.ModuleList(
                [torch.nn.Identity(), torch.nn.Identity(), torch.nn.Linear(256, 4)]
            )
            self.transformer_blocks = torch.nn.ModuleList([block])

        def forward(self, x, y):
            a = self.transformer_blocks[0].attn1
            return a.to_q(x) + a.to_k(y)

    class Tiny(ZImageCollector):
        def _embed(self, text):
            return text

    model = Model()
    collector = Tiny(None, None, model=model, host_family="anima", columns=2)
    assert len(collector.names) == 2
    assert all(".attn1." in n for n in collector.names)
    explicit = Tiny(None, None, model=Model(), host_family="anima", hooks="*ff*", columns=2)
    assert explicit.names and all(".ff." in n for n in explicit.names)
    x, y = torch.randn(3, 256), torch.randn(3, 256)
    collector.covariance_active = True
    model(x, x)
    assert not collector._covariance_seen
    shared = collector.covariance_index.copy()
    data = {
        "sums": {g: t.cpu() for g, t in collector.covariance_sums.items()},
        "counts": collector.covariance_counts.copy(),
        "index": shared,
    }
    assert len(data["sums"]) == 1 and len(set(shared.values())) == 1
    new_file, old_file = tmp_path / "new.pt", tmp_path / "old.pt"
    torch.save(data, new_file)
    torch.save(
        {
            "sums": {n: data["sums"][g].clone() for n, g in shared.items()},
            "counts": {n: 3 for n in shared},
        },
        old_file,
    )
    print(
        f"toy covariance file sizes: old={old_file.stat().st_size}, new={new_file.stat().st_size}"
    )
    assert 0.48 < new_file.stat().st_size / old_file.stat().st_size < 0.52
    model(x, y)
    assert not collector._covariance_seen
    q, k = collector.names
    assert collector.covariance_index[q] != collector.covariance_index[k]
    assert torch.allclose(collector.covariance_sums[collector.covariance_index[q]], 2 * (x.T @ x))
    assert torch.allclose(
        collector.covariance_sums[collector.covariance_index[k]], x.T @ x + y.T @ y
    )
    assert all(v == 6 for v in collector.covariance_counts.values())


def test_old_covariance_rejected_and_estimate_printed(tmp_path, capsys):
    from kilora_editor_chunk.build import _input_bases

    with pytest.raises(ValueError, match="collect again"):
        _input_bases({"sums": {}, "counts": {}}, [], {})

    class Fake:
        host = True
        names = ["hook"]
        shapes = {"hook": [2, 1]}
        input_widths = {"hook": 2}

        def generate(self, prompt, seed):
            raise RuntimeError("stop after estimate")

    with pytest.raises(RuntimeError, match="stop after estimate"):
        collect(Fake(), {"x": ["one"]}, tmp_path)
    output = capsys.readouterr().out
    assert "GPU running sums 16 bytes" in output
    assert "covariance.pt tensor data 16 bytes" in output
