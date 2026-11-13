# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import argparse
import dataclasses
import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from kilora import compose
from kilora.cli import _import, import_license
from kilora.modfile import load_mod, read_manifest
from kilora.multi_import import LoRASource, import_loras, stack_set

DIM, HIDDEN = 12, 20
LAYERS = ["blocks.0.self_attn.q_proj", "blocks.0.mlp.layer1", "blocks.0.mlp.layer2"]


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(DIM, DIM)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(DIM, HIDDEN)
        self.layer2 = nn.Linear(HIDDEN, DIM)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()
        self.mlp = _MLP()

    def forward(self, x):
        x = x + self.self_attn.q_proj(x)
        return x + self.mlp.layer2(torch.nn.functional.gelu(self.mlp.layer1(x)))


class Toy(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.blocks = nn.ModuleList([_Block()])

    def forward(self, x):
        return self.blocks[0](x)


def _lora(model, path, *, rank, seed, alpha=None, layers=LAYERS, license=None):
    gen = torch.Generator().manual_seed(seed)
    sd, weights = {}, {}
    for name in layers:
        m = model.get_submodule(name)
        a = torch.randn(rank, m.in_features, generator=gen) * 0.2
        b = torch.randn(m.out_features, rank, generator=gen) * 0.2
        stem = "lora_unet_" + name.replace(".", "_")
        sd[f"{stem}.lora_down.weight"], sd[f"{stem}.lora_up.weight"] = a, b
        if alpha is not None:
            sd[f"{stem}.alpha"] = torch.tensor(float(alpha))
        weights[name] = (a, b)
    meta = {"modelspec.license": license} if license else None
    save_file(sd, str(path), metadata=meta)
    return weights


@pytest.fixture
def three(tmp_path):
    model = Toy().eval()
    specs = [
        dict(rank=2, seed=1, alpha=1.0),
        dict(rank=4, seed=2, alpha=8.0),
        dict(rank=3, seed=3, alpha=None, layers=LAYERS[1:]),
    ]
    paths, weights, scales = [], [], []
    for i, spec in enumerate(specs):
        p = tmp_path / f"lora{i}.safetensors"
        weights.append(_lora(model, p, **spec))
        paths.append(str(p))
        scales.append(spec["alpha"] / spec["rank"] if spec["alpha"] else 1.0)
    return model, paths, weights, scales


def test_stack_equals_the_sum_of_the_plain_loras(three):
    model, paths, weights, scales = three
    lora_weights = [1.0, 0.5, -1.5]
    sources = [LoRASource(p, weight=w) for p, w in zip(paths, lora_weights, strict=True)]
    stack = stack_set(import_loras(sources, model), name="stack")
    assert stack.num_experts == 1 and stack.name == "stack"
    assert stack.imported_from["composition"] == "stack"
    assert stack.experts["blocks.0.self_attn.q_proj"].down.shape[2] == 2 + 4
    assert stack.experts["blocks.0.mlp.layer1"].down.shape[2] == 2 + 4 + 3
    ref = Toy().eval()
    m = Toy().eval()
    compose(m, [stack], depth=1, mix="stack")
    gen = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for name in LAYERS:
            layer = m.get_submodule(name)
            x = torch.randn(2, 5, layer.in_features, generator=gen, dtype=torch.float32)
            want = ref.get_submodule(name)(x)
            for w, s, lw in zip(weights, scales, lora_weights, strict=True):
                if name in w:
                    a, b = w[name]
                    want = want + lw * s * (x @ a.t() @ b.t())
            torch.testing.assert_close(layer(x), want, rtol=1e-6, atol=1e-6)


def _equal_keys(adapter_set):
    experts = {}
    for t, ex in adapter_set.experts.items():
        experts[t] = dataclasses.replace(
            ex,
            keys_in=ex.keys_in[:1].expand_as(ex.keys_in).clone(),
            keys_out=ex.keys_out[:1].expand_as(ex.keys_out).clone(),
        )
    return dataclasses.replace(adapter_set, experts=experts)


def test_routed_with_equal_keys_equals_stack_at_depth_1(three):
    model, paths, _, _ = three
    n = len(paths)
    routed = _equal_keys(import_loras(paths, model).adapter_set)
    stack_mean = stack_set(import_loras([LoRASource(p, weight=1.0 / n) for p in paths], model))
    stack_full = stack_set(import_loras(paths, model))
    x = torch.randn(2, 6, DIM, generator=torch.Generator().manual_seed(3))
    outs = []
    for s in (routed, stack_mean, stack_full):
        m = Toy().eval()
        compose(m, [s], depth=1, mix="stack")
        with torch.no_grad():
            outs.append(m.blocks[0].mlp.layer1(x) - Toy().eval().blocks[0].mlp.layer1(x))
    torch.testing.assert_close(outs[0], outs[1], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(n * outs[0], outs[2], rtol=1e-5, atol=1e-6)


def test_one_lora_routed_equals_stack(tmp_path):
    model = Toy().eval()
    p = tmp_path / "one.safetensors"
    _lora(model, p, rank=3, seed=5, alpha=2.0)
    result = import_loras([str(p)], model)
    x = torch.randn(1, 4, DIM, generator=torch.Generator().manual_seed(1))
    outs = []
    for s in (result.adapter_set, stack_set(result)):
        m = Toy().eval()
        compose(m, [s], depth=1, mix="stack")
        with torch.no_grad():
            outs.append(m(x))
    torch.testing.assert_close(outs[0], outs[1], rtol=1e-5, atol=1e-6)


def _args(loras, out, **kw):
    base = dict(
        loras=[str(p) for p in loras],
        model="toy",
        output=str(out),
        family=None,
        mode="routed",
        weights=None,
        names=None,
        source_info=None,
        id=None,
        license=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


INFO = [
    {"repo": "alice/ink", "revision": "abc123", "license": "CC-BY-4.0"},
    {"repo": "bob/parchment", "revision": "def456", "license": "CC-BY-4.0"},
    {"repo": "carol/glow", "revision": "0f0f0f", "license": "CC-BY-4.0"},
]


@pytest.mark.parametrize("mode", ["routed", "stack"])
def test_cli_writes_sources_and_composition(three, tmp_path, capsys, mode):
    model, paths, _, _ = three
    info = tmp_path / "info.json"
    info.write_text(json.dumps(INFO))
    out = tmp_path / f"mix-{mode}.kmod"
    args = _args(
        paths,
        out,
        mode=mode,
        weights=[1.0, 0.5, 2.0],
        names=["ink", "parchment", "glow"],
        source_info=str(info),
    )
    assert _import(args, model) == 0
    printed = capsys.readouterr().out
    assert "Expert 0: ink" in printed and "Expert 2: glow" in printed
    assert f"Composition: {mode}." in printed
    manifest = read_manifest(out)
    assert manifest["composition"] == mode
    assert manifest["license"] == "CC-BY-4.0"
    assert manifest["id"] == f"mix-{mode}"
    records = manifest["imported_sources"]
    assert [r["name"] for r in records] == ["ink", "parchment", "glow"]
    assert [r["file"] for r in records] == [f"lora{i}.safetensors" for i in range(3)]
    for r, i in zip(records, INFO, strict=True):
        assert (r["repo"], r["revision"], r["license"]) == (i["repo"], i["revision"], i["license"])
    assert [r["weight"] for r in records] == [1.0, 0.5, 2.0]
    (s,) = load_mod(out).sets
    assert s.num_experts == (3 if mode == "routed" else 1)
    assert s.imported_from["sources"] == records


def test_cli_stack_mod_equals_the_stack_set(three, tmp_path):
    model, paths, _, _ = three
    out = tmp_path / "stack.kmod"
    assert _import(_args(paths, out, mode="stack"), model) == 0
    x = torch.randn(1, 3, DIM, generator=torch.Generator().manual_seed(2))
    outs = []
    for src in (str(out), stack_set(import_loras(paths, model))):
        m = Toy().eval()
        adapter = compose(m, [src], depth=1, mix="stack")
        adapter.settings.gate_mode = "none"
        with torch.no_grad():
            outs.append(m(x))
    torch.testing.assert_close(outs[0], outs[1], rtol=1e-5, atol=1e-6)


def test_license_rule():
    same = [{"license": "MIT"}, {"license": "MIT"}]
    mixed = [{"license": "MIT"}, {"license": "CC-BY-4.0"}]
    assert import_license(same) == "MIT"
    assert import_license(mixed) == "NOASSERTION"
    assert import_license(mixed, "Apache-2.0") == "Apache-2.0"
    assert import_license(same, "Apache-2.0") == "Apache-2.0"
    assert import_license([{"license": None}, {"license": "MIT"}]) == "NOASSERTION"


def test_cli_license_from_files_and_flag(tmp_path, capsys):
    model = Toy().eval()
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    _lora(model, a, rank=2, seed=1, license="MIT")
    _lora(model, b, rank=2, seed=2, license="MIT")
    assert _import(_args([a, b], tmp_path / "same.kmod"), model) == 0
    assert read_manifest(tmp_path / "same.kmod")["license"] == "MIT"
    _lora(model, b, rank=2, seed=2, license="CC-BY-4.0")
    assert _import(_args([a, b], tmp_path / "mixed.kmod"), model) == 0
    assert read_manifest(tmp_path / "mixed.kmod")["license"] == "NOASSERTION"
    args = _args([a, b], tmp_path / "given.kmod", license="Apache-2.0")
    assert _import(args, model) == 0
    assert read_manifest(tmp_path / "given.kmod")["license"] == "Apache-2.0"
    c = tmp_path / "c.safetensors"
    _lora(model, c, rank=2, seed=3)
    assert _import(_args([a, c], tmp_path / "none.kmod"), model) == 0
    assert read_manifest(tmp_path / "none.kmod")["license"] == "NOASSERTION"


def test_cli_rejects_wrong_counts(three, tmp_path, capsys):
    model, paths, _, _ = three
    assert _import(_args(paths, tmp_path / "x.kmod", weights=[1.0]), model) == 1
    assert "--weights needs 3 values" in capsys.readouterr().err
    assert _import(_args(paths, tmp_path / "x.kmod", names=["a", "b"]), model) == 1
    assert "--names needs 3 values" in capsys.readouterr().err
    info = tmp_path / "info.json"
    info.write_text(json.dumps(INFO[:2]))
    assert _import(_args(paths, tmp_path / "x.kmod", source_info=str(info)), model) == 1
    assert "--source-info" in capsys.readouterr().err


def test_cli_parser_has_the_import_command(monkeypatch):
    import kilora.cli as cli

    seen = {}
    monkeypatch.setattr(cli, "_import", lambda args: seen.setdefault("args", args) and 0)
    argv = [
        "import",
        "a.safetensors",
        "b.safetensors",
        "--model",
        "m",
        "--output",
        "o.kmod",
        "--mode",
        "stack",
        "--weights",
        "1",
        "0.5",
        "--names",
        "a",
        "b",
    ]
    cli.main(argv)
    args = seen["args"]
    assert args.loras == ["a.safetensors", "b.safetensors"] and args.mode == "stack"
    assert args.weights == [1.0, 0.5] and args.names == ["a", "b"]


def test_cli_builds_several_mods_from_one_import(three, tmp_path):
    from kilora import load_mod

    model, paths, _, _ = three
    args = _args(paths, tmp_path / "routed.kmod", mode="routed")
    args.build = [f"stack:1/6={tmp_path / 'sixth.kmod'}", f"stack={tmp_path / 'full.kmod'}"]
    assert _import(args, model) == 0
    routed, sixth, full = (
        load_mod(tmp_path / n) for n in ("routed.kmod", "sixth.kmod", "full.kmod")
    )
    assert routed.manifest["composition"] == "routed" and sixth.manifest["composition"] == "stack"
    t = next(iter(full.sets[0].experts))
    torch.testing.assert_close(
        sixth.sets[0].experts[t].up, full.sets[0].experts[t].up / 6, rtol=1e-6, atol=1e-7
    )


def test_two_default_imports_compose(three, tmp_path):
    from kilora.errors import KiLoRAError
    from kilora.multi_import import import_set_name

    model, paths, _, _ = three
    a, b = tmp_path / "a.kmod", tmp_path / "b.kmod"
    assert _import(_args(paths[:1], a), model) == 0
    assert _import(_args(paths[1:2], b), model) == 0
    names = [load_mod(p).sets[0].name for p in (a, b)]
    assert names[0] != names[1] and "imported" not in names
    m = Toy().eval()
    adapter = compose(m, [str(a), str(b)], depth=1)
    assert adapter.set_names == names
    adapter.remove()
    with pytest.raises(KiLoRAError, match="--names"):
        compose(Toy().eval(), [str(a), str(a)], depth=1)
    assert import_set_name(["my lora", "my lora", "x"]) == "my-lora_my-lora-2_x"
    assert import_set_name([]) == "imported"


def test_cli_names_set_the_set_name_and_ids_are_lowercased(three, tmp_path, capsys):
    model, paths, _, _ = three
    out = tmp_path / "a.kmod"
    assert _import(_args(paths[:1], out, names=["regA"], id="RegionA"), model) == 0
    mod = load_mod(out)
    assert mod.sets[0].name == "regA" and mod.manifest["id"] == "regiona"
    assert _import(_args(paths[:1], tmp_path / "b.kmod", id="-bad id"), model) == 1
    assert "for example region-a" in capsys.readouterr().err
