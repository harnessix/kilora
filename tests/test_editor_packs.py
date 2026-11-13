# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

from kilora import editor_packs as ep
from kilora.errors import ModFormatError


def manifest(**over):
    m = {
        "id": "common",
        "version": "1.0.0",
        "role": "common",
        "editor": {"kind": "example-editor", "version": ">=1.0, <2"},
        "base_models": [{"family": "z-image"}],
        "storage": "int8-block",
        "license": "CC-BY-4.0",
        "training": {"state": "trained"},
    }
    m.update(over)
    return m


def family(pid="ink", version="1.0.0", families=("ink",), requires=None):
    return manifest(
        id=pid,
        version=version,
        role="family",
        mod_families=list(families),
        requires=requires or {"common": ">=1.0, <2"},
    )


def write(tmp_path, m, seed=0):
    gen = torch.Generator().manual_seed(seed)
    tensors = {"directions": torch.randn(16, 3, 64, generator=gen)}
    path = tmp_path / f"{m['id']}-{m['version']}{ep.PACK_SUFFIX}"
    return path, ep.write_pack(path, m, tensors), tensors


def test_round_trip_with_int8_block_storage(tmp_path):
    path, m, tensors = write(tmp_path, manifest())
    read, out = ep.read_pack(path)
    assert read.digest == m.digest and read.ref == m.ref
    err = (out["directions"] - tensors["directions"]).abs().max()
    assert float(err) <= float(tensors["directions"].abs().max()) / 127 + 1e-3
    assert out["directions"].shape == tensors["directions"].shape
    assert path.stat().st_size < 16 * 3 * 64 * 2


@pytest.mark.parametrize("storage", ["float32", "bf16", "fp8-row", "int8-block", "nf4-block"])
def test_every_storage_format_reads_back(tmp_path, storage):
    path, _, tensors = write(tmp_path, manifest(id=f"s-{storage}", storage=storage))
    _, out = ep.read_pack(path)
    assert out["directions"].shape == tensors["directions"].shape
    assert torch.allclose(out["directions"], tensors["directions"], atol=0.5)


def test_same_content_gives_the_same_bytes(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a, _, _ = write(tmp_path / "a", manifest(storage="fp8-row"))
    b, _, _ = write(tmp_path / "b", manifest(storage="fp8-row"))
    assert a.read_bytes() == b.read_bytes()


def test_damaged_file_is_refused(tmp_path):
    path, _, _ = write(tmp_path, manifest())
    data = bytearray(path.read_bytes())
    data[-3] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ModFormatError, match="digest does not match"):
        ep.read_pack(path)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"format_version": "2.0"}, "not supported"),
        ({"id": "Bad ID"}, "not valid"),
        ({"version": "1.0"}, "X.Y.Z"),
        ({"role": "extra"}, "role"),
        ({"editor": {"kind": "x", "version": "soon"}}, "version range"),
        ({"base_models": []}, "base_models"),
        ({"storage": "int4"}, "storage"),
        ({"training": {"state": "untrained"}}, "not trained"),
        ({"mod_families": ["ink"]}, "names none"),
    ],
)
def test_manifest_problems_have_plain_messages(change, message):
    m = {
        "format": ep.PACK_FORMAT,
        "format_version": "1.0",
        **manifest(),
        "digest": "sha256-" + "0" * 64,
        **change,
    }
    with pytest.raises(ModFormatError, match=message):
        ep.parse_manifest(m)


def test_family_pack_rules():
    base = {"format": ep.PACK_FORMAT, "format_version": "1.0", "digest": "sha256-" + "0" * 64}
    ok = ep.parse_manifest({**base, **family()})
    assert ok.role == "family" and ok.requires == {"common": ">=1.0, <2"}
    with pytest.raises(ModFormatError, match="names its mod families"):
        ep.parse_manifest({**base, **family(families=())})
    with pytest.raises(ModFormatError, match="requires its common pack"):
        ep.parse_manifest({**base, **manifest(id="ink", role="family", mod_families=["ink"])})


def test_choose_newest_compatible_packs(tmp_path):
    packs = [
        write(tmp_path, manifest(version="1.0.0"))[1],
        write(tmp_path, manifest(version="1.2.0"))[1],
        write(
            tmp_path, manifest(version="2.0.0", editor={"kind": "example-editor", "version": ">=2"})
        )[1],
        write(tmp_path, family(version="1.0.0"))[1],
        write(tmp_path, family(version="1.1.0", requires={"common": ">=1.3"}))[1],
        write(tmp_path, family(pid="parchment", families=("texture",)))[1],
    ]
    chosen = ep.choose_packs(packs, "example-editor", "1.4.0", "z-image", "ink")
    assert [(p.id, p.version) for p in chosen] == [("common", "1.2.0"), ("ink", "1.0.0")]
    assert (
        ep.choose_packs(packs, "example-editor", "1.4.0", "z-image", "texture")[1].id == "parchment"
    )
    assert len(ep.choose_packs(packs, "example-editor", "1.4.0", "z-image", "photo")) == 1
    assert ep.choose_packs(packs, "other-editor", "1.0.0", "z-image", "ink") == []
    assert ep.choose_packs(packs, "example-editor", "1.4.0", "sdxl", "ink") == []
    pinned = ep.choose_packs(
        packs, "example-editor", "1.4.0", "z-image", "ink", pins={"common": "1.0.0"}
    )
    assert pinned[0].version == "1.0.0"


def test_save_record_and_warnings(tmp_path):
    common = write(tmp_path, manifest())[1]
    ink = write(tmp_path, family())[1]
    record = ep.save_record([common, ink])
    assert record == [common.ref.to_manifest(), ink.ref.to_manifest()]
    assert json.loads(json.dumps(record)) == record
    assert ep.check_save_record(record, [common, ink]) == []
    newer = write(tmp_path, manifest(version="1.1.0"), seed=1)[1]
    warnings = ep.check_save_record(record, [newer])
    assert any("Now 1.1.0 is in use" in w for w in warnings)
    assert any("ink, which is not in use" in w for w in warnings)
    assert ep.check_save_record([], [common]) == [
        "The pack common is in use, but the save was made without it."
    ]


def test_index_entry_checks(tmp_path):
    path, m, _ = write(tmp_path, manifest())
    entry = {
        "kind": "editor-pack",
        "version": m.version,
        "role": m.role,
        "url": f"files/{path.name}",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "digest": m.digest,
        "size": path.stat().st_size,
    }
    assert ep.check_index_entry("common", entry) == []
    bad = {**entry, "kind": "mod", "size": 0, "url": "files/x.bin", "sha256": "zz"}
    assert len(ep.check_index_entry("common", bad)) == 4


def test_non_pack_files_are_refused(tmp_path):
    path = tmp_path / "x.kedit-pack.safetensors"
    save_file({"x": torch.zeros(1)}, str(path), metadata={"kilora.manifest": "{}"})
    with pytest.raises(ModFormatError, match="not an editor pack"):
        ep.read_pack(path)


def _three(tmp_path):
    return [
        write(tmp_path, manifest(version="1.2.0"))[1],
        write(tmp_path, family(pid="ink", families=("ink_line",)))[1],
        write(tmp_path, family(pid="ink", version="1.1.0", families=("ink_line",)))[1],
        write(tmp_path, family(pid="parchment", families=("texture",)))[1],
        write(tmp_path, family(pid="pixel", families=("pixel",), requires={"common": ">=9"}))[1],
    ]


def _select(packs, **kw):
    kw.setdefault("mode", "auto")
    kw.setdefault("disabled", ())
    kw.setdefault("base_family", "z-image")
    kw.setdefault("mod_family", None)
    return ep.select_packs(packs, editor_version="1.4.0", **kw)


def test_auto_without_a_family_pack_for_the_mod_takes_every_family_pack(tmp_path):
    packs = _three(tmp_path)
    for mod_family in (None, "vroid"):
        chosen, _ = _select(packs, mod_family=mod_family)
        assert [(p.id, p.version) for p in chosen] == [
            ("common", "1.2.0"),
            ("ink", "1.1.0"),
            ("parchment", "1.0.0"),
        ]
    chosen, _ = _select(packs, mod_family="texture")
    assert [p.id for p in chosen] == ["common", "parchment"]


def test_named_families_next_to_the_common_pack(tmp_path):
    packs = _three(tmp_path)
    chosen, notes = _select(packs, mod_family="texture", families=["ink_line"])
    assert [p.id for p in chosen] == ["common", "ink"]
    chosen, notes = _select(packs, families=["ink_line", "watercolor"])
    assert [p.id for p in chosen] == ["common", "ink"]
    assert any("watercolor" in n for n in notes)
    chosen, _ = _select(packs, mode="common only", families=["ink_line"])
    assert [p.id for p in chosen] == ["common"]
    chosen, _ = _select(packs, families=["ink_line", "texture"], disabled=["parchment"])
    assert [p.id for p in chosen] == ["common", "ink"]
