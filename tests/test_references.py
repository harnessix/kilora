# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import os

import pytest

from kilora import references


def test_index_reuses_hash_across_restart_and_rehashes_change(tmp_path, monkeypatch):
    file = tmp_path / "weights.safetensors"
    file.write_bytes(b"first")
    index_path = tmp_path / "index.json"
    calls = []
    original = references._sha256

    def count(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(references, "_sha256", count)
    expected = hashlib.sha256(b"first").hexdigest()
    assert references.HashIndex(index_path).hash(file) == expected
    assert references.HashIndex(index_path).hash(file) == expected
    assert len(calls) == 1
    stat = file.stat()
    os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert references.HashIndex(index_path).hash(file) == expected
    assert len(calls) == 2
    file.write_bytes(b"second")
    assert references.HashIndex(index_path).hash(file) == hashlib.sha256(b"second").hexdigest()
    assert len(calls) == 3


def test_resolve_and_reject_changed_content(tmp_path, monkeypatch):
    folder = tmp_path / "loras"
    folder.mkdir()
    file = folder / "weights.safetensors"
    file.write_bytes(b"original")
    monkeypatch.setenv("KILORA_HASH_INDEX", str(tmp_path / "index.json"))
    meta = {"sha256": hashlib.sha256(b"original").hexdigest()}
    assert references.resolve(meta, [folder]) == file
    assert references.resolve(meta, [folder]) == file
    file.write_bytes(b"different content")
    with pytest.raises(references.MissingReferenceError):
        references.resolve(meta, [folder])


def test_missing_reference_message(tmp_path, monkeypatch):
    monkeypatch.setenv("KILORA_HASH_INDEX", str(tmp_path / "index.json"))
    meta = {
        "sha256": "a" * 64,
        "filename": "style.safetensors",
        "size": 115 * 1024 * 1024,
        "source_url": "https://example.org/style",
    }
    with pytest.raises(references.MissingReferenceError) as exc:
        references.resolve(meta, [tmp_path / "missing"])
    message = str(exc.value)
    for hint in ("style.safetensors", "115 MB", "aaaaaaaaaaaa", "https://example.org/style"):
        assert hint in message


def test_default_index_path_uses_comfy_user_directory(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from kilora import host

    monkeypatch.delenv("KILORA_HASH_INDEX", raising=False)
    monkeypatch.setattr(
        host,
        "SERVICES",
        host.HostServices(
            comfy=SimpleNamespace(
                folder_paths=SimpleNamespace(get_user_directory=lambda: str(tmp_path))
            )
        ),
    )
    assert references.HashIndex().path == tmp_path / "kilora" / "hash_index.json"


def test_resolve_rechecks_cached_hash_even_when_metadata_matches(tmp_path, monkeypatch):
    file = tmp_path / "weights.kmod"
    file.write_bytes(b"original")
    monkeypatch.setenv("KILORA_HASH_INDEX", str(tmp_path / "index.json"))
    wanted = hashlib.sha256(b"original").hexdigest()
    references.HashIndex().hash(file)
    stat = file.stat()
    file.write_bytes(b"modified")
    os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(references.KiLoRAError, match="Hash mismatch") as exc:
        references.resolve({"sha256": wanted}, [tmp_path])
    assert wanted in str(exc.value)
    assert hashlib.sha256(b"modified").hexdigest() in str(exc.value)


def test_resolve_skips_unreadable_and_prefers_filename(tmp_path, monkeypatch, caplog):
    bad = tmp_path / "a.kmod"
    bad.write_bytes(b"bad")
    good = tmp_path / "b.safetensors"
    good.write_bytes(b"good")
    other = tmp_path / "c.kmod"
    other.write_bytes(b"other")
    monkeypatch.setenv("KILORA_HASH_INDEX", str(tmp_path / "index.json"))
    original = references._sha256
    calls = []

    def read(path):
        calls.append(path.name)
        if path == bad:
            raise PermissionError("denied")
        return original(path)

    monkeypatch.setattr(references, "_sha256", read)
    wanted = hashlib.sha256(b"good").hexdigest()
    assert references.resolve({"sha256": wanted}, [tmp_path]) == good
    assert "denied" in caplog.text
    calls.clear()
    monkeypatch.setattr(references, "_VERIFIED", {})
    assert references.resolve({"sha256": wanted, "filename": good.name}, [tmp_path]) == good
    assert calls == [good.name]
