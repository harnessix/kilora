# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest

import kilora
from kilora import host
from kilora.host import HostServices, PackageExports


def test_register_returns_the_exports(monkeypatch):
    monkeypatch.setattr(host, "SERVICES", host.SERVICES)
    out = kilora.register(HostServices(api=1, host="comfyui", settings={"update": "release"}))
    assert isinstance(out, PackageExports) and out.api == host.EXPORTS_API
    assert "KiLoRAApply" in out.nodes and out.display_names["KiLoRAApply"]
    assert {e.kind for e in out.editors} >= {"path-write"}
    assert any(t.name == "static" for t in out.trainers)
    assert any(m.family == "z-image" for m in out.models)
    assert host.services().host == "comfyui"


def test_other_contract_version_is_refused():
    with pytest.raises(RuntimeError, match="host contract"):
        kilora.register(HostServices(api=99))


def test_editor_package_exports():
    import kilora_editor_delta

    out = kilora_editor_delta.register(HostServices())
    assert out.editors == [kilora_editor_delta.PLUGIN] and not out.nodes


def test_selftest_passes():
    kilora.selftest()
