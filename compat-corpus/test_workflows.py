# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

from kilora.comfyui import NODE_CLASS_MAPPINGS

ROOT = Path(__file__).parent
WORKFLOWS = sorted(ROOT.glob("*/workflows/*.json"))


@pytest.mark.old_mod_corpus
@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: f"{p.parent.parent.name}/{p.name}")
def test_workflow_nodes_and_inputs_exist(path, monkeypatch):
    from kilora.comfyui import nodes

    monkeypatch.setattr(nodes.FOLDERS, "list", lambda folder, extension=None: [])
    graph = json.loads(path.read_text())
    for node_id, node in graph.items():
        cls = NODE_CLASS_MAPPINGS.get(node["class_type"])
        if cls is None:
            assert not node["class_type"].startswith("KiLoRA"), f"{node_id}: node was removed"
            continue
        spec = cls.INPUT_TYPES()
        known = {**spec.get("required", {}), **spec.get("optional", {})}
        for name in node["inputs"]:
            assert name in known, f"{node_id} {node['class_type']}: input {name!r} was removed"
        for name in spec.get("required", {}):
            assert name in node["inputs"], f"{node_id}: new required input {name!r}"
