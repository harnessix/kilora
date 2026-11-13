# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path


def direction_metadata(metadata):
    if "kilora_directions" not in metadata:
        return None
    record = json.loads(metadata["kilora_directions"])
    if not isinstance(record, dict):
        raise ValueError("kilora_directions must contain a JSON object")
    return record


def direction_source(record, path):
    source = record.get("source_file") or record.get("source")
    if source:
        return {"label": str(source), "source_file": source}
    pairs = record.get("pairs", [])
    if pairs:
        labels = []
        for pair in pairs:
            if not isinstance(pair, dict):
                raise ValueError("direction prompt pairs must be JSON objects")
            student = pair.get("student", pair.get("plain", ""))
            target = pair.get("target", pair.get("style", ""))
            labels.append(f"{student} → {target}")
        return {"label": "; ".join(labels), "pairs": pairs}
    return {"label": Path(path).name}
