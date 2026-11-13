# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
from typing import Any

from .modfile import read_manifest

BASE_MODEL_REPOS = {"z-image": "Tongyi-MAI/Z-Image"}

HF_LICENSES = {
    "apache-2.0": "apache-2.0",
    "mit": "mit",
    "cc0-1.0": "cc0-1.0",
    "cc-by-4.0": "cc-by-4.0",
    "cc-by-sa-4.0": "cc-by-sa-4.0",
    "cc-by-nc-4.0": "cc-by-nc-4.0",
    "cc-by-nc-sa-4.0": "cc-by-nc-sa-4.0",
    "openrail++": "openrail++",
}


def _yaml(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def model_card(path: str | os.PathLike[str], repo_file: str | None = None) -> str:
    m = read_manifest(path)
    family = (m.get("base_model") or {}).get("family", "unknown")
    checkpoints = [c for c in (m.get("base_model") or {}).get("checkpoints", []) if "/" in c]
    base_repo = checkpoints[0] if checkpoints else BASE_MODEL_REPOS.get(family)
    spdx = str(m.get("license", "NOASSERTION"))
    hf_license = HF_LICENSES.get(spdx.lower(), "other")
    tags = ["kilora", "lora", family]
    if m.get("mod_family"):
        tags.append(str(m["mod_family"]))
    if m.get("live"):
        tags.append("live-mod")
    tags.append("text-to-image")
    words = list(m.get("trigger_words") or [])
    strength = (m.get("strength") or {}).get("recommended", 1.0)
    depth = (m.get("depth") or {}).get("recommended", 1)
    trained = (m.get("depth") or {}).get("trained")
    file_name = repo_file or os.path.basename(os.fspath(path))
    name = str(m.get("name", m["id"]))

    meta = [
        "---",
        "library_name: kilora",
        f"pipeline_tag: {_yaml('text-to-image')}",
    ]
    if base_repo:
        meta.append(f"base_model: {_yaml(base_repo)}")
    meta.append(f"license: {_yaml(hf_license)}")
    if hf_license == "other":
        meta.append(f"license_name: {_yaml(spdx)}")
    meta.append("tags:")
    meta += [f"  - {_yaml(t)}" for t in tags]
    if words:
        meta.append(f"instance_prompt: {_yaml(', '.join(words))}")
    meta.append("---")

    body = [f"# {name}", ""]
    if m.get("description"):
        body += [str(m["description"]), ""]
    body += [
        "This is a KiLoRA mod. It needs a KiLoRA node in ComfyUI or the `kilora` Python "
        "package. Plain LoRA loaders cannot use it.",
        "",
        "| | |",
        "| --- | --- |",
        f"| File | `{file_name}` |",
        f"| Base model | {family}" + (f" (`{base_repo}`)" if base_repo else "") + " |",
        f"| Recommended strength | {strength:g} |",
        f"| Recommended iteration depth | {depth}"
        + (f" (trained {trained[0]} to {trained[1]})" if trained else "")
        + " |",
    ]
    if words:
        body.append(f"| Trigger words | {', '.join(f'`{w}`' for w in words)} |")
    if m.get("mod_family"):
        body.append(f"| Mod family | {m['mod_family']} |")
    hooks = card_fields(m)["hooks"]
    body.append(f"| Hooks | {hooks['count']} layers: {', '.join(hooks['patterns'])} |")
    body.append(f"| Training data | {source_line(m.get('data_source'))} |")
    body.append(f"| License | {spdx} |")
    if m.get("trained_with"):
        tw = m["trained_with"]
        body.append(f"| Trained with | {tw.get('tool', '')} {tw.get('version', '')} |")
    body.append("")
    if m.get("live"):
        body += [
            "## Live mod",
            "",
            "This mod can keep learning from the images that you pick, with save points and "
            "undo. Use the KiLoRA Mod Editor and KiLoRA Live Save nodes.",
            "",
        ]
    if m.get("license_notice"):
        body += ["## License notice", "", str(m["license_notice"]), ""]
    imported = [s for s in m.get("adapter", {}).get("sets", []) if s.get("imported_from")]
    if imported:
        body += ["## Imported weights", ""]
        for s in imported:
            src = s["imported_from"]
            body.append(
                f"- `{s['name']}` contains weights from `{src.get('original_name', '')}` "
                f"(license: {src.get('original_license', 'unknown')})."
            )
        body.append("")
    prompts = list(m.get("example_prompts") or [])
    body += [
        "## Use in ComfyUI",
        "",
        "1. Install the KiLoRA nodes (search for KiLoRA in ComfyUI Manager).",
        f"2. Put `{file_name}` into `ComfyUI/models/kilora/`.",
        f"3. Add KiLoRA Select (strength {strength:g}) and KiLoRA Apply, and connect the "
        "model output of KiLoRA Apply to your sampler.",
        "",
        "## Use in Python",
        "",
        "```python",
        "from kilora import compose",
        "",
        f'adapter = compose(pipe.transformer, ["{file_name}"], strengths=[{strength:g}], '
        f"depth={depth})",
        "```",
        "",
    ]
    if prompts:
        body += ["## Example prompts", ""] + [f"- {p}" for p in prompts] + [""]
    return "\n".join(meta + [""] + body)


CARD_KEY = "kilora.card"


def _hook_patterns(targets: list[str]) -> list[str]:
    import re

    out: list[str] = []
    for t in targets:
        pattern = re.sub(r"\.\d+\.", ".*.", t, count=1)
        if pattern not in out:
            out.append(pattern)
    return out


def source_line(source: dict[str, Any] | None) -> str:
    if not source:
        return "The training data is not described."
    parts = [f"{source.get('images', '?')} images from {source.get('name', 'a folder')}"]
    if source.get("generators"):
        parts.append(f"made with {', '.join(source['generators'])}")
    if source.get("license"):
        parts.append(f"data license {source['license']}")
    line = ", ".join(parts) + "."
    if source.get("note"):
        line += f" {source['note']}"
    return line


def card_fields(manifest: dict[str, Any]) -> dict[str, Any]:
    m = manifest
    base = m.get("base_model") or {}
    strength = m.get("strength") or {}
    targets = list((m.get("adapter") or {}).get("target_modules", []))
    source = m.get("data_source")
    return {
        "id": m.get("id"),
        "name": m.get("name", m.get("id")),
        "version": m.get("version"),
        "description": m.get("description", ""),
        "trigger": list(m.get("trigger_words") or []),
        "example_prompts": list(m.get("example_prompts") or []),
        "strength": {
            "recommended": strength.get("recommended", 1.0),
            "min": strength.get("min", -2.0),
            "max": strength.get("max", 2.0),
        },
        "depth": (m.get("depth") or {}).get("recommended", 1),
        "hooks": {"count": len(targets), "patterns": _hook_patterns(targets)},
        "base_model": {
            "family": base.get("family", "unknown"),
            "checkpoints": list(base.get("checkpoints", [])),
        },
        "license": m.get("license", "NOASSERTION"),
        "data_source": source,
        "source_line": source_line(source),
        "kind": "live" if m.get("live") else "static",
        "mod_family": m.get("mod_family"),
        "author": m.get("author", ""),
        "learned_with": list(m.get("learned_with") or []),
        "license_notice": m.get("license_notice", ""),
    }


def card_markdown(fields: dict[str, Any]) -> str:
    f = fields
    s = f["strength"]
    rows = [
        f"# {f['name']}",
        "",
        *([f["description"], ""] if f["description"] else []),
        "| | |",
        "| --- | --- |",
        f"| ID | `{f['id']}` {f['version'] or ''} |",
        f"| Kind | {f['kind']} mod |",
        f"| Trigger | {', '.join(f'`{t}`' for t in f['trigger']) or 'none'} |",
        f"| Recommended strength | {s['recommended']:g} (range {s['min']:g} to {s['max']:g}) |",
        f"| Depth | {f['depth']} |",
        f"| Hooks | {f['hooks']['count']} layers: {', '.join(f['hooks']['patterns'])} |",
        f"| Base model | {f['base_model']['family']} |",
        f"| License | {f['license']} |",
        f"| Training data | {f['source_line']} |",
    ]
    if f.get("learned_with"):
        rows.append(
            "| Learned together with | "
            + ", ".join(
                f"{x['id']} (strength {x['strength']:g}, depth {x['depth']})"
                for x in f["learned_with"]
            )
            + " |"
        )
    if f["example_prompts"]:
        rows += ["", "## Example prompts", ""] + [f"- {p}" for p in f["example_prompts"]]
    if f.get("license_notice"):
        rows += ["", "## License notice", "", f["license_notice"]]
    return "\n".join(rows) + "\n"


def embed_card(path: str | os.PathLike[str]) -> dict[str, Any]:
    from safetensors import safe_open

    from .modfile import write_safetensors

    fields = card_fields(read_manifest(path))
    with safe_open(os.fspath(path), framework="pt") as f:
        metadata = dict(f.metadata() or {})
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    metadata.pop("modelspec.hash_sha256", None)
    metadata[CARD_KEY] = json.dumps(fields, sort_keys=True, ensure_ascii=False)
    write_safetensors(path, tensors, metadata)
    return fields


def read_card(path: str | os.PathLike[str]) -> dict[str, Any]:
    from safetensors import safe_open

    with safe_open(os.fspath(path), framework="pt") as f:
        raw = (f.metadata() or {}).get(CARD_KEY)
    return json.loads(raw) if raw else card_fields(read_manifest(path))
