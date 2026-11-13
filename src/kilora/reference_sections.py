# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import math
import re


def _keys(value, required, optional=()):
    if not isinstance(value, dict):
        raise ValueError("metadata must be an object")
    for key in required:
        if key not in value:
            raise ValueError(f"{key} is required")
    for key in value:
        if key not in required and key not in optional:
            raise ValueError(f"unknown key {key!r}; allowed: {sorted((*required, *optional))}")


def _choice(key, value, allowed):
    if value not in allowed:
        raise ValueError(f"{key} must be one of {allowed}, got {value!r}")


def _number(key, value, low, high, integer=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, int if integer else (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(
            f"{key} must be {'an integer' if integer else 'a number'} in [{low}, {high}]"
        )


def validate(kind, meta, tensors):
    if tensors:
        raise ValueError(f"{kind} tensors must be empty")
    if kind == "weight-ref":
        from .compute import RECORD_AXES
        from .roles import POLICIES
        from .timeline import Row, check_row

        _keys(meta, ("sha256", "form", "transforms"), ("size", "filename", "source_url"))
        if not isinstance(meta["sha256"], str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", meta["sha256"]
        ):
            raise ValueError("sha256 must be 64 hex digits")
        if "size" in meta:
            _number("size", meta["size"], 0, 2**63 - 1, integer=True)
        for key in ("filename", "source_url"):
            if key in meta and not isinstance(meta[key], str):
                raise ValueError(f"{key} must be a string")
        _choice("form", meta["form"], ("lora", "mod"))
        t = meta["transforms"]
        _keys(t, ("strength", "axis", "roles", "region", "timeline", "enabled"), ("start", "end"))
        from .entry_window import check_window

        check_window(t.get("start", 0.0), t.get("end", 1.0))
        _number("strength", t["strength"], -2, 2)
        _choice("axis", t["axis"], ("auto", *RECORD_AXES))
        _choice("roles", t["roles"], POLICIES)
        if t["region"] is not None:
            raise ValueError("region must be null; region names are not supported in 1.0")
        if not isinstance(t["enabled"], bool):
            raise ValueError("enabled must be a boolean")
        if not isinstance(t["timeline"], list):
            raise ValueError("timeline must be a list")
        for row in t["timeline"]:
            try:
                _keys(row, ("target", "quantity", "curve", "clock", "interpolation"))
                if not all(isinstance(v, str) for v in row.values()):
                    raise ValueError("row values must be strings")
                check_row(Row(**row))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"timeline: {exc}") from None
    else:
        from .adaptive_gate import GATE_MODES
        from .comfyui.settings import CROSS_SETTINGS, DEFAULTS, ITERATIONS, MIXES
        from .config import ROUTING_SCOPES, ROUTING_SELECTS
        from .depth_schedule import parse_expression
        from .path_walks import PATH_SELECTS

        _keys(meta, ("version", "settings"))
        if type(meta["version"]) is not int or meta["version"] != 1:
            raise ValueError("version must be 1")
        s = meta["settings"]
        _keys(
            s,
            (
                *DEFAULTS,
                "axis_alpha",
                "library",
                "output_gate",
                "gate_kappa",
                "path_width",
                "path_select",
                "route_freeze",
            ),
        )
        if "library" in s and not isinstance(s["library"], bool):
            raise ValueError("library must be a boolean")
        if "axis_alpha" in s:
            _choice("axis_alpha", s["axis_alpha"], ("off", "auto"))
        if "import_score" in s:
            from .comfyui.settings import IMPORT_SCORES

            _choice("import_score", s["import_score"], (None, *IMPORT_SCORES))
        if "lycoris_apply" in s:
            _choice("lycoris_apply", s["lycoris_apply"], ("truncated", "exact"))
        if "candidate_summary" in s:
            _choice("candidate_summary", s["candidate_summary"], ("keys", "read_map"))
        if "candidates" in s:
            _number("candidates", s["candidates"], 0, 4096, integer=True)
        for key in CROSS_SETTINGS:
            if key in s:
                _number(key, s[key], -16, 16, False)
        for key, allowed in (
            ("mix", tuple(MIXES)),
            ("iteration", (None, *ITERATIONS)),
            ("routing_scope", (None, *ROUTING_SCOPES)),
            ("routing_select", (None, *ROUTING_SELECTS)),
            ("output_gate", ("mod", *GATE_MODES)),
            ("path_select", ("mod", *PATH_SELECTS)),
        ):
            _choice(key, s[key], allowed)
        for key, low, high, integer in (
            ("depth", -1, 16, True),
            ("gate_kappa", 0, 16, False),
            ("path_width", 0, 8, True),
            ("route_freeze", 0, 1, False),
        ):
            _number(key, s[key], low, high, integer)
        schedule = s["depth_schedule"]
        if schedule is not None:
            if not isinstance(schedule, str):
                raise ValueError("depth_schedule must be null or an expression string")
            try:
                parse_expression(schedule)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"depth_schedule: {exc}") from None
