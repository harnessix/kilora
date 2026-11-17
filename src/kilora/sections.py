# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .learned_cross import validate as _validate_learned_cross

__all__ = [
    "SECTION_KINDS",
    "SECTION_PREFIX",
    "DenseChange",
    "KronChange",
    "LayerChange",
    "LowRankChange",
    "ProgramChange",
    "Section",
    "SectionKind",
    "decode_sections",
    "encode_sections",
    "fixed_sections",
    "layer_changes",
    "register_section",
]

SECTION_PREFIX = "kilora/section/"


def _on_input(tensor: torch.Tensor, strength: float, x: torch.Tensor) -> torch.Tensor:

    if strength != 1.0:
        tensor = tensor * strength
    if tensor.device == x.device and tensor.dtype == x.dtype:
        return tensor
    return tensor.to(device=x.device, dtype=x.dtype)


class LayerChange:
    strength: float = 1.0

    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LowRankChange(LayerChange):
    def __init__(self, down: torch.Tensor, up: torch.Tensor, strength: float = 1.0) -> None:
        self.down, self.up, self.strength = down.detach(), up.detach(), float(strength)
        self._cast: dict = {}

    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        key = (x.device, x.dtype)
        if key not in self._cast:
            self._cast[key] = (
                _on_input(self.down, 1.0, x),
                _on_input(self.up, self.strength, x),
            )
        down, up = self._cast[key]
        from .fused import add_cast

        return add_cast(y, F.linear(F.linear(x, down), up))


class DenseChange(LayerChange):
    def __init__(self, delta: torch.Tensor, strength: float = 1.0) -> None:
        self.delta, self.strength = delta.detach(), float(strength)
        self._cast: dict = {}

    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        key = (x.device, x.dtype)
        if key not in self._cast:
            self._cast[key] = _on_input(self.delta, self.strength, x)
        return y + F.linear(x, self._cast[key]).to(y.dtype)


class KronChange(LayerChange):
    def __init__(self, w1: torch.Tensor, w2: torch.Tensor, strength: float = 1.0) -> None:
        self.w1, self.w2, self.strength = w1.detach(), w2.detach(), float(strength)
        self._cast: dict = {}

    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        key = (x.device, x.dtype)
        if key not in self._cast:
            self._cast[key] = (
                _on_input(self.w1, self.strength, x),
                _on_input(self.w2, 1.0, x),
            )
        w1, w2 = self._cast[key]
        lead = x.shape[:-1]
        xs = x.reshape(*lead, w1.shape[1], w2.shape[1])
        out = torch.matmul(torch.matmul(xs, w2.t()).transpose(-1, -2), w1.t()).transpose(-1, -2)
        return y + out.reshape(*lead, w1.shape[0] * w2.shape[0]).to(y.dtype)


@dataclass
class Section:
    kind: str
    name: str
    data: Any
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SectionKind:
    kind: str
    since: str
    encode: Any
    decode: Any
    changes: Any
    scale: Any

    @property
    def feature(self) -> str:
        return f"section.{self.kind}"


SECTION_KINDS: dict[str, SectionKind] = {}


def register_section(handler: SectionKind) -> None:
    from .modfile import FEATURES

    SECTION_KINDS[handler.kind] = handler
    FEATURES.setdefault(handler.feature, handler.since)


def _check_path(path: str) -> str:
    if "/" in path or not path:
        raise ValueError(f"a section layer path cannot be empty or contain '/': {path!r}")
    return path


def _lora_encode(data: Mapping[str, tuple[torch.Tensor, torch.Tensor]]) -> dict:
    out = {}
    for path, (down, up) in data.items():
        out[f"{_check_path(path)}/down"] = down.float()
        out[f"{path}/up"] = up.float()
    return out


def _lora_decode(tensors: Mapping[str, torch.Tensor]) -> dict:
    paths = sorted({k.rsplit("/", 1)[0] for k in tensors})
    return {p: (tensors[f"{p}/down"].float(), tensors[f"{p}/up"].float()) for p in paths}


def _delta_encode(data: Mapping[str, torch.Tensor]) -> dict:
    return {f"{_check_path(path)}/delta": d.float() for path, d in data.items()}


def _delta_decode(tensors: Mapping[str, torch.Tensor]) -> dict:
    return {k.rsplit("/", 1)[0]: v.float() for k, v in tensors.items()}


register_section(
    SectionKind(
        kind="fixed-lora",
        since="0.1.1",
        encode=_lora_encode,
        decode=_lora_decode,
        changes=lambda data, s: {p: [LowRankChange(d, u, s)] for p, (d, u) in data.items()},
        scale=lambda data, w: {p: (d, u * w) for p, (d, u) in data.items()},
    )
)


def _lokr_encode(data: Mapping[str, tuple[torch.Tensor, torch.Tensor]]) -> dict:
    out = {}
    for path, (w1, w2) in data.items():
        out[f"{_check_path(path)}/w1"] = w1.float()
        out[f"{path}/w2"] = w2.float()
    return out


def _lokr_decode(tensors: Mapping[str, torch.Tensor]) -> dict:
    paths = sorted({k.rsplit("/", 1)[0] for k in tensors})
    return {p: (tensors[f"{p}/w1"].float(), tensors[f"{p}/w2"].float()) for p in paths}


register_section(
    SectionKind(
        kind="fixed-lokr",
        since="0.1.1",
        encode=_lokr_encode,
        decode=_lokr_decode,
        changes=lambda data, s: {p: [KronChange(a, b, s)] for p, (a, b) in data.items()},
        scale=lambda data, w: {p: (a * w, b) for p, (a, b) in data.items()},
    )
)
register_section(
    SectionKind(
        kind="fixed-delta",
        since="0.1.1",
        encode=_delta_encode,
        decode=_delta_decode,
        changes=lambda data, s: {p: [DenseChange(d, s)] for p, d in data.items()},
        scale=lambda data, w: {p: d * w for p, d in data.items()},
    )
)


def _cross_data(data):
    return dict(_validate_learned_cross(data))


def _cross_changes(data, strength):
    return {}


register_section(
    SectionKind(
        "learned-cross",
        "1.0",
        _cross_data,
        _cross_data,
        _cross_changes,
        lambda data, weight: data,
    )
)


for _kind in ("weight-ref", "logic", "params"):
    register_section(
        SectionKind(
            _kind,
            "1.0",
            lambda data: data if data else {},
            lambda tensors: dict(tensors),
            lambda data, strength: {},
            lambda data, weight: data,
        )
    )


def _validate_record(kind, meta, tensors):
    if kind == "params" and tensors:
        raise ValueError("params: tensors are not allowed")
    if kind == "program":
        from .program import ProgramError, program_identity

        if meta.get("identity") != program_identity(meta):
            raise ProgramError("program identity mismatch")
        expected = {
            name for name, b in meta.get("bindings", {}).items() if b.get("kind") == "embedded"
        }
        if set(tensors) != expected:
            raise ProgramError("program embedded bindings do not match tensors")
    if kind in ("weight-ref", "logic"):
        from .reference_sections import validate

        validate(kind, meta, tensors)


def encode_sections(sections: list[Section]) -> tuple[dict[str, torch.Tensor], list[dict]]:
    tensors: dict[str, torch.Tensor] = {}
    records = []
    from .program import check_program_count

    check_program_count(sum(s.kind == "program" for s in sections))
    if sum(s.kind == "logic" for s in sections) > 1:
        raise ValueError("logic: at most one section is allowed")
    for i, s in enumerate(sections):
        handler = SECTION_KINDS.get(s.kind)
        if handler is None:
            raise ValueError(f"unknown section kind {s.kind!r}; known: {sorted(SECTION_KINDS)}")
        own = handler.encode(s.data)
        _validate_record(s.kind, s.meta, own)
        for name, value in own.items():
            tensors[f"{SECTION_PREFIX}{i}/{name}"] = value.detach().cpu().contiguous()
        records.append(
            {"kind": s.kind, "name": s.name, "meta": dict(s.meta), "tensors": sorted(own)}
        )
    return tensors, records


def decode_sections(records: list, tensors: Mapping[str, torch.Tensor]) -> list[Section]:
    out = []
    from .program import check_program_count

    check_program_count(sum(r.get("kind") == "program" for r in records))
    if sum(r.get("kind") == "logic" for r in records) > 1:
        raise ValueError("logic: at most one section is allowed")
    for i, r in enumerate(records):
        handler = SECTION_KINDS[r["kind"]]
        _validate_record(r["kind"], r.get("meta", {}), r["tensors"])
        own = {name: tensors[f"{SECTION_PREFIX}{i}/{name}"] for name in r["tensors"]}
        out.append(Section(r["kind"], str(r["name"]), handler.decode(own), dict(r.get("meta", {}))))
    return out


def scaled(section: Section, weight: float) -> Section:
    if section.kind == "program":
        meta = dict(section.meta)
        meta["section_strength"] = meta.get("section_strength", 1.0) * float(weight)
        return Section(section.kind, section.name, section.data, meta)
    data = SECTION_KINDS[section.kind].scale(section.data, float(weight))
    return Section(section.kind, section.name, data, dict(section.meta))


def layer_changes(sections: list[Section], strength: float) -> dict[str, list[LayerChange]]:
    out: dict[str, list[LayerChange]] = {}
    for s in sections:
        if s.kind == "program" and "part" in s.meta:
            continue
        if s.kind == "program":
            changes_by_path = {
                path: [
                    ProgramChange(s.meta, s.data, strength * s.meta.get("section_strength", 1.0))
                ]
                for path in s.meta["hooks"]
            }
        else:
            changes_by_path = SECTION_KINDS[s.kind].changes(s.data, float(strength))
        for path, changes in changes_by_path.items():
            for change in changes:
                if "entry_window" in s.meta:
                    change.entry_window = tuple(s.meta["entry_window"])
                if "expert_index" in s.meta:
                    change.expert_index = s.meta["expert_index"]
                    change.expert_count = s.meta["expert_count"]
                    change.mount_kind = s.meta["mount_kind"]
            out.setdefault(path, []).extend(changes)
    return out


def mount_kind(path: str) -> str:
    path = path.lower()
    if "final_layer" in path:
        return "final"
    if "adaln" in path or "_mod." in path or "_mod_" in path:
        return "adaln"
    if "llm_adapter" in path:
        return "text-adapter"
    if "embedder" in path:
        return "embedder"
    return "other"


def fixed_sections(
    name: str,
    low_rank: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    dense: Mapping[str, torch.Tensor],
    weight: float = 1.0,
    *,
    expert_index: int | None = None,
    expert_count: int = 1,
) -> list[Section]:
    out = []
    for kind, values in (("fixed-lora", low_rank), ("fixed-delta", dense)):
        grouped: dict[str, dict] = {}
        for path, value in values.items():
            scaled = (
                (value[0], value[1] * float(weight))
                if kind == "fixed-lora"
                else value * float(weight)
            )
            mount = mount_kind(path) if expert_index is not None else "other"
            grouped.setdefault(mount, {})[path] = scaled
        for mount, data in grouped.items():
            meta = {"layers": len(data)}
            if expert_index is not None:
                meta.update(expert_index=expert_index, expert_count=expert_count, mount_kind=mount)
            out.append(Section(kind, name, data, meta))
    return out


class ProgramChange(LayerChange):
    def __init__(self, program, bindings, strength=1.0):
        self.program = program
        self.bindings = bindings
        for name, spec in program.get("bindings", {}).items():
            if name not in bindings:
                raise ValueError(f"program binding {name} is unresolved")
            tensor = bindings[name]
            if (
                list(tensor.shape) != spec["shape"]
                or str(tensor.dtype).removeprefix("torch.") != spec["dtype"]
            ):
                raise ValueError(f"program binding {name} has the wrong shape or dtype")
        self.strength = float(strength)
        self._validated = {}

    def validate_for(self, in_size, out_size):
        from .program import ProgramError, layer_change_signature, validate

        sizes = (in_size, out_size)
        if sizes not in self._validated:
            validated = validate(self.program, layer_change_signature(sizes))
            for ins in validated.instructions:
                if ins["op"] == "mask":
                    name = self.program.get("name", "<unnamed>")
                    part = self.program.get("part", "-")
                    raise ProgramError(
                        f"program {name}, part {part}, instruction {ins['id']}, op mask: "
                        "regions in programs come later"
                    )
            self._validated[sizes] = validated
        return self._validated[sizes]

    def add(self, x, y):
        from .program import interpret

        validated = self.validate_for(x.shape[-1], y.shape[-1])
        bindings = {
            name: value.to(device=x.device, dtype=x.dtype)
            if value.is_floating_point()
            else value.to(x.device)
            for name, value in self.bindings.items()
        }
        result = interpret(validated, {"x": x, "y": y}, bindings, {"strength": self.strength})
        return y + next(iter(result.values())).to(y.dtype)


register_section(
    SectionKind("program", "1.0", dict, dict, lambda data, strength: {}, lambda data, weight: data)
)


def resolve_program_bindings(sections, search_dirs):
    from safetensors import safe_open

    from .references import resolve

    if not any(s.kind == "program" for s in sections):
        return sections
    refs = {
        s.meta["sha256"]: s.meta for s in sections if s.kind == "weight-ref" and "sha256" in s.meta
    }
    result = []
    for section in sections:
        if section.kind != "program":
            result.append(section)
            continue
        bindings = dict(section.data)
        for name, spec in section.meta.get("bindings", {}).items():
            if spec.get("kind") != "ref":
                continue
            digest = spec.get("sha256")
            if digest not in refs:
                raise ValueError(f"program binding {name}: missing weight-ref {digest}")
            key = spec.get("tensor_key", spec.get("key"))
            if not isinstance(key, str):
                raise ValueError(f"program binding {name}: missing tensor key")
            path = resolve(refs[digest], search_dirs)
            with safe_open(path, framework="pt", device="cpu") as file:
                if key not in set(file.keys()):
                    raise ValueError(f"program binding {name}: tensor {key!r} missing")
                bindings[name] = file.get_tensor(key)
        result.append(Section(section.kind, section.name, bindings, section.meta))
    return result
