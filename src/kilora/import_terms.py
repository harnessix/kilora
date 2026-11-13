# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

PACK_GAIN_MIN = 0.25
PACK_GAIN_EVENTS = 3


def needs_pack_moves(gain: float, loss_gap: float, minimum: float = PACK_GAIN_MIN) -> bool:

    if not all(math.isfinite(value) for value in (gain, loss_gap, minimum)):
        return False
    if loss_gap <= 0 or not 0 <= minimum <= 1:
        return False
    return gain < minimum * loss_gap


def pack_moves_start(gains, minimum: float = PACK_GAIN_MIN) -> bool:

    gains = [float(g) for g in gains]
    if len(gains) < PACK_GAIN_EVENTS:
        return False
    first = sum(gains[:PACK_GAIN_EVENTS])
    last = sum(gains[-PACK_GAIN_EVENTS:])
    return needs_pack_moves(last, first, minimum)


@dataclass(frozen=True)
class PackTerm:
    output_id: str
    input_id: str
    strength: float


@torch.no_grad()
def pack_directions(names, shapes, q, input_basis, target, pack_id):

    if target not in input_basis or target not in names:
        return {}, {}
    if q.ndim != 2 or len(set(names)) != len(names):
        raise ValueError("invalid pack direction layout")
    offset = 0
    output = None
    for name in names:
        shape = tuple(shapes[name])
        if len(shape) != 2 or any(not isinstance(n, int) or n <= 0 for n in shape):
            raise ValueError("invalid pack hook shape")
        size = math.prod(shape)
        if name == target:
            output = q[:, offset : offset + size].reshape(q.shape[0], *shape)
        offset += size
    if offset != q.shape[1]:
        raise ValueError("pack direction width does not match its hooks")
    basis = input_basis[target]
    if basis.ndim != 2 or basis.shape[1] != output.shape[2]:
        raise ValueError("pack input basis does not match its hook")
    if not torch.isfinite(output).all() or not torch.isfinite(basis).all():
        raise ValueError("pack directions must be finite")
    outputs = {
        f"{pack_id}:out:{row}:{column}": output[row, :, column].detach().clone()
        for row in range(output.shape[0])
        for column in range(output.shape[2])
    }
    inputs = {
        f"{pack_id}:in:{column}": basis[:, column].detach().clone()
        for column in range(basis.shape[1])
    }
    return outputs, inputs


@torch.no_grad()
def term_matrices(
    terms: list[PackTerm],
    outputs: Mapping[str, torch.Tensor],
    inputs: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:

    if not terms:
        raise ValueError("at least one term is required")
    up, down = [], []
    out_width = in_width = None
    for term in terms:
        if not math.isfinite(term.strength):
            raise ValueError("term strength must be finite")
        output, input_ = outputs[term.output_id], inputs[term.input_id]
        if output.ndim != 1 or input_.ndim != 1 or not output.numel() or not input_.numel():
            raise ValueError("pack directions must be nonempty vectors")
        if not torch.isfinite(output).all() or not torch.isfinite(input_).all():
            raise ValueError("pack directions must be finite")
        if out_width is None:
            out_width, in_width = output.numel(), input_.numel()
        if output.numel() != out_width or input_.numel() != in_width:
            raise ValueError("pack direction widths differ")
        up.append(output * term.strength)
        down.append(input_)
    return torch.stack(up, dim=1), torch.stack(down)


PACK_TERM_MOVES = 4

TERM_PARTS = ("terms_out", "terms_in", "terms_c", "terms_ids")
TERM_FIXED = ("terms_out", "terms_in", "terms_ids")


def encode_ids(out_ids, in_ids) -> torch.Tensor:
    data = json.dumps({"out": list(out_ids), "in": list(in_ids)}, separators=(",", ":"))
    return torch.tensor(list(data.encode()), dtype=torch.uint8)


def decode_ids(value: torch.Tensor) -> tuple[list[str], list[str]]:
    data = json.loads(bytes(value.to(torch.uint8).tolist()).decode())
    return list(data["out"]), list(data["in"])


def has_terms(parts: Mapping[str, torch.Tensor]) -> bool:
    return "terms_c" in parts


@torch.no_grad()
def term_pair(parts: Mapping[str, torch.Tensor]):
    if not has_terms(parts):
        return None
    c = parts["terms_c"].float()
    if not bool(c.count_nonzero()):
        return None
    return parts["terms_out"].float() @ c, parts["terms_in"].float().T


@torch.no_grad()
def term_change(parts: Mapping[str, torch.Tensor]) -> torch.Tensor | None:
    pair = term_pair(parts)
    return None if pair is None else pair[0] @ pair[1]


@torch.no_grad()
def add_direction(parts: dict, output_id, output, input_id, input_) -> tuple[int, int]:
    if has_terms(parts):
        out_ids, in_ids = decode_ids(parts["terms_ids"])
        p_out, p_in, c = parts["terms_out"], parts["terms_in"], parts["terms_c"]
    else:
        out_ids, in_ids = [], []
        p_out = torch.zeros(output.numel(), 0)
        p_in = torch.zeros(input_.numel(), 0)
        c = torch.zeros(0, 0)
    if output_id not in out_ids:
        out_ids.append(output_id)
        p_out = torch.cat((p_out, output.float().reshape(-1, 1).to(p_out)), 1)
        c = torch.cat((c, torch.zeros(1, c.shape[1], dtype=c.dtype)), 0)
    if input_id not in in_ids:
        in_ids.append(input_id)
        p_in = torch.cat((p_in, input_.float().reshape(-1, 1).to(p_in)), 1)
        c = torch.cat((c, torch.zeros(c.shape[0], 1, dtype=c.dtype)), 1)
    parts.update(terms_out=p_out, terms_in=p_in, terms_c=c, terms_ids=encode_ids(out_ids, in_ids))
    return out_ids.index(output_id), in_ids.index(input_id)


def set_terms(parts: dict, terms: list[PackTerm], outputs, inputs) -> None:
    for term in terms:
        i, j = add_direction(
            parts,
            term.output_id,
            outputs[term.output_id],
            term.input_id,
            inputs[term.input_id],
        )
        parts["terms_c"][i, j] += term.strength


@torch.no_grad()
def change_norm(up: torch.Tensor, down: torch.Tensor) -> float:
    up, down = up.double(), down.double()
    return max(float(torch.trace((up.T @ up) @ (down @ down.T))), 0.0) ** 0.5


@torch.no_grad()
def difference(old: Mapping, new: Mapping):
    a, b = term_pair(old), term_pair(new)
    if a is None and b is None:
        return None
    if (
        a is not None
        and b is not None
        and all(old[k].shape == new[k].shape and torch.equal(old[k], new[k]) for k in TERM_PARTS)
    ):
        return None
    ups, downs = [], []
    if b is not None:
        ups.append(b[0])
        downs.append(b[1])
    if a is not None:
        ups.append(-a[0])
        downs.append(a[1])
    return torch.cat(ups, 1), torch.cat(downs, 0)


@torch.no_grad()
def read_direction_terms(path) -> dict[str, dict[str, torch.Tensor]]:

    from safetensors import safe_open

    from .library.basis import _resolve_groups
    from .library.direction_files import direction_metadata
    from .lora_import import _scale
    from .multi_import import _group

    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    pack_id = digest.hexdigest()
    result = {}
    with safe_open(str(path), framework="pt", device="cpu") as file:
        if direction_metadata(file.metadata() or {}) is None:
            raise ValueError("expected a LoRA file with kilora_directions metadata")
        groups, skipped, layout = _group(dict.fromkeys(file.keys()), str(path))
        if skipped or not groups:
            raise ValueError("direction file contains unsupported or missing weights")
        for mount, keys in _resolve_groups(groups, layout).items():
            if "up" not in keys or "down" not in keys:
                raise ValueError(f"{mount}: direction file must contain plain LoRA weights")
            up = file.get_tensor(keys["up"]).float()
            down = file.get_tensor(keys["down"]).float()
            if (
                up.ndim != 2
                or down.ndim != 2
                or up.shape[1] != down.shape[0]
                or not down.shape[0]
                or not up.shape[0]
                or not down.shape[1]
                or not torch.isfinite(up).all()
                or not torch.isfinite(down).all()
            ):
                raise ValueError(f"{mount}: invalid direction weights")
            alpha = file.get_tensor(keys["alpha"]) if "alpha" in keys else None
            scale = _scale(str(keys), alpha, down.shape[0], None)
            if not math.isfinite(scale):
                raise ValueError(f"{mount}: direction scale must be finite")
            down = down * scale
            norms = torch.linalg.vector_norm(down, dim=1)
            inputs = down / torch.where(norms > 0, norms, torch.ones_like(norms))[:, None]
            rank = down.shape[0]
            result[mount] = {
                "terms_out": up.contiguous(),
                "terms_in": inputs.T.contiguous(),
                "terms_c": torch.diag(norms),
                "terms_ids": encode_ids(
                    [f"{pack_id}:{mount}:out:{i}" for i in range(rank)],
                    [f"{pack_id}:{mount}:in:{i}" for i in range(rank)],
                ),
            }
    return result
