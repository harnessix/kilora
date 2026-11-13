# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Callable, Sequence
from typing import Any

from .errors import KiLoRAError

MAX_DEPTH = 16.0
KINDS = ("self_attn", "cross_attn", "mlp", "other")

PRESETS: dict[str, str] = {
    "flat": "1",
    "late-deep": "1 + 3*smoothstep(0.25, 1, i / max(n - 1, 1))",
    "mid-deep": "1 + 3*(1 - abs(2*i / max(n - 1, 1) - 1))",
    "early-steps": "4 if t < 1/3 else 1",
    "late-steps": "4 if t > 2/3 else 1",
}

VARIABLES = ("i", "n", "t", "self", "cross", "mlp")

_NUM = r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?"
_NUMBER_LIST = re.compile(rf"^\s*{_NUM}(\s*,\s*{_NUM})*\s*,?\s*$")


class DepthScheduleError(KiLoRAError, ValueError):
    pass


def _clamp_depth(value: float) -> float:
    value = float(value)
    if math.isnan(value):
        return 0.0
    return min(max(value, 0.0), MAX_DEPTH)


def _smoothstep(e0: float, e1: float, x: float) -> float:
    if e0 == e1:
        return 0.0 if x < e0 else 1.0
    u = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _need(name: str, args: tuple[float, ...], low: int, high: int | None = None) -> None:
    high = low if high is None else high
    if not low <= len(args) <= high:
        want = str(low) if low == high else f"{low} or more" if high > 99 else f"{low} to {high}"
        raise DepthScheduleError(f"{name}() takes {want} arguments, got {len(args)}")


def _f_min(*a: float) -> float:
    _need("min", a, 1, 999)
    return min(a)


def _f_max(*a: float) -> float:
    _need("max", a, 1, 999)
    return max(a)


def _f_clamp(*a: float) -> float:
    _need("clamp", a, 3)
    return min(max(a[0], a[1]), a[2])


def _f_abs(*a: float) -> float:
    _need("abs", a, 1)
    return abs(a[0])


def _f_floor(*a: float) -> float:
    _need("floor", a, 1)
    return float(math.floor(a[0]))


def _f_ceil(*a: float) -> float:
    _need("ceil", a, 1)
    return float(math.ceil(a[0]))


def _f_round(*a: float) -> float:
    _need("round", a, 1)
    return float(math.floor(a[0] + 0.5))


def _f_step(*a: float) -> float:
    _need("step", a, 2)
    return 1.0 if a[1] >= a[0] else 0.0


def _f_smoothstep(*a: float) -> float:
    _need("smoothstep", a, 3)
    return _smoothstep(*a)


def _f_lerp(*a: float) -> float:
    _need("lerp", a, 3)
    return a[0] + (a[1] - a[0]) * a[2]


FUNCTIONS: dict[str, Callable[..., float]] = {
    "min": _f_min,
    "max": _f_max,
    "clamp": _f_clamp,
    "abs": _f_abs,
    "floor": _f_floor,
    "ceil": _f_ceil,
    "round": _f_round,
    "step": _f_step,
    "smoothstep": _f_smoothstep,
    "lerp": _f_lerp,
}

_BINOPS: dict[type, Callable[[float, float], float]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
}

_CMPOPS: dict[type, Callable[[float, float], bool]] = {
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}


def _check(node: ast.AST, source: str) -> None:
    if isinstance(node, ast.Expression):
        _check(node.body, source)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise DepthScheduleError(f"only numbers are allowed as constants, got {node.value!r}")
    elif isinstance(node, ast.Name):
        if node.id not in VARIABLES:
            hint = " (it is a function, call it)" if node.id in FUNCTIONS else ""
            raise DepthScheduleError(
                f"unknown name {node.id!r}{hint}; variables are {', '.join(VARIABLES)}"
            )
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOPS:
            raise DepthScheduleError(f"operator {type(node.op).__name__} is not allowed")
        _check(node.left, source)
        _check(node.right, source)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
            raise DepthScheduleError(f"operator {type(node.op).__name__} is not allowed")
        _check(_unary_value(node), source)
    elif isinstance(node, ast.BoolOp):
        for v in node.values:
            _check(v, source)
    elif isinstance(node, ast.Compare):
        for op in node.ops:
            if type(op) not in _CMPOPS:
                raise DepthScheduleError(f"comparison {type(op).__name__} is not allowed")
        _check(node.left, source)
        for c in node.comparators:
            _check(c, source)
    elif isinstance(node, ast.IfExp):
        _check(node.test, source)
        _check(node.body, source)
        _check(node.orelse, source)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            name = node.func.id if isinstance(node.func, ast.Name) else ast.dump(node.func)
            raise DepthScheduleError(
                f"unknown function {name!r}; functions are {', '.join(FUNCTIONS)}"
            )
        if node.keywords:
            raise DepthScheduleError("keyword arguments are not allowed")
        for a in node.args:
            if isinstance(a, ast.Starred):
                raise DepthScheduleError("starred arguments are not allowed")
            _check(a, source)
    else:
        raise DepthScheduleError(
            f"{type(node).__name__} is not allowed in a depth expression: {source!r}"
        )


def _unary_value(node: ast.UnaryOp) -> ast.expr:
    (value,) = [c for c in ast.iter_child_nodes(node) if isinstance(c, ast.expr)]
    return value


def _evaluate(node: ast.AST, env: dict[str, float]) -> float:
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.BinOp):
        value = _BINOPS[type(node.op)](_evaluate(node.left, env), _evaluate(node.right, env))
        if isinstance(value, complex):
            raise DepthScheduleError("a depth expression gives a complex number")
        return float(value)
    if isinstance(node, ast.UnaryOp):
        v = _evaluate(_unary_value(node), env)
        if isinstance(node.op, ast.Not):
            return 0.0 if v else 1.0
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return 1.0 if all(_evaluate(v, env) for v in node.values) else 0.0
        return 1.0 if any(_evaluate(v, env) for v in node.values) else 0.0
    if isinstance(node, ast.Compare):
        left = _evaluate(node.left, env)
        for op, comp in zip(node.ops, node.comparators, strict=True):
            right = _evaluate(comp, env)
            if not _CMPOPS[type(op)](left, right):
                return 0.0
            left = right
        return 1.0
    if isinstance(node, ast.IfExp):
        return _evaluate(node.body if _evaluate(node.test, env) else node.orelse, env)
    if isinstance(node, ast.Call):
        assert isinstance(node.func, ast.Name)
        return float(FUNCTIONS[node.func.id](*(_evaluate(a, env) for a in node.args)))
    raise DepthScheduleError(f"{type(node).__name__} is not allowed in a depth expression")


def parse_expression(source: str) -> ast.Expression:
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError as exc:
        raise DepthScheduleError(f"cannot parse depth expression {source!r}: {exc.msg}") from None
    _check(tree, source)
    return tree


def _env(index: int, count: int, kind: str, t: float) -> dict[str, float]:
    return {
        "i": float(index),
        "n": float(count),
        "t": float(t),
        "self": 1.0 if kind == "self_attn" else 0.0,
        "cross": 1.0 if kind == "cross_attn" else 0.0,
        "mlp": 1.0 if kind == "mlp" else 0.0,
    }


class DepthSchedule:
    mode: str
    source: Any
    preset: str | None
    _tree: ast.Expression | None

    def __init__(self, mode: str, source: Any = None, preset: str | None = None) -> None:
        tree = None
        if mode == "list":
            values = tuple(float(v) for v in source)
            if not values:
                raise DepthScheduleError("a depth list needs at least one value")
            if not all(math.isfinite(v) for v in values):
                raise DepthScheduleError("depth list values must be finite numbers")
            source = values
        elif mode == "expr":
            source = str(source)
            tree = parse_expression(source)
        elif mode == "auto":
            source = None
        else:
            raise DepthScheduleError(f"unknown depth schedule mode {mode!r}")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "preset", preset)
        object.__setattr__(self, "_tree", tree)
        if tree is not None:
            self._probe()

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("DepthSchedule is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("DepthSchedule is immutable")

    def __reduce__(self) -> tuple[Any, ...]:
        return (DepthSchedule, (self.mode, self.source, self.preset))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DepthSchedule):
            return NotImplemented
        return (self.mode, self.source, self.preset) == (other.mode, other.source, other.preset)

    def __hash__(self) -> int:
        return hash((self.mode, self.source, self.preset))

    def __repr__(self) -> str:
        if self.preset is not None:
            return f"DepthSchedule(preset={self.preset!r})"
        return f"DepthSchedule({self.mode!r}, {self.source!r})"

    @property
    def is_auto(self) -> bool:
        return self.mode == "auto"

    def _probe(self) -> None:
        for count in (2, 28):
            for index in sorted({0, count // 2, count - 1}):
                for kind in KINDS:
                    for t in (0.0, 0.5, 1.0):
                        self.depth(index, count, kind, t, "")

    def depth(self, index: int, count: int, kind: str, t: float, set_key: str) -> float:

        if self.mode == "list":
            values = self.source
            return _clamp_depth(values[min(max(int(index), 0), len(values) - 1)])
        if self.mode == "auto":
            return MAX_DEPTH
        assert self._tree is not None
        try:
            value = _evaluate(self._tree.body, _env(index, count, kind, t))
        except ZeroDivisionError:
            raise DepthScheduleError(
                f"depth expression {self.source!r} divides by zero at i={index}, n={count}, t={t}"
            ) from None
        except OverflowError:
            raise DepthScheduleError(
                f"depth expression {self.source!r} overflows at i={index}, n={count}, t={t}"
            ) from None
        return _clamp_depth(value)

    def to_dict(self) -> dict[str, Any]:
        if self.preset is not None:
            return {"preset": self.preset}
        if self.mode == "list":
            return {"list": list(self.source)}
        if self.mode == "expr":
            return {"expr": self.source}
        return {"auto": True}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DepthSchedule:
        if not isinstance(data, dict) or len(data) != 1:
            raise DepthScheduleError(f"invalid stored depth schedule: {data!r}")
        ((key, value),) = data.items()
        if key == "preset":
            if value not in PRESETS:
                raise DepthScheduleError(f"unknown depth preset {value!r}")
            return cls("expr", PRESETS[value], preset=value)
        if key == "list":
            return cls("list", value)
        if key == "expr":
            return cls("expr", value)
        if key == "auto" and value is True:
            return cls("auto")
        raise DepthScheduleError(f"invalid stored depth schedule: {data!r}")

    @classmethod
    def from_json(cls, text: str) -> DepthSchedule:
        try:
            data = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise DepthScheduleError(f"invalid stored depth schedule: {exc}") from None
        return cls.from_dict(data)


def compile_schedule(spec: str | Sequence[float] | None) -> DepthSchedule | None:

    if spec is None:
        return None
    if isinstance(spec, DepthSchedule):
        return spec
    if isinstance(spec, str):
        text = spec.strip()
        if not text:
            return None
        key = text.lower()
        if key == "auto":
            return DepthSchedule("auto")
        if key in PRESETS:
            return DepthSchedule("expr", PRESETS[key], preset=key)
        if "," in text and _NUMBER_LIST.match(text):
            return DepthSchedule("list", [float(p) for p in text.split(",") if p.strip()])
        return DepthSchedule("expr", text)
    if isinstance(spec, (bytes, bytearray)):
        raise DepthScheduleError("a depth schedule must be text or a list of numbers")
    try:
        values = [float(v) for v in spec]
    except (TypeError, ValueError):
        raise DepthScheduleError(f"invalid depth schedule {spec!r}") from None
    return DepthSchedule("list", values)
