# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import inspect
import math
import pickle
import re

import pytest

from kilora import depth_schedule as ds
from kilora.depth_schedule import (
    MAX_DEPTH,
    PRESETS,
    DepthSchedule,
    DepthScheduleError,
    compile_schedule,
)


def d(spec, i=0, n=28, kind="self_attn", t=0.0, key="a"):
    return compile_schedule(spec).depth(i, n, kind, t, key)


def test_none_and_empty_give_none():
    assert compile_schedule(None) is None
    assert compile_schedule("") is None
    assert compile_schedule("   ") is None


def test_list_repeats_last_value():
    s = compile_schedule([1, 2, 3])
    assert [s.depth(i, 6, "mlp", 0.0, "k") for i in range(6)] == [1, 2, 3, 3, 3, 3]


def test_comma_string_is_a_list():
    s = compile_schedule("1,1,2,3")
    assert s.mode == "list"
    assert [s.depth(i, 5, "other", 0.5, "k") for i in range(5)] == [1, 1, 2, 3, 3]
    assert compile_schedule(" 0.5 , 1.5, ").source == (0.5, 1.5)


def test_list_is_fractional_and_clamped():
    s = compile_schedule([-2, 0.25, 40, 2.5])
    assert [s.depth(i, 4, "mlp", 0.0, "k") for i in range(4)] == [0.0, 0.25, MAX_DEPTH, 2.5]


def test_list_rejects_bad_values():
    with pytest.raises(DepthScheduleError):
        compile_schedule([])
    with pytest.raises(DepthScheduleError):
        compile_schedule([1, float("nan")])
    with pytest.raises(DepthScheduleError):
        compile_schedule(["a"])


def test_presets():
    n = 28
    flat = compile_schedule("flat")
    assert all(
        flat.depth(i, n, k, t, "") == 1 for i in range(n) for k in ds.KINDS for t in (0, 0.5, 1)
    )

    late = compile_schedule("late-deep")
    vals = [late.depth(i, n, "mlp", 0.5, "") for i in range(n)]
    assert vals == sorted(vals) and vals[0] == 1 and vals[-1] > 3

    mid = compile_schedule("mid-deep")
    vals = [mid.depth(i, n, "mlp", 0.5, "") for i in range(n)]
    assert max(vals) > vals[0] and max(vals) > vals[-1]
    assert vals.index(max(vals)) in (n // 2 - 1, n // 2)

    early = compile_schedule("early-steps")
    assert early.depth(3, n, "mlp", 0.0, "") > 1
    assert early.depth(3, n, "mlp", 0.2, "") > 1
    assert early.depth(3, n, "mlp", 0.5, "") == 1
    assert early.depth(3, n, "mlp", 1.0, "") == 1

    late_s = compile_schedule("late-steps")
    assert late_s.depth(3, n, "mlp", 0.0, "") == 1
    assert late_s.depth(3, n, "mlp", 0.9, "") > 1

    assert compile_schedule("Late-Deep").preset == "late-deep"
    assert set(PRESETS) == {"flat", "late-deep", "mid-deep", "early-steps", "late-steps"}


def test_spec_example_expression():
    f = "1 + 4*smoothstep(0.2, 0.8, t) * (i >= 14)"
    assert d(f, i=13, t=1.0) == 1
    assert d(f, i=14, t=0.0) == 1
    assert d(f, i=14, t=1.0) == 5
    assert d(f, i=20, t=0.5) == pytest.approx(3.0)


def test_variables_and_kind_flags():
    assert d("i", i=7) == 7
    assert d("n / 4", n=28) == 7
    assert d("t * 8", t=0.25) == 2
    f = "1*self + 2*cross + 3*mlp"
    assert d(f, kind="self_attn") == 1
    assert d(f, kind="cross_attn") == 2
    assert d(f, kind="mlp") == 3
    assert d(f + " + 4", kind="other") == 4


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("min(3, 2, 5)", 2),
        ("max(1, 4, 2)", 4),
        ("clamp(9, 1, 3)", 3),
        ("clamp(-1, 1, 3)", 1),
        ("abs(-2.5)", 2.5),
        ("floor(2.7)", 2),
        ("ceil(2.1)", 3),
        ("round(2.5)", 3),
        ("round(2.4)", 2),
        ("step(0.5, 0.4)", 0),
        ("step(0.5, 0.5)", 1),
        ("smoothstep(0, 1, 0.5)", 0.5),
        ("smoothstep(0, 1, -3)", 0),
        ("smoothstep(0, 1, 7)", 1),
        ("smoothstep(2, 2, 3)", 1),
        ("lerp(2, 6, 0.25)", 3),
        ("1 + 2", 3),
        ("5 - 2", 3),
        ("2 * 3", 6),
        ("7 / 2", 3.5),
        ("2 ** 3", 8),
        ("-(-3)", 3),
        ("+3", 3),
        ("(2 < 3) + (3 < 2)", 1),
        ("(2 <= 2) + (2 > 1) + (2 >= 3)", 2),
        ("(2 == 2) + (2 != 2)", 1),
        ("1 < 2 < 3", 1),
        ("1 < 3 < 2", 0),
        ("(1 and 2) * 5", 5),
        ("(1 and 0) + 2", 2),
        ("(0 or 3) * 4", 4),
        ("(0 or 0) + 1", 1),
        ("(not 0) * 3", 3),
        ("(not 2) + 1", 1),
        ("4 if 1 else 2", 4),
        ("4 if 0 else 2", 2),
        ("7 if t > 0.5 else 1", 1),
    ],
)
def test_functions_and_operators(expression, expected):
    assert d(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression, needle",
    [
        ("foo + 1", "foo"),
        ("sqrt(4)", "sqrt"),
        ("__import__('os')", "__import__"),
        ("min", "min"),
        ("x.y", "Attribute"),
        ("(lambda: 1)()", "unknown function"),
        ("[1, 2][0]", "Subscript"),
        ("'a'", "only numbers"),
        ("True + 1", "only numbers"),
        ("min(a=1)", "keyword"),
        ("min(*[1])", "starred"),
        ("5 // 2", "FloorDiv"),
        ("5 % 2", "Mod"),
        ("1 in 2", "In"),
        ("~1", "Invert"),
        ("t.__class__", "Attribute"),
        ("(t := 2)", "NamedExpr"),
    ],
)
def test_unknown_names_and_nodes_rejected(expression, needle):
    with pytest.raises(DepthScheduleError, match=re.escape(needle)):
        compile_schedule(expression)


def test_syntax_error_and_bad_arity():
    with pytest.raises(DepthScheduleError, match="cannot parse"):
        compile_schedule("1 +")
    with pytest.raises(DepthScheduleError, match="clamp"):
        compile_schedule("clamp(1, 2)")
    with pytest.raises(DepthScheduleError, match="lerp"):
        compile_schedule("lerp(1)")


def test_runtime_errors_show_at_compile():
    with pytest.raises(DepthScheduleError, match="zero"):
        compile_schedule("1 / (i - i)")
    with pytest.raises(DepthScheduleError, match="overflows"):
        compile_schedule("10 ** 400")
    with pytest.raises(DepthScheduleError, match="complex"):
        compile_schedule("(-8) ** 0.5")


def test_no_eval_or_exec_in_module():
    source = inspect.getsource(ds)
    for name in ("eval", "exec", "compile", "__import__", "getattr"):
        assert not re.search(rf"(?<![\w.]){name}\s*\(", source), name
    assert not re.search(r"\bexec\b|\bbuiltins\b", source)
    assert re.findall(r"\beval\b", source) == ["eval"]
    assert 'mode="eval"' in source


def test_expression_clamping():
    assert d("-3") == 0
    assert d("100") == MAX_DEPTH
    assert d("10 ** 300") == MAX_DEPTH
    assert d("2.5") == 2.5


def test_auto():
    s = compile_schedule("auto")
    assert s.is_auto and s.mode == "auto"
    assert s.depth(0, 28, "mlp", 0.0, "k") == MAX_DEPTH
    assert not compile_schedule("flat").is_auto


@pytest.mark.parametrize(
    "spec", [[1, 2.5, 3], "1,2,3", "flat", "late-steps", "1 + i/n * self", "auto"]
)
def test_json_round_trip(spec):
    s = compile_schedule(spec)
    back = DepthSchedule.from_json(s.to_json())
    assert back == s and hash(back) == hash(s)
    grid = [(i, k, t) for i in (0, 5, 27) for k in ds.KINDS for t in (0.0, 0.4, 1.0)]
    assert [back.depth(i, 28, k, t, "x") for i, k, t in grid] == [
        s.depth(i, 28, k, t, "x") for i, k, t in grid
    ]


def test_json_rejects_invalid():
    for text in ("[]", "{}", '{"preset": "nope"}', '{"auto": false}', '{"eval": "1"}', "nope"):
        with pytest.raises(DepthScheduleError):
            DepthSchedule.from_json(text)
    with pytest.raises(DepthScheduleError, match="unknown name"):
        DepthSchedule.from_json('{"expr": "os + 1"}')


def test_immutable_and_picklable():
    s = compile_schedule("1 + 2*cross")
    with pytest.raises(AttributeError):
        s.source = "3"
    back = pickle.loads(pickle.dumps(s))
    assert back == s
    assert back.depth(0, 28, "cross_attn", 0, "") == 3


def test_depth_is_a_float():
    for spec in ([2], "flat", "i", "auto"):
        v = compile_schedule(spec).depth(1, 4, "mlp", 0.0, "k")
        assert isinstance(v, float) and not math.isnan(v)
