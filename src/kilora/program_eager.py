# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
import torch.nn.functional as F

from .program_validate import ValidatedProgram

LOWERINGS = {}


def _register(names, fn):
    for name in names.split():
        LOWERINGS[(name, 1, "torch_eager")] = fn(name)


_register("add mul sub div pow lt le gt ge eq", lambda n: lambda a, k, e: getattr(torch, n)(*a))
_register(
    "abs exp log sqrt neg tanh sigmoid relu", lambda n: lambda a, k, e: getattr(torch, n)(a[0])
)
_register("detach", lambda n: lambda a, k, e: a[0].detach())
_register("silu gelu", lambda n: lambda a, k, e: getattr(F, n)(a[0], **k))
_register(
    "matmul",
    lambda n: (
        lambda a, k, e: torch.matmul(a[0], a[1].transpose(-1, -2) if k["transpose_b"] else a[1])
    ),
)
_register("narrow", lambda n: lambda a, k, e: torch.narrow(a[0], e["axis"], 0, e["length"]))
_register("svd_vh", lambda n: lambda a, k, e: torch.linalg.svd(a[0], full_matrices=False).Vh)
_register("svd_values", lambda n: lambda a, k, e: torch.linalg.svd(a[0], full_matrices=False).S)
_register("unflatten", lambda n: lambda a, k, e: torch.unflatten(a[0], e["axis"], e["shape"]))
_register("flatten", lambda n: lambda a, k, e: torch.flatten(a[0], e["axis1"], e["axis2"]))
_register("sum_pair", lambda n: lambda a, k, e: torch.sum(a[0], dim=(e["axis1"], e["axis2"])))
_register("transpose", lambda n: lambda a, k, e: a[0].transpose(e["axis1"], e["axis2"]))
_register("rename_axis", lambda n: lambda a, k, e: a[0])
_register("unsqueeze", lambda n: lambda a, k, e: a[0].unsqueeze(e["before"]))
_register("key_projection", lambda n: lambda a, k, e: torch.einsum("...r,emr->...em", *a))
_register("batch_matmul", lambda n: lambda a, k, e: torch.matmul(a[0], a[1]))
_register("stack stack_rows", lambda n: lambda a, k, e: torch.stack(a, dim=e["before"]))
_register(
    "arange_axis", lambda n: lambda a, k, e: torch.arange(a[0].shape[e["axis"]], device=a[0].device)
)
_register("select", lambda n: lambda a, k, e: torch.select(a[0], e["axis"], k["index"]))
_register(
    "repeat_interleave",
    lambda n: lambda a, k, e: torch.repeat_interleave(a[0], k["repeats"], dim=e["axis"]),
)
_register(
    "scale",
    lambda n: (
        lambda a, k, e: (
            a[0] * (k["value"] * e["strength"] * e["timeline"] if k["by_strength"] else k["value"])
        )
    ),
)
_register("clamp", lambda n: lambda a, k, e: torch.clamp(a[0], min=k["low"], max=k["high"]))
_register("clamp_min", lambda n: lambda a, k, e: torch.clamp(a[0], min=k["low"]))
_register("mask", lambda n: lambda a, k, e: a[0] * e[k["source"]])
_register("output_add", lambda n: lambda a, k, e: a[0])
_register(
    "sum mean", lambda n: lambda a, k, e: getattr(torch, n)(a[0], dim=e["axis"], keepdim=k["keep"])
)
_register(
    "max min",
    lambda n: lambda a, k, e: getattr(torch, n)(a[0], dim=e["axis"], keepdim=k["keep"]).values,
)
_register(
    "vector_norm",
    lambda n: (
        lambda a, k, e: torch.linalg.vector_norm(
            a[0], dim=e["axis"], keepdim=k["keep"], dtype=torch.float32
        )
    ),
)
_register("pow_int", lambda n: lambda a, k, e: a[0].pow(k["exponent"]))
_register("softmax cumsum", lambda n: lambda a, k, e: getattr(torch, n)(a[0], dim=e["axis"]))
_register(
    "sort",
    lambda n: lambda a, k, e: torch.sort(a[0], dim=e["axis"], descending=k["descending"]).values,
)
_register("where", lambda n: lambda a, k, e: torch.where(*a))
_register("argmax", lambda n: lambda a, k, e: torch.argmax(a[0], dim=e["axis"]))
_register("one_hot", lambda n: lambda a, k, e: F.one_hot(a[0], num_classes=e["size"]))
_register("gather", lambda n: lambda a, k, e: torch.gather(a[0], e["axis"], a[1]))
_register("cast", lambda n: lambda a, k, e: a[0].to(getattr(torch, k["dtype"])))


def interpret(validated: ValidatedProgram, inputs, bindings, context=None) -> dict:

    from .program import op_table

    table = op_table(validated.op_set, validated.version)
    values = {**inputs, **bindings}
    context = {"strength": 1.0, "timeline": 1.0, "region": 1.0, "roles": 1.0, **(context or {})}
    for ins in validated.instructions:
        op, attrs = ins["op"], ins["attributes"]
        args = [values[name] for name in ins["args"]]
        env = dict(context)
        if op in {"transpose", "sum_pair", "flatten"}:
            axes = validated.value_types[ins["args"][0]].axes
            env["axis1"], env["axis2"] = axes.index(attrs["axis1"]), axes.index(attrs["axis2"])
        if op in {"unsqueeze", "stack", "stack_rows"}:
            env["before"] = validated.value_types[ins["args"][0]].axes.index(attrs["before"]) + int(
                attrs.get("after", False)
            )
        if "axis" in attrs and op not in {
            "one_hot",
            "rename_axis",
            "unsqueeze",
            "stack",
            "svd_values",
            "svd_vh",
            "stack_rows",
        }:
            env["axis"] = validated.value_types[ins["args"][0]].axes.index(attrs["axis"])
        if op == "narrow":
            source = attrs["size_of"]
            axes = validated.value_types[source].axes
            env["length"] = values[source].shape[axes.index(attrs["new_axis"])]
        if op == "unflatten":
            source = attrs["size_of"]
            axes = validated.value_types[source].axes
            env["shape"] = tuple(
                values[source].shape[axes.index(attrs[key])] for key in ("axis1", "axis2")
            )
        if op == "one_hot":
            source = attrs["size_of"]
            env["size"] = values[source].shape[
                validated.value_types[source].axes.index(attrs["axis"])
            ]
        if op in {"add", "sub", "mul", "div", "pow", "lt", "le", "gt", "ge", "eq", "where"}:
            target = validated.value_types[ins["result"]].axes
            aligned = []
            for name, value in zip(ins["args"], args, strict=True):
                axes = validated.value_types[name].axes
                order = [axes.index(axis) for axis in target if axis in axes]
                if order != list(range(len(axes))):
                    value = value.permute(order)
                shape = iter(value.shape)
                value = value.reshape([next(shape) if axis in axes else 1 for axis in target])
                aligned.append(value)
            args = aligned
        op_version = table[op]["since_version"]
        try:
            lowering = LOWERINGS[(op, op_version, "torch_eager")]
        except KeyError as exc:
            raise ValueError(f"No torch_eager lowering for {op} version {op_version}") from exc
        values[ins["result"]] = lowering(args, attrs, env)
    return {name: values[name] for name in validated.outputs}
