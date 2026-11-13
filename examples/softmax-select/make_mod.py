# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import argparse
import math

import torch

from kilora import load_mod, save_mod
from kilora.program import program_identity
from kilora.sections import Section


def write_example(source, destination, temperature=2.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and greater than zero")
    mod = load_mod(source)
    if any(s.ir_parts for s in mod.sets):
        raise ValueError("source already has a routed program override")
    if not mod.sets:
        raise ValueError("source must contain a routed set")
    if mod.settings.iteration == "accumulate" and mod.settings.gate_mode in {
        "learned",
        "both",
        "auto",
    }:
        raise ValueError("source must use no learned output gate for accumulate iteration")
    typ = {"axes": ["batch", "tokens", "experts"], "dtype": "float32"}
    program = {
        "part": "select",
        "scope": "*",
        "op_set": "kilora.ops",
        "op_set_version": 1,
        "inputs": {"logits": typ, "noise": typ},
        "outputs": {"weights": typ},
        "bindings": {
            "temperature": {"kind": "embedded", "dtype": "float32", "shape": [], "axes": []}
        },
        "instructions": [
            {
                "id": 0,
                "op": "div",
                "args": ["logits", "temperature"],
                "result": "scaled",
                "attributes": {},
            },
            {
                "id": 1,
                "op": "softmax",
                "args": ["scaled"],
                "result": "weights",
                "attributes": {"axis": "experts"},
            },
        ],
    }
    program["identity"] = program_identity(program)
    section = Section(
        "program", "softmax-select", {"temperature": torch.tensor(temperature)}, program
    )
    save_mod(
        destination, mod.sets, mod.settings, sections=[*mod.sections, section], id="softmax-select"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--temperature", type=float, default=2.0)
    args = parser.parse_args()
    write_example(args.source, args.destination, args.temperature)
