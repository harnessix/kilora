# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import json
import shutil

import torch
from test_comfyui import denoiser, folders
from test_path_write import WrapPatcher, _setup, fresh_sessions

from kilora.comfyui import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from kilora.comfyui import nodes as N


def node_info(key, cls):
    info = {
        "input": cls.INPUT_TYPES(),
        "input_order": {k: list(v) for k, v in cls.INPUT_TYPES().items()},
        "output": cls.RETURN_TYPES,
        "output_is_list": getattr(cls, "OUTPUT_IS_LIST", [False] * len(cls.RETURN_TYPES)),
        "output_name": getattr(cls, "RETURN_NAMES", cls.RETURN_TYPES),
        "name": key,
        "display_name": NODE_DISPLAY_NAME_MAPPINGS.get(key, key),
        "description": getattr(cls, "DESCRIPTION", ""),
        "category": getattr(cls, "CATEGORY", "sd"),
        "output_node": getattr(cls, "OUTPUT_NODE", False) is True,
    }
    if hasattr(cls, "OUTPUT_TOOLTIPS"):
        info["output_tooltips"] = cls.OUTPUT_TOOLTIPS
    if getattr(cls, "DEPRECATED", False):
        info["deprecated"] = True
    if getattr(cls, "DEV_ONLY", False):
        info["dev_only"] = True
    info["search_aliases"] = getattr(cls, "SEARCH_ALIASES", [])
    return info


def test_object_info_is_json_and_consistent(folders):
    for key, cls in NODE_CLASS_MAPPINGS.items():
        info = node_info(key, cls)
        json.dumps(info)
        assert len(info["output_name"]) == len(info["output"]), key
        if "output_tooltips" in info:
            assert len(info["output_tooltips"]) == len(info["output"]), key
        assert callable(getattr(cls(), cls.FUNCTION)), key
        for group in info["input"].values():
            for spec in group.values():
                assert isinstance(spec, tuple) and len(spec) in (1, 2), key


def test_deprecated_nodes_are_dev_only(folders):
    for key, cls in NODE_CLASS_MAPPINGS.items():
        info = node_info(key, cls)
        assert info.get("deprecated", False) == info.get("dev_only", False), key


def test_live_record_keeps_another_nodes_wrapper(denoiser, folders, tmp_path):
    _, _, patched, session, _, xs = _setup(denoiser, folders, tmp_path)
    shutil.copy(tmp_path / "path.kedit.safetensors", folders["kilora"] / "p.kedit.safetensors")
    wp = WrapPatcher(patched.model, patched.object_patches, patched.model_options)
    seen = []

    def other(apply_model, args):
        seen.append(1)
        return apply_model(args["input"], args["timestep"], **args["c"])

    wp.set_model_unet_function_wrapper(other)
    plain = wp.sample(xs[0], steps=2)
    seen.clear()
    model, handle = N.KiLoRALiveRecord().record(wp, "p.kedit.safetensors")
    out = model.sample(xs[0], steps=2)
    assert len(seen) == 2 and handle.calls == 2
    assert all(torch.equal(a, b) for a, b in zip(plain, out, strict=True))
