# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid
from pathlib import Path

import torch


def write_map(values: torch.Tensor, folder: Path, name: str) -> Path:
    import numpy as np
    from PIL import Image

    grid = values[0].detach().float().clamp(0, 1).cpu().numpy()
    Path(folder).mkdir(parents=True, exist_ok=True)
    path = Path(folder) / name
    Image.fromarray((grid * 255 + 0.5).astype(np.uint8), mode="L").save(path)
    return path


def _temp_folder() -> Path:
    from ..host import services

    folder_paths = getattr(services().comfy, "folder_paths", None)
    if folder_paths is None:
        import folder_paths
    return Path(folder_paths.get_temp_directory())


def show_measured(values: torch.Tensor, node_id: str | None) -> dict | None:
    if node_id is None:
        return None
    try:
        from server import PromptServer

        server = PromptServer.instance
        folder = _temp_folder()
    except (ImportError, AttributeError):
        return None
    name = f"kilora_depth_{uuid.uuid4().hex[:12]}.png"
    write_map(values, folder, name)
    image = {"filename": name, "subfolder": "", "type": "temp"}
    server.send_sync(
        "executed",
        {
            "node": str(node_id),
            "display_node": str(node_id),
            "output": {"images": [image]},
            "prompt_id": getattr(server, "last_prompt_id", None),
        },
        getattr(server, "client_id", None),
    )
    return image
