# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from ..families import anima_never_target_comfy
from ..presets import anima_name


def basis_mount(mounts, target):
    names = (target, anima_name(target), anima_never_target_comfy(target))
    matches = [name for name in dict.fromkeys(names) if name is not None and name in mounts]
    if len(matches) > 1:
        raise ValueError(f"{target}: multiple basis mounts match: {', '.join(matches)}")
    return matches[0] if matches else None
