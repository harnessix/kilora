# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch
from conftest import TARGET

from kilora import KiLoRAConfig, attach
from kilora.editors import Editor, commit_write
from kilora.live import LiveSpec
from kilora.live_sets import LIVE_PRESETS, make_live_set
from kilora.modfile import save_mod
from kilora.recording import GenerationRecord
from kilora.testing import TINY_FAMILY


def main_set(model, seed=0):
    a = attach(
        model,
        KiLoRAConfig(target_modules=[TARGET], num_experts=4, routes=2, router_dim=8, seed=seed),
        name="main",
    )
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for site in a.sites:
            site.tensors["s0_up"].copy_(
                torch.randn(site.tensors["s0_up"].shape, generator=gen) * 0.2
            )
    main = a.to_sets()[0]
    a.remove()
    return main


def build_live_mod(
    model,
    path,
    mod_id="ink",
    seed=0,
    family="style",
    updater="plugin",
    base_family=TINY_FAMILY,
):
    main = main_set(model, seed)
    live, layout = make_live_set(main, LIVE_PRESETS["light"], base_family)
    gen = torch.Generator().manual_seed(seed)
    for ex in live.experts.values():
        ex.up.copy_(torch.randn(ex.up.shape, generator=gen) * 0.1)
    spec = LiveSpec(layout=layout, updater=updater, live_set=live.name, speed=0.5, speed_max=1.0)
    from kilora.adapter import Settings

    sets = [main, live]
    save_mod(
        path,
        sets,
        Settings(depth=2, softmax_scope="per_set"),
        id=mod_id,
        mod_family=family,
        base_model={"family": base_family},
        live=spec.to_manifest(),
    )
    return path, None, layout


class StepEditor(Editor):
    def __init__(self, seed=0, size=0.1, value=None):
        self.seed, self.size, self.value = seed, size, value

    def next_state(self, state, record):
        gen = torch.Generator().manual_seed(self.seed)
        out = {}
        for t, parts in state.items():
            out[t] = {}
            for p, v in parts.items():
                step = torch.randn(v.shape, generator=gen) * self.size
                out[t][p] = torch.full_like(v, self.value) if self.value is not None else v + step
        return out


def write(session, seed=0, size=0.1, value=None):
    return commit_write(session, StepEditor(seed, size, value), GenerationRecord(images=1))
