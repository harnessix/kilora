# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from test_import_modes import DIM, Toy

from kilora import KiLoRAConfig, attach


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("output_gate", ["none", "separable"])
def test_strengths_set_between_forward_and_backward(depth, output_gate):
    m = Toy()
    x = torch.randn(1, 4, DIM, generator=torch.Generator().manual_seed(1))
    adapter = attach(
        m,
        KiLoRAConfig(
            target_modules=["blocks.0.mlp.layer1"],
            num_experts=3,
            router_dim=8,
            depth=depth,
            train_depth=None,
            output_gate=output_gate,
        ),
    )
    params = [p for p in adapter.parameters() if p.requires_grad]
    assert params
    loss = m(x).pow(2).mean()
    adapter.set_strengths([0.5])
    loss.backward()
    ups = [p for site in adapter.sites for key, p in site.tensors.items() if key.endswith("_up")]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in ups)
    assert all(
        not p.requires_grad and p.grad is None
        for site in adapter.sites
        for key, p in site.tensors.items()
        if "_keys_" in key
    )
    adapter.remove()
