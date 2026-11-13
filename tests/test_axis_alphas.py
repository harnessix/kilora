# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch

from kilora.call_values import axis_alphas
from kilora.compute import LayerSet, run_layer
from kilora.family_spec import TokenLayout
from kilora.region import latent_size, token_layouts


def _set(axis):
    g = torch.Generator().manual_seed(4 if axis == "creative" else 5)
    return LayerSet(
        router=torch.eye(4),
        keys_in=torch.ones(1, 1, 4),
        down=torch.randn(2, 4, generator=g),
        up=torch.randn(2, 4, generator=g),
        form="input",
        axis=axis,
        gate_mode="none",
        per_expert=2,
    )


def _run(y, sets):
    return run_layer(y, y, sets, 1, target="layer")


def test_axis_alpha_composition_and_nesting():
    y = torch.randn(2, 6, 4, generator=torch.Generator().manual_seed(7))
    a, b = _set("creative"), _set("ip")
    one, two = _run(y, [a]), _run(y, [b])
    original = _run(y, [a, b])
    with axis_alphas({}):
        assert torch.equal(_run(y, [a, b]), original)
    with axis_alphas({"creative": 0.0}):
        assert torch.equal(_run(y, [a]), y)
        with axis_alphas({"creative": 0.3, "ip": 0.7}):
            assert torch.allclose(_run(y, [a, b]), y + 0.3 * (one - y) + 0.7 * (two - y))
        assert torch.equal(_run(y, [a]), y)
    assert torch.equal(_run(y, [a, b]), original)


def test_token_map_and_multiple_token_axes():
    y = torch.randn(2, 2, 3, 4)
    s = _set("creative")
    out = _run(y, [s])
    alpha = torch.tensor([[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]])[None].expand(2, -1, -1)
    with axis_alphas({"creative": alpha}):
        assert torch.allclose(_run(y, [s]), y + alpha[..., None] * (out - y))


def test_joint_layout_preserves_text_and_other_tokens():
    y = torch.randn(1, 8, 4)
    s = _set("creative")
    out = _run(y, [s])
    layout = TokenLayout("image2d", (1, 1), text="before")
    with (
        latent_size((2, 2)),
        token_layouts(lambda _: layout, text_tokens=2),
        axis_alphas({"creative": torch.zeros(1, 4)}),
    ):
        got = _run(y, [s])
    expected = out.clone()
    expected[:, 2:6] = y[:, 2:6]
    assert torch.allclose(got, expected)


def test_alpha_map_resizes_to_layer_grid():
    y = torch.randn(1, 4, 4)
    s = _set("creative")
    out = _run(y, [s])
    alpha = torch.tensor([[0.0, 0.0, 1.0, 1.0]]).repeat(1, 4).reshape(1, 16)
    layout = TokenLayout("image2d", (2, 2))
    with latent_size((4, 4)), token_layouts(lambda _: layout), axis_alphas({"creative": alpha}):
        got = _run(y, [s])
    expected_alpha = torch.tensor([0.0, 1.0, 0.0, 1.0])[None, :, None]
    assert torch.allclose(got, y + expected_alpha * (out - y))


def test_alpha_mapping_cache_is_per_call_and_layout(monkeypatch):
    import kilora.family_spec as family_spec

    calls = []
    original = family_spec.map_region

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(family_spec, "map_region", counted)
    y = torch.randn(1, 4, 4)
    alpha = torch.linspace(0, 1, 16).reshape(1, 16)
    layout = TokenLayout("image2d", (2, 2))
    with latent_size((4, 4)), token_layouts(lambda _: layout):
        with axis_alphas({"creative": alpha}):
            first = _run(y, [_set("creative")])
            second = _run(y, [_set("creative")])
            assert len(calls) == 1
            assert torch.equal(first, second)
        with axis_alphas({"creative": alpha}):
            uncached = _run(y, [_set("creative")])
            assert len(calls) == 2
            assert torch.equal(first, uncached)
