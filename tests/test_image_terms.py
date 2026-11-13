# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import torch
from torch import nn

from kilora import prediction_write as pw
from kilora.image_terms import image_terms, image_terms_note
from kilora.recording import CrossFrame, active_recorder

WIDTH, RANK = 4, 2
MOUNTS = ("transformer_blocks.0.mount", "transformer_blocks.1.mount")


class Mount(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.linear = nn.Linear(WIDTH, WIDTH, bias=False, dtype=torch.float64)

    def forward(self, x):
        h = self.linear(x)
        recorder = active_recorder()
        if recorder is not None and recorder.wants_output(self.name):
            recorder.observe_output(self.name, x, h, h)
        return h


class Block(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.mount = Mount(name)

    def forward(self, hidden_states):
        return hidden_states + self.mount(hidden_states)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        self.transformer_blocks = nn.ModuleList(Block(name) for name in MOUNTS)

    def forward(self, hidden_states):
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states


@dataclass
class Sample:
    target: dict
    student: dict
    image: int = 0


def _setup():
    gen = torch.Generator().manual_seed(5)
    frames, responses, p_out, priors = {}, {}, {}, {}
    for name in MOUNTS:
        p_in = torch.linalg.qr(torch.randn(WIDTH, RANK, generator=gen, dtype=torch.float64))[0].T
        p_o = torch.linalg.qr(torch.randn(WIDTH, RANK, generator=gen, dtype=torch.float64))[0].T
        up = torch.randn(WIDTH, RANK, generator=gen, dtype=torch.float64)
        mix = torch.randn(RANK, RANK, generator=gen, dtype=torch.float64)

        def response(z, up=up, mix=mix):
            return (z.double() @ mix.t()).unsqueeze(-2) * up

        frames[name] = [CrossFrame(p_in, p_o)]
        responses[(name, 0)] = (name, response)
        p_out[(name, 0)] = p_o
        zero = torch.zeros(RANK, dtype=torch.float64)
        priors[name] = pw.NodePrior(zero, zero.clone(), torch.eye(RANK, dtype=torch.float64) * 0.1)
    return frames, responses, p_out, priors


def _latent(seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(1, 3, WIDTH, generator=gen, dtype=torch.float64)


def _terms(samples):
    frames, responses, p_out, priors = _setup()
    terms = image_terms(
        None, samples, model=Model(), frames=frames, responses=responses, p_out=p_out
    )
    return terms, priors


def test_equal_calls_give_zero_error_and_no_change():
    x = _latent(1)
    terms, priors = _terms([Sample({"hidden_states": x}, {"hidden_states": x.clone()})])
    assert {t.node for t in terms} == set(MOUNTS)
    assert all(t.source == "images" for t in terms)
    for term in terms:
        assert torch.equal(term.error, torch.zeros_like(term.error))
    result = pw.solve_map(terms, priors)
    for delta in result.delta.values():
        assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-12)
    note = image_terms_note(terms)
    assert "images=1" in note and "samples=1" in note and "mounts=2" in note


def test_one_image_matches_brute_force_least_squares():
    sample = Sample({"hidden_states": _latent(2)}, {"hidden_states": _latent(3)})
    terms, priors = _terms([sample])
    result = pw.solve_map(terms, priors)
    for name in MOUNTS:
        mine = [t for t in terms if t.node == name]
        a = torch.cat([t.response.double() for t in mine])
        e = torch.cat([t.error.double() for t in mine])
        root = torch.linalg.cholesky(priors[name].inv_cov.double())
        lhs = torch.cat([a, root.T])
        rhs = torch.cat([-e, torch.zeros(RANK, dtype=torch.float64)])
        expected = torch.linalg.lstsq(lhs, rhs.unsqueeze(-1)).solution.squeeze(-1)
        assert torch.allclose(result.delta[name], expected, atol=1e-9)
        assert result.delta[name].abs().sum() > 0


def test_opposite_terms_cancel_in_the_shared_solve():
    student = _latent(4)
    shared, own = _latent(5), _latent(6)
    pair = [
        Sample({"hidden_states": student + shared + own}, {"hidden_states": student}, image=0),
        Sample({"hidden_states": student + shared - own}, {"hidden_states": student}, image=1),
    ]
    same = [
        Sample({"hidden_states": student + shared}, {"hidden_states": student}, image=0),
        Sample({"hidden_states": student + shared}, {"hidden_states": student}, image=1),
    ]
    terms, priors = _terms(pair)
    reference, _ = _terms(same)
    solved = pw.solve_map(terms, priors)
    expected = pw.solve_map(reference, priors)
    alone, _ = _terms(pair[:1])
    first = pw.solve_map(alone, priors)
    for name in MOUNTS:
        assert torch.allclose(solved.delta[name], expected.delta[name], atol=1e-9)
        assert not torch.allclose(first.delta[name], expected.delta[name], atol=1e-6)
    assert "images=2" in image_terms_note(terms)


def test_images_weight_is_known_and_old_defaults_stay():
    weights = pw.check_weights({"images": 1.0})
    assert weights["images"] == 1.0
    assert pw.DEFAULT_WEIGHTS["images"] == 1.0
    old = {"depth": 0.25, "cross": 0.25, "hier": 0.25, "clamp": 1.0}
    assert {k: weights[k] for k in old} == old
    assert pw.SOURCES[:4] == ("depth", "cross", "hier", "clamp")
