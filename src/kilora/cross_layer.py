# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch


def jvp_responses(block, x: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:

    if directions.ndim != x.ndim + 1 or directions.shape[1:] != x.shape:
        raise ValueError("directions must have shape [rank, *x.shape]")

    def one(direction):
        return torch.func.jvp(block, (x,), (direction,))[1]

    return torch.vmap(one)(directions).movedim(0, -1)


def pass_back(response: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    if response.ndim != 2 or target.ndim < 1 or target.shape[-1] != response.shape[0]:
        raise ValueError("response and target ranks do not match")
    return target @ torch.linalg.pinv(response.to(target)).T


def variance_weight(variance: float) -> float:
    import math

    if not math.isfinite(variance) or variance < 0:
        return 0.0
    return 1.0 / (1.0 + variance)


class RLSLink:
    def __init__(self, r_in: int, r_out: int, forget: float = 0.99):
        if r_in < 1 or r_out < 1 or not 0 < forget <= 1:
            raise ValueError("ranks must be positive and forget must be in (0, 1]")
        self.r_in, self.r_out, self.forget = r_in, r_out, forget
        self.weights = torch.zeros(r_in, r_out, dtype=torch.float64)
        self.covariance = torch.eye(r_in, dtype=torch.float64) * 1000
        self.mean_in = torch.zeros(r_in, dtype=torch.float64)
        self.mean_out = torch.zeros(r_out, dtype=torch.float64)
        self._mean_weight = 0.0
        self._error_sum = 0.0
        self._error_weight = 0.0

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] != self.r_in:
            raise ValueError("input rank mismatch")
        return z @ self.weights.to(device=z.device, dtype=z.dtype)

    def predict_state(self, z: torch.Tensor) -> torch.Tensor:
        mean_in = self.mean_in.to(device=z.device, dtype=z.dtype)
        mean_out = self.mean_out.to(device=z.device, dtype=z.dtype)
        return self.predict(z - mean_in) + mean_out

    def update(self, z_in: torch.Tensor, z_out: torch.Tensor) -> None:
        if (
            z_in.shape[:-1] != z_out.shape[:-1]
            or z_in.shape[-1] != self.r_in
            or z_out.shape[-1] != self.r_out
        ):
            raise ValueError("observation shapes do not match link ranks")
        inputs = z_in.detach().reshape(-1, self.r_in).to(dtype=torch.float64, device="cpu")
        outputs = z_out.detach().reshape(-1, self.r_out).to(dtype=torch.float64, device="cpu")
        for x, y in zip(inputs, outputs, strict=True):
            old_weight = self.forget * self._mean_weight
            new_weight = old_weight + 1.0
            dx, dy = x - self.mean_in, y - self.mean_out
            residual = dy - dx @ self.weights
            scale = (old_weight / new_weight) ** 0.5
            centered_x, centered_y = scale * dx, scale * dy
            px = self.covariance @ centered_x
            gain = px / (self.forget + centered_x @ px)
            self.weights += torch.outer(gain, centered_y - centered_x @ self.weights)
            self.covariance = (
                self.covariance - torch.outer(gain, centered_x @ self.covariance)
            ) / self.forget
            self.mean_in += dx / new_weight
            self.mean_out += dy / new_weight
            self._mean_weight = new_weight
            self._error_sum = self.forget * self._error_sum + residual.square().mean().item()
            self._error_weight = self.forget * self._error_weight + 1

    def state_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "weights": self.weights.clone(),
            "covariance": self.covariance.clone(),
            "mean_in": self.mean_in.clone(),
            "mean_out": self.mean_out.clone(),
            "mean_weight": torch.tensor(self._mean_weight, dtype=torch.float64),
            "statistics": torch.tensor(
                [self.forget, self._error_sum, self._error_weight], dtype=torch.float64
            ),
        }

    @classmethod
    def from_tensors(cls, state: dict[str, torch.Tensor]) -> RLSLink:
        weights, covariance, statistics = (
            state["weights"],
            state["covariance"],
            state["statistics"],
        )
        if weights.ndim != 2 or covariance.shape != (weights.shape[0], weights.shape[0]):
            raise ValueError("invalid link matrix shapes")
        if statistics.shape != (3,) or not all(
            bool(torch.isfinite(t).all()) for t in (weights, covariance, statistics)
        ):
            raise ValueError("invalid link statistics")
        forget, error_sum, error_weight = statistics.tolist()
        if error_sum < 0 or error_weight < 0:
            raise ValueError("invalid link variance history")
        link = cls(weights.shape[0], weights.shape[1], forget=forget)
        link.weights = weights.detach().to(dtype=torch.float64, device="cpu").clone()
        link.covariance = covariance.detach().to(dtype=torch.float64, device="cpu").clone()
        mean_in = state.get("mean_in", link.mean_in)
        mean_out = state.get("mean_out", link.mean_out)
        mean_weight = state.get("mean_weight", torch.tensor(0.0))
        if (
            mean_in.shape != (link.r_in,)
            or mean_out.shape != (link.r_out,)
            or mean_weight.shape != ()
            or not all(bool(torch.isfinite(t).all()) for t in (mean_in, mean_out, mean_weight))
            or mean_weight.item() < 0
        ):
            raise ValueError("invalid link means")
        link.mean_in = mean_in.detach().to(dtype=torch.float64, device="cpu").clone()
        link.mean_out = mean_out.detach().to(dtype=torch.float64, device="cpu").clone()
        link._mean_weight = mean_weight.item()
        link._error_sum, link._error_weight = error_sum, error_weight
        return link

    @property
    def residual_variance(self) -> float:
        return self._error_sum / self._error_weight if self._error_weight else float("nan")

    @property
    def precision(self) -> float:
        return 1.0 / max(self.residual_variance, 1e-12)


def inner_attention_candidates(targets):
    result = []
    for a, b in zip(targets, targets[1:], strict=False):
        left = a.removesuffix(".weight")
        right = b.removesuffix(".weight")
        if not left.endswith(".to_v") or right != left.removesuffix(".to_v") + ".to_out.0":
            continue
        result.append((a, b))
    return result


def inner_attention_links(targets, weights, shapes):
    result = {}
    for a, b in inner_attention_candidates(targets):
        weight = weights.get(b)
        shape = shapes.get(a)
        if weight is not None and weight.ndim == 2 and shape and weight.shape[1] == shape[1]:
            result[(a, b)] = weight
    return result


def sequential_links(model, targets):

    result = {}
    for a, b in zip(targets, targets[1:], strict=False):
        left, _, i = a.removesuffix(".weight").rpartition(".")
        right, _, j = b.removesuffix(".weight").rpartition(".")
        if left != right or not i.isdigit() or not j.isdigit() or int(i) >= int(j):
            continue
        try:
            parent = model.get_submodule(left) if left else model
            if not isinstance(parent, torch.nn.Sequential):
                continue
            modules = tuple(parent[int(i) + 1 : int(j) + 1])
            pointwise = (
                torch.nn.Linear,
                torch.nn.Identity,
                torch.nn.ReLU,
                torch.nn.GELU,
                torch.nn.SiLU,
                torch.nn.Tanh,
                torch.nn.Sigmoid,
            )
            if not all(type(module) in pointwise for module in modules):
                continue
        except (AttributeError, IndexError):
            continue

        def run(x, modules=modules):
            for module in modules:
                x = module(x)
            return x

        result[(a, b)] = run
    return result


def measured_link(block, samples, basis_in, basis_out):
    responses = []
    with torch.inference_mode(False), torch.no_grad():
        for x, expected in samples:
            x = x.detach().clone()
            expected = expected.to(x).clone()
            incoming = basis_in.to(x)
            outgoing = basis_out.to(x)
            baseline = block(x)
            if not isinstance(baseline, torch.Tensor) or baseline.shape != expected.shape:
                raise ValueError("link replay has a different output shape")
            if not torch.allclose(baseline, expected, atol=0.03, rtol=0.03):
                raise ValueError("link replay does not match recorded output")
            directions = incoming[:, None, :].expand(-1, x.shape[0], -1)
            response = jvp_responses(
                lambda value, outgoing=outgoing: block(value) @ outgoing.T, x, directions
            )
            responses.append(response.detach().float().cpu())
    values = torch.cat(responses, dim=0)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("link response is not finite")
    return values.mean(dim=0), float(values.var(dim=0, unbiased=False).mean())
