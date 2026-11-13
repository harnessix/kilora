# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .backends import Backend, FlowMatchingBackend

TINY_FAMILY = "kilora-tiny-1"


class _Attention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.to_q = nn.Linear(width, width, bias=False)
        self.to_k = nn.Linear(width, width, bias=False)
        self.to_v = nn.Linear(width, width, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(width, width)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
        if mask is not None:
            scores = scores + mask
        weights = torch.softmax(scores, dim=-1)
        return self.to_out[0](weights @ v)


class _FeedForward(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(width, 2 * width, bias=False)
        self.w2 = nn.Linear(2 * width, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(x)))


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = _Attention(width)
        self.norm2 = nn.LayerNorm(width)
        self.feed_forward = _FeedForward(width)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), mask)
        return x + self.feed_forward(self.norm2(x))


class TinyDenoiser(nn.Module):
    def __init__(self, channels: int = 4, width: int = 16, num_layers: int = 2, seed: int = 0):
        super().__init__()
        gen_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        try:
            self.proj_in = nn.Linear(channels, width)
            self.layers = nn.ModuleList([_Block(width) for _ in range(num_layers)])
            self.proj_out = nn.Linear(width, channels)
        finally:
            torch.random.set_rng_state(gen_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x)
        for layer in self.layers:
            h = layer(h)
        return self.proj_out(h)


class TinyFlowBackend(FlowMatchingBackend):
    family = TINY_FAMILY

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def forward(self, items, events, generator):
        x0 = torch.stack(list(items)).float()
        noise = torch.randn(x0.shape, generator=generator)
        s = events.view(-1, 1, 1).float()
        noisy = (1 - s) * x0 + s * noise
        return self.model(noisy), x0 - noise


class TinyLM(nn.Module):
    def __init__(self, vocab: int = 32, width: int = 16, num_layers: int = 2, seed: int = 0):
        super().__init__()
        gen_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        try:
            self.embed = nn.Embedding(vocab, width)
            self.layers = nn.ModuleList([_Block(width) for _ in range(num_layers)])
            self.head = nn.Linear(width, vocab)
        finally:
            torch.random.set_rng_state(gen_state)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embed(tokens)
        n = tokens.shape[-1]
        mask = torch.full((n, n), float("-inf")).triu(1)
        for layer in self.layers:
            h = layer(h, mask)
        return self.head(h)


class TinyLMBackend(Backend):
    family = "kilora-tiny-lm"

    def __init__(self, model: nn.Module, window: int = 8) -> None:
        self.model = model
        self.window = window

    def forward(self, items, events, generator):
        seqs = torch.stack(list(items))
        n = seqs.shape[-1]
        size = min(self.window + 1, n)
        starts = (events.float() * (n - size)).round().long().tolist()
        part = torch.stack([s[i : i + size] for s, i in zip(seqs, starts, strict=True)])
        logits = self.model(part[:, :-1])
        target = torch.nn.functional.one_hot(part[:, 1:], logits.shape[-1]).float()
        return logits, target

    def loss(self, prediction, target):
        logp = torch.log_softmax(prediction.float(), dim=-1)
        return -(logp * target).sum(-1).mean()


def tiny_texts(
    count: int, items: int, length: int = 16, vocab: int = 32, seed: int = 0
) -> dict[str, list[torch.Tensor]]:
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for i in range(count):
        pattern = torch.randint(0, vocab, (4,), generator=gen)
        seqs = []
        for _ in range(items):
            seq = pattern.repeat(length // 4 + 1)[:length].clone()
            flip = torch.rand(length, generator=gen) < 0.1
            seq[flip] = torch.randint(0, vocab, (int(flip.sum()),), generator=gen)
            seqs.append(seq)
        out[f"text-{i}"] = seqs
    return out


def tiny_themes(
    count: int, items: int, tokens: int = 6, channels: int = 4, seed: int = 0
) -> dict[str, list[torch.Tensor]]:
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for i in range(count):
        mean = torch.randn(tokens, channels, generator=gen)
        out[f"theme-{i}"] = [
            mean + 0.3 * torch.randn(tokens, channels, generator=gen) for _ in range(items)
        ]
    return out


def make_live_mod(
    path: Any,
    model: nn.Module | None = None,
    seed: int = 0,
    targets: tuple[str, ...] = ("layers.*.attention.to_out.0",),
) -> nn.Module:
    import dataclasses

    from .adapter import Settings, attach
    from .config import KiLoRAConfig
    from .live import LiveSpec
    from .live_sets import LIVE_PRESETS, make_live_set
    from .modfile import save_mod

    model = model if model is not None else TinyDenoiser(seed=seed).eval()
    config = KiLoRAConfig(target_modules=list(targets), num_experts=4, routes=2, router_dim=8)
    adapter = attach(model, config, name="main")
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for site in adapter.sites:
            up = site.tensors["s0_up"]
            up.copy_(torch.randn(up.shape, generator=gen) * 0.2)
    main = adapter.to_sets()[0]
    initial_routers = {
        int(name.split("_")[1]): value.detach().cpu().clone()
        for name, value in adapter.routers.items()
        if name.startswith("s0_")
    }
    adapter.remove()
    live, layout = make_live_set(
        dataclasses.replace(main, routers=initial_routers), LIVE_PRESETS["light"], TINY_FAMILY
    )
    for ex in live.experts.values():
        ex.up.copy_(torch.randn(ex.up.shape, generator=gen) * 0.1)
    spec = LiveSpec(
        layout=layout, updater={"id": "any"}, live_set=live.name, speed=0.5, speed_max=1.0
    )
    save_mod(
        path,
        [dataclasses.replace(main, frozen=True), live],
        Settings(depth=2, softmax_scope="per_set", routing_select="top1"),
        id="test-live",
        base_model={"family": TINY_FAMILY},
        license="CC0-1.0",
        live=spec.to_manifest(),
    )
    return model


def write_steps(session, seeds, size):
    from kilora.editors import Editor, commit_write
    from kilora.recording import GenerationRecord

    class Step(Editor):
        def __init__(self, seed):
            self.seed = seed

        def next_state(self, state, record):
            gen = torch.Generator().manual_seed(self.seed)
            return {
                t: {p: v + torch.randn(v.shape, generator=gen) * size for p, v in parts.items()}
                for t, parts in state.items()
            }

    session.learning = True
    for seed in seeds:
        commit_write(session, Step(seed), GenerationRecord(images=1))
