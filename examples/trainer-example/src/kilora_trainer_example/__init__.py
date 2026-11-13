# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from kilora import KiLoRAConfig, attach
from kilora.editors import commit_write
from kilora.recording import GenerationRecord
from kilora.testing import TINY_FAMILY, TinyDenoiser, make_live_mod
from kilora.trainers import TrainerPlugin

TARGET = "layers.*.attention.to_out.0"


def train_static(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="example-static")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    torch.manual_seed(args.seed)
    model = TinyDenoiser(seed=args.seed).eval()
    model.requires_grad_(False)
    gen = torch.Generator().manual_seed(args.seed)
    xs = [torch.randn(2, 6, 4, generator=gen) for _ in range(4)]
    with torch.no_grad():
        targets = [model(x) + 0.1 for x in xs]
    adapter = attach(model, KiLoRAConfig(target_modules=[TARGET], num_experts=4, depth=1))
    opt = torch.optim.AdamW(adapter.trainable_parameters(), lr=1e-2)
    losses = []
    for step in range(args.steps):
        adapter.start_step(step)
        x, y = xs[step % len(xs)], targets[step % len(xs)]
        loss = adapter.wrap_loss(torch.nn.functional.mse_loss(model(x), y))
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(loss))
    adapter.save(
        args.out,
        id="example-static",
        name="Example static mod",
        license="CC0-1.0",
        base_model={"family": TINY_FAMILY},
        trained_with={"tool": "kilora-trainer-example", "version": "0.1.0"},
    )
    adapter.remove()
    report = {"first_loss": losses[0], "last_loss": losses[-1], "steps": args.steps}
    Path(f"{args.out}.report.json").write_text(json.dumps(report, indent=1))
    print(f"Saved {args.out}: loss {losses[0]:.4f} -> {losses[-1]:.4f}", file=sys.stderr)
    return 0


def _episode_score(rate: float, seed: int, workdir: Path, writes: int, budget: float) -> float:
    import kilora_pullback as pb

    from kilora.editors import editor_for, register_editor
    from kilora.live import open_session

    register_editor(pb.PLUGIN, "kilora-trainer-example")
    mod = workdir / f"episode-{seed}.kmod"
    if not mod.exists():
        make_live_mod(mod, seed=seed)
    editor_file = workdir / f"rate-{rate:.3f}.kedit.safetensors"
    pb.save_pullback_editor(editor_file, rate=rate, base_family=TINY_FAMILY)
    session = open_session(mod)
    session.learning = True
    gen = torch.Generator().manual_seed(seed)
    session.state = {
        t: {**p, "up": p["up"] + torch.randn(p["up"].shape, generator=gen)}
        for t, p in session.state.items()
    }

    def drift() -> float:
        return sum(
            float((session.state[t]["up"] - session.clean[t]["up"]).norm()) for t in session.state
        )

    start = drift()
    editor = editor_for(editor_file)
    editor.check(session)
    commit_write(session, editor, GenerationRecord(images=1))
    first = 1.0 - drift() / start
    for _ in range(writes - 1):
        commit_write(session, editor, GenerationRecord(images=1))
    left = drift() / start
    return left + max(0.0, first - budget) * 10.0


def train_pullback_rate(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="pullback-rate")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--writes", type=int, default=8)
    p.add_argument("--budget", type=float, default=0.3, help="largest share of drift per write")
    p.add_argument("--family", default=TINY_FAMILY)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    import kilora_pullback as pb

    workdir = Path(f"{args.out}.work")
    workdir.mkdir(parents=True, exist_ok=True)
    rates = [round(0.05 * i, 2) for i in range(1, 21)]
    scores = {
        r: sum(
            _episode_score(r, args.seed + e, workdir, args.writes, args.budget)
            for e in range(args.episodes)
        )
        / args.episodes
        for r in rates
    }
    best = min(rates, key=lambda r: (scores[r], r))
    pb.save_pullback_editor(args.out, rate=best, base_family=args.family)
    report = {"rate": best, "score": scores[best], "scores": scores}
    Path(f"{args.out}.report.json").write_text(json.dumps(report, indent=1))
    print(f"Saved {args.out}: rate {best}, score {scores[best]:.4f}", file=sys.stderr)
    return 0


TRAINERS = [
    TrainerPlugin("example-static", "mod", run=train_static, title="Example static mod"),
    TrainerPlugin("pullback-rate", "editor", run=train_pullback_rate, title="Pullback rate"),
]
