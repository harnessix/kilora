# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import re
import sys

from .errors import KiLoRAError
from .modfile import load_mod


def _check(path: str, mod_path: str | None = None) -> int:
    from .check import check_file

    report = check_file(path, mod=mod_path)
    for line in report.lines():
        print(line)
    if not report.ok:
        return 1
    if report.kind not in ("static mod", "live mod"):
        return 0
    try:
        mod = load_mod(path)
    except KiLoRAError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    m = mod.manifest
    print(f"{m['name']} ({m['id']} {m['version']}), license {m['license']}")
    print(f"format {m['format_version']}, behavior version {m.get('behavior_version', 1)}")
    print(f"base model family: {m['base_model'].get('family')}")
    depth = m.get("depth", {})
    print(f"depth: recommended {depth.get('recommended')}, trained {depth.get('trained')}")
    for s in mod.sets:
        kind = f"keyed, {s.aggregation}" + (", imported" if s.imported_from else "")
        print(
            f"  set {s.name}: {kind}, {len(s.experts)} target modules, "
            f"{s.num_experts} experts each, strength {s.strength:g}"
        )
        if s.imported_from:
            print(
                f"    imported from {s.imported_from.get('original_name')}, "
                f"license {s.imported_from.get('original_license')}"
            )
    print(f"digest {m['digest']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "data":
        from .data import main as data_main

        return data_main(argv[1:])
    if argv and argv[0] == "directions":
        from .library.contrast import main as directions_main

        return directions_main(argv[1:])
    if argv and argv[0] == "library":
        from .library.bake import main as library_main

        return library_main(argv[1:])
    parser = argparse.ArgumentParser(prog="kilora")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("library", help="prepare cached library files: bake")
    sub.add_parser("data", help="check training data: kilora data check FOLDER")
    sub.add_parser("directions", help="extract prompt or LoRA effects: contrast / effect")
    check = sub.add_parser(
        "check", help="check a mod, editor or live-state file and print a summary"
    )
    check.add_argument("file")
    check.add_argument("--mod", default=None, help="also check that an editor or save fits it")
    ex = sub.add_parser("export-live", help="write a live mod for sharing")
    ex.add_argument("mod")
    ex.add_argument("--output", required=True)
    ex.add_argument("--saves-dir", default=None, help="folder with the saves of this mod")
    ex.add_argument("--save", default=None, help="save to export (with --include-learned-state)")
    ex.add_argument(
        "--include-learned-state",
        action="store_true",
        help="include what the mod learned from your picked images",
    )
    ex.add_argument("--new-id", default=None, help="ID of the mod with the learned state")
    il = sub.add_parser("import-lora", help="write a plain LoRA file as a KiLoRA mod")
    il.add_argument("lora")
    il.add_argument("--output", required=True)
    il.add_argument("--model", required=True, help="Diffusers model folder (its transformer)")
    il.add_argument("--family", default="z-image", help="base model family")
    il.add_argument("--weight", type=float, default=1.0, help="the LoRA weight you use elsewhere")
    il.add_argument("--experts", type=int, default=4)
    il.add_argument(
        "--for-training",
        action="store_true",
        help="convert to state-reading experts (A' = A W+) for training with KiLoRA",
    )
    il.add_argument("--id", default=None)
    il.add_argument("--license", default=None, help="default: the LoRA file's license")
    im = sub.add_parser(
        "import", help="import several LoRA files as one mod: routed experts or a stack"
    )
    im.add_argument("loras", nargs="+", metavar="LORA")
    im.add_argument(
        "--model",
        required=True,
        help="Diffusers model folder (its transformer), or a ComfyUI diffusion model or "
        "text encoder .safetensors file",
    )
    im.add_argument("--output", default=None)
    im.add_argument(
        "--build",
        action="append",
        default=None,
        metavar="MODE[:WEIGHT]=OUT",
        help="write one more mod from the same import, for example routed=a.kmod.safetensors "
        "or stack:0.1667=b.kmod.safetensors (WEIGHT multiplies every LoRA); repeat it",
    )
    im.add_argument(
        "--family", default=None, help="base model family (default: found from --model)"
    )
    im.add_argument(
        "--mode",
        choices=("routed", "stack"),
        default="routed",
        help="routed: one expert per LoRA (default); stack: the weighted sum of the LoRAs",
    )
    im.add_argument("--weights", type=float, nargs="+", default=None, help="one per LoRA")
    im.add_argument(
        "--rank",
        type=int,
        default=None,
        help="rank of LoKr and LoHa layers after the SVD cut; default: the largest plain "
        "LoRA rank in the import",
    )
    im.add_argument("--names", nargs="+", default=None, help="one per LoRA")
    im.add_argument(
        "--source-info",
        default=None,
        help="JSON list with one object (repo, revision, license) per LoRA, in file order",
    )
    im.add_argument("--id", default=None)
    im.add_argument(
        "--license", default=None, help="default: the license all LoRAs share, else NOASSERTION"
    )
    pk = sub.add_parser("package", help="write the release folder of one or more mods")
    pk.add_argument("mods", nargs="+")
    pk.add_argument("--output-dir", required=True)
    pk.add_argument(
        "--samples",
        default=None,
        help="folder of kilora render output; <samples>/<mod id>/grid.png is used",
    )
    rd = sub.add_parser("render", help="render the standard showcase grid of a mod")
    rd.add_argument("mod")
    rd.add_argument("--model", required=True, help="Diffusers model id or folder")
    rd.add_argument("--output-dir", required=True, help="a folder per mod is made inside")
    rd.add_argument("--strengths", type=float, nargs="+", default=None)
    rd.add_argument("--size", type=int, default=640)
    rd.add_argument("--steps", type=int, default=8)
    rd.add_argument("--guidance", type=float, default=0.0)
    rd.add_argument("--device", default="cuda")
    card = sub.add_parser("card", help="write a Hugging Face model card for a mod")
    card.add_argument("mod")
    card.add_argument("--output", default="README.md")
    card.add_argument("--repo-file", default=None, help="file name in the repository")
    card.add_argument(
        "--short",
        action="store_true",
        help="write the short KiLoRA card instead of the Hugging Face card",
    )
    card.add_argument(
        "--embed", action="store_true", help="also store the card JSON in the mod's metadata"
    )
    pv = sub.add_parser("preview", help="render a strength and depth grid for a mod")
    pv.add_argument("mod")
    pv.add_argument("--model", required=True, help="Diffusers model id or folder")
    pv.add_argument("--prompt", default=None, help="default: the mod's first example prompt")
    pv.add_argument("--output", default="preview.png")
    pv.add_argument("--strengths", type=float, nargs="+", default=[-1.0, -0.5, 0.0, 0.5, 1.0])
    pv.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    pv.add_argument("--seed", type=int, default=1234)
    pv.add_argument("--steps", type=int, default=8)
    pv.add_argument("--guidance", type=float, default=0.0)
    pv.add_argument("--size", type=int, default=1024)
    pv.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.command == "card":
        from .card import card_fields, card_markdown, embed_card, model_card
        from .modfile import read_manifest

        if args.embed:
            embed_card(args.mod)
        if args.short:
            text = card_markdown(card_fields(read_manifest(args.mod)))
        else:
            text = model_card(args.mod, args.repo_file)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote {args.output}")
        return 0
    if args.command == "preview":
        return _preview(args)
    if args.command == "check":
        return _check(args.file, args.mod)
    if args.command == "export-live":
        return _export_live(args)
    if args.command == "render":
        return _render(args)
    if args.command == "package":
        return _package(args)
    if args.command == "import-lora":
        return _import_lora(args)
    if args.command == "import":
        return _import(args)
    return 2


def _preview(args, pipe=None) -> int:
    import torch

    from .modfile import read_manifest
    from .preview import preview_grid, save_png

    manifest = read_manifest(args.mod)
    prompt = args.prompt
    if prompt is None:
        prompts = manifest.get("example_prompts") or []
        words = manifest.get("trigger_words") or []
        prompt = prompts[0] if prompts else ", ".join(words)
    if not prompt:
        print("error: give --prompt; the mod has no example prompt", file=sys.stderr)
        return 1
    if pipe is None:
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        from .devices import needs_offload

        if args.device == "cuda" and needs_offload():
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(args.device)
    grid = preview_grid(
        pipe,
        args.mod,
        prompt,
        strengths=args.strengths,
        depths=args.depths,
        seed=args.seed,
        steps=args.steps,
        guidance=args.guidance,
        size=args.size,
    )
    save_png(grid, args.output)
    print(f"Wrote {args.output}")
    return 0


def _package(args) -> int:
    import os

    from .modfile import read_manifest
    from .release import package_mod

    failed = 0
    for mod in args.mods:
        try:
            sample = None
            if args.samples:
                sample = os.path.join(args.samples, read_manifest(mod)["id"])
                sample = sample if os.path.isdir(sample) else None
            p = package_mod(mod, args.output_dir, sample_dir=sample)
        except (KiLoRAError, OSError) as exc:
            print(f"{mod}: error: {exc}", file=sys.stderr)
            failed += 1
            continue
        points = "; ".join(p["open_points"]) or "none"
        print(f"{p['mod']['id']}: {len(p['files'])} files in {args.output_dir}. Open: {points}")
    return 1 if failed else 0


def _render(args, pipe=None) -> int:
    import os

    from .modfile import read_manifest
    from .preview import SHOWCASE_STRENGTHS, render_showcase

    if pipe is None:
        import torch
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        from .devices import needs_offload

        if args.device == "cuda" and needs_offload():
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(args.device)
    mod_id = read_manifest(args.mod)["id"]
    out = os.path.join(args.output_dir, mod_id)
    record = render_showcase(
        pipe,
        args.mod,
        out,
        strengths=args.strengths or SHOWCASE_STRENGTHS,
        steps=args.steps,
        guidance=args.guidance,
        size=args.size,
    )
    rows, cols = len(record["rows"]), len(record["columns"])
    print(f"Wrote {out}/grid.png ({rows} prompts by {cols} strengths) and render.json")
    return 0


def _export_live(args) -> int:
    from .live import LEARNED_STATE_WARNING, open_session

    try:
        session = open_session(args.mod, saves_dir=args.saves_dir)
        if args.include_learned_state:
            if args.save is None:
                raise KiLoRAError("--include-learned-state needs --save")
            session.load(args.save)
            print(f"warning: {LEARNED_STATE_WARNING}", file=sys.stderr)
        session.export(
            args.output, include_learned_state=args.include_learned_state, new_id=args.new_id
        )
    except KiLoRAError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    what = "with the learned state" if args.include_learned_state else "with the shipped state"
    print(f"Wrote {args.output} {what}.")
    return 0


def _import_lora(args, model=None) -> int:
    from .lora_import import import_to_mod

    try:
        if model is None:
            from diffusers import AutoModel

            model = AutoModel.from_pretrained(args.model, subfolder="transformer")
        _, report = import_to_mod(
            args.lora,
            model,
            args.output,
            family=args.family,
            weight=args.weight,
            experts=args.experts,
            state_form=args.for_training,
            mod_id=_mod_id(args.id),
            license=args.license,
        )
    except (KiLoRAError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report)
    return 0


def _mod_id(value: str | None) -> str | None:
    if value is None:
        return None
    mod_id = value.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", mod_id):
        raise KiLoRAError(
            f"--id {value!r}: use 1 to 64 letters, digits, '.', '_' or '-', starting with a "
            "letter or digit (for example region-a)"
        )
    return mod_id


def _import_sources(args) -> list:
    import json

    from .multi_import import LoRASource

    n = len(args.loras)
    for flag, values in (("--weights", args.weights), ("--names", args.names)):
        if values is not None and len(values) != n:
            raise KiLoRAError(f"{flag} needs {n} values, one per LoRA, got {len(values)}")
    info: list = [{}] * n
    if args.source_info:
        with open(args.source_info, encoding="utf-8") as f:
            info = json.load(f)
        if not isinstance(info, list) or len(info) != n:
            raise KiLoRAError(f"--source-info must be a JSON list of {n} objects, one per LoRA")
        if not all(isinstance(item, dict) for item in info):
            raise KiLoRAError("--source-info: each entry must be an object")
    return [
        LoRASource(
            path=path,
            weight=args.weights[i] if args.weights else 1.0,
            name=args.names[i] if args.names else None,
            repo=info[i].get("repo"),
            revision=info[i].get("revision"),
            license=info[i].get("license"),
        )
        for i, path in enumerate(args.loras)
    ]


def import_license(sources: list[dict], given: str | None = None) -> str:
    if given:
        return given
    licenses = {s.get("license") or "NOASSERTION" for s in sources}
    return licenses.pop() if len(licenses) == 1 else "NOASSERTION"


def _import(args, model=None) -> int:
    import os
    import re

    from .adapter import imported_schedule_fields, imported_settings
    from .modfile import save_mod
    from .multi_import import import_loras, stack_set
    from .sections import scaled

    try:
        sources = _import_sources(args)
        builds = _import_builds(args)
        family = args.family
        if model is None:
            model, family = _base_model(args.model, family)
        result = import_loras(sources, model, family=family, rank=getattr(args, "rank", None))
        stacked = None
        for mode, weight, output in builds:
            if mode == "routed":
                adapter_set = result.adapter_set
            else:
                stacked = stacked if stacked is not None else stack_set(result)
                adapter_set = stacked
            sections = result.sections
            if weight != 1.0:
                adapter_set = _scaled(adapter_set, weight)
                sections = [scaled(x, weight) for x in sections]
            stem = os.path.splitext(os.path.basename(output))[0]
            mod_id = _mod_id(args.id) or re.sub(r"[^a-z0-9._-]", "-", stem.lower())[:64].strip("-.")
            save_mod(
                output,
                [adapter_set],
                imported_settings(),
                id=mod_id or "import",
                name=stem or mod_id or "import",
                license=import_license(result.sources, args.license),
                base_model={"family": family or "unknown"},
                imported_sources=result.sources,
                composition=mode,
                sections=sections,
                **(imported_schedule_fields() if mode == "routed" else {}),
            )
    except (KiLoRAError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(result.report())
    for mode, weight, output in builds:
        print(
            f"Composition: {mode}"
            + (f" x {weight:g}" if weight != 1.0 else "")
            + f". Wrote {output}."
        )
    return 0


def _import_builds(args) -> list[tuple[str, float, str]]:
    builds = []
    if getattr(args, "output", None):
        builds.append((args.mode, 1.0, args.output))
    for spec in getattr(args, "build", None) or []:
        head, sep, output = spec.partition("=")
        mode, _, weight = head.partition(":")
        if not sep or not output or mode not in ("routed", "stack"):
            raise ValueError(f"--build {spec!r}: use MODE[:WEIGHT]=OUT with routed or stack")
        if "/" in weight:
            num, den = weight.split("/", 1)
            value = float(num) / float(den)
        else:
            value = float(weight) if weight else 1.0
        builds.append((mode, value, output))
    if not builds:
        raise ValueError("give --output or at least one --build")
    return builds


def _scaled(adapter_set, weight: float):
    import dataclasses

    experts = {
        t: dataclasses.replace(ex, up=ex.up * weight) for t, ex in adapter_set.experts.items()
    }
    return dataclasses.replace(adapter_set, experts=experts)


def _base_model(path: str, family: str | None):
    import os

    from .families import checkpoint_tree, diffusers_family

    if os.path.isfile(path):
        model, spec = checkpoint_tree(path, family)
        return model, spec.family
    model = _model_skeleton(path)
    return model, family or diffusers_family(model)


def _model_skeleton(path: str):
    import torch

    try:
        import diffusers
        from diffusers import AutoModel

        config = AutoModel.load_config(path, subfolder="transformer")
        cls = getattr(diffusers, config["_class_name"])
        with torch.device("meta"):
            return cls.from_config(config)
    except Exception:
        from diffusers import AutoModel

        return AutoModel.from_pretrained(path, subfolder="transformer")


if __name__ == "__main__":
    raise SystemExit(main())
