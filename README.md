# KiLoRA

KiLoRA routes several LoRA files together in diffusion models such as Anima, Z-Image and Flux. Each LoRA acts where it responds to the image, instead of all LoRAs adding their full change everywhere. KiLoRA also loads KiLoRA mods, which are routed LoRA experts saved in one file.

The version is 0.0.9.

## Install

ComfyUI: clone this repository into `ComfyUI/custom_nodes/`, install the requirements, and restart ComfyUI.

```bash
git clone https://github.com/harnessix/kilora ComfyUI/custom_nodes/kilora
```

```bash
pip install -r ComfyUI/custom_nodes/kilora/requirements.txt
```

Plain LoRA files stay in `ComfyUI/models/loras/`. Put KiLoRA mods (`.kmod.safetensors`) in `ComfyUI/models/kilora/`.

Python: `pip install "git+https://github.com/harnessix/kilora"`. KiLoRA needs Python 3.10 or later and PyTorch 2.2 or later. On AMD GPUs, install the ROCm build of PyTorch first.

## Agent skills

Skills for coding agents (setup, workflows, LoRA files from prompts, style packs, packaging) are in [harnessix/kilora-skills](https://github.com/harnessix/kilora-skills).

## Start

Use KiLoRA LoRA Loader in place of Load LoRA. The model LoRAs of the chain then route together. To get the same result as Load LoRA, connect KiLoRA Settings (Simple) and set `mix` to `stack`.

## Example workflows

The folder `example_workflows/` has 11 workflows. Drag one into ComfyUI to open it. The LoRA and mod file names in them are placeholders: choose your own files.

- `kilora-lora-mix.json`: three LoRAs routed together at depth 3. LoKr and LoHa files work too.
- `kilora-character-style-background.json`: a character, a style and a background LoRA. The character acts only in its region.
- `kilora-layers.json`: a style for the whole image and a character in a masked area, each with its own prompt.
- `kilora-depth-map.json`: a different depth for the subject and the background, from a mask.
- `kilora-depth-schedule.json`: a depth for each block and sampling step.
- `kilora-timeline.json`: a character mod in the early steps and a style mod in the later steps.
- `kilora-palette-shuffle.json`: a saved set of LoRAs. Shuffle picks two or three of them for each seed.
- `kilora-static-mod-depth1.json`: one KiLoRA mod at depth 1 on Z-Image.
- `kilora-two-mods.json`: two KiLoRA mods with their own signed strengths on Z-Image.
- `kilora-live-mode-stub.json`: live mode on Z-Image. The editor writes the mod after each image.
- `kilora-imported-live-edit.json`: a live edit of two imported LoRA files through a mask, then an export as a plain LoRA.

## Nodes

- Loading: KiLoRA LoRA Loader, KiLoRA Route Loaded LoRAs, KiLoRA Load Mod, KiLoRA Remove Mods.
- Mod lists: Add Mod, Add LoRA, Check, Region, Region Attention, Region Isolation, Token Roles, Two Panels, Suggest LoRAs, and KiLoRA Apply Mod List.
- Settings: KiLoRA Settings, KiLoRA Settings (Simple), KiLoRA Depth Schedule, KiLoRA Depth Map, KiLoRA Timeline, KiLoRA Layer, KiLoRA Layer Filter.
- Palettes and recipes: KiLoRA Palette, KiLoRA Save Palette, KiLoRA Save Recipe, KiLoRA LoRA Library, KiLoRA Similar LoRAs, KiLoRA Clear Local History.
- Information: KiLoRA Trigger Words, KiLoRA Mod Info, KiLoRA Route Record, KiLoRA Route Map.
- Live mods: Record, Write, Settings, Undo/Reset/Save, Editor Packs, Freeze as Static Mod, Export for Sharing. The built-in editor is the direction-set editor in `plugins/kilora-editor-chunk`.
- Direction sets: KiLoRA Direction Nodes.

## Known limits

- The result of routing depends on the seed. Some LoRA pairs, for example two LoRAs that want different subjects, can still mix badly. Try a few seeds, and change the strength of each LoRA.
- The fast GPU kernels need CUDA or ROCm. Some options run on the slower path: candidate selection, learned crosses and custom programs.
- Region Attention and Region Isolation are experimental.
- Similar LoRAs, Suggest LoRAs and the Shuffle `learn_embedding` mode can use an embedding model. No embedding model ships with this release. Without one, these nodes use their default method and say so.
- This release has no static mods and no editor packs. Those ship separately.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
