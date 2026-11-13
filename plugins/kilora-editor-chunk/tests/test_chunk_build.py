# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import torch
from kilora_editor_chunk import ChunkEditor
from kilora_editor_chunk.build import Collector, build, collect

from kilora.editor_packs import read_pack

NAMES = ["layers.0.attention.to_q", "layers.0.attention.to_out.0"]
OUT, RANK = 6, 4


class StandIn(Collector):
    def __init__(self):
        self.names = NAMES
        self.shapes = {h: [OUT, RANK] for h in NAMES}
        g = torch.Generator().manual_seed(0)
        self.w = {h: torch.randn(OUT * RANK, 8, generator=g) for h in NAMES}
        self.calls = 0

    def generate(self, prompt, seed):
        self.calls += 1
        lat = torch.randn(8, generator=torch.Generator().manual_seed(seed + len(prompt)))
        return lat, {h: torch.eye(RANK) * (1 + len(prompt) % 3) for h in NAMES}

    def grad(self, latent, prompt, seed):
        bias = 0.0 if prompt is None else 0.1 * len(prompt)
        return {h: (self.w[h] @ (latent + bias)).reshape(OUT, RANK) for h in NAMES}


PROMPTS = {
    "pixel": [f"a cat, pixel art {i}" for i in range(6)],
    "ink": [f"a tree, ink drawing {i}" for i in range(6)],
}


def test_collect_saves_each_event_and_resumes(tmp_path):
    t = StandIn()
    files = collect(t, PROMPTS, tmp_path / "events", draws=2, log=lambda _: None)
    assert len(files) == 12 and all(f.exists() for f in files)
    files[3].unlink()
    t2 = StandIn()
    collect(t2, PROMPTS, tmp_path / "events", draws=2, log=lambda _: None)
    assert t2.calls == 1


def test_build_writes_orthogonal_packs(tmp_path):
    t = StandIn()
    files = collect(t, PROMPTS, tmp_path / "events", log=lambda _: None)
    r = build(
        files,
        t.names,
        t.shapes,
        tmp_path / "packs",
        "tiny",
        common=4,
        family=3,
        orthogonalize=True,
    )
    assert set(r.packs) == {"common", "pixel", "ink"}
    m, tensors = read_pack(r.packs["ink"])
    assert m.role == "family" and m.requires and "keys.layers.0.attention.to_q" in tensors
    ed = ChunkEditor.load_packs(list(r.packs.values()))
    q = torch.cat([c.q for c in ed.chunks])
    assert torch.allclose(q @ q.t(), torch.eye(q.shape[0]), atol=0.05)
    assert [c.q.shape[0] for c in ed.chunks] == [4, 3, 3]
    assert r.editor_file.exists()


def test_common_only_family_gets_no_family_pack(tmp_path):
    t = StandIn()
    prompts = {**PROMPTS, "_original": [f"a boat, oil painting {i}" for i in range(4)]}
    files = collect(t, prompts, tmp_path / "events", log=lambda _: None)
    r = build(files, t.names, t.shapes, tmp_path / "packs", "tiny", common=4, family=3)
    assert set(r.packs) == {"common", "pixel", "ink"}


def test_unet_collector_gradient_on_a_tiny_unet(tmp_path):
    import pytest

    diffusers = pytest.importorskip("diffusers")
    from kilora_editor_chunk.build import UNetCollector

    from kilora.testing import make_live_mod

    unet = diffusers.UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(8, 16),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=8,
        attention_head_dim=2,
        norm_num_groups=4,
    )
    mod = tmp_path / "live.kmod"
    make_live_mod(mod, model=unet, targets=("*.transformer_blocks.*.attn1.to_out.0",))

    class Pipe:
        scheduler = diffusers.DDPMScheduler()
        text_encoder = torch.nn.Linear(1, 1)

        def __init__(self):
            self.unet = unet

        def encode_prompt(self, text, device, num_images_per_prompt, do_classifier_free_guidance):
            g = torch.Generator().manual_seed(len(text))
            return torch.randn(1, 3, 8, generator=g), None

    t = UNetCollector(Pipe(), str(mod), size=64, steps=2)
    lat = torch.randn(4, 8, 8, generator=torch.Generator().manual_seed(0))
    g_empty, g_full = t.grad(lat, None, 1), t.grad(lat, "a cat", 1)
    assert set(g_empty) == set(t.names) and len(t.names) >= 1
    h = t.names[0]
    assert g_empty[h].shape == tuple(t.shapes[h]) and float(g_empty[h].abs().sum()) > 0
    assert not torch.equal(g_empty[h], g_full[h])
    assert torch.equal(g_empty[h], t.grad(lat, None, 1)[h])


def _tiny_anima():
    import pytest

    diffusers = pytest.importorskip("diffusers")
    from kilora.anima import ensure_mask_resize

    ensure_mask_resize()
    torch.manual_seed(0)
    return diffusers.CosmosTransformer3DModel(
        in_channels=16,
        out_channels=16,
        num_attention_heads=2,
        attention_head_dim=16,
        num_layers=2,
        mlp_ratio=4.0,
        text_embed_dim=12,
        adaln_lora_dim=4,
        max_size=(4, 32, 32),
        patch_size=(1, 2, 2),
        rope_scale=(1.0, 1.0, 1.0),
        concat_padding_mask=True,
        extra_pos_embed_type=None,
    ).eval()


def test_anima_collector_builds_a_pack_on_the_anima_targets(tmp_path):
    import diffusers
    from kilora_editor_chunk.build import AnimaCollector, check_family_targets

    from kilora.presets import get_preset
    from kilora.testing import make_live_mod

    model = _tiny_anima()
    mod = tmp_path / "live.kmod"
    make_live_mod(
        mod,
        model=model,
        targets=("transformer_blocks.*.attn1.to_out.0", "transformer_blocks.*.attn2.to_q"),
    )

    class Pipe:
        scheduler = diffusers.FlowMatchEulerDiscreteScheduler(shift=3.0)
        transformer = model

    def encode(text):
        return torch.randn(5, 12, generator=torch.Generator().manual_seed(len(text)))

    t = AnimaCollector(Pipe(), str(mod), size=64, steps=2, guidance=4.0, encode=encode)
    patterns = get_preset("anima").target_modules
    import fnmatch

    assert t.names and all(any(fnmatch.fnmatchcase(n, p) for p in patterns) for n in t.names)
    lat, keys = t.generate("a cat", 3)
    assert lat.shape == (16, 8, 8) and set(keys) == set(t.names)
    g_empty, g_full = t.grad(lat, None, 1), t.grad(lat, "a cat", 1)
    h = t.names[0]
    assert g_empty[h].shape == tuple(t.shapes[h]) and float(g_empty[h].abs().sum()) > 0
    assert not torch.equal(g_empty[h], g_full[h])
    assert torch.equal(g_empty[h], t.grad(lat, None, 1)[h])
    prompts = {"_common": ["a cat", "a tree", "a boat"], "ink": ["a cat, ink", "a dog, ink"]}
    files = collect(t, prompts, tmp_path / "events", draws=2, log=lambda _: None)
    r = build(files, t.names, t.shapes, tmp_path / "packs", "anima", common=4, family=2)
    m, tensors = read_pack(r.packs["common"])
    assert m.role == "common" and all(any(n in k for k in tensors) for n in t.names)
    assert ChunkEditor.load_packs(list(r.packs.values())).names == t.names
    import pytest

    with pytest.raises(ValueError, match="not z-image targets"):
        check_family_targets(t.names, "z-image")


def test_pack_licence_follows_the_base_family(tmp_path):
    import json

    from kilora_editor_chunk.build import pack_license
    from safetensors import safe_open

    from kilora.presets import ANIMA_NOTICE

    assert pack_license("anima") == ("LicenseRef-CircleStone-Labs-NC-1.2", ANIMA_NOTICE)
    assert pack_license("z-image") == ("Apache-2.0", None)
    assert pack_license("anima", "LicenseRef-Other") == ("LicenseRef-Other", ANIMA_NOTICE)
    t = StandIn()
    files = collect(t, PROMPTS, tmp_path / "events", log=lambda _: None)
    r = build(files, t.names, t.shapes, tmp_path / "packs", "anima", common=4, family=3)
    for f in r.packs.values():
        m, _ = read_pack(f)
        assert m.license == "LicenseRef-CircleStone-Labs-NC-1.2"
        assert m.raw["license_notice"] == ANIMA_NOTICE
    with safe_open(str(r.editor_file), "pt") as f:
        manifest = json.loads(f.metadata()["kilora.manifest"])
    assert manifest["license"] == "LicenseRef-CircleStone-Labs-NC-1.2"
    assert manifest["license_notice"] == ANIMA_NOTICE
    r = build(files, t.names, t.shapes, tmp_path / "zi", "z-image", common=4, family=3)
    m, _ = read_pack(r.packs["common"])
    assert m.license == "Apache-2.0" and "license_notice" not in m.raw


def _tiny_editor():
    q, _ = torch.linalg.qr(
        torch.randn(OUT * RANK * 2, 6, generator=torch.Generator().manual_seed(3))
    )
    rows = q.t()
    tiny = torch.tensor([3e-9, 1e-8, 4e-7])
    keys = {h: torch.eye(RANK) * 1e-9 for h in NAMES}
    from kilora_editor_chunk import Chunk

    return ChunkEditor(
        NAMES,
        {h: [OUT, RANK] for h in NAMES},
        {"to_q": 1.0, "to_out.0": 1.0},
        [
            Chunk("common", rows[:3], torch.tensor([2e-6, 1e-6, 5e-7]), keys),
            Chunk("ink", rows[3:], tiny, keys),
        ],
    )


def test_tiny_eigenvalues_and_keys_survive_the_pack_round_trip(tmp_path):
    ed = _tiny_editor()
    files = ed.save_packs(tmp_path, "1.0.0", "tiny", "Apache-2.0")
    back = ChunkEditor.load_packs(list(files.values()))
    for a, b in zip(ed.chunks, back.chunks, strict=True):
        assert torch.equal(a.eigenvalues.float(), b.eigenvalues)
        for h in NAMES:
            assert torch.equal(a.mean_keys[h].float(), b.mean_keys[h])
        assert (a.q - b.q).abs().max() < 1e-2


def test_the_int8_block_format_keeps_tiny_values(tmp_path):
    from kilora.editor_packs import write_pack

    ed = _tiny_editor()
    files = ed.save_packs(tmp_path / "ok", "1.0.0", "tiny", "Apache-2.0")
    m, t = read_pack(files["ink"])
    raw = {
        k: v for k, v in m.raw.items() if k not in ("format", "format_version", "tensors", "digest")
    }
    write_pack(tmp_path / "old.kedit-pack.safetensors", raw, t)
    _, old = read_pack(tmp_path / "old.kedit-pack.safetensors")
    assert float(old["eigenvalues"].abs().max()) > 0.0


def test_repack_rebuilds_from_the_events_with_the_old_settings(tmp_path):
    from kilora_editor_chunk.build import repack

    t = StandIn()
    files = collect(t, PROMPTS, tmp_path / "events", log=lambda _: None)
    old = build(files, t.names, t.shapes, tmp_path / "old", "tiny", common=4, family=3)
    new = repack(files, list(old.packs.values()), tmp_path / "new")
    assert set(new.packs) == set(old.packs)
    for fam in old.packs:
        mo, to = read_pack(old.packs[fam])
        mn, tn = read_pack(new.packs[fam])
        assert mn.version == mo.version and mn.license == mo.license
        assert tn["q"].shape == to["q"].shape
        assert float(tn["eigenvalues"].min()) > 0
