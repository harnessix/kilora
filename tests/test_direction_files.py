# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from kilora.bake import ScopedSet, plan_targets
from kilora.comfyui.nodes import KiLoRADirectionNodes
from kilora.comfyui.patching import choices_for, detect_family, diffusers_shapes
from kilora.compute import run_layer
from kilora.errors import KiLoRAError
from kilora.import_terms import decode_ids, read_direction_terms, term_change
from kilora.library import basis as builder
from kilora.library.index import LibraryIndex
from kilora.library.nodes import direction_file_adapters

MOUNT = "blocks.0.cross_attn.q_proj"
TARGET = "transformer_blocks.0.attn2.to_q"
STEM = "lora_unet_blocks_0_cross_attn_q_proj"
PAIR = {"student": "a tree", "target": "a tree in ink", "concept": "ink", "cue": ""}


def write_direction(path, record=None, *, alpha=2.0, zero_row=False):
    up = torch.eye(3)[:, :2].contiguous()
    down = torch.tensor([[2.0, 1.0, 0.0, 0.0], [1.0, 3.0, 0.0, 0.0]])
    if zero_row:
        down[1].zero_()
    tensors = {
        STEM + ".lora_up.weight": up,
        STEM + ".lora_down.weight": down,
        STEM + ".alpha": torch.tensor(alpha),
    }
    metadata = {"license": "Apache-2.0"}
    if record is not None:
        metadata["kilora_directions"] = json.dumps(record)
    save_file(tensors, str(path), metadata=metadata)
    return up, down, up @ down * (alpha / 2)


def make_index(tmp_path, folder):
    index = LibraryIndex.open(tmp_path / "index")
    index.sync([folder])
    return index


@pytest.mark.parametrize("record", [{"pairs": [PAIR]}, {"source_file": "sketch.safetensors"}])
def test_basis_keeps_direction_labels_after_save_and_update(tmp_path, record):
    folder = tmp_path / "loras"
    folder.mkdir()
    path = folder / "ink.safetensors"
    write_direction(path, record)
    index = make_index(tmp_path, folder)
    basis = builder.build_basis(index, rank=2, top=2)
    for result in (basis, builder.load_basis(index), builder.update_basis(basis, index)):
        mount = result.mounts[MOUNT]
        for rows in (mount.output_sources, mount.input_sources):
            assert len(rows) == 2
            for row in rows:
                source = row[0]
                assert source["file"] == str(path)
                assert source["weight"] == 1.0
                if "pairs" in record:
                    assert source["pairs"] == [PAIR]
                    assert source["label"] == "a tree → a tree in ink"
                else:
                    assert source["source_file"] == "sketch.safetensors"
                    assert source["label"] == "sketch.safetensors"


@pytest.mark.parametrize("alpha", [2.0, 1.0, -2.0, 0.0])
@pytest.mark.parametrize("zero_row", [False, True])
def test_two_sided_terms_rebuild_change_and_keep_input_overlap(tmp_path, alpha, zero_row):
    path = tmp_path / "ink.safetensors"
    up, down, expected = write_direction(path, {"pairs": [PAIR]}, alpha=alpha, zero_row=zero_row)
    parts = read_direction_terms(path)[MOUNT]
    assert torch.equal(parts["terms_out"], up)
    actual = parts["terms_out"] @ parts["terms_c"] @ parts["terms_in"].T
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(parts["terms_out"].T @ parts["terms_out"], torch.eye(2))
    if alpha and not zero_row:
        inputs = parts["terms_in"]
        torch.testing.assert_close(inputs.norm(dim=0), torch.ones(2))
        assert abs(float(inputs[:, 0] @ inputs[:, 1])) > 0.5
        torch.testing.assert_close(term_change(parts), expected, atol=1e-6, rtol=1e-6)
    ids = decode_ids(parts["terms_ids"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(value.startswith(digest + ":") for group in ids for value in group)
    assert len(set(ids[0] + ids[1])) == 4
    assert torch.equal(down[0], torch.tensor([2.0, 1.0, 0.0, 0.0]))


def test_one_node_writes_the_entire_file_without_svd(tmp_path, monkeypatch):
    path = tmp_path / "ink.safetensors"
    _, _, expected = write_direction(path, {"pairs": [PAIR]})

    def no_svd(*args, **kwargs):
        raise AssertionError("a saved direction must keep its columns")

    monkeypatch.setattr(torch.linalg, "svd", no_svd)
    adapters = direction_file_adapters(path, {TARGET: (4, 3)})
    assert len(adapters) == 1
    assert adapters[0].num_experts == 1
    assert adapters[0].rank == 2
    plan = plan_targets([ScopedSet(adapters[0], scope="independent")], 1)[TARGET]
    layer = plan.to(torch.device("cpu"), torch.float32).sets[0]
    x = torch.tensor([[[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]])
    base = torch.zeros(1, 2, 3)
    actual = run_layer(x, base, [layer], 1)
    torch.testing.assert_close(actual, x @ expected.T, atol=1e-5, rtol=1e-5)


def test_node_file_source_reaches_product_choices(tmp_path):
    path = tmp_path / "ink.safetensors"
    write_direction(path, {"pairs": [PAIR]})
    (chain,) = KiLoRADirectionNodes().add(
        0.7, 1, library_path=str(tmp_path / "unused"), direction_file=str(path)
    )
    assert len(chain) == 1
    assert chain[0].path == str(path)
    assert chain[0].strength == 0.7
    assert list(KiLoRADirectionNodes.INPUT_TYPES()["optional"])[-1] == "direction_file"
    model = nn.Module()
    model.kilora_family = "anima"
    model.blocks = nn.ModuleList([nn.Module()])
    model.blocks[0].cross_attn = nn.Module()
    model.blocks[0].cross_attn.q_proj = nn.Linear(4, 3, bias=False)
    patcher = SimpleNamespace(model=SimpleNamespace(diffusion_model=model))
    choices = choices_for(patcher, chain, detect_family(patcher))
    assert len(choices) == 1
    assert choices[0].strength == 0.7
    assert len(choices[0].adapter_sets) == 1
    assert choices[0].adapter_sets[0].num_experts == 1
    assert not (tmp_path / "unused").exists()


def test_no_match_error_names_direction_file_and_first_three_targets(tmp_path):
    path = tmp_path / "ink.safetensors"
    write_direction(path, {"pairs": [PAIR]})
    (chain,) = KiLoRADirectionNodes().add(1.0, 1, direction_file=str(path))
    model = nn.Module()
    model.kilora_family = "anima"
    model.blocks = nn.ModuleDict()
    for index in (10, 11):
        block = nn.Module()
        block.cross_attn = nn.Module()
        block.cross_attn.q_proj = nn.Linear(4, 3, bias=False)
        block.cross_attn.k_proj = nn.Linear(4, 3, bias=False)
        model.blocks[str(index)] = block
    patcher = SimpleNamespace(model=SimpleNamespace(diffusion_model=model))
    binding = detect_family(patcher)
    targets = list(diffusers_shapes(patcher, binding))
    assert len(targets) == 4
    with pytest.raises(KiLoRAError) as error:
        choices_for(patcher, chain, binding)
    assert str(error.value) == (
        "the direction file has no mounts matching this model; "
        f"file: {path}; model targets: {targets[:3]}\n"
        f"skipped 1 unmatched mounts ({MOUNT})"
    )


def test_plain_library_file_stays_equal_to_existing_read_and_cut(tmp_path):
    folder = tmp_path / "loras"
    folder.mkdir()
    path = folder / "plain.safetensors"
    write_direction(path)
    index = make_index(tmp_path, folder)
    actual = builder.build_basis(index, rank=2, top=2)
    with safe_open(str(path), framework="pt", device="cpu") as file:
        keys = {
            "up": STEM + ".lora_up.weight",
            "down": STEM + ".lora_down.weight",
            "alpha": STEM + ".alpha",
        }
        left, right = builder._directions(file, keys, 2, "cpu", {}, MOUNT)
    record = index.records[index.paths[str(path)]]
    expected = builder._assemble(
        {str(path): {"cluster": record.get("cluster_id")}},
        {str(path): {MOUNT: (left, right)}},
        2,
        2,
        "cpu",
    )
    for field in ("u", "v", "output_values", "input_values"):
        assert torch.equal(
            getattr(actual.mounts[MOUNT], field), getattr(expected.mounts[MOUNT], field)
        )
    for field in ("output_sources", "input_sources"):
        assert getattr(actual.mounts[MOUNT], field) == getattr(expected.mounts[MOUNT], field)
    assert "direction_source" not in actual.files[str(path)]
    with pytest.raises(ValueError, match="kilora_directions"):
        read_direction_terms(path)


def test_missing_file_source_does_not_create_a_library(tmp_path):
    missing = tmp_path / "missing.safetensors"
    (chain,) = KiLoRADirectionNodes().add(1.0, 1, direction_file=str(missing))
    model = nn.Module()
    model.kilora_family = "anima"
    model.blocks = nn.ModuleList([nn.Module()])
    model.blocks[0].cross_attn = nn.Module()
    model.blocks[0].cross_attn.q_proj = nn.Linear(4, 3, bias=False)
    patcher = SimpleNamespace(model=SimpleNamespace(diffusion_model=model))
    with pytest.raises(FileNotFoundError):
        choices_for(patcher, chain, detect_family(patcher))
    assert not missing.exists()


def test_extractor_output_is_read_without_changing_its_columns(tmp_path):
    from kilora.library.contrast import ContrastResult, MountSolution, write_lora

    path = tmp_path / "exported.safetensors"
    up = torch.eye(3)[:2]
    down = torch.tensor([[3.0, 1.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0]])
    result = ContrastResult({TARGET: MountSolution(down, up, 1e-4, 1.0)}, {"pairs": [PAIR]})
    write_lora(path, result)
    parts = read_direction_terms(path)[MOUNT]
    assert torch.equal(parts["terms_out"], up.T)
    torch.testing.assert_close(term_change(parts), up.T @ down, atol=1e-6, rtol=1e-6)


def test_file_identity_is_shared_across_mounts_and_distinct_across_files(tmp_path):
    from safetensors.torch import load_file

    path = tmp_path / "first.safetensors"
    write_direction(path, {"pairs": [PAIR]})
    tensors = load_file(str(path))
    second_stem = "lora_unet_blocks_1_cross_attn_q_proj"
    tensors.update({k.replace(STEM, second_stem): v.clone() for k, v in tensors.copy().items()})
    save_file(tensors, str(path), metadata={"kilora_directions": json.dumps({"pairs": [PAIR]})})
    shapes = {TARGET: (4, 3), TARGET.replace(".0.", ".1."): (4, 3)}
    adapters = direction_file_adapters(path, shapes)
    assert len(adapters) == 2
    assert all(adapter.num_experts == 1 for adapter in adapters)
    assert adapters[0].name == adapters[1].name
    other = tmp_path / "second.safetensors"
    write_direction(other, {"source_file": "other.safetensors"})
    assert direction_file_adapters(other, shapes)[0].name != adapters[0].name
