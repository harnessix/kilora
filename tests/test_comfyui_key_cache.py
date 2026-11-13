# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch

from kilora import KiLoRAConfig, attach
from kilora.comfyui import key_cache
from kilora.fast_keys import PromptCache, fast_state_keys
from kilora.testing import TinyDenoiser

CHANNELS = 4
TOKENS = 12
PROMPTS = ["a red fox", "a castle at night"]
SETTINGS = {"prompts": PROMPTS, "sigmas": [0.8, 0.5], "m_in": 2, "m_out": 3, "seed": 0}


@pytest.fixture(autouse=True)
def cache_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(key_cache, "CACHE_DIR", str(tmp_path / "key-cache"))
    key_cache._FILE_HASHES.clear()
    return tmp_path / "key-cache"


def _lora(tmp_path, content=b"lora weights"):
    path = tmp_path / "style.safetensors"
    path.write_bytes(content)
    return path


def _key_pass(calls, settings=SETTINGS):
    def compute():
        calls.append(1)
        model = TinyDenoiser(channels=CHANNELS, width=16, num_layers=2, seed=0).eval()
        adapter = attach(
            model,
            KiLoRAConfig(
                target_modules=["layers.*.attention.to_out.0"],
                num_experts=4,
                routes=1,
                router_dim=8,
                m_in=settings["m_in"],
                m_out=settings["m_out"],
                depth=1,
                train_depth=None,
            ),
        )
        with torch.no_grad():
            for site in adapter.sites:
                site.tensors["s0_up"].normal_(0, 0.3, generator=torch.Generator().manual_seed(1))

        def run(latents, sigmas, text):
            return model(torch.cat([latents * sigmas.view(-1, 1, 1), text], dim=1))

        def encode(prompt):
            seed = sum(map(ord, prompt))
            return torch.randn(4, CHANNELS, generator=torch.Generator().manual_seed(seed))

        return fast_state_keys(
            adapter,
            run,
            PromptCache(encode),
            settings["prompts"],
            latent_shape=(TOKENS, CHANNELS),
            sigmas=settings["sigmas"],
            m_in=settings["m_in"],
            m_out=settings["m_out"],
            seed=settings["seed"],
        )

    return compute


def test_second_load_does_not_run_the_key_pass(tmp_path, cache_folder):
    lora, calls = _lora(tmp_path), []
    first = key_cache.state_keys(lora, "tiny", SETTINGS, _key_pass(calls))
    assert calls == [1]
    assert len(list(cache_folder.glob("*.safetensors"))) == 1
    key_cache._FILE_HASHES.clear()

    def fail():
        raise AssertionError("the key pass ran on the second load")

    second = key_cache.state_keys(lora, "tiny", SETTINGS, fail)
    assert set(second) == set(first)
    assert second.log_tau == pytest.approx(first.log_tau)
    for target in first:
        for part in key_cache.PARTS:
            torch.testing.assert_close(second[target][part], first[target][part], rtol=0, atol=0)


def test_changed_lora_file_runs_the_key_pass_again(tmp_path):
    lora, calls = _lora(tmp_path), []
    key_cache.state_keys(lora, "tiny", SETTINGS, _key_pass(calls))
    lora.write_bytes(b"other lora weights")
    os.utime(lora, ns=(1, 1))
    key_cache.state_keys(lora, "tiny", SETTINGS, _key_pass(calls))
    assert calls == [1, 1]


def test_same_content_at_a_new_path_hits_the_cache(tmp_path):
    lora, calls = _lora(tmp_path), []
    key_cache.state_keys(lora, "tiny", SETTINGS, _key_pass(calls))
    copy = tmp_path / "renamed.safetensors"
    copy.write_bytes(lora.read_bytes())
    key_cache.state_keys(copy, "tiny", SETTINGS, _key_pass(calls))
    assert calls == [1]


@pytest.mark.parametrize(
    "change",
    [{"m_in": 1}, {"seed": 3}, {"prompts": ["a red fox"]}, {"sigmas": [0.9, 0.4]}],
)
def test_changed_settings_run_the_key_pass_again(tmp_path, change):
    lora, calls = _lora(tmp_path), []
    key_cache.state_keys(lora, "tiny", SETTINGS, _key_pass(calls))
    changed = {**SETTINGS, **change}
    key_cache.state_keys(lora, "tiny", changed, _key_pass(calls, changed))
    assert calls == [1, 1]


def test_changed_family_runs_the_key_pass_again(tmp_path):
    lora, calls = _lora(tmp_path), []
    key_cache.state_keys(lora, "tiny", SETTINGS, _key_pass(calls))
    key_cache.state_keys(lora, "other", SETTINGS, _key_pass(calls))
    assert calls == [1, 1]


def test_broken_cache_file_is_a_miss(tmp_path, cache_folder):
    lora, calls = _lora(tmp_path), []
    key_cache.state_keys(lora, "tiny", SETTINGS, _key_pass(calls))
    (path,) = cache_folder.glob("*.safetensors")
    path.write_bytes(b"not a safetensors file")
    key_cache.state_keys(lora, "tiny", SETTINGS, _key_pass(calls))
    assert calls == [1, 1]


def test_cache_folder_is_under_the_comfyui_user_folder(tmp_path, monkeypatch):
    import types

    from kilora import host

    monkeypatch.setattr(key_cache, "CACHE_DIR", None)
    fp = types.SimpleNamespace(get_user_directory=lambda: str(tmp_path / "user"))
    monkeypatch.setattr(
        host,
        "services",
        lambda: types.SimpleNamespace(comfy=types.SimpleNamespace(folder_paths=fp)),
    )
    assert key_cache.cache_dir() == str(tmp_path / "user" / "kilora" / "key-cache")
