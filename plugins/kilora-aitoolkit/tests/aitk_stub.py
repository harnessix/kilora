# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import sys
import types

import torch


class StubLoRANetwork:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class NetworkConfig:
    def __init__(self, type="lora", linear=4, linear_alpha=4, network_kwargs=None):
        self.type = type
        self.linear = linear
        self.linear_alpha = linear_alpha
        self.network_kwargs = dict(network_kwargs or {})


class BaseModel:
    arch = "zimage"


def install(version="0.13.39"):
    mod_version = types.ModuleType("version")
    mod_version.VERSION = version
    base = types.ModuleType("jobs.process.BaseSDTrainProcess")
    base.LoRASpecialNetwork = StubLoRANetwork
    trainer_mod = types.ModuleType("extensions_built_in.sd_trainer.SDTrainer")

    class SDTrainer:
        def __init__(self, network_config, unet, data, save_path, steps=4, lr=1e-2):
            self.network_config = network_config
            self.unet = unet
            self.data = data
            self.save_path = save_path
            self.steps = steps
            self.lr = lr
            self.network = None

        def run(self):
            cfg = self.network_config
            network_class = sys.modules["jobs.process.BaseSDTrainProcess"].LoRASpecialNetwork
            net = network_class(
                text_encoder=None,
                unet=self.unet,
                lora_dim=cfg.linear,
                multiplier=1.0,
                alpha=cfg.linear_alpha,
                train_unet=True,
                train_text_encoder=False,
                conv_lora_dim=None,
                conv_alpha=None,
                is_sdxl=False,
                is_v2=False,
                is_v3=False,
                is_pixart=False,
                is_auraflow=False,
                is_flux=False,
                is_lumina2=False,
                is_ssd=False,
                is_vega=False,
                dropout=None,
                use_text_encoder_1=True,
                use_text_encoder_2=True,
                use_bias=False,
                is_lorm=False,
                network_config=cfg,
                network_type=cfg.type,
                transformer_only=True,
                is_transformer=True,
                base_model=BaseModel(),
                **cfg.network_kwargs,
            )
            self.network = net
            if isinstance(net, StubLoRANetwork):
                return net
            net.force_to("cpu", dtype=torch.float32)
            net._update_torch_multiplier()
            net.apply_to(None, self.unet, False, True)
            net.prepare_grad_etc(None, self.unet)
            params = net.prepare_optimizer_params(
                text_encoder_lr=self.lr, unet_lr=self.lr, default_lr=self.lr
            )
            opt = torch.optim.AdamW(params)
            x, target = self.data
            for _ in range(self.steps):
                net.multiplier = [1.0] * x.shape[0]
                with net:
                    loss = (self.unet(x) - target).pow(2).mean()
                loss.backward()
                opt.step()
                opt.zero_grad()
            net.multiplier = 0.0
            with torch.no_grad(), net:
                self.prior = self.unet(x)
            with torch.no_grad():
                self.off = self.unet(x)
            net.multiplier = 1.0
            net.save_weights(
                self.save_path, dtype=torch.float32, metadata={}, extra_state_dict=None
            )
            return net

    trainer_mod.SDTrainer = SDTrainer
    for name, module in {
        "version": mod_version,
        "jobs": types.ModuleType("jobs"),
        "jobs.process": types.ModuleType("jobs.process"),
        "jobs.process.BaseSDTrainProcess": base,
        "extensions_built_in": types.ModuleType("extensions_built_in"),
        "extensions_built_in.sd_trainer": types.ModuleType("extensions_built_in.sd_trainer"),
        "extensions_built_in.sd_trainer.SDTrainer": trainer_mod,
    }.items():
        sys.modules[name] = module
    return base


def uninstall():
    for name in [
        n for n in sys.modules if n == "version" or n.startswith(("jobs", "extensions_built_in"))
    ]:
        del sys.modules[name]
