# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import sys
from functools import partial

from kilora.plugin_support import check_trainer_version

from . import SUPPORTED_VERSIONS
from .network import KiLoRANetwork

NETWORK_TYPE = "kilora"
_BASE_PROCESS_MODULE = "jobs.process.BaseSDTrainProcess"
_NETWORK_NAME = "LoRASpecialNetwork"


def installed_version() -> str | None:
    module = sys.modules.get("version") or importlib.import_module("version")
    return getattr(module, "VERSION", None)


def make_process_class():
    trainer = importlib.import_module("extensions_built_in.sd_trainer.SDTrainer").SDTrainer

    class KiLoRATrainProcess(trainer):
        def run(self):
            version = check_trainer_version("AI Toolkit", installed_version(), SUPPORTED_VERSIONS)
            config = getattr(self, "network_config", None)
            if config is None or str(config.type).lower() != NETWORK_TYPE:
                return super().run()
            module = importlib.import_module(_BASE_PROCESS_MODULE)
            original = getattr(module, _NETWORK_NAME)
            setattr(module, _NETWORK_NAME, partial(KiLoRANetwork, tool_version=version))
            try:
                return super().run()
            finally:
                setattr(module, _NETWORK_NAME, original)

    return KiLoRATrainProcess
