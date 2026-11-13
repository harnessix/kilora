# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
class KiLoRAExtension:
    uid = "kilora_trainer"
    name = "KiLoRA Trainer"

    @classmethod
    def get_process(cls):
        from .process import make_process_class

        return make_process_class()


AI_TOOLKIT_EXTENSIONS = [KiLoRAExtension]
