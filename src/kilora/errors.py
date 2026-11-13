# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
class KiLoRAError(Exception):
    pass


class IncompatibleAdapterSetError(KiLoRAError, ValueError):
    pass


class ModFormatError(KiLoRAError, ValueError):
    pass


class UnsupportedFeatureError(KiLoRAError):
    pass


class LoRAImportError(KiLoRAError, ValueError):
    pass
