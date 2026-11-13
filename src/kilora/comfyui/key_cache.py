# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import sys

from .. import key_cache as _key_cache

sys.modules[__name__] = _key_cache
