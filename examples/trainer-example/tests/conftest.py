# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

here = Path(__file__).resolve()
sys.path.insert(0, str(here.parents[1] / "src"))
sys.path.insert(0, str(here.parents[2] / "editor-pullback" / "src"))
