# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import copy
import gc

import pytest

from kilora import compute, ir
from kilora.ir.dispatch import active_overrides


def test_override_scope_restores_nested_and_exceptional_lifetimes():
    initial = compute._IR_ACTIVE_OVERRIDES.get()
    with ir.override_scope():
        assert compute._IR_ACTIVE_OVERRIDES.get() == initial + 1
        with active_overrides({}):
            assert compute._IR_ACTIVE_OVERRIDES.get() == initial + 2
        assert compute._IR_ACTIVE_OVERRIDES.get() == initial + 1
    assert compute._IR_ACTIVE_OVERRIDES.get() == initial
    with pytest.raises(RuntimeError), ir.override_scope():
        raise RuntimeError("test")
    assert compute._IR_ACTIVE_OVERRIDES.get() == initial


def test_set_copy_and_deletion_do_not_change_scope():
    initial = compute._IR_ACTIVE_OVERRIDES.get()
    setting = compute.LayerSet(
        router=None, keys_in=None, ir_parts={"guard": ir.PartOverride("default")}
    )
    duplicate = copy.copy(setting)
    del setting, duplicate
    gc.collect()
    assert compute._IR_ACTIVE_OVERRIDES.get() == initial
