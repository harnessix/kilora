# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
import pytest

from kilora.import_controller import sign_step


def test_repeated_sign_grows_and_caps_step():
    delta = 1.0
    sign = 0
    for _ in range(30):
        step, sign, delta = sign_step(2.0, sign, delta, 1.0)
        assert step == -delta
    assert delta == 8.0


def test_reversal_halves_and_skips():
    step, sign, delta = sign_step(-1.0, 1, 2.0, 1.0)
    assert (step, sign, delta) == (0.0, -1, 1.0)
    step, sign, delta = sign_step(1.0, sign, delta, 1.0)
    assert (step, sign, delta) == (0.0, 1, 0.5)


def test_invalid_input_is_rejected():
    with pytest.raises(ValueError):
        sign_step(float("nan"), 0, 1.0, 1.0)
