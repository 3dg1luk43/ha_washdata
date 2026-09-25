# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""Register item 355: the item-306 ENDING shortening was out of reach for washers.

It fires once a matched run is past `END_GATE_LATE_RATIO x` its programme's own
expected length. A washing machine's programme is load-adaptive, so a run sits
BELOW its profile's mean about half the time by definition: measured over the
`end_gate_eval.py` corpus the median washer reaches only 0.83 of the bar before
it stops, against 1.01 for a dishwasher, and the rule reached 6.8% of washer
cycles. That, not match ambiguity, is why washing machines waited out the full
`min_off_gap` - a median 24.74 min after the last activity against 6.89 for a
dishwasher.

Scoped to washers rather than lowered globally because every early end in the
global sweep was a DISHWASHER. At 0.90, washers only: median end lag
24.74 -> 13.69 min, early ends 0.00%, splits unchanged, dishwashers identical.
"""
from __future__ import annotations

import pytest

from custom_components.ha_washdata.const import (
    DEVICE_TYPE_DISHWASHER,
    DEVICE_TYPE_DRYER,
    DEVICE_TYPE_WASHER_DRYER,
    DEVICE_TYPE_WASHING_MACHINE,
    END_GATE_LATE_RATIO,
    resolve_end_gate_late_ratio,
)


def test_washers_get_a_reachable_bar() -> None:
    assert resolve_end_gate_late_ratio(DEVICE_TYPE_WASHING_MACHINE) == pytest.approx(0.90)
    assert resolve_end_gate_late_ratio(DEVICE_TYPE_WASHER_DRYER) == pytest.approx(0.90)


def test_every_other_device_type_is_unchanged() -> None:
    """The one measured cost of lowering this globally was a dishwasher closing
    5.58 min before its last activity, so only the class that showed zero early
    ends at any ratio moves."""
    for dt in (DEVICE_TYPE_DISHWASHER, DEVICE_TYPE_DRYER, "generic", "other", ""):
        assert resolve_end_gate_late_ratio(dt) == pytest.approx(END_GATE_LATE_RATIO)
        assert resolve_end_gate_late_ratio(dt) == pytest.approx(1.05)


def test_the_gate_resolves_per_device_not_from_the_module_constant() -> None:
    """The detector must call the resolver. Rebinding the imported copy in
    `cycle_detector` used to be how `end_gate_eval --no-shortening` disabled the
    rule, and a gate that still read that name would make the baseline arm lie.
    """
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "cycle_detector.py"
    ).read_text()
    gate = src[src.index("if not _blocked and _elapsed >="):][:120]
    assert "_late_ratio" in gate
    assert "END_GATE_LATE_RATIO * _bar" not in src


def test_the_baseline_arm_of_the_harness_still_disables_the_rule() -> None:
    """`--no-shortening` is the pre-306 arm every A/B in the register rests on.
    Since the gate resolves through `const`, patching `cycle_detector` no longer
    reaches it - the harness has to patch both names on `const`."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1] / "devtools" / "end_gate_eval.py"
    ).read_text()
    assert "_const.END_GATE_LATE_RATIO = 1e9" in src
    assert "_const.END_GATE_LATE_RATIO_BY_DEVICE = {}" in src
    assert "_cd.END_GATE_LATE_RATIO = 1e9" not in src
