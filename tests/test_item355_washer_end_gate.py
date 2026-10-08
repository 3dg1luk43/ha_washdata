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
24.74 -> 16.92 min, early ends 0.00%, splits unchanged, dishwashers identical.
(The first cut of this read 13.69, but ~3.2 min of that came from applying the
0.90 ratio to an ambiguity-RAISED bar, which defeats the bar's purpose and was
corrected in round 33 - see register item 364.)
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
    # The gate reads its bar from `_fallback_shortening_bar` (shared with the
    # item-469b ENDING hold), which resolves the ratio per device.
    assert "if _bar_s is not None and _elapsed >= _bar_s:" in src
    bar = src[src.index("def _fallback_shortening_bar("):]
    bar = bar[: bar.index("def ambiguous_ending_match_defers(")]
    assert "resolve_end_gate_late_ratio(self._config.device_type)" in bar
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


def test_the_device_ratio_never_discounts_a_raised_ambiguity_bar() -> None:
    """Register item 364, found by CodeRabbit round 33 on PR #448.

    When the match is ambiguous and a longer candidate exists, `_bar` stops being
    the expected duration and becomes the LONGEST PLAUSIBLE programme - raised
    precisely to say "a much longer look-alike is still on the table, so past the
    expected end does not mean done". Applying the washer's 0.90 to that would
    shorten the wait 10% before the candidate it represents could even finish,
    and on the shipped washer defaults that drops the wait from `min_off_gap` to
    `max(off_delay, 300)`: one quiet interval from finalising mid-programme and
    recording the rest as a second cycle, which is #288.

    Item 355's measurement was taken against the expected duration and says
    nothing about the raised case, so a raised bar keeps the original 1.05.
    Asserted on `CycleDetector._fallback_shortening_bar`, the gate's bar since
    register item 469 (it used to be read off the source).
    """
    from custom_components.ha_washdata.const import END_GATE_LATE_RATIO
    from custom_components.ha_washdata.cycle_detector import (
        CycleDetector,
        CycleDetectorConfig,
    )

    det = CycleDetector(
        CycleDetectorConfig(min_power=2.0, off_delay=180, device_type="washing_machine"),
        lambda a, b: None,
        lambda c: None,
    )
    washer = resolve_end_gate_late_ratio("washing_machine")
    assert washer < END_GATE_LATE_RATIO
    # Clear match: the device's discounted ratio on the expected duration.
    assert det._fallback_shortening_bar(5000.0, False, 9000.0) == pytest.approx(washer * 5000.0)
    # Ambiguous with a longer look-alike: raised to it, at the undiscounted 1.05.
    assert det._fallback_shortening_bar(5000.0, True, 9000.0) == pytest.approx(
        END_GATE_LATE_RATIO * 9000.0
    )
    # Ambiguous, nothing longer: not raised, the device ratio applies.
    assert det._fallback_shortening_bar(5000.0, True, 4000.0) == pytest.approx(washer * 5000.0)
    # Ambiguous with no candidate durations at all: blocked.
    assert det._fallback_shortening_bar(5000.0, True, 0.0) is None
