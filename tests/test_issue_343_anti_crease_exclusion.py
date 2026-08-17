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

"""Issue #343 (part 1): context-aware suggestions on anti-crease devices.

The stop/start-threshold suggestion is derived from the minimum active power in
the trace. On an anti-crease/anti-wrinkle device the stored trace keeps the
post-cycle tumble-pulse tail, whose ~3 W between-pulse baseline is the global
minimum, so the tuner proposes thresholds just above it and breaks end-detection.
Stop/start thresholds detect the MAIN cycle (the tail is governed by its own
anti_wrinkle_* settings), so for anti-crease-enabled eligible devices the tail is
now excluded from the min-active statistic: everything after the last sample that
reaches anti_wrinkle_max_power (a pulse above it ends anti-wrinkle, so the tail
never does). Gated + guarded so non-anti-crease devices are unaffected.
"""

from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata.suggestion_engine import SuggestionEngine
from custom_components.ha_washdata.const import (
    CONF_STOP_THRESHOLD_W, CONF_ANTI_WRINKLE_ENABLED, CONF_ANTI_WRINKLE_MAX_POWER,
    DEVICE_TYPE_DRYER, DEVICE_TYPE_DISHWASHER,
)

# Main phase (reaches 500 W, min active 200 W) then an anti-crease tail: 60 W tumble
# pulses with a ~3 W baseline between them. 14 points (>= the 10-point minimum).
_MAIN = [500.0, 400.0, 200.0, 200.0, 500.0, 200.0, 200.0, 450.0]
_TAIL = [3.0, 60.0, 3.0, 60.0, 3.0, 60.0]
_CYCLE = {
    "power_data": [[float(i * 5), p] for i, p in enumerate(_MAIN + _TAIL)],
    "start_time": None,
}


def _engine(device_type, options):
    eng = SuggestionEngine(MagicMock(), "e", MagicMock(), device_type)
    eng._entry_options = lambda: options
    return eng


def test_anti_crease_tail_excluded_for_dryer():
    """A dryer with anti-crease on: stop reflects the MAIN min (200 W), not 3 W."""
    eng = _engine(DEVICE_TYPE_DRYER, {CONF_ANTI_WRINKLE_ENABLED: True, CONF_ANTI_WRINKLE_MAX_POWER: 400.0})
    out = eng.run_simulation(_CYCLE)
    stop = out[CONF_STOP_THRESHOLD_W]["value"]
    # 200 W * 0.8 = 160, NOT 3 W * 0.8 = 2.4.
    assert stop == pytest.approx(160.0, abs=1.0), stop


def test_tail_kept_when_anti_crease_disabled():
    """Anti-crease off: behaviour unchanged, the 3 W baseline still drives it."""
    eng = _engine(DEVICE_TYPE_DRYER, {CONF_ANTI_WRINKLE_ENABLED: False, CONF_ANTI_WRINKLE_MAX_POWER: 400.0})
    out = eng.run_simulation(_CYCLE)
    stop = out[CONF_STOP_THRESHOLD_W]["value"]
    assert stop == pytest.approx(2.4, abs=0.5), stop


def test_tail_kept_for_ineligible_device_type():
    """A dishwasher is not an anti-crease device: no exclusion even if the flag is on."""
    eng = _engine(DEVICE_TYPE_DISHWASHER, {CONF_ANTI_WRINKLE_ENABLED: True, CONF_ANTI_WRINKLE_MAX_POWER: 400.0})
    out = eng.run_simulation(_CYCLE)
    stop = out[CONF_STOP_THRESHOLD_W]["value"]
    assert stop == pytest.approx(2.4, abs=0.5), stop


def test_no_high_power_sample_is_safe_noop():
    """If nothing reaches anti_wrinkle_max_power there is no identifiable tail, so
    the whole trace is used (no accidental over-exclusion)."""
    eng = _engine(DEVICE_TYPE_DRYER, {CONF_ANTI_WRINKLE_ENABLED: True, CONF_ANTI_WRINKLE_MAX_POWER: 9999.0})
    out = eng.run_simulation(_CYCLE)
    stop = out[CONF_STOP_THRESHOLD_W]["value"]
    assert stop == pytest.approx(2.4, abs=0.5), stop
