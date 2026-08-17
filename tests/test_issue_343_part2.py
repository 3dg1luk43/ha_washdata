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

"""Issue #343 (part 2): three remaining gaps after the initial stop/start fix.

Gap A — min_power in generate_detection_suggestions() was not stripped;
  anti-crease baseline dragged the p05 down and proposed a noise gate below
  the real draw.

Gap B — off_delay cadence fallback (p95 * 5) was emitted even when
  anti_wrinkle_enabled; the result often exceeds the burst interval,
  resetting the end timer on every tumble burst so the cycle never ends.

Gap C — off_delay pause-based path counted inter-burst anti-crease pauses
  (the ~3 W quiet stretches between 50-145 W bursts) as genuine intra-cycle
  pauses, inflating p95_pause and producing the same over-long off_delay.

Gap D — run_simulation() hardcoded end_energy_threshold = 0.05 Wh regardless
  of cycle energy; on anti-crease devices the baseline power × off_delay
  window exceeds 0.05 Wh so the end gate never fires.
"""

from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata.suggestion_engine import SuggestionEngine
from custom_components.ha_washdata.const import (
    CONF_STOP_THRESHOLD_W,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_END_ENERGY_THRESHOLD,
    CONF_ANTI_WRINKLE_ENABLED,
    CONF_ANTI_WRINKLE_MAX_POWER,
    DEVICE_TYPE_DRYER,
    DEVICE_TYPE_WASHING_MACHINE,
    DEVICE_TYPE_DISHWASHER,
)

# ─── Shared fixture helpers ───────────────────────────────────────────────────

# Canonical anti-crease cycle: main phase at 400-600 W (min active 200 W),
# ending on a high sample so the boundary with the tail is unambiguous.
# Then tail: ~3 W baseline + 60 W tumble bursts (all below 400 W ceiling).
_MAIN_POWERS = [600.0, 400.0, 500.0, 200.0, 200.0, 400.0, 200.0, 500.0, 600.0, 400.0]
_TAIL_POWERS = [3.0, 60.0, 3.0, 60.0, 3.0, 60.0, 3.0]

_ANTI_CREASE_OPTIONS = {
    CONF_ANTI_WRINKLE_ENABLED: True,
    CONF_ANTI_WRINKLE_MAX_POWER: 400.0,
}


def _make_cycle(powers, dt=5.0, start="2026-01-01T00:00:00", **extra):
    """Build a minimal cycle dict with offset-format power_data."""
    pd = [[float(i) * dt, p] for i, p in enumerate(powers)]
    return {
        "power_data": pd,
        "start_time": start,
        "status": "completed",
        "profile_name": "Cotton 60",
        "duration": len(powers) * dt,
        "sampling_interval": dt,
        **extra,
    }


def _anti_crease_cycle(**kw):
    return _make_cycle(_MAIN_POWERS + _TAIL_POWERS, **kw)


def _engine(device_type=DEVICE_TYPE_DRYER, options=None):
    eng = SuggestionEngine(MagicMock(), "e", MagicMock(), device_type)
    eng._entry_options = lambda: (options if options is not None else _ANTI_CREASE_OPTIONS)
    return eng


# ─── Gap A: _strip_anti_crease_readings() ────────────────────────────────────

class TestStripAntiCreaseReadings:
    """The new time-domain strip helper."""

    def test_strips_tail_on_eligible_device(self):
        """Returns only up to the last sample >= max_power."""
        eng = _engine()
        readings = [(float(i), p) for i, p in enumerate(_MAIN_POWERS + _TAIL_POWERS)]
        stripped = eng._strip_anti_crease_readings(readings)
        # Last main-phase sample >= 400 W is at index len(_MAIN_POWERS)-1
        assert len(stripped) == len(_MAIN_POWERS)
        assert all(t <= (len(_MAIN_POWERS) - 1) for t, _ in stripped)

    def test_noop_for_ineligible_device_type(self):
        eng = _engine(device_type=DEVICE_TYPE_DISHWASHER)
        readings = [(float(i), p) for i, p in enumerate(_MAIN_POWERS + _TAIL_POWERS)]
        assert eng._strip_anti_crease_readings(readings) == readings

    def test_noop_when_anti_crease_disabled(self):
        eng = _engine(options={CONF_ANTI_WRINKLE_ENABLED: False, CONF_ANTI_WRINKLE_MAX_POWER: 400.0})
        readings = [(float(i), p) for i, p in enumerate(_MAIN_POWERS + _TAIL_POWERS)]
        assert eng._strip_anti_crease_readings(readings) == readings

    def test_noop_when_no_sample_reaches_ceiling(self):
        """No sample >= very high ceiling → can't identify main phase → no strip."""
        eng = _engine(options={CONF_ANTI_WRINKLE_ENABLED: True, CONF_ANTI_WRINKLE_MAX_POWER: 9999.0})
        readings = [(float(i), p) for i, p in enumerate(_MAIN_POWERS + _TAIL_POWERS)]
        assert eng._strip_anti_crease_readings(readings) == readings

    def test_empty_readings_safe(self):
        eng = _engine()
        assert eng._strip_anti_crease_readings([]) == []


# ─── Gap A continued: min_power in generate_detection_suggestions() ──────────

class TestMinPowerDetectionSuggestion:
    """min_power must exclude the anti-crease tail (#343 gap A)."""

    @staticmethod
    def _long_anti_crease_cycle():
        """~270 s main phase (ramps up to 200-600 W) + anti-crease tail.

        Lead-in at 100 W (below _CLEAN_HIGH_START_RATIO * 600 W = 300 W so
        'started mid-cycle' is not triggered). 100 W * 0.4 = 40 W → capped at
        10 W, making the cap observable in the assertion.
        """
        main = [100.0, 150.0] + [200.0, 400.0, 500.0, 600.0] * 13 + [400.0, 200.0]
        tail = [3.0, 60.0, 3.0, 60.0, 3.0, 60.0, 3.0]
        return _make_cycle(main + tail, dt=5.0)

    def _make_store_with_cycles(self, n=6):
        """Return a mock profile_store whose get_past_cycles() yields n valid anti-crease cycles."""
        store = MagicMock()
        store.get_past_cycles.return_value = [self._long_anti_crease_cycle() for _ in range(n)]
        return store

    def test_min_power_excludes_tail_on_anti_crease_device(self):
        """p05 min-active should reflect the MAIN phase (200 W), not the 3 W baseline."""
        eng = SuggestionEngine(MagicMock(), "e", self._make_store_with_cycles(), DEVICE_TYPE_DRYER)
        eng._entry_options = lambda: _ANTI_CREASE_OPTIONS
        result = eng.generate_detection_suggestions()
        # min_power = max(p05 * 0.4, 1.0) capped at 10.0
        # With tail stripped, p05 min_active ≥ 200 W → p05 * 0.4 = 80 W → capped at 10.0
        # Without strip, p05 min_active ≈ 3 W → p05 * 0.4 = 1.2 W → 1.2
        assert CONF_MIN_POWER in result
        assert result[CONF_MIN_POWER]["value"] == pytest.approx(10.0, abs=0.1)

    def test_min_power_unaffected_on_non_anti_crease_device(self):
        """Dishwasher: no strip, tail baseline still drives the suggestion."""
        store = MagicMock()
        store.get_past_cycles.return_value = [_anti_crease_cycle() for _ in range(6)]
        eng = SuggestionEngine(MagicMock(), "e", store, DEVICE_TYPE_DISHWASHER)
        eng._entry_options = lambda: _ANTI_CREASE_OPTIONS
        result = eng.generate_detection_suggestions()
        if CONF_MIN_POWER in result:
            # Without strip the p05 is 3 W → 3*0.4=1.2, clamped to 1.0
            assert result[CONF_MIN_POWER]["value"] == pytest.approx(1.0, abs=0.5)


# ─── Gap B: cadence-fallback off_delay suppressed ────────────────────────────

class TestOffDelayCadenceSuppressed:
    """When anti_wrinkle_enabled and no pause data, cadence off_delay is withheld."""

    def _eng_with_empty_cycles(self):
        eng = _engine()
        eng.profile_store.get_past_cycles.return_value = []
        eng._current_stop_threshold = lambda opts: 4.0
        return eng

    def test_cadence_off_delay_absent_when_anti_crease_on(self):
        """No CONF_OFF_DELAY suggestion when cadence fallback would fire on anti-crease device."""
        eng = self._eng_with_empty_cycles()
        result = eng.generate_operational_suggestions(p95_dt=81.3, median_dt=30.0)
        assert CONF_OFF_DELAY not in result

    def test_cadence_off_delay_present_when_anti_crease_off(self):
        """Normal (non-anti-crease) path still emits the cadence-based off_delay."""
        eng = _engine(options={CONF_ANTI_WRINKLE_ENABLED: False, CONF_ANTI_WRINKLE_MAX_POWER: 400.0})
        eng.profile_store.get_past_cycles.return_value = []
        eng._current_stop_threshold = lambda opts: 4.0
        result = eng.generate_operational_suggestions(p95_dt=81.3, median_dt=30.0)
        assert CONF_OFF_DELAY in result

    def test_cadence_off_delay_present_for_ineligible_device(self):
        """Dishwasher is not anti-crease eligible; cadence path must run normally."""
        eng = _engine(device_type=DEVICE_TYPE_DISHWASHER)
        eng.profile_store.get_past_cycles.return_value = []
        eng._current_stop_threshold = lambda opts: 4.0
        result = eng.generate_operational_suggestions(p95_dt=81.3, median_dt=30.0)
        assert CONF_OFF_DELAY in result


# ─── Gap C: pause-based off_delay strips anti-crease inter-burst gaps ─────────

class TestOffDelayPauseBasedStrips:
    """_suggest_off_delay_from_pauses must not count anti-crease inter-burst gaps."""

    def _make_anti_crease_cycles(self, n=7):
        """Cycles whose tail has long quiet gaps (180 s) that look like pauses."""
        # Each cycle: main phase at 500 W (30 samples × 5 s = 150 s), then
        # anti-crease tail: 5 s at 3 W, 5 s at 80 W, 180 s at 3 W, 5 s at 80 W, 5 s at 3 W.
        main = [500.0] * 30
        # Tail with a 180-second quiet gap between bursts (dt=5s → 36 samples at 3W)
        tail = [3.0, 80.0] + [3.0] * 36 + [80.0, 3.0]
        return [_make_cycle(main + tail, dt=5.0) for _ in range(n)]

    def test_pause_based_off_delay_stays_sane_with_anti_crease(self):
        """p95 pause from stripped readings is well below the anti-crease burst gap."""
        eng = _engine(options={CONF_ANTI_WRINKLE_ENABLED: True, CONF_ANTI_WRINKLE_MAX_POWER: 200.0})
        cycles = self._make_anti_crease_cycles(7)
        stop_thr = 4.0
        device_floor = 150
        result = eng._suggest_off_delay_from_pauses(cycles, stop_thr, device_floor)
        # With stripping: no resumed pauses in the main phase → returns None
        # (falls back to cadence path, which is then suppressed by gap B fix)
        # With anti-crease tail included: the 180 s quiet gaps would inflate off_delay
        # to > 240 s easily. After strip there are no intra-main pauses → None or floor.
        if result is not None:
            suggested_off_delay = result[0]
            # Must be at or near floor, not inflated by 180s anti-crease gaps
            assert suggested_off_delay <= device_floor + 120, (
                f"off_delay {suggested_off_delay} is too high — "
                "anti-crease inter-burst gaps were counted as real pauses"
            )

    def test_pause_based_returns_none_for_no_main_phase_pauses(self):
        """Main phase has no pauses → _suggest_off_delay_from_pauses returns None."""
        eng = _engine()
        # All clean cycles with continuous main phase (no dips) + anti-crease tail
        cycles = [_anti_crease_cycle() for _ in range(7)]
        result = eng._suggest_off_delay_from_pauses(cycles, 4.0, 150)
        # With stripped readings the main phase is smooth, no resumed pauses
        assert result is None


# ─── Gap D: run_simulation() no longer suggests a hardcoded end_energy ───────

class TestRunSimulationEndEnergy:
    """run_simulation must not suggest a context-free end_energy_threshold (#343 gap D)."""

    _CYCLE = {
        "power_data": [[float(i * 5), p] for i, p in enumerate(_MAIN_POWERS + _TAIL_POWERS)],
        "start_time": None,
    }

    def test_end_energy_not_in_single_cycle_suggestion(self):
        """Hardcoded 0.05 is removed; batch path handles it when data is sufficient."""
        eng = _engine()
        result = eng.run_simulation(self._CYCLE)
        assert CONF_END_ENERGY_THRESHOLD not in result, (
            f"run_simulation should not suggest end_energy; got {result.get(CONF_END_ENERGY_THRESHOLD)}"
        )

    def test_stop_start_still_present(self):
        """The stop/start suggestions from the original fix remain intact."""
        eng = _engine()
        result = eng.run_simulation(self._CYCLE)
        assert CONF_STOP_THRESHOLD_W in result
