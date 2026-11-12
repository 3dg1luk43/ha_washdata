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
"""Issue #429: per-appliance anti-crease finalise ratio.

The finalise into ``STATE_ANTI_WRINKLE`` may only fire once the cycle has
reached a fraction of the matched profile's expected duration. That fraction was
hard-coded at 0.98.

This is a DIFFERENT gate from ``smart_termination_duration_ratio`` (#393): that
one gates Smart Termination, this one the finalise that absorbs the anti-crease
tumble tail. The #393 argument transfers unchanged - ``_expected_duration`` is
the profile's arithmetic mean, so on a sensor-dry program whose runtime follows
the load, about half of all runs are shorter than their own mean and the path
built for exactly that tail can never engage. The reporter measured 15 of 15
matched dryer runs at 0.38-0.92 of expected, all ending by fallback timeout a
median of 27 min after the last real activity.

Per-appliance rather than a new default: their heat-pump dryer has no comparable
tail and the fixed ratio costs it nothing.

Fast, pure-unit tests (no HA boot, no file I/O, no cycle_data replay).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from custom_components.ha_washdata import playground, ws_api
from custom_components.ha_washdata.const import (
    CONF_ANTI_CREASE_FINALIZE_RATIO,
    CONF_CURVE_PREROLL_SECONDS,
    CONF_DEVICE_TYPE,
    CONF_POWER_SENSOR,
    DEFAULT_ANTI_CREASE_FINALIZE_RATIO,
    DEVICE_TYPE_DISHWASHER,
    DEVICE_TYPE_DRYER,
    DOMAIN,
    STATE_ANTI_WRINKLE,
)
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
    TerminationReason,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_is_unchanged_at_098() -> None:
    """Nothing may move until a device is given its own value."""
    assert DEFAULT_ANTI_CREASE_FINALIZE_RATIO == 0.98
    cfg = CycleDetectorConfig(min_power=5.0, off_delay=60)
    assert cfg.anti_crease_finalize_ratio == DEFAULT_ANTI_CREASE_FINALIZE_RATIO


def test_it_is_not_the_smart_termination_ratio() -> None:
    """The two gates are independent; #393 did not make this one configurable."""
    cfg = CycleDetectorConfig(
        min_power=5.0,
        off_delay=60,
        smart_termination_duration_ratio=0.85,
        anti_crease_finalize_ratio=0.75,
    )
    assert cfg.smart_termination_duration_ratio == 0.85
    assert cfg.anti_crease_finalize_ratio == 0.75


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------


def _dt(offset_seconds: float) -> datetime:
    # tz-aware throughout: STATE_ANTI_WRINKLE's own timeout compares against
    # _state_enter_time, and the repo rule is that every time calculation is
    # dt-aware (matching the other anti-crease tests).
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(
        seconds=offset_seconds
    )


def _crease_guard(t: int) -> float:
    """The #429 reporter's crease guard: one 155-275 W tumble of about 20 s every
    3-4 min (here every 210 s) on a 5 W base, above the 4 W stop threshold so the
    off-delay path never closes it (the #296 shape the finalise exists for)."""
    return 200.0 if (t - 7000) % 210 < 30 else 5.0


def _cool_down(_t: int) -> float:
    """A drum that turns without a pause at 200 W after the heating (#514)."""
    return 200.0


def _run_dryer_with_tumble_tail(
    ratio: float,
    tail_end: int = 8000,
    anti_wrinkle_max_duration: float = 60.0,
    tail=_crease_guard,
):
    """A confidently-matched dryer that stops heating at 70% of its expected
    duration and then sits in its crease-guard tumble tail (``tail``).

    Every reading is below ``anti_wrinkle_max_power`` (400 W, so it reads as
    tail, not work) and above ``stop_threshold_w``. The run is fed to
    ``tail_end``; 8000 is 80% of expected, i.e. past a 0.75 ratio but nowhere
    near 0.98. Returns the detector and the ``on_cycle_end`` mock.
    """
    expected = 10000.0
    matcher = Mock(
        side_effect=lambda readings: ("Schranktrocken+", 0.9, expected, "Drying", False)
    )
    on_end = Mock()
    cfg = CycleDetectorConfig(
        min_power=5.0,
        off_delay=60,
        device_type=DEVICE_TYPE_DRYER,
        completion_min_seconds=600,
        start_duration_threshold=0.0,
        start_energy_threshold=0.0,
        start_threshold_w=6.0,
        stop_threshold_w=4.0,
        anti_wrinkle_enabled=True,
        anti_wrinkle_max_power=400.0,
        anti_wrinkle_max_duration=anti_wrinkle_max_duration,
        anti_crease_finalize_ratio=ratio,
    )
    det = CycleDetector(
        config=cfg, on_state_change=Mock(), on_cycle_end=on_end, profile_matcher=matcher
    )

    det.process_reading(1000.0, _dt(0))
    det.process_reading(1000.0, _dt(30))
    det.process_reading(1000.0, _dt(60))
    assert det.matched_profile == "Schranktrocken+"
    assert det._expected_duration == expected

    # Heat phase to 70% of expected.
    for t in range(90, 7000, 30):
        det.process_reading(1000.0, _dt(t))
    for t in range(7000, tail_end, 30):
        det.process_reading(tail(t), _dt(t))
    return det, on_end


def test_tuned_ratio_recognises_the_tail_the_default_sits_through() -> None:
    """The reported case: 0.75 finalises, the shipped 0.98 waits for the timeout."""
    det_low, end_low = _run_dryer_with_tumble_tail(0.75, tail_end=7600)
    assert det_low.state == STATE_ANTI_WRINKLE
    assert end_low.called
    cycle = end_low.call_args[0][0]
    assert cycle.get("termination_reason") == TerminationReason.SMART
    assert cycle["duration"] < 9800, "finalised well short of 98% of expected"

    # Shipped default: still running at 80% of expected, exactly as before - the
    # tail will sit here until the fallback timeout.
    det_hi, end_hi = _run_dryer_with_tumble_tail(0.98)
    assert det_hi.state != STATE_ANTI_WRINKLE
    assert not end_hi.called


# ---------------------------------------------------------------------------
# The coupling the reporter measured: an earlier finalise EXPOSES what follows the
# heating. A lowered ratio opens the gate while a dryer may still be in its
# cool-down, the drum turning without a pause for minutes under
# anti_wrinkle_max_power (#514).
# ---------------------------------------------------------------------------


def test_a_continuous_tumble_is_not_finalised_into_anti_wrinkle() -> None:
    """Register item 393a met this as a fragment: the finalise fired inside a
    continuous drum stretch and the rest of it opened a second cycle (the #429
    reporter saw it twice, 32 Wh and 13 Wh); the tail floor then absorbed it.
    Since #514 a dryer's finalise waits for the first rest, so the stretch stays
    in the cycle and nothing is left to absorb."""
    def cool_then_guard(t: int) -> float:
        return _cool_down(t) if t < 8000 else _crease_guard(t)

    det, on_end = _run_dryer_with_tumble_tail(0.75, tail_end=8000, tail=cool_then_guard)
    assert not on_end.called, "past 0.75 x expected, but still in the cool-down"
    assert det.state != STATE_ANTI_WRINKLE

    det, on_end = _run_dryer_with_tumble_tail(0.75, tail_end=8900, tail=cool_then_guard)
    assert on_end.call_count == 1, "finalised once, after the cool-down"
    assert on_end.call_args[0][0]["duration"] >= 7950, "the cool-down is in the cycle"
    assert det.state == STATE_ANTI_WRINKLE


def test_raising_max_duration_keeps_the_tail_absorbed() -> None:
    """The reporter's own fix: raise the pulse limit past the tumble length. A
    600 s tumble after the finalise stays attached with a 900 s limit."""
    def guard_then_long_tumble(t: int) -> float:
        return 200.0 if 7800 <= t < 8400 else _crease_guard(t)

    det, on_end = _run_dryer_with_tumble_tail(
        0.75, tail_end=8600, anti_wrinkle_max_duration=900.0,
        tail=guard_then_long_tumble,
    )
    assert on_end.call_count == 1
    assert on_end.call_args[0][0]["duration"] < 7800, "finalised in the crease guard"
    assert det.state == STATE_ANTI_WRINKLE, "the whole tail stays attached"


def test_gate_respects_the_ratio_boundary() -> None:
    """The gate must open at the configured fraction, not near it."""
    det, _ = _run_dryer_with_tumble_tail(0.98)
    # 0.79 of expected: still short of the configured 0.80 gate.
    det.config.anti_crease_finalize_ratio = 0.80
    assert det._anticrease_gate_open(_dt(7900)) is False
    assert det._anticrease_gate_open(_dt(8100)) is True


def test_an_out_of_range_stored_ratio_is_held_to_the_documented_minimum() -> None:
    """Only ``ws_set_options`` clamps; the detector must not trust the stored value.

    ``import_config`` strips nulls only, a selective import writes numbers through,
    and the Playground sanitizer just casts to float, so 0.0 can reach the config.
    ``current_duration < expected * 0.0`` is False for every non-negative duration,
    which removes the past-expected discriminator entirely and lets the gate open on
    a mid-wash trough.
    """
    det, _ = _run_dryer_with_tumble_tail(0.98)

    det.config.anti_crease_finalize_ratio = 0.0
    # 10% of expected: nowhere near any legal ratio, so the gate must stay shut.
    assert det._anticrease_gate_open(_dt(1000)) is False
    # The floor is 0.5, so half-way through is still short.
    assert det._anticrease_gate_open(_dt(4900)) is False
    assert det._anticrease_gate_open(_dt(5100)) is True

    # Above the range is held down to 1.0 rather than pushing the gate past the end.
    det.config.anti_crease_finalize_ratio = 4.0
    assert det._anticrease_gate_open(_dt(9900)) is False
    assert det._anticrease_gate_open(_dt(10100)) is True


def test_default_keeps_the_pre_429_behaviour_exactly() -> None:
    """An install that never touches the setting must be byte-identical."""
    det_default, end_default = _run_dryer_with_tumble_tail(
        DEFAULT_ANTI_CREASE_FINALIZE_RATIO
    )
    assert not end_default.called
    assert det_default.state != STATE_ANTI_WRINKLE


# ---------------------------------------------------------------------------
# ws_set_options validation: clamp to [0.50, 1.00]; drop empty/invalid so the
# default applies again.
# ---------------------------------------------------------------------------


def _entry(options: dict) -> MagicMock:
    entry = MagicMock()
    entry.entry_id = "e1"
    entry.data = {CONF_POWER_SENSOR: "sensor.power"}
    entry.options = options
    return entry


def _hass() -> tuple[MagicMock, MagicMock]:
    manager = MagicMock()
    manager.profile_store.async_record_settings_changes = AsyncMock()
    hass = MagicMock()
    hass.data = {DOMAIN: {"e1": manager}}
    return hass, manager


async def _set_options(entry: MagicMock, hass: MagicMock, options: dict) -> dict:
    ws_fn = ws_api.ws_set_options.__wrapped__
    with patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_fn(hass, MagicMock(), {"id": 1, "entry_id": "e1", "options": options})
    return hass.config_entries.async_update_entry.call_args.kwargs["options"]


async def test_in_range_value_is_stored_verbatim() -> None:
    saved = await _set_options(
        _entry({}), _hass()[0], {CONF_ANTI_CREASE_FINALIZE_RATIO: 0.75}
    )
    assert saved[CONF_ANTI_CREASE_FINALIZE_RATIO] == 0.75


@pytest.mark.parametrize(
    ("submitted", "stored"),
    [(1.5, 1.0), (0.1, 0.5), (float("inf"), None), (float("nan"), None)],
)
async def test_out_of_range_is_clamped_or_dropped(submitted, stored) -> None:
    saved = await _set_options(
        _entry({}), _hass()[0], {CONF_ANTI_CREASE_FINALIZE_RATIO: submitted}
    )
    if stored is None:
        assert CONF_ANTI_CREASE_FINALIZE_RATIO not in saved
    else:
        assert saved[CONF_ANTI_CREASE_FINALIZE_RATIO] == stored


@pytest.mark.parametrize("junk", ["", None, "abc"])
async def test_empty_or_non_numeric_drops_the_key(junk) -> None:
    """Dropping the key restores the default; coercing to it would be a value."""
    saved = await _set_options(
        _entry({CONF_ANTI_CREASE_FINALIZE_RATIO: 0.75}),
        _hass()[0],
        {CONF_ANTI_CREASE_FINALIZE_RATIO: junk},
    )
    assert CONF_ANTI_CREASE_FINALIZE_RATIO not in saved


async def test_an_oversized_integer_drops_the_key_instead_of_failing_the_save() -> None:
    """json parses an integer literal of any length; float() on one raises.

    OverflowError is not a ValueError, so it escaped the guard, and the
    ``async_response`` wrapper turns it into ERR_UNKNOWN_ERROR - the whole
    settings save fails instead of this one key falling back to its default
    (register item 194, same shape).
    """
    saved = await _set_options(
        _entry({CONF_ANTI_CREASE_FINALIZE_RATIO: 0.75}),
        _hass()[0],
        {CONF_ANTI_CREASE_FINALIZE_RATIO: 10**400, CONF_CURVE_PREROLL_SECONDS: 10**400},
    )
    assert CONF_ANTI_CREASE_FINALIZE_RATIO not in saved
    assert CONF_CURVE_PREROLL_SECONDS not in saved


async def test_the_two_ratios_do_not_clobber_each_other() -> None:
    """Both submitted at once: each must be validated on its own."""
    from custom_components.ha_washdata.const import (
        CONF_SMART_TERMINATION_DURATION_RATIO,
    )

    saved = await _set_options(
        _entry({}),
        _hass()[0],
        {
            CONF_ANTI_CREASE_FINALIZE_RATIO: 0.75,
            CONF_SMART_TERMINATION_DURATION_RATIO: 0.85,
        },
    )
    assert saved[CONF_ANTI_CREASE_FINALIZE_RATIO] == 0.75
    assert saved[CONF_SMART_TERMINATION_DURATION_RATIO] == 0.85


# ---------------------------------------------------------------------------
# Playground: the field is always a real float, so effective_settings never
# skips it and the sim reproduces the live gate.
# ---------------------------------------------------------------------------


def test_effective_settings_surfaces_the_ratio() -> None:
    cfg = CycleDetectorConfig(
        min_power=5.0, off_delay=60, anti_crease_finalize_ratio=0.75
    )
    eff = playground.effective_settings(cfg, None)
    assert eff[CONF_ANTI_CREASE_FINALIZE_RATIO] == 0.75


def test_playground_base_config_resolves_the_default() -> None:
    """Deliberately NOT device-resolved - the safe value is per machine."""
    manager = MagicMock()
    manager.detector.config = None  # force the fallback build path
    entry = MagicMock()
    entry.data = {}
    for device_type in (DEVICE_TYPE_DRYER, DEVICE_TYPE_DISHWASHER):
        entry.options = {CONF_DEVICE_TYPE: device_type}
        cfg = ws_api._playground_base_config(manager, entry)
        assert cfg.anti_crease_finalize_ratio == DEFAULT_ANTI_CREASE_FINALIZE_RATIO


def test_playground_override_applies_the_ratio() -> None:
    """The Playground must be able to replay a candidate value before adopting it."""
    assert (
        playground._OVERRIDE_FIELD_MAP[CONF_ANTI_CREASE_FINALIZE_RATIO][0]
        == "anti_crease_finalize_ratio"
    )
