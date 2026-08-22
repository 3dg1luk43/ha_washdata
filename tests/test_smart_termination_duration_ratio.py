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
"""Issue #393: per-appliance Smart-Termination duration ratio.

Smart Termination may only end a detected cycle early once it has reached a
fraction of the matched profile's expected (mean) duration. That fraction was
hard-coded (0.98, dishwasher 0.99). Because ``_expected_duration`` is the
profile's arithmetic mean, ~half of the cycles of a load-/temperature-dependent
appliance are shorter than their own mean and can never take the fast path.

This makes the fraction a per-device option, resolved to the device-type default
in the config builder (0.99 dishwasher / 0.98 other) so the field always carries
a real float - never ``None`` - which the playground's ``effective_settings``
relies on. The dishwasher pump-out relief (0.90) is combined via ``min()`` so the
option can only ever loosen the gate, never tighten it.

Fast, pure-unit tests (no HA boot, no file I/O, no cycle_data replay).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from custom_components.ha_washdata import playground, ws_api
from custom_components.ha_washdata.const import (
    CONF_DEVICE_TYPE,
    CONF_POWER_SENSOR,
    CONF_SMART_TERMINATION_DURATION_RATIO,
    DEFAULT_SMART_TERMINATION_DURATION_RATIO,
    DEFAULT_SMART_TERMINATION_DURATION_RATIO_BY_DEVICE,
    DEVICE_TYPE_DISHWASHER,
    DEVICE_TYPE_WASHING_MACHINE,
    DOMAIN,
    STATE_FINISHED,
    STATE_OFF,
)
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def test_scalar_default_is_098():
    assert DEFAULT_SMART_TERMINATION_DURATION_RATIO == 0.98


def test_dishwasher_default_is_099():
    assert (
        DEFAULT_SMART_TERMINATION_DURATION_RATIO_BY_DEVICE[DEVICE_TYPE_DISHWASHER]
        == 0.99
    )


def test_other_device_types_fall_back_to_the_scalar():
    # Only the dishwasher overrides the scalar; everything else resolves to 0.98.
    resolve = DEFAULT_SMART_TERMINATION_DURATION_RATIO_BY_DEVICE.get
    assert resolve(DEVICE_TYPE_WASHING_MACHINE, DEFAULT_SMART_TERMINATION_DURATION_RATIO) == 0.98
    assert resolve("dryer", DEFAULT_SMART_TERMINATION_DURATION_RATIO) == 0.98


def test_config_field_default_is_the_scalar():
    cfg = CycleDetectorConfig(min_power=5.0, off_delay=60)
    assert cfg.smart_termination_duration_ratio == DEFAULT_SMART_TERMINATION_DURATION_RATIO


# ---------------------------------------------------------------------------
# _resolve_smart_ratio (device default + min() pump-out relief)
# ---------------------------------------------------------------------------

def test_non_dishwasher_uses_the_configured_ratio():
    # end-spike state is irrelevant off a dishwasher.
    assert CycleDetector._resolve_smart_ratio("washing_machine", 0.85, True, 999.0, 1000.0) == 0.85
    assert CycleDetector._resolve_smart_ratio("dryer", 0.98, False, 0.0, 1000.0) == 0.98


def test_dishwasher_without_pumpout_uses_the_configured_ratio():
    assert CycleDetector._resolve_smart_ratio("dishwasher", 0.99, False, 0.0, 1000.0) == 0.99


def test_dishwasher_pumpout_below_90pct_uses_the_configured_ratio():
    # A spike at <90% of expected is a mid-cycle rinse drain, not the pump-out.
    assert CycleDetector._resolve_smart_ratio("dishwasher", 0.99, True, 800.0, 1000.0) == 0.99


def test_dishwasher_pumpout_relief_never_tightens_the_gate():
    # Default 0.99: relief loosens to 0.90 once the pump-out is confirmed.
    assert CycleDetector._resolve_smart_ratio("dishwasher", 0.99, True, 900.0, 1000.0) == 0.90
    # A configured value between 0.90 and 0.99 must NOT tighten the relief back up.
    assert CycleDetector._resolve_smart_ratio("dishwasher", 0.95, True, 950.0, 1000.0) == 0.90


def test_dishwasher_configured_below_relief_still_wins():
    # A user who wants an even earlier finish (< 0.90) gets it: min() keeps it.
    assert CycleDetector._resolve_smart_ratio("dishwasher", 0.85, True, 900.0, 1000.0) == 0.85


def test_relief_ignored_when_expected_unknown():
    assert CycleDetector._resolve_smart_ratio("dishwasher", 0.99, True, 900.0, 0.0) == 0.99


def test_out_of_range_configured_ratio_is_clamped():
    # A value persisted by an import / older schema is read unclamped by the manager;
    # _resolve_smart_ratio clamps to the documented [0.50, 1.00] range so a 0.0 can't
    # drop the duration floor and let Smart Termination fire immediately.
    assert CycleDetector._resolve_smart_ratio("washing_machine", 0.0, False, 0.0, 1000.0) == 0.5
    assert CycleDetector._resolve_smart_ratio("washing_machine", 2.0, False, 0.0, 1000.0) == 1.0
    # A clamped-up dishwasher value still gets the pump-out relief.
    assert CycleDetector._resolve_smart_ratio("dishwasher", 0.0, True, 950.0, 1000.0) == 0.5


# ---------------------------------------------------------------------------
# The gate mirror: a lower ratio unblocks the fast path at a duration the
# default blocks. _smart_term_block_reason mirrors the gate's conditions in
# order (returns None when the gate would pass), so it is the canonical, timing-
# free way to prove the ratio's effect (same technique as test_issue_346).
# ---------------------------------------------------------------------------

def test_default_ratio_blocks_a_short_cycle_that_a_lower_ratio_passes():
    expected = 1000.0
    current = 870.0  # 87% of the mean: a real summer / light-load run

    # Default 0.98: the fast path is unreachable for this run.
    assert CycleDetector._smart_term_block_reason(
        current_duration=current, expected=expected, smart_ratio=0.98,
        is_confident=True, ambiguous=False, prefix_ambiguous=False,
    ) == "duration_not_reached"

    # Tuned down to 0.85: the same run now clears the gate.
    assert CycleDetector._smart_term_block_reason(
        current_duration=current, expected=expected, smart_ratio=0.85,
        is_confident=True, ambiguous=False, prefix_ambiguous=False,
    ) is None


# ---------------------------------------------------------------------------
# End-to-end: the config field actually drives when the cycle ends. A washing
# machine that stops at 86% of its profile mean gets the fast finish under a
# tuned-down ratio, while the shipped default keeps waiting.
# ---------------------------------------------------------------------------

def _dt(offset_seconds: float) -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0) + timedelta(seconds=offset_seconds)


def _run_washer_stopping_early(ratio: float):
    """Drive a confident washing-machine cycle that drops to a low standby at 86%
    of a 10000 s expected duration, then feed standby readings to 90%.

    The standby is 3.5 W: below the 4.0 W stop threshold (so the cycle reaches
    ENDING) but energetic enough to trip the end-energy gate over the off-delay
    window (so the plain power-based fallback is held off and Smart Termination is
    the deciding path - the exact situation the FR describes, where a stopped
    machine sits in ENDING). Returns the detector and the on_cycle_end mock.
    """
    expected = 10000.0
    matcher = Mock(side_effect=lambda readings: ("Mix", 0.9, expected, "Washing", False))
    on_end = Mock()
    cfg = CycleDetectorConfig(
        min_power=5.0,
        off_delay=60,
        device_type=DEVICE_TYPE_WASHING_MACHINE,
        completion_min_seconds=600,
        start_duration_threshold=0.0,
        start_energy_threshold=0.0,
        start_threshold_w=6.0,
        stop_threshold_w=4.0,
        min_off_gap=60,  # washer smart-debounce -> max(180, 30) = 180 s
        smart_termination_duration_ratio=ratio,
    )
    det = CycleDetector(config=cfg, on_state_change=Mock(), on_cycle_end=on_end, profile_matcher=matcher)

    det.process_reading(100.0, _dt(0))
    det.process_reading(100.0, _dt(30))
    det.process_reading(100.0, _dt(60))
    assert det.matched_profile == "Mix"
    assert det._expected_duration == expected

    # Run high to 86% of expected.
    for t in range(90, 8600, 30):
        det.process_reading(100.0, _dt(t))
    # Machine stops but keeps a low standby draw; feed it to 90% of expected.
    for t in range(8600, 9000, 30):
        det.process_reading(3.5, _dt(t))
    return det, on_end


def test_tuned_ratio_gives_the_early_finish_the_default_withholds():
    from custom_components.ha_washdata.cycle_detector import TerminationReason

    # Tuned down to 0.85: the machine stopping at 86% gets the fast finish, well
    # short of 98% of the mean, via Smart Termination.
    det_low, end_low = _run_washer_stopping_early(0.85)
    assert det_low.state in (STATE_FINISHED, STATE_OFF)
    assert end_low.called
    cycle = end_low.call_args[0][0]
    assert cycle.get("termination_reason") == TerminationReason.SMART
    assert cycle["duration"] < 9800  # ended before 98% of expected

    # Shipped default 0.98: the same cycle is still waiting at 90% - the exact
    # "structurally unreachable fast path" the FR reports. The standby energy gate
    # holds off the plain fallback, so the cycle is still open, not finished.
    det_hi, end_hi = _run_washer_stopping_early(0.98)
    assert det_hi.state not in (STATE_FINISHED, STATE_OFF)
    assert not end_hi.called


# ---------------------------------------------------------------------------
# ws_set_options validation: clamp to [0.50, 1.00]; drop empty/invalid so the
# device-type default (resolved in the builder) applies again.
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


async def test_in_range_value_is_stored_verbatim():
    entry = _entry({})
    hass, _ = _hass()
    saved = await _set_options(entry, hass, {CONF_SMART_TERMINATION_DURATION_RATIO: 0.85})
    assert saved[CONF_SMART_TERMINATION_DURATION_RATIO] == 0.85


async def test_above_range_is_clamped_to_one():
    entry = _entry({})
    hass, _ = _hass()
    saved = await _set_options(entry, hass, {CONF_SMART_TERMINATION_DURATION_RATIO: 1.5})
    assert saved[CONF_SMART_TERMINATION_DURATION_RATIO] == 1.0


async def test_below_range_is_clamped_to_half():
    entry = _entry({})
    hass, _ = _hass()
    saved = await _set_options(entry, hass, {CONF_SMART_TERMINATION_DURATION_RATIO: 0.1})
    assert saved[CONF_SMART_TERMINATION_DURATION_RATIO] == 0.5


async def test_empty_string_drops_the_key():
    entry = _entry({CONF_SMART_TERMINATION_DURATION_RATIO: 0.85})
    hass, _ = _hass()
    saved = await _set_options(entry, hass, {CONF_SMART_TERMINATION_DURATION_RATIO: ""})
    assert CONF_SMART_TERMINATION_DURATION_RATIO not in saved


async def test_non_numeric_drops_the_key():
    entry = _entry({})
    hass, _ = _hass()
    saved = await _set_options(entry, hass, {CONF_SMART_TERMINATION_DURATION_RATIO: "abc"})
    assert CONF_SMART_TERMINATION_DURATION_RATIO not in saved


async def test_none_drops_the_key():
    entry = _entry({CONF_SMART_TERMINATION_DURATION_RATIO: 0.85})
    hass, _ = _hass()
    saved = await _set_options(entry, hass, {CONF_SMART_TERMINATION_DURATION_RATIO: None})
    assert CONF_SMART_TERMINATION_DURATION_RATIO not in saved


# ---------------------------------------------------------------------------
# Playground: the default is resolved in the config builder (never in the gate),
# so effective_settings never skips a None-valued field (FR requirement #4), and
# the playground fallback base config resolves the device-type default too.
# ---------------------------------------------------------------------------

def test_effective_settings_surfaces_the_ratio():
    cfg = CycleDetectorConfig(min_power=5.0, off_delay=60, smart_termination_duration_ratio=0.9)
    eff = playground.effective_settings(cfg, None)
    assert eff[CONF_SMART_TERMINATION_DURATION_RATIO] == 0.9


def test_playground_base_config_resolves_dishwasher_default():
    manager = MagicMock()
    manager.detector.config = None  # force the fallback build path
    entry = MagicMock()
    entry.data = {}
    entry.options = {CONF_DEVICE_TYPE: DEVICE_TYPE_DISHWASHER}
    cfg = ws_api._playground_base_config(manager, entry)
    assert cfg.smart_termination_duration_ratio == 0.99


def test_playground_base_config_resolves_other_default():
    manager = MagicMock()
    manager.detector.config = None
    entry = MagicMock()
    entry.data = {}
    entry.options = {CONF_DEVICE_TYPE: DEVICE_TYPE_WASHING_MACHINE}
    cfg = ws_api._playground_base_config(manager, entry)
    assert cfg.smart_termination_duration_ratio == 0.98


def test_playground_base_config_honours_an_explicit_option():
    manager = MagicMock()
    manager.detector.config = None
    entry = MagicMock()
    entry.data = {}
    entry.options = {CONF_DEVICE_TYPE: DEVICE_TYPE_DISHWASHER, CONF_SMART_TERMINATION_DURATION_RATIO: 0.8}
    cfg = ws_api._playground_base_config(manager, entry)
    assert cfg.smart_termination_duration_ratio == 0.8
