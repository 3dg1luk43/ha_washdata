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
"""Register item 351: the standby-band spin guard was inert on the devices it exists for.

`_is_standby_band_stuck` finalises a washer sitting on a flat low plateau past its
expected duration. It defers while the matched profile still owes a terminal
high-power block, so the final spin cannot be recorded as a second cycle - but
that deferral runs through `_anticrease_spin_pending`, which needs element 10 of
the match tuple, and the manager only supplied element 10 when
`anti_wrinkle_enabled` was true. That option defaults FALSE, so on a washer the
guard returned False immediately.

Measured over the 273-cycle replay corpus: the band fired 14 times, 11 with the
guard inert, 6 of those with a reading above `min_power` still ahead. Arming it
against `STANDBY_BAND_MAX_FRACTION` of the cycle's own peak took washing-machine
splits from 4.49% to 1.90% with the median end lag unchanged.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from custom_components.ha_washdata.const import (
    DEVICE_TYPE_DISHWASHER,
    DEVICE_TYPE_DRYER,
    DEVICE_TYPE_WASHING_MACHINE,
    STANDBY_BAND_FINALIZE_DEVICE_TYPES,
    STANDBY_BAND_MAX_FRACTION,
)
from custom_components.ha_washdata.cycle_detector import CycleDetector, CycleDetectorConfig
from custom_components.ha_washdata.manager import WashDataManager

T0 = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# manager: element 10 is supplied for a standby-band device with anti-wrinkle off
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_hass() -> Any:
    hass = MagicMock()
    hass.data = {}
    hass.config_entries.async_get_entry = MagicMock()
    return hass


@pytest.fixture
def manager(mock_hass: Any) -> WashDataManager:
    entry = MagicMock()
    entry.entry_id = "e351"
    entry.title = "Washer"
    entry.options = {"power_sensor": "sensor.p"}
    entry.data = {}
    mock_hass.config_entries.async_get_entry.return_value = entry
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        return WashDataManager(mock_hass, entry)


def _arm(manager: WashDataManager, device_type: str, *, anti_wrinkle: bool, peak: float):
    manager.detector.config.device_type = device_type
    manager.detector.config.anti_wrinkle_enabled = anti_wrinkle
    manager.detector.config.anti_wrinkle_max_power = 40.0
    manager.detector._cycle_max_power = peak
    manager.profile_store.profile_terminal_high_block = MagicMock(
        return_value=(0.95, 300.0, 6000.0)
    )
    return manager.profile_store.profile_terminal_high_block


def test_a_washer_with_anti_wrinkle_off_still_gets_the_spin_guard(
    manager: WashDataManager,
) -> None:
    """The whole finding: this returned None, so the guard never armed."""
    block_fn = _arm(manager, DEVICE_TYPE_WASHING_MACHINE, anti_wrinkle=False, peak=2000.0)

    out = manager._terminal_high_for_guards("Eco")

    assert out is not None, "the standby-band guard is still inert"
    # Armed against the plateau ceiling, not the dryer's tumble level.
    assert block_fn.call_args[0][1] == pytest.approx(2000.0 * STANDBY_BAND_MAX_FRACTION)
    # ...and the bar travels with the block so the live counter uses the same one.
    assert len(out) == 4
    assert out[3] == pytest.approx(2000.0 * STANDBY_BAND_MAX_FRACTION)


def test_anti_wrinkle_still_wins_and_keeps_its_own_bar(manager: WashDataManager) -> None:
    """A dryer running anti-wrinkle keeps the tumble level and the 3-tuple."""
    block_fn = _arm(manager, DEVICE_TYPE_DRYER, anti_wrinkle=True, peak=2000.0)

    out = manager._terminal_high_for_guards("Eco")

    assert block_fn.call_args[0][1] == 40.0
    assert len(out) == 3


def test_a_dishwasher_is_not_armed_by_the_standby_path(manager: WashDataManager) -> None:
    """Only STANDBY_BAND_FINALIZE_DEVICE_TYPES reach the standby finalise at all."""
    assert DEVICE_TYPE_DISHWASHER not in STANDBY_BAND_FINALIZE_DEVICE_TYPES
    _arm(manager, DEVICE_TYPE_DISHWASHER, anti_wrinkle=False, peak=2000.0)

    assert manager._terminal_high_for_guards("Eco") is None


def test_a_cycle_with_no_peak_yet_arms_nothing(manager: WashDataManager) -> None:
    """Fail open: a zero peak would make the ceiling 0 W, which matches every
    reading and would hold every cycle open."""
    _arm(manager, DEVICE_TYPE_WASHING_MACHINE, anti_wrinkle=False, peak=0.0)

    assert manager._terminal_high_for_guards("Eco") is None


def test_no_profile_arms_nothing(manager: WashDataManager) -> None:
    _arm(manager, DEVICE_TYPE_WASHING_MACHINE, anti_wrinkle=False, peak=2000.0)
    assert manager._terminal_high_for_guards(None) is None


# ---------------------------------------------------------------------------
# detector: the carried ceiling is what the live seconds are counted against
# ---------------------------------------------------------------------------
def _det(peak: float = 2000.0) -> CycleDetector:
    cfg = CycleDetectorConfig(
        min_power=2.0,
        off_delay=180,
        device_type=DEVICE_TYPE_WASHING_MACHINE,
        stop_threshold_w=2.0,
    )
    cfg.anti_wrinkle_max_power = 40.0
    det = CycleDetector(cfg, lambda a, b: None, lambda p: None)
    det._current_cycle_start = T0
    det._expected_duration = 6300.0
    det._cycle_max_power = peak
    return det


def test_the_quad_is_preserved_through_sanitisation() -> None:
    det = _det()
    assert det._sanitize_terminal_high((0.95, 300.0, 6000.0, 200.0)) == (
        0.95, 300.0, 6000.0, 200.0,
    )


def test_a_broken_ceiling_degrades_to_the_triple_never_to_none() -> None:
    """This method must never DISARM a guard that would otherwise arm."""
    det = _det()
    for bad in ("x", float("inf"), 0.0, -5.0, None):
        assert det._sanitize_terminal_high((0.95, 300.0, 6000.0, bad)) == (
            0.95, 300.0, 6000.0,
        ), bad


def test_the_live_counter_uses_the_carried_ceiling() -> None:
    """The profile's block was measured above 200 W, so the live seconds must be
    counted above 200 W too - not above the dryer's 40 W tumble level, which
    would credit the quiet plateau as spin and release the guard immediately."""
    det = _det()
    # 600 s of plateau at 100 W: above anti_wrinkle_max_power (40), below the
    # standby ceiling (200).
    det._power_readings = [
        (T0 + timedelta(seconds=float(t)), 100.0) for t in range(6000, 6600, 30)
    ]

    assert det._high_power_seconds_since(6000.0) > 0.0, "sanity: 100 W is above 40 W"
    assert det._high_power_seconds_since(6000.0, ceiling_w=200.0) == 0.0


def test_the_spin_guard_holds_on_a_plateau_and_releases_on_the_spin() -> None:
    """End to end through the predicate the standby finalise actually calls."""
    det = _det()
    det._matched_profile = "Eco"
    det._matched_terminal_high = (0.95, 300.0, 6000.0, 200.0)
    plateau = [(T0 + timedelta(seconds=float(t)), 100.0) for t in range(6000, 6400, 20)]

    det._power_readings = list(plateau)
    assert det._anticrease_spin_pending(T0 + timedelta(seconds=6400)) is True

    # ...and once the spin has run, it lets go.
    det._power_readings = plateau + [
        (T0 + timedelta(seconds=float(t)), 900.0) for t in range(6400, 6600, 20)
    ]
    assert det._anticrease_spin_pending(T0 + timedelta(seconds=6600)) is False


# ---------------------------------------------------------------------------
# the Playground replay has to make the same tuple, or the sim diverges
# ---------------------------------------------------------------------------
def test_the_playground_mirrors_the_standby_arm() -> None:
    """`end_gate_eval.py` drives the detector through the Playground, so an arm
    that exists only in the manager is invisible to every measurement made with
    it - which is exactly how this change first measured as a no-op."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "custom_components" / "ha_washdata"
    pg = (root / "playground.py").read_text()
    assert "STANDBY_BAND_FINALIZE_DEVICE_TYPES" in pg
    assert "STANDBY_BAND_MAX_FRACTION" in pg
    assert "_cycle_max_power" in pg
