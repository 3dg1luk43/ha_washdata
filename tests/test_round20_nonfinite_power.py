# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A power sensor reporting nan/inf must not become the authoritative reading.

`float()` accepts `"nan"`, `"inf"` and `"infinity"`. A power reading is compared
against thresholds, and every comparison against `nan` is False, so such a
reading does not raise - it silently switches gates OFF. With `_current_power`
set to nan both the unmatched-cycle watchdog (`< start_threshold_w`) and the
high-power silence deferral (`> min_power`) evaluate False, so a running cycle
loses the guards that decide whether it ends at all.

`options_utils.option_float` already documents and guards exactly this shape for
stored options; these tests hold the two sensor-reading paths to the same rule.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.manager import _finite_power

NON_FINITE = ["nan", "NaN", "inf", "-inf", "Infinity", "infinity"]


@pytest.mark.parametrize("raw", NON_FINITE)
def test_non_finite_strings_are_rejected(raw: str) -> None:
    assert _finite_power(raw) is None


@pytest.mark.parametrize("raw", ["not-a-number", "", None, "unavailable"])
def test_unparseable_values_are_rejected(raw: Any) -> None:
    assert _finite_power(raw) is None


@pytest.mark.parametrize(
    "raw,expected", [("0", 0.0), ("0.0", 0.0), ("2000", 2000.0), ("-5.5", -5.5)]
)
def test_real_readings_pass_through_unchanged(raw: str, expected: float) -> None:
    """No valid reading may change behaviour, including a legitimate 0 W - the
    terminal 0 W row is what marks a cycle end."""
    assert _finite_power(raw) == expected


def test_an_oversized_integer_literal_is_rejected() -> None:
    """`float()` on an unbounded int raises rather than returning inf, the same
    case `option_float` catches OverflowError for."""
    assert _finite_power(10**400) is None


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Washer"
    entry.options = {"power_sensor": "sensor.test_power"}
    entry.data = {}
    return entry


@pytest.fixture
def manager(hass: Any, mock_entry: Any) -> Any:
    from custom_components.ha_washdata.manager import WashDataManager

    hass.config_entries.async_get_entry = MagicMock(return_value=mock_entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = WashDataManager(hass, mock_entry)
        mgr.profile_store.get_suggestions = MagicMock(return_value={})
        return mgr


def _state(value: str) -> Any:
    """A State-shaped stand-in for the event payload path."""
    st = MagicMock()
    st.state = value
    st.last_reported = datetime(2026, 5, 1, 8, 0, 0)
    st.last_updated = st.last_reported
    return st


@pytest.mark.asyncio
async def test_live_power_state_rejects_a_nan_reading(manager: Any) -> None:
    """The reported site. None is the documented "non-numeric" outcome, and it is
    what keeps the nan out of `current_power` and `_resync_power_from_state`."""
    manager.hass.states.async_set("sensor.test_power", "nan")

    assert manager._live_power_state() is None


@pytest.mark.asyncio
async def test_live_power_state_still_accepts_a_real_reading(manager: Any) -> None:
    manager.hass.states.async_set("sensor.test_power", "1234.5")

    live = manager._live_power_state()
    assert live is not None
    assert live[0] == 1234.5


@pytest.mark.asyncio
async def test_current_power_does_not_surface_a_nan(manager: Any) -> None:
    """`current_power` feeds the sensors, the panel and ws_get_devices; it falls
    back to the cache rather than publishing a nan."""
    manager._current_power = 42.0
    manager.hass.states.async_set("sensor.test_power", "inf")

    assert manager.current_power == 42.0


def test_the_event_path_drops_a_nan_before_the_detector_sees_it(
    manager: Any
) -> None:
    """The route a real nan actually arrives by, and the one that would poison the
    detector's own accumulators. `_async_power_changed` is a sync @callback."""
    manager.detector.process_reading = MagicMock()
    manager.diag_buffer.record_power = MagicMock()

    event = MagicMock()
    event.data = {"new_state": _state("nan"), "old_state": _state("1000")}
    manager._async_power_changed(event)

    manager.detector.process_reading.assert_not_called()
    manager.diag_buffer.record_power.assert_not_called()


def test_the_event_path_still_accepts_a_real_reading(manager: Any) -> None:
    """The control: a valid reading is still recorded and fed through."""
    manager.diag_buffer.record_power = MagicMock()

    event = MagicMock()
    event.data = {"new_state": _state("1500"), "old_state": _state("1000")}
    manager._async_power_changed(event)

    manager.diag_buffer.record_power.assert_called_once()
    assert manager.diag_buffer.record_power.call_args.args[0] == 1500.0


# ── every power-reading site, not just the two from round 20 ────────────────
#
# Round 21 found the gap this way: the setup seed (`_async_setup_complete`)
# parsed with a bare `float()` and wrote the result straight into
# `_current_power`. Seeding a nan there is PERMANENT, because
# `_resync_power_from_state` returns early exactly when `_live_power_state()`
# yields None, which is what a non-finite sensor now produces - so the healing
# path added for #409 could never overwrite it. Guarding only the two round-20
# sites left the cache poisonable at setup and reload.


def test_every_power_reading_site_routes_through_the_helper() -> None:
    """Structural guard, and the ONLY coverage for two of the six sites.

    The setup seed and the config-reload re-seed both live inside
    `WashDataManager.async_setup` / the options-update handler, so reaching them
    means booting all of setup; this test pins them instead. If that is ever
    considered too indirect, the fix is to extract those blocks, not to drop the
    assertion.

    The energy-price and energy-sensor readers at the end of the module are
    deliberately excluded: different consumer, different fallback contract, and
    they are recorded as follow-up rather than reshaped inside a review loop.
    """
    import re
    from pathlib import Path

    src = Path(
        "custom_components/ha_washdata/manager.py"
    ).read_text() if Path("custom_components/ha_washdata/manager.py").exists() else (
        Path(__file__).resolve().parents[1]
        / "custom_components" / "ha_washdata" / "manager.py"
    ).read_text()

    bare = [
        m.start() for m in re.finditer(r"float\((?:new_|old_)?state\.state\)", src)
    ]
    # Two known-and-documented exceptions remain (price + energy sensor).
    assert len(bare) == 2, (
        f"expected only the 2 energy/price readers to parse a state directly, "
        f"found {len(bare)}; a power reading must go through _finite_power"
    )


@pytest.mark.asyncio
async def test_resync_cannot_heal_a_poisoned_cache(manager: Any) -> None:
    """Why seeding a nan would be permanent rather than merely wrong once: the
    resync path bails on a non-finite sensor, so nothing overwrites the cache.
    This is the mechanism that makes the seed guard load-bearing."""
    import math

    manager._current_power = math.nan
    manager.hass.states.async_set("sensor.test_power", "nan")

    manager._resync_power_from_state(datetime(2026, 5, 1, 8, 0, 0), True)

    # Still nan: the resync declined, which is exactly why the seed must refuse.
    assert math.isnan(manager._current_power)


@pytest.mark.asyncio
async def test_resync_heals_the_cache_once_the_sensor_recovers(manager: Any) -> None:
    """The control: a real reading after a bad one is picked up normally."""
    import math

    manager._current_power = math.nan
    manager.hass.states.async_set("sensor.test_power", "900")

    manager._resync_power_from_state(datetime(2026, 5, 1, 8, 0, 0), False)

    assert manager._current_power == 900.0


def test_prev_raw_power_keeps_the_cached_value_for_a_nan_old_state(
    manager: Any
) -> None:
    """`prev_raw_power >= min_p` is the "genuine drop from active power" arm of
    `is_low_power`, and a nan there evaluates False and silently disables it.

    That arm is what exempts a real drop to ~0 W from the sampling-interval
    throttle, so the observable consequence is whether the reading reaches the
    detector at all: `_last_reading_time` is set just now, so with the arm
    disabled the throttle returns early and the drop is swallowed.
    """
    manager._current_power = 1200.0
    manager._sampling_interval = 300
    manager._last_reading_time = dt_util.now()
    # min_power must be set explicitly: on the patched detector it would come from
    # MagicMock.__float__, i.e. 1.0, and `power < min_p` would be False so the
    # branch under test never runs. "off" keeps the cycle-active arm out of it, so
    # is_low_power can only become True via prev_raw_power.
    manager.detector.config.min_power = 2.0
    manager.detector.state = "off"
    manager.detector.process_reading = MagicMock()

    event = MagicMock()
    event.data = {"new_state": _state("1.0"), "old_state": _state("nan")}
    manager._async_power_changed(event)

    # The nan old_state was ignored, so prev_raw_power kept the cached 1200 W and
    # the drop was still recognised as genuine, bypassing the throttle.
    manager.detector.process_reading.assert_called_once()
