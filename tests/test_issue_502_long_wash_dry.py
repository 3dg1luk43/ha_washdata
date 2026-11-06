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
"""Issue #502: an 8 h+ wash+dry on a washer-dryer always ended force_stopped.

Midea LSD12X1 on a ~1 s publish-on-change Zigbee plug, programmes of 7.9-8.5 h.

1. The detector's flat 8 h cap cut two runs at exactly 8 h 00 m in their drying
   phase (1300 W heater pulses, ~50 W tumbling); the rest opened a second cycle
   9-10 s later. The cap now scales on the device's longest learned programme
   (and the match's expected duration): max(8 h, 1.5 x that), at most 24 h.
2. The manager's unmatched idle guard force-ended a 7 h 55 min run 24 s after it
   dropped to 0 W at its real end, so a cycle the ENDING fallback was about to
   close as completed was stored force_stopped. It now waits out the detector's
   own end wait first.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from custom_components.ha_washdata import match_rules
from custom_components.ha_washdata.const import (
    CYCLE_DURATION_CAP_BASE_S,
    DEFAULT_UNMATCHED_WATCHDOG_CEILING_BY_DEVICE,
    STATE_ENDING,
    STATE_PAUSED,
    STATE_RUNNING,
)
from custom_components.ha_washdata.cycle_detector import CycleDetector
from custom_components.ha_washdata.detector_config import build_detector_config

T0 = datetime(2026, 10, 25, 18, 39, 59, tzinfo=timezone.utc)

#: The reporter's options (diagnostics export, 0.5.9.1).
OPTIONS: dict[str, Any] = {
    "device_type": "washer_dryer",
    "start_threshold_w": 4.69,
    "stop_threshold_w": 3.75,
    "min_power": 1.0,
    "off_delay": 180,
    "no_update_active_timeout": 600,
    "watchdog_interval": 30,
    "delay_start_detect_enabled": True,
    "delay_timeout_hours": 8.0,
    "profile_match_interval": 40,
}

#: The reporter's profiles as exported (durations in seconds).
PROFILES: dict[str, Any] = {
    "Normal + Dry": {"min_duration": 24206.0, "max_duration": 24899.7, "avg_duration": 24552.85},
    "Conforter": {"avg_duration": 5691.4, "min_duration": 5691.4, "max_duration": 5691.4},
    "Cotton + Dry": {"avg_duration": 28800.517669, "max_duration": 28800.517669},
    "Dry Only": {"avg_duration": 12530.515362, "max_duration": 12530.515362},
    "Towels + Dry": {"avg_duration": 28467.3, "max_duration": 28467.3},
}

PROGRAMME_S = 8.5 * 3600.0  # the longest run in the export: 8 h 18 min + 9 s + 19 min


def _wash_dry_power(t: float) -> float:
    """A synthetic Midea wash+dry: 4 h wash, then the drying pulses to PROGRAMME_S."""
    if t < 60:
        return 25.0
    if t < 4 * 3600:
        if 600 <= t < 1800:
            return 1950.0  # heating the wash water
        if 3.5 * 3600 <= t < 3.6 * 3600:
            return 480.0  # final spin
        return 150.0 + 100.0 * math.sin(t / 37.0)
    # Drying: a ~70 s heater pulse every 3 min, 25-130 W tumbling between.
    if (t - 4 * 3600) % 180 < 70:
        return 1300.0
    return 77.0 + 52.0 * math.sin(t / 11.0)


def _run(detector: CycleDetector, step: float = 5.0) -> None:
    """Feed the programme, a 5 W reading, 0 W, then 30 s keepalives at 0 W."""
    t = 0.0
    while t < PROGRAMME_S:
        detector.process_reading(_wash_dry_power(t), T0 + timedelta(seconds=t))
        t += step
    detector.process_reading(5.0, T0 + timedelta(seconds=t))
    t += 30.0
    for _ in range(60):  # 30 min of 0 W, on the watchdog's 30 s cadence
        detector.process_reading(0.0, T0 + timedelta(seconds=t))
        t += 30.0


def _detector(cycles: list[dict[str, Any]], provider=None) -> CycleDetector:
    config = build_detector_config(OPTIONS, {}, "washer_dryer")
    return CycleDetector(
        config,
        lambda _old, _new: None,
        cycles.append,
        device_name="issue-502",
        longest_programme_provider=provider,
    )


# --- the rule ---------------------------------------------------------------


def test_cap_is_8h_until_the_device_has_learned_a_long_programme() -> None:
    assert CYCLE_DURATION_CAP_BASE_S == 28800
    assert match_rules.cycle_duration_cap_s(0.0, 0.0) == 28800.0
    # A device whose programmes all run under 5 h 20 min keeps exactly 8 h.
    assert match_rules.cycle_duration_cap_s(4 * 3600.0, 5.3 * 3600.0) == 28800.0


def test_cap_scales_on_the_longest_learned_programme_or_the_match() -> None:
    longest = match_rules.longest_programme_s(PROFILES)
    assert longest == pytest.approx(28800.517669)
    assert match_rules.cycle_duration_cap_s(0.0, longest) == pytest.approx(43200.78, abs=0.01)
    # Unmatched on 2026-10-25, the device's longest was Normal + Dry's 24899.7 s:
    # 10.4 h, enough for the 8.3 h run the flat cap cut.
    assert match_rules.cycle_duration_cap_s(0.0, 24899.7) == pytest.approx(37349.55)
    # The match's own expected duration counts when it is the longer one.
    assert match_rules.cycle_duration_cap_s(30000.0, 24899.7) == pytest.approx(45000.0)


@pytest.mark.parametrize(
    "bad", [float("inf"), float("nan"), -5.0, "x", None, 53.78 * 3600, 10**400]
)
def test_corrupt_or_implausible_lengths_do_not_stretch_the_cap(bad: Any) -> None:
    """A 53.8 h washer profile is in the corpus: never more than 1.5 x 16 h."""
    assert match_rules.cycle_duration_cap_s(bad, bad) == 28800.0
    assert match_rules.longest_programme_s({"p": {"avg_duration": bad}}) == 0.0
    assert match_rules.cycle_duration_cap_s(16 * 3600.0, 0.0) == 24 * 3600.0


@pytest.mark.parametrize("profiles", [None, [], "x", {"p": None}, {"p": [1, 2]}])
def test_longest_programme_never_raises(profiles: Any) -> None:
    assert match_rules.longest_programme_s(profiles) == 0.0


def test_unmatched_guard_ceilings_stay_below_the_cap() -> None:
    for ceiling in DEFAULT_UNMATCHED_WATCHDOG_CEILING_BY_DEVICE.values():
        assert ceiling < CYCLE_DURATION_CAP_BASE_S


# --- the cut (detector) -----------------------------------------------------


def test_without_a_learned_programme_an_8h_plus_run_is_cut_at_8h() -> None:
    """The reported cut: force_stopped at 8 h 00 m, the drying rest a second cycle."""
    cycles: list[dict[str, Any]] = []
    _run(_detector(cycles))
    assert cycles[0]["status"] == "force_stopped"
    assert cycles[0]["duration"] == pytest.approx(28800.0, abs=10.0)
    assert len(cycles) == 2


def test_a_learned_long_programme_lifts_the_cap_and_the_run_completes() -> None:
    cycles: list[dict[str, Any]] = []
    _run(_detector(cycles, lambda: match_rules.longest_programme_s(PROFILES)))
    assert len(cycles) == 1
    assert cycles[0]["status"] == "completed"
    assert cycles[0]["duration"] == pytest.approx(PROGRAMME_S, abs=60.0)


def test_a_provider_that_raises_falls_back_to_the_base_cap() -> None:
    def _boom() -> float:
        raise RuntimeError("store gone")

    cycles: list[dict[str, Any]] = []
    _run(_detector(cycles, _boom))
    assert cycles[0]["status"] == "force_stopped"
    assert cycles[0]["duration"] == pytest.approx(28800.0, abs=10.0)


def test_a_zombie_still_stops_at_the_scaled_cap() -> None:
    """Zombie protection stays: a draw that never ends stops at 1.5 x the longest."""
    cycles: list[dict[str, Any]] = []
    det = _detector(cycles, lambda: 28800.0)
    t = 0.0
    while t < 13 * 3600 and not cycles:
        det.process_reading(120.0, T0 + timedelta(seconds=t))
        t += 30.0
    assert cycles and cycles[0]["status"] == "force_stopped"
    assert cycles[0]["duration"] == pytest.approx(43200.0, abs=31.0)


# --- the guard's deferral (detector side) -----------------------------------


def _in_drying(det: CycleDetector) -> float:
    t = 0.0
    while t < 600:
        det.process_reading(1300.0 if t % 180 < 70 else 60.0, T0 + timedelta(seconds=t))
        t += 5.0
    assert det.state == STATE_RUNNING
    return t


def test_ordinary_end_pending_while_the_end_wait_runs() -> None:
    cycles: list[dict[str, Any]] = []
    det = _detector(cycles)
    t = _in_drying(det)
    assert not det.ordinary_end_pending(60.0, 60.0)  # drawing power: nothing pending
    det.process_reading(0.0, T0 + timedelta(seconds=t))
    # The drop itself, before any quiet has accrued: the #502 moment (24 s in).
    assert det.ordinary_end_pending(0.0, 60.0)
    wait = max(det.config.off_delay, det.config.min_off_gap)
    assert wait == 600  # off_delay 180, washer_dryer min_off_gap 600
    for _ in range(int(wait / 30) - 2):
        t += 30.0
        det.process_reading(0.0, T0 + timedelta(seconds=t))
        assert det.state in (STATE_RUNNING, STATE_PAUSED, STATE_ENDING)
        assert det.ordinary_end_pending(0.0, 60.0)


def test_a_standby_draw_at_or_above_stop_is_never_deferred() -> None:
    """The detector cannot end a standby-band hold itself; the guard must."""
    cycles: list[dict[str, Any]] = []
    det = _detector(cycles)
    t = _in_drying(det)
    det.process_reading(4.0, T0 + timedelta(seconds=t))  # stop 3.75 <= 4.0 < start 4.69
    assert not det.ordinary_end_pending(4.0, 60.0)


def test_ordinary_end_not_pending_once_the_wait_and_grace_are_spent() -> None:
    cycles: list[dict[str, Any]] = []
    det = _detector(cycles)
    t = _in_drying(det)
    # A user pause holds ENDING open past its wait (nothing releases it).
    det._verified_pause = True  # noqa: SLF001
    det._user_paused = True  # noqa: SLF001
    for _ in range(30):  # 15 min of 0 W
        det.process_reading(0.0, T0 + timedelta(seconds=t))
        t += 30.0
    assert not cycles
    assert not det.ordinary_end_pending(0.0, 60.0)
    assert det.ordinary_end_pending(0.0, 600.0)


# --- the 24 s force-end (real manager) ----------------------------------------


async def test_unmatched_run_past_7h_that_finishes_is_stored_completed(
    hass, freezer, caplog
) -> None:
    """The 2026-11-04 run: 7.6 h unmatched, then 5 W, 0 W and a silent plug.

    Before #502 the unmatched idle guard (7 h washer-dryer ceiling, power below
    start) force-ended it on the first watchdog tick after the drop and stored it
    force_stopped. Now the ENDING fallback closes it after its 600 s wait.
    """
    from .real_manager import boot, feed, idle_expire, make_entry, record_notify

    record_notify(hass)
    entry = make_entry(hass, {**OPTIONS, "min_off_gap": 600})
    mgr = await boot(hass, entry)
    assert mgr._unmatched_watchdog_ceiling == 25200.0  # noqa: SLF001
    await feed(hass, freezer, 300, int(7.6 * 3600), step=120)
    assert mgr.detector.state == STATE_RUNNING
    assert mgr.detector.expected_duration_seconds == 0  # nothing learned: unmatched
    await feed(hass, freezer, 5, 30, step=30)
    await feed(hass, freezer, 0, 30, step=30)
    await idle_expire(hass, freezer, 20 * 60, step=30)  # silent plug, watchdog ticks

    stored = mgr.profile_store.get_past_cycles()
    assert [c["status"] for c in stored] == ["completed"]
    assert stored[0]["duration"] == pytest.approx(7.6 * 3600, abs=150)
    assert "Unmatched idle cycle" not in caplog.text
    await mgr.async_shutdown()
