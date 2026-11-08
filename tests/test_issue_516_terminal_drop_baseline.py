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
"""#516: normal washer ends were stored ``interrupted`` by the terminal-drop finalize.

The terminal drop fires on a drop earlier than 0.8 x the earliest offset at which
a completed cycle ever went quiet. A washer's stored trace is cut at its last
activity, so its own end never counted. On the reporter's washer (Whirlpool,
"Apply smart models" on, Stop Threshold raised 1.8 -> 2.3 W) no normal cycle had
a sub-threshold pause, and three over-long timeout runs whose standby tail now
read as quiet set the baseline at 5748 s: every ~60 min end (3600 s < 4598 s)
closed as a terminal drop after 90 s, ``interrupted``, with no finish
notification, and an interrupted cycle never seeds the baseline (self-lock).

A completed cycle's end now counts as a legitimate quiet onset. It only caps a
baseline the quiet spans set, so it can only remove fires.
"""
from __future__ import annotations

from typing import Any

from custom_components.ha_washdata.const import TerminationReason
from custom_components.ha_washdata.detector_config import (
    terminal_drop_baseline_for,
    terminal_drop_fires,
)
from custom_components.ha_washdata.profile_store import earliest_sustained_quiet_offset

from .real_manager import boot, feed, make_entry, record_notify

STOP = 2.3
MEGA_FIRST_QUIET = (5748.0, 7000.0, 9290.0)


def _normal_points(minutes: int, step: int = 30) -> list[tuple[float, float]]:
    """A ~60 min wash as the reporter's plug draws it: heat, wash with ~5 W pause
    dips (above the 2.3 W stop threshold, so never quiet), final spin."""
    end = minutes * 60
    pts: list[tuple[float, float]] = []
    for t in range(0, end + 1, step):
        if t < 900:
            p = 2000.0
        elif t > end - 300:
            p = 600.0
        else:
            p = 5.0 if (t - 900) % 600 < 120 else 300.0
        pts.append((float(t), p))
    return pts


def _cycle(cid: str, points: list[tuple[float, float]], status: str = "completed") -> dict[str, Any]:
    return {
        "id": cid,
        "start_time": "2026-10-01T08:00:00+00:00",
        "duration": points[-1][0],
        "status": status,
        "power_data": [[t, p] for t, p in points],
    }


def _normal(i: int) -> dict[str, Any]:
    """Stored as a washer stores it: trace cut at the last activity (keep_tail=False)."""
    return _cycle(f"n{i}", _normal_points(52 + (i % 9) * 2))


def _mega(i: int) -> dict[str, Any]:
    """A 165-247 min timeout run: a normal wash, then a standby tail that read above
    the old 1.8 W threshold and dips under the new 2.3 W one at ``first_quiet``."""
    first_quiet = MEGA_FIRST_QUIET[i]
    end = (165, 200, 247)[i] * 60
    pts = _normal_points(60)
    pts += [(float(t), 2.5) for t in range(3630, int(first_quiet), 30)]
    pts += [(float(t), 2.0) for t in range(int(first_quiet), end + 1, 30)]
    return _cycle(f"m{i}", pts)


def _reporter_store() -> list[dict[str, Any]]:
    return [_normal(i) for i in range(23)] + [_mega(i) for i in range(3)]


def _live(points: list[tuple[float, float]], quiet_s: int = 120) -> list[tuple[float, float]]:
    last = points[-1][0]
    return points + [(last + 30.0 * (k + 1), 0.0) for k in range(quiet_s // 30)]


def _fires(cycles: list[dict[str, Any]], live: list[tuple[float, float]]) -> bool:
    return terminal_drop_fires(live, terminal_drop_baseline_for(cycles, STOP), STOP)


# ── The baseline ────────────────────────────────────────────────────────────


def test_normal_ends_cap_the_baseline() -> None:
    earliest, peak_range = terminal_drop_baseline_for(_reporter_store(), STOP)
    # Was 5748 s (the first mega-cycle's standby quiet), a fire bound of 4598 s:
    # later than every normal end. Now the shortest completed run's end.
    assert earliest is not None and earliest * 0.8 < 3600.0
    assert earliest == min(c["duration"] for c in _reporter_store()[:23])
    assert peak_range == (2000.0, 2000.0)


def test_normal_end_is_not_a_terminal_drop() -> None:
    assert _fires(_reporter_store(), _live(_normal_points(60))) is False


def test_mislabelled_ends_do_not_lock_it() -> None:
    """The self-lock: each mislabelled normal end is ``interrupted`` and never seeds
    the baseline. Completed ends still bound it, so the next normal end is spared."""
    cycles = _reporter_store() + [
        _cycle(f"x{i}", _normal_points(60), status="interrupted") for i in range(5)
    ]
    assert _fires(cycles, _live(_normal_points(60))) is False


def test_genuine_early_stop_still_fires() -> None:
    """A wash stopped 20 min in: earlier than any completed cycle went quiet or ended."""
    assert _fires(_reporter_store(), _live(_normal_points(60)[:41])) is True


def test_an_end_never_creates_a_baseline() -> None:
    """No completed trace shows a quiet span: no baseline, as before (never fires)."""
    normals = [_normal(i) for i in range(5)]
    assert earliest_sustained_quiet_offset(normals, STOP, 60, 3) is None


def test_a_few_watt_blip_does_not_stand_for_an_end() -> None:
    """A stored 92 s, 2.5 W "completed" blip (seen in the corpus) is not a programme
    end: only a cycle that was clearly on (peak >= 5 x stop) caps the baseline."""
    blip = _cycle("blip", [(0.0, 2.5), (31.2, 2.5), (91.8, 2.4)])
    cycles = _reporter_store() + [blip]
    assert earliest_sustained_quiet_offset(cycles, STOP, 60, 3) == min(
        c["duration"] for c in _reporter_store()[:23]
    )


# ── The real path: a booted manager, "Apply smart models" on ───────────────


async def _boot_reporter(hass):
    entry = make_entry(hass, {
        "device_type": "washing_machine",
        "enable_ml_models": True,
        "min_power": 4.0,
        "start_threshold_w": 10.0,
        "stop_threshold_w": STOP,
        "off_delay": 180,
        "min_off_gap": 480,
        "notify_finish_message": "{device}: {status}",
    })
    mgr = await boot(hass, entry)
    mgr.profile_store._data["past_cycles"].extend(_reporter_store())
    return mgr


async def _wash(hass, freezer, minutes: int) -> None:
    """Feed a normal wash cut at ``minutes`` (its end, or a stop), then 15 min of 0 W."""
    pts = _normal_points(60)
    prev = 0.0
    for t, p in pts:
        if t > minutes * 60:
            break
        await feed(hass, freezer, p, int(t - prev) or 30)
        prev = t
    await feed(hass, freezer, 0, 900)
    await hass.async_block_till_done()


async def test_reporter_normal_ends_are_stored_completed(hass, freezer):
    calls = record_notify(hass)
    mgr = await _boot_reporter(hass)
    n0 = len(mgr.profile_store.get_past_cycles())
    # Two in a row: before the fix the first one's ``interrupted`` kept it out of
    # the baseline, so the second fired too.
    for _ in range(2):
        await _wash(hass, freezer, 60)
    closed = mgr.profile_store.get_past_cycles()[n0:]
    assert len(closed) == 2, closed
    for cyc in closed:
        assert cyc["status"] == "completed", (cyc["status"], cyc["termination_reason"])
        assert cyc["termination_reason"] != TerminationReason.TERMINAL_DROP
        assert 3500 < cyc["duration"] < 3700
    # And the user hears about both (an interrupted cycle sends no finish push).
    assert sum(c.get("message") == "Washer: completed" for c in calls) == 2, calls
    await mgr.async_shutdown()


async def test_reporter_genuine_stop_is_still_a_terminal_drop(hass, freezer):
    calls = record_notify(hass)
    mgr = await _boot_reporter(hass)
    n0 = len(mgr.profile_store.get_past_cycles())
    await _wash(hass, freezer, 20)
    closed = mgr.profile_store.get_past_cycles()[n0:]
    assert len(closed) == 1, closed
    assert closed[0]["termination_reason"] == TerminationReason.TERMINAL_DROP
    assert closed[0]["status"] == "interrupted"
    assert 1100 < closed[0]["duration"] < 1300
    assert not any(str(c.get("message", "")).startswith("Washer: ") for c in calls), calls
    await mgr.async_shutdown()
