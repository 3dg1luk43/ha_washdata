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
"""#458: force-stopped cycles kept producing suggestions, and none of them fixed it.

The reporter accepted a stop threshold of 1.76 W on a washer that idles at 2.2 W -
exactly 0.8 x the standby draw, because the batch anchor is 0.8 x the p05 lowest
active power and standby-level samples sat inside the stored cycles (the same
arithmetic gave #445 its 2.56 W on a 3.2 W idle). After that no cycle could end on
its own: every one ran into the 6 h ceiling.

Trace-derived suggestions already skipped force-stopped cycles. What did not:
the cadence model, fed for hours by a plug reporting standby during the stuck
cycle; and the stop threshold itself, which nothing ever moved back above standby
even though the advisory had measured it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata.const import (
    CONF_OFF_DELAY,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
)
from custom_components.ha_washdata.learning import LearningManager, _ended_on_its_own
from custom_components.ha_washdata.suggestion_engine import (
    SuggestionEngine,
    apply_standby_floor,
    reconcile_suggestions,
    standby_stop_floor,
)

IDLE = 2.2
OPTIONS = {CONF_STOP_THRESHOLD_W: 1.76, CONF_START_THRESHOLD_W: 2.31, CONF_OFF_DELAY: 180}


def _clean(cid: str, *, low_w: float = 5.0, low_s: int = 300) -> dict[str, Any]:
    """A cycle from before the bad threshold: a pre-roll sample at standby (the
    one that drags the p05 anchor down to 2.2 W), wash, a low phase, spin."""
    pts: list[list[float]] = [[0.0, IDLE]]
    t = 30.0
    for _ in range(60):
        pts.append([t, 1900.0]); t += 30.0
    for _ in range(low_s // 30):
        pts.append([t, low_w]); t += 30.0
    for _ in range(40):
        pts.append([t, 400.0]); t += 30.0
    return {
        "id": cid, "profile_name": "Cotton", "status": "completed",
        "termination_reason": "timeout", "duration": t,
        "start_time": "2026-09-01T10:00:00+00:00", "power_data": pts,
    }


def _force_stopped(cid: str) -> dict[str, Any]:
    """Six hours, the last few of them at the 2.2 W standby."""
    pts = [[float(t), 1900.0] for t in range(0, 3600, 60)]
    pts += [[float(t), IDLE] for t in range(3600, 21600, 300)]
    return {
        "id": cid, "status": "force_stopped", "termination_reason": "force_stopped",
        "duration": 21600.0, "start_time": "2026-09-20T10:00:00+00:00", "power_data": pts,
    }


def _history(clean: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return clean + [_force_stopped(f"fs{i}") for i in range(8)]


def _engine(cycles: list[dict[str, Any]], options: dict[str, Any] | None = None) -> SuggestionEngine:
    hass = MagicMock()
    entry = MagicMock()
    entry.data = {}
    entry.options = dict(options or OPTIONS)
    hass.config_entries.async_get_entry.return_value = entry
    store = MagicMock()
    store.get_past_cycles.return_value = cycles
    store.get_profiles.return_value = {}
    store.get_suggestions.return_value = {}
    return SuggestionEngine(hass, "entry1", store, device_type="washing_machine")


# ---------------------------------------------------------------------------
# The floor
# ---------------------------------------------------------------------------


def test_the_floor_sits_above_standby_and_clears_the_wash() -> None:
    floor = standby_stop_floor(_history([_clean(f"c{i}") for i in range(6)]), 1.76, 180)
    assert floor is not None
    assert floor["idle_w"] == pytest.approx(IDLE)
    assert floor["floor_w"] == pytest.approx(2.75)
    assert floor["safe"] is True


def test_the_pass_raises_the_stop_threshold_and_the_start_with_it() -> None:
    out = _engine(_history([_clean(f"c{i}") for i in range(6)])).generate_standby_floor_suggestions()
    assert out[CONF_STOP_THRESHOLD_W]["value"] == pytest.approx(2.75)
    assert out[CONF_START_THRESHOLD_W]["value"] == pytest.approx(3.44)
    assert out[CONF_STOP_THRESHOLD_W]["corrective"] is True
    assert out[CONF_STOP_THRESHOLD_W]["reason_key"] == "suggestion.reason.standby_floor"


def test_the_raised_pair_survives_reconciliation() -> None:
    """Start is primary in Rule 1a; raising stop alone would be pulled back to
    0.8 x start, still under standby."""
    out = _engine(_history([_clean(f"c{i}") for i in range(6)])).generate_standby_floor_suggestions()
    adjusted, changed = reconcile_suggestions(dict(out), OPTIONS)
    assert adjusted[CONF_STOP_THRESHOLD_W]["value"] == pytest.approx(2.75)
    assert CONF_STOP_THRESHOLD_W not in changed


def test_the_batch_anchor_no_longer_lands_under_standby() -> None:
    clean = [_clean(f"c{i}") for i in range(6)]
    plain = _engine(clean).run_batch_simulation(clean)
    # The fixture reproduces the report: 0.8 x 2.2 W.
    assert plain[CONF_STOP_THRESHOLD_W]["value"] == pytest.approx(1.76)

    floored = _engine(_history(clean)).run_batch_simulation(_history(clean))
    assert floored[CONF_STOP_THRESHOLD_W]["value"] >= 2.75
    assert floored[CONF_START_THRESHOLD_W]["value"] >= 3.44


def test_a_wash_that_pauses_at_standby_level_is_left_to_the_advisory() -> None:
    """#445's Miele: its dips between tumble bursts ARE its standby level, so a
    threshold above standby would end its cycles mid-wash."""
    hist = _history([_clean(f"c{i}", low_w=2.4, low_s=300) for i in range(6)])
    floor = standby_stop_floor(hist, 1.76, 180)
    assert floor is not None and floor["safe"] is False
    assert _engine(hist).generate_standby_floor_suggestions() == {}


def test_without_a_safe_floor_a_stop_under_standby_is_still_dropped() -> None:
    unsafe = {"idle_w": IDLE, "floor_w": 2.75, "safe": False}
    out = apply_standby_floor({CONF_STOP_THRESHOLD_W: {"value": 1.76, "reason": "x"}}, unsafe)
    assert CONF_STOP_THRESHOLD_W not in out


def test_too_little_clean_history_proposes_nothing() -> None:
    hist = _history([_clean(f"c{i}") for i in range(3)])
    assert standby_stop_floor(hist, 1.76, 180)["safe"] is False


def test_an_appliance_that_reaches_zero_gets_no_floor() -> None:
    clean = [_clean(f"c{i}") for i in range(6)]
    assert standby_stop_floor(clean, 1.76, 180) is None
    assert _engine(clean).generate_standby_floor_suggestions() == {}


# ---------------------------------------------------------------------------
# The cadence model
# ---------------------------------------------------------------------------


def _learning() -> LearningManager:
    store = MagicMock()
    store.get_past_cycles.return_value = []
    lm = LearningManager(MagicMock(), "entry1", store, device_type="washing_machine")
    lm._update_operational_suggestions = MagicMock()
    return lm


def _feed(lm: LearningManager, gap_s: float, n: int) -> None:
    now = datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc)
    for _ in range(n):
        lm.process_power_reading(IDLE, now, now - timedelta(seconds=gap_s))
        now += timedelta(seconds=gap_s)


@pytest.mark.parametrize(
    "cycle", [
        {"status": "force_stopped", "termination_reason": "force_stopped"},
        {"status": "completed", "termination_reason": "user"},
        {"status": "interrupted", "termination_reason": "timeout"},
    ],
)
def test_a_cycle_that_did_not_end_on_its_own_teaches_no_cadence(cycle) -> None:
    lm = _learning()
    _feed(lm, 300.0, 72)  # six hours of standby heartbeats
    assert lm.close_cycle_cadence(cycle) is False
    assert lm._sample_interval_model.count == 0


def test_a_clean_cycle_commits_its_cadence() -> None:
    lm = _learning()
    _feed(lm, 5.0, 40)
    lm._update_operational_suggestions.reset_mock()  # the periodic mid-cycle refresh
    assert lm.close_cycle_cadence({"status": "completed", "termination_reason": "smart"})
    assert lm._sample_interval_model.count == 40
    lm._update_operational_suggestions.assert_called_once()


def test_dropped_intervals_never_reach_the_next_cycle() -> None:
    lm = _learning()
    _feed(lm, 300.0, 30)
    lm.close_cycle_cadence({"status": "force_stopped", "termination_reason": "force_stopped"})
    _feed(lm, 5.0, 25)
    lm.close_cycle_cadence({"status": "completed", "termination_reason": "timeout"})
    assert lm._sample_interval_model.count == 25
    assert lm._sample_interval_model.p95 == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Re-runs and the cooldown
# ---------------------------------------------------------------------------


def test_only_cycles_that_ended_on_their_own_count_as_new_evidence() -> None:
    assert _ended_on_its_own({"status": "completed", "termination_reason": "timeout"})
    assert not _ended_on_its_own({"status": "force_stopped"})
    assert not _ended_on_its_own({"status": "completed", "termination_reason": "user"})


def test_a_correction_is_not_held_back_by_the_cooldown() -> None:
    """The cooldown waits for cycles under the new settings. Under a threshold no
    cycle can finish under, it would wait for ever."""
    lm = _learning()
    store = lm.profile_store
    store.get_past_cycles.return_value = [{}] * 10
    store.get_lifetime_cycle_count.return_value = 10  # the cooldown reads the odometer
    store.get_suggestion_apply_cycle_count.return_value = 9  # applied one cycle ago
    store.get_locked_suggestions.return_value = []
    store.get_suggestions.return_value = {}
    entry = MagicMock()
    entry.data = {}
    entry.options = dict(OPTIONS)
    lm.hass.config_entries.async_get_entry.return_value = entry
    lm.suggestion_engine.apply_suggestions = MagicMock()

    lm._apply_suggestions_and_notify({
        CONF_STOP_THRESHOLD_W: {"value": 2.75, "reason": "x", "corrective": True},
        CONF_OFF_DELAY: {"value": 600, "reason": "y"},
    })

    applied = lm.suggestion_engine.apply_suggestions.call_args.args[0]
    assert CONF_STOP_THRESHOLD_W in applied
    assert CONF_OFF_DELAY not in applied
