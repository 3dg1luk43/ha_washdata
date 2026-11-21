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
"""Register item 637: suggestions that changed at almost every cycle end.

The operational pass sized watchdog, no-update timeout and match interval from the
newest 200 update intervals of the newest cycle (mostly its quiet tail), so they
moved with every cycle; a chronological replay of 713 cycle ends applied them 107,
123 and 68 times. Now: one cadence summary per cycle, stored, and the median over
the last 10; a 25% hold band on those keys; seconds keys need the relative bar.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

from custom_components.ha_washdata.const import (
    CONF_COMPLETION_MIN_SECONDS,
    CONF_WATCHDOG_INTERVAL,
    CADENCE_RECENT_CYCLES,
)
from custom_components.ha_washdata.learning import (
    LearningManager,
    cycle_cadence,
    recent_cadence,
)

NOW = datetime(2026, 11, 16, 12, tzinfo=timezone.utc)


class _Store:
    def __init__(self) -> None:
        self.suggestions: dict[str, Any] = {}
        self.cadences: list[list[float]] = []

    def get_cadence_summaries(self, sensor: Any = None) -> list[list[float]]:
        return [list(r) for r in self.cadences]

    def add_cadence_summary(self, p95: float, median: float, sensor: Any = None) -> None:
        self.cadences = (self.cadences + [[p95, median]])[-CADENCE_RECENT_CYCLES:]

    def get_suggestions(self) -> dict[str, Any]:
        return self.suggestions

    def set_suggestion(self, key, value, reason=None, reason_key=None, reason_params=None, flags=None):
        self.suggestions[key] = {"value": value, "reason": reason}

    def delete_suggestion(self, key) -> None:
        self.suggestions.pop(key, None)

    def get_locked_suggestions(self) -> list[str]:
        return []

    def get_past_cycles(self) -> list[dict[str, Any]]:
        return []

    def get_lifetime_cycle_count(self) -> int:
        return 100

    def get_suggestion_apply_cycle_count(self) -> int:
        return 0

    async def async_save(self) -> None:
        return None


def _learning(mock_hass: Any, store: _Store, options: dict[str, Any]) -> LearningManager:
    entry = MagicMock()
    entry.data = {}
    entry.options = dict(options)
    mock_hass.config_entries.async_get_entry.return_value = entry
    mock_hass.async_create_task = lambda coro, *_a: coro.close()
    return LearningManager(mock_hass, "e", store, device_type="washing_machine")


def test_one_summary_per_cycle_and_the_median_over_the_last_ten() -> None:
    assert cycle_cadence([10.0] * 19) is None  # below 20 intervals
    p95, median = cycle_cadence([10.0] * 90 + [60.0] * 10)
    assert median == 10.0 and p95 == 60.0
    # One odd cycle among ten does not move the result.
    rows = [(20.0, 10.0)] * 9 + [(400.0, 30.0)]
    assert recent_cadence(rows) == (20.0, 10.0)
    assert recent_cadence([]) is None
    assert recent_cadence([("x", 1), (0.0, 5.0)]) is None  # unusable rows are skipped


def test_a_cycle_end_stores_its_summary_and_the_pass_reads_it(mock_hass: Any) -> None:
    store = _Store()
    lm = _learning(mock_hass, store, {})
    for _ in range(25):
        lm.process_power_reading(100.0, NOW, NOW)  # same stamp: ignored as jitter
    lm._pending_intervals.extend((30.0, NOW) for _ in range(250))  # noqa: SLF001
    assert lm.close_cycle_cadence({"status": "completed", "termination_reason": "timeout"})
    assert store.cadences == [[30.0, 30.0]]
    assert lm.cadence() == (30.0, 30.0)
    # A force stop commits nothing.
    lm._pending_intervals.extend((300.0, NOW) for _ in range(250))  # noqa: SLF001
    assert not lm.close_cycle_cadence({"status": "force_stopped"})
    assert store.cadences == [[30.0, 30.0]]


def test_a_whole_cycle_is_held_not_its_last_200_intervals(mock_hass: Any) -> None:
    store = _Store()
    lm = _learning(mock_hass, store, {})
    lm._pending_intervals.extend((10.0, NOW) for _ in range(900))  # noqa: SLF001
    lm._pending_intervals.extend((60.0, NOW) for _ in range(150))  # noqa: SLF001
    lm.close_cycle_cadence({"status": "completed", "termination_reason": "timeout"})
    # The quiet tail (60 s) is the last 150 intervals; over the whole cycle the
    # median stays the 10 s working cadence.
    assert store.cadences[-1][1] == 10.0


def test_the_hold_band_and_the_seconds_bars(mock_hass: Any) -> None:
    store = _Store()
    lm = _learning(mock_hass, store, {CONF_WATCHDOG_INTERVAL: 63, CONF_COMPLETION_MIN_SECONDS: 2692})
    lm.suggestion_engine.apply_suggestions = MagicMock()
    lm._apply_suggestions_and_notify({  # noqa: SLF001
        CONF_WATCHDOG_INTERVAL: {"value": 75, "reason": "r"},          # +19%: held
        CONF_COMPLETION_MIN_SECONDS: {"value": 2698, "reason": "r"},   # +6 s, 0.2%: noise
    })
    lm.suggestion_engine.apply_suggestions.assert_not_called()
    lm._apply_suggestions_and_notify({CONF_WATCHDOG_INTERVAL: {"value": 83, "reason": "r"}})  # +32%
    (sent,), _kw = lm.suggestion_engine.apply_suggestions.call_args
    assert set(sent) == {CONF_WATCHDOG_INTERVAL}
    # A corrective is not held by the band.
    lm.suggestion_engine.apply_suggestions.reset_mock()
    lm._apply_suggestions_and_notify({  # noqa: SLF001
        CONF_WATCHDOG_INTERVAL: {"value": 70, "reason": "r", "corrective": True},
    })
    lm.suggestion_engine.apply_suggestions.assert_called_once()


def test_an_oversized_stored_row_is_skipped_not_fatal() -> None:
    """PR #529 review: float(10**400) raises OverflowError; one bad imported row
    must not throw away the others."""
    rows = [[10**400, 5.0], [20.0, 10.0], [float("inf"), 10.0]]
    assert recent_cadence(rows) == (20.0, 10.0)


def test_a_whole_cycle_commits_in_one_update(mock_hass: Any) -> None:
    """PR #529 review: one stats recompute per committed cycle, not one per interval."""
    lm = _learning(mock_hass, _Store(), {})
    calls = []
    original = lm._sample_interval_model._compute_stats  # noqa: SLF001

    def _count() -> None:
        calls.append(1)
        original()

    lm._sample_interval_model._compute_stats = _count  # noqa: SLF001
    lm._pending_intervals.extend((10.0, NOW) for _ in range(3000))  # noqa: SLF001
    lm.close_cycle_cadence({"status": "completed", "termination_reason": "timeout"})
    assert len(calls) == 1
    assert lm._sample_interval_model.count == 200  # noqa: SLF001
