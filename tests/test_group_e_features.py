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
"""Tests for Group E (Appliance Health & Predictive Maintenance) backend.

E1 — Maintenance log (ProfileStore CRUD + cycles-since / due / recency helpers)
E1 — Advisory suppression when maintenance was recently logged
E2 — Reminders surfaced via manager.maintenance_due + the state-sensor attribute

Pure statistics; the store helpers must never raise.
"""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata.const import (
    CONF_MAINTENANCE_REMINDER_CYCLES,
    DEFAULT_MAINTENANCE_REMINDER_CYCLES,
    MAINTENANCE_EVENT_TYPES,
    MAINTENANCE_RECENT_SUPPRESS_DAYS,
)
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.profile_store import ProfileStore
from custom_components.ha_washdata.sensor import WasherStateSensor


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def store():
    """A real ProfileStore backed by an in-memory _data dict (no file I/O)."""
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(MagicMock(), "entry")
        ps.async_save = AsyncMock()
        yield ps


def _completed_cycle(start_dt, status: str = "completed") -> dict:
    return {"start_time": start_dt.isoformat(), "status": status, "duration": 3600.0}


# ---------------------------------------------------------------------------
# Contract sanity: constants + defaults
# ---------------------------------------------------------------------------


def test_maintenance_event_types_contract():
    assert MAINTENANCE_EVENT_TYPES == (
        "descale",
        "filter_clean",
        "drum_clean",
        "bearing_service",
        "other",
    )


def test_default_reminder_dict_contract():
    assert DEFAULT_MAINTENANCE_REMINDER_CYCLES == {
        "descale": 30,
        "filter_clean": 50,
        "drum_clean": 100,
    }
    # bearing_service / other default off (absent)
    assert "bearing_service" not in DEFAULT_MAINTENANCE_REMINDER_CYCLES
    assert "other" not in DEFAULT_MAINTENANCE_REMINDER_CYCLES


# ---------------------------------------------------------------------------
# E1 — add / get / delete
# ---------------------------------------------------------------------------


async def test_add_and_get_maintenance_event(store):
    entry = await store.async_add_maintenance_event("descale", notes="citric acid")
    assert entry["event_type"] == "descale"
    assert entry["notes"] == "citric acid"
    assert entry["id"] and isinstance(entry["id"], str)
    assert entry["date"]  # defaulted to now
    store.async_save.assert_awaited()

    log = store.get_maintenance_log()
    assert len(log) == 1
    assert log[0]["id"] == entry["id"]


async def test_add_defaults_date_to_now(store):
    before = dt_util.now()
    entry = await store.async_add_maintenance_event("filter_clean")
    parsed = dt_util.parse_datetime(entry["date"])
    assert parsed is not None
    assert parsed >= before - timedelta(seconds=5)


async def test_add_invalid_event_type_raises(store):
    with pytest.raises(ValueError):
        await store.async_add_maintenance_event("bogus_type")
    # nothing persisted
    assert store.get_maintenance_log() == []


async def test_get_maintenance_log_most_recent_first(store):
    old = (dt_util.now() - timedelta(days=10)).isoformat()
    mid = (dt_util.now() - timedelta(days=5)).isoformat()
    new = dt_util.now().isoformat()
    await store.async_add_maintenance_event("descale", date=old)
    await store.async_add_maintenance_event("descale", date=new)
    await store.async_add_maintenance_event("descale", date=mid)

    log = store.get_maintenance_log()
    dates = [e["date"] for e in log]
    assert dates == [new, mid, old]


async def test_delete_maintenance_event(store):
    entry = await store.async_add_maintenance_event("drum_clean")
    assert await store.async_delete_maintenance_event(entry["id"]) is True
    assert store.get_maintenance_log() == []
    # deleting a non-existent id returns False
    assert await store.async_delete_maintenance_event("does-not-exist") is False


def test_get_maintenance_log_never_raises_on_bad_data(store):
    store._data["maintenance_log"] = "not a list"
    assert store.get_maintenance_log() == []


# ---------------------------------------------------------------------------
# E1 — cycles_since_maintenance
# ---------------------------------------------------------------------------


def test_cycles_since_maintenance_no_event_returns_total(store):
    now = dt_util.now()
    store._data["past_cycles"] = [
        _completed_cycle(now - timedelta(days=3)),
        _completed_cycle(now - timedelta(days=2)),
        _completed_cycle(now - timedelta(days=1)),
    ]
    assert store.cycles_since_maintenance("descale") == 3


async def test_cycles_since_maintenance_after_event(store):
    now = dt_util.now()
    # two cycles before the maintenance, three after
    store._data["past_cycles"] = [
        _completed_cycle(now - timedelta(days=10)),
        _completed_cycle(now - timedelta(days=9)),
        _completed_cycle(now - timedelta(days=4)),
        _completed_cycle(now - timedelta(days=3)),
        _completed_cycle(now - timedelta(days=2)),
    ]
    await store.async_add_maintenance_event(
        "descale", date=(now - timedelta(days=5)).isoformat()
    )
    assert store.cycles_since_maintenance("descale") == 3


async def test_cycles_since_maintenance_only_counts_completed(store):
    now = dt_util.now()
    store._data["past_cycles"] = [
        _completed_cycle(now - timedelta(days=2)),
        _completed_cycle(now - timedelta(days=1), status="interrupted"),
        _completed_cycle(now - timedelta(hours=1), status="force_stopped"),
    ]
    # no event → total completed only (1)
    assert store.cycles_since_maintenance("descale") == 1


async def test_cycles_since_maintenance_uses_most_recent_event(store):
    now = dt_util.now()
    store._data["past_cycles"] = [
        _completed_cycle(now - timedelta(days=6)),
        _completed_cycle(now - timedelta(days=1)),
    ]
    await store.async_add_maintenance_event(
        "descale", date=(now - timedelta(days=8)).isoformat()
    )
    await store.async_add_maintenance_event(
        "descale", date=(now - timedelta(days=3)).isoformat()
    )
    # most recent event is 3 days ago → only the 1-day-old cycle counts
    assert store.cycles_since_maintenance("descale") == 1


def test_cycles_since_maintenance_never_raises(store):
    store._data["past_cycles"] = "garbage"
    assert store.cycles_since_maintenance("descale") == 0


# ---------------------------------------------------------------------------
# E1 — get_maintenance_due
# ---------------------------------------------------------------------------


def _seed_completed(store, n: int) -> None:
    now = dt_util.now()
    store._data["past_cycles"] = [
        _completed_cycle(now - timedelta(hours=n - i)) for i in range(n)
    ]


def test_get_maintenance_due_at_threshold(store):
    _seed_completed(store, 30)
    assert store.get_maintenance_due({"descale": 30}) == ["descale"]


def test_get_maintenance_due_over_threshold(store):
    _seed_completed(store, 45)
    assert store.get_maintenance_due({"descale": 30}) == ["descale"]


def test_get_maintenance_due_under_threshold(store):
    _seed_completed(store, 29)
    assert store.get_maintenance_due({"descale": 30}) == []


def test_get_maintenance_due_zero_threshold_off(store):
    _seed_completed(store, 100)
    assert store.get_maintenance_due({"descale": 0}) == []


def test_get_maintenance_due_absent_key_off(store):
    _seed_completed(store, 100)
    # bearing_service not in the config -> not evaluated
    assert store.get_maintenance_due({"descale": 30}) == ["descale"]


def test_get_maintenance_due_multiple(store):
    _seed_completed(store, 60)
    due = store.get_maintenance_due({"descale": 30, "filter_clean": 50, "drum_clean": 100})
    assert set(due) == {"descale", "filter_clean"}  # drum_clean (100) not reached


def test_get_maintenance_due_never_raises(store):
    assert store.get_maintenance_due(None) == []
    assert store.get_maintenance_due({"descale": "not-an-int"}) == []


# ---------------------------------------------------------------------------
# E1 — has_recent_maintenance
# ---------------------------------------------------------------------------


async def test_has_recent_maintenance_within_window(store):
    await store.async_add_maintenance_event(
        "descale", date=(dt_util.now() - timedelta(days=5)).isoformat()
    )
    assert store.has_recent_maintenance("descale") is True


async def test_has_recent_maintenance_outside_window(store):
    await store.async_add_maintenance_event(
        "descale",
        date=(dt_util.now() - timedelta(days=MAINTENANCE_RECENT_SUPPRESS_DAYS + 10)).isoformat(),
    )
    assert store.has_recent_maintenance("descale") is False


async def test_has_recent_maintenance_respects_type(store):
    await store.async_add_maintenance_event("filter_clean")
    assert store.has_recent_maintenance("filter_clean") is True
    assert store.has_recent_maintenance("descale") is False


async def test_has_recent_maintenance_custom_days(store):
    await store.async_add_maintenance_event(
        "descale", date=(dt_util.now() - timedelta(days=15)).isoformat()
    )
    assert store.has_recent_maintenance("descale", days=7) is False
    assert store.has_recent_maintenance("descale", days=30) is True


# ---------------------------------------------------------------------------
# E1 — advisory suppression
# ---------------------------------------------------------------------------


def _prime_advisories(store, *, health: dict, trends: dict) -> None:
    store.compute_profile_health = lambda: health
    store.compute_profile_trends = lambda: trends


async def test_advisory_suppressed_after_recent_descale(store):
    _prime_advisories(
        store,
        health={"Cotton": {"health_status": "poor"}},
        trends={"Eco": {"duration_trend": "up", "duration_slope_pct": 0.1}},
    )
    # Without maintenance: both advisories present.
    assert {a["code"] for a in store.compute_profile_advisories()} == {
        "poor_health",
        "duration_trend_up",
    }
    # Log a descale → both nag advisories suppressed.
    await store.async_add_maintenance_event("descale")
    assert store.compute_profile_advisories() == []


async def test_advisory_suppressed_by_filter_or_drum_clean(store):
    _prime_advisories(
        store,
        health={},
        trends={"Eco": {"duration_trend": "up", "duration_slope_pct": 0.2}},
    )
    await store.async_add_maintenance_event("filter_clean")
    assert store.compute_profile_advisories() == []


async def test_advisory_not_suppressed_by_bearing_service(store):
    # bearing_service is NOT one of the descale/filter/drum triggers.
    _prime_advisories(
        store,
        health={"Cotton": {"health_status": "poor"}},
        trends={},
    )
    await store.async_add_maintenance_event("bearing_service")
    codes = {a["code"] for a in store.compute_profile_advisories()}
    assert "poor_health" in codes


async def test_advisory_energy_trend_not_suppressed(store):
    # energy_trend_up is not a "needs maintenance" nag and must survive suppression.
    _prime_advisories(
        store,
        health={},
        trends={"Cotton": {"duration_trend": "stable", "energy_trend": "up", "energy_slope_pct": 0.09}},
    )
    await store.async_add_maintenance_event("descale")
    codes = {a["code"] for a in store.compute_profile_advisories()}
    assert codes == {"energy_trend_up"}


def test_advisory_not_suppressed_when_maintenance_stale(store):
    _prime_advisories(
        store,
        health={"Cotton": {"health_status": "poor"}},
        trends={},
    )
    store._data["maintenance_log"] = [
        {
            "id": "old",
            "event_type": "descale",
            "date": (dt_util.now() - timedelta(days=200)).isoformat(),
            "notes": "",
        }
    ]
    codes = {a["code"] for a in store.compute_profile_advisories()}
    assert "poor_health" in codes


# ---------------------------------------------------------------------------
# E2 — manager.maintenance_due
# ---------------------------------------------------------------------------


def _manager_stub(options: dict, due: list[str]):
    mgr = MagicMock()
    mgr.config_entry.options = options
    mgr.profile_store.get_maintenance_due.return_value = due
    return mgr


def test_manager_maintenance_due_reads_config():
    cfg = {"descale": 5}
    mgr = _manager_stub({CONF_MAINTENANCE_REMINDER_CYCLES: cfg}, ["descale"])
    result = WashDataManager.maintenance_due.fget(mgr)
    assert result == ["descale"]
    mgr.profile_store.get_maintenance_due.assert_called_once_with(cfg)


def test_manager_maintenance_due_falls_back_to_default():
    mgr = _manager_stub({}, [])
    WashDataManager.maintenance_due.fget(mgr)
    mgr.profile_store.get_maintenance_due.assert_called_once_with(
        DEFAULT_MAINTENANCE_REMINDER_CYCLES
    )


def test_manager_maintenance_due_never_raises():
    mgr = MagicMock()
    mgr.config_entry.options = {}
    mgr.profile_store.get_maintenance_due.side_effect = RuntimeError("boom")
    assert WashDataManager.maintenance_due.fget(mgr) == []


# ---------------------------------------------------------------------------
# E2 — state sensor attribute
# ---------------------------------------------------------------------------


def _state_sensor_manager(maintenance_due):
    mgr = MagicMock()
    mgr.samples_recorded = 0
    mgr.current_program = "none"
    mgr.sub_state = "idle"
    mgr.device_type = "washer"
    mgr.cycle_anomaly = "none"
    mgr.restart_gaps = []
    mgr.maintenance_due = maintenance_due
    return mgr


def test_state_sensor_exposes_maintenance_due():
    sensor = MagicMock()
    sensor._manager = _state_sensor_manager(["descale", "filter_clean"])
    attrs = WasherStateSensor.extra_state_attributes.fget(sensor)
    assert attrs["maintenance_due"] == ["descale", "filter_clean"]


def test_state_sensor_omits_maintenance_due_when_empty():
    sensor = MagicMock()
    sensor._manager = _state_sensor_manager([])
    attrs = WasherStateSensor.extra_state_attributes.fget(sensor)
    assert "maintenance_due" not in attrs
