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
"""Tests for Group F3 backend — the Playground tab.

Covers the two new WebSocket commands and their pure helper logic in
``playground.py``:

- ``run_playground_simulation`` (via :func:`playground.run_playground_batch`)
  replays stored cycles through a fresh headless :class:`CycleDetector` with the
  device's settings (optionally overridden) and returns a per-cycle event log +
  outcome plus an aggregate summary.
- ``get_dtw_debug`` (via :func:`playground.dtw_debug_payload`) returns the
  Stage 2 / DTW / Stage 4 score breakdown, the two resampled traces on a shared
  grid, and the DTW warping path for one cycle vs one profile.

Fast, pure-unit tests (no HA boot, no file I/O).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from custom_components.ha_washdata import playground, ws_api
from custom_components.ha_washdata.const import (
    CONF_COMPLETION_MIN_SECONDS,
    CONF_MIN_OFF_GAP,
    CONF_OFF_DELAY,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    DOMAIN,
)
from custom_components.ha_washdata.cycle_detector import CycleDetectorConfig
from custom_components.ha_washdata.profile_store import ProfileStore


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_trace(dur_s: int = 3600, dt: int = 30, peak: float = 2000.0, base: float = 80.0):
    """A washer-shaped [[offset, power], ...] trace: heat, wash, spin, wash."""
    pts: list[list[float]] = []
    t = 0.0
    while t <= dur_s:
        frac = t / dur_s
        if frac < 0.2:
            p = peak
        elif frac < 0.7:
            p = base
        elif frac < 0.9:
            p = 400.0
        else:
            p = base
        pts.append([round(t, 1), p])
        t += dt
    return pts


def _make_cycle(cid: str, day: int, *, label: str = "Cotton 40", dur: int = 3600) -> dict:
    return {
        "id": cid,
        "start_time": f"2024-01-{day:02d}T00:00:00+00:00",
        "duration": float(dur),
        "profile_name": label,
        "status": "completed",
        "power_data": _make_trace(dur),
    }


def _make_store(cycles: list[dict], profiles: dict) -> ProfileStore:
    """Real ProfileStore with storage stubbed out and _data pre-populated."""
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(MagicMock(), "entry")
    ps._data["past_cycles"] = cycles
    ps._data["profiles"] = profiles
    return ps


def _base_config(**overrides) -> CycleDetectorConfig:
    cfg = dict(
        min_power=5.0,
        off_delay=60,
        completion_min_seconds=600,
        min_off_gap=60,
        start_threshold_w=10.0,
        stop_threshold_w=5.0,
        end_repeat_count=1,
    )
    cfg.update(overrides)
    return CycleDetectorConfig(**cfg)


def _default_store() -> ProfileStore:
    c1 = _make_cycle("c1", 1)
    c2 = _make_cycle("c2", 2)
    return _make_store([c1, c2], {"Cotton 40": {"sample_cycle_id": "c1", "avg_duration": 3600.0}})


# ---------------------------------------------------------------------------
# build_sim_config
# ---------------------------------------------------------------------------

def test_build_sim_config_applies_known_keys():
    base = _base_config()
    out = playground.build_sim_config(
        base,
        {
            CONF_OFF_DELAY: 120,
            CONF_STOP_THRESHOLD_W: 25.0,
            CONF_MIN_OFF_GAP: 480,
            CONF_COMPLETION_MIN_SECONDS: 900,
        },
    )
    assert out.off_delay == 120
    assert out.stop_threshold_w == 25.0
    assert out.min_off_gap == 480
    assert out.completion_min_seconds == 900
    # base object is untouched
    assert base.off_delay == 60 and base.stop_threshold_w == 5.0


def test_build_sim_config_ignores_unknown_and_bad_values():
    base = _base_config()
    out = playground.build_sim_config(
        base,
        {
            "totally_unknown_key": 999,
            CONF_START_THRESHOLD_W: "not-a-number",  # un-coercible -> ignored
            CONF_OFF_DELAY: None,  # None -> ignored
        },
    )
    # nothing valid changed -> same values as base
    assert out.start_threshold_w == base.start_threshold_w
    assert out.off_delay == base.off_delay


def test_build_sim_config_empty_override_returns_base():
    base = _base_config()
    assert playground.build_sim_config(base, {}) is base
    assert playground.build_sim_config(base, None) is base


# ---------------------------------------------------------------------------
# run_playground_batch
# ---------------------------------------------------------------------------

def test_batch_detects_and_matches_labeled_cycles():
    store = _default_store()
    res = playground.run_playground_batch(
        store, ["c1", "c2"], _base_config(), {}, concurrency=5
    )
    summary = res["summary"]
    assert summary["cycles"] == 2
    assert summary["detected"] == 2
    assert summary["missed"] == 0
    assert summary["match_correct"] == 2
    assert summary["match_wrong"] == 0
    assert summary["unmatched"] == 0

    for r in res["results"]:
        oc = r["outcome"]
        assert oc["detected"] is True
        assert oc["match_profile"] == "Cotton 40"
        assert oc["match_correct"] is True
        assert oc["termination_reason"] == "timeout"
        assert oc["stored_duration_s"] == 3600.0
        # event log has state transitions + a match event + an end event
        types = {e["type"] for e in r["events"]}
        assert "state" in types
        assert "matched" in types
        assert "end" in types
        # each event carries the {t, type, detail} shape
        for e in r["events"]:
            assert set(e) == {"t", "type", "detail"}


def test_batch_settings_override_changes_outcome():
    """A start threshold above the trace's peak means the cycle never starts."""
    store = _default_store()
    baseline = playground.run_playground_batch(store, ["c1"], _base_config(), {}, 1)
    assert baseline["results"][0]["outcome"]["detected"] is True

    overridden = playground.run_playground_batch(
        store,
        ["c1"],
        _base_config(),
        {CONF_START_THRESHOLD_W: 5000.0, CONF_STOP_THRESHOLD_W: 4000.0},
        1,
    )
    oc = overridden["results"][0]["outcome"]
    assert oc["detected"] is False
    assert oc["match_correct"] is None
    assert overridden["summary"]["detected"] == 0
    assert overridden["summary"]["missed"] == 1


def test_batch_empty_cycle_ids_defaults_to_recent():
    # Build 25 cycles; recent-20 default should be applied.
    cycles = [_make_cycle(f"c{i}", (i % 27) + 1) for i in range(25)]
    store = _make_store(cycles, {"Cotton 40": {"sample_cycle_id": "c0", "avg_duration": 3600.0}})
    res = playground.run_playground_batch(store, [], _base_config(), {}, concurrency=50)
    # capped at the most recent DEFAULT_RECENT_CYCLES
    assert res["summary"]["requested"] == playground.DEFAULT_RECENT_CYCLES
    assert res["summary"]["cycles"] == playground.DEFAULT_RECENT_CYCLES
    run_ids = {r["cycle_id"] for r in res["results"]}
    # the newest cycles are the tail of the list
    assert run_ids == {c["id"] for c in cycles[-playground.DEFAULT_RECENT_CYCLES:]}


def test_batch_concurrency_clamped_and_caps_batch_size():
    store = _default_store()
    # concurrency below range -> clamped to 1, and only 1 of 2 selected cycles run
    low = playground.run_playground_batch(store, ["c1", "c2"], _base_config(), {}, 0)
    assert low["summary"]["concurrency"] == 1
    assert low["summary"]["cycles"] == 1
    assert low["summary"]["requested"] == 2

    # concurrency above range -> clamped to MAX_BATCH_CYCLES
    high = playground.run_playground_batch(store, [], _base_config(), {}, 999)
    assert high["summary"]["concurrency"] == playground.MAX_BATCH_CYCLES


def test_batch_unknown_cycle_ids_skipped():
    store = _default_store()
    res = playground.run_playground_batch(
        store, ["c1", "ghost", "c2"], _base_config(), {}, concurrency=5
    )
    assert res["summary"]["skipped_ids"] == ["ghost"]
    assert res["summary"]["cycles"] == 2
    assert {r["cycle_id"] for r in res["results"]} == {"c1", "c2"}


def test_batch_empty_store_is_graceful():
    store = _make_store([], {})
    res = playground.run_playground_batch(store, ["nope"], _base_config(), {}, 1)
    assert res["results"] == []
    assert res["summary"]["cycles"] == 0
    assert res["summary"]["skipped_ids"] == ["nope"]


# ---------------------------------------------------------------------------
# dtw_debug_payload
# ---------------------------------------------------------------------------

def test_dtw_debug_returns_full_breakdown():
    store = _default_store()
    out = playground.dtw_debug_payload(store, "c1", "Cotton 40")

    assert out["profile_name"] == "Cotton 40"
    assert out["grid_n"] == playground.MATCH_DTW_RESAMPLE_N
    assert len(out["cycle_trace"]) == out["grid_n"]
    assert len(out["profile_trace"]) == out["grid_n"]
    # every trace point is [t, w]
    assert all(len(p) == 2 for p in out["cycle_trace"])

    for key in ("correlation", "mae_score", "score"):
        assert key in out["stage2"]
    for key in ("l1_score", "ddtw_score", "blend_weight", "blended_score"):
        assert key in out["dtw"]
    for key in ("duration_agreement", "energy_agreement", "final_score"):
        assert key in out["stage4"]

    # warp path is a list of [i, j] index pairs
    assert out["warp_path"]
    assert all(len(p) == 2 for p in out["warp_path"])

    # c1 IS the profile's own sample cycle -> scores are (near) perfect
    assert out["stage2"]["score"] == pytest.approx(1.0, abs=1e-6)
    assert out["stage4"]["final_score"] == pytest.approx(1.0, abs=1e-6)


def test_dtw_debug_defaults_profile_to_cycle_label():
    store = _default_store()
    out = playground.dtw_debug_payload(store, "c2", None)
    assert out["profile_name"] == "Cotton 40"
    assert "stage2" in out


def test_dtw_debug_missing_cycle_errors():
    store = _default_store()
    out = playground.dtw_debug_payload(store, "does-not-exist", None)
    assert out == {"error": "cycle_not_found"}


def test_dtw_debug_missing_profile_errors():
    store = _default_store()
    out = playground.dtw_debug_payload(store, "c1", "Nonexistent Profile")
    assert out["error"] == "profile_not_found"
    assert out["profile_name"] == "Nonexistent Profile"


def test_dtw_debug_unlabeled_cycle_no_profile_errors():
    cycle = _make_cycle("u1", 1)
    cycle["profile_name"] = None
    cycle["label"] = None
    store = _make_store([cycle], {})
    out = playground.dtw_debug_payload(store, "u1", None)
    assert out["error"] == "no_profile"


# ---------------------------------------------------------------------------
# WS handler wiring
# ---------------------------------------------------------------------------

def _make_hass_with_manager(store: ProfileStore, base_config: CycleDetectorConfig):
    manager = MagicMock()
    manager.profile_store = store
    manager.detector.config = base_config

    hass = MagicMock()
    hass.data = {DOMAIN: {"e1": manager}}
    hass.config_entries.async_entries.return_value = []

    async def _exec(fn, *args):
        return fn(*args)

    hass.async_add_executor_job = _exec
    return hass


async def test_ws_run_playground_simulation_sends_result():
    store = _default_store()
    hass = _make_hass_with_manager(store, _base_config())
    connection = MagicMock()
    msg = {
        "id": 7,
        "entry_id": "e1",
        "cycle_ids": ["c1"],
        "settings_override": {},
        "concurrency": 1,
    }
    await ws_api.ws_run_playground_simulation.__wrapped__(hass, connection, msg)

    connection.send_result.assert_called_once()
    payload = connection.send_result.call_args[0][1]
    assert "results" in payload and "summary" in payload
    assert payload["summary"]["cycles"] == 1
    connection.send_error.assert_not_called()


async def test_ws_run_playground_simulation_no_manager():
    hass = MagicMock()
    hass.data = {DOMAIN: {}}
    connection = MagicMock()
    msg = {"id": 1, "entry_id": "missing", "cycle_ids": [], "settings_override": {}, "concurrency": 1}
    await ws_api.ws_run_playground_simulation.__wrapped__(hass, connection, msg)

    connection.send_error.assert_called_once()
    assert connection.send_error.call_args[0][1] == "not_found"


async def test_ws_get_dtw_debug_sends_result():
    store = _default_store()
    hass = _make_hass_with_manager(store, _base_config())
    connection = MagicMock()
    msg = {"id": 3, "entry_id": "e1", "cycle_id": "c1", "profile_name": "Cotton 40"}
    await ws_api.ws_get_dtw_debug.__wrapped__(hass, connection, msg)

    connection.send_result.assert_called_once()
    payload = connection.send_result.call_args[0][1]
    assert payload["profile_name"] == "Cotton 40"
    assert "warp_path" in payload
    connection.send_error.assert_not_called()


async def test_ws_get_dtw_debug_missing_cycle_sends_error():
    store = _default_store()
    hass = _make_hass_with_manager(store, _base_config())
    connection = MagicMock()
    msg = {"id": 4, "entry_id": "e1", "cycle_id": "ghost"}
    await ws_api.ws_get_dtw_debug.__wrapped__(hass, connection, msg)

    connection.send_result.assert_not_called()
    connection.send_error.assert_called_once()
    assert connection.send_error.call_args[0][1] == "cycle_not_found"


async def test_ws_get_dtw_debug_no_manager():
    hass = MagicMock()
    hass.data = {DOMAIN: {}}
    connection = MagicMock()
    msg = {"id": 1, "entry_id": "missing", "cycle_id": "c1"}
    await ws_api.ws_get_dtw_debug.__wrapped__(hass, connection, msg)
    connection.send_error.assert_called_once()
    assert connection.send_error.call_args[0][1] == "not_found"


# ---------------------------------------------------------------------------
# Registration / RBAC wiring
# ---------------------------------------------------------------------------

def test_playground_tab_whitelisted():
    assert "playground" in ws_api._PANEL_TABS


def test_playground_simulation_is_read_level():
    # run_playground_simulation does not start with get_, so it must be
    # explicitly whitelisted to gate at the 'read' level.
    assert "run_playground_simulation" in ws_api._READ_WRITE_COMMANDS
