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

Covers Playground WebSocket commands and their pure helper logic in
``playground.py``:

- override plumbing (``build_sim_config``, ``apply_match_overrides``,
  ``finalize_sweep_1d``).

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
# Registration / RBAC wiring
# ---------------------------------------------------------------------------

def test_playground_tab_whitelisted():
    assert "playground" in ws_api._PANEL_TABS


def test_the_removed_playground_commands_are_gone():
    # 0.5.8 UI removals: the DTW visualizer and the settings presets.
    for cmd in ("get_dtw_debug", "save_playground_preset", "delete_playground_preset"):
        assert not hasattr(ws_api, f"ws_{cmd}")
        assert cmd not in ws_api._READ_WRITE_COMMANDS


def test_the_one_shot_playground_commands_are_gone():
    # Audit PLAYGROUND-17: unused by the panel (it starts registry tasks) and each
    # ran a whole batch in one executor call. The Python functions stay.
    for cmd in ("run_playground_cycle_detail", "run_playground_history", "run_playground_sweep"):
        assert cmd not in ws_api._READ_WRITE_COMMANDS
        assert not hasattr(ws_api, f"ws_{cmd}")


# ─── Playground tab-open cost (lazy suggestions) ──────────────────────────────
#
# `get_playground_settings` is the only fetch the tab makes before it can render, and it
# used to compute the auto-tuner AND ML suggestion sets on every open - statistics over
# every clean cycle - purely to label two buttons. `include_suggestions=False` keeps that
# off the critical path; the panel fetches them in the background afterwards.

@pytest.mark.asyncio
async def test_playground_settings_can_skip_suggestion_computation():
    import inspect as _inspect
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock, patch

    from custom_components.ha_washdata import ws_api
    from custom_components.ha_washdata.cycle_detector import CycleDetectorConfig

    hass = MagicMock()
    hass.data = {}

    async def _exec(fn, *a, **k):
        return await fn(*a, **k) if _inspect.iscoroutinefunction(fn) else fn(*a, **k)

    hass.async_add_executor_job = AsyncMock(side_effect=_exec)

    store = MagicMock()
    store.get_suggestions = MagicMock(return_value={"off_delay": {"value": 240}})
    ml_engine_used = MagicMock()
    manager = MagicMock()
    manager.profile_store = store
    manager.learning_manager = SimpleNamespace(suggestion_engine=ml_engine_used)
    manager.detector = SimpleNamespace(
        config=CycleDetectorConfig(min_power=2.0, off_delay=300, device_type="washing_machine")
    )
    manager._resolve_energy_price = MagicMock(return_value=None)
    entry = SimpleNamespace(entry_id="e", options={"device_type": "washing_machine"}, data={})

    async def _call(include):
        conn = MagicMock()
        sent: dict = {}
        conn.send_result = MagicMock(side_effect=lambda _i, payload: sent.update(payload))
        with patch.object(ws_api, "_get_manager", return_value=manager), \
             patch.object(ws_api, "_get_entry", return_value=entry):
            await ws_api.ws_get_playground_settings.__wrapped__(
                hass, conn, {"id": 1, "entry_id": "e", "include_suggestions": include}
            )
        return sent

    # Opted out: no suggestions computed, and the store is not even read for them.
    store.get_suggestions.reset_mock()
    lean = await _call(False)
    assert lean["classic_suggestions"] == {}
    assert store.get_suggestions.call_count == 0
    # The values the fields actually need are still there.
    assert "effective" in lean and "presets" not in lean

    # Opted in (the default for every other caller): suggestions are read.
    full = await _call(True)
    assert store.get_suggestions.call_count == 1
    assert full["classic_suggestions"].get("off_delay") == 240


def test_playground_snapshots_include_every_evidence_category():
    """A profile sampled from a backfilled cycle must still be a Playground candidate.

    The snapshot pool was built from `past_cycles + reference_cycles`, so such a profile
    produced no candidate at all and the sandbox reported the cycle as unmatched - a wrong
    answer that would have been read as a matcher problem.
    """
    from unittest.mock import MagicMock

    from custom_components.ha_washdata import playground as pg

    store = MagicMock()
    backfilled = {
        "id": "b1", "profile_name": "Cotton 40", "duration": 3600,
        "power_data": [[float(i * 60), 1500.0] for i in range(61)],
    }
    store._data = {
        "profiles": {"Cotton 40": {"avg_duration": 3600, "sample_cycle_id": "b1"}},
        "past_cycles": [], "reference_cycles": [], "backfill_cycles": [backfilled],
    }
    store.iter_evidence_cycles = MagicMock(return_value=[backfilled])
    store._grouped_snapshots = MagicMock(side_effect=lambda snaps: (snaps, {}, {}))

    snaps, _config, _members, _member_snaps = pg._build_match_snapshots(store)

    assert [s["name"] for s in snaps] == ["Cotton 40"]
    store.iter_evidence_cycles.assert_called_once()
