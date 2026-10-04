# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Audit MATCH-DECIDE-15/18: an undecided match is shown, not silent.

While the live match is undecided (no committed program, or a runner-up within
the ambiguity margin) the Status card says "Uncertain: X or Y, ~N% sure". The
figure is a display-only map of the margin (``MATCH_SURE_KNOTS``, fitted by
``devtools/margin_display_fit.py``). The program sensor keeps its raw
``detecting...`` state, which automations compare against. The cycle-end event
carries ``match_margin`` and ``label_applied`` so an automation can tell a
labelled cycle from a best guess shown for display.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import callback

from custom_components.ha_washdata import match_rules, ws_api
from custom_components.ha_washdata.const import (
    DOMAIN,
    EVENT_CYCLE_ENDED,
    MATCH_SURE_KNOTS,
    MATCH_SURE_SINGLE_CANDIDATE,
)
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.profile_store import MatchResult


def _match(best: str | None, margin: float, *, runner: dict | None = None,
           ambiguous: bool = False, conf: float = 0.8) -> MatchResult:
    cands: list[dict[str, Any]] = []
    if best:
        cands.append({"name": best, "score": conf})
    if runner is not None:
        cands.append({"score": conf - margin, **runner})
    return MatchResult(best, conf, 3600.0, None, cands, ambiguous, margin)


# ── the margin -> "~N% sure" map ──────────────────────────────────────────────

def test_the_fitted_knots_are_a_monotone_probability_map() -> None:
    xs = [x for x, _ in MATCH_SURE_KNOTS]
    ys = [y for _, y in MATCH_SURE_KNOTS]
    assert xs == sorted(xs) and len(set(xs)) == len(xs)
    assert ys == sorted(ys)
    assert all(0.0 < y < 1.0 for y in ys)
    assert 0.0 < MATCH_SURE_SINGLE_CANDIDATE < 1.0


def test_sure_pct_is_monotone_clamped_and_rounded_to_five() -> None:
    grid = [i / 1000 for i in range(0, 1001)]
    pcts = [match_rules.display_sure_pct(m) for m in grid]
    assert pcts == sorted(pcts)
    assert all(p % 5 == 0 for p in pcts)
    lo, hi = MATCH_SURE_KNOTS[0][1], MATCH_SURE_KNOTS[-1][1]
    assert pcts[0] == 5 * round(lo * 20)
    assert pcts[-1] == 5 * round(hi * 20)
    # Interpolated between knots, not stepped.
    (x0, y0), (x1, y1) = MATCH_SURE_KNOTS[2], MATCH_SURE_KNOTS[3]
    mid = (x0 + x1) / 2
    assert match_rules.display_sure_pct(mid) == 5 * round(((y0 + y1) / 2) * 20)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "x", -1.0])
def test_sure_pct_never_raises_and_reads_bad_input_as_no_lead(bad: Any) -> None:
    assert match_rules.display_sure_pct(bad) == 5 * round(MATCH_SURE_KNOTS[0][1] * 20)


def test_a_lone_candidate_gets_its_own_figure_not_the_one_sentinel() -> None:
    lone = match_rules.display_sure_pct(None)
    assert lone == 5 * round(MATCH_SURE_SINGLE_CANDIDATE * 20)
    assert lone < match_rules.display_sure_pct(1.0)


# ── when the Status card says "Uncertain" ─────────────────────────────────────

def test_an_uncommitted_match_names_the_top_two() -> None:
    u = match_rules.live_match_uncertainty(
        _match("Cotton 40", 0.03, runner={"name": "Eco 60"}), "detecting..."
    )
    assert u == {
        "top": "Cotton 40", "runner_up": "Eco 60", "margin": 0.03,
        "sure_pct": match_rules.display_sure_pct(0.03),
    }


def test_a_committed_clear_match_is_not_uncertain() -> None:
    assert match_rules.live_match_uncertainty(
        _match("Cotton 40", 0.30, runner={"name": "Eco 60"}), "Cotton 40"
    ) is None


def test_a_committed_match_with_a_close_runner_up_is_uncertain() -> None:
    u = match_rules.live_match_uncertainty(
        _match("Cotton 40", 0.02, runner={"name": "Eco 60"}, ambiguous=True), "Cotton 40"
    )
    assert u is not None and u["runner_up"] == "Eco 60"


def test_after_a_restart_the_restored_placeholder_is_not_a_commitment() -> None:
    assert match_rules.live_match_uncertainty(
        _match("Cotton 40", 0.30, runner={"name": "Eco 60"}), "restored..."
    ) is not None


def test_a_group_runner_up_is_named_by_its_best_member() -> None:
    u = match_rules.live_match_uncertainty(
        _match("Cotton 40", 0.02, runner={"name": "__group__Eco", "group_best_member": "Eco 60"}),
        "detecting...",
    )
    assert u["runner_up"] == "Eco 60"
    bare = match_rules.live_match_uncertainty(
        _match("Cotton 40", 0.02, runner={"name": "__group__Eco"}), "detecting..."
    )
    assert bare["runner_up"] == "Eco"


def test_a_lone_candidate_has_no_runner_up_and_no_margin() -> None:
    u = match_rules.live_match_uncertainty(_match("Cotton 40", 1.0), "detecting...")
    assert u["runner_up"] is None and u["margin"] is None
    assert u["sure_pct"] == match_rules.display_sure_pct(None)


def test_no_winner_shows_nothing() -> None:
    assert match_rules.live_match_uncertainty(None, "detecting...") is None
    assert match_rules.live_match_uncertainty(_match(None, 0.0), "detecting...") is None


# ── manager + get_devices ─────────────────────────────────────────────────────

@pytest.fixture
def manager(hass: Any) -> WashDataManager:
    entry = MagicMock()
    entry.entry_id = "e"
    entry.title = "W"
    entry.options = {"power_sensor": "sensor.p"}
    entry.data = {}
    hass.config_entries.async_get_entry = MagicMock(return_value=entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = WashDataManager(hass, entry)
    ps = mgr.profile_store
    ps.get_suggestions = MagicMock(return_value={})
    ps.get_profiles = MagicMock(return_value={
        "Eco 50": {"avg_duration": 9000}, "Cotton 60": {"avg_duration": 8400},
    })
    ps.async_add_cycle = AsyncMock()
    ps.async_clear_active_cycle = AsyncMock()
    ps.async_rebuild_envelope = AsyncMock()
    ps.async_save = AsyncMock()
    mgr._run_post_cycle_processing = AsyncMock()
    mgr._learning_confidence = 0.6
    mgr._auto_label_confidence = 0.9
    mgr.learning_manager.process_cycle_end = MagicMock()
    return mgr


def test_manager_reports_uncertainty_only_while_a_cycle_runs(manager: WashDataManager) -> None:
    manager._current_program = "detecting..."
    manager._last_match_result = _match("Eco 50", 0.02, runner={"name": "Cotton 60"})
    manager.detector.state = "running"
    u = manager.match_uncertainty
    assert u is not None and (u["top"], u["runner_up"]) == ("Eco 50", "Cotton 60")
    # The sensor-facing program value is untouched (automations compare on it).
    assert manager._current_program == "detecting..."

    manager.detector.state = "off"
    assert manager.match_uncertainty is None

    manager.detector.state = "running"
    manager._manual_program_active = True
    assert manager.match_uncertainty is None


def _get_devices_with(unc: Any) -> dict[str, Any]:
    entry = SimpleNamespace(entry_id="e1", title="Washer", data={}, options={})
    store = SimpleNamespace(
        get_suggestions=lambda: {}, get_locked_suggestions=lambda: [],
        get_pending_feedback=lambda: {}, get_past_cycles=lambda: [],
    )
    mgr = SimpleNamespace(
        detector=SimpleNamespace(state="running", sub_state=None),
        _current_program=None, manual_program_active=False, _time_remaining=None,
        _total_duration=None, _cycle_progress=None, profile_store=store,
        is_user_paused=False, recorder=SimpleNamespace(is_recording=False),
        match_uncertainty=unc,
    )
    hass = MagicMock()
    hass.config_entries.async_entries.return_value = [entry]
    hass.data = {DOMAIN: {"e1": mgr}}
    connection = MagicMock()
    with patch.object(ws_api, "_effective_level", return_value="admin"):
        ws_api.ws_get_devices(hass, connection, {"id": 1, "type": "x"})
    return connection.send_result.call_args[0][1]


def test_get_devices_carries_the_uncertainty_within_its_contract() -> None:
    unc = {"top": "Eco 50", "runner_up": "Cotton 60", "margin": 0.02, "sure_pct": 30}
    payload = _get_devices_with(unc)
    assert payload["devices"][0]["match_uncertainty"] == unc
    assert ws_api._validate_ws_contract("get_devices", payload) == []
    # Anything that is not a dict (a mocked manager attribute) is reported as None.
    assert _get_devices_with(MagicMock())["devices"][0]["match_uncertainty"] is None


# ── the cycle-end event ───────────────────────────────────────────────────────

def _cycle() -> dict[str, Any]:
    return {
        "id": "c1", "start_time": "2026-05-01T08:00:00+00:00", "duration": 9000.0,
        "status": "completed", "power_data": [[i * 600.0, 1500.0] for i in range(16)],
    }


async def _end_event(hass: Any, manager: WashDataManager, final: Any) -> dict[str, Any]:
    manager._notify_fire_events = True
    manager.profile_store.async_match_profile = AsyncMock(return_value=final)
    fired: list[dict[str, Any]] = []

    @callback
    def _on(event: Any) -> None:
        fired.append(event.data)

    hass.bus.async_listen(EVENT_CYCLE_ENDED, _on)
    await manager._async_process_cycle_end(_cycle())
    await hass.async_block_till_done()
    assert fired
    return fired[-1]


async def test_a_labelled_cycle_reports_its_margin(hass: Any, manager: WashDataManager) -> None:
    manager._current_program = "Cotton 60"
    data = await _end_event(hass, manager, _match("Cotton 60", 0.15, runner={"name": "Eco 50"}, conf=0.88))
    assert data["program"] == "Cotton 60"
    assert data["match_margin"] == pytest.approx(0.15)
    assert data["label_applied"] is True


async def test_a_coin_flip_is_named_for_display_but_not_labelled(
    hass: Any, manager: WashDataManager
) -> None:
    manager._current_program = "detecting..."
    data = await _end_event(hass, manager, _match("Eco 50", 0.02, runner={"name": "Cotton 60"}, conf=0.9))
    assert data["program"] == "Eco 50"
    assert data["match_margin"] == pytest.approx(0.02)
    assert data["label_applied"] is False
    assert not data["cycle_data"].get("profile_name")


async def test_no_winner_has_no_margin(hass: Any, manager: WashDataManager) -> None:
    manager._current_program = "detecting..."
    data = await _end_event(hass, manager, _match(None, 0.0))
    assert data["program"] == "unknown"
    assert data["match_margin"] is None
    assert data["label_applied"] is False


async def test_a_hand_picked_program_counts_as_applied(hass: Any, manager: WashDataManager) -> None:
    manager._current_program = "Eco 50"
    manager._manual_program_active = True
    data = await _end_event(hass, manager, _match("Cotton 60", 0.30, runner={"name": "Eco 50"}))
    assert data["program"] == "Eco 50"
    assert data["label_applied"] is True
    assert math.isclose(data["match_margin"], 0.30)
