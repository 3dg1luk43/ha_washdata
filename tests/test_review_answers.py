# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Getting the few review questions that matter answered (2026-11-03 audit item 1).

The backend half of the panel's one-tap answers and its "X or Y?" chips:

- ``learning.review_choices``: the two programmes a review request offers, the
  same two for the panel's buttons, its chart overlay and the finish notification.
- ``LearningManager.async_answer_review``: a pick is a confirmation when it names
  the detected programme, a correction otherwise (the rule the panel's
  ``resolve_feedback`` calls follow).
- ``get_feedbacks`` carries the choices with their usual duration and energy, and
  the cycle's own start, duration and energy.
- ``match_rules.live_torn_choices`` / ``manager.match_choices`` / ``get_devices``:
  the two programmes a live match is torn between, by the cycle-end label margin.

The panel half is ``playwright-tests/tests/review-question-inline.spec.ts``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata import match_rules, ws_api
from custom_components.ha_washdata.const import DOMAIN, MATCH_LABEL_MIN_MARGIN
from custom_components.ha_washdata.learning import review_choices
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.profile_store import MatchResult

from .test_group_a_features import _learning_manager, mock_hass_learning  # noqa: F401

_PROFILES = {"Eco 60": {"avg_duration": 9600}, "Quick 30": {"avg_duration": 2400}, "Cotton 40": {}}


def _request(detected: str | None = "Eco 60", ranking: list | None = None) -> dict[str, Any]:
    return {
        "cycle_id": "c1",
        "detected_profile": detected,
        "ranking": ranking if ranking is not None else [
            {"name": "Eco 60", "score": 0.61}, {"name": "Quick 30", "score": 0.58},
            {"name": "Cotton 40", "score": 0.40},
        ],
    }


# ── review_choices ────────────────────────────────────────────────────────────

def test_choices_are_the_detected_programme_then_the_ranking_runner_up() -> None:
    assert review_choices(_request(), _PROFILES) == ["Eco 60", "Quick 30"]


def test_a_detected_programme_ranked_second_still_comes_first() -> None:
    # Stage 5 can pick a member the ranking does not lead with.
    assert review_choices(_request("Quick 30"), _PROFILES) == ["Quick 30", "Eco 60"]


def test_a_profile_group_is_named_by_its_scoring_member() -> None:
    req = _request(ranking=[
        {"name": "Eco 60"},
        {"name": "__group__Cottons", "group_best_member": "Cotton 40"},
    ])
    assert review_choices(req, _PROFILES) == ["Eco 60", "Cotton 40"]
    # A bare group key (an older stored record) names no profile and is skipped.
    bare = _request(ranking=[{"name": "Eco 60"}, {"name": "__group__Cottons"}, {"name": "Quick 30"}])
    assert review_choices(bare, _PROFILES) == ["Eco 60", "Quick 30"]


def test_a_deleted_profile_is_never_offered() -> None:
    assert review_choices(_request("Gone"), _PROFILES) == ["Eco 60", "Quick 30"]
    assert review_choices(_request(ranking=[{"name": "Gone"}]), _PROFILES) == ["Eco 60"]


@pytest.mark.parametrize("bad", [None, "x", 3, {"ranking": "x"}, {"ranking": [None, 5, "Eco 60"]}])
def test_choices_never_raise(bad: Any) -> None:
    assert review_choices(bad, _PROFILES) == []
    assert review_choices(_request(), None) == []


def test_the_stored_ranking_keeps_a_group_member(mock_hass_learning) -> None:  # noqa: F811
    mgr, store = _learning_manager(mock_hass_learning, labeled_count=6)
    result = SimpleNamespace(ranking=[
        {"name": "Eco 60", "score": 0.6, "group_best_member": "Eco 50"},  # a resolved Stage-5 win
        {"name": "__group__Cottons", "score": 0.55, "group_best_member": "Cotton 40"},
    ])
    mgr.request_cycle_verification("c1", "Eco 60", 0.6, 3600.0, 3500.0, match_result=result)
    ranking = store.pending["c1"]["ranking"]
    assert "group_best_member" not in ranking[0]
    assert ranking[1]["group_best_member"] == "Cotton 40"


# ── async_answer_review ───────────────────────────────────────────────────────

def _pending(mgr_store: Any) -> None:
    store = mgr_store
    store.profiles.update(_PROFILES)
    store.past_cycles.append({"id": "c1", "duration": 2700.0, "profile_name": None})
    store.pending["c1"] = {"cycle_id": "c1", "detected_profile": "Eco 60"}


async def test_picking_the_detected_programme_confirms_it(mock_hass_learning) -> None:  # noqa: F811
    mgr, store = _learning_manager(mock_hass_learning, labeled_count=6)
    _pending(store)
    assert await mgr.async_answer_review("c1", "Eco 60") is True
    rec = store.feedback["c1"]
    assert rec["user_confirmed"] is True and rec["corrected_profile"] is None
    assert store.past_cycles[0]["profile_name"] == "Eco 60"
    assert store.past_cycles[0]["label_source"] == "manual"
    assert "c1" not in store.pending


async def test_picking_the_other_programme_corrects_to_it(mock_hass_learning) -> None:  # noqa: F811
    mgr, store = _learning_manager(mock_hass_learning, labeled_count=6)
    _pending(store)
    assert await mgr.async_answer_review("c1", "Quick 30") is True
    rec = store.feedback["c1"]
    assert rec["user_confirmed"] is False and rec["corrected_profile"] == "Quick 30"
    assert store.past_cycles[0]["profile_name"] == "Quick 30"
    assert store.past_cycles[0]["label_source"] == "manual"
    assert set(store.rebuilt_profiles) == {"Quick 30", "Eco 60"}


async def test_an_unknown_programme_or_no_request_answers_nothing(mock_hass_learning) -> None:  # noqa: F811
    mgr, store = _learning_manager(mock_hass_learning, labeled_count=6)
    _pending(store)
    assert await mgr.async_answer_review("c1", "Gone") is False
    assert await mgr.async_answer_review("c1", "") is False
    assert await mgr.async_answer_review("c2", "Eco 60") is False
    assert "c1" in store.pending and store.feedback == {}


# ── get_feedbacks ─────────────────────────────────────────────────────────────

def _get_feedbacks(pending: dict[str, Any], cycles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    envelopes = {"Eco 60": {"avg_energy": 1.1}, "Quick 30": {"avg_energy": 0.22}}
    store = SimpleNamespace(
        get_pending_feedback=lambda: pending,
        get_profiles=lambda: _PROFILES,
        get_past_cycles=lambda: cycles,
        get_envelope=lambda n: envelopes.get(n),
    )
    hass = SimpleNamespace(data={DOMAIN: {"e1": SimpleNamespace(profile_store=store)}})
    connection = MagicMock()
    ws_api.ws_get_feedbacks(hass, connection, {"id": 1, "type": "x", "entry_id": "e1"})
    payload = connection.send_result.call_args[0][1]
    assert ws_api._validate_ws_contract("get_feedbacks", payload) == []
    return payload["feedbacks"]


def test_get_feedbacks_carries_the_choices_and_the_cycle() -> None:
    cycles = [{"id": "c1", "start_time": "2026-11-03T08:00:00+00:00", "duration": 2700.0,
               "energy_wh": 310.0, "profile_name": None}]
    (fb,) = _get_feedbacks({"c1": _request()}, cycles)
    assert fb["choices"] == [
        {"name": "Eco 60", "avg_duration_s": 9600, "avg_energy_kwh": 1.1},
        {"name": "Quick 30", "avg_duration_s": 2400, "avg_energy_kwh": 0.22},
    ]
    assert fb["cycle"] == {
        "start_time": "2026-11-03T08:00:00+00:00", "duration_s": 2700.0,
        "energy_kwh": 0.31, "profile_name": None,
    }
    # The stored request itself is passed through unchanged.
    assert fb["detected_profile"] == "Eco 60" and len(fb["ranking"]) == 3


def test_an_orphaned_request_has_no_cycle() -> None:
    (fb,) = _get_feedbacks({"c1": _request()}, [])
    assert fb["cycle"] is None
    assert [c["name"] for c in fb["choices"]] == ["Eco 60", "Quick 30"]


# ── live_torn_choices ─────────────────────────────────────────────────────────

def _match(margin: float, *, ambiguous: bool = False, runner: dict | None = None,
           best: str | None = "Cotton 40") -> MatchResult:
    cands: list[dict[str, Any]] = [{"name": best, "score": 0.8}] if best else []
    if runner is not False:
        cands.append({"score": 0.8 - margin, **(runner or {"name": "Eco 60"})})
    return MatchResult(best, 0.8, 3600.0, None, cands, ambiguous, margin)


def test_a_lead_under_the_label_margin_is_torn() -> None:
    m = MATCH_LABEL_MIN_MARGIN - 0.01
    assert match_rules.live_torn_choices(_match(m)) == ["Cotton 40", "Eco 60"]


def test_a_lead_at_the_label_margin_is_not_torn() -> None:
    assert match_rules.live_torn_choices(_match(MATCH_LABEL_MIN_MARGIN)) is None
    assert match_rules.live_torn_choices(_match(0.3)) is None


def test_an_ambiguous_winner_is_torn_whatever_the_margin() -> None:
    # A Stage-5 safeguard flags the winner ambiguous on a wide margin.
    assert match_rules.live_torn_choices(_match(0.3, ambiguous=True)) == ["Cotton 40", "Eco 60"]


def test_nothing_to_offer_without_a_winner_or_a_runner_up() -> None:
    assert match_rules.live_torn_choices(None) is None
    assert match_rules.live_torn_choices(_match(0.0, best=None)) is None
    assert match_rules.live_torn_choices(_match(1.0, runner=False, ambiguous=True)) is None


def test_a_torn_group_runner_up_is_named_by_its_best_member() -> None:
    m = _match(0.01, runner={"name": "__group__Eco", "group_best_member": "Eco 60"})
    assert match_rules.live_torn_choices(m) == ["Cotton 40", "Eco 60"]


@pytest.mark.parametrize("margin", [float("nan"), None, "x"])
def test_a_broken_margin_is_not_torn(margin: Any) -> None:
    result = SimpleNamespace(best_profile="Cotton 40", is_ambiguous=False, ambiguity_margin=margin,
                             candidates=[{"name": "Cotton 40"}, {"name": "Eco 60"}])
    assert match_rules.live_torn_choices(result) is None


# ── manager.match_choices + get_devices ───────────────────────────────────────

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
    mgr.profile_store.async_save = AsyncMock()
    mgr.profile_store.get_profiles = MagicMock(return_value=dict(_PROFILES))
    return mgr


def test_manager_offers_choices_only_while_a_cycle_runs_unpinned(manager: WashDataManager) -> None:
    manager._last_match_result = _match(0.02)
    manager._current_program = "Cotton 40"
    manager.detector.state = "running"
    assert manager.match_choices == ["Cotton 40", "Eco 60"]

    manager.detector.state = "off"
    assert manager.match_choices is None

    manager.detector.state = "running"
    manager._manual_program_active = True
    assert manager.match_choices is None


def test_manager_offers_only_programmes_a_chip_can_pin(manager: WashDataManager) -> None:
    # A bare group key resolves to the group's name, which set_program rejects.
    manager._last_match_result = _match(0.02, runner={"name": "__group__Eco"})
    manager.detector.state = "running"
    assert match_rules.live_torn_choices(manager._last_match_result) == ["Cotton 40", "Eco"]
    assert manager.match_choices is None


def _get_devices_with(choices: Any) -> dict[str, Any]:
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
        match_uncertainty=None, match_choices=choices,
    )
    hass = SimpleNamespace(
        config_entries=SimpleNamespace(async_entries=lambda *_a, **_k: [entry]),
        data={DOMAIN: {"e1": mgr}},
    )
    connection = MagicMock()
    with patch.object(ws_api, "_effective_level", return_value="admin"):
        ws_api.ws_get_devices(hass, connection, {"id": 1, "type": "x"})
    return connection.send_result.call_args[0][1]


def test_get_devices_carries_the_choices_within_its_contract() -> None:
    payload = _get_devices_with(["Cotton 40", "Eco 60"])
    assert payload["devices"][0]["match_choices"] == ["Cotton 40", "Eco 60"]
    assert ws_api._validate_ws_contract("get_devices", payload) == []
    for bad in (None, MagicMock(), ["Cotton 40"], "Cotton 40"):
        assert _get_devices_with(bad)["devices"][0]["match_choices"] is None
