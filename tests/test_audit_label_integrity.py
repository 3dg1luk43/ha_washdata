"""Audit 2026-10-02 MANAGER-01 / MANAGER-02 (= SUGGEST-02, MATCH-DECIDE-01): label integrity.

MANAGER-01: confirming or correcting a cycle from the review queue set only
``profile_name``, so the user's answer kept the matcher's ``label_source`` and the
panel's Auto-label (which ran with ``overwrite=True``) re-matched and replaced it,
recording the user's answer as ``original_auto_label``. It happened in the
maintainer's own store.

MANAGER-02: the cycle-end gate refused a label on margin / ambiguity, then
``learning._maybe_request_feedback`` auto-labelled the same cycle on confidence
alone - no margin, no ambiguity check, no provenance.

Real manager, real ProfileStore, real LearningManager.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.profile_store import (
    MatchResult,
    WashDataStore,
    _repair_answered_feedback_provenance,
)

from .real_manager import boot, feed, make_entry, record_notify


def _res(name: str, conf: float = 0.95, margin: float = 0.3, amb: bool = False) -> MatchResult:
    return MatchResult(
        best_profile=name, confidence=conf, expected_duration=870.0,
        matched_phase=None, candidates=[], is_ambiguous=amb, ambiguity_margin=margin,
    )


async def _one_cycle(hass, freezer, mgr):
    await feed(hass, freezer, 500, 900)
    await feed(hass, freezer, 0, 600)
    await hass.async_block_till_done()
    return mgr.profile_store.get_past_cycles()[-1]


async def test_a_review_correction_survives_bulk_auto_label(hass, freezer):
    record_notify(hass)
    mgr = await boot(hass, make_entry(hass))
    ps = mgr.profile_store
    cyc = await _one_cycle(hass, freezer, mgr)
    await ps.create_profile_standalone("Cotton", avg_duration=870)
    await ps.create_profile_standalone("Synthetics", avg_duration=870)
    cyc["profile_name"] = "Cotton"
    cyc["label_source"] = "auto_match"
    mgr.learning_manager.request_cycle_verification(cyc["id"], "Cotton", 0.8, 870.0, cyc["duration"])

    assert await mgr.learning_manager.async_submit_cycle_feedback(
        cyc["id"], user_confirmed=False, corrected_profile="Synthetics"
    )
    assert cyc["profile_name"] == "Synthetics"
    assert cyc["label_source"] == "manual"
    assert cyc["original_auto_label"] == "Cotton"

    ps.async_match_profile = AsyncMock(return_value=_res("Cotton"))
    await ps.auto_label_cycles(0.75, overwrite=True)
    assert cyc["profile_name"] == "Synthetics"
    await mgr.async_shutdown()


async def test_a_review_confirmation_is_stamped_manual(hass, freezer):
    record_notify(hass)
    mgr = await boot(hass, make_entry(hass))
    ps = mgr.profile_store
    cyc = await _one_cycle(hass, freezer, mgr)
    await ps.create_profile_standalone("Cotton", avg_duration=870)
    mgr.learning_manager.request_cycle_verification(cyc["id"], "Cotton", 0.8, 870.0, cyc["duration"])
    assert await mgr.learning_manager.async_submit_cycle_feedback(cyc["id"], user_confirmed=True)
    assert (cyc["profile_name"], cyc["label_source"]) == ("Cotton", "manual")
    await mgr.async_shutdown()


@pytest.mark.parametrize("margin,amb", [(0.02, True), (0.06, False)])
async def test_learning_does_not_label_what_the_gate_refused(hass, freezer, margin, amb):
    record_notify(hass)
    mgr = await boot(hass, make_entry(hass))
    ps = mgr.profile_store
    for _ in range(6):
        await _one_cycle(hass, freezer, mgr)
    cycles = ps.get_past_cycles()
    await ps.create_profile_standalone("Cotton", avg_duration=870)
    await ps.create_profile_standalone("Cotton Eco", avg_duration=880)
    for c in cycles[:5]:
        c["profile_name"] = "Cotton"
        c["label_source"] = "manual"
    target = dict(cycles[-1])
    target.pop("id", None)
    target.pop("profile_name", None)
    ps.get_past_cycles().pop()  # re-run its end through the real tail
    mgr._current_program = "Cotton"
    mgr._last_match_confidence = 0.95
    mgr._matched_profile_duration = 870.0
    mgr._last_match_result = _res("Cotton", 0.95, margin, amb)
    ps.async_match_profile = AsyncMock(return_value=_res("Cotton", 0.95, margin, amb))

    await mgr._async_process_cycle_end(target)
    await hass.async_block_till_done()

    stored = next(c for c in ps.get_past_cycles() if c.get("id") == target.get("id"))
    assert not stored.get("profile_name")
    # Refused, so the user is asked instead.
    assert target.get("id") in ps.get_pending_feedback()
    await mgr.async_shutdown()


async def test_panel_auto_label_never_overwrites(monkeypatch) -> None:
    from custom_components.ha_washdata import task_registry

    hass = MagicMock()
    hass.data = {}
    manager = MagicMock()
    manager.profile_store.auto_label_cycles = AsyncMock(return_value={"labeled": 2})
    monkeypatch.setattr(ws_api, "_get_manager", MagicMock(return_value=manager))
    # The WS command and the service both start this runner (audit PLATFORM-05).
    task = task_registry.get_registry(hass).create("e", "auto_label", "x")
    await ws_api._auto_label_task(hass, task, "e", 0.8)  # noqa: SLF001
    _args, kwargs = manager.profile_store.auto_label_cycles.call_args
    assert kwargs.get("overwrite") is False
    assert task.state == task_registry.STATE_DONE and task.result == {"labeled": 2}


def _store_data(**cycle: Any) -> dict[str, Any]:
    base = {"id": "c1", "profile_name": "Cotton", "label_source": "auto_match"}
    base.update(cycle)
    return {
        "past_cycles": [base],
        "feedback_history": {},
    }


def test_v15_repair_stamps_confirmed_and_corrected_answers():
    data = _store_data()
    data["feedback_history"]["c1"] = {
        "cycle_id": "c1", "user_confirmed": True, "original_detected_profile": "Cotton",
    }
    assert _repair_answered_feedback_provenance(data) == {"stamped": 1, "restored": 0}
    assert data["past_cycles"][0]["label_source"] == "manual"
    # Idempotent.
    assert _repair_answered_feedback_provenance(data) == {"stamped": 0, "restored": 0}


def test_v15_repair_restores_an_answer_auto_label_replaced():
    # The maintainer's d9166cb41c03: corrected to "Cotton 30", later replaced.
    data = _store_data(
        profile_name="800rpm", label_source="auto_label_service",
        original_auto_label="Cotton 30",
    )
    data["feedback_history"]["c1"] = {
        "cycle_id": "c1", "user_confirmed": False, "corrected_profile": "Cotton 30",
        "original_detected_profile": "800rpm",
    }
    assert _repair_answered_feedback_provenance(data) == {"stamped": 0, "restored": 1}
    cyc = data["past_cycles"][0]
    assert (cyc["profile_name"], cyc["label_source"], cyc["original_auto_label"]) == (
        "Cotton 30", "manual", "800rpm",
    )


def test_v15_repair_leaves_dismissals_and_later_manual_labels_alone():
    data = _store_data(profile_name="Wool", label_source="manual")
    data["past_cycles"].append({"id": "c2", "profile_name": "Eco", "label_source": "auto_match"})
    data["feedback_history"]["c1"] = {
        "cycle_id": "c1", "user_confirmed": False, "corrected_profile": "Cotton",
    }
    data["feedback_history"]["c2"] = {
        "cycle_id": "c2", "user_confirmed": False, "corrected_profile": None,
    }
    assert _repair_answered_feedback_provenance(data) == {"stamped": 0, "restored": 0}
    assert data["past_cycles"][0]["profile_name"] == "Wool"
    assert data["past_cycles"][1]["label_source"] == "auto_match"


async def test_v14_store_migrates_through_the_repair():
    store = WashDataStore(MagicMock(), 15, "ha_washdata.test")
    data = _store_data()
    data["feedback_history"]["c1"] = {
        "cycle_id": "c1", "user_confirmed": True, "original_detected_profile": "Cotton",
    }
    result = await store._async_migrate_func(14, 1, data)
    assert result["past_cycles"][0]["label_source"] == "manual"
