# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Audit item 1, UX 3: a review question answered from the finish notification.

A cycle end that raises a review request puts two mobile actions on the finish
notification the user already gets, named after the request's top two ranked
programmes. A tap resolves the request the way the panel's ``resolve_feedback``
does (confirm the request's programme, correct to the other), stamped
``manual``. The learning pass now runs BEFORE the finish notification, so the
request exists when the payload is built.

Real manager, real ProfileStore, real LearningManager; the notify services are
registered with Home Assistant's real notify schema, so the payload is validated
by the bus itself.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from homeassistant.components.notify.const import NOTIFY_SERVICE_SCHEMA

from custom_components.ha_washdata.const import DOMAIN
from custom_components.ha_washdata.learning import (
    resolve_review_answer,
    review_answer_choices,
)
from custom_components.ha_washdata.profile_store import MatchResult

from .real_manager import boot, feed, make_entry

_EVENT = "mobile_app_notification_action"
_PROFILES = {"Cotton": {}, "Cotton Eco": {}, "Synthetics": {}}


def _req(detected: str | None, *names: str) -> dict[str, Any]:
    return {
        "detected_profile": detected,
        "ranking": [{"name": n, "score": 0.9 - 0.01 * i} for i, n in enumerate(names)],
    }


# ── the pure choice / resolve rules ───────────────────────────────────────────

def test_choices_are_the_ranked_top_two_with_confirm_for_the_detected() -> None:
    req = _req("Cotton", "Cotton", "Cotton Eco", "Synthetics")
    assert review_answer_choices(req, _PROFILES) == [("C", "Cotton"), ("R1", "Cotton Eco")]


def test_choices_skip_group_keys_deleted_and_repeated_names() -> None:
    # The panel's rule (learning.review_choices): the detected programme first, then
    # the next distinct existing one; a group key without its best member is skipped.
    req = _req("Cotton", "__group__Cottons", "Gone", "Cotton", "Cotton", "Synthetics")
    assert review_answer_choices(req, _PROFILES) == [("C", "Cotton"), ("R1", "Synthetics")]


def test_choices_fill_in_the_detected_programme_when_the_ranking_is_short() -> None:
    assert review_answer_choices(_req("Cotton"), _PROFILES) == [("C", "Cotton")]
    assert review_answer_choices(_req("Cotton", "Synthetics"), _PROFILES) == [
        ("C", "Cotton"), ("R1", "Synthetics"),
    ]
    # Ranked below two others, the detected programme still comes first: the
    # notification offers the same two answers as the panel's buttons.
    assert review_answer_choices(
        _req("Cotton", "Synthetics", "Cotton Eco", "Cotton"), _PROFILES
    ) == [("C", "Cotton"), ("R1", "Synthetics")]


def test_choices_never_raise() -> None:
    for bad in (None, "x", {"ranking": "x"}, {"ranking": [None, 3]}, {}):
        assert review_answer_choices(bad, _PROFILES) == []
    assert review_answer_choices(_req("Gone", "Gone"), _PROFILES) == []


def test_resolve_confirms_the_detected_and_corrects_to_the_other() -> None:
    req = _req("Cotton", "Cotton", "Cotton Eco")
    # The key resolves to a programme; LearningManager.async_answer_review then
    # confirms the detected one and corrects to any other.
    assert resolve_review_answer(req, "C", _PROFILES) == "Cotton"
    assert resolve_review_answer(req, "R1", _PROFILES) == "Cotton Eco"
    assert resolve_review_answer(req, "R0", _PROFILES) is None


def test_resolve_rejects_malformed_keys_and_vanished_programmes() -> None:
    req = _req("Cotton", "Cotton", "Gone")
    for key in ("", "R", "Rx", "R9", "R-1", "R²", "X", None, 1):
        assert resolve_review_answer(req, key, _PROFILES) is None
    assert resolve_review_answer(req, "R1", _PROFILES) is None
    assert resolve_review_answer(_req("Gone"), "C", _PROFILES) is None


# ── the real manager ──────────────────────────────────────────────────────────

def _record(hass) -> dict[str, list[dict]]:
    """A mobile and a plain notify target, both validated by the real schema."""
    calls: dict[str, list[dict]] = {"mobile": [], "plain": []}

    def _svc(kind: str):
        async def _handler(call):
            calls[kind].append(dict(call.data))
        return _handler

    hass.services.async_register(
        "notify", "mobile_app_phone", _svc("mobile"), schema=NOTIFY_SERVICE_SCHEMA
    )
    hass.services.async_register(
        "notify", "plain_phone", _svc("plain"), schema=NOTIFY_SERVICE_SCHEMA
    )
    return calls


def _entry(hass, **options):
    return make_entry(hass, {
        "notify_finish_services": ["notify.mobile_app_phone", "notify.plain_phone"],
        **options,
    })


def _res(margin: float) -> MatchResult:
    return MatchResult(
        best_profile="Cotton", confidence=0.95, expected_duration=870.0,
        matched_phase=None, candidates=[], is_ambiguous=margin < 0.05,
        ambiguity_margin=margin,
        ranking=[
            {"name": "Cotton", "score": 0.95, "profile_duration": 870.0},
            {"name": "Cotton Eco", "score": 0.95 - margin, "profile_duration": 880.0},
        ],
    )


async def _cycle_end(hass, freezer, mgr, margin: float) -> dict[str, Any]:
    """Six real cycles; the last one's end re-run with a controlled final match."""
    ps = mgr.profile_store
    for _ in range(6):
        await feed(hass, freezer, 500, 900)
        await feed(hass, freezer, 0, 600)
        await hass.async_block_till_done()
    cycles = ps.get_past_cycles()
    await ps.create_profile_standalone("Cotton", avg_duration=870)
    await ps.create_profile_standalone("Cotton Eco", avg_duration=880)
    for c in cycles[:5]:
        c["profile_name"] = "Cotton"
        c["label_source"] = "manual"
    target = dict(cycles[-1])
    target.pop("id", None)
    target.pop("profile_name", None)
    ps.get_past_cycles().pop()
    mgr._current_program = "Cotton"  # noqa: SLF001
    mgr._last_match_confidence = 0.95  # noqa: SLF001
    mgr._matched_profile_duration = 870.0  # noqa: SLF001
    mgr._last_match_result = _res(margin)  # noqa: SLF001
    ps.async_match_profile = AsyncMock(return_value=_res(margin))
    await mgr._async_process_cycle_end(target)  # noqa: SLF001
    await hass.async_block_till_done()
    return next(c for c in ps.get_past_cycles() if c.get("id") == target.get("id"))


def _finish(calls: list[dict]) -> dict:
    return [c for c in calls if "finished" in str(c.get("message"))][-1]


async def test_a_refused_label_puts_two_answers_on_the_finish_notification(hass, freezer):
    calls = _record(hass)
    entry = _entry(hass)
    mgr = await boot(hass, entry)
    cyc = await _cycle_end(hass, freezer, mgr, margin=0.02)
    assert cyc["id"] in mgr.profile_store.get_pending_feedback()

    prefix = f"REVIEW_WD_{entry.entry_id[:8].upper()}_"
    actions = _finish(calls["mobile"])["data"]["actions"]
    assert actions == [
        {"action": f"{prefix}C_{cyc['id']}", "title": "It was Cotton"},
        {"action": f"{prefix}R1_{cyc['id']}", "title": "It was Cotton Eco"},
    ]
    # Never to a plain notify target (strict-schema platforms reject the key).
    assert "actions" not in (_finish(calls["plain"]).get("data") or {})
    await mgr.async_shutdown()


async def test_no_review_request_means_no_actions(hass, freezer):
    calls = _record(hass)
    mgr = await boot(hass, _entry(hass))
    cyc = await _cycle_end(hass, freezer, mgr, margin=0.3)
    assert cyc["profile_name"] == "Cotton"
    assert cyc["id"] not in mgr.profile_store.get_pending_feedback()
    assert "actions" not in _finish(calls["mobile"])["data"]
    await mgr.async_shutdown()


async def test_a_correction_tap_labels_the_cycle_by_hand(hass, freezer):
    calls = _record(hass)
    entry = _entry(hass)
    mgr = await boot(hass, entry)
    cyc = await _cycle_end(hass, freezer, mgr, margin=0.02)
    correct = _finish(calls["mobile"])["data"]["actions"][1]["action"]

    hass.bus.async_fire(_EVENT, {"action": correct})
    await hass.async_block_till_done()

    assert (cyc["profile_name"], cyc["label_source"]) == ("Cotton Eco", "manual")
    assert cyc["id"] not in mgr.profile_store.get_pending_feedback()
    rec = mgr.profile_store.get_feedback_history()[cyc["id"]]
    assert (rec["user_confirmed"], rec["corrected_profile"]) == (False, "Cotton Eco")

    # A second tap (or the other button) finds no waiting request: nothing changes.
    confirm = _finish(calls["mobile"])["data"]["actions"][0]["action"]
    hass.bus.async_fire(_EVENT, {"action": confirm})
    await hass.async_block_till_done()
    assert cyc["profile_name"] == "Cotton Eco"
    assert mgr.profile_store.get_feedback_history()[cyc["id"]] == rec
    await mgr.async_shutdown()


async def test_a_confirm_tap_confirms_the_detected_programme(hass, freezer):
    calls = _record(hass)
    mgr = await boot(hass, _entry(hass))
    cyc = await _cycle_end(hass, freezer, mgr, margin=0.02)
    confirm = _finish(calls["mobile"])["data"]["actions"][0]["action"]

    hass.bus.async_fire(_EVENT, {"action": confirm})
    await hass.async_block_till_done()

    assert (cyc["profile_name"], cyc["label_source"]) == ("Cotton", "manual")
    assert mgr.profile_store.get_feedback_history()[cyc["id"]]["user_confirmed"] is True
    await mgr.async_shutdown()


async def test_a_request_answered_in_the_panel_ignores_the_tap(hass, freezer):
    calls = _record(hass)
    mgr = await boot(hass, _entry(hass))
    cyc = await _cycle_end(hass, freezer, mgr, margin=0.02)
    correct = _finish(calls["mobile"])["data"]["actions"][1]["action"]
    # The panel's resolve_feedback "confirm".
    assert await mgr.learning_manager.async_submit_cycle_feedback(
        cycle_id=cyc["id"], user_confirmed=True
    )
    rec = dict(mgr.profile_store.get_feedback_history()[cyc["id"]])

    hass.bus.async_fire(_EVENT, {"action": correct})
    await hass.async_block_till_done()

    assert cyc["profile_name"] == "Cotton"
    assert mgr.profile_store.get_feedback_history()[cyc["id"]] == rec
    await mgr.async_shutdown()


async def test_other_devices_and_unrelated_actions_are_ignored(hass, freezer):
    calls = _record(hass)
    mgr = await boot(hass, _entry(hass))
    cyc = await _cycle_end(hass, freezer, mgr, margin=0.02)
    mine = _finish(calls["mobile"])["data"]["actions"][1]["action"]
    other = mine.replace(mgr.entry_id[:8].upper(), "ZZZZZZZZ")
    for action in (other, f"RESUME_WD_{mgr.entry_id[:8].upper()}", None, 7, mine + "x"):
        hass.bus.async_fire(_EVENT, {"action": action})
    await hass.async_block_till_done()
    assert cyc["id"] in mgr.profile_store.get_pending_feedback()
    assert not cyc.get("profile_name")
    await mgr.async_shutdown()


async def test_the_listener_is_removed_on_unload_and_restored_on_setup(hass, freezer):
    calls = _record(hass)
    entry = _entry(hass)
    mgr = await boot(hass, entry)
    cyc = await _cycle_end(hass, freezer, mgr, margin=0.02)
    correct = _finish(calls["mobile"])["data"]["actions"][1]["action"]
    listening = hass.bus.async_listeners().get(_EVENT, 0)
    await mgr.async_shutdown()
    assert hass.bus.async_listeners().get(_EVENT, 0) == listening - 1

    # The next setup (a restart, a reload) still answers the waiting question.
    mgr2 = await boot(hass, entry)
    assert hass.data[DOMAIN][entry.entry_id] is mgr2
    assert mgr2._remove_review_action_listener is not None  # noqa: SLF001
    hass.bus.async_fire(_EVENT, {"action": correct})
    await hass.async_block_till_done()
    stored = next(c for c in mgr2.profile_store.get_past_cycles() if c["id"] == cyc["id"])
    assert (stored["profile_name"], stored["label_source"]) == ("Cotton Eco", "manual")
    await mgr2.async_shutdown()


async def test_a_failing_learning_pass_still_sends_the_finish_notification(hass, freezer):
    calls = _record(hass)
    mgr = await boot(hass, _entry(hass))

    def learning_raises(*_a: Any, **_kw: Any) -> None:
        raise RuntimeError("learning exploded")

    mgr.learning_manager.process_cycle_end = learning_raises
    await feed(hass, freezer, 500, 900)
    await feed(hass, freezer, 0, 600)
    await hass.async_block_till_done()
    assert len([c for c in calls["mobile"] if "finished" in str(c.get("message"))]) == 1
    await mgr.async_shutdown()


async def test_a_rename_keeps_the_ranked_answer_resolvable(hass, freezer):
    calls = _record(hass)
    mgr = await boot(hass, _entry(hass))
    cyc = await _cycle_end(hass, freezer, mgr, margin=0.02)
    correct = _finish(calls["mobile"])["data"]["actions"][1]["action"]
    await mgr.profile_store.update_profile("Cotton Eco", "Eco 40")

    hass.bus.async_fire(_EVENT, {"action": correct})
    await hass.async_block_till_done()
    assert (cyc["profile_name"], cyc["label_source"]) == ("Eco 40", "manual")
    await mgr.async_shutdown()
