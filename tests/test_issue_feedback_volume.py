"""Feedback requests ask only where the answer changes something (register item 433).

Every complete-cycle match between learning_confidence and auto_label_confidence
used to raise a review request, even when the cycle-end gate had already labelled
the cycle with a clear margin: 81% of cycles on the leave-one-out corpus, most of
them correct (91.5%). Now a cycle labelled by that gate is not queued; a refused
gate (small margin / ambiguous) and a profile's warm-up are.
"""

from __future__ import annotations

from .test_group_a_features import _learning_manager, mock_hass_learning  # noqa: F401


def _end(mgr, store, *, labelled: bool, allowed: bool, conf: float = 0.75):
    cyc = {"id": "c1", "duration": 3600.0, "profile_name": "Cotton 60" if labelled else None}
    store.past_cycles.append(cyc)
    mgr._maybe_request_feedback(  # noqa: SLF001
        cyc, detected_profile="Cotton 60", confidence=conf, predicted_duration=3600.0,
        label_allowed=allowed,
    )
    return "c1" in store.pending


def test_a_cycle_the_gate_labelled_is_not_queued(mock_hass_learning):  # noqa: F811
    mgr, store = _learning_manager(mock_hass_learning, labeled_count=6)
    assert _end(mgr, store, labelled=True, allowed=True) is False


def test_a_refused_gate_still_asks(mock_hass_learning):  # noqa: F811
    mgr, store = _learning_manager(mock_hass_learning, labeled_count=6)
    assert _end(mgr, store, labelled=False, allowed=False) is True


def test_warm_up_still_asks_for_a_labelled_cycle(mock_hass_learning):  # noqa: F811
    mgr, store = _learning_manager(mock_hass_learning, labeled_count=1)
    assert _end(mgr, store, labelled=True, allowed=True) is True


def test_low_conformance_alone_does_not_ask(mock_hass_learning):  # noqa: F811
    # Labelled cycles under 0.40 conformance are still 85.6% right leave-one-out:
    # 7 questions per wrong label, and a loose-envelope device re-queued every cycle.
    mgr, store = _learning_manager(mock_hass_learning, labeled_count=6)
    cyc = {"id": "c1", "duration": 3600.0, "profile_name": "Cotton 60", "envelope_conformance": 0.2}
    store.past_cycles.append(cyc)
    mgr._maybe_request_feedback(  # noqa: SLF001
        cyc, detected_profile="Cotton 60", confidence=0.75, predicted_duration=3600.0,
        label_allowed=True,
    )
    assert "c1" not in store.pending


# ── v16 storage migration: drop the leftover requests the rule no longer raises ──

def _rec(cid, detected, scores):
    return {"cycle_id": cid, "detected_profile": detected, "confidence": scores[0],
            "ranking": [{"name": detected, "score": scores[0]}]
            + [{"name": f"other{i}", "score": s} for i, s in enumerate(scores[1:])]}


def _queue_data():
    past = [{"id": f"x{i}", "profile_name": "Cotton", "label_source": "auto_match"} for i in range(3)]
    past += [
        {"id": "clear", "profile_name": "Cotton", "label_source": "auto_match"},
        {"id": "tie", "profile_name": "Cotton", "label_source": "auto_match"},
        {"id": "conflict", "profile_name": "Eco", "label_source": "auto_label_service"},
        {"id": "open", "profile_name": None},
        {"id": "byhand", "profile_name": "Eco", "label_source": "manual"},
        {"id": "warm", "profile_name": "Wool", "label_source": "auto_match"},
    ]
    pending = {
        "clear": _rec("clear", "Cotton", [0.80, 0.60]),
        "tie": _rec("tie", "Cotton", [0.80, 0.77]),
        "conflict": _rec("conflict", "Cotton", [0.80, 0.50]),
        "open": _rec("open", "Cotton", [0.70, 0.66]),
        "byhand": _rec("byhand", "Cotton", [0.80, 0.50]),
        "warm": _rec("warm", "Wool", [0.85, 0.50]),
        "gone": _rec("gone", "Cotton", [0.80, 0.50]),
    }
    return {"past_cycles": past, "pending_feedback": pending, "feedback_history": {}}


def test_v16_cleanup_drops_only_what_the_rule_would_not_ask():
    from custom_components.ha_washdata.profile_store import _dismiss_unneeded_feedback

    data = _queue_data()
    summary = _dismiss_unneeded_feedback(data)
    assert set(data["pending_feedback"]) == {"tie", "conflict", "open", "warm"}
    assert summary == {"dismissed": 1, "answered": 1, "stale": 1, "kept": 4}
    # Not recorded as the user's answer, and no label touched.
    assert data["feedback_history"] == {}
    assert next(c for c in data["past_cycles"] if c["id"] == "clear")["label_source"] == "auto_match"
    # Idempotent.
    assert _dismiss_unneeded_feedback(data)["dismissed"] == 0
    assert set(data["pending_feedback"]) == {"tie", "conflict", "open", "warm"}


async def test_v15_store_migrates_through_the_cleanup():
    from unittest.mock import MagicMock

    from custom_components.ha_washdata.profile_store import WashDataStore

    store = WashDataStore(MagicMock(), 16, "ha_washdata.test")
    result = await store._async_migrate_func(15, 1, _queue_data())  # noqa: SLF001
    assert "clear" not in result["pending_feedback"]
    assert "tie" in result["pending_feedback"]
