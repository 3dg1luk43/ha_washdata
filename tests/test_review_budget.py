"""Review requests stop once a programme's budget is spent (2026-11-03 audit item 1).

A cycle the label gate refused used to be queued every time. Replayed in
chronological order over 13 devices, a programme's first answers and every
correction carry the accuracy; asking only while fewer than
``REVIEW_BUDGET_PER_PROFILE`` requests about it are waiting or were answered
without a correction took 60 questions instead of 105 for the same top-1.
"""

from __future__ import annotations

from custom_components.ha_washdata.const import REVIEW_BUDGET_PER_PROFILE

from .test_group_a_features import _learning_manager, mock_hass_learning  # noqa: F401

_P = "Cotton 60"


def _refused(mgr, store, cid="c1", profile=_P):
    cyc = {"id": cid, "duration": 3600.0, "profile_name": None}
    store.past_cycles.append(cyc)
    mgr._maybe_request_feedback(  # noqa: SLF001
        cyc, detected_profile=profile, confidence=0.75, predicted_duration=3600.0,
        label_allowed=False,
    )
    return cid in store.pending


def _answer(store, cid, *, confirmed=False, corrected=None, detected=_P):
    store.feedback[cid] = {
        "cycle_id": cid, "original_detected_profile": detected,
        "user_confirmed": confirmed, "corrected_profile": corrected,
    }


def test_budget_is_two():
    # The constant the measurement chose (1: 45 questions for +23, 3: 75 for +32).
    assert REVIEW_BUDGET_PER_PROFILE == 2


def test_asks_until_the_budget_is_spent(mock_hass_learning):  # noqa: F811
    mgr, store = _learning_manager(mock_hass_learning, labeled_count=6)
    assert _refused(mgr, store, "c1") is True
    assert _refused(mgr, store, "c2") is True
    # Two waiting requests about the programme: the third refused cycle is not asked.
    assert _refused(mgr, store, "c3") is False
    # Another programme has its own budget.
    assert _refused(mgr, store, "c4", profile="Eco") is True


def test_confirmed_answers_spend_the_budget(mock_hass_learning):  # noqa: F811
    mgr, store = _learning_manager(mock_hass_learning, labeled_count=6)
    _answer(store, "a1", confirmed=True)
    assert _refused(mgr, store, "c1") is True
    _answer(store, "a2", confirmed=True)
    del store.pending["c1"]
    assert _refused(mgr, store, "c2") is False


def test_dismissal_and_relabel_to_the_same_programme_spend_it(mock_hass_learning):  # noqa: F811
    mgr, store = _learning_manager(mock_hass_learning, labeled_count=6)
    _answer(store, "a1")                    # Ignore: no answer named
    _answer(store, "a2", corrected=_P)      # relabelled to what WashData guessed
    assert _refused(mgr, store) is False


def test_corrections_never_spend_it(mock_hass_learning):  # noqa: F811
    # The matcher keeps calling other programmes' cycles this one: keep asking.
    mgr, store = _learning_manager(mock_hass_learning, labeled_count=6)
    for i in range(5):
        _answer(store, f"a{i}", corrected="Eco")
    assert _refused(mgr, store) is True


def test_answers_about_other_programmes_do_not_count(mock_hass_learning):  # noqa: F811
    mgr, store = _learning_manager(mock_hass_learning, labeled_count=6)
    for i in range(3):
        _answer(store, f"a{i}", confirmed=True, detected="Eco")
    store.pending["w1"] = {"cycle_id": "w1", "detected_profile": "Eco"}
    assert _refused(mgr, store) is True


def test_warm_up_is_not_budgeted(mock_hass_learning):  # noqa: F811
    mgr, store = _learning_manager(mock_hass_learning, labeled_count=1)
    _answer(store, "a1", confirmed=True)
    _answer(store, "a2", confirmed=True)
    cyc = {"id": "c1", "duration": 3600.0, "profile_name": _P}
    store.past_cycles.append(cyc)
    mgr._maybe_request_feedback(  # noqa: SLF001
        cyc, detected_profile=_P, confidence=0.75, predicted_duration=3600.0,
        label_allowed=True,
    )
    assert "c1" in store.pending


def test_a_broken_queue_record_never_blocks_the_cycle_end(mock_hass_learning):  # noqa: F811
    mgr, store = _learning_manager(mock_hass_learning, labeled_count=6)
    store.pending["junk"] = "not a dict"
    store.feedback["junk"] = None
    assert _refused(mgr, store) is True
