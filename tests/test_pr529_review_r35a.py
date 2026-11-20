"""Review of PR #529, round 35a: the Minor findings, each reproduced.

* The joint end-gate check never withdrew a cascade: a pending 10 W Start whose
  20 W pulse-ceiling cascade lets a dryer ignore its 10 W pulses stayed, and the
  strict check reported success.
* A hand-picked program Stage 1's lower bound rejected was still read at its
  profile group's score (a sibling's), so a never-scored pick made a deficit.
* A profile deleted while a #530 check matched moved no generation, so the
  trigger could fire naming the deleted profile.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata.const import (
    CONF_ANTI_WRINKLE_MAX_POWER,
    CONF_DEVICE_TYPE,
    CONF_END_ENERGY_THRESHOLD,
    CONF_MIN_OFF_GAP,
    CONF_OFF_DELAY,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
)
from custom_components.ha_washdata.learning import OWNER_BATCH, LearningManager
from custom_components.ha_washdata.suggestion_engine import SuggestionEngine


class _Store:
    """The suggestion half of ProfileStore, in memory."""

    def __init__(self, cycles: list[dict[str, Any]] | None = None) -> None:
        self.suggestions: dict[str, Any] = {}
        self.cycles = list(cycles or [])

    def get_suggestions(self) -> dict[str, Any]:
        return self.suggestions

    def set_suggestion(self, key, value, reason=None, reason_key=None, reason_params=None, flags=None):
        from custom_components.ha_washdata.profile_store import SUGGESTION_FLAG_KEYS  # noqa: PLC0415

        entry: dict[str, Any] = {"value": value, "reason": reason}
        for flag in SUGGESTION_FLAG_KEYS:
            if flags and flags.get(flag):
                entry[flag] = flags[flag]
        self.suggestions[key] = entry

    def delete_suggestion(self, key) -> None:
        self.suggestions.pop(key, None)

    def get_locked_suggestions(self) -> list[str]:
        return []

    def get_past_cycles(self) -> list[dict[str, Any]]:
        return self.cycles

    def get_lifetime_cycle_count(self) -> int:
        return 100

    def get_suggestion_apply_cycle_count(self) -> int:
        return 0

    async def async_save(self) -> None:
        return None


# ── A cascade goes with the entry it follows ─────────────────────────────────


def _pulsed_dryer_cycle() -> dict[str, Any]:
    """300 W, an 800 s 3.8 W phase with a single 10 W reading every 90 s, 300 W."""
    points = [(float(t), 300.0) for t in range(0, 1200, 10)]
    points += [
        (float(t), 10.0 if t > 1200 and (t - 1200) % 90 == 0 else 3.8)
        for t in range(1200, 2000, 10)
    ]
    points += [(float(t), 300.0) for t in range(2000, 3200, 10)]
    points += [(3200.0 + t, 0.5) for t in range(0, 600, 10)]
    return {
        "id": "d1", "status": "completed", "termination_reason": "timeout",
        "power_data": [[t, p] for t, p in points],
    }


# In force: an 8 W pulse ceiling keeps the 10 W pulses resetting the quiet, so
# the pause stays whole at a 360 s Min Off Gap.
_DRYER_OPTS = {
    CONF_DEVICE_TYPE: "dryer", CONF_STOP_THRESHOLD_W: 4.0, CONF_START_THRESHOLD_W: 6.0,
    CONF_OFF_DELAY: 180, CONF_MIN_OFF_GAP: 360, CONF_END_ENERGY_THRESHOLD: 0.05,
    CONF_ANTI_WRINKLE_MAX_POWER: 8.0,
}


def _dryer_engine(store: _Store) -> SuggestionEngine:
    eng = SuggestionEngine(MagicMock(), "e", store, device_type="dryer")
    eng._entry_options = lambda: dict(_DRYER_OPTS)  # noqa: SLF001
    return eng


def _snapshot(store: _Store) -> dict[str, Any]:
    return {k: dict(v) for k, v in store.suggestions.items()}


def test_a_start_whose_pulse_ceiling_cascade_splits_is_withdrawn() -> None:
    """A pending 10 W Start makes reconcile store a 20 W pulse ceiling (2 x start):
    the 10 W pulses are then ignored and the pause splits. The cascade was never a
    candidate, so the check returned [] (strict too) and the set stayed."""
    store = _Store(cycles=[_pulsed_dryer_cycle()])
    eng = _dryer_engine(store)
    store.set_suggestion(CONF_START_THRESHOLD_W, 10.0, reason="r", flags={"owner": OWNER_BATCH})
    eng._reconcile_stored_suggestions()  # noqa: SLF001
    assert store.suggestions[CONF_ANTI_WRINKLE_MAX_POWER]["value"] == 20.0
    assert store.suggestions[CONF_ANTI_WRINKLE_MAX_POWER]["cascade"] is True
    assert eng.joint_end_gate_drops(_snapshot(store), strict=True) == [CONF_START_THRESHOLD_W]
    assert eng.joint_end_gate_drops(_snapshot(store)) == [CONF_START_THRESHOLD_W]


def test_a_source_whose_cascade_adds_no_split_stays_and_strict_fails() -> None:
    """A 10 W Start derives a 20 W pulse ceiling that splits nothing on a pause with
    no pulses; a stale 4 W Stop cascade (no stored entry derives it) does. Withdrawing
    the Start would remove no split: it stays, and the strict check fails rather than
    reporting the set safe."""
    points = [(float(t), 300.0) for t in range(0, 1200, 10)]
    points += [(float(t), 2.0) for t in range(1200, 1700, 10)]
    points += [(float(t), 300.0) for t in range(1700, 2900, 10)]
    points += [(2900.0 + t, 0.5) for t in range(0, 600, 10)]
    cycle = {
        "id": "d2", "status": "completed", "termination_reason": "timeout",
        "power_data": [[t, p] for t, p in points],
    }
    store = _Store(cycles=[cycle])
    eng = SuggestionEngine(MagicMock(), "e", store, device_type="dryer")
    opts = {**_DRYER_OPTS, CONF_STOP_THRESHOLD_W: 1.5, CONF_END_ENERGY_THRESHOLD: 0.2}
    eng._entry_options = lambda: dict(opts)  # noqa: SLF001
    store.set_suggestion(CONF_START_THRESHOLD_W, 10.0, reason="r", flags={"owner": OWNER_BATCH})
    eng._reconcile_stored_suggestions()  # noqa: SLF001
    assert store.suggestions[CONF_ANTI_WRINKLE_MAX_POWER]["cascade"] is True
    store.set_suggestion(CONF_STOP_THRESHOLD_W, 4.0, reason="r", flags={"cascade": True})
    assert eng.joint_end_gate_drops(_snapshot(store)) == []
    with pytest.raises(RuntimeError):
        eng.joint_end_gate_drops(_snapshot(store), strict=True)


def _learning(hass: Any, store: _Store) -> LearningManager:
    entry = MagicMock()
    entry.data = {}
    entry.options = dict(_DRYER_OPTS)
    hass.config_entries.async_get_entry.return_value = entry
    hass.async_create_task = lambda coro, *_a: coro.close()
    return LearningManager(hass, "e", store, device_type="dryer")


async def test_the_manual_check_withdraws_the_start_and_its_cascade(mock_hass: Any) -> None:
    """The strict pass withdraws the Start; the reconcile after it drops the cascade."""
    store = _Store(cycles=[_pulsed_dryer_cycle()])
    lm = _learning(mock_hass, store)
    lm.suggestion_engine.apply_suggestions(
        {CONF_START_THRESHOLD_W: {"value": 10.0, "reason": "r"}}, owner=OWNER_BATCH
    )
    assert CONF_ANTI_WRINKLE_MAX_POWER in store.suggestions
    await lm._enforce_joint_end_gate(strict=True)  # noqa: SLF001
    assert CONF_START_THRESHOLD_W not in store.suggestions
    assert CONF_ANTI_WRINKLE_MAX_POWER not in store.suggestions


def test_a_different_start_stored_during_the_check_is_not_withdrawn(mock_hass: Any) -> None:
    """Start 10.01 W stored while 10 W was judged: its cascade rounds to the same
    20 W, so the end-gate values did not move. It is not the Start judged, so it
    stays and the set is judged again."""
    store = _Store(cycles=[_pulsed_dryer_cycle()])
    lm = _learning(mock_hass, store)
    lm.suggestion_engine.apply_suggestions(
        {CONF_START_THRESHOLD_W: {"value": 10.0, "reason": "r"}}, owner=OWNER_BATCH
    )
    judged = _snapshot(store)
    lm.suggestion_engine.apply_suggestions(
        {CONF_START_THRESHOLD_W: {"value": 10.01, "reason": "r"}}, owner=OWNER_BATCH
    )
    assert store.suggestions[CONF_ANTI_WRINKLE_MAX_POWER]["value"] == 20.0
    assert lm._withdraw_joint_drops(judged, [CONF_START_THRESHOLD_W]) is False  # noqa: SLF001
    assert store.suggestions[CONF_START_THRESHOLD_W]["value"] == 10.01


# ── A pick Stage 1's lower bound rejected is no evidence, family or not ──────


def _family_match(rejected: tuple[str, ...]) -> Any:
    from types import SimpleNamespace  # noqa: PLC0415

    return SimpleNamespace(
        best_profile="Eco",
        candidates=[
            {"name": "Eco", "score": 0.9},
            {"name": "__group__Cotton", "group_best_member": "Cotton 40", "score": 0.6},
            {"name": "Quick", "score": 0.5},
            {"name": "Wool", "score": 0.45},
            {"name": "Synthetics", "score": 0.4},
        ],
        min_ratio_rejected=rejected,
    )


def test_a_rejected_pick_is_not_read_at_its_sibling_s_family_score() -> None:
    """Cotton 90 was never scored (the cycle is still under the min ratio of it);
    the Cotton family's row is Cotton 40's score and bounds nothing of it, so it
    must not make a 0.3 deficit. Scored, the family bound stands."""
    from custom_components.ha_washdata.match_rules import selected_program_deficit  # noqa: PLC0415

    families = {"Cotton 40": "Cotton", "Cotton 90": "Cotton"}
    assert selected_program_deficit(_family_match(("Cotton 90",)), "Cotton 90", families) is None
    hit = selected_program_deficit(_family_match(()), "Cotton 90", families)
    assert hit is not None and hit[0] == "Eco" and hit[1] == pytest.approx(0.3)


# ── A profile deleted while a #530 check matches discards that match ─────────


async def test_the_winner_deleted_during_the_check_discards_it(
    hass: Any, setup_washdata_entry: Any, freezer: Any
) -> None:
    """Synthetics is deleted while the check's match runs; the ranking names it as
    what the cycle looks like. The data generation did not move, so the result
    passed every guard and the trigger fired for a profile that no longer exists.
    The narrower profile set revision discards it; the data generation stays (it
    would discard unrelated envelope builds, suggestion passes and Apply all)."""
    from unittest.mock import AsyncMock, patch  # noqa: PLC0415

    from custom_components.ha_washdata.const import PROGRAM_MISMATCH_PERSISTENCE  # noqa: PLC0415

    from .test_issue_530_program_mismatch import (  # noqa: PLC0415
        _check,
        _mismatches,
        _past_half,
        _result,
    )

    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    store = manager.profile_store
    generation = store._data_generation  # noqa: SLF001
    wrong = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})

    async def match_and_delete(*_a: Any, **_k: Any) -> Any:
        await store.delete_profile("Synthetics")
        return wrong

    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    with patch.object(store, "async_match_profile", new=match_and_delete):
        await _check(hass, manager, state.key)
    assert _mismatches(sent) == []
    assert state.streak == 0 and not state.fired
    assert store._data_generation == generation  # noqa: SLF001
    # A check on the profile set as it now stands still counts.
    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    other = _result("Wool", {"Wool": 0.9, "Cotton": 0.5})
    with patch.object(store, "async_match_profile", AsyncMock(return_value=other)):
        await _check(hass, manager, state.key)
    assert len(_mismatches(sent)) == 1
