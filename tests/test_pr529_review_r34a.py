"""PR #529 review, round 34a.

* #530: a group edit during a program check discarded the check but kept the
  streak; undone before the next check, nothing saw the change and the run went
  on over a match that could not be read.
* #530: a scoring setting changed during a check's trace preparation and changed
  back before it landed compared equal by value, though the ranking was scored
  under the intermediate setting.
* #530: a pick Stage 1's lower bound rejected (the cycle still too short a part
  of it) was read against the fifth score of a full list, a bound it never had.
* #524: a program changed during a PAUSED door dwell, the dwell ran out before
  the detector took it, and nothing armed another once it did.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata import analysis, match_rules
from custom_components.ha_washdata.const import (
    DOMAIN,
    PROGRAM_MISMATCH_PERSISTENCE,
    STATE_PAUSED,
)
from custom_components.ha_washdata.time_utils import utc_now
from tests.test_issue_524_door_ends_cycle import DOOR, OPTS, WASH_S, _paused_washer, _tick
from tests.test_issue_530_program_mismatch import (
    COTTON,
    _check,
    _mismatches,
    _past_half,
    _result,
    _seed,
    _trace,
)

# ── #530: a group edit undone before the next check ─────────────────────────


async def test_a_group_edit_undone_before_the_next_check_restarts_the_run(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    """Grouped during the match and ungrouped again before the next check: the
    map the streak was counted under is back, so only the discarding guard can
    restart the run. One deficit after it must not complete it."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    store = manager.profile_store
    wrong = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})

    async def match_and_group(*_a: Any, **_k: Any) -> Any:
        store._data["profile_groups"] = {"Wash": {"members": ["Cotton", "Synthetics"]}}  # noqa: SLF001
        return wrong

    store._data["profile_groups"] = {}  # noqa: SLF001
    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    with patch.object(store, "async_match_profile", new=match_and_group):
        await _check(hass, manager, state.key)
    assert _mismatches(sent) == []
    assert state.streak == 0
    store._data["profile_groups"] = {}  # noqa: SLF001  (the edit undone)
    with patch.object(store, "async_match_profile", AsyncMock(return_value=wrong)):
        await _check(hass, manager, state.key)
        assert _mismatches(sent) == [] and state.streak == 1
        # A whole run under the unchanged groups still establishes it, once.
        for _ in range(PROGRAM_MISMATCH_PERSISTENCE):
            await _check(hass, manager, state.key)
    assert len(_mismatches(sent)) == 1


# ── #530: a scoring setting changed and changed back during the match ───────


async def test_a_setting_changed_and_back_during_the_match_discards_it(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    """The matcher reads its settings after the trace preparation: a reload there
    and another restoring the value before the match lands scored the ranking
    under the intermediate band, which the value comparison cannot see."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    store = manager.profile_store
    wrong = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})

    async def match_flip_and_back(*_a: Any, **_k: Any) -> Any:
        band = store.dtw_bandwidth
        store.dtw_bandwidth = band + 0.1  # a reload during the preparation...
        store.dtw_bandwidth = band  # ...and one restoring it before the match lands
        return wrong

    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    with patch.object(store, "async_match_profile", new=match_flip_and_back):
        await _check(hass, manager, state.key)
    assert _mismatches(sent) == []
    assert not state.fired and state.streak == 0
    # Unchanged settings still count.
    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    with patch.object(store, "async_match_profile", AsyncMock(return_value=wrong)):
        await _check(hass, manager, state.key)
    assert len(_mismatches(sent)) == 1


async def test_the_settings_revision_moves_on_every_change_only(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """Every ``match_settings`` value set through the store moves the revision
    when it changes; an options reload that keeps them does not."""
    entry = await setup_washdata_entry()
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    rev = store.match_settings_revision
    lo, hi = store.get_duration_ratio_limits()
    store.set_duration_ratio_limits(min_ratio=lo, max_ratio=hi)
    store.dtw_bandwidth = store.dtw_bandwidth
    store.energy_mode = store.energy_mode
    store.device_type = store.device_type
    store.evidence_sources = list(store.evidence_sources)
    assert store.match_settings_revision == rev
    await manager.async_reload_config(entry)
    assert store.match_settings_revision == rev
    changes = (
        lambda: store.set_duration_ratio_limits(min_ratio=lo + 0.05, max_ratio=hi),
        lambda: setattr(store, "dtw_bandwidth", store.dtw_bandwidth + 0.05),
        lambda: setattr(store, "energy_mode", "integrated" if store.energy_mode == "mean" else "mean"),
        lambda: setattr(store, "device_type", "dryer"),
        lambda: setattr(store, "evidence_sources", ["real_cycles"]),
    )
    for change in changes:
        before = store.match_settings()
        change()
        assert store.match_settings() != before
        assert store.match_settings_revision == rev + 1
        rev = store.match_settings_revision


# ── #530: a pick Stage 1's lower bound rejected is no evidence ─────────────


def test_a_min_ratio_rejected_pick_missing_from_a_full_list_is_no_evidence() -> None:
    """Five listed, the pick missing: ranked below them it is bounded by the
    fifth score; rejected by Stage 1's lower bound (the cycle still too short a
    part of it) it was never scored. Past the upper bound it still reads at the
    fifth score (measured: the overrun is what catches 316 wrong picks)."""
    res = _result("Synthetics", {"Synthetics": 0.9, "A": 0.85, "B": 0.8, "C": 0.7, "D": 0.65})
    hit = match_rules.selected_program_deficit(res, "Cotton")
    assert hit is not None and abs(hit[1] - 0.25) < 1e-9
    res.min_ratio_rejected = ("Cotton",)
    assert match_rules.selected_program_deficit(res, "Cotton") is None
    # Another programme's rejection leaves the bound to the pick.
    res.min_ratio_rejected = ("Eco",)
    assert match_rules.selected_program_deficit(res, "Cotton") == hit


async def _add_programme(store: Any, name: str, shape: list[tuple[float, float]]) -> None:
    pts = _trace(shape)
    start = dt_util.utcnow() - timedelta(days=30)
    for j in range(3):
        store._data["past_cycles"].append({  # noqa: SLF001
            "id": f"{name}-{j}", "profile_name": name, "label_source": "manual",
            "status": "completed", "start_time": (start + timedelta(days=j)).isoformat(),
            "end_time": (start + timedelta(days=j, seconds=pts[-1][0])).isoformat(),
            "duration": pts[-1][0], "power_data": [list(p) for p in pts],
        })
    store._data["profiles"][name] = {  # noqa: SLF001
        "avg_duration": pts[-1][0], "sample_cycle_id": f"{name}-0",
    }
    await store.async_rebuild_envelope(name)


async def test_the_match_reports_the_programmes_the_min_ratio_rejected(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """``min_ratio_rejected`` is Stage 1's own lower-bound verdict on the
    snapshots the match scored. 25 min into a run at a 0.5 min ratio: the 60 min
    programme (0.42) is listed there; the 10 min one (2.5x, past the 1.8 upper
    bound) is rejected too, but not there; the 30 min ones are scored."""
    entry = await setup_washdata_entry()
    manager = hass.data[DOMAIN][entry.entry_id]
    await _seed(manager)
    store = manager.profile_store
    await _add_programme(store, "Quick", [(10, 2000.0)])
    await _add_programme(store, "Long", [(20, 2000.0), (40, 200.0)])
    _lo, hi = store.get_duration_ratio_limits()
    store.set_duration_ratio_limits(min_ratio=0.5, max_ratio=hi)
    t0 = dt_util.utcnow()
    readings = [(t0 + timedelta(seconds=t), w) for t, w in _trace(COTTON)[:51]]
    elapsed = (readings[-1][0] - readings[0][0]).total_seconds()
    assert elapsed == 1500.0
    res = await store.async_match_profile(readings, elapsed, in_progress=True)
    listed = {c.get("name") for c in res.candidates}
    assert listed and not listed & {"Quick", "Long"}
    assert res.min_ratio_rejected == ("Long",)
    # The shared rule, as the worker applies it.
    bounds = analysis.stage1_duration_bounds(
        1500.0, {"in_progress": True, "min_duration_ratio": 0.5}
    )
    assert analysis.stage1_duration_rejects(1500.0, 600.0, bounds)
    assert analysis.stage1_duration_rejects(1500.0, 3600.0, bounds)
    assert analysis.stage1_min_ratio_rejects(1500.0, 3600.0, bounds)
    assert not analysis.stage1_min_ratio_rejects(1500.0, 600.0, bounds)
    assert not analysis.stage1_duration_rejects(1500.0, 1800.0, bounds)


# ── #524: a program changed during a PAUSED door dwell ──────────────────────


async def test_a_program_changed_during_the_door_dwell_still_ends_the_cycle(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    """The dwell runs out before the detector takes the new pick; once it does,
    still PAUSED and with the door still open, the door ends the cycle. The
    detector is held in PAUSED (no reading reaches it): this fixture's pause
    turns into ENDING within the minute, a transition that re-arms the dwell on
    its own and would hide the gap."""
    manager, ends = await _paused_washer(hass, setup_washdata_entry, freezer, OPTS)
    hass.states.async_set(DOOR, "on")
    await hass.async_block_till_done()
    assert manager._remove_door_end_dwell is not None  # noqa: SLF001
    # Picked by hand during the dwell: the detector holds Cotton until its next tick.
    manager._manual_program_active = True  # noqa: SLF001
    manager._current_program = "Quick"  # noqa: SLF001
    manager._matched_profile_duration = float(WASH_S)  # noqa: SLF001
    with patch.object(manager.detector, "process_reading", lambda *_a, **_k: None):
        await _tick(hass, freezer, 70, None)  # past the dwell
        assert ends == []
        assert manager.detector.state == STATE_PAUSED
        assert manager.detector.matched_profile == "Cotton"
        # The detector's match tick takes the pick, the door still open.
        manager.detector._try_profile_match(utc_now(), force=True)  # noqa: SLF001
        assert manager.detector.matched_profile == "Quick"
        assert manager.detector.state == STATE_PAUSED
        await _tick(hass, freezer, 70, None)
    assert len(ends) == 1, ends


async def test_a_user_pause_during_the_wait_keeps_the_cycle(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    """The wait for the detector re-arms nothing under a user pause."""
    manager, ends = await _paused_washer(hass, setup_washdata_entry, freezer, OPTS)
    hass.states.async_set(DOOR, "on")
    await hass.async_block_till_done()
    manager._manual_program_active = True  # noqa: SLF001
    manager._current_program = "Quick"  # noqa: SLF001
    manager._is_user_paused = True  # noqa: SLF001
    manager._door_end_dwell_fired(utc_now())  # noqa: SLF001
    assert manager._remove_door_end_dwell is None  # noqa: SLF001
    assert ends == []
