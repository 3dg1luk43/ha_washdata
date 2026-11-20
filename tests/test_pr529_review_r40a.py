"""Review of PR #529, round 40a, each reproduced.

* #530: the check's fraction gate read the stall shown now as programme time.
  ``_match_readings()`` keeps a shown stall in the trace and ``paused_s`` covered
  only the user pause, so a right pick stalled early reached half its duration
  on the stall's clock. The stall shown now leaves the elapsed time as well
  (``CycleDetector.open_halt_s``); finished stalls are cut from the trace already.
* An envelope build was discarded on any match setting changed during its job,
  a Stage-1 ratio included, which no build reads and after which nothing
  rebuilds: a cycle deleted meanwhile stayed in the envelope. Only a setting the
  build reads (``envelope_settings_revision``) discards it now.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata import cycle_detector as cd_mod
from custom_components.ha_washdata.const import DOMAIN, PROGRAM_MISMATCH_PERSISTENCE

from .test_audit_detection_514_halt_followups import _det, _feed, _halt, _match, _wash
from .test_issue_530_program_mismatch import _check, _mismatches, _past_half, _result, _seed

# ── #530: the stall shown now is no programme time ───────────────────────────


def _span(det: Any) -> float:
    live = det._match_readings()  # noqa: SLF001
    return (live[-1][0] - live[0][0]).total_seconds()


def test_the_stall_shown_now_leaves_the_checks_elapsed_time() -> None:
    """30 min of a 90 min programme, then a halt shown as stalled: the trace the
    check matches still spans the halt, and its span less ``open_halt_s`` is the
    30 min of programme time. Banked on the resume, it is cut from the trace and
    not taken off again."""
    det = _det()
    _feed(det, 0, 1800, _wash)
    _match(det, "Cotton", 5400.0)
    _feed(det, 1800, 4200, _halt)
    assert det.stalled
    assert _span(det) == pytest.approx(4170.0)
    assert _span(det) - det.open_halt_s() == pytest.approx(1800.0, abs=30.0)
    with patch.object(cd_mod, "STALL_CURRENT_EXCLUDED_FROM_PROGRESS", False):
        assert det.open_halt_s() == 0.0
    _feed(det, 4200, 4800, _wash)
    assert not det.stalled
    assert det.open_halt_s() == 0.0
    assert _span(det) == pytest.approx(4770.0 - 2400.0, abs=30.0)


def test_a_user_pause_inside_the_stall_is_not_taken_off_twice() -> None:
    """The stall shown and a user pause opened in it: their union, once."""
    det = _det()
    _feed(det, 0, 1800, _wash)
    _match(det, "Cotton", 5400.0)
    _feed(det, 1800, 3000, _halt)
    assert det.stalled
    det.set_user_paused(True, cd_mod.dt_util.as_utc(det._power_readings[-1][0]))  # noqa: SLF001
    _feed(det, 3000, 3600, _halt)
    assert det.open_user_pause_s() > 0
    assert _span(det) - det.open_halt_s() == pytest.approx(1800.0, abs=30.0)


async def test_a_pick_stalled_early_is_not_past_half(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    """Cotton (30 min) picked, the trace ~17 min long but stalled since minute 5:
    five hits in a row and a sixth do not establish a mismatch, since 5 min of
    programme time is not half of 30. Without the stall the sixth does."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    det = manager.detector
    wrong = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})
    det._stall_active = True  # noqa: SLF001  (shown as stalled since minute 5)
    det._stall_run_start = det._current_cycle_start + timedelta(minutes=5)  # noqa: SLF001
    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1
    with patch.object(manager.profile_store, "async_match_profile", AsyncMock(return_value=wrong)):
        await _check(hass, manager, state.key)
    assert _mismatches(sent) == []
    assert state.streak == PROGRAM_MISMATCH_PERSISTENCE and not state.fired
    det._stall_active = False  # noqa: SLF001
    det._stall_run_start = None  # noqa: SLF001
    with patch.object(manager.profile_store, "async_match_profile", AsyncMock(return_value=wrong)):
        await _check(hass, manager, state.key)
    assert len(_mismatches(sent)) == 1


# ── An envelope build is not discarded on a Stage-1 ratio change ─────────────


def _changing_during_the_job(store: Any, change: Any) -> Any:
    real = store._rebuild_envelope_sync  # noqa: SLF001

    def worker(cycles: Any) -> Any:
        change()
        return real(cycles)

    return worker


async def _seeded_store(hass: HomeAssistant, setup_washdata_entry: Any) -> Any:
    entry = await setup_washdata_entry()
    manager = hass.data[DOMAIN][entry.entry_id]
    await _seed(manager)
    store = manager.profile_store
    assert store.get_envelope("Cotton")["cycle_count"] == 3
    return store


async def test_a_cycle_deleted_while_a_ratio_changes_leaves_the_envelope(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """One of Cotton's three cycles deleted while the min ratio changes under the
    rebuild: the deletion is saved and the envelope is the two-cycle one."""
    store = await _seeded_store(hass, setup_washdata_entry)
    revision = store.match_settings_revision
    ratio = _changing_during_the_job(
        store, lambda: store.set_duration_ratio_limits(min_ratio=0.2, max_ratio=1.8)
    )
    with patch.object(store, "_rebuild_envelope_sync", side_effect=ratio):
        assert await store.delete_cycle("Cotton-2")
    assert store.match_settings_revision > revision
    env = store.get_envelope("Cotton")
    assert env["cycle_count"] == 2 and env["shape_cycle_count"] == 2


@pytest.mark.parametrize(
    "change",
    [
        lambda s: setattr(s, "dtw_bandwidth", s.dtw_bandwidth + 0.05),
        lambda s: setattr(s, "energy_mode", "integrated" if s.energy_mode != "integrated" else "mean"),
        lambda s: setattr(s, "device_type", "dryer"),
        lambda s: setattr(s, "evidence_sources", ["real_cycles", "reference_cycles"]),
    ],
)
async def test_a_setting_the_build_reads_still_discards_it(
    hass: HomeAssistant, setup_washdata_entry: Any, change: Any
) -> None:
    """The DTW band, the energy mode, the device type and the evidence sources
    are read by the build: changed under it, the build is discarded as before."""
    store = await _seeded_store(hass, setup_washdata_entry)
    assert len(store.evidence_sources) == 3
    worker = _changing_during_the_job(store, lambda: change(store))
    with patch.object(store, "_rebuild_envelope_sync", side_effect=worker):
        assert await store.delete_cycle("Cotton-2")
    assert store.get_envelope("Cotton")["cycle_count"] == 3
