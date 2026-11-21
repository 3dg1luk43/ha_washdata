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
* A door end dwell that expired while a Pause waited on its power cut declined
  in the pause; when the switch call then failed, the rollback undid the pause
  but left no timer, so the open door never ended the cycle. The rollback now
  judges the open door again (a fresh dwell); a successful pause still cancels.
* The startup suggestion refresh ran every pass under one try: an operational
  scan that raised ended it before the #527 Start Threshold correction, and the
  refresh does not run again on an unchanged store, so a device that cannot
  start a cycle never got it. Each pass runs on its own now; the manual
  analysis still stops at a failed pass and reports it.
* On a template cache miss (a new grid step) ``build_match_snapshots()``
  resampled every uncached sample and member template on the loop: 15-22 ms on
  the largest corpus store here, 12-13 ms of it that resampling. A live match
  now resamples them in one executor job over copied traces and builds again on
  the warm cache; the snapshots are the inline build's.
"""
from __future__ import annotations

import asyncio
import threading
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_fire_time_changed

from custom_components.ha_washdata import cycle_detector as cd_mod
from custom_components.ha_washdata import profile_store as ps_mod
from custom_components.ha_washdata.const import (
    CONF_OFF_DELAY,
    DOMAIN,
    PROGRAM_MISMATCH_PERSISTENCE,
    TerminationReason,
)
from custom_components.ha_washdata.suggestion_engine import SuggestionEngine

from .test_audit_detection_514_halt_followups import _det, _feed, _halt, _match, _wash
from .test_issue_524_door_ends_cycle import DOOR, OPTS, _paused_washer, _tick
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
        lambda s: setattr(s, "device_type", "dryer" if s.device_type != "dryer" else "washing_machine"),
        lambda s: setattr(
            s, "evidence_sources",
            ["real_cycles", "reference_cycles"] if len(s.evidence_sources) == 3
            else ["real_cycles", "reference_cycles", "backfill_cycles"],
        ),
    ],
)
async def test_a_setting_the_build_reads_still_discards_it(
    hass: HomeAssistant, setup_washdata_entry: Any, change: Any
) -> None:
    """The DTW band, the energy mode, the device type and the evidence sources
    are read by the build: changed under every build, each is discarded (one
    changed once is built again, round 41b)."""
    store = await _seeded_store(hass, setup_washdata_entry)
    assert len(store.evidence_sources) == 3
    worker = _changing_during_the_job(store, lambda: change(store))
    with patch.object(store, "_rebuild_envelope_sync", side_effect=worker):
        assert await store.delete_cycle("Cotton-2")
    assert store.get_envelope("Cotton")["cycle_count"] == 3


# ── A failed power-cut pause judges an open door again ───────────────────────

SWITCH = "switch.washer_plug"
CUT_OPTS = {**OPTS, "pause_cuts_power": True, "switch_entity": SWITCH}


async def _door_open_then_pause(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any, *, fails: bool
) -> tuple[Any, list[dict[str, Any]], bool]:
    """A washer PAUSED at its programme end, the door open (a 60 s dwell armed),
    Pause pressed: the switch call takes 70 s, the dwell expires in it."""
    manager, ends = await _paused_washer(hass, setup_washdata_entry, freezer, CUT_OPTS)
    hass.states.async_set(DOOR, "on")
    await hass.async_block_till_done()
    assert manager._remove_door_end_dwell is not None  # noqa: SLF001

    async def turn_off(_call: ServiceCall) -> None:
        freezer.tick(timedelta(seconds=70))
        async_fire_time_changed(hass, dt_util.utcnow())
        await asyncio.sleep(0)
        # Expired inside the pause: declined, and no timer left.
        assert manager._remove_door_end_dwell is None  # noqa: SLF001
        if fails:
            raise HomeAssistantError("plug offline")

    hass.services.async_register("switch", "turn_off", turn_off)
    paused = await manager.async_pause_cycle()
    await hass.async_block_till_done()
    assert ends == []
    return manager, ends, paused


async def test_a_failed_pause_judges_the_open_door_again(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    manager, ends, paused = await _door_open_then_pause(
        hass, setup_washdata_entry, freezer, fails=True
    )
    assert paused is False and manager._is_user_paused is False  # noqa: SLF001
    assert manager._remove_door_end_dwell is not None  # noqa: SLF001  (a fresh dwell)
    await _tick(hass, freezer, 30, None)
    assert ends == []  # inside the fresh dwell
    await _tick(hass, freezer, 40, None)
    assert len(ends) == 1 and ends[0]["termination_reason"] == TerminationReason.DOOR


async def test_a_successful_pause_still_cancels_the_dwell(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    manager, ends, paused = await _door_open_then_pause(
        hass, setup_washdata_entry, freezer, fails=False
    )
    assert paused is True and manager._is_user_paused is True  # noqa: SLF001
    assert manager._remove_door_end_dwell is None  # noqa: SLF001
    await _tick(hass, freezer, 120, None)
    assert ends == []


# ── The startup refresh: a failed pass does not end it ───────────────────────

PASSES = [
    "generate_operational_suggestions",
    "generate_standby_floor_suggestions",
    "generate_start_threshold_suggestions",
    "generate_model_suggestions",
    "generate_detection_suggestions",
    "run_batch_simulation",
]


@pytest.mark.parametrize("failing", PASSES[:-1])
async def test_a_failed_pass_does_not_end_the_refresh(
    hass: HomeAssistant, setup_washdata_entry: Any, failing: str
) -> None:
    """The operational scan (or any other pass) raises: every later pass still
    runs, the Start Threshold correction included, and the refresh never raises."""
    entry = await setup_washdata_entry(options={CONF_OFF_DELAY: 120})
    await hass.async_block_till_done()
    lm = hass.data[DOMAIN][entry.entry_id].learning_manager
    ran: list[str] = []

    def spy(name: str) -> Any:
        real = getattr(SuggestionEngine, name)

        def run(self: Any, *args: Any, **kwargs: Any) -> Any:
            ran.append(name)
            if name == failing:
                raise RuntimeError("scan broke")
            return real(self, *args, **kwargs)

        return run

    with (
        patch.object(lm, "cadence", return_value=(60.0, 30.0)),
        patch.multiple(SuggestionEngine, **{n: spy(n) for n in PASSES}),
    ):
        await lm.async_refresh_suggestions()
    assert ran == PASSES


async def test_the_manual_analysis_still_stops_at_a_failed_pass(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    entry = await setup_washdata_entry(options={CONF_OFF_DELAY: 120})
    await hass.async_block_till_done()
    lm = hass.data[DOMAIN][entry.entry_id].learning_manager
    start = AsyncMock()
    with (
        patch.object(lm, "cadence", return_value=(60.0, 30.0)),
        patch.object(
            SuggestionEngine, "generate_operational_suggestions",
            side_effect=RuntimeError("scan broke"),
        ),
        patch.object(SuggestionEngine, "generate_start_threshold_suggestions", new=start),
        pytest.raises(RuntimeError, match="scan broke"),
    ):
        await lm._async_run_all_passes(raise_errors=True)  # noqa: SLF001
    start.assert_not_called()


# ── Uncached template resampling runs in the executor ────────────────────────

N_PROFILES, N_POINTS, STEP = 5, 1500, 2.0


def _trace_of(k: int) -> list[list[float]]:
    """Profile ``k``: a heater block at its own position, then tumbling."""
    return [
        [i * STEP, 2000.0 if 100 * k <= i < 100 * k + 200 else 150.0 + 50.0 * ((i // 30) % 2)]
        for i in range(N_POINTS)
    ]


def _single_cycle_store(hass: Any) -> Any:
    """Five profiles of one cycle each: every template is a sample-backed one."""
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        store = ps_mod.ProfileStore(hass, "entry", min_duration_ratio=0.1, max_duration_ratio=1.8)
    store.async_save = AsyncMock()
    for k in range(N_PROFILES):
        name = f"P{k}"
        store._data["past_cycles"].append({  # noqa: SLF001
            "id": f"c{k}", "profile_name": name, "status": "completed",
            "start_time": f"2026-11-0{k + 1}T08:00:00+00:00",
            "duration": N_POINTS * STEP, "power_data": _trace_of(k),
        })
        store._data["profiles"][name] = {  # noqa: SLF001
            "avg_duration": N_POINTS * STEP, "sample_cycle_id": f"c{k}",
        }
    return store


def _query() -> list[tuple[float, float]]:
    return [(float(t), float(p)) for t, p in _trace_of(2)[:900]]


def _ranking(result: Any) -> list[tuple[str, float]]:
    return [(c["name"], round(float(c["score"]), 9)) for c in result.candidates]


async def test_a_cold_match_resamples_its_templates_off_the_loop(hass: HomeAssistant) -> None:
    """A match on a cache miss resampled every sample-backed template on the loop.
    Now none on the loop, the ranking is the inline build's, and a second match
    on the same grid resamples nothing."""
    real = ps_mod.resample_uniform
    threads: list[int] = []

    def recording(*args: Any, **kwargs: Any) -> Any:
        threads.append(threading.get_ident())
        return real(*args, **kwargs)

    loop_thread = threading.get_ident()
    store = _single_cycle_store(hass)
    query = _query()
    with patch.object(ps_mod, "resample_uniform", side_effect=recording):
        result = await store.async_match_profile(query, query[-1][0], in_progress=True)
        assert threads and loop_thread not in threads
        threads.clear()
        again = await store.async_match_profile(query, query[-1][0], in_progress=True)
        assert threads == []
    assert _ranking(again) == _ranking(result)
    # The same match with every template resampled inline on the loop, as before.
    inline = _single_cycle_store(hass)
    with patch.object(ps_mod, "_MATCH_SEGMENT_INLINE_POINTS", 10**9):
        expected = await inline.async_match_profile(query, query[-1][0], in_progress=True)
    assert result.best_profile == expected.best_profile == "P2"
    assert _ranking(result) == _ranking(expected)


async def test_a_trace_replaced_during_the_job_is_not_cached(mock_hass: Any) -> None:
    """A trim (a new trace list, its cache entries dropped) while the templates
    resample: the old trace's segment is not cached, the build uses the new one."""
    store = _single_cycle_store(mock_hass)
    cycle = store._data["past_cycles"][0]  # noqa: SLF001
    trimmed = cycle["power_data"][:600]

    async def trimming_job(target: Any, *args: Any) -> Any:
        out = target(*args)
        cycle["power_data"] = trimmed
        store._drop_cached_sample_segments("c0")  # noqa: SLF001
        return out

    mock_hass.async_add_executor_job = trimming_job
    snaps = await store._async_match_snapshots(STEP)  # noqa: SLF001
    fresh = ps_mod._resample_sample_trace(trimmed, STEP)  # noqa: SLF001
    assert fresh is not None
    p0 = next(s for s in snaps if s["name"] == "P0")
    assert p0["sample_power"] == fresh.power.tolist()
    cached = store._cached_sample_segments[("c0", STEP)]  # noqa: SLF001
    assert cached.power.tolist() == fresh.power.tolist()
    # The others were resampled in the job and cached as the build used them.
    p1 = next(s for s in snaps if s["name"] == "P1")
    assert p1["sample_power"] == ps_mod._resample_sample_trace(_trace_of(1), STEP).power.tolist()  # noqa: SLF001
