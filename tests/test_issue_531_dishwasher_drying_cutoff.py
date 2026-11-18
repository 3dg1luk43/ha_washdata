"""Issue #531: a dishwasher's ECO finished ~60 min early, and earlier every month.

The Bauknecht dries 53 min at 3.3 W (its own standby level, under a 4.5 W Stop
Threshold) and beeps at 6.3 W at the real end, 191 min. Smart Termination ends a run
at the programme's learned length, and the run is then stored and learned at that
length: once a few runs were closed inside the drying, the length could only
shrink (190 -> 129 min over nine months). Two fixes, both pinned here:

* a run closed inside the programme's quiet phase (others go on into a long quiet
  and then a terminal event) is a cut-off: it builds neither the envelope nor the
  learned length (`profile_store.cut_short_mask`);
* the 6.3 W beep after the drying is the terminal event although it is 0.3% of the
  peak: a reading over the Stop Threshold that stands out of a long quiet arms the
  end spike (`CycleDetector._is_terminal_blip`), so the run ends at the beep rather
  than the whole spike window after it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import DOMAIN
from custom_components.ha_washdata.cycle_detector import CycleDetector
from custom_components.ha_washdata.detector_config import build_detector_config
from custom_components.ha_washdata.profile_store import _events_after_quiet, cut_short_mask

T0 = datetime(2026, 11, 10, 8, tzinfo=timezone.utc)
BEEP_S = 191.6 * 60


def _eco(*, cut_s: float | None = None, beep: bool = True, hold_s: float = 0.0) -> list[tuple[float, float]]:
    """A synthetic ECO run, 10 s readings: heating, washing, pump-outs to 138 min,
    drying at 3.3 W, the 6.3 W beep for 40 s at 191.6 min, then standby at 3.3 W."""
    pts: list[tuple[float, float]] = []
    t = 0.0
    end = BEEP_S + 40 + hold_s
    while t <= end:
        m = t / 60
        if 33 <= m < 47 or 112 <= m < 124:
            p = 1950.0
        elif m < 110:
            p = 30.0
        elif 130 <= m < 131 or 137.5 <= m < 138.5:
            p = 25.0
        elif BEEP_S <= t < BEEP_S + 40 and beep:
            p = 6.3
        else:
            p = 3.3
        pts.append((t, p))
        t += 10.0
    if cut_s is not None:
        pts = [(t, p) for t, p in pts if t <= cut_s]
    return pts


def _run(points: list[tuple[float, float]], pinned: bool = False) -> tuple[list[float], list[float], float, bool, float]:
    return ([t for t, _ in points], [p for _, p in points], points[-1][0], pinned, max(p for _, p in points))


def test_the_beep_after_the_drying_is_a_terminal_event() -> None:
    events = _events_after_quiet(*_run(_eco(hold_s=600))[:2], 1950.0)
    assert events and abs(events[-1][1] - BEEP_S) < 20  # after the 53 min drying
    assert events[-1][1] - events[-1][0] > 50 * 60


def test_runs_closed_inside_the_drying_are_cut_offs() -> None:
    full = [_run(_eco(hold_s=30)) for _ in range(5)]
    cut = [_run(_eco(cut_s=131 * 60)) for _ in range(12)]
    mask = cut_short_mask(full + cut)
    assert mask == [False] * 5 + [True] * 12
    # The user's word wins: a pinned (golden / corrected) cut run still counts.
    assert cut_short_mask(full + [_run(_eco(cut_s=131 * 60), pinned=True)]) == [False] * 6


def test_a_run_that_only_waited_longer_does_not_make_the_others_cut_offs() -> None:
    """Runs stopped at their last activity, others that waited 15 min more in silence
    (no event after the quiet): nothing was cut short (a corpus Express 60)."""
    on_time = [_run([(t, p) for t, p in _eco(beep=False) if t <= 140 * 60]) for _ in range(6)]
    waited = [_run([(t, p if t <= 140 * 60 else 0.0) for t, p in _eco(beep=False) if t <= 160 * 60]) for _ in range(6)]
    assert not any(cut_short_mask(on_time + waited))


def test_a_load_adaptive_programme_s_short_runs_are_complete() -> None:
    """Each run of an Auto programme is the whole programme scaled to its length,
    drying and final pump-out included (the #375 test's profile, 95-169 min): a long
    run still goes on past a short run's end, but the short one ended on its own
    event, so it is not a cut-off."""
    def scaled(total_min: float) -> tuple:
        pts, t = [], 0.0
        while t <= total_min * 60:
            f = t / (total_min * 60)
            p = 2100.0 if 0.05 <= f < 0.17 or 0.55 <= f < 0.65 else 60.0 if f < 0.80 else 0.5 if f < 0.99 else 30.0
            pts.append((t, p))
            t += 60.0
        return _run(pts)
    assert not any(cut_short_mask([scaled(m) for m in (95, 100, 107, 112, 117, 120, 127, 135, 145, 169)]))


def test_too_few_longer_runs_cut_nothing() -> None:
    assert not any(cut_short_mask([_run(_eco(hold_s=30)) for _ in range(2)]
                                  + [_run(_eco(cut_s=131 * 60)) for _ in range(20)]))


def _replay(expected_min: float, points: list[tuple[float, float]]) -> tuple[float | None, Any]:
    cfg = build_detector_config({"stop_threshold_w": 4.5, "start_threshold_w": 5.0},
                                {"device_type": "dishwasher"}, "dishwasher")
    ends: list[dict] = []
    det = CycleDetector(cfg, lambda _o, _n: None, ends.append)
    matched = False
    for t, p in points + [(points[-1][0] + 10.0 * k, 3.3) for k in range(1, 400)]:
        ts = T0 + timedelta(seconds=t)
        det.process_reading(p, ts)
        if not matched and t >= 20 * 60:
            det.update_match(("ECO", 0.8, expected_min * 60, None, False, False))
            det.set_match_committed(True)
            matched = True
        if ends:
            return t / 60, ends[0]
    return None, None


def test_the_beep_ends_the_run_at_the_programme_s_length() -> None:
    end_min, cycle = _replay(192.0, _eco(hold_s=1800))
    assert cycle is not None and cycle["termination_reason"] == "smart"
    assert BEEP_S / 60 <= end_min <= BEEP_S / 60 + 5  # at the beep, not 30 min later


def test_a_drying_blip_under_the_progress_gate_does_not_end_it() -> None:
    """The same blip at 60% of the programme is not its end (the 85% gate holds)."""
    blip = [(t, 6.3 if 165 * 60 <= t < 165 * 60 + 40 else p) for t, p in _eco(hold_s=1800)]
    end_min, _cycle = _replay(260.0, blip)  # expected 260: 165 min is 63%
    assert end_min is None or end_min > 200


async def test_the_envelope_and_its_length_leave_cut_offs_out(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    """The live expected duration is the envelope's `target_duration`: both it and
    `avg_duration` come from the full runs, so the next run waits for the beep."""
    entry = await setup_washdata_entry(device_type="dishwasher", options={"stop_threshold_w": 4.5})
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    runs = [_eco(hold_s=30)] * 5 + [_eco(cut_s=131 * 60)] * 12
    store._data["profiles"]["ECO"] = {}  # noqa: SLF001
    store._data["past_cycles"] = [  # noqa: SLF001
        {"id": f"c{i}", "profile_name": "ECO", "status": "completed", "termination_reason": "smart",
         "start_time": (T0 + timedelta(days=i)).isoformat(), "duration": pts[-1][0],
         "power_data": [[t, p] for t, p in pts]}
        for i, pts in enumerate(runs)
    ]
    assert await store.async_rebuild_envelope("ECO")
    full_min = _eco(hold_s=30)[-1][0] / 60
    assert abs(store.get_profiles()["ECO"]["avg_duration"] / 60 - full_min) < 1
    assert abs(store._data["envelopes"]["ECO"]["target_duration"] / 60 - full_min) < 1  # noqa: SLF001


async def test_a_store_from_before_the_rule_is_rebuilt_once_at_setup(
    hass: HomeAssistant, hass_storage: dict[str, Any], enable_custom_integrations
) -> None:
    """The reporter's store: its envelope still holds the cut-off length. Without a
    rebuild the first run after the update would end early once more."""
    from pytest_homeassistant_custom_component.common import MockConfigEntry  # noqa: PLC0415

    from custom_components.ha_washdata.const import (  # noqa: PLC0415
        CONFIG_ENTRY_MINOR_VERSION,
        CONFIG_ENTRY_VERSION,
        CUT_SHORT_REBUILD_KEY,
        STORAGE_KEY,
        STORAGE_VERSION,
    )

    hass.states.async_set("sensor.dw_power", "3.3", {"unit_of_measurement": "W"})
    entry = MockConfigEntry(
        domain=DOMAIN, title="Dishwasher",
        data={"name": "Dishwasher", "power_sensor": "sensor.dw_power", "device_type": "dishwasher"},
        options={"stop_threshold_w": 4.5}, unique_id="washdata_dw_531",
        version=CONFIG_ENTRY_VERSION, minor_version=CONFIG_ENTRY_MINOR_VERSION,
    )
    entry.add_to_hass(hass)
    runs = [_eco(hold_s=30)] * 5 + [_eco(cut_s=131 * 60)] * 12
    stale = 131 * 60.0
    hass_storage[f"{STORAGE_KEY}.{entry.entry_id}"] = {
        "version": STORAGE_VERSION, "minor_version": 1, "key": f"{STORAGE_KEY}.{entry.entry_id}",
        "data": {
            # "Imported": known only by its saved envelope (no cycles of its own).
            "profiles": {"ECO": {"avg_duration": stale}, "Imported": {"avg_duration": 3600.0}},
            "envelopes": {"Imported": {
                "time_grid": [0.0, 1800.0, 3600.0], "avg": [5.0, 1900.0, 3.0],
                "min": [0.0, 1800.0, 0.0], "max": [10.0, 2000.0, 6.0], "std": [1.0, 50.0, 1.0],
                "target_duration": 3600.0, "cycle_count": 3,
            }},
            "past_cycles": [
                {"id": f"c{i}", "profile_name": "ECO", "status": "completed",
                 "termination_reason": "smart", "start_time": (T0 + timedelta(days=i)).isoformat(),
                 "duration": pts[-1][0], "power_data": [[t, p] for t, p in pts]}
                for i, pts in enumerate(runs)
            ],
        },
    }
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    assert store._data.get(CUT_SHORT_REBUILD_KEY) is True  # noqa: SLF001
    assert store.get_profiles()["ECO"]["avg_duration"] > stale + 50 * 60
    assert store.cut_short_rebuild_pending() is False  # once only
    # The envelope-only profile is not sent through a rebuild that would drop it.
    assert "Imported" in store._data["envelopes"]  # noqa: SLF001


# PR #529 review (CodeRabbit, 2026-11-18) ------------------------------------------


def test_a_heater_restart_after_a_pause_is_not_a_terminal_event() -> None:
    """Main activity after 700 s of quiet resets the quiet; it is not an end."""
    pts = [(t, 2000.0 if t < 600 or t >= 1300 else 2.0) for t in range(0, 2000, 10)]
    assert _events_after_quiet([t for t, _ in pts], [p for _, p in pts], 2000.0) == []


def test_siblings_still_working_at_the_candidate_s_end_do_not_vote() -> None:
    """Siblings whose quiet starts well after the candidate ended were not drying
    when it ended: a silent complete run stays complete."""
    def run(total_s: float, dry_from: float | None) -> tuple:
        pts = []
        for t in range(0, int(total_s) + 1, 10):
            if dry_from is not None and t >= dry_from:
                p = 25.0 if t >= total_s - 40 else 0.5
            else:
                p = 1950.0 if 600 <= t < 1200 else 30.0
            pts.append((float(t), p))
        return _run(pts)
    complete = [run(3000, None) for _ in range(4)]          # ends silently at 3000 s
    longer = [run(5400, 3000 + 1200) for _ in range(4)]     # dry from 4200 s, pump-out at 5360 s
    assert not any(cut_short_mask(complete + longer)[:4])


def test_mis_captures_do_not_vote_and_a_user_trim_counts_as_corrected() -> None:
    from custom_components.ha_washdata.profile_store import ProfileStore  # noqa: PLC0415

    class _Probe(ProfileStore):
        def __init__(self) -> None:  # noqa: D401 - only the build's own state
            self.device_type = "dishwasher"
            self.dtw_bandwidth = 0.1
            self.energy_mode = "mean"
            self._logger = __import__("logging").getLogger("probe")

    full = _eco(hold_s=30)
    cut = _eco(cut_s=131 * 60)
    # Three 10 W mis-captures with a 1 W rise after a long quiet: no votes.
    miscap = [(t, 10.0 if t < 3000 else (1.0 if t < 4800 else 2.5)) for t, _ in full[: len(full) // 2]]

    def cyc(pts: list, i: int, **extra: Any) -> dict:
        return {"id": f"c{i}", "profile_name": "ECO", "status": "completed", "duration": pts[-1][0],
                "power_data": [[t, p] for t, p in pts], **extra}

    probe = _Probe()
    built = probe._rebuild_envelope_sync([cyc(full, 0), cyc(full, 1), cyc(cut, 2)] + [cyc(miscap, 3 + k) for k in range(3)])  # noqa: SLF001
    assert built is not None and len(built[1]) == 3  # the cut run is kept: 2 full voters only
    trimmed = cyc(cut, 9, meta={"trim": [0.0, cut[-1][0]]})
    built = probe._rebuild_envelope_sync([cyc(full, k) for k in range(5)] + [trimmed])  # noqa: SLF001
    assert built is not None and len(built[1]) == 6  # a user-trimmed run is never a cut-off
    # A manual_duration the build rejects as a unit error (385200 s on a 7860 s
    # trace) is no correction: that run is still a cut-off (PR #529 review).
    absurd = cyc(cut, 10, manual_duration=385200.0)
    built = probe._rebuild_envelope_sync([cyc(full, k) for k in range(5)] + [absurd])  # noqa: SLF001
    assert built is not None and len(built[1]) == 5


def test_a_sensor_outage_is_not_quiet_before_a_blip() -> None:
    """700 s with no reading, then the blip: the quiet was not observed, so the
    beep rule does not arm the end; the run waits as before (PR #529 review)."""
    pts = [(t, p) for t, p in _eco(hold_s=1800) if not (BEEP_S - 700 <= t < BEEP_S)]
    end_min, _cycle = _replay(192.0, pts)
    assert end_min is None or end_min > BEEP_S / 60 + 10


def test_an_unmatched_run_takes_no_blip_as_its_end() -> None:
    from custom_components.ha_washdata.cycle_detector import CycleDetector as _CD  # noqa: PLC0415

    cfg = build_detector_config({"stop_threshold_w": 4.5}, {"device_type": "dishwasher"}, "dishwasher")
    det = _CD(cfg, lambda _o, _n: None, lambda _c: None)
    for t, p in _eco(hold_s=0):
        det.process_reading(p, T0 + timedelta(seconds=t))
    assert det._is_terminal_blip(T0 + timedelta(seconds=BEEP_S + 1), 6.3) is False  # noqa: SLF001


def test_short_outages_do_not_add_up_to_observed_quiet() -> None:
    """A pump reading, 90 s of 3.3 W, a 469 s outage, 60 s more, then the beep: no
    single gap is 600 s, but most of the quiet was not observed (PR #529 review)."""
    pump = BEEP_S - 711
    pts = []
    for t, p in _eco(hold_s=1800):
        if pump <= t < pump + 10:
            p = 25.0  # the last activity before the beep
        if BEEP_S - 529 <= t < BEEP_S - 60:
            continue  # the outage
        pts.append((t, p))
    # The beep at 91% of a 210 min programme: only the blip could end it there.
    end_min, _cycle = _replay(210.0, pts)
    assert end_min is None or end_min > 200
    # The same beep after continuously observed drying does end it.
    end_ok, _ = _replay(210.0, _eco(hold_s=1800))
    assert end_ok is not None and end_ok < BEEP_S / 60 + 5


def test_an_unobserved_gap_is_not_quiet_before_a_terminal_event() -> None:
    """A 2 W reading, 900 s with no reading, then 20 W: not a terminal event."""
    pts = [(float(t), 1900.0) for t in range(0, 600, 10)] + [(600.0, 2.0), (1500.0, 20.0)]
    assert _events_after_quiet([t for t, _ in pts], [p for _, p in pts], 1900.0) == []


async def test_the_one_time_rebuild_stays_pending_after_a_failure_or_a_swap(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    """PR #529 review: a profile whose rebuild raised, or data an undo replaced while
    the pass ran, must not be marked rebuilt."""
    from unittest.mock import patch  # noqa: PLC0415

    from custom_components.ha_washdata.const import CUT_SHORT_REBUILD_KEY  # noqa: PLC0415

    entry = await setup_washdata_entry(device_type="dishwasher", options={"stop_threshold_w": 4.5})
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    store._data["profiles"] = {"A": {}, "B": {}}  # noqa: SLF001
    store._data["past_cycles"] = [  # noqa: SLF001
        {"id": f"{n}{i}", "profile_name": n, "status": "completed", "duration": 600.0,
         "start_time": (T0 + timedelta(days=i)).isoformat(),
         "power_data": [[float(t), 1000.0 if t < 500 else 1.0] for t in range(0, 601, 10)]}
        for n in "AB" for i in range(3)
    ]
    store._data.pop(CUT_SHORT_REBUILD_KEY, None)  # noqa: SLF001
    real = store.async_rebuild_envelope

    async def flaky(name: str) -> bool:
        if name == "B":
            raise RuntimeError("boom")
        return await real(name)

    with patch.object(store, "async_rebuild_envelope", side_effect=flaky):
        assert await store.async_rebuild_for_cut_short_rule() == 1  # A rebuilt
    assert store.cut_short_rebuild_pending()

    async def swapped(name: str) -> bool:
        store._data_generation = getattr(store, "_data_generation", 0) + 1  # noqa: SLF001 - an undo
        return await real(name)

    with patch.object(store, "async_rebuild_envelope", side_effect=swapped):
        await store.async_rebuild_for_cut_short_rule()
    assert store.cut_short_rebuild_pending()
    await store.async_rebuild_for_cut_short_rule()
    assert not store.cut_short_rebuild_pending()
