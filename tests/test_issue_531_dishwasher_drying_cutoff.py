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
            "profiles": {"ECO": {"avg_duration": stale}},
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
    assert store.arm_cut_short_rebuild() is False  # once only
