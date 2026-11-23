"""Issue #531 follow-up: 0.6.1's cut-off rule did not help the reporter.

The ECO programme (191 min, 53 min of drying at 3.3 W, a 6.3 W beep at the end)
still finished ~61 min early, and its learned length stayed at the cut-off ~133 min.
Two reasons, both reproduced by ``devtools/duration_ratchet_eval.py``:

* the full-length runs predate 0.5.8, whose per-programme trace cap (register item
  463) kept only the newest 20 traces: they are summaries without a trace, and the
  rule voted from traced runs only, so no sibling showed the programme's end
  (``--traceless 98``: learned 129 min, every run ended 61 min early);
* a run closed before the low-power pump-outs that follow the last heating
  (123-138 min) got no vote even from traced full runs: their quiet starts after
  the last pump-out, more than ``TERMINAL_BLIP_QUIET_MIN_S`` after its end
  (``--cut-min 127``: learned 127 min with all 42 full runs traced).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import (
    CUT_SHORT_REBUILD_KEY,
    CUT_SHORT_RULE_REVISION,
    DOMAIN,
)
from custom_components.ha_washdata.profile_store import ProfileStore, cut_short_mask

T0 = datetime(2026, 2, 1, 8, tzinfo=timezone.utc)
BEEP_S = 191.6 * 60
FULL_S = BEEP_S + 40


def _eco(*, cut_s: float | None = None) -> list[tuple[float, float]]:
    """A synthetic ECO run, 10 s readings: heating, washing, the last heating to
    124 min, pump-outs at 130 and 137.5 min, drying at 3.3 W, the 6.3 W beep for
    40 s at 191.6 min; cut at ``cut_s`` if given."""
    pts: list[tuple[float, float]] = []
    t = 0.0
    while t <= FULL_S:
        m = t / 60
        if 33 <= m < 47 or 112 <= m < 124:
            p = 1950.0
        elif m < 110:
            p = 30.0
        elif 130 <= m < 131 or 137.5 <= m < 138.5:
            p = 25.0
        elif BEEP_S <= t < BEEP_S + 40:
            p = 6.3
        else:
            p = 3.3
        pts.append((t, p))
        t += 10.0
    if cut_s is not None:
        pts = [(t, p) for t, p in pts if t <= cut_s]
    return pts


def _run(points: list[tuple[float, float]]) -> tuple[list[float], list[float], float, bool, float]:
    return ([t for t, _ in points], [p for _, p in points], points[-1][0], False, max(p for _, p in points))


def _cycle(i: int, points: list[tuple[float, float]], *, traced: bool = True,
           reason: str = "smart") -> dict[str, Any]:
    """A stored ECO cycle; ``traced=False`` is one the pre-0.5.8 trace cap stripped
    (trace, sampling interval and artifacts gone, the summary kept)."""
    cycle: dict[str, Any] = {
        "id": f"c{i:03d}", "profile_name": "ECO", "status": "completed",
        "termination_reason": reason, "label_source": "manual",
        "start_time": (T0 + timedelta(days=i)).isoformat(), "duration": points[-1][0],
    }
    if traced:
        cycle["power_data"] = [[t, p] for t, p in points]
    return cycle


class _Probe(ProfileStore):
    def __init__(self) -> None:  # noqa: D107 - only what the envelope build reads
        self.device_type = "dishwasher"
        self.dtw_bandwidth = 0.1
        self.energy_mode = "mean"
        self._logger = logging.getLogger("probe")


def _built_length_min(cycles: list[dict[str, Any]]) -> tuple[float, float, int]:
    """``(target_duration, avg of the build's durations, shape count)`` in minutes."""
    built = _Probe()._rebuild_envelope_sync(cycles)  # noqa: SLF001
    assert built is not None
    result, durations = built
    return result[5] / 60, sum(durations) / len(durations) / 60, len(durations)


def test_full_runs_without_a_trace_restore_the_programme_s_length() -> None:
    """The reporter's store: every full run lost its trace, every traced run was
    closed by Smart Termination in the drying. The traced runs stay the curve, held
    to the length the untraced runs reached, so the next run waits for the beep."""
    cycles = [_cycle(i, _eco(), traced=False, reason="timeout") for i in range(12)]
    cycles += [_cycle(12 + i, _eco(cut_s=129 * 60)) for i in range(40)]
    target, avg, shape = _built_length_min(cycles)
    assert abs(target - FULL_S / 60) < 1 and abs(avg - FULL_S / 60) < 1
    assert shape == 40  # the traced runs, not dropped


def test_one_traced_full_run_and_the_summaries_leave_the_cut_offs_out() -> None:
    """After the restored length the next run ends at the beep and keeps its trace:
    it witnesses the end, and the summaries at its length add the votes it alone
    lacks (need 6 of 59 runs), so the length does not fall back to the cut-offs."""
    cycles = [_cycle(i, _eco(), traced=False, reason="timeout") for i in range(12)]
    cycles += [_cycle(12 + i, _eco(cut_s=129 * 60)) for i in range(46)]
    cycles.append(_cycle(99, _eco()))
    target, avg, shape = _built_length_min(cycles)
    assert abs(target - FULL_S / 60) < 1 and abs(avg - FULL_S / 60) < 1
    assert shape == 1


def test_a_run_closed_before_the_pump_outs_is_a_cut_off() -> None:
    """Closed at 127 min, before the pump-outs at 130 and 137.5 min: the traced full
    runs' quiet starts at 138.5 min, but they are quiet for most of the 10 min after
    127 min, so they vote."""
    full = [_run(_eco()) for _ in range(5)]
    early = [_run(_eco(cut_s=127 * 60)) for _ in range(12)]
    assert cut_short_mask(full + early) == [False] * 5 + [True] * 12


def test_summaries_alone_do_not_cut_runs_the_appliance_ended() -> None:
    """A run that ended on its own (no Smart Termination) is no right-censored
    observation: old untraced runs that were longer do not make it a cut-off, so a
    programme whose old summaries are too long cannot hold its length up for good."""
    runs = [_run(_eco(cut_s=129 * 60)) for _ in range(10)]
    summaries = [FULL_S] * 10
    assert not any(cut_short_mask(runs, summaries=summaries, censored=[False] * 10))
    assert all(cut_short_mask(runs, summaries=summaries, censored=[True] * 10))


def test_summaries_inside_the_traced_range_need_a_witness() -> None:
    """A load-adaptive programme: untraced runs a little longer than some traced ones
    but no longer than the longest, and no traced sibling showing an end after the
    short ones. The summaries are no evidence of a cut (corpus Auto 45-65)."""
    lengths = (101, 107, 110, 114, 117, 122, 130, 134)

    def plain(total_min: float) -> tuple:
        pts = [(float(t), 1900.0 if 20 * 60 <= t < 40 * 60 else 40.0)
               for t in range(0, int(total_min * 60) + 1, 30)]
        return _run(pts)

    runs = [plain(m) for m in lengths]
    summaries = [m * 60.0 for m in (98, 103, 108, 116, 117, 128)]
    assert not any(cut_short_mask(runs, summaries=summaries, censored=[True] * len(runs)))


def test_a_store_rebuilt_under_0_6_1_is_rebuilt_once_more() -> None:
    probe = _Probe()
    probe._data = {CUT_SHORT_REBUILD_KEY: True}  # noqa: SLF001 - 0.6.1's marker
    assert probe.cut_short_rebuild_pending()
    probe._data[CUT_SHORT_REBUILD_KEY] = CUT_SHORT_RULE_REVISION  # noqa: SLF001
    assert not probe.cut_short_rebuild_pending()


async def test_the_live_store_learns_the_full_length_from_its_summaries(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    """Through the real store: the learned length and the envelope's target (what
    Smart Termination reads) are the full programme's, the usage count every run."""
    entry = await setup_washdata_entry(device_type="dishwasher", options={"stop_threshold_w": 4.5})
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    store._data["profiles"]["ECO"] = {}  # noqa: SLF001
    store._data["past_cycles"] = (  # noqa: SLF001
        [_cycle(i, _eco(), traced=False, reason="timeout") for i in range(12)]
        + [_cycle(12 + i, _eco(cut_s=131 * 60)) for i in range(30)]
    )
    assert await store.async_rebuild_envelope("ECO")
    full_min = FULL_S / 60
    assert abs(store.get_profiles()["ECO"]["avg_duration"] / 60 - full_min) < 1
    env = store._data["envelopes"]["ECO"]  # noqa: SLF001
    assert abs(env["target_duration"] / 60 - full_min) < 1
    assert env["cycle_count"] == 42


async def test_the_reporter_s_0_6_1_store_is_rebuilt_at_the_first_start(
    hass: HomeAssistant, hass_storage: dict[str, Any], enable_custom_integrations
) -> None:
    """0.6.1 had rebuilt this store (marker True) and kept the cut-off length; the
    rule changed, so the first start after the update rebuilds it again and the
    first ECO run already waits for the beep."""
    from pytest_homeassistant_custom_component.common import MockConfigEntry  # noqa: PLC0415

    from custom_components.ha_washdata.const import (  # noqa: PLC0415
        CONFIG_ENTRY_MINOR_VERSION,
        CONFIG_ENTRY_VERSION,
        STORAGE_KEY,
        STORAGE_VERSION,
    )

    hass.states.async_set("sensor.dw_power", "3.3", {"unit_of_measurement": "W"})
    entry = MockConfigEntry(
        domain=DOMAIN, title="Dishwasher",
        data={"name": "Dishwasher", "power_sensor": "sensor.dw_power", "device_type": "dishwasher"},
        options={"stop_threshold_w": 4.5}, unique_id="washdata_dw_531_traceless",
        version=CONFIG_ENTRY_VERSION, minor_version=CONFIG_ENTRY_MINOR_VERSION,
    )
    entry.add_to_hass(hass)
    stale = 129 * 60.0
    cycles = (
        [_cycle(i, _eco(), traced=False, reason="timeout") for i in range(12)]
        + [_cycle(12 + i, _eco(cut_s=stale)) for i in range(30)]
    )
    hass_storage[f"{STORAGE_KEY}.{entry.entry_id}"] = {
        "version": STORAGE_VERSION, "minor_version": 1, "key": f"{STORAGE_KEY}.{entry.entry_id}",
        "data": {
            "profiles": {"ECO": {"avg_duration": stale}},
            "envelopes": {"ECO": {
                "time_grid": [0.0, stale / 2, stale], "avg": [30.0, 1950.0, 3.3],
                "min": [0.0, 1900.0, 0.0], "max": [40.0, 2000.0, 6.0], "std": [1.0, 50.0, 1.0],
                "target_duration": stale, "cycle_count": 30,
            }},
            "past_cycles": cycles,
            CUT_SHORT_REBUILD_KEY: True,
        },
    }
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    assert store._data.get(CUT_SHORT_REBUILD_KEY) == CUT_SHORT_RULE_REVISION  # noqa: SLF001
    assert abs(store.get_profiles()["ECO"]["avg_duration"] / 60 - FULL_S / 60) < 1
    assert abs(store._data["envelopes"]["ECO"]["target_duration"] / 60 - FULL_S / 60) < 1  # noqa: SLF001
    assert len(store._data["past_cycles"]) == 42  # noqa: SLF001 - every run kept


def test_untraced_runs_do_not_raise_the_bar_for_the_traced_vote() -> None:
    """Many summaries do not dilute what traced siblings already show: 10 traced full
    runs still call 30 traced cut-offs cut off (their share is of the 40 traced
    runs), Smart Termination or not."""
    full = [_run(_eco()) for _ in range(10)]
    cut = [_run(_eco(cut_s=129 * 60)) for _ in range(30)]
    mask = cut_short_mask(full + cut, summaries=[FULL_S] * 200, censored=[False] * 40)
    assert mask == [False] * 10 + [True] * 30
