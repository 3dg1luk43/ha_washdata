"""Audit PLAYGROUND-07: imported dishwasher cycles banked their whole end wait.

The import replays unmatched, so ``CycleDetector._keep_tail_cap`` has no expected
duration and returns None, a dishwasher's timeout finish keeps its tail, and the
banked-tail repair skips ``backfill_cycles``: on the corpus an imported dishwasher
cycle ran a median 1.2x what live stored (the Beko exactly +3600 s, its
``min_off_gap``), and once labelled it drags ``target_duration`` and the ETA long.
Now each imported dishwasher candidate is matched on its activity, and the
programme it confidently matches bounds the tail with the live cap's own rule;
the repair's own trim stores it.
"""
from __future__ import annotations

import asyncio
import json
import statistics
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from custom_components.ha_washdata import history_import as hi
from custom_components.ha_washdata import task_registry, ws_api
from custom_components.ha_washdata.const import TERMINAL_QUIET_CAP_S
from custom_components.ha_washdata.cycle_detector import CycleDetectorConfig

sys.path.insert(0, str(Path(__file__).parent))
from test_ws_history_import import T0, _conn, _hass, _manager, _scan, _upload  # noqa: E402

ENTITY = "sensor.washer_power"

DISHWASHER = CycleDetectorConfig(
    min_power=2.0, off_delay=300, device_type="dishwasher", min_off_gap=3600,
    completion_min_seconds=900, start_energy_threshold=0.2, stop_threshold_w=1.2,
)


def _cand(points, duration, reason="timeout"):
    return {"power_data": points, "duration": duration, "termination_reason": reason}


# A wash ending at t=5400 s (pump-out), then a dead plug until the 3600 s timeout.
_RUN = [[float(t), 1900.0 if t < 1200 else 60.0] for t in range(0, 5401, 30)]
_TAIL = [[5400.0 + 30 * k, 0.4] for k in range(1, 121)]


def test_the_cut_keeps_the_programmes_measured_drying_and_no_more():
    cand = _cand(_RUN + _TAIL, 5400.0 + 3600.0)
    assert hi.import_tail_cut_s(cand, DISHWASHER, 900.0) == 5400.0 + 900.0
    # The allowance is capped exactly as live caps it.
    long_wait = _cand(
        _RUN + [[5400.0 + 30 * k, 0.4] for k in range(1, 400)], 5400.0 + 12000.0
    )
    assert hi.import_tail_cut_s(long_wait, DISHWASHER, 99999.0) == 5400.0 + TERMINAL_QUIET_CAP_S


def test_a_run_that_already_dried_before_its_pump_out_keeps_no_allowance():
    # 1200 s of quiet, then the terminal pump-out at 6600 s, then the wait.
    run = [[float(t), 1900.0 if t < 1200 else 60.0] for t in range(0, 5401, 30)]
    run += [[5400.0 + 30 * k, 0.4] for k in range(1, 40)]
    run += [[6600.0, 40.0], [6630.0, 0.4]]
    run += [[6630.0 + 30 * k, 0.4] for k in range(1, 121)]
    cand = _cand(run, 6600.0 + 3600.0)
    assert hi.import_tail_cut_s(cand, DISHWASHER, 1000.0) == 6600.0


def test_no_measured_span_falls_back_to_the_expected_end_and_the_trusted_floor_holds():
    cand = _cand(_RUN + _TAIL, 9000.0)
    assert hi.import_tail_cut_s(cand, DISHWASHER, None, expected_s=6000.0) == 6000.0
    assert hi.import_tail_cut_s(cand, DISHWASHER, None) is None   # no evidence: no cut
    # Never below 0.9 x the length the user vouched for.
    assert hi.import_tail_cut_s(cand, DISHWASHER, 900.0, trusted_min_s=8000.0) == 7200.0


def test_only_a_dishwasher_timeout_finish_is_cut():
    cand = _cand(_RUN + _TAIL, 9000.0)
    washer = CycleDetectorConfig(min_power=2.0, off_delay=300, device_type="washing_machine")
    assert hi.import_tail_cut_s(cand, washer, 900.0) is None
    assert hi.import_tail_cut_s(_cand(_RUN + _TAIL, 9000.0, "smart"), DISHWASHER, 900.0) is None
    # Under a minute to reclaim is left alone, as the repair leaves it.
    assert hi.import_tail_cut_s(_cand(_RUN + _TAIL[:1], 5430.0), DISHWASHER, 10.0) is None


def _dishwasher_csv() -> str:
    """One 90 min dishwasher run, then a 3 h quiet trickle the plug keeps reporting."""
    rows = ["entity_id,state,last_changed"]
    t = 0.0
    while t <= 5400:
        w = 1900 if t < 1200 else 60
        rows.append(f"{ENTITY},{w},{(T0 + timedelta(seconds=t)).isoformat()}")
        t += 30
    while t <= 5400 + 3 * 3600:
        rows.append(f"{ENTITY},0.4,{(T0 + timedelta(seconds=t)).isoformat()}")
        t += 300
    return "\n".join(rows)


def _dishwasher_manager(hass):
    manager = _manager(hass)
    manager.detector = SimpleNamespace(config=DISHWASHER)
    store = manager.profile_store
    store.async_match_profile = AsyncMock(return_value=SimpleNamespace(
        best_profile="Eco", label_confidence=0.9, is_ambiguous=False,
        ambiguity_margin=0.3, expected_duration=6000.0,
    ))
    store.profile_terminal_quiet_seconds = lambda name: 900.0 if name == "Eco" else None
    store.profile_trusted_min_duration = lambda _name: None
    return manager


async def _apply(hass, conn, manager, scan_id, accept):
    reg = task_registry.get_registry(hass)
    with patch.object(ws_api, "_get_manager", return_value=manager):
        ws_api.ws_apply_history_import(
            hass, conn, {"id": 20, "entry_id": "e", "scan_task_id": scan_id, "accept": accept}
        )
        task_id = conn.send_result.call_args.args[1]["task_id"]
        for _ in range(200):
            if reg.get(task_id).state != task_registry.STATE_RUNNING:
                break
            await asyncio.sleep(0)
    return reg.get(task_id)


@pytest.mark.asyncio
async def test_an_imported_dishwasher_cycle_is_stored_without_the_end_wait():
    hass, conn = _hass(), _conn()
    manager = _dishwasher_manager(hass)
    entry = SimpleNamespace(entry_id="e", options={"device_type": "dishwasher"}, data={})
    token = await _upload(hass, conn, manager, _dishwasher_csv())
    scan = await _scan(hass, conn, manager, entry, token)
    assert scan.state == task_registry.STATE_DONE, scan.error
    [seg] = scan.result["segments"]
    # Matched on its activity only (the wait would wreck the duration terms).
    probe, probe_dur = manager.profile_store.async_match_profile.call_args.args[:2]
    assert probe_dur == 5400.0 and max(p[0] for p in probe) == 5400.0
    # The preview shows what will be stored, and how much was trimmed.
    assert seg["duration_s"] == 5400.0 + 900.0
    # Stored live it would have been 5400 + 900; the unmatched replay said 9000.
    assert seg["banked_tail_s"] == 9000.0 - 6300.0

    applied = await _apply(hass, conn, manager, scan.id, [0])
    assert applied.state == task_registry.STATE_DONE, applied.error
    [stored] = manager.profile_store.get_backfill_cycles()
    assert stored["duration"] == 5400.0 + 900.0
    assert stored["power_data"][-1][0] == 5400.0 + 900.0
    assert all(p[0] <= 6300.0 for p in stored["power_data"])
    # Unlabelled: the match only named the programme whose statistics bound the cut.
    assert stored["profile_name"] is None


@pytest.mark.asyncio
async def test_without_a_confident_programme_nothing_is_cut():
    hass, conn = _hass(), _conn()
    manager = _dishwasher_manager(hass)
    manager.profile_store.async_match_profile = AsyncMock(return_value=SimpleNamespace(
        best_profile="Eco", label_confidence=0.9, is_ambiguous=True,
        ambiguity_margin=0.01, expected_duration=6000.0,
    ))
    entry = SimpleNamespace(entry_id="e", options={"device_type": "dishwasher"}, data={})
    token = await _upload(hass, conn, manager, _dishwasher_csv())
    scan = await _scan(hass, conn, manager, entry, token)
    [seg] = scan.result["segments"]
    assert "banked_tail_s" not in seg and seg["duration_s"] == 9000.0


# ─── The corpus: re-import stored traces, compare with what live stored ───────


_BEKO = (
    Path(__file__).parent.parent / "cycle_data" / "KoLSMS" / "dishwasher"
    / "washdata_export_01KWRK9X_424_beko.json"
)


@pytest.mark.slow
def test_reimported_beko_cycles_store_what_live_stored():
    if not _BEKO.exists():
        pytest.skip("cycle_data corpus not present")
    import importlib.util  # noqa: PLC0415

    from custom_components.ha_washdata import playground  # noqa: PLC0415
    from custom_components.ha_washdata.suggestion_engine import _cycle_readings  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location(
        "wd_ege_07", Path(__file__).parent.parent / "devtools" / "end_gate_eval.py"
    )
    ege = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ege
    spec.loader.exec_module(ege)
    doc = json.loads(_BEKO.read_text(encoding="utf-8"))
    data = dict(doc["data"])
    for key in ("past_cycles", "reference_cycles", "backfill_cycles"):
        data[key] = list(data.get(key) or [])
    cfg, store, opts = ege._production(doc, data)  # noqa: SLF001
    # The stored durations as an upgraded store holds them.
    ege._run(store.async_repair_banked_tails(float(cfg.stop_threshold_w), "dishwasher"))  # noqa: SLF001
    raw_ratios, cut_ratios = [], []
    for cyc in data["past_cycles"]:
        if cyc.get("status") != "completed":
            continue
        pts = _cycle_readings(cyc)
        if len(pts) < 20:
            continue
        t0 = playground._cycle_base_time(cyc)  # noqa: SLF001
        samples = [(t0 - timedelta(minutes=20), 0.0)] + [
            (t0 + timedelta(seconds=float(t)), float(p)) for t, p in pts
        ]
        samples.append((samples[-1][0] + timedelta(seconds=30), 0.0))
        runner = hi.build_scan(samples, cfg)
        while not runner.finished:
            runner.step(4000)
        payload = runner.finalize()
        kept = [c for s, c in zip(payload["segments"], payload["cycles"]) if s["accept"]]
        stored = float(cyc["duration"])
        raw_ratios.append(sum(float(c["duration"]) for c in kept) / stored)
        ege._run(hi.async_import_tail_cuts(store, cfg, kept, opts))  # noqa: SLF001
        cut_ratios.append(sum(float(hi.effective_duration(c)) for c in kept) / stored)
    assert len(cut_ratios) >= 10
    # Measured: median 1.251 before, 1.000 after.
    assert statistics.median(raw_ratios) > 1.15
    assert 0.97 <= statistics.median(cut_ratios) <= 1.03
    assert max(cut_ratios) <= 1.05 and min(cut_ratios) >= 0.9
