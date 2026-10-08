"""devtools/start_gate_eval.py: the start-gate harness (register item 488).

The corpus run needs raw histories (``cycle_data/`` is gitignored and holds one);
this needs none. It replays a synthetic day - idle, an isolated blip, a wash with an
aborted door-lock probe before it - through the real detector and pins what each
gate variant must report, plus the three loaders and the CLI, so the harness cannot
rot in CI.
"""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "devtools" / "start_gate_eval.py"
_spec = importlib.util.spec_from_file_location("wd_start_gate_eval", _PATH)
sge = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sge
_spec.loader.exec_module(sge)

T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
WASH = 14400.0  # seconds after T0: the stored (live) start of the wash
OPTIONS = {
    "device_type": "washing_machine",
    "start_threshold_w": 5.0,
    "stop_threshold_w": 2.0,
    "min_power": 2.0,
    "start_duration_threshold": 5.0,
    "start_energy_threshold": 0.2,
    "sampling_interval": 1.0,
}


def _at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def _day() -> tuple[list[tuple[datetime, float | None]], float]:
    """Idle (0.3 W every 30 s), a 1 s 1500 W blip, then a wash after a door-lock probe."""
    rows: list[tuple[float, float]] = []
    t = 0.0
    while t < 7200:
        rows.append((t, 0.3))
        t += 30
    rows += [(7200.0, 1500.0), (7201.0, 1500.0), (7202.0, 0.3)]
    t = 7230.0
    while t < WASH - 60:
        rows.append((t, 0.3))
        t += 30
    # Door lock: 10 W for 15 s, then nothing until the programme really starts.
    rows += [(WASH - 50, 10.0), (WASH - 45, 10.0), (WASH - 40, 10.0), (WASH - 35, 10.0),
             (WASH - 30, 0.0), (WASH - 15, 0.0)]
    t = WASH
    for watts, span in ((200.0, 60), (2000.0, 600), (400.0, 1200)):
        stop = t + span
        while t < stop:
            rows.append((t, watts))
            t += 5
    end = t
    while t < end + 7200:
        rows.append((t, 0.3))
        t += 30
    return [(_at(s), p) for s, p in rows], end


def _device(end: float, stored: bool = True) -> object:
    cycles = []
    if stored:
        cycles = [{"id": "wash", "start_time": _at(WASH).isoformat(),
                   "end_time": _at(end).isoformat(), "duration": end - WASH,
                   "status": "completed"}]
    doc = {"data": {"past_cycles": cycles}, "entry_data": {}, "entry_options": dict(OPTIONS)}
    return sge.load_device(doc, None)


def test_onset_chains_probes_and_stops_at_a_quiet_gap():
    readings = [(_at(s), p) for s, p in (
        (0, 50.0), (200, 0.0), (300, 10.0), (330, 10.0), (360, 0.0), (420, 300.0), (425, 300.0),
    )]
    # 300 -> 420 is one chain (gaps <= 90 s); the 0 s blip is 300 s before it.
    assert sge.onset_before(readings, _at(420), 5.0) == _at(300)
    # A stored start a few seconds before its first reading anchors on that reading.
    assert sge.onset_before(readings, _at(417), 5.0) == _at(300)
    # The lookback bounds the walk.
    assert sge.onset_before(readings, _at(420), 5.0, lookback_s=100) == _at(330)


def test_baseline_and_variants_on_a_synthetic_day():
    readings, end = _day()
    out = sge.evaluate(
        readings, _device(end),
        sweeps=["start_duration_threshold=0.5", "start_energy_threshold=1000",
                "curve_preroll_seconds=300"],
    )
    assert out["truth"] == "stored"
    v = out["variants"]

    base = v["baseline"]["summary"]
    assert base["references"] == 1 and base["missed"] == 0
    # The door-lock probe aborted, so the run starts at the wash; the onset is the probe.
    assert base["late_start_s"]["median"] == pytest.approx(50.0)
    assert base["fidelity_abs_s"]["max"] == pytest.approx(0.0)
    assert base["cycle_probes"] == 1
    # The blip is a probe in idle, not a phantom.
    assert base["phantoms"] == 0 and base["idle_probes"] == 1
    assert base["idle_days"] > 0

    # Looser duration gate: the 1 s blip carries 0.4 Wh and now commits.
    loose = v["start_duration_threshold=0.5"]["summary"]
    assert loose["phantoms"] == 1 and loose["missed"] == 0
    assert loose["phantom_starts"] == [_at(7200).isoformat()]

    # An energy gate the wash can never fill misses it.
    assert v["start_energy_threshold=1000"]["summary"]["missed"] == 1

    # The #430 pre-roll carries the aborted probe into the curve: no late start.
    pre = v["curve_preroll_seconds=300"]["summary"]
    assert pre["late_start_s"]["median"] == pytest.approx(0.0)
    assert pre["fidelity_abs_s"]["max"] == pytest.approx(50.0)


def test_blocks_truth_without_stored_cycles():
    readings, end = _day()
    out = sge.evaluate(readings, _device(end, stored=False))
    assert out["truth"] == "blocks"
    base = out["variants"]["baseline"]["summary"]
    # The blip block is too short to be a cycle; the wash block is the one reference.
    assert base["references"] == 1 and base["missed"] == 0 and base["phantoms"] == 0


def test_the_sampling_throttle_drops_a_start_reading():
    """The manager's throttle, emulated: a reading inside sampling_interval is lost.

    Found on a real washer (sampling 40.3 s): a 5 W reading 20 s after the last
    processed one was dropped, so the start was recorded 20 s late, as live did.
    """
    doc = {"data": {"past_cycles": []}, "entry_data": {},
           "entry_options": {**OPTIONS, "sampling_interval": 40.0}}
    device = sge.load_device(doc, None)
    config, manager = sge.detector_setup(device, {})
    rows = [(0, 3.0), (20, 0.0), (40, 5.0), (60.5, 61.0), (83, 48.0)]
    rows += [(83 + 20 * k, 1500.0) for k in range(1, 30)]
    result = sge.replay([(_at(s), p) for s, p in rows], config, manager)
    assert result.runs and result.runs[0].start == _at(60.5)
    # 0 W after 3 W (>= min_power) is a genuine drop and bypassed the throttle at
    # 20 s, so the 5 W reading 20 s later is the one dropped.
    assert result.readings_processed < len(rows)


def test_loaders(tmp_path):
    readings, _end = _day()
    # Home Assistant's history CSV, with an outage row.
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["entity_id", "state", "last_changed"])
    for ts, p in readings[:5]:
        writer.writerow(["sensor.washer_power", p, ts.isoformat()])
    writer.writerow(["sensor.washer_power", "unavailable", _at(200).isoformat()])
    parsed = sge.readings_from_csv(buf.getvalue(), "sensor.washer_power")
    assert [p for _t, p in parsed] == [0.3, 0.3, 0.3, 0.3, 0.3, None]

    # A diagnostics dump's live trace.
    dump = {"data": {"live_diagnostics": {"power_trace": [
        [_at(30).isoformat(), 2.0], [_at(0).isoformat(), 1.0], ["junk"]]}}}
    assert sge.readings_from_dump(dump) == [(_at(0), 1.0), (_at(30), 2.0)]

    # The recorder's tables, read without a lock.
    db = tmp_path / "home-assistant_v2.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE states_meta (metadata_id INTEGER, entity_id TEXT)")
    con.execute("CREATE TABLE states (state TEXT, last_updated_ts REAL, metadata_id INTEGER)")
    con.executemany("INSERT INTO states_meta VALUES (?, ?)", [(1, "sensor.washer_power"), (2, "x")])
    con.executemany("INSERT INTO states VALUES (?, ?, ?)", [
        ("12.5", _at(10).timestamp(), 1), ("unknown", _at(20).timestamp(), 1),
        ("99", _at(15).timestamp(), 2), ("0", _at(5).timestamp(), 1)])
    con.commit()
    con.close()
    rec = sge.readings_from_recorder(str(db), "sensor.washer_power")
    assert rec == [(_at(5), 0.0), (_at(10), 12.5), (_at(20), None)]


def test_cli_runs_on_a_dump(tmp_path, capsys):
    readings, end = _day()
    dump = {
        "data": {
            "entry": {"data": {"device_type": "washing_machine"}, "options": dict(OPTIONS)},
            "store_export": {"data": {"past_cycles": [{
                "id": "wash", "start_time": _at(WASH).isoformat(),
                "end_time": _at(end).isoformat(), "duration": end - WASH}]}},
            "live_diagnostics": {"power_trace": [[t.isoformat(), p] for t, p in readings]},
        }
    }
    path = tmp_path / "config_entry-ha_washdata-test.json"
    path.write_text(json.dumps(dump), encoding="utf-8")
    out_json = tmp_path / "out.json"
    assert sge.main(["--history", str(path), "--sweep", "start_energy_threshold=0.05,0.5",
                     "--json", str(out_json)]) == 0
    printed = capsys.readouterr().out
    assert "baseline" in printed and "start_energy_threshold=0.5" in printed
    result = json.loads(out_json.read_text(encoding="utf-8"))
    assert result["variants"]["baseline"]["summary"]["references"] == 1


def test_gz_recording_csv_merge_and_clip(tmp_path):
    """The issue-download shapes: gzipped files, #43's recording CSV, two downloads of one plug."""
    import gzip

    rec = "minutes_from_start,watts,timestamp_utc\n0.0,0.0,2026-03-18T10:00:00\n0.3,12.5,2026-03-18T10:00:20\n"
    path = tmp_path / "dishwasher_power.csv.gz"
    path.write_bytes(gzip.compress(rec.encode()))
    readings, doc = sge.load_history(str(path), None)
    stamp = datetime(2026, 3, 18, 10, 0, tzinfo=timezone.utc)
    assert doc is None
    assert readings == [(stamp, 0.0), (stamp + timedelta(seconds=20), 12.5)]

    dump = {"data": {"live_diagnostics": {"power_trace": [[_at(0).isoformat(), 1.0]]}}}
    gz = tmp_path / "config_entry.json.gz"
    gz.write_bytes(gzip.compress(json.dumps(dump).encode()))
    assert sge.load_history(str(gz), None)[0] == [(_at(0), 1.0)]
    assert sge.load_config(gz) == dump

    first = [(_at(0), 0.0), (_at(10), 5.0)]
    second = [(_at(10), 5.0), (_at(20), 0.0)]
    merged = sge.merge([first, second])
    assert merged == [(_at(0), 0.0), (_at(10), 5.0), (_at(20), 0.0)]
    assert sge.clip(merged, _at(5), _at(15)) == [(_at(10), 5.0)]


def test_union_truth_extra_and_exclude():
    """A wiped history (#101): stored misses a real run, union adds it from the blocks."""
    readings, end = _day()
    # A second wash later the same day that no stored cycle covers.
    second = end + 7200
    rows = []
    t = second
    for watts, span in ((200.0, 60), (2000.0, 600), (400.0, 1200)):
        stop = t + span
        while t < stop:
            rows.append((_at(t), watts))
            t += 5
    tail = [(_at(t + 30 * k), 0.3) for k in range(240)]
    readings = readings + rows + tail
    device = _device(end)

    stored = sge.evaluate(readings, device, truth="stored")["variants"]["baseline"]["summary"]
    assert stored["references"] == 1 and stored["phantoms"] == 1

    union = sge.evaluate(readings, device, truth="union")
    s = union["variants"]["baseline"]["summary"]
    assert union["truth"] == "union" and s["references"] == 2 and s["phantoms"] == 0

    # exclude drops the stored record; a hand-labelled window replaces what it overlaps.
    out = sge.evaluate(readings, device, truth="stored", exclude=["wash"],
                       extra=[(_at(WASH - 120), _at(end))])
    assert out["truth"] == "stored+manual"
    rows_out = out["variants"]["baseline"]["rows"]
    assert [r["ref"].split("@")[0] for r in rows_out] == ["manual"]


def test_shipped_defaults_drops_only_the_start_gates():
    device = sge.load_device({"data": {"past_cycles": []}, "entry_data": {},
                              "entry_options": {**OPTIONS, "curve_preroll_seconds": 300}}, None)
    shipped = sge.shipped_defaults(device)
    assert not set(sge.START_GATE_KEYS) & set(shipped.options)
    assert shipped.options["min_power"] == 2.0 and shipped.options["stop_threshold_w"] == 2.0
    config, manager = sge.detector_setup(shipped, {})
    assert config.start_threshold_w == pytest.approx(3.0)  # min_power + 1
    assert config.curve_preroll_seconds == 0.0
    assert manager["sampling_interval"] == 2.0  # the washer's device default


def test_manifest_aggregates_per_device_type(tmp_path, capsys):
    import gzip

    readings, end = _day()
    dump = {
        "data": {
            "entry": {"data": {"device_type": "washing_machine"}, "options": dict(OPTIONS)},
            "store_export": {"data": {"past_cycles": [{
                "id": "wash", "start_time": _at(WASH).isoformat(),
                "end_time": _at(end).isoformat(), "duration": end - WASH}]}},
            "live_diagnostics": {"power_trace": [[t.isoformat(), p] for t, p in readings]},
        }
    }
    sub = tmp_path / "issue1"
    sub.mkdir()
    (sub / "config_entry-ha_washdata-ABC.json.gz").write_bytes(gzip.compress(json.dumps(dump).encode()))
    manifest = tmp_path / "sources.jsonl"
    manifest.write_text("\n".join([
        "# comment lines and blank lines are skipped",
        "",
        json.dumps({"label": "a", "history": "issue1/*ABC.json.gz"}),
        json.dumps({"label": "b", "history": str(sub / "config_entry-ha_washdata-ABC.json.gz"),
                    "aggregate": False}),
    ]), encoding="utf-8")
    sources = sge.read_manifest(manifest)
    assert [s.label for s in sources] == ["a", "b"] and sources[1].aggregate is False
    results = [sge.evaluate_source(s, root=tmp_path, sweeps=["start_energy_threshold=1000"])
               for s in sources]
    agg = sge.aggregate(results)
    base = agg["baseline"]["washing_machine"]
    assert base["sources"] == 1 and base["references"] == 1 and base["missed"] == 0
    assert agg["baseline"]["ALL"]["references"] == 1
    assert agg["start_energy_threshold=1000"]["ALL"]["missed"] == 1

    out_json = tmp_path / "out.json"
    assert sge.main(["--manifest", str(manifest), "--root", str(tmp_path), "--shipped-defaults",
                     "--source", "a", "--json", str(out_json)]) == 0
    printed = capsys.readouterr().out
    assert "washing_machine" in printed and "ALL" in printed and "not aggregated" not in printed
    written = json.loads(out_json.read_text(encoding="utf-8"))
    assert [r["label"] for r in written["sources"]] == ["a"]
    assert written["aggregate"]["baseline"]["ALL"]["references"] == 1


def test_manifest_glob_must_match_one_file(tmp_path):
    with pytest.raises(SystemExit):
        sge._resolve("nothing/*.json", tmp_path)
    (tmp_path / "x-1.json").write_text("{}")
    (tmp_path / "x-2.json").write_text("{}")
    with pytest.raises(SystemExit):
        sge._resolve("x-*.json", tmp_path)
    assert sge._resolve("x-1*.json", tmp_path) == str(tmp_path / "x-1.json")
    assert sge._resolve("sqlite:db/home.db", tmp_path) == "sqlite:" + str(tmp_path / "db/home.db")
