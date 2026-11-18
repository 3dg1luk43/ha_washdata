"""devtools/program_mismatch_eval.py (#530) on a synthetic export.

The corpus run needs ``cycle_data/`` (local only); this replays a tiny two-programme
export through the harness so it cannot rot in CI: the rows it records, the right
pick never firing, a wrong pick firing, and the sweep.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "devtools" / "program_mismatch_eval.py"
_spec = importlib.util.spec_from_file_location("wd_program_mismatch_eval", _PATH)
pme = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pme
_spec.loader.exec_module(pme)

SHAPES = {
    "Cotton": ((15, 2000.0), (25, 200.0)),
    "Synthetics": ((15, 200.0), (15, 1000.0), (10, 200.0)),
}


def _export() -> dict:
    rng = np.random.default_rng(530)
    past, profiles = [], {}
    for p, (name, shape) in enumerate(SHAPES.items()):
        for j in range(4):
            pts, t = [], 0.0
            for minutes, watts in shape:
                for _ in range(minutes * 2):
                    pts.append([t, round(watts * (1 + 0.03 * rng.standard_normal()), 1)])
                    t += 30.0
            pts.append([t, 0.0])
            cid = f"{name}-{j}"
            past.append({
                "id": cid, "profile_name": name, "label_source": "manual",
                "status": "completed", "start_time": f"2026-01-{2 * j + p + 1:02d}T08:00:00+00:00",
                "duration": t, "power_data": pts,
            })
            profiles.setdefault(name, {"avg_duration": t, "sample_cycle_id": cid})
    return {
        "version": 16, "entry_data": {"device_type": "washing_machine", "name": "Synth"},
        "entry_options": {"min_power": 2.0, "off_delay": 180},
        "data": {"profiles": profiles, "past_cycles": past, "envelopes": {}},
    }


def test_a_synthetic_device_replays_and_scores(tmp_path: Path) -> None:
    (tmp_path / "synth").mkdir()
    (tmp_path / "synth" / "export.json").write_text(json.dumps(_export()))
    rows = pme.replay(tmp_path, jobs=1, log=lambda *a, **k: None)
    assert len(rows) == 8
    for row in rows:
        assert set(row["durations"]) == set(SHAPES)
        assert row["interval"] == 300.0
        ts = [t["t"] for t in row["ticks"]]
        assert ts == sorted(set(ts)) and all(t < row["duration"] for t in ts)
        assert all(t["best"] in (*SHAPES, None) for t in row["ticks"])
        assert sum(t["best"] == row["label"] for t in row["ticks"]) >= len(row["ticks"]) // 2

    prep = pme.prepare(rows)
    summary = pme.score(prep, *pme.SHIPPED)
    assert summary["all"]["cycles"] == 8
    assert summary["all"]["false_alarm"] == 0
    assert summary["all"]["pairs"] == 8
    # 4 of 8 fire strictly before the end; two more fired on the check AT the
    # recorded end, which the manager never runs (the cycle is over by then).
    assert summary["all"]["detect"] >= 4
    assert 0.0 < summary["all"]["elapsed_frac"] < 1.0
    assert set(summary) == {"all", "own history", "washing_machine"}

    grid = pme.sweep(prep)
    assert len(grid) == len(pme.SWEEP_MARGINS) * len(pme.SWEEP_PERSISTENCE) * len(pme.SWEEP_FRACTIONS)
    assert all(g["false_alarm_rate"] == 0.0 for g in grid)


def test_the_rows_round_trip_through_the_json_file(tmp_path: Path) -> None:
    rows = [{
        "path": "x/y.json", "id": "c1", "dev": "dryer", "label": "A", "duration": 1200.0,
        "interval": 300.0, "durations": {"A": 1200.0, "B": 1200.0}, "families": {},
        "twins": [], "unmatchable": [],
        "ticks": [
            {"t": 300.0 * k, "best": "B",
             "c": [{"name": "B", "score": 0.9}, {"name": "A", "score": 0.5}]}
            for k in range(1, 5)
        ],
    }]
    out = tmp_path / "rows.json"
    out.write_text(json.dumps({"rows": rows}))
    assert pme.main(["--from-rows", str(out), "--json", str(tmp_path / "o.json")]) == 0
    doc = json.loads((tmp_path / "o.json").read_text())
    # B leads A by 0.4 from the first tick: the right pick (A) is a false alarm
    # once enough ticks and fraction have passed, B as the pick never fires.
    fires = pme.fire_at(pme.pick_sequence(rows[0], "A"), 0.18, 3, 0.5)
    assert fires == 900.0
    assert pme.fire_at(pme.pick_sequence(rows[0], "B"), 0.18, 1, 0.0) is None
    assert doc["summary"]["all"]["cycles"] == 1
    # Only a fire strictly before the recorded end (1200 s) is one the user sees.
    assert pme.fire_at(pme.pick_sequence(rows[0], "A"), 0.18, 3, 0.5, end=900.0) is None
    assert pme.score(pme.prepare(rows), 0.18, 4, 0.5)["all"]["false_alarm"] == 0
    assert pme.score(pme.prepare(rows), 0.18, 3, 0.5)["all"]["false_alarm"] == 1


def test_every_other_programme_is_a_wrong_pick() -> None:
    """A programme without a learned duration or a template can be picked too; the
    manager never checks it, so it is a miss, not a pair left out."""
    row = {
        "path": "x/y.json", "id": "c1", "dev": "washing_machine", "label": "A",
        "duration": 1800.0, "interval": 300.0, "durations": {"A": 1800.0, "B": 1700.0, "U": 1750.0},
        "programs": ["A", "B", "C", "U"], "families": {}, "twins": [], "unmatchable": ["U"],
        "ticks": [
            {"t": 300.0 * k, "best": "A",
             "c": [{"name": "A", "score": 0.9}, {"name": "B", "score": 0.4}]}
            for k in range(1, 6)
        ],
    }
    assert pme._wrong_picks(row) == ["B", "C", "U"]
    assert pme.pick_sequence(row, "C") == [] and pme.pick_sequence(row, "U") == []
    s = pme.score(pme.prepare([row]), 0.18, 3, 0.5)["all"]
    assert (s["pairs"], s["detect"]) == (3, 1)
    # The nearest by duration ratio is a pick with a duration (here U, 1750 s).
    assert pme._nearest_wrong(row) == "U" and s["nearest_detect"] == 0


def test_a_check_needs_a_new_reading(tmp_path: Path) -> None:
    """The manager checks on the first reading at least one interval after the last
    check: a plug silent from 5 to 25 min runs no check in between, where a replay
    every interval fed the persistence rule one unchanged prefix four times."""
    dense = [[30.0 * k, 100.0] for k in range(91)]           # 0 .. 2700 s
    sparse = [p for p in dense if p[0] <= 300 or p[0] >= 1500]
    assert [t for _q, t in pme.check_prefixes(dense, 300.0, 2700.0)] == [
        300.0 * k for k in range(1, 9)]                       # never AT the end
    checks = pme.check_prefixes(sparse, 300.0, 2700.0)
    assert [t for _q, t in checks] == [300.0, 1500.0, 1800.0, 2100.0, 2400.0]
    assert [len(q) for q, _t in checks] == [11, 12, 22, 32, 42]

    doc = _export()
    for cyc in doc["data"]["past_cycles"]:
        cyc["power_data"] = [p for p in cyc["power_data"] if p[0] <= 300 or p[0] >= 1500]
    (tmp_path / "sparse").mkdir()
    (tmp_path / "sparse" / "export.json").write_text(json.dumps(doc))
    rows = pme.replay(tmp_path, jobs=1, log=lambda *a, **k: None)
    assert rows
    for row in rows:
        ts = [t["t"] for t in row["ticks"]]
        assert ts == [300.0, 1500.0, 1800.0, 2100.0], ts


def test_the_device_cache_follows_the_corpus_root(tmp_path: Path) -> None:
    """Two in-process replays of different corpora must not share a device."""
    for root, n in (("one", 4), ("two", 2)):
        doc = _export()
        doc["data"]["past_cycles"] = [c for c in doc["data"]["past_cycles"]
                                      if int(c["id"].split("-")[1]) < n]
        (tmp_path / root / "synth").mkdir(parents=True)
        (tmp_path / root / "synth" / "export.json").write_text(json.dumps(doc))
    pme._worker_init(str(tmp_path / "one"))
    first = pme._device("synth/export.json")
    pme._worker_init(str(tmp_path / "two"))
    second = pme._device("synth/export.json")
    assert second is not first
    assert len(second.cycles) == 4 and len(first.cycles) == 8
