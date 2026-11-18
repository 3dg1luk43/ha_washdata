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
        assert [t["t"] for t in row["ticks"]] == sorted(t["t"] for t in row["ticks"])
        assert all(t["best"] in (*SHAPES, None) for t in row["ticks"])
        assert sum(t["best"] == row["label"] for t in row["ticks"]) >= len(row["ticks"]) // 2

    prep = pme.prepare(rows)
    summary = pme.score(prep, *pme.SHIPPED)
    assert summary["all"]["cycles"] == 8
    assert summary["all"]["false_alarm"] == 0
    assert summary["all"]["pairs"] == 8
    assert summary["all"]["detect"] >= 6
    assert 0.0 < summary["all"]["elapsed_frac"] <= 1.0
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
