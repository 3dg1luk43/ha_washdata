"""devtools/program_mismatch_eval.py (#530) on a synthetic export.

The corpus run needs ``cycle_data/`` (local only); this replays a tiny two-programme
export through the harness so it cannot rot in CI: the rows it records, the right
pick never firing, a wrong pick firing, and the sweep.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

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
            {"t": 300.0 * k, "at": 300.0 * k, "best": "B",
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
            {"t": 300.0 * k, "at": 300.0 * k, "best": "A",
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
    checks = list(pme.check_prefixes(sparse, 300.0, 2700.0))
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


def _write(root: Path, doc: dict | str) -> str:
    (root / "synth").mkdir(parents=True, exist_ok=True)
    (root / "synth" / "export.json").write_text(doc if isinstance(doc, str) else json.dumps(doc))
    return "synth/export.json"


def test_the_checks_come_one_at_a_time() -> None:
    dense = [[30.0 * k, 100.0] for k in range(91)]
    checks = pme.check_prefixes(dense, 300.0, 2700.0)
    assert iter(checks) is checks
    prefix, elapsed = next(checks)
    assert (len(prefix), elapsed) == (11, 300.0)


def _behind(ticks: list[tuple[float, float]], duration: float) -> dict:
    """A row whose label A trails B by 0.4 on every tick ``(t, at)``."""
    return {
        "path": "x/y.json", "id": "c1", "dev": "dryer", "label": "A", "duration": duration,
        "interval": 300.0, "durations": {"A": 1800.0, "B": 1800.0}, "families": {},
        "twins": [], "unmatchable": [],
        "ticks": [{"t": t, "at": at, "best": "B",
                   "c": [{"name": "B", "score": 0.9}, {"name": "A", "score": 0.5}]}
                  for t, at in ticks],
    }


def test_the_end_check_reads_the_stored_offset() -> None:
    """The recorded duration counts from the cycle start, so a trace whose first
    reading is 60 s in ends at the same offset: its reading at 1860 s is past an
    1820 s end though only 1800 s after the first reading. The cadence and the
    rule still read the time since the first reading."""
    pts = [[60.0 + 30.0 * k, 100.0] for k in range(61)]   # 60 .. 1860 s
    checks = list(pme.check_prefixes(pts, 300.0, 1820.0))
    assert [t for _q, t in checks] == [300.0, 600.0, 900.0, 1200.0, 1500.0]
    assert [q[-1][0] for q, _t in checks] == [360.0, 660.0, 960.0, 1260.0, 1560.0]
    # The scoring reads each tick's stored offset too, and reports the fire at it.
    row = _behind([(300.0 * k, 300.0 * k + 60.0) for k in range(1, 7)], 1820.0)
    seq = pme.pick_sequence(row, "A")
    assert pme.fire_at(seq, 0.18, 6, 0.0) == 1860.0
    assert pme.fire_at(seq, 0.18, 6, 0.0, end=row["duration"]) is None
    assert pme.fire_at(seq, 0.18, 5, 0.0, end=row["duration"]) == 1560.0
    # A tick exactly at the end is past it.
    assert pme.fire_at(seq, 0.18, 5, 0.0, end=1560.0) is None


def test_the_reported_fraction_reads_the_cycle_start_clock() -> None:
    """A 3600 s cycle whose first reading is 600 s in: the wrong pick fires on the
    check 1800 s after that reading, 2400 s into the cycle, i.e. two thirds of it
    (read against the first reading it was reported as half). The rule itself still
    reads the 1800 s."""
    row = _behind([(300.0 * k, 300.0 * k + 600.0) for k in range(1, 8)], 3600.0)
    for tick in row["ticks"]:  # A, the label, leads: B is the wrong pick that fires
        tick["best"], tick["c"] = "A", [{"name": "A", "score": 0.9}, {"name": "B", "score": 0.5}]
    s = pme.score(pme.prepare([row]), 0.18, 6, 0.5)["all"]
    assert (s["false_alarm"], s["detect"], s["nearest_detect"]) == (0, 1, 1)
    assert s["elapsed_frac"] == s["nearest_elapsed_frac"] == round(2400.0 / 3600.0, 3)


def test_rows_without_the_stored_offsets_are_refused(tmp_path: Path, capsys) -> None:
    """A rows file written before the ticks kept their offset holds only ``t``, the
    seconds since the first reading. Read as the offset it can put a tick of a
    trace that starts late past the cycle end inside it: such a file is refused,
    to be matched again."""
    row = _behind([(300.0 * k, 300.0 * k + 600.0) for k in range(1, 12)], 3600.0)
    for tick in row["ticks"]:
        del tick["at"]
    out = tmp_path / "rows.json"
    out.write_text(json.dumps({"rows": [row]}))
    assert pme.main(["--from-rows", str(out)]) == 3
    assert "offset" in capsys.readouterr().err
    with pytest.raises(KeyError):
        pme.pick_sequence(row, "A")


def test_the_saved_end_keeps_full_precision(tmp_path: Path) -> None:
    """Every check the replay runs, strictly before the recorded end, is one the
    saved rows keep before it: rounded to 0.1 s, a 2400.04 s cycle was saved as
    2400.0 and its check at 2400 s read as past the end."""
    doc = _export()
    for cyc in doc["data"]["past_cycles"]:
        cyc["duration"] += 0.04
    _write(tmp_path, doc)
    rows = pme.replay(tmp_path, jobs=1, log=lambda *a, **k: None)
    assert len(rows) == 8
    for row in rows:
        assert row["duration"] == 2400.04
        assert row["ticks"][-1]["at"] == 2400.0
        assert all(at < row["duration"] for *_rest, at in pme.pick_sequence(row, row["label"]))


@pytest.mark.parametrize("literal", ["1e400", "1" + "0" * 400, '"3600"', "null"],
                         ids=["inf", "int-overflow", "string", "null"])
def test_a_record_with_an_unusable_duration_is_left_out(tmp_path: Path, literal: str) -> None:
    """An imported ``1e400`` parses as ``inf`` (no check would reach its end, and
    the rows file would hold ``Infinity``, which is not JSON); an integer literal
    ``1`` and 400 zeros parses as an int ``float`` cannot hold, and selecting the
    folds raised OverflowError for the whole corpus; a string or a null makes the
    rebuild's ``duration > 60`` raise. The record is left out of the folds and of
    every rebuild, so the rest of the device is replayed, Cotton's other folds
    included."""
    doc = _export()
    doc["data"]["past_cycles"][0]["duration"] = "__bad__"
    path = _write(tmp_path, json.dumps(doc).replace('"__bad__"', literal))
    rows = pme.replay(tmp_path, jobs=1, log=lambda *a, **k: None)
    assert sorted(r["id"] for r in rows) == sorted(
        [f"Cotton-{j}" for j in range(1, 4)] + [f"Synthetics-{j}" for j in range(4)])
    # The worker's copy of the device is the one the selection read.
    pme._worker_init(str(tmp_path))
    assert "Cotton-0" not in {c["id"] for c, _ in pme._device(path).cycles}
    out = tmp_path / "o.json"
    assert pme.main(["--corpus", str(tmp_path), "--jobs", "1", "--json", str(out)]) == 0

    def strict(token: str) -> None:
        raise ValueError(token)

    doc_out = json.loads(out.read_text(), parse_constant=strict)
    assert len(doc_out["rows"]) == len(rows)


def test_a_fold_never_sees_another_folds_rebuild(tmp_path: Path, monkeypatch) -> None:
    """A rebuild cuts imported dishwasher cycles in place (``meta.tail_cut_s``, #518),
    stood in for here by one that cuts every cycle of the programme it rebuilds:
    the loaded cycles stay as they were, and a fold's row is the same whichever
    folds ran before it (and so whichever worker ran it)."""
    from custom_components.ha_washdata.profile_store import ProfileStore

    real = ProfileStore.async_rebuild_envelope

    async def cutting(self, name):
        for key in ("past_cycles", "reference_cycles", "backfill_cycles"):
            for c in self._data.get(key) or []:
                if c.get("profile_name") != name or (c.get("meta") or {}).get("tail_cut_s"):
                    continue
                keep = c["power_data"][: len(c["power_data"]) * 3 // 4]
                c.setdefault("meta", {})["tail_cut_s"] = c["duration"] - keep[-1][0]
                c["power_data"], c["duration"] = keep, keep[-1][0]
        return await real(self, name)

    monkeypatch.setattr(ProfileStore, "async_rebuild_envelope", cutting)
    path = _write(tmp_path, _export())
    pme._worker_init(str(tmp_path))
    dev = pme._device(path)
    loaded = copy.deepcopy([c for c, _ in dev.cycles])
    _, base = pme._job_base(path)
    idxs = pme.ev.scorable(dev)
    assert len(idxs) == 8
    together = pme._job_folds((path, base, idxs))
    alone = [row for i in reversed(idxs) for row in pme._job_folds((path, base, [i]))]
    assert [c for c, _ in dev.cycles] == loaded
    assert len(together) == 8
    assert sorted(alone, key=lambda r: r["id"]) == sorted(together, key=lambda r: r["id"])
    # Each held-out cycle is replayed as recorded, uncut.
    assert {r["duration"] for r in together} == {2400.0}


@pytest.mark.parametrize("duration", [float("nan"), -5.0, 0.0])
def test_a_recorded_duration_must_be_positive_and_finite(duration: float) -> None:
    pts = [[0.0, 1.0], [60.0, 1.0]]
    assert pme.recorded_duration({"duration": duration}, pts) in (None, 60.0)
    assert pme.recorded_duration({"duration": float("inf")}, pts) is None
    assert pme.recorded_duration({"duration": 10 ** 400}, pts) is None
    assert pme.recorded_duration({"duration": 1234.56}, pts) == 1234.56


def test_a_missing_duration_ends_at_the_final_offset() -> None:
    """Without a recorded duration the end is the trace's final offset, on the
    cycle-start clock the end check reads. The span from the first reading ended a
    trace that starts 600 s in 600 s early, so its last two checks never ran."""
    pts = [[600.0 + 30.0 * k, 100.0] for k in range(101)]   # 600 .. 3600 s
    for missing in ({}, {"duration": 0}, {"duration": None}, {"duration": -5.0}):
        assert pme.recorded_duration(missing, pts) == 3600.0
    assert pme.recorded_duration({"duration": 3000.0}, pts) == 3000.0
    end = pme.recorded_duration({}, pts)
    assert [t for _q, t in pme.check_prefixes(pts, 300.0, end)][-1] == 2700.0


def test_a_fold_never_reads_the_held_out_cycles_duration(tmp_path: Path) -> None:
    """The base profiles are built from every cycle. When a fold's rebuild learns no
    duration for the held-out programme (here its other cycles were interrupted,
    which no rebuild reads), the base's, learned from the held-out cycle itself, set
    the fraction gate: that fold is left out. The folds trained on it keep theirs."""
    doc = _export()
    for cyc in doc["data"]["past_cycles"]:
        if cyc["profile_name"] == "Cotton" and cyc["id"] != "Cotton-0":
            cyc["status"] = "interrupted"
    _write(tmp_path, doc)
    rows = pme.replay(tmp_path, jobs=1, log=lambda *a, **k: None)
    ids = {r["id"] for r in rows}
    assert "Cotton-0" not in ids
    assert {f"Cotton-{j}" for j in range(1, 4)} <= ids
    assert all(r["durations"]["Cotton"] == 2400.0 for r in rows if r["label"] == "Cotton")
