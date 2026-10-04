"""devtools/eta_eval.py: the ETA harness's metrics, plumbing and compare mode.

The corpus run is a devtools job (``cycle_data/`` is gitignored); this needs none.
It pins the per-cycle metrics on a hand-built series, then replays a tiny synthetic
washer leave-one-out through the real Playground so the harness cannot rot in CI.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from custom_components.ha_washdata import playground

_PATH = Path(__file__).resolve().parents[1] / "devtools" / "eta_eval.py"
_spec = importlib.util.spec_from_file_location("wd_eta_eval", _PATH)
eta = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = eta
_spec.loader.exec_module(eta)


def _pt(t: float, rem: float | None, prog: str | None) -> dict:
    return {"t": t, "remaining_s": rem, "matched_profile": prog}


def test_truth_is_the_last_reading_above_stop():
    pts = [(0.0, 0.0), (30.0, 900.0), (60.0, 4.0), (90.0, 200.0), (120.0, 1.0), (900.0, 0.5)]
    assert eta.truth_end(pts, 2.0) == 90.0
    assert eta.truth_end([(0.0, 0.0), (30.0, 1.0)], 2.0) == 0.0


def test_cycle_metrics_reads_what_was_shown_at_each_fraction():
    # 1000 s cycle: wrong programme first at 80 s, right one from 300 s, reverted
    # (no ETA shown) from 700 s, back at 880 s.
    series = [
        _pt(0, None, None), _pt(50, None, None),
        _pt(80, 1200, "Quick"), _pt(240, 1000, "Quick"),
        _pt(300, 750, "Long"), _pt(480, 520, "Long"),
        _pt(700, None, None), _pt(880, 100, "Long"), _pt(1100, None, None),
    ]
    m = eta.cycle_metrics(series, 1000.0, "Long")
    assert m["first_eta_s"] == 80 and m["first_eta_program"] == "Quick"
    assert m["first_eta_right"] is False
    assert m["first_eta_err_s"] == pytest.approx(1200 - (1000 - 80))
    at = m["at"]
    # 10 % = 100 s: the 80 s estimate (latest at or before), wrong programme.
    assert at["0.10"]["t"] == 80 and at["0.10"]["right"] is False
    assert at["0.10"]["err_s"] == pytest.approx(280.0)
    assert at["0.10"]["rel"] == pytest.approx(280.0 / 920.0, abs=1e-4)
    # 50 % = 500 s: the 480 s estimate, right programme, 0 error.
    assert at["0.50"]["t"] == 480 and at["0.50"]["right"] is True
    assert at["0.50"]["err_s"] == pytest.approx(0.0)
    # 75 % = 750 s: the latest point (700 s) shows no ETA -> not covered.
    assert at["0.75"] is None
    # 90 % = 900 s: the 880 s estimate, 20 s early -> -20 s.
    assert at["0.90"]["err_s"] == pytest.approx(-20.0)


def test_cycle_metrics_without_any_eta():
    m = eta.cycle_metrics([_pt(0, None, None), _pt(600, None, None)], 900.0, "Long")
    assert m["first_eta_s"] is None and m["first_eta_right"] is None
    assert all(v is None for v in m["at"].values())


def _row(cid: str, dt: str, first: float | None, err10: float | None, right: bool = True) -> dict:
    at = {eta._fkey(f): None for f in eta.FRACS}  # noqa: SLF001
    if err10 is not None:
        at["0.10"] = {"t": 60.0, "err_s": err10, "rel": err10 / 600.0, "right": right}
    return {"export": "x", "id": cid, "device_type": dt, "first_eta_s": first,
            "first_eta_right": right if first is not None else None, "at": at}


def test_summary_counts_never_and_coverage():
    rows = [_row("a", "dishwasher", 120.0, 60.0), _row("b", "dishwasher", 600.0, None, False),
            _row("c", "dishwasher", None, None), _row("d", "washing_machine", 1800.0, None)]
    s = eta.summarise(rows, "dishwasher")
    assert (s["n"], s["never"]) == (3, 1)
    assert s["first_median_min"] == pytest.approx(6.0)   # median of 2 and 10 min
    assert s["first_right_pct"] == pytest.approx(50.0)
    assert s["at"]["0.10"]["cov_pct"] == pytest.approx(33.3)
    assert s["at"]["0.10"]["mae_min"] == pytest.approx(1.0)
    assert eta.summarise(rows)["n"] == 4


# ------------------------------------------------------------- synthetic device

_SHAPES = {   # minutes, watts: two programmes with different silhouettes
    "Quick": [(6, 2000), (16, 150), (3, 500)],
    "Long": [(14, 2200), (30, 200), (6, 650)],
}


def _cycle(name: str, j: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    scale = 1.0 + 0.03 * (j - 1)
    pts, t = [], 0.0
    for minutes, watts in _SHAPES[name]:
        for _ in range(int(minutes * 60 * scale / 30)):
            pts.append([round(t, 1), round(max(0.0, watts * (1 + 0.05 * rng.standard_normal())), 1)])
            t += 30.0
    pts.append([round(t, 1), 0.0])
    return {
        "id": f"{name}-{j}", "profile_name": name, "status": "completed",
        "label_source": "manual", "termination_reason": "timeout",
        "start_time": f"2026-01-{seed:02d}T08:00:00+00:00", "duration": t, "power_data": pts,
    }


def _export() -> dict:
    past = [_cycle(name, j, 3 * k + j + 1) for k, name in enumerate(_SHAPES) for j in range(3)]
    profiles = {name: {"avg_duration": next(c["duration"] for c in past if c["profile_name"] == name),
                       "sample_cycle_id": f"{name}-0"} for name in _SHAPES}
    return {
        "version": 16, "device_fingerprint": {"device_type": "washing_machine"},
        "entry_data": {"device_type": "washing_machine", "name": "Synth"},
        "entry_options": {"min_power": 2.0, "off_delay": 180},
        "data": {"profiles": profiles, "past_cycles": past, "envelopes": {}},
    }


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("eta_corpus")
    (root / "synth").mkdir()
    (root / "synth" / "export.json").write_text(json.dumps(_export()))
    return root


@pytest.fixture(scope="module")
def run_out(corpus, tmp_path_factory) -> tuple[list[dict], Path]:
    out = tmp_path_factory.mktemp("eta_out") / "rows.json"
    assert eta.main(["--corpus", str(corpus), "--json", str(out)]) == 0
    return json.loads(out.read_text()), out


def test_every_cycle_is_replayed_leave_one_out_and_gets_an_eta(run_out):
    rows, _ = run_out
    assert sorted(r["id"] for r in rows) == sorted(f"{n}-{j}" for n in _SHAPES for j in range(3))
    for r in rows:
        assert r["export"] == "cycle_data/synth/export.json"
        assert r["loo"] is True and r["repair"] is True
        assert r["n_other"] == 2       # the fold kept the programme's other two cycles
        assert r["detected_count"] == 1
        # The first ETA follows the commit, inside one estimator step of it.
        assert r["first_commit_s"] is not None and r["first_eta_s"] is not None
        assert r["first_commit_s"] <= r["first_eta_s"] <= r["first_commit_s"] + 60.0
        # Distinct silhouettes: the programme is right from its first ETA on, and the
        # duration-anchored estimate is within a few minutes once it is shown.
        assert r["first_eta_right"] is True, r
        for f in ("0.50", "0.75", "0.90"):
            assert r["at"][f] is not None and r["at"][f]["right"] is True
            assert abs(r["at"][f]["err_s"]) < 600.0, (r["id"], f, r["at"][f])


def test_a_fold_matches_against_a_store_without_the_replayed_cycle(corpus, monkeypatch):
    flags = {"loo": True, "all_formats": False, "shipped_watchdog": False, "repair": True}
    ctx = eta._prepare(corpus / "synth" / "export.json", flags)  # noqa: SLF001
    target = ctx["targets"][0]
    seen: list[dict] = []
    real = eta.EG._production  # noqa: SLF001

    def _spy(doc, data, **kw):
        seen.append(data)
        return real(doc, data, **kw)

    monkeypatch.setattr(eta.EG, "_production", _spy)
    row = eta._replay_one(ctx, target, "k", flags)  # noqa: SLF001
    assert row is not None and len(seen) == 1
    assert all(c is not target for c in seen[0]["past_cycles"])
    assert len(seen[0]["past_cycles"]) == len(ctx["base"]["past_cycles"]) - 1


def test_pre_404_arm_pins_the_commit_flag_and_restores_it():
    from custom_components.ha_washdata.cycle_detector import CycleDetector

    det = CycleDetector.__new__(CycleDetector)
    with eta._pre_404_cadence(True):  # noqa: SLF001
        det.set_match_committed(False)
        assert det._match_committed is True  # noqa: SLF001
    assert "_match_committed" not in CycleDetector.__dict__
    det.set_match_committed(False)
    assert det._match_committed is False  # noqa: SLF001


def test_the_playground_display_caps_are_restored(run_out):
    assert playground.MAX_SERIES_PER_CYCLE == 600
    assert playground.MAX_EVENTS_PER_CYCLE == 300


def test_compare_pairs_runs_and_lists_moved_first_etas(run_out, tmp_path, capsys):
    rows, before = run_out
    moved = json.loads(json.dumps(rows))
    moved[0]["first_eta_s"] += 300.0
    moved[1]["first_eta_s"] = None
    after = tmp_path / "after.json"
    after.write_text(json.dumps(moved))
    capsys.readouterr()
    assert eta.main(["--compare", str(before), str(after)]) == 0
    out = capsys.readouterr().out
    assert f"paired cycles: {len(rows)}" in out
    assert "moved by >= 1 min: 2 / 6" in out
    assert "never" in out and rows[0]["id"] in out
