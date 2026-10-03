"""devtools/suggestion_loop_eval.py: the loop plumbing and the verdicts (audit F10).

The corpus run is the slow test (``test_suggestion_loop_fixed_point.py``); this one
needs no ``cycle_data/``: it pins the classifier, the reading throttle, the legacy
patch's clean-up, and runs the whole loop on a tiny synthetic export so the harness
cannot rot in CI, where the corpus is absent.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.suggestion_engine import SuggestionEngine

_PATH = Path(__file__).resolve().parents[1] / "devtools" / "suggestion_loop_eval.py"
_spec = importlib.util.spec_from_file_location("wd_suggestion_loop_eval", _PATH)
loop = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = loop
_spec.loader.exec_module(loop)


@pytest.mark.parametrize(
    ("values", "changed", "last", "verdict"),
    [
        ([30], [], 0, "stable"),
        ([30, 60], [0], 1, "converged"),
        ([2.0, 1.2, 1.6], [0, 1], 2, "converged"),     # one reversal, then settled
        ([2, 7, 12, 17], [0, 1, 2], 2, "ladder"),       # still climbing at the end
        ([2, 7, 2], [0, 1], 1, "oscillates"),           # back where it started
        ([2.0, 3.0, 2.5], [0, 1], 1, "oscillates"),     # direction flipped, still moving
        ([2, 7, 12], [0, 1], 1, "unsettled"),           # too few applies to call
        ([30, 30.0000001], [0], 1, "stable"),           # float noise is not a change
    ],
)
def test_classify(values, changed, last, verdict):
    assert loop.classify(values, changed, last)["verdict"] == verdict


def test_converged_reports_reversals():
    out = loop.classify([2.0, 1.2, 1.6], [0, 1], 2)
    assert (out["applies"], out["reversals"], out["values"]) == (2, 1, [2.0, 1.2, 1.6])


def test_throttle_drops_fast_high_readings_and_keeps_every_low_one():
    pts = [(0.0, 100.0), (3.0, 100.0), (6.0, 0.5), (7.0, 100.0), (12.0, 100.0), (14.0, 0.2)]
    assert loop._throttle(pts, 5.0, 2.0) == [  # noqa: SLF001
        (0.0, 100.0), (6.0, 0.5), (12.0, 100.0), (14.0, 0.2)
    ]


def test_post_cycle_level_reads_only_a_kept_tail():
    work = [(float(t), 900.0) for t in range(0, 600, 30)]
    # A Smart-Termination tail: the appliance still drawing 0.8 W after the end.
    kept = work + [(600.0 + 10 * i, 0.8) for i in range(12)]
    assert loop.post_cycle_level(kept, recorded_stop=1.5) == pytest.approx(0.8)
    # A trimmed trace ends on its last active reading: nothing to read.
    assert loop.post_cycle_level(work, recorded_stop=1.5) is None
    # A tail at 0 W, or one too short to be a level, says nothing either.
    assert loop.post_cycle_level(work + [(600.0 + 10 * i, 0.0) for i in range(12)], 1.5) is None
    assert loop.post_cycle_level(work + [(600.0, 0.8), (610.0, 0.8)], 1.5) is None


def test_legacy_patch_restores_the_shipped_engine():
    gen, keys = SuggestionEngine.generate_detection_suggestions, ws_api._SUGGESTION_KEYS  # noqa: SLF001
    with loop.legacy_patch(("sampling_interval", "confidence")):
        assert "sampling_interval" in ws_api._SUGGESTION_KEYS  # noqa: SLF001
        assert SuggestionEngine.generate_detection_suggestions is not gen
    assert SuggestionEngine.generate_detection_suggestions is gen
    assert ws_api._SUGGESTION_KEYS == keys  # noqa: SLF001


# ------------------------------------------------------------- synthetic device

def _trace(seed: int, minutes_heat: int, minutes_wash: int) -> list[list[float]]:
    """Heat, wash with a 3-minute soak pause, spin, off; 30 s plug with jitter."""
    rng = np.random.default_rng(seed)
    phases = [(minutes_heat, 2000.0), (minutes_wash // 2, 150.0), (3, 0.0),
              (minutes_wash // 2, 150.0), (3, 500.0)]
    pts, t = [[0.0, 0.0]], 15.0
    for minutes, watts in phases:
        for _ in range(minutes * 2):
            w = 0.0 if watts == 0 else max(5.0, watts * (1 + 0.05 * rng.standard_normal()))
            pts.append([round(t, 1), round(w, 1)])
            t += 30.0 + float(rng.uniform(-2, 2))
    pts.append([round(t, 1), 0.0])
    return pts


def _export() -> dict:
    past = []
    for j in range(8):
        name, heat, wash = ("Quick", 8, 20) if j % 2 else ("Long", 15, 40)
        pts = _trace(j, heat, wash)
        past.append({
            "id": f"c{j}", "profile_name": name, "label_source": "manual",
            "status": "completed", "termination_reason": "timeout",
            "start_time": f"2026-01-{j + 1:02d}T08:00:00+00:00",
            "end_time": f"2026-01-{j + 1:02d}T09:30:00+00:00",
            "duration": pts[-1][0], "power_data": pts, "match_confidence": 0.8,
        })
    # Labelled cycles but no profiles: the replay skips the matcher (live does the
    # same without real profiles), which keeps this under the fast tier's budget.
    # The matched path is the slow corpus test's job.
    return {
        "version": 16, "entry_data": {"device_type": "washing_machine", "name": "Synth"},
        "entry_options": {"min_power": 2.0, "off_delay": 180},
        "data": {"profiles": {}, "past_cycles": past, "envelopes": {}},
    }


@pytest.fixture(scope="module")
def synth_results(tmp_path_factory):
    root = tmp_path_factory.mktemp("loop_corpus")
    (root / "synth").mkdir()
    (root / "synth" / "export.json").write_text(json.dumps(_export()))
    return loop.run(root, rounds=3, jobs=1)


def test_loop_runs_end_to_end_on_a_synthetic_device(synth_results):
    assert [r["device"] for r in synth_results] == ["synth/export.json"]
    res = synth_results[0]
    first = res["rounds"][0]
    # Every stored cycle went through the real detector, none was lost or split.
    assert first["replay"]["replayed"] == first["replay"]["stored"] == 8
    assert first["replay"]["splits"] == first["replay"]["lost"] == 0
    # The cadence model was fed (>= 20 committed intervals at ~30 s).
    assert first["cadence"] is not None and 25 < first["cadence"][1] < 35
    # The loop applied something and then stopped at a fixed point.
    assert first["applied"]
    assert res["fixed_point"] is True
    assert loop.failures(synth_results) == []


def test_a_muted_key_is_never_applied(synth_results):
    probe = synth_results[0]["lock_probe"]
    assert probe is not None and probe["locked"]
    assert probe["applied"] == []


def test_parallel_replays_give_the_serial_result(synth_results, tmp_path):
    (tmp_path / "synth").mkdir()
    (tmp_path / "synth" / "export.json").write_text(json.dumps(_export()))
    par = loop.run(tmp_path, rounds=3, jobs=2)

    def _strip(rs):
        return json.dumps([
            {**{k: v for k, v in r.items() if k not in ("cpu_s", "rounds")},
             "rounds": [{k: v for k, v in row.items() if k not in ("cpu_s", "memo_hit", "memo_miss")}
                        for row in r["rounds"]]}
            for r in rs
        ], default=str)

    assert _strip(par) == _strip(synth_results)


def test_a_run_leaves_the_process_as_it_found_it(synth_results):
    import logging

    from custom_components.ha_washdata import analysis

    # The fixture ran first: no CRITICAL logger and no memo left behind for the
    # tests that follow (test_ws_contract reads a debug log through caplog).
    assert logging.getLogger("custom_components.ha_washdata").level != logging.CRITICAL
    assert not getattr(analysis.compute_matches_worker, "_loop_eval_memo", False)
