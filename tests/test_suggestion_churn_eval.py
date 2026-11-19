"""devtools/suggestion_churn_eval.py: the legacy arm and a walk on a synthetic device.

The corpus run needs ``cycle_data/`` (local only); this pins the 0.6.0 patch's
clean-up and cadence window and walks a tiny synthetic export through both arms so
the harness cannot rot in CI.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from custom_components.ha_washdata import learning, suggestion_engine

_PATH = Path(__file__).resolve().parents[1] / "devtools" / "suggestion_churn_eval.py"
_spec = importlib.util.spec_from_file_location("wd_suggestion_churn_eval", _PATH)
churn = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = churn
_spec.loader.exec_module(churn)


def test_the_legacy_patch_restores_the_shipped_rules() -> None:
    shipped = (learning.CADENCE_HOLD_REL, learning._SECONDS_SUFFIXES, suggestion_engine.STOP_HOLD_BAND)  # noqa: SLF001
    with churn.legacy_patch(True):
        assert learning._SECONDS_SUFFIXES == ()  # noqa: SLF001
        assert suggestion_engine.STOP_HOLD_BAND == float("inf")
        assert learning.CADENCE_HOLD_REL < shipped[0]
    assert (learning.CADENCE_HOLD_REL, learning._SECONDS_SUFFIXES, suggestion_engine.STOP_HOLD_BAND) == shipped  # noqa: SLF001


def test_the_legacy_cadence_reads_the_newest_200_intervals() -> None:
    assert churn.legacy_cadence([[30.0] * 19]) is None
    p95, median = churn.legacy_cadence([[600.0] * 300, [30.0] * 200])
    assert (p95, median) == (30.0, 30.0)


def test_a_keepalive_moves_the_cadence_clock_but_is_not_timed() -> None:
    """The live cadence model times each real reading against the manager's last
    reading, which a watchdog keepalive advances too (PR #529 review)."""
    real_only = {"power_data": [[0.0, 500.0], [60.0, 500.0], [120.0, 500.0]]}
    assert churn._intervals(real_only) == [60.0, 60.0]  # noqa: SLF001
    kept_alive = {
        "power_data": [[0.0, 500.0], [30.0, 500.0], [60.0, 500.0], [120.0, 500.0]],
        "synthetic_offsets": [30.0],
    }
    assert churn._intervals(kept_alive) == [30.0, 60.0]  # noqa: SLF001


def _export() -> dict:
    """Eight labelled washes on a jittery 30 s plug (a 10-60 s cadence spread)."""
    rng = np.random.default_rng(7)
    past = []
    for j in range(8):
        pts, t = [[0.0, 0.0]], 15.0
        for minutes, watts in ((15, 2000.0), (20, 150.0), (3, 0.0), (20, 150.0), (3, 500.0)):
            for _ in range(minutes * 2):
                w = 0.0 if watts == 0 else max(5.0, watts * (1 + 0.05 * rng.standard_normal()))
                pts.append([round(t, 1), round(w, 1)])
                t += float(rng.choice([10.0, 30.0, 60.0]))
        pts.append([round(t, 1), 0.0])
        past.append({
            "id": f"c{j}", "profile_name": "Wash", "label_source": "manual",
            "status": "completed", "termination_reason": "timeout",
            "start_time": f"2026-01-{j + 1:02d}T08:00:00+00:00",
            "duration": pts[-1][0], "power_data": pts,
        })
    return {
        "version": 16, "entry_data": {"device_type": "washing_machine", "name": "Synth"},
        "entry_options": {"min_power": 2.0, "off_delay": 180},
        "data": {"profiles": {}, "past_cycles": past, "envelopes": {}},
    }


def test_both_arms_walk_a_synthetic_device(tmp_path: Path) -> None:
    (tmp_path / "synth").mkdir()
    (tmp_path / "synth" / "export.json").write_text(json.dumps(_export()))
    shipped = (learning.CADENCE_HOLD_REL, learning._SECONDS_SUFFIXES, suggestion_engine.STOP_HOLD_BAND)  # noqa: SLF001
    rows = churn.run(tmp_path, churn.ARMS, None, jobs=1)
    assert sorted(r["arm"] for r in rows) == ["legacy", "shipped"]
    for row in rows:
        assert row["cycles"] == 8 and row["cycle_ends"] == 8
        assert set(row["applies"]) == set(churn.KEYS)
        for key, history in row["history"].items():
            assert len(history) == row["applies"][key]
    # A walk leaves the shipped rules in place.
    assert (learning.CADENCE_HOLD_REL, learning._SECONDS_SUFFIXES, suggestion_engine.STOP_HOLD_BAND) == shipped  # noqa: SLF001


def test_the_start_threshold_corrective_runs_at_every_cycle_end(tmp_path: Path) -> None:
    """PR #529 review: Start 450 W, Stop 100 W and three 400 W steady loads; the
    shipped corrective pass offers Start 200 W at a cycle end with no cadence (too
    few readings) and no batch (unlabelled). The walk ran only those two passes."""
    past = []
    for j in range(3):
        # 6 intervals a cycle: 18 in all, under the cadence model's 20.
        pts = [[k * 120.0, 400.0 + (k % 3)] for k in range(6)] + [[720.0, 0.0]]
        past.append({
            "id": f"s{j}", "status": "completed", "termination_reason": "smart",
            "start_time": f"2026-02-{j + 1:02d}T08:00:00+00:00",
            "duration": pts[-1][0], "power_data": pts,
        })
    doc = {
        "version": 16, "entry_data": {"device_type": "dryer", "name": "Steady"},
        "entry_options": {"start_threshold_w": 450.0, "stop_threshold_w": 100.0},
        "data": {"profiles": {}, "past_cycles": past, "envelopes": {}},
    }
    (tmp_path / "steady").mkdir()
    (tmp_path / "steady" / "export.json").write_text(json.dumps(doc))
    (row,) = churn.run(tmp_path, ("shipped",), None, jobs=1)
    assert row["cycle_ends"] == 0
    assert row["history"]["start_threshold_w"] == [[450.0, 200.0]]


def _wash(cid: str, day: str, shape: tuple, name: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    pts, t = [[0.0, 0.0]], 15.0
    for minutes, watts in shape:
        for _ in range(minutes * 2):
            pts.append([round(t, 1), round(max(5.0, watts * (1 + 0.05 * rng.standard_normal())), 1)])
            t += 30.0
    pts.append([round(t, 1), 0.0])
    return {
        "id": cid, "profile_name": name, "label_source": "manual", "status": "completed",
        "termination_reason": "timeout", "start_time": f"{day}T08:00:00+00:00",
        "duration": pts[-1][0], "power_data": pts,
    }


def test_each_pass_sees_the_profiles_of_its_history_prefix(tmp_path: Path, monkeypatch) -> None:
    """PR #529 review: the walk read the exported profiles, so a short programme
    first run last capped Profile Match Interval from the first pass. A programme
    now exists from its first stored run, with what the store derives from the runs
    so far; a reference cycle counts from the start only when it was imported
    before the walk."""
    long_shape = ((15, 2000.0), (20, 150.0), (20, 500.0))
    past = [_wash(f"l{j}", f"2026-01-{j + 1:02d}", long_shape, "Long", j) for j in range(6)]
    past.append(_wash("q0", "2026-01-20", ((10, 1500.0),), "Quick", 99))
    early = _wash("r0", "2025-12-01", ((40, 800.0),), "RefEarly", 5)
    late = _wash("r1", "2026-01-10", ((5, 800.0),), "RefLate", 6)
    doc = {
        "version": 16, "entry_data": {"device_type": "washing_machine", "name": "Prefix"},
        "entry_options": {"min_power": 2.0, "off_delay": 180},
        "data": {
            # As exported: every programme known, with its final duration.
            "profiles": {n: {"avg_duration": d} for n, d in (
                ("Long", 3300.0), ("Quick", 600.0), ("RefEarly", 2400.0), ("RefLate", 300.0))},
            "past_cycles": past, "reference_cycles": [early, late], "envelopes": {},
        },
    }
    (tmp_path / "prefix").mkdir()
    (tmp_path / "prefix" / "export.json").write_text(json.dumps(doc))
    seen: list[tuple[int, list[str], float | None]] = []
    real = suggestion_engine.SuggestionEngine._shortest_profile_duration  # noqa: SLF001

    def spy(self):
        out = real(self)
        seen.append((len(self.profile_store.get_past_cycles()),
                     sorted(self.profile_store.get_profiles()), out))
        return out

    monkeypatch.setattr(suggestion_engine.SuggestionEngine, "_shortest_profile_duration", spy)
    churn.run(tmp_path, ("shipped",), None, jobs=1)
    before_quick = [s for s in seen if s[0] < len(past)]
    assert before_quick and all(names == ["Long", "RefEarly"] for _n, names, _s in before_quick)
    # Learned from the runs so far (Long 55 min, RefEarly 40 min), not Quick's 600 s.
    assert all(s is not None and s > 2000.0 for _n, _names, s in before_quick)
    assert all(names == ["Long", "Quick", "RefEarly"] for n, names, _s in seen if n == len(past))
