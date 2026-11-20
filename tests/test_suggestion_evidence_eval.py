"""devtools/suggestion_evidence_eval.py: its 0.6.0 standby rule, patches and a run.

The corpus run needs ``cycle_data/`` (local only); this keeps the harness from
rotting in CI.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from custom_components.ha_washdata import suggestion_engine as se

_PATH = Path(__file__).resolve().parents[1] / "devtools" / "suggestion_evidence_eval.py"
_spec = importlib.util.spec_from_file_location("wd_suggestion_evidence_eval", _PATH)
evidence = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = evidence
_spec.loader.exec_module(evidence)


def _cycle(i: int, last: float, reason: str = "user", label: str = "Wash") -> dict:
    pts = [[10.0 * k, 2000.0 if 20 <= k < 80 else 60.0] for k in range(120)] + [[1200.0, last]]
    return {
        "id": f"c{i}", "status": "completed", "termination_reason": reason, "profile_name": label,
        "start_time": f"2026-01-{i + 1:02d}T08:00:00+00:00", "duration": 1200.0,
        "power_data": pts, "signature": {"max_power": 2000.0},
    }


def test_the_legacy_rule_needs_every_last_sample_above_the_stop() -> None:
    assert evidence.legacy_standby([_cycle(i, 3.4) for i in range(4)], 2.5) == 3.4
    assert evidence.legacy_standby([_cycle(0, 0.0)] + [_cycle(i, 3.4) for i in range(1, 4)], 2.5) is None
    assert evidence.legacy_standby([_cycle(0, 3.4)], 2.5) is None  # fewer than two


def test_a_non_finite_last_sample_is_no_reading() -> None:
    """PR #529 review: two "NaN" last samples gave a NaN level, counted as flagged."""
    assert evidence.legacy_standby([_cycle(i, "NaN") for i in range(2)], 2.5) is None
    assert evidence.legacy_standby([_cycle(0, "inf"), _cycle(1, 3.4)], 2.5) is None
    assert evidence.legacy_standby(
        [_cycle(0, float("nan"))] + [_cycle(i, 3.4) for i in range(1, 3)], 2.5
    ) == 3.4


def test_a_run_reports_every_arm_and_restores_the_constants(tmp_path: Path) -> None:
    export = {
        "version": 16, "entry_data": {"device_type": "washing_machine", "name": "Synth"},
        "entry_options": {"stop_threshold_w": 2.5},
        "data": {"profiles": {}, "envelopes": {},
                 "past_cycles": [_cycle(i, 3.4) for i in range(6)]},
    }
    (tmp_path / "synth").mkdir()
    (tmp_path / "synth" / "export.json").write_text(json.dumps(export))
    shipped = (se._PROGRAMME_LENGTH_RATIO, se._TAIL_SETTLED_FRACTION)  # noqa: SLF001
    rows = evidence.run(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert set(row["length"]) == set(evidence.LENGTH_ARMS)
    assert set(row["standby"]) == set(evidence.TAIL_ARMS)
    assert row["standby"]["legacy"] == 3.4 and row["standby"]["0.25"] == 3.4
    assert (se._PROGRAMME_LENGTH_RATIO, se._TAIL_SETTLED_FRACTION) == shipped  # noqa: SLF001


def _write(root: Path, rel: str, data: dict, options: dict | None = None) -> None:
    doc = {"version": 16, "entry_data": {"device_type": "washing_machine", "name": "Synth"},
           "entry_options": options or {}, "data": {"profiles": {}, "envelopes": {}, **data}}
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(json.dumps(doc))


def test_each_tail_arm_computes_its_own_settled_level(tmp_path: Path) -> None:
    """PR #529 review: the settled-tail cache is keyed by trace, not by the fraction,
    so the 0.25 and 0.5 arms read the 0.1 arm's level. Two hand stops whose tails
    hold 1 W for 20% of the time and 4 W for 80%, under a 2 W Stop Threshold."""
    past = []
    for i in range(2):
        pts = [[60.0 * k, 500.0] for k in range(30)] + [[1800.0, 1.0]]
        pts += [[1920.0 + 60.0 * k, 4.0] for k in range(9)]   # 1920 .. 2400 s
        past.append({"id": f"tail-{i}", "status": "completed", "termination_reason": "user",
                     "profile_name": "Wash", "start_time": f"2026-02-0{i + 1}T08:00:00+00:00",
                     "duration": 2400.0, "power_data": pts})
    _write(tmp_path, "tail/export.json", {"past_cycles": past}, {"stop_threshold_w": 2.0})
    (row,) = evidence.run(tmp_path)
    assert row["standby"] == {"legacy": 4.0, "0.1": None, "0.25": 4.0, "0.5": 4.0}


def test_a_store_package_cannot_drop_a_history_as_its_clone(tmp_path: Path) -> None:
    """PR #529 review: the clone rule kept the larger file, a store package holding
    ten of a history's traces, and dropped the history; the later ``store/`` filter
    then left neither."""
    cycles = [_cycle(i, 0.0, reason="timeout") for i in range(11)]
    for i, c in enumerate(cycles):
        c["power_data"][5][1] = 100.0 + i   # eleven distinct traces
    _write(tmp_path, "store/washing_machine/pkg.json", {"past_cycles": [], "reference_cycles": cycles})
    _write(tmp_path, "hist/export.json", {"past_cycles": cycles[:10]})
    assert [r["device"] for r in evidence.run(tmp_path)] == ["hist/export.json"]


def test_cycles_without_an_id_are_told_apart(tmp_path: Path) -> None:
    """PR #529 review: the clean sets were keyed by ``str(id)``, so every id-less
    cycle was one key "None": one kept with no exemption hid the two the shipped
    band admits from their duration ratios."""

    def run_(dead_s: float) -> dict:
        pts = [[0.0, 0.0]]
        for k in range(1, 60):   # 60 s steps over 3600 s; a dead run from 1200 s
            t = 60.0 * k
            pts.append([t, 0.0 if 1200.0 <= t < 1200.0 + dead_s or t >= 3360.0 else 2000.0])
        pts.append([3600.0, 0.0])
        return {"status": "completed", "termination_reason": "smart", "profile_name": "Wash",
                "start_time": "2026-03-01T08:00:00+00:00", "duration": 3600.0, "power_data": pts}

    past = [run_(900.0), run_(900.0), run_(0.0)]   # two merged-looking, one plain; no ids
    _write(tmp_path, "noid/export.json", {"past_cycles": past}, {"stop_threshold_w": 2.0})
    (row,) = evidence.run(tmp_path)
    assert row["length"]["off"] == {"clean": 1, "mid_restart": 2}
    assert row["length"]["0.8-1.25"] == {"clean": 3, "mid_restart": 0}
    assert row["admitted_ratios"] == [1.0, 1.0]


def _dead_run(i: int, dead_s: float) -> dict:
    """3600 s at 2 kW, 60 s readings, with a dead stretch from 1200 s."""
    pts = [[0.0, 0.0]]
    for k in range(1, 60):
        t = 60.0 * k
        pts.append([t, 0.0 if 1200.0 <= t < 1200.0 + dead_s or t >= 3360.0 else 2000.0])
    pts.append([3600.0, 0.0])
    return {"id": f"d{i}", "status": "completed", "termination_reason": "smart", "profile_name": "Wash",
            "start_time": f"2026-03-0{i + 1}T08:00:00+00:00", "duration": 3600.0, "power_data": pts}


def test_an_admitted_cycle_without_a_duration_is_timed_by_its_trace(tmp_path: Path) -> None:
    """PR #529 review: selection times a cycle stored without a duration by its
    trace span, and the band admits it, but its ratio read the missing duration as
    0.0. A record labelled through ``label`` was likewise left out of the ratios."""
    no_duration = _dead_run(3, 900.0)
    del no_duration["duration"]
    by_label = _dead_run(4, 900.0)
    by_label["label"] = by_label.pop("profile_name")
    past = [_dead_run(0, 900.0), _dead_run(1, 0.0), _dead_run(2, 0.0), no_duration, by_label]
    _write(tmp_path, "span/export.json", {"past_cycles": past}, {"stop_threshold_w": 2.0})
    (row,) = evidence.run(tmp_path)
    assert row["length"]["off"] == {"clean": 2, "mid_restart": 3}
    assert row["length"]["0.8-1.25"] == {"clean": 5, "mid_restart": 0}
    assert row["admitted_ratios"] == [1.0, 1.0, 1.0]
