# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""devtools/terminal_drop_plugpull_eval.py on one tiny synthetic dishwasher export.

The corpus run is a devtools job (``cycle_data/`` is gitignored); this needs none.
Four completed 60 min runs at 2 kW whose first quiet is at 40 min, no profiles: a
cut at 15% (9 min) is an anomalously-early drop at a familiar power, so the
ungated rule fires and closes it in minutes, while the shipped (guarded) rule has
no committed match to fire on and ``off`` never fires.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "devtools" / "terminal_drop_plugpull_eval.py"
_spec = importlib.util.spec_from_file_location("wd_terminal_drop_plugpull_eval", _PATH)
pp = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pp
_spec.loader.exec_module(pp)


def _cycle(i: int) -> dict:
    pts = [[float(t), 2000.0] for t in range(0, 2400, 60)]
    pts += [[float(t), 0.0] for t in range(2400, 2700, 60)]
    pts += [[float(t), 2000.0] for t in range(2700, 3600, 60)] + [[3600.0, 0.0]]
    return {
        "id": f"syn-{i}",
        "start_time": f"2026-04-0{i + 1}T08:00:00+00:00",
        "end_time": f"2026-04-0{i + 1}T09:00:00+00:00",
        "duration": 3600.0,
        "status": "completed",
        "power_data": pts,
    }


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    doc = {
        "device_fingerprint": {"device_type": "dishwasher"},
        "entry_data": {"device_type": "dishwasher"},
        "entry_options": {"device_type": "dishwasher", "min_power": 2.0,
                          "stop_threshold_w": 1.5, "start_threshold_w": 3.0},
        "data": {"past_cycles": [_cycle(i) for i in range(4)], "profiles": {}},
    }
    (tmp_path / "dw.json").write_text(json.dumps(doc))
    return tmp_path


def _run(corpus: Path, rule: str) -> list[dict]:
    jobs = pp.corpus_jobs(corpus, ("dishwasher",), rule, (0.15,), 1)
    assert len(jobs) == 1
    return pp._export_job(jobs[0])  # noqa: SLF001


def test_cut_cycle_appends_the_zero_tail() -> None:
    cyc = _cycle(0)
    pts = [(float(t), float(p)) for t, p in cyc["power_data"]]
    cut, cut_s = pp.cut_cycle(cyc, pts, 0.5)
    assert cut_s == 1800.0
    assert len(cut["power_data"]) == 31 + pp.TAIL_SAMPLES
    assert all(p == 0.0 for _t, p in cut["power_data"][31:])


def test_rules_on_a_synthetic_plug_pull(corpus: Path) -> None:
    ungated = _run(corpus, "ungated")
    assert len(ungated) == 1 and ungated[0]["fired"] is True
    assert ungated[0]["close_min"] is not None and ungated[0]["close_min"] < 10

    for rule in ("off", "shipped"):
        (row,) = _run(corpus, rule)
        assert row["fired"] is False, rule
        assert row["close_min"] is None or row["close_min"] > 10, rule

    # The rule swap is undone afterwards.
    assert pp.playground.terminal_drop_may_fire is pp.detector_config.terminal_drop_may_fire


def test_summarise_counts_fires_and_closes() -> None:
    rows = [
        {"frac": 0.15, "fired": True, "close_min": 4.0, "n_finished": 1},
        {"frac": 0.15, "fired": False, "close_min": 90.0, "n_finished": 1},
        {"frac": 0.15, "fired": False, "close_min": None, "n_finished": 0},
    ]
    (t,) = pp.summarise(rows, (0.15,))
    assert (t["n"], t["fires"], t["never_closed"], t["splits"]) == (3, 1, 1, 0)
    assert t["median_close_min"] == 47.0
