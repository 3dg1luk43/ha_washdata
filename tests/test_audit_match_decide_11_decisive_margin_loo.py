# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Audit MATCH-DECIDE-11 / MATCH-EVAL-06 / -16: decisive_margin_eval measures LOO.

In-sample, every checkpoint was matched against an envelope the cycle itself
helped build (sentinel 99.5% / real margin 92.5% correct; leave-one-out 96.3% /
87.8%). ``--loo`` matches each cycle against a store rebuilt without it, the
``end_gate_eval._production`` pattern; ``--switching`` replays the live switching
state machine and summarises what was displayed.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "devtools"))

import decisive_margin_eval as dme  # noqa: E402


@pytest.fixture(autouse=True)
def _logging_untouched():
    """A harness called as a library must leave logging as it found it: a leaked
    ``logging.disable`` broke tests/test_diag_buffer.py in the same session."""
    root = logging.getLogger()
    before = (logging.root.manager.disable, root.level, list(root.handlers))
    yield
    assert (logging.root.manager.disable, root.level, list(root.handlers)) == before


def _export(path: Path) -> None:
    """Six completed washer cycles over two programmes, a 10 s plug."""
    cycles = []
    for i in range(6):
        long = i % 2 == 0
        n = 360 if long else 180
        trace = [[t * 10.0, 0.5 if t > n - 6 else (1800.0 if t < 40 else 150.0 + (t % 7))]
                 for t in range(n)]
        cycles.append({
            "id": f"c{i}", "start_time": f"2026-09-{10 + i:02d}T08:00:00+00:00",
            "duration": n * 10.0, "status": "completed", "termination_reason": "smart",
            "profile_name": "Cotton" if long else "Quick", "label_source": "manual",
            "power_data": trace,
        })
    cycles.append({**cycles[0], "id": "c-only", "profile_name": "Solo"})
    path.write_text(json.dumps({
        "device_fingerprint": {"device_type": "washing_machine"},
        "entry_data": {}, "entry_options": {"stop_threshold_w": 2.0},
        "data": {"past_cycles": cycles,
                 "profiles": {"Cotton": {}, "Quick": {}, "Solo": {}}},
    }), encoding="utf-8")


@pytest.fixture(name="device")
def _device(tmp_path: Path):
    path = tmp_path / "export.json"
    _export(path)
    dev = dme._device(path, False)  # noqa: SLF001
    assert dev is not None
    return dev


def _ids(store) -> set[str]:
    return {str(c.get("id")) for c in store.iter_stored_cycles()}


def test_loo_matches_against_a_store_without_the_cycle(device) -> None:
    doc, base, store, _cfg, _opts = device
    cyc = base["past_cycles"][0]
    assert dme._fold_store(doc, base, cyc, store, False) is store  # noqa: SLF001
    fold = dme._fold_store(doc, base, cyc, store, True)  # noqa: SLF001
    assert fold is not store
    assert "c0" in _ids(store) and "c0" not in _ids(fold)
    assert _ids(store) - _ids(fold) == {"c0"}
    # The held-out programme's envelope was rebuilt from what is left.
    full = store.get_envelope("Cotton") or {}
    held = fold.get_envelope("Cotton") or {}
    assert int(held.get("cycle_count") or 0) == int(full.get("cycle_count") or 0) - 1


def test_scorable_means_the_programme_keeps_another_cycle(device) -> None:
    _doc, base, _store, _cfg, _opts = device
    by_id = {c["id"]: c for c in base["past_cycles"]}
    assert dme._scorable(base, by_id["c0"])  # noqa: SLF001
    assert not dme._scorable(base, by_id["c-only"])  # noqa: SLF001


@pytest.mark.slow
def test_loo_tally_is_labelled_by_mode(tmp_path: Path, capsys) -> None:
    path = tmp_path / "export.json"
    _export(path)
    for loo in (False, True):
        tally = dme._scan_export((str(path), loo, False))  # noqa: SLF001
        assert tally["cycles"] == 7
        assert tally["checkpoints"] > 0
    dme._print_bypass(tally, True)  # noqa: SLF001
    assert "mode                                : LOO" in capsys.readouterr().out


@pytest.mark.slow
def test_switching_rows_read_the_displayed_programme(tmp_path: Path, capsys) -> None:
    path = tmp_path / "export.json"
    _export(path)
    rows = dme._switching_rows((str(path), True, False))  # noqa: SLF001
    assert len(rows) == 7
    for r in rows:
        assert set(r) >= {"committed", "first_right", "final_right", "switches",
                          "reverts", "wrong_to_right", "right_to_wrong", "scorable",
                          "notify_right", "commit_s"}
        assert r["first_right"] <= r["committed"]
        assert (r["commit_s"] is not None) == r["committed"]
    assert sum(r["scorable"] for r in rows) == 6
    dme._print_switching(rows, True)  # noqa: SLF001
    assert "live switching replay (LOO)" in capsys.readouterr().out
