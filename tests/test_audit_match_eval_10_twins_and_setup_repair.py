# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Harness hygiene (audit MATCH-EVAL-10, wave 7).

1. ``devtools/eval.py`` counted one washer twice: the maintainer's exports
   ``01KBWSV8`` and ``01KXGA3C`` are two config entries on the same plug (47 of 50
   runs start within 2 min of each other). Twin entries are now detected by
   overlapping run start times and, by default, only one of a pair is scored.
2. ``decisive_margin_eval``, ``playground_parity_eval`` and ``eta_eval`` replayed
   without the setup sample repair (``ProfileStore.async_repair_profile_samples``)
   that ``eval.py``, ``end_gate_eval.py`` and every real setup run first, so a
   profile whose only "sample" is another programme's run kept competing with it.

Synthetic exports in tmp_path; no cycle_data/ needed.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "devtools"))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ev = _load("wd_eval_twins", "devtools/eval.py")

_SHAPES = {"Quick": [(8, 2000), (20, 150), (2, 500)], "Long": [(20, 2200), (60, 200), (10, 650)]}


def _cycle(name: str, j: int, seed: int, start: str) -> dict:
    rng = np.random.default_rng(seed)
    pts, t = [], 0.0
    for minutes, watts in _SHAPES[name]:
        for _ in range(int(minutes * 60 * (1.0 + 0.03 * j) / 30)):
            pts.append([round(t, 1), round(max(0.0, watts * (1 + 0.05 * rng.standard_normal())), 1)])
            t += 30.0
    pts.append([round(t, 1), 0.0])
    return {"id": f"{name}-{j}-{seed}", "profile_name": name, "status": "completed",
            "start_time": start, "duration": t, "power_data": pts}


def _export(n_runs: int, seed: int, shift_s: int, day0: int = 1) -> dict:
    """``n_runs`` cycles alternating two programmes, one a day at 08:00 + shift."""
    past = []
    for j in range(n_runs):
        name = ("Quick", "Long")[j % 2]
        start = f"2026-03-{day0 + j:02d}T08:{shift_s // 60:02d}:{shift_s % 60:02d}+00:00"
        past.append(_cycle(name, j, seed + j, start))
    profiles = {n: {"sample_cycle_id": next(c["id"] for c in past if c["profile_name"] == n)}
                for n in _SHAPES}
    return {"entry_data": {"device_type": "washing_machine"}, "entry_options": {},
            "data": {"profiles": profiles, "past_cycles": past, "envelopes": {}}}


# --------------------------------------------------------------- 1. twin entries


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("twins")
    (root / "me").mkdir()
    # Two entries on one plug: same runs, different noise (so not clone files), the
    # second entry created later and holding fewer runs.
    (root / "me" / "a_main.json").write_text(json.dumps(_export(8, 0, 0)))
    (root / "me" / "b_test.json").write_text(json.dumps(_export(6, 500, 75, day0=3)))
    # Same hour every day but another month: never a twin.
    doc = _export(8, 900, 0)
    for c in doc["data"]["past_cycles"]:
        c["start_time"] = c["start_time"].replace("2026-03-", "2026-05-")
    (root / "other.json").write_text(json.dumps(doc))
    return root


def test_twin_entries_are_detected_by_overlapping_run_starts(corpus):
    devices, clones = ev.load_corpus(corpus)
    assert clones == {}
    twins = ev.twin_entries(devices)
    # The entry with more labelled traced cycles is kept.
    assert twins == {"me/b_test.json": {"kept": "me/a_main.json", "overlap": 6, "runs": 6}}


def test_by_default_only_the_kept_twin_is_scored(corpus, tmp_path):
    doc = ev.run_eval(corpus, mode="full", cuts=(1.0,), jobs=1, cache_dir=tmp_path, log=lambda *a, **k: None)
    assert sorted(doc["devices"]) == ["me/a_main.json", "other.json"]
    assert {r["path"] for r in doc["folds"]} == {"me/a_main.json", "other.json"}
    assert doc["meta"]["twin_entries"] == {
        "me/b_test.json": {"kept": "me/a_main.json", "overlap": 6, "runs": 6, "scored": False}}


def test_include_twins_scores_both(corpus, tmp_path):
    doc = ev.run_eval(corpus, mode="full", cuts=(1.0,), jobs=1, cache_dir=tmp_path,
                      log=lambda *a, **k: None, include_twins=True)
    assert "me/b_test.json" in {r["path"] for r in doc["folds"]}
    assert doc["meta"]["twin_entries"]["me/b_test.json"]["scored"] is True


def test_a_few_coincident_starts_are_not_twins():
    a = ev.Device("a", "u", "export", "x", {}, {}, json.loads(json.dumps(_export(8, 0, 0)["data"])))
    b_doc = _export(8, 50, 0, day0=20)["data"]
    # Three runs started together (two appliances on one routine): below TWIN_MIN_RUNS.
    for j, c in enumerate(b_doc["past_cycles"][:3]):
        c["start_time"] = a.data["past_cycles"][j]["start_time"]
    b = ev.Device("b", "u", "export", "y", {}, {}, b_doc)
    assert ev.twin_entries([a, b]) == {}


# ---------------------------------------------------- 2. setup repair in harnesses


def _repair_export(path: Path) -> None:
    """Six washer cycles over two programmes, plus a profile whose only "sample" is
    a Cotton run (no cycle of its own): the setup repair clears that pointer."""
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
    path.write_text(json.dumps({
        "device_fingerprint": {"device_type": "washing_machine"},
        "entry_data": {"device_type": "washing_machine"},
        "entry_options": {"stop_threshold_w": 2.0, "min_power": 2.0},
        "data": {"past_cycles": cycles, "envelopes": {},
                 "profiles": {"Cotton": {}, "Quick": {}, "Ghost": {"sample_cycle_id": "c0"}}},
    }), encoding="utf-8")


@pytest.fixture(name="export")
def _export_path(tmp_path: Path) -> Path:
    path = tmp_path / "export.json"
    _repair_export(path)
    return path


@pytest.fixture(autouse=True)
def _restore_logging():
    before = logging.root.manager.disable
    yield
    logging.disable(before)


def test_decisive_margin_eval_runs_the_setup_repair(export):
    import decisive_margin_eval as dme  # noqa: PLC0415

    dev = dme._device(export, False)  # noqa: SLF001
    assert dev is not None
    _doc, base, store, _cfg, _opts = dev
    assert base["profiles"]["Ghost"]["sample_cycle_id"] is None
    assert store._data["profiles"]["Ghost"]["sample_cycle_id"] is None  # noqa: SLF001


def test_eta_eval_runs_the_setup_repair(export):
    eta = _load("wd_eta_eval_repair", "devtools/eta_eval.py")
    flags = {"loo": True, "all_formats": False, "shipped_watchdog": False, "repair": True}
    ctx = eta._prepare(export, flags)  # noqa: SLF001
    assert ctx is not None
    assert ctx["base"]["profiles"]["Ghost"]["sample_cycle_id"] is None


@pytest.mark.parametrize("entry", ["_match_export", "_replay_export"])
def test_playground_parity_eval_runs_the_setup_repair(export, monkeypatch, entry):
    import end_gate_eval  # noqa: PLC0415
    import playground_parity_eval as ppe  # noqa: PLC0415

    seen: list = []
    real = end_gate_eval._rebuild_envelopes  # noqa: SLF001

    def spy(store, names):
        seen.append(store._data["profiles"]["Ghost"].get("sample_cycle_id"))  # noqa: SLF001
        return real(store, names)

    monkeypatch.setattr(end_gate_eval, "_rebuild_envelopes", spy)
    # Every cycle but the last, so the run stays short.
    getattr(ppe, entry)((str(export), 1, False))
    assert seen and seen[0] is None
