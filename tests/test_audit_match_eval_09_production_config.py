# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Audit MATCH-EVAL-09 / MATCH-EVAL-04 / MATCH-CORE-07 / TESTING-08.

The tuning harnesses measured a pipeline that does not ship: ``prefix_guard_eval``
(upper duration ratio 1.5, no ``energy_mode``, no Stage 5, the stop threshold read
from the top-level ``entry_options``), ``dtw_ab_eval`` (the same partial config)
and ``min_off_gap_eval`` (a hand-rolled detector config with its own defaults).
Each now takes its config from the production builders.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "devtools"))

import dtw_ab_eval  # noqa: E402
import min_off_gap_eval  # noqa: E402
import prefix_guard_eval  # noqa: E402

from custom_components.ha_washdata.analysis import stage4_energy_mode  # noqa: E402
from custom_components.ha_washdata.const import (  # noqa: E402
    DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO,
)
from custom_components.ha_washdata.detector_config import (  # noqa: E402
    build_detector_config,
)


@pytest.fixture(autouse=True)
def _logging_untouched():
    """A harness called as a library must leave logging as it found it: a leaked
    ``logging.disable`` broke tests/test_diag_buffer.py in the same session."""
    root = logging.getLogger()
    before = (logging.root.manager.disable, root.level, list(root.handlers))
    yield
    assert (logging.root.manager.disable, root.level, list(root.handlers)) == before


@pytest.mark.parametrize("device_type", ["washing_machine", "dishwasher", "dryer"])
@pytest.mark.parametrize("options", [{}, {"stop_threshold_w": 4.0, "off_delay": 600,
                                          "smart_termination_duration_ratio": 0.97}])
def test_min_off_gap_replays_the_production_detector_config(
    device_type: str, options: dict
) -> None:
    got = min_off_gap_eval._cfg({}, options, device_type, 1234)  # noqa: SLF001
    want = build_detector_config({**options, "min_off_gap": 1234}, {}, device_type)
    assert dataclasses.asdict(got) == dataclasses.asdict(want)


def test_min_off_gap_replay_ends_one_cycle() -> None:
    cfg = min_off_gap_eval._cfg({}, {}, "washing_machine", 600)  # noqa: SLF001
    trace = [(t * 10.0, 400.0 if t < 200 else 0.5) for t in range(240)]
    ended = min_off_gap_eval._replay(cfg, {"device_type": "washing_machine"}, trace)  # noqa: SLF001
    assert len(ended) == 1


def _export(path: Path, device_type: str) -> None:
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
        "device_fingerprint": {"device_type": device_type},
        "entry_data": {}, "entry_options": {},
        "data": {"past_cycles": cycles, "profiles": {"Cotton": {}, "Quick": {}}},
    }), encoding="utf-8")


@pytest.mark.parametrize("device_type", ["washing_machine", "dishwasher"])
def test_dtw_ab_eval_uses_the_shipped_stage_1_4_config(tmp_path: Path, device_type: str) -> None:
    path = tmp_path / "export.json"
    _export(path, device_type)
    cfg = dtw_ab_eval._shipped_cfg(str(path))  # noqa: SLF001
    assert cfg["max_duration_ratio"] == DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO
    assert cfg["energy_mode"] == stage4_energy_mode(device_type)
    assert cfg["dtw_bandwidth"] > 0


def test_dtw_ab_eval_keeps_the_export_options_without_a_device_type(tmp_path: Path) -> None:
    # No device type anywhere: the fallback replaced the whole document, so the
    # export's own options were dropped and the defaults ran instead.
    path = tmp_path / "export.json"
    _export(path, "washing_machine")
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["device_fingerprint"] = {}
    doc["entry_options"] = {"profile_match_max_duration_ratio": 2.4}
    path.write_text(json.dumps(doc), encoding="utf-8")
    dtw_ab_eval._shipped_cfg_cached.cache_clear()  # noqa: SLF001
    assert dtw_ab_eval._shipped_cfg(str(path))["max_duration_ratio"] == 2.4  # noqa: SLF001


def test_decisive_margin_key_hashes_names_through_a_symlinked_corpus(tmp_path: Path) -> None:
    # cycle_data/ is often a symlink: resolving it left the repo and the fallback
    # printed the raw file name, a contributor's real name under user-Contributed/.
    import decisive_margin_eval as dm  # noqa: PLC0415

    real = tmp_path / "corpus" / "user-Contributed"
    real.mkdir(parents=True)
    (real / "Jane Doe washer.json").write_text("{}", encoding="utf-8")
    link = dm.REPO / "cycle_data" / "user-Contributed" / "Jane Doe washer.json"
    key = dm._key(link)  # noqa: SLF001
    assert "Jane" not in key and key.startswith("cycle_data/user-Contributed/")
    assert "Jane" not in dm._key(real / "Jane Doe washer.json")  # noqa: SLF001


@pytest.mark.slow
def test_prefix_guard_folds_come_from_the_shipped_matcher(tmp_path: Path) -> None:
    path = tmp_path / "export.json"
    _export(path, "washing_machine")
    from custom_components.ha_washdata import analysis  # noqa: PLC0415
    from custom_components.ha_washdata import profile_store as ps  # noqa: PLC0415

    orig, orig_worker = ps._match_prefix_ambiguity, analysis.compute_matches_worker  # noqa: SLF001
    out = prefix_guard_eval._collect_device((str(path), True))  # noqa: SLF001
    # The recorders are removed again.
    assert ps._match_prefix_ambiguity is orig  # noqa: SLF001
    assert analysis.compute_matches_worker is orig_worker
    assert len(out["neg"]) == 6
    for row in out["neg"]:
        assert row["cands"], row
        assert isinstance(row["landscape_paused"], bool)
        # The sweep's rule at the shipped margin is the row's flag (no pauses).
        assert prefix_guard_eval._prefix_fires(  # noqa: SLF001
            row["cands"], row["best_dur"], prefix_guard_eval.SHIPPED_PREFIX[0]
        ) == row["landscape"]
    # The harness scored the removed rule's prefix itself: Quick folds see the
    # longer Cotton programme with a harness prefix score.
    assert any(c.get("ps_wide") is not None for r in out["pos"] + out["neg"]
               for c in r["cands"][1:])
