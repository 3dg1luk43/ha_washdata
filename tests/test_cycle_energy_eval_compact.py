# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``devtools/cycle_energy_eval.py`` reads a raw v19 store too.

From storage v19 a raw store file holds each trace as compact JSON text
(``trace_codec``); the harness skipped those cycles silently (``power_data`` not a
list), so a mixed corpus gave incomplete figures with no warning.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from custom_components.ha_washdata.trace_codec import encode_store_payload

REPO = Path(__file__).resolve().parent.parent


def test_a_raw_v19_store_is_read(tmp_path: Path) -> None:
    pts = [[float(i * 30), 500.0 if i < 100 else 0.0] for i in range(120)]
    cyc = {"id": "a", "start_time": "2026-10-01T08:00:00+00:00", "duration": 3570.0,
           "power_data": pts}
    store = encode_store_payload({"past_cycles": [cyc], "envelopes": {}})
    assert isinstance(store["past_cycles"][0]["power_data"], str)
    (tmp_path / "raw.json").write_text(json.dumps({"version": 19, "data": store}))
    out = subprocess.run(
        [sys.executable, str(REPO / "devtools" / "cycle_energy_eval.py"), "--root", str(tmp_path)],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert out.returncode == 0, out.stderr
    assert "cycles: 1" in out.stdout, out.stdout


def test_a_malformed_first_copy_does_not_hide_a_valid_one(tmp_path: Path) -> None:
    pts = [[float(i * 30), 500.0 if i < 100 else 0.0] for i in range(120)]
    good = {"id": "a", "start_time": "2026-10-01T08:00:00+00:00", "duration": 3570.0,
            "power_data": pts}
    bad = {**good, "power_data": [[p[0], 0.0] for p in pts]}   # no energy: not a row
    (tmp_path / "a_bad.json").write_text(json.dumps({"data": {"past_cycles": [bad]}}))
    (tmp_path / "b_good.json").write_text(json.dumps({"data": {"past_cycles": [good]}}))
    out = subprocess.run(
        [sys.executable, str(REPO / "devtools" / "cycle_energy_eval.py"), "--root", str(tmp_path)],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert out.returncode == 0, out.stderr
    assert "cycles: 1" in out.stdout, out.stdout

