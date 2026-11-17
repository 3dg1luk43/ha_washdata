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
