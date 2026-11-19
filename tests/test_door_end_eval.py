"""devtools/door_end_eval.py (#524): argument checks that need no corpus."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "devtools" / "door_end_eval.py"
_spec = importlib.util.spec_from_file_location("wd_door_end_eval", _PATH)
dee = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = dee
_spec.loader.exec_module(dee)


@pytest.mark.parametrize("dwell", ["nan", "inf", "-inf", "0", "-5", "soon"])
def test_an_unusable_dwell_is_refused(tmp_path: Path, dwell: str) -> None:
    """PR #529 review: ``--dwell nan`` dropped every door moment and reported zero
    exposure, as if the ratio were safe."""
    with pytest.raises(SystemExit) as exc:
        dee.main(["--root", str(tmp_path), "--dwell", dwell])
    assert exc.value.code == 2


@pytest.mark.parametrize("argv", [[], ["--dwell", "30"], ["--dwell", "0.5"]])
def test_the_default_and_a_positive_dwell_run(tmp_path: Path, argv: list[str]) -> None:
    assert dee.main(["--root", str(tmp_path), *argv]) == 0


def _trace() -> list[list[float]]:
    """A run with a mid-cycle quiet spell and a final 0 W reading."""
    pts = [[float(t), 500.0] for t in range(0, 1500, 10)]
    pts += [[float(t), 0.5] for t in range(1500, 2100, 10)]
    pts += [[float(t), 500.0] for t in range(2100, 2700, 10)]
    pts.append([2700.0, 0.0])
    return pts


def test_external_trigger_ends_are_measured(tmp_path: Path) -> None:
    """PR #529 review: a programme whose runs the External End Trigger ended
    (stored ``external`` since 0.6.1, a natural end) dropped out of the report."""

    def cycle(i: int) -> dict:
        return {
            "id": f"c{i}",
            "profile_name": "Cotton",
            "status": "completed",
            "termination_reason": "external",
            "start_time": f"2026-01-0{i + 1}T08:00:00+00:00",
            "duration": 2700.0 + 60 * i,
            "power_data": _trace(),
        }

    doc = {
        "data": {"past_cycles": [cycle(i) for i in range(3)], "profiles": {}},
        "entry_data": {"device_type": "washing_machine"},
        "entry_options": {},
    }
    (tmp_path / "export.json").write_text(json.dumps(doc), encoding="utf-8")
    report = dee.measure(tmp_path, 60.0)
    assert report["washing_machine"]["cycles"] == 3
