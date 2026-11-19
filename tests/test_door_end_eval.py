"""devtools/door_end_eval.py (#524): argument checks that need no corpus."""
from __future__ import annotations

import importlib.util
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
