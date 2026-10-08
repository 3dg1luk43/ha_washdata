"""Audit MATCH-EVAL-11: the committed eval baseline can be checked for staleness.

``devtools/eval_baseline.json`` was taken at code_sha e330991b and nothing noticed
when the matcher moved on. ``eval.py baseline-status`` hashes the matcher sources
(the same ``code_sha`` a run records) and asks git whether the baseline's ``rev``
is an ancestor of HEAD; ``release_check.sh`` prints its verdict as a warning.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("wd_eval_status", _REPO / "devtools" / "eval.py")
ev = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ev
_spec.loader.exec_module(ev)


def _head() -> str:
    return subprocess.run(["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture(autouse=True)
def _fixed_code_sha(monkeypatch):
    """Hashing the matcher sources parses ~15 modules; the logic under test is not that."""
    monkeypatch.setattr(ev, "code_sha", lambda: "c0ffee0000000000")


def _baseline(tmp_path: Path, **meta) -> Path:
    p = tmp_path / "base.json"
    p.write_text(json.dumps({"meta": {"mode": "fast", **meta}}))
    return p


def test_fresh_when_code_and_rev_match(tmp_path):
    out: list[str] = []
    assert ev.baseline_status(_baseline(tmp_path, code_sha=ev.code_sha(), rev=_head()), out=out.append) == 0
    assert out[0].startswith("fresh")


def test_stale_when_the_matcher_sources_moved(tmp_path):
    out: list[str] = []
    assert ev.baseline_status(_baseline(tmp_path, code_sha="0" * 16, rev=_head()), out=out.append) == 1
    assert "matcher sources changed" in out[0] and "eval.py run --mode fast" in out[0]


def test_stale_when_the_rev_is_not_in_this_history(tmp_path):
    out: list[str] = []
    assert ev.baseline_status(_baseline(tmp_path, code_sha=ev.code_sha(), rev="0000000"), out=out.append) == 1
    assert "0000000" in out[0]


def test_unreadable(tmp_path):
    assert ev.baseline_status(tmp_path / "missing.json", out=lambda *_: None) == 2


def test_release_check_runs_it():
    assert "eval.py baseline-status" in (_REPO / "devtools" / "release_check.sh").read_text()
