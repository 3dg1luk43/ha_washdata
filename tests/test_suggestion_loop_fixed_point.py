"""Audit F10 / SUGGEST-16: every shipped suggestion reaches a fixed point on the corpus.

``devtools/suggestion_loop_eval.py`` re-records each corpus device's history under the
options Apply all produces (real detector + matcher replay, the manager's cadence
model, every suggestion pass through the LearningManager gates, the panel's Apply
all) and iterates. A setting that is still moving after ``ROUNDS`` applies is a
ladder or an oscillation: exactly what the 0.5.8 cuts removed (the confidence
thresholds fell, Sampling Interval rose and Completion Minimum crept up with every
apply), and what no formula-level test could see.

``ROUNDS = 5``: the shipped keys settle in at most 2 applies on this corpus (their
statistics depend on the re-detected history only through cycle boundaries, so one
re-detection under the new value is enough to reach the value it implies); a ladder
needs ``LADDER_MIN_APPLIES`` (3) same-direction applies to be told apart from a
correction that overshot once, and the removed Sampling Interval ladder shows that
by its third apply. Five leaves two rounds of margin over the slowest legitimate
convergence without hiding a ladder.

The revert checks run two suggestions removed in 0.5.8 with their old logic and
require the harness to fail them: Sampling Interval as a ladder, Completion Minimum
as erased evidence (it climbs, then stops once it has deleted the programme it was
sized from). If it cannot, a passing fixed-point test proves nothing.

Runtime: ~7 min on 8 cores (the corpus run), ~1.5 min for the revert checks.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

_REPO = Path(__file__).resolve().parents[1]
_CORPUS = _REPO / "cycle_data"
_PATH = _REPO / "devtools" / "suggestion_loop_eval.py"
_spec = importlib.util.spec_from_file_location("wd_suggestion_loop_eval_slow", _PATH)
loop = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = loop
_spec.loader.exec_module(loop)

ROUNDS = 5
#: The AEG washer of #427: 2 s throttle, ~5-10 s plug. The audit measured the old
#: sampling_interval suggestion walking it 2 -> 7 -> 12 -> ... -> 34 s.
_LADDER_DEVICE = "KoLSMS/washing_machine/washdata_export_01KW6VW3_427_aeg.json"

#: A contributed dishwasher (entry id only: contributed file names carry names) whose
#: shortest programme the old Completion Minimum turned interrupted.
_RATCHET_DEVICE = "01KGM619"

needs_corpus = pytest.mark.skipif(not _CORPUS.is_dir(), reason="cycle_data/ corpus not present")


@pytest.fixture(scope="module")
def results() -> list[dict]:
    return loop.run(_CORPUS, rounds=ROUNDS, jobs=min(8, os.cpu_count() or 1))


@needs_corpus
def test_every_shipped_suggestion_reaches_a_fixed_point(results):
    # The whole traced corpus, not a sample: 20+ devices carry enough history.
    assert len(results) >= 20, [r["device"] for r in results]
    assert loop.failures(results) == []
    # Each device stopped because Apply all had nothing left to change, not
    # because it ran out of rounds.
    assert all(r["fixed_point"] for r in results), [
        r["device"] for r in results if not r["fixed_point"]
    ]


@needs_corpus
def test_the_cooldown_expires_and_never_leaks(results):
    for r in results:
        assert not r["cooldown_leaks"], (r["device"], r["cooldown_leaks"])
        assert not any(row["cooldown_active_after_expiry"] for row in r["rounds"]), r["device"]


@needs_corpus
def test_a_muted_setting_is_never_applied(results):
    probed = [r for r in results if r["lock_probe"]]
    assert probed
    for r in probed:
        assert r["lock_probe"]["applied"] == [], (r["device"], r["lock_probe"])


@needs_corpus
@pytest.mark.skipif(not (_CORPUS / _LADDER_DEVICE).exists(), reason="#427 export not in cycle_data/")
def test_revert_check_the_removed_sampling_interval_suggestion_is_a_ladder():
    # Three applies plus the evaluation after them is enough to see it still climbing.
    res = loop.run(
        _CORPUS, rounds=3, jobs=min(8, os.cpu_count() or 1), only=[_LADDER_DEVICE],
        legacy=("sampling_interval",), lock_probe=False,
    )
    assert len(res) == 1
    verdict = res[0]["keys"]["sampling_interval"]
    assert verdict["verdict"] == "ladder", verdict
    assert all(b > a for a, b in zip(verdict["values"], verdict["values"][1:])), verdict
    assert loop.failures(res)


@needs_corpus
def test_revert_check_the_removed_completion_minimum_erases_its_own_evidence():
    """SUGGEST-03: half the p05 of CLEAN durations, and interrupted is not clean.

    Each apply stores the shortest runs interrupted, which drops them from the next
    p05, so the value climbs until the shortest programme is gone. The loop stops
    there, so it is not a ladder; it is caught as erased evidence.
    """
    res = loop.run(
        _CORPUS, rounds=ROUNDS, jobs=min(8, os.cpu_count() or 1), only=[_RATCHET_DEVICE],
        legacy=("completion_min",), lock_probe=False,
    )
    if not res:
        pytest.skip("contributed dishwasher export not in cycle_data/")
    values = res[0]["keys"]["completion_min_seconds"]["values"]
    assert len(values) >= 3 and all(b > a for a, b in zip(values, values[1:])), values
    assert res[0]["erased"], res[0]["keys"]
    assert any("erased" in line for line in loop.failures(res))
