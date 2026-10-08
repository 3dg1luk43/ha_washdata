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

Register item 455: a fixed point can still cost detection. The batch stop/start
anchor read a resting draw as the lowest running power, so Apply all put the stop
threshold under a dishwasher's 0.8 W drying phase (10 Eco cycles stranded, force-
stopped under ``--idle-hold``) and the start under a washer's 3.3 W post-end draw (2
labelled cycles split). Both are failures now, and pinned below.

Runtime (2026-10-05, 4 cores): serially one pooled corpus run, ~3.5 min, and ~1 min
for the revert and idle checks; under xdist ~1000 core-s in per-device units, the
longest ~130 s.

**Under pytest-xdist** the corpus run is one unit per device instead of one pooled
run: each device's four tests share an ``xdist_group``, so ``--dist loadgroup``
hands each device to one worker, which runs it on its own (``jobs=1``: no process
pool inside a worker, and the revert checks likewise) while the others take the
next device. A device's result is the same either way: measured 2026-10-05,
the 21 per-device runs equal the pooled run's 21 results (CPU 666 s per device vs
752 s pooled: each device keeps its matcher memo in one process).
"""
from __future__ import annotations

import functools
import importlib.util
import os
import pickle
import sys
from pathlib import Path

import pytest
from filelock import FileLock

# heavy: under pytest-xdist these go out first (tests/conftest.py), so no worker
# starts a minutes-long device run last.
pytestmark = [pytest.mark.slow, pytest.mark.heavy]

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

_XDIST = bool(os.environ.get("PYTEST_XDIST_WORKER"))


def _jobs() -> int:
    """A process pool when this is the only process; one process per xdist worker."""
    return 1 if _XDIST else min(8, os.cpu_count() or 1)


@functools.cache
def _devices() -> tuple[str, ...]:
    """The corpus devices the loop runs, biggest file first (``loop.run``'s order)."""
    paths = loop.device_paths(_CORPUS)
    return tuple(sorted(paths, key=lambda p: (-(_CORPUS / p).stat().st_size, p)))


def _group(device: str) -> str:
    return f"suggestion-loop-{_devices().index(device)}"


def _slow_selected(config: pytest.Config) -> bool:
    """Would ``-m`` keep slow tests? (Scanning the corpus costs ~2 s per process.)"""
    expr = (config.getoption("markexpr") or "").strip()
    if not expr:
        return True
    try:
        from _pytest.mark.expression import Expression  # noqa: PLC0415

        return Expression.compile(expr).evaluate(lambda name, **_kw: name == "slow")
    except Exception:  # noqa: BLE001 - unknown pytest internals: scan, to be safe
        return True


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """One test per corpus device, grouped per device for ``--dist loadgroup``."""
    if "device" not in metafunc.fixturenames:
        return
    if not _CORPUS.is_dir() or not _slow_selected(metafunc.config):
        # Skipped (no corpus) or deselected (-m "not slow"): one placeholder.
        metafunc.parametrize("device", ["corpus"])
        return
    metafunc.parametrize("device", [
        pytest.param(d, id=loop.ev.public_key(d), marks=pytest.mark.xdist_group(_group(d)))
        for d in _devices()
    ])


@functools.cache
def _pooled() -> dict[str, dict]:
    return {r["device"]: r for r in loop.run(_CORPUS, rounds=ROUNDS, jobs=_jobs())}


_DONE: dict[str, dict] = {}


def _result(device: str, shared: Path) -> dict:
    """The loop's result for one device, computed once per session.

    Serial: one pooled corpus run. Under xdist: this device alone, in this worker,
    under a lock in the session's shared temp dir, so a test of the same device on
    another worker (any ``--dist`` but loadgroup) loads it instead of repeating it.
    """
    key = loop.ev.public_key(device)
    if not _XDIST:
        return _pooled()[key]
    if key not in _DONE:
        out = shared / f"suggestion_loop_{_devices().index(device)}.pickle"
        with FileLock(f"{out}.lock"):
            if out.is_file():
                _DONE[key] = pickle.loads(out.read_bytes())
            else:
                res = loop.run(_CORPUS, rounds=ROUNDS, jobs=1, only=[device])
                # ``only`` matches substrings: exactly this device, or the split is wrong.
                assert [r["device"] for r in res] == [key], [r["device"] for r in res]
                out.write_bytes(pickle.dumps(res[0]))
                _DONE[key] = res[0]
    return _DONE[key]


@pytest.fixture
def result(device, tmp_path_factory) -> dict:
    return _result(device, tmp_path_factory.getbasetemp().parent)


@needs_corpus
def test_the_loop_covers_the_whole_traced_corpus(tmp_path_factory):
    # The whole traced corpus, not a sample: 20+ devices carry enough history.
    devices = _devices()
    assert len(devices) >= 20, devices
    # The muted-setting check below is not vacuous: some device's Apply all moved.
    shared = tmp_path_factory.getbasetemp().parent
    assert any(_result(d, shared)["lock_probe"] for d in devices)


@needs_corpus
def test_every_shipped_suggestion_reaches_a_fixed_point(result):
    assert loop.failures([result]) == []
    # The device stopped because Apply all had nothing left to change, not because
    # it ran out of rounds.
    assert result["fixed_point"], result["device"]


@needs_corpus
def test_the_cooldown_expires_and_never_leaks(result):
    assert not result["cooldown_leaks"], (result["device"], result["cooldown_leaks"])
    assert not any(row["cooldown_active_after_expiry"] for row in result["rounds"]), result["device"]


@needs_corpus
def test_a_muted_setting_is_never_applied(result):
    if result["lock_probe"]:
        assert result["lock_probe"]["applied"] == [], (result["device"], result["lock_probe"])


@needs_corpus
def test_issue_455_apply_all_never_splits_or_strands_a_labelled_cycle(result):
    """(b) a start under the post-end draw splits; (a) a stop under it strands."""
    assert not result["fragmented"], (result["device"], result["fragmented"], result["keys"])
    idle = [row["replay"]["idle_above_stop"] for row in result["rounds"]]
    assert idle[-1] <= idle[0], (result["device"], idle, result["keys"])


@needs_corpus
@pytest.mark.skipif(not (_CORPUS / _LADDER_DEVICE).exists(), reason="#427 export not in cycle_data/")
def test_revert_check_the_removed_sampling_interval_suggestion_is_a_ladder():
    # Three applies plus the evaluation after them is enough to see it still climbing.
    res = loop.run(
        _CORPUS, rounds=3, jobs=_jobs(), only=[_LADDER_DEVICE],
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
        _CORPUS, rounds=ROUNDS, jobs=_jobs(), only=[_RATCHET_DEVICE],
        legacy=("completion_min",), lock_probe=False,
    )
    if not res:
        pytest.skip("contributed dishwasher export not in cycle_data/")
    values = res[0]["keys"]["completion_min_seconds"]["values"]
    assert len(values) >= 3 and all(b > a for a, b in zip(values, values[1:])), values
    assert res[0]["erased"], res[0]["keys"]
    assert any("erased" in line for line in loop.failures(res))


@needs_corpus
def test_issue_455a_the_drying_phase_still_ends_cycles_when_it_is_standby():
    """``--idle-hold``: read every kept tail at or above the new stop as standby.

    The contributed dishwasher rests at 0.8 W through its drying phase; with stop
    0.56 W this run force-stopped its 10 Eco cycles whose Smart Termination fired
    there. Its stop threshold now stays put, so nothing is erased.
    """
    res = loop.run(
        _CORPUS, rounds=ROUNDS, jobs=_jobs(), only=[_RATCHET_DEVICE],
        lock_probe=False, idle_hold=True,
    )
    if not res:
        pytest.skip("contributed dishwasher export not in cycle_data/")
    assert res[0]["erased"] == [], res[0]["erased"]
    stop = res[0]["keys"].get("stop_threshold_w")
    # Above the 0.8 W rest with the standby floor's margin, if it moves at all.
    assert stop is None or stop["values"][-1] >= 1.0, stop
    assert loop.failures(res) == []
