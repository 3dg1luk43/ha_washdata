"""Fast unit tests for the background-task registry."""
from __future__ import annotations

import time

from custom_components.ha_washdata import task_registry as tr


def test_create_update_finish_and_progress():
    reg = tr.TaskRegistry()
    events: list[dict] = []
    unsub = reg.add_listener(events.append)

    t = reg.create("entry1", "pg_sweep", "Optimize off_delay", total=10)
    assert t.state == tr.STATE_RUNNING
    assert t.progress() == 0.0
    assert events[-1]["kind"] == "pg_sweep" and events[-1]["total"] == 10

    reg.update(t, done=5)
    assert t.progress() == 0.5
    assert events[-1]["done"] == 5 and events[-1]["progress"] == 0.5

    reg.finish(t, result={"points": [1, 2, 3]})
    assert t.state == tr.STATE_DONE
    assert t.result == {"points": [1, 2, 3]}
    assert events[-1]["has_result"] is True and events[-1]["state"] == "done"
    # snapshot with result embeds the payload; without, only a flag.
    assert "result" not in t.snapshot()
    assert t.snapshot(include_result=True)["result"] == {"points": [1, 2, 3]}

    unsub()
    reg.update(t, done=10)
    assert events[-1]["done"] == 5  # listener removed -> no new event


def test_progress_none_when_total_unknown():
    reg = tr.TaskRegistry()
    t = reg.create("e", "ml_training", "Learning", total=0)
    assert t.progress() is None
    assert t.eta_s() is None


def test_eta_estimates_from_elapsed_and_progress():
    reg = tr.TaskRegistry()
    t = reg.create("e", "pg_history", "History", total=4)
    t.started_at = time.time() - 10.0  # 10s elapsed
    reg.update(t, done=2)              # 50% -> ETA ~ 10s remaining
    eta = t.eta_s()
    assert eta is not None and 5.0 <= eta <= 20.0
    # Finished tasks have no ETA.
    reg.finish(t)
    assert t.eta_s() is None


def test_cancel_only_flags_running_tasks():
    reg = tr.TaskRegistry()
    t = reg.create("e", "pg_sweep", "s", total=3)
    assert reg.cancel(t.id) is True
    assert t.cancel_requested is True
    reg.finish(t, state=tr.STATE_CANCELLED)
    assert reg.cancel(t.id) is False          # no longer running
    assert reg.cancel("nonexistent") is False


def test_snapshot_filters_by_entry():
    reg = tr.TaskRegistry()
    reg.create("A", "reprocess", "a")
    reg.create("B", "reprocess", "b")
    assert len(reg.snapshot()) == 2
    assert len(reg.snapshot("A")) == 1
    assert reg.snapshot("A")[0]["entry_id"] == "A"


def test_finished_tasks_are_evicted_running_kept():
    reg = tr.TaskRegistry()
    running = reg.create("e", "pg_sweep", "keep-me", total=100)
    for i in range(tr._MAX_FINISHED + 10):
        f = reg.create("e", "pg_history", f"done-{i}")
        reg.finish(f)
    # Running task survives; finished count is capped.
    assert reg.get(running.id) is not None
    finished = [t for t in reg.snapshot() if t["state"] != "running"]
    assert len(finished) <= tr._MAX_FINISHED
