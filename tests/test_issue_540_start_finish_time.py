"""#540: ``{finish_time}`` in the Start Message.

The start message goes out the moment a cycle starts, before any program is
recognised. With a program picked by hand (armed for the cycle) the finish time
is its learned length from the start; otherwise the device's typical run, the
median of its recent completed runs, marked ``~`` as an estimate; empty with no
history at all. The live updates carry the matched estimate later.
"""
from __future__ import annotations

from datetime import timedelta

from homeassistant.util import dt as dt_util

from .real_manager import boot, feed, make_entry, record_notify

TEMPLATE = "{device} | {finish_time} |"


def _starts(calls: list[dict]) -> list[str]:
    return [c["message"] for c in calls if str(c.get("message", "")).startswith("Washer | ")]


def _past(start, seconds: float, status: str = "completed") -> dict:
    return {
        "start_time": start.isoformat(),
        "end_time": (start + timedelta(seconds=seconds)).isoformat(),
        "duration": seconds,
        "status": status,
        "energy_wh": 500.0,
        "power_data": [[0.0, 500.0], [seconds, 0.0]],
    }


def _clock(mgr, seconds: float) -> str:
    return dt_util.as_local(mgr._cycle_start_time + timedelta(seconds=seconds)).strftime("%H:%M")  # noqa: SLF001


async def test_no_history_leaves_it_empty(hass, freezer) -> None:
    calls = record_notify(hass)
    mgr = await boot(hass, make_entry(hass, {"notify_start_message": TEMPLATE}))
    await feed(hass, freezer, 500, 180)
    assert _starts(calls) == ["Washer |  |"]
    await mgr.async_shutdown()


async def test_the_typical_run_is_marked_as_an_estimate(hass, freezer) -> None:
    calls = record_notify(hass)
    mgr = await boot(hass, make_entry(hass, {"notify_start_message": TEMPLATE}))
    t0 = dt_util.utcnow() - timedelta(days=3)
    for i, (seconds, status) in enumerate(
        [(3600, "completed"), (5400, "completed"), (7200, "completed"), (600, "interrupted")]
    ):
        await mgr.profile_store.async_add_cycle(_past(t0 + timedelta(hours=4 * i), seconds, status))
    await feed(hass, freezer, 500, 180)
    # The median of the completed runs (the interrupted one does not count).
    assert _starts(calls) == [f"Washer | ~{_clock(mgr, 5400)} |"]
    await mgr.async_shutdown()


async def test_a_program_picked_by_hand_gives_its_own_length(hass, freezer) -> None:
    calls = record_notify(hass)
    mgr = await boot(hass, make_entry(hass, {"notify_start_message": TEMPLATE}))
    t0 = dt_util.utcnow() - timedelta(days=3)
    await mgr.profile_store.async_add_cycle(_past(t0, 3600))
    mgr.profile_store._data.setdefault("profiles", {})["Cotton"] = {"avg_duration": 7200.0}  # noqa: SLF001
    assert mgr.set_manual_program("Cotton")
    await feed(hass, freezer, 500, 180)
    assert _starts(calls) == [f"Washer | {_clock(mgr, 7200)} |"]
    await mgr.async_shutdown()


async def test_the_default_message_is_unchanged(hass, freezer) -> None:
    calls = record_notify(hass)
    mgr = await boot(hass, make_entry(hass))
    await feed(hass, freezer, 500, 180)
    assert [c["message"] for c in calls if "started" in str(c.get("message", ""))] == ["Washer started."]
    await mgr.async_shutdown()


async def test_a_restored_cycle_counts_from_its_own_start(hass, freezer) -> None:
    """The restart fallback can send the message before a power event has set the
    manager's start: the detector's restored start is used (PR #539 review)."""
    mgr = await boot(hass, make_entry(hass))
    now = dt_util.utcnow()
    mgr._cycle_start_time = None  # noqa: SLF001
    mgr.detector._current_cycle_start = now - timedelta(hours=1)  # noqa: SLF001
    mgr._matched_profile_duration = 7200.0  # noqa: SLF001
    expected = dt_util.as_local(now + timedelta(hours=1)).strftime("%H:%M")
    assert mgr._start_finish_time(now) == expected  # noqa: SLF001
    await mgr.async_shutdown()
