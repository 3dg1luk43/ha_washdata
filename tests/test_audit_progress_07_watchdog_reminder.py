"""Audit 2026-10-02 PROGRESS-07: the "almost done" reminder during a silent tail.

The watchdog refreshes the remaining-time estimate while a publish-on-change plug
is silent (item 128), but only the power path checked the pre-completion reminder.
So when the remaining time crossed the lead inside the silence, the reminder went
out with the next reading, typically the final pump-out, alongside the finish.
The watchdog now runs the same ``notification_rules`` predicate after its refresh.

Runs a REAL manager (real ProfileStore, Store and detector).
"""

from __future__ import annotations

from .real_manager import boot, feed, idle_expire, make_entry, record_notify


def _reminders(calls: list[dict]) -> int:
    return sum("minutes left" in str(c.get("message")) for c in calls)


async def test_the_reminder_fires_while_the_plug_is_silent(hass, freezer):
    calls = record_notify(hass)
    entry = make_entry(hass, {"notify_before_end_minutes": 5})
    mgr = await boot(hass, entry)
    await feed(hass, freezer, 500, 300)  # 5 min into the cycle
    assert mgr.detector.state == "running"
    # Matched to a 15 min programme, as a committed live match leaves it.
    mgr._matched_profile_duration = 900.0  # noqa: SLF001
    mgr._current_program = "Cotton"  # noqa: SLF001
    assert not mgr._notified_pre_completion  # noqa: SLF001

    # Steady draw on a publish-on-change plug: no report at all from here. Only
    # the watchdog runs, and the 5 min lead is crossed at about 10 min elapsed.
    await idle_expire(hass, freezer, 480, step=30)

    assert mgr.detector.state == "running"
    assert mgr._notified_pre_completion  # noqa: SLF001
    assert _reminders(calls) == 1
    await mgr.async_shutdown()
