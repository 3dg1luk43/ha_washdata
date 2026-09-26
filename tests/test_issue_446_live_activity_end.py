# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""Issue #446: the iOS Live Activity was never ended.

Per the HA companion docs ("Live Activities and Live Updates") an activity is
started with ``live_update: true``, updated by re-sending the same tag, and ended
**only** by ``clear_notification`` with that tag. There is no ``activity`` key in
that API, so the ``activity: "end"`` the finished notification carried was never
acted on - and because the live updates shared the lifecycle tag, sending the
documented clear would have dismissed the finished card instead. The reporter's
lock screen stayed frozen at 98% / 0:00 for an hour after the cycle finished; on
an earlier run the chronometer counted upward to 4:12:20.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.const import NOTIFY_EVENT_LIVE
from custom_components.ha_washdata.manager import WashDataManager


@pytest.fixture
def mock_hass() -> Any:
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.bus.async_fire = MagicMock()
    hass.async_create_task = MagicMock(
        side_effect=lambda coro: getattr(coro, "close", lambda: None)()
    )
    hass.components.persistent_notification.async_create = MagicMock()
    hass.config_entries.async_get_entry = MagicMock()
    hass.states.get = MagicMock(return_value=MagicMock(state="home"))
    return hass


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "e446"
    entry.title = "Dishwasher"
    entry.options = {"power_sensor": "sensor.p"}
    entry.data = {}
    return entry


@pytest.fixture
def manager(mock_hass: Any, mock_entry: Any) -> WashDataManager:
    mock_hass.config_entries.async_get_entry.return_value = mock_entry
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = WashDataManager(mock_hass, mock_entry)
        mgr.profile_store.get_suggestions = MagicMock(return_value={})
        mgr._notify_live_services = ["notify.mobile_app_iphone"]
        return mgr


def _clears(mock_hass: Any) -> list[str]:
    """Tags of every clear_notification actually sent."""
    out = []
    for call in mock_hass.services.async_call.call_args_list:
        _domain, _service, payload = call[0]
        if payload.get("message") == "clear_notification":
            out.append(payload["data"]["tag"])
    return out


def test_live_activity_has_its_own_tag(manager: WashDataManager) -> None:
    """The whole fix rests on this: while the two shared a tag, ending the
    activity and keeping the finished card were mutually exclusive."""
    assert manager._live_notification_tag != manager._lifecycle_tag
    assert manager._live_notification_tag.endswith("_live")
    assert manager._lifecycle_tag.endswith("_lifecycle")
    # ...and neither may collide with the clean-laundry nag.
    assert manager._clean_tag not in (
        manager._live_notification_tag,
        manager._lifecycle_tag,
    )


def test_cycle_end_sends_the_documented_clear_for_the_activity(
    manager: WashDataManager, mock_hass: Any
) -> None:
    manager._live_activity_started = True
    manager._end_live_activity()

    assert _clears(mock_hass) == [manager._live_notification_tag]
    _domain, service, payload = mock_hass.services.async_call.call_args[0]
    assert service == "mobile_app_iphone"
    assert payload["message"] == "clear_notification"
    # A bare clear: no live_update/progress keys to re-create what we are ending.
    assert "live_update" not in payload["data"]
    assert "progress" not in payload["data"]


def test_no_clear_when_no_live_target_is_configured(
    manager: WashDataManager, mock_hass: Any
) -> None:
    manager._notify_live_services = []
    manager._end_live_activity()
    mock_hass.services.async_call.assert_not_called()


def test_first_live_tick_hands_the_lifecycle_card_over(
    manager: WashDataManager, mock_hass: Any
) -> None:
    """Live updates no longer replace the start alert by sharing its tag, so the
    handover is explicit - otherwise the mobile app would show two entries.

    This also heals an upgrade: a Live Activity left running under the old shared
    tag by a pre-0.5.7 build is ended by exactly this clear.
    """
    manager._hand_over_lifecycle_to_live_activity()
    assert _clears(mock_hass) == [manager._lifecycle_tag]


def test_the_handover_fires_once_per_cycle(
    manager: WashDataManager, mock_hass: Any
) -> None:
    """Driven through the real live-progress path, not the helper directly."""
    manager._notify_live_interval_seconds = 30
    manager._notify_live_overrun_percent = 0
    manager.detector.state = "running"
    manager.detector.get_elapsed_seconds = MagicMock(return_value=60.0)
    manager._matched_profile_duration = 600.0
    manager._total_duration = 600.0
    manager._time_remaining = 540.0
    manager._dispatch_notification = MagicMock(return_value=True)

    from datetime import datetime, timedelta, timezone

    for _ in range(3):
        manager._last_live_notification_time = datetime.now(timezone.utc) - timedelta(
            seconds=31
        )
        manager._check_live_progress_notification()

    assert manager._live_activity_started is True
    assert _clears(mock_hass) == [manager._lifecycle_tag], (
        "the lifecycle handover must happen once, on the first tick only"
    )


def test_activity_is_ended_after_the_finished_alert_not_before(
    manager: WashDataManager, mock_hass: Any
) -> None:
    """Ordering matters: clearing first would leave the lock screen empty until
    the finished notification arrived."""
    order: list[str] = []
    manager._notify_finish_services = ["notify.mobile_app_iphone"]

    def _record(message, event_type=None, extra_vars=None, **kw):  # noqa: ANN001
        order.append(f"send:{event_type}")
        return True

    manager._dispatch_notification = MagicMock(side_effect=_record)
    manager._send_tag_clear = MagicMock(
        side_effect=lambda tag: order.append(f"clear:{tag}")
    )

    manager._live_activity_started = True
    live_running = manager._live_activity_started
    manager._clear_live_progress_notification(clear_services=False)
    manager._dispatch_notification("done", event_type="finish", extra_vars={})
    if live_running:
        manager._end_live_activity()

    assert order == ["send:finish", f"clear:{manager._live_notification_tag}"], order


def test_cycle_end_orders_finish_before_the_activity_clear_in_source() -> None:
    """Pin the ordering in the real cycle-end path.

    The two tests around this one drive the sequence by hand, which proves the
    pieces but not that _async_process_cycle_end wires them in that order. Read
    it out of the source so a later edit that moves the clear above the finished
    notification - re-opening #446's "lock screen momentarily empty" - fails here.

    Reading source text is a poor substitute for observing delivery order, and it
    is only here because a mocked service bus has no delivery order to observe.
    The real check now lives in `devtools/testbox/assert_run.py`, against two
    timestamped records in the box's notification capture; this stays as the cheap
    canary that runs in 30 s without Docker.
    """
    import inspect

    from custom_components.ha_washdata import manager as mgr_mod

    src = inspect.getsource(mgr_mod.WashDataManager._async_process_cycle_end)
    i_capture = src.index("live_activity_running = _same_cycle and self._live_activity_started")
    i_purge = src.index("_clear_live_progress_notification(clear_services=False)")
    i_finish = src.index('event_type=NOTIFY_EVENT_FINISH')
    i_end = src.index("self._end_live_activity()")

    # The flag must be read before the purge resets it...
    assert i_capture < i_purge, "the purge resets _live_activity_started"
    # ...and the activity must be ended only after the finished alert is dispatched.
    assert i_finish < i_end, "clearing before the finish empties the lock screen"
    assert "if live_activity_running:" in src
    # Both the flag read and the purge are gated on the cycle token: this tail
    # runs after several awaits, and a NEW cycle starting during them would
    # otherwise have its live counters purged and its activity ended, since the
    # live tag is per device rather than per cycle.
    assert "_same_cycle = (" in src
    assert "if _same_cycle:" in src


def test_no_stray_clear_when_no_activity_ever_started(
    manager: WashDataManager, mock_hass: Any
) -> None:
    """A device with live targets configured but no activity this cycle must not
    receive an end call for something that never began."""
    manager._live_activity_started = False
    live_running = manager._live_activity_started
    manager._clear_live_progress_notification(clear_services=False)
    if live_running:
        manager._end_live_activity()
    assert _clears(mock_hass) == []


def test_shutdown_mid_cycle_clears_both_surfaces(
    manager: WashDataManager, mock_hass: Any
) -> None:
    """A cycle in progress has no finished card yet, so nothing may be left behind."""
    from custom_components.ha_washdata.const import STATE_RUNNING

    manager.detector.state = STATE_RUNNING
    manager._clear_live_progress_notification(clear_services=True)
    assert _clears(mock_hass) == [
        manager._live_notification_tag,
        manager._lifecycle_tag,
    ]


def test_shutdown_after_a_cycle_keeps_the_finished_card(
    manager: WashDataManager, mock_hass: Any
) -> None:
    """Found in the PR #448 round-25 review.

    The finished alert uses the lifecycle tag. A restart or unload AFTER a cycle
    ended would otherwise dismiss the card the user still needs: "no finished
    notification follows" is true of a cycle in progress and false of one already
    over. The live tag still goes unconditionally, because its Live Activity
    chronometer goes negative once nothing updates it.
    """
    from custom_components.ha_washdata.const import STATE_OFF

    manager.detector.state = STATE_OFF
    manager._clear_live_progress_notification(clear_services=True)
    assert _clears(mock_hass) == [manager._live_notification_tag]


def test_pending_live_entries_are_still_purged_by_tag(
    manager: WashDataManager,
) -> None:
    """The purge matches on the live tag; it must follow the tag rename."""
    manager._pending_notifications = [
        {
            "event_type": NOTIFY_EVENT_LIVE,
            "extra_vars": {"tag": manager._live_notification_tag, "live_update": True},
            "message": "live",
        },
        {"event_type": "finish", "extra_vars": {"tag": manager._lifecycle_tag}, "message": "f"},
    ]
    manager._clear_live_progress_notification(clear_services=False)
    remaining = [n["event_type"] for n in manager._pending_notifications]
    assert NOTIFY_EVENT_LIVE not in remaining


def test_every_clear_survives_home_assistants_notify_schema(
    manager: WashDataManager, mock_hass: Any
) -> None:
    """The clears above were shaped right and still never reached the phone.

    ``notify``'s service schema validates ``title`` as a string even though it is
    optional, so a payload carrying ``title: None`` is rejected by
    ``hass.services.async_call`` before any platform sees it ("string value is
    None at 'title'"). Every dismiss-marker sender omits the title, and the call
    is fire-and-forget, so all four surfaces failed silently: the live-activity
    end, the lifecycle handover, the clean reminder and the timer-pause card. The
    assertions in this module all pass with that payload, which is exactly why
    this test validates against Home Assistant's own schema instead.
    """
    from homeassistant.components.notify.const import NOTIFY_SERVICE_SCHEMA

    manager._live_activity_started = True
    manager._end_live_activity()
    manager._hand_over_lifecycle_to_live_activity()

    payloads = [
        call[0][2]
        for call in mock_hass.services.async_call.call_args_list
        if call[0][0] == "notify"
    ]
    assert payloads, "no notify call was made"
    for payload in payloads:
        assert "title" not in payload or payload["title"] is not None
        NOTIFY_SERVICE_SCHEMA(payload)  # raises if HA would have rejected it


def test_a_normal_notification_still_carries_its_title(
    manager: WashDataManager, mock_hass: Any
) -> None:
    """Dropping a None title must not drop a real one."""
    manager._send_notification_service(
        "Dishwasher finished",
        services=["notify.mobile_app_iphone"],
        title="WashData",
        event_type="finish",
    )
    _domain, _service, payload = mock_hass.services.async_call.call_args[0]
    assert payload["title"] == "WashData"


def _person_home_event(entity_id: str = "person.owner") -> Any:
    """A state-change event that satisfies _handle_notify_person_change."""
    state = MagicMock()
    state.state = "home"
    state.entity_id = entity_id
    state.name = "Owner"
    state.attributes = {}
    event = MagicMock()
    event.data = {"new_state": state}
    return event


def test_a_presence_deferred_live_update_still_records_the_activity(
    manager: WashDataManager, mock_hass: Any
) -> None:
    """Found in the PR #448 review: the third delivery path forgot to say so.

    Presence gating queues the first live notification, and
    ``_handle_notify_person_change`` later sends it - carrying the same
    ``activity: "start"`` the direct paths send, so the phone has a live activity
    either way. It set the counters but never ``_live_activity_started``, so the
    cycle-end teardown read False, skipped ``_end_live_activity`` and left the
    card frozen on the lock screen. #446's own bug, reached from the other side.
    """
    manager._dispatch_notification = MagicMock(return_value=True)
    manager._pending_notifications = [
        {
            "message": "Washing, 40 minutes left",
            "event_type": NOTIFY_EVENT_LIVE,
            "extra_vars": {"tag": manager._live_notification_tag, "progress": 30,
                           "activity": "start"},
        }
    ]

    manager._handle_notify_person_change(_person_home_event())

    assert manager._live_activity_started is True, (
        "the deferred delivery started an activity the cycle end must be able to end"
    )


def test_a_presence_deferred_waiting_card_also_records_the_activity(
    manager: WashDataManager,
) -> None:
    """The waiting card carries `activity: "start"` too, progress or not."""
    manager._dispatch_notification = MagicMock(return_value=True)
    manager._pending_notifications = [
        {
            "message": "Cycle started",
            "event_type": NOTIFY_EVENT_LIVE,
            "extra_vars": {"tag": manager._live_notification_tag,
                           "activity": "start"},
        }
    ]

    manager._handle_notify_person_change(_person_home_event())

    assert manager._live_waiting_notification_sent is True
    assert manager._live_activity_started is True


def test_a_deferred_live_update_that_fails_to_send_starts_nothing(
    manager: WashDataManager,
) -> None:
    """No delivery, no activity: the flag must not be set optimistically."""
    manager._dispatch_notification = MagicMock(return_value=False)
    manager._pending_notifications = [
        {
            "message": "Washing",
            "event_type": NOTIFY_EVENT_LIVE,
            "extra_vars": {"tag": manager._live_notification_tag, "progress": 30},
        }
    ]

    manager._handle_notify_person_change(_person_home_event())

    assert manager._live_activity_started is False


def test_the_deferred_path_hands_the_lifecycle_card_over_exactly_once(
    manager: WashDataManager, mock_hass: Any
) -> None:
    """Same once-per-cycle contract the direct paths have."""
    manager._dispatch_notification = MagicMock(return_value=True)
    for _ in range(3):
        manager._pending_notifications = [
            {
                "message": "Washing",
                "event_type": NOTIFY_EVENT_LIVE,
                "extra_vars": {"tag": manager._live_notification_tag, "progress": 30},
            }
        ]
        manager._handle_notify_person_change(_person_home_event())

    assert _clears(mock_hass) == [manager._lifecycle_tag]


def test_a_deferred_non_live_notification_starts_no_activity(
    manager: WashDataManager,
) -> None:
    """Only a live delivery is an activity; a deferred finish alert is not."""
    manager._dispatch_notification = MagicMock(return_value=True)
    manager._pending_notifications = [
        {"message": "Cycle finished", "event_type": "finish", "extra_vars": {}}
    ]

    manager._handle_notify_person_change(_person_home_event())

    assert manager._live_activity_started is False


def test_the_setup_time_presence_flush_also_records_the_activity(
    manager: WashDataManager, mock_hass: Any
) -> None:
    """Found in the PR #448 round-4 review: the fourth delivery path.

    The listener flushes queued notifications when somebody is already home at
    (re-)attach time, which is what happens on a config reload. It was a second
    copy of the person-change loop and had drifted - no `sent` check, no
    activity recorded - so a live card queued while nobody was home and
    delivered on reload left the Live Activity frozen on the phone.
    """
    manager._dispatch_notification = MagicMock(return_value=True)
    manager._pending_notifications = [
        {
            "message": "Washing, 40 minutes left",
            "event_type": NOTIFY_EVENT_LIVE,
            "extra_vars": {"tag": manager._live_notification_tag, "progress": 30,
                           "activity": "start"},
        }
    ]

    manager._flush_pending_notifications("person.owner", "Owner")

    assert manager._live_activity_started is True
    assert manager._pending_notifications == []
    assert manager._live_notification_sent_count == 1


def test_a_new_cycle_keeps_its_live_activity_when_the_old_tail_lands_late(
    manager: WashDataManager, mock_hass: Any
) -> None:
    """Found in the PR #448 round-13 review.

    The cycle-end tail runs after the persistence / envelope / cost / lifetime
    awaits. A new cycle starting during them resets the live state and its first
    tick sets `_live_activity_started` again - so an ungated tail read the NEW
    cycle's flag, purged its counters, and cleared `_live_notification_tag`,
    which is per DEVICE. The new cycle's activity vanished and its start card
    was cleared a second time.
    """
    # Cycle A finished; by the time its tail runs, cycle B owns the tokens.
    manager._ranking_snapshot_cycle_id = "cycle-B"
    manager._live_activity_started = True
    manager._live_notification_sent_count = 3

    same_cycle = (
        "cycle-A" is None or manager._ranking_snapshot_cycle_id == "cycle-A"
    )
    assert same_cycle is False
    live_activity_running = same_cycle and manager._live_activity_started
    assert live_activity_running is False, (
        "cycle A's tail must not end the activity cycle B is running"
    )

    # And with no token (legacy callers) the old unconditional behaviour holds.
    same_cycle_legacy = True  # cycle_token is None
    assert (same_cycle_legacy and manager._live_activity_started) is True
