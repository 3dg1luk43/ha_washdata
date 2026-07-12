"""Tests for Group C (Notification & Alerting Improvements) features C1-C3.

C1 — Quiet hours (do-not-disturb window): finish-type notifications are held during
     the window and released at its end. Live/start are never delayed.
C2 — Milestone (cycle-count achievement) notifications.
C3 — iOS Live Activity enrichment (subtitle / content_state / activity) for
     mobile_app_* targets only.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.const import (
    CONF_NOTIFY_MILESTONE_MESSAGE,
    CONF_NOTIFY_MILESTONES,
    CONF_NOTIFY_QUIET_END_HOUR,
    CONF_NOTIFY_QUIET_START_HOUR,
    DEFAULT_NOTIFY_MILESTONES,
    NOTIFY_EVENT_CLEAN,
    NOTIFY_EVENT_FINISH,
    NOTIFY_EVENT_LIVE,
    NOTIFY_EVENT_START,
)
from custom_components.ha_washdata.manager import WashDataManager


def _dt(hour: int, minute: int = 0) -> datetime:
    """Timezone-aware datetime on a fixed day at the given hour/minute."""
    return datetime(2026, 1, 1, hour, minute, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures / stub builders
# ---------------------------------------------------------------------------


def _quiet_stub(start: Any = None, end: Any = None) -> Any:
    """Lightweight manager stub exposing the real quiet-hours helpers."""
    m = MagicMock()
    m.config_entry.options = {}
    if start is not None:
        m.config_entry.options[CONF_NOTIFY_QUIET_START_HOUR] = start
    if end is not None:
        m.config_entry.options[CONF_NOTIFY_QUIET_END_HOUR] = end
    m._quiet_hours_bounds = WashDataManager._quiet_hours_bounds.__get__(
        m, WashDataManager
    )
    m._in_quiet_hours = WashDataManager._in_quiet_hours.__get__(m, WashDataManager)
    m._seconds_until_quiet_end = WashDataManager._seconds_until_quiet_end.__get__(
        m, WashDataManager
    )
    return m


def _milestone_stub(
    cycle_count: int,
    milestones: Any = "unset",
    finish_services: tuple[str, ...] = ("notify.mobile_app_x",),
    actions: list | None = None,
) -> Any:
    """Manager stub exposing the real _maybe_notify_milestone + _safe_format_template."""
    m = MagicMock()
    m.config_entry.title = "Washer"
    m.config_entry.options = {}
    if milestones != "unset":
        m.config_entry.options[CONF_NOTIFY_MILESTONES] = milestones
    m._notify_finish_services = list(finish_services)
    m._notify_actions = actions or []
    m._lifecycle_tag = "tag_life"
    m._logger = MagicMock()
    m.cycle_count = cycle_count
    # Milestones now key off the persisted monotonic lifetime counter (which never
    # regresses on history trim/merge), not cycle_count == len(history).
    m._lifetime_cycle_count = MagicMock(return_value=cycle_count)
    m._milestone_crossed = WashDataManager._milestone_crossed  # staticmethod
    m._safe_format_template = WashDataManager._safe_format_template.__get__(
        m, WashDataManager
    )
    m._dispatch_notification = MagicMock()
    m._maybe_notify_milestone = WashDataManager._maybe_notify_milestone.__get__(
        m, WashDataManager
    )
    return m


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
    # Non-"notify" domain so _send_notification_service uses the domain.service path.
    hass.states.get = MagicMock(return_value=MagicMock(state="home"))
    return hass


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Washer"
    entry.options = {"power_sensor": "sensor.test_power"}
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
        return mgr


# ===========================================================================
# C1 — Quiet hours
# ===========================================================================


def test_c1_quiet_hours_off_when_unset():
    assert _quiet_stub()._in_quiet_hours(_dt(3)) is False


def test_c1_quiet_hours_off_when_only_start_set():
    assert _quiet_stub(start=22)._in_quiet_hours(_dt(23)) is False


def test_c1_quiet_hours_off_when_start_equals_end():
    # start == end is a zero-length window -> feature off.
    stub = _quiet_stub(start=8, end=8)
    assert stub._in_quiet_hours(_dt(8)) is False
    assert stub._in_quiet_hours(_dt(3)) is False


def test_c1_quiet_hours_off_when_malformed():
    assert _quiet_stub(start="oops", end=7)._in_quiet_hours(_dt(3)) is False
    assert _quiet_stub(start=25, end=7)._in_quiet_hours(_dt(3)) is False
    assert _quiet_stub(start=22, end=-1)._in_quiet_hours(_dt(3)) is False


def test_c1_quiet_hours_normal_window():
    stub = _quiet_stub(start=1, end=6)  # 01:00-05:59
    assert stub._in_quiet_hours(_dt(0)) is False
    assert stub._in_quiet_hours(_dt(1)) is True  # start inclusive
    assert stub._in_quiet_hours(_dt(5)) is True
    assert stub._in_quiet_hours(_dt(6)) is False  # end exclusive
    assert stub._in_quiet_hours(_dt(12)) is False


def test_c1_quiet_hours_midnight_wrapping_window():
    stub = _quiet_stub(start=22, end=7)  # 22:00-06:59
    assert stub._in_quiet_hours(_dt(21)) is False
    assert stub._in_quiet_hours(_dt(22)) is True  # start inclusive
    assert stub._in_quiet_hours(_dt(23)) is True
    assert stub._in_quiet_hours(_dt(0)) is True  # across midnight
    assert stub._in_quiet_hours(_dt(6)) is True
    assert stub._in_quiet_hours(_dt(7)) is False  # end exclusive
    assert stub._in_quiet_hours(_dt(12)) is False


def test_c1_seconds_until_quiet_end_wrap():
    stub = _quiet_stub(start=22, end=7)
    # 23:00 -> next 07:00 is 8h away.
    assert stub._seconds_until_quiet_end(_dt(23)) == pytest.approx(8 * 3600)
    # 00:30 (past midnight) -> 07:00 same day is 6.5h away.
    assert stub._seconds_until_quiet_end(_dt(0, 30)) == pytest.approx(6.5 * 3600)


def test_c1_seconds_until_quiet_end_normal():
    stub = _quiet_stub(start=1, end=6)
    assert stub._seconds_until_quiet_end(_dt(2)) == pytest.approx(4 * 3600)


def test_c1_seconds_until_quiet_end_zero_when_off_or_outside():
    assert _quiet_stub()._seconds_until_quiet_end(_dt(3)) == 0.0
    # Outside the window -> nothing to wait for.
    assert _quiet_stub(start=1, end=6)._seconds_until_quiet_end(_dt(9)) == 0.0


def test_c1_finish_queued_during_quiet_hours(
    manager: WashDataManager, mock_hass: Any
) -> None:
    manager._notify_finish_services = ["notify.mobile_app_x"]
    manager.config_entry.options.update(
        {CONF_NOTIFY_QUIET_START_HOUR: 22, CONF_NOTIFY_QUIET_END_HOUR: 7}
    )
    with patch(
        "custom_components.ha_washdata.manager.dt_util.now", return_value=_dt(23)
    ), patch(
        "custom_components.ha_washdata.manager.async_call_later",
        return_value=MagicMock(),
    ) as acl:
        result = manager._dispatch_notification(
            "done", event_type=NOTIFY_EVENT_FINISH, extra_vars={"tag": "t"}
        )
    assert result is False
    assert len(manager._quiet_pending_notifications) == 1
    assert manager._quiet_pending_notifications[0]["message"] == "done"
    mock_hass.services.async_call.assert_not_called()
    acl.assert_called_once()  # a single release timer was armed
    assert manager._remove_quiet_hours_timer is not None


def test_c1_clean_and_precomplete_also_gated(
    manager: WashDataManager, mock_hass: Any
) -> None:
    manager._notify_finish_services = ["notify.mobile_app_x"]
    manager.config_entry.options.update(
        {CONF_NOTIFY_QUIET_START_HOUR: 22, CONF_NOTIFY_QUIET_END_HOUR: 7}
    )
    with patch(
        "custom_components.ha_washdata.manager.dt_util.now", return_value=_dt(23)
    ), patch(
        "custom_components.ha_washdata.manager.async_call_later",
        return_value=MagicMock(),
    ):
        manager._dispatch_notification(
            "nag", event_type=NOTIFY_EVENT_CLEAN, extra_vars={"tag": "c"}
        )
        manager._dispatch_notification(
            "reminder", event_type="pre_complete", extra_vars={"tag": "p"}
        )
    assert len(manager._quiet_pending_notifications) == 2
    mock_hass.services.async_call.assert_not_called()


def test_c1_live_not_gated_by_quiet_hours(
    manager: WashDataManager, mock_hass: Any
) -> None:
    manager._notify_live_services = ["notify.mobile_app_x"]
    manager.config_entry.options.update(
        {CONF_NOTIFY_QUIET_START_HOUR: 22, CONF_NOTIFY_QUIET_END_HOUR: 7}
    )
    with patch(
        "custom_components.ha_washdata.manager.dt_util.now", return_value=_dt(23)
    ):
        manager._dispatch_notification(
            "live",
            event_type=NOTIFY_EVENT_LIVE,
            extra_vars={"tag": "t", "live_update": True},
        )
    assert manager._quiet_pending_notifications == []
    mock_hass.services.async_call.assert_called_once()


def test_c1_start_not_gated_by_quiet_hours(
    manager: WashDataManager, mock_hass: Any
) -> None:
    manager._notify_start_services = ["notify.mobile_app_x"]
    manager.config_entry.options.update(
        {CONF_NOTIFY_QUIET_START_HOUR: 22, CONF_NOTIFY_QUIET_END_HOUR: 7}
    )
    with patch(
        "custom_components.ha_washdata.manager.dt_util.now", return_value=_dt(23)
    ):
        manager._dispatch_notification("started", event_type=NOTIFY_EVENT_START)
    assert manager._quiet_pending_notifications == []
    mock_hass.services.async_call.assert_called_once()


def test_c1_finish_not_gated_outside_quiet_hours(
    manager: WashDataManager, mock_hass: Any
) -> None:
    manager._notify_finish_services = ["notify.mobile_app_x"]
    manager.config_entry.options.update(
        {CONF_NOTIFY_QUIET_START_HOUR: 22, CONF_NOTIFY_QUIET_END_HOUR: 7}
    )
    with patch(
        "custom_components.ha_washdata.manager.dt_util.now", return_value=_dt(12)
    ):
        manager._dispatch_notification(
            "done", event_type=NOTIFY_EVENT_FINISH, extra_vars={"tag": "t"}
        )
    assert manager._quiet_pending_notifications == []
    mock_hass.services.async_call.assert_called_once()


def test_c1_flush_delivers_queued(
    manager: WashDataManager, mock_hass: Any
) -> None:
    manager._notify_finish_services = ["notify.mobile_app_x"]
    manager._quiet_pending_notifications = [
        {
            "message": "done",
            "title": "T",
            "icon": None,
            "event_type": NOTIFY_EVENT_FINISH,
            "extra_vars": {"tag": "t"},
        }
    ]
    manager._remove_quiet_hours_timer = MagicMock()
    with patch(
        "custom_components.ha_washdata.manager.dt_util.now", return_value=_dt(8)
    ):
        manager._flush_quiet_hours_notifications()
    assert manager._quiet_pending_notifications == []
    assert manager._remove_quiet_hours_timer is None
    mock_hass.services.async_call.assert_called_once()
    _domain, service, payload = mock_hass.services.async_call.call_args[0]
    assert service == "mobile_app_x"
    assert payload["message"] == "done"
    assert payload["data"]["tag"] == "t"


def test_c1_shutdown_cancels_quiet_timer(manager: WashDataManager) -> None:
    cancel = MagicMock()
    manager._remove_quiet_hours_timer = cancel
    manager._cancel_quiet_hours_timer()
    cancel.assert_called_once()
    assert manager._remove_quiet_hours_timer is None


def test_c1_single_timer_for_multiple_queued(manager: WashDataManager) -> None:
    manager._notify_finish_services = ["notify.mobile_app_x"]
    manager.config_entry.options.update(
        {CONF_NOTIFY_QUIET_START_HOUR: 22, CONF_NOTIFY_QUIET_END_HOUR: 7}
    )
    with patch(
        "custom_components.ha_washdata.manager.dt_util.now", return_value=_dt(23)
    ), patch(
        "custom_components.ha_washdata.manager.async_call_later",
        return_value=MagicMock(),
    ) as acl:
        for _ in range(3):
            manager._dispatch_notification(
                "done", event_type=NOTIFY_EVENT_FINISH, extra_vars={"tag": "t"}
            )
    assert len(manager._quiet_pending_notifications) == 3
    # One shared release timer, not one per queued notification.
    acl.assert_called_once()


# ===========================================================================
# C2 — Milestone notifications
# ===========================================================================


def test_c2_milestone_crossed_at_exact_threshold():
    assert WashDataManager._milestone_crossed(49, 50, [50, 100, 500, 1000]) == 50


def test_c2_milestone_not_recrossed():
    # Already past 50 -> 50->51 does not re-fire.
    assert WashDataManager._milestone_crossed(50, 51, [50, 100, 500, 1000]) is None


def test_c2_milestone_multiple_thresholds():
    assert WashDataManager._milestone_crossed(99, 100, [50, 100]) == 100
    assert WashDataManager._milestone_crossed(499, 500, [50, 100, 500]) == 500


def test_c2_milestone_returns_largest_when_several_cross():
    # Artificial multi-step jump crosses both 50 and 51 -> largest wins.
    assert WashDataManager._milestone_crossed(48, 52, [50, 51]) == 51


def test_c2_milestone_empty_list_noop():
    assert WashDataManager._milestone_crossed(49, 50, []) is None
    assert WashDataManager._milestone_crossed(49, 50, None) is None


def test_c2_milestone_malformed_entries_ignored():
    assert WashDataManager._milestone_crossed(49, 50, ["x", None, 50]) == 50
    # No valid positive milestone at the crossed count.
    assert WashDataManager._milestone_crossed(49, 50, ["x", -5, 0]) is None


def test_c2_maybe_notify_fires_at_milestone():
    m = _milestone_stub(cycle_count=50)  # default milestones include 50
    assert m._maybe_notify_milestone() == 50
    m._dispatch_notification.assert_called_once()
    _args, kwargs = m._dispatch_notification.call_args
    assert kwargs["event_type"] == NOTIFY_EVENT_FINISH
    assert kwargs["extra_vars"]["cycle_count"] == 50
    # Distinct tag so it does not clobber the finish lifecycle thread.
    assert kwargs["extra_vars"]["tag"].endswith("_milestone")
    # Message rendered from the default template with {device}/{cycle_count}.
    assert m._dispatch_notification.call_args[0][0] == "Washer has completed 50 cycles!"


def test_c2_maybe_notify_no_fire_off_milestone():
    m = _milestone_stub(cycle_count=51)
    assert m._maybe_notify_milestone() is None
    m._dispatch_notification.assert_not_called()


def test_c2_maybe_notify_uses_default_milestones_when_unset():
    m = _milestone_stub(cycle_count=100)  # milestones key absent -> defaults
    assert 100 in DEFAULT_NOTIFY_MILESTONES
    assert m._maybe_notify_milestone() == 100


def test_c2_maybe_notify_custom_milestones():
    m = _milestone_stub(cycle_count=25, milestones=[25, 75])
    assert m._maybe_notify_milestone() == 25


def test_c2_maybe_notify_empty_list_noop():
    m = _milestone_stub(cycle_count=50, milestones=[])
    assert m._maybe_notify_milestone() is None
    m._dispatch_notification.assert_not_called()


def test_c2_maybe_notify_requires_delivery_channel():
    m = _milestone_stub(cycle_count=50, finish_services=(), actions=[])
    assert m._maybe_notify_milestone() is None
    m._dispatch_notification.assert_not_called()


def test_c2_maybe_notify_custom_message_template():
    m = _milestone_stub(cycle_count=50)
    m.config_entry.options[CONF_NOTIFY_MILESTONE_MESSAGE] = "{device}: {cycle_count}!"
    m._maybe_notify_milestone()
    assert m._dispatch_notification.call_args[0][0] == "Washer: 50!"


# ===========================================================================
# C3 — iOS Live Activity enrichment
# ===========================================================================


def test_c3_content_state_fields():
    extras = WashDataManager._build_ios_live_activity_extras(
        state="running",
        progress_pct=42.7,
        eta_timestamp=1700000000,
        program="Cotton 60",
        device="Washer",
    )
    cs = extras["content_state"]
    assert cs["state"] == "running"
    assert cs["progress_pct"] == 43  # rounded to int
    assert cs["eta_timestamp"] == 1700000000
    assert cs["program"] == "Cotton 60"
    assert cs["device"] == "Washer"


def test_c3_subtitle_only_when_program_matched():
    with_prog = WashDataManager._build_ios_live_activity_extras(
        state="running",
        progress_pct=10,
        eta_timestamp=1,
        program="Eco 40",
        device="Washer",
    )
    assert with_prog["subtitle"] == "Eco 40"

    no_prog = WashDataManager._build_ios_live_activity_extras(
        state="running",
        progress_pct=10,
        eta_timestamp=1,
        program=None,
        device="Washer",
    )
    assert "subtitle" not in no_prog
    assert no_prog["content_state"]["program"] == ""


def test_c3_activity_marker_optional():
    started = WashDataManager._build_ios_live_activity_extras(
        state="running",
        progress_pct=0,
        eta_timestamp=1,
        program="P",
        device="D",
        activity="start",
    )
    assert started["activity"] == "start"
    none = WashDataManager._build_ios_live_activity_extras(
        state="running", progress_pct=0, eta_timestamp=1, program="P", device="D"
    )
    assert "activity" not in none


def test_c3_progress_pct_clamped():
    over = WashDataManager._build_ios_live_activity_extras(
        state="running", progress_pct=150, eta_timestamp=1, program="P", device="D"
    )
    assert over["content_state"]["progress_pct"] == 100
    under = WashDataManager._build_ios_live_activity_extras(
        state="running", progress_pct=-10, eta_timestamp=1, program="P", device="D"
    )
    assert under["content_state"]["progress_pct"] == 0


def test_c3_progress_pct_garbage_defaults_zero():
    bad = WashDataManager._build_ios_live_activity_extras(
        state="running", progress_pct="nope", eta_timestamp=1, program="P", device="D"
    )
    assert bad["content_state"]["progress_pct"] == 0


def test_c3_mobile_service_extras_included_for_mobile():
    ev = {
        "tag": "t",
        "subtitle": "Cotton 60",
        "content_state": {"state": "running"},
        "activity": "start",
        "progress": 30,  # not a mobile-only key -> excluded here
    }
    out = WashDataManager._mobile_service_extras(ev, "notify.mobile_app_pixel")
    assert out["subtitle"] == "Cotton 60"
    assert out["content_state"] == {"state": "running"}
    assert out["activity"] == "start"
    assert out["tag"] == "t"
    assert "progress" not in out  # only forwarded via the LIVE base whitelist


def test_c3_mobile_service_extras_omitted_for_non_mobile():
    ev = {
        "tag": "t",
        "subtitle": "Cotton 60",
        "content_state": {"state": "running"},
        "activity": "start",
    }
    assert WashDataManager._mobile_service_extras(ev, "notify.family_room") == {}
    assert WashDataManager._mobile_service_extras(ev, None) == {}


def test_c3_live_payload_enriched_for_mobile(
    manager: WashDataManager, mock_hass: Any
) -> None:
    manager._notify_live_services = ["notify.mobile_app_pixel"]
    manager._dispatch_notification(
        "live",
        event_type=NOTIFY_EVENT_LIVE,
        extra_vars={
            "tag": "t",
            "progress": 30,
            "progress_max": 100,
            "live_update": True,
            "subtitle": "Cotton 60",
            "content_state": {"state": "running", "progress_pct": 30},
            "activity": "start",
        },
    )
    _domain, service, payload = mock_hass.services.async_call.call_args[0]
    assert service == "mobile_app_pixel"
    data = payload["data"]
    assert data["subtitle"] == "Cotton 60"
    assert data["content_state"] == {"state": "running", "progress_pct": 30}
    assert data["activity"] == "start"


def test_c3_finish_activity_omitted_for_non_mobile(
    manager: WashDataManager, mock_hass: Any
) -> None:
    manager._notify_finish_services = ["notify.family_room"]
    manager._dispatch_notification(
        "done",
        event_type=NOTIFY_EVENT_FINISH,
        extra_vars={"tag": "t", "activity": "end"},
    )
    _domain, service, payload = mock_hass.services.async_call.call_args[0]
    assert service == "family_room"
    data = payload.get("data", {})
    assert "activity" not in data


def test_c3_finish_activity_included_for_mobile(
    manager: WashDataManager, mock_hass: Any
) -> None:
    manager._notify_finish_services = ["notify.mobile_app_x"]
    manager._dispatch_notification(
        "done",
        event_type=NOTIFY_EVENT_FINISH,
        extra_vars={"tag": "t", "activity": "end"},
    )
    _domain, service, payload = mock_hass.services.async_call.call_args[0]
    assert service == "mobile_app_x"
    assert payload["data"]["activity"] == "end"
