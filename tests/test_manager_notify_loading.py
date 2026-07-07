"""Unit tests for WashDataManager._load_notify_services.

Covers the new per-event service lists (notify_start_services,
notify_finish_services, notify_live_services) and the backward-compat
migration from the deprecated notify_service + notify_events pair.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from custom_components.ha_washdata.const import (
    CONF_NOTIFY_EVENTS,
    CONF_NOTIFY_FINISH_SERVICES,
    CONF_NOTIFY_LIVE_SERVICES,
    CONF_NOTIFY_SERVICE,
    CONF_NOTIFY_START_SERVICES,
    NOTIFY_EVENT_FINISH,
    NOTIFY_EVENT_LIVE,
    NOTIFY_EVENT_START,
)
from custom_components.ha_washdata.manager import WashDataManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(options: dict) -> MagicMock:
    entry = MagicMock()
    entry.entry_id = "test_notify_entry"
    entry.title = "Test Washer"
    entry.options = options
    entry.data = {}
    return entry


def _make_manager(options: dict) -> WashDataManager:
    entry = _make_entry(options)
    hass = MagicMock()
    hass.data = {}
    hass.config_entries.async_get_entry = MagicMock(return_value=entry)
    with patch("custom_components.ha_washdata.manager.ProfileStore"), patch(
        "custom_components.ha_washdata.manager.CycleDetector"
    ):
        mgr = WashDataManager(hass, entry)
        mgr.profile_store.get_suggestions = MagicMock(return_value={})
    return mgr


# ---------------------------------------------------------------------------
# New API: per-event service lists
# ---------------------------------------------------------------------------


def test_load_notify_services_reads_finish_services():
    mgr = _make_manager({
        "power_sensor": "sensor.p",
        CONF_NOTIFY_FINISH_SERVICES: ["notify.mobile_app_phone"],
    })
    assert mgr._notify_finish_services == ["notify.mobile_app_phone"]
    assert mgr._notify_start_services == []
    assert mgr._notify_live_services == []


def test_load_notify_services_reads_start_services():
    mgr = _make_manager({
        "power_sensor": "sensor.p",
        CONF_NOTIFY_START_SERVICES: ["notify.alexa_media"],
    })
    assert mgr._notify_start_services == ["notify.alexa_media"]
    assert mgr._notify_finish_services == []


def test_load_notify_services_reads_live_services():
    mgr = _make_manager({
        "power_sensor": "sensor.p",
        CONF_NOTIFY_LIVE_SERVICES: ["notify.mobile_app_pixel"],
    })
    assert mgr._notify_live_services == ["notify.mobile_app_pixel"]


def test_load_notify_services_reads_all_three_lists():
    mgr = _make_manager({
        "power_sensor": "sensor.p",
        CONF_NOTIFY_START_SERVICES: ["notify.a"],
        CONF_NOTIFY_FINISH_SERVICES: ["notify.b", "notify.c"],
        CONF_NOTIFY_LIVE_SERVICES: ["notify.d"],
    })
    assert mgr._notify_start_services == ["notify.a"]
    assert mgr._notify_finish_services == ["notify.b", "notify.c"]
    assert mgr._notify_live_services == ["notify.d"]


def test_load_notify_services_none_value_treated_as_empty():
    mgr = _make_manager({
        "power_sensor": "sensor.p",
        CONF_NOTIFY_FINISH_SERVICES: None,
    })
    assert mgr._notify_finish_services == []


# ---------------------------------------------------------------------------
# Backward-compat: legacy notify_service + notify_events
# ---------------------------------------------------------------------------


def test_load_notify_compat_service_alone_populates_all_three():
    """Old single notify_service (no notify_events) → all three lists.

    When notify_events is absent, _old_events is empty/falsy, so
    ``not _old_events`` is True and every list is unconditionally populated.
    """
    mgr = _make_manager({
        "power_sensor": "sensor.p",
        CONF_NOTIFY_SERVICE: "notify.old_service",
    })
    assert mgr._notify_start_services == ["notify.old_service"]
    assert mgr._notify_finish_services == ["notify.old_service"]
    assert mgr._notify_live_services == ["notify.old_service"]


def test_load_notify_compat_service_with_live_event_populates_only_live():
    """Old notify_service + notify_events=[live] → only live list.

    notify_events restricts which events fire.  When set, each list is only
    populated if its event type appears in the list.
    """
    mgr = _make_manager({
        "power_sensor": "sensor.p",
        CONF_NOTIFY_SERVICE: "notify.old_service",
        CONF_NOTIFY_EVENTS: [NOTIFY_EVENT_LIVE],
    })
    assert mgr._notify_start_services == []
    assert mgr._notify_finish_services == []
    assert mgr._notify_live_services == ["notify.old_service"]


def test_load_notify_compat_service_with_finish_only_event():
    """Old notify_service + notify_events=[finish] → only finish list."""
    mgr = _make_manager({
        "power_sensor": "sensor.p",
        CONF_NOTIFY_SERVICE: "notify.old_service",
        CONF_NOTIFY_EVENTS: [NOTIFY_EVENT_FINISH],
    })
    assert mgr._notify_start_services == []
    assert mgr._notify_finish_services == ["notify.old_service"]
    assert mgr._notify_live_services == []


def test_load_notify_compat_skipped_when_new_lists_present():
    """New list present → legacy migration skipped entirely."""
    mgr = _make_manager({
        "power_sensor": "sensor.p",
        CONF_NOTIFY_SERVICE: "notify.old_service",
        CONF_NOTIFY_FINISH_SERVICES: ["notify.new_service"],
    })
    # Old service must not be injected alongside the new list
    assert mgr._notify_finish_services == ["notify.new_service"]
    assert "notify.old_service" not in mgr._notify_start_services
    assert "notify.old_service" not in mgr._notify_live_services


def test_load_notify_compat_empty_service_is_ignored():
    """Empty string notify_service → no migration."""
    mgr = _make_manager({
        "power_sensor": "sensor.p",
        CONF_NOTIFY_SERVICE: "",
        CONF_NOTIFY_EVENTS: [NOTIFY_EVENT_FINISH],
    })
    assert mgr._notify_finish_services == []
    assert mgr._notify_start_services == []
