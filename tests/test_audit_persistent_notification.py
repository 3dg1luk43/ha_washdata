"""Audit 2026-10-02 PLATFORM-01: persistent notifications must reach Home Assistant.

``_pn_create`` used to go through ``hass.components.persistent_notification``.
Home Assistant removed ``hass.components``, so the getattr found nothing and the
helper returned silently: the sidebar fallback for users with no notify target
and the auto-pause timer card never appeared, while 12 test modules that mocked
``hass.components`` kept passing. These tests run against a real ``hass``.
"""

from __future__ import annotations

import re
from pathlib import Path

from homeassistant.components import persistent_notification
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component

from custom_components.ha_washdata.manager import _pn_create, _pn_dismiss

PACKAGE = Path(__file__).resolve().parents[1] / "custom_components" / "ha_washdata"


async def test_pn_create_and_dismiss_reach_a_real_hass(hass: HomeAssistant) -> None:
    assert await async_setup_component(hass, persistent_notification.DOMAIN, {})
    notifications = persistent_notification._async_get_or_create_notifications(hass)

    assert _pn_create(hass, "Washer finished", title="WashData", notification_id="wd_x")
    await hass.async_block_till_done()
    assert "wd_x" in notifications
    assert notifications["wd_x"]["message"] == "Washer finished"

    _pn_dismiss(hass, "wd_x")
    await hass.async_block_till_done()
    assert "wd_x" not in notifications


def test_no_module_reaches_for_hass_components() -> None:
    """``hass.components`` no longer exists; any use of it is dead code."""
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if re.search(r"hass\.components\b|getattr\([^)]*[\"']components[\"']", text):
            offenders.append(path.name)
    assert offenders == []
