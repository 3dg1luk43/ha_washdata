"""Tests for Group G1 - HA Conversation intents.

Fast, pure-unit tests that instantiate :class:`WashDataStatusIntentHandler` and
drive ``async_handle`` with a fabricated intent object + fake hass/managers
(mirroring the MagicMock style used elsewhere in the suite). No HA boot, no I/O.
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.core import Context
from homeassistant.helpers import intent as ha_intent
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata import intents
from custom_components.ha_washdata.const import (
    DOMAIN,
    STATE_CLEAN,
    STATE_FINISHED,
    STATE_OFF,
    STATE_RUNNING,
)
from custom_components.ha_washdata.intents import (
    INTENT_STATUS,
    WashDataStatusIntentHandler,
    async_setup_intents,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeHass:
    """Minimal hass stand-in: only ``.data`` is exercised by the handler."""

    def __init__(self, managers: dict | None = None) -> None:
        self.data = {DOMAIN: managers or {}}


def _make_manager(title, state, *, time_remaining=None, last_end=None):
    """Build a fake manager exposing the accessors the handler reads."""
    manager = MagicMock()
    manager.config_entry.title = title
    manager.check_state.return_value = state
    manager.time_remaining = time_remaining
    manager.last_cycle_end_time = last_end
    return manager


async def _run(hass, *, name=None, language="en", translations=None) -> str:
    """Invoke async_handle with a fabricated intent and return the speech text.

    The HA translation cache is stubbed (returns ``translations`` or nothing) so
    the handler behaves deterministically without a warmed cache.
    """
    slots = {"name": {"value": name}} if name is not None else {}
    intent_obj = ha_intent.Intent(
        hass=hass,
        platform="test",
        intent_type=INTENT_STATUS,
        slots=slots,
        text_input=None,
        context=Context(),
        language=language,
    )
    handler = WashDataStatusIntentHandler()
    with patch.object(
        intents.translation,
        "async_get_translations",
        AsyncMock(return_value=translations or {}),
    ):
        response = await handler.async_handle(intent_obj)
    return response.speech["plain"]["speech"]


# ---------------------------------------------------------------------------
# Response behaviour
# ---------------------------------------------------------------------------

async def test_running_with_estimate():
    hass = _FakeHass({"e1": _make_manager("Washer", STATE_RUNNING, time_remaining=1800)})
    speech = await _run(hass)
    assert "Washer" in speech
    assert "still running" in speech
    assert "30 minutes" in speech


async def test_running_without_estimate():
    hass = _FakeHass({"e1": _make_manager("Washer", STATE_RUNNING, time_remaining=None)})
    speech = await _run(hass)
    assert speech == "Your Washer is still running."
    assert "minutes" not in speech


async def test_idle_not_running():
    hass = _FakeHass({"e1": _make_manager("Washer", STATE_OFF, last_end=None)})
    speech = await _run(hass)
    assert speech == "Your Washer is not running."


async def test_finished_recently():
    end = dt_util.now() - timedelta(minutes=5)
    hass = _FakeHass({"e1": _make_manager("Dishwasher", STATE_FINISHED, last_end=end)})
    speech = await _run(hass)
    assert "Dishwasher" in speech
    assert "finished" in speech
    assert "5 minutes ago" in speech


async def test_finished_clean_state_without_timestamp_says_just_finished():
    hass = _FakeHass({"e1": _make_manager("Dryer", STATE_CLEAN, last_end=None)})
    speech = await _run(hass)
    assert speech == "Your Dryer just finished."


async def test_off_with_recent_end_reports_finished():
    """A completed cycle whose device already returned to OFF still reports."""
    end = dt_util.now() - timedelta(minutes=12)
    hass = _FakeHass({"e1": _make_manager("Washer", STATE_OFF, last_end=end)})
    speech = await _run(hass)
    assert "finished" in speech
    assert "12 minutes ago" in speech


async def test_off_with_stale_end_says_not_running():
    """An end far outside the recent window is treated as idle, not finished."""
    end = dt_util.now() - timedelta(hours=48)
    hass = _FakeHass({"e1": _make_manager("Washer", STATE_OFF, last_end=end)})
    speech = await _run(hass)
    assert speech == "Your Washer is not running."


async def test_unknown_device():
    hass = _FakeHass({"e1": _make_manager("Washer", STATE_RUNNING, time_remaining=600)})
    speech = await _run(hass, name="dryer")
    assert "dryer" in speech
    assert "couldn't find" in speech


async def test_no_devices():
    hass = _FakeHass({})
    speech = await _run(hass)
    assert speech == "I couldn't find any WashData appliances."


async def test_multi_device_selection_by_name_slot():
    managers = {
        "e1": _make_manager("Kitchen Dishwasher", STATE_RUNNING, time_remaining=1200),
        "e2": _make_manager("Laundry Washer", STATE_OFF, last_end=None),
    }
    hass = _FakeHass(managers)

    # Partial, case-insensitive name match picks the right device.
    dish = await _run(hass, name="dish")
    assert "Kitchen Dishwasher" in dish
    assert "still running" in dish
    assert "20 minutes" in dish

    washer = await _run(hass, name="WASHER")
    assert washer == "Your Laundry Washer is not running."


async def test_multi_device_no_name_summarizes_running():
    managers = {
        "e1": _make_manager("Kitchen Dishwasher", STATE_RUNNING, time_remaining=600),
        "e2": _make_manager("Laundry Washer", STATE_OFF, last_end=None),
    }
    hass = _FakeHass(managers)
    speech = await _run(hass)
    # Only the running device is summarized.
    assert "Kitchen Dishwasher" in speech
    assert "still running" in speech
    assert "Laundry Washer" not in speech


async def test_multi_device_no_name_none_running():
    managers = {
        "e1": _make_manager("Kitchen Dishwasher", STATE_OFF, last_end=None),
        "e2": _make_manager("Laundry Washer", STATE_OFF, last_end=None),
    }
    hass = _FakeHass(managers)
    speech = await _run(hass)
    assert speech == "None of your WashData appliances are running."


async def test_handler_never_raises_on_broken_manager():
    """A manager that raises from check_state must still yield a graceful reply."""
    broken = MagicMock()
    broken.config_entry.title = "Washer"
    broken.check_state.side_effect = RuntimeError("boom")
    broken.time_remaining = None
    broken.last_cycle_end_time = None
    hass = _FakeHass({"e1": broken})
    # _describe_device swallows the error -> falls through to "not running".
    speech = await _run(hass)
    assert speech == "Your Washer is not running."


async def test_translation_override_applied():
    """A localized template from the translation cache overrides the default."""
    hass = _FakeHass({"e1": _make_manager("Washer", STATE_RUNNING, time_remaining=None)})
    flat = {
        f"component.{DOMAIN}.intent.{INTENT_STATUS}.running_no_estimate": (
            "Die {device} laeuft noch."
        )
    }
    speech = await _run(hass, language="de", translations=flat)
    assert speech == "Die Washer laeuft noch."


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_async_setup_intents_registers_handler():
    hass = _FakeHass({})
    async_setup_intents(hass)
    registry = hass.data[ha_intent.DATA_KEY]
    assert INTENT_STATUS in registry
    assert isinstance(registry[INTENT_STATUS], WashDataStatusIntentHandler)


def test_slot_schema_accepts_optional_name():
    handler = WashDataStatusIntentHandler()
    # Optional name slot -> empty slots validate fine.
    assert handler.async_validate_slots({}) == {}
    validated = handler.async_validate_slots({"name": {"value": "washer"}})
    assert validated["name"]["value"] == "washer"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def test_manifest_declares_conversation_dependency():
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "conversation" in manifest["dependencies"]
    # Valid JSON with the expected shape.
    assert manifest["domain"] == DOMAIN
