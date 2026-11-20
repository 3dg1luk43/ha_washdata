"""Review of PR #529, round 41b, each reproduced.

* A skipped write of a retired store (its entry unloaded) logged through the
  module logger, without the appliance it belongs to.
"""
from __future__ import annotations

import logging
from typing import Any

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import DOMAIN

# ── A retired store's skipped write names its appliance ──────────────────────


async def test_a_skipped_write_of_a_retired_store_names_the_appliance(
    hass: HomeAssistant, setup_washdata_entry: Any, caplog: pytest.LogCaptureFixture
) -> None:
    entry = await setup_washdata_entry(title="Bosch Upstairs")
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    assert await hass.config_entries.async_unload(entry.entry_id)
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="custom_components.ha_washdata"):
        await store.async_save()
        store.coalesce_saves()
        await store.async_save()
    skipped = [r for r in caplog.records if "unloaded entry's store skipped" in r.getMessage()]
    assert len(skipped) == 2, caplog.text
    assert all(r.getMessage().startswith("[Bosch Upstairs] ") for r in skipped)
    assert all(getattr(r, "wd_device", None) == "Bosch Upstairs" for r in skipped)
