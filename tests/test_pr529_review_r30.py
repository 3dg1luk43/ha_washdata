"""Review of PR #529, round 30: the Minor findings, each reproduced.

* A manual analysis with no cadence withdrew the operational entries, then a
  later pass raised: nothing pushed the non-polling Suggested Settings sensor,
  so it kept showing the withdrawn entry.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from custom_components.ha_washdata.const import CONF_WATCHDOG_INTERVAL, DOMAIN
from custom_components.ha_washdata.suggestion_engine import OWNER_OPERATIONAL


class _FailingEngine:
    """A suggestion engine snapshot whose every scan raises."""

    def __getattr__(self, _name: str) -> Any:
        def fail(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError("scan failed")

        return fail


async def test_a_failed_manual_analysis_still_pushes_its_withdrawal(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    entry = await setup_washdata_entry()
    manager = hass.data[DOMAIN][entry.entry_id]
    lm = manager.learning_manager
    store = manager.profile_store
    store.set_suggestion(CONF_WATCHDOG_INTERVAL, 95, reason="r", flags={"owner": OWNER_OPERATIONAL})
    manager.notify_update()
    await hass.async_block_till_done()
    entity_id = er.async_get(hass).async_get_entity_id("sensor", DOMAIN, f"{entry.entry_id}_suggestions")
    assert hass.states.get(entity_id).state == "1"
    assert lm.cadence() is None  # so the operational pass withdraws instead of running

    client = await hass_ws_client(hass)
    with patch.object(lm.suggestion_engine, "for_job", return_value=_FailingEngine()):
        await client.send_json_auto_id(
            {"type": f"{DOMAIN}/run_suggestion_analysis", "entry_id": entry.entry_id}
        )
        res = await client.receive_json()
    await hass.async_block_till_done()
    assert res["success"] is False and res["error"]["code"] == "unknown_error"
    assert CONF_WATCHDOG_INTERVAL not in store.get_suggestions()
    assert hass.states.get(entity_id).state == "0"


async def test_an_entry_that_becomes_a_dishwasher_runs_the_cut_short_rebuild(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """A washer entry skips the one-time #531 rebuild at setup; a later device type
    change to dishwasher reloads in place and never ran it, so its first cycle read
    the old drying estimates."""
    from unittest.mock import AsyncMock  # noqa: PLC0415

    entry = await setup_washdata_entry()
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    await hass.async_block_till_done()
    assert store.cut_short_rebuild_pending()  # skipped: not a dishwasher

    real = store.async_rebuild_for_cut_short_rule
    with patch.object(
        store, "async_rebuild_for_cut_short_rule", AsyncMock(side_effect=real)
    ) as rebuild:
        hass.config_entries.async_update_entry(
            entry, options={**entry.options, "device_type": "dishwasher"}
        )
        await hass.async_block_till_done()
        assert manager.device_type == "dishwasher"
        assert rebuild.await_count == 1
        assert not store.cut_short_rebuild_pending()

        # Another reload, same type: the marker is set, nothing runs again.
        hass.config_entries.async_update_entry(
            entry, options={**entry.options, "off_delay": 200}
        )
        await hass.async_block_till_done()
    assert rebuild.await_count == 1


async def test_foreign_stop_rows_do_not_evict_the_last_local_one(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """A history merge brought 60 newer Stop rows from another device; the next
    settings save gave them every slot past the newest 50, which the advisory
    ignores, and dropped the local change it reads the threshold in force from."""
    from datetime import datetime, timedelta, timezone  # noqa: PLC0415

    from custom_components.ha_washdata.const import (  # noqa: PLC0415
        CONF_STOP_THRESHOLD_W,
        SETTINGS_CHANGELOG_IMPORTED_KEY,
    )
    from custom_components.ha_washdata.suggestion_engine import (  # noqa: PLC0415
        stop_threshold_changes,
    )

    t0 = datetime(2026, 11, 1, 8, tzinfo=timezone.utc)
    entry = await setup_washdata_entry()
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    await store.async_record_settings_changes(
        [{"key": CONF_STOP_THRESHOLD_W, "old": 5.0, "new": 2.0, "timestamp": t0.isoformat()}]
    )
    store._data["settings_changelog"].extend(  # noqa: SLF001 - as a merge leaves it
        {"key": CONF_STOP_THRESHOLD_W, "old": 9.0, "new": 8.0,
         "timestamp": (t0 + timedelta(hours=1 + i)).isoformat(),
         SETTINGS_CHANGELOG_IMPORTED_KEY: True}
        for i in range(60)
    )
    await store.async_record_settings_changes([{"key": "off_delay", "old": 1, "new": 2}])
    log = store.get_settings_changelog()
    assert len(log) == store.SETTINGS_CHANGELOG_MAX + 1  # the newest 50, then the local row
    assert stop_threshold_changes(log) == [(t0, 5.0)]
