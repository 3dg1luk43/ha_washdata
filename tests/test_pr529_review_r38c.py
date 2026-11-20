"""Review of PR #529, round 38c: the Minor findings in ws_api.py, each reproduced.

* Clearing a stored Smart Termination ratio (or anti-crease ratio, curve
  pre-roll) to "" saved, but its reset removed the override before the
  null-strip, so the settings history had no row and it could not be reverted.
* The wholesale and the selective import refreshed their manager after the
  settings-history await even when a full reload had replaced it.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch

from homeassistant.core import HomeAssistant

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import (
    CONF_ANTI_CREASE_FINALIZE_RATIO,
    CONF_CURVE_PREROLL_SECONDS,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_SMART_TERMINATION_DURATION_RATIO,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    DOMAIN,
    STORAGE_VERSION,
)


async def _set_options(hass: HomeAssistant, client: Any, entry_id: str, options: dict) -> dict:
    await client.send_json_auto_id({
        "type": f"{DOMAIN}/set_options", "entry_id": entry_id, "options": options,
    })
    res = await client.receive_json()
    await hass.async_block_till_done()
    return res


async def test_a_reset_override_is_recorded_and_revertible(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """A 0.75 Smart Termination ratio cleared with "" (and the two other reset
    keys, one with None) is recorded as removed, so Settings History can put it
    back. A reset of a key with no override records nothing."""
    entry = await setup_washdata_entry(options={
        CONF_SMART_TERMINATION_DURATION_RATIO: 0.75,
        CONF_ANTI_CREASE_FINALIZE_RATIO: 0.9,
        CONF_CURVE_PREROLL_SECONDS: 30.0,
        CONF_OFF_DELAY: 120,
    })
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    client = await hass_ws_client(hass)
    res = await _set_options(hass, client, entry.entry_id, {
        CONF_SMART_TERMINATION_DURATION_RATIO: "",
        CONF_ANTI_CREASE_FINALIZE_RATIO: None,
        CONF_CURVE_PREROLL_SECONDS: "",
    })
    assert res["success"], res
    for key in (
        CONF_SMART_TERMINATION_DURATION_RATIO, CONF_ANTI_CREASE_FINALIZE_RATIO,
        CONF_CURVE_PREROLL_SECONDS,
    ):
        assert key not in entry.options
    rows = {r["key"]: r for r in store.get_settings_changelog()}
    assert rows[CONF_SMART_TERMINATION_DURATION_RATIO]["old"] == 0.75
    assert rows[CONF_SMART_TERMINATION_DURATION_RATIO]["new"] is None
    assert rows[CONF_ANTI_CREASE_FINALIZE_RATIO]["old"] == 0.9
    assert rows[CONF_ANTI_CREASE_FINALIZE_RATIO]["new"] is None
    assert rows[CONF_CURVE_PREROLL_SECONDS]["old"] == 30.0
    assert rows[CONF_CURVE_PREROLL_SECONDS]["new"] is None

    # Revert: the history's ``old`` written back puts the override in again.
    res = await _set_options(hass, client, entry.entry_id, {
        CONF_SMART_TERMINATION_DURATION_RATIO:
            rows[CONF_SMART_TERMINATION_DURATION_RATIO]["old"],
    })
    assert res["success"], res
    assert entry.options[CONF_SMART_TERMINATION_DURATION_RATIO] == 0.75

    # No override left to remove: no row.
    before = len(store.get_settings_changelog())
    res = await _set_options(hass, client, entry.entry_id, {
        CONF_ANTI_CREASE_FINALIZE_RATIO: "", CONF_CURVE_PREROLL_SECONDS: None,
    })
    assert res["success"], res
    assert len(store.get_settings_changelog()) == before


_PAIR = {CONF_MIN_POWER: 0.5, CONF_START_THRESHOLD_W: 10.0, CONF_STOP_THRESHOLD_W: 3.0}


class _Replacement:
    """The manager a full reload put in place; records what reached it."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: self.calls.append(name)


@contextmanager
def _reload_during_history_write(manager: Any) -> Iterator[tuple[_Replacement, Any]]:
    """Once the settings history is written, the entry runs another manager."""
    real_get, real_record = ws_api._get_manager, ws_api._record_option_changes  # noqa: SLF001
    replacement = _Replacement()
    retired = {"on": False}

    def get_manager(h: Any, entry_id: str) -> Any:
        return replacement if retired["on"] else real_get(h, entry_id)

    async def record_then_reload(*args: Any, **kwargs: Any) -> None:
        await real_record(*args, **kwargs)
        retired["on"] = True

    with (
        patch.object(ws_api, "_get_manager", side_effect=get_manager),
        patch.object(ws_api, "_record_option_changes", side_effect=record_then_reload),
        patch.object(manager, "notify_update") as notify,
    ):
        yield replacement, notify


async def _send(hass: HomeAssistant, client: Any, payload: dict) -> dict:
    await client.send_json_auto_id(payload)
    res = await client.receive_json()
    await hass.async_block_till_done()
    return res


async def test_import_does_not_refresh_a_retired_manager(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """The imported settings were written and a full reload replaced the manager
    while the history was: the retired one still refreshed. The import's result
    stands, the repairs go to the live one, and without a reload it refreshes."""
    entry = await setup_washdata_entry(options=_PAIR)
    manager = hass.data[DOMAIN][entry.entry_id]
    client = await hass_ws_client(hass)

    def payload(stop: float) -> dict:
        return {
            "type": f"{DOMAIN}/import_config", "entry_id": entry.entry_id,
            "json_data": json.dumps({
                "version": STORAGE_VERSION, "data": {"profiles": {"Eco": {"avg_duration": 3600}}},
                "entry_options": {**_PAIR, CONF_STOP_THRESHOLD_W: stop},
            }),
        }

    with _reload_during_history_write(manager) as (replacement, notify):
        res = await _send(hass, client, payload(1.0))
    assert res["success"], res
    assert res["result"]["success"] is True
    assert entry.options[CONF_STOP_THRESHOLD_W] == 1.0
    assert notify.call_count == 0
    assert "async_schedule_banked_tail_repair" in replacement.calls
    assert "notify_update" not in replacement.calls

    with patch.object(manager, "notify_update") as notify:
        res = await _send(hass, client, payload(2.0))
    assert res["success"], res
    assert notify.call_count == 1


async def test_selective_import_does_not_refresh_a_retired_manager(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """Same for the selective import's settings write."""
    entry = await setup_washdata_entry(options=_PAIR)
    manager = hass.data[DOMAIN][entry.entry_id]
    client = await hass_ws_client(hass)

    def payload(stop: float) -> dict:
        return {
            "type": f"{DOMAIN}/import_config_selective", "entry_id": entry.entry_id,
            "json_data": json.dumps({
                "version": STORAGE_VERSION, "entry_id": "another",
                "device_fingerprint": {"device_type": "washing_machine"},
                "data": {"profiles": {}},
                "entry_options": {CONF_STOP_THRESHOLD_W: stop},
            }),
            "selection": {"categories": ["settings"]}, "apply_settings": True,
        }

    with _reload_during_history_write(manager) as (replacement, notify):
        res = await _send(hass, client, payload(1.0))
    assert res["success"], res
    assert res["result"]["success"] is True
    assert res["result"]["summary"]["settings_applied"] == 1
    assert entry.options[CONF_STOP_THRESHOLD_W] == 1.0
    assert notify.call_count == 0
    assert "async_schedule_banked_tail_repair" in replacement.calls
    assert "notify_update" not in replacement.calls

    with patch.object(manager, "notify_update") as notify:
        res = await _send(hass, client, payload(2.0))
    assert res["success"], res
    assert notify.call_count == 1
