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
"""PR #529 review: what the suggestion passes report.

* The Suggested Settings sensor does not poll. A background pass (a cycle end's,
  the batch) or a refresh (startup, an options change) that changed the stored
  suggestions sent no entity update, so on an idle appliance the sensor kept the
  old count until some unrelated update.
* The manual "analyse now" reported success with the count of a partial run when
  one of its passes raised.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.dispatcher import async_dispatcher_connect

from custom_components.ha_washdata.const import CONF_OFF_DELAY, DOMAIN, SIGNAL_WASHER_UPDATE
from custom_components.ha_washdata.suggestion_engine import OWNER_MODEL, SuggestionEngine


async def _boot(hass: HomeAssistant, setup_washdata_entry: Any) -> tuple[Any, Any, list[int]]:
    entry = await setup_washdata_entry(options={CONF_OFF_DELAY: 120})
    await hass.async_block_till_done()
    manager = hass.data[DOMAIN][entry.entry_id]
    pushes: list[int] = []
    async_dispatcher_connect(
        hass, SIGNAL_WASHER_UPDATE.format(entry.entry_id), lambda: pushes.append(1)
    )
    return entry, manager, pushes


def _sensor_state(hass: HomeAssistant, entry: Any) -> str | None:
    entity_id = er.async_get(hass).async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_suggestions"
    )
    state = hass.states.get(entity_id) if entity_id else None
    return state.state if state else None


def _shown(manager: Any) -> str:
    """What the sensor would show if it were refreshed now."""
    from custom_components.ha_washdata.ws_api import _visible_suggestions  # noqa: PLC0415

    entry = manager.config_entry
    rows = _visible_suggestions(
        manager.profile_store, {**entry.data, **entry.options}, manager.device_type
    )
    return str(len(rows))


async def test_a_background_pass_that_changes_the_suggestions_pushes_the_sensor(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    entry, manager, pushes = await _boot(hass, setup_washdata_entry)
    lm = manager.learning_manager
    assert not manager.profile_store.get_suggestions()
    assert _sensor_state(hass, entry) == "0"

    def propose() -> dict[str, Any]:
        return {CONF_OFF_DELAY: {"value": 600, "reason": "r"}}

    await lm._run_pass(propose, OWNER_MODEL, lm._config_generation)  # noqa: SLF001
    await hass.async_block_till_done()
    assert CONF_OFF_DELAY in manager.profile_store.get_suggestions()
    assert pushes
    assert _shown(manager) != "0"
    assert _sensor_state(hass, entry) == _shown(manager)

    # The same result again changes nothing and pushes nothing.
    pushes.clear()
    await lm._run_pass(propose, OWNER_MODEL, lm._config_generation)  # noqa: SLF001
    await hass.async_block_till_done()
    assert pushes == []

    # The pass withdrawing what it owned is a change too.
    await lm._run_pass(dict, OWNER_MODEL, lm._config_generation)  # noqa: SLF001
    await hass.async_block_till_done()
    assert CONF_OFF_DELAY not in manager.profile_store.get_suggestions()
    assert pushes
    assert _sensor_state(hass, entry) == _shown(manager)


async def test_a_refresh_pushes_the_sensor_once_it_ends(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """Startup and an options change: what the sensor shows depends on the options
    too, so the refresh pushes whatever it changed."""
    _entry, manager, pushes = await _boot(hass, setup_washdata_entry)
    await manager.learning_manager.async_refresh_suggestions()
    await hass.async_block_till_done()
    assert pushes


async def test_no_push_after_an_unload(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    _entry, manager, pushes = await _boot(hass, setup_washdata_entry)
    lm = manager.learning_manager
    manager._is_shutdown = True  # noqa: SLF001
    try:
        await lm._run_pass(  # noqa: SLF001
            lambda: {CONF_OFF_DELAY: {"value": 600, "reason": "r"}},
            OWNER_MODEL, lm._config_generation,  # noqa: SLF001
        )
        await lm.async_refresh_suggestions()
        await hass.async_block_till_done()
    finally:
        manager._is_shutdown = False  # noqa: SLF001
    assert pushes == []


@pytest.mark.parametrize(
    "failing", ["generate_standby_floor_suggestions", "generate_detection_suggestions"]
)
async def test_a_manual_analysis_with_a_failed_pass_is_no_success(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any, failing: str
) -> None:
    """The passes after the failed one never ran: the panel gets the existing
    error reply ("Analysis failed: ..."), not a count of the partial run. The
    background refresh still never raises."""
    entry, manager, _pushes = await _boot(hass, setup_washdata_entry)
    client = await hass_ws_client(hass)
    with patch.object(SuggestionEngine, failing, side_effect=RuntimeError("scan broke")):
        await client.send_json_auto_id(
            {"type": f"{DOMAIN}/run_suggestion_analysis", "entry_id": entry.entry_id}
        )
        res = await client.receive_json()
        assert res["success"] is False
        assert res["error"]["code"] == "unknown_error"
        assert "scan broke" in res["error"]["message"]
        await manager.learning_manager.async_refresh_suggestions()  # never raises
    await client.send_json_auto_id(
        {"type": f"{DOMAIN}/run_suggestion_analysis", "entry_id": entry.entry_id}
    )
    ok = await client.receive_json()
    assert ok["success"] is True and "count" in ok["result"]
