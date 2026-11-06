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
"""#507: only an administrator may change the services and actions WashData runs.

``notify_actions`` is a Home Assistant script and ``notify_*_services`` name the
services WashData calls. Both run with no user in the context, which passes the
check of an admin-only service, and ``set_options`` is open to every user while
RBAC is off (the default): a non-admin could run any admin-only service.
"""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers.service import async_register_admin_service

from custom_components.ha_washdata.const import DOMAIN


async def _send(client: Any, payload: dict[str, Any]) -> dict[str, Any]:
    await client.send_json_auto_id(payload)
    return await client.receive_json()


async def test_a_non_admin_cannot_change_what_notifications_run(
    hass: HomeAssistant, setup_washdata_entry, hass_ws_client,
    hass_access_token, hass_read_only_access_token,
) -> None:
    ran: list[str | None] = []

    async def _probe(call: ServiceCall) -> None:
        ran.append(call.context.user_id)

    async_register_admin_service(hass, "probe", "admin_only", _probe)
    entry = await setup_washdata_entry(
        options={"notify_finish_services": ["notify.phone"]}
    )
    admin = await hass_ws_client(hass, hass_access_token)
    user = await hass_ws_client(hass, hass_read_only_access_token)

    def set_options(client: Any, options: dict[str, Any]) -> Any:
        return _send(client, {
            "type": f"{DOMAIN}/set_options", "entry_id": entry.entry_id, "options": options,
        })

    for options in (
        {"notify_actions": [{"action": "probe.admin_only"}]},
        {"notify_finish_services": ["probe.admin_only"]},
        {"notify_start_services": ["probe.admin_only"], "off_delay": 120},
    ):
        res = await set_options(user, options)
        assert not res["success"] and res["error"]["code"] == "forbidden", options
    await hass.async_block_till_done()
    assert "notify_actions" not in entry.options
    assert entry.options["notify_finish_services"] == ["notify.phone"]
    assert "off_delay" not in entry.options

    # The rest of the form, with those keys unchanged, still saves.
    res = await set_options(user, {"off_delay": 120, "notify_finish_services": ["notify.phone"],
                                   "notify_actions": []})
    assert res["success"], res
    await hass.async_block_till_done()
    assert entry.options["off_delay"] == 120

    # RBAC on with full access to the device changes nothing for these keys.
    on = await _send(admin, {"type": f"{DOMAIN}/set_panel_config",
                             "rbac": {"enabled": True, "default_level": "full"}})
    assert on["success"], on
    res = await set_options(user, {"notify_actions": [{"action": "probe.admin_only"}]})
    assert not res["success"] and res["error"]["code"] == "forbidden"

    # An administrator still can.
    actions = [{"action": "probe.admin_only"}]
    res = await set_options(admin, {"notify_actions": actions})
    assert res["success"], res
    await hass.async_block_till_done()
    assert entry.options["notify_actions"] == actions
    assert ran == []
