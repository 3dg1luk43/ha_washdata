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
"""#478: the automation template library, downloaded from its own repository.

The templates themselves are validated in that repository's CI (inside a
WashData checkout). Here: the parser the panel's list comes from, that a
downloaded file can never reach this server's files, and the
``ha_washdata/blueprints/list`` command (online features only, cached).
"""
from __future__ import annotations

import json

import pytest

from custom_components.ha_washdata import blueprint_catalog, store_account
from custom_components.ha_washdata.const import AUTOMATION_LIBRARY_RAW, DOMAIN

CATALOG_URL = f"{AUTOMATION_LIBRARY_RAW}/catalog.json"
META = {"device_types": ["dryer"], "min_version": "0.5.9", "tags": ["smart plug"], "icon": "plug"}

GOOD = """blueprint:
  name: "WashData: cut the plug"
  description: >-
    Cuts the plug.


    Needs WashData 0.5.9 or newer.
  domain: automation
  author: Alex
  source_url: https://github.com/3dg1luk43/washdata-automations/blob/main/automation/good.yaml
  input:
    washdata_device:
      name: Dryer
      selector:
        device:
          filter:
            - integration: ha_washdata
    timing:
      name: Timing
      input:
        wait: {name: Wait, default: 5, selector: {number: {}}}
triggers:
  - trigger: ha_washdata.cycle_finished
    target:
      device_id: !input washdata_device
actions: []
"""
NEEDS_INPUT = GOOD.replace("default: 5, ", "")


def _url(bp_id: str) -> str:
    return f"{AUTOMATION_LIBRARY_RAW}/automation/{bp_id}.yaml"


def _serve(aioclient_mock, catalog: dict, files: dict[str, str]) -> None:
    aioclient_mock.get(CATALOG_URL, text=json.dumps(catalog))
    for bp_id, text in files.items():
        aioclient_mock.get(_url(bp_id), text=text)


# ─── The parser ──────────────────────────────────────────────────────────────


def test_describe_reads_a_template() -> None:
    item = blueprint_catalog.describe("good", GOOD, META, "0.5.9")
    assert item["name"] == "WashData: cut the plug"
    assert item["description"].startswith("Cuts the plug.")
    assert item["author"] == "Alex"
    assert item["icon"] == "plug"
    assert item["device_types"] == ["dryer"]
    assert item["compatible"] is True
    assert item["tags"] == ["smart plug"]
    assert item["path"] == "ha_washdata/good.yaml"
    assert item["yaml"] == GOOD
    assert [i["key"] for i in item["inputs"]] == ["washdata_device", "wait"]
    assert item["one_click"] is True
    assert blueprint_catalog.describe("good", NEEDS_INPUT, META)["one_click"] is False


@pytest.mark.parametrize("tag", ["!include secrets.yaml", "!secret api_key", "!env_var HOME", "!include_dir_list x"])
def test_describe_never_reads_this_servers_files(tag: str) -> None:
    """A downloaded file is parsed with !input only: HA's loader would resolve
    these against /config and show the result to every signed-in user."""
    with pytest.raises(Exception):
        blueprint_catalog.describe("bad", GOOD.replace("author: Alex", f"author: {tag}"), META)


def test_describe_cuts_long_text_and_drops_odd_values() -> None:
    text = GOOD.replace("author: Alex", "author: " + "A" * 200).replace(
        'name: "WashData: cut the plug"', "name: [1, 2]"
    )
    meta = {**META, "icon": "skull", "tags": ["x" * 50, "b", "c", "d", 5], "device_types": "dryer"}
    item = blueprint_catalog.describe("odd", text, meta)
    assert item["name"] == "odd"
    assert len(item["author"]) == blueprint_catalog.AUTHOR_MAX
    assert item["icon"] == ""
    assert len(item["tags"]) == blueprint_catalog.TAGS_MAX
    assert len(item["tags"][0]) == blueprint_catalog.TAG_MAX
    assert item["device_types"] == ["all"]
    long = GOOD.replace("Cuts the plug.", "word " * 400)
    assert len(blueprint_catalog.describe("long", long, META)["description"]) == blueprint_catalog.DESCRIPTION_MAX


@pytest.mark.parametrize("url", ["javascript:alert(1)", "JavaScript:alert(1)", "http://example.com/x.yaml", "data:text/html,x"])
def test_describe_drops_a_source_url_that_is_not_https(url: str) -> None:
    """The panel links it ("View on GitHub"); only an https URL may reach that href."""
    text = GOOD.replace(
        "https://github.com/3dg1luk43/washdata-automations/blob/main/automation/good.yaml", url
    )
    assert blueprint_catalog.describe("good", text, META)["source_url"] == ""
    assert blueprint_catalog.describe("good", GOOD, META)["source_url"].startswith("https://github.com/")


def test_describe_flags_a_template_for_a_newer_washdata() -> None:
    meta = {**META, "min_version": "0.6.0"}
    assert blueprint_catalog.describe("good", GOOD, meta, "0.5.9")["compatible"] is False
    assert blueprint_catalog.describe("good", GOOD, meta, "0.6.0")["compatible"] is True
    assert blueprint_catalog.describe("good", GOOD, meta, None)["compatible"] is True


@pytest.mark.parametrize("text", [
    "blueprint: [\n",
    GOOD.replace("domain: automation", "domain: script"),
    GOOD.replace("washdata_device:", "device:").replace("!input washdata_device", "!input device"),
])
def test_describe_rejects_what_is_not_a_washdata_automation(text: str) -> None:
    with pytest.raises(Exception):
        blueprint_catalog.describe("bad", text, META)


# ─── The WS command ──────────────────────────────────────────────────────────


async def test_list_is_off_without_online_features(
    hass, setup_washdata_entry, hass_ws_client, aioclient_mock
) -> None:
    """The library is on GitHub: nothing is downloaded until online features are on."""
    await setup_washdata_entry()
    client = await hass_ws_client(hass)
    await client.send_json_auto_id({"type": f"{DOMAIN}/blueprints/list"})
    res = await client.receive_json()
    assert res["success"], res
    assert res["result"] == {"blueprints": [], "disabled": True}
    assert aioclient_mock.call_count == 0


async def test_list_downloads_the_library(
    hass, setup_washdata_entry, hass_ws_client, hass_read_only_access_token, aioclient_mock
) -> None:
    """Any signed-in user gets the documented shape, in catalog order; a broken
    file, a bad id and a missing file are skipped, never requested out of the folder."""
    await setup_washdata_entry()
    await store_account.async_set_online(hass, True)
    _serve(
        aioclient_mock,
        {"good": META, "broken": META, "../escape": META, "missing": META, "second": {**META, "icon": "light"}},
        {"good": GOOD, "broken": "blueprint: [\n", "second": NEEDS_INPUT},
    )
    aioclient_mock.get(_url("missing"), status=404)
    client = await hass_ws_client(hass, hass_read_only_access_token)
    await client.send_json_auto_id({"type": f"{DOMAIN}/blueprints/list"})
    res = await client.receive_json()
    assert res["success"], res
    result = res["result"]
    assert set(result) == {"blueprints"}
    assert [b["id"] for b in result["blueprints"]] == ["good", "second"]
    for item in result["blueprints"]:
        assert set(item) == {
            "id", "name", "description", "author", "icon", "device_types", "min_version",
            "compatible", "tags", "source_url", "path", "yaml", "one_click", "inputs",
        }
        for inp in item["inputs"]:
            assert set(inp) == {"key", "name", "description", "selector", "has_default"}
    assert not [c for c in aioclient_mock.mock_calls if "escape" in str(c[1])]


async def test_list_is_cached_and_refreshes_at_most_once_a_minute(
    hass, setup_washdata_entry, hass_ws_client, aioclient_mock, monkeypatch
) -> None:
    await setup_washdata_entry()
    await store_account.async_set_online(hass, True)
    _serve(aioclient_mock, {"good": META}, {"good": GOOD})
    clock = [1000.0]
    monkeypatch.setattr(blueprint_catalog.time, "monotonic", lambda: clock[0])
    client = await hass_ws_client(hass)

    async def _list(**kw):
        await client.send_json_auto_id({"type": f"{DOMAIN}/blueprints/list", **kw})
        res = await client.receive_json()
        assert res["success"], res
        return res["result"]

    await _list()
    assert aioclient_mock.call_count == 2
    await _list()
    await _list(refresh=True)  # under a minute: the cached list
    assert aioclient_mock.call_count == 2
    clock[0] += blueprint_catalog.REFRESH_MIN_S
    await _list(refresh=True)
    assert aioclient_mock.call_count == 4
    clock[0] += blueprint_catalog.CACHE_TTL_S
    await _list()
    assert aioclient_mock.call_count == 6


async def test_an_unreachable_library_keeps_the_last_good_list(
    hass, setup_washdata_entry, hass_ws_client, aioclient_mock, monkeypatch
) -> None:
    await setup_washdata_entry()
    await store_account.async_set_online(hass, True)
    clock = [1000.0]
    monkeypatch.setattr(blueprint_catalog.time, "monotonic", lambda: clock[0])
    client = await hass_ws_client(hass)

    async def _list():
        await client.send_json_auto_id({"type": f"{DOMAIN}/blueprints/list"})
        res = await client.receive_json()
        assert res["success"], res
        return res["result"]

    aioclient_mock.get(CATALOG_URL, status=500)
    assert await _list() == {"blueprints": [], "error": "unreachable"}
    aioclient_mock.clear_requests()
    _serve(aioclient_mock, {"good": META}, {"good": GOOD})
    assert (await _list())["error"] == "unreachable"  # the failure is cached briefly
    clock[0] += blueprint_catalog.ERROR_TTL_S
    assert [b["id"] for b in (await _list())["blueprints"]] == ["good"]
    aioclient_mock.clear_requests()
    aioclient_mock.get(CATALOG_URL, text="x" * (blueprint_catalog.CATALOG_MAX_BYTES + 1))
    clock[0] += blueprint_catalog.CACHE_TTL_S
    result = await _list()
    assert result["error"] == "unreachable"
    assert [b["id"] for b in result["blueprints"]] == ["good"]
