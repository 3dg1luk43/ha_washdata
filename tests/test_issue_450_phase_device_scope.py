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
"""Regression tests for issue #450.

A phase created from the panel vanished from the catalog, yet creating it again
reported ``duplicate_phase``. Three independent links in that chain:

1. ``async_step_user`` passes no ``options``, so a device added after its last
   schema migration carries ``device_type`` in ``entry.data`` only.
2. ``ws_get_devices`` served bare ``entry.options``, so the panel fell back to
   ``'washing_machine'`` and sent that as the new phase's scope, while
   ``ws_get_phase_catalog`` listed against the manager's real device type.
3. Phases already stored under the wrong scope render nowhere and cannot be
   edited or deleted, but still occupy the name.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.config_flow import ConfigFlow
from custom_components.ha_washdata.const import (
    CONF_DEVICE_TYPE,
    CONF_MIN_POWER,
    CONF_NAME,
    CONF_POWER_SENSOR,
)
from custom_components.ha_washdata.phase_catalog import merge_phase_catalog
from custom_components.ha_washdata.profile_store import ProfileStore


@pytest.fixture
def mock_hass() -> MagicMock:
    hass = MagicMock()
    hass.data = {}

    async def _async_executor(func, *args, **kwargs):
        return func(*args, **kwargs)

    hass.async_add_executor_job = AsyncMock(side_effect=_async_executor)
    return hass


@pytest.fixture
def store(mock_hass: MagicMock) -> ProfileStore:
    with patch("custom_components.ha_washdata.profile_store.WashDataStore") as mock_store_cls:
        instance = ProfileStore(mock_hass, "test_entry")
        instance._store = mock_store_cls.return_value
        instance._store.async_load = AsyncMock(return_value=None)
        instance._store.async_save = AsyncMock()
        return instance


# ---------------------------------------------------------------------------
# Link 1 - a fresh entry keeps device_type in data only
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fresh_entry_has_device_type_in_data_only() -> None:
    """The setup flow passes no options, so entry.options starts empty.

    The data -> options backfill lives in ``async_migrate_entry``, which returns
    early for an entry already at the current version. Anything reading
    ``entry.options`` alone therefore misses the device type of every recently
    added device.
    """
    flow = ConfigFlow()
    captured: dict = {}
    flow.async_create_entry = lambda **kw: captured.update(kw) or {  # type: ignore[assignment]
        "type": "create_entry"
    }

    await flow.async_step_user(
        {
            CONF_NAME: "Dishwasher",
            CONF_DEVICE_TYPE: "dishwasher",
            CONF_POWER_SENSOR: "sensor.dw_power",
            CONF_MIN_POWER: 5.0,
        }
    )

    assert captured["data"][CONF_DEVICE_TYPE] == "dishwasher"
    assert "options" not in captured


# ---------------------------------------------------------------------------
# Link 2 - a mis-scoped phase is invisible but still blocks the name
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_phase_scoped_to_other_device_is_invisible_but_duplicate(
    store: ProfileStore,
) -> None:
    """The exact reported symptom pair, reproduced end to end."""
    await store.async_create_custom_phase("washing_machine", "Klarspulen")

    # Invisible in the catalog the dishwasher entry actually asks for ...
    dishwasher = [p["name"] for p in store.list_phase_catalog("dishwasher")]
    assert "Klarspulen" not in dishwasher

    # ... yet the name is taken, so the retry fails.
    with pytest.raises(ValueError, match="duplicate_phase"):
        await store.async_create_custom_phase("washing_machine", "Klarspulen")


def test_merge_filters_a_phase_scoped_to_a_different_device_type() -> None:
    """``merge_phase_catalog`` is where the phase drops out."""
    custom = [
        {
            "id": "x1",
            "name": "Klarspulen",
            "description": "",
            "device_type": "washing_machine",
            "is_default": False,
        }
    ]
    assert "Klarspulen" in [p["name"] for p in merge_phase_catalog("washing_machine", custom)]
    assert "Klarspulen" not in [p["name"] for p in merge_phase_catalog("dishwasher", custom)]


# ---------------------------------------------------------------------------
# Fix A - the WS create path resolves the scope itself
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ws_create_phase_falls_back_to_manager_device_type(
    store: ProfileStore,
) -> None:
    """An omitted or empty device_type uses the manager's, not a panel guess."""
    from custom_components.ha_washdata import ws_api

    manager = MagicMock()
    manager.device_type = "dishwasher"
    manager.profile_store = store

    hass = MagicMock()
    connection = MagicMock()

    with patch.object(ws_api, "_get_manager", return_value=manager):
        await ws_api.ws_create_phase.__wrapped__(
            hass,
            connection,
            {"id": 1, "entry_id": "e1", "device_type": "", "name": "Klarspulen"},
        )

    connection.send_error.assert_not_called()
    assert "Klarspulen" in [p["name"] for p in store.list_phase_catalog("dishwasher")]


@pytest.mark.asyncio
async def test_ws_create_phase_still_honours_an_explicit_device_type(
    store: ProfileStore,
) -> None:
    """A caller that does send a scope keeps it - the fallback is only a fallback."""
    from custom_components.ha_washdata import ws_api

    manager = MagicMock()
    manager.device_type = "dishwasher"
    manager.profile_store = store

    with patch.object(ws_api, "_get_manager", return_value=manager):
        await ws_api.ws_create_phase.__wrapped__(
            MagicMock(),
            MagicMock(),
            {"id": 1, "entry_id": "e1", "device_type": "dryer", "name": "Nachtrocknen"},
        )

    stored = store._get_shared_custom_phases()
    assert [p["device_type"] for p in stored if p["name"] == "Nachtrocknen"] == ["dryer"]


# ---------------------------------------------------------------------------
# Fix B - ws_get_devices serves merged data+options
# ---------------------------------------------------------------------------

def test_ws_get_devices_options_include_data_only_keys() -> None:
    """The panel's device_type read must survive an entry with empty options."""
    from custom_components.ha_washdata import ws_api

    entry = MagicMock()
    entry.entry_id = "e1"
    entry.title = "Dishwasher"
    entry.data = {
        CONF_NAME: "Creation Time Name",
        CONF_DEVICE_TYPE: "dishwasher",
        CONF_POWER_SENSOR: "sensor.dw_power",
    }
    entry.options = {}

    hass = MagicMock()
    hass.config_entries.async_entries.return_value = [entry]
    hass.data = {}

    sent: dict = {}
    connection = MagicMock()
    connection.user = None

    with patch.object(ws_api, "_effective_level", return_value="full"), patch.object(
        ws_api, "_send_result", side_effect=lambda _c, _i, _t, payload: sent.update(payload)
    ):
        ws_api.ws_get_devices(hass, connection, {"id": 1})

    options = sent["devices"][0]["options"]
    assert options[CONF_DEVICE_TYPE] == "dishwasher"
    # #422: the display name stays pinned to the live title, never the data fossil.
    assert options[CONF_NAME] == "Dishwasher"


# ---------------------------------------------------------------------------
# Fix C - already-stranded phases are re-scoped and become reachable
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_repair_rescopes_stranded_phases(store: ProfileStore) -> None:
    await store.async_create_custom_phase("washing_machine", "Klarspulen")

    repaired = await store.async_repair_custom_phase_scope("dishwasher")

    assert repaired == 1
    assert "Klarspulen" in [p["name"] for p in store.list_phase_catalog("dishwasher")]
    # The name is no longer double-booked: recreating it now reports the
    # duplicate against a phase the user can actually see.
    with pytest.raises(ValueError, match="duplicate_phase"):
        await store.async_create_custom_phase("dishwasher", "Klarspulen")


@pytest.mark.asyncio
async def test_repair_leaves_universal_phases_alone(store: ProfileStore) -> None:
    """An empty device_type means 'every device' and must not be narrowed."""
    await store.async_create_custom_phase("", "Vorspulen")

    assert await store.async_repair_custom_phase_scope("dishwasher") == 0
    assert store._get_shared_custom_phases()[0]["device_type"] == ""
    assert "Vorspulen" in [p["name"] for p in store.list_phase_catalog("dryer")]


@pytest.mark.asyncio
async def test_repair_is_idempotent_and_saves_only_on_change(
    store: ProfileStore,
) -> None:
    await store.async_create_custom_phase("dishwasher", "Klarspulen")
    store._store.async_save.reset_mock()

    assert await store.async_repair_custom_phase_scope("dishwasher") == 0
    store._store.async_save.assert_not_called()

    assert await store.async_repair_custom_phase_scope("") == 0
    store._store.async_save.assert_not_called()


@pytest.mark.asyncio
async def test_repair_does_not_create_two_phases_with_one_name(
    store: ProfileStore,
) -> None:
    """Found in the PR #448 round-21 review.

    A user who changes device type, recreates the name under the new type and
    then triggers the repair would get BOTH in one catalog. Rename and delete
    propagate to profile assignments by NAME
    (``async_delete_custom_phase`` filters on ``name.casefold()``), so deleting
    either would strip the assignments belonging to both.
    """
    await store.async_create_custom_phase("washing_machine", "Klarspulen")
    await store.async_create_custom_phase("dishwasher", "Klarspulen")

    repaired = await store.async_repair_custom_phase_scope("dishwasher")

    assert repaired == 0, "the stranded phase was re-scoped onto a name in use"
    names = [p["name"] for p in store.list_phase_catalog("dishwasher")]
    assert names.count("Klarspulen") == 1
    # Left unreachable, NOT deleted: a phase the user cannot see is recoverable,
    # one removed from under their profiles is not.
    assert len(store._get_shared_custom_phases()) == 2


@pytest.mark.asyncio
async def test_repair_does_not_collide_two_stranded_phases(
    store: ProfileStore,
) -> None:
    """The same guard has to hold within the batch, not just against what was
    already in the target scope."""
    await store.async_create_custom_phase("washing_machine", "Klarspulen")
    await store.async_create_custom_phase("dryer", "klarspulen")

    repaired = await store.async_repair_custom_phase_scope("dishwasher")

    assert repaired == 1
    names = [p["name"].casefold() for p in store.list_phase_catalog("dishwasher")]
    assert names.count("klarspulen") == 1


@pytest.mark.asyncio
async def test_repair_does_not_shadow_a_universal_phase(store: ProfileStore) -> None:
    """An unscoped phase is visible in every catalog, so a stranded phase with
    that name must not be pulled in alongside it."""
    await store.async_create_custom_phase("", "Vorspulen")
    store._get_shared_custom_phases().append(
        {"id": "x", "name": "Vorspulen", "description": "", "device_type": "dryer"}
    )

    assert await store.async_repair_custom_phase_scope("dishwasher") == 0
    names = [p["name"] for p in store.list_phase_catalog("dishwasher")]
    assert names.count("Vorspulen") == 1
