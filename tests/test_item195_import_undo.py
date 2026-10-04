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
"""Register item 195: "Undo last import".

A replace import (wholesale, or the selective wizard's replace / overwrite) used to
replace the device's stored data with no way back. It now writes the store as it was
to ``ha_washdata.<entry_id>.pre_import`` first; ``undo_import`` puts it back and
consumes the snapshot. Real Store, real hass storage for the store-level tests.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.ha_washdata import (
    _ENTRY_STORE_RE,
    _entry_store_keys,
    async_remove_entry,
    ws_api,
)
from custom_components.ha_washdata.const import (
    DOMAIN,
    PRE_IMPORT_STORE_SUFFIX,
    STORAGE_KEY,
    STORAGE_VERSION,
)
from custom_components.ha_washdata.profile_store import ProfileStore, WashDataStore

ENTRY = "01KBWSV8WQZHNZ0STZCPZGZ5K9"
MAIN = f"{STORAGE_KEY}.{ENTRY}"
SNAP = f"{STORAGE_KEY}.{ENTRY}.{PRE_IMPORT_STORE_SUFFIX}"


def _trace(watts: float, n: int = 31, dur: int = 1800) -> list[list[float]]:
    step = dur / (n - 1)
    return [[round(i * step, 1), float(watts)] for i in range(n)]


def _cycle(cid: str, profile: str | None, day: int, watts: float = 400.0) -> dict:
    return {
        "id": cid, "profile_name": profile, "duration": 1800.0, "status": "completed",
        "start_time": f"2026-03-{day:02d}T08:00:00+00:00", "power_data": _trace(watts),
        "label_source": "manual",
    }


def _mine() -> dict[str, Any]:
    """A store with something in every kind of user data an import can clobber."""
    return {
        "profiles": {"Mine": {"avg_duration": 1800.0, "sample_cycle_id": "m1"}},
        "past_cycles": [_cycle("m1", "Mine", 1), _cycle("m2", "Mine", 2), _cycle("m3", None, 3)],
        "reference_cycles": [_cycle("r1", "Mine", 4, 380.0)],
        "backfill_cycles": [_cycle("b1", None, 5, 410.0)],
        "envelopes": {},
        "feedback_history": {"m2": {"user_confirmed": True, "corrected_profile": "Mine"}},
        "pending_feedback": {"m3": {"cycle_id": "m3"}},
        "maintenance_log": [{"id": "x1", "type": "descale", "date": "2026-02-01"}],
        "custom_phases": [{"id": "p1", "name": "Rinse"}],
        "profile_groups": {},
        "suggestions": {"off_delay": {"value": 120}},
        "lifetime_cycle_count": 140,
        "lifetime_energy_wh": 52000.0,
    }


def _payload(*, device_type: str = "washing_machine") -> dict[str, Any]:
    return {
        "version": STORAGE_VERSION,
        "device_fingerprint": {"device_type": device_type},
        "data": {
            "profiles": {"Theirs": {"avg_duration": 1800.0}},
            "past_cycles": [_cycle("t1", "Theirs", 10, 900.0), _cycle("t2", "Theirs", 11, 900.0)],
            "reference_cycles": [],
            "envelopes": {},
        },
        "entry_data": {},
        "entry_options": {"off_delay": 300},
    }


async def _store(hass: HomeAssistant) -> tuple[ProfileStore, dict[str, Any]]:
    ps = ProfileStore(hass, ENTRY)
    await ps.async_load()
    ps._data = _mine()  # noqa: SLF001
    await ps.async_save()
    # What "the previous data" means: the store as it round-trips through JSON.
    return ps, json.loads(json.dumps(ps._data))  # noqa: SLF001


async def test_wholesale_replace_then_undo_restores_the_exact_previous_data(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    ps, before = await _store(hass)
    result = await ps.async_import_data(_payload(), entry_options={"off_delay": 120})
    assert result["restore_point_saved"] is True
    assert set(ps._data["profiles"]) == {"Theirs"}  # noqa: SLF001 - it did replace

    meta = await ps.async_get_pre_import_snapshot()
    assert meta["source"] == "import_config"
    assert meta["counts"] == {
        "profiles": 1, "real_cycles": 3, "reference_cycles": 1, "backfill_cycles": 1,
    }
    assert SNAP in hass_storage

    restored = await ps.async_restore_pre_import_snapshot()
    assert ps._data == before  # noqa: SLF001
    assert hass_storage[MAIN]["data"] == before
    assert restored["entry_options"] == {"off_delay": 120}
    assert restored["counts"]["real_cycles"] == 3

    # Consumed: no second undo over newer data, and the panel stops offering it.
    assert SNAP not in hass_storage
    assert await ps.async_get_pre_import_snapshot() is None
    with pytest.raises(ValueError):
        await ps.async_restore_pre_import_snapshot()


async def test_the_restore_point_survives_a_restart(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    ps, before = await _store(hass)
    await ps.async_import_data(_payload())
    reloaded = ProfileStore(hass, ENTRY)
    await reloaded.async_load()
    assert (await reloaded.async_get_pre_import_snapshot())["counts"]["real_cycles"] == 3
    restored = await reloaded.async_restore_pre_import_snapshot()
    assert reloaded._data == before  # noqa: SLF001
    # No options were captured, so the undo must leave them alone.
    assert restored["entry_options"] is None


async def test_selective_replace_then_undo_restores_the_exact_previous_data(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    ps, before = await _store(hass)
    summary = await ps.async_import_data_selective(
        _payload(),
        selection={"categories": ["profiles", "real_cycles"]},
        mode="replace",
        cycle_destination="real_history",
        local_device_type="washing_machine",
        entry_options={"off_delay": 120},
    )
    assert summary["restore_point_saved"] is True
    assert [c["id"] for c in ps._data["past_cycles"]] != ["m1", "m2", "m3"]  # noqa: SLF001
    assert (await ps.async_get_pre_import_snapshot())["source"] == "import_selective_replace"

    await ps.async_restore_pre_import_snapshot()
    assert ps._data == before  # noqa: SLF001


async def test_an_additive_merge_leaves_the_restore_point_alone(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    ps, _ = await _store(hass)
    summary = await ps.async_import_data_selective(
        _payload(), selection={"categories": ["profiles"]}, mode="merge",
        local_device_type="washing_machine",
    )
    assert summary["restore_point_saved"] is False
    assert SNAP not in hass_storage


async def test_a_merge_that_overwrites_a_profile_saves_one(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    ps, before = await _store(hass)
    payload = _payload()
    payload["data"]["profiles"] = {"Mine": {"avg_duration": 999.0}}
    summary = await ps.async_import_data_selective(
        payload, selection={"categories": ["profiles"]}, mode="merge",
        conflict_resolutions={"Mine": "overwrite"}, local_device_type="washing_machine",
    )
    assert summary["restore_point_saved"] is True
    assert ps._data["profiles"]["Mine"]["avg_duration"] == 999.0  # noqa: SLF001
    await ps.async_restore_pre_import_snapshot()
    assert ps._data == before  # noqa: SLF001


async def test_one_slot_the_next_replace_overwrites_it(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    ps, _ = await _store(hass)
    await ps.async_import_data(_payload())
    between = json.loads(json.dumps(ps._data))  # noqa: SLF001
    second = _payload()
    second["data"]["profiles"] = {"Other": {"avg_duration": 1800.0}}
    second["data"]["past_cycles"] = [_cycle("o1", "Other", 20)]
    await ps.async_import_data(second)
    await ps.async_restore_pre_import_snapshot()
    assert ps._data == between  # noqa: SLF001


async def test_a_refused_import_does_not_burn_the_restore_point(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    ps, before = await _store(hass)
    await ps.async_import_data(_payload())
    slot = copy.deepcopy(hass_storage[SNAP])
    with pytest.raises(ValueError):
        await ps.async_import_data({"version": STORAGE_VERSION, "data": {"profiles": {}}})
    assert hass_storage[SNAP] == slot
    await ps.async_restore_pre_import_snapshot()
    assert ps._data == before  # noqa: SLF001


async def test_undo_with_no_restore_point_errors_cleanly(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    ps, before = await _store(hass)
    assert await ps.async_get_pre_import_snapshot() is None
    with pytest.raises(ValueError, match="no import to undo"):
        await ps.async_restore_pre_import_snapshot()
    assert ps._data == before  # noqa: SLF001


async def test_an_older_snapshot_is_migrated_and_a_newer_one_refused(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    ps, _ = await _store(hass)
    old = _mine()

    def _slot(version: int) -> dict[str, Any]:
        return {"version": 1, "minor_version": 1, "key": SNAP, "data": {
            "created_at": "2026-10-01T10:00:00+00:00", "source": "import_config",
            "store_version": version, "counts": {}, "entry_options": None, "data": old,
        }}

    hass_storage[SNAP] = _slot(STORAGE_VERSION + 1)
    ps._pre_import_meta_loaded = False  # noqa: SLF001
    with pytest.raises(ValueError, match="newer version"):
        await ps.async_restore_pre_import_snapshot()

    # Refused, not consumed: a later upgrade can still use it.
    assert SNAP in hass_storage

    hass_storage[SNAP] = _slot(STORAGE_VERSION - 1)
    fresh = ProfileStore(hass, ENTRY)
    await fresh.async_load()
    migrated = {**old, "migrated": True}
    with patch.object(
        WashDataStore, "_async_migrate_func", AsyncMock(return_value=migrated)
    ) as mig:
        await fresh.async_restore_pre_import_snapshot()
    assert mig.await_args.args[:2] == (STORAGE_VERSION - 1, 1)
    assert fresh._data["migrated"] is True  # noqa: SLF001


async def test_the_snapshot_store_is_removed_with_the_entry(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    ps, _ = await _store(hass)
    await ps.async_import_data(_payload())
    assert SNAP in hass_storage
    assert SNAP in _entry_store_keys(ENTRY)
    assert ps._pre_import_store.key == SNAP  # noqa: SLF001
    assert _ENTRY_STORE_RE.match(SNAP).group(1) == ENTRY  # the orphan sweep finds it

    entry = MockConfigEntry(domain=DOMAIN, entry_id=ENTRY, data={})
    entry.add_to_hass(hass)
    await async_remove_entry(hass, entry)
    await hass.async_block_till_done()
    assert SNAP not in hass_storage
    assert MAIN not in hass_storage


# ── WS layer ────────────────────────────────────────────────────────────────


def _conn() -> MagicMock:
    c = MagicMock()
    c.send_result = MagicMock()
    c.send_error = MagicMock()
    return c


def _ws_hass() -> MagicMock:
    hass = MagicMock()
    hass.data = {}
    hass.config_entries.async_update_entry = MagicMock()
    return hass


def _manager(store: Any) -> MagicMock:
    m = MagicMock()
    m.profile_store = store
    m.notify_update = MagicMock()
    m.async_schedule_banked_tail_repair = MagicMock()
    return m


async def test_ws_undo_without_a_restore_point_is_a_clean_not_found() -> None:
    store = MagicMock()
    store.async_restore_pre_import_snapshot = AsyncMock(
        side_effect=ValueError("There is no import to undo")
    )
    manager, conn = _manager(store), _conn()
    with patch.object(ws_api, "_get_manager", return_value=manager):
        await ws_api.ws_undo_import.__wrapped__(_ws_hass(), conn, {"id": 7, "entry_id": "e"})
    conn.send_result.assert_not_called()
    assert conn.send_error.call_args.args[:2] == (7, "not_found")
    manager.notify_update.assert_not_called()


async def test_ws_undo_restores_options_but_keeps_this_devices_bindings() -> None:
    store = MagicMock()
    store.async_restore_pre_import_snapshot = AsyncMock(return_value={
        "restored_from": "2026-10-01T10:00:00+00:00",
        "counts": {"profiles": 1, "real_cycles": 3},
        "entry_options": {"off_delay": 120, "power_sensor": "sensor.old_plug"},
    })
    store.async_record_settings_changes = AsyncMock()
    manager, conn, hass = _manager(store), _conn(), _ws_hass()
    entry = SimpleNamespace(
        entry_id="e", data={},
        # off_delay + min_power came from the import; power_sensor was re-picked since.
        options={"off_delay": 300, "min_power": 9.0, "power_sensor": "sensor.new_plug"},
    )
    with patch.object(ws_api, "_get_manager", return_value=manager), \
         patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_api.ws_undo_import.__wrapped__(hass, conn, {"id": 3, "entry_id": "e"})
    conn.send_error.assert_not_called()
    payload = conn.send_result.call_args.args[1]
    assert payload == {"success": True, "summary": {
        "restored_from": "2026-10-01T10:00:00+00:00",
        "counts": {"profiles": 1, "real_cycles": 3},
    }}
    options = hass.config_entries.async_update_entry.call_args.kwargs["options"]
    assert options == {"off_delay": 120, "power_sensor": "sensor.new_plug"}
    manager.notify_update.assert_called_once()


async def test_ws_import_config_hands_over_the_options_and_reports_the_snapshot() -> None:
    store = MagicMock()
    store.async_import_data = AsyncMock(return_value={
        "entry_data": {}, "entry_options": {}, "restore_point_saved": True,
    })
    manager, conn, hass = _manager(store), _conn(), _ws_hass()

    async def _exec(func, *args):
        return func(*args)

    hass.async_add_executor_job = AsyncMock(side_effect=_exec)
    entry = SimpleNamespace(entry_id="e", data={}, options={"off_delay": 120})
    with patch.object(ws_api, "_get_manager", return_value=manager), \
         patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_api.ws_import_config.__wrapped__(
            hass, conn, {"id": 1, "entry_id": "e", "json_data": json.dumps(_payload())}
        )
    assert store.async_import_data.await_args.kwargs["entry_options"] == {"off_delay": 120}
    assert conn.send_result.call_args.args[1] == {"success": True, "restore_point_saved": True}


async def test_ws_diagnostics_carries_the_restore_point() -> None:
    meta = {"created_at": "2026-10-01T10:00:00+00:00", "source": "import_config",
            "counts": {"profiles": 1}}
    store = MagicMock()
    store.get_storage_stats = AsyncMock(return_value={"total_cycles": 3})
    store.async_get_pre_import_snapshot = AsyncMock(return_value=meta)
    conn = _conn()
    with patch.object(ws_api, "_get_manager", return_value=_manager(store)):
        await ws_api.ws_get_diagnostics.__wrapped__(_ws_hass(), conn, {"id": 2, "entry_id": "e"})
    assert conn.send_result.call_args.args[1] == {"stats": {"total_cycles": 3}, "import_undo": meta}


def test_undo_is_admin_only_even_with_rbac_off() -> None:
    hass = MagicMock()
    hass.data = {}
    conn = MagicMock()
    conn.user = SimpleNamespace(id="u1", is_admin=False)
    msg = {"id": 1, "type": "ha_washdata/undo_import", "entry_id": "e"}
    assert not ws_api._rbac_ok(hass, conn, msg)  # noqa: SLF001
    assert "undo_import" in ws_api._FULL_COMMANDS  # noqa: SLF001
