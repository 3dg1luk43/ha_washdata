"""Issue #480: a hot wash (or drum clean) reminder that knows when you ran one.

A reminder only ever restarted when it was logged by hand, so "run a hot wash
every month" needed the user to remember logging it too. A maintenance task,
built-in or custom, can now name programs: a finished cycle labelled with one of
them (a confident match or a hand-picked program) logs the task as done, dated at
the cycle's end, so its count restarts from that cycle.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import DOMAIN, MAINTENANCE_PROGRAMS_MAX
from custom_components.ha_washdata.profile_store import ProfileStore

POWER = "sensor.washer_power"
END = "2026-10-20T11:30:00+00:00"


def _store(hass: HomeAssistant, *names: str) -> ProfileStore:
    store = ProfileStore(hass, "e480")
    store._data["profiles"] = {n: {"avg_duration": 3600.0} for n in names}  # noqa: SLF001
    return store


def _cycle(program: str | None, cycle_id: str = "c1", end: str = END) -> dict:
    return {"id": cycle_id, "profile_name": program, "end_time": end, "status": "completed"}


async def test_programs_are_validated_and_stored(hass: HomeAssistant) -> None:
    store = _store(hass, "Cotton 90", "Eco")
    with patch.object(store, "async_save", AsyncMock()):
        assert await store.async_set_maintenance_programs(
            "drum_clean", ["Cotton 90", "Cotton 90", "Eco"]
        ) == ["Cotton 90", "Eco"]
        with pytest.raises(ValueError, match="Unknown maintenance task"):
            await store.async_set_maintenance_programs("custom_nope", ["Eco"])
        with pytest.raises(ValueError, match="Unknown program"):
            await store.async_set_maintenance_programs("drum_clean", ["Cotton 60"])
        with pytest.raises(ValueError, match="list"):
            await store.async_set_maintenance_programs("drum_clean", "Eco")
        assert store.get_maintenance_programs() == {"drum_clean": ["Cotton 90", "Eco"]}
        # An empty list turns it off.
        assert await store.async_set_maintenance_programs("drum_clean", []) == []
        assert store.get_maintenance_programs() == {}


async def test_too_many_programs_are_refused(hass: HomeAssistant) -> None:
    names = [f"P{i}" for i in range(MAINTENANCE_PROGRAMS_MAX + 1)]
    store = _store(hass, *names)
    with patch.object(store, "async_save", AsyncMock()), pytest.raises(ValueError, match="At most"):
        await store.async_set_maintenance_programs("drum_clean", names)


async def test_a_finished_cycle_of_a_listed_program_logs_the_task(hass: HomeAssistant) -> None:
    store = _store(hass, "Cotton 90", "Eco")
    with patch.object(store, "async_save", AsyncMock()):
        task = await store.async_add_maintenance_task("Hot wash", days=30)
        await store.async_set_maintenance_programs("drum_clean", ["Cotton 90"])
        await store.async_set_maintenance_programs(task["id"], ["Cotton 90"])
    store.set_lifetime_cycle_count(40, force=True)

    assert store.log_maintenance_done_by_program(_cycle("Eco")) == []
    assert store.log_maintenance_done_by_program(_cycle(None)) == []
    assert store.log_maintenance_done_by_program(_cycle("Cotton 90")) == ["drum_clean", task["id"]]
    # The same cycle never logs twice (a restart re-running the cycle end).
    assert store.log_maintenance_done_by_program(_cycle("Cotton 90")) == []

    log = store.get_maintenance_log()
    assert {e["event_type"] for e in log} == {"drum_clean", task["id"]}
    for entry in log:
        assert entry["date"] == END
        assert entry["program"] == "Cotton 90"
        assert entry["cycle_id"] == "c1"
        assert entry["cycle_count_at_log"] == 40
    assert next(e for e in log if e["event_type"] == task["id"])["task_name"] == "Hot wash"

    rows = {r["id"]: r for r in store.get_maintenance_status({"drum_clean": 100})}
    assert rows["drum_clean"]["cycles_since"] == 0
    assert rows["drum_clean"]["programs"] == ["Cotton 90"]
    assert rows[task["id"]]["programs"] == ["Cotton 90"]
    assert rows[task["id"]]["due"] is False


async def test_rename_delete_and_task_removal_follow(hass: HomeAssistant) -> None:
    store = _store(hass, "Cotton 90", "Eco")
    with patch.object(store, "async_save", AsyncMock()):
        task = await store.async_add_maintenance_task("Hot wash", days=30)
        await store.async_set_maintenance_programs("drum_clean", ["Cotton 90", "Eco"])
        await store.async_set_maintenance_programs(task["id"], ["Cotton 90"])

        await store.update_profile("Cotton 90", "Cotton 95")
        assert store.get_maintenance_programs() == {
            "drum_clean": ["Cotton 95", "Eco"], task["id"]: ["Cotton 95"],
        }
        await store.delete_profile("Cotton 95")
        assert store.get_maintenance_programs() == {"drum_clean": ["Eco"]}

        await store.async_set_maintenance_programs(task["id"], ["Eco"])
        await store.async_delete_maintenance_task(task["id"])
        assert store.get_maintenance_programs() == {"drum_clean": ["Eco"]}


async def test_a_hand_picked_cycle_logs_the_task_at_cycle_end(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    entry = await setup_washdata_entry(power_sensor=POWER)
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    store._data["profiles"]["Cotton 90"] = {"avg_duration": 3600.0}  # noqa: SLF001
    await store.async_set_maintenance_programs("drum_clean", ["Cotton 90"])

    manager._current_program = "Cotton 90"  # noqa: SLF001
    manager._manual_program_active = True  # noqa: SLF001
    await manager._async_process_cycle_end({  # noqa: SLF001
        "duration": 3600.0, "status": "completed", "energy_wh": 1800.0,
        "start_time": "2026-10-20T10:30:00+00:00", "end_time": END,
    })
    await hass.async_block_till_done()

    log = store.get_maintenance_log()
    assert [(e["event_type"], e["program"]) for e in log] == [("drum_clean", "Cotton 90")]
    assert store.cycles_since_maintenance("drum_clean") == 0


async def test_an_interrupted_cycle_does_not_count(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    entry = await setup_washdata_entry(power_sensor=POWER)
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    store._data["profiles"]["Cotton 90"] = {"avg_duration": 3600.0}  # noqa: SLF001
    await store.async_set_maintenance_programs("drum_clean", ["Cotton 90"])

    manager._current_program = "Cotton 90"  # noqa: SLF001
    manager._manual_program_active = True  # noqa: SLF001
    await manager._async_process_cycle_end({  # noqa: SLF001
        "duration": 200.0, "status": "interrupted", "energy_wh": 10.0,
        "start_time": "2026-10-20T10:30:00+00:00", "end_time": END,
    })
    await hass.async_block_till_done()

    assert store.get_maintenance_log() == []


async def test_an_imported_copy_takes_its_programs_with_it(hass: HomeAssistant) -> None:
    """Selective import "as a copy" renames the file's profile; the task's program
    list follows it instead of naming the unrelated local profile."""
    store = _store(hass, "Cotton 90")
    payload = {
        "version": 11,
        "device_fingerprint": {"device_type": "washing_machine"},
        "data": {
            "profiles": {"Cotton 90": {"avg_duration": 5400.0}},
            "past_cycles": [], "reference_cycles": [],
            "maintenance_log": [], "maintenance_tasks": [],
            "maintenance_programs": {"drum_clean": ["Cotton 90"]},
        },
        "entry_data": {}, "entry_options": {},
    }
    with patch.object(store, "async_save", AsyncMock()), patch.object(
        store, "async_save_pre_import_snapshot", AsyncMock(return_value=True)
    ):
        await store.async_import_data_selective(
            payload, selection={"categories": ["profiles", "maintenance_log"]},
            conflict_resolutions={"Cotton 90": "import_as_copy"},
            local_device_type="washing_machine",
        )
    assert "Cotton 90 (imported)" in store.get_profiles()
    assert store.get_maintenance_programs() == {"drum_clean": ["Cotton 90 (imported)"]}


async def test_a_profile_removed_with_its_last_cycle_or_by_gc_leaves_no_association(
    hass: HomeAssistant,
) -> None:
    store = _store(hass, "Imported 90", "Broken")
    store._data["reference_cycles"] = [  # noqa: SLF001
        {"id": "r1", "profile_name": "Imported 90", "start_time": END, "duration": 60.0}
    ]
    store._data["profiles"]["Broken"]["sample_cycle_id"] = "gone"  # noqa: SLF001
    with patch.object(store, "async_save", AsyncMock()):
        await store.async_set_maintenance_programs("drum_clean", ["Imported 90", "Broken"])
        assert await store.delete_cycle("r1")
    assert "Imported 90" not in store.get_profiles()
    assert store.get_maintenance_programs() == {"drum_clean": ["Broken"]}

    assert store.cleanup_orphaned_profiles() == 1
    assert store.get_maintenance_programs() == {}


async def test_moving_a_programs_last_imported_cycle_away_drops_the_association(
    hass: HomeAssistant,
) -> None:
    store = _store(hass, "Imported 90", "Eco")
    ref = {"id": "r1", "profile_name": "Imported 90", "start_time": END, "duration": 60.0}
    store._data["reference_cycles"] = [ref]  # noqa: SLF001
    with patch.object(store, "async_save", AsyncMock()):
        await store.async_set_maintenance_programs("drum_clean", ["Imported 90"])
    store._relabel_non_real_cycle(ref, "Eco")  # noqa: SLF001
    assert "Imported 90" not in store.get_profiles()
    assert store._data["maintenance_programs"] == {}  # noqa: SLF001


async def test_a_name_that_is_no_longer_a_program_is_not_offered(hass: HomeAssistant) -> None:
    store = _store(hass, "Eco")
    store._data["maintenance_programs"] = {"drum_clean": ["Gone", "Eco"], "descale": ["Gone"]}  # noqa: SLF001
    assert store.get_maintenance_programs() == {"drum_clean": ["Eco"]}


async def test_a_merge_import_unites_the_lists_of_a_task_both_stores_have(
    hass: HomeAssistant,
) -> None:
    store = _store(hass, "Eco", "Cotton 90")
    with patch.object(store, "async_save", AsyncMock()):
        await store.async_set_maintenance_programs("drum_clean", ["Eco"])
    payload = {
        "version": 11,
        "device_fingerprint": {"device_type": "washing_machine"},
        "data": {
            "profiles": {}, "past_cycles": [], "reference_cycles": [],
            "maintenance_log": [], "maintenance_tasks": [],
            "maintenance_programs": {"drum_clean": ["Cotton 90", "Eco"]},
        },
        "entry_data": {}, "entry_options": {},
    }
    with patch.object(store, "async_save", AsyncMock()), patch.object(
        store, "async_save_pre_import_snapshot", AsyncMock(return_value=True)
    ):
        await store.async_import_data_selective(
            payload, selection={"categories": ["maintenance_log"]}, mode="merge",
            local_device_type="washing_machine",
        )
    assert store.get_maintenance_programs() == {"drum_clean": ["Eco", "Cotton 90"]}
