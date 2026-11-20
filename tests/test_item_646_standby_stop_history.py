"""Register item 646: a cycle that ended below an older, higher Stop Threshold.

``detect_standby_above_stop`` counted every cycle that ended on its own as one that
gets below the threshold, but it only got below the one in force then. With the
stop lowered from 5 W to 2 W under a 3 W standby, six such ends outvoted the two
hand stops that showed the 3 W standby, so the advisory, the #458 floor and the
learned standby level all stayed silent. The settings changelog records each Stop
Threshold change, so the threshold in force at each end is known.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

from homeassistant.core import HomeAssistant

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import (
    CONF_STOP_THRESHOLD_W,
    DOMAIN,
    SETTINGS_CHANGELOG_IMPORTED_KEY,
)
from custom_components.ha_washdata.suggestion_engine import (
    detect_standby_above_stop,
    recorded_stop_changes,
    stop_threshold_changes,
)

T0 = datetime(2026, 11, 1, 8, tzinfo=timezone.utc)
CHANGED_AT = T0 + timedelta(days=6, hours=12)


def _program() -> list[list[float]]:
    return [[10.0 * i, 2150.0 if 60 <= i < 120 else 60.0 + i % 11] for i in range(360)]


def _cycle(day: int, reason: str, standby: float | None = None) -> dict[str, Any]:
    """A natural end (cut at its last working reading) or a hand stop kept at ``standby``."""
    pts = _program()
    if standby is not None:
        t = pts[-1][0]
        for _ in range(30):
            t += 10.0
            pts.append([t, standby])
    start = T0 + timedelta(days=day)
    return {
        "id": f"c{day}", "status": "completed", "termination_reason": reason,
        "start_time": start.isoformat(),
        "end_time": (start + timedelta(seconds=pts[-1][0])).isoformat(),
        "duration": pts[-1][0], "power_data": pts, "signature": {"max_power": 2150.0},
    }


def _history() -> list[dict[str, Any]]:
    natural = [_cycle(d, "timeout") for d in range(6)]          # under the old 5 W stop
    return natural + [_cycle(7, "user", 3.0), _cycle(8, "user", 3.0)]  # under 2 W


def _log(old: float, new: float, when: datetime = CHANGED_AT) -> list[dict[str, Any]]:
    return [{"key": CONF_STOP_THRESHOLD_W, "old": old, "new": new, "timestamp": when.isoformat()}]


def test_ends_under_a_higher_stop_say_nothing_about_a_lowered_one() -> None:
    cycles = _history()
    assert detect_standby_above_stop(cycles, 2.0) is None  # nothing recorded: as before
    adv = detect_standby_above_stop(cycles, 2.0, stop_changes=stop_threshold_changes(_log(5.0, 2.0)))
    assert adv is not None
    assert adv["idle_w"] == 3.0 and adv["cycles_above"] == 2 and adv["cycles_checked"] == 2


def test_ends_under_a_lower_or_the_same_stop_still_count_as_below() -> None:
    cycles = _history()
    assert detect_standby_above_stop(cycles, 2.0, stop_changes=stop_threshold_changes(_log(1.5, 2.0))) is None
    # A change before every cycle: they all ended under the current stop.
    early = stop_threshold_changes(_log(5.0, 2.0, T0 - timedelta(days=1)))
    assert detect_standby_above_stop(cycles, 2.0, stop_changes=early) is None


def test_the_changelog_reader_keeps_only_timed_stop_rows_oldest_first() -> None:
    log = [
        {"key": CONF_STOP_THRESHOLD_W, "old": "4", "new": 2, "timestamp": "2026-11-09T10:00:00+01:00"},
        {"key": "off_delay", "old": 180, "new": 600, "timestamp": "2026-11-08T10:00:00+00:00"},
        {"key": CONF_STOP_THRESHOLD_W, "old": 5, "new": 4, "timestamp": "2026-11-07T10:00:00"},
        {"key": CONF_STOP_THRESHOLD_W, "old": 9, "new": 5, "timestamp": "not a time"},
        "junk",
    ]
    changes = stop_threshold_changes(log)
    assert [old for _when, old in changes] == [5.0, 4.0]
    assert changes[0][0].tzinfo is not None
    assert stop_threshold_changes(None) == [] and stop_threshold_changes("x") == []


async def test_every_caller_reads_the_changelog_on_a_real_boot(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    """The Settings advisory, the #458 floor and the learned standby level."""
    entry = await setup_washdata_entry(
        options={CONF_STOP_THRESHOLD_W: 2.0, "start_threshold_w": 10.0, "off_delay": 180}
    )
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    store._data["past_cycles"] = _history()  # noqa: SLF001
    await store.async_record_settings_changes(_log(5.0, 2.0))

    conn = MagicMock()
    ws_api.ws_get_devices(hass, conn, {"id": 1, "type": "ha_washdata/get_devices"})
    devices = conn.send_result.call_args[0][1]["devices"]
    info = next(d for d in devices if d["entry_id"] == entry.entry_id)
    assert info["standby_above_stop"] is not None
    assert info["standby_above_stop"]["idle_w"] == 3.0

    floor = manager.learning_manager.suggestion_engine._standby_floor(dict(entry.options))  # noqa: SLF001
    assert floor is not None and floor["idle_w"] == 3.0

    await manager._async_refresh_standby_level()  # noqa: SLF001
    assert manager.detector._standby_level_w == 3.0  # noqa: SLF001


async def test_a_stop_change_outlives_the_changelog_cap(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    """The changelog keeps the newest 50 changes of any setting; recent Stop
    Threshold rows are kept past it (at most 10), or 50 other changes after the
    lowering would bring the stale ends back."""
    entry = await setup_washdata_entry(options={CONF_STOP_THRESHOLD_W: 2.0, "start_threshold_w": 10.0})
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    store._data["past_cycles"] = _history()  # noqa: SLF001
    await store.async_record_settings_changes(_log(5.0, 2.0))
    later = (CHANGED_AT + timedelta(hours=1)).isoformat()
    await store.async_record_settings_changes(
        [{"key": "off_delay", "old": 180 + i, "new": 181 + i, "timestamp": later} for i in range(60)]
    )
    log = store.get_settings_changelog()
    assert len(log) == store.SETTINGS_CHANGELOG_MAX + 1
    assert log[-1]["key"] == CONF_STOP_THRESHOLD_W
    changes = stop_threshold_changes(log)
    assert detect_standby_above_stop(store.get_past_cycles(), 2.0, stop_changes=changes) is not None

    # Bounded: never more than SETTINGS_CHANGELOG_KEEP_STOP stop rows past the cap.
    await store.async_record_settings_changes(
        [{"key": CONF_STOP_THRESHOLD_W, "old": 2.0, "new": 2.0 + i / 10, "timestamp": later} for i in range(20)]
    )
    await store.async_record_settings_changes(
        [{"key": "off_delay", "old": 1, "new": 2, "timestamp": later} for _ in range(60)]
    )
    tail = store.get_settings_changelog()[store.SETTINGS_CHANGELOG_MAX:]
    assert len(tail) == store.SETTINGS_CHANGELOG_KEEP_STOP
    assert all(row["key"] == CONF_STOP_THRESHOLD_W for row in tail)


async def test_a_merged_newer_stop_row_survives_the_prune(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    """A history merge appends imported rows after the local ones; the prune must
    judge them by their time, not their place (PR #529 review)."""
    entry = await setup_washdata_entry()
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    old = [{"key": CONF_STOP_THRESHOLD_W, "old": 3.0, "new": 2.0 + i / 10,
            "timestamp": (T0 - timedelta(days=30 - i)).isoformat()} for i in range(store.SETTINGS_CHANGELOG_KEEP_STOP)]
    filler = [{"key": "off_delay", "old": 1, "new": 2, "timestamp": (T0 - timedelta(days=40)).isoformat()}
              for _ in range(store.SETTINGS_CHANGELOG_MAX)]
    merged_in = {"key": CONF_STOP_THRESHOLD_W, "old": 5.0, "new": 2.0, "timestamp": CHANGED_AT.isoformat()}
    store._data["settings_changelog"] = filler + old + [merged_in]  # noqa: SLF001 - as a merge leaves it
    await store.async_record_settings_changes([{"key": "off_delay", "old": 2, "new": 3}])
    assert any(r is merged_in or r == merged_in for r in store.get_settings_changelog())


def _history_export(source_entry_id: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": 19,
        "device_fingerprint": {"device_type": "washing_machine"},
        "data": {"settings_changelog": _log(5.0, 2.0)},
        "entry_data": {},
        "entry_options": {},
    }
    if source_entry_id is not None:
        payload["entry_id"] = source_entry_id
    return payload


def test_the_changelog_reader_skips_rows_imported_from_another_device() -> None:
    imported = [{**row, SETTINGS_CHANGELOG_IMPORTED_KEY: True} for row in _log(5.0, 2.0)]
    assert stop_threshold_changes(imported) == []
    assert detect_standby_above_stop(
        _history(), 2.0, stop_changes=stop_threshold_changes(imported)
    ) is None


async def test_another_appliance_s_history_does_not_classify_local_cycles(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    """PR #529 review: History Logs imported from another device put its Stop
    change into this device's changelog, and the advisory read the local natural
    ends as ended under that device's 5 W Stop. The rows stay visible, marked,
    and the advisory is what it was before the import. This device's own export
    brought back is still read."""
    entry = await setup_washdata_entry(
        options={CONF_STOP_THRESHOLD_W: 2.0, "start_threshold_w": 10.0, "off_delay": 180}
    )
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    store._data["past_cycles"] = _history()  # noqa: SLF001

    def _advisory() -> Any:
        return detect_standby_above_stop(
            store.get_past_cycles(), 2.0, stop_changes=recorded_stop_changes(store)
        )

    assert _advisory() is None
    for source in ("another-entry", None):
        store._data["settings_changelog"] = []  # noqa: SLF001
        await store.async_import_data_selective(
            _history_export(source), selection={"categories": ["history_logs"]},
            local_device_type="washing_machine",
        )
        log = store.get_settings_changelog()
        assert len(log) == 1 and log[0][SETTINGS_CHANGELOG_IMPORTED_KEY] is True
        assert _advisory() is None

    store._data["settings_changelog"] = []  # noqa: SLF001
    await store.async_import_data_selective(
        _history_export(entry.entry_id), selection={"categories": ["history_logs"]},
        local_device_type="washing_machine",
    )
    assert SETTINGS_CHANGELOG_IMPORTED_KEY not in store.get_settings_changelog()[0]
    assert _advisory() is not None


async def test_a_wholesale_import_of_history_without_cycles_marks_it_foreign(
    hass: Any, setup_washdata_entry: Any
) -> None:
    """Review of PR #529: a wholesale import that brings a settings history but no
    cycles keeps the local cycles; the payload's Stop rows are not what they ended
    under, so they are marked like the selective import's."""
    from custom_components.ha_washdata.const import DOMAIN, SETTINGS_CHANGELOG_IMPORTED_KEY  # noqa: PLC0415

    entry = await setup_washdata_entry()
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    store._data["past_cycles"] = [{"id": "local", "profile_name": "A", "status": "completed",  # noqa: SLF001
                                   "duration": 3600.0, "power_data": [[0, 100.0], [3600, 0.0]]}]
    row = {"key": "stop_threshold_w", "old": 6.0, "new": 2.0, "timestamp": "2026-11-01T10:00:00+00:00"}
    await store.async_import_data({"version": 19, "data": {"profiles": {"A": {}}, "settings_changelog": [row]}})
    rows = store.get_settings_changelog()
    assert rows and all(r.get(SETTINGS_CHANGELOG_IMPORTED_KEY) for r in rows if r.get("key") == "stop_threshold_w")


def test_the_stop_a_finish_was_decided_under_beats_the_history_at_its_end() -> None:
    """PR #529 review: Stop raised 2 -> 5 W during a run, then restored to 2 W. The
    detector stores the last active reading, from before the raise, as the end, so
    the history at that time reads 2 W, and six ends that only got below 5 W
    outvoted the two hand stops showing the 3 W standby. The Stop the finish was
    decided under is recorded on the cycle and read first; older cycles keep the
    history lookup."""
    natural = [_cycle(d, "timeout") for d in range(6)]
    hand = [_cycle(7, "user", 3.0), _cycle(8, "user", 3.0)]
    raised = CHANGED_AT - timedelta(days=1, hours=1)  # after the last natural end's stamp
    log = [
        {"key": CONF_STOP_THRESHOLD_W, "old": 2.0, "new": 5.0, "timestamp": raised.isoformat()},
        {"key": CONF_STOP_THRESHOLD_W, "old": 5.0, "new": 2.0, "timestamp": CHANGED_AT.isoformat()},
    ]
    changes = stop_threshold_changes(log)
    assert detect_standby_above_stop(natural + hand, 2.0, stop_changes=changes) is None
    for cycle in natural:
        cycle["finish_stop_threshold_w"] = 5.0
    for stop_changes in (changes, None):
        adv = detect_standby_above_stop(natural + hand, 2.0, stop_changes=stop_changes)
        assert adv is not None and adv["idle_w"] == 3.0 and adv["cycles_checked"] == 2
    # Finished under the current stop: still a cycle that gets below it.
    for cycle in natural:
        cycle["finish_stop_threshold_w"] = 2.0
    assert detect_standby_above_stop(natural + hand, 2.0, stop_changes=changes) is None


async def test_a_finished_cycle_records_the_stop_it_was_decided_under(
    hass: HomeAssistant, setup_washdata_entry, freezer
) -> None:
    """The detector stamps the Stop Threshold in force at its finish decision."""
    from homeassistant.util import dt as dt_util  # noqa: PLC0415
    from pytest_homeassistant_custom_component.common import async_fire_time_changed  # noqa: PLC0415

    power = "sensor.washer_power"
    entry = await setup_washdata_entry(power_sensor=power, options={
        "min_power": 2.0, "off_delay": 60, "min_off_gap": 60, "watchdog_interval": 30,
        "completion_min_seconds": 300, CONF_STOP_THRESHOLD_W: 1.5,
    })
    manager = hass.data[DOMAIN][entry.entry_id]

    async def feed(watts: float, seconds: int) -> None:
        for _ in range(seconds // 30):
            freezer.tick(timedelta(seconds=30))
            hass.states.async_set(power, str(watts), {"unit_of_measurement": "W"},
                                  force_update=True)
            async_fire_time_changed(hass, dt_util.utcnow())
            await hass.async_block_till_done()

    await feed(200.0, 15 * 60)
    await feed(0.0, 10 * 60)
    cycles = manager.profile_store.get_past_cycles()
    assert cycles and cycles[-1]["finish_stop_threshold_w"] == 1.5


async def test_undo_import_records_a_stop_it_removes(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    """PR #529 review: an import set Stop to 3 W; the restore point has it unset, so
    Undo Import removes the override and the Stop goes back to its default. Only
    keys the restore wrote were recorded, so the history never saw the change.
    The Stop before the row is the restored one: every cycle the restored store
    holds ended before the import, under it (PR #529 review, round 36b)."""
    entry = await setup_washdata_entry(options={
        "min_power": 2.0, "start_threshold_w": 10.0, CONF_STOP_THRESHOLD_W: 3.0,
    })
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    await ws_api._async_restore_entry_options(  # noqa: SLF001
        hass, entry, {"min_power": 2.0, "start_threshold_w": 10.0},
    )
    assert CONF_STOP_THRESHOLD_W not in entry.options
    rows = [r for r in store.get_settings_changelog() if r["key"] == CONF_STOP_THRESHOLD_W]
    assert len(rows) == 1, rows
    assert rows[0]["old"] == 3.0 and rows[0]["new"] is None
    assert rows[0]["old_effective"] == 1.2  # Stop unset: follows the 2 W Min Power
    assert [old for _when, old in stop_threshold_changes(store.get_settings_changelog())] == [1.2]
