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


def _reference_export(device_type: str) -> dict[str, Any]:
    """An export holding one reference copy of a 3,000 s run corrected to 3,300 s."""
    trace = [[60.0 * i, 1800.0 if i < 10 else 120.0] for i in range(51)]
    ref = {
        "id": "r1", "profile_name": "ECO", "duration": 3000.0, "manual_duration": 3300.0,
        "status": "completed", "start_time": "2026-01-01T10:00:00+00:00",
        "power_data": trace, "meta": {"source": "store:abc"},
    }
    return {
        "version": 19,
        "device_fingerprint": {"device_type": device_type},
        "data": {"profiles": {"ECO": {"avg_duration": 3300.0}}, "past_cycles": [],
                 "reference_cycles": [ref]},
        "entry_data": {}, "entry_options": {},
    }


async def test_a_reimported_reference_copy_keeps_its_length_correction(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """The payload's reference_cycles path dropped ``manual_duration`` (the real
    cycles path carries it): the rebuild read 3,000 s and the run lost the pin
    that keeps it out of the #531 cut-off vote. Another device type's correction
    is not this device's, as for the trim and edit markers."""
    entry = await setup_washdata_entry(device_type="dishwasher")
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    await store.async_import_data_selective(
        _reference_export("dishwasher"),
        selection={"categories": ["profiles", "reference_cycles"]},
        local_device_type="dishwasher",
    )
    ref = store.get_reference_cycles()[0]
    assert ref.get("manual_duration") == 3300.0
    assert store.profile_trusted_min_duration("ECO") == 3300.0

    other = await setup_washdata_entry("Dishwasher 2", device_type="dishwasher")
    store2 = hass.data[DOMAIN][other.entry_id].profile_store
    await store2.async_import_data_selective(
        _reference_export("washing_machine"),
        selection={"categories": ["profiles", "reference_cycles"]},
        local_device_type="dishwasher",
    )
    assert "manual_duration" not in store2.get_reference_cycles()[0]


async def test_the_match_converts_the_trace_off_the_event_loop(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """The #530 check (and the live tick) handed a long cycle's readings to the
    matcher, which converted every one to seconds and resampled them on the loop
    before its executor job: ~3-4.5 ms a match on a 4 h cycle at 2 s."""
    import threading  # noqa: PLC0415
    from datetime import datetime, timezone  # noqa: PLC0415

    on_loop: list[bool] = []
    loop_thread = threading.current_thread()

    class _Stamp(datetime):
        def timestamp(self) -> float:
            on_loop.append(threading.current_thread() is loop_thread)
            return super().timestamp()

    from custom_components.ha_washdata import profile_store  # noqa: PLC0415

    entry = await setup_washdata_entry()
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    t0 = datetime(2026, 11, 1, 8, tzinfo=timezone.utc).timestamp()
    n = profile_store._MATCH_PREP_INLINE_READINGS + 100  # noqa: SLF001 - a long trace
    readings = [
        (_Stamp.fromtimestamp(t0 + 2.0 * i, tz=timezone.utc), 1800.0 if i < n // 2 else 150.0)
        for i in range(n)
    ]
    await store.async_match_profile(readings, 2.0 * (n - 1), in_progress=True, stop_threshold_w=2.0)
    assert len(on_loop) >= len(readings)  # every reading was converted
    assert not any(on_loop)
