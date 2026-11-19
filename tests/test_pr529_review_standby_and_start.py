"""Review of PR #529: the standby advisory's evidence and the steady-load Start fix.

* A legacy trace stamped with ISO times weighed its kept tail at None, so a hand
  stop read its last sample alone (a 6 W pulse over a settled 3 W standby).
* A Stop Threshold row recorded ``old=None`` while Stop was unset and followed Min
  Power, so the cycles that ended under that effective Stop counted as ending
  below a lowered one and outvoted the hand stops showing the standby.
* On a kW sensor the steady-load correction rounded a 0.08 running level to 0.1
  and proposed Start 0.1, above the load it was meant to start on.
* The device-list poll reads the settled-tail cache the manager fills in the
  executor at setup, so a restart does not sort the tails on the event loop.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata import suggestion_engine as se
from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import (
    CONF_MIN_POWER,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    DOMAIN,
    STORAGE_KEY,
    STORAGE_VERSION,
)

T0 = datetime(2026, 11, 1, 8, tzinfo=timezone.utc)


def _hand_stop(day: int, *, iso: bool, hours: float = 0.5) -> dict[str, Any]:
    """Ten minutes at 2 kW, then ``hours`` of a 3 W standby, stopped by hand on a 6 W pulse."""
    start = T0 + timedelta(days=day)
    pts: list[list[Any]] = [[10.0 * i, 2000.0] for i in range(60)]
    pts += [[600.0 + 10.0 * i, 3.0] for i in range(int(hours * 360))]
    pts.append([pts[-1][0] + 10.0, 6.0])
    if iso:
        pts = [[(start + timedelta(seconds=t)).isoformat(), w] for t, w in pts]
        end_s = 600.0 + 10.0 * int(hours * 360)
    else:
        end_s = pts[-1][0]
    return {
        "id": f"h{day}", "status": "completed", "termination_reason": "user",
        "start_time": start.isoformat(),
        "end_time": (start + timedelta(seconds=end_s)).isoformat(),
        "duration": end_s, "power_data": pts, "signature": {"max_power": 2000.0},
    }


@pytest.mark.parametrize("iso", [False, True])
def test_a_legacy_iso_trace_weighs_its_tail_like_an_offset_one(iso: bool) -> None:
    se._TAIL_SETTLED_CACHE.clear()  # noqa: SLF001
    cycles = [_hand_stop(0, iso=iso), _hand_stop(1, iso=iso)]
    assert se._idle_evidence_w(cycles[0], 5.0) == 3.0  # noqa: SLF001
    # Settled at 3 W under a 5 W Stop: below it, so no "still drawing" advisory.
    assert se.detect_standby_above_stop(cycles, 5.0) is None


def _timeout(day: int) -> dict[str, Any]:
    start = T0 + timedelta(days=day)
    pts = [[10.0 * i, 2000.0 if i < 60 else 40.0] for i in range(120)]
    return {
        "id": f"t{day}", "status": "completed", "termination_reason": "timeout",
        "start_time": start.isoformat(),
        "end_time": (start + timedelta(seconds=pts[-1][0])).isoformat(),
        "duration": pts[-1][0], "power_data": pts, "signature": {"max_power": 2000.0},
    }


async def test_a_stop_row_records_the_stop_in_force_while_it_was_unset(
    hass: HomeAssistant, setup_washdata_entry
) -> None:
    """Min Power 10 W, Stop unset (runs at 6 W): six ends timed out under 6 W. Stop
    set to 2 W; two hand stops then show a 3 W standby. Revert keeps ``old``."""
    entry = await setup_washdata_entry(options={CONF_MIN_POWER: 10.0})
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    await ws_api._record_option_changes(  # noqa: SLF001
        hass, entry, {CONF_STOP_THRESHOLD_W: 2.0}, "set_options"
    )
    row = store.get_settings_changelog()[0]
    assert row["key"] == CONF_STOP_THRESHOLD_W
    assert row["old"] is None  # what Revert restores
    assert row["old_effective"] == 6.0

    changed = datetime.fromisoformat(row["timestamp"])
    history = [_timeout(d) for d in range(6)]
    for d, c in enumerate(history):  # all six ended before the change
        c["end_time"] = (changed - timedelta(days=7 - d)).isoformat()
    hand = [_hand_stop(0, iso=False), _hand_stop(1, iso=False)]
    for i, c in enumerate(hand):
        c["end_time"] = (changed + timedelta(hours=1 + i)).isoformat()
    changes = se.stop_threshold_changes(store.get_settings_changelog())
    assert changes[0][1] == 6.0
    adv = se.detect_standby_above_stop(history + hand, 2.0, stop_changes=changes)
    assert adv is not None and adv["idle_w"] == 3.0


def test_a_row_without_the_effective_stop_still_reads_old() -> None:
    log = [{"key": CONF_STOP_THRESHOLD_W, "old": 5.0, "new": 2.0, "timestamp": T0.isoformat()}]
    assert se.stop_threshold_changes(log)[0][1] == 5.0
    log[0]["old_effective"] = "junk"
    assert se.stop_threshold_changes(log)[0][1] == 5.0


async def test_min_power_moving_an_unset_stop_is_recorded_as_a_stop_change(
    hass: HomeAssistant, setup_washdata_entry, hass_ws_client
) -> None:
    """PR #529 review: with Stop unset it follows Min Power (#450), so a save of Min
    Power 10 W -> 3 W moved Stop from 6 W to 1.8 W with no Stop row, and the six
    ends under 6 W were judged as below 1.8 W, outvoting two 3 W hand stops. The
    move is recorded for the advisory, not listed as a Stop change in the panel."""
    entry = await setup_washdata_entry(options={CONF_MIN_POWER: 10.0})
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    client = await hass_ws_client(hass)
    await client.send_json_auto_id({
        "type": f"{DOMAIN}/set_options", "entry_id": entry.entry_id,
        "options": {CONF_MIN_POWER: 3.0},
    })
    res = await client.receive_json()
    assert res["success"], res
    await hass.async_block_till_done()

    rows = [r for r in store.get_settings_changelog() if r["key"] == CONF_STOP_THRESHOLD_W]
    assert len(rows) == 1
    assert rows[0]["old"] is None and rows[0]["new"] is None  # Revert changes nothing
    assert rows[0]["old_effective"] == 6.0
    changes = se.stop_threshold_changes(store.get_settings_changelog())
    assert [old for _when, old in changes] == [6.0]

    changed = changes[0][0]
    history = [_timeout(d) for d in range(6)]
    for d, c in enumerate(history):  # all six ended under the 6 W Stop
        c["end_time"] = (changed - timedelta(days=7 - d)).isoformat()
    hand = [_hand_stop(0, iso=False), _hand_stop(1, iso=False)]
    for i, c in enumerate(hand):
        c["end_time"] = (changed + timedelta(hours=1 + i)).isoformat()
    adv = se.detect_standby_above_stop(history + hand, 1.8, stop_changes=changes)
    assert adv is not None and adv["idle_w"] == 3.0

    await client.send_json_auto_id({
        "type": f"{DOMAIN}/get_settings_changelog", "entry_id": entry.entry_id,
    })
    listed = (await client.receive_json())["result"]["changelog"]
    assert [r["key"] for r in listed] == [CONF_MIN_POWER]


def _steady_kw(day: int, level: float) -> dict[str, Any]:
    """An hour at a steady ``level`` kW, 30 s readings, no idle readings kept."""
    start = T0 + timedelta(days=day)
    pts = [[30.0 * (i + 1), level] for i in range(120)]
    return {"id": f"k{day}", "status": "completed", "start_time": start.isoformat(),
            "duration": pts[-1][0], "power_data": pts}


@pytest.mark.parametrize("level", [0.08, 0.04, 0.012, 0.15])
def test_a_kw_steady_load_start_correction_stays_below_its_running_level(level: float) -> None:
    cycles = [_steady_kw(d, level) for d in range(4)]
    adv = se.start_above_running(cycles, 0.2, 0.01)
    assert adv is not None
    out = se.steady_load_threshold_suggestions(adv)
    start = out[CONF_START_THRESHOLD_W]["value"]
    assert 0 < start < level
    stop = out.get(CONF_STOP_THRESHOLD_W, {}).get("value", 0.01)
    assert 0 < stop < start
    # Applied, the gate no longer fires: the pair is not offered again.
    assert se.start_above_running(cycles, start, stop) is None


def test_watt_levels_round_as_before() -> None:
    assert se._round_level(380.44) == 380.4  # noqa: SLF001
    assert se._round_level(12.36) == 12.4  # noqa: SLF001
    assert se._round_level(0.0834) == 0.083  # noqa: SLF001


async def test_the_first_device_poll_after_a_restart_reads_the_warm_tail_cache(
    hass: HomeAssistant, enable_custom_integrations: None,
    hass_storage: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setup learns the standby level in the executor over the same cycles, which
    fills the cache the poll reads: the poll itself computes no tail."""
    from pytest_homeassistant_custom_component.common import MockConfigEntry  # noqa: PLC0415

    from custom_components.ha_washdata.const import (  # noqa: PLC0415
        CONFIG_ENTRY_MINOR_VERSION,
        CONFIG_ENTRY_VERSION,
    )

    hass.states.async_set("sensor.washer_power", "0", {"unit_of_measurement": "W"})
    entry = MockConfigEntry(
        domain=DOMAIN, title="Washer",
        data={"name": "Washer", "power_sensor": "sensor.washer_power",
              "device_type": "washing_machine"},
        options={CONF_STOP_THRESHOLD_W: 2.0, CONF_START_THRESHOLD_W: 10.0},
        unique_id="washdata_pr529_tail_cache",
        version=CONFIG_ENTRY_VERSION, minor_version=CONFIG_ENTRY_MINOR_VERSION,
    )
    entry.add_to_hass(hass)
    hass_storage[f"{STORAGE_KEY}.{entry.entry_id}"] = {
        "version": STORAGE_VERSION, "minor_version": 1,
        "key": f"{STORAGE_KEY}.{entry.entry_id}",
        "data": {"profiles": {}, "past_cycles": [_hand_stop(d, iso=False, hours=3.0) for d in range(8)]},
    }
    se._TAIL_SETTLED_CACHE.clear()  # noqa: SLF001
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    computed: list[int] = []
    original = se._tail_settled_w_uncached  # noqa: SLF001

    def _counting(cycle: dict[str, Any], raw: list[Any], peak: float) -> float | None:
        computed.append(len(raw))
        return original(cycle, raw, peak)

    monkeypatch.setattr(se, "_tail_settled_w_uncached", _counting)
    conn = MagicMock()
    ws_api.ws_get_devices(hass, conn, {"id": 1, "type": "ha_washdata/get_devices"})
    info = next(d for d in conn.send_result.call_args[0][1]["devices"] if d["entry_id"] == entry.entry_id)
    assert info["standby_above_stop"] is not None and info["standby_above_stop"]["idle_w"] == 3.0
    assert computed == []
