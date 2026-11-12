"""#518 parts 2 and 3: history-imported dishwasher cycles kept their end wait.

The history import replays unmatched, so a dishwasher's timeout finish keeps its
whole ``min_off_gap`` wait (``keep_tail``). The import cuts it only when the
candidate confidently matches an existing programme, so on a first import nothing
was cut, and once labelled into "Eco" the reporter's four cycles (12,084-12,950 s
stored, last activity ~8,470 s) took its ``avg_duration`` from 8,461 to 10,193 s.
Now the envelope rebuild a label triggers cuts them with the same rule and the
programme's statistics, a one-time pass reaches installs that already hold them,
and a backfill cycle can be trimmed by hand (it could only be deleted).

Register item 618: the cut dropped what it removed, so one made at a higher stop
threshold stayed when the threshold was lowered (only Undo import restored it).
It now keeps those readings, and a stop threshold change puts them back and cuts
again from the full trace.
"""
from __future__ import annotations

import asyncio
import copy
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from homeassistant.core import HomeAssistant
from homeassistant.helpers.json import json_dumps

from custom_components.ha_washdata import store_account, trace_codec, ws_api
from custom_components.ha_washdata.const import (
    BACKFILL_TAIL_CUT_KEY,
    BACKFILL_TAIL_TRACE_KEY,
    STORAGE_KEY,
    STORAGE_VERSION,
)
from custom_components.ha_washdata.profile_store import ProfileStore
from custom_components.ha_washdata.signal_processing import integrate_cycle_wh
from custom_components.ha_washdata.store import StoreBridge

STOP = 1.2
# The programme: 20 min heating, wash to 5400 s, 3000 s of silent drying, then the
# terminal pump-out (40 W) at 8400-8430 s. Live, it is stored ending at 8460 s.
_RUN = [[float(t), 1900.0 if t < 1200 else 60.0] for t in range(0, 5401, 30)]
_DRY = [[float(t), 0.4] for t in range(5460, 8400, 60)]
_PUMP = [[8400.0, 40.0], [8430.0, 40.0], [8460.0, 0.4]]
LIVE_S = 8460.0
PUMP_END_S = 8430.0
# An import then waits the dishwasher's 3600 s min_off_gap before its timeout.
_WAIT = [[LIVE_S + 60.0 * k, 0.4] for k in range(1, 61)]
IMPORT_S = LIVE_S + 3600.0


def _live(i: int) -> dict[str, Any]:
    return {
        "id": f"live{i}", "profile_name": "Eco", "status": "completed",
        "termination_reason": "smart", "start_time": f"2026-10-2{i}T08:00:00+00:00",
        "duration": LIVE_S, "power_data": [list(p) for p in _RUN + _DRY + _PUMP],
    }


def _imported(i: int, profile: str | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "id": f"bf{i}", "profile_name": profile, "status": "completed",
        "termination_reason": "timeout", "start_time": f"2026-10-1{i}T08:00:00+00:00",
        "duration": IMPORT_S, "max_power": 1900.0, "energy_wh": 999.0,
        "power_data": [list(p) for p in _RUN + _DRY + _PUMP + _WAIT],
        "meta": {"source": "history_import"}, **extra,
    }


async def _store(
    hass: HomeAssistant, *, device_type: str = "dishwasher", live: int = 3,
    imported: list[dict[str, Any]] | None = None, stop: float | None = STOP,
    key: str = "test_518",
) -> ProfileStore:
    store = ProfileStore(hass, key)
    await store.async_load()
    store.device_type = device_type
    store.stop_threshold_w = stop
    store._data["profiles"]["Eco"] = {"avg_duration": LIVE_S}
    store._data["past_cycles"].extend(_live(i) for i in range(live))
    store._data["backfill_cycles"].extend(imported or [])
    return store


def _bf(store: ProfileStore, cid: str) -> dict[str, Any]:
    cycle, origin = store.find_stored_cycle(cid)
    assert origin == "backfill"
    return cycle


def _avg(store: ProfileStore) -> float:
    return float(store._data["profiles"]["Eco"]["avg_duration"])


async def test_a_label_cuts_the_wait_to_the_pump_out(hass: HomeAssistant) -> None:
    store = await _store(hass, imported=[_imported(0)])
    await store.async_rebuild_envelope("Eco")
    assert _avg(store) == LIVE_S

    await store.assign_profile_to_cycle("bf0", "Eco")
    cycle = _bf(store, "bf0")
    # The programme measured its drying before the pump-out (from the traces, which
    # the wait after it cannot move), so the run ends with that pump-out.
    assert cycle["duration"] == PUMP_END_S
    assert max(p[0] for p in cycle["power_data"]) <= PUMP_END_S
    assert cycle["meta"] == {"source": "history_import", "tail_cut_s": IMPORT_S - PUMP_END_S}
    assert cycle["energy_wh"] < 999.0
    assert cycle["profile_name"] == "Eco"
    # The rebuild the label triggered never learned the wait.
    assert abs(_avg(store) - LIVE_S) < 30


async def test_installs_that_already_hold_them_are_cut_once(hass: HomeAssistant) -> None:
    """Labelled before the cut existed: the one-time pass, then nothing more."""
    labelled = [_imported(i, "Eco") for i in range(4)]
    store = await _store(hass, imported=labelled, stop=None)
    await store.async_rebuild_envelope("Eco")   # the store cannot judge: no cut
    inflated = _avg(store)
    assert inflated > LIVE_S + 1500   # the #518 shape: 8,461 -> 10,193 s
    assert store.backfill_tail_cut_pending() is True   # absent = pending

    store.stop_threshold_w = STOP
    result = await store.async_cut_backfill_tails()
    assert result["cut"] == 4 and result["profiles"] == 1
    assert store.backfill_tail_cut_pending() is False
    assert store._data[BACKFILL_TAIL_CUT_KEY] is False
    assert all(_bf(store, f"bf{i}")["duration"] == PUMP_END_S for i in range(4))
    assert abs(_avg(store) - LIVE_S) < 30 < inflated - LIVE_S

    # Idempotent: a second pass, and any later rebuild, finds nothing to cut.
    before = [(c["duration"], len(c["power_data"]), dict(c["meta"])) for c in labelled]
    store._data[BACKFILL_TAIL_CUT_KEY] = True
    assert (await store.async_cut_backfill_tails())["cut"] == 0
    assert store._cut_backfill_tails_nosave("Eco") == (0, 0.0)
    await store.async_rebuild_envelope("Eco")
    assert [(c["duration"], len(c["power_data"]), dict(c["meta"])) for c in labelled] == before


async def test_a_cut_cycle_is_never_cut_again(hass: HomeAssistant) -> None:
    """The stamp, not the rule's verdict, decides: different statistics later (a
    shorter span, a lower floor) must not take a second slice off the same cycle."""
    store = await _store(hass, imported=[_imported(0, "Eco", meta={
        "source": "history_import", "tail_cut_s": 120.0,
    })])
    assert store._backfill_tail_candidates("Eco") == []
    await store.async_rebuild_envelope("Eco")
    assert _bf(store, "bf0")["duration"] == IMPORT_S


async def test_a_cycle_the_user_trimmed_or_corrected_is_theirs(hass: HomeAssistant) -> None:
    trimmed = _imported(0, "Eco")
    corrected = _imported(1, "Eco", manual_duration=11000.0)
    store = await _store(hass, imported=[trimmed, corrected])
    # The user keeps more than the rule would: their trim is the statement.
    assert await store.trim_cycle_power_data("bf0", 0.0, 11000.0) is True
    assert trimmed["meta"]["edited"] is True
    await store.async_rebuild_envelope("Eco")
    assert (await store.async_cut_backfill_tails())["cut"] == 0
    assert trimmed["duration"] == 10980.0   # snapped to its last sample, not cut
    assert corrected["duration"] == IMPORT_S
    assert "tail_cut_s" not in trimmed["meta"] and "tail_cut_s" not in corrected["meta"]


async def test_only_a_dishwashers_timeout_finish_is_cut(hass: HomeAssistant) -> None:
    washer = await _store(hass, device_type="washing_machine", imported=[_imported(0, "Eco")])
    await washer.async_rebuild_envelope("Eco")
    assert (await washer.async_cut_backfill_tails())["cut"] == 0
    assert _bf(washer, "bf0")["duration"] == IMPORT_S

    smart = _imported(1, "Eco", termination_reason="smart")
    dishwasher = await _store(hass, imported=[smart])
    await dishwasher.async_rebuild_envelope("Eco")
    assert smart["duration"] == IMPORT_S


async def test_without_a_measured_span_the_other_cycles_bound_it_not_the_mean(
    hass: HomeAssistant,
) -> None:
    """No pump-out to measure a drying span on: fall back to the length of the
    profile's OTHER cycles. Never `avg_duration` - it holds the waits being cut."""
    flat_live = [[float(t), 1900.0 if t < 1200 else 60.0] for t in range(0, 5401, 30)]
    flat_live.append([5430.0, 0.4])
    store = await _store(hass, live=0, imported=[
        {**_imported(i, "Eco"), "power_data": [list(p) for p in flat_live] + [
            [5430.0 + 60.0 * k, 0.4] for k in range(1, 61)
        ], "duration": 5430.0 + 3600.0}
        for i in range(4)
    ], stop=None)
    store._data["past_cycles"].extend(
        {**_live(i), "duration": 5430.0, "power_data": [list(p) for p in flat_live]}
        for i in range(2)
    )
    await store.async_rebuild_envelope("Eco")
    assert store.profile_terminal_quiet_seconds("Eco") is None
    assert _avg(store) > 7500   # the inflated mean the cut must not anchor on

    store.stop_threshold_w = STOP
    assert (await store.async_cut_backfill_tails())["cut"] == 4
    assert all(_bf(store, f"bf{i}")["duration"] == 5430.0 for i in range(4))
    assert abs(_avg(store) - 5430.0) < 30


async def test_a_programme_with_no_evidence_waits_for_some(hass: HomeAssistant) -> None:
    """A first import labelled into a new programme: one or two cycles say nothing,
    so nothing is cut (and nothing stamped); the third measures the drying span
    and the rebuild it triggers cuts all three."""
    store = await _store(hass, live=0, imported=[_imported(i) for i in range(3)])
    await store.assign_profile_to_cycle("bf0", "Eco")
    await store.assign_profile_to_cycle("bf1", "Eco")
    assert [_bf(store, f"bf{i}")["duration"] for i in range(2)] == [IMPORT_S] * 2
    assert "tail_cut_s" not in _bf(store, "bf0")["meta"]

    await store.assign_profile_to_cycle("bf2", "Eco")
    assert [_bf(store, f"bf{i}")["duration"] for i in range(3)] == [PUMP_END_S] * 3
    assert abs(_avg(store) - PUMP_END_S) < 30


async def test_the_bulk_auto_label_cuts_too(hass: HomeAssistant) -> None:
    store = await _store(hass, imported=[_imported(0)])
    await store.async_rebuild_envelope("Eco")
    await store.auto_label_cycles(confidence_threshold=0.0)
    cycle = _bf(store, "bf0")
    assert cycle["profile_name"] == "Eco"
    assert cycle["duration"] == PUMP_END_S


async def test_an_import_cut_is_stamped_so_the_label_does_not_cut_it_again(
    hass: HomeAssistant,
) -> None:
    store = await _store(hass)
    cycle = _imported(0)
    store._data["backfill_cycles"].append(cycle)
    store._apply_backfill_tail_cut(cycle, LIVE_S + 900.0)   # what ws_api applies
    assert cycle["meta"]["tail_cut_s"] == 2700.0
    cycle["profile_name"] = "Eco"
    assert store._backfill_tail_candidates("Eco") == []


# ─── Part 3: a backfill cycle can be trimmed ─────────────────────────────────


async def test_a_backfill_cycle_can_be_trimmed_and_its_profile_follows(
    hass: HomeAssistant,
) -> None:
    store = await _store(hass, imported=[_imported(0, "Eco")], stop=None)
    await store.async_rebuild_envelope("Eco")
    assert store._data["profiles"]["Eco"]["max_duration"] == IMPORT_S

    assert await store.trim_cycle_power_data("bf0", 0.0, PUMP_END_S) is True
    cycle = _bf(store, "bf0")
    assert cycle["duration"] == PUMP_END_S
    assert cycle["meta"]["edited"] is True and cycle["meta"]["trim"] == [0.0, PUMP_END_S]
    assert cycle["meta"]["source"] == "history_import"
    assert cycle in store.get_backfill_cycles()   # still an import, not a real cycle
    assert len(store.get_past_cycles()) == 3
    # The profile was rebuilt from the trimmed trace.
    assert store._data["profiles"]["Eco"]["max_duration"] == LIVE_S


async def test_a_community_reference_cycle_stays_as_downloaded(hass: HomeAssistant) -> None:
    store = await _store(hass)
    ref = {**_live(9), "id": "ref0", "meta": {"source": "store:x"}}
    store._data["reference_cycles"].append(ref)
    assert await store.trim_cycle_power_data("ref0", 0.0, 5400.0) is False
    assert ref["duration"] == LIVE_S


# ─── Register item 618: a cut made at another stop threshold is redone ───────

HIGH_STOP = 2.5


def _integrated(cycle: dict[str, Any]) -> dict[str, Any]:
    """The energy a real import stores: integrated over its own trace."""
    ts = np.asarray([p[0] for p in cycle["power_data"]], dtype=float)
    pw = np.asarray([p[1] for p in cycle["power_data"]], dtype=float)
    cycle["energy_wh"] = round(integrate_cycle_wh(ts, pw), 3)
    return cycle


def _standby(i: int, profile: str | None = "Eco") -> dict[str, Any]:
    """The reporter's plug: after the pump-out it idles at 1.7 W for the whole wait,
    quiet at a 2.5 W stop and activity at 1.2 W."""
    cycle = _imported(i, profile)
    cycle["power_data"] = [[t, 1.7 if t > LIVE_S else p] for t, p in cycle["power_data"]]
    return _integrated(cycle)


async def _recut_at(store: ProfileStore, stop: float) -> dict[str, Any]:
    """What a Stop Threshold change does: the manager pins the level and re-arms."""
    store.stop_threshold_w = stop
    store.arm_backfill_tail_cut()
    return await store.async_cut_backfill_tails()


async def test_a_cut_at_a_higher_stop_is_undone_when_the_stop_is_lowered(
    hass: HomeAssistant,
) -> None:
    """At 2.5 W the standby reads as quiet and the wait is cut at the pump-out; at
    1.2 W it is activity to the very end, so the rule declines: the cycle is again
    exactly what the import stored, not stuck with the 2.5 W cut."""
    standby = _standby(2)
    uncut = copy.deepcopy(standby)
    store = await _store(hass, imported=[standby], stop=HIGH_STOP)
    assert (await store.async_cut_backfill_tails())["cut"] == 1
    assert standby["duration"] == PUMP_END_S
    kept = standby[BACKFILL_TAIL_TRACE_KEY]
    assert kept["points"] == [p for p in uncut["power_data"] if p[0] > PUMP_END_S]
    assert kept["stop_w"] == HIGH_STOP and kept["duration"] == IMPORT_S
    assert kept["end_row"] is False   # the cut ends on the pump-out's own sample

    result = await _recut_at(store, STOP)
    assert result["restored"] == 1 and result["cut"] == 0 and result["profiles"] == 1
    for field in ("duration", "power_data", "energy_wh", "meta"):
        assert standby[field] == uncut[field], field
    assert "end_time" not in standby and BACKFILL_TAIL_TRACE_KEY not in standby
    # Declined, so unstamped: the next rebuild may judge it again, and finds nothing.
    await store.async_rebuild_envelope("Eco")
    assert standby["duration"] == IMPORT_S
    # Same level again: nothing to redo.
    assert (await _recut_at(store, STOP))["restored"] == 0


async def test_a_lowered_stop_redoes_the_cut_where_a_fresh_cut_would_end(
    hass: HomeAssistant,
) -> None:
    """No drying span to measure, so the cut falls back to the programme length,
    never before the last activity. At 2.5 W the 1.7 W standby after the run is
    quiet (cut at 5430 s, a terminal row appended); at 1.2 W it is activity up to
    6600 s. The redo must land exactly where a never-cut import cut at 1.2 W does."""
    flat = [[float(t), 1900.0 if t < 1200 else 60.0] for t in range(0, 5401, 30)]

    def _import() -> dict[str, Any]:
        trace = (
            [list(p) for p in flat]
            + [[float(t), 1.7] for t in range(5460, 6601, 60)]
            + [[float(t), 0.4] for t in range(6660, 9031, 60)]
        )
        return _integrated({**_imported(0, "Eco"), "power_data": trace, "duration": 9030.0})

    def _short_live(store: ProfileStore) -> None:
        store._data["past_cycles"].extend(
            {**_live(i), "duration": 5430.0,
             "power_data": [list(p) for p in flat] + [[5430.0, 0.4]]}
            for i in range(2)
        )

    redo = _import()
    store = await _store(hass, live=0, imported=[redo], stop=HIGH_STOP)
    _short_live(store)
    fresh = _import()
    reference = await _store(hass, live=0, imported=[fresh], stop=STOP, key="test_618_fresh")
    _short_live(reference)

    assert (await store.async_cut_backfill_tails())["cut"] == 1
    assert redo["duration"] == 5430.0 and redo["power_data"][-1] == [5430.0, 1.7]
    assert redo[BACKFILL_TAIL_TRACE_KEY]["end_row"] is True

    result = await _recut_at(store, STOP)
    assert result["restored"] == 1 and result["cut"] == 1
    assert (await reference.async_cut_backfill_tails())["cut"] == 1
    assert fresh["duration"] == 6600.0   # the fresh cut, at the new level
    for field in (
        "duration", "power_data", "energy_wh", "signature", "end_time", "meta",
        BACKFILL_TAIL_TRACE_KEY,
    ):
        assert redo[field] == fresh[field], field
    assert redo[BACKFILL_TAIL_TRACE_KEY]["stop_w"] == STOP
    assert store._data["profiles"]["Eco"]["avg_duration"] == (
        reference._data["profiles"]["Eco"]["avg_duration"]
    )


async def test_a_cycle_the_user_changed_after_the_cut_keeps_their_change(
    hass: HomeAssistant,
) -> None:
    corrected, trimmed, untouched = _standby(0), _standby(1), _standby(2)
    store = await _store(hass, imported=[corrected, trimmed, untouched], stop=HIGH_STOP)
    assert (await store.async_cut_backfill_tails())["cut"] == 3
    assert all(BACKFILL_TAIL_TRACE_KEY in c for c in (corrected, trimmed, untouched))
    corrected["manual_duration"] = 9000.0
    assert await store.trim_cycle_power_data("bf1", 0.0, 8400.0) is True

    result = await _recut_at(store, STOP)
    assert result["restored"] == 1
    assert untouched["duration"] == IMPORT_S
    assert corrected["duration"] == PUMP_END_S and corrected["meta"]["tail_cut_s"]
    assert trimmed["duration"] == 8400.0 and trimmed["meta"]["edited"] is True
    assert max(p[0] for p in trimmed["power_data"]) == 8400.0


async def test_the_kept_tail_is_exported_written_compact_and_kept_from_the_panel(
    hass: HomeAssistant,
) -> None:
    standby = _standby(2)
    src = await _store(hass, imported=[standby], stop=HIGH_STOP)
    await src.async_cut_backfill_tails()
    kept = copy.deepcopy(standby[BACKFILL_TAIL_TRACE_KEY])

    # On disk it is compact text, like the trace beside it, and reads back exactly.
    disk = trace_codec.encode_store_payload(src._data)
    assert isinstance(disk["backfill_cycles"][0][BACKFILL_TAIL_TRACE_KEY], str)
    assert trace_codec.decode_store_payload(json.loads(json_dumps(disk))) >= 2
    # The panel's cycle list never carries it.
    assert BACKFILL_TAIL_TRACE_KEY not in ws_api._strip_cycle(standby)

    # Both exports carry it, and the selective import keeps it on the copy.
    assert src.export_data()["data"]["backfill_cycles"][0][BACKFILL_TAIL_TRACE_KEY] == kept
    dw = {"device_type": "dishwasher"}
    cats = {"categories": ["profiles", "backfill_cycles"]}
    payload = json.loads(json_dumps(
        src.export_data(entry_data=dw, entry_options=dw, selection=cats)
    ))
    dst = await _store(hass, imported=[], stop=STOP, key="test_618_dst")
    summary = await dst.async_import_data_selective(
        payload, selection=cats, local_device_type="dishwasher",
    )
    assert summary["backfill_cycles_imported"] == 1
    [copied] = dst.get_backfill_cycles()
    assert copied[BACKFILL_TAIL_TRACE_KEY] == kept and copied["duration"] == PUMP_END_S
    # ...so the target, at its lower stop, can still undo the cut.
    assert (await dst.async_cut_backfill_tails())["restored"] == 1
    assert copied["duration"] == IMPORT_S


class _UploadRecorder:
    """Stands in for the community-store client: records what would be sent."""

    def __init__(self) -> None:
        self.write_lock = asyncio.Lock()
        self.sent: list[Any] = []

    async def upload_reference_cycle(self, *args: Any) -> str:
        self.sent.append(args)
        return "store-cycle"

    async def upload_device_bundle(self, *args: Any) -> dict[str, Any]:
        self.sent.append(args)
        return {"ok": True, "cycle_ids": ["store-cycle"]}

    def last_error(self) -> None:
        return None


async def test_a_shared_cycle_never_carries_the_kept_tail(hass: HomeAssistant) -> None:
    """The uploads are built from the cut trace and whitelisted stats only."""
    standby = _standby(2)
    store = await _store(hass, imported=[standby], stop=HIGH_STOP)
    await store.async_cut_backfill_tails()
    assert standby[BACKFILL_TAIL_TRACE_KEY]["points"]

    hass.data[store_account._DATA_KEY] = {
        "store": MagicMock(async_save=AsyncMock()),
        "data": {"online_enabled": True, "account": {"refresh_token": "rt", "uid": "u1"}},
    }
    client = _UploadRecorder()
    with patch("custom_components.ha_washdata.store.get_client", return_value=client):
        bridge = StoreBridge(hass, store)
    assert (await bridge.share_cycle("bf2", "Eco", "Beko", "X", "dishwasher")) == {
        "store_cycle_id": "store-cycle"
    }
    await bridge.share_device("Beko", "X", "dishwasher", [
        {"local_cycle_id": "bf2", "program": "Eco"},
    ])
    assert len(client.sent) == 2
    sent = json.dumps(client.sent, default=str)
    assert BACKFILL_TAIL_TRACE_KEY not in sent and "end_row" not in sent
    points = client.sent[0][4]
    assert max(p[0] for p in points) <= PUMP_END_S


# ─── Wiring: the manager pushes the level and runs the pass once ─────────────


async def test_a_real_boot_cuts_what_an_older_install_labelled(
    hass: HomeAssistant, enable_custom_integrations: None, hass_storage: dict[str, Any],
) -> None:
    from pytest_homeassistant_custom_component.common import MockConfigEntry

    from custom_components.ha_washdata.const import (
        CONFIG_ENTRY_MINOR_VERSION,
        CONFIG_ENTRY_VERSION,
        DOMAIN,
    )

    hass.states.async_set("sensor.dw_power", "0", {"unit_of_measurement": "W"})
    entry = MockConfigEntry(
        domain=DOMAIN, title="Dishwasher", unique_id="washdata_dw_518",
        data={"name": "Dishwasher", "power_sensor": "sensor.dw_power",
              "device_type": "dishwasher"},
        options={"device_type": "dishwasher", "min_power": 2.0, "stop_threshold_w": STOP},
        version=CONFIG_ENTRY_VERSION, minor_version=CONFIG_ENTRY_MINOR_VERSION,
    )
    entry.add_to_hass(hass)
    # The reporter's plug: after the pump-out one import sits on a 1.7 W standby,
    # above the 1.2 W stop, so at that level its activity never ends.
    standby = _imported(2, "Eco")
    standby["power_data"] = [[t, 1.7 if t > LIVE_S else p] for t, p in standby["power_data"]]
    hass_storage[f"{STORAGE_KEY}.{entry.entry_id}"] = {
        "version": STORAGE_VERSION, "minor_version": 1,
        "key": f"{STORAGE_KEY}.{entry.entry_id}",
        "data": {
            "profiles": {"Eco": {"avg_duration": 10193.0}},
            "past_cycles": [_live(i) for i in range(3)],
            "backfill_cycles": [_imported(i, "Eco") for i in range(2)] + [standby],
        },
    }
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    assert store.stop_threshold_w == manager.detector.config.stop_threshold_w == STOP
    assert store.backfill_tail_cut_pending() is False
    durations = {c["id"]: c["duration"] for c in store.get_backfill_cycles()}
    assert durations == {"bf0": PUMP_END_S, "bf1": PUMP_END_S, "bf2": IMPORT_S}

    # The stop raised above the standby (the reporter's workaround): cut now, not
    # at the programme's next cycle end.
    hass.config_entries.async_update_entry(
        entry, options={**entry.options, "stop_threshold_w": 2.5}
    )
    await hass.async_block_till_done()
    assert store.stop_threshold_w == 2.5
    assert store.backfill_tail_cut_pending() is False
    assert [c["duration"] for c in store.get_backfill_cycles()] == [PUMP_END_S] * 3
    assert abs(float(store._data["profiles"]["Eco"]["avg_duration"]) - LIVE_S) < 30

    # Lowered again (register item 618): the standby is activity at 1.2 W, so the
    # cut made at 2.5 W is undone from its kept readings and the rule declines it.
    hass.config_entries.async_update_entry(
        entry, options={**entry.options, "stop_threshold_w": STOP}
    )
    await hass.async_block_till_done()
    assert store.stop_threshold_w == STOP
    durations = {c["id"]: c["duration"] for c in store.get_backfill_cycles()}
    assert durations == {"bf0": PUMP_END_S, "bf1": PUMP_END_S, "bf2": IMPORT_S}
    assert "tail_cut_s" not in _bf(store, "bf2")["meta"]
    assert await hass.config_entries.async_unload(entry.entry_id)
