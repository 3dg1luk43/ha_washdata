"""#518 parts 2 and 3: history-imported dishwasher cycles kept their end wait.

The history import replays unmatched, so a dishwasher's timeout finish keeps its
whole ``min_off_gap`` wait (``keep_tail``). The import cuts it only when the
candidate confidently matches an existing programme, so on a first import nothing
was cut, and once labelled into "Eco" the reporter's four cycles (12,084-12,950 s
stored, last activity ~8,470 s) took its ``avg_duration`` from 8,461 to 10,193 s.
Now the envelope rebuild a label triggers cuts them with the same rule and the
programme's statistics, a one-time pass reaches installs that already hold them,
and a backfill cycle can be trimmed by hand (it could only be deleted).
"""
from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import (
    BACKFILL_TAIL_CUT_KEY,
    STORAGE_KEY,
    STORAGE_VERSION,
)
from custom_components.ha_washdata.profile_store import ProfileStore

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
) -> ProfileStore:
    store = ProfileStore(hass, "test_518")
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
    assert await hass.config_entries.async_unload(entry.entry_id)
