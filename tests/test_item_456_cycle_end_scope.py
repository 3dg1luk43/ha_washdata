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
"""Register item 456: the cycle end does only what one finished cycle invalidates.

The budgets in ``test_perf_budgets.py`` count the work; these pin that doing less
of it changed nothing a user can see: the stored data is what the full
maintenance pass would have produced, the cycle is on disk as early as before,
a debounced write is never lost on unload, the rebuild memo rebuilds whenever
its inputs move, and the per-profile sensors still follow their profile.
"""

from __future__ import annotations

import copy
import hashlib
from datetime import timedelta
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from custom_components.ha_washdata.const import DOMAIN
from custom_components.ha_washdata.profile_store import ProfileStore

from . import test_perf_budgets as perf
from .real_manager import boot, make_entry
from .test_perf_budgets import (
    CYCLES_PER_PROFILE,
    PROFILES,
    QUIET,
    fire_timers,
    main_key,
    program_trace,
    report,
)

# The budget module's fixtures: the same seeded store and real entity platforms
# the work counts are taken on (and its autouse real clock).
_real_clock = perf._real_clock  # noqa: SLF001
seeded = perf.seeded
washer = perf.washer
store_writes = perf.store_writes

NEW_CYCLES = PROFILES * CYCLES_PER_PROFILE + 1


def dipped_trace(p: int, k: int) -> list[list[float]]:
    """``program_trace`` with a 6-minute near-zero stretch in its wash phase.

    Against the envelope as it stands BEFORE this cycle joins it the stretch is
    not flagged, against the rebuilt one it is a pause: so the stored artifact
    exists only if the post-cycle refresh ran for the labelled profile.
    """
    heat_s = 300 + 60 * p
    return [
        [t, 8.0 if heat_s + 120 <= t < heat_s + 480 else w]
        for t, w in program_trace(p, k)
    ]


async def run_cycle(hass: HomeAssistant, freezer: Any, trace: list[list[float]]) -> None:
    for _t, w in trace:
        await report(hass, freezer, w)
        await fire_timers(hass)


async def settle(hass: HomeAssistant, freezer: Any, steps: int = 8) -> None:
    """Quiet readings until the cycle has ended and every debounced write landed."""
    for _ in range(steps):
        await report(hass, freezer, 0.0)
        await fire_timers(hass)


def without_stamps(envelopes: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {k: v for k, v in env.items() if k != "updated"}
        for name, env in envelopes.items()
    }


async def test_cycle_end_leaves_nothing_for_the_full_pass(
    hass: HomeAssistant, seeded: Any, freezer: Any, hass_storage: dict[str, Any]
) -> None:
    """After a cycle end, the full maintenance finds nothing left to change."""
    entry, mgr = seeded
    ps = mgr.profile_store
    await run_cycle(hass, freezer, dipped_trace(PROFILES - 1, 1))
    await settle(hass, freezer)
    assert len(ps.get_past_cycles()) == NEW_CYCLES
    new = ps.get_past_cycles()[-1]
    assert new["profile_name"] == f"Program {PROFILES - 1}"
    # Only the refreshed list carries this pause (see dipped_trace).
    assert [a["type"] for a in new.get("artifacts", [])] == ["pause"]

    # Everything that reached memory reached the disk.
    assert hass_storage[main_key(entry)]["data"] == ps._data  # noqa: SLF001

    # The global passes, run from scratch (memo cleared, so every profile is
    # really rebuilt): no artifact is stale and no envelope or profile moves.
    envelopes = copy.deepcopy(without_stamps(ps._data["envelopes"]))  # noqa: SLF001
    profiles = copy.deepcopy(ps.get_profiles())
    ps.__dict__.pop("_envelope_built_from", None)
    stats = await ps.async_run_maintenance()
    assert stats["refreshed_artifacts"] == 0
    assert stats["orphaned_profiles"] == 0
    assert without_stamps(ps._data["envelopes"]) == envelopes  # noqa: SLF001
    assert ps.get_profiles() == profiles


async def test_finished_cycle_is_on_disk_before_the_debounced_write(
    hass: HomeAssistant, seeded: Any, freezer: Any,
    hass_storage: dict[str, Any], store_writes: list[tuple[str, int]],
) -> None:
    """One immediate write carries the cycle; the follow-ups' write comes later."""
    entry, mgr = seeded
    ps = mgr.profile_store
    await run_cycle(hass, freezer, program_trace(PROFILES - 1, 1))
    store_writes.clear()

    def main_writes() -> int:
        return sum(k == main_key(entry) for k, _ in store_writes)

    for _ in range(5):
        await report(hass, freezer, 0.0)
        if len(ps.get_past_cycles()) == NEW_CYCLES:
            break
        await fire_timers(hass)
        if len(ps.get_past_cycles()) == NEW_CYCLES:
            break
    assert len(ps.get_past_cycles()) == NEW_CYCLES
    new = ps.get_past_cycles()[-1]

    # The cycle end has run to completion and the clock has not moved since: one
    # write, and it already holds the cycle, its counters and its envelope.
    assert main_writes() == 1
    on_disk = hass_storage[main_key(entry)]["data"]
    assert new["id"] in {c["id"] for c in on_disk["past_cycles"]}
    assert on_disk["lifetime_energy_wh"] == ps.get_lifetime_energy_wh()
    assert on_disk["lifetime_cycle_count"] == ps.get_lifetime_cycle_count()
    assert on_disk["envelopes"][new["profile_name"]] == ps.get_envelope(new["profile_name"])

    # Whatever the follow-ups changed lands with at most one more write.
    await settle(hass, freezer)
    assert main_writes() <= 2
    assert hass_storage[main_key(entry)]["data"] == ps._data  # noqa: SLF001


async def test_coalesced_saves_flush_and_expire(
    hass: HomeAssistant, enable_custom_integrations: None, freezer: Any,
    hass_storage: dict[str, Any],
) -> None:
    entry = make_entry(hass, QUIET)
    mgr = await boot(hass, entry)
    ps = mgr.profile_store
    await ps.async_save()

    def on_disk() -> dict[str, Any]:
        return hass_storage[main_key(entry)]["data"]

    ps.coalesce_saves()
    ps._data["marker"] = 1  # noqa: SLF001
    await ps.async_save()
    assert "marker" not in on_disk()  # debounced
    await ps.async_flush_saves()
    assert on_disk()["marker"] == 1  # flushed at once

    ps._data["marker"] = 2  # noqa: SLF001
    await ps.async_save()
    assert on_disk()["marker"] == 1
    freezer.tick(timedelta(seconds=11))
    await fire_timers(hass)
    assert on_disk()["marker"] == 2  # the debounce fired

    freezer.tick(timedelta(seconds=120))
    ps._data["marker"] = 3  # noqa: SLF001
    await ps.async_save()
    assert on_disk()["marker"] == 3  # the window has closed: immediate again

    # Unload: a pending debounced write lands before the store is let go.
    ps.coalesce_saves()
    ps._data["marker"] = 4  # noqa: SLF001
    await ps.async_save()
    assert on_disk()["marker"] == 3
    await mgr.async_shutdown()
    assert on_disk()["marker"] == 4
    ps._data["marker"] = 5  # noqa: SLF001
    await ps.async_save()
    assert on_disk()["marker"] == 5  # and nothing is debounced after it


async def test_rebuild_memo_rebuilds_whenever_an_input_moves(
    hass: HomeAssistant, seeded: Any, monkeypatch: Any
) -> None:
    _entry, mgr = seeded
    ps = mgr.profile_store
    builds: list[str] = []
    real = ProfileStore._rebuild_envelope_sync

    def counted(self: ProfileStore, cycles: list[Any]) -> Any:
        builds.append("build")
        return real(self, cycles)

    monkeypatch.setattr(ProfileStore, "_rebuild_envelope_sync", counted)
    name = "Program 1"
    cycles = [c for c in ps.get_past_cycles() if c["profile_name"] == name]

    async def rebuilt() -> bool:
        builds.clear()
        assert await ps.async_rebuild_envelope(name)
        return bool(builds)

    assert not await rebuilt()  # seeded store: nothing moved since the seed build

    cycles[0]["manual_duration"] = 999.0
    assert await rebuilt()
    assert not await rebuilt()

    cycles[1]["power_data"] = [list(p) for p in cycles[1]["power_data"]][:-1]
    assert await rebuilt()

    cycles[2]["power_data"].append([cycles[2]["power_data"][-1][0] + 60, 40.0])
    assert await rebuilt()  # in place: length and last point moved

    cycles[0]["ml_review"] = {"golden": True}
    assert await rebuilt()

    ps.get_profiles()[name]["avg_duration"] = 1.0
    assert await rebuilt()  # a hand edit of the stats it writes

    ps.dtw_bandwidth = ps.dtw_bandwidth + 0.05
    assert await rebuilt()

    ps._data["envelopes"][name] = dict(ps._data["envelopes"][name])  # noqa: SLF001
    assert await rebuilt()  # the envelope it built was replaced

    cycles[1]["profile_name"] = "Program 2"
    assert await rebuilt()  # a cycle left the profile
    assert not await rebuilt()


async def test_reference_curve_memo_follows_the_envelope(
    hass: HomeAssistant, seeded: Any, monkeypatch: Any
) -> None:
    _entry, mgr = seeded
    ps = mgr.profile_store
    computed: list[int] = []
    real = ProfileStore._compute_reference_curve

    def counted(env: Any, n: int) -> Any:
        computed.append(n)
        return real(env, n)

    monkeypatch.setattr(ProfileStore, "_compute_reference_curve", staticmethod(counted))
    first = ps.reference_curve("Program 2")
    assert first is not None
    first["points"][0][1] = -1.0  # a consumer mutating its copy
    second = ps.reference_curve("Program 2")
    assert len(computed) == 1
    assert second is not None and second["points"][0][1] != -1.0

    cycle = next(c for c in ps.get_past_cycles() if c["profile_name"] == "Program 2")
    cycle["power_data"] = [[t, w * 2] for t, w in cycle["power_data"]]
    await ps.async_rebuild_envelope("Program 2")
    third = ps.reference_curve("Program 2")
    assert len(computed) == 2
    assert third != second


async def test_profile_count_sensor_still_follows_its_profile(
    hass: HomeAssistant, washer: Any, freezer: Any
) -> None:
    """Off the live signal, but rewritten as soon as the profile's count moves."""
    entry, mgr = washer
    name = f"Program {PROFILES - 1}"
    token = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    entity_id = er.async_get(hass).async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_profile_count_{token}"
    )
    assert entity_id is not None
    assert hass.states.get(entity_id).state == str(CYCLES_PER_PROFILE)
    before = hass.states.get(entity_id).last_reported

    await run_cycle(hass, freezer, program_trace(PROFILES - 1, 1))
    # Thirty live refreshes, none of them a write of this sensor.
    assert hass.states.get(entity_id).last_reported == before

    await settle(hass, freezer)
    assert len(mgr.profile_store.get_past_cycles()) == NEW_CYCLES
    assert hass.states.get(entity_id).state == str(CYCLES_PER_PROFILE + 1)
