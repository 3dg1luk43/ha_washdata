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
"""Which cycle categories count as evidence for a profile (CONF_PROFILE_EVIDENCE_SOURCES).

The same three lists feed four places that must agree, or a profile's curve and its
matching template would describe different things: the envelope build, the matcher's
snapshot pool, the matching-template pick, and `has_real_profiles`.

The dangerous half is what must NOT be gated. Profile garbage collection and sample repair
delete or re-point a profile whose `sample_cycle_id` resolves to nothing, so they read
`iter_stored_cycles` (everything) and never `iter_evidence_cycles`: a cycle the user has
stopped trusting is still a stored cycle, and gating those lookups would destroy a
backfill-only profile the moment someone unticked imported history.
"""
from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.const import (
    EVIDENCE_BACKFILL_CYCLES,
    EVIDENCE_REAL_CYCLES,
    EVIDENCE_REFERENCE_CYCLES,
    PROFILE_EVIDENCE_SOURCES,
)
from custom_components.ha_washdata.profile_store import ProfileStore

BASE = datetime(2026, 5, 1, 8, 0, tzinfo=timezone.utc)
PROFILE = "Cotton 40"


@pytest.fixture
def mock_hass():
    hass = MagicMock()

    async def _exec(func, *args, **kwargs):
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)

    hass.async_add_executor_job = AsyncMock(side_effect=_exec)
    hass.async_create_task = lambda coro, *a: asyncio.create_task(coro)
    return hass


def _trace(watts, n=61, dur=3600):
    step = dur / (n - 1)
    return [[i * step, float(watts)] for i in range(n)]


@pytest.fixture
def store(mock_hass):
    """One labelled cycle of each kind, all on the same profile."""
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(mock_hass, "e", min_duration_ratio=0.0, max_duration_ratio=3.0)
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()

    def _cycle(cid, watts, source=None):
        c = {
            "id": cid, "profile_name": PROFILE, "duration": 3600, "status": "completed",
            "start_time": BASE.isoformat(),
            "end_time": (BASE + timedelta(seconds=3600)).isoformat(),
            "power_data": _trace(watts), "energy_wh": 1000.0,
        }
        if source:
            c["meta"] = {"source": source}
        return c

    ps._data["profiles"] = {PROFILE: {"avg_duration": 3600, "sample_cycle_id": "real1"}}
    ps._data["past_cycles"] = [_cycle("real1", 2000)]
    ps._data["reference_cycles"] = [_cycle("ref1", 1800, "store:abc")]
    ps._data["backfill_cycles"] = [_cycle("back1", 1600, "history_import")]
    return ps


# ─── The selection itself ─────────────────────────────────────────────────────


def test_default_is_every_category(store):
    """An upgrade must change nothing, so the default is the pre-setting behaviour."""
    assert store.evidence_sources == tuple(PROFILE_EVIDENCE_SOURCES)
    assert {c["id"] for c in store.iter_evidence_cycles()} == {"real1", "ref1", "back1"}


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ([EVIDENCE_REAL_CYCLES], {"real1"}),
        ([EVIDENCE_REFERENCE_CYCLES], {"ref1"}),
        ([EVIDENCE_BACKFILL_CYCLES], {"back1"}),
        ([EVIDENCE_REAL_CYCLES, EVIDENCE_BACKFILL_CYCLES], {"real1", "back1"}),
    ],
)
def test_selection_filters_the_evidence_view(store, selection, expected):
    store.evidence_sources = selection
    assert {c["id"] for c in store.iter_evidence_cycles()} == expected
    # The ungated view is unchanged: every cycle still exists.
    assert {c["id"] for c in store.iter_stored_cycles()} == {"real1", "ref1", "back1"}


@pytest.mark.parametrize("selection", [[], None, "", ["nonsense"], {"also_nonsense"}])
def test_an_empty_or_unknown_selection_falls_back_to_everything(store, selection):
    """With nothing allowed every envelope would be empty and no profile could ever
    match, which is worse than ignoring the setting."""
    store.evidence_sources = selection
    assert store.evidence_sources == tuple(PROFILE_EVIDENCE_SOURCES)
    assert len(store.iter_evidence_cycles()) == 3


def test_unknown_names_are_dropped_but_known_ones_kept(store):
    store.evidence_sources = [EVIDENCE_REAL_CYCLES, "past_cycles", "whatever"]
    assert store.evidence_sources == (EVIDENCE_REAL_CYCLES,)


# ─── The four sites that must agree ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_envelope_is_built_only_from_allowed_cycles(store):
    seen: list[list[str]] = []
    real_sync = store._rebuild_envelope_sync

    def _spy(shape_cycles):
        seen.append([c["id"] for c in shape_cycles])
        return real_sync(shape_cycles)

    with patch.object(store, "_rebuild_envelope_sync", side_effect=_spy):
        store.evidence_sources = [EVIDENCE_REAL_CYCLES]
        await store.async_rebuild_envelope(PROFILE)
        store.evidence_sources = [EVIDENCE_REFERENCE_CYCLES, EVIDENCE_BACKFILL_CYCLES]
        await store.async_rebuild_envelope(PROFILE)

    assert seen[0] == ["real1"]
    assert sorted(seen[1]) == ["back1", "ref1"]


@pytest.mark.asyncio
async def test_excluding_every_category_that_has_a_cycle_empties_the_envelope(store):
    """A profile whose only cycles are excluded has no curve - which is what excluding
    them means. It is not deleted, though: see the GC tests below."""
    store._data["backfill_cycles"] = []
    store._data["reference_cycles"] = []
    store.evidence_sources = [EVIDENCE_REFERENCE_CYCLES]  # real1 is now excluded

    assert await store.async_rebuild_envelope(PROFILE) is False
    assert PROFILE not in store._data.get("envelopes", {})
    assert PROFILE in store._data["profiles"]


@pytest.mark.asyncio
async def test_usage_statistics_still_count_the_cycles_the_machine_ran(store):
    """Statistics are not evidence. Unticking "real cycles" removes them from the curve
    without making the profile claim it has never run."""
    store.evidence_sources = [EVIDENCE_BACKFILL_CYCLES]
    await store.async_rebuild_envelope(PROFILE)
    envelope = store.get_envelope(PROFILE)
    assert envelope
    assert envelope["cycle_count"] == 1  # real1, though it shaped nothing


def test_the_matching_template_comes_from_allowed_cycles_only(store):
    store.evidence_sources = [EVIDENCE_BACKFILL_CYCLES]
    assert store._select_reference_cycle_id(PROFILE) == "back1"
    store.evidence_sources = [EVIDENCE_REAL_CYCLES]
    assert store._select_reference_cycle_id(PROFILE) == "real1"


@pytest.mark.asyncio
async def test_the_matcher_pool_reads_the_gated_view(store):
    """The pool and the envelope must agree, so the matcher reads the same gated view."""
    with patch.object(store, "iter_evidence_cycles", wraps=store.iter_evidence_cycles) as spy:
        await store.async_match_profile(_trace(2000), 3600)
    spy.assert_called()


def test_matchability_follows_the_selection(store):
    assert store.has_real_profiles is True
    store._data["past_cycles"] = []
    store.evidence_sources = [EVIDENCE_REAL_CYCLES]
    # Only imported cycles are left and they are excluded: nothing to match against.
    assert store.has_real_profiles is False
    store.evidence_sources = [EVIDENCE_BACKFILL_CYCLES]
    assert store.has_real_profiles is True


def test_the_playground_sees_the_same_candidates_as_the_live_matcher(store):
    """The sandbox must agree with production about what a profile looks like.

    The snapshot pool was built from `past_cycles + reference_cycles`, so a profile
    sampled from a backfilled cycle produced no candidate at all and the Playground
    reported it as unmatched - a wrong answer, not a slow one.
    """
    from custom_components.ha_washdata import playground

    store._data["past_cycles"] = []
    store._data["reference_cycles"] = []
    store._data["profiles"][PROFILE]["sample_cycle_id"] = "back1"

    snaps, _config, _members, _member_snaps = playground._build_match_snapshots(store)
    assert [s["name"] for s in snaps] == [PROFILE]

    # And it honours the gate, so the sandbox cannot match on evidence the user excluded.
    store.evidence_sources = [EVIDENCE_REAL_CYCLES]
    snaps, _config, _members, _member_snaps = playground._build_match_snapshots(store)
    assert snaps == []


# ─── What must NOT be gated ───────────────────────────────────────────────────


def test_profile_gc_ignores_the_selection(store):
    """The failure this guards: unticking imported history would otherwise delete every
    profile built from it, because its sample cycle would look non-existent."""
    store._data["past_cycles"] = []
    store._data["reference_cycles"] = []
    store._data["profiles"][PROFILE]["sample_cycle_id"] = "back1"
    store.evidence_sources = [EVIDENCE_REAL_CYCLES]

    assert store.cleanup_orphaned_profiles() == 0
    assert PROFILE in store._data["profiles"]


@pytest.mark.asyncio
async def test_sample_repair_ignores_the_selection(store):
    """An excluded cycle is still a valid sample target, so repair must not re-point the
    profile at an unrelated real cycle."""
    store._data["reference_cycles"] = []
    store._data["profiles"][PROFILE]["sample_cycle_id"] = "back1"
    store._data["past_cycles"].append({
        "id": "unlabelled", "profile_name": None, "duration": 3600,
        "status": "completed", "start_time": (BASE + timedelta(days=1)).isoformat(),
        "power_data": _trace(500),
    })
    store.evidence_sources = [EVIDENCE_REAL_CYCLES]

    await store.async_repair_profile_samples()

    assert store._data["profiles"][PROFILE]["sample_cycle_id"] == "back1"
    assert next(c for c in store.get_past_cycles() if c["id"] == "unlabelled")["profile_name"] is None


def test_lookups_by_id_ignore_the_selection(store):
    """The inspector, label, delete and power-data paths must still find an excluded
    cycle - it is hidden from profiles, not from the user."""
    store.evidence_sources = [EVIDENCE_REAL_CYCLES]
    cycle, origin = store.find_stored_cycle("back1")
    assert cycle is not None and origin == "backfill"
    assert store.get_cycle_power_data("back1")


# ─── Plumbing: the store cannot read entry options ────────────────────────────
#
# The manager pushes the value in, the way it pushes energy_mode. Changing it changes
# every profile's curve, so it also forces a rebuild: envelopes are otherwise only
# rebuilt on a cycle end or a label change, and the user would tick the box and see
# nothing happen for days.

from custom_components.ha_washdata import manager as mgr_mod  # noqa: E402
from custom_components.ha_washdata.const import (  # noqa: E402
    CONF_DEVICE_TYPE,
    CONF_MIN_POWER,
    CONF_POWER_SENSOR,
    CONF_PROFILE_EVIDENCE_SOURCES,
    DEVICE_TYPE_WASHING_MACHINE,
)
from custom_components.ha_washdata.manager import WashDataManager  # noqa: E402
from homeassistant.util import dt as dt_util  # noqa: E402


@pytest.fixture
def mgr_hass():
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.bus.async_fire = MagicMock()
    hass.async_create_task = MagicMock(
        side_effect=lambda coro: getattr(coro, "close", lambda: None)()
    )
    hass.config_entries.async_get_entry = MagicMock()
    return hass


def _entry(evidence=None):
    entry = MagicMock()
    entry.entry_id = "e_evidence"
    entry.title = "Test Appliance"
    entry.options = {
        CONF_MIN_POWER: 5.0,
        CONF_POWER_SENSOR: "sensor.test_power",
        CONF_DEVICE_TYPE: DEVICE_TYPE_WASHING_MACHINE,
        "notify_finish_services": [],
    }
    if evidence is not None:
        entry.options[CONF_PROFILE_EVIDENCE_SOURCES] = evidence
    entry.data = {}
    return entry


def _manager(hass, entry):
    hass.config_entries.async_get_entry.return_value = entry
    dt_util.now.side_effect = lambda: datetime.now(timezone.utc)
    with patch("custom_components.ha_washdata.manager.ProfileStore"):
        return WashDataManager(hass, entry)


async def _reload(mgr, entry):
    mgr.profile_store.get_duration_ratio_limits.return_value = (0.1, 1.5)
    with patch.object(mgr, "_setup_external_end_trigger", AsyncMock()), \
         patch.object(mgr, "_setup_door_sensor_listener", AsyncMock()), \
         patch.object(mgr, "_setup_notify_people_listener", AsyncMock()), \
         patch.object(mgr, "_setup_maintenance_scheduler", AsyncMock()), \
         patch.object(mgr, "_setup_ml_training_scheduler", MagicMock()), \
         patch.object(mgr, "_attempt_state_restoration", AsyncMock()), \
         patch.object(mgr_mod, "async_dispatcher_send", MagicMock()):
        await mgr.async_reload_config(entry)


def test_construction_pushes_the_option_into_the_store(mgr_hass):
    entry = _entry([EVIDENCE_REAL_CYCLES])
    mgr = _manager(mgr_hass, entry)
    assert mgr.profile_store.evidence_sources == [EVIDENCE_REAL_CYCLES]


@pytest.mark.asyncio
async def test_changing_the_option_forces_a_full_envelope_rebuild(mgr_hass):
    entry = _entry()
    mgr = _manager(mgr_hass, entry)
    # The store is a MagicMock here, so give the change check a real "before" value.
    mgr.profile_store.evidence_sources = tuple(PROFILE_EVIDENCE_SOURCES)
    mgr_hass.async_create_task.reset_mock()

    entry.options[CONF_PROFILE_EVIDENCE_SOURCES] = [EVIDENCE_REAL_CYCLES]
    await _reload(mgr, entry)

    assert mgr.profile_store.async_rebuild_all_envelopes.called


@pytest.mark.asyncio
async def test_an_unchanged_option_does_not_rebuild(mgr_hass):
    """A reload happens on every settings save; rebuilding every profile each time
    would be a needless stall on a low-power host."""
    entry = _entry([EVIDENCE_REAL_CYCLES])
    mgr = _manager(mgr_hass, entry)
    mgr.profile_store.evidence_sources = [EVIDENCE_REAL_CYCLES]

    await _reload(mgr, entry)

    assert not mgr.profile_store.async_rebuild_all_envelopes.called
