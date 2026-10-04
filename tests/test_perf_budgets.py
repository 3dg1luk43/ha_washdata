"""Audit 2026-10-02 PERF-08: performance budgets that count work instead of timing it.

The ``benchmark`` marker gated nothing (one skipped test, one print), so the
event-loop stalls PERF-01/02/03 found could return unnoticed. These run the REAL
manager, ProfileStore, Store, detector and entity platforms on a fixed synthetic
store and count what each hot path does: profile-summary rebuilds and lookups,
entity state writes, store writes and their bytes, matcher runs, envelope parses
and rebuilds, trace decompressions, NumPy calls in the detector. Counts are the
same on every machine, so a regression fails CI deterministically, with no
wall-clock threshold anywhere.

Each budget is the value measured when it was written plus a stated margin, and
names the regression it guards. A deliberate change that moves one re-measures it
and updates the number and its comment in the same commit.
"""

from __future__ import annotations

import collections
import inspect
import sys
from collections.abc import AsyncIterator, Callable, Iterator
from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers import json as json_helper
from homeassistant.helpers import storage as ha_storage
from homeassistant.helpers.entity import Entity
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_fire_time_changed_exact

from custom_components.ha_washdata import PLATFORMS, cycle_detector, profile_store, progress, ws_api
from custom_components.ha_washdata.const import DOMAIN, STORAGE_KEY
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.profile_store import ProfileStore

from .real_manager import POWER, boot, make_entry

# Taken at collection, before any test runs: tests/test_logic_comprehensive.py
# assigns `dt_util.now = MagicMock(...)` and never restores it, which freezes the
# manager's clock (time_utils.utc_now reads dt_util.now) for every later test in
# the process, so a replayed cycle never advances.
_REAL_DT_NOW = dt_util.now

PROFILES = 4  # several budgets scale with this; keep them in step
CYCLES_PER_PROFILE = 3
STEP_S = 60  # plug reporting cadence of the replays
QUIET = {"notify_finish_services": [], "notify_start_services": [], "sampling_interval": 5}


# --------------------------------------------------------------------------- fixture


def program_trace(p: int, k: int) -> list[list[float]]:
    """Heat block, agitation, spin; program ``p`` runs longer than ``p - 1``."""
    heat_s, wash_s, spin_s = 300 + 60 * p, 600 + 120 * p, 300
    pts: list[list[float]] = []
    t = 0.0
    while t <= heat_s + wash_s + spin_s:
        if t < heat_s:
            w = 1800.0 + 50 * p + 10 * k
        elif t < heat_s + wash_s:
            w = 150.0 + 40 * ((t // 120) % 3)
        else:
            w = 500.0 if (t // 60) % 2 else 50.0
        pts.append([t, w])
        t += STEP_S
    return pts


async def seed_store(ps: ProfileStore) -> None:
    """PROFILES labelled programs x CYCLES_PER_PROFILE completed cycles + envelopes."""
    base = dt_util.parse_datetime("2026-09-01T08:00:00+00:00")
    assert base is not None
    for p in range(PROFILES):
        name = f"Program {p}"
        last: dict[str, Any] = {}
        for k in range(CYCLES_PER_PROFILE):
            pts = program_trace(p, k)
            start = base + timedelta(days=p * CYCLES_PER_PROFILE + k)
            last = {
                "start_time": start.isoformat(),
                "end_time": (start + timedelta(seconds=pts[-1][0])).isoformat(),
                "duration": pts[-1][0], "status": "completed",
                "profile_name": name, "power_data": pts,
                # As a live-labelled cycle carries it (else every boot re-matches it).
                "match_confidence": 0.9,
            }
            ps._add_cycle_data(last)  # noqa: SLF001 - the path a finished cycle takes
        ps.get_profiles()[name] = {"avg_duration": last["duration"], "sample_cycle_id": last["id"]}
    await ps.async_rebuild_all_envelopes()
    await ps.async_save()


async def forward_platforms(hass: HomeAssistant, entry: ConfigEntry) -> None:
    # The dev env cannot import `conversation` (no hassil); mark it loaded so the
    # manifest dependency resolves, as it is in a real install.
    hass.config.components.add("conversation")
    entry.mock_state(hass, ConfigEntryState.LOADED)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    await hass.async_block_till_done()


async def teardown(hass: HomeAssistant, entry: ConfigEntry, mgr: WashDataManager) -> None:
    if entry.state is ConfigEntryState.LOADED:
        await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
        entry.mock_state(hass, ConfigEntryState.NOT_LOADED)
    await mgr.async_shutdown()
    await hass.async_block_till_done()


async def report(hass: HomeAssistant, freezer: Any, watts: float) -> None:
    """One plug report STEP_S after the last; returns once its work has settled."""
    freezer.tick(timedelta(seconds=STEP_S))
    hass.states.async_set(POWER, str(watts), {"unit_of_measurement": "W"}, force_update=True)
    await hass.async_block_till_done()


async def fire_timers(hass: HomeAssistant) -> None:
    async_fire_time_changed_exact(hass, dt_util.utcnow())
    await hass.async_block_till_done()


def main_key(entry: ConfigEntry) -> str:
    return f"{STORAGE_KEY}.{entry.entry_id}"


@pytest.fixture(autouse=True)
def _real_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dt_util, "now", _REAL_DT_NOW)


@pytest.fixture
async def seeded(
    hass: HomeAssistant, enable_custom_integrations: None
) -> AsyncIterator[tuple[ConfigEntry, WashDataManager]]:
    """Real manager on the seeded store; no entities (replays count manager work)."""
    entry = make_entry(hass, QUIET)
    mgr = await boot(hass, entry)
    await seed_store(mgr.profile_store)
    yield entry, mgr
    await teardown(hass, entry, mgr)


@pytest.fixture
async def washer(
    hass: HomeAssistant, seeded: tuple[ConfigEntry, WashDataManager]
) -> tuple[ConfigEntry, WashDataManager]:
    """``seeded`` plus all four entity platforms."""
    await forward_platforms(hass, seeded[0])
    return seeded


# ---------------------------------------------------------------------------- spies


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> Callable[..., collections.Counter[str]]:
    """``calls(owner, "name", ...)`` counts calls to each attribute, keyed by name.

    Wraps plain, static and async methods and module functions; every wrapper
    delegates to the original. All installs share one Counter.
    """
    counter: collections.Counter[str] = collections.Counter()

    def install(owner: Any, *names: str) -> collections.Counter[str]:
        for name in names:
            raw = inspect.getattr_static(owner, name)
            real = raw.__func__ if isinstance(raw, staticmethod) else getattr(owner, name)
            if inspect.iscoroutinefunction(real):

                async def wrapper(*a: Any, _n: str = name, _f: Any = real, **kw: Any) -> Any:
                    counter[_n] += 1
                    return await _f(*a, **kw)

            else:

                def wrapper(*a: Any, _n: str = name, _f: Any = real, **kw: Any) -> Any:  # type: ignore[misc]
                    counter[_n] += 1
                    return _f(*a, **kw)

            monkeypatch.setattr(
                owner, name, staticmethod(wrapper) if isinstance(raw, staticmethod) else wrapper
            )
        return counter

    return install


@pytest.fixture
def decompressions(monkeypatch: pytest.MonkeyPatch) -> collections.Counter[str]:
    """Counts ``decompress_power_data`` (a full stored trace -> tuples) in every module."""
    counter: collections.Counter[str] = collections.Counter()
    real = profile_store.decompress_power_data

    def counted(cycle: Any) -> list[tuple[float, float]]:
        counter["decompress"] += 1
        return real(cycle)

    for name, module in list(sys.modules.items()):
        if name.startswith("custom_components.ha_washdata") and (
            getattr(module, "decompress_power_data", None) is real
        ):
            monkeypatch.setattr(module, "decompress_power_data", counted)
    return counter


@pytest.fixture
def state_writes(monkeypatch: pytest.MonkeyPatch) -> collections.Counter[str]:
    """Entity state writes by entity class (pushed and polled writes both end here)."""
    writes: collections.Counter[str] = collections.Counter()
    real = Entity._async_write_ha_state  # noqa: SLF001

    def _write(self: Entity) -> None:
        writes[type(self).__name__] += 1
        real(self)

    monkeypatch.setattr(Entity, "_async_write_ha_state", _write)
    return writes


@pytest.fixture
def store_writes(hass_storage: dict[str, Any]) -> Iterator[list[tuple[str, int]]]:
    """Every Store write as (key, bytes HA would put on disk).

    Wraps the hass_storage mock instead of monkeypatching over it: this fixture
    is torn down before hass_storage, so the real writer is always restored.
    """
    writes: list[tuple[str, int]] = []
    inner = ha_storage.Store._async_write_data  # noqa: SLF001 - the hass_storage mock

    async def _record(store: ha_storage.Store, data: dict[str, Any]) -> None:
        if "data_func" in data:
            data["data"] = data.pop("data_func")()
        _mode, payload = json_helper.prepare_save_json(data, encoder=store._encoder)  # noqa: SLF001
        writes.append((store.key, len(payload)))
        await inner(store, data)

    ha_storage.Store._async_write_data = _record  # noqa: SLF001
    try:
        yield writes
    finally:
        ha_storage.Store._async_write_data = inner  # noqa: SLF001


class _NumpyCounter:
    """Stands in for ``cycle_detector.np`` and counts every NumPy call the detector makes."""

    def __init__(self, real: Any, counter: collections.Counter[str]) -> None:
        self._real, self._counter = real, counter

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._real, name)
        if not callable(attr) or isinstance(attr, type):
            return attr

        def call(*a: Any, **kw: Any) -> Any:
            self._counter["numpy"] += 1
            return attr(*a, **kw)

        return call


def within_budget(*checks: tuple[str, float, float]) -> None:
    """Fail once, listing EVERY exceeded budget as ``name: measured > budget``."""
    over = [f"{name}: {value} > {limit}" for name, value, limit in checks if value > limit]
    assert not over, "over budget: " + "; ".join(over)


# --------------------------------------------------------------------------- PERF-01


async def test_per_notify_work(
    hass: HomeAssistant, washer: Any, calls: Any, state_writes: collections.Counter[str]
) -> None:
    """Ten ``_notify_update`` calls with a matched program: what every power event pays.

    It used to be ``3P + 3`` full ``list_profiles()`` rebuilds per notify (42 on a
    13-profile washer, ~200 ms of event loop per reading).
    """
    _entry, mgr = washer
    mgr._current_program = "Program 2"  # noqa: SLF001 - the program sensor's attribute path
    c = calls(
        ProfileStore, "_build_profile_summaries", "_profiles_fingerprint",
        "_compute_profile_power_profile", "_compute_reference_curve",
    )
    mgr._notify_update()  # noqa: SLF001 - warm-up: settle what setup left pending
    await hass.async_block_till_done()
    c.clear()
    state_writes.clear()
    n = 10
    for _ in range(n):
        mgr._notify_update()  # noqa: SLF001
    await hass.async_block_till_done()

    within_budget(
        # Summaries are rebuilt only when the store changes. Measured 0; budget 0.
        # Guards PERF-01: list_profiles/get_profile rebuilding every profile's stats.
        ("summary rebuilds", c["_build_profile_summaries"], 0),
        # Every profile lookup still fingerprints the store, O(profiles + cycles).
        # Measured 4 per notify, independent of P (the program sensor's options,
        # read twice per write, the select's, the profile-sensor manager's change
        # check); budget 5. Guards item 456 (d): with the profile-count sensors
        # back on the live signal it is P + 4, one lookup per property read 3P + 4.
        ("profile lookups", c["_profiles_fingerprint"], n * 5),
        # The power_profile attribute is memoised per envelope revision. Measured
        # 0; budget 0. Guards a NumPy resample of every profile's envelope per notify.
        ("power_profile builds", c["_compute_profile_power_profile"], 0),
        # The program sensor's reference curve, memoised per envelope revision
        # (item 456 b). Measured 0; budget 0. Guards a full-envelope
        # interpolation per state write (1 per notify unmemoised).
        ("reference curves", c["_compute_reference_curve"], 0),
        # One state write per entity on the live signal. Measured 18 per notify;
        # budget 19 (room for one new entity). The P profile-count sensors have
        # their own signal and are written only when the profile summaries change
        # (item 456 d); guards them, or a duplicate subscription, returning (18 + P).
        ("state writes", sum(state_writes.values()), n * 19),
    )


# --------------------------------------------------------------------------- PERF-02


async def test_idle_writes_only_the_clock(
    hass: HomeAssistant, washer: Any, freezer: Any, state_writes: collections.Counter[str]
) -> None:
    """Ten idle minutes: no plug report, every timer firing."""
    for _ in range(20):
        freezer.tick(timedelta(seconds=30))
        await fire_timers(hass)
    within_budget(
        # HA re-wrote all entities every 30 s, idle included (188 ms of event loop
        # per poll on a 13-profile store). Measured 20 (the elapsed-time sensor,
        # every 30 s); budget 22. Guards PERF-02: one more polled entity adds 20,
        # an idle-time notify ~22 per tick.
        ("idle state writes", sum(state_writes.values()), 22),
    )


# --------------------------------------------------------------------------- PERF-09


async def test_setup_work(
    hass: HomeAssistant, enable_custom_integrations: None, calls: Any,
    store_writes: list[tuple[str, int]],
) -> None:
    """Restart on a steady store: manager setup plus the four entity platforms.

    One setup called list_profiles() 61 times (audit PERF-09, billed to WashData's
    startup time).
    """
    entry = make_entry(hass, QUIET)
    first = await boot(hass, entry)
    await seed_store(first.profile_store)
    await first.async_shutdown()
    await hass.async_block_till_done()
    hass.data[DOMAIN].pop(entry.entry_id)

    c = calls(ProfileStore, "_build_profile_summaries", "async_match_profile")
    store_writes.clear()
    mgr = await boot(hass, entry)
    await forward_platforms(hass, entry)
    checks = (
        # Measured 1; budget 2. Guards an uncached summary path during setup.
        ("summary rebuilds", c["_build_profile_summaries"], 2),
        # Measured 0; budget 0. Guards the boot-time match_confidence backfill
        # re-matching cycles that already carry one (a full match per cycle).
        ("matcher runs", c["async_match_profile"], 0),
        # Measured 0; budget 0. Guards a steady store being rewritten every boot.
        ("main-store writes", sum(k == main_key(entry) for k, _ in store_writes), 0),
    )
    await teardown(hass, entry, mgr)
    within_budget(*checks)


# ------------------------------------------------------------ PERF-01/03/06/07, 297


async def test_running_cycle_work(
    hass: HomeAssistant, seeded: Any, freezer: Any, calls: Any,
    decompressions: collections.Counter[str], store_writes: list[tuple[str, int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The longest program up to its spin (30 readings, 29 min), not allowed to end."""
    entry, mgr = seeded
    c = calls(ProfileStore, "async_match_profile", "async_rebuild_envelope")
    calls(WashDataManager, "_notify_update")
    calls(progress, "_parse_phase_envelope", "power_data_to_offsets")
    monkeypatch.setattr(cycle_detector, "np", _NumpyCounter(cycle_detector.np, c))
    store_writes.clear()

    event_notifies = 0
    for _t, w in program_trace(PROFILES - 1, 1):
        before = c["_notify_update"]
        await report(hass, freezer, w)
        event_notifies += c["_notify_update"] - before
        await fire_timers(hass)
    assert mgr.detector.state == "running"
    assert mgr.current_program == f"Program {PROFILES - 1}"
    main = [b for k, b in store_writes if k == main_key(entry)]
    active = [b for k, b in store_writes if k == f"{main_key(entry)}.active"]

    within_budget(
        # Entity refreshes caused by power events (timers excluded). Measured 42
        # for 30 readings: one per reading (30), the detector's two state changes
        # (2), and on 5 match ticks the result, which lands a loop turn later, plus
        # a watchdog tick inside the same 60 s step (10). Budget 44. Guards item
        # 456 (c): _update_estimates and a match completing inside the reading
        # each refreshing before the reading's own refresh (49), and PERF-01's
        # fan-out (every notify rewrites ~20 entities).
        ("event notifies", event_notifies, 44),
        # One per match interval, half that until a program commits. Measured 8;
        # budget 10. Guards the detector's rate limit (unthrottled: every reading).
        ("matcher runs", c["async_match_profile"], 10),
        # The phase estimate parses the matched envelope once per envelope
        # revision. Measured 1; budget 1. Guards PERF-06 (a re-parse every 5 s).
        ("envelope parses", c["_parse_phase_envelope"], 1),
        # The phase estimate converts only its 60 s window of the running trace.
        # Measured 0; budget 0. Guards PROGRESS-17 (the whole trace converted on
        # every 5 s estimate, 744-1593 points, on the event loop).
        ("whole-trace conversions", c["power_data_to_offsets"], 0),
        # The detector's per-sample cadence statistics are pure Python. Measured
        # 0; budget 1. Guards PERF-07 (np.percentile / np.median on <= 20 values,
        # 40-185 us each, on every sample).
        ("detector NumPy calls", c["numpy"], 1),
        # The per-profile pause / terminal statistics a live match reads are cached
        # on an evidence fingerprint. Measured 6 (each computed once); budget 9.
        # Guards those caches (register item 297: uncached, the profile's traces
        # are decompressed on the event loop on every match tick).
        ("trace decompressions", decompressions["decompress"], 9),
        # Measured 0; budget 0. Guards an envelope rebuild on the live path.
        ("envelope rebuilds", c["async_rebuild_envelope"], 0),
        # The in-flight snapshot has its own file. Measured 0; budget 0. Guards
        # PERF-03 (every trace and envelope rewritten every 60 s: 6.7 MB a save,
        # 1.1 GB over one cycle on a real store, serialised on the event loop).
        ("main-store writes", len(main), 0),
        # A save every > 60 s, so every other reading here. Measured 15; budget
        # 17. Guards the 60 s throttle (a save per reading is 30).
        ("snapshot writes", len(active), 17),
        # Detector state plus the running trace. Measured 4146 bytes after 30
        # readings; budget 5632 (5.5 KB). Guards the snapshot growing back into a
        # store copy, or the trace being serialised twice.
        ("snapshot bytes", max(active, default=0), 5632),
    )


# --------------------------------------------------------------------------- PERF-03


async def test_cycle_end_work(
    hass: HomeAssistant, seeded: Any, freezer: Any, calls: Any,
    decompressions: collections.Counter[str], store_writes: list[tuple[str, int]],
) -> None:
    """The longest program, start to finished: five quiet minutes after its spin."""
    entry, mgr = seeded
    for _t, w in program_trace(PROFILES - 1, 1):
        await report(hass, freezer, w)
        await fire_timers(hass)
    c = calls(ProfileStore, "async_rebuild_envelope", "_rebuild_envelope_sync")
    decompressions.clear()
    store_writes.clear()
    for _ in range(5):
        await report(hass, freezer, 0.0)
        await fire_timers(hass)
    assert len(mgr.profile_store.get_past_cycles()) == PROFILES * CYCLES_PER_PROFILE + 1

    within_budget(
        # Every main-store write rewrites every trace and envelope (HA's Store is
        # whole-file). Measured 1: the cycle end's one immediate write, which here
        # also carries every follow-up's change; budget 2, room for the debounced
        # write a follow-up landing after it makes (suggestion scans, artifact
        # refresh). Guards item 456 (a): six writes before (suggestions x2,
        # lifetime energy, history pass, profile rebuild, maintenance; the audit
        # counted ~10 on a real store, 6.7 MB each).
        ("main-store writes", sum(k == main_key(entry) for k, _ in store_writes), 2),
        # Requests: the labelled profile's, and the learning pass confirming the
        # same label. Measured 2; budget 2. Guards a pass over every profile at
        # cycle end (the full maintenance made it P + 2).
        ("envelope rebuild requests", c["async_rebuild_envelope"], 2),
        # Builds: the second request finds nothing changed. Measured 1; budget 1.
        # Guards the rebuild memo (2 without it).
        ("envelope builds", c["_rebuild_envelope_sync"], 1),
        # The labelled profile's build (one per cycle of it), the new cycle's peak,
        # and that profile's artifacts. Measured 9 (35 before item 456: every
        # envelope rebuilt and every stored cycle's artifacts recomputed; 55
        # before `_cycle_peak` was memoised); budget 11. Guards a pass that
        # decompresses every stored trace again (the store size, not the profile's).
        ("trace decompressions", decompressions["decompress"], 11),
    )


# ----------------------------------------------------------------------------- WS


async def test_get_profiles_work(
    hass: HomeAssistant, seeded: Any, decompressions: collections.Counter[str]
) -> None:
    """``ha_washdata/get_profiles``: health, trends, gaps, advisories, signatures."""
    entry, mgr = seeded
    connection = MagicMock()
    ws_api.ws_get_profiles(
        hass, connection, {"id": 1, "type": "ha_washdata/get_profiles", "entry_id": entry.entry_id}
    )
    await hass.async_block_till_done(wait_background_tasks=True)
    assert connection.send_result.call_count == 1
    within_budget(
        # Measured one per stored cycle (12); budget cycles + 2. Guards a second
        # heuristic decompressing every trace again (2 x cycles).
        ("trace decompressions", decompressions["decompress"],
         len(mgr.profile_store.get_past_cycles()) + 2),
    )
