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
"""PR #448 review round 27: an import re-armed the banked-tail repair and then
nothing ran it.

Both import paths set ``BANKED_TAIL_REPAIR_KEY`` for a payload old enough to
carry banked tails, because an import bypasses ``_async_migrate_func``. Only
``manager.async_setup`` ever spawned the repair, and an import reloads the entry
only when the payload brings options with it - ``async_update_entry`` is not
called for a cycles-only import, nor for a selective one with
``apply_settings=False``. So the imported tails kept feeding ``avg_duration``,
the ETA and Smart Termination until the user next restarted Home Assistant.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from custom_components.ha_washdata.const import BANKED_TAIL_REPAIR_KEY


class _Mgr:
    """The scheduler in isolation: it touches only the store and _spawn_tracked."""

    def __init__(self, pending: bool):
        from custom_components.ha_washdata.manager import WashDataManager

        self.profile_store = MagicMock()
        self.profile_store.banked_tail_repair_pending.return_value = pending
        self._banked_tail_repair_task = None
        self.spawned: list = []
        self._schedule = WashDataManager.async_schedule_banked_tail_repair.__get__(self)
        self._repair_calls = 0

    def _spawn_tracked(self, coro):
        coro.close()  # never awaited here; keep the loop clean
        self.spawned.append(coro)
        task = MagicMock()
        task.done.return_value = False
        return task

    async def _async_repair_banked_tails(self):
        self._repair_calls += 1


def test_an_armed_marker_schedules_the_repair() -> None:
    mgr = _Mgr(pending=True)
    mgr._schedule()
    assert len(mgr.spawned) == 1


def test_a_clear_marker_schedules_nothing() -> None:
    """The marker is the whole test, so the common case costs one dict lookup."""
    mgr = _Mgr(pending=False)
    mgr._schedule()
    assert mgr.spawned == []


def test_a_second_import_does_not_start_a_second_walk() -> None:
    """The repair clears the marker only at the END of its run, so the pending
    check alone would let two imports in quick succession each start a full walk
    of the history and rebuild the same envelopes concurrently."""
    mgr = _Mgr(pending=True)
    mgr._schedule()
    mgr._schedule()
    assert len(mgr.spawned) == 1


def test_a_finished_run_does_not_block_a_later_import() -> None:
    """The in-flight guard must not become a one-shot latch: a later import that
    re-arms the marker has to be able to schedule again."""
    mgr = _Mgr(pending=True)
    mgr._schedule()
    mgr._banked_tail_repair_task.done.return_value = True
    mgr._schedule()
    assert len(mgr.spawned) == 2


@pytest.mark.parametrize(
    "handler, extra, store_call, result",
    [
        ("ws_import_config", {}, "async_import_data", {"entry_options": {}}),
        (
            "ws_import_config_selective",
            {"selection": {}, "mode": "merge", "conflict_resolutions": {},
             "cycle_destination": "reference", "apply_settings": False},
            "async_import_data_selective",
            {},
        ),
    ],
)
async def test_both_ws_import_handlers_schedule_the_repair(
    handler: str, extra: dict, store_call: str, result: dict
) -> None:
    """Driven, not read: a cycles-only import (no options, so no entry write and
    no reload) must still schedule the repair itself. Was a search of each
    handler's source for the call (audit TESTING-13)."""
    from unittest.mock import AsyncMock

    from custom_components.ha_washdata import ws_api

    manager = MagicMock()
    setattr(manager.profile_store, store_call, AsyncMock(return_value=result))
    entry = MagicMock()
    entry.options = {}
    entry.data = {}
    hass = MagicMock()
    hass.data = {}
    hass.async_add_executor_job = AsyncMock(side_effect=lambda fn, *a: fn(*a))
    connection = MagicMock()
    msg = {"id": 1, "entry_id": "e1", "json_data": "{}", **extra}

    with patch.object(ws_api, "_get_manager", return_value=manager), \
            patch.object(ws_api, "_get_entry", return_value=entry):
        await getattr(ws_api, handler).__wrapped__(hass, connection, msg)

    connection.send_error.assert_not_called()
    getattr(manager.profile_store, store_call).assert_awaited_once()
    hass.config_entries.async_update_entry.assert_not_called()
    manager.async_schedule_banked_tail_repair.assert_called_once_with()


async def test_the_import_config_service_schedules_the_repair(
    hass, tmp_path, enable_custom_integrations
) -> None:
    """The legacy `import_config` service reads a file straight into
    `async_import_data`, and writes the entry only when the payload carried
    settings. A cycles-only file must still schedule the repair.

    Driven through the real service on a real hass, where it used to be a search
    of `__init__.py`'s text (audit TESTING-13).
    """
    import json

    from homeassistant.config_entries import ConfigEntryState
    from homeassistant.helpers import device_registry as dr
    from homeassistant.setup import async_setup_component
    from pytest_homeassistant_custom_component.common import MockConfigEntry

    import custom_components.ha_washdata as washdata
    from custom_components.ha_washdata.const import DOMAIN

    assert await async_setup_component(hass, "http", {"http": {}})
    hass.states.async_set("sensor.p", "0")
    entry = MockConfigEntry(
        domain=DOMAIN, title="Washer",
        data={"name": "Washer", "power_sensor": "sensor.p", "device_type": "washing_machine"},
        options={}, version=3, minor_version=11,
    )
    entry.add_to_hass(hass)
    entry.mock_state(hass, ConfigEntryState.LOADED)
    assert await washdata.async_setup_entry(hass, entry)
    await hass.async_block_till_done()
    reg = dr.async_get(hass)
    dev = reg.async_get_device(identifiers={(DOMAIN, entry.entry_id)}) or reg.async_get_or_create(
        config_entry_id=entry.entry_id, identifiers={(DOMAIN, entry.entry_id)}, name="Washer"
    )
    manager = hass.data[DOMAIN][entry.entry_id]
    schedule = MagicMock(wraps=manager.async_schedule_banked_tail_repair)
    manager.async_schedule_banked_tail_repair = schedule

    path = tmp_path / "cycles_only.json"
    path.write_text(json.dumps({
        "version": 12,
        "data": {
            "profiles": {"Cotton": {"avg_duration": 3600.0}},
            "past_cycles": [{
                "id": "c1", "start_time": "2026-01-01T00:00:00+00:00", "duration": 3600.0,
                "status": "completed", "profile_name": "Cotton",
                "power_data": [[0, 100.0], [1800, 500.0], [3600, 0.0]],
            }],
        },
    }))
    hass.config.allowlist_external_dirs = {str(tmp_path)}
    options_before = dict(entry.options)

    await hass.services.async_call(
        DOMAIN, "import_config", {"device_id": dev.id, "path": str(path)}, blocking=True
    )
    await hass.async_block_till_done()

    assert dict(entry.options) == options_before, "a cycles-only import wrote options"
    schedule.assert_called_once_with()
    await hass.config_entries.async_unload(entry.entry_id)


def test_an_old_selective_import_arms_the_marker() -> None:
    """The half this test file exists to connect to: the store really does set
    the marker on an old payload, so the scheduling above is not hypothetical."""
    from custom_components.ha_washdata.profile_store import (
        _export_predates_banked_tail_repair,
    )

    assert _export_predates_banked_tail_repair({"version": 12}) is True
    # v13 re-arms as well since the #424 re-run (STORAGE_VERSION 14).
    assert _export_predates_banked_tail_repair({"version": 13}) is True
    assert _export_predates_banked_tail_repair({"version": 14}) is False


# --------------------------------------------------------------------------
# round 28: the second per-entry WS lock was never released on unload
# --------------------------------------------------------------------------
async def test_unload_releases_both_per_entry_ws_locks() -> None:
    """`_entry_options_lock` is a second per-entry lock built exactly like the
    write lock, and `async_unload_entry` popped only the write one - so every
    removed config entry left an asyncio.Lock in `hass.data` for the lifetime of
    the process.

    Driven through the real `async_unload_entry` (platform unload stubbed), where
    it used to be a search of `__init__.py`'s text (audit TESTING-13). A second
    entry stays loaded so the last-entry panel teardown is not reached.
    """
    from unittest.mock import AsyncMock

    from custom_components.ha_washdata import async_unload_entry, ws_api
    from custom_components.ha_washdata.const import DOMAIN

    manager = MagicMock()
    manager.async_shutdown = AsyncMock()
    hass = MagicMock()
    hass.data = {DOMAIN: {"e1": manager, "e2": MagicMock()}}
    hass.config_entries.async_unload_platforms = AsyncMock(return_value=True)
    entry = MagicMock()
    entry.entry_id = "e1"

    for get_lock in (ws_api._entry_write_lock, ws_api._entry_options_lock):
        get_lock(hass, "e1")
        get_lock(hass, "e2")

    assert await async_unload_entry(hass, entry) is True

    manager.async_shutdown.assert_awaited_once()
    for key in (ws_api._WS_WRITE_LOCKS_KEY, ws_api._WS_OPTIONS_LOCKS_KEY):
        assert set(hass.data[key]) == {"e2"}, key


def test_the_two_lock_keys_are_distinct() -> None:
    """Popping the same key twice would look like a fix and change nothing."""
    from custom_components.ha_washdata.ws_api import (
        _WS_OPTIONS_LOCKS_KEY,
        _WS_WRITE_LOCKS_KEY,
    )

    assert _WS_OPTIONS_LOCKS_KEY != _WS_WRITE_LOCKS_KEY


# --------------------------------------------------------------------------
# round 28: three docs describing pre-item-355/356/353 behaviour
# --------------------------------------------------------------------------
def test_the_end_gate_comment_does_not_claim_a_fixed_ratio() -> None:
    """Item 355 made the bar device-resolved (0.90 for washers), so "never fires
    before 1.05x expected" became false for exactly the device type the change
    was made for."""
    import inspect

    from custom_components.ha_washdata import cycle_detector

    src = inspect.getsource(cycle_detector)
    assert "never fires before 1.05x expected" not in src
    assert "resolve_end_gate_late_ratio" in src


def test_the_tuner_docstring_describes_the_implemented_rule() -> None:
    """Item 356 implemented the three-way template rule; the docstring still
    called it deferred, which would tell a maintainer item 347 is open."""
    import inspect

    from custom_components.ha_washdata.ml import matching_tuner

    doc = inspect.getdoc(matching_tuner._snaps) or ""
    assert "deferred" not in doc
    assert "duration is closest to the profile" not in doc
    assert "three-way" in doc


def test_the_repair_docstring_admits_it_rewrites_reference_cycles() -> None:
    """Item 353 put `reference_cycles` in scope, golden ones included. The first
    paragraph still said only `past_cycles` is touched, which is the paragraph a
    maintainer reads first."""
    import inspect

    from custom_components.ha_washdata.profile_store import ProfileStore

    doc = inspect.getdoc(ProfileStore.async_repair_banked_tails) or ""
    assert "Only ``past_cycles`` is touched" not in doc
    assert "reference_cycles" in doc
    assert "golden" in doc


# --------------------------------------------------------------------------
# round 29: one zero ratio wiped every advisory for every profile
# --------------------------------------------------------------------------
def _advisory_store(avg_duration: float):
    """A store with one profile whose duration gate nothing can satisfy, plus a
    second profile that is genuinely unmatchable so there is an advisory to lose."""
    from unittest.mock import MagicMock as _MM

    from custom_components.ha_washdata.profile_store import ProfileStore

    st = ProfileStore.__new__(ProfileStore)
    st._logger = _MM()
    st._min_duration_ratio = 0.10
    st._max_duration_ratio = 1.8
    cycles = [
        {
            "id": f"c{i}",
            "profile_name": "Long",
            "status": "completed",
            "duration": 120.0,
            "power_data": [[0.0, 100.0], [60.0, 100.0], [120.0, 0.0]],
        }
        for i in range(6)
    ]
    st.iter_evidence_cycles = lambda: iter(cycles)
    st.get_profiles = lambda: {"Long": {"avg_duration": avg_duration}}
    st._stage1_duration_for = lambda name, prof: avg_duration
    st.compute_profile_health = lambda: {}
    st.compute_profile_trends = lambda: {}
    st.unmatchable_profiles = lambda: {"Ghost": "no cycle with power data"}
    return st


def test_a_zero_ratio_does_not_wipe_every_advisory() -> None:
    """`ratio` is `round(dur / avg, 3)`, so a hand-set duration over ~33 h rounds
    it to exactly 0.0 against the 60 s floor. `math.log(0.0)` raised ValueError
    into `compute_profile_advisories`'s broad `except`, which returns [] - so one
    outlier cycle removed every advisory for every profile, the `unmatchable`
    warnings included."""
    st = _advisory_store(avg_duration=500_000.0)

    out = st.compute_profile_advisories()

    codes = {a["code"] for a in out}
    assert "unmatchable" in codes, "the unrelated warning must survive"
    assert "duration_outlier" in codes


def test_the_zero_ratio_case_is_actually_reachable() -> None:
    """Guards the test above against becoming vacuous: if the rounding ever stops
    producing 0.0, the regression it pins is no longer being exercised."""
    st = _advisory_store(avg_duration=500_000.0)

    offenders = st._self_unmatchable_cycles()["Long"]

    assert any(o["ratio"] == 0.0 for o in offenders)


def test_a_normal_outlier_still_ranks_by_distance_from_one() -> None:
    """The clamp must not disturb ordinary ranking.

    The expected string is "0.09", not "0.1": round 35 widened the display to two
    decimals because `.1f` collapsed the informative end of the range (0.04x read
    as "0.0x", i.e. "this cycle had no length"). Under the old format this
    assertion could not tell 0.09 from 0.12, which is the distinction it exists
    to make."""
    st = _advisory_store(avg_duration=1000.0)
    st.iter_evidence_cycles = lambda: iter(
        [
            {"id": "near", "profile_name": "Long", "status": "completed", "duration": 120.0},
            {"id": "far", "profile_name": "Long", "status": "completed", "duration": 90.0},
        ]
        + [
            {"id": f"ok{i}", "profile_name": "Long", "status": "completed", "duration": 1000.0}
            for i in range(4)
        ]
    )

    out = [a for a in st.compute_profile_advisories() if a["code"] == "duration_outlier"]

    assert out and out[0]["message_params"]["ratio"] == "0.09"
    assert "0.09x" in out[0]["message"], "fallback and param must agree"


# --------------------------------------------------------------------------
# round 30: a mid-cycle settings save re-ran the once-per-cycle handover
# --------------------------------------------------------------------------
def _live_mgr():
    """A manager stub carrying only the live-notification flags this touches."""
    from unittest.mock import MagicMock as _MM

    from custom_components.ha_washdata.manager import WashDataManager

    m = WashDataManager.__new__(WashDataManager)
    m._live_notification_sent_count = 7
    m._live_notification_cap = 12
    m._last_live_notification_time = object()
    m._live_waiting_notification_sent = True
    m._live_chronometer_overrun_sent = True
    m._live_activity_started = True
    m._lifecycle_tag = "tag_lifecycle"
    m._notify_live_sticky = False
    m._notify_live_silent = True
    m._send_tag_clear = _MM()
    return m


def test_a_settings_save_does_not_re_run_the_lifecycle_handover() -> None:
    """`async_reload_config` resets live state mid-cycle when the user saves any
    option. Clearing `_live_activity_started` there made the next live tick look
    like the first of a new cycle, so `_record_live_activity_started` re-ran the
    #446 handover and cleared `_lifecycle_tag` - the tag the pre-completion
    reminder rides at priority high."""
    m = _live_mgr()

    m._reset_live_notification_state(keep_activity_started=True)
    m._record_live_activity_started()

    assert m._live_activity_started is True
    m._send_tag_clear.assert_not_called()
    # The counters it exists to reset are still reset.
    assert m._live_notification_sent_count == 0
    assert m._live_waiting_notification_sent is False


def test_a_cycle_boundary_still_resets_the_flag() -> None:
    """Cycle start/end must keep resetting it, or the handover never runs again."""
    m = _live_mgr()

    m._reset_live_notification_state()

    assert m._live_activity_started is False
    m._record_live_activity_started()
    m._send_tag_clear.assert_called_once_with("tag_lifecycle")


def test_preserving_cannot_fake_a_started_activity() -> None:
    """A reload that ENABLES live notifications mid-cycle must still hand over on
    the first real tick: the flag is only ever kept at the value it already had."""
    m = _live_mgr()
    m._live_activity_started = False

    m._reset_live_notification_state(keep_activity_started=True)

    assert m._live_activity_started is False


def test_a_silent_live_update_stays_silent_across_a_reload() -> None:
    """`_apply_live_notification_prefs` gates `silent`/`push` on the same flag, so
    losing it made the next tick alert audibly with notify_live_silent on (#417)."""
    m = _live_mgr()

    m._reset_live_notification_state(keep_activity_started=True)
    extra: dict = {}
    m._apply_live_notification_prefs(extra)

    assert extra.get("silent") is True
    assert extra.get("push") == {"interruption-level": "passive"}


async def test_the_reload_path_actually_passes_the_flag(hass, freezer) -> None:
    """The tests above pin the MECHANISM and all pass with the call site reverted,
    which would pin nothing. This drives the call site: a REAL manager mid-cycle
    with a live activity running, then a settings save through
    `async_reload_config`. The flag must survive and the next live tick must not
    re-run the handover (a `clear_notification` on the lifecycle tag).

    Was a source-text check that the keyword appeared once in manager.py (audit
    TESTING-13). The complement - every cycle boundary still resets it - is
    driven through the real cycle end in test_issue_446_cycle_token_gate.py.
    """
    from .real_manager import boot, feed, make_entry, record_notify

    calls = record_notify(hass)
    entry = make_entry(hass, {"notify_live_services": ["notify.mobile_app_phone"]})
    mgr = await boot(hass, entry)
    # A profile, so live cards are sent at all.
    await feed(hass, freezer, 500, 900)
    await feed(hass, freezer, 0, 300)
    await hass.async_block_till_done()
    await mgr.profile_store.create_profile("Cotton", mgr.profile_store.get_past_cycles()[0]["id"])

    await feed(hass, freezer, 500, 300)
    assert mgr._live_activity_started is True
    lifecycle = mgr._lifecycle_tag

    def lifecycle_clears() -> int:
        return sum(
            1 for c in calls
            if c.get("message") == "clear_notification"
            and (c.get("data") or {}).get("tag") == lifecycle
        )

    handovers = lifecycle_clears()
    assert handovers == 1, "the first live card hands the lifecycle card over once"

    hass.config_entries.async_update_entry(
        entry, options={**entry.options, "notify_live_interval_seconds": 60}
    )
    await mgr.async_reload_config(entry)
    assert mgr._live_activity_started is True, "a settings save is not a cycle boundary"

    await feed(hass, freezer, 500, 600)
    assert mgr.detector.state == "running"
    assert lifecycle_clears() == handovers, "the reload re-ran the lifecycle handover"
    await mgr.async_shutdown()


# --------------------------------------------------------------------------
# round 30: a phase stored under an empty scope is invisible AND unrepeatable
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_an_unresolved_device_type_is_refused_not_stored() -> None:
    """`manager.device_type` reads `options.get(CONF_DEVICE_TYPE, ...)`, and `.get`
    returns a persisted "" or None verbatim rather than the default - the #389
    class `strip_null_options` exists for. Storing under an empty scope is the
    worst outcome available: `list_phase_catalog` never lists it, so the phase is
    invisible, yet it still trips the duplicate check on the next attempt."""
    from unittest.mock import AsyncMock, MagicMock

    from custom_components.ha_washdata import ws_api

    manager = MagicMock()
    manager.device_type = ""
    manager.profile_store.async_create_custom_phase = AsyncMock()
    hass = MagicMock()
    connection = MagicMock()
    msg = {"id": 1, "entry_id": "e1", "device_type": "", "name": "Soak", "description": ""}

    with patch.object(ws_api, "_get_manager", return_value=manager):
        # Wrapped by @async_response; __wrapped__ is the coroutine itself.
        await ws_api.ws_create_phase.__wrapped__(hass, connection, msg)

    manager.profile_store.async_create_custom_phase.assert_not_called()
    assert connection.send_error.call_args[0][1] == "invalid_device_type"


@pytest.mark.asyncio
async def test_a_resolved_device_type_still_creates_the_phase() -> None:
    """The guard must not block the ordinary path."""
    from unittest.mock import AsyncMock, MagicMock

    from custom_components.ha_washdata import ws_api

    manager = MagicMock()
    manager.device_type = "dishwasher"
    manager.profile_store.async_create_custom_phase = AsyncMock()
    hass = MagicMock()
    connection = MagicMock()
    msg = {"id": 1, "entry_id": "e1", "device_type": "", "name": "Soak", "description": ""}

    with patch.object(ws_api, "_get_manager", return_value=manager):
        # Wrapped by @async_response; __wrapped__ is the coroutine itself.
        await ws_api.ws_create_phase.__wrapped__(hass, connection, msg)

    manager.profile_store.async_create_custom_phase.assert_awaited_once()
    assert manager.profile_store.async_create_custom_phase.await_args[0][0] == "dishwasher"
    connection.send_error.assert_not_called()


# --------------------------------------------------------------------------
# round 31: the presence flush dismissed the reminder it had just delivered
# --------------------------------------------------------------------------
def _flush_mgr(pending):
    """A manager stub carrying only what `_flush_pending_notifications` touches,
    recording the ORDER of deliveries and tag clears."""
    from custom_components.ha_washdata.manager import WashDataManager

    m = WashDataManager.__new__(WashDataManager)
    m._pending_notifications = list(pending)
    m._live_activity_started = False
    m._live_waiting_notification_sent = False
    m._live_notification_sent_count = 0
    m._last_live_notification_time = None
    m._lifecycle_tag = "tag_lifecycle"
    m.log: list = []

    def _dispatch(message, **kw):
        m.log.append(("deliver", kw.get("event_type")))
        return True

    def _clear(tag):
        m.log.append(("clear", tag))

    m._dispatch_notification = _dispatch
    m._send_tag_clear = _clear
    return m


def test_a_queued_reminder_is_not_dismissed_by_the_live_handover() -> None:
    """A deferred LIVE entry replaces earlier live ones and is appended LAST, so a
    pre_complete reminder queued earlier sits ahead of it. Flushed in order, the
    reminder was delivered and then cleared a moment later by the #446 handover -
    and it rides `_lifecycle_tag` at priority high, i.e. the card worth having."""
    m = _flush_mgr([
        {"message": "nearly done", "event_type": "pre_complete"},
        {"message": "live", "event_type": "cycle_live", "extra_vars": {"progress": 80}},
    ])

    m._flush_pending_notifications(None, None)

    clear_at = next(i for i, e in enumerate(m.log) if e[0] == "clear")
    deliver_at = next(i for i, e in enumerate(m.log) if e == ("deliver", "pre_complete"))
    assert clear_at < deliver_at, (
        "the lifecycle clear must happen before the reminder is delivered, "
        f"got {m.log}"
    )
    assert m._live_activity_started is True


def test_a_queued_start_card_is_delivered_before_the_clear_not_dropped() -> None:
    """Round 31 DROPPED queued START entries, which was a regression round 32
    caught: "superseded by the live activity" is true only of a mobile target that
    also gets the live card. `_send_tag_clear` only ever addresses
    `_notify_live_services` (and returns early when it is empty), while START has
    its own `_notify_start_services` and may be a telegram or e-mail target the
    clear can never reach - and a notification ACTION fires on delivery, so an
    automation branching on `event_type == "start"` needs the dispatch to happen.

    Delivering START ahead of the clear gets every case right at once: a mobile
    start card is swept up by the handover exactly as before, and every other
    target keeps its notification."""
    m = _flush_mgr([
        {"message": "started", "event_type": "cycle_start"},
        {"message": "live", "event_type": "cycle_live", "extra_vars": {"progress": 10}},
    ])

    m._flush_pending_notifications(None, None)

    start_at = m.log.index(("deliver", "cycle_start"))
    clear_at = next(i for i, e in enumerate(m.log) if e[0] == "clear")
    assert start_at < clear_at, f"start must be delivered before the clear, got {m.log}"
    assert ("deliver", "cycle_live") in m.log


def test_the_reminder_still_survives_alongside_a_start_card() -> None:
    """Both fixes at once: START ahead of the clear, reminder after it."""
    m = _flush_mgr([
        {"message": "started", "event_type": "cycle_start"},
        {"message": "nearly done", "event_type": "pre_complete"},
        {"message": "live", "event_type": "cycle_live", "extra_vars": {"progress": 90}},
    ])

    m._flush_pending_notifications(None, None)

    clear_at = next(i for i, e in enumerate(m.log) if e[0] == "clear")
    assert m.log.index(("deliver", "cycle_start")) < clear_at
    assert m.log.index(("deliver", "pre_complete")) > clear_at


def test_no_live_entry_means_no_handover_and_nothing_dropped() -> None:
    """Without a live entry there is no activity to hand over to, so a queued
    start card and reminder must both still arrive."""
    m = _flush_mgr([
        {"message": "started", "event_type": "cycle_start"},
        {"message": "nearly done", "event_type": "pre_complete"},
    ])

    m._flush_pending_notifications(None, None)

    assert [e for e in m.log if e[0] == "clear"] == []
    assert ("deliver", "cycle_start") in m.log
    assert ("deliver", "pre_complete") in m.log
    assert m._live_activity_started is False


def test_an_already_started_activity_does_not_hand_over_again() -> None:
    m = _flush_mgr([
        {"message": "live", "event_type": "cycle_live", "extra_vars": {"progress": 50}},
    ])
    m._live_activity_started = True

    m._flush_pending_notifications(None, None)

    assert [e for e in m.log if e[0] == "clear"] == []


def test_the_shutdown_docstring_no_longer_claims_both_tags_are_cleared() -> None:
    """Round 25 gated the lifecycle clear on `_CYCLE_IN_PROGRESS_STATES` because it
    carries the FINISHED alert. The docstring still said both tags go outright,
    which is the sentence a maintainer would trust when undoing the gate."""
    import inspect

    from custom_components.ha_washdata.manager import WashDataManager

    doc = inspect.getdoc(WashDataManager._clear_live_progress_notification) or ""
    assert "so both tags are cleared outright" not in doc
    assert "_CYCLE_IN_PROGRESS_STATES" in doc


# --------------------------------------------------------------------------
# round 35: hoisting START reordered the FINISH entry queued before it
# --------------------------------------------------------------------------
def test_a_previous_cycles_finish_card_is_not_left_pinned_by_the_next_start() -> None:
    """Round 32 pulled START entries to the FRONT, which silently reordered every
    lifecycle-tagged entry queued before one. FINISH survives the
    `_clear_live_progress_notification` purge (it drops LIVE, START and
    `pre_complete`, not FINISH), so `[FINISH(A), START(B), LIVE(B)]` is reachable:
    nobody home, cycle A ends, cycle B starts.

    Hoisting sent START(B) first, cleared the tag, then delivered FINISH(A) AFTER
    it - leaving "cycle A finished" pinned to the lifecycle tag for the whole of
    cycle B, with B's own start card already cleared. Delivering in order fixes it
    for free: FINISH(A) lands, START(B) replaces it on the same tag, and the
    handover clears the single card that remains."""
    m = _flush_mgr([
        {"message": "A finished", "event_type": "cycle_finish"},
        {"message": "B started", "event_type": "cycle_start"},
        {"message": "live", "event_type": "cycle_live", "extra_vars": {"progress": 5}},
    ])

    m._flush_pending_notifications(None, None)

    clear_at = next(i for i, e in enumerate(m.log) if e[0] == "clear")
    finish_at = m.log.index(("deliver", "cycle_finish"))
    start_at = m.log.index(("deliver", "cycle_start"))
    assert finish_at < start_at < clear_at, (
        "queue order must be preserved and both must precede the clear, "
        f"got {m.log}"
    )


def test_a_reminder_after_the_start_still_survives_the_clear() -> None:
    """The split must not undo round 31: only entries up to the last START go
    before the handover."""
    m = _flush_mgr([
        {"message": "A finished", "event_type": "cycle_finish"},
        {"message": "B started", "event_type": "cycle_start"},
        {"message": "nearly done", "event_type": "pre_complete"},
        {"message": "live", "event_type": "cycle_live", "extra_vars": {"progress": 90}},
    ])

    m._flush_pending_notifications(None, None)

    clear_at = next(i for i, e in enumerate(m.log) if e[0] == "clear")
    assert m.log.index(("deliver", "cycle_finish")) < clear_at
    assert m.log.index(("deliver", "cycle_start")) < clear_at
    assert m.log.index(("deliver", "pre_complete")) > clear_at


def test_a_live_entry_before_the_last_start_still_waits_for_the_clear() -> None:
    """The dedup only re-appends within one queue, so a LIVE can precede a later
    cycle's START. It still belongs after the handover."""
    m = _flush_mgr([
        {"message": "live", "event_type": "cycle_live", "extra_vars": {"progress": 40}},
        {"message": "B started", "event_type": "cycle_start"},
    ])

    m._flush_pending_notifications(None, None)

    clear_at = next(i for i, e in enumerate(m.log) if e[0] == "clear")
    assert m.log.index(("deliver", "cycle_start")) < clear_at
    assert m.log.index(("deliver", "cycle_live")) > clear_at
