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
    "module, handler",
    [
        ("ws_api", "ws_import_config"),
        ("ws_api", "ws_import_config_selective"),
    ],
)
def test_both_ws_import_handlers_schedule_the_repair(module: str, handler: str) -> None:
    """Asserted on the source: driving these needs a live connection, a manager
    and the per-entry write lock. What matters is that neither handler relies on
    the options write below it to reload the entry."""
    import importlib
    import inspect

    mod = importlib.import_module(f"custom_components.ha_washdata.{module}")
    src = inspect.getsource(getattr(mod, handler))
    assert "async_schedule_banked_tail_repair()" in src


def test_the_import_config_service_schedules_the_repair() -> None:
    """The legacy `import_config` service reads a file straight into
    `async_import_data`, and writes the entry only when the payload carried
    settings."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "__init__.py"
    ).read_text()
    assert "manager.async_schedule_banked_tail_repair()" in src


def test_an_old_selective_import_arms_the_marker() -> None:
    """The half this test file exists to connect to: the store really does set
    the marker on an old payload, so the scheduling above is not hypothetical."""
    from custom_components.ha_washdata.profile_store import (
        _export_predates_banked_tail_repair,
    )

    assert _export_predates_banked_tail_repair({"version": 12}) is True
    assert _export_predates_banked_tail_repair({"version": 13}) is False


# --------------------------------------------------------------------------
# round 28: the second per-entry WS lock was never released on unload
# --------------------------------------------------------------------------
def test_unload_releases_both_per_entry_ws_locks() -> None:
    """`_entry_options_lock` is a second per-entry lock built exactly like the
    write lock, and `async_unload_entry` popped only the write one - so every
    removed config entry left an asyncio.Lock in `hass.data` for the lifetime of
    the process. Asserted on source: the pop sits inside a long unload coroutine
    that needs a fully built entry to reach."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "__init__.py"
    ).read_text()
    assert "hass.data.get(_WS_WRITE_LOCKS_KEY, {}).pop(entry.entry_id, None)" in src
    assert "hass.data.get(_WS_OPTIONS_LOCKS_KEY, {}).pop(entry.entry_id, None)" in src


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
    """The clamp must not disturb ordinary ranking."""
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

    assert out and out[0]["message_params"]["ratio"] == "0.1"


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


def test_the_reload_path_actually_passes_the_flag() -> None:
    """The tests above pin the MECHANISM and all pass with the call site reverted,
    which would pin nothing. `async_reload_config` needs a fully built manager,
    a config entry and a live detector to reach, so the call site is asserted on
    source - the same pattern the other hard-to-reach call sites here use.

    The complement matters too: every OTHER reset is a cycle boundary and must
    keep resetting the flag, so exactly one call site may carry the keyword.
    """
    import inspect

    from custom_components.ha_washdata.manager import WashDataManager

    reload_src = inspect.getsource(WashDataManager.async_reload_config)
    assert "_reset_live_notification_state(keep_activity_started=True)" in reload_src

    whole = inspect.getsource(inspect.getmodule(WashDataManager))
    assert whole.count("_reset_live_notification_state(keep_activity_started=True)") == 1


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
