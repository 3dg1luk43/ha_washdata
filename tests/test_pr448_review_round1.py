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
"""Regressions for the defects found in the PR #448 review.

Each test here pins one thing that shipped broken on the 0.5.7 branch and was
caught in review rather than by the suite. They are grouped in one module
because they share nothing but their provenance.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import DOMAIN
from custom_components.ha_washdata.ml.matching_tuner import _series
from custom_components.ha_washdata.suggestion_engine import detect_standby_above_stop


# --------------------------------------------------------------------------
# ws_api: the standby advisory was computed into a NameError and swallowed
# --------------------------------------------------------------------------
def _call_get_devices(past_cycles, options):
    """Invoke ws_get_devices for one entry and return its device info dict."""
    entry = SimpleNamespace(
        entry_id="e1", title="Washer", data={}, options=dict(options)
    )
    store = SimpleNamespace(
        get_suggestions=lambda: {},
        get_locked_suggestions=lambda: [],
        get_pending_feedback=lambda: {},
        get_past_cycles=lambda: list(past_cycles),
    )
    manager = SimpleNamespace(
        detector=SimpleNamespace(state="off", sub_state=None),
        _current_program=None,
        manual_program_active=False,
        _time_remaining=None,
        _total_duration=None,
        _current_power=None,
        _cycle_progress=None,
        profile_store=store,
        is_user_paused=False,
        recorder=SimpleNamespace(is_recording=False),
    )
    hass = MagicMock()
    hass.config_entries.async_entries.return_value = [entry]
    hass.data = {DOMAIN: {"e1": manager}}
    connection = MagicMock()
    with patch.object(ws_api, "_effective_level", return_value="admin"):
        ws_api.ws_get_devices(hass, connection, {"id": 1, "type": "x"})
    return connection.send_result.call_args[0][1]["devices"][0]


def _idle_cycle(final_w: float, cid: str = "c") -> dict:
    return {
        "id": cid,
        "duration": 3600.0,
        "power_data": [[0.0, 2000.0], [10.0, 2000.0], [20.0, final_w]],
    }


def test_ws_get_devices_actually_reaches_the_standby_advisory():
    """The import lived in another function, so every call raised NameError.

    The surrounding ``except Exception`` swallowed it, which is indistinguishable
    from "no pattern found" - the whole #445 cause-1 card shipped dead.
    """
    dev = _call_get_devices(
        [_idle_cycle(3.4, "a"), _idle_cycle(3.2, "b"), _idle_cycle(3.5, "c")],
        {"stop_threshold_w": 2.56},
    )
    assert dev["standby_above_stop"] is not None
    assert dev["standby_above_stop"]["cycles_above"] == 3
    assert dev["standby_above_stop"]["idle_w"] == pytest.approx(3.4)


def test_ws_get_devices_reports_no_advisory_when_there_is_no_pattern():
    """The key is still set (to None) so a NameError cannot masquerade as silence."""
    dev = _call_get_devices([_idle_cycle(0.0, "a")], {"stop_threshold_w": 2.56})
    assert dev["standby_above_stop"] is None


# --------------------------------------------------------------------------
# suggestion_engine: a cycle stopped by hand mid-wash is not standby evidence
# --------------------------------------------------------------------------
def test_a_cycle_stopped_at_working_power_is_not_standby_evidence():
    """Two force-stops mid-wash used to be reported as "idling at 500 W"."""
    mid_wash = {
        "id": "u",
        "duration": 1200.0,
        "termination_reason": "user",
        "power_data": [[0.0, 2000.0], [10.0, 1800.0], [20.0, 500.0]],
    }
    assert detect_standby_above_stop([mid_wash, dict(mid_wash, id="u2")], 2.56) is None


def test_a_cycle_the_user_stopped_at_standby_still_counts():
    """#445's reporter force-stopped four cycles; those ARE the evidence."""
    res = detect_standby_above_stop(
        [_idle_cycle(3.4, "a"), _idle_cycle(3.3, "b")], 2.56
    )
    assert res is not None and res["cycles_above"] == 2


# --------------------------------------------------------------------------
# matching_tuner: one half-parsed row desynchronised the two arrays
# --------------------------------------------------------------------------
def test_a_row_whose_power_is_unparseable_drops_the_whole_pair():
    """``ts`` used to gain an element ``pw`` did not, so np.interp raised later."""
    cycle = {
        "power_data": [
            [0.0, 10.0],
            [10.0, "nonsense"],
            [20.0, 30.0],
            [30.0, 40.0],
            [40.0, 50.0],
        ]
    }
    ts, pw = _series(cycle)
    assert ts.size == pw.size == 4
    assert list(ts) == [0.0, 20.0, 30.0, 40.0]


def test_an_unbounded_integer_power_drops_the_pair_rather_than_raising():
    """`json` keeps an oversized literal as an int, and float() on one raises."""
    cycle = {
        "power_data": [
            [0.0, 10.0],
            [10.0, 10**400],
            [20.0, 30.0],
            [30.0, 40.0],
            [40.0, 50.0],
        ]
    }
    ts, pw = _series(cycle)
    assert ts.size == pw.size == 4


def test_the_arrays_stay_usable_for_interpolation():
    """The failure mode was downstream: np.interp on unequal arrays."""
    cycle = {"power_data": [[0.0, 1.0], [1.0, None], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]}
    ts, pw = _series(cycle)
    assert np.interp(np.linspace(0.0, 4.0, 10), ts, pw).size == 10


# --------------------------------------------------------------------------
# manager: the banked-tail repair was spawned untracked
# --------------------------------------------------------------------------
def test_the_banked_tail_repair_is_spawned_through_spawn_tracked():
    """It writes to the ProfileStore, so a reload must be able to cancel it.

    ``async_shutdown`` only cancels what is in ``_background_tasks``, and
    ``_spawn_tracked`` is what puts it there. Spawned bare, the repair would walk
    up to 200 stored traces still writing to the store setup had just replaced.
    Asserted on the source because reaching this line needs a fully built manager.
    """
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "manager.py"
    ).read_text()
    assert "self._spawn_tracked(self._async_repair_banked_tails())" in src
    assert "async_create_task(self._async_repair_banked_tails" not in src


# --------------------------------------------------------------------------
# panel: the standby card was pushed after the HTML string was built
# --------------------------------------------------------------------------
def test_the_standby_card_is_pushed_before_the_attention_html_is_built():
    """``attnHtml`` is a plain string, so a later push never reaches the render.

    The card was computed, pushed and then dropped on the floor - dead code that
    no unit test could see, because the array it mutates is correct either way.
    """
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "www"
        / "ha-washdata-panel.js"
    ).read_text()
    push_at = src.index("const sas = dev.standby_above_stop;")
    build_at = src.index("const attnHtml = attn.length")
    assert push_at < build_at, (
        "the standby_above_stop card is pushed onto `attn` after `attnHtml` has "
        "already been joined, so it never renders"
    )
