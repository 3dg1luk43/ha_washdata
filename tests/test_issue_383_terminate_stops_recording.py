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
"""Issues #376 / #383: Force terminate could not clear a forgotten manual recording.

While the recorder runs, ``check_state`` reports running and the detector is fed
nothing, so ``detector.user_stop()`` was a no-op: the device stayed "running" at
0 W, and the recording is persisted, so a reload came back running too. Force
terminate now stops the recording as the Stop Recording button does, keeping the
run for processing.
"""

from __future__ import annotations

from custom_components.ha_washdata.button import WashDataTerminateButton
from custom_components.ha_washdata.const import STATE_RUNNING

from .real_manager import boot, feed, make_entry, record_notify


async def test_force_terminate_stops_a_forgotten_recording(hass, freezer) -> None:
    record_notify(hass)
    entry = make_entry(hass)
    mgr = await boot(hass, entry)
    await mgr.async_start_recording()
    await feed(hass, freezer, 400, 600)
    await feed(hass, freezer, 0, 1800)
    assert mgr.recorder.is_recording
    assert mgr.check_state() == STATE_RUNNING  # pinned at 0 W: the reported bug

    await WashDataTerminateButton(mgr, entry).async_press()
    await hass.async_block_till_done()

    assert not mgr.recorder.is_recording
    assert mgr.check_state() != STATE_RUNNING
    # The run is kept for processing, exactly as Stop Recording keeps it.
    last = mgr.recorder.last_run
    assert last is not None and len(last["data"]) >= 80
    await mgr.async_shutdown()


async def test_force_terminate_without_a_recording_still_ends_the_cycle(hass, freezer) -> None:
    record_notify(hass)
    entry = make_entry(hass)
    mgr = await boot(hass, entry)
    await feed(hass, freezer, 500, 900)
    assert mgr.detector.state == STATE_RUNNING
    await mgr.async_terminate_cycle()
    await hass.async_block_till_done()
    assert mgr.detector.state != STATE_RUNNING
    assert mgr.recorder.last_run is None
    await mgr.async_shutdown()
