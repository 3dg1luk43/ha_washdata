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
"""#511: an envelope rebuild that fails during the sample repair is logged, not swallowed."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.profile_store import ProfileStore

START = datetime(2026, 5, 1, 8, 0, tzinfo=timezone.utc)


async def test_a_failed_rebuild_after_a_repair_is_logged(
    hass: HomeAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    ps = ProfileStore(hass, "e511")
    await ps.async_load()
    ps._data["profiles"] = {"Eco": {"avg_duration": 3600, "sample_cycle_id": "gone"}}  # noqa: SLF001
    ps._data["past_cycles"] = [{  # noqa: SLF001
        "id": "c1", "profile_name": "Eco", "status": "completed", "duration": 3600,
        "start_time": START.isoformat(),
        "end_time": (START + timedelta(seconds=3600)).isoformat(),
        "power_data": [[float(t), 500.0] for t in range(0, 3601, 60)],
    }]

    with patch.object(
        ps, "async_rebuild_envelope", AsyncMock(side_effect=RuntimeError("boom"))
    ), caplog.at_level(logging.WARNING):
        stats = await ps.async_repair_profile_samples()

    assert stats["profiles_repaired"] == 1
    assert ps.get_profiles()["Eco"]["sample_cycle_id"] == "c1"
    assert "Envelope rebuild after repairing profile 'Eco' failed" in caplog.text
