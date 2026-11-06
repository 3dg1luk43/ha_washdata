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
"""#511: an export from a newer WashData is refused, like the undo of one already is.

No migration runs backwards, so a newer store's data was taken as-is and then
saved under this build's version, where no later load would ever migrate it.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata.const import STORAGE_VERSION
from custom_components.ha_washdata.profile_store import (
    ProfileStore,
    build_import_manifest,
    unwrap_import_payload,
)


def _export(version: Any) -> dict[str, Any]:
    return {
        "version": version,
        "data": {
            "profiles": {"Eco": {"avg_duration": 3600}},
            "past_cycles": [{
                "id": "x1", "profile_name": "Eco", "status": "completed", "duration": 3600,
                "start_time": "2026-03-01T08:00:00+00:00",
                "power_data": [[float(t), 400.0] for t in range(0, 3601, 60)],
            }],
        },
    }


async def test_a_newer_export_is_refused_and_changes_nothing(hass: HomeAssistant) -> None:
    ps = ProfileStore(hass, "e511")
    await ps.async_load()
    before = copy.deepcopy(ps._data)  # noqa: SLF001
    newer = _export(STORAGE_VERSION + 1)

    with pytest.raises(ValueError, match="newer version"):
        await ps.async_import_data(newer)
    with pytest.raises(ValueError, match="newer version"):
        await ps.async_import_data_selective(
            newer, selection={"categories": ["profiles"]}, mode="replace"
        )
    assert ps._data == before  # noqa: SLF001
    assert await ps.async_get_pre_import_snapshot() is None
    assert "newer version" in build_import_manifest(
        newer, local_device_type="washing_machine", local_profile_names=[]
    )["error"]


@pytest.mark.parametrize("version", [STORAGE_VERSION, 1, "abc", None])
def test_current_older_and_unreadable_versions_still_unwrap(version: Any) -> None:
    data, _meta = unwrap_import_payload(_export(version))
    assert "Eco" in data["profiles"] or version == 1  # v1 is the flat format
