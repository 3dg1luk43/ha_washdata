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
"""Phase B: provenance code derivation + trim edit marker."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.ha_washdata import store_account
from custom_components.ha_washdata.const import QC_EDITED, QC_MANUAL, QC_RECORDING
from custom_components.ha_washdata.store import derive_qc, online_features_enabled


def test_derive_qc_recording():
    assert derive_qc({"meta": {"source": "recorder"}}) == QC_RECORDING
    assert derive_qc({"meta": {"original_samples": 500}}) == QC_RECORDING


def test_derive_qc_edited():
    assert derive_qc({"meta": {"edited": True}}) == QC_EDITED


def test_derive_qc_manual_default():
    assert derive_qc({}) == QC_MANUAL
    assert derive_qc({"meta": {}}) == QC_MANUAL
    assert derive_qc({"ml_review": {"golden": True}}) == QC_MANUAL


def test_recorder_precedence_over_edited():
    # A trimmed recording is still classed as a recording.
    assert derive_qc({"meta": {"source": "recorder", "edited": True}}) == QC_RECORDING


@pytest.mark.asyncio
async def test_online_features_gating_is_global():
    """Online features are now integration-wide (device-agnostic), not per-entry."""
    hass = MagicMock()
    hass.data = {}
    fake = MagicMock()
    fake.async_save = AsyncMock()
    hass.data[store_account._DATA_KEY] = {"store": fake, "data": {"online_enabled": False, "account": {}}}
    assert online_features_enabled(hass) is False
    await store_account.async_set_online(hass, True)
    assert online_features_enabled(hass) is True
