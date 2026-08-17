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

"""Issue #331 residual: a manual relabel should clear the red dot even when the
cycle has no pending feedback.

The #331 core (relabel resolves pending feedback) shipped in 0.5.3. Residual: a
cycle sitting in the review queue only because of an uncertain quality label or a
force_stopped/interrupted status has no pending feedback, so relabeling did not
clear its red dot - the user had to open the quality "Save evaluation" flow. A
manual (re)label of such a cycle now stamps ml_review.reviewed_at (leaving the
quality label intact), which clears the needs-review flag. Normal cycles are not
touched.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata.learning import LearningManager


class _MockStore:
    def __init__(self):
        self.feedback: dict = {}
        self.pending: dict = {}
        self.cycles: list = []
        self.saved = 0

    def get_feedback_history(self):
        return self.feedback

    def get_pending_feedback(self):
        return self.pending

    def get_past_cycles(self):
        return self.cycles

    async def set_cycle_review(self, cycle_id, **kw):
        cyc = next((c for c in self.cycles if c.get("id") == cycle_id), None)
        if cyc is None:
            raise ValueError("not found")
        rv = dict(cyc.get("ml_review") or {})
        rv["reviewed_at"] = datetime.now(timezone.utc).isoformat()
        cyc["ml_review"] = rv
        self.saved += 1
        return True

    async def async_save(self):
        self.saved += 1


@pytest.fixture
def learning_manager(mock_hass):
    entry = MagicMock()
    entry.options = {}
    entry.title = "Test Entry"
    mock_hass.config_entries.async_get_entry.return_value = entry
    return LearningManager(mock_hass, "test_entry", _MockStore())


@pytest.mark.asyncio
async def test_relabel_marks_uncertain_cycle_reviewed(learning_manager):
    store = learning_manager.profile_store
    store.cycles = [{"id": "cyc1", "status": "completed", "ml_review": {"label": "uncertain", "reviewed_at": None}}]

    resolved = await learning_manager.async_resolve_pending_from_label("cyc1", "Eco 50")

    assert resolved is True
    assert store.cycles[0]["ml_review"]["reviewed_at"] is not None


@pytest.mark.asyncio
async def test_relabel_marks_force_stopped_cycle_reviewed(learning_manager):
    store = learning_manager.profile_store
    store.cycles = [{"id": "cyc1", "status": "force_stopped", "ml_review": None}]

    resolved = await learning_manager.async_resolve_pending_from_label("cyc1", "Eco 50")

    assert resolved is True
    assert store.cycles[0]["ml_review"]["reviewed_at"] is not None


@pytest.mark.asyncio
async def test_relabel_normal_cycle_is_not_marked_reviewed(learning_manager):
    store = learning_manager.profile_store
    store.cycles = [{"id": "cyc1", "status": "completed", "ml_review": None}]

    resolved = await learning_manager.async_resolve_pending_from_label("cyc1", "Eco 50")

    assert resolved is False
    assert store.cycles[0].get("ml_review") in (None, {})
    assert store.saved == 0


@pytest.mark.asyncio
async def test_already_reviewed_cycle_not_restamped(learning_manager):
    store = learning_manager.profile_store
    store.cycles = [{"id": "cyc1", "status": "force_stopped", "ml_review": {"reviewed_at": "2026-01-01T00:00:00+00:00"}}]

    resolved = await learning_manager.async_resolve_pending_from_label("cyc1", "Eco 50")

    assert resolved is False
    assert store.cycles[0]["ml_review"]["reviewed_at"] == "2026-01-01T00:00:00+00:00"
    assert store.saved == 0
