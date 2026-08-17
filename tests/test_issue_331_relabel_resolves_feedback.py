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
"""Issue #331 - manually (re)labelling a review cycle must resolve its pending
feedback and drop it from the "must be reviewed" queue.

Before the fix, ``label_cycle`` only touched ``assign_profile_to_cycle`` and never
cleared ``pending_feedback``, so a re-labelled cycle stayed in review forever with
the stale original detection (``user_response: null``). The learning manager now
exposes ``async_resolve_pending_from_label`` which the WS command + service call
after applying the label.
"""
import pytest
from unittest.mock import MagicMock

from custom_components.ha_washdata.learning import LearningManager
from custom_components.ha_washdata.const import (
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_LEARNING_CONFIDENCE,
    CONF_DURATION_TOLERANCE,
)


class _MockStore:
    """Minimal store exposing just the surface the resolver touches."""

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
        self.saved += 1
        return True

    async def async_save(self):
        self.saved += 1


@pytest.fixture
def learning_manager(mock_hass):
    entry = MagicMock()
    entry.options = {
        CONF_AUTO_LABEL_CONFIDENCE: 0.9,
        CONF_LEARNING_CONFIDENCE: 0.6,
        CONF_DURATION_TOLERANCE: 0.1,
    }
    entry.title = "Test Entry"
    mock_hass.config_entries.async_get_entry.return_value = entry
    return LearningManager(mock_hass, "test_entry", _MockStore())


def _seed_pending(store, cycle_id="cyc1", detected="Chef 70°", confidence=0.55):
    store.pending[cycle_id] = {
        "cycle_id": cycle_id,
        "detected_profile": detected,
        "confidence": confidence,
        "user_response": None,
    }


@pytest.mark.asyncio
async def test_relabel_to_different_profile_resolves_as_correction(learning_manager):
    """The dishwasher case from the issue: Chef 70° -> Auto 45° bis 65°."""
    store = learning_manager.profile_store
    _seed_pending(store)

    resolved = await learning_manager.async_resolve_pending_from_label(
        "cyc1", "Auto 45° bis 65°"
    )

    assert resolved is True
    # Dropped from the review queue.
    assert "cyc1" not in store.pending
    # A correction record was written with the right provenance.
    rec = store.feedback["cyc1"]
    assert rec["user_confirmed"] is False
    assert rec["corrected_profile"] == "Auto 45° bis 65°"
    assert rec["original_detected_profile"] == "Chef 70°"
    assert store.saved == 1


@pytest.mark.asyncio
async def test_relabel_to_same_profile_resolves_as_confirmation(learning_manager):
    store = learning_manager.profile_store
    _seed_pending(store, detected="Eco 50")

    resolved = await learning_manager.async_resolve_pending_from_label("cyc1", "Eco 50")

    assert resolved is True
    assert "cyc1" not in store.pending
    rec = store.feedback["cyc1"]
    assert rec["user_confirmed"] is True
    assert rec["corrected_profile"] is None


@pytest.mark.asyncio
async def test_remove_label_resolves_as_rejection(learning_manager):
    store = learning_manager.profile_store
    _seed_pending(store)

    resolved = await learning_manager.async_resolve_pending_from_label("cyc1", None)

    assert resolved is True
    assert "cyc1" not in store.pending
    rec = store.feedback["cyc1"]
    assert rec["user_confirmed"] is False
    assert rec["corrected_profile"] is None


@pytest.mark.asyncio
async def test_no_pending_feedback_is_noop(learning_manager):
    """Labelling a cycle with no pending feedback must not fabricate a record."""
    store = learning_manager.profile_store

    resolved = await learning_manager.async_resolve_pending_from_label("cyc1", "Eco 50")

    assert resolved is False
    assert store.feedback == {}
    assert store.saved == 0
