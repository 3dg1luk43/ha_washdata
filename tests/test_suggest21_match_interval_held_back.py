"""The match-interval suggestion is held back (audit SUGGEST-21, decision 2026-10-04).

A shorter `profile_match_interval` lets a match tick land inside the end wait and
delays the end (register item 469(b): 6.7 -> 20.2 min on one cycle). The engine
still computes it, but it is neither shown nor applied until 469(b) is fixed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import (
    CONF_MIN_POWER,
    CONF_PROFILE_MATCH_INTERVAL,
)


def test_a_stored_match_interval_suggestion_is_not_shown_or_applied():
    store = MagicMock()
    store.get_suggestions.return_value = {
        CONF_PROFILE_MATCH_INTERVAL: {"value": 60, "reason": "cadence"},
        CONF_MIN_POWER: {"value": 3.5, "reason": "floor"},
    }
    store.get_locked_suggestions.return_value = []
    shown = [k for k, *_ in ws_api._visible_suggestions(store, {CONF_MIN_POWER: 2.0}, "washing_machine")]  # noqa: SLF001
    assert CONF_PROFILE_MATCH_INTERVAL not in shown
    assert CONF_MIN_POWER in shown  # the filter still shows real suggestions
