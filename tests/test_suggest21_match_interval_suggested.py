"""The match-interval suggestion is shown and applied again (audit SUGGEST-21).

It was held back on 2026-10-04 because a shorter `profile_match_interval` let an
ambiguous match tick land in the end wait and delay the end (register item 469(b):
6.7 -> 23.2 min on one cycle). `match_rules.hold_in_ending` keeps such a tick from
deferring the end (tests/test_item469b_ending_ambiguous_hold.py), so the key is back
in the one filter behind the Settings list, the Overview card and Apply all.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import (
    CONF_MIN_POWER,
    CONF_PROFILE_MATCH_INTERVAL,
)


def _store(locked: list[str] | None = None) -> MagicMock:
    store = MagicMock()
    store.get_suggestions.return_value = {
        CONF_PROFILE_MATCH_INTERVAL: {"value": 60.4, "reason": "cadence"},
        CONF_MIN_POWER: {"value": 3.5, "reason": "floor"},
    }
    store.get_locked_suggestions.return_value = locked or []
    return store


def test_a_stored_match_interval_suggestion_is_shown_and_applied_as_an_int():
    shown = {
        k: suggested
        for k, _item, suggested, _cur in ws_api._visible_suggestions(  # noqa: SLF001
            _store(), {CONF_MIN_POWER: 2.0, CONF_PROFILE_MATCH_INTERVAL: 300}, "washing_machine"
        )
    }
    assert shown[CONF_PROFILE_MATCH_INTERVAL] == 60
    assert CONF_MIN_POWER in shown


def test_muting_it_still_hides_it():
    shown = [
        k for k, *_ in ws_api._visible_suggestions(  # noqa: SLF001
            _store([CONF_PROFILE_MATCH_INTERVAL]), {CONF_PROFILE_MATCH_INTERVAL: 300},
            "washing_machine",
        )
    ]
    assert CONF_PROFILE_MATCH_INTERVAL not in shown
