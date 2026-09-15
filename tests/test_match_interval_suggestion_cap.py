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
"""Issue #431: cap the ``profile_match_interval`` suggestion by the shortest profile.

The suggestion was cadence-only (``median_dt * 10``), so a plug reporting every
60 s produced 599 s - longer than ``DEFAULT_PROFILE_MATCH_INTERVAL`` (300 s)
itself, which makes applying the suggestion strictly WORSE than never touching
the setting. With ``match_persistence`` 3 no program could then be committed
before ~30 min, however short it runs.

The cap bounds the *decision budget* (``interval * persistence``), not the
interval alone, at ``MATCH_INTERVAL_SUGGESTION_DECISION_FRAC`` of the shortest
known profile - so raising persistence cannot reintroduce the same wait.

Fast, pure-unit tests (no HA boot, no file I/O, no cycle_data replay).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata.const import (
    CONF_MATCH_PERSISTENCE,
    CONF_PROFILE_MATCH_INTERVAL,
    DEFAULT_MATCH_PERSISTENCE,
    DEFAULT_PROFILE_MATCH_INTERVAL,
    MATCH_INTERVAL_SUGGESTION_DECISION_FRAC,
    MATCH_INTERVAL_SUGGESTION_MIN_S,
)
from custom_components.ha_washdata.suggestion_engine import SuggestionEngine


def _engine(
    profiles: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> SuggestionEngine:
    """A suggestion engine bound to a fake store and an options snapshot.

    ``get_past_cycles`` returns nothing so the off-delay branch falls through to
    its cadence fallback and only the match-interval block under test varies.
    """
    store = MagicMock()
    store.get_profiles.return_value = profiles if profiles is not None else {}
    store.get_past_cycles.return_value = []
    eng = SuggestionEngine(MagicMock(), "entry", store, device_type="washing_machine")
    return eng.for_job(options or {})


def _match_suggestion(
    median_dt: float,
    profiles: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eng = _engine(profiles, options)
    out = eng.generate_operational_suggestions(p95_dt=median_dt, median_dt=median_dt)
    return out[CONF_PROFILE_MATCH_INTERVAL]


# ---------------------------------------------------------------------------
# The reported case
# ---------------------------------------------------------------------------


def test_reported_case_no_longer_exceeds_the_default() -> None:
    """A 60 s plug on a 2579 s program: 599 s was worse than doing nothing."""
    sug = _match_suggestion(60.0, {"Eco": {"avg_duration": 2579.0}})
    assert sug["value"] < DEFAULT_PROFILE_MATCH_INTERVAL, (
        "a suggestion the user can apply with one click must never be worse "
        "than the default they started from"
    )
    # 0.15 * 2579 / 3 = 128.95 -> 128
    assert sug["value"] == 128
    assert sug["reason_key"] == "suggestion.reason.match_interval_capped"


def test_decision_budget_is_the_configured_fraction() -> None:
    """interval * persistence must fit inside the budget, by construction."""
    shortest = 2579.0
    sug = _match_suggestion(60.0, {"Eco": {"avg_duration": shortest}})
    budget = shortest * MATCH_INTERVAL_SUGGESTION_DECISION_FRAC
    assert sug["value"] * DEFAULT_MATCH_PERSISTENCE <= budget


# ---------------------------------------------------------------------------
# When the cap must NOT bind
# ---------------------------------------------------------------------------


def test_dense_sensor_keeps_the_cadence_value() -> None:
    """A 6 s cadence against a 2579 s program is already well inside the budget."""
    sug = _match_suggestion(6.0, {"Eco": {"avg_duration": 2579.0}})
    assert sug["value"] == 60  # 6 * 10, untouched
    assert sug["reason_key"] == "suggestion.reason.match_interval"


def test_no_profiles_leaves_the_suggestion_unchanged() -> None:
    """A fresh install has nothing to cap against and must behave as before."""
    sug = _match_suggestion(60.0, {})
    assert sug["value"] == 600  # 60 * 10, untouched (the reporter's 599 = 59.9 * 10)
    assert sug["reason_key"] == "suggestion.reason.match_interval"


def test_placeholder_durations_are_ignored() -> None:
    """A <=60 s 'profile' is a placeholder, not evidence of a short program.

    Mirrors the ``avg > 60`` guard the model-suggestion generator already uses;
    without it a hand-created profile would collapse the suggestion to its floor.
    """
    sug = _match_suggestion(60.0, {"Stub": {"avg_duration": 30.0}})
    assert sug["value"] == 600
    assert sug["reason_key"] == "suggestion.reason.match_interval"


def test_malformed_profiles_do_not_raise() -> None:
    """Suggestions run in an executor and must never take the panel down."""
    profiles = {
        "bad": None,
        "worse": {"avg_duration": "not a number"},
        "nan": {"avg_duration": float("nan")},
        # json keeps an oversized literal as an unbounded int and float() on one
        # raises OverflowError, which is neither a TypeError nor a ValueError: it
        # used to abort the whole operational-suggestion pass.
        "imported": {"avg_duration": 10**400},
        "good": {"avg_duration": 2579.0},
    }
    sug = _match_suggestion(60.0, profiles)
    assert sug["value"] == 128  # only "good" counted


# ---------------------------------------------------------------------------
# Persistence coupling - the half the reporter raised as "optionally"
# ---------------------------------------------------------------------------


def test_higher_persistence_shrinks_the_interval() -> None:
    """Capping the interval alone would let persistence undo the fix."""
    profiles = {"Eco": {"avg_duration": 2579.0}}
    at3 = _match_suggestion(60.0, profiles, {CONF_MATCH_PERSISTENCE: 3})["value"]
    at6 = _match_suggestion(60.0, profiles, {CONF_MATCH_PERSISTENCE: 6})["value"]
    assert at6 * 6 <= at3 * 3 + 6, "the decision budget must not grow with persistence"
    assert at6 < at3


def test_a_profile_added_mid_pass_does_not_abort_the_suggestions() -> None:
    """`get_profiles()` hands back the live map and this runs in an executor.

    The event loop can label, rename or delete a profile while the pass is
    iterating, which raised "dictionary changed size during iteration" and took
    the whole suggestion pass with it.
    """
    live = {"Eco": {"avg_duration": 2579.0}, "Cotton": {"avg_duration": 7200.0}}

    class _MutatingView(dict):
        """Grows WHILE it is being iterated, which is what the event loop does.

        Mutating before iteration starts would not reproduce anything: the error
        comes from the dict's own iterator noticing the size changed under it.
        """

        def values(self):  # noqa: D102
            view = super().values()

            def _gen():
                for index, value in enumerate(view):
                    if index == 0:
                        self["Added mid-pass"] = {"avg_duration": 1800.0}
                    yield value

            return _gen()

    store = MagicMock()
    store.get_profiles.return_value = _MutatingView(live)
    store.get_past_cycles.return_value = []
    eng = SuggestionEngine(MagicMock(), "entry", store, device_type="washing_machine")
    sug = eng.for_job({}).generate_operational_suggestions(p95_dt=60.0, median_dt=60.0)

    assert CONF_PROFILE_MATCH_INTERVAL in sug


@pytest.mark.parametrize(
    "bad",
    # float("inf") is the one that raises OverflowError rather than ValueError:
    # json parses a bare `Infinity` literal into it, so a hand-edited import can
    # store one, and escaping here aborted every operational suggestion.
    ["", None, "abc", 0, -2, float("inf"), float("-inf"), float("nan")],
)
def test_invalid_persistence_falls_back_to_the_default(bad: Any) -> None:
    """A junk stored value must not divide by zero or invert the cap."""
    profiles = {"Eco": {"avg_duration": 2579.0}}
    sug = _match_suggestion(60.0, profiles, {CONF_MATCH_PERSISTENCE: bad})
    assert sug["value"] >= 10
    assert sug["value"] <= 600


# ---------------------------------------------------------------------------
# Floors and payload shape
# ---------------------------------------------------------------------------


def test_very_short_profile_still_respects_the_10s_floor() -> None:
    """A 61 s program must not drive the matcher into a per-second poll."""
    sug = _match_suggestion(60.0, {"Rinse": {"avg_duration": 61.0}})
    assert sug["value"] == MATCH_INTERVAL_SUGGESTION_MIN_S


def test_the_floor_says_it_broke_the_budget_rule() -> None:
    """When the floor wins, the budget rule does not hold and must not be claimed.

    61 s at persistence 3 caps to 3 s, the floor lifts it back to 10 s, and three
    consecutive matches then take 30 s - about half the program, far past the 15%
    the capped wording promises. The reason is shown beside the value the user is
    asked to accept, so it says the minimum applied instead.
    """
    sug = _match_suggestion(60.0, {"Rinse": {"avg_duration": 61.0}})
    budget = 61.0 * MATCH_INTERVAL_SUGGESTION_DECISION_FRAC

    assert sug["value"] * DEFAULT_MATCH_PERSISTENCE > budget  # the rule is broken
    assert sug["reason_key"] == "suggestion.reason.match_interval_floored"
    assert set(sug["reason_params"]) == {
        "median", "shortest", "persistence", "pct", "cap", "minimum", "budget",
    }
    assert all(isinstance(v, str) for v in sug["reason_params"].values())
    assert "minimum" in sug["reason"]


def test_capped_reason_params_cover_every_placeholder() -> None:
    """The panel resolves reason_key with reason_params; a missing one renders raw."""
    sug = _match_suggestion(60.0, {"Eco": {"avg_duration": 2579.0}})
    assert set(sug["reason_params"]) == {"median", "shortest", "persistence", "pct"}
    assert all(isinstance(v, str) for v in sug["reason_params"].values())
    assert sug["reason"], "English fallback must be present for untranslated panels"
