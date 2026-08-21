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
"""Issue #396: shipped defaults must satisfy the panel's own conflict rules.

Two root causes were fixed:
  1. The cadence rules (watchdog >= 2*sampling, start_duration >= sampling) were
     violated because DEFAULT_SAMPLING_INTERVAL_BY_DEVICE was never wired, so every
     device ran at the coarse 30 s scalar. Sampling is now device-resolved and
     watchdog/start_duration derive from it.
  2. The confidence rule was backwards: the correct ladder is
     unmatch < match < learning < auto_label, but the rule flagged learning ABOVE
     match. It now enforces learning >= match, which the shipped defaults satisfy.

Fast, pure-unit tests.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import (
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_DEVICE_TYPE,
    CONF_LEARNING_CONFIDENCE,
    CONF_PROFILE_MATCH_THRESHOLD,
    CONF_PROFILE_UNMATCH_THRESHOLD,
    CONF_SAMPLING_INTERVAL,
    CONF_SMART_TERMINATION_DURATION_RATIO,
    CONF_START_DURATION_THRESHOLD,
    CONF_WATCHDOG_INTERVAL,
    DEFAULT_AUTO_LABEL_CONFIDENCE,
    DEFAULT_LEARNING_CONFIDENCE,
    DEFAULT_PROFILE_MATCH_THRESHOLD,
    DEFAULT_PROFILE_UNMATCH_THRESHOLD,
    resolve_sampling_interval_default,
    resolve_start_duration_default,
    resolve_watchdog_interval_default,
)
from custom_components.ha_washdata.suggestion_engine import reconcile_suggestions

_ALL_DEVICE_TYPES = [
    "washing_machine", "dryer", "washer_dryer", "dishwasher",
    "air_fryer", "bread_maker", "pump", "generic", "other",
]


# ---------------------------------------------------------------------------
# Cadence: the device-resolved defaults satisfy the panel's own rules
# ---------------------------------------------------------------------------

def test_resolved_cadence_defaults_satisfy_conflict_rules_for_every_device_type():
    for dt in _ALL_DEVICE_TYPES:
        s = resolve_sampling_interval_default(dt)
        w = resolve_watchdog_interval_default(dt)
        sd = resolve_start_duration_default(dt)
        assert w >= 2 * s, f"{dt}: watchdog {w} < 2*sampling {2*s}"
        assert sd >= s, f"{dt}: start_duration {sd} < sampling {s}"


def test_wet_appliances_keep_fast_sampling():
    for dt in ("washing_machine", "washer_dryer", "dishwasher"):
        assert resolve_sampling_interval_default(dt) == 2.0
    assert resolve_sampling_interval_default("pump") == 10.0
    assert resolve_sampling_interval_default("dryer") == 30.0


def test_watchdog_never_below_the_30s_floor():
    # A fast-sampling device must not get an over-aggressive watchdog tick.
    assert resolve_watchdog_interval_default("washing_machine") == 30
    assert resolve_watchdog_interval_default("dryer") == 61


# ---------------------------------------------------------------------------
# Confidence: the shipped defaults form the correct ascending ladder
# ---------------------------------------------------------------------------

def test_shipped_confidence_defaults_form_the_correct_ladder():
    # unmatch < match < learning < auto_label
    assert (
        DEFAULT_PROFILE_UNMATCH_THRESHOLD
        < DEFAULT_PROFILE_MATCH_THRESHOLD
        <= DEFAULT_LEARNING_CONFIDENCE
        <= DEFAULT_AUTO_LABEL_CONFIDENCE
    )
    # and the specific pair the panel rule now enforces (learning >= match)
    assert DEFAULT_LEARNING_CONFIDENCE >= DEFAULT_PROFILE_MATCH_THRESHOLD


# ---------------------------------------------------------------------------
# reconcile_suggestions: the inverted learning>=match invariant
# ---------------------------------------------------------------------------

def test_reconcile_raises_learning_up_to_match_when_below():
    # A suggested learning_confidence below the (current) match threshold is a
    # violation of learning >= match; reconcile raises learning to match.
    out, changed = reconcile_suggestions(
        {CONF_LEARNING_CONFIDENCE: {"value": 0.3, "reason": "x"}},
        {CONF_PROFILE_MATCH_THRESHOLD: 0.4},
    )
    assert out[CONF_LEARNING_CONFIDENCE]["value"] == 0.4
    assert CONF_LEARNING_CONFIDENCE in changed


def test_reconcile_leaves_learning_above_match_untouched():
    out, changed = reconcile_suggestions(
        {CONF_LEARNING_CONFIDENCE: {"value": 0.6, "reason": "x"}},
        {CONF_PROFILE_MATCH_THRESHOLD: 0.4},
    )
    assert out[CONF_LEARNING_CONFIDENCE]["value"] == 0.6
    assert CONF_LEARNING_CONFIDENCE not in changed


def test_reconcile_does_not_touch_defaults():
    # The shipped defaults are already jointly valid: nothing to reconcile.
    out, changed = reconcile_suggestions(
        {CONF_LEARNING_CONFIDENCE: {"value": DEFAULT_LEARNING_CONFIDENCE, "reason": "x"}},
        {CONF_PROFILE_MATCH_THRESHOLD: DEFAULT_PROFILE_MATCH_THRESHOLD},
    )
    assert out[CONF_LEARNING_CONFIDENCE]["value"] == DEFAULT_LEARNING_CONFIDENCE
    assert CONF_LEARNING_CONFIDENCE not in changed


def test_reconcile_preserves_an_engine_proposed_match_raise():
    # match_threshold is a live detection knob. When the engine deliberately raises
    # it above the current auto-label ceiling, reconcile must keep the raise and lift
    # the ceiling to it, not silently discard it by lowering match to auto.
    out, changed = reconcile_suggestions(
        {CONF_PROFILE_MATCH_THRESHOLD: {"value": 0.7, "reason": "tighten"}},
        {CONF_AUTO_LABEL_CONFIDENCE: 0.5},
    )
    assert out[CONF_PROFILE_MATCH_THRESHOLD]["value"] == 0.7
    assert out[CONF_AUTO_LABEL_CONFIDENCE]["value"] == 0.7
    assert out[CONF_AUTO_LABEL_CONFIDENCE].get("cascade") is True
    assert CONF_PROFILE_MATCH_THRESHOLD not in changed


def test_reconcile_still_lowers_match_when_it_was_not_proposed():
    # When only auto is proposed (below the live match), the ladder is enforced by
    # cascading match down: match was not the engine's own proposal here.
    out, changed = reconcile_suggestions(
        {CONF_AUTO_LABEL_CONFIDENCE: {"value": 0.5, "reason": "loosen"}},
        {CONF_PROFILE_MATCH_THRESHOLD: 0.7},
    )
    assert out[CONF_PROFILE_MATCH_THRESHOLD]["value"] == 0.5
    assert out[CONF_PROFILE_MATCH_THRESHOLD].get("cascade") is True


# ---------------------------------------------------------------------------
# ws_get_options exposes the device-resolved cadence defaults
# ---------------------------------------------------------------------------

def _run_get_options(device_type: str) -> dict:
    entry = MagicMock()
    entry.data = {CONF_DEVICE_TYPE: device_type}
    entry.options = {}
    captured = {}

    def _capture(connection, msg_id, cmd, payload):
        captured["payload"] = payload

    with patch.object(ws_api, "_get_entry", return_value=entry), \
         patch.object(ws_api, "_send_result", side_effect=_capture):
        fn = getattr(ws_api.ws_get_options, "__wrapped__", ws_api.ws_get_options)
        fn(MagicMock(), MagicMock(), {"id": 1, "entry_id": "e1"})
    return captured["payload"]


def test_ws_get_options_defaults_for_coarse_device():
    payload = _run_get_options("dryer")
    d = payload["defaults"]
    assert d[CONF_SAMPLING_INTERVAL] == 30.0
    assert d[CONF_WATCHDOG_INTERVAL] == 61
    assert d[CONF_START_DURATION_THRESHOLD] == 30.0
    # #393: the Smart-Termination ratio field is pre-populated from here (0.98 non-dishwasher)
    assert d[CONF_SMART_TERMINATION_DURATION_RATIO] == 0.98


def test_ws_get_options_defaults_for_wet_device():
    payload = _run_get_options("washing_machine")
    d = payload["defaults"]
    assert d[CONF_SAMPLING_INTERVAL] == 2.0
    assert d[CONF_WATCHDOG_INTERVAL] == 30
    assert d[CONF_START_DURATION_THRESHOLD] == 5.0
    assert d[CONF_SMART_TERMINATION_DURATION_RATIO] == 0.98


def test_ws_get_options_smart_termination_default_for_dishwasher():
    # #393: dishwashers pre-populate the conservative 0.99 gate.
    payload = _run_get_options("dishwasher")
    assert payload["defaults"][CONF_SMART_TERMINATION_DURATION_RATIO] == 0.99
