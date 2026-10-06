"""Audit 2026-10-02 SUGGEST-01/03/04/05/07/09/10/11/15: suggestions that made devices worse.

Each test pins one removed harm:

01  the confidence ladder - percentiles of labels that only exist above the
    thresholds in force, so every apply pulled the ladder down (to ~0.6-0.77
    within 1-3 applies on every device with >= 15 auto labels);
03  completion_min_seconds - its own output (interrupted) was dropped from its
    input, so it ratcheted up and erased short programmes;
04  sampling_interval - the stored interval is measured after the throttle the
    option sets, so applying it ratcheted the throttle without bound;
05  the cooldown stamp - len(past_cycles) stops growing at the retention cap, so
    one Apply-all silenced every non-corrective suggestion forever;
07  dishwasher off_delay - fell back to the 1800 s blind prior even with real
    traces showing no pause;
09  max_duration_ratio - p95 + 0.1 landed below the shipped 1.8 everywhere;
10  unset keys - compared against None, so users were told to set the value
    they were already running;
11  a muted key re-created by the reconcile cascade and applied by Apply-all;
15  no_update_active_timeout - no device floor (register item 165).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import (
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_COMPLETION_MIN_SECONDS,
    CONF_LEARNING_CONFIDENCE,
    CONF_MIN_OFF_GAP,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_OFF_DELAY,
    CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
    CONF_PROFILE_MATCH_THRESHOLD,
    CONF_SAMPLING_INTERVAL,
    DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT_BY_DEVICE,
    DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO,
)
from custom_components.ha_washdata.learning import LearningManager
from custom_components.ha_washdata.suggestion_engine import SuggestionEngine


def _trace(duration: float = 3600.0, n: int = 120, peak: float = 1000.0) -> list[list[float]]:
    step = duration / (n - 1)
    out = []
    for i in range(n):
        frac = i / (n - 1)
        p = peak * (frac / 0.1) if frac < 0.1 else (
            peak * max(0.0, (1.0 - frac) / 0.1) if frac > 0.9 else peak
        )
        out.append([round(i * step, 1), round(p, 1)])
    return out


def _cycle(i: int, *, duration: float = 3600.0, name: str = "Cotton", status: str = "completed",
           **extra: Any) -> dict[str, Any]:
    return {
        "id": f"c{i}", "status": status, "profile_name": name, "duration": duration,
        "power_data": _trace(duration), "start_time": "2026-01-01T10:00:00+00:00", **extra,
    }


def _engine(cycles: list[dict], *, device_type: str = "washing_machine",
            options: dict | None = None, profiles: dict | None = None) -> SuggestionEngine:
    hass = MagicMock()
    entry = MagicMock()
    entry.data = {}
    entry.options = dict(options or {})
    hass.config_entries.async_get_entry.return_value = entry
    store = MagicMock()
    store.get_past_cycles.return_value = cycles
    store.get_profiles.return_value = profiles or {}
    store.get_suggestions.return_value = {}
    store.get_reference_cycles.return_value = []
    store.get_backfill_cycles.return_value = []
    eng = SuggestionEngine(hass, "e", store, device_type=device_type)
    eng._entry_options = lambda: dict(options or {})  # noqa: SLF001
    return eng


def test_no_confidence_ladder_and_no_sampling_interval_are_suggested() -> None:
    cycles = [
        _cycle(i, match_confidence=0.95, label_source="auto_match", sampling_interval=7.0)
        for i in range(30)
    ]
    out = _engine(cycles).generate_detection_suggestions()
    for key in (CONF_LEARNING_CONFIDENCE, CONF_AUTO_LABEL_CONFIDENCE,
                CONF_PROFILE_MATCH_THRESHOLD, CONF_SAMPLING_INTERVAL):
        assert key not in out


def test_completion_min_never_exceeds_half_the_shortest_programme() -> None:
    # A 48 min "Quick wash" next to long programmes; the short "Rinse" runs are
    # already stored interrupted - the shape that ratcheted to 4656 s.
    # One Quick run among 30 long ones: the p05 sits in the long cluster, so half
    # of it (~4500 s) would turn every Quick wash interrupted.
    cycles = [_cycle(i, duration=9000.0, name="Cotton") for i in range(30)]
    cycles += [_cycle(100, duration=2870.0, name="Quick")]
    cycles += [_cycle(200 + i, duration=600.0, name="Rinse", status="interrupted")
               for i in range(3)]
    eng = _engine(cycles)
    with patch.object(SuggestionEngine, "_shortest_profile_duration", return_value=2870.0):
        out = eng.generate_detection_suggestions()
    assert out[CONF_COMPLETION_MIN_SECONDS]["value"] <= 2870.0 / 2


def test_max_ratio_is_never_suggested_below_the_shipped_default() -> None:
    profiles = {"Cotton": {"avg_duration": 3600.0}}
    tight = [_cycle(i, duration=3600.0 + i) for i in range(15)]  # p95 ratio ~1.0
    out = _engine(tight, profiles=profiles).generate_model_suggestions()
    assert CONF_PROFILE_MATCH_MAX_DURATION_RATIO not in out
    # A user who stored the old 1.5 is pointed back to the default.
    out = _engine(
        tight, profiles=profiles, options={CONF_PROFILE_MATCH_MAX_DURATION_RATIO: 1.5}
    ).generate_model_suggestions()
    assert out[CONF_PROFILE_MATCH_MAX_DURATION_RATIO]["value"] == (
        DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO
    )
    # A raised min ratio (0.63 on one corpus device) is pointed back too.
    out = _engine(
        tight, profiles=profiles, options={CONF_PROFILE_MATCH_MIN_DURATION_RATIO: 0.63}
    ).generate_model_suggestions()
    assert out[CONF_PROFILE_MATCH_MIN_DURATION_RATIO]["value"] < 0.63


def test_dishwasher_timeout_and_off_delay_keep_their_floors() -> None:
    cycles = [_cycle(i, duration=7200.0) for i in range(8)]  # traced, no pauses
    out = _engine(cycles, device_type="dishwasher").generate_operational_suggestions(30.0, 5.0)
    assert out[CONF_NO_UPDATE_ACTIVE_TIMEOUT]["value"] >= (
        DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT_BY_DEVICE["dishwasher"]
    )
    assert out[CONF_OFF_DELAY]["value"] < 1800


def _learning(store: MagicMock, options: dict) -> LearningManager:
    hass = MagicMock()
    entry = MagicMock()
    entry.data = {}
    entry.options = dict(options)
    hass.config_entries.async_get_entry.return_value = entry
    lm = LearningManager(hass, "e", store, device_type="washing_machine")
    lm.suggestion_engine.apply_suggestions = MagicMock()
    return lm


def test_the_cooldown_lifts_at_the_retention_cap() -> None:
    store = MagicMock()
    store.get_past_cycles.return_value = [{}] * 200       # at the cap, stays 200
    store.get_lifetime_cycle_count.return_value = 250     # 50 cycles since the apply
    store.get_suggestion_apply_cycle_count.return_value = 200
    store.get_locked_suggestions.return_value = []
    store.get_suggestions.return_value = {}
    lm = _learning(store, {CONF_OFF_DELAY: 180})
    lm._apply_suggestions_and_notify({CONF_OFF_DELAY: {"value": 600, "reason": "x"}})
    assert CONF_OFF_DELAY in lm.suggestion_engine.apply_suggestions.call_args.args[0]


def test_an_unset_key_at_its_effective_value_is_not_prompted() -> None:
    store = MagicMock()
    store.get_past_cycles.return_value = []
    store.get_lifetime_cycle_count.return_value = 0
    store.get_suggestion_apply_cycle_count.return_value = 0
    store.get_locked_suggestions.return_value = []
    store.get_suggestions.return_value = {}
    lm = _learning(store, {})  # a fresh entry: options == {}
    lm._apply_suggestions_and_notify({
        CONF_OFF_DELAY: {"value": 180, "reason": "the default"},   # what it runs with
        CONF_MIN_OFF_GAP: {"value": 900, "reason": "a real change"},
    })
    applied = lm.suggestion_engine.apply_suggestions.call_args.args[0]
    assert CONF_OFF_DELAY not in applied
    assert CONF_MIN_OFF_GAP in applied


async def test_apply_all_never_applies_a_muted_key() -> None:
    entry = MagicMock()
    entry.entry_id = "e1"
    entry.data = {}
    entry.options = {CONF_OFF_DELAY: 180, CONF_MIN_OFF_GAP: 480}
    manager = MagicMock()
    manager.device_type = "washing_machine"
    manager.profile_store.get_suggestions.return_value = {
        CONF_OFF_DELAY: {"value": 900},
        CONF_MIN_OFF_GAP: {"value": 900},  # cascade-created by Rule 2, but muted
    }
    manager.profile_store.get_locked_suggestions.return_value = [CONF_MIN_OFF_GAP]
    manager.profile_store.get_lifetime_cycle_count.return_value = 10
    manager.profile_store.clear_suggestions = AsyncMock()
    hass = MagicMock()
    with patch.object(ws_api, "_get_manager", return_value=manager), patch.object(
        ws_api, "_get_entry", return_value=entry
    ), patch.object(ws_api, "_record_option_changes", AsyncMock()):
        await ws_api.ws_apply_suggestions.__wrapped__(
            hass, MagicMock(), {"id": 1, "entry_id": "e1", "keys": [CONF_OFF_DELAY, CONF_MIN_OFF_GAP]}
        )
    _args, kwargs = hass.config_entries.async_update_entry.call_args
    assert kwargs["options"][CONF_OFF_DELAY] == 900
    assert kwargs["options"][CONF_MIN_OFF_GAP] == 480


def test_end_energy_is_raised_only_when_it_forbids_an_end() -> None:
    from custom_components.ha_washdata.const import (
        CONF_END_ENERGY_THRESHOLD,
        CONF_STOP_THRESHOLD_W,
    )

    cycles = [_cycle(i) for i in range(12)]
    opts = {CONF_STOP_THRESHOLD_W: 2.0, CONF_OFF_DELAY: 600, CONF_END_ENERGY_THRESHOLD: 0.01}
    out = _engine(cycles, options=opts).run_batch_simulation(cycles)
    # 2 W over 600 s implies 0.334 Wh; anything below it can never be met.
    assert out[CONF_END_ENERGY_THRESHOLD]["value"] == 0.34
    assert out[CONF_END_ENERGY_THRESHOLD]["corrective"] is True
    opts[CONF_END_ENERGY_THRESHOLD] = 0.5
    out = _engine(cycles, options=opts).run_batch_simulation(cycles)
    assert CONF_END_ENERGY_THRESHOLD not in out


def test_an_oversized_json_integer_does_not_abort_the_batch_pass() -> None:
    """`float(10**400)` raises OverflowError; the end-energy block caught only
    TypeError/ValueError, so one oversized stored option aborted every suggestion."""
    from custom_components.ha_washdata.const import (
        CONF_END_ENERGY_THRESHOLD,
        CONF_STOP_THRESHOLD_W,
    )

    cycles = [_cycle(i) for i in range(12)]
    for key in (CONF_STOP_THRESHOLD_W, CONF_OFF_DELAY):
        opts = {CONF_STOP_THRESHOLD_W: 2.0, CONF_OFF_DELAY: 600, CONF_END_ENERGY_THRESHOLD: 0.01}
        opts[key] = 10**400
        out = _engine(cycles, options=opts).run_batch_simulation(cycles)
        assert CONF_END_ENERGY_THRESHOLD not in out
