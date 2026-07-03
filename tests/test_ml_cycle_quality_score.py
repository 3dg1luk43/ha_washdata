"""Tests for WashDataManager._compute_cycle_quality_score.

Verifies that the quality scoring side-effect (storing ml_quality_score on
cycle_data) behaves correctly under different ML model availability and
opt-in states, without requiring a full HA runtime.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.const import CONF_MIN_POWER


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_manager(*, ml_enabled: bool = True) -> WashDataManager:
    """Return a WashDataManager with minimal mocking for unit tests."""
    hass = MagicMock()
    hass.data = {}
    hass.async_create_task = MagicMock(
        side_effect=lambda coro: getattr(coro, "close", lambda: None)()
    )

    entry = MagicMock()
    entry.entry_id = "test"
    entry.title = "Test Washer"
    entry.options = {
        CONF_MIN_POWER: 2.0,
        "power_sensor": "sensor.test_power",
        "enable_ml_models": ml_enabled,
    }

    hass.config_entries.async_get_entry.return_value = entry

    with (
        patch("custom_components.ha_washdata.manager.ProfileStore"),
        patch("custom_components.ha_washdata.manager.CycleDetector"),
    ):
        mgr = WashDataManager(hass, entry)
        mgr.profile_store.get_suggestions = MagicMock(return_value={})
        mgr.profile_store._data = {}
        mgr.profile_store.get_past_cycles = MagicMock(return_value=[
            {"profile_name": "Cotton 60°", "duration": 3500.0, "energy_wh": 700.0, "max_power": 2100.0},
            {"profile_name": "Cotton 60°", "duration": 3600.0, "energy_wh": 720.0, "max_power": 2150.0},
            {"profile_name": "Cotton 60°", "duration": 3650.0, "energy_wh": 710.0, "max_power": 2050.0},
        ])
        return mgr


def _cycle_data_with_power():
    """Minimal cycle_data dict with a real power trace."""
    pts = [(float(i * 30), 500.0 + i * 10.0) for i in range(20)]
    return {
        "id": "test_cycle",
        "profile_name": "Cotton 60°",
        "duration": 600.0,
        "energy_wh": 50.0,
        "max_power": 690.0,
        "match_confidence": 0.85,
        "power_data": [[o, p] for o, p in pts],
    }


# ---------------------------------------------------------------------------
# ML disabled (opt-out)
# ---------------------------------------------------------------------------


def test_ml_disabled_no_score():
    """No ml_quality_score is added when ML models are opted out."""
    mgr = _make_manager(ml_enabled=False)
    cd = _cycle_data_with_power()

    mgr._compute_cycle_quality_score(cd)

    assert "ml_quality_score" not in cd


# ---------------------------------------------------------------------------
# ML enabled: scorer available → score is stored
# ---------------------------------------------------------------------------


def test_ml_enabled_stores_score():
    """When ML is enabled and scorer works, ml_quality_score is written."""
    mgr = _make_manager(ml_enabled=True)
    cd = _cycle_data_with_power()

    def _fake_resolve(capability, store):
        if capability == "quality":
            return (lambda feats: 0.25, "baseline")
        return (None, None)

    with (
        patch("custom_components.ha_washdata.ml.engine.resolve_scorer", side_effect=_fake_resolve),
        patch("custom_components.ha_washdata.ml.engine.ml_models_enabled", return_value=True),
    ):
        mgr._compute_cycle_quality_score(cd)

    assert "ml_quality_score" in cd
    assert cd["ml_quality_score"] == pytest.approx(0.25, abs=0.001)


def test_ml_enabled_score_is_rounded():
    """Stored score is rounded to 3 decimal places."""
    mgr = _make_manager(ml_enabled=True)
    cd = _cycle_data_with_power()

    def _fake_resolve(capability, store):
        if capability == "quality":
            return (lambda feats: 0.1234567, "baseline")
        return (None, None)

    with (
        patch("custom_components.ha_washdata.ml.engine.resolve_scorer", side_effect=_fake_resolve),
        patch("custom_components.ha_washdata.ml.engine.ml_models_enabled", return_value=True),
    ):
        mgr._compute_cycle_quality_score(cd)

    assert cd["ml_quality_score"] == 0.123


# ---------------------------------------------------------------------------
# ML enabled: scorer unavailable → no score stored
# ---------------------------------------------------------------------------


def test_scorer_returns_none_no_score():
    """If resolve_scorer returns (None, None) the quality score is not set."""
    mgr = _make_manager(ml_enabled=True)
    cd = _cycle_data_with_power()

    with (
        patch("custom_components.ha_washdata.ml.engine.resolve_scorer", return_value=(None, None)),
        patch("custom_components.ha_washdata.ml.engine.ml_models_enabled", return_value=True),
    ):
        mgr._compute_cycle_quality_score(cd)

    assert "ml_quality_score" not in cd


# ---------------------------------------------------------------------------
# Resilience: exceptions must never break cycle storage
# ---------------------------------------------------------------------------


def test_scorer_exception_is_swallowed():
    """An exception from the scorer must not propagate; cycle_data is unchanged."""
    mgr = _make_manager(ml_enabled=True)
    cd = _cycle_data_with_power()

    def _boom(*_a, **_kw):
        raise RuntimeError("model exploded")

    with (
        patch("custom_components.ha_washdata.ml.engine.resolve_scorer", return_value=(_boom, "baseline")),
        patch("custom_components.ha_washdata.ml.engine.ml_models_enabled", return_value=True),
    ):
        mgr._compute_cycle_quality_score(cd)  # must not raise

    assert "ml_quality_score" not in cd


def test_missing_profile_name_skips_scoring():
    """No profile_name → scoring is skipped; no score stored."""
    mgr = _make_manager(ml_enabled=True)
    cd = _cycle_data_with_power()
    del cd["profile_name"]

    called = []

    def _fake_resolve(capability, store):
        if capability == "quality":
            called.append(True)
            return (lambda feats: 0.5, "baseline")
        return (None, None)

    with (
        patch("custom_components.ha_washdata.ml.engine.resolve_scorer", side_effect=_fake_resolve),
        patch("custom_components.ha_washdata.ml.engine.ml_models_enabled", return_value=True),
    ):
        mgr._compute_cycle_quality_score(cd)

    assert "ml_quality_score" not in cd


def test_empty_power_data_skips_scoring():
    """Empty power_data list → scoring is skipped; no score stored."""
    mgr = _make_manager(ml_enabled=True)
    cd = _cycle_data_with_power()
    cd["power_data"] = []

    called = []

    def _fake_resolve(capability, store):
        if capability == "quality":
            called.append(True)
            return (lambda feats: 0.5, "baseline")
        return (None, None)

    with (
        patch("custom_components.ha_washdata.ml.engine.resolve_scorer", side_effect=_fake_resolve),
        patch("custom_components.ha_washdata.ml.engine.ml_models_enabled", return_value=True),
    ):
        mgr._compute_cycle_quality_score(cd)

    assert "ml_quality_score" not in cd


def test_no_past_cycles_for_profile_skips_scoring():
    """If there are no past cycles for the profile, stats can't be computed → skip."""
    mgr = _make_manager(ml_enabled=True)
    # Override past cycles to return empty
    mgr.profile_store.get_past_cycles = MagicMock(return_value=[])
    cd = _cycle_data_with_power()

    def _fake_resolve(capability, store):
        if capability == "quality":
            return (lambda feats: 0.9, "baseline")
        return (None, None)

    with (
        patch("custom_components.ha_washdata.ml.engine.resolve_scorer", side_effect=_fake_resolve),
        patch("custom_components.ha_washdata.ml.engine.ml_models_enabled", return_value=True),
    ):
        mgr._compute_cycle_quality_score(cd)

    assert "ml_quality_score" not in cd


# ---------------------------------------------------------------------------
# Score range
# ---------------------------------------------------------------------------


def test_score_is_in_01_range():
    """Score returned by scorer should be passed through unclipped."""
    for raw_score in (0.0, 0.5, 1.0):
        mgr = _make_manager(ml_enabled=True)
        cd = _cycle_data_with_power()

        with (
            patch(
                "custom_components.ha_washdata.ml.engine.resolve_scorer",
                side_effect=lambda cap, s: (lambda f: raw_score, "baseline") if cap == "quality" else (None, None),
            ),
            patch("custom_components.ha_washdata.ml.engine.ml_models_enabled", return_value=True),
        ):
            mgr._compute_cycle_quality_score(cd)

        assert "ml_quality_score" in cd
        assert 0.0 <= cd["ml_quality_score"] <= 1.0
