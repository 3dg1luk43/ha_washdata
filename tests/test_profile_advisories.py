"""Tests for ProfileStore.compute_profile_advisories.

Consolidates the existing per-profile health/trend signals into a ranked list of
actionable maintenance advisories (surfaced as recommendations in the Profiles
tab, never a notification). Pure statistics; must never raise.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from custom_components.ha_washdata.profile_store import ProfileStore


def _store(*, health: dict, trends: dict) -> ProfileStore:
    store = MagicMock(spec=ProfileStore)
    store.compute_profile_health.return_value = health
    store.compute_profile_trends.return_value = trends
    store.compute_profile_advisories = ProfileStore.compute_profile_advisories.__get__(
        store, ProfileStore
    )
    return store


def test_empty_when_no_signals():
    store = _store(health={}, trends={})
    assert store.compute_profile_advisories() == []


def test_poor_health_warning():
    store = _store(health={"Cotton": {"health_status": "poor"}}, trends={})
    adv = store.compute_profile_advisories()
    assert len(adv) == 1
    assert adv[0]["profile"] == "Cotton"
    assert adv[0]["severity"] == "warning"
    assert adv[0]["code"] == "poor_health"


def test_duration_trend_up_info():
    store = _store(
        health={"Eco": {"health_status": "healthy"}},
        # duration_slope_pct is already %/cycle (compute_profile_trends emits
        # dur_slope * 100), so a real +12%/cycle trend is 12.0, not 0.12.
        trends={"Eco": {"duration_trend": "up", "duration_slope_pct": 12.0}},
    )
    adv = store.compute_profile_advisories()
    assert len(adv) == 1
    assert adv[0]["code"] == "duration_trend_up"
    assert adv[0]["severity"] == "info"
    assert "12%" in adv[0]["message"]
    # Regression guard for the 100x re-scale bug: must not read "1200%".
    assert "1200%" not in adv[0]["message"]


def test_energy_trend_up_when_duration_stable():
    store = _store(
        health={},
        trends={"Cotton": {"duration_trend": "stable", "energy_trend": "up", "energy_slope_pct": 9.0}},
    )
    adv = store.compute_profile_advisories()
    assert len(adv) == 1
    assert adv[0]["code"] == "energy_trend_up"
    # energy_slope_pct is already %/cycle: 9.0 must render as "9%", not "900%".
    assert "9%" in adv[0]["message"]
    assert "900%" not in adv[0]["message"]


def test_poor_health_suppresses_trend_advice_for_same_profile():
    # A profile flagged poor shouldn't also get a (redundant) trend advisory.
    store = _store(
        health={"Cotton": {"health_status": "poor"}},
        trends={"Cotton": {"duration_trend": "up", "duration_slope_pct": 20.0}},
    )
    adv = store.compute_profile_advisories()
    assert len(adv) == 1
    assert adv[0]["code"] == "poor_health"


def test_warnings_ranked_before_info():
    store = _store(
        health={"Bad": {"health_status": "poor"}},
        trends={"Drift": {"duration_trend": "up", "duration_slope_pct": 10.0}},
    )
    adv = store.compute_profile_advisories()
    assert [a["severity"] for a in adv] == ["warning", "info"]


def test_healthy_stable_profile_no_advice():
    store = _store(
        health={"Cotton": {"health_status": "healthy"}},
        trends={"Cotton": {"duration_trend": "stable", "energy_trend": "stable"}},
    )
    assert store.compute_profile_advisories() == []


def test_never_raises():
    store = MagicMock(spec=ProfileStore)
    store.compute_profile_health.side_effect = RuntimeError("boom")
    store.compute_profile_advisories = ProfileStore.compute_profile_advisories.__get__(
        store, ProfileStore
    )
    assert store.compute_profile_advisories() == []
