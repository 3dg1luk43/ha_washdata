"""Tests for Group B (Energy, Cost & Sustainability) features B1-B4.

B1 — HA Energy dashboard entity (lifetime-accumulating energy)
B2 — Per-profile average / total cost in list_profiles()
B3 — New finish-notification template variables (time_finished, cycle_count, vs_typical)
B4 — Peak-rate awareness tip at cycle start
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata.const import (
    CONF_PEAK_RATE_MESSAGE,
    CONF_PEAK_RATE_THRESHOLD,
    DEFAULT_PEAK_RATE_MESSAGE,
)
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.profile_store import ProfileStore


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _energy_store(initial: dict | None = None) -> ProfileStore:
    """ProfileStore mock with real lifetime-energy methods bound and a mutable _data."""
    store = MagicMock(spec=ProfileStore)
    store._data = {} if initial is None else dict(initial)
    store.get_lifetime_energy_wh = ProfileStore.get_lifetime_energy_wh.__get__(
        store, ProfileStore
    )
    store.async_add_lifetime_energy_wh = (
        ProfileStore.async_add_lifetime_energy_wh.__get__(store, ProfileStore)
    )
    return store


def _list_profiles_store(profiles: dict, past_cycles: list[dict]) -> ProfileStore:
    """ProfileStore mock with real list_profiles bound; envelopes stubbed to None."""
    store = MagicMock(spec=ProfileStore)
    store._data = {"profiles": profiles, "past_cycles": past_cycles}
    store.get_envelope = MagicMock(return_value=None)
    store.list_profiles = ProfileStore.list_profiles.__get__(store, ProfileStore)
    return store


def _peak_rate_manager(title: str = "Washer") -> WashDataManager:
    """Minimal manager-like stub exposing the real _peak_rate_tip / _safe_format_template."""
    m = MagicMock()
    m.config_entry.title = title
    m._logger = MagicMock()
    m._safe_format_template = WashDataManager._safe_format_template.__get__(
        m, WashDataManager
    )
    m._peak_rate_tip = WashDataManager._peak_rate_tip.__get__(m, WashDataManager)
    return m


# ---------------------------------------------------------------------------
# B1 — lifetime energy accumulator
# ---------------------------------------------------------------------------


def test_b1_get_lifetime_energy_default_zero():
    store = _energy_store()
    assert store.get_lifetime_energy_wh() == 0.0


def test_b1_get_lifetime_energy_reads_data():
    store = _energy_store({"lifetime_energy_wh": 1234.5})
    assert store.get_lifetime_energy_wh() == 1234.5


def test_b1_get_lifetime_energy_garbage_returns_zero():
    store = _energy_store({"lifetime_energy_wh": "oops"})
    assert store.get_lifetime_energy_wh() == 0.0


async def test_b1_add_lifetime_energy_accumulates_and_persists():
    store = _energy_store()
    await store.async_add_lifetime_energy_wh(500.0)
    assert store._data["lifetime_energy_wh"] == pytest.approx(500.0)
    assert store.async_save.called
    # Second call sums onto the first — no double-count, no reset.
    await store.async_add_lifetime_energy_wh(250.25)
    assert store._data["lifetime_energy_wh"] == pytest.approx(750.25)


async def test_b1_add_lifetime_energy_rounds_to_3_decimals():
    store = _energy_store()
    await store.async_add_lifetime_energy_wh(1.23456)
    assert store._data["lifetime_energy_wh"] == 1.235


async def test_b1_add_lifetime_energy_ignores_negative():
    store = _energy_store({"lifetime_energy_wh": 100.0})
    await store.async_add_lifetime_energy_wh(-50.0)
    # Negative is clamped to 0 (max(0.0, wh)) so the total is unchanged.
    assert store._data["lifetime_energy_wh"] == 100.0


async def test_b1_add_lifetime_energy_ignores_garbage():
    store = _energy_store({"lifetime_energy_wh": 100.0})
    await store.async_add_lifetime_energy_wh("not-a-number")
    assert store._data["lifetime_energy_wh"] == 100.0
    # Bad input returns before saving.
    assert not store.async_save.called


def test_b1_manager_lifetime_energy_kwh_property():
    m = MagicMock()
    m.profile_store.get_lifetime_energy_wh.return_value = 2500.0
    assert WashDataManager.lifetime_energy_kwh.fget(m) == 2.5


def test_b1_manager_lifetime_energy_kwh_rounds():
    m = MagicMock()
    m.profile_store.get_lifetime_energy_wh.return_value = 1234.56
    # 1234.56 Wh -> 1.23456 kWh -> round 3 -> 1.235
    assert WashDataManager.lifetime_energy_kwh.fget(m) == 1.235


# ---------------------------------------------------------------------------
# B2 — per-profile cost aggregates in list_profiles()
# ---------------------------------------------------------------------------


def _profile_of(result: list[dict], name: str) -> dict:
    return next(p for p in result if p["name"] == name)


def test_b2_list_profiles_cost_aggregates():
    profiles = {"Cotton 60": {"avg_duration": 3600}}
    past = [
        {"profile_name": "Cotton 60", "cost": 0.50, "start_time": "2026-01-01"},
        {"profile_name": "Cotton 60", "cost": 0.30, "start_time": "2026-01-02"},
        {"profile_name": "Cotton 60", "cost": 0.40, "start_time": "2026-01-03"},
    ]
    store = _list_profiles_store(profiles, past)
    p = _profile_of(store.list_profiles(), "Cotton 60")
    assert p["total_cost"] == pytest.approx(1.20)
    assert p["avg_cost"] == pytest.approx(0.40)


def test_b2_list_profiles_cost_none_when_absent():
    profiles = {"Eco 30": {}}
    past = [
        {"profile_name": "Eco 30", "start_time": "2026-01-01"},
        {"profile_name": "Eco 30", "start_time": "2026-01-02"},
    ]
    store = _list_profiles_store(profiles, past)
    p = _profile_of(store.list_profiles(), "Eco 30")
    assert p["avg_cost"] is None
    assert p["total_cost"] is None


def test_b2_list_profiles_cost_ignores_none_entries():
    profiles = {"Quick": {}}
    past = [
        {"profile_name": "Quick", "cost": 0.10, "start_time": "a"},
        {"profile_name": "Quick", "start_time": "b"},  # no cost
        {"profile_name": "Quick", "cost": 0.30, "start_time": "c"},
    ]
    store = _list_profiles_store(profiles, past)
    p = _profile_of(store.list_profiles(), "Quick")
    assert p["total_cost"] == pytest.approx(0.40)
    assert p["avg_cost"] == pytest.approx(0.20)


def test_b2_list_profiles_cost_rounded_to_4_decimals():
    profiles = {"P": {}}
    past = [
        {"profile_name": "P", "cost": 0.1},
        {"profile_name": "P", "cost": 0.1},
        {"profile_name": "P", "cost": 0.1},
    ]
    store = _list_profiles_store(profiles, past)
    p = _profile_of(store.list_profiles(), "P")
    # 0.3 / 3 = 0.1 (rounded, not 0.0999999...)
    assert p["avg_cost"] == 0.1
    assert p["total_cost"] == pytest.approx(0.3)


def test_b2_list_profiles_keeps_existing_fields():
    profiles = {"Cotton 60": {"avg_duration": 3600, "min_duration": 3000}}
    past = [{"profile_name": "Cotton 60", "cost": 0.5, "start_time": "x"}]
    store = _list_profiles_store(profiles, past)
    p = _profile_of(store.list_profiles(), "Cotton 60")
    # New keys are additive.
    for field in ("name", "avg_duration", "cycle_count", "avg_cost", "total_cost"):
        assert field in p


# ---------------------------------------------------------------------------
# B3 — finish-notification template variables
# ---------------------------------------------------------------------------


def test_b3_vs_typical_longer():
    assert WashDataManager._format_vs_typical(4320.0, 3600.0) == "20% longer than usual"


def test_b3_vs_typical_shorter():
    assert WashDataManager._format_vs_typical(2880.0, 3600.0) == "20% shorter than usual"


def test_b3_vs_typical_exact_1pct_longer():
    # 36 / 3600 = 1.0% -> rounds to 1
    assert WashDataManager._format_vs_typical(3636.0, 3600.0) == "1% longer than usual"


def test_b3_vs_typical_within_1pct_is_empty():
    # 10 / 3600 = 0.28% -> rounds to 0 -> ""
    assert WashDataManager._format_vs_typical(3610.0, 3600.0) == ""


def test_b3_vs_typical_no_median_is_empty():
    assert WashDataManager._format_vs_typical(3600.0, None) == ""
    assert WashDataManager._format_vs_typical(3600.0, 0) == ""
    assert WashDataManager._format_vs_typical(3600.0, -5) == ""


def test_b3_vs_typical_garbage_is_empty():
    assert WashDataManager._format_vs_typical("nope", 3600.0) == ""


def test_b3_cycle_count_reflects_store():
    # The finish notification passes self.cycle_count as `cycle_count`.
    m = MagicMock()
    m.profile_store.get_past_cycles.return_value = [1, 2, 3, 4]
    assert WashDataManager.cycle_count.fget(m) == 4


def test_b3_time_finished_format(monkeypatch):
    from custom_components.ha_washdata import manager as mgr_mod

    fixed = datetime(2026, 7, 10, 9, 5, tzinfo=timezone.utc)
    monkeypatch.setattr(mgr_mod.dt_util, "now", lambda: fixed)
    # Mirrors the exact expression used in the finish-notification block:
    # zero-padded local wall-clock "HH:MM".
    assert mgr_mod.dt_util.now().strftime("%H:%M") == "09:05"


# ---------------------------------------------------------------------------
# B4 — peak-rate awareness tip
# ---------------------------------------------------------------------------


def test_b4_tip_appended_when_price_at_or_above_threshold():
    m = _peak_rate_manager()
    tip = m._peak_rate_tip({CONF_PEAK_RATE_THRESHOLD: 0.30}, 0.35)
    assert tip == DEFAULT_PEAK_RATE_MESSAGE.format(price="0.350")
    assert tip == "Running at peak rate (0.350/kWh)."


def test_b4_tip_at_exact_threshold():
    m = _peak_rate_manager()
    tip = m._peak_rate_tip({CONF_PEAK_RATE_THRESHOLD: 0.25}, 0.25)
    assert "0.250/kWh" in tip


def test_b4_no_tip_when_price_below_threshold():
    m = _peak_rate_manager()
    assert m._peak_rate_tip({CONF_PEAK_RATE_THRESHOLD: 0.40}, 0.30) == ""


def test_b4_no_tip_when_threshold_unset():
    m = _peak_rate_manager()
    assert m._peak_rate_tip({}, 0.99) == ""
    assert m._peak_rate_tip({CONF_PEAK_RATE_THRESHOLD: None}, 0.99) == ""
    assert m._peak_rate_tip({CONF_PEAK_RATE_THRESHOLD: ""}, 0.99) == ""


def test_b4_no_tip_when_no_price():
    m = _peak_rate_manager()
    assert m._peak_rate_tip({CONF_PEAK_RATE_THRESHOLD: 0.10}, None) == ""


def test_b4_no_tip_when_threshold_non_positive_or_bad():
    m = _peak_rate_manager()
    assert m._peak_rate_tip({CONF_PEAK_RATE_THRESHOLD: 0}, 0.50) == ""
    assert m._peak_rate_tip({CONF_PEAK_RATE_THRESHOLD: -1}, 0.50) == ""
    assert m._peak_rate_tip({CONF_PEAK_RATE_THRESHOLD: "abc"}, 0.50) == ""


def test_b4_custom_message_override():
    m = _peak_rate_manager()
    opts = {CONF_PEAK_RATE_THRESHOLD: 0.10, CONF_PEAK_RATE_MESSAGE: "Expensive: {price}"}
    assert m._peak_rate_tip(opts, 0.50) == "Expensive: 0.500"
