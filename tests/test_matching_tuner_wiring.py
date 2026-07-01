"""Manager glue for on-device matcher tuning: _tune_matching_config persists a
promoted override, leaves the store alone otherwise, and never raises. The real
grid search lives in test_matching_tuner.py (slow); here it is patched so this
stays in the fast suite."""
from __future__ import annotations

import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.manager import WashDataManager

_TUNER = "custom_components.ha_washdata.ml.matching_tuner.tune_matching_config"


def _fake_mgr(mock_hass):
    store = MagicMock()
    store.set_matching_config = AsyncMock()
    return types.SimpleNamespace(hass=mock_hass, profile_store=store, _logger=MagicMock())


async def test_persists_promoted_override(mock_hass):
    fake = _fake_mgr(mock_hass)
    promoted = {
        "promoted": True,
        "config": {"corr_weight": 0.5, "duration_weight": 0.15, "energy_weight": 0.15},
        "baseline_test_top1": 0.70,
        "tuned_test_top1": 0.82,
    }
    with patch(_TUNER, return_value=promoted):
        result = await WashDataManager._tune_matching_config(fake, [{"a": 1}, {"b": 2}])

    fake.profile_store.set_matching_config.assert_awaited_once()
    rec = fake.profile_store.set_matching_config.await_args.args[0]
    assert rec["config"] == promoted["config"]
    assert rec["cycle_count"] == 2
    assert rec["baseline_test_top1"] == 0.70 and rec["tuned_test_top1"] == 0.82
    assert "trained_at" in rec
    assert result["promoted"] is True


async def test_no_persist_when_not_promoted(mock_hass):
    fake = _fake_mgr(mock_hass)
    with patch(_TUNER, return_value={"promoted": False, "reason": "insufficient data"}):
        result = await WashDataManager._tune_matching_config(fake, [])
    fake.profile_store.set_matching_config.assert_not_awaited()
    assert result["promoted"] is False


async def test_swallows_tuner_exception(mock_hass):
    fake = _fake_mgr(mock_hass)
    with patch(_TUNER, side_effect=RuntimeError("boom")):
        result = await WashDataManager._tune_matching_config(fake, [{"a": 1}])
    assert result["promoted"] is False
    assert result["reason"] == "exception"
    fake.profile_store.set_matching_config.assert_not_awaited()
