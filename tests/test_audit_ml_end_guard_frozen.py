"""Audit 2026-10-02 ML-05: the ML end-guard is frozen off, even with ML enabled.

Replayed on 292 real cycles it prevented no premature stop and raised the washer
median end lag 12.2 -> 17.5 min. The manager's provider must answer None (the
detector then keeps its power/energy behaviour) whatever the device's ML option.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from custom_components.ha_washdata import const as C
from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.ml.engine import CONF_ENABLE_ML_MODELS


def test_the_manager_end_guard_provider_is_inert_with_ml_enabled() -> None:
    entry = MagicMock()
    entry.entry_id = "e"
    entry.title = "W"
    entry.options = {"power_sensor": "sensor.p", CONF_ENABLE_ML_MODELS: True}
    entry.data = {"power_sensor": "sensor.p"}
    hass = MagicMock()
    hass.data = {}
    with patch("custom_components.ha_washdata.manager.ProfileStore"):
        mgr = WashDataManager(hass, entry)
    mgr._current_program = "Cotton"
    mgr.profile_store.get_profiles.return_value = {"Cotton": {"avg_duration": 3600}}
    with patch("custom_components.ha_washdata.ml.engine.resolve_scorer",
               return_value=(lambda _f: 0.1, None)) as scorer:
        assert mgr._ml_end_confidence([(0.0, 500.0), (3500.0, 1.0)], 3600.0) is None
    scorer.assert_not_called()
    assert C.ENABLE_ML_END_GUARD is False
