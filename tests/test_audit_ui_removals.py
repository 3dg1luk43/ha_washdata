"""Audit 2026-10-02 UI removals (0.5.8): what the panel stopped showing.

- ML Training tab: a fine-tuned model whose consumer is frozen off changes
  nothing on the device, so it is not reported as "learned" (audit ML-01/02/05/06/07).
- The ML-calibrated setting suggestions are gone end to end (audit SUGGEST-13/18).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from custom_components.ha_washdata import suggestion_engine, ws_api


async def test_ml_status_lists_only_capabilities_something_still_consumes():
    store = MagicMock()
    store.get_ml_model_versions = MagicMock(return_value={
        cap: {"trained_at": "2026-09-01T02:00:00+00:00", "spec": {"kind": "x"}, "new_auc": 0.9}
        for cap in ("end", "quality", "live_match", "remaining_time", "total_energy")
    })
    store.get_ml_training_history = MagicMock(return_value={})
    store.get_matching_config = MagicMock(return_value={})
    store.get_past_cycles = MagicMock(return_value=[])
    manager = MagicMock()
    manager.profile_store = store
    manager._last_ml_training_at = MagicMock(return_value=None)
    manager._ml_training_running = False
    entry = SimpleNamespace(entry_id="e", data={}, options={})
    conn = MagicMock()
    sent: dict = {}
    conn.send_result = MagicMock(side_effect=lambda _i, payload: sent.update(payload))
    with patch.object(ws_api, "_get_manager", return_value=manager), \
         patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_api.ws_get_ml_training_status.__wrapped__(
            MagicMock(), conn, {"id": 1, "entry_id": "e"}
        )
    assert set(sent["on_device_models"]) == {"total_energy"}


def test_the_ml_suggestion_path_is_gone():
    assert not hasattr(suggestion_engine, "MLSuggestionEngine")
    assert not hasattr(ws_api, "_build_settings_comparison")
    assert not hasattr(ws_api, "ENABLE_ML_SUGGESTIONS")
