"""Tests for HA restart gap detection and storage.

When HA restarts while a cycle is active, _attempt_state_restoration records
the dark period as a restart gap.  The gap is stored on cycle_data at cycle
end so the panel can shade the hole in the power trace.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.const import CONF_MIN_POWER, CONF_POWER_SENSOR


@pytest.fixture
def mock_hass() -> Any:
    hass = MagicMock()
    hass.data = {}
    hass.services.async_call = AsyncMock()
    hass.bus.async_fire = MagicMock()
    hass.async_create_task = MagicMock(
        side_effect=lambda coro: getattr(coro, "close", lambda: None)()
    )
    hass.components.persistent_notification.async_create = MagicMock()
    hass.config_entries.async_get_entry = MagicMock()
    return hass


@pytest.fixture
def mock_entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Washer"
    entry.options = {
        CONF_MIN_POWER: 2.0,
        CONF_POWER_SENSOR: "sensor.test_power",
        "completion_min_seconds": 600,
        "notify_finish_services": [],
    }
    return entry


@pytest.fixture
def manager(mock_hass: Any, mock_entry: Any) -> WashDataManager:
    mock_hass.config_entries.async_get_entry.return_value = mock_entry
    with (
        patch("custom_components.ha_washdata.manager.ProfileStore"),
        patch("custom_components.ha_washdata.manager.CycleDetector"),
    ):
        mgr = WashDataManager(mock_hass, mock_entry)
        mgr.profile_store.get_suggestions = MagicMock(return_value={})
        mgr.profile_store._data = {"profiles": {}}
        return mgr


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class TestRestartGapsInit:
    def test_restart_gaps_starts_empty(self, manager: WashDataManager) -> None:
        assert manager._restart_gaps == []

    def test_restart_gaps_is_list(self, manager: WashDataManager) -> None:
        assert isinstance(manager._restart_gaps, list)


class TestRestartGapStoredAtCycleEnd:
    """_async_process_cycle_end must store and clear _restart_gaps."""

    def _build_cycle_data(self, now: datetime) -> dict[str, Any]:
        return {
            "id": "test-cycle-1",
            "power_data": [],
            "duration": 3600,
            "status": "completed",
            "start_time": _iso(now - timedelta(hours=1)),
            "end_time": _iso(now),
        }

    def _stub_manager(self, mgr: WashDataManager) -> None:
        """Replace heavy async/sync calls with no-ops so tests focus on gap logic."""
        mgr.profile_store.async_add_cycle = AsyncMock()
        mgr.profile_store.async_rebuild_envelope = AsyncMock()
        mgr.profile_store.async_match_profile = AsyncMock(return_value=MagicMock(best_profile=None, confidence=0.0, ranking=[]))
        mgr.profile_store.get_past_cycles = MagicMock(return_value=[])
        mgr.profile_store.get_profiles = MagicMock(return_value={})
        mgr.profile_store.detect_cycle_artifacts = MagicMock(return_value=[])
        mgr.profile_store.compute_envelope_conformance = MagicMock(return_value=None)
        mgr.profile_store.confirm_match_ranking_snapshots = MagicMock()
        mgr.learning_manager = MagicMock()
        mgr.learning_manager.process_cycle_end = MagicMock()
        # Disable auto-label path so async_match_profile is not called unconditionally
        mgr._auto_label_confidence = 0

    @pytest.mark.asyncio
    async def test_restart_gap_stored_in_cycle_data(self, manager: WashDataManager) -> None:
        """Restart gaps are copied into cycle_data["restart_gaps"] at cycle end."""
        now = _now_utc()
        gap = {
            "start_ts": _iso(now - timedelta(minutes=5)),
            "end_ts": _iso(now - timedelta(minutes=3)),
            "gap_seconds": 120.0,
            "profile": "Cotton 60",
            "match_confidence": 0.82,
        }
        manager._restart_gaps.append(gap)
        self._stub_manager(manager)
        cycle_data = self._build_cycle_data(now)

        with patch.object(manager, "_run_final_match_from_cycle_data", AsyncMock()):
            with patch.object(manager, "_compute_cycle_quality_score", MagicMock()):
                with patch.object(manager, "_clear_live_progress_notification", MagicMock()):
                    with patch.object(manager, "_reset_live_notification_state", MagicMock()):
                        with patch.object(manager, "_resolve_energy_price", MagicMock(return_value=None)):
                            with patch("custom_components.ha_washdata.manager.dt_util") as mock_dt:
                                mock_dt.now.return_value = now
                                await manager._async_process_cycle_end(cycle_data)

        assert "restart_gaps" in cycle_data
        assert len(cycle_data["restart_gaps"]) == 1
        stored = cycle_data["restart_gaps"][0]
        assert stored["gap_seconds"] == 120.0
        assert stored["profile"] == "Cotton 60"

    @pytest.mark.asyncio
    async def test_restart_gaps_cleared_after_cycle_end(self, manager: WashDataManager) -> None:
        """_restart_gaps is cleared after being stored so the next cycle starts fresh."""
        now = _now_utc()
        manager._restart_gaps.append({
            "start_ts": _iso(now - timedelta(minutes=2)),
            "end_ts": _iso(now - timedelta(minutes=1)),
            "gap_seconds": 60.0,
            "profile": None,
            "match_confidence": None,
        })
        self._stub_manager(manager)
        cycle_data = self._build_cycle_data(now)

        with patch.object(manager, "_run_final_match_from_cycle_data", AsyncMock()):
            with patch.object(manager, "_compute_cycle_quality_score", MagicMock()):
                with patch.object(manager, "_clear_live_progress_notification", MagicMock()):
                    with patch.object(manager, "_reset_live_notification_state", MagicMock()):
                        with patch.object(manager, "_resolve_energy_price", MagicMock(return_value=None)):
                            with patch("custom_components.ha_washdata.manager.dt_util") as mock_dt:
                                mock_dt.now.return_value = now
                                await manager._async_process_cycle_end(cycle_data)

        assert manager._restart_gaps == []

    @pytest.mark.asyncio
    async def test_no_restart_gaps_key_when_none(self, manager: WashDataManager) -> None:
        """cycle_data gets no restart_gaps key when no gaps occurred."""
        now = _now_utc()
        assert manager._restart_gaps == []
        self._stub_manager(manager)
        cycle_data = self._build_cycle_data(now)

        with patch.object(manager, "_run_final_match_from_cycle_data", AsyncMock()):
            with patch.object(manager, "_compute_cycle_quality_score", MagicMock()):
                with patch.object(manager, "_clear_live_progress_notification", MagicMock()):
                    with patch.object(manager, "_reset_live_notification_state", MagicMock()):
                        with patch.object(manager, "_resolve_energy_price", MagicMock(return_value=None)):
                            with patch("custom_components.ha_washdata.manager.dt_util") as mock_dt:
                                mock_dt.now.return_value = now
                                await manager._async_process_cycle_end(cycle_data)

        assert "restart_gaps" not in cycle_data

    @pytest.mark.asyncio
    async def test_ranking_backfill_uses_captured_token_not_live_field(
        self, manager: WashDataManager
    ) -> None:
        """The snapshot back-fill must use the captured cycle_token, not the live
        _ranking_snapshot_cycle_id, which can roll to a newly-started cycle during
        the awaits in _async_process_cycle_end (else cycle A's snapshots go
        unlabelled and cycle B's get mislabelled)."""
        now = _now_utc()
        self._stub_manager(manager)
        cycle_data = self._build_cycle_data(now)
        cycle_data["profile_name"] = "Cotton 60"  # so the confirm block runs
        # Simulate cycle B having started during processing: the live field has
        # already rolled over to B's id, but this call is finishing cycle A.
        manager._ranking_snapshot_cycle_id = "cycle_B"

        with patch.object(manager, "_run_final_match_from_cycle_data", AsyncMock()):
            with patch.object(manager, "_compute_cycle_quality_score", MagicMock()):
                with patch.object(manager, "_clear_live_progress_notification", MagicMock()):
                    with patch.object(manager, "_reset_live_notification_state", MagicMock()):
                        with patch.object(manager, "_resolve_energy_price", MagicMock(return_value=None)):
                            with patch("custom_components.ha_washdata.manager.dt_util") as mock_dt:
                                mock_dt.now.return_value = now
                                await manager._async_process_cycle_end(
                                    cycle_data, cycle_token="cycle_A"
                                )

        manager.profile_store.confirm_match_ranking_snapshots.assert_called_once()
        _, kwargs = manager.profile_store.confirm_match_ranking_snapshots.call_args
        assert kwargs.get("cycle_id") == "cycle_A", (
            "back-fill must use the captured token, not the rolled-over live id 'cycle_B'"
        )


class TestRestartGapStructure:
    """Validate the structure of a restart gap record."""

    def test_gap_has_required_fields(self) -> None:
        now = _now_utc()
        gap = {
            "start_ts": _iso(now - timedelta(minutes=10)),
            "end_ts": _iso(now - timedelta(minutes=8)),
            "gap_seconds": 120.0,
            "profile": "Eco 40",
            "match_confidence": 0.75,
        }
        for key in ("start_ts", "end_ts", "gap_seconds", "profile", "match_confidence"):
            assert key in gap

    def test_gap_with_no_match_confidence(self) -> None:
        """A gap recorded before any profile is matched has None confidence."""
        now = _now_utc()
        gap = {
            "start_ts": _iso(now - timedelta(minutes=5)),
            "end_ts": _iso(now - timedelta(minutes=3)),
            "gap_seconds": 120.0,
            "profile": None,
            "match_confidence": None,
        }
        assert gap["profile"] is None
        assert gap["match_confidence"] is None

    def test_gap_duration_matches_timestamps(self) -> None:
        now = _now_utc()
        start = now - timedelta(seconds=180)
        end = now
        gap_seconds = (end - start).total_seconds()
        gap = {
            "start_ts": _iso(start),
            "end_ts": _iso(end),
            "gap_seconds": round(gap_seconds, 1),
            "profile": None,
            "match_confidence": None,
        }
        assert abs(gap["gap_seconds"] - 180.0) < 1.0
