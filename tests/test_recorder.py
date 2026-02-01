
"""Tests for the CycleRecorder class."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone, timedelta

from custom_components.ha_washdata.recorder import CycleRecorder
from custom_components.ha_washdata.const import DOMAIN

# Helper to mock dt_util
def mock_dt_util(mock_dt):
    mock_dt.now.return_value = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    mock_dt.utcnow.return_value = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    def parse_dt(dt_str):
        return datetime.fromisoformat(dt_str)
    mock_dt.parse_datetime.side_effect = parse_dt
    
    def utc_from_ts(ts):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    mock_dt.utc_from_timestamp.side_effect = utc_from_ts
    return mock_dt

@pytest.mark.asyncio
async def test_recorder_initialization(mock_hass):
    """Test recorder initialization."""
    with patch("custom_components.ha_washdata.recorder.dt_util") as mock_dt:
        mock_dt_util(mock_dt)
        recorder = CycleRecorder(mock_hass, "test_entry")
        assert not recorder.is_recording
        assert recorder.last_run is None

@pytest.mark.asyncio
async def test_start_stop_recording(mock_hass):
    """Test starting and stopping a recording."""
    with patch("custom_components.ha_washdata.recorder.dt_util") as mock_dt:
        mock_dt_util(mock_dt)
        recorder = CycleRecorder(mock_hass, "test_entry")
        
        # Mock store save
        recorder._store.async_save = AsyncMock()
        
        await recorder.start_recording()
        assert recorder.is_recording
        assert recorder._start_time is not None
        assert len(recorder._buffer) == 0
        
        # Add some data
        recorder.process_reading(10.5)
        recorder.process_reading(20.0)
        assert len(recorder._buffer) == 2
        
        # Stop
        result = await recorder.stop_recording()
        assert not recorder.is_recording
        
        assert len(result["data"]) == 2

@pytest.mark.asyncio
async def test_persistence_loading(mock_hass):
    """Test loading state from storage."""
    # Patch RecorderStore (not Store) since CycleRecorder uses RecorderStore
    with patch("custom_components.ha_washdata.recorder.RecorderStore") as mock_store_cls:
        mock_store_instance = mock_store_cls.return_value
        
        mock_data = {
            "is_recording": True,
            "start_time": "2023-01-01T12:00:00+00:00",
            "buffer": [("2023-01-01T12:00:01+00:00", 100.0)],
            "last_run": {"some": "data"}
        }
        mock_store_instance.async_load = AsyncMock(return_value=mock_data)
        
        with patch("custom_components.ha_washdata.recorder.dt_util") as mock_dt:
            mock_dt_util(mock_dt)
            recorder = CycleRecorder(mock_hass, "test_entry")
            await recorder.async_load()
            
            # Note: is_recording flag is restored from storage
            assert recorder._is_recording == True
            assert recorder.last_run == {"some": "data"}

@pytest.mark.asyncio
async def test_trim_suggestions(mock_hass):
    """Test trim suggestion logic."""
    with patch("custom_components.ha_washdata.recorder.dt_util") as mock_dt:
        dt_base = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_dt.now.return_value = dt_base
        mock_dt.parse_datetime.side_effect = datetime.fromisoformat
        
        recorder = CycleRecorder(mock_hass, "test_entry")
        
        # Create fake data
        data = []
        # Head noise (10s of 0W)
        for i in range(10):
            t = dt_base + timedelta(seconds=i)
            data.append((t.isoformat(), 0.0))
            
        # Active (10s of 100W)
        for i in range(10, 20):
            t = dt_base + timedelta(seconds=i)
            data.append((t.isoformat(), 100.0))
            
        # Tail noise (10s of 0W)
        for i in range(20, 30):
            t = dt_base + timedelta(seconds=i)
            data.append((t.isoformat(), 0.0))
            
        head, tail, score = recorder.get_trim_suggestions(data)
        
        assert 9.0 <= head <= 11.0
        assert 9.0 <= tail <= 11.0

