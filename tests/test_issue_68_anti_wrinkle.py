"""Tests for anti-wrinkle state handling (issue #68)."""

from datetime import datetime, timezone, timedelta
from unittest.mock import Mock

import pytest

from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
    STATE_OFF,
    STATE_STARTING,
    STATE_RUNNING,
    STATE_FINISHED,
    STATE_ANTI_WRINKLE,
    DEVICE_TYPE_DRYER,
    DEVICE_TYPE_WASHER_DRYER,
    DEVICE_TYPE_WASHING_MACHINE,
)


def dt(seconds_offset: float) -> datetime:
    """Create a datetime with given offset from test base time."""
    return datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=seconds_offset)


def flush_buffer(detector: CycleDetector, start_t_offset: float, num_readings: int = 80) -> None:
    """Flush detector state machine by sending low readings at 1s intervals."""
    for i in range(1, num_readings + 1):
        detector.process_reading(0.0, dt(start_t_offset + i))


@pytest.fixture
def mock_callbacks() -> dict[str, Mock]:
    """Create mock callbacks."""
    return {
        "on_state_change": Mock(),
        "on_cycle_end": Mock(),
    }


@pytest.fixture
def dryer_config_no_anti_wrinkle() -> CycleDetectorConfig:
    """Create dryer config without anti-wrinkle."""
    return CycleDetectorConfig(
        min_power=5.0,
        off_delay=60,
        device_type=DEVICE_TYPE_DRYER,
        anti_wrinkle_enabled=False,
    )


@pytest.fixture
def dryer_config_with_anti_wrinkle() -> CycleDetectorConfig:
    """Create dryer config with anti-wrinkle enabled."""
    return CycleDetectorConfig(
        min_power=5.0,
        off_delay=60,
        device_type=DEVICE_TYPE_DRYER,
        anti_wrinkle_enabled=True,
        anti_wrinkle_max_power=400.0,
        anti_wrinkle_max_duration=60.0,
        anti_wrinkle_exit_power=0.8,
    )


def test_anti_wrinkle_disabled_default_finish(
    dryer_config_no_anti_wrinkle: CycleDetectorConfig,
    mock_callbacks: dict[str, Mock],
) -> None:
    """Test that when anti-wrinkle is disabled, cycle finishes to FINISHED (not ANTI_WRINKLE)."""
    detector = CycleDetector(
        config=dryer_config_no_anti_wrinkle,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
    )

    # Start and run cycle
    detector.process_reading(500.0, dt(0))
    assert detector.state == STATE_STARTING

    detector.process_reading(500.0, dt(10))
    assert detector.state == STATE_RUNNING

    # Run for 25 mins
    for t in range(10, 1500, 10):
        detector.process_reading(500.0, dt(t))

    # Low power for off_delay -> ENDING
    detector.process_reading(1.0, dt(1501))
    detector.process_reading(1.0, dt(1540))
    flush_buffer(detector, 1540, num_readings=65)

    # Should end cycle and transition to FINISHED (not ANTI_WRINKLE when disabled)
    assert detector.state == STATE_FINISHED
    mock_callbacks["on_cycle_end"].assert_called_once()


def test_anti_wrinkle_cycle_completion_transition(
    dryer_config_with_anti_wrinkle: CycleDetectorConfig,
    mock_callbacks: dict[str, Mock],
) -> None:
    """Test that completed cycle transitions to ANTI_WRINKLE when enabled for dryers."""
    detector = CycleDetector(
        config=dryer_config_with_anti_wrinkle,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
    )

    # Start and run cycle
    detector.process_reading(500.0, dt(0))
    assert detector.state == STATE_STARTING

    detector.process_reading(500.0, dt(10))
    assert detector.state == STATE_RUNNING

    # Run for 25 mins
    for t in range(10, 1500, 10):
        detector.process_reading(500.0, dt(t))

    # Low power for off_delay -> ENDING
    detector.process_reading(1.0, dt(1501))
    detector.process_reading(1.0, dt(1540))
    # Send 65 low-power readings
    flush_buffer(detector, 1540, num_readings=65)

    # Should end cycle and transition to ANTI_WRINKLE (not FINISHED) when enabled
    assert detector.state == STATE_ANTI_WRINKLE
    # But on_cycle_end should still be called (cycle is considered complete)
    mock_callbacks["on_cycle_end"].assert_called_once()
    cycle_data = mock_callbacks["on_cycle_end"].call_args[0][0]
    assert cycle_data["status"] == "completed"


def test_anti_wrinkle_shields_low_power_spike(
    dryer_config_with_anti_wrinkle: CycleDetectorConfig,
    mock_callbacks: dict[str, Mock],
) -> None:
    """Test that low-power bursts in ANTI_WRINKLE are shielded (no new STARTING)."""
    detector = CycleDetector(
        config=dryer_config_with_anti_wrinkle,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
    )

    # First, complete a cycle and enter ANTI_WRINKLE
    detector.process_reading(500.0, dt(0))
    detector.process_reading(500.0, dt(10))
    for t in range(10, 1500, 10):
        detector.process_reading(500.0, dt(t))
    
    detector.process_reading(1.0, dt(1501))
    detector.process_reading(1.0, dt(1540))
    # Send 35 low-power readings to reach ANTI_WRINKLE
    flush_buffer(detector, 1540, num_readings=35)
    
    assert detector.state == STATE_ANTI_WRINKLE
    
    # Reset both mocks to track state changes from this point
    mock_callbacks["on_state_change"].reset_mock()
    mock_callbacks["on_cycle_end"].reset_mock()
    
    # Now simulate anti-wrinkle rotation: 250W for 15s (within thresholds)
    detector.process_reading(250.0, dt(1641))  # Start spike
    assert detector.state == STATE_ANTI_WRINKLE  # Should remain in ANTI_WRINKLE
    
    detector.process_reading(280.0, dt(1646))
    assert detector.state == STATE_ANTI_WRINKLE
    
    detector.process_reading(20.0, dt(1656))  # End spike
    assert detector.state == STATE_ANTI_WRINKLE
    
    # Should NOT have transitioned to STARTING
    mock_callbacks["on_state_change"].assert_not_called()
    mock_callbacks["on_cycle_end"].assert_not_called()


def test_anti_wrinkle_exit_via_user_stop(
    dryer_config_with_anti_wrinkle: CycleDetectorConfig,
    mock_callbacks: dict[str, Mock],
) -> None:
    """Test that ANTI_WRINKLE can be exited via user_stop."""
    detector = CycleDetector(
        config=dryer_config_with_anti_wrinkle,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
    )

    # First, complete a cycle and enter ANTI_WRINKLE
    detector.process_reading(500.0, dt(0))
    detector.process_reading(500.0, dt(10))
    for t in range(10, 1500, 10):
        detector.process_reading(500.0, dt(t))
    
    detector.process_reading(1.0, dt(1501))
    detector.process_reading(1.0, dt(1540))
    flush_buffer(detector, 1540, num_readings=35)
    
    assert detector.state == STATE_ANTI_WRINKLE
    
    # Reset mocks
    mock_callbacks["on_state_change"].reset_mock()
    mock_callbacks["on_cycle_end"].reset_mock()
    
    # Simulate user stop
    detector.user_stop()
    
    # Should exit to OFF via user stop
    assert detector.state == STATE_OFF
    mock_callbacks["on_state_change"].assert_called_with(STATE_ANTI_WRINKLE, STATE_OFF)


def test_anti_wrinkle_not_for_washing_machine(mock_callbacks: dict[str, Mock]) -> None:
    """Test that anti-wrinkle is NOT enabled for washing machines."""
    config = CycleDetectorConfig(
        min_power=5.0,
        off_delay=60,
        device_type=DEVICE_TYPE_WASHING_MACHINE,
        anti_wrinkle_enabled=True,  # Even if enabled in config
        anti_wrinkle_max_power=400.0,
        anti_wrinkle_max_duration=60.0,
        anti_wrinkle_exit_power=0.8,
    )
    
    detector = CycleDetector(
        config=config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
    )

    # Complete a cycle
    detector.process_reading(500.0, dt(0))
    detector.process_reading(500.0, dt(10))
    for t in range(10, 1500, 10):
        detector.process_reading(500.0, dt(t))
    
    detector.process_reading(1.0, dt(1501))
    detector.process_reading(1.0, dt(1540))
    flush_buffer(detector, 1540, num_readings=65)
    
    # Should transition to FINISHED (not ANTI_WRINKLE) for washing machines
    assert detector.state == STATE_FINISHED


def test_anti_wrinkle_enabled_washer_dryer(mock_callbacks: dict[str, Mock]) -> None:
    """Test that anti-wrinkle works for washer-dryer combos too."""
    config = CycleDetectorConfig(
        min_power=5.0,
        off_delay=60,
        device_type=DEVICE_TYPE_WASHER_DRYER,  # Washer-dryer combo
        anti_wrinkle_enabled=True,
        anti_wrinkle_max_power=400.0,
        anti_wrinkle_max_duration=60.0,
        anti_wrinkle_exit_power=0.8,
    )
    
    detector = CycleDetector(
        config=config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
    )

    # Complete a cycle
    detector.process_reading(300.0, dt(0))
    detector.process_reading(300.0, dt(10))
    for t in range(10, 1500, 10):
        detector.process_reading(300.0, dt(t))
    
    detector.process_reading(1.0, dt(1501))
    detector.process_reading(1.0, dt(1540))
    flush_buffer(detector, 1540, num_readings=35)
    
    # Should transition to ANTI_WRINKLE for washer-dryer
    assert detector.state == STATE_ANTI_WRINKLE


def test_anti_wrinkle_interrupted_cycle_no_transition(
    dryer_config_with_anti_wrinkle: CycleDetectorConfig,
    mock_callbacks: dict[str, Mock],
) -> None:
    """Test that anti-wrinkle still works even for interrupted cycles."""
    detector = CycleDetector(
        config=dryer_config_with_anti_wrinkle,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
    )

    # Start cycle
    detector.process_reading(500.0, dt(0))
    detector.process_reading(500.0, dt(10))
    for t in range(10, 400, 10):
        detector.process_reading(500.0, dt(t))
    
    # User stop (ends cycle as completed, so anti-wrinkle applies)
    detector.user_stop()
    
    # Should end in ANTI_WRINKLE for dryers even when user-stopped (completed status)
    assert detector.state == STATE_ANTI_WRINKLE
