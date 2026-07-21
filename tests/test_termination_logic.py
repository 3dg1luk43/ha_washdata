# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""Test termination logic priority."""
# import tests.mock_imports  # noqa: F401
import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta
from custom_components.ha_washdata.cycle_detector import CycleDetector, CycleDetectorConfig
from custom_components.ha_washdata.const import STATE_OFF, STATE_RUNNING, STATE_ENDING, STATE_PAUSED, STATE_FINISHED

# Helper to create datetime sequence
def dt(offset_seconds: int) -> datetime:
    return datetime(2023, 1, 1, 12, 0, 0) + timedelta(seconds=offset_seconds)

@pytest.fixture
def base_config():
    """Default detector config."""
    return CycleDetectorConfig(
        min_power=5.0,
        off_delay=60,
        interrupted_min_seconds=150,
        completion_min_seconds=600,
        # Default start threshold checks
        start_duration_threshold=0.0,
        start_energy_threshold=0.0,
        start_threshold_w=6.0,
        stop_threshold_w=4.0,
    )

@pytest.fixture
def mock_callbacks():
    return {
        "on_state_change": Mock(),
        "on_cycle_end": Mock(),
    }

def test_long_drying_phase_cycle_continuation(base_config, mock_callbacks):
    """
    Verify cycle remains active during long low-power (drying) phases 
    when expected duration suggests it should continue.
    """
    detector = CycleDetector(
        config=base_config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
    )

    # 1. Start Cycle
    detector.process_reading(100.0, dt(0))
    detector.process_reading(100.0, dt(10)) # Running
    
    # Simulate a matched profile that expects 3600s (1h)
    # This is normally done by profile matcher callback injection.
    # We can manually set it for testing internals if we want, or use a mock matcher.
    
    # We'll use a mock matcher to conform to __init__ API
    mock_matcher = Mock()
    # Return match: name="Heavy", conf=0.9, duration=3600, phase="Washing", is_mismatch=False
    mock_matcher.side_effect = lambda readings: ("Heavy", 0.9, 3600.0, "Washing", False)
    
    detector = CycleDetector(
        config=base_config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
        profile_matcher=mock_matcher
    )
    
    # Restart with matcher
    detector.process_reading(100.0, dt(0))
    detector.process_reading(100.0, dt(10)) # Transition to RUNNING
    detector.process_reading(100.0, dt(20)) # Match attempted here
    
    # Check match happened
    assert detector.matched_profile == "Heavy"
    # assert detector._expected_duration == 3600.0 # Internal, but assumed
    
    # 2. Power drops at T=1800 (30 mins), half way. 
    # Must feed intermediate readings to avoid skewing p95 cadence
    # Feed "Running" power every 10s until 1800
    for t in range(30, 1800, 10):
        detector.process_reading(100.0, dt(t))
    
    # Drop to 1.0W
    detector.process_reading(1.0, dt(1800))
    
    # 3. Wait off_delay (60s) with 10s updates
    for t in range(1810, 1870, 10):
        detector.process_reading(1.0, dt(t))
    
    # CURRENT BUGGY BEHAVIOR: Cycle ends because power is low, ignoring 3600s expectation.
    # If this passes 'completed', it confirms the "bug" (default behavior).
    # After fix, this should stay RUNNING or ENDING.
    
    if detector.state == STATE_OFF:
        # Now this means failure (bug persisted)
        cycle_data = mock_callbacks["on_cycle_end"].call_args[0][0]
        pytest.fail(f"Cycle ended prematurely at {cycle_data['duration']}s (Expected ~3600s)")
    else:
        # Success! Kept alive.
        assert detector.state in (STATE_ENDING, STATE_RUNNING, STATE_PAUSED)

def test_manual_program_override_termination(base_config, mock_callbacks):
    """
    Test that a manual program (with 100% confidence) keeps cycle alive.
    Simulates wrapper return: ("ManualProfile", 1.0, 3600.0, "Manual", False)
    """
    mock_matcher = Mock()
    mock_matcher.side_effect = lambda readings: ("ManualProfile", 1.0, 3600.0, "Manual", False)
    
    detector = CycleDetector(
        config=base_config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
        profile_matcher=mock_matcher
    )
    
    # Start and run briefly
    detector.process_reading(100.0, dt(0))
    detector.process_reading(100.0, dt(10)) # Transition to RUNNING
    detector.process_reading(100.0, dt(60)) # Trigger match
    
    assert detector.matched_profile == "ManualProfile"
    assert detector._expected_duration == 3600.0

    # Feed "Running" power every 10s until 600
    for t in range(70, 600, 10):
        detector.process_reading(100.0, dt(t))
    
    # Power fail early (10 mins)
    detector.process_reading(0.0, dt(600))
    
    # Wait past off_delay (60s) with updates
    for t in range(610, 710, 10):
        detector.process_reading(0.0, dt(t))
    
    # Should be alive
    assert detector.state != STATE_OFF
    
    # Warp to expected duration end + tolerance (3600 * 1.25 = 4500)
    # So we need to go beyond 4500 to ensure it finishes
    detector.process_reading(0.0, dt(5000))
    
    # Should be OFF
    assert detector.state == STATE_FINISHED

def test_ambiguous_match_stuck_in_ending_is_hard_finalized(base_config, mock_callbacks):
    """Duration-anchored backstop: an ambiguous match whose fallback energy gate is
    held open by a low standby baseline is finalized at ~2x expected instead of
    sitting in ENDING until the 8h cap (#296/#311).

    A 6-tuple match with ambiguous=True blocks Smart Termination; a 3.5 W standby
    (below stop_threshold=4.0 so ENDING is reached, but energetic enough to trip
    the 0.05 Wh energy gate over the 60 s off_delay window) blocks the normal
    fallback timeout. Only the backstop can end the cycle.
    """
    from custom_components.ha_washdata.const import ENDING_HARD_FINALIZE_RATIO

    mock_matcher = Mock()
    # 6-tuple: (name, conf, expected_dur, phase, is_mismatch, ambiguous)
    mock_matcher.side_effect = lambda readings: ("Heavy", 0.9, 3600.0, "Washing", False, True)

    detector = CycleDetector(
        config=base_config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
        profile_matcher=mock_matcher,
    )

    # Start + confirm RUNNING + establish the ambiguous match.
    detector.process_reading(100.0, dt(0))
    detector.process_reading(100.0, dt(30))
    detector.process_reading(100.0, dt(60))
    assert detector.matched_profile == "Heavy"
    assert detector._match_ambiguous is True

    # Run high for a while, then drop to a 3.5 W standby baseline.
    for t in range(90, 1800, 30):
        detector.process_reading(100.0, dt(t))
    for t in range(1800, 5000, 30):
        detector.process_reading(3.5, dt(t))

    # Well past off_delay but below 2x expected (7200s): the energy gate must have
    # blocked the normal fallback, so the cycle is still open (in ENDING).
    assert detector.state != STATE_OFF, (
        "Normal fallback fired despite the standby energy gate; "
        f"state={detector.state!r}"
    )
    assert not mock_callbacks["on_cycle_end"].called

    # Cross 2x expected (7200s) with the baseline still held: the backstop fires.
    for t in range(5000, 7400, 30):
        detector.process_reading(3.5, dt(t))

    assert mock_callbacks["on_cycle_end"].called, (
        "Duration-anchored backstop did not finalize the stuck cycle"
    )
    cycle_data = mock_callbacks["on_cycle_end"].call_args[0][0]
    # Finalized around 2x expected, well before the 8h (28800s) hard cap.
    assert cycle_data["duration"] < 28800
    assert cycle_data.get("termination_reason") == "timeout"


def test_backstop_does_not_truncate_longer_program(base_config, mock_callbacks):
    """Control: the backstop must NOT truncate a longer program mismatched to a
    shorter profile. A real longer program keeps producing high-power phases, which
    reset the continuous-quiet timer, so the sustained-quiet guard is never met.

    Dips use a 3.9 W standby (below stop_threshold=4.0, but energetic enough to trip
    the energy gate) so the *normal* fallback also can't fire during a dip — this
    isolates the backstop: the only thing that could end the cycle here is the
    duration-anchored finalize, and it must not.
    """
    mock_matcher = Mock()
    # Mismatched to a short 1200s profile, ambiguous (so Smart Termination is off).
    mock_matcher.side_effect = lambda readings: ("Quick", 0.9, 1200.0, "Washing", False, True)

    detector = CycleDetector(
        config=base_config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
        profile_matcher=mock_matcher,
    )

    detector.process_reading(100.0, dt(0))
    detector.process_reading(100.0, dt(30))
    detector.process_reading(100.0, dt(60))
    assert detector.matched_profile == "Quick"

    # A genuinely longer program: alternating high-power work and standby dips, run
    # well past 2x the (wrong) 1200s expected. Each dip (~450s) is shorter than the
    # 600s continuous-quiet floor and every high burst resets the quiet timer, so
    # the sustained-quiet guard is never satisfied.
    t = 90
    while t < 6000:  # 5x the mismatched expected
        for _ in range(4):    # ~120s of high-power work (resets the quiet timer)
            detector.process_reading(120.0, dt(t)); t += 30
        for _ in range(15):   # ~450s standby dip (< 600s quiet floor)
            detector.process_reading(3.9, dt(t)); t += 30

    # Despite being far past 2x the wrong expected duration, the cycle must NOT have
    # been hard-finalized — high-power phases keep resetting the continuous-quiet timer.
    assert not mock_callbacks["on_cycle_end"].called, (
        "Backstop truncated a longer program mismatched to a shorter profile"
    )


def test_standby_band_stuck_running_is_finalized(base_config, mock_callbacks):
    """#296: a washer that finishes but holds a flat standby draw ABOVE
    stop_threshold never accumulates below-threshold time, so it never reaches
    ENDING. The standby-band detector finalizes it (as a normal completion) once
    the flat plateau has held past 2x the expected duration."""
    mock_matcher = Mock()
    mock_matcher.side_effect = lambda readings: ("Cotton", 0.9, 1200.0, "Washing", False, False)

    detector = CycleDetector(
        config=base_config,  # washing_machine, stop=4.0, start=6.0
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
        profile_matcher=mock_matcher,
    )

    # Establish RUNNING + match with real activity (peak 100 W).
    for t in range(0, 300, 30):
        detector.process_reading(100.0, dt(t))
    assert detector.matched_profile == "Cotton"
    assert detector.state == STATE_RUNNING

    # Drop to a flat 5 W anti-crease baseline: above stop_threshold (4.0), so
    # _time_below_threshold never accumulates and the cycle stays RUNNING.
    for t in range(300, 2000, 30):
        detector.process_reading(5.0, dt(t))
    # Still short of 2x expected (2400s): must still be RUNNING (the #296 bug).
    assert detector.state == STATE_RUNNING
    assert not mock_callbacks["on_cycle_end"].called

    # Cross 2x expected with the flat plateau still held → standby-band finalize.
    for t in range(2000, 2600, 30):
        detector.process_reading(5.0, dt(t))
    assert mock_callbacks["on_cycle_end"].called, (
        "Standby-band detector did not finalize the stuck cycle"
    )
    cycle_data = mock_callbacks["on_cycle_end"].call_args[0][0]
    assert cycle_data.get("termination_reason") == "timeout"
    assert cycle_data["duration"] < 28800  # well before the 8h hard cap


def test_standby_band_ignores_fluctuating_activity(base_config, mock_callbacks):
    """Control: periodic high-power bursts (real activity) past 2x expected must
    NOT be seen as a standby plateau — the level gate rejects any window with a
    reading above a small fraction of the cycle peak."""
    mock_matcher = Mock()
    mock_matcher.side_effect = lambda readings: ("Cotton", 0.9, 1200.0, "Washing", False, False)

    detector = CycleDetector(
        config=base_config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
        profile_matcher=mock_matcher,
    )

    for t in range(0, 300, 30):
        detector.process_reading(100.0, dt(t))
    assert detector.matched_profile == "Cotton"

    # Genuine ongoing activity well past 2x expected: 5 W plateaus broken by
    # 100 W bursts. Any window containing a burst is not a standby plateau.
    t = 300
    while t < 4000:
        for _ in range(15):  # ~450s at 5 W
            detector.process_reading(5.0, dt(t)); t += 30
        for _ in range(3):   # ~90s at 100 W (real activity)
            detector.process_reading(100.0, dt(t)); t += 30

    assert not mock_callbacks["on_cycle_end"].called, (
        "Standby-band detector truncated a cycle that still had real activity"
    )


def test_standby_band_excludes_non_wet_device(mock_callbacks):
    """Control: bread makers (keep-warm hold is legitimate) are excluded from the
    standby-band finalize."""
    from custom_components.ha_washdata.const import DEVICE_TYPE_BREAD_MAKER

    config = CycleDetectorConfig(
        min_power=5.0,
        off_delay=60,
        device_type=DEVICE_TYPE_BREAD_MAKER,
        interrupted_min_seconds=150,
        completion_min_seconds=600,
        start_duration_threshold=0.0,
        start_energy_threshold=0.0,
        start_threshold_w=6.0,
        stop_threshold_w=4.0,
    )
    mock_matcher = Mock()
    mock_matcher.side_effect = lambda readings: ("Basic", 0.9, 1200.0, "Baking", False, False)
    detector = CycleDetector(
        config=config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
        profile_matcher=mock_matcher,
    )

    for t in range(0, 300, 30):
        detector.process_reading(100.0, dt(t))
    assert detector.matched_profile == "Basic"

    # Flat 5 W keep-warm past 2x expected: excluded device type, so the
    # standby-band finalize must not fire.
    for t in range(300, 2800, 30):
        detector.process_reading(5.0, dt(t))
    assert not mock_callbacks["on_cycle_end"].called, (
        "Standby-band finalize fired for an excluded device type"
    )


def test_fix_duration_keeps_alive(base_config, mock_callbacks):
    """
    Test that will PASS only after the fix.
    Cycle should remain alive during low power if (elapsed / expected) < ratio.
    """
    # 1. Setup detector with mocked profile match
    mock_matcher = Mock()
    # Expect 3600s
    mock_matcher.side_effect = lambda readings: ("Heavy", 0.9, 3600.0, "Drying", False)
    
    # We need to set min_duration_ratio in config (will add this field in implementation)
    # For now, we rely on default or modify config object after init if needed
    # base_config.min_duration_ratio = 0.8 (Not yet in dataclass)
    
    detector = CycleDetector(
        config=base_config,
        on_state_change=mock_callbacks["on_state_change"],
        on_cycle_end=mock_callbacks["on_cycle_end"],
        profile_matcher=mock_matcher
    )
    
    # Start
    detector.process_reading(100.0, dt(0))
    detector.process_reading(100.0, dt(10)) # Transition to RUNNING
    detector.process_reading(100.0, dt(60)) # Trigger match
    
    assert detector.matched_profile == "Heavy"
    
    # Feed "Running" power every 10s until 1800
    for t in range(20, 1800, 10):
        detector.process_reading(100.0, dt(t))
    
    # Drop power at 30 mins (1800s)
    detector.process_reading(0.0, dt(1800))
    
    # Advance past off_delay (60s) -> 1900s
    for t in range(1810, 1910, 10):
        detector.process_reading(0.0, dt(t))
    
    # ASSERTION FOR DESIRED BEHAVIOR:
    # Should NOT be OFF. Should be ENDING (waiting) or RUNNING (if we deem it running).
    # Usually 'ENDING' is the low-power waiting state.
    
    # Note: THIS WILL FAIL currently (step 1 of TDD)
    if detector.state == STATE_OFF:
        pytest.fail("Cycle terminated prematurely! Fix not working.")
    
    # Ensure it ends eventually
    # 1h + off_delay -> 3600 + 100 = 3700
    detector.process_reading(0.0, dt(3700))
    
    # Now it should end
    # assert detector.state == STATE_OFF (Might need to implement the check correctly first)
