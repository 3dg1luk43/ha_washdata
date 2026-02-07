
import pytest
from datetime import datetime, timezone
from tests.utils.synthesizer import CycleSynthesizer

def test_synthesizer_basic():
    synth = CycleSynthesizer()
    synth.add_phase(100.0, 60.0) # 100W for 60s
    synth.add_gap(30.0) # 0W for 30s
    
    readings = synth.generate(sample_interval=10.0)
    
    # Expected approx 6 points for first phase, 3 for gap
    assert len(readings) >= 9
    assert readings[0][1] == 100.0
    # Last point of first phase should be around 60s offset
    # First point of gap should be after 60s
    
    phase_points = [r for r in readings if r[1] == 100.0]
    gap_points = [r for r in readings if r[1] == 0.0]
    
    assert len(phase_points) >= 6
    assert len(gap_points) >= 3

def test_synthesizer_jitter_and_drop():
    synth = CycleSynthesizer()
    synth.add_phase(100.0, 1000.0)
    
    # Test drop rate
    readings = synth.generate(sample_interval=10.0, drop_rate=0.5)
    # Expected approx 100 points if no drops, so approx 50 with 0.5 drop rate
    assert 30 < len(readings) < 70
    
    # Test jitter
    readings = synth.generate(sample_interval=10.0, jitter=2.0)
    intervals = []
    for i in range(1, len(readings)):
        delta = (readings[i][0] - readings[i-1][0]).total_seconds()
        intervals.append(delta)
    
    assert any(i != 10.0 for i in intervals)
    assert all(8.0 <= i <= 12.0 for i in intervals)

def test_synthesizer_boot_spike():
    synth = CycleSynthesizer()
    synth.add_boot_spike(500.0, 5.0)
    synth.add_gap(10.0)
    
    readings = synth.generate(sample_interval=1.0)
    
    spikes = [r for r in readings if r[1] > 400.0]
    assert len(spikes) >= 5
    
def test_synthesizer_time_warp():
    synth = CycleSynthesizer()
    synth.add_phase(100.0, 100.0)
    
    # Normal duration
    readings_normal = synth.generate(sample_interval=1.0)
    dur_normal = (readings_normal[-1][0] - readings_normal[0][0]).total_seconds()
    
    # Warped duration
    readings_warped = synth.generate(sample_interval=1.0, time_warp=2.0)
    dur_warped = (readings_warped[-1][0] - readings_warped[0][0]).total_seconds()
    
    assert 190 < dur_warped < 210
    assert 90 < dur_normal < 110
