
import pytest
import random
from devtools.mqtt_mock_socket import CycleSynthesizer

def test_synthesizer_amplitude_scaling():
    """Test that amplitude scaling can be applied."""
    # This test will fail because amplitude_scaling is not implemented yet
    template = {
        "power_data": [
            [0, 100],
            [10, 100],
            [20, 0]
        ]
    }
    # We want a parameter for amplitude scaling
    syn = CycleSynthesizer(amplitude_scaling=0.2) # +/- 20%
    
    # Run multiple times to see if we get different peaks
    peaks = []
    for _ in range(100):
        readings = syn.synthesize(template)
        peaks.append(max(readings))
    
    assert min(peaks) < 100
    assert max(peaks) > 100
    assert all(80 <= p <= 120 for p in peaks)

def test_synthesizer_duration_scaling():
    """Test that total duration scaling is applied."""
    template = {
        "power_data": [
            [0, 100],
            [100, 100],
            [200, 0]
        ]
    }
    # Currently variability scales segments, but we want a more explicit 
    # overall duration scaling if possible, or just verify variability works.
    syn = CycleSynthesizer(variability=0.2)
    
    durations = []
    for _ in range(100):
        readings = syn.synthesize(template)
        durations.append(len(readings))
    
def test_synthesizer_early_low_value():
    """Test that low values can arrive earlier than expected."""
    template = {
        "power_data": [
            [0, 100],
            [100, 100],
            [101, 0] # End at 101s
        ]
    }
    # We want a parameter for early low value probability
    syn = CycleSynthesizer(early_low_prob=0.5) 
    
    # We want to see if readings drop to ~0 before the end
    early_drops = 0
    for _ in range(100):
        readings = syn.synthesize(template)
        # Template has 101s. Let's see if it drops before index 101
        # (Assuming 1s samples in synthesize output for this simple test)
        if any(p < 1.0 for p in readings[:100]):
            early_drops += 1
            
    assert early_drops > 0
