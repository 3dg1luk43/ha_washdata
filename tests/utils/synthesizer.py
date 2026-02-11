
import random
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

class CycleSynthesizer:
    """Utility to generate synthetic power usage cycles for testing."""

    def __init__(self):
        self.segments = [] # List of (duration, power, noise)

    def add_phase(self, power: float, duration: float, noise: float = 0.0):
        """Add a constant power phase."""
        self.segments.append({"type": "constant", "power": power, "duration": duration, "noise": noise})
        return self

    def add_gap(self, duration: float):
        """Add a zero-power gap."""
        self.segments.append({"type": "constant", "power": 0.0, "duration": duration, "noise": 0.0})
        return self

    def add_boot_spike(self, power: float, duration: float = 5.0):
        """Add a short high-power boot spike."""
        self.segments.append({"type": "constant", "power": power, "duration": duration, "noise": power * 0.1})
        return self

    def generate(
        self, 
        start_time: datetime = None,
        sample_interval: float = 10.0,
        jitter: float = 0.0,
        drop_rate: float = 0.0,
        time_warp: float = 1.0
    ) -> List[Tuple[datetime, float]]:
        """Generate a sequence of (timestamp, power) readings."""
        if start_time is None:
            start_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        readings = []
        current_time = start_time
        
        for seg in self.segments:
            duration = seg["duration"] * time_warp
            seg_end = current_time + timedelta(seconds=duration)
            
            while current_time < seg_end:
                # Decide if we drop this packet
                if random.random() >= drop_rate:
                    # Add jitter to power
                    p_noise = random.uniform(-seg["noise"], seg["noise"])
                    power = max(0.0, seg["power"] + p_noise)
                    
                    readings.append((current_time, power))
                
                # Advance time with jittered interval
                interval_jitter = random.uniform(-jitter, jitter)
                actual_interval = max(0.1, sample_interval + interval_jitter)
                current_time += timedelta(seconds=actual_interval)
                
        return readings

class PacketDropper:
    """Simple utility to simulate packet loss."""
    def __init__(self, drop_probability: float = 0.0):
        self.drop_prob = drop_probability

    def should_drop(self) -> bool:
        return random.random() < self.drop_prob
