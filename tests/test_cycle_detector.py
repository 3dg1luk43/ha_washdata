"""Test the cycle detector."""
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock

from custom_components.ha_washdata.cycle_detector import CycleDetector, CycleDetectorConfig
from custom_components.ha_washdata.const import STATE_IDLE, STATE_OFF, STATE_RUNNING

class TestCycleDetector(unittest.TestCase):
    def setUp(self):
        self.config = CycleDetectorConfig(min_power=5.0, off_delay=60)
        self.state_changes = []
        self.cycle_ends = []
        
        def on_state_change(old, new):
            self.state_changes.append((old, new))
            
        def on_cycle_end(data):
            self.cycle_ends.append(data)
            
        self.detector = CycleDetector(self.config, on_state_change, on_cycle_end)

    def test_cycle_detection(self):
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        
        # Initial low power -> OFF
        self.detector.process_reading(1.0, start_time)
        self.assertEqual(self.detector.state, STATE_OFF)
        
        # High power -> START
        self.detector.process_reading(10.0, start_time + timedelta(seconds=1))
        self.assertEqual(self.detector.state, STATE_RUNNING)
        self.assertEqual(self.state_changes[-1], (STATE_OFF, STATE_RUNNING))
        
        # Continued high power
        self.detector.process_reading(50.0, start_time + timedelta(minutes=10))
        self.assertEqual(self.detector.state, STATE_RUNNING)
        
        # Low power, but within delay
        self.detector.process_reading(1.0, start_time + timedelta(minutes=10, seconds=30))
        self.assertEqual(self.detector.state, STATE_RUNNING)
        
        # Low power, exceeding delay -> END
        end_time = start_time + timedelta(minutes=10, seconds=61)
        self.detector.process_reading(1.0, end_time)
        self.assertEqual(self.detector.state, STATE_OFF)
        self.assertEqual(self.state_changes[-1], (STATE_RUNNING, STATE_OFF))
        
        self.assertEqual(len(self.cycle_ends), 1)
        self.assertEqual(len(self.cycle_ends[0]["power_data"]), 4) # 3 readings + final check
        self.assertAlmostEqual(self.cycle_ends[0]["duration"], 600.0, delta=1.0) # 10 minutes

if __name__ == "__main__":
    unittest.main()
