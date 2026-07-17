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

import unittest
import logging
import random
import os
import sys

# Ensure we can import from local directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock nicegui and paho.mqtt before importing mqtt_mock_socket
from unittest.mock import MagicMock
sys.modules["nicegui"] = MagicMock()
sys.modules["nicegui.ui"] = MagicMock()
sys.modules["nicegui.events"] = MagicMock()
sys.modules["paho"] = MagicMock()
sys.modules["paho.mqtt"] = MagicMock()
sys.modules["paho.mqtt.client"] = MagicMock()

from mqtt_mock_socket import CycleSynthesizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifySynthesis")

class TestCycleSynthesis(unittest.TestCase):
    def setUp(self):
        # Create a simple step function template
        self.simple_template = {
            "profile_name": "TestProfile",
            "power_data": [
                [0, 100.0],
                [5, 100.0],
                [10, 0.0]
            ]
        }
        
    def test_synthesize_basic(self):
        """Test basic synthesis without jitter or variability."""
        syn = CycleSynthesizer(jitter_w=0.0, variability=0.0)
        readings = syn.synthesize(self.simple_template)
        
        self.assertEqual(len(readings), 11)
        for i, val in enumerate(readings):
            if i < 10:
                self.assertEqual(val, 100.0)
            else:
                self.assertEqual(val, 0.0)
        logger.info("Basic synthesis passed.")

    def test_synthesize_jitter(self):
        """Test that jitter introduces variance but keeps shape."""
        jitter_amount = 5.0
        syn = CycleSynthesizer(jitter_w=jitter_amount, variability=0.0)
        readings = syn.synthesize(self.simple_template)
        
        self.assertEqual(len(readings), 11)
        within_bounds = True
        for i, val in enumerate(readings):
            if i < 10:
                if not (80.0 < val < 120.0):
                    within_bounds = False
        self.assertTrue(within_bounds)
        logger.info("Jitter synthesis passed.")

if __name__ == "__main__":
    unittest.main()
