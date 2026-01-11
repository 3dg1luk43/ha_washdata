
import sys
import os
import json
import logging


# Add devtools to path
sys.path.append("/root/ha_washdata/devtools")

from unittest.mock import MagicMock
sys.modules["paho"] = MagicMock()
sys.modules["paho.mqtt"] = MagicMock()
sys.modules["paho.mqtt.client"] = MagicMock()

from mqtt_mock_socket import CycleLoader, CycleSynthesizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_synthesis():
    json_path = "/root/ha_washdata/cycle_data/me/washing_machine/real-washing-machine.json"
    
    logger.info(f"Testing CycleLoader with {json_path}")
    templates = CycleLoader.load_from_file(json_path)
    logger.info(f"Loaded {len(templates)} cycles")
    
    if not templates:
        logger.error("No templates loaded!")
        sys.exit(1)
        
    # Test valid template
    t = CycleLoader.get_template(templates)
    logger.info(f"Picked template: {t.get('profile_name')} (ID: {t.get('id')})")
    
    # Test Synthesizer
    synth = CycleSynthesizer(jitter_w=5.0, variability=0.2, glitch_prob=0.1)
    logger.info("Synthesizing cycle...")
    
    new_cycle = synth.synthesize(t, sample_rate_s=1.0)
    
    logger.info(f"Synthesized duration: {new_cycle['duration']}s")
    logger.info(f"Sample points: {len(new_cycle['power_readings'])}")
    logger.info(f"Warp factors: {new_cycle['warped_factors']}")
    
    # Verify non-empty
    if not new_cycle['power_readings']:
        logger.error("Generated empty readings!")
        sys.exit(1)
        
    logger.info("Test Passed!")

if __name__ == "__main__":
    test_synthesis()
