# Development Tools

Note: Some examples use "washer" in topic/entity names, but the same tooling applies to other predictable-cycle appliances (e.g., dryers and dishwashers).

**See [../TESTING.md](../TESTING.md) for comprehensive documentation.**

All testing and mock socket documentation has been consolidated into [../TESTING.md](../TESTING.md):

- Mock socket reference & parameters
- Fault injection scenarios
- Testing procedures
- Debugging guide

## Quick Start

```bash
cd /root/ha_washdata/devtools
pip install paho-mqtt
python3 mqtt_mock_socket.py --speedup 720
```

In another terminal:
```bash
mosquitto_pub -t homeassistant/mock_washer_power/cmd -m 'LONG'
```

See [../TESTING.md#mock-socket-reference](../TESTING.md#mock-socket-reference) for full documentation.
