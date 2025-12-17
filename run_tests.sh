#!/bin/bash
# Quick test script for ha_washdata integration
# Runs through all scenarios automatically

set -e

# Load MQTT credentials from secrets.py if it exists
SECRETS_FILE="devtools/secrets.py"
if [ -f "$SECRETS_FILE" ]; then
    # Extract values from secrets.py using grep and cut
    MQTT_HOST=$(grep "^MQTT_HOST = " "$SECRETS_FILE" | cut -d'"' -f2)
    MQTT_PORT=$(grep "^MQTT_PORT = " "$SECRETS_FILE" | cut -d'=' -f2 | xargs)
    MQTT_USERNAME=$(grep "^MQTT_USERNAME = " "$SECRETS_FILE" | cut -d'"' -f2)
    MQTT_PASSWORD=$(grep "^MQTT_PASSWORD = " "$SECRETS_FILE" | cut -d'"' -f2)
    MQTT_USE_TLS=$(grep "^MQTT_USE_TLS = " "$SECRETS_FILE" | cut -d'=' -f2 | xargs)
fi

# Allow command-line overrides
MQTT_HOST="${1:-$MQTT_HOST}"
MQTT_HOST="${MQTT_HOST:-localhost}"
MQTT_PORT="${2:-$MQTT_PORT}"
MQTT_PORT="${MQTT_PORT:-1883}"

MQTT_CMD_TOPIC="homeassistant/mock_washer_power/cmd"

echo "=================================="
echo "HA WashData Integration Test Suite"
echo "=================================="
echo "MQTT: $MQTT_HOST:$MQTT_PORT"
[ -n "$MQTT_USERNAME" ] && echo "Auth: $MQTT_USERNAME"
echo ""

# Check mosquitto_pub availability
if ! command -v mosquitto_pub &> /dev/null; then
    echo "❌ mosquitto_pub not found. Install with: apt-get install mosquitto-clients"
    exit 1
fi

run_test() {
    local name=$1
    local command=$2
    local wait_time=$3
    
    echo "=================================="
    echo "TEST: $name"
    echo "Command: mosquitto_pub -t $MQTT_CMD_TOPIC -m '$command'"
    echo "Wait: ${wait_time}s"
    echo "=================================="
    
    # Build mosquitto_pub command with optional auth
    local mqtt_cmd="mosquitto_pub -h \"$MQTT_HOST\" -p \"$MQTT_PORT\""
    if [ -n "$MQTT_USERNAME" ] && [ -n "$MQTT_PASSWORD" ]; then
        mqtt_cmd="$mqtt_cmd -u \"$MQTT_USERNAME\" -P \"$MQTT_PASSWORD\""
    fi
    mqtt_cmd="$mqtt_cmd -t \"$MQTT_CMD_TOPIC\" -m \"$command\""
    
    eval "$mqtt_cmd"
    sleep "$wait_time"
    echo ""
}

echo "Starting tests... Check HA logs and UI for results."
echo ""

# Normal cycles
run_test "SHORT Normal Cycle" "SHORT" 5
run_test "MEDIUM Normal Cycle" "MEDIUM" 8
run_test "LONG Normal Cycle" "LONG" 12

# Fault scenarios
echo ""
echo "--- FAULT SCENARIOS (watch for watchdog/forced ends) ---"
echo ""

run_test "LONG with Dropout (sensor offline mid-cycle)" "LONG_DROPOUT" 30
run_test "MEDIUM with Glitches (power noise)" "MEDIUM_GLITCH" 10
run_test "SHORT Stuck Phase (infinite loop)" "SHORT_STUCK" 30
run_test "LONG Incomplete (never finishes)" "LONG_INCOMPLETE" 30

# Final summary
echo ""
echo "=================================="
echo "✅ All tests completed!"
echo "=================================="
echo ""
echo "Check HA logs for:"
echo "  - Cycle detection (process_reading)"
echo "  - Watchdog activity (when applicable)"
echo "  - Profile matching results"
echo ""
echo "Check HA UI for:"
echo "  - Completed cycles in recent history"
echo "  - Power graph for each cycle"
echo "  - Program name detection"
echo ""
