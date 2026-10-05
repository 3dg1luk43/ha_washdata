#!/bin/bash
# The mock MQTT plug (devtools/mqtt_mock_socket.py) as a systemd service.
#
#   ./run_mock.sh on       install, enable at boot and start (web UI on :8081)
#   ./run_mock.sh off      stop and disable
#   ./run_mock.sh status   service status
#   ./run_mock.sh logs     follow the log
#
# MOCK_WEB_HOST / MOCK_WEB_PORT change the web UI bind address (default 0.0.0.0, every
# interface) and port (default 8081), e.g. MOCK_WEB_HOST=127.0.0.1 ./run_mock.sh on.
# `on` also stops an instance left by the old nohup start (devtools/.mock_socket.pid),
# which would otherwise fight the service over the same MQTT discovery topics.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT="washdata-mock-socket"
UNIT_FILE="/etc/systemd/system/${UNIT}.service"
HOST="${MOCK_WEB_HOST:-0.0.0.0}"
PORT="${MOCK_WEB_PORT:-8081}"
LEGACY_PID="${REPO}/devtools/.mock_socket.pid"

stop_legacy() {
    [ -f "$LEGACY_PID" ] || return 0
    local pid
    pid="$(cat "$LEGACY_PID")"
    if ps -p "$pid" -o args= 2>/dev/null | grep -q "mqtt_mock_socket.py"; then
        echo "Stopping the old nohup instance (PID $pid)"
        kill "$pid"
    fi
    rm -f "$LEGACY_PID"
}

on() {
    stop_legacy
    cat > "$UNIT_FILE" <<EOF
[Unit]
Description=WashData mock MQTT plugs (devtools/mqtt_mock_socket.py)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${REPO}
Environment=PYTHONUNBUFFERED=1
ExecStart=${REPO}/.venv/bin/python ${REPO}/devtools/mqtt_mock_socket.py --web-host ${HOST} --web-port ${PORT}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable --now "$UNIT"
    # A restart picks up a changed port or code; enable --now leaves a running unit alone.
    systemctl restart "$UNIT"
    local shown="$HOST"
    [ "$HOST" = "0.0.0.0" ] && shown="$(hostname -I | awk '{print $1}')"
    echo "On: http://${shown}:${PORT} (logs: $0 logs)"
}

off() {
    stop_legacy
    if [ -f "$UNIT_FILE" ]; then
        systemctl disable --now "$UNIT"
    fi
    echo "Off"
}

case "${1:-}" in
    on) on ;;
    off) off ;;
    status) systemctl status "$UNIT" --no-pager ;;
    logs) journalctl -u "$UNIT" -f ;;
    *) echo "Usage: $0 {on|off|status|logs}"; exit 1 ;;
esac
