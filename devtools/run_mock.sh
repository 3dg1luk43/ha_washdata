#!/bin/bash

PID_FILE="/root/ha_washdata/devtools/.mock_socket.pid"
LOG_FILE="/root/ha_washdata/devtools/server.log"

start() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "Mock socket is already running (PID: $pid)"
            return 1
        fi
    fi
    
    echo "Starting mock socket..."
    nohup /root/ha_washdata/.venv/bin/python /root/ha_washdata/devtools/mqtt_mock_socket.py --web-port 8080 > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Mock socket started (PID: $!)"
    echo "Logs: $LOG_FILE"
}

stop() {
    if [ ! -f "$PID_FILE" ]; then
        echo "Mock socket is not running (PID file not found)"
        return 1
    fi
    
    pid=$(cat "$PID_FILE")
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "Stopping mock socket (PID: $pid)..."
        kill "$pid"
        rm "$PID_FILE"
        echo "Mock socket stopped"
    else
        echo "Mock socket is not running"
        rm "$PID_FILE"
    fi
}

case "${1:-start}" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    *)
        echo "Usage: $0 {start|stop}"
        exit 1
        ;;
esac