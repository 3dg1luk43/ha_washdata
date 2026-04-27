"""Diagnostic ring buffers for WashData - rolling 24-hour window.

Each WashDataManager owns one DiagBuffer instance that accumulates three
independent time-series for the last 24 hours:

  power_trace   - every raw power-sensor reading, before throttling
  state_history - detector state transitions (off/starting/running/...)
  logs          - DEBUG-and-above log lines emitted by this integration

All buffers are in-memory only (no disk writes) so they impose zero I/O
overhead and vanish cleanly on HA restart.  The 24-hour window caps memory
at a predictable ceiling regardless of sensor polling rate.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any

from homeassistant.util import dt as dt_util

_WINDOW = timedelta(hours=24)

# Hard upper-bound on entries per buffer.  At a 1-second sensor interval,
# 100 000 power entries ~= 27 hours - enough to always cover the window.
_MAX_POWER = 100_000
_MAX_LOGS = 5_000
_MAX_STATES = 2_000

_INTEGRATION_LOGGER_NAME = "custom_components.ha_washdata"


def _ts_iso(unix: float) -> str:
    return datetime.fromtimestamp(unix, tz=timezone.utc).isoformat()


class _LogHandler(logging.Handler):
    """Logging handler that buffers records for a single named device.

    Installed on the integration root logger so it receives all records
    produced anywhere inside *custom_components.ha_washdata*, then keeps
    only the ones whose formatted message contains ``[device_name]`` -
    the prefix injected by :class:`~.log_utils.DeviceLoggerAdapter`.
    """

    def __init__(self, device_name: str) -> None:
        super().__init__()
        # Match the exact prefix added by DeviceLoggerAdapter
        self._tag = f"[{device_name}]"
        # Store (created_float, levelname, message)
        self._buf: deque[tuple[float, str, str]] = deque(maxlen=_MAX_LOGS)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            if self._tag not in msg:
                return
            self._buf.append((record.created, record.levelname, msg))
        except Exception:  # pylint: disable=broad-except
            self.handleError(record)

    def snapshot(self, cutoff: float) -> list[dict[str, Any]]:
        items = list(self._buf)
        return [
            {"ts": _ts_iso(ts), "lvl": lvl, "msg": msg}
            for ts, lvl, msg in items
            if ts >= cutoff
        ]


class DiagBuffer:
    """Per-device diagnostic ring buffer aggregating power, states, and logs.

    Lifecycle::

        # on manager creation
        self.diag_buffer = DiagBuffer(config_entry.title)

        # on each raw power reading
        self.diag_buffer.record_power(watts, timestamp)

        # on each detector state transition
        self.diag_buffer.record_state(old, new, program, timestamp)

        # on manager shutdown
        self.diag_buffer.uninstall()

        # in diagnostics.py
        snapshot = manager.diag_buffer.snapshot()
    """

    def __init__(self, device_name: str) -> None:
        self._device_name = device_name

        # Raw power readings: (unix_ts_float, watts)
        self._power: deque[tuple[float, float]] = deque(maxlen=_MAX_POWER)

        # State transitions: (unix_ts_float, from_state, to_state, program)
        self._states: deque[tuple[float, str, str, str]] = deque(maxlen=_MAX_STATES)

        # Log handler - installed on the integration root logger
        self._log_handler = _LogHandler(device_name)
        logging.getLogger(_INTEGRATION_LOGGER_NAME).addHandler(self._log_handler)

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def record_power(self, watts: float, ts: datetime) -> None:
        """Record one raw power-sensor reading (call *before* any throttling)."""
        self._power.append((ts.timestamp(), watts))

    def record_state(
        self,
        from_state: str,
        to_state: str,
        program: str,
        ts: datetime,
    ) -> None:
        """Record a detector state transition."""
        self._states.append((ts.timestamp(), from_state, to_state, program))

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def redacted_snapshot(self) -> dict[str, Any]:
        """Like :meth:`snapshot` but with identifying fields removed.

        Strips ``device_name`` from the top-level dict and removes the ``msg``
        field from each log entry so raw log text (which contains the device
        name prefix injected by :class:`~.log_utils.DeviceLoggerAdapter`) is
        not included in exported diagnostics.
        """
        data = self.snapshot()
        data.pop("device_name", None)
        data["logs"] = [
            {k: v for k, v in entry.items() if k != "msg"}
            for entry in data.get("logs", [])
        ]
        return data

    def snapshot(self) -> dict[str, Any]:
        """Return all three buffers filtered to the last 24 hours.

        Returned structure::

            {
              "window_hours": 24,
              "device_name": "...",
              "power_trace": [[iso_ts, watts], ...],
              "state_history": [{"ts": ..., "from": ..., "to": ..., "program": ...}, ...],
              "logs": [{"ts": ..., "lvl": ..., "msg": ...}, ...],
            }
        """
        cutoff = (dt_util.now() - _WINDOW).timestamp()
        power_items = list(self._power)
        state_items = list(self._states)
        return {
            "window_hours": 24,
            "device_name": self._device_name,
            "power_trace": [
                [_ts_iso(ts), w]
                for ts, w in power_items
                if ts >= cutoff
            ],
            "state_history": [
                {"ts": _ts_iso(ts), "from": f, "to": t, "program": prog}
                for ts, f, t, prog in state_items
                if ts >= cutoff
            ],
            "logs": self._log_handler.snapshot(cutoff),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def uninstall(self) -> None:
        """Remove the log handler from the integration logger.

        Must be called from ``WashDataManager.async_shutdown()`` to avoid
        accumulating stale handlers across config-entry reloads.
        """
        logging.getLogger(_INTEGRATION_LOGGER_NAME).removeHandler(self._log_handler)
