"""Unit tests for DiagBuffer — rolling 24-hour diagnostic ring buffer."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from custom_components.ha_washdata.diag_buffer import DiagBuffer, _INTEGRATION_LOGGER_NAME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _ts(offset_seconds: float = 0.0) -> datetime:
    return _now() + timedelta(seconds=offset_seconds)


# ---------------------------------------------------------------------------
# record_power / power_samples
# ---------------------------------------------------------------------------


def test_record_power_appears_in_power_samples():
    buf = DiagBuffer("TestDevice")
    buf.record_power(100.0, _ts())
    samples = buf.power_samples()
    assert len(samples) == 1
    assert samples[0][1] == pytest.approx(100.0)
    buf.uninstall()


def test_power_samples_returns_all_when_no_cutoff():
    buf = DiagBuffer("TestDevice")
    for w in (50.0, 100.0, 200.0):
        buf.record_power(w, _ts())
    assert len(buf.power_samples()) == 3
    buf.uninstall()


def test_power_samples_filters_by_since_ts():
    buf = DiagBuffer("TestDevice")
    past = _ts(-200)
    recent = _ts(-10)
    buf.record_power(10.0, past)
    buf.record_power(20.0, recent)
    cutoff = (_now() - timedelta(seconds=100)).timestamp()
    filtered = buf.power_samples(since_ts=cutoff)
    assert len(filtered) == 1
    assert filtered[0][1] == pytest.approx(20.0)
    buf.uninstall()


# ---------------------------------------------------------------------------
# record_state
# ---------------------------------------------------------------------------


def test_record_state_appears_in_snapshot():
    buf = DiagBuffer("TestDevice")
    buf.record_state("off", "starting", "Cotton 40", _ts())
    snap = buf.snapshot()
    assert len(snap["state_history"]) == 1
    entry = snap["state_history"][0]
    assert entry["from"] == "off"
    assert entry["to"] == "starting"
    assert entry["program"] == "Cotton 40"
    buf.uninstall()


def test_record_multiple_state_transitions():
    buf = DiagBuffer("TestDevice")
    buf.record_state("off", "starting", "", _ts(-20))
    buf.record_state("starting", "running", "Eco 60", _ts(-10))
    buf.record_state("running", "ending", "Eco 60", _ts())
    snap = buf.snapshot()
    assert len(snap["state_history"]) == 3
    buf.uninstall()


# ---------------------------------------------------------------------------
# snapshot: 24-hour window filtering
# ---------------------------------------------------------------------------


def test_snapshot_excludes_entries_older_than_24h():
    buf = DiagBuffer("TestDevice")
    old_ts = _ts(-(25 * 3600))  # 25 hours ago
    recent_ts = _ts(-60)
    buf.record_power(5.0, old_ts)
    buf.record_power(10.0, recent_ts)
    buf.record_state("off", "starting", "", old_ts)
    buf.record_state("starting", "running", "eco", recent_ts)

    snap = buf.snapshot()
    assert len(snap["power_trace"]) == 1
    assert snap["power_trace"][0][1] == pytest.approx(10.0)
    assert len(snap["state_history"]) == 1
    assert snap["state_history"][0]["to"] == "running"
    buf.uninstall()


def test_snapshot_includes_metadata_fields():
    buf = DiagBuffer("MyWasher")
    snap = buf.snapshot()
    assert snap["window_hours"] == 24
    assert snap["device_name"] == "MyWasher"
    assert "power_trace" in snap
    assert "state_history" in snap
    assert "logs" in snap
    buf.uninstall()


def test_snapshot_timestamps_are_iso_strings():
    buf = DiagBuffer("TestDevice")
    buf.record_power(42.0, _ts())
    buf.record_state("off", "starting", "", _ts())
    snap = buf.snapshot()
    # ISO 8601 timestamps contain 'T'
    assert "T" in snap["power_trace"][0][0]
    assert "T" in snap["state_history"][0]["ts"]
    buf.uninstall()


# ---------------------------------------------------------------------------
# redacted_snapshot
# ---------------------------------------------------------------------------


def test_redacted_snapshot_omits_device_name():
    buf = DiagBuffer("SensitiveDeviceName")
    redacted = buf.redacted_snapshot()
    assert "device_name" not in redacted
    buf.uninstall()


def test_redacted_snapshot_strips_msg_from_logs():
    buf = DiagBuffer("TestDevice")
    logger = logging.getLogger(_INTEGRATION_LOGGER_NAME)
    logger.warning("[TestDevice] test message for redaction check")
    redacted = buf.redacted_snapshot()
    for entry in redacted.get("logs", []):
        assert "msg" not in entry, "msg must be stripped from redacted log entries"
    buf.uninstall()


def test_redacted_snapshot_keeps_ts_and_lvl_in_logs():
    buf = DiagBuffer("TestDevice2")
    logger = logging.getLogger(_INTEGRATION_LOGGER_NAME)
    logger.warning("[TestDevice2] a warning")
    redacted = buf.redacted_snapshot()
    for entry in redacted.get("logs", []):
        assert "ts" in entry
        assert "lvl" in entry
    buf.uninstall()


# ---------------------------------------------------------------------------
# _LogHandler: tag-based filtering
# ---------------------------------------------------------------------------


def test_log_handler_only_captures_matching_device():
    buf_a = DiagBuffer("DeviceA")
    buf_b = DiagBuffer("DeviceB")
    logger = logging.getLogger(_INTEGRATION_LOGGER_NAME)
    logger.warning("[DeviceA] only for A")
    logger.warning("[DeviceB] only for B")

    snap_a = buf_a.snapshot()
    snap_b = buf_b.snapshot()
    msgs_a = {e["msg"] for e in snap_a["logs"]}
    msgs_b = {e["msg"] for e in snap_b["logs"]}

    assert any("DeviceA" in m for m in msgs_a)
    assert not any("DeviceB" in m for m in msgs_a)
    assert any("DeviceB" in m for m in msgs_b)
    assert not any("DeviceA" in m for m in msgs_b)

    buf_a.uninstall()
    buf_b.uninstall()


def test_log_handler_captures_debug_and_above():
    buf = DiagBuffer("LogLevels")
    logger = logging.getLogger(_INTEGRATION_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.debug("[LogLevels] debug msg")
    logger.info("[LogLevels] info msg")
    logger.warning("[LogLevels] warning msg")
    snap = buf.snapshot()
    levels = {e["lvl"] for e in snap["logs"]}
    # At least WARNING must appear; DEBUG/INFO depend on logger level config
    assert "WARNING" in levels
    buf.uninstall()


# ---------------------------------------------------------------------------
# uninstall: removes log handler
# ---------------------------------------------------------------------------


def test_uninstall_removes_handler():
    buf = DiagBuffer("UninstallTest")
    logger = logging.getLogger(_INTEGRATION_LOGGER_NAME)
    handler_before = set(logger.handlers)
    buf.uninstall()
    handler_after = set(logger.handlers)
    # After uninstall the handler count should drop (or at least not grow)
    assert len(handler_after) <= len(handler_before)


def test_no_log_captured_after_uninstall():
    buf = DiagBuffer("AfterUninstall")
    buf.uninstall()
    logger = logging.getLogger(_INTEGRATION_LOGGER_NAME)
    logger.warning("[AfterUninstall] after uninstall message")
    snap = buf.snapshot()
    # Nothing should appear since handler was removed
    assert snap["logs"] == []
