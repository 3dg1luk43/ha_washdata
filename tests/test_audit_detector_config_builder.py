"""Audit 2026-10-02 F2 / DETECT-12: one detector-config builder for every writer.

The manager built its CycleDetectorConfig twice (constructor and options reload)
and the copies drifted (item 351, 388a); every replay harness hand-rolled a third
with its own fallbacks. `detector_config.build_detector_config` is now the only
builder. These tests pin the constructor and the reload to it, field by field,
over every device type and a spread of option sets.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata import const as C
from custom_components.ha_washdata.detector_config import build_detector_config
from custom_components.ha_washdata.manager import WashDataManager

DEVICES = [
    "washing_machine", "dishwasher", "dryer", "washer_dryer", "pump", "other",
    "generic", "air_fryer", "bread_maker",
]
CHOICES: dict[str, list[Any]] = {
    C.CONF_MIN_POWER: [1.0, 5.5], C.CONF_OFF_DELAY: [60, 600], C.CONF_MIN_OFF_GAP: [120, 3600],
    C.CONF_INTERRUPTED_MIN_SECONDS: [60, 300], C.CONF_COMPLETION_MIN_SECONDS: [120, 600],
    C.CONF_START_DURATION_THRESHOLD: [0.0, 5.0], C.CONF_START_THRESHOLD_W: [3.0, 8.0],
    C.CONF_STOP_THRESHOLD_W: [0.8, 2.0], C.CONF_PROFILE_MATCH_MIN_DURATION_RATIO: [0.05, 0.5],
    C.CONF_PROFILE_MATCH_THRESHOLD: [0.4, 0.6], C.CONF_ANTI_WRINKLE_ENABLED: [True, False],
    C.CONF_SMART_TERMINATION_DURATION_RATIO: [0.95, 0.99], C.CONF_DELAY_TIMEOUT_HOURS: [8, 12],
    C.CONF_END_ENERGY_THRESHOLD: [0.01, 0.1], C.CONF_CURVE_PREROLL_SECONDS: [0.0, 30.0],
}


def _option_sets() -> list[tuple[dict, dict]]:
    rnd = random.Random(11)
    out: list[tuple[dict, dict]] = [({}, {}), ({}, {C.CONF_MIN_POWER: 4.0, C.CONF_OFF_DELAY: 240})]
    for _ in range(20):
        opts = {k: rnd.choice(v) for k, v in CHOICES.items() if rnd.random() < 0.5}
        data = {C.CONF_MIN_POWER: 6.0} if rnd.random() < 0.4 else {}
        out.append((opts, data))
    return out


def _entry(device: str, opts: dict, data: dict) -> Any:
    entry = MagicMock()
    entry.entry_id = "e"
    entry.title = "W"
    entry.options = {"power_sensor": "sensor.p", C.CONF_DEVICE_TYPE: device, **opts}
    entry.data = {"power_sensor": "sensor.p", C.CONF_DEVICE_TYPE: device, **data}
    return entry


def _manager(entry: Any) -> WashDataManager:
    hass = MagicMock()
    hass.data = {}
    with patch("custom_components.ha_washdata.manager.ProfileStore"):
        return WashDataManager(hass, entry)


@pytest.mark.parametrize("device", DEVICES)
def test_constructor_builds_exactly_what_the_builder_builds(device: str) -> None:
    for opts, data in _option_sets():
        entry = _entry(device, opts, data)
        mgr = _manager(entry)
        assert dataclasses.asdict(mgr.detector.config) == dataclasses.asdict(
            build_detector_config(entry.options, entry.data, mgr.device_type)
        ), (device, opts, data)


@pytest.mark.parametrize("device", ["washing_machine", "dishwasher", "dryer"])
async def test_reload_lands_on_the_builder_and_keeps_the_config_object(device: str) -> None:
    mgr = _manager(_entry(device, {}, {}))
    mgr.profile_store.get_duration_ratio_limits = MagicMock(return_value=(0.1, 1.8))
    mgr.profile_store.async_rebuild_all_envelopes = AsyncMock()
    mgr._setup_maintenance_scheduler = AsyncMock()
    live_config = mgr.detector.config
    for opts, data in _option_sets()[:8]:
        entry = _entry(device, opts, data)
        mgr.hass.config_entries.async_get_entry = MagicMock(return_value=entry)
        await mgr.async_reload_config(entry)
        assert mgr.detector.config is live_config
        assert dataclasses.asdict(mgr.detector.config) == dataclasses.asdict(
            build_detector_config(entry.options, entry.data, mgr.device_type)
        ), (device, opts, data)


def test_a_junk_option_falls_back_to_its_default_instead_of_raising() -> None:
    """Item 279 / audit PLATFORM-13 & PLAYGROUND-10: one non-numeric or null option
    used to raise out of the config build, failing setup and every replay."""
    good = build_detector_config({}, {}, "washing_machine")
    bad = build_detector_config(
        {C.CONF_OFF_DELAY: "abc", C.CONF_MIN_OFF_GAP: None, C.CONF_STOP_THRESHOLD_W: float("nan"),
         C.CONF_PROFILE_MATCH_INTERVAL: "300"},
        {C.CONF_MIN_POWER: "x"},
        "washing_machine",
    )
    assert bad.off_delay == good.off_delay
    assert bad.min_off_gap == good.min_off_gap
    assert bad.stop_threshold_w == good.stop_threshold_w
    assert bad.min_power == good.min_power
    assert bad.match_interval == 300


def test_an_oversized_json_integer_falls_back_instead_of_raising() -> None:
    """`json` keeps `10**400` as an unbounded int and `float()` on it raises
    OverflowError, which the junk-value guard did not catch. `1e400` is a float
    literal that parses to inf and never raises, so both are asserted."""
    good = build_detector_config({}, {}, "washing_machine")
    for huge in (10**400, 1e400):
        bad = build_detector_config(
            {C.CONF_OFF_DELAY: huge, C.CONF_STOP_THRESHOLD_W: huge},
            {C.CONF_MIN_POWER: huge},
            "washing_machine",
        )
        assert bad.off_delay == good.off_delay
        assert bad.stop_threshold_w == good.stop_threshold_w
        assert bad.min_power == good.min_power
