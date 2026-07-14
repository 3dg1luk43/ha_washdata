"""Tests for WashDataStore storage migration (v1 → current STORAGE_VERSION).

Each test targets a specific migration step by passing old_major_version = N-1
directly to _async_migrate_func, which lets us test each step in isolation
without real file I/O.  A full-chain test (v1 → current) is also included to
catch regressions where a later step clobbers data that an earlier one set.

Power-data format notes:
  v1/v2: ISO-timestamp format — [["2023-01-01T12:00:00+00:00", watts], ...]
  v3+:   offset format        — [[offset_seconds, power], ...]
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata.const import STORAGE_KEY, STORAGE_VERSION
from custom_components.ha_washdata.profile_store import WashDataStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_START_ISO = "2023-06-01T10:00:00+00:00"


def _make_hass() -> MagicMock:
    hass = MagicMock()
    hass.config.path = lambda *a: "/tmp/" + "/".join(a)
    return hass


def _make_store() -> WashDataStore:
    return WashDataStore(_make_hass(), STORAGE_VERSION, f"{STORAGE_KEY}.test")


def _iso_cycle(cycle_id: str = "c1", watts: list[float] | None = None) -> dict[str, Any]:
    """Build a cycle with ISO-timestamp power_data (pre-v3 format)."""
    if watts is None:
        watts = [0.0, 100.0, 500.0, 800.0, 800.0, 200.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    power_data = [
        [f"2023-06-01T10:00:{i:02d}+00:00", w]
        for i, w in enumerate(watts)
    ]
    return {
        "id": cycle_id,
        "start_time": _START_ISO,
        "end_time": "2023-06-01T10:02:00+00:00",
        "duration": 120.0,
        "power_data": power_data,
        "profile_name": "Standard",
        "max_power": max(watts),
        "termination_reason": "min_off_gap",
    }


def _offset_cycle(cycle_id: str = "c1") -> dict[str, Any]:
    """Build a cycle with already-canonical offset power_data (v3+ format)."""
    return {
        "id": cycle_id,
        "start_time": _START_ISO,
        "duration": 120.0,
        "power_data": [[0.0, 0.0], [10.0, 100.0], [20.0, 500.0], [30.0, 0.0]],
        "profile_name": "Standard",
        "status": "completed",
        "max_power": 500.0,
        "termination_reason": "min_off_gap",
    }


def _recorded_cycle_new(cycle_id: str = "rec1") -> dict[str, Any]:
    """Recorded cycle with explicit meta marker (added in recent builds)."""
    return {
        "id": cycle_id,
        "start_time": _START_ISO,
        "duration": 100.0,
        "power_data": [[0.0, 200.0], [10.0, 600.0], [20.0, 100.0]],
        "profile_name": "Cotton 40",
        "status": "completed",
        "meta": {"source": "recorder", "original_samples": 12},
    }


def _recorded_cycle_old(cycle_id: str = "rec2") -> dict[str, Any]:
    """Old recorded cycle: completed, no max_power, no termination_reason (pre-meta era)."""
    return {
        "id": cycle_id,
        "start_time": _START_ISO,
        "duration": 90.0,
        "power_data": [[0.0, 300.0], [10.0, 700.0]],
        "profile_name": "Quick",
        "status": "completed",
        # Intentionally missing max_power and termination_reason — old recordings
    }


# ---------------------------------------------------------------------------
# v1 → current: signatures computed for ISO-format cycles
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v1_signature_computed_for_iso_format_cycle():
    """v1 → v2: signature is computed for cycles that have ISO-format power_data."""
    store = _make_store()
    cycle = _iso_cycle()
    data = {"past_cycles": [cycle], "profiles": {}}

    result = await store._async_migrate_func(1, 1, data)

    c = result["past_cycles"][0]
    assert "signature" in c, "Signature should be computed during v1→v2 migration"
    assert c["signature"]["max_power"] > 0


@pytest.mark.asyncio
async def test_v1_signature_skipped_for_too_few_points():
    """v1 → v2: cycles with ≤ 10 points do not get a signature (not enough data)."""
    store = _make_store()
    cycle = _iso_cycle(watts=[0.0, 100.0, 0.0])  # only 3 points
    data = {"past_cycles": [cycle], "profiles": {}}

    result = await store._async_migrate_func(1, 1, data)
    c = result["past_cycles"][0]
    assert "signature" not in c


@pytest.mark.asyncio
async def test_v1_signature_skipped_for_empty_power_data():
    """v1 → v2: cycles with empty power_data are silently skipped."""
    store = _make_store()
    cycle = {"id": "empty", "power_data": [], "start_time": _START_ISO}
    data = {"past_cycles": [cycle], "profiles": {}}
    result = await store._async_migrate_func(1, 1, data)
    assert "signature" not in result["past_cycles"][0]


# ---------------------------------------------------------------------------
# v2 → current: ISO power_data converted to offset format; status added
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v2_power_data_converted_to_offset_format():
    """v2 → v3: ISO-timestamp power_data converted to [[offset_s, watts], ...]."""
    store = _make_store()
    cycle = _iso_cycle()
    data = {"past_cycles": [cycle], "profiles": {}}

    result = await store._async_migrate_func(2, 1, data)

    c = result["past_cycles"][0]
    power_data = c["power_data"]
    assert isinstance(power_data[0][0], float), "After migration, offsets must be floats"
    assert power_data[0][0] == pytest.approx(0.0), "First offset must be 0"
    assert power_data[1][0] > 0, "Subsequent offsets must be positive"


@pytest.mark.asyncio
async def test_v2_offset_format_data_not_re_migrated():
    """v2 → v3: cycles already in offset format are left unchanged."""
    store = _make_store()
    cycle = _offset_cycle()
    original_power_data = list(cycle["power_data"])
    data = {"past_cycles": [cycle], "profiles": {}}

    result = await store._async_migrate_func(2, 1, data)
    assert result["past_cycles"][0]["power_data"] == original_power_data


@pytest.mark.asyncio
async def test_v2_status_added_to_cycles_without_status():
    """v2 → v3: cycles missing 'status' get status='completed'."""
    store = _make_store()
    cycle = _offset_cycle()
    del cycle["status"]
    data = {"past_cycles": [cycle], "profiles": {}}

    result = await store._async_migrate_func(2, 1, data)
    assert result["past_cycles"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_v2_existing_status_preserved():
    """v2 → v3: cycles that already have a status retain it."""
    store = _make_store()
    cycle = _offset_cycle()
    cycle["status"] = "interrupted"
    data = {"past_cycles": [cycle], "profiles": {}}

    result = await store._async_migrate_func(2, 1, data)
    assert result["past_cycles"][0]["status"] == "interrupted"


@pytest.mark.asyncio
async def test_v2_profile_device_type_added_if_missing():
    """v2 → v3: profiles without 'device_type' get device_type='washing_machine'."""
    store = _make_store()
    data = {
        "past_cycles": [],
        "profiles": {"Cotton 40": {"avg_duration": 3600.0}},
    }
    result = await store._async_migrate_func(2, 1, data)
    assert result["profiles"]["Cotton 40"]["device_type"] == "washing_machine"


@pytest.mark.asyncio
async def test_v2_profile_device_type_preserved_if_present():
    """v2 → v3: profiles with an existing device_type keep it."""
    store = _make_store()
    data = {
        "past_cycles": [],
        "profiles": {"Quick Wash": {"avg_duration": 1800.0, "device_type": "dishwasher"}},
    }
    result = await store._async_migrate_func(2, 1, data)
    assert result["profiles"]["Quick Wash"]["device_type"] == "dishwasher"


# ---------------------------------------------------------------------------
# v3 → current: phases initialized on profiles; custom_phases dict created
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v3_profile_phases_initialized():
    """v3 → v4: profiles without 'phases' get phases=[]."""
    store = _make_store()
    data = {
        "past_cycles": [],
        "profiles": {"Standard": {"avg_duration": 3600.0}},
    }
    result = await store._async_migrate_func(3, 1, data)
    assert result["profiles"]["Standard"]["phases"] == []


@pytest.mark.asyncio
async def test_v3_profile_existing_phases_preserved():
    """v3 → v4: profiles with existing phases keep them."""
    store = _make_store()
    existing_phases = [{"name": "Wash", "start": 0, "end": 0.5}]
    data = {
        "past_cycles": [],
        "profiles": {"Cotton": {"phases": existing_phases}},
    }
    result = await store._async_migrate_func(3, 1, data)
    assert result["profiles"]["Cotton"]["phases"] == existing_phases


@pytest.mark.asyncio
async def test_v3_custom_phases_initialized_if_missing():
    """v3 → v4: custom_phases is initialized if absent."""
    store = _make_store()
    data = {"past_cycles": [], "profiles": {}}
    result = await store._async_migrate_func(3, 1, data)
    assert "custom_phases" in result
    # Initialized to empty dict at v4 (list normalization happens at v5)
    assert result["custom_phases"] is not None


# ---------------------------------------------------------------------------
# v4 → current: custom_phases normalized to canonical list format
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v4_custom_phases_list_normalized():
    """v4 → v5: custom_phases already a list is normalized (deduplicated)."""
    store = _make_store()
    data = {
        "past_cycles": [],
        "profiles": {},
        "custom_phases": [
            {"name": "Rinse", "device_type": "washing_machine", "description": ""},
            {"name": "Rinse", "device_type": "washing_machine", "description": ""},  # duplicate
            {"name": "Spin", "device_type": "washing_machine", "description": ""},
        ],
    }
    result = await store._async_migrate_func(4, 1, data)
    names = [p["name"] for p in result["custom_phases"]]
    assert names.count("Rinse") == 1, "Duplicates must be removed"
    assert "Spin" in names


@pytest.mark.asyncio
async def test_v4_custom_phases_dict_flattened_to_list():
    """v4 → v5: custom_phases stored as dict (old per-device-type format) is flattened."""
    store = _make_store()
    data = {
        "past_cycles": [],
        "profiles": {},
        "custom_phases": {
            "washing_machine": [
                {"name": "Pre-wash", "description": "Cold soak", "device_type": "washing_machine"},
            ],
            "dryer": [
                {"name": "Cool-down", "description": "", "device_type": "dryer"},
            ],
        },
    }
    result = await store._async_migrate_func(4, 1, data)
    assert isinstance(result["custom_phases"], list)
    names = {p["name"] for p in result["custom_phases"]}
    assert "Pre-wash" in names
    assert "Cool-down" in names


@pytest.mark.asyncio
async def test_v4_custom_phases_none_becomes_empty_list():
    """v4 → v5: custom_phases=None (or missing) becomes []."""
    store = _make_store()
    data = {"past_cycles": [], "profiles": {}, "custom_phases": None}
    result = await store._async_migrate_func(4, 1, data)
    assert result["custom_phases"] == []


# ---------------------------------------------------------------------------
# v5 → current: recorded cycles flagged as golden
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v5_new_recorded_cycle_flagged_golden():
    """v5 → v6: cycles with meta.source='recorder' are flagged as golden."""
    store = _make_store()
    cycle = _recorded_cycle_new()
    data = {"past_cycles": [cycle], "profiles": {}}
    result = await store._async_migrate_func(5, 1, data)
    review = result["past_cycles"][0].get("ml_review", {})
    assert review.get("golden") is True


@pytest.mark.asyncio
async def test_v5_auto_detected_cycle_not_flagged_golden():
    """v5 → v6: auto-detected cycles (with max_power) are not flagged."""
    store = _make_store()
    cycle = _offset_cycle()  # has max_power — auto-detected
    data = {"past_cycles": [cycle], "profiles": {}}
    result = await store._async_migrate_func(5, 1, data)
    review = result["past_cycles"][0].get("ml_review")
    assert not (review and review.get("golden")), "Auto-detected cycle must not be golden"


@pytest.mark.asyncio
async def test_v5_already_golden_cycle_not_double_counted():
    """v5 → v6: already-golden cycles are not re-flagged."""
    store = _make_store()
    cycle = _recorded_cycle_new()
    cycle["ml_review"] = {"golden": True, "quality": "good"}
    data = {"past_cycles": [cycle], "profiles": {}}
    result = await store._async_migrate_func(5, 1, data)
    assert result["past_cycles"][0]["ml_review"]["golden"] is True


# ---------------------------------------------------------------------------
# v7 → current: old recordings without meta marker are flagged (structural check)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v7_old_recorded_cycle_no_meta_flagged_golden():
    """v7 → v8: completed cycles missing max_power AND termination_reason are flagged."""
    store = _make_store()
    cycle = _recorded_cycle_old()  # completed, no max_power, no termination_reason
    data = {"past_cycles": [cycle], "profiles": {}}
    result = await store._async_migrate_func(7, 1, data)
    review = result["past_cycles"][0].get("ml_review", {})
    assert review.get("golden") is True


@pytest.mark.asyncio
async def test_v7_auto_detected_cycle_with_max_power_not_flagged():
    """v7 → v8: completed cycle WITH max_power (auto-detected) is not flagged."""
    store = _make_store()
    cycle = _offset_cycle()  # has max_power
    data = {"past_cycles": [cycle], "profiles": {}}
    result = await store._async_migrate_func(7, 1, data)
    review = result["past_cycles"][0].get("ml_review")
    assert not (review and review.get("golden"))


# ---------------------------------------------------------------------------
# v8 → v9: pre-initialize additive top-level keys
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v8_init_additive_keys():
    """v8 → v9: additive keys are created with zero/empty defaults when absent."""
    store = _make_store()
    data: dict[str, Any] = {"past_cycles": [], "profiles": {}}
    result = await store._async_migrate_func(8, 1, data)
    assert result["lifetime_energy_wh"] == 0.0
    assert result["settings_changelog"] == []
    assert result["maintenance_log"] == []


@pytest.mark.asyncio
async def test_v8_init_additive_keys_idempotent():
    """v8 → v9: existing values for additive keys are preserved (idempotent)."""
    store = _make_store()
    data: dict[str, Any] = {
        "past_cycles": [],
        "profiles": {},
        "lifetime_energy_wh": 5.0,
        "settings_changelog": ["x"],
    }
    result = await store._async_migrate_func(8, 1, data)
    assert result["lifetime_energy_wh"] == 5.0
    assert result["settings_changelog"] == ["x"]
    assert result["maintenance_log"] == []


@pytest.mark.asyncio
async def test_v9_to_v10_adds_reference_cycles():
    """v9 → v10: reference_cycles list is created; past_cycles untouched."""
    store = _make_store()
    data: dict[str, Any] = {"past_cycles": [{"id": "a"}], "profiles": {}}
    result = await store._async_migrate_func(9, 1, data)
    assert result["reference_cycles"] == []
    assert result["past_cycles"] == [{"id": "a"}]


@pytest.mark.asyncio
async def test_v9_to_v10_idempotent():
    """v9 → v10: existing reference_cycles preserved (idempotent)."""
    store = _make_store()
    data: dict[str, Any] = {"past_cycles": [], "profiles": {}, "reference_cycles": [{"id": "r"}]}
    result = await store._async_migrate_func(9, 1, data)
    assert result["reference_cycles"] == [{"id": "r"}]


# ---------------------------------------------------------------------------
# Full chain v1 → current: end-to-end through all migration steps
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_chain_v1_to_current_preserves_cycle_count():
    """v1 → current: cycle count is preserved through all migration steps."""
    store = _make_store()
    cycles = [_iso_cycle(f"c{i}") for i in range(3)]
    data = {
        "past_cycles": cycles,
        "profiles": {"Standard": {"avg_duration": 120.0}},
    }
    result = await store._async_migrate_func(1, 1, data)
    assert len(result["past_cycles"]) == 3


@pytest.mark.asyncio
async def test_full_chain_v1_to_current_power_data_in_offset_format():
    """v1 → current: power_data ends up in canonical offset format."""
    store = _make_store()
    cycle = _iso_cycle()
    data = {"past_cycles": [cycle], "profiles": {}}
    result = await store._async_migrate_func(1, 1, data)
    power_data = result["past_cycles"][0]["power_data"]
    assert isinstance(power_data[0][0], float)
    assert power_data[0][0] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_full_chain_v1_to_current_profile_has_phases_and_device_type():
    """v1 → current: profiles gain both 'phases' and 'device_type' through the full chain."""
    store = _make_store()
    data = {
        "past_cycles": [],
        "profiles": {"Eco 40": {"avg_duration": 4800.0}},
    }
    result = await store._async_migrate_func(1, 1, data)
    profile = result["profiles"]["Eco 40"]
    assert "phases" in profile
    assert profile["device_type"] == "washing_machine"


@pytest.mark.asyncio
async def test_full_chain_v1_to_current_custom_phases_initialized():
    """v1 → current: custom_phases is always present after full migration."""
    store = _make_store()
    data = {"past_cycles": [], "profiles": {}}
    result = await store._async_migrate_func(1, 1, data)
    assert "custom_phases" in result


@pytest.mark.asyncio
async def test_current_version_data_returned_unchanged():
    """Current version data passes through _async_migrate_func without modification."""
    store = _make_store()
    data = {
        "past_cycles": [_offset_cycle()],
        "profiles": {"Standard": {"avg_duration": 3600.0, "phases": [], "device_type": "dryer"}},
        "custom_phases": [],
    }
    # Simulate being called at the current version (no migration branches run)
    result = await store._async_migrate_func(STORAGE_VERSION, 1, data)
    assert result["profiles"]["Standard"]["device_type"] == "dryer"
    assert len(result["past_cycles"]) == 1
