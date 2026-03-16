"""Tests for orphaned diagnostic entity cleanup."""

import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from custom_components.ha_washdata.sensor import cleanup_orphaned_diagnostic_entities


def _mock_registry_entry(entity_id: str, unique_id: str) -> SimpleNamespace:
    return SimpleNamespace(entity_id=entity_id, unique_id=unique_id)


def _build_manager_with_profiles(names: list[str]) -> MagicMock:
    manager = MagicMock()
    manager.profile_store.list_profiles.return_value = [{"name": name} for name in names]
    return manager


def _profile_count_unique_id(entry_id: str, profile_name: str) -> str:
    token = hashlib.sha256(profile_name.encode("utf-8")).hexdigest()[:8]
    return f"{entry_id}_profile_count_{token}"


def test_cleanup_removes_stale_profile_count_and_legacy_wash_phase() -> None:
    """Stale profile counters and legacy wash_phase entities are removed."""
    entry = MagicMock()
    entry.entry_id = "entry1"
    entry.options = {"expose_debug_entities": False}

    manager = _build_manager_with_profiles(["NewName"])
    hass = MagicMock()
    keep_uid = _profile_count_unique_id(entry.entry_id, "NewName")
    stale_uid = _profile_count_unique_id(entry.entry_id, "OldName")

    registry_entries = [
        _mock_registry_entry("sensor.keep_new", keep_uid),
        _mock_registry_entry("sensor.stale_old", stale_uid),
        _mock_registry_entry("sensor.legacy_phase", "entry1_wash_phase"),
    ]

    ent_reg = MagicMock()
    with patch(
        "custom_components.ha_washdata.sensor.entity_registry.async_get",
        return_value=ent_reg,
    ), patch(
        "custom_components.ha_washdata.sensor.entity_registry.async_entries_for_config_entry",
        return_value=registry_entries,
    ):
        removed = cleanup_orphaned_diagnostic_entities(hass, manager, entry)

    assert removed == 2
    removed_ids = [call.args[0] for call in ent_reg.async_remove.call_args_list]
    assert "sensor.stale_old" in removed_ids
    assert "sensor.legacy_phase" in removed_ids
    assert "sensor.keep_new" not in removed_ids


def test_cleanup_removes_stale_debug_diagnostics_when_debug_hidden() -> None:
    """Debug-only diagnostics are removed when debug entities are not exposed."""
    entry = MagicMock()
    entry.entry_id = "entry1"
    entry.options = {"expose_debug_entities": False}

    manager = _build_manager_with_profiles([])
    hass = MagicMock()

    registry_entries = [
        _mock_registry_entry("sensor.debug_info", "entry1_debug_info"),
        _mock_registry_entry("sensor.top_candidates", "entry1_top_candidates"),
        _mock_registry_entry("sensor.match_confidence", "entry1_match_confidence"),
        _mock_registry_entry("binary_sensor.ambiguity", "entry1_ambiguity"),
    ]

    ent_reg = MagicMock()
    with patch(
        "custom_components.ha_washdata.sensor.entity_registry.async_get",
        return_value=ent_reg,
    ), patch(
        "custom_components.ha_washdata.sensor.entity_registry.async_entries_for_config_entry",
        return_value=registry_entries,
    ):
        removed = cleanup_orphaned_diagnostic_entities(hass, manager, entry)

    assert removed == 3
    removed_ids = [call.args[0] for call in ent_reg.async_remove.call_args_list]
    assert "sensor.debug_info" not in removed_ids
    assert "sensor.top_candidates" in removed_ids
    assert "sensor.match_confidence" in removed_ids
    assert "binary_sensor.ambiguity" in removed_ids


def test_cleanup_keeps_debug_diagnostics_when_debug_exposed() -> None:
    """Debug diagnostics are preserved when debug entities are enabled."""
    entry = MagicMock()
    entry.entry_id = "entry1"
    entry.options = {"expose_debug_entities": True}

    manager = _build_manager_with_profiles([])
    hass = MagicMock()

    registry_entries = [
        _mock_registry_entry("sensor.debug_info", "entry1_debug_info"),
        _mock_registry_entry("sensor.top_candidates", "entry1_top_candidates"),
        _mock_registry_entry("sensor.match_confidence", "entry1_match_confidence"),
        _mock_registry_entry("binary_sensor.ambiguity", "entry1_ambiguity"),
        _mock_registry_entry("sensor.suggestions", "entry1_suggestions"),
    ]

    ent_reg = MagicMock()
    with patch(
        "custom_components.ha_washdata.sensor.entity_registry.async_get",
        return_value=ent_reg,
    ), patch(
        "custom_components.ha_washdata.sensor.entity_registry.async_entries_for_config_entry",
        return_value=registry_entries,
    ):
        removed = cleanup_orphaned_diagnostic_entities(hass, manager, entry)

    assert removed == 0
    ent_reg.async_remove.assert_not_called()
