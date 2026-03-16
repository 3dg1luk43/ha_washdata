"""Sensors for WashData."""

from __future__ import annotations

from asyncio import Task
import hashlib
import logging
from typing import Any

from homeassistant.components.sensor import SensorEntity, SensorEntityDescription, SensorDeviceClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.const import EntityCategory
from homeassistant.helpers import entity_registry
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import dt as dt_util

from .const import (
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_DURATION_TOLERANCE,
    DOMAIN,
    CONF_MIN_OFF_GAP,
    CONF_MIN_POWER,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_OFF_DELAY,
    CONF_PROFILE_DURATION_TOLERANCE,
    CONF_PROFILE_MATCH_INTERVAL,
    CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
    CONF_SAMPLING_INTERVAL,
    SIGNAL_WASHER_UPDATE,
    CONF_WATCHDOG_INTERVAL,
    CONF_EXPOSE_DEBUG_ENTITIES,
    STATE_OFF,
    STATE_IDLE,
    STATE_STARTING,
    STATE_RUNNING,
    STATE_PAUSED,
    STATE_ENDING,
    STATE_FINISHED,
    STATE_ANTI_WRINKLE,
    STATE_INTERRUPTED,
    STATE_FORCE_STOPPED,
    STATE_RINSE,
    STATE_UNKNOWN,
)
from .manager import WashDataManager

_LOGGER = logging.getLogger(__name__)


_STATIC_DIAGNOSTIC_SUFFIXES = {
    "debug_info",
    "suggestions",
    "match_confidence",
    "top_candidates",
    "ambiguity",
}


def _profile_count_unique_id(entry_id: str, profile_name: str) -> str:
    """Build deterministic unique_id for a profile count diagnostic sensor."""
    profile_token = hashlib.sha256(profile_name.encode("utf-8")).hexdigest()[:8]
    return f"{entry_id}_profile_count_{profile_token}"


def _expected_diagnostic_unique_ids(manager: WashDataManager, entry: ConfigEntry) -> set[str]:
    """Return expected diagnostic unique_ids for this config entry."""
    expected = {
        f"{entry.entry_id}_debug_info",
        f"{entry.entry_id}_suggestions",
    }

    if entry.options.get(CONF_EXPOSE_DEBUG_ENTITIES):
        expected.update(
            {
                f"{entry.entry_id}_match_confidence",
                f"{entry.entry_id}_top_candidates",
                f"{entry.entry_id}_ambiguity",
            }
        )

    for profile in manager.profile_store.list_profiles():
        profile_name = profile.get("name")
        if isinstance(profile_name, str) and profile_name:
            expected.add(_profile_count_unique_id(entry.entry_id, profile_name))

    return expected


def cleanup_orphaned_diagnostic_entities(
    hass: HomeAssistant, manager: WashDataManager, entry: ConfigEntry
) -> int:
    """Remove stale diagnostic entities for this config entry from entity registry."""
    ent_reg = entity_registry.async_get(hass)
    expected_unique_ids = _expected_diagnostic_unique_ids(manager, entry)

    removed = 0
    entry_prefix = f"{entry.entry_id}_"
    for reg_entry in entity_registry.async_entries_for_config_entry(ent_reg, entry.entry_id):
        unique_id = reg_entry.unique_id or ""
        if not unique_id.startswith(entry_prefix):
            continue

        suffix = unique_id[len(entry_prefix) :]
        is_diagnostic_family = (
            suffix in _STATIC_DIAGNOSTIC_SUFFIXES
            or suffix.startswith("profile_count_")
            or suffix == "wash_phase"
        )
        if not is_diagnostic_family:
            continue

        if unique_id not in expected_unique_ids:
            ent_reg.async_remove(reg_entry.entity_id)
            removed += 1

    if removed:
        _LOGGER.info(
            "Removed %s orphaned diagnostic entities for entry %s",
            removed,
            entry.entry_id,
        )
    return removed


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the sensors."""
    manager: WashDataManager = hass.data[DOMAIN][entry.entry_id]

    entities: list[SensorEntity] = [
        WasherStateSensor(manager, entry),
        WasherProgramSensor(manager, entry),
        WasherCurrentPhaseSensor(manager, entry),
        WasherTimeRemainingSensor(manager, entry),
        WasherTotalDurationSensor(manager, entry),
        WasherProgressSensor(manager, entry),
        WasherPowerSensor(manager, entry),
        WasherElapsedTimeSensor(manager, entry),
        WasherDebugSensor(manager, entry),
        WasherSuggestionsSensor(manager, entry),
    ]

    # Add debug entities if enabled
    if entry.options.get(CONF_EXPOSE_DEBUG_ENTITIES):
        entities.extend(
            [
                WasherMatchConfidenceSensor(manager, entry),
                WasherTopCandidatesSensor(manager, entry),
            ]
        )

    async_add_entities(entities)

    # Reconcile diagnostics at startup so stale unavailable entries are auto-removed.
    cleanup_orphaned_diagnostic_entities(hass, manager, entry)

    # Initialize dynamic profile sensor manager
    profile_sensor_manager = WasherProfileSensorManager(manager, entry, async_add_entities)
    await profile_sensor_manager.async_update()
    entry.async_on_unload(profile_sensor_manager.unsubscribe)


class WasherBaseSensor(SensorEntity):
    """Base sensor for ha_washdata."""

    _attr_has_entity_name = True

    def __init__(self, manager: WashDataManager, entry: ConfigEntry) -> None:
        """Initialize."""
        self._manager = manager
        self._entry = entry
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": entry.title,
            "manufacturer": "WashData",
        }
        self._attr_unique_id = f"{entry.entry_id}_{self.entity_description.key}"

    async def async_added_to_hass(self) -> None:
        """Register callbacks."""
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                SIGNAL_WASHER_UPDATE.format(self._entry.entry_id),
                self._update_callback,
            )
        )

    @callback
    def _update_callback(self) -> None:
        """Update the sensor."""
        self.async_write_ha_state()


class WasherStateSensor(WasherBaseSensor):
    """Sensor for the washing machine state."""

    def __init__(self, manager: WashDataManager, entry: ConfigEntry) -> None:
        """Initialize the state sensor."""
        self.entity_description = SensorEntityDescription(
            key="washer_state",
            translation_key="washer_state",
            device_class=SensorDeviceClass.ENUM,
            options=[
                STATE_OFF,
                STATE_IDLE,
                STATE_STARTING,
                STATE_RUNNING,
                STATE_PAUSED,
                STATE_ENDING,
                STATE_FINISHED,
                STATE_ANTI_WRINKLE,
                STATE_INTERRUPTED,
                STATE_FORCE_STOPPED,
                STATE_RINSE,
                STATE_UNKNOWN,
            ],
        )
        super().__init__(manager, entry)

    @property
    def icon(self) -> str | None:  # type: ignore[override]
        """Return the icon."""
        dtype = self._manager.device_type
        if dtype == "dryer":
            return "mdi:tumble-dryer"
        if dtype == "dishwasher":
            return "mdi:dishwasher"
        if dtype == "ev":
            return "mdi:car-electric"
        if dtype == "coffee_machine":
            return "mdi:coffee-maker"
        if dtype == "air_fryer":
            return "mdi:pot-steam"
        if dtype == "heat_pump":
            return "mdi:heat-pump"
        return "mdi:washing-machine"

    @property
    def native_value(self):  # type: ignore[override]
        return self._manager.check_state()

    @property
    def extra_state_attributes(self):  # type: ignore[override]
        attrs: dict[str, Any] = {
            "samples_recorded": self._manager.samples_recorded,
            "current_program_guess": self._manager.current_program,
            "sub_state": self._manager.sub_state,
        }
        return attrs


class WasherProgramSensor(WasherBaseSensor):
    """Sensor for the current program."""

    def __init__(self, manager: WashDataManager, entry: ConfigEntry) -> None:
        """Initialize the program sensor."""
        self.entity_description = SensorEntityDescription(
            key="washer_program",
            translation_key="washer_program",
            icon="mdi:file-document-outline",
            device_class=SensorDeviceClass.ENUM,
        )
        super().__init__(manager, entry)

    @property
    def options(self) -> list[str] | None:  # type: ignore[override]
        """Return a list of possible options."""
        profiles = self._manager.profile_store.list_profiles()
        # Include current program if not in profiles (e.g. unknown or special states)
        options = [p["name"] for p in profiles]
        curr = self._manager.current_program
        if curr and curr not in options:
            options.append(curr)
        if "none" not in options:
            options.append("none")
        if "unknown" not in options:
            options.append("unknown")
        return options

    @property
    def native_value(self):  # type: ignore[override]
        return self._manager.current_program

    @property
    def extra_state_attributes(self):  # type: ignore[override]
        profile_name = self._manager.current_program
        if not profile_name or profile_name in ("off", "detecting...", "starting", "unknown"):
            return None

        device_type = self._manager.device_type
        if device_type:
            catalog = self._manager.list_phase_catalog(device_type)
            assigned = self._manager.get_profile_phase_ranges_for_device(
                profile_name,
                device_type,
            )
        else:
            catalog = []
            assigned = []
        catalog_view: list[dict[str, Any]] = [
            {
                "name": p.get("name"),
                "description": p.get("description", ""),
                "is_default": bool(p.get("is_default", False)),
            }
            for p in catalog
        ]

        attrs: dict[str, Any] = {
            "active_phase": self._manager.phase_description,
            "phase_catalog": catalog_view,
            "phase_ranges": assigned,
        }
        return attrs


class WasherTimeRemainingSensor(WasherBaseSensor):
    """Sensor for estimated time remaining."""

    def __init__(self, manager: WashDataManager, entry: ConfigEntry) -> None:
        """Initialize the time remaining sensor."""
        self.entity_description = SensorEntityDescription(
            key="time_remaining",
            translation_key="time_remaining",
            icon="mdi:timer-sand",
        )
        super().__init__(manager, entry)

    @property
    def native_unit_of_measurement(self) -> str | None:  # type: ignore[override]
        """Return the unit of measurement."""
        if self._manager.check_state() in (STATE_OFF, STATE_ANTI_WRINKLE):
            return None
        return "min"

    @property
    def native_value(self):  # type: ignore[override]
        if self._manager.check_state() in (STATE_OFF, STATE_ANTI_WRINKLE):
            return None
        if self._manager.time_remaining is not None:
            return int(self._manager.time_remaining / 60)
        return None


class WasherTotalDurationSensor(WasherBaseSensor):
    """Sensor for total predicted duration."""

    def __init__(self, manager: WashDataManager, entry: ConfigEntry) -> None:
        """Initialize the total duration sensor."""
        self.entity_description = SensorEntityDescription(
            key="total_duration",
            translation_key="total_duration",
            device_class=SensorDeviceClass.DURATION,
            icon="mdi:timer-check-outline",
        )
        super().__init__(manager, entry)

    @property
    def native_unit_of_measurement(self) -> str | None:  # type: ignore[override]
        """Return the unit of measurement."""
        if self._manager.check_state() == STATE_OFF:
            return None
        return "min"

    @property
    def native_value(self):  # type: ignore[override]
        if self._manager.check_state() == STATE_OFF:
            return None
        if self._manager.total_duration:
            return int(self._manager.total_duration / 60)
        return None

    @property
    def extra_state_attributes(self):  # type: ignore[override]
        """Return extra state attributes."""
        return {
            "last_updated": self._manager.last_total_duration_update,
        }


class WasherProgressSensor(WasherBaseSensor):
    """Sensor for cycle progress percentage."""

    def __init__(self, manager: WashDataManager, entry: ConfigEntry) -> None:
        """Initialize the progress sensor."""
        self.entity_description = SensorEntityDescription(
            key="cycle_progress",
            translation_key="cycle_progress",
            native_unit_of_measurement="%",
            suggested_display_precision=1,
            icon="mdi:progress-clock",
        )
        super().__init__(manager, entry)

    @property
    def native_value(self):  # type: ignore[override]
        return self._manager.cycle_progress


class WasherPowerSensor(WasherBaseSensor):
    """Sensor for current power usage."""

    def __init__(self, manager: WashDataManager, entry: ConfigEntry) -> None:
        """Initialize the power sensor."""
        self.entity_description = SensorEntityDescription(
            key="current_power",
            translation_key="current_power",
            native_unit_of_measurement="W",
            device_class=SensorDeviceClass.POWER,
            icon="mdi:flash",
        )
        super().__init__(manager, entry)

    @property
    def native_value(self):  # type: ignore[override]
        return self._manager.current_power


class WasherElapsedTimeSensor(WasherBaseSensor):
    """Sensor for elapsed cycle time."""

    def __init__(self, manager: WashDataManager, entry: ConfigEntry) -> None:
        """Initialize the elapsed time sensor."""
        self.entity_description = SensorEntityDescription(
            key="elapsed_time",
            translation_key="elapsed_time",
            native_unit_of_measurement="s",
            device_class=SensorDeviceClass.DURATION,
            icon="mdi:timer-outline",
        )
        super().__init__(manager, entry)

    @property
    def native_value(self):  # type: ignore[override]
        if self._manager.check_state() == STATE_OFF:
            return 0
        start = self._manager.cycle_start_time
        if start:
            delta = dt_util.now() - start
            return int(delta.total_seconds())
        return 0


class WasherDebugSensor(WasherBaseSensor):
    """Sensor for internal debug information."""

    def __init__(self, manager: WashDataManager, entry: ConfigEntry) -> None:
        """Initialize the debug sensor."""
        self.entity_description = SensorEntityDescription(
            key="debug_info",
            translation_key="debug_info",
            icon="mdi:bug",
            entity_registry_enabled_default=False,  # Hidden by default
            entity_category=EntityCategory.DIAGNOSTIC,
        )
        super().__init__(manager, entry)

    @property
    def native_value(self):  # type: ignore[override]
        return self._manager.check_state()

    @property
    def extra_state_attributes(self):  # type: ignore[override]
        """Return various internal states for debugging."""
        detector = self._manager.detector
        stats = self._manager.sample_interval_stats
        # pylint: disable=protected-access
        attrs: dict[str, Any] = {
            "sub_state": detector.sub_state,
            "match_confidence": getattr(self._manager, "_last_match_confidence", 0.0),
            "cycle_id": getattr(detector, "_current_cycle_start", None),
            "samples": detector.samples_recorded,
            "energy_accum": getattr(detector, "_energy_since_idle_wh", 0.0),
            "time_below": getattr(detector, "_time_below_threshold", 0.0),
            "sampling_p95": stats.get("p95"),
            "noise_events": len(getattr(self._manager, "_noise_events", [])),
            "top_candidates": self._manager.top_candidates,
            "last_match_details": self._manager.last_match_details,
        }
        return attrs


class WasherMatchConfidenceSensor(WasherBaseSensor):
    """Sensor for profile match confidence."""

    def __init__(self, manager: WashDataManager, entry: ConfigEntry) -> None:
        self.entity_description = SensorEntityDescription(
            key="match_confidence",
            translation_key="match_confidence",
            icon="mdi:chart-bar",
            state_class="measurement",
            native_unit_of_measurement="%",
            entity_category=EntityCategory.DIAGNOSTIC,
        )
        super().__init__(manager, entry)

    @property
    def native_value(self):  # type: ignore[override]
        conf = getattr(self._manager, "_last_match_confidence", 0.0)
        return int(conf * 100)


class WasherTopCandidatesSensor(WasherBaseSensor):
    """Sensor showing top matching candidates."""

    def __init__(self, manager: WashDataManager, entry: ConfigEntry) -> None:
        self.entity_description = SensorEntityDescription(
            key="top_candidates",
            translation_key="top_candidates",
            icon="mdi:format-list-numbered",
            entity_category=EntityCategory.DIAGNOSTIC,
        )
        super().__init__(manager, entry)

    @property
    def native_value(self):  # type: ignore[override]
        candidates = self._manager.top_candidates
        if not candidates:
            return "none"
        # Return simplified string: "Name (Score), Name (Score)"
        return ", ".join([f"{c['name']} ({c['score']:.2f})" for c in candidates[:3]])

    @property
    def extra_state_attributes(self):  # type: ignore[override]
        return {"candidates": self._manager.top_candidates}


class WasherCurrentPhaseSensor(WasherBaseSensor):
    """Sensor for the current detected phase."""

    def __init__(self, manager: WashDataManager, entry: ConfigEntry) -> None:
        self.entity_description = SensorEntityDescription(
            key="current_phase",
            translation_key="current_phase",
            icon="mdi:water-sync",
        )
        super().__init__(manager, entry)

    @property
    def native_value(self):  # type: ignore[override]
        return self._manager.phase_description


class WasherProfileCountSensor(WasherBaseSensor):
    """Diagnostic sensor showing cycle count for a specific profile."""

    def __init__(
        self, manager: WashDataManager, entry: ConfigEntry, profile_name: str, count: int
    ) -> None:
        """Initialize."""
        self._profile_name = profile_name
        self._profile_token = hashlib.sha256(
            profile_name.encode("utf-8")
        ).hexdigest()[:8]
        # We store initial count, but update callback will refresh it
        self._count = count

        self.entity_description = SensorEntityDescription(
            key=f"profile_count_{self._profile_token}",
            translation_key="profile_cycle_count",
            icon="mdi:counter",
            state_class="total",
            entity_category=EntityCategory.DIAGNOSTIC,
        )
        self._attr_translation_placeholders = {"profile_name": profile_name}
        super().__init__(manager, entry)
        # Override unique ID to be profile specific
        self._attr_unique_id = f"{entry.entry_id}_profile_count_{self._profile_token}"

    @property
    def native_value(self) -> int:  # type: ignore[override]
        """Return the cycle count."""
        # Fetch fresh count from store if available
        profile = self._manager.profile_store.get_profile(self._profile_name)
        if profile:
            return profile.get("cycle_count", 0)
        return 0

    @property
    def available(self) -> bool:  # type: ignore[override]
        """Return True if profile still exists."""
        return self._manager.profile_store.get_profile(self._profile_name) is not None

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:  # type: ignore[override]
        """Return profile statistics."""
        profile = self._manager.profile_store.get_profile(self._profile_name)
        if not profile:
            return None

        avg_energy = profile.get("avg_energy")
        count = profile.get("cycle_count", 0)
        total_energy = (avg_energy * count) if avg_energy is not None else None
        duration_std_dev = profile.get("duration_std_dev")
        consistency_min = (
            float(duration_std_dev) / 60.0
            if isinstance(duration_std_dev, (int, float))
            else None
        )

        # Helper to format duration
        def _to_min(sec: float) -> int:
            return int(sec / 60) if sec else 0

        return {
            "average_consumption_kwh": avg_energy,
            "total_consumption_kwh": total_energy,
            "last_run": profile.get("last_run"),
            "average_length_min": _to_min(profile.get("avg_duration", 0)),
            "min_length_min": _to_min(profile.get("min_duration", 0)),
            "max_length_min": _to_min(profile.get("max_duration", 0)),
            "consistency_min": consistency_min,
        }


class WasherProfileSensorManager:
    """Manages dynamic profile sensors."""

    def __init__(
        self,
        manager: WashDataManager,
        entry: ConfigEntry,
        async_add_entities: AddEntitiesCallback,
    ) -> None:
        """Initialize."""
        self._manager = manager
        self._entry = entry
        self._async_add_entities = async_add_entities
        self._sensors: dict[str, WasherProfileCountSensor] = {}
        self._diagnostics_cleanup_done: bool = False

        # Determine the signal string. It must match SIGNAL_WASHER_UPDATE from const.py
        # which is "washdata_update_{}"
        self._signal = SIGNAL_WASHER_UPDATE.format(entry.entry_id)
        self._update_task: Task[None] | None = None
        self._pending_update: bool = False

        # Register callback for ALL updates (simplest hook we have)
        # Ideally we'd have a specific profile update signal, but general update is fine
        # as long as we debounce or check efficiently.
        self._unsub_dispatcher = async_dispatcher_connect(
            manager.hass,
            self._signal,
            self._update_callback,
        )

        # Handle stale diagnostics that were left in registry by previous naming schemes
        # or profile renames. Run once at initialization instead of on every update.
        cleanup_orphaned_diagnostic_entities(
            self._manager.hass, self._manager, self._entry
        )
        self._diagnostics_cleanup_done = True

    def unsubscribe(self) -> None:
        """Remove the dispatcher subscription."""
        if self._unsub_dispatcher:
            self._unsub_dispatcher()
            self._unsub_dispatcher = None

        # Prevent queued follow-up refreshes after teardown.
        self._pending_update = False
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()
        self._update_task = None

    @callback
    def _update_callback(self) -> None:
        """Handle updates."""
        if self._update_task and not self._update_task.done():
            self._pending_update = True
            return

        task = self._manager.hass.async_create_task(self.async_update())
        self._update_task = task

        def _clear_update_task(done_task: Task[Any]) -> None:
            if self._update_task is done_task:
                self._update_task = None
                if self._unsub_dispatcher is not None and self._pending_update:
                    self._pending_update = False
                    follow = self._manager.hass.async_create_task(self.async_update())
                    self._update_task = follow
                    follow.add_done_callback(_clear_update_task)

        task.add_done_callback(_clear_update_task)

    async def async_update(self) -> None:
        """Reflect profile changes in sensors."""
        profiles = self._manager.profile_store.list_profiles()
        current_names = {p["name"] for p in profiles}
        existing_names = set(self._sensors.keys())

        # Add new
        new_names = current_names - existing_names
        new_entities: list[SensorEntity] = []
        for name in new_names:
            p_data = self._manager.profile_store.get_profile(name)
            count = p_data.get("cycle_count", 0) if p_data else 0
            sensor = WasherProfileCountSensor(self._manager, self._entry, name, count)
            self._sensors[name] = sensor
            new_entities.append(sensor)

        if new_entities:
            self._async_add_entities(new_entities)

        # Remove old (if profile deleted)
        removed_names = existing_names - current_names

        if removed_names:
            ent_reg = entity_registry.async_get(self._manager.hass)
            for name in removed_names:
                sensor = self._sensors.pop(name)
                # Remove from Entity Registry if registered
                if sensor.entity_id:
                    if ent_reg.async_get(sensor.entity_id):
                        ent_reg.async_remove(sensor.entity_id)
                    else:
                        # Fallback for non-registered entities that were attached.
                        if sensor.hass:
                            try:
                                await sensor.async_remove()
                            except Exception as err:  # pylint: disable=broad-exception-caught
                                _LOGGER.debug(
                                    "Failed to remove sensor '%s' via fallback path: %s",
                                    name,
                                    err,
                                )


class WasherSuggestionsSensor(WasherBaseSensor):
    """Sensor for learned settings suggestions."""

    def __init__(self, manager: WashDataManager, entry: ConfigEntry) -> None:
        self.entity_description = SensorEntityDescription(
            key="suggestions",
            translation_key="suggestions",
            icon="mdi:lightbulb-on-outline",
            entity_category=EntityCategory.DIAGNOSTIC,
        )
        super().__init__(manager, entry)

    @staticmethod
    def _applicable_suggestion_keys() -> tuple[str, ...]:
        """Return suggestion keys that can be applied in options flow."""
        return (
            CONF_MIN_POWER,
            CONF_OFF_DELAY,
            CONF_WATCHDOG_INTERVAL,
            CONF_NO_UPDATE_ACTIVE_TIMEOUT,
            CONF_SAMPLING_INTERVAL,
            CONF_PROFILE_MATCH_INTERVAL,
            CONF_AUTO_LABEL_CONFIDENCE,
            CONF_DURATION_TOLERANCE,
            CONF_PROFILE_DURATION_TOLERANCE,
            CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
            CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
            CONF_MIN_OFF_GAP,
        )

    def _count_applicable_suggestions(self, suggestions: dict[str, Any]) -> int:
        """Count only suggestions with values that can be applied from options flow."""
        count = 0
        for key in self._applicable_suggestion_keys():
            entry = suggestions.get(key)
            if isinstance(entry, dict) and entry.get("value") is not None:
                count += 1
        return count

    @property
    def native_value(self):  # type: ignore[override]
        suggestions = self._manager.suggestions
        if not suggestions:
            return 0
        return self._count_applicable_suggestions(suggestions)

    @property
    def extra_state_attributes(self):  # type: ignore[override]
        suggestions: dict[str, Any] = self._manager.suggestions or {}
        count = self._count_applicable_suggestions(suggestions)
        applicable_keys = sorted(
            k for k in self._applicable_suggestion_keys()
            if isinstance(suggestions.get(k), dict) and suggestions[k].get("value") is not None
        )

        attrs: dict[str, Any] = {
            "has_actionable_suggestions": count > 0,
            "suggestions_count": count,
            "suggested_option_keys": applicable_keys,
            "suggestions": suggestions,
        }
        return attrs