"""Sensors for HA WashData."""

from __future__ import annotations

from homeassistant.components.sensor import SensorEntity, SensorEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.const import EntityCategory
from homeassistant.helpers import entity_registry
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import dt as dt_util, slugify

from .const import DOMAIN, SIGNAL_WASHER_UPDATE, CONF_EXPOSE_DEBUG_ENTITIES
from .manager import WashDataManager


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the sensors."""
    manager: WashDataManager = hass.data[DOMAIN][entry.entry_id]

    entities = [
        WasherStateSensor(manager, entry),
        WasherProgramSensor(manager, entry),
        WasherTimeRemainingSensor(manager, entry),
        WasherProgressSensor(manager, entry),
        WasherPowerSensor(manager, entry),
        WasherElapsedTimeSensor(manager, entry),
        WasherDebugSensor(manager, entry),
    ]

    # Add debug entities if enabled
    if entry.options.get(CONF_EXPOSE_DEBUG_ENTITIES):
        entities.extend(
            [
                WasherMatchConfidenceSensor(manager, entry),
                WasherTopCandidatesSensor(manager, entry),
                WasherPhaseSensor(manager, entry),
            ]
        )

    async_add_entities(entities)

    # Initialize dynamic profile sensor manager
    profile_sensor_manager = WasherProfileSensorManager(manager, entry, async_add_entities)
    await profile_sensor_manager.async_update()


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
            "manufacturer": "HA WashData",
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

    def __init__(self, manager, entry):
        """Initialize the state sensor."""
        self.entity_description = SensorEntityDescription(
            key="washer_state", name="State", icon="mdi:washing-machine"
        )
        super().__init__(manager, entry)

    @property
    def native_value(self):
        return self._manager.check_state

    @property
    def extra_state_attributes(self):
        return {
            "samples_recorded": self._manager.samples_recorded,
            "current_program_guess": self._manager.current_program,
            "sub_state": self._manager.sub_state,
        }


class WasherProgramSensor(WasherBaseSensor):
    """Sensor for the current program."""

    def __init__(self, manager, entry):
        """Initialize the program sensor."""
        self.entity_description = SensorEntityDescription(
            key="washer_program", name="Program", icon="mdi:file-document-outline"
        )
        super().__init__(manager, entry)

    @property
    def native_value(self):
        return self._manager.current_program


class WasherTimeRemainingSensor(WasherBaseSensor):
    """Sensor for estimated time remaining."""

    def __init__(self, manager, entry):
        """Initialize the time remaining sensor."""
        self.entity_description = SensorEntityDescription(
            key="time_remaining",
            name="Time Remaining",
            # native_unit_of_measurement="min",  # Removed static unit
            icon="mdi:timer-sand",
        )
        super().__init__(manager, entry)

    @property
    def native_unit_of_measurement(self) -> str | None:
        """Return the unit of measurement."""
        if self._manager.check_state == "off":
            return None
        return "min"

    @property
    def native_value(self):
        if self._manager.check_state == "off":
            return "off"
        if self._manager.time_remaining:
            return int(self._manager.time_remaining / 60)
        return None


class WasherProgressSensor(WasherBaseSensor):
    """Sensor for cycle progress percentage."""

    def __init__(self, manager, entry):
        """Initialize the progress sensor."""
        self.entity_description = SensorEntityDescription(
            key="cycle_progress",
            name="Progress",
            native_unit_of_measurement="%",
            suggested_display_precision=1,
            icon="mdi:progress-clock",
        )
        super().__init__(manager, entry)

    @property
    def native_value(self):
        return self._manager.cycle_progress


class WasherPowerSensor(WasherBaseSensor):
    """Sensor for current power usage."""

    def __init__(self, manager, entry):
        """Initialize the power sensor."""
        self.entity_description = SensorEntityDescription(
            key="current_power",
            name="Current Power",
            native_unit_of_measurement="W",
            device_class="power",
            icon="mdi:flash",
        )
        super().__init__(manager, entry)

    @property
    def native_value(self):
        return self._manager.current_power


class WasherElapsedTimeSensor(WasherBaseSensor):
    """Sensor for elapsed cycle time."""

    def __init__(self, manager, entry):
        """Initialize the elapsed time sensor."""
        self.entity_description = SensorEntityDescription(
            key="elapsed_time",
            name="Elapsed Time",
            native_unit_of_measurement="s",
            device_class="duration",
            icon="mdi:timer-outline",
        )
        super().__init__(manager, entry)

    @property
    def native_value(self):
        if self._manager.check_state == "off":
            return 0
        start = self._manager.cycle_start_time
        if start:
            delta = dt_util.now() - start
            return int(delta.total_seconds())
        return 0


class WasherDebugSensor(WasherBaseSensor):
    """Sensor for internal debug information."""

    def __init__(self, manager, entry):
        """Initialize the debug sensor."""
        self.entity_description = SensorEntityDescription(
            key="debug_info",
            name="Debug Info",
            icon="mdi:bug",
            entity_registry_enabled_default=False,  # Hidden by default
            entity_category=EntityCategory.DIAGNOSTIC,
        )
        super().__init__(manager, entry)

    @property
    def native_value(self):
        return self._manager.check_state

    @property
    def extra_state_attributes(self):
        """Return various internal states for debugging."""
        detector = self._manager.detector
        stats = self._manager.sample_interval_stats
        # pylint: disable=protected-access
        return {
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


class WasherMatchConfidenceSensor(WasherBaseSensor):
    """Sensor for profile match confidence."""

    def __init__(self, manager, entry):
        self.entity_description = SensorEntityDescription(
            key="match_confidence",
            name="Match Confidence",
            icon="mdi:chart-bar",
            state_class="measurement",
            native_unit_of_measurement="%",
            entity_category=EntityCategory.DIAGNOSTIC,
        )
        super().__init__(manager, entry)

    @property
    def native_value(self):
        conf = getattr(self._manager, "_last_match_confidence", 0.0)
        return int(conf * 100)


class WasherTopCandidatesSensor(WasherBaseSensor):
    """Sensor showing top matching candidates."""

    def __init__(self, manager, entry):
        self.entity_description = SensorEntityDescription(
            key="top_candidates",
            name="Top Candidates",
            icon="mdi:format-list-numbered",
            entity_category=EntityCategory.DIAGNOSTIC,
        )
        super().__init__(manager, entry)

    @property
    def native_value(self):
        candidates = self._manager.top_candidates
        if not candidates:
            return "None"
        # Return simplified string: "Name (Score), Name (Score)"
        return ", ".join([f"{c['name']} ({c['score']:.2f})" for c in candidates[:3]])

    @property
    def extra_state_attributes(self):
        return {"candidates": self._manager.top_candidates}


class WasherPhaseSensor(WasherBaseSensor):
    """Sensor for current wash phase."""

    def __init__(self, manager, entry):
        self.entity_description = SensorEntityDescription(
            key="wash_phase",
            name="Phase",
            icon="mdi:washing-machine-alert",
            entity_category=EntityCategory.DIAGNOSTIC,
        )
        super().__init__(manager, entry)

    @property
    def native_value(self):
        return self._manager.phase_description


class WasherProfileCountSensor(WasherBaseSensor):
    """Diagnostic sensor showing cycle count for a specific profile."""

    def __init__(
        self, manager: WashDataManager, entry: ConfigEntry, profile_name: str, count: int
    ) -> None:
        """Initialize."""
        self._profile_name = profile_name
        self._safe_name = slugify(profile_name)
        # We store initial count, but update callback will refresh it
        self._count = count
        
        self.entity_description = SensorEntityDescription(
            key=f"profile_count_{self._safe_name}",
            name=f"Profile: {profile_name} Count",
            icon="mdi:counter",
            native_unit_of_measurement="cycles",
            state_class="total",
            entity_category=EntityCategory.DIAGNOSTIC,
        )
        super().__init__(manager, entry)
        # Override unique ID to be profile specific
        self._attr_unique_id = f"{entry.entry_id}_profile_count_{self._safe_name}"

    @property
    def native_value(self) -> int:
        """Return the cycle count."""
        # Fetch fresh count from store if available
        profile = self._manager.profile_store.get_profile(self._profile_name)
        if profile:
            return profile.get("cycle_count", 0)
        return 0

    @property
    def available(self) -> bool:
        """Return True if profile still exists."""
        return self._manager.profile_store.get_profile(self._profile_name) is not None

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        """Return profile statistics."""
        profile = self._manager.profile_store.get_profile(self._profile_name)
        if not profile:
            return None
            
        avg_energy = profile.get("avg_energy")
        count = profile.get("cycle_count", 0)
        total_energy = (avg_energy * count) if avg_energy is not None else None
        
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
        
        # Determine the signal string. It must match SIGNAL_WASHER_UPDATE from const.py
        # which is "washdata_update_{}"
        self._signal = SIGNAL_WASHER_UPDATE.format(entry.entry_id)

        # Register callback for ALL updates (simplest hook we have)
        # Ideally we'd have a specific profile update signal, but general update is fine
        # as long as we debounce or check efficiently.
        async_dispatcher_connect(
            manager.hass,
            self._signal,
            self._update_callback,
        )

    @callback
    def _update_callback(self) -> None:
        """Handle updates."""
        self._manager.hass.async_create_task(self.async_update())

    async def async_update(self) -> None:
        """Reflect profile changes in sensors."""
        profiles = self._manager.profile_store.list_profiles()
        current_names = {p["name"] for p in profiles}
        existing_names = set(self._sensors.keys())

        # Add new
        new_names = current_names - existing_names
        new_entities = []
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
                        # Fallback for non-registered entities
                        await sensor.async_remove()

