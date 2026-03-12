"""Phase catalog defaults and helpers for WashData."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .const import (
    DEVICE_TYPE_AIR_FRYER,
    DEVICE_TYPE_COFFEE_MACHINE,
    DEVICE_TYPE_DISHWASHER,
    DEVICE_TYPE_DRYER,
    DEVICE_TYPE_EV,
    DEVICE_TYPE_HEAT_PUMP,
    DEVICE_TYPE_WASHER_DRYER,
    DEVICE_TYPE_WASHING_MACHINE,
)

PhaseItem = dict[str, Any]

DEFAULT_PHASES_BY_DEVICE: dict[str, list[PhaseItem]] = {
    DEVICE_TYPE_WASHING_MACHINE: [
        {
            "name": "Pre-Wash",
            "description": "Initial soak or pre-treatment before the main wash.",
            "is_default": True,
        },
        {
            "name": "Wash",
            "description": "Main washing cycle with drum movement and optional heating.",
            "is_default": True,
        },
        {
            "name": "Rinse",
            "description": "Clean-water rinse stage. This phase may repeat multiple times.",
            "is_default": True,
        },
        {
            "name": "Spin",
            "description": "High-speed extraction to remove water from the load.",
            "is_default": True,
        },
        {
            "name": "Soak",
            "description": "Low-activity soaking period between active wash stages.",
            "is_default": True,
        },
        {
            "name": "Anti-Crease",
            "description": "Occasional short tumbles after completion to reduce wrinkles.",
            "is_default": True,
        },
    ],
    DEVICE_TYPE_DRYER: [
        {
            "name": "Heat Up",
            "description": "Initial heater warm-up before full drying begins.",
            "is_default": True,
        },
        {
            "name": "Drying",
            "description": "Main heated tumbling period.",
            "is_default": True,
        },
        {
            "name": "Cool Down",
            "description": "Tumbling without heat near cycle end.",
            "is_default": True,
        },
        {
            "name": "Anti-Wrinkle",
            "description": "Periodic post-cycle tumbling to reduce wrinkles.",
            "is_default": True,
        },
        {
            "name": "Sensor Check",
            "description": "Short low-power pause while dryness is measured.",
            "is_default": True,
        },
    ],
    DEVICE_TYPE_WASHER_DRYER: [
        {
            "name": "Pre-Wash",
            "description": "Initial soak or pre-treatment before the main wash.",
            "is_default": True,
        },
        {
            "name": "Wash",
            "description": "Main washing cycle with drum movement and optional heating.",
            "is_default": True,
        },
        {
            "name": "Rinse",
            "description": "Clean-water rinse stage. This phase may repeat multiple times.",
            "is_default": True,
        },
        {
            "name": "Spin",
            "description": "High-speed extraction before drying transition.",
            "is_default": True,
        },
        {
            "name": "Drain & Switch",
            "description": "Transition period from washing to drying mode.",
            "is_default": True,
        },
        {
            "name": "Heat Up",
            "description": "Initial heater warm-up before full drying begins.",
            "is_default": True,
        },
        {
            "name": "Drying",
            "description": "Main heated tumbling period.",
            "is_default": True,
        },
        {
            "name": "Cool Down",
            "description": "Tumbling without heat near cycle end.",
            "is_default": True,
        },
        {
            "name": "Anti-Wrinkle",
            "description": "Periodic post-cycle tumbling to reduce wrinkles.",
            "is_default": True,
        },
    ],
    DEVICE_TYPE_DISHWASHER: [
        {
            "name": "Pre-Rinse",
            "description": "Initial spray-down before detergent wash.",
            "is_default": True,
        },
        {
            "name": "Wash",
            "description": "Main detergent wash with heating.",
            "is_default": True,
        },
        {
            "name": "Rinse",
            "description": "Clean-water rinse stage. This phase may repeat multiple times.",
            "is_default": True,
        },
        {
            "name": "Dry",
            "description": "Drying stage using heater and/or residual heat.",
            "is_default": True,
        },
        {
            "name": "Sanitize",
            "description": "High-temperature cleaning stage for sanitization programs.",
            "is_default": True,
        },
        {
            "name": "Soak",
            "description": "Extended soak period for heavy soil.",
            "is_default": True,
        },
    ],
    DEVICE_TYPE_COFFEE_MACHINE: [
        {
            "name": "Heat Up",
            "description": "Boiler heating to reach operating temperature.",
            "is_default": True,
        },
        {
            "name": "Brewing",
            "description": "Water pumping through coffee grounds.",
            "is_default": True,
        },
        {
            "name": "Keep Warm",
            "description": "Maintaining temperature after brew completion.",
            "is_default": True,
        },
        {
            "name": "Grinding",
            "description": "Bean grinding stage on machines with integrated grinder.",
            "is_default": True,
        },
        {
            "name": "Steaming",
            "description": "Steam generation for milk frothing.",
            "is_default": True,
        },
        {
            "name": "Idle",
            "description": "Ready/standby period with low power use.",
            "is_default": True,
        },
    ],
    DEVICE_TYPE_EV: [
        {
            "name": "Initialization",
            "description": "Vehicle and charger handshake before power transfer.",
            "is_default": True,
        },
        {
            "name": "Charging",
            "description": "Main charging period at available power.",
            "is_default": True,
        },
        {
            "name": "Taper",
            "description": "Reduced charging rate near high state of charge.",
            "is_default": True,
        },
        {
            "name": "Maintenance",
            "description": "Battery balancing or conditioning activity.",
            "is_default": True,
        },
        {
            "name": "Complete",
            "description": "Charge complete with minimal top-up activity.",
            "is_default": True,
        },
        {
            "name": "Pre-Conditioning",
            "description": "Battery temperature conditioning before or during charge.",
            "is_default": True,
        },
    ],
    DEVICE_TYPE_AIR_FRYER: [
        {
            "name": "Pre-Heat",
            "description": "Initial chamber heating before full cooking.",
            "is_default": True,
        },
        {
            "name": "Cooking",
            "description": "Main cooking phase with active heater and fan.",
            "is_default": True,
        },
        {
            "name": "Pause",
            "description": "Short pause for shaking or inspection.",
            "is_default": True,
        },
        {
            "name": "Cool Down",
            "description": "Fan-only cool-down stage after heating.",
            "is_default": True,
        },
        {
            "name": "Keep Warm",
            "description": "Low-heat holding stage to keep food warm.",
            "is_default": True,
        },
    ],
    DEVICE_TYPE_HEAT_PUMP: [
        {
            "name": "Start-Up",
            "description": "Compressor and system stabilization at cycle start.",
            "is_default": True,
        },
        {
            "name": "Heating",
            "description": "Active heating operation.",
            "is_default": True,
        },
        {
            "name": "Cooling",
            "description": "Active cooling operation.",
            "is_default": True,
        },
        {
            "name": "Defrost",
            "description": "Defrost routine to clear outdoor coil ice.",
            "is_default": True,
        },
        {
            "name": "Standby",
            "description": "Low-activity temperature holding period.",
            "is_default": True,
        },
        {
            "name": "Fan Only",
            "description": "Air circulation without compressor heating/cooling.",
            "is_default": True,
        },
        {
            "name": "Boost",
            "description": "High-output operation for rapid temperature change.",
            "is_default": True,
        },
    ],
}


def normalize_phase_name(name: str) -> str:
    """Normalize and validate phase names."""
    normalized = " ".join(name.strip().split())
    if not normalized:
        raise ValueError("invalid_phase_name")
    if len(normalized) > 48:
        raise ValueError("phase_name_too_long")
    return normalized


def get_default_phase_catalog(device_type: str) -> list[PhaseItem]:
    """Return default phase catalog for a device type."""
    return deepcopy(DEFAULT_PHASES_BY_DEVICE.get(device_type, []))


def get_shared_default_phase_catalog() -> list[PhaseItem]:
    """Return a shared default catalog deduplicated across all device types."""
    merged: list[PhaseItem] = []
    seen: set[str] = set()
    for device_phases in DEFAULT_PHASES_BY_DEVICE.values():
        for item in device_phases:
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            key = name.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(
                {
                    "name": name,
                    "description": str(item.get("description", "")).strip(),
                    "is_default": True,
                }
            )
    return merged


def merge_phase_catalog(device_type: str, custom_phases: list[PhaseItem] | None) -> list[PhaseItem]:
    """Merge device defaults with custom phases for UI/selection."""
    merged = (
        get_default_phase_catalog(device_type)
        if device_type in DEFAULT_PHASES_BY_DEVICE
        else get_shared_default_phase_catalog()
    )
    custom = custom_phases or []
    for item in custom:
        try:
            normalized_name = normalize_phase_name(str(item.get("name", "")))
        except ValueError:
            continue
        merged.append(
            {
                "name": normalized_name,
                "description": str(item.get("description", "")).strip(),
                "is_default": False,
            }
        )
    return [p for p in merged if p.get("name")]
