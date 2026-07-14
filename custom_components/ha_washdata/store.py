"""Community-store bridge: gating + provenance + import/share orchestration.

Pure/near-pure glue between ``store_client`` (network) and ``profile_store`` (local).
Nothing here runs unless online features are enabled.
"""
from __future__ import annotations

from typing import Any

from .const import (
    CONF_ENABLE_ONLINE_FEATURES,
    DEFAULT_ENABLE_ONLINE_FEATURES,
    QC_EDITED,
    QC_MANUAL,
    QC_RECORDING,
)


def online_features_enabled(options: dict[str, Any] | None) -> bool:
    """True when the user has opted into online store features (default off)."""
    if not options:
        return DEFAULT_ENABLE_ONLINE_FEATURES
    return bool(options.get(CONF_ENABLE_ONLINE_FEATURES, DEFAULT_ENABLE_ONLINE_FEATURES))


def derive_qc(cycle: dict[str, Any]) -> int:
    """Derive the obfuscated provenance code for a cycle being uploaded.

    QC_RECORDING - a pure recorder capture.
    QC_EDITED    - trimmed/edited from a detected cycle.
    QC_MANUAL    - a plain detected cycle the user flagged golden by hand.
    Never raises.
    """
    meta = cycle.get("meta") if isinstance(cycle.get("meta"), dict) else {}
    if meta.get("source") == "recorder" or "original_samples" in meta:
        return QC_RECORDING
    if meta.get("edited"):
        return QC_EDITED
    return QC_MANUAL
