"""Opt-in ML scoring bridge for WashData (experimental).

This package holds compact, NumPy-only models trained offline in the
``ml_washdata`` lab and embedded here as base64 blobs (see
``promoted_manifest.json`` for provenance). The integration runtime stays
NumPy-only; no sklearn/torch/scipy are imported.

The single runtime entry point is :func:`resolve_scorer`, which returns a scoring
callable for a capability, preferring an on-device trained spec over the shipped
embedded baseline. All live ML consumers go through it (the panel's ``ml_health``
shadow comparison in ``ws_api`` and :class:`MLSuggestionEngine`), and any new
runtime consumer should too — feature extraction lives in ``feature_extraction``
and gating in :func:`ml_models_enabled`, so there is no separate engine object.

Each model consumes a feature mapping whose keys are the model's
``FEATURE_COLUMNS``; the integration computes those from live data per the
``*_feature_contract.json`` files shipped alongside the model modules.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Mapping

CONF_ENABLE_ML_MODELS = "enable_ml_models"

# Logical capability -> generated model module name (without the _model suffix).
_MODEL_MODULES = {
    "quality": "hybrid_curve_quality_model",
    "live_match": "live_match_commit_model",
    "end": "cycle_end_detector_model",
}


def ml_models_enabled(options: Mapping[str, object] | None) -> bool:
    """True when the user has opted into experimental ML models."""
    if not options:
        return False
    return bool(options.get(CONF_ENABLE_ML_MODELS, False))


def resolve_scorer(capability: str, store: object | None):
    """Return ``(score_fn, source)`` for a capability, preferring an on-device
    trained spec over the shipped embedded baseline.

    ``score_fn`` maps a feature mapping -> float in [0,1]; ``source`` is
    ``"on_device"`` or ``"baseline"``. Returns ``(None, None)`` when neither is
    available. This is the single bridge that lets trained models (Stage 4)
    actually reach inference (ML Lab shadow comparison + MLSuggestionEngine)
    while transparently falling back to the baseline.
    """
    # 1) On-device trained spec from the store.
    if store is not None:
        try:
            versions = store.get_ml_model_versions() or {}  # type: ignore[attr-defined]
            record = versions.get(capability)
            spec = record.get("spec") if isinstance(record, dict) else None
            if isinstance(spec, dict):
                from .trainer import score_spec

                return (lambda feats, _s=spec: float(score_spec(_s, feats)), "on_device")
        except Exception:  # noqa: BLE001 - never let a bad store break inference
            pass
    # 2) Shipped embedded baseline module.
    module_name = _MODEL_MODULES.get(capability)
    if module_name is not None:
        try:
            module = importlib.import_module(f"{__package__}.{module_name}")
            return (lambda feats, _m=module: float(_m.score(feats)), "baseline")
        except Exception:  # noqa: BLE001
            pass
    return (None, None)


def resolve_regressor(capability: str, store: object | None):
    """Return ``(predict_fn, source)`` for a regression capability.

    Regression models (currently only ``"remaining_time"``) have **no** shipped
    embedded baseline - they are trained purely on-device (Stage 4) and stored as
    ``standardized_linear`` specs. This returns ``(None, None)`` until on-device
    training promotes one, so live behaviour is unchanged until then.

    ``predict_fn`` maps a feature mapping -> float in the model's target units
    (for ``remaining_time`` that is a completion fraction in ~[0, 1]).
    """
    if store is None:
        return (None, None)
    try:
        versions = store.get_ml_model_versions() or {}  # type: ignore[attr-defined]
        record = versions.get(capability)
        spec = record.get("spec") if isinstance(record, dict) else None
        if isinstance(spec, dict) and spec.get("kind") == "standardized_linear":
            from .trainer import predict_value_spec

            return (lambda feats, _s=spec: float(predict_value_spec(_s, feats)), "on_device")
    except Exception:  # noqa: BLE001 - never let a bad store break inference
        pass
    return (None, None)


def available_models() -> list[dict[str, object]]:
    """Return provenance for the embedded models, or [] if none are shipped."""
    manifest = Path(__file__).resolve().parent / "promoted_manifest.json"
    if not manifest.exists():
        return []
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    models = payload.get("models")
    return models if isinstance(models, list) else []
