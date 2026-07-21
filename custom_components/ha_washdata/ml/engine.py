# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
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
import logging
from pathlib import Path
from typing import Mapping

_LOGGER = logging.getLogger(__name__)

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
    def _baseline():
        """Resolve the shipped embedded baseline scorer for this capability.

        Kept as a lazily-invoked helper so the baseline module is only imported
        when the on-device spec is absent *or* fails at call time - preserving the
        original "baseline only loaded when needed" semantics.
        """
        module_name = _MODEL_MODULES.get(capability)
        if module_name is None:
            return (None, None)
        try:
            module = importlib.import_module(f"{__package__}.{module_name}")
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                "Failed to load embedded baseline for capability %r: %s",
                capability, exc,
            )
            return (None, None)

        def _baseline_score(feats, _m=module):
            # The embedded baseline must never raise into live inference either
            # (mirrors _on_device_score's call-time guard): on any scoring error
            # log and return a neutral 0.0 so a gate treats the signal as absent
            # rather than letting the exception reach live detection/matching.
            try:
                return float(_m.score(feats))
            except Exception as exc:  # noqa: BLE001 - never raise into live inference
                _LOGGER.warning(
                    "Embedded baseline scorer for capability %r failed at call "
                    "time, returning neutral 0.0: %s", capability, exc,
                )
                return 0.0

        return (_baseline_score, "baseline")

    # 1) On-device trained spec from the store.
    if store is not None:
        try:
            versions = store.get_ml_model_versions() or {}  # type: ignore[attr-defined]
            record = versions.get(capability)
            spec = record.get("spec") if isinstance(record, dict) else None
            # Only treat a spec as a classifier here. A regression spec
            # (standardized_linear) must never be sigmoid-squashed by score_spec;
            # classifier and regression capability keys are disjoint today, but this
            # guard keeps it safe if a key were ever reused.
            if isinstance(spec, dict) and spec.get("kind") != "standardized_linear":
                # Feature-column schema guard: a spec promoted under an older
                # FEATURE_COLUMNS must be dropped rather than silently scoring on a
                # stale/neutral-filled schema. The call-time guard already catches
                # shape mismatches, but this catches them at load time and logs
                # clearly, so users see a single warm-up warning instead of a
                # per-inference warning storm.
                module_name = _MODEL_MODULES.get(capability)
                if module_name is not None:
                    try:
                        _bm = importlib.import_module(f"{__package__}.{module_name}")
                        _expected = list(getattr(_bm, "FEATURE_COLUMNS", []))
                        _stored = list(spec.get("feature_columns") or [])
                        if _expected and _stored and _stored != _expected:
                            _LOGGER.warning(
                                "Promoted spec for %r has stale feature schema "
                                "(%d cols vs current %d); reverting to baseline.",
                                capability, len(_stored), len(_expected),
                            )
                            return _baseline()
                    except Exception:  # noqa: BLE001 - schema check must not break inference
                        pass

                from .trainer import score_spec

                def _on_device_score(feats, _s=spec):
                    # A malformed / dimensionally-incompatible promoted spec must
                    # never raise into live detection/matching: on any call-time
                    # error fall back to the embedded baseline (or a neutral 0.0).
                    try:
                        return float(score_spec(_s, feats))
                    except Exception as exc:  # noqa: BLE001 - never raise into live inference
                        _LOGGER.warning(
                            "Trained scorer for capability %r failed at call time, "
                            "falling back to baseline: %s", capability, exc,
                        )
                        fn, _src = _baseline()
                        if fn is not None:
                            try:
                                return fn(feats)
                            except Exception:  # noqa: BLE001 - baseline must not raise either
                                pass
                        return 0.0

                return (_on_device_score, "on_device")
        except Exception as exc:  # noqa: BLE001 - never let a bad store break inference
            _LOGGER.warning(
                "Failed to load trained spec for capability %r, falling back to baseline: %s",
                capability, exc,
            )
    # 2) Shipped embedded baseline module.
    return _baseline()


def resolve_regressor(capability: str, store: object | None):
    """Return ``(predict_fn, source)`` for a regression capability.

    Regression models (``"remaining_time"`` and ``"total_energy"``) have **no**
    shipped embedded baseline - they are trained purely on-device (Stage 4) and
    stored as ``standardized_linear`` specs. This returns ``(None, None)`` until
    on-device training promotes one, so live behaviour is unchanged until then.

    ``predict_fn`` maps a feature mapping -> float in the model's target units
    (a completion fraction in ~[0, 1] for both regression capabilities).
    """
    if store is None:
        return (None, None)
    try:
        versions = store.get_ml_model_versions() or {}  # type: ignore[attr-defined]
        record = versions.get(capability)
        spec = record.get("spec") if isinstance(record, dict) else None
        if isinstance(spec, dict) and spec.get("kind") == "standardized_linear":
            # Feature-column schema guard for regression specs.
            try:
                from .feature_extraction import PROGRESS_FEATURE_COLUMNS
                _expected_r = list(PROGRESS_FEATURE_COLUMNS)
                _stored_r = list(spec.get("feature_columns") or [])
                if _expected_r and _stored_r and _stored_r != _expected_r:
                    _LOGGER.warning(
                        "Promoted regression spec for %r has stale feature schema "
                        "(%d cols vs current %d); reverting to inert.",
                        capability, len(_stored_r), len(_expected_r),
                    )
                    return (None, None)
            except Exception:  # noqa: BLE001 - schema check must not break inference
                pass

            from .trainer import predict_value_spec

            def _on_device_predict(feats, _s=spec):
                # A malformed / incompatible promoted regression spec must never
                # raise into the live remaining-time / energy estimates: on any
                # call-time error return NaN so the (isfinite-guarded) consumers
                # treat this capability as inert.
                try:
                    return float(predict_value_spec(_s, feats))
                except Exception as exc:  # noqa: BLE001 - never raise into live inference
                    _LOGGER.warning(
                        "Trained regressor for capability %r failed at call time, "
                        "returning inert value: %s", capability, exc,
                    )
                    return float("nan")

            return (_on_device_predict, "on_device")
    except Exception as exc:  # noqa: BLE001 - never let a bad store break inference
        _LOGGER.warning(
            "Failed to load trained regression spec for capability %r, capability will be inert: %s",
            capability, exc,
        )
    return (None, None)


_MANIFEST_MODELS_CACHE: list[dict[str, object]] | None = None


def available_models() -> list[dict[str, object]]:
    """Return provenance for the embedded models, or [] if none are shipped.

    The manifest is a shipped baseline file that never changes at runtime
    (on-device training writes specs into the store, not this file), so the parsed
    result is cached module-side after the first read.
    """
    global _MANIFEST_MODELS_CACHE
    if _MANIFEST_MODELS_CACHE is not None:
        return _MANIFEST_MODELS_CACHE
    manifest = Path(__file__).resolve().parent / "promoted_manifest.json"
    if not manifest.exists():
        return []
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    models = payload.get("models")
    result = models if isinstance(models, list) else []
    _MANIFEST_MODELS_CACHE = result
    return result
