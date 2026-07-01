"""Opt-in ML engine for WashData (experimental).

This package holds compact, NumPy-only models trained offline in the
``ml_washdata`` lab and embedded here as base64 blobs (see
``promoted_manifest.json`` for provenance). The integration runtime stays
NumPy-only; no sklearn/torch/scipy are imported.

These models are **opt-in**. Until a user enables them, nothing here is used and
the proven existing detection/matching/ETA code paths run unchanged. The engine
returns ``None`` whenever the feature flag is off or a model is unavailable, so
callers fall back to the existing behavior.

Wiring (kept out of this module to avoid touching core logic):
    - Add ``CONF_ENABLE_ML_MODELS = "enable_ml_models"`` to const.py and an
      options-flow toggle (default False).
    - Construct ``MLEngine.from_options(entry.options)`` in the manager and call
      the scoring helpers where a decision is made, treating ``None`` as "use the
      existing proven logic".

Each model consumes a feature mapping whose keys are the model's
``FEATURE_COLUMNS``; the integration must compute those from live data per the
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


class MLEngine:
    """Loads models lazily and gates them behind the opt-in flag.

    Prefers an on-device *user-trained* spec (Stage 4) for a capability when one
    is present, and transparently falls back to the shipped embedded baseline
    otherwise. User specs are the NumPy-only ``standardized_logistic`` dicts
    produced by :mod:`.trainer` and persisted in the profile store.
    """

    def __init__(
        self,
        enabled: bool,
        user_models: Mapping[str, Mapping[str, object]] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self._cache: dict[str, object] = {}
        self._user_models: dict[str, Mapping[str, object]] = dict(user_models or {})

    @classmethod
    def from_options(cls, options: Mapping[str, object] | None) -> "MLEngine":
        return cls(ml_models_enabled(options))

    @classmethod
    def from_options_and_store(
        cls, options: Mapping[str, object] | None, store: object
    ) -> "MLEngine":
        """Build an engine that prefers on-device trained models from the store."""
        user: dict[str, Mapping[str, object]] = {}
        try:
            versions = store.get_ml_model_versions() or {}  # type: ignore[attr-defined]
            for capability, record in versions.items():
                spec = record.get("spec") if isinstance(record, dict) else None
                if isinstance(spec, dict):
                    user[capability] = spec
        except Exception:  # noqa: BLE001 - a bad store must never break the engine
            user = {}
        return cls(ml_models_enabled(options), user_models=user)

    def set_user_models(self, user_models: Mapping[str, Mapping[str, object]] | None) -> None:
        """Replace the set of on-device trained specs (capability -> spec)."""
        self._user_models = dict(user_models or {})

    def model_source(self, capability: str) -> str:
        """'on_device' when a user-trained model backs this capability, else 'baseline'."""
        return "on_device" if capability in self._user_models else "baseline"

    def _module(self, capability: str):
        if not self.enabled:
            return None
        if capability not in self._cache:
            module_name = _MODEL_MODULES.get(capability)
            if module_name is None:
                self._cache[capability] = None
            else:
                try:
                    self._cache[capability] = importlib.import_module(f"{__package__}.{module_name}")
                except Exception:  # noqa: BLE001 - a missing/broken model must never break the integration
                    self._cache[capability] = None
        return self._cache[capability]

    def _score(self, capability: str, features: Mapping[str, float]) -> float | None:
        if not self.enabled:
            return None
        spec = self._user_models.get(capability)
        if spec is not None:
            try:
                from .trainer import score_spec

                return float(score_spec(spec, features))
            except Exception:  # noqa: BLE001 - fall back to the embedded baseline
                pass
        module = self._module(capability)
        if module is None:
            return None
        try:
            return float(module.score(features))
        except Exception:  # noqa: BLE001 - degrade gracefully to existing behavior
            return None

    # --- quality / bad-cycle detector ---
    def quality_problem_score(self, features: Mapping[str, float]) -> float | None:
        """P(cycle is a problem) in [0,1], or None when disabled/unavailable."""
        return self._score("quality", features)

    def quality_is_problem(self, features: Mapping[str, float]) -> bool | None:
        return self._threshold("quality", features)

    # --- live program match commit confidence ---
    def live_match_commit_confidence(self, features: Mapping[str, float]) -> float | None:
        """P(top-1 live match is the correct program) in [0,1], or None."""
        return self._score("live_match", features)

    def live_match_should_commit(self, features: Mapping[str, float]) -> bool | None:
        return self._threshold("live_match", features)

    # --- cycle-end detector ---
    def end_confidence(self, features: Mapping[str, float]) -> float | None:
        """P(low-power event is the true end vs a pause) in [0,1], or None."""
        return self._score("end", features)

    def end_is_final(self, features: Mapping[str, float]) -> bool | None:
        return self._threshold("end", features)

    def end_confidence_for_series(
        self,
        points: "list[tuple[float, float]]",
        expectation: Mapping[str, float],
    ) -> float | None:
        """Convenience: P(true end) from a live (offset_s, watt) series + profile expectation.

        Returns None when disabled, when the model is unavailable, or when there
        is no qualifying low-power run yet (so the caller keeps existing logic).
        """
        if not self.enabled:
            return None
        try:
            from .feature_extraction import latest_end_event_features

            features = latest_end_event_features(points, dict(expectation))
        except Exception:  # noqa: BLE001 - never let feature extraction break the integration
            return None
        if features is None:
            return None
        return self.end_confidence(features)

    def live_match_confidence_for_prefix(
        self,
        points: "list[tuple[float, float]]",
        elapsed_s: float,
        top1_distance: float,
        top2_distance: "float | None",
        top1_median_duration_s: float,
        candidate_count: int,
    ) -> float | None:
        """Convenience: P(top-1 match is correct) from the current match ranking.

        Returns None when disabled or the model is unavailable. The caller
        should keep the existing ranking logic when None is returned.

        Args:
            points: Observed prefix (offset_s, watts).
            elapsed_s: Seconds since cycle start.
            top1_distance: Blended RMSE+DTW distance to the top-1 candidate.
            top2_distance: Distance to top-2; pass None when only one candidate.
            top1_median_duration_s: Expected duration of top-1 candidate profile.
            candidate_count: Number of candidate profiles for this device.
        """
        if not self.enabled:
            return None
        try:
            from .feature_extraction import live_match_features

            features = live_match_features(
                points,
                elapsed_s,
                top1_distance,
                top2_distance,
                top1_median_duration_s,
                candidate_count,
            )
        except Exception:  # noqa: BLE001
            return None
        return self.live_match_commit_confidence(features)

    def quality_score_for_cycle(
        self,
        points: "list[tuple[float, float]]",
        profile_median_duration_s: float,
        profile_median_energy_wh: float,
        profile_median_peak_w: float,
        profile_distance: float,
        label_margin: float,
        profile_fit_score: float,
        flag_count: int,
    ) -> float | None:
        """Convenience: P(cycle is a problem) from the completed cycle + match result.

        Returns None when disabled or the model is unavailable. The caller
        should treat None as "no ML signal; use existing quality logic".

        Args:
            points: Complete cycle power trace (offset_s, watts).
            profile_median_duration_s: Matched profile's median duration (s).
            profile_median_energy_wh: Matched profile's median energy (Wh).
            profile_median_peak_w: Matched profile's median peak power (W).
            profile_distance: Shape distance from MatchResult to the profile.
            label_margin: Score gap between top-1 and top-2 candidates (0 = ambiguous).
            profile_fit_score: Profile fit score in [0, 1] from the matcher.
            flag_count: Detection/anomaly flags raised by the existing detector.
        """
        if not self.enabled:
            return None
        try:
            from .feature_extraction import quality_features

            features = quality_features(
                points,
                profile_median_duration_s,
                profile_median_energy_wh,
                profile_median_peak_w,
                profile_distance,
                label_margin,
                profile_fit_score,
                flag_count,
            )
        except Exception:  # noqa: BLE001
            return None
        return self.quality_problem_score(features)

    def _threshold(self, capability: str, features: Mapping[str, float]) -> bool | None:
        if not self.enabled:
            return None
        spec = self._user_models.get(capability)
        if spec is not None:
            try:
                from .trainer import predict_spec

                return bool(predict_spec(spec, features))
            except Exception:  # noqa: BLE001 - fall back to the embedded baseline
                pass
        module = self._module(capability)
        if module is None:
            return None
        try:
            return bool(module.predict(features))
        except Exception:  # noqa: BLE001
            return None
