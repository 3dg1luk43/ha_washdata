"""Tests for the opt-in, embedded ML models package.

These verify that the promoted NumPy-only models load and score inside the
integration, that the engine stays inert until explicitly enabled, and that the
models degrade gracefully. They do not require Home Assistant.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

_ML_DIR = Path(__file__).resolve().parent.parent / "custom_components" / "ha_washdata" / "ml"

from custom_components.ha_washdata.ml import (
    CONF_ENABLE_ML_MODELS,
    available_models,
    ml_models_enabled,
    resolve_scorer,
)

MODEL_MODULES = [
    "hybrid_curve_quality_model",
    "live_match_commit_model",
    "cycle_end_detector_model",
]


def _load(module_name: str):
    return importlib.import_module(f"custom_components.ha_washdata.ml.{module_name}")


@pytest.mark.parametrize("module_name", MODEL_MODULES)
def test_model_module_loads_and_scores(module_name: str) -> None:
    module = _load(module_name)
    assert module.FEATURE_COLUMNS, "model must expose feature columns"
    # All-zero features must produce a finite probability in [0, 1].
    score = module.score({})
    assert 0.0 <= score <= 1.0
    # Scoring is deterministic and order-independent for the same inputs.
    features = {column: 0.3 for column in module.FEATURE_COLUMNS}
    assert module.score(features) == pytest.approx(module.score(dict(reversed(list(features.items())))))
    # predict() agrees with the embedded threshold.
    assert module.predict(features) == (module.score(features) >= module.THRESHOLD)


def test_ml_models_enabled_flag() -> None:
    assert ml_models_enabled(None) is False
    assert ml_models_enabled({}) is False
    assert ml_models_enabled({CONF_ENABLE_ML_MODELS: False}) is False
    assert ml_models_enabled({CONF_ENABLE_ML_MODELS: True}) is True


def test_available_models_manifest() -> None:
    models = available_models()
    names = {model.get("name") for model in models}
    # The three promoted models should be present with provenance.
    assert {"hybrid_curve_quality", "live_match_commit", "cycle_end_detector"} <= names
    for model in models:
        assert model.get("kind") in {"standardized_logistic", "standardized_linear"}
        assert "metrics" in model


def test_unknown_capability_returns_none() -> None:
    score_fn, source = resolve_scorer("does_not_exist", None)
    assert score_fn is None and source is None


@pytest.mark.parametrize("module_name", MODEL_MODULES)
def test_embedded_model_matches_lab_parity_fixtures(module_name: str) -> None:
    """The embedded model must reproduce the lab's scores bit-for-bit.

    This is the cross-repo correctness guarantee for the base64-promotion
    approach: the lab emits golden (features -> expected_score) cases and the
    integration's embedded copy must reproduce them within rounding tolerance.
    """
    module = _load(module_name)
    parity_path = _ML_DIR / f"{module.MODEL_NAME}_parity.json"
    assert parity_path.exists(), f"missing parity fixtures for {module.MODEL_NAME}"
    payload = json.loads(parity_path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    assert cases, "parity fixtures must contain cases"
    for case in cases:
        produced = module.score(case["features"])
        assert produced == pytest.approx(case["expected_score"], abs=1e-5), (
            f"{module.MODEL_NAME}: embedded score {produced} != lab {case['expected_score']}"
        )


def test_parity_fixtures_present_for_all_manifest_models() -> None:
    for model in available_models():
        name = model["name"]
        assert (_ML_DIR / f"{name}_parity.json").exists(), f"no parity fixtures shipped for {name}"
        assert (_ML_DIR / f"{name}_feature_contract.json").exists(), f"no feature contract shipped for {name}"
