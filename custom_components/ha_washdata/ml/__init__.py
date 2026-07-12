"""Opt-in, NumPy-only ML models for WashData (experimental).

Models are trained offline in the ml_washdata lab and embedded here as base64
blobs. They are inert unless the user enables them. See engine.py and README.md.
"""

from .engine import (
    CONF_ENABLE_ML_MODELS,
    available_models,
    ml_models_enabled,
    resolve_regressor,
    resolve_scorer,
)

__all__ = [
    "CONF_ENABLE_ML_MODELS",
    "available_models",
    "ml_models_enabled",
    "resolve_regressor",
    "resolve_scorer",
]
