"""Shared test helpers: a remaining-time regressor spec for the frozen C4 consumer.

On-device training of the ``remaining_time`` head was removed in 0.5.8 (audit
ML-11), but its consumer (``progress.ml_progress_percent``) stays, frozen off, and
its tests patch the flag on. They still need a spec fitted on progress-shaped data,
so this rebuilds the old prefix dataset (each clean cycle cut at fixed elapsed
fractions, target = completion fraction) and runs it through the same regression
gate the ``total_energy`` head uses.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from custom_components.ha_washdata.ml.feature_extraction import (
    PROGRESS_FEATURE_COLUMNS,
    profile_expectations,
    progress_features,
)
from custom_components.ha_washdata.ml.training_task import (
    _PROGRESS_CUT_FRACTIONS,
    _matrix,
    _read_points,
    _train_regression_capability,
)


def remaining_time_rows(
    cycles: list[dict[str, Any]], expectations: dict[str, dict[str, float]] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """``(X, y, columns, groups)``: completion-fraction rows from cycle prefixes."""
    exps = profile_expectations(cycles) if expectations is None else expectations
    columns = list(PROGRESS_FEATURE_COLUMNS)
    rows: list[dict[str, float]] = []
    labels: list[float] = []
    groups: list[int] = []
    for ci, cycle in enumerate(cycles):
        exp = exps.get(cycle.get("profile_name"))
        points = _read_points(cycle)
        if not exp or len(points) < 12:
            continue
        t0, total = points[0][0], points[-1][0] - points[0][0]
        if total <= 60.0:
            continue
        for frac in _PROGRESS_CUT_FRACTIONS:
            prefix = [(o, p) for o, p in points if o <= t0 + frac * total]
            feat = progress_features(prefix, exp) if len(prefix) >= 4 else None
            if feat is None:
                continue
            rows.append(feat)
            labels.append(min(max((prefix[-1][0] - t0) / total, 0.0), 1.0))
            groups.append(ci)
    return _matrix(rows, columns), np.array(labels, dtype=float), columns, np.array(groups, dtype=int)


def trained_remaining_time_spec(cycles: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The promoted ``remaining_time`` spec for ``cycles``, or None if it did not promote."""
    X, y, columns, groups = remaining_time_rows(cycles)
    record = _train_regression_capability(
        "remaining_time", "progress_fraction", "fraction", X, y, columns,
        "2026-07-03T02:00:00+00:00", groups,
    )
    return record.get("spec") if record.get("promoted") else None
