"""On-device training orchestration (Stage 4, gated by ENABLE_ML_TRAINING).

Gathers the user's own labelled cycles, derives training labels from data the
integration already has, fits NumPy-only logistic heads with :mod:`.trainer`,
and promotes a retrained model over the shipped baseline only when it is at
least as good on a held-out split. Nothing here runs unless the training loop
(behind the feature flag + per-device opt-in) invokes it.

Label sources (no manual labelling required to start):
  * end detector  - from trace geometry: a completed cycle's final low-power
    event is a true end (positive); earlier pauses that resumed are non-ends.
  * quality model - from cycle status + optional ML-Lab review labels: clean
    completed / "good" / "golden" -> not a problem; force_stopped / interrupted
    / "bad" / "unusable" -> a problem.
  * live_match    - deferred: needs per-prefix ranking history to accumulate.
"""
from __future__ import annotations

import importlib
import logging
from typing import Any

import numpy as np

_LOGGER = logging.getLogger(__name__)

from ..const import ML_TRAINING_AUC_MARGIN, ML_TRAINING_MIN_POSITIVES
from ..time_utils import power_data_to_offsets
from . import trainer as T

# Capability -> (embedded module name, target label). Mirrors engine._MODEL_MODULES.
_CAPABILITIES = {
    "end": ("cycle_end_detector_model", "cycle_end"),
    "quality": ("hybrid_curve_quality_model", "cycle_quality"),
}

_ACTIVE_FLOOR_RATIO = 0.02
_MIN_ROWS = 40


def _read_points(cycle: dict[str, Any]) -> list[tuple[float, float]]:
    raw = cycle.get("power_data")
    if not isinstance(raw, list) or len(raw) < 2:
        return []
    start_iso = cycle.get("start_time") if isinstance(cycle.get("start_time"), str) else None
    try:
        pairs = power_data_to_offsets(raw, start_iso)
        return [(float(o), float(p)) for o, p in pairs]
    except (TypeError, ValueError):
        return []


def _profile_expectations(cycles: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, list[float]]] = {}
    for c in cycles:
        name = c.get("profile_name")
        if not isinstance(name, str) or not name:
            continue
        s = stats.setdefault(name, {"d": [], "e": [], "p": []})
        for k, field in (("d", "duration"), ("e", "energy_wh"), ("p", "max_power")):
            v = c.get(field)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                s[k].append(float(v))
    out: dict[str, dict[str, float]] = {}
    for name, s in stats.items():
        if not s["d"]:
            continue
        out[name] = {
            "duration": float(np.median(s["d"])),
            "energy": float(np.median(s["e"])) if s["e"] else 500.0,
            "peak": float(np.median(s["p"])) if s["p"] else 500.0,
        }
    return out


def _matrix(rows: list[dict[str, float]], columns: list[str]) -> np.ndarray:
    if not rows:
        return np.empty((0, len(columns)), dtype=float)
    return np.array(
        [[float(r.get(col) or 0.0) for col in columns] for r in rows], dtype=float
    )


def _end_dataset(
    clean: list[dict[str, Any]],
    expectations: dict[str, dict[str, float]],
    stop_thr: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Positives = each completed clean cycle's final end; negatives = pauses that resumed."""
    from .feature_extraction import END_FEATURE_COLUMNS, latest_end_event_features

    rows: list[dict[str, float]] = []
    labels: list[float] = []
    for c in clean:
        exp = expectations.get(c.get("profile_name"))
        if not exp:
            continue
        points = _read_points(c)
        if len(points) < 6:
            continue
        peak = max((p for _, p in points), default=0.0)
        if peak <= 0:
            continue
        active_thr = max(stop_thr, _ACTIVE_FLOOR_RATIO * peak)
        in_low = False
        low_start = 0.0
        for i, (t, p) in enumerate(points):
            if not in_low and p < active_thr:
                in_low = True
                low_start = t
            elif in_low and p >= active_thr:
                if (points[i - 1][0] - low_start) >= 30.0:
                    feat = latest_end_event_features(points[:i], exp)
                    if feat is not None:
                        rows.append(feat)
                        labels.append(0.0)  # resumed -> not the end
                in_low = False
        feat_end = latest_end_event_features(points, exp)
        if feat_end is not None:
            rows.append(feat_end)
            labels.append(1.0)  # trace ends here -> true end
    return _matrix(rows, list(END_FEATURE_COLUMNS)), np.array(labels, dtype=float), list(END_FEATURE_COLUMNS)


def _quality_label(cycle: dict[str, Any]) -> float | None:
    """1 = problem, 0 = clean, None = unknown (skip)."""
    review = cycle.get("ml_review")
    if isinstance(review, dict):
        if review.get("golden"):
            return 0.0  # pinned reference cycle -> definitely clean
        q = review.get("quality")
        if q in ("good", "golden"):
            return 0.0
        if q in ("bad", "unusable"):
            return 1.0
    status = cycle.get("status")
    if status in ("force_stopped", "interrupted"):
        return 1.0
    if status == "completed":
        return 0.0
    return None


def _quality_dataset(
    cycles: list[dict[str, Any]],
    expectations: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Uses ALL cycles (not clean-filtered) so mis-detected cycles are the positives."""
    from .feature_extraction import QUALITY_FEATURE_COLUMNS, quality_features

    rows: list[dict[str, float]] = []
    labels: list[float] = []
    for c in cycles:
        exp = expectations.get(c.get("profile_name"))
        if not exp:
            continue
        label = _quality_label(c)
        if label is None:
            continue
        points = _read_points(c)
        if len(points) < 6:
            continue
        raw_conf = c.get("match_confidence")
        if isinstance(raw_conf, (int, float)) and not isinstance(raw_conf, bool) and raw_conf > 0:
            conf = float(raw_conf)
            proxy_dist, proxy_margin, proxy_fit = max(0.0, 1.0 - conf), conf, conf
        else:
            proxy_dist, proxy_margin, proxy_fit = 0.25, 0.30, 0.75
        try:
            feat = quality_features(
                points, exp["duration"], exp["energy"], exp["peak"],
                proxy_dist, proxy_margin, proxy_fit, 0,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            continue
        rows.append(feat)
        labels.append(label)
    return _matrix(rows, list(QUALITY_FEATURE_COLUMNS)), np.array(labels, dtype=float), list(QUALITY_FEATURE_COLUMNS)


def _holdout_split(
    X: np.ndarray, y: np.ndarray, *, frac: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Seeded split that keeps both classes in the test set when possible."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * frac)))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    # Guarantee both classes present in test; otherwise fall back to all-data eval.
    if len(np.unique(y[test_idx])) < 2 or len(np.unique(y[train_idx])) < 2:
        return X, y, X, y
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def _embedded_module(capability: str):
    module_name = _CAPABILITIES.get(capability, (None, None))[0]
    if module_name is None:
        return None
    try:
        return importlib.import_module(f"{__package__}.{module_name}")
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _baseline_auc(capability: str, X_test: np.ndarray, y_test: np.ndarray, columns: list[str]) -> float | None:
    module = _embedded_module(capability)
    if module is None:
        return None
    try:
        scores = np.array(
            [float(module.score(dict(zip(columns, row)))) for row in X_test], dtype=float
        )
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    return T.auc(y_test, scores)


def _baseline_threshold(capability: str, default: float) -> float:
    module = _embedded_module(capability)
    thr = getattr(module, "THRESHOLD", None) if module is not None else None
    return float(thr) if isinstance(thr, (int, float)) else default


def _train_capability(
    capability: str,
    target: str,
    X: np.ndarray,
    y: np.ndarray,
    columns: list[str],
    trained_at: str,
) -> dict[str, Any]:
    """Fit + gate one capability. Returns a status record (promoted or not)."""
    n = X.shape[0]
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n < _MIN_ROWS or n_pos < ML_TRAINING_MIN_POSITIVES or n_neg < 5:
        return {"capability": capability, "promoted": False,
                "reason": f"insufficient data (rows={n}, pos={n_pos}, neg={n_neg})"}

    X_tr, y_tr, X_te, y_te = _holdout_split(X, y)
    fit = T.fit_logistic(X_tr, y_tr)
    default_thr = _baseline_threshold(capability, 0.5)
    spec_probe = {"center": fit["center"], "scale": fit["scale"], "coef": fit["coef"],
                  "bias": fit["bias"], "feature_columns": columns}
    train_scores = T.score_matrix_spec(spec_probe, X_tr)
    threshold = T.select_threshold(y_tr, train_scores, default=default_thr)

    test_scores = T.score_matrix_spec(spec_probe, X_te)
    new_auc = T.auc(y_te, test_scores)
    metrics = T.binary_metrics(y_te, test_scores, threshold)
    base_auc = _baseline_auc(capability, X_te, y_te, columns)
    baseline = base_auc if base_auc is not None else 0.5

    promote = new_auc >= (baseline - ML_TRAINING_AUC_MARGIN)
    record: dict[str, Any] = {
        "capability": capability,
        "promoted": bool(promote),
        "rows": n, "positives": n_pos, "negatives": n_neg,
        "new_auc": round(new_auc, 4),
        "baseline_auc": round(baseline, 4),
        "threshold": threshold,
        "metrics": metrics,
    }
    if promote:
        record["spec"] = T.build_spec(
            name=capability, target=target, feature_columns=columns,
            fit=fit, threshold=threshold,
            metrics={"holdout": metrics, "auc": round(new_auc, 4), "baseline_auc": round(baseline, 4)},
            trained_at=trained_at, cycle_count=n,
        )
        record["trained_at"] = trained_at
        record["cycle_count"] = n
    else:
        record["reason"] = f"AUC {new_auc:.3f} below baseline {baseline:.3f} - margin"
    return record


def train_from_cycles(
    cycles: list[dict[str, Any]],
    device_type: str | None,
    stop_threshold_w: float = 2.0,
    trained_at: str = "",
) -> dict[str, Any]:
    """Pure function (executor-safe): build datasets, train, gate all capabilities.

    Returns ``{"results": [record, ...], "promoted": {capability: record}}``.
    Caller persists the promoted records via ``profile_store.set_ml_model_version``.
    """
    from ..suggestion_engine import select_clean_cycles

    clean, _excluded = select_clean_cycles(cycles, stop_threshold_w=stop_threshold_w)
    expectations = _profile_expectations(cycles)

    datasets = {
        "end": _end_dataset(clean, expectations, stop_threshold_w),
        "quality": _quality_dataset(cycles, expectations),
    }

    results: list[dict[str, Any]] = []
    promoted: dict[str, Any] = {}
    for capability, (module_name, target) in _CAPABILITIES.items():
        X, y, columns = datasets[capability]
        record = _train_capability(capability, target, X, y, columns, trained_at)
        results.append(record)
        if record.get("promoted") and "spec" in record:
            promoted[capability] = {
                "spec": record["spec"],
                "trained_at": trained_at,
                "cycle_count": record["cycle_count"],
                "metrics": record["metrics"],
                "new_auc": record["new_auc"],
                "baseline_auc": record["baseline_auc"],
            }
    return {"results": results, "promoted": promoted}


async def async_run_training(hass: Any, manager: Any) -> dict[str, Any]:
    """Public entry point: train on this device's cycles and persist winners.

    Offloads the CPU work to an executor thread and persists any promoted model
    specs into the profile store. Returns a summary for logging / the event.
    """
    from ..const import CONF_MIN_POWER, CONF_STOP_THRESHOLD_W

    store = manager.profile_store
    entry = hass.config_entries.async_get_entry(manager.entry_id)
    merged = {**(entry.data if entry else {}), **(entry.options if entry else {})}
    stop_thr = 2.0
    for key in (CONF_STOP_THRESHOLD_W, CONF_MIN_POWER):
        try:
            v = float(merged.get(key))
        except (TypeError, ValueError):
            continue
        if v > 0:
            stop_thr = v
            break

    from homeassistant.util import dt as dt_util

    trained_at = dt_util.now().isoformat()
    cycles = store.get_past_cycles()

    _LOGGER.info(
        "On-device ML training starting: %d cycles, device_type=%s, stop_threshold=%.1fW",
        len(cycles), manager.device_type, stop_thr,
    )
    summary = await hass.async_add_executor_job(
        train_from_cycles, cycles, manager.device_type, stop_thr, trained_at
    )
    for record in summary.get("results", []):
        if record.get("promoted"):
            _LOGGER.info(
                "ML training PROMOTED %s: AUC %.3f vs baseline %.3f (rows=%s, pos=%s)",
                record["capability"], record.get("new_auc", 0), record.get("baseline_auc", 0),
                record.get("rows"), record.get("positives"),
            )
        else:
            _LOGGER.info(
                "ML training kept baseline for %s: %s",
                record["capability"], record.get("reason", "not promoted"),
            )
    for capability, record in summary.get("promoted", {}).items():
        await store.set_ml_model_version(capability, record)
    return summary
