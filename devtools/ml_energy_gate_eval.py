#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The on-device ``total_energy`` regressor: its promotion gate and its served error.

Register: audit 2026-10-02 ML-12 / PROGRESS-16. Two things changed in
``ml/training_task.py`` and this measures both on every export in ``cycle_data/``
(eval.py's corpus, de-cloned):

* **Promotion decision.** OLD = the pre-fix gate, emulated: expectation from the
  median stored ``duration`` / ``energy_wh`` / ``max_power`` (``profile_expectations``),
  a plain 20% group holdout, MAE vs the naive estimate only. NEW = the shipped
  gate: the live expectation (``live_expectations``, with the matcher's duration
  per profile), at least ``ML_TRAINING_MIN_HOLDOUT_CYCLES`` held-out cycles. Both
  run with no incumbent (an export carries none worth scoring).
* **Served error.** What the projection actually gets: grouped 5-fold CV over the
  clean cycles, every model scored on rows built with the LIVE expectation (what
  ``progress.ml_energy_total`` feeds it), trained either on stored-field rows
  (old) or on live rows (new). The naive column is ``elapsed_over_expected``.

    python3 devtools/ml_energy_gate_eval.py [--corpus cycle_data]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import eval as E  # noqa: E402


def _spec(fit: dict) -> dict:
    return {"center": fit["center"], "scale": fit["scale"], "coef": fit["coef"],
            "bias": fit["bias"], "output_center": fit["y_center"], "output_scale": fit["y_scale"]}


def _served(T, Xo, yo, go, Xl, yl, gl) -> tuple[float, float, float, int] | None:
    """(naive, old-trained, new-trained) MAE on live rows, and the row count."""
    common = sorted(set(gl.tolist()) & set(go.tolist()))
    if len(common) < 10:
        return None
    folds = np.array_split(np.random.default_rng(0).permutation(common), 5)
    err_old: list[float] = []
    err_new: list[float] = []
    err_naive: list[float] = []
    for fold in folds:
        test = set(fold.tolist())
        tr_o = np.array([g not in test for g in go])
        tr_l = np.array([g not in test for g in gl])
        te_l = np.array([g in test for g in gl])
        if tr_o.sum() < 10 or tr_l.sum() < 10 or not te_l.any():
            continue
        fit_old = T.fit_ridge(Xo[tr_o], yo[tr_o], alpha=1.0)
        fit_new = T.fit_ridge(Xl[tr_l], yl[tr_l], alpha=1.0)
        truth = yl[te_l]
        for fit, sink in ((fit_old, err_old), (fit_new, err_new)):
            pred = np.clip(T.predict_matrix_spec(_spec(fit), Xl[te_l]), 0.0, 1.0)
            sink.extend(np.abs(pred - truth).tolist())
        err_naive.extend(np.abs(np.clip(Xl[te_l][:, 0], 0.0, 1.0) - truth).tolist())
    if not err_new:
        return None
    return float(np.mean(err_naive)), float(np.mean(err_old)), float(np.mean(err_new)), len(err_new)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--corpus", default=str(Path(__file__).resolve().parent.parent / "cycle_data"))
    args = ap.parse_args()
    logging.disable(logging.WARNING)

    from custom_components.ha_washdata.ml import trainer as T  # noqa: PLC0415
    from custom_components.ha_washdata.ml import training_task as TT  # noqa: PLC0415
    from custom_components.ha_washdata.ml.feature_extraction import (  # noqa: PLC0415
        profile_expectations,
    )
    from custom_components.ha_washdata.suggestion_engine import (  # noqa: PLC0415
        select_clean_cycles,
    )

    devices, _dropped = E.load_corpus(Path(args.corpus))
    promoted = {"old": 0, "new": 0}
    n_dev = 0
    served_rows: list[tuple[float, float, float, int]] = []
    for dev in devices:
        data = E.base_data(dev)
        past = list(data.get("past_cycles") or [])
        if len(past) < 5:
            continue
        n_dev += 1
        st, mgr = E.fresh_store(dev, data, {})
        clean, _ = select_clean_cycles(past, stop_threshold_w=2.0)
        live = TT.live_expectations(
            past, {c.get("profile_name") for c in clean}, TT._expected_durations(st),  # noqa: SLF001
        )
        Xl, yl, cols, gl = TT._energy_dataset(clean, live)  # noqa: SLF001
        Xo, yo, _cols, go = TT._energy_dataset(clean, profile_expectations(past))  # noqa: SLF001
        new = TT._train_regression_capability(  # noqa: SLF001
            "total_energy", "energy_fraction", "fraction", Xl, yl, cols, "", gl,
        )
        saved = TT.ML_TRAINING_MIN_HOLDOUT_CYCLES
        TT.ML_TRAINING_MIN_HOLDOUT_CYCLES = 0  # the pre-fix holdout: 20%, no floor
        try:
            old = TT._train_regression_capability(  # noqa: SLF001
                "total_energy", "energy_fraction", "fraction", Xo, yo, cols, "", go,
            )
        finally:
            TT.ML_TRAINING_MIN_HOLDOUT_CYCLES = saved
        promoted["old"] += bool(old.get("promoted"))
        promoted["new"] += bool(new.get("promoted"))
        served = _served(T, Xo, yo, go, Xl, yl, gl)
        if served:
            served_rows.append(served)
        tail = (f"naive {served[0]:.3f} old {served[1]:.3f} new {served[2]:.3f} (rows {served[3]})"
                if served else "n/a")
        print(f"{dev.key[:44]:44s} {mgr.device_type:16s} cycles {len(past):4d} clean {len(set(gl.tolist())):4d}"
              f" | old {'P' if old.get('promoted') else '-'} new {'P' if new.get('promoted') else '-'}"
              f" {new.get('reason_code', 'promoted'):26s} held out {new.get('held_out_cycles', 0):3d}"
              f" | served MAE {tail}")
    print(f"\npromoted: old {promoted['old']} / new {promoted['new']} of {n_dev} devices")
    if served_rows:
        arr = np.array([r[:3] for r in served_rows])
        weights = np.array([r[3] for r in served_rows])
        naive, old_mae, new_mae = (np.average(arr[:, i], weights=weights) for i in range(3))
        better = int((arr[:, 2] < arr[:, 1] - 5e-4).sum())
        worse = int((arr[:, 2] > arr[:, 1] + 5e-4).sum())
        print(f"served MAE (row-weighted, {len(served_rows)} devices): naive {naive:.4f}  "
              f"old-trained {old_mae:.4f}  new-trained {new_mae:.4f}")
        print(f"new-trained vs old-trained per device: {better} better, {worse} worse, "
              f"{len(served_rows) - better - worse} within 0.0005")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
