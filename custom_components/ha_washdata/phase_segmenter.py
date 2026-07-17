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
"""Unsupervised power-trace phase segmentation (Phase 0 prototype).

Turns a cycle's power trace into an ordered list of :class:`PhaseSegment`
(role, start/end offset, duration, energy, mean/peak power) via a hysteresis
regime classifier (idle / active / high) + minimum-run merge + role assignment
driven by a per-device-type :class:`PhaseModel`.

Constraint: NumPy only, no Home Assistant imports. Pure and executor-safe;
``segment_cycle`` never raises - it returns ``[]`` on malformed input.

This module is INERT: nothing in the live integration imports it yet. It exists
so the Phase-0 side-by-side harness (`devtools/eta_phase_eval.py`) and the unit
tests can exercise the real segmentation logic before any live wiring. See
`docs/superpowers/specs/2026-07-17-phase-segmented-matching-design.md`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .signal_processing import energy_gap_threshold_s, integrate_wh

# Regime tokens (internal, not user-facing).
_IDLE = 0
_ACTIVE = 1
_HIGH = 2

# Role tokens. Physically detectable roles; mapping to display phase names
# (phase_catalog.DEFAULT_PHASES_BY_DEVICE) is a later (Phase 6) concern.
ROLE_HEATING = "heating"
ROLE_WASH = "wash"
ROLE_SPIN = "spin"
ROLE_IDLE = "idle"


@dataclass(frozen=True)
class PhaseSegment:
    """One contiguous phase of a cycle."""

    role: str
    t_start: float          # seconds from cycle start
    t_end: float
    duration_s: float
    energy_wh: float
    mean_w: float
    peak_w: float
    open: bool = False      # True when this is the final segment of a partial cycle


@dataclass(frozen=True)
class PhaseModel:
    """Per-device-type segmentation parameters and role vocabulary.

    ``high`` threshold = ``max(high_w_floor, high_frac * robust_peak)`` where the
    robust peak is the 95th percentile of the trace, so a heating element is
    detected relative to the device's own scale without a single spike inflating
    it. ``active_w`` separates motor/fill activity from idle. A terminal
    ``active`` block whose mean power >= ``spin_min_w`` and which begins in the
    last ``spin_tail_frac`` of the (completed) cycle is labelled ``spin``.
    """

    device_type: str
    high_w_floor: float
    high_frac: float
    active_w: float
    spin_min_w: float
    spin_tail_frac: float
    min_run_s: float
    roles: tuple[str, ...] = field(default=(ROLE_HEATING, ROLE_WASH, ROLE_SPIN, ROLE_IDLE))


# Only washing_machine is intended for live rollout first; dishwasher and
# washer-dryer models exist for OFFLINE evaluation in the Phase-0 harness.
_MODELS: dict[str, PhaseModel] = {
    "washing_machine": PhaseModel(
        device_type="washing_machine",
        high_w_floor=800.0, high_frac=0.5,
        active_w=30.0, spin_min_w=200.0, spin_tail_frac=0.30,
        min_run_s=90.0,
    ),
    "washer_dryer": PhaseModel(
        device_type="washer_dryer",
        high_w_floor=800.0, high_frac=0.5,
        active_w=30.0, spin_min_w=200.0, spin_tail_frac=0.30,
        min_run_s=90.0,
    ),
    # Dishwasher: heating element is the dominant HIGH regime; the long passive
    # dry tail lands in IDLE. Deferred for LIVE rollout (delicate end-of-cycle
    # logic) - present here only for offline measurement.
    "dishwasher": PhaseModel(
        device_type="dishwasher",
        high_w_floor=800.0, high_frac=0.5,
        active_w=25.0, spin_min_w=250.0, spin_tail_frac=0.25,
        min_run_s=120.0,
    ),
}


# Device types phase matching is rolled out for LIVE (validated by the Phase-0
# ETA gate). Dishwasher has a model for OFFLINE evaluation only (its ETA gate was
# a no-go and its end-of-cycle logic is delicate), so it is deliberately excluded
# here even though ``phase_model_for("dishwasher")`` returns a model.
LIVE_PHASE_DEVICE_TYPES: tuple[str, ...] = ("washing_machine", "washer_dryer")


def phase_model_for(device_type: str | None) -> PhaseModel | None:
    """Return the :class:`PhaseModel` for a device type, or ``None`` to fall back.

    ``None`` means the device type has no validated phase model and must use the
    existing whole-cycle pipeline (zero-regression fallback). Returns a model for
    every type with one defined (incl. dishwasher, for the offline harness).
    """
    if not device_type:
        return None
    return _MODELS.get(str(device_type))


def phase_matching_live_supported(device_type: str | None) -> bool:
    """True when phase matching is enabled for LIVE use on this device type.

    Stricter than ``phase_model_for`` - only the Phase-0-validated rollout types
    (washing machine, washer-dryer). Used to gate phase-profile caching and the
    live ETA path; the offline harness uses ``phase_model_for`` directly.
    """
    return (
        device_type in LIVE_PHASE_DEVICE_TYPES
        and phase_model_for(device_type) is not None
    )


def _classify(power: np.ndarray, model: PhaseModel) -> tuple[np.ndarray, float]:
    """Per-sample regime tokens + the robust peak used for the HIGH threshold."""
    robust_peak = float(np.percentile(power, 95)) if power.size else 0.0
    high_thr = max(model.high_w_floor, model.high_frac * robust_peak)
    reg = np.full(power.shape, _IDLE, dtype=int)
    reg[power >= model.active_w] = _ACTIVE
    reg[power >= high_thr] = _HIGH
    return reg, robust_peak


def _runs(reg: np.ndarray) -> list[list[int]]:
    """Contiguous same-regime runs as ``[regime, start_idx, end_idx]`` (inclusive)."""
    runs: list[list[int]] = []
    n = len(reg)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and reg[j + 1] == reg[i]:
            j += 1
        runs.append([int(reg[i]), i, j])
        i = j + 1
    return runs


def _merge_short(runs: list[list[int]], t: np.ndarray, min_run_s: float) -> list[list[int]]:
    """Absorb runs shorter than ``min_run_s`` into their neighbour.

    A short leading run merges forward; every other short run merges into the
    preceding (already-accepted) run. This prevents brief motor spikes during
    tumbling from fragmenting a wash phase, and brief dips from splitting a
    heating block, without erasing genuine phases.
    """
    if not runs:
        return runs
    merged: list[list[int]] = []
    for reg, a, b in runs:
        dur = float(t[b] - t[a])
        if merged and dur < min_run_s:
            merged[-1][2] = b  # extend previous run to absorb this short one
        else:
            merged.append([reg, a, b])
    return merged


def segment_cycle(
    timestamps: "np.ndarray | list[float]",
    power: "np.ndarray | list[float]",
    model: PhaseModel,
    *,
    partial: bool = False,
) -> list[PhaseSegment]:
    """Segment a power trace into ordered phases. Never raises.

    Args:
        timestamps: seconds from cycle start (ascending).
        power: watts, same length as ``timestamps``.
        model: the device-type :class:`PhaseModel`.
        partial: when True the trace is an observed-so-far prefix of a running
            cycle; the final segment is marked ``open=True``.

    Returns:
        List of :class:`PhaseSegment` in time order, or ``[]`` when the input is
        too short/degenerate to segment.
    """
    try:
        t = np.asarray(timestamps, dtype=float)
        w = np.asarray(power, dtype=float)
    except (TypeError, ValueError):
        return []
    if t.size < 4 or w.size != t.size:
        return []
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(w))):
        finite = np.isfinite(t) & np.isfinite(w)
        t, w = t[finite], w[finite]
        if t.size < 4:
            return []
    # Enforce ascending time.
    order = np.argsort(t)
    t, w = t[order], w[order]

    reg, _peak = _classify(w, model)
    runs = _merge_short(_runs(reg), t, model.min_run_s)
    if not runs:
        return []

    gap_s = energy_gap_threshold_s(t)
    total = float(t[-1] - t[0])
    spin_zone_start = t[0] + (1.0 - model.spin_tail_frac) * total if total > 0 else t[-1]

    # First pass: build raw segments with stats.
    raw: list[dict] = []
    for reg_tok, a, b in runs:
        seg_t = t[a:b + 1]
        seg_w = w[a:b + 1]
        dur = float(seg_t[-1] - seg_t[0]) if seg_t.size > 1 else 0.0
        energy = integrate_wh(seg_t, seg_w, max_gap_s=gap_s) if seg_t.size > 1 else 0.0
        raw.append({
            "reg": reg_tok, "a": a, "b": b,
            "t0": float(seg_t[0]), "t1": float(seg_t[-1]),
            "dur": dur, "energy": float(energy),
            "mean": float(np.mean(seg_w)), "peak": float(np.max(seg_w)),
        })

    # Determine which ACTIVE run (if any) is the spin: the last non-idle run,
    # elevated and starting in the terminal zone. Only on a completed cycle -
    # a partial cycle can't know its terminal segment yet.
    spin_idx: Optional[int] = None
    if not partial:
        for idx in range(len(raw) - 1, -1, -1):
            seg = raw[idx]
            if seg["reg"] == _IDLE:
                continue
            if (seg["reg"] == _ACTIVE and seg["mean"] >= model.spin_min_w
                    and seg["t0"] >= spin_zone_start):
                spin_idx = idx
            break  # only inspect the last non-idle run

    out: list[PhaseSegment] = []
    for idx, seg in enumerate(raw):
        if seg["reg"] == _HIGH:
            role = ROLE_HEATING
        elif seg["reg"] == _IDLE:
            role = ROLE_IDLE
        else:  # _ACTIVE
            role = ROLE_SPIN if idx == spin_idx else ROLE_WASH
        out.append(PhaseSegment(
            role=role, t_start=seg["t0"], t_end=seg["t1"],
            duration_s=seg["dur"], energy_wh=seg["energy"],
            mean_w=seg["mean"], peak_w=seg["peak"],
            open=(partial and idx == len(raw) - 1),
        ))
    return out
