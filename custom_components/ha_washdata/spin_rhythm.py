# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Spin/heat rhythm descriptor for washer matching (issue #486).

A washer's programme family shows in its mechanical rhythm, which the shape score
cannot see because the heater dominates correlation and peak-relative MAE. This
module turns a raw power trace into a handful of rhythm figures and grades a cycle
against a profile's own member cycles.

All levels are relative to the trace: heater = readings >= ``SPIN_HEAT_FRAC`` x
peak; drum baseline ``B`` = time-weighted median of the active non-heater
readings; a mechanical event is a run above 1.5 B (6 s gaps bridged) that reaches
``SPIN_LEVEL_K`` x B. Durations are zero-order-hold time on the timestamps, never
sample counts. Pure, NumPy only, never raises (returns None).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .const import (
    SPIN_FEATURE_SCALES,
    SPIN_HEAT_FRAC,
    SPIN_LEVEL_K,
    SPIN_MIN_MEMBERS_TOP2,
)

_ZOH_CAP_S = 300.0
_HEAT_BRIDGE_S = 30.0
_MECH_BRIDGE_S = 6.0
_EPISODE_GAP_S = 90.0
_SUSTAINED_S = 30.0
_SPIKE_S = 15.0
_ACTIVE_FRAC = 0.03
# Programme end before a quiet tail (active_end_s, #486).
_ACTIVE_GRID_STEP_S = 2.0
_ACTIVE_GRID_MAX_POINTS = 20000
_ACTIVE_SUSTAIN_S = 60.0
_ACTIVE_HI_FRAC = 0.08
_TAIL_QUIET_FRAC = 0.04
_TAIL_MIN_S = 60.0
_TAIL_MAX_TRIM = 0.5


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _bridged(t: np.ndarray, dt: np.ndarray, mask: np.ndarray, gap: float) -> list[list[float]]:
    """[start_s, end_s, first_idx, end_idx] of True runs, gaps <= ``gap`` bridged."""
    out: list[list[float]] = []
    for a, b in _runs(mask):
        s, e = float(t[a]), float(t[b - 1] + dt[b - 1])
        if out and s - out[-1][1] <= gap:
            out[-1][1] = e
            out[-1][3] = b
        else:
            out.append([s, e, a, b])
    return out


def rhythm_events(offsets: Any, watts: Any) -> dict[str, Any] | None:
    """Heater blocks and mechanical (spin) events of one trace, or None.

    ``{"heat": [[start_s, end_s, on_s]...], "mech": [[start_s, end_s, peak_w]...],
    "span": s}``, times relative to the first reading.
    """
    try:
        t = np.asarray(offsets, dtype=float)
        w = np.asarray(watts, dtype=float)
        if t.size != w.size or t.size < 8:
            return None
        ok = np.isfinite(t) & np.isfinite(w)
        t, w = t[ok], w[ok]
        if t.size < 8:
            return None
        order = np.argsort(t, kind="mergesort")
        t = t[order] - t[order][0]
        w = w[order]
        span = float(t[-1])
        if span <= 0:
            return None
        dt = np.minimum(np.diff(t, append=t[-1]), _ZOH_CAP_S)
        peak = float(np.max(w))
        if peak <= 0:
            return None
        heat_lvl = SPIN_HEAT_FRAC * peak
        act = (w >= _ACTIVE_FRAC * peak) & (w < heat_lvl)
        if int(act.sum()) < 3 or float(dt[act].sum()) <= 0:
            return None
        aw, ad = w[act], dt[act]
        o = np.argsort(aw, kind="mergesort")
        cum = np.cumsum(ad[o])
        base = float(aw[o][min(int(np.searchsorted(cum, 0.5 * cum[-1])), aw.size - 1)])
        if base <= 0:
            return None
        spin_lvl = min(SPIN_LEVEL_K * base, 0.9 * heat_lvl)
        heat = []
        for s, e, a, b in _bridged(t, dt, w >= heat_lvl, _HEAT_BRIDGE_S):
            seg_w, seg_dt = w[int(a):int(b)], dt[int(a):int(b)]
            heat.append([round(s, 1), round(e, 1), round(float(seg_dt[seg_w >= heat_lvl].sum()), 1)])
        mech = []
        for s, e, a, b in _bridged(t, dt, (w >= 1.5 * base) & (w < heat_lvl), _MECH_BRIDGE_S):
            pk = float(np.max(w[int(a):int(b)]))
            if pk >= spin_lvl:
                mech.append([round(s, 1), round(e, 1), round(pk, 1)])
        return {"heat": heat, "mech": mech, "span": round(span, 1)}
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def rhythm_descriptor(ev: Any, horizon: float | None = None) -> list[float] | None:
    """Feature vector of ``ev`` over ``[0, horizon]`` (whole trace when None).

    ``[ln t_end_min, ln1p heat_min, ln1p sustained_inter_min, spikes per
    intermediate episode, ln1p final_sustained_min]``: when the last spin ends,
    how long the heater ran, Cotton-like ramp+plateau intermediate spins, Easy
    Care-like spike pairs, the final spin's sustained length. Spins count only
    after the last heater block; the last episode seen is the "final" one, so a
    prefix is compared with the same prefix of each member.
    """
    try:
        if not isinstance(ev, dict):
            return None
        span = float(ev.get("span") or 0.0)
        T = span if horizon is None else float(horizon)
        if not math.isfinite(T) or T <= 0:
            return None
        heat = [h for h in ev.get("heat") or [] if float(h[0]) < T]
        last_heat = min(float(heat[-1][1]), T) if heat else 0.0
        heat_on = sum(min(float(h[2]), max(0.0, T - float(h[0]))) for h in heat)
        eps: list[list[tuple[float, float]]] = []
        for s, e, _pk in ev.get("mech") or []:
            s, e = float(s), min(float(e), T)
            if s < last_heat or s >= T:
                continue
            if eps and s - eps[-1][-1][1] <= _EPISODE_GAP_S:
                eps[-1].append((s, e))
            else:
                eps.append([(s, e)])
        fin = eps[-1] if eps else None
        inter = eps[:-1]
        t_end = fin[-1][1] if fin else last_heat
        sus = sum(e - s for ep in inter for s, e in ep if e - s >= _SUSTAINED_S)
        spk = sum(1 for ep in inter for s, e in ep if e - s < _SPIKE_S)
        fin_sus = sum(e - s for s, e in fin) if fin else 0.0
        vec = [
            math.log(max(t_end, 60.0) / 60.0),
            math.log1p(heat_on / 60.0),
            math.log1p(sus / 60.0),
            (spk / len(inter)) if inter else 0.0,
            math.log1p(fin_sus / 60.0),
        ]
        return vec if all(math.isfinite(v) for v in vec) else None
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def member_rhythm_reference(
    members: list[tuple[list[float], list[float], float]],
) -> dict[str, Any] | None:
    """The rhythm events of a profile's member cycles, stored on the envelope."""
    out = []
    for offsets, watts, _dur in members:
        ev = rhythm_events(offsets, watts)
        if ev is not None:
            out.append(ev)
    return {"v": 1, "n": len(out), "events": out} if out else None


def rhythm_agreement(
    query: list[float] | None,
    ref: Any,
    horizon: float | None = None,
) -> float | None:
    """0-1 agreement of a query descriptor with a profile's members, or None.

    Per member: mean over features of a Gaussian kernel at the feature's scale.
    The profile scores its best member (the mean of its best two once it has
    ``SPIN_MIN_MEMBERS_TOP2``), so a multi-modal profile (an option that adds a
    rinse) and a one-cycle profile are both graded on a real cycle, not a median
    no member has.
    """
    try:
        if query is None or not isinstance(ref, dict):
            return None
        evs = ref.get("events") or []
        scales = SPIN_FEATURE_SCALES
        per: list[float] = []
        for ev in evs:
            vec = rhythm_descriptor(ev, horizon)
            if vec is None:
                continue
            ag = 0.0
            for q, m, s in zip(query, vec, scales):
                z = (q - m) / s
                ag += math.exp(-0.5 * z * z)
            per.append(ag / len(scales))
        if not per:
            return None
        per.sort(reverse=True)
        if len(per) >= SPIN_MIN_MEMBERS_TOP2:
            return float((per[0] + per[1]) / 2.0)
        return float(per[0])
    except Exception:  # pylint: disable=broad-exception-caught
        return None



def active_end_s(offsets: Any, watts: Any) -> float | None:
    """Seconds from the first reading to the programme's end when the trace carries
    a QUIET tail after it (a Miele anti-crease phase: base ~3 W, 50-80 W pulses
    until the door opens), else None (issue #486).

    The end is the end of the last ``_ACTIVE_SUSTAIN_S`` window whose median is at
    least ``_ACTIVE_HI_FRAC`` x the trace's sustained peak (its highest rolling
    median), on a step-hold grid, so a lone spike neither counts as activity nor
    sets the peak. Accepted only when no later window's median exceeds
    ``_TAIL_QUIET_FRAC`` x that peak (a washer still rinsing at 100 W after a 2 kW
    heater is not in a tail), the cut is at least ``_TAIL_MIN_S`` and it removes at
    most ``_TAIL_MAX_TRIM`` of the span. Relative to the trace, not watts. Measured
    on the corpus: the #486 Miele median tail 29.7 min, the maintainer's two
    washers 1.6 and 0 min, where "after the last spin" read 62 and 51. Never raises.
    """
    try:
        t = np.asarray(offsets, dtype=float)
        p = np.asarray(watts, dtype=float)
        if t.ndim != 1 or t.shape != p.shape or t.size < 3:
            return None
        ok = np.isfinite(t) & np.isfinite(p)
        t, p = t[ok], p[ok]
        if t.size < 3:
            return None
        order = np.argsort(t, kind="mergesort")
        t, p = t[order], p[order]
        span = float(t[-1] - t[0])
        if not (span > 0 and math.isfinite(span)):
            return None
        g = max(_ACTIVE_GRID_STEP_S, span / _ACTIVE_GRID_MAX_POINTS)
        grid = t[0] + np.arange(int(span // g) + 1, dtype=float) * g
        idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, t.size - 1)
        pg = p[idx]
        k = max(3, int(round(_ACTIVE_SUSTAIN_S / g)) | 1)
        if pg.size < k + 2:
            return None
        med = np.median(np.lib.stride_tricks.sliding_window_view(pg, k), axis=1)
        peak = float(np.max(med))
        if not peak > 0:
            return None
        on = np.flatnonzero(med >= _ACTIVE_HI_FRAC * peak)
        if on.size == 0:
            return None
        end_i = min(int(on[-1]) + k - 1, grid.size - 1)
        active = float(grid[end_i] - t[0])
        if span - active < _TAIL_MIN_S or active < (1.0 - _TAIL_MAX_TRIM) * span:
            return None
        rest = med[end_i + 1:]
        if rest.size and float(np.max(rest)) > _TAIL_QUIET_FRAC * peak:
            return None
        return active if active > 0 else None
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def member_tail_reference(
    members: list[tuple[list[float], list[float], float]],
) -> dict[str, Any] | None:
    """A profile's quiet tail after its programme end, from its own cycles (#486):
    ``{"n", "with_tail", "median_s"}``, stored on the envelope as ``tail_ref``.

    ``members`` are the envelope build's ``(offsets, watts, duration)`` triples. A
    member's tail is its duration after :func:`active_end_s` (0 without one), so
    ``median_s`` is the time the programme usually runs on after it finished and
    ``with_tail`` how many members had a tail at all. Never raises.
    """
    try:
        tails: list[float] = []
        for offsets, watts, duration in members:
            try:
                dur = float(duration)
            except (TypeError, ValueError, OverflowError):
                continue
            if not (math.isfinite(dur) and dur > 0) or len(offsets) < 3:
                continue
            end = active_end_s(offsets, watts)
            if end is None:
                tails.append(0.0)
                continue
            # Both measured from the first reading (member offsets start at it).
            tails.append(max(0.0, dur - end))
        if not tails:
            return None
        return {
            "n": len(tails),
            "with_tail": sum(1 for x in tails if x >= _TAIL_MIN_S),
            "median_s": round(float(np.median(tails)), 1),
        }
    except Exception:  # pylint: disable=broad-exception-caught
        return None
