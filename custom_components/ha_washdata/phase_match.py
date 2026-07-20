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
"""Phase-aware profile matching and phase-resolved ETA (Phase 0 prototype).

Given a cycle segmented into phases (:mod:`phase_segmenter`), this module:

* builds a per-profile :class:`PhaseProfile` (per-role duration/energy priors)
  from the profile's member cycles;
* scores candidate profiles by **per-role** duration + energy agreement, so two
  profiles that differ mainly in heating length (i.e. temperature) are separated
  by a large per-phase signal instead of a small whole-cycle-correlation delta;
* projects **time-remaining** as a per-role budget (Σ expected_role_total −
  consumed_role), which personalises the ETA to the matched variant.

The matcher handles **partial** (observed-so-far) cycles for progressive
narrowing: completed roles are compared fully; the open current role is scored
one-sided (a candidate is only penalised if the observed duration already
*exceeds* its expected total), so the correct larger-heating variant is not
prematurely ruled out while heating is still in progress.

Constraint: NumPy only, no Home Assistant imports. Pure and executor-safe;
consumed live (behind the default-off ``enable_phase_matching`` per-device gate)
by ``ProfileStore.phase_remaining`` and the progress blend. See
`docs/superpowers/specs/2026-07-17-phase-segmented-matching-design.md`.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .phase_segmenter import (
    ROLE_HEATING,
    ROLE_IDLE,
    ROLE_SPIN,
    ROLE_WASH,
    PhaseSegment,
)

# Per-role weight in the match score. Heating dominates because it is the
# temperature discriminator; the wash middle is load-variable; spin/idle are
# weak/noisy signals. Overridable via the match config.
_ROLE_WEIGHTS: dict[str, float] = {
    ROLE_HEATING: 0.50,
    ROLE_WASH: 0.25,
    ROLE_SPIN: 0.15,
    ROLE_IDLE: 0.10,
}
_DUR_SCALE = 0.35        # log-ratio agreement scale for per-role duration
_EN_SCALE = 0.40         # ... and per-role energy
# Agreement score credited when a role is present on one side but absent on the
# other (structural mismatch, e.g. a heating vs no-heating cycle). 0.0 = full
# penalty (structural difference is a strong, wanted discriminator); raise toward
# 1.0 to soften. Config-overridable via ``occ_penalty``.
_OCC_PENALTY = 0.0


@dataclass(frozen=True)
class RoleStat:
    """Aggregated per-role prior across a profile's member cycles."""

    dur_mean: float
    dur_std: float
    dur_p50: float
    en_mean: float
    occurrence: float     # fraction of member cycles that contain this role


@dataclass(frozen=True)
class PhaseProfile:
    """Per-profile phase model: per-role priors + total-duration prior."""

    name: str
    roles: dict[str, RoleStat]
    total_dur_mean: float
    total_dur_std: float
    n_cycles: int


@dataclass(frozen=True)
class PhaseMatchResult:
    name: str
    score: float


def phase_profile_to_dict(profile: PhaseProfile) -> dict:
    """Serialize a :class:`PhaseProfile` for storage (JSON-safe)."""
    return {
        "name": profile.name,
        "n_cycles": profile.n_cycles,
        "total_dur_mean": profile.total_dur_mean,
        "total_dur_std": profile.total_dur_std,
        "roles": {role: asdict(stat) for role, stat in profile.roles.items()},
    }


def phase_profile_from_dict(data: dict | None) -> PhaseProfile | None:
    """Rebuild a :class:`PhaseProfile` from stored form. Never raises."""
    if not isinstance(data, dict):
        return None
    try:
        roles = {
            str(role): RoleStat(
                dur_mean=float(s.get("dur_mean", 0.0)),
                dur_std=float(s.get("dur_std", 0.0)),
                dur_p50=float(s.get("dur_p50", 0.0)),
                en_mean=float(s.get("en_mean", 0.0)),
                occurrence=float(s.get("occurrence", 0.0)),
            )
            for role, s in (data.get("roles") or {}).items()
            if isinstance(s, dict)
        }
        if not roles:
            return None
        return PhaseProfile(
            name=str(data.get("name", "")),
            roles=roles,
            total_dur_mean=float(data.get("total_dur_mean", 0.0)),
            total_dur_std=float(data.get("total_dur_std", 0.0)),
            n_cycles=int(data.get("n_cycles", 0)),
        )
    except (TypeError, ValueError, AttributeError):
        return None


def _agree(observed: float, expected: float, scale: float) -> float:
    """Log-ratio agreement in (0, 1]; 1.0 when equal, sharper for small scale."""
    if observed <= 0.0 or expected <= 0.0:
        # Both ~zero → perfect agreement; one zero → no agreement.
        return 1.0 if observed <= 0.0 and expected <= 0.0 else 0.0
    scale = scale if scale > 1e-9 else 1e-9  # avoid div-by-zero on a 0 config scale
    return 1.0 / (1.0 + abs(math.log(observed / expected)) / scale)


def _role_totals(segments: list[PhaseSegment]) -> dict[str, dict[str, float]]:
    """Sum duration + energy per role across a cycle's segments."""
    totals: dict[str, dict[str, float]] = {}
    for seg in segments:
        acc = totals.setdefault(seg.role, {"dur": 0.0, "en": 0.0})
        acc["dur"] += max(0.0, seg.duration_s)
        acc["en"] += max(0.0, seg.energy_wh)
    return totals


def build_phase_profile(name: str, segmented_cycles: list[list[PhaseSegment]]) -> PhaseProfile | None:
    """Aggregate a profile's member cycles into per-role priors. Never raises.

    Returns ``None`` when no usable cycles are supplied.
    """
    cycles = [c for c in segmented_cycles if c]
    if not cycles:
        return None
    n = len(cycles)
    per_role_durs: dict[str, list[float]] = {}
    per_role_ens: dict[str, list[float]] = {}
    role_count: dict[str, int] = {}
    totals_dur: list[float] = []
    for segs in cycles:
        totals = _role_totals(segs)
        totals_dur.append(sum(v["dur"] for v in totals.values()))
        for role, v in totals.items():
            per_role_durs.setdefault(role, []).append(v["dur"])
            per_role_ens.setdefault(role, []).append(v["en"])
            role_count[role] = role_count.get(role, 0) + 1

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    def _std(xs: list[float], m: float) -> float:
        return float((sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5) if xs else 0.0

    def _p50(xs: list[float]) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        mid = len(s) // 2
        return float(s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0)

    roles: dict[str, RoleStat] = {}
    for role, durs in per_role_durs.items():
        m = _mean(durs)
        roles[role] = RoleStat(
            dur_mean=m, dur_std=_std(durs, m), dur_p50=_p50(durs),
            en_mean=_mean(per_role_ens[role]),
            occurrence=role_count[role] / n,
        )
    tm = _mean(totals_dur)
    return PhaseProfile(
        name=name, roles=roles,
        total_dur_mean=tm, total_dur_std=_std(totals_dur, tm), n_cycles=n,
    )


def match_phase_profiles(
    observed: list[PhaseSegment],
    candidates: list[PhaseProfile],
    config: dict | None = None,
) -> list[PhaseMatchResult]:
    """Rank ``candidates`` for the ``observed`` (full or partial) cycle. Never raises.

    Score = weighted mean over roles of ``sqrt(dur_agree * energy_agree)``, with a
    structural (occurrence-mismatch) penalty. The observed cycle's *open* role
    (partial cycle) is scored one-sided so a larger-heating candidate is not
    ruled out mid-heating.
    """
    if not observed or not candidates:
        return []
    cfg = config or {}
    weights = {**_ROLE_WEIGHTS, **(cfg.get("role_weights") or {})}
    dur_scale = float(cfg.get("dur_scale", _DUR_SCALE))
    en_scale = float(cfg.get("en_scale", _EN_SCALE))
    occ_pen = float(cfg.get("occ_penalty", _OCC_PENALTY))

    totals = _role_totals(observed)
    open_role = next((s.role for s in observed if s.open), None)
    is_partial = open_role is not None

    results: list[PhaseMatchResult] = []
    for cand in candidates:
        all_roles = set(totals) | set(cand.roles)
        num = 0.0
        den = 0.0
        for role in all_roles:
            w = weights.get(role, 0.1)
            if w <= 0:
                continue
            obs = totals.get(role, {"dur": 0.0, "en": 0.0})
            stat = cand.roles.get(role)
            if stat is None:
                # Observed a role the candidate never exhibits: structural miss.
                if obs["dur"] > 0:
                    num += w * occ_pen
                    den += w
                continue
            if role not in totals:
                # Candidate expects a role not yet observed. Future phase on a
                # partial cycle → neutral (skip). On a completed cycle →
                # structural miss (candidate does a phase this cycle never had).
                if not is_partial:
                    num += w * occ_pen
                    den += w
                continue
            if role == open_role:
                # One-sided: only penalise if the in-progress duration/energy
                # already EXCEEDS what this candidate expects for the whole role
                # (a larger-budget candidate must not be ruled out mid-phase).
                # Energy is the cleanest temperature discriminator, so score it
                # one-sided too rather than dropping it (weaker narrowing).
                da = 1.0 if obs["dur"] <= stat.dur_mean else _agree(obs["dur"], stat.dur_mean, dur_scale)
                ea = 1.0 if obs["en"] <= stat.en_mean else _agree(obs["en"], stat.en_mean, en_scale)
                agree = math.sqrt(da * ea)
            else:
                da = _agree(obs["dur"], stat.dur_mean, dur_scale)
                ea = _agree(obs["en"], stat.en_mean, en_scale)
                agree = math.sqrt(da * ea)
            num += w * agree
            den += w
        score = (num / den) if den > 0 else 0.0
        results.append(PhaseMatchResult(name=cand.name, score=float(score)))

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def phase_eta(
    observed: list[PhaseSegment],
    profile: PhaseProfile,
) -> float | None:
    """Project remaining seconds as a per-role budget. Never raises.

    Classifies each profile role against the observed-so-far segments (design §8):

      * **current** (the open, in-progress role): contribute
        ``max(0, dur_mean − consumed)`` using the CONDITIONAL mean (the role is
        known present, so no occurrence discount);
      * **completed** (a role already observed but not the open one): contribute
        **0** — it is done, even if it ran shorter than the profile mean (fixing
        the "phantom remaining" over-count);
      * **future** (a profile role not yet observed): contribute the
        occurrence-weighted prior ``dur_mean × occurrence`` (we do not yet know
        whether it will occur).

    Returns ``None`` when no phase priors are available (caller falls back to the
    current estimator).
    """
    if profile is None or not profile.roles:
        return None
    consumed = _role_totals(observed) if observed else {}
    # The open (in-progress) role. If the caller did not mark one (e.g. a
    # completed trace), treat the last observed role as current so completed
    # roles still contribute 0 rather than a spurious prior.
    open_role = next((s.role for s in (observed or []) if s.open), None)
    if open_role is None and observed:
        open_role = observed[-1].role
    remaining = 0.0
    for role, stat in profile.roles.items():
        done = consumed.get(role, {}).get("dur", 0.0)
        if role == open_role:
            # current: conditional mean (occurrence -> 1, role is present)
            remaining += max(0.0, stat.dur_mean - done)
        elif role in consumed:
            # completed: nothing more to spend on it
            continue
        else:
            # future: unconditional prior (may or may not occur)
            remaining += stat.dur_mean * max(0.0, min(1.0, stat.occurrence))
    return float(max(0.0, remaining))
