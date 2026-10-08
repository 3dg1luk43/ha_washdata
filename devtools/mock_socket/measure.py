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
"""Measurements over a stretch of the plot. Pure.

Both traces are step functions in epoch milliseconds: a reading holds until the next one.
``reported`` is what Home Assistant received (``None`` = the plug was offline from there);
``draw`` is what the appliance really drew. Energy from the reports is what a left
Riemann sum over the sensor would give, so its gap to the true energy is the cost of the
plug's reporting.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

Point = tuple[float, float | None]


def segments(points: Sequence[Point], t0: float, t1: float) -> list[tuple[float, float, float | None]]:
    """``(start, end, W)`` pieces of a step function inside ``[t0, t1]``.

    ``W`` is None before the first point and wherever the trace says no reading.
    """
    if t1 <= t0:
        return []
    times = [p[0] for p in points]
    i = bisect.bisect_right(times, t0) - 1
    current = points[i][1] if i >= 0 else None
    out, cursor = [], t0
    for ts, w in points[i + 1:]:
        if ts >= t1:
            break
        if ts > cursor:
            out.append((cursor, ts, current))
            cursor = ts
        current = w
    out.append((cursor, t1, current))
    return out


def _energy_wh(pieces: list[tuple[float, float, float | None]]) -> float:
    return sum(w * (b - a) for a, b, w in pieces if w is not None) / 3.6e6


@dataclass(frozen=True)
class Measurement:
    start_ms: float
    end_ms: float
    reports: int
    median_interval_s: float | None
    longest_silence_s: float
    energy_reported_wh: float
    energy_true_wh: float | None
    mean_w: float | None
    min_w: float | None
    max_w: float | None
    std_w: float | None
    true_peak_w: float | None
    offline_s: float

    @property
    def duration_s(self) -> float:
        return (self.end_ms - self.start_ms) / 1000.0

    @property
    def error_pct(self) -> float | None:
        if not self.energy_true_wh:
            return None
        return 100.0 * (self.energy_reported_wh - self.energy_true_wh) / self.energy_true_wh

    def as_text(self) -> str:
        """A plain-text summary for the clipboard."""
        fmt = lambda ms: datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M:%S")  # noqa: E731
        rows = [
            ("from", fmt(self.start_ms)), ("to", fmt(self.end_ms)),
            ("duration", hms(self.duration_s)),
            ("energy from reports", f"{self.energy_reported_wh:.2f} Wh"),
            ("energy drawn", "-" if self.energy_true_wh is None else f"{self.energy_true_wh:.2f} Wh"),
            ("reporting error", "-" if self.error_pct is None else f"{self.error_pct:+.2f} %"),
            ("mean power", _w(self.mean_w)), ("min power", _w(self.min_w)),
            ("max power", _w(self.max_w)), ("std dev", _w(self.std_w)),
            ("peak drawn", _w(self.true_peak_w)),
            ("reports", str(self.reports)),
            ("median interval", "-" if self.median_interval_s is None else f"{self.median_interval_s:.1f} s"),
            ("longest silence", hms(self.longest_silence_s)),
            ("offline", hms(self.offline_s)),
        ]
        return "\n".join(f"{k}: {v}" for k, v in rows)


def _w(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f} W"


def hms(seconds: float) -> str:
    s = int(round(max(0.0, seconds)))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def measure(reported: Sequence[Point], draw: Sequence[Point], t0: float, t1: float) -> Measurement:
    t0, t1 = min(t0, t1), max(t0, t1)
    pieces = segments(reported, t0, t1)
    covered = [(a, b, w) for a, b, w in pieces if w is not None]
    span_ms = sum(b - a for a, b, _ in covered)
    mean = std = None
    if span_ms > 0:
        mean = sum(w * (b - a) for a, b, w in covered) / span_ms
        std = math.sqrt(sum((w - mean) ** 2 * (b - a) for a, b, w in covered) / span_ms)
    values = [w for _, _, w in covered]

    stamps = [ts for ts, w in reported if t0 <= ts <= t1 and w is not None]
    intervals = [(b - a) / 1000.0 for a, b in zip(stamps, stamps[1:])]
    edges = [t0, *stamps, t1]
    silence = max((b - a) / 1000.0 for a, b in zip(edges, edges[1:]))

    drawn = segments(draw, t0, t1) if draw else []
    drawn_known = [p for p in drawn if p[2] is not None]
    return Measurement(
        start_ms=t0,
        end_ms=t1,
        reports=len(stamps),
        median_interval_s=sorted(intervals)[len(intervals) // 2] if intervals else None,
        longest_silence_s=silence,
        energy_reported_wh=_energy_wh(pieces),
        energy_true_wh=_energy_wh(drawn) if drawn_known else None,
        mean_w=mean,
        min_w=min(values) if values else None,
        max_w=max(values) if values else None,
        std_w=std,
        true_peak_w=max((w for _, _, w in drawn_known), default=None),
        offline_s=sum((b - a) for a, b, w in pieces if w is None and a >= (reported[0][0] if reported else t1)) / 1000.0,
    )
