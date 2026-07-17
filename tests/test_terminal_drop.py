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
"""Opt-in terminal-drop fast finalize.

A hard cliff-to-~0 that begins EARLIER than a device has ever legitimately gone
quiet (learned from its own completed cycles) is an anomaly - almost certainly a
real stop (plug pulled / cancelled) rather than a soak pause. The detector then
finalizes the cycle quickly instead of waiting out the full soak-bridging
``min_off_gap`` (up to 8 min for washers, 1 h for dishwashers).

Two layers are tested here without Home Assistant or any ML model:
  * ``earliest_sustained_quiet_offset`` - the pure per-device baseline helper.
  * the ``CycleDetector`` wiring - a stub ``terminal_drop_provider`` proves the
    ENDING state shortens the wait to ``TERMINAL_DROP_OFF_DELAY_SECONDS`` and
    that without the provider the full ``min_off_gap`` is honoured.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
)
from custom_components.ha_washdata.const import (
    STATE_INTERRUPTED,
    TERMINAL_DROP_OFF_DELAY_SECONDS,
    TerminationReason,
)
from custom_components.ha_washdata.profile_store import (
    device_active_peak_range,
    earliest_sustained_quiet_offset,
    is_terminal_drop,
)

_BASE = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _cycle(status: str, points: list[tuple[float, float]]) -> dict:
    return {
        "status": status,
        "start_time": _BASE.isoformat(),
        "power_data": [[float(o), float(p)] for o, p in points],
    }


# ── earliest_sustained_quiet_offset (per-device baseline) ───────────────────


def test_baseline_none_when_too_few_clean_cycles() -> None:
    # Only two completed cycles < the min-3 requirement -> cannot judge.
    cycles = [
        _cycle("completed", [(0, 100), (600, 100), (700, 0), (900, 0)]),
        _cycle("completed", [(0, 100), (600, 100), (700, 0), (900, 0)]),
    ]
    assert (
        earliest_sustained_quiet_offset(cycles, stop_threshold_w=5.0,
                                        min_quiet_span_s=60, min_clean_cycles=3)
        is None
    )


def test_baseline_is_earliest_first_quiet_across_completed() -> None:
    # Three completed cycles; the earliest sustained (>=60s) quiet span begins at
    # 700s in the third. The 300s dip in cycle 2 is only 40s long -> not quiet.
    cycles = [
        _cycle("completed", [(0, 200), (900, 200), (1000, 0), (1200, 0)]),
        _cycle("completed", [(0, 200), (300, 0), (340, 200), (900, 0), (1000, 0)]),
        _cycle("completed", [(0, 200), (700, 0), (900, 0)]),
    ]
    offset = earliest_sustained_quiet_offset(
        cycles, stop_threshold_w=5.0, min_quiet_span_s=60, min_clean_cycles=3
    )
    # cycle3 first sustained quiet at 700; cycle2's real quiet is at 900.
    assert offset == 700.0


def test_baseline_ignores_interrupted_cycles() -> None:
    # Interrupted cycles are the anomalies we are trying to catch; including them
    # would poison the baseline (drive it toward 0) and suppress detection.
    cycles = [
        _cycle("interrupted", [(0, 200), (120, 0), (300, 0)]),  # early quiet - ignored
        _cycle("completed", [(0, 200), (800, 0), (1000, 0)]),
        _cycle("completed", [(0, 200), (850, 0), (1000, 0)]),
        _cycle("completed", [(0, 200), (900, 0), (1100, 0)]),
    ]
    offset = earliest_sustained_quiet_offset(
        cycles, stop_threshold_w=5.0, min_quiet_span_s=60, min_clean_cycles=3
    )
    assert offset == 800.0  # the interrupted 120s quiet is excluded


# ── device_active_peak_range (familiarity baseline) ─────────────────────────


def test_peak_range_none_when_too_few_clean() -> None:
    cycles = [_cycle("completed", [(0, 2000), (600, 0)])]
    assert device_active_peak_range(cycles, min_clean_cycles=3) is None


def test_peak_range_min_max_over_completed() -> None:
    cycles = [
        _cycle("completed", [(0, 1800), (600, 0)]),
        _cycle("completed", [(0, 2100), (600, 0)]),
        _cycle("completed", [(0, 1950), (600, 0)]),
        _cycle("interrupted", [(0, 50), (60, 0)]),  # ignored - would skew the floor
    ]
    assert device_active_peak_range(cycles, min_clean_cycles=3) == (1800.0, 2100.0)


# ── is_terminal_drop (pure decision: anomaly + familiarity gates) ────────────

_STOP = 5.0
_EARLINESS = 0.8
_MIN_PEAK = 5.0
_TOL = 0.4


def _prefix_then_drop(peak: float, drop_at: float) -> list[tuple[float, float]]:
    """On at ``peak`` until ``drop_at``, then a sustained cliff to 0."""
    pts = [(o, peak) for o in range(0, int(drop_at), 30)]
    pts += [(drop_at + 30 * i, 0.0) for i in range(4)]
    return pts


def _decide(points, earliest, peak_range) -> bool:
    return is_terminal_drop(
        points, earliest, peak_range, _STOP, _EARLINESS, _MIN_PEAK, _TOL
    )


def test_terminal_when_early_familiar_and_clearly_on() -> None:
    # Peak 2000 within [1500,2100]*tol; drop at 120 < 700*0.8=560 -> terminal.
    pts = _prefix_then_drop(peak=2000.0, drop_at=120.0)
    assert _decide(pts, earliest=700.0, peak_range=(1500.0, 2100.0)) is True


def test_not_terminal_when_baseline_missing() -> None:
    pts = _prefix_then_drop(peak=2000.0, drop_at=120.0)
    assert _decide(pts, earliest=None, peak_range=(1500.0, 2100.0)) is False
    assert _decide(pts, earliest=700.0, peak_range=None) is False


def test_not_terminal_when_never_clearly_on() -> None:
    # Peak 20 < min_peak_ratio(5) * stop(5) = 25 -> device barely on, ignore.
    pts = _prefix_then_drop(peak=20.0, drop_at=120.0)
    assert _decide(pts, earliest=700.0, peak_range=(10.0, 24.0)) is False


def test_deferred_when_power_signature_is_novel() -> None:
    # The familiarity gate (user's request): the drop IS anomalously early, but
    # the cycle peaks at 2000 while this device has only ever peaked ~5000-6000.
    # Power unlike anything seen before -> possibly a NEW program -> defer.
    pts = _prefix_then_drop(peak=2000.0, drop_at=120.0)
    assert _decide(pts, earliest=700.0, peak_range=(5000.0, 6000.0)) is False
    # ...and the mirror case: a peak far ABOVE the historical band is novel too.
    pts_hi = _prefix_then_drop(peak=9000.0, drop_at=120.0)
    assert _decide(pts_hi, earliest=700.0, peak_range=(1500.0, 2100.0)) is False


def test_not_terminal_when_drop_not_anomalously_early() -> None:
    # Familiar + clearly on, but the drop begins at 600 >= 700*0.8=560, i.e. not
    # earlier than this device has legitimately gone quiet -> slow path.
    pts = _prefix_then_drop(peak=2000.0, drop_at=600.0)
    assert _decide(pts, earliest=700.0, peak_range=(1500.0, 2100.0)) is False


def test_not_terminal_without_trailing_quiet() -> None:
    # Still on at the end (no sustained sub-threshold tail) -> nothing to finalize.
    pts = [(o, 2000.0) for o in range(0, 200, 30)]
    assert _decide(pts, earliest=700.0, peak_range=(1500.0, 2100.0)) is False


# ── CycleDetector wiring (fast finalize) ────────────────────────────────────


def _config() -> CycleDetectorConfig:
    # Washing-machine-like: 8-min soak-bridging gap dominates the end wait.
    return CycleDetectorConfig(min_power=5.0, off_delay=180, min_off_gap=480)


def _detector(provider) -> CycleDetector:
    ended: dict = {}
    det = CycleDetector(
        config=_config(),
        on_state_change=lambda *_a: None,
        on_cycle_end=lambda cd: ended.update(cd),
        terminal_drop_provider=provider,
    )
    det._captured_end = ended  # type: ignore[attr-defined]
    return det


def _run_cycle_then_drop(det: CycleDetector, zero_seconds: int) -> None:
    """Feed a short high-power run (reaches RUNNING), then a cliff to 0."""
    t = _BASE
    # ~3 minutes of clearly-ON power (well above 5*stop_threshold) so the state
    # machine confirms RUNNING before the drop.
    for _ in range(6):
        det.process_reading(2000.0, t)
        t += timedelta(seconds=30)
    # Cliff to zero, sampled every 30s, for the requested span.
    for _ in range(int(zero_seconds / 30) + 1):
        det.process_reading(0.0, t)
        t += timedelta(seconds=30)


def test_terminal_provider_finalizes_early() -> None:
    # Provider says "anomalous early drop" -> cycle finalized well before the
    # 480s min_off_gap would have.
    det = _detector(lambda points, dur: True)
    _run_cycle_then_drop(det, zero_seconds=240)
    assert det.state == STATE_INTERRUPTED
    assert det._captured_end["termination_reason"] == TerminationReason.TERMINAL_DROP
    assert det._captured_end["status"] == "interrupted"


def test_no_provider_waits_out_full_gap() -> None:
    # No provider -> proven behavior: still RUNNING/PAUSED/ENDING at the point a
    # terminal drop would have fired, because the 480s gap has not elapsed.
    det = _detector(None)
    _run_cycle_then_drop(det, zero_seconds=240)
    assert det.state != STATE_INTERRUPTED
    assert "termination_reason" not in det._captured_end


def test_provider_false_waits_out_full_gap() -> None:
    # Provider present but says "not anomalous" (e.g. this device does soak this
    # early) -> honour the full soak-bridging wait, do not finalize early.
    det = _detector(lambda points, dur: False)
    _run_cycle_then_drop(det, zero_seconds=240)
    assert det.state != STATE_INTERRUPTED


def test_provider_exception_is_swallowed() -> None:
    def boom(points, dur):
        raise RuntimeError("anomaly model blew up")

    det = _detector(boom)
    # Must not raise and must not finalize early (falls back to slow path).
    _run_cycle_then_drop(det, zero_seconds=240)
    assert det.state != STATE_INTERRUPTED
