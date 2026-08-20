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
"""Issue #364: a running wash must not be finalized at a shorter profile's length.

Reported on a Miele washer: a ~119 min programme was closed with
termination_reason "smart" at 4638 s because the live matcher had locked onto the
shorter "Oberhemden 40" profile (4731 s), and 4638/4731 = 0.980 is exactly the
ratio both Smart-Termination paths fire at. Power at that moment was still
100-165 W, far above the 5 W stop threshold. The remaining rinse/spin was then
recorded as a second cycle.

Root cause: both paths key on `elapsed >= 0.98 * expected` and neither asked
whether the appliance was still working. The #288 prefix-landscape guard could
not help - in one reported case the programme actually running had never been
trained, so no longer candidate existed to notice.

The fix compares the trailing mean power against what the matched profile itself
draws at its own end (`profile_store.profile_tail_power`, pushed to the detector
as element 9 of the match tuple). Shorten-only: it can only block an early
finish, never end a cycle sooner, and with no tail power it is inert.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

from custom_components.ha_washdata.const import (
    DEVICE_TYPE_WASHING_MACHINE,
    STATE_ENDING,
    STATE_FINISHED,
    STATE_PAUSED,
    STATE_RUNNING,
    TerminationReason,
)
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
)

_BASE = datetime(2026, 8, 7, 9, 0, 0, tzinfo=timezone.utc)

# The reported numbers: the matcher committed a 4731 s profile while the machine
# was really running a ~7140 s programme, and 0.98 * 4731 = 4636 s.
SHORT_EXPECTED = 4731.0
SMART_ANCHOR = int(SHORT_EXPECTED * 0.98)
# "Oberhemden"-class washer tail: the last few % of that profile average ~25 W.
PROFILE_TAIL_W = 25.0


def _dt(seconds: float) -> datetime:
    return _BASE + timedelta(seconds=seconds)


def _make_detector(
    completed: list[dict],
    *,
    anti_wrinkle: bool = False,
    match_confidence_threshold: float = 0.4,
) -> CycleDetector:
    config = CycleDetectorConfig(
        min_power=5.0,
        off_delay=150,
        interrupted_min_seconds=150,
        completion_min_seconds=600,
        start_duration_threshold=0.0,
        min_off_gap=480,
        stop_threshold_w=5.0,
        start_threshold_w=6.3,
        device_type=DEVICE_TYPE_WASHING_MACHINE,
        anti_wrinkle_enabled=anti_wrinkle,
        anti_wrinkle_max_power=400.0,
        match_confidence_threshold=match_confidence_threshold,
    )
    return CycleDetector(
        config=config,
        on_state_change=Mock(),
        on_cycle_end=lambda data: completed.append(data),
    )


def _match_tuple(tail_power: float | None) -> tuple:
    """The live match as the manager pushes it: a confident, unambiguous commit to
    the SHORT profile - the situation #364 reports."""
    return (
        "Oberhemden 40",
        0.60,
        SHORT_EXPECTED,
        None,
        False,  # is_confident_mismatch
        False,  # is_ambiguous
        False,  # is_prefix_ambiguous (widened) - the guard the report shows failing
        False,  # is_prefix_ambiguous_full_shape (legacy)
        tail_power,
    )


def _run_wash(detector: CycleDetector, tail_power: float | None) -> None:
    """Drive a real washing cycle up to the short profile's 0.98 anchor.

    Shaped like the reporter's trace: a drum pulsing between a near-zero baseline
    and 100-165 W, with an opening heating burst so `_cycle_max_power` clears
    anti_wrinkle_max_power (the anti-crease gate requires an energetic cycle).
    """
    detector.process_reading(2000.0, _dt(0))
    detector.process_reading(2000.0, _dt(10))
    assert detector.state == STATE_RUNNING
    for t in range(40, SMART_ANCHOR + 1, 10):
        # 3 W dip every third sample, 100-165 W otherwise.
        power = 3.0 if (t // 10) % 3 == 0 else (165.0 if (t // 10) % 2 else 100.0)
        detector.process_reading(power, _dt(t))
        if t % 300 == 0:
            detector.update_match(_match_tuple(tail_power))
    detector.update_match(_match_tuple(tail_power))


def _arm_ending_then_keep_washing(
    detector: CycleDetector, tail_power: float | None
) -> int:
    """Reproduce the exact path-A sequence #364 needs, and return the offset of the
    sample at which Smart Termination is decided.

    This is the non-obvious part of the bug. A single ~60 s dip is enough to walk
    RUNNING -> PAUSED -> ENDING (the dynamic thresholds are only ~30 s / ~45 s at a
    10 s cadence). Once in ENDING, `_time_in_state` accrues on EVERY reading
    regardless of power, and past 0.98 x expected a power burst no longer returns
    the detector to RUNNING - so the machine can resume washing at 150 W for the
    whole 240 s washer debounce while the fast end-path arms itself. The next
    sub-threshold sample then finalizes. That is how a cycle gets closed as "smart"
    with the drum still turning.
    """
    t = SMART_ANCHOR + 10
    for _ in range(6):  # ~60 s quiet: RUNNING -> PAUSED -> ENDING
        detector.process_reading(1.0, _dt(t))
        t += 10
    assert detector.state == STATE_ENDING, detector.state

    for _ in range(30):  # ~300 s back at wash power, still in ENDING
        detector.process_reading(150.0, _dt(t))
        detector.update_match(_match_tuple(tail_power))
        t += 10
    assert detector.state == STATE_ENDING, detector.state
    assert detector._time_in_state >= 240.0

    detector.process_reading(1.0, _dt(t))  # the decision sample
    return t


def test_path_a_blocked_while_still_washing() -> None:
    """The reported case: ENDING Smart Termination must not fire at the short
    profile's 0.98 anchor while the drum is still pulling ~150 W."""
    completed: list[dict] = []
    detector = _make_detector(completed)
    _run_wash(detector, PROFILE_TAIL_W)
    _arm_ending_then_keep_washing(detector, PROFILE_TAIL_W)

    assert not completed, (
        "Smart Termination fired mid-wash: the cycle was split at the shorter "
        f"profile's length ({completed[0]['duration'] if completed else 0:.0f}s)"
    )
    assert detector.state == STATE_ENDING


def test_path_a_fires_when_power_really_drops() -> None:
    """Same trace, but the machine actually finishes: trailing power falls to the
    profile's own tail level, so Smart Termination fires as it always did."""
    completed: list[dict] = []
    detector = _make_detector(completed)
    _run_wash(detector, PROFILE_TAIL_W)

    # Real end: a quiet tail at the profile's tail level.
    for t in range(SMART_ANCHOR + 10, SMART_ANCHOR + 700, 10):
        detector.process_reading(0.0, _dt(t))

    assert completed, "Smart Termination must still end a genuinely finished cycle"
    assert completed[0]["termination_reason"] == TerminationReason.SMART
    assert detector.state == STATE_FINISHED


def test_guard_inert_without_tail_power() -> None:
    """No tail power (short tuple, or a profile with no usable trace) means no
    opinion: behaviour is identical to before the guard existed."""
    completed: list[dict] = []
    detector = _make_detector(completed)
    _run_wash(detector, None)
    assert detector._matched_tail_power is None
    _arm_ending_then_keep_washing(detector, None)

    # Pre-fix behaviour, asserted deliberately: with no tail power the guard has no
    # opinion and the split still happens. This is what pins the fix to the guard
    # rather than to some incidental change in the surrounding state machine - the
    # identical trace with a tail power does NOT split (test above).
    assert completed, "with no tail power the guard must stay out of the way"
    assert completed[0]["termination_reason"] == TerminationReason.SMART


def test_short_tuples_leave_new_fields_at_safe_defaults() -> None:
    """5/6/7-element tuples (the manual-override path and every pre-#364 caller)
    must not switch the guard on, and must not loosen the anti-crease gate."""
    completed: list[dict] = []
    detector = _make_detector(completed)
    detector.process_reading(100.0, _dt(0))
    detector.process_reading(100.0, _dt(10))

    detector.update_match(("P", 0.7, 3600.0, None, False, True, True))
    assert detector._matched_tail_power is None
    # No element 8: the narrow flag mirrors the widened one, so the anti-crease
    # gate is exactly as conservative as it was before #364.
    assert detector._match_prefix_ambiguous is True
    assert detector._match_prefix_ambiguous_full_shape is True

    detector.update_match(("P", 0.7, 3600.0, None, False, False))
    assert detector._match_prefix_ambiguous is False
    assert detector._match_prefix_ambiguous_full_shape is False


def test_non_finite_tail_power_is_ignored() -> None:
    """A NaN/inf/zero/garbage tail power must degrade to "no opinion", never to a
    divide-by-zero or a permanent block."""
    completed: list[dict] = []
    detector = _make_detector(completed)
    detector.process_reading(100.0, _dt(0))
    detector.process_reading(100.0, _dt(10))
    for bad in (float("nan"), float("inf"), 0.0, -5.0, "x", None):
        detector.update_match(
            ("P", 0.7, 3600.0, None, False, False, False, False, bad)
        )
        assert detector._matched_tail_power is None, bad
        assert detector._smart_term_power_plausible(_dt(20)) is True


def test_anticrease_finalize_blocked_mid_wash_then_fires_on_real_tail() -> None:
    """Path B (#296 anti-crease finalize) is reachable straight from RUNNING and
    only needs 180 s below anti_wrinkle_max_power - which a washer's whole wash
    phase satisfies. It must not fire mid-wash, but must still fire on the real
    low-power tumble tail, or #364's fix would re-create the #296 hang.
    """
    completed: list[dict] = []
    detector = _make_detector(completed, anti_wrinkle=True)
    _run_wash(detector, PROFILE_TAIL_W)

    # Still washing at 100-165 W, past 0.98 x the (wrong, short) expected.
    for t in range(SMART_ANCHOR + 10, SMART_ANCHOR + 400, 10):
        detector.process_reading(150.0, _dt(t))
    assert not completed, "anti-crease finalize fired during the wash phase"

    # Now the genuine anti-crease tail: a low baseline with small tumble bursts,
    # averaging around the profile's own tail level.
    t = SMART_ANCHOR + 400
    end = t + 900
    while t < end:
        detector.process_reading(60.0 if (t // 10) % 8 == 0 else 8.0, _dt(t))
        detector.update_match(_match_tuple(PROFILE_TAIL_W))
        t += 10

    assert completed, "anti-crease finalize must still fire on the real tail (#296)"
    assert completed[0]["termination_reason"] == TerminationReason.SMART


def test_match_confidence_threshold_is_honoured() -> None:
    """`profile_match_threshold` was dead config: raising it (the workaround the
    #288 reporter documented) did nothing. Now a threshold above the match's
    confidence suppresses Smart Termination."""
    completed: list[dict] = []
    detector = _make_detector(completed, match_confidence_threshold=0.75)
    _run_wash(detector, None)  # guard inert, so only the threshold can block
    _arm_ending_then_keep_washing(detector, None)

    assert not completed, (
        "conf 0.60 is below the configured 0.75 threshold, so the fast end-path "
        "must be skipped"
    )

    # Default threshold, same trace: the fast path is available again.
    baseline: list[dict] = []
    det2 = _make_detector(baseline, match_confidence_threshold=0.4)
    _run_wash(det2, None)
    _arm_ending_then_keep_washing(det2, None)
    assert baseline and baseline[0]["termination_reason"] == TerminationReason.SMART


def test_block_reason_reports_still_active() -> None:
    """The #346 diagnostic must name the new blocker, so a late finish caused by
    this guard is traceable in the log."""
    reason = CycleDetector._smart_term_block_reason(
        4700.0, SHORT_EXPECTED, 0.98, True, False, False, False
    )
    assert reason == "still_active"
    # Order: the pre-existing reasons still win, so existing logs are unchanged.
    assert (
        CycleDetector._smart_term_block_reason(
            4700.0, SHORT_EXPECTED, 0.98, True, True, False, False
        )
        == "match_ambiguous"
    )
    assert (
        CycleDetector._smart_term_block_reason(
            4700.0, SHORT_EXPECTED, 0.98, True, False, False, True
        )
        is None
    )


def test_tail_power_survives_snapshot_roundtrip() -> None:
    """A restart must not silently disarm the guard, and a pre-#364 snapshot must
    restore the conservative anti-crease behaviour."""
    completed: list[dict] = []
    detector = _make_detector(completed)
    detector.process_reading(100.0, _dt(0))
    detector.process_reading(100.0, _dt(10))
    detector.update_match(
        ("P", 0.7, 3600.0, None, False, False, True, False, PROFILE_TAIL_W)
    )

    snap = detector.get_state_snapshot()
    assert snap["matched_tail_power"] == PROFILE_TAIL_W
    assert snap["match_prefix_ambiguous_full_shape"] is False

    restored = _make_detector([])
    restored.restore_state_snapshot(snap)
    assert restored._matched_tail_power == PROFILE_TAIL_W
    assert restored._match_prefix_ambiguous_full_shape is False

    # Pre-#364 snapshot: no narrow flag, so it must fall back to the widened one
    # rather than defaulting to False (which would LOOSEN the anti-crease gate).
    legacy = dict(snap)
    legacy.pop("match_prefix_ambiguous_full_shape")
    legacy.pop("matched_tail_power")
    older = _make_detector([])
    older.restore_state_snapshot(legacy)
    assert older._match_prefix_ambiguous is True
    assert older._match_prefix_ambiguous_full_shape is True
    assert older._matched_tail_power is None
