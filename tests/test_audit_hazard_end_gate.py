"""Audit 2026-10-02 DETECT-16: the hazard end gate, and LIVE-18's prefix flags.

The ENDING fallback waited max(off_delay, min_off_gap) - a blind per-device
prior for a soak - until 0.90-1.05x expected. It now waits 1.25x the longest
below-stop pause the matched profile's own traced cycles ever resumed from at or
after this quiet's position: shorten-only, unambiguous matches, >= 3 cycles.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from custom_components.ha_washdata.cycle_detector import CycleDetector
from custom_components.ha_washdata.detector_config import build_detector_config
from custom_components.ha_washdata.profile_store import MatchResult, _match_prefix_ambiguity
from custom_components.ha_washdata.signal_processing import resumed_pauses

T0 = datetime(2026, 5, 1, 8, 0, tzinfo=timezone.utc)
# Three traced cycles: a 30 min soak at 40% and a 2 min pause at 92%.
CATALOGUE = (3, ((0.40, 1800.0), (0.92, 120.0)) * 3)


def _det(catalogue=CATALOGUE, *, ambiguous: bool = False, prefix: bool = False) -> CycleDetector:
    det = CycleDetector(
        build_detector_config({"off_delay": 60, "min_off_gap": 2400}, {}, "washing_machine"),
        MagicMock(), MagicMock(),
    )
    det.update_match(("Cotton", 0.8, 7200.0, None, False, ambiguous, prefix, False,
                      None, None, None, 0.0, None, catalogue))
    det._current_cycle_start = T0  # noqa: SLF001
    return det


def _wait(det: CycleDetector, quiet_started_at_s: float, quiet_s: float = 60.0) -> float:
    det._time_below_threshold = quiet_s  # noqa: SLF001
    return det._hazard_wait(T0 + timedelta(seconds=quiet_started_at_s + quiet_s), 2400.0)  # noqa: SLF001


def test_past_every_soak_it_waits_only_for_the_late_pauses() -> None:
    assert _wait(_det(), 0.95 * 7200) == 150.0          # 1.25 x 120 s


def test_before_the_soak_it_waits_the_soak_out_with_margin() -> None:
    assert _wait(_det(), 0.30 * 7200) == 2250.0         # 1.25 x 1800 s


def test_it_never_goes_below_off_delay_or_above_the_old_wait() -> None:
    assert _wait(_det((3, ((0.1, 5.0),) * 3)), 0.95 * 7200) == 60.0


def test_it_does_nothing_without_evidence_or_with_an_ambiguous_match() -> None:
    assert _wait(_det(None), 0.95 * 7200) == 2400.0
    assert _wait(_det((2, ((0.92, 120.0),) * 2)), 0.95 * 7200) == 2400.0
    assert _wait(_det(ambiguous=True), 0.95 * 7200) == 2400.0


def test_the_retired_prefix_fit_element_no_longer_holds_the_hazard_gate() -> None:
    """Element 7 (the #364 prefix-fit flag) was removed in 0.5.8; a legacy tuple
    still setting it gets the hazard wait like any unambiguous match."""
    assert _wait(_det(prefix=True), 0.95 * 7200) == 150.0


def test_a_shorter_tuple_and_a_reset_clear_the_catalogue() -> None:
    det = _det()
    det.update_match(("Cotton", 0.8, 7200.0, None, False, False, False, False))
    assert det._matched_pause_catalogue is None  # noqa: SLF001
    det = _det()
    det.reset()
    assert det._matched_pause_catalogue is None  # noqa: SLF001


def test_resumed_pauses_skip_the_leading_standby_and_the_open_end() -> None:
    pts = [(0, 0.0), (60, 500.0), (120, 0.0), (300, 500.0), (600, 0.0), (900, 0.0)]
    assert resumed_pauses(pts, 2.0) == [(120 / 900, 180.0)]


def test_only_the_full_shape_flag_is_left_and_a_prefix_score_cannot_set_it() -> None:
    """LIVE-18 took the #288 term off the ENDING gates; 0.5.8 removed the #364
    prefix-fit flag those gates read instead. What is left is the #288 term, for
    the anti-crease finalize, and a `prefix_score` no longer reaches it."""
    win = {"name": "Quick", "profile_duration": 2760.0, "shape_score": 0.70, "score": 0.61}
    longer = {"name": "Normal", "profile_duration": 5280.0, "shape_score": 0.70, "score": 0.44}
    assert _match_prefix_ambiguity([win, longer], 2760.0) is True
    poor = dict(longer, shape_score=0.20, prefix_score=0.95)
    assert _match_prefix_ambiguity([win, poor], 2760.0) is False
    assert "is_prefix_ambiguous" not in MatchResult.__dataclass_fields__
    assert "is_prefix_ambiguous_full_shape" in MatchResult.__dataclass_fields__
