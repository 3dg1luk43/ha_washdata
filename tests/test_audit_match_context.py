"""Audit 2026-10-02 DETECT-15: the match handed to the detector is built by name.

It was a 13-, then 14-element positional tuple built in two places (manager and
Playground); items 351/384/387a were each "forgot element N in one producer".
"""

from __future__ import annotations

from unittest.mock import MagicMock

from custom_components.ha_washdata.cycle_detector import CycleDetector, MatchContext
from custom_components.ha_washdata.detector_config import build_detector_config


def test_every_named_field_reaches_its_detector_state() -> None:
    det = CycleDetector(build_detector_config({}, {}, "washing_machine"), MagicMock(), MagicMock())
    det.update_match(MatchContext(
        profile_name="Cotton", confidence=0.7, expected_duration=3600.0,
        is_ambiguous=True, is_prefix_ambiguous=True, is_prefix_ambiguous_full_shape=False,
        tail_power=1.5, terminal_quiet_s=300.0, longest_candidate_s=5400.0,
        trusted_min_s=3000.0, pause_catalogue=(3, ((0.5, 120.0),)),
    ))
    assert det._matched_profile == "Cotton"  # noqa: SLF001
    assert det._expected_duration == 3600.0  # noqa: SLF001
    assert det._match_ambiguous is True  # noqa: SLF001
    assert det._match_prefix_ambiguous is True  # noqa: SLF001
    assert det._match_prefix_ambiguous_full_shape is False  # noqa: SLF001
    assert det._matched_tail_power == 1.5  # noqa: SLF001
    assert det._matched_terminal_quiet_s == 300.0  # noqa: SLF001
    assert det._longest_candidate_duration == 5400.0  # noqa: SLF001
    assert det._matched_trusted_min_s == 3000.0  # noqa: SLF001
    assert det._matched_pause_catalogue == (3, ((0.5, 120.0),))  # noqa: SLF001


def test_the_context_is_the_legacy_sequence() -> None:
    ctx = MatchContext("A", 0.5, 60.0)
    assert len(ctx) == 14 and ctx[0] == "A" and ctx[11] == 0.0 and ctx[13] is None
    assert tuple(ctx.as_sequence()) == (
        "A", 0.5, 60.0, None, False, False, False, False, None, None, None, 0.0, None, None,
    )
