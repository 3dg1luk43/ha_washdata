"""Audit 2026-11-03 item 2: a per-run end expectation from the nearest members.

Every washer / washer-dryer / dryer end gate keyed on one number, the matched
profile's mean stored duration, blind to the load. The match now also carries the
expected duration of THIS run: the mean of the gate profile's two members nearest
to it in energy over the same elapsed stretch, when they agree
(``analysis.nearest_member_duration`` -> ``MatchResult.gate_run_expected`` ->
``match_rules.GateInputs.run_expected`` -> MatchContext element 17). The detector's
Smart Termination bar, late shortening and finish deferral read it below the
profile's (``CycleDetector._gate_expected_s``): shorten-only, floored, never while
the match is ambiguous. The anti-crease finalise keeps the profile's.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock

import pytest

from custom_components.ha_washdata import analysis, match_rules
from custom_components.ha_washdata.const import (
    DEVICE_TYPE_WASHING_MACHINE,
    RUN_EXPECT_FLOOR_RATIO,
    STATE_FINISHED,
    STATE_OFF,
)
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
    MatchContext,
    TerminationReason,
)
from custom_components.ha_washdata.detector_config import build_detector_config
from custom_components.ha_washdata.profile_store import MatchResult, ProfileStore


def _member(watts: float, duration: float, dt: float = 60.0, heater_s: float = 0.0) -> dict:
    n = int(duration / dt)
    k = int(heater_s / dt)
    power = [2000.0] * k + [watts] * (n - k)
    return {"sample_power": power, "sample_span_s": duration, "duration": duration}


def _heated(heater_s: float, duration: float) -> dict:
    """A washer run: a 2 kW heater block first (its length tracks the load), then
    a 100 W drum phase; a longer heater means a longer programme."""
    return _member(100.0, duration, heater_s=heater_s)


# --- analysis.nearest_member_duration ----------------------------------------

def test_the_two_members_nearest_in_energy_set_the_run_expectation() -> None:
    members = [_heated(600.0, 3000.0), _heated(660.0, 3060.0), _heated(1500.0, 4200.0),
               _heated(1800.0, 4800.0)]
    # A finished 3030 s run with a 630 s heater: the two short members are nearest
    # (the long ones had used twice its energy by 3030 s already).
    run = _heated(630.0, 3030.0)["sample_power"]
    out = analysis.nearest_member_duration(run, 3030.0, members)
    assert out == pytest.approx(3030.0)


def test_a_longer_member_is_graded_on_its_own_prefix() -> None:
    """At 2000 s a run of the same power is as far into every member that is still
    running: their prefix energies tie with it, so they are the nearest."""
    members = [_member(1000.0, 1500.0), _member(1000.0, 4000.0), _member(1000.0, 4100.0),
               _member(1000.0, 4050.0)]
    out = analysis.nearest_member_duration([1000.0] * 50, 2000.0, members)
    assert out is not None and out > 4000.0


def test_members_that_disagree_or_sit_far_away_give_nothing() -> None:
    members = [_member(1000.0, 3000.0), _member(1000.0, 3600.0), _member(500.0, 9000.0)]
    # The two nearest (3000 / 3600 s) disagree by ln(1.2) > the spread bar.
    assert analysis.nearest_member_duration([1000.0] * 50, 3300.0, members) is None
    # Nothing within the energy bar of a 2 kW run.
    far = [_member(1000.0, 3000.0), _member(1000.0, 3050.0), _member(1000.0, 3100.0)]
    assert analysis.nearest_member_duration([2000.0] * 50, 3000.0, far) is None


def test_too_few_members_or_garbage_never_raise() -> None:
    two = [_member(1000.0, 3000.0), _member(1000.0, 3000.0)]
    assert analysis.nearest_member_duration([1000.0] * 50, 3000.0, two) is None
    junk = [{"sample_power": None}, {"duration": "x"}, {}, _member(1000.0, 3000.0)]
    assert analysis.nearest_member_duration([1000.0] * 50, 3000.0, junk) is None
    assert analysis.nearest_member_duration([], 3000.0, two * 2) is None
    assert analysis.nearest_member_duration([1000.0] * 5, 0.0, two * 2) is None


# --- the store and match_rules carry it to the detector ------------------------

def test_the_store_reads_the_named_profiles_member_templates() -> None:
    snaps = {"Cotton": {"members": [_heated(600.0, 3000.0), _heated(660.0, 3060.0),
                                    _heated(1500.0, 4200.0)]},
             "Wool": {}}
    run = _heated(630.0, 3030.0)["sample_power"]
    out = ProfileStore._run_expected_for("Cotton", snaps, run, 3030.0)  # noqa: SLF001
    assert out == pytest.approx(3030.0)
    assert ProfileStore._run_expected_for("Wool", snaps, [1000.0] * 50, 3030.0) is None  # noqa: SLF001
    assert ProfileStore._run_expected_for(None, snaps, [1000.0] * 50, 3030.0) is None  # noqa: SLF001


def test_the_envelope_build_records_each_members_programme_end() -> None:
    from custom_components.ha_washdata.profile_store import _member_programme_ends

    # 40 min of 2 kW / 400 W work, then a 20 min anti-crease tail at 3 W.
    work = [(float(t), 2000.0 if t < 1200 else 400.0) for t in range(0, 2400, 10)]
    tail = [(float(t), 3.0) for t in range(2400, 3600, 10)]
    flat = [(float(t), 400.0) for t in range(0, 3600, 10)]
    ends = _member_programme_ends([({"id": "a"}, work + tail), ({"id": "b"}, flat),
                                   ({"id": None}, work + tail), ({"id": "c"}, [])])
    assert set(ends) == {"a"} and 2300.0 <= ends["a"] <= 2500.0
    assert _member_programme_ends([({"id": "b"}, flat)]) is None


def _tick() -> match_rules.MatchTick:
    return match_rules.MatchTick("Cotton", 0.8, 3600.0, None, 1800.0, 0.0, 0.2, True)


def test_gate_inputs_carry_it_only_from_the_gate_view() -> None:
    res = MatchResult("Cotton", 0.8, 3600.0, None, [], False, 0.2, gate_profile="Cotton",
                      gate_confidence=0.8, gate_expected_duration=3600.0,
                      gate_ambiguous=False, gate_prefix_full_shape=False,
                      gate_run_expected=3300.0)
    assert match_rules.detector_inputs(_tick(), res).run_expected == 3300.0
    plain = MatchResult("Cotton", 0.8, 3600.0, None, [], False, 0.2, gate_run_expected=3300.0)
    assert match_rules.detector_inputs(_tick(), plain).run_expected is None
    bad = MatchResult("Cotton", 0.8, 3600.0, None, [], False, 0.2, gate_ambiguous=False,
                      gate_run_expected=float("inf"))
    assert match_rules.detector_inputs(_tick(), bad).run_expected is None


def test_both_producers_send_element_17() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "custom_components" / "ha_washdata"
    for module in ("manager.py", "playground.py"):
        assert "run_expected_s=gate.run_expected" in (root / module).read_text(), module


# --- the detector -------------------------------------------------------------

def _det(run=None, *, ambiguous=False, expected=7200.0) -> CycleDetector:
    det = CycleDetector(build_detector_config({}, {}, "washing_machine"), MagicMock(), MagicMock())
    det.update_match(MatchContext("Cotton", 0.8, expected, is_ambiguous=ambiguous,
                                  run_expected_s=run))
    return det


def test_gates_read_the_shorter_run_expectation_floored() -> None:
    assert _det(6000.0)._gate_expected_s("smart") == 6000.0  # noqa: SLF001
    assert _det(6000.0)._gate_expected_s("late") == 6000.0  # noqa: SLF001
    assert _det(6000.0)._gate_expected_s("defer") == 6000.0  # noqa: SLF001
    floor = RUN_EXPECT_FLOOR_RATIO * 7200.0
    assert _det(1000.0)._gate_expected_s("smart") == floor  # noqa: SLF001


def test_never_longer_never_ambiguous_never_another_gate() -> None:
    assert _det(9000.0)._gate_expected_s("smart") == 7200.0  # noqa: SLF001
    assert _det(6000.0, ambiguous=True)._gate_expected_s("smart") == 7200.0  # noqa: SLF001
    assert _det(None)._gate_expected_s("smart") == 7200.0  # noqa: SLF001
    assert _det(6000.0)._gate_expected_s("anticrease") == 7200.0  # noqa: SLF001
    # The profile's own expected duration, which the anti-crease finalise reads,
    # is untouched.
    assert _det(6000.0)._expected_duration == 7200.0  # noqa: SLF001


def test_a_shorter_tuple_or_a_revoke_clears_it() -> None:
    det = _det(6000.0)
    det.update_match(("Cotton", 0.8, 7200.0, None, False))
    assert det._matched_run_expected_s is None  # noqa: SLF001
    det = _det(6000.0)
    det.update_match(MatchContext(None, 0.0, 0.0, is_confident_mismatch=True))
    assert det._matched_run_expected_s is None  # noqa: SLF001
    det = _det(6000.0)
    det.reset()
    assert det._matched_run_expected_s is None  # noqa: SLF001


def test_it_survives_a_restart_snapshot() -> None:
    det = _det(6000.0)
    snap = det.get_state_snapshot()
    other = _det(None)
    other.restore_state_snapshot(snap)
    assert other._matched_run_expected_s == 6000.0  # noqa: SLF001
    snap.pop("matched_run_expected_s")
    other.restore_state_snapshot(snap)
    assert other._matched_run_expected_s is None  # noqa: SLF001


def _at(offset: float) -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0) + timedelta(seconds=offset)


def _washer_stopping_at_86pct(run_expected):
    """A confident washer on a 10000 s profile that stops at 8600 s on a 3.5 W
    standby (energetic enough to hold the plain fallback's energy gate), so Smart
    Termination decides: test_smart_termination_duration_ratio's scenario."""
    matcher = Mock(side_effect=lambda readings: MatchContext(
        "Mix", 0.9, 10000.0, "Washing", run_expected_s=run_expected,
    ))
    on_end = Mock()
    cfg = CycleDetectorConfig(
        min_power=5.0, off_delay=60, device_type=DEVICE_TYPE_WASHING_MACHINE,
        completion_min_seconds=600, start_duration_threshold=0.0,
        start_energy_threshold=0.0, start_threshold_w=6.0, stop_threshold_w=4.0,
        min_off_gap=60,
    )
    det = CycleDetector(config=cfg, on_state_change=Mock(), on_cycle_end=on_end,
                        profile_matcher=matcher)
    for t in range(0, 8600, 30):
        det.process_reading(100.0, _at(t))
    for t in range(8600, 9000, 30):
        det.process_reading(3.5, _at(t))
    return det, on_end


def test_smart_termination_ends_a_short_run_at_its_own_expectation() -> None:
    det, on_end = _washer_stopping_at_86pct(8700.0)
    assert det.state in (STATE_FINISHED, STATE_OFF)
    assert on_end.call_args[0][0].get("termination_reason") == TerminationReason.SMART
    # Without one the same run still waits for 98% of the profile.
    det, on_end = _washer_stopping_at_86pct(None)
    assert det.state not in (STATE_FINISHED, STATE_OFF)
    assert not on_end.called
