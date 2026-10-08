"""Live bugs the F7 parity harness surfaced once the Playground ran the manager's rules.

1. A dishwasher's verified pause could never be released once the ENDING match
   freeze engaged: the #375 sustained-quiet release ran only inside the manager's
   match tick, and the freeze stops that tick. Every ENDING finalize stays gated on
   `not _verified_pause`, so the cycle ran to the force stop (~8 h on a plug that
   keeps reporting 0 W; three TRON4R dishwasher cycles).
2. A divergence revert handed the detector the profile name "detecting...", which
   became its matched programme with the abandoned winner's expected duration.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

from custom_components.ha_washdata import match_rules
from custom_components.ha_washdata.const import STATE_ENDING
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
    MatchContext,
)

T0 = datetime(2026, 1, 5, 8, 0, tzinfo=timezone.utc)


def _frozen_dishwasher(*, user_paused: bool, elapsed_s: float, quiet_s: float) -> CycleDetector:
    det = CycleDetector(
        config=CycleDetectorConfig(min_power=5.0, off_delay=180, device_type="dishwasher"),
        on_state_change=Mock(),
        on_cycle_end=Mock(),
        profile_matcher=Mock(),
    )
    det._state = STATE_ENDING  # noqa: SLF001
    det._matched_profile = "Eco"  # noqa: SLF001
    det._expected_duration = 3600.0  # noqa: SLF001
    det._current_cycle_start = T0  # noqa: SLF001
    det._power_readings = [(T0, 1800.0), (T0 + timedelta(seconds=elapsed_s), 0.0)]  # noqa: SLF001
    det._time_below_threshold = quiet_s  # noqa: SLF001
    det._time_below_threshold_gapfree = quiet_s  # noqa: SLF001
    det.set_verified_pause(True)
    det.set_user_paused(user_paused)
    return det


def test_a_frozen_dishwasher_releases_its_verified_pause_once_quiet_past_expected():
    det = _frozen_dishwasher(user_paused=False, elapsed_s=4000.0, quiet_s=700.0)
    det._try_profile_match(T0 + timedelta(seconds=4000))  # noqa: SLF001
    assert det._verified_pause is False  # noqa: SLF001
    det._profile_matcher.assert_not_called()  # noqa: SLF001  (still frozen: no re-match)


def test_the_release_keeps_the_375_conditions():
    # Not yet at the expected duration, or not quiet long enough: the pause holds.
    for elapsed, quiet in ((3000.0, 700.0), (4000.0, 400.0)):
        det = _frozen_dishwasher(user_paused=False, elapsed_s=elapsed, quiet_s=quiet)
        det._try_profile_match(T0 + timedelta(seconds=elapsed))  # noqa: SLF001
        assert det._verified_pause is True, (elapsed, quiet)  # noqa: SLF001


def test_a_user_pause_is_never_released_by_the_freeze():
    det = _frozen_dishwasher(user_paused=True, elapsed_s=4000.0, quiet_s=700.0)
    det._try_profile_match(T0 + timedelta(seconds=4000))  # noqa: SLF001
    assert det._verified_pause is True  # noqa: SLF001


def test_a_divergence_revert_revokes_the_detector_match():
    reverted = SimpleNamespace(profile_name=match_rules.DETECTING)
    assert match_rules.detector_match(reverted, SimpleNamespace(is_confident_mismatch=False)) == (None, True)
    kept = SimpleNamespace(profile_name="Eco")
    assert match_rules.detector_match(kept, SimpleNamespace(is_confident_mismatch=False)) == ("Eco", False)

    det = CycleDetector(
        config=CycleDetectorConfig(min_power=5.0, off_delay=180, device_type="dishwasher"),
        on_state_change=Mock(),
        on_cycle_end=Mock(),
    )
    det.update_match(MatchContext(profile_name="Eco", confidence=0.8, expected_duration=3600.0))
    assert det._matched_profile == "Eco"  # noqa: SLF001
    name, revoke = match_rules.detector_match(reverted, SimpleNamespace(is_confident_mismatch=False))
    det.update_match(MatchContext(
        profile_name=name, confidence=0.3, expected_duration=3600.0, is_confident_mismatch=revoke,
    ))
    assert det._matched_profile is None  # noqa: SLF001
    assert det._expected_duration == 0.0  # noqa: SLF001


def test_a_dishwasher_replay_tail_covers_the_end_spike_wait():
    """With Off Delay / Min Off Gap lowered (what Apply all does), the synthetic tail
    ended before the detector's 30-min pump-out wait and force-stopped a cycle that
    ends normally live (register item 455)."""
    from custom_components.ha_washdata.const import DISHWASHER_END_SPIKE_WAIT_SECONDS
    from custom_components.ha_washdata.playground import _tail_span_s

    lowered = SimpleNamespace(off_delay=180, min_off_gap=300, device_type="dishwasher")
    assert _tail_span_s(lowered) > DISHWASHER_END_SPIKE_WAIT_SECONDS
    washer = SimpleNamespace(off_delay=180, min_off_gap=300, device_type="washing_machine")
    assert _tail_span_s(washer) == 300 * 1.5 + 300.0


def test_the_sweep_reports_the_current_value_for_every_parameter():
    """`_sim_config_summary` carries only part of the config, so sweeping e.g.
    Completion Minimum reported `current_value: None` and the panel drew no
    marker for where the device is today."""
    from custom_components.ha_washdata.playground import run_playground_sweep

    store = Mock()
    store.get_past_cycles.return_value = []
    cfg = CycleDetectorConfig(min_power=5.0, off_delay=180, completion_min_seconds=777)
    out = run_playground_sweep(
        store, None, cfg, "completion_min_seconds", [600, 900], "match_accuracy",
        {}, None, 4, prebuilt=([], {}, {}, {}),
    )
    assert out["current_value"] == 777
