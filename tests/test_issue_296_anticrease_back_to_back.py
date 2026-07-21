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
"""Issue #296 follow-up: back-to-back anti-crease cycles must not merge.

A Miele washing machine in "Knitterschutz" (anti-crease) mode holds a constant
~3.2 W baseline plus periodic low-power tumble bursts (<400 W) after the wash
finishes, until the door is opened.  If a second load is started before the door
is opened, the whole sequence (wash -> anti-crease -> wash -> anti-crease) used to
merge into one multi-hour "cycle": the live match drifts to a longer / ambiguous
profile on the growing burst tail, which breaks Smart Termination, so the cycle
never finalises into STATE_ANTI_WRINKLE (which would absorb the tail and split the
next wash).

These tests replay the real user export
(cycle_data/tron4r/washing_machine/ha_washdata_export_01KWFX8C3HVEK7YK6F9N6KAVVS.json)
through a real CycleDetector with the user's exact settings.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
    STATE_ANTI_WRINKLE,
)

pytestmark = pytest.mark.slow

EXPORT_PATH = (
    Path(__file__).parent.parent
    / "cycle_data"
    / "tron4r"
    / "washing_machine"
    / "ha_washdata_export_01KWFX8C3HVEK7YK6F9N6KAVVS.json"
)

# Pflegeleicht 30 target_duration from the export's envelopes (seconds).
WASH1_EXPECTED_S = 88.0 * 60
# Baumwolle 40 target_duration - the longer near-duplicate the match drifts to on
# the growing tail (seconds).
DRIFT_EXPECTED_S = 132.0 * 60


def _load_export() -> dict:
    if not EXPORT_PATH.exists():
        pytest.skip(f"user export fixture not present: {EXPORT_PATH}")
    return json.loads(EXPORT_PATH.read_text())


def _config_from_options(opts: dict) -> CycleDetectorConfig:
    return CycleDetectorConfig(
        min_power=opts["min_power"],
        off_delay=opts["off_delay"],
        device_type=opts["device_type"],
        smoothing_window=opts["smoothing_window"],
        interrupted_min_seconds=opts["interrupted_min_seconds"],
        completion_min_seconds=opts["completion_min_seconds"],
        start_duration_threshold=opts["start_duration_threshold"],
        start_energy_threshold=opts["start_energy_threshold"],
        end_energy_threshold=opts["end_energy_threshold"],
        running_dead_zone=opts["running_dead_zone"],
        end_repeat_count=opts["end_repeat_count"],
        min_off_gap=opts["min_off_gap"],
        start_threshold_w=opts["start_threshold_w"],
        stop_threshold_w=opts["stop_threshold_w"],
        power_off_threshold_w=opts["power_off_threshold_w"],
        power_off_delay=opts["power_off_delay"],
        anti_wrinkle_enabled=opts["anti_wrinkle_enabled"],
        anti_wrinkle_max_power=opts["anti_wrinkle_max_power"],
        anti_wrinkle_max_duration=opts["anti_wrinkle_max_duration"],
        anti_wrinkle_exit_power=opts["anti_wrinkle_exit_power"],
        delay_detect_enabled=opts["delay_start_detect_enabled"],
    )


def _merged_cycle(data: dict) -> dict:
    """The single back-to-back merged cycle in the export (the only >3 h one)."""
    longest = max(data["past_cycles"], key=lambda c: c.get("duration") or 0)
    assert (longest.get("duration") or 0) > 3 * 3600, "expected the merged >3h cycle"
    return longest


def _replay(cfg: CycleDetectorConfig, power_data, matcher):
    """Replay a stored trace + a trailing door-open quiet through a real detector.

    Returns (states_seen, ended_cycles) where ended_cycles is a list of the
    cycle_data dicts emitted by on_cycle_end.
    """
    ended: list[dict] = []
    states: list[str] = []
    det = CycleDetector(
        config=cfg,
        on_state_change=lambda _o, n: states.append(n),
        on_cycle_end=lambda d: ended.append(d),
        profile_matcher=matcher,
    )
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    for off, pw in power_data:
        det.process_reading(float(pw), base + timedelta(seconds=float(off)))
    # Door opened: sustained zero for an hour at 30 s cadence.
    last = power_data[-1][0]
    for i in range(1, 121):
        det.process_reading(0.0, base + timedelta(seconds=last + i * 30))
    return states, ended, det


def _degrading_matcher(readings):
    """Confident 88-min match during the wash; drifts to a longer, ambiguous
    profile once the flat anti-crease tail grows the trace past ~95 min - the
    real field failure that breaks Smart Termination.
    """
    if not readings:
        return None
    elapsed = (readings[-1][0] - readings[0][0]).total_seconds()
    if elapsed <= 95 * 60:
        return ("Pflegeleicht 30", 0.7, WASH1_EXPECTED_S, None, False, False)
    return ("Baumwolle 40", 0.5, DRIFT_EXPECTED_S, None, False, True)


def test_back_to_back_anticrease_does_not_merge() -> None:
    """The merged export replayed with a realistically-degrading matcher must
    split into two washes (not one multi-hour cycle) and pass through
    STATE_ANTI_WRINKLE.
    """
    export = _load_export()
    opts = export["entry_options"]
    cfg = _config_from_options(opts)
    merged = _merged_cycle(export["data"])

    states, ended, _det = _replay(cfg, merged["power_data"], _degrading_matcher)

    durations_min = sorted(round((c["duration"] or 0) / 60, 1) for c in ended)
    assert len(ended) == 2, (
        f"expected the back-to-back run to split into 2 cycles, got "
        f"{len(ended)} (durations {durations_min} min)"
    )
    # Neither recorded cycle is the ~203-min merge.
    assert all((c["duration"] or 0) < 150 * 60 for c in ended), (
        f"a recorded cycle still spans the merge (durations {durations_min} min)"
    )
    assert all(c["status"] == "completed" for c in ended)
    # The split is achieved by finalising the first wash into anti-wrinkle.
    assert STATE_ANTI_WRINKLE in states


def test_confident_match_throughout_also_splits() -> None:
    """Downstream regression guard: if the match stays confident the whole time,
    the cycle already splits correctly (proves the anti-wrinkle machinery)."""
    export = _load_export()
    cfg = _config_from_options(export["entry_options"])
    merged = _merged_cycle(export["data"])

    def confident(readings):
        return ("Pflegeleicht 30", 0.7, WASH1_EXPECTED_S, None, False, False)

    _states, ended, _det = _replay(cfg, merged["power_data"], confident)
    assert len(ended) == 2


def test_manager_path_also_splits() -> None:
    """End-to-end guard for the manager's match path: the manager pushes matches
    via update_match DIRECTLY (bypassing the detector's _try_profile_match), so the
    back-to-back run must still split into two cycles when driven that way.
    """
    export = _load_export()
    cfg = _config_from_options(export["entry_options"])
    merged = _merged_cycle(export["data"])
    power_data = merged["power_data"]

    ended: list[dict] = []
    det = CycleDetector(
        config=cfg,
        on_state_change=lambda _o, _n: None,
        on_cycle_end=lambda d: ended.append(d),
        profile_matcher=None,  # matches come only via update_match, like the manager
    )
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    cycle_start_off = None
    for off, pw in power_data:
        det.process_reading(float(pw), base + timedelta(seconds=float(off)))
        # Track the current cycle's elapsed the way the manager would.
        if det.state in ("starting", "running", "paused", "ending"):
            if cycle_start_off is None:
                cycle_start_off = off
            elapsed = off - cycle_start_off
            if elapsed < WASH1_EXPECTED_S:
                det.update_match(
                    ("Pflegeleicht 30", 0.7, WASH1_EXPECTED_S, None, False, False)
                )
            else:
                # Past expected: manager pushes an ambiguous longer match.
                det.update_match(
                    ("Baumwolle 40", 0.5, DRIFT_EXPECTED_S, None, False, True, True)
                )
        else:
            cycle_start_off = None
    last = power_data[-1][0]
    for i in range(1, 121):
        det.process_reading(0.0, base + timedelta(seconds=last + i * 30))

    durations_min = sorted(round((c["duration"] or 0) / 60, 1) for c in ended)
    assert len(ended) == 2, (
        f"manager-path replay did not split: got {len(ended)} cycle(s) "
        f"(durations {durations_min} min)"
    )
    assert all((c["duration"] or 0) < 150 * 60 for c in ended)


def test_update_match_freeze_preserves_good_match_in_tail() -> None:
    """Unit test of the anti-crease match freeze (#296).

    update_match is the single sink for the detector's own matcher AND the
    manager's async matcher.  Once a matched, energetic cycle is past its expected
    duration and sitting in the low-power tail, a later degraded/ambiguous match
    (which re-matching on the growing tail would produce) must be IGNORED so the
    finalise gate is not blocked - the protection that matters for machines whose
    final spin ends within the confirmation window of the expected duration, where
    Part 1 cannot finalise on the very first past-expected reading.
    """
    from custom_components.ha_washdata.cycle_detector import (
        CycleDetectorConfig as _Cfg,
    )

    cfg = _Cfg(
        min_power=2.0,
        off_delay=60,
        device_type="washing_machine",
        stop_threshold_w=5.0,
        anti_wrinkle_enabled=True,
        anti_wrinkle_max_power=400.0,
    )
    det = CycleDetector(
        config=cfg,
        on_state_change=lambda _o, _n: None,
        on_cycle_end=lambda _d: None,
    )
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    # Drive internal state to "matched, energetic, past-expected, low-power tail".
    det._matched_profile = "Pflegeleicht 30"
    det._expected_duration = 600.0
    det._last_match_confidence = 0.7
    det._match_ambiguous = False
    det._match_prefix_ambiguous = False
    det._cycle_max_power = 2000.0
    det._current_cycle_start = base
    det._power_readings = [
        (base + timedelta(seconds=s), 50.0) for s in range(650, 900, 10)
    ]
    now = det._power_readings[-1][0]

    assert det._in_anticrease_freeze(now) is True

    # A degraded, ambiguous, longer-profile match must be refused.
    det.update_match(("Baumwolle 40", 0.5, 900.0, None, False, True, True))
    assert det.matched_profile == "Pflegeleicht 30"
    assert det._expected_duration == pytest.approx(600.0)

    # A heating burst (above max_power) leaves the tail regime and re-arms matching.
    det._power_readings.append((base + timedelta(seconds=910), 1500.0))
    assert det._in_anticrease_freeze(det._power_readings[-1][0]) is False


def test_dishwasher_is_excluded_from_anticrease_finalize() -> None:
    """The anti-crease finalize/freeze must never engage for a dishwasher (its long
    passive drying tail is handled by the dedicated issue-#43 logic, not anti-crease).

    Verified across the corpus: forcing anti-wrinkle on for all 104 dishwasher cycles
    produced zero anti-crease splits.  This locks the device-type gate.
    """
    from custom_components.ha_washdata.cycle_detector import (
        CycleDetectorConfig as _Cfg,
    )

    cfg = _Cfg(
        min_power=2.0,
        off_delay=1800,
        device_type="dishwasher",
        stop_threshold_w=5.0,
        anti_wrinkle_enabled=True,
        anti_wrinkle_max_power=400.0,
    )
    det = CycleDetector(
        config=cfg,
        on_state_change=lambda _o, _n: None,
        on_cycle_end=lambda _d: None,
    )
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    # Same "past-expected, energetic, low-power tail" state that fires for a washer.
    det._matched_profile = "Eco 50"
    det._expected_duration = 600.0
    det._last_match_confidence = 0.7
    det._match_ambiguous = False
    det._match_prefix_ambiguous = False
    det._cycle_max_power = 2000.0
    det._current_cycle_start = base
    det._power_readings = [
        (base + timedelta(seconds=s), 50.0) for s in range(650, 900, 10)
    ]
    now = det._power_readings[-1][0]

    assert det._anticrease_gate_open(now) is False
    assert det._is_anticrease_tail(now) is False
    assert det._in_anticrease_freeze(now) is False


def test_finalize_never_truncates_high_power_washing() -> None:
    """Regression guard: the anti-crease finalise must never cut a real wash in the
    middle.  Replaying each single wash with a confident match to its own profile,
    the FIRST emitted cycle must contain ALL of the cycle's high-power (heating /
    spin) activity, and any further emitted piece must be a low-power tail (peak at
    or below anti_wrinkle_max_power).

    NB: several recorded cycles already contain an anti-crease tail (the user left
    the door closed), so a split into "wash + low-power tail" is the CORRECT result,
    not a false split.  What must never happen is truncating high-power washing.
    """
    export = _load_export()
    cfg = _config_from_options(export["entry_options"])
    envelopes = export["data"]["envelopes"]
    max_power = float(cfg.anti_wrinkle_max_power)

    checked = 0
    for cycle in export["data"]["past_cycles"]:
        if (cycle.get("duration") or 0) > 3 * 3600:
            continue  # skip the back-to-back merge (covered by the split tests)
        prof = cycle.get("profile_name")
        env = envelopes.get(prof) if prof else None
        if not env or not env.get("target_duration"):
            continue
        expected_s = float(env["target_duration"])
        power_data = cycle["power_data"]
        high_offsets = [off for off, pw in power_data if pw > max_power]
        if not high_offsets:
            continue  # no heating/spin to protect
        last_high = max(high_offsets)

        def confident(_readings, _e=expected_s, _p=prof):
            return (_p, 0.7, _e, None, False, False)

        _states, ended, _det = _replay(cfg, power_data, confident)
        checked += 1
        assert ended, f"cycle (profile {prof}) produced no completed cycle"

        # The first emitted cycle must span all of the high-power activity.
        first = ended[0]
        assert (first["duration"] or 0) >= last_high, (
            f"first cycle (profile {prof}) ends at "
            f"{(first['duration'] or 0)/60:.1f} min but high-power washing runs to "
            f"{last_high/60:.1f} min - real washing was truncated"
        )
        # Any subsequent piece is a separated low-power (anti-crease) tail.
        for extra in ended[1:]:
            assert (extra.get("max_power") or 0) <= max_power * 1.05, (
                f"cycle (profile {prof}) split off a piece with high power "
                f"{extra.get('max_power')}W - not a low-power tail"
            )

    assert checked >= 10, f"expected to check many cycles, only checked {checked}"
