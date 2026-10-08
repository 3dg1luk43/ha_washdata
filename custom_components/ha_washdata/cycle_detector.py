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
"""Cycle detection logic for WashData."""

from __future__ import annotations

import itertools
import logging
import math
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, cast
import numpy as np

from homeassistant.util import dt as dt_util

from . import match_rules
from .log_utils import DeviceLoggerAdapter
from .time_utils import utc_now
from .const import (
    TERMINAL_EVENT_PEAK_FRAC,
    DISHWASHER_QUIET_RELEASE_TERMINAL_MARGIN,
    ANTI_WRINKLE_ELIGIBLE_REASONS,
    TerminationReason,
    STATE_OFF,
    STATE_DELAY_WAIT,
    STATE_STARTING,
    STATE_RUNNING,
    STATE_PAUSED,
    STATE_ENDING,
    STATE_FINISHED,
    STATE_ANTI_WRINKLE,
    STATE_INTERRUPTED,
    STATE_FORCE_STOPPED,
    STATE_UNKNOWN,
    STATE_IDLE,
    DEVICE_TYPE_WASHING_MACHINE,
    DEVICE_TYPE_DRYER,
    DEVICE_TYPE_WASHER_DRYER,
    DEFAULT_MAX_DEFERRAL_SECONDS,
    DEFAULT_DEFER_FINISH_CONFIDENCE,
    DEFAULT_SMART_TERMINATION_DURATION_RATIO,
    DISHWASHER_END_SPIKE_MIN_PROGRESS,
    DISHWASHER_END_SPIKE_QUIET_RELEASE_SECONDS,
    DISHWASHER_END_SPIKE_WAIT_SECONDS,
    DISHWASHER_SMART_TERMINATION_DEBOUNCE_SECONDS,
    SMART_TERM_TAIL_MAX_RATIO,
    SMART_TERM_TAIL_MIN_POINTS,
    SMART_TERM_TAIL_WINDOW_FRAC,
    SMART_TERM_TAIL_WINDOW_MIN_S,
    SMART_TERM_TAIL_WINDOW_S,
    WASHER_SMART_TERMINATION_DEBOUNCE_MAX_SECONDS,
    STARTING_PAUSED_TRUE_OFF_TIMEOUT_SECONDS,
    DISHWASHER_MATCH_FREEZE_QUIET_SECONDS,
    DISHWASHER_MIN_CYCLE_DURATION_S,
    TERMINAL_DROP_OFF_DELAY_SECONDS,
    ENDING_HARD_FINALIZE_RATIO,
    ENDING_HARD_FINALIZE_MIN_QUIET_S,
    GATE_CADENCE_MEDIAN_FACTOR,
    END_GATE_LATE_RATIO,
    resolve_end_gate_late_ratio,
    END_GATE_LATE_SECONDS,
    END_GATE_HAZARD_MARGIN,
    END_GATE_HAZARD_MIN_CYCLES,
    END_GATE_HAZARD_POSITION_SLACK,
    STANDBY_BAND_FINALIZE_DEVICE_TYPES,
    STANDBY_BAND_MIN_RATIO,
    STANDBY_BAND_LOOSE_MIN_RATIO,
    STANDBY_BAND_NEAR_STOP_FACTOR,
    STANDBY_BAND_NEAR_STOP_W,
    DEVICE_TYPE_DISHWASHER,
    TERMINAL_QUIET_CAP_S,
    TRUSTED_LENGTH_FLOOR_FRAC,
    STANDBY_BAND_WINDOW_S,
    STANDBY_BAND_MAX_FRACTION,
    STANDBY_BAND_FLATNESS_FRACTION,
    STANDBY_BAND_FLATNESS_FLOOR_W,
    DEFAULT_ANTI_CREASE_FINALIZE_RATIO,
    ANTI_CREASE_FINALIZE_RATIO_MIN,
    ANTI_CREASE_FINALIZE_RATIO_MAX,
    DEFAULT_CURVE_PREROLL_SECONDS,
    CURVE_PREROLL_MAX_SECONDS,
    PREROLL_CHAIN_BREAK_SECONDS,
    ANTI_CREASE_CONFIRM_WINDOW_S,
    ANTI_CREASE_TERMINAL_HIGH_MIN_FRAC,
    ANTI_CREASE_TERMINAL_MATCH_FRAC,
    ANTI_CREASE_SPIN_WAIT_MAX_RATIO,
)

# The dishwasher end-spike wait window is shared between two code paths
# (Smart Termination's wait branch and _should_defer_finish's no-end-spike
# branch).  They MUST release the cycle at the same instant - sanity-check
# that the constants module loaded a sensible value rather than allowing the
# paths to silently drift if one was changed and the other forgotten.
if DISHWASHER_END_SPIKE_WAIT_SECONDS <= 0:
    # Runtime check (not assert: asserts are stripped under python -O).
    raise ValueError("DISHWASHER_END_SPIKE_WAIT_SECONDS must be positive")

# Opt-in ML end-detection guard (Stage 6). When the manager injects an
# end-confidence provider (only when the user enabled ML models for the device),
# the cycle-end model can defer a *normal* completion if it judges the current
# low-power event to be a pause rather than the true end. This is intentionally
# asymmetric: it can only *delay* a completion, never end a cycle early, and it
# is bounded, so a wrong model can slow a finish but can neither stop one early
# nor hang the cycle. Force-stop / smart-termination / user paths never consult
# it. Overridable emphasis lives here rather than const.py to keep the guard
# self-contained (it is detector-internal policy, not user configuration).
ML_END_GUARD_MIN_CONFIDENCE = 0.5        # P(true end) below this -> treat as a likely pause
ML_END_GUARD_MAX_DEFER_SECONDS = 1800.0  # cap the extra wait the guard may add (30 min)
# The opt-in ML end-guard / terminal-drop providers rebuild the trace and run
# inference on every ENDING-phase evaluation. During a long quiet tail (e.g. a
# dishwasher's up-to-1h soak) that is wasteful, so recompute at most this often
# (data-clock seconds). Safe to cache: the guard only ever *defers* and terminal
# drop only ever *shortens*, so both tolerate a value up to this window stale.
ML_PROVIDER_THROTTLE_SECONDS = 30.0
# ENDING energy gate vs a flat standby (audit DETECT-08, item 95 residual). The
# gate's bar is end_energy_threshold x 3600 / off_delay as a MEAN power (1.0 W at
# 180 s, 0.1 W at 1800 s), so a standby sitting between that bar and the stop
# threshold pinned it forever: an unmatched cycle ran to the 8 h cap. A window is
# a flat standby when every reading of the last max(off_delay,
# STANDBY_BAND_WINDOW_S) seconds is above 0 W and the window spans no more than
# max(FLOOR_W, FRAC x its highest reading). The non-zero floor is what keeps the
# gate's one measured catch: a dishwasher's passive drying flickering 0-0.5 W
# for 25 min before its pump-out (Hatton ECO `44d35b4ca01e`).
#
# Two tiers, like the standby band's. Just under stop (every reading >=
# NEAR_STOP_FRAC x stop_threshold_w) is the item-95 shape and ends after the
# un-shortened max(off_delay, min_off_gap). Any other flat window waits
# LOOSE_QUIET_S: flat non-zero phases far under stop DO resume in real cycles.
# Over every stored trace in the corpus (all formats) the resumed flat non-zero
# sub-stop pauses longer than that device's min_off_gap are two, and both sit at
# about half of stop: an 18.9 min washer soak at 2.7-3.0 W on a 6 W stop (it split
# with one tier at min_off_gap, end_gate_eval --shipped-watchdog) and an 80.8 min
# dishwasher drying phase at 0.7-0.9 W on a 1.5 W stop (it split replayed
# unmatched). Neither reaches 0.75 x stop; 2 h is 1.5x the longer one.
ENDING_FLAT_STANDBY_MIN_READINGS = 3
ENDING_FLAT_STANDBY_SPREAD_FRAC = 0.25
ENDING_FLAT_STANDBY_SPREAD_FLOOR_W = 0.2
ENDING_FLAT_STANDBY_NEAR_STOP_FRAC = 0.75
ENDING_FLAT_STANDBY_LOOSE_QUIET_S = 7200.0
if not 0 < DISHWASHER_END_SPIKE_MIN_PROGRESS < 1:
    raise ValueError("DISHWASHER_END_SPIKE_MIN_PROGRESS must be a fraction in (0, 1)")
from .signal_processing import (
    energy_gap_threshold_s,
    integrate_wh,
    median_fast,
    percentile_linear,
    terminal_event_end,
    terminal_quiet_seen,
)

_LOGGER = logging.getLogger(__name__)

# After a user/external stop the manual-stop lockout swallows the machine's
# spin-down/drain so it is not logged as a fresh cycle. The lockout normally
# clears as soon as power drops to idle. As a safety net, if power instead stays
# at or above the start threshold for longer than any plausible spin-down, the
# device is running a genuinely new (back-to-back) load: release the lockout so
# the new cycle is detected instead of being pinned until the progress-reset
# window expires (issue #267).
STOP_LOCKOUT_RELEASE_SECONDS = 180.0

# Register item 501: how much of start_energy_threshold a standby RE-probe (one
# that began with no reading below stop_threshold_w since the last false start)
# must fill before entities show it as `starting`. Display only: detection never
# reads it. Measured by devtools/start_gate_eval.py (flickers per idle day).
STANDBY_REPROBE_SHOW_ENERGY_FRACTION = 0.5

# Register item 515: a false start out of Finished / Interrupted / Force-Stopped
# returns to that state with its original entry time (as item 504 returns one
# out of DELAY_WAIT), instead of falling to OFF. With it the manager clears the
# cycle end, Clean and the unload nag only when a probe commits (RUNNING), not
# when it begins: a probe that aborted used to lose all three, and a straddling
# standby (#35) probes on almost every reading. The A/B switch for the revert
# checks and `devtools/start_gate_eval.py` (`terminal_lost`).
TERMINAL_PROBE_RETURNS = True

# Discussion #452: two more display-only states, read by `exposed_state` and never
# by detection.
#
# STALLED. A washer that halts mid-cycle (an unbalanced load, a door warning)
# sits at its standby draw, just ABOVE stop_threshold_w, so the cycle never even
# pauses. A stall is a flat run of readings in the standby band's near-stop shape
# (`standby_near_stop_ceiling`, at most STANDBY_BAND_MAX_FRACTION of the cycle's
# peak, spread within the band's flatness limit) that has lasted longer than any
# low stretch the matched programme recorded from that position on
# (END_GATE_HAZARD_MARGIN x the longest in its near-stop pause catalogue, element
# 15; at least STALL_MIN_S) while the programme still owes work: its terminal
# high-power block not yet seen (#399's evidence), or, for a programme without
# one, the run began before STALL_OWES_MAX_POSITION of its expected duration.
# The evidence is the match the run began under (a plateau soon reads to the
# matcher as a finished SHORTER programme) AND the current match (the last tick
# before a real end can be a prefix match of a LONGER one); with no current match
# the first stands. Unmatched (or matched with fewer than
# END_GATE_HAZARD_MIN_CYCLES traced cycles): STALL_UNMATCHED_MIN_S. Shown as
# `paused` with sub-state STALL_SUB_STATE, cleared by the first reading out of
# the band. While shown it reaches detection twice: the anti-crease finalize
# waits (item 511, below), and in the standby-band finalize,
# during a run whose two matches both still owe their terminal high-power block,
# the near-stop tier is held, so the plateau waits for the loose tier
# (STANDBY_BAND_LOOSE_MIN_RATIO x expected) instead of closing a halted wash as
# finished at 1.0x (STALL_HOLDS_STANDBY_BAND, the A/B switch for
# devtools/end_gate_eval.py).
STALL_DEVICE_TYPES = STANDBY_BAND_FINALIZE_DEVICE_TYPES
STALL_MIN_S = 600.0
STALL_UNMATCHED_MIN_S = 1800.0
STALL_OWES_MAX_POSITION = 0.75
STALL_SUB_STATE = "Stalled"
STALL_HOLDS_STANDBY_BAND = True
# Register item 511: a halted programme does not advance. Once a stall is over
# (the wash resumed, or the plug dropped below stop), every duration-based end
# gate (Smart Termination, the hard finalize, the item-306 shortening, the hazard
# gate, the minimum-duration deferral, both standby-band tiers, the anti-crease
# gate and its #399 spin wait) reads the elapsed time LESS that run, from its
# first reading (`_gate_elapsed_s`); the live matcher reads the trace with it cut
# out (`_match_readings`) and the #399 scan offset moves past it. Without that the
# halt carried the clock past each gate's ratio and the resumed wash ended at its
# next quiet. A stall still shown counts as before, so the standby band closes
# a display left on after a real end exactly as the #452 hold above allows (the
# loose tier bounds it), and the anti-crease finalize waits while one is shown. The
# stored duration, the displayed elapsed and the 8 h cap keep the wall clock.
# Excluding the stall in progress as well (the near-stop tier then waited out
# every shown stall) kept 26 more synthetic 45 min halts whole, but held 10 real
# ends with the display left on for 45 min until it went off (up to 57 min late)
# and one left on for twice the programme up to 6 h. The A/B switch for the
# revert checks.
STALL_EXCLUDED_FROM_GATES = True
# Register item 514: progress reads programme time too. Every finished stall
# leaves the elapsed time and the trace the progress estimate reads
# (`progress_elapsed_s` / `progress_trace`, the one input of the manager and the
# Playground replay), as the user pause leaves the elapsed time
# (`manager.net_elapsed_seconds`), and the finished stalls ride in the cycle data
# (`stall_spans`) so the cycle-end label match reads the trace without them
# (`match_rules.final_match_input`). The stall shown now as well
# (STALL_CURRENT_EXCLUDED_FROM_PROGRESS): from the moment it shows, the remaining
# time stops counting down. Measured (`eta_eval --halt-at 0.5 --halt-min 45`, 246
# washer targets): through the halt the ETA fell a median 40.9 min before, 0.5 now;
# its median error 1 / 10 / 30 min after the resume 41.8 / 36.4 / 21.6 -> 13.2 /
# 16.7 / 13.7 min (finished stalls only: 25.0 / 16.7 / 13.7); real corpus 0 of 450
# rows changed. A/B switches for the revert checks.
STALL_EXCLUDED_FROM_PROGRESS = True
STALL_CURRENT_EXCLUDED_FROM_PROGRESS = True
# ...and a finished user pause is programme time stood still too (item 514): the
# gate clock, the #399 scan offset and the live matcher's trace leave it out like
# a finished stall (the union of the two, `_gate_spans`), and it rides in the
# cycle data with them (`halt_spans`). The user pause blocked every finisher
# while it lasted (the verified pause) but the gates read the raw clock after
# the resume, so a pause that cuts the plug's power ended the resumed wash at
# its next quiet. Progress already leaves it out (`manager.net_elapsed_seconds`).
# Measured (`end_gate_eval --halt-at F --halt-min 45 --user-pause`, 258 cycles):
# power cut at 50% / 30% / 90%, splits 48 / 50 / 40 -> 12 / 22 / 33, early ends
# 6 / 5 / 18 -> 5 / 7 / 8 (each new one the unpaused cycle's own end, or a former
# split); paused on the display level, 9 -> 6 and 14 -> 9 (50% / 90%).
USER_PAUSE_EXCLUDED_FROM_GATES = True
# Register item 514: a halt is abrupt. A programme winds down to its end (the last
# spin's ramp, then quiet), so the reading before a display left on is low; a halt
# stops the machine where it is. A flat run whose previous reading was at least
# STALL_ABRUPT_PEAK_FRACTION of the cycle's peak (and twice the band's ceiling)
# began straight out of activity (`_stall_run_abrupt`), and when the MATCHED
# programme it began under still owes work, the current match need not agree: the
# plateau soon reads to the matcher as a finished shorter programme, which vetoed
# the display and the near-stop hold above and closed the halted wash 10 min in.
# Unmatched runs are left to the current match (a short programme's real end looks
# the same). Measured (`end_gate_eval --halt-at/--halt-min`, 258 washer,
# washer-dryer and dryer cycles): at 0.10, 45 min halts at 50% shown 196 -> 198,
# closed inside the halt 75 -> 67, splits 79 -> 71 (30%: 76 -> 61; 90%: 130 -> 118),
# early ends unchanged; a display left on after a real end (100%, 45 min and 2x):
# no new stall flag, 2 closes later, both in runs the old gates had split. At 0.05:
# 50% splits 79 -> 54 and 9 more shown, but one more display-on flag and one more
# real end held (+36 min). 0 disables (the A/B switch).
STALL_ABRUPT_PEAK_FRACTION = 0.10
#
# IDLE. Outside a cycle a two-level appliance (display on vs switched off) shows
# `idle` at its standby level and `off` below it. The level is the user's
# power-off threshold (#284) when one is set: idle at or above it, off below it,
# debounced by power_off_delay like the #284 reset itself. Otherwise the learned
# standby level (`learned_standby_level_w`, set by the manager): idle from
# IDLE_OFF_FRACTION of it, off below IDLE_EXIT_FRACTION of that (hysteresis), each
# held IDLE_DEBOUNCE_S, and only once the appliance has been seen below that off
# level outside a cycle since start-up. A level under IDLE_MIN_STANDBY_W, or at
# or above the start threshold, is not clearly separated from off or from a
# start: no idle at all, exactly as before.
IDLE_MIN_STANDBY_W = 2.0
IDLE_OFF_FRACTION = 0.5
IDLE_EXIT_FRACTION = 0.75
IDLE_DEBOUNCE_S = 60.0


def effective_anticrease_finalize_ratio(value: Any) -> float:
    """The ratio the anti-crease gate will actually use for a stored value.

    Only ``ws_set_options`` range-checks this option: ``import_config`` strips
    nulls only, a selective import writes numbers through, and the Playground
    sanitizer just casts to float. So the value is held to its documented range
    here, at every point of use, and anything unusable falls back to the default
    rather than disarming the gate (a stored ``0.0`` would satisfy the
    past-expected test for every duration).

    Shared with the Playground's config summary so the figure the panel shows and
    the figure the gate applies cannot drift apart.
    """
    try:
        ratio = float(value)
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_ANTI_CREASE_FINALIZE_RATIO
    if not math.isfinite(ratio):
        return DEFAULT_ANTI_CREASE_FINALIZE_RATIO
    return min(
        ANTI_CREASE_FINALIZE_RATIO_MAX, max(ANTI_CREASE_FINALIZE_RATIO_MIN, ratio)
    )


def effective_curve_preroll_seconds(value: Any) -> float:
    """The pre-roll window actually applied for a stored value (0 = off).

    Same reasoning as :func:`effective_anticrease_finalize_ratio`: the detector
    caps the window at ``CURVE_PREROLL_MAX_SECONDS`` wherever it reads it, so the
    summary has to report the capped figure or it describes a sim that did not
    run.
    """
    try:
        window = float(value or 0.0)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(window) or window <= 0:
        return 0.0
    return min(window, CURVE_PREROLL_MAX_SECONDS)


@dataclass(frozen=True)
class MatchContext:
    """One live match as the detector consumes it (audit DETECT-15).

    Producers build this by name; ``update_match`` still reads the positional
    sequence it always did (``as_sequence``), so a field can no longer be put in
    the wrong slot or left out of one producer. Every field after the first five
    defaults to "no evidence", exactly what a shorter legacy tuple meant.
    """

    profile_name: str | None
    confidence: float
    expected_duration: float
    phase_name: str | None = None
    is_confident_mismatch: bool = False
    is_ambiguous: bool = False
    # Element 7 is retired: it carried the #364 prefix-fit flag, removed in 0.5.8
    # (see const.py). The slot stays (always False) so 8-14 keep their positions.
    # Element 8: the #288 full-shape term (anti-crease finalize only).
    is_prefix_ambiguous_full_shape: bool = False
    tail_power: Any = None            # 9 (#364 power guard)
    terminal_high: Any = None         # 10 (#399 anti-crease spin look-ahead)
    terminal_quiet_s: Any = None      # 11 (item 297 keep-tail cap)
    longest_candidate_s: float = 0.0  # 12 (item 330 fallback bar)
    trusted_min_s: Any = None         # 13 (item 384 trusted-length floor)
    pause_catalogue: Any = None       # 14 (DETECT-16 hazard gate)
    # 15 (#452 stall display): the same catalogue below `standby_near_stop_ceiling`,
    # or a zero-argument callable returning it: read only when a flat run lasts
    # long enough to be judged, so the profile's traces are not walked for it on
    # every cycle (`test_perf_budgets`).
    stall_catalogue: Any = None

    def __getitem__(self, index: Any) -> Any:
        return self.as_sequence()[index]

    def __len__(self) -> int:
        return 15

    def as_sequence(self) -> tuple[Any, ...]:
        return (
            self.profile_name, self.confidence, self.expected_duration, self.phase_name,
            self.is_confident_mismatch, self.is_ambiguous, False,
            self.is_prefix_ambiguous_full_shape, self.tail_power, self.terminal_high,
            self.terminal_quiet_s, self.longest_candidate_s, self.trusted_min_s,
            self.pause_catalogue, self.stall_catalogue,
        )


@dataclass
class CycleDetectorConfig:
    """Configuration for cycle detection."""

    min_power: float
    off_delay: int
    device_type: str = DEVICE_TYPE_WASHING_MACHINE
    interrupted_min_seconds: int = 150
    completion_min_seconds: int = 600
    start_duration_threshold: float = 5.0
    start_energy_threshold: float = 0.005
    end_energy_threshold: float = 0.05  # 0.05 Wh (50 mWh) threshold for "still active"
    min_off_gap: int = 60
    start_threshold_w: float = 2.0
    stop_threshold_w: float = 2.0
    # The FINISH-DEFERRAL ratio (`_should_defer_finish`), despite the name: not the
    # matcher's Stage-1 `profile_match_min_duration_ratio`. Built from
    # const.DEFAULT_DEFER_FINISH_RATIO (audit DETECT-02).
    min_duration_ratio: float = 0.8
    # Minimum live-match confidence for a match to be trusted by Smart Termination
    # and the anti-crease gate. Fed from the `profile_match_threshold` option, which
    # up to 0.5.5 was stored and never read - so raising it (the workaround the #288
    # reporter documented) silently did nothing. Default matches the value that was
    # hard-coded at those two sites, so behaviour is unchanged unless the user has
    # deliberately tuned the option.
    match_confidence_threshold: float = 0.4
    # Power-based Off detection (issue #284). Carried on the config so the manager
    # (the single owner of the terminal -> Off transition) can read them live; the
    # detector itself does not act on them. 0 = disabled.
    power_off_threshold_w: float = 0.0
    power_off_delay: float = 30.0
    match_interval: int = 300  # Default profile match interval
    anti_wrinkle_enabled: bool = False
    anti_wrinkle_max_power: float = 400.0
    anti_wrinkle_max_duration: float = 60.0
    anti_wrinkle_exit_power: float = 0.8
    anti_wrinkle_idle_timeout: float = 120.0
    # Dishwasher only: sustained-quiet seconds (after reaching expected duration)
    # that release the end-of-cycle pump-out/drain wait early (#379). Defaults to
    # the shipped constant; per-device configurable so a machine with a long silent
    # passive-drying phase before its final drain can absorb profile drift.
    dishwasher_end_spike_quiet_release: float = DISHWASHER_END_SPIKE_QUIET_RELEASE_SECONDS
    # Fraction of the matched profile's expected (mean) duration Smart Termination
    # requires before it may fire (#393). Device-type-resolved in the manager's
    # config builder (0.99 dishwasher / 0.98 other), so this field always carries a
    # real float - never None - which is what playground.effective_settings() relies
    # on. The dishwasher pump-out relief is combined via min(), so a configured value
    # can only loosen the gate.
    smart_termination_duration_ratio: float = DEFAULT_SMART_TERMINATION_DURATION_RATIO
    # Fraction of the matched profile's expected duration the anti-crease finalise
    # requires before it may fire (#429). A DIFFERENT gate from the ratio above:
    # that one gates Smart Termination, this one the finalise into
    # STATE_ANTI_WRINKLE. Scalar default (no device-type resolution) and always a
    # real float, so playground.effective_settings() never sees None.
    anti_crease_finalize_ratio: float = DEFAULT_ANTI_CREASE_FINALIZE_RATIO
    # How far back readings from aborted start probes may be carried into a
    # committed cycle's curve (#430). 0 disables the whole path, which is the
    # default and keeps the stored-duration convention unchanged.
    curve_preroll_seconds: float = DEFAULT_CURVE_PREROLL_SECONDS
    delay_detect_enabled: bool = False
    # Sustained seconds power must stay in the standby band (between
    # stop_threshold_w and start_threshold_w) before DELAY_WAIT engages.
    # Tuned to filter out brief menu-navigation peaks at the start of a
    # delayed program.
    delay_confirm_seconds: float = 60.0
    delay_timeout_seconds: float = 28800.0


    # Add other fields as needed


def trim_zero_readings(
    readings: list[tuple[datetime, float]],
    threshold: float = 0.5,
    trim_start: bool = True,
    trim_end: bool = True,
) -> list[tuple[datetime, float]]:
    """Trim continuous zero/near-zero readings from start and end of cycle.

    Args:
        readings: List of (timestamp, power) tuples
        threshold: Power values below this are considered "zero"
        trim_start: Whether to trim zeros from the beginning
        trim_end: Whether to trim zeros from the end

    Returns:
        Trimmed list
    """
    if not readings:
        return readings

    start_idx = 0
    if trim_start:
        for i, (_, power) in enumerate(readings):
            if power > threshold:
                start_idx = i
                break
        else:
            # All readings are zero - return single point if list not empty
            return readings[:1] if readings else []

    end_idx = len(readings) - 1
    if trim_end:
        # Find last non-zero reading
        found_end = False
        for i in range(len(readings) - 1, -1, -1):
            if readings[i][1] > threshold:
                end_idx = i
                found_end = True
                break

        if not found_end and trim_start:
            # If all zeros and trim_start was checked, it would return early.
            # But if safety fallback needed:
            end_idx = start_idx
        elif not found_end and not trim_start:
             # Trimming end but not start, and all zeros?
             # Keep first point
            end_idx = 0

    # Return trimmed slice (inclusive of end)
    return readings[start_idx : end_idx + 1]


def standby_near_stop_ceiling(stop_threshold_w: float) -> float:
    """Top of the standby band's near-stop shape (#445 / register item 383).

    A plateau at or above the stop threshold and at most this high is an
    appliance sitting at its standby draw, not one doing work. Shared by the
    standby-band finalize, the stall display (#452) and its pause catalogue.
    """
    stop = float(stop_threshold_w)
    return max(STANDBY_BAND_NEAR_STOP_FACTOR * stop, stop + STANDBY_BAND_NEAR_STOP_W)


#: How many of the most recent stored cycles `learned_standby_level_w` reads.
STANDBY_LEVEL_RECENT_CYCLES = 30


def learned_standby_level_w(
    cycles: Any, stop_threshold_w: float, start_threshold_w: float
) -> float | None:
    """Discussion #452: the standby/display level of a two-level appliance, or None.

    Two measurements the suggestion engine already makes, in order: the level
    every recent cycle was still drawing when it ended (``detect_standby_above_stop``,
    #445), else the resting draw of the clean cycles (``resting_level_w``, item
    455), timed below the near-stop ceiling instead of below stop so that a draw
    just above stop (#452: 4-5 W on a 2.8 W stop) is seen too. None unless the
    level is at least IDLE_MIN_STANDBY_W and below the start threshold, i.e.
    clearly separated from both "off" and a start. Reads the last
    STANDBY_LEVEL_RECENT_CYCLES cycles. Pure, executor-safe, never raises.
    """
    try:
        # Lazy: suggestion_engine imports Home Assistant helpers this module does not need.
        from .suggestion_engine import (  # noqa: PLC0415
            _cycle_readings,
            detect_standby_above_stop,
            resting_level_w,
            select_clean_cycles,
        )

        stop = float(stop_threshold_w)
        start = float(start_threshold_w)
        if not (math.isfinite(stop) and math.isfinite(start)) or stop <= 0:
            return None
        recent = [c for c in list(cycles or [])[-STANDBY_LEVEL_RECENT_CYCLES:] if isinstance(c, dict)]
        adv = detect_standby_above_stop(recent, stop)
        if adv is not None:
            level: float | None = float(adv["idle_w"])
        else:
            clean, _excluded = select_clean_cycles(recent, stop_threshold_w=stop)
            points = [p for p in (_cycle_readings(c) for c in clean) if len(p) >= 5]
            level = resting_level_w(points, standby_near_stop_ceiling(stop))
        if level is None or not math.isfinite(level):
            return None
        if level < IDLE_MIN_STANDBY_W or level >= start:
            return None
        return float(level)
    except Exception:  # noqa: BLE001 - a display statistic must never break setup
        return None


def terminal_high_for_guards(
    store: Any,
    config: "CycleDetectorConfig",
    cycle_max_power: Any,
    profile_name: str | None,
) -> tuple[float, ...] | None:
    """Element 10 of the match tuple: the matched profile's last high-power block.

    Two consumers with two different bars, and the bar has to travel with the
    block (register item 351):

    * **anti-crease** (#399) measures against ``anti_wrinkle_max_power``, the
      dryer's "a tumble is below this" level. Only meaningful while anti-wrinkle
      is on, and it returns a TRIPLE.
    * **the standby-band finalise** (#296 / #445) shares the same predicate and
      used to get nothing at all, because element 10 was supplied only when
      anti-wrinkle was enabled and ``DEFAULT_ANTI_WRINKLE_ENABLED`` is False. So
      on a washing machine ``_anticrease_spin_pending`` returned False at once
      and the machine could finalise on the quiet plateau before its final spin,
      recording that spin as a second cycle. The bar here is a share of the
      cycle's own peak, and it is returned as a QUAD so
      ``_high_power_seconds_since`` counts live seconds against the same number.

    **This lives here, module level, because it had two copies.** The manager
    builds the live match tuple and ``playground`` builds the sim's, and
    ``end_gate_eval.py`` drives the detector through the Playground - so an arm
    present in only one of them is invisible to every measurement made with that
    harness, which is exactly how item 352 first measured as a no-op. The copies
    had already drifted in their error handling before they were merged.

    Total by construction: every failure path returns None, which leaves the
    guard exactly as inert as it was. That is the fail-open direction every input
    here takes, and it is also what lets ``playground`` call it directly while
    keeping its own never-raise contract.
    """
    if not profile_name or store is None:
        return None
    try:
        if config.anti_wrinkle_enabled:
            return store.profile_terminal_high_block(
                profile_name, config.anti_wrinkle_max_power
            )
        if config.device_type not in STANDBY_BAND_FINALIZE_DEVICE_TYPES:
            return None
        ceiling = float(cycle_max_power or 0.0) * STANDBY_BAND_MAX_FRACTION
        if ceiling <= 0:
            return None
        block = store.profile_terminal_high_block(profile_name, ceiling)
        if block is None:
            return None
        return (float(block[0]), float(block[1]), float(block[2]), ceiling)
    except Exception:  # noqa: BLE001 - a guard input must never break matching
        return None


def _merge_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """``(at, length)`` spans sorted and merged where they overlap (item 514: a
    stall shown during a user pause is banked by both)."""
    merged: list[tuple[float, float]] = []
    for at, length in sorted(spans):
        if merged and at <= merged[-1][0] + merged[-1][1]:
            a0, n0 = merged[-1]
            merged[-1] = (a0, max(a0 + n0, at + length) - a0)
        else:
            merged.append((at, length))
    return merged


class CycleDetector:
    """Detects washing machine cycles based on power usage.

    Implements a robust state machine:
    OFF -> STARTING -> RUNNING <-> PAUSED -> ENDING -> OFF
    """

    def __init__(
        self,
        config: CycleDetectorConfig,
        on_state_change: Callable[[str, str], None],
        on_cycle_end: Callable[[dict[str, Any]], None],
        profile_matcher: (
            Callable[
                [list[tuple[datetime, float]]],
                tuple[str | None, float, float, str | None] | None,
            ]
            | None
        ) = None,
        device_name: str = "",
        end_confidence_provider: (
            Callable[[list[tuple[float, float]], float], float | None] | None
        ) = None,
        terminal_drop_provider: (
            Callable[[list[tuple[float, float]], float], bool | None] | None
        ) = None,
    ) -> None:
        """Initialize the cycle detector."""
        self._logger = DeviceLoggerAdapter(_LOGGER, device_name)
        self._config = config
        self._on_state_change = on_state_change
        self._on_cycle_end = on_cycle_end
        self._profile_matcher = profile_matcher
        # Opt-in ML end-guard: (points, expected_duration) -> P(true end) or None.
        # Injected by the manager; None disables the guard (existing behavior).
        self._end_confidence_provider = end_confidence_provider
        # Opt-in terminal-drop detector: (points, expected_duration) -> bool.
        # True means the current low-power event is an anomalously-early hard
        # cliff-to-0 (never seen this early on this device), so the cycle may be
        # finalized without waiting out the full soak-bridging min_off_gap.
        # Injected by the manager; None disables it (existing behavior). Opposite
        # asymmetry to the end-guard: it can only ever *shorten* the end wait.
        self._terminal_drop_provider = terminal_drop_provider
        # Throttle caches for the two providers, scoped to the cycle + expectation:
        # (last_reading_ts, expected_duration, cycle_start, result). Reused only
        # within the recompute window when expected_duration and cycle_start match.
        self._ml_end_cache: tuple[datetime, float, datetime, float | None] | None = None
        self._terminal_drop_cache: tuple[datetime, float, datetime, bool] | None = None
        # Cycle duration (s) at which the ML guard first deferred the current
        # ending episode; bounds how long the guard may keep deferring.
        self._ml_defer_start_duration: float | None = None

        # State
        self._state = STATE_OFF
        self._sub_state: str | None = None
        self._ignore_power_until_idle: bool = False
        # Sustained high-power time accrued while the stop lockout is armed; used
        # to release the lockout for a genuinely new back-to-back load (#267).
        self._lockout_high_seconds: float = 0.0

        # Data
        self._power_readings: list[tuple[datetime, float]] = []  # (time, raw_power)
        # Rolling pre-cycle readings, so a start that took several probes to
        # commit can recover the readings the aborted probes took with them
        # (#430). Only appended to while no cycle is open, trimmed to
        # curve_preroll_seconds, and cleared at every cycle end - a previous
        # cycle's tail must never be carried into the next cycle's curve.
        self._preroll_buffer: list[tuple[datetime, float]] = []
        self._current_cycle_start: datetime | None = None
        self._last_active_time: datetime | None = None
        # Last reading the POWER SENSOR actually sent, as opposed to one the
        # manager injected to advance the quiet timers. Deliberately not reset per
        # cycle: it describes the sensor, not the run.
        self._last_real_reading_time: datetime | None = None
        # When the power sensor's state became unavailable / unknown / non-numeric
        # (register item 266): set by the manager, cleared by the next real
        # reading. Sensor state like the field above, so not reset per cycle.
        self._sensor_outage_since: datetime | None = None
        # Traceback of the last restore_state_snapshot that failed, else None.
        self.restore_error: str | None = None
        self._cycle_max_power: float = 0.0

        # Accumulators (dt-aware)
        self._energy_since_idle_wh: float = 0.0
        self._time_above_threshold: float = 0.0
        self._time_below_threshold: float = 0.0
        # As above, but restarts whenever an outage-sized gap breaks the observed
        # quiet tail, so it counts only quiet WashData actually saw. Used by the
        # dishwasher quiet-release gates so a single low sample after a telemetry
        # dropout can't satisfy them without the quiet having been observed.
        self._time_below_threshold_gapfree: float = 0.0
        # The part of the current below-threshold run that fell inside a sensor
        # outage and so was NOT added to `_time_below_threshold` (item 266). Only
        # the hazard gate reads it, to place the quiet run's start honestly.
        self._time_below_unobserved: float = 0.0
        self._last_process_time: datetime | None = None

        # New State Machine trackers
        self._state_enter_time: datetime | None = None
        self._matched_profile: str | None = None
        self._verified_pause: bool = False
        self._user_paused: bool = False
        # Item 514: when the current user pause began, and the pauses this cycle
        # already resumed from as (seconds from the cycle start, length).
        self._user_pause_since: datetime | None = None
        self._user_pause_spans: list[tuple[float, float]] = []

        self._last_power: float | None = None
        self._time_in_state: float = 0.0

        # Smoothing buffer

        # Adaptive Sampling Tracker
        self._recent_dts: list[float] = []  # Track last 20 dt values
        self._p95_dt: float = 1.0  # Default assumption
        # The cadence as it stood BEFORE the reading currently being processed was
        # folded in. Every gap-vs-outage classification reads this, never _p95_dt:
        # an outage-sized interval that has already widened p95 would raise the very
        # ceiling meant to catch it (a 120 s gap after a 10 s cadence lifts p95 to
        # ~15.5 s -> ceiling 155 s -> the gap passes as observed time).
        self._prior_p95_dt: float = 1.0

        # Profile Matching Tracker
        self._last_match_time: datetime | None = None
        # Whether the manager has committed a program this cycle (set_match_committed).
        self._match_committed: bool = False
        self._expected_duration: float = 0.0
        self._last_match_confidence: float = 0.0
        # Element 12: the longest expected duration among the candidates the
        # matcher still considers plausible. An ambiguous match may mean one of
        # them is materially longer than the winner, so "past the expected end"
        # might be "mid-soak in that longer programme" - but only up to THIS
        # duration. Past it there is no longer programme left to be mid-soak in,
        # and the guard's own rationale is spent.
        self._longest_candidate_duration: float = 0.0
        self._end_spike_seen: bool = False
        self._end_spike_duration: float = 0.0  # cycle duration (s) when _end_spike_seen was last set
        self._match_ambiguous: bool = False  # last live match was ambiguous (gates predictive end)
        # The #288 prefix-landscape term (element 8): a much longer candidate with
        # a good full-envelope shape exists. Read only by the anti-crease finalize
        # (see _anticrease_gate_open). The ENDING gates' own prefix flag (#364
        # prefix fit, element 7) was removed in 0.5.8.
        self._match_prefix_ambiguous_full_shape: bool = False
        # Mean power the matched profile draws over the last few % of its own run
        # (profile_store.profile_tail_power). None = no opinion, guard stays inert.
        self._matched_tail_power: float | None = None
        # (start_frac, seconds, start_offset_s) of the matched profile's own terminal
        # high-power block, from profile_store.profile_terminal_high_block. None = the
        # profile has no such block (or we have no opinion) and the guard stays inert
        # (#399). A two-element payload (pre-item-196 snapshot, Playground, older
        # callers) is still accepted and falls back to start_frac x expected.
        self._matched_terminal_high: (
            tuple[float, float] | tuple[float, float, float] | None
        ) = None
        # Element 11 (register item 297): how long the matched profile is MEASURED to stay quiet
        # after its last real activity. Bounds how much of Smart Termination's
        # confirmation delay may be banked into the stored duration. None means the
        # profile has not been measured, and for a DISHWASHER the previous
        # expected-end cap applies. Only a dishwasher gets that far: every other
        # device type returns `_last_active_time` from `_keep_tail_cap` before
        # this element is read at all, because nothing legitimate follows their
        # last activity - so for them the cap can sit EARLIER than expected_end.
        self._matched_terminal_quiet_s: float | None = None
        # Element 13 (register item 384): the shortest length the user has vouched
        # for in the matched profile (a corrected `manual_duration`, a recorder
        # capture, a golden cycle). A dishwasher's kept tail never ends before
        # TRUSTED_LENGTH_FLOOR_FRAC of it - the same floor the banked-tail repair
        # applies - because element 11 can be wrong in the direction that deletes
        # a drying phase. None: nothing vouched, no floor.
        self._matched_trusted_min_s: float | None = None
        # Element 14 (audit DETECT-16): the matched profile's pause catalogue,
        # (traced cycles, ((start_fraction, seconds), ...)), for the hazard end
        # gate. None: no catalogue, the fallback waits as before.
        self._matched_pause_catalogue: tuple[int, tuple[tuple[float, float], ...]] | None = None
        # Register item 498: the longest pause in the catalogue of every match
        # revoked this cycle, which bounds a verified pause the revoke orphaned
        # (`_release_orphaned_pause`). 0.0: no evidence, the floor applies.
        self._orphaned_pause_evidence_s: float = 0.0
        self._terminal_quiet_memo: tuple[Any, bool] | None = None
        # One-shot per cycle, so the held-finalise reason is visible in the log
        # without repeating it on every reading.
        self._anticrease_spin_wait_logged: bool = False
        # Register item 393a: set while STATE_ANTI_WRINKLE was entered by the #296
        # anti-crease finalise - the level at or under which a reading is that
        # tail's own baseline rather than a burst. None for every other entry.
        self._anticrease_tail_floor_w: float | None = None
        self._last_smart_term_block_reason: str | None = None  # #346 diagnostic throttle

        # Anti-wrinkle tracking (dryers only)
        self._anti_wrinkle_candidate_start: datetime | None = None
        self._anti_wrinkle_candidate_peak: float = 0.0
        self._anti_wrinkle_candidate_start_power: float = 0.0
        self._anti_wrinkle_idle_time: float = 0.0  # Track time spent below exit_power while in ANTI_WRINKLE

        # Delayed-start band tracking.
        # _delay_band_start anchors the first reading in the standby band
        # [stop_threshold_w, start_threshold_w) while still in STATE_OFF.
        # _delay_band_seconds mirrors the anchored elapsed time for
        # diagnostics and tests.
        self._delay_band_start: datetime | None = None
        self._delay_band_seconds: float = 0.0
        # _delay_band_peak is purely diagnostic - surfaced in the log line
        # when the transition fires so users can see what their machine's
        # actual standby plateau looked like.
        self._delay_band_peak: float = 0.0
        # _delay_wait_true_off_seconds tracks sustained "true off" (power
        # below stop_threshold_w) inside DELAY_WAIT, so we can drop back to
        # OFF only when the machine has clearly been switched off rather
        # than briefly dipped.
        self._delay_wait_true_off_seconds: float = 0.0
        # _starting_paused_off_since anchors the first below-stop reading while
        # a user-paused STARTING state is held (issue #306). Elapsed time is
        # measured from this anchor, not accumulated per-dt, so a single large
        # (but sub-outage) interval cannot prematurely credit minutes of quiet time.
        self._starting_paused_off_since: datetime | None = None
        # _delay_wait_high_start anchors the first high-power reading
        # observed inside DELAY_WAIT.  We only transition to STARTING
        # when the high-power streak has lasted at least
        # start_duration_threshold real seconds - measured between two
        # consecutive high readings, not from the dt to the previous
        # (low) reading.  This prevents a single isolated spike from
        # tripping STARTING just because the sampling interval is long.
        self._delay_wait_high_start: datetime | None = None
        self._delay_wait_high_power: float | None = None
        # Preserve a delayed-start candidate across a false STARTING probe
        # that drops back into the standby band without the machine truly
        # turning off.
        self._preserve_delay_band_on_off: bool = False
        # Register item 504: when the current STARTING probe came out of DELAY_WAIT,
        # the moment DELAY_WAIT was entered (its timeout anchor). A false start that
        # falls back into the standby band returns to DELAY_WAIT with this anchor
        # instead of to OFF, so a standby that straddles start_threshold_w keeps
        # waiting until a real start, a true off, or the band's own timeout.
        # None for every other probe. In memory only: a probe restored after a
        # restart falls back to OFF as before.
        self._probe_wait_since: datetime | None = None
        # Item 515: when the current probe came out of a terminal state, that state,
        # its entry time and its sub-state, for the false start to return to. None
        # for every other probe; in memory only (a restored probe falls to OFF).
        self._probe_terminal_from: tuple[str, datetime | None, str | None] | None = None
        # Register item 501: a standby that straddles start_threshold_w (#35: 2-22 W
        # around a 4.24 W threshold) re-probes on almost every reading.
        # `_standby_reprobe` is set when a false start falls back into the band
        # (>= stop_threshold_w) or a terminal state reads the band (item 510), and
        # cleared by any reading below it and by entering any state other than
        # OFF/STARTING. A probe that begins while it is set, or out
        # of DELAY_WAIT (a standby by definition), is HIDDEN: it runs exactly as any
        # other, but `exposed_state` keeps showing the state it began in until it
        # has filled STANDBY_REPROBE_SHOW_ENERGY_FRACTION of the energy gate.
        self._standby_reprobe: bool = False
        self._probe_hidden: bool = False
        self._probe_hidden_state: str = STATE_OFF
        self._probe_hidden_sub_state: str | None = None
        # Discussion #452, display only (see the STALL_* / IDLE_* constants).
        # Element 15: the matched profile's low stretches below the near-stop
        # ceiling, as element 14's (traced, ((fraction, seconds), ...)).
        self._matched_stall_catalogue: tuple[int, tuple[tuple[float, float], ...]] | None = None
        # The current flat near-stop run: its first reading and its spread.
        self._stall_run_start: datetime | None = None
        self._stall_run_lo: float = 0.0
        self._stall_run_hi: float = 0.0
        self._stall_active: bool = False
        # The match as it stood when the current run began (`_begin_stall_run`):
        # a plateau soon looks like a finished SHORTER programme to the matcher,
        # so the evidence is the pre-plateau match's. And (required seconds,
        # owes work, owes its terminal high-power block), computed from it once.
        self._stall_match: tuple[Any, ...] | None = None
        self._stall_eval: tuple[float, bool, bool] | None = None
        # ...and whether the CURRENT match agrees, re-read after every match.
        self._stall_now: tuple[bool, bool] | None = None
        # Item 514: the current run began straight out of activity.
        self._stall_run_abrupt: bool = False
        # Item 511: the runs this cycle ended while shown as stalled, as (elapsed
        # seconds at the run's first reading, run seconds), in order.
        self._stall_spans: list[tuple[float, float]] = []
        # The learned standby level (`learned_standby_level_w`, set by the
        # manager) and the debounced idle/off class it drives.
        self._standby_level_w: float | None = None
        self._idle_shown: bool = False
        self._idle_candidate_since: datetime | None = None
        # A LEARNED level shows idle only once the appliance has been seen at its
        # off level outside a cycle (held IDLE_DEBOUNCE_S) since start-up: an
        # appliance whose switched-off draw reaches the level would otherwise read
        # idle for ever, and an automation waiting for `off` would never fire.
        self._idle_off_seen: bool = False
        self._idle_off_since: datetime | None = None

    @property
    def exposed_state(self) -> str:
        """The state entities show. Detection never reads this.

        ``state``, except: during a hidden probe the state it began in (item 501:
        a standby that keeps crossing the start threshold otherwise wrote ~500
        off/starting rows a day and fired every automation keyed on
        ``starting``); ``paused`` while the open cycle is stalled (#452); and
        ``idle`` instead of ``off`` while a two-level appliance sits at its
        standby level (#452).
        """
        state = self._state
        if self._probe_hidden and state == STATE_STARTING:
            state = self._probe_hidden_state
        if state == STATE_OFF and self._idle_shown and self._idle_bounds() is not None:
            return STATE_IDLE
        if self._stall_active and state in (STATE_RUNNING, STATE_ENDING):
            return STATE_PAUSED
        return state

    @property
    def exposed_sub_state(self) -> str | None:
        """``sub_state`` to show, matching :attr:`exposed_state`."""
        exposed = self.exposed_state
        if exposed == STATE_IDLE:
            return STATE_IDLE.capitalize()
        if self._stall_active and exposed == STATE_PAUSED:
            return STALL_SUB_STATE
        if self._probe_hidden and self._state == STATE_STARTING:
            return self._probe_hidden_sub_state
        return self._sub_state

    @property
    def stalled(self) -> bool:
        """Whether the open cycle is stalled (#452). Display only."""
        return self._stall_active

    def stall_info(self) -> dict[str, Any] | None:
        """Small, event-safe description of the current stall, or None."""
        if not self._stall_active or self._stall_run_start is None:
            return None
        start = self._current_cycle_start
        last = self._last_process_time or self._stall_run_start
        return {
            "stalled_since": self._stall_run_start.isoformat(),
            "stalled_for_s": round(max(0.0, (last - self._stall_run_start).total_seconds())),
            "elapsed_at_stall_s": (
                round((self._stall_run_start - start).total_seconds()) if start else None
            ),
            "plateau_w": round((self._stall_run_lo + self._stall_run_hi) / 2.0, 2),
            # The match the halt began under when the halt has since cost it.
            "matched_profile": self._matched_profile or (
                self._stall_match[0] if self._stall_match else None
            ),
        }

    def set_standby_level(self, level_w: float | None) -> None:
        """The learned standby level for the idle display (#452); None: none."""
        try:
            level = float(level_w) if level_w is not None else None
        except (TypeError, ValueError, OverflowError):
            level = None
        self._standby_level_w = level if level is not None and math.isfinite(level) else None

    def _idle_bounds(self) -> tuple[float, float, float, bool] | None:
        """``(enter_w, exit_w, debounce_s, off level confirmed)`` of the idle
        display, or None (#452)."""
        cfg = self._config
        start = float(cfg.start_threshold_w)
        pot = cfg.power_off_threshold_w
        if isinstance(pot, (int, float)) and 0.0 < float(pot) < float(cfg.stop_threshold_w):
            # The user's own off level (#284): the same boundary and debounce the
            # power-based Off uses, so a reset to off never shows idle first.
            return (float(pot), float(pot), max(0.0, float(cfg.power_off_delay)), True)
        level = self._standby_level_w
        if level is None or level < IDLE_MIN_STANDBY_W or level >= start:
            return None
        enter = IDLE_OFF_FRACTION * level
        return (enter, IDLE_EXIT_FRACTION * enter, IDLE_DEBOUNCE_S, False)

    def _update_idle(self, timestamp: datetime, power: float) -> None:
        """Debounced idle/off class of the latest real reading (#452). O(1).

        A reading at or above the enter level votes idle, one below the exit
        level votes off, one in between keeps the current class. The class
        changes only once the other vote has held for the debounce, timed from
        its first reading, so an excursion shorter than that never shows.
        """
        bounds = self._idle_bounds()
        if bounds is None:
            self._idle_shown = False
            self._idle_candidate_since = None
            return
        enter, exit_w, debounce, confirmed = bounds
        if power >= enter:
            vote = True
        elif power < exit_w:
            vote = False
        else:
            vote = self._idle_shown
        if power < exit_w and self._state in (
            STATE_OFF, STATE_FINISHED, STATE_INTERRUPTED, STATE_FORCE_STOPPED
        ):
            if self._idle_off_since is None:
                self._idle_off_since = timestamp
            if (timestamp - self._idle_off_since).total_seconds() >= debounce:
                self._idle_off_seen = True
        else:
            self._idle_off_since = None
        if vote and not (confirmed or self._idle_off_seen):
            vote = False  # never seen switched off: not yet known to be two-level
        if vote == self._idle_shown:
            self._idle_candidate_since = None
            return
        if self._idle_candidate_since is None:
            self._idle_candidate_since = timestamp
        if (timestamp - self._idle_candidate_since).total_seconds() >= debounce:
            self._idle_shown = vote
            self._idle_candidate_since = None

    @classmethod
    def _sanitize_stall_catalogue(cls, raw: Any) -> Any:
        """Element 15, keeping only the stretches that can set a stall's wait.

        The near-stop catalogue holds every gap between two drum tumbles; a
        stretch under STALL_MIN_S / END_GATE_HAZARD_MARGIN can never raise the
        wait above STALL_MIN_S, so it is dropped. A callable (the producers' lazy
        form) is kept as is and resolved by :meth:`_resolve_stall_catalogue`.
        """
        if callable(raw):
            return raw
        cat = cls._sanitize_pause_catalogue(raw)
        if cat is None:
            return None
        floor = STALL_MIN_S / END_GATE_HAZARD_MARGIN
        return (cat[0], tuple(p for p in cat[1] if p[1] >= floor))

    @staticmethod
    def _sanitize_stall_spans(raw: Any) -> list[tuple[float, float]]:
        """Item 511's finished stalls from a snapshot: finite, non-negative pairs
        in order; anything else is dropped (an old snapshot has none)."""
        return match_rules.sanitize_stall_spans(raw)

    @classmethod
    def _resolve_stall_catalogue(
        cls, raw: Any
    ) -> tuple[int, tuple[tuple[float, float], ...]] | None:
        """Element 15 as data: a lazy producer is called here, once per run."""
        if callable(raw):
            try:
                raw = raw()
            except Exception:  # noqa: BLE001 - missing evidence, not an error
                return None
        return cls._sanitize_stall_catalogue(raw) if raw is not None else None

    def _seed_stall_run(self) -> None:
        """Re-derive the flat near-stop run in progress from the trace (#452)."""
        self._clear_stall()
        if self._state not in (STATE_RUNNING, STATE_PAUSED, STATE_ENDING):
            return
        band = self._stall_band()
        if band is None:
            return
        lo = hi = None
        start = None
        for ts, p in reversed(self._power_readings):
            p = float(p)
            if not band[0] <= p <= band[1]:
                break
            nlo = p if lo is None else min(lo, p)
            nhi = p if hi is None else max(hi, p)
            if nhi - nlo > band[2]:
                break
            lo, hi, start = nlo, nhi, ts
        if start is not None and lo is not None and hi is not None:
            # The restored match stands in for the one the run began under.
            self._begin_stall_run(start, lo)
            self._stall_run_hi = hi

    def _gate_spans(self) -> list[tuple[float, float]]:
        """The spans the gates' programme time leaves out, in order and merged:
        every finished stall (item 511) and every finished user pause (item 514)."""
        stalls = self._stall_spans if STALL_EXCLUDED_FROM_GATES else []
        pauses = self._user_pause_spans if USER_PAUSE_EXCLUDED_FROM_GATES else []
        if not pauses:
            return stalls
        if not stalls:
            return pauses
        return _merge_spans([*stalls, *pauses])

    def _gate_elapsed_s(self, timestamp: datetime) -> float:
        """Elapsed seconds the duration-based end gates read (item 511): the
        cycle's wall clock less every finished stall and user pause (item 514).
        0.0 with no cycle open."""
        start = self._current_cycle_start
        if start is None:
            return 0.0
        elapsed = (timestamp - start).total_seconds()
        spans = self._gate_spans()
        if spans:
            elapsed = max(0.0, elapsed - sum(length for _at, length in spans))
        return elapsed

    def _wall_offset_s(self, offset_s: float) -> float:
        """A programme-time offset from the cycle start as wall-clock seconds
        (item 511): each left-out span that began before it moves it later."""
        shift = 0.0
        for at, length in self._gate_spans():
            if offset_s < at - shift:
                break
            shift += length
        return offset_s + shift

    def _match_readings(self) -> list[tuple[datetime, float]]:
        """The trace the live matcher reads (item 511): each finished stall cut
        out and every later reading moved back by it, so the duration, the
        Stage-1 ratio and the in-progress Stage-4 stretch are programme time and
        the shape has no plateau the programme never made. A stall still shown
        stays in, as before: its evidence includes the current match's reading
        of it. The stored trace keeps every reading."""
        return self._cut_readings(self._gate_spans())

    def _cut_readings(
        self, spans: list[tuple[float, float]]
    ) -> list[tuple[datetime, float]]:
        """The trace with ``spans`` cut out, every later reading moved back by them
        (`match_rules.stall_cut_plan`). The trace itself when there is none."""
        start = self._current_cycle_start
        if not spans or start is None:
            return self._power_readings
        readings = self._power_readings
        plan = match_rules.stall_cut_plan(
            [(ts - start).total_seconds() for ts, _p in readings], spans
        )
        return [
            (readings[i][0] - timedelta(seconds=shift), readings[i][1]) if shift
            else readings[i]
            for i, shift in plan
        ]

    def _progress_spans(
        self, timestamp: datetime | None
    ) -> tuple[list[tuple[float, float]], tuple[float, float] | None]:
        """``(finished stalls, the stall shown now as (at, seconds so far) or None)``
        that progress leaves out (item 514)."""
        if not STALL_EXCLUDED_FROM_PROGRESS:
            return [], None
        start, run = self._current_cycle_start, self._stall_run_start
        current = None
        if (
            STALL_CURRENT_EXCLUDED_FROM_PROGRESS and self._stall_active
            and start is not None and run is not None and timestamp is not None
        ):
            current = (
                max(0.0, (run - start).total_seconds()),
                max(0.0, (dt_util.as_utc(timestamp) - run).total_seconds()),
            )
        return list(self._stall_spans), current

    def progress_elapsed_s(self, elapsed_s: float, timestamp: datetime | None) -> float:
        """``elapsed_s`` less the stalled time progress leaves out (item 514).

        ``elapsed_s`` is the caller's own clock: the manager's net of the user
        pause, the Playground's replay offset. ``timestamp`` is now (the stall
        shown at this moment counts up to it)."""
        finished, current = self._progress_spans(timestamp)
        stalled = sum(length for _at, length in finished)
        if current is not None:
            stalled += current[1]
        return max(0.0, float(elapsed_s) - stalled)

    def progress_trace(self, timestamp: datetime | None) -> list[tuple[datetime, float]]:
        """The trace progress reads (item 514): the live matcher's, and with a
        stall shown now everything from its first reading on is left out too.
        A copy, like :meth:`get_power_trace`."""
        finished, current = self._progress_spans(timestamp)
        if current is not None:
            finished = [*finished, (current[0], math.inf)]
        return list(self._cut_readings(finished))

    def _bank_stall(self, timestamp: datetime) -> None:
        """Keep the run that ends at ``timestamp`` if it is shown as stalled."""
        start, run_start = self._current_cycle_start, self._stall_run_start
        if self._stall_active and start is not None and run_start is not None:
            self._stall_spans.append((
                max(0.0, (run_start - start).total_seconds()),
                max(0.0, (timestamp - run_start).total_seconds()),
            ))

    def _clear_stall(self) -> None:
        self._stall_run_start = None
        self._stall_run_lo = self._stall_run_hi = 0.0
        self._stall_active = False
        self._stall_match = None
        self._stall_eval = None
        self._stall_now = None
        self._stall_run_abrupt = False

    def _begin_stall_run(self, timestamp: datetime, power: float) -> None:
        """A new flat near-stop run starts at this reading: snapshot the match."""
        self._stall_run_start = timestamp
        self._stall_run_lo = self._stall_run_hi = power
        self._stall_eval = None
        self._stall_now = None
        # Item 514: straight out of activity, judged on the reading before this
        # one (a restart re-derives it from the restored trace).
        self._stall_run_abrupt = False
        band = self._stall_band() if STALL_ABRUPT_PEAK_FRACTION > 0 else None
        if band is not None:
            prev = next(
                (float(p) for ts, p in reversed(self._power_readings) if ts < timestamp), None
            )
            self._stall_run_abrupt = prev is not None and prev >= max(
                STALL_ABRUPT_PEAK_FRACTION * float(self._cycle_max_power), 2.0 * band[1]
            )
        self._stall_match = (
            self._matched_profile if self._expected_duration > 0 else None,
            float(self._expected_duration),
            self._matched_terminal_high,
            self._matched_stall_catalogue,
        )

    def _stall_evidence(self) -> tuple[float, bool, bool] | None:
        """``(required s, owes work, owes its spin)`` of the current run, or None."""
        if self._stall_run_start is None:
            return None
        if self._stall_eval is None:
            self._stall_eval = self._stall_evaluate()
        return self._stall_eval

    def _stall_holds_standby_band(self) -> bool:
        """Whether the current flat run says the programme is not done (#452).

        Only the strongest evidence holds a finalize: the match the run began
        under ends on a terminal high-power block (#399) this cycle has not
        produced yet. A matcher that re-reads the halt as a finished SHORTER
        programme cannot release it, and the plain position test the display also
        accepts holds nothing (a programme matched to a longer one ends "early"
        every time). Independent of the display's wait: the finalize's own 10 min
        window and expected-duration gate already say "long enough".
        """
        if not STALL_HOLDS_STANDBY_BAND or self._stall_band() is None:
            return False
        evidence = self._stall_evidence()
        if self._stall_abrupt_owes():
            return True  # item 514: an abrupt halt under a match that owes work
        return bool(evidence and evidence[2] and self._stall_now_evidence()[1])

    def _stall_abrupt_owes(self) -> bool:
        """Item 514: the run began straight out of activity under a MATCHED
        programme that still owes work, so the current match's verdict (the
        plateau read as a finished shorter programme) is not needed."""
        if not (self._stall_run_abrupt and self._stall_match and self._stall_match[0]):
            return False
        evidence = self._stall_evidence()
        return bool(evidence and evidence[1])

    def _stall_now_evidence(self) -> tuple[bool, bool]:
        """``(owes work, owes its spin)`` by the CURRENT match, for the current run.

        Both matches must agree before anything shows or holds: the one the run
        began under can be a prefix match of a longer programme (the last live
        tick before a real end differs from the complete match on 17.5% of
        cycles, audit MATCH-DECIDE-02), and the current one can be the halt read
        as a finished shorter programme. Measured with ``end_gate_eval.py
        --halt-at``: requiring both cut real ends held under a 45 min display-on
        tail 15 -> 6 of 258 and kept 28 of the 48 halts the pre-plateau match
        alone kept open. With no current match the pre-plateau verdict stands.
        """
        if self._stall_now is not None:
            return self._stall_now
        start, run_start = self._current_cycle_start, self._stall_run_start
        expected = float(self._expected_duration)
        block = self._matched_terminal_high
        if not (self._matched_profile and expected > 0 and start and run_start):
            evidence = self._stall_evidence()
            now = (bool(evidence and evidence[1]), bool(evidence and evidence[2]))
        elif block is not None and block[0] >= ANTI_CREASE_TERMINAL_HIGH_MIN_FRAC:
            needed = float(block[1]) * ANTI_CREASE_TERMINAL_MATCH_FRAC
            offset_s = float(block[2]) if len(block) >= 3 else float(block[0]) * expected
            ceiling_w = float(block[3]) if len(block) >= 4 else None
            spin = needed > 0 and self._high_power_seconds_since(
                offset_s, ceiling_w=ceiling_w
            ) < needed
            now = (spin, spin)
        else:
            # Programme time (item 511): an earlier stall is not progress.
            position = self._gate_elapsed_s(run_start) / expected
            now = (position < STALL_OWES_MAX_POSITION, False)
        self._stall_now = now
        return now

    def _set_stalled(self, stalled: bool, timestamp: datetime) -> None:
        """Flip the stall flag (one place, so the eval harness can observe it)."""
        if stalled == self._stall_active:
            return
        self._stall_active = stalled
        if stalled:
            self._logger.info(
                "Cycle stalled: flat %.1f-%.1f W (stop %.2f W) for %.0fs, matched %s; "
                "shown as paused (display only, the cycle stays open)",
                self._stall_run_lo,
                self._stall_run_hi,
                self._config.stop_threshold_w,
                (timestamp - (self._stall_run_start or timestamp)).total_seconds(),
                self._matched_profile,
            )
        else:
            self._logger.debug("Cycle no longer stalled at %s", timestamp)

    def _stall_band(self) -> tuple[float, float, float] | None:
        """``(low_w, high_w, flatness_w)`` a stall plateau must sit in, or None."""
        if self._config.device_type not in STALL_DEVICE_TYPES:
            return None
        stop = float(self._config.stop_threshold_w)
        peak = float(self._cycle_max_power)
        if stop <= 0 or peak <= 0:
            return None
        high = min(standby_near_stop_ceiling(stop), peak * STANDBY_BAND_MAX_FRACTION)
        if high < stop:
            return None
        flat = max(STANDBY_BAND_FLATNESS_FLOOR_W, peak * STANDBY_BAND_FLATNESS_FRACTION)
        return (stop, high, flat)

    def _stall_evaluate(self) -> tuple[float, bool, bool]:
        """``(required seconds, owes work, owes its spin)`` for the current run,
        from the match it began under."""
        start = self._current_cycle_start
        run_start = self._stall_run_start
        name, expected, block, cat = self._stall_match or (None, 0.0, None, None)
        if not (name and expected > 0 and start and run_start):
            return (STALL_UNMATCHED_MIN_S, True, False)
        cat = self._resolve_stall_catalogue(cat)
        position = max(0.0, self._gate_elapsed_s(run_start) / expected)  # item 511
        # Owes work: the terminal high-power block (#399) not yet produced, or,
        # for a programme without one, a run that began well before its end.
        owes_spin = False
        if block is not None and block[0] >= ANTI_CREASE_TERMINAL_HIGH_MIN_FRAC:
            needed = float(block[1]) * ANTI_CREASE_TERMINAL_MATCH_FRAC
            offset_s = float(block[2]) if len(block) >= 3 else float(block[0]) * expected
            ceiling_w = float(block[3]) if len(block) >= 4 else None
            owes_spin = needed > 0 and self._high_power_seconds_since(
                offset_s, ceiling_w=ceiling_w
            ) < needed
            owes = owes_spin
        else:
            owes = position < STALL_OWES_MAX_POSITION
        # Ambiguity is not a reason to refuse here: the display only waits, it
        # never shortens anything (unlike the hazard gate this mirrors).
        if cat is None or cat[0] < END_GATE_HAZARD_MIN_CYCLES:
            return (STALL_UNMATCHED_MIN_S, owes, owes_spin)
        later = [d for f, d in cat[1] if f >= position - END_GATE_HAZARD_POSITION_SLACK]
        need = max(STALL_MIN_S, END_GATE_HAZARD_MARGIN * max(later, default=0.0))
        return (need, owes, owes_spin)

    def _update_stall(self, timestamp: datetime, power: float) -> None:
        """Track the flat near-stop run of an open cycle and flag a stall (#452).

        O(1) per reading; the evidence is computed once per run and match.
        """
        band = self._stall_band()
        if band is None or not (band[0] <= power <= band[1]):
            if self._stall_run_start is not None or self._stall_active:
                self._bank_stall(timestamp)  # item 511
                self._set_stalled(False, timestamp)
                self._clear_stall()
            return
        if self._stall_run_start is None:
            self._begin_stall_run(timestamp, power)
        else:
            lo = min(self._stall_run_lo, power)
            hi = max(self._stall_run_hi, power)
            if hi - lo > band[2] and not self._stall_active:
                # Not flat: a new run starts here. A stall already shown stays
                # shown while readings stay in the band.
                self._begin_stall_run(timestamp, power)
            else:
                self._stall_run_lo, self._stall_run_hi = lo, hi
        run_s = (timestamp - self._stall_run_start).total_seconds()
        if self._stall_eval is None and run_s < min(STALL_MIN_S, STALL_UNMATCHED_MIN_S):
            return
        evidence = self._stall_evidence()
        if evidence is not None:
            self._set_stalled(
                run_s >= evidence[0] and evidence[1]
                and (self._stall_now_evidence()[0] or self._stall_abrupt_owes()),
                timestamp,
            )

    def _hide_probe(self, hidden: bool) -> None:
        """Item 501: whether the probe about to begin is hidden. Called BEFORE the
        transition to STARTING, whose callback already refreshes the entities."""
        self._probe_hidden = hidden
        self._probe_hidden_state = self._state
        self._probe_hidden_sub_state = self._sub_state

    def _return_probe_to_terminal(self, timestamp: datetime, power: float) -> None:
        """Item 515: a false start goes back to the terminal state it began in.

        With that state's entry time and sub-state, and without the probe's
        readings, as ``reset()`` leaves a terminal state: no cycle is open, so a
        later force-stop or "Done now" there records nothing. A reading still in
        the band marks the next probe as a standby re-probe (items 501, 510).
        """
        back = self._probe_terminal_from
        if back is None:
            return
        state, entered, sub_state = back
        self._transition_to(state, timestamp)
        if entered is not None:
            self._state_enter_time = entered
            self._time_in_state = max(0.0, (timestamp - entered).total_seconds())
        self._sub_state = sub_state
        self._power_readings = []
        self._current_cycle_start = None
        self._last_active_time = None
        self._cycle_max_power = 0.0
        self._energy_since_idle_wh = 0.0
        self._time_above_threshold = 0.0
        self._standby_reprobe = power >= self._config.stop_threshold_w

    def _show_probe_with_evidence(self) -> None:
        """Item 501: a hidden probe is shown once it has filled part of the energy gate."""
        if self._probe_hidden and self._energy_since_idle_wh >= (
            STANDBY_REPROBE_SHOW_ENERGY_FRACTION * self._config.start_energy_threshold
        ):
            self._probe_hidden = False

    @property
    def _gate_cadence(self) -> float:
        """Cadence the pause/end gates are sized from (#424/#427).

        ``_p95_dt`` is the 2nd-largest of the last 20 intervals by construction,
        so it tracks the worst *gap* rather than the reporting rate. That is the
        right input for the outage ceilings (a gap must be judged against the
        worst gap we consider normal) but the wrong one for the gates below,
        which multiply it by three: once a publish-on-change plug falls silent at
        standby, the only intervals left are the long quiet ones, p95 collapses
        onto them, and each gate becomes ~3x the silence it is supposed to be
        measuring. The accumulator advances one interval per reading, so the
        cycle then needs ~3 more readings - a gate that is set by, and grows
        with, its own input. Measured on the #427 trace: a 297 s sensor silence
        followed by a 481 s keepalive lifted the end gate from 45 s to 1455 s and
        held the cycle in PAUSED for 25.5 min.

        Capping p95 at a multiple of the *median* keeps the estimate robust: a
        genuinely slow sensor reports slowly every time, so its median equals its
        p95 and the cap never binds (a 300 s-cadence meter keeps its 900 s gate);
        a fast sensor that went quiet has a small median, so the isolated holes
        cannot triple the gate. Only these two gates read it - ``_p95_dt`` itself
        is left alone so every outage ceiling keeps the cadence snapshot it was
        tuned against (register items 213, 215).

        The cap alone was not enough, and the reason is worth keeping: it is
        computed over the same 20-interval window it is meant to protect. Once a
        publish-on-change plug falls silent the watchdog's own 0 W keepalives are
        the only readings left, so the median collapses onto the injection spacing
        too and ``5 x median`` stops binding - the gate then grows with our
        injection rate instead of the plug's. Fixed at the source (register item
        289): ``process_reading`` no longer trains the cadence on synthetic
        readings, so this window describes the sensor and nothing else.
        """
        if len(self._recent_dts) < 5:
            return self._p95_dt
        median_dt = median_fast(self._recent_dts)
        return min(self._p95_dt, GATE_CADENCE_MEDIAN_FACTOR * median_dt)

    @property
    def _dynamic_pause_threshold(self) -> float:
        """Calculate dynamic pause threshold based on sampling cadence."""
        # User requirement: T_pause >= 3 * p95_update_interval
        # Default 15s or 3 * p95
        return max(15.0, 3.0 * self._gate_cadence)

    @property
    def _dynamic_end_threshold(self) -> float:
        """Calculate dynamic end candidate threshold."""
        # Keep this generic for pause->ending transitions across all device types.
        base = 3.0 * self._gate_cadence
        # Ensure end threshold is at least 15s greater than pause threshold
        return max(base, self._dynamic_pause_threshold + 15.0)

    def _update_cadence(self, dt: float) -> None:
        """Update rolling cadence statistics."""
        if dt <= 0.1:
            return
        self._recent_dts.append(dt)
        if len(self._recent_dts) > 20:
            self._recent_dts.pop(0)

        # Calculate p95 if enough samples
        if len(self._recent_dts) >= 5:
            # Pure Python, bit-identical to np.percentile (audit PERF-07): on 20
            # values NumPy's call overhead was the detector's top per-sample cost.
            self._p95_dt = percentile_linear(self._recent_dts, 95)
        else:
            self._p95_dt = max(dt, 1.0)

    def _try_profile_match(self, timestamp: datetime, force: bool = False) -> None:
        """Attempt to invoke the profile matcher if conditions are met.

        Args:
            timestamp: Current timestamp.
            force: If True, run match immediately regardless of interval.
        """
        if not self._profile_matcher:
            return
        if not self._power_readings:
            return

        # Terminal-tail match freeze (dishwashers): once we are in ENDING with a
        # profile already matched and power has been sustained-quiet, the active
        # cycle is over - only the passive drain/dry tail remains. Re-matching on
        # the growing idle tail inflates the observed duration and drifts the
        # Stage-4 duration-agreement toward a LONGER near-duplicate profile,
        # flipping the label and stalling smart-termination on the ambiguity gate.
        # Keep the active-phase match instead. Self-correcting: a real resume sends
        # a high reading that leaves ENDING, so this guard stops applying.
        if (
            self._state == STATE_ENDING
            and self._config.device_type == "dishwasher"
            and self._matched_profile
            and self._time_below_threshold >= DISHWASHER_MATCH_FREEZE_QUIET_SECONDS
        ):
            # The freeze also stops the manager's match tick, the only place the #375
            # sustained-quiet release ran: a verified pause engaged before the freeze
            # was never released, and every ENDING finalize stayed blocked until the
            # force stop (~8 h on a plug that keeps reporting 0 W; found by the F7
            # parity harness on three TRON4R dishwasher cycles). Run the same rule here.
            if self._verified_pause and not self._user_paused and self._power_readings:
                pause = match_rules.decide_pause_release(
                    verified_pause=True,
                    current_matched=self._matched_profile,
                    current_power=float(self._power_readings[-1][1]),
                    stop_threshold_w=float(self._config.stop_threshold_w),
                    user_paused=False,
                    expected_duration=float(self._expected_duration or 0.0),
                    current_duration=(
                        self._power_readings[-1][0] - self._power_readings[0][0]
                    ).total_seconds(),
                    time_below=self._time_below_threshold_gapfree,
                    program=self._matched_profile,
                )
                for level, msg, args in pause.log:
                    self._logger.log(level, msg, *args)
                self._verified_pause = pause.verified_pause
            return

        # Terminal-tail match freeze (anti-crease, #296): once a washer/dryer with
        # anti-wrinkle enabled is past its expected duration and has settled into
        # the low-power tumble tail, re-matching on the growing flat tail drifts the
        # label toward a LONGER near-duplicate (its expected duration grows), which
        # pushes the anti-crease finalize gate out and breaks Smart Termination -
        # the field failure that merges back-to-back washes.  Keep the good
        # pre-tail match instead.  Self-correcting: a new wash's heating burst above
        # anti_wrinkle_max_power leaves the tail regime, so this stops applying and
        # matching re-arms for the next cycle.
        if self._matched_profile and self._in_anticrease_freeze(timestamp):
            return

        # Rate limiting
        if not force and self._last_match_time:
            elapsed = (timestamp - self._last_match_time).total_seconds()
            interval = float(self._config.match_interval)
            if not getattr(self, "_match_committed", True):
                interval /= 2.0
            if elapsed < interval:
                return

        self._last_match_time = timestamp

        # Call the matcher
        try:
            result = self._profile_matcher(self._match_readings())
            # If synchronous result returned, process it.
            # If None returned (async offload), the matcher is responsible for
            # calling update_match later.
            if result:
                self.update_match(result)

        except Exception as e:  # pylint: disable=broad-exception-caught
            self._logger.debug("Profile match failed: %s", e)

    # Maximum reasonable cycle duration accepted by the detector.  Anything
    # longer is rejected as corrupted data and replaced with the
    # _SANITIZE_INVALID_SENTINEL so downstream gates fall through to the
    # unmatched / no-expected-duration path.
    _SANITIZE_MAX_EXPECTED_DURATION = 6 * 3600.0  # 6 hours
    _SANITIZE_INVALID_SENTINEL = 0.0  # 0 == "no valid expected_duration"

    def _sanitize_expected_duration(
        self, raw: Any, *, source: str = "update_match"
    ) -> float:
        """Coerce ``raw`` into a finite float in (0, 6h] or return 0.0.

        The class invariant is that ``self._expected_duration`` is either a
        finite, strictly positive float ≤ 6 hours, or 0.0 meaning "no valid
        expected duration".  Every code path that assigns ``_expected_duration``
        (live profile-match callbacks AND restored snapshots) routes through
        this helper so the gates in STATE_ENDING and ``_should_defer_finish``
        can trust the value without re-validating.

        Emits a DEBUG log line distinguishing the rejection reason - the
        ``<= 0`` and ``> 6h`` markers are part of issue #197's regression
        contract and tests assert on them.
        """
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            self._logger.debug(
                "%s: invalid raw_expected_duration %r, defaulting to 0.0",
                source, raw,
            )
            return self._SANITIZE_INVALID_SENTINEL
        if not math.isfinite(value):
            self._logger.debug(
                "%s: invalid raw_expected_duration %r, defaulting to 0.0",
                source, raw,
            )
            return self._SANITIZE_INVALID_SENTINEL
        if value <= 0:
            self._logger.debug(
                "%s: invalid raw_expected_duration %r (<= 0), defaulting to 0.0",
                source, raw,
            )
            return self._SANITIZE_INVALID_SENTINEL
        if value > self._SANITIZE_MAX_EXPECTED_DURATION:
            self._logger.debug(
                "%s: invalid raw_expected_duration %r (> 6h), defaulting to 0.0",
                source, raw,
            )
            return self._SANITIZE_INVALID_SENTINEL
        return value

    @staticmethod
    def _sanitize_terminal_quiet(raw: Any) -> float | None:
        """Coerce a measured post-activity quiet span into a finite, non-negative
        float, else None (register item 297).

        Same discipline as the two siblings: None means "no opinion", and for a
        DISHWASHER ``_keep_tail_cap`` then behaves exactly as it did before this
        element existed. Other device types never reach that branch - they are
        capped at ``_last_active_time`` higher up - so None changes nothing for
        them either way. Bounded above by ``TERMINAL_QUIET_CAP_S`` so a corrupted
        or hand-edited value cannot license an unbounded tail - the one thing
        this field exists to prevent.
        """
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(value) or value < 0:
            return None
        return min(value, TERMINAL_QUIET_CAP_S)

    @staticmethod
    def _sanitize_positive(raw: Any) -> float | None:
        """A finite positive float from a snapshot, else None."""
        if raw is None or isinstance(raw, bool):
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        return value if math.isfinite(value) and value > 0.0 else None

    @staticmethod
    def _sanitize_trusted_min(raw: Any) -> float | None:
        """Coerce element 13 to a positive duration or None ("nothing vouched")."""
        if raw is None or isinstance(raw, bool):
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(value) or not 0.0 < value <= CycleDetector._SANITIZE_MAX_EXPECTED_DURATION:
            return None
        return value

    @staticmethod
    def _sanitize_pause_catalogue(
        raw: Any,
    ) -> tuple[int, tuple[tuple[float, float], ...]] | None:
        """Coerce element 14 to ``(traced, ((fraction, seconds), ...))`` or None."""
        try:
            traced, pauses = raw
            out = tuple(
                (float(f), float(d)) for f, d in pauses
                if math.isfinite(float(f)) and math.isfinite(float(d)) and float(d) >= 0
            )
            return (int(traced), out) if int(traced) > 0 else None
        except (TypeError, ValueError, OverflowError):
            return None

    def _hazard_wait(self, timestamp: datetime, base: float) -> float:
        """The ENDING fallback wait the matched profile's own pauses justify.

        Audit DETECT-16: "no soak left after 1.05x expected" generalised to "no
        soak this programme has ever shown from this position": END_GATE_HAZARD_
        MARGIN x the longest resumed pause its traced evidence began at or after
        this quiet's start (less a slack), clamped to [off_delay, base]. Shorten-
        only, unambiguous matches only, and only with END_GATE_HAZARD_MIN_CYCLES
        traced cycles; otherwise ``base``. Measured (prototype, 292 cycles): washer
        median lag 16.17 -> 12.50 min, early ends 0 -> 0, splits unchanged.
        """
        return self._hazard_wait_for(
            timestamp, base, self._matched_pause_catalogue,
            self._expected_duration, self._match_ambiguous,
        )

    def _hazard_wait_for(
        self, timestamp: datetime, base: float, cat: Any, expected: float, ambiguous: bool
    ) -> float:
        """:meth:`_hazard_wait` for a match with these values (register item 469b)."""
        if (
            cat is None
            or cat[0] < END_GATE_HAZARD_MIN_CYCLES
            or not self._matched_profile
            or expected <= 0
            or self._current_cycle_start is None
            or ambiguous
        ):
            return base
        elapsed = self._gate_elapsed_s(timestamp)  # programme time (item 511)
        # Where the quiet run began. An outage inside it is still part of the run
        # (item 266), or the outage would read as progress and drop the very pause
        # being waited out from `later`.
        quiet_run = self._time_below_threshold + self._time_below_unobserved
        position = max(0.0, (elapsed - quiet_run) / expected)
        later = [d for f, d in cat[1] if f >= position - END_GATE_HAZARD_POSITION_SLACK]
        need = END_GATE_HAZARD_MARGIN * max(later) if later else 0.0
        return max(float(self._config.off_delay), min(float(base), need))

    @staticmethod
    def _sanitize_longest_candidate(raw: Any) -> float:
        """Coerce the longest plausible candidate duration to a usable bound.

        Unlike the three siblings above, "no opinion" is ``0.0`` rather than
        ``None``: the ENDING gate reads a non-positive bound as "no information"
        and keeps the old refusal, which is the safe direction. Not routed
        through ``_sanitize_expected_duration`` because 0.0 is legitimate here
        and that helper logs it as invalid.
        """
        try:
            value = float(raw or 0.0)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        if not math.isfinite(value):
            return 0.0
        return value if 0.0 < value <= CycleDetector._SANITIZE_MAX_EXPECTED_DURATION else 0.0

    @staticmethod
    def _sanitize_tail_power(raw: Any) -> float | None:
        """Coerce ``raw`` into a finite, positive float, else None (#364).

        None means "no opinion": ``_smart_term_power_plausible`` then leaves both
        Smart-Termination paths exactly as they behaved before the guard existed.
        """
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            # OverflowError alongside the type errors: `json` keeps an integer
            # literal of any length as an unbounded `int`, and `float()` on one
            # raises rather than returning `inf`, so the non-finite filter below is
            # never reached. Both callers are the reason it matters - the restored
            # state snapshot is hand-editable `.storage` JSON, and
            # `restore_state_snapshot`'s one broad `except` answers a raise with
            # `self.reset()`, which discards the WHOLE restored cycle rather than
            # this one field. "No opinion" is the documented contract; a full reset
            # is not.
            return None
        if not math.isfinite(value) or value <= 0:
            return None
        return value

    @staticmethod
    def _sanitize_terminal_high(
        raw: Any,
    ) -> (
        tuple[float, float]
        | tuple[float, float, float]
        | tuple[float, float, float, float]
        | None
    ):
        """Coerce ``raw`` into a ``(start_frac, seconds)`` pair, a
        ``(start_frac, seconds, start_offset_s)`` triple, or a
        ``(start_frac, seconds, start_offset_s, ceiling_w)`` quad, else None (#399).

        The fourth element is the watts the block was MEASURED against (register
        item 351). Anti-crease measures against ``anti_wrinkle_max_power`` and
        sends a triple; the standby-band path measures against a share of the
        cycle's own peak and must say so, because the live counter has to count
        seconds above the same bar or the two halves compare different things.

        None means "no opinion", which leaves ``_anticrease_spin_pending`` inert and
        the anti-crease finalise exactly as it behaved before the guard existed.

        The arity is PRESERVED rather than normalised (register item 196). A
        two-element payload is what a pre-196 state snapshot, an older caller and
        most tests supply, and it has to keep meaning "no absolute offset, fall back
        to ``start_frac x expected``" - handing it a fabricated 0.0 offset would make
        the guard scan the whole cycle. A malformed third element degrades to the
        pair for the same reason the whole method returns None on garbage: it must
        never be able to disarm a guard that would otherwise arm.
        """
        if raw is None:
            return None
        # A str/bytes is iterable, so `list("11")` is `["1", "1"]` and sanitizes to
        # (1.0, 1.0) - a scalar string silently ARMING the guard off a malformed
        # snapshot, which is the one direction this method promises never to go.
        # Rejected before the iteration so it takes the documented garbage path.
        if isinstance(raw, (str, bytes, bytearray)):
            return None
        try:
            values = list(raw)
        except TypeError:
            return None
        if len(values) not in (2, 3, 4):
            return None
        try:
            start_frac = float(values[0])
            seconds = float(values[1])
        except (TypeError, ValueError, OverflowError):
            # Same reason as _sanitize_tail_power above: an unbounded int raises out
            # of float(), and a raise here costs the whole restored cycle state.
            return None
        if not math.isfinite(start_frac) or not math.isfinite(seconds):
            return None
        if not 0.0 <= start_frac <= 1.0 or seconds <= 0:
            return None
        if len(values) == 2:
            return (start_frac, seconds)
        try:
            start_offset = float(values[2])
        except (TypeError, ValueError, OverflowError):
            return (start_frac, seconds)
        if not math.isfinite(start_offset) or start_offset < 0:
            return (start_frac, seconds)
        if len(values) == 3:
            return (start_frac, seconds, start_offset)
        try:
            ceiling = float(values[3])
        except (TypeError, ValueError, OverflowError):
            return (start_frac, seconds, start_offset)
        # A non-positive or non-finite ceiling degrades to the triple rather than
        # to None: the triple still arms the guard against
        # ``anti_wrinkle_max_power``, and this method must never be able to
        # DISARM a guard that would otherwise arm.
        if not math.isfinite(ceiling) or ceiling <= 0:
            return (start_frac, seconds, start_offset)
        return (start_frac, seconds, start_offset, ceiling)

    def _trailing_mean_power(self, timestamp: datetime, window_s: float) -> float | None:
        """Time-weighted mean power over the trailing ``window_s``, or None when
        there are too few samples to judge.

        Time-weighted (not a plain sample mean) so an irregular reporting cadence -
        a plug that only pushes on change, so quiet stretches are sparse - cannot
        bias the result toward whichever regime happened to sample more often.
        """
        window: list[tuple[datetime, float]] = []
        for ts, power in reversed(self._power_readings):
            if (timestamp - ts).total_seconds() > window_s:
                break
            window.append((ts, float(power)))
        window.reverse()
        # Explicit gap handling (energy-integration rule): a reading held across an
        # unobserved outage would dominate the trapezoid - e.g. a stale 2000 W sample
        # 290 s before a 284 s dropout, then 5 W, integrates to ~1000 W though every
        # observed recent reading is 5 W, wrongly blocking termination. So drop
        # everything up to and including the most recent outage-sized gap and judge
        # only the clean contiguous tail, the same ceiling the standby / anti-crease
        # window scans reject a holed window with.
        if len(window) >= 2:
            # O(1) ceiling from the maintained p95 cadence (mirrors energy_gap_threshold_s
            # = clip(10x cadence, 60, 3600)), NOT _outage_threshold_s() which rebuilds a
            # NumPy array from every reading - this runs on the per-reading ENDING /
            # anti-crease path, same reasoning as the gap-free tally at L1006.
            max_gap = min(3600.0, max(60.0, 10.0 * self._prior_p95_dt))
            cut = 0
            for i in range(1, len(window)):
                if (window[i][0] - window[i - 1][0]).total_seconds() > max_gap:
                    cut = i
            window = window[cut:]
        if len(window) < SMART_TERM_TAIL_MIN_POINTS:
            return None
        span = (window[-1][0] - window[0][0]).total_seconds()
        if span <= 0:
            return None
        energy = 0.0
        for (t0, p0), (t1, p1) in zip(window, window[1:]):
            energy += (p0 + p1) / 2.0 * (t1 - t0).total_seconds()
        return energy / span

    def _smart_term_power_plausible(self, timestamp: datetime) -> bool:
        """Whether the appliance looks like it is actually FINISHING (#364).

        Both Smart-Termination paths fire at ``elapsed >= 0.98 * expected`` and
        neither asks whether the machine is still working.  When the matcher has
        locked onto a shorter look-alike profile that anchor lands mid-wash, the
        cycle is cut in half and the remainder is recorded as a second cycle.

        The test: compare the trailing mean power against what the matched profile
        itself draws at its own end.  Drawing several times that level is proof we
        are not at the end of anything - whatever the clock says.  Unlike the
        prefix-landscape guard this needs no longer profile to exist in the pool,
        so it also covers the reported case where the programme actually running
        was never trained.

        Shorten-only and fail-open: any missing input returns True, leaving
        behaviour identical to before the guard.  A False can only ever *block* an
        early finish - the power-based fallback timeout still ends the cycle.
        """
        tail_power = self._matched_tail_power
        if tail_power is None or tail_power <= 0:
            return True
        mean_power = self._trailing_mean_power(timestamp, self._tail_window_s())
        if mean_power is None:
            return True
        return mean_power <= tail_power * SMART_TERM_TAIL_MAX_RATIO

    def _tail_window_s(self) -> float:
        """Trailing window that covers the same FRACTION of the run as the profile
        tail it is compared against.

        A fixed window is not comparable across programme lengths: 300 s is 4% of a
        cotton wash but a third of a 15-minute spin-and-drain, whose trailing mean
        would then be the spin itself while its profile tail is the quiet moment
        after the pump stops. Measured on the full corpus, making this proportional
        is strictly better at every threshold.
        """
        expected = self._expected_duration
        if expected <= 0:
            return SMART_TERM_TAIL_WINDOW_S
        return min(
            SMART_TERM_TAIL_WINDOW_S,
            max(SMART_TERM_TAIL_WINDOW_MIN_S, expected * SMART_TERM_TAIL_WINDOW_FRAC),
        )

    def update_match(self, result: tuple[Any, ...] | list[Any] | Any) -> None:  # type: ignore[misc]
        """Process a match result (synchronously).

        Can be called by the matcher callback directly or asynchronously.
        """
        # Terminal-tail match freeze (anti-crease, #296).  This is the single sink
        # for ALL match updates - the detector's own _try_profile_match AND the
        # manager's async 5-min matcher (manager.py calls update_match directly).
        # Once a washer/dryer with anti-wrinkle enabled is past its expected
        # duration and has settled into the low-power tumble tail, re-matching on
        # the growing flat tail drifts the label toward a LONGER near-duplicate (or
        # flips it ambiguous), which pushes out expected_duration and would block
        # the anti-crease finalize - the field failure that merges back-to-back
        # washes.  Keep the good pre-tail match instead.  Self-correcting: a new
        # wash's heating burst above anti_wrinkle_max_power leaves the tail regime,
        # so this stops applying and matching re-arms for the next cycle.
        if (
            self._matched_profile
            and self._power_readings
            and self._in_anticrease_freeze(self._power_readings[-1][0])
        ):
            return
        if isinstance(result, MatchContext):
            result = result.as_sequence()
        # Register item 469(b): an ambiguous match in ENDING may not make the end
        # later (see `ambiguous_ending_match_defers`). Like the freezes above, the
        # detector keeps the match it has; a revoke still applies.
        if (
            match_rules.HOLD_AMBIGUOUS_IN_ENDING
            and isinstance(result, (list, tuple))
            and len(result) >= 6
            and result[0] is not None
            and bool(result[5])
            and not bool(result[4])
            and self.ambiguous_ending_match_defers(
                result[2], result[1], result[11] if len(result) >= 12 else 0.0,
                result[13] if len(result) >= 14 else None,
            )
        ):
            self._logger.debug(
                "ENDING: ambiguous match %s not applied, %s kept (item 469)",
                result[0], self._matched_profile,
            )
            return
        # Unpack 5 elements (or 4 for backward compatibility if needed, but wrapper is updated)
        # wrapper returns (name, confidence, duration, phase, is_mismatch)
        # Or MatchResult object if refactored, but currently wrapper returns tuple.

        # Register item 498: what a revoke below discards, kept as the evidence
        # that bounds a verified pause it leaves behind.
        revoked_from = self._matched_profile
        revoked_catalogue = self._matched_pause_catalogue

        is_match_mismatch = False
        match_name: str | None = None
        phase_name: str | None = None
        confidence: float = 0.0
        expected_duration: float = 0.0
        ambiguous: bool = False

        if isinstance(result, (list, tuple)):  # type: ignore[misc]
            result_seq = cast(tuple[Any, ...] | list[Any], result)
            # Optional 6th element: whether the live match is ambiguous
            # (top-1 vs top-2 within MATCH_AMBIGUITY_MARGIN). Used to gate the
            # predictive Smart Termination below.
            if len(result_seq) >= 6:
                ambiguous = bool(result_seq[5])
            if len(result_seq) >= 5:
                (
                    raw_name,
                    raw_confidence,
                    raw_expected_duration,
                    raw_phase_name,
                    raw_mismatch,
                ) = result_seq[:5]
                match_name = str(raw_name) if raw_name is not None else None
                try:
                    confidence = float(raw_confidence)
                    if not math.isfinite(confidence):
                        confidence = 0.0
                        self._logger.debug("update_match: invalid raw_confidence %r, defaulting to 0.0", raw_confidence)
                except (TypeError, ValueError, OverflowError):
                    confidence = 0.0
                    self._logger.debug("update_match: invalid raw_confidence %r, defaulting to 0.0", raw_confidence)
                expected_duration = self._sanitize_expected_duration(
                    raw_expected_duration, source="update_match"
                )
                phase_name = str(raw_phase_name) if raw_phase_name is not None else None
                is_match_mismatch = raw_mismatch if isinstance(raw_mismatch, bool) else bool(raw_mismatch)
            else:
                # Fallback for old signature
                if len(result_seq) >= 4:
                    (
                        raw_name,
                        raw_confidence,
                        raw_expected_duration,
                        raw_phase_name,
                    ) = result_seq[:4]
                    match_name = str(raw_name) if raw_name is not None else None
                    try:
                        confidence = float(raw_confidence)
                        if not math.isfinite(confidence):
                            confidence = 0.0
                            self._logger.debug("update_match: invalid raw_confidence %r, defaulting to 0.0", raw_confidence)
                    except (TypeError, ValueError, OverflowError):
                        confidence = 0.0
                        self._logger.debug("update_match: invalid raw_confidence %r, defaulting to 0.0", raw_confidence)
                    expected_duration = self._sanitize_expected_duration(
                        raw_expected_duration, source="update_match"
                    )
                    phase_name = (
                        str(raw_phase_name) if raw_phase_name is not None else None
                    )
                    is_match_mismatch = False

            # Store confidence + ambiguity for Smart Termination checks
            self._last_match_confidence = confidence or 0.0
            self._match_ambiguous = ambiguous
            # Element 8: the #288 full-shape term. Element 7 (the removed #364
            # prefix-fit flag) is read only as its fallback for a 7-element legacy
            # tuple, which carried the #288 verdict there before #364 split it out.
            self._match_prefix_ambiguous_full_shape = (
                bool(result_seq[7]) if len(result_seq) >= 8
                else bool(result_seq[6]) if len(result_seq) >= 7
                else False
            )
            # Element 9 (#364): the matched profile's own tail power level. Absent
            # or non-finite leaves the power-plausibility guard inert - so a shorter
            # tuple (Playground, older callers, most tests) must CLEAR it, not keep
            # the previous match's value: retaining it would let the guard compare
            # the live tail against the wrong profile and block a valid termination.
            self._matched_tail_power = (
                self._sanitize_tail_power(result_seq[8]) if len(result_seq) >= 9 else None
            )
            # Element 10 (#399): the matched profile's own terminal high-power
            # block. Cleared by a shorter tuple for the same reason as element 9 -
            # keeping the previous profile's block would make the anti-crease guard
            # wait for a spin the newly-matched program does not have.
            self._matched_terminal_high = (
                self._sanitize_terminal_high(result_seq[9]) if len(result_seq) >= 10 else None
            )
            # Element 11 (register item 297): cleared by a shorter tuple for the same reason as
            # elements 9 and 10 - a newly matched programme must not inherit the
            # previous one's tail.
            self._matched_terminal_quiet_s = (
                self._sanitize_terminal_quiet(result_seq[10])
                if len(result_seq) >= 11
                else None
            )
            # Element 12: longest plausible candidate duration (see the attribute's
            # own comment). A shorter tuple clears it, like elements 9-11, so a
            # stale value can never license a shortening for a different match.
            # Coerced quietly, not through _sanitize_expected_duration: 0.0 is a
            # legitimate "no candidate durations to compare" here, and that helper
            # logs it as invalid.
            self._longest_candidate_duration = self._sanitize_longest_candidate(
                result_seq[11] if len(result_seq) >= 12 else 0.0
            )
            # Element 13: cleared by a shorter tuple, like elements 9-12.
            self._matched_trusted_min_s = self._sanitize_trusted_min(
                result_seq[12] if len(result_seq) >= 13 else None
            )
            # Element 14: cleared by a shorter tuple, like elements 9-13.
            self._matched_pause_catalogue = self._sanitize_pause_catalogue(
                result_seq[13] if len(result_seq) >= 14 else None
            )
            # Element 15 (#452): likewise. A run already in progress keeps the
            # match it began under (`_stall_match`).
            self._matched_stall_catalogue = self._sanitize_stall_catalogue(
                result_seq[14] if len(result_seq) >= 15 else None
            )
            self._stall_now = None  # the current match is re-read for a run in progress
        else:
            # Assume MatchResult object or similar (future proofing)
            # But for now wrapper returns tuple
            return

        if is_match_mismatch and self._matched_profile:
            # Confident non-match - revert to detecting if previously matched.
            # The expected duration goes with it, as in reset(): the duration-
            # anchored hard finalize, the dishwasher end-spike arm gate, the
            # keep-tail cap and the manager's zombie killer read it without
            # checking `_matched_profile`, so a revoked programme's length kept
            # steering an unmatched cycle.
            self._matched_profile = None
            self._expected_duration = 0.0
            self._match_ambiguous = False
            self._match_prefix_ambiguous_full_shape = False
            self._matched_tail_power = None
            self._matched_terminal_high = None
            self._matched_terminal_quiet_s = None
            self._matched_trusted_min_s = None
            self._matched_pause_catalogue = None
            self._matched_stall_catalogue = None

        elif match_name:
            # If sanitization rejected the expected_duration, treat the match
            # as invalid: setting _matched_profile while _expected_duration is
            # the 0.0 sentinel would let Smart Termination fire on the
            # `current_duration >= 0` always-true comparison.  Drop both so
            # the cycle stays in detecting/unmatched mode.
            if expected_duration == self._SANITIZE_INVALID_SENTINEL:
                self._logger.debug(
                    "update_match: match %r ignored - expected_duration "
                    "sanitized to invalid sentinel; treating as unmatched",
                    match_name,
                )
                self._matched_profile = None
                self._expected_duration = self._SANITIZE_INVALID_SENTINEL
            else:
                self._matched_profile = match_name
                # Sub-state can be set from phase_name if available
                if phase_name:
                    self._sub_state = phase_name
                # Wrapper provides it
                self._expected_duration = expected_duration

        if revoked_from and not self._matched_profile and revoked_catalogue is not None:
            # Either path above that drops the match (a revoke, or a named match
            # with an unusable duration). Max over the cycle's revokes.
            self._orphaned_pause_evidence_s = max(
                self._orphaned_pause_evidence_s,
                max((d for _f, d in revoked_catalogue[1]), default=0.0),
            )

    def _release_orphaned_pause(self) -> None:
        """Release a verified pause a revoked match left behind (register item 498).

        After a revoke nothing could clear an automatic verified pause but high
        power: the 95%-of-span release needs the match, the #375 release its
        expected duration. So it held every ENDING finisher until the 8 h force stop
        (a plug reporting 0 W) or the watchdog's 4.5 h silence limit (a silent plug).
        The bounded wait is :func:`match_rules.orphaned_pause_wait_s`; a user pause
        is never released. Called on every ENDING reading (watchdog keepalives and
        the Playground's replay included), after this reading's match tick.
        """
        if not self._verified_pause or self._user_paused or self._matched_profile:
            return
        longest = self._orphaned_pause_evidence_s
        pause = match_rules.decide_orphaned_pause_release(
            verified_pause=True,
            user_paused=False,
            current_matched=None,
            time_below=self._time_below_threshold_gapfree,
            wait_s=match_rules.orphaned_pause_wait_s(
                off_delay=self._config.off_delay,
                min_off_gap=self._config.min_off_gap,
                longest_pause_s=longest,
            ),
            longest_pause_s=longest,
        )
        for level, msg, args in pause.log:
            self._logger.log(level, msg, *args)
        self._verified_pause = pause.verified_pause

    def set_verified_pause(self, verified: bool) -> None:
        """Set or clear the verified pause flag."""
        self._verified_pause = verified

    def set_match_committed(self, committed: bool) -> None:
        """Mirror whether the caller has committed a program this cycle.

        Until it has, matches run at half ``match_interval`` (audit LIVE-17): at
        the shipped 300 s, 12 resampled points on a 30 s plug need 330 s, so the
        300 s try was empty too and the first commit came a median 25 min in.
        Measured: right programme shown 56.2 -> 61.0% of cycle time, never
        committed 8 -> 4 of 247, +19% matcher CPU. Persistence stays a match
        count, as in the measured arm.
        """
        self._match_committed = bool(committed)

    def set_user_paused(self, paused: bool, timestamp: datetime | None = None) -> None:
        """Mirror the user's Pause Cycle state (set by the manager).

        Kept apart from ``_verified_pause`` on purpose: the envelope auto-pause
        writes that one too, and freezing Smart Termination on it would re-open
        the #375 hang class. Only a USER pause blocks Smart Termination.

        Item 514: a resume banks the pause (from ``timestamp``, now by default)
        as a span the gates' programme time leaves out. A pause start restored
        from a snapshot is kept when the manager re-asserts the pause.
        """
        paused = bool(paused)
        now = dt_util.as_utc(timestamp) if timestamp is not None else utc_now()
        if paused and not self._user_paused:
            if self._user_pause_since is None:
                self._user_pause_since = now
        elif not paused:
            since, start = self._user_pause_since, self._current_cycle_start
            self._user_pause_since = None
            if self._user_paused and since is not None and start is not None:
                begin = max(since, start)
                length = (now - begin).total_seconds()
                if length > 0:
                    self._user_pause_spans = _merge_spans([
                        *self._user_pause_spans,
                        ((begin - start).total_seconds(), length),
                    ])
        self._user_paused = paused

    def mark_sensor_unavailable(self, timestamp: datetime) -> None:
        """Start a sensor outage: the power sensor has no usable value (item 266).

        Called by the manager when the sensor's state becomes unavailable, unknown
        or non-numeric. Until the next real reading nothing is observed, so that
        span is not credited as quiet, and a watchdog keepalive inside it takes no
        decision (see ``process_reading``). Idempotent: an ongoing outage keeps
        its first start.
        """
        if self._sensor_outage_since is None:
            self._sensor_outage_since = dt_util.as_utc(timestamp)
            in_cycle = self._state in (
                STATE_STARTING, STATE_RUNNING, STATE_PAUSED, STATE_ENDING
            )
            self._logger.log(
                logging.INFO if in_cycle else logging.DEBUG,
                "Power sensor unavailable from %s (state %s): quiet time is not "
                "credited until it reports again",
                self._sensor_outage_since,
                self._state,
            )

    def reset(
        self, target_state: str = STATE_OFF, timestamp: datetime | None = None
    ) -> None:
        """Force reset the detector state to target state.

        ``timestamp`` is the reading that caused the reset. The state entered here
        is timed from it (ANTI_WRINKLE's 2 h exit reads ``_state_enter_time``), so a
        replay with historical timestamps must pass it: stamping the host clock
        left a replayed dryer in ANTI_WRINKLE forever (audit DETECT-10).
        """
        # Item 515: the manager's Off expiry of a terminal state is a display change,
        # not a fresh start. Since false starts return to the terminal state, the
        # standby's re-probe flag (item 501) and the probes' pre-roll readings live
        # there now, as they did in OFF, and carry into it (the cycle end emptied
        # the buffer, so it holds nothing of that cycle).
        idle_carry = (
            TERMINAL_PROBE_RETURNS
            and target_state == STATE_OFF
            and self._state in (STATE_FINISHED, STATE_INTERRUPTED, STATE_FORCE_STOPPED)
        )
        carried = (self._standby_reprobe, self._preroll_buffer) if idle_carry else None
        self._transition_to(
            target_state,
            dt_util.as_utc(timestamp) if timestamp is not None else utc_now(),
        )
        self._power_readings = []
        # #430: the pre-roll buffer is pre-CYCLE context, never cross-cycle. A
        # reset that left it populated would let the previous cycle's tail be
        # spliced into the front of the next cycle's curve.
        self._preroll_buffer = []
        self._current_cycle_start = None
        self._last_active_time = None
        self._cycle_max_power = 0.0
        self._energy_since_idle_wh = 0.0
        self._time_above_threshold = 0.0
        # Only reset time_below_threshold if not transitioning to ANTI_WRINKLE
        # (ANTI_WRINKLE needs to track idle time to determine true-off)
        if target_state != STATE_ANTI_WRINKLE:
            self._time_below_threshold = 0.0
            self._time_below_threshold_gapfree = 0.0
            self._time_below_unobserved = 0.0
        self._last_match_time = None
        self._match_committed = False
        self._matched_profile = None
        # Clear stale match state so the next cycle starts with clean defaults.
        # _expected_duration left at 0 tells the dishwasher end-spike gate that
        # no profile is matched yet; stale non-zero would mis-gate the spike check.
        self._expected_duration = 0.0
        self._last_match_confidence = 0.0
        self._match_ambiguous = False
        self._match_prefix_ambiguous_full_shape = False
        self._matched_tail_power = None
        self._matched_terminal_high = None
        self._matched_terminal_quiet_s = None
        self._matched_trusted_min_s = None
        self._matched_pause_catalogue = None
        self._matched_stall_catalogue = None
        self._orphaned_pause_evidence_s = 0.0
        # Element 12 belongs with them: its own comment claims a stale value can
        # never license a shortening for a different match, and that was only
        # true of the tuple path. Left here across a reset, a small positive
        # bound survives into the next cycle, where it is neither greater than
        # `_expected_duration` (so the bar is not raised) nor <= 0 (so the "no
        # information" refusal does not fire) - the one combination that lets an
        # ambiguous match shorten `effective_off_delay` on no evidence.
        self._longest_candidate_duration = 0.0
        self._anticrease_spin_wait_logged = False
        # Per-cycle diagnostic throttle (#346): the "Smart Termination not applied"
        # line only logs when the reason CHANGES. Carrying the previous cycle's
        # reason across a reset swallows the new cycle's very first diagnostic
        # whenever it happens to be blocked for the same reason.
        self._last_smart_term_block_reason = None
        self._ignore_power_until_idle = False  # Reset lockout
        self._lockout_high_seconds = 0.0
        # Clear the verified-pause flag so it can't leak into the next cycle (B6):
        # a stale True would make an early low-power dip look like a verified pause
        # before the first live match of the new cycle runs.
        self._verified_pause = False
        self._anti_wrinkle_candidate_start = None
        self._anti_wrinkle_candidate_peak = 0.0
        self._anti_wrinkle_candidate_start_power = 0.0
        self._anticrease_tail_floor_w = None
        # Reset idle time tracker for anti-wrinkle
        self._anti_wrinkle_idle_time = 0.0
        # Reset delayed-start tracking
        self._delay_band_seconds = 0.0
        self._delay_band_peak = 0.0
        self._delay_wait_true_off_seconds = 0.0
        self._starting_paused_off_since = None
        self._delay_wait_high_start = None
        # Item 501: a forced state is a fresh start, never a standby re-probe.
        self._standby_reprobe = False
        self._probe_hidden = False
        self._probe_wait_since = None
        self._probe_terminal_from = None
        if carried is not None:
            self._standby_reprobe, self._preroll_buffer = carried

    @property
    def state(self) -> str:
        """Return current state."""
        return self._state

    @property
    def sub_state(self) -> str | None:
        """Return current sub-state."""
        return self._sub_state

    @property
    def config(self) -> CycleDetectorConfig:
        """Return current configuration."""
        return self._config

    @property
    def matched_profile(self) -> str | None:
        """Return the name of the matched profile, if any."""
        return self._matched_profile

    @property
    def current_cycle_start(self) -> datetime | None:
        """Return the start timestamp of the current cycle."""
        return self._current_cycle_start

    @property
    def samples_recorded(self) -> int:
        """Return the number of power samples recorded in current cycle."""
        return len(self._power_readings)

    @property
    def expected_duration_seconds(self) -> float:
        """Return the expected duration of the current cycle in seconds."""
        return self._expected_duration

    @staticmethod
    def _smart_term_block_reason(
        current_duration: float,
        expected: float,
        smart_ratio: float,
        is_confident: bool,
        ambiguous: bool,
        power_plausible: bool = True,
    ) -> str | None:
        """Why the Smart-Termination fast end-path did NOT fire, for diagnostics.

        Returns None when the gate would pass, or when no expected duration is known
        yet (nothing meaningful to report). Mirrors the gate's conditions in order so
        the first blocking reason is surfaced. Pure and side-effect-free; the
        detector logs the result (throttled to reason changes) - no behaviour
        change (#346, extended with "still_active" for #364).
        """
        if expected <= 0:
            return None
        if current_duration < expected * smart_ratio:
            return "duration_not_reached"
        if not is_confident:
            return "low_confidence"
        if ambiguous:
            return "match_ambiguous"
        if not power_plausible:
            return "still_active"
        return None

    @staticmethod
    def _resolve_smart_ratio(
        device_type: str,
        configured_ratio: float,
        end_spike_seen: bool,
        end_spike_duration: float,
        expected_duration: float,
    ) -> float:
        """Resolve the Smart-Termination duration-ratio gate (#393).

        ``configured_ratio`` is the per-device option, already resolved in the
        config builder to the device-type default (0.99 dishwasher / 0.98 other)
        unless the user tuned it - so it is always a real float here.

        For a dishwasher whose most-recent in-ENDING spike landed at >=90% of the
        expected duration, that spike is the terminal pump-out (not a mid-cycle
        rinse drain): once it is confirmed the gate is loosened to the 0.90
        pump-out relief, because individual cycles can be a few % shorter than the
        rolling average and still terminate cleanly. Keeping the configured gate
        for spikes at <90% prevents premature closes during the passive Dry phase
        that follows the pre-final-rinse drain. The relief is combined with the
        configured value via ``min()`` so a configured ratio can only ever LOOSEN
        the gate, never tighten it. Pure and side-effect-free (unit-testable).
        """
        # Clamp to the documented [0.50, 1.00] range (the WS write path clamps, but a
        # value persisted by an import or an older schema is read here unclamped): a
        # 0.0 would drop the duration floor entirely and let Smart Termination fire the
        # moment its other conditions pass.
        configured_ratio = min(1.0, max(0.5, configured_ratio))
        if (
            device_type == "dishwasher"
            and end_spike_seen
            and expected_duration > 0
            and end_spike_duration >= expected_duration * 0.90
        ):
            return min(configured_ratio, 0.90)
        return configured_ratio

    def process_reading(
        self,
        power: float,
        timestamp: datetime,
        synthetic: bool = False,
        observed: bool = True,
    ) -> None:
        """Process a new power reading using robust dt-aware logic.

        ``synthetic=True`` marks a reading the *manager* injected rather than one
        the power sensor sent: the watchdog and anti-wrinkle keepalives, which
        exist to advance the quiet timers while a change-only plug says nothing.
        They must keep doing exactly that, so this flag does not change the quiet
        accumulators. What it does change is the two places that reason about what
        the SENSOR did (#424):

        * ``_update_cadence`` is skipped. The cadence estimate feeds
          ``_gate_cadence`` and therefore the pause/end gates, so training it on
          our own injections makes those gates a function of how often we inject -
          see the note on ``_gate_cadence``.
        * **with ``observed=True``**, the gap-free tally treats the interval as
          observed, because the watchdog re-anchored on the sensor's live state
          (``_resync_power_from_state``) before injecting - so the keepalive is a
          moment we looked rather than a hole in the record.

        ``observed=False`` says the sensor state could NOT be read when this
        keepalive was injected (unavailable / unknown / non-finite). The second
        bullet is then false: the interval it closes is a genuine outage, and the
        gap-free tally is reset like any other hole - at ANY step size, since the
        watchdog injects far more often than the outage ceiling.

        A RECORDED outage (``mark_sensor_unavailable``, register item 266) goes
        further, because only it says where the hole began: from its start until
        the next real reading, time is not added to ``_time_below_threshold``
        either, so an outage can no longer run out an end gate. A keepalive inside
        one advances the clock and nothing else - no state-machine step, no
        sample in the trace - so no decision rests on a value nobody read. The
        manager's staleness force-stop still ends a cycle whose sensor stays dead.
        ``observed=False`` alone (a late watchdog tick, item 391) keeps its
        narrower meaning: it resets the gap-free tally only.

        (The `_keep_tail_cap` use this flag was originally added for, register
        item 238, was implemented, measured and reverted: after
        ``_last_active_time`` every reading is below the stop threshold anyway, so
        a plug still reporting cannot separate a drying phase from standby - it
        only shows the plug is chatty. See item 260 for two more that were
        measured and rejected.)
        """
        # Every interval below is a datetime subtraction. Two aware datetimes that
        # share one tzinfo instance - which every `dt_util.now()` stamp does - are
        # subtracted on their WALL-CLOCK fields, so across a DST change dt, elapsed
        # and the stored duration are off by an hour (a spring-forward soak was
        # credited 3960 s of quiet and split the wash; audit DETECT-01). UTC has no
        # transitions, and mixed-zone comparisons elsewhere stay correct.
        timestamp = dt_util.as_utc(timestamp)
        outage_since = self._sensor_outage_since
        if not synthetic:
            self._last_real_reading_time = timestamp
            if outage_since is not None:
                # The sensor spoke: the outage is over (item 266).
                self._sensor_outage_since = None
                self._logger.debug(
                    "Power sensor reporting again after %.0fs unavailable; that "
                    "span was not counted as quiet",
                    (timestamp - outage_since).total_seconds(),
                )

        # Calculate dt (needed by the stop lockout below and the state machine).
        dt = 0.0
        if self._last_process_time:
            dt = (timestamp - self._last_process_time).total_seconds()

        # Sanity check for negative dt
        if dt < 0:
            # Logged because this is the one exit from process_reading that leaves
            # NO trace: the accumulators do not advance, `_power_readings` does not
            # grow, and every downstream gate therefore stays silent too. A cycle
            # wedged behind it looks exactly like a cycle whose end gate is simply
            # not satisfied - which is how much of register item 320 was spent.
            self._logger.debug(
                "Ignoring reading %.1fW: timestamp %s is %.1fs before the last "
                "processed reading %s",
                power,
                timestamp,
                -dt,
                self._last_process_time,
            )
            self._last_process_time = timestamp
            return

        # Manual Stop Lockout:
        # If user/external stop forced an end, ignore the machine's spin-down so
        # it is not logged as a new cycle. The lockout clears the moment power
        # drops to idle. As a safety net, if power instead stays high far longer
        # than any plausible spin-down, treat it as a genuinely new back-to-back
        # load and release the lockout so the cycle is detected immediately
        # rather than pinned until the progress-reset window expires (#267).
        if self._ignore_power_until_idle:
            if power < self._config.start_threshold_w:
                self._ignore_power_until_idle = False
                self._lockout_high_seconds = 0.0
                self._logger.debug(
                    "Power dropped below start threshold. Manual stop lockout cleared."
                )
            else:
                self._lockout_high_seconds += dt
                if self._lockout_high_seconds < STOP_LOCKOUT_RELEASE_SECONDS:
                    # Still within the spin-down window - ignore reading.
                    # The reading is withheld from the state machine, but it is
                    # still a real observation of the power level, so record it
                    # (#403): the accumulator below judges each interval against
                    # the previous observation, and a release reading compared to
                    # a pre-stop sample up to the full lockout window old would
                    # lose the credit for its own interval (#267 back-to-back
                    # start whose stop happened in a low-power trough).
                    self._last_process_time = timestamp
                    self._last_power = power
                    return
                self._ignore_power_until_idle = False
                self._lockout_high_seconds = 0.0
                self._logger.info(
                    "Manual stop lockout released after sustained power "
                    "(>= %.1fs at/above start threshold): treating as a new "
                    "cycle (#267).",
                    STOP_LOCKOUT_RELEASE_SECONDS,
                )
                # Fall through: the state machine will start a new cycle.

        # Snapshot the cadence BEFORE folding this reading in: the gap-free tally
        # below classifies `dt` against a ceiling derived from the cadence, and an
        # outage that has already widened p95 would raise the very threshold that
        # is supposed to catch it (a 120 s gap after a 10 s cadence lifts p95 to
        # ~15.5 s -> ceiling 155 s -> the gap counts as observed quiet).
        self._prior_p95_dt = self._p95_dt
        # Below five intervals `_p95_dt` is just the last interval (or the 1 s
        # default), not a cadence, so nothing can be called outage-sized yet.
        prior_cadence_known = len(self._recent_dts) >= 5
        # Only the SENSOR trains the cadence estimator (#424). The watchdog's 0 W
        # keepalives exist because the plug fell silent, so once it does every
        # interval left in `_recent_dts` is one we manufactured: p95 AND the
        # median both collapse onto the injection spacing, the `5 x median` cap in
        # `_gate_cadence` stops binding, and the pause/end gates - three times that
        # cadence - grow with our own injection rate. Measured on the #424
        # reporter's v0.5.6 cycle: the gate climbed 192 s -> 530 s on injected
        # readings alone, taking the end gate to 1605 s and holding PAUSED for
        # 1060 s. Skipping them leaves the estimate describing the plug, which is
        # the only thing it is supposed to describe; the accumulators below still
        # advance on every reading, synthetic or not.
        if not synthetic:
            self._update_cadence(dt)
        self._last_process_time = timestamp

        # 1b. Pre-roll buffer (#430): record every reading seen while no cycle is
        # open, so a start that needed several probes can recover what the aborted
        # ones took with them. Cheap and bounded; inert while the option is off.
        self._record_preroll(power, timestamp)

        # 2. Accumulators Update
        # Hysteresis Logic
        if self._state in (STATE_OFF, STATE_DELAY_WAIT, STATE_STARTING, STATE_UNKNOWN):
            threshold = self._config.start_threshold_w
        elif self._state in (STATE_FINISHED, STATE_INTERRUPTED, STATE_FORCE_STOPPED):
            # No cycle is open here either, so a terminal state starts on
            # start_threshold_w like OFF (register item 510). It fell to the stop
            # threshold by omission when these states were added: a display left
            # on above stop probed STARTING on its first reading, a probe that
            # cannot commit (STARTING measures against start_threshold_w), and the
            # manager cleared Finished, Clean and the unload nag on it. The higher
            # of the two, so an inverted pair (stop above start, seen in a
            # contributed export) is not made easier to leave than before.
            threshold = max(self._config.start_threshold_w, self._config.stop_threshold_w)
        else:
            threshold = self._config.stop_threshold_w

        is_high = power >= threshold

        # Last observation carried forward (#403): `dt` is the interval that
        # ENDED at this reading, so the appliance sat at the PREVIOUS sample's
        # level for it, not at this one. With a change-only (send-on-delta)
        # power sensor a low -> high crossing carries the whole idle gap, and
        # crediting it at the new high power let a single blip after minutes of
        # silence satisfy both start gates on the next reading. So the interval
        # only counts as high-power evidence when the previous observation was
        # also at or above the threshold those gates measure against. A densely
        # sampled device is unaffected: there the previous sample is already
        # high and the interval keeps its full credit.
        #
        # This is the same principle the surrounding code already applies - the
        # low branch restarts its gap-free tally rather than credit an outage,
        # DELAY_WAIT and the paused-STARTING anchor (#306) anchor on the first
        # high reading, and `integrate_wh`/`energy_gap_threshold_s` drop
        # outage-sized segments - applied to the one branch that still credited
        # unobserved time. An outage heuristic cannot substitute for it: a
        # 511 s gap on a 70 s idle cadence is legitimate change-only silence,
        # well inside the outage ceiling, and only the credit direction
        # separates it from real high-power time.
        #
        # A reading inside the hysteresis band (>= stop_threshold_w but
        # < start_threshold_w) therefore earns no evidence toward the start
        # gates, which is correct: the band is by definition below the
        # threshold the gates measure against, and it is exactly where a
        # waiting machine idles. The cost is one extra report before
        # confirmation on a band-crossing ramp; no start is lost.
        prev_high = self._last_power is not None and self._last_power >= threshold

        # The part of `dt` inside a recorded sensor outage (item 266, audit
        # DETECT-13): from the outage start - or the previous reading, if later -
        # up to this one. 0.0 whenever the sensor never went unavailable.
        unobserved = 0.0
        if outage_since is not None and dt > 0:
            unobserved = min(dt, max(0.0, (timestamp - outage_since).total_seconds()))
        # Outage-sized interval: clip(10x cadence, 60, 3600) like
        # energy_gap_threshold_s, from the maintained p95 (O(1) in this hot path)
        # as it stood BEFORE this reading, so a gap cannot widen its own ceiling.
        outage_ceiling = min(3600.0, max(60.0, 10.0 * self._prior_p95_dt))

        # Carrying the previous level forward assumes somebody was looking (item
        # 266). Time inside a recorded outage, and a sensor interval the cadence
        # calls an outage, is not high-power evidence either: crediting it let a
        # plug that dropped out mid-heat confirm a start and bank its last power
        # for the whole hole, which the stored energy (`integrate_wh` drops
        # outage-sized segments) never counted. Synthetic keepalives are exempt
        # from the size test for the reason given in the low branch below.
        high_dt = 0.0
        if prev_high:
            high_dt = dt - unobserved
            if not synthetic and prior_cadence_known and dt > outage_ceiling:
                high_dt = 0.0
        # ...and the ENERGY for that interval at the level the appliance actually sat
        # at, which is the same argument applied to the second start gate. Crediting
        # it at the NEW reading's power let a sample barely above the threshold,
        # followed by a spike, bank the spike's power for the whole preceding
        # interval and satisfy start_energy_threshold on its own. Computed here, not
        # at the three use sites, because `self._last_power` is overwritten a few
        # lines below - before the two STARTING seeds further down would read it.
        # The sibling paths already do this: the DELAY_WAIT seed is this same
        # accumulator over its high streak (item 504) and the anti-wrinkle window
        # uses the trapezoid average.
        high_step_wh = (
            (self._last_power or 0.0) * (high_dt / 3600.0) if high_dt > 0 else 0.0
        )

        if is_high:
            self._time_above_threshold += high_dt
            self._time_below_threshold = 0.0
            self._time_below_threshold_gapfree = 0.0
            self._time_below_unobserved = 0.0
            # Energy for the guarded interval, computed with high_dt above.
            self._energy_since_idle_wh += high_step_wh
            self._last_active_time = timestamp
        else:
            # Quiet nobody saw is not quiet (item 266). The watchdog keeps
            # injecting while the sensor is unavailable, and crediting those
            # ticks let a dropout that began on a low reading run out the
            # fallback timeout: a washer was closed `completed` mid-wash and the
            # rest recorded as a second cycle. Frozen, not reset, so the quiet
            # observed on either side still counts and `is_waiting_low_power`
            # keeps the manager's staleness force-stop armed for a dead sensor.
            self._time_below_threshold += dt - unobserved
            self._time_below_unobserved += unobserved
            # Gap-free tally: an outage-sized step is unobserved time, so restart
            # the observed-quiet tally from this sample instead of crediting the
            # gap. Ceiling mirrors energy_gap_threshold_s (clip(10x cadence, 60,
            # 3600)) but reuses the maintained p95 cadence to stay O(1) in this
            # per-reading hot path. Uses the cadence as it stood BEFORE this
            # reading, so a gap cannot widen its own acceptance threshold.
            # A synthetic keepalive is normally not an outage: the watchdog
            # resyncs against the sensor's live state before injecting, so the
            # interval it closes IS observed. This matters because the ceiling is
            # derived from the p95 cadence, which synthetic readings no longer
            # train - without the exemption a 106 s keepalive on a 2 s-cadence
            # plug would look like a 106 s hole and reset the tally on every
            # tick, starving the two consumers that can only ever SHORTEN the
            # wait (the dishwasher end-spike quiet release and the ENDING hard
            # finalize).
            #
            # `observed` is what makes that premise true rather than assumed.
            # `_resync_power_from_state` returns early when the sensor is
            # unavailable / unknown / non-finite, but the watchdog injects anyway
            # - it only checks the silence interval. So during a real telemetry
            # outage every keepalive was exempt and the gap-free tally grew
            # through quiet nobody ever saw, which is exactly what that tally
            # exists not to count. The caller now says whether the sensor state
            # could actually be read, and an unread sensor is an outage.
            # An unread sensor is an outage HOWEVER SHORT each keepalive step
            # is, which is why this is its own clause and not a qualifier on the
            # ceiling test. The watchdog injects once per `watchdog_interval`
            # (floor 30 s, effective 30-60 s) and the ceiling is at least 60 s,
            # so during a real outage every individual `dt` sits under the
            # ceiling - qualifying the ceiling test left the tally accumulating
            # exactly as before, which is the bug this is meant to fix.
            # (`outage_ceiling` is computed once, above the high-power credit.)
            if (
                (synthetic and not observed)
                or unobserved > 0
                or (dt > outage_ceiling and not synthetic)
            ):
                self._time_below_threshold_gapfree = 0.0
            else:
                self._time_below_threshold_gapfree += dt
            self._time_above_threshold = 0.0

        self._time_in_state += dt

        self._last_power = power
        if power < self._config.stop_threshold_w:
            # Back on the idle floor: the next probe is a fresh one (item 501).
            self._standby_reprobe = False
        if not synthetic:
            # #452 idle display: real readings only (a keepalive is not a level).
            self._update_idle(timestamp, power)

        # A keepalive inside a recorded outage carries no observation, only the
        # clock (item 266). Stepping the state machine on it would hand every
        # finisher a window of the 0 W the manager injects for an unreadable
        # sensor - the anti-crease finalize reads exactly such a window - and
        # write that invented 0 W into the trace and the matcher's input. So it
        # stops here; the next real reading picks up from the frozen tallies.
        if (
            synthetic
            and outage_since is not None
            and self._state in (STATE_STARTING, STATE_RUNNING, STATE_PAUSED, STATE_ENDING)
        ):
            return

        anti_wrinkle_active = (
            self._config.anti_wrinkle_enabled
            and self._config.device_type in (
                DEVICE_TYPE_WASHING_MACHINE,
                DEVICE_TYPE_DRYER,
                DEVICE_TYPE_WASHER_DRYER,
            )
        )

        # 3. State Machine

        if self._state in (
            STATE_OFF,
            STATE_FINISHED,
            STATE_INTERRUPTED,
            STATE_FORCE_STOPPED,
            STATE_ANTI_WRINKLE,
        ):
            started_from_anti_wrinkle = False
            tail_floor = (
                self._anticrease_tail_floor_w
                if self._state == STATE_ANTI_WRINKLE
                else None
            )
            if (
                anti_wrinkle_active
                and tail_floor is not None
                and is_high
                and power <= tail_floor
            ):
                # Register item 393a: the #296 tail's own baseline between bursts.
                # It can sit above stop_threshold (the 3.3 W Knitterschutz draw
                # against a ~1.5 W threshold), where it never reset the candidate,
                # so every tail left anti-wrinkle after anti_wrinkle_max_duration.
                self._anti_wrinkle_candidate_start = None
                self._anti_wrinkle_candidate_peak = 0.0
                self._anti_wrinkle_candidate_start_power = 0.0
            elif anti_wrinkle_active and self._state == STATE_ANTI_WRINKLE and is_high:
                if self._anti_wrinkle_candidate_start is None:
                    self._anti_wrinkle_candidate_start = timestamp
                    self._anti_wrinkle_candidate_peak = power
                    self._anti_wrinkle_candidate_start_power = power
                else:
                    self._anti_wrinkle_candidate_peak = max(
                        self._anti_wrinkle_candidate_peak, power
                    )

                candidate_duration = (
                    timestamp - self._anti_wrinkle_candidate_start
                ).total_seconds()
                # Register item 393a: after the #296 finalise the tail is a train of
                # sub-max-power drum bursts, and a send-on-change plug can skip the
                # baseline between two of them (the reporter's 20 Knitterschutz
                # tails all hold 60-111 s without a reading under stop_threshold),
                # so the configured burst length opened a second cycle on the tail
                # itself. The finalise accepted ANTI_CREASE_CONFIRM_WINDOW_S of such
                # readings as tail; leaving it on duration takes a burst longer than
                # that. A next wash's heating still leaves at once on the power test.
                burst_limit = float(self._config.anti_wrinkle_max_duration)
                if tail_floor is not None:
                    burst_limit = max(burst_limit, ANTI_CREASE_CONFIRM_WINDOW_S)
                exceeds = (
                    self._anti_wrinkle_candidate_peak
                    > self._config.anti_wrinkle_max_power
                    or power > self._config.anti_wrinkle_max_power
                    or candidate_duration > burst_limit
                )

                if exceeds:
                    candidate_start = self._anti_wrinkle_candidate_start
                    candidate_peak = self._anti_wrinkle_candidate_peak
                    candidate_start_power = self._anti_wrinkle_candidate_start_power
                    self._anti_wrinkle_candidate_start = None
                    self._anti_wrinkle_candidate_peak = 0.0
                    self._anti_wrinkle_candidate_start_power = 0.0
                    self._transition_to(STATE_STARTING, timestamp)
                    started_from_anti_wrinkle = True
                    self._current_cycle_start = candidate_start or timestamp

                    # Preserve the anti-wrinkle candidate window instead of dropping ramp-up samples.
                    if candidate_start and candidate_start < timestamp:
                        start_power = candidate_start_power if candidate_start_power > 0 else power
                        self._power_readings = [(candidate_start, start_power), (timestamp, power)]
                        interval_s = (timestamp - candidate_start).total_seconds()
                        avg_power = (start_power + power) / 2.0
                        self._energy_since_idle_wh = max(0.0, avg_power * (interval_s / 3600.0))
                    else:
                        self._power_readings = [(timestamp, power)]
                        # Guarded interval (#403): the gap between anti-wrinkle
                        # tumbles was spent at the previous (idle) level.
                        self._energy_since_idle_wh = high_step_wh

                    self._cycle_max_power = max(candidate_peak, power)
            elif self._state != STATE_ANTI_WRINKLE:
                self._anti_wrinkle_candidate_start = None
                self._anti_wrinkle_candidate_peak = 0.0
                self._anti_wrinkle_candidate_start_power = 0.0

            if self._state == STATE_ANTI_WRINKLE:
                # Track time in idle. Quiet is below the exit power OR the stop
                # threshold, whichever is higher, so the exit power only matters when
                # set above stop. Not a "true off" level below stop (#285 #296 #325):
                # after 33 of 139 corpus washer/dryer cycles the appliance idled
                # between the 0.8 W default and its stop threshold, which would hold
                # anti-wrinkle (and with it Clean and the unload reminder) for the
                # 2 h cap. The #296 tail's baseline above stop is the tail floor's job.
                effective_exit = max(self._config.anti_wrinkle_exit_power, self._config.stop_threshold_w)
                if power < effective_exit:
                    # Low-power gap invalidates any burst candidate collected while in anti-wrinkle.
                    self._anti_wrinkle_candidate_start = None
                    self._anti_wrinkle_candidate_peak = 0.0
                    self._anti_wrinkle_candidate_start_power = 0.0
                    self._anti_wrinkle_idle_time += dt
                    anti_wrinkle_end_threshold = max(
                        self._dynamic_end_threshold,
                        float(self._config.anti_wrinkle_idle_timeout),
                    )
                    if self._anti_wrinkle_idle_time >= anti_wrinkle_end_threshold:
                        self._transition_to(STATE_OFF, timestamp)
                        return
                else:
                    # Reset idle timer when power rises (burst detected)
                    self._anti_wrinkle_idle_time = 0.0

                # Exit conditions:
                # 1. Idle duration exceeded (handled above), OR
                # 2. Safety timeout (2 hours in anti-wrinkle), OR
                # 3. External trigger (user_stop, external triggers handled by manager)
                if (
                    self._state_enter_time
                    and (timestamp - self._state_enter_time).total_seconds() > 7200
                ):
                    # Safety timeout: 2 hours in anti-wrinkle
                    self._transition_to(STATE_OFF, timestamp)
                return

            # Delayed-start "standby band" detection (only from STATE_OFF).
            #
            # A machine in delayed-start mode sits in a power band between
            # the off-noise floor (stop_threshold_w) and the cycle-start
            # threshold (start_threshold_w) - display, electronics, the
            # occasional anti-damp tumble - for minutes to hours.  We
            # track anchored elapsed time while power is in that band; once
            # it crosses delay_confirm_seconds we transition to DELAY_WAIT.
            #
            # Brief high-power excursions (menu navigation, button presses)
            # don't break the candidate: they fall through to the normal
            # start logic below, and unless they sustain for
            # start_duration_threshold they get aborted as a false start
            # and we re-enter the band on the next reading.  Excursions
            # below stop_threshold_w (machine momentarily idle on the noise
            # floor) DO reset the candidate, because that's the same
            # signal we use to define "off".
            if (
                self._config.delay_detect_enabled
                and self._state == STATE_OFF
                and not started_from_anti_wrinkle
                and self._config.stop_threshold_w < self._config.start_threshold_w
            ):
                in_band = (
                    self._config.stop_threshold_w
                    <= power
                    < self._config.start_threshold_w
                )
                if in_band:
                    if self._delay_band_start is None:
                        self._delay_band_start = timestamp
                        self._delay_band_seconds = 0.0
                    else:
                        self._delay_band_seconds = (
                            timestamp - self._delay_band_start
                        ).total_seconds()
                    self._delay_band_peak = max(self._delay_band_peak, power)
                    if self._delay_band_seconds >= self._config.delay_confirm_seconds:
                        self._logger.info(
                            "Delayed start detected: standby band held for %.0fs "
                            "(peak %.1fW, current %.1fW) → DELAY_WAIT",
                            self._delay_band_seconds,
                            self._delay_band_peak,
                            power,
                        )
                        self._transition_to(STATE_DELAY_WAIT, timestamp)
                        return
                    # Stay in OFF while we accumulate evidence - do not
                    # fall through to the high-power start logic, the
                    # reading is below threshold by definition.
                    return
                elif power < self._config.stop_threshold_w:
                    # Machine genuinely idle: forget any band history.
                    self._delay_band_start = None
                    self._delay_band_seconds = 0.0
                    self._delay_band_peak = 0.0
                    self._preserve_delay_band_on_off = False
                # power >= start_threshold_w: fall through to the normal
                # start path below.  If it turns out to be a brief peak,
                # STATE_STARTING will abort it as a false start and we'll
                # re-enter the band check on the next sample without
                # losing accumulated time (we don't reset on a high
                # excursion - most users' "menu navigation" peaks last
                # less than a sample interval anyway).

            terminal = self._state in (
                STATE_FINISHED, STATE_INTERRUPTED, STATE_FORCE_STOPPED
            )
            if terminal and not is_high and power >= self._config.stop_threshold_w:
                # Item 510: a terminal state holds through the standby band instead
                # of probing on it, so the band reading marks the next probe as a
                # standby re-probe (item 501), as the band probe's abort used to.
                self._standby_reprobe = True

            if is_high and not started_from_anti_wrinkle:
                # Transition to STARTING
                self._preserve_delay_band_on_off = self._delay_band_start is not None
                self._hide_probe(
                    self._standby_reprobe and (self._state == STATE_OFF or terminal)
                )
                back = (
                    (self._state, self._state_enter_time, self._sub_state)
                    if terminal and TERMINAL_PROBE_RETURNS
                    else None
                )
                self._transition_to(STATE_STARTING, timestamp)
                self._probe_terminal_from = back  # item 515
                self._current_cycle_start = timestamp
                self._power_readings = [(timestamp, power)]
                # Seed from the guarded interval (#403), not raw dt: this seed
                # OVERWRITES the accumulator (it has to - entering STARTING from
                # a terminal state carries the previous cycle's total), so an
                # unguarded seed would reinstate the idle gap the accumulator
                # just declined to credit.
                self._energy_since_idle_wh = high_step_wh
                self._cycle_max_power = power
                self._apply_curve_preroll(timestamp, power)
                self._show_probe_with_evidence()
            # NOTE: terminal-state expiry (Finished/Interrupted/Force-Stopped -> Off)
            # is owned solely by the manager (WashDataManager._handle_state_expiry),
            # which has a wall-clock timer that also fires when a change-only power
            # sensor stops reporting, plus the opt-in power-based Off (issue #284).
            # The detector used to auto-expire here after a hardcoded 30 min, but that
            # duplicated the manager timer (a weaker, per-reading subset) and left the
            # manager's bookkeeping (progress, clean overlay, notifications) dangling.
            # ANTI_WRINKLE -> Off is handled by its own idle/timeout logic above.

        elif self._state == STATE_DELAY_WAIT:
            if power >= self._config.start_threshold_w:
                # Power is in cycle-start territory.  Require at least
                # two consecutive high readings spanning
                # start_duration_threshold real seconds before committing
                # to STARTING, so a single isolated spike (a heavy menu
                # interaction, an anti-damp pulse briefly crossing the
                # threshold) doesn't false-trigger.  We anchor on the
                # FIRST high reading instead of accumulating dt, because
                # dt to the previous (low) reading is unrelated to how
                # long the high power has actually persisted.
                self._delay_wait_true_off_seconds = 0.0
                if self._delay_wait_high_start is None:
                    self._delay_wait_high_start = timestamp
                    self._delay_wait_high_power = power
                    # Item 504: the streak's energy from here, credited per guarded
                    # interval by the accumulator above (#403), seeds the probe.
                    # The previous reading was not high, so nothing before the
                    # anchor is lost.
                    self._energy_since_idle_wh = 0.0
                else:
                    elapsed_high = (
                        timestamp - self._delay_wait_high_start
                    ).total_seconds()
                    if elapsed_high >= self._config.start_duration_threshold:
                        # Debug, like any other probe: since item 504 a straddling
                        # standby probes from here hundreds of times a day.
                        self._logger.debug(
                            "Delayed start: probing (power %.1fW sustained ≥ %.1fW for %.0fs)",
                            power,
                            self._config.start_threshold_w,
                            elapsed_high,
                        )
                        self._hide_probe(True)  # out of a standby (item 501)
                        wait_since = self._state_enter_time or timestamp
                        self._transition_to(STATE_STARTING, timestamp)
                        self._probe_wait_since = wait_since  # item 504
                        start_timestamp = self._delay_wait_high_start or timestamp
                        start_power = self._delay_wait_high_power or power
                        self._current_cycle_start = start_timestamp
                        self._power_readings = [(start_timestamp, start_power)]
                        # `_energy_since_idle_wh` already holds the streak's energy
                        # (item 504). It was `start_power * elapsed`, which credited
                        # the first high reading's level for the whole window: a
                        # 22 W blip then 5 W for 20 s banked 0.15 of a 0.2 Wh gate,
                        # and once false starts return here every standby probe
                        # started that way (max aborted probe 0.53 -> 0.78 of the
                        # gate on #35).
                        if timestamp != start_timestamp:
                            self._power_readings.append((timestamp, power))
                        self._cycle_max_power = max(start_power, power)
                        # #430: the buffer records in DELAY_WAIT too, so the
                        # readings between the anchor and this confirmation exist
                        # and would otherwise be dropped - the curve would span the
                        # confirmation window with two points. Never moves the
                        # anchor forward (see _apply_curve_preroll), and is a no-op
                        # while the option is off, which is the default.
                        self._apply_curve_preroll(timestamp, power)
                        self._show_probe_with_evidence()
            else:
                # Power dropped back below start threshold - clear the
                # high-power streak anchor so the next high reading
                # starts a fresh confirmation window.
                self._delay_wait_high_start = None
                self._delay_wait_high_power = None
                if power < self._config.stop_threshold_w:
                    # Power near zero: machine genuinely turned off, not
                    # just waiting.
                    self._delay_wait_true_off_seconds += dt
                    if self._delay_wait_true_off_seconds >= 30.0:
                        self._logger.info(
                            "Delayed start cancelled: power dropped to off (%.1fW) for %.0fs",
                            power,
                            self._delay_wait_true_off_seconds,
                        )
                        self._transition_to(STATE_OFF, timestamp)
                        return
                else:
                    self._delay_wait_true_off_seconds = 0.0

                # Safety timeout
                if (
                    self._state_enter_time
                    and (timestamp - self._state_enter_time).total_seconds()
                    >= self._config.delay_timeout_seconds
                ):
                    self._logger.info(
                        "Delayed start timeout after %.0fh → OFF",
                        self._config.delay_timeout_seconds / 3600.0,
                    )
                    self._transition_to(STATE_OFF, timestamp)

        elif self._state == STATE_STARTING:
            self._power_readings.append((timestamp, power))
            self._cycle_max_power = max(self._cycle_max_power, power)

            if is_high:
                # Power back up - clear any accumulated "true off" hold time.
                self._starting_paused_off_since = None

            self._show_probe_with_evidence()  # item 501

            if self._time_above_threshold >= self._config.start_duration_threshold:
                if self._energy_since_idle_wh >= self._config.start_energy_threshold:
                    self._transition_to(STATE_RUNNING, timestamp)

            # Abort if power drops below threshold before confirmation.
            # Skip the abort when the user has explicitly paused the cycle
            # (issue #306): a user pause sets verified_pause=True, which signals
            # that the low power is intentional, not a false start.
            if not is_high and self._time_below_threshold > 1.0:  # 1s grace period
                if getattr(self, "_verified_pause", False):
                    # User pause holds; wait for Resume Cycle (issue #306).  But a
                    # genuinely paused appliance keeps standby power above the stop
                    # threshold - sustained power *below* it means the machine was
                    # switched off, so fall back to OFF rather than pinning STARTING
                    # forever.
                    if power < self._config.stop_threshold_w:
                        # An outage-sized gap since the last reading is NOT observed
                        # quiet (the machine may still be paused, we just lost
                        # telemetry): reset anchor so only genuinely-sampled
                        # sustained-off time can cancel a paused STARTING.
                        if dt > self._outage_threshold_s():
                            self._starting_paused_off_since = None
                        elif self._starting_paused_off_since is None:
                            # First below-stop reading: anchor the timestamp.
                            # Don't credit the preceding dt interval — we only
                            # know the device is off *now*, not how long before
                            # this sample it went quiet.
                            self._starting_paused_off_since = timestamp
                        observed_off_s = (
                            (timestamp - self._starting_paused_off_since).total_seconds()
                            if self._starting_paused_off_since is not None
                            else 0.0
                        )
                        if observed_off_s >= STARTING_PAUSED_TRUE_OFF_TIMEOUT_SECONDS:
                            self._logger.info(
                                "Paused STARTING cancelled: power off (%.1fW) for "
                                "%.0fs → OFF",
                                power,
                                observed_off_s,
                            )
                            self._transition_to(STATE_OFF, timestamp)
                            return
                    else:
                        # Power recovered — clear the off anchor.
                        self._starting_paused_off_since = None
                else:
                    # False start
                    self._logger.debug(
                        "False start detected: power dropped after %.2fs",
                        self._time_above_threshold,
                    )
                    # Do NOT reset _delay_band_* here — _transition_to(STATE_OFF) will
                    # preserve the band via _preserve_delay_band_on_off if it was set
                    # at STARTING entry (line 838), so a brief high-power peak (menu
                    # navigation) doesn't restart the delayed-start accumulation from zero.
                    # Item 501: back in the band, not on the idle floor, so the next
                    # probe is a standby re-probe and is not shown until it has evidence.
                    self._standby_reprobe = power >= self._config.stop_threshold_w
                    wait_since = self._probe_wait_since
                    waited_s = (
                        (timestamp - wait_since).total_seconds()
                        if wait_since is not None
                        else 0.0
                    )
                    if self._probe_terminal_from is not None:
                        # Item 515: out of Finished / Interrupted / Force-Stopped, back
                        # there whatever the power: a probe that never committed ends
                        # nothing, and the manager's expiry still owns the way to OFF.
                        self._return_probe_to_terminal(timestamp, power)
                    elif (
                        wait_since is not None
                        and self._standby_reprobe
                        and waited_s < self._config.delay_timeout_seconds
                    ):
                        # Item 504: a probe out of DELAY_WAIT that falls back into the
                        # band goes back to waiting, keeping DELAY_WAIT's timeout
                        # anchor. Back to OFF lost it, and on a standby straddling
                        # start_threshold_w the band (OFF only) re-armed DELAY_WAIT a
                        # minute later: off <-> waiting ~16 times per idle day on
                        # #35. A drop below stop_threshold_w still ends the wait,
                        # and so does the timeout, checked here because on such a
                        # standby this abort may be the only in-band reading.
                        self._transition_to(STATE_DELAY_WAIT, timestamp)
                        self._state_enter_time = wait_since
                        self._time_in_state = max(0.0, waited_s)
                    else:
                        if wait_since is not None and self._standby_reprobe:
                            self._logger.info(
                                "Delayed start timeout after %.0fh → OFF",
                                self._config.delay_timeout_seconds / 3600.0,
                            )
                        self._transition_to(STATE_OFF, timestamp)

        elif self._state == STATE_RUNNING:
            self._power_readings.append((timestamp, power))
            self._cycle_max_power = max(self._cycle_max_power, power)
            self._update_stall(timestamp, power)  # #452, display + standby-band hold

            # Anti-crease finalize (#296): a matched cycle past its expected
            # duration that has settled into the low-power tumble tail is done -
            # finalize into anti-wrinkle now instead of letting the periodic
            # bursts keep reviving RUNNING until a second wash merges in.
            if self._maybe_finalize_anticrease_tail(timestamp):
                return

            # Use dynamic threshold
            thresh = self._dynamic_pause_threshold
            if self._time_below_threshold >= thresh:
                self._try_profile_match(timestamp, force=True)  # Refine match on pause
                self._transition_to(STATE_PAUSED, timestamp)

            # Periodic profile matching
            self._try_profile_match(timestamp)

            # Standby-band stuck finalize (#296): an appliance holding a flat
            # low standby draw ABOVE stop_threshold never accumulates
            # _time_below_threshold, so it never reaches PAUSED/ENDING.  Detect
            # the plateau and finalize as a normal completion (so anti-wrinkle
            # still engages).  Cheaply gated on being well past expected before
            # the window scan runs.
            if self._maybe_finalize_standby_band(timestamp, power):
                return

            # Max duration safety
            if (
                self._current_cycle_start
                and (timestamp - self._current_cycle_start).total_seconds() > 28800
            ):  # 8h safety
                self._finish_cycle(
                    timestamp,
                    status="force_stopped",
                    termination_reason=TerminationReason.FORCE_STOPPED,
                )

        elif self._state == STATE_PAUSED:
            self._power_readings.append((timestamp, power))
            self._update_stall(timestamp, power)  # #452

            # Anti-crease finalize (#296) - see the RUNNING branch.
            if self._maybe_finalize_anticrease_tail(timestamp):
                return

            if is_high:
                # Resume to RUNNING
                self._transition_to(STATE_RUNNING, timestamp)
            else:
                # Periodic profile matching during pause
                self._try_profile_match(timestamp)

                thresh = self._dynamic_end_threshold
                if self._time_below_threshold >= thresh:
                    self._transition_to(STATE_ENDING, timestamp)

        elif self._state == STATE_ENDING:
            self._power_readings.append((timestamp, power))
            self._update_stall(timestamp, power)  # #452

            # Hard cap: ENDING must not run longer than RUNNING's 8 h safety limit.
            # Without this a standby baseline can hold the state open indefinitely.
            if (
                self._current_cycle_start
                and (timestamp - self._current_cycle_start).total_seconds() > 28800
            ):
                self._finish_cycle(
                    timestamp,
                    status="force_stopped",
                    termination_reason=TerminationReason.FORCE_STOPPED,
                )
                return

            # Anti-crease finalize (#296) - see the RUNNING branch.  Fires ahead of
            # the is_high end-spike handling so a sub-max_power tail burst finalizes
            # into anti-wrinkle instead of reviving RUNNING.
            if self._maybe_finalize_anticrease_tail(timestamp):
                return

            if is_high:
                # Standby-band finalize (#296/#445) from ENDING too (DETECT-03). A
                # run that dipped long enough to reach ENDING and THEN settled on a
                # flat standby above stop_threshold keeps every reading "high":
                # after 120 s in state each one is kept below as a terminal spike
                # and returns, resetting the quiet timer, so neither the fallback
                # nor the RUNNING-only plateau check could ever end it - the 8 h
                # force-stop did. Same predicate and trim as RUNNING.
                if self._maybe_finalize_standby_band(timestamp, power):
                    return

                # Programme time: a stall is not progress (item 511).
                current_duration = self._gate_elapsed_s(timestamp)

                is_dishwasher = self._config.device_type == "dishwasher"

                # Issue #43: only treat this as a *terminal* end spike (which then
                # pre-arms Smart Termination) when it occurs near the end of the
                # expected cycle.  Mid-cycle spikes - e.g. the dishwasher
                # wash→drying drain wind-down at ~50% of expected duration - must
                # not arm smart termination, otherwise the cycle finishes at 99%
                # of expected *before* the real end-of-cycle pump-out, and that
                # pump-out is then misread as a brand-new cycle.  Without a
                # matched profile (expected==0) the gating is bypassed so the
                # legacy "any spike counts" behaviour is preserved for unmatched
                # cycles (relied on by the dishwasher unmatched-cap path).
                if (
                    self._expected_duration <= 0
                    or current_duration
                    >= self._expected_duration * DISHWASHER_END_SPIKE_MIN_PROGRESS
                ) and self._spike_follows_terminal_quiet(timestamp):
                    self._end_spike_seen = True
                    self._end_spike_duration = current_duration
                    self._logger.debug(
                        "End spike detected (power high in ENDING state, "
                        "%.0fs/%.0fs)",
                        current_duration,
                        self._expected_duration,
                    )
                else:
                    self._logger.debug(
                        "Mid-cycle spike in ENDING ignored for end-spike "
                        "tracking (%.0fs < %.0f%% of expected %.0fs)",
                        current_duration,
                        DISHWASHER_END_SPIKE_MIN_PROGRESS * 100,
                        self._expected_duration,
                    )

                # Sanity check: if expected_duration is unreasonable (>6 hours), use fallback
                max_reasonable = 21600.0  # 6 hours
                effective_expected = self._expected_duration

                if effective_expected <= 0 or effective_expected > max_reasonable:
                    # Fallback: use current duration + buffer if we've run > 3 hours
                    # (Assumes any cycle over 3 hours running is near completion when in ENDING)
                    if current_duration > 10800:  # 3 hours
                        effective_expected = current_duration * 0.99  # Always past threshold
                        self._logger.debug(
                            "End spike check using fallback: expected_duration=%ds is unreasonable, "
                            "using current_duration=%ds as reference",
                            int(self._expected_duration), int(current_duration)
                        )

                past_expected = (
                    effective_expected > 0
                    and current_duration >= (effective_expected * 0.98)
                )

                # If ENDING has already lasted long enough, treat any power burst as
                # terminal (applies to all device types). Dishwashers additionally check
                # proximity to the expected duration.
                long_ending_tail = self._time_in_state >= 120.0
                terminal_spike = long_ending_tail

                if is_dishwasher:
                    near_expected = (
                        effective_expected > 0
                        and current_duration >= (effective_expected * 0.90)
                    )
                    terminal_spike = near_expected or long_ending_tail

                if terminal_spike:
                    self._logger.debug(
                        "End spike kept in ENDING (duration %.0fs/%.0fs, time_in_ending %.0fs)",
                        current_duration,
                        effective_expected,
                        self._time_in_state,
                    )
                    return

                if past_expected:
                    self._logger.debug(
                        "End spike ignored for state transition (past expected duration %.0fs/%.0fs)",
                        current_duration, effective_expected
                    )
                    # Stay in ENDING, the spike is recorded but doesn't resume cycle
                else:
                    # Resume -> RUNNING (spike is genuine mid-cycle activity)
                    self._transition_to(STATE_RUNNING, timestamp)
            else:
                # Periodic profile matching during ending
                self._try_profile_match(timestamp)
                # A verified pause a revoke orphaned is released after a bounded
                # quiet, so the fallback below can end the cycle (item 498).
                self._release_orphaned_pause()

                # --- SMART TERMINATION CHECK ---
                # If we have a confident profile match and duration meets expectations,
                # we terminate early (after appropriate debounce), ignoring long arbitrary timeouts.
                if self._matched_profile:
                    start_time = self._current_cycle_start or timestamp
                    # Programme time (item 511): the halt itself carried the clock
                    # past the ratio, so the resumed wash ended at its next quiet.
                    current_duration = self._gate_elapsed_s(timestamp)

                    # --- ROBUSTNESS UPGRADE ---
                    # 1. Require higher duration ratio for Smart path
                    # 2. Require debounce to be measured FROM entry into ENDING state

                    # Per-appliance configurable gate (#393): the ratio is resolved
                    # in the config builder to the device-type default (0.99
                    # dishwasher / 0.98 other) unless the user tuned it, and the
                    # dishwasher pump-out relief is folded in via a pure helper so
                    # the gate logic stays unit-testable (see _resolve_smart_ratio).
                    smart_ratio = self._resolve_smart_ratio(
                        self._config.device_type,
                        self._config.smart_termination_duration_ratio,
                        getattr(self, "_end_spike_seen", False),
                        getattr(self, "_end_spike_duration", 0.0),
                        self._expected_duration,
                    )

                    is_confident_match = (
                        getattr(self, "_last_match_confidence", 0.0)
                        >= self._config.match_confidence_threshold
                    )
                    # Compute the #364 power-plausibility once per reading and reuse it
                    # for the diagnostic reason and the gate below - the helper walks the
                    # trailing window, so calling it two/three times per reading is waste.
                    _power_plausible = self._smart_term_power_plausible(timestamp)

                    # Gate the predictive end on match certainty.
                    # _match_ambiguous: top-1 vs top-2 score gap is too small to
                    # trust the matched profile's expected duration - fall through
                    # to the power-based fallback timeout instead. (The #364
                    # prefix-fit flag that also blocked here - "a longer candidate
                    # explains this trace better" - was removed in 0.5.8: since
                    # #400 the live matcher scores that prefix itself, and the flag
                    # never fired at a split moment on the corpus. See const.py.)
                    # Surface why the fast end-path is (not) firing, throttled to
                    # reason changes so a stuck cycle's cause is visible in the log
                    # without spamming every reading. Pure diagnostic (#346).
                    _block_reason = self._smart_term_block_reason(
                        current_duration,
                        self._expected_duration,
                        smart_ratio,
                        is_confident_match,
                        self._match_ambiguous,
                        _power_plausible,
                    )
                    if _block_reason != self._last_smart_term_block_reason:
                        self._last_smart_term_block_reason = _block_reason
                        if _block_reason is not None:
                            self._logger.debug(
                                "Smart Termination not applied (%s): dur=%.0fs/%.0fs conf=%.2f "
                                "ambiguous=%s trailing_power=%s profile_tail=%s",
                                _block_reason,
                                current_duration,
                                self._expected_duration * smart_ratio,
                                getattr(self, "_last_match_confidence", 0.0),
                                self._match_ambiguous,
                                self._trailing_mean_power(timestamp, self._tail_window_s()),
                                self._matched_tail_power,
                            )

                    if (
                        current_duration >= (self._expected_duration * smart_ratio)
                        and is_confident_match
                        and not self._match_ambiguous
                        # A user pause is authoritative: every other finisher honours
                        # it, and the paused time itself pushes elapsed past the
                        # ratio, so a washer paused near its end was finished by
                        # Smart Termination while the plug sat at 0 W (DETECT-05).
                        and not getattr(self, "_user_paused", False)
                        # #364: the clock says "done", but if we are still drawing
                        # several times what this profile draws at its own end, the
                        # match is a shorter look-alike and we are mid-wash. Block;
                        # the power-based fallback timeout decides instead.
                        and _power_plausible
                    ):
                        # Dynamic confirmation window
                        if self._config.device_type == "dishwasher":
                            # Fixed - NOT off_delay-derived.  off_delay is sized to
                            # bridge the long drying "pause", but must not delay the
                            # end; see DISHWASHER_SMART_TERMINATION_DEBOUNCE_SECONDS.
                            smart_debounce = DISHWASHER_SMART_TERMINATION_DEBOUNCE_SECONDS
                        elif self._config.device_type in (
                            DEVICE_TYPE_WASHING_MACHINE,
                            DEVICE_TYPE_WASHER_DRYER,
                        ):
                            # Washing machines and washer-dryers have soak and
                            # rinse gaps that can dip for several minutes between
                            # programme phases.  Require quiet time equal to half
                            # the soak-bridging min_off_gap before committing
                            # Smart Termination, so a near-duplicate profile
                            # doesn't cut a long cycle short during a mid-cycle
                            # power trough.  Bounded above so a large suggested /
                            # hand-set min_off_gap can't inflate the quiet-time
                            # requirement and starve end-detection (see
                            # WASHER_SMART_TERMINATION_DEBOUNCE_MAX_SECONDS).
                            smart_debounce = min(
                                WASHER_SMART_TERMINATION_DEBOUNCE_MAX_SECONDS,
                                max(180.0, self._config.min_off_gap * 0.5),
                            )
                        else:
                            smart_debounce = 120.0

                        # Quiet time, not time in ENDING, for everything but a
                        # dishwasher: a wash that resumes after > 120 s in ENDING
                        # stays in ENDING, so `_time_in_state` had the debounce
                        # pre-paid by washing time and Smart fired on the next dip,
                        # turning the final spin into a second cycle (DETECT-04;
                        # 0/295 corpus cycles move). A dishwasher keeps state time,
                        # or every pump-out would add up to 300 s.
                        if self._config.device_type == "dishwasher":
                            _smart_quiet = self._time_in_state
                        else:
                            _smart_quiet = min(
                                self._time_in_state, self._time_below_threshold
                            )
                        if _smart_quiet >= smart_debounce:
                            # --- END SPIKE WAIT PERIOD (Dishwashers) ---
                            # Dishwashers should see the real end-of-cycle
                            # pump-out (which arms _end_spike_seen via the 85%
                            # progress gate) before Smart Termination fires -
                            # otherwise the pump-out arrives AFTER the cycle
                            # has already closed and registers as a brand-new
                            # "ghost" cycle.  User reports (issue #43) showed
                            # the original 5-min past_wait_period escape hatch
                            # closing the cycle ~4 min before the real pump-out
                            # at ~99.5% of expected.  Widen the escape hatch
                            # substantially (DISHWASHER_END_SPIKE_WAIT_SECONDS,
                            # currently 30 min past expected) so it cannot
                            # short-circuit a pump-out that fires within a
                            # reasonable window around expected end, but still
                            # guarantees the cycle terminates eventually for
                            # dishwashers that have no pump-out at all.
                            end_spike_seen = getattr(self, "_end_spike_seen", False)
                            # Release the pump-out wait once EITHER the cycle has run
                            # DISHWASHER_END_SPIKE_WAIT_SECONDS past its expected
                            # duration OR it has already reached its expected duration
                            # AND power has since stayed sustained-quiet for
                            # DISHWASHER_END_SPIKE_QUIET_RELEASE_SECONDS.  The second arm
                            # closes cycles that finish shorter than the profile's
                            # (drifted-up) average and whose terminal pump-out lands
                            # *before* the drop into ENDING, so no in-ENDING end-spike
                            # ever arms - without it they hang to the fallback timeout
                            # (~30-44 min late) and their label can even drift to a longer
                            # near-duplicate profile.  It is gated on
                            # ``current_duration >= expected`` so it can NOT fire during a
                            # long passive-drying phase that precedes a genuinely-late
                            # pump-out (e.g. an ECO cycle quiet from 50%-99% of expected):
                            # while still short of expected the cycle keeps waiting, and a
                            # real pump-out at ~99% arms the end-spike first.  Takes the
                            # SOONER of the two anchors, so it can only ever shorten the
                            # wait, never extend it.
                            _spike_wait = self._dishwasher_end_spike_wait_s()
                            past_wait_period = current_duration >= (
                                self._expected_duration + _spike_wait
                            ) or (
                                current_duration >= self._expected_duration
                                and self._time_below_threshold_gapfree
                                >= self._dishwasher_quiet_release_s()
                            )
                            if (
                                self._config.device_type == "dishwasher"
                                and not end_spike_seen
                                and not past_wait_period
                            ):
                                self._logger.debug(
                                    "Waiting for end spike (duration %.0fs, "
                                    "expected %.0fs + %.0fs wait)",
                                    current_duration,
                                    self._expected_duration,
                                    _spike_wait,
                                )
                                return  # Don't finish yet, wait for spike

                            self._logger.info(
                                "Smart Termination: Profile '%s' match confirmed (duration %.0fs, "
                                "conf %.2f, spike_seen=%s), ending.",
                                self._matched_profile,
                                current_duration,
                                getattr(self, "_last_match_confidence", 0.0),
                                end_spike_seen,
                            )
                            # Keep tail when smart terminating (matches profile
                            # duration), but only as far as the program can
                            # actually reach - see _keep_tail_cap (#424).
                            self._finish_cycle(
                                timestamp,
                                status="completed",
                                termination_reason=TerminationReason.SMART,
                                keep_tail=True,
                                tail_cap=self._keep_tail_cap(start_time),
                            )
                            return

                    # --- DURATION-ANCHORED HARD FINALIZE (backstop) ---
                    # Separate safety net for a matched cycle whose Smart
                    # Termination is blocked (ambiguous / prefix-ambiguous match)
                    # and whose fallback energy gate is held open by a low standby
                    # baseline: without this it sits in ENDING until the 8 h cap /
                    # zombie-kill (#296/#311).  Fires only well past the expected
                    # duration AND after a long *continuous* sub-threshold span, so
                    # it can never truncate a longer program mismatched to a shorter
                    # profile (a real longer program has high-power phases that keep
                    # resetting the quiet timer) — asymmetric, shorten-only.  Not
                    # for user-paused cycles.
                    required_quiet = max(
                        self._ending_hard_finalize_quiet_s(),
                        float(max(self._config.off_delay, self._config.min_off_gap)),
                    )
                    # Require the required_quiet tail to be actually SAMPLED (no
                    # outage-sized gap): otherwise a telemetry outage that inflated
                    # _time_below_threshold could finalize an active cycle early.
                    # Walk in reverse so we can capture the boundary reading (the
                    # first reading outside the window) — an outage right before the
                    # window would be invisible if we only passed in-window timestamps.
                    quiet_ts: list[datetime] = []
                    _boundary_q: datetime | None = None
                    for _ts, _ in reversed(self._power_readings):
                        if (timestamp - _ts).total_seconds() <= required_quiet:
                            quiet_ts.append(_ts)
                        elif quiet_ts:
                            _boundary_q = _ts
                            break
                    if _boundary_q is not None:
                        quiet_ts.append(_boundary_q)
                    if (
                        self._expected_duration > 0
                        and current_duration
                        >= self._expected_duration * ENDING_HARD_FINALIZE_RATIO
                        and self._time_below_threshold >= required_quiet
                        and not self._verified_pause
                        and not self._window_has_outage_gap(quiet_ts)
                    ):
                        self._logger.info(
                            "Duration-anchored finalize: cycle in ENDING at %.0fs "
                            "(%.1fx expected %.0fs), quiet %.0fs - Smart Termination "
                            "was blocked (ambiguous=%s); finalizing.",
                            current_duration,
                            current_duration / self._expected_duration,
                            self._expected_duration,
                            self._time_below_threshold,
                            self._match_ambiguous,
                        )
                        self._finish_cycle(
                            timestamp,
                            status="completed",
                            termination_reason=TerminationReason.TIMEOUT,
                            keep_tail=False,
                        )
                        return

                # --- FALLBACK TIMEOUT CHECK ---
                # Rule: To separate cycles, we must wait at least min_off_gap.
                effective_off_delay = self._hazard_wait(
                    timestamp, max(self._config.off_delay, self._config.min_off_gap)
                )

                # Progress-aware shortening (register item 306). `min_off_gap` is
                # there to bridge mid-cycle soak periods; once the run is past the
                # matched programme's OWN expected length there is no soak left to
                # bridge, so continuing to wait out a blind per-device prior just
                # reports the end late.
                #
                # Measured on `devtools/end_gate_eval.py`, which IS committed.
                # **`cycle_data/` is not**, so every n below depends on the local
                # corpus and two different counts get quoted: cycles REPLAYED,
                # and the subset that reached an ENDING exit, which is the only
                # one that yields a lag and is what the harness table's `n`
                # column reports. Current corpus: 273 replayed, 263 measured.
                # Figures attributed below to "211/221" predate the #445 / #427 /
                # #424 reporter exports being added mid-PR-#448, when the same
                # corpus was 221 replayed / 211 measured. They are the same
                # harness over a smaller corpus, not a drift.
                #
                # Re-cut on the current corpus (`--no-shortening` vs shipped,
                # paired over 273 cycles): median end lag 13.43 -> 12.00 min
                # overall and 27.50 -> 26.00 for washing machines, dishwasher p90
                # 33.50 -> 19.70, and early ends (1.14% / 0.00%) and splits
                # (2.66%) **identical in every scope**. It moves 32 of 273
                # cycles: it buys little because it reaches few, and it costs
                # nothing. Item 306's original figures (427 cycles, washing
                # machines 12.9 -> 7.5 min, splits 3.75%) came from a harness
                # that was never committed and **do not reproduce**; the safety
                # half reproduces exactly. Treat the 427-cycle numbers as
                # unverified; register item 329 holds that reconciliation, taken
                # on the 211-measured corpus. Re-cut anything new with
                # `end_gate_eval.py` rather than a throwaway script.
                #
                # Asymmetric and bounded, in the same spirit as _keep_tail_cap: it
                # can only ever shorten, keeps the user's explicit `off_delay` as
                # the floor (only the blind prior shrinks), and is inert when
                # nothing matched. The bar it waits for is DEVICE-RESOLVED, not a
                # fixed 1.05x (register item 355): `resolve_end_gate_late_ratio`
                # returns 0.90 for `washing_machine` / `washer_dryer` and
                # `END_GATE_LATE_RATIO` (1.05) for everything else - so on a washer
                # the shortening starts BEFORE the expected end. That is the point:
                # a washer's programme is load-adaptive, so a run sits below its
                # profile mean about half the time by definition and the median
                # washer reaches only 0.83 of a 1.05 bar, which put the rule out of
                # reach for the device type that needed it most. Early ends stayed
                # at 0.00% for washers at every ratio measured; the dishwashers,
                # which do produce early ends below 1.0, keep 1.05.
                # Gated on the SAME guard Smart Termination respects. The rule
                # keys on `_expected_duration`, so it must not fire while the
                # matcher says that duration is in doubt: an ambiguous match may
                # mean a much longer look-alike is still plausible, and then "past
                # the expected end" may really be "mid-soak in a longer programme".
                # Without this the fallback timeout walks straight through and
                # re-opens the #288 split-cycle bug - caught by
                # test_smart_termination_blocked_by_ambiguous_match, where a 450 s
                # soak dip sits right at the short profile's end.
                # NOT also gated on `_last_match_confidence >=
                # match_confidence_threshold`, and that is deliberate, not an
                # oversight - the paragraph above says "the SAME guard Smart
                # Termination respects" and means the ambiguity flag.
                # Smart Termination does check confidence, because it ENDS a cycle
                # early on a prediction; this rule only shortens a wait that is
                # already past the programme's own expected end, so the asymmetry
                # is intended. Adding the check was tried (PR #448 round 6) and
                # measured on `devtools/end_gate_eval.py` over 221 replayed real
                # cycles (211 of them measurable, the pre-reporter-export corpus
                # described above): it moves **2 of them**, delaying one by 10.5
                # min and one by 21 min, while early ends (1.42% / 0.00% at the >1
                # and >5 min marks) and splits (3.32%) stay **exactly** where they
                # were. It
                # prevented no split and no early end - pure cost, so it was
                # reverted. The corpus carries only 4 matched cycles under 0.4
                # confidence, so it cannot prove the guard harmless either; the
                # exposure is real (cycle `7c4598310016` is a 3h37m wash matched at
                # 0.372 to a profile named "1:07", i.e. running the shortened gate
                # for over two hours) and it still did not split. Re-run the
                # harness before re-litigating this.
                if (
                    self._matched_profile
                    and self._expected_duration > 0
                    and self._current_cycle_start is not None
                ):
                    _elapsed = self._gate_elapsed_s(timestamp)  # item 511
                    # The bar this run has to clear. Normally the matched
                    # programme's own expected end; while the matcher still thinks
                    # a materially LONGER programme is plausible, that longer one's
                    # end instead (register item 330).
                    #
                    # Blocking outright on the ambiguity flags - which is what
                    # this did until item 330 - was costing almost every cycle the
                    # shortening. Measured on `devtools/end_gate_eval.py`: of the
                    # 94 cycles that ever pass 1.05x their own expected duration,
                    # **84 (89%) were blocked by an ambiguity flag**, so the rule
                    # reached 4.7% of cycles (10 of the 211 measurable on the
                    # pre-reporter-export corpus) and the median cycle still
                    # waited out the full `min_off_gap`.
                    #
                    # The flags were not wrong, they were too coarse. They exist to
                    # protect `_expected_duration` against "this is really a prefix
                    # of something longer" (#288 / #364) - a statement about
                    # DURATION, not about which label wins. (Only `_match_ambiguous`
                    # is left: the #364 prefix-fit flag was removed in 0.5.8.)
                    # Two programmes that score within the ambiguity margin and
                    # run the same length leave "past the expected end" true
                    # either way. So instead of
                    # refusing, raise the bar to the longest duration still in
                    # play: past THAT, no candidate is left for this to be a
                    # mid-soak of, which is exactly the condition the guard was
                    # standing in for.
                    #
                    # Absent information keeps the OLD refusal. A caller that does
                    # not send element 12 (an older Playground, most tests, any
                    # short tuple) leaves `_longest_candidate_duration` at 0.0,
                    # and an ambiguous match with no candidate durations to
                    # compare must block exactly as it did before - otherwise the
                    # #288 split-cycle reproduction
                    # `test_smart_termination_blocked_by_ambiguous_match` walks
                    # straight through, which is how the first draft of this was
                    # caught.
                    #
                    # The bar is `_fallback_shortening_bar` (one implementation
                    # with the item-469b ENDING hold): None, i.e. blocked, for an
                    # ambiguous match with no candidate durations. The bound
                    # itself is derived from the FULL pre-collapse candidate
                    # population, so a longer candidate ranked sixth or lower
                    # cannot hide from the bar - which it could when this read
                    # `MatchResult.candidates`, i.e. `candidates[:5]`.
                    _bar_s = self._fallback_shortening_bar(
                        self._expected_duration,
                        self._match_ambiguous,
                        self._longest_candidate_duration,
                    )
                    # Device-resolved (register item 355): 1.05 is out of reach
                    # for a load-adaptive washer, which reaches a median 0.83 of
                    # its EXPECTED duration before it stops.
                    #
                    # **But never against a RAISED bar.** When the match is
                    # ambiguous and a longer candidate exists, `_bar` is no longer
                    # the expected duration - it is the longest plausible
                    # programme, and it was raised precisely to say "a much longer
                    # look-alike is still on the table, so past the expected end
                    # does not mean done". Discounting that by 0.90 would shorten
                    # the wait 10% BEFORE the candidate it represents could even
                    # finish, and on the shipped washer defaults that drops the
                    # wait from `min_off_gap` to `max(off_delay, 300)` - one quiet
                    # interval away from finalising mid-programme and recording the
                    # rest as a second cycle, which is #288. The item-355 measurement
                    # was taken against the expected duration and says nothing about
                    # this case, so a raised bar keeps the original 1.05.
                    if _bar_s is not None and _elapsed >= _bar_s:
                        effective_off_delay = max(
                            self._config.off_delay,
                            min(self._config.min_off_gap, END_GATE_LATE_SECONDS),
                        )

                # Energy gate always looks back off_delay seconds by default;
                # overridden below for the dishwasher cap case so the window
                # is consistent with the shortened effective_off_delay.
                gate_window = self._config.off_delay

                # Dishwasher-specific: after a terminal end spike (pump-out), an
                # unmatched cycle doesn't need to wait the full min_off_gap (up to
                # 9000s) before closing. Cap at 30 min so cycle 3 ends cleanly
                # ~30 min after the pump-out rather than sitting open for hours.
                if (
                    self._config.device_type == "dishwasher"
                    and not self._matched_profile
                    and self._end_spike_seen
                ):
                    effective_off_delay = min(effective_off_delay, 1800)
                    gate_window = effective_off_delay

                # Opt-in terminal-drop fast finalize (asymmetric, shorten-only):
                # a hard cliff-to-~0 sustained for TERMINAL_DROP_OFF_DELAY_SECONDS
                # that began earlier than this device has ever legitimately gone
                # quiet is almost certainly a real stop (plug pulled / cancelled),
                # not a soak.  Finalize now instead of waiting out the full
                # soak-bridging min_off_gap.  Only consulted when there is a longer
                # wait to shorten and the provider is wired (ML/anomaly opt-in);
                # the energy/defer gates are bypassed because the sustained sub-
                # threshold span already proves the appliance is off, and the
                # anomaly check has ruled out a legitimate early pause.
                if (
                    self._terminal_drop_provider is not None
                    and not self._verified_pause
                    and effective_off_delay > TERMINAL_DROP_OFF_DELAY_SECONDS
                    and self._time_below_threshold >= TERMINAL_DROP_OFF_DELAY_SECONDS
                    and self._is_terminal_drop()
                ):
                    start_time = self._current_cycle_start or timestamp
                    current_duration = (timestamp - start_time).total_seconds()
                    self._logger.info(
                        "Terminal drop: anomalously-early power cliff after %.0fs "
                        "(device never quiet this early) - finalizing without the "
                        "full %.0fs soak wait.",
                        current_duration,
                        effective_off_delay,
                    )
                    self._finish_cycle(
                        timestamp,
                        status="interrupted",
                        termination_reason=TerminationReason.TERMINAL_DROP,
                        keep_tail=False,
                    )
                    return

                if self._time_below_threshold >= effective_off_delay:

                    # Walk from the tail — readings are chronological so we can
                    # break as soon as we exceed the gate window (O(window) not O(n)).
                    recent_window = []
                    for r in reversed(self._power_readings):
                        if (timestamp - r[0]).total_seconds() <= gate_window:
                            recent_window.append(r)
                        else:
                            break
                    recent_window.reverse()

                    if not recent_window:
                        # Check deferred finish for matched profiles
                        start_time = self._current_cycle_start or timestamp
                        current_duration = self._gate_elapsed_s(timestamp)  # item 511

                        if self._should_defer_finish(current_duration):
                            return

                        # For dishwashers, use the timeout timestamp as end_time
                        # (keep_tail=True) so that the stored cycle duration includes
                        # the passive drying phase.  Without this, end_time snaps back
                        # to _last_active_time which may be set by a terminal drain
                        # spike mid-ENDING, producing a falsely short cycle duration.
                        keep_tail = self._config.device_type == "dishwasher"
                        self._finish_cycle(
                            timestamp,
                            status="completed",
                            keep_tail=keep_tail,
                            tail_cap=self._keep_tail_cap(start_time),
                        )
                        return

                    # Compute energy in recent window
                    recent_ts = np.array([r[0].timestamp() for r in recent_window])
                    recent_p = np.array([r[1] for r in recent_window])
                    max_gap_s = energy_gap_threshold_s(recent_ts)
                    recent_e = integrate_wh(recent_ts, recent_p, max_gap_s=max_gap_s)

                    energy_ok = recent_e <= self.config.end_energy_threshold
                    if not energy_ok and self._flat_standby_tail(timestamp):
                        # DETECT-08: the window's energy is a flat standby, not
                        # activity. Never before the full un-shortened wait, so a
                        # flat sub-stop phase the item-306 / hazard shortening
                        # would otherwise cut is still bridged by min_off_gap;
                        # far under stop, not before 2 h (see the constants).
                        self._logger.debug(
                            "Energy gate skipped: %.4fWh is a flat standby "
                            "after %.0fs quiet",
                            recent_e,
                            self._time_below_threshold,
                        )
                        energy_ok = True
                    if energy_ok:
                        start_time = self._current_cycle_start or timestamp
                        current_duration = self._gate_elapsed_s(timestamp)  # item 511

                        if self._should_defer_finish(current_duration):
                            return

                        keep_tail = self._config.device_type == "dishwasher"
                        self._finish_cycle(
                            timestamp,
                            status="completed",
                            keep_tail=keep_tail,
                            tail_cap=self._keep_tail_cap(start_time),
                        )
                    else:

                        self._logger.debug(
                            "Cycle ending prevented by energy gate: %.4fWh > %.4fWh",
                            recent_e,
                            self._config.end_energy_threshold,
                        )

    def _record_preroll(self, power: float, timestamp: datetime) -> None:
        """Buffer a reading seen before a cycle commits (#430).

        Only while no cycle is open - once RUNNING, ``_power_readings`` is the
        curve and this buffer would just duplicate it. STARTING counts as "not
        open": a probe in STARTING may still abort, and those are precisely the
        readings worth keeping.

        Bounded by ``curve_preroll_seconds`` (itself capped at
        ``CURVE_PREROLL_MAX_SECONDS``), so the buffer holds seconds of data, not
        an unbounded history.
        """
        window = effective_curve_preroll_seconds(self._config.curve_preroll_seconds)
        if window <= 0:
            # Option off: keep the buffer empty rather than paying to fill one
            # nothing will read, and so that enabling it mid-run cannot splice in
            # readings from before the option was turned on.
            if self._preroll_buffer:
                self._preroll_buffer = []
            return
        if self._state not in (
            STATE_OFF,
            STATE_STARTING,
            STATE_DELAY_WAIT,
            STATE_UNKNOWN,
        ) and not (
            # Item 515: a false start out of a terminal state returns there, not to
            # OFF; it keeps recording as OFF did, so the next probe can still chain
            # it. The buffer is empty there until a probe (the cycle end reset it).
            self._preroll_buffer
            and self._state in (STATE_FINISHED, STATE_INTERRUPTED, STATE_FORCE_STOPPED)
        ):
            return
        self._preroll_buffer.append((timestamp, float(power)))
        cutoff = timestamp - timedelta(seconds=window)
        # Readings arrive in order, so the stale prefix is contiguous.
        drop = 0
        for ts, _p in self._preroll_buffer:
            if ts < cutoff:
                drop += 1
            else:
                break
        if drop:
            del self._preroll_buffer[:drop]

    def _preroll_for_commit(
        self, timestamp: datetime, power: float
    ) -> list[tuple[datetime, float]]:
        """Readings to prepend to a cycle committing at ``timestamp`` (#430).

        Walks the buffer backwards from the commit and stops at the first quiet
        gap longer than ``PREROLL_CHAIN_BREAK_SECONDS`` - "the same start, probed
        twice" rather than "an unrelated blip earlier". Then anchors on the
        EARLIEST reading in that chain that is at or above ``start_threshold_w``:
        anchoring on the window edge instead would drag standby into the curve
        and move the cycle start to a moment the appliance was not yet doing
        anything.

        Returns [] whenever there is nothing to add, so the caller's fast path is
        a single emptiness test.
        """
        window = effective_curve_preroll_seconds(self._config.curve_preroll_seconds)
        if window <= 0 or not self._preroll_buffer:
            return []

        # Everything strictly before this commit, most recent first.
        prior = [(ts, p) for ts, p in self._preroll_buffer if ts < timestamp]
        if not prior:
            return []

        chain: list[tuple[datetime, float]] = []
        next_ts = timestamp
        for ts, p in reversed(prior):
            if (next_ts - ts).total_seconds() > PREROLL_CHAIN_BREAK_SECONDS:
                break
            chain.append((ts, p))
            next_ts = ts
        if not chain:
            return []
        chain.reverse()  # chronological

        threshold = float(self._config.start_threshold_w)
        anchor = next(
            (i for i, (_ts, p) in enumerate(chain) if p >= threshold), None
        )
        if anchor is None:
            return []  # the chain is all standby - nothing of this cycle in it
        return chain[anchor:]

    def _apply_curve_preroll(self, timestamp: datetime, power: float) -> None:
        """Prepend buffered pre-commit readings to the freshly-started cycle (#430).

        Moves ``_current_cycle_start`` back with them, and that is not optional:
        the stored duration is ``end_time - _current_cycle_start`` while matching
        resamples ``_power_readings``, so a curve that started earlier than the
        pointer would describe a different run from the one whose duration is
        recorded. The two existing back-anchors (the anti-wrinkle candidate
        window and the DELAY_WAIT high-start anchor) move the pointer for exactly
        the same reason.

        **Record-only: this must never make a cycle easier to START.**
        ``_energy_since_idle_wh`` is deliberately left alone. Despite the name it
        is not the cycle's energy - the stored figure is integrated from
        ``power_data`` at persistence, so it picks the pre-roll up for free - it
        is the accumulator the STARTING -> RUNNING gate reads
        (``>= start_energy_threshold``). Feeding it the pre-roll would let an
        aborted probe's energy be re-spent on the next probe's start gate, so two
        blips that each failed the gate could together pass it: exactly the
        phantom cycle #403 was fixed to prevent. The same argument covers
        ``_time_above_threshold``, which is likewise untouched.

        ``_cycle_max_power`` IS updated, because that is a property of the run
        being recorded (it gates the anti-crease path), not of admitting it.
        """
        preroll = self._preroll_for_commit(timestamp, power)
        if not preroll:
            return

        start_ts = preroll[0][0]
        # Callers do not all commit with the pointer at ``timestamp``. The
        # DELAY_WAIT confirmation has already back-anchored it to its first
        # sustained-high reading, and a pre-roll window shorter than
        # ``start_duration_threshold`` would otherwise move that pointer FORWARD
        # and shorten the delayed start. So keep whatever the caller anchored
        # before the chain, and only ever move the pointer earlier.
        earlier = [(ts, p) for ts, p in self._power_readings if ts < start_ts]
        self._power_readings = [*earlier, *preroll, (timestamp, power)]
        self._current_cycle_start = min(
            start_ts, self._current_cycle_start or start_ts
        )
        self._cycle_max_power = max(p for _ts, p in self._power_readings)
        self._logger.debug(
            "Curve pre-roll: carried %d reading(s) covering %.0fs from aborted "
            "start probe(s) into this cycle (start moved back to %s).",
            len(preroll),
            (timestamp - start_ts).total_seconds(),
            start_ts.isoformat(),
        )

    def _transition_to(self, new_state: str, timestamp: datetime) -> None:
        """Handle state transitions."""
        if self._state == new_state:
            return

        old_state = self._state
        self._state = new_state
        self._state_enter_time = timestamp
        self._time_in_state = 0.0
        self._sub_state = new_state.capitalize()  # Default substate

        # Bound each ENDING episode's ML-guard deferral independently: clear the
        # tracker whenever we are not in ENDING (e.g. on resume back to RUNNING).
        if new_state != STATE_ENDING:
            self._ml_defer_start_duration = None

        # Item 501: hiding is per probe, and a re-probe only follows a false start
        # straight back in OFF (any other state means the standby run is over).
        if new_state != STATE_STARTING:
            self._probe_hidden = False
            self._probe_wait_since = None  # item 504: per probe, like the hiding
            self._probe_terminal_from = None  # item 515, likewise
        if new_state not in (STATE_OFF, STATE_STARTING):
            self._standby_reprobe = False
        if new_state not in (STATE_RUNNING, STATE_PAUSED, STATE_ENDING):
            self._clear_stall()  # #452: a stall belongs to one open cycle
            self._stall_spans = []  # item 511
            self._user_pause_spans = []  # item 514
            self._user_pause_since = None

        # Reset energy accumulator on transition to OFF
        if new_state == STATE_OFF:
            self._energy_since_idle_wh = 0.0
            # Also reset idle time tracker when leaving ANTI_WRINKLE
            self._anti_wrinkle_idle_time = 0.0
            if not self._preserve_delay_band_on_off:
                self._delay_band_start = None
                self._delay_band_seconds = 0.0
                self._delay_band_peak = 0.0
            self._delay_wait_true_off_seconds = 0.0
            self._delay_wait_high_start = None
            self._delay_wait_high_power = None
            self._preserve_delay_band_on_off = False
            # Clear the paused-STARTING true-off accumulator so a later STARTING
            # cycle cannot inherit stale hold time and finalize to OFF prematurely
            # (this path is also reached via the paused-STARTING cancellation).
            self._starting_paused_off_since = None

        # Reset end spike tracker when entering ENDING state
        if new_state == STATE_ENDING:
            self._end_spike_seen = False
            self._end_spike_duration = 0.0
        elif new_state == STATE_DELAY_WAIT:
            # Band-accumulation tracker already played its role getting us
            # here; reset it so a future OFF→band cycle starts fresh.
            self._delay_band_start = None
            self._delay_band_seconds = 0.0
            self._delay_band_peak = 0.0
            self._delay_wait_true_off_seconds = 0.0
            self._delay_wait_high_start = None
            self._delay_wait_high_power = None
            self._sub_state = "Waiting to Start"
            self._preserve_delay_band_on_off = False
            # Like OFF: a false start returning here (item 504) leaves no energy.
            self._energy_since_idle_wh = 0.0
        elif new_state == STATE_ANTI_WRINKLE:
            self._anti_wrinkle_candidate_start = None
            self._anti_wrinkle_candidate_peak = 0.0
            self._anti_wrinkle_candidate_start_power = 0.0
            self._anti_wrinkle_idle_time = 0.0  # Reset idle time when entering ANTI_WRINKLE
            self._sub_state = "Anti-Wrinkle"
        elif new_state == STATE_STARTING:
            # Reset idle time if exiting ANTI_WRINKLE to STARTING (high-power burst resumed)
            self._anti_wrinkle_idle_time = 0.0
            # Fresh STARTING cycle: never inherit a prior cycle's true-off hold.
            self._starting_paused_off_since = None
        elif new_state == STATE_RUNNING:
            self._delay_band_start = None
            self._delay_band_seconds = 0.0
            self._delay_band_peak = 0.0
            self._preserve_delay_band_on_off = False

        self._logger.debug("Transition: %s -> %s at %s", old_state, new_state, timestamp)
        self._on_state_change(old_state, new_state)

    def _ml_end_confidence(self) -> float | None:
        """P(the current low-power event is the true end) from the opt-in ML guard.

        Builds the offset-second trace from the current cycle's readings and asks
        the injected provider. Returns None when there is no provider, no cycle
        start, or the provider declines (ML off / unmatched / model unavailable),
        so the caller keeps the existing power/energy-based behavior.
        """
        provider = self._end_confidence_provider
        start = self._current_cycle_start
        if provider is None or start is None or not self._power_readings:
            return None
        # Throttle: reuse the last result within the recompute window, but only when
        # it was computed for THIS cycle and the same expected_duration (which can
        # change under overrun) — otherwise recompute.
        now_ts = self._power_readings[-1][0]
        exp = float(self._expected_duration)
        cache = self._ml_end_cache
        if (
            cache is not None
            and cache[1] == exp
            and cache[2] == start
            and (now_ts - cache[0]).total_seconds() < ML_PROVIDER_THROTTLE_SECONDS
        ):
            return cache[3]
        points = [
            ((ts - start).total_seconds(), float(power))
            for ts, power in self._power_readings
        ]
        try:
            result = provider(points, exp)
        except Exception:  # noqa: BLE001 - ML must never break detection
            result = None
        self._ml_end_cache = (now_ts, exp, start, result)
        return result

    def _flat_standby_tail(self, timestamp: datetime) -> bool:
        """Whether the ENDING quiet is a flat standby the energy gate must not pin.

        DETECT-08. A sampled (no outage hole) window of the last
        max(off_delay, STANDBY_BAND_WINDOW_S) seconds whose readings are all above
        0 W and within max(ENDING_FLAT_STANDBY_SPREAD_FLOOR_W,
        ENDING_FLAT_STANDBY_SPREAD_FRAC x the highest) of each other, after
        max(off_delay, min_off_gap) of quiet when every reading is at least
        ENDING_FLAT_STANDBY_NEAR_STOP_FRAC x stop_threshold_w, else after
        ENDING_FLAT_STANDBY_LOOSE_QUIET_S. A window reaching back past the start
        of the quiet holds an above-stop reading and so is not flat.
        """
        quiet = self._time_below_threshold
        base_wait = float(max(self._config.off_delay, self._config.min_off_gap))
        if quiet < base_wait:
            return False
        span = max(float(self._config.off_delay), STANDBY_BAND_WINDOW_S)
        powers: list[float] = []
        window_ts: list[datetime] = []
        boundary: datetime | None = None
        for ts, p in reversed(self._power_readings):
            if (timestamp - ts).total_seconds() <= span:
                powers.append(float(p))
                window_ts.append(ts)
            else:
                boundary = ts
                break
        if boundary is None or len(powers) < ENDING_FLAT_STANDBY_MIN_READINGS:
            return False
        lo, hi = min(powers), max(powers)
        if lo <= 0.0:
            return False
        if (hi - lo) > max(
            ENDING_FLAT_STANDBY_SPREAD_FLOOR_W, ENDING_FLAT_STANDBY_SPREAD_FRAC * hi
        ):
            return False
        near_stop = lo >= ENDING_FLAT_STANDBY_NEAR_STOP_FRAC * float(
            self._config.stop_threshold_w
        )
        if not near_stop and quiet < max(base_wait, ENDING_FLAT_STANDBY_LOOSE_QUIET_S):
            return False
        return not self._window_has_outage_gap([boundary, *window_ts])

    def _window_has_outage_gap(self, window_ts: list[datetime]) -> bool:
        """Whether a 'sustained window' contains a data-outage-sized hole.

        The span + coverage checks in the standby / anti-crease window scans accept
        e.g. three readings spanning the window even if a long unobserved gap sits
        between them (a sensor dropout, or a sparse burst next to one old reading).
        Finalizing on such a window could wrongly cut an active cycle, so reject it.
        The gap ceiling is the sensor's own data-driven outage threshold
        (``energy_gap_threshold_s`` over the full trace), so a change-only sensor's
        legitimately-sparse stable stretches (tens of seconds between reports) are
        NOT rejected while a genuine dropout is.
        """
        if len(window_ts) < 2:
            return True  # too few points to trust as a sustained window
        max_gap = self._outage_threshold_s()
        ordered = sorted(t.timestamp() for t in window_ts)
        return any((b - a) > max_gap for a, b in itertools.pairwise(ordered))

    def _outage_threshold_s(self) -> float:
        """Sensor-adaptive gap ceiling (seconds): intervals longer than this are
        treated as telemetry outages, not observed quiet. Data-driven from the
        trace's own cadence (`energy_gap_threshold_s`), so a change-only sensor's
        sparse-but-real stable stretches are not mistaken for a dropout.
        """
        all_ts = np.array(
            [r[0].timestamp() for r in self._power_readings], dtype=float
        )
        return energy_gap_threshold_s(all_ts)

    def _standby_band_plateau(self, timestamp: datetime) -> float | None:
        """The stuck plateau's highest reading, or None when the cycle is not stuck.

        Whether a RUNNING cycle is stuck on a flat standby plateau (#296).

        Returns True only when ALL of the following hold, so this can never end
        an active low-power phase:

        * the device is a wet appliance where a stuck baseline is unambiguously
          anomalous (``STANDBY_BAND_FINALIZE_DEVICE_TYPES``);
        * a profile is matched and elapsed >= ``STANDBY_BAND_MIN_RATIO`` x the
          expected duration (well past when it should have ended);
        * not user-paused;
        * the most recent >= ``STANDBY_BAND_WINDOW_S`` of readings are ALL at or
          below ``STANDBY_BAND_MAX_FRACTION`` of the cycle's own peak power AND
          span no more than ``STANDBY_BAND_FLATNESS_FRACTION`` of the peak (a flat
          plateau, not fluctuating activity).

        The expensive window scan runs only after the cheap duration gate passes,
        so normal cycles never pay for it.
        """
        if self._config.device_type not in STANDBY_BAND_FINALIZE_DEVICE_TYPES:
            return None
        if getattr(self, "_verified_pause", False):
            return None
        if not (self._matched_profile and self._expected_duration > 0):
            return None
        start = self._current_cycle_start
        if start is None:
            return None
        current_duration = self._gate_elapsed_s(timestamp)  # item 511
        if current_duration < self._expected_duration * STANDBY_BAND_MIN_RATIO:
            return None
        # #399 interaction, load-bearing since the gate above dropped from 2.0x to
        # 1.0x expected (#445): a washer can sit quiet below anti_wrinkle_max_power
        # for minutes BEFORE its final spin, and that quiet is a flat sub-10%-of-peak
        # plateau like any other. Finalising there is exactly the failure #399 fixed
        # - the spin then arrives and opens a second cycle record. Defer while the
        # matched profile still owes this run its terminal high-power block. Shares
        # the predicate with the anti-crease finalise so the two release together,
        # and it fails open on every missing input (no profile block, non-terminal
        # block, past the ANTI_CREASE_SPIN_WAIT_MAX_RATIO cap), so an appliance that
        # never spins - the #445 Miele, which has no terminal block at all - is not
        # delayed by it.
        #
        # **This used to be inert on the devices it exists for (register item
        # 351).** `_anticrease_spin_pending` needs element 10, and the manager
        # supplied it only when `anti_wrinkle_enabled` was true -
        # `DEFAULT_ANTI_WRINKLE_ENABLED` is False, so on a washer the predicate
        # returned False immediately and none of the above happened. Measured on
        # the 273-cycle replay corpus: the band fired 14 times, 11 with the guard
        # inert, and 6 of those 11 had a reading above `min_power` still ahead,
        # i.e. would split. `terminal_high_for_guards` (module level in this file,
        # shared by the manager's live match tuple and the Playground's sim tuple
        # since round 31 - it used to be two hand-copies) now arms it for
        # `STANDBY_BAND_FINALIZE_DEVICE_TYPES` against a share of the cycle's own
        # peak - the same `STANDBY_BAND_MAX_FRACTION` used below, so the rule is
        # "wait while the profile still owes a block above the plateau you are
        # sitting on" and there is no new tunable. The bar travels WITH the block
        # as element 4, because `_high_power_seconds_since` has to count live
        # seconds above the same number.
        #
        # Measured end to end on `devtools/end_gate_eval.py`: washing-machine
        # splits 4.49% -> 1.90%, median end lag 26.00 -> 24.74 min (it does not
        # cost time - a cycle that used to split now finishes once), match rate
        # 89.7% -> 92.4%, early ends unchanged at 0.00%, dishwashers identical.
        # A ceiling of 0.15 caught the 6th split too but deferred 10 of the 11
        # firings, and a deferral with no spin ahead waits out
        # ANTI_CREASE_SPIN_WAIT_MAX_RATIO (1.25x expected, ~32 min on a 2:09
        # wash), so it bought the last split for three long waits. 0.10 was the
        # maintainer's call.
        if self._anticrease_spin_pending(timestamp):
            return None
        peak = float(self._cycle_max_power)
        if peak <= 0:
            return None

        level_ceiling = peak * STANDBY_BAND_MAX_FRACTION
        # Walk the tail; readings are chronological so we can break once outside
        # the window (O(window), not O(n)).
        window: list[float] = []
        window_ts: list[datetime] = []
        oldest_in_window: datetime | None = None
        saw_older = False  # a reading older than the window exists -> full coverage
        _standby_boundary_ts: datetime | None = None
        for ts, p in reversed(self._power_readings):
            if (timestamp - ts).total_seconds() <= STANDBY_BAND_WINDOW_S:
                window.append(float(p))
                window_ts.append(ts)
                oldest_in_window = ts
            else:
                saw_older = True
                _standby_boundary_ts = ts  # boundary: last reading before the window
                break
        # The plateau must actually SPAN the required window (data exists from
        # before it), not just a couple of recent samples, and have enough points
        # to judge.  `saw_older` (rather than an exact span >= WINDOW check) is
        # robust to sample phase/granularity: with e.g. 30 s sampling the oldest
        # in-window reading is typically only ~570-599 s old, which an exact check
        # would wrongly reject.  A coverage sanity (oldest >= 90% of the window)
        # plus an adjacent-gap check (``_window_has_outage_gap``) guard against a
        # sparse burst of samples sitting next to one old reading across a dropout.
        if (
            oldest_in_window is None
            or not saw_older
            or len(window) < 3
            or (timestamp - oldest_in_window).total_seconds()
            < STANDBY_BAND_WINDOW_S * 0.9
            or self._window_has_outage_gap(
                [_standby_boundary_ts, *window_ts]
                if _standby_boundary_ts is not None
                else window_ts
            )
        ):
            return None
        hi = max(window)
        lo = min(window)
        if hi > level_ceiling:
            return None  # a real active reading in the window - not standby
        flatness_limit = max(
            STANDBY_BAND_FLATNESS_FLOOR_W, peak * STANDBY_BAND_FLATNESS_FRACTION
        )
        if (hi - lo) > flatness_limit:
            return None  # fluctuating - still doing work
        # Two tiers. At or just above the stop threshold is the #445 shape - an
        # appliance whose standby the end gates cannot see - and it closes at the
        # expected duration. Anything else that passed the loose test above (a 0 W
        # soak, a 60 W rinse; flat and under 10% of a heater peak) waits for twice
        # the expected duration, as it did before 0.5.7. See the constants.
        stop = float(self._config.stop_threshold_w)
        near_stop_ceiling = standby_near_stop_ceiling(stop)
        if (
            stop > 0
            and float(np.median(window)) >= stop
            and hi <= near_stop_ceiling
            # #452: a halt says the programme is not done (it stopped on its
            # standby draw before its end), so the near-stop tier waits; the
            # loose tier below still bounds it at STANDBY_BAND_LOOSE_MIN_RATIO x
            # expected.
            and not self._stall_holds_standby_band()
        ):
            return hi
        if current_duration >= self._expected_duration * STANDBY_BAND_LOOSE_MIN_RATIO:
            return hi
        return None

    def _maybe_finalize_standby_band(self, timestamp: datetime, power: float) -> bool:
        """Finish the cycle when it is stuck on a standby plateau; True if it did.

        Called from RUNNING on every reading and from ENDING on every above-stop
        reading (DETECT-03); `_standby_band_plateau` holds every gate.
        """
        plateau_hi = self._standby_band_plateau(timestamp)
        if plateau_hi is None:
            return False
        start_time = self._current_cycle_start or timestamp
        current_duration = (timestamp - start_time).total_seconds()
        # The plateau sits ABOVE stop_threshold, so it keeps advancing
        # _last_active_time and the default keep_tail=False trim would NOT
        # remove it - inflating the stored duration/energy with minutes of
        # standby. Snap the end back to the last reading above the PLATEAU
        # and drop only the plateau run. This used to snap back to the last
        # reading above 10% of the cycle's peak, which cut every lower-power
        # phase after the last heating burst - 31 min of a Miele's rinse and
        # spin in one replayed cycle.
        trim_ceiling = plateau_hi + max(0.5, 0.1 * plateau_hi)
        plateau_start_idx = None
        for i in range(len(self._power_readings) - 1, -1, -1):
            if float(self._power_readings[i][1]) > trim_ceiling:
                plateau_start_idx = i
                break
        if (
            plateau_start_idx is not None
            and plateau_start_idx < len(self._power_readings) - 1
        ):
            self._power_readings = self._power_readings[: plateau_start_idx + 1]
            self._last_active_time = self._power_readings[-1][0]
            current_duration = (self._last_active_time - start_time).total_seconds()
        self._logger.info(
            "Standby-band finalize (%s): flat plateau ~%.1fW (peak %.0fW) held "
            "past expected %.0fs - appliance finished but holds a standby "
            "draw above stop_threshold; finalizing (plateau trimmed, "
            "duration %.0fs).",
            self._state,
            power,
            self._cycle_max_power,
            self._expected_duration,
            current_duration,
        )
        self._finish_cycle(
            timestamp,
            status="completed",
            termination_reason=TerminationReason.TIMEOUT,
            keep_tail=False,
        )
        return True

    def _anticrease_gate_open(self, timestamp: datetime) -> bool:
        """Core anti-crease gate (#296): everything except the current power level
        and the low-power-window check.  Shared by the match freeze
        (``_in_anticrease_freeze``) and the finalise (``_is_anticrease_tail``).

        True only when a genuinely energetic, confidently-matched cycle for an
        anti-wrinkle device is PAST its expected duration - the discriminator that
        separates the post-wash anti-crease tail from a mid-wash low-power trough
        (a washer spends most of its cycle below ``anti_wrinkle_max_power``, but a
        mid-wash trough is always BEFORE the expected duration, the tail after it).

        That "past expected" fraction is per-appliance since #429
        (``anti_crease_finalize_ratio``, default 0.98). **Lowering it trades away
        exactly the guarantee in the paragraph above**, so it is meant for dryers
        whose sensor-dry runtime follows the load and whose tumble tail would
        otherwise sit until the fallback timeout. Nothing downstream can stand in
        for it on a washer: the low-power window check cannot separate a trough
        from a tail (both are below ``anti_wrinkle_max_power`` by definition),
        ``_smart_term_power_plausible`` compares the trailing mean against the
        matched profile's OWN tail level, which is equally low, and
        ``_anticrease_spin_pending`` fails open when the profile carries no
        terminal high block.
        """
        if not self._config.anti_wrinkle_enabled:
            return False
        if self._config.device_type not in (
            DEVICE_TYPE_WASHING_MACHINE,
            DEVICE_TYPE_DRYER,
            DEVICE_TYPE_WASHER_DRYER,
        ):
            return False
        if getattr(self, "_verified_pause", False):
            return False
        if not (self._matched_profile and self._expected_duration > 0):
            return False
        if self._last_match_confidence < self._config.match_confidence_threshold:
            return False
        # The #288 full-shape verdict only (the wider #364 prefix-fit flag never
        # reached this gate, and was removed in 0.5.8): a false block here
        # disables the finalise AND the match freeze, and because
        # the tumble bursts recur faster than off_delay neither the fallback timeout
        # nor ENDING_HARD_FINALIZE can close the cycle - that is the #296 hang.
        if self._match_ambiguous or self._match_prefix_ambiguous_full_shape:
            return False
        if self._cycle_max_power <= float(self._config.anti_wrinkle_max_power):
            return False  # never a hot/energetic cycle - leave low-power programs alone
        # Cheap clock test first, so the trailing-power scan below is skipped for the
        # whole mid-wash phase (it only matters once we are past-expected).
        start = self._current_cycle_start
        if start is None:
            return False
        current_duration = self._gate_elapsed_s(timestamp)  # programme time, item 511
        # Held to the documented 0.50-1.00 range on READ, not at construction: the
        # manager assigns this field directly on an options reload, and the value can
        # arrive from an import or the Playground, neither of which range-checks it.
        # A stored 0.0 would satisfy the test below for every duration and hand the
        # gate a mid-wash trough.
        finalize_ratio = effective_anticrease_finalize_ratio(
            self._config.anti_crease_finalize_ratio
        )
        if current_duration < self._expected_duration * finalize_ratio:
            return False
        # #364: "past expected" only means "past the wash" when expected belongs to
        # the RIGHT profile. A whole washer wash phase sits below
        # anti_wrinkle_max_power, so with a mis-matched shorter profile this gate
        # would open mid-wash. Requiring the trailing power to look like this
        # profile's own tail restores the guarantee the ratio alone used to give.
        if not self._smart_term_power_plausible(timestamp):
            return False
        return True

    def _in_anticrease_freeze(self, timestamp: datetime) -> bool:
        """Whether match updates should be frozen (#296): the anti-crease gate is
        open AND the most recent reading is in the low-power regime (at or below
        ``anti_wrinkle_max_power``).

        Deliberately lighter than ``_is_anticrease_tail`` - it does NOT wait for the
        full ``ANTI_CREASE_CONFIRM_WINDOW_S``, so the confident pre-tail match is
        preserved from the instant the cycle crosses its expected duration in a
        low-power state, before the window accrues.  Without this a match that
        degrades to ambiguous within the first window's worth of tail would
        deadlock both the freeze and the finalise (both require an unambiguous
        match).  Self-correcting: a heating burst above ``anti_wrinkle_max_power``
        leaves the regime and re-arms matching.
        """
        if not self._power_readings:
            return False
        if float(self._power_readings[-1][1]) > float(
            self._config.anti_wrinkle_max_power
        ):
            return False
        return self._anticrease_gate_open(timestamp)

    def _is_anticrease_tail(self, timestamp: datetime) -> bool:
        """Whether a matched, past-expected cycle has settled into the anti-crease
        tumble tail (#296) - the trigger for the finalise into STATE_ANTI_WRINKLE.

        Miele-style "Knitterschutz": after the wash proper ends, the machine holds
        a constant baseline plus periodic sub-``anti_wrinkle_max_power`` tumble
        bursts (no heating) until the door is opened.  Because those bursts recur
        faster than off_delay they keep reviving the cycle out of ENDING, so the
        normal power-off path never finalises it and STATE_ANTI_WRINKLE - which is
        built to absorb the tail and split off the next wash - never engages.
        Recognising the tail lets us finalise into anti-wrinkle directly.

        Requires the core gate (``_anticrease_gate_open``) AND that the most recent
        >= ``ANTI_CREASE_CONFIRM_WINDOW_S`` of readings are ALL at or below
        ``anti_wrinkle_max_power`` (we are in the low-power tail, clear of the final
        spin and not mid-heating).  The expensive window scan runs only after the
        cheap gate passes, so normal cycles never pay for it.
        """
        if not self._anticrease_gate_open(timestamp):
            return False
        # Item 511: a stalled wash is halted, not in its tumble tail (the #296
        # bursts leave the stall band, so a real tail never shows as stalled). The
        # standby band's loose tier still bounds the plateau.
        if STALL_EXCLUDED_FROM_GATES and self._stall_active:
            return False
        # #399: only the finalise, never _anticrease_gate_open. A false block in the
        # shared gate would also kill the match freeze, and because the tumble bursts
        # recur faster than off_delay neither the fallback timeout nor
        # ENDING_HARD_FINALIZE could then close the cycle - that is the #296 hang.
        if self._anticrease_spin_pending(timestamp):
            return False
        max_power = float(self._config.anti_wrinkle_max_power)
        # Walk the tail; readings are chronological so we can break once outside the
        # window (O(window), not O(n)).
        window: list[float] = []
        window_ts: list[datetime] = []
        oldest_in_window: datetime | None = None
        saw_older = False
        _ac_boundary_ts: datetime | None = None
        for ts, p in reversed(self._power_readings):
            if (timestamp - ts).total_seconds() <= ANTI_CREASE_CONFIRM_WINDOW_S:
                window.append(float(p))
                window_ts.append(ts)
                oldest_in_window = ts
            else:
                saw_older = True
                _ac_boundary_ts = ts  # boundary: last reading before the window
                break
        # The low-power tail must actually SPAN the window (data exists from before
        # it) and have enough points to judge - not just a couple of recent samples.
        # ``saw_older`` plus a coverage sanity and an adjacent-gap check
        # (``_window_has_outage_gap``) is robust to sample phase/granularity while
        # rejecting a dropout-sized hole (mirrors _standby_band_plateau).
        if (
            oldest_in_window is None
            or not saw_older
            or len(window) < 3
            or (timestamp - oldest_in_window).total_seconds()
            < ANTI_CREASE_CONFIRM_WINDOW_S * 0.9
            or self._window_has_outage_gap(
                [_ac_boundary_ts, *window_ts]
                if _ac_boundary_ts is not None
                else window_ts
            )
        ):
            return False
        if max(window) > max_power:
            return False  # a heating / high-spin reading in the window - still washing
        return True

    def _anticrease_spin_pending(self, timestamp: datetime) -> bool:
        """Whether the matched profile still owes this run a terminal high-power
        event - i.e. the anti-crease finalise must wait (#399).

        ``_is_anticrease_tail``'s two conditions both look backwards: past expected,
        and quiet for the confirm window. A programme whose final spin lands just
        past 0.98 x expected, after a long sub-``anti_wrinkle_max_power`` rinse
        stretch, satisfies both while the spin is still ahead - so the wash was
        finalised into anti-wrinkle and the spin opened a SECOND cycle record.

        The profile carries the missing information: where its own last high-power
        block sits and how long it runs. If that block is terminal (starts at or
        after ``ANTI_CREASE_TERMINAL_HIGH_MIN_FRAC`` of the profile) and this run
        has not yet produced a comparable amount of high-power time at or after the
        same position, the spin is still ahead.

        Deliberately compares EVENTS, not clock positions: mapping the profile's
        last high sample onto elapsed time and clearing there delays the reported
        finalise by 16 s and then splits the wash anyway, because a run's spin can
        arrive hundreds of seconds later than the profile's (the same
        load-dependent duration spread behind #393).

        Delay-only and bounded: never blocks past
        ``ANTI_CREASE_SPIN_WAIT_MAX_RATIO`` x expected, and fails open on any
        missing input, so it cannot reproduce the #296 hang. The cap reads the
        gates' programme time (item 511).
        """
        block = self._matched_terminal_high
        if block is None:
            return False
        start_frac, block_seconds = block[0], block[1]
        if start_frac < ANTI_CREASE_TERMINAL_HIGH_MIN_FRAC:
            return False  # the profile's tail is genuinely low-power (#296 shape)
        expected = self._expected_duration
        start = self._current_cycle_start
        if expected <= 0 or start is None:
            return False
        current_duration = self._gate_elapsed_s(timestamp)
        if current_duration >= expected * ANTI_CREASE_SPIN_WAIT_MAX_RATIO:
            return False  # cap: waited long enough, let the finalise through
        needed = block_seconds * ANTI_CREASE_TERMINAL_MATCH_FRAC
        if needed <= 0:
            return False
        # Register item 196: scan from the block's ABSOLUTE offset on the profile's
        # own grid when the store supplied one (element 3). `start_frac` is measured
        # against the quiet-TRIMMED span - it has to be, or a capture's idle tail
        # disarms the terminal gate above - while `expected` is the profile's
        # avg_duration, which tracks the UNTRIMMED span. Their product is therefore a
        # systematically LATE offset, and a late offset means the run's own spin sits
        # BEFORE the scan window and is never counted, so the hold ran out the
        # ANTI_CREASE_SPIN_WAIT_MAX_RATIO cap instead of releasing on the event. Over
        # 36 real armed profile/cycle pairs the product recognised the spin 3 times
        # and the absolute offset 14, with zero cases in either where the credited
        # seconds exceeded the run's own terminal block (so no new premature-release
        # exposure). The fallback keeps a pre-196 payload - an old state snapshot, the
        # Playground, older callers - behaving exactly as before.
        offset_s = float(block[2]) if len(block) >= 3 else start_frac * expected
        # Element 4, when the store sent one, is the watts the block was measured
        # against. Count the live seconds above the SAME bar or the two halves
        # describe different things (register item 351).
        ceiling_w = float(block[3]) if len(block) >= 4 else None
        seen = self._high_power_seconds_since(offset_s, ceiling_w=ceiling_w)
        if seen >= needed:
            return False
        if not self._anticrease_spin_wait_logged:
            self._anticrease_spin_wait_logged = True
            self._logger.debug(
                "Finalize held, terminal high-power block still owed: '%s' ends "
                "with a %.0fs block above %.0fW at %.0f%% of its run (scanning "
                "from %.0fs); this cycle has %.0fs of it so far (elapsed %.0fs of "
                "%.0fs expected).",
                self._matched_profile,
                block_seconds,
                (
                    float(self._config.anti_wrinkle_max_power)
                    if ceiling_w is None
                    else ceiling_w
                ),
                start_frac * 100.0,
                offset_s,
                seen,
                current_duration,
                expected,
            )
        return True

    def _high_power_seconds_since(
        self, offset_s: float, ceiling_w: float | None = None
    ) -> float:
        """Seconds this cycle has spent above ``ceiling_w`` at or after ``offset_s``
        from its start (#399); ``anti_wrinkle_max_power`` when None.

        The caller supplies the ceiling so both halves of the comparison use one
        bar: the profile's block was measured against it too. The standby-band
        path passes a share of the cycle's own peak (register item 351), the
        anti-crease path passes nothing and keeps the dryer's tumble level.

        Walks the readings backwards and stops at the offset, so the scan is bounded
        by the tail of the trace rather than its whole length. Each reading covers
        the interval up to the following one, which matches how the profile's own
        block length is measured.

        Two corrections to that per-interval credit, both of which decide whether the
        guard releases:

        * An outage-sized interval is unobserved time, not high-power time. Counting
          it in full let a silent plug bank minutes of "spin" it never reported,
          satisfy ``seen >= needed`` and release the finalise before the real
          terminal spin - the #399 failure, reached by a different route. Same
          treatment (and the same p95-derived ceiling) the tail scan at
          ``_smart_term_tail_stats`` and the gap-free quiet tally already apply.
          Deliberately NOT ``_outage_threshold_s()``, which rebuilds a NumPy array
          from every reading; this runs on the per-reading anti-crease path.
        * When ``offset_s`` falls inside an interval, only the part after the offset
          counts. Breaking out of the loop dropped that remainder entirely, and the
          offset is ``start_frac * expected``, so a boundary reading is the norm
          rather than an edge case.
        """
        start = self._current_cycle_start
        if start is None or not self._power_readings:
            return 0.0
        if self._stall_spans or self._user_pause_spans:
            # The offset is on the profile's grid: programme time (items 511, 514).
            offset_s = self._wall_offset_s(offset_s)
        ceiling = (
            float(self._config.anti_wrinkle_max_power)
            if ceiling_w is None
            else float(ceiling_w)
        )
        max_gap = min(3600.0, max(60.0, 10.0 * self._prior_p95_dt))
        total = 0.0
        readings = self._power_readings
        for i in range(len(readings) - 1, -1, -1):
            ts, power = readings[i]
            elapsed = (ts - start).total_seconds()
            if i + 1 >= len(readings):
                continue  # last reading covers no interval yet
            next_elapsed = (readings[i + 1][0] - start).total_seconds()
            if next_elapsed <= offset_s:
                break  # this interval ends at or before the offset, as do all earlier ones
            interval = next_elapsed - elapsed
            if interval > max_gap:
                if elapsed < offset_s:
                    break
                continue  # unobserved time, not evidence of anything
            if float(power) > ceiling:
                # Credit only the portion at or after the offset.
                total += next_elapsed - max(elapsed, offset_s)
        return total

    def _maybe_finalize_anticrease_tail(self, timestamp: datetime) -> bool:
        """Finalise a cycle that has entered the anti-crease tail into
        STATE_ANTI_WRINKLE (#296).  Returns True if the cycle was finalised.

        Shared by the RUNNING / PAUSED / ENDING branches so the finalise fires no
        matter which state a burst left the detector in.  Uses Smart Termination
        (in ``ANTI_WRINKLE_ELIGIBLE_REASONS``) so ``_finish_cycle`` routes into
        STATE_ANTI_WRINKLE, which then absorbs the tail and splits off any next
        wash on its first heating burst above ``anti_wrinkle_max_power``.
        """
        if not self._is_anticrease_tail(timestamp):
            return False
        start_time = self._current_cycle_start or timestamp
        current_duration = (timestamp - start_time).total_seconds()
        # Register item 393a: the tail's baseline, from the window that was just
        # accepted as tail (read before _finish_cycle clears the readings). Twice
        # its lowest reading, never under the anti-wrinkle exit level.
        tail_floor = max(
            float(self._config.anti_wrinkle_exit_power),
            float(self._config.stop_threshold_w),
            2.0 * min(
                (
                    float(p)
                    for ts, p in self._power_readings
                    if (timestamp - ts).total_seconds() <= ANTI_CREASE_CONFIRM_WINDOW_S
                ),
                default=0.0,
            ),
        )
        self._logger.info(
            "Anti-crease finalize: matched '%s' past expected %.0fs (elapsed %.0fs), "
            "settled into the low-power tumble tail — finalizing into anti-wrinkle.",
            self._matched_profile,
            self._expected_duration,
            current_duration,
        )
        self._finish_cycle(
            timestamp,
            status="completed",
            termination_reason=TerminationReason.SMART,
            keep_tail=True,
            # Capped like the three sibling keep_tail finishes (#424). This path can
            # fire on a window of *watchdog* keepalives: a change-only plug that has
            # gone silent emits nothing, the injected 0 W readings satisfy the "all
            # at or below anti_wrinkle_max_power" window, and `timestamp` is then
            # the moment the watchdog noticed rather than the moment the appliance
            # stopped - post-cycle standby banked as cycle time, which feeds
            # avg_duration and self-amplifies.
            #
            # The tumble tail itself is never cut into, which is why this is safe
            # here: an anti-crease baseline sits ABOVE stop_threshold
            # (const.py:811-812, a ~2.5-3.2 W draw against a ~1.2 W threshold), so
            # every one of those readings refreshes `_last_active_time`, and for a
            # non-dishwasher `_keep_tail_cap` returns exactly that - the last
            # tumble. A real tail therefore loses only the trailing quiet gap
            # between its last reading and this finalize, which is time the
            # appliance drew nothing, and a tail of genuinely-0 W readings is
            # dropped at `_last_active_time`.
            #
            # NB: the cap is no longer `max(expected_end, _last_active_time)` and
            # so is NOT bounded below by the expected end - on a washer or dryer
            # the stored end can now land earlier than the matched profile's
            # expected duration. Shorten-only still holds; "never earlier than
            # expected_end" no longer does, and this paragraph used to claim it.
            tail_cap=self._keep_tail_cap(start_time),
        )
        # After _finish_cycle, whose reset() clears it (register item 393a).
        if self._state == STATE_ANTI_WRINKLE:
            self._anticrease_tail_floor_w = tail_floor
        return True

    def _is_terminal_drop(self) -> bool:
        """Whether the current low-power event is an anomalously-early hard drop.

        Mirrors ``_ml_end_confidence``: builds the offset-second trace from the
        current cycle's readings and asks the injected terminal-drop provider.
        Returns ``False`` when there is no provider, no cycle start, or the
        provider declines/raises (ML off / too little history / not anomalous),
        so the caller keeps the proven soak-bridging end-detection.
        """
        provider = self._terminal_drop_provider
        start = self._current_cycle_start
        if provider is None or start is None or not self._power_readings:
            return False
        # Throttle: reuse within the window, scoped to this cycle + expected_duration.
        now_ts = self._power_readings[-1][0]
        exp = float(self._expected_duration)
        cache = self._terminal_drop_cache
        if (
            cache is not None
            and cache[1] == exp
            and cache[2] == start
            and (now_ts - cache[0]).total_seconds() < ML_PROVIDER_THROTTLE_SECONDS
        ):
            return cache[3]
        points = [
            ((ts - start).total_seconds(), float(power))
            for ts, power in self._power_readings
        ]
        try:
            result = bool(provider(points, exp))
        except Exception:  # noqa: BLE001 - ML must never break detection
            result = False
        self._terminal_drop_cache = (now_ts, exp, start, result)
        return result

    def _should_defer_finish(self, duration: float) -> bool:
        """Check if we should defer termination based on expected duration."""
        # Check explicit verified pause override from manager
        if getattr(self, "_verified_pause", False):
            self._logger.debug("Deferring cycle finish: Verified pause active")
            return True

        # Dishwasher minimum-duration floor: even without a matched profile (e.g.
        # first cycle of a program, or the 5-min matcher hasn't fired yet) a
        # dishwasher cycle should never end before it has crossed the minimum
        # reasonable programme duration.  This prevents a dip during the fill or
        # early wash phase from being read as the end of a complete cycle.
        #
        # A MATCHED profile overrides the blanket constant with its own learned
        # length, because the constant is a stand-in for exactly the knowledge a
        # match supplies - as this comment's own "even without a matched profile"
        # says. Blanket, it is wrong for real hardware: the community catalogue
        # carries a 6.0 min Smeg "Delay- prewash", which a 30 min floor defers by
        # half an hour. The floor still applies unmatched, and a matched profile
        # can only ever LOWER it (`min`), never license a longer deferral - the
        # 39.4 min Electrolux "Rapido" already clears it and is unaffected.
        #
        # Gated on a TRUSTED match, not merely a present one. This floor is an
        # anti-premature-end guard, so the risk is the opposite way round from
        # the ENDING fallback gate (item 329, where a confidence check measured
        # as pure cost): getting this wrong ends a dishwasher during its fill or
        # early-wash dip and records the rest of the programme as a second
        # cycle, which is the expensive failure. A low-confidence match to a
        # short look-alike is exactly how that happens, so it does not get to
        # lower the bar; nor does an ambiguous one. (It also honoured the #364
        # prefix-fit flag until that was removed in 0.5.8.) No-op on the whole
        # corpus either way - every corpus dishwasher profile is over 90 minutes.
        _dw_floor = DISHWASHER_MIN_CYCLE_DURATION_S
        if (
            self._matched_profile
            and self._expected_duration > 0
            and self._last_match_confidence >= self._config.match_confidence_threshold
            and not self._match_ambiguous
        ):
            _dw_floor = min(_dw_floor, float(self._expected_duration))
        if self._config.device_type == "dishwasher" and duration < _dw_floor:
            self._logger.debug(
                "Deferring dishwasher cycle end: elapsed %.0fs < minimum %.0fs",
                duration,
                _dw_floor,
            )
            return True

        if not self._matched_profile or self._expected_duration <= 0:
            return False

        # Safety: Don't defer forever
        if duration > (self._expected_duration + DEFAULT_MAX_DEFERRAL_SECONDS):
            self._logger.warning(
                "Deferral limit exceeded (%.0fs > expected %.0f + %s), allowing finish",
                duration,
                self._expected_duration,
                DEFAULT_MAX_DEFERRAL_SECONDS,
            )
            return False

        # Opt-in ML end-guard (asymmetric anti-premature-stop, bounded). If the
        # cycle-end model judges this low-power event to be more likely a pause
        # than the true end, defer the normal completion - but only for a bounded
        # extra window, so a wrong model can delay, never hang, the cycle. As the
        # low-power run lengthens the model's confidence rises, so a genuine end
        # is released once the model agrees or the cap is reached.
        if (
            self._end_confidence_provider is not None
            and self._last_match_confidence >= DEFAULT_DEFER_FINISH_CONFIDENCE
        ):
            confidence = self._ml_end_confidence()
            if confidence is not None and confidence < ML_END_GUARD_MIN_CONFIDENCE:
                if self._ml_defer_start_duration is None:
                    self._ml_defer_start_duration = duration
                if (duration - self._ml_defer_start_duration) < ML_END_GUARD_MAX_DEFER_SECONDS:
                    self._logger.debug(
                        "Deferring cycle finish: ML end-guard (P(true end)=%.2f < %.2f)",
                        confidence,
                        ML_END_GUARD_MIN_CONFIDENCE,
                    )
                    return True
            elif confidence is not None:
                # Model is confident this is the true end -> stop ML-deferring.
                self._ml_defer_start_duration = None

        # Dishwasher passive drying protection:
        # Dishwashers can have 2+ hour passive drying phases at near-0W.  A terminal
        # drain spike that fires early in the ENDING state (e.g. at 120 min of a
        # 233-min ECO cycle) resets _time_below_threshold, and the subsequent 60-min
        # silence timeout would otherwise end the cycle at ~180 min - well before the
        # real finish.  Defer until the cycle reaches the late-phase threshold (the
        # same one used by the end-spike arm gate, so both move together) so that
        # smart termination can catch the true end (~99% of expected) instead.
        # Confidence may be low this early, so the normal confidence gate is
        # bypassed here.
        if (
            self._config.device_type == "dishwasher"
            and self._matched_profile
            and self._expected_duration > 0
            and duration
            < (self._expected_duration * DISHWASHER_END_SPIKE_MIN_PROGRESS)
        ):
            self._logger.debug(
                "Deferring cycle finish: dishwasher drying phase protection "
                "(%.0fs < %.0f%% of expected %.0fs, profile: %s, conf %.2f)",
                duration,
                DISHWASHER_END_SPIKE_MIN_PROGRESS * 100,
                self._expected_duration,
                self._matched_profile,
                self._last_match_confidence,
            )
            return True

        # Issue #43: dishwasher end-spike wait protection.  Once past the 85%
        # passive-drying gate above, we still keep the cycle deferred until
        # the real end-of-cycle pump-out fires (sets _end_spike_seen=True via
        # the 85% progress gate in STATE_ENDING) or we cross the
        # smart-termination wait window (expected + 30 min) - whichever comes
        # first.  Shares DISHWASHER_END_SPIKE_WAIT_SECONDS with Smart
        # Termination's wait branch so the two paths release the cycle at the
        # same instant.  Beyond the wait window, Smart Termination's
        # past_wait_period kicks in and finalises; below it, the fallback
        # timeout's energy gate is the safety net for cycles whose pump-out
        # never arrives.
        # Mirrors the STATE_ENDING pump-out wait so both paths release together.
        # Keep deferring while we are still inside the wait window, UNLESS the cycle
        # has already reached its expected duration and has since been sustained-quiet
        # for DISHWASHER_END_SPIKE_QUIET_RELEASE_SECONDS - in which case any terminal
        # pump-out has already happened, so a cycle that finished slightly short of the
        # profile's (drifted-up) average is released here instead of hanging to
        # expected + 30 min.  The ``duration >= expected`` gate keeps a long
        # passive-drying phase that still precedes a late pump-out deferred.
        quiet_released = (
            duration >= self._expected_duration
            and self._time_below_threshold_gapfree
            >= self._dishwasher_quiet_release_s()
        )
        if (
            self._config.device_type == "dishwasher"
            and self._matched_profile
            and self._expected_duration > 0
            and not self._end_spike_seen
            and duration
            < (self._expected_duration + self._dishwasher_end_spike_wait_s())
            and not quiet_released
        ):
            # Report the gap-free tally: that is what `quiet_released` above reads,
            # and after a telemetry outage the two diverge - logging the plain one
            # would show quiet time that played no part in the decision.
            self._logger.debug(
                "Deferring cycle finish: dishwasher waiting for end-of-cycle "
                "pump-out (%.0fs < expected %.0fs + %.0fs wait, observed quiet "
                "%.0fs of %.0fs needed, profile: %s)",
                duration,
                self._expected_duration,
                self._dishwasher_end_spike_wait_s(),
                self._time_below_threshold_gapfree,
                self._dishwasher_quiet_release_s(),
                self._matched_profile,
            )
            return True

        # If matched profile, enforce min duration ratio
        ratio = self._config.min_duration_ratio

        # --- STRICTER DEFERRAL ---
        # If we are NOT in a verified pause, but power has been low for a long time (ENDING state),
        # we only defer if we are VERY confident this profile is correct.
        # This prevents hanging on too-long profiles that matched early but are now diverging.
        if self._last_match_confidence < DEFAULT_DEFER_FINISH_CONFIDENCE:
            self._logger.debug(
                "Not deferring finish: confidence %.2f too low for unverified pause (profile: %s)",
                self._last_match_confidence,
                self._matched_profile,
            )
            return False

        # Primary check: Is duration significantly below expectation?
        if duration < (self._expected_duration * ratio):
            self._logger.debug(
                "Deferring cycle finish: duration %.0fs < %.0f%% of expected %.0fs (profile: %s, confidence %.2f)",
                duration,
                ratio * 100,
                self._expected_duration,
                self._matched_profile,
                self._last_match_confidence,
            )
            return True

        # (A "ratio to 1 + profile_duration_tolerance" window used to follow, but both
        # of its branches returned False - the tolerance changed nothing here.)
        return False

    def _fallback_shortening_bar(
        self, expected: float, ambiguous: bool, longest: float
    ) -> float | None:
        """Elapsed seconds from which the ENDING fallback shortens, None when blocked.

        The item 306/330/355 bar for a match with these values: ``expected`` at the
        device's late ratio, or, while the match is ambiguous, the longest candidate
        at the plain ``END_GATE_LATE_RATIO`` when that is longer, and blocked when
        there are no candidate durations to compare (see the gate in
        ``process_reading``). Shared by that gate and
        :meth:`ambiguous_ending_match_defers`, so the two cannot disagree.
        """
        bar = float(expected)
        raised = False
        if ambiguous:
            if longest > bar:
                bar = float(longest)
                raised = True
            elif longest <= 0.0:
                return None
        ratio = (
            END_GATE_LATE_RATIO
            if raised
            else resolve_end_gate_late_ratio(self._config.device_type)
        )
        return ratio * bar

    def _ending_wait_left_s(
        self, expected: float, ambiguous: bool, longest: float, confidence: float, cat: Any
    ) -> float:
        """Seconds until the ENDING fallback would finish under continued quiet, for
        a match with these values (register item 469b).

        The fallback gate in ``process_reading``, run forward: the hazard wait
        before the shortening bar, the shortened wait after it (the gate swaps one
        for the other, so the shortening can also lengthen), then
        ``_should_defer_finish``'s duration floor, which needs the confidence. The
        energy gate and Smart Termination are left out: neither depends on which
        programme is matched in a way an ambiguous tick changes, except that the
        ambiguity blocks Smart Termination (see the caller).
        """
        now = self._power_readings[-1][0]
        elapsed = self._gate_elapsed_s(now)  # the gate's own clock (item 511)
        quiet = float(self._time_below_threshold)
        hazard = self._hazard_wait_for(
            now, max(self._config.off_delay, self._config.min_off_gap), cat, expected, ambiguous
        )
        left = max(0.0, hazard - quiet)
        bar = self._fallback_shortening_bar(expected, ambiguous, longest)
        if bar is not None and elapsed + left >= bar:
            short = max(
                self._config.off_delay,
                min(self._config.min_off_gap, END_GATE_LATE_SECONDS),
            )
            left = max(0.0, short - quiet, bar - elapsed)
        if confidence >= DEFAULT_DEFER_FINISH_CONFIDENCE:
            left = max(left, expected * float(self._config.min_duration_ratio) - elapsed)
        return left

    def ambiguous_ending_match_defers(
        self, expected: Any, confidence: Any, longest: Any, catalogue: Any = None
    ) -> bool:
        """Would an AMBIGUOUS match with these values make this ENDING wait longer?

        Register item 469(b). A tick in ENDING matches a trace that ends in its idle
        tail, which reads as the prefix of a longer programme pausing, so its top-1
        drifts longer and ties with its runner-up. Applied as is, a shorter match
        interval (what "Apply all" suggests) let such a tick defer the end: 01KXGA3C
        62f39dfc34f4, a 29 min wool wash, waited 23.2 min instead of 6.7 min at an
        87 s interval. ``update_match`` refuses such a match when this says the
        fallback would finish later with it (:meth:`_ending_wait_left_s` for both),
        or as late when the current match is clear (only the ambiguous one blocks
        Smart Termination). Not "a longer programme": the shortening bar is not
        monotone in the expected duration. A match to the LONGEST candidate does
        not raise it, so it can end sooner than a shorter ambiguous one whose bar
        the longest raised (01KBWSV8 1dbc19ccba79 / daea1437efcc: 6.7 min applied,
        23-25 min held), and a shorter one can raise it (01KXGA3C 9c2624675652).
        A dishwasher's drying and pump-out waits follow the expected duration
        alone, so there a longer one is refused outright.

        False outside ENDING or without a matched programme: nothing to keep.
        """
        if (
            self._state != STATE_ENDING
            or not self._matched_profile
            or self._expected_duration <= 0
            or self._current_cycle_start is None
            or not self._power_readings
        ):
            return False
        try:
            new_expected = float(expected)
            new_confidence = float(confidence or 0.0)
            new_longest = float(longest or 0.0)
        except (TypeError, ValueError, OverflowError):
            return True
        if not (math.isfinite(new_expected) and new_expected > 0):
            return True  # would unmatch the cycle: the unshortened fallback
        if (
            self._config.device_type == DEVICE_TYPE_DISHWASHER
            and new_expected > self._expected_duration
        ):
            return True
        new_left = self._ending_wait_left_s(
            new_expected, True, new_longest, new_confidence,
            self._sanitize_pause_catalogue(catalogue),
        )
        old_left = self._ending_wait_left_s(
            self._expected_duration, self._match_ambiguous,
            self._longest_candidate_duration, self._last_match_confidence,
            self._matched_pause_catalogue,
        )
        if new_left > old_left + 1.0:
            return True
        return not self._match_ambiguous and new_left >= old_left - 1.0

    def _dishwasher_quiet_release_s(self) -> float:
        """Sustained quiet past expected that releases the pump-out wait (#379).

        The configured value - raised to DISHWASHER_QUIET_RELEASE_TERMINAL_MARGIN x
        the matched profile's measured quiet before its terminal event while this
        run has NOT yet been through that quiet (register item 392). The release
        exists for a pump-out that already happened or never comes; a run still
        short of the quiet its programme always has before the pump-out has not
        reached that point. A run that has been through it - #424's Beko and the
        Hatton ECO, whose element-11 event sits mid-cycle - keeps the configured
        value (or the item-465 floor below). Lengthen-only; the 30 min spike wait
        still bounds the whole wait.

        Also never shorter than the same margin x the longest below-stop pause
        the profile's traced evidence ever came back from (element 14, register
        item 465), whether or not the run has been through its terminal quiet.
        Element 11 is a MEDIAN over the profile's last events, and a cycle closed
        before its pump-out contributes the wrong event: on 01KGM619's Eco half
        the stored cycles end on their last heating block (100-160 s gap), the
        rest on a pump-out 4840-4860 s after it, so the median lands at 2500 s,
        a quiet no cycle has. 1.1 x that released 6 of 6 pump-out cycles 6-11 min
        before their pump-out, and each stored ~8.1k s instead of ~11.1k s, which
        drags `avg_duration`, and with it the next release, earlier. A pause the
        programme has resumed from says "activity can still follow a quiet this
        long" directly. No position filter: the release is only asked past the
        expected end, and the catalogue's fractions are of each cycle's own span,
        which is longer than `expected` on exactly the cycles that kept their
        pump-out. Lengthen-only, bounded by the same spike wait.
        """
        base = max(
            float(self._config.dishwasher_end_spike_quiet_release),
            self._resumed_pause_release_s(),
        )
        quiet = self._matched_terminal_quiet_s
        if not (self._matched_profile and quiet) or not self._power_readings:
            return base
        now = self._power_readings[-1][0]
        if self._spike_follows_terminal_quiet(now, latest=True):
            return base
        return max(base, DISHWASHER_QUIET_RELEASE_TERMINAL_MARGIN * float(quiet))

    def _resumed_pause_release_s(self) -> float:
        """Margin x the longest pause the matched profile resumed from, else 0.

        Read from the hazard gate's catalogue (element 14) with its evidence bar
        (END_GATE_HAZARD_MIN_CYCLES traced cycles). See
        :meth:`_dishwasher_quiet_release_s` (register item 465).
        """
        cat = self._matched_pause_catalogue
        if (
            cat is None
            or cat[0] < END_GATE_HAZARD_MIN_CYCLES
            or not cat[1]
            or not self._matched_profile
        ):
            return 0.0
        return DISHWASHER_QUIET_RELEASE_TERMINAL_MARGIN * max(d for _f, d in cat[1])

    def _spike_follows_terminal_quiet(
        self, timestamp: datetime, *, latest: bool = False
    ) -> bool:
        """May this ENDING spike be the dishwasher's terminal pump-out? (item 392)

        True unless the matched profile's quiet before its terminal event has been
        measured (element 11) and this run has not yet been through it - asked
        with the same `terminal_quiet_seen` the keep-tail cap and the banked-tail
        repair use. On the corpus's "65° full" (934 s measured, 17/17 cycles) a
        24 W fan blip 211 s after the last heating armed the end at 85% of
        expected, and Smart Termination closed the cycle 12 min before the real
        pump-out. Only ever withholds the arm, so it can delay an end, never
        bring one forward; every other device type, and an unmeasured profile,
        is unaffected.
        """
        quiet = self._matched_terminal_quiet_s
        start = self._current_cycle_start
        if (
            self._config.device_type != DEVICE_TYPE_DISHWASHER
            or not self._matched_profile
            or not quiet
            or start is None
            or not self._power_readings
        ):
            return True
        # Memoised per reading: the release asks this on every ENDING reading past
        # the expected end, and the answer only changes when a reading arrives.
        key = (len(self._power_readings), timestamp, latest)
        hit = self._terminal_quiet_memo
        if hit is not None and hit[0] == key:
            return hit[1]
        pts = [((ts - start).total_seconds(), float(pw)) for ts, pw in self._power_readings]
        last_off = (timestamp - start).total_seconds()
        if latest:
            # "Has the run been through it by now?": measured from the last
            # activity, not from a spike that has not happened.
            last_active = self._last_active_time
            if last_active is None:
                return True
            last_off = (last_active - start).total_seconds()
        seen = terminal_quiet_seen(
            pts,
            last_off,
            self._config.stop_threshold_w,
            float(quiet),
            TERMINAL_EVENT_PEAK_FRAC,
        )
        self._terminal_quiet_memo = (key, seen)
        return seen

    def _dishwasher_end_spike_wait_s(self) -> float:
        """Grace past the expected end while waiting for the terminal pump-out.

        Capped at the programme's OWN expected length - the same ``min()`` shape
        register item 331 gave ``DISHWASHER_MIN_CYCLE_DURATION_S``, for the same
        reason. A flat 1800 s is 20% of a 150 min ECO cycle but **five times** a
        6 min Smeg "Delay- prewash", and the community catalogue carries exactly
        that programme. Asymmetric: the cap can only ever SHORTEN the wait, never
        extend it, so no cycle waits longer than it does today.

        A no-op across the maintainer's corpus, where every dishwasher profile is
        >90 min and the cap therefore never binds (register item 357). Unmatched
        cycles keep the flat constant: with no expected duration there is nothing
        to be proportional to.
        """
        expected = float(self._expected_duration or 0.0)
        if expected <= 0:
            return DISHWASHER_END_SPIKE_WAIT_SECONDS
        return min(DISHWASHER_END_SPIKE_WAIT_SECONDS, expected)

    def _ending_hard_finalize_quiet_s(self) -> float:
        """Continuous sub-threshold span the ENDING backstop requires.

        Same cap, same reason: 600 s of required quiet is a third of a 30 min
        programme and longer than a 6 min one, which would disarm the backstop
        entirely on a short programme - the opposite of what a safety net is for.
        The ``off_delay`` / ``min_off_gap`` floor is applied by the caller and is
        unaffected.
        """
        expected = float(self._expected_duration or 0.0)
        if expected <= 0:
            return ENDING_HARD_FINALIZE_MIN_QUIET_S
        return min(ENDING_HARD_FINALIZE_MIN_QUIET_S, expected)

    def _keep_tail_cap(self, start_time: datetime) -> datetime | None:
        """Latest end time a *kept* tail may claim (#424).

        The paths that keep their tail do so because the tail can be real cycle
        time: a dishwasher's near-0 W passive drying phase sits between the last
        drain spike and the actual end of the programme (issue #43), so snapping
        back to ``_last_active_time`` would store a falsely short cycle. But they
        fire on accumulated quiet time, and on a publish-on-change plug that wait
        is minutes of *post-appliance* standby - which stamping ``timestamp`` as
        the end time banked as cycle time.

        Measured on the #424 reporter's dishwasher: every cycle that ended via
        `timeout` stored a 0-29 s tail (237-239 min, matching the appliance), and
        every cycle that ended via `smart` stored a 96-1239 s tail (240-260 min),
        with ``_last_active_time`` unchanged across the whole history - only the
        termination path differed. It also self-amplifies, because the inflated
        duration feeds ``avg_duration``, which raises ``expected_duration``,
        which delays the next Smart Termination further (the second reporter's
        profile had already drifted 63 -> 70.5 min).

        The cap used to be the matched profile's **expected end**. That is the
        mean of these same stored durations, so it moved with the thing it was
        bounding: a banked tail raised ``avg_duration``, the higher average
        allowed a longer tail, and the reported end drifted later every run.
        Measured over 375 cycles from 16 devices, smart-terminated cycles banked a
        median **12.6 min** of post-appliance time (washing machines **22.7 min**,
        p90 40.4 min) against ~0 min for every other termination path, and the
        profiles carried a mean **+5.3%** duration inflation as a result - on the
        #427 reporter's washer, +20.2 min on a 108 min programme, which is also
        why their ETA read 131 min for a ~105 min wash.

        So anchor on the last real activity instead. Where a passive phase can
        legitimately follow it, allow only what this programme has been
        **measured** to do (``profile_terminal_quiet_seconds``, element 11) - a
        statistic taken from the traces, not from the stored durations, so it
        cannot be inflated by the tail it bounds, and gated on having been seen
        repeatedly rather than once.

        **Only a dishwasher has a passive terminal phase.** Every other type ends
        on activity - a washer's spin, a dryer's drum - which ``_last_active_time``
        already marks, so there is nothing legitimate to bank after it. That is
        not a new assumption: the fallback-timeout path beside this one has always
        read ``keep_tail = device_type == "dishwasher"``. Smart Termination was
        the one path that kept a tail for every type, which is exactly where the
        22.7 min washing-machine median came from.

        Within the dishwasher case, two sub-cases, and the difference matters:

        * the run produced its terminal pump-out (``_end_spike_seen``).
          ``_last_active_time`` already sits on it, so that IS the end.
        * it did not, so the programme ended in its passive drying phase. Allow up
          to the profile's measured quiet span past the last activity.

        Falls back to the old expected-end cap when that span has not been
        measured, rather than truncating a drying phase on no evidence - the
        measured corpus shows the pump-out missing in a substantial minority of
        runs on some machines, and in those runs the drying IS the tail. Still
        asymmetric and shorten-only; still None for an unmatched cycle.
        """
        if self._expected_duration <= 0:
            return None
        expected_end = start_time + timedelta(seconds=self._expected_duration)
        last_active = self._last_active_time
        if last_active is None:
            return expected_end
        if self._config.device_type != DEVICE_TYPE_DISHWASHER:
            return last_active
        cap = self._dishwasher_tail_cap(start_time, expected_end, last_active)
        # Never before the length the user has vouched for (register item 384),
        # the floor `ProfileStore.async_repair_banked_tails` applies, so live and
        # the repair store the same duration. Element 11 is measured from the
        # profile's own traces, and on a machine that dries silently AFTER its
        # last activity it reads the pre-drying pause instead: the Hatton ECO
        # export stores 230-235 min (one corrected to 234), and without this the
        # cap snapped three of its four cycles to their last activity at ~120 min.
        # Lengthen-only, and only for a run that has itself lasted that long: a
        # cancelled or short run never reaches the floor, and lifting its cap
        # would bank its whole post-appliance wait (#424). Same outcome as the
        # repair, which never lengthens a cycle.
        trusted = self._matched_trusted_min_s
        if trusted and self._power_readings:
            floor = start_time + timedelta(seconds=TRUSTED_LENGTH_FLOOR_FRAC * trusted)
            if cap < floor <= self._power_readings[-1][0]:
                return floor
        return cap

    def _dishwasher_tail_cap(
        self, start_time: datetime, expected_end: datetime, last_active: datetime
    ) -> datetime:
        """The dishwasher half of :meth:`_keep_tail_cap`, before the trusted floor."""
        # Only a spike LATE enough to be the terminal pump-out licenses snapping
        # the stored end back to the last activity. `_end_spike_seen` is set from
        # DISHWASHER_END_SPIKE_MIN_PROGRESS (0.85), but this file does not treat
        # every such spike as terminal: `_resolve_smart_ratio` relaxes its gate
        # only at `>= expected * 0.90`, because below that the spike can be the
        # pre-final-rinse drain with a passive Dry phase still to come. Capping
        # at `last_active` for an 87% drain cuts that drying off, which lowers
        # `avg_duration`, which makes the NEXT Smart Termination fire earlier -
        # the error compounds in the direction that splits cycles. Same 0.90
        # test here, so a pre-rinse drain falls through to the measured quiet
        # span or the expected-end fallback below.
        if getattr(self, "_end_spike_seen", False) and getattr(
            self, "_end_spike_duration", 0.0
        ) >= self._expected_duration * 0.90:
            return last_active
        quiet = self._matched_terminal_quiet_s
        if quiet is None:
            return max(expected_end, last_active)
        # ...and if the drying ALREADY happened, adding the allowance on top
        # counts the same quiet twice. The 0.90 test above only catches a spike
        # late enough to be unambiguously terminal; a pump-out at 85-90% of
        # expected falls through it, and on a machine that dries BEFORE its final
        # drain that banks a second drying period into the stored duration, which
        # feeds `avg_duration` - the exact drift item 297 exists to remove.
        # `ProfileStore.async_repair_banked_tails` has always asked the trace this
        # question; this path did not, so the same cycle got one duration live and
        # another when the repair re-judged it. Same helper, same 0.5 bar, so the
        # two cannot drift again (register item 347).
        #
        # And it has to ask at the threshold the allowance was MEASURED at (#424):
        # see `terminal_quiet_seen` for the Beko that banked 10 min on every cycle
        # because this test ran at the stop threshold while the statistic it
        # guards is taken at a fraction of the cycle's peak.
        if self._current_cycle_start is not None and self._power_readings:
            _start = self._current_cycle_start
            _pts = [
                ((ts - _start).total_seconds(), float(pw))
                for ts, pw in self._power_readings
            ]
            _last_off = (last_active - _start).total_seconds()
            if terminal_quiet_seen(
                _pts,
                _last_off,
                self._config.stop_threshold_w,
                float(quiet),
                TERMINAL_EVENT_PEAK_FRAC,
            ):
                # ...at the terminal event, which may sit below the stop
                # threshold and so after `last_active` (register item 384).
                return _start + timedelta(
                    seconds=terminal_event_end(_pts, _last_off, TERMINAL_EVENT_PEAK_FRAC)
                )
        return last_active + timedelta(seconds=min(quiet, TERMINAL_QUIET_CAP_S))

    def _finish_cycle(
        self,
        timestamp: datetime,
        status: str = "completed",
        termination_reason: str = TerminationReason.TIMEOUT,
        keep_tail: bool = False,
        tail_cap: datetime | None = None,
    ) -> None:
        """Finalize cycle.

        Args:
            timestamp: Time of completion
            status: Cycle status string
            termination_reason: Reason for termination
            keep_tail: If True, use current timestamp as end time and preserve
                       trailing zero readings (e.g. Smart Termination).
                       If False (default), snap back to last active time and trim
                       trailing zeros (e.g. Timeout).
            tail_cap: Latest end time a kept tail may claim. Ignored when
                      ``keep_tail`` is False or when it is not earlier than
                      ``timestamp``; readings past it are dropped so the stored
                      trace and the stored duration stay consistent.
        """

        # Capture data before reset
        readings = self._power_readings
        if keep_tail:
            end_time = timestamp
            if tail_cap is not None and tail_cap < end_time:
                end_time = tail_cap
                readings = [r for r in readings if r[0] <= end_time]
        else:
            end_time = self._last_active_time or timestamp

        if not self._current_cycle_start:
            self.reset(timestamp=timestamp)
            return

        duration = (end_time - self._current_cycle_start).total_seconds()

        # "Interrupted" logic (short cycle etc)
        if duration < self._config.interrupted_min_seconds:
            status = "interrupted"
        elif duration < self._config.completion_min_seconds:
            status = "interrupted"

        # Trim leading/trailing zero readings for cleaner data
        # If we keep tail, we explicitly do NOT trim end zeros
        trimmed_readings = trim_zero_readings(
            readings,
            threshold=self._config.stop_threshold_w,
            trim_end=not keep_tail,
        )

        # Ensure power_data covers the full duration until end_time
        # (especially important for manual recordings or drying phases with no sensor updates)
        final_readings = list(trimmed_readings)
        if final_readings:
            last_t, last_p = final_readings[-1]
            if last_t < end_time:
                # Not at ACTIVE power when a capped tail ends past the last kept
                # sample: interpolation would read the whole drying allowance (up
                # to TERMINAL_QUIET_CAP_S) as full draw, in the stored trace and in
                # every envelope built from it. Readings past a cap at or after
                # `_last_active_time` are quiet by construction, so the first of
                # them is a real observation of this level - the rule the
                # banked-tail repair applies to the same point (register item 384).
                if last_p >= self._config.stop_threshold_w:
                    later = next(
                        (p for t, p in self._power_readings if t > end_time), None
                    )
                    if later is not None and later < self._config.stop_threshold_w:
                        last_p = later
                final_readings.append((end_time, last_p))

        start_ts = self._current_cycle_start.timestamp()
        # Store timestamps in canonical UTC (#369). Reading timestamps arrive from
        # dt_util.now() (HA-local-aware) while trim/split paths emit UTC, which left
        # past_cycles with a mix of offsets. Normalizing here (instant-preserving)
        # keeps stored cycles consistent and safe for cross-device/store transfer.
        cycle_data: dict[str, Any] = {
            "start_time": dt_util.as_utc(self._current_cycle_start).isoformat(),
            "end_time": dt_util.as_utc(end_time).isoformat(),
            "duration": duration,
            "max_power": self._cycle_max_power,
            "status": status,
            "termination_reason": termination_reason,
            "power_data": [[round(t.timestamp() - start_ts, 1), p] for t, p in final_readings],
        }
        spans = [
            [round(at, 1), round(n, 1)]
            for at, n in _merge_spans([*self._stall_spans, *self._user_pause_spans])
            if at < duration
        ]
        if spans:
            # Item 514: the finished stalls and user pauses, so the cycle-end label
            # match reads the trace in programme time (`match_rules.final_match_input`).
            cycle_data["halt_spans"] = spans

        self._logger.info("Cycle Finished: %s, %.1f min", status, duration / 60)
        self._on_cycle_end(cycle_data)

        target = STATE_FINISHED
        if status == "interrupted":
            target = STATE_INTERRUPTED
        elif status == "force_stopped":
            target = STATE_FORCE_STOPPED
        elif (
            status == "completed"
            and termination_reason in ANTI_WRINKLE_ELIGIBLE_REASONS
            and self._config.anti_wrinkle_enabled
            and self._config.device_type in (
                DEVICE_TYPE_WASHING_MACHINE,
                DEVICE_TYPE_DRYER,
                DEVICE_TYPE_WASHER_DRYER,
            )
        ):
            target = STATE_ANTI_WRINKLE

        self.reset(target_state=target, timestamp=timestamp)

    # Stub methods for compatibility or simpler logic
    def force_end(self, timestamp: datetime) -> None:
        """Force the cycle to end immediately."""
        if self._state != STATE_OFF:
            self._finish_cycle(
                dt_util.as_utc(timestamp),
                status="force_stopped",
                termination_reason=TerminationReason.FORCE_STOPPED,
                keep_tail=False,  # Force stop usually implies snap back to reality
            )
            self._ignore_power_until_idle = False

    def user_stop(self) -> None:
        """Handle user-initiated stop."""
        if self._state != STATE_OFF:
            now = utc_now()
            # "Done now" keeps the tail only while the machine is still running.
            # Pressed after the wash had already gone quiet - which is exactly when
            # a user stops it, because WashData missed the end - the whole wait was
            # banked as cycle time (30 min late -> a 60 min wash stored as 90) and
            # fed avg_duration like item 297's Smart tail. Cap it the same way
            # (audit DETECT-06).
            below_stop = bool(self._power_readings) and (
                self._power_readings[-1][1] < self._config.stop_threshold_w
            )
            self._finish_cycle(
                now,
                status="completed",
                termination_reason=TerminationReason.USER,
                keep_tail=True,  # User implies "Done Now"
                tail_cap=(
                    self._keep_tail_cap(self._current_cycle_start or now)
                    if below_stop
                    else None
                ),
            )
            # Prevent immediate restart if power is still high
            self._ignore_power_until_idle = True
            # Anchor the lockout clock to this stop instant. The next reading's
            # dt is measured from the last processed sample, which predates the
            # stop, so without this the high-power accumulator would count the
            # pre-stop gap and release the lockout early (#267).
            self._lockout_high_seconds = 0.0
            self._last_process_time = now


    def get_power_trace(self) -> list[tuple[datetime, float]]:
        """Return the current power trace."""
        return list(self._power_readings)

    def get_state_snapshot(self) -> dict[str, Any]:
        """Get a snapshot of the current state for persistence."""
        return {
            "state": self._state,
            "sub_state": self._sub_state,
            "current_cycle_start": (
                self._current_cycle_start.isoformat()
                if self._current_cycle_start
                else None
            ),
            "power_readings": [(t.isoformat(), p) for t, p in self._power_readings],
            "accumulated_energy_wh": self._energy_since_idle_wh,
            "time_above": self._time_above_threshold,
            "time_below": self._time_below_threshold,
            "time_below_gapfree": self._time_below_threshold_gapfree,
            "cycle_max_power": self._cycle_max_power,
            "last_active_time": (
                self._last_active_time.isoformat() if self._last_active_time else None
            ),
            "expected_duration": self._expected_duration,
            "matched_profile": self._matched_profile,
            "state_enter_time": (
                self._state_enter_time.isoformat() if self._state_enter_time else None
            ),
            "end_spike_seen": self._end_spike_seen,
            "end_spike_duration": self._end_spike_duration,
            "match_ambiguous": self._match_ambiguous,
            "match_prefix_ambiguous_full_shape": self._match_prefix_ambiguous_full_shape,
            "matched_tail_power": self._matched_tail_power,
            "matched_terminal_high": self._matched_terminal_high,
            "matched_terminal_quiet_s": self._matched_terminal_quiet_s,
            "matched_trusted_min_s": self._matched_trusted_min_s,
            # Element 14. A dishwasher restored into its terminal-tail match freeze
            # never re-matches, so without it the item-465 release floor was gone
            # for the rest of the cycle (the hazard gate only lost a shortening).
            "matched_pause_catalogue": (
                [
                    self._matched_pause_catalogue[0],
                    [list(p) for p in self._matched_pause_catalogue[1]],
                ]
                if self._matched_pause_catalogue is not None
                else None
            ),
            "longest_candidate_duration": self._longest_candidate_duration,
            "ml_defer_start_duration": self._ml_defer_start_duration,
            # Without it a restart dropped the confidence to 0.0: Smart Termination
            # was then blocked as low_confidence, and a dishwasher restored into
            # its terminal-tail match freeze could never re-match (DETECT-09).
            "last_match_confidence": self._last_match_confidence,
            # Register item 393a: the #296 tail's baseline and the idle clock of
            # the anti-wrinkle state. Without the floor a restart mid-tail fell
            # back to the configured burst length, which splits the tail again.
            "anticrease_tail_floor_w": self._anticrease_tail_floor_w,
            "anti_wrinkle_idle_time": self._anti_wrinkle_idle_time,
            # Item 511: the stalls already over, which the end gates' clock
            # excludes (the one in progress is re-derived from the trace).
            "stall_spans": [[at, length] for at, length in self._stall_spans],
            # Item 514: the user pauses already resumed from, and the current one.
            "user_pause_spans": [[at, length] for at, length in self._user_pause_spans],
            "user_pause_since": (
                self._user_pause_since.isoformat() if self._user_pause_since else None
            ),
        }

    @property
    def in_anticrease_tail(self) -> bool:
        """STATE_ANTI_WRINKLE entered by the #296 anti-crease finalise (item 393a)."""
        return (
            self._state == STATE_ANTI_WRINKLE
            and self._anticrease_tail_floor_w is not None
        )

    def get_elapsed_seconds(self) -> float:
        """Return seconds elapsed in current cycle."""
        if self._current_cycle_start:
            return (utc_now() - self._current_cycle_start).total_seconds()
        return 0.0

    def is_waiting_low_power(self) -> bool:
        """Return True if we are pending end/pause due to low power."""
        return (
            self._state in (STATE_RUNNING, STATE_PAUSED, STATE_ENDING)
            and self._time_below_threshold > 0
        )

    def restore_state_snapshot(self, snapshot: dict[str, Any]) -> bool:
        """Restore state from snapshot; False when it could not be restored.

        A snapshot that raises leaves the detector OFF (never half-restored) and is
        logged with its traceback. The caller keeps the snapshot for diagnostics
        instead of deleting it (register item 266 follow-up): until then a restore
        that raised lost the running cycle with one ERROR line and no record.
        """
        try:
            self._state = snapshot.get("state", STATE_OFF)
            self._sub_state = snapshot.get("sub_state")
            # Not persisted (item 501): a restored probe is shown.
            self._standby_reprobe = False
            self._probe_hidden = False
            self._energy_since_idle_wh = snapshot.get("accumulated_energy_wh", 0.0)
            self._time_above_threshold = snapshot.get("time_above", 0.0)
            self._time_below_threshold = snapshot.get("time_below", 0.0)
            # Old snapshots lack the gap-free tally, and the plain value they do
            # carry may already include outage-sized intervals — the exact
            # contamination this field exists to exclude — so it must NOT be used
            # as the fallback. 0.0 is also the honest value on a restore in
            # general: the restart itself is unobserved time (the manager records
            # it as a restart gap), so no quiet observed before it still counts.
            self._time_below_threshold_gapfree = float(
                snapshot.get("time_below_gapfree", 0.0) or 0.0
            )
            # Not persisted: the restart's own gap is in neither tally either.
            self._time_below_unobserved = 0.0
            self._cycle_max_power = snapshot.get("cycle_max_power", 0.0)
            # Sanitize via the same helper as update_match so the class
            # invariant on _expected_duration holds across restarts and the
            # gates in STATE_ENDING / _should_defer_finish can trust the value.
            # If sanitization rejects the snapshot's expected_duration, also
            # clear the matched_profile so we don't restore a half-valid state
            # where Smart Termination can fire on _expected_duration == 0.0.
            restored_match = snapshot.get("matched_profile")
            sanitized_expected = self._sanitize_expected_duration(
                snapshot.get("expected_duration", 0.0),
                source="restore_state_snapshot",
            )
            if (
                restored_match is not None
                and sanitized_expected == self._SANITIZE_INVALID_SENTINEL
            ):
                self._logger.debug(
                    "restore_state_snapshot: dropping matched_profile %r "
                    "because expected_duration sanitized to invalid sentinel",
                    restored_match,
                )
                self._matched_profile = None
            else:
                self._matched_profile = restored_match
            self._expected_duration = sanitized_expected
            self._end_spike_seen = snapshot.get("end_spike_seen", False)
            self._end_spike_duration = float(snapshot.get("end_spike_duration", 0.0))
            self._match_ambiguous = snapshot.get("match_ambiguous", False)
            # A pre-#364 snapshot has no narrow flag: fall back to the old single
            # `match_prefix_ambiguous` (the #364 prefix-fit flag, removed in 0.5.8,
            # and before #364 the #288 verdict itself) so a restart cannot loosen
            # the anti-crease gate. Newer snapshots no longer write that key.
            self._match_prefix_ambiguous_full_shape = snapshot.get(
                "match_prefix_ambiguous_full_shape",
                snapshot.get("match_prefix_ambiguous", False),
            )
            self._matched_tail_power = self._sanitize_tail_power(
                snapshot.get("matched_tail_power")
            )
            self._matched_terminal_high = self._sanitize_terminal_high(
                snapshot.get("matched_terminal_high")
            )
            self._matched_terminal_quiet_s = self._sanitize_terminal_quiet(
                snapshot.get("matched_terminal_quiet_s")
            )
            self._matched_trusted_min_s = self._sanitize_trusted_min(
                snapshot.get("matched_trusted_min_s")
            )
            self._matched_pause_catalogue = self._sanitize_pause_catalogue(
                snapshot.get("matched_pause_catalogue")
            )
            # Unconditionally, because the hazard is the value already on the
            # object, not the one in the snapshot: this restores the ambiguity
            # flags the ENDING gate reads, so leaving the bound untouched pairs
            # them with whatever a previous cycle left behind. A snapshot written
            # before this key existed yields 0.0, i.e. the old refusal.
            self._longest_candidate_duration = self._sanitize_longest_candidate(
                snapshot.get("longest_candidate_duration")
            )
            self._ml_defer_start_duration = snapshot.get("ml_defer_start_duration")
            try:
                self._last_match_confidence = float(
                    snapshot.get("last_match_confidence") or 0.0
                )
            except (TypeError, ValueError, OverflowError):
                self._last_match_confidence = 0.0
            # Item 393a. Meaningful only in the tail; an older snapshot (or any
            # other state) yields None / 0.0, i.e. the configured burst length.
            # The burst candidate is not restored: the restart broke the run.
            in_tail = self._state == STATE_ANTI_WRINKLE
            self._anticrease_tail_floor_w = (
                self._sanitize_positive(snapshot.get("anticrease_tail_floor_w"))
                if in_tail else None
            )
            self._anti_wrinkle_idle_time = (
                self._sanitize_positive(snapshot.get("anti_wrinkle_idle_time")) or 0.0
                if in_tail else 0.0
            )
            self._anti_wrinkle_candidate_start = None
            self._anti_wrinkle_candidate_peak = 0.0
            self._anti_wrinkle_candidate_start_power = 0.0

            # Restore state enter time and recompute time_in_state from it
            enter_time = snapshot.get("state_enter_time")
            if enter_time:
                try:
                    self._state_enter_time = dt_util.parse_datetime(enter_time)
                    if self._state_enter_time:
                        if self._state_enter_time.tzinfo is None:
                            self._state_enter_time = self._state_enter_time.replace(
                                tzinfo=dt_util.now().tzinfo
                            )
                        self._state_enter_time = dt_util.as_utc(self._state_enter_time)
                        elapsed = (utc_now() - self._state_enter_time).total_seconds()
                        self._time_in_state = max(0.0, elapsed)
                except Exception: # pylint: disable=broad-exception-caught
                    self._logger.warning("Failed to parse state enter time")

            start = snapshot.get("current_cycle_start")
            self._current_cycle_start = None
            if start:
                try:
                    dt_start = dt_util.parse_datetime(start)
                    if dt_start and dt_start.tzinfo is None:
                        # Fix Naive Timestamp (Legacy Data)
                        dt_start = dt_start.replace(tzinfo=dt_util.now().tzinfo)
                        self._logger.warning("Restored Naive start_time, assuming local: %s", dt_start)
                    self._current_cycle_start = dt_util.as_utc(dt_start) if dt_start else None
                except Exception:  # pylint: disable=broad-exception-caught
                    self._logger.warning("Failed to parse start time: %s", start)

            readings = snapshot.get("power_readings", [])
            self._power_readings = []

            # Detect naive readings once
            has_naive_readings = False

            for r in readings:
                if isinstance(r, (list, tuple)):
                    reading = cast(list[Any] | tuple[Any, ...], r)
                    if len(reading) < 2:
                        continue
                    try:
                        t = dt_util.parse_datetime(str(reading[0]))
                        if t:
                            if t.tzinfo is None:
                                t = t.replace(tzinfo=dt_util.now().tzinfo)
                                has_naive_readings = True
                            value = float(reading[1])
                            if math.isfinite(value):
                                self._power_readings.append((dt_util.as_utc(t), value))
                    except (TypeError, ValueError, OverflowError) as exc:
                        self._logger.debug("Skipping malformed power reading %s: %s", r, exc)

            if has_naive_readings:
                self._logger.warning(
                    "Restored %d power readings with Naive timestamps (fixed to local)",
                    len(self._power_readings),
                )

            # Restore last active
            last_active = snapshot.get("last_active_time")
            if last_active:
                dt_last = dt_util.parse_datetime(last_active)
                if dt_last and dt_last.tzinfo is None:
                    dt_last = dt_last.replace(tzinfo=dt_util.now().tzinfo)
                self._last_active_time = dt_util.as_utc(dt_last) if dt_last else None
            else:
                self._last_active_time = self._current_cycle_start
            # #452: a stall survives a restart from the restored trace (element
            # 15 is not persisted: the next match tick supplies it again).
            self._stall_spans = self._sanitize_stall_spans(snapshot.get("stall_spans"))
            self._user_pause_spans = self._sanitize_stall_spans(
                snapshot.get("user_pause_spans")
            )
            _since = snapshot.get("user_pause_since")
            _since_dt = dt_util.parse_datetime(_since) if isinstance(_since, str) else None
            self._user_pause_since = dt_util.as_utc(_since_dt) if _since_dt else None
            self._seed_stall_run()

        except Exception:  # pylint: disable=broad-exception-caught
            self.restore_error = traceback.format_exc()
            self._logger.warning(
                "Could not restore the active-cycle snapshot (state %r, cycle start "
                "%r); starting from OFF",
                snapshot.get("state") if isinstance(snapshot, dict) else None,
                snapshot.get("current_cycle_start") if isinstance(snapshot, dict) else None,
                exc_info=True,
            )
            self.reset()
            return False
        self.restore_error = None
        return True