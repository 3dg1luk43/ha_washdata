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
"""Pure live-match DECISIONS: what the manager does with each match result.

Single source of truth for the rules between "the matcher returned a result" and
"the detector receives it" (audit PLAYGROUND-01/02/03):

* the program switching state machine - divergence revert, temporal persistence,
  the initial commit (0.15 / unmatch-threshold floor; an ambiguous winner waits
  ``MATCH_AMBIGUOUS_COMMIT_FACTOR`` x persistence), the decisive-margin switch and
  the persistent switch (clear lead or rising trend), the unmatch revert, the
  score history;
* the envelope verified pause - set on a confirmed expected low-power region,
  released at 95% of the envelope span, on high power, by the #375 sustained-quiet
  release, after a bounded quiet once a revoke has orphaned it (item 498), and
  forced on by a user pause;
* the consistency override, which is also how a confident mismatch (every
  candidate rejected) drops the displayed program;
* the cycle-end label verdict.

``manager.WashDataManager._async_do_perform_matching`` and the Playground's
``playground._DetailSim._matcher`` both call these, so a replay makes the same
decisions the running integration makes. Everything asynchronous (the matcher,
the envelope alignment check) stays with the caller, which passes the result in.

Nothing here touches Home Assistant. Log lines are returned as
``(level, message, args)`` tuples for the caller to emit, so the manager's log
output is unchanged and the Playground can drop it. Moved verbatim from
``manager.py``: every constant, ordering and edge case is the manager's. The
functions do not raise on well-formed matcher output (the manager coerces
``profile_unmatch_threshold`` with ``option_float`` before passing it in).
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .const import (
    DEFAULT_MATCH_REVERT_RATIO,
    END_GATE_HAZARD_MARGIN,
    ENDING_HARD_FINALIZE_MIN_QUIET_S,
    MATCH_AMBIGUOUS_COMMIT_FACTOR,
    MATCH_DECISIVE_MARGIN,
    MATCH_SURE_KNOTS,
    MATCH_SURE_SINGLE_CANDIDATE,
    ORPHANED_PAUSE_MAX_WAIT_S,
)

#: The manager's "no program committed yet" placeholder.
DETECTING = "detecting..."
#: Program values that are not a committed program for the switching rules.
UNCOMMITTED_PROGRAMS = ("detecting...", "off", "starting", "unknown")
#: Program values for which the detector is told nothing is committed (LIVE-17).
_NOT_COMMITTED_FOR_DETECTOR = ("detecting...", "restored...", "off", "starting", "unknown", None)

#: A log line for the caller to emit: ``(level, message, args)``.
LogLine = tuple[int, str, tuple[Any, ...]]


@dataclass
class SwitchState:
    """The manager's per-cycle switching fields, by the names the rules use.

    The manager keeps its own attributes as the source of truth and round-trips
    them through this object (``manager._read_switch_state`` / ``_write_switch_state``);
    the Playground keeps one of these per replay. The two dicts are shared by
    reference, exactly as the manager's own attributes were mutated in place.
    """

    current_program: Any = "off"
    matched_duration: float | None = None
    last_confidence: float = 0.0
    last_member_confidence: float | None = None
    score_history: dict[str, list[float]] = field(default_factory=dict)
    persistence_counter: dict[str, int] = field(default_factory=dict)
    unmatch_counter: int = 0
    current_candidate: str | None = None

    def start_cycle(self) -> None:
        """The reset ``manager._on_state_change`` applies when a new cycle runs."""
        self.current_program = DETECTING
        self.last_confidence = 0.0
        self.last_member_confidence = None
        self.matched_duration = None
        self.score_history = {}
        self.persistence_counter = {}
        self.unmatch_counter = 0
        self.current_candidate = None


@dataclass
class MatchTick:
    """One live match as the switching rules read it, plus what they decided."""

    profile_name: str | None
    confidence: float
    matched_duration: Any
    phase_name: str | None
    current_duration: float
    current_program_score: Any
    match_margin: float
    is_persistent: bool
    should_switch: bool = False
    switch_reason: str = ""
    log: list[LogLine] = field(default_factory=list)


@dataclass
class PauseDecision:
    """The verified-pause flag to push to the detector, and the envelope position.

    ``envelope_position`` is None when this tick did not compute one, in which case
    the manager leaves its attribute as it was.
    """

    verified_pause: Any
    envelope_position: float | None = None
    log: list[LogLine] = field(default_factory=list)


def profile_duration(value: Any) -> float | None:
    """A profile's expected duration in seconds, or None when unusable.

    None is the field's declared "unknown" and every reader of the matched
    duration already guards for it. Non-finite is rejected because ``inf``
    survives a plain ``> 0`` test and ``sensor.py``'s ``int(time_remaining / 60)``
    then raises OverflowError (register items 211/229). OverflowError is caught
    because ``float(10**400)`` raises rather than returning ``inf`` (items 279/280).
    """
    try:
        avg = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(avg) or avg <= 0:
        return None
    return avg


def analyze_trend(history: list[float]) -> bool:
    """True if the score rose in at least 7 of the last 10 intervals.

    Requires at least 5 samples of history to make a determination.
    """
    if len(history) < 5:
        return False
    # Use last 11 points to get 10 intervals (or fewer if history short)
    recent = history[-11:]
    if len(recent) < 2:
        return False
    up_count = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i - 1])
    total_intervals = len(recent) - 1
    # Proportional threshold (7/10 => 0.7)
    return (up_count / total_intervals) >= 0.70


def program_is_committed(program: Any) -> bool:
    """Whether the detector should be told a program is committed (audit LIVE-17)."""
    return program not in _NOT_COMMITTED_FOR_DETECTOR


def detector_match(tick: "MatchTick", result: Any) -> tuple[str | None, bool]:
    """``(profile_name, revoke)`` for the detector's ``update_match`` this tick.

    A divergence revert leaves ``tick.profile_name == "detecting..."``. Handed to the
    detector as is, that string became its matched programme, carrying the
    abandoned winner's expected duration, so Smart Termination and the dishwasher
    freeze kept acting on a match the manager had just dropped (audit F7 finding).
    A revert now revokes the detector's match exactly as a confident mismatch does.
    """
    if tick.profile_name == DETECTING:
        return None, True
    return tick.profile_name, bool(getattr(result, "is_confident_mismatch", False))


def begin_tick(
    state: SwitchState, result: Any, persistence: int, current_duration: float
) -> MatchTick:
    """Read a match result, apply the divergence revert, advance persistence.

    Mutates ``state``. Returns the tick the rest of the rules read; its
    ``profile_name`` becomes ``"detecting..."`` when divergence reverted the
    program (the detector is told through :func:`detector_match`).
    """
    log: list[LogLine] = []
    profile_name = result.best_profile
    confidence = result.confidence

    # Identify current program score from results. Read from the COLLAPSED top 5,
    # so a displayed member of a cohesive Stage-5 family (whose record carries the
    # picked member's or the `__group__` name) and a program ranked sixth read 0.0.
    # Reading every member's own pre-collapse score instead (audit MATCH-DECIDE-03)
    # was measured and NOT shipped: on the two corpus exports with user groups the
    # committed program at cycle end was right 55 -> 53 of 116 (end_gate_eval --loo
    # --all-formats: 333 -> 331 of 460, end timing identical), for 169 -> 163
    # displayed-program changes. Both losses were 0.0 reads switching the display
    # on real margins of 0.002-0.038 that happened to land on the label.
    current_program_score: Any = 0.0
    for c in result.candidates:
        if c.get("name") == state.current_program:
            current_program_score = c.get("score", 0.0)
            break

    # How far clear of the runner-up the winner is. Measured over 594 cycles x 10
    # checkpoints, this separates right from wrong far better than the absolute
    # score does mid-cycle (AUC 0.773 vs 0.535), which is why the mid-cycle switch
    # keys on it. Register item 305. Measured against the best OTHER candidate
    # rather than by list index, so it does not depend on how the Stage-5 collapse
    # orders the rebuilt candidate list.
    match_margin = 1.0
    runner_up = None
    for c in result.candidates:
        if c.get("name") == profile_name:
            continue
        try:
            cs = float(c.get("score", 0.0) or 0.0)
        except (TypeError, ValueError, OverflowError):
            continue
        if runner_up is None or cs > runner_up:
            runner_up = cs
    if runner_up is not None:
        match_margin = float(confidence) - runner_up

    # CASE: Divergence Detection (Score Drop)
    # If the current matched program has a significant drop from its own peak
    # score, consider unmatching it even if it's still the "best" candidate. This
    # catches divergence faster than waiting for the fixed unmatch threshold.
    if (
        state.current_program not in UNCOMMITTED_PROGRAMS
        and profile_name == state.current_program
    ):
        history: list[float] = state.score_history.get(state.current_program, [])
        if len(history) > 3:
            peak_score = max(history)
            if confidence < peak_score * (1.0 - DEFAULT_MATCH_REVERT_RATIO):
                state.unmatch_counter += 1
                if state.unmatch_counter >= persistence:
                    state.current_program = DETECTING
                    state.matched_duration = None
                    state.unmatch_counter = 0
                    state.persistence_counter.pop(profile_name, None)
                    state.current_candidate = None
                    log.append((
                        logging.INFO,
                        "Divergence detected for profile '%s' (confidence %.3f < 60%% of peak %.3f). "
                        "Reverting to detection.",
                        (profile_name, confidence, peak_score),
                    ))
                    # Reset profile_name so Case 3 doesn't re-trigger
                    profile_name = DETECTING

    # Update persistence for the best profile
    if profile_name and profile_name != DETECTING:
        state.persistence_counter[profile_name] = state.persistence_counter.get(profile_name, 0) + 1
        # Check if this is the same candidate as before
        if profile_name != state.current_candidate:
            # Reset counter for old candidate if it wasn't locked in
            state.current_candidate = profile_name
            state.persistence_counter[profile_name] = 1
    else:
        state.current_candidate = None

    is_persistent = bool(
        profile_name and state.persistence_counter.get(profile_name, 0) >= persistence
    )
    return MatchTick(
        profile_name=profile_name,
        confidence=confidence,
        matched_duration=result.expected_duration,
        phase_name=result.matched_phase,
        current_duration=current_duration,
        current_program_score=current_program_score,
        match_margin=match_margin,
        is_persistent=is_persistent,
        log=log,
    )


def decide_switch(
    state: SwitchState,
    tick: MatchTick,
    result: Any,
    persistence: int,
    unmatch_threshold: Any,
) -> list[LogLine]:
    """Cases 1-3 and the switch itself. Mutates ``state`` and ``tick``.

    ``unmatch_threshold`` is passed exactly as the manager holds it (the raw
    option): Case 1 floors it through ``float``, Case 3 compares it as is.
    Returns this step's log lines (``tick.log`` holds :func:`begin_tick`'s).
    """
    log: list[LogLine] = []
    profile_name = tick.profile_name
    confidence = tick.confidence
    is_persistent = tick.is_persistent
    current_program_score = tick.current_program_score
    should_switch = False
    switch_reason = ""

    # Case 1: Initial Match from "detecting..."
    # The commit floor is at least the unmatch threshold: committing a 0.15-0.35
    # match only for Case 3 to drop it four ticks later (and the still-persistent
    # counter to re-commit it at once) made a stable low-confidence top-1 flip
    # program <-> "detecting..." every few ticks, dropping the ETA each time
    # (audit MATCH-DECIDE-05; 38 of 580 cycles).
    # An AMBIGUOUS winner waits MATCH_AMBIGUOUS_COMMIT_FACTOR x as many wins. A
    # near-tie early in a wash is mostly a prefix of one programme resembling
    # another, so committing it at plain persistence showed the wrong programme
    # (and its ETA) first on most washer cycles; the Status card says "Uncertain:
    # X or Y" meanwhile. Not "never": a stable winner of an always-close pair must
    # still get a programme and an ETA. Measured in const.py.
    if (
        profile_name
        and confidence >= max(0.15, float(unmatch_threshold or 0.0))
        and (not result.is_ambiguous or is_persistent)
        and (not state.matched_duration or state.current_program == DETECTING)
    ):
        wins = state.persistence_counter.get(profile_name, 0)
        needed = persistence * (MATCH_AMBIGUOUS_COMMIT_FACTOR if result.is_ambiguous else 1)
        if is_persistent and wins >= needed:
            should_switch = True
            switch_reason = f"initial_match (persistent {wins}x)"
        else:
            log.append((
                logging.DEBUG,
                "Match persistence: %s at %d/%d matches%s. Stay at detecting...",
                (profile_name, wins, needed, " (ambiguous)" if result.is_ambiguous else ""),
            ))

    # Case 2: Mid-cycle override (different profile)
    elif (
        profile_name
        and state.current_program != profile_name
        and state.current_program not in UNCOMMITTED_PROGRAMS
    ):
        # Decisive Margin Override: bypass persistence when the winner is far clear
        # of the runner-up (register item 305). It replaces a `confidence > 0.8`
        # override whose premise is backwards mid-cycle: the query is a PREFIX, and
        # a prefix of a long programme looks exactly like a finished short one, so
        # a score above 0.8 measured 31.5% correct (n=73). Keyed on the margin
        # (mid-cycle AUC 0.773 vs 0.535), replaying 594 cycles lifted end-of-cycle
        # correctness 70.4% -> 72.6% (McNemar p = 0.0044) for 0.14 displayed
        # switches per cycle. The `> current_program_score` guard measured neutral
        # and is kept because switching to something scoring below what is already
        # displayed is never right.
        # The 1.0 sentinel the margin carries when nothing else scored is
        # LOAD-BEARING, not a gap: `devtools/decisive_margin_eval.py --loo` measured
        # a single surviving candidate as the correct programme 96.3% (361/375) of the
        # time, against 87.8% (1028/1171) for the real-margin bypass, over 2636
        # leave-one-out checkpoints on the shipped matcher (99.4% vs 90.0% where the
        # programme keeps another cycle; in-sample flattered both to 99.5% / 92.5%;
        # PR #448 round 6 had 94.0% / 77.8% on a harness that skipped Stage-1
        # re-gridding). Stage 1/2 rejecting every other profile is evidence.
        if (
            tick.match_margin > MATCH_DECISIVE_MARGIN
            and confidence > current_program_score
        ):
            should_switch = True
            switch_reason = (
                f"decisive_margin (margin {tick.match_margin:.3f} > "
                f"{MATCH_DECISIVE_MARGIN}, {confidence:.3f} vs {current_program_score:.3f})"
            )

        # Normal Switch: a persistent challenger that beats the displayed programme
        # by more than 0.05 (against flapping) and either leads its runner-up
        # clearly (not ambiguous) or has a rising score. Until 0.5.8 only the
        # rising score qualified, so a challenger that led clearly tick after tick
        # but held a flat score was never adopted. Measured alone
        # (decisive_margin_eval.py --switching --loo, 291 cycles): programme shown
        # at cycle end right on washers 50.9 -> 55.3%, dishwashers unchanged
        # (96.9%), washer switches per cycle 1.32 -> 1.47; with the commit rule
        # above 1.25, and end timing unchanged (end_gate_eval --loo --all-formats:
        # 1 of 472 ends moved, 3.5 min earlier).
        elif is_persistent:
            if confidence > current_program_score and (
                not result.is_ambiguous
                or analyze_trend(state.score_history.get(profile_name, []))
            ):
                if (confidence - current_program_score) > 0.05:
                    should_switch = True
                    switch_reason = (
                        f"{'clear_lead' if not result.is_ambiguous else 'positive_trend'}"
                        f"_persistent ({confidence:.3f} > {current_program_score:.3f})"
                    )

    # Case 3: Unmatching (confidence drop)
    elif (
        state.current_program not in UNCOMMITTED_PROGRAMS
        and profile_name == state.current_program
        and confidence < unmatch_threshold
    ):
        state.unmatch_counter += 1
        is_unmatch_persistent = state.unmatch_counter >= persistence

        if is_unmatch_persistent:
            state.current_program = DETECTING
            state.matched_duration = None
            state.unmatch_counter = 0
            # Start persistence over: left at its locked value the very next tick
            # re-committed the profile just dropped.
            state.persistence_counter.pop(profile_name, None)
            state.current_candidate = None
            log.append((
                logging.INFO,
                "Unmatched profile '%s' (confidence %.3f < threshold %.3f persistent %dx). "
                "Reverting to detection.",
                (profile_name, confidence, unmatch_threshold, persistence),
            ))
        else:
            log.append((
                logging.DEBUG,
                "Unmatch persistence: %s at %d/%d low-confidence matches. Stay at %s...",
                (profile_name, state.unmatch_counter, persistence, profile_name),
            ))

    # Reset unmatch counter if confidence is healthy
    # AND we didn't just detect a divergence
    elif (
        profile_name == state.current_program
        and confidence >= unmatch_threshold
        and not (
            len(state.score_history.get(state.current_program, [])) > 3
            and confidence < max(state.score_history[state.current_program]) * (1.0 - DEFAULT_MATCH_REVERT_RATIO)
        )
    ):
        state.unmatch_counter = 0

    if should_switch:
        if profile_name is None:
            state.current_program = DETECTING
        else:
            state.current_program = profile_name
        state.last_confidence = confidence
        state.last_member_confidence = result.member_confidence
        state.unmatch_counter = 0  # Reset on switch
        if profile_name in state.persistence_counter:
            state.persistence_counter[profile_name] = persistence  # Lock it in

        state.matched_duration = profile_duration(tick.matched_duration)
        avg_duration = state.matched_duration or 0.0
        log.append((
            logging.INFO,
            "Switching to profile '%s' (reason: %s). Expected duration: %.0fs (%smin)",
            (profile_name, switch_reason, avg_duration, int(avg_duration / 60)),
        ))
    elif profile_name == state.current_program:
        # Same program, but update confidence for sensors
        state.last_confidence = confidence
        state.last_member_confidence = result.member_confidence
    elif not state.matched_duration:
        state.current_program = DETECTING

    tick.should_switch = should_switch
    tick.switch_reason = switch_reason
    return log


def record_scores(state: SwitchState, candidates: Any) -> None:
    """Append every candidate's score to its history (trend analysis, last 20)."""
    for cand in candidates:
        cname = cand.get("name")
        if cname:
            history = state.score_history.setdefault(cname, [])
            history.append(float(cand.get("score", 0.0)))
            if len(history) > 20:
                history.pop(0)


def needs_alignment_check(
    current_matched: Any, current_power: float, stop_threshold_w: float, user_paused: bool
) -> bool:
    """Whether to run the envelope alignment check this tick.

    Only while the detector holds a match and power is low, to confirm whether this
    is a legitimate (auto-detected) pause or a mismatch. Skipped while the user has
    explicitly paused (issue #306): the user pause is authoritative and must not be
    re-judged by the envelope heuristic.
    """
    return bool(current_matched and current_power < stop_threshold_w and not user_paused)


def decide_alignment_pause(
    *,
    verified_pause: Any,
    current_matched: Any,
    alignment: tuple[bool, float] | None,
    envelope_span: Callable[[Any], float],
) -> PauseDecision:
    """Step 1 of the verified pause: what the envelope alignment says.

    ``verified_pause`` is the detector's current flag. ``alignment`` is
    ``(is_confirmed, mapped_time)`` from ``ProfileStore.async_verify_alignment``
    when :func:`needs_alignment_check` said to run it, else None (flag unchanged).
    The caller applies ``envelope_position`` before step 2, as the manager always
    stored it before reading anything else.
    """
    log: list[LogLine] = []
    position: float | None = None
    if alignment is not None:
        is_confirmed, mapped_time = alignment
        if is_confirmed:
            if not verified_pause:
                log.append((
                    logging.INFO,
                    "Envelope verified expected low power phase for %s. Enabling verified pause.",
                    (current_matched,),
                ))
            verified_pause = True
            # Smart Termination within Envelope block. Compare the mapped position
            # against the envelope's OWN time span (not avg_duration, a differently
            # derived trimmed mean): mapped_time is capped at the grid span, so
            # span/avg_duration < 1 would make the 0.95 release unreachable and the
            # cycle would hang to the deferral cap (#348).
            try:
                span = envelope_span(current_matched)
                if span > 0:
                    # The only continuous "how far through this programme are we"
                    # figure independent of elapsed time (state attribute).
                    position = round(min(1.0, max(0.0, mapped_time / span)), 3)
                if span > 0 and (mapped_time / span) > 0.95:
                    verified_pause = False
                    log.append((
                        logging.INFO,
                        "Smart Termination: near end of profile (%.0f/%.0fs). Releasing pause lock.",
                        (mapped_time, span),
                    ))
                else:
                    # Diagnostic (#346): the release is held; show how far the trace
                    # mapped vs the 95% release point (no behaviour change).
                    log.append((
                        logging.DEBUG,
                        "Smart Termination held for %s: mapped %.0f/%.0fs (%.0f%%) below 95%% release%s",
                        (
                            current_matched, mapped_time, span,
                            (100.0 * mapped_time / span) if span > 0 else 0.0,
                            "" if span > 0 else " (envelope span unavailable)",
                        ),
                    ))
            except Exception as e:  # pylint: disable=broad-exception-caught
                log.append((
                    logging.DEBUG,
                    "Smart Termination alignment verification failed: %s",
                    (e,),
                ))
        else:
            if verified_pause:
                log.append((
                    logging.INFO,
                    "Envelope indicates UNEXPECTED low power for %s. Disabling verified pause.",
                    (current_matched,),
                ))
            verified_pause = False
    return PauseDecision(verified_pause=verified_pause, envelope_position=position, log=log)


def decide_pause_release(
    *,
    verified_pause: Any,
    current_matched: Any,
    current_power: float,
    stop_threshold_w: float,
    user_paused: bool,
    expected_duration: float,
    current_duration: float,
    time_below: float,
    program: Any,
) -> PauseDecision:
    """Step 2 of the verified pause: the releases, then the user-pause override.

    ``time_below`` is the detector's GAP-FREE sub-threshold tally: a telemetry
    outage is unobserved time and must not satisfy the quiet floor of the #375
    release. ``program`` only names the cycle in the log line.
    """
    log: list[LogLine] = []

    # --- High Power Clear ---
    if current_power > stop_threshold_w * 10:
        verified_pause = False

    # --- Sustained-quiet release of an auto-detected pause (issue #375) ---
    # When the appliance goes truly silent at the real end, the envelope alignment
    # can keep re-confirming against a long near-zero drying tail baked into the
    # profile by earlier force-stopped cycles, and the >95%-of-span release is
    # unreachable because the trace goes quiet BEFORE that learned tail ends. The
    # flag then freezes True and every ENDING finalize backstop (all gated on
    # `not _verified_pause`) is defeated, so the cycle hangs until the watchdog's
    # silence limit. Release it once the cycle has completed its expected duration
    # AND has been continuously sub-threshold for the finalize quiet floor. A real
    # user pause is authoritative and re-asserted below, so it is never released.
    if (
        verified_pause
        and not user_paused
        and expected_duration > 0
        and current_duration >= expected_duration
        and time_below >= ENDING_HARD_FINALIZE_MIN_QUIET_S
    ):
        log.append((
            logging.INFO,
            "Releasing auto-detected pause for %s: reached expected "
            "duration (%.0fs >= %.0fs) and sustained-quiet %.0fs - "
            "allowing normal cycle finish (issue #375).",
            (current_matched or program, current_duration, expected_duration, time_below),
        ))
        verified_pause = False

    # A user-initiated pause (Pause Cycle button, or the door-open soft pause)
    # stays in force until the user resumes (issue #306). Re-asserting here also
    # repairs the flag after a restart, since the detector state snapshot does not
    # persist _verified_pause.
    if user_paused:
        verified_pause = True

    return PauseDecision(verified_pause=verified_pause, log=log)


#: Register item 498: release a verified pause a revoke has orphaned, after
#: :func:`orphaned_pause_wait_s`. A module flag so the A/B and the revert check can
#: turn it off; always on in production.
RELEASE_ORPHANED_PAUSE = True


def orphaned_pause_wait_s(*, off_delay: Any, min_off_gap: Any, longest_pause_s: Any) -> float:
    """How much gap-free quiet an orphaned verified pause may hold the end for.

    A revoke (divergence revert, or every candidate rejected) drops the match and
    its expected duration but leaves the envelope's verified pause, which neither
    the 95%-of-span release (it needs the match) nor the #375 release (it needs the
    expected duration) can then clear, so it held the cycle until the force stop.

    ``longest_pause_s`` is the longest below-stop pause the revoked programme's
    traced cycles ever resumed from (its pause catalogue, the hazard gate's evidence,
    position-free because the expected duration that placed it is gone). The wait is
    ``END_GATE_HAZARD_MARGIN`` x that, so a soak the device has recorded is still
    bridged, held to ``[floor, ORPHANED_PAUSE_MAX_WAIT_S]``. The floor is
    ``max(off_delay, min_off_gap, ENDING_HARD_FINALIZE_MIN_QUIET_S)``: what the
    unmatched fallback waits anyway, and the #375 release's quiet floor, so a release
    can never end a cycle sooner than if no pause had engaged. The floor wins over
    the cap. Never raises.
    """
    try:
        floor = max(
            float(off_delay or 0.0), float(min_off_gap or 0.0), ENDING_HARD_FINALIZE_MIN_QUIET_S
        )
    except (TypeError, ValueError, OverflowError):
        floor = ENDING_HARD_FINALIZE_MIN_QUIET_S
    if not math.isfinite(floor):
        floor = ENDING_HARD_FINALIZE_MIN_QUIET_S
    try:
        evidence = END_GATE_HAZARD_MARGIN * max(0.0, float(longest_pause_s or 0.0))
    except (TypeError, ValueError, OverflowError):
        evidence = 0.0
    if not math.isfinite(evidence):
        evidence = 0.0
    return max(floor, min(ORPHANED_PAUSE_MAX_WAIT_S, evidence))


def decide_orphaned_pause_release(
    *,
    verified_pause: Any,
    user_paused: bool,
    current_matched: Any,
    time_below: float,
    wait_s: float,
    longest_pause_s: float,
) -> PauseDecision:
    """Release a verified pause no match stands behind any more (register item 498).

    Only an AUTOMATIC pause (a user pause is authoritative, issue #306) with no
    matched programme, once the GAP-FREE quiet tally reaches ``wait_s``
    (:func:`orphaned_pause_wait_s`): a telemetry outage is unobserved time, as for
    the #375 release. A new match ends the orphan state and the normal releases
    apply again. Run by the detector on every ENDING reading, which the manager's
    readings and watchdog keepalives and the Playground's replay all reach.
    """
    if (
        RELEASE_ORPHANED_PAUSE
        and verified_pause
        and not user_paused
        and not current_matched
        and time_below >= wait_s
    ):
        return PauseDecision(
            verified_pause=False,
            log=[(
                logging.INFO,
                "Releasing auto-detected pause left by a revoked match: quiet %.0fs >= "
                "%.0fs (longest recorded pause of the revoked programme %.0fs) - "
                "allowing normal cycle finish (item 498).",
                (time_below, wait_s, longest_pause_s),
            )],
        )
    return PauseDecision(verified_pause=verified_pause)


#: Register item 469(b): an AMBIGUOUS tick in ENDING cannot defer the end (here
#: for the verified pause, ``CycleDetector.ambiguous_ending_match_defers`` for the
#: match). A module flag so the A/B and the revert check can turn both off; always
#: on in production.
HOLD_AMBIGUOUS_IN_ENDING = True


def hold_in_ending(
    *,
    ending: bool,
    is_ambiguous: bool,
    current_matched: Any,
    prev_verified: Any,
    verified_pause: Any,
    user_paused: bool,
) -> PauseDecision:
    """An ambiguous tick in ENDING engages no verified pause (register item 469b).

    Once the detector is in ENDING the run has gone quiet and the end gates are
    counting. A tick there matches a trace that ends in that idle tail, and when its
    top-1 is within ``MATCH_AMBIGUITY_MARGIN`` of its runner-up it is not evidence of
    anything (it licenses no switch and no label). Its envelope alignment could still
    engage a verified pause, which blocks every ENDING finalize: on the AK Willows
    washer-dryer db46776df845 it engaged on the reading the timeout would have fired
    on once "Apply all" set a 45 s match interval (lag 7.0 -> 17.0 min).

    So a pause that was not already on stays off. A release still applies, an
    engaged pause stays, and a user pause is authoritative. The same tick's match
    reaches the detector, which refuses it only if it would wait longer
    (``CycleDetector.ambiguous_ending_match_defers``). Shared by the manager and the
    Playground replay.
    """
    if (
        HOLD_AMBIGUOUS_IN_ENDING
        and ending
        and is_ambiguous
        and current_matched
        and verified_pause
        and not prev_verified
        and not user_paused
    ):
        return PauseDecision(
            verified_pause=False,
            log=[(
                logging.DEBUG,
                "ENDING: ambiguous match engages no verified pause for %s (item 469)",
                (current_matched,),
            )],
        )
    return PauseDecision(verified_pause=verified_pause)


def consistency_override(
    state: SwitchState,
    tick: MatchTick,
    result: Any,
    verified_pause: Any,
    get_profile: Callable[[str], Any],
) -> list[LogLine]:
    """Align the displayed program with a verified pause or a confident mismatch.

    A verified pause alone is not evidence for a NEW program: it used to adopt
    every tick's raw top-1 there, so an ambiguous A/B pair under a user pause
    displayed B, A, B, A... (audit MATCH-DECIDE-06). It needs the same persistence
    and non-ambiguity a normal switch needs. A confident mismatch (every candidate
    rejected) has no name, so it drops the program to "detecting..." - the
    manager-side half of the revoke the detector applies on element 5.
    """
    log: list[LogLine] = []
    profile_name = tick.profile_name
    pause_switch_ok = (
        verified_pause
        and not result.is_ambiguous
        and bool(tick.is_persistent)
    )
    if profile_name != state.current_program and (
        pause_switch_ok or result.is_confident_mismatch
    ):
        if profile_name:
            state.current_program = profile_name
            state.last_confidence = tick.confidence
            state.last_member_confidence = result.member_confidence
            # Try to fetch duration if we switched back to matched
            try:
                prof = get_profile(profile_name)
                if prof:
                    state.matched_duration = profile_duration(prof.get("avg_duration"))
            except Exception as e:  # pylint: disable=broad-exception-caught
                log.append((logging.DEBUG, "Failed to fetch profile duration on switch: %s", (e,)))
        else:
            state.current_program = DETECTING
            state.matched_duration = None
    return log


# Register item 514: the complete-cycle match at cycle end reads the stored trace
# with each finished stall and user pause (``cycle_data["halt_spans"]``, the
# detector's) cut out and the stored duration less them, as the live matcher reads
# the trace (``CycleDetector._match_readings``): a halt is not part of the
# programme. The stored cycle keeps every reading. The A/B switch.
STALL_CUT_FROM_FINAL_MATCH = True


def sanitize_stall_spans(raw: Any) -> list[tuple[float, float]]:
    """Finished stalls as ``(seconds from the cycle start, length)`` pairs (item 511):
    finite, non-negative, in order and not overlapping; anything else is dropped."""
    spans: list[tuple[float, float]] = []
    if not isinstance(raw, (list, tuple)):
        return spans
    for item in raw:
        try:
            at, length = float(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError, KeyError, OverflowError):
            continue
        if (
            math.isfinite(at) and math.isfinite(length) and at >= 0.0 and length > 0.0
            and (not spans or at >= spans[-1][0] + spans[-1][1])
        ):
            spans.append((at, length))
    return spans


def stall_cut_plan(
    offsets: list[float], spans: list[tuple[float, float]]
) -> list[tuple[int, float]]:
    """Which readings stay once each span is cut out, and how far each moves back.

    ``offsets`` are the readings' seconds from the cycle start, in order; ``spans``
    ``(at, length)`` pairs in order (an infinite length cuts to the end). Returns
    ``(index, shift_s)`` per kept reading: one inside ``[at, at + length)`` is
    dropped and every later one moves back by the spans before it. One
    implementation for the live matcher, progress and the cycle-end match.
    """
    cuts = [(at, at + length, length) for at, length in spans]
    out: list[tuple[int, float]] = []
    i, shift = 0, 0.0
    for idx, offset in enumerate(offsets):
        while i < len(cuts) and offset >= cuts[i][1]:
            shift += cuts[i][2]
            i += 1
        if i < len(cuts) and offset >= cuts[i][0]:
            continue  # inside a stall
        out.append((idx, shift))
    return out


def final_match_input(cycle_data: dict[str, Any]) -> tuple[Any, Any] | None:
    """``(power_data, duration)`` for the complete-cycle match, or None if too short.

    The detector stores ``power_data`` as ``[[offset_seconds, power], ...]``,
    offsets relative to the cycle start. Fewer than 10 readings: no final match.
    Finished stalls and user pauses are cut out (item 514,
    ``STALL_CUT_FROM_FINAL_MATCH``) unless that would leave fewer than 10.
    """
    power_data = cycle_data.get("power_data", [])
    duration = cycle_data.get("duration", 0)
    if not power_data or len(power_data) < 10:
        return None
    spans = (
        sanitize_stall_spans(cycle_data.get("halt_spans"))
        if STALL_CUT_FROM_FINAL_MATCH else []
    )
    if spans:
        try:
            offsets = [float(p[0]) for p in power_data]
            stored = float(duration or 0.0)
        except (TypeError, ValueError, IndexError, KeyError, OverflowError):
            return power_data, duration
        plan = stall_cut_plan(offsets, spans)
        if len(plan) >= 10:
            cut = [[round(offsets[i] - shift, 1), power_data[i][1]] for i, shift in plan]
            excluded = sum(length for at, length in spans if at < stored)
            return cut, max(0.0, stored - excluded)
    return power_data, duration


def cycle_end_label_verdict(
    match_result: Any, floor: float, profiles: Any
) -> tuple[str | None, str]:
    """``profile_store.label_verdict`` plus "that profile still exists".

    The auto-label decision at cycle end, on the complete-cycle match (or the last
    live match when the trace was too short for one). Reasons: ``ok``,
    ``no_winner``, ``below_floor``, ``ambiguous``, ``margin``, ``unknown_profile``.
    """
    # Imported here so this module stays importable without Home Assistant.
    from .profile_store import label_verdict  # pylint: disable=import-outside-toplevel

    verdict, reason = label_verdict(match_result, floor)
    if verdict and verdict not in profiles:
        verdict, reason = None, "unknown_profile"
    return verdict, reason


def display_sure_pct(margin: float | None) -> int:
    """The Status card's "~N% sure" for a live top1-top2 margin. DISPLAY ONLY.

    ``MATCH_SURE_KNOTS`` interpolated piecewise-linearly and clamped at both ends;
    ``None`` (no runner-up) reads ``MATCH_SURE_SINGLE_CANDIDATE``. Rounded to 5
    so it reads as the estimate it is. Monotone in the margin, so gating on it
    would equal gating on the margin (audit MATCH-DECIDE-18): never use it as a
    gate.
    """
    if margin is None:
        p = MATCH_SURE_SINGLE_CANDIDATE
    else:
        try:
            m = float(margin)
        except (TypeError, ValueError, OverflowError):
            m = 0.0
        if not math.isfinite(m):
            m = 0.0
        knots = MATCH_SURE_KNOTS
        if m <= knots[0][0]:
            p = knots[0][1]
        elif m >= knots[-1][0]:
            p = knots[-1][1]
        else:
            p = knots[-1][1]
            for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
                if m <= x1:
                    p = y0 + (y1 - y0) * (m - x0) / (x1 - x0)
                    break
    return int(5 * round(p * 20))


def live_match_uncertainty(result: Any, current_program: Any) -> dict[str, Any] | None:
    """Top two of an undecided live match, for the Status card (MATCH-DECIDE-15).

    Undecided: no program committed yet, or the latest tick flags its own winner
    as ambiguous (a runner-up within ``MATCH_AMBIGUITY_MARGIN``, or a Stage-5
    safeguard). ``None`` when decided or there is no winner. A runner-up that is a
    profile group is named by its best member. Display only: nothing that ends,
    labels or switches a cycle reads this.
    """
    best = getattr(result, "best_profile", None) if result is not None else None
    if not best:
        return None
    if program_is_committed(current_program) and getattr(result, "is_ambiguous", False) is not True:
        return None
    runner_up: str | None = None
    candidates = getattr(result, "candidates", None) or []
    for cand in list(candidates)[1:2]:
        if not isinstance(cand, dict):
            continue
        name = cand.get("group_best_member") or cand.get("name")
        if isinstance(name, str) and name.startswith("__group__"):
            name = name[len("__group__"):]
        if name and name != best:
            runner_up = str(name)
    margin: float | None = None
    if runner_up is not None:
        try:
            margin = round(max(0.0, float(getattr(result, "ambiguity_margin", 0.0) or 0.0)), 3)
        except (TypeError, ValueError, OverflowError):
            margin = 0.0
    return {
        "top": str(best),
        "runner_up": runner_up,
        "margin": margin,
        "sure_pct": display_sure_pct(margin),
    }
