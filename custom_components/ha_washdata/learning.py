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
"""Learning and self-tuning logic for WashData."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime
from collections.abc import Callable, Coroutine
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send
import homeassistant.util.dt as dt_util

from .const import (
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_LEARNING_CONFIDENCE,
    CONF_PROFILE_MIN_WARMUP_CYCLES,
    DEFAULT_AUTO_LABEL_CONFIDENCE,
    DEFAULT_LEARNING_CONFIDENCE,
    MIN_SUGGESTION_COOLDOWN_CYCLES,
    MIN_SUGGESTION_REL_DELTA,
    REVIEW_BUDGET_PER_PROFILE,
    TerminationReason,
)
from .suggestion_engine import SuggestionEngine
from .log_utils import DeviceLoggerAdapter
from .profile_store import _AUTO_LABEL_SOURCES
from .detector_config import effective_option_values

if TYPE_CHECKING:
    from .profile_store import ProfileStore


_LOGGER = logging.getLogger(__name__)


def _suggestion_min_abs_delta(key: str) -> float:
    """Return the minimum absolute change that makes a suggestion worth surfacing.

    Both this threshold AND MIN_SUGGESTION_REL_DELTA must be missed for a
    suggestion to be suppressed — either one passing is enough to keep it.
    """
    if key.endswith(("_w", "_power")):
        return 0.3      # Watts: sub-0.3 W changes are below sensor noise
    if key.endswith(("_interval", "_timeout", "_delay", "_gap", "_duration", "_seconds", "_duration_threshold")):
        return 5.0      # Seconds: 5 s is imperceptible to the detector
    if key.endswith(("_ratio", "_tolerance")):
        return 0.02     # Unitless ratio: 0.02 is the minimum meaningful step
    if key.endswith(("_confidence", "_threshold")):
        return 0.02     # Probability (0–1): 0.02 is the minimum meaningful step
    if key.endswith(("_count", "_window", "_repeat")):
        return 1.0      # Integer count: less than 1 is a no-op
    return 0.05


def _ended_on_its_own(cycle: dict[str, Any]) -> bool:
    """Completed, and not cut short by the user (#458; cf. ``select_clean_cycles``)."""
    return (
        cycle.get("status") == "completed"
        and cycle.get("termination_reason") != TerminationReason.USER
    )


def review_choices(pending: Any, profiles: Any) -> list[str]:
    """The two programmes a review request offers as one-tap answers. Never raises.

    The detected programme first, then the stored ranking's next distinct one (a
    profile group named by its best member). Only names of existing profiles, at
    most two, in that order. The panel's answer buttons, its chart overlay and the
    finish notification's actions all read this, so they always name the same two;
    answer one with :meth:`LearningManager.async_answer_review`.
    """
    out: list[str] = []
    try:
        if not isinstance(pending, dict):
            return out
        names = profiles if profiles is not None else {}

        def _add(name: Any) -> None:
            if (
                isinstance(name, str) and name and name not in out
                and name in names and len(out) < 2
            ):
                out.append(name)

        _add(pending.get("detected_profile"))
        ranking = pending.get("ranking")
        for cand in ranking if isinstance(ranking, list) else []:
            if len(out) >= 2:
                break
            if not isinstance(cand, dict):
                continue
            name = cand.get("name")
            if isinstance(name, str) and name.startswith("__group__"):
                name = cand.get("group_best_member")
            _add(name)
    except Exception:  # noqa: BLE001 - a display helper must never raise
        return out
    return out


#: Answer key that confirms a review request's own programme.
REVIEW_ANSWER_CONFIRM = "C"
#: Prefix of the answer key ``R<j>``: the request's stored ranking entry ``j``.
REVIEW_ANSWER_RANKED = "R"


def _ranked_name(cand: Any) -> Any:
    """The programme a stored ranking entry names (a group by its best member)."""
    if not isinstance(cand, dict):
        return None
    name = cand.get("name")
    if isinstance(name, str) and name.startswith("__group__"):
        return cand.get("group_best_member")
    return name


def review_answer_choices(request: Any, profile_names: Any) -> list[tuple[str, str]]:
    """The finish notification's one-tap answers: ``[(key, programme)]``.

    The same two programmes as the panel's buttons (:func:`review_choices`), keyed
    so the phone carries a position, not a name: ``REVIEW_ANSWER_CONFIRM`` for the
    request's own programme, ``R<j>`` for the other, ``j`` being its entry in the
    request's stored ranking. That list is never reordered (a rename rewrites it in
    place), so the key keeps naming the programme the phone shows: deleting it
    makes the tap stale (:func:`resolve_review_answer` returns None) instead of
    moving the key onto the next programme. Pure, never raises.
    """
    try:
        detected = request.get("detected_profile") if isinstance(request, dict) else None
        ranking = request.get("ranking") if isinstance(request, dict) else None
        ranked = [_ranked_name(c) for c in ranking] if isinstance(ranking, list) else []
        out: list[tuple[str, str]] = []
        for name in review_choices(request, profile_names):
            if name == detected:
                out.append((REVIEW_ANSWER_CONFIRM, name))
            elif name in ranked:
                out.append((f"{REVIEW_ANSWER_RANKED}{ranked.index(name)}", name))
        return out
    except Exception:  # noqa: BLE001 - a notification extra must never raise
        return []


def resolve_review_answer(request: Any, key: Any, profile_names: Any) -> str | None:
    """The programme an answer key names on this request, or None when it no
    longer exists or the key is malformed. Never raises."""
    try:
        if not isinstance(request, dict) or not isinstance(key, str):
            return None
        names = profile_names if profile_names is not None else {}
        detected = request.get("detected_profile")
        if key == REVIEW_ANSWER_CONFIRM:
            name = detected
        else:
            idx = key[len(REVIEW_ANSWER_RANKED):]
            if not key.startswith(REVIEW_ANSWER_RANKED) or not (idx.isascii() and idx.isdigit()):
                return None
            ranking = request.get("ranking")
            j = int(idx)
            if not isinstance(ranking, list) or j >= len(ranking):
                return None
            name = _ranked_name(ranking[j])
            if name == detected:
                return None
        return name if isinstance(name, str) and name and name in names else None
    except Exception:  # noqa: BLE001 - an answer that cannot be read answers nothing
        return None


class StatisticalModel:
    """Helper to track running stats for a metric."""

    def __init__(self, max_samples: int = 200) -> None:
        self._samples: list[float] = []
        self._max_samples = max_samples
        self._last_update: datetime | None = None
        self._stats: dict[str, Any] = {"median": None, "p95": None, "count": 0}

    def add_sample(self, value: float, now: datetime) -> None:
        """Add a sample and update stats."""
        self._samples.append(value)
        if len(self._samples) > self._max_samples:
            self._samples = self._samples[-self._max_samples:]
        self._last_update = now
        self._compute_stats()

    def _compute_stats(self) -> None:
        if not self._samples:
            self._stats = {"median": None, "p95": None, "count": 0}
            return

        arr = np.array(self._samples)
        self._stats = {
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "count": int(len(self._samples)),
        }

    @property
    def median(self) -> float | None:
        """Return the median of samples."""
        return self._stats.get("median")

    @property
    def p95(self) -> float | None:
        """Return the 95th percentile of samples."""
        return self._stats.get("p95")

    @property
    def count(self) -> int:
        """Return the number of samples."""
        return self._stats.get("count", 0)


class LearningManager:
    """Manages cycle learning, user feedback, and auto-tuning."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry_id: str,
        profile_store: "ProfileStore",
        device_type: str | None = None,
        device_name: str = "",
        spawn: Callable[[Coroutine[Any, Any, Any]], Any] | None = None,
    ) -> None:
        """Initialize the learning manager.

        ``spawn`` starts every background task this class creates. The manager
        passes its ``_spawn_tracked`` (audit MANAGER-13): these tasks save the
        ProfileStore, and an untracked one survives an entry reload and can write
        the OLD store over the new one under the same Store key. Without it (unit
        tests) a plain ``hass.async_create_task`` is used.
        """
        self._logger = DeviceLoggerAdapter(_LOGGER, device_name)
        self.hass = hass
        self._spawn_fn = spawn
        self.entry_id = entry_id
        self.profile_store = profile_store
        self.device_type = device_type
        self.suggestion_engine = SuggestionEngine(
            hass, entry_id, profile_store, device_type, spawn=spawn
        )

        # Operational Stats
        self._sample_interval_model = StatisticalModel(max_samples=200)
        # Update intervals seen during the CURRENT cycle, held back until it ends
        # (#458). A cycle that never ends on its own - force-stopped after hours of
        # a plug reporting standby at its idle cadence - would otherwise fill the
        # whole 200-sample window with that idle cadence, and the watchdog and
        # no-update timeouts would be re-suggested from it every five minutes.
        # Bounded like the model itself: only its last 200 survive a commit anyway.
        self._pending_intervals: deque[tuple[float, datetime]] = deque(maxlen=200)
        # Set by a cadence commit, consumed by run_due_operational_pass once the
        # cycle is stored, so the pass's pause scan sees the cycle that just ended.
        self._operational_pass_due: bool = False
        self._last_batch_simulation_count: int = 0  # track when to re-run batch
        self._last_suggestions_labeled_count: int = 0  # gate model/detection passes

    def _spawn(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Start a background task through the manager's tracked spawner."""
        if self._spawn_fn is not None:
            return self._spawn_fn(coro)
        return self.hass.async_create_task(coro)

    def _apply_suggestions_and_notify(self, suggestions: dict[str, Any]) -> None:
        """Apply suggestions that pass quality gates."""
        if not suggestions:
            return

        # Quality gate: drop or suppress suggestions that are not worth surfacing.
        entry = self.hass.config_entries.async_get_entry(self.entry_id)
        current_options: dict[str, Any] = {}
        if entry:
            current_options = {**entry.data, **entry.options}

        # Cooldown: how many cycles have elapsed since the user last applied suggestions?
        # Counted on the odometer, not len(past_cycles): at the retention cap the
        # list length stops growing, so one Apply-all silenced every non-corrective
        # suggestion forever (audit SUGGEST-05).
        cycles_now = self.profile_store.get_lifetime_cycle_count()
        last_apply_count = self.profile_store.get_suggestion_apply_cycle_count()
        cooldown_active = (
            last_apply_count > 0
            and (cycles_now - last_apply_count) < MIN_SUGGESTION_COOLDOWN_CYCLES
        )
        # Compare against what each key RUNS with when it is unset (audit
        # SUGGEST-10), not against None, which skipped every gate below.
        effective = effective_option_values(current_options, self.device_type)

        # Locked keys (#343): the user has told the auto-tuner to stop proposing
        # these (e.g. thresholds that break an anti-crease-tuned device). Drop them
        # from the pending map so they never re-surface until unlocked.
        try:
            locked = set(self.profile_store.get_locked_suggestions())
        except Exception:  # pylint: disable=broad-exception-caught
            locked = set()

        existing_suggestions = self.profile_store.get_suggestions()
        filtered_suggestions: dict[str, Any] = {}
        any_deleted = False
        for key, data in suggestions.items():
            if key in locked:
                if key in existing_suggestions:
                    self.profile_store.delete_suggestion(key)
                    any_deleted = True
                continue
            if isinstance(data, dict) and "value" in data:
                current_val = current_options.get(key)
                if current_val is None:
                    current_val = effective.get(key)
                suggested_val = data["value"]
                if current_val is not None and suggested_val is not None:
                    try:
                        cv, sv = float(current_val), float(suggested_val)
                        abs_delta = abs(sv - cv)

                        # Gate 1: exact equality → stale, delete so it doesn't linger.
                        # Gates 1 and 2 save only when a stored key was removed
                        # (#510): the 5-minute pass re-derives the same applied
                        # values every tick of a running cycle, and "deleting" a key
                        # that was never stored rewrote the whole store each time.
                        if abs_delta < 1e-9:
                            if key in existing_suggestions:
                                self.profile_store.delete_suggestion(key)
                                any_deleted = True
                            continue

                        # Gate 2: change too small to be meaningful → delete (noise).
                        rel_delta = abs_delta / max(abs(cv), 1e-3)
                        if (rel_delta < MIN_SUGGESTION_REL_DELTA
                                and abs_delta < _suggestion_min_abs_delta(key)):
                            if key in existing_suggestions:
                                self.profile_store.delete_suggestion(key)
                                any_deleted = True
                            continue

                        # Gate 3: cooldown active → skip update without deleting.
                        # After the user applies suggestions, wait for a few more
                        # cycles before surfacing new ones (avoids immediately
                        # re-suggesting a slightly-different value on the next cycle).
                        if cooldown_active and not data.get("corrective"):
                            continue

                    except (TypeError, ValueError, OverflowError):
                        pass
            filtered_suggestions[key] = data

        if not filtered_suggestions:
            if any_deleted:
                self._spawn(self.profile_store.async_save())
            return

        # A deletion above is unsaved until a write: apply_suggestions saves only
        # when it changes something itself (#510), so tell it.
        self.suggestion_engine.apply_suggestions(filtered_suggestions, changed=any_deleted)

    def process_power_reading(
        self, _power: float, now: datetime, last_reading_time: datetime | None
    ) -> None:
        """Hold this reading's update interval for the cycle-end cadence commit.

        No suggestion pass runs here. It used to, every 5 minutes of a running
        cycle, but everything it reads moves only at a cycle end (the cadence
        model is committed by ``close_cycle_cadence``, the traces it scans are
        stored then), so each one re-derived what the last cycle-end pass had,
        a ~100-trace executor scan for nothing (#510 follow-up). The pass runs
        once per cycle end, in ``run_due_operational_pass``.
        """
        if last_reading_time:
            delta = (now - last_reading_time).total_seconds()
            # Ignore ultra-small jitter (<0.1s) and massive gaps (>1800s - likely downtime)
            if 0.1 < delta < 1800:
                # Held until the cycle's outcome is known (#458): see close_cycle_cadence.
                self._pending_intervals.append((delta, now))

    def discard_cycle_cadence(self) -> None:
        """Drop held intervals without committing them (a new start from idle)."""
        self._pending_intervals.clear()

    def close_cycle_cadence(self, cycle_data: dict[str, Any] | None) -> bool:
        """Commit or drop the update intervals held for the cycle that just ended.

        Committed only when the cycle ended on its own (``status == "completed"``
        and not user-stopped): the same line ``select_clean_cycles`` draws, and for
        the same reason (#458). A force-stopped or user-stopped cycle's intervals
        describe however long the plug sat reporting standby before something gave
        up, not how the appliance reports while it works. Called for EVERY cycle
        end, persisted or not, so one cycle's intervals can never leak into the
        next. Returns whether anything was committed.
        """
        pending = list(self._pending_intervals)
        self._pending_intervals.clear()
        if not pending or not isinstance(cycle_data, dict):
            return False
        if cycle_data.get("status") != "completed":
            return False
        if cycle_data.get("termination_reason") in (
            TerminationReason.USER,
            TerminationReason.FORCE_STOPPED,
        ):
            return False
        for delta, ts in pending:
            self._sample_interval_model.add_sample(delta, ts)
        # The operational pass's one run per cycle: its cadence input changes only
        # here (process_power_reading runs none during the cycle). Not run here:
        # this commit comes before the cycle is stored, and the pass's pause scan
        # reads the stored cycles, so it would miss this one until the next end.
        self._operational_pass_due = True
        return True

    def run_due_operational_pass(self) -> None:
        """Run the operational pass a cadence commit left due, once.

        Called once the ended cycle is stored (``process_cycle_end``) and, for an
        end that stores nothing (ghost, pump-out, failed write), by the manager
        directly: the stored cycles did not change there, so nothing is missed.
        """
        if not self._operational_pass_due:
            return
        self._operational_pass_due = False
        self._update_operational_suggestions()

    def process_cycle_end(
        self,
        cycle_data: dict[str, Any],
        detected_profile: str | None = None,
        confidence: float = 0.0,
        predicted_duration: float | None = None,
        match_result: Any | None = None,
        label_allowed: bool = True,
    ) -> None:
        """Analyze completed cycle for learning.

        ``label_allowed`` is the manager's cycle-end label gate verdict (margin,
        ambiguity and margin owner). When it refused, this pass must not label the
        cycle anyway on confidence alone (audit MANAGER-02 / MATCH-DECIDE-01); it
        asks the user instead.
        
        Args:
            cycle_data: Completed cycle data
            detected_profile: Profile name detected
            confidence: Match confidence score (0.0-1.0)
            predicted_duration: Expected duration in seconds
            match_result: MatchResult from profile_store.async_match_profile() (optional)
        """
        # (No per-cycle stop/start simulation: re-deriving both from the last cycle
        # alone made them a random walk - audit SUGGEST-06. The batch pass below
        # derives them across cycles.)

        # 1. Operational (off delay, watchdog, timeouts): the cycle is stored now,
        # so the off-delay pause scan includes it.
        self.run_due_operational_pass()

        # 1b. Standby above the stop threshold (#458). Every cycle end, whatever its
        # status: a force-stopped cycle is the evidence this pass exists for.
        self._dispatch_scan_and_apply(
            self.suggestion_engine.for_job().generate_standby_floor_suggestions,
            "Standby floor",
        )

        # 2. Check if we should request feedback
        self._maybe_request_feedback(
            cycle_data, detected_profile, confidence, predicted_duration, match_result,
            label_allowed=label_allowed,
        )

        # 3+3b. Heavy per-profile suggestion passes — only run when the labeled
        # cycle count has grown since the last update (skips passes for unlabeled /
        # noise / duplicate ends with no new data).
        # New EVIDENCE, not new rows (#458): a force-stopped or user-stopped cycle
        # is dropped by `select_clean_cycles` inside every pass this gates, so
        # counting it re-ran them on unchanged data.
        labeled_count = sum(
            1 for c in self.profile_store.get_past_cycles()
            if isinstance(c, dict)
            and c.get("profile_name")
            and c.get("profile_name") != "noise"
            and _ended_on_its_own(c)
        )
        if labeled_count > self._last_suggestions_labeled_count:
            self._last_suggestions_labeled_count = labeled_count
            self._update_model_suggestions()
            self._update_detection_suggestions()

        # 4. Run multi-cycle batch simulation when enough new labeled cycles have accumulated
        self._maybe_run_batch_simulation()

    def _maybe_run_batch_simulation(self) -> None:
        """Schedule a batch simulation when enough new labeled cycles have arrived."""
        _BATCH_MIN = 5
        _BATCH_RERUN_DELTA = 5  # Re-run every 5 new labeled cycles

        labeled_cycles = [
            c for c in self.profile_store.get_past_cycles()
            if isinstance(c, dict)
            and c.get("profile_name")
            and c.get("profile_name") != "noise"
            and c.get("power_data")
            and c.get("status") in ("completed", "force_stopped")
        ]
        # The list still carries force-stopped cycles - the min_off_gap merge
        # ceiling needs the user's real turnaround - but only cycles that ended
        # on their own are new evidence for the re-run cadence (#458).
        current_count = sum(1 for c in labeled_cycles if _ended_on_its_own(c))

        if current_count < _BATCH_MIN:
            return
        if (current_count - self._last_batch_simulation_count) < _BATCH_RERUN_DELTA:
            return

        self._last_batch_simulation_count = current_count
        self._spawn(self._async_run_batch_simulation(labeled_cycles))

    async def _async_run_batch_simulation(self, cycles: list[dict[str, Any]]) -> None:
        """Run multi-cycle batch simulation asynchronously."""
        try:
            engine = self.suggestion_engine.for_job()
            new_suggestions = await self.hass.async_add_executor_job(
                engine.run_batch_simulation, cycles
            )
            if new_suggestions:
                self._apply_suggestions_and_notify(new_suggestions)
                self._logger.debug(
                    "Batch simulation (%d cycles) produced suggestions: %s",
                    len(cycles),
                    list(new_suggestions.keys()),
                )
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._logger.error("Batch simulation failed: %s", e)

    def _update_operational_suggestions(self) -> None:
        """Generate suggestions for operational parameters (intervals, timeouts).

        The cadence stats (p95/median) are read on the event loop and captured as
        immutable snapshots; the historical-trace scan inside
        ``generate_operational_suggestions`` is offloaded to an executor thread by
        ``_dispatch_scan_and_apply`` so it never runs on the loop.
        """
        if self._sample_interval_model.count < 20:
            return

        p95 = self._sample_interval_model.p95
        median = self._sample_interval_model.median

        if p95 is None or median is None:
            return

        # Bind the config snapshot here, on the loop, not inside the executor job.
        engine = self.suggestion_engine.for_job()
        self._dispatch_scan_and_apply(
            lambda: engine.generate_operational_suggestions(p95, median),
            "Operational",
        )

    def _update_model_suggestions(self) -> None:
        """Generate suggestions for model parameters (tolerances, ratios).

        The historical-cycle scan inside ``generate_model_suggestions`` is
        offloaded to an executor thread by ``_dispatch_scan_and_apply``.
        """
        self._dispatch_scan_and_apply(
            self.suggestion_engine.for_job().generate_model_suggestions,
            "Model",
        )

    def _dispatch_scan_and_apply(
        self, generate: Callable[[], dict[str, Any]], label: str
    ) -> None:
        """Run a heavy suggestion scan off the event loop, then apply results.

        ``generate`` is a pure suggestion-engine call that scans historical power
        traces (up to ~100-200 cycles) and is too heavy to run on the event loop.
        When a running loop is present (normal operation) the scan is offloaded to
        an executor thread and the resulting suggestions are applied back on the
        loop. In a synchronous context with no running loop (unit tests / direct
        callers) it runs inline so results are observable immediately. ``generate``
        must only read shared state and return suggestions — the state mutation
        (``_apply_suggestions_and_notify``) always runs on the loop.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop: run inline (synchronous callers / unit tests).
            try:
                suggestions = generate()
            except Exception as e:  # pylint: disable=broad-exception-caught
                self._logger.error("%s suggestion pass failed: %s", label, e)
                return
            if suggestions:
                self._apply_suggestions_and_notify(suggestions)
            return
        self._spawn(self._async_scan_and_apply(generate, label))

    async def _async_scan_and_apply(
        self, generate: Callable[[], dict[str, Any]], label: str
    ) -> None:
        """Offload ``generate`` to an executor thread, then apply on the loop."""
        try:
            suggestions = await self.hass.async_add_executor_job(generate)
            if suggestions:
                self._apply_suggestions_and_notify(suggestions)
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._logger.error("%s suggestion pass failed: %s", label, e)

    def _update_detection_suggestions(self) -> None:
        """Generate statistical detection suggestions from clean cycles.

        Offloaded to an executor because it scans power traces across up to 200
        cycles for the clean-cycle health checks.
        """
        self._spawn(self._async_run_detection_suggestions())

    async def _async_run_detection_suggestions(self) -> None:
        """Run the detection-suggestion pass off the event loop."""
        try:
            engine = self.suggestion_engine.for_job()
            new_suggestions = await self.hass.async_add_executor_job(
                engine.generate_detection_suggestions
            )
            if new_suggestions:
                self._apply_suggestions_and_notify(new_suggestions)
                self._logger.debug(
                    "Detection suggestions produced: %s", list(new_suggestions.keys())
                )
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._logger.error("Detection suggestion pass failed: %s", e)

    async def async_run_full_analysis(self) -> dict[str, int]:
        """Run every suggestion pass now (manual trigger from the panel).

        Runs the operational (cadence), model, detection and batch-simulation
        passes over the accumulated cycle history and reconciles the result.
        Returns ``{"count": <actionable suggestions>}``.
        """
        self._logger.info("Manual suggestion analysis requested")
        try:
            model = self._sample_interval_model
            if model.count >= 20 and model.p95 is not None and model.median is not None:
                p95, median = model.p95, model.median
                op = await self.hass.async_add_executor_job(
                    self.suggestion_engine.for_job().generate_operational_suggestions,
                    p95, median
                )
                if op:
                    self._apply_suggestions_and_notify(op)
            model_sug = await self.hass.async_add_executor_job(
                self.suggestion_engine.for_job().generate_model_suggestions
            )
            if model_sug:
                self._apply_suggestions_and_notify(model_sug)
            await self._async_run_detection_suggestions()
            # Snapshot the live cycles list before handing it to the executor.
            cycles = list(self.profile_store.get_past_cycles())
            batch = await self.hass.async_add_executor_job(
                self.suggestion_engine.for_job().run_batch_simulation, cycles
            )
            if batch:
                self._apply_suggestions_and_notify(batch)
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._logger.error("Manual suggestion analysis failed: %s", e)
        count = len(self.profile_store.get_suggestions() or {})
        self._logger.info("Manual suggestion analysis complete: %d suggestion(s)", count)
        return {"count": count}

    def _maybe_request_feedback(
        self,
        cycle_data: dict[str, Any],
        detected_profile: str | None,
        confidence: float,
        predicted_duration: float | None,
        match_result: Any | None = None,
        label_allowed: bool = True,
    ) -> None:
        """Check if feedback should be requested for this completed cycle."""
        if (
            not predicted_duration
            or not detected_profile
            or detected_profile in ("off", "detecting...")
        ):
            # No match was made, don't request feedback
            return

        # Get the cycle ID from the cycle_data
        cycle_id = cycle_data.get("id")
        if not cycle_id:
            self._logger.warning("Cycle data missing ID, cannot request feedback")
            return

        # The user already chose this cycle's programme. Asking them to confirm the
        # matcher's guess on top of it (warm-up, or a gate-refused match) is the
        # request the v16 cleanup drops (_dismiss_unneeded_feedback); never raise it.
        if cycle_data.get("label_source") == "manual":
            self._logger.debug(
                "Cycle %s was labelled by hand; no confirmation needed", cycle_id
            )
            return

        # Get Configured Thresholds
        entry = self.hass.config_entries.async_get_entry(self.entry_id)
        if not entry:
            return

        auto_label_conf = entry.options.get(
            CONF_AUTO_LABEL_CONFIDENCE, DEFAULT_AUTO_LABEL_CONFIDENCE
        )
        learning_conf = entry.options.get(
            CONF_LEARNING_CONFIDENCE, DEFAULT_LEARNING_CONFIDENCE
        )

        # A4: Warmup mode — profiles with fewer than CONF_PROFILE_MIN_WARMUP_CYCLES labeled
        # cycles skip auto-labeling entirely and always request user confirmation.
        # Only applied when confidence would otherwise trigger auto-labeling; cycles
        # already below the learning threshold follow the normal skip path unchanged.
        warmup_request = False
        # ``route_conf`` drives the auto-label/skip routing only; ``confidence``
        # remains the real match score that gets displayed and persisted, so warmup
        # clamping never fabricates the value shown to the user.
        route_conf = confidence
        # Warm-up applies wherever the cycle would otherwise go unasked: the
        # auto-label band, and (since 0.5.8) a cycle the cycle-end gate labelled.
        if confidence >= auto_label_conf or (label_allowed and confidence >= learning_conf):
            _wm_count = self.profile_store.get_profile_labeled_count(detected_profile)
            # Imported reference profiles are trusted downloaded templates: the user
            # expects to match immediately, so they skip the local warm-up gate.
            _imported = self.profile_store.profile_has_reference_cycles(detected_profile)
            _is_warmup = (
                not _imported
                and isinstance(_wm_count, int)
                and _wm_count < CONF_PROFILE_MIN_WARMUP_CYCLES
            )
            if _is_warmup:
                self._logger.info(
                    "Profile '%s' in warmup mode (%d/%d cycles); requiring manual confirmation.",
                    detected_profile, _wm_count, CONF_PROFILE_MIN_WARMUP_CYCLES,
                )
                # A warmup cycle must always request confirmation: never auto-label,
                # and never silently skip — even under a misconfigured inverted
                # (learning_conf >= auto_label_conf) threshold pair.
                warmup_request = True
                # Clamp the ROUTING confidence just below auto_label so we fall through
                # to the feedback-request path, but stay above learning_conf to request
                # (not skip). Only raise toward learning_conf when there is room below
                # auto_label_conf; otherwise an inverted config would push it back to/above
                # auto_label_conf and silently bypass the warmup guard.
                route_conf = auto_label_conf - 0.001
                if learning_conf + 0.001 < auto_label_conf:
                    route_conf = max(route_conf, learning_conf + 0.001)

        # Auto-label if very high confidence - but not when the cycle's power trace
        # is mostly outside the profile envelope band (low conformance = the shape
        # matched but the actual power levels are inconsistent with the profile).
        # (The ML quality gate that also downgraded here, C3, was removed in 0.5.8:
        # it fired on 0 of the auto-label-eligible real cycles, audit ML-06. A
        # legacy `ml_quality_score` on an older cycle is ignored.)
        _conformance = cycle_data.get("envelope_conformance")
        try:
            envelope_suspicious = (
                _conformance is not None
                and float(_conformance) < 0.40
            )
        except (TypeError, ValueError, OverflowError):
            envelope_suspicious = False
        # The cycle-end pass already labelled this cycle with a DIFFERENT programme
        # (the post-cycle match on the complete trace): re-labelling here would
        # overwrite that label with this pass's guess. (A hand-picked label returned
        # at the top.)
        _existing_label = cycle_data.get("profile_name")
        if route_conf >= auto_label_conf and _existing_label and _existing_label != detected_profile:
            self._logger.debug(
                "Cycle %s already labelled '%s' at cycle end; not auto-labelling it",
                cycle_id, _existing_label,
            )
            return
        if route_conf >= auto_label_conf:
            if not label_allowed:
                self._logger.info(
                    "Cycle %s: the label gate refused '%s' (too close to the runner-up "
                    "or flagged ambiguous); requesting confirmation instead of "
                    "auto-labelling it.",
                    cycle_id, detected_profile,
                )
                # Fall through to the feedback-request path below.
            elif envelope_suspicious:
                # Not a review request on its own (register item 433): a cycle
                # the cycle-end gate already labelled returns below unasked.
                self._logger.info(
                    "Envelope conformance for cycle %s is low (%.2f < 0.40); "
                    "not auto-labelling it here.",
                    cycle_id, _conformance,
                )
                # Fall through to feedback-request path below.
            else:
                labeled = self.auto_label_high_confidence(
                    cycle_id=cycle_id,
                    profile_name=detected_profile,
                    confidence=confidence,
                    confidence_threshold=auto_label_conf,
                )
                if labeled:
                    # Rebuild envelope first, then persist (issue #131)
                    self._spawn(
                        self._async_rebuild_and_save_profile(detected_profile)
                    )
                    self._logger.debug("Auto-labeled high-confidence cycle %s", cycle_id)
                return

        # A cycle the cycle-end gate already labelled with this programme (a clear
        # margin over the runner-up, not ambiguous, above the learning floor) needs no
        # confirmation. Measured leave-one-out over 604 cycle ends, those labels are
        # 91.5% right; asking about every 0.6-0.9 match instead put 81% of all
        # cycles in the review queue, most of them already labelled correctly. What
        # predicts a wrong label is a small margin, not a modest confidence: at
        # 0.7-0.9 a clear margin is 93-96% right and a refused gate 33-54%
        # (register item 433). Warm-up still asks. Low envelope conformance does
        # NOT: labelled cycles under 0.40 are still 85.6% right, so asking about them
        # costs 7 questions per wrong label (and on a device with loose envelopes it
        # re-queued nearly every cycle).
        if (
            label_allowed
            and not warmup_request
            and cycle_data.get("profile_name") == detected_profile
        ):
            self._logger.debug(
                "Cycle %s labelled '%s' at cycle end with a clear margin; no confirmation needed",
                cycle_id, detected_profile,
            )
            return

        # Skip low-confidence matches below learning threshold — but a warmup cycle
        # always requests confirmation, even if the thresholds are misconfigured.
        if route_conf < learning_conf and not warmup_request:
            self._logger.debug(
                "Skipping feedback for low-confidence match (conf=%.2f < %.2f)",
                confidence,
                learning_conf,
            )
            return

        # A refused gate asks only while this programme's review budget lasts
        # (REVIEW_BUDGET_PER_PROFILE): the answers that move accuracy are a
        # programme's first ones and every correction, not the tenth confirmation.
        if not warmup_request:
            spent = self._review_budget_spent(detected_profile)
            if spent >= REVIEW_BUDGET_PER_PROFILE:
                self._logger.debug(
                    "Not asking about cycle %s: '%s' already has %d waiting or "
                    "confirmed review requests",
                    cycle_id, detected_profile, spent,
                )
                return

        actual_duration = cycle_data.get("duration", 0)

        # Request feedback via learning manager for moderate confidence
        self.request_cycle_verification(
            cycle_id=cycle_id,
            detected_profile=detected_profile,
            confidence=confidence,
            estimated_duration=predicted_duration,
            actual_duration=actual_duration,
            match_result=match_result,
        )

        # Persist pending feedback request so it survives restart.
        # The pending review is surfaced in the panel's Cycles review queue;
        # WashData intentionally does not raise a persistent notification here.
        self._spawn(self.profile_store.async_save())

    def _review_budget_spent(self, profile_name: str) -> int:
        """Review requests about ``profile_name`` that spend its budget. Never raises.

        Waiting requests, plus answers that did not correct it: a confirmation, a
        dismissal, or a relabel to that same programme. A correction to another
        programme does not count (see ``REVIEW_BUDGET_PER_PROFILE``). Counted from
        the stored queue and answer history, so it survives a restart and follows
        a profile rename, which rewrites both.
        """
        spent = 0
        try:
            # A request whose cycle is gone (not pruned yet) asks nothing.
            stored = {c.get("id") for c in self.profile_store.get_past_cycles()}
            for cid, rec in (self.profile_store.get_pending_feedback() or {}).items():
                if (
                    isinstance(rec, dict) and rec.get("detected_profile") == profile_name
                    and cid in stored
                ):
                    spent += 1
            for rec in (self.profile_store.get_feedback_history() or {}).values():
                if not isinstance(rec, dict) or rec.get("original_detected_profile") != profile_name:
                    continue
                corrected = rec.get("corrected_profile")
                if rec.get("user_confirmed") or not corrected or corrected == profile_name:
                    spent += 1
        except Exception:  # noqa: BLE001 - a count must never block a cycle end
            self._logger.debug("Review budget count failed", exc_info=True)
        return spent

    def request_cycle_verification(
        self,
        cycle_id: str,
        detected_profile: Optional[str],
        confidence: float,
        estimated_duration: Optional[float],
        actual_duration: float,
        match_result: Any | None = None,
    ) -> None:
        """Request user verification for a detected cycle."""
        duration_match_pct = (
            (actual_duration / estimated_duration * 100) if estimated_duration else 0
        )

        # Extract match ranking from MatchResult if available (for UI visualization)
        ranking_summary: list[dict[str, Any]] = []
        if match_result and hasattr(match_result, "ranking") and match_result.ranking:
            for cand in match_result.ranking[:5]:  # Store top 5
                try:
                    entry: dict[str, Any] = {
                        "name": cand.get("name", "Unknown"),
                        "score": float(cand.get("score", 0.0)),
                        "metrics": cand.get("metrics", {}),
                        "profile_duration": float(cand.get("profile_duration", 0.0)),
                    }
                    # A profile group ranks under its `__group__` key; the member
                    # that scored is what a review answer can name (review_choices).
                    if str(entry["name"]).startswith("__group__") and cand.get("group_best_member"):
                        entry["group_best_member"] = str(cand["group_best_member"])
                    ranking_summary.append(entry)
                except (TypeError, ValueError, KeyError, AttributeError, OverflowError):
                    continue

        feedback_req: dict[str, Any] = {
            "cycle_id": cycle_id,
            "detected_profile": detected_profile,
            "confidence": confidence,
            "estimated_duration": estimated_duration,
            "actual_duration": actual_duration,
            "duration_match_pct": duration_match_pct,
            # (No `is_close_match` since 0.5.9: nothing read it, and its tolerance
            # was the retired duration_tolerance setting. Older records keep it.)
            "created_at": dt_util.now().isoformat(),
            "user_response": None,
            "expires_at": None,
            "ranking": ranking_summary,  # Top candidates for UI display
        }

        self.profile_store.add_pending_feedback(cycle_id, feedback_req)

        est_min = int(estimated_duration / 60) if estimated_duration else 0
        self._logger.info(
            "Feedback requested for cycle %s: profile='%s' (conf=%.2f), "
            "est=%smin, actual=%smin (%.0f%%)",
            cycle_id,
            detected_profile,
            confidence,
            est_min,
            int(actual_duration / 60),
            duration_match_pct,
        )

    def auto_label_high_confidence(
        self,
        cycle_id: str,
        profile_name: str,
        confidence: float,
        confidence_threshold: float,
    ) -> bool:
        """Auto-label a cycle with high confidence."""
        if confidence < confidence_threshold:
            return False

        # Reuse existing internal logic
        self._auto_label_cycle(cycle_id, profile_name)

        # Verify it was labeled (cycle found)
        cycles = self.profile_store.get_past_cycles()
        cycle = next((c for c in cycles if c.get("id") == cycle_id), None)

        return bool(cycle and cycle.get("auto_labeled"))

    async def async_submit_cycle_feedback(
        self,
        cycle_id: str,
        user_confirmed: bool,
        corrected_profile: Optional[str] = None,
        corrected_duration: Optional[float] = None,
        notes: str = "",
        dismiss: bool = False,
    ) -> bool:
        """Submit user feedback for a cycle."""
        pending = self.profile_store.get_pending_feedback().get(cycle_id)
        if not pending:
            return False

        # Parse corrected_duration before writing to history so a bad value
        # never leaves a partially-applied state.
        duration_sec: float | None = None
        if corrected_duration is not None:
            try:
                duration_sec = float(corrected_duration)
            except (TypeError, ValueError, OverflowError):
                self._logger.warning(
                    "Invalid corrected_duration %r for cycle %s, ignoring",
                    corrected_duration,
                    cycle_id,
                )

        feedback_record = self._build_feedback_record(
            cycle_id,
            pending,
            user_confirmed=user_confirmed,
            corrected_profile=corrected_profile,
            corrected_duration=duration_sec,
            notes=notes,
        )

        self.profile_store.get_feedback_history()[cycle_id] = feedback_record

        # Track which profiles need envelope rebuild (issue #131)
        profiles_to_rebuild: set[str] = set()

        if dismiss:
            # Just dismiss, no action
            pass
        elif user_confirmed:
            profile_name = pending.get("detected_profile")
            if isinstance(profile_name, str) and profile_name:
                left = self._auto_label_cycle(
                    cycle_id, profile_name, duration_sec, source="manual"
                )
                if left:
                    profiles_to_rebuild.add(left)
                if duration_sec is not None:
                    cycles = self.profile_store.get_past_cycles()
                    confirmed_cycle = next((c for c in cycles if c.get("id") == cycle_id), None)
                    if confirmed_cycle:
                        confirmed_cycle["duration"] = duration_sec
                profiles_to_rebuild.add(profile_name)
        else:
            # Correction path: only use corrected_profile when user_confirmed is False.
            # Duration-only corrections (no profile specified) are handled by the elif branch below.
            target_profile = corrected_profile
            detected_profile_name = pending.get("detected_profile")

            if isinstance(target_profile, str) and target_profile:
                left = self._apply_correction_learning(
                    cycle_id, target_profile, duration_sec
                )
                profiles_to_rebuild.add(target_profile)
                if left:
                    profiles_to_rebuild.add(left)
                if (
                    isinstance(detected_profile_name, str)
                    and detected_profile_name
                    and detected_profile_name != target_profile
                ):
                    profiles_to_rebuild.add(detected_profile_name)
            elif duration_sec is not None:
                # No valid profile could be determined, but a duration correction was
                # explicitly provided - apply it directly to the cycle so the value
                # is never silently dropped.
                cycles = self.profile_store.get_past_cycles()
                cycle_to_fix = next((c for c in cycles if c.get("id") == cycle_id), None)
                if cycle_to_fix:
                    cycle_to_fix["duration"] = duration_sec
                    cycle_to_fix["manual_duration"] = duration_sec
                    existing_profile = cycle_to_fix.get("profile_name")
                    if isinstance(existing_profile, str) and existing_profile:
                        profiles_to_rebuild.add(existing_profile)
                else:
                    self._logger.warning(
                        "Duration correction skipped: cycle %s not found in past_cycles",
                        cycle_id,
                    )

        # Remove from pending (add_pending_feedback was wrapper, remove is direct)
        if cycle_id in self.profile_store.get_pending_feedback():
            del self.profile_store.get_pending_feedback()[cycle_id]

        # Rebuild envelopes for all modified profiles to recalculate min/max/avg (issue #131)
        for profile_name in profiles_to_rebuild:
            try:
                await self.profile_store.async_rebuild_envelope(profile_name)
            except Exception as e:  # pylint: disable=broad-exception-caught
                self._logger.error("Failed to rebuild envelope for profile '%s': %s", profile_name, e)

        # Persist changes
        await self.profile_store.async_save()
        
        # Trigger UI and sensor refresh (Issue #155)
        async_dispatcher_send(self.hass, f"ha_washdata_update_{self.entry_id}")

        return True

    async def async_answer_review(self, cycle_id: str, profile_name: str) -> bool:
        """Answer a review request with the programme the user picked.

        The one-tap answer: the detected programme is a confirmation, any other
        existing programme a correction to it. The panel's answer buttons send the
        same two ``resolve_feedback`` actions by the same rule, so a button and a
        notification action resolve a request identically. False when nothing is
        waiting for ``cycle_id`` or ``profile_name`` is not a profile.
        """
        pending = self.profile_store.get_pending_feedback().get(cycle_id)
        if not pending or not isinstance(profile_name, str) or not profile_name:
            return False
        if profile_name not in self.profile_store.get_profiles():
            return False
        if not any(c.get("id") == cycle_id for c in self.profile_store.get_past_cycles()):
            # The cycle went (a replace import, say) and left its request behind: a
            # tap on its old notification must not record an answer that labels
            # nothing and still spends the programme's review budget.
            self.profile_store.get_pending_feedback().pop(cycle_id, None)
            await self.profile_store.async_save()
            async_dispatcher_send(self.hass, f"ha_washdata_update_{self.entry_id}")
            return False
        if profile_name == pending.get("detected_profile"):
            return await self.async_submit_cycle_feedback(cycle_id, user_confirmed=True)
        return await self.async_submit_cycle_feedback(
            cycle_id, user_confirmed=False, corrected_profile=profile_name
        )

    def _build_feedback_record(
        self,
        cycle_id: str,
        pending: dict[str, Any],
        *,
        user_confirmed: bool,
        corrected_profile: Optional[str] = None,
        corrected_duration: Optional[float] = None,
        notes: str = "",
    ) -> dict[str, Any]:
        """Build a feedback_history record from a pending request + the response.

        Shared by ``async_submit_cycle_feedback`` (explicit confirm/correct/ignore)
        and ``async_resolve_pending_from_label`` (manual relabel), so both write
        an identically-shaped record.
        """
        return {
            "cycle_id": cycle_id,
            "original_detected_profile": pending.get("detected_profile"),
            "original_confidence": pending.get("confidence"),
            "user_confirmed": user_confirmed,
            "corrected_profile": corrected_profile,
            "corrected_duration": corrected_duration,
            "notes": notes,
            "submitted_at": dt_util.now().isoformat(),
        }

    async def async_resolve_pending_from_label(
        self, cycle_id: str, applied_profile: Optional[str]
    ) -> bool:
        """Resolve a pending feedback when the user manually (re)labels a cycle.

        Manually labelling a cycle that is awaiting verification IS the user's
        answer to "did WashData detect the right program?", so it must clear the
        pending feedback and drop the cycle from the review queue (issue #331).

        The label itself has already been applied by ``assign_profile_to_cycle`` /
        ``create_profile`` (which preserve ``label_source="manual"`` and rebuild the
        affected envelopes), so this only records the feedback response and removes
        the pending entry - it deliberately does NOT re-label or rebuild again.

        Returns True when a pending entry existed and was resolved, OR when no
        pending feedback was found but the cycle was sitting in the review queue
        (uncertain quality / force_stopped / interrupted) and was marked reviewed.
        Returns False when there was nothing to resolve or update.
        """
        pending = self.profile_store.get_pending_feedback().get(cycle_id)
        if not pending:
            # No detection feedback to resolve. A manual (re)label is still the user
            # engaging with the cycle, so if it is sitting in the review queue only
            # for an uncertain quality label or a force_stopped/interrupted status
            # (no pending feedback, so it has no resolve buttons), stamp it reviewed
            # to clear the red dot (#331 residual). The quality/label fields are left
            # untouched, so no training signal is lost; normal cycles are not touched.
            try:
                cycle = next(
                    (c for c in self.profile_store.get_past_cycles() if c.get("id") == cycle_id),
                    None,
                )
            except Exception:  # pylint: disable=broad-exception-caught
                cycle = None
            if cycle is not None:
                rv = cycle.get("ml_review") if isinstance(cycle.get("ml_review"), dict) else {}
                already_reviewed = bool(rv.get("reviewed_at"))
                needs_review = (
                    rv.get("label") in ("uncertain", "review")
                    or rv.get("quality") in ("uncertain", "review")
                    or cycle.get("status") in ("force_stopped", "interrupted")
                    or (cycle.get("ml_health") or {}).get("label") in ("uncertain", "review")
                )
                if needs_review and not already_reviewed:
                    try:
                        await self.profile_store.set_cycle_review(cycle_id)
                    except Exception as err:  # pylint: disable=broad-exception-caught
                        self._logger.debug(
                            "set_cycle_review failed for cycle %s: %s", cycle_id, err
                        )
                        return False
                    async_dispatcher_send(self.hass, f"ha_washdata_update_{self.entry_id}")
                    self._logger.info(
                        "Marked cycle %s reviewed from manual label (no pending feedback; "
                        "cleared needs-review red dot)", cycle_id
                    )
                    return True
            return False

        detected = pending.get("detected_profile")
        if applied_profile and applied_profile == detected:
            # User picked the same program WashData detected -> confirmation.
            user_confirmed, corrected_profile = True, None
        elif applied_profile:
            # User picked a different program -> correction.
            user_confirmed, corrected_profile = False, applied_profile
        else:
            # Label removed: the detection was rejected without naming a program.
            user_confirmed, corrected_profile = False, None

        self.profile_store.get_feedback_history()[cycle_id] = self._build_feedback_record(
            cycle_id,
            pending,
            user_confirmed=user_confirmed,
            corrected_profile=corrected_profile,
        )

        del self.profile_store.get_pending_feedback()[cycle_id]

        await self.profile_store.async_save()
        async_dispatcher_send(self.hass, f"ha_washdata_update_{self.entry_id}")
        self._logger.info(
            "Resolved pending feedback for cycle %s from manual label "
            "(detected='%s', applied='%s', confirmed=%s)",
            cycle_id,
            detected,
            applied_profile,
            user_confirmed,
        )
        return True

    def _auto_label_cycle(
        self,
        cycle_id: str,
        profile_name: str,
        manual_duration: float | None = None,
        *,
        source: str = "auto_match",
    ) -> str | None:
        """Write a label and its provenance; returns the profile the cycle left.

        ``source`` is ``"manual"`` when the user answered a review request (confirm
        or correct) and ``"auto_match"`` when the matcher's guess is recorded.
        These paths used to set only ``profile_name``, so a user's correction still
        read as a matcher guess and the panel's Auto-label reverted it - storing the
        user's answer as ``original_auto_label`` (audit MANAGER-01).

        The returned name (None when the label did not move) is the profile whose
        envelope lost a member: it is the cycle's real old label, which need not
        be the profile the review request detected (register item 493).
        """
        cycles = self.profile_store.get_past_cycles()
        cycle = next((c for c in cycles if c.get("id") == cycle_id), None)
        left: str | None = None
        if cycle:
            old = cycle.get("profile_name")
            if (
                source == "manual"
                and old
                and old != profile_name
                and not cycle.get("original_auto_label")
                and cycle.get("label_source") in _AUTO_LABEL_SOURCES
            ):
                cycle["original_auto_label"] = old
            cycle["profile_name"] = profile_name
            cycle["auto_labeled"] = True
            cycle["label_source"] = source
            if manual_duration:
                cycle["manual_duration"] = manual_duration
            if old and old != profile_name:
                # The profile the cycle left must not keep it as its sample.
                self.profile_store.heal_profile_sample(old)
                if isinstance(old, str):
                    left = old
        return left

    def _apply_correction_learning(
        self,
        cycle_id: str,
        corrected_profile: str,
        corrected_duration: Optional[float] = None,
    ) -> str | None:
        """Apply user correction to a cycle (fix for issue #131).

        Note: We do not update avg_duration here with EMA. Instead, the envelope
        rebuild in async_submit_cycle_feedback() will recalculate all statistics
        (min/max/avg) from labeled cycles, ensuring accuracy. Returns the profile
        the cycle left (see :meth:`_auto_label_cycle`).
        """
        left = self._auto_label_cycle(
            cycle_id, corrected_profile, corrected_duration, source="manual"
        )
        if corrected_duration is not None:
            cycles = self.profile_store.get_past_cycles()
            cycle = next((c for c in cycles if c.get("id") == cycle_id), None)
            if cycle:
                cycle["duration"] = corrected_duration
        # Profile stats will be recalculated when envelope is rebuilt
        return left

    async def _async_rebuild_profile_envelope(self, profile_name: str) -> None:
        """Async helper to rebuild a profile's envelope (issue #131 fix).

        This wraps async_rebuild_envelope with error handling for safe task scheduling.
        """
        try:
            await self.profile_store.async_rebuild_envelope(profile_name)
            self._logger.debug("Rebuilt envelope for profile '%s'", profile_name)
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._logger.error("Failed to rebuild envelope for profile '%s': %s", profile_name, e)

    async def _async_rebuild_and_save_profile(self, detected_profile: str) -> None:
        """Rebuild profile envelope then persist in deterministic order."""
        await self._async_rebuild_profile_envelope(detected_profile)
        await self.profile_store.async_save()

    def get_pending_feedback(self) -> dict[str, dict[str, Any]]:
        """Return pending feedback requests."""
        return dict(self.profile_store.get_pending_feedback())

    def get_feedback_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return submitted feedback history."""
        items = list(self.profile_store.get_feedback_history().values())
        items.sort(key=lambda x: x.get("submitted_at", ""), reverse=True)
        return items[:limit]
