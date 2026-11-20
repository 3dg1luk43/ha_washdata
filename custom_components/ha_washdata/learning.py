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
import functools
import itertools
import math
import logging
from collections import deque
from datetime import datetime
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send
import homeassistant.util.dt as dt_util

from .const import (
    CONF_POWER_SENSOR,
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_LEARNING_CONFIDENCE,
    CONF_PROFILE_MIN_WARMUP_CYCLES,
    DEFAULT_AUTO_LABEL_CONFIDENCE,
    DEFAULT_LEARNING_CONFIDENCE,
    MIN_SUGGESTION_COOLDOWN_CYCLES,
    CADENCE_HOLD_REL,
    CADENCE_MIN_INTERVALS,
    CADENCE_RECENT_CYCLES,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_PROFILE_MATCH_INTERVAL,
    CONF_WATCHDOG_INTERVAL,
    MIN_SUGGESTION_REL_DELTA,
    REVIEW_BUDGET_PER_PROFILE,
    TerminationReason,
)
from .suggestion_engine import (
    END_GATE_KEYS,
    OWNER_BATCH,
    OWNER_DETECTION,
    OWNER_MODEL,
    OWNER_OPERATIONAL,
    OWNER_STANDBY_FLOOR,
    OWNER_START_THRESHOLD,
    SuggestionEngine,
)
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


#: Keys the operational pass sizes from the plug cadence: the hold band applies.
CADENCE_KEYS = (CONF_WATCHDOG_INTERVAL, CONF_NO_UPDATE_ACTIVE_TIMEOUT, CONF_PROFILE_MATCH_INTERVAL)
#: Update intervals one cycle can hold (a 24 h cycle at 5 s); bounded, not 200.
_CYCLE_MAX_INTERVALS = 20000
#: The update intervals ``process_power_reading`` collects, exclusive: shorter is
#: jitter, longer is downtime. A stored summary outside them, bounds included
#: (the store rounds to two decimals), was never measured on this plug.
_CADENCE_MIN_INTERVAL_S = 0.1
_CADENCE_MAX_INTERVAL_S = 1800.0
#: Keys the noise gate requires the relative bar for (item 637): on seconds the 5 s
#: absolute bar alone let 2692 -> 2698 s and 480 -> 470 s through. A sub-second
#: change is noise too; the 5 s bar would block 7 -> 9 s on a short key.
_SECONDS_SUFFIXES = ("_interval", "_timeout", "_delay", "_gap", "_duration", "_seconds")
_SECONDS_MIN_ABS = 1.0
#: Full runs a manual analysis makes before it reports that it was superseded.
#: Each retry needs another options save, history wipe, import or undo during
#: the run before it (a run takes about a second on 100 real cycles).
MANUAL_ANALYSIS_ATTEMPTS = 3


class AnalysisSupersededError(RuntimeError):
    """A manual analysis whose every run was superseded: an options refresh or
    a replaced store discarded part of what it computed. The WS handler sends
    it as ``analysis_superseded``."""


class EndGateCheckError(RuntimeError):
    """A manual analysis whose joint end-gate check could not run (PR #529
    review): the stored set is unjudged, so it is no complete analysis. What
    earlier passes stored stays (Apply all judges it again). The WS handler
    sends it as ``analysis_check_failed``."""


def end_gate_evidence_token(store: Any) -> tuple[Any, ...]:
    """What the joint end-gate scan reads (``SuggestionEngine._end_gate_evidence``):
    the newest 200 stored cycles, each trace by identity (an edit replaces it), and
    the store they come from. A verdict taken while it changed judged a cycle
    stored meanwhile, or not, by chance (PR #529 review)."""
    past = list(store.get_past_cycles() or [])[-200:]
    return (
        getattr(store, "_data_generation", 0), len(past),
        tuple(id(c.get("power_data")) if isinstance(c, dict) else id(c) for c in past),
    )


def cycle_cadence(intervals: Any) -> tuple[float, float] | None:
    """``(p95, median)`` of one cycle's update intervals, or None below 20 of them."""
    try:
        values = [float(v) for v in intervals]
    except (TypeError, ValueError):
        return None
    values = [v for v in values if math.isfinite(v) and v > 0]
    if len(values) < CADENCE_MIN_INTERVALS:
        return None
    arr = np.asarray(values)
    return float(np.percentile(arr, 95)), float(np.median(arr))


def _held_cadence(pending: deque[tuple[float, datetime]]) -> tuple[float, float] | None:
    """``cycle_cadence`` of held ``(interval, stamp)`` rows (an executor job).

    Empties ``pending`` (handed over, no longer shared), so its rows are freed
    in the job too rather than on the loop once the job returns.
    """
    summary = cycle_cadence(delta for delta, _ts in pending)
    pending.clear()
    return summary


def recent_cadence(summaries: Any) -> tuple[float, float] | None:
    """The median p95 and median over the last ``CADENCE_RECENT_CYCLES`` summaries.

    A row outside the collected interval range is skipped (PR #529 review): an
    imported ``[1e308, 2]`` made the operational pass's ``int(p95 * 20)``
    overflow, so every Analyse stopped before the Start Threshold correction.
    """
    rows: list[tuple[float, float]] = []
    for row in list(summaries or [])[-CADENCE_RECENT_CYCLES:]:
        try:
            p95, median = float(row[0]), float(row[1])
        except (TypeError, ValueError, IndexError, OverflowError):
            continue
        if (
            _CADENCE_MIN_INTERVAL_S <= p95 <= _CADENCE_MAX_INTERVAL_S
            and _CADENCE_MIN_INTERVAL_S <= median <= _CADENCE_MAX_INTERVAL_S
        ):
            rows.append((p95, median))
    if not rows:
        return None
    return float(np.median([r[0] for r in rows])), float(np.median([r[1] for r in rows]))


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
        self.add_samples((value,), now)

    def add_samples(self, values: Any, now: datetime) -> None:
        """Add samples and update stats once (a cycle commits thousands at a time)."""
        self._samples.extend(values)
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
    def max_samples(self) -> int:
        """How many of the newest samples the stats are taken over."""
        return self._max_samples

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
        on_suggestions_changed: Callable[[], Any] | None = None,
    ) -> None:
        """Initialize the learning manager.

        ``spawn`` starts every background task this class creates. The manager
        passes its ``_spawn_tracked`` (audit MANAGER-13): these tasks save the
        ProfileStore, and an untracked one survives an entry reload and can write
        the OLD store over the new one under the same Store key. Without it (unit
        tests) a plain ``hass.async_create_task`` is used.

        ``on_suggestions_changed`` is called when a background pass or a refresh
        has changed what is stored: the Suggested Settings sensor does not poll
        (PR #529 review). The manager's pushes an entity update.
        """
        self._logger = DeviceLoggerAdapter(_LOGGER, device_name)
        self.hass = hass
        self._spawn_fn = spawn
        self._on_suggestions_changed = on_suggestions_changed
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
        self._pending_intervals: deque[tuple[float, datetime]] = deque(maxlen=_CYCLE_MAX_INTERVALS)
        # Which sensor reported the held intervals and the in-memory model's: a
        # sensor swap is deferred while a cycle runs, so the selected sensor is not
        # always the one reporting (PR #529 review).
        self._pending_sensor: str | None = None
        self._model_sensor: str | None = None
        # Set by a cadence commit, consumed by run_due_operational_pass once the
        # cycle is stored, so the pass's pause scan sees the cycle that just ended.
        self._operational_pass_due: bool = False
        # A cycle's cadence summary is computed in the executor (PR #529 review):
        # the jobs not yet stored, and whether the due pass waits for them. The
        # epoch is bumped by ``reset_cadence_evidence`` (a wipe, Undo Import, an
        # import that replaces the history or the cadence rows), which discards a
        # summary still in flight.
        self._summaries_in_flight: int = 0
        self._operational_pass_waiting: bool = False
        self._cadence_epoch: int = 0
        # ``[sensor, summary, done]`` per committed cycle, in cycle order: a
        # summary is stored only once every earlier one is, so a late job cannot
        # store an older cycle (or the old sensor's) over a newer one (PR #529
        # review).
        self._summary_queue: deque[list[Any]] = deque()
        self._last_batch_simulation_count: int = 0  # track when to re-run batch
        self._last_suggestions_labeled_count: int = 0  # gate model/detection passes
        # Bumped by every refresh (an options reload, a device type change): a pass
        # that started before it applies nothing, since its options snapshot
        # (``for_job``) is no longer the one in force.
        self._config_generation: int = 0
        # Bumped by each refresh, so one can tell a newer refresh has started.
        self._refresh_seq: int = 0
        # One suggestion pass at a time, from its scan to its store (PR #529
        # review): a cycle end spawns several, and each end-gate guard checks its
        # values against what the others have pending. Run side by side, two
        # passes read the same old pending set, and an Off Delay and an End
        # Energy each safe alone were stored together and split a pause.
        self._pass_lock = asyncio.Lock()

    async def _run_pass(
        self,
        generate: Callable[..., dict[str, Any]],
        owner: str | None,
        generation: int,
        *args: Any,
        data_generation: int | None = None,
        strict: bool = False,
    ) -> dict[str, Any] | None:
        """Scan in the executor and apply on the loop, holding ``_pass_lock``.

        Then judge the stored set as a whole (``_enforce_joint_end_gate``): the
        gates and the reconcile after the pass's own guard can remove a pending
        value another entry was judged with. ``strict`` (the manual analysis)
        raises :class:`EndGateCheckError` when that check cannot run.
        """
        # Read before the lock wait, or where the inputs were taken (the batch pass
        # takes its cycles at spawn): a store Undo Import replaced meanwhile is not
        # the one they came from (PR #529 review). None when discarded.
        if data_generation is None:
            data_generation = getattr(self.profile_store, "_data_generation", 0)
        async with self._pass_lock:
            # Superseded while it waited for the lock (several saves during one
            # refresh queue a pass each): both generations only grow, so the scan
            # would be discarded below. Skipped before it scans the history (PR
            # #529 review): None as for a replaced store, nothing applied either way.
            if getattr(self.profile_store, "_data_generation", 0) != data_generation:
                self._logger.debug("Skipping a %s pass queued before the store was replaced", owner)
                return None
            if generation != self._config_generation:
                self._logger.debug("Skipping a %s pass queued before the options changed", owner)
                return {}
            found = await self.hass.async_add_executor_job(generate, *args)
            if getattr(self.profile_store, "_data_generation", 0) != data_generation:
                self._logger.debug("Discarding a %s pass computed against a replaced store", owner)
                return None
            before = self._stored_suggestions()
            try:
                self._apply_if_current(generation, found or {}, owner)
                if generation == self._config_generation:
                    await self._enforce_joint_end_gate(strict=strict)
            finally:
                self._announce_if_changed(before)
            return found or {}

    def _stored_suggestions(self) -> dict[str, Any]:
        """A comparable copy of the stored suggestions."""
        return {
            k: dict(v) if isinstance(v, dict) else v
            for k, v in (self.profile_store.get_suggestions() or {}).items()
        }

    def _announce_suggestions(self) -> None:
        """Tell the manager the stored suggestions changed. Never raises."""
        notify = self._on_suggestions_changed
        if notify is None:
            return
        try:
            notify()
        except Exception:  # pylint: disable=broad-exception-caught
            self._logger.debug("Suggestion update notify failed", exc_info=True)

    def _announce_if_changed(self, before: dict[str, Any]) -> None:
        """``_announce_suggestions`` when the store no longer holds ``before``."""
        if self._on_suggestions_changed is not None and self._stored_suggestions() != before:
            self._announce_suggestions()

    def schedule_joint_end_gate_check(self) -> None:
        """Spawn ``async_check_joint_end_gate`` (tracked, so an unload cancels it)."""
        self._spawn(self.async_check_joint_end_gate())

    async def async_check_joint_end_gate(self) -> None:
        """Re-judge the stored set after a change no pass made (a mute). Never raises."""
        try:
            async with self._pass_lock:
                before = self._stored_suggestions()
                try:
                    await self._enforce_joint_end_gate()
                finally:
                    self._announce_if_changed(before)
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._logger.debug("Joint end-gate check failed: %s", e)

    async def _enforce_joint_end_gate(self, *, strict: bool = False) -> None:
        """Withdraw stored entries that, applied together, split own cycles.

        The trace scan runs in the executor against a snapshot; an entry is only
        withdrawn if it is still the one that was judged. A scan that fails reads
        as nothing to withdraw, unless ``strict``: then it raises
        :class:`EndGateCheckError` (the manual analysis, PR #529 review). So does
        a strict check whose set kept changing while it was judged (a mute, a
        restore, a cycle stored meanwhile): the manual analysis must not report a
        set no check judged.
        """
        # A store replaced meanwhile (Undo Import, a restore), a cycle stored or
        # edited meanwhile, or a set a mute changed, is judged again as it now
        # stands; bounded, so a store that keeps changing cannot spin.
        for _attempt in range(3):
            stored = {
                k: dict(v) for k, v in (self.profile_store.get_suggestions() or {}).items()
                if isinstance(v, dict)
            }
            if not any(k in END_GATE_KEYS for k in stored):
                return
            generation = self._config_generation
            evidence = end_gate_evidence_token(self.profile_store)
            check = self.suggestion_engine.for_job().joint_end_gate_drops
            if not strict:
                drops = await self.hass.async_add_executor_job(check, stored)
            else:
                try:
                    drops = await self.hass.async_add_executor_job(
                        functools.partial(check, stored, strict=True)
                    )
                except Exception as err:
                    raise EndGateCheckError(f"joint end-gate check failed: {err}") from err
            # Judged against options a refresh has since replaced: the next check
            # judges the new set (PR #529 review).
            if generation != self._config_generation:
                return
            # The pending values unchanged is not enough: a cycle that completed
            # during the scan may split under the set it passed (PR #529 review).
            if end_gate_evidence_token(self.profile_store) != evidence:
                continue
            if self._withdraw_joint_drops(stored, drops or []):
                return
        if strict:
            raise EndGateCheckError("the stored set changed during every joint end-gate check")

    def _withdraw_joint_drops(self, judged: dict[str, Any], drops: list[str]) -> bool:
        """Withdraw ``drops`` from the stored set; False when it is not the one judged."""
        current = self.profile_store.get_suggestions() or {}
        # The verdict holds only for the set it judged: if any pending end-gate
        # value moved meanwhile, leave everything for a check of the new set.
        for key in END_GATE_KEYS:
            was = judged.get(key, {}).get("value") if isinstance(judged.get(key), dict) else None
            now = current.get(key, {}).get("value") if isinstance(current.get(key), dict) else None
            if was != now:
                return False
        # So for an entry withdrawn outside them (a start a cascade follows): a
        # different one stored meanwhile is not the one judged (PR #529 review).
        for key in drops:
            was = judged.get(key, {}).get("value") if isinstance(judged.get(key), dict) else None
            now = current.get(key, {}).get("value") if isinstance(current.get(key), dict) else None
            if key not in END_GATE_KEYS and now is not None and was != now:
                return False
        gone = [
            key for key in drops
            if isinstance(current.get(key), dict)
            and current[key].get("value") == judged.get(key, {}).get("value")
        ]
        if not gone:
            return True
        self._logger.info(
            "Suggestion(s) withdrawn: applied together with the rest they would end "
            "own cycles inside a pause the settings in force bridge: %s",
            ", ".join(sorted(gone)),
        )
        for key in gone:
            self.profile_store.delete_suggestion(key)
        self.suggestion_engine.apply_suggestions({}, changed=True)
        return True

    def _spawn(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Start a background task through the manager's tracked spawner."""
        if self._spawn_fn is not None:
            return self._spawn_fn(coro)
        return self.hass.async_create_task(coro)

    def _apply_suggestions_and_notify(
        self, suggestions: dict[str, Any], owner: str | None = None
    ) -> None:
        """Apply suggestions that pass quality gates.

        ``owner`` names the pass that produced them. A pass that runs and no longer
        proposes a key it proposed before withdraws that stored entry (audit
        2026-11-16): a withheld or corrected value otherwise stayed one Apply-all
        away, e.g. a pre-#527 start threshold on the running level. Call it with
        an empty map too, so the withdrawal happens.
        """
        if not suggestions:
            if owner is not None and self._withdraw_owned(owner, suggestions):
                self.suggestion_engine.apply_suggestions({}, changed=True, owner=owner)
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

        any_deleted = owner is not None and self._withdraw_owned(owner, suggestions)
        existing_suggestions = self.profile_store.get_suggestions()
        filtered_suggestions: dict[str, Any] = {}
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
                        # A seconds key needs the relative bar (item 637); others either.
                        rel_delta = abs_delta / max(abs(cv), 1e-3)
                        rel_bar = (
                            CADENCE_HOLD_REL
                            if key in CADENCE_KEYS and not data.get("corrective")
                            else MIN_SUGGESTION_REL_DELTA
                        )
                        small_rel = rel_delta < rel_bar
                        if key.endswith(_SECONDS_SUFFIXES):
                            noise = small_rel or abs_delta < _SECONDS_MIN_ABS
                        else:
                            noise = small_rel and abs_delta < _suggestion_min_abs_delta(key)
                        if noise:
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

        if not filtered_suggestions and not any_deleted:
            return

        # A deletion above is unsaved until a write: apply_suggestions saves only
        # when it changes something itself (#510), so tell it. It also re-derives
        # the cascades of what is left.
        self.suggestion_engine.apply_suggestions(
            filtered_suggestions, changed=any_deleted, owner=owner
        )

    def _withdraw_operational(self) -> None:
        """Withdraw every operational entry (no cadence to run that pass on).

        Pushes the sensor itself when that changed the store (PR #529 review):
        no ``_run_pass`` snapshot covers it, so a manual analysis whose next pass
        raised left the withdrawn entry on the non-polling sensor.
        """
        before = self._stored_suggestions()
        self._apply_suggestions_and_notify({}, owner=OWNER_OPERATIONAL)
        self._announce_if_changed(before)

    def _apply_if_current(
        self, generation: int, suggestions: dict[str, Any], owner: str | None
    ) -> None:
        """Apply a background pass's result unless a refresh superseded its snapshot."""
        if generation != self._config_generation:
            self._logger.debug("Discarding a %s pass computed against replaced options", owner)
            return
        self._apply_suggestions_and_notify(suggestions, owner=owner)

    def _withdraw_owned(self, owner: str, proposed: dict[str, Any]) -> bool:
        """Delete stored entries ``owner`` produced and no longer proposes."""
        deleted = False
        for key, entry in list((self.profile_store.get_suggestions() or {}).items()):
            if isinstance(entry, dict) and entry.get("owner") == owner and key not in proposed:
                self.profile_store.delete_suggestion(key)
                deleted = True
        return deleted

    def process_power_reading(
        self,
        _power: float,
        now: datetime,
        last_reading_time: datetime | None,
        sensor: str | None = None,
    ) -> None:
        """Hold this reading's update interval for the cycle-end cadence commit.

        No suggestion pass runs here. It used to, every 5 minutes of a running
        cycle, but everything it reads moves only at a cycle end (the cadence
        model is committed by ``close_cycle_cadence``, the traces it scans are
        stored then), so each one re-derived what the last cycle-end pass had,
        a ~100-trace executor scan for nothing (#510 follow-up). The pass runs
        once per cycle end, in ``run_due_operational_pass``.
        """
        if sensor is not None and sensor != self._pending_sensor:
            # Another sensor's intervals are not this one's cadence.
            self._pending_intervals.clear()
            self._pending_sensor = sensor
        if last_reading_time:
            delta = (now - last_reading_time).total_seconds()
            # Ignore ultra-small jitter (<0.1s) and massive gaps (>1800s - likely downtime)
            if _CADENCE_MIN_INTERVAL_S < delta < _CADENCE_MAX_INTERVAL_S:
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
        # Handed over whole, not copied: a long cycle holds up to 20000 rows.
        pending = self._pending_intervals
        self._pending_intervals = deque(maxlen=_CYCLE_MAX_INTERVALS)
        if not pending or not isinstance(cycle_data, dict):
            return False
        if cycle_data.get("status") != "completed":
            return False
        if cycle_data.get("termination_reason") in (
            TerminationReason.USER,
            TerminationReason.FORCE_STOPPED,
        ):
            return False
        sensor = self._pending_sensor or self._power_sensor()
        if sensor != self._model_sensor:
            self._sample_interval_model = StatisticalModel(max_samples=200)
            self._model_sensor = sensor
        # The model keeps its newest 200 samples: only those of the cycle reach it.
        tail = list(itertools.islice(reversed(pending), self._sample_interval_model.max_samples))
        tail.reverse()
        self._sample_interval_model.add_samples(
            (delta for delta, _ts in tail), pending[-1][1]
        )
        # The operational pass's one run per cycle: its cadence input changes only
        # here (process_power_reading runs none during the cycle). Not run here:
        # this commit comes before the cycle is stored, and the pass's pause scan
        # reads the stored cycles, so it would miss this one until the next end.
        self._operational_pass_due = True
        self._commit_cycle_summary(pending, sensor)
        return True

    def _commit_cycle_summary(self, pending: deque[tuple[float, datetime]], sensor: str | None) -> None:
        """Store the ended cycle's ``(p95, median)`` cadence summary for ``sensor``.

        It reads every interval of the cycle, not the model's 200 (PR #529 review:
        with it on the loop, the close took 1.7 ms for a 4 h cycle at 2 s and
        4.3 ms at the 20000 cap, on a desktop core), so with a running loop it is
        computed in the executor and stored back on the loop; the due operational
        pass waits for it (``run_due_operational_pass``). Inline without a loop
        (sync callers, tests).
        """
        slot: list[Any] = [sensor, None, False]
        self._summary_queue.append(slot)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            slot[1:] = [_held_cadence(pending), True]
            self._store_finished_summaries()
            return
        self._summaries_in_flight += 1
        coro = self._async_commit_cycle_summary(pending, slot, self._cadence_epoch)
        task = self._spawn(coro)
        if not isinstance(task, asyncio.Future):
            # Nothing runs it (a mocked hass): inline, or the due pass would wait forever.
            coro.close()
            self._summaries_in_flight -= 1
            slot[1:] = [_held_cadence(pending), True]
            self._store_finished_summaries()

    async def _async_commit_cycle_summary(
        self,
        pending: deque[tuple[float, datetime]],
        slot: list[Any],
        epoch: int,
    ) -> None:
        """``_commit_cycle_summary``'s executor job; then the pass that waited for it.

        Dropped only when the cadence epoch moved (``reset_cadence_evidence``): a
        wipe, Undo Import, or an import that replaced the real history or the
        cadence rows (stored, an old 90 s cycle joined an imported 2 s history,
        PR #529 review). Any other store write, a profiles-only import included,
        keeps it (PR #529 review). A summary that finished earlier and waits in
        the queue goes the same way: the reset empties the queue.
        """
        summary: tuple[float, float] | None = None
        try:
            summary = await self.hass.async_add_executor_job(_held_cadence, pending)
        except Exception:  # pylint: disable=broad-exception-caught
            self._logger.debug("Could not summarise the cycle cadence", exc_info=True)
        finally:
            if epoch == self._cadence_epoch:
                self._summaries_in_flight = max(0, self._summaries_in_flight - 1)
                # Finished, even cancelled: a later cycle's summary never waits on it.
                slot[1:] = [summary, True]
        if epoch != self._cadence_epoch:
            # Replaced while it ran: a cycle that ends against the replacement
            # stores its own.
            return
        if self._store_finished_summaries():
            # It may land after the cycle-end write: saved, or a restart with no
            # later write loses it (PR #529 review). Inside the cycle-end burst
            # this joins its one write (``coalesce_saves``).
            try:
                await self.profile_store.async_save()
            except Exception:  # pylint: disable=broad-exception-caught
                self._logger.debug("Could not save the cycle cadence", exc_info=True)
        if not self._summaries_in_flight and self._operational_pass_waiting:
            self.run_due_operational_pass()

    def _store_finished_summaries(self) -> bool:
        """Store the finished summaries at the head of the queue, in cycle order."""
        stored = False
        queue = self._summary_queue
        while queue and queue[0][2]:
            sensor, summary, _done = queue.popleft()
            stored = self._store_cycle_summary(summary, sensor) or stored
        return stored

    def _store_cycle_summary(self, summary: tuple[float, float] | None, sensor: str | None) -> bool:
        """Keep ``summary`` for ``sensor``; whether it was stored. Never raises."""
        if summary is None:
            return False
        try:
            self.profile_store.add_cadence_summary(*summary, sensor=sensor)
        except Exception:  # pylint: disable=broad-exception-caught
            self._logger.debug("Could not store the cycle cadence", exc_info=True)
            return False
        return True

    def cadence(self) -> tuple[float, float] | None:
        """``(p95, median)`` update interval the operational pass sizes from (item 637).

        The median over the stored per-cycle summaries (``recent_cadence``), which
        survive a restart; before any is stored, the newest committed intervals.
        """
        sensor = self._power_sensor()
        try:
            stored = recent_cadence(self.profile_store.get_cadence_summaries(sensor))
        except Exception:  # pylint: disable=broad-exception-caught
            stored = None
        if stored is not None:
            return stored
        model = self._sample_interval_model
        if self._model_sensor is not None and self._model_sensor != sensor:
            return None  # another sensor's intervals
        if model.count < CADENCE_MIN_INTERVALS or model.p95 is None or model.median is None:
            return None
        return model.p95, model.median

    def _power_sensor(self) -> str | None:
        """The entry's power sensor, the identity of its cadence. Never raises."""
        try:
            entry = self.hass.config_entries.async_get_entry(self.entry_id)
            if entry is None:
                return None
            sensor = {**entry.data, **entry.options}.get(CONF_POWER_SENSOR)
            return str(sensor) if sensor else None
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    def reset_cadence_evidence(self) -> None:
        """Forget the update intervals measured so far and the passes run on a
        history a wipe or an import replaced (PR #529 review): without this the
        in-memory model kept sizing the operational pass from before it.

        The stored per-cycle summaries are not touched here: they describe the
        plug, not the history. A wipe clears them, an import takes the payload's
        pair or keeps this device's (``_CADENCE_PAIR_KEYS``), and Undo Import puts
        the restore point's back."""
        self._sample_interval_model = StatisticalModel(max_samples=200)
        self._model_sensor = None
        self._pending_intervals.clear()
        # The labelled-cycle counts the model, detection and batch passes wait to
        # exceed describe the history just replaced: kept, those passes stayed off
        # until the new history outgrew the old one (review of PR #529).
        self._last_suggestions_labeled_count = 0
        self._last_batch_simulation_count = 0
        # A pass already in the executor read the old cadence: its result must not
        # refill the wiped store (PR #529 review). A new cadence commit schedules a
        # pass in the new generation. A summary still in the executor goes too.
        self._config_generation += 1
        self._operational_pass_due = False
        self._cadence_epoch += 1
        self._summaries_in_flight = 0
        self._summary_queue.clear()
        self._operational_pass_waiting = False

    def run_due_operational_pass(self) -> None:
        """Run the operational pass a cadence commit left due, once.

        Called once the ended cycle is stored (``process_cycle_end``) and, for an
        end that stores nothing (ghost, pump-out, failed write), by the manager
        directly: the stored cycles did not change there, so nothing is missed.
        """
        if not self._operational_pass_due:
            return
        if self._summaries_in_flight:
            # The cycle's summary is still in the executor: its commit runs the pass.
            self._operational_pass_waiting = True
            return
        self._operational_pass_due = False
        self._operational_pass_waiting = False
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

        # 1b. Thresholds no cycle can end or start under (#458, #527). Every cycle
        # end, whatever its status: a force-stopped cycle is the evidence these
        # passes exist for.
        self.run_threshold_correctives()

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

    def run_threshold_correctives(self) -> None:
        """The two passes that correct a threshold no cycle can end or start under.

        The standby floor raises a stop threshold under the appliance's standby
        (#458); the start pass lowers a start threshold on a steady load's running
        level (#527). Called at every cycle end and once HA has started: the #527
        fault stops cycles from starting at all, so it never produces a cycle end.
        """
        self._dispatch_scan_and_apply(
            self.suggestion_engine.for_job().generate_standby_floor_suggestions,
            "Standby floor",
            owner=OWNER_STANDBY_FLOOR,
        )
        self._dispatch_scan_and_apply(
            self.suggestion_engine.for_job().generate_start_threshold_suggestions,
            "Start threshold",
            owner=OWNER_START_THRESHOLD,
        )

    def _maybe_run_batch_simulation(self) -> None:
        """Schedule a batch simulation when enough new labeled cycles have arrived."""
        _BATCH_MIN = 5
        _BATCH_RERUN_DELTA = 5  # Re-run every 5 new labeled cycles

        labeled_cycles = self._batch_cycles()
        # The list still carries force-stopped and untraced cycles - the
        # min_off_gap merge ceiling needs the user's real turnaround - but only
        # traced cycles that ended on their own are new evidence for the re-run
        # cadence (#458).
        current_count = sum(
            1 for c in labeled_cycles if c.get("power_data") and _ended_on_its_own(c)
        )

        if current_count < _BATCH_MIN:
            return
        if (current_count - self._last_batch_simulation_count) < _BATCH_RERUN_DELTA:
            return

        self._last_batch_simulation_count = current_count
        self._spawn(self._async_run_batch_simulation(
            labeled_cycles, self._config_generation,
            getattr(self.profile_store, "_data_generation", 0),
        ))

    async def _async_run_batch_simulation(
        self,
        cycles: list[dict[str, Any]],
        generation: int | None = None,
        data_generation: int | None = None,
    ) -> None:
        """Run multi-cycle batch simulation asynchronously."""
        gen = self._config_generation if generation is None else generation
        try:
            engine = self.suggestion_engine.for_job()
            new_suggestions = await self._run_pass(
                engine.run_batch_simulation, OWNER_BATCH, gen, cycles,
                data_generation=data_generation,
            )
            if new_suggestions is None:
                # Its cycles were replaced (Undo Import): the restored history has
                # had no batch, so the next cycle end runs one (PR #529 review).
                self._last_batch_simulation_count = 0
                return
            if new_suggestions:
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
        cad = self.cadence()
        if cad is None:
            # No cadence for this sensor (a new one, a wipe): what the pass stored
            # was measured on another, so it goes (PR #529 review).
            self._withdraw_operational()
            return
        p95, median = cad

        # Bind the config snapshot here, on the loop, not inside the executor job.
        engine = self.suggestion_engine.for_job()
        self._dispatch_scan_and_apply(
            lambda: engine.generate_operational_suggestions(p95, median),
            "Operational",
            owner=OWNER_OPERATIONAL,
        )

    def _update_model_suggestions(self) -> None:
        """Generate suggestions for model parameters (tolerances, ratios).

        The historical-cycle scan inside ``generate_model_suggestions`` is
        offloaded to an executor thread by ``_dispatch_scan_and_apply``.
        """
        self._dispatch_scan_and_apply(
            self.suggestion_engine.for_job().generate_model_suggestions,
            "Model",
            owner=OWNER_MODEL,
        )

    def _dispatch_scan_and_apply(
        self, generate: Callable[[], dict[str, Any]], label: str, owner: str | None = None
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
            self._apply_suggestions_and_notify(suggestions or {}, owner=owner)
            stored = {
                k: dict(v) for k, v in (self.profile_store.get_suggestions() or {}).items()
                if isinstance(v, dict)
            }
            if any(k in END_GATE_KEYS for k in stored):
                self._withdraw_joint_drops(
                    stored, self.suggestion_engine.for_job().joint_end_gate_drops(stored)
                )
            return
        self._spawn(self._async_scan_and_apply(generate, label, owner, self._config_generation))

    async def _async_scan_and_apply(
        self,
        generate: Callable[[], dict[str, Any]],
        label: str,
        owner: str | None = None,
        generation: int | None = None,
    ) -> None:
        """Offload ``generate`` to an executor thread, then apply on the loop."""
        gen = self._config_generation if generation is None else generation
        try:
            await self._run_pass(generate, owner, gen)
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._logger.error("%s suggestion pass failed: %s", label, e)

    def _update_detection_suggestions(self) -> None:
        """Generate statistical detection suggestions from clean cycles.

        Offloaded to an executor because it scans power traces across up to 200
        cycles for the clean-cycle health checks.
        """
        self._spawn(self._async_run_detection_suggestions(self._config_generation))

    async def _async_run_detection_suggestions(
        self, generation: int | None = None, *, raise_errors: bool = False
    ) -> None:
        """Run the detection-suggestion pass off the event loop."""
        gen = self._config_generation if generation is None else generation
        try:
            engine = self.suggestion_engine.for_job()
            new_suggestions = await self._run_pass(
                engine.generate_detection_suggestions, OWNER_DETECTION, gen,
                strict=raise_errors,
            )
            if new_suggestions:
                self._logger.debug(
                    "Detection suggestions produced: %s", list(new_suggestions.keys())
                )
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._logger.error("Detection suggestion pass failed: %s", e)
            if raise_errors:
                raise

    def _generations(self) -> tuple[int, int]:
        """The options and stored-data generations a run's results belong to."""
        return self._config_generation, getattr(self.profile_store, "_data_generation", 0)

    async def async_run_full_analysis(self) -> dict[str, int]:
        """Run every suggestion pass now (manual trigger from the panel).

        Runs the operational (cadence), threshold-corrective (standby floor, start
        threshold), model, detection and batch-simulation passes over the
        accumulated cycle history and reconciles the result. Each pass withdraws
        what it owns and no longer proposes.
        Returns ``{"count": <actionable suggestions>}``. A pass that raises is
        raised to the caller (PR #529 review): the passes after it never ran, so
        the count is no complete analysis and must not be reported as one. So is
        a joint end-gate check that could not run (:class:`EndGateCheckError`).

        A run that an options refresh or a replaced store superseded had results
        discarded (``_run_pass``), so it is no complete analysis either, nor is its
        error one about the current data: it runs again on the current options and
        data, whether it returned or raised, and after
        ``MANUAL_ANALYSIS_ATTEMPTS`` superseded runs raises
        :class:`AnalysisSupersededError` (PR #529 review).
        """
        self._logger.info("Manual suggestion analysis requested")
        for _attempt in range(MANUAL_ANALYSIS_ATTEMPTS):
            started = self._generations()
            try:
                await self._async_run_all_passes(raise_errors=True)
            except Exception:
                # A pass that failed on options or a store replaced under it is
                # superseded like one that returned (PR #529 review): run again.
                if self._generations() == started:
                    raise
                self._logger.debug(
                    "Manual suggestion analysis failed after it was superseded", exc_info=True
                )
                continue
            if self._generations() == started:
                count = len(self.profile_store.get_suggestions() or {})
                self._logger.info(
                    "Manual suggestion analysis complete: %d suggestion(s)", count
                )
                return {"count": count}
            self._logger.debug("Manual suggestion analysis superseded; running it again")
        raise AnalysisSupersededError(
            "the options or the stored data changed during every run"
        )

    async def async_refresh_suggestions(self, *, clear: bool = False) -> None:
        """Re-derive every stored suggestion with the current code and options.

        Run once HA has started and after an options change (audit 2026-11-16): a
        suggestion computed against other options, another device type or an
        older version is otherwise stored until its own pass happens to run, and
        a threshold no cycle can start under never produces the cycle end that
        runs the passes. ``clear`` drops every stored entry first (a device type
        change); otherwise only entries no pass owns (stored before passes owned
        their entries) are dropped. A store replaced during the run (a wipe, an
        import, Undo Import) discards what it computed, so it runs once more.
        Never raises.

        Ends with an entity update whatever it changed: the options or device type
        it follows decide which stored entries the sensor shows (PR #529 review).
        """
        self._config_generation += 1
        self._refresh_seq += 1
        seq = self._refresh_seq
        try:
            for _attempt in range(2):
                data_generation = getattr(self.profile_store, "_data_generation", 0)
                stored = self.profile_store.get_suggestions() or {}
                stale = [
                    key for key, entry in stored.items()
                    if clear or not isinstance(entry, dict) or not entry.get("owner")
                ]
                for key in stale:
                    self.profile_store.delete_suggestion(key)
                if stale:
                    self.suggestion_engine.apply_suggestions({}, changed=True)
                await self._async_run_all_passes()
                # A wipe, import or Undo Import that replaced the store during the
                # run took what its passes computed with it, and with the options
                # unchanged nothing else re-derives them for the replacement (PR #529
                # review: the #527 Start Threshold correction, which a plug in that
                # fault never ends the cycle to recompute). Once more, unless a newer
                # refresh is already doing it.
                if (
                    getattr(self.profile_store, "_data_generation", 0) == data_generation
                    or seq != self._refresh_seq
                ):
                    break
                self._logger.debug("Store replaced during the suggestion refresh; running it again")
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._logger.error("Suggestion refresh failed: %s", e)
        self._announce_suggestions()

    def _batch_cycles(self) -> list[dict[str, Any]]:
        """The cycles the batch pass reads: labelled, completed or force-stopped.

        Force-stopped ones stay in for the min_off_gap merge ceiling (#458), and
        so do cycles with no trace (PR #529 review): their start and end are the
        user's real turnaround, and dropped they fused two gaps into one. The
        batch's own trace checks keep them out of every waveform statistic.
        """
        return [
            c for c in self.profile_store.get_past_cycles()
            if isinstance(c, dict)
            and c.get("profile_name")
            and c.get("profile_name") != "noise"
            and c.get("status") in ("completed", "force_stopped")
        ]

    async def _async_run_all_passes(self, *, raise_errors: bool = False) -> None:
        """Every suggestion pass, in cycle-end order, each with its owner.

        Stops as soon as a refresh (or a history wipe) supersedes the options it
        started on: every later pass would be discarded (PR #529 review). Each
        pass runs on its own: one that raises is logged and the rest still run
        (PR #529 review: an operational scan that raised at startup ended the
        refresh before the #527 Start Threshold correction, and a device that
        cannot start a cycle never reaches the cycle end that runs it again).
        ``raise_errors`` (the manual analysis) instead ends the run and hands
        the error to the caller.
        """
        gen = self._config_generation

        async def _operational() -> None:
            cad = self.cadence()
            if cad is not None:
                p95, median = cad
                await self._run_pass(
                    self.suggestion_engine.for_job().generate_operational_suggestions,
                    OWNER_OPERATIONAL, gen, p95, median, strict=raise_errors,
                )
            else:
                # A power sensor with no cadence yet: entries measured on the old
                # one go rather than wait for a pass that will not run (PR #529 review).
                self._withdraw_operational()

        async def _owned(owner: str, method: str) -> None:
            generate = getattr(self.suggestion_engine.for_job(), method)
            await self._run_pass(generate, owner, gen, strict=raise_errors)

        async def _detection() -> None:
            await self._async_run_detection_suggestions(gen, raise_errors=raise_errors)

        async def _batch() -> None:
            await self._run_pass(
                self.suggestion_engine.for_job().run_batch_simulation,
                OWNER_BATCH, gen, self._batch_cycles(), strict=raise_errors,
            )

        passes: tuple[tuple[str, Callable[[], Awaitable[None]]], ...] = (
            (OWNER_OPERATIONAL, _operational),
            (OWNER_STANDBY_FLOOR, functools.partial(
                _owned, OWNER_STANDBY_FLOOR, "generate_standby_floor_suggestions")),
            (OWNER_START_THRESHOLD, functools.partial(
                _owned, OWNER_START_THRESHOLD, "generate_start_threshold_suggestions")),
            (OWNER_MODEL, functools.partial(_owned, OWNER_MODEL, "generate_model_suggestions")),
            (OWNER_DETECTION, _detection),
            (OWNER_BATCH, _batch),
        )
        for name, run in passes:
            if gen != self._config_generation:
                return
            try:
                await run()
            except Exception as e:  # pylint: disable=broad-exception-caught
                if raise_errors:
                    self._logger.error("Suggestion analysis failed: %s", e)
                    raise
                self._logger.error("Suggestion pass %s failed: %s", name, e)

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
        """Submit user feedback for a cycle.

        False, with nothing written, when no review is pending for the cycle or
        the answer names a programme that no longer exists (deleting a programme
        keeps the requests that detected it, so confirming one labelled the cycle
        with the deleted name).
        """
        pending = self.profile_store.get_pending_feedback().get(cycle_id)
        if not pending:
            return False
        if not dismiss:
            target = pending.get("detected_profile") if user_confirmed else corrected_profile
            if (
                isinstance(target, str)
                and target
                and target not in self.profile_store.get_profiles()
            ):
                self._logger.info(
                    "Review answer for cycle %s names programme %r, which no longer "
                    "exists; nothing recorded",
                    cycle_id,
                    target,
                )
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
