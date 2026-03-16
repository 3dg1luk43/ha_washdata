"""Learning and self-tuning logic for WashData."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING, cast

import numpy as np
from homeassistant.core import HomeAssistant
from homeassistant.helpers import translation
from homeassistant.helpers.dispatcher import async_dispatcher_send
import homeassistant.util.dt as dt_util

from .const import (
    CONF_AUTO_LABEL_CONFIDENCE,
    CONF_DURATION_TOLERANCE,
    CONF_LEARNING_CONFIDENCE,
    CONF_MIN_OFF_GAP,
    CONF_MIN_POWER,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_OFF_DELAY,
    CONF_PROFILE_DURATION_TOLERANCE,
    CONF_PROFILE_MATCH_INTERVAL,
    CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
    CONF_SAMPLING_INTERVAL,
    CONF_WATCHDOG_INTERVAL,
    DEFAULT_AUTO_LABEL_CONFIDENCE,
    DEFAULT_DURATION_TOLERANCE,
    DEFAULT_LEARNING_CONFIDENCE,
    DOMAIN,
    SIGNAL_WASHER_UPDATE,
)
from .suggestion_engine import SuggestionEngine
from .log_utils import DeviceLoggerAdapter

if TYPE_CHECKING:
    from .profile_store import ProfileStore


_LOGGER = logging.getLogger(__name__)


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
    ) -> None:
        """Initialize the learning manager."""
        self._logger = DeviceLoggerAdapter(_LOGGER, device_name)
        self.hass = hass
        self.entry_id = entry_id
        self.profile_store = profile_store
        self.device_type = device_type
        self.suggestion_engine = SuggestionEngine(
            hass, entry_id, profile_store, device_type
        )

        # Operational Stats
        self._sample_interval_model = StatisticalModel(max_samples=200)
        self._last_suggestion_update: datetime | None = None

    def _apply_suggestions_and_notify(self, suggestions: dict[str, Any]) -> None:
        """Apply suggestions and notify once when they become actionable."""
        if not suggestions:
            return

        _actionable_keys = (
            CONF_MIN_POWER,
            CONF_OFF_DELAY,
            CONF_WATCHDOG_INTERVAL,
            CONF_NO_UPDATE_ACTIVE_TIMEOUT,
            CONF_SAMPLING_INTERVAL,
            CONF_PROFILE_MATCH_INTERVAL,
            CONF_AUTO_LABEL_CONFIDENCE,
            CONF_DURATION_TOLERANCE,
            CONF_PROFILE_DURATION_TOLERANCE,
            CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
            CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
            CONF_MIN_OFF_GAP,
        )

        def _count_actionable(s: dict) -> int:
            return sum(
                1 for k in _actionable_keys
                if isinstance(s.get(k), dict) and s[k].get("value") is not None
            )

        current = self.profile_store.get_suggestions()
        before_count = _count_actionable(current) if isinstance(current, dict) else 0

        self.suggestion_engine.apply_suggestions(suggestions)

        updated = self.profile_store.get_suggestions()
        after_count = _count_actionable(updated) if isinstance(updated, dict) else 0

        if before_count == 0 and after_count > 0:
            entry = self.hass.config_entries.async_get_entry(self.entry_id)
            device_title = entry.title if entry else DOMAIN
            self.hass.async_create_task(
                self._async_send_suggestions_ready_notification(device_title, after_count)
            )

    def process_power_reading(
        self, _power: float, now: datetime, last_reading_time: datetime | None
    ) -> None:
        """Ingest power reading metadata for statistical analysis."""
        if last_reading_time:
            delta = (now - last_reading_time).total_seconds()
            # Ignore ultra-small jitter (<0.1s) and massive gaps (>1800s - likely downtime)
            if 0.1 < delta < 1800:
                self._sample_interval_model.add_sample(delta, now)

        # Periodically update suggestions based on operational stats
        if (
            self._last_suggestion_update is None
            or (now - self._last_suggestion_update).total_seconds() > 300  # Check every 5 mins
        ):
            self._update_operational_suggestions(now)

    def process_cycle_end(
        self,
        cycle_data: dict[str, Any],
        detected_profile: str | None = None,
        confidence: float = 0.0,
        predicted_duration: float | None = None,
        match_result: Any | None = None,
    ) -> None:
        """Analyze completed cycle for learning.
        
        Args:
            cycle_data: Completed cycle data
            detected_profile: Profile name detected
            confidence: Match confidence score (0.0-1.0)
            predicted_duration: Expected duration in seconds
            match_result: MatchResult from profile_store.async_match_profile() (optional)
        """
        # 1. Trigger background simulation to find optimal parameters for this cycle
        if cycle_data.get("power_data"):
            # Offload to executor since simulation can be heavy
            self.hass.async_create_task(self._async_run_simulation(cycle_data))

        # 2. Check if we should request feedback
        self._maybe_request_feedback(
            cycle_data, detected_profile, confidence, predicted_duration, match_result
        )

        # 3. Update model-based suggestions (durations etc)
        self._update_model_suggestions(dt_util.now())

    async def _async_run_simulation(self, cycle_data: dict[str, Any]) -> None:
        """Run simulation asynchronously."""
        try:
            # Simulation runner derives optimal thresholds
            # Offload to executor since simulation can be heavy (CPU bound)
            new_suggestions = await self.hass.async_add_executor_job(
                self.suggestion_engine.run_simulation, cycle_data
            )
            if new_suggestions:
                self._apply_suggestions_and_notify(new_suggestions)
                self._logger.debug("Post-cycle simulation completed with suggestions: %s", new_suggestions.keys())
        except Exception as e:
            self._logger.error("Background simulation failed: %s", e)

    def _update_operational_suggestions(self, now: datetime) -> None:
        """Generate suggestions for operational parameters (intervals, timeouts)."""
        if self._sample_interval_model.count < 20:
            return

        p95 = self._sample_interval_model.p95
        median = self._sample_interval_model.median

        if p95 is None or median is None:
            return

        suggestions = self.suggestion_engine.generate_operational_suggestions(p95, median)
        self._apply_suggestions_and_notify(suggestions)
        self._last_suggestion_update = now

    def _update_model_suggestions(self, now: datetime) -> None:
        """Generate suggestions for model parameters (tolerances, ratios)."""
        suggestions = self.suggestion_engine.generate_model_suggestions()
        self._apply_suggestions_and_notify(suggestions)

    async def _async_send_suggestions_ready_notification(
        self, device_title: str, suggestions_count: int
    ) -> None:
        """Send a one-time persistent notification when suggestions become available."""
        try:
            notification_id = f"ha_washdata_suggestions_ready_{self.entry_id}"

            translations = await translation.async_get_translations(
                self.hass, self.hass.config.language, "options", {DOMAIN}
            )

            default_title = "WashData: Suggested Settings Ready ({device})"
            default_msg = (
                "The **Suggested Settings** sensor now reports **{count}** actionable recommendations.\n\n"
                "To review and apply them: **Settings > Devices & Services > WashData > Configure > "
                "Advanced Settings > Apply Suggested Values**.\n\n"
                "Suggestions are optional and shown for review before you save."
            )

            title_template = translations.get(
                f"component.{DOMAIN}.options.error.suggestions_ready_notification_title",
                default_title,
            )
            msg_template = translations.get(
                f"component.{DOMAIN}.options.error.suggestions_ready_notification_message",
                default_msg,
            )

            title = title_template.format(device=device_title)
            message = msg_template.format(count=suggestions_count)

            await self.hass.services.async_call(
                "persistent_notification",
                "create",
                {
                    "message": message,
                    "title": title,
                    "notification_id": notification_id,
                },
            )
        except Exception:  # pylint: disable=broad-exception-caught
            self._logger.exception("Failed to create suggestions-ready notification")

    def _set_suggestion(self, key: str, value: Any, reason: str) -> None:
        """Persist a suggested setting."""
        current: Any = self.profile_store.get_suggestions().get(key, {})
        if isinstance(current, dict):
            current_dict = cast(dict[str, Any], current)
            if current_dict.get("value") == value:
                return  # No change

        self.profile_store.set_suggestion(key, value, reason=reason)
        # We fire a background save task if possible, or rely on next periodic save.
        # Since learning manager doesn't hold reference to hass task creation easily,
        # we can just rely on ProfileStore's periodic save or trigger one if referenced.
        # Ideally ProfileStore handles dirtiness.
        # But wait, Manager calls save periodically. We should just mark it dirty?
        # ProfileStore.async_save() is needed.
        # We'll just trigger it via hass if available.
        if self.hass:
            self.hass.async_create_task(self.profile_store.async_save())

    def _maybe_request_feedback(
        self,
        cycle_data: dict[str, Any],
        detected_profile: str | None,
        confidence: float,
        predicted_duration: float | None,
        match_result: Any | None = None,
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
        duration_tol = entry.options.get(
            CONF_DURATION_TOLERANCE, DEFAULT_DURATION_TOLERANCE
        )

        # Auto-label if very high confidence
        if confidence >= auto_label_conf:
            labeled = self.auto_label_high_confidence(
                cycle_id=cycle_id,
                profile_name=detected_profile,
                confidence=confidence,
                confidence_threshold=auto_label_conf,
            )
            if labeled:
                # Rebuild envelope first, then persist (issue #131)
                self.hass.async_create_task(
                    self._async_rebuild_and_save_profile(detected_profile)
                )
                self._logger.debug("Auto-labeled high-confidence cycle %s", cycle_id)
            return

        # Skip low-confidence matches below learning threshold
        if confidence < learning_conf:
            self._logger.debug(
                "Skipping feedback for low-confidence match (conf=%.2f < %.2f)",
                confidence,
                learning_conf,
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
            duration_tolerance=duration_tol,
            match_result=match_result,
        )

        # Persist pending feedback request so it survives restart
        self.hass.async_create_task(self.profile_store.async_save())

        # Create user-visible notification
        self.hass.async_create_task(
            self._async_send_feedback_notification(
                entry.title, cycle_data, detected_profile, confidence
            )
        )

    async def _async_send_feedback_notification(
        self, device_title: str, cycle_data: dict[str, Any], profile: str, confidence: float
    ) -> None:
        """Send a persistent notification for feedback (Async with translation)."""
        try:
            cycle_id = cycle_data.get("id", "unknown")
            start_ts = cycle_data.get("start_time")
            end_ts = dt_util.now() # Approximate, or pass actual end time

            # Format times
            t_str = ""
            if start_ts:
                try:
                    s_dt = datetime.fromisoformat(str(start_ts)) if isinstance(start_ts, str) else start_ts
                    s_local = dt_util.as_local(s_dt)
                    e_local = dt_util.as_local(end_ts)
                    t_str = f"{s_local.strftime('%H:%M')} - {e_local.strftime('%H:%M')}"
                except Exception:
                    t_str = "Just now"

            notification_id = f"ha_washdata_feedback_{self.entry_id}_{cycle_id}"

            # Load translations (from en.json / localization files)
            # We use "options" category to access the error keys where we stored these strings
            translations = await translation.async_get_translations(
                self.hass, self.hass.config.language, "options", {DOMAIN}
            )

            # Default templates
            default_title = "WashData: Verify Cycle ({device})"
            default_msg = (
                 "**Device**: {device}\n"
                 "**Program**: {program} ({confidence}% confidence)\n"
                 "**Time**: {time}\n\n"
                 "WashData needs your help to verify this detected cycle.\n\n"
                 "Please go to **Settings > Devices & Services > WashData > Configure > Learning Feedbacks** to confirm or correct this result."
            )

            title_template = translations.get(
                f"component.{DOMAIN}.options.error.feedback_notification_title", default_title
            )
            msg_template = translations.get(
                f"component.{DOMAIN}.options.error.feedback_notification_message", default_msg
            )

            # Confidence as percentage
            conf_pct = int(confidence * 100)

            title = title_template.format(device=device_title)
            message = msg_template.format(
                device=device_title,
                program=profile,
                confidence=conf_pct,
                time=t_str
            )

            # Use standard service call
            await self.hass.services.async_call(
                "persistent_notification",
                "create",
                {
                    "message": message,
                    "title": title,
                    "notification_id": notification_id,
                },
            )
        except Exception:  # pylint: disable=broad-exception-caught
            self._logger.exception("Failed to create feedback notification")

    def _send_feedback_notification(
        self, device_title: str, cycle_data: dict[str, Any], profile: str, confidence: float
    ) -> None:
        """Deprecated sync wrapper."""
        self.hass.async_create_task(
            self._async_send_feedback_notification(
                device_title, cycle_data, profile, confidence
            )
        )

    def request_cycle_verification(
        self,
        cycle_id: str,
        detected_profile: Optional[str],
        confidence: float,
        estimated_duration: Optional[float],
        actual_duration: float,
        duration_tolerance: float = 0.10,
        match_result: Any | None = None,
    ) -> None:
        """Request user verification for a detected cycle."""
        duration_match_pct = (
            (actual_duration / estimated_duration * 100) if estimated_duration else 0
        )
        tolerance_pct = duration_tolerance * 100
        is_close_match = (
            estimated_duration and abs(duration_match_pct - 100) <= tolerance_pct
        )

        # Extract match ranking from MatchResult if available (for UI visualization)
        ranking_summary: list[dict[str, Any]] = []
        if match_result and hasattr(match_result, "ranking") and match_result.ranking:
            for cand in match_result.ranking[:5]:  # Store top 5
                try:
                    ranking_summary.append({
                        "name": cand.get("name", "Unknown"),
                        "score": float(cand.get("score", 0.0)),
                        "metrics": cand.get("metrics", {}),
                        "profile_duration": float(cand.get("profile_duration", 0.0)),
                    })
                except (TypeError, ValueError, KeyError, AttributeError):
                    continue

        feedback_req: dict[str, Any] = {
            "cycle_id": cycle_id,
            "detected_profile": detected_profile,
            "confidence": confidence,
            "estimated_duration": estimated_duration,
            "actual_duration": actual_duration,
            "duration_match_pct": duration_match_pct,
            "is_close_match": is_close_match,
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
        cycle = next((c for c in cycles if c["id"] == cycle_id), None)

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
            except (TypeError, ValueError):
                self._logger.warning(
                    "Invalid corrected_duration %r for cycle %s, ignoring",
                    corrected_duration,
                    cycle_id,
                )

        feedback_record: dict[str, Any] = {
            "cycle_id": cycle_id,
            "original_detected_profile": pending["detected_profile"],
            "original_confidence": pending["confidence"],
            "user_confirmed": user_confirmed,
            "corrected_profile": corrected_profile,
            "corrected_duration": duration_sec,
            "notes": notes,
            "submitted_at": dt_util.now().isoformat(),
        }

        self.profile_store.get_feedback_history()[cycle_id] = feedback_record

        # Track which profiles need envelope rebuild (issue #131)
        profiles_to_rebuild: set[str] = set()

        if dismiss:
            # Just dismiss, no action
            pass
        elif user_confirmed:
            profile_name = pending.get("detected_profile")
            if isinstance(profile_name, str) and profile_name:
                self._auto_label_cycle(cycle_id, profile_name, duration_sec)
                if duration_sec is not None:
                    cycles = self.profile_store.get_past_cycles()
                    confirmed_cycle = next((c for c in cycles if c["id"] == cycle_id), None)
                    if confirmed_cycle:
                        confirmed_cycle["duration"] = duration_sec
                profiles_to_rebuild.add(profile_name)
        else:
            # Correction path
            # If corrected_profile is missing, fallback to the original detected profile
            # This ensures duration changes (Issue #155) are saved even if the profile remains unchanged.
            target_profile = corrected_profile or pending.get("detected_profile")
            
            if isinstance(target_profile, str) and target_profile:
                detected_profile = pending.get("detected_profile")

                self._apply_correction_learning(
                    cycle_id, target_profile, duration_sec
                )
                profiles_to_rebuild.add(target_profile)
                if (
                    isinstance(detected_profile, str)
                    and detected_profile
                    and detected_profile != target_profile
                ):
                    profiles_to_rebuild.add(detected_profile)

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

    def _auto_label_cycle(self, cycle_id: str, profile_name: str, manual_duration: float | None = None) -> None:
        cycles = self.profile_store.get_past_cycles()
        cycle = next((c for c in cycles if c["id"] == cycle_id), None)
        if cycle:
            cycle["profile_name"] = profile_name
            cycle["auto_labeled"] = True
            if manual_duration:
                cycle["manual_duration"] = manual_duration

    def _apply_correction_learning(
        self,
        cycle_id: str,
        corrected_profile: str,
        corrected_duration: Optional[float] = None,
    ) -> None:
        """Apply user correction to a cycle (fix for issue #131).

        Note: We do not update avg_duration here with EMA. Instead, the envelope
        rebuild in async_submit_cycle_feedback() will recalculate all statistics
        (min/max/avg) from labeled cycles, ensuring accuracy.
        """
        self._auto_label_cycle(cycle_id, corrected_profile, corrected_duration)
        if corrected_duration is not None:
            cycles = self.profile_store.get_past_cycles()
            cycle = next((c for c in cycles if c["id"] == cycle_id), None)
            if cycle:
                cycle["duration"] = corrected_duration
        # Profile stats will be recalculated when envelope is rebuilt

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
