"""Learning and self-tuning logic for HA WashData."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
from homeassistant.core import HomeAssistant
from homeassistant.helpers import storage, start, translation
import homeassistant.util.dt as dt_util

from .const import (
    CONF_WATCHDOG_INTERVAL,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_OFF_DELAY,
    CONF_PROFILE_MATCH_INTERVAL,
    CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
    CONF_DURATION_TOLERANCE,
    CONF_PROFILE_DURATION_TOLERANCE,
    CONF_LEARNING_CONFIDENCE,
    DEFAULT_LEARNING_CONFIDENCE,
    CONF_AUTO_LABEL_CONFIDENCE,
    DEFAULT_AUTO_LABEL_CONFIDENCE,
    DEFAULT_DURATION_TOLERANCE,
    DOMAIN,
)

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
        self, hass: HomeAssistant, entry_id: str, profile_store: "ProfileStore"
    ) -> None:
        """Initialize the learning manager."""
        self.hass = hass
        self.entry_id = entry_id
        self.profile_store = profile_store

        # Operational Stats
        self._sample_interval_model = StatisticalModel(max_samples=200)
        self._last_suggestion_update: datetime | None = None

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
    ) -> None:
        """Analyze completed cycle for learning."""
        # Check if we should request feedback
        self._maybe_request_feedback(
            cycle_data, detected_profile, confidence, predicted_duration
        )
        
        # Update model-based suggestions (durations etc)
        self._update_model_suggestions(dt_util.now())

    def _update_operational_suggestions(self, now: datetime) -> None:
        """Generate suggestions for operational parameters (intervals, timeouts)."""
        if self._sample_interval_model.count < 20:
            return

        p95 = self._sample_interval_model.p95
        median = self._sample_interval_model.median
        
        if p95 is None or median is None:
            return

        if p95 is None or median is None:
            return
        # Ensure we have defaults available if needed - mostly we just check diffs
        
        # 1. Watchdog Interval
        # OLD: max(1, round(p95)) -> Too strict (e.g. 3s)
        # NEW: max(30, p95 * 10) -> Robust against network blips
        # Rationale: Watchdog kills the cycle. We rarely need it < 30s.
        suggested_watchdog = int(max(30, p95 * 10))
        self._set_suggestion(
            CONF_WATCHDOG_INTERVAL,
            suggested_watchdog,
            f"Based on observed update cadence (p95={p95:.1f}s) * 10 (min 30s buffer)."
        )

        # 2. No Update Timeout
        # OLD: max(off_delay, p95 * 4) -> Too strict
        # NEW: max(60, off_delay * 2, p95 * 20) -> Very robust
        # Rationale: This is "sensor dead" check. Give it plenty of time.
        # off_delay = getattr(self.profile_store, "_off_delay", DEFAULT_OFF_DELAY)
        
        suggested_timeout = int(max(60, p95 * 20))
        self._set_suggestion(
            CONF_NO_UPDATE_ACTIVE_TIMEOUT,
            suggested_timeout,
            f"Based on observed update cadence (p95={p95:.1f}s) * 20 (min 60s)."
        )

        # 3. Off Delay
        # OLD: max(60, p95 * 4) -> OK, but could be smarter.
        # If we have erratic updates, off_delay needs to be higher to bridge gaps.
        suggested_off_delay = int(max(60, p95 * 5))
        self._set_suggestion(
            CONF_OFF_DELAY,
            suggested_off_delay,
            f"Based on observed update cadence (p95={p95:.1f}s) * 5 (min 60s)."
        )

        # 4. Profile Match Interval
        # OLD: max(10, median * 10) -> Reasonable.
        # Keep consistent: ~10 samples per match attempt.
        suggested_match = int(max(10, median * 10))
        self._set_suggestion(
            CONF_PROFILE_MATCH_INTERVAL,
            suggested_match,
            f"Based on observed update cadence (median={median:.1f}s) * 10."
        )

        self._last_suggestion_update = now

    def _update_model_suggestions(self, now: datetime) -> None:
        """Generate suggestions for model parameters (tolerances, ratios)."""
        # (This logic essentially migrates from manager._compute_duration_variance_p95 etc)
        # Re-implemented to be robust.
        if now:  # Use argument to quiet linter (and potentially use for expiry logic)
            pass

        # 1. Duration Tolerance
        # Filter cycles to clean ones (no interrupts, labeled)
        cycles = self.profile_store.get_past_cycles()[-100:]
        profiles = self.profile_store.get_profiles()
        
        ratios = []
        for c in cycles:
            if not c.get("profile_name") or c.get("status") == "interrupted":
                continue
            prof = profiles.get(c["profile_name"])
            if not prof:
                continue
            avg = prof.get("avg_duration", 0)
            dur = c.get("duration", 0)
            if avg > 60 and dur > 60:
                ratios.append(dur / avg)

        if len(ratios) >= 10:
            arr = np.array(ratios)
            # Deviation from 1.0
            deviations = np.abs(arr - 1.0)
            p95_dev = float(np.percentile(deviations, 95))
            
            # Suggest tolerance: cover p95 deviation + small buffer (0.05)
            # Bound between 0.10 and 0.50
            suggested_tol = min(0.50, max(0.10, round(p95_dev + 0.05, 2)))
            
            self._set_suggestion(
                CONF_DURATION_TOLERANCE,
                suggested_tol,
                f"Based on duration variance of {len(ratios)} recent labeled cycles "
                f"(p95 dev={p95_dev:.2f}).",
            )
            self._set_suggestion(
                CONF_PROFILE_DURATION_TOLERANCE,
                suggested_tol,
                f"Based on duration variance of {len(ratios)} recent labeled cycles "
                f"(p95 dev={p95_dev:.2f}).",
            )

            # 2. Match Duration Ratios (Min/Max)
            # Use raw ratios to find bounds
            p05_ratio = float(np.percentile(arr, 5))
            p95_ratio = float(np.percentile(arr, 95))
            
            # Min: slightly below p05, capped at 0.1
            min_r = max(0.1, round(p05_ratio - 0.1, 2))
            # Max: slightly above p95, capped at 3.0
            max_r = min(3.0, round(p95_ratio + 0.1, 2))
            
            if min_r < max_r - 0.2: # Ensure gap
                self._set_suggestion(
                    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
                    min_r,
                    f"Based on labeled cycle durations (p05={p05_ratio:.2f})."
                )
                self._set_suggestion(
                    CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
                    max_r,
                    f"Based on labeled cycle durations (p95={p95_ratio:.2f})."
                )

    def _set_suggestion(self, key: str, value: Any, reason: str) -> None:
        """Persist a suggested setting."""
        current = self.profile_store.get_suggestions().get(key, {})
        if isinstance(current, dict) and current.get("value") == value:
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
            _LOGGER.warning("Cycle data missing ID, cannot request feedback")
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
                # Persist label and rebuild envelope for the profile
                self.hass.async_create_task(self.profile_store.async_save())
                try:
                    self.profile_store.rebuild_envelope(detected_profile)
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
                _LOGGER.debug("Auto-labeled high-confidence cycle %s", cycle_id)
            return

        # Skip low-confidence matches below learning threshold
        if confidence < learning_conf:
            _LOGGER.debug(
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
            _LOGGER.exception("Failed to create feedback notification")

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
    ) -> None:
        """Request user verification for a detected cycle."""
        duration_match_pct = (
            (actual_duration / estimated_duration * 100) if estimated_duration else 0
        )
        tolerance_pct = duration_tolerance * 100
        is_close_match = (
            estimated_duration and abs(duration_match_pct - 100) <= tolerance_pct
        )

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
        }

        self.profile_store.get_pending_feedback()[cycle_id] = feedback_req
        self.profile_store.add_pending_feedback(cycle_id, feedback_req)
        
        est_min = int(estimated_duration / 60) if estimated_duration else 0
        _LOGGER.info(
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

        feedback_record: dict[str, Any] = {
            "cycle_id": cycle_id,
            "original_detected_profile": pending["detected_profile"],
            "original_confidence": pending["confidence"],
            "user_confirmed": user_confirmed,
            "corrected_profile": corrected_profile,
            "corrected_duration": corrected_duration,
            "notes": notes,
            "submitted_at": dt_util.now().isoformat(),
        }

        self.profile_store.get_feedback_history()[cycle_id] = feedback_record
        
        if dismiss:
             # Just dismiss, no action
             pass
        elif user_confirmed:
            profile_name = pending.get("detected_profile")
            if isinstance(profile_name, str) and profile_name:
                self._auto_label_cycle(cycle_id, profile_name)
        else:
            if isinstance(corrected_profile, str) and corrected_profile:
                # corrected_duration is in minutes from UI, convert to seconds
                duration_sec = corrected_duration * 60.0 if corrected_duration else None
                
                self._apply_correction_learning(
                    cycle_id, corrected_profile, duration_sec
                )
                self._auto_label_cycle(cycle_id, corrected_profile, duration_sec)

        # Remove from pending (add_pending_feedback was wrapper, remove is direct)
        if cycle_id in self.profile_store.get_pending_feedback():
            del self.profile_store.get_pending_feedback()[cycle_id]
        
        # self.profile_store.remove_pending_feedback(cycle_id) # Redundant if we delete directly above

        await self.profile_store.async_save()

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
        self._auto_label_cycle(cycle_id, corrected_profile)
        # Update profile avg duration with simple EMA
        if corrected_duration:
            profile = self.profile_store.get_profiles().get(corrected_profile)
            if profile:
                old = profile.get("avg_duration", corrected_duration)
                profile["avg_duration"] = old * 0.8 + corrected_duration * 0.2

    def get_pending_feedback(self) -> dict[str, dict[str, Any]]:
        """Return pending feedback requests."""
        return dict(self.profile_store.get_pending_feedback())

    def get_feedback_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return submitted feedback history."""
        items = list(self.profile_store.get_feedback_history().values())
        items.sort(key=lambda x: x.get("submitted_at", ""), reverse=True)
        return items[:limit]

