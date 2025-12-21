"""Learning and self-tuning logic for HA WashData."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING

from homeassistant.core import HomeAssistant

if TYPE_CHECKING:
    from .profile_store import ProfileStore


_LOGGER = logging.getLogger(__name__)


class LearningManager:
    """Manages cycle learning, user feedback, and auto-tuning."""

    def __init__(self, hass: HomeAssistant, entry_id: str, profile_store: "ProfileStore") -> None:
        """Initialize the learning manager."""
        self.hass = hass
        self.entry_id = entry_id
        self.profile_store = profile_store

        # Feedback history: track user confirmations and corrections.
        # Persist these into ProfileStore so restarts don't lose learning context.
        # Mutable mappings persisted by ProfileStore
        self._feedback_history = self.profile_store.get_feedback_history()
        self._pending_feedback = self.profile_store.get_pending_feedback()

    def request_cycle_verification(
        self,
        cycle_id: str,
        detected_profile: Optional[str],
        confidence: float,
        estimated_duration: Optional[float],
        actual_duration: float,
        duration_tolerance: float = 0.10,
    ) -> None:
        """
        Request user verification for a detected cycle.
        
        Called when cycle finishes and we made a confident match.
        Stores pending feedback request for UI to pick up.
        """
        duration_match_pct = (
            (actual_duration / estimated_duration * 100) if estimated_duration else 0
        )

        tolerance_pct = duration_tolerance * 100
        is_close_match = (
            estimated_duration
            and abs(duration_match_pct - 100) <= tolerance_pct
        )

        feedback_req: dict[str, Any] = {
            "cycle_id": cycle_id,
            "detected_profile": detected_profile,
            "confidence": confidence,
            "estimated_duration": estimated_duration,
            "actual_duration": actual_duration,
            "duration_match_pct": duration_match_pct,
            "is_close_match": is_close_match,
            "created_at": datetime.now().isoformat(),
            "user_response": None,
            "expires_at": None,
        }

        self._pending_feedback[cycle_id] = feedback_req

        # Persist pending feedback so UI/automations can survive restart
        self.profile_store.get_pending_feedback()[cycle_id] = feedback_req

        est_min = int(estimated_duration / 60) if estimated_duration else 0
        _LOGGER.info(
            f"Feedback requested for cycle {cycle_id}: "
            f"profile='{detected_profile}' (conf={confidence:.2f}), "
            f"est={est_min}min, actual={int(actual_duration/60)}min "
            f"({duration_match_pct:.0f}%) - is_close={is_close_match} (tolerance=±{tolerance_pct:.0f}%)"
        )

    def submit_cycle_feedback(
        self,
        cycle_id: str,
        user_confirmed: bool,
        corrected_profile: Optional[str] = None,
        corrected_duration: Optional[float] = None,
        notes: str = "",
    ) -> bool:
        """
        Submit user feedback for a cycle.
        
        Args:
            cycle_id: The cycle ID
            user_confirmed: True if user confirmed the detected profile was correct
            corrected_profile: If user disagrees, the correct profile name
            corrected_duration: If user disagrees on time, the actual duration in seconds
            notes: Optional user notes
            
        Returns:
            True if feedback was processed, False if cycle not found
        """
        pending = self._pending_feedback.get(cycle_id)
        if not pending:
            _LOGGER.warning(f"No pending feedback request for cycle {cycle_id}")
            return False

        feedback_record: dict[str, Any] = {
            "cycle_id": cycle_id,
            "original_detected_profile": pending["detected_profile"],
            "original_confidence": pending["confidence"],
            "user_confirmed": user_confirmed,
            "corrected_profile": corrected_profile,
            "corrected_duration": corrected_duration,
            "notes": notes,
            "submitted_at": datetime.now().isoformat(),
        }

        self._feedback_history[cycle_id] = feedback_record

        # Persist feedback record
        self.profile_store.get_feedback_history()[cycle_id] = feedback_record

        if user_confirmed:
            # User confirmed the detected profile - auto-label the cycle
            profile_name = pending.get("detected_profile")
            if isinstance(profile_name, str) and profile_name:
                self._auto_label_cycle(cycle_id, profile_name)
            
            _LOGGER.info(
                f"User confirmed cycle {cycle_id}: profile='{profile_name}' "
                f"duration={int(pending['actual_duration']/60)}min - auto-labeled"
            )
        else:
            # User provided correction - learn from it and auto-label with correct profile
            _LOGGER.info(
                f"User corrected cycle {cycle_id}: "
                f"detected='{pending['detected_profile']}' -> correct='{corrected_profile}', "
                f"corrected_duration={int(corrected_duration/60) if corrected_duration else 'N/A'}min"
            )

            # Apply correction learning and auto-label with corrected profile
            if isinstance(corrected_profile, str) and corrected_profile and corrected_profile != pending.get("detected_profile"):
                self._apply_correction_learning(cycle_id, corrected_profile, corrected_duration)
                self._auto_label_cycle(cycle_id, corrected_profile)

        # Remove from pending
        del self._pending_feedback[cycle_id]
        pending_map = self.profile_store.get_pending_feedback()
        if cycle_id in pending_map:
            del pending_map[cycle_id]
        return True

    def _apply_correction_learning(
        self,
        cycle_id: str,
        corrected_profile: str,
        corrected_duration: Optional[float] = None,
    ) -> None:
        """Apply learning from user correction."""
        # Fetch the cycle from storage
        cycles = self.profile_store.get_past_cycles()
        cycle = next((c for c in cycles if c["id"] == cycle_id), None)

        if not cycle:
            _LOGGER.warning(f"Cycle {cycle_id} not found in storage")
            return

        # Update the cycle's profile tag
        cycle["profile_name"] = corrected_profile
        cycle["feedback_corrected"] = True

        # Optionally update profile's avg_duration if user provided correction
        if corrected_duration:
            profile = self.profile_store.get_profiles().get(corrected_profile)
            if profile:
                # Calculate weighted average: 80% old, 20% new (conservative learning)
                old_avg = profile.get("avg_duration", corrected_duration)
                profile["avg_duration"] = (
                    old_avg * 0.8 + corrected_duration * 0.2
                )
                _LOGGER.debug(
                    f"Updated profile '{corrected_profile}' avg_duration: "
                    f"{old_avg:.0f}s -> {profile['avg_duration']:.0f}s"
                )

        # Schedule async save
        # (This will be called from manager which has access to hass)

    def _auto_label_cycle(self, cycle_id: str, profile_name: str) -> None:
        """Auto-label a cycle with a profile name."""
        cycles = self.profile_store.get_past_cycles()
        cycle = next((c for c in cycles if c["id"] == cycle_id), None)
        
        if not cycle:
            _LOGGER.warning(f"Cycle {cycle_id} not found for auto-labeling")
            return
        
        cycle["profile_name"] = profile_name
        cycle["auto_labeled"] = True
        _LOGGER.debug(f"Auto-labeled cycle {cycle_id} with profile '{profile_name}'")

    def auto_label_high_confidence(
        self,
        cycle_id: str,
        profile_name: str,
        confidence: float,
        confidence_threshold: float = 0.95,
    ) -> bool:
        """
        Auto-label a cycle if confidence is very high.
        
        Args:
            cycle_id: The cycle ID
            profile_name: The detected profile name
            confidence: Confidence score (0-1)
            confidence_threshold: Threshold for auto-labeling (default 0.95)
            
        Returns:
            True if auto-labeled, False otherwise
        """
        if confidence < confidence_threshold:
            return False
        
        self._auto_label_cycle(cycle_id, profile_name)
        _LOGGER.info(
            f"Auto-labeled cycle {cycle_id} with very high confidence: "
            f"profile='{profile_name}' (confidence={confidence:.3f})"
        )
        return True

    def get_pending_feedback(self) -> dict[str, dict[str, Any]]:
        """Return pending feedback requests (for UI/service discovery)."""
        return dict(self._pending_feedback)

    def get_feedback_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent feedback history."""
        items = list(self._feedback_history.values())
        # Sort by submitted_at descending
        items.sort(
            key=lambda x: x.get("submitted_at", ""), reverse=True
        )
        return items[:limit]

    def get_learning_stats(self) -> dict[str, Any]:
        """Return learning statistics."""
        total_feedback = len(self._feedback_history)
        confirmed = sum(
            1 for f in self._feedback_history.values()
            if f.get("user_confirmed", False)
        )
        corrections = total_feedback - confirmed

        return {
            "total_feedback_submitted": total_feedback,
            "user_confirmed_cycles": confirmed,
            "user_corrected_cycles": corrections,
            "pending_feedback_requests": len(self._pending_feedback),
        }
