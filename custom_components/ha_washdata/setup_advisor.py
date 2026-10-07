"""Pure phase computation for the adoption guidance system.

No HA imports. No side effects. Testable in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class SetupPhaseResult:
    phase: str  # phase0 | phase1a | phase1b | phase1c | phase2 | phase3 | phase4
    message_key: str
    message_params: dict = field(default_factory=dict)
    cta_label_key: str = "setup.cta.start_recording"
    cta_action: str = "open_recorder"
    secondary_label_key: str | None = None
    secondary_action: str | None = None
    skippable: bool = False
    dismissible: bool = False
    step_key: str | None = None  # key used in skipped_steps dict


def compute_setup_phase(
    device_type: str,
    profile_names: list[str],
    past_cycles: list[dict],
    ref_profile_names: set[str],
    coverage_gap: dict | None,
    suggestions: list[dict],
    skipped_steps: dict[str, str | None],
    now: datetime,
    backfill_cycles: list[dict] | None = None,
) -> SetupPhaseResult:
    """Compute the current adoption phase for a device.

    Every cycle argument is the **evidence** view (``iter_evidence_cycles``), not the
    stored one: each check here asks whether a profile can be matched, and a profile
    whose only cycles the user excluded cannot. That keeps phase0 equal to
    ``ProfileStore.has_real_profiles`` being False, which is when the manager skips
    matching and leaves this card to explain why (register item 129d).

    Args:
        device_type: HA device type string (washing_machine, dishwasher, ...).
        profile_names: All profile names stored for this device.
        past_cycles: Live cycles that count as evidence (each may have profile_name
            and meta.source).
        ref_profile_names: Profile names backed by evidence reference cycles
            (store-adopted).
        coverage_gap: Result of profile_store.suggest_coverage_gaps(), or None.
        suggestions: Actionable suggestions from SuggestionEngine (empty list = none).
        skipped_steps: Dict of step_key -> "never" | ISO timestamp | None.
        now: Current aware datetime for snooze comparisons.
        backfill_cycles: Evidence cycles recovered from this machine's raw power
            history (#344). They are its own history, so a profile they back is a
            self-built profile, never a community one; they are never recordings.
    """
    backfill = list(backfill_cycles or [])
    own_cycles = [*past_cycles, *backfill]
    real = _real_profile_names(profile_names, own_cycles)
    has_real = bool(real)
    # Only the recorder writes meta.source == "recorder", and it writes past_cycles.
    has_recorded = _has_recorded_cycles(past_cycles, real)
    has_store = bool(ref_profile_names)
    has_self_cycles = bool(real)  # any own cycle assigned to a real profile

    # ── Phase 0 ──────────────────────────────────────────────────────────────
    if not has_real and not has_store and any(not c.get("profile_name") for c in backfill):
        # Imported history is waiting to be named: labelling it is the next step,
        # recording is the alternative. "Start recording" first told a user who had
        # just imported weeks of cycles that the device had nothing (item 129d).
        return SetupPhaseResult(
            phase="phase0",
            message_key="setup.phase0.generic",
            cta_label_key="setup.cta.label_detected_cycle",
            cta_action="open_cycles_unlabeled",
            secondary_label_key="setup.cta.start_recording",
            secondary_action="open_recorder",
            skippable=False,
            dismissible=False,
        )
    if not has_real and not has_store:
        msg_key = {
            "washing_machine": "setup.phase0.washer",
            "dryer": "setup.phase0.washer",
            "washer_dryer": "setup.phase0.washer",
            "dishwasher": "setup.phase0.dishwasher",
        }.get(device_type, "setup.phase0.generic")
        return SetupPhaseResult(
            phase="phase0",
            message_key=msg_key,
            cta_label_key="setup.cta.start_recording",
            cta_action="open_recorder",
            secondary_label_key="setup.cta.label_detected_cycle",
            secondary_action="open_cycles_unlabeled",
            skippable=False,
            dismissible=False,
        )

    # ── Phase 1c — store download, no self cycles yet ─────────────────────────
    if has_store and not has_self_cycles:
        return SetupPhaseResult(
            phase="phase1c",
            message_key="setup.phase1c.verify",
            message_params={"count": len(ref_profile_names)},
            cta_label_key="setup.cta.view_profiles",
            cta_action="open_profiles",
            skippable=False,
            dismissible=False,
        )

    # ── Phase 2 — coverage gaps / unmatched nudges ───────────────────────────
    if _phase2_active(coverage_gap, skipped_steps, now):
        cg = coverage_gap or {}
        clusters = cg.get("profile_suggestions") or []
        if clusters:
            first = clusters[0]
            return SetupPhaseResult(
                phase="phase2",
                message_key="setup.phase2.cluster",
                message_params={"count": cg.get("unmatched_count", 0),
                                "cycle_ids": first.get("cycle_ids", []),
                                "name": first.get("suggested_name", "")},
                cta_label_key="setup.cta.create_from_cluster",
                cta_action="create_profile_from_cluster",
                skippable=True,
                dismissible=False,
                step_key="setup_skip_phase2",
            )
        last_id = cg.get("last_unmatched_cycle_id")
        return SetupPhaseResult(
            phase="phase2",
            message_key="setup.phase2.unmatched",
            message_params={"cycle_id": last_id or ""},
            cta_label_key="setup.cta.create_profile",
            cta_action=f"open_cycle:{last_id}" if last_id else "open_cycles_unlabeled",
            skippable=True,
            dismissible=False,
            step_key="setup_skip_phase2",
        )

    # ── Phase 3 — tuning items ────────────────────────────────────────────────
    item = _phase3_pending_item(suggestions, skipped_steps, now)
    if item:
        return item

    # ── Phase 1a / 1b — early guidance (device has not yet seen coverage stats) ─
    # Only shown while the gap analyser has not yet flagged a coverage gap:
    # suggest_coverage_gaps() returns {} (empty) when there are no cycles or not
    # enough unmatched cycles, and a dict with suggest_create=True once the device
    # has accumulated enough cycles for the gap to be actionable.  We treat any
    # non-empty dict where suggest_create is True as "past the early stage".
    # None is also accepted (function signature allows it) and treated as no gap.
    # Also skipped once the user has permanently dismissed ("never") or snoozed the
    # Phase 1 guidance via setup_skip_phase1.
    # Auto-graduated when the device is clearly established: ≥2 real profiles, or
    # ≥5 cycles assigned to real profiles.  At that point the "record your first
    # cycle" nudge is stale and misleading regardless of whether the user ever
    # clicked Skip.
    _established = len(real) >= 2 or _real_cycle_count(own_cycles, real) >= 5
    _coverage_gap_actionable = bool(coverage_gap and coverage_gap.get("suggest_create"))
    _phase1_suppressed = _is_step_suppressed("setup_skip_phase1", skipped_steps, now)
    if has_real and not _established and not _coverage_gap_actionable and not _phase1_suppressed:
        if has_recorded:
            first_recorded_profile = _first_recorded_profile_name(past_cycles, real)
            return SetupPhaseResult(
                phase="phase1b",
                message_key="setup.phase1b.recorded",
                message_params={"profile_name": first_recorded_profile or ""},
                cta_label_key="setup.cta.start_recording",
                cta_action="open_recorder",
                secondary_label_key="setup.cta.browse_cycles",
                secondary_action="open_cycles",
                skippable=True,
                dismissible=False,
                step_key="setup_skip_phase1",
            )
        return SetupPhaseResult(
            phase="phase1a",
            message_key="setup.phase1a.labelled",
            cta_label_key="setup.cta.start_recording",
            cta_action="open_recorder",
            secondary_label_key="setup.cta.browse_cycles",
            secondary_action="open_cycles",
            skippable=True,
            dismissible=False,
            step_key="setup_skip_phase1",
        )

    # ── Phase 4 — healthy ─────────────────────────────────────────────────────
    return SetupPhaseResult(
        phase="phase4",
        message_key="setup.phase4.healthy",
        message_params={"profile_count": len(real)},
        skippable=False,
        dismissible=True,
    )


# ── Private helpers ───────────────────────────────────────────────────────────

def _real_profile_names(profile_names: list[str], past_cycles: list[dict]) -> set[str]:
    named = {c.get("profile_name") for c in past_cycles if c.get("profile_name")}
    return {n for n in profile_names if n in named}


def _real_cycle_count(past_cycles: list[dict], real: set[str]) -> int:
    return sum(1 for c in past_cycles if c.get("profile_name") in real)


def _has_recorded_cycles(past_cycles: list[dict], real: set[str]) -> bool:
    return any(
        (c.get("meta") or {}).get("source") == "recorder"
        for c in past_cycles
        if c.get("profile_name") in real
    )


def _first_recorded_profile_name(past_cycles: list[dict], real: set[str]) -> str | None:
    for c in past_cycles:
        if (c.get("profile_name") in real
                and (c.get("meta") or {}).get("source") == "recorder"):
            return c["profile_name"]
    return None


def _is_step_suppressed(step_key: str, skipped_steps: dict, now: datetime) -> bool:
    val = skipped_steps.get(step_key)
    if not val:
        return False
    if val == "never":
        return True
    try:
        until = datetime.fromisoformat(val)
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        return now < until
    except (ValueError, TypeError, OverflowError):
        return False


def _phase2_active(coverage_gap: dict | None, skipped_steps: dict, now: datetime) -> bool:
    if not coverage_gap or not coverage_gap.get("suggest_create"):
        return False
    return not _is_step_suppressed("setup_skip_phase2", skipped_steps, now)


def _phase3_pending_item(
    suggestions: list[dict],
    skipped_steps: dict,
    now: datetime,
) -> SetupPhaseResult | None:
    if suggestions and not _is_step_suppressed("setup_skip_phase3_suggestions", skipped_steps, now):
        return SetupPhaseResult(
            phase="phase3",
            message_key="setup.phase3.suggestions",
            cta_label_key="setup.cta.review_suggestions",
            cta_action="open_suggestions",
            skippable=True,
            dismissible=True,
            step_key="setup_skip_phase3_suggestions",
        )
    return None
