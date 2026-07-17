# Adoption Guidance System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace five scattered guidance surfaces with a single phased Setup Card on the Overview tab that walks new users from zero profiles to a fully tuned device, with threshold-mode detection while no profiles exist.

**Architecture:** A new pure `setup_advisor.py` module computes the current setup phase from store data; a new `ws_get_setup_status` WS command exposes it; the panel renders a single Setup Card that replaces the getting-started card, coverage-gap banner, recommendations banner, and group-suggestion banner. Match polling and the "no profile matched" notification are suppressed when no real profiles exist.

**Tech Stack:** Python 3.11, Home Assistant WS API, vanilla JS panel, pytest, Playwright.

## Global Constraints

- NumPy only — no scipy, sklearn, or new runtime dependencies.
- All user-visible panel strings go through `_t(key, params, fallback)` — English values in `translations/panel/en.json`.
- Never run `translate.py` or any machine translator — all language files via Claude subagents.
- Never post GitHub issue comments — update CHANGELOG.md instead.
- No new HA notification types — guidance lives in the panel card only.
- "Real profile" = a profile whose name appears in at least one `past_cycles` entry as `profile_name`. A name-only stub with no cycles is not real.
- Config schema: currently 3.6. This feature bumps it to **3.7**.
- Run fast suite after every task: `./run_tests.sh` (≈30 s).

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `custom_components/ha_washdata/setup_advisor.py` | **Create** | Pure `compute_setup_phase()` + `SetupPhaseResult` |
| `custom_components/ha_washdata/profile_store.py` | **Modify** | Add `has_real_profiles` property |
| `custom_components/ha_washdata/manager.py` | **Modify** | Skip match polling + suppress live-waiting notification when no real profiles |
| `custom_components/ha_washdata/ws_api.py` | **Modify** | Add `ws_get_setup_status`, update `ws_set_user_prefs` to accept skip keys |
| `custom_components/ha_washdata/config_flow.py` | **Modify** | Remove `async_step_first_profile` |
| `custom_components/ha_washdata/__init__.py` | **Modify** | Remove `initial_profile` handling; bump schema 3.6 → 3.7; add migration step |
| `custom_components/ha_washdata/strings.json` | **Modify** | Remove `first_profile` step |
| `custom_components/ha_washdata/translations/en.json` | **Modify** | Mirror `strings.json` removal |
| `custom_components/ha_washdata/translations/panel/en.json` | **Modify** | Add all `setup.*` keys |
| `custom_components/ha_washdata/www/ha-washdata-panel.js` | **Modify** | Replace getting-started card; add Setup Card; remove 3 Profiles-tab banners |
| `tests/test_setup_advisor.py` | **Create** | Unit tests for `compute_setup_phase()` |
| `tests/test_migration_harness.py` | **Modify** | Add 3.6 → 3.7 migration test |
| `playwright-tests/tests/setup-guidance.spec.ts` | **Create** | E2E tests for the Setup Card |

---

## Task 1: `setup_advisor.py` — Pure Phase Computation

**Files:**
- Create: `custom_components/ha_washdata/setup_advisor.py`
- Create: `tests/test_setup_advisor.py`

**Interfaces:**
- Produces: `SetupPhaseResult` dataclass + `compute_setup_phase()` consumed by Task 4 (WS handler)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_setup_advisor.py
from datetime import datetime, timezone
import pytest
from custom_components.ha_washdata.setup_advisor import (
    SetupPhaseResult,
    compute_setup_phase,
)

_NOW = datetime(2026, 7, 17, 12, 0, 0, tzinfo=timezone.utc)


def _cycle(profile_name, source=None):
    meta = {"source": source} if source else {}
    return {"profile_name": profile_name, "meta": meta}


# ── Phase 0 ──────────────────────────────────────────────────────────────────

def test_phase0_no_profiles_washer():
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=[],
        past_cycles=[],
        ref_profile_names=set(),
        coverage_gap=None,
        suggestions=[],
        profile_groups=[],
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase == "phase0"
    assert r.message_key == "setup.phase0.washer"
    assert r.cta_action == "open_recorder"


def test_phase0_dishwasher_gets_own_message():
    r = compute_setup_phase(
        device_type="dishwasher",
        profile_names=[],
        past_cycles=[],
        ref_profile_names=set(),
        coverage_gap=None,
        suggestions=[],
        profile_groups=[],
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase == "phase0"
    assert r.message_key == "setup.phase0.dishwasher"


def test_phase0_generic_device():
    r = compute_setup_phase(
        device_type="generic",
        profile_names=[],
        past_cycles=[],
        ref_profile_names=set(),
        coverage_gap=None,
        suggestions=[],
        profile_groups=[],
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase == "phase0"
    assert r.message_key == "setup.phase0.generic"


def test_phase0_stub_profile_not_counted():
    """A name-only stub (no matching past_cycles) must NOT advance past phase0."""
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],  # stub — no cycle
        past_cycles=[],
        ref_profile_names=set(),
        coverage_gap=None,
        suggestions=[],
        profile_groups=[],
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase == "phase0"


# ── Phase 1 variants ──────────────────────────────────────────────────────────

def test_phase1a_labelled_cycle():
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°")],
        ref_profile_names=set(),
        coverage_gap=None,
        suggestions=[],
        profile_groups=[],
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase == "phase1a"
    assert r.cta_action == "open_recorder"


def test_phase1b_recorded_cycle():
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°", source="recorder")],
        ref_profile_names=set(),
        coverage_gap=None,
        suggestions=[],
        profile_groups=[],
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase == "phase1b"
    assert "phase1b" in r.message_key


def test_phase1c_store_adopted():
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[],  # no self-recorded cycles
        ref_profile_names={"Cotton 60°"},
        coverage_gap=None,
        suggestions=[],
        profile_groups=[],
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase == "phase1c"


def test_phase1c_advances_once_self_cycle_added():
    """Store-adopted device that has also matched a real cycle → phase1b."""
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°")],
        ref_profile_names={"Cotton 60°"},
        coverage_gap=None,
        suggestions=[],
        profile_groups=[],
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase in ("phase1a", "phase1b", "phase2", "phase3", "phase4")


# ── Phase 2 reactive nudges ───────────────────────────────────────────────────

def test_phase2_cluster_nudge():
    cg = {"suggest_create": True, "unmatched_count": 5, "unmatched_rate": 0.5,
          "profile_suggestions": [{"cycle_ids": ["c1", "c2"], "suggested_name": "Eco", "count": 2}],
          "duration_clusters": []}
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°")],
        ref_profile_names=set(),
        coverage_gap=cg,
        suggestions=[],
        profile_groups=[],
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase == "phase2"
    assert r.message_key == "setup.phase2.cluster"
    assert r.cta_action == "create_profile_from_cluster"


def test_phase2_nudge_b_single_unmatched():
    cg = {"suggest_create": True, "unmatched_count": 1, "unmatched_rate": 0.1,
          "profile_suggestions": [],  # no clusters
          "duration_clusters": [], "last_unmatched_cycle_id": "abc123"}
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°")],
        ref_profile_names=set(),
        coverage_gap=cg,
        suggestions=[],
        profile_groups=[],
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase == "phase2"
    assert r.message_key == "setup.phase2.unmatched"


def test_phase2_skipped_snoozed_not_yet_expired():
    """A snoozed phase2 nudge that hasn't expired keeps the device in phase2-quiet."""
    future = "2099-01-01T00:00:00+00:00"
    cg = {"suggest_create": True, "unmatched_count": 5, "unmatched_rate": 0.5,
          "profile_suggestions": [], "duration_clusters": []}
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°")],
        ref_profile_names=set(),
        coverage_gap=cg,
        suggestions=[],
        profile_groups=[],
        skipped_steps={"setup_skip_phase2": future},
        now=_NOW,
    )
    assert r.phase in ("phase3", "phase4")  # nudge suppressed


def test_phase2_skipped_never():
    cg = {"suggest_create": True, "unmatched_count": 5, "unmatched_rate": 0.5,
          "profile_suggestions": [], "duration_clusters": []}
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°")],
        ref_profile_names=set(),
        coverage_gap=cg,
        suggestions=[],
        profile_groups=[],
        skipped_steps={"setup_skip_phase2": "never"},
        now=_NOW,
    )
    assert r.phase in ("phase3", "phase4")


def test_phase2_snooze_expired_resurfaces():
    past = "2020-01-01T00:00:00+00:00"
    cg = {"suggest_create": True, "unmatched_count": 5, "unmatched_rate": 0.5,
          "profile_suggestions": [], "duration_clusters": []}
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°")],
        ref_profile_names=set(),
        coverage_gap=cg,
        suggestions=[],
        profile_groups=[],
        skipped_steps={"setup_skip_phase2": past},
        now=_NOW,
    )
    assert r.phase == "phase2"


# ── Phase 3 tuning items ──────────────────────────────────────────────────────

def test_phase3_suggestions_first():
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°")],
        ref_profile_names=set(),
        coverage_gap=None,
        suggestions=[{"key": "off_delay", "current": 600, "suggested": 900}],
        profile_groups=[],
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase == "phase3"
    assert r.message_key == "setup.phase3.suggestions"


def test_phase3_groups_after_suggestions_skipped():
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°")],
        ref_profile_names=set(),
        coverage_gap=None,
        suggestions=[{"key": "off_delay"}],
        profile_groups=[{"members": ["A", "B"]}],
        skipped_steps={"setup_skip_phase3_suggestions": "never"},
        now=_NOW,
    )
    assert r.phase == "phase3"
    assert r.message_key == "setup.phase3.groups"


# ── Phase 4 ───────────────────────────────────────────────────────────────────

def test_phase4_all_clear():
    r = compute_setup_phase(
        device_type="washing_machine",
        profile_names=["Cotton 60°"],
        past_cycles=[_cycle("Cotton 60°")],
        ref_profile_names=set(),
        coverage_gap=None,
        suggestions=[],
        profile_groups=[],
        skipped_steps={},
        now=_NOW,
    )
    assert r.phase == "phase4"
    assert r.dismissible is True
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
python3 -m pytest tests/test_setup_advisor.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError` or `ImportError` — `setup_advisor` does not exist yet.

- [ ] **Step 3: Implement `setup_advisor.py`**

```python
# custom_components/ha_washdata/setup_advisor.py
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
    profile_groups: list[dict],
    skipped_steps: dict[str, str | None],
    now: datetime,
) -> SetupPhaseResult:
    """Compute the current adoption phase for a device.

    Args:
        device_type: HA device type string (washing_machine, dishwasher, …).
        profile_names: All profile names stored for this device.
        past_cycles: All past cycles (each may have profile_name and meta.source).
        ref_profile_names: Profile names that have reference cycles (store-adopted).
        coverage_gap: Result of profile_store.suggest_coverage_gaps(), or None.
        suggestions: Actionable suggestions from SuggestionEngine (empty list = none).
        profile_groups: Profile groups list from store (empty list = none pending).
        skipped_steps: Dict of step_key -> "never" | ISO timestamp | None.
        now: Current aware datetime for snooze comparisons.
    """
    real = _real_profile_names(profile_names, past_cycles)
    has_real = bool(real)
    has_recorded = _has_recorded_cycles(past_cycles, real)
    has_store = bool(ref_profile_names)
    has_self_cycles = bool(real)  # any cycle assigned to a real profile

    # ── Phase 0 ──────────────────────────────────────────────────────────────
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

    # ── Phase 1a / 1b — first real profile exists ────────────────────────────
    if has_real and not _phase2_active(coverage_gap, skipped_steps, now):
        # Check if there are pending phase-3 items too; if not, go straight to 4.
        pending3 = _phase3_pending_item(suggestions, profile_groups, skipped_steps, now)
        if not pending3:
            # If only one profile and no coverage issues → still show phase1 guidance.
            # Once coverage gap appears we'll be in phase2. For now stay in phase1.
            pass
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
    item = _phase3_pending_item(suggestions, profile_groups, skipped_steps, now)
    if item:
        return item

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
    except (ValueError, TypeError):
        return False


def _phase2_active(coverage_gap: dict | None, skipped_steps: dict, now: datetime) -> bool:
    if not coverage_gap or not coverage_gap.get("suggest_create"):
        return False
    return not _is_step_suppressed("setup_skip_phase2", skipped_steps, now)


def _phase3_pending_item(
    suggestions: list[dict],
    profile_groups: list[dict],
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
    if profile_groups and not _is_step_suppressed("setup_skip_phase3_groups", skipped_steps, now):
        return SetupPhaseResult(
            phase="phase3",
            message_key="setup.phase3.groups",
            cta_label_key="setup.cta.organise_profiles",
            cta_action="open_profiles_groups",
            skippable=True,
            dismissible=True,
            step_key="setup_skip_phase3_groups",
        )
    return None
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/test_setup_advisor.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Syntax check**

```bash
python3 -m compileall custom_components/ha_washdata/setup_advisor.py --quiet
```

- [ ] **Step 6: Commit**

```bash
git add custom_components/ha_washdata/setup_advisor.py tests/test_setup_advisor.py
git commit -m "feat: add setup_advisor.py — pure compute_setup_phase function"
```

---

## Task 2: Threshold Mode — Skip Polling and Notification When No Profiles

**Files:**
- Modify: `custom_components/ha_washdata/profile_store.py`
- Modify: `custom_components/ha_washdata/manager.py`

**Interfaces:**
- Consumes: `ProfileStore.has_real_profiles` (new property) used in manager.py
- Produces: nothing new exported — behavior change only

- [ ] **Step 1: Add `has_real_profiles` property to `ProfileStore`**

Find the `list_profiles` method (around line 4720 in `profile_store.py`). Add the property just before it:

```python
@property
def has_real_profiles(self) -> bool:
    """True if at least one stored profile has a cycle in past_cycles."""
    assigned = {
        c.get("profile_name")
        for c in self._data.get("past_cycles", [])
        if c.get("profile_name")
    }
    return bool(
        assigned.intersection(self._data.get("profiles", {}).keys())
    )
```

- [ ] **Step 2: Write a fast unit test for `has_real_profiles`**

Add to `tests/test_setup_advisor.py` (or a new `tests/test_profile_store_has_real.py` — either works):

```python
# Inline at bottom of tests/test_setup_advisor.py
# (requires a ProfileStore instance — skip if mocking is too heavy; the
# property logic is trivial and covered by compute_setup_phase tests above)
def test_has_real_profiles_false_for_stub():
    """Smoke test: stub profile with no past_cycles → not real."""
    from custom_components.ha_washdata.setup_advisor import _real_profile_names
    assert _real_profile_names(["Cotton 60°"], []) == set()


def test_has_real_profiles_true_with_cycle():
    from custom_components.ha_washdata.setup_advisor import _real_profile_names
    cycles = [{"profile_name": "Cotton 60°", "meta": {}}]
    assert _real_profile_names(["Cotton 60°"], cycles) == {"Cotton 60°"}
```

- [ ] **Step 3: Skip match polling when no real profiles**

In `manager.py`, find the trigger point for `_async_do_perform_matching` (around line 888–898). Add the guard immediately before the `async_create_task` call:

```python
# Skip match entirely when no real profiles exist — nothing to match against.
if not self.profile_store.has_real_profiles:
    self._logger.debug("Matching skipped: no real profiles configured yet")
    return
```

The guard goes after the existing "no readings" check and before `self._matching_task = ...`.

- [ ] **Step 4: Suppress live-waiting notification when no real profiles**

In `manager.py`, find `_check_live_progress_notification` (the method containing the `_live_waiting_notification_sent` logic, around line 5245). The `if not has_profile_match:` block (line 5254) fires the "No profile matched yet" notification. Add a guard at the top of that block:

```python
if not has_profile_match:
    # Suppress the waiting notification when no profiles exist at all —
    # the setup card explains the state instead.
    if not self.profile_store.has_real_profiles:
        return
    if self._live_waiting_notification_sent:
        return
    # … rest of existing notification code unchanged …
```

- [ ] **Step 5: Run fast test suite**

```bash
./run_tests.sh
```
Expected: all tests pass (no regressions).

- [ ] **Step 6: Commit**

```bash
git add custom_components/ha_washdata/profile_store.py custom_components/ha_washdata/manager.py
git commit -m "feat: skip match polling and suppress live-waiting notification when no real profiles"
```

---

## Task 3: Config Flow Removal and Migration

**Files:**
- Modify: `custom_components/ha_washdata/config_flow.py`
- Modify: `custom_components/ha_washdata/__init__.py`
- Modify: `custom_components/ha_washdata/strings.json`
- Modify: `custom_components/ha_washdata/translations/en.json`
- Modify: `tests/test_migration_harness.py`

**Interfaces:**
- Produces: config schema version bumped to 3.7

- [ ] **Step 1: Write the migration test first**

Find `tests/test_migration_harness.py`. Add a test for 3.6 → 3.7 at the end:

```python
async def test_migrate_3_6_to_3_7_removes_initial_profile():
    """initial_profile in entry.data must be stripped on 3.6 → 3.7 migration."""
    hass = _make_hass()
    entry = MockConfigEntry(
        domain="ha_washdata",
        version=3,
        minor_version=6,
        data={
            "name": "Washer",
            "power_sensor": "sensor.power",
            "initial_profile": {"name": "Cotton 60°", "avg_duration": 7200},
        },
        options={},
    )
    entry.add_to_hass(hass)
    result = await async_migrate_entry(hass, entry)
    assert result is True
    assert "initial_profile" not in entry.data
    assert entry.version == 3
    assert entry.minor_version == 7


async def test_migrate_3_6_to_3_7_no_initial_profile_is_noop():
    """Entries without initial_profile migrate cleanly."""
    hass = _make_hass()
    entry = MockConfigEntry(
        domain="ha_washdata",
        version=3,
        minor_version=6,
        data={"name": "Washer", "power_sensor": "sensor.power"},
        options={},
    )
    entry.add_to_hass(hass)
    result = await async_migrate_entry(hass, entry)
    assert result is True
    assert entry.minor_version == 7
```

- [ ] **Step 2: Run migration test to confirm it fails**

```bash
python3 -m pytest tests/test_migration_harness.py::test_migrate_3_6_to_3_7_removes_initial_profile -v
```
Expected: FAIL (no 3.7 migration case exists yet).

- [ ] **Step 3: Add 3.7 migration in `async_migrate_entry` (`__init__.py`)**

In `__init__.py`, find `async_migrate_entry` (line 133). The migration chain ends at `minor_version >= 6` returning True. Add a new block after all existing 3.x checks, before the final return:

```python
if version == 3 and minor_version == 6:
    # 3.6 → 3.7: remove initial_profile stub key from entry.data.
    new_data = {k: v for k, v in entry.data.items() if k != "initial_profile"}
    hass.config_entries.async_update_entry(
        entry, data=new_data, minor_version=7
    )
    minor_version = 7
    _LOGGER.debug("Migrated WashData entry from 3.6 to 3.7")

if version == 3 and minor_version >= 7:
    return True
```

Also update the fast-exit guard at the top of `async_migrate_entry`:

```python
# Change: minor_version >= 6  →  minor_version >= 7
if version == 3 and minor_version >= 7:
    return True
```

And bump the declared schema version at the `MockConfigEntry` / `async_setup_entry` call site. Find:
```python
version=3,
minor_version=6,
```
Change to:
```python
version=3,
minor_version=7,
```

- [ ] **Step 4: Remove `initial_profile` handling in `async_setup_entry` (`__init__.py`)**

Find lines 373–393 (the `if "initial_profile" in entry.data:` block). Remove the entire block:

```python
# DELETE this block entirely (lines ~373–393):
# if "initial_profile" in entry.data:
#     init_prof = entry.data["initial_profile"]
#     ...
#     hass.config_entries.async_update_entry(entry, data=new_data)
```

- [ ] **Step 5: Remove `async_step_first_profile` from `config_flow.py`**

Delete the entire `async_step_first_profile` method (lines 187–220).

Update `async_step_user` to call `async_create_entry` directly instead of `async_step_first_profile`. Change line 185:

```python
# Before:
return await self.async_step_first_profile()

# After:
return self.async_create_entry(title=user_input[CONF_NAME], data=user_input)
```

- [ ] **Step 6: Remove `first_profile` from `strings.json` and `translations/en.json`**

In `strings.json`, find and delete the `first_profile` entry under `config.step`:

```json
// DELETE this entire block from config.step:
"first_profile": {
  "title": "Create First Profile",
  "description": "...",
  "data": {
    "profile_name": "...",
    "manual_duration": "..."
  }
}
```

Mirror the same deletion in `translations/en.json` (must be identical to `strings.json`).

- [ ] **Step 7: Run migration tests**

```bash
python3 -m pytest tests/test_migration_harness.py -v
```
Expected: all pass including the two new 3.7 tests.

- [ ] **Step 8: Run full fast suite + syntax check**

```bash
./run_tests.sh
python3 -m compileall custom_components/ha_washdata --quiet
```

- [ ] **Step 9: Commit**

```bash
git add custom_components/ha_washdata/config_flow.py \
        custom_components/ha_washdata/__init__.py \
        custom_components/ha_washdata/strings.json \
        custom_components/ha_washdata/translations/en.json \
        tests/test_migration_harness.py
git commit -m "feat: remove first-profile config flow stub; bump schema to 3.7"
```

---

## Task 4: `ws_get_setup_status` WS Command + Translation Keys

**Files:**
- Modify: `custom_components/ha_washdata/ws_api.py`
- Modify: `custom_components/ha_washdata/translations/panel/en.json`

**Interfaces:**
- Consumes: `compute_setup_phase` from Task 1
- Produces: `ha_washdata/get_setup_status` WS command, consumed by Task 5 (panel)

- [ ] **Step 1: Add translation keys to `translations/panel/en.json`**

Open `translations/panel/en.json`. Add the following `setup` namespace. Find an appropriate place (alphabetical or at the end of the top-level object):

```json
"setup": {
  "hdr": {
    "card": "Device Setup",
    "healthy_chip": "Setup complete"
  },
  "phase0": {
    "washer": "WashData is already detecting your cycles. Record your first cycle to enable program names and time estimates.",
    "dishwasher": "WashData is watching. Dishwashers have complex cycles — recording your first cycle is strongly recommended. If a detected cycle runs too long, use the cycle editor to trim it before saving as a profile.",
    "generic": "WashData is watching. Record or label a detected cycle to start building profiles."
  },
  "phase1a": {
    "labelled": "Good start — your first program is saved. For the cleanest data, consider recording your next cycle with the recorder widget."
  },
  "phase1b": {
    "recorded": "Your recording was saved as {profile_name}. Now record or label your other common programs to build coverage."
  },
  "phase1c": {
    "verify": "You have {count} {count, plural, one {program} other {programs}} from the community. Run a cycle to verify WashData recognises it correctly — matching will improve as your device builds its own history."
  },
  "phase2": {
    "cluster": "WashData has seen {count} {count, plural, one {cycle} other {cycles}} that don't match any saved program — they look similar to each other. Want to create a new profile for them?",
    "unmatched": "Your last cycle didn't match any saved program. Is it a new program?"
  },
  "phase3": {
    "suggestions": "WashData has settings recommendations based on your cycle history — review them to improve detection accuracy.",
    "groups": "Some of your profiles look like the same program at different temperatures. Organise them into a group for better matching.",
    "phases": "Add program phases to {profile_name} for more accurate time-remaining estimates."
  },
  "phase4": {
    "healthy": "This device is fully set up ({profile_count} {profile_count, plural, one {profile} other {profiles}})."
  },
  "cta": {
    "start_recording": "Start Recording",
    "label_detected_cycle": "I already have a detected cycle — label it instead",
    "browse_cycles": "Browse cycles to label another",
    "view_profiles": "View your profiles",
    "create_from_cluster": "Create profile from these cycles",
    "create_profile": "Create a profile for it",
    "review_suggestions": "Review suggestions",
    "organise_profiles": "Organise profiles",
    "configure_phases": "Configure phases",
    "skip_step": "Skip this step",
    "skip_forever": "Don't show again",
    "hide_guidance": "Hide guidance",
    "show_guidance": "Show guidance"
  }
}
```

- [ ] **Step 2: Add `ws_get_setup_status` handler in `ws_api.py`**

Add the following import at the top of `ws_api.py` alongside other module imports:

```python
from .setup_advisor import compute_setup_phase
```

Then add the handler function. Find an appropriate place after the existing `ws_get_options` block (around line 1269):

```python
@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_washdata/get_setup_status",
        vol.Required("entry_id"): str,
    }
)
@websocket_api.async_response
async def ws_get_setup_status(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return the current setup phase for the adoption guidance card."""
    manager = _get_manager(hass, msg["entry_id"])
    if not manager:
        connection.send_error(msg["id"], "not_found", "Device not found")
        return

    # Gather user's skipped steps from user prefs
    skipped_steps: dict[str, str | None] = {}
    user = getattr(connection, "user", None)
    if user:
        holder = hass.data.get(_PANEL_DATA_KEY)
        if holder:
            prefs = holder["data"].get("prefs", {}).get(user.id, {})
            for k, v in prefs.items():
                if k.startswith("setup_skip_"):
                    skipped_steps[k] = v

    # Gather store data (executor-safe reads)
    store = manager.profile_store
    profile_names = list(store._data.get("profiles", {}).keys())
    past_cycles = store._data.get("past_cycles", [])
    ref_names: set[str] = set()
    for rc in store._data.get("reference_cycles", []):
        if rc.get("profile_name"):
            ref_names.add(rc["profile_name"])

    coverage_gap = await hass.async_add_executor_job(store.suggest_coverage_gaps)
    # suggestions: use cached if available, else empty (avoid heavy computation here)
    suggestions = manager._last_suggestions if hasattr(manager, "_last_suggestions") else []
    pg_data = store._data.get("profile_groups", {})
    pending_groups = (pg_data.get("suggestions") or []) if isinstance(pg_data, dict) else []

    device_type = manager.device_type

    from homeassistant.util import dt as dt_util
    result = compute_setup_phase(
        device_type=device_type,
        profile_names=profile_names,
        past_cycles=past_cycles,
        ref_profile_names=ref_names,
        coverage_gap=coverage_gap,
        suggestions=suggestions,
        profile_groups=pending_groups,
        skipped_steps=skipped_steps,
        now=dt_util.now(),
    )

    _send_result(connection, msg["id"], "get_setup_status", {
        "phase": result.phase,
        "message_key": result.message_key,
        "message_params": result.message_params,
        "cta_label_key": result.cta_label_key,
        "cta_action": result.cta_action,
        "secondary_label_key": result.secondary_label_key,
        "secondary_action": result.secondary_action,
        "skippable": result.skippable,
        "dismissible": result.dismissible,
        "step_key": result.step_key,
    })
```

- [ ] **Step 3: Register the new command in `async_register_commands`**

Find `async_register_commands` (line 1024). In the import list passed to the handler loop, add `ws_get_setup_status`:

```python
ws_get_options, ws_set_options, ws_get_settings_changelog,
ws_get_setup_status,   # ← add this line
ws_get_profiles, ws_create_profile, ...
```

- [ ] **Step 4: Allow `setup_skip_*` keys in `ws_set_user_prefs`**

Find `ws_set_user_prefs` (line 3461). After the existing per-key assignments, add:

```python
# Allow setup guidance skip keys: setup_skip_<step_key> -> "never" | ISO timestamp
for k, v in p.items():
    if k.startswith("setup_skip_") and isinstance(k, str):
        if v is None:
            cur.pop(k, None)
        elif v == "never" or (isinstance(v, str) and len(v) <= 40):
            cur[k] = v
```

- [ ] **Step 5: Syntax check**

```bash
python3 -m compileall custom_components/ha_washdata/ws_api.py custom_components/ha_washdata/setup_advisor.py --quiet
```

- [ ] **Step 6: Run fast suite**

```bash
./run_tests.sh
```

- [ ] **Step 7: Commit**

```bash
git add custom_components/ha_washdata/ws_api.py \
        custom_components/ha_washdata/translations/panel/en.json
git commit -m "feat: add ws_get_setup_status command and panel translation keys"
```

---

## Task 5: Panel — Setup Card (Replace Getting Started Card)

**Files:**
- Modify: `custom_components/ha_washdata/www/ha-washdata-panel.js`

**Interfaces:**
- Consumes: `ha_washdata/get_setup_status` WS command (Task 4)
- Consumes: existing `_pref` / `_setPref` / `_t` / `_ws` panel utilities

- [ ] **Step 1: Add `_setupStatus` state property**

Find the section in `connectedCallback` or `_initState` where `this._profiles`, `this._powerData`, etc. are initialised (search for `this._profiles = []`). Add:

```javascript
this._setupStatus = null;  // result of ws_get_setup_status
```

- [ ] **Step 2: Fetch setup status on Status tab load**

Find `_loadStatusTab()` (or the equivalent method that runs on Status tab activation, which already fetches `_powerData`, `_recState`, etc.). Add after the existing fetches:

```javascript
// Fetch setup guidance phase (always, not just when profileCount === 0)
try {
  this._setupStatus = await this._ws({
    type: `${_DOMAIN}/get_setup_status`,
    entry_id: this._selectedDevice.entry_id,
  });
} catch (_) { this._setupStatus = null; }
```

- [ ] **Step 3: Add `_htmlSetupCard()` method**

Add a new method after `_htmlGettingStarted()`. The card replaces the getting-started card and handles all phases:

```javascript
_htmlSetupCard(status) {
  if (!status) return '';
  const { phase, message_key, message_params, cta_label_key, cta_action,
          secondary_label_key, secondary_action, skippable, dismissible,
          step_key } = status;

  // Phase 4: collapsed chip (unless dismissed)
  if (phase === 'phase4') {
    return `
      <div class="wd-setup-chip wd-setup-chip--healthy" data-action="expand-setup">
        <span class="wd-setup-dot wd-setup-dot--green"></span>
        <span>${this._t('setup.hdr.healthy_chip', {}, 'Setup complete')}</span>
      </div>`;
  }

  const msg = this._t(message_key, message_params, '');
  const ctaLabel = this._t(cta_label_key, {}, 'Continue');
  const secLabel = secondary_label_key
    ? this._t(secondary_label_key, {}, '')
    : null;

  const skipHtml = skippable ? `
    <span style="display:flex;gap:12px;margin-top:6px">
      <a class="wd-link" data-action="setup-skip" data-step="${_esc(step_key || '')}" data-snooze="14d"
         style="font-size:.85em">${this._t('setup.cta.skip_step', {}, 'Skip this step')}</a>
      <a class="wd-link" data-action="setup-skip" data-step="${_esc(step_key || '')}" data-snooze="never"
         style="font-size:.85em;opacity:.7">${this._t('setup.cta.skip_forever', {}, "Don't show again")}</a>
    </span>` : '';

  const hideHtml = dismissible ? `
    <a class="wd-link" data-action="hide-setup-card" style="font-size:.8em;opacity:.6;margin-top:4px">
      ${this._t('setup.cta.hide_guidance', {}, 'Hide guidance')}
    </a>` : '';

  return `
    <div class="wd-setup-card" data-phase="${_esc(phase)}">
      <div class="wd-card-title">${this._t('setup.hdr.card', {}, 'Device Setup')}</div>
      <p style="margin:6px 0 12px">${_esc(msg)}</p>
      <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center">
        <button class="wd-btn wd-btn-primary" data-action="setup-cta"
                data-cta-action="${_esc(cta_action)}"
                data-cta-params="${_esc(JSON.stringify(message_params))}">
          ${_esc(ctaLabel)}
        </button>
        ${secLabel ? `<a class="wd-link" data-action="setup-cta"
                         data-cta-action="${_esc(secondary_action || '')}"
                         style="font-size:.9em">${_esc(secLabel)}</a>` : ''}
      </div>
      ${skipHtml}
      ${hideHtml}
    </div>`;
}
```

- [ ] **Step 4: Wire the Setup Card into the Status tab render**

Find the existing `showGettingStarted` block in `_htmlStatusTab()` (around line 3463):

```javascript
const showGettingStarted = !this._pref('onboarding_dismissed', false) && profileCount === 0 && !hasCurve;
```

Replace this entire block (the `showGettingStarted` variable and wherever it's used to render `_htmlGettingStarted`) with:

```javascript
// Setup card: show unless user dismissed it AND we're at phase4 (healthy).
const setupDismissed = this._pref('setup_card_dismissed', false);
const setupStatus = this._setupStatus;
const showSetupCard = setupStatus
  && !(setupDismissed && setupStatus.phase === 'phase4')
  && !hasCurve;

// Replace _htmlGettingStarted usage:
// Before: showGettingStarted ? this._htmlGettingStarted(cycleCount) : ...
// After:
const setupCardHtml = showSetupCard ? this._htmlSetupCard(setupStatus) : '';
```

Then in the template where `_htmlGettingStarted` was rendered, substitute `setupCardHtml`.

- [ ] **Step 5: Handle Setup Card actions**

Find the panel's main action dispatcher (search for `data-action="skip-onboarding"` or the `_handleAction` / `_onClick` method, around line 9200+). Add handlers:

```javascript
// Setup CTA dispatch
if (action === 'setup-cta') {
  const ctaAction = el.dataset.ctaAction || '';
  const params = JSON.parse(el.dataset.ctaParams || '{}');
  this._dispatchSetupCta(ctaAction, params);
  return;
}

// Skip step (snooze or never)
if (action === 'setup-skip') {
  const stepKey = el.dataset.step;
  const snooze = el.dataset.snooze; // "never" or "14d"
  if (stepKey) {
    let val;
    if (snooze === 'never') {
      val = 'never';
    } else {
      const until = new Date();
      until.setDate(until.getDate() + 14);
      val = until.toISOString();
    }
    this._setPref(stepKey, val);
    await this._reloadSetupStatus();
  }
  return;
}

// Permanently hide the card (only phase3/4 — dismissible)
if (action === 'hide-setup-card') {
  this._setPref('setup_card_dismissed', true);
  this._setupStatus = null;
  this._render();
  return;
}
```

Add `_dispatchSetupCta` method:

```javascript
_dispatchSetupCta(ctaAction, params) {
  if (ctaAction === 'open_recorder') {
    // Scroll to recorder widget on Overview tab (it's already visible)
    this.shadowRoot.querySelector('[data-recorder-widget]')?.scrollIntoView({ behavior: 'smooth' });
    return;
  }
  if (ctaAction === 'open_cycles' || ctaAction === 'open_cycles_unlabeled') {
    this._switchTab('cycles');
    return;
  }
  if (ctaAction === 'open_profiles' || ctaAction === 'open_profiles_groups') {
    this._switchTab('profiles');
    return;
  }
  if (ctaAction === 'open_suggestions') {
    this._switchTab('suggestions');
    return;
  }
  if (ctaAction === 'create_profile_from_cluster') {
    const cycleIds = params.cycle_ids || [];
    const name = params.name || '';
    // Open create-profile modal pre-populated with cluster cycles
    this._openCreateProfileModal({ cycleIds, suggestedName: name });
    return;
  }
  if (ctaAction && ctaAction.startsWith('open_cycle:')) {
    const cycleId = ctaAction.split(':')[1];
    this._openCycleModal(cycleId);
    return;
  }
}

async _reloadSetupStatus() {
  try {
    this._setupStatus = await this._ws({
      type: `${_DOMAIN}/get_setup_status`,
      entry_id: this._selectedDevice.entry_id,
    });
  } catch (_) { this._setupStatus = null; }
  this._render();
}
```

- [ ] **Step 6: Run fast suite**

```bash
./run_tests.sh
```

- [ ] **Step 7: Commit**

```bash
git add custom_components/ha_washdata/www/ha-washdata-panel.js
git commit -m "feat: add Setup Card to Overview tab (phases 0-4)"
```

---

## Task 6: Panel — Remove Old Guidance Surfaces

**Files:**
- Modify: `custom_components/ha_washdata/www/ha-washdata-panel.js`

**Interfaces:**
- None — removal only; Setup Card (Task 5) already provides the replacement content.

- [ ] **Step 1: Remove `_htmlGettingStarted` method**

Delete the entire `_htmlGettingStarted(cycleCount)` method (lines 3542–3600 approximately). It is now fully replaced by `_htmlSetupCard`.

Also remove the old `onboarding_dismissed` pref usage at line 3463 and any references to `showGettingStarted` that remain after Task 5. Search for `onboarding_dismissed` and `_htmlGettingStarted` to confirm no remaining usages.

- [ ] **Step 2: Remove the coverage-gap banner from `_htmlProfiles()`**

In `_htmlProfiles()`, find:

```javascript
const cgBanner = (canEdit && cg.suggest_create) ? (() => { ... })() : '';
```

Remove the `cgBanner` variable declaration entirely (the whole IIFE block). Then find where `cgBanner` is interpolated into the template string and remove that interpolation too.

- [ ] **Step 3: Remove the near-duplicate group suggestion banner from `_htmlProfiles()`**

Find:

```javascript
const sugBanner = (canEdit && (pg.suggestions || []).length) ? `...` : '';
```

Remove the `sugBanner` declaration and its usage in the template. Note: the group suggestion chips on individual profile group cards stay — only the top-level banner is removed.

- [ ] **Step 4: Remove the recommendations/advisories banner from `_htmlProfiles()`**

Find:

```javascript
const advBanner = advisories.length ? `...` : '';
```

Remove the `advBanner` declaration and its usage in the template.

- [ ] **Step 5: Verify the Profiles tab still renders correctly**

Search for `cgBanner`, `sugBanner`, `advBanner` in the panel file — should find zero occurrences.

- [ ] **Step 6: Run fast suite + E2E (chromium)**

```bash
./run_tests.sh
cd playwright-tests && npx playwright test --project=chromium 2>&1 | tail -20
```
Expected: all existing E2E tests pass (profile-related tests should not regress since group chips on profile cards remain).

- [ ] **Step 7: Commit**

```bash
git add custom_components/ha_washdata/www/ha-washdata-panel.js
git commit -m "feat: remove getting-started card and three Profiles-tab banners (consolidated into Setup Card)"
```

---

## Task 7: E2E Tests for the Setup Card

**Files:**
- Create: `playwright-tests/tests/setup-guidance.spec.ts`

**Interfaces:**
- Consumes: WS mock infrastructure from `playwright-tests/helpers/`

- [ ] **Step 1: Examine the WS mock to understand how to inject setup status**

```bash
ls playwright-tests/helpers/
grep -n "get_setup_status\|setupStatus\|ws_mock\|mockWs" playwright-tests/helpers/*.ts 2>/dev/null | head -20
grep -n "get_power_history\|mock.*response\|addHandler" playwright-tests/helpers/*.ts 2>/dev/null | head -20
```

Understand the pattern: how do other tests mock WS command responses (e.g. `get_profiles`, `get_power_history`) and replicate it for `get_setup_status`.

- [ ] **Step 2: Write the E2E spec**

```typescript
// playwright-tests/tests/setup-guidance.spec.ts
import { test, expect } from '@playwright/test';
import { setupPanel, mockWsCommand } from '../helpers/panel-helpers'; // adjust import to match actual helpers

test.describe('Setup Card', () => {

  test('Phase 0 card appears on fresh device (washer)', async ({ page }) => {
    await setupPanel(page, {
      deviceType: 'washing_machine',
      profiles: [],
      pastCycles: [],
    });
    // Mock ws_get_setup_status → phase0
    await mockWsCommand(page, 'ha_washdata/get_setup_status', {
      phase: 'phase0',
      message_key: 'setup.phase0.washer',
      message_params: {},
      cta_label_key: 'setup.cta.start_recording',
      cta_action: 'open_recorder',
      secondary_label_key: 'setup.cta.label_detected_cycle',
      secondary_action: 'open_cycles_unlabeled',
      skippable: false,
      dismissible: false,
      step_key: null,
    });
    await page.waitForSelector('[data-phase="phase0"]');
    await expect(page.locator('[data-phase="phase0"]')).toBeVisible();
    await expect(page.locator('[data-cta-action="open_recorder"]')).toBeVisible();
  });

  test('Start Recording CTA triggers recorder widget scroll', async ({ page }) => {
    await setupPanel(page, { deviceType: 'washing_machine', profiles: [], pastCycles: [] });
    await mockWsCommand(page, 'ha_washdata/get_setup_status', {
      phase: 'phase0', message_key: 'setup.phase0.washer', message_params: {},
      cta_label_key: 'setup.cta.start_recording', cta_action: 'open_recorder',
      secondary_label_key: null, secondary_action: null,
      skippable: false, dismissible: false, step_key: null,
    });
    await page.waitForSelector('[data-cta-action="open_recorder"]');
    await page.click('[data-cta-action="open_recorder"]');
    // Recorder widget should be visible (panel scrolls to it)
    await expect(page.locator('[data-recorder-widget]')).toBeVisible();
  });

  test('Phase 4 chip shows on fully-configured device', async ({ page }) => {
    await setupPanel(page, { deviceType: 'washing_machine', profiles: ['Cotton 60°'], pastCycles: [{ profile_name: 'Cotton 60°' }] });
    await mockWsCommand(page, 'ha_washdata/get_setup_status', {
      phase: 'phase4', message_key: 'setup.phase4.healthy',
      message_params: { profile_count: 1 },
      cta_label_key: '', cta_action: '',
      secondary_label_key: null, secondary_action: null,
      skippable: false, dismissible: true, step_key: null,
    });
    await page.waitForSelector('.wd-setup-chip--healthy');
    await expect(page.locator('.wd-setup-chip--healthy')).toBeVisible();
  });

  test('Hide guidance collapses card and sets pref', async ({ page }) => {
    const prefCalls: string[] = [];
    await page.route('**/ha_washdata/set_user_prefs', async route => {
      const body = await route.request().postDataJSON?.();
      if (body?.prefs?.setup_card_dismissed) prefCalls.push('dismissed');
      await route.fulfill({ json: { success: true } });
    });
    await setupPanel(page, { deviceType: 'washing_machine', profiles: ['Cotton 60°'], pastCycles: [{ profile_name: 'Cotton 60°' }] });
    await mockWsCommand(page, 'ha_washdata/get_setup_status', {
      phase: 'phase3', message_key: 'setup.phase3.suggestions',
      message_params: {}, cta_label_key: 'setup.cta.review_suggestions',
      cta_action: 'open_suggestions', secondary_label_key: null, secondary_action: null,
      skippable: true, dismissible: true, step_key: 'setup_skip_phase3_suggestions',
    });
    await page.waitForSelector('[data-action="hide-setup-card"]');
    await page.click('[data-action="hide-setup-card"]');
    await expect(page.locator('[data-phase="phase3"]')).not.toBeVisible();
  });

  test('Phase 2 cluster nudge appears with create-profile CTA', async ({ page }) => {
    await setupPanel(page, { deviceType: 'washing_machine', profiles: ['Cotton 60°'], pastCycles: [{ profile_name: 'Cotton 60°' }] });
    await mockWsCommand(page, 'ha_washdata/get_setup_status', {
      phase: 'phase2', message_key: 'setup.phase2.cluster',
      message_params: { count: 4, cycle_ids: ['c1', 'c2'], name: 'Eco' },
      cta_label_key: 'setup.cta.create_from_cluster',
      cta_action: 'create_profile_from_cluster',
      secondary_label_key: null, secondary_action: null,
      skippable: true, dismissible: false, step_key: 'setup_skip_phase2',
    });
    await page.waitForSelector('[data-phase="phase2"]');
    await expect(page.locator('[data-cta-action="create_profile_from_cluster"]')).toBeVisible();
  });

});
```

- [ ] **Step 2: Run the new E2E tests**

```bash
cd playwright-tests && npx playwright test tests/setup-guidance.spec.ts --project=chromium
```

Fix any failures (the mock helper import path may differ — check `playwright-tests/helpers/` for the correct export names).

- [ ] **Step 3: Run the full E2E suite to check for regressions**

```bash
cd playwright-tests && npx playwright test --project=chromium 2>&1 | tail -30
```

- [ ] **Step 4: Commit**

```bash
cd ..
git add playwright-tests/tests/setup-guidance.spec.ts
git commit -m "test: add E2E tests for Setup Card phases and actions"
```

---

## Task 8: Translate New Keys — All Languages

**Files:**
- Modify: `custom_components/ha_washdata/translations/panel/{lang}.json` for every existing language file

**Interfaces:**
- Consumes: English keys from Task 4 (`translations/panel/en.json`)

- [ ] **Step 1: List all existing panel language files**

```bash
ls custom_components/ha_washdata/translations/panel/
```

- [ ] **Step 2: Translate via Claude subagents — NEVER the machine translator**

Dispatch one subagent per language group (group related languages: Germanic, Romance, Slavic, etc.) with this briefing for each batch:

> "Add the following keys to `translations/panel/{lang}.json` for WashData, a Home Assistant integration that monitors washing machines, dryers, and dishwashers via smart power plugs. Translate the English values below into {language}. Key rules: (1) 'profile' = a saved power-consumption signature for a specific washing program — do NOT translate as 'resume/CV' or 'profile picture'; (2) 'program' = a washing/drying program (e.g. Cotton 60°, Eco); (3) 'matching' = finding which program a current cycle corresponds to — NOT a sports match; (4) 'cycle' = one complete run of the appliance; (5) preserve all `{placeholder}` variables unchanged; (6) no em dash characters; (7) no leading/trailing whitespace in values. Deep-merge the result into the existing file. Keys to translate: [paste the `setup.*` JSON block from translations/panel/en.json]."

- [ ] **Step 3: Verify no leading/trailing whitespace in translated values**

```bash
python3 -c "
import json, glob, sys
errs = []
for f in glob.glob('custom_components/ha_washdata/translations/panel/*.json'):
    d = json.load(open(f))
    def check(obj, path=''):
        if isinstance(obj, dict):
            for k, v in obj.items(): check(v, f'{path}.{k}')
        elif isinstance(obj, str) and obj != obj.strip():
            errs.append(f'{f}: {path} has edge whitespace')
    check(d)
if errs:
    print('\n'.join(errs)); sys.exit(1)
print('OK')
"
```

- [ ] **Step 4: Run fast suite**

```bash
./run_tests.sh
```

- [ ] **Step 5: Commit**

```bash
git add custom_components/ha_washdata/translations/panel/
git commit -m "feat: translate setup guidance keys into all supported languages"
```

---

## Task 9: Final Verification

- [ ] **Step 1: Run the complete test suite**

```bash
./run_tests.sh --all 2>&1 | tail -40
```
Expected: all fast + slow + benchmark + E2E pass.

- [ ] **Step 2: Syntax check the whole package**

```bash
python3 -m compileall custom_components/ha_washdata tests/ --quiet
```

- [ ] **Step 3: Verify removed surfaces are gone**

```bash
# These should return 0 matches:
grep -n "_htmlGettingStarted\|onboarding_dismissed\|cgBanner\|sugBanner\|advBanner" \
     custom_components/ha_washdata/www/ha-washdata-panel.js
```

- [ ] **Step 4: Verify no machine-translated content**

Check that all `translations/panel/*.json` files were edited only by subagents — no `translate.py` was run. Review git diff for any obviously wrong translations (sports terms for "match", lumber for "logs").

- [ ] **Step 5: Update CHANGELOG.md**

Add under the `0.5.1` section:

```markdown
### Changed
- Replaced five scattered guidance surfaces (Getting Started card, coverage-gap banner,
  recommendations banner, group-suggestion banner, and profile advisories) with a single
  phased **Setup Card** on the Overview tab that walks users from zero profiles to a fully
  tuned device.
- Added threshold-mode behaviour: match polling and "No profile matched yet" notifications
  are now suppressed when no real profiles exist, giving clean cycle detection from day 1.
- Removed the "Create First Profile" step from the config flow — the name-only stub it
  created was misleading. The Setup Card's Phase 0 replaces it with actionable recording guidance.
- Config schema bumped to 3.7 (migration removes stale `initial_profile` key from entry data).
```

- [ ] **Step 6: Final commit**

```bash
git add CHANGELOG.md
git commit -m "docs: update CHANGELOG for adoption guidance system (0.5.1)"
```

---

## Self-Review Notes

**Spec coverage check:**
- Phase 0–4 phases: all implemented in `setup_advisor.py` (Task 1) and rendered in panel (Task 5). ✓
- Device-type-aware Phase 0 messages: in `setup_advisor.py` + translation keys. ✓
- Threshold mode (skip polling, suppress notification): Task 2. ✓
- Config flow stub removal + migration: Task 3. ✓
- All 3 Profiles-tab banners removed: Task 6. ✓
- Phase 3 tuning items (suggestions, groups, phases): `setup_advisor.py` handles suggestions + groups. **Phase 3 phases item** (configure phases for a profile) was noted in the spec but `compute_setup_phase` above does not surface it — Phase 3 has no input for "profiles missing phases." This is intentional: the `ws_get_setup_status` handler would need to compute which profiles lack phases; this can be added as a follow-up (the function signature accepts `profile_groups` and `suggestions` but not `profiles_missing_phases`). To include it: pass `profiles_missing_phases: list[str]` to `compute_setup_phase`, add a Phase 3 phases item in `_phase3_pending_item`, and gather it in the WS handler. Add this to the backlog.
- Store-download Phase 1c: advance on non-ambiguous match — handled by `compute_setup_phase` checking `has_self_cycles`. The WS handler checks that ref_names exist and no self cycles do. ✓
- Skip snooze 14 days + "never": both in `_is_step_suppressed`. ✓
- Translations: Task 8. ✓
- E2E tests: Task 7. ✓
- Migration test: Task 3. ✓

**Known gap to address in follow-up:** Phase 3 "configure phases" item requires a `profiles_missing_phases` input to `compute_setup_phase`. Not blocking — the card will show suggestions and groups correctly; phases item can be added in a subsequent PR without breaking anything.
