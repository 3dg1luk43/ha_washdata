"""Type-safe contract for the WashData WebSocket API (Group H1).

This module is the **single source of truth** for the shape of every
``ha_washdata/*`` WebSocket command: its request parameters (:data:`WS_COMMANDS`)
and its response payload (:data:`WS_RESPONSE_TYPES`, one ``TypedDict`` per
command). It is deliberately dependency-free — it imports nothing from Home
Assistant and nothing from :mod:`ws_api` — so it is safe to import from tooling
(``devtools/generate_ws_types.py``), from tests, and from ``ws_api`` itself
without creating an import cycle.

Nothing here runs in the hot path. ``ws_api._send_result`` consults the
registry only when the debug contract flag is on; otherwise this module is
imported once and never touched again.

**Keeping the contract in sync:** ``tests/test_ws_contract.py`` asserts that the
set of commands registered in :mod:`ws_api` matches the keys of
:data:`WS_COMMANDS` exactly, so adding, removing, or renaming a WS command must
be mirrored here or the suite fails.

Convention notes:
- ``TypedDict`` totality is used to express required-vs-optional keys: a class
  with ``total=True`` (the default) means every key is always present in the
  response; ``total=False`` means the response has conditional keys. (Because
  this file uses ``from __future__ import annotations``, per-field
  ``Required``/``NotRequired`` markers cannot be introspected reliably, so
  per-class totality is the mechanism the debug validator reads via
  ``__required_keys__`` / ``__optional_keys__``.)
- A handful of commands splat an upstream summary dict into their response and
  therefore have an open-ended key set; those are listed in
  :data:`WS_OPEN_RESPONSES` so the validator does not flag their extra keys.
"""
from __future__ import annotations

from typing import Any, TypedDict


# ─── Shared / trivial responses ────────────────────────────────────────────────

class SuccessResponse(TypedDict):
    """The common ``{"success": bool}`` acknowledgement."""

    success: bool


class OkResponse(TypedDict):
    """The ``{"ok": bool}`` acknowledgement used by the cycle-control commands."""

    ok: bool


# ─── Devices ────────────────────────────────────────────────────────────────────

class DeviceInfo(TypedDict):
    """One entry in :class:`GetDevicesResponse` — live state for a device."""

    entry_id: str
    perm: str
    title: str
    detector_state: str
    sub_state: str | None
    current_program: str | None
    time_remaining_s: float | None
    total_duration_s: float | None
    current_power_w: float | None
    cycle_progress_pct: float | None
    suggestions_count: int
    feedback_count: int
    recording: bool
    is_user_paused: bool
    manual_program: bool
    options: dict[str, Any]


class GetDevicesResponse(TypedDict):
    devices: list[DeviceInfo]


class GetDeviceCyclesResponse(TypedDict):
    entry_id: str
    cycles: list[dict[str, Any]]
    total: int
    has_more: bool


# ─── Settings ──────────────────────────────────────────────────────────────────

class GetOptionsResponse(TypedDict):
    options: dict[str, Any]


class GetSettingsChangelogResponse(TypedDict):
    changelog: list[dict[str, Any]]


# ─── Profiles ──────────────────────────────────────────────────────────────────

class GetProfilesResponse(TypedDict):
    profiles: list[dict[str, Any]]
    profile_health: dict[str, Any]
    profile_trends: dict[str, Any]
    coverage_gaps: dict[str, Any]
    profile_advisories: list[dict[str, Any]]


class CreateProfileResponse(TypedDict):
    success: bool
    name: str


class ProfileGroupInfo(TypedDict):
    name: str
    members: list[str]
    cohesion: float
    cohesive: bool


class GetProfileGroupsResponse(TypedDict):
    groups: list[ProfileGroupInfo]
    min_cohesion: float
    suggestions: list[dict[str, Any]]


class GetProfilePhasesResponse(TypedDict):
    phases: list[dict[str, Any]]


# ─── Maintenance log ───────────────────────────────────────────────────────────

class GetMaintenanceLogResponse(TypedDict):
    log: list[dict[str, Any]]
    due: Any
    event_types: list[str]
    reminders: dict[str, Any]


class AddMaintenanceEventResponse(TypedDict):
    success: bool
    event: dict[str, Any]


# ─── Phase catalog ─────────────────────────────────────────────────────────────

class GetPhaseCatalogResponse(TypedDict):
    phases: list[dict[str, Any]]
    device_type: str | None


# ─── Recording ─────────────────────────────────────────────────────────────────

class GetRecordingStateResponse(TypedDict, total=False):
    """``state`` is always present; the remaining keys depend on the state."""

    state: str
    duration_s: int
    sample_count: int
    start_time: str | None
    end_time: str | None


# ─── Feedbacks ─────────────────────────────────────────────────────────────────

class GetFeedbacksResponse(TypedDict):
    feedbacks: list[dict[str, Any]]


class DismissAllFeedbacksResponse(TypedDict):
    success: bool
    dismissed: int


# ─── Diagnostics ───────────────────────────────────────────────────────────────

class GetDiagnosticsResponse(TypedDict):
    stats: dict[str, Any]


class ClearDebugDataResponse(TypedDict):
    success: bool
    count: int


class ExportConfigResponse(TypedDict):
    json_data: str


# ─── Shared constants ──────────────────────────────────────────────────────────

class GetConstantsResponse(TypedDict):
    device_types: list[dict[str, Any]]
    state_colors: dict[str, Any]
    ml_lab_enabled: bool
    ml_suggestions_enabled: bool
    ml_training_available: bool
    PROFILE_MIN_WARMUP_CYCLES: Any
    store_online_available: bool
    store_online_enabled: bool
    store_web_origin: str


# ─── Suggestions ───────────────────────────────────────────────────────────────

class GetSuggestionsResponse(TypedDict):
    suggestions: list[dict[str, Any]]


class ApplySuggestionsResponse(TypedDict):
    success: bool
    applied: list[str]


class RunSuggestionAnalysisResponse(TypedDict, total=False):
    """``success`` plus whatever the analysis pass reports (e.g. ``count``)."""

    success: bool
    count: int


# ─── Cycle curve / interactive editing ─────────────────────────────────────────

class GetCyclePowerDataResponse(TypedDict, total=False):
    """``cycle_id`` / ``samples`` / ``full_duration_s`` are always present; the
    metadata keys are present only when the cycle is found."""

    cycle_id: str
    samples: list[list[float]]
    full_duration_s: float
    start_time: str | None
    end_time: str | None
    duration: float | None
    profile_name: str | None
    status: str | None
    energy_kwh: float | None
    artifacts: list[dict[str, Any]]
    restart_gaps: list[Any]


class AnalyzeSplitResponse(TypedDict):
    segments: list[list[float]]
    split_offsets: list[float]
    samples: list[list[float]]
    full_duration_s: float


class ApplySplitResponse(TypedDict):
    success: bool
    new_ids: list[str]


class ApplyMergeResponse(TypedDict):
    success: bool
    new_id: str


# ─── Profile envelope / member cycles ──────────────────────────────────────────

class ProfileEnvelope(TypedDict):
    avg: list[list[float]]
    min: list[list[float]]
    max: list[list[float]]
    target_duration: float | None
    avg_energy: float | None
    duration_std_dev: float | None
    cycle_count: int


class GetProfileEnvelopeResponse(TypedDict):
    envelope: ProfileEnvelope | None


class GetProfileCyclesResponse(TypedDict):
    cycles: list[dict[str, Any]]


# ─── Panel config + RBAC ───────────────────────────────────────────────────────

class GetPanelConfigResponse(TypedDict, total=False):
    """``panel`` / ``is_admin`` / ``user`` / ``prefs`` are always present;
    ``rbac`` and ``users`` are added for admins only."""

    panel: dict[str, Any]
    is_admin: bool
    user: dict[str, Any]
    prefs: dict[str, Any]
    rbac: dict[str, Any]
    users: list[dict[str, Any]]


# ─── Live match debug ──────────────────────────────────────────────────────────

class GetMatchDebugResponse(TypedDict):
    confidence: float | None
    ambiguous: bool
    candidates: list[dict[str, Any]]


# ─── Live power history ────────────────────────────────────────────────────────

class GetPowerHistoryResponse(TypedDict, total=False):
    """Base keys are always present; ``cycle_start_iso`` only while a cycle runs."""

    cycle_active: bool
    cycle_elapsed_s: float
    live: list[list[float]]
    raw: list[list[float]]
    restart_gaps: list[Any]
    cycle_start_iso: str


# ─── Logs ──────────────────────────────────────────────────────────────────────

class GetLogsResponse(TypedDict):
    logs: list[dict[str, Any]]


# ─── ML Lab shadow-mode comparison ─────────────────────────────────────────────

class GetMlComparisonResponse(TypedDict, total=False):
    """Two shapes: a disabled report (``enabled=False`` + ``error``) and an
    enabled report; both carry ``cycles`` + ``settings_comparison``."""

    enabled: bool
    error: str
    cycles: list[dict[str, Any]]
    settings_comparison: dict[str, Any]
    cycle_count: int
    evaluated_count: int
    model_source: dict[str, Any]
    profile_stats: dict[str, Any]
    ml_suggestions_enabled: bool


# ─── On-device ML training ─────────────────────────────────────────────────────

class GetMlTrainingStatusResponse(TypedDict):
    available: bool
    enabled: bool
    running: bool
    last_trained: str | None
    cycle_count: int
    min_cycles: int
    interval_days: int
    hour: int
    on_device_models: dict[str, Any]
    matching: dict[str, Any]


# ─── Playground (F3) ───────────────────────────────────────────────────────────

class PlaygroundSummary(TypedDict):
    cycles: int
    requested: int
    concurrency: int
    detected: int
    missed: int
    false_end: int
    match_correct: int
    match_wrong: int
    unmatched: int
    skipped_ids: list[str]


class RunPlaygroundSimulationResponse(TypedDict):
    results: list[dict[str, Any]]
    summary: PlaygroundSummary


class RunPlaygroundCycleDetailResponse(TypedDict, total=False):
    cycle_id: Any
    label: str | None
    duration_s: float | None
    config_summary: dict[str, Any]
    series: list[dict[str, Any]]
    events: list[dict[str, Any]]
    alerts: list[dict[str, Any]]
    outcome: dict[str, Any]
    error: str


class RunPlaygroundHistoryResponse(TypedDict, total=False):
    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    baseline_rows: list[dict[str, Any]]
    baseline_summary: dict[str, Any]
    diff: dict[str, list[str]]


class RunPlaygroundSweepResponse(TypedDict, total=False):
    param: str
    objective: str
    points: list[dict[str, Any]]
    current_value: Any
    best_value: Any
    best_metric: float | None
    param_x: str
    param_y: str
    x_values: list[float]
    y_values: list[float]
    grid: list[list[Any]]
    best: dict[str, Any]
    current: dict[str, Any]
    error: str


class DtwStage2Scores(TypedDict):
    correlation: float
    mae_score: float
    score: float


class DtwScores(TypedDict):
    l1_score: float
    ddtw_score: float
    ensemble_score: float
    blend_weight: float
    blended_score: float


class DtwStage4Scores(TypedDict):
    duration_agreement: float
    energy_agreement: float
    final_score: float


class GetDtwDebugResponse(TypedDict):
    cycle_id: Any
    profile_name: str
    grid_n: int
    cycle_duration_s: float
    profile_duration_s: float
    cycle_trace: list[list[float]]
    profile_trace: list[list[float]]
    stage2: DtwStage2Scores
    dtw: DtwScores
    stage4: DtwStage4Scores
    warp_path: list[list[int]]


class TaskSnapshot(TypedDict, total=False):
    id: str
    entry_id: str
    kind: str
    label: str
    state: str
    done: int
    total: int
    progress: float | None
    eta_s: float | None
    started_at: float
    updated_at: float
    finished_at: float | None
    error: str | None
    has_result: bool
    result: Any


class ListTasksResponse(TypedDict):
    tasks: list[TaskSnapshot]


class CancelTaskResponse(TypedDict):
    cancelled: bool


class StartTaskResponse(TypedDict):
    task_id: str


class SubscribeTasksResponse(TypedDict, total=False):
    """Empty ack for the ``subscribe_tasks`` subscription; the live data arrives
    as ``{"type": "task", "task": TaskSnapshot}`` event messages, not in this
    result."""


# ─── Response-type registry ────────────────────────────────────────────────────

#: Map ``<command>`` (the string after ``ha_washdata/``) -> its response TypedDict.
#: Commands whose response is the trivial ``{"success": True}`` share
#: :class:`SuccessResponse`; the cycle-control commands share :class:`OkResponse`.
class StoreStatusResponse(TypedDict, total=False):
    """Community-store status / identity (no refresh token)."""
    enabled: bool
    connected: bool
    uid: str | None
    name: str | None
    brand: str | None
    model: str | None
    disabled: bool


class StoreItemsResponse(TypedDict, total=False):
    """A list of store docs (devices / profiles / cycles)."""
    items: list
    disabled: bool


class StoreImportResponse(TypedDict, total=False):
    """Result of importing a reference cycle."""
    profile: str
    cycle_id: str
    error: str
    disabled: bool


class StoreUploadResponse(TypedDict, total=False):
    """Result of sharing (uploading) a local cycle."""
    store_cycle_id: str
    error: str
    detail: str | None
    disabled: bool


class StoreSimpleResponse(TypedDict, total=False):
    """Connect/disconnect acknowledgement + identity/error markers."""
    connected: bool
    uid: str | None
    name: str | None
    brand: str | None
    model: str | None
    error: str
    disabled: bool


class StoreQualityResponse(TypedDict, total=False):
    """Device 5-star quality summary (count + average)."""
    avg: float | None
    count: int
    disabled: bool


class StoreConfirmResponse(TypedDict, total=False):
    """Result of confirming a device (drives community auto-approval)."""
    confirmed: bool
    confirmCount: int
    status: str | None
    error: str
    disabled: bool


class StoreOnlineResponse(TypedDict, total=False):
    """Result of toggling integration-wide online features."""
    enabled: bool
    ok: bool
    error: str
    disabled: bool


class StoreDeviceProfilesResponse(TypedDict, total=False):
    """Resolved store deviceId + its profiles (for the Share dialog picker)."""
    device_id: str
    items: list
    disabled: bool


WS_RESPONSE_TYPES: dict[str, type] = {
    "get_devices": GetDevicesResponse,
    "get_device_cycles": GetDeviceCyclesResponse,
    "get_options": GetOptionsResponse,
    "set_options": SuccessResponse,
    "get_settings_changelog": GetSettingsChangelogResponse,
    "get_profiles": GetProfilesResponse,
    "create_profile": CreateProfileResponse,
    "rename_profile": SuccessResponse,
    "delete_profile": SuccessResponse,
    "get_profile_groups": GetProfileGroupsResponse,
    "save_profile_group": SuccessResponse,
    "rename_profile_group": SuccessResponse,
    "delete_profile_group": SuccessResponse,
    "rebuild_envelopes": SuccessResponse,
    "get_profile_phases": GetProfilePhasesResponse,
    "set_profile_phases": SuccessResponse,
    "get_maintenance_log": GetMaintenanceLogResponse,
    "add_maintenance_event": AddMaintenanceEventResponse,
    "delete_maintenance_event": SuccessResponse,
    "label_cycle": SuccessResponse,
    "delete_cycle": SuccessResponse,
    "auto_label_cycles": SuccessResponse,
    "get_phase_catalog": GetPhaseCatalogResponse,
    "create_phase": SuccessResponse,
    "update_phase": SuccessResponse,
    "delete_phase": SuccessResponse,
    "get_recording_state": GetRecordingStateResponse,
    "start_recording": SuccessResponse,
    "stop_recording": SuccessResponse,
    "process_recording": SuccessResponse,
    "discard_recording": SuccessResponse,
    "get_feedbacks": GetFeedbacksResponse,
    "resolve_feedback": SuccessResponse,
    "dismiss_all_feedbacks": DismissAllFeedbacksResponse,
    "get_diagnostics": GetDiagnosticsResponse,
    "reprocess_history": StartTaskResponse,
    "clear_debug_data": ClearDebugDataResponse,
    "wipe_history": SuccessResponse,
    "export_config": ExportConfigResponse,
    "import_config": SuccessResponse,
    "get_constants": GetConstantsResponse,
    "get_suggestions": GetSuggestionsResponse,
    "apply_suggestions": ApplySuggestionsResponse,
    "clear_suggestions": SuccessResponse,
    "run_suggestion_analysis": RunSuggestionAnalysisResponse,
    "get_cycle_power_data": GetCyclePowerDataResponse,
    "trim_cycle": SuccessResponse,
    "analyze_split": AnalyzeSplitResponse,
    "apply_split": ApplySplitResponse,
    "apply_merge": ApplyMergeResponse,
    "get_profile_envelope": GetProfileEnvelopeResponse,
    "get_profile_cycles": GetProfileCyclesResponse,
    "get_panel_config": GetPanelConfigResponse,
    "set_panel_config": SuccessResponse,
    "set_user_prefs": SuccessResponse,
    "get_match_debug": GetMatchDebugResponse,
    "set_program": SuccessResponse,
    "get_power_history": GetPowerHistoryResponse,
    "get_logs": GetLogsResponse,
    "get_ml_comparison": GetMlComparisonResponse,
    "get_ml_training_status": GetMlTrainingStatusResponse,
    "trigger_ml_training": StartTaskResponse,
    "revert_matching_config": SuccessResponse,
    "revert_ml_models": SuccessResponse,
    "set_ml_review": SuccessResponse,
    "pause_cycle": OkResponse,
    "resume_cycle": OkResponse,
    "terminate_cycle": OkResponse,
    "run_playground_simulation": RunPlaygroundSimulationResponse,
    "run_playground_cycle_detail": RunPlaygroundCycleDetailResponse,
    "run_playground_history": RunPlaygroundHistoryResponse,
    "run_playground_sweep": RunPlaygroundSweepResponse,
    "get_dtw_debug": GetDtwDebugResponse,
    "list_tasks": ListTasksResponse,
    "subscribe_tasks": SubscribeTasksResponse,
    "cancel_task": CancelTaskResponse,
    "get_task_result": TaskSnapshot,
    "start_playground_history": StartTaskResponse,
    "start_playground_sweep": StartTaskResponse,
    "store_status": StoreStatusResponse,
    "store_connect": StoreSimpleResponse,
    "store_disconnect": StoreSimpleResponse,
    "store_search_devices": StoreItemsResponse,
    "store_get_profiles": StoreItemsResponse,
    "store_get_cycles": StoreItemsResponse,
    "store_import_cycle": StoreImportResponse,
    "store_upload_cycle": StoreUploadResponse,
    "store_list_brands": StoreItemsResponse,
    "store_get_device_quality": StoreQualityResponse,
    "store_confirm_device": StoreConfirmResponse,
    "store_rate_device": StoreOnlineResponse,
    "store_set_online": StoreOnlineResponse,
    "store_get_device_profiles": StoreDeviceProfilesResponse,
}

#: Commands whose response splats an upstream summary dict and therefore has an
#: open-ended top-level key set. The debug validator still checks required keys
#: for these but does not flag extra keys.
WS_OPEN_RESPONSES: frozenset[str] = frozenset({
    "run_suggestion_analysis",
    # Playground what-if responses carry nested/variant shapes (incl. an error
    # variant); skip strict extra-key validation.
    "run_playground_cycle_detail",
    "run_playground_history",
    "run_playground_sweep",
    # Task snapshot splats a variant key set (result present only when finished).
    "get_task_result",
})


# ─── Request-parameter contract ────────────────────────────────────────────────

def _p(name: str, type_: str, required: bool = True, **extra: Any) -> dict[str, Any]:
    """Build one request-parameter descriptor.

    ``type_`` is a small vocabulary of type names mirroring the voluptuous
    schema: ``str``, ``int``, ``float``, ``bool``, ``dict``, ``list``,
    ``list[str]``, ``list[float]``, and the nullable ``str|null`` / ``float|null``.
    ``extra`` may carry an ``enum`` list of allowed string values.
    """
    param: dict[str, Any] = {"name": name, "required": required, "type": type_}
    param.update(extra)
    return param


# entry_id is required by nearly every command; a tiny helper keeps it terse.
def _entry() -> dict[str, Any]:
    return _p("entry_id", "str", True)


#: Map ``<command>`` -> ``{"params": [<param descriptor>, ...]}``. Hand-maintained
#: to mirror the voluptuous ``@websocket_command`` schemas in :mod:`ws_api`
#: (the implicit ``type`` discriminator is omitted). This is the source of truth
#: ``devtools/generate_ws_types.py`` reads to emit TS types + Markdown docs.
WS_COMMANDS: dict[str, dict] = {
    "get_devices": {"params": []},
    "get_device_cycles": {"params": [
        _entry(),
        _p("limit", "int", False),
        _p("offset", "int", False),
    ]},
    "get_options": {"params": [_entry()]},
    "set_options": {"params": [_entry(), _p("options", "dict")]},
    "get_settings_changelog": {"params": [_entry()]},
    "get_profiles": {"params": [_entry()]},
    "create_profile": {"params": [
        _entry(),
        _p("name", "str"),
        _p("reference_cycle", "str|null", False),
        _p("manual_duration_min", "float|null", False),
    ]},
    "rename_profile": {"params": [
        _entry(),
        _p("profile_name", "str"),
        _p("new_name", "str"),
        _p("manual_duration_min", "float|null", False),
    ]},
    "delete_profile": {"params": [
        _entry(),
        _p("profile_name", "str"),
        _p("unlabel_cycles", "bool", False),
    ]},
    "get_profile_groups": {"params": [_entry()]},
    "save_profile_group": {"params": [
        _entry(),
        _p("name", "str"),
        _p("members", "list[str]"),
    ]},
    "rename_profile_group": {"params": [
        _entry(),
        _p("name", "str"),
        _p("new_name", "str"),
    ]},
    "delete_profile_group": {"params": [_entry(), _p("name", "str")]},
    "rebuild_envelopes": {"params": [_entry()]},
    "get_profile_phases": {"params": [_entry(), _p("profile_name", "str")]},
    "set_profile_phases": {"params": [
        _entry(),
        _p("profile_name", "str"),
        _p("phases", "list"),
    ]},
    "get_maintenance_log": {"params": [_entry()]},
    "add_maintenance_event": {"params": [
        _entry(),
        _p("event_type", "str"),
        _p("date", "str|null", False),
        _p("notes", "str", False),
    ]},
    "delete_maintenance_event": {"params": [_entry(), _p("event_id", "str")]},
    "label_cycle": {"params": [
        _entry(),
        _p("cycle_id", "str"),
        _p("profile_name", "str|null", False),
        _p("new_profile_name", "str|null", False),
    ]},
    "delete_cycle": {"params": [_entry(), _p("cycle_id", "str")]},
    "auto_label_cycles": {"params": [
        _entry(),
        _p("confidence_threshold", "float", False),
    ]},
    "get_phase_catalog": {"params": [_entry(), _p("device_type", "str|null", False)]},
    "create_phase": {"params": [
        _entry(),
        _p("device_type", "str"),
        _p("name", "str"),
        _p("description", "str", False),
    ]},
    "update_phase": {"params": [
        _entry(),
        _p("phase_id", "str"),
        _p("new_name", "str"),
        _p("description", "str", False),
    ]},
    "delete_phase": {"params": [_entry(), _p("phase_id", "str")]},
    "get_recording_state": {"params": [_entry()]},
    "start_recording": {"params": [_entry()]},
    "stop_recording": {"params": [_entry()]},
    "process_recording": {"params": [
        _entry(),
        _p("profile_name", "str"),
        _p("save_mode", "str", enum=["new_profile", "existing_profile"]),
        _p("head_trim", "float", False),
        _p("tail_trim", "float", False),
    ]},
    "discard_recording": {"params": [_entry()]},
    "get_feedbacks": {"params": [_entry()]},
    "resolve_feedback": {"params": [
        _entry(),
        _p("cycle_id", "str"),
        _p("action", "str", enum=["confirm", "correct", "ignore", "delete"]),
        _p("corrected_profile", "str|null", False),
        _p("corrected_duration_min", "float|null", False),
    ]},
    "dismiss_all_feedbacks": {"params": [_entry()]},
    "get_diagnostics": {"params": [_entry()]},
    "reprocess_history": {"params": [_entry()]},
    "clear_debug_data": {"params": [_entry()]},
    "wipe_history": {"params": [_entry()]},
    "export_config": {"params": [_entry()]},
    "import_config": {"params": [_entry(), _p("json_data", "str")]},
    "get_constants": {"params": []},
    "get_suggestions": {"params": [_entry()]},
    "apply_suggestions": {"params": [_entry(), _p("keys", "list[str]")]},
    "clear_suggestions": {"params": [_entry()]},
    "run_suggestion_analysis": {"params": [_entry()]},
    "get_cycle_power_data": {"params": [_entry(), _p("cycle_id", "str")]},
    "trim_cycle": {"params": [
        _entry(),
        _p("cycle_id", "str"),
        _p("start_s", "float"),
        _p("end_s", "float"),
    ]},
    "analyze_split": {"params": [
        _entry(),
        _p("cycle_id", "str"),
        _p("gap_seconds", "int", False),
    ]},
    "apply_split": {"params": [
        _entry(),
        _p("cycle_id", "str"),
        _p("split_offsets", "list[float]"),
        _p("segment_profiles", "list", False),
    ]},
    "apply_merge": {"params": [
        _entry(),
        _p("cycle_ids", "list[str]"),
        _p("target_profile", "str|null", False),
        _p("new_profile_name", "str|null", False),
    ]},
    "get_profile_envelope": {"params": [_entry(), _p("profile_name", "str")]},
    "get_profile_cycles": {"params": [
        _entry(),
        _p("profile_name", "str"),
        _p("limit", "int", False),
    ]},
    "get_panel_config": {"params": []},
    "set_panel_config": {"params": [
        _p("panel", "dict", False),
        _p("rbac", "dict", False),
    ]},
    "set_user_prefs": {"params": [_p("prefs", "dict")]},
    "get_match_debug": {"params": [_entry()]},
    "set_program": {"params": [_entry(), _p("program", "str|null")]},
    "get_power_history": {"params": [_entry(), _p("with_raw", "bool", False)]},
    "get_logs": {"params": [
        _p("level", "str|null", False),
        _p("limit", "int", False),
    ]},
    "get_ml_comparison": {"params": [_entry()]},
    "get_ml_training_status": {"params": [_entry()]},
    "trigger_ml_training": {"params": [_entry()]},
    "revert_matching_config": {"params": [_entry()]},
    "revert_ml_models": {"params": [_entry()]},
    "set_ml_review": {"params": [
        _entry(),
        _p("cycle_id", "str"),
        _p("quality", "str", False, enum=["", "bad", "good", "unusable"]),
        _p("golden", "bool", False),
        _p("tags", "list[str]", False),
        _p("notes", "str", False),
    ]},
    "pause_cycle": {"params": [_entry()]},
    "resume_cycle": {"params": [_entry()]},
    "terminate_cycle": {"params": [_entry()]},
    "run_playground_simulation": {"params": [
        _entry(),
        _p("cycle_ids", "list[str]", False),
        _p("settings_override", "dict", False),
        _p("concurrency", "int", False),
    ]},
    "run_playground_cycle_detail": {"params": [
        _entry(),
        _p("cycle_id", "str"),
        _p("settings_override", "dict", False),
    ]},
    "run_playground_history": {"params": [
        _entry(),
        _p("cycle_ids", "list[str]", False),
        _p("settings_override", "dict", False),
        _p("concurrency", "int", False),
    ]},
    "run_playground_sweep": {"params": [
        _entry(),
        _p("param", "str"),
        _p("values", "list[float]"),
        _p("objective", "str"),
        _p("cycle_ids", "list[str]", False),
        _p("concurrency", "int", False),
        _p("param_y", "str", False),
        _p("values_y", "list[float]", False),
    ]},
    "get_dtw_debug": {"params": [
        _entry(),
        _p("cycle_id", "str"),
        _p("profile_name", "str|null", False),
    ]},
    "list_tasks": {"params": [_p("entry_id", "str|null", False)]},
    "subscribe_tasks": {"params": [_p("entry_id", "str|null", False)]},
    "cancel_task": {"params": [_p("task_id", "str")]},
    "get_task_result": {"params": [_p("task_id", "str")]},
    "start_playground_history": {"params": [
        _entry(),
        _p("cycle_ids", "list[str]", False),
        _p("settings_override", "dict", False),
    ]},
    "start_playground_sweep": {"params": [
        _entry(),
        _p("param", "str"),
        _p("values", "list[float]"),
        _p("objective", "str"),
        _p("param_y", "str|null", False),
        _p("values_y", "list[float]", False),
    ]},
    # Community store (online features)
    "store_status": {"params": [_entry()]},
    "store_connect": {"params": [
        _entry(), _p("refresh_token", "str"), _p("uid", "str"), _p("name", "str|null", False),
    ]},
    "store_disconnect": {"params": [_entry()]},
    "store_search_devices": {"params": [
        _entry(), _p("query", "str|null", False), _p("appliance_type", "str|null", False),
        _p("model_query", "str|null", False), _p("include_pending", "bool", False),
    ]},
    "store_list_brands": {"params": [
        _entry(), _p("query", "str|null", False), _p("include_pending", "bool", False),
    ]},
    "store_get_profiles": {"params": [_entry(), _p("device_id", "str")]},
    "store_get_cycles": {"params": [_entry(), _p("profile_id", "str")]},
    "store_get_device_quality": {"params": [_entry(), _p("device_id", "str")]},
    "store_get_device_profiles": {"params": [_entry(), _p("brand", "str"), _p("model", "str"), _p("appliance_type", "str")]},
    "store_confirm_device": {"params": [_entry(), _p("device_id", "str")]},
    "store_rate_device": {"params": [_entry(), _p("device_id", "str"), _p("rating", "int")]},
    "store_set_online": {"params": [_entry(), _p("enabled", "bool")]},
    "store_import_cycle": {"params": [
        _entry(), _p("cycle_id", "str"),
        _p("target_profile", "str|null", False), _p("new_profile_name", "str|null", False),
    ]},
    "store_upload_cycle": {"params": [
        _entry(), _p("local_cycle_id", "str"), _p("program", "str"), _p("description", "str|null", False),
    ]},
}


#: The domain prefix every command name carries on the wire.
WS_PREFIX = "ha_washdata"


def full_command_name(command: str) -> str:
    """``"get_devices"`` -> ``"ha_washdata/get_devices"``."""
    return f"{WS_PREFIX}/{command}"


__all__ = [
    "WS_COMMANDS",
    "WS_RESPONSE_TYPES",
    "WS_OPEN_RESPONSES",
    "WS_PREFIX",
    "full_command_name",
]
