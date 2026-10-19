# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""0.5.9 option trim: retired settings are fixed at their shipped value or merged,
and every key nothing reads is stripped.

Fixed (measured neutral or better before removal): the unmatch threshold (0.35),
the sampling throttle (the device type's default), the anti-wrinkle exit power
(max(0.8 W, stop)), the ML training hour / minimum cycles / interval (2 / 30 / 7).
Dead: ``duration_tolerance``. Merged: the
external end trigger is on whenever its entity is set. Leftovers of features
removed earlier go too. One set, ``RETIRED_OPTION_KEYS``, extended into the
unreleased 3.11 -> 3.12 step; the import, the undo-import and ``ws_set_options``
strip the same set. ``notify_only_when_home`` / ``notify_people`` are NOT merged:
the people list also fills the ``person_entity_id`` / ``person_name`` variables
of the notification actions, whatever the flag.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.ha_washdata as washdata
from custom_components.ha_washdata import async_migrate_entry, ws_api
from custom_components.ha_washdata.const import (
    CONF_DEVICE_TYPE,
    CONF_EXTERNAL_END_TRIGGER,
    CONF_EXTERNAL_END_TRIGGER_ENABLED,
    CONF_EXTERNAL_END_TRIGGER_INVERTED,
    CONF_MIN_POWER,
    CONF_ML_TRAINING_ENABLED,
    CONF_NOTIFY_EVENTS,
    CONF_NOTIFY_ONLY_WHEN_HOME,
    CONF_NOTIFY_PEOPLE,
    CONF_NOTIFY_SERVICE,
    CONF_OFF_DELAY,
    CONF_POWER_SENSOR,
    CONFIG_ENTRY_MINOR_VERSION,
    CONFIG_ENTRY_VERSION,
    DEFAULT_PROFILE_UNMATCH_THRESHOLD,
    DOMAIN,
    ML_TRAINING_HOUR,
    ML_TRAINING_MIN_CYCLES,
    RETIRED_OPTION_KEYS,
)

from .real_manager import boot, make_entry

TRIGGER = "binary_sensor.washer_done"

# Every retired key read by nothing, with a value a real export carried (or could).
OLD_VALUES: dict[str, Any] = {
    "duration_tolerance": 0.42,
    "anti_wrinkle_exit_power": 0.2,
    "profile_unmatch_threshold": 0.25,
    "sampling_interval": 40.3,
    "ml_training_hour": 1,
    "ml_training_min_cycles": 1,
    "ml_training_interval_days": 1,
    "end_repeat_count": 2.0,
    "smoothing_window": 4,
    "profile_duration_tolerance": 0.35,
    "max_past_cycles": 200,
    "max_full_traces_per_profile": 20,
    "max_full_traces_unlabeled": 20,
    "enable_phase_matching": True,
    "auto_merge_gap_seconds": 600.0,
    "auto_merge_lookback_hours": 3.0,
    "notify_fire_events": True,
}
# Settings that stay, including the pair item 8 leaves alone.
KEPT: dict[str, Any] = {
    CONF_OFF_DELAY: 120,
    CONF_MIN_POWER: 2.0,
    CONF_NOTIFY_PEOPLE: ["person.alex"],
    CONF_NOTIFY_ONLY_WHEN_HOME: False,
    CONF_EXTERNAL_END_TRIGGER_INVERTED: True,
}


def _entry(
    hass: HomeAssistant,
    options: dict[str, Any],
    *,
    minor: int = 11,
    data: dict[str, Any] | None = None,
) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Washer",
        data=data if data is not None else {"name": "Washer"},
        options=dict(options),
        version=3,
        minor_version=minor,
    )
    entry.add_to_hass(hass)
    return entry


def _no_retired_keys(entry: MockConfigEntry) -> None:
    for key in RETIRED_OPTION_KEYS:
        assert key not in entry.options, key
        assert key not in entry.data, key


def test_the_retired_set_is_exactly_the_trimmed_keys() -> None:
    """Guards against a key slipping in (or out) of the one set every path strips."""
    assert RETIRED_OPTION_KEYS == set(OLD_VALUES) | {
        CONF_NOTIFY_SERVICE, CONF_NOTIFY_EVENTS, CONF_EXTERNAL_END_TRIGGER_ENABLED,
    }
    assert not RETIRED_OPTION_KEYS & set(KEPT)
    assert CONF_EXTERNAL_END_TRIGGER not in RETIRED_OPTION_KEYS


# ── 3.11 -> 3.12 ─────────────────────────────────────────────────────────────


async def test_every_retired_key_goes_from_options_and_data(hass: HomeAssistant) -> None:
    entry = _entry(
        hass, {**OLD_VALUES, **KEPT},
        data={"name": "Washer", "sampling_interval": 30.0, "duration_tolerance": 0.1},
    )

    assert await async_migrate_entry(hass, entry) is True

    assert (entry.version, entry.minor_version) == (3, 12) == (
        CONFIG_ENTRY_VERSION, CONFIG_ENTRY_MINOR_VERSION,
    )
    _no_retired_keys(entry)
    assert dict(entry.options) == KEPT
    assert dict(entry.data) == {"name": "Washer"}


@pytest.mark.parametrize(
    ("stored", "kept"),
    [
        ({CONF_EXTERNAL_END_TRIGGER_ENABLED: False, CONF_EXTERNAL_END_TRIGGER: TRIGGER}, None),
        # Absent meant off at runtime (options.get(..., False)), so it is cleared too.
        ({CONF_EXTERNAL_END_TRIGGER: TRIGGER}, None),
        ({CONF_EXTERNAL_END_TRIGGER_ENABLED: True, CONF_EXTERNAL_END_TRIGGER: TRIGGER}, TRIGGER),
        ({CONF_EXTERNAL_END_TRIGGER_ENABLED: True}, None),
        ({CONF_EXTERNAL_END_TRIGGER_ENABLED: False}, None),
    ],
    ids=["off", "flag_absent", "on", "on_without_entity", "off_without_entity"],
)
async def test_the_trigger_flag_folds_into_the_entity(
    hass: HomeAssistant, stored: dict[str, Any], kept: str | None
) -> None:
    """Behaviour unchanged: a trigger that was off loses its entity, one that was
    on keeps it, and the flag goes either way."""
    entry = _entry(
        hass, {CONF_OFF_DELAY: 120, **stored},
        data={"name": "Washer", CONF_EXTERNAL_END_TRIGGER: "binary_sensor.stale"},
    )

    assert await async_migrate_entry(hass, entry) is True

    assert entry.options.get(CONF_EXTERNAL_END_TRIGGER) == kept
    assert CONF_EXTERNAL_END_TRIGGER_ENABLED not in entry.options
    # The runtime read options only: a copy in data was never used and goes.
    assert CONF_EXTERNAL_END_TRIGGER not in entry.data
    assert entry.options[CONF_OFF_DELAY] == 120


async def test_the_migration_is_idempotent(hass: HomeAssistant) -> None:
    entry = _entry(hass, {
        **OLD_VALUES, **KEPT,
        CONF_EXTERNAL_END_TRIGGER_ENABLED: True, CONF_EXTERNAL_END_TRIGGER: TRIGGER,
    })
    assert await async_migrate_entry(hass, entry) is True
    first = (dict(entry.data), dict(entry.options), entry.minor_version)

    assert await async_migrate_entry(hass, entry) is True
    assert (dict(entry.data), dict(entry.options), entry.minor_version) == first
    assert entry.options[CONF_EXTERNAL_END_TRIGGER] == TRIGGER

    # The strip on its own output changes nothing.
    opts, data = dict(entry.options), dict(entry.data)
    assert washdata._retire_options(opts, data) == []  # noqa: SLF001
    assert (data, opts) == (first[0], first[1])


async def test_a_clean_3_11_entry_only_advances(hass: HomeAssistant) -> None:
    entry = _entry(hass, dict(KEPT))
    assert await async_migrate_entry(hass, entry) is True
    assert entry.minor_version == 12
    assert dict(entry.options) == KEPT


# ── One-pass legacy path (below 3.6) ─────────────────────────────────────────


async def test_the_one_pass_path_strips_and_seeds_no_retired_key(hass: HomeAssistant) -> None:
    """It used to SEED duration_tolerance, smoothing_window and
    profile_duration_tolerance; now it drops them like the chain."""
    legacy = _entry(
        hass,
        {CONF_DEVICE_TYPE: "washing_machine", CONF_POWER_SENSOR: "sensor.p",
         **OLD_VALUES, CONF_EXTERNAL_END_TRIGGER_ENABLED: False,
         CONF_EXTERNAL_END_TRIGGER: TRIGGER},
        minor=5,
    )
    bare = _entry(
        hass, {CONF_DEVICE_TYPE: "washing_machine", CONF_POWER_SENSOR: "sensor.p"}, minor=5,
    )

    assert await async_migrate_entry(hass, legacy) is True
    assert await async_migrate_entry(hass, bare) is True

    for entry in (legacy, bare):
        assert (entry.version, entry.minor_version) == (
            CONFIG_ENTRY_VERSION, CONFIG_ENTRY_MINOR_VERSION,
        )
        _no_retired_keys(entry)
    assert CONF_EXTERNAL_END_TRIGGER not in legacy.options
    # The people pair is still seeded (item 8 not merged).
    assert bare.options[CONF_NOTIFY_PEOPLE] == []
    assert bare.options[CONF_NOTIFY_ONLY_WHEN_HOME] is False


# ── Import, undo-import and save cannot bring a key back ─────────────────────


async def test_an_import_strips_every_retired_key(hass: HomeAssistant) -> None:
    entry = _entry(hass, {CONF_OFF_DELAY: 120}, minor=CONFIG_ENTRY_MINOR_VERSION)

    await ws_api.async_apply_imported_entry_options(
        hass, entry,
        {"entry_options": {**OLD_VALUES, CONF_EXTERNAL_END_TRIGGER_ENABLED: True,
                           CONF_OFF_DELAY: 300}},
        "import_config",
    )

    _no_retired_keys(entry)
    assert entry.options[CONF_OFF_DELAY] == 300, "real tunables must still be imported"


async def test_an_undo_import_does_not_restore_them(hass: HomeAssistant) -> None:
    entry = _entry(hass, {CONF_OFF_DELAY: 300}, minor=CONFIG_ENTRY_MINOR_VERSION)

    await ws_api._async_restore_entry_options(  # noqa: SLF001
        hass, entry, {**OLD_VALUES, CONF_EXTERNAL_END_TRIGGER_ENABLED: False, CONF_OFF_DELAY: 120},
    )

    _no_retired_keys(entry)
    assert entry.options[CONF_OFF_DELAY] == 120


async def test_a_save_drops_a_submitted_or_stray_retired_key(hass: HomeAssistant) -> None:
    """An older panel still submits the retired fields; a 3.12 entry from before
    the trim may still store them. Neither survives a save."""
    entry = MockConfigEntry(
        domain=DOMAIN, title="Washer", data={CONF_POWER_SENSOR: "sensor.power"},
        options={CONF_OFF_DELAY: 120, "sampling_interval": 30.0},
        version=3, minor_version=CONFIG_ENTRY_MINOR_VERSION,
    )
    entry.add_to_hass(hass)
    connection = MagicMock()
    with patch.object(ws_api, "_record_option_changes", AsyncMock()):
        await ws_api.ws_set_options.__wrapped__(
            hass, connection,
            {"id": 1, "entry_id": entry.entry_id,
             "options": {CONF_OFF_DELAY: 300, "profile_unmatch_threshold": 0.5,
                         CONF_EXTERNAL_END_TRIGGER_ENABLED: True}},
        )

    connection.send_error.assert_not_called()
    assert dict(entry.options) == {CONF_OFF_DELAY: 300}


def test_a_shared_bundle_carries_none_of_them() -> None:
    from custom_components.ha_washdata.const import sanitize_shared_settings

    assert not set(sanitize_shared_settings(dict(OLD_VALUES))) & RETIRED_OPTION_KEYS


# ── Runtime: a stored value is ignored ───────────────────────────────────────


async def test_the_manager_runs_the_fixed_values_whatever_is_stored(hass: HomeAssistant) -> None:
    mgr = await boot(hass, make_entry(hass, {**OLD_VALUES, CONF_ML_TRAINING_ENABLED: True}))

    assert mgr._unmatch_threshold == DEFAULT_PROFILE_UNMATCH_THRESHOLD == 0.35  # noqa: SLF001
    assert mgr._sampling_interval == 2.0  # the washer's device default, not 40.3  # noqa: SLF001
    assert not hasattr(mgr.detector.config, "anti_wrinkle_exit_power")

    # A stored minimum of 1 cycle no longer lets a scheduled run through.
    result = await mgr.async_run_ml_training(force=False)
    assert result == {"ok": False, "reason": f"need {ML_TRAINING_MIN_CYCLES} cycles, have 0"}

    # The schedule runs at the fixed hour, not the stored one.
    with patch(
        "custom_components.ha_washdata.manager.evt.async_track_time_change",
        return_value=MagicMock(),
    ) as track:
        mgr._setup_ml_training_scheduler()  # noqa: SLF001
    assert track.call_args.kwargs["hour"] == ML_TRAINING_HOUR == 2
    await mgr.async_shutdown()


async def test_feedback_records_stay_readable(hass: HomeAssistant) -> None:
    """New requests no longer carry `is_close_match` (its tolerance was the retired
    duration_tolerance); a stored one that still does is served as before."""
    mgr = await boot(hass, make_entry(hass))
    store = mgr.profile_store
    store.add_pending_feedback("old", {
        "cycle_id": "old", "detected_profile": "Cotton", "confidence": 0.7,
        "estimated_duration": 3600.0, "actual_duration": 3500.0,
        "duration_match_pct": 97.2, "is_close_match": True,
    })
    mgr.learning_manager.request_cycle_verification("new", "Cotton", 0.7, 3600.0, 3500.0)

    pending = store.get_pending_feedback()
    assert pending["old"]["is_close_match"] is True
    assert "is_close_match" not in pending["new"]
    assert pending["new"]["duration_match_pct"] == pytest.approx(3500 / 3600 * 100)
    await mgr.async_shutdown()


async def test_get_options_hides_a_stray_retired_key(hass: HomeAssistant) -> None:
    """The panel's cadence rules read the resolved sampling default; a stale stored
    sampling_interval (a 3.12 entry from before the trim) must not override it."""
    entry = MockConfigEntry(
        domain=DOMAIN, title="Washer",
        data={CONF_POWER_SENSOR: "sensor.power", CONF_DEVICE_TYPE: "dryer"},
        options={CONF_OFF_DELAY: 120, **OLD_VALUES},
        version=3, minor_version=CONFIG_ENTRY_MINOR_VERSION,
    )
    entry.add_to_hass(hass)
    sent: dict[str, Any] = {}
    with patch.object(ws_api, "_send_result", side_effect=lambda _c, _i, _n, payload: sent.update(payload)):
        fn = getattr(ws_api.ws_get_options, "__wrapped__", ws_api.ws_get_options)
        fn(hass, MagicMock(), {"id": 1, "entry_id": entry.entry_id})

    assert not set(sent["options"]) & RETIRED_OPTION_KEYS
    assert sent["options"][CONF_OFF_DELAY] == 120
    assert sent["defaults"]["sampling_interval"] == 30.0  # the dryer's cadence
