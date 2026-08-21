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
"""WS glue for the historical power-data import (issue #344).

Ingest is staged and chunked because Home Assistant builds its WebSocket with aiohttp's
default 4 MiB frame cap and ten days of 5-second data is 5-8 MB of text - and an
over-cap frame is not rejected, it closes the connection. The scan and the apply run as
detached registry tasks; the scan's full traces stay server-side for the same reason.
"""
import asyncio
import inspect
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata import task_registry, ws_api
from custom_components.ha_washdata.const import HISTORY_IMPORT_MAX_BYTES
from custom_components.ha_washdata.cycle_detector import CycleDetectorConfig
from custom_components.ha_washdata.profile_store import ProfileStore

ENTITY = "sensor.washer_power"
T0 = datetime(2026, 5, 1, 8, 0, tzinfo=timezone.utc)


def _conn():
    c = MagicMock()
    c.send_result = MagicMock()
    c.send_error = MagicMock()
    return c


def _hass():
    hass = MagicMock()
    hass.data = {}

    async def _exec(func, *args, **kwargs):
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)

    hass.async_add_executor_job = AsyncMock(side_effect=_exec)
    hass.async_create_task = lambda coro, *a: asyncio.create_task(coro)
    return hass


def _store(hass):
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(hass, "e", min_duration_ratio=0.0, max_duration_ratio=3.0)
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
    return ps


def _manager(hass):
    m = MagicMock()
    m.profile_store = _store(hass)
    m.power_sensor_entity_id = ENTITY
    m.notify_update = MagicMock()
    m.detector = SimpleNamespace(
        config=CycleDetectorConfig(
            min_power=2.0,
            off_delay=300,
            device_type="washing_machine",
            completion_min_seconds=600,
            min_off_gap=480,
            start_energy_threshold=0.2,
        )
    )
    m._resolve_energy_price = MagicMock(return_value=None)
    return m


def _entry():
    return SimpleNamespace(
        entry_id="e",
        options={"device_type": "washing_machine", "sampling_interval": 5.0},
        data={},
    )


def _csv(cycles: int = 2, idle_s: float = 1500.0) -> str:
    """Two 40-minute cycles separated by an idle stretch, as an HA history export."""
    rows = ["entity_id,state,last_changed"]
    elapsed = 0.0
    for index in range(cycles):
        for i in range(480):
            elapsed += 5.0
            watts = 1800 if i < 120 else 400
            rows.append(f"{ENTITY},{watts},{(T0 + timedelta(seconds=elapsed)).isoformat()}")
        elapsed += 5.0
        rows.append(f"{ENTITY},0,{(T0 + timedelta(seconds=elapsed)).isoformat()}")
        if index < cycles - 1:
            elapsed += idle_s
            rows.append(f"{ENTITY},0,{(T0 + timedelta(seconds=elapsed)).isoformat()}")
    return "\n".join(rows)


async def _upload(hass, conn, manager, text, chunks=3):
    """Run the begin + chunk handshake, returning the staging token."""
    with patch.object(ws_api, "_get_manager", return_value=manager):
        ws_api.ws_history_import_begin(hass, conn, {"id": 1, "entry_id": "e"})
        token = conn.send_result.call_args.args[1]["token"]
        size = len(text) // chunks + 1
        for seq, start in enumerate(range(0, len(text), size)):
            ws_api.ws_history_import_chunk(hass, conn, {
                "id": 2 + seq, "entry_id": "e", "token": token,
                "seq": seq, "text": text[start:start + size],
            })
    return token


async def _scan(hass, conn, manager, entry, token):
    reg = task_registry.get_registry(hass)
    with patch.object(ws_api, "_get_manager", return_value=manager), \
         patch.object(ws_api, "_get_entry", return_value=entry):
        ws_api.ws_start_history_import_scan(
            hass, conn, {"id": 9, "entry_id": "e", "token": token}
        )
        task_id = conn.send_result.call_args.args[1]["task_id"]
        await asyncio.sleep(0)
        for _ in range(200):
            if reg.get(task_id).state != task_registry.STATE_RUNNING:
                break
            await asyncio.sleep(0)
    return reg.get(task_id)


# ─── Ingest ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_begin_then_chunks_stage_the_text():
    hass, conn = _hass(), _conn()
    manager = _manager(hass)
    text = _csv()
    token = await _upload(hass, conn, manager, text)
    slot = ws_api._history_staging(hass)["e"]
    assert slot["token"] == token
    assert "".join(slot["chunks"]) == text
    assert conn.send_result.call_args.args[1]["received_bytes"] == len(text.encode())
    assert conn.send_error.call_count == 0


@pytest.mark.asyncio
async def test_out_of_order_chunk_is_rejected():
    """A spliced file would import silently corrupt cycles, so sequence is enforced."""
    hass, conn = _hass(), _conn()
    manager = _manager(hass)
    with patch.object(ws_api, "_get_manager", return_value=manager):
        ws_api.ws_history_import_begin(hass, conn, {"id": 1, "entry_id": "e"})
        token = conn.send_result.call_args.args[1]["token"]
        ws_api.ws_history_import_chunk(hass, conn, {
            "id": 2, "entry_id": "e", "token": token, "seq": 3, "text": "x",
        })
    assert conn.send_error.call_args.args[1] == "invalid_format"


@pytest.mark.asyncio
async def test_a_stale_token_cannot_append():
    hass, conn = _hass(), _conn()
    manager = _manager(hass)
    with patch.object(ws_api, "_get_manager", return_value=manager):
        ws_api.ws_history_import_begin(hass, conn, {"id": 1, "entry_id": "e"})
        # A second begin replaces the slot: one upload per entry, so an abandoned
        # multi-megabyte paste cannot accumulate.
        ws_api.ws_history_import_begin(hass, conn, {"id": 2, "entry_id": "e"})
        ws_api.ws_history_import_chunk(hass, conn, {
            "id": 3, "entry_id": "e", "token": "stale", "seq": 0, "text": "x",
        })
    assert conn.send_error.call_args.args[1] == "not_found"


@pytest.mark.asyncio
async def test_oversized_upload_is_refused_and_the_slot_dropped():
    hass, conn = _hass(), _conn()
    manager = _manager(hass)
    with patch.object(ws_api, "_get_manager", return_value=manager), \
         patch.object(ws_api, "HISTORY_IMPORT_MAX_BYTES", 32):
        ws_api.ws_history_import_begin(hass, conn, {"id": 1, "entry_id": "e"})
        token = conn.send_result.call_args.args[1]["token"]
        ws_api.ws_history_import_chunk(hass, conn, {
            "id": 2, "entry_id": "e", "token": token, "seq": 0, "text": "y" * 64,
        })
    assert conn.send_error.call_args.args[1] == "invalid_format"
    assert "e" not in ws_api._history_staging(hass)


@pytest.mark.asyncio
async def test_recorder_ingest_uses_the_devices_own_entity_in_day_windows():
    """The entity is never client-supplied: this command reads arbitrary history."""
    hass, conn = _hass(), _conn()
    manager = _manager(hass)
    calls: list[tuple] = []

    async def _fake_recorder(_hass, entity_id, start_dt, *, end_dt=None):
        calls.append((entity_id, start_dt, end_dt))
        return [(start_dt.timestamp(), 100.0)]

    with patch.object(ws_api, "_get_manager", return_value=manager), \
         patch.object(ws_api, "_recorder_power", side_effect=_fake_recorder):
        await ws_api.ws_history_import_recorder.__wrapped__(
            hass, conn, {"id": 1, "entry_id": "e", "days": 3}
        )
    payload = conn.send_result.call_args.args[1]
    assert payload["entity_id"] == ENTITY
    assert payload["rows"] == 3
    assert len(calls) == 3                       # one bounded query per day
    assert all(c[0] == ENTITY for c in calls)
    assert all(c[2] is not None for c in calls)   # every query has an end bound
    slot = ws_api._history_staging(hass)["e"]
    assert slot["source"] == "recorder"


# ─── Scan ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scan_finds_candidates_and_keeps_traces_server_side():
    hass, conn = _hass(), _conn()
    manager, entry = _manager(hass), _entry()
    token = await _upload(hass, conn, manager, _csv())
    task = await _scan(hass, conn, manager, entry, token)

    assert task.state == task_registry.STATE_DONE
    result = task.result
    assert [round(s["duration_s"] / 60.0, 1) for s in result["segments"]] == [39.9, 39.9]
    assert all(s["accept"] for s in result["segments"])
    assert result["settings"]["device_type"] == "washing_machine"
    # The preview travels; the traces do not - `get_task_result` ships the result
    # verbatim and a few hundred traces would exceed the 4 MiB frame cap.
    assert "cycles" not in result
    assert all("power_data" not in s for s in result["segments"])
    staged = ws_api._history_staging(hass)["e"]
    assert len(staged["cycles"]) == 2
    assert staged["cycles"][0]["power_data"]
    # The raw upload text is released once it has been parsed.
    assert "chunks" not in staged


@pytest.mark.asyncio
async def test_scan_without_a_staged_upload_errors():
    hass, conn = _hass(), _conn()
    manager, entry = _manager(hass), _entry()
    with patch.object(ws_api, "_get_manager", return_value=manager), \
         patch.object(ws_api, "_get_entry", return_value=entry):
        ws_api.ws_start_history_import_scan(
            hass, conn, {"id": 1, "entry_id": "e", "token": "nope"}
        )
    assert conn.send_error.call_args.args[1] == "not_found"


@pytest.mark.asyncio
async def test_a_stream_with_nothing_usable_is_a_result_not_a_failure():
    """Six months of hourly averages is the common case; the user gets an explanation."""
    hass, conn = _hass(), _conn()
    manager, entry = _manager(hass), _entry()
    hourly = ["entity_id,state,last_changed"] + [
        f"{ENTITY},{300 if h % 3 else 0},{(T0 + timedelta(hours=h)).isoformat()}"
        for h in range(48)
    ]
    token = await _upload(hass, conn, manager, "\n".join(hourly), chunks=1)
    task = await _scan(hass, conn, manager, entry, token)

    assert task.state == task_registry.STATE_DONE
    assert task.result["segments"] == []
    assert task.result["error"] == "no_usable_blocks"
    assert task.result["skipped"]


# ─── Apply ────────────────────────────────────────────────────────────────────


async def _apply(hass, conn, manager, scan_task_id, accept):
    reg = task_registry.get_registry(hass)
    with patch.object(ws_api, "_get_manager", return_value=manager):
        ws_api.ws_apply_history_import(hass, conn, {
            "id": 20, "entry_id": "e", "scan_task_id": scan_task_id, "accept": accept,
        })
        task_id = conn.send_result.call_args.args[1]["task_id"]
        for _ in range(200):
            if reg.get(task_id).state != task_registry.STATE_RUNNING:
                break
            await asyncio.sleep(0)
    return reg.get(task_id)


@pytest.mark.asyncio
async def test_apply_persists_only_the_accepted_candidates():
    hass, conn = _hass(), _conn()
    manager, entry = _manager(hass), _entry()
    token = await _upload(hass, conn, manager, _csv())
    scan = await _scan(hass, conn, manager, entry, token)

    applied = await _apply(hass, conn, manager, scan.id, [1])

    assert applied.state == task_registry.STATE_DONE
    assert applied.result["imported"] == 1
    store = manager.profile_store
    backfilled = store.get_backfill_cycles()
    assert len(backfilled) == 1
    assert store.get_past_cycles() == []
    assert store.get_reference_cycles() == []
    cycle = backfilled[0]
    assert cycle["meta"]["source"] == "history_import"
    assert cycle["profile_name"] is None
    assert cycle["power_data"]
    # Never golden: that flag marks a curated recording, unlocks sharing to the
    # community store and stamps a star. Nothing has verified this cycle.
    assert not (cycle.get("ml_review") or {}).get("golden")
    assert store.get_lifetime_cycle_count() == 0
    # The staging area is released once the import lands.
    assert "e" not in ws_api._history_staging(hass)


@pytest.mark.asyncio
async def test_re_importing_the_same_history_skips_duplicates():
    """`_add_cycle_data` makes a colliding id unique by suffixing, so without an
    explicit check a second import would store a second copy of every cycle."""
    hass, conn = _hass(), _conn()
    manager, entry = _manager(hass), _entry()
    text = _csv()

    token = await _upload(hass, conn, manager, text)
    scan = await _scan(hass, conn, manager, entry, token)
    first = await _apply(hass, conn, manager, scan.id, [0, 1])
    assert first.result == {**first.result, "imported": 2, "duplicates": 0}

    token = await _upload(hass, conn, manager, text)
    scan = await _scan(hass, conn, manager, entry, token)
    second = await _apply(hass, conn, manager, scan.id, [0, 1])

    assert second.result["imported"] == 0
    assert second.result["duplicates"] == 2
    assert len(manager.profile_store.get_backfill_cycles()) == 2


@pytest.mark.asyncio
async def test_apply_dedupes_repeated_accept_indices():
    """A client that sends the same index twice must not store the candidate twice
    (a candidate with a None dedup_key would slip past the in-loop dedup set)."""
    hass, conn = _hass(), _conn()
    manager, entry = _manager(hass), _entry()
    token = await _upload(hass, conn, manager, _csv())
    scan = await _scan(hass, conn, manager, entry, token)

    applied = await _apply(hass, conn, manager, scan.id, [1, 1, 1])

    assert applied.result["imported"] == 1
    assert len(manager.profile_store.get_backfill_cycles()) == 1


@pytest.mark.asyncio
async def test_apply_rejects_a_foreign_scan_task():
    """`get_task_result` is not entry-scoped, so ownership is checked here."""
    hass, conn = _hass(), _conn()
    manager, entry = _manager(hass), _entry()
    token = await _upload(hass, conn, manager, _csv())
    scan = await _scan(hass, conn, manager, entry, token)
    reg = task_registry.get_registry(hass)
    foreign = reg.create("other_entry", "history_import", "Scanning")

    applied = await _apply(hass, conn, manager, foreign.id, [0])
    assert applied.state == task_registry.STATE_ERROR
    assert applied.error == "scan_expired"
    assert manager.profile_store.get_backfill_cycles() == []

    # And a task of the wrong kind cannot be replayed as a scan either.
    wrong_kind = reg.create("e", "reprocess", "Reprocessing")
    applied = await _apply(hass, conn, manager, wrong_kind.id, [0])
    assert applied.error == "scan_expired"
    assert scan.id != wrong_kind.id


@pytest.mark.asyncio
async def test_apply_after_the_staging_area_is_cleared_reports_expiry():
    hass, conn = _hass(), _conn()
    manager, entry = _manager(hass), _entry()
    token = await _upload(hass, conn, manager, _csv())
    scan = await _scan(hass, conn, manager, entry, token)

    ws_api.async_clear_history_import(hass, "e")  # what an entry reload does

    applied = await _apply(hass, conn, manager, scan.id, [0])
    assert applied.state == task_registry.STATE_ERROR
    assert applied.error == "scan_expired"


@pytest.mark.asyncio
async def test_apply_respects_the_total_cycle_cap():
    hass, conn = _hass(), _conn()
    manager, entry = _manager(hass), _entry()
    token = await _upload(hass, conn, manager, _csv())
    scan = await _scan(hass, conn, manager, entry, token)

    with patch.object(ws_api, "HISTORY_IMPORT_MAX_TOTAL_CYCLES", 1):
        applied = await _apply(hass, conn, manager, scan.id, [0, 1])

    assert applied.result["imported"] == 1
    assert applied.result["capped"] is True
    assert len(manager.profile_store.get_backfill_cycles()) == 1


@pytest.mark.asyncio
async def test_apply_with_nothing_accepted_writes_nothing():
    hass, conn = _hass(), _conn()
    manager, entry = _manager(hass), _entry()
    token = await _upload(hass, conn, manager, _csv())
    scan = await _scan(hass, conn, manager, entry, token)

    applied = await _apply(hass, conn, manager, scan.id, [])
    assert applied.result["imported"] == 0
    assert manager.profile_store.get_backfill_cycles() == []


@pytest.mark.asyncio
async def test_scan_refuses_to_run_without_the_live_detector_config():
    """Segmentation thresholds come from the running detector.

    `_playground_base_config`'s fallback uses the scalar defaults (min_off_gap 60,
    off_delay 180) rather than the per-device ones, which on a dishwasher would cut the
    stream at every drying pause. Refusing is better than scanning with wrong thresholds.
    """
    hass, conn = _hass(), _conn()
    manager, entry = _manager(hass), _entry()
    token = await _upload(hass, conn, manager, _csv())
    manager.detector = None

    task = await _scan(hass, conn, manager, entry, token)
    assert task.state == task_registry.STATE_ERROR
    assert task.error == "detector_unavailable"


def test_the_import_commands_are_admin_and_full_only():
    for command in (
        "history_import_begin", "history_import_chunk", "history_import_recorder",
        "start_history_import_scan", "apply_history_import",
    ):
        assert command in ws_api._FULL_COMMANDS, command
        assert command in ws_api._ADMIN_COMMANDS, command
