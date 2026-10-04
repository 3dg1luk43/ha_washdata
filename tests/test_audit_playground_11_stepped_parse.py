"""Audit PLAYGROUND-11: history-import parsing was one multi-second executor job.

At the 32 MiB / 500k-row caps ``parse_history_csv`` was 4.4 s of CPU in ONE job
(the #311 freeze pattern on a Pi), the replay ran 4000 readings (~1 s) per job,
and the recorder's ``truncated`` flag never reached the panel. Now the parser is
stepped a slice of rows per job, the replay takes 1000 readings per job, and
both reads put ``truncated`` in the parse report the review step shows.
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from custom_components.ha_washdata import history_import as hi
from custom_components.ha_washdata import task_registry, ws_api
from custom_components.ha_washdata.const import HISTORY_IMPORT_CHUNK_SAMPLES

sys.path.insert(0, str(Path(__file__).parent))
from test_ws_history_import import (  # noqa: E402
    ENTITY, T0, _conn, _csv, _entry, _hass, _manager, _scan, _upload,
)


def _rows(entities: list[str], n: int, *, naive: bool = False) -> str:
    rows = ["entity_id,state,last_changed"]
    for i in range(n):
        t = T0 + timedelta(seconds=5 * (i % 97) + i // 97)  # out of order, with dups
        stamp = t.replace(tzinfo=None).isoformat() if naive else t.isoformat()
        value = "unavailable" if i % 41 == 0 else ("x" if i % 53 == 0 else str(i % 300))
        rows.append(f"{entities[i % len(entities)]},{value},{stamp}")
    return "\n".join(rows)


def _drive(text: str, step: int, **kw):
    parser = hi.HistoryCsvParser(text, **kw)
    jobs = 0
    while not parser.finished:
        parser.step(step)
        jobs += 1
    return parser.result(), jobs


def _same(a, b) -> bool:
    if isinstance(a, dict) or isinstance(b, dict):
        return a == b
    return a.samples == b.samples and a.report() == b.report() and (
        a.entity_id, a.entity_substituted_from
    ) == (b.entity_id, b.entity_substituted_from)


@pytest.mark.parametrize("entities, wanted", [
    ([ENTITY], ENTITY),
    (["sensor.renamed"], ENTITY),                 # one other entity: read in its place
    (["sensor.a", "sensor.b"], ENTITY),           # several others: an error
    ([ENTITY, "sensor.b"], ENTITY),               # filter the wanted one out
    (["", "sensor.renamed"], ENTITY),             # blank entity cells kept, then substituted
    (["sensor.a", "sensor.b"], None),             # no filter at all
])
@pytest.mark.parametrize("step", [1, 7, 500])
def test_stepped_parse_is_the_one_shot_parse(entities, wanted, step):
    text = _rows(entities, 1200)
    one_shot = hi.parse_history_csv(text, entity_id=wanted)
    stepped, jobs = _drive(text, step, entity_id=wanted)
    assert _same(one_shot, stepped)
    assert jobs >= 1200 // step


def test_the_row_cap_cuts_the_stepped_parse_where_it_cut_the_one_shot():
    text = _rows([ENTITY, "sensor.b"], 1000)
    one_shot = hi.parse_history_csv(text, entity_id=ENTITY, max_rows=333)
    stepped, _jobs = _drive(text, 50, entity_id=ENTITY, max_rows=333)
    assert _same(one_shot, stepped)
    assert stepped.truncated and stepped.report()["truncated"] is True


@pytest.mark.asyncio
async def test_the_scan_parses_a_slice_per_executor_job():
    hass, conn = _hass(), _conn()
    manager, entry = _manager(hass), _entry()
    token = await _upload(hass, conn, manager, _csv())
    steps: list[int] = []
    real_step = hi.HistoryCsvParser.step

    def _counted(self, n=hi.PARSE_STEP_ROWS):
        steps.append(n)
        return real_step(self, n)

    replays: list[int] = []
    real_runner_step = hi.ScanRunner.step

    def _counted_runner(self, n=1):
        replays.append(n)
        return real_runner_step(self, n)

    with patch.object(hi, "PARSE_STEP_ROWS", 200), \
         patch.object(hi.HistoryCsvParser, "step", _counted), \
         patch.object(hi.ScanRunner, "step", _counted_runner):
        task = await _scan(hass, conn, manager, entry, token)
    assert task.state == task_registry.STATE_DONE
    # 962 rows at 200 a job: five parse jobs, not one.
    assert len(steps) >= 5 and set(steps) == {200}
    # The replay takes at most 1000 readings per job (was 4000).
    assert HISTORY_IMPORT_CHUNK_SAMPLES <= 1000
    assert replays and max(replays) <= 1000


@pytest.mark.asyncio
async def test_a_recorder_read_cut_by_the_row_cap_says_so_in_the_report():
    hass, conn = _hass(), _conn()
    manager, entry = _manager(hass), _entry()

    async def _fake_recorder(_hass, _entity, start_dt, *, end_dt=None):
        base = start_dt.timestamp()
        return [(base + 5 * i, 1800.0 if i % 400 < 300 else 0.0) for i in range(2000)]

    with patch.object(ws_api, "_get_manager", return_value=manager), \
         patch.object(ws_api, "_recorder_power", side_effect=_fake_recorder), \
         patch.object(ws_api, "HISTORY_IMPORT_MAX_ROWS", 3000):
        await ws_api.ws_history_import_recorder.__wrapped__(
            hass, conn, {"id": 1, "entry_id": "e", "days": 3}
        )
    payload = conn.send_result.call_args.args[1]
    assert payload["truncated"] is True and payload["rows"] == 3000
    task = await _scan(hass, conn, manager, entry, payload["token"])
    parse = task.result["parse"]
    assert parse["truncated"] is True
    assert parse["source"] == "recorder"
    assert parse["rows_total"] == 3000
    # The same report a CSV gets: the span and the peak are there too.
    assert parse["peak_w"] == 1800.0 and parse["first"] and parse["last"]
