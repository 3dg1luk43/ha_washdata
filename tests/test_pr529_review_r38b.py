"""Review of PR #529, round 38b: the Minor findings, each reproduced.

* A stored cadence row far outside what the plug can report (an imported
  ``[1e308, 2]``) sized the operational pass: ``int(p95 * 20)`` overflowed and
  Analyse stopped before the Start Threshold correction, on every retry.
* A profiles-only wholesale import kept the history and the cadence rows but
  still reset the live cadence: the running cycle's intervals were lost.
* A settings change already merged in as another device's copy stayed marked
  when this device's own backup brought the original later: the standby
  advisory kept ignoring it.
* A #531 cut-off left out of a dishwasher programme's length could still be
  picked as its sample when its length sat nearest the new average.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import DOMAIN, SETTINGS_CHANGELOG_IMPORTED_KEY
from custom_components.ha_washdata.learning import recent_cadence
from custom_components.ha_washdata.suggestion_engine import (
    SuggestionEngine,
    stop_threshold_changes,
)

POWER = "sensor.washer_power"
T0 = datetime(2026, 11, 10, 8, tzinfo=timezone.utc)


# ── A stored cadence row outside the collected range is no cadence ──────────


def test_recent_cadence_keeps_only_rows_the_plug_could_report() -> None:
    # The collection bounds themselves survive the store's two-decimal rounding.
    assert recent_cadence([[0.1, 0.1]]) == (0.1, 0.1)
    assert recent_cadence([[1800.0, 1800.0]]) == (1800.0, 1800.0)
    assert recent_cadence([[1800.01, 5.0]]) is None
    assert recent_cadence([[5.0, 0.09]]) is None
    assert recent_cadence([[1e308, 2.0], [4.0, 2.0]]) == (4.0, 2.0)


async def test_an_overflowing_stored_cadence_does_not_stop_analyse(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    entry = await setup_washdata_entry(power_sensor=POWER)
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    store._data["cadence_summaries"] = [[1e308, 2.0]]  # noqa: SLF001
    store._data["cadence_sensor"] = POWER  # noqa: SLF001
    lm = manager.learning_manager

    ran: list[bool] = []
    original = SuggestionEngine.generate_start_threshold_suggestions

    def spy(self: SuggestionEngine, *args: Any, **kwargs: Any) -> Any:
        ran.append(True)
        return original(self, *args, **kwargs)

    with patch.object(SuggestionEngine, "generate_start_threshold_suggestions", spy):
        result = await lm.async_run_full_analysis()
    assert isinstance(result, dict) and "count" in result
    assert ran  # the Start Threshold correction ran
    assert lm.cadence() is None


# ── A profiles-only import keeps the live cadence ────────────────────────────


async def _a_running_cycle(hass: HomeAssistant, setup_washdata_entry: Any) -> tuple[Any, Any]:
    entry = await setup_washdata_entry(power_sensor=POWER)
    manager = hass.data[DOMAIN][entry.entry_id]
    store = manager.profile_store
    store._data["cadence_summaries"] = [[6.0, 5.0]]  # noqa: SLF001
    store._data["cadence_sensor"] = POWER  # noqa: SLF001
    lm = manager.learning_manager
    now = dt_util.utcnow()
    lm._pending_sensor = POWER  # noqa: SLF001
    lm._pending_intervals.extend((5.0, now) for _ in range(50))  # noqa: SLF001
    return entry, manager


async def _ws_import(hass: HomeAssistant, entry: Any, payload: dict[str, Any]) -> None:
    connection = MagicMock()
    await ws_api.ws_import_config.__wrapped__(hass, connection, {
        "id": 1, "entry_id": entry.entry_id, "json_data": json.dumps(payload),
    })
    await hass.async_block_till_done()
    connection.send_error.assert_not_called()


async def test_a_profiles_only_import_keeps_the_running_cycle_s_intervals(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    entry, manager = await _a_running_cycle(hass, setup_washdata_entry)
    await _ws_import(hass, entry, {"version": 2, "data": {
        "profiles": {"Cotton": {"avg_duration": 3600.0}},
    }})
    assert manager.profile_store.get_profiles().keys() == {"Cotton"}
    lm = manager.learning_manager
    assert len(lm._pending_intervals) == 50  # noqa: SLF001
    assert lm.cadence() == (6.0, 5.0)


async def test_an_import_that_replaces_the_history_still_resets_it(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    entry, manager = await _a_running_cycle(hass, setup_washdata_entry)
    await _ws_import(hass, entry, {"version": 2, "data": {
        "profiles": {"Cotton": {"avg_duration": 3600.0}}, "past_cycles": [],
    }})
    assert len(manager.learning_manager._pending_intervals) == 0  # noqa: SLF001


# ── The local copy of a settings change wins over an imported one ────────────


async def test_this_device_s_own_backup_unmarks_an_imported_copy(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """The change came back from another device first (marked), then from this
    device's own backup: the local row is kept, and the advisory reads it."""
    entry = await setup_washdata_entry()
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    own = {"key": "stop_threshold_w", "old": 2.0, "new": 5.0,
           "timestamp": "2026-11-01T10:00:00+00:00"}
    store._data["settings_changelog"] = [  # noqa: SLF001
        {**own, SETTINGS_CHANGELOG_IMPORTED_KEY: True},
    ]
    payload = {
        "version": 19,
        "entry_id": entry.entry_id,
        "device_fingerprint": {"device_type": "washing_machine"},
        "data": {"settings_changelog": [dict(own)]},
        "entry_data": {},
        "entry_options": {},
    }
    await store.async_import_data_selective(
        payload, selection={"categories": ["history_logs"]},
        local_device_type="washing_machine",
    )
    log = store.get_settings_changelog()
    assert len(log) == 1
    assert SETTINGS_CHANGELOG_IMPORTED_KEY not in log[0]
    assert len(stop_threshold_changes(log)) == 1


# ── A cut-off run is not the programme's sample ──────────────────────────────


def _dishwasher_run(total_s: float, *, quiet_from: float, beep_at: float) -> list[list[float]]:
    """10 s readings: washing at 1950 W, then 3.3 W quiet, a 6.3 W beep at
    ``beep_at`` (its terminal event), 3.3 W to ``total_s``."""
    pts: list[list[float]] = []
    t = 0.0
    while t <= total_s:
        if t < quiet_from:
            p = 1950.0 if (t // 300) % 2 == 0 else 30.0
        elif beep_at <= t < beep_at + 40:
            p = 6.3
        else:
            p = 3.3
        pts.append([t, p])
        t += 10.0
    return pts


async def test_a_cut_off_run_is_not_picked_as_the_sample(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """Three 4200 s runs dry from 2400 s and beep at 4100 s, so a run closed at
    3000 s inside that drying is a cut-off. Two complete 1200 s runs (they beep
    at their own end) bring the kept average to exactly 3000 s, the cut-off's
    length. Through the shipped rebuild."""
    entry = await setup_washdata_entry(device_type="dishwasher", options={"stop_threshold_w": 4.5})
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    full = _dishwasher_run(4200, quiet_from=2400, beep_at=4100)
    cut = [p for p in full if p[0] <= 3000]
    short = _dishwasher_run(1200, quiet_from=400, beep_at=1100)
    runs = {"full0": full, "full1": full, "full2": full, "cut": cut, "short0": short, "short1": short}
    store._data["profiles"]["ECO"] = {}  # noqa: SLF001
    store._data["past_cycles"] = [  # noqa: SLF001
        {"id": cid, "profile_name": "ECO", "status": "completed", "termination_reason": "smart",
         "start_time": (T0 + timedelta(days=i)).isoformat(), "duration": pts[-1][0],
         "power_data": [list(p) for p in pts]}
        for i, (cid, pts) in enumerate(runs.items())
    ]
    assert await store.async_rebuild_envelope("ECO")
    profile = store.get_profiles()["ECO"]
    # The cut-off left the length: 3 x 4200 + 2 x 1200 over five.
    assert abs(profile["avg_duration"] - 3000.0) < 1
    assert store._data["envelopes"]["ECO"]["shape_cycle_count"] == 5  # noqa: SLF001
    assert profile["sample_cycle_id"] in {"full0", "full1", "full2", "short0", "short1"}
