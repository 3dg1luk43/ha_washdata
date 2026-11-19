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
"""PR #529 review: what the suggestion passes report.

* The Suggested Settings sensor does not poll. A background pass (a cycle end's,
  the batch) or a refresh (startup, an options change) that changed the stored
  suggestions sent no entity update, so on an idle appliance the sensor kept the
  old count until some unrelated update.
* The manual "analyse now" reported success with the count of a partial run when
  one of its passes raised, or when an options refresh or a replaced store
  superseded it and discarded what its passes computed.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.dispatcher import async_dispatcher_connect

from custom_components.ha_washdata.const import CONF_OFF_DELAY, DOMAIN, SIGNAL_WASHER_UPDATE
from custom_components.ha_washdata.learning import MANUAL_ANALYSIS_ATTEMPTS
from custom_components.ha_washdata.suggestion_engine import OWNER_MODEL, SuggestionEngine


async def _boot(hass: HomeAssistant, setup_washdata_entry: Any) -> tuple[Any, Any, list[int]]:
    entry = await setup_washdata_entry(options={CONF_OFF_DELAY: 120})
    await hass.async_block_till_done()
    manager = hass.data[DOMAIN][entry.entry_id]
    pushes: list[int] = []
    async_dispatcher_connect(
        hass, SIGNAL_WASHER_UPDATE.format(entry.entry_id), lambda: pushes.append(1)
    )
    return entry, manager, pushes


def _sensor_state(hass: HomeAssistant, entry: Any) -> str | None:
    entity_id = er.async_get(hass).async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_suggestions"
    )
    state = hass.states.get(entity_id) if entity_id else None
    return state.state if state else None


def _shown(manager: Any) -> str:
    """What the sensor would show if it were refreshed now."""
    from custom_components.ha_washdata.ws_api import _visible_suggestions  # noqa: PLC0415

    entry = manager.config_entry
    rows = _visible_suggestions(
        manager.profile_store, {**entry.data, **entry.options}, manager.device_type
    )
    return str(len(rows))


async def test_a_background_pass_that_changes_the_suggestions_pushes_the_sensor(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    entry, manager, pushes = await _boot(hass, setup_washdata_entry)
    lm = manager.learning_manager
    assert not manager.profile_store.get_suggestions()
    assert _sensor_state(hass, entry) == "0"

    def propose() -> dict[str, Any]:
        return {CONF_OFF_DELAY: {"value": 600, "reason": "r"}}

    await lm._run_pass(propose, OWNER_MODEL, lm._config_generation)  # noqa: SLF001
    await hass.async_block_till_done()
    assert CONF_OFF_DELAY in manager.profile_store.get_suggestions()
    assert pushes
    assert _shown(manager) != "0"
    assert _sensor_state(hass, entry) == _shown(manager)

    # The same result again changes nothing and pushes nothing.
    pushes.clear()
    await lm._run_pass(propose, OWNER_MODEL, lm._config_generation)  # noqa: SLF001
    await hass.async_block_till_done()
    assert pushes == []

    # The pass withdrawing what it owned is a change too.
    await lm._run_pass(dict, OWNER_MODEL, lm._config_generation)  # noqa: SLF001
    await hass.async_block_till_done()
    assert CONF_OFF_DELAY not in manager.profile_store.get_suggestions()
    assert pushes
    assert _sensor_state(hass, entry) == _shown(manager)


async def test_a_refresh_pushes_the_sensor_once_it_ends(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """Startup and an options change: what the sensor shows depends on the options
    too, so the refresh pushes whatever it changed."""
    _entry, manager, pushes = await _boot(hass, setup_washdata_entry)
    await manager.learning_manager.async_refresh_suggestions()
    await hass.async_block_till_done()
    assert pushes


async def test_no_push_after_an_unload(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    _entry, manager, pushes = await _boot(hass, setup_washdata_entry)
    lm = manager.learning_manager
    manager._is_shutdown = True  # noqa: SLF001
    try:
        await lm._run_pass(  # noqa: SLF001
            lambda: {CONF_OFF_DELAY: {"value": 600, "reason": "r"}},
            OWNER_MODEL, lm._config_generation,  # noqa: SLF001
        )
        await lm.async_refresh_suggestions()
        await hass.async_block_till_done()
    finally:
        manager._is_shutdown = False  # noqa: SLF001
    assert pushes == []


@pytest.mark.parametrize(
    "failing", ["generate_standby_floor_suggestions", "generate_detection_suggestions"]
)
async def test_a_manual_analysis_with_a_failed_pass_is_no_success(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any, failing: str
) -> None:
    """The passes after the failed one never ran: the panel gets the existing
    error reply ("Analysis failed: ..."), not a count of the partial run. The
    background refresh still never raises."""
    entry, manager, _pushes = await _boot(hass, setup_washdata_entry)
    client = await hass_ws_client(hass)
    with patch.object(SuggestionEngine, failing, side_effect=RuntimeError("scan broke")):
        await client.send_json_auto_id(
            {"type": f"{DOMAIN}/run_suggestion_analysis", "entry_id": entry.entry_id}
        )
        res = await client.receive_json()
        assert res["success"] is False
        assert res["error"]["code"] == "unknown_error"
        assert "scan broke" in res["error"]["message"]
        await manager.learning_manager.async_refresh_suggestions()  # never raises
    await client.send_json_auto_id(
        {"type": f"{DOMAIN}/run_suggestion_analysis", "entry_id": entry.entry_id}
    )
    ok = await client.receive_json()
    assert ok["success"] is True and "count" in ok["result"]


def _count_manual_runs(lm: Any, before_run: Any = None) -> tuple[list[int], Any]:
    """Wrap ``_async_run_all_passes``: count the manual analysis's runs (the
    refresh calls it too, without ``raise_errors``)."""
    runs: list[int] = []
    orig = lm._async_run_all_passes  # noqa: SLF001

    async def run(*args: Any, **kwargs: Any) -> None:
        if kwargs.get("raise_errors"):
            runs.append(1)
            if before_run is not None:
                before_run(len(runs))
        await orig(*args, **kwargs)

    return runs, patch.object(lm, "_async_run_all_passes", side_effect=run)


async def test_a_manual_analysis_superseded_by_an_options_save_runs_again(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """An options save while Analyze now runs starts a refresh: the manual run's
    later passes are discarded (other options), so it runs again on the new
    options before it reports a count."""
    entry, manager, _pushes = await _boot(hass, setup_washdata_entry)
    lm = manager.learning_manager
    passes = {"n": 0}
    orig_pass = lm._run_pass  # noqa: SLF001

    async def run_pass(*args: Any, **kwargs: Any) -> Any:
        passes["n"] += 1
        if passes["n"] == 2:  # the manual run's second pass: the user saves options
            hass.config_entries.async_update_entry(
                entry, options={**entry.options, CONF_OFF_DELAY: 150}
            )
            await hass.async_block_till_done()  # the refresh ran on the new options
        return await orig_pass(*args, **kwargs)

    generation = lm._config_generation  # noqa: SLF001
    runs, wrapped = _count_manual_runs(lm)
    with wrapped, patch.object(lm, "_run_pass", side_effect=run_pass):
        res = await lm.async_run_full_analysis()
    assert lm._config_generation > generation  # noqa: SLF001  (superseded once)
    assert len(runs) == 2
    assert res == {"count": len(manager.profile_store.get_suggestions() or {})}


@pytest.mark.parametrize("supersede", ["options", "store"])
async def test_a_manual_analysis_superseded_on_every_run_is_no_success(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any, supersede: str
) -> None:
    """Superseded on every run (options refreshed, or the store replaced by an
    import or undo): the panel gets ``analysis_superseded``, not a count."""
    entry, manager, _pushes = await _boot(hass, setup_washdata_entry)
    lm = manager.learning_manager
    store = manager.profile_store

    def bump(_n: int) -> None:
        if supersede == "options":
            lm._config_generation += 1  # noqa: SLF001  (a refresh started)
        else:
            store._data_generation += 1  # noqa: SLF001  (an import replaced it)

    client = await hass_ws_client(hass)
    runs, wrapped = _count_manual_runs(lm, bump)
    with wrapped:
        await client.send_json_auto_id(
            {"type": f"{DOMAIN}/run_suggestion_analysis", "entry_id": entry.entry_id}
        )
        res = await client.receive_json()
    assert res["success"] is False
    assert res["error"]["code"] == "analysis_superseded"
    assert len(runs) == MANUAL_ANALYSIS_ATTEMPTS


async def test_a_manual_analysis_whose_joint_check_fails_is_no_success(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """A pass stored an end-gate value, then the joint check of the stored set
    could not run. Non-strict, that read as "nothing to withdraw" and the panel
    got a count; Apply all checks the same set strictly and can refuse it. The
    manual analysis checks strictly too and replies ``analysis_check_failed``;
    what the pass stored stays. A background refresh still never raises, and an
    empty drop list from a check that ran is a success."""
    entry, manager, _pushes = await _boot(hass, setup_washdata_entry)
    lm = manager.learning_manager
    seen: list[bool] = []
    broken = {"on": True}

    def joint(_self: Any, _stored: Any, *, strict: bool = False) -> list[str]:
        seen.append(strict)
        if broken["on"] and strict:
            raise RuntimeError("scan broke")
        return []

    client = await hass_ws_client(hass)
    with patch.object(
        SuggestionEngine, "generate_model_suggestions",
        lambda _self: {CONF_OFF_DELAY: {"value": 600, "reason": "r"}},
    ), patch.object(SuggestionEngine, "joint_end_gate_drops", joint):
        await client.send_json_auto_id(
            {"type": f"{DOMAIN}/run_suggestion_analysis", "entry_id": entry.entry_id}
        )
        res = await client.receive_json()
        assert res["success"] is False
        assert res["error"]["code"] == "analysis_check_failed"
        assert seen and all(seen)
        assert CONF_OFF_DELAY in (manager.profile_store.get_suggestions() or {})
        seen.clear()
        await lm.async_refresh_suggestions()  # never raises
        assert seen and not any(seen)
        broken["on"] = False
        await client.send_json_auto_id(
            {"type": f"{DOMAIN}/run_suggestion_analysis", "entry_id": entry.entry_id}
        )
        ok = await client.receive_json()
    assert ok["success"] is True and ok["result"]["count"] >= 1


@pytest.mark.parametrize("supersede", ["options", "store"])
@pytest.mark.parametrize("failing", ["pass", "joint_check"])
async def test_a_manual_analysis_that_failed_after_it_was_superseded_runs_again(
    hass: HomeAssistant, setup_washdata_entry: Any, hass_ws_client: Any,
    supersede: str, failing: str,
) -> None:
    """PR #529 review: a pass that raised on options or a store replaced while it
    ran answered ``unknown_error`` (or ``analysis_check_failed``), and nothing
    analysed the replacement. Superseded, it runs again like a run that returned;
    with nothing superseded the error still reaches the panel (tests above)."""
    entry, manager, _pushes = await _boot(hass, setup_washdata_entry)
    lm, store = manager.learning_manager, manager.profile_store
    failed: list[int] = []

    def supersede_and_fail() -> None:
        failed.append(1)
        if supersede == "options":
            lm._config_generation += 1  # noqa: SLF001  (a refresh started)
        else:
            store._data_generation += 1  # noqa: SLF001  (Undo Import replaced it)
        raise RuntimeError("read a store being replaced")

    def model(_self: Any) -> dict[str, Any]:
        if failing == "pass" and not failed:
            supersede_and_fail()
        return {CONF_OFF_DELAY: {"value": 600, "reason": "r"}}

    def joint(_self: Any, _stored: Any, *, strict: bool = False) -> list[str]:
        if failing == "joint_check" and strict and not failed:
            supersede_and_fail()
        return []

    client = await hass_ws_client(hass)
    runs, wrapped = _count_manual_runs(lm)
    with wrapped, patch.object(SuggestionEngine, "generate_model_suggestions", model), \
            patch.object(SuggestionEngine, "joint_end_gate_drops", joint):
        await client.send_json_auto_id(
            {"type": f"{DOMAIN}/run_suggestion_analysis", "entry_id": entry.entry_id}
        )
        res = await client.receive_json()
    assert failed == [1]
    assert res["success"] is True, res
    assert len(runs) == 2
    assert res["result"]["count"] == len(store.get_suggestions() or {})
    assert CONF_OFF_DELAY in (store.get_suggestions() or {})
