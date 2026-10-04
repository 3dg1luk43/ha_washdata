# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Audit STORE-21: imported program cards say how often the matcher used them.

Counted locally from the appliance's own recorded cycles: only labels the matcher
set on its own (``_AUTO_LABEL_SOURCES``). A label the user set or corrected is the
user's decision, and reference / backfill cycles are not this appliance's
observed runs. No store write, no prompt.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import DOMAIN
from custom_components.ha_washdata.profile_store import ProfileStore


@pytest.fixture
def store() -> Any:
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(MagicMock(), "entry")
        ps.async_save = AsyncMock()
        yield ps


def _c(i: int, name: str | None, source: str | None) -> dict[str, Any]:
    return {"id": f"c{i}", "start_time": f"2026-05-0{i % 9 + 1}T08:00:00+00:00",
            "duration": 3600.0, "status": "completed", "profile_name": name,
            "label_source": source}


def _seed(store: Any) -> None:
    store._data["profiles"] = {"Imported Eco": {"avg_duration": 3600}, "Mine": {"avg_duration": 3000}}
    store._data["past_cycles"] = [
        _c(1, "Imported Eco", "auto_match"),
        _c(2, "Imported Eco", "auto_label_service"),
        _c(3, "Imported Eco", "auto_label_post"),
        _c(4, "Imported Eco", "manual"),        # the user's decision
        _c(5, "Imported Eco", None),            # provenance unknown
        _c(6, "Mine", "auto_match"),
        _c(7, None, "auto_match"),
    ]
    store._data["reference_cycles"] = [
        {**_c(8, "Imported Eco", "auto_match"), "source": "store"},
    ]
    store._data["backfill_cycles"] = [_c(9, "Imported Eco", "auto_label_backfill")]


def test_counts_only_matcher_labels_on_observed_cycles(store: Any) -> None:
    _seed(store)
    assert store.matcher_label_counts() == {"Imported Eco": 3, "Mine": 1}


def test_an_empty_store_counts_nothing(store: Any) -> None:
    assert store.matcher_label_counts() == {}


async def test_get_profiles_serves_the_counts_within_its_contract(hass: Any, store: Any) -> None:
    _seed(store)
    hass.data[DOMAIN] = {"e1": SimpleNamespace(profile_store=store)}
    connection = MagicMock()
    ws_api.ws_get_profiles(hass, connection, {"id": 1, "type": "ha_washdata/get_profiles", "entry_id": "e1"})
    await hass.async_block_till_done(wait_background_tasks=True)
    payload = connection.send_result.call_args[0][1]
    assert payload["profile_matcher_counts"] == {"Imported Eco": 3, "Mine": 1}
    assert ws_api._validate_ws_contract("get_profiles", payload) == []
