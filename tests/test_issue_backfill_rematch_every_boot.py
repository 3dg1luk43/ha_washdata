"""The match-confidence backfill must not re-run the matcher on every start.

`async_backfill_match_confidence` runs on every setup and re-matches each labelled
cycle that has no `match_confidence`, stamping one only when the matcher agrees
with the label. A cycle it disagrees with (typically one the user relabelled by
hand) stayed unstamped, so it cost one full match per Home Assistant start,
forever (found by the PERF-08 budget tests, audit F8).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.profile_store import ProfileStore


def _store(cycles: list[dict]) -> ProfileStore:
    hass = MagicMock()
    hass.data = {}
    with patch("custom_components.ha_washdata.profile_store.WashDataStore") as cls:
        st = ProfileStore(hass, "e1")
        st._store = cls.return_value  # noqa: SLF001
        st._store.async_save = AsyncMock()  # noqa: SLF001
    st._data["past_cycles"] = cycles  # noqa: SLF001
    st.async_save = AsyncMock()
    return st


def _cycle(cid: str, label: str) -> dict:
    return {"id": cid, "profile_name": label, "duration": 3600.0,
            "power_data": [[i * 60.0, 500.0] for i in range(20)]}


@pytest.mark.asyncio
async def test_a_disagreeing_cycle_is_matched_once_not_every_start():
    st = _store([_cycle("agree", "Cotton"), _cycle("relabelled", "Wool")])
    st.async_match_profile = AsyncMock(
        return_value=SimpleNamespace(best_profile="Cotton", label_confidence=0.8)
    )
    with patch("custom_components.ha_washdata.profile_store.decompress_power_data",
               side_effect=lambda c: c["power_data"]):
        assert await st.async_backfill_match_confidence() == 1
        assert st.async_match_profile.await_count == 2
        # Second start: nothing left to try.
        await st.async_backfill_match_confidence()
    assert st.async_match_profile.await_count == 2
    by_id = {c["id"]: c for c in st._data["past_cycles"]}  # noqa: SLF001
    assert by_id["agree"]["match_confidence"] == 0.8
    assert by_id["relabelled"].get("match_confidence") is None
