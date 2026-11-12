# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A hand-picked programme carries its anti-crease tail (MatchContext element 18).

The manual pin returned 13 elements and ``update_match`` clears element 18 for a
shorter tuple, so a pinned washer whose programme ends in a quiet anti-crease tail
never armed the programme-end finalise (#486) and finalised on the clock instead
(end_gate_eval with every cycle pinned to its label, 311 washers: median lag
8.5 -> 6.5 min, early > 1 min 6 -> 4, splits 5 -> 4).
"""

from __future__ import annotations

from datetime import timedelta

from homeassistant.util import dt as dt_util

from custom_components.ha_washdata import progress as progress_mod
from custom_components.ha_washdata.const import DOMAIN


async def test_a_pinned_programme_supplies_its_anticrease_tail(hass, setup_washdata_entry) -> None:
    entry = await setup_washdata_entry()
    mgr = hass.data[DOMAIN][entry.entry_id]
    store = mgr.profile_store
    store._data["profiles"]["Cotton"] = {"avg_duration": 7200.0}  # noqa: SLF001
    store._data.setdefault("envelopes", {})["Cotton"] = {  # noqa: SLF001
        "avg": [[0.0, 500.0], [7200.0, 5.0]],
        "tail_ref": {"n": 4, "with_tail": 4, "median_s": 900.0},
    }
    mgr._manual_program_active = True  # noqa: SLF001
    mgr._current_program = "Cotton"  # noqa: SLF001
    now = dt_util.utcnow()
    readings = [(now - timedelta(seconds=600), 500.0), (now, 480.0)]

    result = mgr.detector._profile_matcher(readings)  # noqa: SLF001

    want = progress_mod.anticrease_programme_tail(store, "Cotton", "washing_machine")
    assert want == 900.0
    assert len(result) == 18 and result[17] == want
    mgr.detector.update_match(result)
    assert mgr.detector._matched_programme_tail_s == want  # noqa: SLF001
