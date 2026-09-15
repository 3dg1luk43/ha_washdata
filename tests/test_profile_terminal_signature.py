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
"""Per-profile terminal signature: how a programme ends, from its own cycles.

A dishwasher does not stop at its last wash activity. It goes quiet for a passive
drying phase and then usually emits a short, low-power terminal event (the final
pump-out). Measured across the whole `cycle_data/` corpus (146 unique dishwasher
cycles, 10 households) that quiet-plus-event span is a median 11% of the cycle and
reaches 43%, which is why it is cycle content rather than a tail.

This is a pure statistic in the same family as `compute_profile_health` and
`compute_envelope_conformance`: it is reported, never acted on. Register item 238
records the two end-detection rules the same evidence rejected, and the reason is
visible here as `consistency` - the same appliance emits the event in only some of
its runs, so its presence is informative and its absence is not.

Two properties are asserted directly because they are what make the statistic
portable: the detection threshold scales with each cycle's own peak (a fixed
wattage cannot span a 30 W pump-out and a 2000 W wash), and the quiet span is the
quiet PHASE rather than the interval between two samples (measuring the latter
would just report the plug's reporting rate).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from custom_components.ha_washdata.const import TERMINAL_SIGNATURE_MIN_CYCLES
from custom_components.ha_washdata.profile_store import ProfileStore


@pytest.fixture
def store() -> ProfileStore:
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(MagicMock(), "e")
    ps._data["profiles"] = {"Eco": {"avg_duration": 9000.0}}
    return ps


def _cycle(
    cid: str,
    *,
    wash_w: float = 2000.0,
    event_w: float | None = 30.0,
    quiet_s: float = 900.0,
    event_s: float = 60.0,
    step: float = 30.0,
    dry_w: float = 0.4,
) -> dict[str, Any]:
    """A wash block, a quiet drying stretch, then an optional terminal event."""
    points: list[list[float]] = []
    t = 0.0
    for _ in range(20):  # the wash itself
        points.append([t, wash_w])
        t += step
    quiet_end = t + quiet_s
    while t < quiet_end:  # drying: reported, but near zero
        points.append([t, dry_w])
        t += step
    if event_w is not None:
        event_end = t + event_s
        while t < event_end:
            points.append([t, event_w])
            t += step
    points.append([t, 0.0])
    return {
        "id": cid,
        "profile_name": "Eco",
        "status": "completed",
        "duration": t,
        "power_data": points,
    }


def test_the_terminal_event_is_measured(store: ProfileStore) -> None:
    store._data["past_cycles"] = [_cycle(f"c{i}") for i in range(5)]

    sig = store.compute_profile_terminal_signature("Eco")

    assert sig is not None
    assert sig["seen_in"] == 5 and sig["measured"] == 5
    assert sig["consistency"] == 1.0
    assert sig["quiet_before_s"] == pytest.approx(900.0, abs=60.0)
    assert sig["event_watts"] == pytest.approx(30.0)
    # 30 W against a 2000 W wash.
    assert sig["event_watts_frac"] == pytest.approx(0.015, abs=0.002)
    # The event sits at the very end of the cycle.
    assert sig["position_frac"] > 0.9


def test_the_quiet_span_is_the_phase_not_the_sample_gap(store: ProfileStore) -> None:
    """The plug reports right through the drying phase here, every 30 s.

    Measuring the interval before the event would report 30 s - the reporting
    rate - and would read as "no quiet phase at all" on any densely sampled plug.
    """
    store._data["past_cycles"] = [_cycle(f"c{i}", step=30.0) for i in range(4)]

    sig = store.compute_profile_terminal_signature("Eco")

    assert sig is not None
    assert sig["quiet_before_s"] > 600.0, "the drying phase was measured as a sample gap"


def test_intermittency_is_reported_rather_than_hidden(store: ProfileStore) -> None:
    """The finding that stops this being an end signal has to be visible.

    The same appliance emits the event in only some runs, so a consumer must be
    able to see that before trusting the medians.
    """
    store._data["past_cycles"] = [
        _cycle("c1"),
        _cycle("c2"),
        _cycle("c3", event_w=None),
        _cycle("c4", event_w=None),
    ]

    sig = store.compute_profile_terminal_signature("Eco")

    assert sig is not None
    assert sig["seen_in"] == 2 and sig["measured"] == 4
    assert sig["consistency"] == 0.5


def test_a_fixed_wattage_would_miss_a_low_power_appliance(store: ProfileStore) -> None:
    """The threshold scales with the cycle's own peak, which is the whole point.

    The #399 spin extractor looks for a terminal block above `anti_wrinkle_max_power`
    (400 W) and therefore finds nothing on a dishwasher whose pump-out is 33 W.
    Here the entire appliance peaks at 60 W and the event at 4 W.

    Note the drying baseline has to scale down with the appliance too: the
    threshold is a fraction of the cycle peak, so on a 60 W appliance a 0.4 W
    standby would sit ABOVE it and nothing would read as quiet. That bound is
    documented on the method.
    """
    store._data["past_cycles"] = [
        _cycle(f"c{i}", wash_w=60.0, event_w=4.0, dry_w=0.05) for i in range(4)
    ]

    sig = store.compute_profile_terminal_signature("Eco")

    assert sig is not None
    assert sig["seen_in"] == 4
    assert sig["event_watts"] == pytest.approx(4.0)


def test_a_cycle_that_just_stops_reports_no_event(store: ProfileStore) -> None:
    """A washing machine ends AT power: no quiet phase, so no terminal signature."""
    store._data["past_cycles"] = [
        _cycle(f"c{i}", quiet_s=0.0, event_w=None) for i in range(4)
    ]

    sig = store.compute_profile_terminal_signature("Eco")

    assert sig is not None
    assert sig["seen_in"] == 0
    assert sig["quiet_before_s"] is None


def test_too_few_cycles_measures_nothing(store: ProfileStore) -> None:
    """Below the minimum the medians would describe noise, not the programme."""
    store._data["past_cycles"] = [
        _cycle(f"c{i}") for i in range(TERMINAL_SIGNATURE_MIN_CYCLES - 1)
    ]
    assert store.compute_profile_terminal_signature("Eco") is None


def test_it_never_raises(store: ProfileStore) -> None:
    """Same contract as its neighbours: a statistic must not break the panel."""
    store._data["past_cycles"] = [
        {"id": "a", "profile_name": "Eco", "power_data": "garbage"},
        {"id": "b", "profile_name": "Eco", "power_data": [[0.0, 1.0]]},
        {"id": "c", "profile_name": "Eco", "power_data": [["x", "y"], [1, 2]]},
        {"id": "d", "profile_name": "Eco", "power_data": []},
    ]
    result = store.compute_profile_terminal_signature("Eco")
    assert result is None or isinstance(result, dict)
