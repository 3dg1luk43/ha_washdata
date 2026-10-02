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
"""#424 after 0.5.7: two dishwashers, two different causes.

1. **The Beko stored 10 min of standby in every cycle.** `_keep_tail_cap` withholds
   the measured drying allowance when the run already went through its drying
   phase, but it asked that at `stop_threshold_w` (0.96 W) while the allowance is
   measured at 0.4% of the cycle's peak (7.9 W). The Beko ends 15-20 W drain ->
   1.3 W -> 0.3 W, so at 0.96 W its last activity is the 1.3 W wind-down sample,
   preceded by the drain: no quiet found, 611 s banked. The v13 repair used the
   same test, and skipped the dishwasher's timeout finishes altogether.
2. **The second reporter's dishwasher never reached Smart Termination.** A three-
   hour programme sharing the first hour of its two-hour one set the prefix-
   landscape guard on all 21 replayed cycles, so each ended on the fallback - an
   hour on dishwasher defaults. That programme never pauses below 1.44 W until the
   machine switches itself off, so it cannot be what the ENDING quiet belongs to.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata.const import (
    BANKED_TAIL_REPAIR_KEY,
    STORAGE_KEY,
    STORAGE_VERSION,
    TERMINAL_EVENT_PEAK_FRAC,
)
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
)
from custom_components.ha_washdata.profile_store import (
    ProfileStore,
    WashDataStore,
    _match_prefix_ambiguity,
)
from custom_components.ha_washdata.signal_processing import (
    has_resumed_pause,
    quiet_run_before,
    terminal_quiet_seen,
)

T0 = datetime(2026, 9, 27, 11, 31, tzinfo=timezone.utc)
STOP = 0.96
QUIET = 611.5  # the Beko Eco profile's measured quiet_before_s (17/17 cycles)


def _beko(standby_until: float = 0.0) -> list[tuple[float, float]]:
    """The #424 Beko's shape: heating, a 641 s quiet at ~1.2 W (above the 0.96 W
    stop threshold, below 0.4% of peak), a long ~14 W phase, the drain, the 1.3 W
    wind-down sample, then 0.3 W standby."""
    pts = [(float(t), 2000.0) for t in range(0, 3000, 30)]
    pts += [(float(t), 1.2) for t in range(3000, 3641, 60)]
    pts += [(float(t), 14.0) for t in range(3641, 6000, 60)]
    pts += [(6000.0, 18.0), (6015.0, 16.0), (6030.0, 15.0), (6060.0, 1.3)]
    pts += [(float(t), 0.3) for t in range(6160, int(standby_until) + 1, 100)]
    return pts


# ---------------------------------------------------------------------------
# 1. The shared helper
# ---------------------------------------------------------------------------


def test_at_the_stop_threshold_the_beko_shows_no_quiet() -> None:
    """The bug, pinned: the old test finds nothing in front of the 1.3 W sample."""
    assert quiet_run_before(_beko(), 6060.0, STOP) == 0.0


def test_at_the_signature_threshold_the_drying_already_happened() -> None:
    assert terminal_quiet_seen(_beko(), 6060.0, STOP, QUIET, TERMINAL_EVENT_PEAK_FRAC)


def test_a_run_still_in_its_drying_phase_keeps_the_allowance() -> None:
    """The allowance exists for this: activity ended, drying is still to come."""
    pts = [(float(t), 2000.0) for t in range(0, 3000, 30)]
    pts += [(float(t), 0.3) for t in range(3000, 3300, 60)]
    assert not terminal_quiet_seen(pts, 2970.0, STOP, QUIET, TERMINAL_EVENT_PEAK_FRAC)


def test_the_stop_threshold_answer_still_counts() -> None:
    """Shorten-only: a run the item-347 test already judged dried stays judged so."""
    pts = [(float(t), 2000.0) for t in range(0, 3000, 30)]
    pts += [(float(t), 0.2) for t in range(3000, 3700, 60)]
    pts += [(3700.0, 30.0)]
    assert terminal_quiet_seen(pts, 3700.0, STOP, QUIET, TERMINAL_EVENT_PEAK_FRAC)


# ---------------------------------------------------------------------------
# 2. The live cap
# ---------------------------------------------------------------------------


def _dishwasher(points: list[tuple[float, float]]) -> CycleDetector:
    cfg = CycleDetectorConfig(
        min_power=0.1, off_delay=1800, device_type="dishwasher", stop_threshold_w=STOP
    )
    d = CycleDetector(cfg, lambda a, b: None, lambda p: None)
    d._current_cycle_start = T0
    d._expected_duration = 6000.0
    d._matched_terminal_quiet_s = QUIET
    d._end_spike_seen = False
    d._end_spike_duration = 0.0
    d._power_readings = [(T0 + timedelta(seconds=t), p) for t, p in points]
    d._last_active_time = T0 + timedelta(seconds=6060.0)
    return d


def test_the_beko_tail_cap_ends_at_its_last_activity() -> None:
    """0.5.7 stored every Beko cycle 611 s long: 249 min against a 239 min wash."""
    det = _dishwasher(_beko(standby_until=6700.0))
    cap = det._keep_tail_cap(T0)
    assert (cap - T0).total_seconds() == pytest.approx(6060.0)


# ---------------------------------------------------------------------------
# 3. The repair and its re-run
# ---------------------------------------------------------------------------


class _Store(ProfileStore):
    def __init__(self, data):  # pylint: disable=super-init-not-called
        self._data = data
        self._logger = MagicMock()
        self._cached_sample_segments = {}

    def iter_evidence_cycles(self):
        yield from (self._data.get("past_cycles") or [])

    async def async_save(self):
        return None

    async def async_rebuild_envelope(self, profile_name):  # noqa: ARG002
        return True


def _signature(store: _Store) -> _Store:
    store.compute_profile_terminal_signature = MagicMock(
        return_value={"quiet_before_s": QUIET, "seen_in": 17, "measured": 17, "consistency": 1.0}
    )
    return store


def _stored(cid: str, reason: str) -> dict:
    pts = _beko(standby_until=6660.0)
    return {
        "id": cid,
        "profile_name": "Eco",
        "start_time": T0.isoformat(),
        "duration": pts[-1][0],
        "termination_reason": reason,
        "power_data": [[t, p] for t, p in pts],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["smart", "timeout"])
async def test_the_repair_takes_the_banked_standby_back(reason: str) -> None:
    """Both finish paths kept the tail through the same cap on a dishwasher: the
    Beko banked it on smart finishes, the Samsung in the same report on timeout."""
    data = {"past_cycles": [_stored("a", reason)], BANKED_TAIL_REPAIR_KEY: True}
    st = _signature(_Store(data))

    res = await st.async_repair_banked_tails(STOP, "dishwasher")

    assert res["repaired"] == 1
    assert data["past_cycles"][0]["duration"] == pytest.approx(6060.0)


@pytest.mark.asyncio
async def test_a_washer_timeout_is_still_not_ours() -> None:
    """Every other type's timeout already snaps back to the last activity."""
    cycle = _stored("a", "timeout")
    data = {"past_cycles": [cycle], BANKED_TAIL_REPAIR_KEY: True}
    st = _signature(_Store(data))

    res = await st.async_repair_banked_tails(STOP, "washing_machine")

    assert res["repaired"] == 0


def _hass():
    h = MagicMock()
    h.config.config_dir = "/tmp"
    return h


@pytest.mark.asyncio
async def test_v14_re_arms_a_repair_that_already_ran() -> None:
    """The v13 pass cleared the key on every 0.5.7 store, so this must assign."""
    store = WashDataStore(_hass(), STORAGE_VERSION, f"{STORAGE_KEY}.test")
    result = await store._async_migrate_func(13, 1, {"past_cycles": [], BANKED_TAIL_REPAIR_KEY: False})
    assert result[BANKED_TAIL_REPAIR_KEY] is True


@pytest.mark.asyncio
async def test_v14_is_idempotent() -> None:
    store = WashDataStore(_hass(), STORAGE_VERSION, f"{STORAGE_KEY}.test")
    once = await store._async_migrate_func(13, 1, {"past_cycles": []})
    twice = await store._async_migrate_func(13, 1, dict(once))
    assert once == twice


# ---------------------------------------------------------------------------
# 4. Pause evidence for the prefix guard
# ---------------------------------------------------------------------------


def test_a_pause_counts_only_once_power_resumes() -> None:
    assert has_resumed_pause([(0, 100.0), (10, 0.0), (130, 100.0)], 1.44, 60.0)
    assert not has_resumed_pause([(0, 100.0), (10, 0.0), (50, 100.0)], 1.44, 60.0)
    # Still quiet at the end: that is the cycle's own end, not a pause.
    assert not has_resumed_pause([(0, 100.0), (10, 0.0), (900, 0.0)], 1.44, 60.0)


def test_a_change_only_plug_counts_its_silence() -> None:
    """One 0 W row and then nothing until the next activity is still a pause."""
    assert has_resumed_pause([(0, 50.0), (100, 0.0), (400, 50.0)], 1.44, 60.0)


def _profile_cycle(name: str, points: list[tuple[float, float]]) -> dict:
    return {"id": f"{name}-{len(points)}", "profile_name": name, "duration": points[-1][0],
            "power_data": [[t, p] for t, p in points]}


def test_profile_pauses_below_reads_the_evidence() -> None:
    never = [(float(t), 9.2 if t % 120 else 24.0) for t in range(0, 7200, 30)]
    pauses = [(0.0, 900.0), (600.0, 0.0), (1200.0, 900.0), (2400.0, 30.0)]
    st = _Store({"past_cycles": [
        _profile_cycle("Chef", never),
        _profile_cycle("Soak", pauses),
        {"id": "x", "profile_name": "Untraced", "duration": 3600.0},
    ]})
    assert st.profile_pauses_below("Chef", 1.44) is False
    assert st.profile_pauses_below("Soak", 1.44) is True
    assert st.profile_pauses_below("Untraced", 1.44) is None
    assert st.profile_pauses_below("Chef", 0.0) is None


def test_the_pause_cache_follows_the_evidence() -> None:
    data = {"past_cycles": [_profile_cycle("Chef", [(0.0, 50.0), (3600.0, 50.0)])]}
    st = _Store(data)
    assert st.profile_pauses_below("Chef", 1.44) is False
    data["past_cycles"].append(_profile_cycle("Chef", [(0.0, 50.0), (60.0, 0.0), (400.0, 50.0)]))
    assert st.profile_pauses_below("Chef", 1.44) is True


def _cands() -> list[dict]:
    return [
        {"name": "Auto", "profile_duration": 6818.0, "score": 0.58, "shape_score": 0.60},
        {"name": "Chef", "profile_duration": 11013.0, "score": 0.40, "shape_score": 0.477},
    ]


def test_a_longer_programme_that_never_pauses_cannot_block() -> None:
    """The second reporter's case, with their numbers: Chef 70 at 1.62x and 0.477."""
    assert _match_prefix_ambiguity(_cands(), 6818.0) == (True, False)
    assert _match_prefix_ambiguity(_cands(), 6818.0, lambda n: False) == (False, False)


@pytest.mark.parametrize("answer", [True, None])
def test_evidence_of_a_pause_or_no_evidence_keeps_the_guard(answer) -> None:
    assert _match_prefix_ambiguity(_cands(), 6818.0, lambda n: answer) == (True, False)


def test_a_failing_lookup_keeps_the_guard() -> None:
    def boom(_name):
        raise RuntimeError("boom")

    assert _match_prefix_ambiguity(_cands(), 6818.0, boom) == (True, False)


# ---------------------------------------------------------------------------
# 5. The reporters' own exports, through the real detector + matcher
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parent.parent
_TRON = _REPO / "cycle_data" / "tron4r" / "dishwasher" / "dishwasher_export_2026-09-28_v0.5.7_424.json"


@pytest.mark.slow
@pytest.mark.skipif(not _TRON.exists(), reason="reporter export not in cycle_data/")
def test_the_second_reporters_dishwasher_reaches_smart_termination() -> None:
    import sys

    sys.path.insert(0, str(_REPO / "devtools"))
    import end_gate_eval  # noqa: E402  # pylint: disable=import-outside-toplevel

    rows = end_gate_eval._measure_export(_TRON, no_shortening=False)  # noqa: SLF001
    # The five cycles the reporter tabulated on 2026-09-28, all `timeout` live.
    ids = {"ec51d2e7a961", "2ddb931416f4", "de0b0fd8c2c6", "d1a8cfc03436", "c7592ce87b51"}
    recent = [r for r in rows if r["id"] in ids]
    assert len(recent) == 5, json.dumps(rows)[:400]
    # All five reach Smart Termination. `2ddb931416f4` was held at the timeout by
    # top-1/top-2 ambiguity (item 393b) until the 0.5.8 matcher fixes (envelope
    # templates on the query grid, audit MATCH-CORE-01, among them) separated
    # its two programmes.
    smart = {r["id"] for r in recent if r["termination_reason"] == "smart"}
    assert smart == ids, [
        (r["id"], r["termination_reason"]) for r in recent
    ]
