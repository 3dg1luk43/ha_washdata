# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Register item 480: the #399 spin guard armed a programme on ONE member's spin.

``profile_terminal_high_block`` arms the anti-crease spin wait when the envelope's
``max`` band ends in a block above ``anti_wrinkle_max_power``, and the band is a
pointwise max, so a single member whose final spin crossed 400 W arms the whole
programme. Washer spins straddle that level (the held runs peak at 331-394 W, the
members that "spin" at 404-437 W), so most runs of such a programme never produce
the block and are held to the 1.25x cap: 8 of the 14 cap releases on the corpus.

Fix: once the envelope arms, the guard stays armed only when at least
``ANTI_CREASE_SPIN_ARM_MIN_SHARE`` of the profile's completed traced members end
with their own terminal spin above the level (``_member_terminal_spins``); fewer
than ``ANTI_CREASE_SPIN_ARM_MIN_MEMBERS`` evaluated members leave the envelope in
charge, as before.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.const import (
    ANTI_CREASE_SPIN_ARM_MIN_MEMBERS,
    ANTI_CREASE_SPIN_ARM_MIN_SHARE,
    ANTI_CREASE_SPIN_WAIT_MAX_RATIO,
)
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
)
from custom_components.ha_washdata.profile_store import ProfileStore

BASE = datetime(2026, 10, 4, 8, 0, 0, tzinfo=timezone.utc)
EXPECTED = 9000.0
SPIN_S = 60.0
SPIN_AT = 8400.0
TAIL_START = 8800.0


def _store() -> ProfileStore:
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(MagicMock(), "entry")
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
        ps.async_save = AsyncMock()
        return ps


def _wash(spin_peak: float, *, spin_start: float = SPIN_AT, step: float = 20.0) -> list[list[float]]:
    """Heating, a long sub-400 W wash, the final spin at ``spin_peak``, a low rinse."""
    pts: list[list[float]] = []
    t = 0.0
    while t <= EXPECTED:
        if t < 900.0:
            p = 2000.0
        elif spin_start <= t < spin_start + SPIN_S:
            p = spin_peak
        else:
            p = 60.0
        pts.append([t, p])
        t += step
    return pts


def _seeded(peaks: tuple[float, ...]) -> ProfileStore:
    """One member per peak; the envelope max band carries the highest spin."""
    ps = _store()
    ps._data["profiles"] = {"P": {"avg_duration": EXPECTED}}
    ps._data["envelopes"] = {"P": {"max": _wash(max(peaks))}}
    ps._data["past_cycles"] = [
        {"id": f"c{i}", "profile_name": "P", "status": "completed",
         "duration": EXPECTED, "power_data": _wash(peak)}
        for i, peak in enumerate(peaks)
    ]
    return ps


def _mixed(n: int, spinners: int) -> tuple[float, ...]:
    return tuple([437.0] * spinners + [380.0] * (n - spinners))


# ---------------------------------------------------------------------------
# profile_terminal_high_block: arming follows the members
# ---------------------------------------------------------------------------


def test_one_spinning_member_no_longer_arms_a_programme_that_does_not_spin() -> None:
    """1 of 4 members crossed the level: the band arms, the members do not."""
    ps = _seeded(_mixed(4, 1))
    assert 1 / 4 < ANTI_CREASE_SPIN_ARM_MIN_SHARE
    assert ps.profile_terminal_high_block("P", 400.0) is None


def test_a_programme_most_of_whose_runs_spin_stays_armed() -> None:
    ps = _seeded(_mixed(4, 3))
    block = ps.profile_terminal_high_block("P", 400.0)
    assert block is not None
    assert block[0] >= 0.9
    assert block[2] == pytest.approx(SPIN_AT, abs=1.0)


def test_the_share_is_inclusive() -> None:
    """Exactly the bar arms (2 of 4 at 0.5); one fewer spinner does not."""
    n = 4
    k = int(-(-ANTI_CREASE_SPIN_ARM_MIN_SHARE * n // 1))  # ceil
    assert _seeded(_mixed(n, k)).profile_terminal_high_block("P", 400.0) is not None
    assert _seeded(_mixed(n, k - 1)).profile_terminal_high_block("P", 400.0) is None


def test_a_thin_profile_keeps_the_envelope_decision() -> None:
    """Below the member floor the band (or sample) decides, as before 480: one
    member that spun under the level is not evidence the programme never spins."""
    ps = _seeded((380.0,) * (ANTI_CREASE_SPIN_ARM_MIN_MEMBERS - 1))
    ps._data["envelopes"] = {"P": {"max": _wash(437.0)}}
    block = ps.profile_terminal_high_block("P", 400.0)
    assert block is not None and block[0] >= 0.9
    assert block[2] == pytest.approx(SPIN_AT, abs=1.0)


def test_unfinished_members_do_not_vote() -> None:
    """Interrupted / force-stopped traces are not members (item 207's rule)."""
    ps = _seeded(_mixed(2, 2))
    for i in range(4):
        ps._data["past_cycles"].append(
            {"id": f"x{i}", "profile_name": "P", "status": "interrupted",
             "duration": 3000.0, "power_data": _wash(380.0)}
        )
    assert ps.profile_terminal_high_block("P", 400.0) is not None


def test_the_quorum_follows_the_evidence() -> None:
    ps = _seeded(_mixed(4, 3))
    assert ps.profile_terminal_high_block("P", 400.0) is not None
    for i in range(6):
        ps._data["past_cycles"].append(
            {"id": f"n{i}", "profile_name": "P", "status": "completed",
             "duration": EXPECTED, "power_data": _wash(380.0)}
        )
    assert ps.profile_terminal_high_block("P", 400.0) is None


# ---------------------------------------------------------------------------
# End to end through the real detector
# ---------------------------------------------------------------------------


def _config() -> CycleDetectorConfig:
    return CycleDetectorConfig(
        min_power=5.0,
        off_delay=300,
        device_type="washing_machine",
        stop_threshold_w=1.5,
        start_threshold_w=5.0,
        anti_wrinkle_enabled=True,
        anti_wrinkle_max_power=400.0,
        anti_wrinkle_max_duration=60.0,
        min_off_gap=300,
    )


def _live(t: float, spin_peak: float) -> float:
    """This run: its spin under the level, then the Knitterschutz tail."""
    if t < 900.0:
        return 2000.0
    if SPIN_AT <= t < SPIN_AT + SPIN_S:
        return spin_peak
    if t < TAIL_START:
        return 60.0
    return 60.0 if t % 37.0 < 4.0 else 3.3


def _replay(block, spin_peak: float) -> list[dict]:
    ended: list[dict] = []
    det = CycleDetector(
        config=_config(),
        on_state_change=lambda _o, _n: None,
        on_cycle_end=lambda d: ended.append(d),
    )
    match = ("P", 0.9, EXPECTED, None, False, False, False, False, 60.0, block)
    t = 0.0
    while t <= EXPECTED * ANTI_CREASE_SPIN_WAIT_MAX_RATIO + 600.0 and not ended:
        det.process_reading(_live(t, spin_peak), BASE + timedelta(seconds=t))
        if not ended:
            det.update_match(match)
        t += 10.0
    return ended


def test_a_run_spinning_under_the_level_finalises_on_its_tail_not_at_the_cap() -> None:
    block = _seeded(_mixed(4, 1)).profile_terminal_high_block("P", 400.0)
    ended = _replay(block, spin_peak=380.0)
    assert len(ended) == 1
    assert float(ended[0]["duration"]) < EXPECTED * 1.05


def test_before_480_the_same_run_was_held_to_the_cap() -> None:
    """The envelope-only block, for contrast: the hold outlasts the tail."""
    env = _seeded(_mixed(4, 1))
    env._data["past_cycles"] = []
    block = env.profile_terminal_high_block("P", 400.0)
    assert block is not None
    ended = _replay(block, spin_peak=380.0)
    assert len(ended) == 1
    assert float(ended[0]["duration"]) >= EXPECTED * ANTI_CREASE_SPIN_WAIT_MAX_RATIO - 60.0
