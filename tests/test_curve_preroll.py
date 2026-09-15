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
"""Issue #430: fill the missing start of a cycle's curve from a buffer.

A cycle's curve begins at the start probe that finally COMMITS. Earlier probes
that aborted as false starts take their readings with them, so an appliance that
probes repeatedly before settling (programme selection, door lock, first fill)
loses the first 40-217 s of real activity from the front of every curve. #403
makes that more common: the first high reading now earns no evidence toward
either start gate, so a sparse change-only sensor aborts more probes.

Two invariants this locks down:

* Off by default, and with it off the curve is byte-identical to before.
* When it fires, ``_current_cycle_start`` moves back WITH the curve. That is not
  a choice: the stored duration is ``end_time - _current_cycle_start`` while
  matching resamples ``_power_readings``, so a curve starting earlier than the
  pointer would describe a different run from the one whose duration is stored.

Fast, pure-unit tests (no HA boot, no file I/O, no cycle_data replay).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

from custom_components.ha_washdata import playground, ws_api
from custom_components.ha_washdata.const import (
    ANTI_CREASE_FINALIZE_RATIO_MIN,
    CONF_CURVE_PREROLL_SECONDS,
    CONF_POWER_SENSOR,
    CURVE_PREROLL_MAX_SECONDS,
    DEFAULT_CURVE_PREROLL_SECONDS,
    DOMAIN,
    PREROLL_CHAIN_BREAK_SECONDS,
    STATE_DELAY_WAIT,
    STATE_OFF,
    STATE_STARTING,
    STATE_RUNNING,
)
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
)
from custom_components.ha_washdata.signal_processing import (
    energy_gap_threshold_s,
    integrate_wh,
)


def _dt(offset_seconds: float) -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(
        seconds=offset_seconds
    )


def _detector(preroll: float) -> CycleDetector:
    """A detector whose start gates need sustained power.

    ``start_duration_threshold`` of 40 s means a short probe cannot commit, which
    is what produces the aborted probes this feature recovers.
    """
    cfg = CycleDetectorConfig(
        min_power=5.0,
        off_delay=60,
        completion_min_seconds=600,
        start_duration_threshold=40.0,
        start_energy_threshold=0.0,
        start_threshold_w=20.0,
        stop_threshold_w=4.0,
        curve_preroll_seconds=preroll,
    )
    return CycleDetector(
        config=cfg, on_state_change=Mock(), on_cycle_end=Mock(), profile_matcher=None
    )


def _probe_then_commit(det: CycleDetector) -> None:
    """An aborted probe at t=0..20, quiet, then a real start at t=60.

    The probe is 3 readings of genuine activity (door lock / first fill) that the
    detector currently throws away; the quiet gap between it and the commit is
    inside PREROLL_CHAIN_BREAK_SECONDS, so it belongs to the same start.
    """
    det.process_reading(120.0, _dt(0))
    det.process_reading(150.0, _dt(10))
    det.process_reading(130.0, _dt(20))
    det.process_reading(0.5, _dt(30))  # probe aborts
    det.process_reading(0.5, _dt(45))
    for t in range(60, 200, 10):  # the start that sticks
        det.process_reading(500.0, _dt(t))


# ---------------------------------------------------------------------------
# Off by default
# ---------------------------------------------------------------------------


def test_default_is_off() -> None:
    assert DEFAULT_CURVE_PREROLL_SECONDS == 0.0
    cfg = CycleDetectorConfig(min_power=5.0, off_delay=60)
    assert cfg.curve_preroll_seconds == 0.0


def test_disabled_reproduces_the_current_curve_exactly() -> None:
    """With the option off, nothing about the trace may change."""
    det = _detector(0.0)
    _probe_then_commit(det)
    assert det.state == STATE_RUNNING
    assert det._current_cycle_start == _dt(60), "start unchanged"
    assert det._power_readings[0] == (_dt(60), 500.0), "curve unchanged"
    assert det._preroll_buffer == [], "no buffer is kept while the option is off"


# ---------------------------------------------------------------------------
# The reported case
# ---------------------------------------------------------------------------


def test_enabled_recovers_the_aborted_probe() -> None:
    det = _detector(300.0)
    _probe_then_commit(det)
    assert det.state == STATE_RUNNING
    assert det._current_cycle_start == _dt(0), "start moved back to the first probe"
    assert det._power_readings[:3] == [
        (_dt(0), 120.0),
        (_dt(10), 150.0),
        (_dt(20), 130.0),
    ]


def test_curve_and_duration_pointer_agree() -> None:
    """The invariant that forces the start pointer to move with the curve."""
    det = _detector(300.0)
    _probe_then_commit(det)
    assert det._power_readings[0][0] == det._current_cycle_start


def test_preroll_does_not_feed_the_start_energy_gate() -> None:
    """Record-only: pre-roll must never make a cycle easier to START.

    ``_energy_since_idle_wh`` is the accumulator the STARTING -> RUNNING gate
    reads, not the cycle's stored energy. Crediting it with an aborted probe's
    energy would let two blips that each failed the gate pass it together, which
    is the phantom cycle #403 was fixed to prevent.
    """
    off = _detector(0.0)
    on = _detector(300.0)
    for det in (off, on):
        det.process_reading(120.0, _dt(0))
        det.process_reading(150.0, _dt(10))
        det.process_reading(130.0, _dt(20))
        det.process_reading(0.5, _dt(30))
        det.process_reading(500.0, _dt(60))  # commits STARTING here

    assert on._current_cycle_start == _dt(0), "the curve did move back"
    assert on._energy_since_idle_wh == pytest.approx(off._energy_since_idle_wh), (
        "but the start-gate accumulator did not"
    )
    assert on._time_above_threshold == pytest.approx(off._time_above_threshold)


def test_a_repeated_blip_still_cannot_start_a_cycle() -> None:
    """The #403 guarantee, restated against pre-roll: two sub-gate blips must
    not combine into a start."""
    det = _detector(300.0)
    det.config.start_energy_threshold = 0.5  # Wh - one blip alone cannot reach it
    det.process_reading(200.0, _dt(0))  # blip 1
    det.process_reading(0.5, _dt(5))
    det.process_reading(200.0, _dt(40))  # blip 2, same pre-roll chain
    det.process_reading(0.5, _dt(45))
    assert det.state != STATE_RUNNING, "two blips may not add up to a start"


def test_stored_energy_still_covers_the_preroll() -> None:
    """The energy a cycle is stored with is integrated from its trace, so
    extending the trace picks the pre-roll up without touching the gate."""
    det = _detector(300.0)
    _probe_then_commit(det)
    start = det._current_cycle_start
    offsets = np.array(
        [(ts - start).total_seconds() for ts, _p in det._power_readings], dtype=float
    )
    powers = np.array([p for _ts, p in det._power_readings], dtype=float)
    with_preroll = integrate_wh(
        offsets, powers, max_gap_s=energy_gap_threshold_s(offsets)
    )

    bare = _detector(0.0)
    _probe_then_commit(bare)
    b_start = bare._current_cycle_start
    b_off = np.array(
        [(ts - b_start).total_seconds() for ts, _p in bare._power_readings], dtype=float
    )
    b_pow = np.array([p for _ts, p in bare._power_readings], dtype=float)
    without = integrate_wh(b_off, b_pow, max_gap_s=energy_gap_threshold_s(b_off))

    assert with_preroll > without


# ---------------------------------------------------------------------------
# What must NOT be carried
# ---------------------------------------------------------------------------


def test_a_long_quiet_gap_breaks_the_chain() -> None:
    """An unrelated blip earlier in the day is not part of this start."""
    det = _detector(600.0)
    det.process_reading(120.0, _dt(0))  # unrelated blip
    det.process_reading(0.5, _dt(10))
    quiet = PREROLL_CHAIN_BREAK_SECONDS + 30
    for t in range(60, 200, 10):
        det.process_reading(500.0, _dt(10 + quiet + t))
    assert det._current_cycle_start == _dt(10 + quiet + 60)


def test_anchor_is_the_first_active_reading_not_the_window_edge() -> None:
    """Standby ahead of the probe must not become the cycle start."""
    det = _detector(300.0)
    det.process_reading(2.0, _dt(0))  # standby, below start_threshold_w
    det.process_reading(2.0, _dt(20))
    det.process_reading(120.0, _dt(40))  # first genuine activity
    det.process_reading(0.5, _dt(50))
    for t in range(80, 220, 10):
        det.process_reading(500.0, _dt(t))
    assert det._current_cycle_start == _dt(40)
    assert det._power_readings[0] == (_dt(40), 120.0)


def test_an_all_standby_chain_carries_nothing() -> None:
    """Nothing above the start threshold means nothing of this cycle happened yet."""
    det = _detector(300.0)
    det.process_reading(2.0, _dt(0))
    det.process_reading(2.0, _dt(20))
    for t in range(40, 180, 10):
        det.process_reading(500.0, _dt(t))
    assert det._current_cycle_start == _dt(40)


def test_readings_older_than_the_window_are_dropped() -> None:
    """The buffer is bounded by the configured window, not by the chain rule.

    A 15 s window with the aborted probe 30 s back: the chain rule alone would
    happily carry it (the quiet gap is well inside PREROLL_CHAIN_BREAK_SECONDS),
    so the start staying put proves the window is what stopped it.
    """
    det = _detector(15.0)
    det.process_reading(120.0, _dt(0))  # probe, outside the 15 s window
    det.process_reading(130.0, _dt(10))
    det.process_reading(0.5, _dt(20))  # probe aborts
    for t in range(30, 170, 10):
        det.process_reading(500.0, _dt(t))
    assert det._current_cycle_start == _dt(30)
    assert det._power_readings[0] == (_dt(30), 500.0)


def test_previous_cycle_tail_is_never_carried_over() -> None:
    """The buffer is pre-CYCLE context; a reset must empty it."""
    det = _detector(300.0)
    for t in range(0, 1200, 10):
        det.process_reading(500.0, _dt(t))
    assert det.state == STATE_RUNNING
    det.reset()
    assert det._preroll_buffer == []
    assert det.state == STATE_OFF


def _delayed_start_detector(preroll: float, confirm_s: float = 40.0) -> CycleDetector:
    """A detector that parks in DELAY_WAIT, then confirms on sustained power."""
    cfg = CycleDetectorConfig(
        min_power=5.0,
        off_delay=60,
        completion_min_seconds=600,
        start_duration_threshold=confirm_s,
        start_energy_threshold=0.0,
        start_threshold_w=100.0,
        stop_threshold_w=4.0,
        delay_detect_enabled=True,
        delay_confirm_seconds=30.0,
        curve_preroll_seconds=preroll,
    )
    det = CycleDetector(
        config=cfg, on_state_change=Mock(), on_cycle_end=Mock(), profile_matcher=None
    )
    det.process_reading(0.0, _dt(0))
    for t in range(10, 70, 10):  # standby band -> DELAY_WAIT
        det.process_reading(25.0, _dt(t))
    assert det.state == STATE_DELAY_WAIT
    return det


def test_delayed_start_keeps_the_samples_between_anchor_and_confirmation() -> None:
    """The DELAY_WAIT commit built its curve from two points and dropped the rest.

    The anchor is the first sustained-high reading and the confirmation arrives
    `start_duration_threshold` later, so everything measured in between was
    thrown away even though the buffer records in DELAY_WAIT.
    """
    det = _delayed_start_detector(300.0)
    for t in range(100, 160, 10):  # high power: anchor at 100, confirms at 140
        det.process_reading(800.0, _dt(t))

    assert det.state in (STATE_STARTING, STATE_RUNNING)
    assert det._current_cycle_start == _dt(100), "the delayed-start anchor must hold"
    carried = [ts for ts, _p in det._power_readings if _dt(100) <= ts <= _dt(140)]
    assert len(carried) >= 4, (
        "the readings measured during the confirmation window are missing from "
        f"the curve: {[t.second for t in carried]}"
    )


def test_a_short_preroll_window_cannot_move_the_delayed_anchor_forward() -> None:
    """The guard: pre-roll only ever moves the start pointer earlier.

    With a 15 s window and a 40 s confirmation the buffered chain begins well
    after the anchor, and adopting its first reading as the start would shorten
    the delayed cycle by the difference.
    """
    det = _delayed_start_detector(15.0)
    for t in range(100, 160, 10):
        det.process_reading(800.0, _dt(t))

    assert det.state in (STATE_STARTING, STATE_RUNNING)
    assert det._current_cycle_start == _dt(100)
    assert det._power_readings[0][0] == _dt(100)


def test_buffer_is_not_filled_while_a_cycle_runs() -> None:
    """Once RUNNING the curve IS the record; a second copy would be dead weight."""
    det = _detector(300.0)
    for t in range(0, 400, 10):
        det.process_reading(500.0, _dt(t))
    assert det.state == STATE_RUNNING
    before = len(det._preroll_buffer)
    for t in range(400, 800, 10):
        det.process_reading(500.0, _dt(t))
    assert len(det._preroll_buffer) == before


# ---------------------------------------------------------------------------
# ws_set_options validation
# ---------------------------------------------------------------------------


def _entry(options: dict) -> MagicMock:
    entry = MagicMock()
    entry.entry_id = "e1"
    entry.data = {CONF_POWER_SENSOR: "sensor.power"}
    entry.options = options
    return entry


def _hass() -> MagicMock:
    manager = MagicMock()
    manager.profile_store.async_record_settings_changes = AsyncMock()
    hass = MagicMock()
    hass.data = {DOMAIN: {"e1": manager}}
    return hass


async def _set_options(entry: MagicMock, hass: MagicMock, options: dict) -> dict:
    ws_fn = ws_api.ws_set_options.__wrapped__
    with patch.object(ws_api, "_get_entry", return_value=entry):
        await ws_fn(hass, MagicMock(), {"id": 1, "entry_id": "e1", "options": options})
    return hass.config_entries.async_update_entry.call_args.kwargs["options"]


@pytest.mark.parametrize(
    ("submitted", "stored"),
    [
        (300, 300.0),
        (0, 0.0),
        (-5, 0.0),
        (99999, CURVE_PREROLL_MAX_SECONDS),
        (float("inf"), None),
        (float("nan"), None),
        ("", None),
        (None, None),
        ("abc", None),
    ],
)
async def test_validation(submitted, stored) -> None:
    saved = await _set_options(
        _entry({}), _hass(), {CONF_CURVE_PREROLL_SECONDS: submitted}
    )
    if stored is None:
        assert CONF_CURVE_PREROLL_SECONDS not in saved
    else:
        assert saved[CONF_CURVE_PREROLL_SECONDS] == stored


# ---------------------------------------------------------------------------
# Playground
# ---------------------------------------------------------------------------


def test_effective_settings_surfaces_the_window() -> None:
    cfg = CycleDetectorConfig(min_power=5.0, off_delay=60, curve_preroll_seconds=300.0)
    eff = playground.effective_settings(cfg, None)
    assert eff[CONF_CURVE_PREROLL_SECONDS] == 300.0


def test_playground_override_is_wired() -> None:
    assert (
        playground._OVERRIDE_FIELD_MAP[CONF_CURVE_PREROLL_SECONDS][0]
        == "curve_preroll_seconds"
    )


def test_the_sim_summary_reports_the_window_the_sim_applied() -> None:
    """An override the detector clamps must not be echoed back unclamped.

    ``build_sim_config`` passes a non-negative override through as given, while
    every detector read caps it at ``CURVE_PREROLL_MAX_SECONDS``, so the summary
    would describe a sim that did not run.
    """
    cfg = CycleDetectorConfig(
        min_power=5.0,
        off_delay=60,
        curve_preroll_seconds=CURVE_PREROLL_MAX_SECONDS * 4,
        anti_crease_finalize_ratio=0.0,
    )
    summary = playground._sim_config_summary(cfg)

    assert summary["curve_preroll_seconds"] == CURVE_PREROLL_MAX_SECONDS
    # 0.0 is the case that silently disarms the gate; the gate reads 0.5.
    assert summary["anti_crease_finalize_ratio"] == ANTI_CREASE_FINALIZE_RATIO_MIN
