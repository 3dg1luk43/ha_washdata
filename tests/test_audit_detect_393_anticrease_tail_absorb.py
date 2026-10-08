# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Register item 393a: the #296 anti-crease tail opened a second cycle on itself.

TRON4R's Miele washer ends every Pflegeleicht / Jeans / Baumwolle programme with
~30 min of Knitterschutz: a ~3 W baseline and a 40-85 W drum burst every ~37 s.
Its plug reports on change and now and then skips the dip between two bursts, so
all 20 of the reporter's stored tails hold 60-111 s without a reading under
``stop_threshold_w``. After the #296 finalise put the cycle into
STATE_ANTI_WRINKLE, the first such stretch beat ``anti_wrinkle_max_duration``
(60 s) and the tail itself was recorded as a new cycle (``c24effabc3``: finalised
at 91 min, the last 21 min of tumbling became cycle two).

Fix: while the anti-wrinkle state was entered by the #296 finalise, the burst
length that means "a new wash" is at least ``ANTI_CREASE_CONFIRM_WINDOW_S``, the
window the finalise itself accepted as tail. The power test is unchanged, so a
next wash's heating still leaves at once; other entries keep the configured value.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from custom_components.ha_washdata.const import ANTI_CREASE_CONFIRM_WINDOW_S
from custom_components.ha_washdata.cycle_detector import (
    STATE_ANTI_WRINKLE,
    STATE_STARTING,
    CycleDetector,
    CycleDetectorConfig,
)

BASE = datetime(2026, 7, 16, 11, 0, 0, tzinfo=timezone.utc)
EXPECTED = 5400.0
WASH_END = 4500.0     # heating/rinse/spin over; Knitterschutz from here
TAIL_END = 6400.0     # the crease guard stops (~30 min)
PERIOD = 37.0


def _config(**over) -> CycleDetectorConfig:
    cfg = dict(
        min_power=2.0,
        off_delay=150,
        device_type="washing_machine",
        stop_threshold_w=5.0,
        start_threshold_w=6.3,
        anti_wrinkle_enabled=True,
        anti_wrinkle_max_power=400.0,
        anti_wrinkle_max_duration=60.0,
        min_off_gap=480,
    )
    cfg.update(over)
    return CycleDetectorConfig(**cfg)


def _wash(t: float) -> float:
    if t < 900.0:
        return 2200.0
    return 100.0 if int(t) % 120 < 90 else 3.3


def _tail_readings(start: float, end: float, *, skip_every: int = 6) -> list[tuple[float, float]]:
    """Burst, decay, baseline - except that in every ``skip_every`` bursts the plug
    reports the dip after two of them, so three bursts read as one ~76 s run (the
    reporter's 5736-5812 s stretch in ``c24effabc3``).
    """
    out: list[tuple[float, float]] = []
    t, n = start, 0
    while t < end:
        out += [(t, 62.0), (t + 2.0, 37.0)]
        if n % skip_every not in (skip_every - 2, skip_every - 1):
            out += [(t + 4.0, 3.3), (t + 20.0, 3.2)]
        t += PERIOD
        n += 1
    return out


def _run(readings: list[tuple[float, float]], *, config: CycleDetectorConfig | None = None):
    ended: list[dict] = []
    states: list[tuple[float, str]] = []
    det = CycleDetector(
        config=config or _config(),
        on_state_change=lambda _o, n: states.append((cur[0], n)),
        on_cycle_end=lambda d: ended.append(d),
    )
    cur = [0.0]
    match = ("P", 0.9, EXPECTED, None, False, False, False, False, 60.0, None)
    for t, p in readings:
        cur[0] = t
        det.process_reading(p, BASE + timedelta(seconds=t))
        if not ended:
            det.update_match(match)
    return ended, states, det


def _wash_then_tail(**tail_kw) -> list[tuple[float, float]]:
    readings = [(float(t), _wash(float(t))) for t in range(0, int(WASH_END), 10)]
    return readings + _tail_readings(WASH_END, TAIL_END, **tail_kw)


def test_the_tumble_tail_is_absorbed_when_the_plug_skips_a_dip() -> None:
    ended, states, det = _run(_wash_then_tail())
    assert len(ended) == 1
    # Finalised by the #296 path on the tail, before the crease guard stopped.
    assert float(ended[0]["duration"]) < TAIL_END - 300.0
    assert STATE_ANTI_WRINKLE in [s for _t, s in states]
    # ...and nothing after it read the tail as a new wash.
    finalised_at = next(t for t, s in states if s == STATE_ANTI_WRINKLE)
    assert not [s for t, s in states if t > finalised_at and s == STATE_STARTING]
    assert det._current_cycle_start is None


def test_without_skipped_dips_nothing_changes() -> None:
    ended, states, _det = _run(_wash_then_tail(skip_every=10_000))
    assert len(ended) == 1
    finalised_at = next(t for t, s in states if s == STATE_ANTI_WRINKLE)
    assert not [s for t, s in states if t > finalised_at and s == STATE_STARTING]


def test_a_next_wash_still_leaves_the_tail_on_its_heating() -> None:
    readings = _wash_then_tail()
    nxt = readings[-1][0] + 30.0
    readings += [(nxt + i * 10.0, 2200.0) for i in range(30)]
    ended, states, det = _run(readings)
    assert len(ended) == 1
    assert det._current_cycle_start is not None
    # Backdated to the burst the heating followed without a dip, as before.
    assert det._current_cycle_start >= BASE + timedelta(seconds=nxt - 2 * PERIOD)


def test_a_sustained_low_power_start_still_leaves_after_the_window() -> None:
    """A cold next wash that stays under max power still opens a cycle, only
    after the finalise's own window instead of 60 s."""
    readings = _wash_then_tail()
    nxt = readings[-1][0] + 30.0
    readings += [(nxt + i * 10.0, 80.0) for i in range(60)]
    ended, states, _det = _run(readings)
    starts = [t for t, s in states if s == STATE_STARTING and t >= nxt]
    assert starts
    assert starts[0] - nxt > float(_config().anti_wrinkle_max_duration)
    assert starts[0] - nxt <= ANTI_CREASE_CONFIRM_WINDOW_S + 15.0


def test_a_tail_whose_baseline_sits_above_stop_threshold_is_absorbed() -> None:
    """The #296 shape the code comments describe: a ~3 W crease-guard draw over a
    lower stop threshold. No reading ever fell under the exit level, the burst
    candidate never reset, and every such tail left anti-wrinkle after
    ``anti_wrinkle_max_duration`` (113 of 146 finalised replays in
    ``end_gate_eval --anti-wrinkle force --tumble-tail`` grew a tail cycle)."""
    readings = [(float(t), _wash(float(t))) for t in range(0, int(WASH_END), 10)]
    t = WASH_END
    while t < TAIL_END:
        readings.append((t, 60.0 if t % PERIOD < 4.0 else 6.0))
        t += 10.0
    nxt = TAIL_END + 30.0
    readings += [(nxt + i * 10.0, 2200.0) for i in range(30)]
    ended, states, det = _run(readings)
    assert len(ended) == 1
    finalised_at = next(t for t, s in states if s == STATE_ANTI_WRINKLE)
    assert not [s for t, s in states if finalised_at < t < nxt and s == STATE_STARTING]
    # The next load is its own cycle, started on its own heating.
    assert det._current_cycle_start is not None
    assert det._current_cycle_start >= BASE + timedelta(seconds=nxt - 2 * PERIOD)


def test_other_anti_wrinkle_entries_keep_the_configured_burst_length() -> None:
    """A dryer finished by the ordinary end gates is not on a recognised tail."""
    det = CycleDetector(
        config=_config(device_type="dryer"),
        on_state_change=lambda _o, _n: None,
        on_cycle_end=lambda _d: None,
    )
    t = 0.0
    for _ in range(360):
        det.process_reading(2200.0, BASE + timedelta(seconds=t))
        t += 10.0
    for _ in range(150):
        det.process_reading(0.0, BASE + timedelta(seconds=t))
        t += 10.0
        if det.state == STATE_ANTI_WRINKLE:
            break
    assert det.state == STATE_ANTI_WRINKLE
    assert det._anticrease_tail_floor_w is None
    for _ in range(10):
        det.process_reading(80.0, BASE + timedelta(seconds=t))
        t += 10.0
    assert det.state != STATE_ANTI_WRINKLE


_REPO = Path(__file__).resolve().parent.parent
_TRON = (
    _REPO / "cycle_data" / "tron4r" / "washing_machine"
    / "ha_washdata_export_01KWFX8C3HVEK7YK6F9N6KAVVS.json"
)


@pytest.mark.slow
@pytest.mark.skipif(not _TRON.exists(), reason="reporter export not in cycle_data/")
def test_the_reporters_pflegeleicht_wash_stays_one_cycle() -> None:
    """``c24effabc3`` with the reporter's own settings (anti-wrinkle on)."""
    sys.path.insert(0, str(_REPO / "devtools"))
    import end_gate_eval  # noqa: E402  # pylint: disable=import-outside-toplevel

    end_gate_eval._integration()  # noqa: SLF001
    doc = end_gate_eval._load_doc(_TRON, True)  # noqa: SLF001
    data = doc["data"]
    base = {**data, "profiles": {k: dict(v) for k, v in data["profiles"].items()},
            "envelopes": dict(data.get("envelopes") or {})}
    cfg, store, opts = end_gate_eval._production(doc, base)  # noqa: SLF001
    assert cfg.anti_wrinkle_enabled
    end_gate_eval._rebuild_envelopes(store, list(base["profiles"]))  # noqa: SLF001
    cyc = next(c for c in base["past_cycles"] if str(c["id"]).startswith("c24effabc3"))
    sim = end_gate_eval.playground.simulate_cycle_detail(
        cyc, cfg, None, store, opts, price=None, compute_series=False,
    )
    out = sim["outcome"]
    events = [e["detail"] for e in sim["events"] if e["type"] in ("state", "finished")]
    # The #296 finalise still fires on the tail; the tail is not a second cycle.
    assert "running->anti_wrinkle" in events, json.dumps(events)
    assert out["detected_count"] == 1, json.dumps(events)
