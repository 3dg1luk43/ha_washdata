# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fast tests for the Playground stress-tail (idle termination test) feature."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata import playground
from custom_components.ha_washdata.const import TerminationReason
from custom_components.ha_washdata.cycle_detector import CycleDetectorConfig


# ─── helpers ────────────────────────────────────────────────────────────────────


def _cfg(**kwargs) -> CycleDetectorConfig:
    """Minimal CycleDetectorConfig for stress-tail tests."""
    defaults = dict(
        min_power=2.0,
        off_delay=180,
        device_type="washing_machine",
        stop_threshold_w=2.0,
        start_threshold_w=10.0,
        min_off_gap=60,
        completion_min_seconds=60,
    )
    defaults.update(kwargs)
    return CycleDetectorConfig(**defaults)


def _store() -> MagicMock:
    """Minimal store mock."""
    s = MagicMock()
    s._data = {"profiles": {}, "past_cycles": [], "reference_cycles": []}
    s._min_duration_ratio = 0.05
    s._max_duration_ratio = 1.5
    s.dtw_bandwidth = 0.2
    s._matching_overrides.return_value = {}
    s._grouped_snapshots.return_value = ([], {}, {})
    s.get_envelope.return_value = None
    s.phase_remaining.return_value = None
    s.get_lifetime_cycle_count.return_value = 0
    return s


def _make_cycle(
    duration_s: float = 3600.0,
    peak_w: float = 2000.0,
    tail_idle_w: float = 3.2,
    sensor_interval_s: float = 10.0,
) -> dict:
    """Synthetic cycle: hot phase then a standby-draw tail."""
    base = datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
    pts: list[list[float]] = []
    hot_s = duration_s * 0.8
    t = 0.0
    while t < hot_s:
        pts.append([t, peak_w])
        t += sensor_interval_s
    while t <= duration_s:
        pts.append([t, tail_idle_w])
        t += sensor_interval_s
    return {
        "id": "test-stress-cycle-abc",
        "start_time": base.isoformat(),
        "duration": duration_s,
        "status": "completed",
        "power_data": pts,
        "profile_name": None,
        "sampling_interval": sensor_interval_s,
    }


def _run_stress(
    cycle: dict,
    cfg: CycleDetectorConfig,
    store=None,
    stress_idle_w: float | None = None,
) -> dict:
    """Build and run a _DetailSim with stress_tail=True; return finalize()."""
    if store is None:
        store = _store()
    sim = playground._DetailSim(
        cycle, cfg, None, store, {}, None,
        compute_series=True, prebuilt=None,
        stress_tail=True, stress_idle_w=stress_idle_w,
    )
    assert sim.ready, "cycle must have ≥5 readings"
    sim.step(0, sim.n_readings)
    sim.run_stress_tail()
    return sim.finalize()


# ─── tests ──────────────────────────────────────────────────────────────────────


def test_stress_idle_below_threshold_terminates():
    """Idle draw below stop_threshold_w → cycle terminates (not force-stopped).

    The real cycle's tail must be ABOVE stop_threshold so step() doesn't
    terminate it first.  We then override stress_idle_w to below threshold.
    """
    # Real tail 5W (above stop_threshold=2W) → step() keeps cycle RUNNING.
    # stress_idle_w=1.0W (below stop_threshold=2W) → stress tail terminates it.
    cycle = _make_cycle(duration_s=3600.0, tail_idle_w=5.0)
    cfg = _cfg(stop_threshold_w=2.0, off_delay=180, min_off_gap=60)
    d = _run_stress(cycle, cfg, stress_idle_w=1.0)

    st = d["outcome"]["stress"]
    assert st is not None
    assert st["enabled"] is True
    assert st["idle_w"] == pytest.approx(1.0, abs=0.01)
    assert st["idle_above_threshold"] is False
    assert st["terminated"] is True
    assert st["hit_cap"] is False
    assert st["terminated_after_s"] < 3600, "should terminate well within an hour"

    alert_codes = [a["code"] for a in d["alerts"]]
    assert "stress_terminated" in alert_codes
    assert "stress_hit_cap" not in alert_codes


def test_stress_idle_above_threshold_hits_cap():
    """Idle draw above stop_threshold_w → stays RUNNING → 8h safety cap fires."""
    cycle = _make_cycle(duration_s=3600.0, tail_idle_w=5.0)
    cfg = _cfg(stop_threshold_w=2.0, off_delay=180, min_off_gap=60)
    d = _run_stress(cycle, cfg)

    st = d["outcome"]["stress"]
    assert st["idle_above_threshold"] is True
    assert st["hit_cap"] is True
    assert st["terminated"] is True

    alert_codes = [a["code"] for a in d["alerts"]]
    assert "stress_hit_cap" in alert_codes
    assert "stress_terminated" not in alert_codes


def test_stress_quiet_tail_unchanged():
    """stress_tail=False → outcome.stress is None; behaviour unchanged."""
    cycle = _make_cycle(duration_s=600.0, tail_idle_w=1.0)
    cfg = _cfg(stop_threshold_w=2.0, off_delay=60, min_off_gap=60)
    store = _store()

    sim = playground._DetailSim(
        cycle, cfg, None, store, {}, None,
        compute_series=True, prebuilt=None,
        stress_tail=False,
    )
    sim.step(0, sim.n_readings)
    sim.run_tail()
    result = sim.finalize()

    assert result["outcome"]["stress"] is None


def test_stress_deterministic():
    """Two stress runs with same cycle → identical stress outcome."""
    cycle = _make_cycle(duration_s=3600.0, tail_idle_w=1.0)
    cfg = _cfg(stop_threshold_w=2.0, off_delay=180, min_off_gap=60)

    d1 = _run_stress(cycle, cfg)
    d2 = _run_stress(cycle, cfg)

    assert d1["outcome"]["stress"] == d2["outcome"]["stress"]
    assert len(d1["series"]) == len(d2["series"])
    for pt1, pt2 in zip(d1["series"], d2["series"]):
        assert pt1["t"] == pt2["t"]
        assert pt1["power"] == pt2["power"]
        assert pt1["state"] == pt2["state"]


def test_stress_series_has_points_during_tail():
    """Series points exist past the stored duration during the stress tail."""
    cycle = _make_cycle(duration_s=600.0, tail_idle_w=1.0)
    cfg = _cfg(stop_threshold_w=2.0, off_delay=60, min_off_gap=60)
    d = _run_stress(cycle, cfg)

    stored_dur = float(d["duration_s"] or 0)
    stress_pts = [pt for pt in d["series"] if pt["t"] > stored_dur]
    assert len(stress_pts) >= 1, "stress tail must add at least one series point"
    for pt in stress_pts:
        assert "power" in pt
        assert "state" in pt
        assert "t" in pt


def test_stress_manual_override():
    """stress_idle_w override sets the hold level; outcome reflects it."""
    cycle = _make_cycle(duration_s=3600.0, tail_idle_w=3.2)
    cfg = _cfg(stop_threshold_w=2.0, off_delay=180, min_off_gap=60)
    d = _run_stress(cycle, cfg, stress_idle_w=8.0)

    st = d["outcome"]["stress"]
    assert st["manual_override"] is True
    assert st["idle_w"] == pytest.approx(8.0, abs=0.01)
    assert st["idle_above_threshold"] is True


def test_stress_negative_override_clamped_to_zero():
    """A negative stress_idle_w override is clamped to 0 (the schema allows any float),
    so the outcome never reports a nonsensical negative idle draw."""
    cycle = _make_cycle(duration_s=3600.0, tail_idle_w=3.2)
    cfg = _cfg(stop_threshold_w=2.0, off_delay=180, min_off_gap=60)
    d = _run_stress(cycle, cfg, stress_idle_w=-5.0)

    st = d["outcome"]["stress"]
    assert st["idle_w"] == pytest.approx(0.0, abs=0.01)
    assert st["idle_above_threshold"] is False


def test_stress_override_non_finite_falls_back(monkeypatch):
    """A non-finite override (inf/nan) is ignored (auto-derived floor used) rather than
    corrupting the synthetic samples; the run still completes with a valid outcome."""
    import math as _math
    cycle = _make_cycle(duration_s=3600.0, tail_idle_w=1.0)
    cfg = _cfg(stop_threshold_w=2.0, off_delay=180, min_off_gap=60)
    for bad in (float("inf"), float("nan"), -float("inf")):
        d = _run_stress(cycle, cfg, stress_idle_w=bad)
        st = d["outcome"]["stress"]
        assert st is not None and st.get("enabled") is True
        assert _math.isfinite(st["idle_w"]) and st["idle_w"] >= 0.0


def test_stress_override_huge_clamped():
    """An absurdly large override is clamped to the documented ceiling, never inf."""
    from custom_components.ha_washdata.const import PLAYGROUND_STRESS_MAX_IDLE_W
    cycle = _make_cycle(duration_s=3600.0, tail_idle_w=1.0)
    cfg = _cfg(stop_threshold_w=2.0, off_delay=180, min_off_gap=60)
    d = _run_stress(cycle, cfg, stress_idle_w=1e12)
    st = d["outcome"]["stress"]
    assert st["idle_w"] == pytest.approx(PLAYGROUND_STRESS_MAX_IDLE_W, rel=1e-6)
    assert st["idle_above_threshold"] is True


def test_derive_idle_level_returns_floor_not_mean():
    """_derive_idle_level returns the p7 standby floor, not the contaminated mean."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    readings = []
    t = base
    # Pre-window hot phase
    readings.append((t, 2000.0))
    t += timedelta(seconds=70)
    # 60s window: alternating 3.2W standby and 70W bursts
    for i in range(6):
        readings.append((t, 3.2))
        t += timedelta(seconds=5)
        readings.append((t, 70.0))
        t += timedelta(seconds=5)

    cycle = {
        "id": "floor-test",
        "start_time": base.isoformat(),
        "power_data": [[(r[0] - base).total_seconds(), r[1]] for r in readings],
        "duration": (t - base).total_seconds(),
        "status": "completed",
    }
    store = _store()
    sim = playground._DetailSim(
        cycle, _cfg(), None, store, {}, None,
        stress_tail=True,
    )
    idle_w, fluct_w = sim._derive_idle_level()

    # Contaminated mean ≈ (3.2*6 + 70*6)/12 ≈ 36.6 — very different from p7 ≈ 3.2
    assert idle_w < 10.0, f"expected standby floor ~3.2, got {idle_w:.1f}"
    assert fluct_w > 0.0
