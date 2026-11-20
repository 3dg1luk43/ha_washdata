"""devtools/duration_ratchet_eval.py on a synthetic ECO curve (#531).

The replay through the real matcher is in the slow tier. The corpus curve is
local only; this keeps the harness from rotting in CI and pins its #531 verdict:
from a store whose ECO runs were mostly closed inside the drying, the next run
still ends at the programme's real end.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "devtools" / "duration_ratchet_eval.py"
_spec = importlib.util.spec_from_file_location("wd_duration_ratchet_eval", _PATH)
ratchet = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ratchet
_spec.loader.exec_module(ratchet)


def _curve() -> list[tuple[float, float]]:
    """Heating, washing, pump-outs to 138 min, drying at 3.3 W, a 6.3 W beep at 191.6."""
    pts, t = [], 0.0
    while t <= 192.2 * 60:
        m = t / 60
        if 33 <= m < 47 or 112 <= m < 124:
            p = 1950.0
        elif m < 110:
            p = 30.0
        elif 130 <= m < 131 or 137.5 <= m < 138.5:
            p = 25.0
        elif 191.6 <= m < 192.2:
            p = 6.3
        else:
            p = 3.3
        pts.append((t, p))
        t += 10.0
    return pts


@pytest.mark.slow
def test_a_store_of_cut_off_runs_no_longer_ends_the_next_one_early() -> None:
    res = ratchet.run(_curve(), seed=4, cut=8, cut_min=131.0, cycles=2, hold_min=20.0)
    assert res["ended_early"] == 0 and res["no_completed_record"] == 0
    assert (res["seed_full"], res["seed_cut"]) == (4, 8)
    for row in res["rows"]:
        assert row["end_min"] >= res["real_end_min"] - 1
        assert row["learned_after_min"] >= res["real_end_min"] - 2


def test_a_curve_that_never_reaches_the_stop_is_bad_input(tmp_path: Path) -> None:
    flat = tmp_path / "flat.csv"
    flat.write_text("minutes_since_start,power_w\n0,3.3\n1,3.3\n")
    assert ratchet.main(["--curve", str(flat)]) == 2


def test_a_cut_that_does_not_shorten_the_curve_is_bad_input(tmp_path: Path) -> None:
    curve = tmp_path / "c.csv"
    curve.write_text("minutes_since_start,power_w\n" + "".join(f"{m},{1900 if m < 60 else 3.3}\n" for m in range(90)))
    assert ratchet.main(["--curve", str(curve), "--cut", "3", "--cut-min", "131"]) == 2


def test_an_end_the_manager_discards_is_never_learned_from(monkeypatch) -> None:
    """PR #529 review: a replay whose only completed end is one the manager never
    stores (here a dishwasher pump-out) was stored as the run and learned from."""

    def replay(src, *_a, **_k):
        pump = {"status": "completed", "termination_reason": "smart", "duration": 240.0,
                "start_time": src["start_time"], "power_data": [[0.0, 30.0], [240.0, 0.0]]}
        return {"captured": [pump], "ends": [300.0], "discarded": ["pump_out"],
                "committed": [], "base": None, "idle_above_stop": False, "work_end": None}

    monkeypatch.setattr(ratchet.sle, "replay_one", replay)
    res = ratchet.run(_curve(), seed=3, cycles=2, hold_min=0.0)
    assert res["no_completed_record"] == 2 and res["ended_early"] == 0
    for row in res["rows"]:
        assert "stored_min" not in row
        assert row["learned_after_min"] == row["learned_before_min"]
    assert res["learned_last_min"] == res["learned_first_min"]


@pytest.mark.parametrize(
    ("body", "field"),
    [("0,3.3\n1,oops\n", "power_w"), ("0,3.3\nx,6.3\n", "minutes_since_start"),
     ("0,3.3\n1\n", "power_w"), ("0,nan\n", "power_w")],
    ids=["text-power", "text-minutes", "short-row", "nan"],
)
def test_a_malformed_curve_is_bad_input(tmp_path: Path, capsys, body: str, field: str) -> None:
    """PR #529 review: a value that is no number raised ValueError (a traceback)
    instead of exit 2; the message names the file and the field."""
    bad = tmp_path / "bad.csv"
    bad.write_text("minutes_since_start,power_w\n" + body)
    assert ratchet.main(["--curve", str(bad)]) == 2
    err = capsys.readouterr().err
    assert str(bad) in err and field in err


@pytest.mark.parametrize("minutes", ["1e308", "1e200", "20000", "-20000"])
def test_a_time_no_replay_can_stamp_is_bad_input(tmp_path: Path, capsys, minutes: str) -> None:
    """PR #529 review: 1e308 minutes is finite but infinite in seconds, and the
    replay raised OverflowError (a traceback) stamping it; checked in seconds."""
    bad = tmp_path / "far.csv"
    bad.write_text(f"minutes_since_start,power_w\n0,30\n1,1900\n{minutes},6.3\n")
    assert ratchet.main(["--curve", str(bad), "--seed", "1", "--cycles", "1"]) == 2
    err = capsys.readouterr().err
    assert f"{bad}:4" in err and "minutes_since_start" in err
