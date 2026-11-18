"""devtools/duration_ratchet_eval.py on a synthetic ECO curve (#531).

Replays through the real matcher, so it is in the slow tier. The corpus curve is
local only; this keeps the harness from rotting in CI and pins its #531 verdict:
from a store whose ECO runs were mostly closed inside the drying, the next run
still ends at the programme's real end.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

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
