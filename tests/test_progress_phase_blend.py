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
"""Phase-resolved ETA blend in progress.compute_progress (Phase 4)."""
from __future__ import annotations

import pytest

from custom_components.ha_washdata.progress import _compute_progress_base, compute_progress


def _base(**kw):
    args = dict(
        device_type="washing_machine", matched_duration=6000.0, duration_so_far=600.0,
        prev_smoothed=0.0, phase_result=None, ml_pct=None,
    )
    args.update(kw)
    return args


def test_none_phase_remaining_is_byte_identical():
    a = _base()
    base = _compute_progress_base(**a)
    blended = compute_progress(**a, phase_remaining_s=None)
    assert base is not None and blended is not None
    assert (blended.progress, blended.smoothed, blended.remaining, blended.total, blended.source) == (
        base.progress, base.smoothed, base.remaining, base.total, base.source
    )


def test_blend_moves_estimate_toward_phase_early():
    a = _base()  # 10% in, linear base remaining = 5400s
    base = compute_progress(**a, phase_remaining_s=None)  # linear baseline
    # phase says much less time left than the linear base
    blended = compute_progress(**a, phase_remaining_s=3000.0)
    assert blended.source == "phase_blend"
    # blend goes through the percent-domain smoothing: progress rises, remaining
    # falls (toward the phase signal), but is re-scaled by matched_duration.
    assert blended.progress > base.progress
    assert blended.remaining < base.remaining
    assert blended.total == pytest.approx(a["duration_so_far"] + blended.remaining)


def test_blend_leans_on_base_late_in_cycle():
    a = _base(duration_so_far=5400.0)  # 90% in, base remaining = 600s
    base = compute_progress(**a, phase_remaining_s=None)
    # phase disagrees a lot, but late in the cycle the base should dominate
    blended = compute_progress(**a, phase_remaining_s=3000.0)
    # blended remaining stays much closer to base than to the raw phase estimate
    assert abs(blended.remaining - base.remaining) < abs(blended.remaining - 3000.0)


@pytest.mark.parametrize("bad", [-100.0, float("nan"), float("inf")])
def test_invalid_phase_remaining_falls_back_to_base(bad):
    a = _base()
    base = _compute_progress_base(**a)
    blended = compute_progress(**a, phase_remaining_s=bad)
    assert blended.remaining == base.remaining
    assert blended.source == base.source


def test_no_matched_duration_returns_none_regardless():
    a = _base(matched_duration=0.0)
    assert compute_progress(**a, phase_remaining_s=1000.0) is None
