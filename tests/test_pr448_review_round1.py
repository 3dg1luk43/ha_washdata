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
"""Regressions for the defects found in the PR #448 review.

Each test here pins one thing that shipped broken on the 0.5.7 branch and was
caught in review rather than by the suite. They are grouped in one module
because they share nothing but their provenance.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import DOMAIN
from custom_components.ha_washdata.ml.matching_tuner import _series
from custom_components.ha_washdata.suggestion_engine import detect_standby_above_stop


# --------------------------------------------------------------------------
# ws_api: the standby advisory was computed into a NameError and swallowed
# --------------------------------------------------------------------------
def _call_get_devices(past_cycles, options):
    """Invoke ws_get_devices for one entry and return its device info dict."""
    entry = SimpleNamespace(
        entry_id="e1", title="Washer", data={}, options=dict(options)
    )
    store = SimpleNamespace(
        get_suggestions=lambda: {},
        get_locked_suggestions=lambda: [],
        get_pending_feedback=lambda: {},
        get_past_cycles=lambda: list(past_cycles),
    )
    manager = SimpleNamespace(
        detector=SimpleNamespace(state="off", sub_state=None),
        _current_program=None,
        manual_program_active=False,
        _time_remaining=None,
        _total_duration=None,
        _current_power=None,
        _cycle_progress=None,
        profile_store=store,
        is_user_paused=False,
        recorder=SimpleNamespace(is_recording=False),
    )
    hass = MagicMock()
    hass.config_entries.async_entries.return_value = [entry]
    hass.data = {DOMAIN: {"e1": manager}}
    connection = MagicMock()
    with patch.object(ws_api, "_effective_level", return_value="admin"):
        ws_api.ws_get_devices(hass, connection, {"id": 1, "type": "x"})
    return connection.send_result.call_args[0][1]["devices"][0]


def _idle_cycle(final_w: float, cid: str = "c") -> dict:
    return {
        "id": cid,
        "duration": 3600.0,
        "power_data": [[0.0, 2000.0], [10.0, 2000.0], [20.0, final_w]],
    }


def test_ws_get_devices_actually_reaches_the_standby_advisory():
    """The import lived in another function, so every call raised NameError.

    The surrounding ``except Exception`` swallowed it, which is indistinguishable
    from "no pattern found" - the whole #445 cause-1 card shipped dead.
    """
    dev = _call_get_devices(
        [_idle_cycle(3.4, "a"), _idle_cycle(3.2, "b"), _idle_cycle(3.5, "c")],
        {"stop_threshold_w": 2.56},
    )
    assert dev["standby_above_stop"] is not None
    assert dev["standby_above_stop"]["cycles_above"] == 3
    assert dev["standby_above_stop"]["idle_w"] == pytest.approx(3.4)


def test_ws_get_devices_reports_no_advisory_when_there_is_no_pattern():
    """The key is still set (to None) so a NameError cannot masquerade as silence."""
    dev = _call_get_devices([_idle_cycle(0.0, "a")], {"stop_threshold_w": 2.56})
    assert dev["standby_above_stop"] is None


# --------------------------------------------------------------------------
# suggestion_engine: a cycle stopped by hand mid-wash is not standby evidence
# --------------------------------------------------------------------------
def test_a_cycle_stopped_at_working_power_is_not_standby_evidence():
    """Two force-stops mid-wash used to be reported as "idling at 500 W"."""
    mid_wash = {
        "id": "u",
        "duration": 1200.0,
        "termination_reason": "user",
        "power_data": [[0.0, 2000.0], [10.0, 1800.0], [20.0, 500.0]],
    }
    assert detect_standby_above_stop([mid_wash, dict(mid_wash, id="u2")], 2.56) is None


def test_a_cycle_the_user_stopped_at_standby_still_counts():
    """#445's reporter force-stopped four cycles; those ARE the evidence."""
    res = detect_standby_above_stop(
        [_idle_cycle(3.4, "a"), _idle_cycle(3.3, "b")], 2.56
    )
    assert res is not None and res["cycles_above"] == 2


# --------------------------------------------------------------------------
# matching_tuner: one half-parsed row desynchronised the two arrays
# --------------------------------------------------------------------------
def test_a_row_whose_power_is_unparseable_drops_the_whole_pair():
    """``ts`` used to gain an element ``pw`` did not, so np.interp raised later."""
    cycle = {
        "power_data": [
            [0.0, 10.0],
            [10.0, "nonsense"],
            [20.0, 30.0],
            [30.0, 40.0],
            [40.0, 50.0],
        ]
    }
    ts, pw = _series(cycle)
    assert ts.size == pw.size == 4
    assert list(ts) == [0.0, 20.0, 30.0, 40.0]


def test_an_unbounded_integer_power_drops_the_pair_rather_than_raising():
    """`json` keeps an oversized literal as an int, and float() on one raises."""
    cycle = {
        "power_data": [
            [0.0, 10.0],
            [10.0, 10**400],
            [20.0, 30.0],
            [30.0, 40.0],
            [40.0, 50.0],
        ]
    }
    ts, pw = _series(cycle)
    assert ts.size == pw.size == 4


def test_the_arrays_stay_usable_for_interpolation():
    """The failure mode was downstream: np.interp on unequal arrays."""
    cycle = {"power_data": [[0.0, 1.0], [1.0, None], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]}
    ts, pw = _series(cycle)
    assert np.interp(np.linspace(0.0, 4.0, 10), ts, pw).size == 10


# --------------------------------------------------------------------------
# manager: the banked-tail repair was spawned untracked
# --------------------------------------------------------------------------
def test_the_banked_tail_repair_is_spawned_through_spawn_tracked():
    """It writes to the ProfileStore, so a reload must be able to cancel it.

    ``async_shutdown`` only cancels what is in ``_background_tasks``, and
    ``_spawn_tracked`` is what puts it there. Spawned bare, the repair would walk
    up to 200 stored traces still writing to the store setup had just replaced.
    Asserted on the source because reaching this line needs a fully built manager.
    """
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "manager.py"
    ).read_text()
    assert "self._spawn_tracked(self._async_repair_banked_tails())" in src
    assert "async_create_task(self._async_repair_banked_tails" not in src


# --------------------------------------------------------------------------
# panel: the standby card was pushed after the HTML string was built
# --------------------------------------------------------------------------
def test_the_standby_card_is_pushed_before_the_attention_html_is_built():
    """``attnHtml`` is a plain string, so a later push never reaches the render.

    The card was computed, pushed and then dropped on the floor - dead code that
    no unit test could see, because the array it mutates is correct either way.
    """
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "www"
        / "ha-washdata-panel.js"
    ).read_text()
    push_at = src.index("const sas = dev.standby_above_stop;")
    build_at = src.index("const attnHtml = attn.length")
    assert push_at < build_at, (
        "the standby_above_stop card is pushed onto `attn` after `attnHtml` has "
        "already been joined, so it never renders"
    )


# --------------------------------------------------------------------------
# Round 4: the duration and the score described different quiet spans
# --------------------------------------------------------------------------
def test_the_scored_span_is_the_span_whose_duration_is_reported():
    """A low run split by mid-level readings holds several quiet spans.

    ``_measured_quiet_span_s`` returns the LONGEST; ``points[:resume_idx]`` ends
    on the LAST. Pairing one span's duration with another span's P(end) is what
    let ``_ml_off_delay``'s ``score < 0.4`` filter keep the wrong durations.
    """
    from custom_components.ha_washdata.suggestion_engine import (
        _measured_quiet_span,
        _measured_quiet_span_s,
    )

    # 0-300 s: a long quiet span (the one worth measuring).
    # 300-310 s: one 50 W reading, above stop (2.0) but below active_thr.
    # 310-360 s: a short quiet span, the one nearest the resume.
    pts: list[tuple[float, float]] = [(float(t), 0.5) for t in range(0, 310, 10)]
    pts.append((310.0, 50.0))
    pts += [(float(t), 0.5) for t in range(320, 370, 10)]

    span, end_idx = _measured_quiet_span(pts, 0.0, len(pts), 2.0)
    assert span == pytest.approx(_measured_quiet_span_s(pts, 0.0, len(pts), 2.0))
    assert end_idx is not None
    # The winning span is the long one, so its endpoint is the 300 s sample -
    # not the 360 s one the old prefix ended on.
    assert pts[end_idx][0] == pytest.approx(300.0)
    assert span == pytest.approx(300.0)


def test_a_single_unbroken_quiet_span_still_ends_where_it_ends():
    from custom_components.ha_washdata.suggestion_engine import _measured_quiet_span

    pts: list[tuple[float, float]] = [(0.0, 100.0)] + [
        (float(t), 0.5) for t in range(10, 130, 10)
    ]
    span, end_idx = _measured_quiet_span(pts, 0.0, len(pts), 2.0)
    assert span == pytest.approx(120.0)
    assert end_idx == len(pts) - 1


def test_no_quiet_span_reports_no_endpoint():
    """A run that never goes below the stop threshold is not a pause at all."""
    from custom_components.ha_washdata.suggestion_engine import _measured_quiet_span

    pts: list[tuple[float, float]] = [(float(t), 50.0) for t in range(0, 200, 10)]
    assert _measured_quiet_span(pts, 0.0, len(pts), 2.0) == (0.0, None)


def test_the_scored_pause_prefix_ends_on_the_measured_span():
    """End to end through _scored_pauses: the feature extractor is handed the
    prefix that belongs to the duration it is scored against."""
    from custom_components.ha_washdata.suggestion_engine import MLSuggestionEngine

    seen: list[int] = []

    def _feat(prefix, _expectation):
        seen.append(len(prefix))
        return [0.0]

    pts: list[tuple[float, float]] = [(0.0, 1000.0)]
    pts += [(float(t), 0.5) for t in range(10, 310, 10)]   # long quiet span
    pts.append((310.0, 50.0))                               # splits the low run
    pts += [(float(t), 0.5) for t in range(320, 370, 10)]   # short quiet span
    pts += [(float(t), 1000.0) for t in range(370, 700, 10)]  # sustained resume

    engine = MLSuggestionEngine.__new__(MLSuggestionEngine)
    pauses = engine._scored_pauses(
        pts, {"duration": 3600.0, "energy": 800.0, "peak": 1000.0}, 2.0,
        lambda _f: 0.1, _feat,
    )

    assert pauses, "the long quiet span should be reported as a pause"
    assert seen, "the feature extractor was never called"
    # The prefix ends on the 300 s sample (index 30), so its length is 31 -
    # it must not run on to the 360 s sample the resume follows.
    assert seen[0] == 31, f"scored a prefix of {seen[0]} samples, expected 31"


# --------------------------------------------------------------------------
# Round 4: _safe_offset and the prefix bandwidth default
# --------------------------------------------------------------------------
def test_safe_offset_survives_an_unbounded_integer():
    """It is reached from the banked-tail repair, whose caller aborts before
    clearing the repair marker - so one bad row would re-fail on every setup."""
    from custom_components.ha_washdata.profile_store import _safe_offset

    assert _safe_offset(10**400) is None
    assert _safe_offset(float("inf")) is None
    assert _safe_offset("nope") is None
    assert _safe_offset(12.5) == pytest.approx(12.5)


def test_prefix_scoring_and_stage3_share_one_bandwidth_default():
    """Both read the same unmutated config in one match (register item 309)."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "analysis.py"
    ).read_text()
    assert 'config.get("dtw_bandwidth", 0.1)' not in src
    assert src.count('config.get("dtw_bandwidth", DEFAULT_DTW_BANDWIDTH)') == 2


# --------------------------------------------------------------------------
# Round 4: the setup-time presence flush is the same code as the event one
# --------------------------------------------------------------------------
def test_both_presence_flush_paths_go_through_one_body():
    """They were two copies of the same loop and drifted: only one recorded that
    a Live Activity had started, so a queued live card delivered at listener
    (re-)attach left the activity frozen on the phone (#446)."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "manager.py"
    ).read_text()
    assert src.count("def _flush_pending_notifications") == 1
    assert src.count("self._flush_pending_notifications(") == 2


# --------------------------------------------------------------------------
# Round 6, re-measured: both proposed gate changes were measured worse
# --------------------------------------------------------------------------
def test_the_end_gate_is_deliberately_not_gated_on_confidence():
    """Adding the confidence check was tried and measured pure cost.

    ``devtools/end_gate_eval.py`` over 221 replayed real cycles: it moves 2 of
    them (+10.5 min and +21 min of end lag) while early ends (1.42% / 0.00%) and
    splits (3.32%) stay exactly where they were - no split prevented, no early
    end prevented. Smart Termination checks confidence because it ENDS a cycle
    on a prediction; this rule only shortens a wait already past the programme's
    expected end. Pinned so the next review round does not re-add it silently.
    """
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "cycle_detector.py"
    ).read_text()
    block = src.split("--- FALLBACK TIMEOUT CHECK ---", 1)[1].split(
        "gate_window = self._config.off_delay", 1
    )[0]
    # Comments stripped: the block explains at length WHY the confidence check is
    # absent, so a naive substring search finds it in the prose.
    code = "\n".join(
        ln for ln in block.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "match_confidence_threshold" not in code, (
        "the confidence check was measured as pure cost (2 cycles delayed, no "
        "split or early end prevented) - see devtools/end_gate_eval.py"
    )
    # The ambiguity guards are still consulted - since item 330 they raise the
    # bar to the longest plausible candidate rather than refusing outright, and
    # they still refuse when there is no candidate duration to compare.
    assert "_match_prefix_ambiguous" in code
    assert "_match_ambiguous" in code
    assert "_longest_candidate_duration" in code


def test_a_sole_surviving_candidate_still_bypasses_persistence():
    """Requiring a real runner-up was measured worse than the sentinel.

    ``devtools/decisive_margin_eval.py`` over 1977 mid-cycle checkpoints: a
    single surviving candidate occurs at 2.58% of them and is the correct
    programme 94.0% (47/50) of the time, against 77.8% (669/860) for the
    real-margin bypass it would have been held to. Stage 1/2 rejecting every
    other profile is evidence, not the absence of it.
    """
    from custom_components.ha_washdata.const import MATCH_DECISIVE_MARGIN

    def bypasses(runner_up: float | None, confidence: float, current: float) -> bool:
        margin = 1.0 if runner_up is None else confidence - runner_up
        return margin > MATCH_DECISIVE_MARGIN and confidence > current

    assert bypasses(0.40, 0.80, 0.50) is True   # genuinely decisive
    assert bypasses(0.75, 0.80, 0.50) is False  # crowded field, unchanged
    assert bypasses(None, 0.55, 0.0) is True    # sole survivor: 94% correct


def test_the_decisive_margin_bypass_does_not_require_a_runner_up():
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "manager.py"
    ).read_text()
    block = src.split("Decisive Margin Override", 1)[1].split("should_switch = True", 1)[0]
    assert "_runner_up is not None" not in block, (
        "the sole-surviving-candidate case is 94% correct - see "
        "devtools/decisive_margin_eval.py"
    )


def test_both_measurement_harnesses_are_checked_in():
    """The whole reason these two were unmeasurable is that item 306's harness
    was never committed. Do not let that happen again."""
    from pathlib import Path

    devtools = Path(__file__).resolve().parents[1] / "devtools"
    assert (devtools / "end_gate_eval.py").is_file()
    assert (devtools / "decisive_margin_eval.py").is_file()


def test_a_matched_dishwasher_profile_lowers_the_minimum_duration_floor():
    """The 1800 s floor is a stand-in for the knowledge a match supplies.

    Blanket, it is wrong for real hardware: the community catalogue carries a
    6.0 min Smeg "Delay- prewash", which a 30 min floor defers by half an hour.
    A matched profile may only ever LOWER the floor, never raise it - the
    39.4 min Electrolux "Rapido" already clears it and is unaffected.
    """
    from custom_components.ha_washdata.const import DISHWASHER_MIN_CYCLE_DURATION_S

    def floor(
        matched: str | None,
        expected: float,
        conf: float = 0.9,
        ambiguous: bool = False,
        prefix_ambiguous: bool = False,
        threshold: float = 0.4,
    ) -> float:
        out = DISHWASHER_MIN_CYCLE_DURATION_S
        if (
            matched
            and expected > 0
            and conf >= threshold
            and not ambiguous
            and not prefix_ambiguous
        ):
            out = min(out, float(expected))
        return out

    # An UNTRUSTED match may not lower it: this is an anti-premature-end guard,
    # and a weak match to a short look-alike is how a fill dip ends the cycle.
    assert floor("Delay- prewash", 360.0, conf=0.2) == DISHWASHER_MIN_CYCLE_DURATION_S
    assert floor("Delay- prewash", 360.0, ambiguous=True) == DISHWASHER_MIN_CYCLE_DURATION_S
    assert (
        floor("Delay- prewash", 360.0, prefix_ambiguous=True)
        == DISHWASHER_MIN_CYCLE_DURATION_S
    )

    # Unmatched: the blanket floor still applies.
    assert floor(None, 0.0) == DISHWASHER_MIN_CYCLE_DURATION_S
    # A programme shorter than the floor lowers it to its own length.
    assert floor("Delay- prewash", 360.0) == 360.0
    # One that already clears the floor is unaffected...
    assert floor("Rapido", 2364.0) == DISHWASHER_MIN_CYCLE_DURATION_S
    # ...and a long one cannot raise it above the constant.
    assert floor("ECO", 13962.0) == DISHWASHER_MIN_CYCLE_DURATION_S


# --------------------------------------------------------------------------
# Round 8: my own lock fix put settings saves behind ML training
# --------------------------------------------------------------------------
def test_option_writers_do_not_queue_behind_the_long_background_tasks():
    """`_entry_write_lock` is held for the WHOLE run of the detached tasks.

    `_reprocess_task`, `_ml_training_task` and `_rebuild_envelopes_task` each
    acquire it and hold it across their entire multi-await body. Putting a
    Settings save behind that lock - which the round-7 fix did - makes the save
    block for as long as the task runs, minutes on a slow host, with no
    feedback. The option writers need mutual exclusion only against each other.
    """
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "ws_api.py"
    ).read_text()
    assert "def _entry_options_lock(" in src
    # The three option writers take the options lock, never the write lock.
    for marker in (
        'lock = _entry_options_lock(hass, msg["entry_id"])',   # ws_set_options
        'async with _entry_options_lock(hass, entry_id):',      # ws_apply_suggestions
        'async with _entry_options_lock(hass, msg["entry_id"]):',  # store_download
    ):
        assert marker in src, marker
    # Lock ORDER where both are held must be write -> options. The import
    # handlers are the only place both are taken; assert the options lock is
    # acquired INSIDE their write-lock block, not around it.
    for handler in ("async def ws_import_config(", "async def ws_import_config_selective("):
        body = src.split(handler, 1)[1].split("\n@websocket_api", 1)[0]
        w = body.find("_entry_write_lock(")
        o = body.find("_entry_options_lock(")
        assert w != -1 and o != -1, handler
        assert w < o, f"{handler}: options lock must be nested inside the write lock"


def test_the_self_unmatchable_advisory_uses_the_envelope_status_filter():
    """`avg_duration` is built from completed/force_stopped cycles only.

    Judging an interrupted cycle against an average it never contributed to
    flags a merely-cut-off run as needing a re-label or a split.
    """
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "profile_store.py"
    ).read_text()
    block = src.split("def _self_unmatchable_cycles", 1)[1].split("profiles = self.get_profiles()", 1)[0]
    assert 'status' in block and '"completed", "force_stopped"' in block


def test_the_banked_tail_repair_measures_each_profile_once():
    """Per-cycle it is quadratic AND order-dependent: a cycle repaired earlier
    in the loop changes the history later calls measure."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "profile_store.py"
    ).read_text()
    block = src.split("async def async_repair_banked_tails", 1)[1].split(
        "def _apply_repaired_duration", 1
    )[0]
    assert "quiet_by_profile" in block
    assert block.count("self.profile_terminal_quiet_seconds(") == 1


# --------------------------------------------------------------------------
# Round 9: a keepalive during a sensor outage is not observed quiet
# --------------------------------------------------------------------------
def test_an_unobserved_keepalive_still_resets_the_gapfree_tally():
    """The synthetic exemption rests on "the watchdog resynced first".

    `_resync_power_from_state` returns early when the sensor is unavailable,
    unknown or non-finite, but the watchdog injects on the silence interval
    alone - so during a real outage every keepalive was exempt and
    `_time_below_threshold_gapfree` grew through quiet nobody saw. That tally
    feeds the dishwasher end-spike quiet release and the ENDING hard finalize,
    both of which can only SHORTEN the wait, so a plug dropping offline could
    finalize a dishwasher before its terminal pump-out.
    """
    from datetime import datetime, timedelta, timezone

    from custom_components.ha_washdata.cycle_detector import (
        CycleDetector,
        CycleDetectorConfig,
    )

    def _det():
        cfg = CycleDetectorConfig(min_power=2.0, off_delay=60, stop_threshold_w=2.0)
        d = CycleDetector(cfg, lambda a, b: None, lambda c: None)
        base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        # A fast-cadence plug, so the outage ceiling is the 60 s floor.
        for i in range(30):
            d.process_reading(0.0, base + timedelta(seconds=i * 2))
        return d, base + timedelta(seconds=60)

    # The dt that MATTERS is a short one. The watchdog injects once per
    # watchdog_interval (floor 30 s) and the outage ceiling is at least 60 s, so
    # in a real outage every step sits UNDER the ceiling. An earlier version of
    # this test used dt=600 s, which is over the ceiling, and so passed against
    # a fix that did nothing in the case that actually occurs.
    for dt_s in (30, 45, 600):
        # Observed: the interval was seen, so the tally keeps accumulating
        # (the #424/#427 behaviour the exemption exists for).
        det, t = _det()
        before = det._time_below_threshold_gapfree
        det.process_reading(
            0.0, t + timedelta(seconds=dt_s), synthetic=True, observed=True
        )
        assert det._time_below_threshold_gapfree > before, dt_s

        # Unread sensor: an outage at ANY step size, so the tally resets.
        det, t = _det()
        det.process_reading(
            0.0, t + timedelta(seconds=dt_s), synthetic=True, observed=False
        )
        assert det._time_below_threshold_gapfree == 0.0, dt_s


def test_a_real_reading_after_an_outage_still_resets_regardless_of_observed():
    """`observed` only ever qualifies the synthetic exemption; a genuine
    sensor reading across a hole resets as it always did."""
    from datetime import datetime, timedelta, timezone

    from custom_components.ha_washdata.cycle_detector import (
        CycleDetector,
        CycleDetectorConfig,
    )

    cfg = CycleDetectorConfig(min_power=2.0, off_delay=60, stop_threshold_w=2.0)
    det = CycleDetector(cfg, lambda a, b: None, lambda c: None)
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    for i in range(30):
        det.process_reading(0.0, base + timedelta(seconds=i * 2))
    det.process_reading(0.0, base + timedelta(seconds=660))  # real, big hole
    assert det._time_below_threshold_gapfree == 0.0


# --------------------------------------------------------------------------
# Round 14: an imported old backup never got the banked-tail repair
# --------------------------------------------------------------------------
def test_an_old_export_re_arms_the_banked_tail_repair():
    """Only `_async_migrate_func` sets the marker, and imports bypass it.

    A user restoring a pre-0.5.7 backup kept the banked confirmation delay
    permanently: no marker, so `banked_tail_repair_pending()` is False and
    `manager.async_setup` never schedules the repair. Those durations feed
    avg_duration, the ETA and the Smart Termination gate.
    """
    from custom_components.ha_washdata.profile_store import (
        _export_predates_banked_tail_repair,
    )

    assert _export_predates_banked_tail_repair({"version": 12}) is True
    assert _export_predates_banked_tail_repair({"version": 1}) is True
    assert _export_predates_banked_tail_repair({"version": 13}) is False
    assert _export_predates_banked_tail_repair({"version": 14}) is False
    # Unreadable version is treated as old: an idempotent repair on an already
    # repaired history costs one pass, skipping it on an unrepaired one is
    # permanent. (An oversized integer literal is NOT unreadable - Python ints
    # are unbounded, so it parses and simply reads as a very new version.)
    for bad in ({}, {"version": None}, {"version": "x"}, {"version": []}):
        assert _export_predates_banked_tail_repair(bad) is True, bad
    assert _export_predates_banked_tail_repair({"version": 10**400}) is False


def test_the_standby_card_opens_the_section_that_holds_its_setting():
    """The card names the Stop Threshold; `goto-conflicts` only flips the tab
    and keeps whatever section was last open."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "www"
        / "ha-washdata-panel.js"
    ).read_text()
    card = src[src.index("const sas = dev.standby_above_stop;"):]
    card = card[: card.index("const attnHtml")]
    assert 'data-action="goto-standby"' in card
    assert "a === 'goto-standby'" in src
    handler = src[src.index("a === 'goto-standby'"):][:900]
    assert "_settingsSec = 'detection'" in handler


# --------------------------------------------------------------------------
# Round 15 (deferred, then taken): the two panel findings
# --------------------------------------------------------------------------
def _panel_src() -> str:
    from pathlib import Path

    return (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "www"
        / "ha-washdata-panel.js"
    ).read_text()


def test_scrollers_nested_in_a_dialog_are_keyed_apart_from_the_page():
    """`_SCROLLERS` covers `.wd-table-wrap` and `.wd-sd-tree` as well as
    `.wd-modal`, and `_eachScroller` keyed them from all shadow-root matches
    with no dialog scope - so the Cleanup table, the history-import review
    table and the export-select / import-wizard / store-share-device trees all
    shared one offset across different dialogs.
    """
    src = _panel_src()
    each = src[src.index("_eachScroller(sels, fn) {"):]
    each = each[: each.index("\n  _captureScroll")]
    assert "closest('.wd-modal')" in each, "modal-nested scrollers are not scoped"
    assert "_MODAL_KEY_PREFIX" in each
    # ...and the render-time drop must clear those scoped keys too, not just
    # the '.wd-modal|' ones.
    render = src[src.index("const modalKey = this._modalNavKey();"):][:800]
    assert "_MODAL_KEY_PREFIX" in render


def test_a_contributed_appliance_is_saved_to_the_device_that_asked_for_it():
    """The contribute popup is user-paced, so the panel selection can move
    while it is open. `_saveStoreOptions` targets whatever is selected NOW, so
    an ungated save stamps the appliance identity onto another entry and
    reloads it, while the device that asked gets nothing."""
    src = _panel_src()
    # Both popup-open sites record the originating entry...
    assert src.count("this._storeContribEid = eid") == 2
    # ...and both message handlers gate on it.
    listener = src[src.index("washdata-device-created"):]
    listener = listener[: listener.index("washdata-connect")]
    assert listener.count("_storeContribEid") == 2
    assert listener.count("_isActiveEntry") == 2


# --------------------------------------------------------------------------
# manager: the banked-tail repair raced the rest of async_setup (round 16)
# --------------------------------------------------------------------------
def test_the_banked_tail_repair_is_spawned_after_setup_rewrites_the_cycles():
    """Its cycle loop takes no awaits, but the envelope rebuild after it does.

    Every await hands the loop back to the rest of ``async_setup``, which rewrites
    the very cycles the repair is rebuilding from:
    ``async_repair_profile_samples`` can drop a profile or re-point its sample,
    and ``async_migrate_cycles_to_compressed`` replaces ``power_data`` wholesale.
    Spawning last also means the legacy ISO-offset traces that migration converts
    are trimmable by the time the repair sees them. Asserted on source order
    because the race needs a fully built manager and real executor timing.
    """
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "manager.py"
    ).read_text()
    spawn = src.index("self._spawn_tracked(self._async_repair_banked_tails())")
    for later in (
        "await self.profile_store.async_repair_profile_samples()",
        "await self.profile_store.async_migrate_cycles_to_compressed()",
        "await self._setup_maintenance_scheduler()",
    ):
        assert src.index(later) < spawn, f"{later} must run before the repair spawns"


# --------------------------------------------------------------------------
# cycle_detector: the longest-candidate bound outlived its match (round 17)
# --------------------------------------------------------------------------
def _det_with_bound(bound: float = 4200.0):
    from custom_components.ha_washdata.cycle_detector import (
        CycleDetector,
        CycleDetectorConfig,
    )

    cfg = CycleDetectorConfig(min_power=2.0, off_delay=60, stop_threshold_w=2.0)
    det = CycleDetector(cfg, lambda a, b: None, lambda c: None)
    det._longest_candidate_duration = bound
    det._match_ambiguous = True
    det._expected_duration = 3600.0
    return det


def test_reset_clears_the_longest_candidate_bound():
    """Element 12 belongs with elements 9-11, which ``reset`` already clears.

    Its own comment claims a stale value can never license a shortening for a
    different match, and that held only for the tuple path. The dangerous shape
    is a SMALL positive bound: not greater than ``_expected_duration`` so the bar
    is not raised, and not <= 0 so the "no information" refusal never fires.
    """
    from custom_components.ha_washdata.const import STATE_OFF

    det = _det_with_bound(1200.0)
    det.reset(STATE_OFF)
    assert det._longest_candidate_duration == 0.0


def test_the_bound_survives_a_snapshot_round_trip():
    """``restore_state_snapshot`` restores the ambiguity flags the ENDING gate
    reads, so leaving the bound alone pairs them with a previous cycle's value."""
    det = _det_with_bound(4200.0)
    snap = det.get_state_snapshot()
    assert snap["longest_candidate_duration"] == 4200.0

    other = _det_with_bound(999.0)
    other.restore_state_snapshot(snap)
    assert other._longest_candidate_duration == 4200.0


def test_a_snapshot_without_the_bound_restores_no_information():
    """A snapshot written before this key existed must land on 0.0 - the ENDING
    gate reads a non-positive bound as "no information" and keeps the old
    refusal, which is the safe direction. The stale in-memory value must not
    survive in its place."""
    det = _det_with_bound(4200.0)
    snap = det.get_state_snapshot()
    snap.pop("longest_candidate_duration")

    other = _det_with_bound(999.0)
    other.restore_state_snapshot(snap)
    assert other._longest_candidate_duration == 0.0


# --------------------------------------------------------------------------
# profile_store: the artifact refresh mutated self._data from an executor
# --------------------------------------------------------------------------
def test_the_artifact_collector_does_not_mutate_in_the_executor():
    """It runs in an executor over dicts inside ``self._data``. Another coroutine
    can reach ``async_save`` while it is awaited, and adding or removing a key
    from another thread mid-serialisation raises ``dictionary changed size
    during iteration`` on an unrelated save. The worker now only computes.
    """
    from custom_components.ha_washdata.profile_store import ProfileStore

    store = MagicMock()
    store._logger = MagicMock()
    stale = {"id": "c1", "profile_name": None, "artifacts": [{"type": "spike"}]}
    store.iter_stored_cycles.return_value = [stale]
    store._collect_cycle_artifact_updates = (
        ProfileStore._collect_cycle_artifact_updates.__get__(store, ProfileStore)
    )

    pending = store._collect_cycle_artifact_updates()

    assert pending == [(stale, [])]
    assert "artifacts" in stale, "the worker mutated the cycle from the executor"
    assert ProfileStore._apply_cycle_artifact_updates(pending) == 1
    assert "artifacts" not in stale


# --------------------------------------------------------------------------
# __init__: the legacy migration seeded the max duration ratio (round 17)
# --------------------------------------------------------------------------
def test_the_legacy_migration_no_longer_seeds_the_max_duration_ratio():
    """Seeding it is what produced ``_OLD_SEEDED_MAX_DURATION_RATIO`` and the two
    heal sites: entries took the then-default 1.5 as an explicit value, so item
    311's widening reached none of them. Left absent, every reader falls back to
    the constant and the next change to it actually lands.
    """
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "__init__.py"
    ).read_text()
    assert (
        "options.setdefault(\n"
        "        CONF_PROFILE_MATCH_MAX_DURATION_RATIO, "
        "DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO\n"
        "    )"
    ) not in src


def test_the_panel_and_python_agree_on_the_max_duration_ratio_default():
    """Item 311 raised it to 1.8 in Python and the panel schema stayed at 1.5.

    While the legacy migration seeded the key that never showed, because every
    entry carried an explicit value. Removing the seed (round 17) exposes it:
    an entry without the key would render 1.5 while the matcher uses 1.8. The
    constant is published through ``_resolved_option_defaults`` now, and the JS
    literals - the fallback when that payload is unavailable - have to agree.
    """
    from pathlib import Path

    from custom_components.ha_washdata.const import (
        DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO,
    )

    assert (
        ws_api._resolved_option_defaults("washing_machine")[
            "profile_match_max_duration_ratio"
        ]
        == DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO
    )

    panel = (
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "ha_washdata"
        / "www"
        / "ha-washdata-panel.js"
    ).read_text()
    import re

    # Numeric literals only: the key also appears in the cross-field conflict
    # rule, which carries no default.
    literals = re.findall(
        r"profile_match_max_duration_ratio:\s*([0-9]+(?:\.[0-9]+)?)\s*,", panel
    ) + re.findall(
        r"key: 'profile_match_max_duration_ratio'[^\n]*?\bdef:\s*"
        r"([0-9]+(?:\.[0-9]+)?)",
        panel,
    )
    assert literals, "no panel default found for the key"
    for raw in literals:
        assert float(raw) == DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO, (
            f"stale panel default {raw}, Python says "
            f"{DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO}"
        )
