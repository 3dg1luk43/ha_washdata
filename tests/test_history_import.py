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
"""Historical power-data import (issue #344).

The fixture ``tests/fixtures/history_import_sample.csv`` is the Home Assistant history
export attached to issue #344 by the reporter (values rounded to 2 dp to keep the file
small). It is the reference case for the whole feature: 2358 rows in which six months of
hourly long-term-statistics averages precede ten days of real 5-second data, plus the
``unavailable``/``unknown`` rows a plug produces when it drops off the network.

Its measured outcome - 4 completed cycles of 74.4 / 48.3 / 95.5 / 45.8 minutes - is
asserted here, together with the three failure modes that were measured while designing
the pre-pass and that a "simplification" would silently reintroduce:

* feeding the raw stream to one detector produces multi-day ``force_stopped`` junk;
* dropping isolated sparse samples deletes the terminal 0 W row that marks each cycle end;
* trimming a block's trailing edge eats a real cycle's low-power tail.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from custom_components.ha_washdata import history_import as hi
from custom_components.ha_washdata.const import (
    DEVICE_TYPE_DISHWASHER,
    DEVICE_TYPE_PUMP,
    DEVICE_TYPE_WASHING_MACHINE,
)
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
)

FIXTURE = Path(__file__).parent / "fixtures" / "history_import_sample.csv"
ENTITY = "sensor.waschmaschineplug_power"
T0 = datetime(2026, 5, 1, 8, 0, tzinfo=timezone.utc)


def washer_config(**overrides) -> CycleDetectorConfig:
    """The washer defaults the reporter's device runs with."""
    base = {
        "min_power": 2.0,
        "off_delay": 300,
        "device_type": DEVICE_TYPE_WASHING_MACHINE,
        "completion_min_seconds": 600,
        "min_off_gap": 480,
        "start_energy_threshold": 0.2,
    }
    base.update(overrides)
    return CycleDetectorConfig(**base)


def build_stream(cycles: int = 2, idle_style: str = "sparse", idle_s: float = 1500.0):
    """``cycles`` x 40-minute runs at 5 s cadence, separated by an idle stretch.

    ``idle_style='sparse'`` expresses the idle time the way a change-based sensor does -
    a single 0 W row, then nothing until the next cycle. ``'dense'`` emits the 0 W rows a
    polling sensor would. Detection must not care which.
    """
    samples: list[tuple[datetime, float]] = []
    elapsed = 0.0

    def add(step: float, watts: float) -> None:
        nonlocal elapsed
        elapsed += step
        samples.append((T0 + timedelta(seconds=elapsed), watts))

    for index in range(cycles):
        for i in range(480):
            add(5.0, 1800.0 if i < 120 else 400.0)
        add(5.0, 0.0)
        if index < cycles - 1:
            if idle_style == "dense":
                for _ in range(int(idle_s // 5)):
                    add(5.0, 0.0)
            else:
                add(idle_s, 0.0)
    return samples


def replay_blocks(samples, config):
    """The production pipeline: blocks -> gates -> densify -> replay."""
    runner = hi.build_scan(samples, config, sampling_interval_s=5.0)
    assert not isinstance(runner, dict), runner
    while not runner.finished:
        runner.step(4000)
    return runner.finalize()


# ─── Parsing ──────────────────────────────────────────────────────────────────


def test_parse_ha_history_export():
    parsed = hi.parse_history_csv(FIXTURE.read_text(), entity_id=ENTITY)
    assert not isinstance(parsed, dict), parsed
    report = parsed.report()
    assert report["rows_total"] == 2358
    assert report["entities"] == [ENTITY]
    # The four `unavailable` and four `unknown` rows survive as stream breaks rather than
    # being dropped: dropping them would carry the previous value across the hole.
    assert report["breaks"] == 8
    assert len(parsed.readings) == 2350
    assert report["peak_w"] == pytest.approx(2165.0)
    assert report["rows_duplicate"] == 0
    assert report["rows_other_entity"] == 0


def test_parse_accepts_semicolons_bom_and_header_aliases():
    text = (
        "﻿entity_id;value;last_updated\r\n"
        f"{ENTITY};12,5;2026-05-01T08:00:00Z\r\n"
        f"{ENTITY};900;2026-05-01T08:00:05Z\r\n"
    )
    parsed = hi.parse_history_csv(text, entity_id=ENTITY)
    assert not isinstance(parsed, dict), parsed
    assert [round(p, 1) for _, p in parsed.readings] == [12.5, 900.0]


def test_parse_filters_other_entities_and_sorts():
    text = "\n".join(
        [
            "entity_id,state,last_changed",
            f"{ENTITY},100,2026-05-01T08:00:10Z",
            "sensor.other_plug_power,9999,2026-05-01T08:00:00Z",
            f"{ENTITY},50,2026-05-01T08:00:00Z",
            f"{ENTITY},50,2026-05-01T08:00:00Z",
            f"{ENTITY},unavailable,2026-05-01T08:00:20Z",
        ]
    )
    parsed = hi.parse_history_csv(text, entity_id=ENTITY)
    assert not isinstance(parsed, dict), parsed
    assert [p for _, p in parsed.readings] == [50.0, 100.0]  # sorted, deduplicated
    assert parsed.rows_other_entity == 1
    assert parsed.rows_duplicate == 1
    assert parsed.samples[-1][1] is None  # the break is kept, last


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("", "empty_file"),
        ("just some text\n", "missing_columns"),
        ("entity_id,state,last_changed\n", "no_readings"),
    ],
)
def test_parse_error_markers(text, error):
    assert hi.parse_history_csv(text) == {"error": error} or hi.parse_history_csv(text)["error"] == error


def test_parse_reports_missing_entity_rather_than_empty_result():
    text = f"entity_id,state,last_changed\nsensor.other,5,2026-05-01T08:00:00Z\n"
    result = hi.parse_history_csv(text, entity_id=ENTITY)
    assert isinstance(result, dict)
    assert result["error"] == "entity_not_in_file"
    assert result["entities"] == ["sensor.other"]


def test_parse_row_cap_truncates_instead_of_exploding():
    rows = ["entity_id,state,last_changed"] + [
        f"{ENTITY},100,2026-05-01T08:{i // 60:02d}:{i % 60:02d}Z" for i in range(50)
    ]
    parsed = hi.parse_history_csv("\n".join(rows), entity_id=ENTITY, max_rows=10)
    assert not isinstance(parsed, dict), parsed
    assert parsed.truncated is True
    assert len(parsed.samples) == 10


def test_samples_from_readings_handles_recorder_shape():
    samples = hi.samples_from_readings([(1_800_000_100.0, 5.0), (1_800_000_000.0, "bad"), (1_800_000_050.0, 7.0)])
    assert [p for _, p in samples] == [7.0, 5.0]


# ─── The reference case ───────────────────────────────────────────────────────


def test_reporter_export_yields_four_clean_cycles():
    parsed = hi.parse_history_csv(FIXTURE.read_text(), entity_id=ENTITY)
    result = replay_blocks(parsed.samples, washer_config())

    durations = [round(seg["duration_s"] / 60.0, 1) for seg in result["segments"]]
    assert durations == [74.4, 48.3, 95.5, 45.8]
    assert [seg["status"] for seg in result["segments"]] == ["completed"] * 4
    assert all(seg["accept"] for seg in result["segments"])
    assert result["truncated_blocks"] == 0
    assert result["capped"] is False
    # Every one of the six months of hourly averages is accounted for as a skipped span.
    assert len(result["skipped"]) == 120
    assert {item["reason"] for item in result["skipped"]} == {"idle"}
    # The dense window is the last ten days; nothing before it survives the gates.
    assert all(seg["start_time"] >= "2026-07-18" for seg in result["segments"])


def test_energy_is_integrated_per_segment():
    parsed = hi.parse_history_csv(FIXTURE.read_text(), entity_id=ENTITY)
    result = replay_blocks(parsed.samples, washer_config())
    energies = [seg["energy_wh"] for seg in result["segments"]]
    assert all(e > 0 for e in energies)
    # The first run is the hot wash: several times the energy of the others.
    assert energies[0] > 900.0
    assert max(energies[1:]) < energies[0]


def test_raw_replay_without_the_pre_pass_produces_junk():
    """Why the pre-pass exists: one detector over the whole stream cannot cope.

    A change-based history emits no rows while the appliance sits at 0 W, so the detector
    never sees the readings that expire a cycle and its outage handling force-stops
    instead. This is the behaviour the block pre-pass replaces; if this test ever starts
    reporting clean cycles, the pre-pass is no longer load-bearing.
    """
    parsed = hi.parse_history_csv(FIXTURE.read_text(), entity_id=ENTITY)
    captured: list[dict] = []
    detector = CycleDetector(
        washer_config(), lambda *_: None, captured.append, profile_matcher=None
    )
    for timestamp, power in parsed.readings:
        detector.process_reading(power, timestamp)

    assert len(captured) > 10
    assert all(c["status"] == "force_stopped" for c in captured)
    assert max(c["duration"] for c in captured) > 24 * 3600


def test_dropping_isolated_sparse_samples_shortens_a_real_cycle():
    """The pre-filter that must never be added.

    Discarding samples whose neighbours are both far away looks like the tidy way to
    delete the hourly-average region. It also deletes the terminal 0 W row of every
    cycle, because a change-based sensor reports 0 once and then goes silent. The
    floor-independent cut rule means the blocks still come out right, so the damage is
    quiet rather than obvious: the third wash loses the nine minutes between its last
    running sample and the 0 W row that ended it.
    """
    parsed = hi.parse_history_csv(FIXTURE.read_text(), entity_id=ENTITY)
    readings = parsed.readings
    thinned: list[tuple[datetime, float]] = []
    for i, (timestamp, power) in enumerate(readings):
        before = (timestamp - readings[i - 1][0]).total_seconds() if i else float("inf")
        after = (readings[i + 1][0] - timestamp).total_seconds() if i < len(readings) - 1 else float("inf")
        if before <= 60.0 or after <= 60.0:
            thinned.append((timestamp, power))

    assert len(thinned) < len(readings)
    intact = [round(s["duration_s"] / 60.0, 1) for s in replay_blocks(parsed.samples, washer_config())["segments"]]
    damaged = [round(s["duration_s"] / 60.0, 1) for s in replay_blocks(thinned, washer_config())["segments"]]
    assert intact == [74.4, 48.3, 95.5, 45.8]
    assert damaged == [74.4, 48.3, 86.4, 45.8]


def test_trailing_edge_is_not_trimmed():
    """Leading debris is dropped; the trailing low-power tail is part of the cycle."""
    parsed = hi.parse_history_csv(FIXTURE.read_text(), entity_id=ENTITY)
    blocks, _ = hi.find_activity_blocks(parsed.samples, washer_config())
    third = blocks[2]
    trimmed = hi.trim_leading_debris(third)
    assert trimmed.samples[-1] == third.samples[-1]
    # The 95.5-minute wash ends on an isolated 0 W row 906 s after its body; a symmetric
    # trim would remove it and shorten the cycle to 86.4 minutes.
    assert (third.samples[-1][0] - third.samples[-2][0]).total_seconds() > 60.0


# ─── Block segmentation rules ─────────────────────────────────────────────────


def test_cut_threshold_follows_the_device_type():
    assert hi.cut_threshold_s(washer_config()) == 780.0
    dishwasher = washer_config(device_type=DEVICE_TYPE_DISHWASHER, min_off_gap=3600, off_delay=1800)
    # A dishwasher's drying pause may not be mistaken for the end of the cycle.
    assert hi.cut_threshold_s(dishwasher) == 5400.0


@pytest.mark.parametrize("idle_style", ["sparse", "dense"])
def test_reporting_style_does_not_change_the_result(idle_style):
    result = replay_blocks(build_stream(idle_style=idle_style), washer_config())
    assert [round(s["duration_s"] / 60.0, 1) for s in result["segments"]] == [39.9, 39.9]
    assert all(s["status"] == "completed" for s in result["segments"])


def test_densifying_a_quiet_gap_separates_two_cycles():
    """A gap shorter than the block cut threshold must still end a cycle.

    The detector resets its gap-free quiet tally when a step exceeds the outage ceiling,
    because unobserved time may not be credited as quiet. On a change-based history that
    would merge two cycles 10 minutes apart, so the quiet a live sensor would have
    reported is re-inserted first.
    """
    config = washer_config()
    samples = build_stream(idle_s=600.0)  # below the 780 s cut threshold
    blocks, _ = hi.find_activity_blocks(samples, config)
    assert len(blocks) == 1  # too short to cut; the detector must do the separating

    raw = hi.StreamSegmenter(blocks[0].samples, config)
    raw.step(0, len(blocks[0].samples))
    raw.flush_tail()
    assert [round(c["duration"] / 60.0, 1) for c in raw.captured] == [90.0]

    dense = hi.densify_quiet_gaps(blocks[0], config)
    seg = hi.StreamSegmenter(dense, config)
    seg.step(0, len(dense))
    seg.flush_tail()
    assert [round(c["duration"] / 60.0, 1) for c in seg.captured] == [39.9, 39.9]


def test_zero_stop_threshold_still_accrues_quiet_and_densifies():
    """``stop_threshold_w == 0`` is a valid (if degenerate) panel value - the Stop
    Threshold field allows ``min=0``. With no positive quiet level nothing reads as
    quiet (power is clamped >= 0), so block quiet-accrual and gap densification both
    silently no-op and two cycles a few minutes apart inside one block could merge. A
    small positive floor keeps them working; any positive configured value is used as-is.
    """
    # Floor applies only at 0; every positive threshold is preserved exactly.
    assert hi._quiet_threshold(washer_config(stop_threshold_w=0.0)) == hi._HISTORY_IMPORT_MIN_QUIET_W
    assert hi._quiet_threshold(washer_config(stop_threshold_w=0.0)) > 0.0
    assert hi._quiet_threshold(washer_config(stop_threshold_w=3.0)) == 3.0

    config = washer_config(stop_threshold_w=0.0)
    samples = build_stream(idle_s=600.0)  # below the block cut threshold
    blocks, _ = hi.find_activity_blocks(samples, config)
    assert len(blocks) == 1  # short gap: densify must re-insert the quiet, not the cut
    dense = hi.densify_quiet_gaps(blocks[0], config)
    # Synthetic quiet samples were inserted - defeated entirely when quiet_w == 0.
    assert len(dense) > len(blocks[0].samples)


def test_a_standby_floor_above_the_stop_threshold_still_cuts():
    """The floor-independent cut rule.

    An appliance idling above ``stop_threshold_w`` never accumulates quiet, so the quiet
    rule alone would leave a ten-day stream as one block - which can only ever produce
    the detector's 8 h ``force_stopped`` blob.

    The thresholds here are the ones the manager derives for ``min_power=2`` (stop at 60%,
    start at 110%), which the bare dataclass defaults do not reproduce.
    """
    config = washer_config(stop_threshold_w=1.2, start_threshold_w=2.2)
    samples: list[tuple[datetime, float]] = []
    elapsed = 0.0

    def add(step: float, watts: float) -> None:
        nonlocal elapsed
        elapsed += step
        samples.append((T0 + timedelta(seconds=elapsed), watts))

    for index in range(2):
        for i in range(480):
            add(5.0, 1800.0 if i < 120 else 400.0)
        if index == 0:
            for _ in range(60):  # 30 minutes of 1.8 W standby: never "quiet", never active
                add(30.0, 1.8)

    assert config.stop_threshold_w < 1.8 < config.start_threshold_w
    blocks, _ = hi.find_activity_blocks(samples, config)
    assert len(blocks) == 2


def test_sensor_unavailable_breaks_the_stream():
    """A hole in the data may not carry a running load across it."""
    config = washer_config()
    samples: list[tuple[datetime, float | None]] = []
    elapsed = 0.0

    def add(step: float, watts: float | None) -> None:
        nonlocal elapsed
        elapsed += step
        samples.append((T0 + timedelta(seconds=elapsed), watts))

    for _ in range(240):
        add(5.0, 1800.0)
    add(5.0, None)                  # plug drops off the network mid-cycle
    add(6 * 3600.0, None)
    for _ in range(240):
        add(5.0, 1800.0)

    blocks, _ = hi.find_activity_blocks(samples, config)
    assert len(blocks) == 2
    assert all(block.span_s < 3600.0 for block in blocks)


# ─── Gates ────────────────────────────────────────────────────────────────────


def test_hourly_statistics_are_reported_as_skipped_spans():
    text = "\n".join(
        ["entity_id,state,last_changed"]
        + [f"{ENTITY},{300 if h % 3 else 0},2026-01-0{1 + h // 24}T{h % 24:02d}:00:00Z" for h in range(48)]
    )
    parsed = hi.parse_history_csv(text, entity_id=ENTITY)
    result = hi.build_scan(parsed.samples, washer_config(), sampling_interval_s=5.0)
    assert isinstance(result, dict)
    assert result["error"] == "no_usable_blocks"
    assert result["skipped"]


def test_cadence_gate_is_relative_to_the_devices_own_reporting_rate():
    """A plug that legitimately reports once a minute must not be rejected."""
    assert hi.max_median_interval_s(5.0) == 120.0
    assert hi.max_median_interval_s(60.0) == 240.0
    config = washer_config()
    samples = [(T0 + timedelta(seconds=60 * i), 1800.0 if i < 40 else 400.0) for i in range(80)]
    samples.append((T0 + timedelta(seconds=60 * 80), 0.0))
    blocks, _ = hi.find_activity_blocks(samples, config)
    usable, skipped = hi.classify_blocks(blocks, config, sampling_interval_s=60.0)
    assert len(usable) == 1, skipped


def test_sample_floor_follows_the_device_type():
    """A pump cycle is seconds long; a flat floor of 20 samples would discard it."""
    assert hi.min_block_samples(washer_config()) == 20
    assert hi.min_block_samples(washer_config(device_type=DEVICE_TYPE_PUMP)) == 3


def test_over_long_block_is_reported_not_replayed():
    config = washer_config()
    samples = [(T0 + timedelta(seconds=30 * i), 1800.0) for i in range(2000)]  # ~16.6 h
    blocks, _ = hi.find_activity_blocks(samples, config)
    usable, skipped = hi.classify_blocks(blocks, config, sampling_interval_s=30.0)
    assert usable == []
    assert [item["reason"] for item in skipped] == ["too_long"]


# ─── Preview rows ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("status", "accept", "reason"),
    [
        ("completed", True, None),
        ("interrupted", False, "shorter_than_minimum"),
        ("force_stopped", False, "no_clean_end"),
    ],
)
def test_accept_default_follows_the_cycle_status(status, accept, reason):
    """Status is the discriminator: on the reference export every junk detection was
    ``force_stopped`` and every real cycle ``completed``."""
    cycle = {
        "start_time": T0.isoformat(),
        "end_time": (T0 + timedelta(minutes=40)).isoformat(),
        "duration": 2400.0,
        "status": status,
        "termination_reason": "timeout",
        "power_data": [[float(i * 5), 1000.0] for i in range(480)],
    }
    row = hi.summarize_segment(cycle, index=0, completion_min_s=600.0)
    assert (row["accept"], row["reason"]) == (accept, reason)
    assert row["peak_w"] == 1000.0
    assert row["energy_wh"] > 0
    assert len(row["curve"]) <= 60


def test_preview_curve_spans_the_whole_segment():
    """With curve_points < len(powers) < 2*curve_points, floor-division stride kept
    every sample and the [:curve_points] slice showed only the head. Ceiling stride
    must span the trace so the last curve point reflects the end of the cycle."""
    # 100 ramped samples (power == index), curve_points defaults to 60.
    cycle = {
        "start_time": T0.isoformat(),
        "end_time": (T0 + timedelta(minutes=50)).isoformat(),
        "duration": 3000.0,
        "status": "completed",
        "termination_reason": "timeout",
        "power_data": [[float(i * 30), float(i)] for i in range(100)],
    }
    row = hi.summarize_segment(cycle, index=0, completion_min_s=600.0)
    # The final preview point must come from near the end (index ~98), not index 59.
    assert row["curve"][-1] >= 90.0


def test_scan_runner_is_resumable_in_small_chunks():
    """The WS task walks the scan in executor-sized bites; the result must not depend on
    the chunk size."""
    samples = build_stream()
    whole = replay_blocks(samples, washer_config())
    runner = hi.build_scan(samples, washer_config(), sampling_interval_s=5.0)
    steps = 0
    while not runner.finished:
        runner.step(37)
        steps += 1
    piecemeal = runner.finalize()
    assert steps > 10
    assert [s["duration_s"] for s in piecemeal["segments"]] == [
        s["duration_s"] for s in whole["segments"]
    ]


def test_scan_caps_the_number_of_candidates():
    samples = build_stream(cycles=4)
    runner = hi.build_scan(samples, washer_config(), sampling_interval_s=5.0)
    runner.max_segments = 2
    while not runner.finished:
        runner.step(4000)
    result = runner.finalize()
    assert result["found"] == 4
    assert len(result["segments"]) == 2
    assert result["capped"] is True


def test_build_scan_error_markers_never_raise():
    assert hi.build_scan([], washer_config())["error"] == "no_readings"
    assert hi.build_scan([(T0, None)], washer_config())["error"] == "no_readings"


def test_fixture_is_the_reporters_export():
    """Guard the fixture's shape so a well-meaning edit cannot quietly weaken the suite."""
    with FIXTURE.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2358
    assert {row["entity_id"] for row in rows} == {ENTITY}
    assert sum(1 for row in rows if row["state"] in ("unavailable", "unknown")) == 8
    hourly = [row for row in rows if row["last_changed"].endswith(":00:00.000Z")]
    assert len(hourly) > 100  # the long-term-statistics region


def test_dedup_key_accepts_a_numeric_unix_start_time():
    """Legacy cycles store start_time as a numeric unix timestamp. dedup_key must
    derive a key for both formats, else a re-import of the same history is not seen
    as a duplicate and a second copy is stored."""
    from datetime import datetime, timezone

    ts = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
    iso_key = hi.dedup_key(ts.isoformat(), 3600)
    num_key = hi.dedup_key(ts.timestamp(), 3600)  # float unix seconds
    str_num_key = hi.dedup_key(str(ts.timestamp()), 3600)  # numeric-as-string

    assert iso_key is not None
    assert num_key == iso_key
    assert str_num_key == iso_key


def test_dedup_key_rejects_unparseable_start_time():
    assert hi.dedup_key("not a timestamp", 3600) is None
    assert hi.dedup_key(None, 3600) is None
