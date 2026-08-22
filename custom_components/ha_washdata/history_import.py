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
"""Historical power-data import: turn a raw power stream into candidate cycles (issue #344).

Pure, executor-safe, Home-Assistant-free (same contract as :mod:`playground`): nothing
here does I/O, and every top-level entry point returns an ``{"error": ...}`` marker
rather than raising.

**Why a pre-pass exists at all.** A Home Assistant history export - and a recorder read -
is a *change-based* stream: while an appliance sits at a steady 0 W the sensor emits no
rows whatsoever. Feeding such a stream straight into :class:`~.cycle_detector.CycleDetector`
does not work, because the detector never receives the low readings that expire a cycle and
its outage-gap logic force-stops instead. Measured on the export attached to issue #344
(2358 rows, 10 days of 5 s data behind 6 months of hourly averages): a naive replay produced
18 cycles, *every one* ``force_stopped``, one of them 61 980 minutes long, with two real
washes merged into a single blob.

So the stream is pre-segmented into **activity blocks** first, and each block is replayed
through its own fresh detector:

1. :func:`parse_history_csv` - tolerant CSV read, one entity, `unavailable` becomes an
   explicit stream break rather than a silently dropped row (dropping it makes the previous
   value carry forward across the hole, so a plug that dies mid-cycle at 2 kW would look
   like hours of running).
2. :func:`find_activity_blocks` - cut the stream wherever the appliance was demonstrably
   off. Two independent rules, unioned, because neither alone is sufficient: accumulated
   *quiet* time (carried value below the stop threshold) and time since the last
   *active* sample (immune to a standby floor sitting above the stop threshold, which
   would otherwise never accumulate quiet and leave the whole stream as one block).
3. :func:`classify_blocks` - trim leading hourly-average debris, then gate on sample count,
   cadence and span, each rejection carrying a reason the UI can show.
4. :func:`densify_quiet_gaps` - re-insert the samples a live sensor would have emitted
   inside a carried-forward quiet gap, so the detector's gap-free quiet tally accrues the
   way it does live instead of being reset by the outage ceiling.
5. :class:`ScanRunner` - resumable replay across all usable blocks, driven chunk-by-chunk
   from an executor job so the event loop keeps breathing.

The same export then yields exactly 4 ``completed`` cycles of 74.4 / 48.3 / 95.5 / 45.8
minutes. ``tests/test_history_import.py`` locks those numbers.
"""
from __future__ import annotations

import csv
import io
import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

from .const import (
    DEFAULT_SAMPLING_INTERVAL,
    DEVICE_COMPLETION_THRESHOLDS,
    HISTORY_IMPORT_DENSIFY_STEP_S,
    HISTORY_IMPORT_EDGE_GAP_S,
    HISTORY_IMPORT_MAX_BLOCK_SPAN_S,
    HISTORY_IMPORT_MAX_MEDIAN_INTERVAL_S,
    HISTORY_IMPORT_MAX_ROWS,
    HISTORY_IMPORT_MAX_SEGMENTS,
    HISTORY_IMPORT_MIN_BLOCK_SAMPLES,
    HISTORY_IMPORT_SOURCE,
    HISTORY_IMPORT_TAIL_STEP_S,
    STATE_FINISHED,
    STATE_OFF,
)
from .cycle_detector import CycleDetector, CycleDetectorConfig
from .signal_processing import energy_gap_threshold_s, integrate_wh

_LOGGER = logging.getLogger(__name__)

# A power value of None marks a *stream break*: the sensor reported `unavailable` or
# `unknown`, so nothing is known about the appliance from here until the next real
# reading. Breaks are never fed to the detector; they cut the stream and stop the
# previous value being carried across the hole.
Sample = tuple[datetime, float | None]

# Header aliases seen in the wild: HA's own history download uses
# `entity_id,state,last_changed`; hand-rolled exports and InfluxDB dumps vary.
_ENTITY_KEYS = ("entity_id", "entity", "id")
_VALUE_KEYS = ("state", "value", "power", "mean", "w")
_TIME_KEYS = ("last_changed", "last_updated", "time", "timestamp", "date")

# States that mean "nothing is known", as opposed to a number.
_UNKNOWN_STATES = frozenset({"unavailable", "unknown", "none", "null", ""})


# ─── Parsing ──────────────────────────────────────────────────────────────────


@dataclass
class ParsedHistory:
    """Result of reading a raw power history into memory."""

    samples: list[Sample] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    entity_id: str | None = None
    # Set when the file held a single entity that was not the configured sensor and it
    # was read anyway; carries the id that was asked for, so the review UI can say so.
    entity_substituted_from: str | None = None
    rows_total: int = 0
    rows_parsed: int = 0
    rows_non_numeric: int = 0
    rows_other_entity: int = 0
    rows_unordered: int = 0
    rows_duplicate: int = 0
    truncated: bool = False

    @property
    def readings(self) -> list[tuple[datetime, float]]:
        """Just the real readings, breaks removed."""
        return [(t, p) for t, p in self.samples if p is not None]

    def report(self) -> dict[str, Any]:
        """JSON-safe summary for the panel's parse step."""
        readings = self.readings
        powers = [p for _, p in readings]
        return {
            "rows_total": self.rows_total,
            "rows_parsed": self.rows_parsed,
            "rows_non_numeric": self.rows_non_numeric,
            "rows_other_entity": self.rows_other_entity,
            "rows_unordered": self.rows_unordered,
            "rows_duplicate": self.rows_duplicate,
            "truncated": self.truncated,
            "entities": list(self.entities),
            "entity_id": self.entity_id,
            "entity_substituted_from": self.entity_substituted_from,
            "first": readings[0][0].isoformat() if readings else None,
            "last": readings[-1][0].isoformat() if readings else None,
            "peak_w": round(max(powers), 1) if powers else 0.0,
            "mean_w": round(statistics.fmean(powers), 1) if powers else 0.0,
            "breaks": sum(1 for _, p in self.samples if p is None),
        }


def _parse_ts(raw: str) -> datetime | None:
    """Parse an ISO-8601 timestamp, tolerating a trailing ``Z`` and no offset.

    A naive timestamp is read as UTC: HA writes UTC in both the history download and
    the recorder, and assuming local time would move samples across a DST boundary.
    """
    text = (raw or "").strip()
    if not text:
        return None
    if text.endswith(("Z", "z")):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _pick_key(fieldnames: Iterable[str], candidates: Sequence[str]) -> str | None:
    lowered = {str(name).strip().lstrip("﻿").lower(): str(name) for name in fieldnames if name}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def parse_history_csv(
    text: str,
    *,
    entity_id: str | None = None,
    max_rows: int = HISTORY_IMPORT_MAX_ROWS,
) -> ParsedHistory | dict[str, Any]:
    """Read a power-history CSV into an ordered sample list.

    Accepts Home Assistant's history download (``entity_id,state,last_changed``) and the
    common variants of it: a ``;`` delimiter, a UTF-8 BOM, and the header aliases in
    :data:`_VALUE_KEYS` / :data:`_TIME_KEYS`.

    ``entity_id`` restricts the read to one entity - an export can hold several, and
    interleaving two appliances' readings would corrupt detection. When it is given but
    absent from the file the caller gets an error rather than a silent empty result.

    Rows whose state is ``unavailable``/``unknown`` are kept as stream breaks
    (power ``None``), not dropped. Out-of-order rows are sorted and exact-duplicate
    timestamps dropped, because :meth:`CycleDetector.process_reading` discards a reading
    whose timestamp went backwards *and still advances its clock*, which would both lose
    the sample and inflate the next gap.

    Never raises; returns ``{"error": ...}`` instead.
    """
    if not isinstance(text, str) or not text.strip():
        return {"error": "empty_file"}
    try:
        body = text.lstrip("﻿")
        sample = body[:8192]
        try:
            dialect: Any = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = "excel"
        reader = csv.DictReader(io.StringIO(body), dialect=dialect)
        if not reader.fieldnames:
            return {"error": "no_header"}
        value_key = _pick_key(reader.fieldnames, _VALUE_KEYS)
        time_key = _pick_key(reader.fieldnames, _TIME_KEYS)
        entity_key = _pick_key(reader.fieldnames, _ENTITY_KEYS)
        if value_key is None or time_key is None:
            return {"error": "missing_columns"}

        out = ParsedHistory(entity_id=entity_id)
        _wanted_entity = (entity_id or "").strip().casefold()
        entities: dict[str, int] = {}
        rows: list[Sample] = []
        for row in reader:
            out.rows_total += 1
            if out.rows_total > max_rows:
                out.truncated = True
                break
            row_entity = str(row.get(entity_key) or "").strip() if entity_key else ""
            if row_entity:
                entities[row_entity] = entities.get(row_entity, 0) + 1
            # Case/whitespace-insensitive: entity ids are lowercase by convention, but an
            # export that round-tripped through a spreadsheet can differ in case alone,
            # which must not read as "a different appliance".
            if entity_id and row_entity and row_entity.casefold() != _wanted_entity:
                out.rows_other_entity += 1
                continue
            timestamp = _parse_ts(str(row.get(time_key) or ""))
            if timestamp is None:
                out.rows_non_numeric += 1
                continue
            raw_value = str(row.get(value_key) or "").strip()
            if raw_value.lower() in _UNKNOWN_STATES:
                rows.append((timestamp, None))
                continue
            try:
                # A locale-formatted export writes 1234,5 (comma = decimal point). But a
                # single comma followed by exactly three digits is more likely a thousands
                # group ("1,234" is 1234, not 1.234), which is too ambiguous to rewrite: a
                # wrong guess stores a value 1000x off, so leave it and let float() drop it.
                _frac = raw_value.split(",", 1)[1] if raw_value.count(",") == 1 else ""
                _decimal_comma = raw_value.count(",") == 1 and not (
                    len(_frac) == 3 and _frac.isdigit()
                )
                power = float(raw_value.replace(",", ".") if _decimal_comma else raw_value)
            except ValueError:
                out.rows_non_numeric += 1
                continue
            if not math.isfinite(power):
                out.rows_non_numeric += 1
                continue
            rows.append((timestamp, max(0.0, power)))

        out.entities = sorted(entities)
        if (
            entity_id
            and entities
            and _wanted_entity not in {e.casefold() for e in entities}
        ):
            # The configured sensor is not in the file. With exactly ONE entity in it the
            # upload is still unambiguous - the user picked this file for this device, and
            # a renamed entity, a template/helper sensor in front of the plug, or an export
            # taken under the old id all land here - so honour it and record the
            # substitution instead of dead-ending. The error is kept for a MULTI-entity
            # file, where guessing which appliance to read would corrupt detection.
            if len(entities) == 1:
                only = next(iter(entities))
                retry = parse_history_csv(text, entity_id=only, max_rows=max_rows)
                if isinstance(retry, ParsedHistory):
                    retry.entity_substituted_from = entity_id
                return retry
            return {"error": "entity_not_in_file", "entities": out.entities}
        if not rows:
            return {"error": "no_readings"}

        ordered = sorted(range(len(rows)), key=lambda i: rows[i][0])
        out.rows_unordered = sum(1 for pos, i in enumerate(ordered) if pos != i)
        last_ts: datetime | None = None
        for i in ordered:
            timestamp, power = rows[i]
            if last_ts is not None and timestamp == last_ts:
                out.rows_duplicate += 1
                continue
            out.samples.append((timestamp, power))
            last_ts = timestamp
        out.rows_parsed = len(out.samples)
        return out
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("History CSV parse failed: %s", exc)
        return {"error": "parse_failed"}


def samples_from_readings(readings: Iterable[tuple[float, float]]) -> list[Sample]:
    """Convert ``(unix_ts, watts)`` pairs - the recorder read shape - into samples."""
    out: list[Sample] = []
    for raw_ts, raw_power in readings or []:
        try:
            timestamp = datetime.fromtimestamp(float(raw_ts), tz=timezone.utc)
            power = float(raw_power)
        except (TypeError, ValueError, OSError, OverflowError):
            continue
        if math.isfinite(power):
            out.append((timestamp, max(0.0, power)))
    out.sort(key=lambda item: item[0])
    return out


# ─── Block segmentation ───────────────────────────────────────────────────────


@dataclass
class Block:
    """A contiguous stretch of stream in which the appliance was plausibly doing something."""

    samples: list[tuple[datetime, float]]

    @property
    def start(self) -> datetime:
        return self.samples[0][0]

    @property
    def end(self) -> datetime:
        return self.samples[-1][0]

    @property
    def span_s(self) -> float:
        return (self.end - self.start).total_seconds()

    @property
    def median_dt_s(self) -> float:
        gaps = [
            (b[0] - a[0]).total_seconds()
            for a, b in zip(self.samples, self.samples[1:])
            if (b[0] - a[0]).total_seconds() > 0
        ]
        return statistics.median(gaps) if gaps else 0.0

    @property
    def peak_w(self) -> float:
        return max((p for _, p in self.samples), default=0.0)

    def summary(self, *, reason: str | None = None) -> dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "span_s": round(self.span_s, 1),
            "samples": len(self.samples),
            "median_interval_s": round(self.median_dt_s, 1),
            "peak_w": round(self.peak_w, 1),
            **({"reason": reason} if reason else {}),
        }


def cut_threshold_s(config: CycleDetectorConfig) -> float:
    """How long the appliance must look off before the stream is cut in two.

    ``min_off_gap`` is what the live detector itself requires before it will separate two
    cycles (up to an hour on a dishwasher, to bridge drying pauses), plus ``off_delay``.
    Cutting any sooner would hand the detector two halves of one cycle; cutting later is
    harmless because the detector applies the same rule inside the block.
    """
    return max(60.0, float(config.min_off_gap or 0.0) + float(config.off_delay or 0.0))


# Fallback "off" level when stop_threshold_w is 0. The panel allows min=0 for that
# field, but a 0 threshold makes nothing read as quiet (power is clamped >= 0), so block
# quiet-accrual and gap densification would both silently no-op and two cycles a few
# minutes apart inside one block could merge. Any positive configured value is used as-is.
_HISTORY_IMPORT_MIN_QUIET_W = 1.0


def _quiet_threshold(config: CycleDetectorConfig) -> float:
    stop = float(config.stop_threshold_w or 0.0)
    return stop if stop > 0.0 else _HISTORY_IMPORT_MIN_QUIET_W


def _active_threshold(config: CycleDetectorConfig) -> float:
    floor = _quiet_threshold(config)
    return max(floor, float(config.start_threshold_w or 0.0), float(config.min_power or 0.0))


def find_activity_blocks(
    samples: Sequence[Sample],
    config: CycleDetectorConfig,
    *,
    cut_after_s: float | None = None,
) -> tuple[list[Block], list[dict[str, Any]]]:
    """Split a raw stream into activity blocks, dropping the dead air between them.

    Three independent cut rules, unioned:

    * a stream break (``power is None``) - the sensor went away, so nothing may be
      carried across the hole;
    * **quiet accumulation** - the carried-forward value has been below the stop
      threshold for longer than ``cut_after_s``;
    * **no activity** - no sample has reached the start threshold for longer than
      ``cut_after_s``. This rule is what makes the pre-pass work on an appliance whose
      standby draw sits *above* the stop threshold: quiet would never accumulate there,
      and without it the whole stream stays one block and can only ever produce the
      detector's 8 h ``force_stopped`` blob.

    Blocks that never reach the start threshold are dead air and are returned as skipped
    spans instead, so the UI can account for every row.
    """
    quiet_w = _quiet_threshold(config)
    active_w = _active_threshold(config)
    limit = float(cut_after_s if cut_after_s is not None else cut_threshold_s(config))

    blocks: list[Block] = []
    skipped: list[dict[str, Any]] = []
    current: list[tuple[datetime, float]] = []
    quiet_s = 0.0
    idle_s = 0.0

    def _close() -> None:
        nonlocal current
        if not current:
            return
        block = Block(current)
        if block.peak_w >= active_w and len(current) >= 2:
            blocks.append(block)
        else:
            skipped.append(block.summary(reason="idle"))
        current = []

    for timestamp, power in samples:
        if power is None:
            _close()
            quiet_s = idle_s = 0.0
            continue
        if current:
            gap = (timestamp - current[-1][0]).total_seconds()
            carried = current[-1][1]
            quiet_s = quiet_s + gap if carried < quiet_w else 0.0
            idle_s += gap
            if quiet_s > limit or idle_s > limit:
                _close()
                quiet_s = idle_s = 0.0
        current.append((timestamp, power))
        if power >= active_w:
            idle_s = 0.0
        if power >= quiet_w:
            quiet_s = 0.0
    _close()
    return blocks, skipped


def trim_leading_debris(
    block: Block, *, edge_gap_s: float = HISTORY_IMPORT_EDGE_GAP_S
) -> Block:
    """Drop leading samples that stand more than ``edge_gap_s`` from the block body.

    An hourly-average row glued to the head of an otherwise dense block would otherwise
    open the replay with a one-hour gap at running power, which the detector reads as an
    outage. Only the *leading* edge is trimmed: doing the same at the trailing edge eats a
    real cycle's low-power tail (measured: a 95.5-minute wash became 86.4).
    """
    samples = list(block.samples)
    while len(samples) > 2 and (samples[1][0] - samples[0][0]).total_seconds() > edge_gap_s:
        samples.pop(0)
    return Block(samples)


def min_block_samples(config: CycleDetectorConfig) -> int:
    """Sample-count floor for a usable block, derived from the device type.

    A flat 20 would discard whole device classes - a pump cycle can be under 30 seconds
    (``DEVICE_COMPLETION_THRESHOLDS[pump] = 5``) - so the floor is how many samples the
    shortest cycle this device can have would produce at the slowest cadence the
    integration assumes, capped at the generic default.
    """
    completion_s = float(DEVICE_COMPLETION_THRESHOLDS.get(config.device_type, 600))
    expected = int(completion_s / max(1.0, DEFAULT_SAMPLING_INTERVAL))
    return max(3, min(HISTORY_IMPORT_MIN_BLOCK_SAMPLES, expected or 3))


def max_median_interval_s(sampling_interval_s: float | None) -> float:
    """Cadence gate, relative to what this device actually reports.

    Zigbee plugs and Tasmota's ``TelePeriod`` commonly report once a minute, so a fixed
    60 s ceiling would reject perfectly good hardware. The absolute floor still rejects
    HA's hourly long-term statistics.
    """
    observed = float(sampling_interval_s or 0.0)
    return max(HISTORY_IMPORT_MAX_MEDIAN_INTERVAL_S, 4.0 * observed)


def classify_blocks(
    blocks: Sequence[Block],
    config: CycleDetectorConfig,
    *,
    sampling_interval_s: float | None = None,
    max_span_s: float = HISTORY_IMPORT_MAX_BLOCK_SPAN_S,
) -> tuple[list[Block], list[dict[str, Any]]]:
    """Trim and gate blocks, returning ``(usable, skipped_with_reason)``.

    Rejection reasons are part of the contract - the wizard reports them, so a user whose
    export is six months of hourly averages is told that rather than shown zero results.
    Internal gaps are deliberately left alone: the detector's own outage handling is tuned
    for them, and re-splitting on them would cut a real cycle at its quiet mid-phases.
    """
    min_samples = min_block_samples(config)
    max_dt = max_median_interval_s(sampling_interval_s)
    usable: list[Block] = []
    skipped: list[dict[str, Any]] = []
    for raw in blocks:
        block = trim_leading_debris(raw)
        if len(block.samples) < min_samples:
            skipped.append(block.summary(reason="too_few_samples"))
            continue
        if block.median_dt_s > max_dt:
            skipped.append(block.summary(reason="sparse"))
            continue
        if block.span_s > max_span_s:
            skipped.append(block.summary(reason="too_long"))
            continue
        usable.append(block)
    return usable, skipped


def densify_quiet_gaps(
    block: Block,
    config: CycleDetectorConfig,
    *,
    step_s: float = HISTORY_IMPORT_DENSIFY_STEP_S,
) -> list[tuple[datetime, float]]:
    """Re-insert the quiet samples a live sensor would have emitted inside a gap.

    The detector only credits quiet time it actually observed: a step larger than its
    outage ceiling (``max(60, 10 x p95_dt)``) *resets* the gap-free quiet tally, on the
    principle that unobserved time must not be treated as quiet. That is right for a live
    sensor and wrong for a change-based history, where a 500-second gap after a 0 W row
    means five hundred seconds of 0 W. Without this the detector cannot separate two
    cycles whose gap is shorter than the block cut threshold, and they merge.

    Only *quiet* carried values are densified. A gap carrying a running load is left as a
    gap, because a change-based sensor that stops reporting mid-cycle is genuinely
    ambiguous and the detector's outage handling is the right judge of it.
    """
    quiet_w = _quiet_threshold(config)
    step = max(1.0, float(step_s))
    out: list[tuple[datetime, float]] = []
    for current, following in zip(block.samples, block.samples[1:]):
        out.append(current)
        timestamp, power = current
        gap = (following[0] - timestamp).total_seconds()
        if power >= quiet_w or gap <= step * 1.5:
            continue
        for n in range(1, int(gap // step) + 1):
            filled = timestamp + timedelta(seconds=step * n)
            if filled >= following[0]:
                # A synthetic sample landing on (or past) the next real one would only
                # feed the detector a zero-length step.
                break
            out.append((filled, power))
    if block.samples:
        out.append(block.samples[-1])
    return out


# ─── Replay ───────────────────────────────────────────────────────────────────


class StreamSegmenter:
    """Replays one block through a fresh :class:`CycleDetector`, collecting whole cycles.

    Resumable so a long block can be walked from many small executor jobs: ``step`` takes
    a half-open reading range, exactly like the Playground's single-cycle simulator.

    The detector runs with no profile matcher. Import boundaries are therefore purely
    power-based: Smart Termination, the dishwasher end-spike arm and the dryer anti-crease
    gate all need a matched profile with an expected duration, and a fresh install has no
    profiles at all. The cost is that an anti-crease dryer tail can run to the detector's
    8 h cap and be stamped ``force_stopped`` - which is why the accept gate below is
    ``status == "completed"``.
    """

    def __init__(
        self,
        readings: Sequence[tuple[datetime, float]],
        config: CycleDetectorConfig,
    ) -> None:
        self.config = config
        self.readings = list(readings)
        self.captured: list[dict[str, Any]] = []
        self.aborted = False
        self.ready = len(self.readings) >= 2
        self.detector: CycleDetector | None = None
        if self.ready:
            self.detector = CycleDetector(
                config,
                self._on_state_change,
                self._on_cycle_end,
                profile_matcher=None,
                device_name="history-import",
            )

    @property
    def n_readings(self) -> int:
        return len(self.readings)

    def _on_state_change(self, old_state: str, new_state: str) -> None:
        """State transitions are not surfaced; the cycle payload carries what matters."""

    def _on_cycle_end(self, cycle_data: dict[str, Any]) -> None:
        self.captured.append(cycle_data)

    def step(self, i0: int, i1: int) -> None:
        """Replay ``readings[i0:i1]``. Never raises; a failure aborts this block only."""
        if self.aborted or not self.ready or self.detector is None:
            return
        try:
            for timestamp, power in self.readings[i0:i1]:
                self.detector.process_reading(power, timestamp)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.aborted = True
            _LOGGER.debug("History-import replay failed at %s-%s: %s", i0, i1, exc)

    def flush_tail(self) -> bool:
        """Close an open cycle with a synthetic quiet tail.

        Returns ``True`` when the block ended cleanly. A block still running afterwards is
        reported as truncated rather than force-ended: ``force_end`` would stamp
        ``force_stopped`` on what is probably a real cycle the export simply cut short.
        """
        if self.aborted or not self.ready or self.detector is None:
            return False
        try:
            last_ts = self.readings[-1][0]
            span = max(
                float(self.config.off_delay or 0.0), float(self.config.min_off_gap or 0.0)
            ) * 1.5 + 300.0
            step = max(1.0, float(HISTORY_IMPORT_TAIL_STEP_S))
            for n in range(1, int(span / step) + 2):
                self.detector.process_reading(0.0, last_ts + timedelta(seconds=step * n))
                if self.detector.state in (STATE_OFF, STATE_FINISHED):
                    break
            return self.detector.state in (STATE_OFF, STATE_FINISHED)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.aborted = True
            _LOGGER.debug("History-import tail flush failed: %s", exc)
            return False


def summarize_segment(
    cycle_data: dict[str, Any],
    *,
    index: int,
    completion_min_s: float,
    curve_points: int = 60,
) -> dict[str, Any]:
    """Preview row for one detected candidate: what it is, and whether to accept it.

    ``accept`` is the checkbox default, not a filter. Only ``completed`` cycles default to
    accepted: on the issue's export every one of the 18 junk detections was
    ``force_stopped`` and all 4 real cycles were ``completed``, which makes status a clean
    discriminator. A cycle shorter than the device's completion threshold is stamped
    ``interrupted`` by the detector and contributes nothing to an envelope, so it is
    surfaced unchecked with a reason rather than hidden.
    """
    power_data = cycle_data.get("power_data") or []
    offsets = [float(point[0]) for point in power_data]
    powers = [float(point[1]) for point in power_data]
    duration = float(cycle_data.get("duration") or 0.0)
    status = str(cycle_data.get("status") or "")
    energy_wh = 0.0
    if len(offsets) >= 2:
        energy_wh = integrate_wh(offsets, powers, max_gap_s=energy_gap_threshold_s(offsets))

    if status == "completed":
        accept, reason = True, None
    elif status == "interrupted":
        accept, reason = False, "shorter_than_minimum"
    else:
        accept, reason = False, "no_clean_end"

    # Ceiling division so the stride always spans the whole trace. Floor division
    # gives step=1 when curve_points < len(powers) < 2*curve_points, and the [:curve_points]
    # slice below would then show only the first curve_points samples (the head of the
    # cycle) instead of the full shape.
    step = max(1, -(-len(powers) // max(1, curve_points)))
    return {
        "index": index,
        "start_time": cycle_data.get("start_time"),
        "end_time": cycle_data.get("end_time"),
        "duration_s": round(duration, 1),
        "status": status,
        "termination_reason": cycle_data.get("termination_reason"),
        "samples": len(power_data),
        "peak_w": round(max(powers), 1) if powers else 0.0,
        "energy_wh": round(energy_wh, 1),
        "accept": accept,
        "reason": reason,
        "below_minimum": duration < float(completion_min_s or 0.0),
        "curve": [round(p, 1) for p in powers[::step]][:curve_points],
    }


class ScanRunner:
    """Resumable replay of every usable block, driven chunk-by-chunk from an executor.

    ``step`` advances at most ``n`` readings and returns the number consumed, so the WS
    task can report progress and honour a cancel between chunks without ever holding the
    GIL long enough to freeze the panel.
    """

    def __init__(
        self,
        blocks: Sequence[Block],
        config: CycleDetectorConfig,
        *,
        skipped: Sequence[dict[str, Any]] = (),
        parse_report: dict[str, Any] | None = None,
        max_segments: int = HISTORY_IMPORT_MAX_SEGMENTS,
    ) -> None:
        self.config = config
        self.skipped = [dict(item) for item in skipped]
        self.parse_report = dict(parse_report or {})
        self.max_segments = max(1, int(max_segments))
        self._streams = [
            densify_quiet_gaps(block, config) for block in blocks
        ]
        self.total = sum(len(stream) for stream in self._streams)
        self.done = 0
        self._block = 0
        self._cursor = 0
        self._segmenter: StreamSegmenter | None = None
        self._captured: list[dict[str, Any]] = []
        self.truncated_blocks = 0

    @property
    def finished(self) -> bool:
        return self._block >= len(self._streams)

    def step(self, n: int = 1) -> int:
        """Advance up to ``n`` readings across block boundaries. Never raises."""
        budget = max(1, int(n))
        consumed = 0
        while budget > 0 and not self.finished:
            stream = self._streams[self._block]
            if self._segmenter is None:
                self._segmenter = StreamSegmenter(stream, self.config)
            take = min(budget, len(stream) - self._cursor)
            if take > 0:
                self._segmenter.step(self._cursor, self._cursor + take)
                self._cursor += take
                budget -= take
                consumed += take
                self.done += take
            if self._cursor >= len(stream):
                if not self._segmenter.flush_tail():
                    self.truncated_blocks += 1
                self._captured.extend(self._segmenter.captured)
                self._segmenter = None
                self._block += 1
                self._cursor = 0
        return consumed

    def finalize(self, *, partial: bool = False) -> dict[str, Any]:
        """Preview payload plus the full cycle payloads, for the caller to split apart.

        ``cycles`` carries whole traces and must stay server-side: shipping them over the
        WebSocket would blow the 4 MiB frame cap and take the connection down with it.
        """
        completion_min_s = float(self.config.completion_min_seconds or 0.0)
        cycles = self._captured[: self.max_segments]
        segments = [
            summarize_segment(cycle, index=i, completion_min_s=completion_min_s)
            for i, cycle in enumerate(cycles)
        ]
        return {
            "segments": segments,
            "cycles": cycles,
            "skipped": self.skipped,
            "parse": self.parse_report,
            "found": len(self._captured),
            "capped": len(self._captured) > len(cycles),
            "truncated_blocks": self.truncated_blocks,
            "partial": bool(partial),
        }


def build_scan(
    samples: Sequence[Sample],
    config: CycleDetectorConfig,
    *,
    sampling_interval_s: float | None = None,
    parse_report: dict[str, Any] | None = None,
) -> ScanRunner | dict[str, Any]:
    """Blocks + gates + a ready-to-drive :class:`ScanRunner`. Never raises."""
    try:
        readings = [(t, p) for t, p in samples if p is not None]
        if len(readings) < 2:
            return {"error": "no_readings"}
        blocks, idle = find_activity_blocks(samples, config)
        usable, gated = classify_blocks(
            blocks, config, sampling_interval_s=sampling_interval_s
        )
        if not usable:
            return {
                "error": "no_usable_blocks",
                "skipped": idle + gated,
                "parse": dict(parse_report or {}),
            }
        return ScanRunner(
            usable,
            config,
            skipped=idle + gated,
            parse_report=parse_report,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("History-import scan build failed: %s", exc)
        return {"error": "scan_failed"}


# ─── Persistence ──────────────────────────────────────────────────────────────


def _dedup_start_dt(start_time: Any) -> datetime | None:
    """Coerce a stored ``start_time`` into a datetime for the dedup key.

    Mirrors ``profile_store._parse_start_dt`` (kept local so this module stays
    hass-free and import-light): datetime as-is, a numeric unix timestamp (int/float
    or numeric string, the legacy storage format) via ``fromtimestamp``, otherwise an
    ISO-8601 string via ``_parse_ts``.
    """
    if isinstance(start_time, datetime):
        return start_time
    if isinstance(start_time, (int, float)) and not isinstance(start_time, bool):
        try:
            return datetime.fromtimestamp(float(start_time), tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    text = str(start_time or "").strip()
    if not text:
        return None
    parsed = _parse_ts(text)
    if parsed is not None:
        return parsed
    try:
        return datetime.fromtimestamp(float(text), tz=timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def dedup_key(start_time: Any, duration: Any) -> tuple[int, int] | None:
    """Identity of a cycle for re-import detection: its start second and whole seconds.

    ``_add_cycle_data`` derives a cycle id from ``sha256(start_time + duration)`` and, on
    collision, appends a suffix until the id is unique - so re-importing the same export
    would silently store a second copy of every cycle rather than being rejected. Callers
    compare this key against the cycles already stored and skip the matches.

    Rounded to whole seconds so a re-export whose timestamps differ in fractions, or
    whose duration was recomputed, still matches.

    Accepts both storage formats for ``start_time``: an ISO-8601 string (current) and
    a numeric unix timestamp (legacy cycles, per ``profile_store._parse_start_dt``).
    Without the numeric fallback a legacy cycle produced no key, so a re-import of the
    same history was not seen as a duplicate and a second copy was stored.
    """
    parsed = _dedup_start_dt(start_time)
    if parsed is None:
        return None
    try:
        return (int(parsed.timestamp()), int(round(float(duration or 0.0))))
    except (TypeError, ValueError):
        return None


def existing_dedup_keys(cycles: Iterable[dict[str, Any]]) -> set[tuple[int, int]]:
    """Dedup keys for everything already stored, from every cycle list."""
    out: set[tuple[int, int]] = set()
    for cycle in cycles or []:
        key = dedup_key(cycle.get("start_time"), cycle.get("duration"))
        if key is not None:
            out.add(key)
    return out


def build_backfill_cycle(cycle_data: dict[str, Any]) -> dict[str, Any]:
    """Turn a detected segment into a storable ``backfill_cycles`` entry.

    Deliberately minimal. In particular it does **not** set ``ml_review.golden``: that
    flag marks a curated reference recording, unlocks sharing to the community store and
    stamps a star in the UI, and an auto-detected segment nobody has verified has earned
    none of that. The real timestamps are kept (unlike a store import, which stamps
    import time) because when the cycle ran is the whole point of importing it.
    """
    out = {
        key: cycle_data[key]
        for key in (
            "start_time",
            "end_time",
            "duration",
            "status",
            "termination_reason",
            "max_power",
            "power_data",
        )
        if key in cycle_data
    }
    out["profile_name"] = None
    out["meta"] = {"source": HISTORY_IMPORT_SOURCE}
    return out
