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
import operator
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

from .const import (
    BANKED_TAIL_REPAIR_MIN_S,
    CONF_LEARNING_CONFIDENCE,
    DEFAULT_LEARNING_CONFIDENCE,
    DEFAULT_SAMPLING_INTERVAL,
    DEVICE_COMPLETION_THRESHOLDS,
    DEVICE_TYPE_DISHWASHER,
    HISTORY_IMPORT_DENSIFY_STEP_S,
    HISTORY_IMPORT_EDGE_GAP_S,
    HISTORY_IMPORT_MAX_BRIDGE_S,
    HISTORY_IMPORT_MAX_BLOCK_SPAN_S,
    HISTORY_IMPORT_MAX_MEDIAN_INTERVAL_S,
    HISTORY_IMPORT_MAX_ROWS,
    HISTORY_IMPORT_MAX_SEGMENTS,
    HISTORY_IMPORT_MIN_BLOCK_SAMPLES,
    HISTORY_IMPORT_SOURCE,
    HISTORY_IMPORT_TAIL_STEP_S,
    STATE_FINISHED,
    STATE_OFF,
    TERMINAL_EVENT_PEAK_FRAC,
    TERMINAL_QUIET_CAP_S,
    TRUSTED_LENGTH_FLOOR_FRAC,
    TerminationReason,
)
from .cycle_detector import CycleDetector, CycleDetectorConfig
from .options_utils import option_float
from .signal_processing import (
    energy_gap_threshold_s,
    integrate_wh,
    terminal_event_end,
    terminal_quiet_seen,
)

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
    # Rows whose timestamp carried no UTC offset (read as UTC, PLAYGROUND-24).
    rows_naive_time: int = 0
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
            "rows_naive_time": self.rows_naive_time,
            "truncated": self.truncated,
            "entities": list(self.entities),
            "entity_id": self.entity_id,
            "entity_substituted_from": self.entity_substituted_from,
            "first": readings[0][0].isoformat() if readings else None,
            "last": readings[-1][0].isoformat() if readings else None,
            "peak_w": round(max(powers), 1) if powers else 0.0,
            "mean_w": round(statistics.fmean(powers), 1) if powers else 0.0,
            "breaks": sum(1 for _, p in self.samples if p is None),
            "warnings": parse_warnings(powers, self.rows_naive_time),
        }


# Below this peak a stream that still has a shape is far more likely kilowatts than
# an appliance: every supported type draws hundreds of watts or more at its peak.
KW_SUSPECT_PEAK_W = 20.0


def parse_warnings(powers: Sequence[float], naive_rows: int = 0) -> list[str]:
    """What the review step should flag about an otherwise readable stream.

    ``looks_like_kw``: the peak is under :data:`KW_SUSPECT_PEAK_W` yet the readings
    have structure (at least three distinct non-zero values) - a kW sensor, which
    the import (and the live integration) reads as watts, so nothing is ever
    detected. ``naive_timestamps``: some timestamps carried no UTC offset and were
    read as UTC (audit PLAYGROUND-24).
    """
    warnings: list[str] = []
    # The distinct-value set only when the peak qualifies: building it rounded every
    # reading of a watt-scale stream (~0.15 s of the 0.34 s report at the row cap).
    if powers and max(powers) < KW_SUSPECT_PEAK_W:
        positive = {round(p, 6) for p in powers if p > 0}
        if positive and max(positive) < KW_SUSPECT_PEAK_W and len(positive) >= 3:
            warnings.append("looks_like_kw")
    if naive_rows > 0:
        warnings.append("naive_timestamps")
    return warnings


def _parse_ts_naive(raw: str) -> tuple[datetime | None, bool]:
    """:func:`_parse_ts`, plus whether the text carried no UTC offset."""
    text = (raw or "").strip()
    if not text:
        return None, False
    if text.endswith(("Z", "z")):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None, False
    if parsed.tzinfo:
        return parsed, False
    return parsed.replace(tzinfo=timezone.utc), True


def _parse_ts(raw: str) -> datetime | None:
    """Parse an ISO-8601 timestamp, tolerating a trailing ``Z`` and no offset.

    A naive timestamp is read as UTC: HA writes UTC in both the history download and
    the recorder, and assuming local time would move samples across a DST boundary.
    The parse report counts them (``rows_naive_time``) so the panel can say so: a
    file written in local time is shifted by the UTC offset, and across a DST change
    its repeated hour reorders and drops rows (audit PLAYGROUND-24).
    """
    return _parse_ts_naive(raw)[0]


def _pick_key(fieldnames: Iterable[str], candidates: Sequence[str]) -> str | None:
    lowered = {str(name).strip().lstrip("﻿").lower(): str(name) for name in fieldnames if name}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


# Rows read per executor job by :class:`HistoryCsvParser` (audit PLAYGROUND-11).
# ~0.1 s on a desktop, so even a Pi stays well under the multi-second GIL holds
# the one-job parse of a 500k-row file was (4.4 s CPU measured on a desktop).
PARSE_STEP_ROWS = 10_000


class HistoryCsvParser:
    """:func:`parse_history_csv`, resumable: ``step`` reads a slice of rows.

    The WS scan task drives it across many executor jobs (audit PLAYGROUND-11), so a
    file at the 32 MiB / 500k-row caps no longer holds the GIL for seconds in one
    job. :func:`parse_history_csv` drives the same object to completion, so both
    produce the same result. Never raises; :meth:`result` returns the parsed
    history or an ``{"error": ...}`` marker.
    """

    def __init__(
        self,
        text: str,
        *,
        entity_id: str | None = None,
        max_rows: int = HISTORY_IMPORT_MAX_ROWS,
    ) -> None:
        self.entity_id = entity_id
        self.max_rows = max_rows
        self.error: dict[str, Any] | None = None
        self.finished = False
        self.out = ParsedHistory(entity_id=entity_id)
        self._wanted = (entity_id or "").strip().casefold()
        self._entities: dict[str, int] = {}
        # Rows in file order, tagged True for a row this read keeps. Rows of another
        # entity are held (tagged False) only while the file could still turn out to
        # hold that ONE entity alone, which is read in its place (see `result`); the
        # second distinct entity, or the configured one, drops them.
        self._rows: list[tuple[bool, Sample]] = []
        self._may_substitute = bool(entity_id)
        self._other_non_numeric = 0
        self._other_naive = 0
        self._reader: Any = None
        self._keys: tuple[str, str, str | None] = ("", "", None)
        # Line count, for a progress bar (a quoted field can hold a newline, so it
        # is an estimate).
        self.rows_estimate = max(1, text.count("\n")) if isinstance(text, str) else 1
        if not isinstance(text, str) or not text.strip():
            self._fail({"error": "empty_file"})
            return
        try:
            body = text.lstrip("\ufeff")
            sample = body[:8192]
            try:
                dialect: Any = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except csv.Error:
                dialect = "excel"
            reader = csv.DictReader(io.StringIO(body), dialect=dialect)
            if not reader.fieldnames:
                self._fail({"error": "no_header"})
                return
            value_key = _pick_key(reader.fieldnames, _VALUE_KEYS)
            time_key = _pick_key(reader.fieldnames, _TIME_KEYS)
            entity_key = _pick_key(reader.fieldnames, _ENTITY_KEYS)
            if value_key is None or time_key is None:
                self._fail({"error": "missing_columns"})
                return
            self._reader = reader
            self._keys = (value_key, time_key, entity_key)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("History CSV parse failed: %s", exc)
            self._fail({"error": "parse_failed"})

    def _fail(self, error: dict[str, Any]) -> None:
        self.error = error
        self.finished = True
        self._reader = None
        self._rows = []

    def _parse_row(self, row: dict[str, Any]) -> tuple[Sample | None, bool]:
        """One row's sample (None when it is not one; counted by the caller) and
        whether its timestamp carried no offset."""
        value_key, time_key, _entity_key = self._keys
        timestamp, naive = _parse_ts_naive(str(row.get(time_key) or ""))
        if timestamp is None:
            return None, False
        return self._parse_value(row, value_key, timestamp), naive

    @staticmethod
    def _parse_value(row: dict[str, Any], value_key: str, timestamp: datetime) -> Sample | None:
        raw_value = str(row.get(value_key) or "").strip()
        if raw_value.lower() in _UNKNOWN_STATES:
            return (timestamp, None)
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
            return None
        if not math.isfinite(power):
            return None
        return (timestamp, max(0.0, power))

    def step(self, n_rows: int = PARSE_STEP_ROWS) -> int:
        """Read up to ``n_rows`` more rows; returns how many were read."""
        if self.finished or self._reader is None:
            return 0
        out = self.out
        _value_key, _time_key, entity_key = self._keys
        read = 0
        try:
            for row in self._reader:
                read += 1
                out.rows_total += 1
                if out.rows_total > self.max_rows:
                    out.truncated = True
                    self.finished = True
                    break
                row_entity = str(row.get(entity_key) or "").strip() if entity_key else ""
                if row_entity:
                    if row_entity not in self._entities and self._may_substitute and (
                        self._entities or row_entity.casefold() == self._wanted
                    ):
                        # A second entity, or the configured one: no substitution.
                        self._may_substitute = False
                        self._rows = [item for item in self._rows if item[0]]
                    self._entities[row_entity] = self._entities.get(row_entity, 0) + 1
                # Case/whitespace-insensitive: entity ids are lowercase by convention,
                # but an export that round-tripped through a spreadsheet can differ in
                # case alone, which must not read as "a different appliance".
                other = bool(
                    self.entity_id and row_entity and row_entity.casefold() != self._wanted
                )
                if other:
                    out.rows_other_entity += 1
                    if not self._may_substitute:
                        if read >= n_rows:
                            break
                        continue
                parsed, naive = self._parse_row(row)
                if parsed is None:
                    if other:
                        self._other_non_numeric += 1
                    else:
                        out.rows_non_numeric += 1
                else:
                    self._rows.append((not other, parsed))
                    if naive:
                        if other:
                            self._other_naive += 1
                        else:
                            out.rows_naive_time += 1
                if read >= n_rows:
                    break
            else:
                self.finished = True
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("History CSV parse failed: %s", exc)
            self._fail({"error": "parse_failed"})
        return read

    def result(self) -> ParsedHistory | dict[str, Any]:
        """The parsed history once :attr:`finished` (sorted, de-duplicated)."""
        if self.error is not None:
            return self.error
        try:
            while not self.finished:
                self.step(PARSE_STEP_ROWS)
            if self.error is not None:
                return self.error
            out = self.out
            entities = self._entities
            out.entities = sorted(entities)
            entity_id = self.entity_id
            if (
                entity_id
                and entities
                and self._wanted not in {e.casefold() for e in entities}
            ):
                # The configured sensor is not in the file. With exactly ONE entity in
                # it the upload is still unambiguous - the user picked this file for
                # this device, and a renamed entity, a template/helper sensor in front
                # of the plug, or an export taken under the old id all land here - so
                # honour it and record the substitution instead of dead-ending. The
                # error is kept for a MULTI-entity file, where guessing which
                # appliance to read would corrupt detection.
                if len(entities) == 1:
                    out.entity_id = next(iter(entities))
                    out.entity_substituted_from = entity_id
                    out.rows_other_entity = 0
                    out.rows_non_numeric += self._other_non_numeric
                    out.rows_naive_time += self._other_naive
                    rows = [sample for _keep, sample in self._rows]
                else:
                    return {"error": "entity_not_in_file", "entities": out.entities}
            else:
                rows = [sample for keep, sample in self._rows if keep]
            self._rows = []
            if not rows:
                return {"error": "no_readings"}

            stamps = [row[0] for row in rows]
            if all(map(operator.le, stamps, stamps[1:])):
                # Already in order (the usual export): the stable sort is the identity.
                ordered: Sequence[int] = range(len(rows))
            else:
                ordered = sorted(range(len(rows)), key=stamps.__getitem__)
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
    absent from the file the caller gets an error rather than a silent empty result
    (unless the file holds exactly one other entity, which is read in its place).

    Rows whose state is ``unavailable``/``unknown`` are kept as stream breaks
    (power ``None``), not dropped. Out-of-order rows are sorted and exact-duplicate
    timestamps dropped, because :meth:`CycleDetector.process_reading` discards a reading
    whose timestamp went backwards *and still advances its clock*, which would both lose
    the sample and inflate the next gap.

    One call; the WS scan steps a :class:`HistoryCsvParser` across executor jobs
    instead. Never raises; returns ``{"error": ...}`` instead.
    """
    return HistoryCsvParser(text, entity_id=entity_id, max_rows=max_rows).result()


class RecorderReadings:
    """:func:`samples_from_readings`, resumable: ``step`` converts a slice of rows.

    The WS scan task drives it across executor jobs like :class:`HistoryCsvParser`
    (audit PLAYGROUND-11): 500k recorder rows were one ~0.7 s job.
    :func:`samples_from_readings` drives the same object to completion, so both give
    the same samples.
    """

    def __init__(self, readings: Iterable[tuple[float, float]] | None) -> None:
        self._rows: Sequence[tuple[float, float]] = (
            readings if isinstance(readings, (list, tuple)) else list(readings or [])
        )
        self.rows_estimate = max(1, len(self._rows))
        self.done = 0
        self._out: list[Sample] = []

    @property
    def finished(self) -> bool:
        return self.done >= len(self._rows)

    def step(self, n_rows: int = PARSE_STEP_ROWS) -> int:
        """Convert up to ``n_rows`` more rows; returns how many were read."""
        start = self.done
        end = min(len(self._rows), start + max(1, int(n_rows)))
        out = self._out
        for raw_ts, raw_power in self._rows[start:end]:
            try:
                timestamp = datetime.fromtimestamp(float(raw_ts), tz=timezone.utc)
                power = float(raw_power)
            except (TypeError, ValueError, OSError, OverflowError):
                continue
            if math.isfinite(power):
                out.append((timestamp, max(0.0, power)))
        self.done = end
        return end - start

    def result(self) -> list[Sample]:
        """The samples, sorted by time (stable, as the one-shot sort was)."""
        while not self.finished:
            self.step(PARSE_STEP_ROWS)
        self._out.sort(key=lambda item: item[0])
        return self._out


def samples_from_readings(readings: Iterable[tuple[float, float]]) -> list[Sample]:
    """Convert ``(unix_ts, watts)`` pairs - the recorder read shape - into samples."""
    return RecorderReadings(readings).result()


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

    * a stream break (``power is None``) longer than ``HISTORY_IMPORT_MAX_BRIDGE_S``
      (or the cut threshold, if shorter) - the sensor went away, so nothing may be
      carried across the hole. A shorter break is bridged: it is a blip or a restart,
      not the end of anything;
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
    finder = _BlockFinder(samples, config, cut_after_s=cut_after_s)
    finder.step(len(samples))
    return finder.finish()


class _BlockFinder:
    """:func:`find_activity_blocks`, resumable: ``step`` walks a slice of samples.

    :class:`ScanBuilder` drives it across executor jobs (audit PLAYGROUND-11); the
    function drives it in one call, so both cut the stream identically.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        config: CycleDetectorConfig,
        *,
        cut_after_s: float | None = None,
    ) -> None:
        self.samples = samples
        self.quiet_w = _quiet_threshold(config)
        self.active_w = _active_threshold(config)
        self.limit = float(
            cut_after_s if cut_after_s is not None else cut_threshold_s(config)
        )
        self.blocks: list[Block] = []
        self.skipped: list[dict[str, Any]] = []
        self.readings = 0  # samples with a power value (build_scan's first gate)
        self.pos = 0
        self._current: list[tuple[datetime, float]] = []
        self._quiet_s = 0.0
        self._idle_s = 0.0
        self._in_break = False

    @property
    def finished(self) -> bool:
        return self.pos >= len(self.samples)

    def _close(self) -> None:
        current = self._current
        if not current:
            return
        block = Block(current)
        if block.peak_w >= self.active_w and len(current) >= 2:
            self.blocks.append(block)
        else:
            self.skipped.append(block.summary(reason="idle"))
        self._current = []

    def step(self, n: int) -> int:
        """Walk up to ``n`` more samples; returns how many."""
        start = self.pos
        end = min(len(self.samples), start + max(1, int(n)))
        quiet_w, active_w, limit = self.quiet_w, self.active_w, self.limit
        quiet_s, idle_s, in_break = self._quiet_s, self._idle_s, self._in_break
        readings = self.readings
        for timestamp, power in self.samples[start:end]:
            if power is None:
                # Do not cut yet: a 2 s Wi-Fi blip or an HA restart writes the same
                # `unavailable` row, and cutting there split one wash into two
                # "completed" candidates, both pre-ticked (audit PLAYGROUND-06). The
                # next real sample decides: a short hole is bridged as a plain gap
                # (the detector's own outage logic judges it), a long one cuts.
                in_break = True
                continue
            readings += 1
            current = self._current
            if in_break:
                in_break = False
                hole = (
                    (timestamp - current[-1][0]).total_seconds() if current else float("inf")
                )
                if hole > min(limit, HISTORY_IMPORT_MAX_BRIDGE_S):
                    self._close()
                    current = self._current
                    quiet_s = idle_s = 0.0
            if current:
                gap = (timestamp - current[-1][0]).total_seconds()
                carried = current[-1][1]
                quiet_s = quiet_s + gap if carried < quiet_w else 0.0
                idle_s += gap
                if quiet_s > limit or idle_s > limit:
                    self._close()
                    current = self._current
                    quiet_s = idle_s = 0.0
            current.append((timestamp, power))
            if power >= active_w:
                idle_s = 0.0
            if power >= quiet_w:
                quiet_s = 0.0
        self._quiet_s, self._idle_s, self._in_break = quiet_s, idle_s, in_break
        self.readings = readings
        self.pos = end
        return end - start

    def finish(self) -> tuple[list[Block], list[dict[str, Any]]]:
        """Close the last block; ``(blocks, skipped)`` as the function returns them."""
        self._close()
        return self.blocks, self.skipped


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
        _classify_block(raw, min_samples, max_dt, max_span_s, usable, skipped)
    return usable, skipped


def _classify_block(
    raw: Block,
    min_samples: int,
    max_dt: float,
    max_span_s: float,
    usable: list[Block],
    skipped: list[dict[str, Any]],
) -> None:
    """One block of :func:`classify_blocks` (shared with :class:`ScanBuilder`)."""
    block = trim_leading_debris(raw)
    if len(block.samples) < min_samples:
        skipped.append(block.summary(reason="too_few_samples"))
        return
    if block.median_dt_s > max_dt:
        skipped.append(block.summary(reason="sparse"))
        return
    if block.span_s > max_span_s:
        skipped.append(block.summary(reason="too_long"))
        return
    usable.append(block)


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
        streams: Sequence[list[tuple[datetime, float]]] | None = None,
    ) -> None:
        self.config = config
        self.skipped = [dict(item) for item in skipped]
        self.parse_report = dict(parse_report or {})
        self.max_segments = max(1, int(max_segments))
        # ``streams``: the blocks already densified (``ScanBuilder`` does it a slice
        # per executor job); otherwise densified here.
        self._streams = (
            list(streams)
            if streams is not None
            else [densify_quiet_gaps(block, config) for block in blocks]
        )
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


# Samples of block work per executor job when :class:`ScanBuilder` is stepped
# (audit PLAYGROUND-11): ~50 ms on a desktop, against ~0.7-1.7 s for the one job
# ``build_scan`` was at the 500k-row cap.
SCAN_BUILD_STEP_SAMPLES = 50_000


class ScanBuilder:
    """:func:`build_scan`, resumable: ``step`` does a slice of its work.

    Three passes, each cut by a sample budget: the block finder over the stream, the
    gates per block, the quiet-gap densification per usable block. The WS scan task
    drives it across executor jobs; :func:`build_scan` drives it to completion, so both
    return the same runner or the same error. Never raises.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        config: CycleDetectorConfig,
        *,
        sampling_interval_s: float | None = None,
        parse_report: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self._parse_report = parse_report
        self._result: ScanRunner | dict[str, Any] | None = None
        self._finder: _BlockFinder | None = None
        self._phase = "find"
        self._blocks: list[Block] = []
        self._idle: list[dict[str, Any]] = []
        self._usable: list[Block] = []
        self._gated: list[dict[str, Any]] = []
        self._streams: list[list[tuple[datetime, float]]] = []
        self._index = 0
        try:
            self._finder = _BlockFinder(samples, config)
            self._min_samples = min_block_samples(config)
            self._max_dt = max_median_interval_s(sampling_interval_s)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._fail(exc)

    @property
    def finished(self) -> bool:
        return self._result is not None

    def _fail(self, exc: Exception) -> None:
        _LOGGER.debug("History-import scan build failed: %s", exc)
        self._result = {"error": "scan_failed"}

    def step(self, n: int = SCAN_BUILD_STEP_SAMPLES) -> int:
        """Do up to ``n`` samples of work; returns how many. Never raises."""
        budget = max(1, int(n))
        spent = 0
        try:
            while spent < budget and self._result is None:
                spent += self._advance(budget - spent)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._fail(exc)
        return spent

    def _advance(self, budget: int) -> int:
        """One bounded unit of the current pass; returns the samples it cost."""
        if self._phase == "find":
            finder = self._finder
            if finder is None:
                raise RuntimeError("scan builder lost its block finder")
            spent = finder.step(budget) if not finder.finished else 0
            if finder.finished:
                if finder.readings < 2:
                    self._result = {"error": "no_readings"}
                    return max(1, spent)
                self._blocks, self._idle = finder.finish()
                self._finder = None
                self._phase = "classify"
                self._index = 0
            return max(1, spent)
        if self._phase == "classify":
            spent = 0
            while self._index < len(self._blocks) and spent < budget:
                block = self._blocks[self._index]
                _classify_block(
                    block, self._min_samples, self._max_dt,
                    HISTORY_IMPORT_MAX_BLOCK_SPAN_S, self._usable, self._gated,
                )
                spent += max(1, len(block.samples))
                self._index += 1
            if self._index >= len(self._blocks):
                self._blocks = []
                if not self._usable:
                    self._result = {
                        "error": "no_usable_blocks",
                        "skipped": self._idle + self._gated,
                        "parse": dict(self._parse_report or {}),
                    }
                self._phase = "densify"
                self._index = 0
            return max(1, spent)
        # densify
        spent = 0
        while self._index < len(self._usable) and spent < budget:
            block = self._usable[self._index]
            self._streams.append(densify_quiet_gaps(block, self.config))
            spent += max(1, len(block.samples))
            self._index += 1
        if self._index >= len(self._usable):
            self._result = ScanRunner(
                self._usable,
                self.config,
                skipped=self._idle + self._gated,
                parse_report=self._parse_report,
                streams=self._streams,
            )
        return max(1, spent)

    def result(self) -> ScanRunner | dict[str, Any]:
        """The ready runner, or the error marker :func:`build_scan` returns."""
        while self._result is None:
            self.step(SCAN_BUILD_STEP_SAMPLES)
        return self._result


def build_scan(
    samples: Sequence[Sample],
    config: CycleDetectorConfig,
    *,
    sampling_interval_s: float | None = None,
    parse_report: dict[str, Any] | None = None,
) -> ScanRunner | dict[str, Any]:
    """Blocks + gates + a ready-to-drive :class:`ScanRunner`. Never raises.

    One call; the WS scan steps a :class:`ScanBuilder` across executor jobs instead.
    """
    try:
        builder = ScanBuilder(
            samples, config,
            sampling_interval_s=sampling_interval_s,
            parse_report=parse_report,
        )
        builder.step(max(1, len(samples)) * 3)
        return builder.result()
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
    except (TypeError, ValueError, OverflowError):
        return None


def existing_dedup_keys(cycles: Iterable[dict[str, Any]]) -> set[tuple[int, int]]:
    """Dedup keys for everything already stored, from every cycle list."""
    out: set[tuple[int, int]] = set()
    for cycle in cycles or []:
        key = dedup_key(cycle.get("start_time"), cycle.get("duration"))
        if key is not None:
            out.add(key)
    return out


def stored_intervals(cycles: Iterable[dict[str, Any]]) -> list[tuple[float, float]]:
    """``(start, end)`` unix-second spans of every stored cycle, sorted by start."""
    out: list[tuple[float, float]] = []
    for cycle in cycles or []:
        start = _dedup_start_dt(cycle.get("start_time"))
        if start is None:
            continue
        try:
            duration = max(0.0, float(cycle.get("duration") or 0.0))
        except (TypeError, ValueError, OverflowError):
            continue
        t0 = start.timestamp()
        out.append((t0, t0 + duration))
    out.sort()
    return out


def overlaps_stored(
    start_time: Any, duration: Any, intervals: Sequence[tuple[float, float]]
) -> bool:
    """True when a candidate's span overlaps any stored cycle's span.

    The exact ``dedup_key`` alone missed 81% of re-detected cycles: a cycle recorded
    live ends via Smart Termination and a tail trim, the same run replayed from raw
    history ends unmatched on the timeout, so start and duration rarely agree to the
    second (audit PLAYGROUND-05; Beko 14205 s stored vs 17805 s imported). Any real
    overlap means the run is already on record. Touching endpoints do not count.
    """
    start = _dedup_start_dt(start_time)
    if start is None:
        return False
    try:
        t0 = start.timestamp()
        t1 = t0 + max(0.0, float(duration or 0.0))
    except (TypeError, ValueError, OverflowError):
        return False
    for s0, s1 in intervals:
        if s0 >= t1:
            break
        if s1 > t0 and s0 < t1:
            return True
    return False


def mark_already_recorded(
    segments: Iterable[dict[str, Any]], intervals: Sequence[tuple[float, float]]
) -> int:
    """Untick preview rows that overlap a stored cycle; returns how many were marked.

    Shown, never pre-ticked: a labelled copy double-weights the real cycle in its
    profile's envelope (audit PLAYGROUND-05).
    """
    marked = 0
    for seg in segments or []:
        if overlaps_stored(seg.get("start_time"), seg.get("duration_s"), intervals):
            seg["accept"] = False
            seg["reason"] = "already_recorded"
            marked += 1
    return marked


def _last_active_offset(points: Sequence[tuple[float, float]], stop: float) -> float | None:
    for offset, power in reversed(points):
        if power > stop:
            return offset
    return None


def _offset_points(cycle_data: dict[str, Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for point in cycle_data.get("power_data") or []:
        try:
            points.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError, IndexError, OverflowError):
            continue
    return points


def import_tail_probe(
    cycle_data: dict[str, Any], config: CycleDetectorConfig
) -> tuple[list[list[float]], float] | None:
    """The trace to match an imported candidate on, when its tail may be cut.

    Only a dishwasher's timeout finish keeps a tail (``keep_tail``); every other
    finish of an unmatched replay already snaps back to the last activity. The probe
    stops at the last activity: matched with the banked wait still on it, two thirds
    of the corpus candidates found no confident programme (the wait wrecks the
    duration terms), cut at the activity nine in ten do. Never raises.
    """
    try:
        if getattr(config, "device_type", None) != DEVICE_TYPE_DISHWASHER:
            return None
        if cycle_data.get("termination_reason") != TerminationReason.TIMEOUT:
            return None
        points = _offset_points(cycle_data)
        stop = float(getattr(config, "stop_threshold_w", 0.0) or 0.0)
        last_active = _last_active_offset(points, stop)
        if last_active is None or last_active <= 0:
            return None
        probe = [[o, p] for o, p in points if o <= last_active]
        return (probe, last_active) if len(probe) >= 2 else None
    except Exception:  # pylint: disable=broad-exception-caught
        return None


async def async_import_tail_cuts(
    store: Any,
    config: CycleDetectorConfig,
    cycles: Sequence[dict[str, Any]],
    options: dict[str, Any] | None = None,
    segments: Sequence[dict[str, Any]] = (),
) -> int:
    """Stamp ``tail_cut_s`` on every candidate whose banked end wait can be cut.

    Each one is matched (``ProfileStore.async_match_profile`` on its
    :func:`import_tail_probe`) and must pass ``label_verdict`` at the device's
    learning floor - the same bar an auto-label clears - before its programme's
    measured drying span, trusted length and expected duration are applied by
    :func:`import_tail_cut_s`. The match only names the programme whose
    statistics bound the cut: the candidate itself stays unlabelled. Returns how
    many were stamped; never raises (a failed match leaves that candidate as is).
    """
    from .profile_store import label_verdict  # pylint: disable=import-outside-toplevel

    learning_floor = float(
        option_float(
            (options or {}).get(CONF_LEARNING_CONFIDENCE, DEFAULT_LEARNING_CONFIDENCE),
            DEFAULT_LEARNING_CONFIDENCE,
        )
        or 0.0
    )
    rows = {seg.get("index"): seg for seg in segments or () if isinstance(seg, dict)}
    stamped = 0
    stop = float(getattr(config, "stop_threshold_w", 0.0) or 0.0)
    for position, cycle in enumerate(cycles or ()):
        try:
            probe = import_tail_probe(cycle, config)
            if probe is None:
                continue
            result = await store.async_match_profile(
                probe[0], probe[1], stop_threshold_w=stop
            )
            name, _reason = label_verdict(result, learning_floor)
            if not name:
                continue
            cut = import_tail_cut_s(
                cycle,
                config,
                store.profile_terminal_quiet_seconds(name),
                store.profile_trusted_min_duration(name),
                getattr(result, "expected_duration", None),
            )
            if cut is not None:
                cycle["tail_cut_s"] = round(cut, 1)
                stamped += 1
                # The preview row shows what will be stored, and the overlap check
                # (`mark_already_recorded`) reads the same span.
                row = rows.get(position)
                if row is not None:
                    row["banked_tail_s"] = round(float(cycle.get("duration") or 0.0) - cut, 1)
                    row["duration_s"] = round(cut, 1)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("History-import tail match failed: %s", exc)
    return stamped


def effective_duration(cycle_data: dict[str, Any]) -> Any:
    """The duration a candidate will be stored with: its tail cut, else as detected."""
    cut = cycle_data.get("tail_cut_s")
    return cut if cut else cycle_data.get("duration")


def import_tail_cut_s(
    cycle_data: dict[str, Any],
    config: CycleDetectorConfig,
    quiet_s: float | None,
    trusted_min_s: float | None = None,
    expected_s: float | None = None,
) -> float | None:
    """Duration to store for an imported candidate, or None to keep it as detected.

    Audit PLAYGROUND-07: the import replays unmatched, so
    ``CycleDetector._keep_tail_cap`` has no expected duration and returns None, and a
    dishwasher's timeout finish (``keep_tail``) banks the whole quiet wait as cycle
    time - ~20% on the corpus, exactly ``min_off_gap`` on one Beko. The banked-tail
    repair would undo it but skips ``backfill_cycles``. This is that rule for an
    import: the dishwasher half of ``_keep_tail_cap`` / ``async_repair_banked_tails``
    (the same shared ``terminal_quiet_seen`` / ``terminal_event_end`` judgement, the
    same ``TERMINAL_QUIET_CAP_S``, ``TRUSTED_LENGTH_FLOOR_FRAC`` and
    ``BANKED_TAIL_REPAIR_MIN_S``), fed the statistics of the programme
    :func:`async_import_tail_cuts` matched: its measured drying span ``quiet_s``,
    its trusted length, and - where no span was measured - its expected duration,
    the live cap's fallback. (The live cap's end-spike shortcut needs detector state
    the replay does not keep; the trace test covers the same pump-out.) Every other
    device type's timeout already snaps back to its last activity, so there is
    nothing to cut. Shorten-only; never raises.
    """
    try:
        if getattr(config, "device_type", None) != DEVICE_TYPE_DISHWASHER:
            return None
        if cycle_data.get("termination_reason") != TerminationReason.TIMEOUT:
            return None
        quiet = float(quiet_s) if quiet_s is not None else 0.0
        expected = float(expected_s) if expected_s is not None else 0.0
        if not (math.isfinite(quiet) and quiet > 0) and not (
            math.isfinite(expected) and expected > 0
        ):
            return None
        stop = float(getattr(config, "stop_threshold_w", 0.0) or 0.0)
        points = _offset_points(cycle_data)
        if len(points) < 2:
            return None
        last_active = _last_active_offset(points, stop)
        if last_active is None:
            return None
        if not (math.isfinite(quiet) and quiet > 0):
            # No measured drying span: the live cap falls back to the programme's
            # expected end (`_dishwasher_tail_cap`), never before the last activity.
            new_duration = max(expected, last_active)
        elif terminal_quiet_seen(points, last_active, stop, quiet, TERMINAL_EVENT_PEAK_FRAC):
            # The drying already happened before the pump-out: no allowance on top.
            new_duration = terminal_event_end(points, last_active, TERMINAL_EVENT_PEAK_FRAC)
        else:
            new_duration = last_active + min(quiet, TERMINAL_QUIET_CAP_S)
        if trusted_min_s:
            # Never below the length the user has vouched for (register item 384).
            new_duration = max(new_duration, TRUSTED_LENGTH_FLOOR_FRAC * float(trusted_min_s))
        old_duration = float(cycle_data.get("duration") or 0.0)
        if old_duration - new_duration < BANKED_TAIL_REPAIR_MIN_S:
            return None
        return new_duration
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _LOGGER.debug("History-import tail cut failed: %s", exc)
        return None


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
