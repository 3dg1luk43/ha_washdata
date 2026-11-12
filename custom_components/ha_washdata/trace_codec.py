# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compact on-disk form of a store's per-sample arrays (storage v19).

Home Assistant writes a ``Store`` file as JSON with a 2-space indent, so one
``[offset, watts]`` sample of ``power_data`` costs ~62 B on disk against ~14 B as
compact JSON (tron4r's Miele store: 7.8 MB of traces, 1.8 MB compact), and the
profiles' envelopes (per-point avg/min/max/std curves) pay the same indent. From
v19 the ``power_data`` and ``debug_data`` of every cycle in the three cycle lists
(and the kept tail of a cut import, ``tail_cut_trace``, which needs no version: a
build without it keeps the text as it is), and every envelope in ``envelopes``,
are written as ONE JSON string each holding
the same value as compact JSON text, and read back with one parse.

Why text and not a binary fixed-point form: a fixed-point int32 + zlib + base64
form is ~3x smaller again (~4.4 B a sample) but costs ~2x the CPU on every write,
which runs on the event loop, and needs a precision proof per array; the text
costs what the indented write of the same arrays cost and is exact by
construction.

Lossless by construction: the text is what Home Assistant's own serialiser writes
for that value (``json_bytes``: the Store writer's orjson call without the indent),
so a decode returns exactly what loading the old indented file returned. Every
float is bit-identical (orjson writes the shortest repr that round-trips, ``-0.0``
included), ints stay ints, and a value the serialiser cannot write is left as it
is for the Store writer to handle the old way.

The codec lives only at the file boundary: ``WashDataStore`` encodes a shallow
copy of the store when it writes and decodes when it loads, the pre-import
restore point does the same, and every import goes through
``unwrap_import_payload``, which decodes. Memory always holds the decoded lists
and dicts, so no reader ever sees two formats, and exports, diagnostics, the WS API and the
community-store upload keep their readable shape. Pure (no Home Assistant needed
to decode), never raises.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from .const import BACKFILL_TAIL_TRACE_KEY

_ENCODE: Callable[[Any], bytes] | None
try:  # the Store writer's own serialiser, minus the indent
    from homeassistant.helpers.json import json_bytes as _ENCODE
except ImportError:  # pragma: no cover - outside Home Assistant: decode only
    _ENCODE = None

try:
    from orjson import loads as _json_loads
except ImportError:  # pragma: no cover - stdlib parses the same text exactly
    _json_loads = json.loads

#: The three cycle lists of a store (see CLAUDE.md "Three cycle lists").
STORED_CYCLE_LISTS: tuple[str, ...] = ("past_cycles", "reference_cycles", "backfill_cycles")

#: Per-cycle fields written compact, with the type their value has in memory.
#: The third is the end wait a history-import tail cut removed (register item 618):
#: a 1 h wait densified at 30 s is ~120 rows, 1.8 KB a cut cycle on disk against
#: 8.6 KB as indented JSON (KoLSMS Beko export re-imported, 15 cuts).
COMPACT_CYCLE_FIELDS: tuple[tuple[str, type], ...] = (
    ("power_data", list),
    ("debug_data", dict),
    (BACKFILL_TAIL_TRACE_KEY, dict),
)

#: The per-profile envelope map; each envelope (a dict) is written compact as a whole.
ENVELOPES_KEY = "envelopes"


def encode_value(value: Any) -> str | None:
    """``value`` as compact JSON text, or None when it is not worth or not safe to encode.

    Only a non-empty list or dict is encoded. None as well when the serialiser
    rejects the value (the Store writer then reports the same error it always did).
    """
    if _ENCODE is None or not isinstance(value, (list, dict)) or not value:
        return None
    try:
        return _ENCODE(value).decode("utf-8")
    except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        return None  # left to the Store writer, which writes it the old way


def decode_value(value: Any, expected: type) -> Any:
    """The value a compact text stands for, or ``value`` itself when it is not one.

    A string that does not parse to an ``expected`` is returned unchanged, so a
    damaged field is kept as it was rather than dropped.
    """
    if not isinstance(value, str):
        return value
    try:
        parsed = _json_loads(value)
    except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        return value
    return parsed if isinstance(parsed, expected) else value


def encode_store_payload(data: Any) -> Any:
    """The on-disk form of a store payload. Never mutates ``data``.

    A shallow copy of ``data`` whose cycle lists and envelope map are new
    containers; a cycle with a field to encode is a shallow copy with that field
    as text, an envelope is its text, and every other cycle and key is the object
    ``data`` holds. A non-dict is returned as is.
    """
    if not isinstance(data, dict):
        return data
    out = dict(data)
    for key in STORED_CYCLE_LISTS:
        cycles = data.get(key)
        if not isinstance(cycles, list):
            continue
        new_list: list[Any] = []
        for cycle in cycles:
            if isinstance(cycle, dict):
                copied: dict[str, Any] | None = None
                for field, kind in COMPACT_CYCLE_FIELDS:
                    value = cycle.get(field)
                    # Only the declared kind: decoding accepts nothing else, so a
                    # list-valued debug_data (an import) stayed JSON text on load.
                    if not isinstance(value, kind):
                        continue
                    text = encode_value(value)
                    if text is not None:
                        if copied is None:
                            copied = dict(cycle)
                        copied[field] = text
                if copied is not None:
                    cycle = copied
            new_list.append(cycle)
        out[key] = new_list
    envelopes = data.get(ENVELOPES_KEY)
    if isinstance(envelopes, dict):
        new_map: dict[Any, Any] = {}
        for name, envelope in envelopes.items():
            text = encode_value(envelope) if isinstance(envelope, dict) else None
            new_map[name] = envelope if text is None else text
        out[ENVELOPES_KEY] = new_map
    return out


def decode_store_payload(data: Any) -> int:
    """Turn every compact field of a store payload back into its value, in place.

    ``data``'s cycle lists and envelope map are replaced by new containers, and a
    decoded cycle by a shallow copy, so containers shared with someone else (an
    import payload) are never changed. Idempotent: a decoded store has nothing
    left to decode. Returns how many values were decoded.
    """
    if not isinstance(data, dict):
        return 0
    decoded = 0
    for key in STORED_CYCLE_LISTS:
        cycles = data.get(key)
        if not isinstance(cycles, list):
            continue
        new_list: list[Any] | None = None
        for index, cycle in enumerate(cycles):
            if not isinstance(cycle, dict):
                continue
            copied: dict[str, Any] | None = None
            for field, kind in COMPACT_CYCLE_FIELDS:
                raw = cycle.get(field)
                if not isinstance(raw, str):
                    continue
                value = decode_value(raw, kind)
                if value is raw:
                    continue
                if copied is None:
                    copied = dict(cycle)
                copied[field] = value
                decoded += 1
            if copied is not None:
                if new_list is None:
                    new_list = list(cycles)
                new_list[index] = copied
        if new_list is not None:
            data[key] = new_list
    envelopes = data.get(ENVELOPES_KEY)
    if isinstance(envelopes, dict):
        new_map: dict[Any, Any] | None = None
        for name, raw in envelopes.items():
            if not isinstance(raw, str):
                continue
            value = decode_value(raw, dict)
            if value is raw:
                continue
            if new_map is None:
                new_map = dict(envelopes)
            new_map[name] = value
            decoded += 1
        if new_map is not None:
            data[ENVELOPES_KEY] = new_map
    return decoded


def count_compact_fields(data: Mapping[str, Any]) -> tuple[int, int]:
    """``(cycle fields, envelopes)`` :func:`encode_store_payload` writes as text."""
    if not isinstance(data, Mapping):
        return 0, 0
    fields = 0
    for key in STORED_CYCLE_LISTS:
        cycles = data.get(key)
        if not isinstance(cycles, list):
            continue
        for cycle in cycles:
            if isinstance(cycle, dict):
                fields += sum(
                    1 for field, kind in COMPACT_CYCLE_FIELDS
                    if isinstance(cycle.get(field), kind) and cycle.get(field)
                )
    envelopes = data.get(ENVELOPES_KEY)
    count = (
        sum(1 for e in envelopes.values() if isinstance(e, dict) and e)
        if isinstance(envelopes, dict) else 0
    )
    return fields, count
