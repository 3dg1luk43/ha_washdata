# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Storage v19 compact traces and envelopes, proven lossless on the whole local corpus.

Every store payload in ``cycle_data/`` (exports, diagnostics downloads, store
files, gzipped or not) goes through the codec and the real Store serialiser and
back; each cycle field and envelope must come out with the same type and every
float with the same 64-bit pattern as the parse of today's indented file.
Local-only data.
"""
from __future__ import annotations

import gzip
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
from homeassistant.helpers.json import _orjson_default_encoder
from homeassistant.util.json import json_loads

from custom_components.ha_washdata import trace_codec

from .test_storage_v19_compact_traces import _bits

pytestmark = pytest.mark.slow

CORPUS = Path(__file__).resolve().parent.parent / "cycle_data"


def _docs(path: Path) -> list[Any]:
    try:
        raw = path.read_bytes()
        if path.suffix == ".gz":
            raw = gzip.decompress(raw)
            if path.name.endswith(".zip.gz"):
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    return [json.loads(zf.read(n)) for n in zf.namelist() if n.endswith(".json")]
        return [json.loads(raw)]
    except (OSError, ValueError, zipfile.BadZipFile):
        return []


def _stores(obj: Any) -> list[dict[str, Any]]:
    """Every dict holding one of the three cycle lists, however deep."""
    out: list[dict[str, Any]] = []
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if any(isinstance(cur.get(k), list) for k in trace_codec.STORED_CYCLE_LISTS):
                out.append(cur)
            stack.extend(v for v in cur.values() if isinstance(v, (dict, list)))
        elif isinstance(cur, list):
            stack.extend(v for v in cur if isinstance(v, dict))
    return out


def test_codec_is_bit_exact_on_every_stored_cycle_and_envelope_of_the_corpus() -> None:
    if not CORPUS.is_dir():
        pytest.skip("cycle_data corpus not present (local-only)")
    files = sorted(
        p for p in CORPUS.rglob("*")
        if p.is_file() and (p.name.endswith((".json", ".json.gz", ".zip.gz")))
    )
    fields = samples = envelopes = 0
    for path in files:
        for doc in _docs(path):
            for store in _stores(doc):
                payload = {k: store.get(k) for k in trace_codec.STORED_CYCLE_LISTS if isinstance(store.get(k), list)}
                if isinstance(store.get(trace_codec.ENVELOPES_KEY), dict):
                    payload[trace_codec.ENVELOPES_KEY] = store[trace_codec.ENVELOPES_KEY]
                    envelopes += sum(isinstance(e, dict) and bool(e) for e in store[trace_codec.ENVELOPES_KEY].values())
                disk = json_loads(_orjson_default_encoder(trace_codec.encode_store_payload(payload)))
                trace_codec.decode_store_payload(disk)
                assert _bits(disk) == _bits(payload), path
                for key in trace_codec.STORED_CYCLE_LISTS:
                    for cycle in payload.get(key) or []:
                        if isinstance(cycle, dict) and isinstance(cycle.get("power_data"), list):
                            fields += 1
                            samples += len(cycle["power_data"])
    if not fields:
        pytest.skip("no stored cycle traces under cycle_data/")
    assert samples > 1_000_000  # the corpus at the time of writing: 1.57 M samples
    assert envelopes > 100
