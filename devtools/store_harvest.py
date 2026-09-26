#!/usr/bin/env python3
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
"""Harvest community-store reference cycles into ``cycle_data/`` as a test corpus.

Why this exists
---------------
Detection and matching changes are only as trustworthy as the corpus they are
validated against. Locally that corpus is whatever exports happen to have been
collected by hand; the community store already holds far more, contributed for
exactly this kind of reuse.

Read budget (read this before raising any limit)
------------------------------------------------
The store runs on Firestore's Spark tier: **50,000 document reads per day, shared
by every WashData user in the world**, and it has been close to that ceiling
before. A naive full walk is easily five figures of reads, so this tool:

* counts every document it reads and stops dead at ``--max-reads`` (default
  :data:`DEFAULT_MAX_READS`). Note the floor: **enumerating the catalogue is one
  query returning every device** (~500 documents today), so any run costs that
  before it fetches a single cycle. A full harvest is ~500 + ~350 = under 1000
  reads, about 2% of the daily budget - cheap occasionally, not something to put
  on a timer;
* is **incremental** - a manifest records what has already been harvested, and a
  device whose profile/cycle counts are unchanged is skipped without fetching its
  bundle;
* never asks for star ratings (``get_device_bundle`` already skips them: they are
  browse-only decoration and cost one aggregation *per cycle*);
* defaults to ``--dry-run``-able planning via ``--limit``;
* serialises device fetches, so it cannot burst against the store's rate limiter.

Provenance
----------
Store cycles are **reference cycles**, not observed ones. They are written to
``reference_cycles`` - never ``past_cycles`` - through the integration's own
``_add_reference_cycle_nosave``, so a harvested file has exactly the shape a real
store download produces (``ml_review.golden``, ``meta.source = "store:<id>"``,
import-time timestamps). Mixing them into ``past_cycles`` would fake provenance
and corrupt every lifetime statistic that reads it; see CLAUDE.md's three-cycle-
list table.

Usage
-----
    python3 devtools/store_harvest.py --dry-run
    python3 devtools/store_harvest.py --appliance-type dishwasher --limit 20
    python3 devtools/store_harvest.py --max-reads 5000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_OUT = REPO / "cycle_data" / "store"

#: The store calls a washing machine a "washer" (store.py:_STORE_APPLIANCE_TYPE).
#: Harvested files are consumed as if they were exports, so the device type has to
#: be written in the integration's vocabulary or every device-type-conditional
#: path (keep_tail, min_off_gap defaults, standby-band finalize) resolves wrongly.
_LOCAL_DEVICE_TYPE = {"washer": "washing_machine"}
MANIFEST_NAME = "_harvest_manifest.json"

#: Conservative slice of the store's shared 50k/day Spark budget.
DEFAULT_MAX_READS = 3000
#: Pause between device bundles, so a long run cannot look like a scraper.
DEVICE_DELAY_S = 0.25

#: Longest trace span accepted from the store. This is the detector's own
#: corrupt-data bound (``CycleDetector._SANITIZE_MAX_EXPECTED_DURATION``), reused
#: rather than invented: a trace the matcher would reject as corrupted is worse
#: than useless in a validation corpus, because it silently widens whatever it is
#: averaged into. Community uploads really do contain them - 1 of the first 73
#: harvested spanned 77 hours for a wool wash.
MAX_TRACE_SPAN_S = 6 * 3600.0


class ReadBudget(Exception):
    """Raised to unwind the walk once the read ceiling is reached."""


class Counter:
    """Every document this run has read, and the ceiling it must not cross."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.reads = 0

    def charge(self, n: int, what: str) -> None:
        """Bill ``n`` documents. Raises once the ceiling is genuinely exceeded.

        Strictly greater, not >=: enumerating the catalogue is a fixed, unavoidable
        cost (one query, ~500 documents today) and a budget set exactly to it should
        still be allowed to spend it rather than abort having read everything and
        kept nothing.
        """
        self.reads += max(0, int(n))
        if self.reads > self.limit:
            raise ReadBudget(
                f"read budget of {self.limit} exceeded while fetching {what} "
                f"({self.reads} documents). Re-run to continue: the manifest makes "
                f"this incremental."
            )


def _local_device_type(store_type: Any) -> str:
    """Store appliance type -> the integration's ``device_type``."""
    text = str(store_type or "").strip()
    return _LOCAL_DEVICE_TYPE.get(text, text)


def _slug(value: Any) -> str:
    """Filesystem-safe path segment."""
    text = "".join(
        ch if (ch.isalnum() or ch in "-_.") else "-" for ch in str(value or "").strip()
    )
    while "--" in text:
        text = text.replace("--", "-")
    return text.strip("-").lower() or "unknown"


def _make_store():
    """A ProfileStore with just enough state to build reference cycles.

    ``_add_reference_cycle_nosave`` is the integration's own builder, so reusing it
    is what guarantees a harvested file matches a real store download instead of
    drifting into a second, parallel shape.
    """
    from custom_components.ha_washdata.profile_store import ProfileStore

    store = ProfileStore.__new__(ProfileStore)
    store._data = {  # noqa: SLF001 - deliberate: this is the documented shape
        "profiles": {},
        "past_cycles": [],
        "reference_cycles": [],
        "backfill_cycles": [],
        "envelopes": {},
    }
    store._logger = SimpleNamespace(
        debug=lambda *a, **k: None,
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    return store


def _trace_span(points: Any) -> float:
    """Wall-clock span of a raw ``[[offset, watts], ...]`` trace, or 0.0."""
    try:
        offsets = [float(p[0]) for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
    except (TypeError, ValueError):
        return 0.0
    return (max(offsets) - min(offsets)) if len(offsets) >= 2 else 0.0


def _device_export(
    device: dict[str, Any], bundle: dict[str, Any]
) -> tuple[dict[str, Any] | None, int]:
    """Turn one store bundle into an export-shaped document.

    Returns ``(document_or_None, n_rejected)``.
    """
    store = _make_store()
    imported = 0
    rejected = 0
    for prof in bundle.get("profiles") or []:
        program = str(prof.get("program") or prof.get("program_lc") or "").strip()
        if not program:
            continue
        for cyc in prof.get("cycles") or []:
            points = cyc.get("importable")
            if not points:
                continue
            span = _trace_span(points)
            if span <= 0 or span > MAX_TRACE_SPAN_S:
                rejected += 1
                continue
            cid = store._add_reference_cycle_nosave(  # noqa: SLF001
                program,
                points,
                {
                    "store_cycle_id": cyc.get("id"),
                    "store_uploaded_at": cyc.get("createdAt"),
                    "sampling_interval": (cyc.get("trace") or {}).get("sampleIntervalSec"),
                },
            )
            if cid:
                imported += 1
    if not imported:
        return None, rejected

    settings = bundle.get("settings") if isinstance(bundle.get("settings"), dict) else {}
    appliance_type = _local_device_type(device.get("applianceType"))
    return {
        "version": 12,
        "source": "community_store",
        "harvested_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "store_device_id": bundle.get("device_id"),
        "device_fingerprint": {
            "device_type": appliance_type,
            "brand": device.get("brand"),
            "model": device.get("model"),
        },
        "entry_data": {"device_type": appliance_type},
        # The device's shared tunables, which is what a user adopting this package
        # would apply. Anything absent falls back to the device-type default, the
        # same as a real entry.
        "entry_options": {"device_type": appliance_type, **settings},
        "data": store._data,  # noqa: SLF001
        "_counts": {
            "profiles": len(bundle.get("profiles") or []),
            "reference_cycles": imported,
            "rejected_traces": rejected,
        },
    }, rejected


async def _harvest(args: argparse.Namespace) -> int:
    import aiohttp

    from custom_components.ha_washdata.store_client import StoreClient

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / MANIFEST_NAME
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:  # noqa: BLE001 - a corrupt manifest just means a full pass
            manifest = {}

    budget = Counter(args.max_reads)
    written = skipped = empty = rejected_total = 0
    stopped_early = False

    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # `hass` is only ever used to obtain a session, which we inject.
        client = StoreClient(SimpleNamespace(), session=session)
        try:
            devices = await client.search_devices(
                brand=args.brand,
                appliance_type=args.appliance_type,
                include_pending=args.include_pending,
            )
            budget.charge(len(devices), "device list")
            total = len(devices)

            # Skip devices that report no shared content, WITHOUT fetching them.
            # This is the difference between ~800 reads and five figures: of 500
            # catalogue entries only 86 report any cycle at all.
            #
            # These counters are contributor-maintained and known to under-report
            # (see _DEVICE_LIST_FIELDS), so a zero is "nothing claimed", not
            # "nothing there" - which is why --min-cycles 0 exists to sweep the
            # whole catalogue when the budget allows.
            if args.min_cycles > 0:
                devices = [
                    d for d in devices
                    if int(d.get("cycleCount") or 0) >= args.min_cycles
                ]
            # Richest first, so a run that hits the ceiling has still collected the
            # most useful devices.
            devices.sort(key=lambda d: -int(d.get("cycleCount") or 0))
            claimed = sum(int(d.get("cycleCount") or 0) for d in devices)
            print(
                f"catalog: {total} device(s) matched, {len(devices)} with "
                f">= {args.min_cycles} reported cycle(s), {claimed} cycles claimed"
            )

            if args.limit:
                devices = devices[: args.limit]

            for i, dev in enumerate(devices, 1):
                dev_id = dev.get("id")
                if not dev_id:
                    continue
                label = (
                    f"{dev.get('brand')} {dev.get('model')} "
                    f"[{_local_device_type(dev.get('applianceType'))}]"
                )
                # Incremental: the device doc carries the counters the store keeps
                # up to date, so an unchanged device costs nothing to skip.
                fingerprint = {
                    "profiles": dev.get("profileCount"),
                    "cycles": dev.get("cycleCount"),
                    "updated": dev.get("updatedAt"),
                }
                prev = manifest.get(dev_id)
                if prev and prev.get("fingerprint") == fingerprint and not args.force:
                    skipped += 1
                    if args.verbose:
                        print(f"  [{i}/{len(devices)}] skip (unchanged) {label}")
                    continue

                if args.dry_run:
                    print(f"  [{i}/{len(devices)}] WOULD fetch {label}")
                    written += 1
                    continue

                bundle = await client.get_device_bundle(
                    dev_id, include_pending=args.include_pending
                )
                n_cycles = sum(
                    len(p.get("cycles") or []) for p in (bundle.get("profiles") or [])
                )
                # 1 device doc + 1 profiles query + the cycle docs themselves.
                budget.charge(1 + len(bundle.get("profiles") or []) + n_cycles, label)

                doc, n_rejected = _device_export(dev, bundle)
                rejected_total += n_rejected
                if doc is None:
                    empty += 1
                    manifest[dev_id] = {"fingerprint": fingerprint, "file": None}
                    if args.verbose:
                        print(f"  [{i}/{len(devices)}] no importable traces: {label}")
                else:
                    sub = out_dir / _slug(_local_device_type(dev.get("applianceType")))
                    sub.mkdir(parents=True, exist_ok=True)
                    path = sub / f"{_slug(dev.get('brand'))}__{_slug(dev.get('model'))}.json"
                    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
                    manifest[dev_id] = {
                        "fingerprint": fingerprint,
                        "file": str(path.relative_to(out_dir)),
                        "reference_cycles": doc["_counts"]["reference_cycles"],
                    }
                    written += 1
                    print(
                        f"  [{i}/{len(devices)}] {label}: "
                        f"{doc['_counts']['reference_cycles']} cycles -> "
                        f"{path.relative_to(REPO)}"
                    )
                await asyncio.sleep(DEVICE_DELAY_S)
        except ReadBudget as stop:
            stopped_early = True
            print(f"\nSTOPPED: {stop}")
        finally:
            if not args.dry_run:
                manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))

    print(
        f"\n{'would write' if args.dry_run else 'wrote'} {written} device file(s), "
        f"skipped {skipped} unchanged, {empty} with no importable trace.\n"
        f"rejected {rejected_total} trace(s) as corrupt "
        f"(span <= 0 or > {MAX_TRACE_SPAN_S / 3600:.0f} h).\n"
        f"documents read: {budget.reads} / {budget.limit}"
    )
    if not args.dry_run:
        print(f"manifest: {manifest_path.relative_to(REPO)}")
    return 1 if stopped_early else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--brand", help="only this brand")
    ap.add_argument("--appliance-type", help="e.g. dishwasher, washing_machine")
    ap.add_argument("--limit", type=int, default=0, help="at most N devices this run")
    ap.add_argument(
        "--min-cycles",
        type=int,
        default=1,
        help=(
            "only fetch devices reporting at least N cycles (default: 1). These "
            "counters under-report, so 0 sweeps the whole catalogue - far more "
            "reads for a handful of extra devices"
        ),
    )
    ap.add_argument(
        "--max-reads",
        type=int,
        default=DEFAULT_MAX_READS,
        help=(
            "hard ceiling on Firestore document reads. The store's whole daily "
            f"budget is 50000, shared by every user (default: {DEFAULT_MAX_READS})"
        ),
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="re-fetch unchanged devices")
    ap.add_argument(
        "--include-pending",
        action="store_true",
        default=True,
        help="include recordings awaiting community approval (default: on)",
    )
    ap.add_argument("--no-include-pending", dest="include_pending", action="store_false")
    ap.add_argument("--dry-run", action="store_true", help="plan only, read nothing but the device list")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    try:
        return asyncio.run(_harvest(args))
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
