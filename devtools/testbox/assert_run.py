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
"""Acceptance checks for one replayed cycle on the test box.

Each check states what it proves and why a unit test cannot. Run by smoke.sh
after the replay; exits non-zero on the first failing check's category so it can
gate a release the way run_tests.sh does.

    ./assert_run.py <entry_id> [--slug test_dishwasher] [--baseline baseline.json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import aiohttp

from hactl import CAPTURE_FILE, LOG_FILE, WS, _read_jsonl, rest

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
INFO = "\033[36m ..  \033[0m"


class Checks:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.count = 0

    def check(self, ok: bool, title: str, detail: str = "") -> bool:
        self.count += 1
        print(f"  [{PASS if ok else FAIL}] {title}")
        if detail and not ok:
            for line in detail.splitlines():
                print(f"         {line}")
        if not ok:
            self.failures.append(title)
        return ok

    def note(self, text: str) -> None:
        print(f"  [{INFO}] {text}")


def _live(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r for r in records
        if isinstance(r.get("data"), dict) and r["data"].get("live_update") is True
        and r["message"] != "clear_notification"
    ]


def _clears(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r.get("message") == "clear_notification"]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("entry_id")
    parser.add_argument("--slug", default="test_dishwasher")
    parser.add_argument("--baseline", default="")
    args = parser.parse_args()

    baseline: dict[str, Any] = {}
    if args.baseline and Path(args.baseline).exists():
        baseline = json.loads(Path(args.baseline).read_text())
    since = baseline.get("since")

    records = _read_jsonl(CAPTURE_FILE, since)
    checks = Checks()

    async with aiohttp.ClientSession() as session:
        # ── the cycle itself ────────────────────────────────────────────────
        print("\ncycle")
        states = {
            s["entity_id"]: s
            for s in await rest(session, "GET", "/api/states")
            if s["entity_id"].startswith(("sensor.", "binary_sensor."))
        }
        state_entity = f"sensor.{args.slug}_state"
        checks.check(
            state_entity in states,
            f"{state_entity} exists",
            f"present: {[e for e in states if args.slug in e][:8]}",
        )
        async with WS(session) as ws:
            cycles = await ws.cmd(
                "ha_washdata/get_device_cycles", entry_id=args.entry_id
            )
        stored = cycles.get("cycles", cycles) if isinstance(cycles, dict) else cycles
        # `total` is the device's whole cycle count; the `cycles` list is one page
        # (limit defaults to 50), so counting the page compares two capped lists
        # and never grows.
        total = cycles.get("total", len(stored)) if isinstance(cycles, dict) else len(stored)
        before = baseline.get("cycle_count", 0)
        checks.check(
            total > before,
            f"a cycle was recorded ({before} -> {total})",
            "The replay ran but nothing was stored, so detection never completed.",
        )
        if stored:
            newest = stored[0] if stored[0].get("start_time", "") >= stored[-1].get("start_time", "") else stored[-1]
            checks.note(
                f"newest: {newest.get('profile_name')} | "
                f"{float(newest.get('duration') or 0) / 60:.1f} min | "
                f"status={newest.get('status')} | "
                f"reason={newest.get('termination_reason')}"
            )
            checks.check(
                float(newest.get("duration") or 0) > 0,
                "the recorded cycle has a duration",
            )

        # ── notifications actually delivered ───────────────────────────────
        # This whole section is the part no mocked hass can do: these records
        # exist only because Home Assistant accepted the payload, validated it
        # against the real notify schema and handed it to a platform.
        print("\nnotifications delivered")
        checks.note(f"{len(records)} captured since the baseline")
        for record in records[:40]:
            data = record.get("data") or {}
            tag = data.get("tag", "-")
            marks = ",".join(
                k for k in ("live_update", "chronometer", "sticky", "silent")
                if data.get(k)
            )
            print(
                f"         {record['service']:<20} {record['message'][:52]:<52} "
                f"tag={str(tag).rsplit('_', 1)[-1]:<10} {marks}"
            )

        checks.check(
            any(r["message"] != "clear_notification" for r in records),
            "something was delivered at all",
            "Nothing reached a notify platform. Either no notify target is "
            "configured or every call was rejected on the bus.",
        )
        # NB: a dismiss marker legitimately carries no title (it is addressed by
        # its tag), and the capture cannot tell "no title key" from "title None"
        # - the platform receives kwargs.get("title") either way. The real
        # null-title failure is a REJECTED call, which shows up as a
        # MultipleInvalid in the log check below and as a missing record here.
        titled = [r for r in records if r["message"] != "clear_notification"]
        checks.check(
            all(r.get("title") for r in titled),
            "every content notification carries a title",
            f"untitled: {[r['message'][:40] for r in titled if not r.get('title')]}",
        )

        live = _live(records)
        checks.check(bool(live), f"live updates were delivered ({len(live)})")
        checks.check(
            all(r["service"].startswith("mobile_app_") for r in live),
            "live updates went to mobile targets only",
            "Non-mobile targets received live payloads they cannot render: "
            f"{sorted({r['service'] for r in live if not r['service'].startswith('mobile_app_')})}",
        )
        if live:
            tags = {(r.get("data") or {}).get("tag") for r in live}
            checks.check(
                len(tags) == 1 and all(t and t.endswith("_live") for t in tags),
                "live updates share one dedicated tag",
                f"tags seen: {tags}",
            )

        # #454: one configured colour, three companion-app keys, mobile only.
        # A MagicMock accepts any payload, so the unit suite can prove the keys
        # are BUILT but never that they survive a real notify service call; and
        # `color` in particular is a documented Android key, so a schema that
        # rejected it would only show up here.
        colour_keys = ("color", "notification_icon_color", "progress_bar_color")
        mobile = [r for r in titled if r["service"].startswith("mobile_app_")]
        plain = [r for r in titled if not r["service"].startswith("mobile_app_")]
        checks.check(
            bool(mobile)
            and all(
                all((r.get("data") or {}).get(k) == "#4CAF50" for k in colour_keys)
                for r in mobile
            ),
            f"the notification colour reached every mobile payload ({len(mobile)})",
            "missing/wrong: "
            + str([
                {k: (r.get("data") or {}).get(k) for k in colour_keys}
                for r in mobile
                if any((r.get("data") or {}).get(k) != "#4CAF50" for k in colour_keys)
            ][:3]),
        )
        checks.check(
            all(
                not any(k in (r.get("data") or {}) for k in colour_keys)
                for r in plain
            ),
            "the colour never reached a non-mobile target",
            "Strict-schema platforms received mobile-only colour keys: "
            + str(sorted({r["service"] for r in plain
                          if any(k in (r.get("data") or {}) for k in colour_keys)})),
        )

        # The MATCHED live payload, which is what a user sees for most of a
        # cycle: the Android progress bar and countdown, and the iOS Live
        # Activity's content_state. Until the matcher fires, live updates carry
        # only the "no profile matched yet" text, so a run that never matched
        # proves nothing about these keys.
        matched_live = [
            r for r in live
            if "progress" in (r.get("data") or {})
        ]
        checks.check(
            bool(matched_live),
            f"a matched live update was delivered ({len(matched_live)})",
            "Every live update was the pre-match placeholder, so the progress / "
            "chronometer / content_state payload was never exercised. Check that "
            "profile_match_interval is compressed for this run.",
        )
        for record in matched_live[:1]:
            data = record["data"]
            checks.check(
                data.get("progress_max", 0) > 0
                and 0 <= data.get("progress", -1) <= data["progress_max"],
                "the progress bar is within its own maximum",
                f"progress={data.get('progress')} max={data.get('progress_max')}",
            )
            checks.check(
                data.get("chronometer") is True and int(data.get("when", 0)) > 0,
                "the countdown carries a chronometer and an absolute end time",
                f"chronometer={data.get('chronometer')} when={data.get('when')}",
            )
            checks.check(
                int(data.get("time_remaining_seconds", -1)) >= 0
                and int(data.get("cycle_seconds", 0)) > 0,
                "remaining and total are both present and sane",
                f"remaining={data.get('time_remaining_seconds')} "
                f"total={data.get('cycle_seconds')}",
            )

        clears = _clears(records)
        cleared_tags = {(r.get("data") or {}).get("tag") for r in clears}
        checks.check(
            bool(clears),
            f"dismissals were delivered ({len(clears)})",
            "Not one clear_notification arrived. This is the #446 failure mode: "
            "the live card stays on the phone forever. It is invisible to the "
            "unit suite, which only checks the payload it built itself.",
        )
        checks.check(
            any(t and t.endswith("_live") for t in cleared_tags),
            "the live activity was ended at cycle end",
            f"cleared tags: {cleared_tags}",
        )
        checks.check(
            any(t and t.endswith("_lifecycle") for t in cleared_tags),
            "the lifecycle card was handed over to the live one",
            f"cleared tags: {cleared_tags}",
        )
        finish = [
            r for r in records
            if (r.get("data") or {}).get("tag", "").endswith("_lifecycle")
            and r["message"] != "clear_notification"
        ]
        checks.check(bool(finish), f"start/finish alerts were delivered ({len(finish)})")

        # #446 ordering: the finished alert must be delivered BEFORE the live
        # activity is ended, or the lock screen is momentarily empty. The unit
        # suite can only assert this by reading manager.py's source text
        # (test_issue_446...::..._in_source), because a mocked bus has no delivery
        # order to inspect. Here it is a fact about two captured records.
        live_clear = [
            r for r in clears
            if (r.get("data") or {}).get("tag", "").endswith("_live")
        ]
        if live_clear and finish:
            # Compare the EARLIEST of each: one dispatch fans out to every
            # configured target as separate tasks, so the per-target delivery
            # timestamps interleave with anything sent immediately after. What
            # #446 requires is that the finished alert was dispatched first, and
            # that is what the first delivery of each shows.
            checks.check(
                min(r["ts"] for r in finish) <= min(r["ts"] for r in live_clear),
                "the finished alert was delivered before the activity ended",
                "The activity was cleared first, so the lock screen went empty "
                "before the finished notification arrived.",
            )
        else:
            checks.note(
                "ordering of finish vs activity-end not observable in this run "
                "(no finish alert yet - see register item 320)"
            )

        # ── the log ─────────────────────────────────────────────────────────
        print("\nHome Assistant log")
        log = LOG_FILE.read_text(errors="replace") if LOG_FILE.exists() else ""
        rejected = [
            line for line in log.splitlines()
            if "MultipleInvalid" in line or "not a valid value" in line
        ]
        checks.check(
            not rejected,
            "no service call was rejected by a schema",
            "\n".join(rejected[-6:]),
        )
        ours = [
            line for line in log.splitlines()
            if "ERROR" in line and "ha_washdata" in line
        ]
        checks.check(not ours, "no ha_washdata errors", "\n".join(ours[-8:]))
        tracebacks = log.count("Traceback (most recent call last)")
        checks.check(tracebacks == 0, f"no tracebacks ({tracebacks} found)")

    print(
        f"\n{checks.count - len(checks.failures)}/{checks.count} checks passed"
        + (f"\nfailed: {'; '.join(checks.failures)}" if checks.failures else "")
    )
    return 1 if checks.failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
