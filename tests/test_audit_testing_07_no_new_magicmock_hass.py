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
"""Audit TESTING-07: no NEW ``hass = MagicMock()`` in the tests.

A MagicMock hass lies: an unconfigured power read is a silent 1 W, a lock is a
no-op, ``async_update_entry`` changes nothing, every service payload is accepted
(register item 316), and module-local copies disagree on whether a scheduled
coroutine runs at all. The existing sites are a known debt, recorded per file in
``tests/fixtures/magicmock_hass_allowlist.json``; this ratchet stops the count
from rising. For new tests use the real ``hass`` fixture (``setup_washdata_entry``
in conftest.py, or ``tests/real_manager.py``), or conftest's ``mock_hass``.

A file that drops sites passes; shrink its entry with
``python tests/test_audit_testing_07_no_new_magicmock_hass.py --update``.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
ALLOWLIST = TESTS / "fixtures" / "magicmock_hass_allowlist.json"

# `hass = MagicMock(...)`, `self.hass = Mock()`, `mock_hass = AsyncMock()`, ...
_SITE = re.compile(
    r"\b\w*hass\w*\s*=\s*(?:unittest\.mock\.|mock\.)?"
    r"(?:MagicMock|Mock|AsyncMock|NonCallableMagicMock)\("
)


def count_sites() -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(TESTS.rglob("*.py")):
        if path == Path(__file__).resolve():
            continue  # this file spells the pattern out on purpose
        n = len(_SITE.findall(path.read_text(encoding="utf-8")))
        if n:
            counts[path.relative_to(TESTS).as_posix()] = n
    return counts


def test_no_new_magicmock_hass_sites() -> None:
    allowed: dict[str, int] = json.loads(ALLOWLIST.read_text(encoding="utf-8"))
    grown = {
        rel: (n, allowed.get(rel, 0))
        for rel, n in count_sites().items()
        if n > allowed.get(rel, 0)
    }
    assert not grown, (
        "new `hass = MagicMock()` site(s) (found, allowed): "
        + ", ".join(f"tests/{rel} {n}>{a}" for rel, (n, a) in sorted(grown.items()))
        + ". A MagicMock hass accepts any call, so the test can pass against code "
        "Home Assistant rejects. Use the real `hass` fixture (setup_washdata_entry in "
        "tests/conftest.py, or tests/real_manager.py) or conftest's mock_hass."
    )


def test_the_pattern_catches_the_usual_spellings() -> None:
    for line in ("hass = MagicMock()", "    self.hass = Mock()",
                 "mock_hass = MagicMock(spec=HomeAssistant)", "hass=AsyncMock()",
                 "flow.hass = unittest.mock.MagicMock()"):
        assert _SITE.search(line), line
    for line in ("hass = real_hass", "manager = MagicMock()", "hass.data = MagicMock()"):
        assert not _SITE.search(line), line


if __name__ == "__main__" and "--update" in sys.argv:
    ALLOWLIST.write_text(json.dumps(count_sites(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {ALLOWLIST}")
