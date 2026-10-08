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
"""Audit TESTING-14 / TESTING-18 / TESTING-07: the test runner's own configuration.

* TESTING-18: a hung test fails after 60 s in the fast tier and 900 s in the slow
  tier, however the suite is invoked.
* TESTING-14: skips are listed with their reason, and skips of tests that replay
  the gitignored cycle_data/ corpus are counted on one summary line.
* TESTING-07: conftest's ``mock_hass`` runs what it is asked to schedule.
"""
from __future__ import annotations

import shlex
from types import SimpleNamespace

import pytest

from tests import conftest


def test_fast_tier_timeout_and_skip_reasons_are_configured(pytestconfig) -> None:
    assert pytestconfig.getini("timeout") == "60"
    chars = [a[2:] for a in shlex.split(" ".join(pytestconfig.getini("addopts"))) if a.startswith("-r")]
    # Skips listed; and -r REPLACES pytest's default "fE", so those must stay too.
    assert chars and {"s", "f", "E"} <= set(chars[-1]), chars


class _Item:
    def __init__(self, *markers: str) -> None:
        self._markers = {m: getattr(pytest.mark, m).mark for m in markers}
        self.added: list = []

    def get_closest_marker(self, name):
        return self._markers.get(name)

    def add_marker(self, mark) -> None:
        self.added.append(mark.mark)


def test_slow_tests_get_the_long_timeout_and_nothing_else_does() -> None:
    slow, bench, fast, pinned = _Item("slow"), _Item("benchmark"), _Item(), _Item("slow", "timeout")
    conftest.pytest_collection_modifyitems(None, [slow, bench, fast, pinned])
    assert [m.args for m in slow.added] == [(conftest.SLOW_TEST_TIMEOUT_S,)]
    assert [m.args for m in bench.added] == [(conftest.SLOW_TEST_TIMEOUT_S,)]
    assert fast.added == [] and pinned.added == []  # an explicit timeout wins


def test_corpus_skips_are_counted_on_one_line(tmp_path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_replay.py").write_text('DATA = "cycle_data/me/x.json"\n')
    (tmp_path / "tests/test_node.py").write_text('# needs node\n')
    skipped = [
        SimpleNamespace(location=("tests/test_replay.py", 1, "a"), nodeid="tests/test_replay.py::a"),
        SimpleNamespace(location=("tests/test_replay.py", 2, "b"), nodeid="tests/test_replay.py::b"),
        SimpleNamespace(location=("tests/test_node.py", 1, "c"), nodeid="tests/test_node.py::c"),
    ]
    lines: list[str] = []
    reporter = SimpleNamespace(
        stats={"skipped": skipped}, write_line=lambda line, **_: lines.append(line)
    )
    conftest.pytest_terminal_summary(reporter, 0, SimpleNamespace(rootpath=tmp_path))
    assert lines == [
        "cycle_data: 2 test(s) skipped in 1 module(s) that replay the private corpus "
        "(cycle_data/ NOT present); reasons in the SKIPPED lines"
    ]

    lines.clear()
    reporter.stats = {"skipped": skipped[2:]}
    conftest.pytest_terminal_summary(reporter, 0, SimpleNamespace(rootpath=tmp_path))
    assert lines == []  # a skip that has nothing to do with the corpus says nothing


async def test_mock_hass_runs_a_scheduled_coroutine(mock_hass) -> None:
    ran: list[str] = []

    async def _work() -> None:
        ran.append("done")

    task = mock_hass.async_create_task(_work())
    await task
    assert ran == ["done"]


def test_mock_hass_runs_a_coroutine_scheduled_from_sync_code(mock_hass) -> None:
    ran: list[str] = []

    async def _child() -> None:
        ran.append("child")

    async def _work() -> None:
        ran.append("parent")
        mock_hass.async_create_task(_child())

    mock_hass.async_create_task(_work())
    assert ran == ["parent", "child"]
