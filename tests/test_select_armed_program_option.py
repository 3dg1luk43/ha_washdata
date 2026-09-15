# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The program select must show a program armed for the next cycle (#411).

`_update_state` reads the armed program through
``getattr(self._manager, "armed_program", None)``. `WashDataManager.armed_program`
is a *property*, so that yields ``str | None`` and the
``armed in self._attr_options`` membership test works. This was raised in review
as storing a bound method, which would make the branch dead and leave the select
on Auto forever; these tests pin the real behaviour so the question is settled by
execution rather than by reading.

Adding a call - ``armed_program()`` - would raise
``TypeError: 'str' object is not callable`` as soon as a program is armed, so
`test_arming_a_program_shows_it_in_the_select` fails under that change too.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from custom_components.ha_washdata.select import OPTION_AUTO, WashDataProgramSelect


def _entry() -> Any:
    entry = MagicMock()
    entry.entry_id = "entry-1"
    entry.title = "Test Washer"
    return entry


class _FakeManager:
    """Mirrors the manager's real accessor shapes.

    `armed_program` and `manual_program_active` are declared as properties here
    for the same reason they are on `WashDataManager`: the behaviour under test is
    what `select.py`'s `getattr(...)` sees, and a plain attribute would not
    exercise that. `test_manager_armed_program_is_a_property_not_a_method` pins
    the real class to the same shape.
    """

    entry_id = "entry-1"
    device_type = "washing_machine"

    def __init__(self, *, armed: str | None, current: str, manual: bool) -> None:
        self._armed_program = armed
        self._current = current
        self._manual = manual
        self.profile_store = MagicMock()
        self.profile_store.list_profiles.return_value = [
            {"name": "Eco 50"}, {"name": "Cotton 60"}
        ]

    @property
    def current_program(self) -> str:
        return self._current

    @property
    def manual_program_active(self) -> bool:
        return self._manual

    @property
    def armed_program(self) -> str | None:
        return getattr(self, "_armed_program", None)


def _manager(*, armed: str | None, current: str = "off",
             manual: bool = False) -> Any:
    return _FakeManager(armed=armed, current=current, manual=manual)


def _select(mgr: Any) -> WashDataProgramSelect:
    sel = WashDataProgramSelect(mgr, _entry())
    sel.async_write_ha_state = MagicMock()
    return sel


def test_manager_armed_program_is_a_property_not_a_method() -> None:
    """The premise the review finding rested on."""
    from custom_components.ha_washdata.manager import WashDataManager

    assert isinstance(
        WashDataManager.__dict__["armed_program"], property
    ), "armed_program must stay a property; select.py reads it via getattr"


def test_arming_a_program_shows_it_in_the_select() -> None:
    """The armed branch: idle appliance, program pinned for the next cycle."""
    sel = _select(_manager(armed="Eco 50"))
    sel._update_state()

    assert sel._attr_current_option == "Eco 50"
    # Not a bound method, and not silently coerced to a string.
    assert isinstance(sel._attr_current_option, str)


def test_no_armed_program_falls_back_to_auto() -> None:
    """The control, so the test above cannot pass by always returning the name."""
    sel = _select(_manager(armed=None))
    sel._update_state()

    assert sel._attr_current_option == OPTION_AUTO


def test_an_armed_program_that_no_longer_exists_falls_back_to_auto() -> None:
    """`armed in self._attr_options` is what makes a deleted profile safe."""
    sel = _select(_manager(armed="Deleted Program"))
    sel._update_state()

    assert sel._attr_current_option == OPTION_AUTO


def test_a_manual_override_still_wins_over_an_armed_program() -> None:
    """Branch order: a program running under manual override is shown, not the arm."""
    sel = _select(_manager(armed="Eco 50", current="Cotton 60", manual=True))
    sel._update_state()

    assert sel._attr_current_option == "Cotton 60"
