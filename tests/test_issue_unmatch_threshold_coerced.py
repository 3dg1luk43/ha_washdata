"""A non-numeric `profile_unmatch_threshold` must not break matching (audit F7 finding).

The option was used raw: `float()` in the initial-commit check and `<` in the
unmatch check, so a hand-edited or imported string raised inside every match tick
and the cycle never got a program. Both reads (setup and reload) now coerce it.
"""

from __future__ import annotations

from custom_components.ha_washdata.const import (
    CONF_PROFILE_UNMATCH_THRESHOLD,
    DEFAULT_PROFILE_UNMATCH_THRESHOLD,
)

from .real_manager import boot, make_entry


async def test_a_garbage_unmatch_threshold_falls_back_to_the_default(hass):
    mgr = await boot(hass, make_entry(hass, {CONF_PROFILE_UNMATCH_THRESHOLD: "abc"}))
    assert mgr._unmatch_threshold == DEFAULT_PROFILE_UNMATCH_THRESHOLD  # noqa: SLF001


async def test_a_numeric_string_is_read_as_its_number(hass):
    mgr = await boot(hass, make_entry(hass, {CONF_PROFILE_UNMATCH_THRESHOLD: "0.5"}))
    assert mgr._unmatch_threshold == 0.5  # noqa: SLF001
