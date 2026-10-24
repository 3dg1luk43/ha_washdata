"""A non-numeric `profile_unmatch_threshold` must not break matching (audit F7 finding).

The option was used raw: `float()` in the initial-commit check and `<` in the
unmatch check, so a hand-edited or imported string raised inside every match tick
and the cycle never got a program. Both reads (setup and reload) then coerced it.

Since 0.5.9 it is not a setting: forcing 0.35 on every corpus entry that set
another value was neutral or better on the LOO switching replay, so a stored
value of any kind is ignored and the threshold is the fixed default.
"""

from __future__ import annotations

import pytest

from custom_components.ha_washdata.const import (
    CONF_PROFILE_UNMATCH_THRESHOLD,
    DEFAULT_PROFILE_UNMATCH_THRESHOLD,
)

from .real_manager import boot, make_entry


@pytest.mark.parametrize("stored", ["abc", "0.5", 0.25, None])
async def test_a_stored_unmatch_threshold_is_ignored(hass, stored):
    options = {} if stored is None else {CONF_PROFILE_UNMATCH_THRESHOLD: stored}
    mgr = await boot(hass, make_entry(hass, options))
    assert DEFAULT_PROFILE_UNMATCH_THRESHOLD == 0.35
    assert mgr._unmatch_threshold == DEFAULT_PROFILE_UNMATCH_THRESHOLD  # noqa: SLF001
