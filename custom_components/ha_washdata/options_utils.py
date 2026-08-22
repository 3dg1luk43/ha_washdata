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
"""Pure normalization helpers for ``entry.options``.

No Home Assistant imports, so the same rules apply on both write (the panel's
``ws_set_options``) and read-back (integration setup) without pulling the WS
layer into the setup path.

An option is either present with a real value or absent - absent is what makes
``options.get(key, DEFAULT)`` hand back the compiled default. A persisted
``None`` looks like "not set" to a human but is returned verbatim by ``.get()``,
so the ``float()`` / ``int()`` casts that build ``CycleDetectorConfig`` raise
``TypeError`` and the entry can never be set up again. Dropping the key is
therefore the faithful way to store "not set", and it is what every reader
already expects.

The rule is uniform: no option is ever persisted as ``None``. See
``NULL_MEANINGFUL_OPTION_KEYS`` for when a future key would need an exception,
and why none does today.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Dropping a key and storing None are the same thing to every reader that takes
# no default (`.get(key)`, `.get(key) or None`, `.get(key, "")` behind a falsy
# check) and to `.get(key) or entry.data.get(key)`, where the `or` swallows both.
# They differ only where an absent key falls through to a *non-None* default. The
# rule for a future key is therefore "does an absent key fall through to a
# non-None default, AND is None at that read a state the integration supports?" -
# not "is it nullable?".
#
# Today nothing qualifies, so the set is empty. `power_sensor` is the only key
# whose default argument is a non-None `entry.data` lookup
# (`options.get(CONF_POWER_SENSOR, entry.data.get(CONF_POWER_SENSOR))`), but a
# None there is not a supported state: the config flow requires a sensor and
# writes it into `entry.data`, the panel's field is not clearable (an emptied
# non-clearable field is dropped from the submission, and the selector-clearing
# loop in `ws_set_options` deliberately omits this key), and a stored None makes
# `async_setup` raise `AttributeError` inside `hass.states.get(None)`
# (`None.lower()`) - i.e. the same bricked entry this module exists to prevent,
# in a form no `except TypeError` can catch. Falling through to the sensor in
# `entry.data` is the correct recovery, so this key is stripped like the rest.
NULL_MEANINGFUL_OPTION_KEYS: frozenset[str] = frozenset()


def strip_null_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``options`` with unset-meaning ``None`` values dropped.

    The input is never mutated. Keys in ``NULL_MEANINGFUL_OPTION_KEYS`` keep their
    ``None``; every other key holding one is removed, so the read falls through to
    the compiled default instead of handing ``None`` to a numeric cast.
    """
    return {
        k: v
        for k, v in options.items()
        if v is not None or k in NULL_MEANINGFUL_OPTION_KEYS
    }


def has_null_options(options: Mapping[str, Any]) -> bool:
    """True when ``options`` holds at least one unset-meaning ``None``.

    Lets a caller skip the rewrite (and the entry reload it schedules) when there is
    nothing to clean.
    """
    return any(
        v is None and k not in NULL_MEANINGFUL_OPTION_KEYS for k, v in options.items()
    )
