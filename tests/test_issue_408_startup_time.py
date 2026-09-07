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
"""Regression guards for issue #408: WashData reported a 65 s startup time.

``hass.import_executor`` is ``max_workers=1`` and shared with every other
integration importing during startup, so an *awaited* job on it costs however
deep that queue happens to be. Real user diagnostics showed 35-95 s of
``config_entry_setup`` against well under a second of actual work.

``async_setup_entry`` opens with the ML module warm-up (issue #328) on that
executor. Two properties keep it out of the user's startup number, and both are
easy to regress by "simplifying" the helper:

1. It is coalesced onto ONE shared future per HA instance. ``preload_models()``
   is idempotent, so a per-entry job buys nothing but another trip through the
   queue.
2. The wait is wrapped in ``async_pause_setup(WAIT_IMPORT_PACKAGES)`` so HA
   credits it back, the same way HA core reports its own heavy imports
   (``workday``, ``holiday``, ``stream``, ``mqtt``, ...). Without it the queue
   depth is billed to WashData as "Integration startup time".
"""

import asyncio
import time

import pytest
from homeassistant.core import CoreState
from homeassistant.setup import (
    SetupPhases,
    async_get_domain_setup_times,
    async_get_setup_timings,
    async_start_setup,
)

from custom_components.ha_washdata import (
    ML_PRELOAD_FUTURE_KEY,
    _async_preload_ml_modules,
)
from custom_components.ha_washdata.const import DOMAIN

# Long enough to dominate the ~1 ms of real warm-up work, short enough to stay
# in the fast suite.
_QUEUE_BLOCK_SECONDS = 0.3


def _count_import_jobs(hass):
    """Wrap the import executor so submissions can be counted (real Futures kept)."""
    calls: list[str] = []
    original = hass.async_add_import_executor_job

    def counting(target, *args):
        calls.append(getattr(target, "__name__", repr(target)))
        return original(target, *args)

    hass.async_add_import_executor_job = counting
    return calls


async def test_ml_warmup_is_one_job_for_many_entries(hass):
    """N config entries must share ONE import-executor job, not queue N of them."""
    calls = _count_import_jobs(hass)

    await asyncio.gather(*(_async_preload_ml_modules(hass) for _ in range(3)))
    # A device added later (sequential setup) must reuse the memo too.
    await _async_preload_ml_modules(hass)

    assert len(calls) == 1, (
        f"expected a single coalesced ML warm-up job, got {len(calls)}: {calls}. "
        "Each extra job is a full trip through HA's single-threaded import "
        "executor, which is what made setup look like it took a minute (#408)."
    )
    assert hass.data.get(ML_PRELOAD_FUTURE_KEY) is not None


async def test_ml_warmup_wait_is_credited_back(hass):
    """The import-queue wait must be recorded as a negative (credited) phase."""
    hass.set_state(CoreState.starting)  # HA only records timings while starting

    # Occupy the single import-executor thread, exactly as a real startup does.
    hass.import_executor.submit(time.sleep, _QUEUE_BLOCK_SECONDS)

    group = "entry_408"
    started = time.perf_counter()
    with async_start_setup(
        hass, integration=DOMAIN, phase=SetupPhases.CONFIG_ENTRY_SETUP, group=group
    ):
        await _async_preload_ml_modules(hass)
    elapsed = time.perf_counter() - started

    phases = dict(async_get_domain_setup_times(hass, DOMAIN)[group])
    credit = phases.get(SetupPhases.WAIT_IMPORT_PACKAGES)

    assert credit is not None, (
        "no WAIT_IMPORT_PACKAGES phase recorded: the ML warm-up await is no "
        "longer wrapped in async_pause_setup(), so HA bills the import-executor "
        "queue depth to WashData as startup time (#408)."
    )
    assert credit < 0, f"expected a negative credit, got {credit}"
    # The blocked executor is what we waited on, so the credit must cover it.
    assert -credit >= _QUEUE_BLOCK_SECONDS * 0.5, (
        f"credit {credit:.3f}s does not account for the {_QUEUE_BLOCK_SECONDS}s "
        f"blocked import executor (elapsed {elapsed:.3f}s)"
    )

    # What HA actually displays: elapsed minus the credited waits.
    displayed = async_get_setup_timings(hass)[DOMAIN]
    assert displayed < elapsed, (
        f"displayed startup time {displayed:.3f}s did not drop below the "
        f"{elapsed:.3f}s spent waiting on the import queue"
    )


async def test_ml_warmup_failure_clears_memo_for_retry(hass):
    """A failed warm-up must not be memoized forever (broken install can heal)."""

    def boom(*_args):
        future = hass.loop.create_future()
        future.set_exception(RuntimeError("import exploded"))
        return future

    hass.async_add_import_executor_job = boom

    await _async_preload_ml_modules(hass)  # must not raise: ML just stays inert

    assert ML_PRELOAD_FUTURE_KEY not in hass.data, (
        "a failed warm-up stayed memoized, so no later setup or reload can retry it"
    )
