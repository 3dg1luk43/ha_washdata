"""Audit 2026-10-02 PLATFORM-06: `binary_sensor.*_running` means "a cycle is in progress".

It used to be on only in `running`, so it turned off during every soak, pause and
the end wait - and automations that treat "off" as "done" fired mid-cycle.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata.binary_sensor import WasherRunningBinarySensor
from custom_components.ha_washdata.const import (
    STATE_ANTI_WRINKLE,
    STATE_CLEAN,
    STATE_ENDING,
    STATE_FINISHED,
    STATE_IDLE,
    STATE_OFF,
    STATE_PAUSED,
    STATE_RUNNING,
    STATE_STARTING,
    STATE_USER_PAUSED,
)


def _sensor(state: str) -> WasherRunningBinarySensor:
    manager = MagicMock()
    manager.check_state.return_value = state
    entry = MagicMock()
    entry.entry_id = "e"
    entry.title = "Washer"
    return WasherRunningBinarySensor(manager, entry)


@pytest.mark.parametrize(
    "state", [STATE_RUNNING, STATE_PAUSED, STATE_USER_PAUSED, STATE_ENDING]
)
def test_on_while_a_cycle_is_in_progress(state: str) -> None:
    assert _sensor(state).is_on is True


@pytest.mark.parametrize(
    "state",
    [STATE_OFF, STATE_IDLE, STATE_STARTING, STATE_FINISHED, STATE_CLEAN, STATE_ANTI_WRINKLE],
)
def test_off_otherwise(state: str) -> None:
    assert _sensor(state).is_on is False
