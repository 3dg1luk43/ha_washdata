"""Audit 2026-10-02 PLATFORM-16: the conftest service guard fails closed."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.conftest import _validate_recorded_service_calls


def test_an_unknown_outbound_domain_fails_instead_of_passing_unchecked() -> None:
    hass = MagicMock()
    hass.services.async_call("light", "turn_on", {"entity_id": "light.x"})
    with pytest.raises(AssertionError, match="no schema registered"):
        _validate_recorded_service_calls(hass)


def test_our_own_services_are_checked_against_their_schemas() -> None:
    hass = MagicMock()
    hass.services.async_call("ha_washdata", "trim_cycle", {"cycle_id": "c"})  # no device_id
    with pytest.raises(AssertionError):
        _validate_recorded_service_calls(hass)
