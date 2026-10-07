"""Audit 2026-10-02 PLATFORM-10: every service has a schema and one resolver.

No service passed `schema=`: `profile_name: 123` raised AttributeError, a
non-numeric `trim_start_s` a ValueError traceback, `unlabel_cycles: "false"` was
truthy; nine handlers raised a bare ValueError for an unknown device, and every
one took an arbitrary entry of the device rather than ours.
"""

from __future__ import annotations

import pytest
import voluptuous as vol
import yaml
from pathlib import Path

from custom_components.ha_washdata import _SERVICE_SCHEMAS

_YAML = Path(__file__).resolve().parents[1] / "custom_components" / "ha_washdata" / "services.yaml"


def test_every_service_in_services_yaml_has_a_schema() -> None:
    assert set(yaml.safe_load(_YAML.read_text())) == set(_SERVICE_SCHEMAS)


def test_values_are_typed_instead_of_crashing_the_handler() -> None:
    out = _SERVICE_SCHEMAS["label_cycle"]({"device_id": "d", "cycle_id": "c", "profile_name": 123})
    assert out["profile_name"] == "123"
    assert _SERVICE_SCHEMAS["delete_profile"](
        {"device_id": "d", "profile_name": "P", "unlabel_cycles": "false"}
    )["unlabel_cycles"] is False
    with pytest.raises(vol.Invalid):
        _SERVICE_SCHEMAS["trim_cycle"]({"device_id": "d", "cycle_id": "c", "trim_start_s": "x"})
    with pytest.raises(vol.Invalid):
        _SERVICE_SCHEMAS["auto_label_cycles"]({"device_id": "d", "confidence_threshold": 7})
    with pytest.raises(vol.Invalid):
        _SERVICE_SCHEMAS["record_start"]({})


def test_an_undeclared_key_an_automation_already_sends_is_kept() -> None:
    out = _SERVICE_SCHEMAS["submit_cycle_feedback"]({"cycle_id": "c", "dismiss": "true"})
    assert out["dismiss"] is True


def test_auto_label_accepts_an_empty_threshold() -> None:
    # services.yaml says "Leave empty to use the device's Auto-Label Confidence";
    # an automation passing null must reach the handler's None fallback.
    out = _SERVICE_SCHEMAS["auto_label_cycles"]({"device_id": "d", "confidence_threshold": None})
    assert out["confidence_threshold"] is None
    assert _SERVICE_SCHEMAS["auto_label_cycles"](
        {"device_id": "d", "confidence_threshold": "0.8"}
    )["confidence_threshold"] == 0.8


@pytest.mark.parametrize(
    ("service", "data"),
    [
        ("auto_label_cycles", {"confidence_threshold": 10**400}),
        ("trim_cycle", {"cycle_id": "c", "trim_start_s": 10**400}),
        ("trim_cycle", {"cycle_id": "c", "trim_end_s": 10**400}),
        ("submit_cycle_feedback", {"cycle_id": "c", "corrected_duration": 10**400}),
    ],
)
def test_an_oversized_integer_is_a_validation_error(service, data) -> None:
    # vol.Coerce(float) lets OverflowError escape (voluptuous catches only
    # ValueError/TypeError), so the call failed with a traceback, not Invalid.
    with pytest.raises(vol.Invalid):
        _SERVICE_SCHEMAS[service]({"device_id": "d", **data})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -60, 90000])
def test_corrected_duration_is_held_to_the_selector_range(bad) -> None:
    # YAML automations bypass services.yaml's 0-86400 selector, and NaN, inf or a
    # negative value reached the stored cycle's duration.
    with pytest.raises(vol.Invalid):
        _SERVICE_SCHEMAS["submit_cycle_feedback"]({"cycle_id": "c", "corrected_duration": bad})
    out = _SERVICE_SCHEMAS["submit_cycle_feedback"]({"cycle_id": "c", "corrected_duration": "3600"})
    assert out["corrected_duration"] == 3600.0
