"""DeviceLoggerAdapter attaches a structured device field for the Logs filter."""
from __future__ import annotations

import logging

from custom_components.ha_washdata.log_utils import DeviceLoggerAdapter


def test_adapter_prefixes_message_and_sets_structured_device(caplog):
    logger = logging.getLogger("custom_components.ha_washdata.test_adapter")
    adapter = DeviceLoggerAdapter(logger, "Kitchen dishwasher")
    with caplog.at_level(logging.INFO, logger=logger.name):
        adapter.info("something happened")
    rec = caplog.records[-1]
    # Human-readable prefix preserved …
    assert rec.getMessage() == "[Kitchen dishwasher] something happened"
    # … and the structured field the Logs page filters on is set.
    assert getattr(rec, "wd_device", None) == "Kitchen dishwasher"


def test_adapter_preserves_caller_extra(caplog):
    logger = logging.getLogger("custom_components.ha_washdata.test_adapter2")
    adapter = DeviceLoggerAdapter(logger, "Dryer")
    with caplog.at_level(logging.INFO, logger=logger.name):
        adapter.info("msg", extra={"custom_field": 7})
    rec = caplog.records[-1]
    assert getattr(rec, "wd_device", None) == "Dryer"
    assert getattr(rec, "custom_field", None) == 7


def test_adapter_does_not_mutate_caller_extra():
    """The caller's extra dict must be left untouched (adapter copies before override)."""
    logger = logging.getLogger("custom_components.ha_washdata.test_adapter3")
    adapter = DeviceLoggerAdapter(logger, "Washer")
    caller_extra = {"custom_field": 1}
    _, kwargs = adapter.process("msg", {"extra": caller_extra})
    # The adapter injected its own field into the returned copy …
    assert kwargs["extra"]["wd_device"] == "Washer"
    assert kwargs["extra"]["custom_field"] == 1
    # … but the caller's original dict was not mutated.
    assert caller_extra == {"custom_field": 1}
    assert "wd_device" not in caller_extra
