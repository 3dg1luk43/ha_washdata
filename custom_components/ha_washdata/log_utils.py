"""Logging utilities for WashData."""
from __future__ import annotations

import logging


class DeviceLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that prepends the device name to every log message.

    Usage::

        _LOGGER = logging.getLogger(__name__)

        class MyClass:
            def __init__(self, device_name: str) -> None:
                self._logger = DeviceLoggerAdapter(_LOGGER, device_name)

            def do_thing(self) -> None:
                self._logger.info("Something happened")
                # emits: "[My Device] Something happened"
    """

    def __init__(self, logger: logging.Logger, device_name: str) -> None:
        super().__init__(logger, {"device_name": device_name})

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        device = self.extra.get("device_name") or "unknown"  # type: ignore[union-attr]
        return f"[{device}] {msg}", kwargs
