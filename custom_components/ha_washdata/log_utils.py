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
        # Also attach the device name as a structured field (record.wd_device) so
        # the Logs page can filter by device, not just parse the "[device]" prefix.
        src = kwargs.get("extra")
        # Shallow-copy so we never mutate the caller's dict; the adapter owns the
        # reserved wd_device field but preserves every other caller-supplied extra.
        extra = dict(src) if isinstance(src, dict) else {}
        extra["wd_device"] = device
        kwargs["extra"] = extra
        return f"[{device}] {msg}", kwargs
