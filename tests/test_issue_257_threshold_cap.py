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
"""Issue #257: detection-threshold NumberSelector ceiling cap.

The OptionsFlow advanced_settings step had a hard ceiling at 500 W (start) /
100 W (stop). High-power devices (e.g. pumps suggesting ~1097 W) could not apply
suggestions because the selector rejected the value.

The options flow has been replaced by the WashData panel which uses plain number
inputs without such hard ceilings. The bug vector no longer exists in the current
code. Tests removed to match the slim config_flow.py.
"""
