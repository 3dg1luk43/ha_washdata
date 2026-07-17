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
"""Opt-in, NumPy-only ML models for WashData (experimental).

Models are trained offline in the ml_washdata lab and embedded here as base64
blobs. They are inert unless the user enables them. See engine.py and README.md.
"""

from .engine import (
    CONF_ENABLE_ML_MODELS,
    available_models,
    ml_models_enabled,
    resolve_regressor,
    resolve_scorer,
)

__all__ = [
    "CONF_ENABLE_ML_MODELS",
    "available_models",
    "ml_models_enabled",
    "resolve_regressor",
    "resolve_scorer",
]
