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
"""Issue #328: blocking ``importlib.import_module`` inside the event loop.

``resolve_scorer`` is called from the event loop (live matching, end detection,
quality gating) and used to import the embedded baseline module lazily on every
call, which Home Assistant flags as a blocking call. The modules are now imported
once from an import executor (``preload_models``) and cached, so no live call
imports anything.
"""
from __future__ import annotations

import importlib
import sys

import pytest

from custom_components.ha_washdata.ml import engine


@pytest.fixture(autouse=True)
def _warm_cache():
    """Every test below assumes setup already ran the executor preload."""
    # Drop any negative cache entry a failing test left behind so the tests stay
    # independent of each other.
    for name, module in list(engine._MODULE_CACHE.items()):
        if module is None:
            del engine._MODULE_CACHE[name]
    engine.preload_models()


@pytest.fixture(name="no_imports")
def _no_imports(monkeypatch: pytest.MonkeyPatch):
    """Make any ``importlib.import_module`` call fail loudly."""

    def _boom(name, *args, **kwargs):  # noqa: ANN001
        raise AssertionError(f"blocking import in the event loop: {name}")

    monkeypatch.setattr(importlib, "import_module", _boom)
    monkeypatch.setattr(engine.importlib, "import_module", _boom)


def test_preload_puts_every_model_in_sys_modules() -> None:
    engine.preload_models()
    package = engine.__package__
    for module_name in engine._MODEL_MODULES.values():
        assert f"{package}.{module_name}" in sys.modules
    for sibling in engine._SIBLING_MODULES:
        assert f"{package}.{sibling}" in sys.modules
    # The manifest read (a blocking file open) is warmed too.
    assert engine._MANIFEST_MODELS_CACHE is not None


@pytest.mark.parametrize("capability", ["quality", "live_match", "end"])
def test_resolve_scorer_after_preload_does_not_import(
    capability: str, no_imports: None
) -> None:
    score_fn, source = engine.resolve_scorer(capability, None)
    assert source == "baseline"
    assert 0.0 <= score_fn({}) <= 1.0


def test_stale_spec_schema_guard_does_not_import(no_imports: None) -> None:
    """The on-device schema guard also reads the baseline through the cache."""

    class _Store:
        def get_ml_model_versions(self):
            return {
                "end": {
                    "spec": {
                        "kind": "standardized_logistic",
                        "feature_columns": ["gone_column"],
                    }
                }
            }

    score_fn, source = engine.resolve_scorer("end", _Store())
    assert source == "baseline"
    assert 0.0 <= score_fn({}) <= 1.0


def test_failed_module_import_is_cached_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken model warns once instead of retrying the import per inference."""
    monkeypatch.setattr(engine, "_MODULE_CACHE", {})
    calls: list[str] = []

    def _fail(name):  # noqa: ANN001
        calls.append(name)
        raise ImportError("nope")

    monkeypatch.setattr(engine.importlib, "import_module", _fail)
    assert engine._load_model_module("cycle_end_detector_model") is None
    assert engine._load_model_module("cycle_end_detector_model") is None
    assert len(calls) == 1
    score_fn, source = engine.resolve_scorer("end", None)
    assert (score_fn, source) == (None, None)


async def test_setup_preloads_through_the_import_executor() -> None:
    """Setup must warm the cache off the loop, and survive a preload failure."""
    from custom_components.ha_washdata import _async_preload_ml_modules

    calls: list[object] = []

    class _Hass:
        async def async_add_import_executor_job(self, target, *args):
            calls.append(target)
            return target(*args)

    await _async_preload_ml_modules(_Hass())
    assert len(calls) == 1
    assert engine._MANIFEST_MODELS_CACHE is not None

    class _BrokenHass:
        async def async_add_import_executor_job(self, target, *args):
            raise RuntimeError("executor gone")

    # A failed preload only means ML stays inert; it must never fail setup.
    await _async_preload_ml_modules(_BrokenHass())
