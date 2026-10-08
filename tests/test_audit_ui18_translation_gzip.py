# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Audit 2026-10-02 UI-18: panel translation files are served pre-compressed.

The panel fetches its English and user-language dictionaries (130-250 KB each) on
every cold load. aiohttp serves a ``.gz`` sibling only when one exists, and the
translations directory was registered without one, so both went out uncompressed.
"""

from __future__ import annotations

import gzip
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from custom_components.ha_washdata import frontend as fe


def _tree(tmp_path: Path) -> Path:
    d = tmp_path / "translations" / "panel"
    d.mkdir(parents=True)
    (d / "en.json").write_text('{"msg": {"hello": "Hello"}}\n' * 200)
    (d / "de.json").write_text('{"msg": {"hello": "Hallo"}}\n' * 200)
    return d


def test_every_translation_gets_a_gz_that_decompresses_to_it(tmp_path: Path) -> None:
    d = _tree(tmp_path)
    assert fe._prepare_translations(d) is True
    for name in ("en.json", "de.json"):
        gz = d / f"{name}.gz"
        assert gz.is_file()
        assert gzip.decompress(gz.read_bytes()) == (d / name).read_bytes()


def test_a_current_gz_is_kept(tmp_path: Path) -> None:
    """35 files at level 9 cost ~0.6 s per start; an unchanged one is not redone."""
    d = _tree(tmp_path)
    fe._prepare_translations(d)
    gz = d / "en.json.gz"
    stamp = gz.stat().st_mtime_ns
    old = stamp - 10_000_000_000
    os.utime(gz, ns=(old, old))
    fe._prepare_translations(d)
    assert gz.stat().st_mtime_ns == old


def test_a_stale_gz_is_rebuilt_even_when_it_looks_newer(tmp_path: Path) -> None:
    """Freshness is decided by content: an update can restore an older source
    mtime from the release archive, so a newer .gz is not proof of a current one."""
    d = _tree(tmp_path)
    fe._prepare_translations(d)
    src, gz = d / "de.json", d / "de.json.gz"
    src.write_text('{"msg": {"hello": "Servus"}}\n')
    old = gz.stat().st_mtime - 100
    os.utime(src, (old, old))
    fe._prepare_translations(d)
    assert gzip.decompress(gz.read_bytes()) == src.read_bytes()


def test_a_corrupt_gz_is_rebuilt(tmp_path: Path) -> None:
    d = _tree(tmp_path)
    (d / "en.json.gz").write_bytes(b"\x1f\x8b not really gzip")
    fe._prepare_translations(d)
    assert gzip.decompress((d / "en.json.gz").read_bytes()) == (d / "en.json").read_bytes()


def test_an_orphaned_gz_is_removed(tmp_path: Path) -> None:
    """aiohttp serves the .gz sibling on its own existence, so a dropped
    language's leftover archive would keep being served."""
    d = _tree(tmp_path)
    fe._prepare_translations(d)
    (d / "de.json").unlink()
    fe._prepare_translations(d)
    assert not (d / "de.json.gz").exists()
    assert (d / "en.json.gz").exists()


def test_missing_directory_is_reported_not_raised(tmp_path: Path) -> None:
    assert fe._prepare_translations(tmp_path / "nope") is False


def test_cache_buster_ignores_the_generated_gz(tmp_path: Path, monkeypatch) -> None:
    """The .gz siblings are written at startup; folding their mtime into the
    cache buster would change the panel URL (and refetch it) on every restart."""
    base = tmp_path / "ha_washdata"
    (base / "www").mkdir(parents=True)
    (base / "manifest.json").write_text('{"version": "1.0.0"}')
    (base / "www" / fe.CARD_NAME).write_text("card")
    d = base / "translations" / "panel"
    d.mkdir(parents=True)
    (d / "en.json").write_text("{}")
    monkeypatch.setattr(fe, "__file__", str(base / "frontend.py"))
    before = fe.get_cache_buster(fe.CARD_NAME)
    gz = d / "en.json.gz"
    gz.write_bytes(gzip.compress(b"{}"))
    future = os.stat(d / "en.json").st_mtime_ns + 60_000_000_000
    os.utime(gz, ns=(future, future))
    assert fe.get_cache_buster(fe.CARD_NAME) == before


async def test_panel_registration_compresses_the_translations(tmp_path: Path, monkeypatch) -> None:
    """The wiring: the translations directory goes through _prepare_translations
    (in the executor) before it is registered as a static path."""
    js = tmp_path / "panel.js"
    js.write_text("x")
    monkeypatch.setattr(fe, "_prepare_asset", lambda name: js)
    monkeypatch.setattr(fe, "get_cache_buster", lambda name=None: "v")
    seen: list[Path] = []
    monkeypatch.setattr(fe, "_prepare_translations", lambda d: seen.append(d) or True)
    register = AsyncMock()
    monkeypatch.setattr(fe, "_async_register_path", register)
    import homeassistant.components.frontend as ha_frontend

    monkeypatch.setattr(ha_frontend, "async_register_built_in_panel", MagicMock())

    hass = MagicMock()
    hass.data = {}

    async def _executor(fn, *args):
        return fn(*args)

    hass.async_add_executor_job = _executor
    assert await fe._do_register_panel(hass, js) is True
    assert seen and seen[0].name == fe.PANEL_TRANSLATIONS_DIRNAME
    urls = [c.args[1] for c in register.await_args_list]
    assert fe.PANEL_TRANSLATIONS_URL in urls
