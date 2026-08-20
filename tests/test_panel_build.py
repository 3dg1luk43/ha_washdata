# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gate for the minified panel/card build artifacts.

The readable sources in ``www/`` are the source of truth; ``devtools/build_panel.mjs``
produces the ``*.min.js`` files that are actually served, and records the source
hashes in ``build-manifest.json``.

Two layers protect that arrangement, and this module tests both:

1. **Runtime** -- ``frontend._resolve_asset`` only serves an artifact it can prove
   was built from the bytes currently on disk, so a forgotten rebuild degrades to a
   larger download rather than to stale code.
2. **CI** -- ``test_build_artifacts_are_current`` fails when an artifact is stale,
   so the rebuild is not forgotten in the first place.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from custom_components.ha_washdata import frontend as fe

WWW = Path(fe.__file__).parent / "www"
MANIFEST = WWW / fe.BUILD_MANIFEST_NAME
REBUILD_HINT = "Run: node devtools/build_panel.mjs"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def test_build_manifest_exists() -> None:
    """The manifest is committed; without it nothing can be verified."""
    assert MANIFEST.is_file(), f"{MANIFEST} missing. {REBUILD_HINT}"
    assert _manifest().get("assets"), "manifest has no assets"


@pytest.mark.parametrize("source_name", ["ha-washdata-panel.js", "ha-washdata-card.js"])
def test_build_artifacts_are_current(source_name: str) -> None:
    """Fail when a source changed without rebuilding its minified artifact.

    This is the gate that keeps a stale bundle from being committed. It compares
    content hashes, not mtimes, so a git checkout (which rewrites mtimes) cannot
    produce a false pass or a false failure.
    """
    entry = _manifest()["assets"].get(source_name)
    assert entry, f"no manifest entry for {source_name}. {REBUILD_HINT}"

    source = WWW / source_name
    artifact = WWW / entry["artifact"]
    assert artifact.is_file(), f"{artifact.name} missing. {REBUILD_HINT}"

    assert _sha256(source) == entry["source_sha256"], (
        f"{source_name} changed since the last build. {REBUILD_HINT}"
    )
    assert _sha256(artifact) == entry["artifact_sha256"], (
        f"{entry['artifact']} was modified by hand. {REBUILD_HINT}"
    )
    # A build that got bigger means something went wrong with minification.
    assert artifact.stat().st_size < source.stat().st_size


@pytest.mark.parametrize("source_name", ["ha-washdata-panel.js", "ha-washdata-card.js"])
def test_minified_artifact_keeps_license_notice(source_name: str) -> None:
    """AGPL requires the notice to travel with the distributed artifact.

    esbuild strips `//` comments, so the banner is re-attached by the build script;
    this asserts it is actually there.
    """
    artifact = WWW / _manifest()["assets"][source_name]["artifact"]
    head = artifact.read_text(errors="replace")[:1200]
    assert "AGPL-3.0-or-later" in head
    assert "Lukas Bandura" in head


def test_resolver_uses_minified_build_when_current() -> None:
    """With everything in order, the smaller artifact is what gets served."""
    served = fe._resolve_asset("ha-washdata-panel.js")
    assert served.name.endswith(".min.js"), (
        "expected the minified build to be served; "
        f"got {served.name} (stale build?) {REBUILD_HINT}"
    )


def test_served_asset_report_names_the_variant_in_use(tmp_path: Path) -> None:
    """Both variants are served at the same URL, so this report is the only way to
    answer "is the minified bundle actually live?" without hashing an HTTP response.

    It is what `frontend_assets` in a diagnostics download shows, so a bug report
    tells us which bundle the reporter was running.
    """
    www = _fake_www(tmp_path)
    fe._SERVED_ASSETS.clear()

    fe._prepare_asset("asset.js", www)
    report = fe.served_asset_report()["asset.js"]
    assert report["minified"] is True
    assert report["serving"] == "asset.min.js"
    assert report["bytes"] == (www / "asset.min.js").stat().st_size

    # Editing the source without rebuilding flips it back to the readable file, and
    # the report has to say so rather than keep claiming the artifact.
    (www / "asset.js").write_text("// changed\nconst a = 2;\n")
    fe._prepare_asset("asset.js", www)
    report = fe.served_asset_report()["asset.js"]
    assert report["minified"] is False
    assert report["serving"] == "asset.js"

    # A copy is returned, so a caller cannot mutate the record it is reporting on.
    fe.served_asset_report()["asset.js"]["serving"] = "tampered"
    assert fe.served_asset_report()["asset.js"]["serving"] == "asset.js"


def _fake_www(tmp_path: Path) -> Path:
    """A miniature www/ with one source, one artifact and a matching manifest."""
    www = tmp_path / "www"
    www.mkdir()
    src = www / "asset.js"
    src.write_text("// readable source\nconst a = 1;\n")
    art = www / "asset.min.js"
    art.write_text("const a=1;\n")
    (www / fe.BUILD_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "assets": {
                    "asset.js": {
                        "artifact": "asset.min.js",
                        "source_sha256": _sha256(src),
                        "artifact_sha256": _sha256(art),
                    }
                }
            }
        )
    )
    return www


def test_resolver_prefers_artifact_in_fake_tree(tmp_path: Path) -> None:
    www = _fake_www(tmp_path)
    assert fe._resolve_asset("asset.js", www).name == "asset.min.js"


def test_resolver_falls_back_when_source_edited(tmp_path: Path) -> None:
    """Editing the source without rebuilding must serve the source, not the artifact."""
    www = _fake_www(tmp_path)
    (www / "asset.js").write_text("// readable source\nconst a = 2;  // new work\n")
    assert fe._resolve_asset("asset.js", www).name == "asset.js"


def test_resolver_falls_back_when_artifact_tampered(tmp_path: Path) -> None:
    www = _fake_www(tmp_path)
    (www / "asset.min.js").write_text("const a=999;\n")
    assert fe._resolve_asset("asset.js", www).name == "asset.js"


def test_resolver_falls_back_when_artifact_missing(tmp_path: Path) -> None:
    www = _fake_www(tmp_path)
    (www / "asset.min.js").unlink()
    assert fe._resolve_asset("asset.js", www).name == "asset.js"


def test_resolver_falls_back_without_manifest(tmp_path: Path) -> None:
    www = _fake_www(tmp_path)
    (www / fe.BUILD_MANIFEST_NAME).unlink()
    assert fe._resolve_asset("asset.js", www).name == "asset.js"


def test_ensure_gzip_matches_served_bytes(tmp_path: Path) -> None:
    """The .gz must decompress to exactly the file being served."""
    target = tmp_path / "asset.js"
    target.write_text("const x = 1;\n" * 500)
    fe._ensure_gzip(target)
    gz = tmp_path / "asset.js.gz"
    assert gz.is_file()
    assert gzip.decompress(gz.read_bytes()) == target.read_bytes()
    assert gz.stat().st_size < target.stat().st_size


def test_ensure_gzip_is_a_noop_when_fresh(tmp_path: Path) -> None:
    target = tmp_path / "asset.js"
    target.write_text("const x = 1;\n" * 500)
    fe._ensure_gzip(target)
    gz = tmp_path / "asset.js.gz"
    stamp = gz.stat().st_mtime_ns
    fe._ensure_gzip(target)
    assert gz.stat().st_mtime_ns == stamp


def test_ensure_gzip_regenerates_when_stale(tmp_path: Path) -> None:
    """A .gz older than its source is rewritten rather than served as-is.

    aiohttp serves a pre-compressed sibling on existence alone, so a stale .gz
    would be delivered (and cached for a month) as if it were the real file.
    """
    target = tmp_path / "asset.js"
    target.write_text("original\n")
    fe._ensure_gzip(target)
    gz = tmp_path / "asset.js.gz"

    target.write_text("updated content\n")
    old = target.stat().st_mtime - 10
    os.utime(gz, (old, old))

    fe._ensure_gzip(target)
    assert gzip.decompress(gz.read_bytes()) == b"updated content\n"


def test_ensure_gzip_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "asset.js"
    target.write_text("data\n" * 100)
    fe._ensure_gzip(target)
    assert not list(tmp_path.glob("*.tmp*"))


def test_ensure_gzip_survives_readonly_dir(tmp_path: Path) -> None:
    """A read-only install must serve uncompressed, not raise."""
    target = tmp_path / "asset.js"
    target.write_text("data\n")
    os.chmod(tmp_path, 0o500)
    try:
        fe._ensure_gzip(target)  # must not raise
    finally:
        os.chmod(tmp_path, 0o700)


def test_prepare_asset_returns_served_path_and_compresses(tmp_path: Path) -> None:
    www = _fake_www(tmp_path)
    served = fe._prepare_asset("asset.js", www)
    assert served.name == "asset.min.js"
    assert (www / "asset.min.js.gz").is_file()
