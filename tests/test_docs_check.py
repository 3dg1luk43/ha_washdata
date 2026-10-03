"""devtools/docs_check.py: the docs consistency gate (audit F9).

The first test is the gate itself: the real CLAUDE.md / README.md / register against the
real code and the committed baseline. The rest pin each check's behaviour on a tiny
synthetic tree, so a regex that silently stops matching cannot turn the gate into a no-op.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "devtools" / "docs_check.py"
_spec = importlib.util.spec_from_file_location("wd_docs_check", _PATH)
dc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = dc  # dataclasses resolve their module through sys.modules
_spec.loader.exec_module(dc)


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.mark.skipif(
    shutil.which("git") is None or not (dc.ROOT / ".git").exists(), reason="needs a git checkout"
)
def test_docs_match_the_code() -> None:
    report = dc.run()
    assert report.failures == [], "\n".join(
        [*report.failures, "run: python3 devtools/docs_check.py", dc.FIX_HINT]
    )
    # The checks ran against something: an empty pass would also be "clean".
    assert report.stats["anchors checked"] > 100
    assert report.stats["constant claims checked"] >= 10
    assert report.stats["register ids"] > 300


def test_anchor_check(tmp_path: Path) -> None:
    _write(tmp_path, f"{dc.COMPONENT}/mod.py", (
        "LIMIT_LOW = 1\n"
        "first, second = 1, 2\n"
        "def alive():\n    pass\n"
        "class Box:\n"
        "    size: int = 3\n"
        "    def lid(self):\n        self.hinge = 1\n"
        "call(\n    kwarg=1,\n)\n"
    ))
    _write(tmp_path, f"{dc.COMPONENT}/ml/engine.py", "def resolve_scorer():\n    pass\n")
    _write(tmp_path, "tests/test_x.py", "def test_ok():\n    pass\n")
    _write(tmp_path, "CLAUDE.md", (
        "`mod.py:alive` `mod.py::Box.lid` `mod.py:Box.hinge` `mod.py:Box.size` `mod.py:second`\n"
        "`mod.py:LIMIT_*` `ml/engine.py:resolve_scorer` `test_x.py::test_ok` `mod.py:123` `mod.py:L9`\n"
        "`mod.py:dead` `mod.py:Box.nope` `mod.py:kwarg` `missing.py:foo` `mod.py:NOPE_*`\n"
        "```\npytest tests/test_x.py::test_function_name\n```\n"
    ))
    report = dc.Report()
    dc.check_anchors(tmp_path, report)
    dead = {anchor for _doc, anchor, _line, _why in report.dead_anchors}
    assert dead == {"mod.py:dead", "mod.py:Box.nope", "mod.py:kwarg", "missing.py:foo", "mod.py:NOPE_*"}
    assert report.stats["anchors checked"] == 13  # line anchors are not counted


def test_constant_check(tmp_path: Path) -> None:
    _write(tmp_path, f"{dc.COMPONENT}/const.py", (
        "MATCH_MARGIN = 0.05\nML_FLAG = False\nSTORAGE_VERSION = 16\n"
        "CONFIG_ENTRY_VERSION = 3\nCONFIG_ENTRY_MINOR_VERSION = 11\nWINDOW_S = 2 * 60\n"
        "MATCH_CORR_WEIGHT = 0.45\n"
    ))
    _write(tmp_path, "CLAUDE.md", (
        "ok: `WINDOW_S` (120 s) and `ML_FLAG = False`, `MATCH_MARGIN` must stay at 0.05,\n"
        "(45% correlation / 55% MAE), `MATCH_MARGIN = 0.04 -> 0.05` history, `UNKNOWN_X` (9).\n"
        "bad: `MATCH_MARGIN` (0.06), **Frozen off** (`ML_FLAG`) fine, `ML_FLAG = True`,\n"
        "config schema v1->3.10.\n"
        "2. **Storage migration** - v1->15\n   (`STORAGE_VERSION` in `const.py`). Recent: v14->v15.\n"
        "## Next\n"
    ))
    report = dc.Report()
    dc.check_constants(tmp_path, report)
    text = "\n".join(report.failures)
    assert len(report.failures) == 5, text
    assert "MATCH_MARGIN stated as 0.06" in text
    assert "ML_FLAG stated as True" in text
    assert "v1->3.10" in text
    assert "stated as v1->15" in text
    assert "newest storage step listed is v15" in text


def test_register_ids_and_baseline(tmp_path: Path) -> None:
    _write(tmp_path, dc.REFERENCE, (
        "## 7. Register\n| # | Status | Kind | Short |\n|---|---|---|---|\n"
        "| 5 | FIXED | CODE | a |\n| 5 | OPEN | - | b |\n| 6 | FALSE POSITIVE | DOC | c |\n"
        "| 5-6 | FIXED | CODE | summary row |\n"
        "## 8. Index\n| 6 | FIXED | CODE | outside section 7 |\n"
    ))
    report = dc.Report()
    dc.check_register(tmp_path, report)
    assert report.register_counts == {"5": 2, "6": 1}

    dc.apply_baseline(report, {})
    assert any("register id 5 appears 2 times" in f for f in report.failures)

    report.failures.clear()
    dc.apply_baseline(report, dc.baseline_from(report))
    assert report.failures == []


def test_em_dash_ratchet_and_round_trip() -> None:
    dash = "\N{EM DASH}"
    assert dash.encode() == dc.EM_DASH
    report = dc.Report(em_dash={"a.md": 3, "b.py": 1, "c.md": 1})
    report.dead_anchors.append(("CLAUDE.md", "x.py:gone", 4, "gone is not defined"))
    dc.apply_baseline(report, {"em_dash": {"a.md": 3, "c.md": 2}})
    assert report.failures == [
        "CLAUDE.md:4: dead anchor `x.py:gone` - gone is not defined",
        "b.py: 1 em dash(es), baseline 0 - use ' - ' or '->' (U+2014 is banned repo-wide)",
    ]
    assert any("fell in 1 file" in n for n in report.notes)

    fresh = dc.Report(em_dash=dict(report.em_dash), dead_anchors=list(report.dead_anchors))
    dc.apply_baseline(fresh, dc.baseline_from(fresh))
    assert fresh.failures == []
