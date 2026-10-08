"""devtools/docs_check.py: the docs consistency gate (audit F9).

The first test is the gate itself: the real CLAUDE.md / README.md / register / deep-dives
against the real code and the committed baseline. The rest pin each check's behaviour on a tiny
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
    # Most anchors lived in the register; since the DOCS-03 split its archive is
    # history and is not anchor-checked, so only a handful remain.
    assert report.stats["anchors checked"] >= 1
    assert report.stats["constant claims checked"] >= 10
    # docs/internal (register, deep-dives) is local-only and absent from a CI checkout.
    if (dc.ROOT / "docs/internal").is_dir():
        assert report.stats["register ids"] > 300
        assert report.stats["deep-dive identifiers checked"] > 1000


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
    _write(tmp_path, dc.REGISTER_OPEN, (
        "# Open\n| # | Status | Kind | Owner | Summary |\n|---|---|---|---|---|\n"
        "| 5 | OPEN | CODE | detection | b |\n"
        "## Detail\n| 6 | FIXED | CODE | a quoted row, not the table |\n"
    ))
    _write(tmp_path, dc.REGISTER_ARCHIVE, (
        "# Archive\n| # | Status | Kind | Short |\n|---|---|---|---|\n"
        "| 5 | FIXED | CODE | a |\n| 6 | FALSE POSITIVE | DOC | c |\n"
        "| 5-6 | FIXED | CODE | summary row |\n"
        "## Dated notes\n| 6 | FIXED | CODE | outside the table |\n"
    ))
    report = dc.Report()
    dc.check_register(tmp_path, report)
    assert report.register_counts == {"5": 2, "6": 1}
    assert report.failures == []

    dc.apply_baseline(report, {})
    assert any("id 5 appears 2 times" in f for f in report.failures)

    report.failures.clear()
    dc.apply_baseline(report, dc.baseline_from(report))
    assert report.failures == []


def test_register_open_and_archive_hold_the_right_rows(tmp_path: Path) -> None:
    """Audit DOCS-03: OPEN.md is a short work list, ARCHIVE.md holds no open item."""
    _write(tmp_path, dc.REGISTER_OPEN, (
        "| # | Status | Kind | Owner | Summary |\n|---|---|---|---|---|\n"
        f"| 7 | FIXED | CODE | x | done |\n| 8 | OPEN | CODE | x | {'y' * 301} |\n"
        "| 9 | PARTIAL | NOTE | x | fine |\n"
    ))
    _write(tmp_path, dc.REGISTER_ARCHIVE, "| 10 | OPEN | CODE | still open |\n| 11 | FIXED | CODE | ok |\n")
    report = dc.Report()
    dc.check_register(tmp_path, report)
    text = "\n".join(report.failures)
    assert len(report.failures) == 3, text
    assert "item 7 is FIXED; closed items belong in ARCHIVE.md" in text
    assert "item 8 summary is 301 chars" in text
    assert "item 10 is OPEN; open items belong in OPEN.md" in text


def test_an_empty_open_list_is_fine_but_a_missing_table_is_not(tmp_path: Path) -> None:
    _write(tmp_path, dc.REGISTER_ARCHIVE, "| 1 | FIXED | CODE | done |\n")
    _write(tmp_path, dc.REGISTER_OPEN, "| # | Status | Kind | Owner | Summary |\n|---|---|---|---|---|\n\n## Detail\n")
    report = dc.Report()
    dc.check_register(tmp_path, report)
    assert report.failures == []
    _write(tmp_path, dc.REGISTER_OPEN, "# Open\nnothing here\n")
    report = dc.Report()
    dc.check_register(tmp_path, report)
    assert any("OPEN.md: no register rows found" in f for f in report.failures)


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


def test_deep_dive_identifier_check(tmp_path: Path) -> None:
    _write(tmp_path, f"{dc.COMPONENT}/mod.py", (
        "LIMIT_LOW = 1\n"
        "def alive():\n    pass\n"
        "class Box:\n    def lid(self):\n        self._hinge = 21_600\n"
        "# _comment_only is named in a comment\n"
    ))
    _write(tmp_path, f"{dc.COMPONENT}/ml/engine.py", "def resolve_scorer():\n    pass\n")
    _write(tmp_path, f"{dc.COMPONENT}/www/panel.js", "function _jsHelper() { return PANEL_KEY_X; }\n")
    _write(tmp_path, f"{dc.COMPONENT}/www/panel.min.js", "MIN_ONLY_NAME\n")  # derived build: not a source
    _write(tmp_path, "devtools/eval.py", "def _tool_fn():\n    pass\n")
    _write(tmp_path, "devtools/node_modules/pkg/x.py", "NODE_ONLY_NAME = 1\n")
    _write(tmp_path, "tests/test_x.py", "TEST_ONLY_NAME = 1\n")  # tests are not a source either
    _write(tmp_path, f"{dc.DEEP_DIVES}/01-x.md", (
        "# Title\n"                                                                  # 1
        "Alive: `alive()`, `Box.lid()`, `self._hinge`, `LIMIT_LOW`, `LIMIT_*`, `resolve_scorer(x)`,\n"
        "`_jsHelper()`, `PANEL_KEY_X`, `_tool_fn`, `_comment_only`, `LIMIT_LOW/HIGH_X`.\n"
        "Unchecked: `plain_name`, `CamelCase`, `_parity.json`, `*_model.py`, `test_{a,b}_cols`,\n"
        "`<id>_state`, `alive/_tail_dead`, `21_600`, `mod.py:alive`.\n"     # 5
        "Dead: `gone()`, `_dead`, `DEAD_CONST`, `NOPE_*`, `MIN_ONLY_NAME`, `NODE_ONLY_NAME`,\n"
        "`TEST_ONLY_NAME`, `obj.method_gone(1)`, `wrapped_dead(a,\nb)`.\n"   # 7-8
        "```\nfenced_dead()\n```\n"                                         # 9-11
        "\n`hist_dead()` was removed in 0.5.8 (register item 411),\n"       # 12-13
        "and `hist_wrapped_dead` with it.\n"                                 # 14
        "\n`no_cite_dead()` was removed in 0.5.8.\n"                         # 15-16
        "\n- `item_hist_dead()` removed in 0.5.2 (item 27).\n"               # 17-18
        "- `item_live_dead()` is still described.\n"                         # 19
        "\n| `_row_hist_dead` | removed in 0.5.8, item 418 |\n"               # 20-21
        "| `_row_live_dead` | current |\n"                                    # 22
        "\n## Old stack (removed in 0.5.8, item 411)\n"                      # 23-24
        "`section_dead()` and\n\n### Detail\n`subsection_dead()`.\n"        # 25-28
        "\n## Current\n`after_section_dead()`\n"                             # 29-31
    ))
    report = dc.Report()
    dc.check_deep_dive_identifiers(tmp_path, report)
    dead = {(name, line) for _doc, name, line in report.dead_identifiers}
    assert dead == {
        ("gone", 6), ("_dead", 6), ("DEAD_CONST", 6), ("NOPE_", 6), ("MIN_ONLY_NAME", 6),
        ("NODE_ONLY_NAME", 6), ("TEST_ONLY_NAME", 7), ("method_gone", 7), ("wrapped_dead", 7),
        ("no_cite_dead", 16), ("item_live_dead", 19), ("_row_live_dead", 22),
        ("after_section_dead", 31),
    }
    assert {doc for doc, _name, _line in report.dead_identifiers} == {f"{dc.DEEP_DIVES}/01-x.md"}

    dc.apply_baseline(report, {})
    assert len(report.failures) == len(report.dead_identifiers)
    assert "01-x.md:6: `gone` is not in the code" in report.failures[0]

    # The ratchet: a baselined name passes anywhere in its doc; one that resolves is a note.
    baseline = dc.baseline_from(report)
    assert baseline["deep_dive_identifiers"][f"{dc.DEEP_DIVES}/01-x.md"][:2] == ["DEAD_CONST", "MIN_ONLY_NAME"]
    baseline["deep_dive_identifiers"][f"{dc.DEEP_DIVES}/01-x.md"].append("since_fixed")
    report.failures.clear()
    dc.apply_baseline(report, baseline)
    assert report.failures == []
    assert any("`since_fixed` now resolves or is gone" in n for n in report.notes)
