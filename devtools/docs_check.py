#!/usr/bin/env python3
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Docs consistency check (audit item F9).

Hand-maintained docs drift; this keeps the checkable part of them honest. Four
checks, all stdlib-only, deterministic and network-free (CI runs this before any
``pip install``):

1. **Symbol anchors.** Every ``file.py:symbol`` / ``file.py::symbol`` in CLAUDE.md,
   README.md, docs/internal/INTEGRATION_REFERENCE.md and register/OPEN.md must name
   a file under
   ``custom_components/ha_washdata/``, ``devtools/`` or ``tests/`` that defines
   ``symbol`` (AST: def/class, an assignment target, a ``self.x =`` attribute;
   ``Class.member`` is looked up inside the class). ``NAME_*`` and a trailing ``_``
   are prefixes. Line anchors (``file.py:123``) and anchors inside fenced code
   blocks (command examples) are not checked.
2. **Constant values.** In CLAUDE.md and README.md, `` `NAME` (value) ``,
   ``NAME = value``, `` `NAME` ... stays at value `` and ``**Frozen off** (`NAME`)``
   must agree with ``const.py``; the storage migration range ``v1->N`` and every
   ``vA->vB`` step must top out at ``STORAGE_VERSION``; the config entry
   ``schema v1->X.Y`` must equal ``CONFIG_ENTRY_VERSION.CONFIG_ENTRY_MINOR_VERSION``.
3. **Register ids.** Rows ``| <id> | <STATUS> | <KIND> |`` in
   docs/internal/register/OPEN.md and ARCHIVE.md must have unique ids across both
   files. Range rows (``135-140``) are summaries and are skipped. OPEN.md holds only
   OPEN/PARTIAL rows with a summary of at most 300 characters; ARCHIVE.md holds none
   (audit DOCS-03: the register had grown too big to read).
4. **Em dash ratchet.** U+2014 per git-tracked text file must not rise (repo rule:
   no em dashes anywhere). A file not in the baseline is allowed zero.
5. **Deep-dive identifiers.** In ``docs/internal/reference/*.md`` (outside fenced
   code blocks), every identifier inside an inline code span that is a call
   (``name(``), private (``_name``, ``self._name``) or an UPPER_SNAKE constant must
   occur as a word in ``custom_components/ha_washdata/**/*.py``,
   ``custom_components/ha_washdata/www/*.js`` (not the derived ``*.min.js``) or
   ``devtools/**/*.py``. ``NAME_*``, a trailing ``_`` and the head of a slash
   shorthand (``A_MIN/MAX_B``) are prefixes; a file name (``_x.py``), a glob or
   placeholder suffix (``*_model``, ``{x}_y``, ``<id>_z``) and the tail of a slash
   shorthand are not checked. History is how a deep-dive names removed code on
   purpose: a block (paragraph, list item, table row or heading) that says
   "removed in X.Y.Z" (or deleted / retired) AND cites a register item ("item 411")
   is exempt; such a heading exempts its whole section. Same contract as FIXED
   register rows: the register keeps the detail, the deep-dive keeps one pointer.

Known historical failures live in ``devtools/docs_check_baseline.json`` (dead anchors
that FIXED register rows name on purpose, the duplicate register ids, the em dash
counts, deep-dive identifiers not yet refreshed). Only NEW failures fail; an entry
that is no longer needed is reported as a note so the baseline can be tightened.

    python3 devtools/docs_check.py                    # report; exit 1 on any failure
    python3 devtools/docs_check.py --update-baseline  # rewrite the baseline from the tree

Exit codes: 0 clean, 1 at least one failure, 2 setup error (no git checkout,
unreadable baseline). With ``--update-baseline`` the exit code covers what the
baseline cannot absorb (constant mismatches).
"""
from __future__ import annotations

import argparse
import ast
import bisect
import json
import operator
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "devtools" / "docs_check_baseline.json"
COMPONENT = "custom_components/ha_washdata"
REFERENCE = "docs/internal/INTEGRATION_REFERENCE.md"
REGISTER_OPEN = "docs/internal/register/OPEN.md"
REGISTER_ARCHIVE = "docs/internal/register/ARCHIVE.md"
REGISTER_DOCS = (REGISTER_OPEN, REGISTER_ARCHIVE)
REGISTER_OPEN_STATUSES = frozenset({"OPEN", "PARTIAL"})
REGISTER_OPEN_SUMMARY_MAX = 300
# The archive is history: its FIXED rows name removed code on purpose, so it is
# not anchor-checked. OPEN.md is a work list and must point at live code.
ANCHOR_DOCS = ("CLAUDE.md", "README.md", REFERENCE, REGISTER_OPEN)
CONSTANT_DOCS = ("CLAUDE.md", "README.md")
ANCHOR_ROOTS = (COMPONENT, "devtools", "tests")
DEEP_DIVES = "docs/internal/reference"
# (directory, suffix, recursive): the files whose words a deep-dive identifier must be among.
IDENT_SOURCES = ((COMPONENT, ".py", True), (f"{COMPONENT}/www", ".js", False), ("devtools", ".py", True))
EM_DASH = "\N{EM DASH}".encode()  # spelled by name so this file holds none
TEXT_SUFFIXES = frozenset({
    ".md", ".py", ".js", ".mjs", ".cjs", ".ts", ".json", ".yaml", ".yml",
    ".html", ".css", ".sh", ".txt", ".toml", ".cfg", ".ini",
})

# `path/file.py:symbol`, `file.py::Class.member`, `const.py:STANDBY_BAND_*`.
ANCHOR_RE = re.compile(
    r"(?<![\w/.-])((?:[\w.-]+/)*[A-Za-z_][\w-]*\.py)(::?)"
    r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)(\*?)"
)
LINE_ANCHOR_RE = re.compile(r"L\d+")
FENCE_RE = re.compile(r"^\s*(```|~~~)")

_NUM = r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?"
_VALUE = rf"{_NUM}|True|False"
CONST_PATTERNS = (
    # `NAME` (0.12)   `NAME` (480 s, ...)
    re.compile(rf"`([A-Z][A-Z0-9_]{{2,}})`\s*\(\s*({_VALUE})\s*(?:s|W|Wh|%|x)?\s*[,);]"),
    # NAME = 0.12   `NAME=False`
    re.compile(rf"(?<![\w.])([A-Z][A-Z0-9_]{{2,}})\s*==?\s*({_VALUE})(?![\w.])"),
    # `NAME` itself must stay at 0.05
    re.compile(rf"`([A-Z][A-Z0-9_]{{2,}})`[^`.\n]{{0,30}}?\b(?:stays?|kept|pinned|fixed) at\s+({_NUM})(?![\w.])"),
)
FROZEN_OFF_RE = re.compile(r"\*\*Frozen off\*\*\s*\(`([A-Z][A-Z0-9_]{2,})`")
HISTORY_ARROW_RE = re.compile("\\s*(?:->|\N{RIGHTWARDS ARROW})")
STORAGE_RANGE_RE = re.compile(r"v1->(\d+)\s*\(\s*`STORAGE_VERSION`")
STORAGE_STEP_RE = re.compile(r"\bv(\d+)->v?(\d+)\b")
CONFIG_RANGE_RE = re.compile(r"schema v1->(\d+)\.(\d+)")

# | <id> | <STATUS> | <KIND> | ...   (KIND is occasionally "-")
# A register row: id, status, then the rest of the line.
REGISTER_ROW_FULL_RE = re.compile(r"^\|\s*(\d+[a-z]?(?:-\d+[a-z]?)?)\s*\|\s*([A-Z][^|\n]*)\|([^\n]*)$", re.M)

# Deep-dive identifiers (check 5).
CODE_SPAN_RE = re.compile(r"`([^`]+)`")
IDENT_RE = re.compile(r"(?<![\w$])[A-Za-z_]\w*")
UPPER_SNAKE_RE = re.compile(r"[A-Z][A-Z0-9]*_(?:[A-Z0-9]+_?)*")  # NAME_ (a prefix) included
CALL_AFTER_RE = re.compile(r"\s*\(")
FILE_AFTER_RE = re.compile(r"\.(?:py|js|mjs|cjs|ts|json|md|ya?ml|sh|txt|csv)\b")
HEADING_RE = re.compile(r"^(#{1,6})\s")
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s")
REMOVED_IN_RE = re.compile(r"\b(?:removed|deleted|retired)\s+in\s+v?\d+\.\d+(?:\.\d+)?\b", re.I)
REGISTER_CITE_RE = re.compile(r"\bitems?\s+\d+[a-z]?\b", re.I)

# Numbers CLAUDE.md states in prose, tied to the constant they come from:
# (pattern, ((CONST or "1-CONST", scale), ...)) - one entry per capture group,
# doc value == scale * const value. A pattern that no longer matches is skipped.
PROSE_CLAIMS = (
    (re.compile(rf"`\[min_duration_ratio, max_duration_ratio\]`\s*\(\s*({_NUM})x-({_NUM})x"),
     (("DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO", 1), ("DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO", 1))),
    (re.compile(rf"\(({_NUM})% correlation / ({_NUM})% MAE\)"),
     (("MATCH_CORR_WEIGHT", 100), ("1-MATCH_CORR_WEIGHT", 100))),
    (re.compile(r"profile matching every (\d+) min"), (("DEFAULT_PROFILE_MATCH_INTERVAL", 1 / 60),)),
)


FIX_HINT = (
    "-> fix the doc (or the code). Only history may be accepted instead: a FIXED register row "
    "naming a since-removed symbol goes in the baseline via --update-baseline; a deep-dive names "
    "removed code in a block that says 'removed in X.Y.Z' and cites the register item."
)


@dataclass
class Report:
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)
    # Raw findings, before the baseline is applied (what --update-baseline writes).
    dead_anchors: list[tuple[str, str, int, str]] = field(default_factory=list)  # doc, anchor, line, why
    register_counts: Counter = field(default_factory=Counter)
    em_dash: dict[str, int] = field(default_factory=dict)
    dead_identifiers: list[tuple[str, str, int]] = field(default_factory=list)  # doc, name, line


# ── helpers ──────────────────────────────────────────────────────────────────


def _strip_fences(text: str) -> str:
    """Blank fenced code blocks, keeping line numbers stable."""
    out, fenced = [], False
    for line in text.split("\n"):
        if FENCE_RE.match(line):
            fenced = not fenced
            out.append("")
        else:
            out.append("" if fenced else line)
    return "\n".join(out)


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _git_ls_files(root: Path) -> list[str]:
    res = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True
    )
    return [p for p in res.stdout.decode("utf-8", "surrogateescape").split("\0") if p]


class _PyIndex:
    """Python files under the anchor roots, and whether one defines a name."""

    def __init__(self, root: Path) -> None:
        self.by_name: dict[str, list[Path]] = defaultdict(list)
        for base in ANCHOR_ROOTS:
            for p in (root / base).rglob("*.py"):
                if "__pycache__" not in p.parts and "node_modules" not in p.parts:
                    self.by_name[p.name].append(p)
        self._src: dict[Path, str] = {}
        self._ast: dict[Path, tuple[set[str], dict[str, set[str]]]] = {}

    def resolve(self, ref: str) -> list[Path]:
        ref = ref.removeprefix("./")
        cands = self.by_name.get(ref.rsplit("/", 1)[-1], [])
        if "/" in ref:
            cands = [p for p in cands if p.as_posix().endswith("/" + ref)]
        return cands

    def defines(self, path: Path, symbol: str, *, prefix: bool) -> bool:
        head, *rest = symbol.split(".")
        if path not in self._src:
            self._src[path] = path.read_text(encoding="utf-8")
        src = self._src[path]
        if head not in src:
            return False
        if not rest and _defined_on_a_line(src, head, prefix):
            return True
        if path not in self._ast:
            self._ast[path] = _ast_definitions(src)
        names, classes = self._ast[path]
        if not rest:
            return any(n.startswith(head) for n in names) if prefix else head in names
        members = classes.get(head, set())
        return any(n.startswith(rest[0]) for n in members) if prefix else rest[0] in members


# One source line that defines ``name``: a def/class, or a plain/annotated/self. assignment.
# The assignment needs whitespace before ``=`` so a keyword argument on its own line
# (``    name=value,``) is not mistaken for one; an unspaced assignment falls to the AST.
_DEF_LINE = (
    r"[ \t]*(?:(?:async[ \t]+)?def|class)[ \t]+{name}"
    r"|[ \t]*(?:self\.|cls\.)?{name}(?:[ \t]*:[^=]*)?[ \t]+=(?!=)"
)


def _defined_on_a_line(src: str, name: str, prefix: bool) -> bool:
    """Fast path: test only the lines that contain ``name`` (str.find, not a regex scan).

    Parsing every anchored module costs ~1 s and a multiline regex over them about
    the same, so the AST is kept as the fallback for what a line cannot show
    (tuple targets, loop variables, class membership).
    """
    line_re = re.compile(_DEF_LINE.format(name=re.escape(name) + (r"\w*" if prefix else r"\b")))
    i = src.find(name)
    while i >= 0:
        start = src.rfind("\n", 0, i) + 1
        end = src.find("\n", i)
        end = len(src) if end < 0 else end
        if line_re.match(src, start, end):
            return True
        i = src.find(name, end)
    return False


def _target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"):
        return [node.attr]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [n for elt in node.elts for n in _target_names(elt)]
    if isinstance(node, ast.Starred):
        return _target_names(node.value)
    return []


def _scan(stmts: list, sinks: list[set[str]], classes: dict[str, set[str]]) -> None:
    """Statement-level walk (expressions are skipped: ~10x fewer nodes than ast.walk)."""
    for node in stmts:
        found: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.append(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                found += _target_names(t)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor)):
            found += _target_names(node.target)
        for sink in sinks:
            sink.update(found)
        inner = sinks
        if isinstance(node, ast.ClassDef):
            inner = [*sinks, classes.setdefault(node.name, set())]
        for attr in ("body", "orelse", "finalbody", "handlers", "cases"):
            child = getattr(node, attr, None)
            if isinstance(child, list):
                _scan(child, inner, classes)


def _ast_definitions(source: str) -> tuple[set[str], dict[str, set[str]]]:
    """Every name a module binds at any depth, and per class what its body binds."""
    names: set[str] = set()
    classes: dict[str, set[str]] = {}
    try:
        _scan(ast.parse(source).body, [names], classes)
    except SyntaxError:
        # A file mid-edit still has its def/class lines.
        names = set(re.findall(r"^\s*(?:async\s+)?(?:def|class)\s+(\w+)", source, re.M))
    return names, classes


_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def _const_values(path: Path) -> dict[str, object]:
    """Module-level scalar constants of const.py, evaluated without importing it."""
    values: dict[str, object] = {}

    def ev(node: ast.AST) -> object:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else +v
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        raise ValueError

    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Name):
                try:
                    values[t.id] = ev(value)
                except Exception:  # noqa: BLE001 - non-scalar or not evaluable: not checkable
                    values.pop(t.id, None)
    return values


def _same(doc_value: str, actual: object) -> bool:
    if doc_value in ("True", "False"):
        return isinstance(actual, bool) and actual == (doc_value == "True")
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return False
    return abs(float(doc_value) - float(actual)) <= 1e-9 * max(1.0, abs(float(actual)))


# ── checks ───────────────────────────────────────────────────────────────────


def check_anchors(root: Path, report: Report) -> None:
    index = _PyIndex(root)
    checked = 0
    for doc in ANCHOR_DOCS:
        path = root / doc
        if not path.is_file():
            continue
        text = _strip_fences(path.read_text(encoding="utf-8"))
        for m in ANCHOR_RE.finditer(text):
            ref, symbol, star = m.group(1), m.group(3), m.group(4)
            if LINE_ANCHOR_RE.fullmatch(symbol):
                continue
            checked += 1
            anchor = f"{ref}{m.group(2)}{symbol}{star}"
            files = index.resolve(ref)
            if not files:
                problem = f"no {ref} under {', '.join(ANCHOR_ROOTS)}"
            elif any(index.defines(f, symbol, prefix=bool(star) or symbol.endswith("_")) for f in files):
                continue
            else:
                problem = f"{symbol} is not defined in " + ", ".join(
                    p.relative_to(root).as_posix() for p in files
                )
            report.dead_anchors.append((doc, anchor, _line_of(text, m.start()), problem))
    report.stats["anchors checked"] = checked


def check_constants(root: Path, report: Report) -> None:
    values = _const_values(root / COMPONENT / "const.py")
    checked = 0
    for doc in CONSTANT_DOCS:
        path = root / doc
        if not path.is_file():
            continue
        text = _strip_fences(path.read_text(encoding="utf-8"))
        seen: set[tuple[str, int]] = set()
        claims: list[tuple[str, str, int]] = []
        for pattern in CONST_PATTERNS:
            for m in pattern.finditer(text):
                if HISTORY_ARROW_RE.match(text, m.end()):
                    continue  # "NAME 0.3 -> 0.5" is history, not a claim
                claims.append((m.group(1), m.group(2), m.start(1)))
        claims += [(m.group(1), "False", m.start(1)) for m in FROZEN_OFF_RE.finditer(text)]
        checked += _check_prose(doc, text, values, report)
        for name, value, pos in claims:
            if name not in values or (name, pos) in seen:
                continue
            seen.add((name, pos))
            checked += 1
            if not _same(value, values[name]):
                report.failures.append(
                    f"{doc}:{_line_of(text, pos)}: {name} stated as {value}, const.py has {values[name]!r}"
                )
        checked += _check_versions(doc, text, values, report)
    report.stats["constant claims checked"] = checked


def _check_prose(doc: str, text: str, values: dict[str, object], report: Report) -> int:
    checked = 0
    for pattern, consts in PROSE_CLAIMS:
        for m in pattern.finditer(text):
            for group, (name, scale) in enumerate(consts, start=1):
                base = values.get(name.removeprefix("1-"))
                if isinstance(base, bool) or not isinstance(base, (int, float)):
                    continue
                expected = scale * ((1 - base) if name.startswith("1-") else base)
                checked += 1
                if not _same(m.group(group), expected):
                    report.failures.append(
                        f"{doc}:{_line_of(text, m.start(group))}: '{m.group(0)}' states {m.group(group)}, "
                        f"const.py gives {round(expected, 6)} ({name} x {scale:g})"
                    )
    return checked


def _check_versions(doc: str, text: str, values: dict[str, object], report: Report) -> int:
    checked = 0
    storage = values.get("STORAGE_VERSION")
    for m in STORAGE_RANGE_RE.finditer(text):
        checked += 1
        if int(m.group(1)) != storage:
            report.failures.append(
                f"{doc}:{_line_of(text, m.start())}: storage migration stated as v1->{m.group(1)}, "
                f"const.py STORAGE_VERSION is {storage}"
            )
    # The storage-migration paragraph lists the recent steps; its newest must be current.
    start = text.find("**Storage migration**")
    if start >= 0:
        end = text.find("\n#", start)
        end = end if end > 0 else len(text)
        steps = [(int(m.group(2)), m.start()) for m in STORAGE_STEP_RE.finditer(text, start, end)]
        if steps:
            checked += 1
            newest, pos = max(steps)
            if newest != storage:
                report.failures.append(
                    f"{doc}:{_line_of(text, pos)}: newest storage step listed is v{newest}, "
                    f"const.py STORAGE_VERSION is {storage}"
                )
    want = f"{values.get('CONFIG_ENTRY_VERSION')}.{values.get('CONFIG_ENTRY_MINOR_VERSION')}"
    for m in CONFIG_RANGE_RE.finditer(text):
        checked += 1
        if f"{m.group(1)}.{m.group(2)}" != want:
            report.failures.append(
                f"{doc}:{_line_of(text, m.start())}: config entry schema stated as v1->{m.group(1)}.{m.group(2)}, "
                f"const.py CONFIG_ENTRY_VERSION.CONFIG_ENTRY_MINOR_VERSION is {want}"
            )
    return checked


def _register_table(text: str) -> str:
    """The register table: everything before the first ``## `` heading after it.

    OPEN.md's per-item detail and ARCHIVE.md's pre-split prose follow the table
    under their own headings and may quote rows of other tables.
    """
    end = re.search(r"^## ", text, re.M)
    return text[: end.start()] if end else text


def check_register(root: Path, report: Report) -> None:
    ids: list[str] = []
    for rel in REGISTER_DOCS:
        path = root / rel
        if not path.is_file():
            report.failures.append(f"{rel}: missing (the register lives in register/OPEN.md + ARCHIVE.md)")
            continue
        table = _register_table(path.read_text(encoding="utf-8"))
        rows = [m for m in REGISTER_ROW_FULL_RE.finditer(table) if "-" not in m.group(1)]
        if not rows:
            report.failures.append(f"{rel}: no register rows found (format changed?)")
        ids += [m.group(1) for m in rows]
        for m in rows:
            status, rest = m.group(2).strip(), m.group(3)
            line = _line_of(table, m.start())
            if rel == REGISTER_OPEN:
                if status not in REGISTER_OPEN_STATUSES:
                    report.failures.append(
                        f"{rel}:{line}: item {m.group(1)} is {status}; closed items belong in ARCHIVE.md"
                    )
                summary = rest.rsplit("|", 2)[-2].strip() if rest.count("|") >= 2 else rest.strip()
                if len(summary) > REGISTER_OPEN_SUMMARY_MAX:
                    report.failures.append(
                        f"{rel}:{line}: item {m.group(1)} summary is {len(summary)} chars "
                        f"(max {REGISTER_OPEN_SUMMARY_MAX}); put the detail under ## Detail"
                    )
            elif status in REGISTER_OPEN_STATUSES:
                report.failures.append(
                    f"{rel}:{line}: item {m.group(1)} is {status}; open items belong in OPEN.md"
                )
    report.register_counts = Counter(ids)
    report.stats["register ids"] = len(ids)


def check_em_dash(root: Path, report: Report) -> None:
    counts: dict[str, int] = {}
    for rel in _git_ls_files(root):
        if Path(rel).suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            n = (root / rel).read_bytes().count(EM_DASH)
        except OSError:
            continue  # deleted in the working tree
        if n:
            counts[rel] = n
    report.em_dash = counts
    report.stats["em dashes"] = sum(counts.values())


_WORD_BYTES = frozenset(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
# bytes.translate + split tokenizes ~4 MB of source in a tenth of the time a regex takes.
_NON_WORD_TO_SPACE = bytes(c if c in _WORD_BYTES else 0x20 for c in range(256))


def _code_words(root: Path) -> set[str]:
    """Every identifier-shaped word in IDENT_SOURCES (comments and strings included)."""
    words: set[bytes] = set()
    for base, suffix, recursive in IDENT_SOURCES:
        for dirpath, dirnames, filenames in os.walk(root / base):
            dirnames[:] = [] if not recursive else [
                d for d in dirnames if d not in ("node_modules", "__pycache__") and not d.startswith(".")
            ]
            for name in filenames:
                if name.endswith(suffix) and not name.endswith(".min.js"):
                    data = (Path(dirpath) / name).read_bytes()
                    words.update(data.translate(_NON_WORD_TO_SPACE).split())
    return {w.decode("ascii") for w in words}


def _blocks(lines: list[str]) -> list[tuple[list[int], bool]]:
    """Paragraphs, list items, table rows and headings as line indexes, each with its history flag.

    A block is history when it says "removed in X.Y.Z" and cites a register item; a
    history heading makes everything up to the next heading of its level history too.
    """
    def is_history(text: str) -> bool:
        return bool(REMOVED_IN_RE.search(text) and REGISTER_CITE_RE.search(text))

    blocks: list[tuple[list[int], bool]] = []
    block: list[int] = []
    section: int | None = None  # level of the heading that opened a history section

    def flush() -> None:
        if block:
            text = " ".join(lines[i] for i in block)
            blocks.append((list(block), section is not None or is_history(text)))
            block.clear()

    for i, line in enumerate(lines):
        heading = HEADING_RE.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            if section is not None and level <= section:
                section = None
            if section is None and is_history(line):
                section = level
            blocks.append(([i], section is not None))
            continue
        if not line.strip():
            flush()
            continue
        row = line.lstrip().startswith("|")
        if row or LIST_ITEM_RE.match(line):
            flush()
        block.append(i)
        if row:
            flush()
    flush()
    return blocks


def _span_identifiers(span: str) -> list[tuple[str, bool]]:
    """(name, is_prefix) for each call, private name or UPPER_SNAKE constant in a code span."""
    out: list[tuple[str, bool]] = []
    for m in IDENT_RE.finditer(span):
        name, start, end = m.group(), m.start(), m.end()
        before = span[start - 1] if start else ""
        after = span[end: end + 1]
        if before in ("*", "}", ">") or (before == "/" and start > 1 and _is_word(span[start - 2])):
            continue  # glob / placeholder suffix, or the tail of a slash shorthand
        if FILE_AFTER_RE.match(span, end):
            continue  # a file name such as _parity.json
        call = bool(CALL_AFTER_RE.match(span, end))
        if not (call or name.startswith("_") or UPPER_SNAKE_RE.fullmatch(name)):
            continue
        prefix = after == "*" or name.endswith("_") or (after == "/" and _is_word(span[end + 1: end + 2]))
        out.append((name, prefix))
    return out


def _is_word(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def check_deep_dive_identifiers(root: Path, report: Report) -> None:
    docs = sorted((root / DEEP_DIVES).glob("*.md"))
    words = _code_words(root) if docs else set()
    ordered = sorted(words)
    checked = 0
    for path in docs:
        doc = path.relative_to(root).as_posix()
        lines = _strip_fences(path.read_text(encoding="utf-8")).split("\n")
        for idxs, history in _blocks(lines):
            text = "\n".join(lines[i] for i in idxs)
            if history or "`" not in text:
                continue
            for m in CODE_SPAN_RE.finditer(text):  # a span may wrap onto the next line
                line = idxs[0] + text.count("\n", 0, m.start()) + 1
                for name, prefix in _span_identifiers(m.group(1).replace("\n", " ")):
                    checked += 1
                    if prefix:
                        j = bisect.bisect_left(ordered, name)
                        if j < len(ordered) and ordered[j].startswith(name):
                            continue
                    elif name in words:
                        continue
                    report.dead_identifiers.append((doc, name, line))
    report.stats["deep-dive identifiers checked"] = checked


# ── baseline ─────────────────────────────────────────────────────────────────


def _id_key(i: str) -> tuple[int, str]:
    m = re.match(r"(\d+)(.*)", i)
    return (int(m.group(1)), m.group(2)) if m else (0, i)


def baseline_from(report: Report) -> dict:
    return {
        "_comment": (
            "Ratchet for devtools/docs_check.py: known failures that predate the check. "
            "Regenerate with `python3 devtools/docs_check.py --update-baseline` and review "
            "the diff - an entry should only ever shrink or disappear."
        ),
        "dead_anchors": {doc: sorted(anchors) for doc, anchors in sorted(_anchors_by_doc(report).items())},
        "register_duplicates": {
            i: n for i, n in sorted(report.register_counts.items(), key=lambda kv: _id_key(kv[0])) if n > 1
        },
        "em_dash": dict(sorted(report.em_dash.items())),
        "deep_dive_identifiers": {
            doc: sorted(names) for doc, names in sorted(_identifiers_by_doc(report).items())
        },
    }


def _identifiers_by_doc(report: Report) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for doc, name, _line in report.dead_identifiers:
        out[doc].add(name)
    return out


def _anchors_by_doc(report: Report) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for doc, anchor, _line, _why in report.dead_anchors:
        out[doc].add(anchor)
    return out


def apply_baseline(report: Report, baseline: dict) -> None:
    known_anchors = {doc: set(v) for doc, v in baseline.get("dead_anchors", {}).items()}
    for doc, anchor, line, problem in report.dead_anchors:
        if anchor not in known_anchors.get(doc, set()):
            report.failures.append(f"{doc}:{line}: dead anchor `{anchor}` - {problem}")
    found = _anchors_by_doc(report)
    for doc, anchors in known_anchors.items():
        for anchor in sorted(anchors - found.get(doc, set())):
            report.notes.append(f"baseline dead anchor {doc}: `{anchor}` now resolves or is gone")

    known_dups = baseline.get("register_duplicates", {})
    for i, n in sorted(report.register_counts.items(), key=lambda kv: _id_key(kv[0])):
        allowed = known_dups.get(i, 1)
        if n > allowed:
            report.failures.append(
                f"register: id {i} appears {n} times across OPEN.md + ARCHIVE.md (baseline {allowed}); give the new row a fresh id"
            )
        elif n < allowed:
            report.notes.append(f"register id {i} now appears {n} times (baseline {allowed})")
    for i in sorted(set(known_dups) - set(report.register_counts), key=_id_key):
        report.notes.append(f"register id {i} no longer present (baseline {known_dups[i]})")

    known_em = baseline.get("em_dash", {})
    for rel, n in sorted(report.em_dash.items()):
        allowed = known_em.get(rel, 0)
        if n > allowed:
            report.failures.append(
                f"{rel}: {n} em dash(es), baseline {allowed} - use ' - ' or '->' (U+2014 is banned repo-wide)"
            )
    shrunk = sum(1 for rel, n in known_em.items() if report.em_dash.get(rel, 0) < n)
    if shrunk:
        report.notes.append(f"em dash count fell in {shrunk} file(s); --update-baseline locks that in")

    known_ids = {doc: set(v) for doc, v in baseline.get("deep_dive_identifiers", {}).items()}
    for doc, name, line in report.dead_identifiers:
        if name not in known_ids.get(doc, set()):
            report.failures.append(
                f"{doc}:{line}: `{name}` is not in the code - describe what the code does now, "
                "or make the block history ('removed in X.Y.Z' + 'item N')"
            )
    found_ids = _identifiers_by_doc(report)
    for doc, names in sorted(known_ids.items()):
        for name in sorted(names - found_ids.get(doc, set())):
            report.notes.append(f"baseline deep-dive identifier {doc}: `{name}` now resolves or is gone")


def load_baseline(path: Path = BASELINE_PATH) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def collect(root: Path = ROOT) -> Report:
    report = Report()
    check_anchors(root, report)
    check_constants(root, report)
    check_register(root, report)
    check_em_dash(root, report)
    check_deep_dive_identifiers(root, report)
    return report


def run(root: Path = ROOT, baseline: dict | None = None) -> Report:
    """Run every check and apply the baseline. ``report.failures`` empty means clean."""
    report = collect(root)
    apply_baseline(report, load_baseline(root / BASELINE_PATH.relative_to(ROOT)) if baseline is None else baseline)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--update-baseline", action="store_true", help="rewrite the baseline from the current tree")
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    try:
        if args.update_baseline:
            report = collect(ROOT)
            BASELINE_PATH.write_text(json.dumps(baseline_from(report), indent=2) + "\n", encoding="utf-8")
            print(f"wrote {BASELINE_PATH.relative_to(ROOT)}")
            apply_baseline(report, load_baseline())
        else:
            report = run(ROOT)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"docs_check: setup error: {exc}", file=sys.stderr)
        return 2
    for line in report.failures:
        print(f"FAIL  {line}")
    if report.failures:
        print(FIX_HINT)
    for line in report.notes:
        print(f"note  {line}")
    stats = ", ".join(f"{k} {v}" for k, v in report.stats.items())
    verdict = f"{len(report.failures)} failure(s)" if report.failures else "ok"
    print(f"docs_check: {verdict} ({stats}; {time.perf_counter() - t0:.2f}s)")
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
