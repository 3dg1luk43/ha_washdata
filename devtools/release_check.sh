#!/usr/bin/env bash
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Release preflight: everything that must be true before a tag is cut.
#
# Why this exists: the panel ships as a *generated* artifact. The readable
# `www/ha-washdata-panel.js` is the source of truth, `devtools/build_panel.mjs`
# minifies it, and `frontend.py` only serves the minified file when its recorded
# source hash still matches the bytes on disk. That design fails safe -- a
# forgotten rebuild means users download the larger readable file, never stale
# code -- but "fails safe" is not "correct": a release that ships a stale bundle
# silently loses the 32% size win for every user until someone notices.
#
# The same class of problem applies to the other generated/paired files: the WS
# type artifacts, the version in three places, and strings.json vs its English
# translation. This script checks them all in one place so a release cannot
# depend on anybody remembering.
#
#   devtools/release_check.sh                 verify only (safe; used by CI)
#   devtools/release_check.sh --fix           regenerate artifacts instead of failing
#   devtools/release_check.sh --tag v0.5.5    also require the tag to match the version
#   devtools/release_check.sh --full          add the slow suite and the E2E suite
#
# Exit code is the number of failed checks, so `if release_check.sh; then tag; fi`
# works. Every failure prints the exact command that fixes it.
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FIX=0
FULL=0
WANT_TAG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fix)  FIX=1; shift ;;
    --full) FULL=1; shift ;;
    --tag)
      if [[ $# -lt 2 ]]; then echo "--tag needs a value (e.g. --tag v0.5.5)" >&2; exit 2; fi
      WANT_TAG="$2"; shift 2 ;;
    -h|--help) sed -n '6,30p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

FAILED=0
PY="./.venv/bin/python"
[[ -x "$PY" ]] || PY="python3"

pass() { printf '  \033[32mok\033[0m    %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; [[ -n "${2:-}" ]] && printf '        -> %s\n' "$2"; FAILED=$((FAILED + 1)); }
skip() { printf '  \033[33mskip\033[0m  %s\n' "$1"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$1"; }

# ── 1. generated panel/card artifacts ────────────────────────────────────────
head_ "Build artifacts"
if ! command -v node >/dev/null 2>&1; then
  fail "node not available" "install Node so the panel bundle can be built/verified"
elif [[ $FIX -eq 1 ]]; then
  if node devtools/build_panel.mjs >/dev/null; then pass "panel + card rebuilt"
  else fail "panel build failed" "node devtools/build_panel.mjs"; fi
else
  if node devtools/build_panel.mjs --check >/dev/null 2>&1; then pass "panel + card artifacts current"
  else fail "minified artifacts are stale or missing" "node devtools/build_panel.mjs (then commit www/*.min.js + build-manifest.json)"; fi
fi

# ── 2. generated WS contract artifacts ───────────────────────────────────────
if [[ $FIX -eq 1 ]]; then
  if "$PY" devtools/generate_ws_types.py >/dev/null; then pass "WS types + docs regenerated"
  else fail "WS type generation failed" "$PY devtools/generate_ws_types.py"; fi
else
  if "$PY" devtools/generate_ws_types.py --check >/dev/null 2>&1; then pass "ws-types.d.ts + docs/WS_API.md current"
  else fail "WS type artifacts are out of date" "$PY devtools/generate_ws_types.py"; fi
fi

# ── 3. version agreement ─────────────────────────────────────────────────────
# manifest.json is what HACS and Home Assistant report; the CHANGELOG's top
# heading is what humans read; the tag is what GitHub publishes. All three
# disagreeing is the classic release slip, and only the first is machine-checked
# by anything else.
head_ "Version"
VERSION_REPORT=$(WANT_TAG="$WANT_TAG" "$PY" - <<'PY'
import json, os, re, sys
mf = json.load(open("custom_components/ha_washdata/manifest.json"))["version"]
m = re.search(r"^##\s+v?(\d[^\s]*)", open("CHANGELOG.md", encoding="utf-8").read(), re.M)
cl = m.group(1) if m else None
want = (os.environ.get("WANT_TAG") or "").lstrip("v") or None
print(f"manifest={mf}")
print(f"changelog={cl}")
bad = []
if cl is None:
    bad.append("no '## <version>' heading found in CHANGELOG.md")
elif mf != cl:
    bad.append(f"manifest.json says {mf} but the top CHANGELOG entry is {cl}")
if want and want != mf:
    bad.append(f"tag says {want} but manifest.json says {mf}")
for b in bad:
    print("PROBLEM:" + b)
sys.exit(1 if bad else 0)
PY
)
VERSION_RC=$?
MF_VER=$(sed -n 's/^manifest=//p' <<<"$VERSION_REPORT")
if [[ $VERSION_RC -eq 0 ]]; then
  pass "version agrees everywhere (${MF_VER})"
else
  while IFS= read -r line; do
    [[ "$line" == PROBLEM:* ]] && fail "${line#PROBLEM:}" "bump custom_components/ha_washdata/manifest.json and/or the CHANGELOG heading"
  done <<<"$VERSION_REPORT"
fi

# ── 4. translation invariants ────────────────────────────────────────────────
# strings.json and translations/en.json are required to be identical for the HA
# key namespace (see CLAUDE.md); every other language file must at least parse,
# because one malformed file breaks integration startup for that locale.
head_ "Translations"
if "$PY" - <<'PY'
import json, sys
from pathlib import Path
root = Path("custom_components/ha_washdata")
a = json.loads((root / "strings.json").read_text(encoding="utf-8"))
b = json.loads((root / "translations/en.json").read_text(encoding="utf-8"))
if a != b:
    print("strings.json and translations/en.json have diverged", file=sys.stderr)
    sys.exit(1)
PY
then pass "strings.json == translations/en.json"
else fail "strings.json and translations/en.json diverged" "keep them identical for the HA key namespace"; fi

BAD_JSON=$("$PY" - <<'PY'
import json
from pathlib import Path
bad = []
for p in sorted(Path("custom_components/ha_washdata").rglob("*.json")):
    try:
        json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        bad.append(f"{p}: {exc}")
print("\n".join(bad))
PY
)
if [[ -z "$BAD_JSON" ]]; then pass "every shipped .json parses"
else fail "malformed JSON in the shipped component" "$BAD_JSON"; fi

# Leading/trailing whitespace in a translation value is rejected by HA's own
# validator, which turns a cosmetic slip into a failed release.
WS_BAD=$("$PY" - <<'PY'
import json
from pathlib import Path
bad = []
def walk(node, path, origin):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, f"{path}.{k}" if path else k, origin)
    elif isinstance(node, str) and node != node.strip():
        bad.append(f"{origin}:{path}")
for p in sorted(Path("custom_components/ha_washdata").rglob("translations/**/*.json")):
    try:
        walk(json.loads(p.read_text(encoding="utf-8")), "", p)
    except Exception:
        pass  # parse errors are reported by the check above
print("\n".join(bad[:10]))
PY
)
if [[ -z "$WS_BAD" ]]; then pass "no edge whitespace in translation values"
else fail "translation values with leading/trailing whitespace" "$WS_BAD"; fi

# A translation that drops or renames a {placeholder} makes Home Assistant fail its own
# placeholder validation at startup for that locale -- and the error names the key, not
# the language, so it is miserable to track down. Compare every language against English.
PH_BAD=$(PH_LAYER=translations "$PY" - <<'PHPY'
import json, re
from pathlib import Path
PH = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")


def flat(node, prefix=""):
    out = {}
    for k, v in (node or {}).items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flat(v, key))
        else:
            out[key] = v
    return out


bad = []
import os
LAYER = os.environ["PH_LAYER"]
for d in (LAYER,):
    root = Path("custom_components/ha_washdata") / d
    en_file = root / "en.json"
    if not en_file.is_file():
        continue
    en = flat(json.loads(en_file.read_text(encoding="utf-8")))
    for p in sorted(root.glob("*.json")):
        if p.stem in ("en", ".translation-locks"):
            continue
        try:
            other = flat(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue  # malformed files are reported by the parse check above
        for key, val in other.items():
            src = en.get(key)
            if not isinstance(src, str) or not isinstance(val, str):
                continue
            want, got = set(PH.findall(src)), set(PH.findall(val))
            if want != got:
                missing = ", ".join(sorted(want - got)) or "-"
                extra = ", ".join(sorted(got - want)) or "-"
                bad.append(f"{p.name}:{key} missing[{missing}] unexpected[{extra}]")
print("\n".join(bad[:12]))
PHPY
)
if [[ -z "$PH_BAD" ]]; then pass "HA-layer translation placeholders match English"
else fail "HA-layer translations with mismatched {placeholders}" "$PH_BAD"; fi

# Panel layer: same comparison, but a mismatch there only renders a literal "{n}" or drops
# a value rather than failing startup, and 219 predate this check. Counted, not blocking.
PH_PANEL=$(PH_LAYER=translations/panel "$PY" -c '
import json, os, re
from pathlib import Path
PH = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")
def flat(node, prefix=""):
    out = {}
    for k, v in (node or {}).items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict): out.update(flat(v, key))
        else: out[key] = v
    return out
root = Path("custom_components/ha_washdata") / os.environ["PH_LAYER"]
en = flat(json.loads((root / "en.json").read_text(encoding="utf-8")))
n = 0
langs = set()
for p in sorted(root.glob("*.json")):
    if p.stem in ("en", ".translation-locks"): continue
    try: other = flat(json.loads(p.read_text(encoding="utf-8")))
    except Exception: continue
    for key, val in other.items():
        src = en.get(key)
        if isinstance(src, str) and isinstance(val, str) and set(PH.findall(src)) != set(PH.findall(val)):
            n += 1; langs.add(p.stem)
print(f"{n} {len(langs)}" if n else "")
')
if [[ -z "$PH_PANEL" ]]; then pass "panel translation placeholders match English"
else printf '  \033[33mwarn\033[0m  panel placeholder drift: %s value(s) across %s language(s)\n' $PH_PANEL
     printf '        -> cosmetic (renders a literal {n} or drops a value); pre-existing, needs a translation pass\n'
fi

# ── 5. code health ───────────────────────────────────────────────────────────
head_ "Code"
if "$PY" -m compileall -q custom_components/ha_washdata tests >/dev/null 2>&1; then pass "python compiles"
else fail "python syntax error" "$PY -m compileall custom_components/ha_washdata tests"; fi

if command -v node >/dev/null 2>&1; then
  if node --check custom_components/ha_washdata/www/ha-washdata-panel.js 2>/dev/null \
     && node --check custom_components/ha_washdata/www/ha-washdata-card.js 2>/dev/null; then
    pass "panel + card JS parse"
  else
    fail "panel or card JS does not parse" "node --check custom_components/ha_washdata/www/ha-washdata-panel.js"
  fi
  # Catches load-time TDZ and template ReferenceErrors that node --check cannot;
  # a backtick in panel CSS blanks the whole panel at runtime and only this sees it.
  if node devtools/panel_smoke.js >/dev/null 2>&1; then pass "panel render smoke"
  else fail "panel render smoke failed" "node devtools/panel_smoke.js"; fi
else
  skip "JS checks (node unavailable)"
fi

# ── 6. tests ─────────────────────────────────────────────────────────────────
# A suite failure prints its tail. Swallowing it entirely meant a CI log said only
# "fast suite failed" with no way to tell a real regression from a collection error -
# and a collection error takes the WHOLE suite with it, so "failed" can mean "nothing
# ran at all" (v0.5.5: a missing optional devtools import did exactly that).
run_suite() {
  local label="$1" out
  shift
  if out=$("$@" 2>&1); then
    pass "$label"
  else
    fail "$label failed" "$*"
    printf '%s\n' "$out" | tail -25 | sed 's/^/        | /'
  fi
}

head_ "Tests"
run_suite "fast suite" "$PY" -m pytest tests/ -q

if [[ $FULL -eq 1 ]]; then
  run_suite "slow suite" "$PY" -m pytest tests/ -q -m slow
  if command -v npx >/dev/null 2>&1; then
    if (cd playwright-tests && npx playwright test --reporter=dot >/dev/null 2>&1); then pass "E2E suite"
    else fail "E2E suite failed" "cd playwright-tests && npx playwright test"; fi
    # The bundle users actually receive must work, not just the readable source.
    if (cd playwright-tests && PANEL_BUILD=min npx playwright test --reporter=dot >/dev/null 2>&1); then pass "E2E against the minified bundle"
    else fail "E2E failed against the minified bundle" "cd playwright-tests && PANEL_BUILD=min npx playwright test"; fi
  else
    skip "E2E (npx unavailable)"
  fi
else
  skip "slow + E2E suites (pass --full to include)"
fi

# ── 7. release hygiene ───────────────────────────────────────────────────────
head_ "Release hygiene"
if git rev-parse --git-dir >/dev/null 2>&1; then
  DIRTY=$(git status --porcelain --untracked-files=no)
  if [[ -z "$DIRTY" ]]; then
    pass "working tree clean"
  else
    # A warning, not a failure: --fix legitimately dirties the tree, and this
    # script is meant to be usable mid-work.
    printf '  \033[33mwarn\033[0m  uncommitted changes to tracked files (%s file(s))\n' "$(wc -l <<<"$DIRTY")"
    printf '        -> commit the generated artifacts too: www/*.min.js, www/build-manifest.json\n'
  fi
  for art in custom_components/ha_washdata/www/ha-washdata-panel.min.js \
             custom_components/ha_washdata/www/ha-washdata-card.min.js \
             custom_components/ha_washdata/www/build-manifest.json; do
    if git ls-files --error-unmatch "$art" >/dev/null 2>&1; then
      pass "tracked: $(basename "$art")"
    else
      fail "$(basename "$art") is not tracked by git" "git add $art  (it is served to users, so it must ship)"
    fi
  done
else
  skip "git checks (not a repository)"
fi

printf '\n'
if [[ $FAILED -eq 0 ]]; then
  printf '\033[32mRelease preflight passed.\033[0m %s\n' "${MF_VER:+version $MF_VER}"
else
  printf '\033[31mRelease preflight failed: %d check(s).\033[0m\n' "$FAILED"
fi
exit "$FAILED"
