"""Regression tests for .github/workflows/issue_validator.yml (issues #399, #400).

Both reports were filed with the bug template's *text* but over the API rather than the
web issue form. GitHub applies an issue form's ``labels:`` only on the web path, so they
arrived with no labels at all - and the validator's ``bug`` gate skipped them outright.
Their headings were also ``##`` (hand-written) where the form renders ``###``, so every
field extractor missed, which would have produced a bogus "everything is missing" comment
had the job run.

Actions cannot be exercised locally, so the job scripts are extracted from the workflow
and executed by ``devtools/issue_validator_harness.mjs`` against a stubbed GitHub API.
The tests run the shipped script text - not a copy - so they fail if the workflow drifts.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
WORKFLOW = REPO / ".github" / "workflows" / "issue_validator.yml"
HARNESS = REPO / "devtools" / "issue_validator_harness.mjs"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is required to run the workflow scripts"
)


def _script(job: str) -> str:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return data["jobs"][job]["steps"][0]["with"]["script"]


def _run(scenarios: list[dict]) -> dict[str, dict]:
    spec = {
        "restoreScript": _script("restore-template-label"),
        "validateScript": _script("validate-bug-report"),
        "scenarios": scenarios,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(spec, fh)
        path = fh.name
    try:
        proc = subprocess.run(
            ["node", str(HARNESS), path],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    finally:
        Path(path).unlink(missing_ok=True)
    assert proc.returncode == 0, f"harness failed:\n{proc.stderr}"
    return {r["name"]: r for r in json.loads(proc.stdout)}


def _bug_body(heading: str, version: str = "0.5.4") -> str:
    """A complete bug report at the given heading level."""
    h = heading
    return "\n".join(
        [
            f"{h} Before You Submit",
            "",
            "- [x] I have searched the existing issues.",
            "",
            f"{h} Describe the Bug",
            "",
            "The anti-crease finalize splits a wash in two.",
            "",
            f"{h} Device Type",
            "",
            "Washing Machine",
            "",
            f"{h} Device Brand & Model",
            "",
            "Electrolux front-loader",
            "",
            f"{h} WashData Integration Version",
            "",
            version,
            "",
            f"{h} Home Assistant Version",
            "",
            "2026.8.1",
            "",
            f"{h} Reproduction Steps",
            "",
            "1. Run a program whose final spin sits in the last 2%.",
            "",
            f"{h} Expected Behavior",
            "",
            "A wash should be recorded as one cycle.",
            "",
            f"{h} Logs / Error Evidence",
            "",
            "```",
            "2026-08-21 20:48:07.210 INFO [cycle_detector] Anti-crease finalize: matched",
            "  'Delicate 30C' past expected 5966s (elapsed 5848s) - finalizing.",
            "```",
            "",
            f"{h} Additional Context",
            "",
            "Not a duplicate of #288.",
        ]
    )


def _scenario(name: str, body: str, labels: list[str] | None = None) -> dict:
    return {
        "name": name,
        "body": body,
        "labels": labels or [],
        "latestRelease": "v0.5.5",
    }


def test_api_filed_bug_report_gets_its_template_label_back():
    """#399/#400: a bug body posted over the API arrives unlabelled; restore it."""
    res = _run([_scenario("api-bug", _bug_body("##"))])["api-bug"]
    assert res["restore"]["addedLabels"] == ["bug"]


def test_api_filed_bug_report_is_actually_validated():
    """The validator must run for it, and flag only the outdated version."""
    res = _run([_scenario("api-bug", _bug_body("##"))])["api-bug"]
    assert len(res["validate"]["comments"]) == 1
    comment = res["validate"]["comments"][0]
    assert "Outdated version" in comment
    assert "`0.5.4`" in comment and "v0.5.5" in comment
    # The `##` headings must not read as missing fields (the second, latent failure).
    for field in (
        "**Device Brand & Model** is missing",
        "**Home Assistant Version** is missing",
        "**Logs / Error Evidence** are missing",
    ):
        assert field not in comment, f"{field!r} wrongly reported for a ## body"
    assert res["validate"]["addedLabels"] == ["more info required"]


@pytest.mark.parametrize("heading", ["##", "###", "####"])
def test_fields_are_read_at_any_heading_level(heading):
    """The web form renders `###`; a hand-written body may use any level."""
    res = _run([_scenario("h", _bug_body(heading, version="0.5.5"), ["bug"])])["h"]
    # Complete report on the current version: nothing to complain about.
    assert res["validate"]["comments"] == []
    assert res["validate"]["addedLabels"] == []


def test_web_form_report_with_blank_fields_still_flagged():
    """The pre-existing `###` path keeps working, placeholders and all."""
    body = "\n".join(
        [
            "### Describe the Bug",
            "",
            "It breaks.",
            "",
            "### Device Brand & Model",
            "",
            "_No response_",
            "",
            "### WashData Integration Version",
            "",
            "_No response_",
            "",
            "### Home Assistant Version",
            "",
            "_No response_",
            "",
            "### Logs / Error Evidence",
            "",
            "_No response_",
        ]
    )
    res = _run([_scenario("blank", body, ["bug"])])["blank"]
    comment = res["validate"]["comments"][0]
    for field in ("Device Brand & Model", "WashData Integration Version",
                  "Home Assistant Version", "Logs / Error Evidence"):
        assert field in comment
    assert res["validate"]["addedLabels"] == ["more info required"]


def test_feature_request_is_labelled_and_not_bug_validated():
    """A restored label must match the template the body actually used."""
    body = "\n".join(
        [
            "## Before You Submit",
            "",
            "- [x] yes",
            "",
            "## Problem or Motivation",
            "",
            "I want a thing.",
            "",
            "## Proposed Solution",
            "",
            "Add the thing.",
            "",
            "## Related Device Type",
            "",
            "Washing Machine",
            "",
            "## Alternatives Considered",
            "",
            "None.",
            "",
            "## Contributing an Implementation",
            "",
            "- [ ] no",
        ]
    )
    res = _run([_scenario("fr", body)])["fr"]
    assert res["restore"]["addedLabels"] == ["feature request"]
    assert res["validate"]["comments"] == []
    assert res["validate"]["addedLabels"] == []


def test_free_form_issue_is_never_labelled():
    """No template headings: leave it alone for close-templateless-issues to handle."""
    body = "# Summary\n\nIt is broken.\n\n# Impact\n\nBad.\n"
    res = _run([_scenario("free", body)])["free"]
    assert res["restore"]["addedLabels"] == []
    assert res["validate"]["comments"] == []


def test_a_single_stray_heading_does_not_mislabel():
    """One distinctive heading is below the two-hit threshold."""
    body = "# Notes\n\nSee also:\n\n## Proposed Solution\n\nmaybe do X.\n"
    res = _run([_scenario("stray", body)])["stray"]
    assert res["restore"]["addedLabels"] == []


def test_existing_template_label_is_left_alone():
    """Web-form issues already carry the label; the restore job must be a no-op."""
    res = _run([_scenario("kept", _bug_body("###", version="0.5.5"), ["bug"])])["kept"]
    assert res["restore"]["addedLabels"] == []


def test_job_triggers_are_scoped_correctly():
    """The YAML gates are not exercised by the harness, so pin them here.

    ``restore-template-label`` must stay opened/reopened-only (an ``edited`` run would
    re-add a label a maintainer removed on purpose), and ``validate-bug-report`` must
    keep ``always()`` or a skipped ``needs`` would skip it too.
    """
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    restore = data["jobs"]["restore-template-label"]["if"]
    assert "'opened'" in restore and "'reopened'" in restore
    assert "'edited'" not in restore

    validate = data["jobs"]["validate-bug-report"]
    assert validate["needs"] == "restore-template-label"
    assert "always()" in validate["if"]
    # The `bug` gate moved into the script; leaving it in the `if:` would re-break #399.
    assert "labels.*.name" not in validate["if"]
