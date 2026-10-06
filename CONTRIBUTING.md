# Contributing to WashData

Thanks for helping. Please read the [Code of Conduct](CODE_OF_CONDUCT.md) first.

## Contributor PR flow (non-translation PRs)

**Agree the work before you write code.** WashData does not accept unsolicited pull requests. The maintainer is a volunteer, often with
unreleased work in progress, so a surprise PR easily duplicates or conflicts with it.

1. **Open an issue** with a template: [Bug report](https://github.com/3dg1luk43/ha_washdata/issues/new?template=bug_report.yml),
   [Feature request](https://github.com/3dg1luk43/ha_washdata/issues/new?template=feature_request.yml) or
   [Documentation](https://github.com/3dg1luk43/ha_washdata/issues/new?template=documentation.yml).
   Blank issues are disabled.
2. **Tick the box** saying you want to build it ("Contributing a Fix" / "Contributing an Implementation").
3. **Wait for the `accepted` label.** Only the maintainer adds it; use the issue to agree scope and
   approach.
4. **Open the PR**, fill in the template, and link the issue with `Closes #NNN`.

Closed automatically:

- an issue that skips the template (a template with an empty field is not closed; the bot asks
  you to fill it in)
- a PR with no linked `accepted` issue, or opened before its issue was accepted
- a PR whose template is deleted or left empty

If a bot or the maintainer closed your issue and the problem is still there, comment `/reopen` on it.

**Translations are the exception:** they need no issue or label (see below). Documentation PRs go
through the same flow as code.

## Bug reports

Include your WashData and Home Assistant versions, steps to reproduce, and the error text from the
Home Assistant log (a few full lines are enough). A diagnostics download from the device helps most.

## Development setup

Python 3.13.

```bash
git clone https://github.com/YOUR_USERNAME/ha_washdata.git
cd ha_washdata
git submodule update --init --recursive   # translation tooling
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
./devtools/install_hooks.sh               # pre-commit hook for the panel build
```

Before you push:

```bash
./run_tests.sh                                              # fast suite
python3 -m compileall custom_components/ha_washdata tests/ -q
```

More test modes, the real-Home-Assistant test box and the mock MQTT plug:
[Testing](https://github.com/3dg1luk43/ha_washdata/wiki/Testing). Architecture and the detailed
rules: [CLAUDE.md](CLAUDE.md) and [How it works](https://github.com/3dg1luk43/ha_washdata/wiki/Implementation-Details).

## Rules for code changes

- **NumPy only.** No SciPy, scikit-learn or other libraries at runtime, ML code included.
- **Local only.** No external API calls outside the opt-in Community Store.
- **Time:** `dt_util.now()` for timestamps, UTC (`time_utils.utc_now()`) for any interval.
- **No UI text in code.** Home Assistant strings go in `strings.json` and `translations/en.json`;
  panel strings go through `_t()` with the English text in `translations/panel/en.json`.
- **Event data stays under 32 KB:** never put power traces in a fired event.
- **Panel edits:** after changing `www/*.js`, run `node devtools/build_panel.mjs` and commit the
  rebuilt `.min.js` files and `build-manifest.json` in the same commit. The pre-commit hook and CI
  check this.
- **New services** need entries in `services.yaml`, `strings.json` and a schema.
- **Bug fixes come with a test** that reproduces the issue (`tests/test_issue_<number>_*.py`).
- **User-facing changes get a CHANGELOG entry:** one or two lines, symptom and fix.
- One fix or feature per PR. Screenshots for UI changes.

PR titles: `[FIX]`, `[FEATURE]`, `[REFACTOR]`, `[DOCS]`, `[TEST]`, `[PERF]`, e.g.
`[FIX] Handle a power sensor going unavailable mid-cycle`.

## Translations

Translate on [GitLocalize](https://gitlocalize.com/repo/10819); it opens the PR for you. To fix a
bad translation, correct it there instead of opening an issue.

- Keep every `{placeholder}` exactly as it is.
- Think appliance, not sport: a "match" is a recognised program, "logs" are diagnostic output.
- **No machine translation.** It has corrupted these files before.

## License

Contributions are licensed under [AGPL-3.0-or-later](LICENSE), like the project. You keep the
copyright in your own work.
