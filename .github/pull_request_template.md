## Description

<!-- Brief summary of the changes. What problem does this solve? -->

Fixes #(issue number)

## Type of Change

<!-- Mark the relevant option with an "x" -->

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 🔄 Refactor (code reorganization with no behavior change)
- [ ] 📚 Documentation update
- [ ] 🧪 Test additions/improvements
- [ ] ⚡ Performance improvement
- [ ] 🎨 UI/UX improvement
- [ ] 🌍 Translation/Localization

## Changes Made

<!-- Describe what you changed and why. Be specific. -->

- [ ] Change 1
- [ ] Change 2
- [ ] Change 3

## Testing

<!-- How have you tested these changes? -->

- [ ] Unit tests added/updated
- [ ] Manual testing completed
- [ ] Tested with mock socket: `python3 devtools/mqtt_mock_socket.py`
- [ ] Ran: `./run_tests.sh`

<!-- For bugs, include reproduction steps: -->
**Tested on:**
- Home Assistant version: X.Y.Z
- WashData version: X.Y.Z
- Device type(s): [Washing Machine / Dryer / Dishwasher / Other]

## Breaking Changes?

- [ ] This PR includes **breaking changes**

<!-- If breaking changes, describe them and migration path: -->
**If breaking:**
Describe the breaking change and how to migrate...

## Checklist

<!-- Ensure these before submitting -->

- [ ] My code follows the project's code standards (PEP 8, type hints)
- [ ] I've added/updated docstrings for new functions/classes
- [ ] I've added corresponding tests (if applicable)
- [ ] I've updated documentation (README, CHANGELOG, etc. if applicable)
- [ ] I've checked that `python3 -m compileall custom_components tests` passes
- [ ] I've run `./run_tests.sh` and all tests pass
- [ ] **No hardcoded UI strings** — all user-facing text is in `strings.json` and `translations/`
- [ ] I've reviewed my own code for quality
- [ ] I've synced with upstream: `git fetch upstream && git rebase upstream/main`

## For Translations

<!-- If this is a translation PR, include: -->

- [ ] Language: [e.g., Russian, Spanish, French]
- [ ] I've verified JSON syntax: `python3 -m json.tool custom_components/ha_washdata/translations/[lang].json`
- [ ] All keys match the English translation file

## Screenshots / Demo

<!-- Add screenshots, GIFs, or describe the visual impact if applicable -->

<!-- Example for UI changes:
Before:
[screenshot]

After:
[screenshot]
-->

## Related Issues

<!-- Link related issues or discussions -->

- Related to: #(issue number)
- Closes: #(issue number)

## Notes for Reviewers

<!-- Any additional context, concerns, or guidance for the reviewer? -->

---

**Thank you for contributing to WashData!** 🙏

Before hitting submit, please make sure:
1. ✅ PR title clearly describes the change
2. ✅ Description is detailed enough for reviewers to understand
3. ✅ Tests pass locally
4. ✅ You've reviewed the [CONTRIBUTING.md](../CONTRIBUTING.md) guide
5. ✅ You've read our [Code of Conduct](../CODE_OF_CONDUCT.md)
