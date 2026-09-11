> [!CAUTION]
> **Do not open this PR unless your linked issue already carries the `accepted` label.**
>
> WashData does not accept unsolicited code. Open an
> [issue](https://github.com/3dg1luk43/ha_washdata/issues/new/choose) first, tick "I plan to submit a
> PR", and wait for the `accepted` label so scope is agreed before you write anything.
>
> Closed automatically: PRs linking no issue, PRs opened in parallel with a brand-new issue, and PRs
> with this template deleted or unfilled. **Translation PRs (incl. GitLocalize) are exempt from all of
> the above.**

Closes #<!-- must already have the `accepted` label -->

## What and why

<!-- What problem does this solve, and how? -->

## Type

<!-- Delete what does not apply -->
Bug fix / Feature / Refactor / Docs / Tests / Performance / UI / Translation

## Testing

<!-- How did you verify this? Include HA version, WashData version and device type for a bug fix. -->

- [ ] `./run_tests.sh` passes
- [ ] Tests added or updated
- [ ] Manually verified

## Checklist

- [ ] The linked issue already had the `accepted` label (or this is a translation PR)
- [ ] No hardcoded UI strings - user-facing text is in `strings.json` / `translations/`
- [ ] Rebuilt panel artifacts if I touched `www/*.js` (`node devtools/build_panel.mjs`)
- [ ] Docs and CHANGELOG updated if the change is user-visible
- [ ] Breaking change? Describe it and the migration path here:

## Notes for reviewers

<!-- Anything else worth knowing. Screenshots welcome for UI changes. -->
