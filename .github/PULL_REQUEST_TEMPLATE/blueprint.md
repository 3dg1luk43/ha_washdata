# Automation template PR

Automation templates are Home Assistant blueprints that ship with WashData. A PR that only touches
`custom_components/ha_washdata/blueprints/` needs no accepted issue, but every template is reviewed
before it is merged. See [CONTRIBUTING](../../CONTRIBUTING.md#automation-templates).

## Description

<!-- What does the template do, for which appliances, and why is it useful? -->

## Type of Change

- [x] Automation template

## Testing

<!-- Your Home Assistant and WashData versions, the appliance, and what happened when it ran. -->

- [ ] I ran it on my own Home Assistant through at least one real cycle
- [ ] `pytest tests/test_issue_478_blueprints.py` passes

## Checklist

- [ ] One template per PR: a YAML file in `blueprints/automation/` and its entry in `blueprints/catalog.json`
- [ ] The `washdata_device` input is a device selector filtered to `integration: ha_washdata`, and the triggers are WashData device triggers
- [ ] Every other input has a default where a sensible one exists (then it installs in one click)
- [ ] The description says plainly what it switches or sends, and any risk
- [ ] It never starts a heating appliance's program (restoring a plug's power is stated in the description and can be limited), and a new cycle starting cancels anything still waiting
- [ ] `source_url` points at the file's own path on `main`
- [ ] No em dash characters

## Notes for reviewers

<!-- Anything else worth knowing. -->
