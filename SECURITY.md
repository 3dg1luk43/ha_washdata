# Security Policy

## Report a vulnerability

**Do not open a public issue.** Report it privately through
[GitHub Security Advisories](https://github.com/3dg1luk43/ha_washdata/security/advisories)
("Report a vulnerability"). There is no security e-mail address.

Include:

- what the problem is and what an attacker could do with it
- steps to reproduce
- affected versions (release tags, e.g. `v0.5.6 - v0.5.7`)
- a suggested fix, if you have one, and whether you want to be credited

What to expect: an acknowledgement within 48 hours, confirmation within a week, and a fix in a new
release (critical issues first). The advisory is published on GitHub once the fix is out, and
reporters are credited unless they ask not to be. Please do not disclose the issue publicly before
the fix is released.

## Supported versions

| Version | Security fixes |
| :--- | :--- |
| Latest release of the current minor line (e.g. `0.5.x`) | Yes |
| Previous minor line | Best effort, for high-impact issues |
| Older | No |

## What WashData sends and stores

- **Local by default.** Detection, matching, learning and the optional ML models run inside Home
  Assistant. No telemetry, no downloads of models or firmware.
- **Stored data** (programs, cycles, power traces, exports) lives in your Home Assistant config
  folder, unencrypted, protected by the file system.
- **Notifications** go wherever your chosen notify service sends them (for example a phone push
  through a cloud service). Without notify targets, automations or the Community Store's online
  features, nothing leaves Home Assistant.
- **Community Store** (off by default): when you enable online features, browsing fetches shared
  setups, and only what you choose to share is uploaded (program curves, reference traces, appliance
  brand, model and type). Your name is shown only if you turn that on. Turning online features off
  clears the stored credential.

## Access control

Every panel command is checked on the server, not only hidden in the UI. Services apply the same
checks.

- Optional per-user access per device (None / Read / Edit / Full), set in the panel's Access Control.
  Off by default; administrators always have full access.
- Administrator-only, even with access control off: export and import (panel and services),
  wiping history, reprocessing history, clearing debug data, the ML training service, panel configuration, and
  every Community Store account, sharing, rating and publishing action.
- Background tasks check access to the device they belong to. A path given to the export or import
  service must be in Home Assistant's `allowlist_external_dirs`; an export there never overwrites a file.

## Dependencies

NumPy is the only third-party runtime dependency (`manifest.json`). The ML code is NumPy-only as
well.

## Electrical safety

Not a software issue, but the real risk: a smart plug on a washer, dryer or dishwasher carries the
heating load for hours. Use a plug rated in watts at your mains voltage for the appliance's peak
power (16 A is 3680 W at 230 V, but 15 A at 120 V is only 1800 W) or a hardwired module, and check
it regularly. See
[Smart plugs](https://github.com/3dg1luk43/ha_washdata/wiki/Smart-Plugs).
