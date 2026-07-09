# Security Policy

## Reporting a Vulnerability

**⚠️ DO NOT open a public issue to report security vulnerabilities.**

If you discover a security vulnerability in WashData, please report it responsibly:

### Private Reporting (Recommended)

1. **GitHub Security Advisory**: Report via GitHub's private vulnerability disclosure
   - Go to: https://github.com/3dg1luk43/ha_washdata/security/advisories
   - Click **"Report a vulnerability"**
   - Describe the issue with steps to reproduce

2. **Email**: Private email reporting is not currently published.
   - Use GitHub Security Advisories above for confidential reporting.

### What to Include

Please provide:

- Clear description of the vulnerability
- Potential impact (how severe is it?)
- Steps to reproduce (if applicable)
- Your suggested fix or workaround (if any)
- Whether you want attribution in the security advisory

### Supported Versions

Security fixes are provided for currently supported release lines, identified by
semantic version tags (for example `v0.5.0`):

- Supported: latest patch release on the current minor line (for example `0.5.x`)
- Best effort: previous minor line when impact is high and patching is low-risk
- End-of-life (EOL): older minors and all unsupported majors

EOL versions may still be triaged, but fixes are not guaranteed and are usually
delivered only in a newer supported release.

Backport policy:

- Critical vulnerabilities may be backported to another supported line
- High/medium vulnerabilities are normally fixed in the latest supported line
- Reporters should expect fix timelines to follow the response timeline below,
  with critical issues prioritized

Include affected versions in reports using release tags or semver ranges.
Example:

`Affected versions: v0.4.0 - v0.4.9 (fixed in v0.5.0)`

### Response Timeline

We aim to:

- **Acknowledge** your report within 48 hours
- **Investigate** and confirm the vulnerability within 1 week
- **Develop** and test a fix within 2 weeks
- **Release** a patched version (if critical) or include in next release
- **Publish** a security advisory on GitHub

---

## Security Considerations

### ⚠️ Electrical Safety (Not a Software Issue, But Important)

This integration monitors **high-power appliances** via smart plugs. **Electrical hazards are real:**

- **Fire Risk**: Cheap smart plugs rated for <10A may overheat under sustained high loads (W = A × V; e.g., 10 A ≈ 1200 W at 120 V or ≈ 2300–2500 W at 230–250 V) - check your region's voltage
- **Recommendations**:
  - Use plugs rated for **peak power** of your appliance
  - For washing machines/dryers: **16A+** plugs or hardwired modules
  - Inspect hardware regularly for damage
  - Use reputable brands (Shelly, Sonoff, etc.)

⚡ **The WashData team is NOT responsible for electrical damage caused by improper hardware.**

### Data Privacy

- **Core processing is local**: Detection logic and matching run inside your Home Assistant instance
- **No tracking**: WashData does not collect or send usage analytics
- **Configuration backups**: Exports are JSON files stored on your device/instance
- **Power history**: Power readings are stored on your Home Assistant instance
- **Notification routing is configurable**: If you configure per-event notify targets (`notify_start/finish/live_services`) or build your own automations on WashData's cycle events, payloads may be delivered by your selected notify integration (for example mobile-app push, email, or other cloud-backed providers). Without any notify target or automation, data remains local to Home Assistant.
- **Management panel & access control**: The full-screen panel is backed by a WebSocket API, and every command is authorised **server-side** against a per-user access level (none / read / edit / full, per device). Access control is off by default (single-user setups are unaffected); when enabled, read-only users cannot mutate data and destructive or export/import actions require full access. Authorisation is enforced on the server, not merely hidden in the UI.
- **On-device ML**: The optional, experimental ML subsystem is **NumPy-only** and runs entirely on your instance - it downloads no models, calls no external services, and sends no data out. It is off by default.

### Home Assistant Security

To secure your Home Assistant installation:

- Use **strong passwords** for HA accounts
- Enable **two-factor authentication** (2FA) if available
- Keep Home Assistant **updated** to the latest version
- Restrict network access to your HA instance
- Use VPN for remote access (don't expose HA directly to the internet)
- Review **trusted devices** and sessions regularly

---

## Known Security Guidelines

### What WashData DOES

✅ Process core power readings and detection locally (no external calls); optional notification or integration features may transmit payloads externally  
✅ Use NumPy for calculations (stable, well-audited library)  
✅ Store config/profiles locally in Home Assistant  
✅ Support manual profile creation (no auto-learning from external sources)  

### What WashData DOES NOT DO

❌ Send analytics/telemetry to cloud services on its own  
❌ Phone home with usage analytics  
❌ Download profiles or firmware updates  
❌ Run arbitrary user code  
ℹ️ Store configuration and power history locally without encryption (relies on filesystem permissions)  
ℹ️ Optional per-event notify targets, or user-built automations on WashData's cycle events, can forward notification payloads outside the instance; this depends on user configuration.  
ℹ️ From the management panel a user with sufficient access can create or delete Home Assistant automations and import a configuration (which overwrites data); these are explicit, permission-gated user actions (see access control below), not autonomous behaviour.  

### Dependencies

WashData's runtime dependencies are minimal:

- **numpy**: Numerical computations (security-conscious library). This is the **only** third-party runtime dependency - including the ML subsystem, which is deliberately NumPy-only (no scikit-learn / PyTorch / SciPy ship at runtime).
- **Home Assistant**: Core framework (actively maintained). WashData declares `frontend`, `websocket_api`, and (optional) `recorder` as `after_dependencies`.

All dependencies are specified in `manifest.json`.

---

## Vulnerability Disclosure Policy

We follow **responsible disclosure** principles:

1. **Private First**: Security issues are reported privately and handled confidentially
2. **Timely Response**: We respond quickly and work toward fixes in good faith
3. **Coordinated Release**: Fixes are released promptly after verification
4. **Transparency**: Security advisories are published on GitHub's security tab
5. **Credit**: Reporters are credited unless they request anonymity

### Do NOT

- Publicly disclose the vulnerability before a fix is available
- Exploit the vulnerability for malicious purposes
- Demand payment for not disclosing
- Disclose other private information (like user credentials)

---

## Security Best Practices

### For Users

- ✅ Install WashData **only from HACS** or the official GitHub repository
- ✅ Keep Home Assistant **up-to-date**
- ✅ Use **strong, unique passwords** for HA
- ✅ Review **installed custom components** regularly
- ✅ Monitor **HA logs** for errors or unusual activity

### For Contributors

- ✅ Never commit secrets, API keys, or credentials
- ✅ Review code before submission (security awareness)
- ✅ Use **type hints** (helps catch subtle bugs)
- ✅ Sanitize user input (prevent injection attacks)
- ✅ Follow PEP 8 and linting standards

---

## Security Advisories

Published security advisories are available at:

- GitHub Advisory Database: https://github.com/3dg1luk43/ha_washdata/security/advisories
- GitHub Security Tab: https://github.com/3dg1luk43/ha_washdata/security

---

## Questions?

If you have security questions (not a vulnerability):

- Post in **GitHub Discussions** (public, but respectful)
- Tag questions with `[SECURITY]` for visibility
- Do not share sensitive details publicly

---

**Thank you for helping keep WashData secure.** 🛡️

*Last Updated: 2026-07-02*
