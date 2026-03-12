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

- **Fire Risk**: Cheap smart plugs rated for <10A may overheat under sustained loads >2500W
- **Recommendations**:
  - Use plugs rated for **peak power** of your appliance
  - For washing machines/dryers: **16A+** plugs or hardwired modules
  - Inspect hardware regularly for damage
  - Use reputable brands (Shelly, Sonoff, etc.)

⚡ **The WashData team is NOT responsible for electrical damage caused by improper hardware.**

### Data Privacy

- **No cloud dependencies**: All data stays local to your Home Assistant instance
- **No tracking**: WashData does not collect or send usage analytics
- **Configuration backups**: Exports are JSON files stored on your device
- **Power history**: Power readings are stored only on your HA instance

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

✅ Process power readings locally (no external calls)  
✅ Use NumPy for calculations (stable, well-audited library)  
✅ Store config/profiles locally in Home Assistant  
✅ Support manual profile creation (no auto-learning from external sources)  

### What WashData DOES NOT DO

❌ Send any data to cloud services  
❌ Phone home with usage analytics  
❌ Download profiles or firmware updates  
❌ Run arbitrary user code  
❌ Store sensitive data unencrypted on disk  

### Dependencies

WashData's dependencies are minimal:

- **numpy**: Numerical computations (security-conscious library)
- **Home Assistant**: Core framework (actively maintained)

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

*Last Updated: 2026-03-11*
