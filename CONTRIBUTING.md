# Contributing to WashData

Thank you for your interest in contributing to WashData! 🎉 This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Before contributing, please review our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold these standards.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Types of Contributions](#types-of-contributions)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Git Commit Messages](#git-commit-messages)
- [Localization & Translations](#localization--translations)
- [Questions & Support](#questions--support)

---

## Getting Started

### Prerequisites

- Python 3.11+
- Home Assistant development environment knowledge (helpful but not required)
- Git and GitHub account

### Fork & Clone

1. **Fork the repository** on GitHub
2. **Clone your fork locally**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ha_washdata.git
   cd ha_washdata
   ```
3. **Add upstream remote** to stay in sync:
   ```bash
   git remote add upstream https://github.com/3dg1luk43/ha_washdata.git
   ```
4. **Initialize translator submodule** (required for translation tooling):
   ```bash
   git submodule update --init --recursive
   ```
   This populates `scripts/ha_integration_translator`.

### Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or for bug fixes:
git checkout -b fix/brief-description
```

Use descriptive branch names (e.g., `feature/cycle-detection-improvement`, `fix/timezone-bug`).

---

## Development Setup

### 1. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

On Windows, if VS Code does not auto-detect the interpreter from `.venv`, set local
workspace overrides to `.venv\\Scripts\\python.exe` (and use that interpreter for pytest).

### 2. Install Dependencies

```bash
pip install -r requirements-dev.txt
```

### 3. Syntax Check

Verify your Python code before committing:

```bash
python3 -m compileall custom_components/ha_washdata tests/ --quiet
```

### 4. Run Tests Locally

```bash
./run_tests.sh
# or manually:
pytest tests/ -v
```

### 5. Testing with Mock Socket

To simulate a washing machine with power readings:

```bash
python3 devtools/mqtt_mock_socket.py --default LONG --variability 0.15
```

For full mock testing guide, see [TESTING.md](TESTING.md).

---

## Types of Contributions

### 🐛 Bug Reports

Found a bug? Open an issue using our [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml). Include:

- Clear description of the issue
- Steps to reproduce
- Your WashData version and Home Assistant version
- Relevant logs or error messages

### ✨ Feature Requests

Have an idea? Open an issue using our [feature request template](.github/ISSUE_TEMPLATE/feature_request.yml). Describe:

- What the feature should do
- Why it would be useful
- How it should work (with examples if possible)

### 🌍 Translations

**IMPORTANT**: Submit translations via Pull Request, not as issues.

See [Localization & Translations](#localization--translations) section below.

### 📚 Documentation

Improve READMEs, guides, or docstrings:

1. Edit the relevant `.md` or `.py` file
2. Submit a PR with clear changes
3. Documentation improvements are always welcome!

### 🔍 Code Review

Review open PRs and provide constructive feedback. Even experienced contributors value a second set of eyes.

---

## Pull Request Process

### Before You Start

1. **Sync with upstream**: Ensure your branch is up-to-date with `main`
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Check for existing work**: Search issues and PRs to avoid duplicate efforts

3. **For large changes**: Open an issue first to discuss with maintainers

### Making Your Changes

1. **Write clean, focused code**:
   - One feature/fix per PR (don't mix unrelated changes)
   - Follow the project's coding standards (see below)
   - Include comments for complex logic

2. **Test your changes**:
   ```bash
   ./run_tests.sh
   python3 -m compileall custom_components/ha_washdata tests/ --quiet
   ```

3. **Update documentation**:
   - Docstrings for functions/classes
   - README if adding UI components
   - CHANGELOG if it's a user-facing change

### Submitting Your PR

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request** on GitHub with:
   - Clear title describing the change
   - Detailed description (use our PR template)
   - Reference any related issues: `Closes #123`
   - Screenshots for UI changes

3. **Respond to reviews**:
   - Be open to feedback
   - Make requested changes promptly
   - If you disagree, explain your reasoning
   - Re-request review after making changes

### PR Title Format

- `[FEATURE]` - New features
- `[FIX]` - Bug fixes
- `[REFACTOR]` - Code reorganization (no behavior change)
- `[DOCS]` - Documentation only
- `[TEST]` - Test improvements
- `[PERF]` - Performance improvements

Example: `[FIX] Handle timezone-aware datetime in cycle detection`

---

## Coding Standards

### Python Style

- **PEP 8** compliance (with Black formatter preferences)
- **Type hints** for better IDE support
- **Docstrings** for all public functions/classes (Google style)
- **No hardcoded UI strings** - use `strings.json` and `translations/` for all user-facing text

Example:

```python
async def async_method(self, device_id: str, config: dict[str, Any]) -> None:
    """
    Perform an async operation.
    
    Args:
        device_id: The device identifier
        config: Configuration dictionary
        
    Returns:
        None
    """
    # Implementation
```

### Project Guardrails

- **ONLY NumPy** for calculations (no SciPy or ML libraries)
- **No external API calls** - all processing must be local
- **Timezone-aware datetimes** - ALWAYS use `dt_util.now()`
- **No inline UI strings** - use translation keys instead
- **Respect 32KB event data limit** - exclude large data from fired events

See [copilot-instructions.md](.github/copilot-instructions.md) for full technical details.

### File Organization

- New feature files go in `custom_components/ha_washdata/`
- Tests go in `tests/` mirroring the module structure
- Documentation additions go in `doc/` folder
- Do NOT modify core architecture files without discussion

---

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_cycle_detector.py -v

# With coverage
pytest tests/ --cov=custom_components.ha_washdata
```

### Writing Tests

- Use `pytest` framework
- Mock external dependencies (Home Assistant services, etc.)
- Aim for >80% code coverage on new code
- Test both happy-path and edge cases

Example test structure:

```python
import pytest
from unittest.mock import Mock, AsyncMock

@pytest.fixture
def mock_manager():
    """Fixture providing a mock WashDataManager."""
    return Mock()

@pytest.mark.asyncio
async def test_cycle_detection(mock_manager):
    """Test that cycles are detected correctly."""
    mock_manager.some_method.return_value = "expected_value"
    assert mock_manager.some_method() == "expected_value"
```

---

## Git Commit Messages

Write clear, descriptive commit messages:

```text
type: brief summary (50 chars max)

Longer explanation of what changed and why. Wrap at 72 characters.
Explain the problem you're solving, not just the code changes.

- Bullet point for major changes
- Another detail

Fixes #123
```

**Types**:
- `feat:` New feature
- `fix:` Bug fix
- `refactor:` Code reorganization
- `docs:` Documentation
- `test:` Test additions/improvements
- `perf:` Performance improvements
- `chore:` Build, config, dependencies

**Examples**:
- `feat: Add predictive end-time calculation`
- `fix: Handle null power readings gracefully`
- `docs: Update TESTING.md with mock socket guide`

---

## Localization & Translations

### 🚨 CRITICAL: TRANSLATIONS VIA PR, NOT ISSUES

**Do NOT open an issue to report bad translations.** Instead:

1. **Edit the translation file** for your language
   - Location: `custom_components/ha_washdata/translations/[language-code].json`
   - Example: `translations/ru.json`, `translations/es.json`

2. **Submit a Pull Request** with your corrections
   - Include a brief description of what was corrected

3. **Use translation script** (for Home Assistant translations):
   ```bash
   source .venv_translation/bin/activate
   python3 scripts/ha_integration_translator/translate.py custom_components/ha_washdata/translations --all
   ```

4. **Protect manually curated keys from auto-overwrite** (recommended):
   - Add key paths to `custom_components/ha_washdata/translations/.translation-locks.json`
   - Exact key lock example: `options.step.manage_profiles.title`
   - Prefix lock example: `options.common_text.*`
   - On next automated translation run, locked existing keys are preserved.

### Adding a New Language

1. Copy `translations/en.json` to `translations/[new-language-code].json`
2. Translate all values (keep keys unchanged)
3. Include in strings.json if needed
4. Submit PR with translations

**Note**: Translations are validated automatically. Ensure JSON is valid before submitting.

---

## Questions & Support

### Getting Help

- **Question about contributing?** → Open a Discussion on GitHub
- **Found a bug?** → Open an Issue with the bug report template
- **Have a feature idea?** → Open an Issue with the feature request template
- **Need development help?** → Reach out to maintainers via discussion

### Communication

- **Be respectful** and assume good intentions
- **Search first** - your question may already be answered
- **Provide context** - share relevant code/logs
- **Be patient** - maintainers are volunteers

---

## Recognition

Contributors to WashData are recognized in:

- [CHANGELOG.md](CHANGELOG.md) for significant contributions
- GitHub's contributor graph
- Project README acknowledgments (major contributors)

Thank you for making WashData better! 🌟

---

**Last Updated**: 2026-03-11
