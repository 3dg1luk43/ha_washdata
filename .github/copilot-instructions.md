# WashData - AI development instructions

**All project guidance lives in [`CLAUDE.md`](../CLAUDE.md) at the repository root.** Read that file
first: it covers the architecture, the build and test commands, the generated files that must be
regenerated rather than hand-edited, and the hard rules (NumPy-only runtime, no machine translation,
timezone-aware datetimes, the three cycle lists).

For deeper detail, `docs/internal/INTEGRATION_REFERENCE.md` is the canonical engineering reference and
carries the discrepancy and tech-debt register.

This file is deliberately a pointer. A second copy of the project description drifts out of date the
moment the first one changes, which is exactly what happened to the version of this file that stood
here until 0.5.6 (it still described 0.4.3).
