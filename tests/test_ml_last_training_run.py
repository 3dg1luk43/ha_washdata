"""Regression test: "Last trained" reflects the last training RUN, not the last
model promotion.

Bug: `_last_ml_training_at()` derived the timestamp from the newest promoted
model's `trained_at`, so a "Train now" that completed without promoting anything
(the fresh attempt didn't beat the baseline on held-out data) left the displayed
date stuck at the last promotion. The fix persists a last-run timestamp that
advances on every completed run.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.manager import WashDataManager
from custom_components.ha_washdata.profile_store import ProfileStore


# ---------------------------------------------------------------------------
# Store round-trip
# ---------------------------------------------------------------------------


@pytest.fixture
def store():
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(MagicMock(), "entry")
        ps.async_save = AsyncMock()
        yield ps


async def test_last_training_run_round_trip(store):
    assert store.get_ml_last_training_run() is None
    await store.set_ml_last_training_run("2026-07-03T10:00:00+00:00")
    assert store.get_ml_last_training_run() == "2026-07-03T10:00:00+00:00"


# ---------------------------------------------------------------------------
# _last_ml_training_at semantics
# ---------------------------------------------------------------------------


def _bound(*, last_run, versions):
    mgr = MagicMock()
    mgr.profile_store.get_ml_last_training_run.return_value = last_run
    mgr.profile_store.get_ml_model_versions.return_value = versions
    return WashDataManager._last_ml_training_at.__get__(mgr, WashDataManager)


def test_prefers_last_run_over_promotion_date():
    # A run happened Jul 3, but the only promoted model is from Jul 1.
    # "Last trained" must reflect the run (Jul 3), not the stale promotion.
    fn = _bound(
        last_run="2026-07-03T19:00:00+00:00",
        versions={"end": {"trained_at": "2026-07-01T07:59:00+00:00"}},
    )
    result = fn()
    assert result is not None
    assert result.isoformat().startswith("2026-07-03")


def test_falls_back_to_promotion_when_no_run_recorded():
    # Pre-fix installs never recorded a run -> fall back to newest promotion.
    fn = _bound(
        last_run=None,
        versions={
            "end": {"trained_at": "2026-07-01T07:59:00+00:00"},
            "quality": {"trained_at": "2026-06-20T00:00:00+00:00"},
        },
    )
    result = fn()
    assert result is not None
    assert result.isoformat().startswith("2026-07-01")


def test_none_when_never_trained():
    fn = _bound(last_run=None, versions={})
    assert fn() is None


def test_ignores_malformed_run_timestamp():
    # A bad stored value falls through to the promotion fallback.
    fn = _bound(
        last_run="not-a-date",
        versions={"end": {"trained_at": "2026-07-01T07:59:00+00:00"}},
    )
    result = fn()
    assert result is not None
    assert result.isoformat().startswith("2026-07-01")
