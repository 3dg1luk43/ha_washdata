"""Register item 570: the matching template follows the envelope's golden rule.

``build_match_snapshots`` pinned a profile's template to its sample cycle whenever
ANY of its cycles carried ``ml_review.golden``, the raw flag. Older versions forced
that flag on every community-store import, and the envelope build stopped honouring
it in a profile with cycles of the user's own (``_effective_golden_flags``, item
408 / audit STORE-04), but the template did not: one old import replaced the
envelope template of a profile with 20 of the user's own cycles. The pin (and its
mirror in ``_stage1_duration_for``) now reads ``_pins_golden_template``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from custom_components.ha_washdata.profile_store import ProfileStore, _pins_golden_template


@pytest.fixture
def store(mock_hass):
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(mock_hass, "e", min_duration_ratio=0.1, max_duration_ratio=1.8)
        ps._store.async_save = AsyncMock()  # noqa: SLF001
        yield ps


def _cycle(cid: str, k: int, *, duration: float = 3600.0, golden: bool = False,
           store_import: bool = False) -> dict:
    t = np.arange(0.0, duration + 1, 30.0)
    w = np.where(t < 900, 2000.0 + 40 * k, np.where(t < duration - 600, 180.0 + 10 * np.sin(t / 50 + k), 600.0))
    c = {
        "id": cid, "profile_name": "Eco", "status": "completed", "duration": duration,
        "start_time": f"2026-01-0{k + 1}T08:00:00+00:00",
        "power_data": [[float(a), float(b)] for a, b in zip(t, w)],
    }
    if golden:
        c["ml_review"] = {"golden": True}
    if store_import:
        c["meta"] = {"source": "store:abc"}
    return c


async def _snap(store: ProfileStore, past: list[dict], refs: list[dict], sample: str) -> dict:
    store._data["profiles"]["Eco"] = {"avg_duration": 3700.0, "sample_cycle_id": sample}  # noqa: SLF001
    store._data["past_cycles"] = past  # noqa: SLF001
    store._data["reference_cycles"] = refs  # noqa: SLF001
    assert await store.async_rebuild_envelope("Eco")
    # The rebuild re-picks the sample and the mean; the test pins the sample an old
    # store kept, and a rolling mean apart from the envelope's target.
    store._data["profiles"]["Eco"].update(sample_cycle_id=sample, avg_duration=3700.0)  # noqa: SLF001
    return next(s for s in store.build_match_snapshots(30.0) if s["name"] == "Eco")


def _uses_envelope(store: ProfileStore, snap: dict) -> bool:
    env = store._data["envelopes"]["Eco"]  # noqa: SLF001
    return "sample_dt" not in snap and snap["sample_power"] == store._envelope_template_on_grid(  # noqa: SLF001
        "Eco", env, 30.0
    )


async def test_an_old_forced_golden_import_no_longer_replaces_the_users_envelope(store) -> None:
    own = [_cycle(f"r{k}", k) for k in range(4)]
    imp = _cycle("s0", 7, duration=4800.0, golden=True, store_import=True)
    snap = await _snap(store, own, [imp], sample="s0")

    assert _uses_envelope(store, snap)
    env = store._data["envelopes"]["Eco"]  # noqa: SLF001
    # Stage 1 divides by what the snapshot is sized by: the envelope's target.
    assert store._stage1_duration_for("Eco", store._data["profiles"]["Eco"]) == pytest.approx(  # noqa: SLF001
        float(env["target_duration"])
    )


async def test_the_users_own_golden_cycle_still_pins_the_template(store) -> None:
    own = [_cycle("r0", 0, golden=True)] + [_cycle(f"r{k}", k) for k in range(1, 4)]
    snap = await _snap(store, own, [_cycle("s0", 7, golden=True, store_import=True)], sample="r0")

    assert "sample_dt" in snap and not _uses_envelope(store, snap)
    # The pinned profile keeps its own avg_duration.
    assert store._stage1_duration_for("Eco", store._data["profiles"]["Eco"]) == pytest.approx(3700.0)  # noqa: SLF001


async def test_a_store_only_profile_keeps_what_it_had(store) -> None:
    flagged = [_cycle("s0", 0, golden=True, store_import=True),
               _cycle("s1", 1, store_import=True), _cycle("s2", 2, store_import=True)]
    snap = await _snap(store, [], flagged, sample="s0")
    assert "sample_dt" in snap, "a forced-golden import still pins a profile with nothing else"

    unflagged = [_cycle(f"s{k}", k, store_import=True) for k in range(3)]
    snap = await _snap(store, [], unflagged, sample="s0")
    assert _uses_envelope(store, snap), "an unflagged store-only profile stays on its envelope"


def test_pins_golden_template_rule() -> None:
    own, own_gold = _cycle("r0", 0), _cycle("r1", 1, golden=True)
    imp, imp_gold = _cycle("s0", 2, store_import=True), _cycle("s1", 3, golden=True, store_import=True)
    assert not _pins_golden_template([own, imp_gold])
    assert _pins_golden_template([own_gold, imp_gold])
    assert _pins_golden_template([imp, imp_gold])
    assert not _pins_golden_template([imp, dict(imp, id="s9")])
    assert not _pins_golden_template([own])
    assert not _pins_golden_template([])
