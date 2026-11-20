"""Review of PR #529, round 39a, each reproduced.

* #530: the check's user-pause guard covered only the dispatch. A check still
  scoring when the user pressed Pause counted on landing, so the sixth fired in
  the pause. It now lands as nothing; the run so far is kept.
* A real profile named like a group key (``__group__G``) that won was relabelled
  to a member of group G (Stage 5, the end gates' view, the hazard contenders):
  its own name left the ranking, so a pick of it read as a deficit. Only a row
  with the collapse's ``group_best_member`` provenance is a family now.
* A group member renamed while the matcher prepared the trace left the groups it
  read at the call naming the old name (a rename moves no data generation), so
  the cohesive group was skipped. The profile set revision re-reads them too.
* #531: an envelope build across a device type changed and back (no data
  generation) kept a dishwasher's cut-off runs, was memoised, and the sweep
  saved its marker. A build a match setting changed under is now discarded and
  the sweep keeps the profile pending.
* devtools/program_mismatch_eval.py: alias twins came from ``eval.trace_hash``
  (1 s / 1 W), so two programmes' runs a sub-watt apart were twins and a valid
  wrong pick was dropped. They come from trace content (``run_hash``) now.
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata import match_rules
from custom_components.ha_washdata.const import (
    CUT_SHORT_REBUILD_KEY,
    DOMAIN,
    PROGRAM_MISMATCH_PERSISTENCE,
)
from custom_components.ha_washdata.profile_store import ProfileStore, _ambiguity_contenders

from .test_issue_530_program_mismatch import _check, _mismatches, _past_half, _result
from .test_issue_531_dishwasher_drying_cutoff import T0, _eco
from .test_program_mismatch_eval import _export as _pme_export
from .test_program_mismatch_eval import pme

# ── #530: a check that lands in a user pause does not count ──────────────────


async def test_a_check_landing_in_a_user_pause_does_not_count(
    hass: HomeAssistant, setup_washdata_entry: Any, freezer: Any
) -> None:
    """Five hits, then Pause pressed while the sixth check scores: nothing fires
    and the five are kept. After the resume the next hit establishes it."""
    _entry, manager, sent, state = await _past_half(hass, setup_washdata_entry, freezer)
    wrong = _result("Synthetics", {"Synthetics": 0.9, "Cotton": 0.5})
    state.streak = PROGRAM_MISMATCH_PERSISTENCE - 1

    async def paused_while_scoring(*_a: Any, **_k: Any) -> Any:
        assert await manager.async_pause_cycle()
        return wrong

    with patch.object(manager.profile_store, "async_match_profile", new=paused_while_scoring):
        await _check(hass, manager, state.key)
    assert _mismatches(sent) == []
    assert state.streak == PROGRAM_MISMATCH_PERSISTENCE - 1 and not state.fired
    # A check completing without a pause still counts.
    assert await manager.async_resume_cycle()
    with patch.object(manager.profile_store, "async_match_profile", AsyncMock(return_value=wrong)):
        await _check(hass, manager, state.key)
    assert len(_mismatches(sent)) == 1


# ── A profile named like a group key is no family ────────────────────────────

N = 200
DUR_30, DUR_40, DUR_X = 5940.0, 6044.0, 4000.0


def _block(hot_start: float, hot_end: float) -> list[float]:
    return [
        1900.0 if (0.027 <= i / N < 0.037 or hot_start <= i / N < hot_end) else 60.0
        for i in range(N)
    ]


# Group G: a cohesive 30/40 pair (test_issue_400_group_best_member's shapes).
CURVE_30, CURVE_40 = _block(0.20, 0.218), _block(0.20, 0.30)
# The profile literally named "__group__G": another programme entirely.
CURVE_X = [
    800.0 if 0.05 <= i / N < 0.45 else (400.0 if 0.6 <= i / N < 0.8 else 20.0)
    for i in range(N)
]


def _envelope(curve: list[float], dur: float) -> dict[str, Any]:
    step = dur / (N - 1)
    return {"avg": [[i * step, p] for i, p in enumerate(curve)], "cycle_count": 3}


@pytest.fixture
def store(mock_hass: Any) -> Any:
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(mock_hass, "entry")
    ps.async_save = AsyncMock()
    ps._min_duration_ratio = 0.10  # noqa: SLF001
    ps.energy_mode = "integrated"
    ps._data["profiles"] = {  # noqa: SLF001
        "D30": {"avg_duration": DUR_30}, "D40": {"avg_duration": DUR_40},
        "__group__G": {"avg_duration": DUR_X},
    }
    ps._data["envelopes"] = {  # noqa: SLF001
        "D30": _envelope(CURVE_30, DUR_30), "D40": _envelope(CURVE_40, DUR_40),
        "__group__G": _envelope(CURVE_X, DUR_X),
    }
    ps._data["profile_groups"] = {"G": {"members": ["D30", "D40"]}}  # noqa: SLF001
    return ps


@pytest.mark.parametrize("in_progress", [True, False])
async def test_a_profile_named_like_a_group_key_keeps_its_win(
    store: Any, in_progress: bool
) -> None:
    """A run of the real "__group__G" profile: it wins and stays itself, not a
    member of group G; the family's own entry is still listed as the family."""
    upto = 120 if in_progress else N
    live = [[i * (DUR_X / N), p] for i, p in enumerate(CURVE_X[:upto])]
    res = await store.async_match_profile(live, DUR_X * upto / N, in_progress=in_progress)
    assert res.best_profile == "__group__G"
    top = res.ranking[0]
    assert top["name"] == "__group__G" and "stage5_group" not in top
    assert any(c.get("group_best_member") in ("D30", "D40") for c in res.ranking[1:])
    # The #530 check reads a pick of it as the winner, not a deficit.
    families = match_rules.program_families(store.get_profile_groups())
    assert match_rules.selected_program_deficit(res, "__group__G", families) is None


def test_the_end_gates_view_keeps_a_profile_named_like_a_group_key(store: Any) -> None:
    """``_single_template_gate``: the same relabel on the single-template ranking."""
    view = [
        {"name": "__group__G", "score": 0.9, "shape_score": 0.9, "profile_duration": DUR_X},
        {"name": "Other", "score": 0.5, "shape_score": 0.5, "profile_duration": 3000.0},
    ]
    with patch.object(store, "_stage5_pick_member", return_value=("D40", 0.9, DUR_40)) as pick:
        gate = store._single_template_gate(  # noqa: SLF001
            view, CURVE_X[:120], DUR_X * 0.6, {"__group__G": ["D30", "D40"]}, {}, None,
        )
    assert not pick.called
    assert gate["profile"] == "__group__G" and gate["expected"] == DUR_X
    # The family's own entry still resolves to its member.
    family = [{**view[0], "group_best_member": "D40"}, view[1]]
    with patch.object(store, "_stage5_pick_member", return_value=("D40", 0.9, DUR_40)):
        gate = store._single_template_gate(  # noqa: SLF001
            family, CURVE_X[:120], DUR_X * 0.6, {"__group__G": ["D30", "D40"]}, {}, None,
        )
    assert gate["profile"] == "D40"


def test_the_hazard_contenders_do_not_expand_a_profile_named_like_a_group_key() -> None:
    ranking = [
        {"name": "__group__G", "score": 0.90, "profile_duration": DUR_X},
        {"name": "Other", "score": 0.88, "profile_duration": 3000.0},
    ]
    groups = {"__group__G": ["D30", "D40"]}
    durations = {"D30": DUR_30, "D40": DUR_40}
    out = _ambiguity_contenders(ranking, groups, durations, "__group__G", DUR_X, False)
    assert out == (("__group__G", DUR_X), ("Other", 3000.0))
    family = [{**ranking[0], "group_best_member": "D40"}, ranking[1]]
    out = _ambiguity_contenders(family, groups, durations, "D40", DUR_40, False)
    assert out == (("D40", DUR_40), ("D30", DUR_30), ("Other", 3000.0))


# ── A member renamed during the trace preparation keeps its group ────────────


async def test_a_member_renamed_during_the_preparation_keeps_the_group(
    store: Any, mock_hass: Any
) -> None:
    """D30 renamed to D35 while the long trace is prepared in the executor: the
    templates are built under D35 and D40, so the groups must be read again
    (a rename moves no data generation, only ``profile_set_revision``)."""
    del store._data["profiles"]["__group__G"], store._data["envelopes"]["__group__G"]  # noqa: SLF001
    step = 5.0
    live = [
        [k * step, CURVE_30[min(N - 1, int(k * step / DUR_30 * N))]]
        for k in range(int(0.6 * DUR_30 / step))
    ]
    assert len(live) > 512  # prepared in the executor (_MATCH_PREP_INLINE_READINGS)
    run = mock_hass.async_add_executor_job.side_effect
    renamed: list[bool] = []

    async def rename_during_preparation(target: Any, *args: Any) -> Any:
        if not renamed and getattr(target, "__name__", "") == "_prepare_match_trace":
            renamed.append(True)
            await store.update_profile("D30", "D35")
        return await run(target, *args)

    with patch.object(mock_hass, "async_add_executor_job", side_effect=rename_during_preparation):
        res = await store.async_match_profile(live, live[-1][0], in_progress=True)
    assert renamed
    assert store.get_profile_groups()["G"]["members"] == ["D35", "D40"]
    # The family was formed under the renamed membership: its winner carries the
    # group, and its siblings do not stand side by side in the ranking.
    assert res.ranking[0].get("stage5_group") == "__group__G"
    assert res.best_profile in ("D35", "D40")
    assert not {"D35", "D40"} <= {c.get("name") for c in res.ranking}


# ── #531: a device type changed and back during an envelope build ────────────


async def _eco_store(hass: HomeAssistant, setup_washdata_entry: Any) -> Any:
    """A dishwasher with five full ECO runs and twelve cut inside the drying,
    from before the #531 rule (no marker, no envelope)."""
    entry = await setup_washdata_entry(device_type="dishwasher", options={"stop_threshold_w": 4.5})
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    await hass.async_block_till_done()
    runs = [_eco(hold_s=30)] * 5 + [_eco(cut_s=131 * 60)] * 12
    store._data["profiles"] = {"ECO": {}}  # noqa: SLF001
    store._data["envelopes"] = {}  # noqa: SLF001
    store._data["past_cycles"] = [  # noqa: SLF001
        {"id": f"c{i}", "profile_name": "ECO", "status": "completed",
         "termination_reason": "smart", "start_time": (T0 + timedelta(days=i)).isoformat(),
         "duration": pts[-1][0], "power_data": [[t, p] for t, p in pts]}
        for i, pts in enumerate(runs)
    ]
    store._data.pop(CUT_SHORT_REBUILD_KEY, None)  # noqa: SLF001
    return store


def _type_flips(store: Any, times: int) -> Any:
    """The envelope worker, run while the device type reads washing_machine: a
    reconfigure away and back in place (no data generation) over the job."""
    real = store._rebuild_envelope_sync  # noqa: SLF001
    calls = {"n": 0}

    def flipped(cycles: Any) -> Any:
        calls["n"] += 1
        if calls["n"] > times:
            return real(cycles)
        store.device_type = "washing_machine"
        try:
            return real(cycles)
        finally:
            store.device_type = "dishwasher"

    return flipped


async def test_a_build_across_a_type_change_and_back_is_discarded(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """Built as a washer (the cut-offs kept) and finished as a dishwasher again:
    not written, not memoised, so the next request builds under the rule."""
    store = await _eco_store(hass, setup_washdata_entry)
    with patch.object(store, "_rebuild_envelope_sync", side_effect=_type_flips(store, 1)):
        assert await store.async_rebuild_envelope("ECO") is False
        assert "ECO" not in store._data["envelopes"]  # noqa: SLF001
        assert "ECO" not in store.__dict__.get("_envelope_built_from", {})
        assert await store.async_rebuild_envelope("ECO") is True
    assert store._data["envelopes"]["ECO"]["shape_cycle_count"] == 5  # noqa: SLF001


async def test_the_cut_short_sweep_keeps_a_discarded_build_pending(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """The sweep counted the washer-built envelope done and saved the marker; it
    now retries the profile, and saves the marker only once it is built under
    the rule."""
    store = await _eco_store(hass, setup_washdata_entry)
    with patch.object(store, "_rebuild_envelope_sync", side_effect=_type_flips(store, 1)):
        await store.async_rebuild_for_cut_short_rule(device_type="dishwasher")
    assert not store.cut_short_rebuild_pending()
    env = store._data["envelopes"]["ECO"]  # noqa: SLF001
    assert env["shape_cycle_count"] == 5
    full_min = _eco(hold_s=30)[-1][0] / 60
    assert abs(store.get_profiles()["ECO"]["avg_duration"] / 60 - full_min) < 1


async def test_the_cut_short_marker_waits_while_a_build_stays_discarded(
    hass: HomeAssistant, setup_washdata_entry: Any
) -> None:
    """Discarded on every round: nothing written, the marker not saved."""
    store = await _eco_store(hass, setup_washdata_entry)
    with patch.object(store, "_rebuild_envelope_sync", side_effect=_type_flips(store, 99)):
        assert await store.async_rebuild_for_cut_short_rule(device_type="dishwasher") == 0
    assert store.cut_short_rebuild_pending()
    assert "ECO" not in store._data["envelopes"]  # noqa: SLF001


# ── devtools/program_mismatch_eval.py: alias twins by trace content ──────────


def _sub_watt_export() -> dict[str, Any]:
    """Cotton-0 and Synthetics-0 are two runs on one sample grid whose readings
    differ by 0.2 W: one trace at ``eval.trace_hash``'s 1 W, two by content."""
    doc = _pme_export()
    past = doc["data"]["past_cycles"]
    cotton = next(c for c in past if c["id"] == "Cotton-0")
    synth = next(c for c in past if c["id"] == "Synthetics-0")
    cotton["power_data"] = [[t, float(int(p)) + 0.1] for t, p in cotton["power_data"]]
    synth["power_data"] = [[t, p + 0.2] for t, p in cotton["power_data"]]
    synth["duration"] = cotton["duration"]
    return doc


def test_alias_twins_are_the_same_trace_content_not_a_sub_watt_neighbour(
    tmp_path: Path,
) -> None:
    (tmp_path / "synth").mkdir()
    (tmp_path / "synth" / "export.json").write_text(json.dumps(_sub_watt_export()))
    dev = pme.ev.load_device(tmp_path, "synth/export.json")
    assert pme.ev.alias_classes(dev)  # the coarse hash calls them one trace...
    assert pme.alias_classes(dev) == {}  # ...the content hash does not
    rows = pme.replay(tmp_path, jobs=1, log=lambda *a, **k: None)
    cotton = next(r for r in rows if r["label"] == "Cotton")
    assert cotton["twins"] == []
    assert "Synthetics" in pme._wrong_picks(cotton)  # noqa: SLF001
    # The same trace content under two names is still one run.
    doc = _sub_watt_export()
    past = doc["data"]["past_cycles"]
    synth = next(c for c in past if c["id"] == "Synthetics-0")
    synth["power_data"] = [list(p) for p in next(c for c in past if c["id"] == "Cotton-0")["power_data"]]
    (tmp_path / "synth" / "export.json").write_text(json.dumps(doc))
    dev = pme.ev.load_device(tmp_path, "synth/export.json")
    assert pme.alias_classes(dev) == {
        "Cotton": frozenset({"Cotton", "Synthetics"}),
        "Synthetics": frozenset({"Cotton", "Synthetics"}),
    }
