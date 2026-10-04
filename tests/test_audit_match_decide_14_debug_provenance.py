"""Audit MATCH-DECIDE-14: the match diagnostics describe ONE result.

``get_match_debug`` (the panel's Live Match Debug card) and the debug sensor's
attributes showed ``_last_match_confidence`` - the COMMITTED program's confidence,
moved only when the switching rules commit or confirm it - next to the newest
result's ambiguity flag and candidate rows. Mid-cycle the two describe different
matches: the card could read "92% Clear" over a table whose winner scored 41% and
was ambiguous. Confidence, ambiguity and candidates now come from the same result.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.profile_store import MatchResult, ProfileStore
from custom_components.ha_washdata.sensor import WasherDebugSensor


def _result() -> MatchResult:
    cands = [
        {"name": "Eco", "score": 0.41, "profile_duration": 7200.0, "metrics": {}},
        {"name": "Quick", "score": 0.39, "profile_duration": 1800.0, "metrics": {}},
    ]
    return MatchResult("Eco", 0.41, 7200.0, None, cands, True, 0.02, ranking=cands, query_duration_s=1500.0)


def _manager() -> SimpleNamespace:
    store = ProfileStore.__new__(ProfileStore)  # only get_match_candidates_summary is used
    return SimpleNamespace(
        _last_match_result=_result(),
        _last_match_confidence=0.92,     # the committed program's, from an earlier tick
        _last_match_ambiguous=False,
        profile_store=store,
    )


def test_ws_get_match_debug_reports_one_result():
    conn = MagicMock()
    with patch.object(ws_api, "_get_manager", return_value=_manager()):
        ws_api.ws_get_match_debug(MagicMock(), conn, {"id": 7, "entry_id": "e"})
    out = conn.send_result.call_args[0][1]
    assert out["confidence"] == 0.41
    assert out["ambiguous"] is True
    assert [c["profile_name"] for c in out["candidates"]] == ["Eco", "Quick"]
    assert out["candidates"][0]["confidence_pct"] == 41.0


def test_ws_get_match_debug_before_any_match():
    conn = MagicMock()
    mgr = _manager()
    mgr._last_match_result = None  # noqa: SLF001
    with patch.object(ws_api, "_get_manager", return_value=mgr):
        ws_api.ws_get_match_debug(MagicMock(), conn, {"id": 7, "entry_id": "e"})
    assert conn.send_result.call_args[0][1] == {"confidence": None, "ambiguous": False, "candidates": []}


def test_debug_sensor_confidence_is_the_listed_results():
    mgr = MagicMock()
    mgr._last_match_result = _result()  # noqa: SLF001
    mgr._last_match_confidence = 0.92  # noqa: SLF001
    mgr.sample_interval_stats = {}
    mgr.top_candidates = [{"name": "Eco", "score": 0.41}]
    entry = SimpleNamespace(entry_id="e", title="t")
    attrs = WasherDebugSensor(mgr, entry).extra_state_attributes
    assert attrs["match_confidence"] == 0.41
    # With no result there is nothing to disagree with: the committed figure stands.
    mgr._last_match_result = None  # noqa: SLF001
    assert WasherDebugSensor(mgr, entry).extra_state_attributes["match_confidence"] == 0.92
