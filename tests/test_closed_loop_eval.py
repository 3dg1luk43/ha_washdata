# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""devtools/closed_loop_eval.py: the pure parts (seeding, arms, tallies, compare).

The replay itself runs the real store and is measured, not unit-tested; these
pin what its numbers mean: which cycle seeds a programme, what each arm stores,
which devices are replayed (twin entries and store packages never), and how two
runs are paired.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "devtools"))

import closed_loop_eval as cl  # noqa: E402
import eval as E  # noqa: E402


def _cyc(name: str, day: int, hour: int = 8) -> dict:
    return {"profile_name": name, "start_time": f"2026-09-{day:02d}T{hour:02d}:00:00+00:00",
            "power_data": [[0, 1.0]] * 8}


def test_split_seeds_first_run_of_each_programme_by_start_time() -> None:
    a1, a2, b1, a3, b2 = _cyc("A", 3), _cyc("A", 5), _cyc("B", 4), _cyc("A", 9), _cyc("B", 7)
    seeds, seq = cl.split_seeds([a3, b2, a2, b1, a1])
    assert seeds == [a1, b1]
    assert seq == [a2, b2, a3]
    seeds, seq = cl.split_seeds([a3, b2, a2, b1, a1], per_programme=2)
    assert seeds == [a1, b1, a2, b2]
    assert seq == [a3]


def test_split_seeds_is_stable_on_equal_starts() -> None:
    x, y, z = _cyc("A", 3), _cyc("A", 3), _cyc("A", 3)
    seeds, seq = cl.split_seeds([x, y, z])
    assert seeds[0] is x and seq[0] is y and seq[1] is z


def test_arm_label() -> None:
    # oracle: always the truth, whatever the gate said
    assert cl.arm_label("oracle", None, "B", 0.1, 0.6, "A") == "A"
    # gate: the verdict, else nothing (the user never answers)
    assert cl.arm_label("gate", "B", "B", 0.9, 0.6, "A") == "B"
    assert cl.arm_label("gate", None, "B", 0.8, 0.6, "A") is None
    # review: a refused match at the learning floor is answered truthfully
    assert cl.arm_label("review", None, "B", 0.6, 0.6, "A") == "A"
    assert cl.arm_label("review", None, "B", 0.59, 0.6, "A") is None
    assert cl.arm_label("review", None, None, 0.9, 0.6, "A") is None
    assert cl.arm_label("review", "B", "B", 0.9, 0.6, "A") == "B"


def _row(ok: bool, lab: str | None, truth: str = "A", ok50: bool = False) -> dict:
    return {"ok": ok, "ok50": ok50, "lab": lab, "truth": truth, "reason": "ok"}


def test_tally_and_format() -> None:
    rows = [_row(True, "A", ok50=True), _row(False, "B"), _row(False, None), _row(True, None)]
    t = cl.tally(rows)
    assert t == {"n": 4, "ok": 2, "ok50": 1, "lab": 2, "wrong": 1}
    s = cl.fmt_tally(t)
    assert "top1 50.0%" in s and "top1@50 25.0%" in s
    assert "labelled 50.0%" in s and "wrong-labels 1 (50.0% of labels)" in s


def _dev(path: str, cycles: list[dict], dtype: str = "washing_machine") -> E.Device:
    return E.Device(path, path.split("/")[0], "export", "x", {"device_type": dtype}, {},
                    {"past_cycles": cycles})


def test_select_devices_skips_store_twins_small_and_single_programme() -> None:
    many = [_cyc("A" if i % 2 else "B", 1 + i) for i in range(12)]
    keep = _dev("u/washer.json", many)
    dryer = _dev("u/dryer.json", many, "dryer")
    devs = [
        keep, dryer,
        _dev("store/washing_machine/pkg.json", many),       # a download, not a chronology
        _dev("me/twin.json", many),                          # the same runs as another entry
        _dev("u/small.json", many[:9]),                      # under MIN_CYCLES
        _dev("u/one.json", [_cyc("A", 1 + i) for i in range(12)]),   # one programme
    ]
    twins = {"me/twin.json": {"kept": "u/washer.json"}}
    assert cl.select_devices(devs, twins) == [keep, dryer]
    assert cl.select_devices(devs, twins, {"dryer"}) == [dryer]
    assert cl.select_devices(devs, twins, None, "washer") == [keep]


def test_select_devices_counts_only_the_cycles_the_replay_uses() -> None:
    """Every trace under MIN_POINTS: the replay drops them all, so the device
    must not be selected and reported with empty rows."""
    short = [dict(_cyc("A" if i % 2 else "B", 1 + i), power_data=[[0, 1.0]] * 3) for i in range(12)]
    assert cl.select_devices([_dev("u/short.json", short)], {}) == []


def _doc(rows: list[dict], path: str = "u/washer.json") -> dict:
    return {"meta": {}, "devices": [{"path": path, "dev": "washing_machine", "seeds": 2,
                                     "arms": {"gate": rows}}]}


def test_compare_pairs_rows_and_counts_flips() -> None:
    before = _doc([_row(True, "A"), _row(False, None), _row(False, "B", ok50=True)])
    after = _doc([_row(False, "A"), _row(True, "A"), _row(True, "A")])
    lines: list[str] = []
    assert cl.compare_docs(before, after, out=lines.append) == 0
    dev_line = next(x for x in lines if x.startswith("u/washer.json"))
    assert "top1 +33.3pp (+2 -1 p=1.00)" in dev_line
    assert "@50 -33.3pp (+0 -1 p=1.00)" in dev_line
    assert "labelled +1" in dev_line and "wrong -1" in dev_line
    assert any(x.startswith("TOTAL gate:") for x in lines)


def test_compare_refuses_unpaired_sequences_and_reports_nothing_to_pair() -> None:
    lines: list[str] = []
    before = _doc([_row(True, "A", truth="A")])
    after = _doc([_row(True, "B", truth="B")])
    assert cl.compare_docs(before, after, out=lines.append) == 2
    assert any("sequences differ" in x for x in lines)
    assert cl.compare_docs(_doc([]), _doc([], path="other.json"), out=lines.append) == 2


def test_load_doc_reads_the_audit_legacy_list_and_hides_contributor_names(tmp_path: Path) -> None:
    rows = [_row(True, "A")]
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps([["user-Contributed/x/Some Name.json", 3, {"gate": rows}]]))
    doc = cl.load_doc(p)
    d = doc["devices"][0]
    assert d["path"] == E.public_key("user-Contributed/x/Some Name.json")
    assert "Some Name" not in d["path"] and d["seeds"] == 3 and d["arms"]["gate"] == rows
    new = tmp_path / "new.json"
    new.write_text(json.dumps(_doc(rows, path=d["path"])))
    assert cl.main(["--compare", str(p), str(new)]) == 0


# ---- the shipped arms: the real learning pass in the loop ----

def test_shipped_arms_are_known_and_not_in_the_default_run() -> None:
    assert set(cl.ANSWER_SHARE) == {"shipped", "shipped_p50"}
    assert set(cl.ALL_ARMS) == set(cl.ARMS) | set(cl.ANSWER_SHARE)
    assert "shipped" not in cl.ARMS     # the default run stays oracle,gate,review


def test_review_answer_is_the_truth_as_submit_feedback_arguments() -> None:
    assert cl.review_answer("A", "A") == (True, None)        # confirm
    assert cl.review_answer("B", "A") == (False, "A")       # correct to the truth
    assert cl.review_answer(None, "A") == (False, "A")


def test_user_answers_every_request_or_a_fixed_half() -> None:
    rng = cl.answer_rng(0, "u/washer.json")
    assert all(cl.user_answers("shipped", rng) for _ in range(50))
    a = [cl.user_answers("shipped_p50", cl.answer_rng(0, "u/washer.json")) for _ in range(3)]
    assert len(set(a)) == 1                  # the same seed and path, the same coin
    rng = cl.answer_rng(0, "u/washer.json")
    draws = [cl.user_answers("shipped_p50", rng) for _ in range(400)]
    assert 150 < sum(draws) < 250
    other = cl.answer_rng(1, "u/washer.json")
    assert [cl.user_answers("shipped_p50", other) for _ in range(400)] != draws


def test_tally_counts_questions_only_for_rows_that_carry_them() -> None:
    rows = [dict(_row(True, "A"), q=True, ans=True), dict(_row(False, None), q=True, ans=False),
            dict(_row(False, "B"), q=False, ans=False)]
    t = cl.tally(rows)
    assert (t["q"], t["ans"]) == (2, 1)
    assert "questions 2 answered 1" in cl.fmt_tally(t)
    assert "q" not in cl.tally([_row(True, "A")])


def test_compare_shipped_arm_reports_the_question_delta() -> None:
    def doc(rows: list[dict]) -> dict:
        return {"meta": {}, "devices": [{"path": "u/w.json", "dev": "washing_machine",
                                         "seeds": 1, "arms": {"shipped": rows}}]}
    before = doc([dict(_row(True, "A"), q=True, ans=True), dict(_row(True, "A"), q=True, ans=True)])
    after = doc([dict(_row(True, "A"), q=True, ans=True), dict(_row(True, "A"), q=False, ans=False)])
    lines: list[str] = []
    assert cl.compare_docs(before, after, out=lines.append) == 0
    assert any(x.startswith("TOTAL shipped:") and "questions -1" in x for x in lines)


class _FakeStore:
    def __init__(self) -> None:
        self.pending: dict = {}
        self.rebuilt: list[str] = []

    def get_profiles(self) -> dict:
        return {"A": {"avg_duration": 5400.0}}

    def get_pending_feedback(self) -> dict:
        return self.pending

    async def async_rebuild_envelope(self, name: str) -> None:
        self.rebuilt.append(name)


class _FakeLearning:
    """Records the manager's call; raises a request when told to."""

    def __init__(self, store: _FakeStore, raise_it: bool) -> None:
        self.store, self.raise_it, self.calls, self.answers = store, raise_it, [], []

    def process_cycle_end(self, cycle: dict, **kw) -> None:
        self.calls.append((dict(cycle), kw))
        if self.raise_it:
            self.store.pending[cycle["id"]] = {"detected_profile": kw["detected_profile"]}

    async def async_submit_cycle_feedback(self, cid, confirmed, corrected_profile=None) -> bool:
        self.answers.append((cid, confirmed, corrected_profile))
        return True


class _Res:
    best_profile = "B"


async def test_shipped_step_stores_the_gate_label_then_runs_the_pass_and_answers() -> None:
    st, data = _FakeStore(), {"past_cycles": []}
    lm = _FakeLearning(st, raise_it=True)
    # A hand label in the export must not reach the pass: it would skip the cycle.
    c2 = {"profile_name": "A", "label_source": "manual", "auto_labeled": True}
    lab, raised, answered = await cl._shipped_step(  # noqa: SLF001
        lm, st, data, c2, 7, _Res(), None, 0.7, 3600.0, "A", "shipped", cl.answer_rng(0, "x"))
    stored, kw = lm.calls[0]
    assert data["past_cycles"] == [c2] and stored["id"] == "closed-loop-7"
    assert "profile_name" not in stored and "label_source" not in stored and "auto_labeled" not in stored
    assert kw["detected_profile"] == "B" and kw["label_allowed"] is False
    assert kw["confidence"] == 0.7 and kw["predicted_duration"] == 3600.0   # B has no record
    assert (raised, answered) == (True, True)
    assert lm.answers == [("closed-loop-7", False, "A")]                    # corrected to the truth
    assert st.rebuilt == []                                                   # nothing labelled by the gate


async def test_shipped_step_gate_label_is_stored_and_unasked_cycles_are_not_answered() -> None:
    st, data = _FakeStore(), {"past_cycles": []}
    lm = _FakeLearning(st, raise_it=False)
    c2 = {"id": "c9", "profile_name": "A"}
    lab, raised, answered = await cl._shipped_step(  # noqa: SLF001
        lm, st, data, c2, 0, _Res(), "A", 0.8, 3600.0, "A", "shipped", cl.answer_rng(0, "x"))
    stored, kw = lm.calls[0]
    assert stored["profile_name"] == "A" and stored["label_source"] == "auto_match"
    assert kw["label_allowed"] is True and kw["predicted_duration"] == 5400.0
    assert st.rebuilt == ["A"] and (lab, raised, answered) == ("A", False, False)
    assert lm.answers == []
