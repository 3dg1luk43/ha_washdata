"""Audit MATCH-EVAL-08: devtools/eval.py flags within-file alias twins.

The corpus loader dropped clone FILES only. Inside one store package the same trace
can sit under two programme names ("Coton 60°" / "Katoen 60°": 17 twin groups in 6
packages), and a fold whose label has such a twin counts the twin as a matcher
error (~3pp of pooled top-1 in the audit). Nothing is dropped and top1 is unchanged:
rows carry ``aok`` / ``tw``, ``alias_top1`` is reported beside top1 (summary and
compare) and ``meta.alias_twins`` names the labels.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "devtools" / "eval.py"
_spec = importlib.util.spec_from_file_location("wd_eval_alias", _PATH)
ev = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ev
_spec.loader.exec_module(ev)


def _cycle(cid: str, name: str, heat: float, k: int) -> dict:
    rng = np.random.default_rng(k)
    pts, t = [], 0.0
    for minutes, watts in ((10, heat), (30, 200), (5, 600)):
        for _ in range(int(minutes * 2)):
            pts.append([t, round(watts * (1 + 0.03 * rng.standard_normal()), 1)])
            t += 30.0
    pts.append([t, 0.0])
    return {"id": cid, "profile_name": name, "status": "completed", "duration": t,
            "start_time": f"2026-01-{k + 1:02d}T08:00:00+00:00", "power_data": pts}


def _export() -> dict:
    a = [_cycle(f"a{k}", "Coton 60", 2000.0, k) for k in range(3)]
    twin = copy.deepcopy(a[0])
    twin.update(id="k0", profile_name="Katoen 60")            # same trace, other name
    k1 = _cycle("k1", "Katoen 60", 2000.0, 7)
    c = [_cycle(f"c{k}", "Quick", 900.0, 20 + k) for k in range(3)]
    past = a + [twin, k1] + c
    profiles = {n: {"avg_duration": 2700.0, "sample_cycle_id": s}
                for n, s in (("Coton 60", "a0"), ("Katoen 60", "k0"), ("Quick", "c0"))}
    return {"version": 14, "entry_data": {"device_type": "washing_machine", "name": "S"},
            "entry_options": {}, "data": {"profiles": profiles, "past_cycles": past, "envelopes": {}}}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("alias_corpus")
    (root / "store").mkdir()
    (root / "store" / "pkg.json").write_text(json.dumps(_export()))
    return root


def test_alias_classes_are_per_file_and_transitive(corpus):
    dev = ev.load_device(corpus, "store/pkg.json")
    cls = ev.alias_classes(dev)
    assert cls == {"Coton 60": frozenset({"Coton 60", "Katoen 60"}),
                   "Katoen 60": frozenset({"Coton 60", "Katoen 60"})}
    # Transitive: A~B and B~C put all three in one class.
    dev.hashes = ["h1", "h1", "h2", "h2"]
    dev.cycles = [({"profile_name": n}, "past") for n in ("A", "B", "B", "C")]
    assert ev.alias_classes(dev)["A"] == frozenset({"A", "B", "C"})


def test_rows_are_flagged_and_top1_is_untouched():
    aliases = {"p": {"A": frozenset({"A", "B"}), "B": frozenset({"A", "B"})}}
    rows = [
        {"path": "p", "label": "A", "top1": "B", "ok": False},
        {"path": "p", "label": "A", "top1": "C", "ok": False},
        {"path": "p", "label": "C", "top1": "C", "ok": True},
        {"path": "q", "label": "A", "top1": "B", "ok": False},   # another file: no twin there
    ]
    ev.mark_alias_twins(rows, aliases)
    assert [(r["aok"], r["tw"]) for r in rows] == [(True, True), (False, True), (True, False), (False, False)]
    assert all(r["ok"] is o for r, o in zip(rows, (False, False, True, False)))


@pytest.mark.slow
def test_run_reports_alias_top1_and_the_twins(corpus, tmp_path):
    doc = ev.run_eval(corpus, mode="full", cuts=(1.0,), jobs=1, cache_dir=tmp_path, log=lambda *a, **k: None)
    assert doc["meta"]["alias_twins"] == {"store/pkg.json": [["Coton 60", "Katoen 60"]]}
    m = doc["metrics"]["ALL"]
    assert m["twin_folds@1.0"] == sum(1 for r in doc["folds"] if r["label"] in ("Coton 60", "Katoen 60"))
    assert m["alias_top1@1.0"] >= m["top1@1.0"]
    assert all(r["aok"] == (r["ok"] or (r["tw"] and r["top1"] in ("Coton 60", "Katoen 60")))
               for r in doc["folds"])
    lines: list[str] = []
    assert ev.compare(doc, doc, out=lines.append) == 0
    assert any("alias_top1" in ln for ln in lines)
