"""devtools/eval.py: leave-one-cycle-out plumbing and the compare gate.

Runs the real harness core (manager-built ProfileStore, async_rebuild_envelope,
async_match_profile) on a tiny synthetic corpus written to tmp_path, so it needs no
cycle_data/. The real-corpus numbers are the committed devtools/eval_baseline.json.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "devtools" / "eval.py"
_spec = importlib.util.spec_from_file_location("wd_eval_harness", _PATH)
ev = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ev   # dataclasses resolve their module through sys.modules
_spec.loader.exec_module(ev)

# minutes, watts per phase: four programmes with clearly different silhouettes.
_SHAPES = {
    "Quick": [(8, 2000), (20, 150), (2, 500)],
    "Long": [(20, 2200), (60, 200), (10, 650)],
    "Eco": [(30, 800), (25, 100), (5, 400)],
    "Single": [(10, 1500), (30, 300), (5, 700)],
}


def _cycle(name: str, j: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    scale = 1.0 + 0.03 * (j - 1)
    pts, t = [], 0.0
    for minutes, watts in _SHAPES[name]:
        for _ in range(int(minutes * 60 * scale / 30)):
            pts.append([round(t, 1), round(max(0.0, watts * (1 + 0.05 * rng.standard_normal())), 1)])
            t += 30.0
    pts.append([round(t, 1), 0.0])
    return {
        "id": f"{name}-{j}", "profile_name": name, "status": "completed",
        "start_time": f"2026-01-{j + 1:02d}T08:00:00+00:00", "duration": t, "power_data": pts,
    }


def _export(names_counts: dict[str, int], device_type: str = "washing_machine", seed: int = 0) -> dict:
    past, refs = [], []
    for name, n in names_counts.items():
        for j in range(n):
            seed += 1
            c = _cycle(name, j, seed)
            if name == "Eco" and j == 2:
                refs.append(c)          # a store reference cycle
            else:
                past.append(c)
    by_id = {c["id"]: c for c in past + refs}
    by_id["Quick-0"]["label_source"] = "manual"
    if "Long-0" in by_id:
        by_id["Long-0"]["label_source"] = "auto_match"
    profiles = {name: {"avg_duration": by_id[f"{name}-0"]["duration"], "sample_cycle_id": f"{name}-0"}
                for name in names_counts}
    return {
        "version": 14, "entry_data": {"device_type": device_type, "name": "Synth"}, "entry_options": {},
        "data": {"profiles": profiles, "past_cycles": past, "reference_cycles": refs, "envelopes": {}},
    }


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("corpus")
    main = _export({"Quick": 3, "Long": 3, "Eco": 3, "Single": 1})
    (root / "synth" / "washing_machine").mkdir(parents=True)
    (root / "synth" / "washing_machine" / "export.json").write_text(json.dumps(main))
    # A clone of the same entry (exported twice) and a one-programme device.
    (root / "synth" / "washing_machine" / "zz_clone.json").write_text(json.dumps(main))
    (root / "other").mkdir()
    (root / "other" / "one.json").write_text(json.dumps(_export({"Quick": 3}, "dishwasher", seed=100)))
    return root


def _quiet(*_a, **_k) -> None:
    return None


@pytest.fixture(scope="module")
def result(corpus, tmp_path_factory) -> dict:
    return ev.run_eval(corpus, mode="full", cuts=(1.0, 0.5), jobs=1,
                       cache_dir=tmp_path_factory.mktemp("cache"), log=_quiet)


def test_corpus_drops_clone_files_and_single_programme_devices(result):
    assert result["meta"]["dropped_clones"] == {
        "synth/washing_machine/zz_clone.json": "synth/washing_machine/export.json"}
    assert list(result["devices"]) == ["synth/washing_machine/export.json"]
    info = result["devices"]["synth/washing_machine/export.json"]
    # The store config is the one WashDataManager builds: shipped defaults, device energy mode.
    assert (info["min_ratio"], info["max_ratio"], info["energy_mode"]) == (0.10, 1.8, "integrated")
    assert (info["auto_label_conf"], info["learning_conf"]) == (0.9, 0.6)


def test_every_scorable_fold_is_matched_once_per_cut(result):
    rows = result["folds"]
    keys = [(r["path"], r["id"], r["cut"]) for r in rows]
    assert len(keys) == len(set(keys)) == 9 * 2      # the singleton programme has no right answer
    assert {r["label"] for r in rows} == {"Quick", "Long", "Eco"}
    assert all(r["top1"] for r in rows)
    m = result["metrics"]["ALL"]
    assert m["top1@1.0"] == 100.0 and m["n@1.0"] == 9
    assert m["no_candidate@1.0"] == 0.0
    by_prov = {s: v["n@1.0"] for s, v in result["metrics"].items() if s.startswith("prov=")}
    assert by_prov == {"prov=manual": 1, "prov=auto": 1, "prov=reference": 1, "prov=unknown": 6}


def test_fold_holds_the_cycle_out_of_its_own_envelope(corpus):
    dev = ev.load_device(corpus, "synth/washing_machine/export.json")
    base = ev.run_coro(ev.rebuild_base(dev, {}))
    full = ev.base_data(dev)
    full["profiles"], full["envelopes"] = base["profiles"], base["envelopes"]
    sample = full["profiles"]["Quick"]["sample_cycle_id"]
    held = next(c for c, _ in dev.cycles if c["id"] == sample)   # the hardest case: the template
    fd = ev.fold_data(full, held)
    st, _ = ev.fresh_store(dev, fd, {})
    ev.run_coro(st.async_rebuild_envelope("Quick"))
    assert all(c is not held for k, _ in ev.LISTS for c in fd[k])
    assert fd["envelopes"]["Quick"]["cycle_count"] == 2
    assert fd["profiles"]["Quick"]["sample_cycle_id"] not in (None, sample)
    # The shared base is untouched, so the next fold starts from the full store.
    assert full["envelopes"]["Quick"]["cycle_count"] == 3
    assert full["profiles"]["Quick"]["sample_cycle_id"] == sample
    assert any(c is held for c in full["past_cycles"])


def test_cached_rerun_is_identical_and_writes_nothing_else(corpus, result, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "cache"
    cold = ev.run_eval(corpus, mode="full", cuts=(1.0, 0.5), jobs=1, cache_dir=cache, log=_quiet)
    warm = ev.run_eval(corpus, mode="full", cuts=(1.0, 0.5), jobs=1, cache_dir=cache, log=_quiet)
    assert cold["folds"] == warm["folds"] == result["folds"]
    assert cold["metrics"] == warm["metrics"] == result["metrics"]
    # The stubbed Store saved nothing (no ./MagicMock/ tree); only the cache dir exists.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cache"]
    out = tmp_path / "cache" / "r.json"
    ev.write_result(cold, out)
    assert json.loads(out.read_text())["folds"] == cold["folds"]


def test_config_override_reaches_store_and_constants(corpus):
    from custom_components.ha_washdata import const, manager

    before = const.MATCH_LABEL_MIN_MARGIN
    doc = ev.run_eval(corpus, mode="fast", cuts=(1.0,), jobs=1, use_cache=False, log=_quiet, override={
        "options": {"profile_match_max_duration_ratio": 1.23},
        "const": {"MATCH_LABEL_MIN_MARGIN": 5.0},
    })
    assert doc["devices"]["synth/washing_machine/export.json"]["max_ratio"] == 1.23
    assert doc["metrics"]["ALL"]["label_coverage@1.0"] == 0.0
    assert doc["metrics"]["ALL"]["learn_coverage@1.0"] == 0.0
    assert doc["meta"]["override"]["const"] == {"MATCH_LABEL_MIN_MARGIN": 5.0}
    assert const.MATCH_LABEL_MIN_MARGIN == before
    assert manager.MATCH_LABEL_MIN_MARGIN == before   # every rebinding was undone
    with pytest.raises(SystemExit):
        with ev.const_overrides({"NOT_A_CONSTANT": 1}):
            pass


def _doc(rows: list[dict]) -> dict:
    return {"meta": {"rev": "x", "code_sha": "y", "corpus_manifest": "m", "mode": "fast",
                     "harness_version": 1}, "tolerances": ev.DEFAULT_TOLERANCES, "folds": rows}


def _rows(n: int = 60) -> list[dict]:
    return [{"path": f"u/dev{i % 6}.json", "id": f"c{i}", "cut": cut, "dev": "washing_machine",
             "ok": i % 5 != 0, "gok": i % 5 != 0, "rank": 1 if i % 5 else 2, "top1": "A",
             "lok": i % 2 == 0, "lokl": True, "mg": 0.1 + (i % 7) / 50, "conf": 0.8, "g": None, "gin": None}
            for i in range(n) for cut in (1.0, 0.5)]


def test_compare_passes_identical_and_fails_injected_regression():
    lines: list[str] = []
    base = _doc(_rows())
    assert ev.compare(base, copy.deepcopy(base), out=lines.append) == 0
    worse = copy.deepcopy(base)
    flipped = 0
    for r in worse["folds"]:
        if r["cut"] == 1.0 and r["ok"] and flipped < 3:
            r["ok"] = r["gok"] = False
            flipped += 1
    lines.clear()
    assert ev.compare(base, worse, out=lines.append) == 1
    assert any("top1@1.0" in ln and "FAIL" in ln for ln in lines)
    # A looser tolerance file accepts the same drop; nothing paired is exit 2.
    assert ev.compare(base, worse, tol={"top1@1.0": 10.0}, out=lines.append) == 0
    assert ev.compare(base, _doc([]), out=lines.append) == 2


def test_contributor_file_names_never_reach_the_result():
    key = ev.public_key("user-Contributed/ha_washdata/Dishwasher/config_entry-x - Jane Doe.json")
    assert key.startswith("user-Contributed/") and "Jane" not in key
    assert ev.public_key("store/dishwasher/smeg__dwa6d16x.json") == "store/dishwasher/smeg__dwa6d16x.json"


def test_paired_statistics():
    assert ev.mcnemar_p(4, 0) == 0.125
    assert ev.mcnemar_p(25, 1) < 1e-4
    assert ev.mcnemar_p(0, 0) == 1.0
    assert ev.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert ev.auc([0.5], [0.5]) == 0.5
    lo, hi = ev.cluster_bootstrap(["a", "a", "b", "b"], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1])
    assert lo == hi == 100.0


def test_cli_compare_exit_codes(tmp_path):
    base = _doc(_rows())
    worse = copy.deepcopy(base)
    for r in worse["folds"]:
        r["ok"] = False
    (tmp_path / "b.json").write_text(json.dumps(base))
    (tmp_path / "n.json").write_text(json.dumps(worse))
    assert ev.main(["compare", str(tmp_path / "b.json"), str(tmp_path / "b.json")]) == 0
    assert ev.main(["compare", str(tmp_path / "b.json"), str(tmp_path / "n.json")]) == 1
    assert ev.main(["run", "--corpus", str(tmp_path / "missing")]) == 2
    assert sorted(os.listdir(tmp_path)) == ["b.json", "n.json"]
