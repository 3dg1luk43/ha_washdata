#!/usr/bin/env python3
"""Accuracy of the live projected-energy figure, leave-one-out on the real corpus.

Register: audit 2026-10-02 PROGRESS-04. ``progress.projected_energy`` used to
divide the energy so far by the TIME fraction; heaters front-load energy, so it
read washers 1.89x high at 25%. It now divides by the matched profile's
cumulative-energy share at that progress (``progress.envelope_energy_fraction``).

Per scorable cycle (eval.py's corpus, de-cloning and production store config):
rebuild its programme's envelope WITHOUT it, then at each elapsed fraction f
compare ``E(f) / f`` (old) and ``E(f) / F(f)`` (new) against the cycle's own
integrated total. Progress is the oracle time fraction, so this isolates the
divisor from the progress estimate. Reports median projected/actual and MAPE.

    python3 devtools/energy_projection_eval.py [--corpus cycle_data] [--jobs 6]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval as E  # noqa: E402

FRACTIONS = (0.03, 0.10, 0.25, 0.50, 0.75, 0.90)


async def _device_rows(dev: E.Device) -> list[tuple[str, float, float, float]]:
    from custom_components.ha_washdata import progress as P  # noqa: PLC0415
    from custom_components.ha_washdata.signal_processing import (  # noqa: PLC0415
        energy_gap_threshold_s,
        integrate_wh,
    )

    base = await E.rebuild_base(dev, {})
    full = E.base_data(dev)
    full["profiles"], full["envelopes"] = base["profiles"], base["envelopes"]
    rows = []
    for i in E.scorable(dev):
        cyc, _ev = dev.cycles[i]
        label = cyc["profile_name"]
        pts = E.query_points(cyc)
        t = np.array([p[0] for p in pts], dtype=float)
        w = np.array([p[1] for p in pts], dtype=float)
        t = t - t[0]
        if t[-1] < 600:
            continue
        gap = energy_gap_threshold_s(t)
        total = integrate_wh(t, w, max_gap_s=gap)
        if total <= 1.0:
            continue
        st, mgr = E.fresh_store(dev, E.fold_data(full, cyc), {})
        await st.async_rebuild_envelope(label)
        for f in FRACTIONS:
            m = t <= f * t[-1]
            if m.sum() < 2:
                continue
            so_far = integrate_wh(t[m], w[m], max_gap_s=gap)
            if so_far <= 0:
                continue
            frac = P.envelope_energy_fraction(st, label, f * 100.0)
            old = max(so_far / f, so_far) / total
            new = max(so_far / frac, so_far) / total if frac is not None else old
            rows.append((mgr.device_type, f, old, new))
    return rows


def _job(path: str) -> list[tuple[str, float, float, float]]:
    E._quiet_logs()  # noqa: SLF001
    dev = E.load_device(Path(_ROOT[0]), path)
    return E.run_coro(_device_rows(dev)) if dev is not None else []


_ROOT: list[str] = []


def _init(root: str) -> None:
    _ROOT[:] = [root]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", default=str(E.REPO / "cycle_data"))
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    root = Path(args.corpus)
    devices, _dropped = E.load_corpus(root)
    if not devices:
        print("no corpus")
        return 2
    rows: list[tuple[str, float, float, float]] = []
    with ProcessPoolExecutor(args.jobs, initializer=_init, initargs=(str(root),)) as ex:
        for part in ex.map(_job, [d.path for d in devices]):
            rows.extend(part)
    by: dict[tuple[str, float], list[tuple[float, float]]] = defaultdict(list)
    for dev, f, old, new in rows:
        by[(dev, f)].append((old, new))
        by[("ALL", f)].append((old, new))
    print(f"{'device':16s} {'at':>4s} {'n':>5s}  {'old median':>10s} {'old MAPE':>9s}  "
          f"{'new median':>10s} {'new MAPE':>9s}")
    for dev in sorted({k[0] for k in by}, key=lambda d: (d != "ALL", d)):
        for f in FRACTIONS:
            v = by.get((dev, f))
            if not v:
                continue
            a = np.array(v)
            print(f"{dev:16s} {int(f * 100):3d}% {len(a):5d}  {np.median(a[:, 0]):10.2f} "
                  f"{np.mean(np.abs(a[:, 0] - 1)) * 100:8.1f}%  {np.median(a[:, 1]):10.2f} "
                  f"{np.mean(np.abs(a[:, 1] - 1)) * 100:8.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
