"""Tables for the residual study (docs/beyond-mega/02-residual.md, 2026-11-25).

Collects every GCN results.json under the given directories and prints, per dataset:
  depth  × form (plain / pre / post) → FP32 test, QAT test, avg bits, FP32 best epoch,
  FP32 / QAT seconds per epoch.
Form names: plain = no residual; pre = ReLU(ÃXW + X); post = ReLU(ÃXW) + X.

    python -m mega_pt.res_table runs/res4 runs/resstudy runs/hw runs/ext
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict


def form(a: dict) -> str:
    if not a.get("residual", False):
        return "plain"
    return a.get("res_form", "pre")


def fmt(s: dict, k: str, scale: float = 1.0, nd: int = 2) -> str:
    if k not in s:
        return "–"
    m, sd = s[k]["mean"] * scale, s[k]["std"] * scale
    return f"{m:.{nd}f} ± {sd:.{nd}f}"


def collect(dirs: list[str]) -> dict:
    rows = defaultdict(dict)   # (dataset) -> {(L, form, lam, phase): summary}
    for d in dirs:
        for p in glob.glob(os.path.join(d, "**", "results.json"), recursive=True):
            r = json.load(open(p))
            a, s = r["args"], r["summary"]
            if a.get("model") != "gcn" or a.get("norm") != "fold":
                continue
            phase = "fp32" if a.get("fp32_only") else "qat"
            key = (a.get("layers", 2), form(a), a.get("lam", 0.0), phase)
            n = len(r["per_seed"])
            prev = rows[a["dataset"]].get(key)
            if prev is None or n > prev[1]:          # keep the run with the most seeds
                rows[a["dataset"]][key] = (s, n, os.path.dirname(p), a)
    return rows


def main(argv=None):
    dirs = (argv or sys.argv[1:]) or ["runs/res4", "runs/resstudy"]
    rows = collect(dirs)
    for ds in sorted(rows):
        print(f"\n## {ds}")
        print("| L | form | λ | phase | seeds | FP32 | QAT | avg bits | FP32 best ep | s/ep FP32 | s/ep QAT | run |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for key in sorted(rows[ds], key=lambda k: (k[0], ["plain", "pre", "post"].index(k[1]), k[2], k[3])):
            s, n, run, a = rows[ds][key]
            L, f, lam, phase = key
            spe_f = (f"{s['fp32_seconds']['mean'] / a['epochs_fp32']:.2f}"
                     if "fp32_seconds" in s else "–")
            spe_q = (f"{s['qat_seconds']['mean'] / a['epochs_qat']:.2f}"
                     if "qat_seconds" in s else "–")
            best = f"{s['fp32_best_epoch']['mean']:.0f}" if s.get("fp32_best_epoch", {}).get("mean", -1) >= 0 else "–"
            print(f"| {L} | {f} | {lam:g} | {phase} | {n} | {fmt(s, 'fp32_test', 100)} | "
                  f"{fmt(s, 'q_test', 100)} | {fmt(s, 'avg_bits_sparse')} | {best} | {spe_f} | {spe_q} | "
                  f"{os.path.relpath(run)} |")


if __name__ == "__main__":
    main()
