"""Tables for the residual study (docs/beyond-mega/02-residual.md, 2026-11-25 / 26).

Collects every results.json (fold norm) under the given directories and prints, per
(dataset, model):
  depth × form (plain / pre / post) × λ × phase → FP32 test, QAT test, avg bits,
  FP32 best epoch, FP32 / QAT seconds per epoch.
Form names: plain = no residual; pre = ReLU(ÃXW + X); post = ReLU(ÃXW) + X.
Values are % accuracy, or % ROC-AUC for Minesweeper / Tolokers / Questions (Graph.metric).

    python -m mega_pt.res_table runs/res4 runs/resstudy runs/res5
    python -m mega_pt.res_table --curves runs/res5      # epochs-to-threshold, plain vs residual

--curves (needs fp32_val_curve, recorded since 2026-11-26): for each (dataset, model, L) with
both a plain and a residual run, the threshold is thr = 0.99 × min(best val of the two), a
level both reach. Reported: mean over seeds of the first epoch with val ≥ thr, per form, and
the same for the 2-layer plain run's best val (thr2 = 0.99 × that), if reached at all.
"""

from __future__ import annotations

import glob
import json
import os
import statistics
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
    """{(dataset, model): {(L, form, lam, phase): (summary, n_seeds, run_dir, args, per_seed)}}"""
    rows = defaultdict(dict)
    for d in dirs:
        for p in glob.glob(os.path.join(d, "**", "results.json"), recursive=True):
            r = json.load(open(p))
            a, s = r["args"], r["summary"]
            if a.get("norm") != "fold":
                continue
            phase = "fp32" if a.get("fp32_only") else "qat"
            key = (a.get("layers", 2), form(a), a.get("lam", 0.0), phase)
            n = len(r["per_seed"])
            grp = rows[(a["dataset"], a.get("model", "gcn"))]
            prev = grp.get(key)
            if prev is None or n > prev[1]:          # keep the run with the most seeds
                grp[key] = (s, n, os.path.dirname(p), a, r["per_seed"])
    return rows


def _order(k):
    return (k[0], ["plain", "pre", "post"].index(k[1]), k[2], k[3])


def print_tables(rows):
    for ds, model in sorted(rows):
        print(f"\n## {ds} / {model}")
        print("| L | form | λ | phase | seeds | FP32 | QAT | avg bits | FP32 best ep | s/ep FP32 | s/ep QAT | run |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for key in sorted(rows[(ds, model)], key=_order):
            s, n, run, a, _ = rows[(ds, model)][key]
            L, f, lam, phase = key
            spe_f = (f"{s['fp32_seconds']['mean'] / a['epochs_fp32']:.2f}"
                     if "fp32_seconds" in s else "–")
            spe_q = (f"{s['qat_seconds']['mean'] / a['epochs_qat']:.2f}"
                     if "qat_seconds" in s else "–")
            best = f"{s['fp32_best_epoch']['mean']:.0f}" if s.get("fp32_best_epoch", {}).get("mean", -1) >= 0 else "–"
            print(f"| {L} | {f} | {lam:g} | {phase} | {n} | {fmt(s, 'fp32_test', 100)} | "
                  f"{fmt(s, 'q_test', 100)} | {fmt(s, 'avg_bits_sparse')} | {best} | {spe_f} | {spe_q} | "
                  f"{os.path.relpath(run)} |")


def _first_reach(curve, thr):
    for ep, v in enumerate(curve):
        if v >= thr:
            return ep
    return None


def _mean_reach(per_seed, thr):
    """Mean first epoch over seeds; '–' if any seed never reaches thr ('n/k' reached)."""
    eps = [_first_reach(r["fp32_val_curve"], thr) for r in per_seed]
    got = [e for e in eps if e is not None]
    if not got:
        return "never"
    txt = f"{statistics.fmean(got):.0f}"
    return txt if len(got) == len(eps) else f"{txt} ({len(got)}/{len(eps)})"


def print_curves(rows):
    print("| dataset | model | L | form | best val | ep → 0.99·min(best) | ep → 0.99·best(L=2) |")
    print("|---|---|---|---|---|---|---|")
    for ds, model in sorted(rows):
        grp = rows[(ds, model)]
        with_curve = {k: v for k, v in grp.items()
                      if k[3] == "fp32" and all("fp32_val_curve" in r for r in v[4])}
        l2 = with_curve.get((2, "plain", 0.0, "fp32"))
        thr2 = 0.99 * statistics.fmean(max(r["fp32_val_curve"]) for r in l2[4]) if l2 else None
        if l2:                                       # reference: the 2-layer model itself
            b2 = statistics.fmean(max(r["fp32_val_curve"]) for r in l2[4])
            print(f"| {ds} | {model} | 2 | plain | {100 * b2:.2f} | – | {_mean_reach(l2[4], thr2)} |")
        for L in sorted({k[0] for k in with_curve if k[0] > 2}):
            forms = {k[1]: v for k, v in with_curve.items() if k[0] == L}
            if "plain" not in forms or len(forms) < 2:
                continue
            best = {f: statistics.fmean(max(r["fp32_val_curve"]) for r in v[4]) for f, v in forms.items()}
            thr = 0.99 * min(best.values())
            for f in sorted(forms, key=["plain", "pre", "post"].index):
                ps = forms[f][4]
                r2 = _mean_reach(ps, thr2) if thr2 else "–"
                print(f"| {ds} | {model} | {L} | {f} | {100 * best[f]:.2f} | {_mean_reach(ps, thr)} | {r2} |")


def main(argv=None):
    argv = list(argv if argv is not None else sys.argv[1:])
    curves = "--curves" in argv
    dirs = [a for a in argv if a != "--curves"] or ["runs/res4", "runs/resstudy"]
    rows = collect(dirs)
    print_curves(rows) if curves else print_tables(rows)


if __name__ == "__main__":
    main()
