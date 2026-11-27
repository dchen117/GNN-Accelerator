"""Tables for the residual study (docs/beyond-mega/02-residual.md, 2026-11-25 / 26).

Collects every results.json (fold norm) under the given directories and prints, per
(dataset, model):
  depth × form (plain / pre / post) × λ × phase → FP32 test, QAT test, bits/nz (average bits
  of a non-zero hidden-layer value, metrics.avg_bits_nz), hidden density, the old sparse
  average (bits/nz × density), FP32 best epoch, FP32 / QAT seconds per epoch.
Form names: plain = no residual; pre = ReLU(ÃXW + X); post = ReLU(ÃXW) + X.
Values are % accuracy, or % ROC-AUC for Minesweeper / Tolokers / Questions (Graph.metric).

    python -m mega_pt.res_table runs/res4 runs/resstudy runs/res5
    python -m mega_pt.res_table --curves runs/res5      # epochs-to-threshold, plain vs residual
    python -m mega_pt.res_table --minbits runs/res5q runs/minbits   # min bits within 1 pt of FP32

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


def bits_nz(r: dict) -> float | None:
    """avg_bits_nz of one per-seed row; recomputed for results.json files written before it
    existed: avg_bits_sparse = Σ nnz·b / Σ N·F and all hidden inputs have the same F, so
    avg_bits_nz = avg_bits_sparse / mean_l(L<l>_density) over hidden inputs l ≥ 1 (checked
    against quant_report on saved models, 2026-11-27)."""
    if "avg_bits_nz" in r:
        return r["avg_bits_nz"]
    dens = [r[k] for k in sorted(r) if k.startswith("L") and k.endswith("_density") and k != "L0_density"]
    if "avg_bits_sparse" not in r or not dens:
        return None
    return r["avg_bits_sparse"] / statistics.fmean(dens)


def hid_density(r: dict) -> float | None:
    dens = [r[k] for k in sorted(r) if k.startswith("L") and k.endswith("_density") and k != "L0_density"]
    return statistics.fmean(dens) if dens else None


def _ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": statistics.fmean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}


def collect(dirs: list[str]) -> dict:
    """{(dataset, model): {(L, form, lam, phase): (summary, n_seeds, run_dir, args, per_seed)}}"""
    rows = defaultdict(dict)
    for d in dirs:
        for p in glob.glob(os.path.join(d, "**", "results.json"), recursive=True):
            r = json.load(open(p))
            a, s = r["args"], dict(r["summary"])
            for k, fn in (("avg_bits_nz", bits_nz), ("hid_density", hid_density)):
                m = _ms([fn(row) for row in r["per_seed"]])
                if m:
                    s[k] = m
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
        print("| L | form | λ | phase | seeds | FP32 | QAT | bits/nz | density | old sparse avg | FP32 best ep | s/ep FP32 | s/ep QAT | run |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for key in sorted(rows[(ds, model)], key=_order):
            s, n, run, a, _ = rows[(ds, model)][key]
            L, f, lam, phase = key
            spe_f = (f"{s['fp32_seconds']['mean'] / a['epochs_fp32']:.2f}"
                     if "fp32_seconds" in s else "–")
            spe_q = (f"{s['qat_seconds']['mean'] / a['epochs_qat']:.2f}"
                     if "qat_seconds" in s else "–")
            best = f"{s['fp32_best_epoch']['mean']:.0f}" if s.get("fp32_best_epoch", {}).get("mean", -1) >= 0 else "–"
            print(f"| {L} | {f} | {lam:g} | {phase} | {n} | {fmt(s, 'fp32_test', 100)} | "
                  f"{fmt(s, 'q_test', 100)} | {fmt(s, 'avg_bits_nz')} | {fmt(s, 'hid_density', 100, 0)}% | "
                  f"{fmt(s, 'avg_bits_sparse')} | {best} | {spe_f} | {spe_q} | "
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


def print_minbits(dirs, tol=1.0):
    """Minimum-bits study (2026-11-27). For each (dataset, model, L, form): QAT @ bits/nz at
    every memory target found (--target-dense-bits; 'floor' = the 2-bit-floor recipe runs,
    e.g. runs/res5q at λ = 1e-7), and the lowest bits/nz whose mean QAT is within `tol`
    points of the mean FP32 ('–' if none)."""
    grp = defaultdict(dict)
    fp32 = {}
    for d in dirs:
        for p in glob.glob(os.path.join(d, "**", "results.json"), recursive=True):
            r = json.load(open(p))
            a, s = r["args"], r["summary"]
            if "q_test" not in s or a.get("norm") != "fold":
                continue
            tb = a.get("target_dense_bits")
            reach = r["m_target_kb"] > 0 and tb is not None and tb > 2.0
            level = tb if reach else "floor"
            k = (a["dataset"], a.get("model", "gcn"), a.get("layers", 2), form(a))
            nz = _ms([bits_nz(x) for x in r["per_seed"]])
            grp[k][level] = (100 * s["q_test"]["mean"], 100 * s["q_test"]["std"], nz["mean"])
            fp32[k] = 100 * s["fp32_test"]["mean"]
    levels = sorted({lv for g in grp.values() for lv in g if lv != "floor"})
    cols = ["floor"] + levels
    print(f"QAT @ bits/nz per memory target (dense bits); 'min' = lowest bits/nz within {tol} pt of FP32")
    print("| dataset | model | L | form | FP32 | " + " | ".join(str(c) for c in cols) + " | min bits/nz | QAT − FP32 there |")
    print("|---|---|---|---|---|" + "---|" * len(cols) + "---|---|")
    for k in sorted(grp, key=lambda k: (k[0], k[1], k[2], ["plain", "pre", "post"].index(k[3]))):
        cells, ok = [], []
        for c in cols:
            v = grp[k].get(c)
            if v is None:
                cells.append("–")
                continue
            cells.append(f"{v[0]:.2f} ± {v[1]:.2f} @ {v[2]:.2f}")
            if v[0] >= fp32[k] - tol:
                ok.append((v[2], v[0] - fp32[k]))
        best = min(ok) if ok else None
        tail = f"**{best[0]:.2f}** | {best[1]:+.2f}" if best else "– | –"
        print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {fp32[k]:.2f} | " + " | ".join(cells) + f" | {tail} |")


def main(argv=None):
    argv = list(argv if argv is not None else sys.argv[1:])
    curves = "--curves" in argv
    dirs = [a for a in argv if not a.startswith("--")] or ["runs/res4", "runs/resstudy"]
    if "--minbits" in argv:
        return print_minbits(dirs)
    rows = collect(dirs)
    print_curves(rows) if curves else print_tables(rows)


if __name__ == "__main__":
    main()
