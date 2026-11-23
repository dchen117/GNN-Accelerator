"""MEGA vs SGCN, as in MEGA's Fig.14/16/17, next to the paper's ratios.

    python -m sgcnsim.compare runs/hw/megasim_m3.json runs/hw/sgcn.json

MEGA's figures normalize everything to HyGCN, so MEGA's advantage over SGCN is the ratio of
the two bars. PAPER holds both bars per GCN dataset (read from 400-dpi renders of Fig.14/16/17,
p.11; the SGCN series was identified by recomputing the stated geomeans, see
docs/sgcn/00-design.md §Targets). MEGA compares with SGCN on GCN only.
"""

from __future__ import annotations

import argparse
import json
import math

# dataset → (MEGA, SGCN) bars, normalized to HyGCN
PAPER = {
    "speedup": {"cora": (43.0, 15.4), "citeseer": (58.7, 25.9), "pubmed": (29.5, 12.9),
                "nell": (52.7, 20.0), "reddit": (78.5, 5.1)},
    "dram": {"cora": (98.0, 34.5), "citeseer": (188.0, 25.3), "pubmed": (254.6, 38.4),
             "nell": (239.5, 10.2), "reddit": (108.8, 16.8)},
    "energy": {"cora": (46.6, 16.2), "citeseer": (57.9, 24.5), "pubmed": (114.1, 15.6),
               "nell": (86.1, 13.3), "reddit": (103.2, 17.9)},
}
STATED_GEOMEAN = {"speedup": 3.6, "dram": 7.3, "energy": 4.5}     # p.10–11 text


def geo(v):
    v = list(v)
    return math.exp(sum(map(math.log, v)) / len(v)) if v else float("nan")


def ratios(mega: dict, sgcn: dict) -> dict:
    return {"speedup": sgcn["cycles"] / mega["cycles"],
            "dram": sgcn["dram_MB_total"] / mega["dram_MB_total"],
            "energy": sgcn["energy"]["total_uJ"] / mega["energy"]["total_uJ"]}


def table(mega_runs: dict, sgcn_runs: dict) -> str:
    rows = ["| dataset | MEGA µs | SGCN µs | speedup (paper) | MEGA MB | SGCN MB | DRAM red. (paper) "
            "| MEGA µJ | SGCN µJ | energy sav. (paper) |", "|" + "---|" * 10]
    ours = {k: [] for k in PAPER}
    paper_same = {k: [] for k in PAPER}
    for name, s in sgcn_runs.items():
        if name not in mega_runs:
            continue
        m = mega_runs[name]
        ds = name.split("_", 1)[1]
        r = ratios(m, s)
        p = {k: PAPER[k][ds][0] / PAPER[k][ds][1] for k in PAPER} if ds in PAPER["speedup"] else {}
        for k in PAPER:
            ours[k].append(r[k])
            if p:
                paper_same[k].append(p[k])
        f = lambda k: f"{r[k]:.2f}× ({p[k]:.2f}×)" if p else f"{r[k]:.2f}×"
        rows.append(f"| {ds} | {m['time_us']:.1f} | {s['time_us']:.1f} | {f('speedup')} | "
                    f"{m['dram_MB_total']:.2f} | {s['dram_MB_total']:.2f} | {f('dram')} | "
                    f"{m['energy']['total_uJ']:.1f} | {s['energy']['total_uJ']:.1f} | {f('energy')} |")
    paper_all = {k: geo(a / b for a, b in PAPER[k].values()) for k in PAPER}
    rows.append("")
    rows.append("geomean, ours: " + ", ".join(f"{k} {geo(ours[k]):.2f}×" for k in PAPER))
    rows.append("geomean, paper on the same datasets: " +
                ", ".join(f"{k} {geo(paper_same[k]):.2f}×" for k in PAPER))
    rows.append("geomean, paper on all 5 GCN datasets: " +
                ", ".join(f"{k} {paper_all[k]:.2f}× (stated {STATED_GEOMEAN[k]}×)" for k in PAPER))
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mega")
    ap.add_argument("sgcn")
    a = ap.parse_args(argv)
    load = lambda p: {r["run"]: r for r in json.load(open(p))}
    print(table(load(a.mega), load(a.sgcn)))


if __name__ == "__main__":
    main()
