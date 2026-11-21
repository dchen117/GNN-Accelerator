"""Timing simulation of MEGA on trained runs (M2).

Usage:
    python -m megasim.sim runs/hw/gcn_cora [runs/hw/... ] [--seed 0] [--json out.json]

For each run: export the program (S-07), partition with METIS and relabel (Q-27, S-08),
run the functional model on the relabelled program (it gives every layer's X̄ and must still
match the reference accuracy), then the timing model (megasim.timing). Prints cycles, unit
utilizations, DRAM traffic per stream and per sweep.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from . import functional, partition, timing
from .program import load_program


def simulate_run(run_dir: str, seed: int = 0, hw: timing.HW = timing.HW()) -> dict:
    t0 = time.time()
    prog, _ = load_program(run_dir, seed)
    part = partition.partition(prog)
    prog = partition.relabel(prog, part.order)
    t1 = time.time()
    func = functional.run(prog)
    t2 = time.time()
    rep = timing.simulate(prog, part.ptr, func, hw)
    t3 = time.time()
    out = rep.summary()
    pred = func[-1].logits_code.argmax(1)
    out.update(nodes=prog.graph.n, edges_with_self=int(prog.graph.deg.sum()),
               subgraph_cap=part.cap, metis_runs=part.metis_runs,
               test_acc=float((pred[prog.test_idx] == prog.y[prog.test_idx]).mean()),
               time_export_s=round(t1 - t0, 1), time_functional_s=round(t2 - t1, 1),
               time_timing_s=round(t3 - t2, 1))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", nargs="+")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    out = []
    for r in a.runs:
        res = simulate_run(r, a.seed)
        out.append(res)
        print(json.dumps(res, indent=1, default=float))
    print("\nSUMMARY  (cycles at 1 GHz; util = busy / total)")
    for r in out:
        u = " ".join(f"{k}={v:.2f}" for k, v in r["util"].items())
        print(f"  {r['run']:<20} K={r['subgraphs']:<4} {r['cycles']:>10,} cyc  {r['time_us']:8.1f} us"
              f"  DRAM {r['dram_MB_total']:7.2f} MB  {u}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
