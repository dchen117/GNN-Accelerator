"""Timing and energy simulation of SGCN (MEGA Tab.V configuration) on trained GCN runs.

    python -m sgcnsim.sim runs/hw/gcn_cora [...] [--dram ramulator|uncontended] [--json out.json]

Per run: FP32 workload (workload.py) → access plan + cache simulation (timing.build_plan) →
schedule with Ramulator HBM1 timing (the same fixed point and DRAM model as MEGA, S-13) →
energy (energy.py). The JSON has the same keys as `python -m megasim.sim`, so

    python -m megasim.report runs/hw/megasim_m3.json --baseline runs/hw/sgcn.json

prints MEGA's speedup / DRAM reduction / energy saving over SGCN (Fig.14/16/17 style).
"""

from __future__ import annotations

import argparse
import json
import time

from megasim.timing import IdealDram

from . import energy, timing
from .workload import load_workload


RAM_ITERS, RAM_TOL, RAM_RELAX = 10, 0.01, 0.5


def simulate_run(run_dir: str, seed: int = 0, hw: timing.HW = timing.HW(),
                 dram: str = "ramulator", verbose: bool = False) -> dict:
    t0 = time.time()
    w = load_workload(run_dir, seed)
    plan = timing.build_plan(w, hw)
    t1 = time.time()
    if dram == "ramulator":
        from megasim.dram.ramulator import fixed_point
        # damped fixed point (G-06): SGCN's aggregation is DRAM-bound and the undamped
        # iteration oscillates; damped, it settles into a ±3 % band, recorded as `spread`
        rep = fixed_point(lambda d: timing.schedule(plan, hw, d), hw, iters=RAM_ITERS,
                          tol=RAM_TOL, relax=RAM_RELAX, verbose=verbose)
        h = rep.info["ramulator"]["cycles_history"][-4:]
        rep.info["ramulator"]["spread_last4"] = (max(h) - min(h)) / rep.total_cycles
    else:
        rep = timing.schedule(plan, hw)
    t2 = time.time()
    ideal = timing.schedule(plan, hw, IdealDram(hw))
    db = rep.dram_bytes
    out = {
        "run": w.name, "design": "sgcn", "dram_model": dram,
        "subgraphs": "—", "w0_tile": plan.w0_tile,
        "cycles": int(round(rep.total_cycles)), "time_us": rep.total_cycles / hw.clock_ghz / 1e3,
        "cycles_ideal_dram": float(ideal.total_cycles),
        "dram_stall_share": 1 - ideal.total_cycles / rep.total_cycles,
        "dram_MB": {k: v / 1e6 for k, v in db.items()}, "dram_MB_total": sum(db.values()) / 1e6,
        "phases": [{k: (float(v) if not isinstance(v, (dict, str)) else v) for k, v in p.items()}
                   for p in rep.phases],
        "cache_hit_rate": sum(p["hits"] for p in rep.phases) / max(sum(p["accesses"] for p in rep.phases), 1),
        "energy": energy.energy(rep),
        "nodes": w.n, "edges_with_self": w.edges, "dims": w.dims,
        "fp32_test_acc": w.fp32_test_acc,
        "x_density": [float((h != 0).mean()) for h in w.hidden],
        "x0_density": w.x0.nnz / (w.n * w.dims[0]),
        "time_plan_s": round(t1 - t0, 1), "time_timing_s": round(t2 - t1, 1)}
    if "ramulator" in rep.info:
        out["ramulator"] = rep.info["ramulator"]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", nargs="+")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None)
    p.add_argument("--dram", choices=["ramulator", "uncontended"], default="ramulator")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    out = []
    for r in a.runs:
        res = simulate_run(r, a.seed, dram=a.dram, verbose=a.verbose)
        out.append(res)
        print(json.dumps(res, indent=1, default=float))
    print("\nSUMMARY (SGCN, cycles at 1 GHz)")
    for r in out:
        print(f"  {r['run']:<14} {r['cycles']:>11,} cyc {r['time_us']:9.1f} us  DRAM "
              f"{r['dram_MB_total']:8.2f} MB  hit {r['cache_hit_rate']:.2f}  "
              f"E {r['energy']['total_uJ']:9.1f} uJ  stall {r['dram_stall_share']:.1%}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
