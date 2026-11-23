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


def calibrate(plan, hw: timing.HW) -> tuple[list, float]:
    """Ramulator numbers for BackfillDram (G-06): each phase's sustained bandwidth, from a
    replay of that phase's transfers all issued at cycle 0 (in schedule order, so the DMA
    FIFOs keep the HBM saturated), and the unloaded latency of one line."""
    from megasim.dram import ramulator as R
    ideal = timing.schedule(plan, hw, IdealDram(hw))
    bw = []
    for ph in ideal.phases:
        a, b = ph["log"]
        tr = R.trace_from_log([(0.0, s, ad) for _, s, ad in ideal.dram.log[a:b]], hw)
        if tr.addr.size == 0:
            bw.append(hw.dram_bytes_per_cycle)
            continue
        _, last, _ = R.replay(tr)
        bw.append(tr.addr.size * hw.line / (last.max() + 1))
    first, _, _ = R.replay(R.trace_from_log([(0.0, {"sg_topo": 1.0}, {})], hw))
    return bw, float(first[0] + 1)


def simulate_ramulator(plan, hw: timing.HW):
    """Schedule with the Ramulator-calibrated time-ordered server, then replay the resulting
    trace through Ramulator at its issue times as a check: Ramulator should finish the
    traffic about when the schedule does (`replay_end_over_total`)."""
    from megasim.dram import ramulator as R
    bw, lat = calibrate(plan, hw)
    rep = timing.schedule(plan, hw, timing.BackfillDram(hw, bw, lat))
    tr = R.trace_from_log(rep.dram.log, hw)
    _, last, st = R.replay(tr)
    ram = {"phase_bw_GBps": [b * hw.clock_ghz for b in bw],
           "phase_bw_efficiency": [b / hw.dram_bytes_per_cycle for b in bw],
           "unloaded_latency": lat, "lines": int(tr.addr.size),
           "replay_end_over_total": float((last.max() + 1) / rep.total_cycles),
           "row_hit_rate": st["row_hits"] / max(st["row_hits"] + st["row_misses"] + st["row_conflicts"], 1),
           **{k: v for k, v in st.items() if k != "driver"}}
    return rep, ram


def simulate_run(run_dir: str, seed: int = 0, hw: timing.HW = timing.HW(),
                 dram: str = "ramulator", verbose: bool = False, residual: bool = False) -> dict:
    t0 = time.time()
    w = load_workload(run_dir, seed)
    plan = timing.build_plan(w, hw, residual=residual)
    t1 = time.time()
    if dram == "ramulator":
        rep, ram = simulate_ramulator(plan, hw)
    else:
        rep, ram = timing.schedule(plan, hw), None
    t2 = time.time()
    ideal = timing.schedule(plan, hw, IdealDram(hw))
    db = rep.dram_bytes
    out = {
        "run": w.name, "design": "sgcn", "dram_model": dram,
        "subgraphs": "—", "w0_tile": plan.w0_tile, "residual_layers": plan.residual_layers,
        "cycles": int(round(rep.total_cycles)), "time_us": rep.total_cycles / hw.clock_ghz / 1e3,
        "cycles_ideal_dram": float(ideal.total_cycles),
        "dram_stall_share": 1 - ideal.total_cycles / rep.total_cycles,
        "dram_MB": {k: v / 1e6 for k, v in db.items()}, "dram_MB_total": sum(db.values()) / 1e6,
        "phases": [{k: (float(v) if not isinstance(v, (dict, str, tuple)) else v) for k, v in p.items()}
                   for p in rep.phases],
        "cache_hit_rate": sum(p["hits"] for p in rep.phases) / max(sum(p["accesses"] for p in rep.phases), 1),
        "energy": energy.energy(rep),
        "nodes": w.n, "edges_with_self": w.edges, "dims": w.dims,
        "fp32_test_acc": w.fp32_test_acc,
        "x_density": [float((h != 0).mean()) for h in w.hidden],
        "x0_density": w.x0.nnz / (w.n * w.dims[0]),
        "time_plan_s": round(t1 - t0, 1), "time_timing_s": round(t2 - t1, 1)}
    if ram:
        out["ramulator"] = ram
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", nargs="+")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None)
    p.add_argument("--dram", choices=["ramulator", "uncontended"], default="ramulator")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--residual", action="store_true",
                   help="+X⁽ˡ⁾ residual on layers with F_in = F_out (G-08; none in MEGA's 2-layer GCNs)")
    a = p.parse_args(argv)
    out = []
    for r in a.runs:
        res = simulate_run(r, a.seed, dram=a.dram, verbose=a.verbose, residual=a.residual)
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
