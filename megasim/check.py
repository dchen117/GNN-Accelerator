"""Bit-exact check of the functional model against the PyTorch integer reference (Q-30).

Usage:
    python -m megasim.check runs/hw/gcn_cora [runs/hw/gin_cora ...] [--seed 0] [--json out.json]

For each run: export the hardware program, run megasim.functional, run
mega_pt.hw_stats.int_forward(fmt="fx16") on the same model, and require exact equality of
  X̄⁽⁰⁾, every layer's Tile Accumulator output, AU partial sums, the Encoder's X̄⁽ˡ⁺¹⁾,
  and the logits.
Also prints the width checks (Shifter-Acc, Tile Accumulator, AU) and the per-layer memory
numbers of the stored format (Q-19), and the test accuracy of the simulated logits.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from . import functional
from .program import SparseRows, load_program


def _rows_from_stats(st: dict, n: int, F: int) -> SparseRows:
    return SparseRows.from_coo(st["r"].numpy(), st["c"].numpy(), np.rint(st["v"].numpy()), n, F)


def check_run(run_dir: str, seed: int = 0) -> dict:
    from mega_pt.hw_stats import int_forward

    t0 = time.time()
    prog, (model, g, ops) = load_program(run_dir, seed)
    t1 = time.time()
    sim = functional.run(prog)
    t2 = time.time()
    stats: dict = {}
    ref = int_forward(model, g.x, ops, "fx16", stats).numpy()
    n = prog.graph.n

    res = {"run": run_dir, "model": prog.model, "nodes": n, "edges_with_self": int(prog.graph.deg.sum()),
           "time_export_s": round(t1 - t0, 1), "time_sim_s": round(t2 - t1, 1)}
    ok = True
    for l, (layer, r) in enumerate(zip(prog.layers, sim)):
        st = stats[l]
        x_in = prog.x0 if l == 0 else sim[l - 1].x_out
        eq = {
            "x_in": x_in.equals(_rows_from_stats(st, n, layer.F_in)),
            "acc": np.array_equal(r.acc, st["acc"].numpy().astype(np.int64)),
            "psum": np.array_equal(r.psum, st["psum"].numpy().astype(np.int64)),
        }
        if layer.last:
            eq["logits"] = np.array_equal(r.logits_code / 2.0 ** r.logits_shift, ref)
        ok &= all(eq.values())
        bad = [k for k, v in r.checks.items() if k.endswith(("overflow", "saturate")) and v]
        ok &= not bad
        res[f"L{l}"] = {"equal": eq, "checks": r.checks, "memory": r.memory}
    pred = (sim[-1].logits_code).argmax(1)
    res["sim_test_acc"] = float((pred[prog.test_idx] == prog.y[prog.test_idx]).mean())
    res["bit_exact"] = bool(ok)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", nargs="+")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    torch.set_num_threads(4)
    out = []
    for r in a.runs:
        res = check_run(r, a.seed)
        out.append(res)
        print(json.dumps(res, indent=1, default=str))
    print("\nSUMMARY")
    for res in out:
        print(f"  {res['run']:<32} bit_exact={res['bit_exact']}  test_acc={res['sim_test_acc']:.4f}"
              f"  sim {res['time_sim_s']}s")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, default=str)
    return 0 if all(r["bit_exact"] for r in out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
