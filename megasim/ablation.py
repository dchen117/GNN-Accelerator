"""Fig.19(b) DRAM ablation of MEGA, reconstructed in our accounting (diagnostic for the
DRAM-accounting question, docs/simulator/00-design.md S-15).

MEGA's Fig.19(b) adds its techniques one at a time to a HyGCN-like A(XW) design (GCN,
averaged over the datasets) and states the DRAM reduction of each step (p.12 §VI-D-2):

    S0 HyGCN-C               32-bit dense features/weights, no partition          64.2
    S1 + DQ, Bitmap storage   bitmap + values at the highest bitwidth (8 bit)     11.0  (5.8×)
    S2 + Adaptive-Package     values at their own bitwidth, in packages            4.4  (2.5×)
    S3 + Condense-Edge (MEGA) METIS subgraphs + Sparse Buffer regions              1.0  (4.4×;
                                                                         NELL alone 11.6×)

The paper does not describe S0–S2 further. We build each step from the M3 traffic model
by swapping only the streams the step changes, so the ratios show which DRAM accounting
can produce the paper's numbers:

  X̄⁽⁰⁾       S0: N·F0·4 B dense.  S1: bitmaps + 8 bit per non-zero.  S2/S3: bitmaps + packages.
  X̄⁽ˡ⁾ l≥1  fused (S-10, never in DRAM) or round trip (written by the Encoder, read by the
             next combination), selected by `hidden`. S0 is dense 32-bit in the round trip.
  W̄          S0: 32 bit, read once.  S1–S3: the M3 numbers (4 bit; row cache misses).
  B̄ rows    S0: 32 bit.  S1–S3: 4 bit (Q-09).
             S0–S2 (no Condense-Edge, no partition, Fig.12(b)): every layer's B̄ is written
             once (unless it fits on chip, as in M3) and aggregation runs in windows of
             `cap` consecutive dataset-order nodes. Each window reads each source it needs
             with its own 128 B access(es) (Fig.12: "the granularity of each DRAM access is
             128B"), so a 64 B row wastes half of the access, as Fig.12 describes.
             S3: M3's b_int + sparse_w/sparse_r streams.
  logits     S0: 32 bit; others 16 bit.
  topology   the same bytes in every step (M3's csc + eid + meta).

`x0_bitmap` selects how X̄⁽⁰⁾'s per-node bitmaps are counted (S1–S3): "dense" (Q-19,
F bits per node) or "none" (not counted), to measure how much of the gap they explain.

    python -m megasim.ablation runs/hw/gcn_cora runs/hw/gcn_citeseer ...
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from . import functional, partition, timing
from .program import load_program

PAPER = {"S0/S1": 5.8, "S1/S2": 2.5, "S2/S3": 4.4, "S0/S3": 64.2}
ACCESS = 128          # bytes per naive DRAM access (Fig.12)


def naive_agg_bytes(g, cap: int, row_bytes: int) -> float:
    """Aggregation reads without partition/Condense: windows of `cap` consecutive nodes
    (dataset order); each (window, distinct source) pair costs ⌈row/128⌉ 128-B accesses."""
    dst = np.repeat(np.arange(g.n), np.diff(g.dst_ptr))
    win = dst // cap
    pairs = np.unique(win * g.n + g.dst_src).size
    return pairs * -(-row_bytes // ACCESS) * ACCESS


def ablate(run_dir: str, seed: int = 0, hw: timing.HW = timing.HW()) -> dict:
    prog0, _ = load_program(run_dir, seed)                 # dataset order (S0–S2)
    part = partition.partition(prog0)
    prog = partition.relabel(prog0, part.order)            # METIS order (S3)
    func = functional.run(prog)
    rep = timing.simulate(prog, part.ptr, func, hw)        # M3 streams (M2 server is enough:
    B = rep.dram_bytes                                     # traffic does not depend on timing)
    n, L = prog.graph.n, len(prog.layers)
    F = [prog.layers[0].F_in] + [l.F_out for l in prog.layers]
    xs = [prog.x0] + [r.x_out for r in func[:-1]]
    get = lambda k: float(B.get(k, 0.0))

    topo = get("csc") + get("eid") + get("meta")
    w4 = get("weights") + get("w_miss")
    w32 = sum(F[l] * F[l + 1] * 4 for l in range(L))
    logits16 = get("logits")

    # X̄ sizes (bytes) per layer: bitmap, packages (own bitwidth), 8-bit values
    bm, pk, v8 = [], [], []
    for l in range(L):
        _, pb = timing.package_lines(xs[l], prog.layers[l].bits, part.ptr, hw)
        _, bb = timing.bitmap_lines(xs[l], part.ptr, hw)
        bm.append(bb.sum() / 8)
        pk.append(pb.sum() / 8)
        v8.append(xs[l].val.size * 1.0)

    # B̄ traffic: M3 (Condense) vs naive windows
    b_m3 = get("b_int_w") + get("b_int_r") + get("sparse_w") + get("sparse_r")
    b_naive4, b_naive32 = 0.0, 0.0
    for l in range(L):
        fo = F[l + 1]
        onchip = n * timing.row_bits(fo) <= hw.comb_buf // 2 * 8
        if not onchip:
            b_naive4 += n * fo * 4 / 8 + naive_agg_bytes(prog0.graph, part.cap, fo * 4 // 8)
        # 32-bit B never fits a 48 KB half for these graphs
        b_naive32 += n * fo * 4 + naive_agg_bytes(prog0.graph, part.cap, fo * 4)

    out = {"run": prog.name, "MB": {}, "ratios": {}}
    for hidden in ("fused", "roundtrip"):
        for x0bm in ("dense", "none"):
            rt = hidden == "roundtrip"
            hid = lambda arr: 2 * sum(arr[1:]) if rt else 0.0     # write + read, l ≥ 1
            bm0 = bm[0] if x0bm == "dense" else 0.0
            bmh = hid(bm)
            s0 = (n * F[0] * 4 + (2 * sum(n * F[l] * 4 for l in range(1, L)) if rt else 0)
                  + w32 + b_naive32 + 2 * logits16 + topo)
            s1 = bm0 + v8[0] + bmh + hid(v8) + w4 + b_naive4 + logits16 + topo
            s2 = bm0 + pk[0] + bmh + hid(pk) + w4 + b_naive4 + logits16 + topo
            s3 = bm0 + pk[0] + bmh + hid(pk) + w4 + b_m3 + logits16 + topo
            key = f"{hidden}/x0bitmap={x0bm}"
            out["MB"][key] = {s: v / 2 ** 20 for s, v in zip(("S0", "S1", "S2", "S3"), (s0, s1, s2, s3))}
            out["ratios"][key] = {"S0/S1": s0 / s1, "S1/S2": s1 / s2, "S2/S3": s2 / s3, "S0/S3": s0 / s3}
    out["parts_MB"] = {k: v / 2 ** 20 for k, v in {
        "x0_bitmap": bm[0], "x0_pkg": pk[0], "x0_8bit": v8[0], "hidden_bitmap": sum(bm[1:]),
        "hidden_pkg": sum(pk[1:]), "hidden_8bit": sum(v8[1:]), "w4": w4, "b_m3": b_m3,
        "b_naive4": b_naive4, "b_naive32": b_naive32, "logits16": logits16, "topo": topo}.items()}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", nargs="+")
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    res = [ablate(r) for r in a.runs]
    keys = list(res[0]["ratios"])
    print("paper (GCN average):", PAPER, " NELL S2/S3 = 11.6")
    for k in keys:
        print(f"\n== {k}")
        for r in res:
            q = r["ratios"][k]
            print(f"  {r['run']:<14} " + "  ".join(f"{s}={v:6.2f}" for s, v in q.items())
                  + f"   S3={r['MB'][k]['S3']:.2f} MB")
        g = {s: float(np.exp(np.mean([np.log(r['ratios'][k][s]) for r in res]))) for s in PAPER}
        print("  geomean        " + "  ".join(f"{s}={v:6.2f}" for s, v in g.items()))
    for r in res:
        print(r["run"], {k: round(v, 3) for k, v in r["parts_MB"].items()})
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
