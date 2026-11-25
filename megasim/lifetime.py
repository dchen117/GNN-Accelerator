"""How long subgraph k's X̄⁽¹⁾ must stay on chip for a residual (docs/beyond-mega/02-residual.md).

Under our sweep schedule (Q-28), X̄⁽ˡ⁾_k is combined in sweep l and needed by the Encoder in
sweep l+1, after every other subgraph has passed through the Input Buffer: keeping it on chip
means holding the whole layer. This also measures a "wavefront" schedule (not adopted): aggregate
subgraph k as soon as every subgraph holding one of its sources has been combined, and drop
X̄⁽ˡ⁾_k then. Sizes are packages + bitmaps of layer 1's input, per subgraph.

    python -m megasim.lifetime runs/hw/gcn_cora runs/hw/gcn_pubmed ...
"""

from __future__ import annotations

import argparse

import numpy as np

from . import functional, partition, timing
from .program import load_program


def lifetime(run_dir: str, seed: int = 0, hw: timing.HW = timing.HW()) -> dict:
    prog0, _ = load_program(run_dir, seed)
    part = partition.partition(prog0)
    prog = partition.relabel(prog0, part.order)
    g, ptr = prog.graph, part.ptr
    K = ptr.size - 1
    sub = np.repeat(np.arange(K), np.diff(ptr))
    dst = np.repeat(np.arange(g.n), np.diff(g.dst_ptr))
    # last subgraph that must be combined before subgraph k can be aggregated
    ready = np.zeros(K, dtype=np.int64)
    np.maximum.at(ready, sub[dst], sub[g.dst_src])
    ready = np.maximum(ready, np.arange(K))
    x1 = functional.run(prog)[0].x_out
    _, pb = timing.package_lines(x1, prog.layers[1].bits, ptr, hw)
    _, bb = timing.bitmap_lines(x1, ptr, hw)
    sb = (pb + bb) / 8                                 # bytes of X̄⁽¹⁾ per subgraph
    # wavefront: after combining subgraph t, subgraphs ≤ t not yet aggregatable are live
    live = np.array([sb[(np.arange(K) <= t) & (ready > t)].sum() for t in range(K)])
    span = ready - np.arange(K)
    return {"run": prog.name, "subgraphs": K, "whole_KB": sb.sum() / 1024,
            "wavefront_peak_KB": live.max() / 1024,
            "span_median": float(np.median(span)), "span_max": int(span.max())}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", nargs="+")
    a = p.parse_args(argv)
    for r in a.runs:
        d = lifetime(r)
        print(f"{d['run']:<20} K={d['subgraphs']:4d}  whole X̄1={d['whole_KB']:7.1f} KB  "
              f"wavefront peak={d['wavefront_peak_KB']:7.1f} KB  "
              f"distance to last source median={d['span_median']:.0f} max={d['span_max']}")


if __name__ == "__main__":
    main()
