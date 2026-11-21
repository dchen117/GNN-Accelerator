"""METIS partitioning and node relabelling (Q-27, S-08).

    part = partition(prog)                  # subgraph sizes / boundaries
    prog2 = relabel(prog, part.order)       # the program the hardware actually runs

Subgraph capacity (Q-25): one Aggregation Buffer half is 64 KB of 16-bit partial sums, so a
subgraph can hold at most 65536 / (2 · F2) destinations, where F2 is the widest aggregated
layer output (128 → 256 nodes, 256 → 128). The same partition serves every layer.

Number of parts (S-08, ours): METIS is run with K = ⌈N / capacity⌉ parts and its default
imbalance tolerance (ufactor = 30, parts up to 3 % above average). If any part exceeds the
capacity, K is increased by one and METIS rerun, until every part fits. So the partition has
the fewest subgraphs that respect the hard buffer limit.
Why not a tight tolerance: ufactor = 1 (0.1 %) almost triples the edge cut on Cora (0.98
external rows per node instead of 0.35), i.e. Condense-Edge traffic; a few percent of
emptier subgraphs costs far less.

Relabelling (Q-27): nodes are renumbered so that subgraph k is the contiguous ID range
[ptr[k], ptr[k+1]); inside a subgraph the original order is kept (stable). Ascending-ID
Condense matching and contiguous internal reads both need this.

METIS sees the symmetrised graph without self-loops (GraphSage's sampled graph is directed;
partitioning only needs connectivity).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pymetis

from .program import Graph, Program, SparseRows

AGG_HALF_BYTES = 64 * 1024     # Aggregation Buffer 128 KB, ping-pong (Q-25)
PSUM_BYTES = 2                 # 16-bit partial sums


def capacity(prog: Program) -> int:
    """Destinations per subgraph: one Aggregation Buffer half / (2 B · widest F2)."""
    f2 = max(layer.F_out for layer in prog.layers)
    return AGG_HALF_BYTES // (PSUM_BYTES * f2)


@dataclass
class Partition:
    part: np.ndarray       # (n,) subgraph of every ORIGINAL node id
    order: np.ndarray      # (n,) new id → original id
    ptr: np.ndarray        # (K+1,) subgraph k = new ids [ptr[k], ptr[k+1])
    cap: int
    metis_runs: int

    @property
    def num_subgraphs(self) -> int:
        return self.ptr.size - 1


def _metis(g: Graph, k: int, seed: int) -> np.ndarray:
    if k == 1:
        return np.zeros(g.n, dtype=np.int64)
    src = np.repeat(np.arange(g.n), np.diff(g.src_ptr))
    dst = g.src_dst
    m = src != dst
    s, d = np.concatenate([src[m], dst[m]]), np.concatenate([dst[m], src[m]])
    key = np.unique(s * g.n + d)                         # symmetric, no duplicates
    s, d = key // g.n, key % g.n
    xadj = np.concatenate([[0], np.cumsum(np.bincount(s, minlength=g.n))])
    _, part = pymetis.part_graph(k, pymetis.CSRAdjacency(xadj, d),
                                 options=pymetis.Options(seed=seed))
    return np.asarray(part, dtype=np.int64)


def partition(prog: Program, cap: int | None = None, seed: int = 0, max_tries: int = 50) -> Partition:
    cap = cap or capacity(prog)
    n = prog.graph.n
    k = math.ceil(n / cap)
    for tries in range(1, max_tries + 1):
        part = _metis(prog.graph, k, seed)
        sizes = np.bincount(part, minlength=k)
        if sizes.max() <= cap:
            break
        k += 1
    else:
        raise RuntimeError(f"no partition with parts ≤ {cap} after {max_tries} METIS runs")
    # drop empty parts (METIS may leave some) and order subgraphs by part id
    used = np.flatnonzero(sizes)
    remap = np.full(k, -1)
    remap[used] = np.arange(used.size)
    part = remap[part]
    order = np.lexsort((np.arange(n), part))            # by subgraph, stable inside
    ptr = np.concatenate([[0], np.cumsum(np.bincount(part))])
    return Partition(part=part, order=order, ptr=ptr, cap=cap, metis_runs=tries)


def relabel(prog: Program, order: np.ndarray) -> Program:
    """The same program with node new_id = position in `order` (order[new] = old)."""
    n = prog.graph.n
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    g = prog.graph
    src = np.repeat(np.arange(n), np.diff(g.src_ptr))
    graph = Graph.from_edges(inv[src], inv[g.src_dst], n)

    x = prog.x0
    nnz = x.nnz_per_row[order]
    ptr = np.concatenate([[0], np.cumsum(nnz)])
    # index of every non-zero in the old arrays: old row start + rank inside the row
    sel = np.repeat(x.ptr[order] - ptr[:-1], nnz) + np.arange(ptr[-1])
    x0 = SparseRows(n, x.F, ptr, x.col[sel], x.val[sel])

    layers = [replace(l, bits=l.bits[order], deg_idx=l.deg_idx[order]) for l in prog.layers]
    return replace(prog, graph=graph, x0=x0, layers=layers,
                   y=None if prog.y is None else prog.y[order],
                   test_idx=None if prog.test_idx is None else np.sort(inv[prog.test_idx]))
