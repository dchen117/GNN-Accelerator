"""Random hardware programs for tests and debugging (no trained run needed).

    prog = random_program(seed=0, n=600, dims=(300, 128, 7))

The program is internally consistent (bitwidths come from degree tables, the Encoder's output
bitwidths equal the next layer's input bitwidths, scale tables are positive), so the
functional and timing models run on it exactly as on an exported run. The numbers are not
meant to be a meaningful GNN: the scales are drawn so that values stay in range and spread
over the levels.
"""

from __future__ import annotations

import numpy as np

from .program import FixedTable, Graph, Layer, Program, SparseRows


def random_graph(rng, n: int, avg_deg: float, communities: int) -> Graph:
    """Undirected graph with community structure (so METIS has something to find), plus
    self-loops (A+I, Q-01)."""
    m = int(n * avg_deg / 2)
    comm = rng.integers(0, communities, size=n)
    a = rng.integers(0, n, size=m)
    same = rng.random(m) < 0.9                       # 90 % of edges stay in the community
    pool = [np.flatnonzero(comm == c) for c in range(communities)]
    b = np.where(same, [rng.choice(pool[comm[i]]) for i in a], rng.integers(0, n, size=m))
    keep = a != b
    src = np.concatenate([a[keep], b[keep], np.arange(n)])
    dst = np.concatenate([b[keep], a[keep], np.arange(n)])
    key = np.unique(src * n + dst)
    return Graph.from_edges(key // n, key % n, n)


def random_program(seed: int = 0, n: int = 600, dims=(300, 128, 7), avg_deg: float = 4.0,
                   x0_density: float = 0.05, communities: int = 8) -> Program:
    rng = np.random.default_rng(seed)
    g = random_graph(rng, n, avg_deg, communities)
    deg_values, deg_idx = np.unique(g.deg, return_inverse=True)
    K = deg_values.size
    L = len(dims) - 1
    # bitwidth tables: lower degree → more bits (MEGA's trend), 2..8
    bits_tabs = [np.clip(8 - np.round(np.log2(deg_values)).astype(int) - rng.integers(0, 3), 2, 8)
                 for _ in range(L)]
    b0 = bits_tabs[0][deg_idx]
    lim = 2 ** (b0 - 1) - 1
    mask = rng.random((n, dims[0])) < x0_density
    x = rng.integers(1, lim[:, None] + 1, size=(n, dims[0])) * mask
    layers = []
    for l in range(L):
        F_in, F_out = dims[l], dims[l + 1]
        W = rng.integers(-7, 8, size=(F_in, F_out)).astype(np.int8)
        # rescale so that acc · α' · c lands around a few B̄ levels
        typ = 7 * 4 * np.sqrt(max(F_in * (x0_density if l == 0 else 0.5), 1))
        alpha_p = FixedTable.from_values(rng.uniform(0.5, 1.5, K) / typ)
        c = FixedTable.from_values(rng.uniform(0.5, 2.0, F_out))
        gamma = FixedTable.from_values(rng.uniform(0.05, 0.2, F_out))
        enc = out_bits = None
        if l < L - 1:
            enc = FixedTable.from_values(rng.uniform(1.0, 4.0, K) / np.sqrt(deg_values))
            out_bits = bits_tabs[l + 1]
        layers.append(Layer(F_in, F_out, W, bits_tabs[l][deg_idx], deg_idx, deg_values,
                            bits_tabs[l], alpha_p, c, gamma, enc, out_bits))
    y = rng.integers(0, dims[-1], size=n)
    return Program(name=f"synthetic{seed}", model="gcn", graph=g, x0=SparseRows.from_dense(x),
                   layers=layers, y=y, test_idx=np.arange(0, n, 3))
