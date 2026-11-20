"""Hardware program: everything the MEGA hardware needs to run one trained model (S-07).

    prog = load_program("runs/hw/gcn_cora", seed=0)

A program is plain NumPy (no torch) so the simulator does not depend on the training code:

  graph      binary A+I (fold mode, Q-11) in CSC order: for each source j, its destinations.
             For GraphSage it is the fixed evaluation sample (the graph the hardware sees).
  x0         X̄⁽⁰⁾, quantized offline (Q-10), as CSR (indptr, col, val) of non-zero integers.
  layers[l]  W̄ (int8, F_l × F_{l+1}), per-node bitwidth b and degree-table index (Q-26),
             and the fixed-point tables of Q-16:
               alpha_p  α'_d = α_d · s_d          (combination rescale, degree table)
               c        c_j = β_j / γ_j           (combination rescale, column table)
               enc      e_d = t_d / α^{l+1}_d     (Encoder, degree table; not on the last layer)
               gamma    γ_j                       (Encoder column table; logits on the last layer)

Fixed-point format (Q-16, S-05): 16-bit unsigned codes and one binary point per table, chosen
so the largest entry uses all 16 bits. value = code / 2**shift.

The degree tables are *compacted*: they list only the degrees that occur (Q-26), and each node
stores a 16-bit index into them. Node order is the dataset's order here; METIS relabelling
(Q-27) is applied on top in M2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

FX_BITS = 16          # Q-16: 16-bit unsigned scale codes


# --------------------------------------------------------------------------- fixed point
@dataclass
class FixedTable:
    """A table of positive factors in 16-bit fixed point: value = codes / 2**shift."""
    codes: np.ndarray     # int64, each in [0, 2**16)
    shift: int            # fractional bits, shared by the whole table

    @classmethod
    def from_values(cls, v: np.ndarray, bits: int = FX_BITS) -> "FixedTable":
        """Round-to-nearest with the binary point set by the largest entry.
        Must match mega_pt.hw_stats.to_fixed(v, "fx16") exactly (the bit-exact check uses it)."""
        v = np.asarray(v, dtype=np.float64)
        m = float(v.max())
        shift = bits - (math.floor(math.log2(m)) + 1)          # largest entry < 2**bits
        codes = np.floor(v * 2.0 ** shift + 0.5).astype(np.int64)
        assert codes.max() < 2 ** bits and codes.min() >= 0, (codes.min(), codes.max())
        return cls(codes, shift)

    def values(self) -> np.ndarray:
        return self.codes / 2.0 ** self.shift


# --------------------------------------------------------------------------- containers
@dataclass
class Graph:
    """Binary A+I (self-loops included, Q-01) of the graph the hardware runs on."""
    n: int
    deg: np.ndarray       # (n,) in-degree incl. self-loop = CSR row length (Q-01)
    # CSR by destination (what psum_i = Σ_j A_ij B̄_j reads)
    dst_ptr: np.ndarray
    dst_src: np.ndarray
    # CSC by source (the AU's outer-product walk, Q-25): destinations of each source
    src_ptr: np.ndarray
    src_dst: np.ndarray

    @classmethod
    def from_edges(cls, src: np.ndarray, dst: np.ndarray, n: int) -> "Graph":
        o = np.lexsort((src, dst))
        dst_ptr = np.concatenate([[0], np.cumsum(np.bincount(dst, minlength=n))])
        o2 = np.lexsort((dst, src))
        src_ptr = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=n))])
        return cls(n=n, deg=np.diff(dst_ptr), dst_ptr=dst_ptr, dst_src=src[o],
                   src_ptr=src_ptr, src_dst=dst[o2])


@dataclass
class SparseRows:
    """Row-major non-zeros of an integer feature matrix (n × F), columns ascending per row."""
    n: int
    F: int
    ptr: np.ndarray       # (n+1,)
    col: np.ndarray       # int64
    val: np.ndarray       # int64, never 0

    @classmethod
    def from_dense(cls, x: np.ndarray) -> "SparseRows":
        r, c = np.nonzero(x)
        return cls.from_coo(r, c, x[r, c], *x.shape)

    @classmethod
    def from_coo(cls, r, c, v, n, F) -> "SparseRows":
        r, c, v = (np.asarray(a, dtype=np.int64) for a in (r, c, v))
        keep = v != 0
        r, c, v = r[keep], c[keep], v[keep]
        o = np.lexsort((c, r))
        ptr = np.concatenate([[0], np.cumsum(np.bincount(r, minlength=n))])
        return cls(n=n, F=F, ptr=ptr, col=c[o], val=v[o])

    @property
    def nnz_per_row(self) -> np.ndarray:
        return np.diff(self.ptr)

    def rows(self) -> np.ndarray:
        """Row index of every stored non-zero."""
        return np.repeat(np.arange(self.n), self.nnz_per_row)

    def equals(self, o: "SparseRows") -> bool:
        return (self.n == o.n and self.F == o.F and np.array_equal(self.ptr, o.ptr)
                and np.array_equal(self.col, o.col) and np.array_equal(self.val, o.val))


@dataclass
class Layer:
    F_in: int
    F_out: int
    W: np.ndarray                 # int8 (F_in, F_out), levels −7..7 (Q-08)
    bits: np.ndarray              # (n,) input bitwidth per node, 2..8 (Q-07)
    deg_idx: np.ndarray           # (n,) index into the compacted degree table (Q-26)
    deg_values: np.ndarray        # degrees present, ascending (table row k ↔ degree deg_values[k])
    bits_table: np.ndarray        # (K,) b per table row
    alpha_p: FixedTable           # (K,) α'_d
    c: FixedTable                 # (F_out,) c_j
    gamma: FixedTable             # (F_out,) γ_j
    enc: FixedTable | None        # (K,) e_d, None on the last layer
    out_bits_table: np.ndarray | None = None   # (K,) bitwidth of the NEXT layer's input (Encoder qmax)

    @property
    def last(self) -> bool:
        return self.enc is None


@dataclass
class Program:
    name: str
    model: str                    # gcn | gin | sage
    graph: Graph
    x0: SparseRows
    layers: list[Layer] = field(default_factory=list)
    y: np.ndarray | None = None
    test_idx: np.ndarray | None = None


# --------------------------------------------------------------------------- export
def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


@torch.no_grad()
def load_program(run_dir: str, seed: int = 0) -> tuple[Program, tuple]:
    """Export a trained run into a Program. Also returns (model, g, ops) so callers can run
    the PyTorch integer reference on the same inputs."""
    from mega_pt.hw_stats import load_run              # training-side code, only used here

    model, g, ops, args = load_run(run_dir, seed)
    n = g.num_nodes
    deg = _np(ops.deg).astype(np.int64)
    a = ops.a_bin.coalesce()
    dst, src = (_np(i).astype(np.int64) for i in a.indices())
    graph = Graph.from_edges(src, dst, n)
    assert np.array_equal(graph.deg, deg), "deg must equal the A+I row length (Q-01)"

    s = _np(ops.src_scale).astype(np.float64)
    t = _np(ops.dst_scale).astype(np.float64)
    deg_values, deg_idx = np.unique(deg, return_inverse=True)
    # s_d, t_d depend on the degree only, so the compacted table can hold them per row.
    first = np.full(deg_values.size, -1)
    first[deg_idx[::-1]] = np.arange(n)[::-1]                # one node per table row
    s_tab, t_tab = s[first], t[first]

    xbar0 = model.x_quant[0](g.x, ops.deg)[1]
    if xbar0.is_sparse:
        xbar0 = xbar0.coalesce()
        r, c = _np(xbar0.indices())
        x0 = SparseRows.from_coo(r, c, np.rint(_np(xbar0.values())), n, xbar0.shape[1])
    else:
        x0 = SparseRows.from_dense(np.rint(_np(xbar0)).astype(np.int64))

    L = len(model.weights)
    layers = []
    for l in range(L):
        Wbar = np.rint(_np(model.w_quant[l](model.weights[l])[1])).astype(np.int8)
        q = model.x_quant[l]
        bits_all = _np(q.node_bits(torch.as_tensor(deg_values), rounded=True)).astype(np.int64)
        alpha = _np(q.scale).astype(np.float64)[deg_values]
        beta = _np(model.w_quant[l].scale).astype(np.float64)
        gamma = _np(model.b_quant[l].scale).astype(np.float64)
        enc = out_bits = None
        if l < L - 1:
            qn = model.x_quant[l + 1]
            enc = FixedTable.from_values(t_tab / _np(qn.scale).astype(np.float64)[deg_values])
            out_bits = _np(qn.node_bits(torch.as_tensor(deg_values), rounded=True)).astype(np.int64)
        layers.append(Layer(
            F_in=model.dims[l], F_out=model.dims[l + 1], W=Wbar,
            bits=bits_all[deg_idx], deg_idx=deg_idx, deg_values=deg_values, bits_table=bits_all,
            alpha_p=FixedTable.from_values(alpha * s_tab), c=FixedTable.from_values(beta / gamma),
            gamma=FixedTable.from_values(gamma), enc=enc, out_bits_table=out_bits))
    prog = Program(name=run_dir.rstrip("/").split("/")[-1], model=args.model, graph=graph,
                   x0=x0, layers=layers, y=_np(g.y), test_idx=_np(g.test_idx))
    return prog, (model, g, ops)
