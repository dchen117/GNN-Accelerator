"""Functional model of the MEGA datapath (S-04): the exact integers every unit produces.

One layer, in hardware order:

  1. Input Buffer → Decoder     X̄ rows are read back from the Adaptive-Package stream
                                (megasim.package), the bitmaps give their feature positions.
  2. Combination Tiles          bit-serial row product (Q-22/23/24):
                                  * a pass = the node's next ≤128 non-zeros, split evenly over
                                    the 4 tiles (≤32 each), Q-23 "nnz" feeding;
                                  * per tile, b bit-planes LSB first; each C-PE ANDs the plane
                                    with its W̄ column (BSEs), sums 8 at a time (adder tree)
                                    and shift-accumulates (Shifter-Acc, 16-bit signed), the
                                    MSB plane subtracted (two's complement);
                                  * Tile Accumulator: 4-input add of the tiles' results, then a
                                    32-bit accumulator over passes.
  3. Rescale (Q-16)             B̄ = clip(round(acc · α'_d · c_j), ±7), integer codes + shift.
  4. Aggregation Units (Q-25)   psum_i = Σ_{j ∈ N(i) ∪ {i}} B̄_j, 16-bit saturating.
  5. Encoder (Q-16/26)          X̄' = min(round(ReLU(psum) · e_d · γ_j), qmax_d), packed into
                                the next layer's packages. Last layer: logits = psum · γ_j.
                                Residual layers (Q-36): + X̄ · r_d before ReLU.

The whole graph is processed at once per bit-plane (no cycle loop); the tile/pass split is
reproduced so every accumulator can be range-checked against the widths of Q-22/Q-25.
Overflow or saturation is not modelled as wrapping: it is counted and reported, and the
functional check then fails, which is what we want to know.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from . import package
from .program import FixedTable, Layer, Program, SparseRows

TILES, TILE_NNZ, PASS_NNZ = 4, 32, 128        # Q-21/23
SHIFT_ACC_BITS, TILE_ACC_BITS, AU_BITS = 16, 32, 16   # Q-22, Q-25
B_LEVEL = 7                                    # B̄ is 4-bit signed (Q-09)


# --------------------------------------------------------------------------- helpers
def round_shift(num: np.ndarray, s: int) -> np.ndarray:
    """round-half-away(num / 2**s) for integers (Eq.(2) rounding, Q-16)."""
    assert s > 0
    mag = (np.abs(num) + (1 << (s - 1))) >> s
    return np.where(num < 0, -mag, mag)


def _check_product_fits(a: np.ndarray, *tables: FixedTable) -> None:
    """acc · A · C must fit int64 (and float64's 53 bits, so int_forward agrees exactly)."""
    m = float(np.abs(a).max(initial=0))
    for t in tables:
        m *= float(t.codes.max())
    assert m < 2.0 ** 53, f"requantization product {m:.3g} exceeds 2^53"


@dataclass
class LayerResult:
    acc: np.ndarray                    # (n, F_out) Tile Accumulator output, int64
    B: np.ndarray                      # (n, F_out) B̄, int8 range
    psum: np.ndarray                   # (n, F_out) AU partial sums
    x_out: SparseRows | None           # next layer's X̄ (None on the last layer)
    logits_code: np.ndarray | None     # last layer: psum · γ code, value = code / 2**shift
    logits_shift: int | None
    checks: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- 1. storage
def store_and_reload(x: SparseRows, bits: np.ndarray) -> tuple[SparseRows, dict]:
    """Pack X̄ into Adaptive-Packages, decode it back, and return the decoded rows plus the
    memory numbers of Q-19 (values-only and stored bits, packages and bitmaps)."""
    b_val = np.repeat(bits, x.nnz_per_row)
    ps = package.encode(x.val, b_val)
    decoded = package.decode(ps.data, b_val)
    assert np.array_equal(decoded, x.val), "package round trip failed"
    elems = x.n * x.F
    val_bits = int((x.nnz_per_row * bits).sum())
    bm_bits = package.bitmap_bits(x.n, x.F)
    mem = {"elements": elems, "nnz": int(x.val.size), "density": x.val.size / elems,
           "value_bits": val_bits, "package_bits": ps.total_bits, "bitmap_bits": bm_bits,
           "packages": ps.num_packages, "modes": ps.mode_histogram(),
           "pad_fraction": 1 - val_bits / max(ps.total_bits, 1),
           # Tab.VI-comparable (values only, Q-15) and stored (packages + bitmaps, Q-19)
           "bits_per_elem_values": val_bits / elems,
           "bits_per_elem_stored": (ps.total_bits + bm_bits) / elems,
           "cr_stored": 32 * elems / (ps.total_bits + bm_bits)}
    return SparseRows(x.n, x.F, x.ptr, x.col, decoded), mem


# --------------------------------------------------------------------------- 2. combination
def _pass_tile(x: SparseRows):
    """Pass and tile of every non-zero under nnz feeding (Q-23): pass p holds ranks
    128p … 128p+127; its s non-zeros go to the tiles in consecutive shares of ⌈s/4⌉."""
    nnz = x.nnz_per_row
    row = x.rows()
    rank = np.arange(x.val.size) - x.ptr[row]
    p = rank // PASS_NNZ
    s = np.minimum(PASS_NNZ, nnz[row] - p * PASS_NNZ)
    share = -(-s // TILES)
    return row, p, (rank % PASS_NNZ) // share


def combination(x: SparseRows, bits: np.ndarray, W: np.ndarray,
                chunk_elems: int = 4_000_000) -> tuple[np.ndarray, dict]:
    """Bit-serial X̄·W̄ (Q-22). Returns acc (n, F_out) and the range checks."""
    n, F_out = x.n, W.shape[1]
    Wd = W.astype(np.int64)
    row, p, tile = _pass_tile(x)
    b_val = bits[row]
    code = x.val & ((1 << b_val) - 1)                 # b-bit two's complement code
    acc = np.zeros((n, F_out), dtype=np.int64)
    sa_max = ta_max = 0                               # Shifter-Acc / Tile Accumulator |max|
    passes = -(-x.nnz_per_row // PASS_NNZ)
    # chunk over rows so that (segments × F_out) stays bounded
    seg_per_row = np.maximum(passes, 1) * TILES
    cum = np.cumsum(seg_per_row)
    r0 = 0
    while r0 < n:
        r1 = int(np.searchsorted(cum, cum[r0] - seg_per_row[r0] + max(1, chunk_elems // F_out),
                                 side="right"))
        r1 = max(r1, r0 + 1)
        lo, hi = x.ptr[r0], x.ptr[r1]
        if hi > lo:
            rr = row[lo:hi]
            pmax = int(p[lo:hi].max()) + 1
            seg_key = ((rr - r0) * pmax + p[lo:hi]) * TILES + tile[lo:hi]
            useg, seg = np.unique(seg_key, return_inverse=True)
            # Shifter-Acc: one (node, pass, tile) segment, b planes LSB first
            sa = np.zeros((useg.size, F_out), dtype=np.int64)
            bb, cc, cl = b_val[lo:hi], code[lo:hi], x.col[lo:hi]
            for k in range(int(bb.max())):
                bit = (cc >> k) & 1
                w = np.where(k == bb - 1, -(1 << k), 1 << k) * (k < bb)
                coef = bit * w
                nzk = coef != 0
                if nzk.any():
                    M = sp.csr_matrix((coef[nzk], (seg[nzk], cl[nzk])), shape=(useg.size, x.F))
                    sa += M @ Wd                                   # adder tree + shift-acc
                sa_max = max(sa_max, int(np.abs(sa).max()))        # running value after plane k
            # Tile Accumulator: 4-input add per pass, then running sum over passes
            pas_key = useg // TILES
            upas, pinv = np.unique(pas_key, return_inverse=True)
            per_pass = np.zeros((upas.size, F_out), dtype=np.int64)
            np.add.at(per_pass, pinv, sa)
            prow = upas // pmax
            # running sum across the passes of each row (rows are contiguous and in pass order)
            run = np.cumsum(per_pass, axis=0)
            first = np.searchsorted(prow, prow)                    # first pass of each row
            base = np.where((first > 0)[:, None], run[np.maximum(first - 1, 0)], 0)
            ta_max = max(ta_max, int(np.abs(run - base).max()))
            np.add.at(acc, r0 + prow, per_pass)
        r0 = r1
    checks = {"shift_acc_max": sa_max, "shift_acc_overflow": sa_max >= 2 ** (SHIFT_ACC_BITS - 1),
              "tile_acc_max": ta_max, "tile_acc_overflow": ta_max >= 2 ** (TILE_ACC_BITS - 1)}
    return acc, checks


# --------------------------------------------------------------------------- 3-5
def rescale(acc: np.ndarray, layer: Layer) -> np.ndarray:
    """Combination rescale (Q-16): B̄ = clip(round(acc · α'_d · c_j), ±7)."""
    _check_product_fits(acc, layer.alpha_p, layer.c)
    num = acc * layer.alpha_p.codes[layer.deg_idx][:, None] * layer.c.codes[None, :]
    return np.clip(round_shift(num, layer.alpha_p.shift + layer.c.shift), -B_LEVEL, B_LEVEL)


def aggregate(B: np.ndarray, prog: Program) -> tuple[np.ndarray, dict]:
    """AUs (Q-25): psum_i = Σ_j A_ij B̄_j with binary A+I (fold, Q-11), 16-bit saturating.
    Saturation is order-dependent; if Σ_j |B̄_j| < 2^15 for every (i, column), no partial sum
    in ANY order can saturate and the exact sum is the hardware result. That bound is checked;
    if it fails the affected entries are reported (none so far: NELL's worst is 7·4550)."""
    g = prog.graph
    A = sp.csr_matrix((np.ones(g.dst_src.size, dtype=np.int64), g.dst_src, g.dst_ptr),
                      shape=(g.n, g.n))
    psum = A @ B
    bound = A @ np.abs(B)
    lim = 2 ** (AU_BITS - 1)
    return psum, {"au_psum_max": int(np.abs(psum).max()), "au_bound_max": int(bound.max()),
                  "au_may_saturate": int((bound >= lim).sum())}


def encode_next(psum: np.ndarray, layer: Layer, x_in: SparseRows | None = None) -> SparseRows:
    """Encoder QN units (Q-16/26): ReLU, X̄' = min(round(psum · e_d · γ_j), qmax_d).

    Residual layers (Q-36, ours): X̄' = min(round(ReLU(psum · e_d · γ_j + X̄ · r_d)), qmax_d),
    X̄ = this layer's own (decoded) input. The two products have different binary points;
    the one with fewer fractional bits is shifted left to align them before the add, then
    ReLU and one rounding. ReLU now acts on the sum, so it moves after the multiply."""
    _check_product_fits(np.maximum(psum, 0), layer.enc, layer.gamma)
    num = psum * layer.enc.codes[layer.deg_idx][:, None] * layer.gamma.codes[None, :]
    sh = layer.enc.shift + layer.gamma.shift
    if layer.res is not None:
        assert x_in is not None and x_in.F == psum.shape[1], "residual needs F_in = F_out"
        xr = x_in.to_dense() * layer.res.codes[layer.deg_idx][:, None]
        d = sh - layer.res.shift
        if d >= 0:
            xr = xr << d
        else:
            num, sh = num << -d, layer.res.shift
        num = num + xr
        assert float(np.abs(num).max(initial=0)) < 2.0 ** 53, "residual sum exceeds 2^53"
    q = round_shift(np.maximum(num, 0), sh)
    qmax = (1 << (layer.out_bits_table[layer.deg_idx] - 1)) - 1
    return SparseRows.from_dense(np.minimum(q, qmax[:, None]))


def run_layer(x: SparseRows, layer: Layer, prog: Program) -> LayerResult:
    x_dec, mem = store_and_reload(x, layer.bits)
    acc, checks = combination(x_dec, layer.bits, layer.W)
    B = rescale(acc, layer)
    psum, agg_checks = aggregate(B, prog)
    checks.update(agg_checks)
    if layer.last:
        return LayerResult(acc, B, psum, None, psum * layer.gamma.codes[None, :],
                           layer.gamma.shift, checks, mem)
    return LayerResult(acc, B, psum, encode_next(psum, layer, x_dec), None, None, checks, mem)


def run(prog: Program) -> list[LayerResult]:
    out, x = [], prog.x0
    for layer in prog.layers:
        res = run_layer(x, layer, prog)
        out.append(res)
        x = res.x_out
    return out
