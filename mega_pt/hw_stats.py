"""Hardware design numbers from a trained quantized model (input for simulator decisions).

Usage:
    python -m mega_pt.hw_stats runs/hw/gcn_cora [--seed 0] [--json out.json]

The run dir must hold model_seed<s>.pt (train.py --save-model). This script:

1. Re-runs the model as an **integer-only reference** (`int_forward`): the computation the
   hardware does, with integer X̄, W̄, B̄ and partial sums, and the scales applied only at
   the two requantization points:
     combination rescale (Q-16):  B̄  = clip(round(acc · α'_{d_i} · c_j), ±7)
         acc = X̄_i · W̄_{:,j} (integer), α'_d = α_d · s_d (GCN fold: s_d = d^-1/2, Q-11),
         c_j = β_j / γ_j (computed offline, per output column)
     Encoder requantization (Q-26): X̄' = clip(round(ReLU(psum) · e_{d_i} · γ_j), 0..qmax_{d_i})
         psum = (A+I) B̄ (integer AU sums), e_d = t_d / α^{l+1}_d (GCN t_d = d^-1/2,
         GraphSage 1/d, GIN 1), γ_j the B scale of this layer
     Residual layers (ours, Q-36): X̄' = clip(round(ReLU(psum · e_d · γ_j + X̄ · r_d)), 0..qmax_d)
         X̄ = this layer's own input, r_d = α^l_d / α^{l+1}_d (step ratio, per degree)
   With exact (float64) factors it must match the fake-quant model's predictions (checked
   and printed as `int_vs_fake_agree`). Then each factor is rounded to a fixed-point format
   to measure what the hardware scale format costs (Q-16):
     "fx<k>"   one shared binary point per table (plain fixed point), k-bit unsigned values
     "dy<k>"   per-entry k-bit mantissa + shift (dyadic, like integer-only inference in
               TFLite); costs a small shifter per entry.
2. Measures integer accumulator ranges (Q-22): adder tree, per-tile-pass Shifter-Acc
   (32 non-zeros or one 32-feature slice) and the full-row accumulator, including every
   running intermediate value, plus the AU partial sums (Q-25).
3. Estimates BSE utilization (Q-23) for the two ways of feeding a Combination Tile:
     "slice": each tile takes a fixed 32-feature slice of X (4 tiles = 128 features per
              pass); only that slice's non-zeros go to its 8 BSEs, in groups of 8.
     "nnz":   each pass takes the next ≤128 NON-ZEROS of the row (any feature positions)
              and splits them evenly over the 4 tiles (≤32 each), so groups of 8 are full
              except at the end of the row.
   Cycles per node (one 32-column output block) = Σ over passes of max over tiles of
   (groups × b), the bit-serial cost.

Everything here is our own analysis tooling; the numbers feed docs/mega-kb/04-open-questions.md.
"""

from __future__ import annotations

import argparse
import json
import math
from types import SimpleNamespace

import torch

from .data import load_dataset
from .metrics import accuracy
from .model import GraphOps, QGCN
from .quant import qmax_from_bits

TILES, C_SLICE, N_BSE = 4, 32, 8      # Q-21: 4 tiles, c = 32, n = 8
W_MAX, B_MAX_LVL = 7, 7               # 4-bit signed levels (Q-08, Q-09)


# --------------------------------------------------------------------------- loading
def load_run(run_dir: str, seed: int = 0):
    ck = torch.load(f"{run_dir}/model_seed{seed}.pt", weights_only=False)   # our own file
    a = SimpleNamespace(**ck["args"])
    assert a.norm == "fold", "integer reference implements the decided fold mode (Q-11) only"
    g = load_dataset(a.dataset, feat_norm=a.feat_norm).use_split(seed)
    if a.model == "sage":   # the same fixed evaluation sample as train.py
        ops = GraphOps.from_graph(g, "sage", a.samples, torch.Generator().manual_seed(seed))
    else:
        ops = GraphOps.from_graph(g, a.model)
    model = QGCN(ck["dims"], ck["max_degree"], norm=a.norm, frac_bits=a.frac_bits,
                 dropout=a.dropout, x_param=a.x_param, residual=getattr(a, "residual", False),
                 res_form=getattr(a, "res_form", "pre"))
    model.load_state_dict(ck["state"])
    model.quantize = True
    model.eval()
    return model, g, ops, a


# --------------------------------------------------------------------------- fixed point
def to_fixed(v: torch.Tensor, fmt: str | None) -> torch.Tensor:
    """Round positive factors v to a hardware format. fmt None = exact.
    fx<k>: shared binary point for the whole table, chosen so max(v) uses all k bits.
    dy<k>: per-entry k-bit mantissa in [2^(k-1), 2^k) and its own power-of-two shift."""
    if fmt is None:
        return v
    k = int(fmt[2:])
    v = v.double()
    if fmt.startswith("fx"):
        # fractional bits: largest entry < 2^k. (floor+1, not ceil: an exact power of two
        # would otherwise round to 2^k, a (k+1)-bit code. megasim.program mirrors this.)
        f = k - (math.floor(math.log2(v.max().item())) + 1)
        return torch.floor(v * 2.0 ** f + 0.5) / 2.0 ** f
    e = torch.floor(torch.log2(v)) - (k - 1)                      # per-entry exponent
    return torch.floor(v / 2.0 ** e + 0.5) * 2.0 ** e


def round_half_away(v: torch.Tensor) -> torch.Tensor:
    return torch.sign(v) * torch.floor(v.abs() + 0.5)


# --------------------------------------------------------------------------- integer model
def _xbar_coo(xbar: torch.Tensor):
    """(row, col, val) of the non-zeros, sorted by row then col."""
    if xbar.is_sparse:
        xbar = xbar.coalesce()
        r, c = xbar.indices()
        v = xbar.values()
    else:
        r, c = xbar.nonzero(as_tuple=True)
        v = xbar[r, c]
    keep = v != 0
    return r[keep], c[keep], v[keep].double()


def _dense(xbar: torch.Tensor) -> torch.Tensor:
    return (xbar.to_dense() if xbar.is_sparse else xbar).double()


def _matmul_int(r, c, v, Wbar, n):
    out = torch.zeros(n, Wbar.shape[1], dtype=torch.float64)
    step = max(1, 4_000_000 // Wbar.shape[1])          # chunk: Reddit has 1.4e8 non-zeros
    for i in range(0, r.numel(), step):
        out.index_add_(0, r[i:i + step], v[i:i + step, None] * Wbar[c[i:i + step]])
    return out


@torch.no_grad()
def int_forward(model: QGCN, x, ops: GraphOps, fmt: str | None = None, stats: dict | None = None):
    """Integer-only forward (fold mode). Returns logits (float, for argmax) and fills `stats`
    with per-layer integer ranges and the data needed for the utilization estimate."""
    deg = ops.deg
    s, t = ops.src_scale.double(), ops.dst_scale.double()
    a_bin = ops.a_bin.double()
    L = len(model.weights)
    xbar = model.x_quant[0](x, deg)[1]          # X^(0) quantized offline (preprocessing, Q-10)
    for l in range(L):
        r, c, v = _xbar_coo(xbar)
        _, Wbar = model.w_quant[l](model.weights[l])
        Wbar = Wbar.double()
        beta, gamma = model.w_quant[l].scale.double(), model.b_quant[l].scale.double()
        alpha = model.x_quant[l].scale.double()                         # per degree
        acc = _matmul_int(r, c, v, Wbar, deg.numel())                   # combination, integer
        a_prime = to_fixed((alpha[deg] * s), fmt)                       # degree-table factor
        cj = to_fixed(beta / gamma, fmt)                                # column factor
        Bbar = round_half_away(acc * a_prime[:, None] * cj[None, :]).clamp(-B_MAX_LVL, B_MAX_LVL)
        psum = torch.sparse.mm(a_bin, Bbar)                             # AUs, integer
        if stats is not None:
            stats[l] = {"r": r, "c": c, "v": v, "Wbar": Wbar, "acc": acc, "psum": psum,
                        "bits": model.x_quant[l].node_bits(deg, rounded=True), "F": model.dims[l],
                        "tables": {"alpha_prime": alpha[deg] * s, "c": beta / gamma, "gamma": gamma}}
        if l < L - 1:
            q_next = model.x_quant[l + 1]
            e = to_fixed(t / q_next.scale.double()[deg], fmt)
            g_fx = to_fixed(gamma, fmt)
            qmax = qmax_from_bits(q_next.node_bits(deg, rounded=True)).double()
            pre = psum * e[:, None] * g_fx[None, :]
            if l in model.residual_layers and model.res_form == "post":
                pre = pre.clamp_min(0)          # ReLU before the residual add (res_form post)
            if l in model.residual_layers:
                # residual: the layer's own input X̄^(l), moved from step α^l_d to α^{l+1}_d
                r_d = to_fixed(alpha[deg] / q_next.scale.double()[deg], fmt)
                pre = pre + _dense(xbar) * r_d[:, None]
                if stats is not None:
                    stats[l]["tables"]["res_deg"] = alpha[deg] / q_next.scale.double()[deg]
            xbar = torch.minimum(round_half_away(pre.clamp_min(0)), qmax[:, None])
            if stats is not None:
                stats[l]["tables"]["enc_deg"] = t / q_next.scale.double()[deg]
        else:
            # logits: the per-row factor t_i does not change argmax; γ_j does
            xbar = psum * to_fixed(gamma, fmt)[None, :]
    return xbar


# --------------------------------------------------------------------------- analyses
def accumulator_ranges(st: dict) -> dict:
    """Largest |value| each accumulator ever holds (running sums included), per layer."""
    r, c, v, Wbar, acc = st["r"], st["c"], st["v"], st["Wbar"], st["acc"]
    out = {"acc_full_max": acc.abs().max().item(), "psum_max": st["psum"].abs().max().item()}
    # Running (prefix) sums along each row, in feature order = the order the hardware visits
    # the non-zeros. Chunked over rows so Reddit fits in memory.
    run_max = pass_max = slice_max = nnz32_max = 0.0
    n = acc.shape[0]
    bounds = torch.searchsorted(r, torch.arange(0, n + 1))
    avg_nnz = max(1.0, r.numel() / max(n, 1))
    rows_per_chunk = max(1, int(4_000_000 / (avg_nnz * Wbar.shape[1])))   # ~32 MB per tensor
    for r0 in range(0, n, rows_per_chunk):
        lo, hi = bounds[r0].item(), bounds[min(n, r0 + rows_per_chunk)].item()
        if hi == lo:
            continue
        rr, cc = r[lo:hi], c[lo:hi]
        contrib = v[lo:hi, None] * Wbar[cc]                       # (nnz, F2)
        cs = contrib.cumsum(0)
        first = torch.searchsorted(rr, rr, right=False)              # first nnz of each row
        base = torch.where((first > 0)[:, None], cs[(first - 1).clamp_min(0)], torch.zeros_like(cs[:1]))
        run = cs - base                                              # running sum within row
        run_max = max(run_max, run.abs().max().item())
        # per 32-feature slice and per 128-feature pass (the "slice" scheme)
        for width, name in ((C_SLICE, "slice"), (C_SLICE * TILES, "pass")):
            key = rr * (10 ** 9) + torch.div(cc, width, rounding_mode="floor")
            _, inv = torch.unique_consecutive(key, return_inverse=True)
            part = torch.zeros(int(inv.max()) + 1, Wbar.shape[1], dtype=torch.float64).index_add_(0, inv, contrib)
            m = part.abs().max().item()
            if name == "slice":
                slice_max = max(slice_max, m)
            else:
                pass_max = max(pass_max, m)
        # per chunk of 32 consecutive non-zeros (the "nnz" scheme)
        rank = torch.arange(lo, hi) - bounds[rr].clone()
        key = rr * (10 ** 9) + torch.div(rank, C_SLICE, rounding_mode="floor")
        _, inv = torch.unique_consecutive(key, return_inverse=True)
        part = torch.zeros(int(inv.max()) + 1, Wbar.shape[1], dtype=torch.float64).index_add_(0, inv, contrib)
        nnz32_max = max(nnz32_max, part.abs().max().item())
    out.update(acc_running_max=run_max, tile_slice32_max=slice_max, pass128_max=pass_max,
               tile_nnz32_max=nnz32_max)
    return out


def bse_utilization(st: dict) -> dict:
    """Cycles per node and BSE utilization for one 32-column output block (Q-23/Q-24)."""
    r, c, bits, F = st["r"], st["c"], st["bits"].long(), st["F"]
    n = bits.numel()
    nnz = torch.bincount(r, minlength=n)
    # "slice": groups per (row, slice); a pass = 4 consecutive slices on the 4 tiles
    sl = torch.div(c, C_SLICE, rounding_mode="floor")
    key = r * (10 ** 6) + sl
    uk, cnt = torch.unique_consecutive(key, return_counts=True)
    groups = torch.div(cnt + N_BSE - 1, N_BSE, rounding_mode="floor")
    row = torch.div(uk, 10 ** 6, rounding_mode="floor")
    pas = torch.div(uk % (10 ** 6), TILES, rounding_mode="floor")
    pkey = row * (10 ** 6) + pas
    upk, pinv = torch.unique(pkey, return_inverse=True)
    pmax = torch.zeros(upk.numel(), dtype=torch.long).scatter_reduce_(0, pinv, groups, reduce="amax")
    prow = torch.div(upk, 10 ** 6, rounding_mode="floor")
    pcyc = (pmax * bits[prow]).clamp_min(1)             # a pass needs ≥1 cycle (bitmap read)
    slice_cyc = torch.zeros(n, dtype=torch.long).index_add_(0, prow, pcyc)
    n_pass = math.ceil(F / (C_SLICE * TILES))
    busy = torch.bincount(prow, minlength=n)              # passes with ≥1 non-zero
    slice_cyc = slice_cyc + (n_pass - busy)               # empty passes: 1 cycle each
    # "nnz": a pass takes the next ≤128 non-zeros and splits them evenly over the 4 tiles
    # (≤32 each); a tile needs ceil(share / 8) groups. Full passes: 4 groups.
    full = torch.div(nnz, C_SLICE * TILES, rounding_mode="floor")
    rem = nnz - full * C_SLICE * TILES
    share = torch.div(rem + TILES - 1, TILES, rounding_mode="floor")
    nnz_groups = full * (C_SLICE // N_BSE) + torch.div(share + N_BSE - 1, N_BSE, rounding_mode="floor")
    work = (nnz * bits).sum().item()                  # BSE-cycles of useful work
    out = {"density": nnz.sum().item() / (n * F), "mean_nnz": nnz.float().mean().item(),
           "mean_bits": bits.float().mean().item()}
    def put(name, cyc):
        cyc = cyc.sum().item()
        out[f"{name}_cycles_per_node"] = cyc / n
        out[f"{name}_bse_util"] = work / (cyc * TILES * N_BSE) if cyc else float("nan")
    put("slice", slice_cyc)                     # each tile reads its own 32-bit word per pass
    compute = nnz_groups * bits
    put("nnz_noscan", compute)                  # compute only (unlimited bitmap scan rate)
    # The decoder must scan the whole (32-bit padded) bitmap to find the non-zeros; scanning
    # node k+1 overlaps computing node k, so a node takes max(scan, compute) cycles.
    f_pad = math.ceil(F / 32) * 32
    for w in (64, 128, 256, 512):
        put(f"nnz_scan{w}", torch.clamp(compute, min=math.ceil(f_pad / w)))
    return out


def range_summary(v: torch.Tensor) -> dict:
    v = v[v > 0].double()
    return {"min": v.min().item(), "max": v.max().item(),
            "log2_span": math.log2(v.max().item() / v.min().item())}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_dir")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--formats", default="fx8,fx12,fx16,dy4,dy6,dy8,dy12",
                   help="fixed-point formats to test for the scale factors")
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)

    model, g, ops, args = load_run(a.run_dir, a.seed)
    with torch.no_grad():
        fake = model(g.x, ops)
    stats: dict = {}
    exact = int_forward(model, g.x, ops, None, stats)
    res = {"run": a.run_dir, "fake_test": accuracy(fake, g.y, g.test_idx),
           "int_exact_test": accuracy(exact, g.y, g.test_idx),
           "int_vs_fake_agree": (exact.argmax(1) == fake.argmax(1)).float().mean().item()}
    for fmt in a.formats.split(","):
        res[f"int_{fmt}_test"] = accuracy(int_forward(model, g.x, ops, fmt), g.y, g.test_idx)
    for l, st in stats.items():
        res[f"L{l}"] = {**accumulator_ranges(st), **bse_utilization(st),
                        **{f"range_{k}": range_summary(v) for k, v in st["tables"].items()}}
    print(json.dumps(res, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
