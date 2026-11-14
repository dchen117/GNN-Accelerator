"""Evaluation metrics: accuracy, average bits / CR, memory, and hardware-range checks.

Average-bits definitions (Q-15):
  avg_bits_sparse  (REPORTED, our Q-15 decision) — hidden layer inputs only (X^(1)..), zeros
                   cost 0 bits:  Σ_i nnz(X̄_i)·b_i / Σ_i dim.   CR = 32 / avg_bits_sparse.
  avg_bits_dense_all     dimension-weighted dense average over ALL quantized layer inputs
                         (incl. X^(0)); our earlier definition, logged for comparison.
  avg_bits_dense_hidden  dense average over hidden layer inputs only.

Hardware checks (Q-11 / Q-25):
  psum_max_abs     largest |integer partial sum| in the AUs (fold / per_source modes);
                   compare with 32767 (signed 16-bit, p.8 §V-D).
  b_lost_*         fraction of non-zero B entries that quantize to 0, over all source rows
                   and over the top-1%-degree rows (the "fold" risk).
"""

from __future__ import annotations

import torch

from .model import QGCN, GraphOps

INT16_MAX = 32767


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor, idx: torch.Tensor) -> float:
    return (logits[idx].argmax(1) == y[idx]).float().mean().item()


@torch.no_grad()
def quant_report(model: QGCN, x: torch.Tensor, ops: GraphOps) -> dict:
    """Eval-mode pass with recording; returns a flat dict of scalar metrics."""
    assert model.quantize, "quant_report needs the quantized model"
    model.eval()
    _, recs = model(x, ops, record=True)
    out: dict = {}

    num = {"all": 0.0, "hid": 0.0}      # Σ bits (dense)
    den = {"all": 0.0, "hid": 0.0}      # Σ elements
    sparse_num = 0.0
    top = ops.deg >= torch.quantile(ops.deg.float(), 0.99)  # top-1% degree nodes

    for l, r in enumerate(recs):
        xbar, nb = r["xbar"], r["node_bits"]          # (N,F), (N,)
        N, Fdim = xbar.shape
        dense_bits = (nb * Fdim).sum().item()
        num["all"] += dense_bits
        den["all"] += N * Fdim
        nnz_per_node = (xbar != 0).sum(1).float()
        out[f"L{l}_density"] = nnz_per_node.sum().item() / (N * Fdim)
        out[f"L{l}_mean_bits"] = nb.mean().item()
        if l >= 1:  # hidden layer inputs
            num["hid"] += dense_bits
            den["hid"] += N * Fdim
            sparse_num += (nnz_per_node * nb).sum().item()

        B, Bbar = r["B"], r["Bbar"]
        nz = B != 0
        lost = nz & (Bbar == 0)
        out[f"L{l}_b_lost_all"] = lost.sum().item() / max(nz.sum().item(), 1)
        out[f"L{l}_b_lost_topdeg"] = lost[top].sum().item() / max(nz[top].sum().item(), 1)
        if r["psum"] is not None:
            m = r["psum"].abs().max().item()
            out[f"L{l}_psum_max_abs"] = m
            out[f"L{l}_psum_overflow16"] = bool(m > INT16_MAX)

    out["avg_bits_sparse"] = sparse_num / den["hid"] if den["hid"] else float("nan")
    out["cr_sparse"] = 32.0 / out["avg_bits_sparse"] if out["avg_bits_sparse"] > 0 else float("inf")
    out["avg_bits_dense_all"] = num["all"] / den["all"]
    out["avg_bits_dense_hidden"] = num["hid"] / den["hid"] if den["hid"] else float("nan")
    out["memory_kb_eq4"] = model.memory_kb(ops.deg).item()      # continuous b (what the loss sees)
    out["memory_kb_rounded"] = num["all"] / 8 / 1024             # rounded b actually used
    return out
