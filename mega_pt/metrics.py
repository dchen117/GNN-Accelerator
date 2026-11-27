"""Evaluation metrics: accuracy, average bits / CR, memory, and hardware-range checks.

Average-bits definitions (Q-15):
  avg_bits_sparse  (REPORTED, our Q-15 decision) — hidden layer inputs only (X^(1)..), zeros
                   cost 0 bits:  Σ_i nnz(X̄_i)·b_i / Σ_i dim.   CR = 32 / avg_bits_sparse.
  avg_bits_dense_all     dimension-weighted dense average over ALL quantized layer inputs
                         (incl. X^(0)); our earlier definition, logged for comparison.
  avg_bits_dense_hidden  dense average over hidden layer inputs only.
  avg_bits_nz      (REPORTED from 2026-11-27) — hidden layer inputs only, average over the
                   NON-ZERO entries:  Σ_i nnz(X̄_i)·b_i / Σ_i nnz(X̄_i). "How many bits does a
                   stored value take", independent of sparsity. avg_bits_sparse = avg_bits_nz ×
                   hidden density, so avg_bits_sparse mixes bit width and ReLU sparsity (a
                   0.5-bit average is 3 bits × 17% density) and was hard to read. Older
                   results.json files lack it; res_table.bits_nz() recomputes it from
                   avg_bits_sparse and the L<l>_density fields (exact: hidden dims are equal).

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


def roc_auc(logits: torch.Tensor, y: torch.Tensor, idx: torch.Tensor) -> float:
    """ROC-AUC of class 1 for a binary task (Mann-Whitney U, ties count 1/2).
    Score = logit1 - logit0 (monotone in softmax p1, so the AUC is the same)."""
    s = (logits[idx, 1] - logits[idx, 0]).double()
    t = y[idx] == 1
    n1, n0 = int(t.sum()), int((~t).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = torch.argsort(s)
    ss = s[order]
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.double)
    # average ranks over ties
    uniq, inv, cnt = torch.unique_consecutive(ss, return_inverse=True, return_counts=True)
    if len(uniq) < len(ss):
        csum = torch.cumsum(cnt, 0).double()
        avg = csum - (cnt.double() - 1) / 2
        ranks[order] = avg[inv]
    return ((ranks[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)).item()


def score(logits: torch.Tensor, y: torch.Tensor, idx: torch.Tensor, metric: str = "acc") -> float:
    """The dataset's evaluation metric (Graph.metric): accuracy or ROC-AUC."""
    return roc_auc(logits, y, idx) if metric == "auc" else accuracy(logits, y, idx)


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
    nnz_hid = 0.0                       # Σ non-zero hidden-input entries
    top = ops.deg >= torch.quantile(ops.deg.float(), 0.99)  # top-1% degree nodes

    for l, r in enumerate(recs):
        nb, Fdim, nnz_per_node = r["node_bits"], r["F"], r["nnz_per_node"]   # (N,), int, (N,)
        N = nb.numel()
        dense_bits = (nb * Fdim).sum().item()
        num["all"] += dense_bits
        den["all"] += N * Fdim
        out[f"L{l}_density"] = nnz_per_node.sum().item() / (N * Fdim)
        out[f"L{l}_mean_bits"] = nb.mean().item()
        if l >= 1:  # hidden layer inputs
            num["hid"] += dense_bits
            den["hid"] += N * Fdim
            sparse_num += (nnz_per_node * nb).sum().item()
            nnz_hid += nnz_per_node.sum().item()

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
    out["avg_bits_nz"] = sparse_num / nnz_hid if nnz_hid else float("nan")
    out["avg_bits_dense_all"] = num["all"] / den["all"]
    out["avg_bits_dense_hidden"] = num["hid"] / den["hid"] if den["hid"] else float("nan")
    out["memory_kb_eq4"] = model.memory_kb(ops.deg).item()      # continuous b (what the loss sees)
    out["memory_kb_rounded"] = num["all"] / 8 / 1024             # rounded b actually used
    return out
