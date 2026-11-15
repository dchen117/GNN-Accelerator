"""GCN with MEGA's Degree-Aware quantization (PyTorch "fake-quant" model).

Layer l computes, in MEGA's execution order Ã(XW) (p.3 §II-A):

    X_q  = DegreeAware_l(X)          Eq.(2), per-node (α, b) looked up by degree   [Encoder]
    W_q  = Column4bit_l(W)           Eq.(2), 4 bit, per-column β                    (Q-08)
    B    = X_q W_q                   = (X̄ W̄) ⊙ (s_X ⊗ s_W), Eq.(3)               [Combination]
    B_q  = Column4bit'_l(B)          4 bit, per-column γ                            (Q-09)
    H    = Ã B_q                     aggregation, see NORM MODES                    [Aggregation]
    X'   = ReLU(H)                   (not on the last layer, Q-10)

Fake-quant in float gives the same numbers as the integer hardware path as long as no
extra rounding happens between the quantization points (float error aside). The scale
multiplications (α·β, γ, d^-1/2) are treated as exact; their hardware precision is open
(Q-16).

NORM MODES (Q-11, pending test). Ã = D^-1/2 (A+I) D^-1/2 with d = in-degree incl. self-loop.
  "fp32"       H = Ã · Q(B)
               Ideal reference: real-valued edge weights. Not what our hardware does.
  "fold"       H = D^-1/2 · (A+I) · Q(D^-1/2 B)
               Source factor folded into the combination rescale BEFORE B is quantized,
               destination factor folded into the Encoder. Binary A, zero extra hardware.
               Risk: rows of high-degree sources shrink by 1/sqrt(d) and may round to 0.
  "per_source" H = D^-1/2 · (A+I) · (w ⊙ Q(B)),  w = Q1.f fixed point of d^-1/2
               B quantized unscaled; each AU multiplies a source row by w_j once and reuses
               the product for all of j's out-edges. Risk: w precision, 16-bit overflow.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Graph
from .quant import ColumnQuantizer, DegreeAwareQuantizer, fixed_point_q1

NORM_MODES = ("fp32", "fold", "per_source")


def _dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Dropout that also accepts a sparse COO x (NELL's X^(0)): drops stored non-zeros."""
    if not x.is_sparse:
        return F.dropout(x, p, training)
    if not training or p == 0:
        return x
    x = x.coalesce()
    return torch.sparse_coo_tensor(x.indices(), F.dropout(x.values(), p, True), x.shape,
                                   is_coalesced=True, check_invariants=False)


def _matmul(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """X W for dense or sparse COO X (the Combination step, Eq.(3))."""
    return torch.sparse.mm(x, W) if x.is_sparse else x @ W


def _nnz_per_node(xbar: torch.Tensor, n: int) -> torch.Tensor:
    """Number of non-zero quantized features in each node's row (for the sparse bits metric)."""
    if xbar.is_sparse:
        xbar = xbar.coalesce()
        rows = xbar.indices()[0][xbar.values() != 0]
        return torch.bincount(rows, minlength=n).float()
    return (xbar != 0).sum(1).float()
ETA = 8 * 1024  # Eq.(4): bits -> KB


@dataclass
class GraphOps:
    """Precomputed graph operators shared by all layers and epochs."""

    deg: torch.Tensor        # (N,) int64, in-degree incl. self-loop (Q-01)
    a_bin: torch.Tensor      # sparse (N,N) binary A+I, rows = destinations
    a_norm: torch.Tensor     # sparse (N,N) D^-1/2 (A+I) D^-1/2
    dinv_sqrt: torch.Tensor  # (N,) d^-1/2

    @classmethod
    def from_graph(cls, g: Graph) -> "GraphOps":
        dinv = g.deg.float().pow(-0.5)
        vals = dinv[g.edge_dst] * dinv[g.edge_src]
        return cls(deg=g.deg, a_bin=g.adjacency(), a_norm=g.adjacency(vals), dinv_sqrt=dinv)


class QGCN(nn.Module):
    """GCN, no bias (Q-08). `quantize=False` gives the plain FP32 GCN used for pretraining.

    dims = [F, hidden, ..., classes]. The input x may be dense or sparse COO (NELL);
    hidden activations are always dense. Layer l has a DegreeAwareQuantizer for its input
    (Q-03, Q-10: including X^(0)), a weight quantizer and a B quantizer.
    """

    def __init__(self, dims: list[int], max_degree: int, norm: str = "fp32",
                 frac_bits: int = 8, dropout: float = 0.5, x_param: str = "range"):
        super().__init__()
        assert norm in NORM_MODES, norm
        self.dims, self.norm, self.frac_bits, self.dropout = dims, norm, frac_bits, dropout
        self.quantize = False
        L = len(dims) - 1
        self.weights = nn.ParameterList(
            [nn.Parameter(nn.init.xavier_uniform_(torch.empty(dims[l], dims[l + 1]))) for l in range(L)])
        self.x_quant = nn.ModuleList([DegreeAwareQuantizer(max_degree, x_param) for _ in range(L)])
        self.w_quant = nn.ModuleList([ColumnQuantizer(dims[l + 1]) for l in range(L)])
        self.b_quant = nn.ModuleList([ColumnQuantizer(dims[l + 1]) for l in range(L)])

    # ------------------------------------------------------------------ helpers
    def weight_params(self):
        return list(self.weights)

    def quant_params(self):
        return [p for m in (*self.x_quant, *self.w_quant, *self.b_quant) for p in m.parameters()]

    def alpha_params(self):
        """log α (or log clip range) tables of the degree-aware quantizers."""
        return [q.log_p for q in self.x_quant]

    def bit_params(self):
        """continuous b tables of the degree-aware quantizers."""
        return [q.bits for q in self.x_quant]

    def column_scale_params(self):
        """log β (weights) and log γ (B) per-column scales."""
        return [q.log_scale for q in (*self.w_quant, *self.b_quant)]

    @torch.no_grad()
    def clamp_quant_params_(self):
        for m in (*self.x_quant, *self.w_quant, *self.b_quant):
            m.clamp_()

    def source_weights(self, ops: GraphOps) -> torch.Tensor:
        """w_j for per_source mode: Q1.f fixed point of d_j^-1/2."""
        return fixed_point_q1(ops.dinv_sqrt, self.frac_bits)

    def memory_kb(self, deg: torch.Tensor) -> torch.Tensor:
        """Eq.(4) inner term: (1/η) Σ_l Σ_i dim^l · b_i^l with CONTINUOUS b (Q-04),
        dense storage, over each layer's input X^(0..L-1) (Q-10)."""
        total = 0.0
        for l, q in enumerate(self.x_quant):
            total = total + self.dims[l] * q.node_bits(deg, rounded=False).sum()
        return total / ETA

    # ------------------------------------------------------------------ forward
    def _aggregate(self, B: torch.Tensor, l: int, ops: GraphOps, rec: dict | None):
        """Quantize B (if quantizing) and aggregate according to the norm mode."""
        if not self.quantize:
            return torch.sparse.mm(ops.a_norm, B)
        d = ops.dinv_sqrt.unsqueeze(1)
        if self.norm == "fp32":
            Bq, Bbar = self.b_quant[l](B)
            H = torch.sparse.mm(ops.a_norm, Bq)
            psum = None  # no integer datapath defined for real-valued Ã
        elif self.norm == "fold":
            Bq, Bbar = self.b_quant[l](d * B)          # source factor before quantization
            H = d * torch.sparse.mm(ops.a_bin, Bq)     # destination factor in the Encoder
            psum = (lambda: torch.sparse.mm(ops.a_bin, Bbar)) if rec is not None else None
        else:  # per_source
            Bq, Bbar = self.b_quant[l](B)
            w = self.source_weights(ops).unsqueeze(1)
            H = d * torch.sparse.mm(ops.a_bin, w * Bq)
            if rec is not None:
                w_int = torch.floor(ops.dinv_sqrt * 2.0 ** self.frac_bits + 0.5).unsqueeze(1)
                psum = lambda: torch.sparse.mm(ops.a_bin, w_int * Bbar)  # integer AU sums
            else:
                psum = None
        if rec is not None:
            rec["B"], rec["Bbar"] = B.detach(), Bbar
            # Integer partial sums the AUs would hold (exact in float32 for these ranges).
            rec["psum"] = psum().detach() if psum is not None else None
        return H

    def forward(self, x: torch.Tensor, ops: GraphOps, record: bool = False):
        """Returns logits, plus a list of per-layer records when record=True."""
        records = []
        L = len(self.weights)
        for l in range(L):
            rec = {} if record else None
            if self.quantize:
                x, xbar = self.x_quant[l](x, ops.deg)
                W, _ = self.w_quant[l](self.weights[l])
                if rec is not None:
                    rec["F"] = x.shape[1]
                    rec["nnz_per_node"] = _nnz_per_node(xbar, x.shape[0])
                    rec["node_bits"] = self.x_quant[l].node_bits(ops.deg, rounded=True).detach()
            else:
                W = self.weights[l]
            # Dropout AFTER the quantizer: the quantizer then sees the same value
            # distribution in training and eval (dropout's 1/(1-p) rescale would otherwise
            # bias the learned α). Our choice; the paper doesn't say.
            x = _dropout(x, self.dropout, self.training)
            B = _matmul(x, W)                            # combination, Eq.(3)
            x = self._aggregate(B, l, ops, rec)          # aggregation
            if l < L - 1:
                x = F.relu(x)                            # Encoder ReLU (not on logits, Q-10)
            if record:
                records.append(rec)
        return (x, records) if record else x

    # ------------------------------------------------------------------ init
    @torch.no_grad()
    def calibrate(self, x: torch.Tensor, ops: GraphOps) -> None:
        """Initialize all scales from one FP32 eval pass (Q-04, Q-08, Q-09).

        Runs the FP32 network layer by layer and initializes, per layer:
          α table from the layer input X (per degree), β from W, γ from B as the
          selected norm mode will see it (fold quantizes D^-1/2 B, not B).
        """
        was_training, was_quant = self.training, self.quantize
        self.eval()
        self.quantize = False
        L = len(self.weights)
        for l in range(L):
            self.x_quant[l].init_from_activations(x, ops.deg)
            self.w_quant[l].init_from_tensor(self.weights[l])
            B = _matmul(x, self.weights[l])
            B_seen = ops.dinv_sqrt.unsqueeze(1) * B if self.norm == "fold" else B
            self.b_quant[l].init_from_tensor(B_seen)
            x = torch.sparse.mm(ops.a_norm, B)
            if l < L - 1:
                x = F.relu(x)
        self.train(was_training)
        self.quantize = was_quant
