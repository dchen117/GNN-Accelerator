"""GCN / GIN / GraphSage with MEGA's Degree-Aware quantization (PyTorch "fake-quant" model).

All three models use Eq.(1)'s single-W form, X' = σ(Ã X W) (p.3 says the three "can be
uniformly represented" by it). Per Q-12 they differ only in Ã (see GraphOps.build):
  gcn   Ã = D^-1/2 (A+I) D^-1/2                         (Tab.III: Add aggregation)
  gin   Ã = A+I, i.e. GIN-0 (ε = 0) with ONE linear layer per GNN layer, sum aggregation
  sage  Ã = D_s^-1 (A_s+I): mean over the node itself plus up to 25 uniformly sampled
        in-neighbors (Tab.III: Mean, 25 samples); single W, no concat
Write s = source factor and t = destination factor, so Ã = diag(t) (A+I) diag(s).
The text below says d^-1/2 for GCN; for GIN s = t = 1, for GraphSage s = 1, t = 1/d.

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

NORM MODES (Q-11: decided "fold"). Written for GCN, Ã = D^-1/2 (A+I) D^-1/2 with d = in-degree
incl. self-loop. In code the source factor is ops.src_scale and the destination factor
ops.dst_scale, so the same modes serve GIN (no-op factors) and GraphSage (1/d in the Encoder).
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
from .quant import B_INIT, ColumnQuantizer, DegreeAwareQuantizer, fixed_point_q1

NORM_MODES = ("fp32", "fold", "per_source")


def _dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Dropout that also accepts a sparse COO x (NELL's X^(0)): drops stored non-zeros.

    Very large dense inputs (Reddit's 233K x 602 X^(0)) use a mask from torch.rand: the same
    dropout (keep with prob 1-p, scale by 1/(1-p)), but 2.5x faster on CPU, where
    F.dropout's bernoulli_ is effectively single-threaded (2.4 s vs 1.0 s per call). Smaller
    tensors keep F.dropout so earlier results stay bit-reproducible (same random stream)."""
    if not x.is_sparse:
        if training and p > 0 and x.numel() > 50_000_000:
            return x * (torch.rand_like(x) >= p) / (1 - p)
        return F.dropout(x, p, training)
    if not training or p == 0:
        return x
    x = x.coalesce()
    return torch.sparse_coo_tensor(x.indices(), F.dropout(x.values(), p, True), x.shape,
                                   is_coalesced=True, check_invariants=False)


def _matmul(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """X W for dense or sparse COO X (the Combination step, Eq.(3)).

    Sparse X is done as gather + scatter-add over the non-zeros instead of torch.sparse.mm:
    when X's values need a gradient (QAT: they depend on α and b), sparse.mm's backward
    materializes a dense N x F gradient -- 16 GB for NELL, which got the process OOM-killed.
    Here autograd memory is O(nnz x hidden) (~220 MB for NELL)."""
    if not x.is_sparse:
        return x @ W
    x = x.coalesce()
    row, col = x.indices()
    contrib = x.values().unsqueeze(1) * W[col]                 # (nnz, out)
    return torch.zeros(x.shape[0], W.shape[1], dtype=W.dtype).index_add(0, row, contrib)


def _nnz_per_node(xbar: torch.Tensor, n: int) -> torch.Tensor:
    """Number of non-zero quantized features in each node's row (for the sparse bits metric)."""
    if xbar.is_sparse:
        xbar = xbar.coalesce()
        rows = xbar.indices()[0][xbar.values() != 0]
        return torch.bincount(rows, minlength=n).float()
    return (xbar != 0).sum(1).float()
ETA = 8 * 1024  # Eq.(4): bits -> KB


MODELS = ("gcn", "gin", "sage")


def sample_neighbors(src: torch.Tensor, dst: torch.Tensor, num_nodes: int, k: int,
                     generator: torch.Generator | None = None,
                     large_graph_edges: int = 20_000_000) -> tuple[torch.Tensor, torch.Tensor]:
    """GraphSage neighbor sampling (Q-12): for every destination node keep at most k of its
    in-neighbors, chosen uniformly without replacement, plus its self-loop (always kept, so
    the sampled degree is min(d-1, k) + 1, which counts the self-loop as Q-01 does).

    Small graphs (E <= large_graph_edges; every MEGA dataset except Reddit): give each
    non-self edge a random key in [0, 1), sort by dst + key, keep the first k of each group.

    Large graphs (Reddit, 5e7-1e8 edges): that sort costs ~6 s and ~7 GB per epoch, so we use
    _sample_floyd instead: same distribution (exact uniform k-subsets), O(N·k) work. It needs
    the edges grouped by destination, which the large-graph loaders guarantee (CSC order,
    as the hardware stores the graph)."""
    self_loop = src == dst
    s, d = src[~self_loop], dst[~self_loop]
    loops = torch.arange(num_nodes)
    if d.numel() <= large_graph_edges:
        key = d.double() + torch.rand(d.numel(), generator=generator, dtype=torch.float64)
        order = torch.argsort(key)
        s, d = s[order], d[order]
        first = torch.searchsorted(d, d, right=False)          # index where each dst's group starts
        keep = (torch.arange(d.numel()) - first) < k
        return torch.cat([s[keep], loops]), torch.cat([d[keep], loops])
    assert bool((d[1:] >= d[:-1]).all()), "large graphs must have edges grouped by destination"
    pick = _sample_floyd(d, num_nodes, k, generator)
    return torch.cat([s[pick], loops]), torch.cat([d[pick], loops])


def _sample_floyd(d: torch.Tensor, num_nodes: int, k: int, generator) -> torch.Tensor:
    """Indices of the kept edges, for edges grouped by destination d.
    Nodes with ≤ k edges keep all of them. For the others, Floyd's algorithm draws a uniform
    random k-subset of the offsets 0..D-1, vectorized over nodes (k steps, one per element):
        for j = D-k .. D-1:  t = uniform{0..j};  add t, or j if t was already chosen."""
    count = torch.bincount(d, minlength=num_nodes)
    start = torch.cumsum(count, 0) - count
    keep_all = torch.nonzero(count[d] <= k).squeeze(1)
    hi = torch.nonzero(count > k).squeeze(1)
    D = count[hi]
    chosen = torch.empty(hi.numel(), k, dtype=torch.long)
    for j in range(k):
        jj = D - k + j                                         # current upper bound (inclusive)
        t = (torch.rand(hi.numel(), generator=generator, dtype=torch.float64) * (jj + 1)).long()
        t = torch.minimum(t, jj)                               # guard against rand() == 1 - eps rounding
        dup = (chosen[:, :j] == t.unsqueeze(1)).any(1)
        chosen[:, j] = torch.where(dup, jj, t)
    return torch.cat([keep_all, (start[hi].unsqueeze(1) + chosen).reshape(-1)])


@dataclass
class GraphOps:
    """Graph operators for one forward pass: Ã = diag(dst_scale) · (A+I) · diag(src_scale).

    GCN/GIN build this once. GraphSage builds one per sampled graph (train.py resamples every
    training epoch and uses one fixed sample, seeded by the run seed, for evaluation)."""

    deg: torch.Tensor        # (N,) int64, in-degree incl. self-loop (Q-01), of THIS graph
                             # (for GraphSage: the sampled degree, which is what the hardware's
                             # CSC column length and therefore its degree-table lookup sees)
    a_bin: torch.Tensor      # sparse (N,N) binary A+I, rows = destinations (what hardware stores)
    a_norm: torch.Tensor     # sparse (N,N) real-valued Ã (FP32 reference)
    src_scale: torch.Tensor  # (N,) source factor s_j: GCN d^-1/2, GIN 1, GraphSage 1
    dst_scale: torch.Tensor  # (N,) destination factor t_i: GCN d^-1/2, GIN 1, GraphSage 1/d

    @classmethod
    def build(cls, src: torch.Tensor, dst: torch.Tensor, num_nodes: int, model: str = "gcn") -> "GraphOps":
        assert model in MODELS, model
        deg = torch.bincount(dst, minlength=num_nodes)
        d = deg.float()
        if model == "gcn":
            s = t = d.pow(-0.5)
        elif model == "gin":                                 # GIN-0: plain sum over A+I
            s = t = torch.ones(num_nodes)
        else:                                                # sage: mean = (1/d) Σ over A+I
            s, t = torch.ones(num_nodes), 1.0 / d
        idx = torch.stack([dst, src])
        shape = (num_nodes, num_nodes)
        # Indices come from the loader (validated there) or are a subset of them (sampling).
        a_bin = torch.sparse_coo_tensor(idx, torch.ones(src.numel()), shape,
                                        check_invariants=False).coalesce()
        a_norm = torch.sparse_coo_tensor(idx, t[dst] * s[src], shape,
                                         check_invariants=False).coalesce()
        return cls(deg=deg, a_bin=a_bin, a_norm=a_norm, src_scale=s, dst_scale=t)

    @classmethod
    def from_graph(cls, g: Graph, model: str = "gcn", samples: int | None = None,
                   generator: torch.Generator | None = None) -> "GraphOps":
        """samples=k: GraphSage neighbor sampling (k neighbors + self), see sample_neighbors."""
        src, dst = g.edge_src, g.edge_dst
        if samples:
            src, dst = sample_neighbors(src, dst, g.num_nodes, samples, generator)
        return cls.build(src, dst, g.num_nodes, model)


class QGCN(nn.Module):
    """Eq.(1) GNN, no bias (Q-08): GCN, GIN-0 or GraphSage-mean depending on the GraphOps passed
    in (the layer itself is identical, Q-12). `quantize=False` gives the plain FP32 model used
    for pretraining. max_degree must cover ops.deg (GraphSage: samples + 1).

    dims = [F, hidden, ..., classes]. The input x may be dense or sparse COO (NELL);
    hidden activations are always dense. Layer l has a DegreeAwareQuantizer for its input
    (Q-03, Q-10: including X^(0)), a weight quantizer and a B quantizer.
    """

    def __init__(self, dims: list[int], max_degree: int, norm: str = "fp32",
                 frac_bits: int = 8, dropout: float = 0.5, x_param: str = "alpha"):
        super().__init__()
        assert norm in NORM_MODES, norm
        self.dims, self.norm, self.frac_bits, self.dropout = dims, norm, frac_bits, dropout
        self.quantize = False
        # Debug ablation (--quant-parts): which quantizers act in the forward pass when
        # quantize=True. "xwb" = all (the real model). Dropping a letter passes that tensor
        # through in float; its integer version is still computed for the metrics.
        self.quant_parts = "xwb"
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
        """w_j for per_source mode: Q1.f fixed point of the source factor (GCN d_j^-1/2;
        exactly 1.0 for GIN and GraphSage)."""
        return fixed_point_q1(ops.src_scale, self.frac_bits)

    def memory_kb(self, deg: torch.Tensor) -> torch.Tensor:
        """Eq.(4) inner term: (1/η) Σ_l Σ_i dim^l · b_i^l with CONTINUOUS b (Q-04),
        dense storage, over each layer's input X^(0..L-1) (Q-10)."""
        total = 0.0
        for l, q in enumerate(self.x_quant):
            total = total + self.dims[l] * q.node_bits(deg, rounded=False).sum()
        return total / ETA

    # ------------------------------------------------------------------ forward
    def _quant_b(self, l: int, t: torch.Tensor):
        tq, tbar = self.b_quant[l](t)
        return (tq if "b" in self.quant_parts else t), tbar

    def _aggregate(self, B: torch.Tensor, l: int, ops: GraphOps, rec: dict | None):
        """Quantize B (if quantizing) and aggregate according to the norm mode."""
        if not self.quantize:
            return torch.sparse.mm(ops.a_norm, B)
        s, t = ops.src_scale.unsqueeze(1), ops.dst_scale.unsqueeze(1)
        if self.norm == "fp32":
            Bq, Bbar = self._quant_b(l, B)
            H = torch.sparse.mm(ops.a_norm, Bq)
            psum = None  # no integer datapath defined for real-valued Ã
        elif self.norm == "fold":
            Bq, Bbar = self._quant_b(l, s * B)         # source factor before quantization
            H = t * torch.sparse.mm(ops.a_bin, Bq)     # destination factor in the Encoder
            psum = (lambda: torch.sparse.mm(ops.a_bin, Bbar)) if rec is not None else None
        else:  # per_source
            Bq, Bbar = self._quant_b(l, B)
            w = self.source_weights(ops).unsqueeze(1)
            H = t * torch.sparse.mm(ops.a_bin, w * Bq)
            if rec is not None:
                w_int = torch.floor(ops.src_scale * 2.0 ** self.frac_bits + 0.5).unsqueeze(1)
                psum = lambda: torch.sparse.mm(ops.a_bin, w_int * Bbar)  # integer AU sums
            else:
                psum = None
        if rec is not None:
            rec["B"], rec["Bbar"] = B.detach(), Bbar
            # Integer partial sums the AUs would hold (exact in float32 for these ranges).
            rec["psum"] = psum().detach() if psum is not None else None
        return H

    def forward(self, x: torch.Tensor, ops: GraphOps, record: bool = False, refit: bool = False):
        """Returns logits, plus a list of per-layer records when record=True.
        refit=True: re-fit each layer's clip ranges on its actual input first (refit_clips_)."""
        records = []
        L = len(self.weights)
        for l in range(L):
            rec = {} if record else None
            if self.quantize:
                if refit:
                    self.x_quant[l].refit_clip_(x, ops.deg)
                xq, xbar = self.x_quant[l](x, ops.deg)
                x = xq if "x" in self.quant_parts else x
                Wq, _ = self.w_quant[l](self.weights[l])
                W = Wq if "w" in self.quant_parts else self.weights[l]
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

    @torch.no_grad()
    def refit_clips_(self, x: torch.Tensor, ops: GraphOps) -> None:
        """MSE re-fit of every degree-aware clip range at the current rounded b (train.py
        --clip-refit). Runs one quantized eval pass, layer by layer, so layer l's range is fit
        on the input that the (already re-fit) quantized layers before it produce."""
        was_training = self.training
        self.eval()
        self(x, ops, refit=True)
        self.train(was_training)

    # ------------------------------------------------------------------ init
    @torch.no_grad()
    def calibrate(self, x: torch.Tensor, ops: GraphOps, x_init: str = "max",
                  init_bits: list[float] | None = None, wb_init: str = "max") -> None:
        """Initialize all scales from one FP32 eval pass (Q-04, Q-08, Q-09).

        Runs the FP32 network layer by layer and initializes, per layer:
          α table from the layer input X (per degree), β from W, γ from B as the
          selected norm mode will see it (fold quantizes D^-1/2 B, not B).
        x_init: "max" or "mse" clip-range init for the α tables (see
          DegreeAwareQuantizer.init_from_activations).
        init_bits: starting bitwidth per layer (all degrees); default B_INIT everywhere.
        wb_init: "max" or "mse" init for the W and B column scales (ColumnQuantizer).
        """
        was_training, was_quant = self.training, self.quantize
        self.eval()
        self.quantize = False
        L = len(self.weights)
        for l in range(L):
            self.x_quant[l].init_from_activations(x, ops.deg, x_init,
                                                 bits=init_bits[l] if init_bits else B_INIT)
            self.w_quant[l].init_from_tensor(self.weights[l], wb_init)
            B = _matmul(x, self.weights[l])
            B_seen = ops.src_scale.unsqueeze(1) * B if self.norm == "fold" else B
            self.b_quant[l].init_from_tensor(B_seen, wb_init)
            x = torch.sparse.mm(ops.a_norm, B)
            if l < L - 1:
                x = F.relu(x)
        self.train(was_training)
        self.quantize = was_quant
