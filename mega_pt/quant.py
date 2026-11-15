"""Quantizers for the MEGA Degree-Aware PyTorch model.

All quantizers implement Eq.(2) of the paper (p.5 §IV), a signed, round-half-away-from-zero,
clipped uniform quantizer:

    x̄ = sign(x) · min( ⌊|x|/α + 0.5⌋ , 2^(b−1) − 1 )          x_q = α · x̄

Decisions implemented here (docs/mega-kb/04-open-questions.md):
  Q-02  one (s, b) entry per integer degree 0..d_max.
  Q-03  one table per quantized layer input (each layer builds its own DegreeAwareQuantizer).
  Q-04  b continuous, clamped to [B_MIN, B_MAX]; forward uses round(b), backward is STE.
  Q-05  STE for rounding; α gets the LSQ-equivalent gradient from autograd, no grad scaling.
        Scales are learned in log space, by default as a clip range c with α = c/q_max(b)
        (implementation choices; see DegreeAwareQuantizer).
  Q-06  signed everywhere (X, W, B).
  Q-07  feature bitwidth range 2..8.
  Q-08  W: fixed 4 bit, per-column scale β.
  Q-09  B: fixed 4 bit, per-column scale γ.

Every quantizer returns BOTH the fake-quantized float tensor (used for training) and the
integer tensor x̄ (used for metrics and for checking hardware value ranges).
"""

from __future__ import annotations

import torch
import torch.nn as nn

B_MIN, B_MAX = 2.0, 8.0   # Q-07: feature bitwidth range
B_INIT = 4.0              # Q-04: initial bitwidth
W_BITS = 4                # Q-08 / Q-09: weights and B are fixed 4 bit
EPS = 1e-8                # floor for scales so |x|/α never divides by zero


def round_half_away_ste(v: torch.Tensor) -> torch.Tensor:
    """⌊v + 0.5⌋ for v >= 0 (the rounding in Eq.(2)), with a straight-through gradient.

    Forward: the rounded value. Backward: gradient 1, as if no rounding happened (Q-05).
    Only ever called on non-negative inputs (|x|/α, or b), where ⌊v+0.5⌋ is round-half-up.
    """
    return v + (torch.floor(v + 0.5) - v).detach()


def qmax_from_bits(b: torch.Tensor) -> torch.Tensor:
    """Largest integer level 2^(b−1) − 1 for (rounded) bitwidth b. Eq.(2)."""
    return torch.pow(2.0, b - 1.0) - 1.0


def eq2_quantize(x: torch.Tensor, alpha: torch.Tensor, qmax: torch.Tensor):
    """Eq.(2). alpha and qmax must broadcast against x.

    Returns (x_q, x_bar):
      x_q   = α · x̄, differentiable (STE through rounding; see Q-05)
      x_bar = integer levels as float, detached (for metrics / range checks)

    Gradients autograd produces from this formula:
      d x_q / d x : 1 inside the clip range, 0 when clipped
      d x_q / d α : sign(x)·(round(|x|/α) − |x|/α) inside, sign(x)·qmax when clipped (= LSQ)
      d x_q / d qmax : sign(x)·α when clipped, else 0 -> this is how the task loss reaches b
    """
    alpha = alpha.clamp_min(EPS)
    level = round_half_away_ste(x.abs() / alpha)
    level = torch.minimum(level, qmax)
    x_bar = torch.sign(x) * level
    return x_bar * alpha, x_bar.detach()


class DegreeAwareQuantizer(nn.Module):
    """Per-node quantizer whose (α, b) are looked up by node degree. Eq.(2), Q-02/Q-03/Q-04.

    Parameters (one entry per degree d = 0..d_max)
      log_p[d] : log of the learned scale parameter, see `param` below
      bits[d]  : continuous bitwidth b, clamped to [B_MIN, B_MAX]

    param = "range" (default): log_p = log c_d, the clipping range, and
                α_d = c_d / (2^(round(b_d)−1) − 1).
            "alpha": log_p = log α_d directly (our first version).
    Both give exactly Eq.(2) in the forward pass, and the hardware stores the same thing (one
    α per degree, computed from c and b after training). They differ only in what gradient
    descent sees. With "alpha", lowering b at fixed α shrinks the clip range 2× per bit: on
    Cora's binary features, b 4->2 at α = 1/7 maps every 1 to 1/7, so the task loss resists
    any bit reduction on layer 0 (where 2 bits would be lossless) and the memory penalty
    squeezes the hidden layer instead. With "range", lowering b keeps the range and only
    coarsens the grid. Measured comparison in mega_pt/README.md. Implementation choice,
    not from the paper (the paper learns "s" and b, Eq.(2), and does not say how).

    Why log space: Adam's step is ~lr in absolute terms. α is often ~0.01, so a linear α
    with lr 0.01 changes ~100% per step and training diverged in our first test. In log
    space the same lr is a ~1% relative change.

    Degree 0 has an entry like any other (Q-02). With self-loops counted (Q-01) it is never
    used, so it just keeps its initial value.
    """

    def __init__(self, max_degree: int, param: str = "range"):
        super().__init__()
        assert param in ("range", "alpha"), param
        self.param = param
        self.log_p = nn.Parameter(torch.zeros(max_degree + 1))
        self.bits = nn.Parameter(torch.full((max_degree + 1,), B_INIT))

    def alpha_table(self) -> torch.Tensor:
        """α per degree (shape d_max+1), differentiable w.r.t. log_p and (in "range" mode,
        through the STE-rounded q_max) w.r.t. b."""
        if self.param == "alpha":
            return self.log_p.exp()
        b = round_half_away_ste(self.bits.clamp(B_MIN, B_MAX))
        return self.log_p.exp() / qmax_from_bits(b)

    @property
    def scale(self) -> torch.Tensor:
        """α per degree (what the hardware would store)."""
        return self.alpha_table()

    @torch.no_grad()
    def init_from_activations(self, x: torch.Tensor, deg: torch.Tensor) -> None:
        """Q-04 init: α_d = max|x| over nodes of degree d / Q_max(B_INIT); b = B_INIT.
        (In "range" mode: c_d = max|x|, which gives the same α at b = B_INIT.)

        Degrees with no nodes get the layer-wide max. Max-based init means nothing is
        clipped at the start; the losses then push α and b down.
        """
        qmax = qmax_from_bits(torch.tensor(B_INIT)).item()
        if x.is_sparse:   # NELL's X^(0): max over each row's stored non-zeros
            x = x.coalesce()
            row_max = torch.zeros(deg.numel(), dtype=x.dtype)
            row_max.scatter_reduce_(0, x.indices()[0], x.values().abs(), reduce="amax")
        else:
            row_max = x.abs().amax(dim=1)                   # (N,)
        global_max = row_max.max().clamp_min(EPS)
        per_deg = torch.zeros_like(self.log_p)
        per_deg.scatter_reduce_(0, deg, row_max, reduce="amax", include_self=True)
        present = torch.bincount(deg, minlength=per_deg.numel()) > 0
        per_deg = torch.where(present & (per_deg > 0), per_deg, global_max)
        self.log_p.copy_((per_deg / qmax).log() if self.param == "alpha" else per_deg.log())
        self.bits.fill_(B_INIT)

    @torch.no_grad()
    def clamp_(self) -> None:
        """Call after every optimizer step: keep b in [B_MIN, B_MAX] (Q-04/Q-07).
        (α > 0 is automatic with the log parameterization.)"""
        self.bits.clamp_(B_MIN, B_MAX)

    def node_bits(self, deg: torch.Tensor, rounded: bool) -> torch.Tensor:
        """Per-node bitwidth b_{d_i}. Continuous for Eq.(4) (Q-04), rounded for the forward."""
        b = self.bits.clamp(B_MIN, B_MAX)[deg]
        return round_half_away_ste(b) if rounded else b

    def forward(self, x: torch.Tensor, deg: torch.Tensor):
        """Returns (x_q, x_bar). For a sparse COO x (NELL's X^(0), too big to store densely)
        only the stored non-zeros are quantized -- zeros stay zero under Eq.(2) anyway --
        and both outputs are sparse with the same pattern."""
        if x.is_sparse:
            x = x.coalesce()
            idx = x.indices()
            d = deg[idx[0]]                                           # degree of each nnz's row
            alpha = self.alpha_table()[d]
            qmax = qmax_from_bits(self.node_bits(d, rounded=True))
            xq, xbar = eq2_quantize(x.values(), alpha, qmax)
            # check_invariants=False: idx comes from an already-coalesced, validated tensor.
            return (torch.sparse_coo_tensor(idx, xq, x.shape, is_coalesced=True, check_invariants=False),
                    torch.sparse_coo_tensor(idx, xbar, x.shape, is_coalesced=True, check_invariants=False))
        alpha = self.alpha_table()[deg].unsqueeze(1)                  # (N, 1)
        qmax = qmax_from_bits(self.node_bits(deg, rounded=True)).unsqueeze(1)
        return eq2_quantize(x, alpha, qmax)


class ColumnQuantizer(nn.Module):
    """Fixed-bitwidth quantizer with one learnable scale per column. Used for W (scale β,
    Q-08) and for B = XW (scale γ, Q-09). Bitwidth 4 -> integer levels −7..+7."""

    def __init__(self, num_cols: int, bits: int = W_BITS):
        super().__init__()
        self.bits = bits
        self.log_scale = nn.Parameter(torch.zeros(num_cols))   # log-parameterized, see above
        self.register_buffer("qmax", qmax_from_bits(torch.tensor(float(bits))))

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.exp()

    @torch.no_grad()
    def init_from_tensor(self, t: torch.Tensor) -> None:
        """Q-08/Q-09 init: per-column max|t| / Q_max."""
        self.log_scale.copy_((t.abs().amax(dim=0).clamp_min(EPS) / self.qmax).log())

    @torch.no_grad()
    def clamp_(self) -> None:
        pass  # nothing to clamp: bitwidth fixed, scale positive by construction

    def forward(self, t: torch.Tensor):
        return eq2_quantize(t, self.scale.unsqueeze(0), self.qmax)


def fixed_point_q1(w: torch.Tensor, frac_bits: int) -> torch.Tensor:
    """Unsigned Q1.f fixed point (Q-11 option b): round(w·2^f)/2^f.

    Q1.f has one integer bit so 1.0 is exact (needed: degree-1 nodes have w = 1, and GIN /
    GraphSage load w = 1.0). Values below 2^-(f+1) round to 0 -- that loss is what the
    per-source experiments measure.
    """
    s = 2.0 ** frac_bits
    return torch.floor(w * s + 0.5) / s
