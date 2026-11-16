"""Unit tests for the MEGA PyTorch model. Run:  .venv/bin/python -m pytest -q tests

Each test checks one property stated in the paper or one of our recorded decisions, so a
failure points at a specific equation / Q-ID.
"""

import torch

from mega_pt.data import Graph
from mega_pt.model import GraphOps, QGCN
from mega_pt.quant import (ColumnQuantizer, DegreeAwareQuantizer, eq2_quantize,
                           fixed_point_q1, qmax_from_bits)


def test_eq2_levels_signed_round_half_away():
    """Eq.(2): b=2 -> levels {-1,0,1}; b=4 -> ±7; ties round away from zero."""
    x = torch.tensor([-2.6, -0.5, -0.49, 0.0, 0.49, 0.5, 1.5, 9.0])
    _, xbar = eq2_quantize(x, torch.tensor(1.0), qmax_from_bits(torch.tensor(4.0)))
    assert xbar.tolist() == [-3, -1, 0, 0, 0, 1, 2, 7]
    _, xbar2 = eq2_quantize(x, torch.tensor(1.0), qmax_from_bits(torch.tensor(2.0)))
    assert xbar2.tolist() == [-1, -1, 0, 0, 0, 1, 1, 1]


def test_eq2_gradients_ste_and_lsq():
    """Q-05: d/dx = 1 inside the range, 0 when clipped; d/dα = round(v) - v inside."""
    x = torch.tensor([0.3, 0.8, 50.0], requires_grad=True)
    a = torch.tensor(0.5, requires_grad=True)
    xq, _ = eq2_quantize(x, a, torch.tensor(7.0))
    xq.sum().backward()
    assert x.grad.tolist() == [1.0, 1.0, 0.0]
    v = torch.tensor([0.6, 1.6])
    expected = (torch.floor(v + 0.5) - v).sum() + 7.0      # two inside + one clipped
    assert torch.isclose(a.grad, expected)


def test_eq3_integer_reformulation():
    """Eq.(3): (S_X X̄)(W̄ S_W) == (X̄ W̄) ⊙ (s_X ⊗ s_W)."""
    torch.manual_seed(0)
    xbar = torch.randint(-3, 4, (5, 6)).float()
    wbar = torch.randint(-7, 8, (6, 4)).float()
    sx, sw = torch.rand(5), torch.rand(4)
    lhs = (sx[:, None] * xbar) @ (wbar * sw[None, :])
    rhs = (xbar @ wbar) * torch.outer(sx, sw)
    assert torch.allclose(lhs, rhs, atol=1e-5)


def test_degree_table_lookup_and_bit_clamp():
    """Q-02/Q-04/Q-07: per-degree lookup; b clamped to [2, 8]; forward uses round(b)."""
    q = DegreeAwareQuantizer(max_degree=3)
    with torch.no_grad():
        q.bits.copy_(torch.tensor([4.0, 1.0, 2.6, 9.0]))
    q.clamp_()
    assert torch.allclose(q.bits, torch.tensor([4.0, 2.0, 2.6, 8.0]))
    deg = torch.tensor([1, 2, 3])
    assert q.node_bits(deg, rounded=True).tolist() == [2.0, 3.0, 8.0]
    assert torch.allclose(q.node_bits(deg, rounded=False), torch.tensor([2.0, 2.6, 8.0]))


def test_degree_init_no_clipping():
    """Q-04 init: α_d = max|x| of degree-d nodes / 7, so nothing clips at b=4."""
    q = DegreeAwareQuantizer(max_degree=3)
    x = torch.tensor([[1.0, -2.0], [0.5, 0.1], [7.0, 0.0]])
    deg = torch.tensor([1, 1, 3])
    q.init_from_activations(x, deg)
    assert torch.allclose(q.scale[1], torch.tensor(2.0 / 7))
    assert torch.allclose(q.scale[3], torch.tensor(1.0))
    assert torch.allclose(q.scale[2], torch.tensor(1.0))   # absent degree -> global max/7
    _, xbar = q(x, deg)
    assert xbar.abs().max() <= 7


def test_column_quantizer_4bit():
    """Q-08/Q-09: per-column scale, levels within ±7, column max maps to 7."""
    w = torch.tensor([[0.7, -0.2], [-1.4, 0.1]])
    cq = ColumnQuantizer(2)
    cq.init_from_tensor(w)
    _, wbar = cq(w)
    assert wbar.abs().amax(0).tolist() == [7.0, 7.0]


def test_q1f_fixed_point():
    """Q-11(b): Q1.f represents 1.0 exactly; tiny values round to 0."""
    w = torch.tensor([1.0, 0.5, 0.3, 0.001])
    assert fixed_point_q1(w, 4).tolist() == [1.0, 0.5, 0.3125, 0.0]


def _toy_graph():
    # 4 nodes, path 0-1-2-3, plus self-loops
    src = torch.tensor([0, 1, 1, 2, 2, 3, 0, 1, 2, 3])
    dst = torch.tensor([1, 0, 2, 1, 3, 2, 0, 1, 2, 3])
    deg = torch.bincount(dst)
    x = torch.randn(4, 5)
    idx = torch.arange(4)
    return Graph("toy", x, torch.tensor([0, 1, 0, 1]), src, dst, deg, idx, idx, idx)


def test_norm_modes_agree_without_quantization_error():
    """Q-11: fold and per_source compute the same Ã B as fp32 when nothing rounds.
    We force this by using huge-precision w (f=30) and comparing the un-quantized algebra."""
    g = _toy_graph()
    ops = GraphOps.from_graph(g)
    B = torch.randn(4, 3)
    d = ops.dinv_sqrt[:, None]
    ref = torch.sparse.mm(ops.a_norm, B)
    fold = d * torch.sparse.mm(ops.a_bin, d * B)
    w = fixed_point_q1(ops.dinv_sqrt, 30)[:, None]
    per_source = d * torch.sparse.mm(ops.a_bin, w * B)
    assert torch.allclose(ref, fold, atol=1e-6)
    assert torch.allclose(ref, per_source, atol=1e-6)


def test_model_runs_all_modes_and_records():
    g = _toy_graph()
    ops = GraphOps.from_graph(g)
    for norm in ("fp32", "fold", "per_source"):
        m = QGCN([5, 8, 2], int(g.deg.max()), norm=norm, frac_bits=6)
        m.calibrate(g.x, ops)
        m.quantize = True
        m.eval()
        logits, recs = m(g.x, ops, record=True)
        assert logits.shape == (4, 2)
        assert len(recs) == 2
        for r in recs:
            assert r["Bbar"].abs().max() <= 7
            assert (r["psum"] is None) == (norm == "fp32")


def test_sparse_input_matches_dense():
    """NELL path: a sparse COO X^(0) must give the same logits, metrics inputs and
    gradients (weights, α, b) as the same X stored densely."""
    g = _toy_graph()
    ops = GraphOps.from_graph(g)
    x_dense = g.x * (torch.rand_like(g.x) > 0.5)               # make it actually sparse
    x_sparse = x_dense.to_sparse_coo()
    outs = []
    for x in (x_dense, x_sparse):
        torch.manual_seed(0)
        m = QGCN([5, 8, 2], int(g.deg.max()), norm="fold")
        m.calibrate(x, ops)
        m.quantize = True
        m.eval()
        logits, recs = m(x, ops, record=True)
        logits.sum().backward()
        outs.append((logits.detach(), recs[0]["nnz_per_node"],
                     m.weights[0].grad, m.x_quant[0].log_p.grad, m.x_quant[0].bits.grad))
    for a, b in zip(*outs):
        assert torch.allclose(a, b, atol=1e-5), (a, b)


def test_range_param_keeps_clip_range_when_bits_drop():
    """x_param="range": α = c / q_max(round(b)), so lowering b keeps the largest
    representable value at c. "alpha" mode keeps α instead, so the range shrinks."""
    x = torch.tensor([[1.0, 0.0, 1.0]])
    deg = torch.tensor([1])
    for param, expect_max in (("range", 1.0), ("alpha", 1.0 / 7)):
        q = DegreeAwareQuantizer(max_degree=1, param=param)
        q.init_from_activations(x, deg)                      # b = 4: α = 1/7 in both modes
        assert torch.allclose(q.scale[1], torch.tensor(1.0 / 7))
        with torch.no_grad():
            q.bits.fill_(2.0)                                 # q_max = 1
        xq, _ = q(x, deg)
        assert torch.allclose(xq.max(), torch.tensor(expect_max)), param


def _mse(q, x, deg):
    xq, _ = q(x, deg)
    return ((x - xq) ** 2).sum().item()


def test_mse_init_beats_max_on_heavy_tailed_input():
    """Q-04 amendment: --x-init mse clips a few outliers so most values get a finer grid.
    On a heavy-tailed input its quantization error must be below max-based init's."""
    torch.manual_seed(0)
    x = torch.distributions.Exponential(1.0).sample((200, 16)) ** 2   # heavy tail
    deg = torch.randint(1, 4, (200,))
    errs = {}
    for method in ("max", "mse"):
        q = DegreeAwareQuantizer(max_degree=3)
        q.init_from_activations(x, deg, method)
        errs[method] = _mse(q, x, deg)
    assert errs["mse"] < 0.8 * errs["max"], errs


def test_refit_clip_follows_bitwidth_and_keeps_sparse_dense_equal():
    """refit_clip_ (--clip-refit): after b drops 4 -> 2, re-fitting the range at the new b must
    lower the error vs keeping the 4-bit range, give the range picked by an MSE init done
    directly at 2 bits, and treat a sparse input exactly like its dense copy.
    Uses param="range", the mode the re-fit is paired with (--x-param range --clip-refit K);
    the default is "alpha" since 2026-11-16."""
    torch.manual_seed(1)
    x = torch.relu(torch.randn(300, 8)) ** 2
    deg = torch.randint(1, 5, (300,))
    q = DegreeAwareQuantizer(max_degree=4, param="range")
    q.init_from_activations(x, deg, "mse")             # fit for b = 4
    with torch.no_grad():
        q.bits.fill_(2.0)
    stale = _mse(q, x, deg)
    q.refit_clip_(x, deg)
    assert _mse(q, x, deg) < stale
    direct = DegreeAwareQuantizer(max_degree=4, param="range")
    direct.init_from_activations(x, deg, "mse", bits=2)
    assert torch.allclose(q.log_p, direct.log_p)
    qs = DegreeAwareQuantizer(max_degree=4, param="range")
    qs.load_state_dict(q.state_dict())
    qs.refit_clip_(x.to_sparse(), deg)
    assert torch.allclose(qs.log_p, q.log_p)


def test_lsq_scaling_changes_only_the_scale_gradient():
    """--lsq: the forward pass is unchanged, the scale's gradient is multiplied by
    g_d = 1/sqrt(N_d · Q_P), and b's gradient is untouched."""
    torch.manual_seed(2)
    x = torch.randn(50, 6)
    deg = torch.randint(1, 4, (50,))
    grads, outs = {}, {}
    for lsq in (False, True):
        q = DegreeAwareQuantizer(max_degree=3)
        q.init_from_activations(x, deg, "max")
        q.lsq = lsq
        xq, _ = q(x, deg)
        (xq * torch.linspace(-1, 1, 6)).sum().backward()
        grads[lsq] = (q.log_p.grad.clone(), q.bits.grad.clone())
        outs[lsq] = xq.detach()
    assert torch.equal(outs[False], outs[True])
    n = torch.bincount(deg, minlength=4).float() * x.shape[1]
    g = 1.0 / torch.sqrt(n.clamp_min(1) * 7.0)                     # B_INIT = 4 -> Q_P = 7
    assert torch.allclose(grads[True][0], grads[False][0] * g, atol=1e-7)
    assert torch.allclose(grads[True][1], grads[False][1])
