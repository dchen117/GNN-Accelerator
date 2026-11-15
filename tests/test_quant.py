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
                     m.weights[0].grad, m.x_quant[0].log_scale.grad, m.x_quant[0].bits.grad))
    for a, b in zip(*outs):
        assert torch.allclose(a, b, atol=1e-5), (a, b)
