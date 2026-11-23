"""Unit tests for megasim M2: partition/relabel, unit cycle counts, traffic, schedule, and the
cycle-stepped tile check (S-03). Run: python -m pytest tests/test_megasim_timing.py -q"""

import numpy as np
import pytest

from megasim import functional, partition, tile_cycle, timing
from megasim.program import SparseRows
from megasim.synthetic import random_program


@pytest.fixture(scope="module")
def prog():
    return random_program(seed=0, n=700, dims=(300, 128, 7))


# --------------------------------------------------------------------------- partition
def test_partition_capacity_and_relabel(prog):
    part = partition.partition(prog)
    assert part.cap == 256
    assert np.diff(part.ptr).max() <= part.cap and part.ptr[-1] == prog.graph.n
    p2 = partition.relabel(prog, part.order)
    # every subgraph is a contiguous id range
    assert np.array_equal(np.repeat(np.arange(part.num_subgraphs), np.diff(part.ptr)),
                          part.part[part.order])
    # relabelling only permutes the results
    a, b = functional.run(prog), functional.run(p2)
    for ra, rb in zip(a, b):
        assert np.array_equal(ra.psum[part.order], rb.psum)


def test_partition_retries_until_parts_fit(prog):
    part = partition.partition(prog, cap=100)
    assert np.diff(part.ptr).max() <= 100 and part.num_subgraphs >= 7


# --------------------------------------------------------------------------- unit cycles
def test_comb_cycles_closed_form():
    hw = timing.HW()
    # nnz 0, 5, 32, 128, 130 at b = 4, F = 300 (scan = ⌈320/256⌉ = 2), F_out = 64 (2 blocks)
    x = np.zeros((5, 300), dtype=np.int64)
    for i, k in enumerate([0, 5, 32, 128, 130]):
        x[i, :k] = 1
    c = timing.comb_cycles(SparseRows.from_dense(x), np.full(5, 4), 64, hw)
    # waves on the busiest tile: 0, 1 (⌈5/4⌉=2 ≤ 8), 1 (8 each), 4, 4 + 1
    assert list(c) == [2, 1 * 4 * 2, 1 * 4 * 2, 4 * 4 * 2, 5 * 4 * 2]


@pytest.mark.parametrize("seed", range(3))
def test_tile_cycle_model_matches_closed_form(seed):
    """S-03: the cycle-stepped tiles give X̄·W̄ and exactly closed form + 3 drain cycles."""
    rng = np.random.default_rng(seed)
    n, F, F2 = 10, 300, 70
    bits = rng.integers(2, 9, n)
    lim = 2 ** (bits - 1) - 1
    x = rng.integers(-lim[:, None], lim[:, None] + 1, (n, F)) * (rng.random((n, F)) < rng.random((n, 1)))
    W = rng.integers(-7, 8, (F, F2)).astype(np.int8)
    xs = SparseRows.from_dense(x)
    acc, cyc, st = tile_cycle.simulate(xs, bits, W)
    assert np.array_equal(acc, x @ W.astype(np.int64))
    closed = timing.comb_cycles(xs, bits, F2, timing.HW(scan_bits=10 ** 9))
    closed[xs.nnz_per_row == 0] = 0
    assert cyc == closed.sum() + tile_cycle.DRAIN and st["xbar_stalls"] == 0


def test_tile_cycle_b1_is_crossbar_bound():
    xs = SparseRows.from_dense(-np.ones((1, 64), dtype=np.int64))
    _, cyc, st = tile_cycle.simulate(xs, np.array([1]), np.ones((64, 32), dtype=np.int8))
    assert st["xbar_stalls"] > 0 and cyc > 2 + tile_cycle.DRAIN


def test_fused_enc_comb_equals_recurrence():
    rng = np.random.default_rng(1)
    comb = rng.integers(1, 20, size=50)
    ptr = np.array([0, 20, 20, 50])
    e = 4
    got = timing.fused_enc_comb(comb, e, ptr)
    for k in range(3):
        enc_done = done = 0
        for c in comb[ptr[k]:ptr[k + 1]]:
            enc_done += e
            done = max(done, enc_done) + c
        assert got[k] == done


def test_agg_edges_per_cycle():
    hw = timing.HW()
    assert timing.agg_cycles_per_edge_group(128, hw) == (2, 1)
    assert timing.agg_cycles_per_edge_group(256, hw) == (1, 1)
    assert timing.agg_cycles_per_edge_group(7, hw) == (32, 1)
    assert timing.agg_cycles_per_edge_group(512, hw) == (1, 2)


# --------------------------------------------------------------------------- traffic
def test_lines_touched():
    assert list(timing.lines_touched(np.array([0, 0, 500, 512, 7]), np.array([0, 512, 24, 1, 1024]),
                                     512)) == [0, 1, 2, 1, 3]


def test_condense_traffic_conserves_rows(prog):
    part = partition.partition(prog)
    p2 = partition.relabel(prog, part.order)
    K = part.num_subgraphs
    sub = np.repeat(np.arange(K), np.diff(part.ptr))
    hw = timing.HW()
    se = timing.subgraph_edges(p2.graph, sub, K, 128, hw)
    ct = timing.condense_traffic(se, sub, K, 128, hw)      # asserts every row reaches its area
    rows = se.ext_rows.sum()
    # each row is 64 B = 1 line; partial flushes can only add lines
    assert ct.flush_lines_by_src_sub.sum() + ct.end_flush_lines >= rows
    assert ct.read_lines.sum() == rows
    # external pairs are exactly the (target, source) pairs with different subgraphs
    g = p2.graph
    src = np.repeat(np.arange(g.n), np.diff(g.src_ptr))
    ext = sub[src] != sub[g.src_dst]
    assert rows == np.unique(sub[g.src_dst[ext]] * g.n + src[ext]).size


def test_weight_row_cache():
    hw = timing.HW(weight_buf=4 * 64 + 2 * 5 + 128 * 4)   # tables (2 degrees, 128 cols) + 4 rows of 64 B
    from megasim.synthetic import random_program as rp
    p = rp(seed=2, n=50, dims=(40, 128, 7))
    lay = p.layers[0]
    x = SparseRows.from_dense(np.array([[1, 1, 0, 0, 1, 0] + [0] * 34,     # rows 0,1,4
                                        [1, 0, 0, 0, 1, 0] + [0] * 34]))   # rows 0,4
    lay = type(lay)(**{**lay.__dict__, "deg_values": np.array([1, 2])})
    w = timing.weight_traffic(x, lay, np.array([0, 2]), hw)
    assert not w["resident"] and w["slots"] == 4
    # slot 0: rows 0, 4, 0, 4 → all four miss; slot 1: row 1 → miss
    assert w["misses"] == 5


# --------------------------------------------------------------------------- schedule
def test_simulate_runs_and_bounds(prog):
    part = partition.partition(prog)
    p2 = partition.relabel(prog, part.order)
    func = functional.run(p2)
    hw = timing.HW()
    rep = timing.simulate(p2, part.ptr, func, hw)
    assert len(rep.sweeps) == len(p2.layers) + 1
    for s in rep.sweeps:
        # a sweep lasts at least as long as its busiest unit and its DRAM time
        assert s.cycles >= max(s.busy.values()) and s.cycles >= s.dram_busy
    # infinite bandwidth and no latency can only make it faster
    fast = timing.simulate(p2, part.ptr, func, timing.HW(dram_bytes_per_cycle=1e12, dram_latency=0))
    assert fast.total_cycles <= rep.total_cycles
    assert rep.dram_bytes == fast.dram_bytes            # traffic does not depend on timing


# --------------------------------------------------------------------------- S-15 X̄⁽⁰⁾ format
def test_x0_position_format(prog):
    """X̄⁽⁰⁾ non-zero positions in DRAM: F-bit bitmaps (Q-19) or 16-bit column indices,
    "auto" taking the smaller. Only the DRAM stream changes, never the timing."""
    part = partition.partition(prog)
    p2 = partition.relabel(prog, part.order)
    x = p2.x0
    hw_b, hw_i, hw_a = (timing.HW(x0_format=f) for f in ("bitmap", "index", "auto"))
    nb, lb = timing.x0_position_stream(x, part.ptr, hw_b)
    ni, li = timing.x0_position_stream(x, part.ptr, hw_i)
    assert (nb, ni) == ("x0_bitmap", "x0_index")
    # 16 bits per non-zero, contiguous: lines ≈ 2·nnz / 64 (+ boundary lines per subgraph)
    assert 2 * x.val.size / 64 <= li.sum() <= 2 * x.val.size / 64 + part.num_subgraphs
    na, la = timing.x0_position_stream(x, part.ptr, hw_a)
    assert la.sum() == min(lb.sum(), li.sum())
    func = functional.run(p2)
    rb = timing.simulate(p2, part.ptr, func, hw_b, dram=timing.IdealDram(hw_b))
    ri = timing.simulate(p2, part.ptr, func, hw_i, dram=timing.IdealDram(hw_i))
    assert rb.total_cycles == ri.total_cycles
    assert rb.dram_bytes["x0_bitmap"] == lb.sum() * 64 and ri.dram_bytes["x0_index"] == li.sum() * 64
