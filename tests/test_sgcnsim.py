"""Tests for sgcnsim (SGCN in MEGA's Tab.V configuration). Run:
    python -m pytest tests/test_sgcnsim.py -q
Ramulator tests are skipped unless megasim/dram/build.sh has been run."""

import numpy as np
import pytest
import scipy.sparse as sp

from megasim.dram import ramulator as R
from megasim.timing import IdealDram
from sgcnsim import cache, energy, timing as T
from sgcnsim.workload import Workload

HW = T.HW()


def synthetic(n=300, dims=(200, 128, 5), deg=4, x0_density=0.02, h_density=0.5, seed=0):
    rng = np.random.default_rng(seed)
    src, dst = [], []
    for v in range(n):
        nb = rng.choice(n, size=deg, replace=False)
        src += [v, *nb]
        dst += [v] * (deg + 1)
    src, dst = np.array(src), np.array(dst)
    ptr = np.concatenate([[0], np.cumsum(np.bincount(dst, minlength=n))])
    x0 = sp.random(n, dims[0], density=x0_density, format="csr", random_state=seed)
    h = rng.random((n, dims[1])).astype(np.float32)
    h[rng.random(h.shape) > h_density] = 0
    return Workload("syn_test", n, list(dims), ptr, src, x0, [h])


def test_beicsr_layout_and_lines():
    lay = T.beicsr_layout(128, HW)
    assert [b for b in lay.bounds] == [(0, 96), (96, 128)]
    assert list(lay.bitmap) == [12, 4] and list(lay.off) == [0, 448] and lay.stride == 640
    nnz = np.array([[0, 0], [13, 15], [96, 32]])
    # bitmap is always read; 12 + 4·13 = 64 B is one line, 4 + 4·15 = 64 B one line
    assert T.slice_lines(lay, nnz, 3, HW).tolist() == [[1, 1], [1, 1], [7, 3]]
    assert T.slice_lines(T.dense_layout(128, HW), None, 1, HW).tolist() == [[6, 2]]


def test_w0_tile_choice():
    # Cora / CiteSeer / PubMed / NELL input widths against ¾ of the 392 KB cache (294 KB)
    assert [T.w0_tile(f, 128, HW) for f in (1433, 3703, 500, 61278)] == [32, 16, 128, 128]


def test_compiled_cache_matches_python():
    lines = np.random.default_rng(1).integers(0, 3000, 50000).astype(np.uint64)
    assert np.array_equal(cache.lru(lines, 37, 16), cache.lru_python(lines, 37, 16))


def test_plan_accounting():
    w = synthetic()
    plan = T.build_plan(w, HW)
    A, B, C = plan.phases
    # every line an engine reads is one cycle (one 64 B line per cycle per engine)
    for p in (B, C):
        assert p.engine_cycles.sum() == p.acc_addr.size
    # phase A: each X0 non-zero reads its W0 tile row (wt/16 lines) once per tile
    assert A.acc_addr.size == w.x0.nnz * plan.w0_tile // 16 * (128 // plan.w0_tile)
    # phase B: every edge reads the dense T row (6 + 2 lines)
    assert B.acc_addr.size == w.edges * 8
    # misses partition the accesses
    for p in plan.phases:
        assert p.miss_ptr[-1] == (p.hit == 0).sum() == p.miss_addr.size
    assert C.comb_cycles.sum() == w.n * 128 * 5 / HW.comb_macs


def test_schedule_traffic_and_ideal_bound():
    w = synthetic()
    plan = T.build_plan(w, HW)
    rep = T.schedule(plan, HW)
    miss = sum(int((p.hit == 0).sum()) for p in plan.phases)
    assert rep.dram.lines["sg_w"] + rep.dram.lines["sg_t_r"] + rep.dram.lines["sg_x_r"] == miss
    ideal = T.schedule(plan, HW, IdealDram(HW))
    assert ideal.total_cycles <= rep.total_cycles
    # compute bound: each phase lasts at least its busiest engine
    for ph in rep.phases:
        assert ph["cycles"] >= ph["busy"]["comb"]
    e = energy.energy(rep)
    assert set(e["uJ"]) == {"dram", "sram", "pu", "leakage"} and all(v > 0 for v in e["uJ"].values())


def test_backfill_server_books_in_time_order():
    d = T.BackfillDram(HW, [256.0], 0)
    assert d.xfer(100, {"x": 4}) == (100, 101)        # 4 lines = 256 B = 1 cycle
    assert d.xfer(10, {"x": 8}) == (10, 12)           # called later, issued earlier: not blocked
    assert d.xfer(9, {"x": 12}) == (9, 14)            # 9–10, then the gap after 12
    assert d.xfer(99, {"x": 8}) == (99, 102)          # splits around the booking at 100
    assert d.starts == [9, 99] and d.ends == [14, 102]
    half = T.BackfillDram(HW, [128.0], 5)             # half bandwidth, latency 5
    assert half.xfer(0, {"x": 4}) == (5, 7)


@pytest.mark.skipif(not (R.DRIVER.exists() and R.CONFIG.exists()), reason="run megasim/dram/build.sh")
def test_ramulator_calibrated_schedule():
    from sgcnsim.sim import simulate_ramulator
    w = synthetic()
    plan = T.build_plan(w, HW)
    rep, ram = simulate_ramulator(plan, HW)
    assert all(0 < e <= 1 for e in ram["phase_bw_efficiency"])
    assert ram["num_read_reqs"] + ram["num_write_reqs"] == 2 * ram["lines"]
    assert rep.total_cycles >= T.schedule(plan, HW, IdealDram(HW)).total_cycles
    # Ramulator, fed the schedule's own issue times, finishes the traffic about when it ends
    assert 0.8 < ram["replay_end_over_total"] < 1.2
