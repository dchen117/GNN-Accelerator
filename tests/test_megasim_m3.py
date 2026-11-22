"""Tests for megasim M3: Ramulator replay (S-13), SRAM accesses and energy (S-14).
Run: python -m pytest tests/test_megasim_m3.py -q
The Ramulator tests are skipped unless megasim/dram/build.sh has been run."""

import numpy as np
import pytest

from megasim import cacti, energy, functional, partition, timing
from megasim.dram import ramulator as R
from megasim.synthetic import random_program

HW = timing.HW()
needs_driver = pytest.mark.skipif(not (R.DRIVER.exists() and R.CONFIG.exists()),
                                  reason="run megasim/dram/build.sh first")


@pytest.fixture(scope="module")
def setup():
    p = random_program(seed=0, n=700, dims=(300, 128, 7))
    part = partition.partition(p)
    p2 = partition.relabel(p, part.order)
    return p2, part.ptr, functional.run(p2)


# --------------------------------------------------------------------------- trace
def test_trace_preserves_lines_and_order():
    log = [(5.0, {"eid": 0.25, "csc": 3.0}, {}), (2.0, {"eid": 0.25}, {}),
           (9.0, {"eid": 0.5, "sparse_w": 2.0}, {}),
           (9.0, {"w_miss": 2.0}, {"w_miss": np.array([640, 64])})]
    tr = R.trace_from_log(log, HW)
    # fractional eID lines accumulate to exactly 1 line overall
    assert tr.addr.size == 3 + 1 + 2 + 2
    assert list(tr.batch_t) == sorted(tr.batch_t) and list(tr.order) == [1, 0, 2, 3]
    # sequential stream: consecutive lines inside its own region
    csc = tr.addr[(tr.addr >> R.REGION_BITS) == R.REGION["csc"]]
    assert np.array_equal(np.diff(csc.astype(np.int64)), [64, 64])
    # explicit addresses kept, writes flagged
    w = tr.addr[(tr.addr >> R.REGION_BITS) == R.REGION["w_miss"]] - (R.REGION["w_miss"] << R.REGION_BITS)
    assert list(w) == [640, 64]
    assert tr.write.sum() == 2


def test_latency_dram_uses_measured_latencies():
    d = timing.LatencyDram(HW, np.array([10.0, 0.0, 7.0]), np.array([30.0, 0.0, 9.0]))
    assert d.xfer(100, {"csc": 4}) == (110, 130)
    assert d.xfer(50, {}) == (50, 50)                    # empty transfer
    assert d.xfer(200, {"logits": 1}) == (207, 209)
    assert d.free == 209 and len(d.log) == 3


# --------------------------------------------------------------------------- Ramulator
@needs_driver
def test_hbm_bandwidth_sequential_and_random():
    n = 16384                                            # 1 MB of 64 B lines
    tr = R.trace_from_log([(0.0, {"x0_pkg": float(n)}, {})], HW)
    _, last, st = R.replay(tr)
    ideal = n * 64 / HW.dram_bytes_per_cycle
    assert ideal <= last[0] < ideal / 0.85               # sequential: ≥ 85 % of 256 GB/s
    assert st["row_hits"] / (st["num_read_reqs"]) > 0.95
    rng = np.random.default_rng(0)
    a = rng.integers(0, 1 << 22, n) * 64
    tr = R.trace_from_log([(0.0, {"w_miss": float(n)}, {"w_miss": a})], HW)
    _, last_r, _ = R.replay(tr)
    assert last_r[0] > 1.5 * last[0]                     # scattered lines are much slower


@needs_driver
def test_simulate_ramulator_converges_and_keeps_traffic(setup):
    p, ptr, func = setup
    m2 = timing.simulate(p, ptr, func, HW)
    rep = R.simulate_ramulator(p, ptr, func, HW)
    info = rep.info["ramulator"]
    assert info["converged"]
    assert rep.dram_bytes == m2.dram_bytes               # bytes come from the traffic model
    assert info["num_read_reqs"] + info["num_write_reqs"] == 2 * info["lines"]
    for s in rep.sweeps:
        assert s.cycles >= max(s.busy.values())
    # a DRAM that answers instantly can only be faster (the schedule is max-plus monotone)
    ideal = timing.simulate(p, ptr, func, HW, dram=timing.IdealDram(HW))
    assert ideal.total_cycles <= rep.total_cycles and ideal.total_cycles <= m2.total_cycles


# --------------------------------------------------------------------------- energy
def test_sram_accesses_consistent(setup):
    p, ptr, func = setup
    rep = timing.simulate(p, ptr, func, HW)
    a = rep.info["sram"]
    assert a["input_buf"]["read"] == a["input_buf"]["write"] > 0
    assert a["sparse_buf"]["read"] == a["sparse_buf"]["write"]
    # every edge does a read-modify-write of F2 partial sums
    e = sum(timing.subgraph_edges(p.graph, np.repeat(np.arange(ptr.size - 1), np.diff(ptr)),
                                  ptr.size - 1, l.F_out, HW).edges.sum() * l.F_out * 2
            for l in p.layers)
    assert a["agg_buf"]["write"] == e


def test_energy_categories(setup):
    p, ptr, func = setup
    rep = timing.simulate(p, ptr, func, HW)
    e = energy.energy(rep)
    assert set(e["uJ"]) == {"dram", "sram", "pu", "leakage"}
    assert all(v > 0 for v in e["uJ"].values())
    assert e["uJ"]["dram"] == pytest.approx(sum(rep.dram_bytes.values()) * 8 * 7.0 * 1e-6)
    assert e["uJ"]["leakage"] == pytest.approx(e["sram_leak_mW"] * rep.total_cycles * 1e-6)
    assert abs(sum(e["share"].values()) - 1) < 1e-9


def test_cacti_cache_matches_tab_iv_sizes():
    b = cacti.load()
    assert set(b) == set(cacti.TAB_IV)
    for k, (kb, area, mw) in cacti.TAB_IV.items():
        assert b[k]["size"] == kb * 1024 and b[k]["banks"] == cacti.BANKS[k]
        assert 0.7 * area < b[k]["area_mm2"] < 1.3 * area      # S-14 calibration
