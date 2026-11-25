"""Unit tests for megasim (functional model). Run: python -m pytest tests/test_megasim.py -q

The end-to-end bit-exact check against mega_pt.hw_stats.int_forward needs trained runs and
is a CLI instead: python -m megasim.check runs/hw/*
"""

import numpy as np
import pytest

from megasim import functional, package
from megasim.program import FixedTable, SparseRows


# --------------------------------------------------------------------------- packages
def test_package_bytes_by_hand():
    # Two 3-bit nodes (values 1,2,-1 and 3), then one 2-bit node (value -2).
    # Package 1: b=3, 4 values = 5+12 = 17 bits → short (64). Header 00|010.
    # Package 2: b=2, 1 value = 7 bits → short. Header 00|001.
    ps = package.encode(np.array([1, 2, -1, 3, -2]), np.array([3, 3, 3, 3, 2]))
    assert list(ps.mode) == [0, 0] and list(ps.count) == [4, 1]
    bits = np.unpackbits(ps.data)
    assert bits.size == 128
    p1 = "".join(map(str, bits[:17]))
    assert p1 == "00" + "010" + "001" + "010" + "111" + "011"
    p2 = "".join(map(str, bits[64:71]))
    assert p2 == "00" + "001" + "10"
    assert not bits[17:64].any() and not bits[71:].any()          # zero padding


def test_package_full_and_spanning():
    # 100 values at b=4: capacity 46 per long package → 46, 46, 8 (last fits 5+32 → short)
    v = np.full(100, 3)
    ps = package.encode(v, np.full(100, 4))
    assert list(ps.count) == [46, 46, 8] and list(ps.mode) == [2, 2, 0]
    assert np.array_equal(package.decode(ps.data, np.full(100, 4)), v)


def test_package_capacities():
    assert [package.capacity(b) for b in range(2, 9)] == [93, 62, 46, 37, 31, 26, 23]


@pytest.mark.parametrize("seed", range(5))
def test_package_round_trip_random(seed):
    rng = np.random.default_rng(seed)
    b = np.repeat(rng.integers(2, 9, size=60), rng.integers(1, 30, size=60))
    v = np.array([rng.integers(-(2 ** (x - 1)), 2 ** (x - 1)) for x in b])
    v[v == 0] = 1
    ps = package.encode(v, b)
    assert np.array_equal(package.decode(ps.data, b), v)
    assert ps.data.size * 8 == int(sum(package.MODE_LENGTHS[m] for m in ps.mode))


# --------------------------------------------------------------------------- arithmetic
def test_round_shift_matches_round_half_away():
    num = np.arange(-40, 41)
    for s in (1, 2, 3):
        ref = np.sign(num / 2 ** s) * np.floor(np.abs(num / 2 ** s) + 0.5)
        assert np.array_equal(functional.round_shift(num, s), ref.astype(np.int64))


def test_fixed_table_power_of_two_max():
    t = FixedTable.from_values(np.array([1.0, 0.25]))
    assert t.codes.max() < 2 ** 16 and np.allclose(t.values(), [1.0, 0.25])


@pytest.mark.parametrize("seed", range(3))
def test_combination_equals_matmul(seed):
    """Bit-serial tile/pass computation == plain integer X̄·W̄, incl. rows with >128 non-zeros
    (several passes) and negative values (MSB plane subtracted)."""
    rng = np.random.default_rng(seed)
    n, F, F2 = 30, 400, 20
    dens = rng.uniform(0.01, 0.9, size=n)[:, None]
    bits = rng.integers(2, 9, size=n)
    lim = 2 ** (bits - 1) - 1
    x = rng.integers(-lim[:, None], lim[:, None] + 1, size=(n, F)) * (rng.random((n, F)) < dens)
    W = rng.integers(-7, 8, size=(F, F2)).astype(np.int8)
    acc, checks = functional.combination(SparseRows.from_dense(x), bits, W, chunk_elems=500)
    assert np.array_equal(acc, x @ W.astype(np.int64))
    assert not checks["shift_acc_overflow"] and not checks["tile_acc_overflow"]


def test_pass_tile_split():
    # one row with 130 non-zeros: pass 0 = 128 → 32 per tile; pass 1 = 2 → share 1, tiles 0 and 1
    x = SparseRows.from_dense(np.ones((1, 130), dtype=np.int64))
    _, p, tile = functional._pass_tile(x)
    assert np.array_equal(np.bincount(tile[p == 0]), [32, 32, 32, 32])
    assert list(tile[p == 1]) == [0, 1]


@pytest.mark.parametrize("res_scale", [0.25, 1e-6])   # r_d with fewer / more fractional bits than e·γ
def test_encoder_residual_matches_float(res_scale):
    """Q-36: the Encoder's residual add aligns the two binary points exactly; the result
    equals min(round_half_away(ReLU(psum·e·γ + x̄·r)), qmax) computed in float64."""
    from types import SimpleNamespace
    rng = np.random.default_rng(0)
    n, F, K = 50, 16, 4
    deg_idx = rng.integers(0, K, n)
    enc = FixedTable.from_values(rng.uniform(0.01, 0.2, K))
    gamma = FixedTable.from_values(rng.uniform(0.05, 1.0, F))
    res = FixedTable.from_values(rng.uniform(0.5, 1.0, K) * res_scale)
    out_bits = np.array([2, 4, 6, 8])
    layer = SimpleNamespace(enc=enc, gamma=gamma, res=res, deg_idx=deg_idx, out_bits_table=out_bits)
    psum = rng.integers(-200, 200, (n, F))
    xin = rng.integers(0, 16, (n, F)) * (rng.random((n, F)) < 0.5)
    got = functional.encode_next(psum, layer, SparseRows.from_dense(xin)).to_dense()
    pre = (psum * enc.values()[deg_idx][:, None] * gamma.values()[None, :]
           + xin * res.values()[deg_idx][:, None])
    pre = np.maximum(pre, 0)
    want = np.minimum(np.sign(pre) * np.floor(np.abs(pre) + 0.5),
                      ((1 << (out_bits[deg_idx] - 1)) - 1)[:, None])
    assert np.array_equal(got, want.astype(np.int64))
