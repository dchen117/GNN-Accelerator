"""Cycle-stepped model of the 4 Combination Tiles, for checking only (S-03).

It steps the Q-24 schedule cycle by cycle on a few nodes and checks two things against the
fast models:
  * the integers: the Tile Accumulator output equals X̄·W̄ (and functional.combination);
  * the cycles: the steady-state time equals the closed form of timing.comb_cycles without
    the scan bound, i.e. Σ_nodes blocks · Σ_pass max_tile waves·b, plus a constant drain.

Per tile (Q-24): the BSE array is 8 rows (non-zeros of one wave) × 32 C-PEs, in two halves
of 16 C-PEs. The crossbar delivers one half's weights for one wave per cycle.
  cycle t       crossbar → left half (C-PEs 0-15): weights of wave w; left processes plane 0
  t+1 … t+b−1   left processes planes 1 … b−1 with the same weights
  cycle t+1     crossbar → right half (C-PEs 16-31); right processes plane 0 (forwarded from
                the left half), then planes 1 … b−1 up to t+b
  cycle t+b     the left half takes the next wave
The crossbar is a resource: with b = 1 the left half's next request and the right half's
request collide, and the model stalls (b ≥ 2 never collides, which is Q-24's argument).

Order of work: for each node, for each 32-column block, for each pass (≤128 non-zeros, split
over the 4 tiles in shares of ⌈s/4⌉, Q-23): every tile processes its waves (≤8 non-zeros).
The 4 tiles meet at a barrier per (node, block, pass), because the Tile Accumulator adds the
4 tiles' Shifter-Acc results of one pass; a tile's left half may start the next pass while
its right half still finishes (the 1-cycle skew overlaps).

The constant drain after the last wave is 1 (half skew) + 1 (adder tree) + 1 (Shifter-Acc
register) = 3 cycles here; the crossbar read is in the same cycle as plane 0 (Q-24 says so),
so Q-24's "4" fill counts one more stage than this model needs. It is paid once per sweep.
"""

from __future__ import annotations

import numpy as np

from .program import SparseRows

TILES, ROWS, CPES, HALF, PASS_NNZ = 4, 8, 32, 16, 128
DRAIN = 3


def _steps(x: SparseRows, bits: np.ndarray, F_out: int):
    """Yield (node, block, b, [per-tile list of waves]) per pass; a wave is (cols, codes)."""
    blocks = -(-F_out // CPES)
    for i in range(x.n):
        cols = x.col[x.ptr[i]:x.ptr[i + 1]]
        vals = x.val[x.ptr[i]:x.ptr[i + 1]]
        b = int(bits[i])
        codes = vals & ((1 << b) - 1)
        for blk in range(blocks):
            for p0 in range(0, cols.size, PASS_NNZ):
                pc, pv = cols[p0:p0 + PASS_NNZ], codes[p0:p0 + PASS_NNZ]
                share = -(-pc.size // TILES)
                tiles = []
                for t in range(TILES):
                    tc, tv = pc[t * share:(t + 1) * share], pv[t * share:(t + 1) * share]
                    tiles.append([(tc[w:w + ROWS], tv[w:w + ROWS]) for w in range(0, tc.size, ROWS)])
                yield i, blk, b, tiles


def simulate(x: SparseRows, bits: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, int, dict]:
    """Returns (acc (n, F_out), total cycles, stats)."""
    n, F_out = x.n, W.shape[1]
    blocks = -(-F_out // CPES)
    Wp = np.zeros((W.shape[0], blocks * CPES), dtype=np.int64)
    Wp[:, :F_out] = W
    acc = np.zeros((n, blocks * CPES), dtype=np.int64)
    stats = {"xbar_stalls": 0, "shift_acc_max": 0}

    # A job is one wave on one tile: (step id, node, block, b, cols, codes).
    # Half state: [job, plane]. Each half has its own 16 Shifter-Accs; they hold the running
    # sum of one step (node, block, pass) and are emptied into the Tile Accumulator when the
    # half starts a wave of another step, and at the end.
    left, right = [None] * TILES, [None] * TILES
    waiting = [None] * TILES                   # wave started on the left, right half not yet
    sa = np.zeros((TILES, CPES), dtype=np.int64)
    sa_owner = [[None, None] for _ in range(TILES)]    # (node, block, step) per half
    halves = ((0, 0, HALF), (1, HALF, CPES))

    def flush(tl, h):
        own = sa_owner[tl][h]
        if own is not None:
            node, blk, _ = own
            lo, hi = halves[h][1], halves[h][2]
            acc[node, blk * CPES + lo: blk * CPES + hi] += sa[tl, lo:hi]
            sa[tl, lo:hi] = 0
            sa_owner[tl][h] = None

    def start(tl, h, job):
        own = (job[1], job[2], job[0])
        if sa_owner[tl][h] != own:
            flush(tl, h)
            sa_owner[tl][h] = own
        return [job, 0]

    def step_half(tl, h, st):
        """Process one bit-plane; returns True when the wave is finished on this half."""
        (sid, node, blk, b, cols, codes), k = st
        lo, hi = halves[h][1], halves[h][2]
        bit = (codes >> k) & 1
        wk = -(1 << k) if k == b - 1 else (1 << k)
        sa[tl, lo:hi] += wk * (bit @ Wp[cols, blk * CPES + lo: blk * CPES + hi])
        stats["shift_acc_max"] = max(stats["shift_acc_max"], int(np.abs(sa[tl, lo:hi]).max()))
        st[1] += 1
        return st[1] == b

    t, last_busy = 0, -1
    steps = list(_steps(x, bits, F_out))
    for sid, (i, blk, b, tiles) in enumerate(steps + [(None, None, None, None)]):  # sentinel drains
        queues = [[(sid, i, blk, b, c, v) for c, v in w] for w in tiles] if tiles is not None else \
            [[] for _ in range(TILES)]
        final = tiles is None
        # Barrier: this step's waves enter only after every tile's left half has finished the
        # previous step (the loop below runs until then); right halves may still be busy.
        while any(queues) or any(l is not None for l in left) or \
                (final and (any(w is not None for w in waiting) or any(r is not None for r in right))):
            for tl in range(TILES):
                xbar_used = False
                if right[tl] is None and waiting[tl] is not None:     # older wave first
                    right[tl] = start(tl, 1, waiting[tl])
                    waiting[tl] = None
                    xbar_used = True
                if left[tl] is None and queues[tl]:
                    if xbar_used or waiting[tl] is not None:
                        stats["xbar_stalls"] += 1
                    else:
                        left[tl] = start(tl, 0, queues[tl].pop(0))
                if left[tl] is not None:
                    last_busy = t
                    job, k0 = left[tl]
                    if step_half(tl, 0, left[tl]):
                        left[tl] = None
                    if k0 == 0:
                        # plane 0 done: it is forwarded to the right half, which takes the
                        # wave's weights from the crossbar from the next cycle on
                        waiting[tl] = job
                if right[tl] is not None:
                    last_busy = t
                    if step_half(tl, 1, right[tl]):
                        right[tl] = None
            t += 1
    for tl in range(TILES):
        flush(tl, 0)
        flush(tl, 1)
    total = last_busy + 1 + 2                    # + adder tree + Shifter-Acc register
    return acc[:, :F_out], total, stats
