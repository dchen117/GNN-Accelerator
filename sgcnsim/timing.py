"""SGCN dataflow, cache and timing model (docs/sgcn/00-design.md, G-xx).

    plan = build_plan(workload, hw)            # access streams + cache simulation (timing-free)
    rep = schedule(plan, hw)                   # M2-style DRAM server (megasim.timing.Dram)
    rep = sgcnsim.sim.simulate(...)            # with Ramulator (fixed point, as MEGA)

A 2-layer GCN runs as three phases (G-03):

    A  comb0   T = X⁽⁰⁾·W⁽⁰⁾ on the sparse aggregation engines (SGCN §V-F "optionally"; X⁽⁰⁾ in
               CSR, W⁽⁰⁾ rows through the cache, column-tiled so a W⁽⁰⁾ tile fits half the cache);
               T is written dense.
    B  agg0    X⁽¹⁾ = ReLU(Ã·T): sparse aggregation over T (dense slices), ReLU + compressor
               write X⁽¹⁾ in sliced BEICSR.
    C  agg1    logits = (Ã·X⁽¹⁾)·W⁽¹⁾: sparse aggregation over BEICSR slices, then the 16-MAC
               combination engine; logits written dense.
Deeper models repeat C with ReLU + BEICSR output.

Work is split into strips of 32 destination vertices (sparsity-aware cooperation, SGCN §V-C):
strip k goes to aggregation engine k mod 4. Inside a strip the engine walks slice by slice
(feature slicing, C = 96), and for each slice over the strip's vertices and their in-edges.
Every engine consumes one 64 B line per cycle (SIMD16 × 32 bit, Fig.8), so an engine's
compute cycles for a strip = the cache lines it reads.

The access sequence (all phases, strips in order k = 0, 1, …) goes through one LRU cache
simulation before any timing; hits/misses therefore do not depend on timing, and neither does
the DRAM call sequence, which is what the Ramulator fixed point needs (megasim/dram/ramulator.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from megasim import timing as mt
from megasim.dram.ramulator import REGION, REGION_BITS

from .cache import lru
from .workload import Workload


@dataclass(frozen=True)
class HW:
    """SGCN in MEGA's Tab.V configuration (G-01). DRAM fields are what megasim's DRAM models use."""
    agg_engines: int = 4           # Tab.V: 4 SIMD16 for aggregation
    simd: int = 16                 # lanes per engine: one 64 B line of 32-bit values per cycle
    comb_macs: int = 16            # Tab.V: 16 MACs for combination
    cache_bytes: int = 392 * 1024  # MEGA's on-chip capacity (§VI-A-3: same buffer capacity)
    ways: int = 16                 # SGCN Tab.III
    slice: int = 96                # C, SGCN's default slice (§V-B)
    strip: int = 32                # SAC strip height (§V-C)
    word: int = 4                  # 32-bit features and weights
    dram_bytes_per_cycle: float = 256.0
    dram_latency: int = 100        # M2 server only (Ramulator replaces it)
    line: int = 64
    clock_ghz: float = 1.0

    @property
    def sets(self) -> int:
        return self.cache_bytes // self.line // self.ways


# --------------------------------------------------------------------------- layouts
def slice_bounds(F: int, C: int) -> list[tuple[int, int]]:
    return [(lo, min(lo + C, F)) for lo in range(0, F, C)]


def align(x, a: int = 64):
    return -(-np.asarray(x) // a) * a


@dataclass
class Layout:
    """Row-major feature layout: row v starts at v·stride; slice s at off[s] inside the row.
    In-place BEICSR (SGCN §V-A/B): every slice reserves its dense size + bitmap, aligned to a
    line, and a read/write touches only the lines its bitmap + non-zeros occupy."""
    bounds: list
    off: np.ndarray        # (S,) byte offset of each slice in a row
    stride: int            # bytes per row
    bitmap: np.ndarray     # (S,) bitmap bytes per slice (0 for dense)


def beicsr_layout(F: int, hw: HW) -> Layout:
    b = slice_bounds(F, hw.slice)
    bm = np.array([-(-(hi - lo) // 8) for lo, hi in b])
    size = align(bm + hw.word * np.array([hi - lo for lo, hi in b]), hw.line)
    return Layout(b, np.concatenate([[0], np.cumsum(size)[:-1]]), int(size.sum()), bm)


def dense_layout(F: int, hw: HW) -> Layout:
    b = slice_bounds(F, hw.slice)
    size = align(hw.word * np.array([hi - lo for lo, hi in b]), hw.line)
    return Layout(b, np.concatenate([[0], np.cumsum(size)[:-1]]), int(size.sum()),
                  np.zeros(len(b), dtype=np.int64))


def slice_lines(lay: Layout, nnz: np.ndarray | None, n: int, hw: HW) -> np.ndarray:
    """(n, S) lines touched per row and slice. nnz=None: dense (every value)."""
    width = np.array([hi - lo for lo, hi in lay.bounds])
    vals = np.broadcast_to(width, (n, width.size)) if nnz is None else nnz
    return (-(-(lay.bitmap + hw.word * vals) // hw.line)).astype(np.int64)


def slice_nnz(x: np.ndarray, lay: Layout) -> np.ndarray:
    return np.stack([(x[:, lo:hi] != 0).sum(1) for lo, hi in lay.bounds], 1).astype(np.int64)


def _expand(offsets: np.ndarray, counts: np.ndarray, line: int) -> np.ndarray:
    """For each (offset, count): offset, offset+line, …, (count lines)."""
    rep = np.repeat(offsets, counts)
    start = np.repeat(np.cumsum(counts) - counts, counts)
    return rep + (np.arange(rep.size) - start) * line


def w0_tile(F0: int, F1: int, hw: HW) -> int:
    """Column-tile width of W⁽⁰⁾ for phase A (G-04): the widest power-of-two divisor of F1
    (≥ 16, one line) whose F0 × w tile fits ¾ of the cache (W⁽⁰⁾ is the only cached data in
    phase A; ¼ is margin for set conflicts); F1 (no tiling) if none does, since a tile that
    cannot stay resident only multiplies the X⁽⁰⁾ re-reads."""
    best = None
    w = 16
    while w <= F1:
        if F1 % w == 0 and F0 * w * hw.word <= hw.cache_bytes * 3 // 4:
            best = w
        w *= 2
    return best or F1


# --------------------------------------------------------------------------- plan
@dataclass
class Phase:
    name: str
    items: int                           # work items (strips; × tiles in phase A)
    engine_cycles: np.ndarray            # (items,) aggregation-engine cycles
    seq_reads: dict                      # stream → (items,) lines read sequentially
    miss_stream: str                     # cache-miss stream name
    acc_ptr: np.ndarray                  # (items+1,) into the phase's access arrays
    acc_addr: np.ndarray                 # (A,) region offsets of every cached line access
    write_stream: str
    write_lines: np.ndarray              # (items,) lines written
    write_ptr: np.ndarray | None = None  # explicit write addresses (None: sequential)
    write_addr: np.ndarray | None = None
    comb_cycles: np.ndarray | None = None
    comb_macs: float = 0.0
    pre_read: dict = field(default_factory=dict)  # stream → lines read at phase start (weights)
    hit: np.ndarray | None = None        # (A,) filled by the cache simulation
    miss_ptr: np.ndarray | None = None
    miss_addr: np.ndarray | None = None


@dataclass
class Plan:
    workload: str
    n: int
    dims: list
    w0_tile: int
    phases: list


def _per_item(item_of_access: np.ndarray, items: int) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(np.bincount(item_of_access, minlength=items))])


def _phase_comb0(w: Workload, hw: HW) -> Phase:
    x0 = w.x0.tocsr()
    x0.sort_indices()
    n, (F0, F1) = w.n, w.dims[:2]
    K = -(-n // hw.strip)
    wt = w0_tile(F0, F1, hw)
    J, L = F1 // wt, wt * hw.word // hw.line
    row = np.repeat(np.arange(n), np.diff(x0.indptr))
    strip_nz = row // hw.strip
    rows = np.bincount(np.arange(n) // hw.strip, minlength=K)
    nnz = np.bincount(strip_nz, minlength=K)
    addr, item = [], []
    for j in range(J):                                   # tile-outer, strips inner
        base = (j * F0 + x0.indices.astype(np.int64)) * wt * hw.word
        addr.append(_expand(base, np.full(base.size, L), hw.line))
        item.append(np.repeat(j * K + strip_nz, L))
    addr, item = np.concatenate(addr), np.concatenate(item)
    T = dense_layout(F1, hw)
    v = np.arange(n)
    w_addr = [_expand(v * T.stride + j * wt * hw.word, np.full(n, L), hw.line) for j in range(J)]
    w_item = [np.repeat(j * K + v // hw.strip, L) for j in range(J)]
    return Phase(
        name="A comb0 X0·W0", items=J * K, engine_cycles=np.tile(nnz * L, J).astype(float),
        seq_reads={"sg_x0": np.tile((hw.word * rows + 2 * hw.word * nnz) / hw.line, J)},
        miss_stream="sg_w", acc_ptr=_per_item(item, J * K), acc_addr=addr,
        write_stream="sg_t_w", write_lines=np.tile(rows * L, J).astype(float),
        write_ptr=_per_item(np.concatenate(w_item), J * K), write_addr=np.concatenate(w_addr),
        comb_macs=float(x0.nnz) * F1), wt


def _gather(w: Workload, hw: HW, src_lines: np.ndarray, lay: Layout, base: int):
    """Aggregation reads in strip → slice → edge order: (item ptr, addresses, cycles, topo)."""
    n, S = w.n, len(lay.bounds)
    K = -(-n // hw.strip)
    dst = np.repeat(np.arange(n), np.diff(w.ptr))
    e_strip = dst // hw.strip
    E = w.src.size
    e_idx = np.tile(np.arange(E), S)
    s_idx = np.repeat(np.arange(S), E)
    order = np.lexsort((e_idx, s_idx, e_strip[e_idx]))    # strip, then slice, then edge
    e_idx, s_idx = e_idx[order], s_idx[order]
    u = w.src[e_idx]
    cnt = src_lines[u, s_idx]
    addr = _expand(base + u * lay.stride + lay.off[s_idx], cnt, hw.line)
    item = np.repeat(e_strip[e_idx], cnt)
    rows = np.bincount(np.arange(n) // hw.strip, minlength=K)
    edges = np.bincount(e_strip, minlength=K)
    ptr = _per_item(item, K)
    topo = (hw.word * rows + 2 * hw.word * edges) / hw.line   # row ptr + (col idx, Ã value)
    return ptr, addr, np.diff(ptr).astype(float), topo, rows, K


def _beicsr_writes(x: np.ndarray, lay: Layout, base: int, hw: HW, K: int):
    n = x.shape[0]
    ln = slice_lines(lay, slice_nnz(x, lay), n, hw)               # (n, S)
    v = np.repeat(np.arange(n), lay.off.size)
    s = np.tile(np.arange(lay.off.size), n)
    addr = _expand(base + v * lay.stride + lay.off[s], ln.ravel(), hw.line)
    item = np.repeat(v // hw.strip, ln.ravel())
    ptr = _per_item(item, K)
    return ptr, addr, np.diff(ptr).astype(float)


X_LAYER_STRIDE = 1 << 26       # region offset between BEICSR layers inside sg_x (64 MB)


def build_plan(w: Workload, hw: HW = HW()) -> Plan:
    L = w.layers
    phases = []
    A, wt = _phase_comb0(w, hw)
    phases.append(A)
    # B: aggregate T (dense), ReLU, compress X⁽¹⁾ (or logits for a 1-layer model)
    T = dense_layout(w.dims[1], hw)
    ptr, addr, cyc, topo, rows, K = _gather(w, hw, slice_lines(T, None, w.n, hw), T, 0)
    if L > 1:
        X1 = beicsr_layout(w.dims[1], hw)
        wp, wa, wl = _beicsr_writes(w.hidden[0], X1, X_LAYER_STRIDE, hw, K)
        phases.append(Phase("B agg0 Ã·T", K, cyc, {"sg_topo": topo}, "sg_t_r", ptr, addr,
                            "sg_x_w", wl, wp, wa))
    else:
        phases.append(Phase("B agg0 Ã·T", K, cyc, {"sg_topo": topo}, "sg_t_r", ptr, addr,
                            "logits", rows * w.dims[1] * hw.word / hw.line))
    # C: layers l ≥ 1, aggregation first, then the combination engine
    for l in range(1, L):
        Fi, Fo = w.dims[l], w.dims[l + 1]
        X = beicsr_layout(Fi, hw)
        sl = slice_lines(X, slice_nnz(w.hidden[l - 1], X), w.n, hw)
        ptr, addr, cyc, topo, rows, K = _gather(w, hw, sl, X, l * X_LAYER_STRIDE)
        comb = rows * Fi * Fo / hw.comb_macs
        pre = {"sg_wl": Fi * Fo * hw.word / hw.line}
        if l < L - 1:
            Xo = beicsr_layout(Fo, hw)
            wp, wa, wl = _beicsr_writes(w.hidden[l], Xo, (l + 1) * X_LAYER_STRIDE, hw, K)
            ph = Phase(f"C agg{l}+comb{l}", K, cyc, {"sg_topo": topo}, "sg_x_r", ptr, addr,
                       "sg_x_w", wl, wp, wa, comb, float(w.n) * Fi * Fo, pre)
        else:
            ph = Phase(f"C agg{l}+comb{l}", K, cyc, {"sg_topo": topo}, "sg_x_r", ptr, addr,
                       "logits", rows * Fo * hw.word / hw.line, None, None, comb,
                       float(w.n) * Fi * Fo, pre)
        phases.append(ph)
    simulate_cache(phases, hw)
    return Plan(workload=w.name, n=w.n, dims=list(w.dims), w0_tile=wt, phases=phases)


def simulate_cache(phases: list, hw: HW) -> None:
    """One LRU run over all phases' cached reads, in execution order (G-05)."""
    lines = np.concatenate([(np.uint64(REGION[p.miss_stream] << REGION_BITS)
                             + p.acc_addr.astype(np.uint64)) >> np.uint64(6) for p in phases])
    hit = lru(lines, hw.sets, hw.ways)
    i = 0
    for p in phases:
        p.hit = hit[i:i + p.acc_addr.size]
        i += p.acc_addr.size
        miss = p.hit == 0
        p.miss_addr = p.acc_addr[miss]
        p.miss_ptr = np.concatenate([[0], np.cumsum(miss)])[p.acc_ptr]


# --------------------------------------------------------------------------- schedule
class BackfillDram(mt.Dram):
    """Bandwidth server booked in TIME order (G-06): a transfer issued at t takes the earliest
    free bandwidth at or after t, filling gaps between transfers booked earlier, possibly
    split over several gaps. Unlike megasim's FCFS server (which serves in call order), a read
    issued early but called after a late posted write is not blocked by it.

    `bw[k]` is phase k's sustained bandwidth in bytes/cycle, measured with Ramulator by
    replaying that phase's own accesses at saturation (sim.calibrate); `latency` is
    Ramulator's unloaded read latency. Data of a transfer: first = start of its first slot +
    latency, last = end of its last slot + latency."""

    def __init__(self, hw, bw: list, latency: float):
        super().__init__(hw)
        self.bw, self.latency, self.phase = bw, latency, 0
        self.starts: list = []         # sorted, disjoint busy intervals
        self.ends: list = []

    def set_phase(self, k: int) -> None:
        self.phase = k

    def _time(self, i: int, t: float, dur: float) -> tuple[float, float]:
        import bisect
        need = dur * self.hw.dram_bytes_per_cycle / self.bw[self.phase]
        S, E = self.starts, self.ends
        j = bisect.bisect_right(E, t)                 # first interval ending after t
        cur, first = t, None
        while need > 1e-12:
            if j < len(S) and S[j] <= cur:            # inside a busy interval: skip it
                cur = max(cur, E[j])
                j += 1
                continue
            gap_end = S[j] if j < len(S) else float("inf")
            take = min(need, gap_end - cur)
            if first is None:
                first = cur
            # book [cur, cur + take): merge with neighbours
            a, b = cur, cur + take
            if j > 0 and E[j - 1] >= a:
                j -= 1
                E[j] = b
            else:
                S.insert(j, a)
                E.insert(j, b)
            if j + 1 < len(S) and S[j + 1] <= E[j]:
                E[j] = E[j + 1]
                del S[j + 1], E[j + 1]
            need -= take
            cur = b
        self.free = max(self.free, cur + self.latency)
        return first + self.latency, cur + self.latency


class UncontendedDram(mt.Dram):
    """Every transfer gets the fixed latency + its own bytes at full bandwidth, with no queueing
    behind other transfers (G-06). Only the starting point of the Ramulator fixed point and a
    quick debugging mode: megasim's FCFS server (S-12) books transfers in call order, and SGCN's
    call order (engine items, then their posted writes) is not time order, so a write booked at
    its late ready time would block reads issued earlier — an artefact, not DRAM behaviour.
    Ramulator (sim.py default) models contention properly."""

    def _time(self, i: int, t: float, dur: float) -> tuple[float, float]:
        self.free = max(self.free, t + self.hw.dram_latency + dur)
        return t + self.hw.dram_latency, t + self.hw.dram_latency + dur


@dataclass
class Report:
    hw: HW
    total_cycles: float
    phases: list            # per phase: name, start, end, busy, accesses, hits
    dram: mt.Dram
    info: dict = field(default_factory=dict)

    @property
    def dram_bytes(self) -> dict:
        return {k: v * self.hw.line for k, v in self.dram.lines.items()}


def schedule(plan: Plan, hw: HW = HW(), dram: mt.Dram | None = None) -> Report:
    """Event recurrence over the work items (G-06). Engine e runs items e, e+4, …; the reads of
    an item are issued when the engine starts its previous item (one item of prefetch), the
    item starts when its first data is there and ends no earlier than its last data + 1. The
    combination engine (phase C) takes items in order once their aggregation is done. Writes
    are posted when an item's output is ready. Phases are separated by a barrier that also
    waits for the writes."""
    dram = dram or UncontendedDram(hw)
    E = hw.agg_engines
    t = 0.0
    out = []
    for k, p in enumerate(plan.phases):
        if hasattr(dram, "set_phase"):
            dram.set_phase(k)
        log0 = len(dram.log)
        start = t
        w_ready = t
        if p.pre_read:
            _, w_ready = dram.xfer(t, p.pre_read)
        free = [t] * E
        last_start = [t] * E
        comb_free, end = t, t
        busy_eng, busy_comb = 0.0, 0.0
        for i in range(p.items):
            e = i % E
            streams = {k: v[i] for k, v in p.seq_reads.items()}
            a, b = p.miss_ptr[i], p.miss_ptr[i + 1]
            addrs = None
            if b > a:
                streams[p.miss_stream] = b - a
                addrs = {p.miss_stream: p.miss_addr[a:b]}
            first, last = dram.xfer(last_start[e], streams, addrs)
            reads = any(v > 0 for v in streams.values())
            s = max(free[e], first) if reads else free[e]
            fin = max(s + p.engine_cycles[i], last + 1 if reads else s)
            busy_eng += p.engine_cycles[i]
            free[e], last_start[e] = fin, s
            ready = fin
            if p.comb_cycles is not None:
                cs = max(comb_free, fin, w_ready)
                comb_free = cs + p.comb_cycles[i]
                busy_comb += p.comb_cycles[i]
                ready = comb_free
            if p.write_ptr is not None:
                wa, wb = p.write_ptr[i], p.write_ptr[i + 1]
                _, wl = dram.xfer(ready, {p.write_stream: wb - wa},
                                  {p.write_stream: p.write_addr[wa:wb]})
            else:
                _, wl = dram.xfer(ready, {p.write_stream: p.write_lines[i]})
            end = max(end, ready, wl)
        end = max(end, comb_free, *free)
        out.append({"name": p.name, "start": start, "end": end, "cycles": end - start,
                    "busy": {"agg_engines": busy_eng / E, "comb": busy_comb},
                    "accesses": int(p.acc_addr.size), "hits": int(p.hit.sum()),
                    "items": p.items, "log": (log0, len(dram.log))})
        t = end
    return Report(hw=hw, total_cycles=t, phases=out, dram=dram,
                  info={"w0_tile": plan.w0_tile,
                        # systolic-array MACs only (phase A's MACs run on the aggregation
                        # engines and are counted from their busy cycles)
                        "comb_macs": sum(p.comb_macs for p in plan.phases
                                         if p.comb_cycles is not None)})
