"""Timing model of MEGA (S-02, S-09 … S-12): per-unit cycle counts, the Q-28 sweep schedule,
DRAM traffic per stream, and a bandwidth + latency DRAM model.

    part  = partition.partition(prog); prog = partition.relabel(prog, part.order)
    func  = functional.run(prog)                    # X̄⁽ˡ⁾ of every layer (sizes, nnz)
    rep   = timing.simulate(prog, part.ptr, func)   # TimingReport

Structure
---------
1. Per-unit cycle counts (closed forms, S-02), vectorized over nodes:
     combination  per node: max(scan, blocks · Σ_pass max_tile waves·b)          Q-23/24
                  scan = ⌈F_pad / 256⌉ (bitmap read once; the bit-planes are replayed
                  from the Bit FIFO for each 32-column block, S-09)
     aggregation  per (subgraph, source): ⌈edges / k⌉ · ⌈F2′ / 256⌉, k = max(1, 256 // F2′)  Q-25
     encoder      per node: ⌈F2 / 32⌉                                              Q-26
     condense     per node: ⌈batches / 16⌉ compare + 2 per matching batch            Q-27
2. DRAM traffic per stream (S-10). Every stream lives in its own contiguous region and is
   read/written in 64 B lines; a contiguous byte range [a, b) costs the lines it touches.
3. The sweep schedule of Q-28 as an event recurrence over subgraphs (S-11):
     sweep 0      combine layer 0 for every subgraph (Condense fills the Sparse regions)
     sweep l      per subgraph k: aggregate layer l−1 → Encoder → combine layer l (fused,
                  node by node) → Condense for layer l
     sweep L      aggregate layer L−1, Encoder writes the logits
   Ping-pong buffers let subgraph k+1's aggregation overlap subgraph k's Encoder/combination
   (Aggregation Buffer halves), and k+1's inputs load while k is aggregated.
4. DRAM (S-12, M2): one FCFS server at 256 B/cycle (256 GB/s at 1 GHz) with a fixed latency
   of 100 cycles per transfer. Reads block the unit that consumes them; writes are posted
   (they take bandwidth but nobody waits on them, except the end of a sweep). M3 replaces this
   server with Ramulator 2.0 on the same request streams.

All numbers are in cycles of the 1 GHz clock.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from . import package
from .functional import LayerResult
from .program import Graph, Program, SparseRows


# --------------------------------------------------------------------------- parameters
@dataclass(frozen=True)
class HW:
    """MEGA hardware parameters. Sources: Tab.IV (buffers, units), Q-21 … Q-29."""
    tiles: int = 4                 # Combination Tiles (Q-21)
    bse_rows: int = 8              # n: non-zeros per wave
    cpes: int = 32                 # m: output columns per tile = column-block width
    pass_nnz: int = 128            # non-zeros per pass (Q-23)
    scan_bits: int = 256           # bitmap bits scanned per cycle (Q-23)
    aus: int = 256                 # Aggregation Units (Q-25)
    qn_units: int = 32             # Encoder QN units (Q-26)
    fill: int = 4                  # combination pipeline fill/drain per sweep (Q-24)
    batch: int = 16                # ID FIFOs = subgraphs per Condense batch (Q-27)
    condense_match_cycles: int = 2 # per matching batch: load heads + match, write back + min (ours)
    # buffers in bytes (Tab.IV)
    input_buf: int = 64 * 1024
    weight_buf: int = 48 * 1024
    comb_buf: int = 96 * 1024
    sparse_buf: int = 32 * 1024
    edge_buf: int = 24 * 1024
    agg_buf: int = 128 * 1024
    # DRAM (Q-29; M2 model)
    dram_bytes_per_cycle: float = 256.0    # 256 GB/s at 1 GHz
    dram_latency: int = 100                # cycles per transfer (ours, M2 only)
    line: int = 64                         # access granularity in bytes
    clock_ghz: float = 1.0

    @property
    def region_bytes(self) -> int:       # Sparse Buffer: 16 equal regions (Q-27)
        return self.sparse_buf // self.batch


# --------------------------------------------------------------------------- helpers
def lines_touched(start_bit, nbits, line_bits: int) -> np.ndarray:
    """Lines a contiguous bit range touches (0 for an empty range)."""
    start_bit, nbits = np.asarray(start_bit), np.asarray(nbits)
    end = start_bit + nbits
    n = -(-end // line_bits) - start_bit // line_bits
    return np.where(nbits > 0, n, 0)


def seg_sum(v: np.ndarray, ptr: np.ndarray) -> np.ndarray:
    """Sum of v over each segment [ptr[k], ptr[k+1])."""
    c = np.concatenate([[0], np.cumsum(v, dtype=np.float64)])
    return c[ptr[1:]] - c[ptr[:-1]]


def row_bits(F: int) -> int:
    """One B̄ row (4-bit signed, Q-09) packed: F · 4 bits."""
    return 4 * F


def pow2ceil(x: int) -> int:
    return 1 << max(0, math.ceil(math.log2(x)))


# --------------------------------------------------------------------------- 1. unit cycles
def comb_cycles(x: SparseRows, bits: np.ndarray, F_out: int, hw: HW) -> np.ndarray:
    """Cycles per node in the Combination Tiles (Q-23/24, S-09).
    A full pass (128 non-zeros) gives each tile 32 = 4 waves; a partial pass of r non-zeros
    gives each tile ⌈r/4⌉, i.e. ⌈⌈r/4⌉/8⌉ waves on the busiest tile. A wave takes b cycles.
    Every 32-column block of W̄ replays the node's planes. The bitmap scan (once per node,
    overlapped with the previous node's compute) is the lower bound."""
    nnz = x.nnz_per_row
    full = nnz // hw.pass_nnz
    rem = nnz - full * hw.pass_nnz
    share = -(-rem // hw.tiles)
    waves = full * (hw.pass_nnz // hw.tiles // hw.bse_rows) + -(-share // hw.bse_rows)
    blocks = -(-F_out // hw.cpes)
    compute = waves * bits * blocks
    f_pad = -(-x.F // package.BITMAP_WORD) * package.BITMAP_WORD
    scan = -(-f_pad // hw.scan_bits)
    return np.maximum(compute, scan).astype(np.int64)


def agg_cycles_per_edge_group(F2: int, hw: HW) -> tuple[int, int]:
    """(edges per cycle, cycles per edge group) for an aggregated width F2 (Q-25)."""
    f2p = pow2ceil(F2)
    return max(1, hw.aus // f2p), -(-f2p // hw.aus)


@dataclass
class SubgraphEdges:
    """Edges into each subgraph, split by source (the AU's CSC walk, Q-25)."""
    agg_cycles: np.ndarray         # (K,) aggregation cycles at the given F2
    sources: np.ndarray            # (K,) distinct sources (internal + external)
    edges: np.ndarray              # (K,) edges incl. self-loops
    ext_rows: np.ndarray           # (K,) external sources = rows Condense placed for k
    pair_src: np.ndarray           # external (target subgraph, source) pairs, by source
    pair_dst_sub: np.ndarray


def subgraph_edges(g: Graph, sub: np.ndarray, K: int, F2: int, hw: HW) -> SubgraphEdges:
    src = np.repeat(np.arange(g.n), np.diff(g.src_ptr))
    t = sub[g.src_dst]                                     # target subgraph of each edge
    key, cnt = np.unique(t * g.n + src, return_counts=True)   # (target sub, source) groups
    ksub, ksrc = key // g.n, key % g.n
    k_e, cyc_e = agg_cycles_per_edge_group(F2, hw)
    cyc = -(-cnt // k_e) * cyc_e
    ext = sub[ksrc] != ksub
    o = np.lexsort((ksub[ext], ksrc[ext]))               # combination order: by source
    return SubgraphEdges(
        agg_cycles=np.bincount(ksub, weights=cyc, minlength=K).astype(np.int64),
        sources=np.bincount(ksub, minlength=K), edges=np.bincount(t, minlength=K),
        ext_rows=np.bincount(ksub[ext], minlength=K),
        pair_src=ksrc[ext][o], pair_dst_sub=ksub[ext][o])


def condense_cycles(se: SubgraphEdges, sub: np.ndarray, n: int, K: int, hw: HW) -> np.ndarray:
    """Per node: compare its ID with all batch minimums (16 per cycle), then 2 cycles for each
    batch whose minimum equals it (B2 with batch-minimum skipping, Q-27)."""
    nb = -(-K // hw.batch)
    batch_key = np.unique(se.pair_src * nb + se.pair_dst_sub // hw.batch)
    matches = np.bincount(batch_key // nb, minlength=n)
    return (-(-nb // 16) + hw.condense_match_cycles * matches).astype(np.int64)


def fused_enc_comb(comb: np.ndarray, e: int, ptr: np.ndarray) -> np.ndarray:
    """Time per subgraph of the Encoder feeding combination node by node (Q-28 fusion).
    Node i is encoded at (i+1)·e and combined after it and after node i−1, so the subgraph
    finishes at max_i [(i+1)·e + Σ_{j≥i} comb_j] (a max-plus recurrence in closed form)."""
    out = np.zeros(ptr.size - 1)
    c = np.concatenate([[0], np.cumsum(comb, dtype=np.float64)])
    for k in range(ptr.size - 1):
        a, b = ptr[k], ptr[k + 1]
        if b == a:
            continue
        i = np.arange(b - a)
        suffix = c[b] - c[a + i]
        out[k] = float(np.max((i + 1) * e + suffix))
    return out


# --------------------------------------------------------------------------- 2. traffic
def package_lines(x: SparseRows, bits: np.ndarray, ptr: np.ndarray, hw: HW) -> tuple[np.ndarray, np.ndarray]:
    """Lines and exact bits of each subgraph's slice of the package stream. A subgraph reads
    every package that holds one of its values (boundary packages are read by both
    neighbours, Q-19's per-subgraph start pointer)."""
    K = ptr.size - 1
    ps = package.encode(x.val, np.repeat(bits, x.nnz_per_row))
    if ps.num_packages == 0:
        return np.zeros(K, dtype=np.int64), np.zeros(K, dtype=np.int64)
    plen = np.asarray(package.MODE_LENGTHS)[ps.mode]
    pkg_of_val = np.repeat(np.arange(ps.num_packages), ps.count)
    v0, v1 = x.ptr[ptr[:-1]], x.ptr[ptr[1:]]               # value range of each subgraph
    has = v1 > v0
    p0 = pkg_of_val[np.minimum(v0, pkg_of_val.size - 1)]
    p1 = pkg_of_val[np.maximum(v1 - 1, 0)]
    start = ps.start_bit[p0]
    nbits = np.where(has, ps.start_bit[p1] + plen[p1] - start, 0)
    return lines_touched(start, nbits, hw.line * 8), nbits


def bitmap_lines(x: SparseRows, ptr: np.ndarray, hw: HW) -> tuple[np.ndarray, np.ndarray]:
    wbits = -(-x.F // package.BITMAP_WORD) * package.BITMAP_WORD
    nbits = np.diff(ptr) * wbits
    return lines_touched(ptr[:-1] * wbits, nbits, hw.line * 8), nbits


def contiguous_lines(ptr: np.ndarray, bits_per_node: int, hw: HW) -> np.ndarray:
    """Lines of a per-node array of fixed width, read subgraph by subgraph."""
    return lines_touched(ptr[:-1] * bits_per_node, np.diff(ptr) * bits_per_node, hw.line * 8)


@dataclass
class CondenseTraffic:
    flush_lines_by_src_sub: np.ndarray    # (K,) lines written while combining subgraph k
    end_flush_lines: int                  # lines flushed at the end of the sweep
    read_lines: np.ndarray                # (K,) lines read when k is aggregated
    flushes: int


def condense_traffic(se: SubgraphEdges, sub: np.ndarray, K: int, F2: int, hw: HW) -> CondenseTraffic:
    """Sparse Buffer regions as write-combining buffers (Q-27, S-10): region r = t mod 16
    holds rows for one target subgraph t at a time and flushes them to t's contiguous DRAM
    area when it is full, when a row for another subgraph of the same region arrives, or at
    the end of the sweep. Rows are packed (F2 · 4 bits each)."""
    rb = row_bits(F2)
    line_bits = hw.line * 8
    region_rows = max(1, hw.region_bytes * 8 // rb)
    occ_sub = [-1] * hw.batch
    occ_rows = [0] * hw.batch
    area_off = np.zeros(K, dtype=np.int64)               # bits written so far to t's area
    by_src = np.zeros(K, dtype=np.int64)
    nflush = 0

    def flush(r: int) -> int:
        nonlocal nflush
        t, nb = occ_sub[r], occ_rows[r] * rb
        ln = int(lines_touched(area_off[t], nb, line_bits))
        area_off[t] += nb
        occ_rows[r] = 0
        nflush += 1
        return ln

    for j, t in zip(se.pair_src.tolist(), se.pair_dst_sub.tolist()):
        r = t % hw.batch
        if occ_rows[r] and occ_sub[r] != t:
            by_src[sub[j]] += flush(r)
        occ_sub[r] = t
        occ_rows[r] += 1
        if occ_rows[r] == region_rows:
            by_src[sub[j]] += flush(r)
    end = sum(flush(r) for r in range(hw.batch) if occ_rows[r])
    read = lines_touched(np.zeros(K, dtype=np.int64), se.ext_rows * rb, line_bits)
    assert np.array_equal(area_off, se.ext_rows * rb)
    return CondenseTraffic(by_src, end, read, nflush)


def weight_traffic(x: SparseRows, layer, ptr: np.ndarray, hw: HW) -> dict:
    """W̄ and the per-layer tables (Q-35). If they fit in the Weight Buffer, W̄ is loaded once at
    the start of the sweep. Otherwise the buffer is a direct-mapped cache of W̄ rows (S-10):
    row r goes to slot r mod S; a node fetches each of its non-zero rows once (it is reused
    over the node's passes and column blocks), and a fetch misses if the slot holds another row."""
    rbytes = -(-row_bits(layer.F_out) // 8)
    w_bytes = layer.F_in * rbytes
    tab_bytes = layer.deg_values.size * 5 + layer.F_out * 4   # (b, α′, e) + (c, γ) entries
    load = -(-(w_bytes + tab_bytes) // hw.line)
    K = ptr.size - 1
    if w_bytes + tab_bytes <= hw.weight_buf:
        return {"resident": True, "load_lines": load, "miss_lines": np.zeros(K, dtype=np.int64),
                "misses": 0, "accesses": int(x.val.size)}
    row_lines = -(-rbytes // hw.line)
    slots = (hw.weight_buf - tab_bytes) // (row_lines * hw.line)
    assert slots >= 1, "Weight Buffer cannot hold the tables and one W̄ row"
    col = x.col
    slot = col % slots
    o = np.argsort(slot, kind="stable")                  # node order kept inside each slot
    s, c = slot[o], col[o]
    miss_sorted = np.ones(c.size, dtype=bool)
    miss_sorted[1:] = (s[1:] != s[:-1]) | (c[1:] != c[:-1])
    miss = np.empty_like(miss_sorted)
    miss[o] = miss_sorted
    row = x.rows()
    per_node = np.bincount(row[miss], minlength=x.n)
    miss_lines = (seg_sum(per_node, ptr) * row_lines).astype(np.int64)
    tab_load = -(-tab_bytes // hw.line)
    return {"resident": False, "load_lines": tab_load, "miss_lines": miss_lines,
            "misses": int(miss.sum()), "accesses": int(col.size), "slots": int(slots)}


# --------------------------------------------------------------------------- 3-4. schedule
class Dram:
    """FCFS bandwidth server with a fixed latency (S-12). Transfers are booked in the order the
    schedule issues them; each occupies the server for bytes / bandwidth cycles."""

    def __init__(self, hw: HW):
        self.hw = hw
        self.free = 0.0
        self.busy = 0.0
        self.lines = defaultdict(float)          # stream → lines (read + write)

    def xfer(self, t: float, streams: dict) -> tuple[float, float]:
        """Book one transfer made of several streams (name → lines) issued at time t.
        Returns (first data, last data); (t, t) if empty."""
        n = float(sum(streams.values()))
        for k, v in streams.items():
            self.lines[k] += float(v)
        if n == 0:
            return t, t
        start = max(t, self.free)
        dur = n * self.hw.line / self.hw.dram_bytes_per_cycle
        self.free = start + dur
        self.busy += dur
        return start + self.hw.dram_latency, start + self.hw.dram_latency + dur


@dataclass
class SweepReport:
    name: str
    start: float
    end: float
    busy: dict = field(default_factory=dict)       # unit → busy cycles
    dram_lines: dict = field(default_factory=dict)  # stream → lines
    dram_busy: float = 0.0
    notes: dict = field(default_factory=dict)

    @property
    def cycles(self) -> float:
        return self.end - self.start


@dataclass
class TimingReport:
    name: str
    hw: HW
    num_subgraphs: int
    sweeps: list
    total_cycles: float
    dram_bytes: dict
    info: dict

    def summary(self) -> dict:
        busy = defaultdict(float)
        for s in self.sweeps:
            for k, v in s.busy.items():
                busy[k] += v
        return {"run": self.name, "subgraphs": self.num_subgraphs,
                "cycles": int(self.total_cycles),
                "time_us": self.total_cycles / (self.hw.clock_ghz * 1e3),
                "util": {k: v / self.total_cycles for k, v in busy.items()},
                "dram_MB": {k: v / 2 ** 20 for k, v in self.dram_bytes.items()},
                "dram_MB_total": sum(self.dram_bytes.values()) / 2 ** 20,
                "sweeps": [{"name": s.name, "cycles": int(s.cycles),
                            "busy": {k: int(v) for k, v in s.busy.items()},
                            "dram_busy": int(s.dram_busy), **s.notes} for s in self.sweeps],
                **self.info}


def simulate(prog: Program, ptr: np.ndarray, func: list[LayerResult], hw: HW = HW()) -> TimingReport:
    """Run the Q-28 schedule on a relabelled program whose subgraph k is node range
    [ptr[k], ptr[k+1]). `func` is functional.run(prog) (gives each layer's X̄)."""
    g, L, K = prog.graph, len(prog.layers), ptr.size - 1
    n = g.n
    sub = np.repeat(np.arange(K), np.diff(ptr))
    sizes = np.diff(ptr)
    xs = [prog.x0] + [r.x_out for r in func[:-1]]
    dram = Dram(hw)
    sweeps: list[SweepReport] = []
    info: dict = {"layers": []}

    # ---- per-layer precomputation
    comb = [comb_cycles(xs[l], prog.layers[l].bits, prog.layers[l].F_out, hw) for l in range(L)]
    edges = [subgraph_edges(g, sub, K, prog.layers[l].F_out, hw) for l in range(L)]
    cond = [condense_cycles(edges[l], sub, n, K, hw) for l in range(L)]
    ctraf = [condense_traffic(edges[l], sub, K, prog.layers[l].F_out, hw) for l in range(L)]
    wtraf = [weight_traffic(xs[l], prog.layers[l], ptr, hw) for l in range(L)]
    meta_lines = contiguous_lines(ptr, 16, hw)                    # 16-bit degree index (Q-26)
    # eID lists (4 B each) + 12 B of metadata per subgraph: resident in half the Edge Buffer
    # if they fit (loaded once), else streamed in every combination sweep (S-10)
    eid_bytes = edges[0].pair_src.size * 4 + K * 12
    eid_resident = eid_bytes <= hw.edge_buf // 2
    eid_lines_by_sub = np.bincount(sub[edges[0].pair_src], minlength=K) * 4 / hw.line
    # internal B̄ rows: stay in the Combination Buffer if the whole layer fits one half
    b_onchip = [n * row_bits(prog.layers[l].F_out) <= hw.comb_buf // 2 * 8 for l in range(L)]
    b_lines = [contiguous_lines(ptr, row_bits(prog.layers[l].F_out), hw) for l in range(L)]

    for l in range(L):
        lay = prog.layers[l]
        li = {"layer": l, "F_in": lay.F_in, "F_out": lay.F_out,
              "comb_cycles_per_node": float(comb[l].mean()),
              "agg_cycles_per_subgraph": float(edges[l].agg_cycles.mean()),
              "ext_rows_per_node": float(edges[l].ext_rows.sum() / n),
              "condense_flushes": ctraf[l].flushes, "w_resident": wtraf[l]["resident"],
              "w_miss_rate": wtraf[l]["misses"] / max(wtraf[l]["accesses"], 1),
              "b_onchip": b_onchip[l]}
        if l > 0:   # encoded X̄⁽ˡ⁾ of one subgraph must fit one Input Buffer half (fusion)
            pl, pb = package_lines(xs[l], lay.bits, ptr, hw)
            _, bb = bitmap_lines(xs[l], ptr, hw)
            sub_bytes = (pb + bb) / 8
            li["x_sub_max_bytes"] = float(sub_bytes.max())
            li["x_spill_subgraphs"] = int((sub_bytes > hw.input_buf // 2).sum())
        info["layers"].append(li)
    info["eid_resident"] = bool(eid_resident)

    t = 0.0
    # ================================================================ sweep 0: combine layer 0
    s = SweepReport("comb L0", t, t)
    lines0 = dict(dram.lines)
    busy0 = dram.busy
    x0_pkg, _ = package_lines(prog.x0, prog.layers[0].bits, ptr, hw)
    x0_bm, _ = bitmap_lines(prog.x0, ptr, hw)
    w = wtraf[0]
    first_w = {"weights": w["load_lines"]}
    if eid_resident:
        first_w["eid"] = -(-eid_bytes // hw.line)
    _, w_done = dram.xfer(t, first_w)
    comb_end_prev = w_done
    comb_start_prev = t
    pending_w: dict = {}
    pending_t = 0.0
    comb0_sub, cond0_sub = seg_sum(comb[0], ptr), seg_sum(cond[0], ptr)
    for k in range(K):
        rd = {"x0_pkg": x0_pkg[k], "x0_bitmap": x0_bm[k], "meta": meta_lines[k],
              "w_miss": w["miss_lines"][k]}
        if not eid_resident:
            rd["eid"] = eid_lines_by_sub[k]
        first, done = dram.xfer(comb_start_prev, rd)       # prefetch into the other half
        if pending_w:
            dram.xfer(pending_t, pending_w)
        cs = max(comb_end_prev, first)
        work = max(comb0_sub[k], cond0_sub[k])
        ce = max(cs + work, done)
        pending_w = {"sparse_w": ctraf[0].flush_lines_by_src_sub[k]}
        if not b_onchip[0]:
            pending_w["b_int_w"] = b_lines[0][k]
        pending_t = ce
        comb_start_prev, comb_end_prev = cs, ce
    pending_w["sparse_w"] = pending_w.get("sparse_w", 0) + ctraf[0].end_flush_lines
    dram.xfer(pending_t, pending_w)
    s.end = max(comb_end_prev, dram.free) + hw.fill
    s.busy = {"comb": float(comb[0].sum()), "condense": float(cond[0].sum())}
    s.dram_lines = {k: v - lines0.get(k, 0) for k, v in dram.lines.items()}
    s.dram_busy = dram.busy - busy0
    sweeps.append(s)
    t = s.end

    # ================================================================ sweeps 1 … L
    for l in range(1, L + 1):
        prev = prog.layers[l - 1]                       # the layer being aggregated
        fused = l < L                                   # combine layer l in the same sweep
        s = SweepReport(f"agg L{l-1}" + (f" + comb L{l}" if fused else " + logits"), t, t)
        lines0, busy0 = dict(dram.lines), dram.busy
        if fused:
            w = wtraf[l]
            _, w_done = dram.xfer(t, {"weights": w["load_lines"]})
        else:
            w_done = t
        e = -(-prev.F_out // hw.qn_units)               # Encoder cycles per node (Q-26)
        if fused:
            fused_t = fused_enc_comb(comb[l], e, ptr)
            cond_t = seg_sum(cond[l], ptr)
        csc_bytes = edges[l - 1].edges * 1 + edges[l - 1].sources * 2   # 8-bit dst index + 16-bit count
        csc_off = np.concatenate([[0], np.cumsum(csc_bytes)])
        csc_lines = lines_touched(csc_off[:-1] * 8, csc_bytes * 8, hw.line * 8)
        logit_lines = contiguous_lines(ptr, 16 * prev.F_out, hw)   # 16-bit logits
        agg_end = [t] * K
        enc_end = [t] * K
        agg_start_prev, agg_end_prev, ec_end_prev = t, t, w_done
        pending_w, pending_t = {}, t
        for k in range(K):
            rd = {"csc": csc_lines[k], "sparse_r": ctraf[l - 1].read_lines[k]}
            if not b_onchip[l - 1]:
                rd["b_int_r"] = b_lines[l - 1][k]
            if fused:
                rd["meta"] = meta_lines[k]
            first, done = dram.xfer(agg_start_prev, rd)
            if pending_w:
                dram.xfer(pending_t, pending_w)
            a_s = max(agg_end_prev, enc_end[k - 2] if k >= 2 else t, first)
            a_e = max(a_s + edges[l - 1].agg_cycles[k], done)
            if fused:
                crd = {"w_miss": w["miss_lines"][k]}
                if not eid_resident:
                    crd["eid"] = eid_lines_by_sub[k]
                _, cdone = dram.xfer(a_s, crd)
                ec_s = max(a_e, ec_end_prev)
                enc_end[k] = ec_s + sizes[k] * e
                ec_e = max(ec_s + max(fused_t[k], cond_t[k]), cdone)
                pending_w = {"sparse_w": ctraf[l].flush_lines_by_src_sub[k]}
                if not b_onchip[l]:
                    pending_w["b_int_w"] = b_lines[l][k]
            else:
                ec_s = max(a_e, ec_end_prev)
                ec_e = enc_end[k] = ec_s + sizes[k] * e
                pending_w = {"logits": logit_lines[k]}
            pending_t = ec_e
            agg_start_prev, agg_end_prev, ec_end_prev = a_s, a_e, ec_e
            agg_end[k] = a_e
        if fused:
            pending_w["sparse_w"] = pending_w.get("sparse_w", 0) + ctraf[l].end_flush_lines
        dram.xfer(pending_t, pending_w)
        s.end = max(ec_end_prev, dram.free) + (hw.fill if fused else 0)
        s.busy = {"agg": float(edges[l - 1].agg_cycles.sum()), "enc": float(n * e)}
        if fused:
            s.busy.update(comb=float(comb[l].sum()), condense=float(cond[l].sum()))
        s.dram_lines = {k: v - lines0.get(k, 0) for k, v in dram.lines.items()}
        s.dram_busy = dram.busy - busy0
        sweeps.append(s)
        t = s.end

    dram_bytes = {k: v * hw.line for k, v in dram.lines.items() if v}
    return TimingReport(prog.name, hw, K, sweeps, t, dram_bytes, info)
