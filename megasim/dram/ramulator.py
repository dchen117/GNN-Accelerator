"""Ramulator 2.1 HBM replay of the timing model's DRAM traffic (S-13).

    rep = simulate_ramulator(prog, ptr, func)        # TimingReport with Ramulator timing
    rep.info["ramulator"]                            # row hits, latency, iterations, ...

How it works
------------
1. timing.simulate runs with the M2 bandwidth server and logs every DRAM transfer
   ("batch": issue cycle + lines per stream, call order).
2. `trace_from_log` gives every line a byte address: each stream has its own 256 MB region
   and is read/written sequentially inside it (packages, bitmaps, CSC, Sparse areas, B̄ rows
   are all contiguous per subgraph, S-10); W̄ row misses use their real row addresses.
3. The C++ driver (replay.cpp) feeds the lines to Ramulator's HBM1 model at their issue
   cycles and returns, per batch, the cycle its first and last line completed.
4. timing.simulate runs again with a LatencyDram: batch i now takes the latencies Ramulator
   measured for it. The new issue times go back to step 2, until the total cycle count
   changes by less than `tol` (fixed-point iteration). The call sequence is the same in every
   run (it does not depend on timing), so batch i is always the same transfer.

The bytes moved never change (they come from the traffic model); only stalls do.

Build first: megasim/dram/build.sh (needs Ramulator 2.1 built; see that script).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .. import timing

HERE = Path(__file__).resolve().parent
DRIVER = HERE / "replay"
CONFIG = HERE / "hbm1_256GBps.yaml"

REGION_BITS = 28                     # 256 MB per stream region; 16 regions = the 4 GB stack
# stream → region. A stream that is written in one sweep and read in the next shares the
# region with its reader (same addresses, same row behaviour).
REGION = {"x0_pkg": 0, "x0_bitmap": 1, "meta": 2, "weights": 3, "w_miss": 3, "eid": 4,
          "csc": 5, "b_int_w": 6, "b_int_r": 6, "sparse_w": 7, "sparse_r": 7, "logits": 8,
          # SGCN (sgcnsim/timing.py): CSR topology, CSR X⁽⁰⁾, W⁽⁰⁾ rows (cache misses), dense
          # T = X⁽⁰⁾W⁽⁰⁾, BEICSR features, systolic-array weights
          "sg_topo": 9, "sg_x0": 10, "sg_w": 11, "sg_t_w": 12, "sg_t_r": 12, "sg_x_w": 13,
          "sg_x_r": 13, "sg_wl": 14}
WRITES = {"b_int_w", "sparse_w", "logits", "sg_t_w", "sg_x_w"}
# sequential streams start at an offset inside the region so they do not overlap explicit
# addresses (W̄ misses use l · 16 MB + row offset, layers < 8)
SEQ_OFFSET = {"weights": 1 << 27}


@dataclass
class Trace:
    batch_t: np.ndarray       # (B,) int64 issue cycle, sorted
    batch_ptr: np.ndarray     # (B+1,) int64
    addr: np.ndarray          # (R,) uint64 byte address of each 64 B line
    write: np.ndarray         # (R,) uint8
    order: np.ndarray         # (B,) sorted position → index in the log


def trace_from_log(log: list, hw: timing.HW) -> Trace:
    """Line addresses for every logged transfer. Fractional line counts (e.g. 4 B eIDs) are
    accumulated per stream so the totals match the traffic model."""
    region = 1 << REGION_BITS
    cursor = {k: 0 for k in REGION}       # bytes issued so far per sequential stream
    acc = {k: 0.0 for k in REGION}        # fractional lines issued so far
    addrs, writes, counts = [], [], []
    for t, streams, explicit in log:
        n_batch = 0
        for name, v in streams.items():
            n = int(round(acc[name] + v) - round(acc[name]))
            acc[name] += v
            if n <= 0:
                continue
            base = REGION[name] << REGION_BITS
            if name in explicit:
                a = np.asarray(explicit[name], dtype=np.int64)
                assert a.size == n, (name, a.size, n)
                a = base + (a % region)
            else:
                off = SEQ_OFFSET.get(name, 0)
                a = base + (off + cursor[name] + np.arange(n, dtype=np.int64) * hw.line) % region
                cursor[name] += n * hw.line
            addrs.append(a)
            writes.append(np.full(n, name in WRITES, dtype=np.uint8))
            n_batch += n
        counts.append(n_batch)
    t = np.array([np.ceil(e[0]) for e in log], dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    # per-batch request blocks in log order, then reordered by issue time (stable)
    ptr_log = np.concatenate([[0], np.cumsum(counts)])
    addr_log = np.concatenate(addrs) if addrs else np.zeros(0, dtype=np.int64)
    wr_log = np.concatenate(writes) if writes else np.zeros(0, dtype=np.uint8)
    order = np.argsort(t, kind="stable")
    sel = np.concatenate([np.arange(ptr_log[b], ptr_log[b + 1]) for b in order]) if order.size else \
        np.zeros(0, dtype=np.int64)
    return Trace(batch_t=t[order], batch_ptr=np.concatenate([[0], np.cumsum(counts[order])]),
                 addr=addr_log[sel].astype(np.uint64), write=wr_log[sel], order=order)


def replay(trace: Trace, workdir: str | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run the driver. Returns (first, last) completion cycles indexed like the log (−1 for
    empty batches) and Ramulator's stats."""
    if not DRIVER.exists() or not CONFIG.exists():
        raise FileNotFoundError(f"{DRIVER} / {CONFIG} missing: run megasim/dram/build.sh")
    with tempfile.TemporaryDirectory(dir=workdir) as d:
        p = os.path.join(d, "tr")
        trace.batch_t.astype(np.int64).tofile(p + ".batch_t")
        trace.batch_ptr.astype(np.int64).tofile(p + ".batch_ptr")
        trace.addr.astype(np.uint64).tofile(p + ".addr")
        trace.write.astype(np.uint8).tofile(p + ".write")
        o = os.path.join(d, "out")
        subprocess.run([str(DRIVER), str(CONFIG), p, o], check=True, capture_output=True)
        first_s = np.fromfile(o + ".first", dtype=np.int64)
        last_s = np.fromfile(o + ".last", dtype=np.int64)
        stats = parse_stats(Path(o + ".stats").read_text())
    first = np.empty_like(first_s)
    last = np.empty_like(last_s)
    first[trace.order], last[trace.order] = first_s, last_s
    return first, last, stats


def parse_stats(text: str) -> dict:
    """Ramulator's stats dump is YAML; sum the per-channel controller counters."""
    raw = yaml.safe_load(text)
    ms = raw.get("MemorySystem", raw.get("memory_system", raw))
    ctrls = []

    def walk(node):
        if isinstance(node, dict):
            if "row_hits" in node:
                ctrls.append(node)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(ms)
    keys = ["row_hits", "row_misses", "row_conflicts", "num_read_reqs", "num_write_reqs",
            "read_latency", "num_read_reqs_served", "num_write_reqs_served"]
    out = {k: int(sum(float(c.get(k, 0)) for c in ctrls)) for k in keys}
    out["channels"] = len(ctrls)
    out["avg_read_latency"] = out["read_latency"] / max(out["num_read_reqs_served"], 1)
    out["driver"] = raw.get("driver", {})
    return out


def latencies(trace_t: np.ndarray, first: np.ndarray, last: np.ndarray):
    """Per-batch latency to first and last completion (0 for empty batches)."""
    empty = last < 0
    lf = np.where(empty, 0, first + 1 - trace_t)       # data usable the cycle after
    ll = np.where(empty, 0, last + 1 - trace_t)
    return lf.astype(np.float64), ll.astype(np.float64)


def fixed_point(run, hw, iters: int = 6, tol: float = 1e-3, workdir: str | None = None,
                verbose: bool = False):
    """Ramulator timing for any timing model (see module doc). `run(dram)` simulates with the
    given DRAM back-end (None = the M2 server) and returns a report with `.dram` (whose `.log`
    is the transfer log), `.total_cycles` and an `.info` dict. The call sequence of `run` must
    not depend on timing. Used for MEGA (simulate_ramulator) and SGCN (sgcnsim)."""
    rep = run(None)
    history = [rep.total_cycles]
    stats = {}
    for it in range(iters):
        tr = trace_from_log(rep.dram.log, hw)
        first, last, stats = replay(tr, workdir)
        t_log = np.array([np.ceil(e[0]) for e in rep.dram.log], dtype=np.int64)
        lf, ll = latencies(t_log, first, last)
        rep = run(timing.LatencyDram(hw, lf, ll))
        history.append(rep.total_cycles)
        if verbose:
            print(f"  ramulator iteration {it + 1}: {rep.total_cycles:.0f} cycles")
        if abs(history[-1] - history[-2]) <= tol * history[-2]:
            break
    lines = tr.addr.size
    rep.info["ramulator"] = {
        "iterations": len(history) - 1, "cycles_history": [float(h) for h in history],
        "converged": bool(abs(history[-1] - history[-2]) <= tol * history[-2]),
        "lines": int(lines), "row_hit_rate": stats["row_hits"] / max(
            stats["row_hits"] + stats["row_misses"] + stats["row_conflicts"], 1),
        **{k: v for k, v in stats.items() if k != "driver"},
        "refused_tx": int(stats["driver"].get("refused_tx", 0)),
        "replay_cycles": int(stats["driver"].get("cycles", 0))}
    return rep


def simulate_ramulator(prog, ptr, func, hw: timing.HW = timing.HW(), iters: int = 6,
                       tol: float = 1e-3, workdir: str | None = None,
                       verbose: bool = False) -> timing.TimingReport:
    """MEGA's timing model with Ramulator's HBM timing (fixed point, see module doc)."""
    return fixed_point(lambda d: timing.simulate(prog, ptr, func, hw, dram=d), hw, iters, tol,
                       workdir, verbose)
