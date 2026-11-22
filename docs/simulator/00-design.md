# MEGA simulator — design

Status: **M1 (functional model) done and bit-exact on all 8 saved models**, 2026-11-20.
**M2 (timing model) done**, 2026-11-21: partition + relabel, per-unit cycle counts, the Q-28
sweep schedule, Condense-Edge, DRAM streams with a bandwidth model, and the cycle-stepped tile
check (results: `megasim/README.md`). **M3 done**, 2026-11-22: Ramulator 2.1 HBM1 replay,
energy (DRAM / SRAM / PU / leakage) and the Fig.14/16/17/18 quantities. The hardware decisions it implements
are Q-16 … Q-30 and Q-35 in `docs/mega-kb/04-open-questions.md`; this file records only the
decisions about *how the simulator is built* (S-xx). Each lists why and what it costs.

---

## 1. Goal

A simulator of MEGA, used as the baseline for our own design. It must report what the paper
reports (Fig.14 speedup, Fig.16 DRAM accesses, Fig.17/18 energy) and be fair: our own design
will later run through the same DRAM model, the same energy numbers and the same datasets.

It is split into two parts that share one data layout:

| Part | What it answers | Check |
|---|---|---|
| **Functional model** (`megasim/functional.py`) | the exact integers every unit produces, and the bytes it stores | bit-exact against `mega_pt.hw_stats.int_forward` (Q-30) |
| **Timing model** (`megasim/timing.py`) | cycles per unit, overlap, stalls, DRAM traffic | per-unit closed forms checked against a small cycle-stepped model of one tile (S-03) |

## 2. Decisions

### S-01 — Python + NumPy, vectorized over nodes
- Why: same toolchain as `mega_pt/` (loaders, trained runs, `int_forward`), so the bit-exact
  check needs no glue. NumPy over nodes is fast enough for Reddit (233 K nodes).
- Cost: a per-cycle loop in Python is far too slow for the full graphs (Reddit is about 10⁹
  cycles). That is why S-02 does not step cycles for the whole run.
- Alternative: C++ cycle-stepped simulator. Not needed for the reasons in S-02; can be revisited
  if a unit turns out to have timing that depends on cycle-level interaction.

### S-02 — Timing: exact per-unit cycle counts + an event model at node / subgraph granularity
- Every compute unit we decided on has **data-dependent but contention-free** timing: a node's
  combination cost is Σ passes max over tiles (waves·b), bounded by the bitmap scan (Q-23/24);
  the AUs take ⌈deg_out / k⌉ cycles per source (Q-25); the Encoder takes F2/32 cycles per node
  (Q-26). These closed forms give the same number a cycle-stepped model of the same RTL would.
- What *does* interact — units waiting on each other through ping-pong buffers, and DRAM
  latency/bandwidth — is modelled as a discrete-event pipeline whose events are nodes (inside a
  subgraph) and subgraphs (between buffers), per the Q-28 schedule.
- DRAM: each unit emits a request stream (address, bytes, read/write) per Q-29's regions. M2
  uses a bandwidth + fixed-latency model; M3 replays the same streams through Ramulator 2.0
  (HBM, 256 GB/s). Both designs use the same path.
- Why: this is the standard way accelerator papers' "cycle-accurate" simulators are built for
  graph-sized inputs, and it keeps the run time to minutes.
- Cost: no modelling below the node level (e.g. a crossbar conflict inside a wave). Under our
  decisions there are none (Q-24: b ≥ 2 keeps the crossbar off the critical path).

### S-03 — A cycle-stepped model of one Combination Tile, for checking only
- A small model that steps the Fig.11 systolic array (8 BSE rows × 32 C-PEs, two halves, weight
  reuse over bit-planes) cycle by cycle on random nodes. Its cycle count and outputs must equal
  the closed form of S-02 and the functional model. (M2.)
- Why: Q-24's timing is ours; this is the cheapest way to make sure the closed form matches the
  schedule we wrote down.

### S-04 — The functional model computes bit-planes, not per-cycle values
- The Decoder emits b-bit two's-complement values as bit-planes, LSB first; each C-PE ANDs a
  plane with its weights, sums with the adder tree and shift-accumulates, subtracting on the
  MSB plane (Q-22). The model does exactly this, but for all nodes at once per plane:
  acc = Σ_k w_k · (P_k · W̄), w_k = 2^k, w_{b−1} = −2^{b−1}, P_k ∈ {0,1}.
- The pass/tile split (≤ 128 non-zeros per pass, ≤ 32 per tile, Q-23) is reproduced for the
  width checks: every Shifter-Acc (16-bit), Tile Accumulator (32-bit) and AU (16-bit
  saturating) value is range-checked, and any overflow/saturation is counted.
- Why: bit-exact results without a cycle loop.

### S-05 — Scale factors are integer codes with a per-table shift
- Hardware fixed point (Q-16): code = round(v · 2^f), f = 16 − ⌈log₂ max v⌉ per table.
  Requantization is integer: (acc · A · C + 2^(s−1)) >> s with sign handled as
  round-half-away, s = f_A + f_C. This is the same number `int_forward(fmt="fx16")` computes in
  float64 (the products stay below 2⁵³), so the check is exact equality, not a tolerance.

### S-06 — The storage format is materialized as real bytes
- Packages (Q-17/18/19) are written into a byte stream and decoded back by the Decoder model;
  the round trip is checked. The memory numbers (stored bits, CR_stored) come from these bytes,
  not from a formula.
- Bitmaps are *counted* (F padded to 32 bits per node) but by default not materialized, because
  NELL's X⁽⁰⁾ bitmap alone is about 0.5 GB. The non-zero positions carry the same information;
  `--materialize-bitmaps` builds them for small graphs.

### S-07 — Inputs come from trained runs
- `megasim/program.py` turns a run directory (`model_seed<s>.pt`, from `train.py --save-model`)
  into a hardware program: W̄ per layer, per-node bitwidths and degree-table indices (Q-26),
  the fixed-point tables, X̄⁽⁰⁾ (quantized offline, Q-10), and the graph (A+I, binary under
  fold, Q-11; for GraphSage the fixed evaluation sample).
- METIS partitioning and relabelling (Q-27) happen here too (M2), so the functional and timing
  models see the same node order.

### S-08 — Partition: fewest subgraphs that fit, METIS's default imbalance, one partition for all layers
- Capacity per subgraph = one Aggregation Buffer half / (2 B · F2) with F2 the **widest
  aggregated layer** (Q-25), because the same node order serves every layer (relabelling per
  layer would need every row re-laid in DRAM between layers).
  - Consequence: NELL's output layer has 186 classes, so its capacity is 176, not 256, and it
    gets 383 subgraphs instead of Q-27's estimate of 257. Sage-Cora (F2 = 256) gets 22.
- K starts at ⌈N / capacity⌉; METIS runs with its default tolerance (ufactor 30 = 3 %); if a
  part overflows, K += 1 and METIS reruns (NELL: 10 runs; the others: 1).
- Why not a tight tolerance: ufactor = 1 almost triples the edge cut on Cora (0.98 external
  rows per node instead of 0.35), i.e. three times the Condense-Edge traffic.
- Relabelling: subgraph k = contiguous id range, original order inside. The functional model
  on the relabelled program gives the permuted outputs exactly (tested).

### S-09 — Combination cost per node: one bitmap scan, planes replayed per column block
- Node cycles = max(⌈F_pad / 256⌉, blocks · Σ_pass max_tile waves·b) (Q-23/24). The bitmap is
  scanned once per node, overlapped with the previous node's compute; for each of the
  ⌈F2/32⌉ column blocks the bit-planes are replayed from the Bit FIFO (Q-21).
- Assumption (ours, not stated by the paper): the Bit FIFO holds one node's planes
  (≤ nnz · b bits; ≤ 256 · 8 bits on hidden layers). `mega_pt/hw_stats.py` charged the scan
  per block; that only matters for sparse X⁽⁰⁾ with F2 > 32.
- Checked by S-03: the cycle-stepped tiles give exactly this count + 3 drain cycles, with no
  crossbar stall at b ≥ 2 (and stalls at b = 1, as Q-24 argues).
- Drain: 3 cycles (half skew, adder tree, Shifter-Acc register) in the cycle model, because
  Q-24 puts the crossbar read in the same cycle as plane 0. The timing model keeps Q-24's 4
  per sweep; the difference is 1 cycle per sweep.

### S-10 — DRAM traffic: 64 B lines, one region per stream, and the residency rules
Every stream has its own contiguous region; a read or write of a contiguous range costs the
64 B lines it touches. Per stream:

| Stream | When | Size / rule |
|---|---|---|
| `x0_pkg`, `x0_bitmap` | sweep 0, per subgraph | the subgraph's slice of the package stream (boundary packages read by both neighbours) and its bitmaps (F padded to 32 bits) |
| `meta` | every combination sweep | 16-bit degree-table index per node (Q-26) |
| `weights` | start of each combination sweep | W̄ + tables, if they fit the 48 KB Weight Buffer |
| `w_miss` | during combination | otherwise the Weight Buffer is a **direct-mapped cache of W̄ rows** (row r → slot r mod S); a node fetches each of its non-zero rows once |
| `b_int_w`, `b_int_r` | write in sweep l, read in sweep l+1 | B̄ rows of each subgraph (F2 · 4 bits each), contiguous; kept on chip instead if the whole layer fits one Combination Buffer half (48 KB): true for every output layer here |
| `sparse_w`, `sparse_r` | Condense flushes / aggregation | write-combining regions (Q-27); flushes at full, region switch, and end of sweep; partial lines counted as whole lines |
| `eid` | combination sweeps | eID lists (4 B each) + 12 B per subgraph; loaded once if they fit half the Edge Buffer (12 KB), else streamed every combination sweep |
| `csc` | aggregation sweeps | per subgraph: 1 B destination index per edge (≤ 256 destinations) + 2 B count per source |
| `logits` | final sweep | 16-bit per class |

- **X̄⁽ˡ⁾ for l ≥ 1 never goes to DRAM**: under the Q-28 fusion the Encoder writes a
  subgraph's packages into the Input Buffer and combination consumes them at once. The model
  checks that one subgraph's packages + bitmaps fit one Input Buffer half (32 KB); largest
  seen 16.7 KB, so no spills.
- Why direct-mapped for the W̄ cache: it is the simplest hardware and the traffic is exact and
  vectorizable (an LRU over Reddit's 140 M accesses is not). Q-35 left the policy open.
- Why the Sparse regions flush at the end of a sweep: the next sweep's Condense needs them
  for the next layer while aggregation reads this layer's rows; a region cannot serve both.

### S-11 — The schedule as an event recurrence over subgraphs
Per sweep l ≥ 1, for subgraph k (Q-28):
- inputs (CSC, external rows, internal B̄ rows, metadata) are issued when subgraph k−1 starts
  aggregating (the other buffer half), and consumed as they stream in;
- aggregation of k starts after aggregation of k−1, after the Encoder has drained subgraph
  k−2's Aggregation Buffer half, and after its first data arrives; it ends no earlier than its
  last data;
- Encoder + combination of k start after aggregation of k and after combination of k−1; the
  fused time is max_i [(i+1)·e + Σ_{j≥i} comb_j] (node i is encoded at (i+1)·e, e = ⌈F2/32⌉);
  Condense runs alongside and stretches the stage if it is slower;
- writes (B̄ rows, Sparse flushes, logits) are posted when the stage ends.
Sweep 0 is the same without aggregation; the final sweep has no combination. A sweep ends when
its last stage and all its writes are done (the next sweep reads them), plus the fill.

### S-12 — DRAM in M2: one FCFS bandwidth server with a fixed latency
- 256 B/cycle (256 GB/s at 1 GHz), 100 cycles latency per transfer (ours, typical HBM read
  latency); transfers are booked in issue order; writes take bandwidth but block nobody.
- M3 replays the same streams through Ramulator (S-13). Traffic (bytes) does not depend on the
  timing model, only stalls do (tested). The M2 server stays available (`--dram m2`).

### S-13 — DRAM in M3: Ramulator 2.1 HBM1, coupled by fixed-point iteration
- **Simulator:** Ramulator 2.1 (CMU-SAFARI/ramulator2, commit 7bead6cb, 2026-11-04; it is the
  current release of "Ramulator 2.0" and still models HBM1). Built outside the repo;
  `megasim/dram/build.sh` builds our C++ driver `megasim/dram/replay.cpp` against it and
  exports the config `megasim/dram/hbm1_256GBps.yaml` from `megasim/dram/hbm_config.py`.
- **Device (Q-29):** HBM1, 1 stack, 8 channels × 128 bit, preset `HBM1_4Gb` / `HBM1_2Gbps`
  (2 Gb/s per pin → 32 GB/s per channel → 256 GB/s). The command clock is 1 GHz, equal to
  the accelerator clock, so one memory tick = one cycle. Controller: Ramulator's HBM12 with
  FR-FCFS, open-row policy and all-bank refresh, 32-entry read and write queues per channel
  (Ramulator defaults).
  - Why 2 Gb/s: HBM1 at 1 Gb/s gives only 128 GB/s per stack. The paper states both "HBM1.0"
    and "256 GB/s"; HBM1's 2 Gb/s grade gives both with one stack, which is what Q-29 decided.
    (HyGCN, whose setup MEGA copies, does not give its stack count either.)
- **Address mapping (ours):** from MSB to LSB, row | bank | column[5:1] | bank group | channel
  | column[0] | byte. 64 B lines go round the 8 channels, then a channel's 4 bank groups,
  then along the open 2 KB row; both 32 B halves of a line (HBM1 BL2 = 32 B) stay in one
  row. Implemented in `replay.cpp` `remap()`; Ramulator decodes the result with RoBaRaCoCh.
  - Why: a sequential stream reaches 92.6 % of 256 GB/s (1 MB in 4421 cycles; the ideal is
    4096). Ramulator's RoBaRaCoCh alone keeps a stream in one bank group, where tCCD_L = 2
    halves the column rate (69 %). MOP4CLXOR reaches 84 %, but its XOR splits the two halves
    of a line over two banks, so scattered 64 B W̄ fetches open two rows each.
  - Random 64 B lines reach 39 % of peak (1 MB in 10.5 k cycles), the usual HBM behaviour.
- **Addresses:** each stream has its own 256 MB region and is walked sequentially (packages,
  bitmaps, CSC, Sparse areas, B̄ rows are contiguous per subgraph, S-10). W̄ row misses use
  their real row addresses (`w_miss_addrs`). A stream written in one sweep and read in the
  next (B̄, Sparse) shares a region with its reader.
- **Requester (ours):** the DMA keeps one FIFO per channel. A transfer's lines enter at its
  issue cycle and are offered to the controller at up to 2 × 32 B per channel per cycle (one
  line, i.e. the channel's peak). A refused request (queue full) is retried the next cycle.
  Reads complete when Ramulator returns the data. Writes complete when their WR issues
  (Ramulator's convention).
- **Coupling:** the timing model makes the same sequence of DRAM transfers whatever the
  timing (the issue times change, the transfers do not). So:
  1. run the timing model with the M2 server and log every transfer;
  2. replay the log in Ramulator and measure each transfer's latency to first and last data;
  3. rerun the timing model with those latencies (`timing.LatencyDram`), which gives new
     issue times;
  4. repeat steps 2–3 until the total time changes by less than 0.1 %.

  This converges in 3–6 iterations here, because DRAM is lightly loaded and the system is
  compute-bound.
  - Why not run Ramulator inside the event loop: the event recurrence computes subgraph k's
    times before issuing k+1's transfers, but some of k's posted writes are issued later than
    k+1's reads. A lock-step co-simulation would need the event model rewritten as a cycle
    loop; the fixed point gives the same answer at the converged schedule.
  - Cost: queueing interactions are those of the previous iteration's schedule; at
    convergence the two agree to within 0.1 % in time.
- **DRAM stall share** (Fig.20(a)) = 1 − (time with `timing.IdealDram`: data arrives the cycle
  it is requested) / time. It is not "the M2 server with infinite bandwidth": that server books
  transfers in call order, so a read called after a later-issued posted write waits for it.
  This is a small artifact of the M2 server, about 0.04 % on NELL; Ramulator queues by
  arrival time and does not have it.

### S-14 — Energy: DRAM pJ/bit, CACTI 7.0 SRAM, Tab.IV PU power
Fig.18's four categories:
- **DRAM:** 7 pJ/bit for every byte moved. This is HyGCN's HBM1.0 figure (HyGCN cites
  [32, 41] for it), and MEGA says its DRAM energy "follows the methodology in [56]" =
  HyGCN. Fig.18's DRAM column equals Fig.16's access ratio exactly, which also means a fixed
  energy per bit.
  - Alternative: Ramulator 2.1's Ayna HBM power model. It does not support HBM1, and it
    would not be MEGA's methodology.
- **SRAM:** CACTI 7.0 (HewlettPackard/cacti, 1ffd8df) at 28 nm, as MEGA states.
  - Each Tab.IV buffer is modelled as a scratch RAM with one R/W port and a 64 B access.
  - Configuration chosen by matching Tab.IV's measured area. We swept cell type
    (itrs-hp / lop / lstp) and bank count (1–32). itrs-lop with 8 banks (Aggregation,
    Combination) and 4 banks (the rest) matches areas within 0–24 % (1.48 vs 1.67 mm²), and
    its leakage within 2–23 % of Tab.IV's buffer powers (151 vs 141 mW).
  - Results cached in `megasim/cacti_28nm.json`; `python -m megasim.cacti` regenerates them.
  - Energy = bytes read / 64 × read energy + bytes written / 64 × write energy.
    `timing.sram_accesses` counts the bytes per buffer; its docstring lists what counts as
    an access.
- **PU:** Tab.IV power of each unit × the cycles that unit is busy (BSEs, crossbar and
  Decoder with combination; AUs with aggregation; Encoder; Condense). "Others" is charged
  for the whole run.
  - Ours: Tab.IV's numbers are read as a unit's power while working, and idle units are
    treated as clock-gated. The paper does not say how power becomes energy.
- **Leakage:** CACTI leakage of all buffers × run time. PU leakage stays inside the PU
  numbers.
- **Check:** on-chip power from the model, (SRAM + PU + leakage) / time, can be compared
  with Tab.IV's 195 mW (README, M3 results).

## 3. Milestones

| | Content | State |
|---|---|---|
| M1 | program export, Adaptive-Package encode/decode, functional datapath, bit-exact check on all `runs/hw` models | **done** 2026-11-20 |
| M2 | partition + relabel, per-unit timing, Q-28 event schedule, Condense-Edge (B2 + batch-min skip), DRAM streams with a bandwidth model, one-tile cycle-stepped check (S-03) | **done** 2026-11-21 |
| M3 | Ramulator 2.1 HBM1 replay (S-13), energy with CACTI 7.0 SRAM, 7 pJ/bit DRAM and Tab.IV PU power (S-14), reports matching Fig.14/16/17/18 (`megasim/report.py`) | **done** 2026-11-22 |
