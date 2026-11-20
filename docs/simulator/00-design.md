# MEGA simulator — design

Status: **M1 (functional model) done and bit-exact on all 8 saved models**, 2026-11-20
(results: `megasim/README.md`). M2 (timing) next. The hardware decisions it implements
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
| **Timing model** (`megasim/timing.py`, M2) | cycles per unit, overlap, stalls, DRAM traffic | per-unit closed forms checked against a small cycle-stepped model of one tile (S-03) |

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

## 3. Milestones

| | Content | State |
|---|---|---|
| M1 | program export, Adaptive-Package encode/decode, functional datapath, bit-exact check on all `runs/hw` models | **done** 2026-11-20 |
| M2 | partition + relabel, per-unit timing, Q-28 event schedule, Condense-Edge (B2 + batch-min skip), DRAM streams with a bandwidth model, one-tile cycle-stepped check (S-03) | — |
| M3 | Ramulator 2.0 HBM, energy (CACTI-style SRAM per access from Tab.IV power, DRAM pJ/bit, Q-29), reports matching Fig.14/16/17/18 | — |
