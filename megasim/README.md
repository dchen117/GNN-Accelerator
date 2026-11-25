# megasim — MEGA simulator

The simulator of the MEGA baseline. Design and the reasons for each choice:
`docs/simulator/00-design.md` (S-xx). The hardware it implements: Q-16 … Q-30, Q-35 in
`docs/mega-kb/04-open-questions.md`.

State (2026-11-23): **M1 (functional, bit-exact), M2 (timing) and M3 (Ramulator 2.1 HBM1,
energy, Fig.14/16/17/18 quantities) are done.** The DRAM accounting was checked against the
paper (S-15): X̄⁽⁰⁾'s non-zero positions now go to DRAM as column indices where that is
smaller than the bitmaps. The simulator is ready to serve as the
baseline; our own design will go through the same DRAM model, energy numbers and reports.

## Run

```bash
# bit-exact check of the functional model against mega_pt.hw_stats.int_forward (Q-30)
python -m megasim.check runs/hw/* --json runs/hw/megasim_check.json
```

One-time setup for the DRAM model (S-13): build Ramulator 2.1, then our replay driver.
CACTI is only needed to regenerate `cacti_28nm.json`.

```bash
git clone https://github.com/CMU-SAFARI/ramulator2.git ~/ext/ramulator2
```

```bash
cd ~/ext/ramulator2 && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8
```

```bash
RAMULATOR=~/ext/ramulator2 megasim/dram/build.sh
```

```bash
# timing + energy: partition, functional run, Q-28 schedule, Ramulator HBM replay, energy
python -m megasim.sim runs/hw/* --json runs/hw/megasim_m3.json
```

```bash
# the same with the M2 bandwidth + latency DRAM server (no Ramulator needed)
python -m megasim.sim runs/hw/* --dram m2 --json runs/hw/megasim_timing.json
```

```bash
# Fig.14/16/17/18 quantities as a table (add --baseline other.json for ratios)
python -m megasim.report runs/hw/megasim_m3.json
```

```bash
# Fig.19(b) DRAM ablation in our accounting (S-15; diagnostic)
python -m megasim.ablation runs/hw/gcn_cora runs/hw/gcn_citeseer runs/hw/gcn_pubmed runs/hw/gcn_nell
```

```bash
# regenerate the SRAM numbers (needs CACTI 7: github.com/HewlettPackard/cacti, `make`)
CACTI=~/ext/cacti/cacti python -m megasim.cacti
```

```bash
python -m pytest tests/test_megasim.py tests/test_megasim_timing.py tests/test_megasim_m3.py -q
```

A run directory needs `model_seed<s>.pt`, written by `python -m mega_pt.train ... --save-model`.

## Code map

| File | What | Decisions |
|---|---|---|
| `program.py` | Exports a trained run into a hardware program: W̄, per-node bitwidth and compacted degree-table index, 16-bit fixed-point tables (α′, c, e, γ), X̄⁽⁰⁾, binary A+I in CSR and CSC | S-05, S-07, Q-10, Q-11, Q-16, Q-26 |
| `package.py` | Adaptive-Package encoder (greedy, smallest Mode) and Decoder (walks headers, uses the node bitwidths to find where a padded package ends); bitmap sizes | S-06, Q-17, Q-18, Q-19 |
| `functional.py` | The datapath per layer: store/reload, bit-serial combination with the pass/tile split, Tile Accumulator, rescale, AUs, Encoder; range checks on every accumulator | S-04, Q-16, Q-22, Q-23, Q-25, Q-26 |
| `check.py` | CLI: runs `functional` and `int_forward(fmt="fx16")` on the same model and compares every intermediate exactly | Q-30 |
| `partition.py` | METIS partition (fewest subgraphs that fit the Aggregation Buffer half) and relabelling to contiguous subgraph ranges | S-08, Q-25, Q-27 |
| `timing.py` | Per-unit cycle counts, DRAM traffic per stream, Condense region simulation, W̄ row cache, SRAM access counts, the sweep schedule as an event recurrence, DRAM back-ends (FCFS server, `LatencyDram`, `IdealDram`) | S-02, S-09 … S-14, Q-23 … Q-29, Q-35 |
| `tile_cycle.py` | Cycle-stepped model of the 4 Combination Tiles (crossbar, two C-PE halves, barrier per pass); checks the integers and the closed-form cycle count | S-03, Q-24 |
| `dram/hbm_config.py`, `dram/hbm1_256GBps.yaml` | Ramulator 2.1 config: HBM1 8 × 128 bit at 2 Gb/s, FR-FCFS, open row | S-13, Q-29 |
| `dram/replay.cpp`, `dram/build.sh` | C++ driver: replays the transfer log through Ramulator (per-channel DMA FIFOs, our address mapping), returns each transfer's first/last completion | S-13 |
| `dram/ramulator.py` | Transfer log → line addresses (one region per stream), runs the driver, fixed-point iteration with `timing.LatencyDram` | S-13 |
| `cacti.py`, `cacti_28nm.json` | CACTI 7.0 runs per buffer at 28 nm (organization calibrated to Tab.IV area) and their cached results | S-14 |
| `energy.py` | DRAM (7 pJ/bit), SRAM (CACTI per access), PU (Tab.IV power × busy cycles), leakage | S-14 |
| `ablation.py` | Fig.19(b) DRAM steps rebuilt from the M3 streams under several accountings (diagnostic) | S-15 |
| `lifetime.py` | How long a subgraph's X̄⁽¹⁾ would have to stay on chip for a residual (diagnostic; residual models are not simulated yet) | Q-36 |
| `report.py` | Fig.14/16/17/18-style table from a `sim.py --json` file, optionally as ratios to another design | — |
| `sim.py` | CLI: export → partition → relabel → functional → timing (Ramulator or M2 DRAM) → energy, per run | — |
| `synthetic.py` | Random but consistent programs, for tests and debugging without trained runs | — |

## Debugging

- `functional.run_layer` returns a `LayerResult` with every intermediate (`acc`, `B`, `psum`,
  `x_out`, `logits_code`), the width checks (`checks`) and the memory numbers (`memory`).
- `check.py` prints which intermediate first differs (`L<l>.equal`). The order inside a layer
  is x_in → acc → psum → (next x_in | logits).
- Any `*_overflow` / `au_may_saturate` that is true also fails the check: the model does not
  wrap or saturate, it reports.
- `package.decode` asserts on a header whose Bitwidth disagrees with the node's degree table,
  which is the first symptom of an encoder/decoder mismatch.
- Timing: `timing.simulate` returns a `TimingReport`; `.sweeps[i]` has each sweep's length,
  busy cycles per unit and DRAM lines per stream, `.info["layers"]` the per-layer inputs
  (cycles per node, external rows, W̄ residency/miss rate, Input Buffer fit). A sweep that is
  much longer than its busiest unit and its `dram_busy` points at a dependency stall.
- `tile_cycle.simulate` is slow (Python per cycle); use it on a handful of nodes only.
- DRAM: `report.dram.log` is the list of transfers (issue cycle, lines per stream, explicit
  addresses). `ramulator.trace_from_log` + `ramulator.replay` replay any log by hand; the
  driver's `.stats` (Ramulator's YAML) lands in `rep.info["ramulator"]` (row hits/misses/
  conflicts, average read latency, refused requests = full controller queues). If the fixed
  point does not converge (`converged: false`), `cycles_history` shows whether it oscillates.
- `replay` raises "batch did not complete" if a request never returned. That means a driver or
  config mismatch: `kChannels` / `kChannelShift` in `replay.cpp` must match `hbm_config.py`.
- Energy: `energy.energy(rep)` gives per-buffer (`sram_uJ`) and per-unit (`pu_uJ`) splits;
  `rep.info["sram"]` holds the bytes behind them.
- Without trained runs: `megasim.synthetic.random_program()` gives a program both models run on.

## Results: M1 functional check (2026-11-20)

All 8 saved models in `runs/hw` are **bit-exact** (X̄⁽⁰⁾, acc, psum, X̄⁽¹⁾, logits), and the
simulated test accuracy equals `hw_stats`' fx16 integer accuracy. Raw output:
`runs/hw/megasim_check.json` (written by the command above; `runs/` is not committed). The NELL model is the collapsed run (1.2 %, see `mega_pt/README.md`);
it is kept as a size and functional test only.

Per layer (L0 = X̄⁽⁰⁾, L1 = X̄⁽¹⁾). *values/el* is the Tab.VI-comparable number (Q-15), *stored/el*
includes package headers and padding and the bitmaps (Q-19). SA, TA = largest Shifter-Acc and
Tile Accumulator value; AU bound = largest Σ|B̄| into one partial sum (< 32,768 means no
saturation in any order).

| Run | L | density | values/el | stored/el | CR stored | pad % | SA max | TA max | AU bound |
|---|---|---|---|---|---|---|---|---|---|
| gcn_cora | 0 | 0.013 | 0.025 | 1.031 | 31.0 | 3.3 | 45 | 126 | 394 |
| | 1 | 0.690 | 2.532 | 3.744 | 8.5 | 7.7 | 545 | 1874 | 356 |
| gcn_citeseer | 0 | 0.009 | 0.017 | 1.020 | 31.4 | 3.2 | 150 | 330 | 170 |
| | 1 | 0.696 | 2.251 | 3.428 | 9.3 | 7.3 | 254 | 911 | 123 |
| gcn_pubmed | 0 | 0.066 | 0.132 | 1.160 | 27.6 | 3.2 | 75 | 128 | 316 |
| | 1 | 0.608 | 1.730 | 2.885 | 11.1 | 8.2 | 360 | 1256 | 314 |
| gin_cora | 0 | 0.013 | 0.025 | 1.031 | 31.0 | 3.3 | 75 | 177 | 450 |
| | 1 | 0.715 | 2.625 | 3.836 | 8.3 | 7.4 | 416 | 1292 | 361 |
| gin_citeseer | 0 | 0.009 | 0.017 | 1.020 | 31.4 | 3.1 | 45 | 139 | 190 |
| | 1 | 0.573 | 1.605 | 2.739 | 11.7 | 7.7 | 136 | 476 | 197 |
| gin_amazon-ratings | 0 | 0.722 | 1.451 | 2.566 | 12.5 | 3.2 | 680 | 737 | 247 |
| | 1 | 0.281 | 0.614 | 1.708 | 18.7 | 13.2 | 270 | 875 | 236 |
| sage_cora | 0 | 0.013 | 0.025 | 1.031 | 31.0 | 4.3 | 48 | 104 | 90 |
| | 1 | 0.659 | 2.074 | 3.175 | 10.1 | 4.6 | 708 | 2735 | 132 |

What this shows:
- **Widths hold with margin.** Shifter-Acc ≤ 708 (16-bit limit 32,767), Tile Accumulator
  ≤ 2,735 (32-bit), AU bound ≤ 2,124 incl. NELL. No overflow or saturation anywhere.
- **The bitmap costs about 1 bit per element in the stored format.** On hidden layers the
  stored size is values + ~1.0 bit (bitmap) + 5–13 % package padding, so CR_stored is roughly
  8–19× where the values-only number alone would suggest 12–20×. On X⁽⁰⁾ of bag-of-words
  sets the bitmap is nearly everything (CR_stored ≈ 31). This is MEGA's stated format (Q-19);
  it matters for the DRAM traffic M2 will report.
- **Most packages are full long ones** (192 bits); short/medium ones appear where the bitwidth
  changes between consecutive nodes, which is frequent in hidden layers (nodes are in dataset
  order; METIS relabelling in M2 will change this).

## Results: M2 timing (2026-11-21)

Cycles at 1 GHz; DRAM = 256 B/cycle + 100 cycles latency (S-12). Sweeps: comb L0 / agg L0 +
comb L1 / agg L1 + logits. Util = busy cycles / total. Raw output: `runs/hw/megasim_timing.json`.

| Run | Subgraphs | kcycles | per sweep (k) | comb util | agg util | DRAM MB | largest DRAM streams |
|---|---|---|---|---|---|---|---|
| gcn_cora | 11 | 59.0 | 21.9 / 32.9 / 4.2 | 0.91 | 0.19 | 1.70 | w_miss 40%, x0_bitmap 27%, b_int_w 10% |
| gcn_citeseer | 14 | 91.6 | 51.6 / 35.4 / 4.6 | 0.94 | 0.12 | 5.03 | w_miss 59%, x0_bitmap 29%, b_int_w 4% |
| gcn_pubmed | 80 | 441.8 | 244.2 / 159.6 / 38.0 | 0.91 | 0.23 | 7.20 | sparse_w 22%, x0_bitmap 17%, b_int_w 17% |
| gcn_nell | 383 | 16,742.5 | 15,781.4 / 476.2 / 484.8 | 0.96 | 0.03 | 556.28 | x0_bitmap 86%, logits 4%, sparse_w 3% |
| gin_cora | 11 | 60.4 | 21.9 / 34.3 / 4.2 | 0.91 | 0.19 | 1.70 | w_miss 40%, x0_bitmap 27%, b_int_w 10% |
| gin_citeseer | 14 | 82.6 | 51.6 / 26.4 / 4.6 | 0.94 | 0.13 | 5.03 | w_miss 59%, x0_bitmap 29%, b_int_w 4% |
| gin_amazon-ratings | 99 | 1,576.2 | 1,427.8 / 115.1 / 33.4 | 0.96 | 0.09 | 6.98 | b_int_w 22%, b_int_r 22%, x0_pkg 19% |
| sage_cora | 22 | 98.1 | 43.8 / 49.6 / 4.7 | 0.94 | 0.17 | 4.21 | w_miss 63%, x0_bitmap 11%, b_int_w 8% |

What this shows:
- **MEGA is compute-bound (combination) everywhere.** The Combination Tiles are busy 91–96 %
  of the time; DRAM is busy at most ~37 % of a sweep (CiteSeer sweep 0). This matches the
  paper's remark that DRAM cycles are mostly overlapped (§VI-D-2).
- **NELL is bound by the bitmap scan**: 61,278 features → 240 cycles per node at 256 bits/cycle
  (Q-23), 94 % of the run. In this M2 accounting its X⁽⁰⁾ bitmaps were 86 % of the DRAM traffic
  (480 MB). Since S-15 (2026-11-23), DRAM carries 16-bit column indices instead (0.8 MB); the
  scan time is unchanged.
- **W̄⁽⁰⁾ misses dominate DRAM on Cora/CiteSeer/Sage-Cora** (40–63 %): W̄⁽⁰⁾ is 90–231 KB
  against the 48 KB Weight Buffer (Q-35); the direct-mapped row cache misses on 22–46 % of
  accesses.
- **Hidden-layer X̄ never reaches DRAM**: the Q-28 fusion keeps it in the Input Buffer (largest
  subgraph 16.7 KB of 32 KB). So the Adaptive-Package compression of hidden layers saves
  on-chip space, not DRAM traffic, in this schedule; DRAM sees X⁽⁰⁾, W̄, and 4-bit B̄ rows.
  - Fig.19's 2.5× from Adaptive-Package: a hidden-layer DRAM round trip does not explain it
    either (1.07–1.14×, checked 2026-11-23 with `megasim.ablation`). Fusion stays; see S-15.
- NELL has 383 subgraphs, not 257: its 186-class output layer sets the capacity to 176 (S-08).

## Results: M3 Ramulator + energy (2026-11-22)

`python -m megasim.sim runs/hw/* --json runs/hw/megasim_m3.json`, then `python -m megasim.report`.
HBM1 256 GB/s in Ramulator 2.1 (S-13), energy per S-14. *DRAM stall* = share of the time lost to
DRAM (Fig.20(a)): 1 − time with data arriving instantly / time.

Updated 2026-11-23 with S-15's X̄⁽⁰⁾ position format (column indices where smaller; times are
unchanged). The original bitmap accounting gave Cora 1.70 MB / 119.2 µJ, CiteSeer 5.03 / 323.3,
NELL 556.3 / 36,331, GIN-Cora 1.70, GIN-CiteSeer 5.03, Sage-Cora 4.21 (`--x0-format bitmap`).

| Run | K | Time µs | DRAM MB | Energy µJ | DRAM % | SRAM % | PU % | Leakage % | Row hit | DRAM stall | X̄⁽⁰⁾ positions |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gcn_cora | 11 | 58.8 | 1.33 | 97.8 | 80.1 | 9.2 | 1.6 | 9.1 | 0.91 | 0.8 % | index |
| gcn_citeseer | 14 | 91.0 | 3.76 | 249.0 | 88.7 | 4.9 | 0.9 | 5.5 | 0.95 | 0.3 % | index |
| gcn_pubmed | 80 | 441.9 | 7.20 | 573.5 | 73.8 | 12.5 | 2.1 | 11.6 | 0.87 | 0.2 % | bitmap |
| gcn_nell | 383 | 16,687 | 76.19 | 8,140 | 55.0 | 9.5 | 4.5 | 31.0 | 0.91 | 0.0 % | index |
| gin_cora | 11 | 60.0 | 1.33 | 98.0 | 79.9 | 9.2 | 1.6 | 9.2 | 0.91 | 0.5 % | index |
| gin_citeseer | 14 | 82.7 | 3.76 | 247.2 | 89.3 | 4.8 | 0.8 | 5.1 | 0.95 | 1.2 % | index |
| gin_amazon-ratings | 99 | 1,570 | 6.98 | 895.9 | 45.8 | 23.6 | 4.2 | 26.5 | 0.86 | 0.0 % | bitmap |
| sage_cora | 22 | 97.5 | 3.84 | 260.3 | 86.7 | 6.7 | 1.0 | 5.7 | 0.94 | 0.5 % | index |

Findings:
- **Ramulator changes the time by at most 0.7 % compared with the M2 server** (−0.7 … +0.1 %).
  - Average read latency is 50–64 cycles, below M2's assumed 100.
  - The fixed point converges in 1–5 iterations.
  - DRAM stalls are 0–1.2 % of the time. This agrees with MEGA's remark that most DRAM cycles
    are overlapped (§VI-D-2) and with Fig.20(a), where MEGA has the lowest stall share.
- **DRAM dominates energy** (74–89 %), except on amazon-ratings (46 %) and NELL (55 %).
  - Amazon-ratings: W̄ is resident, X⁽⁰⁾ is dense and small, and the run is long, so leakage
    and SRAM weigh more.
  - NELL: without its input bitmaps in DRAM, its 16.7 ms run makes leakage 31 %. Fig.1 cites
  90.2 % DRAM energy for a prior accelerator. MEGA's own split is only given relative to
  HyGCN (Fig.18).
- **On-chip power** (SRAM + PU + leakage over the run) is 220–356 mW, compared with Tab.IV's
  195 mW.
  - CACTI's leakage alone is 151 mW: that is the itrs-lop cell choice, which matches Tab.IV
    buffer area and power best (S-14).
  - SRAM dynamic energy is mostly Aggregation Buffer read-modify-writes (16-bit partial sums
    per edge) and Weight Buffer crossbar reads.
- High row-hit rates (0.86–0.97): almost all streams are sequential by construction (S-10).
  W̄ row misses are the scattered part.
