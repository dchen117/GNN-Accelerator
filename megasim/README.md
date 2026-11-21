# megasim — MEGA simulator

The simulator of the MEGA baseline. Design and the reasons for each choice:
`docs/simulator/00-design.md` (S-xx). The hardware it implements: Q-16 … Q-30, Q-35 in
`docs/mega-kb/04-open-questions.md`.

State (2026-11-21): **M1, the functional model, is done and bit-exact. M2, the timing model, is
done** (cycles, unit utilization, DRAM traffic per stream). M3 (Ramulator 2.0, energy, the
Fig.14/16/17/18 reports) is next.

## Run

```bash
# bit-exact check of the functional model against mega_pt.hw_stats.int_forward (Q-30)
python -m megasim.check runs/hw/* --json runs/hw/megasim_check.json
```

```bash
# timing: METIS partition + relabel, functional run, Q-28 schedule, DRAM streams (M2)
python -m megasim.sim runs/hw/* --json runs/hw/megasim_timing.json
```

```bash
python -m pytest tests/test_megasim.py tests/test_megasim_timing.py -q
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
| `timing.py` | Per-unit cycle counts, DRAM traffic per stream, Condense region simulation, W̄ row cache, the sweep schedule as an event recurrence, FCFS DRAM server | S-02, S-09 … S-12, Q-23 … Q-29, Q-35 |
| `tile_cycle.py` | Cycle-stepped model of the 4 Combination Tiles (crossbar, two C-PE halves, barrier per pass); checks the integers and the closed-form cycle count | S-03, Q-24 |
| `sim.py` | CLI: export → partition → relabel → functional → timing, per run | — |
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
  (Q-23), 94 % of the run, and its X⁽⁰⁾ bitmaps are 86 % of the DRAM traffic (480 MB).
- **W̄⁽⁰⁾ misses dominate DRAM on Cora/CiteSeer/Sage-Cora** (40–63 %): W̄⁽⁰⁾ is 90–231 KB
  against the 48 KB Weight Buffer (Q-35); the direct-mapped row cache misses on 22–46 % of
  accesses.
- **Hidden-layer X̄ never reaches DRAM**: the Q-28 fusion keeps it in the Input Buffer (largest
  subgraph 16.7 KB of 32 KB). So the Adaptive-Package compression of hidden layers saves
  on-chip space, not DRAM traffic, in this schedule; DRAM sees X⁽⁰⁾, W̄, and 4-bit B̄ rows.
- NELL has 383 subgraphs, not 257: its 186-class output layer sets the capacity to 176 (S-08).
