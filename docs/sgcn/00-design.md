# SGCN simulator (sgcnsim) — design and decisions

SGCN: Yoo et al., "SGCN: Exploiting Compressed-Sparse Features in Deep Graph Convolutional
Network Accelerators", HPCA 2023 (arXiv 2301.10388). We simulate it as MEGA's comparison
point in MEGA's Tab.V configuration, so the two can be compared the way MEGA's Fig.14/16/17
compare them.

Policy (from the user, 2026-11-23): open questions of the SGCN paper are ours to decide,
favouring **simplicity and adherence** to SGCN; hardware sizes come from MEGA Tab.V. Each
decision below is tagged:
- [SGCN] the SGCN paper states it;
- [MEGA] MEGA states it for its SGCN comparison;
- [OURS] our choice, with the reason.

Code: `sgcnsim/` (`workload.py`, `timing.py`, `cache.py` + `cache.cpp`, `energy.py`,
`cacti.py`, `sim.py`, `compare.py`). It reuses megasim's DRAM models and Ramulator fixed point
(`megasim/dram/ramulator.py`), energy constants and CACTI wrapper, so both designs see the
same DRAM, the same SRAM tool and the same pJ/bit.

## G-01 Configuration

| | Value | Source |
|---|---|---|
| Aggregation | 4 engines × SIMD16 (32-bit), one 64 B line per cycle each | [MEGA] Tab.V "4 SIMD16"; [SGCN] Fig.8 (16 multipliers, one cache line per step) |
| Combination | 16 32-bit MACs | [MEGA] Tab.V |
| On-chip memory | one global cache, 392 KB, 16-way, LRU, 64 B lines | [MEGA] §VI-A-3 "same on-chip buffer capacity (392KB)"; [SGCN] Tab.III 16-way LRU |
| Precision | 32-bit features and weights | [MEGA] Tab.V; [SGCN] §VI-A |
| DRAM | HBM1, 256 GB/s, Ramulator 2.1, same config and address mapping as MEGA (S-13) | [MEGA] §VI-A-3 |
| Clock | 1 GHz | both |
| Slice C | 96 features | [SGCN] §V-B default |
| SAC strip | 32 vertices | [SGCN] §V-C |

## G-02 Workload: the FP32 model of the same run [OURS]

SGCN computes in 32-bit, so it runs the FP32 GCN. Each `runs/hw/gcn_*` run was fine-tuned from
an FP32 pretraining cached in `runs/fp32_cache_hw`. `workload.py` loads it, runs the forward
pass in eval mode and takes X⁽¹⁾'s zero pattern from the real ReLU output. The graph (A+I,
GCN normalization) and X⁽⁰⁾ are the same as MEGA's.

| | Cora | CiteSeer | PubMed | NELL |
|---|---|---|---|---|
| X⁽⁰⁾ density | 1.27% | 0.85% | 10.0% | 0.01% |
| X⁽¹⁾ density (FP32 ReLU) | 88.6% | 94.2% | 85.5% | 51.6% |

- This matches SGCN's own observation (Fig.1–2): shallow GCNs without residual connections
  have only 5–30% intermediate sparsity. MEGA's 2-layer, 128-hidden models are of that kind.
- MEGA's quantized X̄⁽¹⁾ is much sparser (MEGA Fig.5), but that sparsity comes from MEGA's own
  quantization and is not available to a 32-bit design.
- The NELL FP32 checkpoint is a weak pretraining (28.0% test accuracy; QAT recovered it for
  MEGA). Only its zero pattern matters here, but X⁽¹⁾'s 52% density may differ from a
  well-trained FP32 NELL model.
- MEGA has no residual connections, so SGCN's residual input S is absent.

## G-03 Layer dataflow

- **Layer 0: combination first, on the sparse aggregation engines** [SGCN] §V-F ("Optionally,
  when the input feature is extremely sparse, the combination of the first layer is performed
  on the sparse aggregator engine"), §VII-B (CSR for such inputs).
  - [OURS] We always use this option. Aggregation first would make the 16-MAC combination
    engine multiply dense Ã·X⁽⁰⁾ rows (F0 = 500 … 61,278 wide) by W⁽⁰⁾. On Cora that is
    2708·1433·128/16 = 31 M cycles, against about 0.1 M this way. Every input in MEGA's
    datasets is ≥ 90% sparse.
  - X⁽⁰⁾ is stored in CSR (row pointer + 4 B column + 4 B value per non-zero) [SGCN] §VII-B,
    streamed sequentially.
  - Each non-zero x[v,f] reads row f of W⁽⁰⁾ through the cache and adds x·W[f,:] into T[v,:].
  - T = X⁽⁰⁾W⁽⁰⁾ is written to DRAM dense (it is pre-activation, nothing to compress).
- **Then aggregation of T**: Ã·T on the aggregation engines over dense slices. The ReLU +
  compressor [SGCN] §V-E writes X⁽¹⁾ in sliced BEICSR.
- **Layers l ≥ 1: aggregation first** [SGCN] Tab.I. Ã·X⁽ˡ⁾ over BEICSR slices, then the
  combination engine (16 MACs), then ReLU + compressor. The last layer writes dense logits.
- Phases are separated by a barrier, since a layer needs all of its input written.

## G-04 Tiling [OURS, within SGCN's description]

- **W⁽⁰⁾ column tiles (phase A).** SGCN tiles "both graph topology and feature" with sizes
  from offline analysis (§V-F, citing GCNAX). We pick the widest power-of-two column tile
  (≥ 16 columns, one line) whose F0 × w tile of W⁽⁰⁾ fits ¾ of the cache. W⁽⁰⁾ is the only
  cached data in phase A, so the remaining ¼ is margin for set conflicts. X⁽⁰⁾ is re-streamed
  once per tile. If no tile fits, there is no tiling, because a non-resident tile only
  multiplies the X⁽⁰⁾ re-reads. Result: Cora 32, CiteSeer 16, PubMed 128 (none), NELL 128
  (none).
  - A first version used ½ of the cache. CiteSeer then ran untiled, with its 1.9 MB W⁽⁰⁾
    missing 22.5 MB, against 8.5 MB total with 16-column tiles.
- **Topology: one tile, the whole graph.** SAC strips are interleaved over it.
  - With 392 KB, a topology tiling that keeps a column tile's features *and* a row tile's
    partial sums on chip re-reads the features once per row tile.
  - Cora estimate: about 7 × 7 tiles, so 7 × 1.39 MB of T reads, against 4.1 MB of misses
    untiled.
  - A GCNAX-style offline analysis would therefore choose one tile for these graphs.
  - [MEGA] Tab.V also lists SGCN with "Graph Partition: No".
- **Feature slicing (C = 96)** [SGCN] §V-B:
  - Each slice is stored in place: bitmap + values, with its dense size reserved and aligned
    to a 64 B line.
  - A read or write touches ⌈(bitmap + 4·nnz)/64⌉ lines.
  - Inside a strip, the engine walks slice by slice, so one slice of the neighbours is live
    at a time.

## G-05 Cache and traffic [OURS where marked]

- Cached: everything accessed at random, i.e. W⁽⁰⁾ rows (phase A), T rows, and BEICSR slices.
- [OURS] Streamed past the cache, never cached:
  - sequential streams (CSR topology: row pointer + column index + Ã value per edge; CSR
    X⁽⁰⁾; the systolic weights W⁽ˡ⁾, loaded once per layer);
  - writes (T, X⁽ˡ⁺¹⁾, logits; write-no-allocate).
- [OURS] One LRU simulation runs over every phase's cached reads, in the fixed execution order:
  strip k = 0, 1, … (engine k mod 4), slice, then edge. Hits and misses are thus independent
  of timing, and so is the DRAM call order the Ramulator fixed point needs (S-13). The
  ordering detail (4 engines interleaved at strip granularity) is a simplification.

## G-06 Timing [OURS]

- Engine e runs items e, e+4, … (an item is a strip; in phase A, a strip × W⁽⁰⁾ tile).
- Compute: one 64 B line per engine per cycle, hit or miss, so a strip's cycles equal its
  lines. Phase A uses the same rule: one W⁽⁰⁾ row segment line × the non-zero per cycle.
- Prefetch: an item's reads are issued when the engine starts its previous item, i.e. one item
  ahead [SGCN] §III-B (readers with prefetch buffers).
- The item starts at max(engine free, first data) and ends at max(start + cycles, last data + 1).
- The combination engine takes items in order: start = max(free, aggregation done, weights
  loaded), cycles = rows·F_in·F_out/16.
- Writes are posted when the output is ready. A phase ends when all items and writes are done.
- **DRAM model: Ramulator-calibrated, time-ordered bandwidth server** (`timing.BackfillDram`,
  `sim.simulate_ramulator`).
  - Calibration: each phase's transfers are replayed through Ramulator, all issued at cycle 0
    in schedule order. Lines × 64 / cycles gives the phase's sustained HBM bandwidth (it
    includes the phase's row-buffer locality). One isolated line gives the unloaded latency
    (31 cycles).
  - The server books each transfer at the earliest free bandwidth at or after its issue time.
    It fills gaps between earlier bookings, so a read issued early is not blocked by a write
    that was called first but issued later.
  - Check: the final schedule's trace is replayed through Ramulator at its real issue times.
    Ramulator finishes the traffic when the schedule ends (`replay_end_over_total` ≈ 1.00).
  - **Why not MEGA's fixed point (S-13)?** We tried it first (`megasim.dram.ramulator.fixed_point`,
    generalized for any timing model):
    - Undamped, SGCN oscillates (PubMed: 2.7 M ↔ 88 M cycles). Its aggregation is DRAM-bound,
      so the measured per-transfer latencies are mostly queueing. A schedule that issues too
      fast measures long queues, which makes the next schedule issue far too slowly.
    - Damped (`relax = 0.5`), it settles on Cora (290 k cycles; the server above gives 291 k)
      and in a ±4 % band on CiteSeer (418 k – 453 k; the server gives 421 k).
    - PubMed still swings by ±18 %, and NELL starts at 415 M cycles.
    - MEGA keeps the fixed point: its DRAM is mostly idle (stall ≤ 1.2 %) and it converges
      undamped. Both couplings take their timing from the same Ramulator HBM1 configuration
      and address mapping.
  - `--dram uncontended` (debugging only): latency + transfer time, no queueing.
    - megasim's FCFS server is not used anywhere for SGCN. It serves in call order, and
      SGCN's call order (an item's reads, then its posted write) is not time order, so a
      write booked at its late ready time would block reads issued earlier.

## G-07 Energy (same categories and constants as MEGA, S-14)

- **DRAM:** 7 pJ/bit.
- **SRAM:** cache reads (one per consumed line) and fills (one per miss) × CACTI 7.0 at
  28 nm, itrs-lop, 392 KB 16-way 64 B, 2 banks (`sgcnsim/cacti_28nm.json`): read 120.6 pJ,
  write 113.1 pJ.
  - 2 banks gives an area of 1.44 mm², closest to MEGA's measured 1.67 mm² for the same
    392 KB.
  - The read energy is higher than MEGA's scratchpads (21–43 pJ) because of the 16-way
    parallel tag/data lookup, CACTI's default access mode.
- **PU:** 1.99 pJ per 32-bit MAC: Horowitz ISSCC'14 45 nm mult 3.1 + add 0.1 pJ, × 28/45.
  - Aggregation engines are charged 16 MACs per consumed line, padding lanes included. The
    combination engine is charged its useful MACs.
  - SGCN's own power figures are for 8 + 8 engines at 45→32 nm and do not transfer to Tab.V's
    configuration.
- **Leakage:** cache leakage from CACTI (108 mW) × run time.

## G-08 Residual connections [SGCN] Eq.(2), §V-F; [OURS] details

`timing.build_plan(w, hw, residual=True)`, CLI `python -m sgcnsim.sim ... --residual`.
MEGA's 2-layer GCNs have none, so it is off in our MEGA comparison.

- [SGCN] X⁽ˡ⁺¹⁾ = ReLU(Ã·X⁽ˡ⁾·W⁽ˡ⁾ + S⁽ˡ⁾): "the registers at the systolic arrays are initialized
  with S instead of zero", then ReLU and the compressor.
- [OURS] S⁽ˡ⁾ = X⁽ˡ⁾ (identity residual, as in the ResGCN-style models SGCN evaluates). It applies
  to aggregation-first layers (phase C) with F_in = F_out.
  - Layer 0 and the output layer have F_in ≠ F_out, and a projection residual is not described.
- [OURS] Cost: each strip's own X⁽ˡ⁾ rows (all slices, BEICSR) are read through the cache with
  the strip's other reads. There are no extra MACs (register initialization) and no extra
  aggregation-engine cycles: the rows feed the systolic array, not the SIMD engines.
  - The rows were usually just read through the strip's self-loop edges, so they mostly hit.
    The test sees under 10 % extra misses.
- The residual's real effect on SGCN is the sparser X⁽ˡ⁾ it produces (SGCN Fig.2). That comes
  from the workload's hidden features. `load_workload` builds MEGA's 2-layer GCNs only, so a
  residual model needs its own workload (hidden features from a model trained with the
  connections).

## Targets (MEGA p.10–11, GCN only)

The SGCN series was read from 400-dpi renders of Fig.14/16/17 and identified by recomputing
the stated geomeans:
- Fig.14's legend order matches its bars.
- In Fig.16/17 the bar order differs from the legend; the geomean check settles which bar is
  SGCN.

| MEGA ÷ SGCN | Cora | CiteSeer | PubMed | NELL | Reddit | geomean (stated) |
|---|---|---|---|---|---|---|
| speedup (Fig.14: 43.0/15.4, 58.7/25.9, 29.5/12.9, 52.7/20.0, 78.5/5.1) | 2.79 | 2.27 | 2.29 | 2.64 | 15.39 | 3.58 (3.6) |
| DRAM reduction (Fig.16: SGCN 34.5, 25.3, 38.4, 10.2, 16.8) | 2.84 | 7.43 | 6.63 | 23.48 | 6.48 | 7.34 (7.3) |
| energy saving (Fig.17: SGCN 16.2, 24.5, 15.6, 13.3, 17.9) | 2.88 | 2.36 | 7.31 | 6.47 | 5.77 | 4.51 (4.5) |

- [DERIVED] The stated 3.6× / 7.3× / 4.5× are geomeans over the five GCN datasets. The
  "Geomean" bars in Fig.14/16/17 (SGCN 13.9 / 22.5 / 17.1) are over those five too.
- We have no GCN-Reddit run (Reddit was trained for GraphSage only), so we compare on 4
  datasets.

## Results (2026-11-23, after megasim S-15)

Commands:

```bash
python -m sgcnsim.sim runs/hw/gcn_{cora,citeseer,pubmed,nell} --json runs/hw/sgcn.json
```

```bash
python -m sgcnsim.compare runs/hw/megasim_m3.json runs/hw/sgcn.json
```

MEGA uses S-15's X̄⁽⁰⁾ position format (`docs/simulator/00-design.md`). Paper values (MEGA ÷
SGCN) are in parentheses.

| dataset | MEGA µs | SGCN µs | speedup | MEGA MB | SGCN MB | DRAM reduction | MEGA µJ | SGCN µJ | energy saving |
|---|---|---|---|---|---|---|---|---|---|
| Cora | 58.8 | 291.0 | 4.95× (2.79×) | 1.33 | 14.09 | 10.56× (2.84×) | 97.8 | 935.1 | 9.56× (2.88×) |
| CiteSeer | 91.0 | 418.0 | 4.60× (2.27×) | 3.76 | 22.35 | 5.94× (7.43×) | 249.0 | 1483.1 | 5.96× (2.36×) |
| PubMed | 441.9 | 2998.1 | 6.79× (2.29×) | 7.20 | 132.20 | 18.35× (6.63×) | 573.5 | 9397.1 | 16.39× (7.31×) |
| NELL | 16,687 | 99,784 | 5.98× (2.64×) | 76.19 | 378.25 | 4.96× (23.48×) | 8,164 | 36,742 | 4.50× (6.47×) |

Geomeans:

| | speedup | DRAM reduction | energy saving |
|---|---|---|---|
| Ours (4 datasets) | 5.51× | 8.70× | 8.05× |
| Paper, the same 4 datasets | 2.48× | 7.57× | 4.24× |
| Paper, all 5 (stated) | 3.6× | 7.3× | 4.5× |
| Ours with MEGA's original bitmap accounting | 5.51× | 4.63× | 4.94× |

**MEGA outperforms SGCN on every dataset in time, DRAM and energy**, as in the paper. The DRAM
geomean is close to the paper's. Speedup is about 2.2× larger than the paper's, and energy
about 1.9× larger.

SGCN breakdown:

| | Cora | CiteSeer | PubMed | NELL |
|---|---|---|---|---|
| phase A / B / C (k cycles) | 99 / 39 / 153 | 212 / 45 / 161 | 1986 / 513 / 500 | 1149 / 791 / 97,845 |
| HBM efficiency per phase (Ramulator) | 0.72 / 0.58 / 0.53 | 0.80 / 0.57 / 0.53 | 0.68 / 0.46 / 0.42 | 0.55 / 0.64 / 0.62 |
| cache hit rate | 0.76 | 0.83 | 0.84 | 0.47 |
| DRAM stall share | 4.0 % | 4.6 % | 10.6 % | 0.4 % |

On every dataset, Ramulator's replay of the schedule's own trace ends when the schedule does
(1.00).

### Why our MEGA ÷ SGCN differs from the paper

**Speed: the gap is on the MEGA side, and we cannot close it without leaving Tab.V or the
accepted MEGA micro-architecture.**
- SGCN's time has a floor that our modelling choices do not set. Tab.V's compute fixes two
  terms:
  - X⁽⁰⁾·W⁽⁰⁾ needs nnz(X⁽⁰⁾)·F1 MACs on the 4 × SIMD16 engines. That is already the cheapest
    option (G-03): aggregation first would put dense Ã·X⁽⁰⁾ on the 16 MACs.
  - The output layer needs N·F1·C MACs on the 16-MAC combination engine.
- Even with SGCN's phases overlapping perfectly, the busiest unit bounds its time. Dividing
  by the paper's speedup bounds the paper's MEGA from below:

  | | Cora | CiteSeer | PubMed | NELL |
  |---|---|---|---|---|
  | SGCN busiest unit (k cycles): SIMD engines / 16 MACs | 153 / 152 | 263 / 160 | 2417 / 473 | 1932 / 97,843 |
  | paper's MEGA must take at least (µs) | 55 | 116 | 1,056 | 37,062 |
  | our MEGA (µs) | 58.8 | 91.0 | 441.9 | 16,687 |

- So on PubMed and NELL the paper's MEGA is at least 2.2–2.4× slower than ours, whatever the
  SGCN model.
  - Our MEGA is combination-bound (91–97 % BSE utilization), so the difference lies in how
    fast the Combination Tiles are taken to be (Q-21/Q-23/Q-24) or in the per-node
    bitwidths.
  - Bitwidths explain at most part of it. The hidden-layer bits per element (M1 table in
    `megasim/README.md`, against Tab.VI's average bits) are higher than Tab.VI on Cora
    (2.53 vs 1.70) and CiteSeer (2.25 vs 1.87), and lower on PubMed (1.73 vs 2.50).
    - Fewer bits could account for up to ~1.4× on PubMed, not 2.4×.
    - NELL has no Tab.VI entry.
  - We leave the accepted tile model as it is. A slower interpretation would only be a
    guess fitted to the ratio.
- With our SGCN at its own floor, our speedups sit 1.8–3.0× above the paper's.

**DRAM: close on average, with per-dataset scatter we cannot attribute.**
- Cora (10.6× vs 2.84×) and PubMed (18.4× vs 6.6×): SGCN reads a neighbour's row per edge.
  - It has a 392 KB cache, no reordering (Tab.V "Graph Partition: No") and 86–94 %-dense FP32
    hidden features.
  - So about E × 512 B per layer is inherent: 55 MB per layer on PubMed.
  - The paper's ratio would need either a 2–3× lighter SGCN or a 2–3× heavier MEGA there.
    Nothing in either paper says which.
- NELL (5.0× vs 23.5×): our SGCN's NELL traffic is W⁽⁰⁾ rows (106 MB), dense T (129 MB) and
  logits (49 MB). MEGA's is logits (23 MB, 16-bit × 186 classes) and Condense regions (27 MB).
- Fig.19 cannot calibrate MEGA's DRAM: no single accounting reproduces it (S-15).

**Energy** follows DRAM: DRAM is 58–84 % of SGCN's energy and 55–89 % of MEGA's. SGCN also pays
120 pJ per cache read (16-way lookup, G-07).

### What changed on 2026-11-23

- **NELL used to be the exception** (MEGA 0.68× SGCN's DRAM, energy tied). That came from
  480 MB of dense 61,278-bit input bitmaps in MEGA's DRAM accounting.
  - The paper's own numbers rule this out (S-15). MEGA now reads NELL's positions as 16-bit
    indices, as SGCN does with CSR.
  - Our earlier note bounded the paper's MEGA at "≈ 67 MB" from HyGCN's 16 GB input. That
    inverted a lower bound and is corrected in S-15.
- Reddit is still missing (no GCN-Reddit run).

### GCN on Reddit: feasibility (checked 2026-11-23)

Reddit is the paper's largest MEGA-vs-SGCN gap (15.4× speedup), so it matters, but it does not
fit this container (15 GB RAM, 4 CPUs).

Measured:
- 232,965 nodes, 114.6 M edges, degree up to 21,658 (mean 493).
- Loading the graph and building GCN's normalized adjacency already peaks at **12.3 GB RSS**.

What each step needs:
1. **Train GCN-Reddit (FP32 + MEGA QAT).**
   - Full-graph GCN (Tab.III: no sampling). Inductive training on the train subgraph (~50 M
     edges) helps, but evaluation, and the hardware, need the full graph.
   - Estimate: 1–3 h per seed and well over 15 GB.
   - Fidelity check to run: hub nodes (21 k in-edges) against the 16-bit Aggregation Unit
     partial sums (Q-25). megasim reports any overflow, but a model that saturates would be
     a finding in itself.
2. **megasim.** The code works unchanged in principle.
   - Four int64 edge arrays are 3.7 GB, and `np.unique` over (subgraph, source) pairs roughly
     doubles that at its peak.
   - The Condense traffic loop is pure Python over the external pairs (tens of millions):
     minutes.
   - Ramulator replays roughly 100 M lines per fixed-point iteration.
   - Feasible on a ≥ 64 GB machine. Here it would need int32 indices and chunking.
3. **sgcnsim needs a streaming mode.**
   - SGCN reads about 8 lines per edge per layer, so the plan materializes ~1.9 G cached line
     addresses: 15 GB as int64 before any miss list.
   - Needed:
     - generate and simulate the cache strip-chunk by strip-chunk, keeping only per-item
       miss counts;
     - calibrate each phase's bandwidth (`BackfillDram`) from a sampled prefix replay
       instead of the full trace;
     - skip the full-trace Ramulator check, or run it on a sample.
   - About a day of work. The results should stay identical on the small datasets, which
     gives a regression check.

Recommendation: run Reddit on a larger cloud machine (≥ 64 GB RAM, ≥ 16 cores). Do the
sgcnsim streaming mode first; it can be developed and verified here on the small datasets.
