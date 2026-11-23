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

## Results (2026-11-23)

Commands:

```bash
python -m sgcnsim.sim runs/hw/gcn_{cora,citeseer,pubmed,nell} --json runs/hw/sgcn.json
```

```bash
python -m sgcnsim.compare runs/hw/megasim_m3.json runs/hw/sgcn.json
```

Paper values (MEGA ÷ SGCN) are in parentheses.

| dataset | MEGA µs | SGCN µs | speedup | MEGA MB | SGCN MB | DRAM reduction | MEGA µJ | SGCN µJ | energy saving |
|---|---|---|---|---|---|---|---|---|---|
| Cora | 58.8 | 291.0 | 4.95× (2.79×) | 1.70 | 14.09 | 8.29× (2.84×) | 119.2 | 935.1 | 7.84× (2.88×) |
| CiteSeer | 91.0 | 418.0 | 4.60× (2.27×) | 5.03 | 22.35 | 4.45× (7.43×) | 323.3 | 1483.1 | 4.59× (2.36×) |
| PubMed | 441.9 | 2998.1 | 6.79× (2.29×) | 7.20 | 132.20 | 18.35× (6.63×) | 573.5 | 9397.1 | 16.39× (7.31×) |
| NELL | 16,687 | 99,784 | 5.98× (2.64×) | 556.28 | 378.25 | **0.68×** (23.48×) | 36,331 | 36,742 | **1.01×** (6.47×) |

Geomeans:

| | speedup | DRAM reduction | energy saving |
|---|---|---|---|
| Ours (4 datasets) | 5.51× | 4.63× | 4.94× |
| Paper, the same 4 datasets | 2.48× | 7.57× | 4.24× |
| Paper, all 5 (stated) | 3.6× | 7.3× | 4.5× |

SGCN breakdown:

| | Cora | CiteSeer | PubMed | NELL |
|---|---|---|---|---|
| phase A / B / C (k cycles) | 99 / 39 / 153 | ≈ 213 / 45 / 160 | ≈ 2000 / 500 / 500 | 1149 / 791 / 97,845 |
| HBM efficiency per phase (Ramulator) | 0.72 / 0.58 / 0.53 | | | |
| cache hit rate | 0.76 | 0.83 | 0.84 | 0.47 |
| DRAM stall share | 4.0 % | 4.6 % | 10.6 % | 0.4 % |

Across all datasets, the per-phase HBM efficiency is 0.42–0.74, and Ramulator's replay ends
when the schedule does (1.00).

Findings:

- **Speed: MEGA is faster on every dataset, by 4.6–6.8×.** The paper gives 2.3–2.8×, so we
  overstate the gap about 2×. Most of SGCN's time is in two places:
  - Layer 1's combination on Tab.V's 16 MACs: rows × 128 × classes / 16, which is 52 % of
    Cora's time and 98 % of NELL's (186 classes).
  - Layer 0's X⁽⁰⁾·W⁽⁰⁾ on the 4 SIMD16 engines (G-03).
  - Neither depends on our DRAM or cache choices. Only a larger combination engine than
    Tab.V's "16 MACs for combination" would shrink them.
- **DRAM: MEGA moves 4.5–18× less on Cora, CiteSeer and PubMed** (paper 2.8–7.4×). SGCN's
  aggregation misses the cache on 70–90 % of neighbour reads. The graphs are not reordered
  (Tab.V "Graph Partition: No"), and FP32 hidden features are 86–94 % dense (G-02), so
  BEICSR saves little.
- **NELL is the exception: SGCN moves 0.68× MEGA's DRAM, and their energy ties.** This is a
  MEGA-side modelling issue, not an SGCN one.
  - Our MEGA reads 480 MB of dense 61,278-bit per-node input bitmaps (86 % of its NELL
    traffic). SGCN reads X⁽⁰⁾ as CSR (3.7 MB).
  - The paper's MEGA cannot be doing that. HyGCN reading NELL's X⁽⁰⁾ once as dense FP32 is
    already 16 GB, and Fig.16's 239.5× reduction leaves MEGA ≈ 67 MB in total, less than
    those bitmaps alone.
  - This belongs with the open MEGA DRAM-accounting question (hidden-layer fusion, Fig.19),
    which is on hold.
- **Energy follows DRAM** (DRAM is 58–89 % of SGCN's energy). NELL also pays 29 % leakage
  over its 100 ms run.
- Reddit is missing (no GCN-Reddit run). In the paper it is the largest gap (15.4× speedup).
