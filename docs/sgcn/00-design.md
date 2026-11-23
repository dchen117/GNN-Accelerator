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
  (≥ 16 columns, one line) whose F0 × w tile of W⁽⁰⁾ fits half the cache. X⁽⁰⁾ is re-streamed
  once per tile. If no tile fits, there is no tiling, because a non-resident tile only
  multiplies the X⁽⁰⁾ re-reads. Result: Cora 32, CiteSeer 128 (none), PubMed 64, NELL 128
  (none).
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
- DRAM models:
  - The evaluation uses Ramulator (`megasim.dram.ramulator.fixed_point`, generalized from
    MEGA's), **damped**: latencies = 0.5 · measured + 0.5 · previous, up to 10 iterations,
    stop at < 1 % change.
    - Undamped, SGCN oscillates (PubMed: 2.7 M ↔ 88 M cycles). Its aggregation is DRAM-bound,
      so a schedule that issues too fast measures long queueing latencies, and those make the
      next schedule issue far too slowly.
    - Damped, it settles into a band of about ±3 % (CiteSeer: 418 k – 453 k cycles over
      iterations 5–12). `ramulator.spread_last4` in the JSON records the band. We report the
      last iterate.
    - MEGA's undamped iteration converges (its DRAM is mostly idle), so MEGA's numbers are
      unchanged.
  - The starting point and `--dram uncontended` use a latency + transfer-time model without
    queueing. megasim's FCFS server is not used here: SGCN's call order (reads of an item,
    then its posted write) is not time order, and a write booked at its late ready time would
    block earlier reads. That is an artefact, not DRAM behaviour.

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
