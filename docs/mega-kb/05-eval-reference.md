# 05 — Evaluation Reference (targets for validating our baseline)

Tags: [STATED] / [DERIVED] / [INTERP] / [UNSPECIFIED]. As agreed, other accelerators are not
described here. Their names appear only where MEGA's numbers are normalized to them.

Figure values marked "(hi-res)" were read from 250–300 dpi renders. Fig.14/16/17 transcriptions
were checked by recomputing the geomeans (see §5).

## 1. Datasets — Tab.II (p.9) [STATED]

| Dataset | #Node | #Edge | Feature length | Average degree |
|---|---|---|---|---|
| Cora | 2,708 | 10,556 | 1,433 | 3.90 |
| CiteSeer | 3,327 | 9,104 | 3,703 | 2.74 |
| PubMed | 19,717 | 88,648 | 500 | 4.50 |
| NELL | 65,755 | 251,550 | 61,278 | 3.83 |
| Reddit | 232,965 | 114,615,892 | 602 | 491.99 |

- [DERIVED] Average degree = #Edge / #Node (Cora 10556/2708 = 3.90 ✓; Reddit 114615892/232965 =
  491.99 ✓). So #Edge counts directed edges (each undirected edge twice), with no self-loops
  included. [INTERP] This hints at how "degree" is counted but does not settle Q-01.
- [STATED] Sources: citation datasets [58], NELL [5], Reddit [20] (p.9 §VI-A-1). Splits are not
  given (Q-14).

## 2. Models — Tab.III (p.9) [STATED]

| Model | Sample nodes | Layers | Hidden units | Aggregation |
|---|---|---|---|---|
| GCN | — | 2 | 128 | Add |
| GIN | — | 2 | 128 | Add |
| GraphSage | 25 | 2 | 256 | Mean |

## 3. Algorithm-level targets

### 3.1 Tab.VI (p.10): accuracy (%) / average bits / CR [STATED]

| Dataset | Model | FP32 | DQ-INT4 (4 bit, 8×) | **Degree-Aware (ours)** | Avg bits | CR |
|---|---|---|---|---|---|---|
| Cora | GCN | 81.5±0.7 | 78.3±1.7 | **80.9±0.6** | 1.70 | 18.6× |
| Cora | GIN | 77.6±1.1 | 69.9±3.4 | **77.8±1.6** | 2.37 | 13.4× |
| Cora | GraphSage | 79.7±0.5 | — | **79.7±1.0** | 3.40 | 9.4× |
| CiteSeer | GCN | 71.1±0.7 | 66.9±2.4 | **70.6±1.1** | 1.87 | 17.0× |
| CiteSeer | GIN | 66.1±0.9 | 60.8±1.2 | **65.1±1.7** | 2.54 | 12.6× |
| PubMed | GCN | 78.9±0.7 | 62.5±2.4 | **78.3±0.6** | 2.50 | 12.8× |
| Reddit | GraphSage | 95.2±0.1 | — | **95.6±0.2** | 2.74 | 11.7× |

- [DERIVED] CR ≈ 32/avg-bits with small mismatches (Q-15).
- [STATED] Other λ trade-off point: CiteSeer GCN 71.2% at 2.24 bits (p.10 §VI-B).
- [STATED] DQ-INT8 on CiteSeer GCN: 71.0% at 8 bits, vs Degree-Aware 70.6% at 1.87 bits (p.10 §VI-B).
- [UNSPECIFIED] No accuracy/bits for NELL (any model), GCN-Reddit, GIN-PubMed, GIN-NELL/Reddit,
  GraphSage-CiteSeer/PubMed/NELL (Q-31).
- **Suggested acceptance check for the PyTorch model** [INTERP — our proposal, not the
  paper's]: for each Tab.VI row, match accuracy within the reported std at ≤ the reported
  average bits.

### 3.2 Tab.I (p.3): DQ on GIN / CiteSeer vs bitwidth [STATED]

| | FP32 | 8bit | 7bit | 6bit | 5bit | 4bit |
|---|---|---|---|---|---|---|
| Acc | 66.1±0.9 | 67.5±1.4 | 65.0±2.4 | 63.8±3.3 | 63.7±3.4 | 60.8±2.1 |
| CR | 1× | 4× | 4.6× | 5.3× | 6.4× | 8× |

### 3.3 Fig.5 (p.4): node-feature density [STATED, read from figure]

| | Cora | CiteSeer | PubMed | NELL | Reddit |
|---|---|---|---|---|---|
| GCN | 44% | 55% | 41% | 12% | 54% |
| GIN | 63% | 79% | 84% | 33% | 19% |
| GraphSage | 79% | 88% | 71% | 56% | 51% |

[UNSPECIFIED] which layer's X, and whether this is before or after quantization.

### 3.4 Other algorithm-level facts
- [STATED] GAT (not in the baseline): CR 16.5× (from 9.6×); CiteSeer FP32 72.5%, Degree-Aware
  71.9% (p.13 §VII-3).
- [STATED] Training overhead: 2.04× FP32 time, memory up to +12.8%, one RTX 3090. DQ = 1.58× the
  Degree-Aware training time (p.12–13 §VII-1).

## 4. Hardware configuration

See `03-hardware-mapping.md` §7 for Tab.IV. Summary: 4×8×32 BSEs, 256 AUs, 392 KB SRAM, 1 GHz,
28nm, 1.869 mm², 194.98 mW, HBM1.0 256 GB/s.

## 5. Hardware-level results (MEGA values; all normalized to HyGCN = 1.0)

### 5.1 Speedup — Fig.14 (p.11, hi-res) [STATED]

| Model | Cora | CiteSeer | PubMed | NELL | Reddit |
|---|---|---|---|---|---|
| GCN | 43.0 | 58.7 | 29.5 | 52.7 | 78.5 |
| GIN | 32.4 | 42.1 | 19.2 | — | — |
| GraphSage | 50.7 | — | — | — | 16.4 |

Geomean **38.3×**. [DERIVED] Geomean of the 10 values = 38.25 ✓.

### 5.2 DRAM access reduction — Fig.16 (p.11, hi-res) [STATED]

| Model | Cora | CiteSeer | PubMed | NELL | Reddit |
|---|---|---|---|---|---|
| GCN | 98.0 | 188.0 | 254.6 | 239.5 | 108.8 |
| GIN | 144.0 | 74.9 | 236.2 | — | — |
| GraphSage | 20.8 | — | — | — | 33.6 |

Geomean **108.1×**. [DERIVED] recomputed 108.09 ✓.

### 5.3 Energy savings — Fig.17 (p.11, hi-res) [STATED]

| Model | Cora | CiteSeer | PubMed | NELL | Reddit |
|---|---|---|---|---|---|
| GCN | 46.6 | 57.9 | 114.1 | 86.1 | 103.2 |
| GIN | 26.7 | 26.6 | 37.9 | — | — |
| GraphSage | 30.7 | — | — | — | 26.7 |

Geomean **47.6×**. [DERIVED] recomputed 47.65 ✓.

### 5.4 Energy breakdown, GCN — Fig.18 (p.11, hi-res) [STATED]
HyGCN energy per component, normalized to MEGA = 1.0:

| | DRAM | SRAM | PU | Leakage |
|---|---|---|---|---|
| Cora | 98.0 | 2.7 | 90.8 | 108.5 |
| CiteSeer | 188.0 | 2.8 | 118.6 | 144.2 |
| PubMed | 254.6 | 13.3 | 59.1 | 74.7 |
| NELL | 239.5 | 2.2 | 117.0 | 134.0 |
| Reddit | 108.8 | 8.8 | 105.8 | 125.8 |

[DERIVED] The DRAM column equals the GCN DRAM-access reductions in Fig.16 exactly, which implies
DRAM energy scales linearly with access count in their model.

### 5.5 Headline ratios vs other accelerators (p.1, p.10–11) [STATED]
- Speedup 38.3× / 7.1× / 4.0× / 3.6×; energy 47.6× / 7.2× / 5.4× / 4.5×; DRAM 108.1× / 10.5× /
  8.4× / 7.3× (vs HyGCN / GCNAX / GROW / SGCN).
- 2.8× over GCNAX run with DQ-INT8 (p.10 §VI-C-1).
- Against two accelerators in their original (larger) configurations: 4.68× and 2.53× on GCN
  (p.11, Fig.15).

## 6. Ablation and sensitivity

### 6.1 Fig.19 (p.12) [STATED] — GCN, averaged
| Step | Speedup (cumulative) | DRAM access (relative) |
|---|---|---|
| HyGCN-C (A(XW) order, no quantization) | 1.0 | 64.2 |
| + Degree-Aware quantization, Bitmap storage | 4.8 (4.8×) | 11.0 (5.8× less) |
| + Adaptive-Package | 22.4 (4.7×) | 4.4 (2.5× less) |
| + Condense-Edge (= MEGA) | 24.8 (1.1×) | 1.0 (4.4× less) |

- [STATED] Condense-Edge's speedup is small "because most cycles of the DRAM accesses can be
  overlapped". Its DRAM benefit is larger on sparser graphs, e.g. NELL (0.0073% density,
  11.6×) (p.12 §VI-D-2).
- [DERIVED] NELL density check: 251,550 / 65,755² = 5.8e-5 = 0.0058%, not 0.0073%.
  [UNSPECIFIED] how the paper's density was computed (e.g. with self-loops:
  (251550+65755)/65755² = 0.0073% ✓). [INTERP] So the paper's figure likely includes self-loops.
- [STATED] Fig.20(a): MEGA has the lowest DRAM-induced pipeline stall share. Fig.20(b):
  Condense-Edge gives lower DRAM access than Naive / METIS / GCoD on Cora, CiteSeer, PubMed
  (p.12). Exact values are not transcribed.

### 6.2 Fig.21 (p.12) [STATED] — package-length settings
Settings explored (short, medium, long) in bits: (16,24,32), (64,128,192), (160,192,296),
(192,296,400), (400,512,800). Normalized DRAM access is within 1.00–1.25 of each dataset's
optimum. Chosen: (64,128,192). PubMed's optimum is (400,512,800).

### 6.3 Fig.22 (p.12) [STATED] — speedup vs compression ratio (Cora, normalized to HyGCN)
| CR | 5.9 | 7.4 | 10.1 | 12.8 | 18.8 |
|---|---|---|---|---|---|
| GCN | 21.3 | 28.7 | 34.7 | 38.8 | 43.0 |
| GIN | 15.6 | 21.5 | 25.5 | 28.5 | 33.1 |

- [DERIVED] The GCN value at CR 18.8 (43.0) equals Fig.14's GCN-Cora value. The GIN maximum
  (33.1) differs from Fig.14's GIN-Cora (32.4), whose Tab.VI CR is 13.4. So the GIN points
  here are not the Tab.VI configuration.

## 7. Motivation-figure numbers [STATED]
- Fig.1 (p.1): DRAM stall up to 86.2% of cycles; DRAM energy 90.2% (Reddit) — for a prior
  accelerator (p.2 §I).
- Fig.6 (p.4): DRAM access (MB) with Naive / METIS / Condense on Cora, CiteSeer, PubMed. Values
  were not transcribed (small labels; available on request).
