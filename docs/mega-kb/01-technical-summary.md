# 01 — Technical Summary

Tags: [STATED] / [DERIVED] / [INTERP] / [UNSPECIFIED]. See README for conventions.

## 1. Problem being solved

- **[STATED]** GNN inference has two phases (MPNN framework): **aggregation** (collect
  neighbor features over a sparse, irregular graph) and **combination** (transform features,
  e.g. with an MLP). p.1 §I; p.2 §II-A, Fig.2.
- **[STATED]** Even with dedicated accelerators, **DRAM access is the bottleneck**:
  "the stall caused by DRAM access accounts for up to 86.2% of the overall execution cycles
  on HyGCN" and "the energy consumption of DRAM access also dominates ... e.g., 90.2% for
  HyGCN on Reddit." p.2 §I, Fig.1 (p.1).
- **[STATED]** Existing GNN quantization (DQ [47], LP [65]) works to 8 bit without accuracy
  loss but degrades at lower bitwidths. For GIN on CiteSeer with DQ, accuracy drops up to 5.3%
  at 4 bit. p.2 §I; p.3 §III-A, Tab.I.
- **[STATED]** Existing methods use one bitwidth for all nodes ("data-independent"),
  ignoring graph topology. p.3 §III-A.

## 2. Key observation (motivation for the algorithm)

- **[STATED]** "as the in-degree of a node increases, the feature value gets larger after
  aggregation," shown for GCN and GIN on Cora (Fig.3, average over 100 runs, in-degree
  bins [1,10], [11,20], [21,30], [31,40], [41,168]). p.3 §III-A, Fig.3.
- **[STATED]** In-degrees in real graphs often follow a power law: most nodes have low
  in-degree. So one bitwidth (e.g. 8 bit) for all nodes wastes storage. p.4 §III-A.
- **[STATED]** Manually assigning bitwidths per node is infeasible at scale, so bitwidth
  and scale are **learned** and assigned according to degree. p.4 §III-A.
- **[STATED]** Claimed novelty: "we are the first to introduce mixed-precision
  quantization to GNN accelerator design." p.4 §III-A.

## 3. Hardware challenges identified (motivation for the architecture)

1. **Storage of fine-grained mixed-precision sparse features** — [STATED] existing sparse
   formats (COO, CSR, Bitmap) must store at the highest bitwidth, giving "non-negligible
   storage underutilization" (Fig.4). Feature density varies widely across datasets/models
   (Fig.5: e.g. NELL GCN 12%, CiteSeer GraphSage 88%), making locating non-zeros and
   decoding costly. p.4 §III-B-1, Figs.4–5.
2. **Irregular memory access** — [STATED] graph partitioning (METIS, used by GROW/GCoD)
   improves locality inside subgraphs but leaves "sparse connections" between subgraphs.
   These still cause considerable DRAM accesses (Fig.6). p.4 §III-B-2, Fig.6.

## 4. Contributions (as listed by the authors, p.2 §I)

1. **Degree-Aware mixed-precision quantization** — learn quantization parameters (scale,
   bitwidth) per node degree, with a memory-size penalty. [STATED] p.2; detailed p.5 §IV.
2. **MEGA accelerator** — heterogeneous architecture with separate dataflows for the
   combination and aggregation phases. Includes the **Adaptive-Package** storage format
   and **Condense-Edge** scheduling. [STATED] p.2; detailed p.5–9 §V.
3. **Results** — up to 15.8% accuracy improvement and up to 18.6× compression ratio vs.
   8× for DQ. 38.3×/7.1×/4.0×/3.6× speedup and 47.6×/7.2×/5.4×/4.5× energy savings vs.
   HyGCN/GCNAX/GROW/SGCN. Implemented in 28nm. [STATED] p.2, p.1 Abstract.
   - [DERIVED] The "15.8%" matches Tab.VI PubMed GCN: Ours 78.3 − DQ 62.5 = 15.8 points
     (p.10). The "18.6×" is Tab.VI Cora GCN.

## 5. Algorithm: Degree-Aware Quantization (summary; equations in `02-math-reference.md`)

- **[STATED]** Base layer: X^(l) = σ(Ã X^(l−1) W^(l−1)), Eq.(1). Computed in the order
  **Ã(XW)** (combination first), which "greatly reduces the number of MACs" when input
  feature dimension ≫ hidden dimension. p.3 §II-A.
- **[STATED]** Each node i's features are quantized with a scale α_i and bitwidth b_i
  looked up from learnable tables s = (s_1..s_d), b = (b_1..b_d) **indexed by the node's
  degree d_i**. Here d is the max degree. Eq.(2), p.5 §IV.
- **[STATED]** Weights W: 4 bit for all GNNs, one learnable scale β per output column.
  p.5 §IV.
- **[STATED]** Combination becomes an integer matmul followed by elementwise scaling by
  the outer product of node scales and column scales, Eq.(3). The aggregation input B = XW
  is quantized "as the quantization way of W", so aggregation is also integer. p.5 §IV.
- **[STATED]** Training loss = task loss + λ·(memory-size penalty)², Eq.(4)–(5). This lets
  accuracy be traded against memory. p.5 §IV.
- **[STATED]** Training cost: 2.04× FP32 training time on average, memory up to 12.8% more,
  on one RTX 3090. DQ costs 1.58× the training time of the Degree-Aware method. p.12–13 §VII-1.

## 6. Architecture overview (details in `03-hardware-mapping.md`)

**[STATED]** p.5 §V-A, Fig.8: a heterogeneous design with

- **Combination Engine** — 4 Combination Tiles (Fig.8). Each has a Decoder and a
  Combination Unit (Bit FIFO, Weight Reg, C-PEs made of Bit-Serial Engines). It computes
  B = XW in a **row-product, bit-serial** manner over mixed-precision sparse X and 4-bit
  dense W. p.6–7 §V-C.
- **Aggregation Engine** — Encoder, Condense Unit, and an Aggregation Tile of Aggregation
  Units (AUs). It computes AB in an **outer-product** manner with A in CSC. Outputs are
  quantized/ReLU'd/encoded to Adaptive-Package by the Encoder and fed back as the next
  layer's combination input. p.7–8 §V-D.
- **Buffers** — Input, Weight, Combination, Sparse, Edge, Aggregation buffers, all
  ping-pong. Plus a Controller. p.6 §V-A, Fig.8.
- **Adaptive-Package format** — packages of same-bitwidth non-zero values from successive
  nodes, with a 2-bit Mode (length class) and 3-bit Bitwidth header. Separate per-node
  bitmaps give non-zero positions. p.6 §V-B.
- **Condense-Edge scheduling** — the graph is METIS-partitioned offline into subgraphs and
  aggregated one subgraph at a time. While the current subgraph is aggregated, combined
  node features needed by other subgraphs' cross-partition ("sparse") edges are copied
  contiguously into a Sparse Buffer region per subgraph, so later fetches are contiguous.
  p.8–9 §V-E, Alg.1, Fig.13.

## 7. Evaluation methodology (summary; numbers in `05-eval-reference.md`)

- **[STATED]** Models: GCN, GIN, GraphSage. Datasets: Cora, CiteSeer, PubMed, NELL, Reddit.
  p.9 §VI-A-1, Tab.II–III.
- **[STATED]** Algorithm baselines: FP32 and DQ-INT4. CR = "the ratio of the average
  bitwidth / 32" (see Q-15 on this definition). p.9 §VI-A-2.
- **[STATED]** Hardware: Verilog RTL synthesized with Synopsys DC, TSMC 28nm, 1GHz.
  SRAM modeled with CACTI-7.0. DRAM is HBM1.0 at 256GB/s, modeled with Ramulator; DRAM
  energy follows HyGCN's methodology [56]. Performance from in-house **cycle-accurate
  simulators** "validated with the HDL design at the cycle level." p.9 §VI-A-3.
- **[STATED]** Fairness: all accelerators get the same 392KB on-chip buffer. Arithmetic is
  normalized via BitOPs (32-bit fixed-point multiply = 1024 BitOPs). Processing-unit areas
  are matched. p.9–10 §VI-A-3.

## 8. Headline results (summary)

| Metric | Value | Source |
|---|---|---|
| Best compression | 18.6× (Cora GCN, 1.70 avg bits, 80.9% vs FP32 81.5%) | p.10 Tab.VI |
| Speedup (geomean) vs HyGCN / GCNAX / GROW / SGCN | 38.3× / 7.1× / 4.0× / 3.6× | p.10 §VI-C-1, Fig.14 |
| DRAM access reduction | 108.1× / 10.5× / 8.4× / 7.3× | p.11 §VI-C-2, Fig.16 |
| Energy savings | 47.6× / 7.2× / 5.4× / 4.5× | p.11 §VI-C-3, Fig.17 |
| Area / power | 1.869 mm² / 194.98 mW | p.10 Tab.IV |

**Ablation** [STATED] p.12 §VI-D, Fig.19 (GCN, relative to HyGCN-C): quantization with
Bitmap storage gives 4.8×. Adding Adaptive-Package gives a further 4.7× (22.4×). Adding
Condense-Edge gives a further 1.1× (24.8×). DRAM access: 5.8× from quantization, 2.5×
from Adaptive-Package, 4.4× from Condense-Edge.

## 9. Extensions discussed (not part of the baseline design)

- **[STATED]** Condense-Edge works without METIS partitioning with a small loss: 3.50×
  speedup (↓3% vs. with METIS) and 3.95× energy (↓14%) "compared with the state-of-the-art
  accelerator" (which one is unspecified — Q-32). p.13 §VII-2.
- **[STATED]** GAT: 16.5× CR (from 9.6×), CiteSeer FP32 72.5% vs Degree-Aware 71.9%.
  Estimated 1.5% area overhead using A³'s softmax. p.13 §VII-3. Not evaluated in hardware.
