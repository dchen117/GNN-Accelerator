# 04 — Open Questions, Gaps, and Inconsistencies

Each entry lists what the paper says (with citations), what is missing or conflicting, and
which implementation path it blocks:

- `[PT]` = PyTorch quantization model; `[SIM]` = cycle-accurate simulator.
- **Severity:** `blocking` = can't implement faithfully without a decision;
  `affects-fidelity` = can implement but results may differ; `minor` = editorial/cosmetic.
- **Decision:** empty until we decide. When we fill one in, it is *our* choice, not the
  paper's. Record the rationale and date.

The paper is never "corrected" here. Possible readings are listed but not chosen.

---

## A. Algorithm (Degree-Aware Quantization)

### Q-01 — Which degree? `[PT][SIM]` blocking
- Says: "nodes with higher in-degrees" (p.2 §I); "quantizes nodes with different
  in-degrees" (p.5 §IV opening); but Eq.(2) text says "d_i is the degree of the i-th node" and
  "d is the maximum degree" (p.5). Fig.3 uses in-degree (p.3).
- Missing: in-degree vs total degree, for directed vs undirected graphs. Whether self-loops
  (added by GCN normalization) are counted. Whether sampled degree is used for GraphSage.
- Decision: —

### Q-02 — Granularity and size of the degree table `[PT][SIM]` blocking
- Says: s = (s_1..s_d), b = (b_1..b_d), "d is the maximum degree of nodes in a graph" (p.5).
  Fig.7(a) shows one row per degree (p.5).
- Missing: whether there is one entry per integer degree up to the maximum. Reddit has an
  average degree of 491.99 (Tab.II), so the max degree is large. Fig.3 uses degree *bins*.
  Also missing: handling of degree-0 nodes (no s_0) and of degrees not present in the graph.
- Decision: —

### Q-03 — Per-layer quantization parameters? `[PT][SIM]` blocking
- Says: Eq.(4) uses b_i^l, "the quantization bitwidth for node i in l-th layer" (p.5).
  Eq.(2) defines s, b without a layer index.
- Missing: whether each layer has its own (s, b) table.
- Decision: —

### Q-04 — How is the discrete bitwidth learned? `[PT]` blocking
- Says: bitwidth is "learnable" (p.5); Eq.(2) uses `[b_i]`, which is never defined.
- Missing: the parameterization of b (continuous + rounding?), the gradient estimator, the
  allowed range (1–8? 2–8? see Q-07), initialization of s and b, and whether Eq.(4) uses the
  rounded or continuous b.
- Decision: —

### Q-05 — Gradient through rounding `[PT]` blocking
- Says: Eq.(2) uses ⌊|x|/α + 0.5⌋ (p.5).
- Missing: the gradient estimator for rounding and clipping (STE or other). Gradients w.r.t.
  α (e.g. LSQ-style [13]? cited in related work p.3 but not stated as used).
- Decision: —

### Q-06 — Signed vs unsigned integer range `[PT][SIM]` blocking
- Says: Eq.(2) is sign(x)·min(round(|x|/α), 2^(b−1)−1) (p.5). The Encoder applies ReLU before
  packing (p.8 §V-D), so stored activations are ≥ 0.
- Conflict: Fig.9 (p.6) stores values beyond the signed range (2-bit `11` = 3 > 1; 3-bit `110` =
  6 > 3). With Eq.(2), b = 1 gives max level 0, so every value becomes zero. Yet 1-bit nodes
  appear in Fig.9 and the Bitwidth field supports 1 bit.
- Missing: whether post-ReLU activations use an unsigned b-bit range (0..2^b−1). How signed
  values are handled in the bit-serial datapath.
- Decision: —

### Q-07 — Minimum feature bitwidth: 1 or 2? `[PT][SIM]` affects-fidelity
- Says: "Bitwidth denotes the bitwidth (ranging from 1 to 8bit)" (p.6 §V-B); "X is from 1 to
  8bit" (p.7 §V-C-2); Fig.9 Node4 is 1-bit; **but** "the minimum bitwidth of the node features
  is 2bit in Degree-Aware method" (p.7 §V-C-2). Fig.10(a) shows 2Bit…8Bit selectors only.
- [DERIVED] Tab.VI reports average bits below 2 (Cora GCN 1.70, CiteSeer GCN 1.87; p.10).
  This is impossible if every node is ≥ 2 bits, unless "average bitwidth" counts something
  other than a per-node mean (see Q-15).
- Decision: —

### Q-08 — Weight quantization details `[PT]` affects-fidelity
- Says: W quantized to 4 bits for all GNNs, per-column learnable β, "integer representation W̄
  as Eq 2" (p.5).
- Missing: whether b = 4 is fixed (not learned), whether β is initialized like α, and whether
  bias terms exist or are quantized.
- Decision: —

### Q-09 — Quantization of B = XW before aggregation `[PT][SIM]` blocking
- Says: "we quantize B as the quantization way of W" (p.5 §IV). The Combination Engine
  results are "quantized to 4bit" (p.7 §V-D).
- Missing: per-column or per-tensor scales, whether they are learned, where they are applied,
  and the rounding mode. In hardware, which unit requantizes B (C-PE output? Aggregation
  Engine input?).
- Decision: —

### Q-10 — First-layer input and last-layer output `[PT][SIM]` blocking
- Says: X in XW is "obtained from the aggregation phase of the previous layer" (p.5). The
  Encoder quantizes aggregated outputs for the next layer (p.8).
- Missing: how raw input features X^(0) are quantized (degree-aware? what bitwidth? are they
  preprocessed offline?). Whether the final layer's output is quantized and whether ReLU is
  applied. Which layers Eq.(4)'s Σ_l covers.
- Decision: —

### Q-11 — Adjacency normalization and edge weights `[PT][SIM]` blocking
- Says: Eq.(1) uses normalized Ã (p.3). §IV says A ∈ {0,1}^(N×N) (p.5). §V-D: features are
  "multiplied with the corresponding edge weight of A broadcasted to each AU, which is stored
  in the Edge Buffer using the CSC format" (p.8). AUs do "integer addition and
  multiplication" (p.7).
- Missing: whether normalization coefficients are stored as edge weights, their format and
  precision, whether they are quantized, and whether normalization is folded into scales.
- Decision: —

### Q-12 — Model-specific operators (GIN, GraphSage) `[PT][SIM]` blocking
- Says: GCN/GIN/GraphSage "can be uniformly represented" by Eq.(1) (p.3). Tab.III: GIN uses Add
  aggregation, GraphSage uses Mean aggregation with 25 sampled neighbors, 256 hidden units (p.9).
- Missing: GIN's (1+ε) self term and multi-layer MLP; GraphSage's concat/separate self weight,
  sampling scheme, and mean in hardware; how degree is defined under sampling; whether
  quantization applies inside the MLP.
- Decision: —

### Q-13 — Loss hyperparameters `[PT]` blocking
- Says: Eq.(4)–(5), η = 8·1024 (p.5). λ is tuned to trade accuracy for size; e.g. lower λ gives
  CiteSeer GCN 71.2% at 2.24 bits (p.10 §VI-B).
- Missing: λ and M_target for every Tab.VI entry, and whether M_target is in KB
  ([DERIVED] from η).
- Decision: —

### Q-14 — Training setup `[PT]` blocking
- Says: Fig.3 averages over 100 runs (p.3). Tab.VI reports mean ± std (p.10). Training on one
  RTX 3090, 2.04× FP32 time (p.12 §VII-1).
- Missing: optimizer, learning rate, epochs, dropout, data splits (standard Planetoid? random?),
  number of runs behind the ± values, whether training starts from a pretrained FP32 model,
  and whether training is transductive or inductive for Reddit. No code release is mentioned.
- Decision: —

### Q-15 — Compression-ratio definition `[PT]` affects-fidelity
- Says: "take the ratio of the average bitwidth / 32 as the theoretical compression ratio"
  (p.9 §VI-A-2).
- [DERIVED] Reported CRs are ≈ 32 / avg-bits (the inverse), but not exactly: Cora GCN 32/1.70 =
  18.82 vs 18.6× reported; Cora GIN 32/2.37 = 13.50 vs 13.4×; CiteSeer GCN 32/1.87 = 17.11 vs
  17.0× (p.10 Tab.VI). Fig.22 (p.12) labels the top Cora point as CR **18.8**.
- Missing: how "average bitwidth" is computed (per-node mean? weighted by dim^l? over layers?
  includes zeros/sparsity?).
- Decision: —

### Q-16 — Where scales are applied in hardware `[SIM]` affects-fidelity
- Says: Eq.(3) needs (X̄W̄) ⊙ (s_X ⊗ s_W) (p.5). QN units use "scales from the Weight Buffer"
  (p.8).
- Missing: the unit that multiplies by α_i β_j, the scale format (FP? fixed-point?), and
  whether rescale and requantize are merged into one multiplier.
- Decision: —

## B. Hardware — storage format

### Q-17 — Do package lengths include the header? `[SIM]` affects-fidelity
- Says: (short, medium, long) = (64, 128, 192) bits (p.6 §V-B). Header = Mode 2b + Bitwidth 3b.
- [DERIVED] In Fig.9(d), the length labels (8/20/4 bits) cover Val Array + padding only.
- Missing: explicit statement for the 64/128/192 values.
- Decision: —

### Q-18 — Mode selection rule and encoding `[SIM]` affects-fidelity
- Says: Mode 00/01/10 = short/medium/long (p.6). Selected "based on the sparsity and
  quantization bitwidth of each node" (p.6); Encoder selects "according to the bit length of the
  values in the package register" (p.8).
- Conflict: the text says Package1 uses "the short mode", but Fig.9(d) Package1's Mode is `01`
  (medium) (p.6).
- Missing: the exact rule (smallest class that fits?), the meaning of code `11`, and whether a
  package can close early to pick a shorter mode.
- Decision: —

### Q-19 — Addressing nodes, bitmaps, and spanning `[SIM]` blocking
- Says: per-node bitmaps are stored separately from packages (p.6). Packages hold successive
  nodes (p.6).
- [DERIVED] A node can exceed one package (e.g. 128 non-zeros × 4 bit = 512 bits > 192), so
  spanning must exist.
- Missing: the per-node offset/pointer scheme, how the Decoder finds where node i starts, the
  bitmap memory layout (and whether bitmaps count in the CR/memory numbers), and DRAM
  alignment.
- Decision: —

### Q-20 — Fig.9(b) Node1 bitmap `minor`
- [DERIVED] Node1 = 01 10 00 11 00 → non-zero pattern 11010, but Fig.9(b) shows 11011 (p.6). The
  other three rows match. Likely a figure error. Doesn't affect implementation.
- Decision: —

## C. Hardware — compute engines

### Q-21 — Values of m, n, c and the meaning of "4 × 8 × 32" `[SIM]` blocking
- Says: Tab.IV BSEs "4 × 8 × 32", Crossbar "32 × 8(64bit)", Encoder "32 QN units" (= #m) (p.10).
  Crossbar = c × n(m/2 × 4 bits) (p.7). 4 Combination Tiles (Fig.8, p.5).
- [DERIVED] A consistent reading is 4 tiles, n = 8, m = 32, c = 32 (see 03 §4.4). Not stated
  explicitly.
- Missing: confirmation, and how F (e.g. 1433, 61278) is split into c-wide slices across tiles
  and passes.
- Decision: —

### Q-22 — Bit-serial details `[SIM]` affects-fidelity
- Says: (feature bits, shift bit) pairs; example (0101,0), (1011,1) (p.7).
  [DERIVED] LSB-plane first.
- Missing: signed handling (two's complement MSB weight? sign-magnitude?), Shifter-Acc width,
  accumulator precision before requantization, and where the 4 tiles' partial sums are added.
- Decision: —

### Q-23 — Weight index generation and crossbar control `[SIM]` affects-fidelity
- Says: Parallel Sum Unit converts the bitindex to W row indices (p.7, Fig.10(b)).
- Missing: how many bitmap bits are processed per cycle, how n non-zeros are grouped per C-PE
  pass, and the tail case when the number of non-zeros isn't a multiple of n.
- Decision: —

### Q-24 — C-PE systolic timing `[SIM]` blocking
- Says: C-PEs are split into two groups; features flow left → right; Fig.11 shows m=2, n=2 over
  4 cycles (p.7).
- Missing: the general schedule for m = 32 (16 per group?), pipeline fill/drain, and cycles per
  row as a function of (non-zeros, bitwidth, F2).
- Decision: —

### Q-25 — Aggregation Unit operation `[SIM]` blocking
- Says: 256 AUs, outer product, CSC, 16-bit partial sums, free AUs can serve other nodes
  (p.8, Tab.IV).
- Missing: the AU-to-feature mapping when F2 = 128 or 256 vs 256 AUs, 16-bit overflow handling,
  mean (division) for GraphSage, the edge-weight broadcast rate, and Aggregation Buffer
  capacity management (128 KB) for large subgraphs.
- Decision: —

### Q-26 — Encoder parameter lookup `[SIM]` affects-fidelity
- Says: QN units quantize using scales from the Weight Buffer, apply ReLU, make the bitindex,
  and do greedy packing (p.8).
- Missing: how a node's degree → (α, b) is looked up (is the degree stored per node? where?)
  and the throughput of packing.
- Decision: —

## D. Hardware — scheduling and memory system

### Q-27 — Condense-Edge capacity and corner cases `[SIM]` blocking
- Says: eID FIFO has 8 entries (p.9). 16 ID FIFOs (Tab.IV). eIDs stored in the Edge Buffer
  (p.8). Sparse Buffer regions spill to DRAM when full (Alg.1 L16).
- Missing: FIFO refill when a subgraph has more than 8 eIDs, behavior when SubNum > 16, region
  sizes, the number of METIS partitions per dataset, and the reload order when reading spilled
  regions back. Alg.1 L4 (partition inside the per-node routine) is presentational.
- [DERIVED] Head-only matching needs nodes to arrive in ascending ID order (see Q-28).
- Decision: —

### Q-28 — Global schedule / phase fusion `[SIM]` blocking
- Says: subgraphs are aggregated one after another. Combined nodes go to aggregation and also
  to the Combination/Sparse Buffers. Encoded outputs "can be forwarded to the Combination
  Engine to fuse the two phases". Buffers are ping-pong (p.6 §V-A, p.8 §V-E).
- Missing: the order in which nodes are combined (all nodes before/while aggregating subgraph
  1?), how combination of later subgraphs interleaves, the granularity of layer fusion, and
  the Controller's policy.
- Decision: —

### Q-29 — DRAM model details `[SIM]` affects-fidelity
- Says: HBM1.0, 256 GB/s, Ramulator, DRAM energy "following the methodology in [56]"
  (p.9 §VI-A-3). The 128B access granularity is used as an example (p.8).
- Missing: the Ramulator config (channels, timing), the address mapping of packages, bitmaps,
  edges, and Sparse Buffer regions, and the per-access energy values.
- Decision: —

### Q-30 — Simulator availability `[SIM]` affects-fidelity
- Says: "We implement cycle-accurate simulators ... validated with the HDL design at the cycle
  level" (p.9).
- Missing: no code or artifact release is stated.
- Decision: —

## E. Evaluation reporting

### Q-31 — Missing accuracy/bitwidth entries for hardware-evaluated cases `[SIM]` affects-fidelity
- Says: Fig.14/16/17 evaluate GCN on {Cora, CiteSeer, PubMed, NELL, Reddit}, GIN on {Cora,
  CiteSeer, PubMed}, GraphSage on {Cora, Reddit} (p.11).
- Missing: Tab.VI (p.10) has no entries for GCN-NELL, GCN-Reddit, or GIN-PubMed, so the
  bitwidths used in those hardware runs are unknown. It also has no NELL accuracy at all.
- Decision: —

### Q-32 — Unnamed comparison baseline in §VII-2 `minor`
- Says: "compared with the state-of-the-art accelerator, our MEGA without partitioning
  on-average obtains 3.50× (↓3% ...) and 3.95× (↓14%)" (p.13). The accelerator is not named.
- Decision: —

### Q-33 — Tab.I vs Tab.VI std mismatch `minor`
- Says: GIN CiteSeer DQ 4-bit = 60.8±2.1 (Tab.I, p.3) vs 60.8±1.2 (Tab.VI, p.10).
- Decision: —

### Q-34 — Area total `minor`
- [DERIVED] Component areas sum to 1.874 mm² vs "Measured Total" 1.869 mm² (Tab.IV, p.10).
  Power sums exactly.
- Decision: —

---

## Blocking summary by path

| Path | Blocking questions |
|---|---|
| `[PT]` PyTorch model | Q-01, Q-02, Q-03, Q-04, Q-05, Q-06, Q-09, Q-10, Q-11, Q-12, Q-13, Q-14 |
| `[SIM]` simulator | Q-01, Q-02, Q-03, Q-06, Q-09, Q-10, Q-11, Q-12, Q-19, Q-21, Q-24, Q-25, Q-27, Q-28 |

[DERIVED] For GCN only, Q-12 drops out of the PyTorch path. The remaining PT blockers are
training-recipe choices (Q-04, Q-05, Q-13, Q-14) and semantic choices (Q-01–Q-03, Q-06,
Q-09–Q-11). Tab.VI accuracy targets can serve as the acceptance check.
