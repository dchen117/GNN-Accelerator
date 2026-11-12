# 03 — Hardware Mapping (algorithm → MEGA hardware)

Tags: [STATED] / [DERIVED] / [INTERP] / [UNSPECIFIED]. Section numbers refer to the paper.

---

## 1. Mapping table

| Algorithm step (see 02) | Hardware unit | Dataflow | Source |
|---|---|---|---|
| Store quantized sparse X^(l) (Eq.2 output) | Input Buffer, DRAM; **Adaptive-Package** format + per-node bitmaps | — | p.6 §V-A, §V-B |
| Store W̄ (4-bit) and scales | Weight Buffer → Weight Reg | — | p.6 §V-A; p.7 §V-C-2; p.8 §V-D (scales "from the Weight Buffer") |
| Unpack X̄ rows, locate non-zeros | **Decoder** (Package Division Unit, Bit Selectors, Weight Index Generator) | — | p.6–7 §V-C-1, Fig.10(a)(b) |
| X̄W̄ (integer part of Eq.3) | **Combination Unit**: Bit FIFO, Weight Reg, crossbar, #m C-PEs × #n BSEs | **row product, bit-serial** | p.7 §V-C-2, Fig.10(c), Fig.11 |
| Sum partial results across feature slices | Combination Tiles' outputs "added together" | — | p.6 §V-A |
| Rescale by s_X ⊗ s_W (Eq.3) | [UNSPECIFIED] → Q-16 | — | — |
| Quantize B "like W" (4 bit) | [UNSPECIFIED] which unit → Q-09/Q-16 | — | p.5 §IV; p.7 §V-D ("quantized to 4bit") |
| AB (aggregation) | **Aggregation Tile** of **AUs** (256), A in **Edge Buffer (CSC)**, 16-bit partial sums in **Aggregation Buffer** | **outer product** | p.7–8 §V-D, Fig.12(a) |
| Graph partition | METIS, **offline** | per-subgraph | p.8 §V-E |
| Cross-subgraph ("sparse connection") features | **Condense Unit** + **Sparse Buffer** (Alg.1) | reorder into contiguous regions | p.8–9 §V-E, Fig.13 |
| σ = ReLU, Eq.(2) quantization of next-layer X, bitmap generation, packing | **Encoder**: #m QN units + package register | greedy packing | p.8 §V-D |
| Overlap of phases | ping-pong buffers; "fuse the two phases" | pipeline | p.6 §V-A |

## 2. Top-level workflow (p.6 §V-A, Fig.8) — [STATED]

1. Encoded input nodes and weights are prefetched from DRAM into the **Input Buffer** and
   **Weight Buffer**.
2. Node features are read from the Input Buffer and dispatched to the **parallel Combination
   Tiles**. "each tile is responsible for processing a slice of features in a node."
3. After decoding, features go to the Combination Unit and compute B = XW with the
   corresponding weights, using a **row-product** dataflow ("more suitable for the wide range
   of sparsity variation in features").
4. "The results from different Combination Tiles can be added together to generate the final
   results or partial sums of one node once slices in all tiles have been processed."
5. The graph is divided into subgraphs, which are aggregated one after another. A node that
   finishes combination is sent to the Aggregation Engine. It is also stored in the
   **Combination Buffer**, or reordered into the **Sparse Buffer** by the Condense Unit.
6. Aggregation uses an **outer-product** dataflow over A (Edge Buffer) to reuse each generated
   node. Partial sums are kept in the **Aggregation Buffer**.
7. Aggregated nodes are quantized and encoded by the **Encoder** "and can be forwarded to the
   Combination Engine to fuse the two phases."
8. "all the buffers use the ping-pong mechanism for pipeline processing."

- [UNSPECIFIED] The exact schedule: whether combination of all nodes in a layer finishes
  before aggregation of subgraph 1 completes, the order in which nodes are combined, and how
  "fusing" across layers is sequenced. → **Q-28**.
- [UNSPECIFIED] How the input feature dimension F (e.g. 1433 for Cora, 61278 for NELL) is
  tiled across the 4 tiles × c-wide slices, and where cross-tile partial sums are accumulated
  when F > 4c. → **Q-21, Q-28**.

## 3. Adaptive-Package storage format (p.6 §V-B, Fig.9)

### Stated design — [STATED]
- A **package** is the primitive storage unit and has 3 segments:
  **Mode (2 bits) | Bitwidth (3 bits) | Val Array (adaptive length)**.
  - Mode: length class of the Val Array. "00 for short, 01 for medium, and 10 for long."
  - Bitwidth: the bitwidth of the values in this package, "ranging from 1 to 8bit".
  - Val Array: **non-zero values only**.
- **Bitmaps:** "Each node has its own bitmap indices to record the locations of the non-zero
  values, which are stored separately from the packages" (Fig.9(b)).
- **Shared bitwidth per package:** all values in a package have the same bitwidth. "Each
  package stores the non-zero values of successive nodes until the package is full or the
  quantization bitwidth of the node changes."
- **Adaptive length:** when the bitwidth changes before a package is full, the rest is
  zero-padded. To limit padding, there are three lengths (short, medium, long), selected
  "based on the sparsity and quantization bitwidth of each node."
- **Lengths:** "We empirically set (short, medium, long) = (64bits, 128bits, 192bits)".
  Fig.21 (p.12) explores alternatives; optimal settings differ per dataset (PubMed's optimum
  is (400, 512, 800)), and (64, 128, 192) was chosen across datasets. p.12 §VI-E-1.
- **Encoder packing rule (p.8 §V-D):** "the package register continuously stores non-zero
  values in successive nodes until the maximum package length is reached or the node's
  bitwidth changes. Then we select the appropriate Mode according to the bit length of the
  values in the package register."

### Fig.9 example — [STATED figure content], checked at high resolution
| Node | Features (Fig.9a) | Bitwidth | Non-zeros | Bitmap shown (Fig.9b) |
|---|---|---|---|---|
| Node1 | 01 10 00 11 00 | 2 | 01, 10, 11 | **1 1 0 1 1** |
| Node2 | 100 010 001 000 000 | 3 | 100, 010, 001 | 1 1 1 0 0 |
| Node3 | 101 110 000 001 000 | 3 | 101, 110, 001 | 1 1 0 1 0 |
| Node4 | 0 0 1 0 1 | 1 | 1, 1 | 0 0 1 0 1 |

- Fig.9(c) fixed 32-bit packages: P1 = 01 10 11 + 26 bits pad; P2 = Node2+Node3 values
  (18 bits) + 14 pad; P3 = 1 1 + 30 pad.
- Fig.9(d) adaptive packages: P1 = `01 | 010 | 01 10 11 | 2 pad` (label "8bits"),
  P2 = `10 | 011 | 100 010 001 101 110 001 | 2 pad` ("20bits"),
  P3 = `00 | 001 | 1 1 | 2 pad` ("4bits").
- **[DERIVED]** In the figure, the length labels (8, 20, 4) equal Val Array + padding and
  **exclude the 5-bit header**. [INTERP] This suggests 64/128/192 may also be Val-Array-only
  lengths, but the text does not say. → **Q-17**.
- **[DERIVED] Inconsistency:** Node1's non-zero pattern is 1 1 0 1 0, but the bitmap shows
  1 1 0 1 1. → **Q-20**.
- **[DERIVED] Inconsistency:** the text says "by using the short mode, we can reduce the
  padding from 26bits to 2bits in Package1", but Fig.9(d) Package1's Mode field is `01`,
  which is *medium* under the stated encoding. Package3 (`00`) is the short one. → **Q-18**.
- **[DERIVED]** Node2's and Node3's values share Package2 because both are 3-bit and
  consecutive. This illustrates "successive nodes until ... bitwidth changes."
- **[DERIVED]** The figure's integer codes exceed Eq.(2)'s signed range. For example, Node3
  has 3-bit value `110` = 6 > 2^(3−1)−1 = 3, and Node1 has 2-bit `11` = 3 > 1. So either
  the figure is schematic or stored values are unsigned magnitudes. → **Q-06**.

### Not specified
- How a node's bitmap and its values are located in memory (pointers/offsets per node or per
  package), and how bitmaps are stored/compressed. → **Q-19**.
- Whether a node's values can span multiple packages. [DERIVED] They must sometimes: e.g.
  128 non-zero 4-bit values = 512 bits > 192-bit long package. → **Q-19**.
- Use of Mode code `11`. → **Q-18**.

## 4. Combination Engine (p.6–7 §V-C, Figs.10–11)

### 4.1 Decoder (Fig.10(a)(b)) — [STATED]
- **Package Reg** caches a package from the Input Buffer.
- **Package Division Unit** extracts Mode and Bitwidth and sends the non-zero values to a
  **Bit Selector** (Fig.10(a) shows 2Bit, 3Bit, …, 8Bit selectors). The selector picks the
  valid bits and emits **(feature bits, shift bit)** pairs to the Combination Unit.
- Example (p.7): Val Array `10, 01, 10, 11` at 2 bits produces pairs **(0101, 0)** and
  **(1011, 1)**. "Shift bit is used to control the number of shifts in bit-serial computation."
  - **[DERIVED]** (0101, 0) collects the LSBs of the four values (0,1,0,1) and (1011, 1)
    collects their MSBs (1,0,1,1). So the decoder emits **bit-planes across n values,
    LSB-plane first**, and the shift bit is the bit position. Signed handling is not shown
    (all example values are non-negative). → **Q-22**.
- "Since features in each package use the same bitwidth, the data format is fixed, which
  can substantially decrease the hardware complexity of the bit selectors."
- **Weight Index Generator:** a **Parallel Sum Unit** converts a node's bitindex (bitmap) into
  the row indices of W for each non-zero (prefix sum, Fig.10(b)). These indices drive
  weight loading.
  - [DERIVED] from Fig.10(b): bitmap 1,0,1,0,1,1,0,0 → prefix sums 1,1,2,2,3,4,4,4 → AND with
    bitmap gives weight indices 1,0,2,0,3,4,0,0 (checked at high resolution). [INTERP] Entry p
    (row p of W) holds k = the 1-based rank of the non-zero at feature position p, with 0 meaning
    "zero feature, skip row". So the k-th non-zero is paired with W row p. How this list is
    turned into crossbar addresses is not described. → **Q-23**.

### 4.2 Combination Unit (Fig.10(c)) — [STATED]
- Computes **B = XW** (X is 1 to 8 bit) with **row product** + **bit-serial**.
- Composed of a **Bit FIFO**, a **Weight Reg**, and **#m C-PEs**. #m C-PEs perform a
  vector-matrix product of **one row of X** with **m columns of W**. Each C-PE produces one
  of the m output features of the same row of B.
- **W̄ is dense and 4-bit**. It is buffered in the Weight Reg and delivered via a **crossbar**
  "according to the row indices from the Decoder".
- Each **C-PE** = **#n BSEs** + **Adder Tree** + **Shifter-Acc**. Different BSEs handle
  different non-zero elements of the same row of X.
- **BSE** = an AND unit + 3 registers (weight, feature bit, result).
- C-PEs are split into **two equal groups**. "C-PEs in each group share the same features."
  Feature bits flow from the left group to the right group for reuse.
- Per cycle within a C-PE: AND(feature bit, weight) in each BSE → Adder Tree sum → shift by
  the shift bit → accumulate in Shifter-Acc until the final output.
- **Weight reuse:** "the weights are reused across different bits of the node features. Given
  that the minimum bitwidth of the node features is 2bit in Degree-Aware method, each weight
  is reused at least twice, which reduces the weight bandwidth requirement ... by half."
  (contradicts the 1–8-bit range → **Q-07**).
- "Because the density of W is always 100%, the BSEs in the same row are all busy during the
  systolic process."
- **Crossbar:** "A c × n(m/2 × 4 bits) crossbar is used to unicast corresponding rows of W̄ to
  C-PEs, where c is the dimension of the feature slice processed in each Combination Tile."

### 4.3 Fig.11 dataflow example (m=2, n=2) — [STATED]
- Cycle 1: first feature bits and corresponding weights are loaded into BSE1, BSE2 of C-PE1.
- Cycle 2: C-PE1's BSEs do the AND and load the second bits of the same non-zero values.
  The crossbar loads weights only to C-PE2, because C-PE1's weights are reused across bits.
  C-PE1's bits are forwarded to C-PE2.
- Cycles 3–4 proceed similarly.
- [INTERP] With b-bit features, each group of n non-zeros occupies a C-PE for b cycles, and
  the weight load for the next C-PE is staggered by one cycle (systolic). The exact
  steady-state throughput formula is not given. → **Q-24**.

### 4.4 Parameter values — what Tab.IV implies
**[STATED]** Tab.IV (p.10): "BSEs … 4 × 8 × 32"; "Crossbar … 32 × 8(64bit)"; "Encoder … 32 QN
units"; Fig.8 shows 4 Combination Tiles; Tab.V calls MEGA's compute "4x8x32 BSEs".

**[DERIVED]** If the crossbar c × n(m/2 × 4 bits) = 32 × 8(64 bit), then c = 32, n = 8, and
m/2 × 4 = 64 → **m = 32**. The Encoder has "#m QN units" = 32, which agrees with m = 32.
So "4 × 8 × 32" = 4 tiles × n=8 BSEs × m=32 C-PEs = **1024 BSEs**.
**[INTERP]** This assignment is consistent with every stated number but is not stated
explicitly. In particular, c = 32 and m = 32 are numerically equal, so the crossbar's
"32" could be either. → **Q-21**.

**[DERIVED]** The fairness note on p.9 says "32bit fixed-point multiplication equivalent to
1024 BitOPs". 1024 BSEs is also the BSE count above. This is consistent but not stated as
related.

## 5. Aggregation Engine (p.7–8 §V-D, Fig.12(a)) — [STATED]

- Components: **Encoder**, **Condense Unit**, **Aggregation Tile** of **AUs** (Tab.IV: 256
  AUs). AUs "perform integer addition and multiplication of scalar".
- "Considering the results (quantized to 4bit) from Combination Engine are node-by-node and
  nearly 100% dense, the Aggregation Engine calculates the AB in an outer product manner to
  fully reuse the generated nodes features."
- "Different dimensions of features in a node are unicasted to different AUs and multiplied
  with the corresponding edge weight of A broadcasted to each AU, which is stored in the Edge
  Buffer using the CSC format."
- Example Fig.12(a)(b): combined Node1 partially aggregates Node1, Node3, Node4, Node6 in
  sequence, i.e. it walks the CSC column of node 1.
- "If the features of one node cannot occupy all units, free units can be assigned to
  aggregate other nodes."
- "The multiplied results are added with **16bit partial sums** buffered in Aggregation
  Buffer."
- When a node's aggregation completes, it is sent to the Encoder.
- [UNSPECIFIED] Edge-weight precision and whether A is binary or normalized (Q-11). Overflow
  or saturation of the 16-bit sums. Mean aggregation (GraphSage) division. How sampling (25
  neighbors) is applied. How completion of a node is detected across subgraphs. → **Q-11,
  Q-12, Q-25**.

### Encoder — [STATED] p.8 §V-D
- **#m QN (Quantization-Nonlinear) units**: quantize #m values "using the scales from the
  Weight Buffer", apply **ReLU**, and generate a **bitindex** (zero / non-zero).
- Non-zeros are concatenated into a **package register** (Val Array). Greedy packing until the
  max package length is reached or the node's bitwidth changes. Mode is chosen "according to
  the bit length of the values in the package register". Header = Mode + Bitwidth.
- The package is written back to the **Input Buffer** as the next layer's combination input.
- [UNSPECIFIED] How the QN unit knows the node's degree-indexed (α, b). Whether the scale table
  lives in the Weight Buffer per layer. Last-layer handling (ReLU/no ReLU, output format).
  → **Q-26**.

## 6. Condense-Edge scheduling (p.8–9 §V-E, Alg.1, Figs.12–13)

### Motivation example (Fig.12(c)(d)) — [STATED]
- Assume a 128B DRAM access granularity, "i.e., a continuous memory region of two nodes where
  each node has 128-dimension features and is quantized to 4bit".
  - [DERIVED] 2 × 128 × 4 bit = 1024 bit = 128 B. ✓
- Without Condense-Edge, aggregating Subgraph2 via sparse connections needs Node1 (for Node3)
  and Node6 (for Node4). That is 2 DRAM accesses (Node1&2, Node6&7), each half-used.
  With Condense-Edge, Node1 and Node6 are already contiguous, so 1 access.

### Algorithm 1 (transcribed) — [STATED] p.8
```
1: Initialize: eID_list = {eID_FIFO_1, ..., eID_FIFO_n}: container of the sparse connections
   eIDs; eID_FIFO_i stores eIDs of the i-th subgraph in ascending order;
   Address_list: stores the initial memory addresses of the Sparse Buffer corresponding
   to different subgraphs.
2: Outputs: None (reorder the nodes features)
3: Condense-Edge(x, nID, SubNum):
4:   Partition the graph using METIS and initialize eID_list. subID = 0
5:   Store x into the Combination Buffer
6:   while subID < SubNum do
7:     eID_FIFO = eID_list[subID]
8:     if eID_FIFO[0] == nID then
9:       eID_FIFO.pop()            {# invalid the eID have matched}
10:      ptr = Address_list[subID] and store the node features x into Sparse Buffer
         according to the ptr      {# reorder the nodes features in Sparse Buffer}
11:      Increment ptr and Address_list[subID] = ptr
12:      subID = subID + 1         {# compare with the next subgraph}
13:    else
14:      subID = subID + 1
15:    end if
16:    If the corresponding region of the Sparse Buffer is full, write the data back to
       DRAM and reinitialize the ptr.
17:  end while
18: end
```

### Semantics — [STATED] p.8–9
- METIS partitioning is **offline**, so SubNum and each subgraph's sparse-connection IDs
  (eIDs) are known in advance and **stored in the Edge Buffer**. Example: eIDs 3 and 6 for
  Subgraph1.
  - [INTERP] eID_FIFO_i holds the IDs of **external** nodes whose features subgraph i needs
    (Subgraph1 = {1, 2} in Fig.12(d) needs nodes 3 and 6). This matches the matching
    description: "If there is a match, it implies that the node features will be utilized by
    this sparse connection".
- The Sparse Buffer is divided into **SubNum regions**, one per subgraph. Address_list holds
  each region's start address.
- When node x (ID nID) is produced by the Combination Engine, nID is compared with the
  **head** of each eID FIFO (eIDs are sorted ascending, and matched IDs are invalidated/popped).
  On a match, x is appended to that subgraph's region. A node is stored once per subgraph
  even if several nodes in that subgraph need it.
- "Regardless of whether there is a match or not, the node features need to be written to
  the Combination Buffer".
- When later aggregating another subgraph, the Sparse Buffer reads that subgraph's
  contiguous sparse-connection features from DRAM. On an eID match, features come from the
  Sparse Buffer. Otherwise they come from the Combination Buffer.
- **Condense Unit (Fig.13):** eID FIFOs (each **8 entries**, one subgraph each), one comparator
  (Cmp) per FIFO for parallel matching, a MUX indexed by subID, an Address Reg holding
  Ptr1..PtrK, and the Sparse Buffer with an Enable to DRAM. Tab.IV: **16 ID FIFOs**.
- **[DERIVED]** Head-only matching is correct only if nodes arrive at the Condense Unit in
  **ascending nID order** (or the same order used to sort the eIDs). [UNSPECIFIED] whether the
  Combination Engine guarantees this order. → **Q-27, Q-28**.
- [UNSPECIFIED] Behavior when a subgraph has more than 8 eIDs (FIFO refill from Edge Buffer?)
  or SubNum > 16. Region sizing (Sparse Buffer = 32 KB total). The number of METIS partitions
  per dataset. → **Q-27**.
- [DERIVED] Alg.1 as printed has `Partition the graph using METIS` *inside* the per-node
  function (L4) and resets subID there. [INTERP] L4 is an offline/one-time initialization
  written into the function for presentation. → **Q-27**.

## 7. On-chip configuration (p.10 Tab.IV) — [STATED]

| Component | Area (mm²) | Power (mW) | Config |
|---|---|---|---|
| BSEs | 0.053 | 14.70 | 4 × 8 × 32 |
| Aggregation Unit | 0.100 | 28.92 | 256 |
| Crossbar | 0.027 | 5.56 | 32 × 8 (64bit) |
| Condense Unit | 0.002 | 1.19 | 16 ID FIFOs |
| Encoder | 0.010 | 1.81 | 32 QN units |
| Decoder | 0.003 | 0.75 | — |
| Others | 0.004 | 0.80 | — |
| **Processing Unit total** | 0.199 (11%) | 53.73 (28%) | — |
| Aggregation Buffer | 0.540 | 46.56 | 128 KB |
| Combination Buffer | 0.452 | 35.19 | 96 KB |
| Input Buffer | 0.220 | 22.88 | 64 KB |
| Edge Buffer | 0.119 | 9.44 | 24 KB |
| Sparse Buffer | 0.154 | 12.86 | 32 KB |
| Weight Buffer | 0.190 | 14.32 | 48 KB |
| **Data Buffer total** | 1.67 (89%) | 141.25 (72%) | 392 KB |
| **Measured Total (28nm)** | 1.869 | 194.98 | — |

- [DERIVED] Component sums check: PU area 0.199 ✓, PU power 53.73 ✓; buffer area 1.675 ≈ 1.67 ✓,
  power 141.25 ✓, size 392 KB ✓; total power 53.73 + 141.25 = 194.98 ✓; total area
  0.199 + 1.675 = 1.874 vs "Measured" 1.869 (−0.005; [INTERP] rounding/measurement, minor →
  Q-34).
- Other stated parameters: 1 GHz clock; TSMC 28nm; HBM1.0 at 256 GB/s; Ramulator; CACTI-7.0
  (p.9 §VI-A-3).
- [UNSPECIFIED] Number of banks/ports per buffer, ping-pong split of each buffer, Controller
  design. → **Q-28, Q-29**.

## 8. What a cycle-accurate simulator would need (summary of gaps)

[DERIVED] from the sections above. To model MEGA cycle-by-cycle, the paper leaves open:
the tile/slice schedule for large F (Q-21, Q-28); the C-PE steady-state timing (Q-24); AU
assignment and CSC walk timing (Q-25); Encoder/package write timing (Q-17–Q-19, Q-26);
Condense-Edge capacity/overflow behavior and partition counts (Q-27); DRAM address layout
and Ramulator configuration (Q-29). The authors' simulator is not stated to be released
(Q-30).
