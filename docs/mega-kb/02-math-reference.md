# 02 — Mathematical Reference

Tags: [STATED] / [DERIVED] / [INTERP] / [UNSPECIFIED]. Equations are transcribed from the
paper as faithfully as plain text allows; the original typesetting is on p.3 and p.5.

---

## Eq.(1) — GNN layer (p.3 §II-A)

```
X^(l) = σ( Ã · X^(l−1) · W^(l−1) )                                          (1)
```

**[STATED]** "Ã is the normalized adjacency matrix, X^(l−1) and W^(l−1) are the feature
map including all nodes features and the learned parameters of the (l−1)-th layer,
respectively, and the σ is an activation function, e.g., ReLU." The paper says the forward
pass of GCN, GIN and GraphSage "can be uniformly represented" by Eq.(1).

**[STATED]** Execution order: of (ÃX)W and Ã(XW), MEGA uses **Ã(XW)**. This reduces MACs
for 2–3-layer GNNs whose input feature dimension is much larger than the hidden dimension. p.3.

- **[INTERP]** Eq.(1) is a simplification for GIN and GraphSage. GIN has a (1+ε) self term
  and an MLP; GraphSage has separate self/neighbor weights, mean aggregation and sampling
  (Tab.III, p.9). The paper does not say how these map onto Eq.(1) or the quantizer. → **Q-12**.

---

## Eq.(2) — Degree-Aware quantizer (p.5 §IV)

Plain-text transcription:

```
x̄_i = sign(x_i) · {  ⌊ |x_i| / α_i + 0.5 ⌋ ,       if |x| < α_i (2^([b_i]−1) − 1)
                   {  2^([b_i]−1) − 1 ,             if |x_i| ≥ α_i (2^([b_i]−1) − 1)       (2)
```

with **α_i = s_{d_i}**, **b_i = b_{d_i}**, where d_i is the degree of node i. [STATED] p.5.

Then **X_q = S_X · X̄**, with **S_X = diag(α_1, …, α_N)**. [STATED] p.5.

### What it does
- **[DERIVED]** This is a symmetric, round-to-nearest, clipped uniform quantizer with step
  α_i. The largest representable magnitude is (2^(b−1) − 1) integer levels: sign plus
  (b−1) magnitude bits. Values with magnitude ≥ α_i(2^(b−1)−1) saturate.
  - b = 2 → levels {−1, 0, +1}; b = 4 → {−7..+7}; b = 8 → {−127..+127}.
  - **b = 1 → 2^0 − 1 = 0, so every value quantizes to 0.** → see **Q-06 / Q-07**.
- **[DERIVED]** All F features of a node share one (α_i, b_i), i.e. per-node (per-row)
  quantization, with the parameters shared by all nodes of the same degree.

### Notation issues in the printed equation
- **[STATED]** The first case's condition is printed as `|x|`, the second as `|x_i|`.
  [INTERP] Both mean the same element; we take x_i elementwise over the F features of node i.
- **[UNSPECIFIED]** `[b_i]` (square brackets) is never defined. [INTERP] It most likely means
  rounding a continuous learnable bitwidth to an integer, since b is called "learnable".
  The paper does not state the rounding operator or the gradient through it. → **Q-04**.
- **[UNSPECIFIED]** Gradient of ⌊·+0.5⌋ during training (e.g. straight-through estimator)
  is not stated. → **Q-05**.
- **[UNSPECIFIED]** Whether "degree" means in-degree, and whether self-loops count. §I and
  §IV's opening say "in-degree"; Eq.(2)'s text says "degree". → **Q-01**.
- **[UNSPECIFIED]** Table size: "d is the maximum degree of nodes in a graph", so there is
  one (s, b) pair per degree value 1..d. Fig.7(a) also shows one row per degree (1, 2, 3).
  How degree-0 nodes are handled, and whether large-degree graphs (Reddit: avg degree
  491.99, Tab.II) really use one entry per degree, is not stated. → **Q-02**.

---

## Weight quantization (p.5 §IV, text between Eq.(2) and Eq.(3))

- **[STATED]** "we quantize W to the same bitwidth of 4bits for all GNNs in this paper."
- **[STATED]** Each column j of W has its own learnable scale β_j: s_W = (β_1, …, β_{F2}),
  where F2 is the output dimension of the layer.
- **[STATED]** "We can obtain the integer representation W̄ as Eq 2". The quantized weight
  is W_q = W̄ · S_W, with S_W = diag(β_1, …, β_{F2}).
- **[DERIVED]** Applying Eq.(2) with b = 4 gives integer weights in {−7..+7}. Fig.10(c) and
  §V-C-2 store W̄ as 4-bit values.
- **[UNSPECIFIED]** Whether the weight bitwidth is fixed at exactly 4 or learnable-then-clamped
  is not stated. → **Q-08**.

---

## Eq.(3) — Integer reformulation of the combination phase (p.5 §IV)

```
X · W  ≈  X_q · W_q  =  (S_X · X̄) · (W̄ · S_W)
                     =  (X̄ · W̄) ⊙ (s_X ⊗ s_W)                                 (3)
```

**[STATED]** "⊙ denotes an element-wise multiplication, and ⊗ denotes the outer product."
Example in Fig.7(b): y_ij multiplied by α_i·β_j.

- **[DERIVED]** (S_X X̄)(W̄ S_W) = S_X (X̄ W̄) S_W. Element (i, j) is α_i β_j (X̄W̄)_ij, which is
  exactly (X̄W̄) ⊙ (s_X ⊗ s_W), where s_X = (α_1..α_N) and s_W = (β_1..β_{F2}). The integer
  part X̄W̄ is what the hardware computes. The scale product is applied afterward.
- **[UNSPECIFIED]** Where in hardware the α_i β_j rescaling happens, and in what precision
  (the AUs do "integer addition and multiplication of scalar"; QN units use "scales from the
  Weight Buffer"). → **Q-16**.

### Aggregation phase quantization (p.5 §IV)
- **[STATED]** "For the aggregation phase, i.e., AB, where B = XW and A ∈ {0,1}^(N×N) is the
  adjacency matrix, we quantize B as the quantization way of W. Then the aggregation phase
  can also be performed by integer operations."
- **[STATED]** §V-D (p.7) says the Combination Engine results are "quantized to 4bit".
- **[INTERP]** "as the quantization way of W" plausibly means 4 bits with per-column scales.
  This is not stated. → **Q-09**.
- **[STATED vs STATED inconsistency]** Eq.(1) uses normalized Ã; §IV says A ∈ {0,1}; §V-D says
  AUs multiply by "the corresponding edge weight of A". → **Q-11**.

### Where X comes from
- **[STATED]** "In the combination phase, the core calculation is XW, where X is the feature
  map obtained from the aggregation phase of the previous layer." p.5 §IV.
- **[STATED]** In hardware, the Encoder applies ReLU, quantizes, and encodes aggregated
  nodes as the input of the next layer's combination. p.8 §V-D.
- **[INTERP]** So the degree-aware quantizer of Eq.(2) is applied to σ(Ã B) per layer, i.e.
  on the activations entering each combination. This matches Fig.3's "feature values ...
  after aggregation". How the first layer's raw input features are quantized is not
  stated. → **Q-10**.

---

## Eq.(4) — Memory penalty (p.5 §IV)

```
L_memory = ( (1/η) · Σ_{l=1..L} Σ_{i=1..N} dim^l · b_i^l  −  M_target )²       (4)
```

**[STATED]** "L is the number of layers in the GNNs, N is the total number of nodes, dim^l is
the length of the node features in l-th layer, b_i^l is the quantization bitwidth for node i
in l-th layer, M_target is the target memory size on the total node features memory size,
and η = 8 ∗ 1024, which is a constant to convert the unit of memory size to KB."

- **[DERIVED]** Σ dim^l · b_i^l is the total node-feature storage in **bits**, assuming dense
  storage (every feature stored at b_i^l bits). Dividing by 8·1024 converts bits → KB. So
  M_target is in KB.
- **[DERIVED]** The penalty is two-sided (squared): it pushes memory *toward* M_target, not
  just below it.
- **[DERIVED]** The penalty counts dense storage. It ignores sparsity and the
  Adaptive-Package overhead, and it ignores weight memory.
- **[INTERP]** The layer index l on b_i^l implies a separate degree-indexed bitwidth table per
  layer. §IV defines s, b without a layer index. → **Q-03**.
- **[UNSPECIFIED]** The value of M_target for any experiment. → **Q-13**.
- **[UNSPECIFIED]** Whether b_i^l inside Eq.(4) is the rounded [b] or the continuous value
  (needed for gradients to reach b). → **Q-04**.

## Eq.(5) — Total loss (p.5 §IV)

```
L_total = L_task + λ · L_memory                                                 (5)
```

**[STATED]** "L_task is the task-related loss function and λ is a penalty factor on
L_memory." The model and quantization parameters are trained jointly by this loss.

- **[STATED]** λ trades accuracy for compression. Example: with a lower penalty, CiteSeer GCN
  reaches 71.2% at 2.24 bits, vs 70.6% at 1.87 bits in Tab.VI. p.10 §VI-B.
- **[UNSPECIFIED]** λ values, optimizer, schedule, epochs, initialization. → **Q-13, Q-14**.

---

## Symbol table

| Symbol | Shape / type | Meaning | Learned? | Source |
|---|---|---|---|---|
| N | scalar | number of nodes | — | p.5 |
| F | scalar | node feature dimension (of the layer input) | — | p.5 |
| F2 | scalar | output dimension of the current layer | — | p.5 |
| X, X^(l) | N×F real | node feature map (layer l) | — | p.3, p.5 |
| x_i | F real | features of node i | — | p.5 |
| W, W^(l) | F×F2 real | layer weights | yes (model) | p.3, p.5 |
| Ã | N×N | normalized adjacency (Eq.1) | — | p.3 |
| A | N×N, {0,1} | adjacency in §IV aggregation | — | p.5 |
| B | N×F2 | B = XW, input to aggregation | — | p.5 |
| σ | fn | activation, e.g. ReLU | — | p.3 |
| d | scalar int | maximum node degree in the graph | — | p.5 |
| d_i | int | degree of node i (in-degree? Q-01) | — | p.5 |
| s = (s_1..s_d) | d real | scale table indexed by degree | **yes** | p.5 |
| b = (b_1..b_d) | d (real?) | bitwidth table indexed by degree | **yes** (method Q-04) | p.5 |
| α_i = s_{d_i} | real | scale of node i | via s | p.5 |
| b_i = b_{d_i} | int-ish | bitwidth of node i; `[b_i]` undefined | via b | p.5 |
| x̄_i, X̄ | ints | integer (fixed-point) features | — | p.5 |
| S_X | N×N diag | diag(α_1..α_N) | — | p.5 |
| s_X | N | vector (α_1..α_N), used in ⊗ | — | p.5 Eq.3 |
| X_q | N×F | S_X · X̄ (dequantized features) | — | p.5 |
| β_j | real | scale of column j of W | **yes** | p.5 |
| s_W | F2 | (β_1..β_{F2}) | yes | p.5 |
| S_W | F2×F2 diag | diag(β_1..β_{F2}) | — | p.5 |
| W̄ | F×F2 ints | 4-bit integer weights | — | p.5 |
| W_q | F×F2 | W̄ · S_W | — | p.5 |
| ⊙, ⊗ | ops | elementwise product, outer product | — | p.5 |
| L | scalar | number of GNN layers | — | p.5 Eq.4 |
| dim^l | int | node feature length in layer l | — | p.5 Eq.4 |
| b_i^l | — | bitwidth of node i in layer l | — | p.5 Eq.4 |
| M_target | KB ([DERIVED]) | target node-feature memory | hyper-param | p.5 Eq.4 |
| η | 8·1024 | bits→KB conversion | constant | p.5 Eq.4 |
| λ | scalar | memory penalty factor | hyper-param | p.5 Eq.5 |
| L_task | — | task loss (e.g. cross-entropy; not specified) | — | p.5 Eq.5 |

## How the equations connect

```
            degree d_i of node i  (graph structure, fixed)
                      │ index
                      ▼
   learnable tables s, b ──► α_i, b_i ──► Eq.(2) quantize x_i ──► X̄ (ints), S_X
                                  │                                   │
                                  │                       W ─(Eq.2, b=4, β_j)─► W̄, S_W
                                  │                                   │
                                  │                                   ▼
                                  │                Eq.(3): XW ≈ (X̄W̄) ⊙ (s_X ⊗ s_W) = B
                                  │                                   │ quantize B "like W"
                                  │                                   ▼
                                  │                    aggregation  A·B (integer) ──► σ ──► next X
                                  │                                                         │
                                  ▼                                                         │
   Eq.(4): L_memory = ((1/η) Σ_l Σ_i dim^l b_i^l − M_target)²       (loops back to Eq.2 ◄───┘
                                  │                                   for the next layer)
                                  ▼
   Eq.(5): L_total = L_task + λ L_memory   ──► gradients update W, s, β, and b
```

- **[DERIVED]** Eq.(2) is the only place b enters the forward pass. Eq.(4) is the only place
  memory size enters the loss. The bitwidth table therefore gets gradient from both the
  task loss (through quantization error) and the memory penalty. *How* gradient flows
  through the discrete b is unspecified (Q-04).
- **[DERIVED]** Because b is indexed by degree, Eq.(4)'s Σ_i b_i^l equals
  Σ_k (count of nodes with degree k) · b_k^l. High-frequency (low) degrees dominate the
  memory term. This matches the power-law argument on p.4.

## Worked example (Fig.7, p.5) — [STATED] figure content

- Fig.7(a): a table with columns Degree | Scale | Bitwidth, rows (1, s1, b1), (2, s2, b2),
  (3, s3, b3). Nodes in a small graph are annotated with their degree.
- Fig.7(b): the N×F integer feature matrix (x̄ entries, rows scaled by α_i) times the F×F2
  weight matrix (columns scaled by β_j) gives outputs y_ij scaled by α_i β_j.

## Compression ratio (CR) — definition used in evaluation

- **[STATED]** "we count the average bitwidths for node features and take the ratio of the
  average bitwidth / 32 as the theoretical compression ratio, named 'CR'." p.9 §VI-A-2.
- **[DERIVED]** Tab.VI values are ≈ 32 / (average bits), the inverse of the stated ratio:
  Cora GCN 32/1.70 = 18.8 (table: 18.6×), Cora GIN 32/2.37 = 13.5 (13.4×),
  CiteSeer GCN 32/1.87 = 17.1 (17.0×). Others match to rounding. Fig.22 (p.12) labels the
  top Cora compression ratio as **18.8**. → **Q-15**.
- **[UNSPECIFIED]** How "average bitwidth" is averaged (over nodes? weighted by dim^l?
  over layers?). → **Q-15**.
