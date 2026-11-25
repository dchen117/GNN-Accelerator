# 04 — Open Questions, Gaps, and Inconsistencies

**Status (2026-11-20):** Q-01–Q-15 have decisions (Q-11 decided as fold on Cora + NELL; Reddit, which is GraphSage-only, cannot test it). GCN matches Tab.VI on Cora, CiteSeer and PubMed (Q-04 amended: MSE clip init + re-fit). Q-16–Q-30 (simulator) were walked
through with the user on 2026-11-20. All Q-16–Q-30 entries are decided, plus Q-35. Q-31 onward are open. The PyTorch model implementing them is in `mega_pt/`.

Each entry lists what the paper says (with citations), what is missing or conflicting, and
which implementation path it blocks:

- `[PT]` = PyTorch quantization model; `[SIM]` = cycle-accurate simulator.
- **Severity:** `blocking` = can't implement faithfully without a decision;
  `affects-fidelity` = can implement but results may differ; `minor` = editorial/cosmetic.
- **Decision:** empty until we decide. When we fill one in, it is *our* choice, not the
  paper's. Record the rationale and date.

The paper is never "corrected" here. Possible readings are listed but not chosen.

**Fidelity rule (user, 2026-11-16).** This model is the MEGA *baseline* for comparing against
our own design, so it must stay fair to MEGA. Do not deviate from what the paper states about
its algorithm or hardware, even when a deviation would be simpler or more accurate. Choose
freely only where the paper is silent or inconsistent, and prefer the reading closest to what
it does state. An outcome of training that removes a key MEGA idea (e.g. every degree learning
the same bitwidth) is a finding to report, never a design choice to adopt. Every decision below
should say which kind it is: filling a gap, or departing from a stated detail.

---

## A. Algorithm (Degree-Aware Quantization)

### Q-01 — Which degree? `[PT][SIM]` blocking
- Says: "nodes with higher in-degrees" (p.2 §I); "quantizes nodes with different
  in-degrees" (p.5 §IV opening); but Eq.(2) text says "d_i is the degree of the i-th node" and
  "d is the maximum degree" (p.5). Fig.3 uses in-degree (p.3).
- Missing: in-degree vs total degree, for directed vs undirected graphs. Whether self-loops
  (added by GCN normalization) are counted. Whether sampled degree is used for GraphSage.
- Decision: **In-degree, counting self-loops, for directed and undirected graphs. GraphSage uses the
  sampled in-degree (sampled neighbors + self).** (2026-11-14)
  - Why: matches "in-degree" in §I/§IV and Fig.3. Counting self-loops makes the quantizer's degree
    the same d that GCN normalization uses, which is what lets normalization fold into
    degree-indexed scales (Q-11). Consequence: every node has degree ≥ 1.
  - Trade-off: GraphSage's degree changes with each sample, so sampling must also be done at
    inference with a fixed seed; otherwise a node's table entry would change between runs.

### Q-02 — Granularity and size of the degree table `[PT][SIM]` blocking
- Says: s = (s_1..s_d), b = (b_1..b_d), "d is the maximum degree of nodes in a graph" (p.5).
  Fig.7(a) shows one row per degree (p.5).
- Missing: whether there is one entry per integer degree up to the maximum. Reddit has an
  average degree of 491.99 (Tab.II), so the max degree is large. Fig.3 uses degree *bins*.
  Also missing: handling of degree-0 nodes (no s_0) and of degrees not present in the graph.
- Decision: **One (s, b) entry per integer degree 0..d_max. Degree 0 gets an entry like any other.**
  (2026-11-14)
  - Why: the literal reading of "d is the maximum degree" and Fig.7(a); no binning rule to invent.
  - Trade-off: rare degrees get few gradient updates, and degrees absent from the graph keep their
    initial values. Harmless here because training and inference use the same graph. With
    self-loops (Q-01), entry 0 is unused for GCN/GIN.

### Q-03 — Per-layer quantization parameters? `[PT][SIM]` blocking
- Says: Eq.(4) uses b_i^l, "the quantization bitwidth for node i in l-th layer" (p.5).
  Eq.(2) defines s, b without a layer index.
- Missing: whether each layer has its own (s, b) table.
- Decision: **Each quantized layer input has its own (s, b) table.** (2026-11-14)
  - Why: Eq.(4) indexes b_i^l by layer; activations differ in scale per layer.
  - Trade-off: L tables to store on chip instead of one (small: d_max+1 entries each).

### Q-04 — How is the discrete bitwidth learned? `[PT]` blocking
- Says: bitwidth is "learnable" (p.5); Eq.(2) uses `[b_i]`, which is never defined.
- Missing: the parameterization of b (continuous + rounding?), the gradient estimator, the
  allowed range (1–8? 2–8? see Q-07), initialization of s and b, and whether Eq.(4) uses the
  rounded or continuous b.
- Decision: **b is a continuous parameter, clamped to [2, 8]. The forward pass uses round(b); the
  backward pass uses a straight-through estimator (STE). Eq.(4) uses the continuous b.
  Initialization: train FP32 first; b = 4.0; α_d = max|x| / Q_max(4) = max|x|/7 over nodes of
  degree d (global layer max for degrees with no nodes).** (2026-11-14)
  - Why: simplest learnable-bitwidth scheme consistent with `[b_i]`. Max-based α starts with no
    clipping, so b and α are only pushed down by the losses.
  - Note [DERIVED]: under Eq.(2) the task loss reaches b only through clipped values (b sets the
    clip level); the memory penalty is what pushes b down.
  - **Amended 2026-11-15** (implementation of the decision, not a change to it): the degree tables
    learn a clip range c_d, with α_d = c_d / (2^(round(b_d)−1) − 1), instead of learning α_d
    directly. The initialization is the same (c_d = max|x|, so α_d = max|x|/7 at b = 4). The
    forward pass is still exactly Eq.(2), and the hardware still stores one α per degree.
    - Why: with α learned directly, lowering b shrinks the clip range 2× per bit. On Cora's binary
      features, b 4→2 maps every 1 to 1/7. The task loss then blocked bit reduction on layer 0,
      where 2 bits are lossless, and the penalty crushed the hidden layer instead. After the
      change, the Cora λ sweep matches Tab.VI (81.0 ± 1.1 at 1.60 bits). Numbers in
      `mega_pt/README.md`. The old behaviour is kept as `--x-param alpha`.
    - Trade-off: none in hardware. In optimization, b's gradient now also flows through α for
      unclipped values, so the note above (task loss reaches b only via clipping) no longer holds.
  - **Amended again 2026-11-15** (implementation, not a change to the decision): the clip range is
    **initialized by MSE and re-fit by MSE during QAT instead of learned by gradient.**
    - Init: c_d is the fraction of the per-degree max (grid 0.05…1.0) that minimizes
      Σ(x − Q(x))² over the nodes of degree d at b = 4. Replaces c_d = max|x|.
    - QAT: every 10 epochs (and after the last one) every c_d is re-fit the same way at that
      degree's current round(b), on the current layer input. b is still learned (Eq.(5)).
    - Why: max-based init rounds most heavy-tailed inputs to 0 (PubMed PTQ 73.4 → 79.5 with MSE
      init; FP32 79.1). And with 60–140 training nodes the learned range gets almost no useful
      gradient, drifts upward, and can't follow b when the penalty lowers it, so CiteSeer and
      PubMed collapsed at λ = 1e-7 (57%). With re-fit: PubMed 78.2 at 0.89 bits, CiteSeer 70.3 at
      1.32 bits (3 seeds); both match Tab.VI. Details in `mega_pt/README.md`.
    - Trade-off: none in hardware (one α per degree either way). In optimization, c is no longer
      trained for the task loss directly; it minimizes reconstruction error, which is a proxy.
      Old behaviour: `--x-init max --clip-refit 0`.
    - **Fidelity note 2026-11-16:** the re-fit departs from a stated detail (p.4 §III-A: the
      scale is "learned"). Tested alternatives with the scale learned by gradient
      (`mega_pt/README.md`, "Fidelity check"): LSQ gradient scaling on c does not help (c stays
      near its 4-bit init and cannot follow b). Learning α directly (`--x-param alpha`, no
      re-fit) works: Cora 81.0 @ 2.58 sparse bits (λ 1e-5), CiteSeer 67.8 @ 2.05 (λ 1e-6),
      PubMed 79.7 @ 1.57 (λ 1e-6), with bits that vary by degree.
  - **Amended 2026-11-16 (user decision): α is learned directly by gradient, no re-fit**
    (`--x-param alpha --clip-refit 0`, now the default). The MSE clip init stays.
    - Why: fidelity to MEGA as a baseline. The paper states the scale is learned (Eq.(2),
      p.4 §III-A), and its key idea is per-degree bitwidths. With the re-fit every degree ended
      at b = 2, i.e. uniform bits, which removes that idea; learned α keeps bits that vary by
      degree (2–5 in the hidden layer).
    - Trade-off: on GCN (3 seeds) CiteSeer is 1–3 points below Tab.VI (67.8 vs 70.6) and Cora
      needs more bits for Tab.VI accuracy (2.58 vs 1.70); PubMed beats Tab.VI. Not tuned yet.
      The re-fit stays as an option (`--x-param range --clip-refit 10`) for comparison.
    - The 2026-11-15 note above (task loss reaches b only via clipping under α-direct) holds
      again for the default.

### Q-05 — Gradient through rounding `[PT]` blocking
- Says: Eq.(2) uses ⌊|x|/α + 0.5⌋ (p.5).
- Missing: the gradient estimator for rounding and clipping (STE or other). Gradients w.r.t.
  α (e.g. LSQ-style [13]? cited in related work p.3 but not stated as used).
- Decision: **STE for rounding (gradient 1). Clipped values pass no gradient to x. α gets the gradient
  that autograd produces from `sign(x)·min(round_ste(|x|/α), Q_max)·α`, which equals the LSQ
  gradient without LSQ's gradient scaling.** (2026-11-14)
  - Why: the paper cites LSQ [13]; writing the quantizer this way gives LSQ's gradient for free.
  - Trade-off: no gradient scaling; add it only if α training is unstable.
  - 2026-11-16: LSQ's gradient scaling tested as `--lsq` (off by default). With Adam it cancels
    (per-entry constant factor); with SGD it did not fix the learned clip range. Not adopted.

### Q-06 — Signed vs unsigned integer range `[PT][SIM]` blocking
- Says: Eq.(2) is sign(x)·min(round(|x|/α), 2^(b−1)−1) (p.5). The Encoder applies ReLU before
  packing (p.8 §V-D), so stored activations are ≥ 0.
- Conflict: Fig.9 (p.6) stores values beyond the signed range (2-bit `11` = 3 > 1; 3-bit `110` =
  6 > 3). With Eq.(2), b = 1 gives max level 0, so every value becomes zero. Yet 1-bit nodes
  appear in Fig.9 and the Bitwidth field supports 1 bit.
- Missing: whether post-ReLU activations use an unsigned b-bit range (0..2^b−1). How signed
  values are handled in the bit-serial datapath.
- Decision: **Signed everywhere: X, W and B all use Eq.(2) as written (sign + magnitude levels
  −(2^(b−1)−1)..+(2^(b−1)−1)).** (2026-11-14)
  - Why: one datapath and one Encoder rule for all tensors; Reddit's negative raw inputs need no
    special case; signed bit-serial costs little (sign XOR + conditional negate).
  - Trade-off: post-ReLU hidden features are ≥ 0, so their sign bit is always 0: one bit per
    hidden value is wasted. Fig.9's stored values do not fit the signed range (remains an
    inconsistency). A signed 1-bit value can only be 0, which is consistent with Q-07's 2-bit
    minimum.

### Q-07 — Minimum feature bitwidth: 1 or 2? `[PT][SIM]` affects-fidelity
- Says: "Bitwidth denotes the bitwidth (ranging from 1 to 8bit)" (p.6 §V-B); "X is from 1 to
  8bit" (p.7 §V-C-2); Fig.9 Node4 is 1-bit; **but** "the minimum bitwidth of the node features
  is 2bit in Degree-Aware method" (p.7 §V-C-2). Fig.10(a) shows 2Bit…8Bit selectors only.
- [DERIVED] Tab.VI reports average bits below 2 (Cora GCN 1.70, CiteSeer GCN 1.87; p.10).
  This is impossible if every node is ≥ 2 bits, unless "average bitwidth" counts something
  other than a per-node mean (see Q-15).
- Decision: **Feature bitwidths range over 2–8 bits.** (2026-11-14)
  - Why: follows the p.7 statement and Fig.10(a)'s 2Bit…8Bit selectors; with signed X (Q-06) a
    1-bit value is always 0, so 2 bits is the smallest useful width. One fewer decoder mode.
  - Trade-off: Tab.VI's 1.70 / 1.87 average bits are only reachable under the sparse
    average-bits definition (Q-15). Note: the paper's accuracy numbers may come from software
    that did not fully mirror the hardware, so not every inconsistency must be reconciled.

### Q-08 — Weight quantization details `[PT]` affects-fidelity
- Says: W quantized to 4 bits for all GNNs, per-column learnable β, "integer representation W̄
  as Eq 2" (p.5).
- Missing: whether b = 4 is fixed (not learned), whether β is initialized like α, and whether
  bias terms exist or are quantized.
- Decision: **W: fixed 4-bit signed (levels −7..+7), learnable per-column β initialized to
  max|W_col|/7 from the FP32-pretrained weights. No bias terms.** (2026-11-14)
  - Why: the paper fixes 4 bits; max-based init mirrors Q-04. No bias keeps Eq.(3) a pure
    integer matmul plus scaling.
  - Trade-off: dropping bias may cost a little accuracy versus a standard GCN.

### Q-09 — Quantization of B = XW before aggregation `[PT][SIM]` blocking
- Says: "we quantize B as the quantization way of W" (p.5 §IV). The Combination Engine
  results are "quantized to 4bit" (p.7 §V-D).
- Missing: per-column or per-tensor scales, whether they are learned, where they are applied,
  and the rounding mode. In hardware, which unit requantizes B (C-PE output? Aggregation
  Engine input?).
- Decision: **B = XW is requantized to 4-bit signed (Eq.(2) rounding, round-half-away) with a learnable
  per-column scale γ, initialized max|B_col|/7 from a calibration pass. In hardware this happens
  after the Combination Tiles' partial sums are accumulated and before B is written to the
  Combination Buffer and Sparse Buffer.** (2026-11-14)
  - Why: "quantize B as the quantization way of W" (p.5) and "quantized to 4bit" (p.7).
  - Trade-off: one requantization point; the per-column γ must be stored with the weights.
  - **Amended 2026-11-16 (init only; applies to β of Q-08 too):** γ and β are initialized by MSE
    (the fraction of max|col| that minimizes Σ(t − Q(t))² per column) instead of max/7
    (`--wb-init mse`, default). Still learned by gradient during QAT.
    - Why: on NELL a few large entries per column (hub rows) set max/7 and most entries round to
      0. NELL seed 2 PTQ: 21.0 → 56.0 (FP32 58.1); NELL QAT λ = 0: 54.56 ± 9.01 → 58.93 ± 3.50.
      Neutral elsewhere (3 seeds): Cora 81.67 → 81.87, CiteSeer 70.33 → 69.93, PubMed
      78.97 → 78.93.
    - Trade-off: none in hardware (one scale per column either way). Old: `--wb-init max`.

### Q-10 — First-layer input and last-layer output `[PT][SIM]` blocking
- Says: X in XW is "obtained from the aggregation phase of the previous layer" (p.5). The
  Encoder quantizes aggregated outputs for the next layer (p.8).
- Missing: how raw input features X^(0) are quantized (degree-aware? what bitwidth? are they
  preprocessed offline?). Whether the final layer's output is quantized and whether ReLU is
  applied. Which layers Eq.(4)'s Σ_l covers.
- Decision: **X⁽⁰⁾ is quantized degree-aware and signed, with its own layer-0 table. The last layer's
  output (logits) has no ReLU and no feature quantization and is not counted in Eq.(4). Eq.(4)'s
  Σ_l covers each layer's input: X⁽⁰⁾ (dim F) and X⁽¹⁾ (dim hidden) for a 2-layer model.**
  (2026-11-14)
  - Why: the hardware's first combination consumes Adaptive-Package input; Eq.(4)'s dim^l
    weighting implies the high-dimensional input is counted. Logits go to softmax and are never
    re-packed.
  - Trade-off: X⁽⁰⁾ dominates Eq.(4) (e.g. Cora 1433 vs 128 dims), so the memory penalty mostly
    acts on the input layer.

### Q-11 — Adjacency normalization and edge weights `[PT][SIM]` decided (fold; Reddit cannot test it, see below)
- Says: Eq.(1) uses normalized Ã (p.3). §IV says A ∈ {0,1}^(N×N) (p.5). §V-D: features are
  "multiplied with the corresponding edge weight of A broadcasted to each AU, which is stored
  in the Edge Buffer using the CSC format" (p.8). AUs do "integer addition and
  multiplication" (p.7).
- Missing: whether normalization coefficients are stored as edge weights, their format and
  precision, whether they are quantized, and whether normalization is folded into scales.
- Decision: **Pending test.** (2026-11-14) A is stored binary (A+I); the destination factor d_i^(−1/2) is
  folded into the Encoder's per-degree scale (free, no precision loss). Two candidates for the
  source factor d_j^(−1/2):
  - (a) **Fold**: multiply into the combination rescale (α′_d = α_d/√d) before B is quantized.
    Zero added hardware. Risk: B rows of high-degree sources shrink by 1/√d and may round to
    0 under the per-column γ.
  - (b) **Per-source scaling**: quantize B unscaled, then each AU multiplies B̄_jk by a fixed-point
    w_j = 1/√d_j (unsigned Q1.f, so 1.0 is exact) **once per source row**, keeps the product in a
    (double-buffered) register and reuses it for all of j's out-edges. Uses the multiplier the
    AUs already have ("integer addition and multiplication", p.7), so the only added area is the
    product register. Risk: w precision (f) and 16-bit partial-sum overflow (Q-25).
  - Shared design: a per-layer, degree-indexed w table. GIN-0 and GraphSage load w = 1.0
    (GraphSage's 1/d_i goes to the Encoder), so other GNNs run without per-source scaling on the
    same hardware. The AU multiplier can also run per-edge (no reuse) for future per-edge
    weights (GIN with ε≠0, GAT).
  - Rejected: per-edge fixed-point weights (same precision as (b) for GCN, but a multiply per
    edge, f extra bits per edge in buffers and DRAM).
  - **Decision rule** (ours): choose (a) if its accuracy is within ~0.5 points of FP32 Ã (or
    within Tab.VI's std); otherwise (b) with the smallest f meeting that bound without
    overflowing 16-bit partial sums. Also test on NELL/Reddit, where high-degree nodes are
    common, against our own FP32 baseline.
  - The PyTorch model implements all three modes (`--norm fp32|fold|per_source`).
  - **Interim result (2026-11-14, Cora, 3 seeds; table in `mega_pt/README.md`):** fold 80.93 ± 0.76
    vs ideal-Ã 80.27 ± 0.50, so fold meets the rule on Cora. per_source f=8 overflows 16-bit
    partial sums (max 49,822); f=4 fits (3,104). Not closed: NELL/Reddit still to test.
  - **Decision (2026-11-15): fold**, by the rule above. Reconfirm on Reddit (not run).
    - NELL result (3 seeds, λ = 0; table in `mega_pt/README.md`): fold 55.87 ± 9.87 vs ideal Ã
      54.56 ± 5.05 vs per_source f=4 38.53 ± 1.29 (FP32 model: 60.85 ± 2.40).
    - per_source is ruled out on NELL. Q1.4 rounds w to 0 for degree > 1,024: 21 hubs, which
      source 12% of the edges. f ≥ 6 fixes that, but [DERIVED estimate] multiplies the measured
      f=4 L0 partial sum (29,897) by about 4, past the 16-bit accumulator (Q-25).
    - Not closed:
      - Reddit is untested. It needs GraphSage (Q-12), and its download host is blocked here.
        **Update 2026-11-17:** Reddit now loads (DGL's S3 bucket). Tab.VI's Reddit model is
        GraphSage, whose normalization has no source factor (s = 1, t = 1/d in the Encoder),
        so fold and per_source are identical there. Reddit cannot reconfirm Q-11. It does show
        the hubs are harmless under sampling: max |psum| 182 (`mega_pt/README.md`, 2026-11-17).
      - NELL QAT itself is noisy (one seed collapses in every mode).
    - The per-source w table and AU-reuse design above stay documented as the fallback.

### Q-12 — Model-specific operators (GIN, GraphSage) `[PT][SIM]` blocking
- Says: GCN/GIN/GraphSage "can be uniformly represented" by Eq.(1) (p.3). Tab.III: GIN uses Add
  aggregation, GraphSage uses Mean aggregation with 25 sampled neighbors, 256 hidden units (p.9).
- Missing: GIN's (1+ε) self term and multi-layer MLP; GraphSage's concat/separate self weight,
  sampling scheme, and mean in hardware; how degree is defined under sampling; whether
  quantization applies inside the MLP.
- Decision: **GIN: GIN-0 (ε = 0, A+I), one linear layer per GNN layer, sum aggregation. GraphSage: mean
  over sampled neighbors plus self, D⁻¹(A+I), single W, 25 samples, sampling also at inference
  with a fixed seed.** (2026-11-14) Deferred until GCN works.
  - Why: fits Eq.(1)'s single-W form and one combination pass per layer.
  - Trade-off: differs from the original GIN (MLP, learnable ε) and GraphSage (separate self
    weight, concat); accuracy may differ from those models.
  - **Implemented 2026-11-16** (`mega_pt/model.py` GraphOps, `--model gin|sage`). Choices the
    paper does not state, made here (ours):
    - One layer form for all three: Ã = diag(t)·(A+I)·diag(s). GIN: s = t = 1. GraphSage:
      s = 1, t = 1/d. So under the decided `fold` mode (Q-11) GIN and GraphSage need no
      source scaling, and GraphSage's 1/d sits in the Encoder's per-degree scale, as
      the Q-11 shared-design note planned.
    - Sampling: uniform, without replacement, at most 25 in-neighbors per node, plus the
      self-loop, which is always kept. One sampled graph is shared by both layers, because
      the hardware stores one CSC graph. Training draws a new sample every epoch (standard
      GraphSage). Evaluation, calibration and metrics use one fixed sample seeded by the run
      seed.
    - Degree under sampling: the **sampled** in-degree incl. self-loop (≤ 26). The hardware
      only sees the stored, sampled graph, so that is the degree its tables can be indexed
      by. Degree tables therefore have 27 entries for GraphSage.
    - Hidden size from Tab.III: 128 for GIN, 256 for GraphSage. The training recipe is the
      same as GCN (Q-14).

### Q-13 — Loss hyperparameters `[PT]` blocking
- Says: Eq.(4)–(5), η = 8·1024 (p.5). λ is tuned to trade accuracy for size; e.g. lower λ gives
  CiteSeer GCN 71.2% at 2.24 bits (p.10 §VI-B).
- Missing: λ and M_target for every Tab.VI entry, and whether M_target is in KB
  ([DERIVED] from η).
- Decision: **M_target is in KB. Eq.(4) stays dense as printed; M_target and λ are tuned per Tab.VI row
  until the *sparse* average bits (Q-15) hits the Tab.VI value.** (2026-11-14)
  - Why: keeps Eq.(4) exactly as printed while matching the reported metric.
  - Trade-off: the penalty and the reported metric measure different things, so M_target can't
    be computed directly from Tab.VI; it needs a sweep. Scale note: Cora's dense memory is
    ≈ 516·b̄ KB, so a 50 KB miss costs 2500·λ; λ around 1e-5–1e-3 is the starting range.
  - **Sweep result 2026-11-15** (`mega_pt/README.md`, results log):
    - **How to set it.** Put M_target at the 2-bit dense floor (`--target-dense-bits 2.0`) and
      use λ as the only knob. A "reachable" target with a large λ doesn't work as a constraint:
      Eq.(4) uses continuous b, and layer 0's continuous b settles just under 2.5 (it rounds to 2)
      with 11× the hidden layer's dimension, which uses up the hidden layer's budget.
    - **Cora.** λ = 1e-7 gives 81.00 ± 1.14 at 1.60 sparse bits, which meets Tab.VI (80.9 ± 0.6
      at 1.70). The useful λ range is much lower than the scale note above suggested: 1e-5
      already costs 4 points. Adam turns any persistent penalty gradient into a ~lr step per
      epoch, so λ mostly sets *when* the task gradient wins, not how hard the penalty pulls.
    - **CiteSeer and PubMed** lost about 3 points even at λ = 0 and collapsed at λ = 1e-7; the
      ablation pointed at the degree-aware X quantizer. **Resolved 2026-11-15** by the Q-04
      MSE clip init + re-fit: λ = 1e-7 now matches Tab.VI on all three (10-seed numbers in
      `mega_pt/README.md`). The same λ = 1e-7 works for every dataset, so per-dataset tuning was
      not needed.

### Q-14 — Training setup `[PT]` blocking
- Says: Fig.3 averages over 100 runs (p.3). Tab.VI reports mean ± std (p.10). Training on one
  RTX 3090, 2.04× FP32 time (p.12 §VII-1).
- Missing: optimizer, learning rate, epochs, dropout, data splits (standard Planetoid? random?),
  number of runs behind the ± values, whether training starts from a pretrained FP32 model,
  and whether training is transductive or inductive for Reddit. No code release is mentioned.
- Decision: **Standard Planetoid public splits (Cora: 140/500/1000). Adam, lr 0.01, weight decay 5e-4,
  dropout 0.5, 200 epochs, hidden 128. FP32 pretraining, then quantization-aware fine-tuning
  initialized from it. 10 seeds for mean ± std. Reddit (later): the standard GraphSAGE inductive
  split.** (2026-11-14)
  - **Reddit implemented 2026-11-17:** standard split (153,431 / 23,831 / 55,703), inductive:
    training sees only the train-induced subgraph, and evaluation uses the full graph. The
    memory term counts the evaluation graph's degrees, which are what the hardware stores.
    Features `--feat-norm none` (already standardized). Result: 93.58 ± 0.02 @ 0.66 bits
    (3 seeds, λ = 1e-9) vs Tab.VI 95.6 ± 0.2 @ 2.74; table in `mega_pt/README.md`.
  - Why: [INTERP] the paper's FP32 GCN Cora 81.5% equals the standard-split result; Kipf & Welling
    hyperparameters are the common default. Pretraining provides the FP32 column and α/β init.
  - Trade-off: the paper's 2.04× training time suggests a recipe we can't verify.
  - **Amended 2026-11-14 after first runs** (details in `mega_pt/README.md`): raw input
    features instead of row normalization (row normalization made many layer-0 values round to 0;
    QAT 75.9 vs 79.7 on Cora). QAT uses lr 1e-3 for weights and for log α, 1e-2 for b and for
    log β/γ (α at 1e-2 drifted up and the training loss rose). Scales are learned in log space.
  - **Amended 2026-11-15: back to row normalization for every dataset.** With the Q-04 MSE clip
    init + re-fit, row normalization no longer hurts Cora (λ = 1e-7: row 81.77 vs raw 81.00,
    3 seeds) and is needed on CiteSeer (below). One preprocessing for all datasets.
  - **Caveat 2026-11-15 (superseded by the amendment above):** the raw-feature choice was measured on Cora only, and before the
    Q-04 clip-range fix. FP32 on CiteSeer is 68.8 (raw) vs 71.4 (row-normalized; Tab.VI 71.1),
    so the choice should be revisited per dataset. NELL needs row normalization plus
    dropout 0.1, wd 1e-5 and 400 FP32 epochs to reach 62.5% FP32 (raw counts with our default
    recipe: 57.2%; Kipf & Welling report 66.0%). Tab.VI has no NELL accuracy to compare with.

### Q-15 — Compression-ratio definition `[PT]` affects-fidelity
- Says: "take the ratio of the average bitwidth / 32 as the theoretical compression ratio"
  (p.9 §VI-A-2).
- [DERIVED] Reported CRs are ≈ 32 / avg-bits (the inverse), but not exactly: Cora GCN 32/1.70 =
  18.82 vs 18.6× reported; Cora GIN 32/2.37 = 13.50 vs 13.4×; CiteSeer GCN 32/1.87 = 17.11 vs
  17.0× (p.10 Tab.VI). Fig.22 (p.12) labels the top Cora point as CR **18.8**.
- Missing: how "average bitwidth" is computed (per-node mean? weighted by dim^l? over layers?
  includes zeros/sparsity?).
- Decision: **Average bits = sparse average over hidden features only:
  Σ_i (non-zero elements of X̄_i × b_i) / Σ_i dim, over the hidden-layer inputs (X⁽¹⁾…),
  excluding X⁽⁰⁾. CR = 32 / average bits. The PyTorch model also logs the dense
  dimension-weighted average over all layers for comparison.** (2026-11-14)
  - Why: with a 2-bit minimum (Q-07), no dense definition can reach Tab.VI's 1.70; the sparse
    hidden-only definition can (e.g. Fig.5 Cora GCN density 44% × ~3.9 bits ≈ 1.70) [INTERP].
    Replaces our earlier dense definition.
  - Trade-off: differs from Eq.(4)'s dense count (see Q-13).

### Q-16 — Where scales are applied in hardware `[SIM]` affects-fidelity
- Says: Eq.(3) needs (X̄W̄) ⊙ (s_X ⊗ s_W) (p.5). QN units use "scales from the Weight Buffer"
  (p.8).
- Missing: the unit that multiplies by α_i β_j, the scale format (FP? fixed-point?), and
  whether rescale and requantize are merged into one multiplier.
- Decision: **Scales are FP32 in training and fixed point in hardware. There are two requantization
  points, and each multiplies an integer by a degree-table factor and a column factor.**
  (2026-11-20; the user chose fixed point and offline c_j, and we chose the format.)
  - **Combination rescale:** B̄_ij = clip(round(acc_ij · α′_{d_i} · c_j), ±7).
    - acc = X̄_i·W̄_{:,j} is the integer from the Tile Accumulator (Q-22).
    - α′_d = α_d·s_d: the GCN fold source factor d^(−1/2) (Q-11) is folded into the degree
      table. It is 1 for GIN and GraphSage.
    - c_j = β_j/γ_j is computed offline (user).
    - Unit: 32 lanes, one per C-PE output column, after the Tile Accumulator, with two
      multipliers per lane.
  - **Encoder:** X̄′_ij = clip(round(ReLU(psum_ij) · e_{d_i} · γ_j), 0..qmax_{d_i}).
    - e_d = t_d / α^(l+1)_d: GCN t = d^(−1/2), GraphSage t = 1/d, GIN t = 1.
  - **Last layer:** logits = psum·γ_j, written out with no ReLU and no quantization (Q-10).
    t_i is dropped because a positive per-row factor does not change the argmax.
  - **Format (ours, measured):** 16-bit unsigned values with one binary point per table, set by
    a per-table shift register. Rounding is round-half-away (Eq.(2)), applied once after both
    multiplies, which run at full width.
    - Measurement: the integer-only reference in `mega_pt/hw_stats.py` (`int_forward`)
      reproduces the fake-quant model's predictions (agreement ≥ 0.9999 on 6 models).
    - With 16-bit tables it is exact on all of them; 12 bits stays within 0.6 points; 8 bits
      loses up to 0.8.
    - Each table spans ≤ 6.6 octaves, so a shared binary point still leaves ≥ 9 significant
      bits for the smallest entry.
    - A per-entry mantissa + shift ("dyadic") was no better, so it is not used.
    - Numbers: `mega_pt/README.md`, 2026-11-20.
  - **Fidelity:** this fills a gap; the paper says only "scales from the Weight Buffer" (p.8).
    The combination rescale unit is not listed in Tab.IV, so we count it as ours.

## B. Hardware — storage format

### Q-17 — Do package lengths include the header? `[SIM]` affects-fidelity
- Says: (short, medium, long) = (64, 128, 192) bits (p.6 §V-B). Header = Mode 2b + Bitwidth 3b.
- [DERIVED] In Fig.9(d), the length labels (8/20/4 bits) cover Val Array + padding only.
- Missing: explicit statement for the 64/128/192 values.
- Decision: **64/128/192 are total package lengths, including the 5-bit header**, so the Val
  Array holds at most 59/123/187 bits. (2026-11-20, user.)
  - Why: every package is then a multiple of 8 bytes, so packages stored back to back stay
    byte-aligned (Q-19).
  - Trade-off: Fig.9(d)'s labels count the Val Array only; we treat the figure as schematic.
    This fills a gap, since the text does not state it.

### Q-18 — Mode selection rule and encoding `[SIM]` affects-fidelity
- Says: Mode 00/01/10 = short/medium/long (p.6). Selected "based on the sparsity and
  quantization bitwidth of each node" (p.6); Encoder selects "according to the bit length of the
  values in the package register" (p.8).
- Conflict: the text says Package1 uses "the short mode", but Fig.9(d) Package1's Mode is `01`
  (medium) (p.6).
- Missing: the exact rule (smallest class that fits?), the meaning of code `11`, and whether a
  package can close early to pick a shorter mode.
- Decision: **Greedy packing as stated on p.8, and Mode = the smallest length that fits.**
  (2026-11-20, user.)
  - The package register takes successive non-zeros until either the next value would exceed
    187 bits or the next node's bitwidth differs.
  - Mode is then the smallest of 64/128/192 total that holds 5 + k·b bits, and the rest is
    zero-padded.
  - Code `11` is unused (reserved).
  - The Package1 text/figure conflict is moot: the rule decides.
  - A full package holds 93, 62, 46, 37, 31, 26 and 23 values at b = 2…8.

### Q-19 — Addressing nodes, bitmaps, and spanning `[SIM]` blocking
- Says: per-node bitmaps are stored separately from packages (p.6). Packages hold successive
  nodes (p.6).
- [DERIVED] A node can exceed one package (e.g. 128 non-zeros × 4 bit = 512 bits > 192), so
  spanning must exist.
- Missing: the per-node offset/pointer scheme, how the Decoder finds where node i starts, the
  bitmap memory layout (and whether bitmaps count in the CR/memory numbers), and DRAM
  alignment.
- Decision: (2026-11-20; the user chose the bitmap + counter, contiguous byte-aligned packages and
  bitmaps in the memory numbers; we chose the layout details.)
  - **Packages:** one contiguous byte stream per layer. All lengths are multiples of 8 bytes.
  - **Bitmaps:** a separate contiguous array, F bits per node, padded to a multiple of 32 bits
    (ours: whole words for the decoder; at most 31 bits wasted per node).
  - **Finding node i:** nodes are read in ID order.
    - The Decoder takes popcount(bitmap_i) values and keeps a counter of the values left in
      the current package.
    - So a node can span packages, and a package can hold several nodes.
    - There are no per-node pointers. Ours: one start pointer (package byte offset, value
      index) per subgraph, used to restart a stream or prefetch.
  - **Memory / CR (user):** count packages (headers and padding included) and bitmaps.
    - Report the *stored* bits per element and CR_stored = 32/that, next to the
      Tab.VI-comparable values-only numbers (Q-15).
  - **Consequence [DERIVED]:** the bitmap costs 1 bit per feature, which dominates sparse inputs.
    - Cora X⁽⁰⁾: 1,440 bitmap bits vs about 36 value bits per node.
    - NELL X⁽⁰⁾: 7.5 KB of bitmap per node, 482 MB in total.
    - So X⁽⁰⁾'s stored CR is ≤ 32 by construction. This is MEGA's stated format (bitmaps per
      node, p.6), so we report it and do not fix it.
  - **Amended 2026-11-23 (DRAM only; `docs/simulator/00-design.md` S-15).** In DRAM,
    X̄⁽⁰⁾'s positions take the smaller of the bitmaps and 16-bit column indices: indices for
    Cora/CiteSeer/NELL, bitmaps for PubMed and Amazon-Ratings.
    - Why: the paper's NELL numbers (Condense-Edge 11.6×, 23.5× less DRAM than SGCN) rule out
      482 MB of input bitmaps in DRAM.
    - Unchanged: the bitmaps on chip, the combination's bitmap scan, hidden layers, and the
      memory/CR numbers above. `--x0-format bitmap` gives the original accounting.
    - Needs an index → bitmap expander in front of the Input Buffer, which MEGA does not
      describe (S-15). Per node, DRAM holds a 16-bit count + 16-bit indices.

### Q-20 — Fig.9(b) Node1 bitmap `minor`
- [DERIVED] Node1 = 01 10 00 11 00 → non-zero pattern 11010, but Fig.9(b) shows 11011 (p.6). The
  other three rows match. Likely a figure error. Doesn't affect implementation.
- Decision: **Figure error; ignore.** (2026-11-20, user.)

## C. Hardware — compute engines

### Q-21 — Values of m, n, c and the meaning of "4 × 8 × 32" `[SIM]` blocking
- Says: Tab.IV BSEs "4 × 8 × 32", Crossbar "32 × 8(64bit)", Encoder "32 QN units" (= #m) (p.10).
  Crossbar = c × n(m/2 × 4 bits) (p.7). 4 Combination Tiles (Fig.8, p.5).
- [DERIVED] A consistent reading is 4 tiles, n = 8, m = 32, c = 32 (see 03 §4.4). Not stated
  explicitly.
- Missing: confirmation, and how F (e.g. 1433, 61278) is split into c-wide slices across tiles
  and passes.
- Decision: **4 tiles, n = 8, m = 32, c = 32** (the [DERIVED] reading). (2026-11-20, user.)
  - F > 128: several passes per node. Under Q-23's non-zero feeding a pass is the node's next
    ≤ 128 **non-zeros** (≤ 32 per tile), not 128 feature positions. c = 32 is the tile's
    Weight Reg / crossbar input count.
  - Confirmed 2026-11-20 with m = 32 C-PEs × n = 8 BSEs per tile. The user's first reading was
    8 C-PEs × 32 BSEs; that would give m = 8, contradicting the crossbar's m/2 × 4 = 64 bits
    and the Encoder's "#m = 32 QN units".
  - Output columns (ours): m = 32 C-PEs produce 32 columns of B, so F2 takes F2/32 column
    blocks (128 → 4, 256 → 8, ≤ 32 classes → 1).
  - For each column block, the node's bit-planes are replayed from the Bit FIFO with that
    block's weights.

### Q-22 — Bit-serial details `[SIM]` affects-fidelity
- Says: (feature bits, shift bit) pairs; example (0101,0), (1011,1) (p.7).
  [DERIVED] LSB-plane first.
- Missing: signed handling (two's complement MSB weight? sign-magnitude?), Shifter-Acc width,
  accumulator precision before requantization, and where the 4 tiles' partial sums are added.
- Decision: (2026-11-20; the user asked for simple sign handling, and we chose the widths.)
  - **Sign:** values are stored as b-bit two's complement (Q-06's levels fit).
    - The Decoder emits bit-planes LSB first, and on the MSB plane the Shifter-Acc subtracts
      (two's-complement weight −2^(b−1)).
    - W̄ is 4-bit two's complement. A BSE ANDs the feature bit with the weight, which selects
      the weight or 0.
  - **Adder tree:** 8 × [−7, 7] gives [−56, 56], 7-bit signed.
  - **Shifter-Acc:** one tile pass is ≤ 32 non-zeros at ≤ 8 bits.
    - Worst case 32·7·127 = 28,448, so 16-bit signed can never overflow. Measured max 878.
  - **Tile Accumulator** (ours, the separate component the user asked for):
    - Per output lane: a 4-input adder of the tiles' 16-bit results, then a 32-bit
      accumulator across passes.
    - Worst case F·7·127; for NELL (F = 61,278) that is 5.4e7 < 2³¹, so it never overflows on
      any of our datasets. Measured running max 2,735.
    - Its output goes to the Rescale unit (Q-16).
  - Why worst-case widths rather than measured ones: the extra width is cheap, and the
    simulator then needs no overflow model for combination.

### Q-23 — Weight index generation and crossbar control `[SIM]` affects-fidelity
- Says: Parallel Sum Unit converts the bitindex to W row indices (p.7, Fig.10(b)).
- Missing: how many bitmap bits are processed per cycle, how n non-zeros are grouped per C-PE
  pass, and the tail case when the number of non-zeros isn't a multiple of n.
- Decision: **Tail BSEs are disabled** (user). **Non-zero feeding with a 256-bit/cycle bitmap scan**
  (proposed by us, accepted by the user 2026-11-20).
  - **Decoders (ours):** one Decoder per tile, as Fig.8 shows.
    - The node's bitmap and package stream are broadcast to all four.
    - Each Decoder computes the same 256-bit prefix sums (the global rank of every non-zero)
      and keeps only the ranks in its tile's share of the pass.
    - Cost: four prefix-sum units instead of one shared dispatcher. Chosen because the paper
      states a Decoder per tile.
  - The two options measured (`mega_pt/hw_stats.py`, cycles per node per 32-column block,
    BSE utilization):
    - **slice:** tile t takes a fixed 32-feature slice. A pass costs max over tiles of
      waves·b, and empty passes still cost 1 cycle.
    - **nnz:** a pass takes the node's next ≤ 128 non-zeros, at any feature positions, split
      evenly over the 4 tiles. The Weight Index Generator's indices fetch those W rows (Fig.10(b):
      "indices drive weight loading").
  - Hidden layers (55–70% dense): slice 75–80% utilization, nnz 82–92%.
  - X⁽⁰⁾ of bag-of-words sets (1–7% dense):
    - slice: 4–26% (Cora 20.7 cycles/node, CiteSeer 47.8).
    - nnz: limited by the bitmap scan. Cora 12.0 at 128 bits/cycle and 6.0 at 256;
      CiteSeer 29.0 and 15.0.
  - **Proposal: nnz feeding with a 256-bit/cycle scan** (one Parallel Sum Unit over 256
    bitmap bits, pipelined with compute).
    - 256 = the largest hidden width (Tab.III, GraphSage), so hidden layers are never
      scan-bound; only sparse X⁽⁰⁾ is. A wider scan helps only X⁽⁰⁾.
  - Fidelity: the paper says "each tile is responsible for processing a slice of features in a
    node" (p.6). nnz reads "slice" as a slice of the node's non-zero features. This differs from
    the 32-feature-wide slices first assumed in Q-21; the user accepted it.

### Q-24 — C-PE systolic timing `[SIM]` blocking
- Says: C-PEs are split into two groups; features flow left → right; Fig.11 shows m=2, n=2 over
  4 cycles (p.7).
- Missing: the general schedule for m = 32 (16 per group?), pipeline fill/drain, and cycles per
  row as a function of (non-zeros, bitwidth, F2).
- Decision: **Ours, accepted by the user 2026-11-20.**
  - **Terms:** a tile's BSEs form an 8 × 32 array: n = 8 BSE rows (non-zeros) × m = 32
    C-PEs (output columns).
    - A **wave** is ≤ 8 non-zeros of the node, one per BSE row. All 32 C-PEs process the same
      wave, each against its own weight column, so one wave yields 8 products per output
      column per bit-plane.
    - The paper's two "groups" are the two halves of the C-PEs (16 + 16), not sets of
      features.
  - **Weight loading:**
    - The crossbar outputs n × (m/2 × 4 bits) = 8 rows × 16 weights per cycle: one wave's
      weights for half of the C-PEs.
    - Cycle t: the left half (C-PEs 1–16) gets the wave's weights and its first bit-plane.
    - Cycle t+1: the right half gets the weights, and the *same* bit-plane is forwarded to it
      from the left half (Fig.11). Meanwhile the left half reuses its weights on the next
      bit-plane.
    - So both halves process the same features, one cycle apart. Each C-PE keeps its weights
      for b cycles.
    - With b ≥ 2, the crossbar alternates halves and is never the bottleneck. That is the
      paper's stated reason for the 2-bit minimum halving weight bandwidth (p.7).
  - **Steady state:** one wave per b cycles per tile per 32-column block. A 32-non-zero tile
    share is 4 waves.
  - **Node cost per column block:** Σ over passes of max over tiles of (waves·b), bounded
    below by the bitmap scan (Q-23).
  - **Fill/drain:** 1 cycle (crossbar) + 1 (half skew) + 1 (adder tree) + 1 (Shifter-Acc).
    Paid once per sweep if nodes stream back to back.

### Q-25 — Aggregation Unit operation `[SIM]` blocking
- Says: 256 AUs, outer product, CSC, 16-bit partial sums, free AUs can serve other nodes
  (p.8, Tab.IV).
- Missing: the AU-to-feature mapping when F2 = 128 or 256 vs 256 AUs, 16-bit overflow handling,
  mean (division) for GraphSage, the edge-weight broadcast rate, and Aggregation Buffer
  capacity management (128 KB) for large subgraphs.
- Decision: **Ours, accepted by the user 2026-11-20.**
  - **Mapping:** outer product over the source's CSC column. Each cycle processes
    k = ⌊256/F2′⌋ edges of the current source, where F2′ is F2 rounded up to a power of 2:
    - 128 → 2 edges/cycle;
    - 256 → 1;
    - 7 classes → 32.
  - **A is binary under fold (Q-11).** AUs add B̄ into the destination's 16-bit partial sum.
    The multiplier is unused but kept for the per_source fallback. CSC stores indices only.
  - **Overflow:** saturating 16-bit adds. The simulator counts saturations. Measured max 479.
  - **GraphSage mean:** 1/d sits in the Encoder factor e_d (Q-12, Q-16), so there is no divider.
  - **Capacity:** the Aggregation Buffer is 128 KB in ping-pong, so 64 KB is active.
    - Destination capacity = 64 KB / (2 B · F2): 256 nodes at F2 = 128, 128 at 256.
    - This sets the subgraph size (Q-27).
  - **Completion:** a subgraph's destinations finish together, after its internal and
    external sources have been processed. The Encoder then drains them in node order.

### Q-26 — Encoder parameter lookup `[SIM]` affects-fidelity
- Says: QN units quantize using scales from the Weight Buffer, apply ReLU, make the bitindex,
  and do greedy packing (p.8).
- Missing: how a node's degree → (α, b) is looked up (is the degree stored per node? where?)
  and the throughput of packing.
- Decision: (2026-11-20; the user chose degree in metadata and a degree-indexed table in the
  Weight Buffer; we chose the details.)
  - **Per-node metadata:** a 16-bit index into a *compacted* per-layer degree table that lists
    only the degrees present (Cora 37 entries, NELL 281, sampled GraphSage ≤ 26).
    - An index rather than the raw degree avoids a lookup CAM.
    - It is streamed with the node's data from DRAM.
  - **Degree-table entry:** b (3 bits), α′_d (16 bits) and e_d (16 bits), plus a per-table
    shift (Q-16).
  - **Column tables:** c_j and γ_j, 16 bits each.
  - **Size:** all tables fit easily in the Weight Buffer; Cora's are about 1 KB in total.
  - **Packing (user):** the 32 QN units handle 32 features per cycle: quantize, ReLU, produce
    32 bitmap bits, and append the non-zeros (≤ 32·b bits) to the package register.
    - F2 = 128 takes 4 cycles per node.
    - A package is emitted when full or when the bitwidth changes (Q-18).

## D. Hardware — scheduling and memory system

### Q-27 — Condense-Edge capacity and corner cases `[SIM]` blocking
- Says: eID FIFO has 8 entries (p.9). 16 ID FIFOs (Tab.IV). eIDs stored in the Edge Buffer
  (p.8). Sparse Buffer regions spill to DRAM when full (Alg.1 L16).
- Missing: FIFO refill when a subgraph has more than 8 eIDs, behavior when SubNum > 16, region
  sizes, the number of METIS partitions per dataset, and the reload order when reading spilled
  regions back. Alg.1 L4 (partition inside the per-node routine) is presentational.
- [DERIVED] Head-only matching needs nodes to arrive in ascending ID order (see Q-28).
- Decision: (2026-11-20; the user and we decided the parts below as marked.)
  - **Order:** FIFO refill keeps ascending order (user). Ours: when a FIFO drops to 4 entries,
    load the next 4 eIDs from the Edge Buffer.
  - **Regions:** equal sized (user): 32 KB / 16 = 2 KB, which is 32 nodes at F2 = 128 (4-bit).
    - A full region spills to DRAM, so subgraph k's spilled rows are contiguous there.
    - They are read back in order when k is aggregated.
  - **Partitioning:** Alg.1 L4 is presentational; METIS runs offline (user).
  - **METIS partitions (ours):** SubNum = ⌈N / capacity⌉ with the capacity from Q-25:
    Cora 11, CiteSeer 13, PubMed 78, NELL 257, Reddit about 1,821, Roman-Empire 89,
    Amazon-Ratings 96, Paris 446.
    - Nodes are relabeled so each subgraph is a contiguous ID range. Ascending-order matching
      and contiguous internal reads need this.
  - **Batching when SubNum > 16: B2 with batch-minimum skipping** (ours, accepted by the user
    2026-11-20).
    - **Batches:** subgraphs are grouped 16 per batch.
    - **Per-subgraph metadata** in the Edge Buffer, 12 B: head eID, read pointer into its
      sorted eID list, and write pointer to its Sparse area in DRAM.
    - **Per-batch register:** the smallest head eID among its 16 subgraphs.
    - **Per combined node nID:**
      - compare nID with the batch minimums (16 per cycle);
      - only a batch whose minimum equals nID is loaded into the 16 FIFO heads/comparators,
        matched (Alg.1 L7–L12), refilled and written back;
      - its minimum is then recomputed (16-way min).
      - Batch 0 can stay resident when SubNum ≤ 16, which is plain Alg.1.
    - **Regions:** the 16 Sparse Buffer regions (2 KB each) are write-combining buffers.
      Region r serves subgraphs r, r+16, … and flushes to that subgraph's contiguous DRAM area
      when it fills or its subgraph changes. Each subgraph's external rows are still read back
      as one contiguous stream.
    - **Why skip instead of visiting every batch (measured, METIS via pymetis, ⌈N/256⌉·1.05
      parts):** external (subgraph, node) pairs are few, at 0.01–1.14 per node.

      | Dataset | Parts | Batches | External pairs per node | Batches with a match, per node |
      |---|---|---|---|---|
      | Cora | 12 | 1 | 0.36 | 0.27 |
      | CiteSeer | 14 | 1 | 0.11 | 0.10 |
      | PubMed | 81 | 6 | 0.89 | 0.57 |
      | Amazon-Ratings | 101 | 7 | 0.21 | 0.18 |
      | Roman-Empire | 93 | 6 | 0.01 | 0.01 |
      | Paris | 469 | 30 | 0.15 | 0.14 |
      | NELL | 270 | 17 | 1.14 | 0.88 |

      - Visiting every batch for every node would move ⌈SubNum/16⌉ × 128 B of head metadata
        (read + write) per node. On PubMed that is 768 B, against 57 B of external rows per
        node: 13× more bytes than the data it reorders.
      - With skipping, metadata moves only for the 0.01–0.88 batches per node that match,
        about 2 B of on-chip SRAM per byte of row data the Condense step places. The
        comparison is about 1 cycle per 16 batches (8 cycles for Reddit's ~114).
    - **Whether Condense-Edge itself pays off** against fetching external rows directly
      depends on the DRAM model: row-buffer locality, and access granularity vs the 64 B row.
      It is MEGA's mechanism, so the baseline keeps it, and the simulator reports its traffic.
    - Not measured: Reddit (sampled graph)
  - **Simulator (2026-11-21, S-08):** one partition serves all layers, so the capacity comes from
    the widest aggregated layer. NELL's 186-class output gives 176 destinations, hence 383
    subgraphs (not 257); Sage-Cora (F2 = 256) gets 22. METIS runs with its default imbalance:
    a tight one (ufactor 1) almost triples Cora's external rows (0.98 vs 0.35 per node).

### Q-28 — Global schedule / phase fusion `[SIM]` blocking
- Says: subgraphs are aggregated one after another. Combined nodes go to aggregation and also
  to the Combination/Sparse Buffers. Encoded outputs "can be forwarded to the Combination
  Engine to fuse the two phases". Buffers are ping-pong (p.6 §V-A, p.8 §V-E).
- Missing: the order in which nodes are combined (all nodes before/while aggregating subgraph
  1?), how combination of later subgraphs interleaves, the granularity of layer fusion, and
  the Controller's policy.
- Decision: **Ours, accepted by the user 2026-11-20.** Nodes are combined in node order (user), and the
  Controller runs a static sweep schedule computed offline.
  - **Sweep 0:** combine layer 0 for all nodes in ID (= subgraph) order. Condense fills the
    Sparse regions.
  - **Sweep l (1 … L−1):** for each subgraph k:
    - aggregate layer l−1, with internal sources from the Combination Buffer/DRAM and external
      ones from k's Sparse region;
    - the Encoder then feeds layer l's combination of k's nodes node by node, which is the
      fusion of p.6;
    - Condense runs for layer l.
  - **Final sweep:** aggregate layer L−1 and write the logits.
  - **Overlap:** ping-pong buffers overlap subgraph k's combination with subgraph k+1's
    aggregation. Granularity is the node inside a subgraph, and the subgraph for overlap.
  - Why aggregation of layer l cannot fuse with combination of the *same* layer: subgraph k's
    external sources may be combined anywhere in the sweep.

### Q-29 — DRAM model details `[SIM]` affects-fidelity
- Says: HBM1.0, 256 GB/s, Ramulator, DRAM energy "following the methodology in [56]"
  (p.9 §VI-A-3). The 128B access granularity is used as an example (p.8).
- Missing: the Ramulator config (channels, timing), the address mapping of packages, bitmaps,
  edges, and Sparse Buffer regions, and the per-access energy values.
- Decision: **Our own choice, applied identically to MEGA and to our design** (2026-11-20, user).
  **Ours, accepted by the user 2026-11-20:**
  - Ramulator 2.0 HBM: 1 stack, 8 channels × 128 bit, 256 GB/s, FR-FCFS.
  - Each data stream (packages, bitmaps, metadata, W̄, CSC, eIDs, Sparse regions, B rows,
    outputs) gets its own contiguous address region.
  - Energy: a fixed pJ/bit per access, taken from one published HBM figure (to be chosen and
    cited).
  - **Implemented 2026-11-22 (M3, `docs/simulator/00-design.md` S-13/S-14):**
    - Ramulator 2.1 with HBM1 at 2 Gb/s per pin. 1 Gb/s would give only 128 GB/s per stack.
    - Address mapping (ours): 64 B lines interleaved over the channels, then over the bank
      groups. A sequential stream reaches 92.6 % of peak.
    - DRAM energy = **7 pJ/bit**, HyGCN's HBM1.0 figure. MEGA's [56] is HyGCN, so this is the
      figure MEGA's stated methodology gives.

### Q-30 — Simulator availability `[SIM]` affects-fidelity
- Says: "We implement cycle-accurate simulators ... validated with the HDL design at the cycle
  level" (p.9).
- Missing: no code or artifact release is stated.
- Decision: **None is released; we build our own simulator** (user, 2026-11-20).
  - Functional check: the simulator's integer outputs must match the integer-only reference
    `mega_pt/hw_stats.int_forward` bit for bit.
  - **Status 2026-11-20:** the functional model (`megasim/`, design in
    `docs/simulator/00-design.md`) matches it bit for bit on all 8 saved models.

### Q-35 — W̄ larger than the Weight Buffer `[SIM]` blocking
- Says: Weight Buffer 48 KB (Tab.IV). Weights are "prefetched from DRAM into ... Weight Buffer"
  (p.6).
- [DERIVED] At 4 bits, W⁽⁰⁾ is larger than the buffer for Cora (90 KB), CiteSeer (231 KB), NELL
  (3.8 MB) and Reddit-GraphSage (75 KB). It fits for PubMed, the extra datasets and every
  hidden layer.
- Missing: the dataflow when W does not fit.
- Decision: **The Weight Buffer acts as a row cache of W̄** (proposed by us, accepted by the user
  2026-11-20).
  - Nodes are the outer loop.
  - Rows a node needs are fetched from DRAM by index; the indices are known early from the
    bitmap, so they can be prefetched.
  - When W fits, it simply stays resident. The simulator measures the traffic.
  - Alternatives:
    - W row blocks as the outer loop: 32-bit partial B rows for every node would go to DRAM
      between blocks.
    - Node blocks: the same as the row cache, only coarser.

### Q-36 — Residual connections `[PT][SIM]` beyond MEGA
- Says: nothing. MEGA's models are 2-layer, without residuals (Tab.III).
- Decision: **Ours, user 2026-11-25. This is not the baseline, so the fidelity rule above does
  not apply.** Full design and results: `docs/beyond-mega/02-residual.md`.
  - The form is the identity residual X⁽ˡ⁺¹⁾ = ReLU(Ã·X̄⁽ˡ⁾W̄ + X̄⁽ˡ⁾·Δ⁽ˡ⁾) on
    hidden → hidden layers.
  - It is added in the Encoder through a per-degree step ratio r_d.
  - Models are 4-layer GCN.
  - Every added component (Decoder, tables, buffers, DRAM streams) is counted.

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
