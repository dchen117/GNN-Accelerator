# 02 — Residual connections on MEGA (ours)

**Status (2026-11-25):** R1 (model + integer reference) and R2 (megasim functional model) implemented. R1 done. R3–R4 planned.
Pointer entry: `docs/mega-kb/04-open-questions.md` Q-36.

> **Bits in this document = bits/nz (changed 2026-11-27).** Every "bits" figure here is the
> average bit width of a **non-zero** hidden-layer value, `metrics.avg_bits_nz` =
> Σ nnz(X̄ᵢ)·bᵢ / Σ nnz(X̄ᵢ), over hidden layer inputs X⁽¹⁾… (b ranges 2–8, so the floor is
> 2.00). Earlier versions used the *sparse* average (`avg_bits_sparse`, Q-15) =
> bits/nz × hidden density, where zeros count as 0 bits. That number mixed bit width with ReLU
> sparsity (0.47 "bits" = 2.1 bits × 22% density) and was hard to read. Old results.json files
> are converted exactly by `res_table.bits_nz` (checked against a direct recomputation on
> saved models). Memory also depends on density and on the number of hidden layers: a 4-layer
> model stores three hidden inputs, the 2-layer model one. `res_table` prints bits/nz, density
> and the old sparse average side by side. The MEGA-baseline docs keep the sparse average:
> it is the definition that matches the paper's Tab.VI numbers (Q-15).

## Scope and rules (user, 2026-11-25)

- MEGA has no residual connections (Tab.III: 2-layer GCN/GIN/GraphSage). This work is **not
  the MEGA baseline**. The fidelity rule of `04-open-questions.md` does not apply here:
  - Deviating from MEGA's hardware is allowed.
  - Where a better design for residuals exists, we take it.
- **No free hardware.** Every added component or overhead is counted: area, energy and cycles.
  Examples: a second Decoder, an extra degree table, a new buffer, extra DRAM streams.
  "Small, so ignored" is not allowed.
- The baseline (no `--residual`, 2 layers) is unchanged by this work.

## Decisions

| # | Decision | Why |
|---|---|---|
| 1 | **Identity residual on hidden → hidden layers:** X⁽ˡ⁺¹⁾ = ReLU(Ã·X̄⁽ˡ⁾W̄ + X⁽ˡ⁾_q) for l ≥ 1, l < L−1, F_in = F_out (`mega_pt.model.residual_layers`). | Same form as our SGCN residual (`docs/sgcn/00-design.md` G-08), so the two accelerators run the same network. No projection: a learned X·W_res would cost a second combination. The logits layer never gets one (it never has F_in = F_out here anyway). |
| 2 | **The residual is the quantized input** X⁽ˡ⁾_q = X̄⁽ˡ⁾ · Δ⁽ˡ⁾_d (taken before dropout). | That is what the hardware stores, so training sees exactly what the hardware computes. No new quantizer. Eq.(4)'s memory term is unchanged: X̄⁽ˡ⁾ is already counted. |
| 3 | **Added in the Encoder:** X̄⁽ˡ⁺¹⁾ = min(round(ReLU(psum · e_d · γ_j + X̄⁽ˡ⁾ · r_d)), qmax_d), with **r_d = Δ⁽ˡ⁾_d / Δ⁽ˡ⁺¹⁾_d**. | The residual and the output are on different per-degree steps. r_d moves the residual to the output's step. It is one per-degree table entry and one multiply-add per value in each QN unit. The alternative was seeding the Aggregation Buffer with X̄⁽ˡ⁾ (as SGCN seeds its systolic registers). That needs a per-column (γ_j) and per-node rescale of every value, plus an extra rounding. |
| 4 | **Models:** GCN only, 4 layers (F → 128 → 128 → 128 → C), so layers 1 and 2 carry a residual. | User. GIN/GraphSage later if needed. |
| 5 | **Datasets:** Cora, CiteSeer, PubMed (MEGA's own, for continuity) plus Roman-Empire and Amazon-Ratings (heterophilic) and City-Networks Paris (long range). NELL is skipped at 4 layers. | User. Dataset background: `01-extra-datasets.md`. |
| 6 | **Training recipe unchanged:** hidden 128, 200 FP32 + 200 QAT epochs, M_target at the 2-bit dense floor. λ per dataset as for the 2-layer runs (Cora 1e-5, CiteSeer/PubMed 1e-6, extra datasets 1e-7). 3 seeds. | User. Only revisit if 4-layer runs without residual fail to train. |

### What the Encoder computes (derivation)

Layer l, in hardware order (`mega_pt/hw_stats.py` int_forward; `megasim/functional.py`):

1. Combination: acc = X̄⁽ˡ⁾ · W̄ (integer).
2. Rescale: B̄ = clip(round(acc · α′_d · c_j), ±7), so the real B ≈ B̄ · γ_j. GCN's source
   factor d_j^(−1/2) is folded into α′_d (Q-11).
3. Aggregation: psum_i = Σ_{j ∈ N(i) ∪ {i}} B̄_j (integer).
4. Encoder:
   - The real aggregate is H_i = t_i · γ_j · psum_i. t_i is the destination factor (GCN
     d_i^(−1/2)).
   - The next input is X̄⁽ˡ⁺¹⁾ = min(round(ReLU(H_i) / Δ⁽ˡ⁺¹⁾_d), qmax_d).
   - t and Δ depend only on the degree, so e_d = t_d / Δ⁽ˡ⁺¹⁾_d is one table entry.
   - With the residual, H_i gains X̄⁽ˡ⁾_i · Δ⁽ˡ⁾_d. Dividing by Δ⁽ˡ⁺¹⁾_d gives the X̄⁽ˡ⁾ · r_d term.

**Check (R1):** on a 4-layer residual Cora smoke model, the integer reference agrees with the
fake-quant model on every prediction (exact factors). r_d spans 0.20–0.27 (0.37 octaves), and
16-bit fixed point is lossless; 12-bit costs 0.2 points. The unit test
`tests/test_quant.py::test_residual_layers_and_integer_reference` checks the agreement on the
logits.

## Dataflow: where X̄⁽ˡ⁾ is when the Encoder needs it

Under our schedule (Q-28), subgraph k's X̄⁽ˡ⁾ is consumed by combination in sweep l. The
Encoder needs it again in sweep l+1, after k's aggregation. k cannot be aggregated in sweep l
because some of its sources are in subgraphs not yet combined. Between the two uses, every
other subgraph's X̄⁽ˡ⁾ passes through the Input Buffer.

Measured on the saved 2-layer models: layer 1's X̄, packages + bitmaps
(`python -m megasim.lifetime runs/hw/gcn_cora ...`).

| Run | Whole X̄⁽¹⁾ | Wavefront peak* | Subgraph distance to its last source (median / max) |
|---|---|---|---|
| GCN Cora | 158 KB | 144 KB | 5 / 10 of 11 |
| GCN CiteSeer | 179 KB | 87 KB | 2 / 12 of 14 |
| GCN PubMed | 890 KB | 791 KB | 37 / 78 of 80 |
| GIN Amazon-Ratings | 655 KB | 248 KB | 17 / 72 of 99 |

\* Wavefront: a schedule we have not adopted, which aggregates subgraph k as soon as every
subgraph holding one of its sources has been combined, and drops X̄⁽ˡ⁾_k then.

- **Keeping X̄⁽ˡ⁾ on chip needs a buffer of the whole layer.** That is 158–890 KB here, and it
  grows with N. MEGA's whole SRAM is 392 KB, and its Input Buffer is 64 KB. Even the
  wavefront needs 87–791 KB.
- **Starting point for R3: a DRAM round trip.** X̄⁽ˡ⁾ is written in sweep l and subgraph k's
  rows are prefetched in sweep l+1 with its CSC and B̄ rows.
  - This is the same trade our SGCN makes: it reads residual rows through its cache.
  - Cost: one write and one read of the compressed X̄⁽ˡ⁾ per residual layer.
- **Improvements to evaluate in R3.** Each must count its hardware.
  - A residual buffer of S KB: it keeps the first S KB of X̄⁽ˡ⁾ and saves 2·S of DRAM per
    residual layer. Cost: CACTI area, leakage and access energy.
  - A schedule that shortens X̄⁽ˡ⁾'s lifetime, such as the wavefront above.
  - Writing X̄⁽ˡ⁾ to DRAM costs nothing extra in compute: the Encoder already produces the
    packages. The write is pure DRAM traffic.

## Hardware accounting checklist (R3)

Each line is counted, not assumed free:

- Encoder: the r_d table (one entry per degree, per residual layer) and one multiply-add per
  QN unit. Area and energy from the existing Encoder figures, scaled, or a synthesized
  estimate. Cycles: zero extra only if it pipelines, and the model must show that.
- Decoder for the residual rows. The Encoder needs X̄⁽ˡ⁾ decoded:
  - either a second Decoder (area and energy counted),
  - or the existing one time-shared with combination (cycles counted).
- Input Buffer or a dedicated buffer holding the prefetched residual rows of subgraph k.
  - Size check, as for `x_spill_subgraphs`.
  - Extra SRAM reads and writes in the energy model.
- DRAM: new streams `x_res_w` and `x_res_r` in Ramulator (bandwidth, stalls, energy).

Related: the S-15 index → bitmap expander, a baseline component, is counted the same way
since 2026-11-25 (user): `energy.ADDED_UNITS`, `timing.expander_cycles`, and
`docs/simulator/00-design.md` S-15. Residual units will use the same mechanism: a gate-count
area estimate, power from the power density of a comparable Tab.IV unit, and charged busy
cycles.

## Milestones

1. **R1** (done in code; runs in `runs/res4/`): `--layers`, `--residual` in `mega_pt`; the
   integer reference; 4-layer GCN ± residual on the six datasets.
2. **R2** (done): `megasim.program` exports r_d (`Layer.res`, 16-bit fixed point like the other
   tables) and `functional.encode_next` adds X̄ · r_d.
   - The two products have different binary points: the one with fewer fractional bits is
     shifted left, then the sum gets ReLU and one rounding.
   - Bit-exact against `int_forward` (`python -m megasim.check <run>`) on the 4-layer residual
     Cora smoke model.
   - With the residual term removed, the check fails from layer 2 on, so it does test the add.
   - `tests/test_megasim.py::test_encoder_residual_matches_float` covers both shift orderings.
   - `timing.simulate` refuses residual programs until R3.
3. **R3:** timing, DRAM streams, buffer checks, the components above, energy and area; then
   the improvements.
4. **R4:** MEGA vs SGCN (`sgcnsim --residual`) on the same 4-layer models.

## Results

R1, 3 seeds, QAT test % (FP32 in brackets) @ bits/nz. Full table with PTQ and
std: `mega_pt/README.md` results log, 2026-11-25. Runs: `runs/res4/`.

| Dataset | 2-layer | 4-layer plain | 4-layer residual |
|---|---|---|---|
| Cora | 81.67 | 78.13 (78.67) @ 2.47 | **79.33** (80.23) @ 2.39 |
| CiteSeer | 70.35 | 60.70 (66.53) @ 3.04 | **62.60** (70.20) @ 3.42 |
| PubMed | 78.97 | 77.70 (75.83) @ 2.01 | **78.47** (76.33) @ 2.01 |
| Roman-Empire | 46.53 | 33.69 (32.94) @ 2.55 | **49.55** (53.84) @ 2.97 |
| Amazon-Ratings | 47.92 | **46.73** (46.68) @ 2.14 | 46.51 (47.28) @ 2.13 |
| City-Networks Paris | 24.18 | **27.35** (33.89) @ 2.00 | 26.78 (33.44) @ 2.00 |

- The residual is what makes 4 layers trainable. Roman-Empire, heterophilic, is the clear win:
  the 4-layer residual model beats the 2-layer one. On MEGA's citation graphs depth does not
  pay even with the residual: the 2-layer model stays best, which is the usual result for them.
- Paris (long range) gains from depth itself (+3.2 points over 2 layers, plain); the residual
  adds nothing there (−0.6). Its quantization gap is the largest (FP32 → QAT −6.5), as at 2
  layers: every input sits at the 2-bit dense floor.
- Quantization drops more at depth on CiteSeer (7.6 points), Paris (6.5) and Roman-Empire (4.3), with the
  recipe unchanged (decision 6). Tuning λ for 4 layers is possible later; it does not affect
  the hardware work.
- `megasim.check` is bit-exact on seed 0 of all six residual models, so R3 runs on real
  4-layer models.

## Residual study (2026-11-25): depth, form, quantization loss, training time

The user asked four questions after R1. Runs: `runs/resstudy/` (64 jobs) and `runs/resstudy2/`
(follow-ups). Tables: `python -m mega_pt.res_table runs/res4 runs/resstudy runs/resstudy2`.
Forms: **plain** = no residual; **pre** = ReLU(ÃXW + X) (R1's form); **post** = ReLU(ÃXW) + X
(`--res-form post`). Both forms cost the same in the Encoder: one multiply-add per value, and
only the ReLU moves. Both are bit-exact in megasim (unit tests).
Seeds: 3; Paris 2 (its seed std is 0.1–0.3).

### Q1. Residual at 2 layers

- **Not defined for an identity residual.** Layer 0 is F → 128 and layer 1 is 128 → C;
  neither has F_in = F_out.
- **3 layers** (F → 128 → 128 → C, one residual layer) is the shallowest identity-residual
  network.
- **A projection skip** (+ X·W_s) would work at 2 layers.
  - On the last layer it is GraphSage's self weight; W_s is 128 × C, so it is cheap.
  - On layer 0 it would double the largest combination.
  - Not built yet.

QAT test % (FP32 in brackets), recipe λ:

| Dataset | 2-layer | 3 plain | 3 pre | 3 post | 4 plain | 4 pre | 4 post |
|---|---|---|---|---|---|---|---|
| Cora | **81.67** | 80.33 (81.90) | 80.33 (81.83) | 80.40 (82.53) | 78.13 | 79.33 | 78.40 |
| CiteSeer | **70.35** | 66.53 (68.70) | 66.97 (70.13) | 67.03 (69.70) | 60.70 | 62.60 | 63.30 |
| PubMed | 78.97 | **80.40** (77.90) | 79.40 (77.70) | 80.00 (77.67) | 77.70 | 78.47 | 78.53 |
| Roman-Empire | 46.53 | 36.02 (34.51) | **51.45** (52.91) | 51.30 (51.99) | 33.69 | 49.55 | 50.56 |
| Amazon-Ratings | **47.92** | 46.62 (46.52) | 47.56 (48.08) | 47.63 (47.88) | 46.73 | 46.51 | 46.72 |
| Paris | 24.18 | 26.58 (32.62) | 26.23 (31.83) | 25.76 (30.78) | **27.35** | 26.78 | 25.89 |

- At 3 layers the residual matters only on Roman-Empire: +15 points, and 51.45 is the best
  Roman-Empire result at any depth, 5 points over the 2-layer model.
- Elsewhere it is within about 1 point of plain.

### Q2. Why the residual does not help everywhere: deeper networks and the post form

FP32 test %, the same 200 epochs (QAT would add the quantization confound):

| Dataset | 2 | 4 plain / pre / post | 8 plain / pre / post | 16 pre / post |
|---|---|---|---|---|
| Cora | 81.65 | 78.67 / 80.23 / 80.83 | 34.43 / 79.53 / 80.17 | 71.17 / 74.67 |
| CiteSeer | 71.23 | 66.53 / 70.20 / 68.93 | 26.73 / 67.03 / 65.33 | 57.67 / 63.13 |
| PubMed | 78.99 | 75.83 / 76.33 / 76.97 | 55.60 / 76.47 / 76.07 | 69.83 / 71.13 |
| Roman-Empire | 47.43 | 32.94 / 53.84 / 52.38 | 19.67 / 46.32 / 50.11 | 25.15 / 39.16 |
| Amazon-Ratings | 47.90 | 46.68 / 47.28 / 47.53 | 45.33 / 45.73 / 46.99 | 41.56 / 44.16 |
| Paris | 27.01 | 33.89 / 33.44 / 31.83 | 32.86 / 31.93 / 30.57 | – |

1. **The residual stops the collapse but does not make depth pay.**
   - Without it, 8 layers collapse: Cora 34%, CiteSeer 27%, PubMed 56%, Roman-Empire 20%.
   - With it, 8 layers stay within 1–3 points of 4 layers.
   - But every dataset's best depth is 2–4, with or without the residual. More layers
     (8, 16) do not reveal a residual benefit; they lose accuracy.
2. **On the citation graphs (Cora, CiteSeer, PubMed) depth is not the bottleneck.**
   - Their FP32 best-val epoch is 9–60 out of 200: the models overfit about 20 training
     nodes per class within a few dozen epochs.
   - Extra layers only add parameters and smoothing.
   - This is the usual result for these graphs. The residual can recover what depth loses,
     not what 2 layers already reach.
3. **Roman-Empire is where the residual matters.** Even at 3 layers it gains 15–19 points.
   - On a heterophilic graph a node's neighbors mostly have other labels, so a node needs its
     own features separated from the aggregate (ego/neighbor separation).
   - The identity residual carries the node's own features past the averaging.
   - Amazon-Ratings is heterophilic too, but GCN gains little there from either depth or the
     residual (all within 1.5 points).
4. **Paris gains from depth itself:** plain 4 layers beats 2 layers by 7 points (FP32).
   - The residual costs 0.5–2 points there, more for post.
   - Its long-range signal needs more hops, not the self features.
5. **pre vs post.**
   - At 8–16 layers, post is better on Cora, Roman-Empire, Amazon-Ratings, and at 16 layers
     on CiteSeer and PubMed (Roman-Empire 16 layers: 39.2 vs 25.2).
   - pre is better on Paris and at 8 layers on CiteSeer and PubMed.
   - At 3–4 layers they are within about 1 point; post is ahead on Roman-Empire, CiteSeer and
     PubMed at 4 layers.
   - **Post stores more non-zeros, not wider values.** Bits/nz are the same (Amazon-Ratings,
     4 layers: 2.17 post vs 2.13 pre; Roman-Empire 2.94 vs 2.97; Paris 2.00 vs 2.00), but
     post's hidden inputs are denser (28% vs 22%, 44% vs 30%, 31% vs 24%). ReLU(·) + X keeps
     every non-zero of X, whereas pre lets the ReLU zero some of them. (An earlier version of
     this note read the higher *sparse* average as "post needs a wider range"; with bits/nz
     that explanation does not hold.)
   - In hardware the extra non-zeros cost memory and DRAM traffic. post is not a free win.
6. **Under-training.** Roman-Empire, Amazon-Ratings and Paris peak at epochs 134–198 at 8–16
   layers, so those deep numbers are lower bounds. Follow-up with 600 FP32 epochs:

**Follow-up: 600 FP32 epochs** (`runs/resstudy2`, same recipe otherwise; FP32 test %, best-val
epoch in brackets):

| Dataset, L, form | 200 epochs | 600 epochs | seeds |
|---|---|---|---|
| Paris 4 plain | 33.92 (≈190) | 35.42 (599) | 1 |
| Paris 8 plain | 32.86 (192) | 35.46 (556) | 1 |
| Roman-Empire 4 pre | 53.84 | **56.41** (535) | 3 |
| Roman-Empire 8 pre | 46.32 (189) | 54.89 (589) | 3 |
| Roman-Empire 8 post | 50.11 (192) | 53.81 (566) | 3 |
| Roman-Empire 16 post | 39.16 (198) | 46.61 (590) | 3 |
| Amazon-Ratings 8 post | 46.99 (153) | 47.34 (284) | 3 |

- **Most of the depth penalty on Roman-Empire and Paris was under-training.** With 600 epochs
  the 8-layer models close to within 1.5 points of 4 layers (Roman-Empire) or tie them
  (Paris), but **they still do not beat 4 layers.** Deep models need ~3× the epochs to get
  there, and 16 layers stay 10 points behind.
- Even at 600 epochs the best-val epoch is near the end (535–599), so these are still lower
  bounds; the 4-layer models are under-trained too.
- Amazon-Ratings converges at ~280 epochs and gains only 0.35.
- Caveat: the 2-layer references are also 200-epoch runs. A fair depth comparison would
  give every depth the same longer schedule; not done (the recipe stays at 200 for the
  baseline comparison).

Formulas the literature uses for depth, not tried here:
- An initial residual, + α·X⁽¹⁾ as in GCNII.
- A separate self weight, X·W_self (GraphSage's root weight; ego/neighbor separation).
  Platonov et al. find this matters most on Roman-Empire / Amazon-Ratings.

Both change the hardware: X⁽¹⁾ must be kept for all layers, or a second combination is needed.
Evaluate them only if depth is a goal for the design.

### Q3. Why quantization loses more at depth: the memory penalty, mostly

4 layers, QAT test % (bits/nz in brackets); λ = 0 removes the memory penalty:

| Dataset, form | FP32 | QAT λ = 0 (bits) | QAT recipe λ (bits) |
|---|---|---|---|
| CiteSeer plain | 66.53 | 64.30 (4.93) | 60.70 (3.04) |
| CiteSeer pre | 70.20 | 66.77 (5.08) | 62.60 (3.42) |
| Roman-Empire plain | 32.94 | 33.38 (4.63) | 33.69 (2.55) |
| Roman-Empire pre | 53.84 | **53.84** (5.00) | 49.55 (2.97) |
| Paris plain | 33.92 | 30.82 (4.23) | 27.35 (2.00) |
| Paris pre | 33.57 | 29.76 (4.58) | 26.78 (2.00) |
| Cora pre | 80.23 | 79.23 (4.77) | 79.33 (2.39) |
| PubMed pre | 76.33 | 77.13 (4.63) | 78.47 (2.01) |
| Amazon-Ratings pre | 47.28 | 47.18 (4.99) | 46.51 (2.13) |

- **The penalty causes most of the extra loss.** Removing it:
  - Roman-Empire: residual QAT equals FP32 (53.84), so all 4.3 points were the penalty.
  - CiteSeer: recovers 4.2 of the 7.6 points.
  - Paris: recovers 3.5 of 6.5.
  - Cora, PubMed, Amazon-Ratings: the penalty costs ≤ 0.7, and on PubMed it helps (+1.3,
    regularization).
- **Why the same λ is harsher at depth.**
  - M_target is the 2-bit dense floor and b ≥ 2, so the target can never be met.
  - The penalty λ(M − M_target)² keeps pushing every layer down; its gradient is
    2λ(M − M_target).
  - A 4-layer model has 3 hidden inputs instead of 1, so M − M_target, and the push, is larger.
  - Paris and PubMed end at the 2.00-bit floor (bits/nz 2.00–2.01), Amazon-Ratings at 2.13.
- **What remains at λ = 0 is real quantization error,** and it is larger at depth on CiteSeer
  (3.4 points) and Paris (3.1–3.8). At 2 layers Paris loses 0.6.
  - Errors compound through more quantized layers.
  - B̄ is 4 bits (MEGA): 20–45% of B̄ entries round to zero on these layers (`L*_b_lost_all`).
- **Answer: yes, λ should be set per depth.** The recipe λ was tuned for 2 layers.

**λ sweep at 4 layers** (`runs/resstudy2`; QAT test %, bits/nz in brackets; FP32 from
the shared cache):

| Dataset, form (FP32) | λ = 0 | 1e-9 | 1e-8 | 1e-7 | 1e-6 |
|---|---|---|---|---|---|
| CiteSeer pre (70.20) | 66.77 (5.08) | – | 65.90 (4.61) | 64.13 (4.03) | 62.60 (3.42) ← recipe |
| Roman-Empire pre (53.84) | 53.84 (5.00) | **54.12** (4.93) | 52.21 (3.79) | 49.55 (2.97) ← recipe | – |
| Paris plain (33.92) | 30.82 (4.23) | – | **28.41 (2.01)** | 27.35 (2.00) ← recipe | – |

- **The trade-off is smooth on CiteSeer and Roman-Empire:** each 10× in λ costs ~1–2.5 points
  and saves ~0.5–1.1 bits per value. There is no free lunch; λ picks a point on the curve.
- **Paris λ = 1e-8 is a free win:** +1.06 points at the same 2.0 bits (the floor) as the recipe. The
  recipe λ over-pushes Paris 4 layers; 1e-8 reaches the same floor more gently.
- λ ≤ 1e-9 is effectively λ = 0 (same bits, same accuracy).
- **Implication for comparisons:** a depth's accuracy and its bits must be read together.
  Comparing 2- vs 4-layer (or plain vs residual) at one λ mixes the depth effect with a
  different point on the accuracy/memory curve. Better options: match bits (choose λ per
  configuration to hit a common bits/nz or memory), or normalize the penalty per layer, e.g.
  λ·Σₗ(Mₗ − Mₗ,target)² / (L − 1). Not implemented yet.

### Q4. Training time

- Every configuration runs the same 200 FP32 + 200 QAT epochs; the residual does not change
  the epoch count.
- Controlled timing: one job at a time, 4 threads, 20 + 20 epochs, seed 0. Seconds per epoch,
  FP32 / QAT:

| Dataset | 2 layers | 3 plain / pre / post | 4 plain / pre / post | 8 plain / pre / post |
|---|---|---|---|---|
| Cora | 0.78* / 0.13 | 0.16 / 0.15 / 0.15 · 0.15 / 0.14 / 0.14 | 0.16 / 0.16 / 0.16 · 0.16 / 0.16 / 0.15 | 0.20 / 0.20 / 0.19 · 0.21 / 0.20 / 0.20 |
| CiteSeer | 0.31 / 0.52 | 0.33 / 0.32 / 0.31 · 0.52 / 0.51 / 0.52 | 0.33 / 0.34 / 0.33 · 0.54 / 0.57 / 0.55 | 0.39 / 0.38 / 0.37 · 0.66 / 0.60 / 0.58 |
| PubMed | 0.34 / 0.41 | 0.41 / 0.42 / 0.42 · 0.47 / 0.46 / 0.46 | 0.44 / 0.48 / 0.45 · 0.57 / 0.57 / 0.57 | 0.72 / 0.76 / 0.78 · 0.99 / 0.96 / 0.99 |
| Roman-Empire | 0.27 / 0.30 | 0.34 / 0.33 / 0.34 · 0.43 / 0.44 / 0.44 | 0.42 / 0.41 / 0.41 · 0.54 / 0.51 / 0.52 | 0.68 / 0.71 / 0.70 · 0.98 / 0.94 / 1.00 |
| Amazon-Ratings | 0.30 / 0.32 | 0.40 / 0.42 / 0.42 · 0.46 / 0.46 / 0.46 | 0.49 / 0.53 / 0.52 · 0.61 / 0.64 / 0.61 | 0.91 / 0.87 / 0.93 · 1.21 / 1.19 / 1.20 |
| Paris | 0.69 / 1.55 | 1.26 / 1.27 / 1.34 · 2.84 / 3.03 / 2.98 | 1.89 / 1.85 / 1.91 · 4.36 / 4.51 / 4.81 | 4.35 / 4.32 / 4.17 · 9.87 / 9.52 / 9.66 |

\* includes warm-up (first job of the benchmark; Cora 3–8 layers take 0.15–0.20 s, so 2 layers is about 0.15).

- **The residual adds no measurable time,** pre or post, at any depth (within noise, ≤ 5%).
  The add is one elementwise op next to a sparse matmul.
- **Time grows about linearly with depth.** Paris is superlinear, from its large N and the
  memory traffic of the hidden activations.
- **Epochs to converge** (FP32 best-val epoch, mean of seeds; `fp32_best_epoch` in results):

| Dataset | 3 plain / pre / post | 8 plain / pre / post | 16 pre / post |
|---|---|---|---|
| Cora | 19 / 18 / 17 | 103 / 57 / 13 | 183 / 168 |
| CiteSeer | 88 / 56 / 71 | 71 / 20 / 12 | 156 / 176 |
| PubMed | 123 / 52 / 53 | 124 / 17 / 51 | 118 / 169 |
| Roman-Empire | 172 / 190 / 167 | 188 / 189 / 192 | 195 / 198 |
| Amazon-Ratings | 142 / 177 / 175 | 134 / 164 / 153 | 194 / 194 |
| Paris | 182 / 186 / 176 | 192 / 191 / 196 | – |

- The residual usually converges **earlier** on the citation graphs at 8 layers (Cora 57 / 13
  vs 103 plain; PubMed 17 / 51 vs 124).
- Roman-Empire, Amazon-Ratings and Paris are still improving at epoch 200 at every depth.
- At 16 layers everything peaks late.

## Residual benefit study (2026-11-26/27): where does the residual pay?

User goal: show that the residual benefits *some* domain / dataset / task, on more than
Roman-Empire. Six questions; answers below. Runs: `runs/res5/` (102 FP32 screening jobs,
3 seeds, 200 epochs, per-epoch val curves) and `runs/res5q/` (QAT on the winners, recipe
λ = 1e-7). Tables: `python -m mega_pt.res_table runs/res5 runs/res5q`; convergence:
`python -m mega_pt.res_table --curves runs/res5`.

New datasets (Platonov et al. 2023, same source as Roman-Empire / Amazon-Ratings; pinned in
`mega_pt/data.py`): **Minesweeper** (10k nodes, a 100×100 grid, label = is a mine, features =
one-hot count of neighbouring mines, 50% of them hidden), **Tolokers** (11.8k crowd workers,
edge = worked on the same task, label = banned), **Questions** (49k users of a Q&A site, edge
= answered each other, label = still active). All three are binary and imbalanced, so the
metric is **ROC-AUC** (as in the paper; `metrics.roc_auc`, `Graph.metric = "auc"`). The same
three-model recipe is used: hidden 128 (GraphSage 256), dropout, lr 0.01, 200 epochs.

### Q1 + Q4. Datasets and models where the residual helps (FP32, 3 seeds)

Test % (accuracy, or ROC-AUC for Minesweeper / Tolokers / Questions). **Bold** = the
residual model beats the best plain model of any depth by more than 1 std.

| Dataset | Model | 2 layers | 4 plain | 4 residual | 8 plain | 8 residual |
|---|---|---|---|---|---|---|
| Minesweeper | GCN | 72.18 | 69.66 | **78.14** | 65.75 | 75.90 |
| Minesweeper | GraphSage | 72.18 | 69.72 | **77.44** | 65.81 | 74.70 |
| Minesweeper | GIN | 72.29 | 68.02 | 68.17 | 61.31 | 60.16 |
| Tolokers | GCN | 67.55 | 67.59 | **72.59** | 62.67 | 71.54 |
| Tolokers | GraphSage | 71.78 | 71.82 | **74.35** | – | – |
| Tolokers | GIN | 69.22 | 68.65 | 68.73 | – | – |
| Questions | GCN | 64.35 | 59.14 | 64.07 | 48.59 | 61.56 |
| Questions | GraphSage | 72.87 | 72.27 | 74.23 | – | – |
| Questions | GIN | 62.75 | 62.84 | 62.63 | – | – |
| Roman-Empire | GCN | 47.43 | 32.90 | **53.87** | 19.67 | 46.32 |
| Roman-Empire | GraphSage | 45.46 | 32.32 | **50.95** | 18.71 | 38.95 |
| Roman-Empire | GIN | 45.17 | 14.78 | 14.86 | 13.92 | 13.81 |
| Amazon-Ratings | GCN | 47.90 | 46.85 | 47.30 | 45.33 | 45.73 |
| Amazon-Ratings | GraphSage | 47.98 | 47.29 | 47.43 | – | – |
| Amazon-Ratings | GIN | 45.11 | 36.76 | 36.63 | – | – |
| Cora | GCN | 81.57 | 78.63 | 80.23 | 34.43 | 79.53 |
| Cora | GraphSage | 81.70 | 79.53 | 81.30 | – | – |
| Cora | GIN | 78.07 | 73.47 | 73.87 | – | – |
| CiteSeer | GCN | 71.10 | 67.13 | 70.03 | 26.73 | 67.03 |
| CiteSeer | GraphSage | 71.60 | 64.93 | 67.87 | – | – |
| CiteSeer | GIN | 67.83 | 63.37 | 62.43 | – | – |
| PubMed | GCN | 79.33 | 75.83 | 76.33 | 55.60 | 76.47 |
| PubMed | GraphSage | 78.27 | 76.83 | 76.87 | – | – |
| PubMed | GIN | 76.70 | 73.93 | 76.73 | – | – |

(GCN Tolokers 3 layers: plain 69.39, residual 72.12. GCN Questions 3 layers: 58.49 / 62.64.)

**Q1 answer.** Yes: **Minesweeper (+6 AUC over the best plain model) and Tolokers (+3–5)**
join Roman-Empire, for both GCN and GraphSage. **Questions** is weaker: GraphSage +1.4 (within
the 2-layer std of 1.6), GCN only recovers the 2-layer level. In all four the 4-layer residual
model is the best configuration we have trained. What they share: Q2.

**Q4 answer.**
- **GraphSage: yes**, the same pattern as GCN, on the same four datasets. Nothing changes in
  the hardware: the residual is added in the Encoder after the aggregation, where the three
  models differ only in the per-degree factors (`GraphOps.src_scale / dst_scale`). GIN and
  GraphSage residual models are bit-exact in megasim (`megasim.check`, smoke runs 2026-11-26).
- **GIN: no benefit anywhere.** Deep GIN does not train in our setup: on Roman-Empire and
  Amazon-Ratings the 4-layer model's best val epoch is 1–3 and accuracy falls to the majority
  class (14.8%, 36.8%). Cause: our GIN is GIN-0 with one linear layer and a plain **sum** over
  A+I (MEGA's form), with no normalization. Each layer multiplies the activation scale by
  ~deg, so after 3 layers it is ~deg³ (Amazon-Ratings: mean degree 8, Tolokers 89). The
  identity term X is then negligible next to the aggregate, so the residual cannot help.
  The published GIN uses a 2-layer MLP with batch norm per layer, which bounds the scale.
- **Learning rate is not the cause** (`runs/res5q`, 4 layers, `--lr 1e-3` instead of 1e-2,
  FP32, 3 seeds): Roman-Empire plain 14.82 / residual 15.10, Amazon-Ratings 36.62 / 36.64,
  Minesweeper 67.95 / 67.96. Still at the class prior on the first two, and identical with or
  without the residual. Making GIN benefit needs a normalized aggregation (e.g. mean over
  A+I, or GIN-ε with a per-layer scale), which is a change to the model, not the residual.
  Not done: GIN would no longer be MEGA's GIN, and GCN/GraphSage already make the case.

### Q2. What task are we doing, and which tasks should benefit?

- **What we do:** transductive, full-graph **node classification** (GraphSage: sampled
  neighbours), one graph, labels on a subset of nodes, MEGA's setting. Citation graphs use
  the Planetoid split: ~20 labels per class (≤ 5% of nodes).
- **Where the residual paid:** Roman-Empire, Minesweeper, Tolokers, Questions. What they
  share:
  1. **Heterophily / ego-vs-neighbour signal.** The label depends on the node's *own*
     features as well as on (often dissimilar) neighbours: a Minesweeper cell's own
     revealed count; a Roman-Empire word's own embedding vs. its context. Without a
     residual, GCN/GraphSage average the node into its neighbourhood at every layer and
     lose that signal (oversmoothing): 4 plain layers are *worse* than 2. With it, the node's
     representation survives each averaging step (Platonov et al. make the same point with
     ego/neighbour separation).
  2. **Enough labels.** 50% of nodes are training nodes, so a deeper model does not simply
     overfit. On Cora / CiteSeer / PubMed (≤ 5% labelled) depth overfits by epoch 10–60
     whatever the form; the residual only stops the collapse.
  3. **Possibly, information beyond 2 hops** (hypothesis, not tested): a Minesweeper cell's
     label is constrained by the counts of its neighbours, which depend on *their*
     neighbours, so 3–4 hops can help.
- **Where it did not pay:** Amazon-Ratings (heterophilous, but the signal is weak for every
  model, ~48%), Paris (long-range: depth helps, the identity residual does not), citation.
- **Tasks that should benefit more, not available in MEGA's setting:**
  - Long-range benchmarks (LRGB: Peptides, PascalVOC-SP), where 6–10+ layers are needed.
    Graph-level tasks need batching of many small graphs and a pooling/readout stage:
    new hardware.
  - Large, well-labelled node classification: **ogbn-arxiv** (169k nodes, 54% train), where
    3-layer GCN with residual connections is the usual baseline, and ogbn-proteins (deep
    ResGCN / DeeperGCN). Not run: the environment's network policy blocks
    `snap.stanford.edu`.
  - Graph classification with GIN (molecules), where residual/JK connections are standard.

### Q3. Other benefits: convergence speed (FP32 val curves)

First epoch at which val reaches **0.99 × the plain model's best val at the same depth**
(a level both forms reach), mean of seeds; and first epoch at which val reaches 0.99 × the
**2-layer** model's best (shallow-model quality). From `res_table --curves runs/res5`.

| Dataset, model, 4 layers | plain → own best | residual → plain's best | residual → 2-layer level | 2-layer → own level |
|---|---|---|---|---|
| Roman-Empire GCN | 138 | **19** | 73 | 100 |
| Roman-Empire GraphSage | 162 | **34** | 101 | 105 |
| Tolokers GCN | 104 | **44** | 46 | 71 |
| Tolokers GraphSage | 101 | **75** | 71 | 57 |
| Questions GCN | 150 | **7** | 149 (1/3 seeds) | 12 |
| Questions GraphSage | 98 | **72** | 90 | 96 |
| Minesweeper GCN | 5 | 4 | 25 | 11 |
| Cora GCN | 59 | **15** | 10 | 32 |
| CiteSeer GCN | 55 | **7** | 8 (1/3) | 59 |

- **At the same depth the residual model converges 1.3–20× faster** to the plain model's
  quality (and then keeps improving), on almost every dataset.
- **Against the 2-layer model it is about even in epochs** (Tolokers / Roman-Empire GCN are
  faster, Minesweeper slower). In wall-clock it is slower: a 4-layer epoch costs 1.5–2× a
  2-layer epoch (Q4 of the previous study). So the honest claim is "deeper *and* no slower
  to train per epoch than the shallow model", not "faster than the shallow model".
- Other benefits observed or worth testing:
  - **Robustness to depth:** at 8 layers plain GCN collapses (Cora 34%, CiteSeer 27%) while
    the residual stays within 1–3 points of the best. A design that must fix the layer count
    in hardware does not have to tune depth per dataset.
  - **Seed stability:** residual std is often lower (Cora 4 layers 0.29 vs 1.76; Tolokers
    1.01 vs 2.49).
  - **Quantization robustness:** mixed. The benefit survives QAT on Minesweeper and
    Tolokers, but the residual model loses more accuracy to quantization than the 2-layer
    model at the same λ. See "Does the benefit survive quantization?" below.

### Does the benefit survive quantization? (QAT, λ = 1e-7, 3 seeds)

Same recipe as the rest of the study (`--target-dense-bits 2.0`, λ = 1e-7, FP32 models from
the screening cache, so the FP32 column equals the screening table). Runs in `runs/res5q`;
`python -m mega_pt.res_table runs/res5q`. QAT = quantized test score (% ROC-AUC, or %
accuracy for Roman-Empire); bits = bits/nz (floor 2.00).

| Dataset, model | 2 layers QAT (bits) | 4 plain QAT (bits) | 4 residual QAT (bits) | residual FP32 → QAT |
|---|---|---|---|---|
| Minesweeper GCN | 72.06 (2.99) | 69.35 (2.80) | **75.58** (2.45) | 78.14 → 75.58 |
| Minesweeper GraphSage | 71.97 (2.67) | 67.89 (2.34) | **74.94** (2.44) | 77.44 → 74.94 |
| Tolokers GCN | 68.07 (2.62) | 63.82 (2.33) | **69.91** (2.37) | 72.59 → 69.91 |
| Tolokers GraphSage | 72.64 (2.58) | 71.45 (2.26) | **73.71** (2.01) | 74.35 → 73.71 |
| Questions GCN | **60.90** (2.02) | 53.84 (2.01) | 58.72 (2.01) | 64.07 → 58.72 |
| Questions GraphSage | **75.46** (2.00) | 63.93 ± 4.06 (2.02) | 71.89 (2.00) | 74.23 → 71.89 |
| Roman-Empire GraphSage | **45.81** (2.58) | 33.06 (2.16) | 44.29 (2.14) | 50.95 → 44.29 |
| Roman-Empire GCN (earlier runs) | 46.53 (3.16), `runs/ext` λ=1e-7 s0–s2 | 33.69 (2.55) | **49.55** (2.97) | 53.84 → 49.55 |

Std ≤ 1.5 except where shown. Findings:
- **Minesweeper and Tolokers keep the benefit after quantization**, for both models: the
  4-layer residual model beats the 2-layer model by +1.1 to +3.5 points, at fewer bits per
  value (2.0–2.45 vs 2.6–3.0). Total memory is still higher: it stores three hidden inputs
  instead of one. Roman-Empire GCN also keeps it (+3.0 points, 2.97 vs 3.16 bits).
- **The residual always beats 4 plain layers after QAT** (+2 to +11 points), so it remains
  the right choice for a fixed 4-layer design.
- **But the residual model loses more to quantization**: 0.6–6.7 points, against ≈ 0
  for the 2-layer model (which even gains on Questions/Tolokers GraphSage, a regularization
  effect). On Questions and Roman-Empire GraphSage this erases the FP32 gain.
- **This is a λ (bit-budget) issue, not a defect of the residual.** On Roman-Empire GCN at 4
  layers, lowering λ closes the gap completely: λ = 1e-7 → 49.55 at 2.97 bits, 1e-8 → 52.21
  at 3.79, 1e-9 → 54.12 at 4.93 (FP32 53.84; `runs/resstudy2`). The skip path carries X̄⁽ˡ⁾
  at its own (low) bit width, so every layer re-adds its quantization error instead of
  averaging it away over neighbours. A 4-layer model at the 2-layer model's λ is
  pushed to fewer bits than the 2-layer model, so the comparison above is slightly
  unfavourable to the residual. The two cases that lose (Questions, Roman-Empire GraphSage)
  sit at or near the 2-bit floor (2.00, 2.14), where λ can push no further.
- **Next step if this matters:** compare at matched bits (pick λ per depth so the average bits
  equal the 2-layer model's), or give the skip operand its own bit-width floor. Both cost
  more memory traffic. That traffic is what R3 measures.

### Q5. How the residual is handled in software, and what the hardware receives

- **Training (`mega_pt.model.QGCN.forward`).** The residual operand is the layer's
  *fake-quantized input* X_q = X̄⁽ˡ⁾ · Δ⁽ˡ⁾_d, the same tensor the combination consumes,
  taken before dropout (decision 2). It is added in float to the aggregate (pre: before the
  ReLU; post: after). There is no explicit rescaling during training: X_q and the aggregate
  are both real-valued, and the next layer's quantizer re-quantizes the sum with its own
  per-degree step Δ⁽ˡ⁺¹⁾_d. Gradients flow through X_q by the straight-through estimator,
  as for the combination input.
- **Export (`megasim.program`).** The scale change is made explicit here. For each residual
  layer the program gets one table r_d = Δ⁽ˡ⁾_d / Δ⁽ˡ⁺¹⁾_d (one entry per degree present,
  16-bit fixed point with a shared binary point, like MEGA's other degree tables), plus the
  `res_post` flag. 16 bits is lossless (megasim bit-exact runs); 12 bits cost 0.2 points on
  the R1 Cora smoke model.
- **Hardware (`megasim.functional.encode_next`).** The Encoder computes
  X̄⁽ˡ⁺¹⁾ = min(round(ReLU(psum·e_d·γ_j + X̄⁽ˡ⁾·r_d)), qmax_d): both products are integers
  with their own binary points; the one with fewer fractional bits is shifted to align, then
  one add, ReLU, one rounding, clip. So **the residual is not a new tensor passed in**: it is
  X̄⁽ˡ⁾, the layer's own compressed input, which already exists (it was written by the
  previous layer's Encoder and read by this layer's combination). What is new is (a) the r_d
  table and (b) getting X̄⁽ˡ⁾ back to the Encoder after aggregation, through DRAM
  (`x_res_w`/`x_res_r`) and a Decoder (R3 checklist above). The PyTorch integer reference
  (`hw_stats.int_forward`) and megasim are bit-exact for GCN, GIN and GraphSage.

### Q6. Specialize or generalize the residual?

Variants, by what they need in MEGA's dataflow (X·W first, then Ã·B; Encoder at the end):

| Variant | Formula (per layer) | Extra hardware vs. ours | Fits the Encoder add? |
|---|---|---|---|
| Identity, pre / post (ours) | ReLU(ÃXW + X) / ReLU(ÃXW) + X | r_d table, X̄⁽ˡ⁾ round trip | yes |
| Scaled / gated identity | ÃXW + s·X, s learned per layer or per channel | none: s folds into r_d (per layer) or into a per-column table like γ_j | yes |
| Initial residual (APPNP-style) | (1−α)ÃXW + α·X⁽¹⁾ | keep X̄⁽¹⁾ (one hidden layer) for all layers instead of X̄⁽ˡ⁾ | yes (different source address) |
| GCNII | σ(((1−α)ÃX + αX⁽¹⁾)((1−β)I + βW)) | W acts *after* the mix; in X·W-first order that is a second combination (X⁽¹⁾·W) per layer | no (2× combination) |
| Projection / self weight (GraphSage root weight) | ÃXW + X·W_self | second weight matrix: a wider combination X·[W W_self], and the self half bypasses aggregation to the Encoder | partly (2× combination, but no extra aggregation) |
| Dense / JK connections | concat or max of all layers | stores all layers, wider inputs | no |

Recommendation:
- **Generalize the cheap axis, specialize the expensive one.** Build one Encoder "skip
  add" that takes (source layer index, per-degree table, optional per-column table,
  pre/post flag). That one unit covers identity (pre/post), scaled/gated identity and the
  initial residual at almost no extra cost: the per-column table reuses γ_j's structure,
  and the source index only changes which X̄ rows are fetched. All of these are
  model-agnostic (GCN / GraphSage / GIN differ only in the degree factors, already folded
  into e_d and r_d) and use the same quantization rule: align binary points, add, round once.
- **Do not build variants that put a matrix multiply on the skip path** (GCNII, projection)
  unless a target workload needs them: each doubles the combination work and adds weights,
  which is MEGA's dominant cost. The exception worth evaluating is the projection skip,
  which is the only way to get a residual at 2 layers (F_in ≠ F_out) and is GraphSage's
  standard root weight.
- **Quantization is what makes the generalization cheap or not.** Every variant that adds a
  stored, already-quantized tensor costs one aligned multiply-add per value. Anything that
  needs a *new* tensor (X⁽¹⁾·W, concatenations) needs a new quantizer, new range/bit tables
  and new memory in Eq.(4).
- **GIN needs a different fix first:** normalization (or mean aggregation, which is
  GraphSage) to bound the activation scale; a residual alone does not help it (Q4).
