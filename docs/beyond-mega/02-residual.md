# 02 — Residual connections on MEGA (ours)

**Status (2026-11-25):** R1 (model + integer reference) and R2 (megasim functional model) implemented. R1 done. R3–R4 planned.
Pointer entry: `docs/mega-kb/04-open-questions.md` Q-36.

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

R1, 3 seeds, QAT test % (FP32 in brackets) @ sparse avg bits. Full table with PTQ and
std: `mega_pt/README.md` results log, 2026-11-25. Runs: `runs/res4/`.

| Dataset | 2-layer | 4-layer plain | 4-layer residual |
|---|---|---|---|
| Cora | 81.67 | 78.13 (78.67) @ 1.32 | **79.33** (80.23) @ 1.34 |
| CiteSeer | 70.35 | 60.70 (66.53) @ 2.14 | **62.60** (70.20) @ 2.55 |
| PubMed | 78.97 | 77.70 (75.83) @ 0.82 | **78.47** (76.33) @ 0.86 |
| Roman-Empire | 46.53 | 33.69 (32.94) @ 0.84 | **49.55** (53.84) @ 0.90 |
| Amazon-Ratings | 47.92 | **46.73** (46.68) @ 0.53 | 46.51 (47.28) @ 0.47 |
| City-Networks Paris | 24.18 | **27.35** (33.89) @ 0.46 | 26.78 (33.44) @ 0.48 |

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
   - **Post needs more bits:** sparse avg 0.61 vs 0.47 (Amazon-Ratings, 4 layers), 1.31 vs 0.90
     (Roman-Empire), 0.62 vs 0.48 (Paris). ReLU(·) + X only adds non-negative terms, so the
     values grow layer by layer and need a wider range.
   - In hardware the bits cost memory and DRAM traffic. post is not a free win.
6. **Under-training.** Roman-Empire, Amazon-Ratings and Paris peak at epochs 134–198 at 8–16
   layers, so those deep numbers are lower bounds. Follow-up with 600 FP32 epochs below.

Formulas the literature uses for depth, not tried here:
- An initial residual, + α·X⁽¹⁾ as in GCNII.
- A separate self weight, X·W_self (GraphSage's root weight; ego/neighbor separation).
  Platonov et al. find this matters most on Roman-Empire / Amazon-Ratings.

Both change the hardware: X⁽¹⁾ must be kept for all layers, or a second combination is needed.
Evaluate them only if depth is a goal for the design.

### Q3. Why quantization loses more at depth: the memory penalty, mostly

4 layers, QAT test % (FP32 in brackets); λ = 0 removes the memory penalty:

| Dataset, form | FP32 | QAT λ = 0 (bits) | QAT recipe λ (bits) |
|---|---|---|---|
| CiteSeer plain | 66.53 | 64.30 (4.11) | 60.70 (2.14) |
| CiteSeer pre | 70.20 | 66.77 (4.48) | 62.60 (2.55) |
| Roman-Empire plain | 32.94 | 33.38 (1.89) | 33.69 (0.84) |
| Roman-Empire pre | 53.84 | **53.84** (1.88) | 49.55 (0.90) |
| Paris plain | 33.92 | 30.82 (1.18) | 27.35 (0.46) |
| Paris pre | 33.57 | 29.76 (1.22) | 26.78 (0.48) |
| Cora pre | 80.23 | 79.23 (3.72) | 79.33 (1.34) |
| PubMed pre | 76.33 | 77.13 (3.22) | 78.47 (0.86) |
| Amazon-Ratings pre | 47.28 | 47.18 (1.37) | 46.51 (0.47) |

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
  - Every layer of Paris, PubMed and Amazon-Ratings ends at 2.00 bits.
- **What remains at λ = 0 is real quantization error,** and it is larger at depth on CiteSeer
  (3.4 points) and Paris (3.1–3.8). At 2 layers Paris loses 0.6.
  - Errors compound through more quantized layers.
  - B̄ is 4 bits (MEGA): 20–45% of B̄ entries round to zero on these layers (`L*_b_lost_all`).
- **Answer: yes, λ should be set per depth.** The recipe λ was tuned for 2 layers. λ sweep at
  4 layers: below.

### Q4. Training time

- Every configuration runs the same 200 FP32 + 200 QAT epochs; the residual does not change
  the epoch count.
- Controlled timing: one job at a time, 4 threads, 20 + 20 epochs, seed 0. Seconds per epoch,
  FP32 / QAT:

| Dataset | 2 layers | 3 plain / pre / post | 4 plain / pre / post | 8 plain / pre / post |
|---|---|---|---|---|
| Cora | 0.16* / 0.13 | 0.16 / 0.15 / 0.15 · 0.15 / 0.14 / 0.14 | 0.16 / 0.16 / 0.16 · 0.16 / 0.16 / 0.15 | 0.20 / 0.20 / 0.19 · 0.21 / 0.20 / 0.20 |
| CiteSeer | 0.31 / 0.52 | 0.33 / 0.32 / 0.31 · 0.52 / 0.51 / 0.52 | 0.33 / 0.34 / 0.33 · 0.54 / 0.57 / 0.55 | 0.39 / 0.38 / 0.37 · 0.66 / 0.60 / 0.58 |
| PubMed | 0.34 / 0.41 | 0.41 / 0.42 / 0.42 · 0.47 / 0.46 / 0.46 | 0.44 / 0.48 / 0.45 · 0.57 / 0.57 / 0.57 | 0.72 / 0.76 / 0.78 · 0.99 / 0.96 / 0.99 |
| Roman-Empire | 0.27 / 0.30 | 0.34 / 0.33 / 0.34 · 0.43 / 0.44 / 0.44 | 0.42 / 0.41 / 0.41 · 0.54 / 0.51 / 0.52 | 0.68 / 0.71 / 0.70 · 0.98 / 0.94 / 1.00 |
| Amazon-Ratings | 0.30 / 0.32 | 0.40 / 0.42 / 0.42 · 0.46 / 0.46 / 0.46 | 0.49 / 0.53 / 0.52 · 0.61 / 0.64 / 0.61 | 0.91 / 0.87 / 0.93 · 1.21 / 1.19 / 1.20 |
| Paris | 0.69 / 1.55 | 1.26 / 1.27 / 1.34 · 2.84 / 3.03 / 2.98 | 1.89 / 1.85 / 1.91 · 4.36 / 4.51 / 4.81 | 4.35 / 4.32 / 4.17 · 9.87 / 9.52 / 9.66 |

\* includes warm-up (first job of the benchmark).

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
