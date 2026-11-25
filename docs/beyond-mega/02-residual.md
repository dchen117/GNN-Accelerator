# 02 — Residual connections on MEGA (ours)

**Status (2026-11-25):** R1 (model + integer reference) and R2 (megasim functional model) implemented. R1 runs in progress. R3–R4 planned.
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

Related: the S-15 index → bitmap expander (`docs/simulator/00-design.md`) is a component we
currently leave unmodelled. Under the rule above it should be counted too. This is noted for
R3, where the logic cost model gets built anyway.

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

R1 accuracy tables go here when the runs finish.
