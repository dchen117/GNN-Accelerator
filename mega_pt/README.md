# mega_pt — PyTorch model of MEGA's Degree-Aware quantization (GCN)

A "fake-quant" PyTorch model of MEGA's algorithm (HPCA 2024, arXiv:2311.09775). It
reproduces the accuracy and average-bits numbers in Tab.VI, and it measures the
hardware-relevant value ranges that will feed the simulator later.

Every design choice the paper leaves open is a recorded decision in
[`docs/mega-kb/04-open-questions.md`](../docs/mega-kb/04-open-questions.md). Code comments
cite those Q-IDs and the paper's equation numbers.

## Setup

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

The CPU-only torch wheel is much smaller. Locally, you can install that one instead:

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The datasets download automatically on first use to `data/planetoid/`, which is gitignored.

## Run

Quick check that the data loads (edge counts should match Tab.II: Cora 10556, CiteSeer 9104,
PubMed 88648, NELL 251550):

```bash
.venv/bin/python -m mega_pt.data
```

Unit tests (each test names the equation or Q-ID it checks):

```bash
.venv/bin/python -m pytest -q tests
```

One configuration, 10 seeds (the setting that matches Tab.VI on Cora, CiteSeer and PubMed;
the defaults already select `--norm fold --feat-norm row --x-init mse --clip-refit 10 --lr-bits 2e-2`):

```bash
.venv/bin/python -m mega_pt.train --dataset citeseer --lam 1e-7 --target-dense-bits 2.0
```

Debug ablation, quantizing only some tensors (here: only the degree-aware features):

```bash
.venv/bin/python -m mega_pt.train --dataset pubmed --norm fold --lam 0 --seeds 3 --quant-parts x
```

All options:

```bash
.venv/bin/python -m mega_pt.train --help
```

Outputs go to `runs/<name>/`:
- `results.json`: arguments, mean ± std summary, and per-seed metrics.
- `bits_seed<k>.json`: the learned per-degree b and α tables for each layer.

## What the pipeline does (per seed)

1. **FP32 pretraining:** a plain 2-layer GCN, hidden 128, no bias. It keeps the epoch with the
   best validation accuracy. This gives the `fp32_test` column.
2. **Calibration:** initializes the W/B column scales β, γ (Q-08, Q-09) and each degree's clip
   range (Q-04) from the FP32 model by minimizing quantization MSE (b = 4 for the degree tables). Accuracy right
   after calibration is reported as `ptq_test`, which is useful when debugging initialization.
3. **QAT:** fine-tunes with `L = CE + λ·(M − M_target)²` (Eq.(4)–(5)). Weights, b, β and γ are
   learned by gradient. The degree tables' clip ranges are re-fit by MSE at the current
   rounded b every 10 epochs (`--clip-refit`) and once more at the end. It reports the **last**
   epoch (`q_test`), so the bits and the accuracy describe the same model.

## Code map

| File | Contents |
|---|---|
| `data.py` | Planetoid loader (Cora/CiteSeer/PubMed/NELL) with a whitelist unpickler. Public split, self-loops, in-degree (Q-01, Q-14). NELL's X^(0) is sparse |
| `quant.py` | Eq.(2) quantizer with STE, the degree-indexed (α, b) table, 4-bit column quantizer, Q1.f fixed point |
| `model.py` | `QGCN` and `GraphOps`. The three normalization modes (Q-11) are described in the module docstring |
| `metrics.py` | Accuracy, average-bits definitions (Q-15), memory, partial-sum / lost-B checks |
| `train.py` | CLI: pretraining → calibration → QAT, multiple seeds, results JSON |
| `ptq_frontier.py` | Debug: PTQ accuracy vs fixed per-layer bitwidth (no QAT), MSE clip per b |

## Metrics in `results.json`

| Key | Meaning |
|---|---|
| `avg_bits_sparse`, `cr_sparse` | **Reported metric (Q-15):** hidden-layer inputs only, zeros cost 0 bits. CR = 32 / bits |
| `avg_bits_dense_all` | Dimension-weighted dense bits over all quantized inputs, including X⁽⁰⁾ (for comparison) |
| `memory_kb_eq4` | Eq.(4) memory with continuous b, as the loss sees it |
| `L{l}_density`, `L{l}_mean_bits` | Non-zero fraction and mean rounded bitwidth of layer l's quantized input |
| `L{l}_b_lost_all`, `L{l}_b_lost_topdeg` | Fraction of non-zero B entries quantized to 0, over all rows and over the top-1%-degree rows (the risk of `fold`) |
| `L{l}_psum_max_abs`, `L{l}_psum_overflow16` | Largest integer AU partial sum, and whether it exceeds signed 16-bit (Q-25). Only for `fold` and `per_source` |

## Implementation choices beyond the recorded decisions

These are ours. The paper is silent on all of them. Each was made after a measured problem,
and the measurement is given.

| Choice | Why (measured on Cora, seeds 0–1, no memory penalty, 2026-11-14) |
|---|---|
| Scales are learned as log α, log β, log γ | With linear α at Adam lr 0.01, α changed by about 100% per step and QAT accuracy collapsed (q_test ≈ 40%). |
| Degree tables learn the **clip range** c_d, with α_d = c_d / (2^(round(b_d)−1) − 1) (`--x-param range`, default; added 2026-11-15) | With α learned directly, lowering b at fixed α shrinks the clip range 2× per bit. On Cora's binary features, b 4→2 maps every 1 to α = 1/7, so the task loss blocked bit reduction on layer 0, where 2 bits are lossless. The memory penalty then crushed layer 1 instead (density 10%, q_test ≈ 75–78%). With the clip range, layer 0 goes to 2 bits even without a penalty: Cora λ=0 gives 80.83 ± 0.91 at 2.46 sparse bits, vs 80.57 at 3.80. The forward pass and the stored α are unchanged; this only changes optimization. |
| **MSE init for the W/B column scales** (`--wb-init mse`, default since 2026-11-16) | NELL: a few hub rows set each B column's max/7, and most entries round to 0. NELL seed 2 PTQ 21.0 → 56.0 (FP32 58.1); NELL QAT λ = 0, 3 seeds: 54.56 ± 9.01 → **58.93 ± 3.50** (FP32 60.85). Neutral on the rest (3 seeds, λ = 1e-6/1e-7): Cora 81.67 → 81.87, CiteSeer 70.33 → 69.93, PubMed 78.97 → 78.93. Amends Q-08/Q-09 init only. |
| `--lr-bits 2e-2` (default since 2026-11-15) | At 1e-2, b moves about 2 bits in 200 epochs, and the earlier runs ended before the memory converged. |
| **MSE clip init** for the degree tables (`--x-init mse`, default; added 2026-11-15) | max/7 rounds most entries of heavy-tailed inputs (TF-IDF, hidden ReLU outputs) to 0. PubMed PTQ (before QAT): 73.4 (max) → 79.5 (MSE), with FP32 at 79.1. c_d is picked from 39 fractions of the per-degree max (0.05…1.0) to minimize Σ(x − Q(x))² at b = 4. |
| **Clip ranges re-fit by MSE, not learned** (`--clip-refit 10`, default; added 2026-11-15) | With 60–140 training nodes the task loss is ~0.05 within 50 QAT epochs, so the learned clip range gets almost no useful gradient. It drifts upward (more values round to 0) and can't follow b when the penalty lowers it: the MSE-best range at 2 bits is far below the 4-bit one. PubMed λ = 1e-7: 57.2 (learned; also 57–62 with lr-alpha 1e-2/3e-2) → **78.17** (re-fit). Re-fitting all tables costs 0.45 s on PubMed. b is still learned (Eq.(5)); only how c is chosen changes, and the hardware still stores one α per degree. |
| M_target at the 2-bit dense floor (`--target-dense-bits 2.0`); λ is the knob | A "reachable" target (`--target-layer-bits 2,2.6`) with a large λ does not act as a clean constraint. Eq.(4) uses continuous b (Q-04), and layer 0's continuous b settles just below 2.5 (it still rounds to 2). With 11× more dimensions, that slack eats the hidden layer's whole budget, so layer 1 collapses to 2 bits (Cora: 75.4%, 0.21 bits). |
| α learning rate 1e-3 (`--lr-alpha`; unused with `--clip-refit`) | At 1e-2, α drifted upward and the *training* loss rose (0.24 → 1.07; test 79% → 54%). This is consistent with the STE gradient being biased on values that round to 0. At 1e-3 it was stable. |
| QAT weight learning rate 1e-3 (`--lr-qat-weights`) | 1e-2 overfits during the extra 200 epochs: 79.7/78.9 vs 80.4/79.5 test. |
| Row-normalized input features (`--feat-norm row`, default again since 2026-11-15) | 2026-11-14 we switched to raw features: with max/7 init, row normalization made many layer-0 values round to 0 (Cora QAT 75.9 vs 79.7). With MSE init + re-fit that problem is gone (Cora λ = 1e-7: row 81.77 ± 0.67 vs raw 81.00 ± 0.46, 3 seeds), and row is needed on CiteSeer (FP32 71.4 row vs 68.8 raw, 5 seeds; Tab.VI 71.1). One preprocessing for every dataset is simpler. **Amends Q-14** (back to Kipf's preprocessing). |
| Dropout after the quantizer | Applying dropout's 1/(1−p) rescale before the quantizer would make the quantizer see different values in training and eval, which biases α. |
| Reported epoch = last QAT epoch | b is still moving during QAT. Best-validation selection could report bits that differ from the final model. |

## Results log

### 2026-11-14 — Q-11 normalization comparison, Cora, 3 seeds, λ=1e-5, dense target 2.0 bits

| `--norm` | q_test (%) | avg_bits_sparse | CR | B lost, top-1% deg (L0 / L1) | max AU psum (L0 / L1) |
|---|---|---|---|---|---|
| fp32 (ideal Ã) | 80.27 ± 0.50 | 1.88 | 17.0× | 33% / 13% | n/a |
| fold | **80.93 ± 0.76** | 1.91 | 16.8× | **65% / 28%** | 378 / 420 |
| per_source f=8 | 80.37 ± 0.40 | 1.89 | 17.0× | 32% / 13% | **49,822 / 22,307 (overflows 16 bit)** |
| per_source f=4 | 80.77 ± 0.46 | 1.85 | 17.4× | 32% / 14% | 3,104 / 1,389 |

FP32 for the same seeds: 81.50 ± 0.82. Tab.VI reports 81.5 ± 0.7 (FP32) and 80.9 ± 0.6 at 1.70 bits.

- **As predicted, fold loses more B on high-degree rows** (65% vs 33%), but it costs no accuracy on
  Cora. It meets the Q-11 decision rule here.
- **per_source with f=8 overflows signed 16-bit partial sums.** f=4 fits, with no accuracy loss.
- **Caveat:** Cora's maximum degree is 169. The rule also requires NELL and Reddit, where
  high-degree nodes are common, before Q-11 is closed. Three seeds is a small sample.

### 2026-11-15 — λ sweep (Q-13), all `--norm fold`, 3 seeds, `--lr-bits 2e-2`, 200 QAT epochs

**Cora: matches Tab.VI.** M_target is at the 2-bit dense floor (`--target-dense-bits 2.0`), clip-range parameterization.

| λ | q_test (%) | avg_bits_sparse | L1 mean b | L1 density |
|---|---|---|---|---|
| 0 | 80.83 ± 0.91 | 2.46 | 3.58 | 0.65 |
| **1e-7** | **81.00 ± 1.14** | **1.60 (CR 20×)** | 2.95 | 0.53 |
| 3e-7 | 80.73 ± 0.49 | 1.65 | 2.98 | 0.55 |
| 1e-6 | 80.43 ± 0.55 | 1.62 | 2.95 | 0.54 |
| 3e-6 | 80.33 ± 0.21 | 1.51 | 2.87 | 0.51 |
| 1e-5 | 77.17 ± 0.38 | 0.44 | 2.15 | 0.18 |

Tab.VI: 80.9 ± 0.6 at 1.70 bits, so λ = 1e-7 meets the acceptance check (accuracy within std, at
≤ the reported bits). λ from 1e-7 to 3e-6 is a flat plateau; 1e-5 falls off a cliff. Layer 0 is
at 2 bits in every run (lossless for binary features).

Before the clip-range fix (`--x-param alpha`), the same sweep gave 78.4–79.6%. Each fix is
documented in "Implementation choices" above.

**CiteSeer and PubMed: not matched yet with the recipe of that day** (resolved later the same day,
see "Quantizer fix" below). Both lost about 3 points to QAT *even with
λ = 0*, and the memory penalty is far more sensitive than on Cora.

| Dataset (FP32 ours / Tab.VI) | Setting | q_test (%) | avg_bits_sparse |
|---|---|---|---|
| CiteSeer raw (68.5 / 71.1) | λ = 0 | 67.37 ± 0.31 | 2.95 |
| | λ = 1e-8 | 66.57 ± 1.14 | 2.09 |
| | λ = 1e-7 | 65.97 ± 1.31 | 1.26 |
| CiteSeer row-norm (71.1 / 71.1) | λ = 0 | 68.00 ± 1.90 | 1.52 |
| | λ = 1e-7 | 22.87 (collapse) | 0.09 |
| PubMed (79.1 / 78.9) | λ = 0 | 75.93 ± 0.31 | 2.00 |
| | λ = 1e-8 | 63.57 ± 8.75 | 0.39 |
| | λ = 1e-7 | 25.33 (collapse) | 0.02 |

Diagnosis so far:
- **Not the normalization.** PubMed with λ = 0: fold 75.93, ideal Ã 75.20, per_source f=4 75.47.
- **Mostly the degree-aware X quantizer** (`--quant-parts` ablation, PubMed, λ = 0):
  W only 79.20, B only 78.10, W+B 78.27, X only 76.70, all 75.93.
- **Learning the X tables helps, but not enough.** With the tables frozen at their
  initialization: 74.27. Learning only b: 76.60. Learning only the range: 74.77.
  α lr 1e-2 / 3e-2: 74.50 / 76.67 (within noise).
- **The penalty is sensitive** because Eq.(4)'s gradient on each degree's b scales with the
  number of nodes of that degree × the layer dimension, and Adam turns any persistent push into
  ~lr per step. PubMed has 7× Cora's nodes, and its real-valued TF-IDF features are not
  lossless at 2 bits, unlike Cora's binary ones.
- **Feature preprocessing matters on CiteSeer.** FP32 is 68.8 raw vs 71.4 row-normalized
  (5 seeds; Tab.VI: 71.1). The "raw features" choice above was measured on Cora only, and under
  the old α parameterization.
- Next things to try (done, see below): better X initialization than max/7 (e.g. MSE- or
  percentile-based clip), longer QAT at a lower bits learning rate, and Eq.(4) normalized per layer.

### 2026-11-15 — Q-11 normalization comparison on NELL, 3 seeds, λ = 0

Setting: `--feat-norm row --dropout 0.1 --wd 1e-5 --epochs-fp32 400 --lr-bits 2e-2` (our default
recipe gets only 57.2% FP32 on NELL; see Q-14). NELL's maximum degree is 4,550, vs 169 for Cora.

| `--norm` | Per-seed q_test (%) | Mean q_test | Max AU psum L0 / L1 (mean) |
|---|---|---|---|
| FP32 model | 62.5 / 61.9 / 58.1 | 60.85 ± 2.40 | — |
| fp32 (ideal Ã) | 56.6 / 58.3 / 48.8 | 54.56 ± 5.05 | n/a |
| **fold** | **61.2 / 61.9 / 44.5** | **55.87 ± 9.87** | 6,756 / 5,354 |
| per_source f=4 | 39.8 / 38.5 / 37.3 | 38.53 ± 1.29 | 29,897 / 6,817 |

- **per_source f=4 fails on NELL.** Q1.4 rounds w = d^(−1/2) to 0 for degree > 1,024. On NELL
  that is 21 hub nodes, which source 38,125 of 317,305 edges (12%), so their messages vanish.
  At f=6 no w rounds to 0 (max relative error 33%). But w_int is 4× larger, so the L0 partial
  sum, already 29,897 at f=4, would exceed the signed 16-bit limit (32,767). This is an
  estimate (4 × 29,897 ≈ 120k); f=6 was not run. So no f satisfies both precision and the
  16-bit accumulator on NELL.
- **fold ≥ ideal Ã on average** (55.87 vs 54.56), and better on 2 of 3 seeds. An
  interpretation, not tested: with unscaled B, hub rows set the per-column γ, and the
  1/√d fold shrinks them so γ fits typical rows.
- **Seed 2 is unstable in every mode.** PTQ (before QAT) drops to 21–39%, and QAT recovers
  only part of it. Mostly fixed by MSE init of the W/B column scales (2026-11-16 below).
- Per-seed time: about 22 minutes with 3 jobs in parallel on 4 CPUs.

### 2026-11-15 — Quantizer fix: MSE clip init + MSE re-fit (Q-04 amended); Tab.VI matched

**Final, 10 seeds** (`--norm fold --feat-norm row --x-init mse --clip-refit 10 --lr-bits 2e-2
--wb-init max`, M_target at the 2-bit dense floor, 200 QAT epochs). avg bits = `avg_bits_sparse`.
These runs predate the `--wb-init mse` default (2026-11-16); the 3-seed checks of that change
are in "Implementation choices" and are neutral on these datasets.

| Dataset | FP32 (ours) | λ = 1e-7 | λ = 1e-6 | Tab.VI |
|---|---|---|---|---|
| Cora | 81.65 ± 0.46 | 81.70 ± 0.61 @ 1.99 | **81.67 ± 0.40 @ 1.08** | 80.9 ± 0.6 @ 1.70 |
| CiteSeer | 71.23 ± 0.54 | 70.59 ± 0.80 @ 1.27 | **70.35 ± 0.69 @ 0.97** | 70.6 ± 1.1 @ 1.87 |
| PubMed | 78.99 ± 0.60 | 79.05 ± 0.86 @ 0.88 | **78.97 ± 0.57 @ 0.85** | 78.3 ± 0.6 @ 2.50 |

λ = 1e-6 meets the acceptance check on all three (accuracy within the paper's std, at fewer
bits). At λ = 1e-7 Cora's accuracy matches but its bits (1.99) are above 1.70.

**Observation (not a decision):** at λ = 1e-6 every degree entry in both layers ends at b = 2
(L0 and L1 mean b = 2.00). The degree-aware part that remains is the per-degree α (clip range).
So on these three datasets a single 2-bit width plus a per-degree scale reaches Tab.VI accuracy.
Whether this argues for dropping per-degree bitwidths in hardware is open: it should be checked
on GIN/GraphSage and Reddit first, and the paper's Fig.10 shows non-uniform bits.

**How we got there** (3 seeds unless noted, fold, λ = 0 unless noted):

1. **Max-based init was the PTQ loss.** max/7 rounds most heavy-tailed entries to 0.
   PubMed PTQ (accuracy right after calibration): 73.4 → **79.5** with MSE init (FP32 79.1).
   PubMed QAT λ = 0: 75.93 → 78.00. Cora unchanged (81.07). CiteSeer raw 67.37 → 67.87.
2. **Low bits are not the problem.** PTQ only, all degrees at a fixed b, MSE clip per b
   (`python -m mega_pt.ptq_frontier`, 2 seeds; raw PubMed): PubMed (4,2) bits 78.7 at 0.95 sparse bits, (2,2) 77.2 at 0.54;
   CiteSeer row (2,3) 71.2 at 2.34, (2,2) 67.9 at 1.21. Yet QAT with λ = 1e-7 collapsed
   PubMed to 57.2 at (2.2, 2.1) bits.
3. **The learned clip range was what broke QAT.** With b frozen (`--lr-bits 0`) at (2,2),
   PubMed QAT fell from 77.4 to 71.9 and sparse bits dropped 0.54 → 0.31: the range grew and
   zeroed more of the hidden layer. Freezing the range too (`--lr-alpha 0`): PubMed (4,2)
   78.97, CiteSeer row (2,3) 70.93, Cora (2,2) 80.40. A higher range lr did not help
   (PubMed λ = 1e-7 with lr-alpha 1e-2 / 3e-2: 62.0 / 59.3).
4. **Fix: re-fit the range by MSE at the current round(b)** every 10 epochs instead of learning
   it. b keeps learning (Eq.(5)), and the range now follows it:

| Setting (3 seeds) | Before (learned range) | After (`--clip-refit 10`) |
|---|---|---|
| PubMed raw, λ = 0 | 78.00 @ 1.94 | 78.97 ± 0.60 @ 3.17 |
| PubMed raw, λ = 1e-7 | 57.20 @ 0.21 | **78.17 ± 0.40 @ 0.89** |
| PubMed row, λ = 1e-7 | — | 78.97 ± 1.27 @ 0.89 |
| CiteSeer row, λ = 0 | 69.77 @ 3.79 | 71.00 ± 0.46 @ 4.17 |
| CiteSeer row, λ = 1e-7 | 48.70 @ 0.35 | **70.33 ± 1.40 @ 1.32** |
| Cora raw, λ = 1e-7 | 81.00 @ 1.60 | 81.00 ± 0.46 @ 1.34 |
| Cora row, λ = 1e-7 | — | 81.77 ± 0.67 @ 1.99 |

5. **Row normalization is fine again**, so it is the default for every dataset (Q-14 amended).

### 2026-11-16 — NELL with the quantizer fix, 3 seeds, λ = 0

Same NELL recipe as the Q-11 comparison above (`--dropout 0.1 --wd 1e-5 --epochs-fp32 400`),
now with the defaults `--x-init mse --clip-refit 10 --feat-norm row --norm fold`.

| W/B init | Per-seed q_test (%) | Mean q_test | Sparse bits |
|---|---|---|---|
| max/7 (Q-09 as first decided) | 59.3 / 60.2 / 44.2 | 54.56 ± 9.01 | 1.24 |
| **MSE (`--wb-init mse`)** | **60.7 / 61.2 / 54.9** | **58.93 ± 3.50** | 1.53 |

FP32: 62.5 / 61.9 / 58.1 (60.85 ± 2.40).
- The degree-aware X tables alone were not NELL's problem: seed 2, PTQ only, quantizing X only
  gives 56.9 (FP32 58.1), but W+B only gives 48.2 and everything 21.0 (`--quant-parts`,
  `--epochs-qat 0 --seed-start 2`). MSE W/B init brings "everything" to 56.0.
- Still open: seed 2 loses ~3 points in QAT, and an MSE re-fit of the clip ranges *right after
  calibration* (0 QAT epochs) dropped seed 2 from 56.0 to 44.9 — on NELL's 186 classes, MSE on
  the hidden layer seems to be a weaker proxy than on the other datasets. Not investigated
  further; NELL has no GCN accuracy in Tab.VI.

## Known limitations / next steps

- **b convergence:** Adam moves b by about `--lr-bits` per step, so 200 epochs allow about 2 bits
  of change. With a strong penalty, the memory may still be heading toward M_target at the
  end (check the `mem` log lines). Use more `--epochs-qat` or a higher `--lr-bits` if needed.
- **λ / M_target sweep (Q-13):** GCN matches Tab.VI on Cora, CiteSeer and PubMed with one
  setting, λ = 1e-6 and M_target at the 2-bit floor (10 seeds; 2026-11-15 "Quantizer fix").
- **Scope:** GCN only. GIN and GraphSage (Q-12) and Reddit are not implemented yet. NELL is
  loaded (sparse X^(0), PyG construction, Tab.II sizes reproduced).
- **NELL source:** the official hosts are blocked from our cloud environment, so the loader
  fetches a GitHub LFS mirror and verifies each file's sha256 (see `data.py`).
- **Scales treated as exact:** the scale multiplications (α·β, γ, d^(−1/2)) are assumed
  exact. Their hardware precision is still open (Q-16).
