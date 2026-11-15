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

One configuration, 10 seeds (the Cora setting that matches Tab.VI):

```bash
.venv/bin/python -m mega_pt.train --dataset cora --norm fold --lam 1e-7 --target-dense-bits 2.0 --lr-bits 2e-2
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
2. **Calibration:** initializes α per degree, β and γ from the FP32 model using max/7
   (Q-04, Q-08, Q-09). Accuracy right after calibration is reported as `ptq_test`, which is
   useful when debugging initialization.
3. **QAT:** fine-tunes with `L = CE + λ·(M − M_target)²` (Eq.(4)–(5)). It reports the **last**
   epoch (`q_test`), so the bits and the accuracy describe the same model.

## Code map

| File | Contents |
|---|---|
| `data.py` | Planetoid loader (Cora/CiteSeer/PubMed/NELL) with a whitelist unpickler. Public split, self-loops, in-degree (Q-01, Q-14). NELL's X^(0) is sparse |
| `quant.py` | Eq.(2) quantizer with STE, the degree-indexed (α, b) table, 4-bit column quantizer, Q1.f fixed point |
| `model.py` | `QGCN` and `GraphOps`. The three normalization modes (Q-11) are described in the module docstring |
| `metrics.py` | Accuracy, average-bits definitions (Q-15), memory, partial-sum / lost-B checks |
| `train.py` | CLI: pretraining → calibration → QAT, multiple seeds, results JSON |

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
| `--lr-bits 2e-2` in the sweeps (default still 1e-2) | At 1e-2, b moves about 2 bits in 200 epochs, and the earlier runs ended before the memory converged. |
| M_target at the 2-bit dense floor (`--target-dense-bits 2.0`); λ is the knob | A "reachable" target (`--target-layer-bits 2,2.6`) with a large λ does not act as a clean constraint. Eq.(4) uses continuous b (Q-04), and layer 0's continuous b settles just below 2.5 (it still rounds to 2). With 11× more dimensions, that slack eats the hidden layer's whole budget, so layer 1 collapses to 2 bits (Cora: 75.4%, 0.21 bits). |
| α learning rate 1e-3 (`--lr-alpha`) | At 1e-2, α drifted upward and the *training* loss rose (0.24 → 1.07; test 79% → 54%). This is consistent with the STE gradient being biased on values that round to 0. At 1e-3 it was stable. |
| QAT weight learning rate 1e-3 (`--lr-qat-weights`) | 1e-2 overfits during the extra 200 epochs: 79.7/78.9 vs 80.4/79.5 test. |
| Raw input features (`--feat-norm none`, default) | Row normalization (Kipf's preprocessing) makes layer-0 values small and uneven, so many round to 0 (L0 density fell from 1.27% to 0.84%). QAT then scored 75.9 vs 79.7. Raw Cora/CiteSeer features are binary and exact at any α. FP32 accuracy: 82.4 (raw) vs 81.9 (row). This **amends Q-14**. |
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

**CiteSeer and PubMed: not matched yet (open).** Both lose about 3 points to QAT *even with
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
- Next things to try: better X initialization than max/7 (e.g. MSE- or percentile-based
  clip), longer QAT at a lower bits learning rate, and Eq.(4) normalized per layer.

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
  only part of it. NELL QAT is affected by the same open problem as CiteSeer and PubMed:
  QAT losses even at λ = 0.
- Per-seed time: about 22 minutes with 3 jobs in parallel on 4 CPUs.

## Known limitations / next steps

- **b convergence:** Adam moves b by about `--lr-bits` per step, so 200 epochs allow about 2 bits
  of change. With a strong penalty, the memory may still be heading toward M_target at the
  end (check the `mem` log lines). Use more `--epochs-qat` or a higher `--lr-bits` if needed.
- **λ / M_target sweep (Q-13):** Cora matches Tab.VI (λ = 1e-7). CiteSeer and PubMed do not yet
  (see the 2026-11-15 results log).
- **Scope:** GCN only. GIN and GraphSage (Q-12) and Reddit are not implemented yet. NELL is
  loaded (sparse X^(0), PyG construction, Tab.II sizes reproduced).
- **NELL source:** the official hosts are blocked from our cloud environment, so the loader
  fetches a GitHub LFS mirror and verifies each file's sha256 (see `data.py`).
- **Scales treated as exact:** the scale multiplications (α·β, γ, d^(−1/2)) are assumed
  exact. Their hardware precision is still open (Q-16).
