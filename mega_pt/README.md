# mega_pt — PyTorch model of MEGA's Degree-Aware quantization (GCN, GIN, GraphSage)

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

The datasets download automatically on first use to `data/`, which is gitignored. Reddit
is a 1.4 GB download (from DGL's S3 bucket; `data.dgl.ai` and `data.pyg.org` are blocked here)
and needs about 12 GB of RAM to train. The extra evaluation datasets (Roman-Empire,
Amazon-Ratings, City-Networks Paris) are described in
[`docs/beyond-mega/01-extra-datasets.md`](../docs/beyond-mega/01-extra-datasets.md).
Every non-Planetoid file is pinned by sha256 (`data.EXTRA_FILES`).

## Run

Quick check that the data loads (edge counts should match Tab.II: Cora 10556, CiteSeer 9104,
PubMed 88648, NELL 251550):

```bash
.venv/bin/python -m mega_pt.data
```

Any dataset by name (Reddit should print 114615892 edges and a 153431/23831/55703 split):

```bash
.venv/bin/python -m mega_pt.data reddit roman-empire amazon-ratings city-paris
```

Unit tests (each test names the equation or Q-ID it checks):

```bash
.venv/bin/python -m pytest -q tests
```

One configuration, 10 seeds. The defaults select `--norm fold --feat-norm row --x-init mse
--x-param alpha --clip-refit 0 --lr-bits 2e-2`: α and b are both learned by gradient, as the
paper states (default since 2026-11-16, Q-04). Best λ so far (3 seeds): 1e-6 for CiteSeer and
PubMed, 1e-5 for Cora:

```bash
.venv/bin/python -m mega_pt.train --dataset citeseer --lam 1e-6 --target-dense-bits 2.0
```

GIN and GraphSage (Q-12; Tab.III hidden sizes 128 / 256 are the defaults):

```bash
.venv/bin/python -m mega_pt.train --model gin --dataset citeseer --lam 1e-6 --target-dense-bits 2.0
```

```bash
.venv/bin/python -m mega_pt.train --model sage --dataset cora --lam 1e-6 --target-dense-bits 2.0
```

All three use Eq.(1)'s single-W layer and differ only in Ã: GIN-0 sums over A+I, GraphSage
averages over the node plus 25 sampled in-neighbors (`--samples`). GraphSage draws a new
sample every training epoch and evaluates on one fixed sample seeded by the run seed.

GraphSage on Reddit (Tab.VI). Training is inductive by default for Reddit (`--inductive on`:
the training graph is induced by the training nodes). One seed takes about 70 minutes on 3
CPU threads. `--fp32-cache` reuses the FP32-pretrained model when only QAT settings change:

```bash
.venv/bin/python -m mega_pt.train --model sage --dataset reddit --lam 1e-8 --target-dense-bits 2.0 --seeds 3 --fp32-cache runs/fp32_cache
```

Extra evaluation datasets (ours, not in MEGA). Roman-Empire and Amazon-Ratings have 10
published splits, and seed s uses split s % 10. Features are z-scored per column by default
for these (`--feat-norm std`):

```bash
.venv/bin/python -m mega_pt.train --dataset roman-empire --lam 1e-6 --target-dense-bits 2.0
```

The earlier MSE re-fit setting (matches Tab.VI at fewer bits, but every degree ends at 2 bits)
is kept as an option:

```bash
.venv/bin/python -m mega_pt.train --dataset citeseer --lam 1e-6 --target-dense-bits 2.0 --x-param range --clip-refit 10
```

Debug ablation, quantizing only some tensors (here: only the degree-aware features):

```bash
.venv/bin/python -m mega_pt.train --dataset pubmed --norm fold --lam 0 --seeds 3 --quant-parts x
```

Deeper models and residual connections (ours, not MEGA; `docs/beyond-mega/02-residual.md`).
`--layers 4` gives F → 128 → 128 → 128 → C. `--residual` adds the identity residual on the
hidden → hidden layers (here layers 1 and 2), using the layer's quantized input.
`--fp32-cache` is needed if SGCN will run the model: sgcnsim simulates the FP32 network:

```bash
.venv/bin/python -m mega_pt.train --dataset roman-empire --layers 4 --residual --lam 1e-7 --target-dense-bits 2.0 --seeds 3 --save-model --out-dir runs/res4 --fp32-cache runs/res4/fp32_cache
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
3. **QAT:** fine-tunes with `L = CE + λ·(M − M_target)²` (Eq.(4)–(5)). Weights, b, α, β and γ
   are learned by gradient (Adam). With the option `--clip-refit K`, the degree tables' clip
   ranges are instead re-fit by MSE at the current rounded b every K epochs and once more at
   the end. It reports the **last** epoch (`q_test`), so the bits and the accuracy describe
   the same model.

## Code map

| File | Contents |
|---|---|
| `data.py` | Planetoid loader (Cora/CiteSeer/PubMed/NELL) with a whitelist unpickler. Public split, self-loops, in-degree (Q-01, Q-14). NELL's X^(0) is sparse. Reddit and the extra datasets: sha256-pinned downloads, vectorized edge building (edges grouped by destination), per-column z-score, multiple splits, train-induced subgraph |
| `quant.py` | Eq.(2) quantizer with STE, the degree-indexed (α, b) table, 4-bit column quantizer, Q1.f fixed point |
| `model.py` | `QGCN` (the Eq.(1) layer for all three models) and `GraphOps` (Ã per model, GraphSage neighbor sampling, Q-12). The three normalization modes (Q-11) are described in the module docstring. Optional identity residual on hidden → hidden layers (`residual_layers`, Q-36) |
| `metrics.py` | Accuracy, average-bits definitions (Q-15), memory, partial-sum / lost-B checks |
| `train.py` | CLI: pretraining → calibration → QAT, multiple seeds, results JSON |
| `ptq_frontier.py` | Debug: PTQ accuracy vs fixed per-layer bitwidth (no QAT), MSE clip per b |
| `hw_stats.py` | Simulator design numbers from a saved model (`train.py --save-model`): integer-only reference forward (`int_forward`, the computation the hardware does, Q-16), fixed-point scale formats, accumulator ranges (Q-22), BSE utilization of the two tile-feeding schemes (Q-23). `python -m mega_pt.hw_stats runs/hw/<run>` |
| `part_stats.py` | METIS partition statistics for Condense-Edge (Q-27): parts sized to the Aggregation Buffer, external (subgraph, node) pairs, batches touched per node. `python -m mega_pt.part_stats cora pubmed` |

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
| `--lsq` (off): LSQ gradient scaling g = 1/sqrt(N·Q_P) on the degree tables' learned scale; `--alpha-opt sgd` to use SGD for it | Tested 2026-11-16 for fidelity (paper cites LSQ [13]). Did not help; see Results log. |
| **Degree tables learn α_d directly** (`--x-param alpha --clip-refit 0`, default since 2026-11-16) | Fidelity: the paper says the scale is learned (Eq.(2), p.4). The re-fit below matched Tab.VI but every degree ended at 2 bits; learned α keeps degree-varying bits (2–5 in the hidden layer). Cost: CiteSeer 67.8 vs Tab.VI 70.6, Cora 2.58 vs 1.70 bits. See Results log "Fidelity check". |
| Option: degree tables learn the **clip range** c_d, with α_d = c_d / (2^(round(b_d)−1) − 1) (`--x-param range`; was the default 2026-11-15 → 11-16) | With α learned directly, lowering b at fixed α shrinks the clip range 2× per bit. On Cora's binary features, b 4→2 maps every 1 to α = 1/7, so the task loss blocked bit reduction on layer 0, where 2 bits are lossless. The memory penalty then crushed layer 1 instead (density 10%, q_test ≈ 75–78%). With the clip range, layer 0 goes to 2 bits even without a penalty: Cora λ=0 gives 80.83 ± 0.91 at 2.46 sparse bits, vs 80.57 at 3.80. The forward pass and the stored α are unchanged; this only changes optimization. |
| **MSE init for the W/B column scales** (`--wb-init mse`, default since 2026-11-16) | NELL: a few hub rows set each B column's max/7, and most entries round to 0. NELL seed 2 PTQ 21.0 → 56.0 (FP32 58.1); NELL QAT λ = 0, 3 seeds: 54.56 ± 9.01 → **58.93 ± 3.50** (FP32 60.85). Neutral on the rest (3 seeds, λ = 1e-6/1e-7): Cora 81.67 → 81.87, CiteSeer 70.33 → 69.93, PubMed 78.97 → 78.93. Amends Q-08/Q-09 init only. |
| `--lr-bits 2e-2` (default since 2026-11-15) | At 1e-2, b moves about 2 bits in 200 epochs, and the earlier runs ended before the memory converged. |
| **MSE clip init** for the degree tables (`--x-init mse`, default; added 2026-11-15; applies to α mode too) | max/7 rounds most entries of heavy-tailed inputs (TF-IDF, hidden ReLU outputs) to 0. PubMed PTQ (before QAT): 73.4 (max) → 79.5 (MSE), with FP32 at 79.1. c_d is picked from 39 fractions of the per-degree max (0.05…1.0) to minimize Σ(x − Q(x))² at b = 4. |
| Option: **clip ranges re-fit by MSE, not learned** (`--clip-refit 10`; was the default 2026-11-15 → 11-16, replaced by learned α for fidelity) | With 60–140 training nodes the task loss is ~0.05 within 50 QAT epochs, so the learned clip range gets almost no useful gradient. It drifts upward (more values round to 0) and can't follow b when the penalty lowers it: the MSE-best range at 2 bits is far below the 4-bit one. PubMed λ = 1e-7: 57.2 (learned; also 57–62 with lr-alpha 1e-2/3e-2) → **78.17** (re-fit). Re-fitting all tables costs 0.45 s on PubMed. b is still learned (Eq.(5)); only how c is chosen changes, and the hardware still stores one α per degree. |
| M_target at the 2-bit dense floor (`--target-dense-bits 2.0`); λ is the knob | A "reachable" target (`--target-layer-bits 2,2.6`) with a large λ does not act as a clean constraint. Eq.(4) uses continuous b (Q-04), and layer 0's continuous b settles just below 2.5 (it still rounds to 2). With 11× more dimensions, that slack eats the hidden layer's whole budget, so layer 1 collapses to 2 bits (Cora: 75.4%, 0.21 bits). |
| α learning rate 1e-3 (`--lr-alpha`; unused with `--clip-refit`; the α-mode results used it) | At 1e-2, α drifted upward and the *training* loss rose (0.24 → 1.07; test 79% → 54%). This is consistent with the STE gradient being biased on values that round to 0. At 1e-3 it was stable. |
| QAT weight learning rate 1e-3 (`--lr-qat-weights`) | 1e-2 overfits during the extra 200 epochs: 79.7/78.9 vs 80.4/79.5 test. |
| Row-normalized input features (`--feat-norm row`, default again since 2026-11-15) | 2026-11-14 we switched to raw features: with max/7 init, row normalization made many layer-0 values round to 0 (Cora QAT 75.9 vs 79.7). With MSE init + re-fit that problem is gone (Cora λ = 1e-7: row 81.77 ± 0.67 vs raw 81.00 ± 0.46, 3 seeds), and row is needed on CiteSeer (FP32 71.4 row vs 68.8 raw, 5 seeds; Tab.VI 71.1). One preprocessing for every dataset is simpler. **Amends Q-14** (back to Kipf's preprocessing). |
| **Per-column z-score** for the extra datasets (`--feat-norm std`, their default; added 2026-11-17) | Row normalization is meaningless for embeddings (Roman-Empire / Amazon-Ratings rows are ~50% negative, some sum to ~0) and for Paris's mixed units. FP32 GCN, 3 seeds: Roman-Empire 47.43 vs 42.73 raw; Amazon-Ratings 47.90 vs 43.34 raw; City-Networks Paris see Results log. Reddit stays raw: it ships already standardized. Details: `docs/beyond-mega/01-extra-datasets.md`. |
| **Inductive Reddit training** (`--inductive on`, Reddit default) | Q-14's decision (the standard GraphSAGE protocol). The training graph keeps only edges between training nodes. The Eq.(4) memory term then uses the *evaluation* graph's degrees, because val/test nodes are isolated (degree 1) in the training graph and would misstate the stored graph's memory. |
| **Floyd sampling** on graphs with > 2·10⁷ edges (Reddit) | The sort-based sampler took 6 s and ~7 GB per Reddit epoch; Floyd's algorithm, vectorized over nodes, takes 1.7 s. Same distribution (exact uniform k-subsets; unit-tested). Smaller graphs keep the sort path, so earlier results are unchanged. |
| Dropout mask from `torch.rand` for inputs with > 5·10⁷ entries (Reddit X⁽⁰⁾) | `F.dropout` is effectively single-threaded on CPU: 2.4 s vs 1.0 s per call. Same dropout; smaller tensors keep `F.dropout` (bit-reproducible). |
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

### 2026-11-16 — Fidelity check: can the scale be learned by gradient? (LSQ, α-direct)

Why: the paper says the scales are *learned* (p.4 §III-A). The MSE re-fit above replaces
that, which departs from a stated detail (fidelity rule, `docs/mega-kb/04-open-questions.md`).
All runs: `--clip-refit 0` (scale learned by gradient), MSE init, W/B MSE init, 3 seeds (0–2),
dense target 2.0 bits. Tab.VI: Cora 80.9 @ 1.70, CiteSeer 70.6 @ 1.87, PubMed 78.3 @ 2.50.

**1. LSQ gradient scaling on the clip range c (`--x-param range`), λ = 1e-6.** q_test %:

| Variant | CiteSeer | PubMed |
|---|---|---|
| MSE re-fit (current default), reference | 70.27 @ 0.96 | 78.93 @ 0.86 |
| c learned, Adam | 48.6 | 28.2 |
| c learned, Adam + `--lsq` | 49.1 | 28.9 |
| c learned, SGD(0.9) + `--lsq`, lr 1e-2 | 63.4 | 30.9 |
| same, lr 1e-1 / lr 1 (seed 0 only) | 63.9 / 63.7 | 39.7 / 36.8 |

- LSQ does not help. With Adam it is a no-op by construction: Adam divides each entry's
  gradient by its own running RMS, which cancels a per-entry constant g. With SGD it changes
  step sizes, but no learning rate fixes it.
- Cause, from the final tables (PubMed seed 0): the learned c ends at about 0.10–0.12 in
  every variant, close to its 4-bit init. The MSE-best c after b drops to 2 is about 0.04.
  The range does not follow b, so at 2 bits most inputs round to 0 (sparse bits 0.04–0.07).
  The training loss is near 0 after FP32 pretraining on 60–140 labels, so the task gradient
  on c is too weak to move it 2–3×. Gradient scaling can only resize that signal.

**2. Learning α directly (`--x-param alpha`, the literal Eq.(2) parameterization),** Adam, no
re-fit, no LSQ. Here lowering b at fixed α also shrinks the clip range, so the clipping cost
reaches b's gradient directly.

| Dataset | λ | q_test % | sparse bits | dense bits |
|---|---|---|---|---|
| Cora | 1e-7 | 81.50 ± 0.36 | 3.87 | 3.04 |
| Cora | 1e-6 | 81.10 ± 0.26 | 3.26 | 2.35 |
| Cora | 1e-5 | 80.97 ± 0.49 | 2.58 | 2.14 |
| CiteSeer | 1e-7 | 69.43 ± 2.40 | 3.36 | 2.74 |
| CiteSeer | 3e-7 | 67.80 ± 1.47 | 3.11 | 2.38 |
| CiteSeer | 1e-6 | 67.80 ± 0.61 | 2.05 | 2.04 |
| PubMed | 1e-7 | 79.63 ± 0.31 | 2.67 | 2.47 |
| PubMed | 1e-6 | 79.73 ± 0.75 | 1.57 | 2.14 |

- Learned bits now **vary with degree**, mostly in the hidden layer (e.g. PubMed λ = 1e-6,
  layer 1, degrees 1–12: 4,3,2,3,3,3,3,3,3,3,4,3; Cora layer 1 mostly 4–5). With the re-fit,
  every present degree ended at 2 bits.
- vs Tab.VI: PubMed matches (better accuracy, fewer bits). Cora matches in accuracy at λ = 1e-5
  but at 2.58 vs 1.70 bits. CiteSeer is 1–3 points short (PTQ already loses 1.7 of FP32's 71.1;
  QAT loses more).
- **Decision (user, 2026-11-16): learned α is the default** (`--x-param alpha --clip-refit 0`),
  for fidelity to MEGA as a baseline: the scale is learned as stated, and the per-degree
  bitwidths stay non-uniform. The re-fit remains available as `--x-param range --clip-refit 10`.
  Still to do: tune CiteSeer (lr-alpha, λ) and Cora (λ between 1e-5 and 1e-4), then 10-seed runs.

### 2026-11-16 — GIN and GraphSage (Q-12), first results

Default settings (learned α, MSE init, fold, row features, 200 + 200 epochs), M_target at the
2-bit dense floor, 3 seeds (0–2), no λ tuning beyond the grid below. GIN = GIN-0 with one
linear layer and sum over A+I, hidden 128. GraphSage = mean over self + 25 sampled
neighbors with a single W, hidden 256. q_test %, avg bits = `avg_bits_sparse`:

| Model / dataset | FP32 (ours) | PTQ | λ = 0 | λ = 1e-7 | λ = 1e-6 | λ = 1e-5 | Tab.VI |
|---|---|---|---|---|---|---|---|
| GIN Cora | 78.07 ± 1.87 | 75.67 | 76.73 @ 3.73 | 77.13 @ 3.29 | 76.80 @ 2.91 | **76.63 ± 0.12 @ 1.92** | 77.8 ± 1.6 @ 2.37 (FP32 77.6) |
| GIN CiteSeer | 67.83 ± 0.06 | 62.63 | 63.87 @ 4.29 | 61.67 @ 2.82 | 59.80 @ 2.22 | **65.33 ± 0.90 @ 1.57** | 65.1 ± 1.7 @ 2.54 (FP32 66.1) |
| GraphSage Cora | 81.70 ± 0.17 | 82.07 | 81.63 @ 4.61 | 81.97 @ 4.14 | 81.57 @ 3.60 | **81.53 ± 0.71 @ 1.81** | 79.7 ± 1.0 @ 3.40 (FP32 79.7) |

- **λ = 1e-5 meets the acceptance check on all three rows**: accuracy within Tab.VI's std
  at fewer bits. GIN Cora is 1.2 below the paper's mean (inside its ±1.6).
- **Per-degree bits stay non-uniform** in the hidden layer (seed 0, λ = 1e-5, layer 1,
  degrees 1–15): GIN Cora 4,3,4,4,4,3,4,4,4,2,4,4,4,3,3; GIN CiteSeer 2–3; GraphSage Cora
  3–4. Layer 0 (binary / row-normalized input) goes to 2 bits except degree 1.
- **GIN CiteSeer is the hard case, as in the paper.** PTQ at 4 bits already loses 5.2
  points (Tab.I: DQ-INT4 loses 5.3). Accuracy is not monotonic in λ (λ = 1e-6 is worst,
  59.8). We have not investigated this; with 3 seeds it may partly be noise.
- **GraphSage FP32 is 2 points above Tab.VI** (81.7 vs 79.7). Our GraphSage is the Q-12
  single-W simplification, without the separate self weight or the concat. That could
  explain the gap, but we have not tested it. Quantized accuracy matches our own FP32.
- Integer partial sums stay far below 16-bit: max |psum| is 479 (GIN, sum aggregation)
  and 148 (GraphSage).
- Not run: 10 seeds, GIN PubMed and GraphSage CiteSeer/PubMed (not in Tab.VI), Reddit
  (blocked).

### 2026-11-17 — GraphSage on Reddit (Tab.VI row), inductive, 3 seeds

Settings: same defaults as above (learned α, MSE init, fold, 200 + 200 epochs, M_target at
the 2-bit dense floor), GraphSage-mean with a single W, hidden 256, 25 sampled neighbors,
`--feat-norm none` (features ship standardized), inductive training on the train-induced
subgraph, evaluation on the full graph. About 1.7 h per seed on 3 CPU threads (1.1 h when the
FP32 model comes from `--fp32-cache`), 11.5 GB peak RAM.

**λ sweep, seed 0 only** (FP32 93.72; q_test %, avg bits = `avg_bits_sparse` / `avg_bits_dense_all`):

| λ | q_test | sparse bits | dense bits | per-degree bits |
|---|---|---|---|---|
| 0 | 93.70 | 1.37 | 4.66 | no memory penalty |
| 1e-9 | 93.57 | 0.74 | 2.28 | layer 1 non-uniform: degrees 1–4 at 3 bits, 5–25 at 2, degree 26 at 3 |
| 1e-8 | 93.50 | 0.49 | 2.00 | all degrees at 2 bits |
| 1e-7 | 93.54 | 0.49 | 2.00 | all degrees at 2 bits |

- λ ≥ 1e-8 drives every degree to the 2-bit floor. That is uniform 2 bits, the outcome the
  fidelity constraint rules out as "MEGA", even though accuracy barely moves. **We chose
  λ = 1e-9**: the largest λ in the grid that still gives degree-dependent bits. Reddit needs
  a much smaller λ than Cora (1e-5) because the memory term scales with N·F (233K nodes ×
  602 features), so λ·M is about 1,000× larger at the same λ.
- Degree 26 means "full sample" (25 neighbors + self). 88% of nodes are there, so its bit
  width dominates the averages.

**Result, λ = 1e-9, seeds 0–2:**

| | FP32 (ours) | PTQ (4 bit) | MEGA (ours) | avg bits sparse / dense | CR (sparse) | Tab.VI |
|---|---|---|---|---|---|---|
| GraphSage Reddit | 93.70 ± 0.01 | 93.43 ± 0.02 | **93.58 ± 0.02** | 0.66 ± 0.08 / 2.28 | 48.9× | 95.6 ± 0.2 @ 2.74, 11.7× (FP32 95.2) |

- **Quantization costs 0.12 points against our own FP32**, at fewer average bits than
  Tab.VI's 2.74. On that measure the result matches the paper's claim.
- **Our absolute accuracy is 2 points below Tab.VI**, because our FP32 model is 1.5 points
  below theirs (93.7 vs 95.2). The likely cause is the Q-12 single-W GraphSage, which has no
  separate self weight and no concat. Standard GraphSAGE-mean reaches about 95 on Reddit
  with them. We have not tested this. On Cora the same simplification came out 2 points
  *above* Tab.VI, so the direction is dataset-dependent. Tab.VI's MEGA (95.6) is also above
  its own FP32 (95.2), which our runs never show on Reddit.
- **Integer partial sums:** max |psum| is 182 on both layers and all seeds, far inside
  16 bits (Q-25), even though Reddit is the dataset with hub nodes. Sampling caps the
  in-degree at 26.
- **Q-11 on Reddit:** GraphSage's normalization is t = 1/d into the Encoder and s = 1 (no
  source factor), so fold and per_source compute the same thing. The Q-11 choice is a GCN
  question and cannot be reconfirmed with Tab.VI's Reddit model, which is GraphSage only.
- Not run: 10 seeds (std is already 0.02 over 3), GCN/GIN on Reddit (not in Tab.VI), a
  transductive variant.

### 2026-11-17 — Extra datasets (Roman-Empire, Amazon-Ratings, City-Networks Paris), 3 seeds

Why these datasets and how they are prepared: `docs/beyond-mega/01-extra-datasets.md`.
MEGA has no numbers for them, so the reference is our own FP32 model. Settings are the
defaults (learned α, MSE init, fold, `--feat-norm std`, 2 layers, 200 + 200 epochs,
M_target at the 2-bit dense floor). Seeds 0–2 use published splits 0–2 (Paris has one
split). q_test % @ `avg_bits_sparse`; dense bits in brackets where they are above 2.0:

| Model / dataset | FP32 (ours) | PTQ (4 bit) | λ = 0 | λ = 1e-7 | λ = 1e-6 | λ = 1e-5 |
|---|---|---|---|---|---|---|
| GCN Roman-Empire | 47.43 ± 0.72 | 46.01 | 46.88 @ 2.27 [3.97] | **46.53 ± 0.24 @ 1.32 [2.71]** | 46.72 @ 1.13 [2.21] | 46.28 @ 0.83 |
| GCN Amazon-Ratings | 47.90 ± 0.18 | 47.75 | 48.52 @ 1.95 [4.85] | **47.92 ± 0.43 @ 0.71 [2.15]** | 47.76 @ 0.52 | 47.89 @ 0.47 |
| GCN Paris | 27.01 ± 0.09 | 18.47 | 26.43 @ 1.63 [4.94] | 24.18 ± 0.19 @ 0.56 | 24.19 @ 0.56 | 24.18 @ 0.57 |
| GIN Roman-Empire | 45.17 ± 0.43 | 39.78 | 44.02 @ 1.87 [3.64] | **46.49 ± 0.49 @ 1.14 [2.61]** | 45.09 @ 0.79 | 45.21 @ 0.70 |
| GIN Amazon-Ratings | 45.11 ± 0.68 | 29.47 | 39.11 @ 1.41 [3.84] | **40.02 ± 0.32 @ 0.76 [2.13]** | 39.16 @ 0.60 | 39.56 @ 0.59 |
| GIN Paris | 26.96 ± 0.07 | 19.38 | 25.97 @ 1.06 [4.32] | 24.92 ± 0.16 @ 0.38 | 24.79 @ 0.37 | 24.88 @ 0.37 |
| GraphSage Roman-Empire | 45.46 ± 0.55 | 44.35 | 45.85 @ 2.20 [4.53] | **45.81 ± 1.05 @ 1.04 [2.69]** | 46.68 @ 0.76 | 46.75 @ 0.76 |
| GraphSage Amazon-Ratings | 47.98 ± 0.26 | 47.71 | 48.58 @ 1.97 [4.92] | **48.33 ± 0.37 @ 0.63 [2.14]** | 48.18 @ 0.48 | 48.24 @ 0.47 |
| GraphSage Paris | 27.63 ± 0.24 | 18.74 | 26.99 @ 1.66 [4.90] | 24.60 ± 0.10 @ 0.57 | 24.73 @ 0.57 | 24.56 @ 0.57 |

Bold: the operating point picked by the Reddit rule (the largest λ that still gives
degree-dependent bits). Raw per-run numbers: `runs/ext/*/results.json`.

- **Roman-Empire and Amazon-Ratings quantize like MEGA's own datasets** for GCN and
  GraphSage. At λ = 1e-7 they stay within 0.9 points of FP32, at 0.6–1.3 sparse bits, with
  non-uniform per-degree bits (dense 2.1–2.7). GIN on Roman-Empire beats its FP32 at
  λ = 1e-7. Differences between λ ≤ 1e-6 are mostly within the seed std.
- **λ ≥ 1e-6 gives uniform 2 bits on nearly every row**, the same pattern as Reddit. Those
  columns are a uniform-2-bit reference, not MEGA's degree-aware scheme.
- **Paris costs 2.4–3 points at every λ > 0, for all three models**, and every λ > 0 ends
  at 2 bits for every degree. With degrees 2–9 there is nothing for degree-aware
  allocation to exploit, as expected (see the dataset doc). With the λ = 0 bit budget
  (about 5 dense bits) the loss shrinks to 0.6–1.0. Paris is therefore precision-sensitive,
  and MEGA's memory penalty pushes it to a bit width it can't take. PTQ at 4 bits loses
  8–9 points, the worst of any dataset. Its 37 z-scored features include coordinates, where
  small differences carry the label.
- **GIN on Amazon-Ratings loses about 6 points even at λ = 0**, so the bit budget is not the
  cause. PTQ already loses 15.6. Amazon-Ratings has the heaviest degree tail of the three
  (max 133), and GIN sums over neighbors without normalization, so high-degree aggregates
  are much larger than low-degree ones (max |psum| up to 479). Not investigated; the
  per-degree scale ranges are the first thing to check.
- **FP32 accuracies are low in absolute terms** (Paris 27%, 10 classes). That is expected
  for 2-layer models with no residuals or input/output MLPs on these benchmarks, and it is
  the gap the planned residual experiments target. We have not compared against the
  datasets' published baselines, which use deeper models with residuals.
- Integer partial sums stay at or below 479 (16 bits is ample).
- Not run: 10 seeds/splits, residual variants, deeper models.

### 2026-11-20 — Simulator design numbers (`hw_stats.py`), seed 0, one model each

Purpose: back the Q-16, Q-22 and Q-23 decisions in `docs/mega-kb/04-open-questions.md` with
measurements. Models were trained with the defaults at λ = 1e-5 (Cora GCN/GIN/GraphSage, GIN
CiteSeer), 1e-6 (GCN CiteSeer/PubMed) and 1e-7 (GIN Amazon-Ratings), `--save-model`; outputs
are in `runs/hw/<run>/hw_stats.json`.

**Integer-only reference and the scale format (Q-16)**, test accuracy (%):

| Model | fake-quant | integer, exact scales | prediction agreement | 8-bit | 12-bit | 16-bit | dyadic 6-bit | dyadic 8-bit |
|---|---|---|---|---|---|---|---|---|
| GCN Cora | 81.2 | 81.2 | 1.0000 | 81.3 | 81.2 | 81.2 | 81.5 | 81.4 |
| GCN CiteSeer | 68.2 | 68.2 | 1.0000 | 67.6 | 68.2 | 68.2 | 67.5 | 68.2 |
| GCN PubMed | 80.5 | 80.5 | 0.9999 | 80.2 | 80.6 | 80.5 | 80.2 | 80.4 |
| GIN Cora | 76.7 | 76.7 | 1.0000 | 76.3 | 76.6 | 76.7 | 76.6 | 76.3 |
| GIN CiteSeer | 65.4 | 65.4 | 1.0000 | 64.6 | 65.1 | 65.4 | 65.4 | 65.2 |
| GraphSage Cora | 80.9 | 80.9 | 1.0000 | 81.2 | 80.9 | 81.0 | 81.0 | 81.2 |
| GIN Amazon-Ratings | 40.2 | 40.2 | 0.9994 | 40.5 | 40.2 | 40.3 | 40.7 | 39.9 |

- **What the columns mean:**
  - "8/12/16-bit": every factor table (α′_d, c_j, e_d, γ_j) is stored as unsigned fixed
    point with one binary point per table.
  - "dyadic k-bit": each entry has its own k-bit mantissa and shift.
- **The integer reference reproduces the fake-quant model** (it is the bit-exact target for
  the simulator).
- **16-bit tables are exact to within ±0.1**, and 12 bits stays within 0.3. Decision:
  16-bit (Q-16).
- **Table ranges are narrow:** each table spans at most 6.6 octaves (log2 max/min).

**Accumulators and BSE utilization (Q-22, Q-23).** Per layer input: L0 = X⁽⁰⁾, L1 = hidden.
"Running max" is the largest |value| the row accumulator ever holds; "tile" is one tile
pass of ≤ 32 non-zeros. Cycles are per node per 32-column output block; util = useful BSE
cycles / available.

| Model | Layer | density | mean bits | running max | tile max | AU psum max | slice cyc / util | nnz, scan 128 | nnz, scan 256 |
|---|---|---|---|---|---|---|---|---|---|
| GCN Cora | L0 | 0.013 | 2.00 | 129 | 126 | 394 | 20.7 / 0.06 | 12.0 / 0.09 | 6.0 / 0.19 |
| | L1 | 0.690 | 3.66 | 1,878 | 878 | 340 | 12.6 / 0.80 | 11.8 / 0.86 | 11.8 / 0.86 |
| GCN CiteSeer | L0 | 0.009 | 2.00 | 330 | 330 | 170 | 47.8 / 0.04 | 29.0 / 0.07 | 15.0 / 0.13 |
| | L1 | 0.696 | 3.23 | 911 | 478 | 123 | 11.3 / 0.80 | 10.5 / 0.86 | 10.5 / 0.86 |
| GCN PubMed | L0 | 0.066 | 2.00 | 128 | 101 | 314 | 8.0 / 0.26 | 4.0 / 0.51 | 3.1 / 0.67 |
| | L1 | 0.608 | 2.84 | 1,256 | 647 | 291 | 9.2 / 0.75 | 8.0 / 0.86 | 8.0 / 0.86 |
| GIN Cora | L1 | 0.715 | 3.68 | 1,294 | 659 | 335 | 13.2 / 0.79 | 12.3 / 0.85 | 12.3 / 0.85 |
| GraphSage Cora | L1 | 0.659 | 3.13 | 2,735 | 765 | 132 | 21.5 / 0.77 | 18.0 / 0.92 | 18.0 / 0.92 |
| GIN Amazon-Ratings | L0 | 0.722 | 2.01 | 755 | 296 | 186 | 21.4 / 0.64 | 14.6 / 0.93 | 14.6 / 0.93 |
| | L1 | 0.281 | 2.20 | 875 | 468 | 236 | 4.1 / 0.60 | 3.5 / 0.70 | 3.5 / 0.70 |

- **Every measured value fits the worst-case-safe widths** of Q-22 (16-bit Shifter-Acc,
  32-bit Tile Accumulator) and the 16-bit AU sums, with large margins.
- **Feeding tiles by non-zeros ("nnz") beats fixed 32-feature slices everywhere.**
  - Hidden layers gain 5–15 points of utilization.
  - Sparse bag-of-words X⁽⁰⁾ gains 2–4× in cycles, once the bitmap scan is 256 bits/cycle.
  - At 128 bits/cycle the scan limits X⁽⁰⁾. Hidden layers (≤ 256 features) are never
    scan-bound at 256.
- **Low X⁽⁰⁾ utilization is intrinsic**: Cora averages 18 non-zeros out of 1,433 features,
  but the decoder still has to scan the whole bitmap.
- Not measured here: Reddit (dense standardized features, 602-wide).
- **GCN NELL (λ = 1e-7) collapsed** to 1.2% test accuracy: X⁽⁰⁾ kept on average 1 non-zero
  per node. NELL QAT is known to be noisy (2026-11-15 notes). So its accuracy and range
  numbers are not used.
  - Its structural number still holds: with a 61,278-wide bitmap, X⁽⁰⁾ takes 240 scan cycles
    per node at 256 bits/cycle (480 with fixed slices) against about 2 compute cycles.
  - NELL's layer 0 is therefore bound entirely by bitmap scanning in MEGA's format (Q-19, Q-23).

### 2026-11-25 — 4-layer GCN ± identity residual (Q-36, R1), 3 seeds

`--layers 4` (F → 128 → 128 → 128 → C), with and without `--residual` (layers 1 and 2), recipe
unchanged: `--target-dense-bits 2.0`; λ: Cora 1e-5, CiteSeer and PubMed 1e-6, extra datasets 1e-7;
`--save-model --out-dir runs/res4 --fp32-cache runs/res4/fp32_cache`. Columns: FP32 / PTQ / QAT
test % and sparse avg bits; the 2-layer column is the QAT row of the λ-sweep / extra-datasets logs.

| Dataset | Residual | FP32 | PTQ | QAT | avg bits | 2-layer QAT |
|---|---|---|---|---|---|---|
| Cora | no | 78.67 ± 1.72 | 77.77 ± 1.92 | 78.13 ± 0.47 | 1.32 | 81.67 |
| Cora | yes | 80.23 ± 0.29 | 79.83 ± 0.61 | **79.33 ± 0.12** | 1.34 | |
| CiteSeer | no | 66.53 ± 2.02 | 63.63 ± 2.53 | 60.70 ± 2.38 | 2.14 | 70.35 |
| CiteSeer | yes | 70.20 ± 0.72 | 69.27 ± 1.03 | **62.60 ± 1.44** | 2.55 | |
| PubMed | no | 75.83 ± 0.70 | 76.13 ± 0.49 | 77.70 ± 0.52 | 0.82 | 78.97 |
| PubMed | yes | 76.33 ± 1.14 | 76.97 ± 1.66 | **78.47 ± 0.47** | 0.86 | |
| Roman-Empire | no | 32.94 ± 0.63 | 31.46 ± 0.80 | 33.69 ± 0.17 | 0.84 | 46.53 |
| Roman-Empire | yes | 53.84 ± 0.98 | 52.14 ± 1.27 | **49.55 ± 0.62** | 0.90 | |
| Amazon-Ratings | no | 46.68 ± 0.22 | 46.33 ± 0.19 | **46.73 ± 0.11** | 0.53 | 47.92 |
| Amazon-Ratings | yes | 47.28 ± 0.33 | 46.81 ± 0.50 | 46.51 ± 0.23 | 0.47 | |
| Paris | no | 33.89 ± 0.11 | 22.23 ± 0.47 | **27.35 ± 0.31** | 0.46 | 24.18 |
| Paris | yes | 33.44 ± 0.24 | 20.89 ± 1.27 | 26.78 ± 0.16 | 0.48 | |

- **The residual fixes depth.** Plain 4-layer GCN loses to the 2-layer one everywhere, and
  collapses on Roman-Empire (33.7 vs 46.5). With the residual, Roman-Empire's 4-layer model
  beats the 2-layer one (49.6 vs 46.5; FP32 53.8 vs 47.4), and Cora, CiteSeer and PubMed gain
  0.8–1.9 points over plain 4-layer. Amazon-Ratings is a wash (−0.2, about 1σ).
- **Paris (long range) gains from depth itself, not from the residual:** plain 4-layer FP32
  33.9 vs 2-layer 27.0, QAT 27.4 vs 24.2. Four hops reach further; the residual adds nothing
  (−0.6). Its QAT drop is the largest (FP32 → QAT −6.5; PTQ −11.7): every input sits at the
  2-bit dense floor (`avg_bits_dense_all` 2.00), the same gap as the 2-layer Paris runs.
- **QAT costs more at depth on CiteSeer and Roman-Empire:** FP32 → QAT drops 7.6 and 4.3
  points with the residual (2-layer: under 1). The memory penalty is set for 2 layers; we kept
  the recipe on purpose (decision 6), so this is a known gap, not tuned.
- **Bit-exact:** `python -m megasim.check` passes on seed 0 of all six residual runs
  and on plain Cora (integer reference = functional model). The integer reference's accuracy
  differs from fake-quant q_test by ≤ 0.05 points (Roman-Empire 48.89 vs 48.87, Amazon-Ratings
  46.25 vs 46.30, Paris 26.85 vs 26.84). The integer reference uses 16-bit fixed-point scale tables, fake-quant uses
  float factors; a handful of borderline nodes flip.

## Known limitations / next steps

- **b convergence:** Adam moves b by about `--lr-bits` per step, so 200 epochs allow about 2 bits
  of change. With a strong penalty, the memory may still be heading toward M_target at the
  end (check the `mem` log lines). Use more `--epochs-qat` or a higher `--lr-bits` if needed.
- **λ / M_target sweep (Q-13):** with the default learned α (3 seeds, 2026-11-16 "Fidelity
  check"): PubMed beats Tab.VI at λ = 1e-6, Cora matches its accuracy at λ = 1e-5 but with
  more bits, CiteSeer is 1–3 points short. Reddit GraphSage (λ = 1e-9) is 2 points below Tab.VI
  in absolute terms but only 0.12 below our FP32. Not tuned further yet; no 10-seed runs yet. The
  MSE re-fit option matched Tab.VI on all three at λ = 1e-6 (10 seeds; 2026-11-15).
- **Scope:** GCN, GIN and GraphSage (Q-12). Reddit is loaded from DGL's S3 bucket (Tab.II
  sizes reproduced); NELL is loaded (sparse X^(0), PyG construction, Tab.II sizes
  reproduced). Extra evaluation datasets: Roman-Empire, Amazon-Ratings, City-Networks Paris.
- **NELL source:** the official hosts are blocked from our cloud environment, so the loader
  fetches a GitHub LFS mirror and verifies each file's sha256 (see `data.py`).
- **Scales treated as exact:** the scale multiplications (α·β, γ, d^(−1/2)) are assumed
  exact. Their hardware precision is still open (Q-16).
