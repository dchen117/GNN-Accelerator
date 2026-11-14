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
PubMed 88648):

```bash
.venv/bin/python -m mega_pt.data
```

Unit tests (each test names the equation or Q-ID it checks):

```bash
.venv/bin/python -m pytest -q tests
```

One configuration, 10 seeds:

```bash
.venv/bin/python -m mega_pt.train --dataset cora --norm fold --lam 1e-5 --target-dense-bits 2.0
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
| `data.py` | Planetoid loader with a whitelist unpickler. Public split, self-loops, in-degree (Q-01, Q-14) |
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

## Known limitations / next steps

- **b convergence:** Adam moves b by about `--lr-bits` per step, so 200 epochs allow about 2 bits
  of change. With a strong penalty, the memory may still be heading toward M_target at the
  end (check the `mem` log lines). Use more `--epochs-qat` or a higher `--lr-bits` if needed.
- **λ / M_target sweep (Q-13):** not done yet. A dense target of 2.0 bits at λ=1e-5 gave a
  sparse average of 1.76 on Cora, close to Tab.VI's 1.70.
- **Scope:** GCN only. GIN and GraphSage (Q-12), and NELL and Reddit, are not implemented yet.
- **Scales treated as exact:** the scale multiplications (α·β, γ, d^(−1/2)) are assumed
  exact. Their hardware precision is still open (Q-16).
