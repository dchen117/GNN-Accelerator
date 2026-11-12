# MEGA Knowledge Base

Reference notes on **"MEGA: A Memory-Efficient GNN Accelerator Exploiting Degree-Aware
Mixed-Precision Quantization"**, Z. Zhu, F. Li, G. Li, Z. Liu, Z. Mo, Q. Hu, X. Liang, J. Cheng.
HPCA 2024. Source used: **arXiv:2311.09775v1** (16 Nov 2023), 15 PDF pages.

The goal is to implement MEGA as a project baseline. This knowledge base records only what
the paper says, plus clearly labeled derivations and interpretations. It does **not** fill gaps.
Gaps are listed in [`04-open-questions.md`](04-open-questions.md) and are decided there
(or in our implementation notes), never silently.

## Files

| File | Contents |
|---|---|
| [`01-technical-summary.md`](01-technical-summary.md) | Problem, motivation, contributions, method, architecture overview |
| [`02-math-reference.md`](02-math-reference.md) | Eq. (1)–(5), symbol table, equation dependency graph, worked examples |
| [`03-hardware-mapping.md`](03-hardware-mapping.md) | How each algorithm step maps to engines, units, buffers, data formats, Algorithm 1 |
| [`04-open-questions.md`](04-open-questions.md) | Unresolved details and inconsistencies (Q-01 … ), and which implementation path each blocks |
| [`05-eval-reference.md`](05-eval-reference.md) | Datasets, model configs, accuracy/CR targets, hardware config, MEGA's reported results |

## Labeling rules (apply to every file)

| Tag | Meaning |
|---|---|
| **[STATED]** | The paper says this explicitly. Always followed by a citation; key claims also quote the paper. |
| **[DERIVED]** | Follows mechanically from stated facts. The arithmetic/logic is shown so it can be checked. |
| **[INTERP]** | Our reading of something ambiguous. Gives the reason, and links a Q-ID when it matters for implementation. |
| **[UNSPECIFIED]** | The paper does not say. Links to a Q-ID. Never filled in. |

Unlabeled text is only organizational (headings, pointers between files).

## Citation format

`p.5 §IV Eq.(2)`, `p.7 §V-C-2 Fig.10(c)`, `p.10 Tab.IV`, `p.8 Alg.1 L9`.

- **p.N = PDF page N of the arXiv v1 file** (1–15). The arXiv version has no printed page
  numbers; page numbers in the IEEE HPCA proceedings version will be different.
- Section numbers follow the paper (§III-B-1 = Section III, subsection B, item 1).
- Alg.1 Lk = line k of Algorithm 1.

## Paper map (PDF page → content)

| Page | Content |
|---|---|
| p.1 | Abstract, §I Introduction, Fig.1 (cycle/energy breakdown of prior accelerators) |
| p.2 | §I contd., contributions, Fig.2 (GNN illustration), §II-A GNNs |
| p.3 | Eq.(1), A(XW) order, §II-B Quantization, §II-C HW acceleration, Tab.I, Fig.3, §III-A |
| p.4 | §III-A contd., §III-B challenges (storage, irregular access), Figs.4–6 |
| p.5 | Fig.7, **§IV Degree-Aware Quantization, Eq.(2)–(5)**, Fig.8, §V-A start |
| p.6 | §V-A workflow, Fig.9, **§V-B Adaptive-Package**, §V-C-1 Decoder |
| p.7 | Figs.10–11, Decoder contd., **§V-C-2 Combination Unit**, §V-D start |
| p.8 | Fig.12, **§V-D Aggregation Engine / Encoder**, **§V-E Condense-Edge**, **Algorithm 1** |
| p.9 | Fig.13 (Condense Unit), Condense-Edge contd., Tab.II–III, §VI-A setup |
| p.10 | **Tab.IV (MEGA config)**, Tab.V, **Tab.VI (accuracy/CR)**, Tab.VII, §VI-B accuracy, §VI-C-1 speedup |
| p.11 | Figs.14–18, §VI-C contd. (speedup, DRAM, energy), §VI-D start |
| p.12 | Figs.19–22, §VI-D ablation, §VI-E sensitivity, §VII-1 |
| p.13 | §VII contd. (Condense-Edge without partitioning, GAT), §VIII Conclusion, references [1]–[18] |
| p.14–15 | References [19]–[65] |

## Index of key claims

| Claim | Where |
|---|---|
| DRAM stalls up to 86.2% of cycles (HyGCN); DRAM energy 90.2% (HyGCN, Reddit) | p.2 §I, Fig.1 (p.1) |
| Nodes with higher in-degree have larger average feature values after aggregation | p.3 §III-A, Fig.3 |
| Learned per-degree (scale, bitwidth) quantization | p.5 §IV Eq.(2) |
| W quantized to 4 bits for all GNNs, per-column learnable scale | p.5 §IV |
| Memory-size penalty in loss | p.5 §IV Eq.(4)–(5) |
| Heterogeneous Combination / Aggregation engines, A(XW) order | p.5 §V-A, Fig.8 |
| Adaptive-Package: Mode(2b) + Bitwidth(3b) + Val Array; lengths 64/128/192 bits | p.6 §V-B |
| Bit-serial row-product combination; weights reused across bits | p.7 §V-C-2 |
| Outer-product aggregation, 16-bit partial sums, CSC edges | p.8 §V-D |
| Condense-Edge scheduling (Alg.1), Condense Unit | p.8–9 §V-E, Fig.13 |
| 28nm, 1GHz, 392KB on-chip, 1.869 mm², 194.98 mW | p.9 §VI-A-3, p.10 Tab.IV |
| Up to 18.6× compression (Cora GCN) vs DQ-INT4 8× | p.2, p.10 Tab.VI |
| 38.3×/7.1×/4.0×/3.6× speedup; 47.6×/7.2×/5.4×/4.5× energy (vs HyGCN/GCNAX/GROW/SGCN) | p.1 Abstract, p.10–11 §VI-C |

## Implementation-path focus

Agreed first target: **whichever is faster between a PyTorch quantization model and a
cycle-accurate simulator**. Assessment ([DERIVED] from what each path needs):
the **PyTorch model** is faster. It depends only on §IV (Eq.2–5) plus public datasets,
and Tab.VI gives accuracy/bitwidth targets. The simulator depends on most of §V,
where many open questions remain. `04-open-questions.md` tags each question with
the path(s) it blocks: `[PT]` = PyTorch model, `[SIM]` = simulator.

## Working notes

- Verified transcriptions: Fig.14/16/17 MEGA values reproduce the paper's stated geomeans
  (38.3×, 108.1×, 47.6×) to within rounding. See `05-eval-reference.md`.
- Figures 9, 14, 16–17 were read from high-resolution renders, not just the page images.
