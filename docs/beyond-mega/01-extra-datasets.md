# Extra evaluation datasets (beyond MEGA)

**Status (2026-11-17):** decided. The user chose Roman-Empire, Amazon-Ratings and
City-Networks (Paris) as extra evaluation datasets, on top of MEGA's own (Cora, CiteSeer,
PubMed, NELL, Reddit). Loaders are in `mega_pt/data.py`; run with `--dataset roman-empire |
amazon-ratings | city-paris`.

This file is **not** part of the MEGA knowledge base (`docs/mega-kb/`), which records only
what the paper says. Everything here is our own choice. MEGA evaluates on none of these
datasets, so there are no paper numbers to match.

## Why these datasets

- **Goal:** later, test whether **residual connections** change accuracy, and how they
  interact with MEGA's degree-aware quantization. Residuals are not implemented yet.
- **Where residuals should matter:**
  - Heterophilic graphs, where neighbors tend to have *different* labels. Here a node's own
    features have to survive aggregation.
  - Long-range tasks, which need many layers, where deep stacks over-smooth without
    residuals.
- **Evidence:** Luo et al., *Classic GNNs are Strong Baselines* (NeurIPS 2024) ablated
  residual connections in classic GNNs.
  - On Roman-Empire, GCN loses 16.43 points without residuals. Their NeurIPS slides say
    24.43, so check the table before citing the exact figure.
  - The drop on Roman-Empire is the largest of their datasets, and Minesweeper also drops
    noticeably.
  - We have not seen an Amazon-Ratings number; it was chosen for its degree profile (below).
- **Long-range:** City-Networks (Liang et al., NeurIPS 2025) labels each road junction by
  its 16-hop eccentricity. A model needs a receptive field of many hops, so many layers,
  which is the setting where residuals are normally needed.

## What "compatible with our setup" required

| Requirement | Why |
|---|---|
| One graph, node classification, full-batch | Our pipeline (and MEGA's Eq.(1) model) is single-graph transductive. LRGB's node tasks (PascalVOC-SP, COCO-SP) are thousands of small graphs. |
| Fits CPU / 15 GB RAM | Our environment has 4 cores and no GPU. |
| Downloadable from our cloud environment | GitHub raw and Amazon S3 work. SNAP (ogbn-arxiv), Zenodo, Dropbox, data.pyg.org and data.dgl.ai are blocked. |
| Accuracy metric | Our metrics are accuracy. Minesweeper, Tolokers and Questions use ROC-AUC. |
| A spread of node degrees | Otherwise MEGA's degree-aware bit allocation has nothing to exploit. |

## The three datasets

These numbers were measured by our loader after symmetrizing, dropping duplicates and
adding one self-loop per node (Q-01). Degree includes the self-loop.

| | Roman-Empire | Amazon-Ratings | City-Networks Paris |
|---|---|---|---|
| Source | Platonov et al., ICLR 2023 | Platonov et al., ICLR 2023 | Liang et al., NeurIPS 2025 |
| Nodes | 22,662 (words) | 24,492 (products) | 114,127 (road junctions) |
| Directed edges, no self-loops | 65,854 | 186,100 | 365,022 |
| Features | 300 (fastText embedding) | 300 (fastText mean of review text) | 37 ("augmented": coordinates, road-segment stats) |
| Classes | 18 (syntactic role) | 5 (rating) | 10 (quantiles of 16-hop eccentricity) |
| Degree min / mean / max | 3 / 3.9 / 15 | 6 / 8.6 / 133 | 2 / 4.2 / 9 |
| Adjusted homophily | −0.05 (heterophilic) | +0.14 (mildly heterophilic) | — (long-range) |
| Splits | 10 fixed, 50 / 25 / 25 % | 10 fixed, 50 / 25 / 25 % | 1 fixed, 10 / 10 / 80 % |
| Metric | accuracy | accuracy | accuracy |

Adjusted homophily follows Platonov et al.: 0 means edges join same-label nodes as often as
chance would, and below 0 means less often.

**What each dataset brings:**
- **Roman-Empire:** the clearest documented residual effect, but a narrow degree range
  (max 15). MEGA's per-degree tables only get about 13 distinct degrees to work with.
- **Amazon-Ratings:** heterophilic with a heavy-tailed degree distribution, so it exercises
  MEGA's degree-aware bits the way Cora and PubMed do.
- **City-Networks Paris:**
  - It is the long-range case: road networks have huge diameters.
  - Its degrees are nearly flat (2–9), so the degree tables have little to exploit; a
    near-uniform bit allocation is the expected outcome, not a bug.
  - It is the smallest of the four cities. The others (Shanghai 184K, LA 241K, London 569K
    nodes) can be added with `load_city(<name>)` once a fixed hash is pinned in `EXTRA_FILES`.

## Protocol decisions (ours)

| Topic | Decision | Why / trade-off |
|---|---|---|
| Files | Platonov's `.npz` and the City-Networks `.tar.gz` straight from the authors' GitHub repos, pinned by sha256 in `data.EXTRA_FILES`. | Same files as PyG's `HeterophilousGraphDataset` and `CityNetwork`. A hash mismatch stops the run instead of silently changing the data. |
| Graph | Bidirected, duplicates merged, one self-loop per node. | Platonov's code does `to_bidirected`, and road segments are undirected. One degree definition (Q-01) for every dataset. |
| Splits | Run seed *s* uses published split *s* mod 10 (heterophilic sets). City: the one published split. | Matches the authors' protocol of reporting the mean over the 10 splits when we run 10 seeds. With 3 seeds, only splits 0–2 are used. |
| City labels / features | 16-hop labels, augmented features. | The defaults of the authors' and PyG's loaders. |
| Feature preprocessing | Per-column z-score over all nodes (`--feat-norm std`, the default for these datasets). | See the next section. |
| Models / recipe | Unchanged: GCN, GIN-0, GraphSage-mean (Q-12), 2 layers, Q-14 training recipe and Q-13 loss. | We first measure MEGA as specified. Residuals and depth come later and will be separate, labeled changes. |

### Why z-score instead of row normalization (Q-14's choice for MEGA's datasets)

- **Row normalization doesn't apply here.** Q-14 row-normalizes MEGA's bag-of-words
  features. That is meaningless for the new sets:
  - Roman-Empire and Amazon-Ratings rows are word embeddings, about half of the values
    negative, and some rows sum to ~0, so dividing by the sum blows up.
  - Paris mixes units: 0/1 flags, counts up to 15, segment lengths up to ~885, and raw
    latitude/longitude (~48.8 / ~2.3) with a spread of only ~0.03.
- **Neither set of authors normalizes**, but both put an FP32 input linear layer in front of
  the GNN. MEGA has no such layer: X⁽⁰⁾ goes straight into the degree-aware quantizer, which
  has one scale per degree, not per column. Raw Paris features would let the length column
  set the clip range and flush the 0/1 flags to zero at 4 bits or less.
- **Hardware:** z-scoring is an offline, one-time change of the input data, the same for
  MEGA and for our design. It adds nothing to the hardware.
- **Labels:** statistics are taken over all nodes, as is usual for transductive
  preprocessing. It uses features only, never labels.
- **Reddit (`--feat-norm none`):** the files ship already standardized (each column has mean
  ≈ 0 and std ≈ 1, from GraphSAGE's preprocessing), so we keep them as distributed.
- **Validated:** see the FP32 none-vs-std check in `mega_pt/README.md`.

## Considered and not used

| Dataset | Why not |
|---|---|
| Minesweeper | Synthetic grid where every degree is ≤ 8, ROC-AUC metric. Strong residual effect in Luo et al., so it is a candidate if we add an AUC metric. |
| Tolokers, Questions | ROC-AUC (binary). Heavy-tailed degrees (max 2,138 / 1,539) would suit MEGA. Candidates if we add AUC. |
| Chameleon / Squirrel (filtered) | 890 / 2,223 nodes, results noisy. The unfiltered versions leak duplicate nodes between train and test (Platonov et al.). |
| Texas, Cornell, Wisconsin, Actor | 183–251 nodes (WebKB), or weak signal (Actor). |
| LRGB (PascalVOC-SP, COCO-SP, Peptides) | Many-graph or graph-level tasks; Dropbox-hosted (blocked). |
| ogbn-arxiv | Good large homophilic check where residuals are standard, but the SNAP host does not resolve here. |
| City-Networks Shanghai / LA / London | Same task, larger. Add after Paris works. |

## Sources

- Luo, Shi, Wu. *Classic GNNs are Strong Baselines: Reassessing GNNs for Node
  Classification*. NeurIPS 2024. https://arxiv.org/abs/2406.08993
- Platonov et al. *A critical look at the evaluation of GNNs under heterophily: Are we
  really making progress?* ICLR 2023. Data: https://github.com/yandex-research/heterophilous-graphs
- Liang, Sáez de Ocáriz Borde, Sripathmanathan, Bronstein, Dong. *Towards Quantifying
  Long-Range Interactions in Graph Machine Learning: a Large Graph Dataset and a
  Measurement*. NeurIPS 2025. https://arxiv.org/abs/2503.09008. Data:
  https://github.com/LeonResearch/City-Networks
