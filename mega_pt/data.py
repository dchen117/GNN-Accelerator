"""Datasets: MEGA's (Cora, CiteSeer, PubMed, NELL, Reddit) and our extra evaluation sets
(Roman-Empire, Amazon-Ratings, City-Networks Paris; see docs/beyond-mega/01-extra-datasets.md).

Decisions implemented here (see docs/mega-kb/04-open-questions.md):
  Q-01  degree = in-degree with self-loops counted.
  Q-14  standard Planetoid public split (Kipf & Welling / Yang et al.); Reddit: the standard
        GraphSAGE split (inductive training is handled in train.py).
Every graph is made undirected, de-duplicated, and given exactly one self-loop per node,
whatever the source format, so the degree definition (Q-01) is the same for all datasets.

The raw files are the original Planetoid pickles from github.com/kimiyoung/planetoid.
They are Python-2 pickles, and unpickling untrusted data can run arbitrary code, so we
load them with a whitelist unpickler that only accepts the few classes these files use
(numpy arrays, a scipy CSR matrix, list, defaultdict).
"""

from __future__ import annotations

import os
import pickle
import tarfile
import zipfile
import urllib.request
from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp
import torch

PLANETOID_URL = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/ind.{name}.{suffix}"
SUFFIXES = ["x", "y", "tx", "ty", "allx", "ally", "graph", "test.index"]
# NELL (Planetoid "nell.0.001" split). The official hosts (cs.cmu.edu, data.pyg.org) are
# blocked from our cloud environment, so we fetch the identical files from a public GitHub
# mirror that stores them with Git LFS, and check each file against the sha256 recorded in
# the LFS pointer (pinned commit). Files are byte-identical to the originals iff the hashes match.
NELL_MIRROR = ("https://media.githubusercontent.com/media/D61-IA/FisherGCN/"
               "a58c1613f1aca7077ef90af6e51a8021548cdb4c/data/ind.nell.0.001.{suffix}")
NELL_SHA256 = {
    "x": "284de725538a6703f3b26ac8f252dc1c43a2fd0df46e78a618a2391a0c9ed63a",
    "y": "88926c6959052cffd51f1d4218b11c35403c0c43f21a52489fbec4676a6a205e",
    "tx": "93016916f57043f6402d9c15654328dbda143f8adcf2b1afd2638caff76808bf",
    "ty": "07a2a8a95063e14b05592322362011993ce95e7ddb87c8246ebd57607768a5c5",
    "allx": "092de8541bfd3a6a9a5364ea73224e08c3b0605033c69dc926f26004a3e93681",
    "ally": "27de2d69a83b0f37272f1189acb209449e59224e6647c296d3b7e27c35d99b1c",
    "graph": "865faeed5d7037fb21122844ec09292b9c8b59710dcf2dc126cc13069a70365b",
    "test.index": "c546df8c960fc961b75d9566a1431c16d9a064725effdcd70a615ec0a31fe907",
}
DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "planetoid")


# ---------------------------------------------------------------------------
# Safe loading
# ---------------------------------------------------------------------------
class _WhitelistUnpickler(pickle.Unpickler):
    """Only resolve the globals the Planetoid files actually reference.

    Verified with pickletools: numpy.core.multiarray._reconstruct, numpy.ndarray,
    numpy.dtype, scipy.sparse.csr.csr_matrix, __builtin__.list, collections.defaultdict.
    Python-2 module names (e.g. "__builtin__") can reach find_class unmapped, so both
    spellings are accepted.
    """

    def find_class(self, module, name):
        if (module, name) in {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct")}:
            from numpy._core.multiarray import _reconstruct  # numpy >= 2 location
            return _reconstruct
        if (module, name) == ("numpy", "ndarray"):
            return np.ndarray
        if (module, name) == ("numpy", "dtype"):
            return np.dtype
        if module in ("scipy.sparse.csr", "scipy.sparse._csr") and name == "csr_matrix":
            return sp.csr_matrix
        if (module, name) in {("builtins", "list"), ("__builtin__", "list")}:
            return list
        if (module, name) == ("collections", "defaultdict"):
            import collections
            return collections.defaultdict
        raise pickle.UnpicklingError(f"blocked global {module}.{name} in Planetoid file")


def _load_pickle(path: str):
    with open(path, "rb") as f:
        # latin1: the files were written by Python 2 with numpy arrays inside.
        return _WhitelistUnpickler(f, encoding="latin1").load()


def _download(name: str, root: str) -> None:
    os.makedirs(root, exist_ok=True)
    for s in SUFFIXES:
        dst = os.path.join(root, f"ind.{name}.{s}")
        if not os.path.exists(dst):
            url = PLANETOID_URL.format(name=name, suffix=s)
            print(f"[data] downloading {url}")
            urllib.request.urlretrieve(url, dst)


def _download_nell(root: str) -> None:
    import hashlib
    os.makedirs(root, exist_ok=True)
    for s, want in NELL_SHA256.items():
        dst = os.path.join(root, f"ind.nell.0.001.{s}")
        if not os.path.exists(dst):
            url = NELL_MIRROR.format(suffix=s)
            print(f"[data] downloading {url}")
            urllib.request.urlretrieve(url, dst)
        with open(dst, "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
        if got != want:
            raise RuntimeError(f"{dst}: sha256 {got} != expected {want}; delete it and retry")


# ---------------------------------------------------------------------------
# Graph container
# ---------------------------------------------------------------------------
@dataclass
class Graph:
    """Everything the model needs, already as torch tensors.

    x         (N, F) float32 node features X^(0): dense, or sparse COO for NELL
              (65,755 x 61,278 would be 16 GB dense)
    y         (N,)   int64 class labels
    edge_src  (E,)   int64 source node j of each edge j -> i, self-loops INCLUDED
    edge_dst  (E,)   int64 destination node i
    deg       (N,)   int64 in-degree counting the self-loop (Q-01), so deg >= 1
    train/val/test_idx  int64 index tensors of the public split (for datasets with several
              published splits: split 0; see `splits` / `use_split`)
    splits    optional list of (train_idx, val_idx, test_idx), one per published split
              (Roman-Empire / Amazon-Ratings have 10). train.py runs seed s on split s % 10.
    """

    name: str
    x: torch.Tensor
    y: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    deg: torch.Tensor
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    splits: list | None = None

    def use_split(self, i: int) -> "Graph":
        """Same graph, with train/val/test set to published split i (mod the number of splits)."""
        if not self.splits:
            return self
        tr, va, te = self.splits[i % len(self.splits)]
        return replace(self, train_idx=tr, val_idx=va, test_idx=te)

    def train_subgraph(self) -> "Graph":
        """Inductive training graph (GraphSAGE's Reddit protocol): keep only edges whose two
        ends are both training nodes. Node ids and x are unchanged (val/test nodes simply
        become isolated, keeping only their self-loop), so the same indices work for both."""
        is_tr = torch.zeros(self.num_nodes, dtype=torch.bool)
        is_tr[self.train_idx] = True
        keep = (is_tr[self.edge_src] & is_tr[self.edge_dst]) | (self.edge_src == self.edge_dst)
        src, dst = self.edge_src[keep], self.edge_dst[keep]
        return replace(self, name=self.name + "[train-induced]", edge_src=src, edge_dst=dst,
                       deg=torch.bincount(dst, minlength=self.num_nodes))

    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]

    @property
    def num_features(self) -> int:
        return self.x.shape[1]

    @property
    def num_classes(self) -> int:
        return int(self.y.max()) + 1

    def adjacency(self, values: torch.Tensor | None = None) -> torch.Tensor:
        """Sparse (N, N) matrix with A[i, j] = value of edge j -> i (rows = destinations).

        values=None gives the binary A+I the hardware stores (Q-11).
        """
        if values is None:
            values = torch.ones(self.edge_src.numel())
        idx = torch.stack([self.edge_dst, self.edge_src])
        n = self.num_nodes
        # check_invariants=True: validate indices once here (cheap) and silence torch's warning.
        return torch.sparse_coo_tensor(idx, values, (n, n), check_invariants=True).coalesce()

    def summary(self) -> str:
        e_no_self = self.edge_src.numel() - self.num_nodes
        return (f"{self.name}: N={self.num_nodes} E(directed, no self-loops)={e_no_self} "
                f"F={self.x.shape[1]} classes={self.num_classes} "
                f"deg(with self) min/mean/max={int(self.deg.min())}/{self.deg.float().mean():.2f}/{int(self.deg.max())} "
                f"split={len(self.train_idx)}/{len(self.val_idx)}/{len(self.test_idx)}")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _build_edges(graph: dict, n: int):
    """Symmetrize, drop duplicates and any existing self-loops, then add exactly one
    self-loop per node (Q-01). Edge j->i is stored as (src=j, dst=i).
    Returns (src, dst, deg) numpy arrays; deg = in-degree incl. the self-loop."""
    pairs = set()
    for u, nbrs in graph.items():
        for v in nbrs:
            if u != v:
                pairs.add((u, v))
                pairs.add((v, u))
    pairs = np.array(sorted(pairs), dtype=np.int64)
    loops = np.arange(n, dtype=np.int64)
    src = np.concatenate([pairs[:, 0], loops])
    dst = np.concatenate([pairs[:, 1], loops])
    return src, dst, np.bincount(dst, minlength=n)


DATASETS = ("cora", "citeseer", "pubmed", "nell", "reddit",
            "roman-empire", "amazon-ratings", "city-paris")
# Input preprocessing when --feat-norm is not given. MEGA's bag-of-words sets keep the Q-14
# choice (row). The others hold embeddings / mixed-unit columns where a row sum is not
# meaningful (it can be ~0); there we z-score each column. See docs/beyond-mega/01-extra-datasets.md.
DEFAULT_FEAT_NORM = {"cora": "row", "citeseer": "row", "pubmed": "row", "nell": "row",
                     "reddit": "none", "roman-empire": "std", "amazon-ratings": "std",
                     "city-paris": "std"}


def load_dataset(name: str, feat_norm: str | None = None) -> Graph:
    """Entry point used by train.py. feat_norm=None picks DEFAULT_FEAT_NORM[name]."""
    name = name.lower()
    feat_norm = feat_norm or DEFAULT_FEAT_NORM[name]
    if name == "nell":
        return load_nell(feat_norm=feat_norm)
    if name in ("roman-empire", "amazon-ratings"):
        return load_heterophilous(name, feat_norm=feat_norm)
    if name == "city-paris":
        return load_city("paris", feat_norm=feat_norm)
    if name == "reddit":
        return load_reddit(feat_norm=feat_norm)
    return load_planetoid(name, feat_norm=feat_norm)


def load_nell(root: str | None = None, feat_norm: str = "none") -> Graph:
    """NELL with the Planetoid 'nell.0.001' split, built the way PyTorch Geometric does it.

    The pickles hold features only for the 8,922 labeled-pool nodes (allx) and the 969
    test nodes (tx), all 5,414-dim. The other 55,864 graph nodes are knowledge-base
    relation nodes with no features; following PyG, each gets its own one-hot feature
    column. 5,414 + 55,864 = 61,278 features and 65,755 nodes, matching Tab.II -- which
    is why we believe the paper used this construction (the paper does not say).
    Test indices are not contiguous, so tx/ty are padded with zeros as for CiteSeer.

    Labels: 210 one-hot columns; nodes without a label (relation nodes) get class 0 and are
    never in train/val/test. The split is the public one: train = first 105 nodes, val = the
    next 500, test = the 969 test indices.

    X^(0) is returned as a sparse COO tensor. Feature values are raw counts (1, 2, ...);
    feat_norm="row" divides each row by its sum.
    """
    root = root or os.path.join(os.path.dirname(DEFAULT_ROOT), "planetoid_nell")
    _download_nell(root)
    pre = os.path.join(root, "ind.nell.0.001.")
    obj = {s: _load_pickle(pre + s) for s in ("x", "y", "tx", "ty", "allx", "ally", "graph")}
    with open(pre + "test.index") as f:
        test_idx_reorder = np.array([int(line) for line in f if line.strip()])
    test_idx_range = np.sort(test_idx_reorder)
    n = len(obj["graph"])                                     # 65,755
    allx, tx = obj["allx"].tocoo(), obj["tx"].tocoo()
    n_all, f_base = allx.shape                                # 8,922 x 5,414

    # Feature rows: allx at 0..n_all-1; test row k (in file order) belongs to node
    # test_idx_reorder[k]. (Equivalent to Kipf's vstack + reorder.)
    rows = [allx.row, test_idx_reorder[tx.row]]
    cols = [allx.col, tx.col]
    vals = [allx.data, tx.data]
    # One-hot columns for featureless nodes (not in allx, not in test).
    has_feat = np.zeros(n, dtype=bool)
    has_feat[:n_all] = True
    has_feat[test_idx_reorder] = True
    iso = np.flatnonzero(~has_feat)
    rows.append(iso)
    cols.append(f_base + np.arange(iso.size))
    vals.append(np.ones(iso.size))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    vals = np.concatenate(vals).astype(np.float32)
    n_feat = f_base + iso.size
    if feat_norm == "row":
        rs = np.bincount(rows, weights=vals, minlength=n)
        vals = (vals / rs[rows]).astype(np.float32)
    elif feat_norm != "none":
        raise ValueError(feat_norm)
    x = torch.sparse_coo_tensor(torch.from_numpy(np.stack([rows, cols])).long(),
                                torch.from_numpy(vals), (n, n_feat), check_invariants=True).coalesce()

    y = np.zeros(n, dtype=np.int64)
    y[:n_all] = obj["ally"].argmax(1)
    y[test_idx_reorder] = obj["ty"].argmax(1)

    src, dst, deg = _build_edges(obj["graph"], n)
    n_train = obj["y"].shape[0]                               # 105
    return Graph(
        name="nell",
        x=x,
        y=torch.from_numpy(y),
        edge_src=torch.from_numpy(src),
        edge_dst=torch.from_numpy(dst),
        deg=torch.from_numpy(deg).long(),
        train_idx=torch.arange(n_train),
        val_idx=torch.arange(n_train, n_train + 500),
        test_idx=torch.from_numpy(test_idx_range).long(),
    )

def load_planetoid(name: str, root: str = DEFAULT_ROOT, feat_norm: str = "row") -> Graph:
    """Load Cora / CiteSeer / PubMed with the public split.

    feat_norm:
      "row"  - divide each node's features by their sum (Kipf & Welling's preprocessing,
               part of the standard recipe chosen in Q-14). Default.
      "none" - raw features (binary for Cora/CiteSeer, TF-IDF for PubMed).
    The paper does not say which it used (Q-14), so this is a switch.
    """
    name = name.lower()
    assert name in ("cora", "citeseer", "pubmed"), name
    _download(name, root)
    obj = {s: _load_pickle(os.path.join(root, f"ind.{name}.{s}")) for s in SUFFIXES if s != "test.index"}
    with open(os.path.join(root, f"ind.{name}.test.index")) as f:
        test_idx_reorder = [int(line) for line in f if line.strip()]
    test_idx_range = np.sort(test_idx_reorder)

    tx, ty = obj["tx"], obj["ty"]
    if name == "citeseer":
        # CiteSeer has test indices with no feature row (isolated nodes). Standard fix
        # from Kipf & Welling: insert zero rows so indices line up. These nodes are not
        # in the test set.
        full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_ext = sp.lil_matrix((len(full), tx.shape[1]))
        tx_ext[test_idx_range - min(test_idx_range), :] = tx
        ty_ext = np.zeros((len(full), ty.shape[1]))
        ty_ext[test_idx_range - min(test_idx_range), :] = ty
        tx, ty = tx_ext, ty_ext

    features = sp.vstack((obj["allx"], tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    labels = np.vstack((obj["ally"], ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    x = np.asarray(features.todense(), dtype=np.float32)
    if feat_norm == "row":
        s = x.sum(1, keepdims=True)
        s[s == 0] = 1.0
        x = x / s
    elif feat_norm != "none":
        raise ValueError(feat_norm)
    y = labels.argmax(1)

    src, dst, deg = _build_edges(obj["graph"], x.shape[0])

    n_train = obj["y"].shape[0]
    return Graph(
        name=name,
        x=torch.from_numpy(x),
        y=torch.from_numpy(y).long(),
        edge_src=torch.from_numpy(src),
        edge_dst=torch.from_numpy(dst),
        deg=torch.from_numpy(deg).long(),
        train_idx=torch.arange(n_train),
        val_idx=torch.arange(n_train, n_train + 500),
        test_idx=torch.from_numpy(test_idx_range).long(),
    )


# ---------------------------------------------------------------------------
# Extra datasets (Reddit + docs/beyond-mega/01-extra-datasets.md)
# ---------------------------------------------------------------------------
DATA_ROOT = os.path.dirname(DEFAULT_ROOT)
# Pinned files. The hashes were recorded on first download (2026-11-17) from these URLs;
# a mismatch means the upstream file changed and results may no longer be comparable.
EXTRA_FILES = {
    # Platonov et al. 2023 (ICLR), github.com/yandex-research/heterophilous-graphs
    "roman-empire": ("https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/roman_empire.npz",
                     "a58ba741d123bf892fe5c872138d07463d75a2e9012360b8dd78ac2d4766d428"),
    "amazon-ratings": ("https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/amazon_ratings.npz",
                       "4c3a3e3b9d9f6cba0fede4625a00aad8c5721c1a36ed771367f446763241c7dd"),
    # Liang et al. 2025 (NeurIPS), github.com/LeonResearch/City-Networks (same files PyG 2.7 uses)
    "city-paris": ("https://raw.githubusercontent.com/LeonResearch/City-Networks/main/data/paris.tar.gz",
                   "e7b12f028aceb8057b8cbb65d5c89d95dba3122483c0419cec9872e9d652b41a"),
    # Hamilton et al. 2017 Reddit, as packaged by DGL (data.dgl.ai is blocked from our cloud
    # environment; this is the S3 bucket behind it). PyG's Reddit is the same data.
    "reddit": ("https://dgl-data.s3.amazonaws.com/dataset/reddit.zip",
               "9a16353c28f8ddd07148fc5ac9b57b818d7911ea0fbe9052d66d49fc32b372bf"),
}


def _fetch(name: str) -> str:
    """Download EXTRA_FILES[name] into data/<name>/ once, verify sha256, return the path."""
    url, want = EXTRA_FILES[name]
    root = os.path.join(DATA_ROOT, name)
    os.makedirs(root, exist_ok=True)
    dst = os.path.join(root, os.path.basename(url))
    if not os.path.exists(dst):
        print(f"[data] downloading {url}")
        urllib.request.urlretrieve(url, dst + ".part")
        os.replace(dst + ".part", dst)
    import hashlib
    h = hashlib.sha256()
    with open(dst, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    if h.hexdigest() != want:
        raise RuntimeError(f"{dst}: sha256 {h.hexdigest()} != expected {want}; delete it and retry")
    return dst


def _edges_from_matrix(a: sp.spmatrix):
    """Vectorized version of _build_edges for large graphs (Reddit: 1.1e8 stored edges, far too
    many for a Python set, and too many for int64 sort-based dedup in 15 GB of RAM).
    Symmetrize (A + Aᵀ, so duplicates merge), drop self-loops, then add one self-loop per node.
    Returns (src, dst, deg) int64 like _build_edges. Edges come out grouped by DESTINATION
    (CSC order, as the hardware stores the graph), which sample_neighbors needs on big graphs."""
    n = a.shape[0]
    a = a.tocsr().astype(np.float32)
    a = (a + a.T).tocsr()                           # union of both directions; sums duplicates
    a.setdiag(0)
    a.eliminate_zeros()
    a.data[:] = 1.0
    a = (a + sp.identity(n, dtype=np.float32, format="csr")).tocoo()   # exactly one self-loop
    # Row r of the symmetric matrix lists r's neighbors; use it as destination r (in-edges).
    dst = a.row.astype(np.int64)
    src = a.col.astype(np.int64)
    del a
    return src, dst, np.bincount(dst, minlength=n)


def _edges_from_pairs(src: np.ndarray, dst: np.ndarray, n: int):
    """Edge list (src[k] -> dst[k]) to (src, dst, deg); see _edges_from_matrix."""
    a = sp.coo_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(n, n))
    return _edges_from_matrix(a)


def _normalize_features(x: np.ndarray, feat_norm: str) -> np.ndarray:
    """row  = divide each row by its sum (Q-14, bag-of-words sets).
    std  = z-score each column over ALL nodes (transductive, uses no labels); constant
           columns are left at 0. Used where rows are embeddings or mixed-unit columns.
    none = raw."""
    x = np.asarray(x, dtype=np.float32)
    if feat_norm == "row":
        s = x.sum(1, keepdims=True)
        s[s == 0] = 1.0
        return x / s
    if feat_norm == "std":
        mu = x.mean(0, keepdims=True, dtype=np.float64).astype(np.float32)
        sd = x.std(0, keepdims=True, dtype=np.float64).astype(np.float32)
        sd[sd == 0] = 1.0
        x = x - mu                                      # float32 throughout (Reddit X is 0.56 GB)
        x /= sd
        return x
    if feat_norm == "none":
        return x
    raise ValueError(feat_norm)


def _graph(name, x, y, src, dst, deg, splits) -> Graph:
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long)
    splits = [(t(a), t(b), t(c)) for a, b, c in splits]
    return Graph(name=name, x=torch.from_numpy(x), y=t(y), edge_src=torch.from_numpy(src),
                 edge_dst=torch.from_numpy(dst), deg=torch.from_numpy(deg).long(),
                 train_idx=splits[0][0], val_idx=splits[0][1], test_idx=splits[0][2],
                 splits=splits if len(splits) > 1 else None)


def load_heterophilous(name: str, feat_norm: str = "std") -> Graph:
    """Roman-Empire / Amazon-Ratings (Platonov et al. 2023). Accuracy is their metric for both.
    The .npz holds node_features, node_labels, edges (E,2) and 10 boolean split masks
    (50/25/25). The authors make the graph bidirected; _edges_from_pairs does the same."""
    z = np.load(_fetch(name), allow_pickle=False)
    x = _normalize_features(z["node_features"], feat_norm)
    e = z["edges"]
    src, dst, deg = _edges_from_pairs(e[:, 0], e[:, 1], x.shape[0])
    splits = [tuple(np.flatnonzero(z[k][i]) for k in ("train_masks", "val_masks", "test_masks"))
              for i in range(z["train_masks"].shape[0])]
    return _graph(name, x, z["node_labels"], src, dst, deg, splits)


def load_city(city: str = "paris", feat_norm: str = "std") -> Graph:
    """City-Networks (Liang et al. 2025), following the authors' / PyG's CityNetwork defaults:
    augmented node features (37 columns), labels = 10 quantile classes of the 16-hop
    eccentricity, one fixed split (10/10/80). Files are torch tensors, read weights_only."""
    name = f"city-{city}"
    path = _fetch(name)
    out = os.path.join(os.path.dirname(path), "raw")
    base = os.path.join(out, city)
    if not os.path.exists(os.path.join(base, "edge_indices.pt")):
        with tarfile.open(path) as tf:
            tf.extractall(out, filter="data")          # "data" filter: no absolute paths / links out
    L = lambda f: torch.load(os.path.join(base, f), weights_only=True)
    x = _normalize_features(L("node_features_augmented.pt").numpy(), feat_norm)
    ei = L("edge_indices.pt").numpy()
    src, dst, deg = _edges_from_pairs(ei[0], ei[1], x.shape[0])
    split = tuple(np.flatnonzero(L(f"{k}_mask.pt").numpy()) for k in ("train", "valid", "test"))
    return _graph(name, x, L("10-chunk_16-hop_node_labels.pt").numpy(), src, dst, deg, [split])


def load_reddit(feat_norm: str = "std") -> Graph:
    """Reddit (GraphSAGE): 232,965 posts, 41 subreddits, 602 features, the standard split
    (node_types 1/2/3 = train/val/test; 153,431 / 23,831 / 55,703).
    The zip holds reddit_data.npz (feature, label, node_types) and reddit_graph.npz (a scipy
    sparse matrix saved with save_npz, loaded without pickle). Extracted once to data/reddit/."""
    path = _fetch("reddit")
    root = os.path.dirname(path)
    if not os.path.exists(os.path.join(root, "reddit_graph.npz")):
        with zipfile.ZipFile(path) as zf:
            for m in ("reddit_data.npz", "reddit_graph.npz"):
                zf.extract(m, root)                    # fixed member names only
    # graph first, while memory is free; then features (stored float64 -> float32)
    src, dst, deg = _edges_from_matrix(sp.load_npz(os.path.join(root, "reddit_graph.npz")))
    z = np.load(os.path.join(root, "reddit_data.npz"), allow_pickle=False)
    x = _normalize_features(z["feature"].astype(np.float32), feat_norm)
    nt = z["node_types"]
    split = tuple(np.flatnonzero(nt == k) for k in (1, 2, 3))
    return _graph("reddit", x, z["label"], src, dst, deg, [split])


if __name__ == "__main__":
    # Quick check: python -m mega_pt.data   (edge counts should match Tab.II of the paper:
    # Cora 10556, CiteSeer 9104, PubMed 88648, NELL 251550)
    import sys
    for ds in sys.argv[1:] or ("cora", "citeseer", "pubmed", "nell"):
        print(load_dataset(ds).summary())
