"""Planetoid datasets (Cora, CiteSeer, PubMed, NELL) with the standard public split.

Decisions implemented here (see docs/mega-kb/04-open-questions.md):
  Q-01  degree = in-degree with self-loops counted.
  Q-14  standard Planetoid public split (Kipf & Welling / Yang et al.).

The raw files are the original Planetoid pickles from github.com/kimiyoung/planetoid.
They are Python-2 pickles, and unpickling untrusted data can run arbitrary code, so we
load them with a whitelist unpickler that only accepts the few classes these files use
(numpy arrays, a scipy CSR matrix, list, defaultdict).
"""

from __future__ import annotations

import os
import pickle
import urllib.request
from dataclasses import dataclass

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
    train/val/test_idx  int64 index tensors of the public split
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


def load_dataset(name: str, feat_norm: str = "none") -> Graph:
    """Entry point used by train.py: cora / citeseer / pubmed / nell."""
    if name.lower() == "nell":
        return load_nell(feat_norm=feat_norm)
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


if __name__ == "__main__":
    # Quick check: python -m mega_pt.data   (edge counts should match Tab.II of the paper:
    # Cora 10556, CiteSeer 9104, PubMed 88648, NELL 251550)
    for ds in ("cora", "citeseer", "pubmed", "nell"):
        print(load_dataset(ds).summary())
