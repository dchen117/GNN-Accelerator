"""Planetoid citation datasets (Cora, CiteSeer, PubMed) with the standard public split.

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


# ---------------------------------------------------------------------------
# Graph container
# ---------------------------------------------------------------------------
@dataclass
class Graph:
    """Everything the model needs, already as torch tensors.

    x         (N, F) float32 dense node features X^(0)
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
# Loader
# ---------------------------------------------------------------------------
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

    # Edges: symmetrize, drop duplicates and any existing self-loops, then add exactly one
    # self-loop per node (Q-01). Edge j->i is stored as (src=j, dst=i).
    n = x.shape[0]
    pairs = set()
    for u, nbrs in obj["graph"].items():
        for v in nbrs:
            if u != v:
                pairs.add((u, v))
                pairs.add((v, u))
    pairs = np.array(sorted(pairs), dtype=np.int64)
    loops = np.arange(n, dtype=np.int64)
    src = np.concatenate([pairs[:, 0], loops])
    dst = np.concatenate([pairs[:, 1], loops])
    deg = np.bincount(dst, minlength=n)  # in-degree incl. self-loop

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
    # Cora 10556, CiteSeer 9104, PubMed 88648)
    for ds in ("cora", "citeseer", "pubmed"):
        print(load_planetoid(ds).summary())
