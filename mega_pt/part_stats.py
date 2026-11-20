"""METIS partition statistics for the Condense-Edge decisions (Q-27).

Usage:  python -m mega_pt.part_stats cora pubmed nell ...

Partitions each graph into K = ceil(N / 256 * 1.05) parts. 256 is the Aggregation Buffer's
destination capacity at F2 = 128 (Q-25); the 5% slack absorbs METIS's imbalance so that no
part exceeds 256. It then counts:
  ext_pairs        distinct (subgraph, external source node) pairs = rows Condense-Edge places
                   in Sparse regions, once per subgraph that needs them
  batches_touched  per node, the number of 16-subgraph batches holding a subgraph that needs
                   it = batch visits under B2 with batch-minimum skipping
Self-loops are ignored. GraphSage's sampled graphs are not covered.
"""

import math
import sys

import numpy as np
import pymetis

from .data import load_dataset

CAP = 256        # destinations per subgraph at F2 = 128 (Q-25)
BATCH = 16       # eID FIFOs in the Condense Unit (Tab.IV)


def stats(name: str) -> dict:
    g = load_dataset(name)
    n = g.num_nodes
    src, dst = g.edge_src.numpy(), g.edge_dst.numpy()
    m = src != dst
    order = np.argsort(src[m], kind="stable")
    s, d = src[m][order], dst[m][order]
    xadj = np.concatenate([[0], np.cumsum(np.bincount(s, minlength=n))])
    k = math.ceil(n / CAP * 1.05)
    _, part = pymetis.part_graph(k, xadj=xadj, adjncy=d)
    part = np.asarray(part)
    ext = (part[src] != part[dst]) & m
    pairs = np.unique(part[dst[ext]].astype(np.int64) * n + src[ext])
    sub, node = pairs // n, pairs % n
    touched = np.unique((sub // BATCH) * n + node).size
    return {"dataset": name, "N": n, "parts": k, "max_part": int(np.bincount(part).max()),
            "batches": math.ceil(k / BATCH), "ext_edge_frac": float(ext.sum() / m.sum()),
            "ext_pairs_per_node": pairs.size / n, "batches_touched_per_node": touched / n}


if __name__ == "__main__":
    for name in sys.argv[1:]:
        print(stats(name))
