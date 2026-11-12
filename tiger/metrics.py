"""Recall@k and NDCG@k for one held-out target per user (full ranking)."""
from __future__ import annotations

import numpy as np


def evaluate(recs: list[list[int]], targets: list[int], ks=(5, 10), mask=None) -> dict:
    t = np.asarray(targets)
    rank = np.array([r.index(x) if x in r else 10 ** 9 for r, x in zip(recs, t)])
    if mask is not None:
        rank = rank[np.asarray(mask)]
    out = {"users": int(len(rank))}
    for k in ks:
        hit = rank < k
        out[f"recall@{k}"] = round(float(hit.mean()), 4) if len(rank) else None
        out[f"ndcg@{k}"] = round(float(np.where(hit, 1 / np.log2(rank + 2), 0).mean()), 4) if len(rank) else None
    return out
