"""Amazon Reviews 2023 (McAuley Lab) 5-core benchmark: loading, user sample, sequences, leave-last-out split.

Files (per category: Beauty_and_Personal_Care, Sports_and_Outdoors, Toys_and_Games):
  benchmark/5core/rating_only/{Category}.csv.gz      user_id, parent_asin, rating, timestamp
  raw/meta_categories/meta_{Category}.jsonl.gz       item metadata (title, store, categories, price, features)

Protocol (as in TIGER, Rajput et al. 2023): every interaction counts as an implicit "purchase"; each user's items are
sorted by time; the last item is the test target, the second last the validation target, the rest is training
history; at most MAX_HIST items of history are used. Full ranking over all items (no negative sampling at test).

The 2023 categories are large (Beauty 5-core: 730K users, 208K items, 6.6M ratings), so a random sample of users is
used to fit a Colab session; the item set is every item those users interacted with. Items that appear only as a
test target (never in any training sequence) are "cold" items: a model with one embedding per item ID cannot rank
them, a model that recommends by content-derived semantic IDs can.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

MAX_HIST = 20
CATEGORIES = {"beauty": "Beauty_and_Personal_Care", "sports": "Sports_and_Outdoors", "toys": "Toys_and_Games"}


def load_ratings(path: Path, n_users: int | None = None, seed: int = 0, min_len: int = 5) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["user_id", "parent_asin", "timestamp"])
    df = df.drop_duplicates(["user_id", "parent_asin"], keep="first")
    n = df.groupby("user_id").size()
    users = n.index[n >= min_len].values
    if n_users and 0 < n_users < len(users):      # 0 or None = all users
        users = np.random.default_rng(seed).choice(users, n_users, replace=False)
    return df[df["user_id"].isin(set(users))].sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)


def sequences(df: pd.DataFrame) -> tuple[list[str], dict[str, int], list[list[int]]]:
    """-> item list (index = item id), asin -> id, per-user item-id sequences in time order."""
    items = sorted(df["parent_asin"].unique())
    idx = {a: i for i, a in enumerate(items)}
    seqs = [[idx[a] for a in g] for _, g in df.groupby("user_id", sort=True)["parent_asin"]]
    return items, idx, seqs


def split(seqs: list[list[int]]) -> dict:
    """Leave-last-out: train sequences (all but the last two), validation and test (history, target) pairs."""
    train = [s[:-2] for s in seqs]
    val = [(s[:-2][-MAX_HIST:], s[-2]) for s in seqs]
    test = [(s[:-1][-MAX_HIST:], s[-1]) for s in seqs]
    return {"train": train, "val": val, "test": test}


class TrainExamples:
    """Every prefix of every training sequence as (user index, history, next item), built on the fly: the full
    Beauty data has over 5 million prefixes, too many to keep as Python lists."""

    def __init__(self, train_seqs: list[list[int]]):
        self.seqs = train_seqs
        u, t = [], []
        for i, s in enumerate(train_seqs):
            n = max(len(s) - 1, 0)
            u.append(np.full(n, i, np.int32)); t.append(np.arange(1, n + 1, dtype=np.int32))
        self.u = np.concatenate(u) if u else np.zeros(0, np.int32)
        self.t = np.concatenate(t) if t else np.zeros(0, np.int32)

        self.flat = np.concatenate([np.asarray(x, np.int64) for x in train_seqs]) if train_seqs else np.zeros(0, np.int64)
        self.off = np.r_[0, np.cumsum([len(x) for x in train_seqs])].astype(np.int64)

    def __len__(self):
        return len(self.u)

    def batch(self, js):
        """Vectorised: users (B,), history item matrix (B, MAX_HIST) right-aligned with -1 padding, targets (B,)."""
        u = self.u[js].astype(np.int64); t = self.t[js].astype(np.int64)
        start, end = self.off[u], self.off[u] + t
        pos = end[:, None] - MAX_HIST + np.arange(MAX_HIST)[None, :]
        valid = pos >= np.maximum(start, end - MAX_HIST)[:, None]
        hist = np.where(valid, self.flat[np.clip(pos, 0, None)], -1)
        return u, hist, self.flat[end]

    def __getitem__(self, j):
        u, t = int(self.u[j]), int(self.t[j])
        s = self.seqs[u]
        return u, s[max(0, t - MAX_HIST):t], s[t]


def train_examples(train_seqs: list[list[int]]) -> TrainExamples:
    return TrainExamples(train_seqs)


def summary(df, items, seqs, sp) -> dict:
    lens = np.array([len(s) for s in seqs])
    seen = np.zeros(len(items), bool)
    for s in sp["train"]:
        seen[s] = True
    for h, _ in sp["val"]:
        seen[h] = True
    test_cold = float(np.mean([not seen[t] for _, t in sp["test"]]))
    counts = np.bincount(np.concatenate([np.asarray(s) for s in sp["train"] if len(s)]), minlength=len(items)) if len(seqs) else np.zeros(0)
    return {"users": len(seqs), "items": len(items), "interactions": int(len(df)),
            "seq_len": {"median": float(np.median(lens)), "mean": round(float(lens.mean()), 2), "max": int(lens.max())},
            "density_pct": round(100 * len(df) / max(len(seqs) * len(items), 1), 4),
            "test_targets_cold_pct": round(100 * test_cold, 2),
            "train_item_count": {"median": float(np.median(counts)), "items_never_in_train_pct": round(100 * float((counts == 0).mean()), 2)}}


def item_text(meta: dict) -> str:
    """The item's content as one string (the fields TIGER uses: title, brand, categories, price)."""
    parts = []
    if meta.get("title"):
        parts.append(f"Title: {meta['title']}")
    if meta.get("store"):
        parts.append(f"Brand: {meta['store']}")
    cats = [c for c in (meta.get("categories") or []) if c]
    if cats:
        parts.append("Category: " + " > ".join(cats[-3:]))
    if meta.get("price") not in (None, "", "None"):
        parts.append(f"Price: {meta['price']}")
    feats = [f for f in (meta.get("features") or []) if f][:2]
    if feats:
        parts.append("Features: " + "; ".join(feats))
    return ". ".join(parts)[:1000] or "Unknown item"


def load_meta(path: Path, wanted: set[str]) -> dict[str, dict]:
    """Streams the (large) metadata file and keeps only the wanted items."""
    out = {}
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            a = d.get("parent_asin")
            if a in wanted:
                out[a] = {"title": d.get("title") or "", "store": d.get("store") or "",
                          "categories": d.get("categories") or [], "price": d.get("price"),
                          "features": d.get("features") or [], "main_category": d.get("main_category") or ""}
                if len(out) == len(wanted):
                    break
    return out


def fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for u, a, t in df[["user_id", "parent_asin", "timestamp"]].itertuples(index=False):
        h.update(f"{u}|{a}|{t}\n".encode())
    return h.hexdigest()
