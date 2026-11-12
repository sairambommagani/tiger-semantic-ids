"""Inference: a purchase history (item IDs) -> next-item recommendations generated as semantic IDs (CPU)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import seq2seq as S
from .data import MAX_HIST
from .versioning import resolve, verify

DISCLAIMER = ("Research prototype. Trained on the 5-core data of one Amazon Reviews 2023 category; recommendations "
              "come only from items in that catalogue.")


class Recommender:
    def __init__(self, root: Path):
        from transformers import T5ForConditionalGeneration
        self.dir = resolve(Path(root))
        bad = verify(self.dir)
        if bad:
            raise RuntimeError(f"model files changed since training: {bad}")
        self.card = json.loads((self.dir / "version.json").read_text())
        self.metrics = json.loads((self.dir / "metrics.json").read_text())
        self.items = json.loads((self.dir / "items.json").read_text())
        self.idx = {it["asin"]: i for i, it in enumerate(self.items)}
        sem = np.array([it["semantic_id"] for it in self.items])
        self.vocab = S.Vocab(sem, int(self.metrics["semantic_ids"]["codebook_size"]))
        self.model = T5ForConditionalGeneration.from_pretrained(self.dir / "tiger_model").eval()
        f = self.dir / "samples.json"
        self.samples = json.loads(f.read_text()) if f.exists() else []

    def item(self, i: int) -> dict:
        it = self.items[i]
        return {"asin": it["asin"], "title": it["title"], "brand": it["brand"], "category": it["category"],
                "semantic_id": it["semantic_id"], "train_count": it["train_count"]}

    def search(self, q: str, k: int = 20) -> list[dict]:
        q = q.lower().strip()
        if not q:
            return []
        hits = [i for i, it in enumerate(self.items) if q in it["title"].lower() or q in it["brand"].lower() or q == it["asin"].lower()]
        hits.sort(key=lambda i: -self.items[i]["train_count"])
        return [self.item(i) for i in hits[:k]]

    def recommend(self, history: list[str], k: int = 10, beams: int = 20) -> dict:
        import torch
        t0 = time.time()
        if not history:
            raise ValueError("history is empty: give at least one item ID (parent_asin)")
        unknown = [a for a in history if a not in self.idx]
        known = [self.idx[a] for a in history if a in self.idx][-MAX_HIST:]
        if not known:
            raise ValueError(f"none of the items are in this model's catalogue: {unknown[:5]}")
        recs, _ = S.recommend(self.model, self.vocab, [(-1, known, None)], torch.device("cpu"), k=k, beams=max(beams, k))
        test = self.metrics["results"]["tiger"]["test"]
        first = {self.items[i]["semantic_id"][0] for i in known}
        shared = [self.items[r]["semantic_id"][0] in first for r in recs[0]]
        return {"history": [self.item(i) for i in known], "unknown_items": unknown,
                "recommendations": [{**self.item(r), "rank": n + 1, "shares_first_code_with_history": bool(s)}
                                    for n, (r, s) in enumerate(zip(recs[0], shared))],
                "model_version": self.card["name"], "category": self.metrics.get("category_name"),
                "reliability": (f"on {test['users']:,} held-out users: Recall@10 {test['recall@10']}, NDCG@10 {test['ndcg@10']}"),
                "seconds": round(time.time() - t0, 3), "disclaimer": DISCLAIMER}
