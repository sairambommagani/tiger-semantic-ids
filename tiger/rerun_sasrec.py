"""Re-run only the SASRec baseline for a category, with exactly the same split and the same evaluation users as
tiger/train.py (same seed), and write the results next to the model versions.

Used for the first Sports and Beauty runs, where SASRec produced NaN scores in evaluation mode (key-padding mask on
left-padded sequences, fixed in tiger/baselines.py); TIGER and popularity results of those runs are not affected.

  python -m tiger.rerun_sasrec --ratings Sports_and_Outdoors.csv.gz --out artifacts_sports --category sports
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import baselines as B
from . import data as D
from . import metrics as MT


def run(ratings, out, category, n_users=0, eval_users=20000, epochs=15, seed=0, log=print) -> dict:
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = D.load_ratings(ratings, n_users, seed)
    items, _, seqs = D.sequences(df)
    sp = D.split(seqs)
    rng = np.random.default_rng(seed)
    ev = rng.choice(len(seqs), min(eval_users, len(seqs)), replace=False)       # same draw as tiger/train.py
    vsub = ev[:min(2000, len(ev))]
    seen = np.zeros(len(items), bool)
    for s in sp["train"]:
        seen[s] = True
    for h, _ in sp["val"]:
        seen[h] = True
    test_t = [sp["test"][u][1] for u in ev]
    cold = np.array([not seen[t] for t in test_t])
    vh, vt = [sp["val"][u][0] for u in vsub], [sp["val"][u][1] for u in vsub]
    sas, hist = B.train_sasrec(sp["train"], len(items), dev, epochs=epochs, seed=seed,
                               eval_fn=lambda mm: MT.evaluate(B.sasrec_topk(mm, vh, dev), vt)["recall@10"], log=log)
    rec = B.sasrec_topk(sas, [sp["test"][u][0] for u in ev], dev)
    res = {"category": category, "params_millions": round(sum(p.numel() for p in sas.parameters()) / 1e6, 2),
           "train_minutes": round((time.time() - t0) / 60, 1),
           "val": MT.evaluate(B.sasrec_topk(sas, [sp["val"][u][0] for u in ev], dev), [sp["val"][u][1] for u in ev]),
           "test": MT.evaluate(rec, test_t), "test_cold": MT.evaluate(rec, test_t, mask=cold),
           "test_warm": MT.evaluate(rec, test_t, mask=~cold), "history": hist,
           "note": "re-run with the fixed SASRec (no key-padding mask); same split and evaluation users as the TIGER run"}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "sasrec_fixed.json").write_text(json.dumps(res, indent=2))
    log(f"  sasrec (fixed) {category}: test Recall@5 {res['test']['recall@5']}  Recall@10 {res['test']['recall@10']}  "
        f"NDCG@5 {res['test']['ndcg@5']}  NDCG@10 {res['test']['ndcg@10']}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--eval-users", type=int, default=20000)
    a = ap.parse_args()
    run(a.ratings, a.out, a.category, eval_users=a.eval_users, epochs=a.epochs, log=lambda *x: print(*x, flush=True))
