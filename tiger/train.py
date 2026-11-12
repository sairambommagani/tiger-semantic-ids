"""TIGER pipeline for one category (Google Colab GPU; the tests run a tiny version on CPU).

  python -m tiger.train --ratings Beauty_and_Personal_Care.csv.gz --meta meta_Beauty_and_Personal_Care.jsonl.gz \
      --category beauty --users 50000 --out artifacts

Steps
  1. ratings -> user sample -> time-ordered sequences -> leave-last-out split (tiger/data.py)
  2. item text (title, brand, categories, price, features) -> Sentence-T5 embedding (tiger/text.py)
  3. RQ-VAE -> semantic IDs (3 codewords + collision token); usage, collisions, category purity (tiger/rqvae.py)
  4. TIGER seq2seq trained on every prefix of the training sequences; early stopping on validation Recall@10
  5. baselines on the same split: popularity, SASRec; optional ablation: TIGER with random IDs
  6. test (once): Recall@5/10, NDCG@5/10 on all test users of the evaluation sample, and separately on cold items
     (never seen in training) and warm items; invalid-ID rate without the trie constraint
  7. a new model version folder (TIGER weights, item table with semantic IDs, RQ-VAE, metrics, model card)
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
from . import rqvae as RQ
from . import seq2seq as S
from . import versioning
from .text import DEFAULT_ENCODER, embed


def run(ratings, meta, art_root, category="beauty", n_users=50000, encoder=DEFAULT_ENCODER, rq_steps=6000,
        codes=256, epochs=20, sasrec_epochs=30, eval_users=10000, beams=20, random_ablation=False, bs=256, lr=1e-3, eval_every=2,
        model_kw=None, bump="minor", notes="", seed=0, cold_frac=0.0, cold_eval=5000, variants=None, skip_sasrec=False,
        log=print) -> dict:
    """variants: TIGER variants to train, from "tiger" (semantic IDs) and "tiger_random_ids"; the first is saved.
    cold_frac > 0: cold-start experiment as in the TIGER paper - that share of the test-target items is removed from
    all training sequences and histories (their semantic IDs still come from their text); results on users whose
    test item is one of them are reported as test_cold."""
    variants = list(variants or (["tiger", "tiger_random_ids"] if random_ablation else ["tiger"]))
    t_start = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))
    df = D.load_ratings(ratings, n_users, seed)
    items, idx, seqs = D.sequences(df)
    sp = D.split(seqs)
    summ = D.summary(df, items, seqs, sp)
    log("data: " + json.dumps(summ))

    t0 = time.time()
    m = D.load_meta(meta, set(items)) if meta else {}
    texts = [D.item_text(m.get(a, {})) for a in items]
    leaf = [(m.get(a, {}).get("categories") or ["?"])[-1] for a in items]
    summ["items_with_metadata_pct"] = round(100 * sum(a in m for a in items) / len(items), 2)
    log(f"metadata for {summ['items_with_metadata_pct']}% of items ({time.time() - t0:.0f} s); example: {texts[0][:150]}")
    t0 = time.time()
    import hashlib
    key = hashlib.sha256(("\n".join(texts) + encoder).encode()).hexdigest()[:16]
    cdir = Path(art_root).resolve().parent / "cache"                  # outside the artifacts folder (not zipped)
    cdir.mkdir(parents=True, exist_ok=True)
    cache = cdir / f"emb_{category}_{key}.npy"                       # reused if the run is repeated
    if cache.exists():
        emb = np.load(cache)
        log(f"loaded cached embeddings {cache.name}")
    else:
        emb = embed(texts, encoder, dev=str(dev))
        np.save(cache, emb)
    log(f"embedded {len(texts)} items with {encoder}: dim {emb.shape[1]} ({time.time() - t0:.0f} s)")

    rq, rq_hist = RQ.train(emb, dev, codes=codes, steps=rq_steps, seed=seed, log=log)
    raw = rq.codes_for(emb)
    sem = RQ.add_collision_token(raw)
    if sem[:, -1].max() >= rq.codes:
        raise RuntimeError(f"too many items share one code ({sem[:, -1].max() + 1}); the collision token needs < {rq.codes}")
    st = RQ.id_stats(raw, rq.codes)
    rnd_raw = RQ.random_ids(len(items), raw.shape[1], rq.codes, seed)
    sem_info = {"levels": int(raw.shape[1]), "codebook_size": int(rq.codes), **st,
                "max_collision_token": int(sem[:, -1].max()),
                "purity_semantic": RQ.prefix_purity(raw, leaf), "purity_random": RQ.prefix_purity(rnd_raw, leaf)}
    log("semantic IDs: " + json.dumps(sem_info))

    rng = np.random.default_rng(seed)
    ev = rng.choice(len(seqs), min(eval_users, len(seqs)), replace=False)
    vsub = ev[:min(2000, len(ev))]
    cold_set = set()
    if cold_frac > 0:
        tgt_items = np.unique([t for _, t in sp["test"]])
        cold_set = set(np.random.default_rng(seed + 7).choice(tgt_items, int(round(cold_frac * len(tgt_items))), replace=False).tolist())
        n_before = sum(len(x) for x in sp["train"])
        sp["train"] = [[i for i in x if i not in cold_set] for x in sp["train"]]
        sp["val"] = [([i for i in h if i not in cold_set], t) for h, t in sp["val"]]
        sp["test"] = [([i for i in h if i not in cold_set], t) for h, t in sp["test"]]
        cu = np.array([u for u, (_, t) in enumerate(sp["test"]) if t in cold_set and u not in set(ev.tolist())])
        cu = np.random.default_rng(seed + 8).permutation(cu)[:cold_eval]
        summ["cold_start"] = {"cold_items": len(cold_set), "share_of_test_items_pct": round(100 * cold_frac, 1),
                              "train_interactions_removed": int(n_before - sum(len(x) for x in sp["train"])),
                              "extra_cold_eval_users": int(len(cu))}
        log("cold-start experiment: " + json.dumps(summ["cold_start"]))
        ev = np.concatenate([ev, cu]).astype(int)
    in_rand = np.zeros(len(ev), bool)
    in_rand[:min(eval_users, len(seqs))] = True
    seen = np.zeros(len(items), bool)
    for s_ in sp["train"]:
        seen[s_] = True
    for h, _ in sp["val"]:
        seen[h] = True
    test_t = [sp["test"][u][1] for u in ev]
    cold = np.array([(t in cold_set) if cold_set else (not seen[t]) for t in test_t])
    examples = D.train_examples(sp["train"])
    # the validation target is also a training example for the test step (history -> second-last item)
    log(f"training examples: {len(examples):,} (prefixes); evaluation users: {int(in_rand.sum()):,} random"
        f" + {int((~in_rand).sum()):,} extra cold; cold test targets {int(cold.sum()):,}")

    def scores(recs):
        vt_all = [sp["val"][u][1] for u in ev]
        return {"test": MT.evaluate([r for r, k in zip(recs, in_rand) if k], [t for t, k in zip(test_t, in_rand) if k]),
                "test_cold": MT.evaluate(recs, test_t, mask=cold),
                "test_warm": MT.evaluate(recs, test_t, mask=in_rand & ~cold)}

    def tiger_variant(name, ids):
        voc = S.Vocab(ids, rq.codes)
        mdl = S.build(voc, **(model_kw or {}))
        vrows = [(int(u), sp["val"][u][0], None) for u in vsub]
        vt = [sp["val"][u][1] for u in vsub]

        def vfn(mm):
            recs, _ = S.recommend(mm, voc, vrows, dev, k=10, beams=beams)
            return MT.evaluate(recs, vt)["recall@10"]
        t1 = time.time()
        ck = cdir / f"ckpt_{category}_{name}_{key}{'_cold' + str(cold_frac) if cold_set else ''}.pt"
        mdl, h = S.train(mdl, voc, examples, vrows, dev, epochs=epochs, bs=bs, lr=lr, eval_fn=vfn, seed=seed, ckpt=ck,
                         eval_every=eval_every, log=log)
        mins = round((time.time() - t1) / 60, 1)
        t1 = time.time()
        vrec, _ = S.recommend(mdl, voc, [(int(u), sp["val"][u][0], None) for u in ev], dev, k=10, beams=beams)
        rows = [(int(u), sp["test"][u][0], None) for u in ev]
        long, _ = S.recommend(mdl, voc, rows, dev, k=beams, beams=beams)     # all distinct items from the beams
        recs = [r[:10] for r in long]
        sec = time.time() - t1
        eps = {}
        if cold_set:   # as in the paper: allow up to e x K new (cold) items from the beams into the top K
            for e in (0.1, 0.2):
                n = int(round(e * 10))
                r_e = []
                for r in long:
                    c = [i for i in r if i in cold_set][:n]
                    w = [i for i in r if i not in cold_set][:10 - len(c)]
                    r_e.append(sorted(c + w, key=r.index))
                eps[f"epsilon_{e}"] = scores(r_e)
        _, inv = S.recommend(mdl, voc, rows[:1000], dev, k=10, beams=beams, constrained=False)
        res = {"params_millions": round(sum(p.numel() for p in mdl.parameters()) / 1e6, 2), "train_minutes": mins,
               "users_per_s": round(2 * len(ev) / max(sec, 1e-9), 1),
               "val": MT.evaluate([r for r, k in zip(vrec, in_rand) if k], [sp["val"][u][1] for u, k in zip(ev, in_rand) if k]),
               **scores(recs), **({"cold_slots": eps} if eps else {}),
               "invalid_id_rate_unconstrained": round(inv / (min(1000, len(rows)) * beams), 4)}
        log(f"  {name}: test Recall@10 {res['test']['recall@10']}  NDCG@10 {res['test']['ndcg@10']}  "
            f"cold Recall@10 {res['test_cold']['recall@10']}  invalid IDs without trie {res['invalid_id_rate_unconstrained']}")
        for k_, v_ in eps.items():
            log(f"  {name} with {k_} cold slots: cold Recall@10 {v_['test_cold']['recall@10']}  test Recall@10 {v_['test']['recall@10']}")
        return mdl, voc, h, res

    results, hist = {}, {}
    tiger = voc = None
    for name in variants:
        log(f"training {name}")
        ids = sem if name == "tiger" else RQ.add_collision_token(rnd_raw)
        mdl, vv, hist[name], results[name] = tiger_variant(name, ids)
        if tiger is None:
            tiger, voc, saved_ids = mdl, vv, ids

    if not skip_sasrec:
        log("training SASRec")
        t1 = time.time()
        vh = [sp["val"][u][0] for u in vsub]
        vt = [sp["val"][u][1] for u in vsub]
        sas, hist["sasrec"] = B.train_sasrec(sp["train"], len(items), dev, epochs=sasrec_epochs, seed=seed,
                                             eval_fn=lambda mm: MT.evaluate(B.sasrec_topk(mm, vh, dev), vt)["recall@10"], log=log)
        srec = B.sasrec_topk(sas, [sp["test"][u][0] for u in ev], dev)
        svrec = B.sasrec_topk(sas, [sp["val"][u][0] for u, k in zip(ev, in_rand) if k], dev)
        results["sasrec"] = {"params_millions": round(sum(p.numel() for p in sas.parameters()) / 1e6, 2),
                             "train_minutes": round((time.time() - t1) / 60, 1),
                             "val": MT.evaluate(svrec, [sp["val"][u][1] for u, k in zip(ev, in_rand) if k]), **scores(srec)}
    pop = B.popularity(sp["train"], len(items))
    results["popularity"] = {"val": MT.evaluate([pop] * int(in_rand.sum()), [sp["val"][u][1] for u, k in zip(ev, in_rand) if k]),
                             **scores([pop] * len(ev))}
    for k, r in results.items():
        log(f"  {k:18s} test Recall@5 {r['test']['recall@5']}  Recall@10 {r['test']['recall@10']}  NDCG@5 {r['test']['ndcg@5']}  "
            f"NDCG@10 {r['test']['ndcg@10']}  | cold R@10 {r['test_cold']['recall@10']}  warm R@10 {r['test_warm']['recall@10']}")

    version, vdir = versioning.new_version(art_root, bump, arch=category + ("_coldstart" if cold_set else "")
                                           + ("" if variants[0] == "tiger" else "_randomids"))
    tiger.save_pretrained(vdir / "tiger_model", safe_serialization=True)
    torch.save({"state": rq.state_dict(), "in_dim": int(emb.shape[1]), "codes": int(rq.codes)}, vdir / "rqvae.pt")
    counts = np.bincount(np.concatenate([np.asarray(s, int) for s in sp["train"] if len(s)]), minlength=len(items))
    table = [{"asin": a, "title": (m.get(a, {}).get("title") or "")[:200], "brand": m.get(a, {}).get("store") or "",
              "category": leaf[i], "semantic_id": [int(x) for x in saved_ids[i]], "train_count": int(counts[i])}
             for i, a in enumerate(items)]
    (vdir / "items.json").write_text(json.dumps(table))
    tr = {u: r for u, r in zip(ev.tolist(), S.recommend(tiger, voc, [(int(u), sp["test"][u][0], None) for u in ev[:200]], dev, k=10, beams=beams)[0])}
    demo = [{"history": [items[i] for i in sp["test"][u][0]], "target": items[sp["test"][u][1]],
             "target_cold": bool(not seen[sp["test"][u][1]]), "tiger_hit@10": sp["test"][u][1] in tr[u]}
            for u in ev[:200].tolist()]
    demo = sorted(demo, key=lambda d: (not d["tiger_hit@10"], len(d["history"])))[:6] + [d for d in demo if not d["tiger_hit@10"]][:2]
    (vdir / "samples.json").write_text(json.dumps(demo))
    metrics = {"deployed": variants[0], "variants": variants, "cold_frac": cold_frac, "category": category, "category_name": D.CATEGORIES.get(category, category),
               "encoder": encoder, "data": summ, "data_sha256": D.fingerprint(df), "semantic_ids": sem_info,
               "rqvae_history": rq_hist, "results": results, "history": hist, "beams": beams,
               "eval_users": int(len(ev)), "minutes": round((time.time() - t_start) / 60, 1)}
    (vdir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    card = versioning.register(art_root, vdir, version, bump, metrics["data_sha256"], summ, metrics, notes)
    for f in cdir.glob(f"ckpt_{category}_*"):     # finished: checkpoints are no longer needed
        f.unlink()
    log(f"saved {vdir.name} in {metrics['minutes']} min")
    return {"version": card, "metrics": metrics}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--meta")
    ap.add_argument("--category", default="beauty")
    ap.add_argument("--users", type=int, default=0, help="0 = all users")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--rq-steps", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--sasrec-epochs", type=int, default=30)
    ap.add_argument("--eval-users", type=int, default=20000)
    ap.add_argument("--eval-every", type=int, default=1, help="validation every N epochs")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--beams", type=int, default=20)
    ap.add_argument("--random-ablation", action="store_true")
    ap.add_argument("--variants", default="", help="comma list: tiger,tiger_random_ids (first one is saved)")
    ap.add_argument("--cold-frac", type=float, default=0.0, help="cold-start experiment: share of test items removed from training")
    ap.add_argument("--skip-sasrec", action="store_true")
    ap.add_argument("--bump", default="minor", choices=["major", "minor", "patch"])
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    out = run(a.ratings, a.meta, a.out, a.category, a.users, a.encoder, a.rq_steps, 256, a.epochs, a.sasrec_epochs,
              a.eval_users, a.beams, a.random_ablation, bs=a.batch, lr=a.lr, eval_every=a.eval_every, notes=a.notes,
              cold_frac=a.cold_frac, variants=[v for v in a.variants.split(",") if v] or None, skip_sasrec=a.skip_sasrec, log=lambda *x: print(*x, flush=True))
    r = out["metrics"]["results"]
    print(json.dumps({k: {"test": v["test"], "cold_recall@10": v["test_cold"]["recall@10"]} for k, v in r.items()}, indent=2))


if __name__ == "__main__":
    main()
