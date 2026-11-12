"""Tests on tiny synthetic Amazon-like files (no download, no GPU, TF-IDF embeddings instead of Sentence-T5)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tiger import data as D                                  # noqa: E402
from tiger import metrics as MT                              # noqa: E402
from tiger import rqvae as RQ                                # noqa: E402


def test_data_split(tmp_path):
    from scripts.make_synthetic import write
    r, m = write(tmp_path)
    df = D.load_ratings(r, n_users=100)
    assert df["user_id"].nunique() == 100
    items, idx, seqs = D.sequences(df)
    sp = D.split(seqs)
    for s, tr, (vh, vt), (th, tt) in zip(seqs, sp["train"], sp["val"], sp["test"]):
        assert tt == s[-1] and vt == s[-2] and th[-1] == vt and tr == s[:-2]
    ex = D.train_examples(sp["train"])
    assert all(len(h) <= D.MAX_HIST for _, h, _ in ex)
    meta = D.load_meta(m, set(items))
    assert set(meta) == set(items) and "Title:" in D.item_text(meta[items[0]])


def test_collision_token_and_metrics():
    c = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 4]])
    s = RQ.add_collision_token(c)
    assert s[:, -1].tolist() == [0, 1, 0] and len({tuple(x) for x in s}) == 3
    st = RQ.id_stats(c, 8)
    assert st["unique_full_code_pct"] == round(100 * 2 / 3, 2) and st["max_items_per_code"] == 2
    r = MT.evaluate([[5, 1, 2], [3, 4, 9]], [1, 7])
    assert r["recall@5"] == 0.5 and abs(r["ndcg@5"] - 0.5 / np.log2(3)) < 1e-4


def test_pipeline_tiny_and_api(tmp_path, monkeypatch):
    from scripts.make_synthetic import write
    from tiger import train, versioning
    r, m = write(tmp_path / "data")
    out = train.run(r, m, tmp_path / "artifacts", n_users=None, encoder="tfidf", rq_steps=300, codes=16, epochs=2,
                    sasrec_epochs=2, eval_users=100, beams=10, random_ablation=True,
                    model_kw=dict(d_model=32, layers=1, heads=2, d_kv=16, d_ff=64), notes="test", log=lambda *a: None)
    card = out["version"]
    assert card["version"] == "1.0.0" and card["name"].startswith("tiger_v1.0.0_beauty_")
    vdir = tmp_path / "artifacts" / card["name"]
    for f in ("tiger_model/config.json", "items.json", "rqvae.pt", "metrics.json", "samples.json", "version.json"):
        assert (vdir / f).exists(), f
    assert versioning.verify(vdir) == []
    mt = json.loads((vdir / "metrics.json").read_text())
    assert {"tiger", "tiger_random_ids", "sasrec", "popularity"} == set(mt["results"])
    items = json.loads((vdir / "items.json").read_text())
    assert len({tuple(i["semantic_id"]) for i in items}) == len(items)          # every item has a unique ID

    monkeypatch.setenv("TIGER_ARTIFACTS", str(tmp_path / "artifacts"))
    import importlib
    import tiger.api as api
    importlib.reload(api)
    from fastapi.testclient import TestClient
    with TestClient(api.app) as c:
        hist = [items[0]["asin"], items[1]["asin"]]
        j = c.post("/recommend", json={"history": hist, "k": 5}).json()
        assert len(j["recommendations"]) <= 5 and all(x["asin"] in {i["asin"] for i in items} for x in j["recommendations"])
        assert j["model_version"] == card["name"]
        assert c.post("/recommend", json={"history": ["NOPE"]}).status_code == 422
        assert c.post("/recommend", json={"history": []}).status_code == 422
        assert c.get("/search", params={"q": "shampoo"}).json()
        assert c.get(f"/items/{items[0]['asin']}").json()["semantic_id"] == items[0]["semantic_id"]
        assert c.get("/items/NOPE").status_code == 404
        assert isinstance(c.get("/samples").json(), list)
        assert c.get("/model-info").json()["version"]["name"] == card["name"]
        assert "semantic IDs" in c.get("/").text


def test_rerun_sasrec_same_eval_users(tmp_path):
    from scripts.make_synthetic import write
    from tiger import rerun_sasrec
    r, _ = write(tmp_path / "data")
    res = rerun_sasrec.run(r, tmp_path / "out", "beauty", eval_users=100, epochs=2, log=lambda *a: None)
    assert res["test"]["users"] == 100 and (tmp_path / "out" / "sasrec_fixed.json").exists()
    assert res["test"]["recall@10"] is not None
