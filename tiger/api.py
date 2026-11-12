"""FastAPI service and web page (laptop, CPU).

  python -m uvicorn tiger.api:app --port 8014   ->   http://localhost:8014   (interactive API docs: /docs)

  POST /recommend      {"history": ["B0...", ...], "k": 10}  -> next items with their semantic IDs
  GET  /search?q=...   find items by title / brand / ID (to build a history)
  GET  /items/{asin}   one item with its semantic ID
  GET  /samples        held-out users' histories (and the item they bought next) for the demo
  GET  /model-info     model card, semantic-ID statistics and all results
  GET  /health         status and model version
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .infer import DISCLAIMER, Recommender

ART = Path(os.environ.get("TIGER_ARTIFACTS", "artifacts"))
UI = Path(__file__).parent / "static" / "index.html"
_cache: dict = {}


def rec() -> Recommender:
    if "r" not in _cache:
        try:
            _cache["r"] = Recommender(ART)
        except (FileNotFoundError, RuntimeError) as e:
            raise HTTPException(503, str(e))
    return _cache["r"]


@asynccontextmanager
async def lifespan(_app):
    try:
        r = rec()
        print(f"model {r.card['name']} loaded ({len(r.items):,} items) - http://localhost:8014", flush=True)
    except Exception as e:
        print(f"model not loaded: {getattr(e, 'detail', e)}", flush=True)
    yield


app = FastAPI(title="Generative retrieval with semantic IDs (TIGER)", version="1.0", description=DISCLAIMER,
              lifespan=lifespan)


class History(BaseModel):
    history: list[str] = Field(..., min_length=1, max_length=50, description="item IDs (parent_asin), oldest first")
    k: int = Field(10, ge=1, le=20)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui():
    return HTMLResponse(UI.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    r = _cache.get("r")
    return {"status": "ready" if r else "model not loaded", "model_version": r.card["name"] if r else None}


@app.get("/model-info")
def model_info():
    import json
    r = rec()
    m = {k: v for k, v in r.metrics.items() if k not in ("history",)}
    fixed = ART / "sasrec_fixed.json"                  # SASRec re-run after the evaluation bug fix (same users)
    if fixed.exists() and "sasrec" in m.get("results", {}):
        f = json.loads(fixed.read_text())
        m["results"] = {**m["results"], "sasrec": {k: f[k] for k in ("val", "test", "test_cold", "test_warm")}}
        m["sasrec_note"] = f.get("note")
    return {"version": r.card, "metrics": m}


@app.get("/samples")
def samples():
    r = rec()
    return [{**s, "history_items": [r.item(r.idx[a]) for a in s["history"] if a in r.idx],
             "target_item": r.item(r.idx[s["target"]]) if s["target"] in r.idx else None} for s in r.samples]


@app.get("/search")
def search(q: str = Query(..., min_length=2, max_length=100)):
    return rec().search(q)


@app.get("/items/{asin}")
def item(asin: str):
    r = rec()
    if asin not in r.idx:
        raise HTTPException(404, "item not in this model's catalogue")
    return r.item(r.idx[asin])


@app.post("/recommend")
def recommend(h: History):
    try:
        return rec().recommend(h.history, h.k)
    except ValueError as e:
        raise HTTPException(422, str(e))
