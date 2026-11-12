"""Baselines on the same split and the same full-ranking evaluation.

popularity  most frequent training items
SASRec      self-attentive sequential recommender (Kang & McAuley 2018): one learned embedding per item ID, 2 causal
            transformer blocks, d 64, BCE with one sampled negative per position, full ranking at test.
            It cannot rank an item it never saw in training (its embedding stays random) - the comparison with
            TIGER on cold items shows the value of content-based semantic IDs.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

MAXLEN = 20


def popularity(train_seqs, n_items, k=10):
    cnt = np.bincount(np.concatenate([np.asarray(s, int) for s in train_seqs if len(s)]), minlength=n_items)
    return list(np.argsort(-cnt)[:k])


class SASRec(nn.Module):
    def __init__(self, n_items, d=64, blocks=2, heads=2, dropout=0.2, maxlen=MAXLEN):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, d, padding_idx=0)       # id 0 = padding, item i -> i + 1
        self.pos = nn.Embedding(maxlen, d)
        layer = nn.TransformerEncoderLayer(d, heads, d * 2, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, blocks)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.maxlen = maxlen
        # small initial embeddings: with the default N(0, 1) init, the many rarely-trained items of a large catalogue
        # keep big random vectors and flood the top of a full ranking
        nn.init.normal_(self.item.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        with torch.no_grad():
            self.item.weight[0].zero_()

    def forward(self, seq):                                            # (B, T) item ids + 1, left-padded
        T = seq.shape[1]
        h = self.drop(self.item(seq) * self.item.embedding_dim ** 0.5 + self.pos(torch.arange(T, device=seq.device)))
        h = h * (seq != 0).unsqueeze(-1)                   # padding positions carry no signal (as in the original SASRec)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=seq.device), 1)
        # causal mask only: a key-padding mask on left-padded sequences leaves the first rows with every key masked,
        # which gives NaN in evaluation mode
        return self.norm(self.enc(h, mask=mask))

    def scores(self, seq):
        return self(seq)[:, -1] @ self.item.weight[1:].T


def _pad(seqs, maxlen):
    out = np.zeros((len(seqs), maxlen), np.int64)
    for i, s in enumerate(seqs):
        s = list(s)[-maxlen:]
        if s:
            out[i, -len(s):] = np.asarray(s) + 1
    return out


def train_sasrec(train_seqs, n_items, dev, epochs=30, bs=256, lr=1e-3, seed=0, eval_fn=None, eval_every=5, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = [s for s in train_seqs if len(s) >= 2]
    X = torch.as_tensor(_pad([s[:-1] for s in data], MAXLEN))
    Y = torch.as_tensor(_pad([s[1:] for s in data], MAXLEN))
    m = SASRec(n_items).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr, betas=(0.9, 0.98))
    best, best_state, hist = -1.0, None, []
    for ep in range(epochs):
        m.train()
        t0, tot, nb = time.time(), 0.0, 0
        for i in range(0, len(X), bs):
            b = torch.arange(i, min(i + bs, len(X)))
            x, y = X[b].to(dev), Y[b].to(dev)
            neg = torch.randint(1, n_items + 1, y.shape, device=dev)
            h = m(x)
            pos_l = (h * m.item(y)).sum(-1); neg_l = (h * m.item(neg)).sum(-1)
            mask = (y != 0).float()
            loss = -((torch.nn.functional.logsigmoid(pos_l) + torch.nn.functional.logsigmoid(-neg_l)) * mask).sum() / mask.sum()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        perm = rng.permutation(len(X)); X, Y = X[perm], Y[perm]
        rec = {"epoch": ep + 1, "train_loss": round(tot / nb, 4), "seconds": round(time.time() - t0, 1)}
        if eval_fn and ((ep + 1) % eval_every == 0 or ep + 1 == epochs):
            rec["val_recall@10"] = eval_fn(m)
            if rec["val_recall@10"] > best:
                best = rec["val_recall@10"]
                best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            log(f"  sasrec epoch {ep + 1}/{epochs}: loss {rec['train_loss']:.4f}  val Recall@10 {rec['val_recall@10']:.4f}")
        hist.append(rec)
    if best_state:
        m.load_state_dict(best_state)
    return m, hist


@torch.no_grad()
def sasrec_topk(m, histories, dev, k=10, bs=512):
    m.eval()
    out = []
    for i in range(0, len(histories), bs):
        x = torch.as_tensor(_pad(histories[i:i + bs], MAXLEN), device=dev)
        out += torch.topk(m.scores(x), k, dim=1).indices.cpu().tolist()
    return out
