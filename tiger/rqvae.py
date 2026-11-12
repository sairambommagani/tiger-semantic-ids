"""RQ-VAE: turns each item's content embedding into a semantic ID (a short tuple of codewords).

encoder  MLP 768 -> 512 -> 256 -> 128 -> 32 (latent z)
quantise residual quantisation with L codebooks of K vectors each: c1 = nearest codeword to z, c2 = nearest to the
         residual z - e(c1), ... ; z_hat = e(c1) + e(c2) + e(c3)
decoder  MLP back to the embedding; loss = reconstruction MSE + codebook + 0.25 x commitment (straight-through)
Codebooks are initialised with k-means on the first batch; unused ("dead") codewords are re-seeded from random
latents during training, so the codebooks stay used.

Items that share all L codewords get an extra collision token (0, 1, 2, ...) so every item has a unique ID:
  semantic ID = (c1, c2, c3, collision)
Similar items share prefixes: c1 is a coarse cluster, c2 refines it, c3 refines further.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(dims):
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.SiLU()]
    return nn.Sequential(*layers[:-1])


class RQVAE(nn.Module):
    def __init__(self, in_dim=768, hidden=(512, 256, 128), latent=32, levels=3, codes=256, beta=0.25):
        super().__init__()
        self.enc = mlp([in_dim, *hidden, latent])
        self.dec = mlp([latent, *hidden[::-1], in_dim])
        self.codebooks = nn.ParameterList([nn.Parameter(torch.randn(codes, latent) * 0.1) for _ in range(levels)])
        self.levels, self.codes, self.beta = levels, codes, beta

    @torch.no_grad()
    def kmeans_init(self, x, iters=20):
        r = self.enc(x)
        for cb in self.codebooks:
            c = r[torch.randperm(len(r))[:self.codes]].clone()
            for _ in range(iters):
                a = torch.cdist(r, c).argmin(1)
                for k in range(self.codes):
                    m = a == k
                    if m.any():
                        c[k] = r[m].mean(0)
            cb.data.copy_(c)
            r = r - c[torch.cdist(r, c).argmin(1)]

    def quantise(self, z):
        r, zq, codes, closs = z, torch.zeros_like(z), [], 0.0
        for cb in self.codebooks:
            idx = torch.cdist(r, cb).argmin(1)
            e = cb[idx]
            closs = closs + F.mse_loss(e, r.detach()) + self.beta * F.mse_loss(r, e.detach())
            zq = zq + e
            r = r - e.detach()
            codes.append(idx)
        return zq, torch.stack(codes, 1), closs

    def forward(self, x):
        z = self.enc(x)
        zq, codes, closs = self.quantise(z)
        zq = z + (zq - z).detach()                         # straight-through estimator
        return self.dec(zq), codes, closs

    @torch.no_grad()
    def codes_for(self, emb) -> np.ndarray:
        """Raw item embeddings (numpy) -> (N, levels) codes, with the training standardisation."""
        x = torch.as_tensor(np.asarray(emb), dtype=torch.float32, device=self.mu.device)
        return self.encode((x - self.mu) / self.sd)

    @torch.no_grad()
    def encode(self, x, bs=4096):
        self.eval()
        return torch.cat([self.quantise(self.enc(x[i:i + bs]))[1] for i in range(0, len(x), bs)]).cpu().numpy()


def train(emb: np.ndarray, dev, levels=3, codes=256, latent=32, steps=6000, bs=1024, lr=1e-3, seed=0, log=print):
    torch.manual_seed(seed)
    x = torch.as_tensor(emb, dtype=torch.float32, device=dev)
    # Sentence embeddings are anisotropic: all items sit in a narrow cone (cosine ~0.7-0.9 to each other), so the
    # raw vectors differ only in tiny residuals and the codebooks collapse onto one code. Centre and scale each
    # dimension first (stored with the model, so new items are encoded the same way).
    mu, sd = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp(min=1e-6)
    x = (x - mu) / sd
    codes = min(codes, max(2, len(x) // 4))
    m = RQVAE(x.shape[1], latent=latent, levels=levels, codes=codes).to(dev)
    m.register_buffer("mu", mu.clone()); m.register_buffer("sd", sd.clone())
    m.kmeans_init(x[torch.randperm(len(x))[:min(len(x), 20000)]])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    t0 = time.time()
    hist = []
    for step in range(1, steps + 1):
        m.train()
        b = x[torch.randint(0, len(x), (min(bs, len(x)),), device=dev)]
        rec, c, closs = m(b)
        rloss = F.mse_loss(rec, b)
        loss = rloss + closs
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == steps:
            with torch.no_grad():                          # re-seed dead codewords
                allc = torch.as_tensor(m.encode(x), device=dev)
                used = [torch.bincount(allc[:, l], minlength=m.codes) > 0 for l in range(levels)]
                r = m.enc(x[torch.randperm(len(x))[:4096]])
                for l, cb in enumerate(m.codebooks):
                    dead = (~used[l]).nonzero().flatten()
                    if len(dead) and step < steps:
                        cb.data[dead] = r[torch.randint(0, len(r), (len(dead),))] + 0.01 * torch.randn(len(dead), cb.shape[1], device=dev)
                    r = r - cb[torch.cdist(r, cb).argmin(1)]
                st = id_stats(allc.cpu().numpy(), m.codes)
            hist.append({"step": step, "recon_mse": round(rloss.item(), 5), "codebook_loss": round(closs.item(), 5),
                         "unique_prefix_pct": st["unique_full_code_pct"], "usage": st["codebook_usage_pct"]})
            log(f"  rq-vae step {step}/{steps}: recon {rloss.item():.5f}  codebook {closs.item():.5f}  "
                f"unique items {st['unique_full_code_pct']}%  codebook usage {st['codebook_usage_pct']}  ({time.time() - t0:.0f} s)")
    if st["unique_full_code_pct"] < 50:
        raise RuntimeError(f"RQ-VAE collapsed: only {st['unique_full_code_pct']}% distinct codes, usage "
                           f"{st['codebook_usage_pct']} - check the embeddings")
    return m, hist


def add_collision_token(codes: np.ndarray) -> np.ndarray:
    """(N, L) -> (N, L + 1): last column numbers the items that share the same L codewords."""
    out = np.zeros((len(codes), codes.shape[1] + 1), np.int64)
    out[:, :-1] = codes
    seen = {}
    for i, c in enumerate(map(tuple, codes)):
        out[i, -1] = seen.get(c, 0)
        seen[c] = out[i, -1] + 1
    return out


def id_stats(codes: np.ndarray, k: int) -> dict:
    full = [tuple(c) for c in codes]
    _, cnt = np.unique(np.asarray(full), axis=0, return_counts=True)
    return {"codebook_usage_pct": [round(100 * len(np.unique(codes[:, l])) / k, 1) for l in range(codes.shape[1])],
            "unique_full_code_pct": round(100 * len(cnt) / len(codes), 2),
            "max_items_per_code": int(cnt.max()), "items_with_collision_pct": round(100 * float(cnt[cnt > 1].sum()) / len(codes), 2)}


def random_ids(n: int, levels: int, k: int, seed: int = 0) -> np.ndarray:
    """Ablation: random codes with the same shape (no content meaning)."""
    return np.random.default_rng(seed).integers(0, k, (n, levels))


def prefix_purity(codes: np.ndarray, labels: list[str]) -> dict:
    """How often items that share the first codeword (and the first two) share the same leaf category: shows that
    semantic IDs group similar items."""
    lab = np.asarray(labels)
    out = {}
    for depth in (1, 2):
        keys = [tuple(c[:depth]) for c in codes]
        groups = {}
        for i, k_ in enumerate(keys):
            groups.setdefault(k_, []).append(i)
        pur, tot = 0, 0
        for g in groups.values():
            if len(g) < 2:
                continue
            _, c = np.unique(lab[g], return_counts=True)
            pur += c.max(); tot += len(g)
        out[f"prefix{depth}_majority_category_pct"] = round(100 * pur / max(tot, 1), 1)
    _, c = np.unique(lab, return_counts=True)
    out["chance_pct"] = round(100 * c.max() / len(lab), 1)
    return out
