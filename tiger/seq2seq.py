"""TIGER generative retriever: a small T5 encoder-decoder over semantic-ID tokens.

Input  : [user token] + the semantic IDs of the last <= 20 items, flattened (4 tokens per item)
Output : the 4 tokens of the next item's semantic ID, then end-of-sequence
Vocabulary: pad, eos, USER_BUCKETS hashed user tokens, and one block of K tokens per ID position.
Inference: beam search over the decoder, constrained by a trie of valid IDs, so every generated ID is a real item
(the unconstrained invalid-ID rate is reported too). The top-k distinct items from the beams are the recommendations.

Model size (as in the paper): 4 encoder + 4 decoder layers, d_model 128, 6 heads of 64, feed-forward 1024, dropout 0.1.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

PAD, EOS = 0, 1
USER_BUCKETS = 2000


class Vocab:
    def __init__(self, sem_ids: np.ndarray, k: int):
        self.sem = np.asarray(sem_ids, np.int64)               # (n_items, L+1)
        self.L1 = self.sem.shape[1]
        self.k = max(k, int(self.sem.max()) + 1)
        self.base = 2 + USER_BUCKETS
        self.size = self.base + self.L1 * self.k
        self.tok = self.base + np.arange(self.L1)[None, :] * self.k + self.sem    # (n_items, L1) token ids
        self.lookup = {tuple(t): i for i, t in enumerate(self.tok.tolist())}
        self.trie = {}
        for t in self.tok.tolist():
            for d in range(self.L1):
                self.trie.setdefault(tuple(t[:d]), set()).add(t[d])
        self.trie = {p: sorted(v) for p, v in self.trie.items()}

    def user_token(self, u) -> int:
        return 2 + (hash(("u", u)) % USER_BUCKETS if not isinstance(u, (int, np.integer)) else int(u) % USER_BUCKETS)

    def encode_history(self, user, hist) -> list[int]:
        return [self.user_token(user)] + [int(x) for i in hist for x in self.tok[i]]


def build(vocab: Vocab, d_model=128, layers=4, heads=6, d_kv=64, d_ff=1024, dropout=0.1):
    from transformers import T5Config, T5ForConditionalGeneration
    cfg = T5Config(vocab_size=vocab.size, d_model=d_model, d_kv=d_kv, d_ff=d_ff, num_layers=layers,
                   num_decoder_layers=layers, num_heads=heads, dropout_rate=dropout, pad_token_id=PAD,
                   eos_token_id=EOS, decoder_start_token_id=PAD, feed_forward_proj="relu", tie_word_embeddings=True)
    return T5ForConditionalGeneration(cfg)


def _batch(vocab, rows, dev):
    """rows: (user, history, target) -> padded tensors."""
    src = [vocab.encode_history(u, h) for u, h, _ in rows]
    L = max(len(s) for s in src)
    ids = torch.full((len(src), L), PAD, dtype=torch.long)
    att = torch.zeros((len(src), L), dtype=torch.long)
    for i, s in enumerate(src):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
    lab = None
    if rows[0][2] is not None:
        lab = torch.tensor([list(vocab.tok[t]) + [EOS] for _, _, t in rows], dtype=torch.long)
    return ids.to(dev), att.to(dev), (lab.to(dev) if lab is not None else None)


def _batch_fast(vocab, users, hist, targets, dev):
    """Vectorised version of _batch for training: hist (B, H) item ids, -1 = padding."""
    B, H = hist.shape
    tok = np.where(hist[..., None] >= 0, vocab.tok[np.clip(hist, 0, None)], PAD).reshape(B, H * vocab.L1)
    ut = (2 + (users % USER_BUCKETS)).astype(np.int64)[:, None]
    ids = np.concatenate([ut, tok], 1)
    att = (ids != PAD).astype(np.int64)
    lab = np.concatenate([vocab.tok[targets], np.full((B, 1), EOS)], 1)
    return (torch.as_tensor(ids, device=dev), torch.as_tensor(att, device=dev), torch.as_tensor(lab, device=dev))


@torch.no_grad()
def recommend(model, vocab, rows, dev, k=10, beams=20, bs=128, constrained=True) -> tuple[list[list[int]], int]:
    """rows: (user, history, _). -> top-k item ids per row, number of invalid generated IDs (unconstrained only)."""
    model.eval()
    out, invalid = [], 0
    trie = vocab.trie

    def allowed(_b, sent):
        p = tuple(sent.tolist()[1:])
        return trie.get(p, [EOS]) if len(p) < vocab.L1 else [EOS]
    for i in range(0, len(rows), bs):
        ids, att, _ = _batch(vocab, [(u, h, None) for u, h, _ in rows[i:i + bs]], dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
          g = model.generate(input_ids=ids, attention_mask=att, num_beams=beams, num_return_sequences=beams,
                             max_new_tokens=vocab.L1 + 1, do_sample=False, early_stopping=True,
                             prefix_allowed_tokens_fn=allowed if constrained else None)
        g = g.view(len(ids), beams, -1).cpu().numpy()
        for row in g:
            rec = []
            for seq in row:
                t = tuple(int(x) for x in seq[1:1 + vocab.L1])
                it = vocab.lookup.get(t)
                if it is None:
                    invalid += 1
                    continue
                if it not in rec:
                    rec.append(it)
                if len(rec) == k:
                    break
            out.append(rec)
    return out, invalid


def train(model, vocab, examples, val_rows, dev, epochs=20, bs=256, lr=1e-3, warmup=0.05, eval_every=2,
          eval_fn=None, seed=0, patience=3, ckpt=None, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = max(1, epochs * int(np.ceil(len(examples) / bs)))
    w = max(1, int(warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / w, max(0.0, (steps - s) / max(steps - w, 1))))
    amp = dev.type == "cuda"                                # mixed precision: about 2-3x faster on a T4
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    skipped = 0
    best, best_state, hist, bad, start = -1.0, None, [], 0, 0
    if ckpt is not None and Path(ckpt).exists():           # resume after a Colab disconnect
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        best, best_state, hist, bad, start = c["best"], c["best_state"], c["hist"], c["bad"], c["epoch"]
        log(f"  resumed from {ckpt} after epoch {start}")
    for ep in range(start, epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        order = np.random.default_rng(seed + ep).permutation(len(examples))
        for i in range(0, len(order), bs):
            if hasattr(examples, "batch"):
                ids, att, lab = _batch_fast(vocab, *examples.batch(order[i:i + bs]), dev)
            else:
                ids, att, lab = _batch(vocab, [examples[j] for j in order[i:i + bs]], dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):                     # fp16 overflow: skip the batch (the scaler adapts)
                sched.step(); skipped += 1
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); n += 1
            if n % 1000 == 0:
                log(f"    step {n}: loss {tot / n:.4f} ({time.time() - t0:.0f} s)")
        rec = {"epoch": ep + 1, "train_loss": round(tot / max(n, 1), 4), "seconds": round(time.time() - t0, 1),
               "skipped_batches": skipped}
        if eval_fn and ((ep + 1) % eval_every == 0 or ep + 1 == epochs):
            r = eval_fn(model)
            rec["val_recall@10"] = r
            if r > best + 1e-4:
                best, bad = r, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        hist.append(rec)
        log(f"  tiger epoch {ep + 1}/{epochs}: loss {tot / n:.4f}" +
            (f"  val Recall@10 {rec['val_recall@10']:.4f}" if "val_recall@10" in rec else "") + f"  ({rec['seconds']:.0f} s)")
        if ckpt is not None:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "best": best,
                        "best_state": best_state, "hist": hist, "bad": bad, "epoch": ep + 1}, ckpt)
        if bad >= patience:
            log("  early stop")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist
