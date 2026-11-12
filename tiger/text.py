"""Item content embeddings.

Default: Sentence-T5 (sentence-transformers/sentence-t5-base, 768-d), the encoder family TIGER used.
Fallback for offline tests: TF-IDF + truncated SVD (no download).
"""
from __future__ import annotations

import numpy as np

DEFAULT_ENCODER = "sentence-transformers/sentence-t5-base"


def embed(texts: list[str], encoder: str = DEFAULT_ENCODER, dev: str | None = None, bs: int = 256, log=print) -> np.ndarray:
    if encoder == "tfidf":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        x = TfidfVectorizer(min_df=1, sublinear_tf=True).fit_transform(texts)
        k = max(2, min(64, x.shape[1] - 1, x.shape[0] - 1))
        e = TruncatedSVD(k, random_state=0).fit_transform(x)
        return (e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)).astype(np.float32)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(encoder, device=dev)
    return m.encode(texts, batch_size=bs, show_progress_bar=False, convert_to_numpy=True,
                    normalize_embeddings=True).astype(np.float32)
