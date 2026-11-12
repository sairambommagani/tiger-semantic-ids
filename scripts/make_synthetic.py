"""Tiny fake Amazon 2023 benchmark files (ratings csv + metadata jsonl.gz), for the offline tests.

Users mostly buy within one product family, in order (e.g. cleanser -> toner -> serum), so a sequence model can
learn it, and item text names the family, so semantic IDs can group it.

  python -m scripts.make_synthetic out_dir
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FAMILIES = ["shampoo hair", "lipstick makeup", "serum skin", "perfume fragrance", "nail polish"]


def write(root: Path, users: int = 300, per_family: int = 12, seed: int = 0) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    items = [(f"B{f:02d}{j:03d}", fam, j) for f, fam in enumerate(FAMILIES) for j in range(per_family)]
    rows = []
    for u in range(users):
        f = rng.integers(len(FAMILIES))
        start = rng.integers(0, per_family - 6)
        n = rng.integers(5, 9)
        fam_items = [it for it in items if it[1] == FAMILIES[f]]
        for t in range(n):
            j = min(start + t, per_family - 1) if rng.random() < 0.8 else rng.integers(per_family)
            rows.append({"user_id": f"U{u:05d}", "parent_asin": fam_items[j][0], "rating": 5.0,
                         "timestamp": 1_600_000_000_000 + t * 86_400_000 + int(rng.integers(1000))})
    r = root / "ratings.csv.gz"
    pd.DataFrame(rows).to_csv(r, index=False, compression="gzip")
    m = root / "meta.jsonl.gz"
    with gzip.open(m, "wt") as f:
        for a, fam, j in items:
            f.write(json.dumps({"parent_asin": a, "title": f"{fam.split()[0].title()} number {j} {fam}",
                                "store": f"Brand{j % 3}", "categories": ["Beauty", fam.split()[1].title(), fam.split()[0].title()],
                                "price": float(5 + j), "features": [f"great {fam}"]}) + "\n")
        f.write(json.dumps({"parent_asin": "UNUSED", "title": "not in ratings"}) + "\n")
    return r, m


if __name__ == "__main__":
    print(write(Path(sys.argv[1] if len(sys.argv) > 1 else "data_synthetic")))
