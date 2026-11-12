"""Model versions (same standard as the COUGHVID and chest X-ray projects): semantic versioning, one folder per training run,
a model card with the training-data hash, metrics and file hashes, never overwritten.

  MAJOR new architecture family / inputs / outputs   MINOR retrained   PATCH threshold or calibration fix
  folder: artifacts/tiger_v<MAJOR.MINOR.PATCH>_<arch>_<YYYYMMDD>/ ; artifacts/LATEST names the served version

  python -m tiger.versioning --list
  python -m tiger.versioning --use 1.0.0        # serve another version (rollback)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import stat
import time
from pathlib import Path

PROJECT = "tiger"
SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def parse(v: str):
    m = SEMVER.match(str(v).lstrip("v"))
    if not m:
        raise ValueError(f"version '{v}' is not MAJOR.MINOR.PATCH")
    return tuple(int(x) for x in m.groups())


def bump(v, part):
    if v is None:
        return "1.0.0"
    a, b, c = parse(v)
    return {"major": f"{a + 1}.0.0", "minor": f"{a}.{b + 1}.0", "patch": f"{a}.{b}.{c + 1}"}[part]


def load_registry(root: Path) -> dict:
    p = Path(root) / "registry.json"
    return json.loads(p.read_text()) if p.exists() else {"project": PROJECT, "versions": []}


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def new_version(root: Path, part: str = "minor", arch: str = "model") -> tuple[str, Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    vs = [v["version"] for v in load_registry(root)["versions"]]
    v = bump(max(vs, key=parse) if vs else None, part)
    d = root / f"{PROJECT}_v{v}_{arch}_{time.strftime('%Y%m%d')}"
    if d.exists() and not (d / "version.json").exists():          # unfinished earlier run of the same version
        import shutil
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=False)
    return v, d


def code_sha256() -> str:
    pkg = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(pkg.glob("*.py")):
        h.update(p.name.encode() + p.read_bytes())
    return h.hexdigest()


def register(root: Path, vdir: Path, version: str, part: str, data_sha256: str, data_summary: dict,
             metrics: dict, notes: str = "") -> dict:
    root, vdir = Path(root), Path(vdir)
    if (vdir / "version.json").exists():
        raise FileExistsError(f"{vdir.name} is already registered - versions are never overwritten")
    dep = metrics.get("deployed")
    res = metrics.get("results", {})
    from importlib import metadata
    soup = {}
    for pkg in ("torch", "torchvision", "numpy", "scikit-learn", "transformers", "sentence-transformers", "scikit-learn"):
        try:
            soup[pkg] = metadata.version(pkg)
        except Exception:
            soup[pkg] = None
    card = {"name": vdir.name, "project": PROJECT, "version": version, "bump": part, "arch": dep,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "notes": notes, "code_sha256": code_sha256(),
            "training_data_sha256": data_sha256, "data": data_summary,
            "category": metrics.get("category"),
            "semantic_ids": metrics.get("semantic_ids"),
            "val_recall10": res.get(dep, {}).get("val", {}).get("recall@10"),
            "test_recall10": res.get(dep, {}).get("test", {}).get("recall@10"),
            "test": res.get(dep, {}).get("test"),
            "metrics_file": "metrics.json", "python": platform.python_version(), "soup": soup,
            "file_sha256": {p.relative_to(vdir).as_posix(): file_sha256(p) for p in sorted(vdir.rglob("*")) if p.is_file()}}
    (vdir / "version.json").write_text(json.dumps(card, indent=2))
    for p in vdir.rglob("*"):
        if p.is_file():
            try:                                   # read-only; some file systems (Google Drive) ignore it
                p.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
            except OSError:
                pass
    reg = load_registry(root)
    reg["versions"] = [v for v in reg["versions"] if v["version"] != version] + [
        {k: card[k] for k in ("name", "version", "bump", "arch", "created", "notes", "val_recall10", "test_recall10")}]
    reg["versions"].sort(key=lambda v: parse(v["version"]))
    (root / "registry.json").write_text(json.dumps(reg, indent=2))
    (root / "LATEST").write_text(vdir.name)
    return card


def resolve(root: Path) -> Path:
    root = Path(root)
    if (root / "LATEST").exists():
        return root / (root / "LATEST").read_text().strip()
    cands = sorted(p for p in root.glob(f"{PROJECT}_v*") if (p / "model.pt").exists())
    if not cands:
        raise FileNotFoundError(f"no model version in {root} - train on Colab and copy the version folder here")
    return cands[-1]


def verify(vdir: Path) -> list[str]:
    card = json.loads((Path(vdir) / "version.json").read_text())
    return [n for n, h in card.get("file_sha256", {}).items()
            if not (Path(vdir) / n).exists() or file_sha256(Path(vdir) / n) != h]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--use")
    a = ap.parse_args()
    root = Path(a.artifacts)
    if a.use:
        hit = [v["name"] for v in load_registry(root)["versions"] if v["version"] == a.use.lstrip("v")]
        if not hit:
            raise SystemExit(f"version {a.use} not found")
        (root / "LATEST").write_text(hit[0])
    cur = (root / "LATEST").read_text().strip() if (root / "LATEST").exists() else None
    for v in load_registry(root)["versions"]:
        print(f"{'*' if v['name'] == cur else ' '} {v['name']:40s} val R@10 {v['val_recall10']}  test R@10 "
              f"{v['test_recall10']}  {v['notes']}")


if __name__ == "__main__":
    main()
