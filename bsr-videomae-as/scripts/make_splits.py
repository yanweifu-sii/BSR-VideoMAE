#!/usr/bin/env python
"""Build per-subject split manifests from preprocessed benchmark folders.

Expected layout (produced by ``preprocess/process_*.py``)::

    <data-root>/<corpus>_fewshot/subject<N>/{train,val|validation,test}/*.pkl

Each pkl is the dict written by ``preprocess/utils.py::save_eeg``:
``{"eeg": {CH: [L]}, "label": <int|str>}``.

The generated manifest matches the schema consumed by ``finetune.py``:
``{schema, corpus, subject, labels, train, validation, test, full_train}`` with
rows ``{path, sha256, label}``.  Paths are absolute for the local machine; the
lock generator (``make_lock.py``) pins the per-subject manifest hashes so a
public user can re-generate a consistent lock locally.

Usage::

    python scripts/make_splits.py --data-root /data/eeg
    python scripts/make_splits.py --data-root /data/eeg --corpus bcic2a
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

CORPORA = ("bcic2a", "seed", "seedvii", "shumi")
#: corpora whose protocol requires a non-empty official validation split.
NEEDS_VAL = ("seed", "shumi")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(value):
    """finetune.py normalizes pickle labels with str(...); keep manifests identical."""
    if hasattr(value, "item"):
        value = value.item()
    return str(value)


def read_label(path: Path):
    with path.open("rb") as f:
        sample = pickle.load(f)
    if not isinstance(sample, dict) or "label" not in sample:
        raise SystemExit(f"{path}: pkl is not a save_eeg dict with a label")
    return jsonable(sample["label"])


def split_rows(directory: Path):
    rows = []
    for path in sorted(directory.glob("*.pkl")):
        rows.append({"path": str(path.resolve()), "sha256": sha_file(path),
                     "label": read_label(path)})
    return rows


def build_subject(corpus: str, subject_dir: Path) -> dict:
    sub = subject_dir.name
    splits = {"train": subject_dir / "train",
              "validation": subject_dir / "val",
              "test": subject_dir / "test"}
    if not splits["validation"].is_dir():
        splits["validation"] = subject_dir / "validation"
    rows = {}
    for key, directory in splits.items():
        rows[key] = split_rows(directory) if directory.is_dir() else []
    if not rows["train"] or not rows["test"]:
        raise SystemExit(f"{corpus}/{sub}: need non-empty train and test directories")
    if corpus in NEEDS_VAL and not rows["validation"]:
        raise SystemExit(f"{corpus}/{sub}: strict protocol needs a non-empty val/validation split")
    labels = sorted({jsonable(r["label"]) for part in ("train", "validation", "test")
                     for r in rows[part]}, key=str)
    manifest = {"schema": "splits-dirs-v22", "corpus": corpus, "subject": sub,
                "labels": labels, "train": rows["train"],
                "validation": rows["validation"], "test": rows["test"],
                "full_train": rows["train"], "selection_eligible": True}
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True,
                        help="root containing <corpus>_fewshot/subject<N>/...")
    parser.add_argument("--out", default="manifests/splits_dirs")
    parser.add_argument("--corpus", action="append", choices=CORPORA, default=[])
    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out)
    corpora = args.corpus or list(CORPORA)
    total = 0
    for corpus in corpora:
        root = data_root / f"{corpus}_fewshot"
        if not root.is_dir():
            raise SystemExit(f"missing {root}")
        subject_dirs = sorted([p for p in root.iterdir()
                               if p.is_dir() and p.name.startswith("subject")],
                              key=lambda p: int(p.name.replace("subject", "")))
        if not subject_dirs:
            raise SystemExit(f"no subject* dirs under {root}")
        for subject_dir in subject_dirs:
            manifest = build_subject(corpus, subject_dir)
            dst = out_root / corpus / f"{subject_dir.name}.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(manifest, indent=1) + "\n")
            total += 1
        print(f"{corpus}: {len(subject_dirs)} subjects -> {out_root / corpus}")
    print(f"wrote {total} subject manifests under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
