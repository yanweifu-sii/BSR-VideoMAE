#!/usr/bin/env python
"""Generate a local ``final_v3_lock.json`` for a downloaded checkpoint.

The released ``manifests/lock_template.json`` carries the four benchmark
configurations from the paper/backbone release; this script fills in

  * ``checkpoint_sha256`` / ``checkpoint_path`` for the local checkpoint, and
  * per-subject ``split_sha256`` for the locally generated split manifests,

so ``finetune.py --stage fixed|strict`` can validate that the run matches the
preregistered protocol on *your* data copy.

Usage::

    python scripts/make_splits.py --data-root /data/eeg
    python scripts/make_lock.py --ckpt ckpts/S2B2A_c70ceb7e.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

TEMPLATE_DEFAULT = Path("manifests/lock_template.json")
OUT_DEFAULT = Path("manifests/final_v3_lock.json")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--splits-root", default="manifests/splits_dirs")
    parser.add_argument("--template", default=str(TEMPLATE_DEFAULT))
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    args = parser.parse_args()
    ckpt = Path(args.ckpt).resolve()
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint missing: {ckpt}")
    ckpt_sha = sha_file(ckpt)
    lock = json.loads(Path(args.template).read_text())
    splits_root = Path(args.splits_root)
    missing = []
    for corpus, task in lock["tasks"].items():
        corpus_root = splits_root / corpus
        subject_files = sorted(corpus_root.glob("subject*.json"),
                               key=lambda p: int(p.stem.replace("subject", "")))
        if not subject_files:
            missing.append(corpus)
            continue
        task["checkpoint_sha256"] = ckpt_sha
        task["checkpoint_path"] = str(ckpt)
        task["split_sha256"] = {corpus: {p.stem: sha_file(p) for p in subject_files}}
    if missing:
        raise SystemExit(f"no split manifests for: {missing}; run make_splits.py first")
    if "checkpoint_sha256" in lock:
        lock["checkpoint_sha256"] = ckpt_sha
    if "single_backbone_sha256" in lock:
        lock["single_backbone_sha256"] = ckpt_sha
    lock["checkpoint_path"] = str(ckpt)
    lock["generated_by"] = "scripts/make_lock.py"
    lock["ckpt_sha256"] = ckpt_sha
    Path(args.out).write_text(json.dumps(lock, indent=1) + "\n")
    print(f"wrote {args.out}: ckpt {ckpt_sha[:16]}..., "
          f"splits {sum(len(t['split_sha256'][c]) for c, t in lock['tasks'].items())} subject files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
