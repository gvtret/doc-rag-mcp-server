#!/usr/bin/env python3
"""One-time download of the BAAI/bge-m3 embedding model into the repo.

Downloads a HF snapshot into ``<dest>.staging``, validates the
sentence-transformers marker files, fingerprints the weights (WEIGHTS.sha256
via doc_rag.raglib.model_paths), then atomically swaps it into place:

    <dest>.staging -> <dest>  (existing <dest> moved to <dest>.old first)

Examples:
    python scripts/fetch_model.py
    python scripts/fetch_model.py --model BAAI/bge-m3 --dest models/bge-m3 --force

Exit codes: 0 on success, 1 on any download/validation failure (the
incomplete staging directory is always removed).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

# Make doc_rag.raglib.model_paths importable both when run from a repo
# checkout (src/ layout, not installed) and from an installed package.
# This repo's src/ must be prepended BEFORE the first doc_rag import:
# if another checkout's src/ is already on sys.path (e.g. an editable
# install of the main workspace), doc_rag.raglib resolves there as a
# namespace package and its __path__ never picks up later sys.path
# entries, so a fallback insert after a failed import does not help.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from doc_rag.raglib.model_paths import MODEL_MARKER_FILES, model_fingerprint, weight_files

# Large derived artefacts we never need server-side: ONNX exports and
# documentation images ship inside the bge-m3 repo but are not weights.
_IGNORE_PATTERNS = ["onnx/*", "imgs/*"]

_DEFAULT_MODEL = "BAAI/bge-m3"
_DEFAULT_DEST = "models/bge-m3"


def download_model(model: str, dest: Path, *, force: bool = False) -> dict[str, Any]:
    """Download ``model`` into ``dest``, returning dest/fingerprint/file count.

    The snapshot lands in ``<dest>.staging``; on any error that directory
    is removed in ``finally`` and the exception propagates. When ``dest``
    already exists it is renamed to ``<dest>.old`` first and removed after
    the staging directory takes its place.
    """
    dest = Path(dest).expanduser()
    staging = Path(str(dest) + ".staging")
    old = Path(str(dest) + ".old")
    # The HF read timeout from the spec; total wall-clock budget is unbounded.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

    if force and staging.exists():
        shutil.rmtree(staging)

    try:
        snapshot_download(
            repo_id=model,
            local_dir=staging,
            ignore_patterns=_IGNORE_PATTERNS,
        )
        for name in MODEL_MARKER_FILES:
            if not (staging / name).is_file():
                raise RuntimeError(f"incomplete snapshot: missing {name} in {staging}")
        # Also writes the WEIGHTS.sha256.meta.json inventory sidecar.
        fingerprint = model_fingerprint(staging)
        if fingerprint is None:
            raise RuntimeError(f"no weight files found in {staging}")
        _remove_stale_cache(staging)

        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if old.exists():
                shutil.rmtree(old)
            os.replace(dest, old)
        try:
            os.replace(staging, dest)
        except OSError:
            # The swap failed after dest was moved aside: restore the
            # working model dir so the machine is never left without one.
            if old.exists() and not dest.exists():
                os.replace(old, dest)
            raise
        if old.exists():
            shutil.rmtree(old)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    weight_files_n = len(weight_files(dest))
    return {"dest": str(dest), "fingerprint": fingerprint, "weight_files": weight_files_n}


def _remove_stale_cache(snapshot_dir: Path) -> None:
    """Drop the .cache/huggingface bookkeeping left inside the snapshot."""
    cache = snapshot_dir / ".cache" / "huggingface"
    if cache.exists():
        shutil.rmtree(cache, ignore_errors=True)
        # Leave no empty .cache parent behind either; a non-empty
        # .cache (foreign content) must not fail an otherwise valid
        # download.
        try:
            snapshot_dir.joinpath(".cache").rmdir()
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download the offline embedding model into the repo.")
    p.add_argument("--model", default=_DEFAULT_MODEL, help=f"HF repo id (default {_DEFAULT_MODEL})")
    p.add_argument(
        "--dest", default=_DEFAULT_DEST, help=f"destination directory (default {_DEFAULT_DEST})"
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="remove a leftover staging directory before downloading",
    )
    args = p.parse_args(argv)

    try:
        info = download_model(args.model, Path(args.dest), force=args.force)
    except Exception as e:
        print(f"fetch_model: FAILED: {e}", file=sys.stderr)
        return 1
    print(
        f"fetch_model: OK: {args.model} -> {info['dest']} "
        f"(fingerprint {info['fingerprint'][:12]}..., {info['weight_files']} weight files)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
