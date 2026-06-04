#!/usr/bin/env python3
"""Reproducible benchmarks for the doc-rag pipeline.

Generates a synthetic corpus of N chunks (pure Python, no real
documents required), then measures three things end-to-end:

  1. Embedding encode throughput (CPU or GPU, whatever sentence-transformers
     finds — controlled via env DOC_RAG_BENCH_DEVICE=cpu|cuda).
  2. FAISS index build (IndexFlatIP) for the embedded vectors.
  3. FAISS reconstruct-on-delete throughput — the deletion-prune path
     that is the headline optimisation in this project.

The benchmark deliberately does **not** exercise the document parser:
parsing speed depends on document format and OCR availability, which
is orthogonal to the chunk-pipeline speed this script is here to
measure.

Output is a single JSON document on stdout (also written to
`build/bench/<size>-<device>-<ts>.json` if --save is given). The JSON
schema is stable enough to track across releases; see
`docs/bench-results.md` for the published numbers.

Usage:
    python scripts/bench.py --size 1000
    python scripts/bench.py --size 10000 --save
    DOC_RAG_BENCH_DEVICE=cuda python scripts/bench.py --size 25000 --save

Exit status:
    0 — benchmark completed
    1 — missing required runtime dependencies (sentence-transformers
        or faiss); pip install -e .[faiss,embeddings] to fix
    2 — bad arguments
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _have_runtime_deps() -> bool:
    try:
        import faiss  # noqa: F401
        import numpy  # noqa: F401
        from sentence_transformers import SentenceTransformer  # noqa: F401
        return True
    except Exception:
        return False


# A tiny pool of plausible engineering-document sentences. Combined
# at random into chunks of roughly target_words words each. We do not
# import Lorem Ipsum: the resulting text has the right token density
# for embedding models trained on technical English.
_SENTENCE_POOL = [
    "The acceptance criterion for input voltage shall be 230 V plus or minus five percent.",
    "Operation under nominal load is permitted for ambient temperatures between fifteen and twenty-five degrees Celsius.",
    "The protective enclosure must satisfy ingress protection class IP54 or higher.",
    "Measurement uncertainty is reported with a coverage factor of two.",
    "The control loop tunes via a proportional integral derivative regulator with an anti-windup branch.",
    "Calibration certificates remain valid for a period of twelve months from the date of issue.",
    "Network traffic on the supervisory channel shall not exceed ten megabits per second sustained.",
    "Each output relay is rated for one million mechanical operations under resistive load.",
    "Diagnostic events are written to the device journal in compliance with the host facility log policy.",
    "Firmware update procedures require a signed bundle authenticated against the vendor public key.",
    "The redundancy scheme operates in active-passive mode with a heartbeat interval of one second.",
    "An alarm is raised when the rolling average exceeds the configured limit by more than twenty percent.",
]


def _make_synthetic_chunks(n: int, target_words: int = 80, seed: int = 0) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    chunks: List[Dict[str, Any]] = []
    for i in range(n):
        words: List[str] = []
        # Splice random sentences until we hit the target word count.
        while len(words) < target_words:
            words.extend(rng.choice(_SENTENCE_POOL).split())
        text = " ".join(words[: target_words + rng.randint(-5, 5)])
        chunks.append(
            {
                "chunk_id": f"doc-{i // 50:04d}:{i % 50:04d}",
                "doc_id": f"doc-{i // 50:04d}",
                "text": text,
                "source_file": f"doc-{i // 50:04d}.pdf",
            }
        )
    return chunks


def _bench(
    *,
    n_chunks: int,
    model_name: str,
    device: str,
    batch_size: int,
    deletion_fraction: float,
) -> Dict[str, Any]:
    import numpy as np
    import faiss  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore

    chunks = _make_synthetic_chunks(n_chunks)

    # 1. Encode
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name, device=device)
    t_model_load = time.perf_counter() - t0

    texts = [c["text"] for c in chunks]
    t0 = time.perf_counter()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype("float32")
    t_encode = time.perf_counter() - t0
    dim = int(vectors.shape[1])

    # 2. FAISS build
    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    t_index_build = time.perf_counter() - t0

    # 3. FAISS reconstruct-on-delete
    n_delete = max(1, int(n_chunks * deletion_fraction))
    doomed_doc_ids = {chunks[i]["doc_id"] for i in range(0, n_delete)}
    chunk_ids = [c["chunk_id"] for c in chunks]
    kept_pairs = [(i, cid) for i, cid in enumerate(chunk_ids) if cid.rsplit(":", 1)[0] not in doomed_doc_ids]

    t0 = time.perf_counter()
    new_idx = faiss.IndexFlatIP(dim)
    kept_vecs = np.zeros((len(kept_pairs), dim), dtype=np.float32)
    for new_i, (old_pos, _) in enumerate(kept_pairs):
        kept_vecs[new_i] = index.reconstruct(int(old_pos))
    new_idx.add(kept_vecs)
    t_reconstruct = time.perf_counter() - t0

    return {
        "model_load_s": round(t_model_load, 3),
        "encode_s": round(t_encode, 3),
        "encode_chunks_per_s": round(n_chunks / max(t_encode, 1e-9), 2),
        "encode_ms_per_chunk": round(1000 * t_encode / max(n_chunks, 1), 3),
        "index_build_s": round(t_index_build, 3),
        "reconstruct_s": round(t_reconstruct, 3),
        "reconstruct_chunks_kept": len(kept_pairs),
        "reconstruct_chunks_removed": n_chunks - len(kept_pairs),
        "vector_dim": dim,
    }


def _platform_info() -> Dict[str, Any]:
    import multiprocessing

    info = {
        "ts_local": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": multiprocessing.cpu_count(),
    }
    # Try to capture cuda/torch info if importable; not required.
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = None
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=1000, help="number of synthetic chunks (default 1000)")
    parser.add_argument(
        "--model",
        default=os.environ.get("DOC_RAG_BENCH_MODEL", "BAAI/bge-small-en-v1.5"),
        help="sentence-transformers model id (default bge-small for fast smoke benches; bge-large for prod-equivalent)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DOC_RAG_BENCH_DEVICE", "cpu"),
        choices=("cpu", "cuda", "auto"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--deletion-fraction",
        type=float,
        default=0.1,
        help="fraction of doc_ids to mark for deletion in the reconstruct benchmark",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="also write results to build/bench/<size>-<device>-<ts>.json",
    )
    args = parser.parse_args()

    if not _have_runtime_deps():
        sys.stderr.write(
            "missing runtime deps. pip install -e .[faiss,embeddings] then retry.\n"
        )
        return 1

    device = args.device
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    plat = _platform_info()
    plat["device"] = device
    plat["model"] = args.model
    plat["batch_size"] = args.batch_size

    print(f"# bench: size={args.size} device={device} model={args.model}", file=sys.stderr)
    results = _bench(
        n_chunks=args.size,
        model_name=args.model,
        device=device,
        batch_size=args.batch_size,
        deletion_fraction=args.deletion_fraction,
    )

    out = {
        "schema_version": 1,
        "size": args.size,
        "platform": plat,
        "results": results,
    }
    sys.stdout.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")

    if args.save:
        bench_dir = _REPO_ROOT / "build" / "bench"
        bench_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        fname = f"{args.size}-{device}-{ts}.json"
        (bench_dir / fname).write_text(
            json.dumps(out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"# saved → {bench_dir / fname}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
