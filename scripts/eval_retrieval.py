#!/usr/bin/env python3
"""Retrieval-quality eval harness for doc-rag.

Runs a golden query set against the retrieval functions and reports
recall@10, MRR@10 and nDCG@10 at document level (and at chunk level when
relevant_chunk_ids are given). This is the measurement tool that turns
retrieval-quality changes from guesswork into before/after numbers.

The three search modes are exercised by calling hybrid_search /
semantic_search / lexical_search DIRECTLY: mode selection in doc_search()
is config-driven, so going through it would measure the config, not the
modes.

Usage:
    python scripts/eval_retrieval.py --root /path/to/corpus
    python scripts/eval_retrieval.py --root . --modes semantic,hybrid,lexical
    python scripts/eval_retrieval.py --root . --queries eval/queries_golden.yaml

Output: a human-readable table on stdout plus a JSON report under
build/eval/<ts>-<modes>.json with per-query and aggregate results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DEFAULT_TOP_K = 10
DEFAULT_MODES = ["semantic", "hybrid", "lexical"]
METRIC_K = 10


def load_queries(path: Path) -> list[dict[str, Any]]:
    """Read the golden queries YAML: list of {query, relevant_doc_ids,
    relevant_chunk_ids (optional)}."""
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a YAML list of query objects")
    out: list[dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict) or not item.get("query"):
            raise ValueError(f"{path}: entry #{i} must be a dict with a 'query' key")
        rel_docs = item.get("relevant_doc_ids") or []
        rel_chunks = item.get("relevant_chunk_ids") or []
        if not rel_docs and not rel_chunks:
            raise ValueError(
                f"{path}: entry #{i} ('{item['query'][:40]}') has neither "
                "relevant_doc_ids nor relevant_chunk_ids"
            )
        out.append(
            {
                "query": str(item["query"]),
                "relevant_doc_ids": [str(d) for d in rel_docs],
                "relevant_chunk_ids": [str(c) for c in rel_chunks],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Metrics (standard definitions, binary relevance gains)
# ---------------------------------------------------------------------------


def chunk_doc_id(chunk_id: str) -> str:
    """chunk_id -> doc_id: the part before the first colon."""
    return chunk_id.split(":")[0]


def doc_recall(ranked_doc_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    found = len(set(ranked_doc_ids[:k]) & relevant)
    return found / len(relevant)


def mrr(ranked: list[str], relevant: set[str], k: int) -> float:
    """1 / rank of the first relevant item, 0 when none in top-k."""
    for i, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def ndcg(ranked: list[str], relevant: set[str], k: int) -> float:
    """nDCG@k with binary gains: DCG = sum(rel_i / log2(i+1)) for position
    i starting at 1 (so the first position has gain 1/log2(2) = 1); IDCG
    is the ideal ordering = min(len(relevant), k) relevant items at the
    top."""
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def aggregate_scores(
    ranked_lists: list[list[str]], relevant_sets: list[set[str]], k: int
) -> dict[str, float]:
    """Macro-average recall@k, MRR@k, nDCG@k over per-query ranked lists.

    Queries with no relevance info for this level are skipped entirely
    (denominator = evaluated queries, not all queries).
    """
    recalls, mrrs, ndcgs = [], [], []
    for ranked, rel in zip(ranked_lists, relevant_sets, strict=False):
        if not rel:
            continue
        recalls.append(doc_recall(ranked, rel, k))
        mrrs.append(mrr(ranked, rel, k))
        ndcgs.append(ndcg(ranked, rel, k))
    n = len(recalls)
    if n == 0:
        return {"n_evaluated": 0, f"recall@{k}": 0.0, f"mrr@{k}": 0.0, f"ndcg@{k}": 0.0}
    return {
        "n_evaluated": n,
        f"recall@{k}": sum(recalls) / n,
        f"mrr@{k}": sum(mrrs) / n,
        f"ndcg@{k}": sum(ndcgs) / n,
    }


# ---------------------------------------------------------------------------
# Search-mode runners
# ---------------------------------------------------------------------------


def run_mode(
    mode: str,
    cfg: dict[str, Any],
    chunks: list[dict[str, Any]],
    query: str,
    top_k: int,
    namespace: str = "default",
) -> list[dict[str, Any]]:
    """Run one query in the given mode, calling the retrieval functions
    directly (doc_search would follow the config's retrieval_mode)."""
    from doc_rag.server import retrieval

    if mode == "semantic":
        res = retrieval.semantic_search(cfg, chunks, query, top_k, namespace=namespace)
        # None = index/model unavailable; fall through to lexical so the
        # eval still produces numbers but the JSON marks the degradation.
        return res if res is not None else []
    if mode == "hybrid":
        return retrieval.hybrid_search(cfg, chunks, query, top_k, namespace=namespace)
    if mode == "lexical":
        return retrieval.lexical_search(chunks, query, top_k)
    raise ValueError(f"unknown mode '{mode}'")


def evaluate_mode(
    mode: str,
    cfg: dict[str, Any],
    chunks: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    top_k: int,
    namespace: str = "default",
) -> dict[str, Any]:
    """Evaluate one mode over all queries; returns per-query + aggregate."""
    doc_ranked: list[list[str]] = []
    doc_relevant: list[set[str]] = []
    chunk_ranked: list[list[str]] = []
    chunk_relevant: list[set[str]] = []
    per_query: list[dict[str, Any]] = []

    for q in queries:
        t0 = time.time()
        results = run_mode(mode, cfg, chunks, q["query"], top_k, namespace)
        elapsed = time.time() - t0

        ranked_doc_ids: list[str] = []
        ranked_chunk_ids: list[str] = []
        for r in results:
            cid = str(r.get("chunk_id", ""))
            if cid:
                ranked_chunk_ids.append(cid)
            doc_id = str(r.get("doc_id", "") or chunk_doc_id(cid))
            if doc_id:
                ranked_doc_ids.append(doc_id)
        # Doc-level ranking should count each document once: a document
        # spread over many chunks would otherwise dominate nDCG/MRR.
        dedup_doc_ids: list[str] = []
        seen: set[str] = set()
        for d in ranked_doc_ids:
            if d not in seen:
                seen.add(d)
                dedup_doc_ids.append(d)

        want_docs = set(q["relevant_doc_ids"])
        want_chunks = set(q["relevant_chunk_ids"])
        per_query.append(
            {
                "query": q["query"],
                "relevant_doc_ids": sorted(want_docs),
                "relevant_chunk_ids": sorted(want_chunks),
                "returned_doc_ids": dedup_doc_ids,
                "returned_chunk_ids": ranked_chunk_ids,
                "latency_s": round(elapsed, 3),
            }
        )
        doc_ranked.append(dedup_doc_ids)
        doc_relevant.append(want_docs)
        chunk_ranked.append(ranked_chunk_ids)
        chunk_relevant.append(want_chunks)

    return {
        "mode": mode,
        "per_query": per_query,
        "doc_level": aggregate_scores(doc_ranked, doc_relevant, METRIC_K),
        "chunk_level": aggregate_scores(chunk_ranked, chunk_relevant, METRIC_K),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_table(aggregates: dict[str, dict[str, Any]], top_k: int) -> None:
    """Human-readable summary table on stdout."""
    levels = [("doc_level", "doc"), ("chunk_level", "chunk")]
    header = f"{'mode':<10} {'level':<6} {'recall@k':>9} {'MRR@k':>9} {'nDCG@k':>9} {'n_eval':>7}"
    print(f"\nRetrieval eval (top_k={top_k})")
    print(header)
    print("-" * len(header))
    for mode in aggregates:
        for level_key, level_name in levels:
            agg = aggregates[mode][level_key]
            print(
                f"{mode:<10} {level_name:<6} "
                f"{agg.get(f'recall@{METRIC_K}', 0.0):>9.3f} "
                f"{agg.get(f'mrr@{METRIC_K}', 0.0):>9.3f} "
                f"{agg.get(f'ndcg@{METRIC_K}', 0.0):>9.3f} "
                f"{agg.get('n_evaluated', 0):>7}"
            )


def save_report(
    out_dir: Path,
    modes: list[str],
    top_k: int,
    queries_path: Path,
    results: dict[str, dict[str, Any]],
    corpus_meta: dict[str, Any],
) -> Path:
    """Persist the full report (per-query + aggregate) as JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{ts}-{','.join(modes)}.json"
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "queries_file": str(queries_path),
        "modes": modes,
        "top_k": top_k,
        "metric_k": METRIC_K,
        "corpus": corpus_meta,
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieval-quality eval for doc-rag")
    parser.add_argument("--root", default=None, help="Corpus root (sets DOC_RAG_ROOT)")
    parser.add_argument("--queries", default="eval/queries_golden.yaml", help="Queries YAML")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES), help="Comma-separated modes")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Ranking depth")
    parser.add_argument("--namespace", default="default", help="Index namespace")
    args = parser.parse_args(argv)

    # DOC_RAG_ROOT must be set BEFORE the first retrieval import/call:
    # project_root() reads it at config load time and caches the config.
    if args.root:
        os.environ["DOC_RAG_ROOT"] = os.path.abspath(os.path.expanduser(args.root))

    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    invalid = [m for m in modes if m not in {"semantic", "hybrid", "lexical"}]
    if invalid:
        parser.error(f"unknown modes: {','.join(invalid)}")

    from doc_rag.server import retrieval

    queries_path = Path(args.queries)
    if not queries_path.is_absolute():
        queries_path = (_REPO_ROOT / queries_path).resolve()
    queries = load_queries(queries_path)

    cfg = retrieval.load_config()
    chunks = retrieval.load_chunks(cfg)
    if not chunks:
        print(
            f"ERROR: no chunks under {cfg.get('_root')} (build/chunks_jsonl/chunks.jsonl)",
            file=sys.stderr,
        )
        return 1

    results: dict[str, dict[str, Any]] = {}
    for mode in modes:
        print(f"running mode '{mode}' over {len(queries)} queries...", flush=True)
        results[mode] = evaluate_mode(mode, cfg, chunks, queries, args.top_k, args.namespace)

    print_table(results, args.top_k)

    doc_ids = {str(c.get("doc_id", "") or chunk_doc_id(str(c.get("chunk_id", "")))) for c in chunks}
    path = save_report(
        Path(cfg.get("_root", str(_REPO_ROOT))) / "build" / "eval",
        modes,
        args.top_k,
        queries_path,
        results,
        {"n_chunks": len(chunks), "n_documents": len(doc_ids)},
    )
    print(f"\nJSON report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
