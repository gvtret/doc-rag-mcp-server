"""Tests for scripts/eval_retrieval.py (retrieval-quality eval harness).

Two layers, mirroring how the harness works:

1. Metric unit tests with hand-computed expected values (recall, MRR,
   nDCG with binary gains).
2. An end-to-end run over a synthetic 3-document corpus in tmp_path
   (DOC_RAG_ROOT): the harness must score recall@10 == 1.0 for queries
   whose targets are trivially findable, and the JSON report must land
   in <root>/build/eval/.

The script is loaded by file location via importlib: scripts/ is not a
package (same pattern as tests/test_fetch_model.py). Retrieval runs use
lexical mode only -- no embedding model, no FAISS -- by stubbing
semantic_search, which keeps the metric math identical to the real run.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_eval_script():
    spec = importlib.util.spec_from_file_location(
        "eval_retrieval", ROOT / "scripts" / "eval_retrieval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Metric unit tests (hand-computed)
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_doc_recall_full_hit(self):
        mod = load_eval_script()
        ranked = ["d1", "d2", "d3"]
        assert mod.doc_recall(ranked, {"d1", "d2"}, 10) == 1.0

    def test_doc_recall_partial_and_miss(self):
        mod = load_eval_script()
        # 1 of 3 relevant docs in top-k
        assert mod.doc_recall(["d1", "x", "y"], {"d1", "d2", "d3"}, 10) == pytest.approx(1 / 3)
        assert mod.doc_recall(["x", "y", "z"], {"d1"}, 10) == 0.0

    def test_doc_recall_respects_k(self):
        mod = load_eval_script()
        # Relevant doc at position 4 is outside top-3
        assert mod.doc_recall(["x", "y", "z", "d1"], {"d1"}, 3) == 0.0
        assert mod.doc_recall(["x", "y", "z", "d1"], {"d1"}, 4) == 1.0

    def test_mrr_first_position(self):
        mod = load_eval_script()
        assert mod.mrr(["d1", "x"], {"d1"}, 10) == 1.0

    def test_mrr_second_position(self):
        mod = load_eval_script()
        assert mod.mrr(["x", "d1"], {"d1"}, 10) == pytest.approx(0.5)

    def test_mrr_miss_beyond_k(self):
        mod = load_eval_script()
        assert mod.mrr(["x", "y", "d1"], {"d1"}, 2) == 0.0

    def test_ndcg_perfect(self):
        """All relevant docs at the top -> nDCG == 1.0."""
        mod = load_eval_script()
        assert mod.ndcg(["d1", "d2", "x", "y"], {"d1", "d2"}, 10) == pytest.approx(1.0)

    def test_ndcg_hand_computed_single_relevant_at_position_2(self):
        """One relevant doc at rank 2:
        DCG = 1/log2(3) = 0.6309; IDCG = 1/log2(2) = 1.0.
        nDCG = 0.6309."""
        mod = load_eval_script()
        expected = 1.0 / math.log2(3)
        assert mod.ndcg(["x", "d1", "y", "z"], {"d1"}, 10) == pytest.approx(expected)

    def test_ndcg_hand_computed_two_relevant_positions_2_and_4(self):
        """Relevant at ranks 2 and 4 out of 2 relevant docs:
        DCG = 1/log2(3) + 1/log2(5) = 0.63093 + 0.43068 = 1.06161
        IDCG = 1/log2(2) + 1/log2(3) = 1.63093
        nDCG = 0.65082."""
        mod = load_eval_script()
        dcg = 1.0 / math.log2(3) + 1.0 / math.log2(5)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        assert mod.ndcg(["x", "d1", "y", "d2", "z"], {"d1", "d2"}, 10) == pytest.approx(dcg / idcg)

    def test_ndcg_more_relevant_than_k_caps_ideal(self):
        """3 relevant docs, k=2, both retrieved at top: IDCG uses
        min(len(rel), k) = 2 -> nDCG = 1.0."""
        mod = load_eval_script()
        assert mod.ndcg(["d1", "d2"], {"d1", "d2", "d3"}, 2) == pytest.approx(1.0)

    def test_ndcg_empty_relevant(self):
        mod = load_eval_script()
        assert mod.ndcg(["d1"], set(), 10) == 0.0

    def test_chunk_doc_id_prefix(self):
        mod = load_eval_script()
        assert mod.chunk_doc_id("rfc5424_24d91f90a0:012") == "rfc5424_24d91f90a0"


# ---------------------------------------------------------------------------
# Aggregate helper
# ---------------------------------------------------------------------------


def test_aggregate_scores_macro_average():
    mod = load_eval_script()
    ranked = [["d1", "x"], ["x", "d1"]]
    relevant = [{"d1"}, {"d1"}]
    agg = mod.aggregate_scores(ranked, relevant, 10)
    assert agg["n_evaluated"] == 2
    assert agg["recall@10"] == pytest.approx(1.0)
    assert agg["mrr@10"] == pytest.approx((1.0 + 0.5) / 2)
    # nDCG: 1.0 and 1/log2(3) / 1.0
    assert agg["ndcg@10"] == pytest.approx((1.0 + 1.0 / math.log2(3)) / 2)


def test_aggregate_scores_skips_empty_relevance():
    mod = load_eval_script()
    ranked = [["d1"], ["x"]]
    relevant = [{"d1"}, set()]  # second query has no chunk-level labels
    agg = mod.aggregate_scores(ranked, relevant, 10)
    assert agg["n_evaluated"] == 1


# ---------------------------------------------------------------------------
# Golden-file loading
# ---------------------------------------------------------------------------


def test_load_queries_valid_file(tmp_path):
    mod = load_eval_script()
    p = tmp_path / "q.yaml"
    p.write_text(
        "- query: hello world\n"
        "  relevant_doc_ids: [doc-a]\n"
        "- query: second\n"
        "  relevant_doc_ids: [doc-b]\n"
        "  relevant_chunk_ids: [doc-b:0]\n",
        encoding="utf-8",
    )
    qs = mod.load_queries(p)
    assert len(qs) == 2
    assert qs[0]["relevant_doc_ids"] == ["doc-a"]
    assert qs[1]["relevant_chunk_ids"] == ["doc-b:0"]


def test_load_queries_rejects_entry_without_relevance(tmp_path):
    mod = load_eval_script()
    p = tmp_path / "q.yaml"
    p.write_text("- query: nothing relevant here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="neither"):
        mod.load_queries(p)


def test_load_queries_rejects_non_list(tmp_path):
    mod = load_eval_script()
    p = tmp_path / "q.yaml"
    p.write_text("queries: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="list"):
        mod.load_queries(p)


# ---------------------------------------------------------------------------
# End-to-end on a synthetic corpus (DOC_RAG_ROOT in tmp_path)
# ---------------------------------------------------------------------------


def _write_corpus(root: Path) -> list[dict[str, Any]]:
    """3 synthetic docs, each with a distinctive lexical marker."""
    docs = [
        ("doc-alpha", "alpha transformer substation dispatch protocol", 4),
        ("doc-beta", "beta optical fibre communication link budget", 4),
        ("doc-gamma", "gamma grounding lightning protection earth", 4),
    ]
    chunks: list[dict[str, Any]] = []
    for doc_id, text, n in docs:
        for i in range(n):
            chunks.append(
                {
                    "chunk_id": f"{doc_id}:{i:03d}",
                    "doc_id": doc_id,
                    "text": f"{text} part {i} marker",
                    "source_file": f"{doc_id}.pdf",
                }
            )
    chunks_dir = root / "build" / "chunks_jsonl"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    with (chunks_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return chunks


def _write_queries(root: Path, *, with_chunk_labels: bool = False) -> Path:
    lines = [
        "- query: transformer substation dispatch\n  relevant_doc_ids: [doc-alpha]\n",
        "- query: optical fibre link budget\n  relevant_doc_ids: [doc-beta]\n",
        "- query: grounding lightning protection\n  relevant_doc_ids: [doc-gamma]\n",
    ]
    if with_chunk_labels:
        lines[0] = (
            "- query: transformer substation dispatch\n"
            "  relevant_doc_ids: [doc-alpha]\n"
            "  relevant_chunk_ids: [doc-alpha:000, doc-alpha:001]\n"
        )
    q = root / "queries.yaml"
    q.write_text("".join(lines), encoding="utf-8")
    return q


def _setup_corpus(root: Path) -> Path:
    """Config + corpus + queries; returns the queries path."""
    (root / "config").mkdir()
    (root / "config" / "config.yaml").write_text(
        "paths:\n  chunks_dir: build/chunks_jsonl\n"
        "  index_dir: build/index\n"
        "  manifest_path: build/manifest.json\n"
        "embeddings:\n  model_name: stub\n"
        "index:\n  backend: faiss\n"
        "mcp:\n  retrieval_mode: semantic\n",
        encoding="utf-8",
    )
    _write_corpus(root)
    return _write_queries(root)


def _stub_semantic_to_lexical(monkeypatch) -> None:
    """No embedding model in tests: stub semantic_search to return the
    lexical ranking so semantic/hybrid metrics match lexical."""
    import doc_rag.server.retrieval as retrieval

    def fake_semantic(cfg, chunks, query, top_k, namespace="default", filters=None):
        return retrieval.lexical_search(chunks, query, top_k, filters=filters)

    monkeypatch.setattr(retrieval, "semantic_search", fake_semantic)


def test_eval_end_to_end_recall_one(tmp_path, monkeypatch, capsys):
    """Full pipeline: 3-doc synthetic corpus, 3 queries with exact
    lexical matches -> recall@10 == 1.0 at doc level for all modes,
    JSON report written under <root>/build/eval/. No query carries
    chunk labels -> no chunk_level key in the report and the table
    prints the pending-annotation placeholder."""
    mod = load_eval_script()

    root = tmp_path
    queries_path = _setup_corpus(root)
    monkeypatch.setenv("DOC_RAG_ROOT", str(root))
    _stub_semantic_to_lexical(monkeypatch)

    rc = mod.main(
        [
            "--queries",
            str(queries_path),
            "--modes",
            "semantic,hybrid,lexical",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "recall@k" in out  # table header label
    assert "chunk-level: pending operator annotation" in out

    reports = sorted((root / "build" / "eval").glob("*.json"))
    assert reports, "JSON report missing under build/eval/"
    report = json.loads(reports[-1].read_text(encoding="utf-8"))
    assert report["modes"] == ["semantic", "hybrid", "lexical"]
    assert report["corpus"]["n_chunks"] == 12
    for mode in ("semantic", "hybrid", "lexical"):
        mode_res = report["results"][mode]
        assert mode_res["degraded"] is False
        agg = mode_res["doc_level"]
        assert agg["n_evaluated"] == 3
        assert agg["recall@10"] == pytest.approx(1.0)
        assert agg["mrr@10"] == pytest.approx(1.0)
        assert agg["ndcg@10"] == pytest.approx(1.0)
        # No chunk labels in the evaluated set -> chunk metrics omitted.
        assert "chunk_level" not in mode_res


def test_eval_chunk_labels_present(tmp_path, monkeypatch):
    """When at least one query has relevant_chunk_ids, chunk_level is
    computed (kept in code; gated on label availability)."""
    mod = load_eval_script()
    root = tmp_path
    _setup_corpus(root)
    # Rewrite queries with chunk labels on the first query.
    q = root / "queries.yaml"
    q.write_text(
        "- query: transformer substation dispatch\n"
        "  relevant_doc_ids: [doc-alpha]\n"
        "  relevant_chunk_ids: [doc-alpha:000, doc-alpha:001]\n"
        "- query: optical fibre link budget\n"
        "  relevant_doc_ids: [doc-beta]\n"
        "- query: grounding lightning protection\n"
        "  relevant_doc_ids: [doc-gamma]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DOC_RAG_ROOT", str(root))

    from doc_rag.server import retrieval as retrieval_mod

    cfg = {"_root": str(root), "paths": {"chunks_dir": "build/chunks_jsonl"}}
    chunks = retrieval_mod.load_chunks(cfg)
    res = mod.evaluate_mode("lexical", cfg, chunks, mod.load_queries(q), top_k=10)
    # The lexical query ranks all four doc-alpha chunks above everything
    # else, so both labelled chunks land in top-10 -> recall 1.0.
    assert res["chunk_level"]["n_evaluated"] == 1
    assert res["chunk_level"]["recall@10"] == pytest.approx(1.0)


def test_eval_per_query_records_rankings(tmp_path, monkeypatch):
    mod = load_eval_script()
    root = tmp_path
    queries_path = _setup_corpus(root)
    monkeypatch.setenv("DOC_RAG_ROOT", str(root))

    from doc_rag.server import retrieval as retrieval_mod

    cfg = {"_root": str(root), "paths": {"chunks_dir": "build/chunks_jsonl"}}
    chunks = retrieval_mod.load_chunks(cfg)
    res = mod.evaluate_mode(
        "lexical",
        cfg,
        chunks,
        mod.load_queries(queries_path),
        top_k=10,
    )
    pq = res["per_query"]
    assert [r["query"] for r in pq] == [
        "transformer substation dispatch",
        "optical fibre link budget",
        "grounding lightning protection",
    ]
    assert pq[0]["returned_doc_ids"][0] == "doc-alpha"
    assert "doc-alpha" in pq[0]["returned_doc_ids"]


def test_main_rejects_unknown_mode(tmp_path, monkeypatch, capsys):
    mod = load_eval_script()
    queries_path = _setup_corpus(tmp_path)
    monkeypatch.setenv("DOC_RAG_ROOT", str(tmp_path))
    with pytest.raises(SystemExit):
        mod.main(["--modes", "bogus", "--queries", str(queries_path)])


def test_main_rejects_top_k_below_metric_k(tmp_path, monkeypatch, capsys):
    """--top-k < 10 would silently relabel a shallower ranking as @10;
    metrics are fixed @10, so smaller values must fail loudly."""
    mod = load_eval_script()
    queries_path = _setup_corpus(tmp_path)
    monkeypatch.setenv("DOC_RAG_ROOT", str(tmp_path))
    with pytest.raises(SystemExit):
        mod.main(["--top-k", "2", "--queries", str(queries_path)])
    assert "--top-k must be >= 10" in capsys.readouterr().err


def test_main_accepts_top_k_equal_10(tmp_path, monkeypatch):
    """Boundary: --top-k 10 is the minimum accepted value."""
    mod = load_eval_script()
    queries_path = _setup_corpus(tmp_path)
    monkeypatch.setenv("DOC_RAG_ROOT", str(tmp_path))
    _stub_semantic_to_lexical(monkeypatch)
    assert mod.main(["--top-k", "10", "--queries", str(queries_path), "--modes", "lexical"]) == 0


def test_root_flag_sets_env_before_retrieval_import():
    """--root must set DOC_RAG_ROOT BEFORE retrieval is imported/called
    (project_root() reads and caches it at config load time). Verified
    in a fresh subprocess: a clean interpreter runs main(["--root", X])
    with a stub-free lexical mode, then asserts the env var the script
    set matches X and that the report was written under X/build/eval."""
    import os
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _setup_corpus(root)
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
            "import importlib.util\n"
            f"spec = importlib.util.spec_from_file_location('eval_retrieval', {str(ROOT / 'scripts' / 'eval_retrieval.py')!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            f"rc = mod.main(['--root', {str(root)!r}, '--modes', 'lexical', '--queries', {str(root / 'queries.yaml')!r}])\n"
            "import os\n"
            "assert rc == 0\n"
            f"assert os.environ['DOC_RAG_ROOT'] == {str(root)!r}, os.environ.get('DOC_RAG_ROOT')\n"
            "import glob\n"
            "assert glob.glob(os.environ['DOC_RAG_ROOT'] + '/build/eval/*.json'), 'no report under --root'\n"
            "print('ok')\n"
        )
        # No DOC_RAG_ROOT in the subprocess environment: only --root may
        # provide it, proving the flag runs before any retrieval call.
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=120,
            env={"PATH": os.environ.get("PATH", "")},
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip().endswith("ok")


def test_eval_degraded_mode_marked(tmp_path, monkeypatch, capsys):
    """semantic_search returning None (no model/index) must mark the
    mode degraded: true in JSON and print the DEGRADED suffix -- zeros
    of an unavailable mode must not read as bad retrieval."""
    mod = load_eval_script()
    root = tmp_path
    queries_path = _setup_corpus(root)
    monkeypatch.setenv("DOC_RAG_ROOT", str(root))

    import doc_rag.server.retrieval as retrieval

    monkeypatch.setattr(retrieval, "semantic_search", lambda *a, **kw: None)

    rc = mod.main(["--queries", str(queries_path), "--modes", "semantic,lexical"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DEGRADED" in out

    reports = sorted((root / "build" / "eval").glob("*.json"))
    report = json.loads(reports[-1].read_text(encoding="utf-8"))
    assert report["results"]["semantic"]["degraded"] is True
    # The available mode is not degraded.
    assert report["results"]["lexical"]["degraded"] is False
    # Degraded semantic doc_level still present (zeros) for the record.
    assert report["results"]["semantic"]["doc_level"]["n_evaluated"] == 3


def test_load_queries_derives_doc_ids_from_chunk_ids(tmp_path):
    """Entry with only relevant_chunk_ids: doc-level relevance derived
    from the chunk_id prefix, not silently skipped."""
    mod = load_eval_script()
    p = tmp_path / "q.yaml"
    p.write_text(
        "- query: chunks only\n  relevant_chunk_ids: [doc-x:3, doc-y:0, doc-x:7]\n",
        encoding="utf-8",
    )
    qs = mod.load_queries(p)
    assert qs[0]["relevant_doc_ids"] == ["doc-x", "doc-y"]
    assert qs[0]["relevant_chunk_ids"] == ["doc-x:3", "doc-y:0", "doc-x:7"]
