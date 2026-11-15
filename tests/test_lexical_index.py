"""Task 14: precomputed BM25 lexical index + RRF weights.

Pins the required properties before the implementation exists (TDD):
(a) BM25 length normalization, (b) RU word forms match, (c) a NEGATIVE
case (no shared stem must not match), (d) code tokens searchable whole,
(e) mtime cache invalidation, (f) filters (Task 7 contract), (g) RRF
weights make the weighted list dominate.
"""

from __future__ import annotations

import json
import time

import pytest

from doc_rag.server import retrieval
from doc_rag.server.lexical_index import LexicalIndex, get_lexical_index
from doc_rag.server.retrieval import _rrf_fusion, lexical_search

# ---------------------------------------------------------------------------
# (a) BM25 length normalization: a long doc (x5 repeated text) must not beat
# a short relevant one merely by being long.
# ---------------------------------------------------------------------------


def test_bm25_long_doc_does_not_win_by_length() -> None:
    """Short relevant doc beats a much longer one with the same term tf=1:
    BM25's length normalization must dominate when the long doc carries
    no extra term frequency."""
    short = {"chunk_id": "a:0", "doc_id": "a", "text": "прибор учёта электроэнергии"}
    filler = " ".join(["посторонний", "текстовый", "наполнитель"] * 40)
    long_doc = {"chunk_id": "b:0", "doc_id": "b", "text": f"{filler} прибор учёта"}
    chunks = [short, long_doc]
    out = lexical_search(chunks, query="прибор", top_k=2)
    assert out, "no results at all"
    assert out[0]["chunk_id"] == "a:0"


def test_bm25_length_normalization_via_index() -> None:
    """Same tf, wildly different lengths: the shorter doc must outscore
    the longer one (BM25 saturates tf, punishes relative length)."""
    idx = LexicalIndex.build(
        [
            {"chunk_id": "a:0", "doc_id": "a", "text": "прибор"},
            {
                "chunk_id": "b:0",
                "doc_id": "b",
                "text": "прибор " + "наполнитель " * 40,
            },
        ]
    )
    out = idx.search("прибор", top_k=2)
    assert out[0][0] == 0  # short doc first


# ---------------------------------------------------------------------------
# (b) RU word forms «прибора»/«приборы» must match a doc with «прибор».
# ---------------------------------------------------------------------------


def test_ru_word_forms_match() -> None:
    chunks = [
        {"chunk_id": "a:0", "doc_id": "a", "text": "Описание прибора учёта"},
        {"chunk_id": "b:0", "doc_id": "b", "text": "Совершенно другой текст"},
    ]
    out = lexical_search(chunks, query="приборы", top_k=2)
    assert [r["chunk_id"] for r in out] == ["a:0"]

    idx = LexicalIndex.build(chunks)
    hits = idx.search("прибора", top_k=2)
    assert hits and hits[0][0] == 0


# ---------------------------------------------------------------------------
# (c) NEGATIVE: «прибой» must NOT match «прибор» after normalization.
# ---------------------------------------------------------------------------


def test_ru_negative_no_stem_match() -> None:
    chunks = [
        {"chunk_id": "a:0", "doc_id": "a", "text": "прибой на берегу"},
        {"chunk_id": "b:0", "doc_id": "b", "text": "прибор учёта"},
    ]
    out = lexical_search(chunks, query="прибой", top_k=2)
    assert [r["chunk_id"] for r in out] == ["a:0"]
    # And the reverse direction: query «прибор» must not hit «прибой».
    idx = LexicalIndex.build(chunks)
    hits = idx.search("прибор", top_k=2)
    assert [h[0] for h in hits] == [1]


# ---------------------------------------------------------------------------
# (d) Code token «62056-5-3» searchable as a whole.
# ---------------------------------------------------------------------------


def test_code_token_searchable_whole() -> None:
    chunks = [
        {
            "chunk_id": "a:0",
            "doc_id": "a",
            "text": "Стандарт IEC 62056-5-3 описывает интерфейс",
        },
        {"chunk_id": "b:0", "doc_id": "b", "text": "другой документ без кодов"},
    ]
    out = lexical_search(chunks, query="62056-5-3", top_k=2)
    assert out and out[0]["chunk_id"] == "a:0"

    idx = LexicalIndex.build(chunks)
    hits = idx.search("62056-5-3", top_k=2)
    assert hits and hits[0][0] == 0


def test_code_token_prefers_doc_with_code() -> None:
    """A code query must rank the doc containing the code above one that
    merely mentions neighbouring numbers."""
    chunks = [
        {"chunk_id": "a:0", "doc_id": "a", "text": "см. также ГОСТ 32144-2013"},
        {"chunk_id": "b:0", "doc_id": "b", "text": "IEC 62056-5-3 ED4"},
    ]
    out = lexical_search(chunks, query="62056-5-3", top_k=2)
    assert out[0]["chunk_id"] == "b:0"


# ---------------------------------------------------------------------------
# (e) Singleton cache keyed by chunks.jsonl mtime.
# ---------------------------------------------------------------------------


def test_singleton_cache_invalidates_on_mtime(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        json.dumps({"chunk_id": "a:0", "doc_id": "a", "text": "первый текст"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    idx1 = get_lexical_index(
        [{"chunk_id": "a:0", "doc_id": "a", "text": "первый текст"}], str(chunks_path)
    )
    # Same file: same object served from cache.
    assert (
        get_lexical_index(
            [{"chunk_id": "a:0", "doc_id": "a", "text": "первый текст"}], str(chunks_path)
        )
        is idx1
    )

    time.sleep(0.05)  # ensure st_mtime strictly increases
    chunks_path.write_text(
        json.dumps({"chunk_id": "b:0", "doc_id": "b", "text": "второй текст"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    idx2 = get_lexical_index(
        [{"chunk_id": "b:0", "doc_id": "b", "text": "второй текст"}], str(chunks_path)
    )
    assert idx2 is not idx1
    hits = idx2.search("второй", top_k=2)
    assert hits and hits[0][0] == 0


# ---------------------------------------------------------------------------
# (f) Filters (Task 7 contract) work through the index path.
# ---------------------------------------------------------------------------


def test_filters_via_index_path() -> None:
    chunks = [
        {"chunk_id": "doc-a:0", "doc_id": "doc-a", "section_path": "doc-a/sec0", "text": "needle"},
        {"chunk_id": "doc-a:1", "doc_id": "doc-a", "section_path": "doc-a/sec1", "text": "needle"},
        {"chunk_id": "doc-b:0", "doc_id": "doc-b", "section_path": "doc-b/sec0", "text": "needle"},
    ]
    out = lexical_search(chunks, query="needle", top_k=5, filters={"doc_id": "doc-b"})
    assert {r["chunk_id"] for r in out} == {"doc-b:0"}

    out = lexical_search(chunks, query="needle", top_k=5, filters={"doc_id": "doc-zzz"})
    assert out == []

    out = lexical_search(chunks, query="needle", top_k=5, filters={"section_path": "doc-a/sec1"})
    assert {r["chunk_id"] for r in out} == {"doc-a:1"}

    chunks[2]["is_table"] = True
    out = lexical_search(chunks, query="needle", top_k=5, filters={"is_table": True})
    assert {r["chunk_id"] for r in out} == {"doc-b:0"}

    with pytest.raises(ValueError, match="unsupported filter keys"):
        lexical_search(chunks, query="needle", top_k=5, filters={"table_only": True})


# ---------------------------------------------------------------------------
# (g) RRF weights: weights=(2.0, 1.0) ranks the first (semantic) list higher.
# ---------------------------------------------------------------------------


def _rrf_item(cid: str) -> dict:
    return {"chunk_id": cid}


def test_rrf_weights_semantic_first() -> None:
    # Both lists contain both chunks but in opposite orders; with equal
    # weights the fused scores would tie. weights=(2.0, 1.0) must put the
    # semantic list's winner on top.
    sem = [_rrf_item("s1"), _rrf_item("s2")]
    lex = [_rrf_item("s2"), _rrf_item("s1")]
    out = _rrf_fusion(sem, lex, k=60, weights=[2.0, 1.0])
    assert [r["chunk_id"] for r in out] == ["s1", "s2"]


def test_rrf_weights_none_matches_old_behavior() -> None:
    """weights=None must reproduce the plain RRF: each list contributes
    1/(k + rank + 1), ranks are 0-based."""
    sem = [_rrf_item("s1"), _rrf_item("s2"), _rrf_item("s3")]
    lex = [_rrf_item("s3"), _rrf_item("s2")]
    out = _rrf_fusion(sem, lex, k=60)
    fused = {r["chunk_id"]: r["score"] for r in out}
    # s2: sem rank 1 -> 1/62, lex rank 1 -> 1/62 (0-based ranks);
    # _rrf_fusion rounds the fused score to 6 decimals (old behavior).
    assert fused["s2"] == pytest.approx(2 / 62, abs=1e-6)
    assert fused["s1"] == pytest.approx(1 / 61, abs=1e-6)


def test_hybrid_search_passes_weights() -> None:
    """hybrid_search must wire lexical_weight/semantic_weight into RRF:
    a heavy lexical weight must lift a lexical-only hit above a
    semantic-only hit in the fused ranking."""
    chunks = [
        {"chunk_id": "lex-hit:0", "doc_id": "lex-hit", "text": "квантовый датчик давления"},
        {"chunk_id": "sem-hit:0", "doc_id": "sem-hit", "text": "что-то unrelated"},
    ]
    lex_rank = [
        {
            "chunk_id": "lex-hit:0",
            "doc_id": "lex-hit",
            "text": "квантовый датчик давления",
            "score": 1.0,
        }
    ]

    def fake_semantic(cfg, ch, query, top_k, namespace="default", filters=None):
        return [
            {"chunk_id": "sem-hit:0", "doc_id": "sem-hit", "text": "что-то unrelated", "score": 1.0}
        ]

    orig_sem = retrieval.semantic_search
    retrieval.semantic_search = fake_semantic  # type: ignore[assignment]
    try:
        out = retrieval.hybrid_search(
            {},
            chunks,
            "квантовый датчик",
            top_k=2,
            lexical_weight=3.0,
            semantic_weight=1.0,
        )
    finally:
        retrieval.semantic_search = orig_sem  # type: ignore[assignment]
        assert lex_rank  # silence lint on the fixture var
    assert [r["chunk_id"] for r in out] == ["lex-hit:0", "sem-hit:0"]
