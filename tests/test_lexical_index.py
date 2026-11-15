"""Task 14: precomputed BM25 lexical index + RRF weights.

Pins the required properties before the implementation exists (TDD):
(a) BM25 length normalization, (b) RU word forms match, (c) a NEGATIVE
case (no shared stem must not match), (d) code tokens searchable whole,
(e) mtime cache invalidation, (f) filters (Task 7 contract), (g) RRF
weights make the weighted list dominate. Plus review-round additions:
code-token prefix expansion, EN stemming, longest-suffix rule, stem
length guard, filter-semantics matrix against retrieval, cache bypass
for metadata builds.
"""

from __future__ import annotations

import json
import time

import pytest

from doc_rag.server import retrieval as retrieval_module
from doc_rag.server.lexical_index import (
    LexicalIndex,
    _normalize,
    get_lexical_index,
    reset_lexical_cache,
)
from doc_rag.server.retrieval import (
    _chunk_passes_filters,
    _rrf_fusion,
    lexical_search,
)


@pytest.fixture(autouse=True)
def _clean_lexical_cache():
    """Isolate the lexical singleton between tests (mtime-keyed, so two
    tests writing the same tmp_path pattern could otherwise share state)."""
    reset_lexical_cache()
    yield
    reset_lexical_cache()


# ---------------------------------------------------------------------------
# (a) BM25 length normalization: a long doc (same term tf=1) must not beat
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
# (b) RU word forms match the stem; suffix rule details.
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


def test_ru_suffix_one_longest_match_only() -> None:
    """One suffix per token, the LONGEST match wins (order-independent):
    'прибора' cuts only 'а' (longest matching suffix list hit 'а' vs
    nothing longer), 'приборами' cuts 'ами' in one pass -> 'прибор'."""
    assert _normalize("прибора") == "прибор"
    assert _normalize("приборами") == "прибор"
    assert _normalize("приборными") == "приборн"  # 'ыми' longest, one pass
    # Order independence is structural (max by length), spot-check two
    # suffixes where the shorter one is later in the list:
    assert _normalize("данные") == "данн"  # 'ые' (4), not 'е' (1) first


def test_ru_short_word_stem_guard() -> None:
    """The stem guard: token >= 5 letters AND stem after cut >= 4.
    'вода' stays whole (stem 'вод' would be 3); 'шифра'/'шифры' cut to
    'шифр' (stem 4) and unify with the query 'шифр'. 'шифров' matches no
    listed suffix ('ов' is not in the brief's list) and stays whole."""
    assert _normalize("вода") == "вода"
    assert _normalize("шифра") == "шифр"
    assert _normalize("шифры") == "шифр"
    assert _normalize("шифров") == "шифров"


def test_ru_short_word_forms_match_via_index() -> None:
    """'шифра'/'шифры' in a document unify with query 'шифр' (stem 4 —
    the >= 4 stem guard from the review round)."""
    chunks = [
        {"chunk_id": "a:0", "doc_id": "a", "text": "Описание шифра ГОСТ"},
        {"chunk_id": "b:0", "doc_id": "b", "text": "Совершенно другой текст"},
    ]
    out = lexical_search(chunks, query="шифры", top_k=2)
    assert [r["chunk_id"] for r in out] == ["a:0"]


# ---------------------------------------------------------------------------
# (c) NEGATIVE: no shared stem must not match.
# ---------------------------------------------------------------------------


def test_ru_negative_no_stem_match() -> None:
    chunks = [
        {"chunk_id": "a:0", "doc_id": "a", "text": "прибой на берегу"},
        {"chunk_id": "b:0", "doc_id": "b", "text": "прибор учёта"},
    ]
    out = lexical_search(chunks, query="прибой", top_k=2)
    assert [r["chunk_id"] for r in out] == ["a:0"]
    # And the reverse direction: query "прибор" must not hit "прибой".
    idx = LexicalIndex.build(chunks)
    hits = idx.search("прибор", top_k=2)
    assert [h[0] for h in hits] == [1]


# ---------------------------------------------------------------------------
# (d) Code tokens: searchable whole, and PREFIX queries find them.
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


def test_code_prefix_query_finds_longer_codes() -> None:
    """Prefix code queries find chunks holding longer codes: '34.12'
    (substring of '34.12-2018') and '62056-5' (substring of '62056-5-3').
    The old scorer served these via raw substring matching; the index
    serves them via code-term prefix expansion (review round 2)."""
    chunks = [
        {
            "chunk_id": "a:0",
            "doc_id": "a",
            "text": "Национальный стандарт ГОСТ Р 34.12-2015",
        },
        {
            "chunk_id": "b:0",
            "doc_id": "b",
            "text": "Стандарт IEC 62056-5-3 описывает интерфейс",
        },
        {"chunk_id": "c:0", "doc_id": "c", "text": "Совершенно посторонний текст"},
    ]
    idx = LexicalIndex.build(chunks)
    hits = idx.search("34.12", top_k=3)
    assert hits and hits[0][0] == 0  # chunk with '34.12-2015'

    hits = idx.search("62056-5", top_k=3)
    assert hits and hits[0][0] == 1  # chunk with '62056-5-3'

    # Through the retrieval wrapper too (fresh-build path).
    out = lexical_search(chunks, query="62056-5", top_k=3)
    assert out and out[0]["chunk_id"] == "b:0"


def test_code_expansion_prefers_frequent_terms() -> None:
    """Expansion is capped and sorted by df: with two candidate vocabulary
    matches the more frequent one contributes more BM25 mass."""
    chunks = []
    for i in range(6):
        chunks.append({"chunk_id": f"d{i}:0", "doc_id": f"d{i}", "text": f"Код 10-20-{i} вариант"})
    chunks.append({"chunk_id": "rare:0", "doc_id": "rare", "text": "Код 10-20-99 редкий"})
    idx = LexicalIndex.build(chunks)
    hits = idx.search("10-20", top_k=10)
    assert len(hits) >= 6  # all chunks whose codes contain '10-20'


# ---------------------------------------------------------------------------
# (d2) EN stemming: s/es/ies for words >= 4 letters.
# ---------------------------------------------------------------------------


def test_en_stemming_forms() -> None:
    assert _normalize("devices") == "devic"  # 'es'
    assert _normalize("protocols") == "protocol"  # 's'
    assert _normalize("entries") == "entry"  # 'ies' -> 'y'
    # EN word guard is >= 4: 'bus' (3 letters) stays whole, 'buses' cuts.
    assert _normalize("bus") == "bus"
    assert _normalize("buses") == "bus"
    assert _normalize("ies") == "ies"  # 3 letters: below guard, unchanged


def test_en_stemming_matches_via_index() -> None:
    """Query and document sides use the SAME normalizer: query 'devices'
    and doc 'devices' both land on stem 'devic'. A singular query
    'device' stays 'device' (no s/es suffix), so it does NOT match the
    'devic' stem — the light stemmer only unifies plural/verb-s forms,
    both directions must use the same plural form."""
    chunks = [
        {"chunk_id": "a:0", "doc_id": "a", "text": "supported devices list"},
        {"chunk_id": "b:0", "doc_id": "b", "text": "совсем другой текст"},
    ]
    out = lexical_search(chunks, query="devices", top_k=2)
    assert [r["chunk_id"] for r in out] == ["a:0"]
    # Singular query with a plural-only doc does NOT match (asymmetry is
    # documented in the module docstring; both sides share one normalizer).
    out = lexical_search(chunks, query="device", top_k=2)
    assert out == []


# ---------------------------------------------------------------------------
# (e) Singleton cache keyed by chunks.jsonl mtime; metadata builds bypass.
# ---------------------------------------------------------------------------


def test_singleton_cache_invalidates_on_mtime(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    chunks1 = [{"chunk_id": "a:0", "doc_id": "a", "text": "первый текст"}]
    chunks_path.write_text(json.dumps(chunks1[0], ensure_ascii=False) + "\n", encoding="utf-8")
    idx1 = get_lexical_index(chunks1, str(chunks_path))
    # Same file: same object served from cache.
    assert get_lexical_index(chunks1, str(chunks_path)) is idx1

    time.sleep(0.05)  # ensure st_mtime strictly increases
    chunks2 = [{"chunk_id": "b:0", "doc_id": "b", "text": "второй текст"}]
    chunks_path.write_text(json.dumps(chunks2[0], ensure_ascii=False) + "\n", encoding="utf-8")
    idx2 = get_lexical_index(chunks2, str(chunks_path))
    assert idx2 is not idx1
    hits = idx2.search("второй", top_k=2)
    assert hits and hits[0][0] == 0


def test_metadata_build_bypasses_cache(tmp_path) -> None:
    """A metadata-tied build must not be cached and must not poison the
    singleton for subsequent metadata-free callers (review round 2)."""
    chunks_path = tmp_path / "chunks.jsonl"
    chunks = [
        {"chunk_id": "a:0", "doc_id": "a", "text": "текст один"},
        {"chunk_id": "b:0", "doc_id": "b", "text": "текст два"},
    ]
    chunks_path.write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8"
    )
    meta = {"a:0": {"doc_id": "a", "section_path": "", "is_table": False}}
    idx_meta = get_lexical_index(chunks, str(chunks_path), metadata=meta)
    # Metadata build is fresh, never cached:
    assert get_lexical_index(chunks, str(chunks_path), metadata=meta) is not idx_meta
    # And the metadata-free caller got the singleton, not the meta build:
    idx_plain = get_lexical_index(chunks, str(chunks_path))
    assert get_lexical_index(chunks, str(chunks_path)) is idx_plain
    assert idx_plain is not idx_meta


# ---------------------------------------------------------------------------
# (f) Filters (Task 7 contract) work through the index path — and match
# the retrieval dict-path semantics on a matrix of cases (review round 2).
# ---------------------------------------------------------------------------


_FILTER_MATRIX = [
    # (filters, chunk_overrides, expected_pass)
    ({"doc_id": "d1"}, {}, False),  # chunk d2
    ({"doc_id": "d2"}, {}, True),  # chunk d2
    ({"doc_ids": ["d1", "d3"]}, {}, False),
    ({"doc_ids": ["d2", "d3"]}, {}, True),
    ({"doc_ids": ("d2",)}, {}, True),
    ({"doc_ids": {"d2"}}, {}, True),
    ({"section_path": "d2/sec"}, {}, True),  # prefix match
    ({"section_path": "d2/secX"}, {}, False),
    ({"section_path": ""}, {}, True),  # empty prefix matches everything
    ({"is_table": True}, {"is_table": True}, True),
    ({"is_table": True}, {}, False),
    ({"is_table": False}, {}, True),
    ({"doc_id": "d2", "is_table": False}, {}, True),  # AND of keys
    ({"doc_id": "d1", "is_table": False}, {}, False),
]


@pytest.mark.parametrize("filters,overrides,expected", _FILTER_MATRIX)
def test_filter_semantics_match_retrieval(filters, overrides, expected) -> None:
    """LexicalIndex._meta_passes must agree with
    retrieval._chunk_passes_filters on the whole matrix (the three
    filter implementations are intentionally parallel copies)."""
    chunk = {
        "chunk_id": "d2:0",
        "doc_id": "d2",
        "section_path": "d2/sec0",
        "text": "needle",
    }
    chunk.update(overrides)
    assert _chunk_passes_filters(chunk, filters) is expected

    idx = LexicalIndex.build([chunk])
    hits = idx.search("needle", top_k=5, filters=filters)
    assert bool(hits) is expected


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


def test_hybrid_search_passes_weights(monkeypatch) -> None:
    """hybrid_search must wire lexical_weight/semantic_weight into RRF:
    a heavy lexical weight must lift a lexical-only hit above a
    semantic-only hit in the fused ranking."""
    chunks = [
        {"chunk_id": "lex-hit:0", "doc_id": "lex-hit", "text": "квантовый датчик давления"},
        {"chunk_id": "sem-hit:0", "doc_id": "sem-hit", "text": "что-то unrelated"},
    ]

    def fake_semantic(cfg, ch, query, top_k, namespace="default", filters=None):
        return [
            {"chunk_id": "sem-hit:0", "doc_id": "sem-hit", "text": "что-то unrelated", "score": 1.0}
        ]

    monkeypatch.setattr(retrieval_module, "semantic_search", fake_semantic)
    out = retrieval_module.hybrid_search(
        {},
        chunks,
        "квантовый датчик",
        top_k=2,
        lexical_weight=3.0,
        semantic_weight=1.0,
    )
    assert [r["chunk_id"] for r in out] == ["lex-hit:0", "sem-hit:0"]


def test_lexical_search_finds_match_and_scores() -> None:
    chunks = [
        {"doc_id": "a", "chunk_id": "a:0", "text": "hello world"},
        {"doc_id": "b", "chunk_id": "b:0", "text": "something else"},
        {"doc_id": "c", "chunk_id": "c:0", "text": "world world world"},
    ]
    out = lexical_search(chunks, query="world", top_k=2)
    assert len(out) == 2
    assert isinstance(out[0].get("score"), float)
