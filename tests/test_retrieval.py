from __future__ import annotations

from doc_rag.server.retrieval import lexical_search


def test_lexical_search_finds_match_and_scores() -> None:
    chunks = [
        {"doc_id": "a", "chunk_id": "a:0", "text": "hello world"},
        {"doc_id": "b", "chunk_id": "b:0", "text": "something else"},
        {"doc_id": "c", "chunk_id": "c:0", "text": "world world world"},
    ]
    out = lexical_search(chunks, query="world", top_k=2)
    assert len(out) == 2
    assert out[0]["doc_id"] in {"a", "c"}
    assert isinstance(out[0].get("score"), float)

