from __future__ import annotations

from doc_rag.server.retrieval import (
    annotation_from_markdown,
    document_preview,
    indexed_catalog,
    lexical_search,
)


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


def test_indexed_catalog_keys() -> None:
    c = indexed_catalog()
    assert "documents" in c
    assert "document_count" in c
    assert isinstance(c["documents"], list)
    assert "lexical_search_ready" in c
    assert "semantic_search_ready" in c


def test_annotation_from_markdown_uses_first_heading() -> None:
    md = "# My Title\n\nFirst paragraph here.\n\nMore text."
    title, preview = annotation_from_markdown(md)
    assert title == "My Title"
    assert "First paragraph" in preview


def test_annotation_from_markdown_empty() -> None:
    title, preview = annotation_from_markdown("")
    assert title == ""
    assert "Пустой" in preview


def test_document_preview_rejects_empty_doc_id() -> None:
    out = document_preview("")
    assert out.get("ok") is False
    assert "empty" in str(out.get("error", "")).lower()
