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


def test_load_config_rejects_invalid_index_backend(tmp_path, monkeypatch) -> None:
    """load_config raises ValueError for invalid index.backend."""
    import yaml

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    cfg_file = cfg_dir / "config.yaml"
    cfg_file.write_text(yaml.safe_dump({"index": {"backend": "mongodb"}}))

    monkeypatch.setattr("doc_rag.server.retrieval.project_root", lambda: str(tmp_path))
    # Clear cache
    monkeypatch.setattr("doc_rag.server.retrieval._CFG_CACHE", {})

    import pytest as _pytest

    with _pytest.raises(ValueError, match="Invalid index.backend"):
        from doc_rag.server.retrieval import load_config as _lc

        _lc()


def test_load_config_rejects_invalid_retrieval_mode(tmp_path, monkeypatch) -> None:
    """load_config raises ValueError for invalid mcp.retrieval_mode."""
    import yaml

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    cfg_file = cfg_dir / "config.yaml"
    cfg_file.write_text(yaml.safe_dump({"mcp": {"retrieval_mode": "vector_only"}}))

    monkeypatch.setattr("doc_rag.server.retrieval.project_root", lambda: str(tmp_path))
    monkeypatch.setattr("doc_rag.server.retrieval._CFG_CACHE", {})

    import pytest as _pytest

    with _pytest.raises(ValueError, match="Invalid mcp.retrieval_mode"):
        from doc_rag.server.retrieval import load_config as _lc

        _lc()
