from __future__ import annotations

import pytest

# ── search_tool: structured citations ──────────────────────────────────

def test_build_citations_returns_indexed_list() -> None:
    from doc_rag.server.search_tool import _build_citations

    results = [
        {
            "chunk_id": "a:0",
            "source_file": "doc.pdf",
            "section_path": "3.1",
            "doc_id": "a",
            "score": 0.95,
            "text": "hello world " * 20,
        },
        {
            "chunk_id": "b:0",
            "source_file": "other.pdf",
            "section_path": "",
            "doc_id": "b",
            "score": 0.8,
            "text": "short",
        },
    ]
    citations = _build_citations(results)
    assert len(citations) == 2
    assert citations[0]["index"] == 1
    assert citations[0]["chunk_id"] == "a:0"
    assert citations[0]["source_file"] == "doc.pdf"
    assert citations[0]["section_path"] == "3.1"
    assert citations[0]["score"] == 0.95
    assert citations[1]["index"] == 2
    assert citations[1]["source_file"] == "other.pdf"


def test_build_citations_empty() -> None:
    from doc_rag.server.search_tool import _build_citations

    assert _build_citations([]) == []


def test_doc_search_tool_returns_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    from doc_rag.server import search_tool as st

    monkeypatch.setattr(
        st,
        "doc_search",
        lambda query, top_k, namespace="default", filters=None: [
            {
                "chunk_id": "x:0",
                "source_file": "test.pdf",
                "section_path": "1.0",
                "doc_id": "x",
                "score": 0.9,
                "text": "test content",
            }
        ],
    )
    monkeypatch.setattr(st, "load_config", lambda: {"mcp": {"retrieval_mode": "lexical"}})
    monkeypatch.setattr(st, "indexed_catalog", lambda: {"semantic_search_ready": False})

    content = st.doc_search_tool({"query": "test"})
    assert len(content) >= 2
    last = content[-1]["text"]
    assert "Citations:" in last
    assert "test.pdf" in last


def test_tools_list_doc_search_has_full_schema() -> None:
    pytest.importorskip("fastapi")
    from doc_rag.server.mcp_http import _handle_jsonrpc

    _, out = _handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    tools = out["result"]["tools"]
    search = next(t for t in tools if t["name"] == "doc_search")
    props = search["inputSchema"]["properties"]
    for key in ("query", "top_k", "namespace", "doc_id", "section_path", "tables_only"):
        assert key in props, f"doc_search missing param: {key}"
