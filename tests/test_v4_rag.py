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


# ── rag_generate: context assembly with max_context_tokens ─────────────

def test_format_context_truncates_long_chunks() -> None:
    from doc_rag.raglib.rag_generate import _format_context

    long_text = "word " * 5000
    results = [
        {"chunk_id": "c:0", "source_file": "big.pdf", "section_path": "", "text": long_text, "score": 0.9},
    ]
    ctx, sources = _format_context(results, max_context_tokens=100)
    assert len(sources) == 1
    assert len(ctx) < len(long_text) + 200
    assert ctx.endswith("...")


def test_format_context_respects_budget_multiple_chunks() -> None:
    from doc_rag.raglib.rag_generate import _format_context

    results = [
        {"chunk_id": f"c:{i}", "source_file": "f.pdf", "section_path": "", "text": f"chunk {i} " * 100, "score": 0.8}
        for i in range(10)
    ]
    ctx, sources = _format_context(results, max_context_tokens=200)
    assert len(sources) < 10
    assert len(sources) >= 1


def test_format_context_no_truncation_when_within_budget() -> None:
    from doc_rag.raglib.rag_generate import _format_context

    results = [
        {"chunk_id": "s:0", "source_file": "s.pdf", "section_path": "", "text": "short", "score": 0.9},
    ]
    ctx, sources = _format_context(results, max_context_tokens=10000)
    assert len(sources) == 1
    assert "short" in ctx
    assert not ctx.endswith("...")


# ── generate_tool: MCP tool call ───────────────────────────────────────

def test_doc_generate_tool_empty_query() -> None:
    from doc_rag.server.generate_tool import doc_generate_tool

    result = doc_generate_tool({"query": ""})
    assert len(result) == 1
    assert "Empty query" in result[0]["text"]


def test_doc_generate_tool_llm_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    from doc_rag.server import generate_tool as gt
    from doc_rag.raglib import rag_generate as rg

    monkeypatch.setattr(rg, "_get_rag_config", lambda cfg: None)

    from doc_rag.server import retrieval as ret

    monkeypatch.setattr(
        ret,
        "doc_search",
        lambda q, k, namespace="default", filters=None: [],
    )

    from doc_rag.server.generate_tool import doc_generate_tool

    result = doc_generate_tool({"query": "test"})
    texts = " ".join(r["text"] for r in result)
    assert "Error" in texts or "not configured" in texts.lower() or "LLM" in texts


def test_doc_generate_tool_format_citations() -> None:
    from doc_rag.server.generate_tool import _format_citations

    sources = [
        {"index": 1, "source_file": "a.pdf", "section_path": "2.1", "chunk_id": "a:0", "score": 0.95},
        {"index": 2, "source_file": "b.pdf", "section_path": "", "chunk_id": "b:0", "score": 0.8},
    ]
    out = _format_citations(sources)
    assert "[1]" in out
    assert "a.pdf" in out
    assert "§2.1" in out
    assert "[2]" in out
    assert "b.pdf" in out


# ── mcp_http: doc_generate registered in tools/list ───────────────────

def test_tools_list_includes_doc_generate() -> None:
    pytest.importorskip("fastapi")
    from doc_rag.server.mcp_http import _handle_jsonrpc

    st, out = _handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    assert st == 200
    tools = out["result"]["tools"]
    names = [t["name"] for t in tools]
    assert "doc_search" in names
    assert "doc_generate" in names


def test_tools_list_doc_generate_has_max_context_tokens() -> None:
    pytest.importorskip("fastapi")
    from doc_rag.server.mcp_http import _handle_jsonrpc

    _, out = _handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    tools = out["result"]["tools"]
    gen = next(t for t in tools if t["name"] == "doc_generate")
    props = gen["inputSchema"]["properties"]
    assert "max_context_tokens" in props
    assert props["max_context_tokens"]["type"] == "integer"


def test_tools_list_doc_search_has_full_schema() -> None:
    pytest.importorskip("fastapi")
    from doc_rag.server.mcp_http import _handle_jsonrpc

    _, out = _handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    tools = out["result"]["tools"]
    search = next(t for t in tools if t["name"] == "doc_search")
    props = search["inputSchema"]["properties"]
    for key in ("query", "top_k", "namespace", "doc_id", "section_path", "tables_only"):
        assert key in props, f"doc_search missing param: {key}"
