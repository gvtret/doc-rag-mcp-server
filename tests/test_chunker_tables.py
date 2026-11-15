"""Tests for table chunking (task 12).

Covers:
- a table within the 504-token cap stays ONE chunk whose text starts
  with the section_path line + header line, with no summary sibling;
- an oversized (~14k real tokens) table splits at row boundaries:
  every chunk starts with the same prefix, every chunk fits the hard
  cap, and no line is cut mid-row;
- no `is_table_summary` chunks and no Russian "Таблица (...)" summaries
  are emitted anywhere;
- an empty table produces no chunk at all (no summary, no crash).

The oversized-table test needs the real bge-m3 tokenizer and skips
cleanly on machines without a local model checkout.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from doc_rag.raglib.blocks import Block
from doc_rag.raglib.chunker_recursive import chunk_recursive
from doc_rag.raglib.chunking_meta import (
    CHUNK_MAX_TOKENS,
    count_tokens,
    reset_count_tokens_for_tests,
)

_HEADER = "| параметр | ед. изм. | значение | примечание |"
_ROW = "| датчик | температура | давление | состояние канала измерения |"
_SECTION = "Параметры измерений"
# prefix injected into every table chunk text
_PREFIX = _SECTION + "\n" + _HEADER


def _git_common_dir() -> Path | None:
    """Repo root / git common dir, discovered via git (see test_chunker_tokens)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        if out:
            p = Path(out)
            if not p.is_absolute():
                p = Path.cwd() / p
            return p.resolve()
    except (OSError, subprocess.SubprocessError):
        pass
    return Path(__file__).resolve().parents[1]


def _find_model_root() -> Path | None:
    """Nearest dir that carries a real bge-m3 checkout, or None."""
    roots = [_git_common_dir(), Path(__file__).resolve()]
    seen: set[Path] = set()
    for root in roots:
        if root is None:
            continue
        for candidate in [root, *root.parents]:
            if candidate in seen:
                continue
            seen.add(candidate)
            for sub in ("models/bge-m3", "build/models/bge-m3"):
                if (candidate / sub / "tokenizer.json").is_file():
                    return candidate
    return None


_MODEL_ROOT = _find_model_root()


@pytest.fixture(autouse=True)
def _real_model_root(monkeypatch: pytest.MonkeyPatch) -> None:
    if _MODEL_ROOT is not None:
        monkeypatch.setenv("DOC_RAG_ROOT", str(_MODEL_ROOT))


@pytest.fixture(autouse=True)
def _reset_tokenizer_cache():
    reset_count_tokens_for_tests()
    yield
    reset_count_tokens_for_tests()


def _block(**kwargs) -> Block:
    defaults = dict(
        block_id="tmp:0000",
        doc_id="test",
        type="paragraph",
        text="hello world",
        source_backend="docling",
    )
    defaults.update(kwargs)
    return Block(**defaults)


def _assert_no_summary(chunks: list[dict]) -> None:
    """Grep-style guard: no summary chunks, no Russian summary text."""
    for c in chunks:
        assert not c.get("is_table_summary"), "summary chunk leaked"
        assert "is_table_summary" not in c, "summary key leaked"
        assert "Таблица (" not in c["text"], f"Russian summary text leaked: {c['text'][:80]!r}"


def test_small_table_single_chunk_with_prefix() -> None:
    """Table within the cap -> ONE chunk, prefixed, no summary sibling."""
    table = _block(type="table", text=_HEADER + "\n| a | b | c | d |\n| e | f | g | h |")
    blocks = [
        _block(type="heading", level=1, text=_SECTION),
        table,
    ]
    chunks = chunk_recursive(blocks, target_tokens=512)
    table_chunks = [c for c in chunks if c.get("is_table")]
    assert len(table_chunks) == 1
    chunk = table_chunks[0]
    # text starts with section_path line, then the table header line
    assert chunk["text"].startswith(_PREFIX + "\n")
    assert chunk["text"].count("\n") >= 2  # data rows survive
    assert not chunk.get("is_table_fragment"), "small table is not a fragment"
    assert chunk["section_path"] == _SECTION
    # whole chunk fits the hard cap
    assert count_tokens(chunk["text"]) <= CHUNK_MAX_TOKENS
    _assert_no_summary(chunks)


def test_oversized_table_splits_at_row_boundaries() -> None:
    """~14k-token table -> chunks <= cap, prefixed, rows never cut mid-line."""
    if _MODEL_ROOT is None:
        pytest.skip("bge-m3 tokenizer not available")
    # 900 identical dense Cyrillic rows: ~14k real tokens overall
    table_text = _HEADER + "\n" + "\n".join([_ROW] * 900)
    assert count_tokens(table_text) > 14000
    blocks = [
        _block(type="heading", level=1, text=_SECTION),
        _block(type="table", text=table_text),
    ]
    chunks = chunk_recursive(blocks, target_tokens=512)
    table_chunks = [c for c in chunks if c.get("is_table")]
    assert len(table_chunks) > 1, "oversized table must split"

    rows_seen = 0
    for c in table_chunks:
        real = count_tokens(c["text"])
        assert real <= CHUNK_MAX_TOKENS, f"chunk has {real} real tokens > {CHUNK_MAX_TOKENS}"
        assert c.get("is_table_fragment"), "split piece must be flagged as fragment"
        lines = c["text"].split("\n")
        # prefix: section_path line, then header line
        assert lines[0] == _SECTION, f"chunk must start with section_path: {lines[0]!r}"
        assert lines[1] == _HEADER, "second line must be the table header"
        # every data line is a complete table row: no mid-line cuts
        data = lines[2:]
        assert data, "each piece carries at least one data row"
        assert all(line == _ROW for line in data), "row text must be intact (no mid-row cuts)"
        rows_seen += len(data)
    _assert_no_summary(chunks)
    # no rows lost or duplicated across the split
    assert rows_seen == 900, f"expected 900 data rows across chunks, got {rows_seen}"


def test_no_summary_chunks_for_mixed_content() -> None:
    """Tables interleaved with paragraphs emit no summary chunks at all."""
    blocks = [
        _block(type="heading", level=1, text="Intro"),
        _block(text="Первый абзац перед таблицей."),
        _block(type="table", text=_HEADER + "\n" + _ROW),
        _block(text="Абзац после таблицы."),
    ]
    chunks = chunk_recursive(blocks, target_tokens=512)
    assert len(chunks) >= 3
    _assert_no_summary(chunks)
    assert all(not c.get("is_table_summary") for c in chunks)


def test_empty_table_no_chunks() -> None:
    """Empty table -> no chunk, no summary, no crash."""
    for text in ("", "\n\n"):
        chunks = chunk_recursive([_block(type="table", text=text)], target_tokens=512)
        assert not [c for c in chunks if c.get("is_table")], f"text={text!r}"
        _assert_no_summary(chunks)
