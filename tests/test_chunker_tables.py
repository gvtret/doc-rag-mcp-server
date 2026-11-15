"""Tests for table chunking (task 12).

Covers:
- a table within the 504-token cap stays ONE chunk whose text is
  exactly the section_path line + header line + data rows, with no
  summary sibling;
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

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

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
    row_a, row_b = "| a | b | c | d |", "| e | f | g | h |"
    table = _block(type="table", text=_HEADER + "\n" + row_a + "\n" + row_b)
    blocks = [
        _block(type="heading", level=1, text=_SECTION),
        table,
    ]
    chunks = chunk_recursive(blocks, target_tokens=512)
    table_chunks = [c for c in chunks if c.get("is_table")]
    assert len(table_chunks) == 1
    chunk = table_chunks[0]
    # The exact expected line set: section_path, header, then both data
    # rows, in order, with nothing else mixed in.
    assert chunk["text"].split("\n") == [_SECTION, _HEADER, row_a, row_b]
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


# --------------------------------------------------------------------------
# Pipeline persistence (T12 review finding 3): is_table_fragment survives
# into chunks.jsonl, and is absent for whole (unsplit) tables.
# --------------------------------------------------------------------------


def _make_pipeline_root(base: Path) -> Path:
    """Minimal corpus root with recursive chunking, like test_atomic_writes."""
    import textwrap

    config_dir = base / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    for d in [
        "build/docs_md",
        "build/chunks",
        "build/blocks",
        "build/index",
        "sources/archived",
        "sources/incoming",
    ]:
        (base / d).mkdir(parents=True, exist_ok=True)
    (base / "config" / "config.yaml").write_text(
        textwrap.dedent("""\
        paths:
          sources_archived: sources/archived
          sources_incoming: sources/incoming
          docs_md_dir: build/docs_md
          chunks_dir: build/chunks
          manifest_path: build/manifest.json
          blocks_dir: build/blocks
          index_dir: build/index
        parsing:
          pdf_backend: docling
        chunking:
          strategy: recursive
          target_tokens: 512
          overlap_tokens: 64
        embeddings:
          model_name: BAAI/bge-small-en-v1.5
          device: cpu
        sources:
          archive_after_ingest: false
          incremental_ingest: false
    """)
    )
    return base / "config" / "config.yaml"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def test_pipeline_persists_is_table_fragment(tmp_path: Path) -> None:
    """chunks.jsonl carries is_table_fragment on split-table pieces and
    NOT on whole tables (pipeline copy sites, review finding 3)."""
    import json as _json

    from doc_rag.raglib.pipeline import rebuild

    config_path = _make_pipeline_root(tmp_path)
    # One doc with an oversized table: .md/.txt sources go through the
    # baseline parser (single paragraph block), so patch parse_document
    # to return a real table block list.
    archived = tmp_path / "sources" / "archived"
    src = archived / "tables.txt"
    src.write_text("table document")

    table_small = _HEADER + "\n" + _ROW
    # Distinct rows: _postprocess_chunks md5-dedups identical chunk texts,
    # so identical rows across fragments would collapse the pieces before
    # we can assert on them (see report "concerns").
    table_big = (
        _HEADER
        + "\n"
        + "\n".join(
            f"| датчик {i:03d} | температура | давление | состояние канала |" for i in range(900)
        )
    )

    def fake_parse(cfg, path):
        from doc_rag.raglib.blocks import blocks_to_markdown

        blocks = [
            Block(
                block_id="tmp:0000",
                doc_id="tmp",
                type="heading",
                level=1,
                text=_SECTION,
                source_backend="direct",
            ),
            Block(
                block_id="tmp:0001",
                doc_id="tmp",
                type="table",
                text=table_small,
                source_backend="direct",
            ),
            Block(
                block_id="tmp:0002",
                doc_id="tmp",
                type="table",
                text=table_big,
                source_backend="direct",
            ),
        ]
        return {
            "markdown": blocks_to_markdown(blocks, "tables"),
            "tables": [],
            "stats": {},
            "blocks": blocks,
        }

    from doc_rag.raglib import pipeline

    with patch.object(pipeline, "parse_document", side_effect=fake_parse):
        rebuild(str(config_path), force=True, skip_index=True)

    chunks_path = tmp_path / "build" / "chunks" / "chunks.jsonl"
    persisted = _read_jsonl(chunks_path)
    assert persisted, "no chunks persisted"
    table_chunks = [c for c in persisted if c.get("is_table")]
    assert table_chunks, "no is_table chunks persisted"
    _assert_no_summary(persisted)

    fragments = [c for c in table_chunks if c.get("is_table_fragment")]
    wholes = [c for c in table_chunks if not c.get("is_table_fragment")]
    # the oversized table split -> fragments exist; the small table did not
    assert fragments, "fragments of the oversized table must be persisted"
    assert len(fragments) > 1, "oversized table must yield multiple pieces"
    assert any(count_tokens(c["text"]) <= CHUNK_MAX_TOKENS for c in fragments)
    # each fragment still carries the prefixed header line
    for c in fragments:
        lines = c["text"].split("\n")
        assert lines[0] == _SECTION and lines[1] == _HEADER
    # the small table persisted whole: exactly one non-fragment table chunk
    # whose text is the full table (header + one row) and has no fragment flag
    assert len(wholes) == 1, f"expected exactly 1 whole table chunk, got {len(wholes)}"
    assert _json.dumps(wholes[0], ensure_ascii=False).count("is_table_fragment") == 0
    assert wholes[0]["text"].startswith(_PREFIX)
    _assert_no_summary(table_chunks)
