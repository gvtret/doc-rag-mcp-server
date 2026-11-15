"""Context prefixes in embedding texts (task 13).

The chunker drops section headings from chunk text, so a chunk from
section "5.3 Requirements for metering equipment" does not contain
those words unless the body repeats them -- structural documents lose
recall. build_embedding_text() prepends "{title_hint} > {section_path}"
to the EMBEDDING input only: stored chunks.jsonl text and served
doc_search/semantic_search results stay raw.

Covered here:
- build() with title_hints embeds prefixed texts (recording stub
  embedder), for both full build and incremental update;
- chunks without section_path embed as plain text;
- table chunks get a title-only prefix (task 12 already put the
  section_path INTO the table text; repeating it would duplicate it);
- an over-budget prefix is truncated to the 40-token budget, title
  first, then section_path, keeping the separator valid;
- served semantic_search results carry the raw, prefix-free text;
- title_hints=None keeps the old plain-text behaviour.
"""

from __future__ import annotations

import json

import numpy as np

from doc_rag.raglib.embedding_text import build_embedding_text
from doc_rag.raglib.vector_index import VectorIndex


def _make_cfg(tmp_path) -> dict:
    return {
        "_root": str(tmp_path),
        "index": {"backend": "faiss", "metric": "ip"},
        "paths": {"index_dir": "build/index", "chunks_dir": "build/chunks_jsonl"},
        "embeddings": {
            "model_name": "test-model",
            "device": "cpu",
            "batch_size": 4,
            "normalize": True,
        },
        "pipeline_version": "1.0.0",
    }


class _RecordingEmbedder:
    """Deterministic stub embedder that records every encoded text.

    Texts mentioning "zeppelin" map to axis 1, everything else to axis 0,
    so a "zeppelin" query retrieves exactly the zeppelin chunk.
    """

    def __init__(self, dim: int = 8):
        self._dim = dim
        self.texts: list[str] = []

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        self.texts.extend(str(t) for t in texts)
        out = []
        for t in texts:
            v = np.zeros((1, self._dim), dtype="float32")
            v[0, 1 if "zeppelin" in t.lower() else 0] = 1.0
            out.append(v)
        return np.vstack(out)


# ---------------------------------------------------------------------------
# (a) build() with title_hints embeds prefixed texts
# ---------------------------------------------------------------------------


def test_build_embeds_prefixed_text_full_and_incremental(tmp_path, monkeypatch) -> None:
    emb = _RecordingEmbedder()
    monkeypatch.setattr("doc_rag.raglib.vector_index._load_embedder", lambda cfg: emb)
    idx = VectorIndex(_make_cfg(tmp_path))

    chunks = [
        {
            "doc_id": "d0",
            "chunk_id": "d0:0",
            "section_path": "5.3 Requirements for metering equipment",
            "text": "zeppelin body text",
        }
    ]
    assert idx.build(chunks, title_hints={"d0": "GOST Manual"}) is True
    assert emb.texts == [
        "GOST Manual > 5.3 Requirements for metering equipment\n\nzeppelin body text"
    ]

    # Incremental update: only the NEW chunk is embedded, also prefixed.
    emb.texts.clear()
    new_chunks = [
        {"doc_id": "d0", "chunk_id": "d0:1", "section_path": "5.4 Tests", "text": "more body"}
    ]
    assert idx.build(new_chunks, title_hints={"d0": "GOST Manual"}) is True
    assert emb.texts == ["GOST Manual > 5.4 Tests\n\nmore body"]


# ---------------------------------------------------------------------------
# (b) no section_path -> plain text (title alone does not prefix)
# ---------------------------------------------------------------------------


def test_build_without_section_path_embeds_plain_text(tmp_path, monkeypatch) -> None:
    emb = _RecordingEmbedder()
    monkeypatch.setattr("doc_rag.raglib.vector_index._load_embedder", lambda cfg: emb)
    idx = VectorIndex(_make_cfg(tmp_path))

    chunks = [{"doc_id": "d0", "chunk_id": "d0:0", "text": "plain body"}]
    assert idx.build(chunks, title_hints={"d0": "Some Title"}) is True
    assert emb.texts == ["plain body"]


# ---------------------------------------------------------------------------
# (c) table chunk -> title-only prefix (section_path already in the text)
# ---------------------------------------------------------------------------


def test_table_chunk_gets_title_only_prefix(tmp_path, monkeypatch) -> None:
    emb = _RecordingEmbedder()
    monkeypatch.setattr("doc_rag.raglib.vector_index._load_embedder", lambda cfg: emb)
    idx = VectorIndex(_make_cfg(tmp_path))

    chunks = [
        {
            "doc_id": "d0",
            "chunk_id": "d0:0",
            "is_table": True,
            "section_path": "5.3 Parameters",
            "text": "zeppelin table body",
        }
    ]
    assert idx.build(chunks, title_hints={"d0": "GOST Manual"}) is True
    assert emb.texts == ["GOST Manual\n\nzeppelin table body"]


def test_table_chunk_without_title_embeds_plain_text(tmp_path, monkeypatch) -> None:
    emb = _RecordingEmbedder()
    monkeypatch.setattr("doc_rag.raglib.vector_index._load_embedder", lambda cfg: emb)
    idx = VectorIndex(_make_cfg(tmp_path))

    chunks = [
        {"doc_id": "d0", "chunk_id": "d0:0", "is_table": True, "section_path": "5.3", "text": "t"}
    ]
    assert idx.build(chunks, title_hints=None) is True
    assert emb.texts == ["t"]


# ---------------------------------------------------------------------------
# (d) prefix budget: 40 tokens, title cut first, then section_path
# ---------------------------------------------------------------------------


def test_prefix_budget_truncates_title_first(monkeypatch) -> None:
    import doc_rag.raglib.embedding_text as et

    # Deterministic token meter: 1 token per 2 characters.
    monkeypatch.setattr(et, "count_tokens", lambda text, tokenizer_path=None: len(text) // 2)

    title = "T" * 200  # 100 tokens
    sp = "S" * 60  # 30 tokens
    out = build_embedding_text({"text": "body", "section_path": sp}, title)
    prefix, body = out.split("\n\n", 1)
    assert body == "body"
    assert et.count_tokens(prefix) <= 40
    # The section_path survived in full; the title was cut first.
    assert prefix.endswith(" > " + sp)
    title_part = prefix[: -len(" > " + sp)]
    assert title_part.startswith("T")
    assert len(title_part) < len(title)


def test_prefix_budget_drops_title_before_cutting_section_path(monkeypatch) -> None:
    import doc_rag.raglib.embedding_text as et

    monkeypatch.setattr(et, "count_tokens", lambda text, tokenizer_path=None: len(text) // 2)

    title = "T" * 400
    sp = "S" * 200  # alone: 100 tokens, already over the budget
    out = build_embedding_text({"text": "body", "section_path": sp}, title)
    prefix, body = out.split("\n\n", 1)
    assert body == "body"
    assert et.count_tokens(prefix) <= 40
    # No dangling separator: the title (and the " > ") are gone entirely.
    assert " > " not in prefix
    assert set(prefix) == {"S"}


def test_prefix_budget_section_path_only(monkeypatch) -> None:
    import doc_rag.raglib.embedding_text as et

    monkeypatch.setattr(et, "count_tokens", lambda text, tokenizer_path=None: len(text) // 2)

    sp = "S" * 100  # 50 tokens > 40
    out = build_embedding_text({"text": "body", "section_path": sp}, None)
    prefix, body = out.split("\n\n", 1)
    assert body == "body"
    assert et.count_tokens(prefix) <= 40
    assert set(prefix) == {"S"}


def test_title_hint_sanitized_strip_and_80_chars(monkeypatch) -> None:
    import doc_rag.raglib.embedding_text as et

    monkeypatch.setattr(
        et, "count_tokens", lambda text, tokenizer_path=None: max(1, len(text) // 4)
    )
    out = build_embedding_text({"text": "body", "section_path": "5.3"}, "  " + "x" * 100 + "  ")
    prefix, body = out.split("\n\n", 1)
    assert body == "body"
    assert prefix == "x" * 80 + " > 5.3"


# ---------------------------------------------------------------------------
# (e) served results keep the raw, prefix-free text
# ---------------------------------------------------------------------------


def test_semantic_search_serves_raw_text(tmp_path, monkeypatch) -> None:
    from doc_rag.server import retrieval

    emb = _RecordingEmbedder()
    monkeypatch.setattr("doc_rag.raglib.vector_index._load_embedder", lambda cfg: emb)
    retrieval._VECTOR_INDEX_CACHE.clear()

    cfg = _make_cfg(tmp_path)
    chunks = [
        {
            "doc_id": "d0",
            "chunk_id": "d0:0",
            "section_path": "5.3 Requirements",
            "text": "zeppelin marker text",
        },
        {"doc_id": "d1", "chunk_id": "d1:0", "text": "unrelated filler"},
    ]
    idx = VectorIndex(cfg)
    assert idx.build(chunks, title_hints={"d0": "GOST Manual"}) is True

    out = retrieval.semantic_search(cfg, chunks, "zeppelin", top_k=1)
    assert out is not None
    assert [c["chunk_id"] for c in out] == ["d0:0"]
    assert out[0]["text"] == "zeppelin marker text"
    assert "GOST Manual" not in out[0]["text"]
    assert "5.3 Requirements" not in out[0]["text"]


# ---------------------------------------------------------------------------
# (f) title_hints=None -> plain text (backward compatible)
# ---------------------------------------------------------------------------


def test_build_without_title_hints_is_backward_compatible(tmp_path, monkeypatch) -> None:
    """title_hints=None keeps the old plain-text behaviour for chunks
    without section_path; a section_path still gets its prefix (that is
    the recall fix itself, independent of titles)."""
    emb = _RecordingEmbedder()
    monkeypatch.setattr("doc_rag.raglib.vector_index._load_embedder", lambda cfg: emb)
    idx = VectorIndex(_make_cfg(tmp_path))

    chunks = [
        {
            "doc_id": "d0",
            "chunk_id": "d0:0",
            "section_path": "5.3 Requirements",
            "text": "section body",
        },
        {"doc_id": "d1", "chunk_id": "d1:0", "text": "plain body"},
    ]
    assert idx.build(chunks) is True
    assert emb.texts == ["5.3 Requirements\n\nsection body", "plain body"]


# ---------------------------------------------------------------------------
# Pipeline wiring: doc_id -> title map from the manifest
# ---------------------------------------------------------------------------


def test_pipeline_title_hints_from_manifest(tmp_path) -> None:
    from doc_rag.raglib.pipeline import _title_hints_for_index

    p = tmp_path / "manifest.json"
    p.write_text(
        json.dumps(
            {
                "documents": [
                    {"doc_id": "a", "title_hint": "  GOST Manual  "},
                    {"doc_id": "b", "title_hint": "x" * 100},
                    {"doc_id": "c"},
                    {"doc_id": "", "title_hint": "lost"},
                    {"doc_id": "d", "title_hint": "   "},
                    "garbage",
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    hints = _title_hints_for_index(str(p))
    assert hints == {"a": "GOST Manual", "b": "x" * 80}

    assert _title_hints_for_index(str(tmp_path / "missing.json")) == {}

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert _title_hints_for_index(str(broken)) == {}


def test_pipeline_build_index_passes_title_hints(tmp_path, monkeypatch) -> None:
    """_build_index_with_progress forwards title_hints into vi.build()."""
    from doc_rag.raglib import pipeline

    captured: dict = {}

    class _FakeVI:
        def set_progress_callback(self, cb):
            captured["cb"] = cb

        def build(self, chunks, force=False, log=print, title_hints=None):
            captured["title_hints"] = title_hints
            return True

    monkeypatch.setattr(pipeline, "_get_vector_index", lambda cfg: _FakeVI())
    pipeline._build_index_with_progress(
        {}, [{"chunk_id": "d:0", "text": "t"}], force=True, title_hints={"d": "T"}
    )
    assert captured["title_hints"] == {"d": "T"}


def test_embedding_text_version_bumped() -> None:
    from doc_rag.raglib.chunking_meta import EMBEDDING_TEXT_VERSION

    assert EMBEDDING_TEXT_VERSION == "2"
