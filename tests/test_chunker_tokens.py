"""Tests for real-tokenizer chunk sizing (task 11).

Covers:
- count_tokens matches a direct tokenizer.encode and differs from len//4;
- chunk_recursive emits no chunk above the 504-token hard cap, even for a
  single ~14k-token table block;
- graceful fallback when tokenizer.json is missing;
- thread-safety of the lazy tokenizer singleton.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from doc_rag.raglib.chunker_recursive import chunk_recursive
from doc_rag.raglib.chunking_meta import (
    CHUNK_MAX_TOKENS,
    count_tokens,
    reset_count_tokens_for_tests,
)

# Fixed literal text: a Cyrillic technical phrase repeated enough times to
# be unambiguous about real-token vs char/4 estimation.
_PHRASE = (
    "Настоящий раздел описывает порядок подключения измерительных цепей "
    "и требования к заземлению оборудования подстанции.\n\n"
)
TEXT = _PHRASE * 40  # ~2048 chars of Cyrillic technical prose


def _find_model_root() -> Path | None:
    """Nearest dir that carries a real bge-m3 checkout.

    Worktrees share the git dir but not the (untracked) models/, so the
    tokenizer usually lives in the main checkout found via the git
    common dir; falls back to walking up from this file.
    """
    from pathlib import Path as _P

    roots = [_P(__file__).resolve()]
    git_common = _P("/home/trgv/doc-rag/.git")
    if git_common.exists():
        # Worktree: the main checkout is the parent of the common git dir.
        roots.append(git_common.parent)
    for root in roots:
        for candidate in [root, *root.parents]:
            for sub in ("models/bge-m3", "build/models/bge-m3"):
                if (candidate / sub / "tokenizer.json").is_file():
                    return candidate
    return None


@pytest.fixture(autouse=True)
def _real_model_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point model discovery at the nearest checkout with a real model."""
    root = _find_model_root()
    if root is not None:
        monkeypatch.setenv("DOC_RAG_ROOT", str(root))


@pytest.fixture(autouse=True)
def _reset_tokenizer_cache():
    """Isolate the tokenizer singleton (cache + WARN latch) between tests."""
    reset_count_tokens_for_tests()
    yield
    reset_count_tokens_for_tests()


def _real_tokenizer():
    from doc_rag.raglib.model_paths import resolve_model_path

    model_dir = resolve_model_path({})
    assert model_dir is not None, "real bge-m3 model dir must resolve for these tests"
    tok_path = model_dir / "tokenizer.json"
    assert tok_path.is_file(), f"missing tokenizer.json under {model_dir}"
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(tok_path))


def test_count_tokens_matches_direct_encode() -> None:
    tok = _real_tokenizer()
    expected = len(tok.encode(TEXT).ids)
    got = count_tokens(TEXT)
    assert got == expected
    # The whole point of the change: real count differs from len//4 on
    # Cyrillic text (bge-m3 tokenizes it far less efficiently than /4).
    assert got != len(TEXT) // 4


def test_count_tokens_empty() -> None:
    assert count_tokens("") == 0


def test_chunker_caps_table_at_504_real_tokens() -> None:
    # Build a markdown-ish table whose real token count is ~14k: each row
    # of dense Cyrillic table syntax is ~15-16 real tokens.
    row = "| датчик | температура | давление | состояние канала измерения |\n"
    table_text = "| параметр | ед. изм. | значение | примечание |\n" + row * 900
    assert count_tokens(table_text) > 14000

    from doc_rag.raglib.blocks import Block

    blocks = [
        Block(
            block_id="t:0",
            doc_id="doc",
            type="table",
            text=table_text,
            source_backend="docling",
        )
    ]
    chunks = chunk_recursive(blocks, target_tokens=512)
    assert len(chunks) > 1
    for c in chunks:
        real = count_tokens(c["text"])
        assert real <= CHUNK_MAX_TOKENS, f"chunk has {real} real tokens > {CHUNK_MAX_TOKENS}"


def test_chunker_caps_paragraphs_at_504_real_tokens() -> None:
    from doc_rag.raglib.blocks import Block

    para = (
        "Измерительные цепи подключаются через клеммные сборки с обязательной "
        "проверкой сопротивления изоляции перед вводом в эксплуатацию.\n\n"
    )
    blocks = [
        Block(
            block_id="p:0",
            doc_id="doc",
            type="paragraph",
            text=para * 80,
            source_backend="docling",
        )
    ]
    chunks = chunk_recursive(blocks, target_tokens=512)
    assert len(chunks) > 1
    for c in chunks:
        assert count_tokens(c["text"]) <= CHUNK_MAX_TOKENS


def test_fallback_without_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    # Point model resolution at a root with no model dir: no tokenizer.json
    # anywhere -> len//4 fallback, value equals the documented formula.
    monkeypatch.setenv("DOC_RAG_ROOT", "/nonexistent/doc-rag-root")
    got = count_tokens(TEXT)
    assert got == max(1, len(TEXT) // 4)
    # Same call again: stable, and the WARN latch must not raise.
    assert count_tokens(TEXT) == got


def test_count_tokens_thread_safety() -> None:
    # 8 threads racing on the lazy singleton must all agree and produce the
    # same value as a single-threaded call.
    expected = count_tokens(TEXT)
    results: list[int] = []
    lock = threading.Lock()

    def worker() -> None:
        v = count_tokens(TEXT)
        with lock:
            results.append(v)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [expected] * 8
