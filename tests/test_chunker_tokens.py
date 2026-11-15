"""Tests for real-tokenizer chunk sizing (task 11).

Covers:
- count_tokens matches a direct tokenizer.encode and differs from len//4;
- chunk_recursive emits no chunk above the 504-token hard cap, even for a
  single ~14k-token table block;
- graceful fallback when tokenizer.json is missing;
- thread-safety of the lazy tokenizer singleton.

Tests that need the real bge-m3 tokenizer skip cleanly on machines
without a local model checkout (CI): the fallback-behaviour test does
NOT skip -- it exercises exactly the missing-model path.
"""

from __future__ import annotations

import subprocess
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


def _git_common_dir() -> Path | None:
    """Repo root / git common dir, discovered via git, no hardcoded paths.

    Worktrees share the git dir but not the (untracked) models/, so the
    tokenizer usually lives in the main checkout the common dir points
    at. Falls back to walking up from this file.
    """
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
    """Point model discovery at the nearest checkout with a real model."""
    if _MODEL_ROOT is not None:
        monkeypatch.setenv("DOC_RAG_ROOT", str(_MODEL_ROOT))


@pytest.fixture(autouse=True)
def _reset_tokenizer_cache():
    """Isolate the tokenizer singleton (cache + seeded path) between tests."""
    reset_count_tokens_for_tests()
    yield
    reset_count_tokens_for_tests()


def _real_tokenizer():
    if _MODEL_ROOT is None:
        pytest.skip("bge-m3 tokenizer not available")
    from doc_rag.raglib.model_paths import resolve_model_path

    model_dir = resolve_model_path({})
    tok_path = model_dir / "tokenizer.json" if model_dir else None
    if tok_path is None or not tok_path.is_file():
        pytest.skip("bge-m3 tokenizer not available")
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(tok_path))


def _require_tokenizer() -> None:
    if _MODEL_ROOT is None:
        pytest.skip("bge-m3 tokenizer not available")


def test_count_tokens_matches_direct_encode() -> None:
    tok = _real_tokenizer()
    expected = len(tok.encode(TEXT).ids)
    got = count_tokens(TEXT)
    assert got == expected
    # The whole point of the change: real count differs from len//4 on
    # Cyrillic text (bge-m3 tokenizes it far less efficiently than /4).
    assert got != len(TEXT) // 4


def test_configure_tokenizer_seeds_model_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # configure_tokenizer(cfg) must seed the default path so that
    # count_tokens honours embeddings.model_path from the cfg.
    pytest.importorskip("tokenizers")
    if _MODEL_ROOT is None:
        pytest.skip("bge-m3 tokenizer not available")
    from tokenizers import Tokenizer

    from doc_rag.raglib.chunking_meta import _default_tokenizer_file, configure_tokenizer

    model_dir = _MODEL_ROOT / "models" / "bge-m3"
    if not (model_dir / "tokenizer.json").is_file():
        model_dir = _MODEL_ROOT / "build" / "models" / "bge-m3"
    cfg = {"embeddings": {"model_path": str(model_dir)}}
    configure_tokenizer(cfg)
    seeded = _default_tokenizer_file()
    assert seeded is not None and seeded == str(model_dir / "tokenizer.json")
    # count_tokens without explicit path uses the seeded tokenizer.
    tok_expect = len(Tokenizer.from_file(seeded).encode(TEXT).ids)
    assert count_tokens(TEXT) == tok_expect


def test_count_tokens_empty() -> None:
    assert count_tokens("") == 0


def test_chunker_caps_table_at_504_real_tokens() -> None:
    _require_tokenizer()
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
    table_chunks = [c for c in chunks if c.get("is_table")]
    assert table_chunks, "table text must survive as table chunk(s)"
    assert all(c.get("is_table_fragment") for c in table_chunks), (
        "cap-split table pieces must carry the fragment marker"
    )
    for c in chunks:
        real = count_tokens(c["text"])
        assert real <= CHUNK_MAX_TOKENS, f"chunk has {real} real tokens > {CHUNK_MAX_TOKENS}"


def test_chunker_caps_paragraphs_at_504_real_tokens() -> None:
    _require_tokenizer()
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
    # Deliberately NOT skipped without a model: this IS the missing-model
    # path.
    monkeypatch.setenv("DOC_RAG_ROOT", "/nonexistent/doc-rag-root")
    got = count_tokens(TEXT)
    assert got == max(1, len(TEXT) // 4)
    # Same call again: stable, and the WARN latch must not raise.
    assert count_tokens(TEXT) == got


def test_fixed_chunk_text_size_band() -> None:
    # _chunk_text (fixed strategy) must produce chunks in a reasonable
    # real-token band around the target: median >= 300 at target 512
    # (ratio 3 chars/token), and every chunk <= the 504 hard cap. The
    # char window only shapes sizes; the _split_text_token_budget pass
    # provides the guarantee.
    _require_tokenizer()
    from doc_rag.raglib.pipeline import _chunk_text

    para = (
        "Измерительные цепи подключаются через клеммные сборки с обязательной "
        "проверкой сопротивления изоляции перед вводом в эксплуатацию. "
        "Персонал обязан использовать средства защиты и соблюдать наряд-допуск. "
    )
    text = para * 40  # ~30k chars of Russian prose
    chunks = _chunk_text(text, target_tokens=512, overlap_tokens=64)
    assert len(chunks) > 1
    sizes = sorted(count_tokens(c) for c in chunks)
    median = sizes[len(sizes) // 2]
    assert median >= 300, f"median chunk {median} real tokens, want >= 300"
    for c in chunks:
        assert count_tokens(c) <= CHUNK_MAX_TOKENS


def test_count_tokens_thread_safety() -> None:
    _require_tokenizer()
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
