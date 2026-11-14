"""Tests for atomic writes of index, manifest, and chunks (crash safety).

A crash mid-rebuild must leave the previous faiss.index / index_meta.json /
manifest.json / chunks.jsonl intact and mutually consistent, never truncated
or half-written.
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from doc_rag.raglib.atomic import atomic_write_bytes, atomic_write_text

# -- helpers --


def _make_root(tmp_path: Path) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    for d in [
        "build/docs_md",
        "build/chunks",
        "build/blocks",
        "build/index",
        "sources/archived",
        "sources/incoming",
    ]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "config.yaml").write_text(
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
          strategy: fixed
          target_tokens: 512
          overlap_tokens: 64
        embeddings:
          model_name: BAAI/bge-small-en-v1.5
          device: cpu
    """)
    )
    return tmp_path / "config" / "config.yaml"


def _faiss_or_skip():
    try:
        import faiss  # type: ignore
    except ImportError:
        pytest.skip("faiss not installed")
    return faiss


# -- atomic helper basics --


class TestAtomicWrite:
    def test_write_bytes_and_text(self, tmp_path: Path) -> None:
        p = tmp_path / "sub" / "out.bin"
        atomic_write_bytes(str(p), b"\x00\x01")
        assert p.read_bytes() == b"\x00\x01"

        t = tmp_path / "out.json"
        atomic_write_text(str(t), '{"a": "\u043f\u0440\u0438\u0432\u0435\u0442"}')
        assert t.read_text(encoding="utf-8") == '{"a": "\u043f\u0440\u0438\u0432\u0435\u0442"}'
        assert (
            json.loads(t.read_text(encoding="utf-8"))["a"] == "\u043f\u0440\u0438\u0432\u0435\u0442"
        )

    def test_tmp_file_in_same_directory(self, tmp_path: Path) -> None:
        """The temp file must be created in the target's directory (same fs)."""
        import tempfile as tempfile_mod

        seen: list[str] = []
        real_mkstemp = tempfile_mod.mkstemp

        def spy_mkstemp(*args, **kwargs):
            fd, path = real_mkstemp(*args, **kwargs)
            seen.append(os.path.dirname(path))
            return fd, path

        with patch("doc_rag.raglib.atomic.tempfile.mkstemp", side_effect=spy_mkstemp):
            atomic_write_text(str(tmp_path / "f.txt"), "x")
        assert seen and seen[0] == str(tmp_path)

    def test_failure_leaves_old_file_intact_and_no_tmp(self, tmp_path: Path) -> None:
        """If writing fails, the old file stays and no .tmp litter remains."""
        p = tmp_path / "data.json"
        p.write_text('{"old": true}', encoding="utf-8")
        old_stat = p.stat()

        def failing_replace(src, dst):
            raise OSError("simulated crash before replace")

        with patch("doc_rag.raglib.atomic.os.replace", side_effect=failing_replace):
            with pytest.raises(OSError):
                atomic_write_text(str(p), '{"new": true}')

        assert p.read_text(encoding="utf-8") == '{"old": true}'
        assert p.stat().st_mtime_ns == old_stat.st_mtime_ns
        leftovers = [f for f in os.listdir(tmp_path) if f != "data.json"]
        assert leftovers == [], f"tmp litter left behind: {leftovers}"

    def test_write_failure_mid_stream_leaves_old_file(self, tmp_path: Path) -> None:
        """A crash while the data is being written (before replace) is safe."""
        p = tmp_path / "blob.bin"
        p.write_bytes(b"PREVIOUS")

        real_open = open

        def failing_fdopen(fd, mode="r", *a, **kw):
            fh = real_open(fd, mode, *a, **kw)
            if "w" in mode or "b" in mode:
                real_write = fh.write
                fh.write = lambda data: (_ for _ in ()).throw(RuntimeError("crash mid-write"))
                _ = real_write  # keep reference alive
            return fh

        with patch("doc_rag.raglib.atomic.os.fdopen", side_effect=failing_fdopen):
            with pytest.raises(RuntimeError):
                atomic_write_bytes(str(p), b"NEW")

        assert p.read_bytes() == b"PREVIOUS"


# -- faiss save() crash safety --


class TestFaissSaveAtomic:
    def _make_store(self, tmp_path: Path):
        from doc_rag.raglib.vectorstore.faiss_store import FaissVectorStore

        faiss = _faiss_or_skip()
        store = FaissVectorStore(
            index_path=str(tmp_path / "faiss.index"),
            meta_path=str(tmp_path / "index_meta.json"),
            metric="ip",
        )
        store.init(dim=4, model_name="test-model")
        vecs = np.eye(4, dtype=np.float32)
        store.add(vecs, ["a", "b", "c", "d"])
        return faiss, store

    def test_save_produces_consistent_pair(self, tmp_path: Path) -> None:
        faiss, store = self._make_store(tmp_path)
        store.save()
        assert (tmp_path / "faiss.index").exists()
        meta = json.loads((tmp_path / "index_meta.json").read_text(encoding="utf-8"))
        assert meta["chunk_ids"] == ["a", "b", "c", "d"]
        idx = faiss.read_index(str(tmp_path / "faiss.index"))
        assert idx.ntotal == 4
        # No .tmp litter
        leftovers = [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
        assert leftovers == []

    def test_crash_during_write_index_keeps_old_pair(self, tmp_path: Path) -> None:
        """write_index failing mid-save must not corrupt the previous files."""
        faiss, store = self._make_store(tmp_path)
        store.save()

        old_index_bytes = (tmp_path / "faiss.index").read_bytes()
        old_meta = (tmp_path / "index_meta.json").read_text(encoding="utf-8")

        def crashing_write_index(index, path):
            # Simulate a crash in the middle of writing to the target:
            # truncate whatever path was handed to us, then die.
            with open(path, "wb") as f:
                f.write(b"PART")
            raise RuntimeError("simulated crash mid write_index")

        with patch.object(faiss, "write_index", side_effect=crashing_write_index):
            with pytest.raises(RuntimeError, match="simulated crash"):
                store.save()

        assert (tmp_path / "faiss.index").read_bytes() == old_index_bytes
        assert (tmp_path / "index_meta.json").read_text(encoding="utf-8") == old_meta

    def test_crash_during_meta_write_keeps_old_pair(self, tmp_path: Path) -> None:
        faiss, store = self._make_store(tmp_path)
        store.save()
        old_index_bytes = (tmp_path / "faiss.index").read_bytes()
        old_meta = (tmp_path / "index_meta.json").read_text(encoding="utf-8")

        with patch(
            "doc_rag.raglib.vectorstore.faiss_store.atomic_write_text",
            side_effect=OSError("simulated crash during meta write"),
        ):
            with pytest.raises(OSError):
                store.save()

        assert (tmp_path / "faiss.index").read_bytes() == old_index_bytes
        assert (tmp_path / "index_meta.json").read_text(encoding="utf-8") == old_meta


# -- _append_chunks flush/fsync --


class TestAppendChunks:
    def test_records_survive_after_sync_failure(self, tmp_path: Path) -> None:
        """A crash after some records were written must not lose them."""
        from doc_rag.raglib.pipeline import _append_chunks

        chunks_path = str(tmp_path / "chunks.jsonl")
        first = [{"chunk_id": "a:0", "text": "first"}]
        _append_chunks(chunks_path, first)
        assert (
            json.loads(Path(chunks_path).read_text(encoding="utf-8").splitlines()[0])["chunk_id"]
            == "a:0"
        )

        with patch("doc_rag.raglib.pipeline.os.fsync", side_effect=OSError("crash on fsync")):
            with pytest.raises(OSError):
                _append_chunks(chunks_path, [{"chunk_id": "b:0", "text": "second"}])

        lines = Path(chunks_path).read_text(encoding="utf-8").splitlines()
        assert [json.loads(ln)["chunk_id"] for ln in lines] == ["a:0", "b:0"]
        # No partial (incomplete JSON) line may be present
        for line in lines:
            json.loads(line)  # raises on truncated record

    def test_append_then_read_roundtrip(self, tmp_path: Path) -> None:
        from doc_rag.raglib.pipeline import _append_chunks

        chunks_path = str(tmp_path / "chunks.jsonl")
        _append_chunks(chunks_path, [{"chunk_id": "a:0", "text": "\u0442\u0435\u043a\u0441\u0442"}])
        _append_chunks(chunks_path, [{"chunk_id": "a:1", "text": "more"}])
        lines = Path(chunks_path).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["text"] == "\u0442\u0435\u043a\u0441\u0442"


# -- manifest/chunks writes during rebuild --


class TestRebuildManifestAtomic:
    def _seed_corpus(self, tmp_path: Path, n: int = 2) -> Path:
        config_path = _make_root(tmp_path)
        archived = tmp_path / "sources" / "archived"
        for i in range(n):
            (archived / f"doc{i}.txt").write_text(f"Document {i} content. " * 20)
        return config_path

    def test_manifest_write_failure_keeps_previous_manifest(self, tmp_path: Path) -> None:
        """If the final manifest write fails, the old manifest stays intact."""
        from doc_rag.raglib.pipeline import rebuild

        config_path = self._seed_corpus(tmp_path)
        manifest_path = tmp_path / "build" / "manifest.json"

        rebuild(str(config_path), force=True, skip_index=True)
        first_manifest = manifest_path.read_text(encoding="utf-8")
        assert json.loads(first_manifest)["documents"], "no documents in first manifest"

        # Add a new file and make the manifest write fail (crash simulation).
        (tmp_path / "sources" / "archived" / "extra.txt").write_text("Extra content. " * 20)

        with patch(
            "doc_rag.raglib.pipeline.atomic_write_text",
            side_effect=OSError("simulated crash during manifest write"),
        ):
            with pytest.raises(OSError):
                rebuild(str(config_path), force=False, skip_index=True)

        # The old manifest is untouched (not truncated, not half-written).
        assert manifest_path.read_text(encoding="utf-8") == first_manifest

    def test_incremental_manifest_failure_keeps_previous_manifest(self, tmp_path: Path) -> None:
        """Crash during the per-file incremental manifest write keeps the old one.

        The incremental manifest write is inside a per-file try/except (the
        rebuild continues to the next file), so the failure is logged rather
        than raised, but the manifest on disk must still be the intact
        previous version, never a truncated file.
        """
        from doc_rag.raglib.pipeline import rebuild

        config_path = self._seed_corpus(tmp_path)
        manifest_path = tmp_path / "build" / "manifest.json"

        rebuild(str(config_path), force=True, skip_index=True)

        (tmp_path / "sources" / "archived" / "extra.txt").write_text("Extra content. " * 20)

        calls = {"n": 0}

        def failing_first_write(path, text):
            if str(path).endswith("manifest.json"):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("simulated crash during incremental manifest write")
            from doc_rag.raglib.atomic import atomic_write_text as real_write

            return real_write(path, text)

        with patch("doc_rag.raglib.pipeline.atomic_write_text", side_effect=failing_first_write):
            # The per-file handler catches the incremental-write error;
            # the rebuild continues and its final manifest write succeeds.
            rebuild(str(config_path), force=False, skip_index=True)

        # The failed incremental attempt left the old file intact; the final
        # successful write replaced it wholesale with valid JSON.
        assert calls["n"] >= 2, "expected an incremental write followed by the final one"
        final = manifest_path.read_text(encoding="utf-8")
        assert json.loads(final)["documents"], "manifest must remain valid JSON"

    def test_manifest_content_identical_to_serialization_flags(self, tmp_path: Path) -> None:
        """Manifest JSON must keep the exact old flags: ensure_ascii=False, indent=2."""
        from doc_rag.raglib.pipeline import rebuild

        config_path = self._seed_corpus(tmp_path)
        (tmp_path / "sources" / "archived" / "\u0434\u043e\u043a.txt").write_text(
            "\u0420\u0443\u0441\u0441\u043a\u0438\u0439 \u0442\u0435\u043a\u0441\u0442 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430. "
            * 20
        )
        rebuild(str(config_path), force=True, skip_index=True)

        manifest_path = tmp_path / "build" / "manifest.json"
        raw = manifest_path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        # ensure_ascii=False: Cyrillic must appear literally, not as \uXXXX
        assert "\u0434\u043e\u043a" in raw
        # indent=2: nested structures are indented
        assert '\n  "documents"' in raw or '\n  "schema_version"' in raw
        assert parsed["documents"]
