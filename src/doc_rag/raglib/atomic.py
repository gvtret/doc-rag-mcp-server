"""Atomic file writes for crash-safe index/manifest persistence.

Every write goes to a temporary file in the SAME directory as the target
(so os.replace is a same-filesystem rename), is fsync'ed, then replaces
the target atomically. A crash mid-write leaves the old file intact
instead of a truncated or missing one.
"""

from __future__ import annotations

import os
import tempfile


def fsync_dir(path: str) -> None:
    """Flush a directory's metadata (the rename itself) to disk."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_and_replace(fd: int, tmp_path: str, path: str) -> None:
    """fsync the tmp file, preserve the target's mode, then rename it over."""
    # mkstemp creates the file with 0600; an existing target's permissions
    # must survive the replacement (and a fresh file should be 0644).
    if os.path.exists(path):
        os.fchmod(fd, os.stat(path).st_mode & 0o7777)
    else:
        os.fchmod(fd, 0o644)
    os.fsync(fd)
    os.replace(tmp_path, path)


def atomic_write_bytes(path: str, data: bytes) -> None:
    """Write bytes to path atomically: tmp file in the same dir, fsync, replace."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=directory, prefix=os.path.basename(path) + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            _fsync_and_replace(f.fileno(), tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    fsync_dir(directory)


def atomic_write_text(path: str, text: str) -> None:
    """Write UTF-8 text to path atomically. See atomic_write_bytes."""
    atomic_write_bytes(path, text.encode("utf-8"))
