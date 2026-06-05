"""Magic-bytes file-type detection for the v1.5 parser dispatch.

Pre-v1.5 the pipeline chose a parser based on the file's extension. That
breaks the moment someone drops `report.pdf` into `sources/incoming/`
and it turns out to be a renamed `.docx` (or a `.zip`, or `.doc`, …).
v1.5 sniffs the actual content via [`filetype`](https://github.com/h2non/filetype.py)
— a pure-Python, MIT-licensed magic-bytes library, no `libmagic`
dependency.

`detect_supported_extension(path)` returns one of the five supported
extensions (`.pdf`, `.docx`, `.doc`, `.md`, `.txt`), preferring content
sniffing when it disagrees with the filename. Plain-text formats
(`.md`, `.txt`) have no magic bytes and are inferred from the filename
extension as a fallback.
"""

from __future__ import annotations

from pathlib import Path

#: Supported extensions in priority order — what the parser dispatch
#: knows how to handle.
_SUPPORTED: tuple[str, ...] = (".pdf", ".docx", ".doc", ".md", ".txt")

#: Map filetype-library MIME strings to our supported extensions.
_MIME_TO_EXT: dict[str, str] = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".doc",
    # filetype reports .zip for any PKZIP container — DOCX is one, so we
    # check the zip path against `is_docx_zip()` below when we see
    # `application/zip`. .epub / .xlsx are also zips; they map to
    # "unsupported" below, not to ".docx".
    "application/zip": ".zip",
}


def _filetype_guess(path: str | Path) -> object | None:
    """Wrap `filetype.guess()` with a defensive try/except.

    Returns the filetype `Kind` object (whose `.extension` and `.mime`
    we look at) or `None` if filetype is not installed or fails.
    """
    try:
        import filetype  # type: ignore

        return filetype.guess(str(path))
    except Exception:
        return None


def _is_docx_zip(path: str | Path) -> bool:
    """Quick check whether a ZIP is specifically a DOCX (Word XML inside)."""
    try:
        import zipfile

        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
        # DOCX always carries word/document.xml; .xlsx has xl/workbook.xml,
        # .pptx has ppt/presentation.xml, .odt has content.xml without the
        # word/ prefix.
        return "word/document.xml" in names
    except Exception:
        return False


def detect_supported_extension(path: str | Path) -> str | None:
    """Resolve the parser-dispatch extension for `path`.

    Returns one of `.pdf` / `.docx` / `.doc` / `.md` / `.txt` (always
    lowercase, leading dot), or `None` if the file's content does not
    match any supported format and the filename extension is not
    informative either.

    Resolution order:
      1. **Magic bytes** via `filetype.guess()`. PDF, DOCX (as zip with
         word/document.xml), and DOC (CFB) are detected by content.
      2. **Filename extension** as a fallback for the plain-text formats
         (`.md` and `.txt`) that have no magic bytes, and when filetype
         is unavailable.
      3. **None** when neither path yields a supported extension.
    """
    p = Path(path)
    if not p.is_file():
        return None

    name_ext = p.suffix.lower()

    kind = _filetype_guess(p)
    if kind is not None:
        mime = getattr(kind, "mime", "") or ""
        mapped = _MIME_TO_EXT.get(mime)
        if mapped == ".zip":
            # Resolve the zip ambiguity: DOCX is the only supported zip.
            return ".docx" if _is_docx_zip(p) else None
        if mapped in _SUPPORTED:
            return mapped
        # `filetype` returned something we don't support (jpg, mp3, …).
        # Fall through to filename extension in case the user just
        # renamed a `.txt` to a media file by accident — extremely rare,
        # but trivially correct.

    if name_ext in _SUPPORTED:
        return name_ext

    return None


def filename_extension_disagrees_with_content(path: str | Path) -> bool:
    """True when the filename extension does not match the sniffed type.

    Useful for surfacing a warning during ingest: `report.pdf` that is
    really a `.docx` will still be parsed correctly, but the operator
    probably wants to know about it.
    """
    p = Path(path)
    name_ext = p.suffix.lower()
    sniffed = detect_supported_extension(p)
    if sniffed is None or name_ext not in _SUPPORTED:
        return False
    return sniffed != name_ext


def is_supported(path: str | Path) -> bool:
    """True if `detect_supported_extension(path)` returns a known type."""
    return detect_supported_extension(path) is not None


__all__ = [
    "detect_supported_extension",
    "filename_extension_disagrees_with_content",
    "is_supported",
]


# Expose the canonical extension list for callers that want to enumerate.
SUPPORTED_EXTENSIONS: tuple[str, ...] = _SUPPORTED


def _diagnostic_summary(path: str | Path) -> dict[str, str | None]:
    """Debug aid: report both the name-based and content-based view.

    Not used by production code; useful when investigating an ingest
    failure on a misnamed file.
    """
    p = Path(path)
    kind = _filetype_guess(p)
    return {
        "filename_extension": p.suffix.lower(),
        "filetype_mime": getattr(kind, "mime", None),
        "filetype_extension": ".%s" % kind.extension if kind is not None else None,  # noqa: UP031
        "resolved_extension": detect_supported_extension(p),
    }
