"""Embedding-text construction: context prefixes for corpus chunks.

Section headings are dropped from chunk text by the chunker, so a chunk
from section "5.3 Requirements for metering equipment" does not contain
those words unless the body repeats them -- recall on structural
documents suffers. build_embedding_text() prepends
"{title_hint} > {section_path}" to the EMBEDDING input only. The stored
chunk text and everything served to the LLM stays raw; queries are
encoded without a prefix (the query side must mirror what the corpus
vectors were built from, which is the prefixed text, but a bare query
already has no headings to mirror).

Table chunks (is_table=True) get a title-only prefix: task 12's table
chunker already embeds the section_path IN the table text, so adding
it to the prefix would duplicate it.

The prefix carries a budget of PREFIX_TOKEN_BUDGET tokens, measured
with chunking_meta.count_tokens. When the combined prefix is over
budget, title_hint is truncated first (it is the least specific part),
then section_path; the separator stays valid (no dangling " > "), and
trimmed parts carry no trailing whitespace.

SEPARATE FROM CHUNK_BUDGET: the prefix lives ON TOP of the chunk
text budget (CHUNK_MAX_TOKENS = 504 in chunking_meta), so a saturated
chunk embeds up to 504 + 40 + separator tokens. embeddings.max_seq_length
must therefore be >= 512 + PREFIX_TOKEN_BUDGET (40) + headroom -- the
deployed config uses 576. Without that headroom the model silently
truncates the tail of the embedding input (no error, lost tokens).
"""

from __future__ import annotations

from typing import Any

from doc_rag.raglib.chunking_meta import count_tokens

# Token budget for the whole prefix (title_hint + separator + section_path).
PREFIX_TOKEN_BUDGET = 40

# title_hint is operator-level metadata (derived from a filename); cap it
# so a pathological filename cannot eat the whole budget.
TITLE_HINT_MAX_CHARS = 80


def _sanitize_title_hint(title_hint: Any) -> str:
    """Strip and cap title_hint; non-strings collapse to ""."""
    if not isinstance(title_hint, str):
        return ""
    return title_hint.strip()[:TITLE_HINT_MAX_CHARS]


def _truncate_to_budget(prefix: str, budget: int) -> str:
    """Truncate `prefix` to at most `budget` tokens, cutting at char level.

    NOT word-whole: binary search finds the largest character cut whose
    token count fits the budget (count_tokens is monotone in text length
    for both the real tokenizer and the len//4 fallback, so bisect is
    exact). A cut may land mid-word; callers rstrip the result.
    """
    if count_tokens(prefix) <= budget:
        return prefix
    # Binary search on the character cut point: count_tokens is monotone,
    # so bisect finds the largest cut that still fits the budget.
    lo, hi = 0, len(prefix)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if count_tokens(prefix[:mid]) <= budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return prefix[:best]


def _trim_prefix(title: str, section_path: str) -> tuple[str, str]:
    """Fit "{title} > {section_path}" into the budget.

    Title is truncated first; if even the empty title leaves the prefix
    over budget (i.e. section_path alone is too big), the title is
    dropped entirely and section_path is truncated on its own. Returns
    the possibly-truncated (title, section_path).
    """
    sep = " > "
    combined = f"{title}{sep}{section_path}" if title else section_path
    if count_tokens(combined) <= PREFIX_TOKEN_BUDGET:
        return title, section_path

    if title:
        # Largest title cut whose WHOLE prefix still fits. Measuring the
        # combined string (not title and rest separately) keeps the
        # guarantee exact for any token meter, including ones where the
        # count of a concatenation differs from the sum of its parts.
        lo, hi = 0, len(title)
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if count_tokens(title[:mid] + sep + section_path) <= PREFIX_TOKEN_BUDGET:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best > 0:
            # rstrip: a char-level cut can leave trailing whitespace, which
            # would otherwise render the separator as "  > ".
            return title[:best].rstrip(), section_path
        # No title cut fits: drop the title, budget section_path alone.
        return "", _truncate_to_budget(section_path, PREFIX_TOKEN_BUDGET).rstrip()

    # section_path only (title absent or dropped): truncate it.
    return "", _truncate_to_budget(section_path, PREFIX_TOKEN_BUDGET).rstrip()


def build_embedding_text(chunk: dict[str, Any], title_hint: str | None) -> str:
    """Build the text to EMBED for `chunk`, with a context prefix.

    "{title_hint} > {section_path}\\n\\n{text}" when both are present;
    "{section_path}\\n\\n{text}" when only section_path is; plain text
    otherwise. Table chunks (is_table=True) take a title-only prefix
    because their text already contains the section_path (task 12).
    The prefix is truncated to PREFIX_TOKEN_BUDGET tokens, title first.
    """
    text = str(chunk.get("text", ""))
    is_table = bool(chunk.get("is_table"))
    section_path = str(chunk.get("section_path", "") or "").strip()

    title = _sanitize_title_hint(title_hint)
    if is_table:
        # section_path is already IN the table text (task 12): prefixing
        # it again would duplicate it. Title only, same 40-token budget
        # as the full prefix (a filename-derived title can be long).
        if title:
            return f"{_truncate_to_budget(title, PREFIX_TOKEN_BUDGET).rstrip()}\n\n{text}"
        return text

    if not section_path:
        # A lone title does not prefix: without a section_path there is no
        # structural context to add, and the plain text keeps embedding
        # behaviour identical to pre-task chunks.
        return text

    prefix_title, prefix_sp = _trim_prefix(title, section_path)
    if prefix_title:
        prefix = f"{prefix_title} > {prefix_sp}"
    else:
        prefix = prefix_sp
    return f"{prefix}\n\n{text}"
