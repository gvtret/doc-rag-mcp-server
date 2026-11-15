"""Precomputed BM25 lexical index over chunks (Task 14).

Replaces the per-query DF recomputation in retrieval.lexical_search,
which walked every chunk twice per query (6-8 s at 20k chunks). The
index is built once per chunks.jsonl revision (mtime-keyed singleton)
and serves BM25 scoring plus the Task 7 metadata filters in
milliseconds.

No third-party NLP: stemming is a hand-rolled suffix normalizer with an
explicit RU suffix list and s/es handling for EN, per the task brief
(dependency freeze: no snowball/nltk/pymorphy/rank_bm25).

BM25: k1 = 1.2, b = 0.75, idf(t) = log(1 + (N - df + 0.5)/(df + 0.5)).
Code tokens ("62056-5-3", "32144-2013") survive tokenization whole, are
exempt from stemming and get a x1.5 query weight bonus. The exact-phrase
bonus from the old scorer is kept, normalized: min(5.0, 2.0 + 3.0*count).

Code-term prefix expansion: a query code token that is itself not in
postings (or additionally to the exact hit) is expanded to the <= 50
most frequent vocabulary terms CONTAINING it as a substring, each
carrying the same 1.5x code weight ("34.12" matches text "34.12-2018",
"62056-5" matches "62056-5-3" — the old scorer served these via raw
substring phrase matching; the whole-phrase count still applies via the
phrase bonus when the query string appears verbatim).

RU suffix removal nuance vs the brief: the brief says "words >= 5
letters"; implemented as "the token is >= 5 letters AND the remaining
stem after suffix removal is >= 4 letters" (cut >= _MIN_STEM_LEN - 1
with _MIN_STEM_LEN = 5). This keeps "шифра"/"шифры" -> "шифр" (stem 4)
while preventing "вода" -> "вод" garbage cuts on 4-letter stems. Without
the stem guard, single-letter stems ("и" from "имя"-class words) would
join unrelated postings.

Memory: the singleton for the real corpus (20498 chunks, vocab ~84k)
adds ~150-160 MB RSS on top of the already-loaded chunk dicts (measured via
ru_maxrss delta; the chunk dicts themselves are another ~780 MB process
peak shared with the rest of the server). postings tuples, per-chunk
metadata and the lowercase phrase_haystacks copy dominate the index
footprint; phrase_haystacks keeps the verbatim lowercase text per chunk
on purpose for the exact-phrase bonus (a deduplicated/interned
representation was considered and rejected to preserve old-scorer
behavior).
"""

from __future__ import annotations

import math
import os
import re
import threading
from typing import Any

K1 = 1.2
B = 0.75

# Explicit RU suffix list, applied once per token: the LONGEST matching
# suffix wins (max by length, list order irrelevant). A token is eligible
# when it has >= _MIN_STEM_LEN letters and the stem left after the cut
# has >= _MIN_STEM_LEN - 1 letters (see module docstring for the nuance).
_RU_SUFFIXES: tuple[str, ...] = (
    "ования",
    "ироваться",
    "ированный",
    "ировала",
    "ирует",
    "ировал",
    "ация",
    "ыми",
    "ями",
    "ами",
    "ным",
    "ного",
    "ого",
    "его",
    "ому",
    "ему",
    "ый",
    "ой",
    "ых",
    "ий",
    "ая",
    "ое",
    "ые",
    "у",
    "ю",
    "а",
    "я",
    "о",
    "е",
    "ы",
    "и",
)

_RU_RE = re.compile(r"[а-яё]")
_MIN_STEM_LEN = 5
_STEM_MIN_CUT = _MIN_STEM_LEN - 1  # shortest stem left after a cut

# Code-like tokens: "62056-5-3", "32144-2013", dates "2021-12-03".
_CODE_TOKEN_RE = re.compile(r"^\d+(?:[.-]\d+)+$")

# Tokenizer: code tokens first (kept whole), then word runs. \w covers
# latin, cyrillic and digits; the code alternative wins at the same
# position because it is listed first.
_TOKEN_RE = re.compile(r"\d+(?:[.-]\d+)+|\w+")

# Cap on code-prefix expansion: the <= 50 most frequent vocabulary terms
# containing the query code token as substring.
_CODE_EXPANSION_CAP = 50


def is_code_token(token: str) -> bool:
    return _CODE_TOKEN_RE.match(token) is not None


def _normalize(token: str) -> str:
    """One normalization pass: RU suffix removal (longest match wins, one
    per token; token >= 5 letters, stem after the cut >= 4) or EN s/es
    stripping (words >= 4 letters, "ies" -> "y"). Code tokens pass
    through untouched."""
    if is_code_token(token):
        return token
    if _RU_RE.search(token):
        if len(token) >= _MIN_STEM_LEN:
            best = max(
                (suf for suf in _RU_SUFFIXES if token.endswith(suf)),
                key=len,
                default="",
            )
            if best and len(token) - len(best) >= _STEM_MIN_CUT:
                return token[: len(token) - len(best)]
        return token
    # EN: light stemming only.
    if len(token) >= 4:
        if token.endswith("ies"):
            return token[:-3] + "y"
        if token.endswith("es"):
            return token[:-2]
        if token.endswith("s"):
            return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Tokenizer shared by index build and query side: lowercase, code
    tokens kept whole, length > 1, then normalize()."""
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if len(raw) <= 1:
            continue
        out.append(_normalize(raw))
    return out


class LexicalIndex:
    """BM25 postings index over a chunk list.

    postings: term -> [(chunk_idx, tf)]; doc_len/avgdl/df support the
    BM25 formula. Chunk metadata (doc_id, section_path, is_table) is
    captured at build time so search() can pre-filter (Task 7 contract)
    without touching the chunk dicts.
    """

    def __init__(
        self,
        postings: dict[str, list[tuple[int, int]]],
        doc_len: list[int],
        df: dict[str, int],
        chunk_meta: list[dict[str, Any]],
        phrase_haystacks: list[str],
    ) -> None:
        self._postings = postings
        self._doc_len = doc_len
        self._df = df
        self._chunk_meta = chunk_meta
        self._phrase_haystacks = phrase_haystacks
        self._n = len(doc_len)
        self._avgdl = (sum(doc_len) / self._n) if self._n else 0.0
        # Per-query-token cache of code-prefix expansions: code token ->
        # (exact term list, expanded term list with per-term df).
        self._code_expansion_cache: dict[str, tuple[list[str], list[tuple[str, int]]]] = {}

    @classmethod
    def build(
        cls,
        chunks: list[dict[str, Any]],
        metadata: dict[str, dict[str, Any]] | None = None,
    ) -> LexicalIndex:
        """Build the index from chunk dicts.

        metadata: optional chunk_id -> {doc_id, section_path, is_table}
        map (chunk_meta_from_chunks shape). When omitted, metadata is
        derived from the chunk dicts themselves.
        """
        postings: dict[str, list[tuple[int, int]]] = {}
        doc_len: list[int] = []
        df: dict[str, int] = {}
        chunk_meta: list[dict[str, Any]] = []
        phrase_haystacks: list[str] = []

        for i, ch in enumerate(chunks):
            cid = str(ch.get("chunk_id", ""))
            text = str(ch.get("text", ""))
            toks = tokenize(text)
            doc_len.append(len(toks))
            phrase_haystacks.append(text.lower())

            if metadata is not None:
                meta = metadata.get(cid) or {}
                doc_id = str(meta.get("doc_id", "") or cid.split(":")[0])
                sec = str(meta.get("section_path", ""))
                is_table = bool(meta.get("is_table"))
            else:
                doc_id = str(ch.get("doc_id", "") or cid.split(":")[0])
                sec = str(ch.get("section_path", ""))
                is_table = bool(ch.get("is_table"))
            chunk_meta.append(
                {"chunk_id": cid, "doc_id": doc_id, "section_path": sec, "is_table": is_table}
            )

            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            for t, c in tf.items():
                postings.setdefault(t, []).append((i, c))
                df[t] = df.get(t, 0) + 1

        return cls(postings, doc_len, df, chunk_meta, phrase_haystacks)

    # -- filters (Task 7 contract) -----------------------------------------
    # NOTE: filter semantics exist in three intentionally parallel copies:
    # raglib.vector_index._chunk_matches_filters (vector path),
    # retrieval._chunk_passes_filters (legacy/semantic dict path) and
    # LexicalIndex._meta_passes (this index path). Keys and semantics are
    # identical (doc_id equality, doc_ids membership, section_path prefix,
    # is_table bool; doc_id falls back to the chunk_id prefix); a matrix
    # test in tests/test_lexical_index.py pins this copy's behavior to
    # retrieval._chunk_passes_filters. Keep the three in sync.

    @staticmethod
    def _validate_filter_keys(filters: dict[str, Any]) -> None:
        from doc_rag.raglib.vector_index import _supported_filter_keys

        unknown = set(filters) - _supported_filter_keys()
        if unknown:
            raise ValueError(
                f"unsupported filter keys: {sorted(unknown)}; "
                f"supported: {sorted(_supported_filter_keys())}"
            )

    @staticmethod
    def _meta_passes(meta: dict[str, Any], filters: dict[str, Any]) -> bool:
        doc_id = str(meta.get("doc_id", ""))
        if "doc_id" in filters and doc_id != str(filters["doc_id"]):
            return False
        if "doc_ids" in filters:
            want = filters["doc_ids"]
            want_list = (
                [str(d) for d in want] if isinstance(want, (list, tuple, set)) else [str(want)]
            )
            if doc_id not in want_list:
                return False
        if "section_path" in filters:
            sec = str(meta.get("section_path", "") or "")
            if not sec.startswith(str(filters["section_path"])):
                return False
        if "is_table" in filters and bool(meta.get("is_table")) != bool(filters["is_table"]):
            return False
        return True

    def _allowed_indices(self, filters: dict[str, Any]) -> list[int] | None:
        """Pre-filter before scoring: a filter matching nothing must stay
        empty, not degrade to unfiltered results."""
        if not filters:
            return None
        return [i for i, m in enumerate(self._chunk_meta) if self._meta_passes(m, filters)]

    def _code_expansion(self, code_token: str) -> tuple[list[str], list[tuple[str, int]]]:
        """Expand a query code token to vocabulary terms containing it as
        a substring.

        Returns (exact_terms, expanded_terms_with_df): exact terms are the
        verbatim postings hits (usually the token itself), expanded terms
        are the <= _CODE_EXPANSION_CAP most frequent vocabulary matches
        (a single pass over the ~85k-term vocabulary), each carrying its
        df so the caller can compute BM25 idf. Results are cached per
        index instance.
        """
        cached = self._code_expansion_cache.get(code_token)
        if cached is not None:
            return cached

        exact = [t for t in (code_token,) if t in self._postings]
        # Single pass over the vocabulary; vocabulary is ~85k terms so
        # this stays in the tens-of-milliseconds range.
        matches: list[tuple[str, int]] = []
        for term, plist in self._postings.items():
            if code_token in term and term != code_token:
                matches.append((term, len(plist)))
        # Most frequent first: a partial code should prefer terms that
        # actually occur in many chunks over rare split variants.
        matches.sort(key=lambda x: x[1], reverse=True)
        matches = matches[:_CODE_EXPANSION_CAP]

        out = (exact, matches)
        self._code_expansion_cache[code_token] = out
        return out

    # -- search ------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[int, float]]:
        """Score matching chunks, return [(chunk_idx, score)] best-first.

        Filters pre-filter the candidate set (Task 7 semantics, fail loud
        on unknown keys). Scoring: BM25 over postings + normalized exact
        phrase bonus; query code tokens weigh 1.5x. A query with no
        usable tokens (single-character, e.g. "0") falls back to a pure
        phrase scan over the allowed set, matching the old scorer.
        """
        q = (query or "").strip()
        if not q or self._n == 0:
            return []

        if filters:
            self._validate_filter_keys(filters)
        allowed = self._allowed_indices(filters or {})
        if allowed is not None and not allowed:
            return []

        # Query terms: normalized, code tokens kept whole with a 1.5x
        # weight (dedup by normalized term, first weight wins). Code
        # tokens additionally expand to vocabulary terms containing them
        # as a substring ("34.12" -> "34.12-2018"), see module docstring.
        q_weights: dict[str, float] = {}
        for p in _TOKEN_RE.findall(q.lower()):
            if len(p) <= 1:
                continue
            term = _normalize(p)
            w = 1.5 if is_code_token(p) else 1.0
            if term not in q_weights:
                q_weights[term] = w

        # Code-prefix expansion: every code term in the query gets its
        # substring matches from the vocabulary, at the same 1.5x weight.
        expanded: dict[str, int] = {}
        for term in list(q_weights):
            if not is_code_token(term):
                continue
            _exact, matches = self._code_expansion(term)
            for match_term, _df in matches:
                if match_term not in q_weights:
                    expanded.setdefault(match_term, len(q_weights))

        allowed_set = set(allowed) if allowed is not None else None
        q_lower = q.lower()

        if not q_weights and not expanded:
            # Old-scorer fallback: single-character queries scored only
            # via the exact phrase bonus.
            return self._phrase_scan(q_lower, top_k, allowed_set)

        # Accumulate BM25 over postings (O(postings) not O(N*V)). Exact
        # code terms keep their query weight; expanded terms share the
        # weight of the code token that produced them (also 1.5x).
        scores: dict[int, float] = {}
        all_terms: list[tuple[str, float]] = list(q_weights.items())
        all_terms.extend((t, 1.5) for t in expanded)
        for term, qw in all_terms:
            plist = self._postings.get(term)
            if not plist:
                continue
            df = self._df.get(term, 0)
            if df <= 0:
                continue
            idf = math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))
            for idx, tf in plist:
                if allowed_set is not None and idx not in allowed_set:
                    continue
                dl = self._doc_len[idx]
                denom_norm = 1.0 - B + B * (dl / self._avgdl if self._avgdl > 0 else 0.0)
                contrib = qw * idf * (tf * (K1 + 1.0)) / (tf + K1 * denom_norm)
                scores[idx] = scores.get(idx, 0.0) + contrib

        # Exact phrase bonus on candidates only (keeps latency low). Note:
        # with filters the idf of rare terms can dominate, but the phrase
        # bonus is applied AFTER filtering so a filtered run keeps the
        # same relative term weighting as an unfiltered one.
        for idx in scores:
            count = self._phrase_haystacks[idx].count(q_lower)
            if count > 0:
                scores[idx] += min(5.0, 2.0 + 3.0 * count)

        scored = [(idx, s) for idx, s in scores.items() if s > 0.0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(1, top_k)]

    def _phrase_scan(
        self, q_lower: str, top_k: int, allowed_set: set[int] | None
    ) -> list[tuple[int, float]]:
        """Phrase-only scoring pass over the whole allowed set (single-char
        queries carry no tokens; the exact match alone must decide)."""
        scored: list[tuple[int, float]] = []
        for i, hay in enumerate(self._phrase_haystacks):
            if allowed_set is not None and i not in allowed_set:
                continue
            count = hay.count(q_lower)
            if count > 0:
                scored.append((i, min(5.0, 2.0 + 3.0 * count)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(1, top_k)]


# ---------------------------------------------------------------------------
# Singleton cache (mirrors the _CHUNKS_CACHE pattern in retrieval.py)
# ---------------------------------------------------------------------------

_LEXICAL_CACHE: dict[str, Any] = {"path": None, "mtime": None, "index": None}
# Guards the build-and-store critical section of get_lexical_index so
# concurrent first requests build the index once, not twice.
_LEXICAL_LOCK = threading.Lock()


def get_lexical_index(
    chunks: list[dict[str, Any]],
    chunks_path: str | None = None,
    metadata: dict[str, dict[str, Any]] | None = None,
) -> LexicalIndex:
    """Return a LexicalIndex for the corpus at chunks_path.

    Cache key is (path, mtime) like _CHUNKS_CACHE: a rewritten
    chunks.jsonl invalidates the index on the next call. Without a path
    (synthetic chunk lists in tests), builds fresh every time.

    IMPORTANT: metadata-filtered builds are NEVER cached — the singleton
    has no key dimension for the metadata map, so caching a build tied to
    one metadata snapshot would serve stale filter semantics to other
    callers. Passing metadata != None always bypasses the cache and never
    populates it.
    """
    if metadata is not None:
        # Metadata-tied build: cache would leak this map to other callers.
        return LexicalIndex.build(chunks, metadata)

    if not chunks_path:
        return LexicalIndex.build(chunks)

    try:
        mt = os.stat(chunks_path).st_mtime
    except Exception:
        mt = None

    # Double-checked locking (chunking_meta.py style): a first hit is a
    # cheap unlocked read; only the build-and-store critical section is
    # serialized so two concurrent first requests do not each build a
    # ~150 MB index.
    if (
        _LEXICAL_CACHE.get("path") == chunks_path
        and _LEXICAL_CACHE.get("mtime") == mt
        and isinstance(_LEXICAL_CACHE.get("index"), LexicalIndex)
    ):
        return _LEXICAL_CACHE["index"]

    with _LEXICAL_LOCK:
        # Re-check inside the lock: another thread may have finished the
        # build while this one waited.
        if (
            _LEXICAL_CACHE.get("path") == chunks_path
            and _LEXICAL_CACHE.get("mtime") == mt
            and isinstance(_LEXICAL_CACHE.get("index"), LexicalIndex)
        ):
            return _LEXICAL_CACHE["index"]
        idx = LexicalIndex.build(chunks)
        if mt is not None:
            _LEXICAL_CACHE["path"] = chunks_path
            _LEXICAL_CACHE["mtime"] = mt
            _LEXICAL_CACHE["index"] = idx
    return idx


def reset_lexical_cache() -> None:
    """Drop the singleton (used by invalidate_index_cache and tests)."""
    _LEXICAL_CACHE.update({"path": None, "mtime": None, "index": None})
