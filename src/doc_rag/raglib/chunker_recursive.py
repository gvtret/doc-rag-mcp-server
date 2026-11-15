"""Structure-aware recursive chunker.

Walks typed blocks from the parser, grouping siblings into chunks while
respecting headings as hard boundaries and tables as atomic units.

The fixed-size splitter (`_chunk_text`) works on raw Markdown text and
has no awareness of document structure. This module uses the typed
`Block` layer (v1.5+) to produce chunks that preserve section context
and keep tables intact.

Chunk sizing uses the real bge-m3 tokenizer (`count_tokens`) rather than
the len//4 estimate: Cyrillic text tokenizes at roughly 2.5-3 chars per
token, so the old estimate undershot by ~25% and pushed chunk tails past
the model's 512-token window. A hard cap (CHUNK_MAX_TOKENS = 504) is
enforced by a final safety pass over every emitted chunk, tables
included; T12 refines how oversized tables are split along row lines.
"""

from __future__ import annotations

import logging
from typing import Any

from doc_rag.raglib.blocks import Block
from doc_rag.raglib.chunking_meta import CHUNK_MAX_TOKENS, count_tokens

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """Real bge-m3 token count (chunk sizing unit).

    Kept as a thin wrapper: the name is used across the module and in
    tests, but the semantics changed from len//4 to tokenizer-accurate.
    """
    return count_tokens(text)


class ChunkNode:
    """Internal tree node for the recursive splitter."""

    __slots__ = ("blocks", "heading_level", "heading_text", "children")

    def __init__(
        self,
        blocks: list[Block] | None = None,
        heading_level: int = 0,
        heading_text: str = "",
    ) -> None:
        self.blocks: list[Block] = blocks or []
        self.heading_level = heading_level
        self.heading_text = heading_text
        self.children: list[ChunkNode] = []


def _build_tree(blocks: list[Block]) -> list[ChunkNode]:
    """Build a tree of ChunkNodes from a flat block list.

    Headings create new child nodes at their level. Blocks between
    headings accumulate in the current node.
    """
    root = ChunkNode()
    stack: list[ChunkNode] = [root]
    current = root

    for block in blocks:
        if block.type == "heading" and block.level is not None:
            level = block.level
            # Pop back to the right parent level
            while len(stack) > 1 and stack[-1].heading_level >= level:
                stack.pop()
            parent = stack[-1]
            child = ChunkNode(heading_level=level, heading_text=block.text)
            parent.children.append(child)
            stack.append(child)
            current = child
            current.blocks.append(block)
        else:
            current.blocks.append(block)

    return root.children if root.children else [root]


def _count_node_tokens(node: ChunkNode) -> int:
    total = 0
    for b in node.blocks:
        total += _estimate_tokens(b.text)
    for c in node.children:
        total += _count_node_tokens(c)
    return total


def _section_path(node: ChunkNode, ancestors: list[str] | None = None) -> str:
    """Build the dotted section path from root to this node."""
    if ancestors is None:
        ancestors = []
    parts = [a for a in ancestors if a]
    if node.heading_text:
        parts.append(node.heading_text)
    return ".".join(parts) if parts else ""


def _render_table_summary(table: Block) -> str:
    """Create a short text summary of a table block for retrieval."""
    lines = [line.strip() for line in table.text.split("\n") if line.strip()]
    if not lines:
        return f"Таблица ({table.block_id})"
    # First line is usually headers
    header = lines[0] if lines else ""
    cell_count = sum(line.count("|") for line in lines)
    return f"Таблица ({len(lines)} строк, {cell_count} ячеек): {header[:200]}"


def _split_text_token_budget(text: str, cap: int) -> list[str]:
    """Split `text` into pieces each within `cap` real tokens.

    Strategy: prefer breaking at the largest \\n\\n boundary whose prefix
    fits the cap; if a piece has no usable boundary, hard-split at the
    char offset that keeps the prefix under the cap (found by binary
    search on count_tokens). Recurses until every piece fits.
    """
    if _estimate_tokens(text) <= cap:
        return [text]
    # Longest prefix boundary under the cap
    best = -1
    pos = 0
    while True:
        idx = text.find("\n\n", pos)
        if idx == -1:
            break
        if _estimate_tokens(text[:idx]) <= cap:
            best = max(best, idx)
            pos = idx + 1
        else:
            break
    if best > 0:
        head = text[:best].rstrip("\n")
        tail = text[best:].lstrip("\n")
        parts = [p for p in [head, tail] if p.strip()]
        if len(parts) == 2:
            return _split_text_token_budget(parts[0], cap) + _split_text_token_budget(parts[1], cap)
    # No \\n\\n boundary fits -- hard split by binary search on char offset
    n = len(text)
    lo, hi = 1, n
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _estimate_tokens(text[:mid]) <= cap:
            lo = mid
        else:
            hi = mid - 1
    head = text[:lo].rstrip()
    tail = text[lo:].lstrip()
    if not head.strip():
        # Degenerate guard: a single token larger than the whole prefix
        head, tail = text[:1], text[1:]
    parts = [p for p in [head, tail] if p.strip()]
    if len(parts) == 2:
        return _split_text_token_budget(parts[0], cap) + _split_text_token_budget(parts[1], cap)
    return [text]


def _enforce_cap(text: str, cap: int) -> list[str]:
    """Final safety pass: guarantee a chunk text fits the hard cap."""
    if not text.strip():
        return []
    if _estimate_tokens(text) <= cap:
        return [text]
    return _split_text_token_budget(text, cap)


def _chunk_node(
    node: ChunkNode,
    target_tokens: int,
    ancestors: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Recursively produce chunk dicts from a ChunkNode tree.

    The grouping budget is min(target_tokens, CHUNK_MAX_TOKENS): groups
    are merged with real-token counts and must never be assembled above
    the hard cap, otherwise the final safety pass would have to re-split
    what grouping just built.
    """
    budget = min(target_tokens, CHUNK_MAX_TOKENS)
    path = _section_path(node, ancestors)
    chunks: list[dict[str, Any]] = []

    # First, recurse into children (sub-headings)
    for child in node.children:
        parent_path = (
            ancestors + [node.heading_text]
            if ancestors is not None
            else ([node.heading_text] if node.heading_text else [])
        )
        child_chunks = _chunk_node(child, target_tokens, parent_path)
        chunks.extend(child_chunks)

    # Now process this node's own blocks (not in children)
    # Separate blocks into groups: tables are atomic, others are mergeable
    groups: list[list[Block]] = []
    current_group: list[Block] = []

    for block in node.blocks:
        if block.type == "heading" and block.level is not None:
            # Headings are handled by the tree structure, skip here
            continue
        if block.type == "table":
            # Flush current group, then add table as its own atomic group
            if current_group:
                groups.append(current_group)
                current_group = []
            groups.append([block])
        else:
            current_group.append(block)
    if current_group:
        groups.append(current_group)

    # Merge groups into chunks respecting target_tokens
    pending: list[Block] = []
    pending_tokens = 0

    def _flush_pending() -> None:
        if not pending:
            return
        text = "\n\n".join(b.text.strip() for b in pending if b.text.strip())
        if text:
            chunk: dict[str, Any] = {
                "text": text,
                "section_path": path,
            }
            chunks.append(chunk)
        pending.clear()

    for group in groups:
        group_tokens = sum(_estimate_tokens(b.text) for b in group)

        if len(group) == 1 and group[0].type == "table":
            # Atomic table -- oversized ones are split by the final cap
            # pass in chunk_recursive (T12 will split them along rows).
            _flush_pending()
            table_block = group[0]
            # Main table chunk
            chunks.append(
                {
                    "text": table_block.text.strip(),
                    "section_path": path,
                    "is_table": True,
                }
            )
            # Table summary sibling for retrieval
            summary_text = _render_table_summary(table_block)
            if summary_text != table_block.text.strip():
                chunks.append(
                    {
                        "text": summary_text,
                        "section_path": path,
                        "is_table_summary": True,
                    }
                )
        elif pending_tokens + group_tokens <= budget:
            pending.extend(group)
            pending_tokens += group_tokens
        else:
            _flush_pending()
            # If single group is larger than the budget, split it
            if group_tokens > budget and len(group) > 1:
                # Split the group at roughly the midpoint
                mid = len(group) // 2
                pending.extend(group[:mid])
                pending_tokens += sum(_estimate_tokens(b.text) for b in group[:mid])
                _flush_pending()
                pending.extend(group[mid:])
                pending_tokens = sum(_estimate_tokens(b.text) for b in pending)
            elif group_tokens > budget and len(group) == 1:
                # Single oversized block -- split its text by paragraph breaks
                block = group[0]
                paragraphs = block.text.split("\n\n")
                if len(paragraphs) > 1:
                    for para in paragraphs:
                        para_tokens = _estimate_tokens(para)
                        if pending_tokens + para_tokens <= budget:
                            pending.append(
                                Block(
                                    block_id=block.block_id,
                                    doc_id=block.doc_id,
                                    type=block.type,
                                    text=para,
                                    source_backend=block.source_backend,
                                    level=block.level,
                                    page=block.page,
                                )
                            )
                            pending_tokens += para_tokens
                        else:
                            _flush_pending()
                            pending.append(
                                Block(
                                    block_id=block.block_id,
                                    doc_id=block.doc_id,
                                    type=block.type,
                                    text=para,
                                    source_backend=block.source_backend,
                                    level=block.level,
                                    page=block.page,
                                )
                            )
                            pending_tokens = _estimate_tokens(para)
                else:
                    # No paragraph breaks -- hard split at the token budget
                    block = group[0]
                    for segment in _split_text_token_budget(block.text, budget):
                        segment = segment.strip()
                        if not segment:
                            continue
                        _flush_pending()
                        pending.append(
                            Block(
                                block_id=block.block_id,
                                doc_id=block.doc_id,
                                type=block.type,
                                text=segment,
                                source_backend=block.source_backend,
                                level=block.level,
                                page=block.page,
                            )
                        )
                        pending_tokens = _estimate_tokens(segment)
            else:
                pending.extend(group)
                pending_tokens += group_tokens

    _flush_pending()
    return chunks


def chunk_recursive(
    blocks: list[Block],
    target_tokens: int = 512,
) -> list[dict[str, Any]]:
    """Produce chunks from typed blocks using structure-aware splitting.

    Chunk sizes are measured in real bge-m3 tokens; every emitted chunk
    (tables included) is guaranteed to be at most CHUNK_MAX_TOKENS real
    tokens via a final safety pass.

    Args:
        blocks: Ordered list of Block objects from the parser.
        target_tokens: Target chunk size in real tokens (soft grouping
            budget; the hard cap CHUNK_MAX_TOKENS is enforced separately).

    Returns:
        List of chunk dicts with keys: text, section_path, optionally
        is_table / is_table_summary / is_table_fragment.
    """
    if not blocks:
        return []

    tree = _build_tree(blocks)
    chunks: list[dict[str, Any]] = []

    for root_node in tree:
        root_chunks = _chunk_node(root_node, target_tokens)
        chunks.extend(root_chunks)

    # Final safety pass: no chunk may exceed the hard cap, tables included.
    # Grouping above already uses real-token budgets, so this rarely
    # splits; it guarantees the invariant for tables and edge cases.
    capped: list[dict[str, Any]] = []
    for c in chunks:
        pieces = _enforce_cap(c["text"], CHUNK_MAX_TOKENS)
        if not pieces:
            continue
        if len(pieces) == 1 and pieces[0] == c["text"]:
            capped.append(c)
            continue
        is_table = bool(c.get("is_table"))
        for piece in pieces:
            replacement = dict(c)
            replacement["text"] = piece
            if is_table:
                # Fragment of a cap-split table: keep is_table semantics
                # (the piece still carries table text) and flag it so T12
                # can distinguish whole tables from split fragments.
                replacement["is_table_fragment"] = True
            capped.append(replacement)
    if len(capped) != len(chunks):
        logger.info(
            "chunk_recursive: hard cap %d tokens split %d chunk(s)",
            CHUNK_MAX_TOKENS,
            len(capped) - len(chunks),
        )
    chunks = capped

    # Assign sequential IDs
    for i, c in enumerate(chunks):
        c["chunk_idx"] = i

    logger.info(
        "chunk_recursive: %d blocks -> %d chunks (target=%d tokens, cap=%d)",
        len(blocks),
        len(chunks),
        target_tokens,
        CHUNK_MAX_TOKENS,
    )
    return chunks
