"""Structure-aware recursive chunker.

Walks typed blocks from the parser, grouping siblings into chunks while
respecting headings as hard boundaries and tables as atomic units.

The fixed-size splitter (`_chunk_text`) works on raw Markdown text and
has no awareness of document structure. This module uses the typed
`Block` layer (v1.5+) to produce chunks that preserve section context
and keep tables intact.
"""

from __future__ import annotations

import logging
from typing import Any

from doc_rag.raglib.blocks import Block

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


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


def _count_tokens(node: ChunkNode) -> int:
    total = 0
    for b in node.blocks:
        total += _estimate_tokens(b.text)
    for c in node.children:
        total += _count_tokens(c)
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


def _chunk_node(
    node: ChunkNode,
    target_tokens: int,
    ancestors: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Recursively produce chunk dicts from a ChunkNode tree."""
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
            # Atomic table
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
        elif pending_tokens + group_tokens <= target_tokens:
            pending.extend(group)
            pending_tokens += group_tokens
        else:
            _flush_pending()
            # If single group is larger than target, split it
            if group_tokens > target_tokens and len(group) > 1:
                # Split the group at roughly the midpoint
                mid = len(group) // 2
                pending.extend(group[:mid])
                pending_tokens += sum(_estimate_tokens(b.text) for b in group[:mid])
                _flush_pending()
                pending.extend(group[mid:])
                pending_tokens = sum(_estimate_tokens(b.text) for b in pending)
            elif group_tokens > target_tokens and len(group) == 1:
                # Single oversized block — split its text by paragraph breaks
                block = group[0]
                paragraphs = block.text.split("\n\n")
                if len(paragraphs) > 1:
                    for para in paragraphs:
                        para_tokens = _estimate_tokens(para)
                        if pending_tokens + para_tokens <= target_tokens:
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
                    # No paragraph breaks — hard split by character count
                    text = block.text
                    window = target_tokens * CHARS_PER_TOKEN
                    for i in range(0, len(text), window):
                        segment = text[i : i + window].strip()
                        if segment:
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

    Args:
        blocks: Ordered list of Block objects from the parser.
        target_tokens: Target chunk size in tokens (~4 chars each).

    Returns:
        List of chunk dicts with keys: text, section_path, optionally
        is_table or is_table_summary.
    """
    if not blocks:
        return []

    tree = _build_tree(blocks)
    chunks: list[dict[str, Any]] = []

    for root_node in tree:
        root_chunks = _chunk_node(root_node, target_tokens)
        chunks.extend(root_chunks)

    # Assign sequential IDs
    for i, c in enumerate(chunks):
        c["chunk_idx"] = i

    logger.info(
        "chunk_recursive: %d blocks → %d chunks (target=%d tokens)",
        len(blocks),
        len(chunks),
        target_tokens,
    )
    return chunks
