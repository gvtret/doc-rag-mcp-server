# `build/blocks/<doc_id>.jsonl` — schema v1

`blocks.jsonl` is the typed intermediate layer introduced in v1.5. Every
parser backend (PyMuPDF, python-docx, antiword, Docling, Unstructured…)
emits a list of `Block` records on output, and every downstream step —
Markdown derivation, chunk splitting, quality checks (v1.7), recursive
chunker (v1.9) — consumes the same shape.

Until v1.5 the intermediate layer was an unstructured Markdown string;
the new typed layer preserves heading hierarchy, table structure,
positional metadata, and per-block confidence so that later releases
can do better chunking and quality reporting without having to re-parse
documents.

## File format

- One file per document at `build/blocks/<doc_id>.jsonl`.
- One JSON object per line (JSON Lines / NDJSON), UTF-8.
- File ends with a newline.
- Order is **reading order**, not necessarily source order — parsers
  that know layout (Docling, PyMuPDF multi-column) reorder; parsers
  that don't (antiword) emit source order.

The version number is **`schema_version: 1`** and is stored in
`build/manifest.json` under the top-level field
`blocks_schema_version`. Individual block records do **not** carry the
version — it is one number per build, not per row.

## Block record — fields

| Field | Type | Required | Description |
|---|---|---|---|
| `block_id` | string | yes | Stable identifier within the document. Recommended form: `<doc_id>:<index>` where `<index>` is a zero-padded sequence number (`doc-aaa:0000`). |
| `doc_id` | string | yes | Parent document's id. Same as the `doc_id` in `manifest.json`. |
| `type` | string | yes | One of: `heading`, `paragraph`, `list_item`, `table`, `formula`, `figure`, `code`, `quote`, `other`. See "Type semantics" below. |
| `text` | string | yes | The block's text content. For `table`, a human-readable rendering. For `figure`, the caption (may be empty). For `formula`, the LaTeX source if available, otherwise the raw text. |
| `level` | integer \| null | no | For `heading`, the heading level (1 = H1, 6 = H6). For `list_item`, the nesting depth (0 = top-level). Null for other types. |
| `page` | integer \| null | no | Source page (1-based). Null for formats without a page concept (`.md`, `.txt`, `.docx` that the parser cannot map to pages). |
| `bbox` | [number, number, number, number] \| null | no | `[x0, y0, x1, y1]` in PDF coordinate units (default: points). Null for non-PDF formats. |
| `source_backend` | string | yes | Which parser produced this block. One of: `pymupdf`, `pypdf2`, `python-docx`, `antiword`, `catdoc`, `docling`, `unstructured`, `direct` (for `.md` / `.txt`). |
| `confidence` | number | no | `0.0`–`1.0`. ML-based backends (Docling layout, OCR) report their model confidence; non-ML backends omit the field (which is interpreted as `1.0`). |
| `metadata` | object | no | Free-form key/value pairs for backend-specific extras. See "Metadata conventions" below. |

Unknown fields are tolerated on read — older builds reading newer files
must ignore fields they don't recognise, not raise. New required
fields trigger a `schema_version` bump.

## Type semantics

| Type | What it represents |
|---|---|
| `heading` | A section title. `level` indicates the depth (1 = top-level). |
| `paragraph` | A run of body text (the most common type). |
| `list_item` | One item in a bulleted or numbered list. `level` indicates nesting depth. The marker (`-`, `1.`, `a)`) is **not** part of `text`. |
| `table` | A tabular block. `text` is a human-readable rendering (e.g. pipe-separated rows for Markdown derivation). The structured cell grid lives in `metadata.cells` when the backend can provide it. |
| `formula` | A mathematical formula. `text` carries LaTeX when the backend can produce it (Docling), otherwise the raw recognised characters. |
| `figure` | An image, diagram, or chart. `text` carries the caption (which may be empty). Image bytes are not embedded; if the backend stores them on disk, the path goes in `metadata.image_path`. |
| `code` | A code listing or pre-formatted text. |
| `quote` | A blockquote. |
| `other` | Anything the parser can extract as text but cannot classify. Avoid emitting this; prefer one of the specific types. |

## Metadata conventions

`metadata` is intentionally open, but a few keys are standardised so
downstream code can rely on them:

- `metadata.cells` — for `type=table`, a list of rows where each row is
  a list of cell strings. Optional.
- `metadata.image_path` — for `type=figure`, a path (relative to the
  repo root) where the image bytes live, if the backend chose to
  persist them. Optional.
- `metadata.heading_path` — for any block, the breadcrumb of enclosing
  headings (e.g. `["1. Introduction", "1.2 Scope"]`). Set by the
  derivation step, not by the parser. Used by the v1.9 recursive
  chunker.
- `metadata.original_index` — original index within the parser's
  raw output, useful for debugging reading-order issues.

Backends are free to add their own keys (`metadata.docling_label`,
`metadata.tableformer_score`, etc.) — they will not collide with
standardised keys as long as they avoid the names above.

## Example

For a small DOCX with a heading, a paragraph and a 2×2 table:

```json
{"block_id":"doc-aaa:0000","doc_id":"doc-aaa","type":"heading","text":"Section 1","level":1,"page":1,"bbox":null,"source_backend":"python-docx"}
{"block_id":"doc-aaa:0001","doc_id":"doc-aaa","type":"paragraph","text":"Lead paragraph.","page":1,"bbox":null,"source_backend":"python-docx"}
{"block_id":"doc-aaa:0002","doc_id":"doc-aaa","type":"table","text":"A1 | B1\nA2 | B2","page":1,"bbox":null,"source_backend":"python-docx","metadata":{"cells":[["A1","B1"],["A2","B2"]]}}
```

## Versioning

This schema is SemVer-protected — bumps follow the same rules as
`manifest.schema_version`:

- **MAJOR (schema_version bump):** adding a required field, removing a
  field, changing the meaning of a field's value. Older builds refuse
  to read newer files (the same `ManifestSchemaTooNew` mechanism is
  reused, parameterised on `blocks_schema_version`).
- **MINOR (no bump):** adding an optional field, adding a new value to
  the `type` enum, adding a `metadata.*` convention.
- **PATCH (no bump):** clarifying docs.

Future schema versions are tracked in `docs/schemas/blocks-vN.md`
files; this one is `v1`.
