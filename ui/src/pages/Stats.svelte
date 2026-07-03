<script lang="ts">
  import { t } from "../lib/i18n";

  interface DocStat {
    doc_id: string;
    filename: string;
    source_file: string;
    size_bytes: number;
    size_mb: number;
    chunk_count: number;
    quality_score: number;
    quality_warnings: number;
    pages: number;
    blocks: Record<string, number>;
    total_blocks: number;
    edition_year: number | null;
    coverage: any;
  }

  let docs = $state<DocStat[]>([]);
  let loading = $state(true);
  let sortBy = $state<"filename" | "score" | "chunks" | "size">("filename");
  let sortOrder = $state<"asc" | "desc">("asc");
  let selectedDoc = $state<DocStat | null>(null);

  async function loadStats() {
    try {
      const r = await fetch("/ui/doc-stats", { credentials: "same-origin" });
      const d = await r.json();
      if (d.ok) {
        docs = d.documents;
      }
    } catch {}
    loading = false;
  }

  $effect(() => {
    loadStats();
  });

  function sortedDocs() {
    const sorted = [...docs];
    sorted.sort((a, b) => {
      let va: any, vb: any;
      switch (sortBy) {
        case "score": va = a.quality_score; vb = b.quality_score; break;
        case "chunks": va = a.chunk_count; vb = b.chunk_count; break;
        case "size": va = a.size_bytes; vb = b.size_bytes; break;
        default: va = a.filename; vb = b.filename;
      }
      if (typeof va === "string") return sortOrder === "asc" ? va.localeCompare(vb) : vb.localeCompare(va);
      return sortOrder === "asc" ? va - vb : vb - va;
    });
    return sorted;
  }

  function toggleSort(key: typeof sortBy) {
    if (sortBy === key) {
      sortOrder = sortOrder === "asc" ? "desc" : "asc";
    } else {
      sortBy = key;
      sortOrder = "asc";
    }
  }

  function qualityColor(score: number): string {
    if (score >= 0.8) return "ok";
    if (score >= 0.5) return "warn";
    return "err";
  }

  function formatSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  }

  function totalChunks() { return docs.reduce((s, d) => s + d.chunk_count, 0); }
  function totalBlocks() { return docs.reduce((s, d) => s + d.total_blocks, 0); }
  function totalSize() { return docs.reduce((s, d) => s + d.size_bytes, 0); }
  function avgScore() { return docs.length ? (docs.reduce((s, d) => s + d.quality_score, 0) / docs.length) : 0; }
</script>

<section>
  <h2>{t("sidebar.stats")}</h2>

  {#if loading}
    <p class="muted">{t("documents.loading")}</p>
  {:else}
    <div class="summary">
      <span>Документов: <strong>{docs.length}</strong></span>
      <span class="sep">·</span>
      <span>Чанков: <strong>{totalChunks()}</strong></span>
      <span class="sep">·</span>
      <span>Блоков: <strong>{totalBlocks()}</strong></span>
      <span class="sep">·</span>
      <span>Размер: <strong>{formatSize(totalSize())}</strong></span>
      <span class="sep">·</span>
      <span>Средний score: <strong class={avgScore() >= 0.8 ? "ok" : "warn"}>{avgScore().toFixed(2)}</strong></span>
    </div>

    <table class="stats-table">
      <thead>
        <tr>
          <th class="sortable" onclick={() => toggleSort("filename")}>
            Файл {sortBy === "filename" ? (sortOrder === "asc" ? "↑" : "↓") : ""}
          </th>
          <th class="num sortable" onclick={() => toggleSort("size")}>
            Размер {sortBy === "size" ? (sortOrder === "asc" ? "↑" : "↓") : ""}
          </th>
          <th class="num sortable" onclick={() => toggleSort("chunks")}>
            Чанки {sortBy === "chunks" ? (sortOrder === "asc" ? "↑" : "↓") : ""}
          </th>
          <th class="num">Блоки</th>
          <th class="num">Стр.</th>
          <th class="sortable" onclick={() => toggleSort("score")}>
            Score {sortBy === "score" ? (sortOrder === "asc" ? "↑" : "↓") : ""}
          </th>
          <th>Warnings</th>
          <th>Блоки по типам</th>
        </tr>
      </thead>
      <tbody>
        {#each sortedDocs() as d (d.doc_id)}
          <tr class="clickable" onclick={() => selectedDoc = selectedDoc?.doc_id === d.doc_id ? null : d}>
            <td title={d.source_file}>{d.filename}</td>
            <td class="num">{formatSize(d.size_bytes)}</td>
            <td class="num">{d.chunk_count}</td>
            <td class="num">{d.total_blocks}</td>
            <td class="num">{d.pages || "—"}</td>
            <td>
              <span class="q-badge q-{qualityColor(d.quality_score)}">
                {d.quality_score.toFixed(2)}
              </span>
            </td>
            <td class="num">{d.quality_warnings}</td>
            <td class="blocks-cell">
              {#each Object.entries(d.blocks) as [type, count]}
                {#if count > 0}
                  <span class="block-tag">{type}: {count}</span>
                {/if}
              {/each}
            </td>
          </tr>
          {#if selectedDoc?.doc_id === d.doc_id}
            <tr class="detail-row">
              <td colspan="8">
                <div class="detail-box">
                  <div><strong>doc_id:</strong> {d.doc_id}</div>
                  <div><strong>sha256:</strong> <code>{/* d.sha256 would need to be in API */ ""}</code></div>
                  <div><strong>Путь:</strong> {d.source_file}</div>
                  <div><strong>Размер:</strong> {formatSize(d.size_bytes)}</div>
                  <div><strong>Чанков:</strong> {d.chunk_count}</div>
                  <div><strong>Блоков:</strong> {d.total_blocks} ({Object.entries(d.blocks).map(([k,v]) => `${k}:${v}`).join(", ")})</div>
                  <div><strong>Страниц:</strong> {d.pages || "—"}</div>
                  <div><strong>Score:</strong> {d.quality_score.toFixed(2)} ({d.quality_warnings} warnings)</div>
                </div>
              </td>
            </tr>
          {/if}
        {/each}
      </tbody>
    </table>
  {/if}
</section>

<style>
  section { padding: 16px 24px; }
  h2 { margin: 0 0 12px; font-size: 1rem; }
  .summary {
    display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap;
    color: var(--text-secondary); font-size: 0.85rem;
  }
  .summary .sep { color: var(--text-faint); }
  .summary .ok { color: var(--accent-ok); }
  .summary .warn { color: var(--accent-warn); }
  .stats-table {
    width: 100%; border-collapse: collapse; background: var(--bg-elevated);
    border: 1px solid var(--border-subtle); font-size: 0.82rem;
  }
  .stats-table th {
    padding: 6px 8px; text-align: left; border-bottom: 1px solid var(--border-strong);
    color: var(--text-faint); font-family: ui-monospace, monospace; font-size: 0.72rem;
    letter-spacing: 0.04em;
  }
  .stats-table td { padding: 5px 8px; border-bottom: 1px solid var(--border-subtle); color: var(--text-secondary); }
  .stats-table tbody tr:hover td { background: var(--bg-base); color: var(--text-primary); }
  .stats-table .clickable { cursor: pointer; }
  .stats-table .num { text-align: right; font-variant-numeric: tabular-nums; }
  .sortable { cursor: pointer; user-select: none; }
  .sortable:hover { color: var(--text-primary); }
  .blocks-cell { display: flex; flex-wrap: wrap; gap: 3px; }
  .block-tag {
    display: inline-block; padding: 1px 4px; font-size: 0.7rem;
    background: var(--bg-base); border-radius: 2px; color: var(--text-muted);
    font-family: ui-monospace, monospace;
  }
  .detail-row td { padding: 0; }
  .detail-box {
    padding: 12px 16px; background: var(--bg-base); border-top: 1px solid var(--border-subtle);
    font-size: 0.82rem; line-height: 1.6;
  }
  .detail-box code { font-size: 0.75rem; word-break: break-all; }
  .q-badge {
    display: inline-block; padding: 1px 6px; font-size: 10px; font-weight: 600;
    border-radius: 2px; font-family: ui-monospace, monospace;
  }
  .q-ok { color: var(--bg-base); background: var(--accent-ok); }
  .q-warn { color: var(--bg-base); background: var(--accent-warn); }
  .q-err { color: #fff; background: var(--accent-error); }
  .muted { color: var(--text-muted); }
</style>
