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

  const BLOCK_TYPES = [
    { key: "heading",     label: "Заголовки",      color: "#6b9bd2", desc: "Названия разделов и подразделов" },
    { key: "paragraph",   label: "Абзацы текста",  color: "#8bab6e", desc: "Основной текст документа" },
    { key: "table",       label: "Таблицы",         color: "#d4a76a", desc: "Структурированные данные в строках и столбцах" },
    { key: "list_item",   label: "Списки",          color: "#b07cc4", desc: "Нумерованные и маркированные списки" },
    { key: "figure",      label: "Рисунки",         color: "#d47a7a", desc: "Изображения, схемы, графики" },
    { key: "code",        label: "Код",              color: "#7ac4c4", desc: "Программный код и технические примеры" },
    { key: "formula",     label: "Формулы",         color: "#c4b47a", desc: "Математические и инженерные формулы" },
    { key: "page_footer", label: "Колонтитулы",     color: "#999",    desc: "Номера страниц, шапки/подвалы" },
  ];

  const BLOCK_COLOR_MAP: Record<string, string> = {};
  const BLOCK_LABEL_MAP: Record<string, string> = {};
  for (const bt of BLOCK_TYPES) {
    BLOCK_COLOR_MAP[bt.key] = bt.color;
    BLOCK_LABEL_MAP[bt.key] = bt.label;
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

  function blockBarSegments(blocks: Record<string, number>): { color: string; pct: number; label: string; count: number }[] {
    const total = Object.values(blocks).reduce((s, v) => s + v, 0);
    if (total === 0) return [];
    const segments: { color: string; pct: number; label: string; count: number }[] = [];
    for (const bt of BLOCK_TYPES) {
      const count = blocks[bt.key] || 0;
      if (count > 0) {
        segments.push({ color: bt.color, pct: (count / total) * 100, label: bt.label, count });
      }
    }
    for (const [key, count] of Object.entries(blocks)) {
      if (count > 0 && !BLOCK_TYPES.find(bt => bt.key === key)) {
        segments.push({ color: "#666", pct: (count / total) * 100, label: key, count });
      }
    }
    return segments;
  }

  // Global block type distribution for summary
  let globalBlockDist = $derived(() => {
    const dist: Record<string, number> = {};
    for (const d of docs) {
      for (const [k, v] of Object.entries(d.blocks)) {
        dist[k] = (dist[k] || 0) + v;
      }
    }
    return dist;
  });
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

    <!-- Legend -->
    <div class="legend">
      {#each BLOCK_TYPES as bt}
        {@const gDist = globalBlockDist()}
        {#if (gDist[bt.key] || 0) > 0}
          <span class="legend-item">
            <span class="legend-dot" style="background:{bt.color}"></span>
            <span class="legend-label" title={bt.desc}>{bt.label}</span>
          </span>
        {/if}
      {/each}
    </div>

    <table class="table stats-table">
      <thead>
        <tr>
          <th class="sortable" title="Название файла" onclick={() => toggleSort("filename")}>
            Файл {sortBy === "filename" ? (sortOrder === "asc" ? "↑" : "↓") : ""}
          </th>
          <th class="num sortable" title="Размер исходного файла" onclick={() => toggleSort("size")}>
            Размер {sortBy === "size" ? (sortOrder === "asc" ? "↑" : "↓") : ""}
          </th>
          <th class="num sortable" title="Количество текстовых блоков (чанков) в индексе" onclick={() => toggleSort("chunks")}>
            Чанки {sortBy === "chunks" ? (sortOrder === "asc" ? "↑" : "↓") : ""}
          </th>
          <th class="num" title="Общее количество структурных блоков (абзацы, заголовки, таблицы...)">Блоки</th>
          <th class="num" title="Количество страниц в документе">Стр.</th>
          <th class="sortable" title="Показатель качества парсинга от 0 до 1. Зелёный ≥0.8, жёлтый ≥0.5, красный <0.5" onclick={() => toggleSort("score")}>
            Score {sortBy === "score" ? (sortOrder === "asc" ? "↑" : "↓") : ""}
          </th>
          <th title="Количество предупреждений при парсинге">Проблемы</th>
          <th class="blocks-col" title="Визуальное распределение блоков по типам (заголовки, абзацы, таблицы...)">Состав блоков</th>
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
              <span class="badge {qualityColor(d.quality_score)}">
                {d.quality_score.toFixed(2)}
              </span>
            </td>
            <td class="num">{d.quality_warnings || "—"}</td>
            <td class="blocks-cell">
              <div class="block-bar" title={Object.entries(d.blocks).filter(([,v]) => v > 0).map(([k,v]) => `${BLOCK_LABEL_MAP[k] || k}: ${v}`).join(", ")}>
                {#each blockBarSegments(d.blocks) as seg}
                  <div
                    class="block-segment"
                    style="width:{seg.pct}%;background:{seg.color}"
                    title="{seg.label}: {seg.count} ({seg.pct.toFixed(0)}%)"
                  ></div>
                {/each}
              </div>
              <span class="blocks-total">{d.total_blocks}</span>
            </td>
          </tr>
          {#if selectedDoc?.doc_id === d.doc_id}
            <tr class="detail-row">
              <td colspan="8">
                <div class="detail-box">
                  <div class="detail-meta">
                    <div><strong>doc_id:</strong> {d.doc_id}</div>
                    <div><strong>Исходный файл:</strong> {d.source_file}</div>
                    <div><strong>Размер:</strong> {formatSize(d.size_bytes)} · <strong>Страниц:</strong> {d.pages || "—"} · <strong>Чанков:</strong> {d.chunk_count}</div>
                    <div><strong>Score:</strong> {d.quality_score.toFixed(2)} · <strong>Предупреждений:</strong> {d.quality_warnings || 0}</div>
                  </div>

                  <div class="detail-blocks-title">Состав документа по типам блоков:</div>
                  <div class="detail-bars">
                    {#each BLOCK_TYPES as bt}
                      {@const count = d.blocks[bt.key] || 0}
                      {#if count > 0}
                        {@const pct = d.total_blocks > 0 ? (count / d.total_blocks) * 100 : 0}
                        <div class="detail-bar-row">
                          <span class="detail-bar-label">{bt.label}</span>
                          <div class="detail-bar-track">
                            <div class="detail-bar-fill" style="width:{pct}%;background:{bt.color}"></div>
                          </div>
                          <span class="detail-bar-count">{count}</span>
                          <span class="detail-bar-pct">{pct.toFixed(0)}%</span>
                        </div>
                      {/if}
                    {/each}
                    {#each Object.entries(d.blocks) as [key, count]}
                      {#if count > 0 && !BLOCK_TYPES.find(bt => bt.key === key)}
                        {@const pct = d.total_blocks > 0 ? (count / d.total_blocks) * 100 : 0}
                        <div class="detail-bar-row">
                          <span class="detail-bar-label">{key}</span>
                          <div class="detail-bar-track">
                            <div class="detail-bar-fill" style="width:{pct}%;background:#666"></div>
                          </div>
                          <span class="detail-bar-count">{count}</span>
                          <span class="detail-bar-pct">{pct.toFixed(0)}%</span>
                        </div>
                      {/if}
                    {/each}
                  </div>
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
    display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap;
    color: var(--text-secondary); font-size: 0.85rem;
  }
  .summary .sep { color: var(--text-faint); }
  .summary .ok { color: var(--accent-ok); }
  .summary .warn { color: var(--accent-warn); }

  /* Legend */
  .legend {
    display: flex; gap: 12px; margin-bottom: 12px; flex-wrap: wrap;
  }
  .legend-item {
    display: flex; align-items: center; gap: 4px; font-size: 0.78rem; color: var(--text-secondary);
  }
  .legend-dot {
    width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0;
  }
  .legend-label { cursor: help; }

  /* Table */
  .stats-table { width: auto; max-width: 100%; font-size: 0.82rem; }
  .stats-table .clickable { cursor: pointer; }
  .stats-table .num { text-align: right; font-variant-numeric: tabular-nums; }
  .sortable { cursor: pointer; user-select: none; }
  .sortable:hover { color: var(--text-primary); }

  /* Stacked bar in table */
  .blocks-cell { display: flex; align-items: center; gap: 6px; }
  .block-bar {
    flex: 1; height: 14px; display: flex; border-radius: 2px; overflow: hidden;
    background: var(--bg-base);
  }
  .block-segment { height: 100%; min-width: 1px; }
  .blocks-total {
    font-family: ui-monospace, monospace; font-size: 0.72rem; color: var(--text-muted);
    min-width: 28px; text-align: right;
  }

  /* Detail row */
  .detail-row td { padding: 0; }
  .detail-box {
    padding: 14px 18px; background: var(--bg-base); border-top: 1px solid var(--border-subtle);
    font-size: 0.82rem; line-height: 1.6;
  }
  .detail-meta { margin-bottom: 12px; }
  .detail-blocks-title {
    font-weight: 600; margin-bottom: 8px; font-size: 0.8rem; color: var(--text-primary);
  }
  .detail-bars { display: flex; flex-direction: column; gap: 4px; }
  .detail-bar-row {
    display: flex; align-items: center; gap: 8px; font-size: 0.78rem;
  }
  .detail-bar-label {
    width: 120px; text-align: right; color: var(--text-secondary); flex-shrink: 0;
  }
  .detail-bar-track {
    flex: 1; height: 12px; background: var(--bg-elevated); border-radius: 2px; overflow: hidden;
  }
  .detail-bar-fill { height: 100%; border-radius: 2px; transition: width 0.2s; }
  .detail-bar-count {
    width: 36px; text-align: right; font-family: ui-monospace, monospace;
    font-size: 0.72rem; color: var(--text-primary);
  }
  .detail-bar-pct {
    width: 32px; text-align: right; font-family: ui-monospace, monospace;
    font-size: 0.72rem; color: var(--text-muted);
  }

  .muted { color: var(--text-muted); }
</style>
