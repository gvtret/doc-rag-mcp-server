<script lang="ts">
  import { appState } from "../lib/state.svelte";
  import { t } from "../lib/i18n";
  import {
    fetchQuality,
    fetchAllDocuments,
    rebuildDocs,
    deleteDocuments,
    type QualityDoc,
    type AllDocument,
    type ManageResult,
  } from "../lib/api";

  const indexed = $derived(appState.status?.indexed);
  const running = $derived(appState.status?.running ?? false);

  let allDocs = $state<AllDocument[]>([]);
  let pendingDocs = $state<AllDocument[]>([]);
  let qualityMap = $state<Record<string, QualityDoc>>({});
  let loading = $state(true);
  let selected = $state<Set<string>>(new Set());
  let busy = $state(false);
  let banner: { tone: "ok" | "err" | "info"; text: string } | null = $state(null);
  let rebuildProgress = $state<{ current_doc: string | null; docs_done: number; docs_total: number | null } | null>(null);

  async function refresh() {
    try {
      const [docs, qual] = await Promise.all([fetchAllDocuments(), fetchQuality()]);
      if (docs.ok) {
        allDocs = docs.indexed;
        pendingDocs = docs.pending;
        rebuildProgress = docs.rebuild_progress ?? null;
      }
      if (qual.ok && qual.documents) {
        const m: Record<string, QualityDoc> = {};
        for (const d of qual.documents) m[d.doc_id] = d;
        qualityMap = m;
      }
    } catch {}
    loading = false;
  }

  $effect(() => {
    refresh();
    const iv = setInterval(refresh, 5000);
    return () => clearInterval(iv);
  });

  function toggleSelect(key: string) {
    const next = new Set(selected);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    selected = next;
  }

  function toggleSelectAll() {
    const allKeys = [
      ...allDocs.map((d) => d.doc_id ?? ""),
      ...pendingDocs.map((d) => d.path ?? d.basename),
    ].filter(Boolean);
    if (selected.size === allKeys.length) {
      selected = new Set();
    } else {
      selected = new Set(allKeys);
    }
  }

  async function doRebuildSelected() {
    if (busy || running || selected.size === 0) return;
    const selectedDocIds: string[] = [];
    const selectedFilenames: string[] = [];
    for (const key of selected) {
      const idxDoc = allDocs.find((d) => d.doc_id === key);
      if (idxDoc?.doc_id) {
        selectedDocIds.push(idxDoc.doc_id);
      } else {
        const pendDoc = pendingDocs.find((d) => (d.path ?? d.basename) === key);
        if (pendDoc) selectedFilenames.push(pendDoc.basename);
      }
    }
    const total = selectedDocIds.length + selectedFilenames.length;
    if (total === 0) return;
    if (!confirm(t("documents.rebuild_confirm", { n: total }))) return;
    busy = true;
    banner = null;
    try {
      const r: ManageResult = await rebuildDocs(selectedDocIds, selectedFilenames);
      if (r.ok) {
        banner = {
          tone: "info",
          text: r.message ?? t("documents.rebuild_started", { n: total }),
        };
        selected = new Set();
      } else {
        banner = { tone: "err", text: r.error ?? t("manage.error") };
      }
    } catch (e) {
      banner = { tone: "err", text: e instanceof Error ? e.message : String(e) };
    } finally {
      busy = false;
    }
  }

  async function doDeleteSelected() {
    if (busy || running || selected.size === 0) return;
    const docIds: string[] = [];
    for (const key of selected) {
      const idxDoc = allDocs.find((d) => d.doc_id === key);
      if (idxDoc?.doc_id) docIds.push(idxDoc.doc_id);
    }
    if (docIds.length === 0) return;
    if (!confirm(`Удалить ${docIds.length} документ(ов) из индекса? Это действие необратимо.`)) return;
    busy = true;
    banner = null;
    try {
      const r: ManageResult = await deleteDocuments(docIds);
      if (r.ok) {
        banner = { tone: "ok", text: `Удалено: ${r.deleted?.length ?? 0} документ(ов).` };
        selected = new Set();
        await refresh();
      } else {
        banner = { tone: "err", text: r.error ?? t("manage.error") };
      }
    } catch (e) {
      banner = { tone: "err", text: e instanceof Error ? e.message : String(e) };
    } finally {
      busy = false;
    }
  }

  function qualityColor(score: number): string {
    if (score >= 0.8) return "ok";
    if (score >= 0.5) return "warn";
    return "err";
  }

  function ocrTitle(ocr: { pages_recognized?: number; confidence?: number }): string {
    const parts = [t("documents.ocr_tooltip_base")];
    if (typeof ocr.pages_recognized === "number" && ocr.pages_recognized > 0) {
      parts.push(t("documents.ocr_tooltip_pages", { n: ocr.pages_recognized }));
    }
    if (typeof ocr.confidence === "number") {
      parts.push(t("documents.ocr_tooltip_confidence", { v: ocr.confidence.toFixed(2) }));
    }
    return parts.join(" · ");
  }
</script>

<section>
  {#if banner}
    <div class="banner {banner.tone}">{banner.text}</div>
  {/if}

  {#if indexed?.error}
    <p class="error">
      {t("documents.error_read_index")} <code>{indexed.error}</code>
    </p>
  {:else if loading}
    <p class="muted">{t("documents.loading")}</p>
  {:else}
    <div class="summary mono">
      <span>{t("documents.indexed")} <strong class="ok">{allDocs.length}</strong></span>
      <span class="sep">·</span>
      <span>{t("documents.pending")} <strong class={pendingDocs.length > 0 ? "warn" : ""}>{pendingDocs.length}</strong></span>
      <span class="sep">·</span>
      <span>
        {t("documents.lexical")} <strong class={indexed?.lexical_search_ready ? "ok" : "warn"}>
          {indexed?.lexical_search_ready ? t("documents.active") : t("search.off")}
        </strong>
      </span>
      <span class="sep">·</span>
      <span>
        {t("documents.semantic")} <strong class={indexed?.semantic_search_ready ? "ok" : "warn"}>
          {indexed?.semantic_search_ready ? t("documents.active") : t("search.off")}
        </strong>
      </span>
      {#if running}
        <span class="sep">·</span>
        <span class="warn">{t("footer.running")}…</span>
      {/if}
      {#if indexed?.pipeline_version}
        <span class="sep">·</span>
        <span class="muted">{t("documents.pipeline")} {indexed.pipeline_version}</span>
      {/if}
    </div>

    {#if rebuildProgress}
      <div class="progress-bar">
        <div class="progress-info mono">
          <span>{t("manage.info_running")} <strong>{rebuildProgress.current_doc ?? "…"}</strong></span>
          <span>{rebuildProgress.docs_done} / {rebuildProgress.docs_total ?? "?"}</span>
        </div>
        {#if rebuildProgress.docs_total && rebuildProgress.docs_total > 0}
          <div class="progress-track">
            <div
              class="progress-fill"
              style="width: {(rebuildProgress.docs_done / rebuildProgress.docs_total * 100).toFixed(1)}%"
            ></div>
          </div>
        {/if}
      </div>
    {/if}

    {#if allDocs.length === 0 && pendingDocs.length === 0}
      <p class="muted">{t("documents.no_documents")}</p>
    {:else}
      {#if selected.size > 0}
        <div class="toolbar">
          <span class="muted">{t("documents.selected")} {selected.size}</span>
          <button
            type="button"
            class="btn primary"
            disabled={busy || running}
            title="Пересобрать выбранные документы заново: парсинг + чанкинг + индексация."
            onclick={() => void doRebuildSelected()}
          >
            {t("documents.rebuild_selected")}
          </button>
          <button
            type="button"
            class="btn danger"
            disabled={busy || running}
            title="Удалить выбранные документы из индекса и манифеста. Необратимо."
            onclick={() => void doDeleteSelected()}
          >
            Удалить
          </button>
          <button type="button" class="btn" onclick={() => (selected = new Set())}>
            {t("documents.reset")}
          </button>
        </div>
      {/if}
      <table class="table doc-table">
        <thead>
          <tr>
            <th class="check">
              <input
                type="checkbox"
                checked={selected.size > 0 &&
                  allDocs.every((d) => selected.has(d.doc_id ?? "")) &&
                  pendingDocs.every((d) => selected.has(d.path ?? d.basename))}
                onchange={toggleSelectAll}
              />
            </th>
            <th class="num">##</th>
            <th>{t("documents.column_file")}</th>
            <th>{t("documents.column_doc_id")}</th>
            <th class="num">{t("documents.column_size")}</th>
            <th class="num">{t("documents.column_chunks")}</th>
            <th class="num">{t("documents.column_year")}</th>
            <th>{t("documents.column_status")}</th>
            <th>{t("documents.column_quality")}</th>
          </tr>
        </thead>
        <tbody>
          {#each allDocs as d, i (`${d.doc_id ?? d.basename}-${i}`)}
            <tr class:selected-row={selected.has(d.doc_id ?? "")}>
              <td class="check">
                <input
                  type="checkbox"
                  checked={selected.has(d.doc_id ?? "")}
                  onchange={() => toggleSelect(d.doc_id ?? "")}
                />
              </td>
              <td class="num muted">{String(i + 1).padStart(2, "0")}</td>
              <td title={d.source_file ?? ""}>
                {#if d.coverage?.ocr?.applied}
                  <span class="badge ok" title={ocrTitle(d.coverage.ocr)}>OCR</span>
                {/if}
                {d.basename}
              </td>
              <td class="mono small">{d.doc_id ?? "—"}</td>
              <td class="num">{d.size ? `${(d.size / 1024).toFixed(0)} КБ` : "—"}</td>
              <td class="num">{d.chunk_count ?? "—"}</td>
              <td class="num">{d.edition_year ?? "—"}</td>
              <td><span class="s-badge {d.status === 'processing' ? 's-processing' : d.status === 'processed' ? 's-processed' : 's-ok'}">
                {#if d.status === "processing"}
                  {t("documents.status_processing")}
                {:else if d.status === "processed"}
                  {t("documents.status_processed")}
                {:else}
                  {t("documents.status_indexed")}
                {/if}
              </span></td>
              <td>
                {#if d.doc_id && qualityMap[d.doc_id]}
                  {@const q = qualityMap[d.doc_id]}
                  <span
                    class="badge {qualityColor(q.score)}"
                    title="точность: {q.score.toFixed(2)} · предупреждения: {q.warning_count}"
                  >
                    {q.score >= 0.8 ? "OK" : q.score >= 0.5 ? "WARN" : "ERR"}
                  </span>
                {:else}
                  <span class="muted">—</span>
                {/if}
              </td>
            </tr>
          {/each}
          {#each pendingDocs as d, i (`${d.path ?? d.basename}-${i}`)}
            <tr class="pending-row">
              <td class="check">
                <input
                  type="checkbox"
                  checked={selected.has(d.path ?? d.basename)}
                  onchange={() => toggleSelect(d.path ?? d.basename)}
                />
              </td>
              <td class="num muted">{String(allDocs.length + i + 1).padStart(2, "0")}</td>
              <td title={d.path ?? ""}>{d.basename}</td>
              <td class="mono small muted">—</td>
              <td class="num muted">{d.size ? `${(d.size / 1024).toFixed(0)} КБ` : "—"}</td>
              <td class="num muted">—</td>
              <td class="num muted">—</td>
              <td><span class="s-badge {d.status === 'processing' ? 's-processing' : d.status === 'processed' ? 's-processed' : 's-pending'}">
                {#if d.status === "processing"}
                  {t("documents.status_processing")}
                {:else if d.status === "processed"}
                  {t("documents.status_processed")}
                {:else}
                  {t("documents.status_pending")}
                {/if}
              </span></td>
              <td><span class="muted">—</span></td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  {/if}
</section>

<style>
  section {
    padding: 16px 24px;
  }
  .banner {
    padding: 6px 10px;
    border-radius: 3px;
    font-size: 0.85rem;
    border-left: 3px solid;
    margin-bottom: 8px;
  }
  .banner.ok {
    background: rgba(31, 120, 95, 0.08);
    border-color: var(--accent-ok);
    color: var(--text-primary);
  }
  .banner.err {
    background: rgba(187, 58, 52, 0.08);
    border-color: var(--accent-error);
    color: var(--text-primary);
  }
  .banner.info {
    background: var(--bg-elevated);
    border-color: var(--border-strong);
    color: var(--text-secondary);
  }
  .progress-bar {
    margin-bottom: 10px;
    padding: 8px 10px;
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 3px;
  }
  .progress-info {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin-bottom: 4px;
  }
  .progress-track {
    height: 4px;
    background: var(--border-subtle);
    border-radius: 2px;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%;
    background: var(--accent-ok);
    border-radius: 2px;
    transition: width 0.3s ease;
  }
  .toolbar {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    padding: 6px 10px;
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 3px;
  }
  .summary {
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
    flex-wrap: wrap;
    color: var(--text-secondary);
    font-size: 0.85rem;
  }
  .summary .sep {
    color: var(--text-faint);
  }
  .summary .ok {
    color: var(--accent-ok);
  }
  .summary .warn {
    color: var(--accent-warn);
  }
  .muted {
    color: var(--text-muted);
  }
  .error {
    color: var(--accent-error);
  }
  .doc-table {
    max-width: 100%;
    font-size: 0.85rem;
  }
  .doc-table th.check,
  .doc-table td.check {
    text-align: center;
    width: 32px;
    padding: 6px 4px;
  }
  .doc-table tbody tr.selected-row td {
    background: rgba(31, 120, 95, 0.06);
  }
  .doc-table td.num,
  .doc-table th.num {
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
  .small {
    font-size: 0.72rem;
  }
  .pending-row td {
    opacity: 0.55;
  }
  .s-badge {
    display: inline-block;
    padding: 1px 6px;
    font-size: 10px;
    font-weight: 600;
    border-radius: 2px;
    font-family: ui-monospace, "JetBrains Mono", "Fira Code", SFMono-Regular,
      Menlo, Consolas, monospace;
    letter-spacing: 0.05em;
  }
  .s-ok {
    color: var(--bg-base);
    background: var(--accent-ok);
  }
  .s-pending {
    color: var(--text-primary);
    background: var(--border-strong);
  }
  .s-processing {
    color: var(--bg-base);
    background: var(--accent-warn);
  }
  .s-processed {
    color: var(--bg-base);
    background: var(--accent-ok);
    opacity: 0.7;
  }
</style>
