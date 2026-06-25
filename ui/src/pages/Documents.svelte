<script lang="ts">
  import { appState } from "../lib/state.svelte";
  import {
    fetchQuality,
    fetchAllDocuments,
    rebuildDocs,
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

  async function refresh() {
    try {
      const [docs, qual] = await Promise.all([fetchAllDocuments(), fetchQuality()]);
      if (docs.ok) {
        allDocs = docs.indexed;
        pendingDocs = docs.pending;
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
    const selectedDocIds = [...selected].filter((k) =>
      allDocs.some((d) => d.doc_id === k),
    );
    if (selectedDocIds.length === 0) return;
    if (
      !confirm(
        `Пересобрать ${selectedDocIds.length} документ(ов)? Их текущий индекс будет перезаписан.`,
      )
    )
      return;
    busy = true;
    banner = null;
    try {
      const r: ManageResult = await rebuildDocs(selectedDocIds);
      if (r.ok) {
        banner = {
          tone: "info",
          text: r.message ?? `Rebuild запущен для ${selectedDocIds.length} документ(ов).`,
        };
        selected = new Set();
      } else {
        banner = { tone: "err", text: r.error ?? "Ошибка" };
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
</script>

<section>
  {#if banner}
    <div class="banner {banner.tone}">{banner.text}</div>
  {/if}

  {#if indexed?.error}
    <p class="error">
      Не удалось прочитать индекс: <code>{indexed.error}</code>
    </p>
  {:else if loading}
    <p class="muted">Загрузка…</p>
  {:else}
    <div class="summary mono">
      <span>indexed: <strong class="ok">{allDocs.length}</strong></span>
      <span class="sep">·</span>
      <span>pending: <strong class={pendingDocs.length > 0 ? "warn" : ""}>{pendingDocs.length}</strong></span>
      <span class="sep">·</span>
      <span>
        lex: <strong class={indexed?.lexical_search_ready ? "ok" : "warn"}>
          {indexed?.lexical_search_ready ? "ready" : "off"}
        </strong>
      </span>
      <span class="sep">·</span>
      <span>
        sem: <strong class={indexed?.semantic_search_ready ? "ok" : "warn"}>
          {indexed?.semantic_search_ready ? "ready" : "off"}
        </strong>
      </span>
      {#if running}
        <span class="sep">·</span>
        <span class="warn">ingest/rebuild…</span>
      {/if}
      {#if indexed?.pipeline_version}
        <span class="sep">·</span>
        <span class="muted">pipeline {indexed.pipeline_version}</span>
      {/if}
    </div>

    {#if allDocs.length === 0 && pendingDocs.length === 0}
      <p class="muted">Пока ни одного документа.</p>
    {:else}
      {#if selected.size > 0}
        <div class="toolbar">
          <span class="muted">Выбрано: {selected.size}</span>
          <button
            type="button"
            class="btn"
            disabled={busy || running}
            onclick={() => void doRebuildSelected()}
          >
            Rebuild selected
          </button>
          <button type="button" class="btn secondary" onclick={() => (selected = new Set())}>
            Сбросить
          </button>
        </div>
      {/if}
      <table>
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
            <th>FILENAME</th>
            <th>DOC_ID</th>
            <th class="num">CHUNKS</th>
            <th class="num">YEAR</th>
            <th>STATUS</th>
            <th>QUALITY</th>
          </tr>
        </thead>
        <tbody>
          {#each allDocs as d, i (d.doc_id ?? d.basename + i)}
            <tr class:selected-row={selected.has(d.doc_id ?? "")}>
              <td class="check">
                <input
                  type="checkbox"
                  checked={selected.has(d.doc_id ?? "")}
                  onchange={() => toggleSelect(d.doc_id ?? "")}
                />
              </td>
              <td class="num muted">{String(i + 1).padStart(2, "0")}</td>
              <td title={d.source_file ?? ""}>{d.basename}</td>
              <td class="mono small">{d.doc_id ?? "—"}</td>
              <td class="num">{d.chunk_count ?? "—"}</td>
              <td class="num">{d.edition_year ?? "—"}</td>
              <td><span class="s-badge s-ok">indexed</span></td>
              <td>
                {#if d.doc_id && qualityMap[d.doc_id]}
                  {@const q = qualityMap[d.doc_id]}
                  <span
                    class="q-badge q-{qualityColor(q.score)}"
                    title="score: {q.score.toFixed(2)} · warnings: {q.warning_count}"
                  >
                    {q.score >= 0.8 ? "OK" : q.score >= 0.5 ? "WARN" : "ERR"}
                  </span>
                {:else}
                  <span class="muted">—</span>
                {/if}
              </td>
            </tr>
          {/each}
          {#each pendingDocs as d, i (d.path ?? d.basename + i)}
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
              <td class="num muted">—</td>
              <td class="num muted">—</td>
              <td><span class="s-badge s-pending">pending</span></td>
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
    background: rgba(34, 197, 94, 0.1);
    border-color: var(--accent-ok);
    color: var(--text-primary);
  }
  .banner.err {
    background: rgba(239, 68, 68, 0.1);
    border-color: var(--accent-error);
    color: var(--text-primary);
  }
  .banner.info {
    background: rgba(59, 130, 246, 0.08);
    border-color: var(--accent-info);
    color: var(--text-secondary);
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
  .btn {
    background: var(--bg-elevated);
    color: var(--text-primary);
    border: 1px solid var(--border-strong);
    border-radius: 3px;
    padding: 4px 10px;
    font-size: 0.82rem;
    font-family: ui-monospace, "JetBrains Mono", "Fira Code", SFMono-Regular,
      Menlo, Consolas, monospace;
    cursor: pointer;
  }
  .btn:hover:not(:disabled) {
    border-color: var(--text-muted);
  }
  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .btn.secondary {
    color: var(--text-muted);
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
  table {
    width: 100%;
    border-collapse: collapse;
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    font-size: 0.85rem;
  }
  thead {
    background: var(--bg-base);
  }
  th {
    padding: 7px 10px;
    text-align: left;
    border-bottom: 1px solid var(--border-strong);
    color: var(--text-faint);
    font-family: ui-monospace, "JetBrains Mono", "Fira Code", SFMono-Regular,
      Menlo, Consolas, monospace;
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    font-weight: 500;
  }
  th.check,
  td.check {
    text-align: center;
    width: 32px;
    padding: 6px 4px;
  }
  td {
    padding: 6px 10px;
    border-bottom: 1px solid var(--border-subtle);
    color: var(--text-secondary);
  }
  tbody tr:hover td {
    background: var(--bg-base);
    color: var(--text-primary);
  }
  tbody tr.selected-row td {
    background: rgba(59, 130, 246, 0.06);
  }
  tbody tr:last-child td {
    border-bottom: none;
  }
  td.num,
  th.num {
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
  .q-badge {
    display: inline-block;
    padding: 1px 6px;
    font-size: 10px;
    font-weight: 600;
    border-radius: 2px;
    cursor: help;
    font-family: ui-monospace, "JetBrains Mono", "Fira Code", SFMono-Regular,
      Menlo, Consolas, monospace;
    letter-spacing: 0.05em;
  }
  .q-ok {
    color: var(--bg-base);
    background: var(--accent-ok);
  }
  .q-warn {
    color: var(--bg-base);
    background: var(--accent-warn);
  }
  .q-err {
    color: #fff;
    background: var(--accent-error);
  }
</style>
