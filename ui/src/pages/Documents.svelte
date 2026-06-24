<script lang="ts">
  import { appState } from "../lib/state.svelte";
  import { fetchQuality, fetchAllDocuments, type QualityDoc, type AllDocument } from "../lib/api";

  const indexed = $derived(appState.status?.indexed);
  const running = $derived(appState.status?.running ?? false);

  let allDocs = $state<AllDocument[]>([]);
  let pendingDocs = $state<AllDocument[]>([]);
  let qualityMap = $state<Record<string, QualityDoc>>({});
  let loading = $state(true);

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

  function qualityColor(score: number): string {
    if (score >= 0.8) return "ok";
    if (score >= 0.5) return "warn";
    return "err";
  }

</script>

<section>
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
      <table>
        <thead>
          <tr>
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
            <tr>
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
  td {
    padding: 6px 10px;
    border-bottom: 1px solid var(--border-subtle);
    color: var(--text-secondary);
  }
  tbody tr:hover td {
    background: var(--bg-base);
    color: var(--text-primary);
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
