<script lang="ts">
  import { appState } from "../lib/state.svelte";

  const documents = $derived(appState.status?.indexed?.documents ?? []);
  const indexed = $derived(appState.status?.indexed);

  function shortenSha(s: string | null | undefined): string {
    if (!s) return "—";
    return s.length > 16 ? s.slice(0, 16) + "…" : s;
  }
</script>

<section>
  {#if indexed?.error}
    <p class="error">
      Не удалось прочитать индекс: <code>{indexed.error}</code>
    </p>
  {:else if !indexed}
    <p class="muted">Загрузка…</p>
  {:else}
    <div class="summary mono">
      <span>docs: <strong>{indexed.document_count ?? 0}</strong></span>
      <span class="sep">·</span>
      <span>
        lex: <strong
          class={indexed.lexical_search_ready ? "ok" : "warn"}>{indexed.lexical_search_ready ? "ready" : "off"}</strong
        >
      </span>
      <span class="sep">·</span>
      <span>
        sem: <strong
          class={indexed.semantic_search_ready ? "ok" : "warn"}>{indexed.semantic_search_ready ? "ready" : "off"}</strong
        >
      </span>
      {#if indexed.pipeline_version}
        <span class="sep">·</span>
        <span class="muted">pipeline {indexed.pipeline_version}</span>
      {/if}
    </div>

    {#if documents.length === 0}
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
            <th>OCR</th>
            <th>SHA256</th>
          </tr>
        </thead>
        <tbody>
          {#each documents as d, i (d.doc_id)}
            <tr>
              <td class="num muted">{String(i + 1).padStart(2, "0")}</td>
              <td title={d.source_file ?? ""}>{d.basename ?? "—"}</td>
              <td class="mono small">{d.doc_id}</td>
              <td class="num">{d.chunk_count ?? "—"}</td>
              <td class="num">{d.edition_year ?? "—"}</td>
              <td>
                {#if d.coverage?.ocr?.applied}
                  <span
                    class="ocr-badge"
                    title="страниц: {d.coverage.ocr.pages_recognized ?? '?'} · уверенность: {(d.coverage.ocr.confidence ?? 0).toFixed(2)}"
                  >
                    OCR
                  </span>
                {:else}
                  <span class="muted">—</span>
                {/if}
              </td>
              <td class="mono small muted">{shortenSha(d.sha256)}</td>
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
  .ocr-badge {
    display: inline-block;
    padding: 1px 6px;
    font-size: 10px;
    font-weight: 600;
    color: var(--bg-base);
    background: var(--accent-warn);
    border-radius: 2px;
    cursor: help;
    font-family: ui-monospace, "JetBrains Mono", "Fira Code", SFMono-Regular,
      Menlo, Consolas, monospace;
    letter-spacing: 0.05em;
  }
</style>
