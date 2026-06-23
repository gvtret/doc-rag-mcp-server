<script lang="ts">
  import { appState } from "../lib/state.svelte";

  type Tab = "ingest" | "http";
  let tab = $state<Tab>("ingest");
  let paused = $state(false);
  let lineCount = $state(200);
  let levelFilter = $state<string>("all");
  let container: HTMLDivElement;

  const logTail = $derived(appState.status?.log_tail ?? []);
  const httpTail = $derived(appState.status?.http_log_tail ?? []);

  const levels = ["all", "debug", "info", "warning", "error", "critical"];

  const activeLines = $derived.by(() => {
    const raw = tab === "ingest" ? logTail : httpTail;
    const sliced = raw.slice(-lineCount);
    if (levelFilter === "all") return sliced;
    return sliced.filter((l) =>
      l.toLowerCase().includes(`[${levelFilter}]`) ||
      l.toLowerCase().includes(levelFilter),
    );
  });

  $effect(() => {
    if (container && !paused) {
      container.scrollTop = container.scrollHeight;
    }
  });

  function lineClass(line: string): string {
    const lower = line.toLowerCase();
    if (lower.includes("[error]") || lower.includes("[critical]")) return "err";
    if (lower.includes("[warning]")) return "warn";
    if (lower.includes("[info]")) return "info";
    return "";
  }
</script>

<section>
  <div class="toolbar">
    <div class="tabs">
      <button
        type="button"
        class="tab"
        class:active={tab === "ingest"}
        onclick={() => (tab = "ingest")}
      >
        Ingest Log
      </button>
      <button
        type="button"
        class="tab"
        class:active={tab === "http"}
        onclick={() => (tab = "http")}
      >
        HTTP Log
      </button>
    </div>

    <div class="controls">
      <label class="ctrl-label">
        Уровень:
        <select bind:value={levelFilter}>
          {#each levels as lv (lv)}
            <option value={lv}>{lv}</option>
          {/each}
        </select>
      </label>
      <label class="ctrl-label">
        Строк:
        <select bind:value={lineCount}>
          <option value={50}>50</option>
          <option value={100}>100</option>
          <option value={200}>200</option>
          <option value={500}>500</option>
        </select>
      </label>
      <button
        type="button"
        class="btn"
        class:active-btn={paused}
        onclick={() => (paused = !paused)}
      >
        {paused ? "▶ Resume" : "⏸ Pause"}
      </button>
    </div>
  </div>

  <div class="log-container" bind:this={container}>
    {#if activeLines.length === 0}
      <p class="muted empty">Нет строк для отображения.</p>
    {:else}
      {#each activeLines as line, i (i)}
        <div class="line {lineClass(line)}">{line}</div>
      {/each}
    {/if}
  </div>

  <div class="status-bar mono">
    <span class="muted">lines: {activeLines.length}</span>
    <span class="sep">·</span>
    <span class="muted">source: {tab === "ingest" ? "log_tail" : "http_log_tail"}</span>
    {#if paused}
      <span class="sep">·</span>
      <span class="warn-text">paused</span>
    {/if}
  </div>
</section>

<style>
  section {
    display: flex;
    flex-direction: column;
    height: 100%;
    padding: 16px 24px;
    box-sizing: border-box;
    gap: 8px;
  }
  .toolbar {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }
  .tabs {
    display: flex;
    gap: 4px;
    border-bottom: 1px solid var(--border-strong);
  }
  .tab {
    background: transparent;
    color: var(--text-muted);
    border: 1px solid transparent;
    border-bottom: none;
    border-radius: 3px 3px 0 0;
    padding: 6px 14px;
    font-size: 0.85rem;
    font-family: ui-monospace, "JetBrains Mono", "Fira Code", SFMono-Regular, Menlo, Consolas, monospace;
    cursor: pointer;
  }
  .tab:hover {
    color: var(--text-primary);
  }
  .tab.active {
    color: var(--text-primary);
    border-color: var(--border-strong);
    background: var(--bg-elevated);
    margin-bottom: -1px;
  }
  .controls {
    display: flex;
    gap: 10px;
    align-items: center;
    margin-left: auto;
  }
  .ctrl-label {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 0.8rem;
    color: var(--text-muted);
    font-family: ui-monospace, "JetBrains Mono", "Fira Code", SFMono-Regular, Menlo, Consolas, monospace;
  }
  .ctrl-label select {
    background: var(--bg-elevated);
    color: var(--text-primary);
    border: 1px solid var(--border-strong);
    border-radius: 3px;
    padding: 3px 6px;
    font-size: 0.8rem;
    font-family: inherit;
  }
  .btn {
    background: var(--bg-elevated);
    color: var(--text-primary);
    border: 1px solid var(--border-strong);
    border-radius: 3px;
    padding: 4px 10px;
    font-size: 0.8rem;
    font-family: ui-monospace, "JetBrains Mono", "Fira Code", SFMono-Regular, Menlo, Consolas, monospace;
    cursor: pointer;
  }
  .btn:hover {
    border-color: var(--text-muted);
  }
  .btn.active-btn {
    border-color: var(--accent-warn);
    color: var(--accent-warn);
  }
  .log-container {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    background: #0d1117;
    border: 1px solid var(--border-strong);
    border-radius: 3px;
    padding: 8px 12px;
    font-family: ui-monospace, "JetBrains Mono", "Fira Code", SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.78rem;
    line-height: 1.5;
  }
  .line {
    white-space: pre-wrap;
    word-break: break-all;
    color: #c9d1d9;
  }
  .line.err {
    color: #f85149;
  }
  .line.warn {
    color: #d29922;
  }
  .line.info {
    color: #8b949e;
  }
  .empty {
    padding: 24px;
    text-align: center;
  }
  .muted {
    color: var(--text-muted);
  }
  .status-bar {
    display: flex;
    gap: 8px;
    align-items: center;
    font-size: 0.75rem;
    color: var(--text-muted);
  }
  .sep {
    color: var(--text-faint);
  }
  .warn-text {
    color: var(--accent-warn);
  }
</style>
