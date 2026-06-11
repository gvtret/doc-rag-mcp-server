<script lang="ts">
  import { appState } from "../lib/state.svelte";

  const health = $derived.by(() => {
    const h = appState.health;
    if (!h) return { label: "?", level: "unknown" as const };
    if (h.ready) return { label: "ready", level: "ok" as const };
    if (h.has_manifest) return { label: "degraded", level: "warn" as const };
    return { label: "down", level: "error" as const };
  });

  const searchMode = $derived.by(() => {
    const idx = appState.status?.indexed;
    if (!idx) return "?";
    if (idx.semantic_search_ready) return "semantic";
    if (idx.lexical_search_ready) return "lexical*";
    return "none";
  });

  const currentOp = $derived.by(() => {
    const s = appState.status;
    if (!s?.running) return { label: "idle", detail: "" };
    const parts: string[] = [];
    if (s.current_doc) parts.push(s.current_doc);
    if (typeof s.docs_done === "number" && s.docs_total != null) {
      parts.push(`${s.docs_done}/${s.docs_total}`);
    }
    return { label: s.job || "busy", detail: parts.join(" · ") };
  });

  const eta = $derived(appState.status?.eta_human ?? null);

  const lastOp = $derived.by(() => {
    if (appState.lastJobName === null) return null;
    const ms = appState.lastJobDurationMs;
    const dur = ms !== null ? humanizeMs(ms) : null;
    const ok = appState.lastJobOk;
    return { name: appState.lastJobName, dur, ok };
  });

  function humanizeMs(ms: number): string {
    const s = Math.round(ms / 1000);
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60);
    const rem = s % 60;
    if (m < 60) return rem ? `${m}m ${String(rem).padStart(2, "0")}s` : `${m}m`;
    const h = Math.floor(m / 60);
    const mm = m % 60;
    return mm ? `${h}h ${String(mm).padStart(2, "0")}m` : `${h}h`;
  }

  let now = $state<Date>(new Date());
  $effect(() => {
    const id = window.setInterval(() => (now = new Date()), 1000);
    return () => window.clearInterval(id);
  });
  const clock = $derived(now.toLocaleTimeString([], { hour12: false }));
  const dateStr = $derived(now.toLocaleDateString());

  const docCount = $derived(appState.status?.indexed?.document_count ?? null);
</script>

<footer class="page-footer">
  <span class="cell">
    <span class="dot dot-{health.level}" aria-hidden="true"></span>
    <span class="k">HEALTH</span>
    <span class="v">{health.label}</span>
  </span>

  <span class="cell">
    <span class="k">SEARCH</span>
    <span class="v">{searchMode}</span>
  </span>

  <span class="cell">
    <span class="k">JOB</span>
    <span class="v {appState.status?.running ? 'running' : ''}">
      {currentOp.label}
    </span>
    {#if currentOp.detail}
      <span class="muted">· {currentOp.detail}</span>
    {/if}
  </span>

  {#if eta}
    <span class="cell">
      <span class="k">ETA</span>
      <span class="v">~{eta}</span>
    </span>
  {/if}

  {#if lastOp}
    <span class="cell">
      <span class="k">LAST</span>
      <span class="v">{lastOp.name}</span>
      {#if lastOp.ok !== null}
        <span class="dot dot-{lastOp.ok ? 'ok' : 'error'}"></span>
      {/if}
      {#if lastOp.dur}
        <span class="muted">{lastOp.dur}</span>
      {/if}
    </span>
  {/if}

  <span class="cell">
    <span class="k">DOCS</span>
    <span class="v">{docCount ?? "—"}</span>
  </span>

  <span class="cell push-right">
    <span class="muted">{dateStr}</span>
    <span class="v">{clock}</span>
  </span>
</footer>

<style>
  .page-footer {
    grid-area: footer;
    background: var(--bg-elevated);
    color: var(--text-secondary);
    padding: 5px 14px;
    display: flex;
    align-items: center;
    gap: 18px;
    font-family: ui-monospace, "JetBrains Mono", "Fira Code", SFMono-Regular,
      Menlo, Consolas, monospace;
    font-size: 0.75rem;
    border-top: 1px solid var(--border-strong);
    flex-wrap: wrap;
  }
  .cell {
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }
  .k {
    color: var(--text-faint);
    letter-spacing: 0.04em;
  }
  .v {
    color: var(--text-primary);
  }
  .v.running {
    color: var(--accent-info);
  }
  .muted {
    color: var(--text-muted);
  }
  .dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .dot-ok {
    background: var(--accent-ok);
    box-shadow: 0 0 6px rgba(34, 197, 94, 0.5);
  }
  .dot-warn {
    background: var(--accent-warn);
    box-shadow: 0 0 6px rgba(245, 158, 11, 0.5);
  }
  .dot-error {
    background: var(--accent-error);
    box-shadow: 0 0 6px rgba(239, 68, 68, 0.5);
  }
  .dot-unknown {
    background: var(--text-faint);
  }
  .push-right {
    margin-left: auto;
  }
</style>
