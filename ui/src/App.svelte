<script lang="ts">
  /**
   * v2.2.0 — stub page mounted at /ui-next/.
   *
   * Goal: prove the toolchain (Vite + Svelte 5 + TS, FastAPI static
   * mount, bootstrap.sh + Dockerfile + CI build) works end-to-end on
   * the production server. No business logic yet; the real /ui
   * migration lands in v2.2.1+.
   *
   * Fetches the existing JSON `/ui/status` endpoint to show that the
   * SPA can talk to the FastAPI backend through the same origin —
   * the same path used by the legacy inline UI.
   */
  type StatusPayload = {
    running?: boolean;
    job?: string | null;
    last_ok?: boolean | null;
    last_error?: string | null;
    docs_done?: number;
    docs_total?: number | null;
    indexed?: {
      document_count?: number;
      manifest_present?: boolean;
      semantic_search_ready?: boolean;
    };
  };

  let status = $state<StatusPayload | null>(null);
  let error = $state<string | null>(null);
  let lastFetchedAt = $state<Date | null>(null);

  async function loadStatus() {
    try {
      const r = await fetch("/ui/status", { credentials: "same-origin" });
      if (!r.ok) {
        throw new Error(`HTTP ${r.status}`);
      }
      status = (await r.json()) as StatusPayload;
      lastFetchedAt = new Date();
      error = null;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  $effect(() => {
    void loadStatus();
    const id = window.setInterval(() => void loadStatus(), 5000);
    return () => window.clearInterval(id);
  });

  function fmtBool(v: boolean | null | undefined): string {
    if (v === true) return "yes";
    if (v === false) return "no";
    return "—";
  }
</script>

<main>
  <h1>doc-rag — UI <span class="muted">(Svelte stub, v2.2.0)</span></h1>

  <p class="muted">
    This is the Svelte-based UI stub introduced in v2.2.0 to prove the
    toolchain. The real <code>/ui</code> migration lands in v2.2.1+ —
    until then the legacy inline page at
    <a href="/ui">/ui</a> stays canonical.
  </p>

  {#if error}
    <p style="color: #b91c1c">
      <strong>Failed to load <code>/ui/status</code>:</strong>
      <code>{error}</code>
    </p>
  {/if}

  {#if status}
    <h2>Live status (polled every 5 s)</h2>
    <ul>
      <li>
        Background job running: <strong>{fmtBool(status.running)}</strong>
        {#if status.job}
          (<code>{status.job}</code>)
        {/if}
      </li>
      <li>
        Last result OK: <strong>{fmtBool(status.last_ok)}</strong>
      </li>
      {#if status.last_error}
        <li>
          Last error: <code>{status.last_error}</code>
        </li>
      {/if}
      <li>
        Indexed documents:
        <strong>{status.indexed?.document_count ?? "—"}</strong>
      </li>
      <li>
        Semantic search ready:
        <strong>{fmtBool(status.indexed?.semantic_search_ready)}</strong>
      </li>
    </ul>
    {#if lastFetchedAt}
      <p class="muted">
        Updated at <code>{lastFetchedAt.toLocaleTimeString()}</code>.
      </p>
    {/if}
  {:else if !error}
    <p class="muted">Loading…</p>
  {/if}
</main>
