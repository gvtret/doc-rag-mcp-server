<script lang="ts">
  import { appState } from "../lib/state.svelte";
  import { t } from "../lib/i18n";
  import LangSwitcher from "./LangSwitcher.svelte";
  import type { PageKey } from "../lib/types";

  const titleKeys: Record<PageKey, string> = {
    documents: "sidebar.documents",
    config: "sidebar.config",
    manage: "sidebar.manage",
    logs: "sidebar.logs",
    docs: "sidebar.docs",
  };
</script>

<header class="page-header">
  <h1>
    <span class="prompt">//</span>
    {t(titleKeys[appState.page])}
  </h1>
  <div class="submenu-slot" aria-label="Подменю страницы">
    <!-- v2.2.2+: per-page submenu lands here -->
  </div>
  <LangSwitcher />
</header>

<style>
  .page-header {
    grid-area: header;
    background: var(--bg-base);
    border-bottom: 1px solid var(--border-subtle);
    padding: 10px 24px;
    display: flex;
    align-items: center;
    gap: 24px;
  }
  h1 {
    margin: 0;
    font-size: 1rem;
    font-weight: 500;
    color: var(--text-primary);
    font-family: ui-monospace, "JetBrains Mono", "Fira Code", SFMono-Regular,
      Menlo, Consolas, monospace;
  }
  .prompt {
    color: var(--text-faint);
    margin-right: 6px;
  }
  .submenu-slot {
    flex: 1;
    display: flex;
    gap: 12px;
    align-items: center;
  }
</style>
