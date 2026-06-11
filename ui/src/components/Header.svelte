<script lang="ts">
  import { appState } from "../lib/state.svelte";
  import type { PageKey } from "../lib/types";

  const titles: Record<PageKey, string> = {
    documents: "Документы",
    config: "Конфигурация",
    manage: "Управление",
    logs: "Логи",
  };
</script>

<header class="page-header">
  <h1>
    <span class="prompt">//</span>
    {titles[appState.page]}
  </h1>
  <div class="submenu-slot" aria-label="Подменю страницы">
    <!-- v2.2.2+: per-page submenu lands here -->
  </div>
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
