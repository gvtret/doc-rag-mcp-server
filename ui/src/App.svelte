<script lang="ts">
  import Sidebar from "./components/Sidebar.svelte";
  import Header from "./components/Header.svelte";
  import Footer from "./components/Footer.svelte";
  import Documents from "./pages/Documents.svelte";
  import Config from "./pages/Config.svelte";
  import Manage from "./pages/Manage.svelte";
  import Logs from "./pages/Logs.svelte";
  import Docs from "./pages/Docs.svelte";
  import { appState } from "./lib/state.svelte";

  // Single polling loop, owned by App. Each child reads from
  // `appState` directly via runes — no props drilling.
  $effect(() => {
    appState.start();
    return () => appState.stop();
  });
</script>

<div class="shell">
  <Sidebar />
  <Header />

  <main class="content">
    {#if appState.page === "documents"}
      <Documents />
    {:else if appState.page === "config"}
      <Config />
    {:else if appState.page === "manage"}
      <Manage />
    {:else if appState.page === "logs"}
      <Logs />
    {:else if appState.page === "docs"}
      <Docs />
    {/if}
  </main>

  <Footer />
</div>

<style>
  .shell {
    display: grid;
    grid-template-columns: 220px 1fr;
    grid-template-rows: auto 1fr auto;
    grid-template-areas:
      "sidebar header"
      "sidebar content"
      "footer  footer";
    /* Exact viewport height so the footer stays glued to the bottom
       of the window. Internal scrolling lives on `.content` only. */
    height: 100vh;
  }
  .content {
    grid-area: content;
    background: var(--bg-base);
    color: var(--text-primary);
    overflow: auto;
  }
</style>
