<script lang="ts">
  import { onMount } from "svelte";
  import LogTable from "$lib/components/LogTable.svelte";
  import { getLogs, getAgents } from "$lib/controllers/logs-controller";

  let agents: any[] = [];
  let selectedAgent = "";
  let logs: any[] = [];
  let loading = false;
  let error = "";
  let res = null;
  let logsCount = null;

  // Load agent list on mount
  onMount(async () => {
    try {
      const res = await getAgents();
      agents = res.agents?.data?.affected_items ?? [];
    } catch (e: any) {
      error = e.message ?? "Failed to load agents.";
      console.error("Error loading agents:", error);
    }
  });

  // Fetch logs for selected agent
  async function fetchLogs() {
    if (!selectedAgent) return;
    loading = true;
    error = "";
    logs = [];

    try {
      res = await getLogs(selectedAgent, 100);
      console.log("Fetched logs:", res);
      logsCount = res["count"];
      logs = res["logs"];
    } catch (e: any) {
      error = e.message ?? "Failed to fetch logs.";
      console.error("Error fetching logs:", error);
    } finally {
      loading = false;
    }
  }
</script>

<!-- Page title -->
<h2 class="h3 fw-bold mb-4">Logs</h2>

<!-- Agent selection and button -->
<div class="d-flex align-items-center gap-3 mb-4 flex-wrap">
  <!-- Agent selector -->
  <select
    bind:value={selectedAgent}
    class="form-select w-auto"
  >
    <option value="">Select agent...</option>
    {#each agents as agent}
      <option value={agent.id}>{agent.name} ({agent.id})</option>
    {/each}
  </select>

  <!-- Load button -->
  <button
    on:click={fetchLogs}
    class="btn btn-primary"
    disabled={!selectedAgent || loading}
  >
    {loading ? "Loading..." : "Load Logs"}
  </button>
</div>

<!-- Error message -->
{#if error}
  <div class="alert alert-danger" role="alert">
    {error}
  </div>
{/if}

<!-- Logs display -->
{#if logs.length > 0}
  <LogTable {logs} />
{:else if loading}
  <div class="d-flex align-items-center gap-2 text-secondary">
    <div class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></div>
    <span>Loading logs...</span>
  </div>
{:else}
  <p class="text-muted">No logs loaded.</p>
{/if}
