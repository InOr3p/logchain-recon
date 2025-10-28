<script lang="ts">
  import { onMount } from "svelte";
  import { getLogs, getAgents } from "$lib/controllers/logs-controller";
  import { logs, selectedLogs, showAlert, agents } from "$lib/stores/generalStores";
  
  const ITEMS_PER_PAGE = 10;
  let currentPage = 1;
  
  $: totalPages = Math.ceil($logs.length / ITEMS_PER_PAGE);
  $: paginatedLogs = $logs.slice(
    (currentPage - 1) * ITEMS_PER_PAGE,
    currentPage * ITEMS_PER_PAGE
  );
  
  function changePage(page: number) {
    if (page >= 1 && page <= totalPages) {
      currentPage = page;
    }
  }
  
  function nextPage() {
    if (currentPage < totalPages) currentPage++;
  }
  
  function prevPage() {
    if (currentPage > 1) currentPage--;
  }
  
  function toggleLogSelection(log: any) {
    const index = $selectedLogs.indexOf(log);
    if (index > -1) {
      $selectedLogs = $selectedLogs.filter(l => l !== log);
    } else {
      $selectedLogs = [...$selectedLogs, log];
    }
  }
  
  function selectAllLogs() {
    $selectedLogs = [...$logs];
  }
  
  function deselectAllLogs() {
    $selectedLogs = [];
  }

  let selectedAgent = "";
  let loading = false;
  let error = "";
  let res = null;
  let logsCount = null;

  onMount(async () => {
    // Only fetch agents if the store is empty
    if ($agents.length > 0) return; 

    try {
      const res = await getAgents();
      $agents = res.agents?.data?.affected_items ?? []; 
    } catch (e: any) {
      error = e.message ?? "Failed to load agents.";
      console.error("Error loading agents:", error);
      showAlert("Error loading agents", "danger", 5000);
    }
  });

  async function fetchLogs() {
    if (!selectedAgent) return;
    loading = true;
    error = "";
    $logs = []; 
    $selectedLogs = [];
    currentPage = 1; // Reset to first page

    try {
      res = await getLogs(selectedAgent, 100);
      console.log("Fetched logs:", res);
      logsCount = res["count"];
      $logs = res["logs"]; 
    } catch (e: any) {
      error = e.message ?? "Failed to fetch logs.";
      console.error("Error fetching logs:", error);
      showAlert("Error fetching logs", "danger", 5000)
    } finally {
      loading = false;
    }
  }
  
  $: console.log('Selected logs:', $selectedLogs);
</script>

<div class="d-flex flex-column flex-grow-1">

  <!-- 
    Header Section
  -->
  <div class="d-flex justify-content-between align-items-center flex-shrink-0 m-3">
    <h3 class="mb-0"><strong>Logs</strong></h3>
    <div class="d-flex align-items-center gap-3">
      <select
        bind:value={selectedAgent}
        class="form-select w-auto"
      >
        <option value="">Select agent...</option>
        {#each $agents as agent}
          <option value={agent.id}>{agent.name} ({agent.id})</option>
        {/each}
      </select>

      <button
        on:click={fetchLogs}
        class="btn btn-primary"
        disabled={!selectedAgent || loading}
      >
        {loading ? "Loading..." : "Load Logs"}
      </button>
    </div>
  </div>

  <!-- 
    Scrollable Content Section
  -->
  <div class="flex-grow-1 overflow-auto" style="min-height: 0;">

    {#if $logs.length > 0}
      <div class="container-fluid my-4">
        <!-- Selection Controls -->
        <div class="selection-controls mb-3">
          <div class="d-flex align-items-center gap-3">
            <span class="text-muted">
              <strong>{$selectedLogs.length}</strong> of <strong>{$logs.length}</strong> logs selected
            </span>
            <button class="btn btn-sm btn-primary" on:click={selectAllLogs}>
              Select All ({$logs.length})
            </button>
            {#if $selectedLogs.length > 0}
              <button class="btn btn-sm btn-outline-secondary" on:click={deselectAllLogs}>
                Clear
              </button>
            {/if}
          </div>
        </div>
        
        <!-- Table Card -->
        <div class="table-card">
          <div class="table-responsive">
            <table class="table table-dark table-hover align-middle mb-0">
              <!-- thead -->
              <thead>
                <tr>
                  <th style="width: 40px;"></th>
                  <th>Created At</th>
                  <th>Description</th>
                  <th>Rule ID</th>
                  <th>Level</th>
                  <th>Agent IP</th>
                  <th>Source IP</th>
                  <th>Groups</th>
                  <th>NIST 800-53</th>
                  <th>GDPR</th>
                </tr>
              </thead>
              <!-- tbody -->
              <tbody>
                {#if paginatedLogs.length > 0}
                  {#each paginatedLogs as log}
                    <tr
                      class:selected={$selectedLogs.includes(log)}
                      on:click={() => toggleLogSelection(log)}
                      style="cursor: pointer;"
                    >
                      <td style="text-align: center;">
                        <input
                          type="checkbox"
                          checked={$selectedLogs.includes(log)}
                          on:click|stopPropagation={() => toggleLogSelection(log)}
                        />
                      </td>
                      <td class="text-muted">{log.timestamp}</td>
                      <td>{log.rule_description ?? "—"}</td>
                      <td><span class="badge-code">{log.rule_id ?? "—"}</span></td>
                      <td>
                        <span class="badge-level level-{log.rule_level || 0}">
                          {log.rule_level ?? "—"}
                        </span>
                      </td>
                      <td class="font-mono">{log.agent_ip ?? "—"}</td>
                      <td class="font-mono">{log.data_srcip ?? "—"}</td>
                      <td class="text-muted small">
                        {#if log.rule_groups && log.rule_groups.length > 0}
                          {log.rule_groups.join(", ")}
                        {:else}
                          —
                        {/if}
                      </td>
                      <td class="text-muted small">
                        {#if log.rule_nist_800_53 && log.rule_nist_800_53.length > 0}
                          {log.rule_nist_800_53.join(", ")}
                        {:else}
                          —
                        {/if}
                      </td>
                      <td class="text-muted small">
                        {#if log.rule_gdpr && log.rule_gdpr.length > 0}
                          {log.rule_gdpr.join(", ")}
                        {:else}
                          —
                        {/if}
                      </td>
                    </tr>
                  {/each}
                {:else}
                  <tr>
                    <td colspan="10" class="text-center empty-state">
                      <div class="py-4">
                        <svg class="empty-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                            d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
                        </svg>
                        <p class="mb-0 mt-2">No logs available</p>
                      </div>
                    </td>
                  </tr>
                {/if}
              </tbody>
            </table>
          </div>
        </div>

        <!-- Pagination Controls -->
        {#if totalPages > 1}
          <div class="pagination-wrapper">
            <button class="page-btn" on:click={prevPage} disabled={currentPage === 1}>
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="15 18 9 12 15 6"></polyline>
              </svg>
            </button>
            
            <div class="page-numbers">
              {#each Array(totalPages) as _, i}
                {#if i + 1 === 1 || i + 1 === totalPages || (i + 1 >= currentPage - 1 && i + 1 <= currentPage + 1)}
                  <button 
                    class="page-num" 
                    class:active={currentPage === i + 1}
                    on:click={() => changePage(i + 1)}
                  >
                    {i + 1}
                  </button>
                {:else if i + 1 === currentPage - 2 || i + 1 === currentPage + 2}
                  <span class="ellipsis">...</span>
                {/if}
              {/each}
            </div>
            
            <button class="page-btn" on:click={nextPage} disabled={currentPage === totalPages}>
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="9 18 15 12 9 6"></polyline>
              </svg>
            </button>
          </div>
        {/if}
      </div>
    {:else if loading}
      <div class="d-flex align-items-center gap-2 text-secondary p-3">
        <div class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></div>
        <span>Loading logs...</span>
      </div>
    {:else}
      <p class="text-muted p-3">No logs loaded.</p>
    {/if}

  </div> <!-- End Scrollable Content -->
</div> <!-- End Root Container -->


<style>
  .table-card {
    background: #1e1e1e;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    overflow: hidden;
  }
  .table {
    color: #e0e0e0 !important;
    margin-bottom: 0;
    background-color: #1a1a1a !important;
  }
  .table tbody {
    background-color: #1a1a1a !important;
  }
  .table tbody tr {
    background-color: #1a1a1a !important;
  }
  .table tbody td {
    background-color: #1a1a1a !important;
  }
  .table thead th {
    background: #252525;
    color: #b0c4ff;
    font-weight: 600;
    font-size: 0.875rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1px solid #333;
    padding: 1rem 0.75rem;
    white-space: nowrap;
  }
  .table tbody td {
    padding: 0.875rem 0.75rem;
    border-bottom: 1px solid #2a2a2a;
    font-size: 0.9375rem;
    vertical-align: middle;
    color: #f0f0f0 !important;
  }
  .table tbody tr:last-child td {
    border-bottom: none;
  }
  .table tbody tr {
    transition: background-color 0.15s ease;
    background-color: #1a1a1a !important;
  }
  .table tbody tr:hover {
    background-color: #242424 !important;
  }
  .table tbody tr:hover td {
    background-color: #242424 !important;
  }
  tr.selected {
    background-color: #2b2b2b !important;
  }
  .font-mono {
    font-family: 'Courier New', monospace;
    font-size: 0.875rem;
    color: #e0e0e0 !important;
  }
  .text-muted {
    color: #a0a0a0 !important;
  }
  .badge-code {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    background: #2a2a2a;
    border: 1px solid #333;
    border-radius: 4px;
    font-family: 'Courier New', monospace;
    font-size: 0.8125rem;
    color: #b0c4ff;
  }
  .badge-level {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8125rem;
    font-weight: 600;
    text-align: center;
    min-width: 2rem;
  }
  .level-1, .level-2, .level-3 {
    background: #1e3a5f;
    color: #60a5fa;
  }
  .level-4, .level-5, .level-6 {
    background: #7c3f00;
    color: #fbbf24;
  }
  .level-7, .level-8, .level-9, .level-10 {
    background: #7f1d1d;
    color: #f87171;
  }
  .empty-state {
    color: #6b7280;
  }
  .empty-icon {
    width: 48px;
    height: 48px;
    opacity: 0.5;
    margin: 0 auto;
  }

  /* New Pagination Styles */
  .pagination-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0.5rem;
    margin-top: 1.5rem;
    padding-top: 1rem;
    border-top: 1px solid #2a2a2a;
  }
  
  .page-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    background: #252525;
    border: 1px solid #333;
    border-radius: 6px;
    color: #e0e0e0;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  
  .page-btn:hover:not(:disabled) {
    background: #2a2a2a;
    border-color: #444;
  }
  
  .page-btn:disabled {
    opacity: 0.3;
    cursor: not-allowed;
  }
  
  .page-numbers {
    display: flex;
    gap: 0.25rem;
  }
  
  .page-num {
    min-width: 36px;
    height: 36px;
    padding: 0 0.5rem;
    background: #252525;
    border: 1px solid #333;
    border-radius: 6px;
    color: #e0e0e0;
    cursor: pointer;
    transition: all 0.2s ease;
    font-size: 0.875rem;
  }
  
  .page-num:hover {
    background: #2a2a2a;
    border-color: #444;
  }
  
  .page-num.active {
    background: #3b82f6;
    border-color: #3b82f6;
    color: white;
  }
  
  .ellipsis {
    display: flex;
    align-items: center;
    padding: 0 0.5rem;
    color: #666;
  }

  @media (max-width: 768px) {
    .table thead th {
      font-size: 0.8125rem;
      padding: 0.75rem 0.5rem;
    }
    .table tbody td {
      font-size: 0.875rem;
      padding: 0.75rem 0.5rem;
    }
  }
</style>