<!-- ClassificationPanel.svelte -->
<script lang="ts">
	import LogDetailModal from '$lib/components/LogDetailModal.svelte';
  import { classifyLogs } from '$lib/controllers/logs-controller';
  import type { LogItem, Prediction } from '$lib/schema/models';
  import { selectedLogs, showAlert } from '$lib/stores/generalStores';
  
  let predictions: Prediction[] = [];
  let isLoading = false;
  let errorMessage = '';
  let hoveredRowId: string | null = null;
  
  // Modal state
  let showModal = false;
  let selectedLog: LogItem | null = null;
  
  // Pagination
  let currentPage = 1;
  let itemsPerPage = 10;
  
  $: totalPages = Math.ceil(predictions.length / itemsPerPage);
  $: paginatedPredictions = predictions.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );
  $: attackCount = predictions.filter(p => p.is_attack).length;
  
  // Check if a log is currently selected
  $: isLogSelected = (logId: string) => {
    return $selectedLogs.some(log => log.id === logId);
  };
  
  function goToPage(page: number) {
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
  
  function viewCompleteLog(logId: string, event?: MouseEvent) {
    // Stop propagation to prevent row click
    if (event) {
      event.stopPropagation();
    }
    
    // Find the log from our classified logs list
    const log = classifiedLogs.find(l => l.id === logId);
    if (log) {
      selectedLog = log;
      showModal = true;
    }
  }

  function getLogDescriptionById(logId: string) {
    const log = classifiedLogs.find(l => l.id === logId);
    if (log) {
      return log["rule_description"]
    }
    return "N/A"
  }
  
  function closeModal() {
    showModal = false;
    selectedLog = null;
  }
  
  // Store all classified logs separately so we can reference them
  let classifiedLogs: LogItem[] = [];
  
  function toggleLogSelection(logId: string, event: MouseEvent) {
    // Prevent row click when clicking the view button or any child element of it
    const target = event.target as HTMLElement;
    if (target.closest('.view-log-btn') || target.closest('.actions-cell')) {
      return;
    }
    
    // Find the log from our classified logs list
    const log = classifiedLogs.find(l => l.id === logId);
    if (!log) return;
    
    const isCurrentlySelected = isLogSelected(logId);
    
    if (isCurrentlySelected) {
      // Remove from selection
      selectedLogs.update(logs => logs.filter(l => l.id !== logId));
    } else {
      // Add to selection
      selectedLogs.update(logs => [...logs, log]);
    }
  }
  
  function selectAllAttacks() {
    // Get all attack predictions
    const attackPredictions = predictions.filter(p => p.is_attack);
    
    // Get the corresponding logs from our classified logs
    const attackLogs = attackPredictions
      .map(pred => classifiedLogs.find(log => log.id === pred.id))
      .filter((log): log is LogItem => log !== undefined);
    
    // Update selectedLogs with unique logs
    selectedLogs.update(logs => {
      const existingIds = new Set(logs.map(l => l.id));
      const newLogs = attackLogs.filter(log => !existingIds.has(log.id));
      return [...logs, ...newLogs];
    });
  }
  
  async function handleClassify() {
    if ($selectedLogs.length === 0) {
      errorMessage = 'Please select one or more logs to classify.';
      showAlert(errorMessage, "danger", 5000);
      return;
    }
    
    isLoading = true;
    errorMessage = '';
    predictions = [];
    currentPage = 1;
    
    try {
      const logsToClassify: LogItem[] = $selectedLogs.map(log => ({
        id: log.id,
        rule_id: log.rule_id,
        rule_groups: log.rule_groups || [],
        rule_nist_800_53: log.rule_nist_800_53 || [],
        rule_gdpr: log.rule_gdpr || [],
        rule_level: log.rule_level,
        rule_firedtimes: log.rule_firedtimes || 1,
        agent_ip: log.agent_ip,
        data_srcip: log.data_srcip,
        timestamp: log.timestamp,
        unix_timestamp: log.timestamp,
        rule_description: log.rule_description,
      }));
      
      // Store the classified logs for later reference
      classifiedLogs = logsToClassify;
      
      const results = await classifyLogs(logsToClassify);
      predictions = results;
      
    } catch (error) {
      if (error instanceof Error) {
        errorMessage = error.message;
      } else {
        errorMessage = 'An unknown error occurred.';
      }
      showAlert(errorMessage, "danger", 5000);
    } finally {
      isLoading = false;
      $selectedLogs = [];
    }
  }
</script>

<div class="classification-panel">
  <!-- Header -->
  <div class="header">
    <h1>Classify logs</h1>
    <p class="subtitle">Classify logs as <code>benign</code> or <code>attack</code> by using XGBoost model</p>
  </div>

  <button class="btn btn-primary mt-4" on:click={handleClassify} disabled={isLoading}>
    {#if isLoading}
      <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
      Classifying...
    {:else}
      Classify Selected Logs ({$selectedLogs.length})
    {/if}
  </button>
  
  {#if predictions.length > 0}
    <div class="results">
      <div class="results-header">
        <h3>Classification Results</h3>
        <div class="stats">
          <span class="results-count">{predictions.length} logs analyzed</span>
          <span class="attack-count" class:has-attacks={attackCount > 0}>
            {attackCount} attack{attackCount !== 1 ? 's' : ''} detected
          </span>
          {#if attackCount > 0}
            <button class="select-attacks-btn" on:click={selectAllAttacks}>
              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="20 6 9 17 4 12"></polyline>
              </svg>
              Select All Attacks
            </button>
          {/if}
        </div>
      </div>
      
      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th style="width: 40px;"></th>
              <th>Log ID</th>
              <th>Description</th>
              <th>Status</th>
              <th>Confidence</th>
              <th style="width: 150px;">Actions</th>
            </tr>
          </thead>
          <tbody>
            {#each paginatedPredictions as pred}
              <tr
                class:selected={isLogSelected(pred.id)}
                on:mouseenter={() => hoveredRowId = pred.id}
                on:mouseleave={() => hoveredRowId = null}
                on:click={(e) => toggleLogSelection(pred.id, e)}
              >
                <td class="checkbox-cell">
                  <div class="custom-checkbox" class:checked={isLogSelected(pred.id)}>
                    {#if isLogSelected(pred.id)}
                      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="20 6 9 17 4 12"></polyline>
                      </svg>
                    {/if}
                  </div>
                </td>
                <td class="log-id">{pred.id}</td>
                <td>{getLogDescriptionById(pred.id)}</td>
                <td>
                  <span class="badge" class:attack={pred.is_attack} class:safe={!pred.is_attack}>
                    {pred.prediction_label}
                  </span>
                </td>
                <td>
                  <div class="confidence-wrapper">
                    <div class="confidence-bar">
                      <div 
                        class="confidence-fill" 
                        class:attack={pred.is_attack}
                        style="width: {pred.prediction_score * 100}%"
                      ></div>
                    </div>
                    <span class="confidence-text">{(pred.prediction_score * 100).toFixed(1)}%</span>
                  </div>
                </td>
                <td class="actions-cell">
                  <button 
                    class="view-log-btn"
                    class:visible={hoveredRowId === pred.id}
                    on:click={(e) => viewCompleteLog(pred.id, e)}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                      <circle cx="12" cy="12" r="3"></circle>
                    </svg>
                    View Log
                  </button>
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
      
      {#if totalPages > 1}
        <div class="pagination">
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
                  on:click={() => goToPage(i + 1)}
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
  {/if}
</div>

<!-- Use the modal component -->
<LogDetailModal 
  bind:showModal 
  log={selectedLog} 
  onClose={closeModal} 
/>

<style>
  .classification-panel {
    padding: 1.5rem;
  }
  
  .results {
    margin-top: 1.5rem;
  }
  
  .results-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }
  
  .results-header h3 {
    margin: 0;
    font-size: 1.25rem;
    color: #f0f0f0;
  }
  
  .stats {
    display: flex;
    gap: 0.75rem;
    align-items: center;
  }
  
  .results-count {
    font-size: 0.875rem;
    color: #888;
    padding: 0.25rem 0.75rem;
    background: #252525;
    border-radius: 12px;
  }
  
  .attack-count {
    font-size: 0.875rem;
    color: #86efac;
    padding: 0.25rem 0.75rem;
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 12px;
    font-weight: 500;
  }
  
  .attack-count.has-attacks {
    color: #fca5a5;
    background: rgba(239, 68, 68, 0.1);
    border-color: rgba(239, 68, 68, 0.3);
  }
  
  .select-attacks-btn {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 6px;
    color: #fca5a5;
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  
  .select-attacks-btn:hover {
    background: rgba(239, 68, 68, 0.25);
    border-color: rgba(239, 68, 68, 0.5);
  }
  
  .table-container {
    overflow-x: auto;
    border-radius: 8px;
    background: #202020;
    border: 1px solid #333;
  }
  
  table {
    width: 100%;
    border-collapse: collapse;
  }
  
  thead {
    background: #252525;
  }
  
  th {
    padding: 1rem;
    text-align: left;
    font-weight: 600;
    font-size: 0.875rem;
    color: #aaa;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  
  tbody tr {
    border-bottom: 1px solid #2a2a2a;
    transition: background 0.2s ease;
    cursor: pointer;
  }
  
  tbody tr:hover {
    background: #252525;
  }
  
  tbody tr.selected {
    background: rgba(59, 130, 246, 0.1);
    border-color: rgba(59, 130, 246, 0.3);
  }
  
  tbody tr.selected:hover {
    background: rgba(59, 130, 246, 0.15);
  }
  
  tbody tr:last-child {
    border-bottom: none;
  }
  
  td {
    padding: 1rem;
    font-size: 0.9rem;
  }
  
  .checkbox-cell {
    padding: 1rem 0.5rem 1rem 1rem;
  }
  
  .custom-checkbox {
    width: 20px;
    height: 20px;
    border: 2px solid #444;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
    background: #1a1a1a;
  }
  
  .custom-checkbox.checked {
    background: #3b82f6;
    border-color: #3b82f6;
  }
  
  .custom-checkbox svg {
    color: white;
  }
  
  tbody tr:hover .custom-checkbox:not(.checked) {
    border-color: #666;
    background: #252525;
  }
  
  .log-id {
    font-family: 'Courier New', monospace;
    color: #b0b0b0;
  }
  
  .badge {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
  }
  
  .badge.attack {
    background: rgba(239, 68, 68, 0.15);
    color: #fca5a5;
    border: 1px solid rgba(239, 68, 68, 0.3);
  }
  
  .badge.safe {
    background: rgba(34, 197, 94, 0.15);
    color: #86efac;
    border: 1px solid rgba(34, 197, 94, 0.3);
  }
  
  .confidence-wrapper {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }
  
  .confidence-bar {
    flex: 1;
    height: 8px;
    background: #2a2a2a;
    border-radius: 4px;
    overflow: hidden;
  }
  
  .confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #22c55e, #16a34a);
    transition: width 0.3s ease;
  }
  
  .confidence-fill.attack {
    background: linear-gradient(90deg, #ef4444, #dc2626);
  }
  
  .confidence-text {
    min-width: 50px;
    text-align: right;
    font-weight: 600;
    color: #d0d0d0;
    font-size: 0.875rem;
  }
  
  .view-log-btn {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.375rem 0.75rem;
    background: #2a2a2a;
    border: 1px solid #333;
    border-radius: 4px;
    color: #b0c4ff;
    font-size: 0.8rem;
    cursor: pointer;
    transition: all 0.2s ease;
    opacity: 0;
    pointer-events: none;
    white-space: nowrap;
  }
  
  .view-log-btn.visible {
    opacity: 1;
    pointer-events: auto;
  }
  
  .view-log-btn:hover {
    background: #333;
    border-color: #3b82f6;
    color: #3b82f6;
  }
  
  .pagination {
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

  h1 {
    font-size: 2.5rem;
    font-weight: 700;
    color: #e0e0e0;
    margin: 0 0 0.5rem 0;
  }
</style>