<script lang="ts">
  export let logs: any[] = [];
  
  const ITEMS_PER_PAGE = 20;
  let currentPage = 1;
  
  $: totalPages = Math.ceil(logs.length / ITEMS_PER_PAGE);
  $: paginatedLogs = logs.slice(
    (currentPage - 1) * ITEMS_PER_PAGE,
    currentPage * ITEMS_PER_PAGE
  );
  
  function changePage(newPage: number) {
    currentPage = Math.max(1, Math.min(newPage, totalPages));
  }
</script>

<div class="container-fluid my-4">
  <!-- Table Card -->
  <div class="table-card">
    <div class="table-responsive">
      <table class="table table-dark table-hover align-middle mb-0">
        <thead>
          <tr>
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
        <tbody>
          {#if paginatedLogs.length > 0}
            {#each paginatedLogs as log}
              <tr>
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
                <td class="text-muted small">{log.rule_nist_800_53 ?? "—"}</td>
                <td class="text-muted small">{log.rule_gdpr ?? "—"}</td>
              </tr>
            {/each}
          {:else}
            <tr>
              <td colspan="9" class="text-center empty-state">
                <div class="py-4">
                  <svg class="empty-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
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
  <div class="d-flex justify-content-between align-items-center mt-4">
    <div class="page-info">
      Page <strong>{currentPage}</strong> of <strong>{totalPages}</strong>
      <span class="text-muted ms-2">({logs.length} total logs)</span>
    </div>
    <nav aria-label="Table pagination">
      <ul class="pagination mb-0">
        <li class="page-item {currentPage === 1 ? 'disabled' : ''}">
          <button
            class="page-link"
            on:click={() => changePage(currentPage - 1)}
            disabled={currentPage === 1}
            aria-label="Previous page"
          >
            <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
              <path fill-rule="evenodd" d="M11.354 1.646a.5.5 0 0 1 0 .708L5.707 8l5.647 5.646a.5.5 0 0 1-.708.708l-6-6a.5.5 0 0 1 0-.708l6-6a.5.5 0 0 1 .708 0z"/>
            </svg>
          </button>
        </li>
        {#each Array(totalPages) as _, i}
          <li class="page-item {currentPage === i + 1 ? 'active' : ''}">
            <button
              class="page-link"
              on:click={() => changePage(i + 1)}
              aria-label="Page {i + 1}"
              aria-current={currentPage === i + 1 ? 'page' : undefined}
            >
              {i + 1}
            </button>
          </li>
        {/each}
        <li class="page-item {currentPage === totalPages ? 'disabled' : ''}">
          <button
            class="page-link"
            on:click={() => changePage(currentPage + 1)}
            disabled={currentPage === totalPages}
            aria-label="Next page"
          >
            <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
              <path fill-rule="evenodd" d="M4.646 1.646a.5.5 0 0 1 .708 0l6 6a.5.5 0 0 1 0 .708l-6 6a.5.5 0 0 1-.708-.708L10.293 8 4.646 2.354a.5.5 0 0 1 0-.708z"/>
            </svg>
          </button>
        </li>
      </ul>
    </nav>
  </div>
</div>

<style>
  /* Table Card Container */
  .table-card {
    background: #1e1e1e;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    overflow: hidden;
  }

  /* Table Base Styles */
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

  /* Table Header */
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

  /* Table Body */
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

  /* Hover Effect */
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

  /* Typography Utilities */
  .font-mono {
    font-family: 'Courier New', monospace;
    font-size: 0.875rem;
    color: #e0e0e0 !important;
  }
  
  .text-muted {
    color: #a0a0a0 !important;
  }

  /* Badge Styles */
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

  /* Level Color Coding */
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

  /* Empty State */
  .empty-state {
    color: #6b7280;
  }

  .empty-icon {
    width: 48px;
    height: 48px;
    opacity: 0.5;
    margin: 0 auto;
  }

  /* Pagination Info */
  .page-info {
    color: #9ca3af;
    font-size: 0.9375rem;
  }

  .page-info strong {
    color: #e0e0e0;
  }

  /* Pagination Buttons */
  .pagination {
    gap: 0.25rem;
  }

  .page-link {
    background: #1e1e1e;
    border: 1px solid #333;
    color: #b0c4ff;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    transition: all 0.15s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 2.5rem;
  }

  .page-link:hover:not(:disabled) {
    background: #2a2a2a;
    border-color: #3b82f6;
    color: #3b82f6;
  }

  .page-item.active .page-link {
    background: #3b82f6;
    border-color: #3b82f6;
    color: #fff;
    font-weight: 600;
  }

  .page-link:disabled {
    background: #1a1a1a;
    border-color: #2a2a2a;
    color: #4b5563;
    cursor: not-allowed;
    opacity: 0.5;
  }

  /* Responsive Table */
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