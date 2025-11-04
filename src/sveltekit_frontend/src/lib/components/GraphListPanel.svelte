<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  
  export let graphs: string[] = [];
  export let selectedGraph: string | null = null;
  export let title: string = "Graphs";
  export let emptyStateMessage: string = "No graphs available";
  export let emptyStateHint: string = "Build graphs first";
  export let isLoading: boolean = false;
  export let iconColor: string = "#3b82f6";
  export let showAttackIcon: boolean = false;
  
  const dispatch = createEventDispatcher();
  
  function handleSelect(graphPath: string) {
    dispatch('select', graphPath);
  }
  
  function formatGraphName(path: string): string {
    const filename = path.split('/').pop()?.replace('.npz', '') || '';
    return filename.replace('inference_', '').replace(/_/g, ' ');
  }
</script>

<div class="left-panel">
  <h2>{title} ({graphs.length})</h2>
  
  {#if isLoading}
    <div class="loading-state">
      <div class="spinner-border"></div>
      <p>Building graphs...</p>
    </div>
  {:else if graphs.length === 0}
    <div class="empty-state">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"/>
        <line x1="12" y1="8" x2="12" y2="12"/>
        <line x1="12" y1="16" x2="12.01" y2="16"/>
      </svg>
      <p>{emptyStateMessage}</p>
      <p class="hint">{emptyStateHint}</p>
    </div>
  {:else}
    <div class="graph-list">
      {#each graphs as graphPath}
        <button
          class="graph-item"
          class:selected={selectedGraph === graphPath}
          on:click={() => handleSelect(graphPath)}
          title={graphPath}
        >
          <div class="graph-icon" style="background-color: {iconColor}" class:attack={showAttackIcon && selectedGraph === graphPath}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="5" cy="5" r="2"/>
              <circle cx="19" cy="5" r="2"/>
              <circle cx="12" cy="19" r="2"/>
              <line x1="6.5" y1="6.5" x2="10.5" y2="17.5"/>
              <line x1="13.5" y1="17.5" x2="17.5" y2="6.5"/>
            </svg>
          </div>
          <div class="graph-info">
            <div class="graph-name">{formatGraphName(graphPath)}</div>
            <div class="graph-path">{graphPath.split('/').pop()}</div>
          </div>
        </button>
      {/each}
    </div>
  {/if}
</div>

<style>
  .left-panel {
    background: #1e1e1e;
    border-radius: 8px;
    border: 1px solid #333;
    overflow: hidden;
  }
  
  h2 {
    font-size: 1.25rem;
    font-weight: 600;
    color: #e0e0e0;
    padding: 1.25rem;
    margin: 0;
    border-bottom: 1px solid #333;
  }
  
  .loading-state,
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem 2rem;
    color: #666;
  }
  
  .empty-state svg {
    width: 64px;
    height: 64px;
    margin-bottom: 1rem;
    opacity: 0.5;
  }
  
  .empty-state p {
    margin: 0.25rem 0;
    font-size: 1rem;
    color: #999;
  }
  
  .empty-state .hint {
    font-size: 0.875rem;
    color: #666;
  }
  
  .spinner-border {
    width: 40px;
    height: 40px;
    border: 3px solid #333;
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-bottom: 1rem;
  }
  
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
  
  .graph-list {
    max-height: 650px;
    overflow-y: auto;
  }
  
  .graph-item {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.875rem 1.25rem;
    border: none;
    border-bottom: 1px solid #333;
    background: #1e1e1e;
    cursor: pointer;
    transition: background 0.2s;
    text-align: left;
  }
  
  .graph-item:hover {
    background: #2b2b2b;
  }
  
  .graph-item.selected {
    background: #0d0d0d;
    border-left: 3px solid #3b82f6;
  }
  
  .graph-icon {
    flex-shrink: 0;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 6px;
    transition: background 0.2s;
  }
  
  .graph-icon.attack {
    background: #ef4444 !important;
  }
  
  .graph-icon svg {
    width: 20px;
    height: 20px;
    color: white;
  }
  
  .graph-info {
    flex: 1;
    min-width: 0;
  }
  
  .graph-name {
    font-weight: 600;
    color: #e0e0e0;
    margin-bottom: 0.25rem;
    text-transform: capitalize;
  }
  
  .graph-path {
    font-size: 0.75rem;
    color: #666;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
</style>