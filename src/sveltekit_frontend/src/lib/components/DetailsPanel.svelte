<script lang="ts">
	import { formatDate } from '$lib/utils';
  import { createEventDispatcher } from 'svelte';
  
  export let selectedElement: any = null;
  export let show: boolean = false;
  export let isAttackMode: boolean = false;
  
  const dispatch = createEventDispatcher();
  
  function close() {
    dispatch('close');
  }
  
  function formatProbability(prob: number): string {
    return `${(prob * 100).toFixed(1)}%`;
  }
</script>

{#if show && selectedElement}
  <div class="details-panel">
    <div class="panel-header">
      <h3>
        {selectedElement.type === 'node' ? 'Node Details' : 'Edge Details'}
      </h3>
      <button class="close-btn" on:click={close}>
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
    </div>
    
    <div class="panel-content">
      {#if selectedElement.type === 'node'}
        <!-- Node Details -->
        <div class="detail-section">
          <div class="detail-label">Node ID</div>
          <div class="detail-value">{selectedElement.id}</div>
        </div>
        
        <div class="detail-section">
          <div class="detail-label">Description</div>
          <div class="detail-value">{selectedElement.description}</div>
        </div>
        
        <div class="detail-section">
          <div class="detail-label">Rule ID</div>
          <div class="detail-value">{selectedElement.ruleId}</div>
        </div>
        
        {#if selectedElement.ruleGroups && selectedElement.ruleGroups.length > 0}
          <div class="detail-section">
            <div class="detail-label">Rule Groups</div>
            <div class="tags">
              {#each selectedElement.ruleGroups as group}
                <span class="tag">{group}</span>
              {/each}
            </div>
          </div>
        {/if}
        
        {#if selectedElement.nist && selectedElement.nist.length > 0}
          <div class="detail-section">
            <div class="detail-label">NIST 800-53</div>
            <div class="tags">
              {#each selectedElement.nist as nist}
                <span class="tag nist">{nist}</span>
              {/each}
            </div>
          </div>
        {/if}
        
        {#if isAttackMode && selectedElement.probability > 0}
          <div class="detail-section">
            <div class="detail-label">Max Incident Probability</div>
            <div class="probability-badge" class:high={selectedElement.probability > 0.8}>
              {formatProbability(selectedElement.probability)}
            </div>
          </div>
        {/if}
        
        <div class="detail-section">
          <div class="detail-label">Timestamp</div>
          <div class="detail-value">{formatDate(selectedElement.timestamp)}</div>
        </div>
      {:else}
        <!-- Edge Details -->
        {#if isAttackMode && selectedElement.probability > 0}
          <div class="detail-section">
            <div class="detail-label">Attack Probability</div>
            <div class="probability-badge large" class:high={selectedElement.probability > 0.8}>
              {formatProbability(selectedElement.probability)}
            </div>
          </div>
          
          <div class="detail-section">
            <div class="detail-label">Timestamp</div>
            <div class="detail-value">{formatDate(selectedElement.timestamp)}</div>
          </div>
        {:else}
          <div class="detail-section">
            <div class="detail-label">Edge ID</div>
            <div class="detail-value">{selectedElement.id}</div>
          </div>
        {/if}
        
        <div class="detail-section">
          <div class="detail-label">
            <span class="node-badge source">Source Node</span>
          </div>
          <div class="node-details">
            <div class="node-detail-item">
              <span class="detail-item-label">ID:</span>
              <span class="detail-item-value">{selectedElement.source.id}</span>
            </div>
            <div class="node-detail-item">
              <span class="detail-item-label">Description:</span>
              <span class="detail-item-value">{selectedElement.source.description}</span>
            </div>
            <div class="node-detail-item">
              <span class="detail-item-label">Timestamp:</span>
              <span class="detail-item-value">{formatDate(selectedElement.source.timestamp)}</span>
            </div>
          </div>
        </div>
        
        <div class="detail-section">
          <div class="detail-label">
            <span class="node-badge target">Target Node</span>
          </div>
          <div class="node-details">
            <div class="node-detail-item">
              <span class="detail-item-label">ID:</span>
              <span class="detail-item-value">{selectedElement.target.id}</span>
            </div>
            <div class="node-detail-item">
              <span class="detail-item-label">Description:</span>
              <span class="detail-item-value">{selectedElement.target.description}</span>
            </div>
            <div class="node-detail-item">
              <span class="detail-item-label">Timestamp:</span>
              <span class="detail-item-value">{formatDate(selectedElement.target.timestamp)}</span>
            </div>
          </div>
        </div>
      {/if}
    </div>
  </div>
{/if}

<style>
  .details-panel {
    position: absolute;
    top: 1rem;
    right: 1rem;
    width: 320px;
    background: #1e1e1e;
    border: 1px solid #444;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
    z-index: 1000;
    animation: slideIn 0.2s ease-out;
  }
  
  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translateX(20px);
    }
    to {
      opacity: 1;
      transform: translateX(0);
    }
  }
  
  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem;
    border-bottom: 1px solid #333;
    background: #252525;
    border-radius: 8px 8px 0 0;
  }
  
  .panel-header h3 {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
    color: #e0e0e0;
  }
  
  .close-btn {
    background: none;
    border: none;
    color: #999;
    cursor: pointer;
    padding: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: color 0.2s;
  }
  
  .close-btn:hover {
    color: #e0e0e0;
  }
  
  .panel-content {
    padding: 1rem;
    max-height: 400px;
    overflow-y: auto;
  }
  
  .detail-section {
    margin-bottom: 1rem;
  }
  
  .detail-section:last-child {
    margin-bottom: 0;
  }
  
  .detail-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
  }
  
  .detail-value {
    font-size: 0.9rem;
    color: #e0e0e0;
    background: #0d0d0d;
    padding: 0.5rem;
    border-radius: 4px;
    border: 1px solid #333;
    word-break: break-word;
  }
  
  .tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }
  
  .tag {
    font-size: 0.75rem;
    padding: 0.25rem 0.5rem;
    background: #0d0d0d;
    border: 1px solid #444;
    border-radius: 4px;
    color: #a0a0a0;
  }
  
  .tag.nist {
    background: rgba(59, 130, 246, 0.1);
    border-color: rgba(59, 130, 246, 0.3);
    color: #60a5fa;
  }
  
  .probability-badge {
    display: inline-block;
    font-size: 0.9rem;
    font-weight: 700;
    padding: 0.5rem 0.75rem;
    background: rgba(239, 68, 68, 0.2);
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: 6px;
    color: #fca5a5;
  }
  
  .probability-badge.large {
    font-size: 1.25rem;
    padding: 0.75rem 1rem;
  }
  
  .probability-badge.high {
    background: rgba(239, 68, 68, 0.3);
    border-color: rgba(239, 68, 68, 0.6);
    color: #ef4444;
    animation: pulse 2s ease-in-out infinite;
  }
  
  @keyframes pulse {
    0%, 100% {
      opacity: 1;
    }
    50% {
      opacity: 0.7;
    }
  }
  
  .node-details {
    background: #0d0d0d;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 0.75rem;
  }
  
  .node-detail-item {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid #252525;
  }
  
  .node-detail-item:last-child {
    border-bottom: none;
    padding-bottom: 0;
  }
  
  .node-detail-item:first-child {
    padding-top: 0;
  }
  
  .detail-item-label {
    font-size: 0.7rem;
    color: #999;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  
  .detail-item-value {
    font-size: 0.85rem;
    color: #e0e0e0;
    word-break: break-word;
  }

  .node-badge {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 0.5rem;
  }
  
  .node-badge.source {
    background: rgba(59, 130, 246, 0.2);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.3);
  }
  
  .node-badge.target {
    background: rgba(34, 197, 94, 0.2);
    color: #86efac;
    border: 1px solid rgba(34, 197, 94, 0.3);
  }
</style>