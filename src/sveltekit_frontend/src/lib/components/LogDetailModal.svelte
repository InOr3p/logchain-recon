<script lang="ts">
  import type { LogItem } from '$lib/schema/models';
  
  export let showModal: boolean = false;
  export let log: LogItem | null = null;
  export let onClose: () => void;
  
  function handleOverlayClick() {
    onClose();
  }
  
  function handleContentClick(event: MouseEvent) {
    event.stopPropagation();
  }
</script>

{#if showModal && log}
  <div class="modal-overlay" on:click={handleOverlayClick}>
    <div class="modal-content" on:click={handleContentClick}>
      <div class="modal-header">
        <h3>Log Details</h3>
        <button class="close-btn" on:click={onClose}>
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        </button>
      </div>
      
      <div class="modal-body">
        <div class="log-details">
          <div class="detail-item">
            <span class="detail-label">Log ID:</span>
            <span class="detail-value">{log.id}</span>
          </div>
          
          <div class="detail-item">
            <span class="detail-label">Rule ID:</span>
            <span class="detail-value">{log.rule_id}</span>
          </div>
          
          <div class="detail-item">
            <span class="detail-label">Rule Description:</span>
            <span class="detail-value">{log.rule_description}</span>
          </div>
          
          <div class="detail-item">
            <span class="detail-label">Rule Level:</span>
            <span class="detail-value">{log.rule_level}</span>
          </div>
          
          <div class="detail-item">
            <span class="detail-label">Fired Times:</span>
            <span class="detail-value">{log.rule_firedtimes || 1}</span>
          </div>
          
          <div class="detail-item">
            <span class="detail-label">Agent IP:</span>
            <span class="detail-value">{log.agent_ip}</span>
          </div>
          
          <div class="detail-item">
            <span class="detail-label">Source IP:</span>
            <span class="detail-value">{log.data_srcip || 'N/A'}</span>
          </div>
          
          <div class="detail-item">
            <span class="detail-label">Created at:</span>
            <span class="detail-value">{log.timestamp}</span>
          </div>
          
          {#if log.rule_groups && log.rule_groups.length > 0}
            <div class="detail-item">
              <span class="detail-label">Rule Groups:</span>
              <div class="detail-tags">
                {#each log.rule_groups as group}
                  <span class="tag">{group}</span>
                {/each}
              </div>
            </div>
          {/if}
          
          {#if log.rule_nist_800_53 && log.rule_nist_800_53.length > 0}
            <div class="detail-item">
              <span class="detail-label">NIST 800-53:</span>
              <div class="detail-tags">
                {#each log.rule_nist_800_53 as nist}
                  <span class="tag">{nist}</span>
                {/each}
              </div>
            </div>
          {/if}
          
          {#if log.rule_gdpr && log.rule_gdpr.length > 0}
            <div class="detail-item">
              <span class="detail-label">GDPR:</span>
              <div class="detail-tags">
                {#each log.rule_gdpr as gdpr}
                  <span class="tag">{gdpr}</span>
                {/each}
              </div>
            </div>
          {/if}
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.75);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    backdrop-filter: blur(4px);
  }
  
  .modal-content {
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 12px;
    width: 90%;
    max-width: 700px;
    max-height: 85vh;
    display: flex;
    flex-direction: column;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
  }
  
  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.5rem;
    border-bottom: 1px solid #2a2a2a;
  }
  
  .modal-header h3 {
    margin: 0;
    font-size: 1.25rem;
    color: #f0f0f0;
  }
  
  .close-btn {
    background: transparent;
    border: none;
    color: #888;
    cursor: pointer;
    padding: 0.25rem;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: color 0.2s ease;
  }
  
  .close-btn:hover {
    color: #f0f0f0;
  }
  
  .modal-body {
    padding: 1.5rem;
    overflow-y: auto;
  }
  
  .log-details {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }
  
  .detail-item {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  
  .detail-label {
    font-size: 0.75rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
  }
  
  .detail-value {
    color: #e0e0e0;
    font-size: 0.95rem;
    font-family: 'Courier New', monospace;
    padding: 0.75rem;
    background: #202020;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    word-break: break-word;
  }
  
  .detail-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }
  
  .tag {
    padding: 0.375rem 0.75rem;
    background: #252525;
    border: 1px solid #333;
    border-radius: 4px;
    color: #b0c4ff;
    font-size: 0.8rem;
    font-family: 'Courier New', monospace;
  }
</style>