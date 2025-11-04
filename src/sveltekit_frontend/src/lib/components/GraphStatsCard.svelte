<script lang="ts">
  export let stats: Array<{ label: string; value: string | number }> = [];
  export let variant: 'primary' | 'danger' = 'primary';
</script>

<div class="graph-stats">
  {#each stats as stat}
    <div class="stat-card" class:danger={variant === 'danger'}>
      <div class="stat-label">{stat.label}</div>
      <div class="stat-value">{stat.value}</div>
    </div>
  {/each}
</div>

<style>
  .graph-stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
  }
  
  .stat-card {
    background: green;
    padding: 1rem;
    border-radius: 8px;
    color: white;
  }
  
  .stat-card.danger {
    background: #ef4444;
  }
  
  .stat-label {
    font-size: 0.75rem;
    opacity: 0.9;
    margin-bottom: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  
  .stat-value {
    font-size: 1.5rem;
    font-weight: 700;
  }
</style>