<script lang="ts">
  import { apiPost } from "$lib/controllers/client-api";
  import { agents, showAlert } from "$lib/stores/generalStores";
  // Import the SequenceBar
  import SequenceBar from "$lib/components/SequenceBar.svelte";
	import { getAgents } from "$lib/controllers/logs-controller";

  let isRefreshing = false;

  async function handleRefreshToken() {
    isRefreshing = true;
    
    try {
      apiPost('/auth/refresh-token', null);
      console.log('Token refreshed successfully');
      showAlert("Token refreshed successfully", "success", 5000)
      try {
        const res = await getAgents();
        $agents = res.agents?.data?.affected_items ?? [];
      } catch (e: any) {
        showAlert("Error loading agents", "danger", 5000)
      }
    } catch (error) {
      console.error('Failed to refresh token:', error);
      showAlert("Failed to refresh the token", "danger", 5000)
    } finally {
      isRefreshing = false;
    }
  }
</script>

<!-- Navbar -->
<nav class="navbar navbar-expand-lg navbar-dark bg-black shadow-sm px-3">
  <a class="navbar-brand fw-bold text-light" href="/">LogChain Recon</a>
  
  <button
    class="navbar-toggler"
    type="button"
    data-bs-toggle="collapse"
    data-bs-target="#navbarNav"
    aria-controls="navbarNav"
    aria-expanded="false"
    aria-label="Toggle navigation"
  >
    <span class="navbar-toggler-icon"></span>
  </button>

  <div class="collapse navbar-collapse" id="navbarNav">
    
    <!-- Sequence Bar positioned right after the title -->
    <div class="sequence-bar-container ms-lg-4">
      <SequenceBar />
    </div>
    
    <!-- Right-aligned items -->
    <ul class="navbar-nav ms-auto">
      <!-- Empty for now, can add more nav items here if needed -->
    </ul>
    
    <!-- Refresh Token Button -->
    <button
      class="btn btn-refresh ms-lg-3 my-2 my-lg-0"
      on:click={handleRefreshToken}
      disabled={isRefreshing}
      aria-label="Refresh token"
      title="Refresh Token"
    >
      <svg
        class="refresh-icon"
        class:spinning={isRefreshing}
        width="18"
        height="18"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          stroke-width="2"
          d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
        />
      </svg>
      <span class="ms-2">Refresh Token</span>
    </button>
  </div>
</nav>

<style>
  .sequence-bar-container {
    display: flex;
    align-items: center;
  }

  .btn-refresh {
    display: flex;
    align-items: center;
    background-color: #1e1e1e;
    border: 1px solid #333;
    color: #b0c4ff;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    font-size: 0.9rem;
    font-weight: 500;
    transition: all 0.2s ease;
    flex-shrink: 0;
  }

  .btn-refresh:hover:not(:disabled) {
    background-color: #2a2a2a;
    border-color: #3b82f6;
    color: #3b82f6;
  }

  .btn-refresh:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .refresh-icon {
    transition: transform 0.2s ease;
  }

  .refresh-icon.spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }

  /* Responsive: hide text on small screens */
  @media (max-width: 768px) {
    .btn-refresh span {
      display: none;
    }
    
    .btn-refresh {
      padding: 0.5rem;
    }
  }

  /* Mobile layout adjustments */
  @media (max-width: 991.98px) {
    .sequence-bar-container {
      margin-top: 1rem;
      margin-bottom: 0.5rem;
      margin-left: 0 !important;
      width: 100%;
      justify-content: center;
      overflow-x: auto;
      overflow-y: hidden;
    }

    .btn-refresh {
      margin-left: 0 !important;
      width: fit-content;
      margin: 0 auto;
    }
  }
</style>