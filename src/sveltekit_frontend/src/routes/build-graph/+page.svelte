<script lang="ts">
  import { buildGraphs, getGraphData } from '$lib/controllers/graphs-controller';
  import type { GraphData } from '$lib/schema/models';
  import { graphFiles, selectedLogs, showAlert, logs } from '$lib/stores/generalStores';
  import { tick } from 'svelte';
  
	import GraphVisualizer from '$lib/components/GraphVisualizer.svelte';
	import GraphListPanel from '$lib/components/GraphListPanel.svelte';
	import GraphStatsCard from '$lib/components/GraphStatsCard.svelte';
	import DetailsPanel from '$lib/components/DetailsPanel.svelte';

  // State management
  let selectedGraph: string | null = null;
  let graphData: GraphData | null = null;
  let isLoading = false;
  let isLoadingGraph = false;
  let error: string | null = null;
  
  // Details panel
  let selectedElement: any = null;
  let showDetailsPanel = false;
  
  let cytoscapeVisualizer: GraphVisualizer;
  
  // Reactive statements
  $: selectedLogsCount = $selectedLogs.length;
  $: statsData = graphData ? [
    { label: 'Agent IP', value: graphData.agent_ip },
    { label: 'Nodes', value: graphData.num_nodes },
    { label: 'Edges', value: graphData.num_edges }
  ] : [];
  
  // Build graphs from selected logs
  async function handleBuildGraphs() {
    if ($selectedLogs.length === 0) {
      error = 'Please select logs to build graphs';
      showAlert(error, "danger", 5000);
      return;
    }
    
    isLoading = true;
    error = null;
    $graphFiles = [];
    selectedGraph = null;
    graphData = null;
    
    try {
      const response = await buildGraphs($selectedLogs);
      $graphFiles = response.graph_files || [];
      const message = response.message || `Built ${$graphFiles.length} graphs`;
      
      showAlert(message, "success", 5000);

      // Auto-select first graph
      if ($graphFiles.length > 0) {
        await selectGraph($graphFiles[0]);
      }
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to build graphs';
      console.error('Error building graphs:', err);
      showAlert(error, "danger", 5000);
    } finally {
      isLoading = false;
    }
  }
  
  // Load graph data from .npz file path
  async function selectGraph(graphPath: string) {
    selectedGraph = graphPath;
    graphData = null;
    isLoadingGraph = true;
    showDetailsPanel = false;
    selectedElement = null;
    
    try {
      graphData = await getGraphData(graphPath);
      await tick();
    } catch (err) {
      error = `Failed to load graph: ${err instanceof Error ? err.message : 'Unknown error'}`;
      console.error('Error loading graph:', err);
      showAlert(error, "danger", 5000);
    } finally {
      isLoadingGraph = false;
    }
  }
  
  // Handle node/edge clicks from Cytoscape
  function handleNodeClick(event: CustomEvent) {
    const node = event.detail;
    const nodeId = node.data('id');
    const log = $selectedLogs.find(l => l.id === nodeId);
    
    selectedElement = {
      type: 'node',
      id: nodeId,
      label: node.data('label') || nodeId,
      description: log?.rule_description || 'N/A',
      ruleId: log?.rule_id || 'N/A',
      ruleGroups: log?.rule_groups || [],
      nist: log?.rule_nist_800_53 || [],
      gdpr: log?.rule_gdpr || [],
      timestamp: log?.timestamp || 'N/A',
      probability: 0
    };
    
    showDetailsPanel = true;
    highlightNode(node);
  }
  
  function handleEdgeClick(event: CustomEvent) {
    const edge = event.detail;
    const sourceNode = edge.source();
    const targetNode = edge.target();
    
    const sourceLog = $selectedLogs.find(log => log.id === sourceNode.data('id'));
    const targetLog = $selectedLogs.find(log => log.id === targetNode.data('id'));
    
    selectedElement = {
      type: 'edge',
      id: edge.id(),
      source: {
        id: sourceNode.id(),
        label: sourceNode.data('label'),
        description: sourceLog?.rule_description || 'N/A',
        timestamp: sourceLog?.timestamp || 'N/A'
      },
      target: {
        id: targetNode.id(),
        label: targetNode.data('label'),
        description: targetLog?.rule_description || 'N/A',
        timestamp: targetLog?.timestamp || 'N/A'
      }
    };
    
    showDetailsPanel = true;
  }
  
  function handleBackgroundClick() {
    closeDetailsPanel();
  }
  
  function closeDetailsPanel() {
    showDetailsPanel = false;
    selectedElement = null;
    if (cytoscapeVisualizer) {
      cytoscapeVisualizer.resetStyles(false);
    }
  }
  
  function highlightNode(node: any) {
    // Highlight logic could be moved to CytoscapeVisualizer
  }
</script>

<div class="container">
  <!-- Header -->
  <div class="header">
    <h1>Graph Builder</h1>
    <p class="subtitle">Build, store and visualize log graphs</p>
  </div>
  
  <!-- Controls -->
  <div class="controls">
    <div class="control-group">
      <span class="log-count">
        {selectedLogsCount} {selectedLogsCount === 1 ? 'log' : 'logs'} selected
      </span>
    </div>
    
    <button 
      on:click={handleBuildGraphs} 
      class="btn btn-primary"
      disabled={isLoading || selectedLogsCount === 0}
    >
      {isLoading ? 'Building Graphs...' : 'Build Graphs'}
    </button>
  </div>
  
  <!-- Main Content -->
  <div class="main-content">
    <!-- Left Panel -->
    <GraphListPanel
      graphs={$graphFiles}
      {selectedGraph}
      title="Generated Graphs"
      emptyStateMessage="No graphs built yet"
      emptyStateHint="Select logs and click 'Build Graphs' to start"
      {isLoading}
      iconColor="green"
      on:select={(e) => selectGraph(e.detail)}
    />
    
    <!-- Right Panel -->
    <div class="right-panel">
      <h2>Graph Visualization</h2>
      
      {#if !selectedGraph}
        <div class="empty-state">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
          </svg>
          <p>Select a graph to visualize</p>
        </div>
      {:else if isLoadingGraph}
        <div class="loading-state">
          <div class="spinner-border"></div>
          <p>Loading graph data...</p>
        </div>
      {:else if graphData}
        <div class="graph-visualization">
          <!-- Graph Stats -->
          <GraphStatsCard stats={statsData} variant="primary" />
          
          <!-- Graph Visualization -->
          <div class="graph-canvas-wrapper">
            <div class="graph-canvas">
              <GraphVisualizer
                bind:this={cytoscapeVisualizer}
                {graphData}
                isAttackMode={false}
                on:nodeClick={handleNodeClick}
                on:edgeClick={handleEdgeClick}
                on:backgroundClick={handleBackgroundClick}
              />
            </div>
            
            <!-- Details Panel -->
            <DetailsPanel
              bind:show={showDetailsPanel}
              {selectedElement}
              isAttackMode={false}
              on:close={closeDetailsPanel}
            />
          </div>
        </div>
      {/if}
    </div>
  </div>
</div>

<style>
  .container {
    max-width: 1600px;
    margin: 0 auto;
    padding: 2rem;
    font-family: "Inter", sans-serif;
  }
  
  .header {
    margin-bottom: 2rem;
  }
  
  h1 {
    font-size: 2.5rem;
    font-weight: 700;
    color: #e0e0e0;
    margin: 0 0 0.5rem 0;
  }
  
  .subtitle {
    color: #999;
    font-size: 1rem;
    margin: 0;
  }
  
  .controls {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
    padding: 1rem;
    background: #1e1e1e;
    border-radius: 8px;
    border: 1px solid #333;
  }
  
  .control-group {
    display: flex;
    gap: 0.75rem;
    align-items: center;
  }
  
  .log-count {
    color: #e0e0e0;
    font-weight: 500;
    padding: 0.5rem 1rem;
    background: #0d0d0d;
    border-radius: 6px;
    border: 1px solid #333;
  }
  
  .btn {
    padding: 0.625rem 1.25rem;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 0.95rem;
  }
  
  .btn-primary {
    background-color: green;
    border-color: green;
    color: #fff;
  }
  
  .btn-primary:hover:not(:disabled) {
    background-color: darkgreen;
    border-color: darkgreen;
  }
  
  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
 
  .main-content {
    display: grid;
    grid-template-columns: 350px 1fr;
    gap: 1.5rem;
    min-height: 600px;
  }
  
  .right-panel {
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
  
  .empty-state,
  .loading-state {
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
  
  .graph-visualization {
    padding: 1.5rem;
  }
  
  .graph-canvas-wrapper {
    position: relative;
  }
  
  .graph-canvas {
    background: #0d0d0d;
    border-radius: 8px;
    padding: 0;
    margin-bottom: 2rem;
    min-height: 500px;
    border: 1px solid #333;
  }
</style>