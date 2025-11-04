<script lang="ts">
  import { predictAttackGraph, getGraphData } from '$lib/controllers/graphs-controller';
  import type { AttackGraphData, GraphData } from '$lib/schema/models';
  import { addPredictedGraph, graphFiles, logs, showAlert } from '$lib/stores/generalStores';
  import { tick } from 'svelte';

  import GraphVisualizer from '$lib/components/GraphVisualizer.svelte';
  import GraphListPanel from '$lib/components/GraphListPanel.svelte';
  import GraphStatsCard from '$lib/components/GraphStatsCard.svelte';
  import DetailsPanel from '$lib/components/DetailsPanel.svelte';
  
  // State management
  let selectedGraphPath: string | null = null;
  let graphData: GraphData | null = null;
  let attackGraphData: AttackGraphData | null = null;
  let isLoadingGraph = false;
  let isLoadingPrediction = false;
  let error: string | null = null;
  let threshold: number = 0.9;
  
  // Selected node/edge for details
  let selectedElement: any = null;
  let showDetailsPanel = false;
  
  let cytoscapeVisualizer: GraphVisualizer;
  
  // Reactive statements
  $: availableGraphs = $graphFiles;
  $: availableLogsCount = $logs.length;
  $: statsData = attackGraphData ? [
    { label: 'Attack Nodes', value: attackGraphData.important_nodes || attackGraphData.total_nodes },
    { label: 'Attack Edges', value: attackGraphData.important_edges || attackGraphData.total_edges },
    { label: 'Threshold', value: formatProbability(threshold) }
  ] : graphData ? [
    { label: 'Agent IP', value: graphData.agent_ip },
    { label: 'Nodes', value: graphData.num_nodes },
    { label: 'Edges', value: graphData.num_edges }
  ] : [];
  
  // Select a graph and load its data
  async function selectGraph(graphPath: string) {
    selectedGraphPath = graphPath;
    graphData = null;
    attackGraphData = null;
    isLoadingGraph = true;
    showDetailsPanel = false;
    selectedElement = null;
    error = null;
    
    try {
      graphData = await getGraphData(graphPath);
      await tick();
      showAlert('Graph loaded successfully', "success", 3000);
    } catch (err) {
      error = `Failed to load graph: ${err instanceof Error ? err.message : 'Unknown error'}`;
      console.error('Error loading graph:', err);
      showAlert(error, "danger", 5000);
    } finally {
      isLoadingGraph = false;
    }
  }
  
  // Get logs that are part of the current graph
  function getGraphLogs(): any[] {
    if (!graphData || !graphData.log_ids) {
      return [];
    }
    
    const logIdsInGraph = new Set(graphData.log_ids);
    const graphLogs = $logs.filter(log => logIdsInGraph.has(log.id));
    
    console.log('Graph log IDs:', Array.from(logIdsInGraph));
    console.log('Found logs:', graphLogs.length);
    
    return graphLogs;
  }
  
  // Predict attack graph
  async function handlePredictAttacks() {
    if (!selectedGraphPath) {
      error = 'Please select a graph first';
      showAlert(error, "danger", 5000);
      return;
    }
    
    if (!graphData) {
      error = 'Graph data not loaded';
      showAlert(error, "danger", 5000);
      return;
    }
    
    const graphLogs = getGraphLogs();
    
    if (graphLogs.length === 0) {
      error = 'No logs found for this graph';
      showAlert(error, "warning", 5000);
      return;
    }
    
    isLoadingPrediction = true;
    error = null;
    attackGraphData = null;
    showDetailsPanel = false;
    selectedElement = null;
    
    try {
      console.log(`Predicting attacks for ${selectedGraphPath} with ${graphLogs.length} logs`);
      
      let summarize = true;

      const response = await predictAttackGraph(
        selectedGraphPath,
        graphLogs,
        threshold,
        summarize
      );
      
      if (response.success && response.graph) {
        attackGraphData = response.graph;
        const message = response.message || 'Attack prediction completed';
        showAlert(message, "success", 5000);
        
        if (selectedGraphPath) {
          addPredictedGraph(selectedGraphPath, response.graph);
        }
        
        await tick();
      } else {
        error = response.message || 'No attack edges detected above threshold';
        showAlert(error, "info", 5000);
        attackGraphData = null;
      }
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to predict attacks';
      console.error('Error predicting attacks:', err);
      showAlert(error, "danger", 5000);
      attackGraphData = null;
    } finally {
      isLoadingPrediction = false;
    }
  }
  
  // Reset to original graph view
  function resetToOriginalGraph() {
    attackGraphData = null;
    showDetailsPanel = false;
    selectedElement = null;
    
    if (cytoscapeVisualizer) {
      cytoscapeVisualizer.resetStyles(false);
    }
  }
  
  // Handle node click
  function handleNodeClick(event: CustomEvent) {
    const node = event.detail;
    const nodeId = node.data('id');
    const nodeData = node.data();
    
    if (attackGraphData) {
      const attackNode = attackGraphData.sample_nodes?.find(n => n.id === nodeId);
      
      selectedElement = {
        type: 'node',
        id: nodeId,
        label: nodeData.label || nodeId,
        description: attackNode?.description || 'N/A',
        ruleId: attackNode?.rule_id || 'N/A',
        ruleGroups: attackNode?.rule_groups || [],
        nist: attackNode?.nist_800_53 || [],
        gdpr: attackNode?.gdpr || [],
        timestamp: attackNode?.timestamp || 'N/A',
        probability: attackNode?.max_incident_prob || 0
      };
    } else {
      const log = $logs.find(l => l.id === nodeId);
      
      selectedElement = {
        type: 'node',
        id: nodeId,
        label: nodeData.label || nodeId,
        description: log?.rule_description || 'N/A',
        ruleId: log?.rule_id || 'N/A',
        ruleGroups: log?.rule_groups || [],
        nist: log?.rule_nist_800_53 || [],
        gdpr: log?.rule_gdpr || [],
        timestamp: log?.timestamp || 'N/A',
        probability: 0
      };
    }
    
    showDetailsPanel = true;
  }
  
  // Handle edge click
  function handleEdgeClick(event: CustomEvent) {
    const edge = event.detail;
    const edgeData = edge.data();
    const sourceNode = edge.source();
    const targetNode = edge.target();
    
    if (attackGraphData) {
      const attackEdge = attackGraphData.sample_edges?.find(
        e => e.source_log_id === sourceNode.data('id') && 
             e.dest_log_id === targetNode.data('id')
      );
      
      const sourceNodeData = attackGraphData.sample_nodes?.find(n => n.id === sourceNode.data('id'));
      const targetNodeData = attackGraphData.sample_nodes?.find(n => n.id === targetNode.data('id'));
      
      selectedElement = {
        type: 'edge',
        id: edgeData.id,
        source: {
          id: sourceNode.data('id'),
          label: sourceNode.data('label') || sourceNode.data('id'),
          description: sourceNodeData?.description || 'N/A',
          timestamp: sourceNodeData?.timestamp || 'N/A'
        },
        target: {
          id: targetNode.data('id'),
          label: targetNode.data('label') || targetNode.data('id'),
          description: targetNodeData?.description || 'N/A',
          timestamp: targetNodeData?.timestamp || 'N/A'
        },
        probability: attackEdge?.edge_prob || 0,
        timestamp: attackEdge?.timestamp || 'N/A'
      };
    } else {
      const sourceLog = $logs.find(l => l.id === sourceNode.data('id'));
      const targetLog = $logs.find(l => l.id === targetNode.data('id'));
      
      selectedElement = {
        type: 'edge',
        id: edgeData.id,
        source: {
          id: sourceNode.data('id'),
          label: sourceNode.data('label') || sourceNode.data('id'),
          description: sourceLog?.rule_description || 'N/A',
          timestamp: sourceLog?.timestamp || 'N/A'
        },
        target: {
          id: targetNode.data('id'),
          label: targetNode.data('label') || targetNode.data('id'),
          description: targetLog?.rule_description || 'N/A',
          timestamp: targetLog?.timestamp || 'N/A'
        },
        probability: 0,
        timestamp: 'N/A'
      };
    }
    
    showDetailsPanel = true;
  }
  
  function handleBackgroundClick() {
    closeDetailsPanel();
  }
  
  // Close details panel
  function closeDetailsPanel() {
    showDetailsPanel = false;
    selectedElement = null;
    
    if (cytoscapeVisualizer) {
      cytoscapeVisualizer.resetStyles(!!attackGraphData);
    }
  }
  
  // Format probability as percentage
  function formatProbability(prob: number): string {
    return `${(prob * 100).toFixed(1)}%`;
  }
</script>

<div class="container">
  <!-- Header -->
  <div class="header">
    <h1>Attack Sequence Predictor</h1>
    <p class="subtitle">Predict and visualize attack sequences from log graphs</p>
  </div>
  
  <!-- Controls -->
  <div class="controls">
    <div class="control-group">
      <span class="log-count">
        {availableLogsCount} {availableLogsCount === 1 ? 'log' : 'logs'} available
      </span>
      
      {#if graphData}
        <span class="graph-log-count">
          {getGraphLogs().length} logs in selected graph
        </span>
      {/if}
      
      <div class="threshold-control">
        <label for="threshold">Threshold:</label>
        <input 
          type="range" 
          id="threshold" 
          bind:value={threshold} 
          min="0.5" 
          max="1.0" 
          step="0.05"
          disabled={isLoadingPrediction}
        />
        <span class="threshold-value">{threshold.toFixed(2)}</span>
      </div>
    </div>
    
    <div class="button-group">
      {#if attackGraphData}
        <button 
          on:click={resetToOriginalGraph} 
          class="btn btn-secondary"
          disabled={isLoadingPrediction}
        >
          Reset View
        </button>
      {/if}
      
      <button 
        on:click={handlePredictAttacks} 
        class="btn btn-danger"
        disabled={isLoadingPrediction || !selectedGraphPath || !graphData}
      >
        {isLoadingPrediction ? 'Predicting...' : 'Predict Attacks'}
      </button>
    </div>
  </div>
  
  <!-- Main Content -->
  <div class="main-content">
    <!-- Left Panel: Graph Selection -->
    <GraphListPanel
      graphs={availableGraphs}
      selectedGraph={selectedGraphPath}
      title="Available Graphs"
      emptyStateMessage="No graphs available"
      emptyStateHint="Build graphs first in the Graph Builder"
      isLoading={false}
      iconColor="#3b82f6"
      showAttackIcon={!!attackGraphData}
      on:select={(e) => selectGraph(e.detail)}
    />
    
    <!-- Right Panel: Graph Visualization -->
    <div class="right-panel">
      <h2>
        {attackGraphData ? 'Attack Graph' : 'Graph Visualization'}
      </h2>
      
      {#if !selectedGraphPath}
        <div class="empty-state">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
            <line x1="12" y1="9" x2="12" y2="13"/>
            <line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
          <p>Select a graph to visualize</p>
        </div>
      {:else if isLoadingGraph}
        <div class="loading-state">
          <div class="spinner-border"></div>
          <p>Loading graph data...</p>
        </div>
      {:else if isLoadingPrediction}
        <div class="loading-state">
          <div class="spinner-border danger"></div>
          <p>Analyzing attack sequences...</p>
        </div>
      {:else if graphData}
        <div class="graph-visualization">
          <!-- Graph Stats Panel -->
          <GraphStatsCard 
            stats={statsData} 
            variant={attackGraphData ? 'danger' : 'primary'} 
          />
          
          <!-- Cytoscape Graph Visualization -->
          <div class="graph-canvas-wrapper">
            <div class="graph-canvas">
              <GraphVisualizer
                bind:this={cytoscapeVisualizer}
                {graphData}
                {attackGraphData}
                isAttackMode={!!attackGraphData}
                on:nodeClick={handleNodeClick}
                on:edgeClick={handleEdgeClick}
                on:backgroundClick={handleBackgroundClick}
              />
            </div>
            
            <!-- Element Details Panel -->
            <DetailsPanel
              bind:show={showDetailsPanel}
              {selectedElement}
              isAttackMode={!!attackGraphData}
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
    gap: 1.5rem;
    align-items: center;
    flex-wrap: wrap;
  }
  
  .button-group {
    display: flex;
    gap: 0.75rem;
  }
  
  .log-count,
  .graph-log-count {
    color: #e0e0e0;
    font-weight: 500;
    padding: 0.5rem 1rem;
    background: #0d0d0d;
    border-radius: 6px;
    border: 1px solid #333;
    font-size: 0.9rem;
  }
  
  .graph-log-count {
    background: #1a1a2e;
    border-color: #3b82f6;
  }
  
  .threshold-control {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }
  
  .threshold-control label {
    color: #999;
    font-size: 0.9rem;
    font-weight: 500;
  }
  
  .threshold-control input[type="range"] {
    width: 150px;
  }
  
  .threshold-value {
    color: #e0e0e0;
    font-weight: 600;
    min-width: 45px;
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
  
  .btn-danger {
    background-color: #ef4444;
    color: #fff;
  }
  
  .btn-danger:hover:not(:disabled) {
    background-color: #dc2626;
  }
  
  .btn-secondary {
    background-color: #4b5563;
    color: #fff;
  }
  
  .btn-secondary:hover:not(:disabled) {
    background-color: #374151;
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
  
  .spinner-border {
    width: 40px;
    height: 40px;
    border: 3px solid #333;
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-bottom: 1rem;
  }
  
  .spinner-border.danger {
    border-top-color: #ef4444;
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