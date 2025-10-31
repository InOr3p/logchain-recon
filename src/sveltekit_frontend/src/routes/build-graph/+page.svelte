<script lang="ts">
  import { buildGraphs, getGraphData } from '$lib/controllers/graphs-controller';
  import type { GraphData } from '$lib/schema/models';
  import { selectedLogs, showAlert } from '$lib/stores/generalStores';
  import { tick } from 'svelte';
  import cytoscape from 'cytoscape';

  // Cytoscape
  let cyContainer: HTMLElement;
  let cy: any = null;
  
  // State management
  let graphFiles: string[] = [];
  let selectedGraph: string | null = null;
  let graphData: GraphData | null = null;
  let isLoading = false;
  let isLoadingGraph = false;
  let error: string | null = null;
  
  // Edge detail panel
  let selectedEdge: any = null;
  let showEdgeDetails = false;
  
  // Reactive statement to get the current selected logs count
  $: selectedLogsCount = $selectedLogs.length;
  
  // Reactive statement to visualize graph when data changes
  $: if (graphData && cyContainer) {
    setTimeout(() => {
      visualizeGraph(graphData);
    }, 0);
  }
  
  // Build graphs from selected logs
  async function handleBuildGraphs() {
    if ($selectedLogs.length === 0) {
      error = 'Please select logs to build graphs';
      showAlert(error, "danger", 5000);
      return;
    }
    
    isLoading = true;
    error = null;
    graphFiles = [];
    selectedGraph = null;
    graphData = null;
    
    // Clear cytoscape
    if (cy) {
      cy.elements().remove();
    }
    
    try {
      const response = await buildGraphs($selectedLogs);
      graphFiles = response.graph_files || [];
      const message = response.message || `Built ${graphFiles.length} graphs`;
      
      showAlert(message, "success", 5000);

      // Auto-select first graph
      if (graphFiles.length > 0) {
        await selectGraph(graphFiles[0]);
      }
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to build graphs';
      console.error('Error building graphs:', err);
      showAlert(error, "danger", 5000);
    } finally {
      isLoading = false;
    }
  }
  
  // Load graph data from .npz file path and visualize
  async function selectGraph(graphPath: string) {
    selectedGraph = graphPath;
    graphData = null;
    isLoadingGraph = true;
    showEdgeDetails = false;
    selectedEdge = null;
    
    // Clear existing cy instance to force re-initialization
    if (cy) {
      cy.destroy();
      cy = null;
    }
    
    try {
      graphData = await getGraphData(graphPath);
      // Wait for DOM to update
      await tick();
    } catch (err) {
      error = `Failed to load graph: ${err instanceof Error ? err.message : 'Unknown error'}`;
      console.error('Error loading graph:', err);
      showAlert(error, "danger", 5000);
    } finally {
      isLoadingGraph = false;
    }
  }
  
  // Handle edge click
  function handleEdgeClick(edge: any) {
    const sourceNode = edge.source();
    const targetNode = edge.target();
    
    // Get log details from selectedLogs
    const sourceLog = $selectedLogs.find(log => log.id === sourceNode.data('label'));
    const targetLog = $selectedLogs.find(log => log.id === targetNode.data('label'));
    
    selectedEdge = {
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
      },
      // Add any additional edge data you have
      data: edge.data()
    };
    
    showEdgeDetails = true;
    
    // Highlight the selected edge
    cy.edges().style({
      'line-color': '#b0c4ff',
      'width': 2,
      'opacity': 0.5
    });
    
    edge.style({
      'line-color': '#22c55e',
      'width': 4,
      'opacity': 1
    });
  }
  
  // Close edge details panel
  function closeEdgeDetails() {
    showEdgeDetails = false;
    selectedEdge = null;
    
    // Reset all edge styles
    if (cy) {
      cy.edges().style({
        'line-color': '#b0c4ff',
        'width': 2,
        'opacity': 0.5
      });
    }
  }
  
  // Visualize graph using Cytoscape
  function visualizeGraph(data: GraphData) {
    console.log('visualizeGraph called', { 
      cyExists: !!cy, 
      containerExists: !!cyContainer,
      numNodes: data.num_nodes,
      numEdges: data.num_edges 
    });

    if (!cyContainer) {
      console.error('Container not available');
      showAlert("Error: cannot visualize graph", "danger", 5000);
      return;
    }

    // Initialize or reset Cytoscape
    if (!cy) {
      console.log('Initializing Cytoscape...');
      cy = cytoscape({
        container: cyContainer,
        style: [
          {
            selector: 'node',
            style: {
              'background-color': '#3b82f6',
              'label': 'data(label)',
              'color': '#e0e0e0',
              'text-valign': 'center',
              'text-halign': 'center',
              'font-size': '10px',
              'width': '30px',
              'height': '30px'
            }
          },
          {
            selector: 'edge',
            style: {
              'width': 2,
              'line-color': '#b0c4ff',
              'opacity': 0.5,
              'curve-style': 'bezier',
              'target-arrow-shape': 'triangle',
              'target-arrow-color': '#b0c4ff'
            }
          },
          {
            selector: 'edge:selected',
            style: {
              'line-color': '#22c55e',
              'width': 4,
              'opacity': 1,
              'target-arrow-color': '#22c55e'
            }
          },
          {
            selector: 'node:selected',
            style: {
              'background-color': 'green',
              'border-width': 3,
              'border-color': '#fff'
            }
          }
        ],
        layout: {
          name: 'grid'
        }
      });
      
      // Add click handler for edges
      cy.on('tap', 'edge', function(evt: any) {
        handleEdgeClick(evt.target);
      });
      
      // Add click handler for background (to deselect)
      cy.on('tap', function(evt: any) {
        if (evt.target === cy) {
          closeEdgeDetails();
        }
      });
    }

    if (!cy) {
      console.error('Failed to initialize Cytoscape');
      showAlert("Error: cannot visualize graph", "danger", 5000);
      return;
    }
    
    // Clear existing elements
    cy.elements().remove();
    
    // Add nodes
    const nodes = [];
    for (let i = 0; i < data.num_nodes; i++) {
      nodes.push({
        data: { 
          id: `node${i}`,
          label: data.log_ids[i] || `N${i}`
        }
      });
    }
    
    console.log('Adding nodes:', nodes.length);
    
    // Add edges
    const edges = [];
    const edgeIndex = data.edge_index;
    for (let i = 0; i < edgeIndex[0].length; i++) {
      const source = edgeIndex[0][i];
      const target = edgeIndex[1][i];
      edges.push({
        data: {
          id: `edge${i}`,
          source: `node${source}`,
          target: `node${target}`,
          // Store edge features if available
          edgeIndex: i,
          sourceIndex: source,
          targetIndex: target
        }
      });
    }
    
    console.log('Adding edges:', edges.length);
    
    // Add all elements to cytoscape
    cy.add(nodes);
    cy.add(edges);
    
    console.log('Elements added. Total nodes:', cy.nodes().length, 'Total edges:', cy.edges().length);
    
    // Apply force-directed layout
    const layout = cy.layout({
      name: 'cose',
      animate: true,
      animationDuration: 500,
      nodeRepulsion: 8000,
      idealEdgeLength: 100,
      edgeElasticity: 100,
      nestingFactor: 1.2,
      gravity: 1,
      numIter: 1000,
      initialTemp: 200,
      coolingFactor: 0.95,
      minTemp: 1.0
    });
    
    layout.run();
    
    // Fit to viewport after layout completes
    layout.on('layoutstop', () => {
      console.log('Layout complete, fitting to viewport');
      cy.fit(50);
    });
  }
  
  // Format graph filename for display
  function formatGraphName(path: string): string {
    const filename = path.split('/').pop()?.replace('.npz', '') || '';
    return filename.replace('inference_', '').replace(/_/g, ' ');
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
    <!-- Left Panel: Graph List -->
    <div class="left-panel">
      <h2>Generated Graphs ({graphFiles.length})</h2>
      
      {#if isLoading}
        <div class="loading-state">
          <div class="spinner-border"></div>
          <p>Building graphs...</p>
        </div>
      {:else if graphFiles.length === 0}
        <div class="empty-state">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          <p>No graphs built yet</p>
          <p class="hint">Select logs and click "Build Graphs" to start</p>
        </div>
      {:else}
        <div class="graph-list">
          {#each graphFiles as graphPath}
            <button
              class="graph-item"
              class:selected={selectedGraph === graphPath}
              on:click={() => selectGraph(graphPath)}
              title={graphPath}
            >
              <div class="graph-icon">
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
    
    <!-- Right Panel: Graph Visualization -->
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
          <!-- Graph Info Panel -->
          <div class="graph-stats">
            <div class="stat-card">
              <div class="stat-label">Agent IP</div>
              <div class="stat-value">{graphData.agent_ip}</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Nodes</div>
              <div class="stat-value">{graphData.num_nodes}</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Edges</div>
              <div class="stat-value">{graphData.num_edges}</div>
            </div>
          </div>
          
          <!-- Cytoscape Graph Visualization -->
          <div class="graph-canvas-wrapper">
            <div class="graph-canvas">
              <div bind:this={cyContainer} class="cytoscape-container"></div>
            </div>
            
            <!-- Edge Details Panel -->
            {#if showEdgeDetails && selectedEdge}
              <div class="edge-details-panel">
                <div class="panel-header">
                  <h3>Edge Details</h3>
                  <button class="close-btn" on:click={closeEdgeDetails}>
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                      <line x1="18" y1="6" x2="6" y2="18"></line>
                      <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                  </button>
                </div>
                
                <div class="panel-content">
                  <div class="detail-section">
                    <div class="detail-label">Edge ID</div>
                    <div class="detail-value">{selectedEdge.id}</div>
                  </div>
                  
                  <div class="detail-section">
                    <div class="detail-label">
                      <span class="node-badge source">Source Node</span>
                    </div>
                    <div class="node-details">
                      <div class="node-detail-item">
                        <span class="detail-item-label">ID:</span>
                        <span class="detail-item-value">{selectedEdge.source.label}</span>
                      </div>
                      <div class="node-detail-item">
                        <span class="detail-item-label">Description:</span>
                        <span class="detail-item-value">{selectedEdge.source.description}</span>
                      </div>
                      <div class="node-detail-item">
                        <span class="detail-item-label">Timestamp:</span>
                        <span class="detail-item-value">{selectedEdge.source.timestamp}</span>
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
                        <span class="detail-item-value">{selectedEdge.target.label}</span>
                      </div>
                      <div class="node-detail-item">
                        <span class="detail-item-label">Description:</span>
                        <span class="detail-item-value">{selectedEdge.target.description}</span>
                      </div>
                      <div class="node-detail-item">
                        <span class="detail-item-label">Timestamp:</span>
                        <span class="detail-item-value">{selectedEdge.target.timestamp}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            {/if}
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
  
  .left-panel,
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
    border-left: 3px solid green;
  }
  
  .graph-icon {
    flex-shrink: 0;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: green;
    border-radius: 6px;
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
  
  .graph-visualization {
    padding: 1.5rem;
  }
  
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
  
  .cytoscape-container {
    width: 100%;
    height: 500px;
  }
  
  /* Edge Details Panel */
  .edge-details-panel {
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
    font-family: 'Courier New', monospace;
    background: #0d0d0d;
    padding: 0.5rem;
    border-radius: 4px;
    border: 1px solid #333;
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