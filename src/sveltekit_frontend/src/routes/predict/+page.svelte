<script lang="ts">
  import { predictAttackGraph, getGraphData } from '$lib/controllers/graphs-controller';
  import type { AttackGraphData, GraphData } from '$lib/schema/models';
  import { graphFiles, logs, showAlert } from '$lib/stores/generalStores';
  import { tick } from 'svelte';
  import cytoscape from 'cytoscape';

  // Cytoscape
  let cyContainer: HTMLElement;
  let cy: any = null;
  
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
  
  // Reactive statements
  $: availableGraphs = $graphFiles;
  $: availableLogsCount = $logs.length;
  
  // Reactive statement to visualize graph when data changes
  $: if (attackGraphData && cyContainer) {
    setTimeout(() => {
      visualizeAttackGraph(attackGraphData);
    }, 0);
  } else if (graphData && cyContainer && !attackGraphData) {
    setTimeout(() => {
      visualizeOriginalGraph(graphData);
    }, 0);
  }
  
  // Select a graph and load its data
  async function selectGraph(graphPath: string) {
    selectedGraphPath = graphPath;
    graphData = null;
    attackGraphData = null;
    isLoadingGraph = true;
    showDetailsPanel = false;
    selectedElement = null;
    error = null;
    
    // Clear existing cy instance
    if (cy) {
      cy.destroy();
      cy = null;
    }
    
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
    
    // Extract log IDs from the graph
    const logIdsInGraph = new Set(graphData.log_ids);
    
    // Filter logs from the store that match the graph's log IDs
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
    
    // Get logs that are part of this graph
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
    
    // Clear cytoscape
    if (cy) {
      cy.destroy();
      cy = null;
    }
    
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
        
        // Wait for DOM update
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
    
    if (cy) {
      cy.destroy();
      cy = null;
    }
    
    if (graphData && cyContainer) {
      setTimeout(() => {
        visualizeOriginalGraph(graphData);
      }, 0);
    }
  }
  
  // Handle node click
  function handleNodeClick(node: any) {
    const nodeId = node.data('id');
    const nodeData = node.data();
    
    if (attackGraphData) {
      // Find the node in attack graph data
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
      // Original graph node
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
    
    // Highlight the selected node
    cy.nodes().style({
      'background-color': attackGraphData ? '#ef4444' : '#3b82f6',
      'border-width': 2
    });
    
    node.style({
      'background-color': '#22c55e',
      'border-width': 4,
      'border-color': '#16a34a'
    });
  }
  
  // Handle edge click
  function handleEdgeClick(edge: any) {
    const edgeData = edge.data();
    const sourceNode = edge.source();
    const targetNode = edge.target();
    
    if (attackGraphData) {
      // Find edge in attack graph data
      const attackEdge = attackGraphData.sample_edges?.find(
        e => e.source_log_id === sourceNode.data('id') && 
             e.dest_log_id === targetNode.data('id')
      );
      
      // Find source and target node data
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
      
      // Highlight the selected edge (attack)
      cy.edges().style({
        'line-color': '#ef4444',
        'width': 2,
        'opacity': 0.6,
        'target-arrow-color': '#ef4444'
      });
      
      edge.style({
        'line-color': '#22c55e',
        'width': 5,
        'opacity': 1,
        'target-arrow-color': '#22c55e'
      });
    } else {
      // Original graph edge
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
      
      // Highlight the selected edge (normal)
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
    
    showDetailsPanel = true;
  }
  
  // Close details panel
  function closeDetailsPanel() {
    showDetailsPanel = false;
    selectedElement = null;
    
    // Reset all styles
    if (cy) {
      if (attackGraphData) {
        cy.nodes().style({
          'background-color': '#ef4444',
          'border-width': 2,
          'border-color': '#dc2626'
        });
        
        cy.edges().style({
          'line-color': '#ef4444',
          'width': 2,
          'opacity': 0.6,
          'target-arrow-color': '#ef4444'
        });
      } else {
        cy.nodes().style({
          'background-color': '#3b82f6',
          'border-width': 2
        });
        
        cy.edges().style({
          'line-color': '#b0c4ff',
          'width': 2,
          'opacity': 0.5
        });
      }
    }
  }
  
  // Visualize original graph (before prediction)
  function visualizeOriginalGraph(data: GraphData) {
    console.log('visualizeOriginalGraph called', {
      cyExists: !!cy,
      containerExists: !!cyContainer,
      numNodes: data.num_nodes,
      numEdges: data.num_edges
    });

    if (!cyContainer) {
      console.error('Container not available');
      return;
    }

    // Initialize Cytoscape
    if (!cy) {
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
            selector: 'node:selected',
            style: {
              'background-color': '#22c55e',
              'border-width': 4,
              'border-color': '#16a34a'
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
          }
        ]
      });
      
      // Add event handlers
      cy.on('tap', 'node', function(evt: any) {
        handleNodeClick(evt.target);
      });
      
      cy.on('tap', 'edge', function(evt: any) {
        handleEdgeClick(evt.target);
      });
      
      cy.on('tap', function(evt: any) {
        if (evt.target === cy) {
          closeDetailsPanel();
        }
      });
    }
    
    // Clear existing elements
    cy.elements().remove();
    
    // Add nodes
    const nodes = [];
    for (let i = 0; i < data.num_nodes; i++) {
      nodes.push({
        data: {
          id: data.log_ids[i] || `node${i}`,
          label: data.log_ids[i] ? data.log_ids[i].substring(0, 8) + '...' : `N${i}`
        }
      });
    }
    
    // Add edges
    const edges = [];
    const edgeIndex = data.edge_index;
    for (let i = 0; i < edgeIndex[0].length; i++) {
      const source = edgeIndex[0][i];
      const target = edgeIndex[1][i];
      const sourceId = data.log_ids[source] || `node${source}`;
      const targetId = data.log_ids[target] || `node${target}`;
      
      edges.push({
        data: {
          id: `edge${i}`,
          source: sourceId,
          target: targetId
        }
      });
    }
    
    cy.add(nodes);
    cy.add(edges);
    
    // Apply layout
    const layout = cy.layout({
      name: 'cose',
      animate: true,
      animationDuration: 500,
      nodeRepulsion: 8000,
      idealEdgeLength: 100,
      edgeElasticity: 100
    });
    
    layout.run();
    layout.on('layoutstop', () => {
      cy.fit(50);
    });
  }
  
  // Visualize attack graph using Cytoscape
  function visualizeAttackGraph(data: AttackGraphData) {
    console.log('visualizeAttackGraph called', {
      cyExists: !!cy,
      containerExists: !!cyContainer,
      numNodes: data.important_nodes,
      numEdges: data.important_edges
    });

    if (!cyContainer) {
      console.error('Container not available');
      return;
    }

    // Initialize Cytoscape with attack styling
    if (!cy) {
      cy = cytoscape({
        container: cyContainer,
        style: [
          {
            selector: 'node',
            style: {
              'background-color': '#ef4444',
              'label': 'data(label)',
              'color': '#e0e0e0',
              'text-valign': 'center',
              'text-halign': 'center',
              'font-size': '11px',
              'font-weight': '600',
              'width': '40px',
              'height': '40px',
              'border-width': 2,
              'border-color': '#dc2626'
            }
          },
          {
            selector: 'edge',
            style: {
              'width': 2,
              'line-color': '#ef4444',
              'opacity': 0.6,
              'curve-style': 'bezier',
              'target-arrow-shape': 'triangle',
              'target-arrow-color': '#ef4444',
              'label': 'data(probability)',
              'font-size': '9px',
              'color': '#fca5a5',
              'text-background-color': '#1e1e1e',
              'text-background-opacity': 0.8,
              'text-background-padding': '2px'
            }
          },
          {
            selector: 'edge:selected',
            style: {
              'line-color': '#22c55e',
              'width': 5,
              'opacity': 1,
              'target-arrow-color': '#22c55e'
            }
          },
          {
            selector: 'node:selected',
            style: {
              'background-color': '#22c55e',
              'border-width': 4,
              'border-color': '#16a34a'
            }
          }
        ]
      });
      
      // Add event handlers
      cy.on('tap', 'node', function(evt: any) {
        handleNodeClick(evt.target);
      });
      
      cy.on('tap', 'edge', function(evt: any) {
        handleEdgeClick(evt.target);
      });
      
      cy.on('tap', function(evt: any) {
        if (evt.target === cy) {
          closeDetailsPanel();
        }
      });
    }
    
    // Clear existing elements
    cy.elements().remove();
    
    // Add nodes from sample_nodes
    const nodes = [];
    const nodeIds = new Set<string>();
    
    if (data.sample_nodes) {
      for (const node of data.sample_nodes) {
        nodes.push({
          data: {
            id: node.id,
            label: node.id.substring(0, 8) + '...',
            fullLabel: node.id,
            description: node.description,
            probability: node.max_incident_prob
          }
        });
        nodeIds.add(node.id);
      }
    }
    
    // Add edges from sample_edges
    const edges = [];
    if (data.sample_edges) {
      for (let i = 0; i < data.sample_edges.length; i++) {
        const edge = data.sample_edges[i];
        
        // Only add edge if both nodes exist
        if (nodeIds.has(edge.source_log_id) && nodeIds.has(edge.dest_log_id)) {
          edges.push({
            data: {
              id: `edge${i}`,
              source: edge.source_log_id,
              target: edge.dest_log_id,
              probability: `${(edge.edge_prob * 100).toFixed(0)}%`,
              rawProbability: edge.edge_prob,
              timestamp: edge.timestamp
            }
          });
        }
      }
    }
    
    cy.add(nodes);
    cy.add(edges);
    
    console.log('Attack graph elements added. Nodes:', cy.nodes().length, 'Edges:', cy.edges().length);
    
    // Apply hierarchical layout for attack sequence
    const layout = cy.layout({
      name: 'breadthfirst',
      directed: true,
      animate: true,
      animationDuration: 500,
      spacingFactor: 1.5,
      avoidOverlap: true,
      nodeDimensionsIncludeLabels: true
    });
    
    layout.run();
    layout.on('layoutstop', () => {
      cy.fit(50);
    });
  }
  
  // Format graph filename for display
  function formatGraphName(path: string): string {
    const filename = path.split('/').pop()?.replace('.npz', '') || '';
    return filename.replace('inference_', '').replace(/_/g, ' ');
  }
  
  // Format probability as percentage
  function formatProbability(prob: number): string {
    return `${(prob * 100).toFixed(1)}%`;
  }
</script>

<div class="container">
  <!-- Header -->
  <div class="header">
    <h1>Attack Predictor</h1>
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
    <div class="left-panel">
      <h2>Available Graphs ({availableGraphs.length})</h2>
      
      {#if availableGraphs.length === 0}
        <div class="empty-state">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          <p>No graphs available</p>
          <p class="hint">Build graphs first in the Graph Builder</p>
        </div>
      {:else}
        <div class="graph-list">
          {#each availableGraphs as graphPath}
            <button
              class="graph-item"
              class:selected={selectedGraphPath === graphPath}
              on:click={() => selectGraph(graphPath)}
              title={graphPath}
            >
              <div class="graph-icon" class:attack={selectedGraphPath === graphPath && attackGraphData}>
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
          <div class="graph-stats">
            {#if attackGraphData}
              <div class="stat-card danger">
                <div class="stat-label">Attack Nodes</div>
                <div class="stat-value">{attackGraphData.important_nodes || attackGraphData.total_nodes}</div>
              </div>
              <div class="stat-card danger">
                <div class="stat-label">Attack Edges</div>
                <div class="stat-value">{attackGraphData.important_edges || attackGraphData.total_edges}</div>
              </div>
              <div class="stat-card danger">
                <div class="stat-label">Threshold</div>
                <div class="stat-value">{formatProbability(threshold)}</div>
              </div>
            {:else}
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
            {/if}
          </div>
          
          <!-- Cytoscape Graph Visualization -->
          <div class="graph-canvas-wrapper">
            <div class="graph-canvas">
              <div bind:this={cyContainer} class="cytoscape-container"></div>
            </div>
            
            <!-- Element Details Panel -->
            {#if showDetailsPanel && selectedElement}
              <div class="details-panel">
                <div class="panel-header">
                  <h3>
                    {selectedElement.type === 'node' ? 'Node Details' : 'Edge Details'}
                  </h3>
                  <button class="close-btn" on:click={closeDetailsPanel}>
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
                    
                    {#if attackGraphData && selectedElement.probability > 0}
                      <div class="detail-section">
                        <div class="detail-label">Max Incident Probability</div>
                        <div class="probability-badge" class:high={selectedElement.probability > 0.8}>
                          {formatProbability(selectedElement.probability)}
                        </div>
                      </div>
                    {/if}
                    
                    <div class="detail-section">
                      <div class="detail-label">Timestamp</div>
                      <div class="detail-value">{selectedElement.timestamp}</div>
                    </div>
                  {:else}
                    <!-- Edge Details -->
                    {#if attackGraphData && selectedElement.probability > 0}
                      <div class="detail-section">
                        <div class="detail-label">Attack Probability</div>
                        <div class="probability-badge large" class:high={selectedElement.probability > 0.8}>
                          {formatProbability(selectedElement.probability)}
                        </div>
                      </div>
                      
                      <div class="detail-section">
                        <div class="detail-label">Timestamp</div>
                        <div class="detail-value">{selectedElement.timestamp}</div>
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
                          <span class="detail-item-value">{selectedElement.source.timestamp}</span>
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
                          <span class="detail-item-value">{selectedElement.target.timestamp}</span>
                        </div>
                      </div>
                    </div>
                  {/if}
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
  
  .checkbox-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #999;
    font-size: 0.9rem;
    cursor: pointer;
  }
  
  .checkbox-label input[type="checkbox"] {
    cursor: pointer;
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
  
  .spinner-border.danger {
    border-top-color: #ef4444;
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
    background: #3b82f6;
    border-radius: 6px;
    transition: background 0.2s;
  }
  
  .graph-icon.attack {
    background: #ef4444;
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
  
  /* Details Panel */
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