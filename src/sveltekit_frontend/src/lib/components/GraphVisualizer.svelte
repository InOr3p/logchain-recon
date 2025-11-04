<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';
  import cytoscape from 'cytoscape';
  import type { GraphData, AttackGraphData } from '$lib/schema/models';
  
  export let graphData: GraphData | null = null;
  export let attackGraphData: AttackGraphData | null = null;
  export let isAttackMode: boolean = false;
  
  const dispatch = createEventDispatcher();
  
  let cyContainer: HTMLElement;
  let cy: any = null;
  
  $: if ((graphData || attackGraphData) && cyContainer) {
    setTimeout(() => {
      if (isAttackMode && attackGraphData) {
        visualizeAttackGraph(attackGraphData);
      } else if (graphData) {
        visualizeGraph(graphData);
      }
    }, 0);
  }
  
  function initializeCytoscape(isAttack: boolean = false) {
    const nodeColor = isAttack ? '#ef4444' : '#3b82f6';
    const edgeColor = isAttack ? '#ef4444' : '#b0c4ff';
    
    cy = cytoscape({
      container: cyContainer,
      style: [
        {
          selector: 'node',
          style: {
            'background-color': nodeColor,
            'label': 'data(label)',
            'color': '#e0e0e0',
            'text-valign': 'center',
            'text-halign': 'center',
            'font-size': isAttack ? '11px' : '10px',
            'font-weight': isAttack ? '600' : 'normal',
            'width': isAttack ? '40px' : '30px',
            'height': isAttack ? '40px' : '30px',
            'border-width': 2,
            'border-color': isAttack ? '#dc2626' : nodeColor
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 2,
            'line-color': edgeColor,
            'opacity': isAttack ? 0.6 : 0.5,
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'target-arrow-color': edgeColor,
            'label': isAttack ? 'data(probability)' : '',
            'font-size': '9px',
            'color': '#fca5a5',
            'text-background-color': '#1e1e1e',
            'text-background-opacity': 0.8,
            'text-background-padding': '2px'
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
            'width': isAttack ? 5 : 4,
            'opacity': 1,
            'target-arrow-color': '#22c55e'
          }
        }
      ]
    });
    
    // Event handlers
    cy.on('tap', 'node', (evt: any) => {
      dispatch('nodeClick', evt.target);
    });
    
    cy.on('tap', 'edge', (evt: any) => {
      dispatch('edgeClick', evt.target);
    });
    
    cy.on('tap', (evt: any) => {
      if (evt.target === cy) {
        dispatch('backgroundClick');
      }
    });
  }
  
  function visualizeGraph(data: GraphData) {
    if (!cyContainer) return;
    
    if (!cy) {
      initializeCytoscape(false);
    }
    
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
          target: targetId,
          edgeIndex: i,
          sourceIndex: source,
          targetIndex: target
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
      edgeElasticity: 100,
      nestingFactor: 1.2,
      gravity: 1,
      numIter: 1000,
      initialTemp: 200,
      coolingFactor: 0.95,
      minTemp: 1.0
    });
    
    layout.run();
    layout.on('layoutstop', () => {
      cy.fit(50);
    });
  }
  
  function visualizeAttackGraph(data: AttackGraphData) {
    if (!cyContainer) return;
    
    if (!cy) {
      initializeCytoscape(true);
    }
    
    cy.elements().remove();
    
    // Add nodes
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
    
    // Add edges
    const edges = [];
    if (data.sample_edges) {
      for (let i = 0; i < data.sample_edges.length; i++) {
        const edge = data.sample_edges[i];
        
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
    
    // Apply hierarchical layout
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
  
  export function resetStyles(isAttack: boolean = false) {
    if (!cy) return;
    
    if (isAttack) {
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
  
  onDestroy(() => {
    if (cy) {
      cy.destroy();
    }
  });
</script>

<div bind:this={cyContainer} class="cytoscape-container"></div>

<style>
  .cytoscape-container {
    width: 100%;
    height: 500px;
  }
</style>