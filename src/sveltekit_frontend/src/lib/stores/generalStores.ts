import { writable } from "svelte/store";
import type { AttackGraphData, AttackReport } from '$lib/schema/models';


export let alertMessage = writable('');
export let alertColor = writable('danger');
export let alertVisible = writable(false);

export function showAlert(message = "", color = 'danger', duration = 5000) {
    alertMessage.set(message);
    alertColor.set(color);
    alertVisible.set(true);

    setTimeout(() => {
    alertVisible.set(false);
    }, duration);
}

export const logs = writable<any[]>([]);
export const selectedLogs = writable<any[]>([]);
export const agents = writable<any[]>([]);
export const graphFiles = writable<any[]>([]);
export const selectedGraphPath = writable<any[]>([]);
export const generatedReports = writable<Map<string, AttackReport>>(new Map());


/**
 * Store for predicted attack graphs
 * Maps graph_path -> AttackGraphData
 */
export const predictedGraphs = writable<Map<string, AttackGraphData>>(new Map());

/**
 * Add a predicted graph to the store
 */
export function addPredictedGraph(graphPath: string, graphData: AttackGraphData) {
  predictedGraphs.update(graphs => {
    graphs.set(graphPath, graphData);
    return graphs;
  });
}

/**
 * Remove a predicted graph from the store
 */
export function removePredictedGraph(graphPath: string) {
  predictedGraphs.update(graphs => {
    graphs.delete(graphPath);
    return graphs;
  });
}

/**
 * Clear all predicted graphs
 */
export function clearPredictedGraphs() {
  predictedGraphs.set(new Map());
}

/**
 * Get a specific predicted graph
 */
export function getPredictedGraph(graphPath: string): AttackGraphData | undefined {
  let graph: AttackGraphData | undefined;
  predictedGraphs.subscribe(graphs => {
    graph = graphs.get(graphPath);
  })();
  return graph;
}