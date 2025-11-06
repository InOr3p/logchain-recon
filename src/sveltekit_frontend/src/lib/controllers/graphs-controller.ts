import type { AttackGraphData, BuildGraphsResponse, GenerateReportRequest, GenerateReportResponse, GraphData, LogItem, PredictAttackGraphResponse, ReportHealthResponse } from "$lib/schema/models";
import { apiFetch, apiPost } from "./client-api";

/**
 * Fetches the data for a specific graph file.
 * Note: This endpoint needs to be implemented in your FastAPI backend.
 * @param graphPath The path to the .npz graph file
 * @returns A promise that resolves to GraphData
 */
export async function getGraphData(graphPath: string): Promise<GraphData> {
  // Encode the path as a query parameter
  const encodedPath = encodeURIComponent(graphPath);
  return apiFetch(`/graphs/data?path=${encodedPath}`);
}

/**
 * Sends a list of logs to the backend to build inference graphs.
 * @param logs An array of LogItem objects to build graphs from.
 * @returns A promise that resolves to BuildGraphsResponse with graph file paths.
 */
export async function buildGraphs(logs: LogItem[]): Promise<BuildGraphsResponse> {
  // Send logs to /graphs/build endpoint
  const response: BuildGraphsResponse = await apiPost("/graphs/build", { logs });
  return response;
}

/**
 * Predicts attack sequences in a graph using the EdgePredictor model.
 * @param graphPath The path/filename of the graph file in graph_cache (e.g., "graph_chunk_0.npz")
 * @param logs An array of LogItem objects that are part of the graph
 * @param threshold Probability threshold for attack edges (0.0-1.0)
 * @param summarize Whether to return a summarized graph or full graph
 * @returns A promise that resolves to PredictAttackGraphResponse
 */
export async function predictAttackGraph(
  graphPath: string,
  logs: LogItem[],
  threshold: number = 0.9,
  summarize: boolean = true
): Promise<PredictAttackGraphResponse> {
  // Extract just the filename if full path is provided
  const filename = graphPath.split('/').pop() || graphPath;
  
  const requestBody = {
    graph_path: filename,
    logs: logs,
    threshold: threshold,
    summarize: summarize
  };
  
  console.log('Predicting attack graph:', {
    graphPath: filename,
    numLogs: logs.length,
    threshold,
    summarize
  });
  
  const response: PredictAttackGraphResponse = await apiPost("/graphs/predict", requestBody);
  
  console.log('Attack prediction response:', {
    success: response.success,
    message: response.message,
    hasGraph: !!response.graph,
    totalNodes: response.graph?.total_nodes,
    totalEdges: response.graph?.total_edges
  });
  
  return response;
}


/**
 * Generates an attack analysis report from an attack graph summary.
 * @param graphSummary Summarized attack graph data from prediction
 * @param modelName Optional LLM model name to use
 * @returns A promise that resolves to GenerateReportResponse
 */
export async function generateReport(
  graphSummary: AttackGraphData,
  llm_engine: string,
  modelName?: string
): Promise<GenerateReportResponse> {
  const requestBody: GenerateReportRequest = {
    graph_summary: graphSummary,
    model_name: modelName,
    llm_engine: llm_engine
  };
  
  console.log('Generating report for attack graph:', {
    totalNodes: graphSummary.total_nodes,
    totalEdges: graphSummary.total_edges,
    model: modelName || 'default'
  });
  
  const response: GenerateReportResponse = await apiPost("/graphs/generate-report", requestBody);
  
  console.log('Report generation response:', {
    success: response.success,
    hasReport: !!response.report,
    error: response.error
  });
  
  return response;
}

/**
 * Checks the health status of the report generation service.
 * @returns A promise that resolves to ReportHealthResponse
 */
export async function checkReportHealth(llm_engine: string): Promise<ReportHealthResponse> {
  return apiFetch(`/graphs/report/health?llm_engine=${llm_engine}`);
}