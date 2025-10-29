import type { BuildGraphsResponse, GraphData, LogItem } from "$lib/schema/models";
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