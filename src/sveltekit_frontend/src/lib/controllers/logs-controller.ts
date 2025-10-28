import type { AgentLogsResponse, AgentsResponse, ClassifyResponse, LogItem, Prediction } from "$lib/schema/models";
import { apiFetch, apiPost } from "./client-api";

/**
 * Fetches the latest logs for a specific agent.
 */
export async function getLogs(agentId: string, limit: number = 50): Promise<AgentLogsResponse> {
  // Add the return type for better type safety
  return apiFetch(`/agents/${agentId}/logs/latest?limit=${limit}`);
}

/**
 * Fetches the list of all available agents.
 */
export async function getAgents(): Promise<AgentsResponse> {
  // Add the return type
  return apiFetch("/agents");
}

/**
 * [NEW FUNCTION]
 * Sends a list of logs to the backend for classification.
 * @param logs An array of LogItem objects to be classified.
 * @returns A promise that resolves to an array of Prediction objects.
 */
export async function classifyLogs(logs: LogItem[]): Promise<Prediction[]> {
  // 1. Use the apiPost function
  // 2. Send the logs array as the body
  // 3. The endpoint is /logs/classify (which matches your logs.py router)
  const response: ClassifyResponse = await apiPost("/logs/classify", logs);

  // 4. Return the nested 'predictions' array as per your schema
  return response.predictions;
}