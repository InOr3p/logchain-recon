import { apiFetch } from "./client-api";

export async function getLogs(agentId: string, limit: number = 50) {
  return apiFetch(`/agents/${agentId}/logs/latest?limit=${limit}`);
}

export async function getAgents() {
  return apiFetch("/agents");
}
