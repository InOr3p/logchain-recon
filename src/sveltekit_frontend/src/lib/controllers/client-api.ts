export const BASE_URL = "http://127.0.0.1:8000"; // FastAPI backend

export async function apiFetch(endpoint: string, options: RequestInit = {}) {
  const res = await fetch(`${BASE_URL}${endpoint}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Error ${res.status}: ${text}`);
  }

  return res.json();
}

export async function apiPost(endpoint: string, body: any) {
  return apiFetch(endpoint, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

