/**
 * api.ts — REST client for the LibraryAI .NET backend.
 *
 * All authenticated requests read the JWT from localStorage and attach it
 * as a Bearer token. The base URL is empty because Vite proxies /api to
 * the .NET backend during development (see vite.config.ts).
 */

const API_BASE = "/api";

/** Build Authorization header from the stored JWT token. */
function authHeaders(): HeadersInit {
  const token = localStorage.getItem("token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function login(email: string, password: string) {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: "Login failed" }));
    throw new Error(err.error || "Login failed");
  }
  return res.json();
}

export async function register(username: string, email: string, password: string) {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, email, password }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ errors: ["Registration failed"] }));
    throw new Error(err.errors?.join(", ") || "Registration failed");
  }
  return res.json();
}

// --- Memory ---
export async function saveFact(fact: string) {
  const res = await fetch(`${API_BASE}/memory/facts`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ fact }),
  });
  return res.json();
}

export async function listFacts(): Promise<{ facts: string[] }> {
  const res = await fetch(`${API_BASE}/memory/facts`, {
    headers: authHeaders(),
  });
  return res.json();
}

export async function deleteFact(keyword: string) {
  const res = await fetch(`${API_BASE}/memory/facts?keyword=${encodeURIComponent(keyword)}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  return res.json();
}

export async function wipeMemory() {
  await fetch(`${API_BASE}/memory/wipe`, {
    method: "POST",
    headers: authHeaders(),
  });
}

export async function clearBuffer() {
  await fetch(`${API_BASE}/memory/clear-buffer`, {
    method: "POST",
    headers: authHeaders(),
  });
}

// --- Documents ---
export async function uploadDocument(file: File) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/documents/upload`, {
    method: "POST",
    headers: authHeaders(),
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: "Upload failed" }));
    throw new Error(err.detail || err.error || "Upload failed");
  }
  return res.json();
}

export async function listDocuments(): Promise<{ documents: string[] }> {
  const res = await fetch(`${API_BASE}/documents`, {
    headers: authHeaders(),
  });
  return res.json();
}

export async function removeDocument(filename: string) {
  const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(filename)}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  return res.json();
}
