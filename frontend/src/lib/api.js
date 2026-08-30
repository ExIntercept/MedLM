export const API_BASE =
  import.meta.env.VITE_API_BASE ||
  (typeof window !== "undefined" && window.location.port === "5173"
    ? "http://127.0.0.1:8600"
    : "");

const TOKEN_KEY = "medintake_token";

export function getToken() {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token) {
  try {
    localStorage.setItem(TOKEN_KEY, token);
  } catch {
    /* ignore storage failures (private browsing, etc.) */
  }
}

export function clearToken() {
  try {
    localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* ignore */
  }
}

/**
 * fetch() wrapper that attaches the bearer token when present. On a 401 it
 * clears the stored token and dispatches a window event so AuthContext (which
 * can't be imported here without a cycle) can force a logout/redirect to the
 * login screen — used for both "never logged in" and "token expired mid-session".
 */
async function authFetch(path, options = {}) {
  const token = getToken();
  const headers = { ...(options.headers || {}) };
  if (token) headers.Authorization = `Bearer ${token}`;

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });
  if (res.status === 401) {
    clearToken();
    window.dispatchEvent(new CustomEvent("auth:unauthorized"));
  }
  return res;
}

export async function getHealth() {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error(`health check failed: ${res.status}`);
  return res.json();
}

// --------------------------------------------------------------------------
// Auth
// --------------------------------------------------------------------------
export async function register(username, password, email) {
  const res = await fetch(`${API_BASE}/api/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password, email: email || undefined }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `registration failed: ${res.status}`);
  }
  return res.json();
}

export async function login(username, password) {
  const form = new URLSearchParams();
  form.set("username", username);
  form.set("password", password);
  const res = await fetch(`${API_BASE}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: form,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `login failed: ${res.status}`);
  }
  const data = await res.json();
  setToken(data.access_token);
  return data;
}

export function logout() {
  clearToken();
}

export async function getMe() {
  const res = await authFetch("/api/auth/me");
  if (!res.ok) throw new Error(`failed to load current user: ${res.status}`);
  return res.json();
}

/** Returns { generated_at, model, dataset_version, num_cases, metrics, cases } from the
 * latest run of scripts/run_benchmark.py. Throws with a 404-friendly message if it hasn't run yet. */
export async function getEvaluationSummary() {
  const res = await fetch(`${API_BASE}/api/evaluation/summary`);
  if (res.status === 404) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || "No benchmark results yet.");
  }
  if (!res.ok) throw new Error(`failed to load evaluation summary: ${res.status}`);
  return res.json();
}

/** Returns the raw docs/BENCHMARK_REPORT.md contents as a string. */
export async function getEvaluationReport() {
  const res = await fetch(`${API_BASE}/api/evaluation/report`);
  if (res.status === 404) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || "No benchmark report yet.");
  }
  if (!res.ok) throw new Error(`failed to load benchmark report: ${res.status}`);
  const data = await res.json();
  return data.markdown;
}

// --------------------------------------------------------------------------
// Conversations (protected)
// --------------------------------------------------------------------------
/** Returns [{ id, date, subject, patient_profile, messages }, ...]. */
export async function getConversations() {
  const res = await authFetch("/api/conversations");
  if (!res.ok) throw new Error(`failed to load conversations: ${res.status}`);
  return res.json();
}

/** Returns { id, date, subject, patient_profile, messages } for one conversation. */
export async function getConversationDetail(conversationId) {
  const res = await authFetch(`/api/conversations/${conversationId}`);
  if (!res.ok) throw new Error(`failed to load conversation ${conversationId}: ${res.status}`);
  return res.json();
}

export async function createConversation(title, patientProfile) {
  const res = await authFetch("/api/conversations/new", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, patient_profile: patientProfile }),
  });
  if (!res.ok) throw new Error(`failed to create conversation: ${res.status}`);
  return res.json();
}

/** Reuses the same hard-rule guardrail engine as chat, applied to the Prescriptions tab. */
export async function checkMedications(medications, conditions) {
  const res = await authFetch("/api/medications/check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ medications, conditions }),
  });
  if (!res.ok) throw new Error(`medication check failed: ${res.status}`);
  return res.json();
}

/** Browses the underlying guideline corpus via the same hybrid retrieval pipeline as chat.
 * In "patient" mode, the backend prioritizes high-level clinical overview content over
 * technical/internal sections (dosage tables, adverse-reaction lists). */
export async function searchEvidence(query, mode = "clinician") {
  const res = await authFetch(
    `/api/evidence/search?q=${encodeURIComponent(query)}&mode=${encodeURIComponent(mode)}`
  );
  if (!res.ok) throw new Error(`evidence search failed: ${res.status}`);
  const data = await res.json();
  return data.results;
}

/**
 * Parse one raw SSE frame (everything between two blank lines) into
 * { event, data }. Multiple `data:` lines within a frame are joined with
 * "\n", matching the SSE spec (same behavior as native EventSource).
 */
function parseSSEFrame(frame) {
  let event = "message";
  const dataLines = [];
  for (const rawLine of frame.split("\n")) {
    if (!rawLine || rawLine.startsWith(":")) continue;
    if (rawLine.startsWith("event:")) {
      event = rawLine.slice(6).trim();
    } else if (rawLine.startsWith("data:")) {
      dataLines.push(rawLine.slice(5).replace(/^ /, ""));
    }
  }
  return { event, data: dataLines.join("\n") };
}

/**
 * Streams /api/chat/stream via fetch + ReadableStream (manual SSE parsing,
 * since EventSource does not support POST bodies or custom headers). Calls
 * `onEvent({event, data})` for every frame as it arrives.
 */
export async function streamChat(payload, onEvent, { signal } = {}) {
  const token = getToken();
  const headers = { "Content-Type": "application/json" };
  if (token) headers.Authorization = `Bearer ${token}`;

  const res = await fetch(`${API_BASE}/api/chat/stream`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
    signal,
  });
  if (res.status === 401) {
    clearToken();
    window.dispatchEvent(new CustomEvent("auth:unauthorized"));
    throw new Error("Your session has expired. Please log in again.");
  }
  if (!res.ok || !res.body) {
    throw new Error(`chat stream failed: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    // sse-starlette (and SSE servers generally) emit "\r\n" line endings, so
    // frames are separated by "\r\n\r\n" — which does NOT contain the substring
    // "\n\n". Without normalizing first, frames never split mid-stream: nothing
    // renders live, and everything is eventually parsed as one concatenated
    // blob, which is what produces "Unexpected non-whitespace character after
    // JSON" (the first payload parses fine, then trailing data from every
    // other event that got glued on breaks the parse).
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      if (!frame.trim()) continue;
      onEvent(parseSSEFrame(frame));
    }
  }
  if (buffer.trim()) {
    onEvent(parseSSEFrame(buffer));
  }
}
