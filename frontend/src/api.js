// Thin client for the FastAPI backend. It only transports messages and maps HTTP
// problems to readable text; every weather and safety decision happens on the server.

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");
const REQUEST_TIMEOUT_MS = 90_000;
export const MAX_MESSAGE_CHARS = 1000;

const FALLBACK_MESSAGES = {
  network: "Couldn't reach the assistant. Check your connection and that the server is running, then try again.",
  timeout: "The assistant took too long to respond. Please try again.",
  server: "Something went wrong on our side. Please try again.",
  message: `Please enter a question (up to ${MAX_MESSAGE_CHARS} characters).`,
  session: "This conversation can't be continued. Please start a new conversation.",
};

/**
 * Send one chat turn.
 * @returns {Promise<{ok: true, data: object} | {ok: false, message: string}>}
 *   ok=true includes answers the server marks as failures (e.g. weather unavailable),
 *   because those still carry a user-facing answer.
 */
export async function sendChat(message, sessionId) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  let response;
  try {
    response = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(sessionId ? { message, session_id: sessionId } : { message }),
      signal: controller.signal,
    });
  } catch (error) {
    return { ok: false, message: error.name === "AbortError" ? FALLBACK_MESSAGES.timeout : FALLBACK_MESSAGES.network };
  } finally {
    clearTimeout(timer);
  }

  let body = null;
  try {
    body = await response.json();
  } catch {
    // non-JSON body (e.g. a proxy error page): handled below without showing it
  }

  if (body && typeof body.answer === "string") {
    return { ok: true, data: body }; // 200, or 503 with an honest answer from the graph
  }
  if (response.status === 422) {
    const fields = body?.error?.fields ?? [];
    const sessionProblem = fields.some((f) => f.field === "session_id");
    return { ok: false, message: sessionProblem ? FALLBACK_MESSAGES.session : FALLBACK_MESSAGES.message };
  }
  if (response.status === 503 && body?.error?.message) {
    return { ok: false, message: body.error.message };
  }
  return { ok: false, message: FALLBACK_MESSAGES.server };
}
