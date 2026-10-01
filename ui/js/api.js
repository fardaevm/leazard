const TOKEN_KEY = "leazard_token";

export const session = {
  get token() { return localStorage.getItem(TOKEN_KEY); },
  set(token) { localStorage.setItem(TOKEN_KEY, token); document.dispatchEvent(new Event("authchange")); },
  clear() { localStorage.removeItem(TOKEN_KEY); document.dispatchEvent(new Event("authchange")); },
};

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function messageFrom(data, status) {
  const detail = data?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return "Some required information is missing. Please check the form and try again.";
  if (status >= 500) return "Something went wrong on our side. Please try again.";
  return "That didn't work. Please try again.";
}

/** Returns parsed JSON, or the raw Response when `raw` is set. Throws ApiError (status 0 = network). */
export async function api(path, { method = "GET", json, body, raw = false, signal } = {}) {
  const headers = {};
  if (session.token) headers.Authorization = `Bearer ${session.token}`;
  if (json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(json);
  }

  let res;
  try {
    res = await fetch(path, { method, headers, body, signal });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new ApiError(0, "We can't reach Leazard right now. Check your connection and try again.");
  }

  // HTTPBearer answers 403 when the header is missing; treat both as "signed out".
  if ((res.status === 401 || res.status === 403) && !path.startsWith("/auth/")) {
    session.clear();
    throw new ApiError(401, "Your session has expired. Please sign in again.");
  }
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new ApiError(res.status, messageFrom(data, res.status));
  }
  return raw ? res : res.json();
}
