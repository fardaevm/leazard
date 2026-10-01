// The only way screens build DOM. Children are appended as text nodes, never parsed as HTML,
// because lease content and server messages are untrusted.
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value == null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : String(child));
  }
  return node;
}

export function formatDate(iso) {
  if (!iso) return "";
  // Backend timestamps are naive UTC.
  const date = new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso}Z`);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

export function formatScore(score) {
  return typeof score === "number" && Number.isFinite(score) ? score.toFixed(1) : null;
}

export function isAbort(err) {
  return err?.name === "AbortError";
}
