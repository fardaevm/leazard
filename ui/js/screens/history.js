import { api } from "../api.js";
import { requireSignIn } from "../auth.js";
import { el, formatDate, formatScore, isAbort } from "../dom.js";
import { refresh } from "../router.js";

function historyItem(item) {
  const score = formatScore(item.risk_score);
  const meta = [item.address, formatDate(item.created_at)].filter(Boolean).join(" · ");
  return el("li", {}, el("a", { class: "card history-item", href: `#/results/${encodeURIComponent(item.id)}` },
    el("div", { class: "stack tight" },
      el("strong", {}, item.original_filename || "Untitled lease"),
      meta ? el("span", { class: "muted small" }, meta) : null),
    el("span", { class: "history-score" },
      score ? el("strong", {}, score) : el("span", { class: "muted" }, "No score"),
      score ? el("span", { class: "muted small" }, " / 10") : null),
  ));
}

export async function renderHistory({ root, signal }) {
  const loading = el("p", { class: "muted", role: "status" }, "Loading your leases…");
  root.append(loading);
  let items;
  try {
    items = await api("/history", { signal });
  } catch (err) {
    if (isAbort(err)) return;
    if (err.status === 401) return requireSignIn(refresh, "Sign in to see your saved leases.");
    loading.remove();
    root.append(el("div", { class: "stack" }, el("h1", {}, "Couldn't load your leases"), el("p", {}, err.message)));
    return;
  }

  loading.remove();
  const list = Array.isArray(items) ? items : [];
  root.append(el("div", { class: "stack loose" },
    el("h1", {}, "Your leases"),
    list.length
      ? el("ul", { class: "card-list" }, list.map(historyItem))
      : el("div", { class: "card stack" },
          el("h2", {}, "No leases yet"),
          el("p", { class: "muted" }, "Analyze a lease and it will be saved here, private to your account."),
          el("div", {}, el("a", { class: "btn primary", href: "#/" }, "Analyze a lease"))),
  ));
}
