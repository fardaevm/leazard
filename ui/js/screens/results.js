import { api } from "../api.js";
import { requireSignIn } from "../auth.js";
import { el, formatDate, formatScore, isAbort } from "../dom.js";
import { refresh } from "../router.js";

const SEVERITY_RANK = { high: 0, medium: 1, low: 2 };

const TERMS = [
  ["Monthly rent", (l) => l.rent_amount],
  ["Security deposit", (l) => l.deposit_amount],
  ["Lease type", (l) => ({ fixed: "Fixed term", month_to_month: "Month-to-month" })[l.lease_type]],
  ["Start date", (l) => l.start_date],
  ["End date", (l) => l.end_date],
  ["Notice period", (l) => (l.notice_period_days == null || l.notice_period_days === "" ? null : `${l.notice_period_days} days`)],
  ["Late fees", (l) => l.late_fee_policy],
];

const severityKey = (flag) => String(flag.severity || "").trim().toLowerCase();
const rank = (flag) => SEVERITY_RANK[severityKey(flag)] ?? 3;

function summarize(flags, standard) {
  // Older records have no standard_clauses; don't claim "0 standard" for them.
  return standard ? `${flags.length} to review, ${standard.length} standard.` : `${flags.length} to review.`;
}

function citationList(citations) {
  const seen = new Set();
  const items = [];
  for (const c of citations || []) {
    const name = String(c?.source || "").trim();
    if (!name || seen.has(name)) continue;
    seen.add(name);
    const safeUrl = typeof c.url === "string" && /^https?:\/\//i.test(c.url) ? c.url : null;
    items.push(el("li", {}, safeUrl ? el("a", { href: safeUrl, target: "_blank", rel: "noopener noreferrer" }, name) : name));
  }
  return items.length ? el("div", {}, el("p", { class: "small" }, el("strong", {}, "Sources")), el("ul", { class: "sources" }, items)) : null;
}

function flagCard(flag) {
  const key = severityKey(flag);
  const label = flag.severity ? String(flag.severity) : "Unrated";
  const why = String(flag.why_it_matters || "").trim();
  const sources = citationList(flag.citations);
  const quote = String(flag.lease_quote || "").trim();
  return el("li", {}, el("article", { class: "card flag" },
    el("div", { class: "row" },
      el("span", { class: `badge${key in SEVERITY_RANK ? ` sev-${key}` : ""}` }, label, el("span", { class: "sr-only" }, " severity")),
      el("h3", {}, flag.category || "Lease clause")),
    flag.finding ? el("p", {}, flag.finding) : null,
    quote ? el("blockquote", { class: "quote" }, el("span", { class: "sr-only" }, "Your lease says: "), `“${quote}”`) : null,
    why || sources ? el("details", {}, el("summary", {}, "Why it matters"),
      el("div", { class: "stack" }, why ? el("p", {}, why) : null, sources)) : null,
  ));
}

function standardItem(item) {
  const quote = String(item.lease_quote || "").trim();
  return el("li", {}, el("article", { class: "flag" },
    el("h3", {}, item.category || "Lease clause"),
    item.finding ? el("p", {}, item.finding) : null,
    quote ? el("blockquote", { class: "quote" }, el("span", { class: "sr-only" }, "Your lease says: "), `“${quote}”`) : null,
  ));
}

function standardSection(items) {
  if (!items.length) return null;
  return el("details", { class: "card" },
    el("summary", {}, `Looks standard (${items.length})`),
    el("ul", { class: "card-list" }, items.map(standardItem)));
}

function keyTerms(lease) {
  const rows = TERMS.map(([name, get]) => [name, get(lease)])
    .filter(([, value]) => value != null && String(value).trim() !== "")
    .map(([name, value]) => el("tr", {}, el("th", { scope: "row" }, name), el("td", {}, String(value))));
  if (!rows.length) return null;
  return el("section", { class: "stack", "aria-labelledby": "terms-h" },
    el("h2", { id: "terms-h" }, "Key terms"),
    el("table", { class: "terms" }, el("tbody", {}, rows)));
}

function letterCard(text, hasIssues) {
  const letter = el("pre", { class: "letter", tabindex: "0", role: "region", "aria-label": "Draft email to your landlord" }, text);
  const status = el("span", { class: "muted small", role: "status" });
  const copy = el("button", { type: "button", class: "btn no-print" }, "Copy email");
  let timer = 0;
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text);
      status.textContent = "Copied";
    } catch {
      getSelection().selectAllChildren(letter);
      status.textContent = document.execCommand("copy") ? "Copied" : "Text selected. Press Ctrl+C (or Cmd+C) to copy.";
    }
    clearTimeout(timer);
    timer = setTimeout(() => { status.textContent = ""; }, 4000);
  });
  return el("section", { class: "card stack", "aria-labelledby": "letter-h" },
    el("h2", { id: "letter-h" }, hasIssues ? "Negotiate before you sign" : "Questions before you sign"),
    el("p", { class: "muted small" }, hasIssues
      ? "A draft email to your landlord about the issues above. Fill in the bracketed parts before sending."
      : "A draft email with a few questions for your landlord. Fill in the bracketed parts before sending."),
    letter,
    el("div", { class: "row" }, copy, status));
}

function pdfButton(fileId, signal) {
  const status = el("span", { class: "small", role: "status" });
  const button = el("button", { type: "button", class: "btn" }, "View original PDF");
  button.addEventListener("click", async () => {
    // Open synchronously so popup blockers allow it; the Bearer header rules out a plain link.
    const win = window.open("", "_blank");
    if (win) win.opener = null;
    status.textContent = "Opening…";
    try {
      const res = await api(`/uploads/${encodeURIComponent(fileId)}`, { raw: true, signal });
      const url = URL.createObjectURL(new Blob([await res.blob()], { type: "application/pdf" }));
      setTimeout(() => URL.revokeObjectURL(url), 60_000);
      if (win) {
        win.location.href = url;
        status.textContent = "";
      } else {
        status.replaceChildren(el("a", { href: url, target: "_blank", rel: "noopener" }, "Open the PDF"));
      }
    } catch (err) {
      win?.close();
      if (isAbort(err)) return;
      if (err.status === 401) return requireSignIn(refresh);
      status.textContent = err.status === 404 ? "The original PDF is no longer available." : err.message;
    }
  });
  return el("div", { class: "row" }, button, status);
}

function expandForPrint(signal) {
  let opened = [];
  const before = () => {
    opened = [...document.querySelectorAll("details:not([open])")];
    opened.forEach((d) => { d.open = true; });
  };
  const after = () => opened.forEach((d) => { d.open = false; });
  window.addEventListener("beforeprint", before, { signal });
  window.addEventListener("afterprint", after, { signal });
}

function renderRecord(root, rec, signal) {
  const risk = rec.risk_json || {};
  const lease = rec.lease_json || {};
  const flags = (Array.isArray(risk.flags) ? risk.flags : []).slice().sort((a, b) => rank(a) - rank(b));
  const standard = Array.isArray(risk.standard_clauses) ? risk.standard_clauses : null;
  const hasIssues = flags.some((f) => ["high", "medium"].includes(severityKey(f)));
  const score = formatScore(risk.risk_score ?? rec.risk_score);
  const skipped = Number(risk.skipped_categories) || 0;
  const meta = [rec.original_filename, formatDate(rec.created_at)].filter(Boolean).join(" · ");

  const print = el("button", { type: "button", class: "btn primary" }, "Download report");
  print.addEventListener("click", () => window.print());
  expandForPrint(signal);

  root.append(el("div", { class: "stack loose" },
    el("header", { class: "stack tight" },
      meta ? el("p", { class: "muted small" }, meta) : null,
      el("h1", {}, rec.address || "Your lease")),

    el("section", { class: "card stack tight", "aria-label": "Risk score" },
      score
        ? el("p", { class: "score" }, el("span", { class: "score-num" }, score), el("span", { class: "muted" }, "out of 10"))
        : el("p", { class: "muted" }, "No score available for this lease."),
      risk.risk_label ? el("p", { class: "score-label" }, `${risk.risk_label} risk`) : null,
      flags.length || standard?.length ? el("p", {}, summarize(flags, standard)) : null,
      skipped > 0 ? el("p", { class: "muted small" }, `${skipped} ${skipped === 1 ? "area" : "areas"} couldn't be analyzed.`) : null),

    el("section", { class: "stack", "aria-labelledby": "flags-h" },
      el("h2", { id: "flags-h" }, "What we found"),
      flags.length
        ? el("ul", { class: "card-list" }, flags.map(flagCard))
        : el("div", { class: "card stack tight" },
            el("h3", {}, "Nothing risky stood out"),
            el("p", { class: "muted" }, "We didn't flag any clauses in the areas we checked. It's still worth reading the full lease and asking about anything unclear before you sign.")),
      standardSection(standard || [])),

    keyTerms(lease),
    rec.letter_text ? letterCard(rec.letter_text, hasIssues) : null,

    el("div", { class: "row no-print" }, print, rec.file_id ? pdfButton(rec.file_id, signal) : null),
    el("p", { class: "muted small disclaimer" }, "Informational only, not legal advice."),
  ));
}

export async function renderResults({ root, signal, params: [leaseId] }) {
  const loading = el("p", { class: "muted", role: "status" }, "Loading your results…");
  root.append(loading);
  try {
    const rec = await api(`/history/${encodeURIComponent(leaseId)}`, { signal });
    loading.remove();
    renderRecord(root, rec, signal);
  } catch (err) {
    if (isAbort(err)) return;
    if (err.status === 401) return requireSignIn(refresh, "Sign in to see your results.");
    root.replaceChildren(el("div", { class: "stack" },
      el("h1", {}, err.status === 404 ? "We couldn't find this lease" : "Couldn't load your results"),
      el("p", {}, err.status === 404 ? "It may have been removed or belong to another account." : err.message),
      el("p", {}, el("a", { href: "#/history" }, "Go to your history"))));
  }
}
