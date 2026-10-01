import { api } from "../api.js";
import { requireSignIn } from "../auth.js";
import { el, isAbort } from "../dom.js";
import { go, refresh } from "../router.js";
import { activeJob, draft } from "../state.js";

const POLL_MS = 1500;
// Must match the step labels in agent.JOB_STEPS; the API reports the label, not a key.
const STEPS = ["Checking your ZIP", "Reading your lease", "Finding risk areas", "Checking California law", "Writing your negotiation email"];

const ERRORS = {
  out_of_scope: { title: "We don't cover this area yet", retry: "Check a different ZIP" },
  scanned_pdf: {
    title: "This PDF looks like a scan",
    body: "We can only read PDFs with selectable text. If you signed electronically or received the lease by email, download it as a PDF from that service instead of scanning or photographing a printout. Then try again.",
    retry: "Choose a different file",
    hideFileNote: true,
  },
  extraction_failed: { title: "We couldn't read this PDF", retry: "Try again" },
  timeout: { title: "This took longer than expected", retry: "Try again" },
  interrupted: { title: "The analysis was interrupted", retry: "Try again" },
  analysis_failed: { title: "We couldn't finish the analysis", retry: "Try again" },
};

function notifyForm() {
  const input = el("input", { id: "notify-email", type: "email", autocomplete: "email", required: true, "aria-describedby": "notify-error" });
  const err = el("p", { id: "notify-error", class: "error", hidden: true }, "Enter a valid email address.");
  const form = el("form", { class: "stack", novalidate: true },
    el("div", {}, el("label", { for: "notify-email" }, "Email me when my area is supported"), input, err),
    el("div", {}, el("button", { type: "submit", class: "btn" }, "Notify me")));
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const valid = input.value.trim() !== "" && input.checkValidity();
    err.hidden = valid;
    input.setAttribute("aria-invalid", String(!valid));
    if (!valid) return input.focus();
    // TODO(backend): POST { email, zip_code } to a waitlist endpoint once one exists.
    const thanks = el("p", { role: "status" }, "Thanks. We'll email you when we cover your area.");
    form.replaceWith(thanks);
  });
  return form;
}

function renderError(root, job) {
  const copy = ERRORS[job.error_code] || ERRORS.analysis_failed;
  const body = copy.body || job.error_message || "Something went wrong. Please try again.";
  const retry = el("button", { type: "button", class: "btn primary" }, copy.retry);
  retry.addEventListener("click", () => go("#/"));
  root.replaceChildren(el("div", { class: "stack loose", role: "alert" },
    el("div", { class: "stack" }, el("h1", {}, copy.title), el("p", {}, body)),
    job.error_code === "out_of_scope" ? notifyForm() : null,
    el("div", { class: "row" }, retry,
      draft.file && !copy.hideFileNote ? el("p", { class: "muted small" }, `Your file "${draft.file.name}" is still selected.`) : null),
  ));
  const heading = root.querySelector("h1");
  heading.tabIndex = -1;
  heading.focus();
}

export function renderAnalyzing({ root, signal, params: [jobId] }) {
  const bar = el("span");
  const progress = el("div", {
    class: "progress", role: "progressbar", "aria-label": "Analysis progress",
    "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": "0",
  }, bar);
  const items = STEPS.map((label) => el("li", { class: "step" }, label, el("span", { class: "sr-only" })));
  const live = el("p", { class: "sr-only", "aria-live": "polite" });
  const note = el("p", { class: "muted small", "aria-live": "polite" });
  let current = -1;
  let timer = 0;

  function update(job) {
    const pct = Math.max(0, Math.min(100, Number(job.progress) || 0));
    bar.style.width = `${pct}%`;
    progress.setAttribute("aria-valuenow", String(pct));
    const idx = job.status === "done" ? STEPS.length : STEPS.indexOf(job.step);
    if (idx === -1 || idx === current) return;
    current = idx;
    items.forEach((li, i) => {
      li.className = `step${i < idx ? " done" : i === idx ? " current" : ""}`;
      if (i === idx) li.setAttribute("aria-current", "step");
      else li.removeAttribute("aria-current");
      li.lastChild.textContent = i < idx ? " (done)" : i === idx ? " (in progress)" : "";
    });
    if (idx < STEPS.length) live.textContent = `${STEPS[idx]}…`;
  }

  async function poll() {
    try {
      const job = await api(`/jobs/${encodeURIComponent(jobId)}`, { signal });
      note.textContent = "";
      update(job);
      if (job.status === "done") {
        activeJob.clear();
        live.textContent = "Done. Opening your results.";
        return go(`#/results/${job.lease_id}`, { replace: true });
      }
      if (job.status === "error") {
        activeJob.clear();
        return renderError(root, job);
      }
    } catch (err) {
      if (isAbort(err)) return;
      if (err.status === 401) return requireSignIn(refresh, "Your session has expired. Sign in to keep watching your analysis.");
      if (err.status === 404) {
        activeJob.clear();
        return renderError(root, { error_code: "analysis_failed", error_message: "We couldn't find this analysis. It may belong to another account." });
      }
      note.textContent = "Having trouble reaching the server. Still trying…";
    }
    if (!signal.aborted) timer = setTimeout(poll, POLL_MS);
  }

  signal.addEventListener("abort", () => clearTimeout(timer));
  root.append(el("div", { class: "stack loose" },
    el("div", { class: "stack" }, el("h1", {}, "Analyzing your lease"), el("p", { class: "muted" }, "Usually under a minute.")),
    progress,
    el("ol", { class: "steps" }, items),
    live, note,
  ));
  poll();
}
