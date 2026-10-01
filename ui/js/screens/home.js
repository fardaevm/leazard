import { api, session } from "../api.js";
import { signIn } from "../auth.js";
import { el } from "../dom.js";
import { go } from "../router.js";
import { activeJob, draft } from "../state.js";

const MAX_BYTES = 10 * 1024 * 1024;

function fileError(file) {
  if (!file) return "Choose your lease PDF.";
  const isPdf = file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
  if (!isPdf) return "That file isn't a PDF. Please choose a PDF of your lease.";
  if (file.size > MAX_BYTES) return "That file is over 10 MB. Please choose a smaller PDF.";
  if (file.size === 0) return "That file is empty. Please choose another PDF.";
  return "";
}

function zipError(zip) {
  return /^\d{5}$/.test(zip) ? "" : "Enter a 5-digit ZIP code.";
}

function formatSize(bytes) {
  return bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function renderHome({ root, signal }) {
  const fileInput = el("input", {
    id: "lease", type: "file", class: "sr-only", accept: "application/pdf,.pdf",
    "aria-describedby": "lease-error",
  });
  const dropTitle = el("span", { class: "drop-title" });
  const dropHint = el("span", { class: "muted small" });
  const drop = el("label", { for: "lease", class: "drop" }, dropTitle, dropHint);
  const fileErr = el("p", { id: "lease-error", class: "error", hidden: true });
  const fileField = el("div", {}, fileInput, drop, fileErr);

  const zipInput = el("input", {
    id: "zip", name: "zip_code", type: "text", class: "zip", inputmode: "numeric",
    autocomplete: "postal-code", maxlength: "5", value: draft.zip, "aria-describedby": "zip-error",
  });
  const zipErr = el("p", { id: "zip-error", class: "error", hidden: true });

  const submit = el("button", { type: "submit", class: "btn primary block" }, "Analyze my lease");
  const formErr = el("p", { class: "error form-error", role: "alert", hidden: true });

  function showFile() {
    const file = draft.file;
    dropTitle.textContent = file ? file.name : "Choose your lease PDF";
    dropHint.textContent = file ? `${formatSize(file.size)} · Click to choose a different file` : "or drag it here · PDF up to 10 MB";
    drop.classList.toggle("has-file", Boolean(file));
  }

  function setError(node, input, message) {
    node.textContent = message;
    node.hidden = !message;
    input.setAttribute("aria-invalid", String(Boolean(message)));
    if (input === fileInput) fileField.classList.toggle("field-invalid", Boolean(message));
  }

  function pickFile(file) {
    draft.file = file || null;
    showFile();
    setError(fileErr, fileInput, file ? fileError(file) : "");
  }

  fileInput.addEventListener("change", () => pickFile(fileInput.files[0]));
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("is-over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("is-over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("is-over");
    pickFile(e.dataTransfer.files[0]);
  });
  zipInput.addEventListener("input", () => {
    zipInput.value = zipInput.value.replace(/\D/g, "").slice(0, 5);
    draft.zip = zipInput.value;
    if (zipErr.textContent && !zipError(draft.zip)) setError(zipErr, zipInput, "");
  });

  function showFormError(message, { withJobLink = false } = {}) {
    const jobId = activeJob.get();
    formErr.replaceChildren(message);
    if (withJobLink && jobId) formErr.append(" ", el("a", { href: `#/analyzing/${jobId}` }, "See its progress"));
    formErr.hidden = !message;
  }

  async function upload() {
    submit.disabled = true;
    submit.textContent = "Uploading…";
    const body = new FormData();
    body.append("lease", draft.file);
    body.append("zip_code", draft.zip);
    try {
      const { job_id } = await api("/jobs", { method: "POST", body, signal });
      activeJob.set(job_id);
      go(`#/analyzing/${job_id}`);
    } catch (err) {
      if (signal.aborted) return;
      submit.disabled = false;
      submit.textContent = "Analyze my lease";
      if (err.status === 401) {
        if (await signIn("Your session has expired. Sign in again to continue.")) upload();
        return;
      }
      showFormError(err.message, { withJobLink: err.status === 429 });
    }
  }

  const form = el("form", { class: "stack loose", novalidate: true }, fileField,
    el("div", {}, el("label", { for: "zip" }, "ZIP code of the rental"), zipInput, zipErr),
    el("div", { class: "stack" }, submit, formErr,
      el("p", { class: "muted small" }, "Informational only, not legal advice. Your lease is private to your account.")),
  );

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    showFormError("");
    draft.zip = zipInput.value.trim();
    const fErr = fileError(draft.file);
    const zErr = zipError(draft.zip);
    setError(fileErr, fileInput, fErr);
    setError(zipErr, zipInput, zErr);
    if (fErr || zErr) return (fErr ? fileInput : zipInput).focus();
    if (!(await signIn())) return;
    upload();
  });

  const resume = activeJob.get() && session.token
    ? el("p", { class: "card small" }, "You have an analysis in progress. ",
        el("a", { href: `#/analyzing/${activeJob.get()}` }, "See its progress"))
    : null;

  showFile();
  root.append(el("div", { class: "stack loose" },
    resume,
    el("div", { class: "stack hero" },
      el("h1", {}, "Upload your lease before you sign. See what's risky in 60 seconds."),
      el("p", { class: "lede" }, "We read your lease, flag clauses that could cost you, and draft an email to negotiate them.")),
    form,
  ));
}
