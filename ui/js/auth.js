import { api, session } from "./api.js";
import { go } from "./router.js";

const dialog = document.getElementById("auth");
const form = document.getElementById("auth-form");
const title = document.getElementById("auth-title");
const sub = document.getElementById("auth-sub");
const user = document.getElementById("auth-user");
const pass = document.getElementById("auth-pass");
const passHint = document.getElementById("auth-pass-hint");
const errorEl = document.getElementById("auth-error");
const submit = document.getElementById("auth-submit");
const toggle = document.getElementById("auth-toggle");
const DEFAULT_SUB = sub.textContent;

let mode = "login";
let waiters = [];

function setMode(next) {
  mode = next;
  const register = mode === "register";
  title.textContent = register ? "Create your account" : "Sign in to continue";
  submit.textContent = register ? "Create account" : "Sign in";
  toggle.textContent = register ? "I already have an account" : "Create an account";
  pass.autocomplete = register ? "new-password" : "current-password";
  passHint.hidden = !register;
  showError("");
}

function showError(message) {
  errorEl.textContent = message;
  errorEl.hidden = !message;
}

function validate() {
  const name = user.value.trim();
  if (name.length < 3) return [user, "Username must be at least 3 characters."];
  if (!pass.value) return [pass, "Enter your password."];
  if (mode === "register" && pass.value.length < 6) return [pass, "Password must be at least 6 characters."];
  return null;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const invalid = validate();
  user.setAttribute("aria-invalid", String(invalid?.[0] === user));
  pass.setAttribute("aria-invalid", String(invalid?.[0] === pass));
  if (invalid) {
    showError(invalid[1]);
    invalid[0].focus();
    return;
  }

  const credentials = { username: user.value.trim(), password: pass.value };
  submit.disabled = true;
  showError("");
  try {
    if (mode === "register") await api("/auth/register", { method: "POST", json: credentials });
    const { access_token } = await api("/auth/login", { method: "POST", json: credentials });
    session.set(access_token);
    dialog.close();
  } catch (err) {
    showError(err.message);
  } finally {
    submit.disabled = false;
  }
});

toggle.addEventListener("click", () => setMode(mode === "login" ? "register" : "login"));
document.getElementById("auth-cancel").addEventListener("click", () => dialog.close());

dialog.addEventListener("close", () => {
  const signedIn = Boolean(session.token);
  waiters.forEach((resolve) => resolve(signedIn));
  waiters = [];
});

/** Opens the dialog (if needed) and resolves true once signed in, false if dismissed. */
export function signIn(message) {
  if (session.token) return Promise.resolve(true);
  if (!dialog.open) {
    form.reset();
    setMode("login");
    sub.textContent = message || DEFAULT_SUB;
    [user, pass].forEach((input) => input.removeAttribute("aria-invalid"));
    dialog.showModal();
  }
  return new Promise((resolve) => waiters.push(resolve));
}

/** For screens that need auth: re-sign-in, then retry; dismissing goes home. */
export async function requireSignIn(retry, message) {
  if (await signIn(message)) retry();
  else go("#/");
}
