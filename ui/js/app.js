import { session } from "./api.js";
import { signIn } from "./auth.js";
import { el } from "./dom.js";
import { go, route, start } from "./router.js";
import { renderAnalyzing } from "./screens/analyzing.js";
import { renderHistory } from "./screens/history.js";
import { renderHome } from "./screens/home.js";
import { renderResults } from "./screens/results.js";
import { activeJob } from "./state.js";

const nav = document.getElementById("nav");

function renderNav() {
  if (session.token) {
    const signOut = el("button", { type: "button", class: "btn link" }, "Sign out");
    signOut.addEventListener("click", () => {
      session.clear();
      activeJob.clear();
      go("#/");
    });
    nav.replaceChildren(el("a", { href: "#/history" }, "History"), signOut);
  } else {
    const signInBtn = el("button", { type: "button", class: "btn link" }, "Sign in");
    signInBtn.addEventListener("click", () => signIn());
    nav.replaceChildren(signInBtn);
  }
}

route(/^\/$/, renderHome);
route(/^\/analyzing\/([\w-]+)$/, renderAnalyzing);
route(/^\/results\/(\d+)$/, renderResults);
route(/^\/history$/, renderHistory);

document.addEventListener("authchange", renderNav);
renderNav();
start(document.getElementById("app"));
