const routes = [];
let root = null;
let controller = null;
let firstRender = true;

export function route(pattern, render) {
  routes.push({ pattern, render });
}

export function go(hash, { replace = false } = {}) {
  if (location.hash === hash) return refresh();
  if (replace) {
    history.replaceState(null, "", hash);
    refresh();
  } else {
    location.hash = hash;
  }
}

// Each render gets an AbortSignal; navigating away aborts it, which cancels fetches and polling.
export function refresh() {
  controller?.abort();
  controller = new AbortController();
  const path = location.hash.slice(1) || "/";
  const match = routes.find((r) => r.pattern.test(path));
  if (!match) return go("#/", { replace: true });

  const { signal } = controller;
  const moveFocus = !firstRender;
  firstRender = false;
  root.replaceChildren();
  window.scrollTo(0, 0);

  Promise.resolve(match.render({ root, signal, params: path.match(match.pattern).slice(1) })).then(() => {
    const heading = root.querySelector("h1");
    if (signal.aborted || !heading) return;
    document.title = path === "/" ? "Leazard · Check your lease before you sign" : `${heading.textContent} · Leazard`;
    if (moveFocus) {
      heading.tabIndex = -1;
      heading.focus();
    }
  });
}

export function start(rootEl) {
  root = rootEl;
  window.addEventListener("hashchange", refresh);
  refresh();
}
