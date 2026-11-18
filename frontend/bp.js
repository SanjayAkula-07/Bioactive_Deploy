/* BioPredict shared: API base, login gate, sidebar navigation */
(function () {
  window.BP_API = (location.protocol === "file:") ? "http://127.0.0.1:8000" : location.origin;

  const PAGES = {
    "single-predictor": "predict.html",
    "batch-screening": "batch.html",
    "model-benchmarks": "benchmarks.html",
    "documentation-and-guide": "docs.html",
    "login-and-security": "login.html"
  };
  const here = location.pathname.split("/").pop() || "index.html";

  // Demo login gate: a session flag set on the login page.
  // NOTE: this is a front-end gate for the demo UI, not server-side authentication.
  function loggedIn() {
    try { return !!sessionStorage.getItem("bp_user"); } catch (e) { return true; }
  }
  window.BP_login = function (name) {
    try { sessionStorage.setItem("bp_user", name || "researcher"); } catch (e) {}
  };
  window.BP_logout = function () {
    try { sessionStorage.removeItem("bp_user"); } catch (e) {}
    location.href = "login.html";
  };
  if (here !== "login.html" && !loggedIn()) {
    location.replace("login.html");
    return;
  }

  window.BP_escape = function (s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, c => (
      { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  };

  document.addEventListener("DOMContentLoaded", () => {
    const nav = document.querySelector("aside nav");
    if (nav) {
      const activeCls = (nav.getAttribute("data-active-classes") ||
        "bg-emerald-50 text-emerald-700 font-semibold border border-emerald-200 shadow-sm").split(" ");
      nav.querySelectorAll("a[data-path]").forEach(a => {
        const target = PAGES[a.getAttribute("data-path")];
        if (!target) return;
        a.setAttribute("href", target);
        const on = target === here;
        a.classList.remove(...activeCls);
        if (on) { a.classList.add(...activeCls); a.setAttribute("aria-current", "page"); }
        else { a.removeAttribute("aria-current"); }
      });
    }
    // Logout link (icon text "logout") in the sidebar footer
    document.querySelectorAll("aside a, aside button").forEach(el => {
      if (/\blogout\b/.test(el.textContent)) {
        el.addEventListener("click", e => { e.preventDefault(); window.BP_logout(); });
        if (el.tagName === "A") el.setAttribute("href", "login.html");
      }
    });
  });
})();
