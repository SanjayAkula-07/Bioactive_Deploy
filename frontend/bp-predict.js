/* Single molecule predictor: live /predict + 3Dmol viewer */
(function () {
  const $ = id => document.getElementById(id);
  const input = $("compound-input"), btn = $("predict-btn"), clearBtn = $("clear-btn");
  const smilesPrev = $("resolved-smiles-preview");
  const container = $("mol-canvas-container");
  let viewer = null, spinning = false, style = "stick", lastMol = "";

  // ---- 3D viewer -------------------------------------------------
  const vdiv = document.createElement("div");
  vdiv.style.cssText = "position:absolute;inset:0;width:100%;height:100%";
  container.querySelectorAll(":scope > svg").forEach(s => s.style.display = "none");
  container.prepend(vdiv);
  // remove made-up HUD text (angles / RMSD vs PDB) from the mockup
  container.querySelectorAll("span").forEach(s => {
    if (/RMSD|Angles/.test(s.textContent)) { const hud = s.closest("div"); if (hud) hud.remove(); }
  });
  const note = document.createElement("div");
  note.style.cssText = "position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#64748b;font-size:13px;pointer-events:none";
  note.textContent = "Run a prediction to see the 3D structure";
  container.appendChild(note);

  function applyStyle() {
    if (!viewer) return;
    viewer.setStyle({}, style === "stick"
      ? { stick: { radius: 0.15 }, sphere: { scale: 0.28 } }
      : { sphere: { scale: 0.9 } });
    viewer.render();
  }
  function render3D(molblock) {
    lastMol = molblock || "";
    if (!window.$3Dmol) { note.textContent = "3D viewer library failed to load (check internet)"; note.style.display = "flex"; return; }
    if (!lastMol) { note.textContent = "No 3D coordinates available"; note.style.display = "flex"; return; }
    if (!viewer) viewer = $3Dmol.createViewer(vdiv, { backgroundColor: "0xf8fafc" });
    viewer.clear();
    viewer.addModel(lastMol, "mol");
    applyStyle();
    viewer.zoomTo();
    viewer.render();
    if (spinning) viewer.spin("y", 1);
    note.style.display = "none";
  }
  $("toggle-spin").addEventListener("click", () => {
    spinning = !spinning;
    $("toggle-spin").classList.toggle("bg-emerald-50", spinning);
    if (viewer) viewer.spin(spinning ? "y" : false, 1);
  });
  $("reset-cam").addEventListener("click", () => { if (viewer) { viewer.zoomTo(); viewer.render(); } });
  $("switch-style").addEventListener("click", () => {
    style = style === "stick" ? "sphere" : "stick";
    const label = $("switch-style").querySelector("span:last-child");
    if (label) label.textContent = style === "stick" ? "Ball & Stick" : "Spacefill (CPK)";
    applyStyle();
  });

  // ---- prediction ------------------------------------------------
  function setVerdict(active, pActive) {
    const badge = $("verdict-badge"), bar = $("confidence-bar");
    $("verdict-text").textContent = active ? "ACTIVE" : "INACTIVE";
    badge.className = "inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full border shadow-sm " +
      (active ? "bg-emerald-50 border-emerald-200 text-emerald-800" : "bg-rose-50 border-rose-200 text-rose-800");
    bar.className = "h-full rounded-full transition-all duration-700 shadow-sm bg-gradient-to-r " +
      (active ? "from-sky-500 to-emerald-500" : "from-rose-300 to-rose-500");
    bar.style.width = (pActive * 100).toFixed(1) + "%";
    $("confidence-percentage").textContent = (pActive * 100).toFixed(1) + "%";
  }
  function showError(msg) {
    let box = $("bp-error");
    if (!box) {
      box = document.createElement("div");
      box.id = "bp-error";
      box.className = "mt-3 px-3 py-2 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-sm";
      btn.parentNode.insertBefore(box, btn.nextSibling);
    }
    box.textContent = msg || "";
    box.style.display = msg ? "block" : "none";
  }

  let busy = false;
  async function predict(value) {
    value = (value || "").trim();
    if (!value || busy) { if (!value) showError("Please enter a molecule name or SMILES string."); return; }
    busy = true; showError("");
    const orig = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<span class="material-symbols-outlined text-2xl animate-spin">refresh</span><span>Calculating fingerprints &amp; inference...</span>';
    const t0 = performance.now();
    try {
      const res = await fetch(BP_API + "/predict", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles: value })
      });
      const d = await res.json();
      if (!res.ok || d.detail) throw new Error(typeof d.detail === "string" ? d.detail : "Server error (" + res.status + ")");
      const active = d.prediction === "Active";
      let pActive = (typeof d.active_prob === "number") ? d.active_prob
        : (typeof d.score === "number" ? (active ? d.score : 1 - d.score) : (active ? 1 : 0));
      setVerdict(active, pActive);
      $("target-compound-name").textContent = d.molecule_name || value;
      smilesPrev.textContent = d.smiles || "";
      const mi = d.mol_info || {};
      $("prop-mw").textContent = mi.mol_weight != null ? Number(mi.mol_weight).toFixed(2) : "N/A";
      $("prop-logp").textContent = mi.logp != null ? mi.logp : "N/A";
      $("prop-hac").textContent = mi.heavy_atoms != null ? mi.heavy_atoms : (mi.num_atoms != null ? mi.num_atoms : "N/A");
      const lat = $("bp-latency"); if (lat) lat.textContent = Math.round(performance.now() - t0);
      render3D(d.mol_3d);
    } catch (err) {
      showError(err.message || "Prediction failed.");
    } finally {
      busy = false; btn.disabled = false; btn.innerHTML = orig;
    }
  }

  btn.addEventListener("click", () => predict(input.value));
  input.addEventListener("keydown", e => { if (e.key === "Enter") predict(input.value); });
  clearBtn.addEventListener("click", () => { input.value = ""; input.focus(); });
  document.querySelectorAll(".example-btn").forEach(b => b.addEventListener("click", () => {
    input.value = b.getAttribute("data-compound") || b.getAttribute("data-smiles");
    predict(input.value);
  }));

  // Initial state: blank until the user runs a prediction
  $("target-compound-name").textContent = "—";
  $("verdict-text").textContent = "—";
  $("verdict-badge").className = "inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full border shadow-sm bg-slate-50 border-slate-200 text-slate-600";
  $("confidence-percentage").textContent = "—";
  $("confidence-bar").style.width = "0%";
  ["prop-mw", "prop-logp", "prop-hac"].forEach(i => $(i).textContent = "—");
  smilesPrev.textContent = "";
  predict(input.value);   // run the pre-filled example once on load
})();
