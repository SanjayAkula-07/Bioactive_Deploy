/* Batch virtual screening: live /predict_batch */
(function () {
  const $ = id => document.getElementById(id);
  const esc = BP_escape;
  const SAMPLE = "id,molecule_name,smiles,expected_activity\n" +
    "1,Ethanol,CCO,Inactive\n2,Aspirin,CC(=O)Oc1ccccc1C(=O)O,Active\n3,Benzene,c1ccccc1,Inactive\n" +
    "4,Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,Active\n5,Celecoxib_Analog,CS(=O)(=O)c1ccc(-c2csc(CC(=O)O)c2-c2ccc(F)cc2)cc1,Active\n" +
    "6,Acetaminophen,CC(=O)Nc1ccc(O)cc1,Active\n7,Naproxen,COc1ccc2cc(C(C)C(=O)O)ccc2c1,Active\n" +
    "8,Caffeine,Cn1cnc2c1c(=O)n(C)c(=O)n2C,Inactive\n9,Invalid_Sample,INVALID_CHEM_STRING,Error\n10,Propane,CCC,Inactive\n";

  let file = null, names = [], results = [], activeOnly = false;
  const tbody = $("screeningTableBody");

  // ---- tidy mock text from the design ----------------------------
  const dz = $("dropZone");
  dz.innerHTML = dz.innerHTML
    .replace(/Upload CSV containing SMILES or Compound Name column/, "Upload a CSV file with a <code>smiles</code> column")
    .replace(/\(<span[^>]*>\.csv, \.tsv, \.smi<\/span> supported up to 250MB\)/, "(<span class=\"font-label-code font-medium\">.csv</span>, max 5 MB)")
    .replace(/Tanimoto Deduplication/, "Row-level SMILES validation");
  document.querySelectorAll("span").forEach(s => {
    const t = s.textContent.trim();
    if (t === "Pipeline v2.4") s.textContent = "XGBoost · ECFP4";
    if (t === "Engine Latency" ) { const v = s.nextElementSibling; if (v) { v.id = "bp-latency"; v.textContent = "—"; } }
    if (t === "Parallel Workers") { const box = s.parentElement, sep = box.previousElementSibling; if (sep) sep.remove(); box.remove(); }
  });

  // ---- helpers to update the stat cards --------------------------
  function card(label) {
    return [...document.querySelectorAll("span")].find(s => s.textContent.trim() === label)?.closest("div.p-space-lg");
  }
  function setStat(label, n, total, unit) {
    const c = card(label); if (!c) return;
    const nums = c.querySelectorAll("span.text-3xl"); if (nums[0]) nums[0].textContent = n;
    const u = nums[0]?.nextElementSibling;
    if (u) u.textContent = unit + (total ? ` (${Math.round(100 * n / total)}%)` : "");
    const bar = c.querySelector(".h-1\\.5 > div");
    if (bar) { bar.className = bar.className.replace(/\bw-\S+/g, "").trim(); bar.style.width = (total ? 100 * n / total : 0) + "%"; }
  }
  function setLabelValue(label, text) {
    const s = [...document.querySelectorAll("span")].find(x => x.textContent.trim() === label);
    const v = s && s.nextElementSibling; if (v) v.textContent = text;
  }
  function stats() {
    const total = results.length;
    const act = results.filter(r => r.prediction === "Active").length;
    const inact = results.filter(r => r.prediction === "Inactive").length;
    const err = results.filter(r => r.status !== "ok").length;
    setStat("Total Screened", total, total, "Compounds");
    setStat("Predicted Active", act, total, "Compounds");
    setStat("Predicted Inactive", inact, total, "Compounds");
    setStat("Validation Errors", err, total, "Invalid SMILES");
    const ok = total ? Math.round(100 * (total - err) / total) : 0;
    document.querySelectorAll("span").forEach(s => {
      if (/^Displaying /.test(s.textContent.trim())) s.textContent = `Displaying ${total} of ${total} items`;
      if (/RDKit MolCheck/.test(s.textContent)) s.textContent = `RDKit MolCheck: ${ok}% valid`;
    });
  }
  function fileCard() {
    $("loadedFileName").textContent = file ? file.name : "No file loaded";
    const sub = $("loadedFileName").nextElementSibling;
    if (sub) sub.textContent = file ? (file.size / 1024).toFixed(1) + " KB • CSV" : "—";
    setLabelValue("Molecules Detected:", file ? names.length + " compounds" : "—");
  }

  // ---- CSV parsing (names only; predictions come from the API) ---
  function parseCSV(text) {
    const rows = []; let cur = [], f = "", q = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (q) { if (c === '"') { if (text[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += c; }
      else if (c === '"') q = true;
      else if (c === ",") { cur.push(f); f = ""; }
      else if (c === "\n" || c === "\r") { if (c === "\r" && text[i + 1] === "\n") i++; cur.push(f); f = ""; if (cur.some(x => x !== "")) rows.push(cur); cur = []; }
      else f += c;
    }
    if (f !== "" || cur.length) { cur.push(f); rows.push(cur); }
    return rows;
  }
  async function setFile(f) {
    if (!/\.csv$/i.test(f.name)) { toast("Please choose a .csv file"); return; }
    file = f;
    const rows = parseCSV(await f.text());
    const head = (rows[0] || []).map(h => h.trim().toLowerCase());
    const ni = ["molecule_name", "name", "compound_name", "compound", "id"].map(k => head.indexOf(k)).find(i => i >= 0) ?? -1;
    names = rows.slice(1).map((r, i) => (ni >= 0 && r[ni]) ? r[ni] : "Compound " + (i + 1));
    fileCard(); results = []; render(); stats();
  }

  // ---- table -----------------------------------------------------
  function render() {
    const q = $("searchInput").value.trim().toLowerCase();
    const rows = results.filter(r => {
      const nm = (names[r.row - 1] || "").toLowerCase();
      const hit = !q || nm.includes(q) || (r.smiles || "").toLowerCase().includes(q);
      return hit && (!activeOnly || r.prediction === "Active");
    });
    if (!results.length) {
      tbody.innerHTML = '<tr><td colspan="7" class="py-10 text-center text-sm text-[#64748b]">' +
        (file ? "Click “Process Virtual Screening” to run predictions." : "Upload a CSV or click “Load Sample CSV” to begin.") + "</td></tr>";
      return;
    }
    tbody.innerHTML = rows.map(r => {
      const nm = names[r.row - 1] || "Compound " + r.row;
      const ok = r.status === "ok", act = r.prediction === "Active";
      const badge = !ok
        ? '<span class="px-2.5 py-0.5 rounded-full bg-[#fef2f2] text-[#dc2626] border border-[#fecaca] font-label-badge text-xs font-semibold">ERROR</span>'
        : act
          ? '<span class="px-2.5 py-0.5 rounded-full bg-[#dcfce7] text-[#15803d] border border-[#86efac] font-label-badge text-xs font-bold shadow-sm">ACTIVE</span>'
          : '<span class="px-2.5 py-0.5 rounded-full bg-[#f1f5f9] text-[#475569] border border-[#e2e8f0] font-label-badge text-xs font-semibold">INACTIVE</span>';
      const status = ok
        ? '<span class="inline-flex items-center gap-1.5 text-[#059669] font-label-badge text-xs font-semibold"><span class="w-2 h-2 rounded-full bg-[#059669]"></span> Valid</span>'
        : `<span class="inline-flex items-center gap-1.5 text-[#dc2626] font-label-badge text-xs font-semibold" title="${esc(r.error)}"><span class="w-2 h-2 rounded-full bg-[#dc2626]"></span> Invalid</span>`;
      const score = ok && r.score != null ? (r.score * 100).toFixed(1) + "%" : "—";
      return `<tr class="hover:bg-[#f8fafc] transition-colors ${ok && act ? "bg-[#ecfdf5]/40" : ""}">
        <td class="py-3.5 px-space-md font-label-code text-xs text-[#64748b]">${String(r.row).padStart(2, "0")}</td>
        <td class="py-3.5 px-space-md font-semibold text-[#0f172a]">${esc(nm)}</td>
        <td class="py-3.5 px-space-md font-label-code text-xs text-[#0284c7] hover:underline cursor-pointer" data-copy="${esc(r.smiles)}">${esc(r.smiles)}</td>
        <td class="py-3.5 px-space-md">${status}</td>
        <td class="py-3.5 px-space-md">${badge}</td>
        <td class="py-3.5 px-space-md text-right font-label-code text-xs font-semibold text-[#0f172a]">${score}</td>
        <td class="py-3.5 px-space-md text-center">${ok ? `<button class="p-1.5 rounded-lg hover:bg-[#f1f5f9] text-[#64748b] hover:text-[#059669] transition-colors" data-view="${esc(r.smiles)}" data-name="${esc(nm)}" title="Inspect 3D conformation"><span class="material-symbols-outlined text-lg">visibility</span></button>` : ""}</td>
      </tr>`;
    }).join("");
  }
  tbody.addEventListener("click", e => {
    const c = e.target.closest("[data-copy]"); if (c) copySmiles(c.getAttribute("data-copy"));
    const v = e.target.closest("[data-view]"); if (v) previewMol(v.getAttribute("data-name"), v.getAttribute("data-view"));
  });

  // ---- actions ---------------------------------------------------
  function toast(msg) {
    $("toastMessage").textContent = msg; const t = $("toastNotification");
    t.classList.remove("translate-y-24", "opacity-0"); t.classList.add("translate-y-0", "opacity-100");
    setTimeout(() => { t.classList.add("translate-y-24", "opacity-0"); t.classList.remove("translate-y-0", "opacity-100"); }, 2600);
  }
  window.copySmiles = s => (navigator.clipboard ? navigator.clipboard.writeText(s) : Promise.reject()).then(() => toast("SMILES copied: " + s)).catch(() => toast("Copy failed"));

  $("btnLoadSample").addEventListener("click", () => setFile(new File([SAMPLE], "sample_molecules.csv", { type: "text/csv" })));
  $("btnClearData").addEventListener("click", () => { file = null; names = []; results = []; activeOnly = false; $("searchInput").value = ""; fileCard(); render(); stats(); });
  const pick = document.createElement("input"); pick.type = "file"; pick.accept = ".csv"; pick.style.display = "none"; document.body.appendChild(pick);
  pick.addEventListener("change", () => { if (pick.files[0]) setFile(pick.files[0]); pick.value = ""; });
  dz.addEventListener("click", () => pick.click());
  dz.addEventListener("dragover", e => { e.preventDefault(); });
  dz.addEventListener("drop", e => { e.preventDefault(); if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); });

  $("btnProcessScreening").addEventListener("click", async () => {
    if (!file) { toast("Load a CSV first"); return; }
    const b = $("btnProcessScreening"), orig = b.innerHTML; b.disabled = true;
    b.innerHTML = '<span class="material-symbols-outlined text-lg animate-spin">refresh</span><span>Screening...</span>';
    const t0 = performance.now();
    try {
      const fd = new FormData(); fd.append("file", file);
      const res = await fetch(BP_API + "/predict_batch", { method: "POST", body: fd });
      const d = await res.json();
      if (!res.ok) throw new Error(typeof d.detail === "string" ? d.detail : "Batch failed (" + res.status + ")");
      results = d.results || []; activeOnly = false; render(); stats();
      const lat = $("bp-latency"); if (lat && results.length) lat.textContent = Math.round((performance.now() - t0) / results.length) + " ms/mol";
      toast("Screened " + results.length + " compounds");
    } catch (err) { toast(err.message || "Batch failed"); }
    finally { b.disabled = false; b.innerHTML = orig; }
  });

  $("searchInput").addEventListener("input", render);
  $("filterActiveBtn").addEventListener("click", () => { activeOnly = true; render(); toast("Showing only active predictions"); });
  $("filterResetBtn").addEventListener("click", () => { activeOnly = false; $("searchInput").value = ""; render(); });
  $("btnExportCSV").addEventListener("click", () => {
    if (!results.length) { toast("Nothing to export yet"); return; }
    const q = v => '"' + String(v == null ? "" : v).replace(/"/g, '""') + '"';
    const lines = [["Row", "Name", "SMILES", "Status", "Prediction", "Confidence", "Error"].join(",")]
      .concat(results.map(r => [r.row, q(names[r.row - 1] || ""), q(r.smiles), r.status, r.prediction || "", r.score == null ? "" : r.score, q(r.error)].join(",")));
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
    a.download = "biopredict_screening_" + Date.now() + ".csv"; a.click();
  });
  const sdf = $("btnDownloadSDF"); if (sdf) sdf.style.display = "none";   // not provided by the API

  // ---- modal 3D preview (uses /predict for the 3D block) ---------
  const modal = $("quickMolModal");
  const stage = modal.querySelector(".h-64");
  stage.querySelectorAll(":scope > svg").forEach(s => s.remove());
  stage.querySelectorAll("span").forEach(s => { if (/MMFF94/.test(s.textContent)) { s.closest("div").remove(); } });
  const mdiv = document.createElement("div"); mdiv.style.cssText = "position:absolute;inset:0"; stage.prepend(mdiv);
  const props = [...modal.querySelectorAll(".grid > div")];
  let mviewer = null;
  window.closeModal = () => { modal.classList.add("hidden"); modal.classList.remove("flex"); };
  modal.addEventListener("click", e => { if (e.target === modal) closeModal(); });
  window.previewMol = async (name, smiles) => {
    $("modalCompoundName").textContent = name; $("modalCompoundSmiles").textContent = smiles;
    modal.classList.remove("hidden"); modal.classList.add("flex");
    props.forEach(p => { p.querySelector("span:last-child").textContent = "…"; });
    try {
      const res = await fetch(BP_API + "/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ smiles }) });
      const d = await res.json(); if (!res.ok) throw new Error(d.detail);
      const mi = d.mol_info || {};
      const vals = [mi.mol_weight != null ? mi.mol_weight + " g/mol" : "N/A", mi.logp != null ? mi.logp : "N/A", mi.heavy_atoms != null ? mi.heavy_atoms : "N/A"];
      const labs = ["Mol Weight", "LogP", "Heavy Atoms"];
      props.forEach((p, i) => { p.querySelector("span:first-child").textContent = labs[i]; p.querySelector("span:last-child").textContent = vals[i]; });
      if (window.$3Dmol && d.mol_3d) {
        if (!mviewer) mviewer = $3Dmol.createViewer(mdiv, { backgroundColor: "0xf8fafc" });
        mviewer.clear(); mviewer.addModel(d.mol_3d, "mol");
        mviewer.setStyle({}, { stick: { radius: 0.15 }, sphere: { scale: 0.28 } });
        mviewer.zoomTo(); mviewer.render(); mviewer.spin("y", 1);
      }
    } catch (e) { toast("Preview failed: " + (e.message || "error")); }
  };

  // initial state
  fileCard(); render(); stats();
})();
