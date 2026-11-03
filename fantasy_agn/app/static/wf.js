/* FANTASY: step-by-step interface. Every action writes Python code that runs in the
   session (shown in the session panel, hidden by default). */
"use strict";
const $ = (id) => document.getElementById(id);
const TOKEN = document.querySelector('meta[name="fantasy-token"]').content;
const STEPS = [["read", "Read"], ["prepare", "Prepare"], ["model", "Model"], ["fit", "Fit"], ["results", "Results"]];
const MODE_LABEL = { single: "single spectrum", sample: "sample", multi: "multi-epoch", classify: "classification" };
const C = {
  // the palette of the figures (plotting.COLORS): the six line colours differ for normal vision and the
  // colour-vision deficiencies; several broad components are steps of the broad blue; the model is ink
  // over grey data, and selections use the red accent
  data: "#9b9a95", model: "#0b0b0b", continuum: "#52514e", host: "#008300", narrow: "#1baf7a", feii: "#e34948",
  outflow: "#eda100", broad: ["#2a78d6", "#104281", "#6da7ec", "#1c5cab", "#9ec5f4"], balmer: "#86b6ef",
  bowen: "#4a3aa7", other: ["#898781", "#86b6ef", "#c9c8c2"], accent: "#e34948",
};
let S = null;                 // server state
let page = null;              // current step
let wasBusy = false, pollTimer = null, pendingGo = null;
let lastError = null;

// ------------------------------------------------------------------ helpers
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const withToken = (url) => url + (url.includes("?") ? "&" : "?") + "t=" + encodeURIComponent(TOKEN);
function fmtNum(v, d = 4) {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(2);
  return Number(v.toPrecision(d)).toString();
}
function pm(v, e, d = 4) {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  return fmtNum(v, d) + (e !== null && e !== undefined && isFinite(e) && e > 0 ? " ± " + fmtNum(e, 2) : "");
}
function toast(msg, ms = 2600) {
  const t = $("toast"); t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(toast.timer); toast.timer = setTimeout(() => t.classList.add("hidden"), ms);
}
async function api(path, options = {}) {
  const headers = Object.assign({ "X-Fantasy-Token": TOKEN }, options.headers || {});
  const res = await fetch(path, Object.assign({}, options, { headers }));
  const type = res.headers.get("Content-Type") || "";
  const body = type.includes("json") ? await res.json() : await res.text();
  if (!res.ok) throw new Error((body && body.error) || res.statusText);
  return body;
}
async function post(path, body) {
  try {
    const st = await api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
    setState(st);
    return true;
  } catch (e) { showError(e.message); return false; }
}
function download(url) {
  const a = document.createElement("a"); a.href = withToken(url); a.download = ""; document.body.appendChild(a); a.click(); a.remove();
}
function showError(msg) {
  const b = $("banner");
  b.innerHTML = `<pre>${esc(msg)}</pre><button id="banner-session">Show in session</button><button id="banner-x">Dismiss</button>`;
  b.classList.remove("hidden");
  $("banner-session").onclick = () => setSession(true);
  $("banner-x").onclick = () => { b.classList.add("hidden"); lastError = S && S.error; };
}
function prettyLine(n) {
  const greek = { a: "α", b: "β", g: "γ", d: "δ", e: "ε" };
  let m = /^(H|Ly|Pa|Br|Pf)([abgde])$/.exec(n);
  if (m) return m[1] + greek[m[2]];
  m = /^H2-(\d-\d)([SQ]\(\d+\))$/.exec(n);
  if (m) return `H₂ ${m[1]} ${m[2]}`;
  m = /^(\[?)([A-Z][a-z]?)([IVX]+)(\]?)(\d+)$/.exec(n);
  if (m) return `${m[1]}${m[2]} ${m[3]}${m[4]} ${m[5]}`;
  return n;
}

// ------------------------------------------------------------------- state
async function refresh() {
  try { setState(await api("/api/state")); }
  catch (e) {
    if (e.message === "forbidden") {          // a restarted server has a new token
      showError("The FANTASY server was restarted: reload the page to continue.");
      return;
    }
    /* server not reachable: keep trying */
  }
  schedule();
}
function schedule() {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(refresh, S && S.busy ? 400 : 3000);
}
function setState(st) {
  const prev = S;
  S = st;
  const busyNow = !!S.busy;
  if (wasBusy && !busyNow) onIdle(prev);
  wasBusy = busyNow;
  if (S.error && S.error !== lastError) { showError(S.error); lastError = S.error; pendingGo = null; }
  if (!S.error) { lastError = null; $("banner").classList.add("hidden"); }
  render();
  schedule();
}
function onIdle() {
  if (pendingGo && !S.error) { const g = pendingGo; pendingGo = null; go(g); }
  else if (page) loadPageData(true);
  if (pendingSuggest && !S.error) { pendingSuggest = false; classInfo = null; if (page === "model") renderModel(); }
  if (pendingSuggest && S.error) pendingSuggest = false;
}
const mode = () => (S && S.wf && S.wf.mode) || "single";
const loaded = () => S && S.wf && S.wf.loaded;
const prepared = () => S && S.wf && S.wf.prepared;
const fitDone = () => S && S.wf && S.wf.fit;

function stepDone(step) {
  if (!S) return false;
  if (step === "read") return !!loaded();
  if (step === "prepare") return !!prepared();
  if (step === "model") return ["sample", "classify"].includes(mode()) ? !!prepared() || (mode() === "sample" && !!loaded()) : !!S.model;
  return !!fitDone();
}
function stepSub(step) {
  const L = loaded(), P = prepared(), F = fitDone();
  if (step === "read") return L ? `${L.n} ${L.n === 1 ? "file" : "files"}` : "—";
  if (step === "prepare") {
    if (!P) return "z, E(B−V), masks";
    return [P.z != null ? "z " + fmtNum(P.z, 4) : null, P.ebv != null ? "E(B−V) " + fmtNum(P.ebv, 3) : null].filter(Boolean).join(" · ") || `${P.n} prepared`;
  }
  if (step === "model") {
    if (mode() === "classify") return "not needed";
    if (mode() === "sample") return "per class";
    return S.model ? `${S.model.n_free} free params` : "components";
  }
  if (step === "fit") {
    if (S.busy && page === "fit") return "running…";
    if (!F) return "not run";
    if (F.kind === "classify") return `${F.n} classified`;
    if (F.kind === "sample") return `${F.n} fitted`;
    return F.rchi2 != null ? `χ²ᵣ ${F.rchi2.toFixed(3)}` : "done";
  }
  if (!F) return "—";
  if (F.kind === "classify" || F.kind === "sample") return `${F.n} spectra`;
  return F.epochs && F.epochs.length > 1 ? `${F.epochs.length} epochs` : "ready";
}

// ------------------------------------------------------------------ render
function render() {
  if (!S) return;
  $("pyver").textContent = "python " + S.python;
  const chip = $("pychip");
  chip.classList.toggle("busy", !!S.busy); chip.classList.toggle("err", !!S.error && !S.busy);
  const L = loaded();
  let obj = "";
  if (L && L.rows.length) {
    const names = L.rows.map((r) => r.name);
    obj = names.length === 1 ? names[0] : commonName(names);
  }
  $("objlabel").textContent = obj ? `${obj} · ${MODE_LABEL[mode()]}` : MODE_LABEL[mode()];
  renderStepper();
  renderCells();
  if (page === "read") renderRead();
  if (page === "prepare") renderPrepare();
  if (page === "model") renderModel();
  if (page === "fit") renderFit();
  if (page === "results") { renderResid(); if (S.wf.residuals && S.wf.residuals.id !== lastResidId) { lastResidId = S.wf.residuals.id; if (resultData && !resultData.error && rv === "fit") drawResultPlot(); } renderResults(); if (fitDone() && fitDone().kind === "joint" && lcData && !lcData.computing) renderReverb(); }
}
// one name for the spectra of one object: the common start and end, cut at separators ('spec-7339-…-0725')
function commonName(names) {
  if (names.length < 2) return names[0] || "";
  let p = names[0], q = names[0];
  for (const n of names) { while (!n.startsWith(p)) p = p.slice(0, -1); while (!n.endsWith(q)) q = q.slice(1); }
  const cut = p.search(/[_\-.\s][^_\-.\s]*$/);
  p = cut >= 0 && /[^_\-.\s]$/.test(p) && !names.every((n) => /[_\-.\s]/.test(n[p.length] || "")) ? p.slice(0, cut + 1) : p;
  const m = q.match(/^[^_\-.\s]*(?=[_\-.\s])/);
  q = m && !names.every((n) => /[_\-.\s]/.test(n[n.length - q.length - 1] || "")) ? q.slice(m[0].length) : q;
  q = q.replace(/\.(fits?|txt|dat|csv)$/i, "");
  if (p.length + q.length < 4) return commonPrefix(names) || names[0];
  if (!q) return p.replace(/[_\-.\s]+$/, "");
  if (!p) return q.replace(/^[_\-.\s]+/, "");
  return p + "…" + q;
}
function commonPrefix(names) {
  let p = names[0] || "";
  for (const n of names) while (!n.startsWith(p)) p = p.slice(0, -1);
  p = p.replace(/[_\-\s.]+$/, "").replace(/^t(?=AT\d)/, "");
  return p.length >= 3 ? p : "";
}
function renderStepper() {
  const nav = $("stepper");
  nav.innerHTML = STEPS.map(([id, label], i) => {
    const done = stepDone(id) && id !== page, cur = id === page;
    return (i ? '<span class="stepsep">›</span>' : "") +
      `<button class="step ${done ? "done" : ""} ${cur ? "cur" : ""}" data-go="${id}">` +
      `<span class="num">${done ? "✓" : i + 1}</span><span><b>${label}</b><small>${esc(stepSub(id))}</small></span></button>`;
  }).join("");
}
function go(step) {
  if (!STEPS.some(([id]) => id === step)) step = "read";
  page = step;
  document.querySelectorAll(".page").forEach((p) => p.classList.toggle("hidden", p.dataset.step !== step));
  if (location.hash !== "#" + step) history.replaceState(null, "", "#" + step);
  $("main").scrollTop = 0;
  render();
  loadPageData(false);
}
function loadPageData(force) {
  if (page === "prepare") loadPreview(force);
  if (page === "model") enterModel(force);
  if (page === "results") loadResult(force);
}

// =================================================================== 1 READ
let srcTab = "files";
function renderRead() {
  const m = mode();
  document.querySelectorAll(".mode").forEach((el) => {
    const on = el.dataset.mode === m;
    el.classList.toggle("on", on); el.querySelector("input").checked = on;
  });
  $("read-hint").textContent = { single: "fa.read(path)", sample: "files → fa.fit_sample(files)", multi: "[fa.read(f) for f in files]", classify: "fa.classify(fa.read(path))" }[m];
  const multi = m === "multi";
  $("groupbox").classList.toggle("hidden", srcTab === "desi" || srcTab === "sdss");
  $("groupwrap").classList.toggle("hidden", !multi);
  document.querySelector(".srcbody").classList.toggle("nogroup", $("groupbox").classList.contains("hidden"));
  $("fileinput").multiple = m !== "single";
  $("droptitle").textContent = $("fileinput").multiple ? "Drop spectra or choose files" : "Drop a spectrum or choose a file";
  if (S.wf.group && document.activeElement !== $("groupsel")) $("groupsel").value = S.wf.group;
  const L = loaded();
  $("loadedbox").classList.toggle("hidden", !L);
  $("toprepare").disabled = !L || !!S.busy;
  if (!L) { $("toprepare").textContent = "Prepare →"; return; }
  const n = L.n, unit = multi ? (n === 1 ? "epoch" : "epochs") : (n === 1 ? "spectrum" : "spectra");
  $("loadedtitle").textContent = `Loaded · ${n} ${unit}` + (L.shown < n ? ` (${L.shown} shown)` : "");
  $("toprepare").textContent = `Prepare ${n === 1 ? "spectrum" : n + " spectra"} →`;
  const sig = JSON.stringify([L.id, L.rows.length, multi]);
  if ($("loadedtbl").dataset.sig !== sig) {
    $("loadedtbl").dataset.sig = sig;
    $("loadedtbl").innerHTML = `<thead><tr><th><input type="checkbox" id="selall"></th><th>${multi ? "Epoch" : "#"}</th><th>Date</th><th>File</th><th>Format</th><th class="num">z</th><th>RA / Dec</th></tr></thead><tbody>` +
      L.rows.map((r) => `<tr><td><input type="checkbox" data-k="${r.k}"></td><td class="mono">#${r.k}</td><td class="mono">${esc(r.date || "—")}</td>` +
        `<td class="mono">${esc(r.file)}</td><td class="mono muted">${esc(r.format)}</td><td class="num">${r.z != null ? fmtNum(r.z, 5) : "—"}</td>` +
        `<td>${r.ra != null ? `<span class="mono">${fmtNum(r.ra, 7)}, ${fmtNum(r.dec, 7)}</span>` : '<span class="badge missing">missing</span>'}</td></tr>`).join("") + "</tbody>";
    $("selall").onchange = (e) => { document.querySelectorAll('#loadedtbl input[data-k]').forEach((c) => (c.checked = e.target.checked)); updateRemove(); };
    document.querySelectorAll('#loadedtbl input[data-k]').forEach((c) => (c.onchange = updateRemove));
    updateRemove();
  }
  const noCoords = L.rows.some((r) => r.ra == null), noZ = L.rows.some((r) => r.z == null);
  const notes = [];
  if (noCoords) notes.push("Some files carry no coordinates: give RA/Dec or E(B−V) in the next step.");
  if (noZ) notes.push("No redshift in the files: enter z in the next step.");
  $("loadednote").textContent = notes.join(" ");
}
function updateRemove() {
  const any = document.querySelectorAll('#loadedtbl input[data-k]:checked').length > 0;
  $("removesel").disabled = !any || !S.wf.loaded.rows.every((r) => r.path);
}
async function readSpectra(src) {
  const body = Object.assign({ mode: mode(), group: $("groupsel").value, format: $("fmtsel").value }, src);
  if (await post("/api/wf/read", body)) toast("Reading…", 1200);
}
async function uploadFiles(files) {
  if (!files.length) return;
  const batch = "batch_" + new Date().toISOString().replace(/[-:T.Z]/g, "").slice(0, 14);
  const paths = [];
  for (let i = 0; i < files.length; i++) {
    toast(`Uploading ${i + 1} / ${files.length}…`, 10000);
    try {
      const res = await api(`/api/upload?filename=${encodeURIComponent(files[i].name)}&batch=${batch}`, { method: "POST", body: files[i] });
      paths.push(res.path);
    } catch (e) { showError(e.message); return; }
  }
  $("toast").classList.add("hidden");
  readSpectra({ paths });
}
function initRead() {
  document.querySelectorAll(".mode").forEach((el) => (el.onclick = (e) => {
    e.preventDefault();
    const m = el.dataset.mode;
    if (m === mode()) return;
    S.wf.mode = m; render();
    post("/api/wf/mode", { mode: m });
  }));
  document.querySelectorAll("#srctabs button").forEach((b) => (b.onclick = () => {
    srcTab = b.dataset.src;
    document.querySelectorAll("#srctabs button").forEach((x) => x.classList.toggle("on", x === b));
    document.querySelectorAll(".srcpane").forEach((p) => p.classList.toggle("hidden", p.dataset.src !== srcTab));
    renderRead();
  }));
  $("fileinput").onchange = (e) => uploadFiles([...e.target.files]);
  const drop = $("drop");
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); uploadFiles([...e.dataTransfer.files]); };
  $("pathbtn").onclick = () => $("pathin").value.trim() && readSpectra({ pattern: $("pathin").value.trim() });
  $("pathin").onkeydown = (e) => e.key === "Enter" && $("pathbtn").click();
  $("desibtn").onclick = () => $("desiin").value.trim() && readSpectra({ targetid: $("desiin").value.trim() });
  $("desiin").onkeydown = (e) => e.key === "Enter" && $("desibtn").click();
  $("sdssbtn").onclick = () => $("sdssin").value.trim() && readSpectra({ sdss: $("sdssin").value.trim() });
  $("sdssin").onkeydown = (e) => e.key === "Enter" && $("sdssbtn").click();
  $("groupsel").onchange = () => {
    const L = loaded();
    if (L && mode() === "multi" && L.rows.every((r) => r.path)) readSpectra({ paths: L.rows.map((r) => r.path) });
  };
  $("removesel").onclick = () => {
    const drop = new Set([...document.querySelectorAll('#loadedtbl input[data-k]:checked')].map((c) => +c.dataset.k));
    const keep = S.wf.loaded.rows.filter((r) => !drop.has(r.k)).map((r) => r.path);
    if (!keep.length) { post("/api/reset", {}); return; }
    readSpectra({ paths: keep });
  };
  $("toprepare").onclick = () => go("prepare");
}

// ================================================================ 2 PREPARE
let prep = null, prepFor = null;   // form values, for the spectra read (loaded id)
let pk = 0, pkind = "prepared", masking = false, prepTimer = null;
function prepDefaults() {
  const L = loaded(), saved = S.wf.settings.prepare;
  if (saved) return Object.assign({}, saved, { masks: (saved.masks || []).slice() });
  const zs = L ? L.rows.map((r) => r.z) : [];
  const hasZ = zs.length && zs.every((z) => z != null);
  const hasCoords = L && L.rows.every((r) => r.ra != null);
  return { z: null, z_source: hasZ ? "file" : "manual", rest_frame: true, extinction: hasCoords ? "coords" : "value",
    ebv: null, ra: "", dec: "", wmin: null, wmax: null, telluric: true, sky_lines: false, absorption: false, masks: [], flux_unit: "file" };
}
function readPrepForm() {
  const num = (id) => ($(id).value.trim() === "" ? null : Number($(id).value.trim().replace(",", ".")));
  prep.z = num("zin"); prep.z_source = $("zsrc").value; prep.rest_frame = $("restin").checked;
  prep.ebv = num("ebvin"); prep.ra = $("rain").value.trim(); prep.dec = $("decin").value.trim();
  prep.wmin = num("wminin"); prep.wmax = num("wmaxin");
  prep.telluric = $("tellin").checked; prep.sky_lines = $("skyin").checked; prep.absorption = $("absin").checked;
  prep.flux_unit = $("fusel").value === "custom" ? ($("fuin").value.trim() || "file") : $("fusel").value;
}
function writePrepForm() {
  $("zin").value = prep.z ?? ""; $("zsrc").value = prep.z_source; $("restin").checked = prep.rest_frame;
  $("zin").disabled = prep.z_source === "file";
  $("ebvin").value = prep.ebv ?? ""; $("rain").value = prep.ra || ""; $("decin").value = prep.dec || "";
  $("wminin").value = prep.wmin ?? ""; $("wmaxin").value = prep.wmax ?? "";
  $("tellin").checked = prep.telluric; $("skyin").checked = prep.sky_lines; $("absin").checked = prep.absorption;
  const fu = prep.flux_unit || "file", listed = [...$("fusel").options].some((o) => o.value === fu);
  $("fusel").value = listed ? fu : "custom"; $("fuin").value = listed ? "" : fu;
  $("fuin").classList.toggle("hidden", $("fusel").value !== "custom");
  renderFluxUnit();
  setExt(prep.extinction, false);
  renderMasks();
}
// the flux unit: what the file says, and a choice when it says nothing (luminosities need it)
const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
function unitText(u) {
  if (u == null) return "unknown";
  const e = Math.floor(Math.log10(u) + 1e-9), m = u / 10 ** e;
  return `${Math.abs(m - 1) < 1e-6 ? "" : +m.toPrecision(3) + " × "}10${String(e).replace(/./g, (c) => SUP[c])} erg s⁻¹ cm⁻² Å⁻¹`;
}
function renderFluxUnit() {
  const L = loaded();
  if (!L) return;
  const units = [...new Set(L.rows.map((r) => r.flux_unit ?? null))], r0 = L.rows[0];
  const unknown = units.includes(null), chosen = $("fusel").value !== "file";
  let txt;
  if (chosen) txt = "Your choice replaces the unit of the file for all spectra.";
  else if (unknown) txt = `<b class="warnword">Not in the file${units.length > 1 ? " for some spectra" : ""}.</b> Choose it: without it there are no luminosities and black-hole masses. Many text files give it in a header (e.g. “flux in 1e-16 erg/s/cm2/Å”).`;
  else if (units.length > 1) txt = "The files have different units: " + units.map(unitText).join(", ") + ".";
  else txt = `From the file: <b>${esc(unitText(units[0]))}</b>${r0.flux_unit_source ? ` <span class="muted">(${esc(r0.flux_unit_source === "values in erg/s/cm2/A" ? "values in erg s⁻¹ cm⁻² Å⁻¹, stored in units of 10⁻¹⁷" : r0.flux_unit_source)})</span>` : ""}.`;
  $("funote").innerHTML = txt;
  $("fubox").classList.toggle("needs", unknown && !chosen);
}
function setExt(kind, run = true) {
  prep.extinction = kind;
  document.querySelectorAll("#extseg button").forEach((b) => b.classList.toggle("on", b.dataset.ext === kind));
  $("ext-coords").classList.toggle("hidden", kind !== "coords");
  $("ext-value").classList.toggle("hidden", kind !== "value");
  $("ext-none").classList.toggle("hidden", kind !== "none");
  const L = loaded();
  const has = L && L.rows.every((r) => r.ra != null);
  $("coordnote").textContent = has ? "Position from the files (overwrite to change). E(B−V) from the SFD map (Schlafly & Finkbeiner 2011)."
    : "The files have no RA/Dec: enter them (degrees or hh:mm:ss ±dd:mm:ss), or enter E(B−V).";
  if (run) schedulePrepare();
}
function renderMasks() {
  $("maskchips").innerHTML = prep.masks.map((m, i) => `<span class="chip">${Math.round(m[0])}–${Math.round(m[1])} Å<button data-i="${i}" title="Remove">×</button></span>`).join("");
  $("maskchips").querySelectorAll("button").forEach((b) => (b.onclick = () => { prep.masks.splice(+b.dataset.i, 1); renderMasks(); schedulePrepare(); }));
}
function prepValid() {
  if (prep.z_source === "manual" && prep.rest_frame && (prep.z === null || !isFinite(prep.z))) return "Enter the redshift z.";
  if (prep.extinction === "value" && (prep.ebv === null || !isFinite(prep.ebv))) return "Enter E(B−V), or choose From coordinates.";
  if (prep.extinction === "coords") {
    const L = loaded();
    const has = L && L.rows.every((r) => r.ra != null);
    if (!has && (!prep.ra || !prep.dec)) return "Enter RA and Dec, or choose Enter value.";
  }
  return null;
}
function schedulePrepare() {
  clearTimeout(prepTimer);
  prepTimer = setTimeout(runPrepare, 350);
}
function runPrepare() {
  readPrepForm();
  const bad = prepValid();
  $("prepstatus").textContent = bad || "";
  if (bad || S.busy) { if (!bad && S.busy) prepTimer = setTimeout(runPrepare, 500); return false; }
  const body = Object.assign({}, prep, { apply_to: $("applyto").value || "all" });
  post("/api/wf/prepare", body);
  return true;
}
function renderPrepare() {
  const L = loaded();
  if (!L) { $("prepstatus").textContent = "Read spectra first."; return; }
  if (!prep || prepFor !== L.id) { prep = prepDefaults(); prepFor = L.id; writePrepForm(); loadPreview.tried = false; pk = 0; }
  const n = L.shown;
  const opts = `<option value="all">all ${n} ${mode() === "multi" ? "epochs" : "spectra"}</option>` +
    L.rows.map((r) => `<option value="${r.k}">#${r.k}${r.date ? " · " + r.date : ""}</option>`).join("");
  if ($("applyto").dataset.sig !== String(n)) { $("applyto").innerHTML = opts; $("applyto").dataset.sig = String(n); }
  $("applybox").classList.toggle("hidden", n < 2);
  $("pprev").disabled = pk <= 0; $("pnext").disabled = pk >= n - 1;
  $("pprev").parentElement.querySelectorAll("button").forEach((b) => b.classList.toggle("hidden", n < 2));
  const P = prepared();
  if (P && !$("prepstatus").textContent.startsWith("Enter")) {
    $("prepstatus").textContent = `${P.n} prepared · z ${P.z != null ? fmtNum(P.z, 5) : "—"} · E(B−V) ${P.ebv != null ? fmtNum(P.ebv, 3) : "—"}` +
      (P.range ? ` · ${Math.round(P.range[0])}–${Math.round(P.range[1])} Å` : "");
  }
  document.querySelectorAll("#pkind button").forEach((b) => b.classList.toggle("on", b.dataset.kind === (P ? pkind : "raw")));
  $("tomodel").disabled = !!S.busy;
}
let previewSig = null;
async function loadPreview(force) {
  const P = prepared(), L = loaded();
  if (!L) return;
  if (!prep || prepFor !== L.id) renderPrepare();
  if (!P && prep && !prepValid() && !S.busy && !S.error && !loadPreview.tried) { loadPreview.tried = true; runPrepare(); }
  const kind = P ? pkind : "raw";
  const sig = JSON.stringify([P && P.id, L.id, pk, kind]);
  if (!force && sig === previewSig) return;
  previewSig = sig;
  let d;
  try { d = await api(`/api/wf/preview?k=${pk}&kind=${kind}`); } catch (e) { return; }
  if (!d || !d.x) return;
  const z = P && P.z ? P.z : 0;
  $("ptitle").textContent = `${mode() === "multi" ? "Epoch " : ""}#${d.k}${d.date ? " · " + d.date : ""} · ${d.name}`;
  $("paxis").textContent = d.rest_frame ? "rest-frame λ (Å)" + (P && P.ebv != null ? " · dereddened" : "") : "observed λ (Å)";
  const bands = (prep && prep.telluric ? d.telluric : []).map((b) => ({ lo: b.lo, hi: b.hi, label: b.label }));
  plotSpectrum($("pplot"), {
    x: d.x, layers: [{ y: d.flux, color: "#2a2f38", width: 1.1, range: true }],
    hatch: d.masked, bands, xlabel: "",
    onSelect: masking ? (lo, hi) => { prep.masks.push([lo, hi]); renderMasks(); setMasking(false); schedulePrepare(); } : null,
  });
  void z;
}
function setMasking(on) {
  masking = on;
  $("maskbtn").textContent = on ? "Drag on the plot to mask a range (Esc: cancel)" : "+ Mask a range (drag on plot)";
  $("pplot").classList.toggle("masking", on);
  if (on && pkind !== "prepared") { pkind = "prepared"; }
  previewSig = null; loadPreview(true);
}
function initPrepare() {
  ["zin", "ebvin", "rain", "decin", "wminin", "wmaxin"].forEach((id) => ($(id).onchange = schedulePrepare));
  ["restin", "tellin", "skyin", "absin"].forEach((id) => ($(id).onchange = schedulePrepare));
  $("fusel").onchange = () => { $("fuin").classList.toggle("hidden", $("fusel").value !== "custom"); renderFluxUnit(); if ($("fusel").value !== "custom") schedulePrepare(); else $("fuin").focus(); };
  $("fuin").onchange = () => { renderFluxUnit(); schedulePrepare(); };
  $("zsrc").onchange = () => { $("zin").disabled = $("zsrc").value === "file"; schedulePrepare(); };
  document.querySelectorAll("#extseg button").forEach((b) => (b.onclick = () => setExt(b.dataset.ext)));
  document.querySelectorAll("#pkind button").forEach((b) => (b.onclick = () => { pkind = b.dataset.kind; renderPrepare(); loadPreview(true); }));
  $("pprev").onclick = () => { pk = Math.max(0, pk - 1); renderPrepare(); loadPreview(true); };
  $("pnext").onclick = () => { pk = Math.min(loaded().shown - 1, pk + 1); renderPrepare(); loadPreview(true); };
  $("maskbtn").onclick = () => setMasking(!masking);
  $("applyto").onchange = () => {};
  $("tomodel").onclick = () => {
    readPrepForm();
    const bad = prepValid();
    if (bad) { $("prepstatus").textContent = bad; return; }
    const saved = JSON.stringify(S.wf.settings.prepare || null), now = JSON.stringify(Object.assign({}, prep));
    if (prepared() && sameSettings(saved, now)) { go("model"); return; }
    pendingGo = "model"; runPrepare();
  };
}
function sameSettings(a, b) {
  try {
    const x = JSON.parse(a), y = JSON.parse(b);
    if (!x) return false;
    return ["z", "z_source", "rest_frame", "extinction", "ebv", "wmin", "wmax", "telluric", "sky_lines", "absorption", "flux_unit"]
      .every((k) => (x[k] ?? null) === (y[k] ?? null)) && JSON.stringify(x.masks || []) === JSON.stringify(y.masks || []);
  } catch (e) { return false; }
}

// ================================================================== 3 MODEL
let selComp = null, pendingSuggest = false, guessSig = null, modelTried = false, startMode = null, classInfo = null;
let LINE_LIB = [];
let LINE_SETS = [];
function presetLabel() {
  for (let i = S.cells.length - 1; i >= 0; i--) {
    const m = /^model = fa\.presets\.(\w+)\((.*)\)$/m.exec(S.cells[i].code);
    if (m) return { name: m[1], args: m[2], code: `fa.presets.${m[1]}(${m[2]})` };
    if (/^model = fa\.(Model|load_model)\(/m.test(S.cells[i].code)) return null;     // built here, or a saved model
  }
  return null;
}
// where the current model comes from: a preset or one built here
function modelOrigin() {
  for (let i = S.cells.length - 1; i >= 0; i--) {
    if (/^model = fa\.presets\./m.test(S.cells[i].code)) return "preset";
    if (/^model = fa\.Model\(/m.test(S.cells[i].code)) return "custom";
    if (/^model = fa\.load_model\(/m.test(S.cells[i].code)) return "saved";
  }
  return S.model ? "preset" : null;
}
function modelForm() {
  const opt = {};
  document.querySelectorAll("#optbar input[data-opt]").forEach((c) => { if (!c.closest(".check").classList.contains("na")) opt[c.dataset.opt] = c.checked; });
  return Object.assign({ action: "preset", model: $("presetsel").value, broad_components: +$("bcount").textContent, host: $("hostsel").value }, opt);
}
function syncOptionAvailability() {
  const d = (S.model_defaults || {})[$("presetsel").value] || {};
  const has = (k) => (d.options || []).includes(k);
  $("broadbox").classList.toggle("hidden", !has("broad_components"));
  $("hostbox").classList.toggle("hidden", !has("host"));
  for (const k of ["outflow", "coronal", "oi6046", "red"]) $("o-" + k).classList.toggle("hidden", !has(k));
  $("o-stratified").classList.toggle("hidden", !d.stratified);
}
function setPresetDefaults(label) {
  const d = (S.model_defaults || {})[label] || {};
  $("bcount").textContent = d.broad_components ?? 2;
  document.querySelector('#optbar input[data-opt="outflow"]').checked = !!d.outflow;
  document.querySelector('#optbar input[data-opt="stratified"]').checked = false;
  for (const k of ["coronal", "oi6046", "red"]) document.querySelector(`#optbar input[data-opt="${k}"]`).checked = !!(d.values || {})[k];
}
// set the controls from the preset cell (made here, by the recommendation, or edited in the session)
let syncedPreset = null;
function syncFromPreset() {
  const pl = presetLabel();
  if (!pl || pl.code === syncedPreset) return;
  syncedPreset = pl.code;
  const args = {};
  for (const m of pl.args.matchAll(/(\w+)=('[^']*'|"[^"]*"|[^,]+)/g)) args[m[1]] = m[2].trim().replace(/^['"]|['"]$/g, "");
  const defs = S.model_defaults || {};
  let best = null, bestScore = -1;
  for (const [label, d] of Object.entries(defs)) {
    if (d.preset !== pl.name) continue;
    const kw = d.kw || {};
    const ok = Object.entries(kw).every(([k, v]) => String(args[k]) === (v === true ? "True" : v === false ? "False" : String(v)));
    const score = ok ? Object.keys(kw).length : -1;
    if (score > bestScore) { best = label; bestScore = score; }
  }
  if (!best) return;
  $("presetsel").value = best;
  setPresetDefaults(best);
  if (args.broad_components) $("bcount").textContent = args.broad_components;
  const box = (k) => document.querySelector(`#optbar input[data-opt="${k}"]`);
  for (const k of ["outflow", "coronal", "oi6046", "red"]) if (k in args) box(k).checked = args[k] === "True";
  box("stratified").checked = args.broad_profile === "stratified";
  const host = args.host;
  $("hostsel").value = host === undefined || host === "True" ? "default" : host === "False" ? "none" : host;
  if (!$("hostsel").value) $("hostsel").value = "default";
  syncOptionAvailability();
}
function makePreset() {
  syncOptionAvailability();
  startMode = "preset";
  post("/api/model", modelForm());
}
function enterModel(force) {
  const m = mode();
  const special = m === "classify" || m === "sample";
  $("model-note").classList.toggle("hidden", !special);
  ["optbar", "modelgrid", "classcard"].forEach((id) => $(id).classList.toggle("hidden", special));
  if (m === "classify") $("model-note").textContent = "Classification fits its own model (stellar populations + power law, then the emission lines); there is nothing to set up here.";
  if (m === "sample") $("model-note").textContent = "In a sample each spectrum is classified and gets the model its class needs; you can also choose one model for all of them in the Fit step.";
  $("tofit").textContent = m === "classify" ? "Set up classification →" : "Set up fit →";
  if (special) return;
  if (!$("presetsel").options.length) {
    $("presetsel").innerHTML = S.models.map((l) => `<option>${esc(l)}</option>`).join("");
    $("hostsel").innerHTML = '<option value="default">preset default</option><option value="none">none</option><option value="pca">PCA eigenspectra</option>' +
      (S.templates || []).map((t) => `<option value="${esc(t.name)}">${esc(t.name)}${t.available ? "" : " (download)"}</option>`).join("");
    const rec = S.recommended && S.recommended.model;
    $("presetsel").value = rec && S.models.includes(rec) ? rec : (S.models.includes("optical") ? "optical" : S.models[0]);
    setPresetDefaults($("presetsel").value);
    syncOptionAvailability();
  }
  syncFromPreset();
  if (!S.model && prepared() && !S.busy && !modelTried) { modelTried = true; makePreset(); return; }
  renderModel();
  loadGuess(force);
}
const COMP_NAME = { PowerLaw: "power law", BrokenPowerLaw: "broken power law", Blackbody: "blackbody", StellarPopulation: "stellar populations",
  HostGalaxy: "galaxy eigenspectra", FeII: "Fe II template", BalmerContinuum: "Balmer continuum", HighOrderBalmer: "high-order Balmer lines",
  Calibration: "calibration polynomial", Extinction: "dust extinction", Polynomial: "polynomial", Absorption: "absorption troughs" };
function compDesc(c) {
  if (c.lines && c.lines.length) return c.lines.map(prettyLine).join(", ");
  return COMP_NAME[c.type] || c.type;
}
function compKindLabel(c) {
  if (c.type === "Lines") return `${c.absorption ? "absorption" : "emission"} lines${c.role && !c.absorption ? " · " + c.role : ""}`;
  return COMP_NAME[c.type] || c.type;
}
function sharedPatterns() {
  const txt = ($("sharedin").value || "").trim();
  if (txt) return txt.split(",").map((s) => s.trim()).filter(Boolean);
  const pats = ["narrow.*"];
  if (S.model) S.model.components.filter((c) => c.kind === "host").forEach((c) => pats.push(c.name + ".*"));
  // with a shared broad-line shape the broad components (and Fe II) keep their shape in all epochs
  if (S.model && $("sharedshape").checked)
    S.model.components.filter((c) => c.role === "broad" || c.type === "FeII").forEach((c) => pats.push(c.name + ".*"));
  return pats;
}
const BOWEN_RE = /^(NIII|OIII3|HeII)\d+$/;
function compColor(c, all) {
  if (c.role === "broad" && c.lines && c.lines.length && c.lines.every((l) => BOWEN_RE.test(l))) return C.bowen;
  if (c.role === "broad") { const i = all.filter((x) => x.role === "broad").indexOf(c); return C.broad[i % C.broad.length]; }
  if (c.role === "narrow") return C.narrow;
  if (c.role === "outflow") return C.outflow;
  if (c.kind === "host") return C.host;
  if (c.kind === "continuum") return C.continuum;
  if (c.type === "FeII") return C.feii;
  const o = all.filter((x) => !x.role && !["host", "continuum"].includes(x.kind)).indexOf(c);
  return C.other[o % C.other.length];
}
const PROFILE_NAME = { gaussian: "Gaussian", lorentzian: "Lorentzian", voigt: "Voigt", "gauss-hermite": "Gauss–Hermite", stratified: "Stratified BLR" };
// ---- classification: a card at the top of the step, not a link
const CLASS_NAME = { type1: "Type 1 AGN", type1_host: "Type 1 AGN with a strong host", seyfert2: "Seyfert 2", liner: "LINER", composite: "Composite (AGN + star formation)",
  starforming: "Star-forming galaxy", retired: "Retired galaxy", passive: "Passive galaxy", unknown: "Not classified" };
async function loadClassInfo() {
  try { const d = await api("/api/wf/classify?k=0"); classInfo = d && d.label ? d : null; } catch (e) { classInfo = null; }
  renderClassCard();
}
function renderClassCard() {
  const rec = S.recommended, done = S.wf.classified && rec;
  const busy = S.busy && pendingSuggest;
  $("classcard").classList.toggle("done", !!done);
  if (busy) {
    $("cctitle").innerHTML = '<span class="spin"></span> Classifying the spectrum…';
    $("ccsub").textContent = "Fitting the starlight, testing for broad Hα and measuring the narrow-line ratios.";
    $("ccbtns").innerHTML = "";
    return;
  }
  if (!done) {
    $("cctitle").textContent = "Not sure which model to use?";
    $("ccsub").textContent = "Classify the spectrum: FANTASY fits the starlight, tests for broad Hα and places the narrow lines on the BPT diagrams, then recommends a model.";
    $("ccbtns").innerHTML = '<button class="btn" id="ccrun">Classify spectrum</button>';
    $("ccrun").onclick = () => { pendingSuggest = true; post("/api/wf/suggest", {}); renderClassCard(); };
    return;
  }
  const inUse = modelOrigin() === "preset" && $("presetsel").value === rec.model && (!rec.broad_components || +$("bcount").textContent === rec.broad_components);
  const what = `${rec.model}${rec.broad_components ? `, ${rec.broad_components} broad component${rec.broad_components > 1 ? "s" : ""}` : ""}`;
  $("cctitle").innerHTML = `<span class="clsdot" style="background:${CLASS_COLOR[rec.label] || "#6b7280"}"></span>${esc(CLASS_NAME[rec.label] || rec.label)}` +
    (rec.uncertain ? ' <span class="badge warn">uncertain</span>' : "") + `<span class="ccrec">Recommended model: <b>${esc(what)}</b></span>`;
  $("ccsub").textContent = classInfo && classInfo.reason ? classInfo.reason : "";
  $("ccbtns").innerHTML = (inUse ? '<span class="badge ok inuse">✓ In use</span>' : '<button class="btn primary" id="ccuse">Use recommended model</button>') +
    '<button class="btn" id="ccview">View classification</button>';
  if ($("ccuse")) $("ccuse").onclick = applyRecommendation;
  $("ccview").onclick = showClassDialog;
  if (!classInfo || classInfo.name !== (prepared() && S.wf.loaded && S.wf.loaded.rows[0] && S.wf.loaded.rows[0].name) && !loadClassInfo.pending) {
    loadClassInfo.pending = true; loadClassInfo().finally(() => (loadClassInfo.pending = false));
  }
}
function setStartMode(m) {
  startMode = m;
  document.querySelectorAll("#startseg button").forEach((b) => b.classList.toggle("on", b.dataset.s === m));
  $("presetopts").classList.toggle("hidden", m !== "preset");
  $("customopts").classList.toggle("hidden", m !== "custom");
  $("savedopts").classList.toggle("hidden", m !== "saved");
  if (m === "saved") refreshSaved();
}
// the saved models: the library (~/.fantasy_agn/models) and models saved in other folders
let savedModels = [], modelFolders = [];
async function refreshSaved() {
  try { const d = await api("/api/models"); savedModels = d.models || []; modelFolders = d.folders || []; } catch (e) { savedModels = []; }
  const cur = $("savedsel").value;
  $("savedsel").innerHTML = savedModels.length
    ? savedModels.map((m) => `<option value="${esc(m.path)}" title="${esc(m.path)}">${esc(m.name)} · ${m.n_components} components, ${m.n_free} free` +
        `${m.library ? "" : " · " + esc(shortPath(m.folder))}</option>`).join("")
    : '<option value="">no saved models yet</option>';
  if (savedModels.some((m) => m.path === cur)) $("savedsel").value = cur;
  $("loadsaved").disabled = !savedModels.length;
  $("sv-folders").innerHTML = modelFolders.map((f) => `<option value="${esc(f.label)}">${f.library ? "library" : ""}</option>`).join("");
}
const shortPath = (p) => { const f = modelFolders.find((x) => x.path === p); return f ? f.label : p; };
const lastFolder = () => { try { return localStorage.getItem("fantasy.modelFolder") || ""; } catch (e) { return ""; } };
function savePreview() {
  const name = $("sv-name").value.trim().replace(/[^\w\- ]/g, "").replace(/ /g, "_");
  const folder = $("sv-folder").value.trim() || "~/.fantasy_agn/models";
  $("sv-path").textContent = name ? `${folder.replace(/\/$/, "")}/${name}.json` : "";
}
async function saveModel() {
  if (!S.model) return showError("Create a model first.");
  await refreshSaved();
  const pl = presetLabel();
  $("sv-name").value = (pl ? pl.name : "my_model") + "_custom";
  $("sv-folder").value = lastFolder();
  savePreview();
  $("savedlg").classList.remove("hidden");
  $("sv-name").select();
}
async function submitSave(e, overwrite) {
  if (e) e.preventDefault();
  const name = $("sv-name").value.trim(), folder = $("sv-folder").value.trim();
  if (!name) return;
  let res;
  try {
    res = await api("/api/model", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "save", name, folder, overwrite: !!overwrite }) });
  } catch (err) { return showError(err.message); }
  if (res.exists) {                                         // a model of that name is already there
    if (confirm(`${res.exists} exists. Replace it?`)) return submitSave(null, true);
    return;
  }
  setState(res);
  try { localStorage.setItem("fantasy.modelFolder", folder); } catch (err) { /* private mode */ }
  $("savedlg").classList.add("hidden");
  toast(`Model saved: ${$("sv-path").textContent}`);
  refreshSaved();
}
function renderModel() {
  if (mode() === "classify" || mode() === "sample") return;
  const M = S.model;
  const pl = presetLabel();
  const loaded = [...S.cells].reverse().map((c) => /^model = fa\.(load_model\([^)]*\)|Model\(|presets\.)/m.exec(c.code)).find(Boolean);
  const saved = loaded && loaded[1].startsWith("load_model") ? loaded[1].replace(/'[^']*\/([^'/]+)\.json'/, "'$1'") : null;
  $("model-hint").textContent = pl ? pl.code : saved ? `fa.${saved}` : M ? "fa.Model(…)" : "fa.Model()";
  if ($("presetsel").options.length) syncFromPreset();
  setStartMode(startMode || modelOrigin() || "preset");
  const own = modelOrigin() === "custom";
  $("startempty").textContent = own ? "Start again" : "Start a new model";
  $("startempty").className = own ? "btn" : "btn primary";
  $("customnote").innerHTML = own ? `Your own model, ${S.model ? S.model.components.length : 0} components. Add more with <b>Add component</b>, or start again from a power law.`
    : "Start with a power-law continuum and add the lines, templates and continuum components the spectrum needs with <b>Add component</b>. You can also start from a preset and change it: every model can be edited.";
  renderClassCard();
  $("tofit").disabled = !M || !!S.busy;
  if (!M) { $("comps").innerHTML = '<li class="muted empty">No model yet: choose a preset or start your own.</li>'; $("compcard").classList.add("hidden"); return; }
  $("compcard").classList.remove("hidden");
  $("nfree").textContent = `${M.n_free} free parameters`;
  if (!selComp || !M.components.some((c) => c.name === selComp)) {
    selComp = (M.components.find((c) => c.role === "broad") || M.components[0]).name;
  }
  const pats = mode() === "multi" ? sharedPatterns() : [];
  const sig = JSON.stringify([M.components, selComp, pats]);
  if ($("comps").dataset.sig !== sig) {
    $("comps").dataset.sig = sig;
    $("comps").innerHTML = M.components.map((c) => {
      const shared = pats.some((p) => p.split(".")[0] === c.name || (p.endsWith("*") && c.name.startsWith(p.split(".")[0]) && p.split(".")[0] !== "" && p.startsWith(c.name)));
      const badges = [c.profile && c.profile !== "gaussian" ? `<span class="badge muted">${esc(PROFILE_NAME[c.profile] || c.profile)}</span>` : "",
        c.absorption ? '<span class="badge k-absorption">absorption</span>' : "",
        c.tied_to ? `<span class="badge info" title="shift and width tied to ${esc(c.tied_to)}">= ${esc(c.tied_to)}</span>` : "",
        shared ? '<span class="badge info">shared</span>' : ""].join("");
      return `<li data-c="${esc(c.name)}" class="${c.name === selComp ? "sel" : ""}"><span class="sw" style="background:${compColor(c, M.components)}"></span>` +
        `<span class="cname">${esc(c.name)}</span><span class="cbadges">${badges}</span>` +
        `<button class="rm" data-rm="${esc(c.name)}" title="Remove ${esc(c.name)}">×</button><span class="cdesc">${esc(compDesc(c))}</span></li>`;
    }).join("");
    $("comps").querySelectorAll("li").forEach((li) => (li.onclick = (e) => {
      if (e.target.dataset.rm) { removeComp(e.target.dataset.rm); return; }
      selComp = li.dataset.c; renderModel(); drawGuess();
    }));
  }
  renderCompCard();
  renderParams();
}
function removeComp(name) {
  if (confirm(`Remove ${name} from the model?`)) post("/api/model", { action: "remove", name });
}
// the selected component: its profile, its kinematics and its parameters
function renderCompCard() {
  const M = S.model, c = M.components.find((x) => x.name === selComp);
  $("csw").style.background = compColor(c, M.components);
  $("ptabtitle").textContent = c.name;
  $("cdesc").textContent = compKindLabel(c);
  const isLines = c.type === "Lines";
  $("profbox").classList.toggle("hidden", !isLines || !c.rebuildable);
  if (isLines && c.rebuildable) {
    const profs = ["gaussian", "lorentzian", "voigt", "gauss-hermite"].concat(c.role === "broad" || c.profile === "stratified" ? ["stratified"] : []);
    $("profseg").innerHTML = profs.map((p) => `<button data-p="${p}" class="${p === c.profile ? "on" : ""}" title="${esc(PROFILE_HELP[p])}">${profileIcon(p, 26, 12)}${esc(PROFILE_NAME[p])}</button>`).join("");
    $("profseg").querySelectorAll("button").forEach((b) => (b.onclick = () => {
      if (b.dataset.p === c.profile) return;
      post("/api/model", { action: "profile", component: c.name, profile: b.dataset.p }).then((ok) => ok && toast(`${c.name}: ${PROFILE_NAME[b.dataset.p]} profile`));
    }));
  }
  const kin = (n) => ["shift", "fwhm"].filter((k) => M.params.some((p) => p.name === `${n}.${k}`));
  const mine = kin(c.name);
  const targets = M.components.filter((x) => x.name !== c.name && kin(x.name).length && x.tied_to !== c.name).map((x) => x.name);
  $("kinbox").classList.toggle("hidden", !mine.length);
  $("kinsel").innerHTML = `<option value="">free (fitted)</option>` + targets.map((t) => `<option value="${esc(t)}" ${t === c.tied_to ? "selected" : ""}>same as ${esc(t)}</option>`).join("");
  $("compctl").classList.toggle("hidden", $("profbox").classList.contains("hidden") && !mine.length);
}
function paramLabel(name) {
  const rest = name.slice(selComp.length + 1).split(".");
  if (rest.length === 2) return `${prettyLine(rest[0])} · ${rest[1]}`;
  return rest.join(".");
}
function renderParams() {
  const M = S.model;
  const ps = M.params.filter((p) => p.name.startsWith(selComp + ".") && !p.linear);
  const sig = JSON.stringify(ps);
  if ($("paramtbl").dataset.sig === sig) return;
  $("paramtbl").dataset.sig = sig;
  if (!ps.length) { $("paramtbl").innerHTML = '<tbody><tr><td class="muted">Only linear amplitudes: solved exactly in the fit.</td></tr></tbody>'; return; }
  const suffix = (n) => n.split(".").pop();
  $("paramtbl").innerHTML = "<thead><tr><th>Parameter</th><th>Initial</th><th>Min</th><th>Max</th><th>Fixed</th><th>Tie to</th></tr></thead><tbody>" +
    ps.map((p) => {
      const cands = M.params.filter((q) => q.name !== p.name && !q.linear && suffix(q.name) === suffix(p.name)).map((q) => q.name);
      if (p.tie && !cands.includes(p.tie)) cands.unshift(p.tie);
      const dis = p.tie ? "disabled" : "";
      return `<tr data-p="${esc(p.name)}" class="${p.tie ? "tied" : ""}"><td class="pname" title="${esc(p.description || p.name)}">${esc(paramLabel(p.name))}${p.unit ? ` <small>${esc(p.unit)}</small>` : ""}</td>` +
        `<td><input data-f="value" value="${fmtNum(p.value, 6)}" ${dis}></td><td><input data-f="min" value="${fmtNum(p.min, 6)}" ${dis}></td>` +
        `<td><input data-f="max" value="${fmtNum(p.max, 6)}" ${dis}></td><td><input type="checkbox" data-f="fixed" ${p.fixed ? "checked" : ""} ${dis}></td>` +
        `<td><select data-f="tie"><option value="">—</option>${cands.map((c) => `<option ${c === p.tie ? "selected" : ""} value="${esc(c)}">${esc(c)}</option>`).join("")}</select></td></tr>`;
    }).join("") + "</tbody>";
  $("paramtbl").querySelectorAll("tr[data-p]").forEach((tr) => tr.querySelectorAll("input,select").forEach((el) => (el.onchange = () => {
    const p = S.model.params.find((q) => q.name === tr.dataset.p);
    const val = (f) => tr.querySelector(`[data-f="${f}"]`);
    const change = { name: p.name, value: val("value").value, min: val("min").value, max: val("max").value,
      fixed: val("fixed").checked, tie: val("tie").value || null, factor: p.factor };
    tr.classList.add("changed");
    post("/api/model", { action: "params", changes: [change] });
  })));
}
async function loadGuess(force) {
  if (!S.model || !prepared()) return;
  const sig = JSON.stringify([S.model.params.map((p) => [p.value, p.min, p.max, p.fixed, p.tie]), S.model.components.map((c) => c.name), prepared().id]);
  if (!force && sig === guessSig && loadGuess.data) { drawGuess(); return; }
  guessSig = sig;
  $("gtitle").textContent = "Initial guess · computing…";
  try { loadGuess.data = await api("/api/wf/guess"); } catch (e) { loadGuess.data = null; }
  drawGuess();
}
function drawGuess() {
  const d = loadGuess.data;
  if (!d || !d.x) {
    $("gtitle").textContent = "Initial guess";
    $("gplot").innerHTML = `<div class="empty">${esc((d && d.error) || "Prepare the spectra and choose a model to see the starting point of the fit.")}</div>`;
    return;
  }
  const n = (S.wf.loaded || {}).shown || 1;
  $("gtitle").textContent = `Initial guess · ${n > 1 ? "epoch #" + d.k + (n > 1 && mode() === "multi" ? " (reference)" : "") : d.name}`;
  $("ghl").textContent = "highlighted: " + selComp;
  const comps = S.model.components;
  const layers = [{ y: d.flux, color: C.data, width: 1, range: true, name: "data" }];
  for (const c of comps) {
    if (!d.components[c.name]) continue;
    const on = c.name === selComp;
    layers.push({ y: d.components[c.name], color: on ? compColor(c, comps) : "#c4c8cf", width: on ? 2.2 : 1, dash: c.kind === "continuum" ? "4 3" : null, name: c.name, top: on });
  }
  layers.push({ y: d.model, color: C.model, width: 1.1, name: "model", range: true, opacity: 0.9 });
  layers.sort((a, b) => (a.top ? 1 : 0) - (b.top ? 1 : 0));
  plotSpectrum($("gplot"), { x: d.x, layers, zero: true });
}
function applyRecommendation() {
  const r = S.recommended;
  if (!r || !S.models.includes(r.model)) return;
  $("presetsel").value = r.model;
  setPresetDefaults(r.model);
  if (r.broad_components) $("bcount").textContent = r.broad_components;
  toast(`${CLASS_NAME[r.label] || r.label}${r.uncertain ? " (uncertain)" : ""}: using ${r.model}`, 4000);
  makePreset();
}
function showClassDialog() {
  const kind = showClassDialog.kind || "diagrams";
  document.querySelectorAll("#classdlgkind button").forEach((b) => b.classList.toggle("on", b.dataset.k === kind));
  const url = (fmt, dl) => `/api/wf/classify.${fmt}?k=0&kind=${kind}&v=${S.cells.length}${dl ? "&dl=1" : ""}`;
  $("classdlgname").textContent = classInfo ? ` · ${classInfo.name}` : "";
  $("classdlgimg").src = withToken(url("png"));
  $("classdlgpng").onclick = () => download(url("png", true));
  $("classdlgpdf").onclick = () => download(url("pdf", true));
  $("classdlg").classList.remove("hidden");
}

// ---------------------------------------------------- add a component
const PROFILE_HELP = { gaussian: "Gaussian: the default line shape", lorentzian: "Lorentzian: narrow core, extended wings (e.g. NLS1 broad lines)",
  voigt: "Voigt: Gaussian convolved with a Lorentzian (two widths)", "gauss-hermite": "Gauss–Hermite: Gaussian with skewness h3 and kurtosis h4 (asymmetric lines)",
  stratified: "Stratified BLR: sum of Gaussians from a radially stratified, virialised BLR (core plus power-law wings)" };
function profileIcon(p, w = 54, h = 24) {
  const n = 41, pts = [];
  for (let i = 0; i < n; i++) {
    const x = -3 + (6 * i) / (n - 1);
    let y;
    if (p === "lorentzian") y = 1 / (1 + (x / 0.55) ** 2);
    else if (p === "voigt") y = 0.6 * Math.exp(-x * x / 0.9) + 0.4 / (1 + (x / 0.5) ** 2);
    else if (p === "gauss-hermite") { const g = Math.exp(-x * x / 1.4); y = g * (1 + 0.32 * (x ** 3 - 1.5 * x) / 1.5); }
    else if (p === "stratified") y = 0.55 * Math.exp(-x * x / 0.25) + 0.45 / (1 + (x / 0.9) ** 2);
    else y = Math.exp(-x * x / 1.2);
    pts.push(`${((i / (n - 1)) * w).toFixed(1)},${(h - 1 - Math.max(0, y) * (h - 3)).toFixed(1)}`);
  }
  return `<svg class="pico" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-hidden="true"><polyline points="${pts.join(" ")}" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>`;
}
const ADD_TYPES = [
  ["Lines", [["emission", "Emission lines", "Broad, narrow or outflow lines from the line list"],
             ["absorption", "Absorption lines", "Lines with negative flux: Na I D, Ca II H & K, Mg II, …"]]],
  ["Continuum", [["PowerLaw", "Power law", "f ∝ λ^α, the accretion-disc continuum"], ["BrokenPowerLaw", "Broken power law", "Two slopes, blue and red of the pivot"],
                 ["Blackbody", "Blackbody", "A thermal continuum, e.g. of a tidal disruption event"], ["Polynomial", "Polynomial", "A smooth additive continuum term"],
                 ["BalmerContinuum", "Balmer continuum", "Bound-free emission below 3646 Å"]]],
  ["Templates", [["FeII", "Fe II template", "FANTASY Fe II multiplets, broadened and shifted"], ["HighOrderBalmer", "High-order Balmer lines", "H7 to H50 near the Balmer edge"],
                 ["StellarPopulation", "Host galaxy", "Stellar population templates with velocity dispersion"]]],
  ["Absorption and dust", [["Absorption", "Absorption troughs", "Troughs at any wavelength that absorb the light behind them (BALs)"],
                           ["Extinction", "Dust extinction", "Reddening of the AGN light, E(B−V) fitted"]]],
];
const ROLE_DEF = { broad: { fwhm: [3000, 1000, 15000], shift: [0, -3000, 3000] }, narrow: { fwhm: [400, 50, 1200], shift: [0, -500, 500] },
  outflow: { fwhm: [1200, 300, 3000], shift: [-300, -2000, 500] }, absorption: { fwhm: [300, 30, 5000], shift: [0, -3000, 1000] } };
let addType = "emission", addState = null;
function uniqueName(base) {
  const names = new Set(S.model ? S.model.components.map((c) => c.name) : []);
  if (!names.has(base)) return base;
  for (let i = 2; ; i++) if (!names.has(base + i)) return base + i;
}
function openAdd(type) {
  addType = type || addType;
  $("addtypes").innerHTML = ADD_TYPES.map(([g, items]) => `<li class="grp">${esc(g)}</li>` + items.map(([id, t, d]) =>
    `<li data-t="${id}" class="${id === addType ? "on" : ""}"><b>${esc(t)}</b><small>${esc(d)}</small></li>`).join("")).join("");
  $("addtypes").querySelectorAll("li[data-t]").forEach((li) => (li.onclick = () => openAdd(li.dataset.t)));
  renderAddForm();
  $("adddlg").classList.remove("hidden");
}
function closeAdd() { $("adddlg").classList.add("hidden"); }
function kinOptions(selfLabel) {
  const M = S.model;
  const opts = (M ? M.components : []).filter((c) => M.params.some((p) => p.name === `${c.name}.shift`) && M.params.some((p) => p.name === `${c.name}.fwhm`));
  return `<option value="">${esc(selfLabel)}</option>` + opts.map((c) => `<option value="${esc(c.name)}">same as ${esc(c.name)} (${esc(compKindLabel(c))})</option>`).join("");
}
function limitsRow(label, key, v, unit) {
  return `<div><label class="lbl">${label} <small>${unit}</small></label><div class="row three">` +
    `<input class="mono" data-lim="${key}" data-i="0" value="${v[0]}" title="start"><input class="mono" data-lim="${key}" data-i="1" value="${v[1]}" title="minimum">` +
    `<input class="mono" data-lim="${key}" data-i="2" value="${v[2]}" title="maximum"></div><small class="muted">start · min · max</small></div>`;
}
function renderAddForm() {
  const f = $("addform"), t = addType;
  const isLines = t === "emission" || t === "absorption";
  if (isLines) {
    // a fresh form each time: Gaussian, and the kind of line the model still lacks
    const roles = S.model ? S.model.components.filter((c) => c.type === "Lines" && !c.absorption).map((c) => c.role) : [];
    const role = t === "absorption" ? "absorption" : !roles.includes("broad") ? "broad" : !roles.includes("narrow") ? "narrow" : "broad";
    addState = { role, lines: [], type: t, profile: "gaussian" };
    const d = ROLE_DEF[role];
    f.innerHTML = `<h3>${t === "absorption" ? "Absorption lines" : "Emission lines"}</h3>
      <div><label class="lbl">Lines</label>
        <div class="linepick"><div class="chosen" id="lp-chosen"></div>
          <div class="lpsets" id="lp-sets"></div>
          <input id="lp-search" placeholder="Search the line list: Hb, OIII, 5007, He II, coronal, Bowen…">
          <div class="lplist" id="lp-list"></div></div></div>
      <div class="row two">
        ${t === "emission" ? `<div><label class="lbl">Kind of line</label><div class="seg small" id="lp-role">
          <button type="button" data-r="broad">Broad</button><button type="button" data-r="narrow">Narrow</button><button type="button" data-r="outflow">Outflow</button></div></div>` : "<div></div>"}
        <div><label class="lbl" for="f-name">Component name</label><input id="f-name" class="mono"></div></div>
      <div><label class="lbl">Profile</label><div class="profcards" id="lp-prof"></div></div>
      <div class="row two">${limitsRow("Width (FWHM)", "fwhm", d.fwhm, "km/s")}${limitsRow("Velocity shift", "shift", d.shift, "km/s")}</div>
      <div><label class="lbl" for="lp-kin">Shift and width</label><select id="lp-kin">${kinOptions("fitted, the same for all these lines")}<option value="__own">fitted separately for each line</option></select></div>
      <div class="formfoot"><span class="muted small" id="f-msg"></span><span class="grow"></span><button type="button" class="btn" id="f-cancel">Cancel</button><button class="btn primary" type="submit">Add lines</button></div>`;
    const setRole = (r) => {
      addState.role = r;
      document.querySelectorAll("#lp-role button").forEach((b) => b.classList.toggle("on", b.dataset.r === r));
      const dd = ROLE_DEF[r];
      f.querySelectorAll("[data-lim]").forEach((el) => (el.value = dd[el.dataset.lim][+el.dataset.i]));
      if (!$("f-name").dataset.edited) $("f-name").value = uniqueName(r === "absorption" ? "abs" : r);
      renderProfiles();
    };
    document.querySelectorAll("#lp-role button").forEach((b) => (b.onclick = () => setRole(b.dataset.r)));
    $("f-name").oninput = () => ($("f-name").dataset.edited = "1");
    $("lp-search").oninput = renderLinePick;
    setRole(role);
    renderLinePick();
  } else {
    const def = { PowerLaw: "continuum", BrokenPowerLaw: "continuum", Blackbody: "blackbody", Polynomial: "poly", BalmerContinuum: "balmer_cont",
      FeII: "feii", HighOrderBalmer: "balmer_lines", StellarPopulation: "host", Absorption: "absorption", Extinction: "extinction" }[t];
    const title = ADD_TYPES.flatMap(([, it]) => it).find((x) => x[0] === t);
    let extra = "";
    if (["PowerLaw", "BrokenPowerLaw", "Blackbody"].includes(t)) extra = `<div><label class="lbl" for="f-pivot">Pivot wavelength <small>Å</small></label><input id="f-pivot" class="mono" value="5100"></div>`;
    if (t === "Polynomial") extra = `<div><label class="lbl" for="f-order">Order</label><input id="f-order" class="mono" value="2"></div>`;
    if (t === "FeII") {
      // the model that covers the fitted range: optical, or one of the UV ones below 3500 Å
      const rng = (S.model && S.model.wave_range) || (prepared() && prepared().range) || null;
      const uv = rng && rng[1] < 3600;
      extra = `<div><label class="lbl" for="f-feiikind">Fe II model</label><select id="f-feiikind">` +
        `<option value="optical"${uv ? "" : " selected"}>optical, 3700–10 900 Å (Kovačević et al. 2010)</option>` +
        `<option value="uv"${uv ? " selected" : ""}>UV around Mg II, 2690–3005 Å (Kovačević-Dojčinović &amp; Popović 2015)</option>` +
        `<option value="uv_extended">UV extended, 2200–3500 Å (grouped by lower term, NIST data; new)</option></select></div>` +
        `<div class="row two">${limitsRow("Width (FWHM)", "fwhm", [2000, 1000, 6000], "km/s")}<div><label class="lbl" for="lp-kin">Shift and width</label><select id="lp-kin">${kinOptions("fitted")}</select></div></div>`;
    }
    if (t === "StellarPopulation") extra = `<div><label class="lbl" for="f-tpl">Templates</label><select id="f-tpl">${(S.templates || []).map((x) => `<option value="${esc(x.name)}">${esc(x.name)}${x.available ? "" : " (download)"}</option>`).join("")}</select></div>`;
    if (t === "Extinction") extra = `<div><label class="lbl" for="f-law">Extinction law</label><select id="f-law"><option value="F99">Fitzpatrick (1999)</option><option value="calzetti">Calzetti (2000)</option></select></div>` +
      (S.model && S.model.components.some((c) => c.type === "Extinction") ? '<p class="callout">The model already has dust extinction; only one is allowed.</p>' : "");
    if (t === "Absorption") extra = `<div><label class="lbl">Troughs</label><table class="tbl troughs" id="f-troughs"><thead><tr><th>Rest wavelength <small>Å</small></th><th>Width (FWHM) <small>km/s</small></th><th></th></tr></thead><tbody></tbody></table>
      <button type="button" class="btn small dashed" id="f-addtr">+ Trough</button></div>`;
    f.innerHTML = `<h3>${esc(title[1])}</h3><p class="note">${esc(title[2])}.</p>
      <div><label class="lbl" for="f-name">Component name</label><input id="f-name" class="mono" value="${esc(uniqueName(def))}"></div>${extra}
      <div class="formfoot"><span class="muted small" id="f-msg"></span><span class="grow"></span><button type="button" class="btn" id="f-cancel">Cancel</button><button class="btn primary" type="submit">Add</button></div>`;
    if (t === "Absorption") {
      const addRow = (w = "", v = "") => {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td><input class="mono" data-w value="${w}" placeholder="e.g. 5893"></td><td><input class="mono" data-v value="${v}" placeholder="e.g. 300"></td><td><button type="button" class="rm" title="Remove">×</button></td>`;
        tr.querySelector(".rm").onclick = () => tr.remove();
        $("f-troughs").tBodies[0].appendChild(tr);
      };
      $("f-addtr").onclick = () => addRow();
      addRow();
    }
  }
  $("f-cancel").onclick = closeAdd;
}
function renderProfiles() {
  const profs = ["gaussian", "lorentzian", "voigt", "gauss-hermite"].concat(addState.role === "broad" ? ["stratified"] : []);
  if (!profs.includes(addState.profile)) addState.profile = "gaussian";
  $("lp-prof").innerHTML = profs.map((p) => `<button type="button" data-p="${p}" class="${p === addState.profile ? "on" : ""}" title="${esc(PROFILE_HELP[p])}">${profileIcon(p)}<span>${esc(PROFILE_NAME[p])}</span></button>`).join("");
  $("lp-prof").querySelectorAll("button").forEach((b) => (b.onclick = () => { addState.profile = b.dataset.p; renderProfiles(); }));
}
const LINE_TYPE_ORDER = ["recombination", "permitted", "bowen", "semi-forbidden", "forbidden", "coronal", "molecular", "feature"];
const LINE_TYPE_NAME = { recombination: "Recombination (H, He)", permitted: "Permitted", bowen: "Bowen fluorescence", "semi-forbidden": "Semi-forbidden",
  forbidden: "Forbidden", coronal: "Coronal", molecular: "Molecular (H₂)", feature: "Dust and other features" };
function renderLinePick() {
  const rng = (S.model && S.model.wave_range) || (prepared() && prepared().range) || null;
  const q = ($("lp-search").value || "").trim().toLowerCase().replace(/[\[\]\s]/g, "");
  const chosen = addState.lines;
  $("lp-chosen").innerHTML = chosen.length ? chosen.map((n) => `<span class="lchip">${esc(prettyLine(n))}<button type="button" data-x="${esc(n)}" title="Remove">×</button></span>`).join("")
    : '<span class="muted small">Click lines below to choose them.</span>';
  $("lp-chosen").querySelectorAll("[data-x]").forEach((b) => (b.onclick = () => { addState.lines = chosen.filter((x) => x !== b.dataset.x); renderLinePick(); }));
  const inRange = (n) => { const l = LINE_LIB.find((x) => x.name === n); return l && (!rng || (l.wave >= rng[0] - 5 && l.wave <= rng[1] + 5)); };
  const sets = LINE_SETS.map((st) => ({ ...st, here: st.lines.filter(inRange) })).filter((st) => st.here.length);
  $("lp-sets").innerHTML = sets.length ? '<small class="muted">Line lists of FANTASY 1:</small> ' + sets.map((st) => {
    const all = st.here.every((n) => chosen.includes(n));
    return `<button type="button" data-set="${esc(st.key)}" class="${all ? "on" : ""}" title="${esc(st.here.map(prettyLine).join(", "))}">${esc(st.label)} <small>${st.here.length}</small></button>`;
  }).join("") : "";
  $("lp-sets").querySelectorAll("[data-set]").forEach((b) => (b.onclick = () => {
    const st = sets.find((x) => x.key === b.dataset.set);
    const all = st.here.every((n) => chosen.includes(n));
    addState.lines = all ? chosen.filter((n) => !st.here.includes(n)) : chosen.concat(st.here.filter((n) => !chosen.includes(n)));
    renderLinePick();
  }));
  let ls = LINE_LIB.filter((l) => !rng || (l.wave >= rng[0] - 5 && l.wave <= rng[1] + 5));
  if (q) ls = LINE_LIB.filter((l) => (l.name + prettyLine(l.name) + (l.ion || "") + (l.type || "") + (l.series || "") + Math.round(l.wave)).toLowerCase().replace(/[\[\]\s]/g, "").includes(q));
  const groups = {};
  ls.forEach((l) => (groups[l.type || "other"] = groups[l.type || "other"] || []).push(l));
  const rank = (g) => { const i = LINE_TYPE_ORDER.indexOf(g); return i < 0 ? 99 : i; };
  const order = Object.keys(groups).sort((a, b) => rank(a) - rank(b));
  $("lp-list").innerHTML = (ls.length ? "" : `<p class="muted small">No line matches${rng && !q ? " in the fitted range" : ""}.</p>`) +
    (rng && !q ? `<p class="muted small">Lines in the fitted range ${Math.round(rng[0])}–${Math.round(rng[1])} Å; search to see all.</p>` : "") +
    order.map((g) => `<div class="lpg"><small>${esc(LINE_TYPE_NAME[g] || g)}</small><div>` + groups[g].sort((a, b) => a.wave - b.wave).map((l) =>
      `<button type="button" data-l="${esc(l.name)}" class="${chosen.includes(l.name) ? "on" : ""}" title="${esc((l.ion || "") + " " + (l.series || ""))}">${esc(prettyLine(l.name))} <small>${l.wave.toFixed(0)}</small></button>`).join("") + "</div></div>").join("");
  $("lp-list").querySelectorAll("button[data-l]").forEach((b) => (b.onclick = () => {
    const n = b.dataset.l;
    addState.lines = chosen.includes(n) ? chosen.filter((x) => x !== n) : chosen.concat([n]);
    renderLinePick();
  }));
}
function addSubmit(e) {
  e.preventDefault();
  const t = addType, f = $("addform");
  const lim = (key) => [0, 1, 2].map((i) => +f.querySelector(`[data-lim="${key}"][data-i="${i}"]`).value);
  const name = $("f-name").value.trim();
  let comp;
  if (t === "emission" || t === "absorption") {
    if (!addState.lines.length) { $("f-msg").textContent = "Choose at least one line."; return; }
    const kin = $("lp-kin").value;
    comp = { type: "Lines", name, lines: addState.lines, role: t === "absorption" ? "narrow" : addState.role, absorption: t === "absorption",
      profile: addState.profile, fwhm: lim("fwhm"), shift: lim("shift"), own_kinematics: kin === "__own", tie_to: kin && kin !== "__own" ? kin : null };
  } else {
    comp = { type: t, name };
    if ($("f-pivot")) comp.pivot = $("f-pivot").value;
    if ($("f-order")) comp.order = $("f-order").value;
    if ($("f-tpl")) comp.templates = $("f-tpl").value;
    if ($("f-law")) comp.law = $("f-law").value;
    if (t === "FeII") { comp.fwhm = lim("fwhm"); comp.tie_to = $("lp-kin").value || null; comp.feii_kind = $("f-feiikind").value; }
    if (t === "Absorption") {
      const rows = [...$("f-troughs").tBodies[0].rows].map((tr) => [tr.querySelector("[data-w]").value.trim(), tr.querySelector("[data-v]").value.trim()]).filter((r) => r[0]);
      if (!rows.length || rows.some((r) => !isFinite(+r[0]) || !isFinite(+r[1]) || !r[1])) { $("f-msg").textContent = "Give the wavelength and width of each trough."; return; }
      comp.troughs = rows.map((r) => r.join(":")).join(", ");
    }
  }
  $("f-msg").textContent = "Adding…";
  post("/api/model", { action: "add", component: comp }).then((ok) => {
    if (!ok) { $("f-msg").textContent = ""; return; }
    selComp = name; closeAdd(); toast(`Added ${name}`);
  });
}
function initModel() {
  $("presetsel").onchange = () => { setPresetDefaults($("presetsel").value); makePreset(); };
  $("hostsel").onchange = makePreset;
  $("bminus").onclick = () => { $("bcount").textContent = Math.max(1, +$("bcount").textContent - 1); makePreset(); };
  $("bplus").onclick = () => { $("bcount").textContent = Math.min(3, +$("bcount").textContent + 1); makePreset(); };
  document.querySelectorAll("#optbar input[data-opt]").forEach((c) => (c.onchange = makePreset));
  document.querySelectorAll("#startseg button").forEach((b) => (b.onclick = () => {
    setStartMode(b.dataset.s);
    if (b.dataset.s === "preset" && modelOrigin() === "custom") makePreset();     // back to the chosen preset
  }));
  $("startempty").onclick = () => {
    if (S.model && modelOrigin() === "custom" && !confirm("Start again? The components you added are removed.")) return;
    startMode = "custom"; selComp = null; post("/api/model", { action: "empty" });
  };
  $("addcomp").onclick = () => openAdd();
  $("savemodel").onclick = saveModel;
  $("saveform").onsubmit = (e) => submitSave(e);
  $("closesave").onclick = $("cancelsave").onclick = () => $("savedlg").classList.add("hidden");
  $("savedlg").onclick = (e) => { if (e.target.id === "savedlg") $("savedlg").classList.add("hidden"); };
  $("sv-name").oninput = $("sv-folder").oninput = savePreview;
  $("sv-lib").onclick = () => { $("sv-folder").value = ""; savePreview(); };
  $("dlmodel").onclick = () => (S.model ? download("/api/model.json") : showError("Create a model first."));
  $("loadsaved").onclick = async () => {
    const path = $("savedsel").value, m = savedModels.find((x) => x.path === path);
    if (path && (await post("/api/model", { action: "load", path }))) { startMode = "saved"; selComp = null; toast(`Model ${m ? m.name : path} loaded`); }
  };
  $("loadfile").onclick = () => $("modelfile").click();
  $("modelfile").onchange = async (e) => {
    const f = e.target.files[0]; e.target.value = "";
    if (!f) return;
    const name = f.name.replace(/\.json$/i, "");
    if (await post("/api/model", { action: "upload", name, text: await f.text() })) {
      startMode = "saved"; selComp = null; toast(`Model ${name} loaded and added to your saved models`); refreshSaved();
    }
  };
  $("closeadd").onclick = closeAdd;
  $("adddlg").onclick = (e) => { if (e.target.id === "adddlg") closeAdd(); };
  $("addform").onsubmit = addSubmit;
  $("rmcomp").onclick = () => removeComp(selComp);
  $("kinsel").onchange = () => {
    const to = $("kinsel").value, c = selComp;
    post("/api/model", to ? { action: "tie", component: c, to } : { action: "untie", component: c })
      .then((ok) => ok && toast(to ? `${c}: shift and width tied to ${to}` : `${c}: shift and width fitted freely`));
  };
  document.querySelectorAll("#classdlgkind button").forEach((b) => (b.onclick = () => { showClassDialog.kind = b.dataset.k; showClassDialog(); }));
  $("closeclass").onclick = () => $("classdlg").classList.add("hidden");
  $("classdlg").onclick = (e) => { if (e.target.id === "classdlg") $("classdlg").classList.add("hidden"); };
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closeAdd(); $("classdlg").classList.add("hidden"); } });
  $("tofit").onclick = () => go("fit");
}

// ==================================================================== 4 FIT
let lastRun = null, runLog = [], fitInit = false, fitRefPending = null;
const errValue = () => (document.querySelector('input[name="errs"]:checked') || {}).value || "covariance";
function setErr(v) { const r = document.querySelector(`input[name="errs"][value="${v}"]`); if (r) r.checked = true; }
function fitForm() {
  return {
    epochs: (document.querySelector('input[name="epochs"]:checked') || {}).value || "joint",
    reference: $("refsel").value, n_jobs: +$("njobs").value || -1, shared: $("sharedin").value.trim(),
    calibration_degree: +$("calsel").value, host_per_epoch: $("hostepoch").checked, shared_shape: $("sharedshape").checked, rescale_errors: $("rescale").checked,
    errors: errValue(), scale_errors: $("scaleerr").checked, uncertainty: $("modelunc").checked, absorbers: $("absorb").checked,
    sample_model: $("smodel").value, ...(mode() === "sample" ? { n_jobs: +$("snjobs").value || -1 } : {}),
  };
}
// what will be fitted, with a way back to change it
function renderFitWhat() {
  const m = mode(), L = loaded(), P = prepared(), M = S.model;
  if (!L) { $("fitwhat").innerHTML = '<p class="muted">Read the spectra first (step 1).</p>'; return; }
  const dates = L.rows.map((r) => r.date).filter(Boolean);
  const name = L.rows.length === 1 ? L.rows[0].name : commonName(L.rows.map((r) => r.name));
  const spectra = m === "multi" ? `${L.shown} epochs of <b>${esc(name)}</b>${dates.length > 1 ? ` · ${esc(dates[0])} → ${esc(dates[dates.length - 1])}` : ""}`
    : m === "single" ? `<b>${esc(name)}</b>` : `${L.n} spectra`;
  const rows = [[m === "single" ? "Spectrum" : "Spectra", spectra + (P && P.z != null ? ` · z ${fmtNum(P.z, 4)}` : ""), "read"]];
  if (P && m !== "sample") rows.push(["Prepared", [P.rest_frame ? "rest frame" : "observed frame", P.ebv != null ? `E(B−V) ${fmtNum(P.ebv, 3)}` : "no Galactic extinction",
    P.flux_unit ? "flux unit known" : '<span class="warnword">flux unit unknown</span>'].join(" · "), "prepare"]);
  if (m === "single" || m === "multi") {
    const pl = presetLabel();
    const rng = M && (M.wave_range || (P && P.range));
    rows.push(["Model", M ? `<b>${esc(pl ? pl.name : "your own model")}</b> · ${M.components.length} components · ${M.n_free} free parameters` +
      (rng ? ` · ${Math.round(rng[0])}–${Math.round(rng[1])} Å` : "") : '<span class="warnword">no model yet</span>', "model"]);
  } else if (m === "classify") rows.push(["Model", "the classification fits its own (stars + power law, then the emission lines)", null]);
  $("fitwhat").innerHTML = `<h3 class="sect">What will be fitted</h3><dl>` + rows.map(([k, v, step]) =>
    `<dt>${k}</dt><dd>${v}${step ? ` <a href="#" class="textlink small" data-go="${step}">change</a>` : ""}</dd>`).join("") + "</dl>";
}
function runLabel() {
  const m = mode(), L = loaded(), n = L ? L.shown : 0, f = fitForm();
  if (m === "classify") return `Classify ${n === 1 ? "the spectrum" : `${n} spectra`}`;
  if (m === "sample") return `Fit ${L ? L.n : ""} spectra`;
  if (m === "multi") return f.epochs === "joint" ? `Run the joint fit of ${n} epochs` : `Fit the ${n} epochs`;
  return "Run the fit";
}
function renderFit() {
  const m = mode(), L = loaded();
  const n = L ? L.shown : 0;
  if (!fitInit && S) {
    fitInit = true;
    $("njobs").value = Math.max(1, Math.min(n || 1, (S.cpus || 2) - 1));
    $("snjobs").value = Math.max(1, (S.cpus || 2) - 1);
    $("smodel").innerHTML = '<option value="auto">chosen by the classification</option>' + S.models.map((l) => `<option>${esc(l)}</option>`).join("");
    const o = S.wf.settings.fit_opts;           // the settings of the last run
    if (o) {
      const r = document.querySelector(`input[name="epochs"][value="${o.epochs}"]`); if (r) r.checked = true;
      if (o.n_jobs > 0) { $("njobs").value = o.n_jobs; $("snjobs").value = o.n_jobs; }
      $("sharedin").value = o.shared || ""; $("calsel").value = String(o.calibration_degree || 0);
      $("hostepoch").checked = !!o.host_per_epoch; $("rescale").checked = !!o.rescale_errors;
      if (o.shared_shape !== undefined) $("sharedshape").checked = !!o.shared_shape;
      setErr(o.errors || "covariance"); $("scaleerr").checked = o.scale_errors !== false;
      $("modelunc").checked = !!o.uncertainty; $("absorb").checked = !!o.absorbers;
      if (o.sample_model) $("smodel").value = o.sample_model;
      fitRefPending = o.reference;
    }
  }
  const joint = m === "multi" && fitForm().epochs === "joint";
  $("epochsbox").classList.toggle("hidden", m !== "multi");
  $("samplebox").classList.toggle("hidden", m !== "sample");
  $("uncbox").classList.toggle("hidden", m === "classify" || m === "sample");
  $("advbox").classList.toggle("hidden", m === "classify" || m === "sample");
  $("uncbox-model").classList.toggle("hidden", m !== "single");
  $("uncbox-abs").classList.toggle("hidden", m !== "single");
  ["adv-host", "adv-rescale", "adv-shared"].forEach((id) => $(id).classList.toggle("hidden", !joint));
  $("adv-jobs").classList.toggle("hidden", m !== "multi");
  $("jointopts").classList.toggle("hidden", !joint);
  // Monte Carlo and MCMC are for independent fits
  document.querySelectorAll('input[name="errs"]').forEach((r) => {
    const off = joint && (r.value === "mc" || r.value === "mcmc");
    r.disabled = off; r.closest(".choice").classList.toggle("off", off);
  });
  if (joint && ["mc", "mcmc"].includes(errValue())) setErr("covariance");
  $("errnote").classList.toggle("hidden", !joint);
  $("errnote").textContent = "The joint fit gives covariance errors; for Monte Carlo or MCMC errors fit each epoch on its own.";
  if (L && $("refsel").dataset.sig !== String(L.id)) {
    $("refsel").dataset.sig = String(L.id);
    $("refsel").innerHTML = '<option value="auto">automatic (highest S/N)</option>' + L.rows.map((r) => `<option value="${r.k}">#${r.k}${r.date ? " · " + r.date : ""}</option>`).join("");
    if (fitRefPending != null) $("refsel").value = String(fitRefPending);
  }
  if (fitInit && lastRun === null && S.wf.settings.fit_opts && fitDone()) lastRun = JSON.stringify(fitForm());
  $("fit-hint").textContent = m === "classify" ? "[fa.classify(s) for s in spectra]" : m === "sample" ? "fa.fit_sample(files, …)"
    : m === "single" ? "fa.fit(spec, model)" : joint ? "fa.fit_epochs(spectra, model, …)" : "fa.fit_many(spectra, model, …)";
  renderFitWhat();
  const blocked = !prepared() && m !== "sample" ? "Prepare the spectra first." : (["single", "multi"].includes(m) && !S.model) ? "Create the model first." : "";
  const changed = lastRun !== null && JSON.stringify(fitForm()) !== lastRun;
  $("runfit").textContent = S.busy ? "Running…" : `▶ ${runLabel()}${fitDone() && !changed ? " again" : ""}`;
  $("runfit").disabled = !!S.busy || !!blocked;
  $("runnote").textContent = blocked || (S.busy ? "" : fitDone() && changed ? "The settings changed since the last run." : "");
  renderProgress();
  $("toresults").disabled = !fitDone() || !!S.busy;
}
function renderProgress() {
  const F = fitDone(), list = $("proglist");
  const cell = S.cells[S.cells.length - 1];
  const running = S.busy && cell && cell.code.startsWith("# fit");
  const failed = !running && S.error && cell && cell.code.startsWith("# fit");
  if (running && S.progress) {
    const p = S.progress, i = runLog.findIndex((x) => x.task === p.task);
    const entry = { task: p.task, done: p.done, total: p.total, detail: p.detail, t: p.time - p.start };
    if (i < 0) runLog.push(entry); else runLog[i] = entry;
  }
  // the progress card while running (and when it failed); the summary card when done
  $("progcard").classList.toggle("hidden", !running && !failed);
  $("donecard").classList.toggle("hidden", running || failed || !F);
  $("epochtblbox").classList.toggle("hidden", running || failed || !(F && F.epochs && F.epochs.length > 1 && F.chi2 != null));
  if (running || failed) {
    const items = runLog.map((e, i) => {
      const live = running && i === runLog.length - 1;
      const frac = e.total ? ` · ${e.done ?? 0} / ${e.total}` : (e.done ? ` · ${e.done}` : "");
      return `<li class="${live ? "run" : "ok"}">${esc(e.task)}${frac}${e.detail ? " · " + esc(e.detail) : ""}<span class="t">${e.t.toFixed(1)} s</span></li>`;
    });
    if (running) {
      const el = S.now - cell.start;
      if (!runLog.length) items.push(`<li class="run">Fitting<span class="t">${el.toFixed(1)} s</span></li>`);
      $("progstate").textContent = "running"; const last = runLog[runLog.length - 1];
      $("progbar").style.width = last && last.total ? `${Math.min(100, (100 * (last.done || 0)) / last.total)}%` : "35%";
    } else {
      items.push(`<li class="err">${esc(S.error)}</li>`); $("progstate").textContent = "failed"; $("progbar").style.width = "0";
    }
    list.innerHTML = items.join("");
    return;
  }
  if (!F) return;
  renderDone(F);
}
function renderDone(F) {
  let head, body = "";
  if (F.kind === "classify") head = `Classified ${F.n} ${F.n === 1 ? "spectrum" : "spectra"}: ${esc([...new Set(F.labels.map((l) => CLASS_NAME[l] || l))].join(", "))}`;
  else if (F.kind === "sample") head = `Fitted ${F.n - F.failed} of ${F.n} spectra` + (F.failed ? ` · <span class="warnword">${F.failed} failed</span>` : "");
  else {
    const q = F.rchi2;
    const [word, cls] = q <= 1.5 ? ["good", "ok"] : q <= 3 ? ["acceptable", "warn"] : ["poor", "bad"];
    const what = F.kind === "joint" ? `Joint fit of ${F.epochs.length} epochs` : F.kind === "fits" ? `${F.epochs.length - F.failed} of ${F.epochs.length} epochs fitted` : "Fit";
    head = `${what} · <span class="q ${cls}">χ²ᵣ ${q.toFixed(2)}, ${word}</span>`;
    const facts = [`${F.time.toFixed(1)} s`, `χ² ${F.chi2.toFixed(0)} for ${F.dof} degrees of freedom`];
    if (F.kind === "single") facts.push(`errors: ${esc(F.error_method || "none")}`);
    if (F.error_scale > 1) facts.push(`errors scaled ×${F.error_scale.toFixed(2)}`);
    body += `<p class="muted small">${facts.join(" · ")}</p>`;
    const warn = [];
    if (q > 1.5) warn.push("χ²ᵣ is well above 1: look at the residuals before trusting the line widths.");
    if (F.success === false) warn.push("The optimizer did not converge.");
    if (F.failed) warn.push(`${F.failed} epoch${F.failed > 1 ? "s" : ""} could not be fitted.`);
    // parameters at a limit, once each, with the epochs where they are
    const at = new Map(), ne = F.epochs.length;
    F.epochs.forEach((e) => (e.limits || []).forEach((p) => at.set(p, (at.get(p) || []).concat([e.k]))));
    const lim = [...at.entries()].map(([p, ks]) => `<li><code>${esc(p)}</code>${ne > 1 ? ` <span class="muted">${ks.length === ne ? `all ${ne} epochs` : "epoch" + (ks.length > 1 ? "s " : " ") + ks.join(", ")}</span>` : ""}</li>`);
    if (warn.length || lim.length) body += `<div class="callout"><div class="grow">${warn.length ? `<div>${warn.map(esc).join(" ")}</div>` : ""}` +
      (lim.length ? `<div><b>${lim.length} parameter${lim.length > 1 ? "s" : ""} at a limit</b>: the limit, not the data, sets ${lim.length > 1 ? "them" : "it"}. Widen the limit, fix the parameter or change the model.<ul>${lim.join("")}</ul></div>` : "") +
      "</div>" + (lim.length ? `<button class="btn small" id="limopen">Open ${esc([...at.keys()][0].split(".")[0])} in the Model step</button>` : "") + "</div>";
    renderDone.first = [...at.keys()][0];
  }
  $("donecard").innerHTML = `<div class="donehead"><span class="tick">✓</span><div class="grow"><b>${head}</b>${body}</div>` +
    `<button class="btn primary" id="seeres">See the results →</button></div>`;
  $("seeres").onclick = () => go("results");
  if ($("limopen")) $("limopen").onclick = () => { selComp = renderDone.first.split(".")[0]; go("model"); };
  if (F.epochs && F.epochs.length > 1 && F.chi2 != null) {
    $("epochtbl").innerHTML = '<thead><tr><th>Epoch</th><th>Date</th><th>Spectrum</th><th class="num">χ²ᵣ</th><th class="num">Flux scale</th><th>Parameters at a limit</th></tr></thead><tbody>' +
      F.epochs.map((e) => `<tr><td class="mono">#${e.k}${e.reference ? '<span class="badge ref">REF</span>' : ""}</td><td class="mono">${esc(e.date || "—")}</td>` +
        `<td class="mono">${esc(e.name)}</td><td class="num">${e.failed ? "failed" : e.rchi2.toFixed(3)}</td><td class="num">${e.scale ? e.scale.toFixed(3) : "—"}</td>` +
        `<td class="mono small">${(e.limits || []).map((x) => `<span class="badge warn">${esc(x)}</span>`).join(" ")}</td></tr>`).join("") + "</tbody>";
  }
}
function initFit() {
  document.querySelectorAll('input[name="epochs"], input[name="errs"]').forEach((r) => (r.onchange = renderFit));
  ["refsel", "njobs", "sharedin", "calsel", "sharedshape", "hostepoch", "rescale", "scaleerr", "modelunc", "absorb", "smodel", "snjobs"].forEach((id) => ($(id).onchange = () => { renderFit(); if (id === "sharedin" || id === "sharedshape") $("comps").dataset.sig = ""; }));
  $("runfit").onclick = () => {
    lastRun = JSON.stringify(fitForm()); runLog = [];
    post("/api/wf/fit", fitForm());
  };
  $("toresults").onclick = () => go("results");
}

// ================================================================ 5 RESULTS
let rk = 0, rtab = "lines", resultData = null, resultSig = null, sk = 0, ck = 0, rv = "fit", cfigKind = "steps";
let classData = null, samplePts = null, lastResidId = null;
// the views of the Results step that apply to this fit
function resultViews() {
  const F = fitDone(), kind = F && F.kind;
  if (!F) return [];
  if (kind === "classify") return ["class"];
  if (kind === "sample") return ["fit", "figure"];
  const v = ["fit", "figure", "lines"];
  if (resultData && resultData.corner_params && resultData.corner_params.length) v.push("posterior");
  if (S.wf.classified) v.push("class");
  const comps = (S.model && S.model.components) || [];
  if (comps.some((c) => c.type === "FeII") && comps.some((c) => c.role === "broad" && (c.lines || []).includes("Hb"))) v.push("ms");
  if (kind === "joint") v.push("time");
  return v;
}
const VIEW_HINT = { lines: "r.plot_lines(['Hb', 'Ha'], residuals=True)", ms: "r.main_sequence()", figure: "r.plot(xlim=…, group=…, label_lines=…)", posterior: "r.samples", class: "c.plot_steps() · c.plot()", time: "me.light_curve() · me.reverberation(line)" };
function showView() {
  const F = fitDone(), kind = F && F.kind, views = resultViews();
  if (!views.includes(rv)) rv = views[0] || "fit";
  document.querySelectorAll("#rnav button").forEach((b) => { b.classList.toggle("hidden", !views.includes(b.dataset.v)); b.classList.toggle("on", b.dataset.v === rv); });
  $("rnav").classList.toggle("hidden", views.length < 2);
  $("res-fit").classList.toggle("hidden", !F || rv !== "fit" || kind === "classify");
  $("res-sample").classList.toggle("hidden", kind !== "sample" || !["fit", "figure"].includes(rv));
  $("res-figure").classList.toggle("hidden", rv !== "figure");
  $("res-lines").classList.toggle("hidden", rv !== "lines");
  $("res-post").classList.toggle("hidden", rv !== "posterior");
  $("res-class").classList.toggle("hidden", rv !== "class");
  $("res-time").classList.toggle("hidden", rv !== "time");
  $("res-ms").classList.toggle("hidden", rv !== "ms");
  $("etabs").classList.toggle("hidden", rv === "class" || rv === "time" || rv === "ms");
  $("res-hint").textContent = VIEW_HINT[rv] || (kind === "joint" ? "me.epochs[k].plot() · me.epochs[k].lines" : kind === "fits" ? "fits[k].plot() · fits[k].lines"
    : kind === "sample" ? "sample.table · sample.load(k).plot()" : "r.plot() · r.lines");
  // downloads that apply
  $("dlwrap").classList.toggle("hidden", !F);
  const fitLike = ["single", "joint", "fits"].includes(kind);
  $("dl-lines").classList.toggle("hidden", kind === "classify");
  $("dl-lines").dataset.dl = kind === "sample" ? "/api/sample.csv" : "/api/wf/lines.csv";
  ["dl-fits", "dl-comp", "dl-plots"].forEach((id) => $(id).classList.toggle("hidden", !fitLike));
  $("dl-comp").querySelector("small").textContent = `CSV of the ${F && F.epochs && F.epochs.length > 1 ? "shown epoch" : "fit"}: data, model, every component, host, AGN`;
}
async function loadResult(force) {
  const F = fitDone();
  const kind = F ? F.kind : null;
  $("hideund").closest("label").classList.toggle("hidden", rtab !== "lines");
  document.querySelector('#rtabs [data-rt="all"]').classList.toggle("hidden", kind === "sample" || kind === "single");
  showView();
  if (!F) { $("etabs").innerHTML = '<p class="muted">Run the fit first (step 4).</p>'; return; }
  if (rv === "class") return loadClassify(force);
  if (rv === "time") return loadTime();
  if (rv === "ms") return loadMainSeq();
  const sig = JSON.stringify([F.chi2, F.time, F.n, kind === "sample" ? sk : rk]);
  if (!force && sig === resultSig && resultData) { renderResults(); return; }
  resultSig = sig;
  if (kind === "sample") {
    $("etabs").innerHTML = "";
    try { samplePts = await api("/api/wf/sample_points"); } catch (e) { samplePts = null; }
    renderSampleResults();
  }
  if (!resultData || resultData.k !== (kind === "sample" ? sk : rk)) {
    $("rplot").innerHTML = '<div class="empty">Loading the fit…</div>'; $("rlegend").innerHTML = ""; $("rsummary").innerHTML = "";
  }
  try { resultData = await api(kind === "sample" ? `/api/wf/sample?k=${sk}` : `/api/wf/result?k=${rk}`); } catch (e) { resultData = null; }
  showView();
  renderResults();
  if (kind !== "sample" && resultData && !resultData.error) loadMore(resultData);
  $("residcard").classList.toggle("hidden", !["joint", "fits", "single"].includes(kind));
  renderResid();
}
function renderResults() {
  const F = fitDone();
  const kind = F && F.kind;
  if (!resultData || kind === "classify") return;
  const R = resultData;
  if (R.epochs) {
    $("etabs").innerHTML = R.epochs.length > 1 ? R.epochs.map((e) => `<button data-k="${e.k}" class="${e.k === R.k ? "on" : ""} ${e.failed ? "failed" : ""}"><b>#${e.k}</b>${esc(e.date || e.name)}${e.reference ? '<span class="badge ref">REF</span>' : ""}</button>`).join("") : "";
    $("etabs").querySelectorAll("button").forEach((b) => (b.onclick = () => { rk = +b.dataset.k; loadResult(true); }));
  }
  $("rtitle").textContent = kind === "sample" ? `${R.name}${R.cls ? " · " + R.cls : ""}${R.preset ? " · " + R.preset : ""}` : (R.name || "");
  if (kind === "sample") $("ssel").textContent = `Fit of #${R.k} · ${R.name}`;
  if (R.error) { $("rplot").innerHTML = `<div class="empty">${esc(R.error)}</div>`; $("rtbl").innerHTML = ""; $("rlegend").innerHTML = ""; $("rsummary").innerHTML = ""; return; }
  renderSummary();
  if (rv === "fit") drawResultPlot();
  if (rv === "figure") showFigure();
  if (rv === "lines") showLines();
  if (rv === "posterior") renderCorner();
  renderResultTable();
}
// ---- the quasar main sequence: every epoch in the R_FeII - FWHM(Hβ) plane
let msData = null;
function mixColor(a, b, f) {
  const p = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const [x, y] = [p(a), p(b)];
  return "#" + x.map((v, i) => Math.round(v + (y[i] - v) * f).toString(16).padStart(2, "0")).join("");
}
async function loadMainSeq() {
  $("msplot").innerHTML = '<div class="empty">Measuring R<sub>Fe II</sub> and FWHM(Hβ) with errors…</div>';
  try { msData = await api("/api/wf/mainseq"); } catch (e) { msData = null; }
  drawMainSeq();
}
function drawMainSeq() {
  const d = msData;
  if (!d || !d.points) { $("msplot").innerHTML = '<div class="empty">No fit with Fe II and broad Hβ.</div>'; return; }
  const P = d.points, many = P.length > 1;
  const good = P.filter((p) => p.r_feii !== null && p.fwhm !== null);
  // several epochs: shaded by date (light = early, dark = late) and joined in time order; FWHM in 1000 km/s
  const t = good.map((p) => p.mjd).filter((v) => v !== null), t0 = Math.min(...t), t1 = Math.max(...t);
  const shade = (p) => (p.mjd === null || !(t1 > t0) ? C.broad[0] : p.k === rk ? C.accent : mixColor("#9ec5f4", "#104281", (p.mjd - t0) / (t1 - t0)));
  const pts = good.map((p) => ({ x: p.r_feii, y: p.fwhm / 1000, xerr: p.r_feii_err, yerr: p.fwhm_err / 1000, k: p.k,
    sel: !many || p.k === rk, color: many ? shade(p) : C.accent,
    tip: `${many ? `#${p.k} ${p.date || p.name}: ` : ""}R_FeII = ${p.r_feii.toFixed(2)}, FWHM = ${Math.round(p.fwhm)} km/s, ${p.type || p.population}` }));
  const path = many ? good.slice().sort((a, b) => (a.mjd ?? a.k) - (b.mjd ?? b.k)) : [];
  let xlim = [0, 3], ylim = [0, 16];
  $("mszoomwrap").classList.toggle("hidden", good.length < 2);
  if (good.length > 1 && $("mszoom").checked) {
    const xs = pts.flatMap((p) => [p.x - (p.xerr || 0), p.x + (p.xerr || 0)]), ys = pts.flatMap((p) => [p.y - (p.yerr || 0), p.y + (p.yerr || 0)]);
    const dx = Math.max(Math.max(...xs) - Math.min(...xs), 0.05), dy = Math.max(Math.max(...ys) - Math.min(...ys), 0.3);
    xlim = [Math.max(0, Math.min(...xs) - 0.15 * dx), Math.max(...xs) + 0.15 * dx];
    ylim = [Math.max(0, Math.min(...ys) - 0.15 * dy), Math.max(...ys) + 0.15 * dy];
  }
  const curves = d.grid.map((g) => ({ x: g.x, y: g.y.map((v) => v / 1000), dash: g.y[0] === 4000 && g.y[1] === 4000 ? null : "2 3" }));
  if (path.length > 1) curves.push({ x: path.map((p) => p.r_feii), y: path.map((p) => p.fwhm / 1000), dash: "1 2" });
  plotScatter($("msplot"), { xlim, ylim, curves, texts: d.types.map((tt) => ({ x: tt.x - 0.12, y: tt.y / 1000 + 0.6, t: tt.t })), points: pts,
    xlabel: "R_FeII = F(Fe II 4434–4684) / F(Hβ)", ylabel: "FWHM(Hβ)  10³ km/s",
    onClick: many ? (k) => { rk = k; drawMainSeq(); } : null, empty: pts.length ? null : "no R_FeII or FWHM(Hβ)" });
  $("mstbl").innerHTML = '<thead><tr>' + (many ? "<th>#</th><th>Date</th>" : "") + '<th class="num">R<sub>Fe II</sub></th><th class="num">FWHM(Hβ) km/s</th><th>Type</th></tr></thead><tbody>' +
    P.map((p) => `<tr data-k="${p.k}" class="${many && p.k === rk ? "sel" : ""}">` + (many ? `<td class="mono">#${p.k}</td><td class="mono">${esc(p.date || p.name)}</td>` : "") +
      `<td class="num">${p.r_feii === null ? "—" : pm(p.r_feii, p.r_feii_err)}</td><td class="num">${p.fwhm === null ? "—" : pm(p.fwhm, p.fwhm_err)}</td>` +
      `<td>${esc(p.type ? `${p.type} (Pop. ${p.population})` : p.population ? `Pop. ${p.population}` : "")}</td></tr>`).join("") + "</tbody>";
  if (many) {
    $("mstbl").querySelectorAll("tr[data-k]").forEach((tr) => (tr.onclick = () => { rk = +tr.dataset.k; drawMainSeq(); }));
    const sel = $("mstbl").querySelector("tr.sel");
    if (sel) sel.scrollIntoView({ block: "nearest" });
  }
  $("msnote").textContent = many ? "Light to dark: early to late epochs, joined in time order; the chosen epoch in red. Click a point or a row to choose it." : "";
  $("mszoom").onchange = drawMainSeq;
  $("mspng").onclick = () => download(`/api/wf/mainseq.png?k=${rk}&dl=1`);
  $("mspdf").onclick = () => download(`/api/wf/mainseq.pdf?k=${rk}&dl=1`);
}
// ---- the summary: fit quality, the main numbers, and what to look at
function prettyQuantity(n) {
  return esc(n).replace(/\bHb\b/g, "Hβ").replace(/\bHa\b/g, "Hα").replace(/\bHg\b/g, "Hγ").replace(/sigma\*/g, "σ★")
    .replace(/M_BH/g, "M<sub>BH</sub>").replace(/ A\b/g, " Å").replace(/^(?!log )[a-z]/, (c) => c.toUpperCase());
}
function renderSummary() {
  const R = resultData, F = fitDone();
  if (!R || !R.fit) { $("rsummary").innerHTML = ""; $("rlimits").classList.add("hidden"); return; }
  const q = R.fit.rchi2;
  const [word, cls] = q <= 1.5 ? ["good", "ok"] : q <= 3 ? ["acceptable", "warn"] : ["poor: look at the residuals", "bad"];
  const tiles = [`<div class="tile fitq ${cls}" title="reduced χ² = ${R.fit.chi2.toFixed(1)} / ${R.fit.dof} degrees of freedom"><small>Fit quality, reduced χ²</small><b>${q.toFixed(2)}</b><span>${word}</span></div>`];
  if (R.key === undefined) tiles.push('<div class="tile wait"><small>Main results</small><b>…</b><span>computing the errors</span></div>');
  else (R.key || []).slice(0, 5).forEach((k) => {
    const [v, e] = String(k.value).split(" ± ");
    tiles.push(`<div class="tile"><small>${prettyQuantity(k.name)}</small><b>${esc(v)}</b><span>${e ? "± " + esc(e) : ""} ${esc(k.unit || "")}</span></div>`);
  });
  const noUnit = prepared() && prepared().flux_unit == null;
  $("rsummary").innerHTML = tiles.join("") + (R.key && R.key.length > 5 ? `<a href="#" class="textlink moreq" id="moreq">All ${R.key.length} quantities with their errors →</a>` : "") +
    (noUnit ? '<p class="note unitnote">The flux unit is not known, so there are no luminosities or black-hole masses. <a href="#" class="textlink" data-go="prepare">Set it in Prepare</a></p>' : "");
  if ($("moreq")) $("moreq").onclick = (e) => { e.preventDefault(); rtab = "continuum"; renderResultTable(); $("rtabs").scrollIntoView({ behavior: "smooth", block: "start" }); };
  // parameters that ended at a limit: say so, and lead to where they are changed
  const ep = F && F.epochs ? F.epochs.find((e) => e.k === R.k) || F.epochs[0] : null;
  const lim = (ep && ep.limits) || [];
  $("rlimits").classList.toggle("hidden", !lim.length);
  if (lim.length) {
    $("rlimits").innerHTML = `<div class="grow"><b>${lim.length} parameter${lim.length > 1 ? "s" : ""} ended at a limit</b>: ${lim.slice(0, 6).map((p) => `<code>${esc(p)}</code>`).join(", ")}${lim.length > 6 ? " …" : ""}.` +
      ` The value is then set by the limit, not by the data: widen the limit, fix the parameter, or change the model.</div><button class="btn small" id="limfix">Open ${esc(lim[0].split(".")[0])} in the Model step</button>`;
    $("limfix").onclick = () => { selComp = lim[0].split(".")[0]; go("model"); };
  }
}
function drawResultPlot() {
  const R = resultData;
  if (!R || !R.plot) return;
  drawFit($("rplot"), $("rlegend"), R.plot, hiddenSeries, () => drawResultPlot());
  $("raxis").textContent = `rest-frame λ ${Math.round(R.plot.range[0])}–${Math.round(R.plot.range[1])} Å`;
}
// ---- the publication figure, redrawn as the options change
function figUrl(fmt, dl) {
  const q = new URLSearchParams({ k: rk });
  $("figopts").querySelectorAll("[data-o]").forEach((el) => {
    const v = el.type === "checkbox" ? (el.checked ? "1" : "0") : el.value.trim();
    if (v !== "") q.set(el.dataset.o, v);
  });
  if (dl) q.set("dl", "1"); else q.set("v", S.cells.length);
  return `/api/wf/figure.${fmt}?${q}`;
}
// what the figure draws: the data, the model and each kind of component in the model, each with its colour
const FIG_KIND = [["data", "Data", "#9b9a95"], ["model", "Model", "#0b0b0b"], ["continuum", "Continuum", "#52514e"], ["host", "Host galaxy", "#008300"],
  ["balmer_cont", "Balmer continuum", "#86b6ef"], ["balmer_lines", "High-order Balmer lines", "#86b6ef"], ["feii", "Fe II", "#e34948"],
  ["narrow", "Narrow lines", "#1baf7a"], ["broad", "Broad lines", "#2a78d6"], ["outflow", "Outflow", "#eda100"], ["other", "Other lines", "#898781"]];   // plotting.COLORS
function figKinds() {
  const have = new Set(["data", "model"]);
  ((S.model && S.model.components) || []).forEach((c) => {
    const t = c.type;
    if (t === "Lines") have.add(c.absorption ? "other" : ["narrow", "broad", "outflow"].includes(c.role) ? c.role : "other");
    else if (t === "FeII") have.add("feii");
    else if (t === "HighOrderBalmer") have.add("balmer_lines");
    else if (t === "BalmerContinuum") have.add("balmer_cont");
    else if (t === "StellarPopulation" || t === "HostGalaxy") have.add("host");
    else if (["PowerLaw", "BrokenPowerLaw", "Blackbody", "Polynomial"].includes(t)) have.add("continuum");
  });
  return FIG_KIND.filter((k) => have.has(k[0]));
}
function renderDrawList() {
  const kinds = figKinds(), sig = kinds.map((k) => k[0]).join();
  const box = $("fo-draw");
  if (box.dataset.sig === sig) return;
  box.dataset.sig = sig;
  // per kind: shown, fill under the curve (Fe II by default), colour, line width and line style
  const lw0 = { data: 0.6, model: 0.9 };
  box.innerHTML = '<div class="drawrow head"><span title="Fill under the curve">fill</span><span>colour</span><span>width</span><span>style</span></div>' +
    kinds.map(([k, label, color]) => {
      const show = k === "data" || k === "model" ? k : `show_${k}`, lname = esc(label.toLowerCase());
      const fill = k === "data" || k === "model" ? "<span></span>"
        : `<input type="checkbox" data-o="fill_${k}" ${k === "feii" ? "checked" : ""} title="Fill under the ${lname} curve">`;
      const style = k === "data" ? "<span></span>"
        : `<select data-o="ls_${k}" title="Line style of ${lname}"><option value="">—</option><option value="solid">solid</option>` +
          `<option value="dashed">dashed</option><option value="dotted">dotted</option><option value="dashdot">dash-dot</option></select>`;
      return `<div class="drawrow"><label><input type="checkbox" data-o="${show}" checked> ${esc(label)}</label>${fill}` +
        `<input type="color" data-o="color_${k}" value="${color}" title="Colour of ${lname}">` +
        `<input type="number" data-o="lw_${k}" min="0.1" max="6" step="0.1" placeholder="${lw0[k] ?? 1}" title="Line width of ${lname} (points)">${style}</div>`;
    }).join("");
  box.querySelectorAll("[data-o]").forEach((el) => (el[el.type === "checkbox" || el.tagName === "SELECT" ? "onchange" : "oninput"] = figChanged));
}
function showFigure() {
  renderDrawList();
  $("fo-labelopts").classList.toggle("hidden", $("fo-labels").value === "0");
  $("fo-title").disabled = $("fo-notitle").checked;
  const sample = fitDone() && fitDone().kind === "sample";
  $("figopts").classList.toggle("hidden", sample);
  const url = withToken(sample ? `/api/sample_fig.png?k=${sk}&kind=fit` : figUrl("png"));
  if ($("rpub").dataset.url === url) return;
  $("rpub").dataset.url = url;
  $("figwait").classList.remove("hidden");
  $("rpub").onload = $("rpub").onerror = () => $("figwait").classList.add("hidden");
  $("rpub").src = url;
}
// ---- line profiles: a grid of the ticked lines, redrawn as the choice changes
let lineList = null, lineKey = null, lineSel = [];
function linesUrl(fmt, dl) {
  const q = new URLSearchParams({ k: rk, lines: lineSel.join(",") });
  $("lineopts").querySelectorAll("[data-l]").forEach((el) => {
    const v = el.type === "checkbox" ? (el.checked ? "1" : "0") : el.value;
    if (v !== "") q.set(el.dataset.l, v);
  });
  if (dl) q.set("dl", "1"); else q.set("v", S.cells.length);
  return `/api/wf/linegrid.${fmt}?${q}`;
}
async function showLines() {
  const key = rk + ":" + S.cells.length;
  if (lineKey !== key) {
    lineKey = key;
    try { lineList = await api(`/api/wf/line_list?k=${rk}`); } catch (e) { lineList = { lines: [] }; }
    const names = lineList.lines.map((d) => d.line);
    lineSel = lineSel.filter((l) => names.includes(l));
    if (!lineSel.length && lineList.main) lineSel = [lineList.main];
    $("lo-lines").innerHTML = lineList.lines.map((d) => `<label class="check linechk"><input type="checkbox" value="${esc(d.line)}" ${lineSel.includes(d.line) ? "checked" : ""}>` +
      `<span>${esc(d.label)}</span><small>${d.roles.join(" + ")}</small></label>`).join("") || '<p class="note small">No fitted lines.</p>';
    $("lo-lines").querySelectorAll("input").forEach((el) => (el.onchange = () => {
      lineSel = el.checked ? lineSel.concat(el.value) : lineSel.filter((l) => l !== el.value);
      const order = lineList.lines.map((d) => d.line);        // panels by wavelength
      lineSel.sort((a, b) => order.indexOf(a) - order.indexOf(b));
      linesChanged();
    }));
  }
  const none = !lineSel.length;
  $("linesempty").classList.toggle("hidden", !none);
  $("rlinesimg").classList.toggle("hidden", none);
  if (none) return;
  const url = withToken(linesUrl("png"));
  if ($("rlinesimg").dataset.url === url) return;
  $("rlinesimg").dataset.url = url;
  $("linewait").classList.remove("hidden");
  $("rlinesimg").onload = $("rlinesimg").onerror = () => $("linewait").classList.add("hidden");
  $("rlinesimg").src = url;
}
let lineTimer = null;
function linesChanged() { clearTimeout(lineTimer); lineTimer = setTimeout(showLines, 350); }
let figTimer = null;
function figChanged() { clearTimeout(figTimer); figTimer = setTimeout(showFigure, 450); }
// ---- the posterior: the corner plot, drawn when the view opens and when the choice changes
function renderCorner() {
  const P = (resultData && resultData.corner_params) || [];
  const sig = JSON.stringify([rk, P]);
  if ($("cornerparams").dataset.sig !== sig) {
    $("cornerparams").dataset.sig = sig;
    // shown first: the kinematics of the broad lines, then Fe II and the narrow lines
    const rank = (p) => (/^broad\./.test(p) ? 0 : /^broad/.test(p) ? 1 : /^fe/.test(p) ? 2 : /^narrow/.test(p) ? 3 : 4);
    const on = new Set(P.map((p, i) => [rank(p), i, p]).sort((a, b) => a[0] - b[0] || a[1] - b[1]).slice(0, 5).map((x) => x[2]));
    $("cornerparams").innerHTML = P.map((p) => `<label><input type="checkbox" value="${esc(p)}" ${on.has(p) ? "checked" : ""}> ${esc(p)}</label>`).join("");
    $("cornerparams").querySelectorAll("input").forEach((c) => (c.onchange = () => { clearTimeout(figTimer); figTimer = setTimeout(drawCorner, 500); }));
  }
  drawCorner();
}
function cornerUrl(fmt, dl) {
  const ps = [...$("cornerparams").querySelectorAll("input:checked")].map((c) => c.value).join(",");
  return `/api/wf/corner.${fmt}?k=${rk}&params=${encodeURIComponent(ps)}${dl ? "&dl=1" : ""}`;
}
function drawCorner() {
  const url = withToken(cornerUrl("png"));
  if ($("cornerimg").dataset.url === url) return;
  $("cornerimg").dataset.url = url;
  $("cornerwait").classList.remove("hidden");
  $("cornerimg").onload = $("cornerimg").onerror = () => $("cornerwait").classList.add("hidden");
  $("cornerimg").src = url;
}
const hiddenSeries = new Set();
// a fit (data, components, model, residuals, labels) with a clickable legend
function drawFit(el, legendEl, P, hidden, redraw) {
  const layers = [{ y: P.flux, color: C.data, width: 1, range: true, name: "data", label: "data", hidden: hidden.has("data") }];
  let bi = 0, oi = 0;
  for (const s of P.series) {
    let color = C.other[oi % C.other.length], dash = null;
    if (s.bowen) color = C.bowen;                       // Bowen fluorescence + He II lines
    else if (s.role === "feii") color = C.feii;
    else if (s.role === "balmer") color = C.balmer;
    else if (s.role === "broad") color = C.broad[bi++ % C.broad.length];
    else if (s.role === "narrow") color = C.narrow;
    else if (s.role === "outflow") color = C.outflow;
    else if (s.role === "host") color = C.host;
    else if (s.role === "continuum") { color = C.continuum; dash = "5 4"; }
    else oi++;
    if (s.role === "balmer") dash = "6 3";              // told apart from the broad blues by the dash
    const label = s.role === "broad" && s.lines && s.lines.length ? `${s.name} (${s.lines.slice(0, 3).join(", ")}${s.lines.length > 3 ? "…" : ""})` : s.name;
    layers.push({ y: s.y, color, width: 1.3, dash, fill: s.role === "feii", name: s.name, label, hidden: hidden.has(s.name) });
  }
  layers.push({ y: P.model, color: C.model, width: 1.2, name: "model", label: "model", range: true, hidden: hidden.has("model") });
  plotSpectrum(el, { x: P.x, layers, zero: true, resid: P.resid, labels: P.labels, hatch: P.masked,
    bands: el.id === "rplot" ? residBands() : [] });
  legend(legendEl, layers, hidden, redraw);
}
function legend(el, layers, hidden, redraw) {
  el.innerHTML = layers.map((l) => `<span data-s="${esc(l.name)}" class="${l.hidden ? "off" : ""}"><i class="${l.dash ? "dash" : ""}" style="border-color:${l.color}"></i>${esc(l.label || l.name)}</span>`).join("");
  el.querySelectorAll("span").forEach((s) => (s.onclick = () => { const n = s.dataset.s; hidden.has(n) ? hidden.delete(n) : hidden.add(n); redraw(); }));
}
// ------------------------------------------------- light curves, reverberation
let lcData = null, lcKind = "flux", lcTimer = null;
async function loadTime() {
  try { lcData = await api("/api/wf/lightcurve"); } catch (e) { lcData = null; }
  clearTimeout(lcTimer);
  if (lcData && lcData.computing) {
    $("lcgrid").innerHTML = '<p class="muted">Computing the light curves (errors of every line in every epoch)…</p>';
    lcTimer = setTimeout(loadTime, 3000);
    return;
  }
  drawLC();
  renderReverb();
}
function drawLC() {
  const L = lcData;
  if (!L || !L.series) { $("lcgrid").innerHTML = '<p class="muted">No light curves.</p>'; return; }
  document.querySelectorAll("#lckind button").forEach((b) => b.classList.toggle("on", b.dataset.k === lcKind));
  const series = L.series.filter((d) => d.kind === lcKind || (lcKind === "flux" && d.line.startsWith("cont")));
  $("lcgrid").innerHTML = series.map((d, i) => `<div class="lc"><b>${esc(d.label)}</b> <span class="muted small">${esc(lcKind === "flux" ? (d.line.startsWith("cont") ? "flux density" : "broad flux") : lcKind.toUpperCase())}${d.unit ? " (" + esc(d.unit) + ")" : ""}</span><div class="plot" id="lc-${i}"></div></div>`).join("");
  const days = L.mjd.map((m) => (m === null ? null : m - L.t0));
  series.forEach((d, i) => {
    const pts = [], cx = [], cy = [];
    d.y.forEach((y, k) => {
      if (y === null || days[k] === null) return;
      pts.push({ x: days[k], y, yerr: d.err[k], k, sel: k === rk, color: d.line.startsWith("cont") ? C.continuum : C.broad[0],
        tip: `#${k} ${L.dates[k] || ""} · ${fmtNum(y, 4)} ± ${fmtNum(d.err[k], 2)}` });
      cx.push(days[k]); cy.push(y);
    });
    if (!pts.length) return;
    const ys = pts.flatMap((p) => [p.y - (p.yerr || 0), p.y + (p.yerr || 0)]);
    let y0 = Math.min(...ys), y1 = Math.max(...ys); const pad = (y1 - y0 || Math.abs(y1) || 1) * 0.12;
    const xs = pts.map((p) => p.x); let x0 = Math.min(...xs), x1 = Math.max(...xs); const px = (x1 - x0 || 1) * 0.05;
    const el = $("lc-" + i);
    plotScatter(el, { xlim: [x0 - px, x1 + px], ylim: [y0 - pad, y1 + pad], curves: [{ x: cx, y: cy, dash: "2 3" }], points: pts,
      xlabel: `days since MJD ${L.t0.toFixed(0)}`, ylabel: "", onClick: (k) => { rk = k; loadResult(true); } });
    resizeObs.observe(el);
  });
}
function renderReverb() {
  const L = lcData;
  if (L && L.lines && $("rvline").dataset.sig !== L.lines.join()) {
    $("rvline").dataset.sig = L.lines.join();
    $("rvline").innerHTML = L.lines.map((l) => `<option value="${esc(l)}">${esc(prettyLine(l))}</option>`).join("");
    if (L.lines.includes("Hb")) $("rvline").value = "Hb";
  }
  const R = S.wf.reverb;
  $("rvrun").disabled = !!S.busy;
  $("rvrun").textContent = S.busy && S.cells[S.cells.length - 1].code.startsWith("# reverberation") ? "Measuring…" : "▶ Measure lag";
  $("rvout").classList.toggle("hidden", !R);
  if (!R) return;
  $("rvwarn").classList.toggle("hidden", !R.warnings.length);
  $("rvwarn").innerHTML = `<div><ul style="margin:0">${R.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></div>`;
  $("rvtbl").innerHTML = `<thead><tr><th>${esc(prettyLine(R.line))}: quantity</th><th class="num">Value</th><th class="num">Error</th><th>Unit</th></tr></thead><tbody>` +
    R.rows.map((r) => `<tr><td>${esc(r[0])}</td><td class="num">${esc(r[1])}</td><td class="num">${esc(r[2])}</td><td class="muted">${esc(r[3])}</td></tr>`).join("") + "</tbody>";
  $("rvlags").innerHTML = '<thead><tr><th>Method</th><th class="num">Lag rest (d)</th><th class="num">r max</th><th class="num">outside peak</th></tr></thead><tbody>' +
    R.lags.map((q) => `<tr class="${q.adopted ? "sel" : ""}"><td>${esc(q.method)}${q.adopted ? ' <span class="badge ok">adopted</span>' : ""}</td><td class="num">${esc(q.lag)}</td><td class="num">${esc(q.r_max)}</td><td class="num">${esc(q.outside)}</td></tr>`).join("") + "</tbody>";
  const src = withToken(`/api/wf/reverb.png?v=${R.id}`);
  if ($("rvfig").dataset.src !== src) { $("rvfig").dataset.src = src; $("rvfig").src = src; }
}
function initTime() {
  document.querySelectorAll("#lckind button").forEach((b) => (b.onclick = () => { lcKind = b.dataset.k; drawLC(); }));
  $("rvrun").onclick = () => post("/api/wf/reverb", {
    line: $("rvline").value, n_mc: +$("rvmc").value || 1000, detrend: $("rvdetrend").checked,
    methods: [...document.querySelectorAll(".rvform [data-m]:checked")].map((c) => c.dataset.m),
    mjd_min: $("rvmin").value.trim() || null, mjd_max: $("rvmax").value.trim() || null,
  });
}

// the slow parts (main numbers with errors, the table of all epochs) arrive after the plot
async function loadMore(R) {
  const mine = R;
  if (R.key === undefined) {
    try { const d = await api(`/api/wf/result_more?k=${R.k}&part=key`); if (resultData === mine) { R.key = d.key || []; R.bh = d.bh || []; renderResultTable(); renderSummary(); } } catch (e) { R.key = []; renderSummary(); }
  }
  if (R.all === undefined && R.epochs && R.epochs.length > 1) {
    try { const d = await api("/api/wf/result_more?part=all"); if (resultData === mine) { R.all = d.all; renderResultTable(); } } catch (e) { R.all = null; }
  }
}
// ------------------------------------------------------------- residuals
const RES_TINT = { emission: "rgba(15,143,126,.16)", absorption: "rgba(212,71,42,.14)", continuum: "rgba(168,117,0,.12)", sky: "rgba(120,126,140,.16)" };
const appliedRes = new Set();
function residRows() {
  const Rz = S.wf.residuals;
  if (!Rz) return [];
  const kind = fitDone() && fitDone().kind;
  // features of the shown epoch (all-epoch analysis: where found; one-epoch analysis: that epoch)
  return Rz.rows.filter((r) => Rz.scope === "all" ? (kind === "single" || r.epochs.includes(rk)) : Rz.scope === rk);
}
function residBands() {
  return residRows().map((r) => ({ lo: r.lo, hi: r.hi, label: String(r.i + 1), tint: RES_TINT[r.kind] }));
}
function renderResid() {
  const F = fitDone(), kind = F && F.kind;
  if (!F || !["joint", "fits", "single"].includes(kind)) return;
  const many = kind !== "single";
  const scopeSig = many ? `${rk}` : "single";
  if ($("rscope").dataset.sig !== scopeSig) {
    $("rscope").dataset.sig = scopeSig;
    $("rscope").innerHTML = many ? `<option value="all">all epochs</option><option value="${rk}">epoch #${rk}</option>` : '<option value="all">this fit</option>';
  }
  const obj = kind === "joint" ? "me" : kind === "fits" ? "fits" : "r";
  $("residhint").textContent = `fa.analyze_residuals(${obj})`;
  const busy = S.busy && S.cells[S.cells.length - 1].code.startsWith("# residuals");
  $("residrun").disabled = !!S.busy;
  $("residrun").textContent = busy ? "Analysing…" : "Analyse residuals";
  const Rz = S.wf.residuals;
  $("residbox").classList.toggle("hidden", !Rz);
  if (!Rz) return;
  const sig = JSON.stringify([Rz.id, [...appliedRes], S.busy]);
  if ($("residtbl").dataset.sig === sig) return;
  $("residtbl").dataset.sig = sig;
  $("residnote").textContent = Rz.rows.length ? `${Rz.rows.length} feature${Rz.rows.length > 1 ? "s" : ""} above the threshold` +
    (Rz.scope === "all" && many ? ` in ${Rz.n_epochs} epochs (most frequent first; the plot marks those of the shown epoch).` : ".")
    : "No significant residual features: the model describes the data down to the noise.";
  $("residtbl").innerHTML = '<thead><tr><th>#</th><th class="num">λ (Å)</th><th>Kind</th><th class="num">σ</th><th class="num">FWHM</th><th>Line</th>' +
    (many && Rz.scope === "all" ? '<th class="num">Epochs</th>' : "") + "<th>Suggestion</th><th></th></tr></thead><tbody>" +
    Rz.rows.map((r) => {
      const done = appliedRes.has(`${Rz.id}:${r.i}`);
      return `<tr class="${done ? "applied" : ""}"><td class="mono">${r.i + 1}</td><td class="num">${fmtNum(r.center, 5)}</td>` +
        `<td><span class="badge k-${r.kind}">${esc(r.kind)}</span></td><td class="num">${r.sigma.toFixed(1)}</td><td class="num">${Math.round(r.fwhm)}</td>` +
        `<td>${r.line ? `<b>${esc(prettyLine(r.line))}</b>${r.in_model ? ' <small class="muted">in model</small>' : ""}` : '<span class="muted">—</span>'}</td>` +
        (many && Rz.scope === "all" ? `<td class="num">${r.epochs.length}/${Rz.n_epochs}</td>` : "") +
        `<td class="sugg">${esc(r.suggestion)}${r.code ? `<code>${esc(r.code)}</code>` : ""}</td>` +
        `<td>${r.action ? `<button class="btn small" data-apply="${r.i}" ${done || S.busy ? "disabled" : ""}>${done ? "Applied" : "Apply"}</button>` : ""}</td></tr>`;
    }).join("") + "</tbody>";
  $("residtbl").querySelectorAll("[data-apply]").forEach((b) => (b.onclick = () => applyResid(Rz.rows[+b.dataset.apply], Rz.id)));
  const n = [...appliedRes].filter((k) => k.startsWith(Rz.id + ":")).length;
  $("residfoot").classList.toggle("hidden", !n);
  $("residapplied").textContent = n ? `${n} change${n > 1 ? "s" : ""} applied to the ${Rz.rows.some((r) => r.action && r.action.type === "mask") ? "model / masks" : "model"}: refit to see the effect.` : "";
}
async function applyResid(r, id) {
  const a = r.action;
  let ok = false;
  if (a.type === "lines") ok = await post("/api/model", { action: "add", component: { type: "Lines", name: a.name, lines: a.lines, role: a.role } });
  else if (a.type === "component") ok = await post("/api/model", { action: "add", component: { type: a.component } });
  else if (a.type === "mask") {
    if (!prep) prep = prepDefaults();
    prep.masks.push(a.range);
    ok = await post("/api/wf/prepare", Object.assign({}, prep, { apply_to: "all" }));
  }
  if (ok) { appliedRes.add(`${id}:${r.i}`); toast("Applied: " + (r.code || r.suggestion)); renderResid(); }
}
function initResid() {
  $("residrun").onclick = () => post("/api/wf/residuals", { scope: $("rscope").value, nsigma: +$("rnsig").value });
  $("residrefit").onclick = () => go("fit");
}

function renderResultTable() {
  const R = resultData, t = $("rtbl");
  if (!R || R.error) return;
  document.querySelectorAll("#rtabs button").forEach((b) => b.classList.toggle("on", b.dataset.rt === rtab));
  $("hideund").closest("label").classList.toggle("hidden", rtab !== "lines");
  if (rtab === "lines") t.innerHTML = lineTable(R.lines, $("hideund").checked);
  else if (rtab === "continuum") {
    t.innerHTML = '<thead><tr><th>Quantity</th><th class="num">Value</th><th>Unit / status</th></tr></thead><tbody>' +
      (R.key === undefined ? '<tr><td colspan="3" class="muted">Computing the main numbers and their errors…</td></tr>' : "") +
      (R.key || []).map((q) => `<tr><td><b>${esc(q.name)}</b></td><td class="num">${esc(q.value)}</td><td class="muted">${esc(q.unit)}</td></tr>`).join("") +
      (R.bh && R.bh.length ? '<tr class="grouphead"><td colspan="3"><b>Black-hole mass, every recipe</b> <span class="muted small">log M<sub>BH</sub>/M<sub>☉</sub> ± statistical error; the recipes scatter by the stated dex around reverberation masses</span></td></tr>' +
        R.bh.map((b) => `<tr><td>${esc(prettyLine(b.line))} · ${b.width === "sigma" ? "σ<sub>line</sub>" : "FWHM"} · ${b.luminosity === "broad line" ? "L(" + esc(prettyLine(b.line)) + ")" : "L<sub>" + esc(b.luminosity.slice(1)) + "</sub>"}<br><small class="muted">${esc(b.reference)} <code>recipe="${esc(b.recipe)}"</code></small></td>` +
          `<td class="num">${esc(b.value)}</td><td class="muted">dex, scatter ${b.scatter} dex</td></tr>`).join("") : "") +
      (R.continuum || []).map((p) => `<tr><td class="mono">${esc(p.name)}</td><td class="num">${esc(p.value)}</td><td>${statusBadge(p.status)} <span class="muted">${esc(p.unit)}</span></td></tr>`).join("") + "</tbody>";
  } else if (rtab === "all") {
    const A = R.all;
    if (A === undefined && R.epochs && R.epochs.length > 1) { t.innerHTML = '<tbody><tr><td class="muted">Computing the light curve of all epochs (errors of every line in every epoch)…</td></tr></tbody>'; return; }
    t.innerHTML = !A ? '<tbody><tr><td class="muted">One spectrum: nothing to compare across epochs.</td></tr></tbody>' :
      `<thead><tr>${A.columns.map((c) => `<th class="${c === "name" ? "" : "num"}">${esc(c)}</th>`).join("")}</tr></thead><tbody>` +
      A.rows.map((r) => `<tr>${r.map((v, i) => `<td class="${A.columns[i] === "name" ? "mono" : "num"}">${esc(v)}</td>`).join("")}</tr>`).join("") + "</tbody>";
  } else {
    t.innerHTML = '<thead><tr><th>Parameter</th><th class="num">Value</th><th class="num">Min</th><th class="num">Max</th><th>Unit</th><th>Status</th></tr></thead><tbody>' +
      (R.params || []).map((p) => `<tr><td class="mono">${esc(p.name)}</td><td class="num">${esc(p.value)}</td><td class="num">${fmtNum(p.min)}</td><td class="num">${fmtNum(p.max)}</td><td class="muted">${esc(p.unit)}</td><td>${statusBadge(p.status)}</td></tr>`).join("") + "</tbody>";
  }
}
function lineTable(lines, hideUnd) {
  // grouped by component; a width and shift shared by the lines of a component are given once, in its header
  const rows = lines.filter((r) => !hideUnd || r.detected);
  const groups = [];
  rows.forEach((r) => { let g = groups.find((x) => x.name === r.component); if (!g) groups.push((g = { name: r.component, role: r.role, rows: [] })); g.rows.push(r); });
  const same = (g, k) => g.rows.length > 1 && g.rows.every((r) => r[k] !== null && Math.abs(r[k] - g.rows[0][k]) <= 1e-6 * Math.max(1, Math.abs(g.rows[0][k])));
  return '<thead><tr><th>Line</th><th class="num">Flux</th><th class="num">FWHM (km/s)</th><th class="num">Shift (km/s)</th><th class="num">EW (Å)</th><th>Flag</th></tr></thead><tbody>' +
    groups.map((g) => {
      const sw = same(g, "fwhm"), ss = same(g, "shift"), r0 = g.rows[0];
      const head = `<tr class="grouphead"><td colspan="2"><b class="mono">${esc(g.name)}</b>${g.role ? ` <span class="muted small">${esc(g.role)}</span>` : ""}</td>` +
        `<td class="num">${sw ? pm(r0.fwhm, r0.fwhm_err) : ""}</td><td class="num">${ss ? pm(r0.shift, r0.shift_err) : ""}</td><td></td><td></td></tr>`;
      return head + g.rows.map((r) => `<tr class="${r.detected ? "" : "dim"}"><td class="lname">${esc(r.label)}</td>` +
        `<td class="num">${r.detected ? pm(r.flux, r.flux_err) : "0"}</td><td class="num">${sw ? "" : pm(r.fwhm, r.fwhm_err)}</td><td class="num">${ss ? "" : pm(r.shift, r.shift_err)}</td>` +
        `<td class="num">${r.detected ? pm(r.ew, r.ew_err) : "0"}</td><td>${r.flags.map((f) => `<span class="badge ${f.kind}">${esc(f.text)}</span>`).join(" ")}</td></tr>`).join("");
    }).join("") + "</tbody>";
}
function statusBadge(s) {
  const kind = /limit|bound/.test(s) ? "warn" : s === "free" ? "ok" : "muted";
  return `<span class="badge ${kind}">${esc(s)}</span>`;
}
// ---------------------------------------------------------- classification
const CLASS_COLOR = { type1: "#2563c9", type1_host: "#3a9bd5", seyfert2: "#d4472a", liner: "#a87500", composite: "#7b3fd0",
  starforming: "#0f8f7e", retired: "#b8458f", passive: "#6b7280", unknown: "#9aa1ad" };
const hidC = { star: new Set(), ha: new Set(), lines: new Set() };
async function loadClassify(force) {
  let d;
  try { d = await api(`/api/wf/classify?k=${ck}`); } catch (e) { return; }
  if (!d || !d.labels) return;
  classData = d;
  $("etabs").innerHTML = "";
  const many = d.labels.length > 1;
  $("cmulti").classList.toggle("hidden", !many);
  if (many) {
    $("ctbl").innerHTML = '<thead><tr><th>#</th><th>Date</th><th>Spectrum</th><th>Class</th></tr></thead><tbody>' +
      d.labels.map((x) => `<tr data-k="${x.k}" class="${x.k === d.k ? "sel" : ""}"><td class="mono">#${x.k}</td><td class="mono">${esc(x.date || "—")}</td><td class="mono">${esc(x.name)}</td>` +
        `<td><span class="badge ${x.uncertain ? "warn" : "ok"}">${esc(x.label)}${x.uncertain ? " (uncertain)" : ""}</span></td></tr>`).join("") + "</tbody>";
    $("ctbl").querySelectorAll("tr[data-k]").forEach((tr) => (tr.onclick = () => { ck = +tr.dataset.k; loadClassify(true); }));
    const pts = d.labels.filter((x) => x.points.nii).map((x) => ({ x: x.points.nii.x, y: x.points.nii.y, k: x.k, sel: x.k === d.k,
      color: CLASS_COLOR[x.label] || "#6b7280", tip: `#${x.k} ${x.name}: ${x.label}` }));
    plotScatter($("cbptall"), Object.assign({}, d.diagrams.nii, { points: pts, xlabel: "log [N II]/Hα", ylabel: "log [O III]/Hβ",
      onClick: (k) => { ck = k; loadClassify(true); } }));
  }
  $("cverdict").innerHTML = `<div><b>${esc(d.name)}: <span class="badge ${d.uncertain ? "warn" : "ok"}">${esc(CLASS_NAME[d.label] || d.label)}${d.uncertain ? " (uncertain)" : ""}</span></b>` +
    (d.reason ? `<div class="small" style="margin-top:4px">${esc(d.reason)}</div>` : "") + "</div>";
  drawClassPlots();
  const diag = { nii: ["log [N II]/Hα", "log [O III]/Hβ"], sii: ["log [S II]/Hα", "log [O III]/Hβ"], oi: ["log [O I]/Hα", "log [O III]/Hβ"], whan: ["log [N II]/Hα", "log EW(Hα) [Å]"] };
  for (const [key, [xl, yl]] of Object.entries(diag)) {
    const p = d.points[key];
    // several epochs: all of them (coloured by class), the chosen one outlined; one spectrum: its point
    const pts = many ? d.labels.filter((x) => x.points[key]).map((x) => Object.assign({ k: x.k, sel: x.k === d.k, color: CLASS_COLOR[x.label] || "#6b7280",
      tip: `#${x.k} ${x.date || ""} ${x.name}: ${x.label} (x = ${x.points[key].x.toFixed(3)}, y = ${x.points[key].y.toFixed(3)}${x.points[key].fit ? ", narrow lines of the fit" : ""})` }, x.points[key]))
      : p ? [Object.assign({ color: C.accent, sel: true, tip: `${d.name}: x = ${p.x.toFixed(3)}, y = ${p.y.toFixed(3)}${p.fit ? " (narrow lines of the fit)" : ""}` }, p)] : [];
    plotScatter($("cd-" + key), Object.assign({}, d.diagrams[key], { xlabel: xl, ylabel: yl, points: pts,
      onClick: many ? (k) => { ck = k; loadClassify(true); } : null,
      empty: pts.length ? null : "ratio not measured" }));
  }
  $("cmeas").innerHTML = '<thead><tr><th>Measurement</th><th class="num">Value</th></tr></thead><tbody>' +
    d.rows.map((r) => `<tr><td class="mono">${esc(r.name)}</td><td class="num">${esc(r.value)}</td></tr>`).join("") + "</tbody>";
  showClassFig();
  void force;
}
function drawClassPlots() {
  const d = classData;
  if (d.starlight) {
    const L = [{ y: d.starlight.flux, color: C.data, width: 1, range: true, name: "data" },
      { y: d.starlight.model, color: C.host, width: 1.5, name: "stars + power law" }].map((l) => Object.assign(l, { hidden: hidC.star.has(l.name) }));
    $("cstart").textContent = " · " + d.starlight.title;
    plotSpectrum($("cstar"), { x: d.starlight.x, layers: L, zero: true, bands: hidC.star.has("masked lines") ? [] : d.starlight.masks.map((m) => ({ lo: m[0], hi: m[1], tint: "rgba(58,155,213,.16)" })) });
    legend($("cstarleg"), L.concat([{ name: "masked lines", color: "rgba(58,155,213,.5)" }]), hidC.star, drawClassPlots);
  } else $("cstar").innerHTML = '<div class="empty">No stellar fit.</div>';
  if (d.halpha) {
    const L = [{ y: d.halpha.flux, color: C.data, width: 1, range: true, name: "data − stars" },
      { y: d.halpha.narrow, color: C.broad[0], width: 1.5, name: "narrow only" }];
    if (d.halpha.broad) L.push({ y: d.halpha.broad, color: C.model, width: 1.5, name: "+ broad" });
    L.forEach((l) => (l.hidden = hidC.ha.has(l.name)));
    $("chat").textContent = " · " + d.halpha.title;
    plotSpectrum($("cha"), { x: d.halpha.x, layers: L, zero: true });
    legend($("chaleg"), L, hidC.ha, drawClassPlots);
  } else $("cha").innerHTML = '<div class="empty">Hα is outside the spectrum.</div>';
  if (d.lines_fit) drawFit($("clines"), $("clinesleg"), d.lines_fit.plot, hidC.lines, drawClassPlots);
  else $("clines").innerHTML = '<div class="empty">No emission-line fit.</div>';
}
function showClassFig() {
  document.querySelectorAll("#cfigkind button").forEach((b) => b.classList.toggle("on", b.dataset.k === cfigKind));
  const url = withToken(`/api/wf/classify.png?k=${classData.k}&kind=${cfigKind}&v=${S.cells.length}`);
  $("cfig").src = url; $("cfigdl").href = url;
  $("cfigdl").download = `${classData.name}_${cfigKind === "steps" ? "classification" : "diagrams"}.png`;
  $("cfigpdf").onclick = () => download(`/api/wf/classify.pdf?k=${classData.k}&kind=${cfigKind}&dl=1`);
}
// ----------------------------------------------------------------- sample
function renderSampleResults() {
  const sm = S.sample;
  if (!sm) return;
  $("sstats").innerHTML = `<div><small>spectra</small><b>${sm.n}</b></div><div><small>failed</small><b>${sm.failed}</b></div>` +
    `<div style="grid-column: span 2"><small>classes</small><b style="font-size:14px">${Object.entries(sm.classes).map(([k, v]) => `${esc(k)} ${v}`).join(" · ")}</b></div>`;
  const sig = String(sm.id) + ":" + sk;
  if ($("stbl").dataset.sig !== sig) {
    $("stbl").dataset.sig = sig;
    $("stbl").innerHTML = `<thead><tr><th>#</th>${sm.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>` +
      sm.rows.map((r, i) => `<tr data-k="${i}" class="${i === sk ? "sel" : ""}"><td class="mono">${i}</td>${r.map((v) => `<td class="mono">${esc(v)}</td>`).join("")}</tr>`).join("") + "</tbody>";
    $("stbl").querySelectorAll("tr[data-k]").forEach((tr) => (tr.onclick = () => selectSample(+tr.dataset.k)));
  }
  const P = samplePts;
  if (P && P.nii) {
    const pts = [];
    P.nii.forEach((x, i) => { const y = P.oiii[i]; if (x !== null && y !== null) pts.push({ x, y, k: i, sel: i === sk, color: CLASS_COLOR[(P.cls || [])[i]] || "#6b7280", tip: `#${i} ${(P.names || [])[i] || ""}: ${(P.cls || [])[i] || ""}` }); });
    plotScatter($("sbpt"), Object.assign({}, P.nii_curves, { points: pts, xlabel: "log [N II]/Hα", ylabel: "log [O III]/Hβ", onClick: selectSample,
      empty: pts.length ? null : "no narrow-line ratios in this sample" }));
  }
}
function selectSample(k) {
  sk = k;
  const tr = $("stbl").querySelector(`tr[data-k="${k}"]`);
  if (tr) tr.scrollIntoView({ block: "nearest" });
  loadResult(true);
}
function initResults() {
  document.querySelectorAll("#rtabs button").forEach((b) => (b.onclick = () => { rtab = b.dataset.rt; renderResultTable(); }));
  $("hideund").onchange = renderResultTable;
  $("refine").onclick = () => go("model");
  document.querySelectorAll("#rnav button").forEach((b) => (b.onclick = () => { rv = b.dataset.v; loadResult(); }));
  document.querySelectorAll("#cfigkind button").forEach((b) => (b.onclick = () => { cfigKind = b.dataset.k; if (classData) showClassFig(); }));
  $("dlbtn").onclick = (e) => { e.stopPropagation(); $("dlmenu").classList.toggle("hidden"); };
  $("dlmenu").onclick = (e) => {
    const a = e.target.closest("a");
    if (!a) return;
    e.preventDefault();
    if (a.id === "dl-comp") download(`/api/wf/components.csv?k=${rk}`);     // the others: the page's [data-dl] handler
    $("dlmenu").classList.add("hidden");
  };
  document.addEventListener("click", (e) => { if (!e.target.closest("#dlwrap")) $("dlmenu").classList.add("hidden"); });
  $("figopts").onsubmit = (e) => e.preventDefault();
  $("figopts").querySelectorAll("[data-o]").forEach((el) => (el[el.type === "checkbox" || el.tagName === "SELECT" ? "onchange" : "oninput"] = figChanged));
  $("figopts").querySelectorAll("[data-fmt]").forEach((b) => (b.onclick = () => download(figUrl(b.dataset.fmt, true))));
  $("figreset").onclick = () => { $("figopts").reset(); showFigure(); };
  $("lineopts").onsubmit = (e) => e.preventDefault();
  $("lineopts").querySelectorAll("[data-l]").forEach((el) => (el.onchange = linesChanged));
  $("lineopts").querySelectorAll("[data-lfmt]").forEach((b) => (b.onclick = () => lineSel.length && download(linesUrl(b.dataset.lfmt, true))));
  $("cornerpng").onclick = () => download(cornerUrl("png", true));
  $("cornerpdf").onclick = () => download(cornerUrl("pdf", true));
}

// scatter diagram (BPT, WHAN): curves, region names and points with error bars; points can be clicked
function plotScatter(el, cfg) {
  const W = el.clientWidth || 300, H = el.clientHeight || 260, m = { l: 44, r: 10, t: 10, b: 36 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  // the diagram's own range, widened to show every point with its error bars
  let [x0, x1] = cfg.xlim, [y0, y1] = cfg.ylim;
  for (const p of cfg.points || []) {
    const xe = p.xerr || 0, ye = p.yerr || 0;
    if (!isFinite(p.x) || !isFinite(p.y)) continue;
    x0 = Math.min(x0, p.x - xe); x1 = Math.max(x1, p.x + xe); y0 = Math.min(y0, p.y - ye); y1 = Math.max(y1, p.y + ye);
  }
  if (x0 < cfg.xlim[0]) x0 -= 0.05 * (x1 - x0);
  if (x1 > cfg.xlim[1]) x1 += 0.05 * (x1 - x0);
  if (y0 < cfg.ylim[0]) y0 -= 0.05 * (y1 - y0);
  if (y1 > cfg.ylim[1]) y1 += 0.05 * (y1 - y0);
  const X = (v) => m.l + ((v - x0) / (x1 - x0)) * pw, Y = (v) => m.t + ph - ((v - y0) / (y1 - y0)) * ph;
  const uid = el.id;
  let s = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}"><defs><clipPath id="c-${uid}"><rect x="${m.l}" y="${m.t}" width="${pw}" height="${ph}"/></clipPath></defs>`;
  const xt = niceTicks(x0, x1, Math.max(3, Math.floor(pw / 70))), yt = niceTicks(y0, y1, Math.max(3, Math.floor(ph / 45)));
  s += `<g clip-path="url(#c-${uid})">`;
  for (const c of cfg.curves) s += `<path d="${c.x.map((v, i) => (i ? "L" : "M") + X(v).toFixed(1) + " " + Y(c.y[i]).toFixed(1)).join("")}" fill="none" stroke="#52514e" stroke-width="1" ${c.dash ? `stroke-dasharray="${c.dash}"` : ""}/>`;
  for (const t of cfg.texts || []) s += `<text x="${X(t.x)}" y="${Y(t.y)}" text-anchor="${X(t.x) > m.l + 0.7 * pw ? "end" : "start"}" style="font:11px var(--sans);fill:#898781">${esc(t.t)}</text>`;
  const pts = (cfg.points || []).slice().sort((a, b) => (a.sel ? 1 : 0) - (b.sel ? 1 : 0));
  for (const p of pts) {
    const px = X(p.x), py = Y(p.y), col = p.color || C.accent;
    if (p.xerr) s += `<line x1="${X(p.x - p.xerr)}" x2="${X(p.x + p.xerr)}" y1="${py}" y2="${py}" stroke="${col}" stroke-width="1.2"/>`;
    if (p.yerr) s += `<line x1="${px}" x2="${px}" y1="${Y(p.y - p.yerr)}" y2="${Y(p.y + p.yerr)}" stroke="${col}" stroke-width="1.2"/>`;
    s += `<circle class="pt" data-k="${p.k ?? ""}" data-tip="${esc(p.tip || "")}" cx="${px}" cy="${py}" r="${p.sel ? 6 : 4}" fill="${col}" fill-opacity="${p.sel ? 1 : 0.75}" stroke="${p.sel ? "#0b0b0b" : "#fff"}" stroke-width="${p.sel ? 1.5 : 1}"/>`;
  }
  s += "</g>";
  s += `<g class="axis"><rect x="${m.l}" y="${m.t}" width="${pw}" height="${ph}"/>` +
    yt.map((v) => `<line x1="${m.l}" x2="${m.l + 5}" y1="${Y(v)}" y2="${Y(v)}"/><line x1="${m.l + pw - 5}" x2="${m.l + pw}" y1="${Y(v)}" y2="${Y(v)}"/>` +
      `<text x="${m.l - 5}" y="${Y(v) + 3.5}" text-anchor="end">${+v.toFixed(2)}</text>`).join("") +
    xt.map((v) => `<line x1="${X(v)}" x2="${X(v)}" y1="${m.t}" y2="${m.t + 5}"/><line x1="${X(v)}" x2="${X(v)}" y1="${m.t + ph - 5}" y2="${m.t + ph}"/>` +
      `<text x="${X(v)}" y="${m.t + ph + 14}" text-anchor="middle">${+v.toFixed(2)}</text>`).join("") +
    `<text x="${m.l + pw / 2}" y="${H - 3}" text-anchor="middle">${esc(cfg.xlabel || "")}</text>` +
    `<text transform="translate(11 ${m.t + ph / 2}) rotate(-90)" text-anchor="middle">${esc(cfg.ylabel || "")}</text></g>`;
  if (cfg.empty) s += `<text x="${m.l + pw / 2}" y="${m.t + ph / 2}" text-anchor="middle" style="font:12px var(--sans);fill:#7b8290">${esc(cfg.empty)}</text>`;
  s += "</svg><div class=\"tip hidden\"></div>";
  el.innerHTML = s;
  plots.set(el, { scatter: cfg });
  const tip = el.querySelector(".tip");
  el.querySelectorAll("circle.pt").forEach((c) => {
    c.onmouseenter = () => { tip.textContent = c.dataset.tip; tip.classList.remove("hidden"); tip.style.left = Math.min(+c.getAttribute("cx") + 10, W - tip.offsetWidth - 4) + "px"; tip.style.top = Math.max(0, +c.getAttribute("cy") - 28) + "px"; };
    c.onmouseleave = () => tip.classList.add("hidden");
    if (cfg.onClick && c.dataset.k !== "") c.onclick = () => cfg.onClick(+c.dataset.k);
  });
}

// ================================================================== plotting
function niceTicks(lo, hi, n) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((f) => f * mag).find((s) => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * span; v += step) out.push(+v.toPrecision(12));
  return out;
}
function quant(arr, q) {
  const a = arr.filter((v) => v !== null && isFinite(v)).sort((x, y) => x - y);
  if (!a.length) return 0;
  return a[Math.min(a.length - 1, Math.max(0, Math.floor(q * (a.length - 1))))];
}
function lowerIndex(x, v) { let lo = 0, hi = x.length - 1; while (lo < hi) { const m = (lo + hi) >> 1; if (x[m] < v) lo = m + 1; else hi = m; } return lo; }
const plots = new Map();
function plotSpectrum(el, cfg) {
  const prev = plots.get(el);
  const sameX = prev && prev.cfg && prev.cfg.x.length === cfg.x.length && prev.cfg.x[0] === cfg.x[0] && prev.cfg.x[cfg.x.length - 1] === cfg.x[cfg.x.length - 1];
  const st = { cfg, view: sameX && prev.cfg ? prev.view : null };
  plots.set(el, st);
  drawPlotEl(el);
}
function drawPlotEl(el) {
  const st = plots.get(el); if (!st) return;
  if (st.scatter) return plotScatter(el, st.scatter);
  const cfg = st.cfg, x = cfg.x;
  const W = el.clientWidth || 600, H = el.clientHeight || 340;
  const rh = cfg.resid ? 60 : 0;
  const m = { l: 56, r: 12, t: cfg.labels ? 24 : 12, b: 24 };
  const ph = H - m.t - m.b - (rh ? rh + 10 : 0), pw = W - m.l - m.r;
  if (!x.length) { el.innerHTML = '<div class="empty">No data</div>'; return; }
  const [x0, x1] = st.view || [x[0], x[x.length - 1]];
  const i0 = Math.max(0, lowerIndex(x, x0) - 1), i1 = Math.min(x.length - 1, lowerIndex(x, x1) + 1);
  // y range: the "range" layers (data, model) in view, ignoring masked pixels
  const masked = (v) => (cfg.hatch || []).some((h) => v >= h[0] && v <= h[1]);
  let vals = [];
  for (const L of cfg.layers) if (L.range && !L.hidden) for (let i = i0; i <= i1; i++) if (L.y[i] !== null && !masked(x[i])) vals.push(L.y[i]);
  let y0 = quant(vals, 0.002), y1 = quant(vals, 1);
  if (cfg.zero) y0 = Math.min(y0, 0);
  const pad = (y1 - y0) || 1;
  y1 += pad * (cfg.labels ? 0.14 : 0.06); y0 -= pad * 0.04;
  const X = (v) => m.l + ((v - x0) / (x1 - x0)) * pw, Y = (v) => m.t + ph - ((v - y0) / (y1 - y0)) * ph;
  const uid = el.id || "p";
  let s = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}"><defs><pattern id="h-${uid}" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="#f4f3ef"/><line x1="0" y1="0" x2="0" y2="6" stroke="rgba(137,135,129,.35)" stroke-width="1.5"/></pattern>` +
    `<clipPath id="c-${uid}"><rect x="${m.l}" y="${m.t - 2}" width="${pw}" height="${ph + 4}"/></clipPath><clipPath id="cr-${uid}"><rect x="${m.l}" y="${m.t + ph + 10}" width="${pw}" height="${rh}"/></clipPath></defs>`;
  const xt = niceTicks(x0, x1, Math.max(3, Math.floor(pw / 110))), yt = niceTicks(y0, y1, Math.max(3, Math.floor(ph / 60)));
  s += `<g clip-path="url(#c-${uid})">`;
  for (const h of cfg.hatch || []) if (h[1] >= x0 && h[0] <= x1) s += `<rect x="${X(h[0])}" y="${m.t}" width="${Math.max(2, X(h[1]) - X(h[0]))}" height="${ph}" fill="url(#h-${uid})"/>`;
  for (const b of cfg.bands || []) if (b.hi >= x0 && b.lo <= x1) s += `<rect x="${X(b.lo)}" y="${m.t}" width="${Math.max(2, X(b.hi) - X(b.lo))}" height="${ph}" fill="${b.tint || `url(#h-${uid})`}" opacity="${b.tint ? 1 : 0.8}"/>`;
  if (cfg.zero && y0 < 0) s += `<line x1="${m.l}" x2="${m.l + pw}" y1="${Y(0)}" y2="${Y(0)}" stroke="#898781" stroke-width=".7"/>`;
  const base = Y(Math.max(0, y0));
  for (const L of cfg.layers) {
    if (L.hidden || !L.fill) continue;        // light fill under a layer (the Fe II pseudo-continuum), below all lines
    let d = "", run = [];
    const close = () => { if (run.length > 1) d += `M${run[0][0]} ${base}` + run.map((p) => `L${p[0]} ${p[1]}`).join("") + `L${run[run.length - 1][0]} ${base}Z`; run = []; };
    for (let i = i0; i <= i1; i++) {
      const v = L.y[i];
      if (v === null || v === undefined) { close(); continue; }
      run.push([X(x[i]).toFixed(1), Y(v).toFixed(1)]);
    }
    close();
    s += `<path d="${d}" fill="${L.color}" fill-opacity=".13" stroke="none"/>`;
  }
  for (const L of cfg.layers) {
    if (L.hidden) continue;
    let d = "", pen = false;
    for (let i = i0; i <= i1; i++) {
      const v = L.y[i];
      if (v === null || v === undefined) { pen = false; continue; }
      d += (pen ? "L" : "M") + X(x[i]).toFixed(1) + " " + Y(v).toFixed(1);
      pen = true;
    }
    s += `<path d="${d}" fill="none" stroke="${L.color}" stroke-width="${L.width || 1.2}" ${L.dash ? `stroke-dasharray="${L.dash}"` : ""} opacity="${L.opacity ?? 1}" stroke-linejoin="round"/>`;
  }
  s += "</g>";
  for (const b of cfg.bands || []) if (b.label && b.hi >= x0 && b.lo <= x1) s += `<text x="${(X(b.lo) + X(b.hi)) / 2}" y="${m.t - 2}" text-anchor="middle" class="plab" style="fill:#52514e;font-size:10px">${esc(b.label)}</text>`;
  // line labels: strongest first, above the model, without overlaps
  if (cfg.labels) {
    const model = cfg.layers.find((L) => L.name === "model");
    const placed = [];
    for (const lab of cfg.labels) {
      if (lab.x < x0 || lab.x > x1 || !model) continue;
      const px = X(lab.x);
      let top = Infinity;
      const j0 = lowerIndex(x, lab.x - 12), j1 = lowerIndex(x, lab.x + 12);
      for (let j = j0; j <= j1; j++) if (model.y[j] !== null) top = Math.min(top, Y(model.y[j]));
      if (!isFinite(top)) continue;
      const w = lab.text.length * 6.2;
      let py = Math.max(m.t + 8, top - 8);
      const hits = (y) => placed.some((p) => Math.abs(p.x - px) < (p.w + w) / 2 + 4 && Math.abs(p.y - y) < 14);
      if (hits(py)) py -= 14;                       // one step up, else leave it out
      if (hits(py) || py < m.t - 6) continue;
      placed.push({ x: px, y: py, w });
      const lx = Math.min(Math.max(px, m.l + w / 2), m.l + pw - w / 2);     // keep the label inside the plot
      s += `<text x="${lx}" y="${py}" text-anchor="middle" class="plab">${esc(lab.text)}</text>`;
    }
  }
  // frame with inward ticks on all four sides (journal style, no grid)
  const xa = m.t + ph + (rh ? rh + 10 : 0);
  const xtick = (top, bottom) => xt.map((v) => `<line x1="${X(v)}" x2="${X(v)}" y1="${top}" y2="${top + 5}"/><line x1="${X(v)}" x2="${X(v)}" y1="${bottom - 5}" y2="${bottom}"/>`).join("");
  s += '<g class="axis">' + `<rect x="${m.l}" y="${m.t}" width="${pw}" height="${ph}"/>` +
    yt.map((v) => `<line x1="${m.l}" x2="${m.l + 5}" y1="${Y(v)}" y2="${Y(v)}"/><line x1="${m.l + pw - 5}" x2="${m.l + pw}" y1="${Y(v)}" y2="${Y(v)}"/>` +
      `<text x="${m.l - 6}" y="${Y(v) + 3.5}" text-anchor="end">${fmtNum(v, 3)}</text>`).join("") + xtick(m.t, m.t + ph);
  if (rh) s += `<rect x="${m.l}" y="${m.t + ph + 10}" width="${pw}" height="${rh}"/>` + xtick(m.t + ph + 10, xa);
  s += xt.map((v) => `<text x="${X(v)}" y="${xa + 15}" text-anchor="middle">${v}</text>`).join("") + "</g>";
  if (rh) {
    const r0 = m.t + ph + 10, R = (v) => r0 + rh / 2 - (Math.max(-5, Math.min(5, v)) / 5) * (rh / 2);
    let d = "", pen = false;
    for (let i = i0; i <= i1; i++) { const v = cfg.resid[i]; if (v === null) { pen = false; continue; } d += (pen ? "L" : "M") + X(x[i]).toFixed(1) + " " + R(v).toFixed(1); pen = true; }
    s += `<g clip-path="url(#cr-${uid})"><rect x="${m.l}" y="${R(1)}" width="${pw}" height="${R(-1) - R(1)}" fill="#e6e5e1"/>` +
      `<line x1="${m.l}" x2="${m.l + pw}" y1="${R(3)}" y2="${R(3)}" stroke="#e6e5e1" stroke-dasharray="3 3"/><line x1="${m.l}" x2="${m.l + pw}" y1="${R(-3)}" y2="${R(-3)}" stroke="#e6e5e1" stroke-dasharray="3 3"/>` +
      `<line x1="${m.l}" x2="${m.l + pw}" y1="${R(0)}" y2="${R(0)}" stroke="#0b0b0b" stroke-width=".8"/><path d="${d}" fill="none" stroke="#52514e" stroke-width=".8"/></g>`;
    s += `<g class="axis"><text x="${m.l - 6}" y="${r0 + rh / 2 + 3.5}" text-anchor="end">res</text></g>`;
  }
  s += `<line class="cross" x1="0" x2="0" y1="${m.t}" y2="${xa}" stroke="#1d2128" stroke-opacity=".25" visibility="hidden"/>`;
  s += `<rect class="selrect" x="0" y="${m.t}" width="0" height="${ph}" fill="${cfg.onSelect ? "rgba(212,71,42,.18)" : "rgba(31,111,120,.12)"}" visibility="hidden"/>`;
  s += `<rect class="hit" x="${m.l}" y="${m.t}" width="${pw}" height="${xa - m.t}" fill="transparent"/></svg><div class="tip hidden"></div>`;
  el.innerHTML = s;
  const svg = el.querySelector("svg"), hit = svg.querySelector(".hit"), cross = svg.querySelector(".cross"), sel = svg.querySelector(".selrect"), tip = el.querySelector(".tip");
  const inv = (px) => x0 + ((px - m.l) / pw) * (x1 - x0);
  let drag = null;
  const pos = (e) => { const r = svg.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  hit.onmousemove = (e) => {
    const [px, py] = pos(e);
    cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.setAttribute("visibility", "visible");
    if (drag) { const a = Math.min(drag, px), b = Math.max(drag, px); sel.setAttribute("x", a); sel.setAttribute("width", b - a); sel.setAttribute("visibility", "visible"); }
    const wv = inv(px), i = Math.min(x.length - 1, lowerIndex(x, wv));
    const parts = [`λ ${x[i].toFixed(1)}`];
    for (const L of cfg.layers) if (!L.hidden && (L.name === "data" || L.name === "model" || L.range) && L.y[i] !== null) parts.push(`${L.name || "flux"} ${fmtNum(L.y[i], 4)}`);
    if (cfg.resid && cfg.resid[i] !== null) parts.push(`res ${cfg.resid[i].toFixed(2)}σ`);
    tip.textContent = parts.join("  ·  ");
    tip.classList.remove("hidden");
    tip.style.left = Math.min(px + 12, W - tip.offsetWidth - 4) + "px"; tip.style.top = Math.max(0, py - 30) + "px";
  };
  hit.onmouseleave = () => { cross.setAttribute("visibility", "hidden"); tip.classList.add("hidden"); };
  hit.onmousedown = (e) => { drag = pos(e)[0]; e.preventDefault(); };
  window.addEventListener("mouseup", function up(e) {
    window.removeEventListener("mouseup", up);
    if (drag === null) return;
    const px = Math.max(m.l, Math.min(m.l + pw, pos(e)[0])), a = Math.min(drag, px), b = Math.max(drag, px);
    drag = null; sel.setAttribute("visibility", "hidden");
    if (b - a < 4) return;
    if (cfg.onSelect) cfg.onSelect(Math.round(inv(a)), Math.round(inv(b)));
    else { st.view = [inv(a), inv(b)]; drawPlotEl(el); }
  });
  hit.ondblclick = () => { st.view = null; drawPlotEl(el); };
}
const resizeObs = new ResizeObserver((entries) => entries.forEach((en) => plots.has(en.target) && drawPlotEl(en.target)));
["pplot", "gplot", "rplot", "cstar", "cha", "clines", "cd-nii", "cd-sii", "cd-oi", "cd-whan", "cbptall", "sbpt"].forEach((id) => resizeObs.observe($(id)));

// ============================================================= session panel
function setSession(on) {
  $("session").classList.toggle("hidden", !on);
  $("pychip").classList.toggle("on", on);
  try { localStorage.setItem("fantasy.session", on ? "1" : "0"); } catch (e) { /* private mode */ }
  if (on) { cellSig = null; renderCells(); const c = $("cells"); c.scrollTop = c.scrollHeight; }
  setTimeout(() => plots.forEach((_, el) => drawPlotEl(el)), 30);
}
const PY_RE = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')|\b(\d+\.?\d*(?:e[+-]?\d+)?)\b|\b(import|from|as|for|in|if|elif|else|def|return|lambda|with|not|and|or|is|None|True|False|while|try|except|class|print)\b|(\b[A-Za-z_]\w*(?=\())/g;
function highlight(code) {
  let out = "", last = 0;
  code.replace(PY_RE, (mm, cm, st, nu, kw, fn, off) => {
    out += esc(code.slice(last, off));
    const cls = cm ? "cm" : st ? "st" : nu ? "nu" : kw ? (["None", "True", "False"].includes(kw) ? "nu" : "kw") : "fn";
    out += `<span class="${cls}">${esc(mm)}</span>`;
    last = off + mm.length;
    return mm;
  });
  return out + esc(code.slice(last));
}
let cellSig = null, editIndex = null;
function renderCells() {
  if ($("session").classList.contains("hidden") || !S) return;
  const sig = JSON.stringify(S.cells.map((c) => [c.n, c.running, c.out.length, c.out.length ? c.out[c.out.length - 1].text.length : 0, c.code.length])) + editIndex;
  if (sig === cellSig) return;
  cellSig = sig;
  const box = $("cells"), atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  box.innerHTML = S.cells.map((c, i) => {
    const n = c.running ? "*" : c.n ?? " ";
    const last = i === S.cells.length - 1;
    let h = `<div class="cell ${last ? "last" : ""} ${c.running ? "running" : ""} ${editIndex === i ? "editing" : ""}" data-i="${i}" title="Click to edit and run again"><span class="p">In [${n}]</span><pre>${highlight(c.code)}</pre></div>`;
    for (const o of c.out) h += `<div class="cell out ${o.kind === "err" ? "err" : ""}"><span class="p">Out[${n}]</span><pre>${esc(o.text)}</pre></div>`;
    return h;
  }).join("");
  box.querySelectorAll(".cell[data-i]").forEach((el) => (el.onclick = () => {
    const i = +el.dataset.i;
    if (i === 0) return;
    editIndex = i; $("codein").value = S.cells[i].code; $("inprompt").textContent = `In [${S.cells[i].n ?? " "}] ✎`;
    autosize(); $("codein").focus(); cellSig = null; renderCells();
  }));
  if (atBottom) box.scrollTop = box.scrollHeight;
}
function autosize() { const t = $("codein"); t.style.height = "auto"; t.style.height = Math.min(220, t.scrollHeight + 2) + "px"; }
async function runInput() {
  const code = $("codein").value;
  if (!code.trim() || S.busy) return;
  const body = { code };
  if (editIndex !== null) body.index = editIndex;
  if (await post("/api/exec", body)) { $("codein").value = ""; editIndex = null; $("inprompt").textContent = "In [ ]"; autosize(); }
}
function initSession() {
  $("pychip").onclick = () => setSession($("session").classList.contains("hidden"));
  $("closesession").onclick = () => setSession(false);
  $("codein").oninput = autosize;
  $("codein").onkeydown = (e) => {
    if (e.key === "Enter" && e.shiftKey) { e.preventDefault(); runInput(); }
    if (e.key === "Escape") { editIndex = null; $("codein").value = ""; $("inprompt").textContent = "In [ ]"; autosize(); cellSig = null; renderCells(); }
  };
  $("copyscript").onclick = async () => {
    try { await navigator.clipboard.writeText(await api("/api/session.py")); toast("Session copied as a Python script"); }
    catch (e) { showError("Could not copy: " + e.message); }
  };
  document.addEventListener("keydown", (e) => {
    if (e.key === "`" && e.ctrlKey) { e.preventDefault(); setSession($("session").classList.contains("hidden")); }
    if (e.key === "Escape" && masking) setMasking(false);
    if (e.key === "Escape" && !$("docs").classList.contains("hidden")) $("docs").classList.add("hidden");
  });
  let saved = null;
  try { saved = localStorage.getItem("fantasy.session"); } catch (e) { /* private mode */ }
  setSession(saved === "1" || new URLSearchParams(location.search).has("session"));
}

// ===================================================================== docs
let docs = null;
async function openDocs() {
  $("docs").classList.remove("hidden");
  if (!docs) docs = await api("/api/docs");
  renderDocList();
  if (!$("docbody").innerHTML) showDoc(docs[0].id);
}
function renderDocList() {
  if (!docs) return;                       // typed before the pages arrived: openDocs renders them
  const q = $("docsearch").value.trim().toLowerCase();
  let sec = null, html = "";
  for (const d of docs) {
    if (q && !(d.title + " " + d.html).toLowerCase().includes(q)) continue;
    if (d.section !== sec) { sec = d.section; html += `<li class="sec">${esc(sec)}</li>`; }
    html += `<li data-id="${esc(d.id)}">${esc(d.title)}</li>`;
  }
  $("doclist").innerHTML = html;
  $("doclist").querySelectorAll("li[data-id]").forEach((li) => (li.onclick = () => showDoc(li.dataset.id)));
}
function showDoc(id) {
  const d = docs.find((x) => x.id === id);
  if (!d) return;
  $("docbody").innerHTML = `<h1>${esc(d.title)}</h1>` + d.html;
  $("docbody").scrollTop = 0;
  $("doclist").querySelectorAll("li[data-id]").forEach((li) => li.classList.toggle("on", li.dataset.id === id));
}

// ===================================================================== init
function initTop() {
  document.addEventListener("click", (e) => {
    const g = e.target.closest("[data-go]");
    if (g) { e.preventDefault(); go(g.dataset.go); }
    const d = e.target.closest("[data-dl]");
    if (d) { e.preventDefault(); download(d.dataset.dl); $("moremenu").classList.add("hidden"); }
    if (!e.target.closest(".menuwrap")) $("moremenu").classList.add("hidden");
  });
  $("morebtn").onclick = () => $("moremenu").classList.toggle("hidden");
  $("newbtn").onclick = async () => {
    if (!confirm("Start a new analysis? The current session (code, spectra and fits) is cleared; save it first from the ⋯ menu if you need it.")) return;
    prep = null; modelTried = false; lastRun = null; runLog = []; resultData = null; fitInit = false; rk = 0; pk = 0;
    $("presetsel").innerHTML = "";
    if (await post("/api/reset", {})) go("read");
  };
  document.querySelector('[data-act="open-project"]').onclick = () => $("projectfile").click();
  $("projectfile").onchange = async (e) => {
    const f = e.target.files[0]; if (!f) return;
    try { setState(await api("/api/project", { method: "POST", body: f })); toast("Project opened"); go(guessPage()); }
    catch (err) { showError(err.message); }
  };
  $("docslink").onclick = (e) => { e.preventDefault(); openDocs(); };
  $("closedocs").onclick = () => $("docs").classList.add("hidden");
  $("docs").onclick = (e) => { if (e.target === $("docs")) $("docs").classList.add("hidden"); };
  $("docsearch").oninput = renderDocList;
}
function guessPage() {
  const h = location.hash.slice(1);
  if (STEPS.some(([id]) => id === h)) return h;
  if (fitDone()) return "results";
  if (S.model && prepared()) return "model";
  if (prepared()) return "model";
  if (loaded()) return "prepare";
  return "read";
}
async function main() {
  initTop(); initRead(); initPrepare(); initModel(); initFit(); initResults(); initTime(); initResid(); initSession();
  try { S = await api("/api/state"); } catch (e) { showError("Cannot reach the FANTASY server: " + e.message); return; }
  wasBusy = !!S.busy;
  api("/api/lines").then((ls) => (LINE_LIB = ls)).catch(() => {});
  api("/api/linesets").then((ls) => (LINE_SETS = ls)).catch(() => {});
  go(guessPage());
  schedule();
}
main();
