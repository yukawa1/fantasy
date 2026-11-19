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
  bowen: "#4a3aa7", coronal: "#b8459c", other: ["#898781", "#86b6ef", "#c9c8c2"], accent: "#e34948",
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
// a value and its error as HTML, rounded to two significant digits of the error, with powers of ten written out
function sciPM(v, e) {
  if (v === null || v === undefined || !isFinite(v)) return { v: "—", e: "" };
  const ok = e !== null && e !== undefined && isFinite(e) && e > 0;
  const exp = v !== 0 ? Math.floor(Math.log10(Math.abs(v))) : 0;
  const p = ok ? Math.floor(Math.log10(e)) - 1 : exp - 2;
  const pow = (x) => `<span class="pow">×10<sup>${x}</sup></span>`;
  if (Math.abs(exp) >= 4) {
    const d = Math.max(0, Math.min(6, exp - p)), s = 10 ** exp;
    return { v: (v / s).toFixed(d) + pow(exp), e: ok ? `± ${(e / s).toFixed(d)}${pow(exp)}` : "" };
  }
  const d = Math.max(0, Math.min(6, -p)), r = (x) => (p > 0 ? Math.round(x / 10 ** p) * 10 ** p : x);
  return { v: r(v).toFixed(d), e: ok ? `± ${r(e).toFixed(d)}` : "" };
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
async function post(path, body, quiet) {
  try {
    const st = await api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
    if (requestError) { requestError = false; $("banner").classList.add("hidden"); }   // the action went through
    setState(st);
    return true;
  } catch (e) { lastRequestError = e.message; if (!quiet) showError(e.message, true); return false; }
}
function download(url) {
  const a = document.createElement("a"); a.href = withToken(url); a.download = ""; document.body.appendChild(a); a.click(); a.remove();
}
function initActionMenu(buttonId, menuId, refresh) {
  const trigger = $(buttonId), menu = $(menuId), wrap = trigger.parentElement;
  const choices = () => Array.from(menu.querySelectorAll('[role="menuitem"]:not(:disabled):not(.hidden)'));
  function close(restore) {
    menu.classList.add("hidden");
    trigger.setAttribute("aria-expanded", "false");
    if (restore) trigger.focus();
  }
  function open(last = false) {
    if (refresh) refresh();
    menu.classList.remove("hidden");
    trigger.setAttribute("aria-expanded", "true");
    const items = choices();
    (last ? items[items.length - 1] : items[0])?.focus();
  }
  trigger.onclick = () => menu.classList.contains("hidden") ? open() : close(true);
  trigger.onkeydown = (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault(); open(e.key === "ArrowUp");
    }
  };
  menu.onkeydown = (e) => {
    if (e.key === "Escape") { e.preventDefault(); close(true); return; }
    if (e.key === "Tab") { close(true); return; }  // allow native Tab/Shift+Tab from the trigger
    const items = choices(), i = items.indexOf(document.activeElement);
    const next = { ArrowDown: i + 1, ArrowUp: i - 1, Home: 0, End: items.length - 1 }[e.key];
    if (next !== undefined && items.length) {
      e.preventDefault(); items[(next + items.length) % items.length].focus();
    }
  };
  menu.addEventListener("click", (e) => {
    const item = e.target.closest('[role="menuitem"]');
    if (!item) return;
    if (item.disabled || item.classList.contains("hidden")) { e.stopPropagation(); return; }
    close(true);
  });
  document.addEventListener("click", (e) => { if (!wrap.contains(e.target)) close(false); });
  wrap.addEventListener("focusout", (e) => { if (!wrap.contains(e.relatedTarget)) close(false); });
}
// An error of a request (e.g. no files match the path) stays until dismissed or the next action goes through;
// one of a cell (S.error) goes when the session has none.
let requestError = false, lastRequestError = "";
function showError(msg, fromRequest) {
  requestError = !!fromRequest;
  const b = $("banner");
  b.innerHTML = `<pre>${esc(msg)}</pre><button id="banner-session">Show in session</button><button id="banner-x">Dismiss</button>`;
  b.classList.remove("hidden");
  $("banner-session").onclick = () => setSession(true);
  $("banner-x").onclick = () => { b.classList.add("hidden"); requestError = false; lastError = S && S.error; };
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
  if (window.MB) MB.updateContext();
  const busyNow = !!S.busy;
  if (wasBusy && !busyNow) onIdle(prev);
  wasBusy = busyNow;
  if (S.error && S.error !== lastError) { showError(S.error); lastError = S.error; pendingGo = null; }
  if (!S.error) { lastError = null; if (!requestError) $("banner").classList.add("hidden"); }
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
  $("stopbtn").classList.toggle("hidden", !S.busy);
  $("stopbtn").disabled = !!S.stopping;
  $("stopbtn").textContent = S.stopping ? "Stopping…" : "■ Stop";
  const L = loaded();
  let obj = "";
  if (L && L.rows.length) {
    const names = L.rows.map((r) => r.name);
    obj = names.length === 1 ? names[0] : commonName(names);
  }
  $("objlabel").textContent = obj ? `${obj} · ${MODE_LABEL[mode()]}` : MODE_LABEL[mode()];
  renderStepper();
  renderCells();
  if (page === "home") renderHome();
  if (page === "read") renderRead();
  if (page === "prepare") renderPrepare();
  if (page === "model") renderModel();
  if (page === "fit") renderFit();
  if (page === "results") { if (S.sample && fitDone() && fitDone().kind === "sample") renderSampleResults(); renderResid(); renderLines(); renderRobust(); renderFeii(); renderSim(); renderJointChecks(); renderCompare(); renderChecks(); renderCorr(); if (S.wf.residuals && S.wf.residuals.id !== lastResidId) { lastResidId = S.wf.residuals.id; if (resultData && !resultData.error && rv === "fit") drawResultPlot(); } renderResults(); if (fitDone() && ["joint", "fits"].includes(fitDone().kind) && lcData && !lcData.computing) renderReverb(); }
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
  // rebuilt only when it changes: replacing the buttons on every state update loses clicks made meanwhile
  const html = STEPS.map(([id, label], i) => {
    const done = stepDone(id) && id !== page, cur = id === page;
    return (i ? '<span class="stepsep">›</span>' : "") +
      `<button class="step ${done ? "done" : ""} ${cur ? "cur" : ""}" data-go="${id}">` +
      `<span class="num">${done ? "✓" : i + 1}</span><span><b>${label}</b><small>${esc(stepSub(id))}</small></span></button>`;
  }).join("");
  if (nav.dataset.html !== html) { nav.innerHTML = html; nav.dataset.html = html; }
}
const PAGES = ["home", ...STEPS.map(([id]) => id)];     // Home (the opening screen) comes before the steps
function go(step) {
  if (!PAGES.includes(step)) step = "home";
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
let nextReadMode = null;
function agoText(t) {
  const s = Math.max(0, Date.now() / 1000 - t);
  if (s < 90) return "a minute ago";
  if (s < 3600) return `${Math.round(s / 60)} minutes ago`;
  if (s < 86400 * 1.5) return `${Math.round(s / 3600)} hours ago`;
  return new Date(t * 1000).toLocaleString();
}
function renderRestore() {
  const box = $("restorebox"), A = S.autosaved || {}, last = A.last;
  const show = last && S.cells.length < 2 && !restoreHidden;
  box.classList.toggle("hidden", !show);
  if (!show) return;
  const sig = JSON.stringify(A);
  if (box.dataset.sig === sig) return;
  box.dataset.sig = sig;
  const what = (m) => [m.spectrum ? `<b>${esc(m.spectrum)}</b>` : "a session",
    m.epochs > 1 ? `${m.epochs} epochs` : null, `${m.cells} ${m.cells === 1 ? "step" : "steps"}`,
    m.fitted ? "fitted" : "not fitted", agoText(m.saved)].filter(Boolean).join(" · ");
  box.innerHTML = `<span>Last session: ${what(last)}</span><span class="grow"></span>` +
    (A.previous ? `<button class="btn small" data-restore="previous" title="${esc("The one before: " + what(A.previous).replace(/<[^>]+>/g, ""))}">Open the one before</button>` : "") +
    `<button class="btn small primary" data-restore="last">Restore last session</button>` +
    `<button class="btn small" id="restorehide" title="Start a new analysis; the saved session stays on disk until a new one replaces it">Dismiss</button>`;
  box.querySelectorAll("[data-restore]").forEach((b) => (b.onclick = async () => {
    b.disabled = true;
    const st = await post("/api/restore", { which: b.dataset.restore });
    if (st) { toast("Session restored"); go(fitDone() ? "results" : S.model && prepared() ? "model" : prepared() ? "model" : loaded() ? "prepare" : "read"); }
  }));
  $("restorehide").onclick = () => { restoreHidden = true; renderRestore(); };
}
let restoreHidden = false;
// Home: the opening screen (hero.js), the way in, and the last session to restore
// "Try it on this galaxy" reads the bundled Seyfert 1 as a single spectrum, on the Read page
window.fantasyTryExample = (path) => {
  if (!path || path.startsWith("__")) return;
  go("read");
  readSpectra({ pattern: path, mode: "single" });
};
function renderHome() {
  renderRestore();
  if (window.fantasyHero) window.fantasyHero.show(true);
  // with an analysis under way, the first button returns to it
  const L = loaded();
  $("herocontinue").hidden = !L;
  $("herostart").hidden = !!L;
  if (L) $("herocontinue").dataset.go = fitDone() ? "results" : prepared() ? "model" : "prepare";
}
function renderRead() {
  if (nextReadMode === mode() || !loaded()) nextReadMode = null;
  const m = nextReadMode || mode();
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
  $("toprepare").disabled = !L || !!S.busy || !!nextReadMode;
  if (!L) { $("toprepare").textContent = "Prepare →"; return; }
  const n = L.n, unit = multi ? (n === 1 ? "epoch" : "epochs") : (n === 1 ? "spectrum" : "spectra");
  $("loadedtitle").textContent = `Loaded · ${n} ${unit}` + (L.shown < n ? ` (${L.shown} shown)` : "");
  $("toprepare").textContent = `Prepare ${n === 1 ? "spectrum" : n + " spectra"} →`;
  const sig = JSON.stringify([L.id, L.rows.length, multi]);
  if ($("loadedtbl").dataset.sig !== sig) {
    $("loadedtbl").dataset.sig = sig;
    $("loadedtbl").innerHTML = `<thead><tr><th><input type="checkbox" id="selall"></th><th>${multi ? "Epoch" : "#"}</th><th>Date</th><th>File</th><th>Format</th><th class="num">z</th><th>RA / Dec</th></tr></thead><tbody>` +
      L.rows.map((r) => `<tr data-row="${r.k}"><td><input type="checkbox" data-k="${r.k}"></td><td class="mono">#${r.k}</td><td class="mono">${esc(r.date || "—")}</td>` +
        `<td class="mono">${esc(r.file)}</td><td class="mono muted">${esc(r.format)}</td><td class="num">${r.z != null ? fmtNum(r.z, 5) : "—"}</td>` +
        `<td>${r.ra != null ? `<span class="mono">${fmtNum(r.ra, 7)}, ${fmtNum(r.dec, 7)}</span>` : '<span class="badge missing">missing</span>'}</td></tr>`).join("") + "</tbody>";
    $("selall").onchange = (e) => { document.querySelectorAll('#loadedtbl input[data-k]').forEach((c) => (c.checked = e.target.checked)); updateRemove(); };
    document.querySelectorAll('#loadedtbl input[data-k]').forEach((c) => (c.onchange = updateRemove));
    document.querySelectorAll("#loadedtbl tbody tr").forEach((tr) => (tr.onclick = (e) => {
      const k = S.wf.catalog?.loaded_rows?.[+tr.dataset.row];
      if (e.target.tagName !== "INPUT" && k != null) catalogPick(k);
    }));
    updateRemove();
  }
  const noCoords = L.rows.some((r) => r.ra == null), noZ = L.rows.some((r) => r.z == null);
  const notes = [];
  if (nextReadMode) notes.push(nextReadMode === "single" ? "Read one spectrum to start a single-spectrum analysis." :
    "Read at least two spectra to start a multi-epoch analysis.");
  if (noCoords) notes.push("Some files carry no coordinates: give RA/Dec or E(B−V) in the next step.");
  if (noZ) notes.push("No redshift in the files: enter z in the next step.");
  $("loadednote").textContent = notes.join(" ");
  updateRemove();
  renderCatalogCard();
}
function updateRemove() {
  const any = document.querySelectorAll('#loadedtbl input[data-k]:checked').length > 0;
  $("removesel").disabled = !any || !!S.busy || !!nextReadMode;
}
async function readSpectra(src) {
  const remote = !!(src.targetid || src.sdss);
  const body = Object.assign({ mode: nextReadMode || mode(), group: remote ? "auto" : $("groupsel").value,
    format: remote ? "auto" : $("fmtsel").value }, src);
  const box = $("readerr");
  box.classList.add("hidden");
  if (await post("/api/wf/read", body, true)) toast("Reading…", 1200);
  else {                                     // in the open source tab, under the box it was typed in
    const pane = document.querySelector(".srcpane:not(.hidden)");
    if (pane) pane.appendChild(box);
    box.textContent = lastRequestError;
    box.classList.remove("hidden");
    box.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
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
    if (S.busy) return;
    nextReadMode = null;
    if (m === mode()) { renderRead(); return; }
    const L = loaded();
    if (L && ((m === "single" && L.n !== 1) || (m === "multi" && L.n < 2))) {
      nextReadMode = m;
      renderRead();
      return;
    }
    post("/api/wf/mode", { mode: m, loaded_id: L && L.id });
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
  $("pathin").onkeydown = (e) => { if (e.key === "Enter") $("pathbtn").click(); };   // returns nothing: false would cancel every other key
  $("desibtn").onclick = () => $("desiin").value.trim() && readSpectra({ targetid: $("desiin").value.trim() });
  $("desiin").onkeydown = (e) => { if (e.key === "Enter") $("desibtn").click(); };   // returns nothing: false would cancel every other key
  $("sdssbtn").onclick = () => $("sdssin").value.trim() && readSpectra({ sdss: $("sdssin").value.trim() });
  $("sdssin").onkeydown = (e) => { if (e.key === "Enter") $("sdssbtn").click(); };   // returns nothing: false would cancel every other key
  $("groupsel").onchange = () => {
    const L = loaded();
    if (L && !nextReadMode && mode() === "multi")
      post("/api/wf/mode", { mode: "multi", group: $("groupsel").value, loaded_id: L.id });
  };
  $("removesel").onclick = () => {
    const indices = [...document.querySelectorAll('#loadedtbl input[data-k]:checked')].map((c) => +c.dataset.k);
    post("/api/wf/remove", { indices, loaded_id: S.wf.loaded.id });
  };
  $("toprepare").onclick = () => go("prepare");
}

// ================================================================ 2 PREPARE
let prep = null, prepFor = null;   // form values, for the spectra read (loaded id)
let pk = 0, pkind = "prepared", masking = false, prepTimer = null;
function appliedPreparation(scope = "all") {
  const settings = S.wf.settings;
  return settings.prepare_rows?.[scope] || settings.prepare;
}
function prepDefaults(scope = "all") {
  const L = loaded(), saved = appliedPreparation(scope);
  if (saved) return Object.assign({}, saved, { masks: (saved.masks || []).slice() });
  const zs = L ? L.rows.map((r) => r.z) : [];
  const hasZ = zs.length && zs.every((z) => z != null);
  const hasCoords = L && L.rows.every((r) => r.ra != null);
  // epochs of one object: one redshift (the files' own redshifts would shift the epochs against each other)
  return { z: null, z_source: hasZ ? (mode() === "multi" && zs.length > 1 ? "median" : "file") : "manual", rest_frame: true, extinction: hasCoords ? "coords" : "value",
    ebv: null, ra: "", dec: "", wmin: null, wmax: null, telluric: true, sky_lines: false, absorption: false, masks: [], flux_unit: "file", resolution: "file", resolution_unit: "kms" };
}
function readPrepForm() {
  const num = (id) => ($(id).value.trim() === "" ? null : Number($(id).value.trim().replace(",", ".")));
  prep.z = num("zin"); prep.z_source = $("zsrc").value; prep.rest_frame = $("restin").checked;
  prep.ebv = num("ebvin"); prep.ra = $("rain").value.trim(); prep.dec = $("decin").value.trim();
  prep.wmin = num("wminin"); prep.wmax = num("wmaxin");
  prep.telluric = $("tellin").checked; prep.sky_lines = $("skyin").checked; prep.absorption = $("absin").checked;
  prep.flux_unit = $("fusel").value === "custom" ? ($("fuin").value.trim() || "file") : $("fusel").value;
  const rv = $("resin").value.trim().replace(",", ".");
  prep.resolution = $("ressel").value === "file" || rv === "" ? "file" : rv;
  prep.resolution_unit = $("ressel").value === "R" ? "R" : "kms";
}
function writePrepForm() {
  $("zin").value = prep.z ?? ""; $("zsrc").value = prep.z_source; $("restin").checked = prep.rest_frame;
  $("zin").disabled = prep.z_source !== "manual";
  $("ebvin").value = prep.ebv ?? ""; $("rain").value = prep.ra || ""; $("decin").value = prep.dec || "";
  $("wminin").value = prep.wmin ?? ""; $("wmaxin").value = prep.wmax ?? "";
  $("tellin").checked = prep.telluric; $("skyin").checked = prep.sky_lines; $("absin").checked = prep.absorption;
  const fu = prep.flux_unit || "file", listed = [...$("fusel").options].some((o) => o.value === fu);
  $("fusel").value = listed ? fu : "custom"; $("fuin").value = listed ? "" : fu;
  $("fuin").classList.toggle("hidden", $("fusel").value !== "custom");
  renderFluxUnit();
  const given = prep.resolution != null && prep.resolution !== "file";
  $("ressel").value = given ? (prep.resolution_unit === "R" ? "R" : "kms") : "file"; $("resin").value = given ? prep.resolution : "";
  $("resin").classList.toggle("hidden", !given);
  renderResolution();
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
// the unit of an integrated line flux (flux density x Å), "" when the flux unit is not known
function lineFluxUnit(fluxUnit = prepared()?.flux_unit) {
  return fluxUnit ? unitText(fluxUnit).replace(" Å⁻¹", "") : "";
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
// finding the redshift: the candidates of the last search; a click on one takes it as the redshift
let zfShown = "";
// a candidate redshift under the pointer: its lines on the spectrum plot of this step
let prevShown = null;
function markCandidate(zc) {
  const st = plots.get($("pplot")), Z = S.wf.redshift, d = prevShown;
  if (!st || !st.cfg || !d) return;
  if (zc === null || !Z || !Z.marks) { if (st.cfg.marks) { st.cfg.marks = null; drawPlotEl($("pplot")); } return; }
  // the plot is in the observed frame, or in the rest frame of the redshift it was prepared with
  const f = (1 + zc) / (d.rest_frame && d.z ? 1 + d.z : 1);
  const air = d.medium === "air" && !d.rest_frame, toAir = (w) => w / (1 + 2.735182e-4 + 131.4182 / w ** 2 + 2.76249e8 / w ** 4);
  st.cfg.marks = Z.marks.map(([n, w, t]) => ({ x: air ? toAir(w * f) : w * f, text: n, kind: t }));
  st.cfg.marksTitle = `z = ${zc.toFixed(5)}`;
  drawPlotEl($("pplot"));
}
function renderZNote() {
  const L = loaded(), P = prepared(), many = L && L.rows.length > 1, src = $("zsrc").value;
  $("zsrc").querySelector('[value="median"]').hidden = !many;
  if (!many && src === "median") $("zsrc").value = "file";
  const zs = many ? L.rows.map((r) => r.z).filter((z) => z) : [];
  let spread = 0, rms = 0;
  if (zs.length > 1) {
    const med = zs.slice().sort((a, b) => a - b)[Math.floor(zs.length / 2)], c = 299792.458 / (1 + med);
    spread = (Math.max(...zs) - Math.min(...zs)) * c;
    const mean = zs.reduce((a, b) => a + b, 0) / zs.length;
    rms = Math.sqrt(zs.reduce((a, b) => a + (b - mean) ** 2, 0) / zs.length) * c;
  }
  let txt = "";
  if (src === "median" && many) txt = `All spectra are moved to the rest frame with one redshift, the median of the files${P && P.z != null ? ` (z = ${fmtNum(P.z, 6)})` : ""}.` +
    (spread > 1 ? ` The files differ by up to ${Math.round(spread)} km/s (rms ${Math.round(rms)}).` : "");
  else if (src === "file" && spread > 15) txt = `<b class="warnword">The redshifts of the files differ by up to ${Math.round(spread)} km/s</b> (rms ${Math.round(rms)}). Each spectrum is moved with its own, so the same line sits at different wavelengths in different epochs. For epochs of one object choose <b>one for all</b>, or fit wavelength offsets.`;
  $("znote").innerHTML = txt;
  $("znote").classList.toggle("hidden", !txt);
}
function renderZFind() {
  const L = loaded();
  if (!L) return;
  renderZNote();
  const Z = S.wf.redshift, cell = S.cells[S.cells.length - 1];
  const running = S.busy && cell && cell.code.startsWith("# find redshift");
  const noZ = L.rows.some((r) => r.npix > 0 && (r.z === null || r.z === 0));
  $("zfind").disabled = !!S.busy;
  $("zfind").textContent = running ? "Searching…" : Z ? "Search again" : "Find the redshift";
  $("zfind").classList.toggle("primary", noZ && !Z);
  $("zfindnote").innerHTML = Z ? "" : noZ ? '<b class="warnword">No redshift in the file.</b> Enter it, or let FANTASY find it: the spectrum is compared with galaxy and quasar templates at every redshift (a few seconds).'
    : "Checks the redshift of the file, or finds it when there is none: the spectrum is compared with galaxy and quasar templates at every redshift.";
  $("zfindout").classList.toggle("hidden", !Z);
  if (!Z) { zfShown = ""; return; }
  const cur = Number(String($("zin").value).replace(",", "."));
  const sig = JSON.stringify([Z.id, !!S.busy, $("zin").value, $("zsrc").value]);
  if (zfShown === sig) return;
  zfShown = sig;
  const kms = (a, b) => 299792.458 * Math.abs(a - b) / (1 + b);
  const how = `${Z.kind} templates${Z.method === "narrow lines" ? ", refined on the narrow lines" : ""}`;
  const fileNote = Z.file_z ? ` The file says ${Z.file_z.toFixed(5)}${kms(Z.z, Z.file_z) < 300 ? ": they agree." : `: <b class="warnword">${Math.round(kms(Z.z, Z.file_z))} km/s apart.</b>`}` : "";
  const head = `<div class="zhead"><div class="zbig"><small>best redshift</small><b>${Z.z.toFixed(5)}</b>${Z.error ? `<span>± ${Z.error.toFixed(5)}</span>` : ""}</div>` +
    (Z.secure ? '<span class="badge ok">secure</span>' : '<span class="badge warn">not secure</span>') + `</div>` +
    `<p class="note small zhow">${how}.${fileNote}${Z.secure ? "" : " Another redshift fits almost as well: point at each to see its lines on the spectrum before you choose."}</p>`;
  const chart = `<div class="zchart" id="zchart"><canvas id="zcv"></canvas><div class="ztip hidden" id="ztip"></div></div>` +
    `<div class="zchart-legend"><span><i style="background:#3b6fb6"></i>quasar</span><span><i style="background:#1b7f5c"></i>galaxy</span><span class="muted">χ² per pixel, lower is better; gaps: no physical fit</span></div>`;
  const maxd = Math.max(Z.threshold * 2, ...Z.candidates.map((c) => c.dchi2 || 0));
  const cands = Z.candidates.map((c, i) => {
    const on = $("zsrc").value === "manual" && isFinite(cur) && kms(cur, c.z) < 30;
    const close = i > 0 && c.dchi2 < Z.threshold;
    const pk = /^(\d+) of (\d+)$/.exec(c.peaks || "");
    const peaks = pk ? `<span class="zpk" title="${pk[1]} of the ${pk[2]} emission peaks of the spectrum sit on a known line at this redshift">` +
      (+pk[2] <= 16 ? Array.from({ length: +pk[2] }, (_, j) => `<i class="${j < +pk[1] ? "y" : ""}"></i>`).join("") : `<em style="--f:${pk[1] / pk[2]}"></em>`) +
      `<small>${pk[1]}/${pk[2]} peaks</small></span>` : "";
    const fit = i === 0 ? `<span class="zfit best"><small>best fit</small></span>`
      : `<span class="zfit${close ? " close" : ""}" title="χ² per pixel above the best; below ${Z.threshold} the two are hard to tell apart"><em><i style="width:${Math.min(100, 100 * c.dchi2 / maxd).toFixed(1)}%"></i><b style="left:${(100 * Z.threshold / maxd).toFixed(1)}%"></b></em><small>+${c.dchi2.toFixed(2)} χ²/px${close ? " · close" : ""}</small></span>`;
    return `<button type="button" class="zcand${on ? " on" : ""}${i ? "" : " first"}" data-z="${c.z}" title="${esc(c.lines)}">` +
      `<span class="zrank">${i + 1}</span><span class="zmain"><b>${c.z.toFixed(5)}</b><span class="zkind k-${esc(c.kind)}">${esc(c.kind)}</span></span>${fit}${peaks}</button>`;
  }).join("");
  $("zfindout").innerHTML = head + chart + `<div class="zlist">${cands}</div>` +
    `<p class="note small">Click a redshift, or a dip of the curve, to use it for ${L.shown > 1 ? "all spectra" : "the spectrum"}; point at one to mark its lines on the spectrum. Drag on the curve to zoom, double-click to zoom out. ` +
    `<a href="#" class="textlink" id="zfindfig">Figure</a></p>`;
  const useZ = (z) => { markCandidate(null); $("zsrc").value = "manual"; $("zin").disabled = false; $("zin").value = z.toFixed(5); zfShown = ""; renderZFind(); schedulePrepare(); };
  $("zfindout").querySelectorAll(".zcand").forEach((b) => {
    b.onclick = () => useZ(+b.dataset.z);
    b.onmouseenter = b.onfocus = () => { markCandidate(+b.dataset.z); zChart.hot = +b.dataset.z; drawZChart(); };
    b.onmouseleave = b.onblur = () => { markCandidate(null); zChart.hot = null; drawZChart(); };
  });
  $("zfindfig").onclick = (e) => { e.preventDefault(); window.open(withToken(`/api/wf/redshift.png?v=${Z.id}`), "_blank"); };
  zChart.use = useZ;
  setupZChart(Z);
}
// the chi2 of the redshift search against z (GET /api/wf/redshift_curve): hover to scan the lines of any
// redshift over the spectrum, click a dip to take it, drag to zoom
const zChart = { data: null, id: null, view: null, hover: null, hot: null, use: null, drag: null };
async function setupZChart(Z) {
  if (zChart.id !== Z.id) {
    zChart.id = Z.id; zChart.data = null; zChart.view = null;
    try { const d = await api("/api/wf/redshift_curve"); if (d && d.id === Z.id) zChart.data = d; } catch (e) { zChart.data = null; }
  }
  const cv = $("zcv");
  if (!cv) return;
  const zOf = (ev) => { const r = cv.getBoundingClientRect(), g = zChart.geo; return g ? g.inv(ev.clientX - r.left) : null; };
  cv.onmousemove = (ev) => {
    const x = zOf(ev);
    if (x === null) return;
    if (zChart.drag) zChart.drag.x1 = x;
    zChart.hover = x;
    markCandidate(10 ** x - 1);
    drawZChart();
  };
  cv.onmouseleave = () => { zChart.hover = null; zChart.drag = null; markCandidate(null); drawZChart(); };
  cv.onmousedown = (ev) => { const x = zOf(ev); if (x !== null) zChart.drag = { x0: x, x1: x }; };
  cv.onmouseup = (ev) => {
    const d = zChart.drag, g = zChart.geo;
    zChart.drag = null;
    if (!d || !g) return;
    if (Math.abs(g.px(d.x1) - g.px(d.x0)) > 6) { zChart.view = [Math.min(d.x0, d.x1), Math.max(d.x0, d.x1)]; drawZChart(); return; }
    const z = zDip(d.x0);
    if (z !== null) zChart.use(z);
  };
  cv.ondblclick = () => { zChart.view = null; drawZChart(); };
  drawZChart();
}
// the redshift a click at log(1+z) = x means: a candidate within 8 pixels, else the lowest point nearby
function zDip(x) {
  const g = zChart.geo, Z = S.wf.redshift;
  if (!g || !Z) return null;
  const near = Z.candidates.map((c) => [Math.abs(g.px(Math.log10(1 + c.z)) - g.px(x)), c.z]).sort((a, b) => a[0] - b[0])[0];
  if (near && near[0] < 8) return near[1];
  let best = null;
  for (const c of Object.values(zChart.data ? zChart.data.curves : {}))
    c.x.forEach((xi, i) => { if (Math.abs(g.px(xi) - g.px(x)) < 8 && (!best || c.y[i] < best[1])) best = [xi, c.y[i]]; });
  return best ? 10 ** best[0] - 1 : null;
}
function drawZChart() {
  const cv = $("zcv"), Z = S && S.wf && S.wf.redshift;
  if (!cv || !Z) return;
  const w = cv.clientWidth, h = cv.clientHeight, dpr = window.devicePixelRatio || 1;
  if (!w) return;
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const D = zChart.data, curves = D ? Object.entries(D.curves) : [];
  const COL = { quasar: "#3b6fb6", galaxy: "#1b7f5c" };
  if (!curves.length) { ctx.fillStyle = "#7b8290"; ctx.font = "11px system-ui"; ctx.fillText(D ? "no curve" : "loading…", 8, h / 2); zChart.geo = null; return; }
  const all = curves.flatMap(([, c]) => c.x);
  const [x0, x1] = zChart.view || [Math.min(...all), Math.max(...all)];
  const ys = curves.flatMap(([, c]) => c.y.filter((_, i) => c.x[i] >= x0 && c.x[i] <= x1)).sort((a, b) => a - b);
  const lo = ys[0], hi = ys[Math.floor(0.85 * (ys.length - 1))] || lo + 1, pad = (hi - lo) * 0.06 || 0.1;
  const L = 12, R = w - 12, T = 12, B = h - 18;
  const px = (x) => L + (x - x0) / (x1 - x0 || 1) * (R - L), py = (y) => B - 7 - (Math.min(y, hi + pad) - lo) / (hi + pad - lo) * (B - 7 - T);
  zChart.geo = { px, inv: (p) => x0 + (p - L) / (R - L) * (x1 - x0) };
  // z ticks
  ctx.font = "10px system-ui"; ctx.textAlign = "center";
  const span = 10 ** x1 - 10 ** x0, steps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2];
  const st = steps.find((s) => span / s <= 6) || 2;
  for (let z = Math.ceil((10 ** x0 - 1) / st) * st; z <= 10 ** x1 - 1 + 1e-9; z += st) {
    const p = px(Math.log10(1 + z));
    ctx.strokeStyle = "#eef0f3"; ctx.beginPath(); ctx.moveTo(p, T); ctx.lineTo(p, B); ctx.stroke();
    ctx.fillStyle = "#7b8290"; ctx.fillText(+z.toFixed(4) + "", p, h - 5);
  }
  ctx.save(); ctx.beginPath(); ctx.rect(L - 2, 0, R - L + 4, B + 1); ctx.clip();
  // the current redshift
  const cur = Number(String($("zin").value).replace(",", "."));
  if (isFinite(cur) && cur > -1) { const p = px(Math.log10(1 + cur)); ctx.strokeStyle = "#d4472a"; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(p, T); ctx.lineTo(p, B); ctx.stroke(); ctx.setLineDash([]); }
  for (const [kind, c] of curves) {
    ctx.strokeStyle = COL[kind] || "#898781"; ctx.lineWidth = 1.2; ctx.beginPath();
    // no line across a gap of the scan (redshifts where the templates do not cover the spectrum)
    const dx = c.x.slice(1).map((x, i) => x - c.x[i]).sort((a, b) => a - b), gap = 6 * (dx[dx.length >> 1] || 1);
    let on = false;
    c.x.forEach((x, i) => { if (x < x0 - (x1 - x0) * 0.02 || x > x1 + (x1 - x0) * 0.02 || (i && x - c.x[i - 1] > gap)) on = false; if (x < x0 - (x1 - x0) * 0.02 || x > x1 + (x1 - x0) * 0.02) return; const X = px(x), Y = py(c.y[i]); on ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); on = true; });
    ctx.stroke();
  }
  const d = zChart.drag;
  if (d && Math.abs(px(d.x1) - px(d.x0)) > 6) { ctx.fillStyle = "rgba(212,71,42,0.10)"; ctx.fillRect(Math.min(px(d.x0), px(d.x1)), T, Math.abs(px(d.x1) - px(d.x0)), B - T); }
  if (zChart.hover !== null && !d) { const p = px(zChart.hover); ctx.strokeStyle = "#1d2128"; ctx.lineWidth = 0.8; ctx.beginPath(); ctx.moveTo(p, T); ctx.lineTo(p, B); ctx.stroke(); }
  ctx.restore();
  // the candidates at their dips
  Z.candidates.forEach((c, i) => {
    const x = Math.log10(1 + c.z);
    if (x < x0 || x > x1) return;
    const cv2 = D.curves[c.kind];
    let y = B - 10;
    if (cv2) { let k = 0, d = Infinity; cv2.x.forEach((xi, j) => { if (Math.abs(xi - x) < d) { d = Math.abs(xi - x); k = j; } }); y = py(cv2.y[k]); }
    const hot = zChart.hot !== null && Math.abs(zChart.hot - c.z) < 1e-6;
    ctx.fillStyle = i === 0 ? "#d4472a" : hot ? "#1d2128" : "#fff";
    ctx.strokeStyle = i === 0 ? "#d4472a" : "#1d2128"; ctx.lineWidth = 1.2;
    const cy = Math.max(T, Math.min(B - 7, y));
    ctx.beginPath(); ctx.arc(px(x), cy, 6.5, 0, 7); ctx.fill(); ctx.stroke();
    ctx.fillStyle = i === 0 || hot ? "#fff" : "#1d2128"; ctx.font = "bold 9px system-ui"; ctx.textBaseline = "middle";
    ctx.fillText(i + 1, px(x), cy + 0.5); ctx.textBaseline = "alphabetic";
  });
  ctx.strokeStyle = "#cfd3d9"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(L, B + 0.5); ctx.lineTo(R, B + 0.5); ctx.stroke();
  const tip = $("ztip");
  if (zChart.hover === null) { tip.classList.add("hidden"); return; }
  const vals = curves.filter(([, c]) => zChart.hover >= c.x[0] && zChart.hover <= c.x[c.x.length - 1]).map(([kind, c]) => { let k = 0, dd = Infinity; c.x.forEach((xi, j) => { if (Math.abs(xi - zChart.hover) < dd) { dd = Math.abs(xi - zChart.hover); k = j; } }); return `<span style="color:${COL[kind]}">${kind} ${c.y[k].toFixed(2)}</span>`; });
  tip.innerHTML = `<b>z = ${(10 ** zChart.hover - 1).toFixed(4)}</b>${vals.join("")}`;
  tip.classList.remove("hidden");
  const p = px(zChart.hover);
  tip.style.left = Math.max(0, Math.min(w - tip.offsetWidth, p - tip.offsetWidth / 2)) + "px";
}
// the instrumental resolution: what the file says; without it the widths include the instrument
function renderResolution() {
  const L = loaded();
  if (!L) return;
  const r0 = L.rows[0], known = L.rows.every((r) => r.resolution != null), mode = $("ressel").value;
  const px = r0.pixel_kms ? ` One pixel is ${Math.round(r0.pixel_kms)} km/s.` : "";
  const v = Number($("resin").value.trim().replace(",", "."));
  let txt;
  if (mode === "R") txt = isFinite(v) && v > 0 ? `R = ${fmtNum(v, 5)} is a FWHM of <b>${Math.round(299792.458 / v)} km/s</b>, used for all spectra and removed from the fitted widths.${px}` : "Give the resolving power, e.g. 2000.";
  else if (mode === "kms") txt = isFinite(v) && v > 0 ? `<b>${fmtNum(v, 4)} km/s</b> FWHM, used for all spectra and removed from the fitted widths.${px}` : "Give the FWHM of an unresolved line in km/s, e.g. 150.";
  else if (known) txt = `From the file: <b>${Math.round(r0.resolution)} km/s</b> FWHM${r0.resolution_varies ? " (median; it changes along the spectrum and each line uses its own value)" : ""}. It is removed from the fitted widths.${px}`;
  else txt = `<b class="warnword">Not in the file${L.rows.some((r) => r.resolution != null) ? " for some spectra" : ""}.</b> The fitted widths then include the instrumental broadening, and a narrow line can become narrower than a pixel, which makes its flux unreliable.${px} Give it if you know it (the instrument's R, or the FWHM of a sky or arc line).`;
  $("resnote").innerHTML = txt;
  $("resbox").classList.toggle("needs", !known && mode === "file");
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
  extLookupProblem();
  if (run) schedulePrepare();
}
// the E(B-V) lookup failed (the dust service is down, or no internet): say so where the choice is made
function extLookupProblem() {
  const failed = prep && prep.extinction === "coords" && typeof S.error === "string" && S.error.includes("no Galactic E(B-V) found");
  if (!failed) return;
  const down = S.error.includes("cannot be reached") || S.error.includes("urlopen error");
  $("coordnote").innerHTML = `<b class="warnword">E(B−V) could not be looked up${down ? ": the IRSA dust service does not answer (it may be down, or this computer is offline)" : ""}.</b> ` +
    `The spectra are not prepared. <a href="#" class="textlink" id="extbyhand">Enter E(B−V) by hand</a> (the SFD value of Schlafly &amp; Finkbeiner 2011 for this position), ` +
    `<a href="#" class="textlink" id="extnone">go on without the correction</a> (fluxes and luminosities are then too low), or try again later. ` +
    `For work offline, install the maps: <code>pip install sfdmap2</code> and set <code>SFD_DIR</code>.`;
  $("extbyhand").onclick = (e) => { e.preventDefault(); setExt("value", false); $("ebvin").focus(); };
  $("extnone").onclick = (e) => { e.preventDefault(); setExt("none"); };
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
  const opts = `<option value="all">all ${L.n} ${mode() === "multi" ? "epochs" : "spectra"}</option>` +
    L.rows.map((r) => `<option value="${r.k}">#${r.k}${r.date ? " · " + r.date : ""}</option>`).join("");
  if ($("applyto").dataset.sig !== String(L.id)) { $("applyto").innerHTML = opts; $("applyto").dataset.sig = String(L.id); }
  $("applybox").classList.toggle("hidden", n < 2);
  $("pprev").disabled = pk <= 0; $("pnext").disabled = pk >= n - 1;
  $("pprev").parentElement.querySelectorAll("button").forEach((b) => b.classList.toggle("hidden", n < 2));
  extLookupProblem();
  renderZFind();
  const P = prepared();
  if (P && !$("prepstatus").textContent.startsWith("Enter")) {
    const count = P.n < L.n ? `${P.n} previewed · ${L.n - P.n} prepared at fit time` : `${P.n} prepared`;
    $("prepstatus").textContent = `${count} · z ${P.z != null ? fmtNum(P.z, 5) : "—"} · E(B−V) ${P.ebv != null ? fmtNum(P.ebv, 3) : "—"}` +
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
  prevShown = d;
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
  $("ressel").onchange = () => { const f = $("ressel").value === "file"; $("resin").classList.toggle("hidden", f); if (f) $("resin").value = ""; renderResolution(); if (f) schedulePrepare(); else { $("resin").placeholder = $("ressel").value === "R" ? "2000" : "150"; if ($("resin").value.trim()) schedulePrepare(); else $("resin").focus(); } };
  $("resin").onchange = () => { renderResolution(); schedulePrepare(); };
  $("zfind").onclick = () => post("/api/wf/find_redshift", { k: pk || 0 });
  $("zsrc").onchange = () => { $("zin").disabled = $("zsrc").value !== "manual"; schedulePrepare(); };
  document.querySelectorAll("#extseg button").forEach((b) => (b.onclick = () => setExt(b.dataset.ext)));
  document.querySelectorAll("#pkind button").forEach((b) => (b.onclick = () => { pkind = b.dataset.kind; renderPrepare(); loadPreview(true); }));
  $("pprev").onclick = () => { pk = Math.max(0, pk - 1); renderPrepare(); loadPreview(true); };
  $("pnext").onclick = () => { pk = Math.min(loaded().shown - 1, pk + 1); renderPrepare(); loadPreview(true); };
  $("maskbtn").onclick = () => setMasking(!masking);
  $("applyto").onchange = () => {
    clearTimeout(prepTimer);
    const scope = $("applyto").value;
    prep = prepDefaults(scope);
    writePrepForm();
    if (scope !== "all") { pk = Number(scope); renderPrepare(); loadPreview(true); }
  };
  $("tomodel").onclick = () => {
    readPrepForm();
    const bad = prepValid();
    if (bad) { $("prepstatus").textContent = bad; return; }
    const saved = JSON.stringify(appliedPreparation($("applyto").value) || null), now = JSON.stringify(prep);
    if (prepared() && sameSettings(saved, now)) { go("model"); return; }
    pendingGo = "model"; runPrepare();
  };
}
function sameSettings(a, b) {
  try {
    const x = JSON.parse(a), y = JSON.parse(b);
    if (!x) return false;
    return ["z", "z_source", "rest_frame", "extinction", "ebv", "ra", "dec", "wmin", "wmax", "telluric", "sky_lines", "absorption", "flux_unit", "resolution", "resolution_unit"]
      .every((k) => String(x[k] ?? "") === String(y[k] ?? "")) && JSON.stringify(x.masks || []) === JSON.stringify(y.masks || []);
  } catch (e) { return false; }
}

// ================================================================== 3 MODEL
let selComp = null, pendingSuggest = false, guessSig = null, modelTried = false, startMode = null, classInfo = null;
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
  document.querySelectorAll("#optbar input[data-opt]").forEach((c) => { if (!c.closest(".check").classList.contains("hidden")) opt[c.dataset.opt] = c.checked; });
  const d = (S.model_defaults || {})[$("presetsel").value] || {};
  const cont = (d.options || []).includes("continuum") ? { continuum: $("contsel").value } : {};
  return Object.assign({ action: "preset", model: $("presetsel").value, broad_components: +$("bcount").textContent, host: $("hostsel").value }, cont, opt);
}
function syncOptionAvailability() {
  const d = (S.model_defaults || {})[$("presetsel").value] || {};
  const has = (k) => (d.options || []).includes(k);
  $("broadbox").classList.toggle("hidden", !has("broad_components"));
  $("hostbox").classList.toggle("hidden", !has("host"));
  $("contbox").classList.toggle("hidden", !has("continuum"));
  for (const k of ["outflow", "coronal", "oi6046", "red"]) $("o-" + k).classList.toggle("hidden", !has(k));
  $("o-stratified").classList.toggle("hidden", !d.stratified);
}
function setPresetDefaults(label) {
  const d = (S.model_defaults || {})[label] || {};
  $("bcount").textContent = d.broad_components ?? 2;
  document.querySelector('#optbar input[data-opt="outflow"]').checked = !!d.outflow;
  document.querySelector('#optbar input[data-opt="stratified"]').checked = false;
  $("contsel").value = (d.values || {}).continuum === "broken" ? "broken" : "powerlaw";
  for (const k of ["coronal", "oi6046", "red"]) document.querySelector(`#optbar input[data-opt="${k}"]`).checked = !!(d.values || {})[k];
}
// set the controls from the preset cell (made here, by the recommendation, or edited in the session)
function describePreset(pl) {
  if (!pl) return { label: null, args: {} };
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
  return { label: best, args };
}
let syncedPreset = null;
function syncFromPreset() {
  const pl = presetLabel();
  if (!pl || pl.code === syncedPreset) return;
  syncedPreset = pl.code;
  const {label: best, args} = describePreset(pl);
  if (!best) return;
  $("presetsel").value = best;
  setPresetDefaults(best);
  if (args.broad_components) $("bcount").textContent = args.broad_components;
  const box = (k) => document.querySelector(`#optbar input[data-opt="${k}"]`);
  for (const k of ["outflow", "coronal", "oi6046", "red"]) if (k in args) box(k).checked = args[k] === "True";
  box("stratified").checked = args.broad_profile === "stratified";
  if (args.continuum) $("contsel").value = args.continuum === "broken" ? "broken" : "powerlaw";
  const host = args.host;
  $("hostsel").value = host === undefined || host === "True" ? "default" : host === "False" ? "none" : host;
  if (!$("hostsel").value) $("hostsel").value = "default";
  syncOptionAvailability();
}
// the preset options as the active preset made them, in the shape of modelForm(); null if not a known preset
function appliedPresetForm() {
  const { label, args } = describePreset(presetLabel());
  const d = label && (S.model_defaults || {})[label];
  if (!d) return null;
  const v = d.values || {}, has = (k) => (d.options || []).includes(k);
  const f = { action: "preset", model: label, broad_components: +(args.broad_components ?? d.broad_components ?? 2),
    host: args.host === undefined || args.host === "True" ? "default" : args.host === "False" ? "none" : args.host };
  if (has("continuum")) f.continuum = (args.continuum ?? v.continuum) === "broken" ? "broken" : "powerlaw";
  f.outflow = "outflow" in args ? args.outflow === "True" : !!d.outflow;
  f.stratified = args.broad_profile === "stratified";
  for (const k of ["coronal", "oi6046", "red"]) f[k] = k in args ? args[k] === "True" : !!v[k];
  return f;
}
// what in the controls differs from the model in use: [] when nothing does
function presetChanges() {
  const a = appliedPresetForm();
  if (!a) return [];
  const f = modelForm();
  if (f.model !== a.model) return ["model"];
  const d = (S.model_defaults || {})[f.model] || {}, has = (k) => (d.options || []).includes(k);
  const out = [];
  if (has("broad_components") && f.broad_components !== a.broad_components) out.push("broad_components");
  if (has("host") && f.host !== a.host) out.push("host");
  if (has("continuum") && f.continuum !== a.continuum) out.push("continuum");
  for (const k of ["outflow", "stratified", "coronal", "oi6046", "red"]) if (k in f && f[k] !== a[k]) out.push(k);
  return out;
}
function makePreset() {
  syncOptionAvailability();
  startMode = "preset";
  post("/api/model", modelForm());
}
// ------------------------------------------------- the Model step of a sample: the model of each class, per spectrum
const SP_WIN = [["C IV", 1450, 1720], ["Mg II", 2650, 3050], ["Hβ", 4800, 5050], ["Hα", 6500, 6650]];
const SP_ROUTE = { optical: "optical", uv_mgii: "UV · Mg II", uv_civ: "UV · C IV", none: "no line model" };
const SP_ROUTE_TIP = { optical: "classified by fa.classify when the sample is fitted", uv_mgii: "ends below 5200 Å: the Mg II model", uv_civ: "ends below 5200 Å: the C IV model" };
let spSig = null;
async function loadSamplePlan() {
  const sig = JSON.stringify([S.cells.length, !!S.sample]);
  if (spSig === sig && $("splan").innerHTML) return;
  spSig = sig;
  let P;
  try { P = await api("/api/wf/sample_plan"); } catch (e) { $("splan").innerHTML = `<div class="card note-card">${esc(e.message || e)}</div>`; return; }
  renderSamplePlan(P);
}
function renderSamplePlan(P) {
  const sp = P.spectra, n = P.n_input || sp.length, cn = (c) => CLASS_NAME[c] || c, preview = n > sp.length;
  const byRoute = {}, byCls = Object.assign({}, P.class_counts || {});
  sp.forEach((x) => { byRoute[x.route] = (byRoute[x.route] || 0) + 1; if (x.cls && !P.class_counts) byCls[x.cls] = (byCls[x.cls] || 0) + 1; });
  const flow = `<div class="sp-flow">` +
    `<div class="sp-node"><small>1</small><b>${n} spectr${n === 1 ? "um" : "a"}</b><span>read and prepared</span></div><i class="sp-arrow"></i>` +
    `<div class="sp-node"><small>2</small><b>Rest frame ends below ${Math.round(P.uv_limit)} Å?</b><span>yes: Hβ and [O III] are cut off, the UV model of Mg II (else C IV) · <b>${(byRoute.uv_mgii || 0) + (byRoute.uv_civ || 0)}</b></span><span>no: <code>fa.classify</code> at fit time · <b>${byRoute.optical || 0}</b></span>${preview ? `<span>counted in the first ${sp.length} spectra</span>` : ""}</div><i class="sp-arrow"></i>` +
    `<div class="sp-node"><small>3</small><b>The model of the class</b><span>${P.fitted ? `${Object.keys(byCls).length} classes in this sample` : "from the table below"}; narrow-line galaxies with χ²ᵣ &gt; 1.2 get the extensions the BIC prefers (<code>fa.refine</code>)</span></div></div>`;
  const one = P.one_model ? `<div class="callout warn small"><div>The Fit step is set to fit <b>every</b> spectrum with <code>${esc(P.one_model.label)}</code>: the classes below are not used. Choose “by class” there to use them.</div></div>` : "";
  // coverage of each spectrum against the windows the route depends on
  const lo = 1300, hi = 9000, X = (w) => ((Math.min(hi, Math.max(lo, w)) - lo) / (hi - lo) * 100).toFixed(2);
  const scale = `<div class="sp-cov sp-scale">${SP_WIN.map(([nm, a, b]) => `<span class="sp-win" style="left:${X(a)}%;width:${(X(b) - X(a)).toFixed(2)}%"><em>${nm}</em></span>`).join("")}` +
    `<span class="sp-lim" style="left:${X(P.uv_limit)}%"><em>${Math.round(P.uv_limit)} Å</em></span></div>`;
  const rows = sp.map((x) => `<tr data-k="${x.k}"><td class="mono">#${x.k}</td><td class="mono sp-name">${esc(x.name)}</td><td class="num mono">${x.z === null ? "—" : x.z.toFixed(4)}</td>` +
    `<td class="sp-covcell"><div class="sp-cov">${SP_WIN.map(([nm, a, b]) => `<span class="sp-win" style="left:${X(a)}%;width:${(X(b) - X(a)).toFixed(2)}%"></span>`).join("")}` +
    (x.blue !== null ? `<span class="sp-bar r-${x.route}" style="left:${X(x.blue)}%;width:${(X(x.red) - X(x.blue)).toFixed(2)}%" title="${Math.round(x.blue)}–${Math.round(x.red)} Å in the rest frame"></span>` : "") +
    `<span class="sp-lim" style="left:${X(P.uv_limit)}%"></span></div></td>` +
    `<td><span class="sp-route r-${x.route}" title="${esc(x.why || SP_ROUTE_TIP[x.route] || "")}">${esc(SP_ROUTE[x.route] || x.route)}</span></td>` +
    (P.fitted ? `<td>${x.cls ? `<span class="sgdot" style="background:${CLASS_COLOR[x.cls] || "#9aa1ad"}"></span>${esc(cn(x.cls))}` : '<span class="muted">—</span>'}</td><td class="mono sp-preset">${esc(x.preset || "—")}</td>` : "") + `</tr>`).join("");
  const table = `<div class="card flush tblwrap"><table class="tbl sp-tbl"><thead><tr><th>#</th><th>Spectrum</th><th class="num">z</th><th>Rest-frame coverage ${scale}</th><th>Route</th>${P.fitted ? "<th>Class (fit)</th><th>Model used</th>" : ""}</tr></thead><tbody>${rows}</tbody></table></div>`;
  const cards = P.routes.map((r) => {
    const used = byCls[r.cls] || 0;
    return `<div class="sp-card${P.fitted && !used ? " unused" : ""}${used ? " used" : ""}"><div class="sp-chead"><span class="sgdot" style="background:${CLASS_COLOR[r.cls] || "#9aa1ad"}"></span><b>${esc(cn(r.cls))}</b><span class="grow"></span>` +
      (P.fitted ? `<span class="badge ${used ? "ok" : "muted"}">${used ? `${used} in this sample` : "none"}</span>` : "") + `</div>` +
      `<code class="sp-code">${esc(r.code)}</code><p>${esc(r.about)}</p></div>`;
  }).join("");
  $("splan").innerHTML = `<div class="card sp-hero"><div class="cs-top"><div><small>Sample · model by class</small><h2>Each spectrum gets the model of its class</h2>` +
    `<p class="cs-sub">The class comes from <code>fa.classify</code> when the sample is fitted (<code>fa.fit_sample</code>), and picks a preset from <code>Classification.RECOMMENDED</code>. Nothing to build here: to fit every spectrum with one model instead, choose it in the Fit step.</p></div></div>${flow}${one}</div>` +
    `<div class="sp-sec"><b>Your spectra</b><span class="muted small">${preview ? `the first ${sp.length} of ${n}, read for the preview (the fit reads them all); ` : ""}the bar: the rest-frame range of each spectrum; the shaded windows: the lines the route depends on</span></div>${table}` +
    `<div class="sp-sec"><b>Class → model</b><span class="muted small">the preset each class gets${P.fitted ? "; the badge: how many spectra of this sample got it" : ""}</span></div><div class="sp-grid">${cards}</div>`;
}
function enterModel(force) {
  const m = mode();
  const special = m === "classify" || m === "sample";
  $("model-note").classList.toggle("hidden", !special);
  ["optbar", "modelgrid", "classcard"].forEach((id) => $(id).classList.toggle("hidden", special));
  if (m === "classify") $("model-note").textContent = "Classification fits its own model (stellar populations + power law, then the emission lines); there is nothing to set up here.";
  if (m === "sample") $("model-note").textContent = "In a sample each spectrum is classified and gets the model its class needs; you can also choose one model for all of them in the Fit step.";
  $("model-note").classList.toggle("hidden", !special || m === "sample");
  $("splan").classList.toggle("hidden", m !== "sample");
  if (m === "sample") loadSamplePlan();
  $("tofit").textContent = m === "classify" ? "Set up classification →" : "Set up fit →";
  if (special) {
    $("tofit").disabled = !!S.busy || !(prepared() || (m === "sample" && loaded()));
    return;
  }
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
  if (!S.model && prepared() && !S.busy && modelTried !== prepared().id) {
    modelTried = prepared().id; makePreset(); return;
  }
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
const BOWEN_RE = /^(NIII|OIII3|HeII)\d+$/, CORONAL_RE = /^\[Fe(VII|X|XI|XIV)\]\d+$/;
// coronal lines (plotting.is_coronal): named coronal..., or of [Fe VII], [Fe X], [Fe XI], [Fe XIV] lines only
const isCoronal = (c) => c.role === "narrow" && (/^coronal/.test(c.name) || (!!(c.lines && c.lines.length) && c.lines.every((l) => CORONAL_RE.test(l))));
function compColor(c, all) {
  if (isCoronal(c)) return C.coronal;
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
  try { const d = await api("/api/wf/classify?k=0&stage=preliminary"); classInfo = d && d.label ? d : null; } catch (e) { classInfo = null; }
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
  const active = describePreset(presetLabel());
  const broad = active.args.broad_components ?? S.model_defaults?.[active.label]?.broad_components;
  const inUse = active.label === rec.model && (!rec.broad_components || Number(broad) === rec.broad_components);
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
  const openingSaved = m === "saved" && $("savedopts").classList.contains("hidden");
  startMode = m;
  document.querySelectorAll("#startseg button").forEach((b) => b.classList.toggle("on", b.dataset.s === m));
  $("presetopts").classList.toggle("hidden", m !== "preset");
  $("customopts").classList.toggle("hidden", m !== "custom");
  $("savedopts").classList.toggle("hidden", m !== "saved");
  if (openingSaved) refreshSaved();
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
  if (window.MB) MB.render();
  if (mode() === "classify" || mode() === "sample") return;
  const M = S.model;
  const pl = presetLabel();
  const loaded = [...S.cells].reverse().map((c) => /^model = fa\.(load_model\([^)]*\)|Model\(|presets\.)/m.exec(c.code)).find(Boolean);
  const saved = loaded && loaded[1].startsWith("load_model") ? loaded[1].replace(/'[^']*\/([^'/]+)\.json'/, "'$1'") : null;
  $("model-hint").textContent = pl ? pl.code : saved ? `fa.${saved}` : M ? "fa.Model(…)" : "fa.Model()";
  if ($("presetsel").options.length) syncFromPreset();
  setStartMode(startMode || modelOrigin() || "preset");
  renderClassCard();
  $("tofit").disabled = !M || !M.components.length || !!S.busy;
  for (const id of ["savemodel", "dlmodel"]) $(id).disabled = !M || !!S.busy;
  $("usepreset").disabled = !!S.busy;
  $("startempty").disabled = !!S.busy;
  if (!M || !M.components.length) {
    $("comps").innerHTML = '<li class="muted empty">No active components.</li>';
    delete $("comps").dataset.sig;
    $("nfree").textContent = "";
    $("ghl").textContent = "";
    guessSig = loadGuess.data = null;
    drawGuess(); renderSharing(); renderModelSum(); return;
  }
  $("nfree").textContent = `${M.n_free} free parameters`;
  if (!M.components.some((c) => c.name === selComp)) {
    selComp = (M.components.find((c) => c.role === "broad") || M.components[0]).name;
  }
  const SH = mode() === "multi" && !$("sharedin").value.trim() ? S.wf.sharing || [] : null;
  const pats = mode() === "multi" && !SH ? sharedPatterns() : [];
  const sig = JSON.stringify([M.components, pats, SH, selComp]);
  if ($("comps").dataset.sig !== sig) {
    $("comps").dataset.sig = sig;
    $("comps").innerHTML = M.components.map((c) => {
      const row = SH ? SH.find((x) => x.name === c.name) : null;
      const byPat = pats.some((p) => p.split(".")[0] === c.name || (p.endsWith("*") && c.name.startsWith(p.split(".")[0]) && p.split(".")[0] !== "" && p.startsWith(c.name)));
      const shared = row ? (row.flux && row.shape !== false ? "shared" : row.flux ? "fluxes shared" : row.shape ? "shape shared" : "") : byPat ? "shared" : "";
      const badges = [c.profile && c.profile !== "gaussian" ? `<span class="badge muted">${esc(PROFILE_NAME[c.profile] || c.profile)}</span>` : "",
        c.absorption ? '<span class="badge k-absorption">absorption</span>' : "",
        c.tied_to ? `<span class="badge info" title="shift and width tied to ${esc(c.tied_to)}">= ${esc(c.tied_to)}</span>` : "",
        shared ? `<span class="badge info" title="between the epochs of the joint fit">${shared}</span>` : ""].join("");
      return `<li><button class="comp-preview" data-c="${esc(c.name)}" aria-pressed="${c.name === selComp}" title="Highlight ${esc(c.name)} in the initial guess"><span class="sw" style="background:${compColor(c, M.components)}"></span>` +
        `<span class="cname">${esc(c.name)}</span><span class="cbadges">${badges}</span><span class="cdesc">${esc(compDesc(c))}</span></button></li>`;
    }).join("");
    $("comps").querySelectorAll("button").forEach((button) => (button.onclick = () => {
      selComp = button.dataset.c;
      $("comps").dataset.sig = JSON.stringify([M.components, pats, SH, selComp]);
      $("comps").querySelectorAll("button").forEach((row) => row.setAttribute("aria-pressed", String(row === button)));
      drawGuess();
    }));
  }
  renderSharing();
  renderModelSum();
}
// ---- the model in use: what it is, where it came from, whether it was fitted, what it fits
const MSUM_KINDS = [["continuum", "Continuum"], ["host", "Host galaxy"], ["feii", "Fe II"], ["narrow", "Narrow lines"], ["coronal", "Coronal lines"], ["broad", "Broad lines"],
  ["outflow", "Outflow"], ["balmer", "Balmer"], ["absorption", "Absorption"], ["other", "Other"]];
function msumKind(c) {
  if (c.absorption) return "absorption";
  if (c.kind === "host") return "host";
  if (c.kind === "continuum" && !/^balmer/.test(c.name)) return "continuum";
  if (c.type === "FeII") return "feii";
  if (isCoronal(c)) return "coronal";
  if (["narrow", "broad", "outflow"].includes(c.role)) return c.role;
  if (["BalmerContinuum", "HighOrderBalmer"].includes(c.type)) return "balmer";
  return "other";
}
function renderModelSum() {
  const M = S.model, box = $("msum"), m = mode();
  const show = !!(M && M.components.length) && !["classify", "sample"].includes(m);
  box.classList.toggle("hidden", !show);
  if (!show) return;
  // the cell that made the model, and the cells that changed it since
  let made = -1;
  for (let i = S.cells.length - 1; i >= 0; i--) if (/^model\s*=\s*fa\./m.test(S.cells[i].code)) { made = i; break; }
  const edits = made < 0 ? [] : S.cells.slice(made + 1).map((c, j) => ({ i: made + 1 + j, code: c.code }))
    .filter((c) => /^model\s*(\.|=\s*model)/m.test(c.code) && !/^# (fit|trial fit)/.test(c.code));
  let fitted = -1;
  for (let i = S.cells.length - 1; i > made; i--) if (/^# (fit|trial fit)/.test(S.cells[i].code)) { fitted = i; break; }
  const lastEdit = edits.length ? edits[edits.length - 1].i : made;
  const origin = modelOrigin(), pl = presetLabel(), F = fitDone();
  // the title: the preset in words, or how the model was built
  let title, bits = [];
  if (origin === "preset" && pl) {
    const { label, args } = describePreset(pl);
    const opt = label && [...$("presetsel").options].find((o) => o.value === label);
    const nm = opt ? opt.textContent.split(" · ")[0] : pl.name;
    title = `${nm.charAt(0).toUpperCase()}${nm.slice(1)} preset`;
    if (args.broad_components) bits.push(`${args.broad_components} broad component${args.broad_components === "1" ? "" : "s"}`);
    if (args.outflow === "True") bits.push("outflow wing"); else if (args.outflow === "False") bits.push("no outflow wing");
    if (args.broad_profile === "stratified") bits.push("stratified BLR");
    if (args.continuum) bits.push(args.continuum === "broken" ? "broken power law" : "power law");
    if (args.host === "False") bits.push("no host"); else if (args.host && args.host !== "True") bits.push(`host: ${args.host}`);
    for (const [k, t] of [["coronal", "coronal lines"], ["oi6046", "O I 6046"], ["red", "red side to 7500 Å"]]) if (args[k] === "True") bits.push(t);
  } else if (origin === "saved") title = "A saved model";
  else title = "A model built in the editor";
  bits.push(`fitted over ${M.wave_range ? `${Math.round(M.wave_range[0])}–${Math.round(M.wave_range[1])} Å` : "the whole spectrum"}`);
  // the state: fitted with this model, or not yet
  let state;
  if (fitted > lastEdit && F && F.rchi2 != null && ["single", "joint", "fits"].includes(F.kind))
    state = `<span class="msum-state ok"><i></i>Fitted with this model · χ²ᵣ ${F.rchi2.toFixed(2)}</span>`;
  else if (F && F.rchi2 != null) state = '<span class="msum-state warn"><i></i>Not fitted yet: the results are from an earlier model</span>';
  else state = '<span class="msum-state"><i></i>Not fitted yet</span>';
  // the preset options above, changed but not applied
  const changes = origin === "preset" && startMode !== "custom" && startMode !== "saved" ? presetChanges() : [];
  const pending = changes.length > 0;
  $("usepreset").classList.toggle("primary", pending);
  let pendingText = "";
  if (pending) {
    const so = $("presetsel").selectedOptions[0];
    const name = so ? so.textContent.split(" · ")[0] : $("presetsel").value;
    const NAMES = { broad_components: "broad components", host: "host galaxy", continuum: "continuum", outflow: "outflow wing",
      stratified: "stratified BLR", coronal: "coronal lines", oi6046: "O I 6046", red: "red side" };
    pendingText = changes[0] === "model"
      ? `<b>${esc(name)}</b> is selected above, but the model in use is still the one below.`
      : `The options above changed (${changes.map((k) => NAMES[k] || k).join(", ")}), but the model in use is still the one below.`;
  }
  // where it came from
  const madeLine = made >= 0 ? S.cells[made].code.split("\n").find((l) => /^model\s*=/.test(l)) || "" : "";
  const from = origin === "preset" && pl ? `Made by <code>${esc(pl.code)}</code>`
    : origin === "saved" ? `Loaded by <code>${esc(madeLine.replace(/^model\s*=\s*/, ""))}</code>`
      : "Built with <b>Edit model</b> (<code>fa.Model(…)</code>)";
  const editList = edits.map((e) => e.code.split("\n").filter((l) => /^model/.test(l))
    .map((l) => `<li><code>${esc(l.length > 90 ? l.slice(0, 88) + "…" : l)}</code></li>`).join("")).join("");
  // what it fits, by kind, and the free parameters of each component
  const free = {};
  M.params.forEach((p) => { if (!p.fixed && !p.tie) { const c = p.name.split(".")[0]; free[c] = (free[c] || 0) + 1; } });
  const rows = MSUM_KINDS.map(([k, label]) => {
    const cs = M.components.filter((c) => msumKind(c) === k);
    if (!cs.length) return "";
    const lines = [...new Set(cs.flatMap((c) => c.lines || []))];
    let what;
    if (k === "feii") what = cs.map((c) => `template, ${Math.max(0, (c.n_params || 2) - 2)} groups`).join("; ");
    else if (k === "broad") what = `${cs.length} component${cs.length > 1 ? "s" : ""}: ${lines.map(prettyLine).join(", ")}`;
    else if (lines.length) what = `${lines.length} line${lines.length > 1 ? "s" : ""}: ${lines.slice(0, 6).map(prettyLine).join(", ")}${lines.length > 6 ? ", …" : ""}`;
    else what = cs.map((c) => compDesc(c)).join(", ");
    const ties = cs.filter((c) => c.tied_to).map((c) => `${esc(c.name)} moves with ${esc(c.tied_to)}`);
    const n = cs.reduce((a, c) => a + (free[c.name] || 0), 0);
    return `<tr><td><span class="sw" style="background:${compColor(cs[0], M.components)}"></span>${label}</td>` +
      `<td>${esc(what)}${ties.length ? ` <span class="muted">· ${ties.join(", ")}</span>` : ""}</td><td class="num mono">${n}</td></tr>`;
  }).join("");
  const bar = M.components.filter((c) => free[c.name]).map((c) =>
    `<span style="flex:${free[c.name]};background:${compColor(c, M.components)}" title="${esc(c.name)}: ${free[c.name]} free parameters"></span>`).join("");
  const recipe = made >= 0 ? [S.cells[made].code.split("\n").filter((l) => !/^r = fits = |^model$/.test(l)).join("\n")]
    .concat(edits.map((e) => e.code)).join("\n") : "";
  const html = `<div class="msum-head"><div class="grow"><div class="msum-eyebrow">The model in use</div><div class="msum-title">${esc(title)}</div>` +
    `<div class="msum-sub">${esc(bits.join(" · "))}</div></div>${state}</div>` +
    (pending ? `<div class="callout warn msum-pending"><div class="grow">${pendingText} <b>Use preset</b> builds the new one, and the fit then uses it.</div><button class="btn small primary" id="msum-apply">Use preset</button></div>` : "") +
    `<p class="msum-from">${from}${edits.length ? `, then changed ${edits.length} time${edits.length > 1 ? "s" : ""}:` : origin === "preset" ? ", unchanged." : "."}</p>` +
    (editList ? `<ul class="msum-edits">${editList}</ul>` : "") +
    `<div class="msum-budget"><div class="msum-bar">${bar}</div><span class="mono small muted">${M.n_free} free parameters</span></div>` +
    `<table class="msum-tbl"><thead><tr><th>It fits</th><th></th><th class="num">free</th></tr></thead><tbody>${rows}</tbody></table>` +
    (recipe ? `<details class="codebox msum-code"><summary>Python code of this model</summary><pre class="code">${esc(recipe)}</pre></details>` : "");
  if (box.dataset.sig === html) return;
  const open = box.querySelector("details") && box.querySelector("details").open;
  box.dataset.sig = html; box.innerHTML = html;
  if (open && box.querySelector("details")) box.querySelector("details").open = true;
  if ($("msum-apply")) $("msum-apply").onclick = () => makePreset();
}
// what is the same in all epochs: one row per component, fluxes and shape
function renderSharing() {
  const SH = mode() === "multi" ? S.wf.sharing : null, M = S.model;
  $("sharecard").classList.toggle("hidden", !(SH && SH.length && M));
  if (!SH || !SH.length || !M) return;
  const override = $("sharedin").value.trim();
  const sig = JSON.stringify([SH, override, !!S.busy]);
  if ($("sharetbl").dataset.sig === sig) return;
  $("sharetbl").dataset.sig = sig;
  const cell = (r, part) => {
    const v = r[part], d = r["default_" + part];
    if (v === null) return '<td class="muted">—</td>';
    const b = (val, label) => `<button data-c="${esc(r.name)}" data-part="${part}" data-v="${val ? 1 : 0}" class="${v === val ? "on" : ""}" ${S.busy || override ? "disabled" : ""}>${label}</button>`;
    const per = r.host && part === "flux" ? "amplitude per epoch" : "per epoch";
    return `<td><span class="seg mini">${b(true, "the same")}${b(false, per)}</span>${v !== d ? '<span class="chg" title="not the default">changed</span>' : ""}</td>`;
  };
  $("sharetbl").innerHTML = `<thead><tr><th>Component</th><th>Fluxes</th><th>Shape <span class="muted small">shift, width, profile</span></th></tr></thead><tbody>` +
    SH.map((r) => { const c = M.components.find((x) => x.name === r.name);
      return `<tr><td><span class="sw" style="background:${c ? compColor(c, M.components) : "#999"}"></span> <b class="mono">${esc(r.name)}</b>` +
        `${r.lines && r.lines.length ? ` <span class="muted small">${r.lines.map(prettyLine).join(", ")}${c && c.lines.length > r.lines.length ? ", …" : ""}</span>` : ""}</td>${cell(r, "flux")}${cell(r, "shape")}</tr>`; }).join("") + "</tbody>";
  $("sharetbl").querySelectorAll("button[data-c]").forEach((b) => (b.onclick = () => post("/api/wf/sharing", { component: b.dataset.c, part: b.dataset.part, shared: b.dataset.v === "1" })));
  const anchors = SH.filter((r) => r.flux).map((r) => r.name), scaledHost = SH.filter((r) => r.host && r.flux === false);
  $("sharenote").innerHTML = override ? `<span class="warnword">A name pattern is set in the Fit step (Advanced options): it replaces this table.</span>`
    : anchors.length ? `The flux scale of each epoch is measured on: <b>${anchors.map(esc).join(", ")}</b>.` +
      (scaledHost.length ? ` The host galaxy keeps one spectrum, with its own amplitude in every epoch (different apertures or seeing).` :
        SH.some((r) => r.host) ? ` <span class="muted">Spectra through different apertures or in different seeing contain different amounts of host light: let its amplitude vary per epoch.</span>` : "")
      : `<span class="warnword">No fluxes are shared, so nothing fixes the flux scale between the epochs: share the narrow lines, or fit each epoch on its own.</span>`;
  $("sharereset").disabled = !S.wf.sharing_custom || !!S.busy;
}
function sharingSummary() {
  const SH = S.wf.sharing || [];
  if ($("sharedin").value.trim()) return `Shared between the epochs: the name pattern <code>${esc($("sharedin").value.trim())}</code> (Advanced options).`;
  const names = (rows) => rows.map((r) => esc(r.name)).join(", ");
  const both = SH.filter((r) => r.flux && r.shape !== false), f = SH.filter((r) => r.flux && r.shape === false), sh = SH.filter((r) => r.shape && !r.flux);
  const sc = SH.filter((r) => r.host && r.flux === false);
  const parts = [both.length ? "fluxes and shape of " + names(both) : "", f.length ? "fluxes of " + names(f) : "",
    sh.length ? "shape of " + names(sh.filter((r) => !(r.host && r.flux === false))) : "",
    sc.length ? "the spectrum of " + names(sc) + " (its amplitude per epoch)" : ""].filter((x) => x && !/of $/.test(x));
  return `<b>The same in all epochs</b>: ${parts.join("; ") || "nothing"}` +
    `${both.length || f.length ? "" : ' <span class="warnword">(no fluxes: nothing fixes the flux scale)</span>'}. Everything else is fitted in every epoch. <a href="#" class="textlink" data-go="model">change</a>`;
}
async function loadGuess(force) {
  if (!S.model || !prepared()) return;
  const sig = JSON.stringify([S.model, prepared().id, S.wf.reference]);
  if (!force && sig === guessSig && loadGuess.data) { drawGuess(); return; }
  guessSig = sig;
  $("gtitle").textContent = "Initial guess · computing…";
  let result;
  try { result = await api("/api/wf/guess"); } catch (e) { result = null; }
  if (guessSig !== sig || !S.model || !prepared()) return;
  loadGuess.data = result;
  drawGuess();
}
function drawGuess() {
  const d = loadGuess.data;
  if (!S.model || !d || !d.x) {
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
  if (!r || !S.models.includes(r.model) || !MB.confirmReplace()) return;
  $("presetsel").value = r.model;
  setPresetDefaults(r.model);
  if (r.broad_components) $("bcount").textContent = r.broad_components;
  toast(`${CLASS_NAME[r.label] || r.label}${r.uncertain ? " (uncertain)" : ""}: using ${r.model}`, 4000);
  makePreset();
}
function showClassDialog() {
  const kind = showClassDialog.kind || "diagrams";
  document.querySelectorAll("#classdlgkind button").forEach((b) => b.classList.toggle("on", b.dataset.k === kind));
  const url = (fmt, dl) => `/api/wf/classify.${fmt}?k=0&kind=${kind}&stage=preliminary&v=${S.cells.length}${dl ? "&dl=1" : ""}`;
  $("classdlgname").textContent = classInfo ? ` · ${classInfo.name}` : "";
  $("classdlgimg").src = withToken(url("png"));
  $("classdlgpng").onclick = () => download(url("png", true));
  $("classdlgpdf").onclick = () => download(url("pdf", true));
  $("classdlg").classList.remove("hidden");
}

function editFitComponent(parameter) {
  selComp = parameter.split(".")[0];
  go("model");
  if (S.model && prepared()) MB.openCurrent(selComp);
}
function initModel() {
  $("presetsel").onchange = () => { setPresetDefaults($("presetsel").value); syncOptionAvailability(); renderModelSum(); };
  $("bminus").onclick = () => { $("bcount").textContent = Math.max(1, +$("bcount").textContent - 1); renderModelSum(); };
  $("bplus").onclick = () => { $("bcount").textContent = Math.min(3, +$("bcount").textContent + 1); renderModelSum(); };
  for (const id of ["hostsel", "contsel"]) $(id).addEventListener("change", () => renderModelSum());
  document.querySelectorAll("#optbar input[data-opt]").forEach((c) => c.addEventListener("change", () => renderModelSum()));
  document.querySelectorAll("#startseg button").forEach((b) => (b.onclick = () => setStartMode(b.dataset.s)));
  $("usepreset").onclick = () => { if (MB.confirmReplace()) makePreset(); };
  $("startempty").onclick = () => MB.openNew();
  $("openmb").onclick = () => MB.openCurrent(selComp);
  $("savemodel").onclick = saveModel;
  $("saveform").onsubmit = (e) => submitSave(e);
  $("closesave").onclick = $("cancelsave").onclick = () => $("savedlg").classList.add("hidden");
  $("savedlg").onclick = (e) => { if (e.target.id === "savedlg") $("savedlg").classList.add("hidden"); };
  $("sv-name").oninput = $("sv-folder").oninput = savePreview;
  $("sv-lib").onclick = () => { $("sv-folder").value = ""; savePreview(); };
  $("dlmodel").onclick = () => (S.model ? download("/api/model.json") : showError("Create a model first."));
  $("loadsaved").onclick = async () => {
    const path = $("savedsel").value, m = savedModels.find((x) => x.path === path);
    if (path && MB.confirmReplace() && (await post("/api/model", { action: "load", path }))) { startMode = "saved"; selComp = null; toast(`Model ${m ? m.name : path} loaded`); }
  };
  $("loadfile").onclick = () => $("modelfile").click();
  $("modelfile").onchange = async (e) => {
    const f = e.target.files[0]; e.target.value = "";
    if (!f || !MB.confirmReplace()) return;
    const name = f.name.replace(/\.json$/i, "");
    if (await post("/api/model", { action: "upload", name, text: await f.text() })) {
      startMode = "saved"; selComp = null; toast(`Model ${name} loaded and added to your saved models`); refreshSaved();
    }
  };
  document.querySelectorAll("#classdlgkind button").forEach((b) => (b.onclick = () => { showClassDialog.kind = b.dataset.k; showClassDialog(); }));
  $("closeclass").onclick = () => $("classdlg").classList.add("hidden");
  $("classdlg").onclick = (e) => { if (e.target.id === "classdlg") $("classdlg").classList.add("hidden"); };
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("classdlg").classList.add("hidden"); });
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
    from_trial: $("fromtrial").checked, wavelength_offsets: $("offsel").value === "1",
    epoch_broadening: $("broadsel").value === "1",
    sample_model: $("smodel").value, prune_feii: $("sprune").checked, ...(mode() === "sample" ? { n_jobs: +$("snjobs").value || -1 } : {}),
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
  if (P) rows.push([m === "sample" && L.n > 1 ? "Prepared preview" : "Prepared", [P.rest_frame ? "rest frame" : "observed frame", P.ebv != null ? `E(B−V) ${fmtNum(P.ebv, 3)}` : "no Galactic extinction",
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
      $("hostepoch").checked = false;      // now in the sharing table of the Model step $("rescale").checked = !!o.rescale_errors;
      if (o.shared_shape !== undefined) $("sharedshape").checked = !!o.shared_shape;
      setErr(o.errors || "covariance"); $("scaleerr").checked = o.scale_errors !== false;
      $("modelunc").checked = !!o.uncertainty; $("absorb").checked = !!o.absorbers;
      if (o.sample_model) $("smodel").value = o.sample_model;
      $("sprune").checked = !!o.prune_feii;
      if (o.from_trial !== undefined) $("fromtrial").checked = !!o.from_trial;
      if (o.wavelength_offsets !== undefined && o.epochs === "joint") $("offsel").value = o.wavelength_offsets ? "1" : "0";
      if (o.epoch_broadening !== undefined && o.epochs === "joint") $("broadsel").value = o.epoch_broadening ? "1" : "0";
      fitRefPending = o.reference;
    }
  }
  const joint = m === "multi" && fitForm().epochs === "joint";
  $("epochsbox").classList.toggle("hidden", m !== "multi");
  // one epoch first: the trial fit, and whether the epochs start from it
  const T = S.wf.trial, multi = m === "multi" && n > 1;
  $("trialbox").classList.toggle("hidden", !multi);
  if (multi && L && $("trialsel").dataset.sig !== String(L.id)) {
    $("trialsel").dataset.sig = String(L.id);
    $("trialsel").innerHTML = '<option value="auto">the epoch with the highest S/N</option>' + L.rows.map((r) => `<option value="${r.k}">#${r.k}${r.date ? " · " + r.date : ""}</option>`).join("");
  }
  const lastCell = S.cells[S.cells.length - 1], trialRunning = S.busy && lastCell && lastCell.code.startsWith("# trial fit");
  $("trialrun").disabled = !!S.busy || !prepared() || !S.model;
  $("trialrun").textContent = trialRunning ? "Fitting…" : T ? "Fit this epoch again" : "Fit this epoch";
  $("trialnote").innerHTML = !T ? "" : `Tried on <b>#${T.k}${T.date ? " · " + esc(T.date) : ""}</b>: χ²<sub>ν</sub> ${T.rchi2.toFixed(2)}.` +
    (T.model_changed ? ' <span class="warnword">The model has changed since: try it again.</span>' : "") +
    (T.active ? ' <a href="#" class="textlink" id="trialopen">Look at the result →</a>' : " The fit of all epochs came after it.");
  if ($("trialopen")) $("trialopen").onclick = (e) => { e.preventDefault(); go("results"); };
  $("fromtrialbox").classList.toggle("hidden", !(multi && T));
  if (T) $("fromtrialnote").textContent = `of #${T.k}${T.date ? " · " + T.date : ""}${T.model_changed ? " (the parameters the changed model still has)" : ""}`;
  $("samplebox").classList.toggle("hidden", m !== "sample");
  $("uncbox").classList.toggle("hidden", m === "classify" || m === "sample");
  $("advbox").classList.toggle("hidden", m === "classify" || m === "sample");
  $("uncbox-model").classList.toggle("hidden", m !== "single");
  $("uncbox-abs").classList.toggle("hidden", m !== "single");
  ["adv-rescale", "adv-shared"].forEach((id) => $(id).classList.toggle("hidden", !joint));
  $("adv-jobs").classList.toggle("hidden", m !== "multi");
  $("jointopts").classList.toggle("hidden", !joint);
  if (joint) {
    const P = prepared(), spread = P && P.z_spread_kms;
    $("offnote").innerHTML = ($("offsel").value === "1"
      ? "A velocity of up to ±150 km/s is fitted for every epoch (the reference epoch is 0), measured on the shared narrow lines like the flux scale. It absorbs differences of wavelength calibration and of the redshift each file was moved to the rest frame with."
      : "The epochs are taken to be on the same wavelength scale.") +
      (spread > 15 && !P.z_one ? ` <b class="warnword">The redshifts of the files differ by ${Math.round(spread)} km/s</b>, so the epochs are shifted against each other${$("offsel").value === "1" ? ": the offsets take that out." : ": fit the offsets, or choose one redshift for all in Prepare."}` : "");
  }
  if (joint) {
    const L = loaded(), res = L ? L.rows.map((r) => r.resolution).filter((v) => v) : [];
    const lo = res.length ? Math.min(...res) : 0, hi = res.length ? Math.max(...res) : 0;
    const known = res.length === (L ? L.rows.length : 0) && !(prepared() || {}).resolution_given;
    $("broadnote").innerHTML = ($("broadsel").value === "1"
      ? "An extra line width (FWHM in quadrature, up to 300 km/s; an epoch may also be sharper) is fitted for every epoch against the reference, measured on the shared narrow lines. Each epoch's widths are then intrinsic."
      : "Every epoch is fitted at the resolution of its spectrum" + (known ? (hi > 1.05 * lo ? ` (from the files: ${Math.round(lo)}–${Math.round(hi)} km/s).` : ` (${Math.round(lo)} km/s from the files).`) : ".")) +
      ($("broadsel").value !== "1" && !known ? ' <b class="warnword">The resolution of each epoch is not known from the files:</b> if it differs (slit, seeing, another spectrograph), the shared narrow lines fit some epochs badly and bias their flux scales; fit the broadening.' : "");
  }
  if (joint) { $("sharesum").innerHTML = sharingSummary(); const a = $("sharesum").querySelector("[data-go]"); if (a) a.onclick = (e) => { e.preventDefault(); go("model"); }; }
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
  const afterTrial = m === "multi" && fitDone() && fitDone().kind === "single";       // only one epoch was fitted so far
  $("runfit").textContent = S.busy ? "Running…" : `▶ ${runLabel()}${fitDone() && !changed && !afterTrial ? " again" : ""}`;
  $("runfit").disabled = !!S.busy || !!blocked;
  $("runnote").textContent = blocked || (S.busy ? "" : afterTrial ? "Only the trial epoch is fitted so far." : fitDone() && changed ? "The settings changed since the last run." : "");
  $("runnote").title = $("runnote").textContent;      // the footer may cut it short
  renderProgress();
  $("toresults").disabled = !fitDone() || !!S.busy;
  // one primary button in the footer: Run until a fit of these settings exists, then the results
  const current = !!fitDone() && !changed && !afterTrial && !S.busy;
  $("runfit").classList.toggle("primary", !current);
  $("toresults").classList.toggle("primary", current);
}
function renderProgress() {
  const F = fitDone(), list = $("proglist");
  const cell = S.cells[S.cells.length - 1];
  const isFit = cell && (cell.code.startsWith("# fit") || cell.code.startsWith("# trial fit"));
  const running = S.busy && isFit;
  const failed = !running && S.error && isFit;
  if (running && S.progress) {
    const p = S.progress, i = runLog.findIndex((x) => x.task === p.task);
    const entry = { task: p.task, done: p.done, total: p.total, detail: p.detail, t: p.time - p.start };
    if (i < 0) runLog.push(entry); else runLog[i] = entry;
  }
  // the progress card while running (and when it failed); the summary card when done
  $("progcard").classList.toggle("hidden", !running && !failed);
  $("donecard").classList.toggle("hidden", running || failed || !F);
  $("epochtblbox").classList.toggle("hidden", running || failed || !(F && F.epochs && F.epochs.length > 1 && F.chi2 != null));
  // the fit of one spectrum is shown as it converges (fitlive.js); runs of many spectra keep the list
  const liveFit = !!running && (mode() === "single" || cell.code.startsWith("# trial fit"));
  if (window.FL) FL.run(liveFit, running ? S.now - cell.start : 0);
  // a sample: one tile per spectrum as the processes finish them (sampletiles.js), kept once the fit is done
  const sampleRun = !!running && mode() === "sample" && cell.code.startsWith("# fit");
  if (window.ST) ST.run(sampleRun, !running && !failed && !!F && F.kind === "sample", running ? S.now - cell.start : 0);
  $("proglist").classList.toggle("hidden", liveFit);
  $("progbar").parentElement.classList.toggle("hidden", liveFit);
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
  $("donecard").innerHTML = `<div class="donehead"><span class="tick">✓</span><div class="grow"><b>${head}</b>${body}</div></div>`;
  if ($("limopen")) $("limopen").onclick = () => editFitComponent(renderDone.first);
  if (F.epochs && F.epochs.length > 1 && F.chi2 != null) {
    $("epochtbl").innerHTML = '<thead><tr><th>Epoch</th><th>Date</th><th>Spectrum</th><th class="num">χ²ᵣ</th><th class="num">Flux scale</th><th>Parameters at a limit</th></tr></thead><tbody>' +
      F.epochs.map((e) => `<tr><td class="mono">#${e.k}${e.reference ? '<span class="badge ref">REF</span>' : ""}</td><td class="mono">${esc(e.date || "—")}</td>` +
        `<td class="mono">${esc(e.name)}</td><td class="num">${e.failed ? "failed" : e.rchi2.toFixed(3)}</td><td class="num">${e.scale ? e.scale.toFixed(3) : "—"}</td>` +
        `<td class="mono small">${(e.limits || []).map((x) => `<span class="badge warn">${esc(x)}</span>`).join(" ")}</td></tr>`).join("") + "</tbody>";
  }
}
function initFit() {
  document.querySelectorAll('input[name="epochs"], input[name="errs"]').forEach((r) => (r.onchange = renderFit));
  ["refsel", "njobs", "sharedin", "calsel", "sharedshape", "hostepoch", "rescale", "scaleerr", "modelunc", "absorb", "smodel", "snjobs", "sprune", "fromtrial", "offsel", "broadsel"].forEach((id) => ($(id).onchange = () => { renderFit(); if (id === "sharedin" || id === "sharedshape") $("comps").dataset.sig = ""; }));
  $("sharereset").onclick = () => post("/api/wf/sharing", { reset: true });
  $("trialrun").onclick = () => { runLog = []; post("/api/wf/fit", Object.assign(fitForm(), { trial: $("trialsel").value })); };
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
// the fit the views of one spectrum show: the selected spectrum of a sample, else the selected epoch
const fitK = () => (fitDone() && fitDone().kind === "sample" ? sk : rk);
function resultViews() {
  const F = fitDone(), kind = F && F.kind;
  if (!F) return [];
  if (kind === "classify") return ["class"];
  if (kind === "sample") return ["fit", "atlas", "sdiag", "figure", "checks", "lines"];
  const v = ["fit"];
  if ((kind === "joint" || kind === "fits") && (F.epochs || []).length > 1) v.push("atlas");
  v.push("anatomy");
  if (resultData && resultData.has_feii) v.push("feii");     // Results › Fe II (feii.js)
  v.push("degen", "checks", "figure", "lines");
  if (resultData && resultData.corner_params && resultData.corner_params.length) v.push("posterior");
  v.push("class");                       // the classification, or a button to run it
  if (S.wf.has_ms) v.push("ms");
  if (kind === "joint" || (kind === "fits" && (F.epochs || []).length > 1)) v.push("time");   // separate fits: fa.EpochSet
  if (["joint", "fits"].includes(kind) && S.wf.corr) v.push("corr");
  return v;
}
const VIEW_HINT = { feii: "r.feii() · r.params (feii.*)", anatomy: "r.model_flux(components=True)", degen: "r.cov · r.samples · r.params", sdiag: "sample.table", corr: "ep = fa.epoch_table(me) · fa.epoch_correlations(ep)", checks: "fa.analyze_residuals(r) · fa.robustness(r) · fa.compare_fits(a, b) · fa.prune_feii(r) · fa.simulation_check(r)", lines: "r.plot_lines(['Hb', 'Ha'], residuals=True)", ms: "r.main_sequence()", figure: "r.plot(xlim=…, group=…, label_lines=…)", posterior: "r.samples", class: "fa.classify_fit(r) · c.plot()", time: "me.light_curve() · me.reverberation(line)" };
function showView() {
  const F = fitDone(), kind = F && F.kind, views = resultViews();
  if (!views.includes(rv)) rv = views[0] || "fit";
  document.querySelectorAll("#rnav button").forEach((b) => { b.classList.toggle("hidden", !views.includes(b.dataset.v)); b.classList.toggle("on", b.dataset.v === rv); });
  $("rnav").classList.toggle("hidden", views.length < 2);
  $("res-fit").classList.toggle("hidden", !F || rv !== "fit" || kind === "classify");
  $("res-sample").classList.toggle("hidden", kind !== "sample" || !["fit", "figure", "checks", "lines"].includes(rv));
  $("res-checks").classList.toggle("sample", kind === "sample");
  $("res-sdiag").classList.toggle("hidden", rv !== "sdiag");
  $("res-checks").classList.toggle("hidden", rv !== "checks");
  $("res-corr").classList.toggle("hidden", rv !== "corr");
  if (rv === "corr") setTimeout(renderCorr, 0);
  $("res-figure").classList.toggle("hidden", rv !== "figure");
  $("res-lines").classList.toggle("hidden", rv !== "lines");
  $("res-anatomy").classList.toggle("hidden", rv !== "anatomy");
  $("res-degen").classList.toggle("hidden", rv !== "degen");
  $("res-feii").classList.toggle("hidden", rv !== "feii");
  $("res-atlas").classList.toggle("hidden", rv !== "atlas");
  $("res-post").classList.toggle("hidden", rv !== "posterior");
  $("res-class").classList.toggle("hidden", rv !== "class");
  $("res-time").classList.toggle("hidden", rv !== "time");
  $("res-ms").classList.toggle("hidden", rv !== "ms");
  $("etabs").classList.toggle("hidden", rv === "class" || rv === "time" || rv === "ms" || rv === "corr" || rv === "atlas");
  $("res-hint").textContent = rv === "corr" && kind === "fits" ? "ep = fa.epoch_table(fits) · fa.epoch_correlations(ep)"
    : rv === "atlas" ? (kind === "sample" ? "sample.table · sample.load(k)" : kind === "joint" ? "me.epochs[k] · me.epochs[k].key_results()" : "fits[k] · fits[k].key_results()") : VIEW_HINT[rv] || (kind === "joint" ? "me.epochs[k].plot() · me.epochs[k].lines" : kind === "fits" ? "fits[k].plot() · fits[k].lines"
    : kind === "sample" ? "sample.table · sample.load(k).plot()" : "r.plot() · r.lines");
  renderDownloads();
}
function renderDownloads() {
  const F = fitDone(), kind = F?.kind;
  $("dlwrap").classList.toggle("hidden", !F);
  const fitLike = ["single", "joint", "fits"].includes(kind);
  $("dl-sample").classList.toggle("hidden", kind !== "sample");
  const epochs = F?.epochs || [], anyFit = epochs.some((e) => !e.failed);
  for (const id of ["dl-lines", "dl-fits", "dl-comp", "dl-plots", "dl-report"]) {
    const item = $(id);
    item.classList.toggle("hidden", !fitLike);
    item.disabled = !anyFit || (id === "dl-comp" && !!epochs.find((e) => e.k === rk)?.failed);
  }
  $("dl-comp").dataset.dl = `/api/wf/components.csv?k=${rk}`;
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
  if (rv === "atlas") return drawAtlas();
  if (rv === "time") return loadTime();
  if (rv === "ms") return loadMainSeq();
  if (kind === "sample" && S.sample && SG.id !== S.sample.id) {     // a new sample: open its first fitted spectrum
    Object.assign(SG, { id: S.sample.id, show: "all", cls: null, q: "", sort: null, dir: 1 });
    $("ssearch").value = "";
    sk = Math.max(0, S.sample.status.findIndex((st) => st !== "failed"));
  }
  if (rv === "sdiag") {
    try { samplePts = await api("/api/wf/sample_points"); } catch (e) { samplePts = null; }
    return drawSampleDiagrams();
  }
  const sig = JSON.stringify([kind, F.chi2, F.time, F.n, kind === "sample" ? S.sample?.id : null,
    kind === "sample" ? sk : rk]);
  if (!force && sig === resultSig && resultData) { renderResults(); return; }
  resultSig = sig;
  if (kind === "sample") {
    $("etabs").innerHTML = "";
    renderSampleResults();
  }
  if (kind === "sample" || !resultData || resultData.k !== rk) {
    $("rplot").innerHTML = '<div class="empty">Loading the fit…</div>'; $("rlegend").innerHTML = ""; $("rsummary").innerHTML = "";
  }
  let detail;
  try { detail = await api(kind === "sample" ? `/api/wf/sample?k=${sk}` : `/api/wf/result?k=${rk}`); } catch (e) { detail = null; }
  if (resultSig !== sig) return;
  resultData = detail;
  showView();
  renderResults();
  if (kind !== "sample" && resultData && !resultData.error) loadMore(resultData);
  $("residcard").classList.toggle("hidden", !["joint", "fits", "single"].includes(kind));
  $("robcard").classList.toggle("hidden", !(S.wf.robust && kind === "single"));
  $("cmpcard").classList.toggle("hidden", !(S.wf.compare && kind === "single"));
  renderFeii();
  renderSim();
  renderResid();
  renderChecks();
}
function renderResults() {
  const F = fitDone();
  const kind = F && F.kind;
  if (!resultData || kind === "classify") return;
  const catText = kind !== "sample" && S.wf.catalog && S.wf.catalog.text;
  $("rcatnote").classList.toggle("hidden", !catText);
  if (catText) $("rcatnote").textContent = "Catalogs: " + catText;
  const R = resultData;
  if (R.epochs) {
    $("etabs").innerHTML = R.epochs.length > 1 ? R.epochs.slice().sort((a, b) => String(a.date || a.name).localeCompare(String(b.date || b.name))).map((e) => `<button data-k="${e.k}" class="${e.k === R.k ? "on" : ""} ${e.failed ? "failed" : ""}"><b>#${e.k}</b>${esc(e.date || e.name)}${e.reference ? '<span class="badge ref">REF</span>' : ""}</button>`).join("") : "";
    $("etabs").querySelectorAll("button").forEach((b) => (b.onclick = () => { rk = +b.dataset.k; loadResult(true); }));
  }
  $("rtitle").textContent = kind === "sample" ? `${R.name}${R.cls ? " · " + R.cls : ""}${R.preset ? " · " + R.preset : ""}` : (R.name || "");
  $("res-fit").classList.toggle("sgfailed", kind === "sample" && !!R.error);
  if (kind === "sample" && rv === "checks") loadSampleChecks(sk);
  if (R.error) {
    const note = kind === "sample" && S.sample ? S.sample.notes[R.k] : "";
    $("rplot").innerHTML = `<div class="empty"><p><b>This spectrum was not fitted.</b></p>${note && note !== R.error ? `<p>${esc(note)}</p>` : ""}<p class="mono small muted">${esc(R.error)}</p></div>`;
    $("rtbl").innerHTML = ""; $("rlegend").innerHTML = ""; $("rsummary").innerHTML = "";
    if (rv === "figure") showFigure();
    return;
  }
  renderSummary();
  if (rv === "fit") drawResultPlot();
  if (rv === "anatomy") drawAnatomy();
  if (rv === "degen") drawDegen();
  if (rv === "feii" && window.FX) FX.show(rk, `${rk}|${R.fit && R.fit.chi2}|${R.name}`);
  if (rv === "figure") showFigure();
  if (rv === "lines") { if (window.LV) LV.show(fitK(), `${R.fit && R.fit.chi2}|${R.name}|${S.cells.length}`); if ($("lv-paper").open) showLines(); }
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
  const pts = good.map((p) => ({ x: p.r_feii, y: p.fwhm / 1000, xerr: p.r_feii_err, yerr: p.fwhm_err == null ? null : p.fwhm_err / 1000, k: p.k,
    sel: !many || p.k === rk, color: many ? shade(p) : C.accent,
    tip: `${many ? `#${p.k} ${p.date || p.name}: ` : ""}R_FeII = ${p.r_feii.toFixed(2)}, FWHM = ${Math.round(p.fwhm)} km/s, ${p.type || p.population}` }));
  const path = many ? good.slice().sort((a, b) => (a.mjd ?? a.k) - (b.mjd ?? b.k)) : [];
  let xlim = [0, 3], ylim = [0, 16];
  $("mszoomwrap").classList.toggle("hidden", good.length < 2);
  if (good.length > 1 && $("mszoom").checked) {
    const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);       // the points set the zoom; a huge error bar runs off the plot
    const dx = Math.max(Math.max(...xs) - Math.min(...xs), 0.05), dy = Math.max(Math.max(...ys) - Math.min(...ys), 0.3);
    xlim = [Math.max(0, Math.min(...xs) - 0.15 * dx), Math.max(...xs) + 0.15 * dx];
    ylim = [Math.max(0, Math.min(...ys) - 0.15 * dy), Math.max(...ys) + 0.15 * dy];
  }
  const curves = d.grid.map((g) => ({ x: g.x, y: g.y.map((v) => v / 1000), dash: g.y[0] === 4000 && g.y[1] === 4000 ? null : "2 3" }));
  if (path.length > 1) curves.push({ x: path.map((p) => p.r_feii), y: path.map((p) => p.fwhm / 1000), dash: "1 2" });
  plotScatter($("msplot"), { xlim, ylim, pointsOnly: good.length > 1 && $("mszoom").checked, curves, texts: d.types.map((tt) => ({ x: tt.x - 0.12, y: tt.y / 1000 + 0.6, t: tt.t })), points: pts,
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
  // a joint fit: how well the flux scales of the epochs are fixed; above 2% the library warns too
  const cal = F && F.kind === "joint" ? F.calibration_error : null;
  if (cal !== null && cal !== undefined) {
    const bad = cal > 0.02;
    tiles.push(`<div class="tile fitq ${bad ? "warn" : "ok"}" title="${bad ? "This noise is added to every light curve and can hide real variability." : "The error of the flux scale of each epoch, from the shared narrow lines."}">` +
      `<small>Flux scale between epochs</small><b>${(100 * cal).toFixed(1)}%</b><span>median error per epoch</span></div>`);
  }
  // the redshift of the narrow lines of all epochs together, and how far the epochs were off in wavelength
  if (F && F.kind === "joint" && F.z_best) {
    const [zb, ze] = F.z_best, O = F.offsets;
    tiles.push(`<div class="tile" title="The redshift of the reference epoch corrected by the shift of the shared narrow lines. Use it as the redshift of the object, or to prepare the spectra again with one redshift."><small>Redshift, narrow lines</small><b>${zb.toFixed(5)}</b>` +
      `<span>${ze ? "± " + ze.toFixed(5) : ""}${O ? ` · offsets ${O.rms.toFixed(0)} km/s rms` : ""}</span></div>`);
  }
  if (F && F.kind === "joint" && F.broadening) {
    const B = F.broadening;
    tiles.push(`<div class="tile" title="The extra line width of each epoch against the reference (FWHM in quadrature; negative: sharper). It is the difference of resolution between the epochs; the widths of each epoch are intrinsic."><small>Broadening between epochs</small><b>${Math.round(B.median)} km/s</b>` +
      `<span>median · from ${Math.round(B.min)} to ${Math.round(B.max)}</span></div>`);
  }
  if (R.key === undefined) tiles.push('<div class="tile wait"><small>Main results</small><b>…</b><span>computing the errors</span></div>');
  else (R.key || []).slice(0, 5).forEach((k) => {
    const [v, e] = String(k.value).split(" ± ");
    const f = sciPM(parseFloat(v), e === undefined ? NaN : parseFloat(e));
    const num = isFinite(parseFloat(v)) ? f : { v: esc(v), e: e ? "± " + esc(e) : "" };
    if (k.scatter && isFinite(parseFloat(v))) {
      // a black-hole mass: the error of the fit and the scatter of the recipe, which is the larger by far
      const fitErr = e === undefined ? null : Math.max(parseFloat(e), 0.01);
      tiles.push(`<div class="tile" title="The first error is that of the fit; the second is the intrinsic scatter of the single-epoch recipe, which applies to every mass made this way."><small>${prettyQuantity(k.name)}</small><b>${parseFloat(v).toFixed(2)}</b>` +
        `<span>${fitErr === null ? "" : `± ${fitErr.toFixed(2)} fit · `}± ${(+k.scatter).toFixed(2)} recipe, dex</span></div>`);
      return;
    }
    const unit = k.unit || (/ flux$/.test(k.name) ? lineFluxUnit(R.flux_unit) : "");
    tiles.push(`<div class="tile"><small>${prettyQuantity(k.name)}</small><b>${num.v}</b><span>${num.e} ${esc(unit)}</span></div>`);
  });
  const noUnit = R.flux_unit == null;
  $("rsummary").innerHTML = tiles.join("") + (R.key && R.key.length > 5 ? `<a href="#" class="textlink moreq" id="moreq">All ${R.key.length} quantities with their errors →</a>` : "") +
    (noUnit ? '<p class="note unitnote">The flux unit is unknown for this spectrum, so luminosities and black-hole masses are unavailable. <a href="#" class="textlink" data-go="prepare">Set it in Prepare</a> and refit.</p>' : "");
  if ($("moreq")) $("moreq").onclick = (e) => { e.preventDefault(); rtab = "continuum"; renderResultTable(); $("rtabs").scrollIntoView({ behavior: "smooth", block: "start" }); };
  // a trial fit of one epoch of a multi-epoch analysis: what it is, and the way on
  const T = S.wf.trial, trial = T && T.active && F && F.kind === "single";
  $("rtrial").classList.toggle("hidden", !trial);
  if (trial) {
    $("rtrial").innerHTML = `<div class="grow"><b>Trial fit of one epoch</b>: #${T.k}${T.date ? " · " + esc(T.date) : ""}, one of ${T.n}. Check it here (Checks, Line profiles); change the model and try again until it is right. ` +
      `The fit of all epochs then starts from this solution.</div><button class="btn small" id="trialmodel">Change the model</button><button class="btn primary small" id="trialall">Fit all ${T.n} epochs →</button>`;
    $("trialmodel").onclick = () => go("model");
    $("trialall").onclick = () => go("fit");
  }
  // parameters that ended at a limit: say so, and lead to where they are changed
  const ep = F && F.epochs ? F.epochs.find((e) => e.k === R.k) || F.epochs[0] : null;
  const lim = (ep && ep.limits) || [];
  $("rlimits").classList.toggle("hidden", !lim.length);
  if (lim.length) {
    $("rlimits").innerHTML = `<div class="grow"><b>${lim.length} parameter${lim.length > 1 ? "s" : ""} ended at a limit</b>: ${lim.slice(0, 6).map((p) => `<code>${esc(p)}</code>`).join(", ")}${lim.length > 6 ? " …" : ""}.` +
      ` The value is then set by the limit, not by the data: widen the limit, fix the parameter, or change the model.</div><button class="btn small" id="limfix">Open ${esc(lim[0].split(".")[0])} in the Model step</button>`;
    $("limfix").onclick = () => editFitComponent(lim[0]);
  }
  // lines narrower than the data can sample
  const und = (ep && ep.undersampled) || [];
  $("rsampling").classList.toggle("hidden", !und.length);
  if (und.length) {
    const unknown = und.some((u) => !u.resolution_known);
    $("rsampling").innerHTML = `<div class="grow"><b>${und.length > 1 ? "Lines" : "A line"} narrower than the data can sample</b>: ${und.slice(0, 4).map((u) => `<code>${esc(u.parameter)}</code> (${u.pixels.toFixed(1)} pixels: ${Math.round(u.fwhm)} km/s against a pixel of ${Math.round(u.pixel)} km/s)`).join(", ")}.` +
      ` Below about 1.2 pixels the flux depends on where the peak falls between two pixels, so it is unreliable.` +
      (unknown ? " The instrumental resolution is not known: set it in Prepare, or raise the lower limit of the width." : " Raise the lower limit of the width.") +
      `</div><button class="btn small" id="sampfix">${unknown ? "Set the resolution in Prepare" : `Open ${esc(und[0].parameter.split(".")[0])} in the Model step`}</button>`;
    $("sampfix").onclick = () => { if (unknown) go("prepare"); else editFitComponent(und[0].parameter); };
  }
}
function drawResultPlot() {
  const R = resultData;
  if (!R || !R.plot) return;
  drawFit($("rplot"), $("rlegend"), R.plot, hiddenSeries, () => drawResultPlot());
  renderZoomChips($("rplot"), $("rzoom"));
  $("rlinkhint").classList.toggle("hidden", !resultViews().includes("lines"));
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
  if (dl && fmt === "png") q.set("dpi", $("fo-dpi").value);
  return `/api/wf/figure.${fmt}?${q}`;
}
// what the figure draws: the data, the model and each kind of component in the model, each with its colour
const FIG_KIND = [["data", "Data", "#9b9a95"], ["model", "Model", "#0b0b0b"], ["continuum", "Continuum", "#52514e"], ["host", "Host galaxy", "#008300"],
  ["balmer_cont", "Balmer continuum", "#86b6ef"], ["balmer_lines", "High-order Balmer lines", "#86b6ef"], ["feii", "Fe II", "#e34948"],
  ["narrow", "Narrow lines", "#1baf7a"], ["coronal", "Coronal lines", "#b8459c"], ["broad", "Broad lines", "#2a78d6"], ["outflow", "Outflow", "#eda100"], ["other", "Other lines", "#898781"]];   // plotting.COLORS
function figKinds() {
  const have = new Set(["data", "model"]);
  ((S.model && S.model.components) || []).forEach((c) => {
    const t = c.type;
    if (t === "Lines") have.add(c.absorption ? "other" : isCoronal(c) ? "coronal" : ["narrow", "broad", "outflow"].includes(c.role) ? c.role : "other");
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
  loadStyle(box, "data-o", "fa-figstyle");       // the remembered colours and styles of these kinds
}
// quick zoom ranges around the main lines of the fit, within its wavelength range
const ZOOMS = [["Mg II", 2650, 2950], ["Hγ", 4250, 4450], ["Hβ", 4650, 5150], ["Hα", 6350, 6800]];
function renderZoomQuick() {
  const R = resultData, range = R && R.plot && R.plot.range;
  if (!range) { $("fo-zoomquick").innerHTML = ""; return; }
  const have = ZOOMS.filter((z) => z[1] >= range[0] - 50 && z[2] <= range[1] + 50);
  const sig = have.map((z) => z[0]).join();
  if ($("fo-zoomquick").dataset.sig === sig) return;
  $("fo-zoomquick").dataset.sig = sig;
  $("fo-zoomquick").innerHTML = have.map((z) => `<button type="button" class="chip" data-z="${z[1]}-${z[2]}">+ ${z[0]}</button>`).join("");
  $("fo-zoomquick").querySelectorAll("[data-z]").forEach((b) => (b.onclick = () => {
    const cur = $("fo-zoom").value.split(",").map((s) => s.trim()).filter(Boolean);
    if (!cur.includes(b.dataset.z)) cur.push(b.dataset.z);
    $("fo-zoom").value = cur.slice(-4).join(", ");
    figChanged();
  }));
}
function showFigure() {
  renderDrawList();
  renderZoomQuick();
  $("fo-zoomsub").classList.toggle("hidden", !$("fo-zoom").value.trim());
  $("fo-labelopts").classList.toggle("hidden", $("fo-labels").value === "0");
  $("fo-title").disabled = $("fo-notitle").checked;
  const sample = fitDone() && fitDone().kind === "sample";
  $("figopts").classList.toggle("hidden", sample);
  const url = withToken(sample ? `/api/sample_fig.png?k=${sk}&kind=fit` : figUrl("png"));
  const none = sample && S.sample && !S.sample.figs[sk];              // a spectrum that was not fitted has no figure
  $("rpub").classList.toggle("hidden", !!none);
  $("figwait").textContent = none ? "This spectrum was not fitted, so it has no figure." : "Drawing…";
  $("figwait").classList.toggle("hidden", !none);
  if (none) return;
  if ($("rpub").dataset.url === url) return;
  $("rpub").dataset.url = url;
  $("figwait").classList.remove("hidden");
  $("rpub").onload = $("rpub").onerror = () => $("figwait").classList.add("hidden");
  figSummary();
  saveStyle($("figopts"), "data-o", "fa-figstyle");
  $("rpub").style.maxWidth = sample ? "" : previewWidth($("fo-width").value);
  $("rpub").src = url;
  refreshCode("figcodebox", "figcode", figCodeUrl());
}
// ---- line profiles: a grid of the ticked lines, redrawn as the choice changes
let lineList = null, lineKey = null, lineSel = [];
function linesUrl(fmt, dl) {
  const q = new URLSearchParams({ k: fitK(), lines: lineSel.join(",") });
  $("lineopts").querySelectorAll("[data-l]").forEach((el) => {
    const v = el.type === "checkbox" ? (el.checked ? "1" : "0") : el.value;
    if (v !== "") q.set(el.dataset.l, v);
  });
  if (dl) q.set("dl", "1"); else q.set("v", S.cells.length);
  if (dl && fmt === "png") q.set("dpi", $("lo-dpi").value);
  return `/api/wf/linegrid.${fmt}?${q}`;
}
// the lines of the figure as toggle chips, by group, with quick choices
function renderLinePick() {
  const L = (lineList && lineList.lines) || [];
  if (!L.length) { $("lo-lines").innerHTML = '<p class="note small">No fitted lines.</p>'; return; }
  const kind = (d) => d.roles.includes("broad") ? "broad" : /^\[Fe(VII|X|XI|XIV)\]/.test(d.line) ? "coronal" : "narrow";
  const chip = (d) => `<label class="lpchip k-${kind(d)}" title="${esc(d.label)}: ${esc(d.roles.join(" + "))}"><input type="checkbox" value="${esc(d.line)}" ${lineSel.includes(d.line) ? "checked" : ""}>` +
    `<span><i></i>${esc(d.label.replace(" λ", " "))}</span></label>`;
  const groups = [["broad", "Broad", L.filter((d) => d.roles.includes("broad"))], ["narrow", "Narrow", L.filter((d) => !d.roles.includes("broad"))]];
  const shown = window.LV && LV.state ? LV.state() : null;
  const above = shown ? (shown.mode === "cmp" ? shown.lines.split(",") : [shown.line]).filter(Boolean) : [];
  $("lo-lines").innerHTML =
    `<div class="lpick-top"><span class="muted small">${plural(lineSel.length, "line", "lines")} chosen</span><span class="grow"></span>` +
    (above.length ? `<button type="button" class="linkbtn" data-pick="above" title="${esc(above.join(", "))}">Same as above</button>` : "") +
    `<button type="button" class="linkbtn" data-pick="none">Clear</button></div>` +
    groups.filter((g) => g[2].length).map(([k, t, ls]) => `<div class="lpick-group"><div class="lpick-head"><span>${t}</span><small>${ls.length}</small>` +
      `<button type="button" class="linkbtn" data-pick="${k}">all</button></div><div class="lpick-chips">${ls.map(chip).join("")}</div></div>`).join("");
  const order = L.map((d) => d.line), set = (sel) => { lineSel = [...new Set(sel)].filter((l) => order.includes(l)).sort((a, b) => order.indexOf(a) - order.indexOf(b)); renderLinePick(); linesChanged(); };
  $("lo-lines").querySelectorAll("input").forEach((el) => (el.onchange = () => set(el.checked ? lineSel.concat(el.value) : lineSel.filter((l) => l !== el.value))));
  $("lo-lines").querySelectorAll("[data-pick]").forEach((b) => (b.onclick = () => {
    const p = b.dataset.pick;
    if (p === "none") set([]);
    else if (p === "above") set(above);
    else set(lineSel.concat((groups.find((g) => g[0] === p) || [, , []])[2].map((d) => d.line)));
  }));
}
async function showLines() {
  const key = fitK() + ":" + S.cells.length;
  if (lineKey !== key) {
    lineKey = key;
    try { lineList = await api(`/api/wf/line_list?k=${fitK()}`); } catch (e) { lineList = { lines: [] }; }
    const names = lineList.lines.map((d) => d.line);
    lineSel = lineSel.filter((l) => names.includes(l));
    if (!lineSel.length && lineList.main) lineSel = [lineList.main];
    renderLinePick();
  }
  const none = !lineSel.length;
  const F = fitDone(), epochs = !!(F && ["joint", "fits"].includes(F.kind) && F.epochs && F.epochs.length > 1);
  $("lo-acrossbox").classList.toggle("hidden", !epochs);
  if (!epochs) $("lo-across").value = "";
  lineSummary();
  saveStyle($("lineopts"), "data-l", "fa-linestyle");
  $("linesempty").classList.toggle("hidden", !none);
  $("rlinesimg").classList.toggle("hidden", none);
  if (none) return;
  const url = withToken(linesUrl("png"));
  if ($("rlinesimg").dataset.url === url) return;
  $("rlinesimg").dataset.url = url;
  $("linewait").classList.remove("hidden");
  $("rlinesimg").onload = $("rlinesimg").onerror = () => $("linewait").classList.add("hidden");
  $("rlinesimg").style.maxWidth = previewWidth($("lo-width").value);
  $("rlinesimg").src = url;
  refreshCode("linecodebox", "linecode", lineCodeUrl());
}
let lineTimer = null;
function linesChanged() { clearTimeout(lineTimer); lineTimer = setTimeout(showLines, 350); }
// ---- side panels: segmented buttons bound to hidden inputs, section summaries, remembered open sections
function initSegs(form, changed) {
  // hidden inputs keep no default of their own through form.reset(): remember it here
  form.querySelectorAll(".seg[data-seg]").forEach((seg) => { const h = $(seg.dataset.seg); h.dataset.def = h.value; });
  form.querySelectorAll(".seg[data-seg]").forEach((seg) => seg.querySelectorAll("button").forEach((b) => (b.onclick = () => {
    $(seg.dataset.seg).value = b.dataset.v;
    syncSegs(form);
    changed();
  })));
  syncSegs(form);
}
function resetForm(form) {
  form.reset();
  form.querySelectorAll(".seg[data-seg]").forEach((seg) => { const h = $(seg.dataset.seg); if (h.dataset.def !== undefined) h.value = h.dataset.def; });
  syncSegs(form);
}
function syncSegs(form) {
  form.querySelectorAll(".seg[data-seg]").forEach((seg) => {
    const v = $(seg.dataset.seg).value;
    seg.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.v === v));
  });
}
const segText = (id) => { const b = document.querySelector(`.seg[data-seg="${id}"] button.on`); return b ? b.textContent : ""; };
const optText = (id) => {
  const el = $(id), o = el && el.options[el.selectedIndex];
  if (!o) return "";
  const g = o.parentElement.label || "";
  return (g && g !== "Any journal" ? g.replace(" / AJ", "") + ", " : "") + o.textContent.split(" · ")[0];
};
function initSections(form) {
  form.querySelectorAll("details.psec[data-sec]").forEach((d) => {
    try { const v = localStorage.getItem("fa-sec-" + d.dataset.sec); if (v !== null) d.open = v === "1"; } catch (e) { /* storage off */ }
    d.addEventListener("toggle", () => { try { localStorage.setItem("fa-sec-" + d.dataset.sec, d.open ? "1" : "0"); } catch (e) { /* storage off */ } });
  });
}
function figSummary() {
  const v = (k) => ($("figopts").querySelector(`[data-o="${k}"]`) || {}).value || "";
  $("sum-fsize").textContent = [optText("fo-width"), segText("fo-height").toLowerCase(), segText("fo-font").toLowerCase()].filter(Boolean).join(" · ");
  const zooms = v("zoom").split(",").filter((s) => s.trim()).length;
  $("sum-frange").textContent = (v("xmin") && v("xmax") ? `${v("xmin")}–${v("xmax")} Å` : "whole fit") + (zooms ? ` · ${zooms} zoom panel${zooms > 1 ? "s" : ""}` : "");
  const shows = [...$("fo-draw").querySelectorAll('input[type=checkbox][data-o^="show_"]')];
  $("sum-fcomp").textContent = shows.length ? `${shows.filter((c) => c.checked).length} of ${shows.length} kinds drawn` : "";
  $("sum-fleg").textContent = `legend ${segText("fo-legend").toLowerCase()} · ${v("label_lines") === "0" ? "no line names" : segText("fo-labels").toLowerCase() + " named"}`;
}
function lineSummary() {
  $("sum-llines").textContent = lineSel.length ? lineSel.map(prettyLine).join(", ") : "none chosen";
  const ov = $("lo-mode").value === "overlay";
  $("sum-lshow").textContent = ov ? ($("lo-across").value === "epochs" ? "one line in every epoch" : "overlaid") : "each in a panel";
  $("sum-laxes").textContent = ov ? ($("lo-win").value ? `± ${$("lo-win").value} km/s` : "velocity") : `${segText("lo-x").toLowerCase()}${$("lo-win").value ? ` · ± ${$("lo-win").value} km/s` : ""}`;
  $("sum-lsize").textContent = [optText("lo-width"), segText("lo-font").toLowerCase()].filter(Boolean).join(" · ");
}
// the figure style is remembered between sessions and spectra (not the ranges, title or zoom, which belong to a spectrum)
const STYLE_SKIP = new Set(["xmin", "xmax", "ymin", "ymax", "title", "notitle", "zoom", "mode"]);
function saveStyle(form, attr, key) {
  const out = {};
  form.querySelectorAll(`[${attr}]`).forEach((el) => {
    const k = el.getAttribute(attr);
    if (!STYLE_SKIP.has(k)) out[k] = el.type === "checkbox" ? el.checked : el.value;
  });
  try { localStorage.setItem(key, JSON.stringify(out)); } catch (e) { /* storage off */ }
}
function loadStyle(form, attr, key) {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(key) || "null"); } catch (e) { saved = null; }
  if (!saved) return false;
  form.querySelectorAll(`[${attr}]`).forEach((el) => {
    const k = el.getAttribute(attr);
    if (!(k in saved) || STYLE_SKIP.has(k)) return;
    if (el.type === "checkbox") el.checked = !!saved[k]; else el.value = saved[k];
  });
  syncSegs(form);
  return true;
}
// the Python code of a figure, fetched when its box is open and the figure changes
async function refreshCode(box, pre, url) {
  if (!$(box).open) return;
  try { $(pre).textContent = (await api(url)).code; } catch (e) { $(pre).textContent = "# " + e.message; }
}
const figCodeUrl = () => figUrl("png").replace("/api/wf/figure.png", "/api/wf/figure_code");
// ---- the Python code of the plot of the other Results views (viewcode.py), following what the view shows
let fitRange = null;
const OWN_CODE = new Set(["figure", "corr", "sdiag"]);     // these views have a code box of their own (or no plot)
function viewCodeUrl() {
  const F = fitDone(), kind = F && F.kind;
  if (!F || OWN_CODE.has(rv) || (kind === "sample" && !["atlas", "lines", "checks"].includes(rv)) || (kind === "classify" && rv !== "class")) return null;
  const q = new URLSearchParams({ view: rv, k: fitK(), v: S.cells.length });     // v: new code after a new fit
  const set = (o) => Object.entries(o).forEach(([k, v]) => { if (v !== null && v !== undefined && v !== "") q.set(k, v); });
  if (rv === "fit") set({ xmin: fitRange && fitRange[0], xmax: fitRange && fitRange[1], hide: [...hiddenSeries].join(",") });
  else if (rv === "anatomy" && AN.F) {
    const whole = AN.view[0] <= AN.F.x[0] && AN.view[1] >= AN.F.x[AN.F.x.length - 1];
    set({ xmin: whole ? null : Math.round(AN.view[0]), xmax: whole ? null : Math.round(AN.view[1]), off: [...AN.off].join(","), solo: AN.solo, peel: AN.peel || null });
  } else if (rv === "feii" && window.FX && FX.state) set(FX.state());
  else if (rv === "lines" && window.LV) set(LV.state());
  else if (rv === "degen") set({ th: DG.th, scope: DG.scope });
  else if (rv === "atlas" && AT.data) set({ order: atShown().map((o) => o.k).join(","), common: AT.common ? 1 : 0, same: AT.same ? 1 : 0 });
  else if (rv === "posterior") set({ params: [...$("cornerparams").querySelectorAll("input:checked")].map((c) => c.value).join(",") });
  else if (rv === "class") { q.set("k", ck); set({ kind: cfigKind }); }
  else if (rv === "ms") set({ zoom: $("mszoom").checked ? 1 : 0 });
  else if (rv === "time") set({ a: LX.A, b: LX.B });
  return `/api/wf/view_code?${q}`;
}
let vcodeUrl = null;
function refreshViewCode(force) {
  const url = viewCodeUrl();
  $("vcodebox").classList.toggle("hidden", !url || page !== "results");
  if (!url || !$("vcodebox").open || (url === vcodeUrl && !force)) return;
  vcodeUrl = url;
  refreshCode("vcodebox", "vcode", url);
}
const lineCodeUrl = () => linesUrl("png").replace("/api/wf/linegrid.png", "/api/wf/lines_code");
// a one-column figure is previewed at about its printed size relative to a two-column one, not blown up
const previewWidth = (w) => (/double/.test(w || "double") ? "100%" : "560px");
const LINE_MODE_NOTE = { grid: "Each line alone: the data with everything else in the model subtracted, its components, their sum and the residuals.",
  overlay: "The profiles on one velocity axis, each scaled to its peak: compare the widths and shapes of lines (broad Hβ against Hα or Mg II). The dots mark the half-maximum points." };
let lineWidthTouched = false;
function setLineMode(m) {
  $("lo-mode").value = m;
  $("lo-modeseg").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.mode === m));
  document.querySelectorAll("#lineopts .lo-grid").forEach((el) => el.classList.toggle("hidden", m !== "grid"));
  document.querySelectorAll("#lineopts .lo-ov").forEach((el) => el.classList.toggle("hidden", m !== "overlay"));
  $("lo-modenote").textContent = LINE_MODE_NOTE[m];
  if (!lineWidthTouched) $("lo-width").value = m === "overlay" ? "single" : "double";
}
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
    else if (s.role === "coronal") color = C.coronal;
    else if (s.role === "outflow") color = C.outflow;
    else if (s.role === "host") color = C.host;
    else if (s.role === "continuum") { color = C.continuum; dash = "5 4"; }
    else oi++;
    if (s.role === "balmer") dash = "6 3";              // told apart from the broad blues by the dash
    const label = s.role === "broad" && s.lines && s.lines.length ? `${s.name} (${s.lines.slice(0, 3).join(", ")}${s.lines.length > 3 ? "…" : ""})` : s.name;
    layers.push({ y: s.y, color, width: 1.3, dash, fill: s.role === "feii", name: s.name, label, hidden: hidden.has(s.name) });
  }
  layers.push({ y: P.model, color: C.model, width: 1.2, name: "model", label: "model", range: true, hidden: hidden.has("model") });
  const main = el.id === "rplot";
  plotSpectrum(el, { x: P.x, layers, zero: true, resid: P.resid, labels: P.labels, hatch: P.masked,
    bands: main ? residBands() : [], rich: main, onLabel: main && resultViews().includes("lines") ? openLineProfile : null });
  legend(legendEl, layers, hidden, redraw, main ? el : null);
}
function legend(el, layers, hidden, redraw, plotEl) {
  el.innerHTML = layers.map((l) => `<span data-s="${esc(l.name)}" class="${l.hidden ? "off" : ""}"><i class="${l.dash ? "dash" : ""}" style="border-color:${l.color}"></i>${esc(l.label || l.name)}</span>`).join("");
  el.querySelectorAll("span").forEach((s) => {
    s.onclick = () => { const n = s.dataset.s; hidden.has(n) ? hidden.delete(n) : hidden.add(n); plotFocus.delete(plotEl); redraw(); };
    // pointing at an entry brings that curve forward and fades the others
    if (plotEl) {
      s.onmouseenter = () => { if (hidden.has(s.dataset.s) || s.dataset.s === "data") return; plotFocus.set(plotEl, s.dataset.s); drawPlotEl(plotEl); };
      s.onmouseleave = () => { if (plotFocus.get(plotEl) === undefined) return; plotFocus.delete(plotEl); drawPlotEl(plotEl); };
    }
  });
}
const plotFocus = new Map();       // plot element -> the layer whose legend entry is under the pointer
// a line label of the Results plot: that line in Results › Line profiles
function openLineProfile(line) {
  if (window.LV) LV.open(line);
  rv = "lines"; loadResult();
}
// quick ranges of the Results plot: the regions of the main lines the spectrum covers
const ZOOM_REGIONS = [["Lyα", 1170, 1290], ["C IV", 1450, 1650], ["C III]", 1820, 1980], ["Mg II", 2650, 2950], ["Hγ", 4220, 4460],
  ["Hβ · [O III]", 4600, 5120], ["Fe II", 5100, 5600], ["He I", 5800, 5950], ["Hα · [N II]", 6400, 6780], ["[S II]", 6680, 6770]];
function renderZoomChips(el, box) {
  const st = plots.get(el);
  if (!st || !st.cfg || !st.cfg.x.length) { box.innerHTML = ""; return; }
  const x = st.cfg.x, lo = x[0], hi = x[x.length - 1], v = st.view;
  const regs = ZOOM_REGIONS.filter(([, a, b]) => a >= lo - 20 && b <= hi + 20);
  if (!regs.length) { box.innerHTML = ""; return; }
  const on = (a, b) => v && Math.abs(v[0] - a) < 1 && Math.abs(v[1] - b) < 1;
  box.innerHTML = `<button type="button" class="${v ? "" : "on"}" data-z="">All</button>` +
    regs.map(([n, a, b]) => `<button type="button" class="${on(a, b) ? "on" : ""}" data-z="${a},${b}">${esc(n)}</button>`).join("");
  box.querySelectorAll("button").forEach((btn) => (btn.onclick = () => {
    const z = btn.dataset.z ? btn.dataset.z.split(",").map(Number) : null;
    st.view = z; drawPlotEl(el); el.dispatchEvent(new CustomEvent("plotview", { detail: z }));
  }));
}
// ------------------------------------------------- light curves, reverberation
// Light curves & lags: a driver and a responder light curve, the responder shifted back by tau, and the
// lag of every method (ICCF, von Neumann, ZDCF, DRW) with its own curve; the adopted one drives the readout
let lcData = null, lcTimer = null;
const LXM = { iccf: "#1d2128", vn: "#8a919d", zdcf: "#7b3fd0", drw: "#d4472a" };
const LXN = { iccf: "ICCF", vn: "von Neumann", zdcf: "ZDCF", drw: "DRW" };
const LXL = ["#2a78d6", "#1baf7a", "#104281", "#6da7ec", "#4a3aa7", "#86b6ef", "#1c5cab", "#0f8f7e"];
const LX = { sig: null, D: null, S: {}, A: null, B: null, tau: 0, view: null, seasons: [], hoverK: null, adopt: "iccf",
  pairs: {}, pending: {}, touched: false, playing: null, raf: null, ccf: { key: null } };
async function loadTime() {
  try { lcData = await api("/api/wf/lightcurve"); } catch (e) { lcData = null; }
  clearTimeout(lcTimer);
  if (lcData && lcData.computing) {
    $("lx-msg").textContent = "Computing the light curves (errors of every line in every epoch)…";
    $("lx-body").classList.add("hidden"); $("lx-scrub").classList.add("hidden");
    lcTimer = setTimeout(loadTime, 3000);
    return;
  }
  if (lcData && lcData.feii_pending) lcTimer = setTimeout(loadTime, 4000);      // the Fe II light curve follows
  lxInit(lcData);
  renderReverb();
}
function lxInit(L) {
  if (!L || !L.series || !L.series.length) { $("lx-msg").textContent = "No light curves."; $("lx-body").classList.add("hidden"); $("lx-scrub").classList.add("hidden"); return; }
  const sig = JSON.stringify([L.names, L.mjd, L.series.map((s) => s.name)]);
  $("lx-msg").textContent = ""; $("lx-body").classList.remove("hidden"); $("lx-scrub").classList.remove("hidden");
  if (sig === LX.sig) return lxDraw();
  const LXO = ["Hb", "Ha", "MgII2798", "CIV1549", "Hg", "HeII4686"], rank = (ln) => (LXO.indexOf(ln) + 1 || 99);
  const lines = (L.lines || []).slice().sort((a, b) => rank(a) - rank(b));           // the main lines first
  const D = { mjd: L.mjd, dates: L.dates || [], z: L.z || 0, lag_range: L.lag_range || [-100, 100], series: [] };
  L.series.forEach((s) => {
    const group = s.line.startsWith("cont") ? "continuum" : s.line === "feii" ? "Fe II" : "broad line";
    const unit = s.kind === "flux" ? (s.unit || (group === "continuum" ? "flux/Å" : lineFluxUnit() || "flux")) : s.unit;
    const d = { key: s.name, label: s.label + (group === "broad line" && s.kind !== "flux" ? " " + s.kind.toUpperCase() : group === "broad line" ? " flux" : ""),
      line: group === "broad line" ? s.line : null, lineLabel: s.label, group, qty: s.kind, unit, y: s.y, e: s.err, src: s.source || null, window: s.window || null,
      color: group === "continuum" ? C.continuum : group === "Fe II" ? C.feii : LXL[Math.max(0, lines.indexOf(s.line)) % LXL.length] };
    D.series.push(d);
  });
  D.idx = D.mjd.map((_, k) => k).filter((k) => D.mjd[k] !== null).sort((a, b) => D.mjd[a] - D.mjd[b]);
  LX.D = D; LX.S = {}; D.series.forEach((s) => (LX.S[s.key] = s));
  LX.pairs = {}; LX.pending = {}; LX.ccf = { key: null };
  // observing seasons: gaps longer than 100 days
  let seasons = [], cur = [D.idx[0]];
  for (let j = 1; j < D.idx.length; j++) { const k = D.idx[j]; if (D.mjd[k] - D.mjd[D.idx[j - 1]] > 100) { seasons.push(cur); cur = []; } cur.push(k); }
  seasons.push(cur);
  LX.seasons = seasons.filter((ks) => ks.length && ks[0] !== undefined).map((ks) => ({ ks, t0: D.mjd[ks[0]], t1: D.mjd[ks[ks.length - 1]], label: (D.dates[ks[0]] || String(Math.round(D.mjd[ks[0]]))).slice(0, 4) }));
  const big = LX.seasons.filter((s) => s.ks.length >= 3);
  // the campaign: the seasons with at least 3 epochs, when isolated epochs lie outside them
  const camp = big.length > 1 && big.reduce((n, s) => n + s.ks.length, 0) < D.idx.length ? [big[0].t0, big[big.length - 1].t1] : null;
  LX.campaign = camp;
  const segs = big.length > 1 ? [["all", "all epochs"]].concat(camp ? [["rm", `${big[0].label}–${big[big.length - 1].label}`]] : []).concat(big.map((s) => [`s${LX.seasons.indexOf(s)}`, s.label])) : [];
  $("lx-seasons").innerHTML = segs.map(([k, t]) => `<button type="button" data-k="${k}">${esc(t)}</button>`).join("");
  $("lx-seasons").classList.toggle("hidden", !segs.length);
  $("lx-seasons").querySelectorAll("button").forEach((b) => (b.onclick = () => lxSetView(b.dataset.k)));
  $("lx-tau").min = D.lag_range[0]; $("lx-tau").max = D.lag_range[1];
  // default comparison: the continuum at 5100 A (or the first) driving the broad flux of the main line
  const conts = D.series.filter((s) => s.group === "continuum"), fluxes = D.series.filter((s) => s.group === "broad line" && s.qty === "flux");
  // the driver: the continuum window at 5100 A (measured on the data), else the fitted continuum
  const drv = conts.find((s) => s.window && s.window[0] >= 5000 && s.window[0] <= 5200) || LX.S.cont5100 || conts.find((s) => s.src === "window") || conts[0] || D.series[0];
  fluxes.sort((a, b) => rank(a.line) - rank(b.line));
  const rsp = fluxes[0] || D.series.find((s) => s !== drv);
  const Q = fluxes.slice(0, 4).map((s) => [drv.key, s.key, `${drv.label.replace("continuum ", "")} → ${s.lineLabel}`]);
  if (LX.S.feii) Q.push([drv.key, "feii", `${drv.label.replace("continuum ", "")} → Fe II`]);
  const cw = conts.filter((s) => s.src === "window").length > 1 ? conts.filter((s) => s.src === "window") : conts.filter((s) => s.src !== "window");
  if (cw.length > 1) Q.push([cw[0].key, cw[cw.length - 1].key, `${cw[0].label.replace("continuum ", "")} → ${cw[cw.length - 1].label.replace("continuum ", "")} (disc)`]);
  if (rsp && rsp.line && LX.S[`${rsp.line}_fwhm`]) Q.push([rsp.key, `${rsp.line}_fwhm`, `${rsp.lineLabel} flux → FWHM (breathing)`]);
  $("lx-quick").innerHTML = Q.map(([a, b, t]) => `<button type="button" class="lx-qc" data-a="${esc(a)}" data-b="${esc(b)}">${esc(t)}</button>`).join("");
  $("lx-quick").querySelectorAll(".lx-qc").forEach((q) => (q.onclick = () => lxSetPair(q.dataset.a, q.dataset.b)));
  const keep = LX.sig !== null && LX.S[LX.A] && LX.S[LX.B];         // a new Fe II curve: keep the comparison
  LX.sig = sig;
  if (!keep) lxSetView(camp ? "rm" : "all", true);
  if (!(drv && rsp)) { $("lx-msg").textContent = "Only one light curve: nothing to compare."; $("lx-scrub").classList.add("hidden"); return; }
  keep ? lxSetPair(LX.A, LX.B, true) : lxSetPair(drv.key, rsp.key);
}
// ------------------------------------------------------------------ the pickers
function lxSpark(s, W, H) {
  const D = LX.D, ks = D.idx.filter((k) => s.y[k] !== null), ys = ks.map((k) => s.y[k]);
  if (ks.length < 2) return "";
  const lo = Math.min(...ys), hi = Math.max(...ys), t0 = D.mjd[ks[0]], t1 = D.mjd[ks[ks.length - 1]];
  return ks.map((k, i) => `${i ? "L" : "M"}${((D.mjd[k] - t0) / ((t1 - t0) || 1) * W).toFixed(1)},${(H - 1 - (s.y[k] - lo) / ((hi - lo) || 1) * (H - 2)).toFixed(1)}`).join("");
}
function lxOpt(s, cur, other) {
  if (!s) return "<span></span>";
  const txt = s.qty === "flux" ? (s.group === "broad line" ? "flux" : s.group === "Fe II" ? "Fe II flux" : s.label.replace("continuum ", "")) : s.qty.toUpperCase();
  return `<button type="button" class="lx-opt${s.key === other ? " dis" : ""}${s.key === cur ? " on" : ""}" data-k="${esc(s.key)}" title="${esc(s.label)}">${esc(txt)}<svg viewBox="0 0 64 14" preserveAspectRatio="none"><path d="${lxSpark(s, 64, 14)}" fill="none" stroke="currentColor" stroke-width="1" opacity=".7"/></svg></button>`;
}
function lxMenu(which) {
  const D = LX.D, box = $(which === "a" ? "lx-pa" : "lx-pb"), cur = which === "a" ? LX.A : LX.B, other = which === "a" ? LX.B : LX.A;
  const o = (s) => lxOpt(s, cur, other), lines = [...new Set(D.series.filter((s) => s.group === "broad line").map((s) => s.line))];
  const wins = D.series.filter((s) => s.group === "continuum" && s.src === "window"), fits = D.series.filter((s) => s.group === "continuum" && s.src !== "window");
  box.querySelector(".lx-menu").innerHTML = (wins.length ? `<h4 title="me.window_continuum(): the mean of the data minus the fitted lines and Fe II in line-free windows (Shen et al. 2011); includes the host">Continuum windows (data − lines, Fe II)</h4><div class="lx-row">${wins.map(o).join("")}</div>` : "") +
    (fits.length ? `<h4>Fitted AGN continuum</h4><div class="lx-row">${fits.map(o).join("")}</div>` : "") +
    (LX.S.feii ? `<h4>Fe II</h4><div class="lx-row">${o(LX.S.feii)}</div>` : "") +
    (lines.length ? `<h4>Broad lines</h4><div class="lx-grid"><span></span><span class="muted small" style="text-align:center">flux</span><span class="muted small" style="text-align:center">FWHM</span><span class="muted small" style="text-align:center">EW</span>
    ${lines.sort((a, b) => (["Hb", "Ha", "MgII2798", "CIV1549", "Hg", "HeII4686"].indexOf(a) + 1 || 99) - (["Hb", "Ha", "MgII2798", "CIV1549", "Hg", "HeII4686"].indexOf(b) + 1 || 99)).map((ln) => { const f = D.series.find((s) => s.line === ln); return `<span class="lx-ln"><i style="background:${f.color}"></i>${esc(f.lineLabel)}</span>` + ["flux", "fwhm", "ew"].map((q) => o(D.series.find((s) => s.line === ln && s.qty === q))).join(""); }).join("")}</div>` : "");
  box.querySelectorAll(".lx-opt").forEach((b) => (b.onclick = (e) => { e.stopPropagation(); box.classList.remove("open"); which === "a" ? lxSetPair(b.dataset.k, LX.B) : lxSetPair(LX.A, b.dataset.k); }));
  const s = LX.S[cur];
  box.querySelector(":scope > button").innerHTML = `<i style="background:${s.color}"></i><span>${esc(s.label)}<small>${esc(s.group === "continuum" ? (s.src === "window" ? "window, data − lines" : "fitted AGN") : s.group)} · ${esc(s.unit)}</small></span><span class="chev">▾</span>`;
}
function lxSetPair(a, b, quiet) {
  if (a === b || !LX.S[a] || !LX.S[b]) return;
  const same = a === LX.A && b === LX.B;
  LX.A = a; LX.B = b; LX.ccf.key = null; lxStop();
  lxMenu("a"); lxMenu("b");
  $("lx-quick").querySelectorAll(".lx-qc").forEach((q) => q.classList.toggle("on", q.dataset.a === a && q.dataset.b === b));
  lxFetchPair();
  lxAdoptButtons();
  const ln = LX.S[b].group === "broad line" && LX.S[b].qty === "flux" ? LX.S[b].line : null;   // Measure lag: the responder's line
  if (!same && ln && [...$("rvline").options].some((o) => o.value === ln)) $("rvline").value = ln;
  if (same && quiet) return lxDraw();
  LX.touched = false;
  const P = lxPair(), m = P && (P.methods.find((q) => q.key === LX.adopt) || P.methods.find((q) => q.key === "iccf"));
  if (m) return lxSetTau(m.tau);
  const c = lxCCF(lxInView());                                          // until the lags arrive: the live ICCF
  lxSetTau(isFinite(c.cen) ? c.cen : 0);
}
const lxPairKey = () => `${LX.sig}|${LX.A}|${LX.B}`;
const lxPair = () => LX.pairs[lxPairKey()] || null;
async function lxFetchPair() {
  const key = lxPairKey(), a = LX.A, b = LX.B;
  if (LX.pairs[key] || LX.pending[key]) return;
  LX.pending[key] = true;
  try { LX.pairs[key] = await api(`/api/wf/lagpair?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`); }
  catch (e) { LX.pairs[key] = { methods: [], zdcf: null, error: e.message || String(e) }; }
  delete LX.pending[key];
  if (lxPairKey() !== key) return;
  lxAdoptButtons();
  const m = lxPair().methods.find((q) => q.key === LX.adopt);
  if (!LX.touched && m) lxAnimateTo(m.tau); else lxDraw();
}
function lxSetView(k, quiet) {
  $("lx-seasons").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.k === k));
  if (k === "all") LX.view = null;
  else if (k === "rm") LX.view = LX.campaign;
  else { const s = LX.seasons[+k.slice(1)]; LX.view = [s.t0, s.t1]; }
  LX.hoverK = null; LX.ccf.key = null;
  if (!quiet) lxDraw();
}
const lxInView = () => LX.D.idx.filter((k) => (!LX.view || (LX.D.mjd[k] >= LX.view[0] - 1 && LX.D.mjd[k] <= LX.view[1] + 1)) && LX.S[LX.A].y[k] !== null && LX.S[LX.B].y[k] !== null);
function lxStats(a) { const m = a.reduce((s, v) => s + v, 0) / a.length, sd = Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / Math.max(1, a.length - 1)); return { m, sd }; }
function lxFvar(f, e) { const { m, sd } = lxStats(f), me2 = e.reduce((s, v) => s + v * v, 0) / e.length, x = sd * sd - me2; return x > 0 ? Math.sqrt(x) / Math.abs(m) : 0; }
function lxTicks(lo, hi, n) { const span = hi - lo || 1, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0)), step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= step0); const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10)); return out; }
// ------------------------------------------------------------------ the browser's ICCF (as fantasy_agn.rm.iccf, with a minimum overlap)
function lxInterp(x, xs, ys) { let j = 0; while (j < xs.length - 2 && xs[j + 1] < x) j++; const t = (x - xs[j]) / ((xs[j + 1] - xs[j]) || 1); return ys[j] + (ys[j + 1] - ys[j]) * t; }
function lxPearson(a, b) { const n = a.length; if (n < 3) return NaN; const ma = a.reduce((s, v) => s + v, 0) / n, mb = b.reduce((s, v) => s + v, 0) / n; let c = 0, va = 0, vb = 0; for (let i = 0; i < n; i++) { c += (a[i] - ma) * (b[i] - mb); va += (a[i] - ma) ** 2; vb += (b[i] - mb) ** 2; } return va > 0 && vb > 0 ? c / Math.sqrt(va * vb) : NaN; }
function lxIccfAt(ks, lag) {
  if (ks.length < 3) return NaN;
  const D = LX.D, t = ks.map((k) => D.mjd[k]), f1 = ks.map((k) => LX.S[LX.A].y[k]), f2 = ks.map((k) => LX.S[LX.B].y[k]), lo = t[0], hi = t[t.length - 1];
  const a1 = [], b1 = [], a2 = [], b2 = [];
  t.forEach((ti, i) => { const x = ti + lag; if (x >= lo && x <= hi) { a1.push(f1[i]); b1.push(lxInterp(x, t, f2)); } const y = ti - lag; if (y >= lo && y <= hi) { a2.push(lxInterp(y, t, f1)); b2.push(f2[i]); } });
  const need = Math.max(5, Math.ceil(ks.length / 3));
  const r = [a1.length >= need ? lxPearson(a1, b1) : NaN, a2.length >= need ? lxPearson(a2, b2) : NaN].filter(isFinite);
  return r.length ? r.reduce((s, v) => s + v, 0) / r.length : NaN;
}
function lxCCF(ks) {
  const key = `${LX.A}|${LX.B}|${ks.join(",")}`;
  if (LX.ccf.key === key) return LX.ccf;
  const lo = +$("lx-tau").min, hi = +$("lx-tau").max, step = Math.max(0.5, (hi - lo) / 300), lags = [], r = [];
  for (let L = lo; L <= hi + 1e-9; L += step) { lags.push(L); r.push(lxIccfAt(ks, L)); }
  let kmax = -1; r.forEach((v, i) => { if (isFinite(v) && (kmax < 0 || v > r[kmax])) kmax = i; });
  let s = 0, w = 0; if (kmax >= 0) for (let i = 0; i < r.length; i++) if (r[i] >= 0.8 * r[kmax]) { s += lags[i] * r[i]; w += r[i]; }
  LX.ccf = { key, lags, r, peak: kmax >= 0 ? lags[kmax] : NaN, rmax: kmax >= 0 ? r[kmax] : NaN, cen: w ? s / w : NaN };
  return LX.ccf;
}
// ------------------------------------------------------------------ the two light curves
function lxCurves() {
  const D = LX.D, svg = $("lx-lc"), LM = { l: 58, r: 16, t: 22 }, PH = 160, GAP = 22, W = svg.clientWidth || 900, H = LM.t + 2 * PH + GAP * 2 + 26;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("height", H); svg.innerHTML = "";
  const ks = lxInView();
  if (ks.length < 3) { anEl("text", { x: W / 2, y: H / 2, "text-anchor": "middle" }, svg).textContent = "fewer than 3 epochs with both light curves"; return; }
  const t0 = LX.view ? LX.view[0] : D.mjd[ks[0]], t1 = LX.view ? LX.view[1] : D.mjd[ks[ks.length - 1]], pad = (t1 - t0) * 0.02 || 5;
  const x = (t) => LM.l + (t - t0 + pad) / (t1 - t0 + 2 * pad) * (W - LM.l - LM.r);
  const inv = (px) => t0 - pad + (px - LM.l) / (W - LM.l - LM.r) * (t1 - t0 + 2 * pad);
  if (LX.seasons.length > 1) LX.seasons.forEach((s, i) => { if (s.t1 < t0 - pad || s.t0 > t1 + pad) return;
    anEl("rect", { x: x(s.t0) - 4, y: LM.t, width: Math.max(8, x(s.t1) - x(s.t0) + 8), height: H - LM.t - 26, fill: i % 2 ? "#f7f8fa" : "#fbfbfc" }, svg);
    anEl("text", { x: (x(Math.max(s.t0, t0)) + x(Math.min(s.t1, t1))) / 2, y: 12, "text-anchor": "middle", style: "font-weight:500;fill:#4a5160" }, svg).textContent = s.label; });
  const mk = (key) => { const s = LX.S[key], st = lxStats(ks.map((k) => s.y[k])), m = st.m || 1; return { s, m: st.m, val: (k) => 100 * (s.y[k] / m - 1), err: (k) => 100 * (s.e[k] || 0) / Math.abs(m), f: lxFvar(ks.map((k) => s.y[k]), ks.map((k) => s.e[k] || 0)) }; };
  const P = [mk(LX.A), mk(LX.B)], segs = (LX.seasons.length ? LX.seasons : [{ ks }]).map((s) => s.ks.filter((k) => ks.includes(k)));
  let y0 = LM.t;
  P.forEach((p, pi) => {
    const vals = ks.map(p.val), errs = ks.map(p.err).sort((a, b) => a - b);
    let lo = Math.min(...vals), hi = Math.max(...vals); const pd = Math.max((hi - lo) * 0.12, 1.2 * (errs[Math.floor(errs.length / 2)] || 0), 0.5); lo -= pd; hi += pd;
    const y = (v) => y0 + 18 + (1 - (v - lo) / (hi - lo)) * (PH - 22);
    anEl("text", { x: LM.l, y: y0 + 11, class: "ph" }, svg).textContent = `${pi ? "responder" : "driver"} · ${p.s.label}`;
    anEl("text", { x: W - LM.r, y: y0 + 11, class: "pm", "text-anchor": "end" }, svg).textContent = `${p.f > 0 ? `F_var ${(100 * p.f).toFixed(1)}%` : "varies within its errors"} · mean ${fmtNum(p.m, 4)} ${p.s.unit}`;
    for (const v of lxTicks(lo, hi, 4)) { anEl("line", { x1: LM.l, x2: W - LM.r, y1: y(v), y2: y(v), stroke: v === 0 ? "#cfd3d9" : "#f0f1f3" }, svg);
      anEl("text", { x: LM.l - 6, y: y(v) + 3, "text-anchor": "end" }, svg).textContent = `${v > 0 ? "+" : ""}${v}%`; }
    const g = anEl("g", {}, svg);
    if (pi === 0 && $("lx-ghost").checked) {            // the responder shifted back by tau, scaled to the driver
      const sb = lxStats(ks.map(P[1].val)), sa = lxStats(ks.map(p.val));
      segs.forEach((sk) => {
        const pts = sk.map((k) => [x(D.mjd[k] - LX.tau), y(Math.max(lo, Math.min(hi, sa.m + (P[1].val(k) - sb.m) / (sb.sd || 1) * sa.sd)))]).filter((q) => q[0] >= LM.l - 4 && q[0] <= W - LM.r + 4);
        if (pts.length > 1) anEl("polyline", { points: pts.map((q) => q.join(",")).join(" "), fill: "none", stroke: "#d4472a", "stroke-width": 5, opacity: 0.16, "stroke-linejoin": "round", "stroke-linecap": "round" }, g);
        pts.forEach(([px, py]) => anEl("circle", { cx: px, cy: py, r: 2.4, fill: "#d4472a", opacity: 0.75 }, g));
      });
    }
    segs.forEach((sk) => { if (sk.length > 1) anEl("polyline", { points: sk.map((k) => `${x(D.mjd[k])},${y(p.val(k))}`).join(" "), fill: "none", stroke: p.s.color, "stroke-width": 1, opacity: 0.45 }, g); });
    ks.forEach((k) => { const v = p.val(k), e = p.err(k), on = k === LX.hoverK, sel = k === rk;
      anEl("line", { x1: x(D.mjd[k]), x2: x(D.mjd[k]), y1: y(v - e), y2: y(v + e), stroke: p.s.color, opacity: 0.45 }, g);
      anEl("circle", { cx: x(D.mjd[k]), cy: y(v), r: on ? 5.5 : sel ? 4.6 : 3.2, fill: p.s.color, stroke: on || sel ? "#1d2128" : "#fff", "stroke-width": on || sel ? 1.5 : 0.8 }, g); });
    anEl("line", { x1: LM.l, x2: W - LM.r, y1: y0 + PH, y2: y0 + PH, stroke: "#e2e4e8" }, svg);
    y0 += PH + GAP;
  });
  const yt = H - 26;
  for (const v of lxTicks(t0, t1, 8)) { anEl("line", { x1: x(v), x2: x(v), y1: yt, y2: yt + 4, stroke: "#b8bec7" }, svg); anEl("text", { x: x(v), y: yt + 14, "text-anchor": "middle" }, svg).textContent = `MJD ${v}`; }
  if (LX.hoverK !== null) anEl("line", { x1: x(D.mjd[LX.hoverK]), x2: x(D.mjd[LX.hoverK]), y1: LM.t, y2: yt, stroke: "#1d2128", opacity: 0.3 }, svg);
  const near = (e) => { const r = svg.getBoundingClientRect(), t = inv((e.clientX - r.left) * (W / r.width)); let best = null; ks.forEach((k) => { if (best === null || Math.abs(D.mjd[k] - t) < Math.abs(D.mjd[best] - t)) best = k; }); return best; };
  svg.onmousemove = (e) => { const k = near(e); if (k !== null && k !== LX.hoverK) { LX.hoverK = k; lxHover(ks); lxCurves(); } };
  svg.onmouseleave = () => { LX.hoverK = null; lxHover(ks); lxCurves(); };
  svg.onclick = (e) => { const k = near(e); if (k !== null) { rk = k; loadResult(true); } };
}
function lxHover(ks) {
  const k = LX.hoverK, D = LX.D;
  if (k === null) { $("lx-hover").textContent = "% change from the mean of the epochs shown · hover for an epoch, click to open it"; return; }
  const pc = (key) => { const st = lxStats(ks.map((j) => LX.S[key].y[j])), v = 100 * (LX.S[key].y[k] / (st.m || 1) - 1); return `${v > 0 ? "+" : ""}${v.toFixed(1)}%`; };
  $("lx-hover").innerHTML = `<b>#${k}</b> ${esc(D.dates[k] || "")} · MJD ${D.mjd[k].toFixed(1)} · driver ${pc(LX.A)} · responder ${pc(LX.B)} <span class="muted">· click to open this epoch</span>`;
}
// ------------------------------------------------------------------ each method's own curve
function lxLin(xs, ys, t) {
  for (let i = 0; i < xs.length - 1; i++) if (xs[i] <= t && t <= xs[i + 1]) { const a = ys[i], b = ys[i + 1]; if (a === null || b === null || !isFinite(a) || !isFinite(b)) return NaN; return a + (b - a) * (t - xs[i]) / ((xs[i + 1] - xs[i]) || 1); }
  return NaN;
}
function lxBest(y, lower) { let k = -1; y.forEach((v, i) => { if (v !== null && isFinite(v) && (k < 0 || (lower ? v < y[k] : v > y[k]))) k = i; }); return k; }
function lxAvailable(key) {
  if (key === "iccf") return true;
  const P = lxPair(); if (!P) return false;
  if (key === "zdcf") return !!(P.zdcf && P.zdcf.tau.filter((v) => v !== null).length > 1);
  const m = P.methods.find((q) => q.key === key); return !!(m && m.curve);
}
function lxAdoptButtons() {
  const P = lxPair();
  if (P && !lxAvailable(LX.adopt)) LX.adopt = "iccf";
  $("lx-adopt").innerHTML = ["iccf", "vn", "zdcf", "drw"].map((k) => `<button type="button" data-k="${k}" class="${k === LX.adopt ? "on" : ""}" ${lxAvailable(k) ? "" : `disabled title="${P ? "not measured for this pair" : "measuring…"}"`}><i style="background:${LXM[k]}"></i>${LXN[k]}</button>`).join("");
  $("lx-adopt").querySelectorAll("button").forEach((b) => (b.onclick = () => lxAdopt(b.dataset.k)));
}
function lxAdopt(k) {
  if (!lxAvailable(k)) return;
  LX.adopt = k; lxAdoptButtons(); lxStop();
  const box = document.querySelector(`.rvform [data-m="${k}"]`);           // the measurement below uses it too
  if (box) box.checked = true;
  const m = lxPair() && lxPair().methods.find((q) => q.key === k);
  if (m) lxAnimateTo(m.tau); else lxDraw();
}
function lxAdopted(ks) {
  const P = lxPair(), k = LX.adopt, m = P && P.methods.find((q) => q.key === k);
  if (k === "iccf" || !P) {
    const c = lxCCF(ks);
    return { key: "iccf", lags: c.lags, y: c.r, at: (t) => lxIccfAt(ks, t), lower: false, lo: -0.6, hi: 1, ticks: [-0.5, 0, 0.5, 1], corr: true, axis: "correlation r", sym: "r",
      best: c.peak, bestv: c.rmax, mark: c.cen, markLabel: `ICCF centroid ${isFinite(c.cen) ? c.cen.toFixed(1) : "—"} d (epochs shown)`, note: "ICCF, live: recomputed for the epochs shown" };
  }
  if (k === "zdcf") {
    const Z = P.zdcf, o = Z.tau.map((t, i) => [t, Z.dcf[i]]).filter((q) => q[0] !== null && q[1] !== null).sort((a, b) => a[0] - b[0]);
    const lags = o.map((q) => q[0]), y = o.map((q) => q[1]), b = lxBest(y, false);
    return { key: k, lags, y, at: (t) => lxLin(lags, y, t), lower: false, lo: -0.6, hi: 1, ticks: [-0.5, 0, 0.5, 1], corr: true, points: true, axis: "ZDCF", sym: "DCF",
      best: lags[b], bestv: y[b], mark: m ? m.tau : NaN, markLabel: m ? `ZDCF lag ${m.tau.toFixed(1)} d` : "", note: "ZDCF, all epochs: equal-population bins" };
  }
  const lags = m.curve.lags, ym = Math.max(...m.curve.y.filter((v) => v !== null)), y = k === "drw" ? m.curve.y.map((v) => (v === null ? v : v / ym)) : m.curve.y;
  const fin = y.filter((v) => v !== null), b = lxBest(y, k === "vn");
  if (k === "drw") return { key: k, lags, y, at: (t) => lxLin(lags, y, t), lower: false, lo: 0, hi: 1.08, ticks: [0, 0.5, 1], axis: "posterior ÷ peak", sym: "p",
    best: lags[b], bestv: y[b], mark: m.tau, markLabel: `DRW lag ${m.tau.toFixed(1)} d`, note: "DRW, all epochs: the likelihood of each lag, weighted for the overlap" };
  const mn = Math.min(...fin), mx = Math.max(...fin), pad = (mx - mn) * 0.08 || 0.1;
  return { key: k, lags, y, at: (t) => lxLin(lags, y, t), lower: true, lo: mn - pad, hi: mx + pad, ticks: lxTicks(mn, mx, 4), axis: "VN ÷ median (lower ↑)", sym: "VN",
    best: lags[b], bestv: y[b], worst: mx, mark: m.tau, markLabel: `von Neumann lag ${m.tau.toFixed(1)} d`, note: "von Neumann, all epochs: its minimum is the lag, so the axis runs downwards" };
}
function lxScore(a, v) {
  if (!isFinite(v)) return NaN;
  if (a.lower) return (a.worst - v) / ((a.worst - a.bestv) || 1);
  return a.bestv > 0 ? v / a.bestv : NaN;
}
function lxCorr() {
  const CM = { l: 84, r: 14, t: 12, b: 28 }, svg = $("lx-ccf"), W = svg.clientWidth || 700, H = 240;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = '<defs><pattern id="lx-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke="#e6e8ec" stroke-width="2"/></pattern></defs>';
  const ks = lxInView(), a = lxAdopted(ks), lo = +$("lx-tau").min, hi = +$("lx-tau").max, col = LXM[a.key], P = lxPair();
  const x = (L) => CM.l + (L - lo) / (hi - lo) * (W - CM.l - CM.r);
  const y = (v) => { const u = (Math.max(a.lo, Math.min(a.hi, v)) - a.lo) / (a.hi - a.lo); return CM.t + (a.lower ? u : 1 - u) * (H - CM.t - CM.b); };
  const inv = (px) => lo + (px - CM.l) / (W - CM.l - CM.r) * (hi - lo), base = a.lower ? a.hi : a.lo;
  $("lx-axisnote").textContent = a.note;
  $("lx-lbiccf").style.display = a.corr && a.key !== "iccf" ? "" : "none";
  $("lx-lbzdcf").style.display = a.corr && a.key !== "zdcf" && P && P.zdcf ? "" : "none";
  for (const r of a.ticks) { anEl("line", { x1: CM.l, x2: W - CM.r, y1: y(r), y2: y(r), stroke: r === 0 && a.corr ? "#cfd3d9" : "#f0f1f3" }, svg); anEl("text", { x: CM.l - 6, y: y(r) + 3, "text-anchor": "end" }, svg).textContent = r; }
  for (const L of lxTicks(lo, hi, 8)) { anEl("line", { x1: x(L), x2: x(L), y1: H - CM.b, y2: H - CM.b + 4, stroke: "#b8bec7" }, svg); anEl("text", { x: x(L), y: H - CM.b + 15, "text-anchor": "middle" }, svg).textContent = L + " d"; }
  if (lo < 0 && hi > 0) anEl("line", { x1: x(0), x2: x(0), y1: CM.t, y2: H - CM.b, stroke: "#cfd3d9", "stroke-dasharray": "3 3" }, svg);
  anEl("text", { x: 14, y: (H - CM.b + CM.t) / 2, "text-anchor": "middle", transform: `rotate(-90 14 ${(H - CM.b + CM.t) / 2})`, style: `fill:${col}` }, svg).textContent = a.axis;
  anEl("text", { x: W - CM.r, y: H - 2, "text-anchor": "end" }, svg).textContent = "τ, observed frame (responder after the driver →)";
  if (a.corr && a.key !== "iccf" && $("lx-showiccf").checked) {         // the other correlation, faint
    const c = lxCCF(ks), pts = c.lags.map((L, i) => (isFinite(c.r[i]) ? `${x(L)},${y(c.r[i])}` : null)).filter(Boolean);
    if (pts.length > 1) anEl("polyline", { points: pts.join(" "), fill: "none", stroke: LXM.iccf, "stroke-width": 1.2, opacity: 0.35 }, svg);
  }
  const Z = P && P.zdcf;
  const zpts = (op) => { const g = anEl("g", { opacity: op }, svg);
    Z.tau.forEach((t, i) => { if (t === null || Z.dcf[i] === null) return;
      const cx = x(t), cy = y(Z.dcf[i]);
      anEl("line", { x1: x(t - (Z.sl[i] || 0)), x2: x(t + (Z.sh[i] || 0)), y1: cy, y2: cy, stroke: LXM.zdcf, opacity: 0.4 }, g);
      anEl("line", { x1: cx, x2: cx, y1: y(Z.dcf[i] - (Z.el[i] || 0)), y2: y(Z.dcf[i] + (Z.eh[i] || 0)), stroke: LXM.zdcf, opacity: 0.6 }, g);
      anEl("rect", { x: cx - 2.6, y: cy - 2.6, width: 5.2, height: 5.2, fill: LXM.zdcf, transform: `rotate(45 ${cx} ${cy})` }, g); }); };
  if (Z && a.corr && a.key !== "zdcf" && $("lx-showzdcf").checked) zpts(0.45);
  const pts = a.lags.map((L, i) => (a.y[i] !== null && isFinite(a.y[i]) && L >= lo && L <= hi ? [x(L), y(a.y[i])] : null)).filter(Boolean);
  if (pts.length > 1) {
    if (!a.points) anEl("path", { d: `M${pts[0][0]},${y(base)}L` + pts.map((q) => q.join(",")).join("L") + `L${pts[pts.length - 1][0]},${y(base)}Z`, fill: col, opacity: 0.07 }, svg);
    anEl("polyline", { points: pts.map((q) => q.join(",")).join(" "), fill: "none", stroke: col, "stroke-width": a.points ? 1 : 1.8, opacity: a.points ? 0.5 : 1 }, svg);
    const fin = a.lags.filter((L, i) => a.y[i] !== null && isFinite(a.y[i]));
    [[lo, fin[0]], [fin[fin.length - 1], hi]].forEach(([p, q]) => { if (q - p > (hi - lo) / 60) { anEl("rect", { x: x(p), y: CM.t, width: x(q) - x(p), height: H - CM.t - CM.b, fill: "url(#lx-hatch)" }, svg);
      if (x(q) - x(p) > 70) anEl("text", { x: (x(p) + x(q)) / 2, y: CM.t + 24, "text-anchor": "middle" }, svg).textContent = a.key === "zdcf" ? "no bins" : "too little overlap"; } });
  } else anEl("text", { x: (CM.l + W) / 2, y: H / 2, "text-anchor": "middle" }, svg).textContent = "too little overlap at every shift";
  if (a.points && Z) zpts(1);
  if (isFinite(a.mark)) {
    anEl("line", { x1: x(a.mark), x2: x(a.mark), y1: CM.t, y2: H - CM.b, stroke: col, "stroke-dasharray": "1 3" }, svg);
    const right = x(a.mark) > W - 220;
    anEl("text", { x: x(a.mark) + (right ? -4 : 4), y: CM.t + 10, "text-anchor": right ? "end" : "start", style: `fill:${col}` }, svg).textContent = a.markLabel;
  }
  const v = a.at(LX.tau);
  anEl("line", { x1: x(LX.tau), x2: x(LX.tau), y1: CM.t, y2: H - CM.b, stroke: "#d4472a", "stroke-width": 2 }, svg);
  anEl("circle", { cx: x(LX.tau), cy: isFinite(v) ? y(v) : y(base), r: 7, fill: "#fff", stroke: "#d4472a", "stroke-width": 2.5 }, svg);
  let drag = false;
  const move = (e) => { const b = svg.getBoundingClientRect(); LX.touched = true; lxSetTau(inv((e.clientX - b.left) * (W / b.width))); };
  svg.onpointerdown = (e) => { drag = true; svg.setPointerCapture(e.pointerId); lxStop(); move(e); };
  svg.onpointermove = (e) => { if (drag) move(e); };
  svg.onpointerup = () => (drag = false);
  lxMethods(x, W, CM);
  return { a, v };
}
function lxMethods(x, W, CM) {
  const svg = $("lx-meth"), P = lxPair(), M = P ? P.methods : [], row = 26, H = Math.max(1, M.length) * row + 8, lo = +$("lx-tau").min, hi = +$("lx-tau").max;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("height", H); svg.innerHTML = "";
  if (!P) { anEl("text", { x: CM.l, y: 18 }, svg).textContent = "measuring the lag with every method (ICCF, von Neumann, ZDCF, DRW)…"; return; }
  if (!M.length) { anEl("text", { x: CM.l, y: 18 }, svg).textContent = P.error || "no lag could be measured for this pair"; return; }
  M.forEach((m, i) => {
    const yc = 4 + i * row + row / 2, col = LXM[m.key] || "#9aa1ad", on = m.key === LX.adopt;
    if (on) anEl("rect", { x: 2, y: yc - row / 2 + 1, width: W - 4, height: row - 2, rx: 5, fill: "#f6f7f9" }, svg);
    const g = anEl("g", { opacity: on ? 1 : 0.55, style: "cursor:pointer" }, svg);
    g.onclick = () => lxAdopt(m.key);
    anEl("rect", { x: 0, y: yc - row / 2, width: W, height: row, fill: "transparent" }, g);
    anEl("text", { x: CM.l - 6, y: yc + 3, "text-anchor": "end", style: `fill:${col};font-weight:${on ? 700 : 500}` }, g).textContent = (on ? "● " : "") + m.method;
    const nb = 100, bw = (hi - lo) / nb, h = new Array(nb).fill(0);
    m.samples.forEach((v) => { const b = Math.floor((v - lo) / bw); if (b >= 0 && b < nb) h[b]++; });
    const hm = Math.max(1, ...h);
    h.forEach((n, b) => { if (n) anEl("rect", { x: x(lo + b * bw), y: yc + 9 - 18 * n / hm, width: Math.max(1, x(lo + bw) - x(lo) - 0.5), height: 18 * n / hm, fill: col, opacity: on ? 0.5 : 0.3 }, g); });
    if (m.tau !== null) {
      anEl("line", { x1: x(Math.max(lo, m.tau - (m.lo || 0))), x2: x(Math.min(hi, m.tau + (m.hi || 0))), y1: yc + 11, y2: yc + 11, stroke: col, "stroke-width": on ? 2.2 : 1.6 }, g);
      anEl("circle", { cx: x(m.tau), cy: yc + 11, r: on ? 4.2 : 3.4, fill: col }, g);
    }
  });
}
const lxFmt = (v) => (!isFinite(v) ? "—" : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3));
const lxPm = (m) => `${m.tau.toFixed(1)} <span class="muted">−${(m.lo || 0).toFixed(1)} +${(m.hi || 0).toFixed(1)}</span>`;
function lxReadout({ a, v }) {
  const D = LX.D, tau = LX.tau, rest = tau / (1 + D.z), P = lxPair(), M = P ? P.methods : [], s = lxScore(a, v), ad = M.find((m) => m.key === a.key), col = LXM[a.key];
  const verdict = !isFinite(v) ? "No value of this curve at this shift." : s >= 0.97 ? `<b>Locked.</b> The ${LXN[a.key]} curve is at its optimum here.`
    : tau < 0 && s >= 0.5 ? "A negative τ: the responder would change <i>before</i> the driver." : s < 0.5 ? "The curves do not line up at this shift." : "Getting closer: keep sliding towards the optimum.";
  const lineKey = LX.S[LX.B].group === "broad line" && LX.S[LX.B].qty === "flux" && LX.S[LX.A].group === "continuum" ? LX.S[LX.B].line : null;
  $("lx-readout").innerHTML = `<div><h3>${esc(LX.S[LX.B].label)} after ${esc(LX.S[LX.A].label)}</h3><div class="note small">τ = ${tau.toFixed(1)} d observed · ${rest.toFixed(1)} d in the rest frame</div></div>
    <div class="lx-big" style="color:${col}">${a.sym} = ${lxFmt(v)} <small>${a.lower ? "minimum" : "best"} ${lxFmt(a.bestv)} at ${isFinite(a.best) ? a.best.toFixed(1) : "—"} d</small></div>
    <div class="lx-rbar"><div style="width:${isFinite(s) ? Math.max(0, Math.min(1, s)) * 100 : 0}%;background:${col}"></div></div>
    <div class="small">${verdict}</div>
    ${ad ? `<div class="small">Adopted: <b>${esc(ad.method)} ${ad.tau.toFixed(1)} −${(ad.lo || 0).toFixed(1)} +${(ad.hi || 0).toFixed(1)} d</b> observed, ${(ad.tau / (1 + D.z)).toFixed(1)} d in the rest frame.</div>` : ""}
    <div class="lx-snap"><div class="muted small">${P ? `Lags of all ${P.n || D.idx.length} epochs (observed frame); click a method to adopt it, again to jump to its lag:` : "Measuring the lags of all epochs…"}</div>
      ${M.map((m, i) => `<button type="button" class="btn small" data-i="${i}"><span style="color:${LXM[m.key]}">${esc(m.method)}${m.key === a.key ? ' <span class="badge ok">adopted</span>' : ""}</span><b>${lxPm(m)} d</b></button>`).join("")}
      ${isFinite(a.best) ? `<button type="button" class="btn small" data-i="best"><span>${a.lower ? "minimum" : "peak"} of the ${LXN[a.key]} curve${a.key === "iccf" ? " (live)" : ""}</span><b>${a.best.toFixed(1)} d</b></button>` : ""}</div>
    ${lineKey ? `<p class="note small">For the published lag, rms spectrum and mass: <b>Measure lag</b> below runs <code>me.reverberation(${esc(JSON.stringify(lineKey).replace(/"/g, "'"))}${LX.S[LX.A].window ? `, continuum=window ${LX.S[LX.A].window.join("–")} Å` : ""}${a.key !== "iccf" ? `, primary='${a.key}'` : ""})</code>${LX.S[LX.A].window ? "" : ", with the correlated errors of each fit"}.</p>` : ""}
    <p class="note small">ZDCF: the z-transformed discrete correlation function (Alexander 1997), FANTASY's port of pyZDCF (Jankov et al. 2022). DRW here uses the two light curves only.</p>`;
  $("lx-readout").querySelectorAll("[data-i]").forEach((b) => (b.onclick = () => {
    lxStop(); LX.touched = true; const i = b.dataset.i;
    if (i === "best") return lxAnimateTo(a.best);
    const m = M[+i]; if (m.key !== LX.adopt && lxAvailable(m.key)) return lxAdopt(m.key);
    lxAnimateTo(m.tau);
  }));
}
// ------------------------------------------------------------------ wiring
function lxSetTau(v) { LX.tau = Math.max(+$("lx-tau").min, Math.min(+$("lx-tau").max, v)); $("lx-tau").value = LX.tau; lxDraw(); }
function lxDraw() {
  if (LX.raf || !LX.D || !LX.A) return;
  LX.raf = requestAnimationFrame(() => { LX.raf = null; if (rv !== "time") return; lxCurves(); lxReadout(lxCorr()); });
}
function lxAnimateTo(v) { if (!isFinite(v)) return; lxStop(); const a = LX.tau, t0 = performance.now(), dur = 700; const step = (now) => { const u = Math.min(1, (now - t0) / dur), e = 1 - (1 - u) ** 3; lxSetTau(a + (v - a) * e); LX.playing = u < 1 ? requestAnimationFrame(step) : null; }; LX.playing = requestAnimationFrame(step); }
function lxStop() { if (LX.playing) cancelAnimationFrame(LX.playing); LX.playing = null; $("lx-play").classList.remove("on"); $("lx-play").textContent = "▶ sweep"; }
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
  const sigCell = (q) => q.sig === null || q.sig === undefined ? "—" :
    `<span class="${q.sig_value < 2 ? "warnword" : q.sig_value < 3 ? "" : "okword"}" title="${q.p_side === null ? "" : Math.round(100 * q.p_side) + "% of the lag distribution is on this side of zero"}">${esc(q.sig)}</span>`;
  $("rvlags").innerHTML = '<thead><tr><th>Method</th><th class="num">Lag rest (d)</th><th class="num" title="how clearly the lag differs from zero: the share of its distribution on its side of zero, as a one-sided Gaussian significance">Significance</th><th class="num">r max</th><th class="num" title="share of the lag distribution in other peaks (seasonal aliases)">outside peak</th></tr></thead><tbody>' +
    R.lags.map((q) => `<tr class="${q.adopted ? "sel" : ""}"><td>${esc(q.method)}${q.adopted ? ' <span class="badge ok">adopted</span>' : ""}</td><td class="num">${esc(q.lag)}</td><td class="num">${sigCell(q)}</td><td class="num">${esc(q.r_max)}</td><td class="num">${esc(q.outside)}</td></tr>`).join("") + "</tbody>";
  const A = R.lags.find((q) => q.adopted);
  $("rvwarn").classList.toggle("hidden", !R.warnings.length);
  if (A && A.sig_value !== null && A.sig_value !== undefined && R.warnings.length) $("rvwarn").innerHTML = `<div><b>Read the lag and the rms width with care.</b><ul style="margin:4px 0 0">${R.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></div>`;
  const src = withToken(`/api/wf/reverb.png?v=${R.id}`);
  if ($("rvfig").dataset.src !== src) { $("rvfig").dataset.src = src; $("rvfig").src = src; }
}
function initTime() {
  ["lx-pa", "lx-pb"].forEach((id) => { $(id).querySelector(":scope > button").onclick = (e) => { e.stopPropagation(); const open = $(id).classList.contains("open"); document.querySelectorAll(".lx-pick").forEach((p) => p.classList.remove("open")); if (!open) $(id).classList.add("open"); }; });
  document.addEventListener("click", (e) => { if (!e.target.closest(".lx-pick")) document.querySelectorAll(".lx-pick").forEach((p) => p.classList.remove("open")); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") document.querySelectorAll(".lx-pick").forEach((p) => p.classList.remove("open")); });
  $("lx-swap").onclick = () => lxSetPair(LX.B, LX.A);
  $("lx-tau").oninput = (e) => { lxStop(); LX.touched = true; lxSetTau(+e.target.value); };
  $("lx-play").onclick = () => {
    if (LX.playing) return lxStop();
    LX.touched = true; $("lx-play").classList.add("on"); $("lx-play").textContent = "■ stop";
    const lo = +$("lx-tau").min, hi = +$("lx-tau").max, t0 = performance.now(), dur = 8000;
    const step = (now) => { lxSetTau(lo + (hi - lo) * (((now - t0) % dur) / dur)); LX.playing = requestAnimationFrame(step); };
    LX.playing = requestAnimationFrame(step);
  };
  ["lx-ghost", "lx-showiccf", "lx-showzdcf"].forEach((id) => ($(id).onchange = lxDraw));
  window.addEventListener("resize", () => LX.D && lxDraw());
  $("rvrun").onclick = () => {
    const methods = [...document.querySelectorAll(".rvform [data-m]:checked")].map((c) => c.dataset.m);
    post("/api/wf/reverb", {
      line: $("rvline").value, n_mc: +$("rvmc").value || 1000, detrend: $("rvdetrend").checked, methods,
      primary: methods.includes(LX.adopt) ? LX.adopt : null,
      window: LX.S[LX.A] && LX.S[LX.A].window && LX.S[LX.B] && LX.S[LX.B].line === $("rvline").value ? LX.S[LX.A].window : null,
      mjd_min: $("rvmin").value.trim() || null, mjd_max: $("rvmax").value.trim() || null,
    });
  };
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
// the leftover flux (data - model) around each suggestion of Checks (GET /api/wf/check_snips)
const CS = { sig: null, data: null, loading: null };
function checkSnips(part) {
  const L = S.wf.linescan, Rz = S.wf.residuals, sig = JSON.stringify([L && L.id, Rz && Rz.id, rk]);
  if (CS.sig === sig) return CS.data && CS.data[part];
  if (CS.loading !== sig) {
    CS.loading = sig;
    api(`/api/wf/check_snips?k=${rk}`).then((d) => { if (CS.loading !== sig) return; CS.sig = sig; CS.data = d; $("lsbox").dataset.sig = ""; $("residtbl").dataset.sig = ""; renderLines(); renderResid(); })
      .catch(() => { CS.sig = sig; CS.data = null; });
  }
  return null;
}
function snipSvg(sn, o = {}) {
  const W = 300, H = 76, P = 4;
  const xs = sn.x.concat(sn.add ? sn.add.x : []), x0 = Math.min(...xs), x1 = Math.max(...xs);
  let lo = 0, hi = 0;
  sn.d.forEach((d, i) => { lo = Math.min(lo, d - sn.e[i]); hi = Math.max(hi, d + sn.e[i]); });
  if (sn.add) sn.add.y.forEach((v) => { hi = Math.max(hi, v); });
  const pad = (hi - lo) * 0.08 || 1;
  lo -= pad; hi += pad;
  const X = (x) => (P + (x - x0) / (x1 - x0 || 1) * (W - 2 * P)).toFixed(1), Y = (y) => (P + (hi - y) / (hi - lo) * (H - 2 * P)).toFixed(1);
  let g = "";
  if (o.band) g += `<rect x="${X(o.band[0])}" y="0" width="${Math.max(1, X(o.band[1]) - X(o.band[0])).toFixed(1)}" height="${H}" fill="${o.tint || "rgba(212,71,42,.12)"}"/>`;
  g += `<line x1="0" x2="${W}" y1="${Y(0)}" y2="${Y(0)}" stroke="#cfd3d9" stroke-width="1"/>`;
  // runs of pixels without a masked gap: no line across a gap
  const dx = sn.x.slice(1).map((x, i) => x - sn.x[i]).sort((a, b) => a - b), gap = 3 * (dx[dx.length >> 1] || 1), runs = [[]];
  sn.x.forEach((x, i) => { if (i && x - sn.x[i - 1] > gap) runs.push([]); runs[runs.length - 1].push(i); });
  for (const run of runs) {
    if (run.length < 2) continue;
    const up = run.map((i) => `${X(sn.x[i])},${Y(sn.d[i] + sn.e[i])}`), dn = run.map((i) => `${X(sn.x[i])},${Y(sn.d[i] - sn.e[i])}`).reverse();
    g += `<polygon points="${up.concat(dn).join(" ")}" fill="#9b9a95" opacity=".22"/>`;
    g += `<polyline points="${run.map((i) => `${X(sn.x[i])},${Y(sn.d[i])}`).join(" ")}" fill="none" stroke="#4a5160" stroke-width="1.1" stroke-linejoin="round"/>`;
  }
  if (sn.add) g += `<polygon points="${X(sn.add.x[0])},${Y(0)} ${sn.add.x.map((x, i) => `${X(x)},${Y(sn.add.y[i])}`).join(" ")} ${X(sn.add.x[sn.add.x.length - 1])},${Y(0)}" fill="rgba(212,71,42,.13)" stroke="#d4472a" stroke-width="1.4" stroke-dasharray="4 2"/>`;
  if (o.split && sn.now !== undefined) {
    g += `<line x1="${X(sn.now)}" x2="${X(sn.now)}" y1="0" y2="${H}" stroke="#7b8290" stroke-width="1" stroke-dasharray="2 2"/>`;
    if (sn.to !== undefined) {
      const y = 9, a = +X(sn.now), b = +X(sn.to), dir = b > a ? -1 : 1;
      g += `<line x1="${X(sn.to)}" x2="${X(sn.to)}" y1="0" y2="${H}" stroke="#d4472a" stroke-width="1.3"/>` +
        `<line x1="${a}" x2="${b}" y1="${y}" y2="${y}" stroke="#d4472a" stroke-width="1.3"/><path d="M${b},${y} l${dir * 5},-3.5 v7 z" fill="#d4472a"/>`;
    }
  }
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="data minus model around the feature">${g}</svg>`;
}
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
  // no analysis of this fit (a new fit makes the last one stale): back to the description, not the old outcome
  if (!$("residnote").dataset.intro) $("residnote").dataset.intro = $("residnote").textContent;
  if (!Rz) { if ($("residtbl").dataset.sig) { $("residtbl").dataset.sig = ""; $("residnote").textContent = $("residnote").dataset.intro; $("residfoot").classList.add("hidden"); } return; }
  const sig = JSON.stringify([Rz.id, [...appliedRes], S.busy]);
  if ($("residtbl").dataset.sig === sig) return;
  $("residtbl").dataset.sig = sig;
  $("residnote").textContent = Rz.rows.length ? `${Rz.rows.length} feature${Rz.rows.length > 1 ? "s" : ""} above the threshold` +
    (Rz.scope === "all" && many ? ` in ${Rz.n_epochs} epochs (most frequent first; the plot marks those of the shown epoch).` : ".")
    : "No significant residual features: the model describes the data down to the noise.";
  const snR = checkSnips("resid");
  $("residtbl").innerHTML = Rz.rows.map((r) => {
      const done = appliedRes.has(`${Rz.id}:${r.i}`), sn = snR && snR[r.i];
      return `<div class="sgcard${done ? " applied" : ""}"><div class="sghead"><span class="sgno">${r.i + 1}</span>` +
        `<b>${r.line ? esc(prettyLine(r.line)) : `${fmtNum(r.center, 5)} Å`}</b>${r.in_model ? '<small class="muted">in model</small>' : ""}<span class="grow"></span><span class="badge k-${r.kind}">${esc(r.kind)}</span></div>` +
        `<div class="sgplot">${sn ? snipSvg(sn, { band: [r.lo, r.hi], tint: RES_TINT[r.kind] }) + (many && sn.ep !== undefined ? `<span class="sgep">epoch #${sn.ep}</span>` : "") : '<span class="muted small">…</span>'}</div>` +
        `<div class="sgmeta"><span><b>${r.sigma.toFixed(1)}σ</b> significance</span><span>${fmtNum(r.center, 5)} Å</span><span>FWHM ${Math.round(r.fwhm)} km/s</span>` +
        (many && Rz.scope === "all" ? `<span>in ${r.epochs.length}/${Rz.n_epochs} epochs</span>` : "") + `</div>` +
        `<p class="sgtext">${esc(r.suggestion)}</p>${r.code ? `<code class="sgcode">${esc(r.code)}</code>` : ""}` +
        (r.action ? `<div class="sgfoot"><button class="btn small${done ? "" : " primary"}" data-apply="${r.i}" ${done || S.busy ? "disabled" : ""}>${done ? "Applied ✓" : "Apply"}</button></div>` : "") + `</div>`;
    }).join("");
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
  else if (a.type === "split") ok = await post("/api/model", { action: "split", component: a.component, lines: a.lines, name: a.name });
  else if (a.type === "mask") {
    if (!prep) prep = prepDefaults();
    prep.masks.push(a.range);
    ok = await post("/api/wf/prepare", Object.assign({}, prep, { apply_to: "all" }));
  }
  if (ok) { appliedRes.add(`${id}:${r.i}`); toast("Applied: " + (r.code || r.suggestion)); renderResid(); }
}
// ------------------------------------------------------------- missing lines (fa.suggest_lines, instant)
const LS_TYPE = { forbidden: "forbidden", coronal: "coronal", "semi-forbidden": "semi-forbidden", permitted: "permitted", recombination: "recombination", bowen: "Bowen" };
const lsName = (r) => r.lines.length > 1 ? `${prettyLine(r.lines[0])} + ${prettyLine(r.lines[r.lines.length - 1]).replace(/^\[?[A-Za-z ]+\]?\s*/, "")}` : prettyLine(r.line);
function renderLines() {
  const L = S.wf.linescan, box = $("lsbox");
  box.classList.toggle("hidden", !L);
  if (!L) return;
  const cell = S.cells[S.cells.length - 1];
  const running = !!(S.busy && cell && cell.code.startsWith("# add lines and refit"));
  if (lsPick.id !== L.id) { lsPick.id = L.id; lsPick.set.clear(); }      // a new search: nothing selected
  const sig = JSON.stringify([L.id, L.rows.map((r) => r.added), [...lsPick.set], !!S.busy, running, S.progress && [S.progress.task, S.progress.done]]);
  if (box.dataset.sig === sig) return;
  box.dataset.sig = sig;
  const epoch = L.trial !== null && L.trial !== undefined, multi = !!L.n_epochs;
  $("lshint").textContent = multi ? `fa.suggest_lines(${(fitDone() || {}).kind === "fits" ? "fits[k]" : "me.epochs[k]"}) for every k` : "fa.suggest_lines(r)";
  $("lsrefit").classList.add("hidden");
  if (running) {                     // the rows of the fit being replaced stay, greyed, until the new fit is in
    $("lsnote").textContent = `Changing ${lsAdding.map(prettyLine).join(", ")} and fitting ${epoch ? `epoch #${L.trial} ` : ""}again with the options of the last fit…` +
      (S.progress && S.progress.task ? ` ${S.progress.task}${S.progress.total ? ` ${S.progress.done ?? 0} / ${S.progress.total}` : ""}` : "");
  }
  box.classList.toggle("busy", running);
  $("lstbl").parentElement.classList.toggle("hidden", !L.rows.length);
  if (!L.rows.length) {
    if (!running) $("lsnote").textContent = "None: no library line the model lacks would lower the BIC by more than 10 at the width and shift of the model's line components, and the lines of each component share its velocity.";
    $("lsall").classList.add("hidden");
    return;
  }
  const left = L.rows.filter((r) => !r.added);
  const nAdd = left.filter((r) => r.action === "add").length, nSplit = left.length - nAdd;
  const what = [nAdd ? `${plural(nAdd, "library line", "library lines")} the model lacks` : "",
    nSplit ? `${plural(nSplit, "line", "lines")} of the model whose velocity differs from the other lines of ${nSplit > 1 ? "their" : "its"} component` : ""].filter(Boolean).join(", and ");
  if (!running) $("lsnote").textContent = left.length
    ? `Changes that lower the BIC by more than 10, measured on the residuals of ${multi ? `each of the ${L.n_epochs} epochs, most frequent first (the sketch: the shown epoch, or the epoch where the change helps most)` : "this fit"} (no refit): ${what}. ` +
      (nAdd ? `${multi ? "An added line takes" : "Add & refit adds a line with"} the width and shift of the component it was found with (one parameter, its flux). ` : "") +
      (nSplit ? `${multi ? "A separated line moves" : "Separate & refit moves the line"} (with its doublet) to a component of its own, with its own shift and width (Model.split_lines). ` : "") +
      (multi ? "Select the changes you want: they are made together and all epochs are fitted again once, with the options of the last fit (and any change applied below); the list comes from the new fit."
        : `Then it fits again with the options of the last fit${epoch ? ` (epoch #${L.trial} only, the trial fit; the epochs are fitted together from the Fit step)` : ""}; the list comes from the new fit.`)
    : "All of them are in the model, but this fit is older than the model: fit again to see the effect.";
  const picked = left.filter((r) => lsPick.set.has(r.line));
  if (multi) {
    $("lsall").disabled = !picked.length || !!S.busy;
    $("lsall").classList.toggle("hidden", !left.length && !running);
    $("lsall").classList.toggle("primary", !!picked.length);
    $("lsall").textContent = running ? "Refitting…" : picked.length ? `Apply ${plural(picked.length, "change", "changes")} & refit all epochs` : "Select changes to apply";
    $("lspick").classList.toggle("hidden", left.length < 2 || running);
    $("lspick").textContent = picked.length === left.length ? "Clear" : "Select all";
  } else {
    $("lspick").classList.add("hidden");
    $("lsall").classList.remove("primary");
    $("lsall").disabled = !left.length || !!S.busy;
    $("lsall").classList.toggle("hidden", left.length < 2);
    $("lsall").textContent = running && lsAdding.length > 1 ? "Refitting…" : "Apply all & refit";
  }
  if (!left.length && !running) { $("lsrefit").classList.remove("hidden"); $("lsrefit").disabled = !!S.busy; }
  const snL = checkSnips("lines"), maxB = Math.max(30, ...L.rows.map((r) => Math.abs(r.dbic || 0)));
  $("lstbl").innerHTML = L.rows.map((r, i) => {
      const now = running && lsAdding.includes(r.line), split = r.action === "split", sn = snL && snL[i];
      const change = split
        ? `own velocity <b>${r.dv > 0 ? "+" : "−"}${Math.abs(r.dv).toFixed(0)} km/s</b> from <span class="mono">${esc(r.component)}</span>`
        : `width and shift of <span class="mono">${esc(r.component)}</span> <small class="muted">${esc(r.role)}</small>`;
      const pick = multi && lsPick.set.has(r.line) && !r.added;
      const btn = now ? "Refitting…" : r.added ? (split ? "Separated ✓" : "Added ✓") : multi ? (pick ? "Selected ✓" : "Select") : (split ? "Separate & refit" : "Add & refit");
      const db = Math.abs(r.dbic || 0), w = Math.log10(1 + db) / Math.log10(1 + maxB) * 100, tick = Math.log10(11) / Math.log10(1 + maxB) * 100;
      return `<div class="sgcard${r.added || now ? " applied" : ""}${now ? " running" : ""}${pick ? " picked" : ""}"><div class="sghead"><b>${esc(lsName(r))}</b><small class="muted">${esc(r.ion)} · ${esc(LS_TYPE[r.type] || r.type)}</small><span class="grow"></span>` +
        `<span class="sgact ${split ? "split" : "add"}">${split ? "separate" : "add"}</span></div>` +
        `<div class="sgplot">${sn ? snipSvg(sn, { split }) + (multi && sn.ep !== undefined ? `<span class="sgep">epoch #${sn.ep}</span>` : "") : '<span class="muted small">…</span>'}</div>` +
        `<div class="sgbic" title="how much the BIC drops with this change; more than 10 (the tick) is strong evidence"><span>ΔBIC</span><em><i style="width:${w.toFixed(1)}%"></i><b style="left:${tick.toFixed(1)}%"></b></em><strong>−${db.toFixed(0)}</strong></div>` +
        `<div class="sgmeta"><span>${r.wave.toFixed(1)} Å</span>${r.snr === null ? "" : `<span>S/N <b>${r.snr.toFixed(1)}</b>${multi ? ` <small class="muted">(#${r.ep})</small>` : ""}</span>`}` +
        `${multi && r.epochs ? `<span title="epochs where it lowers the BIC by more than 10: ${r.epochs.map((k) => "#" + k).join(", ")}">in <b>${r.epochs.length}/${L.n_epochs}</b> epochs</span>` : ""}</div>` +
        `<p class="sgtext">${split ? "Give it " : "Add it with the "}${change}</p>` +
        `<div class="sgfoot"><button class="btn small${r.added || now || (multi && !pick) ? "" : " primary"}" data-ls="${i}" ${r.added || S.busy ? "disabled" : ""}${multi ? ` aria-pressed="${pick}"` : ""}>${btn}</button></div></div>`;
    }).join("");
  $("lstbl").querySelectorAll("[data-ls]").forEach((b) => (b.onclick = () => {
    const ln = L.rows[+b.dataset.ls].line;
    if (!multi) return addLines([ln]);
    lsPick.set.has(ln) ? lsPick.set.delete(ln) : lsPick.set.add(ln);    // a multi-epoch refit is expensive: collect, then apply once
    renderLines();
  }));
}
let lsAdding = [];
const lsPick = { id: null, set: new Set() };       // multi-epoch: the suggestions chosen for one refit
async function addLines(lines) {
  lsAdding = lines;
  if (await post("/api/wf/addlines", { lines, refit: true })) toast(`Changed ${lines.map(prettyLine).join(", ")}: fitting again…`);
}
// ------------------------------------------------------------- robustness
let rbShown = "";
// numbers to the precision of the error bar (two significant digits of the error)
const rbDec = (err) => (err && isFinite(err) && err > 0 ? Math.max(0, 1 - Math.floor(Math.log10(err))) : null);
const rbNum = (v, err) => { const d = rbDec(err); return d === null ? fmtNum(v, 4) : v.toFixed(Math.min(d, 6)); };
const rbFmt = (v, u, err) => (v === null || v === undefined ? "—" : rbNum(v, err) + (u ? " " + u : ""));
const rbRange = (lo, hi, u, err) => (lo === null || hi === null ? "—" : `${rbNum(lo, err)} – ${rbNum(hi, err)}${u ? " " + u : ""}`);
const rbText = (s) => esc(s).replace(/dchi2/g, "Δχ²");
const RB_BADGE = { "ok": ["ok", "stable"], "fragile FWHM": ["warn", "fragile: shoulder at half max"], "not bounded": ["missing", "not bounded"], "error bar too small": ["missing", "error bar too small"], "no error bar": ["", "no error bar"],
  "measured": ["ok", "measured"], "weakly bounded": ["warn", "weakly bounded"], "best at its limit": ["warn", "at its limit"],
  "two solutions": ["missing", "two solutions"], "second minimum": ["warn", "second minimum"] };
function rbBadge(flag) {
  const b = RB_BADGE[flag] || (flag.startsWith("not excluded") ? ["missing", "slides to its limit"] : ["", flag]);
  return `<span class="badge ${b[0]}">${esc(b[1])}</span>`;
}
function rbScan(p) {                              // dchi2 against the fixed value, clipped at 12, with the 68% and 95% levels
  const pts = p.scan.filter((s) => s[0] !== null && s[1] !== null);
  if (pts.length < 2) return "";
  const W = 150, H = 44, top = 12, x0 = Math.min(...pts.map((s) => s[0])), x1 = Math.max(...pts.map((s) => s[0]));
  const X = (v) => 3 + (W - 6) * (x1 > x0 ? (v - x0) / (x1 - x0) : 0.5), Y = (d) => 3 + (H - 6) * (1 - Math.min(Math.max(d, 0), top) / top);
  const path = pts.map((s, i) => `${i ? "L" : "M"}${X(s[0]).toFixed(1)},${Y(s[1]).toFixed(1)}`).join("");
  return `<svg class="rbscan" viewBox="0 0 ${W} ${H}" role="img" aria-label="chi-square scan of ${esc(p.label)}"><title>Δχ² when ${esc(p.label)} is fixed (refitting everything else), from ${fmtNum(x0, 4)} to ${fmtNum(x1, 4)}; dotted: Δχ² = 1 and 4</title>` +
    `<line class="lev" x1="0" x2="${W}" y1="${Y(1)}" y2="${Y(1)}"/><line class="lev" x1="0" x2="${W}" y1="${Y(4)}" y2="${Y(4)}"/>` +
    `<path d="${path}" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>`;
}
function rbParamRow(p) {
  return `<tr><td><b>${esc(p.label)}</b><br><span class="rbnote mono">${esc(p.name)}</span></td><td class="num">${rbFmt(p.value, p.unit, p.error)}</td>` +
    `<td class="num">${rbRange(p.lo68, p.hi68, "", p.error)}</td><td class="num">${rbRange(p.lo95, p.hi95, "", p.error)}</td><td>${rbScan(p)}</td>` +
    `<td>${rbBadge(p.flag)}${p.note ? `<br><span class="rbnote">${rbText(p.note)}</span>` : ""}</td></tr>`;
}
function renderRobust() {
  const R = S.wf.robust;
  if (!R) return;
  const cell = S.cells[S.cells.length - 1];
  const running = S.busy && cell && cell.code.startsWith("# robustness");
  const p = S.progress;
  $("robrun").disabled = !!S.busy;
  $("robrun").textContent = running ? (p && p.task === "robustness scans" && p.total ? `Checking… ${p.done || 0}/${p.total} fits`
    : p && p.task === "robustness: results" && p.total ? `Scanning results… ${p.done || 0}/${p.total}` : "Checking…") : R.id ? "Check again" : "Check robustness";
  $("robout").classList.toggle("hidden", !R.id);
  if (!R.id) return;
  const sig = JSON.stringify([R.id, !!S.busy]);
  if (rbShown === sig) return;
  rbShown = sig;
  const shaky = R.quantities.filter((q) => q.flag !== "ok");
  const flagged = R.params.filter((q) => q.flag !== "measured"), fine = R.params.filter((q) => q.flag === "measured");
  const head = !R.quantities.length ? "" : shaky.length
    ? `${shaky.length} of your ${R.quantities.length} results change more between acceptable fits than their error bars say.`
    : `Your ${R.quantities.length} results are stable within this model: they change no more than their error bars between acceptable fits. Another continuum or line profile can still move them: test that with Compare with another fit below.`;
  const better = R.improved ? `<div class="rbwarn"><b>The scans found a better fit</b> (χ² ${fmtNum(R.chi2[0], 6)} → ${fmtNum(R.chi2[1], 6)}): your fit is not the best solution.
      <div style="margin-top:6px"><button class="btn primary small" id="robapply">Fit again from the better solution →</button></div></div>` : "";
  const qrows = R.quantities.map((q) => `<tr><td><b>${esc(q.label)}</b></td><td class="num">${rbNum(q.value, q.error)}${q.error !== null ? ` ± ${fmtNum(q.error, 2)}` : ""}${q.unit ? " " + q.unit : ""}</td>` +
    `<td class="num">${rbRange(q.lo68, q.hi68, q.unit, q.error)}</td><td class="num">${rbRange(q.lo95, q.hi95, q.unit, q.error)}</td><td>${rbScan(q)}</td><td>${rbBadge(q.flag)}</td></tr>`).join("");
  const phead = '<thead><tr><th>Parameter</th><th class="num">Best fit</th><th class="num">68% interval</th><th class="num">95% interval</th><th>Δχ² scan</th><th></th></tr></thead>';
  $("robout").innerHTML = `<div class="rbhead">${head || `${R.params.length} parameters scanned.`}</div>${better}` +
    (qrows ? `<div class="rbgroup"><h4>Your results over all acceptable fits</h4><div class="tblwrap"><table class="tbl"><thead><tr><th>Result</th><th class="num">Best fit ± error</th><th class="num">68% range</th><th class="num">95% range</th><th>Δχ² scan</th><th></th></tr></thead><tbody>${qrows}</tbody></table></div>` +
      `<p class="note">${R.quantities.some((q) => q.flag === "fragile FWHM") ? "<i>Fragile</i>: the broad profile has a shoulder near half maximum, so a small change of the fit (another continuum, another decomposition) can move the FWHM a lot; σ is the safer width. " : ""}Acceptable: Δχ² ≤ 1 (68%) or ≤ 4 (95%) from the best fit. The ranges come from ${R.n_fits} refits; the scans visit many but not all acceptable fits, so a range can be wider, never narrower. Results come from the whole line profile, so they do not depend on how the line is split into components.</p></div>` : "") +
    (flagged.length ? `<div class="rbgroup"><h4>Parameters to look at (${flagged.length})</h4><div class="tblwrap"><table class="tbl">${phead}<tbody>${flagged.map(rbParamRow).join("")}</tbody></table></div>` +
      `<p class="note"><i>Weakly bounded</i>: the data prefer the best value, but values up to a limit are only a few Δχ² worse. <i>Slides to its limit</i>: not excluded at 95% up to a limit, so the limit sets the error. <i>Second minimum / two solutions</i>: another value fits almost as well. These values depend on the model and its limits; report them with care.</p></div>`
      : `<p class="note">Every scanned parameter is measured: a closed 95% interval and no second solution.</p>`) +
    (R.alternatives && R.alternatives.length ? `<div class="rbgroup"><h4>Other solutions found (${R.alternatives.length})</h4>` +
      R.alternatives.map((a) => `<div class="rbalt"><span class="grow">${esc(a.label)} = ${fmtNum(a.value, 4)}: a second minimum at Δχ² = ${a.dchi2.toFixed(1)}</span>` +
        `<button class="btn small" data-alt="${a.i}" ${S.busy ? "disabled" : ""}>Pin to compare</button></div>`).join("") +
      `<p class="note">Pin a solution, then compare it with the current fit below to see what changes.</p></div>` : "") +
    (fine.length ? `<details class="rbmore"><summary>Measured parameters (${fine.length})</summary><div class="tblwrap"><table class="tbl">${phead}<tbody>${fine.map(rbParamRow).join("")}</tbody></table></div></details>` : "");
  $("robout").querySelectorAll("[data-alt]").forEach((b) => (b.onclick = () => post("/api/wf/pin_alternative", { i: +b.dataset.alt })));
  if ($("robapply")) $("robapply").onclick = async () => {
    if (await post("/api/wf/robust_apply", {})) { toast("Model set to the better solution: fit it again"); go("fit"); }
  };
}
function initRobust() {
  $("robrun").onclick = () => post("/api/wf/robust", {});
}
// ------------------------------------------------------------- simulation check
let simShown = "";
const SIM_BADGE = { "ok": ["ok", "recovered"], "biased": ["missing", "biased"], "error bar too small": ["warn", "error bar too small"],
  "error bar too large": ["warn", "error bar too large"], "no error bar": ["", "no error bar"], "not measured": ["", "not measured"] };
function renderSim() {
  const M = S.wf.simcheck, kind = fitDone() && fitDone().kind;
  $("simcard").classList.toggle("hidden", !(M && kind === "single"));
  if (!M || kind !== "single") return;
  const cell = S.cells[S.cells.length - 1];
  const running = S.busy && cell && cell.code.startsWith("# simulation check");
  const p = S.progress;
  $("simrun").disabled = !!S.busy;
  $("simrun").textContent = running ? (p && p.task === "simulation check" && p.total ? `Fitting… ${p.done || 0}/${p.total}` : "Simulating…") : M.id ? "Run again" : "Run simulations";
  $("simout").classList.toggle("hidden", !M.id);
  if (!M.id) return;
  const sig = JSON.stringify([M.id, !!S.busy]);
  if (simShown === sig) return;
  simShown = sig;
  const biased = M.rows.filter((r) => r.flag === "biased"), errs = M.rows.filter((r) => r.flag.startsWith("error bar"));
  const head = biased.length ? `${biased.length} of ${M.rows.length} results come back shifted: the fit is biased for them at this signal-to-noise ratio.`
    : errs.length ? `No result is biased, but ${errs.length} of ${M.rows.length} have an error bar of the wrong size.`
      : `All ${M.rows.length} results come back unbiased, with error bars of the right size.`;
  const pct = (r) => (r.bias !== null && r.value && r.scatter !== null && Math.abs(r.value) > 3 * r.scatter ? ` <span class="rbnote">(${r.bias > 0 ? "+" : ""}${(100 * r.bias / Math.abs(r.value)).toFixed(1)}%)</span>` : "");
  const rows = M.rows.map((r) => { const b = SIM_BADGE[r.flag] || ["", r.flag]; const u = r.unit ? " " + r.unit : "";
    return `<tr><td><b>${esc(r.label)}</b></td><td class="num">${r.value === null ? "—" : rbNum(r.value, r.scatter) + u}</td>` +
      `<td class="num">${r.mean === null ? "—" : rbNum(r.mean, r.scatter) + u}${pct(r)}</td>` +
      `<td class="num">${r.bias_sd === null ? "—" : (r.bias_sd > 0 ? "+" : "") + r.bias_sd.toFixed(2)}</td>` +
      `<td class="num">${r.scatter === null ? "—" : fmtNum(r.scatter, 2)}</td><td class="num">${r.error === null ? "—" : fmtNum(r.error, 2)}</td>` +
      `<td class="num">${r.cover68 === null ? "—" : Math.round(100 * r.cover68) + "%"}</td><td><span class="badge ${b[0]}">${esc(b[1])}</span></td></tr>`; }).join("");
  $("simout").innerHTML = `<div class="rbhead">${head}</div>` +
    `<p class="note">${M.n} simulated spectra fitted${M.failed ? ` (${M.failed} could not be fitted)` : ""}, median χ²<sub>ν</sub> ${M.rchi2 === null ? "—" : M.rchi2.toFixed(2)}` +
    `${M.pruned !== null ? `; every simulation was pruned of undetected Fe II groups at ${M.pruned}σ, as this fit was` : ""}.</p>` +
    `<div class="tblwrap"><table class="tbl"><thead><tr><th>Result</th><th class="num">In the fit (the truth)</th><th class="num">Mean of the simulations</th>` +
    `<th class="num" title="mean minus truth, in units of the scatter">Bias / scatter</th><th class="num" title="standard deviation over the simulations: the real statistical error">Scatter</th>` +
    `<th class="num" title="median error bar reported by the simulated fits">Error bar</th><th class="num" title="how often the truth is within one error bar; 68% for correct errors">Truth within error</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>` +
    `<p class="note"><i>Biased</i>: the mean of the simulations differs from the truth by more than 0.3 of the scatter, and by more than chance allows with ${M.n} spectra. ` +
    `Compare <i>Scatter</i> with <i>Error bar</i>: the scatter is what the result really varies by.` +
    (biased.some((r) => /Fe II/.test(r.label)) && M.pruned === null && S.wf.feii ? " Fe II is among the biased results: try check 4, then run this again to see whether it helped." : "") + `</p>`;
}
function initSim() {
  $("simrun").onclick = () => post("/api/wf/simcheck", { n: +$("simn").value });
  $("calrun").onclick = () => post("/api/wf/calcheck", {});
  $("jsimrun").onclick = () => post("/api/wf/simcheck", { n: +$("jsimn").value });
}
// ------------------------------------------------------------- checks of a joint fit
let jointShown = "";
function renderJointChecks() {
  const J = S.wf.jointcheck;
  $("jointcard").classList.toggle("hidden", !J);
  if (!J) return;
  const cell = S.cells[S.cells.length - 1], p = S.progress;
  const calRunning = S.busy && cell && cell.code.startsWith("# calibration check");
  const simRunning = S.busy && cell && cell.code.startsWith("# simulation check");
  const prog = (task) => (p && p.task === task && p.total ? ` ${p.done || 0}/${p.total}` : "");
  $("calrun").disabled = !!S.busy || !J.intercalibrated;
  $("calrun").title = J.intercalibrated ? "" : "the epochs were not intercalibrated";
  $("calrun").textContent = calRunning ? "Fitting…" + prog("calibration check") : J.cal.id ? "Check again" : "Check the calibration";
  $("jsimrun").disabled = !!S.busy;
  $("jsimrun").textContent = simRunning ? "Simulating…" + prog("simulation check") : J.sim.id ? "Run again" : "Run simulations";
  const sig = JSON.stringify([J.cal.id, J.sim.id, !!S.busy]);
  if (jointShown === sig) return;
  jointShown = sig;
  const C = J.cal;
  $("calout").classList.toggle("hidden", !C.id);
  if (C.id) {
    const bad = C.rows.filter((r) => r.flag !== "ok" && r.flag !== "not measured");
    const head = bad.length ? `${plural(bad.length, "shared line scatters", "shared lines scatter")} more than the errors between the epochs: about ${(100 * (C.excess || 0)).toFixed(1)}% calibration noise is not in the error bars.`
      : `The shared lines agree within their errors in all ${J.n_epochs} epochs: the flux scales are consistent.`;
    const pct = (v) => (v === null ? "—" : (100 * v).toFixed(1) + "%");
    const rows = C.rows.map((r) => `<tr><td><b>${esc(r.label)}</b></td><td class="num">${r.snr === null ? "—" : Math.round(r.snr)}</td>` +
      `<td class="num">${r.rchi2 === null ? "—" : r.rchi2.toFixed(2)}</td><td class="num">${r.p === null ? "—" : r.p < 1e-4 ? "<10⁻⁴" : r.p.toPrecision(2)}</td>` +
      `<td class="num">${pct(r.scatter)}</td><td class="num">${pct(r.error)}</td><td class="num">${pct(r.excess)}</td>` +
      `<td><span class="badge ${r.flag === "ok" ? "ok" : r.flag === "not measured" ? "" : "warn"}">${esc(r.flag === "scatters more than its errors" ? "scatters" : r.flag)}</span></td></tr>`).join("");
    const outl = C.outliers.length ? `<p class="note">Epochs off by more than 3σ (${C.n_outliers}): ` +
      C.outliers.map((o) => `<b>${o.epoch}</b> <span class="muted">${esc(o.name)}</span> ${esc(o.line)} ${o.pull > 0 ? "+" : ""}${o.pull.toFixed(1)}σ`).join(" · ") +
      `${C.n_outliers > C.outliers.length ? " …" : ""}. Look at them in the light curve; leave them out or fit them on their own if their spectrum is bad.</p>` : "";
    $("calout").innerHTML = `<div class="rbhead">${head}</div>` +
      `<div class="tblwrap"><table class="tbl"><thead><tr><th>Shared line</th><th class="num">S/N</th><th class="num" title="scatter of the epochs' own fluxes around their mean, in units of their errors">χ²/dof</th><th class="num">p</th>` +
      `<th class="num" title="standard deviation of the epochs' own fluxes">Scatter</th><th class="num" title="median error of one epoch">Error</th>` +
      `<th class="num" title="scatter to add in quadrature to the errors for χ²/dof = 1: calibration noise not in the error bars">Excess</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>` + outl +
      `<p class="note">The flux-scale error of the joint fit is ${pct(C.calibration_error)} per epoch (median).` +
      (bad.length ? " Try fitting an offset and a broadening per epoch (Fit step), or a tilt in the calibration; if the scatter stays, add the excess to the light-curve errors." : "") +
      (C.failed ? ` ${C.failed} epochs could not be fitted on their own.` : "") + `</p>`;
  }
  const M = J.sim;
  $("jsimout").classList.toggle("hidden", !M.id);
  if (M.id) {
    const biased = M.rows.filter((r) => r.flag === "biased"), errs = M.rows.filter((r) => r.flag.startsWith("error bar"));
    const head = biased.length ? `${biased.length} of ${M.rows.length} results come back shifted in the simulations.`
      : errs.length ? `Nothing is biased, but ${errs.length} of ${M.rows.length} have an error bar of the wrong size.`
        : `All ${M.rows.length} results come back unbiased, with error bars of the right size.`;
    const rows = M.rows.map((r) => { const b = SIM_BADGE[r.flag] || ["", r.flag];
      return `<tr><td><b>${esc(r.label)}</b></td><td class="num">${r.bias_sd === null ? "—" : (r.bias_sd > 0 ? "+" : "") + r.bias_sd.toFixed(2)}</td>` +
        `<td class="num">${r.spread === null ? "—" : r.spread.toFixed(2)}</td><td class="num">${r.cover68 === null ? "—" : Math.round(100 * r.cover68) + "%"}</td>` +
        `<td class="num">${r.n}</td><td><span class="badge ${b[0]}">${esc(b[1])}</span></td></tr>`; }).join("");
    $("jsimout").innerHTML = `<div class="rbhead">${head}</div>` +
      `<div class="tblwrap"><table class="tbl"><thead><tr><th>Result</th><th class="num" title="mean of (fit − truth) / error bar">Bias / error bar</th>` +
      `<th class="num" title="standard deviation of (fit − truth) / error bar: 1 for right error bars">Scatter / error bar</th>` +
      `<th class="num" title="how often the truth is within one error bar; 68% for correct errors">Truth within error</th><th class="num" title="values compared (epochs × sets)">n</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>` +
      `<p class="note">${M.n} sets of ${J.n_epochs} epochs fitted${M.failed ? ` (${M.failed} could not be fitted)` : ""}. A scatter above 1 means the error bars are too small by that factor.</p>`;
  }
}
// ------------------------------------------------------------- weak Fe II
let feShown = "";
function renderFeii() {
  const F = S.wf.feii, kind = fitDone() && fitDone().kind;
  const show = F && (kind === "single" || (kind === "fits" && F.kind === "fits"));
  $("feiicard").classList.toggle("hidden", !show);
  if (!show) return;
  const cell = S.cells[S.cells.length - 1];
  const running = S.busy && cell && cell.code.startsWith("# prune Fe II");
  const p = S.progress;
  $("feiirun").disabled = !!S.busy || !F.errors || (!F.pruned && !F.weak);
  $("feiirun").textContent = running ? (p && p.task === "pruning Fe II" && p.total ? `Fitting again… ${p.done || 0}/${p.total}` : "Fitting again…")
    : F.pruned ? (F.kind === "fits" ? "Prune again from the full fits" : "Prune again from the full fit") : "Drop undetected groups";
  const sig = JSON.stringify([F.id, !!S.busy]);
  if (feShown === sig) return;
  feShown = sig;
  if (F.kind === "fits") {                       // epochs fitted one by one: a line per epoch
    if (!F.errors) { $("feiiout").innerHTML = '<p class="note">These fits have no errors, so the groups cannot be judged: fit with errors (Fit step) first.</p>'; return; }
    const n = F.rows.length, withWeak = F.rows.filter((r) => r.weak).length;
    const head = F.pruned ? `${F.weak} Fe II groups were dropped in ${withWeak} of ${n} epochs.`
      : F.weak ? `${F.weak} of ${F.groups} Fe II groups in ${n} epochs are below 2σ (${withWeak} epochs have at least one).`
        : `Every Fe II group of the ${n} epochs is detected (above 2σ): there is nothing to drop.`;
    const erows = F.rows.map((r) => `<tr><td>#${r.k}</td><td><b>${esc(r.date || r.name)}</b></td><td class="num">${r.groups ?? "—"}</td><td class="num">${r.weak ?? "—"}</td>` +
      `<td>${r.weak === null ? "" : r.weak ? `<span class="badge warn">${F.pruned ? "dropped" : "not detected"}</span>` : '<span class="badge ok">all detected</span>'}</td></tr>`).join("");
    $("feiiout").innerHTML = `<div class="rbhead">${head}</div>` +
      (F.pruned ? `<p class="note">The light curves, lags and the Main sequence tab now use the pruned fits. A group dropped in one epoch and kept in the next adds scatter to the Fe II light curve: compare with the fits with every group before trusting a change.</p>` +
        (F.can_undo ? `<div class="row"><span class="grow"></span><button class="btn small" id="feiiundo" ${S.busy ? "disabled" : ""}>Back to the fits with every group</button></div>` : "") : "") +
      `<details class="rbmore" ${n <= 12 ? "open" : ""}><summary>Epochs (${n})</summary><div class="tblwrap mid"><table class="tbl"><thead><tr><th></th><th>Epoch</th><th class="num">Fe II groups</th>` +
      `<th class="num">${F.pruned ? "Dropped" : "Below 2σ"}</th><th></th></tr></thead><tbody>${erows}</tbody></table></div></details>`;
    if ($("feiiundo")) $("feiiundo").onclick = () => post("/api/wf/prune_feii_undo", {});
    return;
  }
  if (!F.errors) { $("feiiout").innerHTML = '<p class="note">This fit has no errors, so the groups cannot be judged: fit with errors (Fit step) first.</p>'; return; }
  const kept = F.rows.filter((r) => r.kept).length;
  const head = F.pruned ? `${F.rows.length - kept} of ${F.rows.length} Fe II groups were dropped; this fit has ${kept}.`
    : F.weak ? `${F.weak} of ${F.rows.length} Fe II groups are below 2σ: the data do not detect them.`
      : `All ${F.rows.length} Fe II groups are detected (above 2σ): there is nothing to drop.`;
  const pm = (v, e, d = 4) => (v === null || v === undefined ? "—" : fmtNum(v, d) + (e !== null && e !== undefined ? ` ± ${fmtNum(e, 2)}` : ""));
  const chg = (a, b) => (a && b ? ` <span class="rbnote">(${b > a ? "+" : ""}${(100 * (b / a - 1)).toFixed(1)}%)</span>` : "");
  let cmp = "";
  if (F.pruned && F.before) {
    const a = F.before, b = F.now, line = F.line ? `Broad ${esc(F.line)}` : "";
    const row = (label, x, y, ch) => `<tr><td><b>${label}</b></td><td class="num">${x}</td><td class="num">${y}${ch || ""}</td></tr>`;
    cmp = `<div class="rbgroup"><h4>What changed</h4><div class="tblwrap"><table class="tbl"><thead><tr><th>Result</th><th class="num">Every group</th><th class="num">Without the undetected groups</th></tr></thead><tbody>` +
      row(esc(a.band_label || "Fe II") + " flux", pm(a.flux, a.flux_err), pm(b.flux, b.flux_err), chg(a.flux, b.flux)) +
      (a.r_feii !== null && a.r_feii !== undefined ? row(esc(a.ratio_label || "R(Fe II)"), pm(a.r_feii, a.r_feii_err, 3), pm(b.r_feii, b.r_feii_err, 3), chg(a.r_feii, b.r_feii)) : "") +
      (line && a.line_flux !== undefined ? row(`${line} flux`, pm(a.line_flux, a.line_flux_err), pm(b.line_flux, b.line_flux_err), chg(a.line_flux, b.line_flux)) +
        row(`${line} FWHM`, pm(a.line_fwhm, a.line_fwhm_err) + " km/s", pm(b.line_fwhm, b.line_fwhm_err) + " km/s", chg(a.line_fwhm, b.line_fwhm)) : "") +
      (a.continuum !== undefined ? row("Continuum", pm(a.continuum, null), pm(b.continuum, null), chg(a.continuum, b.continuum)) : "") +
      row("BIC (lower is better)", fmtNum(a.bic, 6), fmtNum(b.bic, 6), ` <span class="rbnote">(${(b.bic - a.bic).toFixed(1)})</span>`) +
      `</tbody></table></div><p class="note">A lower BIC says the data do not need the dropped groups; it does not say the remaining Fe II is unbiased.</p>` +
      `<div class="row"><span class="grow"></span><button class="btn small" id="feiiundo" ${S.busy ? "disabled" : ""}>Back to the fit with every group</button></div></div>`;
  }
  const rows = F.rows.map((r) => `<tr><td><b>${esc(r.label)}</b><br><span class="rbnote mono">${esc(r.name)}</span></td><td class="num">${pm(r.amplitude, r.error)}</td>` +
    `<td class="num">${r.amplitude !== null && r.error ? (r.amplitude / r.error).toFixed(1) : "—"}</td>` +
    `<td>${r.kept ? '<span class="badge ok">detected</span>' : `<span class="badge warn">${F.pruned ? "dropped" : "not detected"}</span>`}</td></tr>`).join("");
  $("feiiout").innerHTML = `<div class="rbhead">${head}</div>${cmp}` +
    `<details class="rbmore" ${F.pruned || F.weak ? "open" : ""}><summary>Fe II groups (${F.rows.length})${F.pruned ? ", as they were before pruning" : ""}</summary><div class="tblwrap"><table class="tbl">` +
    `<thead><tr><th>Group</th><th class="num">Strength ± error</th><th class="num">Strength / error</th><th></th></tr></thead><tbody>${rows}</tbody></table></div></details>`;
  if ($("feiiundo")) $("feiiundo").onclick = () => post("/api/wf/prune_feii_undo", {});
}
function initFeii() {
  $("feiirun").onclick = () => post("/api/wf/prune_feii", { nsigma: +$("feiinsig").value });
  $("smsqall").onchange = () => drawSampleDiagrams();
}
// ------------------------------------------------------------- pin and compare
let cmpShown = "";
function renderCompare() {
  const C = S.wf.compare;
  if (!C) return;
  const cell = S.cells[S.cells.length - 1];
  const running = S.busy && cell && cell.code.startsWith("# compare");
  $("pinbtn").disabled = !!S.busy || C.pins.some((p) => p.current);
  $("pinbtn").textContent = C.pins.some((p) => p.current) ? "Pinned" : "Pin this fit";
  const sig = JSON.stringify([C, !!S.busy]);
  if (cmpShown === sig) return;
  cmpShown = sig;
  $("pinlist").innerHTML = C.pins.length ? `<div class="tblwrap"><table class="tbl"><thead><tr><th>Pinned fit</th><th class="num">χ²</th><th class="num">Reduced χ²</th><th class="num">Free params</th><th>Components</th><th></th></tr></thead><tbody>` +
    C.pins.map((p, i) => `<tr><td><b>${esc(p.label)}</b>${p.current ? ' <span class="badge ok">current fit</span>' : ""}</td><td class="num">${fmtNum(p.chi2, 6)}</td>` +
      `<td class="num">${p.rchi2 === null ? "—" : p.rchi2.toFixed(3)}</td><td class="num">${p.n_free}</td><td class="small muted">${esc(p.components)}</td>` +
      `<td style="white-space:nowrap">${p.current ? "" : `<button class="btn small primary" data-cmp="${i}" ${S.busy || !p.same ? "disabled" : ""} title="${p.same ? "" : "another spectrum"}">${running ? "Comparing…" : "Compare with current"}</button> `}` +
      `<button class="btn small" data-unpin="${i}" ${S.busy ? "disabled" : ""}>Remove</button></td></tr>`).join("") + "</tbody></table></div>"
    : '<p class="muted small">No pinned fit yet.</p>';
  $("pinlist").querySelectorAll("[data-cmp]").forEach((b) => (b.onclick = () => post("/api/wf/compare", { label: C.pins[+b.dataset.cmp].label })));
  $("pinlist").querySelectorAll("[data-unpin]").forEach((b) => (b.onclick = () => post("/api/wf/unpin", { label: C.pins[+b.dataset.unpin].label })));
  const K = C.cmp;
  $("cmpout").classList.toggle("hidden", !K);
  if (!K) return;
  const st = K.stats, [la, lb] = K.labels;
  const equal = Math.abs(st.dchi2) < Math.max(4, 0.01 * st.chi2[0]) && st.rms < 0.3;
  const head = `${esc(la)} vs ${esc(lb)}: the total models differ by ${st.rms.toFixed(3)} times the error (RMS) on the same ${st.n_pixels} pixels; ` +
    `χ² ${fmtNum(st.chi2[0], 6)} vs ${fmtNum(st.chi2[1], 6)} (Δχ² = ${st.dchi2 >= 0 ? "+" : ""}${st.dchi2.toFixed(1)}), free parameters ${st.n_free[0]} vs ${st.n_free[1]}.` +
    (equal ? " <b>The data cannot tell these fits apart.</b>" : "");
  const rows = K.rows.map((r) => {
    const big = r.sig !== null && Math.abs(r.sig) > 1.5;
    const d = rbDec(Math.min(r.ea || Infinity, r.eb || Infinity));
    const f = (v) => (v === null ? "—" : d === null ? fmtNum(v, 4) : v.toFixed(Math.min(d, 6)));
    return `<tr><td><b>${esc(r.label)}</b></td><td class="num">${f(r.a)}${r.ea !== null ? " ± " + fmtNum(r.ea, 2) : ""}</td>` +
      `<td class="num">${f(r.b)}${r.eb !== null ? " ± " + fmtNum(r.eb, 2) : ""}</td>` +
      `<td class="num">${r.diff === null ? "—" : (r.diff > 0 ? "+" : "") + f(r.diff)}${r.unit ? " " + r.unit : ""}${r.rel !== null ? ` (${r.rel > 0 ? "+" : ""}${Math.round(100 * r.rel)}%)` : ""}</td>` +
      `<td class="num">${r.sig === null ? "—" : `<span class="badge ${big ? "missing" : "ok"}">${Math.abs(r.sig).toFixed(1)}σ</span>`}</td></tr>`;
  }).join("");
  $("cmpout").innerHTML = `<p class="rbhead" style="font-weight:400;font-size:14px">${head}</p>` +
    (K.warnings.length ? `<div class="rbwarn">${K.warnings.map((w) => esc(w)).join("<br>")}</div>` : "") +
    `<div class="tblwrap"><table class="tbl"><thead><tr><th>Result</th><th class="num">${esc(la)}</th><th class="num">${esc(lb)}</th><th class="num">Change</th><th class="num">In errors</th></tr></thead><tbody>${rows}</tbody></table></div>` +
    `<img class="cmpfig" alt="Both total fits with their difference in units of the error, the change of each kind of component, and both broad-line profiles with their half-maximum crossings" src="${withToken(`/api/wf/compare.png?v=${K.id}`)}">` +
    `<div class="ckfoot"><a class="btn small" href="${withToken(`/api/wf/compare.pdf?v=${K.id}`)}" download="comparison.pdf">Download PDF</a>` +
    `<span class="muted small">Left: both fits and their difference; where the flux went between components. Right: the broad ${K.line ? esc(K.line) : "line"} profiles alone, with the half-maximum points of the FWHM.</span></div>`;
}
// ------------------------------------------------------------- the checks as lines that open
const CHECKS = [["residcard", ["# residuals", "# add lines and refit"]], ["robcard", "# robustness"], ["cmpcard", "# compare"], ["feiicard", "# prune Fe II"], ["simcard", "# simulation check"], ["jointcard", ["# calibration check", "# simulation check"]]];
const ckOpen = {}, ckRunning = {};     // the user's choice per card (until a reload), and which card was running
const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;
// the status of each check, from the state the cards already show: state "none" | "ok" | "attention"
function checkStatus() {
  const w = S.wf, out = {};
  const Rz = w.residuals;
  out.residcard = !Rz ? { state: "none", text: "not run" } : Rz.rows.length ? { state: "attention", text: plural(Rz.rows.length, "feature", "features") } : { state: "ok", text: "nothing found" };
  const nL = w.linescan ? w.linescan.rows.filter((r) => !r.added).length : 0;
  if (!Rz && w.linescan && !nL && !w.linescan.rows.length) out.residcard = { state: "none", text: "no lines to add" };
  if (nL) out.residcard = { state: "attention", text: (Rz && Rz.rows.length ? out.residcard.text + " · " : "") + plural(nL, "line change", "line changes") };
  const R = w.robust;
  if (!R || !R.id) out.robcard = { state: "none", text: "not run" };
  else if (R.improved) out.robcard = { state: "attention", text: "better fit found" };
  else {
    const n = R.quantities.filter((q) => q.flag !== "ok").length + R.params.filter((q) => q.flag !== "measured").length;
    out.robcard = n ? { state: "attention", text: `${n} to look at` } : { state: "ok", text: "stable" };
  }
  const C = w.compare;
  out.cmpcard = C && C.cmp ? { state: "ok", text: "compared" } : C && C.pins.length ? { state: "none", text: `${C.pins.length} pinned` } : { state: "none", text: "no pinned fit" };
  const Fe = w.feii;
  if (!Fe) out.feiicard = { state: "none", text: "" };
  else if (!Fe.errors) out.feiicard = { state: "none", text: "needs errors" };
  else if (Fe.pruned) { const n = Fe.kind === "fits" ? Fe.weak : Fe.rows.filter((r) => !r.kept).length; out.feiicard = { state: "ok", text: `${plural(n, "group", "groups")} dropped` }; }
  else if (Fe.weak > 0) out.feiicard = { state: "attention", text: `${plural(Fe.weak, "group", "groups")} not detected` };
  else out.feiicard = { state: "ok", text: "all detected" };
  const M = w.simcheck;
  if (!M || !M.id) out.simcard = { state: "none", text: "not run" };
  else {
    const b = M.rows.filter((r) => r.flag === "biased").length, e = M.rows.filter((r) => r.flag.startsWith("error bar")).length;
    out.simcard = b ? { state: "attention", text: `${b} biased` } : e ? { state: "attention", text: `${plural(e, "error bar", "error bars")} off` } : { state: "ok", text: "recovered" };
  }
  const J = w.jointcheck;
  if (!J || (!J.cal.id && !J.sim.id)) out.jointcard = { state: "none", text: "not run" };
  else {
    const cb = J.cal.id ? J.cal.rows.filter((r) => r.flag !== "ok" && r.flag !== "not measured").length : 0;
    const sb = J.sim.id ? J.sim.rows.filter((r) => r.flag !== "ok" && r.flag !== "not measured").length : 0;
    out.jointcard = cb ? { state: "attention", text: "calibration noise" } : sb ? { state: "attention", text: `${sb} off in simulations` }
      : { state: "ok", text: J.cal.id && J.sim.id ? "consistent · recovered" : J.cal.id ? "consistent" : "recovered" };
  }
  return out;
}
// ---- Checks of one spectrum of a sample: fa.analyze_residuals and fa.suggest_lines of its fit (read only: each
// spectrum has its own model, so the changes are given as code that refits that spectrum)
const SC = { k: null, sig: null, D: null, use: new Set() };
async function loadSampleChecks(k) {
  const sig = `${S.sample && S.sample.id}|${k}`;
  if (SC.sig === sig) return;                     // already shown: a poll must not rebuild it (a selection in the code)
  Object.assign(SC, { sig, k, D: null, use: new Set(), code: "" });
  renderSampleChecks();
  let d;
  try { d = await api(`/api/wf/sample_checks?k=${k}`); } catch (e) { d = { error: String(e.message || e) }; }
  if (SC.sig !== sig) return;
  SC.D = d;
  (d.lines || []).forEach((r, i) => { if ((r.dbic ?? 0) < -10) SC.use.add(`l${i}`); });
  renderSampleChecks();
  renderChecks();
}
async function scCode() {                         // the refit code of the chosen changes (combined on the server)
  const ids = (p) => [...SC.use].filter((id) => id[0] === p).map((id) => id.slice(1)).join(",");
  const sig = `${SC.sig}|${[...SC.use].sort()}`;
  if (SC.codeSig === sig) return;
  SC.codeSig = sig;
  let code;
  try { code = (await api(`/api/wf/sample_checks_code?k=${SC.k}&lines=${ids("l")}&resid=${ids("r")}`)).code; } catch (e) { code = `# ${e.message || e}`; }
  if (SC.codeSig === sig && $("sccode")) $("sccode").textContent = SC.code = code;
}
function renderSampleChecks() {
  const box = $("scheckcard");
  if (!box) return;
  const D = SC.D, R = resultData;
  if (R && R.error) { $("scbody").innerHTML = '<p class="muted">This spectrum was not fitted, so there is nothing to check.</p>'; $("schead").textContent = R.name || ""; return; }
  if (!D) { $("scbody").innerHTML = '<div class="empty">Looking at the residuals of this fit…</div>'; return; }
  if (D.error) { $("scbody").innerHTML = `<p class="muted">${esc(D.error)}</p>`; return; }
  $("schead").textContent = `${D.name} · χ²ᵣ ${D.rchi2 != null ? D.rchi2.toFixed(2) : "—"}`;
  const maxB = Math.max(30, ...D.lines.map((r) => Math.abs(r.dbic || 0)));
  const btn = (id) => `<button class="btn small${SC.use.has(id) ? " primary" : ""}" data-scuse="${id}" aria-pressed="${SC.use.has(id)}">${SC.use.has(id) ? "In the code ✓" : "Use"}</button>`;
  const lines = D.lines.map((r, i) => {
    const split = r.action === "split", db = Math.abs(r.dbic || 0);
    const w = Math.log10(1 + db) / Math.log10(1 + maxB) * 100, tick = Math.log10(11) / Math.log10(1 + maxB) * 100;
    const change = split ? `own velocity <b>${r.dv > 0 ? "+" : "−"}${Math.abs(r.dv).toFixed(0)} km/s</b> from <span class="mono">${esc(r.component)}</span>`
      : `width and shift of <span class="mono">${esc(r.component)}</span> <small class="muted">${esc(r.role)}</small>`;
    return `<div class="sgcard${SC.use.has(`l${i}`) ? " picked" : ""}"><div class="sghead"><b>${esc(lsName(r))}</b><small class="muted">${esc(r.ion)} · ${esc(LS_TYPE[r.type] || r.type)}</small><span class="grow"></span>` +
      `<span class="sgact ${split ? "split" : "add"}">${split ? "separate" : "add"}</span></div>` +
      `<div class="sgplot">${r.snip ? snipSvg(r.snip, { split }) : '<span class="muted small">no data here</span>'}</div>` +
      `<div class="sgbic" title="how much the BIC drops with this change; more than 10 (the tick) is strong evidence"><span>ΔBIC</span><em><i style="width:${w.toFixed(1)}%"></i><b style="left:${tick.toFixed(1)}%"></b></em><strong>−${db.toFixed(0)}</strong></div>` +
      `<div class="sgmeta"><span>${r.wave.toFixed(1)} Å</span>${r.snr === null ? "" : `<span>S/N <b>${r.snr.toFixed(1)}</b></span>`}</div>` +
      `<p class="sgtext">${split ? "Give it " : "Add it with the "}${change}</p><div class="sgfoot">${btn(`l${i}`)}</div></div>`;
  }).join("");
  const resid = D.resid.map((r, i) => `<div class="sgcard${SC.use.has(`r${i}`) ? " picked" : ""}"><div class="sghead"><span class="sgno">${i + 1}</span>` +
      `<b>${r.line ? esc(prettyLine(r.line)) : `${fmtNum(r.center, 5)} Å`}</b>${r.in_model ? '<small class="muted">in model</small>' : ""}<span class="grow"></span><span class="badge k-${r.kind}">${esc(r.kind)}</span></div>` +
      `<div class="sgplot">${r.snip ? snipSvg(r.snip, { band: [r.lo, r.hi], tint: RES_TINT[r.kind] }) : '<span class="muted small">no data here</span>'}</div>` +
      `<div class="sgmeta"><span><b>${r.sigma.toFixed(1)}σ</b> significance</span><span>${fmtNum(r.center, 5)} Å</span><span>FWHM ${Math.round(r.fwhm)} km/s</span></div>` +
      `<p class="sgtext">${esc(r.suggestion)}</p>${r.code && r.code.startsWith("model.") ? `<div class="sgfoot">${btn(`r${i}`)}</div>` : r.code ? `<code class="sgcode">${esc(r.code)}</code>` : ""}</div>`).join("");
  const nStrong = D.lines.filter((r) => (r.dbic ?? 0) < -10).length;
  $("scbody").innerHTML =
    `<h3 class="sc-h">Lines to add or separate <small class="muted">fa.suggest_lines(r)</small></h3>` +
    (D.lines.length ? `<p class="muted small">${D.lines.length} change${D.lines.length > 1 ? "s" : ""} the BIC prefers${nStrong ? `; the ${nStrong} with ΔBIC below −10 are in the code below` : ""}.</p><div class="sggrid">${lines}</div>` : '<p class="muted small">No line is missing and none needs a velocity of its own.</p>') +
    `<h3 class="sc-h">Does the model miss anything? <small class="muted">fa.analyze_residuals(r)</small></h3>` +
    (D.resid.length ? `<p class="muted small">${D.resid.length} feature${D.resid.length > 1 ? "s" : ""} above 4σ in the residuals.</p><div class="sggrid">${resid}</div>` : `<p class="muted small">${D.resid_error ? esc(D.resid_error) : "No significant residual features: the model describes the data down to the noise."}</p>`) +
    `<h3 class="sc-h">Refit this spectrum with the changes</h3><p class="muted small">Each spectrum of a sample has its own model, so the changes are not applied here: run this code (in the session's notebook or a script) and keep the change when the BIC drops by more than 10.</p>` +
    `<pre class="code" id="sccode">${esc(SC.code || "")}</pre><button class="btn small" type="button" id="sccopy">Copy</button>`;
  SC.codeSig = null; scCode();
  $("scbody").querySelectorAll("[data-scuse]").forEach((b) => (b.onclick = () => {
    const id = b.dataset.scuse; SC.use.has(id) ? SC.use.delete(id) : SC.use.add(id); renderSampleChecks();
  }));
  $("sccopy").onclick = async () => { try { await navigator.clipboard.writeText($("sccode").textContent); toast("Code copied"); } catch (e) { toast("Select the code and copy it"); } };
}
function renderChecks() {
  const st = checkStatus(), cell = S.cells[S.cells.length - 1];
  let n = 0, no = 0;
  CHECKS.forEach(([id, tag]) => {
    const card = $(id), s = st[id];
    if (!card.classList.contains("hidden")) { const sn = card.querySelector(".stepno"); if (sn) sn.textContent = ++no; }   // 1, 2, … over the cards shown
    const running = !!(S.busy && cell && [].concat(tag).some((t) => cell.code.startsWith(t)) && !card.classList.contains("hidden"));
    if (running && !ckRunning[id]) ckOpen[id] = true;     // a run opens its card and leaves it open
    ckRunning[id] = running;
    const open = id in ckOpen ? ckOpen[id] : s.state === "attention";
    card.classList.toggle("collapsed", !open);
    card.querySelector(".cktoggle").setAttribute("aria-expanded", String(open));
    const chip = $(id + "-status");
    chip.className = "badge ckstatus" + (running ? "" : s.state === "ok" ? " ok" : s.state === "attention" ? " warn" : " muted");
    chip.textContent = running ? "running" : s.text;
    if (s.state === "attention" && !card.classList.contains("hidden")) n++;
  });
  if (fitDone() && fitDone().kind === "sample")       // a sample: the checks of the shown spectrum, once looked at
    n = SC.D && !SC.D.error && SC.k === sk ? SC.D.lines.filter((r) => (r.dbic ?? 0) < -10).length + SC.D.resid.length : 0;
  $("checkcount").textContent = n || "";
  $("checkcount").classList.toggle("hidden", !n);
  $("checkcount").title = n ? `${plural(n, "check needs", "checks need")} attention` : "";
}
function initChecks() {
  CHECKS.forEach(([id]) => ($(id).querySelector(".cktoggle").onclick = () => { ckOpen[id] = $(id).classList.contains("collapsed"); renderChecks(); }));
}
// ------------------------------------------------------------- correlations over epochs
// The epoch table comes from the server once; the map, the fit and the time lag are all computed here, so every
// change is immediate. Matplotlib draws only what is saved.
const CX = { id: null, D: null, by: {}, x: null, y: null, lag: 0, off: new Set(), map: [], asked: false, loading: false, frame: 0, fit: null };
const cxStats = {
  // nonlinear least squares by simplex (Nelder-Mead), in the same form as fantasy_agn.plotting.fit_line
  simplex(f, x0, step) {
    const n = x0.length;
    let s = [x0.slice()];
    for (let i = 0; i < n; i++) { const p = x0.slice(); p[i] += step[i]; s.push(p); }
    let v = s.map(f);
    for (let it = 0; it < 500; it++) {
      const o = v.map((_, i) => i).sort((a, b) => v[a] - v[b]);
      s = o.map((i) => s[i]); v = o.map((i) => v[i]);
      if (Math.abs(v[n] - v[0]) < 1e-8) break;
      const c = new Array(n).fill(0);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) c[j] += s[i][j] / n;
      const at = (t) => c.map((x, j) => x + t * (s[n][j] - x));
      const r = at(-1), fr = f(r);
      if (fr < v[0]) { const e = at(-2), fe = f(e); if (fe < fr) { s[n] = e; v[n] = fe; } else { s[n] = r; v[n] = fr; } }
      else if (fr < v[n - 1]) { s[n] = r; v[n] = fr; }
      else {
        const k = at(0.5), fk = f(k);
        if (fk < v[n]) { s[n] = k; v[n] = fk; }
        else for (let i = 1; i <= n; i++) { s[i] = s[i].map((x, j) => s[0][j] + 0.5 * (x - s[0][j])); v[i] = f(s[i]); }
      }
    }
    return s[0];
  },
  // y = a + b (x - x0) with errors in both coordinates and an intrinsic scatter; bootstrap over the points
  fitLine(x, y, sx, sy, nBoot = 200) {
    const n = x.length, sorted = [...x].sort((a, b) => a - b);
    const x0 = n % 2 ? sorted[n >> 1] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
    const mean = (a) => a.reduce((p, c) => p + c, 0) / a.length;
    const sd = (a) => { const m = mean(a); return Math.sqrt(a.reduce((p, c) => p + (c - m) ** 2, 0) / a.length); };
    // solved in units of the spread of the points, so shifts in km/s behave like log fluxes
    const kx = sd(x) || 1, ky = sd(y) || 1;
    const X = x.map((v) => (v - x0) / kx), Y = y.map((v) => v / ky), SX = sx.map((v) => v / kx), SY = sy.map((v) => v / ky);
    const solve = (xi, yi, sxi, syi, start) => cxStats.simplex(([a, b, ls]) => {
      let t = 0; const s2 = Math.exp(2 * ls);
      for (let i = 0; i < xi.length; i++) {
        const d = yi[i] - a - b * xi[i], w = syi[i] * syi[i] + b * b * sxi[i] * sxi[i] + s2;
        t += 0.5 * (d * d / w + Math.log(w));
      }
      return t;
    }, start, [0.05, 0.05, 0.3]);
    const mX = mean(X), mY = mean(Y);
    let sxx = 0, sxy = 0;
    X.forEach((v, i) => { sxx += (v - mX) ** 2; sxy += (v - mX) * (Y[i] - mY); });
    const b0 = sxx > 0 ? sxy / sxx : 0;
    const best = solve(X, Y, SX, SY, [mY - b0 * mX, b0, Math.log(0.5)]);
    let seed = 12345;
    const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
    const B = [];
    for (let k = 0; k < nBoot; k++) {
      const a = [], b = [], c = [], d = [];
      for (let i = 0; i < n; i++) { const j = Math.floor(rnd() * n); a.push(X[j]); b.push(Y[j]); c.push(SX[j]); d.push(SY[j]); }
      B.push(solve(a, b, c, d, best));
    }
    const back = (p) => [p[0] * ky, p[1] * ky / kx, Math.exp(p[2]) * ky];
    const [a, b, s] = back(best), boot = B.map(back);
    return { a, b, s, ea: sd(boot.map((p) => p[0])), eb: sd(boot.map((p) => p[1])), es: sd(boot.map((p) => p[2])), x0, boot };
  },
  rank(a) {
    const o = a.map((v, i) => [v, i]).sort((p, q) => p[0] - q[0]), r = new Array(a.length);
    for (let i = 0; i < o.length;) {
      let j = i;
      while (j + 1 < o.length && o[j + 1][0] === o[i][0]) j++;
      for (let k = i; k <= j; k++) r[o[k][1]] = (i + j) / 2 + 1;
      i = j + 1;
    }
    return r;
  },
  pearson(x, y) {
    const n = x.length, mx = x.reduce((a, b) => a + b, 0) / n, my = y.reduce((a, b) => a + b, 0) / n;
    let xx = 0, yy = 0, xy = 0;
    for (let i = 0; i < n; i++) { xx += (x[i] - mx) ** 2; yy += (y[i] - my) ** 2; xy += (x[i] - mx) * (y[i] - my); }
    return xx > 0 && yy > 0 ? xy / Math.sqrt(xx * yy) : NaN;
  },
  lnGamma(z) {
    const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let y = z, t = z + 5.5, s = 1.000000000190015;
    t -= (z + 0.5) * Math.log(t);
    for (const v of c) s += v / ++y;
    return -t + Math.log(2.5066282746310005 * s / z);
  },
  // the regularized incomplete beta function, by its continued fraction
  betainc(x, a, b) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const front = Math.exp(cxStats.lnGamma(a + b) - cxStats.lnGamma(a) - cxStats.lnGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    const cf = (x, a, b) => {
      const tiny = 1e-300, fix = (v) => (Math.abs(v) < tiny ? tiny : v);
      let c = 1, d = 1 / fix(1 - (a + b) * x / (a + 1)), h = d;
      for (let m = 1; m <= 500; m++) {
        const m2 = 2 * m;
        let k = m * (b - m) * x / ((a + m2 - 1) * (a + m2));
        d = 1 / fix(1 + k * d); c = fix(1 + k / c); h *= d * c;
        k = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1));
        d = 1 / fix(1 + k * d); c = fix(1 + k / c);
        const del = d * c; h *= del;
        if (Math.abs(del - 1) < 1e-14) break;
      }
      return h;
    };
    return x < (a + 1) / (a + b + 2) ? front * cf(x, a, b) / a : 1 - front * cf(1 - x, b, a) / b;
  },
  // Spearman's rho and its two-sided p (Student t with n - 2 degrees of freedom), as scipy.stats.spearmanr
  spearman(x, y) {
    const n = x.length;
    if (n < 3) return null;
    const r = cxStats.pearson(cxStats.rank(x), cxStats.rank(y));
    if (!isFinite(r)) return null;
    const df = n - 2, t2 = (r * r * df) / Math.max((1 + r) * (1 - r), 1e-300);
    return { rho: r, p: cxStats.betainc(df / (df + t2), df / 2, 0.5), n };
  },
};

// the values of one quantity with the time lag applied (the continuum, as fantasy_agn.correlations.shift_continuum)
function cxSeries(c) {
  const lagged = CX.lag > 0 && /^(cont|logL)/.test(c.name);
  if (!lagged) return { v: c.v, e: c.e };
  const t = CX.D.t, o = t.map((_, i) => i).sort((a, b) => t[a] - t[b]);
  const interp = (arr) => {
    const k = o.filter((i) => arr[i] !== null);
    return t.map((ti) => {
      const w = ti - CX.lag;
      if (w < t[o[0]] || w > t[o[o.length - 1]] || k.length < 2) return null;
      if (w <= t[k[0]]) return arr[k[0]];
      if (w >= t[k[k.length - 1]]) return arr[k[k.length - 1]];
      let lo = 0, hi = k.length - 1;
      while (hi - lo > 1) { const m = (lo + hi) >> 1; if (t[k[m]] <= w) lo = m; else hi = m; }
      const f = (w - t[k[lo]]) / (t[k[hi]] - t[k[lo]]);
      return arr[k[lo]] + f * (arr[k[hi]] - arr[k[lo]]);
    });
  };
  const hasErr = c.e.some((v) => v !== null);
  return { v: interp(c.v), e: hasErr ? interp(c.e) : c.e };
}
// the epochs where both are measured and not left out: axis values (log10 where the quantity is logarithmic), errors
function cxPair(nx, ny) {
  const a = CX.by[nx], b = CX.by[ny], A = cxSeries(a), B = cxSeries(b), out = [];
  const axis = (v, e, lg) => (lg ? [Math.log10(v), (e || 0) / (v * Math.LN10)] : [v, e || 0]);
  for (let k = 0; k < CX.D.t.length; k++) {
    if (CX.off.has(k)) continue;
    const u = A.v[k], w = B.v[k];
    if (u === null || w === null || (a.log && u <= 0) || (b.log && w <= 0)) continue;
    const [x, sx] = axis(u, A.e[k], a.log), [y, sy] = axis(w, B.e[k], b.log);
    out.push({ k, x, sx, y, sy, u, w });
  }
  return out;
}
function cxCorr(nx, ny) {
  const p = cxPair(nx, ny);
  const s = cxStats.spearman(p.map((d) => d.u), p.map((d) => d.w));
  if (!s) return { rho: null, p: null, n: p.length, trivial: false };
  const r = cxStats.pearson(p.map((d) => d.u), p.map((d) => d.w));
  const same = (/^(cont|logL)/.test(nx) && /^(cont|logL)/.test(ny)) || Math.abs(r) > 1 - 1e-9;
  return { ...s, trivial: same };
}
const cxP = (p) => (p === null ? "—" : p < 1e-10 ? "< 10⁻¹⁰" : fmtNum(p, 2));
const cxRho = (r) => (r === null ? "—" : (r > 0 ? "+" : "") + r.toFixed(2));
const cxUnit = (n) => (/_(fwhm|sigma|shift)$/.test(n) ? " [km/s]" : /_ew$/.test(n) ? " [Å]" : "");
const cxAxis = (n) => (CX.by[n].log ? "log " : "") + CX.by[n].label + cxUnit(n);
const cxQuery = () => `lag=${CX.lag}&off=${[...CX.off].sort((a, b) => a - b).join(",")}`;
const cxMix = (a, b, f) => "#" + [0, 2, 4].map((i) => {
  const u = parseInt(a.substr(i, 2), 16), v = parseInt(b.substr(i, 2), 16);
  return Math.round(u + (v - u) * f).toString(16).padStart(2, "0");
}).join("");
function cxTicks(a, b, want = 6) {
  const raw = (b - a) / want, mag = 10 ** Math.floor(Math.log10(raw)), r = raw / mag;
  const step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag, out = [];
  for (let v = Math.ceil(a / step) * step; v <= b + step * 1e-6; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return { out, digits: Math.max(0, -Math.floor(Math.log10(step) + 1e-9)) };
}
function cxDrawMap() {
  const cols = CX.map, n = cols.length, L = 104, T = 6, C = 50, B = 92;
  if (n < 2) { $("cxmap").innerHTML = ""; return; }
  const W = L + C * (n - 1) + 8, H = T + C * (n - 1) + B;
  let s = `<svg class="cxsvg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Spearman rank correlation of every pair of quantities">`, npairs = 0;
  for (let i = 1; i < n; i++) {
    const y = T + C * (i - 1);
    s += `<text class="cxlab" x="${L - 8}" y="${y + C / 2 + 4}" text-anchor="end">${esc(CX.by[cols[i]].label)}</text>`;
    for (let j = 0; j < i; j++) {
      const x = L + C * j, R = cxCorr(cols[j], cols[i]), on = new Set([CX.x, CX.y]).has(cols[i]) && new Set([CX.x, CX.y]).has(cols[j]) && CX.x !== CX.y;
      const usable = R.rho !== null && !R.trivial;
      if (usable) npairs++;
      const fill = !usable ? "#f0efec" : cxMix("f0efec", R.rho < 0 ? "185FA5" : "A32D2D", Math.abs(R.rho));
      const dark = usable && Math.abs(R.rho) > 0.55, txt = R.trivial ? "=" : usable && R.p < 0.05 ? R.rho.toFixed(2) : "";
      s += `<g class="cxcell" data-i="${i}" data-j="${j}"><rect x="${x + 1}" y="${y + 1}" width="${C - 2}" height="${C - 2}" rx="3" fill="${fill}"${on ? ' class="on"' : ""}></rect>` +
        (txt ? `<text x="${x + C / 2}" y="${y + C / 2 + 4}" text-anchor="middle" class="${dark ? "cxnum light" : "cxnum"}">${txt}</text>` : "") + "</g>";
    }
  }
  for (let j = 0; j < n - 1; j++) {
    const x = L + C * j + C / 2, y = T + C * (n - 1) + 10;
    s += `<text class="cxlab" transform="translate(${x + 3} ${y}) rotate(-45)" text-anchor="end">${esc(CX.by[cols[j]].label)}</text>`;
  }
  $("cxmap").innerHTML = s + "</svg>";
  $("cx-chance").textContent = npairs ? `Of ${npairs} pairs, about ${fmtNum(npairs * 0.05, 1)} reach p < 0.05 by chance.` : "";
}
function cxDrawPlot() {
  const nx = CX.x, ny = CX.y, pts = nx === ny ? [] : cxPair(nx, ny), N = CX.D.t.length;
  const set = (id, v, e) => { $(id).innerHTML = `${v}${e ? ` <small>${e}</small>` : ""}`; };
  $("cx-title").textContent = `${CX.by[ny].label} against ${CX.by[nx].label}`;
  set("cx-n", `${pts.length}<span class="muted"> of ${N}</span>`);
  if (pts.length < 3) {
    ["cx-slope", "cx-scatter", "cx-rho", "cx-p"].forEach((id) => set(id, "—"));
    $("cxplot").innerHTML = `<p class="muted cxnone">${nx === ny ? "Choose two different quantities." : "Too few epochs have both quantities" + (CX.lag ? " at this time lag." : ".")}</p>`;
    $("cx-eq").textContent = "";
    CX.fit = null;
    return;
  }
  const X = pts.map((d) => d.x), Y = pts.map((d) => d.y), F = cxStats.fitLine(X, Y, pts.map((d) => d.sx), pts.map((d) => d.sy));
  const R = cxStats.spearman(pts.map((d) => d.u), pts.map((d) => d.w));
  CX.fit = F;
  const sc = (v, e) => { const s = sciPM(v, e); return [s.v, s.e]; };
  set("cx-slope", ...sc(F.b, F.eb));
  if (F.s < 1e-3 * (Math.max(...Y) - Math.min(...Y))) set("cx-scatter", "about 0");
  else set("cx-scatter", ...sc(F.s, F.es));
  set("cx-rho", R ? cxRho(R.rho) : "—");
  set("cx-p", R ? cxP(R.p) : "—");
  const W = 640, H = 370, l = 66, r = 16, t = 12, b = 50, pw = W - l - r, ph = H - t - b;
  const pad = (a) => { const lo = Math.min(...a), hi = Math.max(...a), d = hi - lo || Math.abs(hi) || 1; return [lo - 0.08 * d, hi + 0.08 * d]; };
  const [x1, x2] = pad(X), [y1, y2] = pad(Y);
  const px = (v) => l + ((v - x1) / (x2 - x1)) * pw, py = (v) => t + ph - ((v - y1) / (y2 - y1)) * ph;
  const clipx = (v) => Math.min(l + pw, Math.max(l, px(v))), clipy = (v) => Math.min(t + ph, Math.max(t, py(v)));
  let s = `<svg class="cxsvg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(CX.by[ny].label)} against ${esc(CX.by[nx].label)} with the fitted line">`;
  const tx = cxTicks(x1, x2), ty = cxTicks(y1, y2, 5);
  tx.out.forEach((v) => { s += `<line class="cxgl" x1="${px(v)}" x2="${px(v)}" y1="${t}" y2="${t + ph}"></line><text class="cxtick" x="${px(v)}" y="${t + ph + 18}" text-anchor="middle">${v.toFixed(tx.digits)}</text>`; });
  ty.out.forEach((v) => { s += `<line class="cxgl" x1="${l}" x2="${l + pw}" y1="${py(v)}" y2="${py(v)}"></line><text class="cxtick" x="${l - 8}" y="${py(v) + 4}" text-anchor="end">${v.toFixed(ty.digits)}</text>`; });
  s += `<rect class="cxframe" x="${l}" y="${t}" width="${pw}" height="${ph}"></rect>`;
  s += `<text class="cxaxis" x="${l + pw / 2}" y="${H - 8}" text-anchor="middle">${esc(cxAxis(nx))}</text>`;
  s += `<text class="cxaxis" transform="translate(16 ${t + ph / 2}) rotate(-90)" text-anchor="middle">${esc(cxAxis(ny))}</text>`;
  s += `<clipPath id="cxclip"><rect x="${l}" y="${t}" width="${pw}" height="${ph}"></rect></clipPath><g clip-path="url(#cxclip)">`;
  // the 68% range of the line over the bootstrap samples
  const xs = Array.from({ length: 41 }, (_, i) => x1 + ((x2 - x1) * i) / 40), lo = [], hi = [];
  xs.forEach((v) => {
    const q = F.boot.map((p) => p[0] + p[1] * (v - F.x0)).sort((a, c) => a - c);
    lo.push(q[Math.floor(0.16 * q.length)]); hi.push(q[Math.floor(0.84 * q.length)]);
  });
  s += `<polygon class="cxband" points="${xs.map((v, i) => `${px(v)},${py(hi[i])}`).concat(xs.map((v, i) => `${px(v)},${py(lo[i])}`).reverse()).join(" ")}"></polygon>`;
  s += "</g>";
  pts.forEach((d) => {
    s += `<line class="cxerr" x1="${clipx(d.x - d.sx)}" x2="${clipx(d.x + d.sx)}" y1="${py(d.y)}" y2="${py(d.y)}"></line>` +
      `<line class="cxerr" x1="${px(d.x)}" x2="${px(d.x)}" y1="${clipy(d.y - d.sy)}" y2="${clipy(d.y + d.sy)}"></line>`;
  });
  s += `<g clip-path="url(#cxclip)"><line class="cxline" x1="${px(x1)}" x2="${px(x2)}" y1="${py(F.a + F.b * (x1 - F.x0))}" y2="${py(F.a + F.b * (x2 - F.x0))}"></line></g>`;
  const T = CX.D.t.filter((v) => v !== null), t0 = Math.min(...T), dt = Math.max(...T) - t0 || 1;
  pts.forEach((d) => {
    s += `<g class="cxpt" data-k="${d.k}"><circle class="hit" cx="${px(d.x)}" cy="${py(d.y)}" r="10"></circle>` +
      `<circle class="dot" cx="${px(d.x)}" cy="${py(d.y)}" r="5" fill="${cxMix("85B7EB", "0C447C", (CX.D.t[d.k] - t0) / dt)}"></circle></g>`;
  });
  $("cxplot").innerHTML = s + "</svg>";
  const lg = (n) => (CX.by[n].log ? "log " : "") + CX.by[n].label;
  const sign = F.b < 0 ? "−" : "+";
  $("cx-eq").textContent = `${lg(ny)} = ${fmtNum(F.a, 4)} ${sign} ${fmtNum(Math.abs(F.b), 3)} (${lg(nx)} − ${fmtNum(F.x0, 4)})`;
}
function cxDraw() {
  cancelAnimationFrame(CX.frame);
  CX.frame = requestAnimationFrame(() => {
    if (!CX.D) return;
    cxDrawMap(); cxDrawPlot();
    const out = CX.off.size;
    $("cx-restore").classList.toggle("hidden", !out);
    $("cx-restore").textContent = `Restore ${out} left-out epoch${out === 1 ? "" : "s"}`;
    $("cx-x").value = CX.x; $("cx-y").value = CX.y;
    $("cx-lagv").value = CX.lag;
    $("cx-lag").value = CX.lag;
    $("cx-mapsum").textContent = `${CX.map.length} of ${CX.D.columns.length}`;
    refreshCode("cx-codebox", "cx-code", `/api/wf/relation_code?x=${encodeURIComponent(CX.x)}&y=${encodeURIComponent(CX.y)}&${cxQuery()}`);
    refreshCode("cx-mapcodebox", "cx-mapcode", `/api/wf/correlations_code?cols=${encodeURIComponent(CX.map.join(","))}&${cxQuery()}`);
  });
}
function cxPick(D) {
  // the first view: the Baldwin effect of the main broad line (EW against luminosity)
  const names = D.columns.map((c) => c.name), lum = names.find((n) => n.startsWith("logL")) || names.find((n) => n.startsWith("cont")) || names[0];
  const ew = D.map.find((n) => n.endsWith("_ew")) || D.map.find((n) => n !== lum) || names[1];
  if (!names.includes(CX.x)) CX.x = lum;
  if (!names.includes(CX.y)) CX.y = ew;
}
async function cxLoad(id) {
  CX.loading = true;
  try {
    const D = await api("/api/wf/correlations_data");
    if (!D.columns) return;
    CX.D = D; CX.id = id; CX.by = Object.fromEntries(D.columns.map((c) => [c.name, c]));
    CX.off = new Set(); CX.lag = 0; CX.map = D.map.slice();
    cxPick(D);
    const opts = D.columns.map((c) => `<option value="${esc(c.name)}">${esc(c.label)}</option>`).join("");
    $("cx-x").innerHTML = opts; $("cx-y").innerHTML = opts;
    const span = Math.max(...D.t) - Math.min(...D.t);
    $("cx-lag").max = Math.max(10, Math.ceil(span / 2));
    $("cx-pick").innerHTML = D.columns.map((c) => `<label class="check cxchip"><input type="checkbox" value="${esc(c.name)}" ${CX.map.includes(c.name) ? "checked" : ""}><span>${esc(c.label)}</span></label>`).join("");
    $("cx-pick").querySelectorAll("input").forEach((el) => (el.onchange = () => {
      if (!el.checked && CX.map.length <= 2) { el.checked = true; toast("The map needs at least two quantities"); return; }
      const order = D.columns.map((c) => c.name);
      CX.map = el.checked ? CX.map.concat(el.value) : CX.map.filter((c) => c !== el.value);
      CX.map.sort((a, b) => order.indexOf(a) - order.indexOf(b));
      cxDraw();
    }));
    $("cxstart").classList.add("hidden");
    $("cxmain").classList.remove("hidden");
    cxDraw();
  } catch (e) { toast(e.message); } finally { CX.loading = false; }
}
function renderCorr() {
  const C = S.wf.corr;
  if (!C || rv !== "corr") return;
  if (C.id) {
    CX.asked = false;
    if (CX.id !== C.id && !CX.loading) cxLoad(C.id);
    return;
  }
  CX.id = null; CX.D = null;
  $("cxmain").classList.add("hidden");
  $("cxstart").classList.remove("hidden");
  // measuring starts by itself when the tab is opened; if that failed, the button tries again
  if (!CX.asked && !S.busy) cxMeasure();
  const cell = S.cells[S.cells.length - 1], p = S.progress;
  const running = CX.posting || (CX.asked && S.busy && (!cell || cell.code.startsWith("# correlations")));
  const known = running && p && p.task === "measuring the epochs" && p.total;
  $("cxmsg").textContent = running ? (known ? `Measuring the epochs: ${p.done || 0} of ${p.total}` : `Measuring ${C.n_epochs} epochs…`) : S.busy ? "Waiting for the running step to finish." : "Could not measure the epochs.";
  $("cxbar").style.width = known ? `${Math.round((100 * (p.done || 0)) / p.total)}%` : "0%";
  $("cxbarwrap").classList.toggle("hidden", !running);
  $("cxretry").classList.toggle("hidden", running || !!S.busy);
}
async function cxMeasure() {
  CX.asked = CX.posting = true;
  await post("/api/wf/correlations", {});
  CX.posting = false;
  renderCorr();
}
function cxTip(wrap, ev, html) {
  const tip = wrap.querySelector(".cxtip");
  if (!html) { tip.classList.add("hidden"); return; }
  tip.innerHTML = html;
  tip.classList.remove("hidden");
  const b = wrap.getBoundingClientRect(), x = ev.clientX - b.left, y = ev.clientY - b.top;
  tip.style.left = Math.min(x + 12, b.width - tip.offsetWidth - 4) + "px";
  tip.style.top = (y > tip.offsetHeight + 16 ? y - tip.offsetHeight - 10 : y + 14) + "px";
}
function initCorr() {
  const lag = (v) => { CX.lag = Math.max(0, +v || 0); cxDraw(); };
  $("cx-lag").oninput = (e) => lag(e.target.value);
  $("cx-lagv").onchange = (e) => lag(e.target.value);
  $("cx-x").onchange = (e) => { CX.x = e.target.value; cxDraw(); };
  $("cx-y").onchange = (e) => { CX.y = e.target.value; cxDraw(); };
  $("cx-swap").onclick = () => { [CX.x, CX.y] = [CX.y, CX.x]; cxDraw(); };
  $("cx-restore").onclick = () => { CX.off.clear(); cxDraw(); };
  $("cxretry").onclick = cxMeasure;
  const mapw = $("cxmap").parentElement, plotw = $("cxplot").parentElement;
  $("cxmap").onclick = (e) => {
    const g = e.target.closest(".cxcell");
    if (!g) return;
    CX.x = CX.map[+g.dataset.j]; CX.y = CX.map[+g.dataset.i]; cxDraw();
  };
  $("cxmap").onmousemove = (e) => {
    const g = e.target.closest(".cxcell");
    if (!g) return cxTip(mapw, e, null);
    const a = CX.map[+g.dataset.j], b = CX.map[+g.dataset.i], R = cxCorr(a, b);
    cxTip(mapw, e, `<b>${esc(CX.by[b].label)}</b> against <b>${esc(CX.by[a].label)}</b><br>` +
      (R.trivial ? "The same number twice, not a result" : `ρ = ${cxRho(R.rho)}, p ${R.p !== null && R.p < 1e-10 ? "" : "= "}${cxP(R.p)}, ${R.n} epochs`));
  };
  $("cxmap").onmouseleave = (e) => cxTip(mapw, e, null);
  $("cxplot").onclick = (e) => {
    const g = e.target.closest(".cxpt");
    if (!g) return;
    CX.off.add(+g.dataset.k); cxTip(plotw, e, null); cxDraw();
  };
  $("cxplot").onmousemove = (e) => {
    const g = e.target.closest(".cxpt");
    if (!g) return cxTip(plotw, e, null);
    const k = +g.dataset.k, val = (n) => { const c = CX.by[n], v = cxSeries(c).v[k]; return `${esc(c.label)}: ${fmtNum(v, 4)}`; };
    cxTip(plotw, e, `<b>${esc(CX.D.dates[k] || CX.D.names[k])}</b> · MJD ${fmtNum(CX.D.t[k], 6)}<br>${val(CX.x)}<br>${val(CX.y)}<br><span class="muted">Click to leave this epoch out</span>`);
  };
  $("cxplot").onmouseleave = (e) => cxTip(plotw, e, null);
  document.querySelectorAll("[data-cxsave]").forEach((b) => (b.onclick = () => {
    const [what, fmt] = b.dataset.cxsave.split("."), dpi = fmt === "png" ? "&dpi=300" : "";
    const q = what === "relation" ? `x=${encodeURIComponent(CX.x)}&y=${encodeURIComponent(CX.y)}` : `cols=${encodeURIComponent(CX.map.join(","))}`;
    download(`/api/wf/${what}.${fmt}?${q}&${cxQuery()}&dl=1${dpi}`);
  }));
  ["cx-codebox", "cx-mapcodebox"].forEach((id) => $(id).addEventListener("toggle", cxDraw));
}
function initCompare() {
  $("pinbtn").onclick = async () => { if (await post("/api/wf/pin", { label: $("pinlabel").value })) $("pinlabel").value = ""; };
}
function initResid() {
  $("residrun").onclick = () => post("/api/wf/residuals", { scope: $("rscope").value, nsigma: +$("rnsig").value });
  $("residrefit").onclick = () => go("fit");
  $("lsall").onclick = () => {
    const L = S.wf.linescan;
    if (!L) return;
    const left = L.rows.filter((r) => !r.added);
    addLines((L.n_epochs ? left.filter((r) => lsPick.set.has(r.line)) : left).map((r) => r.line));
  };
  $("lspick").onclick = () => {
    const L = S.wf.linescan;
    if (!L) return;
    const left = L.rows.filter((r) => !r.added).map((r) => r.line);
    if (left.every((ln) => lsPick.set.has(ln))) lsPick.set.clear(); else left.forEach((ln) => lsPick.set.add(ln));
    renderLines();
  };
  $("lsrefit").onclick = () => go("fit");   // only when the lines were added without a refit (from code)
  initRobust();
  initFeii();
  initSim();
  initChecks();
  initCompare();
  initCorr();
}

function renderResultTable() {
  const R = resultData, t = $("rtbl");
  if (!R || R.error) return;
  document.querySelectorAll("#rtabs button").forEach((b) => b.classList.toggle("on", b.dataset.rt === rtab));
  $("hideund").closest("label").classList.toggle("hidden", rtab !== "lines");
  if (rtab === "lines") t.innerHTML = lineTable(R.lines, $("hideund").checked, R.flux_unit);
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
      `<thead><tr>${A.columns.map((c) => `<th class="nowrap ${["name", "Spectrum", "Date", "date"].includes(c) ? "" : "num"}">${esc(c)}</th>`).join("")}</tr></thead><tbody>` +
      A.rows.map((r) => `<tr>${r.map((v, i) => `<td class="nowrap ${["name", "Spectrum"].includes(A.columns[i]) ? "mono" : ["Date", "date"].includes(A.columns[i]) ? "" : "num"}">${esc(v)}</td>`).join("")}</tr>`).join("") + "</tbody>";
  } else {
    t.innerHTML = '<thead><tr><th>Parameter</th><th class="num">Value</th><th class="num">Min</th><th class="num">Max</th><th>Unit</th><th>Status</th></tr></thead><tbody>' +
      (R.params || []).map((p) => `<tr><td class="mono">${esc(p.name)}</td><td class="num">${esc(p.value)}</td><td class="num">${fmtNum(p.min)}</td><td class="num">${fmtNum(p.max)}</td><td class="muted">${esc(p.unit)}</td><td>${statusBadge(p.status)}</td></tr>`).join("") + "</tbody>";
  }
}
function lineTable(lines, hideUnd, fluxUnit) {
  // grouped by component; a width and shift shared by the lines of a component are given once, in its header
  const rows = lines.filter((r) => !hideUnd || r.detected);
  const groups = [];
  rows.forEach((r) => { let g = groups.find((x) => x.name === r.component); if (!g) groups.push((g = { name: r.component, role: r.role, rows: [] })); g.rows.push(r); });
  const same = (g, k) => g.rows.length > 1 && g.rows.every((r) => r[k] !== null && Math.abs(r[k] - g.rows[0][k]) <= 1e-6 * Math.max(1, Math.abs(g.rows[0][k])));
  const fu = lineFluxUnit(fluxUnit);
  return `<thead><tr><th>Line</th><th class="num">Flux${fu ? ` <span class="muted small">(${esc(fu)})</span>` : ""}</th>` + '<th class="num">FWHM (km/s)</th><th class="num">Shift (km/s)</th><th class="num">EW (Å)</th><th>Flag</th></tr></thead><tbody>' +
    groups.map((g) => {
      const sw = same(g, "fwhm"), ss = same(g, "shift"), r0 = g.rows[0];
      const head = `<tr class="grouphead"><td colspan="2"><b class="mono">${esc(g.name)}</b>${g.role ? ` <span class="muted small">${esc(g.role)}</span>` : ""}</td>` +
        `<td class="num">${sw ? pm(r0.fwhm, r0.fwhm_err) : ""}</td><td class="num">${ss ? pm(r0.shift, r0.shift_err) : ""}</td><td></td><td></td></tr>`;
      return head + g.rows.map((r) => {
        const hasEstimate = r.detected || r.flags.some((f) => f.text === "weak (<2σ)");
        return `<tr class="${r.detected ? "" : "dim"}"><td class="lname">${esc(r.label)}</td>` +
        `<td class="num">${r.limit && (!r.detected || r.flux < 2 * (r.flux_err || 0)) ? `<span title="upper limit: ${esc(r.limit.how)}">&lt; ${fmtNum(r.limit.value, 3)}</span>${hasEstimate ? ` <span class="muted small">(${pm(r.flux, r.flux_err)})</span>` : ""}` : hasEstimate ? pm(r.flux, r.flux_err) : "—"}</td><td class="num">${sw ? "" : pm(r.fwhm, r.fwhm_err)}</td><td class="num">${ss ? "" : pm(r.shift, r.shift_err)}</td>` +
        `<td class="num">${hasEstimate ? pm(r.ew, r.ew_err) : "—"}</td><td>${r.flags.map((f) => `<span class="badge ${f.kind}">${esc(f.text)}</span>`).join(" ")}</td></tr>`;
      }).join("");
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
const DIAG_NAME = { nii: "[N II] BPT", sii: "[S II] BPT", oi: "[O I] BPT", whan: "WHAN" };
const REG_TONE = { "star-forming": "sf", SF: "sf", composite: "comp", AGN: "agn", Seyfert: "agn", sAGN: "agn", LINER: "liner", wAGN: "liner", retired: "ret", passive: "ret" };
// several epochs (or spectra): the class over all of them in words, a timeline coloured by class, how each diagram
// places them, and the broad-line S/N against the detection limit of fa.classify_fit
function renderClassSummary(d) {
  const L = d.labels, n = L.length, el = $("csum");
  const name = (c) => CLASS_NAME[c] || c, fin = d.stage === "final";
  const count = {};
  L.forEach((x) => (count[x.label] = (count[x.label] || 0) + 1));
  const order = Object.keys(count).sort((a, b) => count[b] - count[a]);
  const top = order[0], ks = (c) => L.filter((x) => x.label === c).map((x) => "#" + x.k).join(", ");
  const t1 = L.filter((x) => x.label.startsWith("type1")), unc = L.filter((x) => x.uncertain).length;
  const head = order.length === 1 ? `${esc(name(top))} in ${n === 2 ? "both" : "all " + n} epochs` : `${esc(name(top))} in ${count[top]} of ${n} epochs`;
  let sub = order.length === 1 ? "The class does not change between the epochs." :
    "The class changes: " + order.slice(1).map((c) => `<b>${esc(name(c))}</b> in ${ks(c)}`).join("; ") + ".";
  if (unc) sub += ` ${unc === n ? "Every class is" : plural(unc, "class is", "classes are")} uncertain.`;
  let call = "";
  if (t1.length && t1.length < n)
    call = `<div class="callout warn small"><div><b>Broad lines in some epochs only</b> (${t1.map((x) => "#" + x.k).join(", ")} of ${n}). A changing-look candidate, ` +
      `or a broad line near the detection limit: see its S/N below, against the limit of ${CS_SNR} that <code>fa.classify_fit</code> asks.</div></div>`;
  else if (order.length > 1 && order.every((c) => c.startsWith("type1")))
    call = `<p class="note small">Only the host share changes between “Type 1” and “with a strong host”: the host fraction at 5100 Å sits near the limit.</p>`;
  // timeline: by MJD when known, else in order
  const W = Math.max(300, Math.round(el.clientWidth - 36)), H = 54, m = { l: 14, r: 14 };   // drawn at the card's width
  const t = L.map((x, i) => (x.mjd ?? i)), t0 = Math.min(...t), t1v = Math.max(...t), span = t1v - t0 || 1;
  const X = (v) => m.l + (v - t0) / span * (W - m.l - m.r);
  const line = `<line x1="${m.l}" x2="${W - m.r}" y1="22" y2="22" class="cs-axis"/>`;
  const dots = L.map((x, i) => `<g class="cs-ep${x.k === d.k ? " sel" : ""}" data-k="${x.k}"><title>#${x.k} ${esc(x.date || "")} ${esc(x.name)}: ${esc(name(x.label))}${x.uncertain ? " (uncertain)" : ""}</title>` +
    `<circle cx="${X(t[i]).toFixed(1)}" cy="22" r="${x.k === d.k ? 8 : 6.5}" fill="${CLASS_COLOR[x.label] || "#9aa1ad"}"${x.uncertain ? ' stroke-dasharray="2 2"' : ""}/></g>`).join("");
  const lab = (v, s, a) => `<text x="${X(v).toFixed(1)}" y="${H - 6}" text-anchor="${a}">${esc(s || "")}</text>`;
  const i0 = t.indexOf(t0), i1 = t.indexOf(t1v);
  const tl = `<svg class="cs-tl" viewBox="0 0 ${W} ${H}">${line}${dots}${lab(t0, L[i0].date, "start")}${n > 1 ? lab(t1v, L[i1].date, "end") : ""}</svg>`;
  const legendC = order.map((c) => `<span class="cs-key"><i style="background:${CLASS_COLOR[c] || "#9aa1ad"}"></i>${esc(name(c))} <b>${count[c]}</b></span>`).join("");
  // each diagram: where the epochs fall
  const chips = Object.entries(DIAG_NAME).map(([key, nm]) => {
    const tally = {};
    L.forEach((x) => { const r = x.regions && x.regions[key]; if (r) tally[r] = (tally[r] || 0) + 1; });
    const rs = Object.keys(tally).sort((a, b) => tally[b] - tally[a]), got = rs.reduce((a, r) => a + tally[r], 0);
    const txt = rs.length ? rs.map((r) => `${esc(r)} <span class="muted">${tally[r]}</span>`).join(" · ") + (got < n ? ` <span class="muted">· ${n - got} not measured</span>` : "") : "not measured";
    return `<button type="button" class="cvchip ${rs.length ? "t-" + (REG_TONE[rs[0]] || "x") : "none"}${rs.length > 1 ? " mixed" : ""}" data-d="${key}"><small>${nm}</small><b>${txt}</b></button>`;
  }).join("");
  // broad-line S/N over time
  const sn = L.map((x, i) => ({ x, v: x.m && x.m.broad_snr, t: t[i] })).filter((p) => p.v !== null && p.v !== undefined);
  let ser = "";
  if (sn.length) {
    const SH = 92, mt = 8, mb = 18, ml = 34, top = Math.max(CS_SNR * 2, ...sn.map((p) => p.v)) * 1.1;
    const Xs = (v) => ml + (v - t0) / span * (W - ml - m.r), Y = (v) => mt + (1 - Math.max(0, v) / top) * (SH - mt - mb);
    const ticks = niceTicks(0, top, 3).map((v) => `<text x="${ml - 5}" y="${Y(v) + 3}" text-anchor="end">${v}</text><line x1="${ml}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" class="cs-grid"/>`).join("");
    ser = `<div class="cs-serhead"><b>Broad ${esc(prettyLine(sn[0].x.broad_line || "Ha"))} S/N</b><span class="muted small">the dashed line: S/N ${CS_SNR}, the limit for a broad line</span></div>` +
      `<svg class="cs-ser" viewBox="0 0 ${W} ${SH}">${ticks}<line x1="${ml}" x2="${W - m.r}" y1="${Y(CS_SNR)}" y2="${Y(CS_SNR)}" class="cs-lim"/>` +
      sn.map((p) => `<circle cx="${Xs(p.t).toFixed(1)}" cy="${Y(p.v).toFixed(1)}" r="${p.x.k === d.k ? 5.5 : 4}" fill="${CLASS_COLOR[p.x.label] || "#9aa1ad"}" class="cs-ep${p.x.k === d.k ? " sel" : ""}" data-k="${p.x.k}"><title>#${p.x.k} ${esc(p.x.date || "")}: S/N ${p.v.toFixed(1)}${p.x.m.broad_fwhm ? `, FWHM ${Math.round(p.x.m.broad_fwhm)} km/s` : ""}</title></circle>`).join("") + `</svg>`;
  }
  el.innerHTML = `<div class="cs-top"><div><small>${n} epochs${L[i0].date ? ` · ${esc(L[i0].date)} – ${esc(L[i1].date)}` : ""} · ${fin ? "final classes, from the fit" : "preliminary classes"}</small>` +
    `<h2>${head}</h2><p class="cs-sub">${sub}</p></div><div class="cs-keys">${legendC}</div></div>${call}` +
    `<div class="cs-tlwrap">${tl}</div><div class="cv-chips cs-chips">${chips}</div>${ser}`;
  el.querySelectorAll(".cs-ep").forEach((g) => (g.onclick = () => { ck = +g.dataset.k; loadClassify(true); }));
  el.querySelectorAll(".cvchip").forEach((b) => (b.onclick = () => {
    const c = $("cd-" + b.dataset.d).closest(".plotcard");
    c.scrollIntoView({ behavior: "smooth", block: "center" });
    c.classList.remove("flash"); void c.offsetWidth; c.classList.add("flash");
  }));
}
const CS_SNR = 5;       // the broad-line S/N fa.classify_fit asks (broad_snr)
async function loadClassify(force) {
  let d = null;
  if (S.wf.classified || S.wf.final_class) {
    if (!classData) { $("cmulti").classList.add("hidden"); $("csum").classList.add("hidden"); $("cverdict").innerHTML = '<div class="cv-hero cv-wait"><div class="cv-main"><i></i><i></i><i></i></div><div class="cv-chips"><i></i><i></i><i></i><i></i></div></div>'; }
    try { d = await api(`/api/wf/classify?k=${ck}`); } catch (e) { return; }
  }
  const none = !d || !d.labels, busy = S.busy && pendingSuggest;
  $("cempty").classList.toggle("hidden", !none);
  $("cbody").classList.toggle("hidden", none);
  $("etabs").innerHTML = "";
  if (none) {
    $("crun").disabled = !!S.busy;
    $("crun").innerHTML = busy ? '<span class="spin"></span> Classifying…' : `Classify ${(S.wf.prepared && S.wf.prepared.n > 1) ? "the spectra" : "the spectrum"}`;
    $("crun").onclick = () => { pendingSuggest = true; post("/api/wf/suggest", {}); loadClassify(true); };
    return;
  }
  classData = d;
  const many = d.labels.length > 1, fin = d.stage === "final", hasPre = fin && d.labels.some((x) => x.preliminary);
  $("cmulti").classList.toggle("hidden", !many);
  $("csum").classList.toggle("hidden", !many);
  if (many) renderClassSummary(d);
  if (many) {
    $("ctbl").innerHTML = `<thead><tr><th>#</th><th>Date</th><th>Spectrum</th><th>${fin ? "Class (fit)" : "Class"}</th>${hasPre ? "<th>Preliminary</th>" : ""}</tr></thead><tbody>` +
      d.labels.map((x) => `<tr data-k="${x.k}" class="${x.k === d.k ? "sel" : ""}"><td class="mono">#${x.k}</td><td class="mono">${esc(x.date || "—")}</td><td class="mono">${esc(x.name)}</td>` +
        `<td><span class="badge ${x.uncertain ? "warn" : "ok"}">${esc(x.label)}${x.uncertain ? " (uncertain)" : ""}</span></td>` +
        (hasPre ? `<td class="muted">${esc(x.preliminary || "—")}${x.preliminary && x.preliminary !== x.label ? " ≠" : ""}</td>` : "") + "</tr>").join("") + "</tbody>";
    $("ctbl").querySelectorAll("tr[data-k]").forEach((tr) => (tr.onclick = () => { ck = +tr.dataset.k; loadClassify(true); }));
    const pts = d.labels.filter((x) => x.points.nii).map((x) => ({ x: x.points.nii.x, y: x.points.nii.y, k: x.k, sel: x.k === d.k,
      color: CLASS_COLOR[x.label] || "#6b7280", tip: `#${x.k} ${x.name}: ${x.label}` }));
    plotScatter($("cbptall"), Object.assign({}, d.diagrams.nii, { points: pts, xlabel: "log [N II]/Hα", ylabel: "log [O III]/Hβ",
      onClick: (k) => { ck = k; loadClassify(true); } }));
  }
  const pre = d.preliminary;
  // the verdict first: the class in words, the evidence it rests on, and where the point falls in each diagram
  const regs = d.regions || {};
  const evidence = (d.reason || "").split(/;\s*/).filter(Boolean);
  $("cverdict").innerHTML = `<div class="cv-hero"><div class="cv-main"><small>${esc(d.name)} · ${fin ? "final class, from the fit" : "preliminary class"}</small>` +
    `<h2>${esc(CLASS_NAME[d.label] || d.label)}${d.uncertain ? ' <span class="badge warn">uncertain</span>' : ""}</h2>` +
    (evidence.length ? `<ul class="cv-ev">${evidence.map((t) => `<li>${esc(t)}</li>`).join("")}</ul>` : "") + `</div>` +
    `<div class="cv-chips">${Object.entries(DIAG_NAME).map(([k, n]) => `<button type="button" class="cvchip ${regs[k] ? "t-" + (REG_TONE[regs[k]] || "x") : "none"}" data-d="${k}"><small>${n}</small><b>${regs[k] ? esc(regs[k]) : "not measured"}</b></button>`).join("")}</div></div>` +
    `<p class="cv-foot small muted">` + (fin ? `From the fitted values: broad lines summed over their components, narrow (+ outflow) line ratios and EW(Hα) of the fit (<code>fa.classify_fit(r)</code>). ` +
      (pre ? `The preliminary classification (<code>fa.classify</code>, before the fit) said <b>${esc(CLASS_NAME[pre.label] || pre.label)}</b>${pre.label === d.label ? ": the fit agrees." : "."}` : "No preliminary classification was made (Model step › Classify spectrum).")
      : `Preliminary: <code>fa.classify</code> fits the spectrum on its own, before the model is chosen. The final class comes from the fit (step 4).`) + `</p>`;
  $("cverdict").querySelectorAll(".cvchip").forEach((b) => (b.onclick = () => {
    const el = $("cd-" + b.dataset.d).closest(".plotcard");
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash");
  }));
  const steps = !!(d.starlight || d.halpha || d.lines_fit);
  $("cprelim").classList.toggle("hidden", !steps);
  $("cprelimhead").classList.toggle("hidden", !fin);
  if (fin && pre) $("cprelimtxt").textContent = `· ${CLASS_NAME[pre.label] || pre.label}${pre.uncertain ? " (uncertain)" : ""} · fa.classify, its own quick fits made before the model was chosen`;
  drawClassPlots();
  const diag = { nii: ["log [N II]/Hα", "log [O III]/Hβ"], sii: ["log [S II]/Hα", "log [O III]/Hβ"], oi: ["log [O I]/Hα", "log [O III]/Hβ"], whan: ["log [N II]/Hα", "log EW(Hα) [Å]"] };
  for (const [key, [xl, yl]] of Object.entries(diag)) {
    const p = d.points[key];
    // several epochs: all of them (coloured by class), the chosen one outlined; one spectrum: its point
    const pts = many ? d.labels.filter((x) => x.points[key]).map((x) => Object.assign({ k: x.k, sel: x.k === d.k, color: CLASS_COLOR[x.label] || "#6b7280",
      tip: `#${x.k} ${x.date || ""} ${x.name}: ${x.label} (x = ${x.points[key].x.toFixed(3)}, y = ${x.points[key].y.toFixed(3)}${x.points[key].fit && !fin ? ", narrow lines of the fit" : ""})` }, x.points[key]))
      : p ? [Object.assign({ color: C.accent, sel: true, tip: `${d.name}: x = ${p.x.toFixed(3)}, y = ${p.y.toFixed(3)}${p.fit && !fin ? " (narrow lines of the fit)" : ""}` }, p)] : [];
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
  const url = withToken(`/api/wf/classify.png?k=${classData.k}&kind=${cfigKind}&stage=${classData.stage || "final"}&v=${S.cells.length}`);
  $("cfig").src = url; $("cfigdl").href = url;
  $("cfigdl").download = `${classData.name}_${cfigKind === "steps" ? "classification" : "diagrams"}.png`;
  $("cfigpdf").onclick = () => download(`/api/wf/classify.pdf?k=${classData.k}&kind=${cfigKind}&stage=${classData.stage || "final"}&dl=1`);
}
// ----------------------------------------------------------------- sample
const SG = { id: null, show: "all", cls: null, q: "", sort: null, dir: 1, order: [] };
const CLASS_LABEL = { type1: "Type 1", type1_host: "Type 1 + host", seyfert2: "Seyfert 2", liner: "LINER", composite: "Composite", starforming: "Star-forming", retired: "Retired", passive: "Passive", unknown: "Unknown", uv_mgii: "UV, Mg II", uv_civ: "UV, C IV" };
const sgClass = (c) => CLASS_LABEL[c] || c || "";
function sgCell(col, v) {
  if (v === null) return "";
  if (col.key === "cls") return `<span class="sgdot" style="background:${CLASS_COLOR[v] || "#9aa1ad"}"></span>${esc(sgClass(v))}`;
  if (col.key === "preset") return `<span title="${esc(v)}">${esc(String(v).split("(")[0])}${/\(.+\)/.test(v) ? " +" : ""}</span>`;
  return col.digits === null ? esc(v) : esc(Number(v).toFixed(col.digits));
}
// the rows to show, in the order shown (blank values last, whichever way the column is sorted)
function sgVisible() {
  const sm = S.sample, q = SG.q.trim().toLowerCase(), name = sm.columns.findIndex((c) => c.key === "name"), cls = sm.columns.findIndex((c) => c.key === "cls");
  let out = sm.rows.map((_, i) => i).filter((i) => {
    const st = sm.status[i];
    if (SG.show === "fitted" && st === "failed") return false;
    if (SG.show === "catflag") { if (!sm.catalog || !sm.catalog.flags[i]) return false; }
    else if (SG.show !== "all" && SG.show !== "fitted" && st !== SG.show) return false;
    if (SG.cls && sm.rows[i][cls] !== SG.cls) return false;
    return !q || String(sm.rows[i][name]).toLowerCase().includes(q);
  });
  if (SG.sort !== null) {
    const c = SG.sort;
    out = out.slice().sort((a, b) => {
      const u = sm.rows[a][c], v = sm.rows[b][c];
      if (u === null || v === null) return u === v ? a - b : u === null ? 1 : -1;
      return (typeof u === "number" ? u - v : String(u).localeCompare(String(v))) * SG.dir || a - b;
    });
  }
  return out;
}
function renderSampleResults() {
  const sm = S.sample;
  if (!sm) return;
  const sig = JSON.stringify([sm.id, sk, SG.show, SG.cls, SG.q, SG.sort, SG.dir, sm.catalog ? sm.catalog.stamp : 0]);
  if ($("stbl").dataset.sig === sig) return;
  $("stbl").dataset.sig = sig;
  const tile = (key, label, value, note) => `<button type="button" class="${SG.show === key ? "on" : ""} ${key}" data-show="${key}"><small>${label}</small><b>${value}</b>${note ? `<i>${note}</i>` : ""}</button>`;
  $("sstats").innerHTML = tile("all", "Spectra", sm.n) + tile("fitted", "Fitted", sm.fitted, `${Math.round((100 * sm.fitted) / (sm.n || 1))}%`) +
    tile("check", "To check", sm.check, sm.check ? "poor fit, odd width or uncertain class" : "") + tile("failed", "Not fitted", sm.failed) +
    (sm.catalog ? tile("catflag", "Catalogs disagree", sm.catalog.flagged) : "") +
    `<div><small>Median χ²ᵣ</small><b>${sm.median_rchi2 === null ? "—" : sm.median_rchi2.toFixed(2)}</b></div>`;
  $("sstats").querySelectorAll("[data-show]").forEach((b) => (b.onclick = () => { SG.show = SG.show === b.dataset.show ? "all" : b.dataset.show; renderSampleResults(); }));
  $("sfails").classList.toggle("hidden", !sm.failures.length);
  if (sm.failures.length) {
    const fix = sm.failures.some((f) => f.fixable);
    $("sfails").innerHTML = `<div class="grow"><b>${sm.failed} of ${sm.n} spectra were not fitted.</b><ul>${sm.failures.map((f) => `<li>${f.n} × ${esc(f.reason)}</li>`).join("")}</ul>` +
      (fix ? `<span>Those need a UV model: choose one for all of them (Mg II or C IV) in the Fit step and fit again.</span>` : "") + `</div>` +
      (fix ? '<button class="btn small" type="button" id="sfixmodel">Choose a model</button>' : "");
    if (fix) $("sfixmodel").onclick = () => go("fit");
  }
  const classes = Object.entries(sm.classes);
  $("sclasses").innerHTML = classes.length ? `<button type="button" class="${SG.cls ? "" : "on"}" data-c="">All classes</button>` +
    classes.map(([c, n]) => `<button type="button" class="${SG.cls === c ? "on" : ""}" data-c="${esc(c)}"><span class="sgdot" style="background:${CLASS_COLOR[c] || "#9aa1ad"}"></span>${esc(sgClass(c))} <small>${n}</small></button>`).join("") : "";
  $("sclasses").querySelectorAll("button").forEach((b) => (b.onclick = () => { SG.cls = b.dataset.c || null; renderSampleResults(); }));
  const order = sgVisible();
  SG.order = order;
  $("scount").textContent = `${order.length} shown`;
  const span = sm.columns.length - sm.columns.findIndex((c) => c.key === "z") - 1;
  const zAt = sm.columns.findIndex((c) => c.key === "z");
  $("stbl").innerHTML = `<thead><tr><th aria-label="status"></th>${sm.columns.map((c, i) => `<th data-c="${i}" class="${c.digits === null ? "" : "num"} sort ${SG.sort === i ? "sorted" : ""}">${esc(c.label)}${SG.sort === i ? (SG.dir > 0 ? " ↑" : " ↓") : ""}</th>`).join("")}</tr></thead><tbody>` +
    order.map((i) => {
      const st = sm.status[i], r = sm.rows[i], note = sm.notes[i];
      const dot = `<td title="${esc(note || "fitted")}"><span class="sgst ${st}"></span></td>`;
      const cell = (c, j) => `<td class="${c.digits === null ? "" : "num"}">${sgCell(c, r[j])}</td>`;
      const body = st === "failed" && zAt >= 0
        ? sm.columns.slice(0, zAt + 1).map(cell).join("") + `<td colspan="${span}" class="sgnote">${esc(note)}</td>`
        : sm.columns.map(cell).join("");
      return `<tr data-k="${i}" class="${i === sk ? "sel" : ""} ${st === "failed" ? "dim" : ""}">${dot}${body}</tr>`;
    }).join("") + (order.length ? "" : `<tr><td colspan="${sm.columns.length + 1}" class="muted">Nothing matches.</td></tr>`) + "</tbody>";
  $("stbl").querySelectorAll("tr[data-k]").forEach((tr) => (tr.onclick = () => selectSample(+tr.dataset.k)));
  $("stbl").querySelectorAll("th[data-c]").forEach((th) => (th.onclick = () => {
    const c = +th.dataset.c;
    if (SG.sort === c) { if (SG.dir < 0) { SG.sort = null; SG.dir = 1; } else SG.dir = -1; } else { SG.sort = c; SG.dir = 1; }
    renderSampleResults();
  }));
  const at = order.indexOf(sk), name = sm.columns.findIndex((c) => c.key === "name");
  $("ssel").textContent = sm.rows[sk] ? String(sm.rows[sk][name]) : "";
  $("sselnote").textContent = (at >= 0 ? `${at + 1} of ${order.length}` : "not in the filtered list") + (sm.notes[sk] && sm.status[sk] !== "failed" ? ` · ${sm.notes[sk]}` : "");
  $("sprev").disabled = at <= 0; $("snext").disabled = at < 0 || at >= order.length - 1;
}
function drawSampleDiagrams() {
  const P = samplePts;
  if (!P || rv !== "sdiag") return;
  const open = (k) => { rv = "fit"; selectSample(k); };
  const dot = (i) => CLASS_COLOR[(P.cls || [])[i]] || "#6b7280";
  const tip = (i, text) => `${(P.names || [])[i] || ""} · ${sgClass((P.cls || [])[i])} · ${text}`;
  const bpt = [], msq = [];
  (P.nii || []).forEach((x, i) => { const y = P.oiii[i]; if (x !== null && y !== null) bpt.push({ x, y, k: i, sel: i === sk, color: dot(i), tip: tip(i, `${x.toFixed(2)}, ${y.toFixed(2)}`) }); });
  const hasAll = (P.r_feii_all || []).some((v) => v !== null);
  $("smsqallbox").classList.toggle("hidden", !hasAll);
  const all = hasAll && $("smsqall").checked, RF = all ? P.r_feii_all : P.r_feii, HW = all ? P.hb_fwhm_all : P.hb_fwhm;
  (RF || []).forEach((x, i) => { const y = HW && HW[i]; if (x !== null && y !== null && y !== undefined) msq.push({ x, y: y / 1000, k: i, sel: i === sk, color: dot(i), tip: tip(i, `R(Fe II) ${x.toFixed(2)}, FWHM ${Math.round(y)} km/s`) }); });
  plotScatter($("sbpt"), Object.assign({}, P.nii_curves, { points: bpt, xlabel: "log [N II]/Hα", ylabel: "log [O III]/Hβ", onClick: open, empty: bpt.length ? null : "No spectrum has all four narrow lines" }));
  plotScatter($("smsq"), { xlim: [0, 2.5], ylim: [0, 16], points: msq, xlabel: "R(Fe II)", ylabel: "FWHM(Hβ) 10³ km/s", onClick: open, empty: msq.length ? null : "No spectrum has Fe II and broad Hβ",
    curves: (P.ms_grid || []).map((g) => ({ x: g.x, y: g.y.map((v) => v / 1000), dash: g.y[0] === 4000 && g.y[1] === 4000 ? null : "2 3" })),
    texts: (P.ms_types || []).map((t) => ({ x: t.x - 0.12, y: t.y / 1000 + 0.6, t: t.t })) });
  const seen = [...new Set(bpt.concat(msq).map((p) => (P.cls || [])[p.k]))];
  $("sdiaglegend").innerHTML = seen.map((c) => `<span><span class="sgdot" style="background:${CLASS_COLOR[c] || "#6b7280"}"></span>${esc(sgClass(c) || "no class")}</span>`).join("");
}
// --- what public catalogs know about the object (sent only on click)
function catalogRowHtml(r, again) {
  const fmt = (v, d) => (v === undefined || v === null ? "—" : (+v).toFixed(d));
  const line = (label, key) => r[`${key}_name`] === undefined ? "" :
    `<tr><td>${label}</td><td>${esc(r[`${key}_name`])}</td><td>${esc([r[`${key}_type`], r[`${key}_activity`]].filter(Boolean).join(" · ") || "—")}</td><td class="num">${fmt(r[`${key}_z`], 4)}</td><td class="num">${fmt(r[`${key}_sep`], 1)}″</td></tr>`;
  const rows = line("SIMBAD", "simbad") + line("NED", "ned") + line("Milliquas", "milliquas");
  const pos = `${(+r.ra).toFixed(5)}%20${(+r.dec).toFixed(5)}`;
  const simbad = `https://simbad.harvard.edu/simbad/sim-coo?Coord=${pos}&Radius=5&Radius.unit=arcsec`;
  const ned = `https://ned.ipac.caltech.edu/conesearch?search_type=Near%20Position%20Search&in_csys=Equatorial&in_equinox=J2000&ra=${(+r.ra).toFixed(5)}&dec=${(+r.dec).toFixed(5)}&radius=0.05`;
  const flags = r.disagreements ? r.disagreements.match(/[^.]+(?:\.\d+[^.]*)*\.(?:\s|$)/g) || [r.disagreements] : [];
  const wise = r.wise_name === undefined ? "not found" :
    ["w1", "w2", "w3", "w4"].map((b) => `${b.toUpperCase()} ${fmt(r[`wise_${b}`], 2)}`).join(" · ") +
    (r.wise_w1_w2 !== undefined ? `<br>W1−W2 = ${fmt(r.wise_w1_w2, 2)}${r.wise_w1_w2 >= 0.8 ? ' <span class="badge muted">AGN-like colour</span>' : ""}` : "");
  const radio = [r.first_mjy !== undefined ? `FIRST ${fmt(r.first_mjy, 1)} mJy` : "", r.nvss_mjy !== undefined ? `NVSS ${fmt(r.nvss_mjy, 1)} mJy` : ""].filter(Boolean).join(" · ") || "not found";
  const xray = [r.xmm_name, r.chandra_name, r.rosat_name].filter(Boolean).map(esc).join("<br>") || "not found";
  return `<div class="cathead"><b>Catalogs</b>${r.catalog_class ? `<span class="badge muted">${esc(r.catalog_class)}</span>` : ""}<span class="muted small">${esc(r.catalog_name || "")}</span><span class="grow"></span>` +
    `<a href="${simbad}" target="_blank" rel="noopener">SIMBAD ↗</a><a href="${ned}" target="_blank" rel="noopener">NED ↗</a>${again}</div>` +
    (flags.length ? `<div class="catflags"><ul>${flags.map((f) => `<li>${esc(f.trim())}</li>`).join("")}</ul></div>` : "") +
    `<div class="catgrid"><div>${rows ? `<table class="tbl"><thead><tr><th>Catalog</th><th>Name</th><th>Type</th><th class="num">z</th><th class="num">Offset</th></tr></thead><tbody>${rows}</tbody></table>` : '<p class="muted small">Not listed in SIMBAD, NED or Milliquas within 3″.</p>'}</div>` +
    `<dl class="catside"><dt>Infrared (AllWISE)</dt><dd>${wise}</dd><dt>Radio, 1.4 GHz</dt><dd>${radio}</dd><dt>X-ray</dt><dd>${xray}</dd></dl></div>` +
    (r.unreachable ? `<p class="muted small catfoot">Could not reach: ${esc(r.unreachable)}. Those results are missing, not negative.</p>` : "");
}
const CAT = { k: 0, fov: 60, row: null, rowKey: "", sky: null, skyKey: "", sig: "" };
async function catalogLoad(c) {
  const rowKey = `${c.stamp}:${CAT.k}`;
  if (CAT.rowKey !== rowKey) {
    CAT.rowKey = rowKey; CAT.row = null; CAT.sig = "";
    const row = await api(`/api/wf/catalog_row?k=${CAT.k}`).catch(() => null);
    if (CAT.rowKey !== rowKey) return;
    CAT.row = row; CAT.sig = ""; renderCatalogCard();
  }
  const skyKey = `${rowKey}:${CAT.fov}`;
  if (CAT.skyKey !== skyKey) {
    CAT.skyKey = skyKey; CAT.sky = "loading"; CAT.sig = ""; renderCatalogCard();
    const sky = await api(`/api/wf/sky?k=${CAT.k}&fov=${CAT.fov}`).catch(() => ({ image: null }));
    if (CAT.skyKey !== skyKey) return;
    CAT.sky = sky; CAT.sig = ""; renderCatalogCard();
  }
}
function catalogPick(k) {
  CAT.k = k;
  renderCatalogCard();
}
function catalogSkyHtml() {
  const sky = CAT.sky, marks = '<i class="h l"></i><i class="h r"></i><i class="v t"></i><i class="v b"></i>';
  const body = sky === "loading" || !sky ? "Loading image…" : sky.image ? `<img src="${sky.image}" alt="Sky image around the object">${marks}` : "No survey image at this position";
  const sizes = [[30, "30″"], [60, "1′"], [180, "3′"]].map(([v, l]) => `<button type="button" data-fov="${v}" class="${CAT.fov === v ? "on" : ""}">${l}</button>`).join("");
  return `<div><div class="catsky">${body}</div><div class="catskyfoot">${sizes}<span class="grow"></span><span>${sky && sky.survey ? esc(sky.survey) + " · N up, E left" : ""}</span></div></div>`;
}
function renderCatalogCard() {
  const el = $("catcard"), c = S.wf && S.wf.catalog;
  if (!el) return;
  el.classList.toggle("hidden", !c);
  document.querySelectorAll("#loadedtbl tbody tr").forEach((tr) => {
    tr.classList.toggle("catsel", c?.loaded_rows?.[+tr.dataset.row] === CAT.k);
  });
  if (!c) return;
  const many = c.n > 1;
  if (c.done) { CAT.k = Math.min(CAT.k, c.n - 1); catalogLoad(c); }
  const cell = S.cells[S.cells.length - 1], looking = !!(S.busy && cell && cell.code.startsWith("# catalogs"));
  const p = S.progress || {}, status = looking ? `Looking up… ${p.done || 0} of ${p.total || "?"}` : "";
  const sig = JSON.stringify([c.done, c.stamp, c.n, c.n_flagged, c.has_coords, CAT.k, CAT.fov, looking, status, !!S.busy, CAT.rowKey, !!CAT.row, CAT.sky && CAT.sky.image ? CAT.skyKey : String(CAT.sky && CAT.sky !== "loading")]);
  if (CAT.sig === sig) return;
  CAT.sig = sig;
  const btn = (cls, text) => `<button class="btn ${cls}" type="button" id="catgo" ${S.busy ? "disabled" : ""}>${looking ? esc(status) : text}</button>`;
  $("loadedtbl").classList.toggle("pick", !!c.done && many);
  if (!c.done) {
    el.innerHTML = c.has_coords || many
      ? `<div class="catintro"><div><b>What is known about ${many ? "these objects" : "this object"}?</b><p class="note">A sky image, with type, name and redshift from SIMBAD, NED and Milliquas, WISE, radio and X-ray matches, and a warning where they disagree with this spectrum. Only the sky position is sent.</p></div>${btn("", many ? `Look up all ${c.n} objects` : "Look up")}</div>`
      : `<div class="catintro"><div><b>Catalogs</b><p class="note">This file has no sky position, so the catalogs cannot be searched.</p></div></div>`;
  } else {
    const r = CAT.row;
    const bar = many ? `<div class="catbar"><span>${c.found} of ${c.n} found${c.n_flagged ? ` · ${c.n_flagged} disagree:` : " · no disagreements"}</span>${c.flagged.map((f) => `<button type="button" data-catk="${f.k}" class="${f.k === CAT.k ? "on" : ""}">${esc(f.name)}</button>`).join("")}</div>` : "";
    const name = many && r ? `<p class="small muted" style="margin:8px 0 0">Showing <b>${esc(r.name || "")}</b>. Click a row in the list above to see another.</p>` : "";
    const empty = r && r.note ? '<p class="muted small">No sky position in this file.</p>' : "";
    el.innerHTML = bar + name + (!r ? '<p class="muted small">Loading…</p>' : empty ||
      `<div class="catlayout">${catalogSkyHtml()}<div>${catalogRowHtml(r, btn("small", "Look up again"))}</div></div>`);
    el.querySelectorAll("[data-fov]").forEach((b) => (b.onclick = () => { CAT.fov = +b.dataset.fov; CAT.sig = ""; catalogLoad(c); renderCatalogCard(); }));
    el.querySelectorAll("[data-catk]").forEach((b) => (b.onclick = () => catalogPick(+b.dataset.catk)));
  }
  const go_ = $("catgo");
  if (go_) go_.onclick = () => post("/api/wf/catalogs", {});
}
// open spectrum k of the sample fit in Results (from the tiles of the Fit step)
function openSampleFit(k) {
  if (S.sample) {
    Object.assign(SG, { id: S.sample.id, show: "all", cls: null, q: "", sort: null, dir: 1 });
    if ($("ssearch")) $("ssearch").value = "";
  }
  sk = k; rv = "fit";
  go("results");
}
function selectSample(k) {
  sk = k;
  renderSampleResults();
  const tr = $("stbl").querySelector(`tr[data-k="${k}"]`);
  if (tr && rv === "fit") tr.scrollIntoView({ block: "nearest" });
  loadResult(true);
}
function stepSample(d) {
  const at = SG.order.indexOf(sk), k = SG.order[at + d];
  if (at >= 0 && k !== undefined) selectSample(k);
}
function initSample() {
  $("sprev").onclick = () => stepSample(-1);
  $("snext").onclick = () => stepSample(1);
  $("ssearch").oninput = (e) => { SG.q = e.target.value; renderSampleResults(); };
  $("stblwrap").addEventListener("keydown", (e) => {
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    e.preventDefault(); stepSample(e.key === "ArrowDown" ? 1 : -1);
  });
}
function initResults() {
  initSample();
  document.querySelectorAll("#rtabs button").forEach((b) => (b.onclick = () => { rtab = b.dataset.rt; renderResultTable(); }));
  $("hideund").onchange = renderResultTable;
  $("refine").onclick = () => go("model");
  document.querySelectorAll("#rnav button").forEach((b) => (b.onclick = () => { rv = b.dataset.v; loadResult(); }));
  $("tochecks").onclick = () => { rv = "checks"; loadResult(); window.scrollTo({ top: 0, behavior: "smooth" }); };
  // a range zoomed in the interactive plot can be taken over by the publication figure
  $("rplot").addEventListener("plotview", (e) => {
    const v = e.detail;
    renderZoomChips($("rplot"), $("rzoom"));
    fitRange = v ? [Math.round(v[0]), Math.round(v[1])] : null;
    $("userange").classList.toggle("hidden", !v);
    if (v) $("userange").dataset.range = `${Math.round(v[0])},${Math.round(v[1])}`;
  });
  $("userange").onclick = () => {
    const [a, b] = $("userange").dataset.range.split(",");
    document.querySelector('#figopts [data-o="xmin"]').value = a;
    document.querySelector('#figopts [data-o="xmax"]').value = b;
    rv = "figure"; loadResult(); figChanged();
  };
  document.querySelectorAll("#cfigkind button").forEach((b) => (b.onclick = () => { cfigKind = b.dataset.k; if (classData) showClassFig(); }));
  initActionMenu("dlbtn", "dlmenu", renderDownloads);
  $("figopts").onsubmit = (e) => e.preventDefault();
  $("figopts").querySelectorAll("[data-o]").forEach((el) => (el[el.type === "checkbox" || el.tagName === "SELECT" ? "onchange" : "oninput"] = figChanged));
  $("figopts").querySelectorAll("[data-fmt]").forEach((b) => (b.onclick = () => download(figUrl(b.dataset.fmt, true))));
  $("figreset").onclick = () => { try { localStorage.removeItem("fa-figstyle"); } catch (e) { /* storage off */ } resetForm($("figopts")); $("fo-draw").dataset.sig = ""; showFigure(); };
  initSegs($("figopts"), figChanged);
  initSegs($("lineopts"), linesChanged);
  initSections($("figopts"));
  initSections($("lineopts"));
  loadStyle($("figopts"), "data-o", "fa-figstyle");
  if (loadStyle($("lineopts"), "data-l", "fa-linestyle")) lineWidthTouched = true;
  $("linereset").onclick = () => { try { localStorage.removeItem("fa-linestyle"); } catch (e) { /* storage off */ } resetForm($("lineopts")); lineWidthTouched = false; setLineMode("grid"); linesChanged(); };
  $("lineopts").onsubmit = (e) => e.preventDefault();
  $("lineopts").querySelectorAll("[data-l]").forEach((el) => (el.onchange = linesChanged));
  $("lineopts").querySelectorAll("[data-lfmt]").forEach((b) => (b.onclick = () => lineSel.length && download(linesUrl(b.dataset.lfmt, true))));
  $("lo-modeseg").querySelectorAll("button").forEach((b) => (b.onclick = () => { setLineMode(b.dataset.mode); linesChanged(); }));
  $("lo-width").addEventListener("change", () => (lineWidthTouched = true));
  $("lo-win").oninput = linesChanged;
  setLineMode("grid");
  $("vcodebox").addEventListener("toggle", () => refreshViewCode(true));
  setInterval(() => refreshViewCode(false), 600);          // follows every change of the view while the box is open
  $("figcodebox").addEventListener("toggle", () => refreshCode("figcodebox", "figcode", figCodeUrl()));
  $("linecodebox").addEventListener("toggle", () => lineSel.length && refreshCode("linecodebox", "linecode", lineCodeUrl()));
  document.querySelectorAll("[data-copy]").forEach((b) => (b.onclick = async () => {
    try { await navigator.clipboard.writeText($(b.dataset.copy).textContent); toast("Code copied"); } catch (e) { toast("Select the code and copy it"); }
  }));
  $("cornerpng").onclick = () => download(cornerUrl("png", true));
  $("cornerpdf").onclick = () => download(cornerUrl("pdf", true));
}

// scatter diagram (BPT, WHAN): curves, region names and points with error bars; points can be clicked
function plotScatter(el, cfg) {
  const W = el.clientWidth || 300, H = el.clientHeight || 260, m = { l: 44, r: 10, t: 10, b: 36 };
  let pw = W - m.l - m.r; const ph = H - m.t - m.b;
  // the diagram's own range, widened to show every point with its error bars (pointsOnly: the range given is final)
  let [x0, x1] = cfg.xlim, [y0, y1] = cfg.ylim;
  for (const p of cfg.pointsOnly ? [] : cfg.points || []) {
    const xe = p.xerr || 0, ye = p.yerr || 0;
    if (!isFinite(p.x) || !isFinite(p.y)) continue;
    x0 = Math.min(x0, p.x - xe); x1 = Math.max(x1, p.x + xe); y0 = Math.min(y0, p.y - ye); y1 = Math.max(y1, p.y + ye);
  }
  if (x0 < cfg.xlim[0]) x0 -= 0.05 * (x1 - x0);
  if (x1 > cfg.xlim[1]) x1 += 0.05 * (x1 - x0);
  if (y0 < cfg.ylim[0]) y0 -= 0.05 * (y1 - y0);
  if (y1 > cfg.ylim[1]) y1 += 0.05 * (y1 - y0);
  // room for the tick labels of the y axis (a light curve that barely varies needs many digits)
  const ytl = niceTicks(y0, y1, Math.max(3, Math.floor(ph / 45)));
  const room = 14 + 6.4 * Math.max(...ytl.map((v) => tickText(v, ytl).length));
  if (room > m.l) { m.l = Math.min(room, 0.4 * W); pw = W - m.l - m.r; }
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
    const inX = (v) => Math.max(m.l, Math.min(m.l + pw, v)), inY = (v) => Math.max(m.t, Math.min(m.t + ph, v));   // error bars stop at the frame
    if (p.xerr) s += `<line x1="${inX(X(p.x - p.xerr))}" x2="${inX(X(p.x + p.xerr))}" y1="${py}" y2="${py}" stroke="${col}" stroke-width="1.2"/>`;
    if (p.yerr) s += `<line x1="${px}" x2="${px}" y1="${inY(Y(p.y - p.yerr))}" y2="${inY(Y(p.y + p.yerr))}" stroke="${col}" stroke-width="1.2"/>`;
    s += `<circle class="pt" data-k="${p.k ?? ""}" data-tip="${esc(p.tip || "")}" cx="${px}" cy="${py}" r="${p.sel ? 6 : 4}" fill="${col}" fill-opacity="${p.sel ? 1 : 0.75}" stroke="${p.sel ? "#0b0b0b" : "#fff"}" stroke-width="${p.sel ? 1.5 : 1}"/>`;
  }
  s += "</g>";
  s += `<g class="axis"><rect x="${m.l}" y="${m.t}" width="${pw}" height="${ph}"/>` +
    yt.map((v) => `<line x1="${m.l}" x2="${m.l + 5}" y1="${Y(v)}" y2="${Y(v)}"/><line x1="${m.l + pw - 5}" x2="${m.l + pw}" y1="${Y(v)}" y2="${Y(v)}"/>` +
      `<text x="${m.l - 5}" y="${Y(v) + 3.5}" text-anchor="end">${tickText(v, yt)}</text>`).join("") +
    xt.map((v) => `<line x1="${X(v)}" x2="${X(v)}" y1="${m.t}" y2="${m.t + 5}"/><line x1="${X(v)}" x2="${X(v)}" y1="${m.t + ph - 5}" y2="${m.t + ph}"/>` +
      `<text x="${X(v)}" y="${m.t + ph + 14}" text-anchor="middle">${tickText(v, xt)}</text>`).join("") +
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
// a tick label with as many decimals as the tick spacing needs (never the same text twice)
function tickText(v, ticks) {
  const step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : Math.abs(v) || 1;
  const d = Math.max(0, Math.min(8, Math.ceil(-Math.log10(step) - 1e-9)));
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e6 || a < 1e-4) && d > 6) return v.toExponential(2);
  return v.toFixed(d);
}
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
// for the y range only: a one- or two-pixel spike (cosmic ray, sky residual) must not set the scale, and emission
// lines are wider than that, so a running median of five pixels keeps them (when zoomed far in, every pixel counts)
function despike(v) {
  if (v.length < 40) return v;
  return v.map((_, i) => v.slice(Math.max(0, i - 2), i + 3).sort((a, b) => a - b)[Math.min(2, Math.min(i, v.length - 1 - i) + 1)]);
}
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
  for (const L of cfg.layers) {
    if (!L.range || L.hidden) continue;
    const v = [];
    for (let i = i0; i <= i1; i++) if (L.y[i] !== null && !masked(x[i])) v.push(L.y[i]);
    vals.push(...despike(v));
  }
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
    const foc = plotFocus.get(el), dim = foc !== undefined && L.name !== foc && L.name !== "data";
    s += `<path d="${d}" fill="none" stroke="${L.color}" stroke-width="${(L.width || 1.2) + (foc === L.name ? 1.2 : 0)}" ${L.dash ? `stroke-dasharray="${L.dash}"` : ""} opacity="${dim ? (L.name === "model" ? 0.35 : 0.15) : L.opacity ?? 1}" stroke-linejoin="round"/>`;
  }
  s += "</g>";
  for (const b of cfg.bands || []) if (b.label && b.hi >= x0 && b.lo <= x1) s += `<text x="${(X(b.lo) + X(b.hi)) / 2}" y="${m.t - 2}" text-anchor="middle" class="plab" style="fill:#52514e;font-size:10px">${esc(b.label)}</text>`;
  // marked lines (a candidate redshift): dashed lines with their names, alternately high and low
  if (cfg.marks) {
    let k = 0;
    for (const mk of cfg.marks) {
      if (mk.x < x0 || mk.x > x1) continue;
      const px = X(mk.x), col = mk.kind === "a" ? "#1f6f78" : "#d4472a";
      s += `<line x1="${px}" x2="${px}" y1="${m.t}" y2="${m.t + ph}" stroke="${col}" stroke-width="1" stroke-dasharray="4 3" opacity=".8"/>` +
        `<text x="${px + 3}" y="${m.t + 11 + 11 * (k++ % 3)}" class="plab" style="fill:${col};font-size:10px">${esc(mk.text)}</text>`;
    }
    if (cfg.marksTitle) s += `<text x="${m.l + pw - 4}" y="${m.t + ph - 6}" text-anchor="end" class="plab" style="fill:#d4472a;font-size:11px">${esc(cfg.marksTitle)}</text>`;
  }
  // line labels: strongest first, above the model, without overlaps
  let labS = "";
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
      labS += cfg.onLabel && lab.line ? `<text x="${lx}" y="${py}" text-anchor="middle" class="plab link" data-line="${esc(lab.line)}"><title>${esc(lab.text)}: open in Line profiles</title>${esc(lab.text)}</text>`
        : `<text x="${lx}" y="${py}" text-anchor="middle" class="plab">${esc(lab.text)}</text>`;
    }
  }
  if (!cfg.onLabel) s += labS;
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
  s += `<rect class="hit" x="${m.l}" y="${m.t}" width="${pw}" height="${xa - m.t}" fill="transparent"/>${cfg.onLabel ? labS : ""}</svg><div class="tip${cfg.rich ? " rich" : ""} hidden"></div>`;
  el.innerHTML = s;
  if (cfg.onLabel) el.querySelectorAll(".plab.link").forEach((t) => (t.onclick = () => cfg.onLabel(t.dataset.line)));
  const svg = el.querySelector("svg"), hit = svg.querySelector(".hit"), cross = svg.querySelector(".cross"), sel = svg.querySelector(".selrect"), tip = el.querySelector(".tip");
  const inv = (px) => x0 + ((px - m.l) / pw) * (x1 - x0);
  let drag = null;
  const pos = (e) => { const r = svg.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  hit.onmousemove = (e) => {
    const [px, py] = pos(e);
    cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.setAttribute("visibility", "visible");
    if (drag) { const a = Math.min(drag, px), b = Math.max(drag, px); sel.setAttribute("x", a); sel.setAttribute("width", b - a); sel.setAttribute("visibility", "visible"); }
    const wv = inv(px), i = Math.min(x.length - 1, lowerIndex(x, wv));
    if (cfg.rich) {
      // every curve under the pointer with its colour, the strongest component first, and the nearest labelled line
      const near = (cfg.labels || []).filter((l) => Math.abs(l.x - x[i]) < 25).sort((a, b) => Math.abs(a.x - x[i]) - Math.abs(b.x - x[i]))[0];
      const row = (L) => `<div><i style="background:${L.color}"></i><span>${esc(L.label || L.name)}</span><b>${fmtNum(L.y[i], 4)}</b></div>`;
      const vis = cfg.layers.filter((L) => !L.hidden && L.y[i] !== null && L.y[i] !== undefined);
      const mod = cfg.layers.find((L) => L.name === "model"), floor = 1e-3 * Math.abs(mod && mod.y[i] !== null ? mod.y[i] : 1);
      const comps = vis.filter((L) => L.name !== "data" && L.name !== "model" && Math.abs(L.y[i]) > floor).sort((a, b) => b.y[i] - a.y[i]);
      const top = vis.filter((L) => L.name === "data" || L.name === "model");
      const r = cfg.resid && cfg.resid[i] !== null ? cfg.resid[i] : null;
      tip.innerHTML = `<div class="tiphead"><b>${x[i].toFixed(1)} Å</b>${near ? `<span>${esc(near.text)}</span>` : ""}</div>` + top.map(row).join("") +
        (r !== null ? `<div><i class="res"></i><span>residual</span><b class="${Math.abs(r) > 3 ? "hot" : ""}">${r.toFixed(2)}σ</b></div>` : "") +
        (comps.length ? `<div class="tipsep"></div>` + comps.slice(0, 8).map(row).join("") : "");
      tip.classList.remove("hidden");
      const tw = tip.offsetWidth, th = tip.offsetHeight;
      tip.style.left = (px + 16 + tw > W ? px - 16 - tw : px + 16) + "px";
      tip.style.top = Math.max(0, Math.min(H - th, py - th / 2)) + "px";
      return;
    }
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
    else { st.view = [inv(a), inv(b)]; drawPlotEl(el); el.dispatchEvent(new CustomEvent("plotview", { detail: st.view })); }
  });
  hit.ondblclick = () => { st.view = null; drawPlotEl(el); el.dispatchEvent(new CustomEvent("plotview", { detail: null })); };
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
// the pages come from the server (app/guides.py and the docstrings): their HTML is trusted, titles may hold entities
const plainText = (h) => { const t = document.createElement("div"); t.innerHTML = h; return t.textContent; };
function renderDocList() {
  if (!docs) return;                       // typed before the pages arrived: openDocs renders them
  const q = $("docsearch").value.trim().toLowerCase();
  let sec = null, html = "";
  for (const d of docs) {
    if (q && !plainText(d.title + " " + d.html).toLowerCase().includes(q)) continue;
    if (d.section !== sec) { sec = d.section; html += `<li class="sec">${esc(sec)}</li>`; }
    html += `<li data-id="${esc(d.id)}">${d.title}</li>`;
  }
  $("doclist").innerHTML = html || '<li class="empty">nothing found</li>';
  $("doclist").querySelectorAll("li[data-id]").forEach((li) => (li.onclick = () => showDoc(li.dataset.id)));
  const on = $("doclist").querySelector("li.on, li[data-id]");
  if (q && on && !$("docbody").dataset.id) showDoc(on.dataset.id);
}
function showDoc(id, anchor) {
  const i = docs.findIndex((x) => x.id === id);
  if (i < 0) return;
  const d = docs[i], prev = docs[i - 1], next = docs[i + 1];
  const box = document.createElement("div");
  box.innerHTML = d.html;
  // "On this page": the h2 headings, given ids
  const heads = [...box.querySelectorAll("h2")];
  heads.forEach((h, k) => (h.id = `s-${k}`));
  const toc = heads.length > 1
    ? `<nav class="doctoc"><b>On this page</b>${heads.map((h) => `<a data-anchor="${h.id}">${h.innerHTML}</a>`).join("")}</nav>` : "";
  const link = (p, dir) => p ? `<a class="docnext ${dir}" data-doc="${esc(p.id)}"><small>${dir === "prev" ? "&larr; Previous" : "Next &rarr;"}</small>` +
    `<span>${p.title}</span></a>` : "<span></span>";
  $("docbody").dataset.id = id;
  $("docbody").innerHTML = `<header class="dochead"><span class="docsec">${esc(d.section)}</span><h1>${d.title}</h1></header>` +
    `<div class="docgrid"><div class="doccontent">${box.innerHTML}</div>${toc}</div>` +
    `<footer class="docfoot">${link(prev, "prev")}${link(next, "next")}</footer>`;
  $("docbody").querySelectorAll("[data-doc]").forEach((a) => (a.onclick = (e) => { e.preventDefault(); showDoc(a.dataset.doc); }));
  $("docbody").querySelectorAll("[data-anchor]").forEach((a) => (a.onclick = () => {
    const t = $(a.dataset.anchor); if (t) t.scrollIntoView({ behavior: "smooth", block: "start" });
  }));
  $("docbody").scrollTop = 0;
  if (anchor && $(anchor)) $(anchor).scrollIntoView();
  $("doclist").querySelectorAll("li[data-id]").forEach((li) => li.classList.toggle("on", li.dataset.id === id));
  const cur = $("doclist").querySelector("li.on"); if (cur) cur.scrollIntoView({ block: "nearest" });
}

// ===================================================================== new versions
// A newer FANTASY on PyPI shows as a pill in the header (checked once a day by the server); the panel
// says what changed, how this copy is updated, and installs it. The new version runs after a restart.
let UPD = null;
async function checkUpdate(force) {
  try { UPD = await api("/api/update" + (force ? "?force=1" : "")); } catch (e) { UPD = null; }
  if (UPD) $("curver").textContent = UPD.current;
  const show = !!(UPD && UPD.newer);
  $("updwrap").classList.toggle("hidden", !show);
  if (show) $("updbtn").textContent = `New: FANTASY ${UPD.latest}`;
  if (force) {
    if (!UPD || !UPD.checked) toast("The update check is off (FANTASY_NO_UPDATE_CHECK)", 3000);
    else if (!UPD.latest) toast("PyPI could not be reached: check the internet connection", 3000);
    else if (!show) toast(`FANTASY ${UPD.current} is the latest version`, 2500);
    else { renderUpdate(); $("updpanel").classList.remove("hidden"); }
  }
}
function renderUpdate(result) {
  const u = UPD;
  if (!u) return;
  const how = { "uv-tool": "installed with the FANTASY installer", pip: "installed with pip", source: "running from a source checkout" }[u.kind] || "";
  let h = `<h3>FANTASY ${esc(u.latest)} is published</h3>
    <div>You have ${esc(u.current)}, ${how}. <a href="${esc(u.changelog)}" target="_blank" rel="noopener">What's new</a> ·
      <a href="${esc(u.url)}" target="_blank" rel="noopener">PyPI</a></div>
    <div><span class="muted">${u.can_install ? "Install it here, or run in a terminal:" : "Update it in a terminal:"}</span><code>${esc(u.command)}</code></div>`;
  if (result) {
    h += result.ok
      ? `<div class="ok"><b>Installed.</b> Restart FANTASY to use it: save the project first (⋯ › Save project), stop this app (Ctrl+C in its terminal) and start it again with <code>fantasy app</code>.</div>`
      : `<div class="bad"><b>The update did not install.</b> Run the command above in a terminal.</div>`;
    if (result.output) h += `<details${result.ok ? "" : " open"}><summary class="muted">Output</summary><pre>${esc(result.output)}</pre></details>`;
  }
  h += `<div class="row">${u.can_install && !(result && result.ok) ? `<button class="btn primary" id="updinstall">Install ${esc(u.latest)}</button>` : ""}
    <button class="btn" id="updcopy">Copy command</button><button class="btn" id="updlater">Later</button></div>`;
  $("updpanel").innerHTML = h;
  if ($("updinstall")) $("updinstall").onclick = async () => {
    const b = $("updinstall"); b.disabled = true; b.textContent = "Installing… (up to a minute)";
    let res;
    try { res = await api("/api/update", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ version: u.latest }) }); }
    catch (e) { res = { ok: false, output: e.message }; }
    renderUpdate(res);
  };
  $("updcopy").onclick = async () => {
    try { await navigator.clipboard.writeText(u.command); toast("Copied", 1200); }
    catch (e) { const r = document.createRange(); r.selectNodeContents($("updpanel").querySelector("code")); getSelection().removeAllRanges(); getSelection().addRange(r); }
  };
  $("updlater").onclick = () => $("updpanel").classList.add("hidden");
}
function initUpdate() {
  $("updbtn").onclick = () => { renderUpdate(); $("updpanel").classList.toggle("hidden"); };
  document.querySelector('[data-act="check-update"]').onclick = () => checkUpdate(true);
  document.addEventListener("click", (e) => { if (!e.target.closest("#updwrap") && !e.target.closest('[data-act="check-update"]')) $("updpanel").classList.add("hidden"); });
  setTimeout(() => checkUpdate(false), 2500);        // after the page is up; the server asks PyPI at most daily
}

// ===================================================================== init
function initTop() {
  document.addEventListener("click", (e) => {
    const g = e.target.closest("[data-go]");
    if (g) { e.preventDefault(); go(g.dataset.go); }
    const d = e.target.closest("[data-dl]");
    if (d && !d.disabled) { e.preventDefault(); download(d.dataset.dl); }
  });
  initActionMenu("morebtn", "moremenu");
  $("stopbtn").onclick = async () => {
    $("stopbtn").disabled = true;
    await post("/api/stop", {});
  };
  $("newbtn").onclick = async () => {
    if (!confirm("Start a new analysis? The current session (code, spectra and fits) is cleared. It stays auto-saved, and the home page offers to restore it.")) return;
    restoreHidden = false; prep = null; modelTried = false; lastRun = null; runLog = []; resultData = null; fitInit = false; rk = 0; pk = 0;
    $("presetsel").innerHTML = "";
    if (await post("/api/reset", {})) go("home");
  };
  document.querySelector('[data-act="open-project"]').onclick = () => $("projectfile").click();
  // How to cite: from the ⋯ menu and the Results page
  document.querySelectorAll('[data-act="cite"]').forEach((b) => (b.onclick = () => {
    $("citever").textContent = S && S.version ? ` ${S.version}` : "";
    $("citedlg").showModal();
  }));
  $("citeclose").onclick = () => $("citedlg").close();
  $("citedlg").addEventListener("click", (e) => { if (e.target === $("citedlg")) $("citedlg").close(); });
  $("citecopy").onclick = async () => {
    try { await navigator.clipboard.writeText($("citebib").textContent); toast("BibTeX copied"); } catch (e) { toast("Select the BibTeX and copy it"); }
  };
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
  if (PAGES.includes(h)) return h;
  if (fitDone()) return "results";
  if (S.model && prepared()) return "model";
  if (prepared()) return "model";
  if (loaded()) return "prepare";
  return "home";
}
async function main() {
  initTop(); initUpdate(); initRead(); initPrepare(); initModel(); initFit(); initResults(); initTime(); initResid(); initSession();
  try { S = await api("/api/state"); } catch (e) { showError("Cannot reach the FANTASY server: " + e.message); return; }
  wasBusy = !!S.busy;
  go(guessPage());
  schedule();
}
main();

// ================================================================ Results › Anatomy
// The fit as stacked components (r.model_flux(components=True)), the share of each in the flux of the view,
// the composition at any wavelength, residuals in sigma and "peeling": the data minus the top components.
const AN = { F: null, key: null, view: null, peel: 0, off: new Set(), solo: null, hover: null, X: null, W: 0 };
const AN_GROUPS = [["continuum", "Continuum"], ["host", "Host galaxy"], ["balmer", "Balmer continuum"], ["feii", "Fe II"],
  ["broad", "Broad lines"], ["narrow", "Narrow lines"], ["coronal", "Coronal lines"], ["outflow", "Outflow"], ["other", "Other"]];
const AN_COLOR = { continuum: C.continuum, host: C.host, balmer: C.balmer, feii: C.feii, broad: C.broad[0], narrow: C.narrow, coronal: C.coronal, outflow: C.outflow, other: C.other[0] };
const AN_WINDOWS = [["Mg II", 2600, 3020], ["Blue", 3700, 4500], ["Hβ + [O III]", 4600, 5150], ["Hα + [N II]", 6350, 6800], ["C IV", 1450, 1720]];
const AN_M = { l: 56, r: 14, t: 34, b: 26 };
const anEl = (tag, attrs, parent) => { const e = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
const anFmt = (v) => v == null || !isFinite(v) ? "—" : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2);

function anatomyData(R) {
  const P = R.plot, n = P.x.length;
  const layers = [];
  for (const [role, name] of AN_GROUPS) {
    const ss = P.series.filter((s) => (s.role === role) || (role === "other" && !AN_COLOR[s.role]));
    if (!ss.length) continue;
    const y = new Array(n).fill(0);
    ss.forEach((s) => s.y.forEach((v, i) => (y[i] += v || 0)));
    if (!y.some((v) => v > 0)) continue;
    layers.push({ role, name, parts: ss.map((s) => s.name), y });
  }
  const flux = P.flux.map((v) => (v == null ? NaN : v)), model = P.model.map((v) => (v == null ? NaN : v));
  const res = P.resid.map((v) => (v == null ? NaN : v));
  const err = flux.map((f, i) => (isFinite(res[i]) && res[i] !== 0 ? (f - model[i]) / res[i] : NaN));
  return { x: P.x, flux, model, res, err, layers, labels: P.labels || [], masked: P.masked || [], name: R.name, fit: R.fit };
}
function drawAnatomy() {
  const R = resultData;
  if (!R || !R.plot) return;
  const key = `${R.k}|${R.fit && R.fit.chi2}|${R.name}`;
  if (AN.key !== key) {
    AN.F = anatomyData(R); AN.key = key;
    AN.view = [AN.F.x[0], AN.F.x[AN.F.x.length - 1]]; AN.peel = 0; AN.off = new Set(); AN.solo = null;
    $("an-peelr").max = Math.max(0, AN.F.layers.length - 1); $("an-peelr").value = 0;
    const lo = AN.view[0], hi = AN.view[1];
    const wins = AN_WINDOWS.filter(([, a, b]) => a >= lo - 50 && b <= hi + 50);
    $("an-zoom").innerHTML = '<button data-a="full" class="on">Full</button>' + wins.map(([t, a, b]) => `<button data-a="${a},${b}">${t}</button>`).join("");
    $("an-zoom").querySelectorAll("button").forEach((b) => (b.onclick = () => {
      AN.view = b.dataset.a === "full" ? [lo, hi] : b.dataset.a.split(",").map(Number).map((v, j) => (j ? Math.min(v, hi) : Math.max(v, lo)));
      anZoomButtons(b.dataset.a); anDraw();
    }));
    $("an-peelr").oninput = (e) => { AN.peel = +e.target.value; anMain(); anResid(); anPeel(); };
  }
  anDraw();
}
function anZoomButtons(which) { $("an-zoom").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.a === which)); }
function anDraw() { anBudget(); anMain(); anResid(); anMini(); anLegend(); anPeel(); anSide(); }
function anVisible(L) { return !AN.off.has(L.name) && (AN.solo === null || AN.solo === L.name); }
function anActive() { return AN.F.layers.slice(0, AN.F.layers.length - AN.peel); }
function anPeeled() { return AN.F.layers.slice(AN.F.layers.length - AN.peel); }
function anRange() {
  const x = AN.F.x; let a = 0, b = x.length - 1;
  while (a < b && x[a] < AN.view[0]) a++;
  while (b > a && x[b] > AN.view[1]) b--;
  return [a, b];
}
function anNear(w) { const x = AN.F.x; let lo = 0, hi = x.length - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (x[m] < w) lo = m; else hi = m; } return w - x[lo] < x[hi] - w ? lo : hi; }
function anShares(a, b) {
  const x = AN.F.x, tot = {}; let all = 0;
  for (const L of AN.F.layers) {
    let s = 0;
    for (let i = a; i <= b; i++) s += Math.max(L.y[i], 0) * (((x[Math.min(i + 1, b)] - x[Math.max(i - 1, a)]) / 2) || 1);
    tot[L.name] = s; all += s;
  }
  return AN.F.layers.map((L) => ({ name: L.name, role: L.role, f: all > 0 ? tot[L.name] / all : 0 }));
}
function anTicks(a, b, n) {
  const span = b - a, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const st = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const out = []; for (let t = Math.ceil(a / st) * st; t <= b + 1e-9; t += st) out.push(+t.toFixed(10)); return out;
}
function anBudget() {
  const [a, b] = anRange(), sh = anShares(a, b);
  $("an-range").textContent = `${AN.view[0].toFixed(0)}–${AN.view[1].toFixed(0)} Å`;
  const box = $("an-budget"); box.innerHTML = "";
  for (const s of sh) {
    const d = document.createElement("div");
    d.style.flex = `0 0 ${(s.f * 100).toFixed(3)}%`; d.style.background = AN_COLOR[s.role];
    d.style.opacity = AN.solo && AN.solo !== s.name ? 0.25 : 1;
    d.title = `${s.name}: ${(s.f * 100).toFixed(1)}%`;
    const px = box.clientWidth * s.f, full = `${s.name} ${(s.f * 100).toFixed(0)}%`, pct = `${(s.f * 100).toFixed(0)}%`;
    d.textContent = px > full.length * 7 + 12 ? full : px > pct.length * 7 + 8 ? pct : "";
    d.onclick = () => { AN.solo = AN.solo === s.name ? null : s.name; anDraw(); };
    box.appendChild(d);
  }
}
function anMain() {
  const F = AN.F, svg = $("an-plot"), M = AN_M; svg.innerHTML = "";
  const W = (AN.W = svg.clientWidth), H = svg.clientHeight;
  if (!W) return;
  const [a, b] = anRange(), act = anActive(), rem = anPeeled();
  const shown = F.flux.map((f, i) => f - rem.reduce((s, L) => s + L.y[i], 0));       // data minus the peeled layers
  const vals = []; for (let i = a; i <= b; i++) if (isFinite(shown[i])) vals.push(shown[i]);
  vals.sort((p, q) => p - q);
  let top = 0; for (let i = a; i <= b; i++) top = Math.max(top, act.reduce((s, L) => s + (anVisible(L) ? Math.max(L.y[i], 0) : 0), 0));
  const hi = Math.max(top, vals[Math.floor(vals.length * 0.995)] || 1) * 1.08;
  const lo = Math.min(0, vals[Math.floor(vals.length * 0.005)] || 0);
  const X = (AN.X = (w) => M.l + (w - AN.view[0]) / (AN.view[1] - AN.view[0]) * (W - M.l - M.r));
  const Y = (f) => M.t + (1 - (f - lo) / (hi - lo)) * (H - M.t - M.b);
  const g = anEl("g", {}, svg);
  for (const t of anTicks(lo, hi, 5)) { anEl("line", { x1: M.l, x2: W - M.r, y1: Y(t), y2: Y(t), stroke: "#eef0f3" }, g); anEl("text", { x: M.l - 8, y: Y(t) + 4, "text-anchor": "end" }, g).textContent = anFmt(t); }
  for (const t of anTicks(AN.view[0], AN.view[1], 8)) anEl("text", { x: X(t), y: H - 7, "text-anchor": "middle" }, g).textContent = t.toFixed(0);
  const mid = M.t + (H - M.t - M.b) / 2;
  anEl("text", { x: 14, y: mid, transform: `rotate(-90 14 ${mid})`, "text-anchor": "middle" }, g).textContent = "F_λ";
  const defs = anEl("defs", {}, svg);
  anEl("rect", { x: M.l, y: M.t - 2, width: W - M.l - M.r, height: H - M.t - M.b + 2 }, anEl("clipPath", { id: "an-clip" }, defs));
  const plot = anEl("g", { "clip-path": "url(#an-clip)" }, svg);
  if (AN.solo) plot.classList.add("dim");
  for (const [m0, m1] of F.masked) if (m1 > AN.view[0] && m0 < AN.view[1]) anEl("rect", { x: X(m0), y: M.t, width: Math.max(1, X(m1) - X(m0)), height: H - M.t - M.b, fill: "rgba(123,130,144,.10)" }, plot);
  const idx = []; for (let i = Math.max(0, a - 1); i <= Math.min(F.x.length - 1, b + 1); i++) idx.push(i);
  let base = idx.map(() => 0);
  act.forEach((L, li) => {
    const vis = anVisible(L);
    const topv = idx.map((i, j) => base[j] + (vis ? Math.max(L.y[i], 0) : 0));
    if (vis) {
      const gid = "an-grad" + li, col = AN_COLOR[L.role];
      const lg = anEl("linearGradient", { id: gid, x1: 0, x2: 0, y1: 0, y2: 1 }, defs);
      anEl("stop", { offset: "0%", "stop-color": col, "stop-opacity": 0.62 }, lg);
      anEl("stop", { offset: "100%", "stop-color": col, "stop-opacity": 0.28 }, lg);
      const lay = anEl("g", { class: "layer" + (AN.solo === L.name ? " hot" : "") }, plot);
      const edge = idx.map((i, j) => `${j ? "L" : "M"}${X(F.x[i]).toFixed(1)},${Y(topv[j]).toFixed(1)}`).join("");
      let d = edge; for (let j = idx.length - 1; j >= 0; j--) d += `L${X(F.x[idx[j]]).toFixed(1)},${Y(base[j]).toFixed(1)}`;
      anEl("path", { d: d + "Z", fill: `url(#${gid})` }, lay);
      anEl("path", { d: edge, fill: "none", stroke: col, "stroke-width": 1.1 }, lay);
    }
    base = topv;
  });
  let dd = ""; for (let i = a; i <= b; i++) if (isFinite(shown[i])) dd += `${dd ? "L" : "M"}${X(F.x[i]).toFixed(1)},${Y(shown[i]).toFixed(1)}`;
  anEl("path", { d: dd, fill: "none", stroke: "#1d2128", "stroke-width": 0.75, "stroke-opacity": 0.8, class: "data" }, plot);
  if (lo < 0) anEl("line", { x1: M.l, x2: W - M.r, y1: Y(0), y2: Y(0), stroke: "#9aa1ad", "stroke-dasharray": "3 3" }, plot);
  // line labels, two rows so that they do not overlap
  // the strongest first, so that a crowded region keeps the important lines; each needs the room of its text
  const taken = [[], []];
  F.labels.filter((l) => l.x > AN.view[0] && l.x < AN.view[1]).forEach((l) => {
    const x = X(l.x), half = (l.text.length * 6.2) / 2 + 5;
    const free = (row) => taken[row].every(([a, b]) => x + half < a || x - half > b);
    const row = free(0) ? 0 : free(1) ? 1 : -1;
    if (row < 0) return;
    taken[row].push([x - half, x + half]);
    anEl("line", { x1: x, x2: x, y1: M.t - 4, y2: H - M.b, stroke: "#cfd3d9", "stroke-dasharray": "2 3" }, svg);
    anEl("text", { x, y: 12 + row * 13, "text-anchor": "middle", class: "lbl" }, svg).textContent = l.text;
  });
  const cross = anEl("line", { y1: M.t, y2: H - M.b, stroke: "#1d2128", visibility: "hidden" }, svg);
  const dot = anEl("circle", { r: 3.5, fill: "#fff", stroke: "#1d2128", "stroke-width": 1.5, visibility: "hidden" }, svg);
  const brush = anEl("rect", { y: M.t, height: H - M.t - M.b, fill: "rgba(29,33,40,.07)", stroke: "rgba(29,33,40,.35)", "stroke-dasharray": "3 3", visibility: "hidden" }, svg);
  let x0 = null;
  const inv = (px) => AN.view[0] + (px - M.l) / (W - M.l - M.r) * (AN.view[1] - AN.view[0]);
  svg.onmousemove = (e) => {
    const px = e.clientX - svg.getBoundingClientRect().left;
    if (px < M.l || px > W - M.r) return;
    const i = anNear(inv(px)); AN.hover = i;
    for (const c of [cross, document.getElementById("an-rcross")]) if (c) { c.setAttribute("x1", X(F.x[i])); c.setAttribute("x2", X(F.x[i])); c.setAttribute("visibility", "visible"); }
    if (isFinite(shown[i])) { dot.setAttribute("cx", X(F.x[i])); dot.setAttribute("cy", Y(shown[i])); dot.setAttribute("visibility", "visible"); }
    if (x0 !== null) { brush.setAttribute("x", Math.min(x0, px)); brush.setAttribute("width", Math.abs(px - x0)); brush.setAttribute("visibility", "visible"); }
    anSide();
  };
  svg.onmouseleave = () => {
    for (const c of [cross, dot, document.getElementById("an-rcross")]) if (c) c.setAttribute("visibility", "hidden");
    AN.hover = null; x0 = null; brush.setAttribute("visibility", "hidden"); anSide();
  };
  svg.onmousedown = (e) => { x0 = e.clientX - svg.getBoundingClientRect().left; e.preventDefault(); };
  svg.onmouseup = (e) => {
    const x1 = e.clientX - svg.getBoundingClientRect().left;
    if (x0 !== null && Math.abs(x1 - x0) > 8) { AN.view = [inv(Math.min(x0, x1)), inv(Math.max(x0, x1))]; anZoomButtons(null); x0 = null; anDraw(); return; }
    x0 = null; brush.setAttribute("visibility", "hidden");
  };
  svg.ondblclick = () => { AN.view = [F.x[0], F.x[F.x.length - 1]]; anZoomButtons("full"); anDraw(); };
}
function anResid() {
  const F = AN.F, svg = $("an-resid"), M = AN_M; svg.innerHTML = "";
  const W = AN.W, h = svg.clientHeight, X = AN.X;
  if (!W || !X) return;
  const [a, b] = anRange(), lim = 5, top = 6, bot = 6;
  const y = (v) => top + (1 - (Math.max(-lim, Math.min(lim, v)) + lim) / (2 * lim)) * (h - top - bot);
  anEl("rect", { x: M.l, y: y(1), width: W - M.l - M.r, height: y(-1) - y(1), fill: "#e3e7ee" }, svg);
  for (const v of [3, -3]) anEl("line", { x1: M.l, x2: W - M.r, y1: y(v), y2: y(v), stroke: "#b7bdc7", "stroke-dasharray": "3 3" }, svg);
  anEl("line", { x1: M.l, x2: W - M.r, y1: y(0), y2: y(0), stroke: "#7b8290", "stroke-width": 0.8 }, svg);
  for (const v of [-3, 0, 3]) anEl("text", { x: M.l - 8, y: y(v) + 4, "text-anchor": "end" }, svg).textContent = (v > 0 ? "+" : "") + v + "σ";
  anEl("text", { x: 14, y: h / 2, transform: `rotate(-90 14 ${h / 2})`, "text-anchor": "middle" }, svg).textContent = "residual";
  anEl("rect", { x: M.l, y: 0, width: W - M.l - M.r, height: h }, anEl("clipPath", { id: "an-rclip" }, anEl("defs", {}, svg)));
  const plot = anEl("g", { "clip-path": "url(#an-rclip)" }, svg);
  let d = "", s = "", n = 0, s2 = 0, out = 0;
  const k = 7;
  for (let i = a; i <= b; i++) {
    const r = F.res[i];
    if (isFinite(r)) {
      d += `${d ? "L" : "M"}${X(F.x[i]).toFixed(1)},${y(r).toFixed(1)}`;
      n++; s2 += r * r;
      if (Math.abs(r) > 3) { out++; anEl("circle", { cx: X(F.x[i]), cy: y(r), r: 1.8, fill: "#c0362c" }, plot); }
    }
    let m = 0, c = 0; for (let j = Math.max(0, i - k); j <= Math.min(F.x.length - 1, i + k); j++) if (isFinite(F.res[j])) { m += F.res[j]; c++; }
    if (c) s += `${s ? "L" : "M"}${X(F.x[i]).toFixed(1)},${y(m / c).toFixed(1)}`;
  }
  anEl("path", { d, fill: "none", stroke: "#9b9a95", "stroke-width": 0.7 }, plot);
  anEl("path", { d: s, fill: "none", stroke: C.accent, "stroke-width": 1.6, "stroke-linejoin": "round" }, plot);
  anEl("line", { y1: 0, y2: h, stroke: "#1d2128", visibility: "hidden", id: "an-rcross" }, svg);
  $("an-residstat").textContent = n ? `χ²/N in view ${(s2 / n).toFixed(2)} · ${out} pixel${out === 1 ? "" : "s"} beyond 3σ (noise alone ~${Math.round(0.0027 * n)})` : "";
  svg.onmousemove = (e) => $("an-plot").onmousemove(e);
  svg.onmouseleave = () => $("an-plot").onmouseleave();
}
function anMini() {
  const F = AN.F, svg = $("an-mini"), M = AN_M; svg.innerHTML = "";
  const w = svg.clientWidth, h = svg.clientHeight; if (!w) return;
  const w0 = F.x[0], w1 = F.x[F.x.length - 1];
  const x = (v) => M.l + (v - w0) / (w1 - w0) * (w - M.l - M.r);
  const mx = Math.max(...F.model.filter(isFinite)) * 1.1, y = (f) => 4 + (1 - Math.max(f, 0) / mx) * (h - 8);
  let d = ""; F.x.forEach((v, i) => { if (isFinite(F.flux[i])) d += `${d ? "L" : "M"}${x(v).toFixed(1)},${y(F.flux[i]).toFixed(1)}`; });
  anEl("rect", { x: M.l, y: 0, width: w - M.l - M.r, height: h, fill: "#f7f8fa", rx: 4 }, svg);
  anEl("path", { d, fill: "none", stroke: C.data, "stroke-width": 0.7 }, svg);
  anEl("rect", { x: x(AN.view[0]), y: 0, width: Math.max(2, x(AN.view[1]) - x(AN.view[0])), height: h, fill: "rgba(212,71,42,.08)", stroke: "var(--accent)", "stroke-width": 1.2, rx: 3 }, svg);
  anEl("text", { x: M.l - 8, y: h / 2 + 4, "text-anchor": "end" }, svg).textContent = "all";
  svg.onclick = (e) => {
    const px = e.clientX - svg.getBoundingClientRect().left, c = w0 + (px - M.l) / (w - M.l - M.r) * (w1 - w0), half = (AN.view[1] - AN.view[0]) / 2;
    AN.view = [Math.max(w0, c - half), Math.min(w1, c + half)]; anZoomButtons(null); anDraw();
  };
}
function anLegend() {
  const box = $("an-legend"); box.innerHTML = "";
  AN.F.layers.forEach((L) => {
    const c = document.createElement("span"); c.className = "chip" + (AN.off.has(L.name) ? " off" : "");
    c.innerHTML = `<i style="background:${AN_COLOR[L.role]}"></i>${esc(L.name)}${L.parts.length > 1 ? `<span class="muted small">×${L.parts.length}</span>` : ""}`;
    c.title = L.parts.join(", ") + " · click: hide/show";
    c.onclick = () => { AN.off.has(L.name) ? AN.off.delete(L.name) : AN.off.add(L.name); anDraw(); };
    c.onmouseenter = () => { AN.solo = L.name; anMain(); anResid(); anBudget(); };
    c.onmouseleave = () => { AN.solo = null; anMain(); anResid(); anBudget(); };
    box.appendChild(c);
  });
}
function anPeel() {
  const names = AN.F.layers.map((L) => L.name);
  const short = (s) => s.replace(" lines", "").replace(" continuum", " cont.").replace(" galaxy", "");
  $("an-peelsteps").innerHTML = ["all"].concat(names.slice(1).reverse().map((s) => "− " + short(s))).map((s, k) => `<span class="${k === AN.peel ? "on" : ""}">${esc(s)}</span>`).join("");
  const rem = anPeeled().map((L) => L.name.toLowerCase()).reverse(), act = anActive().map((L) => L.name.toLowerCase());
  $("an-peelnote").innerHTML = AN.peel === 0
    ? "The data (black) over the stacked model: each band is one component, the continuum at the bottom. Move the slider to take components off the top."
    : `<b>Data − ${esc(rem.join(" − "))}</b>: what is left for ${esc(act.join(" + "))} to explain.` +
      (act.length === 1 ? " Only the continuum remains: structure left here is what the model gives to the other components." : "");
}
function anSide() {
  const F = AN.F; let rows, title, at, note;
  if (AN.hover !== null) {
    const i = AN.hover;
    rows = F.layers.map((L) => ({ name: L.name, role: L.role, v: Math.max(L.y[i], 0) }));
    const tot = rows.reduce((s, r) => s + r.v, 0); rows.forEach((r) => (r.f = tot > 0 ? r.v / tot : 0));
    title = "Composition at λ";
    at = `λ ${F.x[i].toFixed(1)} Å · data ${anFmt(F.flux[i])} ± ${anFmt(F.err[i])} · model ${anFmt(F.model[i])}`;
    const r = F.res[i], best = rows.slice().sort((p, q) => q.f - p.f)[0];
    note = (isFinite(r) ? `Residual here: <b style="color:${Math.abs(r) > 3 ? "#c0362c" : "inherit"}">${r >= 0 ? "+" : ""}${r.toFixed(1)}σ</b>. ` : "") +
      (best.f > 0.5 ? `Dominated by the ${esc(best.name.toLowerCase())}.` : "No single component dominates: these components share the flux here, and can trade it in the fit.");
  } else {
    const [a, b] = anRange(); rows = anShares(a, b);
    title = "Composition in view"; at = `${AN.view[0].toFixed(0)}–${AN.view[1].toFixed(0)} Å · integrated flux`;
    const lines = rows.filter((r) => ["broad", "narrow", "outflow"].includes(r.role)).reduce((s, r) => s + r.f, 0);
    note = `Emission lines carry <b>${(lines * 100).toFixed(0)}%</b> of the model flux in this range, the continuum components the rest. Hover the spectrum for any wavelength.`;
  }
  $("an-sidetitle").textContent = title; $("an-sideat").textContent = at; $("an-sidenote").innerHTML = note;
  const svg = $("an-donut"); svg.innerHTML = "";
  const R0 = 84, r0 = 58, cx = 100, cy = 100; let ang = -Math.PI / 2;
  const tot = rows.reduce((s, r) => s + r.f, 0) || 1;
  const pt = (rad, a) => `${(cx + rad * Math.cos(a)).toFixed(2)},${(cy + rad * Math.sin(a)).toFixed(2)}`;
  rows.forEach((r) => {
    const f = r.f / tot; if (f <= 0.0005) return;
    const a1 = ang + f * 2 * Math.PI - (f < 0.999 ? 0.012 : 0), big = f > 0.5 ? 1 : 0;
    const d = f > 0.999 ? `M${pt(R0, 0)}A${R0},${R0} 0 1 1 ${pt(R0, Math.PI)}A${R0},${R0} 0 1 1 ${pt(R0, 0)}M${pt(r0, 0)}A${r0},${r0} 0 1 0 ${pt(r0, Math.PI)}A${r0},${r0} 0 1 0 ${pt(r0, 0)}`
      : `M${pt(R0, ang)}A${R0},${R0} 0 ${big} 1 ${pt(R0, a1)}L${pt(r0, a1)}A${r0},${r0} 0 ${big} 0 ${pt(r0, ang)}Z`;
    anEl("path", { d, fill: AN_COLOR[r.role], "fill-rule": "evenodd", opacity: AN.solo && AN.solo !== r.name ? 0.25 : 1 }, svg);
    ang += f * 2 * Math.PI;
  });
  const best = rows.slice().sort((p, q) => q.f - p.f)[0];
  if (best) {
    anEl("text", { x: cx, y: cy + 2, "text-anchor": "middle", class: "c1" }, svg).textContent = `${(best.f / tot * 100).toFixed(0)}%`;
    anEl("text", { x: cx, y: cy + 20, "text-anchor": "middle", class: "c2" }, svg).textContent = best.name.toLowerCase();
  }
  $("an-sidetable").innerHTML = rows.map((r) => `<tr><td><i style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${AN_COLOR[r.role]};margin-right:7px"></i>${esc(r.name)}</td>` +
    `<td class="n">${(r.f / tot * 100).toFixed(1)}%</td><td class="anbar"><span style="width:${Math.max(1, r.f / tot * 70)}px;background:${AN_COLOR[r.role]}"></span></td></tr>`).join("");
}
window.addEventListener("resize", () => { if (rv === "anatomy" && AN.F) anDraw(); });

// ------------------------------------------------------------------ Results › Degeneracies
// The free parameters around a wheel, grouped by component; a ribbon joins two parameters whose
// correlation (from the MCMC samples, else the covariance) is at least the threshold.
const DG = { F: null, key: null, th: 0.3, scope: "between", hover: null, pinNode: null, pinPair: null };
const DG_FAM = ["continuum", "host", "balmer", "feii", "broad", "other", "narrow", "outflow"];
const DG_FAMNAME = { continuum: "continuum", host: "host", balmer: "Balmer", feii: "Fe II", narrow: "narrow", outflow: "outflow" };
const DG_POS = "#c8432f", DG_NEG = "#2a62c0", DG_R = 250;
const dgFmt = (v) => v == null || !isFinite(v) ? "—" : Math.abs(v) >= 1000 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : Math.abs(v) >= 0.1 ? v.toFixed(3) : v === 0 ? "0" : v.toExponential(2);
const dgPol = (r, a) => [r * Math.cos(a), r * Math.sin(a)];
function dgColor(g) { return g.family === "broad" ? C.broad[g.n % C.broad.length] : AN_COLOR[g.family] || C.other[0]; }
const dgMainLine = (g) => ["Hβ", "Hα", "Mg II 2798", "C IV 1549", "Lyα"].find((l) => g.lines.includes(l)) || (g.lines.length === 1 ? g.lines[0] : null);
function dgGroupName(g, all) {        // the family's name when it is the only one of it, else the component's
  const k = (g.name.match(/(\d+)$/) || [])[1];
  if (g.family === "broad") return dgMainLine(g) ? `broad ${dgMainLine(g)}${k ? " " + k : ""}` : g.name.replace(/_/g, " ").replace(/\bhe\b/, "He");
  return DG_FAMNAME[g.family] && all.filter((x) => x.family === g.family).length === 1 ? DG_FAMNAME[g.family] : g.name.replace(/_/g, " ");
}
function dgLabel(p) {          // "broad Hα · flux", not "broad Hα · Hα flux"
  const l = p.g.lines && dgMainLine(p.g), s = l && p.short.startsWith(l + " ") && p.g.label.endsWith(l) ? p.short.slice(l.length + 1) : p.short;
  return `${p.g.label} · ${s}`;
}

function dgPrepare(D) {
  const groups = D.groups.map((g, i) => ({ ...g, i, color: dgColor(g), label: dgGroupName(g, D.groups) }))
    .sort((a, b) => (DG_FAM.indexOf(a.family) - DG_FAM.indexOf(b.family)) || (a.i - b.i));
  const byName = Object.fromEntries(groups.map((g) => [g.name, g]));
  const P = D.params.map((p, i) => ({ ...p, i, g: byName[p.comp] || { name: p.comp, label: p.comp, color: C.other[0] },
    rel: p.err != null && p.value ? Math.abs(p.err / p.value) : null }));
  const gi = (p) => groups.indexOf(p.g);
  const order = P.slice().sort((a, b) => (gi(a) - gi(b)) || (a.i - b.i));
  const used = groups.filter((g) => order.some((p) => p.g === g));
  const gap = 1.6, slots = order.length + gap * used.length;
  let s = gap / 2;
  order.forEach((p, j) => {
    if (j && p.g !== order[j - 1].g) s += gap;
    p.ang = (s + 0.5) / slots * 2 * Math.PI - Math.PI / 2; s += 1;
  });
  const arcs = used.map((g) => { const ps = order.filter((p) => p.g === g), h = 0.45 / slots * 2 * Math.PI; return { g, a0: ps[0].ang - h, a1: ps[ps.length - 1].ang + h }; });
  const pairs = [];
  for (let i = 0; i < P.length; i++) for (let j = i + 1; j < P.length; j++) {
    const r = D.corr[i][j];
    if (Math.abs(r) >= 0.2) pairs.push({ a: P[i], b: P[j], r });
  }
  pairs.sort((x, y) => Math.abs(y.r) - Math.abs(x.r));
  P.forEach((p) => (p.ent = 0));
  pairs.forEach((q) => { if (q.a.comp !== q.b.comp) { q.a.ent = Math.max(q.a.ent, Math.abs(q.r)); q.b.ent = Math.max(q.b.ent, Math.abs(q.r)); } });
  return { D, P, order, arcs, pairs, limits: D.limits || [] };
}
async function drawDegen() {
  const kind = fitDone() && fitDone().kind;
  const k = kind === "sample" ? sk : rk, key = `${k}|${resultData && resultData.fit && resultData.fit.chi2}|${resultData && resultData.name}`;
  if (DG.key !== key) {
    DG.key = key; DG.F = null; DG.hover = DG.pinNode = DG.pinPair = null;
    // while the correlations load: a faint wheel turning, not an empty panel
    $("dg-wheel").innerHTML = '<g class="dg-wait"><circle r="250" fill="none" stroke="#e2e4e8" stroke-width="14"/><circle r="250" fill="none" stroke="#c9ced6" stroke-width="14" stroke-dasharray="260 1311" stroke-linecap="round"/>' +
      '<text y="10" text-anchor="middle" fill="#7b8290" font-size="30">Loading the correlations…</text></g>';
    $("dg-side").innerHTML = '<div class="dg-skel"><i></i><i></i><i></i><i></i><i></i></div>';
    let D = null;
    try { D = await api(`/api/wf/degeneracy?k=${k}`); } catch (e) { D = { error: String(e.message || e) }; }
    if (DG.key !== key) return;                 // another fit was opened meanwhile
    if (!D || D.error) {
      $("dg-src").textContent = "";
      $("dg-side").innerHTML = `<div class="empty"><p><b>No correlations to show.</b></p><p class="muted small">${esc((D && D.error) || "no fit")}</p></div>`;
      return;
    }
    DG.F = dgPrepare(D);
    $("dg-src").textContent = `${DG.F.P.length} free parameters · correlations from the ${D.source}`;
  }
  if (DG.F) dgDraw();
}
function dgShown() { return DG.F.pairs.filter((q) => Math.abs(q.r) >= DG.th && (DG.scope === "all" || q.a.comp !== q.b.comp)); }
function dgDraw() { dgWheel(); dgSide(); }
function dgArc(r0, r1, a0, a1) {
  const big = a1 - a0 > Math.PI ? 1 : 0, [x0, y0] = dgPol(r1, a0), [x1, y1] = dgPol(r1, a1), [x2, y2] = dgPol(r0, a1), [x3, y3] = dgPol(r0, a0);
  return `M${x0},${y0}A${r1},${r1} 0 ${big} 1 ${x1},${y1}L${x2},${y2}A${r0},${r0} 0 ${big} 0 ${x3},${y3}Z`;
}
function dgWheel() {
  const F = DG.F, svg = $("dg-wheel"), R = DG_R, S = dgShown();
  svg.innerHTML = "";
  for (const A of F.arcs) {
    anEl("path", { d: dgArc(R + 6, R + 16, A.a0, A.a1), fill: A.g.color }, svg);
    const am = (A.a0 + A.a1) / 2, [x, y] = dgPol(R + 112, am);
    anEl("text", { x, y, class: "grp", "text-anchor": "middle", "dominant-baseline": "middle" }, svg).textContent = A.g.label;
  }
  const cg = anEl("g", {}, svg);
  S.slice().reverse().forEach((q) => {
    const [x0, y0] = dgPol(R - 4, q.a.ang), [x1, y1] = dgPol(R - 4, q.b.ang);
    let da = Math.abs(q.a.ang - q.b.ang); if (da > Math.PI) da = 2 * Math.PI - da;
    const kk = 1 - da / Math.PI, mid = (q.a.ang + q.b.ang) / 2 + (Math.abs(q.a.ang - q.b.ang) > Math.PI ? Math.PI : 0);
    const [cx, cy] = dgPol(R * kk * 0.55, mid), a = Math.abs(q.r), t = Math.max(0, (a - 0.2) / 0.8);
    const path = anEl("path", { d: `M${x0},${y0}Q${cx},${cy} ${x1},${y1}`, fill: "none", stroke: q.r > 0 ? DG_POS : DG_NEG,
      "stroke-width": (0.6 + 6.5 * Math.pow(t, 1.6)).toFixed(2), "stroke-linecap": "round", opacity: (0.15 + 0.75 * Math.pow(t, 1.2)).toFixed(2), class: "chord" }, cg);
    path.dataset.a = q.a.i; path.dataset.b = q.b.i;
    anEl("title", {}, path).textContent = `${dgLabel(q.a)} ↔ ${dgLabel(q.b)}: ρ = ${q.r.toFixed(2)}`;
    path.onmouseenter = () => dgFocusPair(q);
    path.onmouseleave = () => dgFocusPair(null);
    path.onclick = () => { DG.pinPair = DG.pinPair === q ? null : q; DG.pinNode = null; dgFocusPair(DG.pinPair); };
  });
  const ng = anEl("g", {}, svg);
  F.order.forEach((p) => {
    const g = anEl("g", { class: "node" }, ng); g.dataset.i = p.i;
    const [x, y] = dgPol(R, p.ang);
    const rr = p.rel == null ? 2.5 : Math.max(2.2, Math.min(6.5, 2.2 + 2.2 * Math.log10(1 + 30 * p.rel)));
    anEl("circle", { cx: x, cy: y, r: rr, fill: p.lim ? "#fff" : p.g.color, stroke: p.lim ? "#b42318" : "#fff", "stroke-width": p.lim ? 1.8 : 1 }, g);
    const deg = p.ang * 180 / Math.PI, flip = deg > 90 && deg < 270, [tx, ty] = dgPol(R + 22, p.ang);
    const t = anEl("text", { x: tx, y: ty, "text-anchor": flip ? "end" : "start", "dominant-baseline": "middle", transform: `rotate(${flip ? deg + 180 : deg} ${tx} ${ty})` }, g);
    t.textContent = p.short;
    if (p.ent >= 0.9) t.setAttribute("style", "font-weight:600;fill:#1d2128");
    g.onmouseenter = () => { DG.hover = p; dgFocusNode(p); dgSide(); };
    g.onmouseleave = () => { DG.hover = null; dgFocusNode(DG.pinNode); dgSide(); };
    g.onclick = () => { DG.pinNode = DG.pinNode === p ? null : p; DG.pinPair = null; dgFocusNode(DG.pinNode); dgSide(); };
  });
  anEl("text", { x: 0, y: -8, "text-anchor": "middle", style: "font: 600 26px var(--sans); fill: #1d2128" }, svg).textContent = S.length;
  anEl("text", { x: 0, y: 14, "text-anchor": "middle", style: "font: 11px var(--mono); fill: #7b8290" }, svg).textContent = `pair${S.length === 1 ? "" : "s"} with |ρ| ≥ ${DG.th.toFixed(2)}`;
  dgFocusNode(DG.pinNode); if (DG.pinPair) dgFocusPair(DG.pinPair);
}
function dgFocusNode(p) {
  const svg = $("dg-wheel");
  svg.classList.toggle("focus", !!p);
  if (!p) { svg.querySelectorAll(".hot,.mate").forEach((e) => e.classList.remove("hot", "mate")); return; }
  const mates = new Set();
  svg.querySelectorAll(".chord").forEach((c) => { const on = +c.dataset.a === p.i || +c.dataset.b === p.i; c.classList.toggle("hot", on); if (on) { mates.add(+c.dataset.a); mates.add(+c.dataset.b); } });
  svg.querySelectorAll(".node").forEach((n) => { n.classList.toggle("hot", +n.dataset.i === p.i); n.classList.toggle("mate", mates.has(+n.dataset.i)); });
}
function dgFocusPair(q) {
  const svg = $("dg-wheel"), use = q || DG.pinPair;
  if (!use) { dgFocusNode(DG.hover || DG.pinNode); dgSide(); return; }
  svg.classList.add("focus");
  svg.querySelectorAll(".chord").forEach((c) => c.classList.toggle("hot", +c.dataset.a === use.a.i && +c.dataset.b === use.b.i));
  svg.querySelectorAll(".node").forEach((n) => { const i = +n.dataset.i; n.classList.toggle("hot", i === use.a.i || i === use.b.i); n.classList.remove("mate"); });
  dgSide(use);
}
function dgWhy(q) {
  const A = q.a, B = q.b, sa = A.short, sb = B.short, neg = q.r < 0;
  const base = (c) => c.replace(/2$/, ""), fw = (s) => /fwhm|shift|sigma/.test(s), fl = (s) => /flux|amp|norm/.test(s);
  if (base(A.comp) === base(B.comp) && A.comp !== B.comp) {
    if (fl(sa) && fl(sb)) return "Two components of the same line share its flux: their sum is well measured, the split between them is not.";
    return "Two components of the same line: what one takes in width or flux, the other gives back. Their sum is measured; the decomposition is not unique.";
  }
  if (A.comp === B.comp && ((/fwhm/.test(sa) && fl(sb)) || (/fwhm/.test(sb) && fl(sa))))
    return neg ? "A narrower profile needs more flux to fill the same peak." : "A wider profile carries more flux in its wings: width and flux rise together.";
  if (A.g.family === "feii" || B.g.family === "feii") return "Fe II multiplets overlap this line: flux given to one is taken from the other.";
  if (["continuum", "host", "balmer"].includes(A.g.family) || ["continuum", "host", "balmer"].includes(B.g.family))
    return neg ? "Continuum against a smooth component: flux moves between them under the same pixels." : "Both follow the same continuum level.";
  if (fl(sa) && fl(sb)) return neg ? "Two features under the same pixels: flux given to one is taken from the other." : "They rise and fall together in the fit.";
  if (fw(sa) || fw(sb)) return neg ? "Position or width of one shifts flux away from the other." : "Position or width of one pulls the other along.";
  return neg ? "They compete for the same pixels." : "They move together in the fit.";
}
const dgStrength = (r) => { const a = Math.abs(r); return a >= 0.9 ? "strong degeneracy" : a >= 0.7 ? "strong" : a >= 0.5 ? "moderate" : "weak"; };
const dgCol = (r) => (r > 0 ? DG_POS : DG_NEG);
function dgSide(pair) {
  const F = DG.F, side = $("dg-side");
  if (!F) return;
  const S = dgShown(), p = !pair && (DG.hover || DG.pinNode);
  const pick = () => side.querySelectorAll("tr.pick").forEach((tr) => (tr.onclick = () => { DG.pinPair = F.pairs[+tr.dataset.k]; DG.pinNode = null; dgFocusPair(DG.pinPair); }));
  if (pair) {
    const A = pair.a, B = pair.b, badge = (x) => (x.lim ? '<span class="dg-badge">limit</span>' : "");
    side.innerHTML = `<div><h3>${esc(dgLabel(A))} ↔ ${esc(dgLabel(B))}</h3><div class="at">ρ = ${pair.r.toFixed(2)} · ${dgStrength(pair.r)}</div></div>
      <svg id="dg-ellipse" viewBox="0 0 320 250"></svg>
      <table><tr><td class="pn">${esc(dgLabel(A))}</td><td class="n">${dgFmt(A.value)} ± ${dgFmt(A.err)}${badge(A)}</td></tr>
      <tr><td class="pn">${esc(dgLabel(B))}</td><td class="n">${dgFmt(B.value)} ± ${dgFmt(B.err)}${badge(B)}</td></tr></table>
      <div class="note">${dgWhy(pair)} The ellipses are the 68% and 95% regions of the two together, in units of each one's error: ${Math.abs(pair.r) > 0.8 ? "a thin diagonal ellipse means only a combination of the two is measured." : "a round one would mean independent."}</div>`;
    dgEllipse(pair);
    return;
  }
  if (p) {
    const mine = F.pairs.filter((q) => q.a === p || q.b === p).slice(0, 10);
    side.innerHTML = `<div><h3>${esc(dgLabel(p))}</h3><div class="at">${dgFmt(p.value)} ± ${dgFmt(p.err)} ${esc(p.unit)}${p.rel != null ? ` · ${(p.rel * 100).toFixed(1)}% error` : ""}${p.lim ? '<span class="dg-badge">at a limit</span>' : ""}</div></div>
      <table>${mine.map((q) => { const o = q.a === p ? q.b : q.a; return `<tr class="pick" data-k="${F.pairs.indexOf(q)}"><td><span class="pn">${esc(dgLabel(o))}</span></td><td class="n" style="color:${dgCol(q.r)}">${q.r > 0 ? "+" : ""}${q.r.toFixed(2)}</td></tr>`; }).join("") || '<tr><td class="why">no correlation above 0.2: independent of the rest</td></tr>'}</table>
      <div class="note">${mine.length && Math.abs(mine[0].r) >= 0.9 ? "Its error holds only while its strongest partner stays as fitted: change one and the other follows." : "Click a row for the pair."}</div>`;
    pick();
    return;
  }
  const top = F.pairs.filter((q) => q.a.comp !== q.b.comp).slice(0, 7);
  const ent = F.P.filter((x) => x.ent >= 0.9).length, lim = F.P.filter((x) => x.lim).length + F.limits.length;
  side.innerHTML = `<div><h3>What trades with what</h3><div class="at">the strongest correlations between components</div></div>
    <div class="dg-stat"><div><b>${ent}</b><span>parameters with |ρ| ≥ 0.9 to another component</span></div><div><b>${S.length}</b><span>pairs shown</span></div><div><b>${lim}</b><span>at a limit</span></div></div>
    <table>${top.map((q) => `<tr class="pick" data-k="${F.pairs.indexOf(q)}"><td><span class="pn">${esc(dgLabel(q.a))}</span><br><span class="pn">↔ ${esc(dgLabel(q.b))}</span><div class="why">${dgWhy(q)}</div></td>
      <td class="n" style="color:${dgCol(q.r)}">${dgGlyph(q.r)}${q.r > 0 ? "+" : ""}${q.r.toFixed(2)}</td></tr>`).join("") || '<tr><td class="why">no correlation above 0.2 between components</td></tr>'}</table>
    ${F.limits.length ? `<div class="note">At a limit or at zero, so without an error and not in the wheel: <span class="mono">${F.limits.map(esc).join(", ")}</span>.</div>` : ""}
    <div class="note">Hover a parameter for its partners, a ribbon or a row for the pair. A strong ribbon means the data constrain a combination of the two, not each one alone: report their sum, tie one, or expect the errors to be larger than they look.</div>`;
  pick();
}
// the 68% ellipse of a pair in units of the errors, small: its tilt and thinness are the correlation
function dgGlyph(r) {
  const k = 2.3, rx = Math.sqrt(k * (1 + r)), ry = Math.sqrt(k * Math.max(1 - r, 0.004)), sc = 9 / Math.sqrt(2 * k), col = dgCol(r);
  return `<svg class="dg-glyph" viewBox="-12 -12 24 24" aria-hidden="true"><line x1="-11" x2="11" y1="0" y2="0" stroke="#e2e4e8"/><line y1="-11" y2="11" x1="0" x2="0" stroke="#e2e4e8"/>` +
    `<ellipse rx="${(rx * sc).toFixed(2)}" ry="${(ry * sc).toFixed(2)}" transform="rotate(-45)" fill="${col}" fill-opacity=".25" stroke="${col}" stroke-width="1.2"/></svg>`;
}
function dgEllipse(q) {
  const svg = $("dg-ellipse"); if (!svg) return;
  const W = 320, H = 250, m = { l: 46, r: 12, t: 12, b: 34 }, cx = m.l + (W - m.l - m.r) / 2, cy = m.t + (H - m.t - m.b) / 2;
  const sc = Math.min(W - m.l - m.r, H - m.t - m.b) / 2 / 3.6;
  anEl("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "#fafbfc", stroke: "#e2e4e8" }, svg);
  for (const v of [-2, -1, 0, 1, 2]) {
    anEl("line", { x1: cx + v * sc, x2: cx + v * sc, y1: m.t, y2: H - m.b, stroke: v ? "#eef0f3" : "#cfd3d9" }, svg);
    anEl("line", { x1: m.l, x2: W - m.r, y1: cy - v * sc, y2: cy - v * sc, stroke: v ? "#eef0f3" : "#cfd3d9" }, svg);
    anEl("text", { x: cx + v * sc, y: H - m.b + 13, "text-anchor": "middle" }, svg).textContent = v ? (v > 0 ? "+" : "−") + Math.abs(v) + "σ" : dgFmt(q.a.value);
    if (v) anEl("text", { x: m.l - 5, y: cy - v * sc + 3, "text-anchor": "end" }, svg).textContent = (v > 0 ? "+" : "−") + Math.abs(v) + "σ";
  }
  anEl("text", { x: m.l - 5, y: cy + 3, "text-anchor": "end" }, svg).textContent = dgFmt(q.b.value);
  anEl("text", { x: cx, y: H - 4, "text-anchor": "middle" }, svg).textContent = q.a.short;
  anEl("text", { x: 11, y: cy, "text-anchor": "middle", transform: `rotate(-90 11 ${cy})` }, svg).textContent = q.b.short;
  // [[1, r], [r, 1]] has its axes on the diagonals with variances 1 ± r; 68% and 95% in 2-d are Δχ² = 2.30 and 6.18
  const r = q.r, col = dgCol(r);
  for (const [k, op] of [[6.18, 0.12], [2.30, 0.28]])
    anEl("ellipse", { cx, cy, rx: Math.sqrt(k * (1 + r)) * sc, ry: Math.sqrt(k * Math.max(1 - r, 1e-4)) * sc, transform: `rotate(-45 ${cx} ${cy})`, fill: col, "fill-opacity": op, stroke: col, "stroke-width": 1.2 }, svg);
  anEl("circle", { cx, cy, r: 3, fill: "#1d2128" }, svg);
}
$("dg-th").oninput = (e) => { DG.th = +e.target.value; $("dg-thv").textContent = DG.th.toFixed(2); DG.pinPair = null; if (DG.F) dgDraw(); };
document.querySelectorAll("#dg-scope button").forEach((b) => (b.onclick = () => {
  document.querySelectorAll("#dg-scope button").forEach((x) => x.classList.toggle("on", x === b)); DG.scope = b.dataset.s; if (DG.F) dgDraw();
}));

// ------------------------------------------------------------------ Results › Atlas
// Every fit of a sample (coloured by class) or every epoch of a multi-epoch fit (coloured by date) as a
// small card: data, model, broad lines, host and residuals, ordered by a number, with flags for the fits
// worth a look. A card opens its fit.
const AT = { data: null, key: null, sort: null, dir: -1, off: new Set(), flagged: false, common: false, same: false };
const AT_SORTS = {
  sample: [["rchi2", "χ²ν"], ["struct", "residual structure"], ["z", "redshift"], ["fwhm", "broad FWHM"], ["rfeii", "optical R(Fe II)"],
    ["mbh", "log M_BH"], ["edd", "log L/L_Edd"], ["lum", "log λL_λ"], ["o3hb", "log [O III]/Hβ"]],
  epochs: [["mjd", "date"], ["lum", "continuum log λL_λ"], ["bflux", "broad-line flux"], ["fwhm", "broad FWHM"], ["rchi2", "χ²ν"], ["struct", "residual structure"]],
};
const AT_FMT = { rchi2: (v) => v.toFixed(2), struct: (v) => v.toFixed(1) + "σ", z: (v) => v.toFixed(3), fwhm: (v) => (v / 1000).toFixed(1) + "k",
  rfeii: (v) => v.toFixed(2), mbh: (v) => v.toFixed(2), edd: (v) => v.toFixed(2), lum: (v) => v.toFixed(2), o3hb: (v) => v.toFixed(2),
  bflux: (v) => fmtNum(v, 3), mjd: (v) => String(Math.round(v)) };
const AT_RAMP = ["#3b2c84", "#2a62c0", "#1f9aa8", "#3fae6a", "#c5b72c", "#eda100"];
function atFlags(o) {
  const f = [];
  if (o.error) return f;
  if (o.rchi2 > 2) f.push(["χ²ν " + o.rchi2.toFixed(1), "the model misses structure the errors cannot explain"]);
  if (o.struct > 10) f.push(["residual stretch " + o.struct.toFixed(0) + "σ", "40 pixels in a row sit off the model by this much (noise alone gives about 4σ): a missing or misplaced component"]);
  if (o.width_limits?.length) f.push(["Broad width at a bound", o.width_limits.join(", ")]);
  if (o.rfeii != null && o.rfeii > 3) f.push(["Optical R(Fe II) " + o.rfeii.toFixed(1), "Inspect the Fe II blend, broad Hβ and continuum decomposition."]);
  if (o.mbh != null && (o.mbh > 10.5 || o.mbh < 5.5)) f.push(["M_BH 10^" + o.mbh.toFixed(1), "Check the broad-line width, luminosity and mass calibration."]);
  if ((o.limits || []).length >= 3) f.push([o.limits.length + " at limits", "parameters at a limit: " + o.limits.slice(0, 8).join(", ")]);
  return f;
}
function atRampColor(t) {
  t = Math.max(0, Math.min(1, t)) * (AT_RAMP.length - 1);
  const i = Math.min(AT_RAMP.length - 2, Math.floor(t)), u = t - i;
  const h = (s) => [1, 3, 5].map((j) => parseInt(s.slice(j, j + 2), 16)), a = h(AT_RAMP[i]), b = h(AT_RAMP[i + 1]);
  return "#" + a.map((v, j) => Math.round(v + (b[j] - v) * u).toString(16).padStart(2, "0")).join("");
}
async function drawAtlas() {
  const F = fitDone(), key = JSON.stringify([F && F.kind, F && F.chi2, F && F.time, F && F.n]);
  if (AT.key !== key) {
    AT.key = key; AT.data = null; AT.off = new Set(); AT.flagged = false;
    $("at-grid").innerHTML = '<div class="empty">Loading the cards…</div>'; $("at-dist").innerHTML = ""; $("at-chips").innerHTML = "";
    let d = null;
    try { d = await api("/api/wf/atlas"); } catch (e) { d = { error: String(e.message || e) }; }
    if (AT.key !== key) return;
    if (!d || !d.cards) { $("at-grid").innerHTML = `<div class="empty">${esc((d && d.error) || "No fits to show.")}</div>`; return; }
    const ep = d.kind === "epochs", mj = d.cards.map((c) => c.mjd).filter((v) => v != null);
    const m0 = Math.min(...mj), m1 = Math.max(...mj);
    d.cards.forEach((c) => {
      c.f = atFlags(c);
      c.color = ep ? (c.mjd != null && m1 > m0 ? atRampColor((c.mjd - m0) / (m1 - m0)) : C.broad[0]) : CLASS_COLOR[c.cls] || "#6b7280";
    });
    AT.data = d;
    AT.sort = ep ? "mjd" : "rchi2"; AT.dir = ep ? 1 : -1;
    AT.common = ep; AT.same = ep;
    $("at-common").checked = AT.common; $("at-same").checked = AT.same;
    $("at-sort").innerHTML = AT_SORTS[d.kind].map(([k, t]) => `<button data-k="${k}">${t}</button>`).join("");
    $("at-sort").querySelectorAll("button").forEach((b) => (b.onclick = () => { AT.sort = b.dataset.k; atRender(); }));
  }
  if (AT.data) atRender();
}
function atShown() {
  const k = AT.sort;
  return AT.data.cards.filter((o) => !AT.off.has(o.cls) && (!AT.flagged || o.f.length))
    .sort((a, b) => { const x = a[k], y = b[k]; if (x == null && y == null) return a.k - b.k; if (x == null) return 1; if (y == null) return -1; return AT.dir * (x - y); });
}
function atRender() {
  $("at-sort").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.k === AT.sort));
  $("at-dir").querySelectorAll("button").forEach((b) => b.classList.toggle("on", +b.dataset.d === AT.dir));
  $("at-dir").querySelector('[data-d="-1"]').textContent = AT.sort === "mjd" ? "latest first" : "high first";
  $("at-dir").querySelector('[data-d="1"]').textContent = AT.sort === "mjd" ? "earliest first" : "low first";
  atChips(); atDist(); atGrid();
}
function atChips() {
  const d = AT.data, ep = d.kind === "epochs", nf = d.cards.filter((o) => o.f.length).length;
  let h = `<span class="at-chip flag ${AT.flagged ? "on" : ""}" id="at-flag" title="fits flagged by χ²ν, residual structure, a width at a limit, R_FeII, the black-hole mass or parameters at limits">⚑ worth a look <b>${nf}</b></span>`;
  if (ep) {
    const ds = d.cards.map((c) => c.date).filter(Boolean).sort();
    h += `<span class="at-ramp">${esc(ds[0] || "")}<i style="background:linear-gradient(90deg,${AT_RAMP.join(",")})"></i>${esc(ds[ds.length - 1] || "")}</span>`;
  } else {
    const n = {}; d.cards.forEach((o) => (n[o.cls] = (n[o.cls] || 0) + 1));
    h += Object.keys(n).sort((a, b) => n[b] - n[a]).map((c) => `<span class="at-chip ${AT.off.has(c) ? "off" : ""}" data-c="${esc(c)}" title="click: hide or show · alt-click: only this class"><i style="background:${CLASS_COLOR[c] || "#9aa1ad"}"></i>${esc(CLASS_LABEL[c] || c || "failed")} <b>${n[c]}</b></span>`).join("");
  }
  $("at-chips").innerHTML = h;
  $("at-flag").onclick = () => { AT.flagged = !AT.flagged; atRender(); };
  $("at-chips").querySelectorAll(".at-chip[data-c]").forEach((ch) => (ch.onclick = (e) => {
    const c = ch.dataset.c;
    if (e.altKey || e.metaKey) AT.off = new Set(AT.data.cards.map((o) => o.cls).filter((x) => x !== c));
    else AT.off.has(c) ? AT.off.delete(c) : AT.off.add(c);
    atRender();
  }));
}
function atDist() {
  const svg = $("at-dist"), W = svg.clientWidth || 800, H = 64; svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.innerHTML = "";
  const k = AT.sort, title = (AT_SORTS[AT.data.kind].find((s) => s[0] === k) || [k, k])[1], fmt = AT_FMT[k] || ((v) => fmtNum(v, 3));
  const all = atShown(), xs = all.filter((o) => o[k] != null);
  if (!xs.length) { anEl("text", { x: 8, y: 30 }, svg).textContent = `no fit has ${title}`; return; }
  let lo = Math.min(...xs.map((o) => o[k])), hi = Math.max(...xs.map((o) => o[k])); if (hi === lo) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.04, m = { l: 14, r: 14 }, x = (v) => m.l + (v - lo + pad) / (hi - lo + 2 * pad) * (W - m.l - m.r);
  anEl("line", { x1: m.l, x2: W - m.r, y1: 44, y2: 44, stroke: "#cfd3d9" }, svg);
  for (let j = 0; j <= 5; j++) { const v = lo + (hi - lo) * j / 5; anEl("line", { x1: x(v), x2: x(v), y1: 44, y2: 48, stroke: "#cfd3d9" }, svg); anEl("text", { x: x(v), y: 59, "text-anchor": "middle" }, svg).textContent = k === "mjd" ? String(Math.round(v)) : fmt(v); }
  anEl("text", { x: W - m.r, y: 10, "text-anchor": "end" }, svg).textContent = `${title}${all.length - xs.length ? ` · ${all.length - xs.length} without it` : ""}`;
  const placed = [];
  xs.slice().sort((a, b) => a[k] - b[k]).forEach((o) => {
    const cx = x(o[k]); let lvl = 0;
    while (placed.some((p) => p.lvl === lvl && Math.abs(p.cx - cx) < 9)) lvl++;
    placed.push({ cx, lvl });
    const c = anEl("circle", { cx, cy: 38 - Math.min(lvl, 4) * 8.5 - Math.max(0, lvl - 4) * 2, r: 4, fill: o.color, stroke: o.f.length ? "#b42318" : "#fff", "stroke-width": o.f.length ? 1.6 : 1, id: `at-dot-${o.k}` }, svg);
    anEl("title", {}, c).textContent = `${o.name} · ${fmt(o[k])}`;
    c.onmouseenter = () => atHover(o.k, true); c.onmouseleave = () => atHover(null);
    c.onclick = () => atOpen(o);
  });
}
function atHover(k, scroll) {
  document.querySelectorAll(".at-card.hl").forEach((m) => m.classList.remove("hl"));
  document.querySelectorAll("#at-dist circle").forEach((c) => c.setAttribute("r", 4));
  if (k == null) return;
  const card = document.querySelector(`.at-card[data-k="${k}"]`), dot = $(`at-dot-${k}`);
  if (card) { card.classList.add("hl"); if (scroll) card.scrollIntoView({ block: "nearest", behavior: "smooth" }); }
  if (dot) dot.setAttribute("r", 7);
}
function atSVG(o, W, H, wr, smax) {
  const svg = anEl("svg", { viewBox: `0 0 ${W} ${H}`, height: H });
  const n = o.flux.length, rh = Math.round(H * 0.2), ph = H - rh - 3;
  const [w0, w1] = wr, xw = (j) => o.w0 + (o.w1 - o.w0) * (j + 0.5) / n, x = (j) => (xw(j) - w0) / (w1 - w0) * W;
  const sc = smax || o.scale || 1, ymax = 1.05, y = (v) => 2 + (1 - Math.max(0, Math.min(v / sc, ymax)) / ymax) * (ph - 4);
  const path = (arr, f = y) => { let d = "", pen = false; arr.forEach((v, j) => { if (v == null) { pen = false; return; } d += `${pen ? "L" : "M"}${x(j).toFixed(1)},${f(v).toFixed(1)}`; pen = true; }); return d; };
  if (AT.common) for (const lw of [1549, 2798, 4861, 5007, 6563]) if (lw > w0 && lw < w1) { const xx = (lw - w0) / (w1 - w0) * W; anEl("line", { x1: xx, x2: xx, y1: 0, y2: ph, stroke: "#eef0f3" }, svg); }
  const area = (arr) => { const d = path(arr); return d ? d + `L${x(n - 1)},${y(0)}L${x(0)},${y(0)}Z` : ""; };
  if (o.host) anEl("path", { d: area(o.host), fill: C.host, opacity: 0.09 }, svg);
  if (o.broad) anEl("path", { d: area(o.broad), fill: o.color, opacity: 0.14 }, svg);
  anEl("path", { d: path(o.flux), fill: "none", stroke: "#b9b8b3", "stroke-width": 0.9 }, svg);
  anEl("path", { d: path(o.model), fill: "none", stroke: o.color, "stroke-width": 1.3, "stroke-linejoin": "round" }, svg);
  const r0 = ph + 3, ry = (v) => r0 + rh / 2 - Math.max(-4, Math.min(4, v)) / 4 * (rh / 2);
  anEl("rect", { x: 0, y: r0, width: W, height: rh, fill: "#f6f7f9", rx: 2 }, svg);
  anEl("rect", { x: 0, y: ry(1), width: W, height: ry(-1) - ry(1), fill: "#eaeef4" }, svg);
  anEl("path", { d: path(o.resid, ry), fill: "none", stroke: o.f.length ? "#b42318" : "#4a5160", "stroke-width": 0.9 }, svg);
  return svg;
}
function atGrid() {
  const k = AT.sort, fmt = AT_FMT[k] || ((v) => fmtNum(v, 3)), list = atShown(), g = $("at-grid"), ep = AT.data.kind === "epochs";
  g.innerHTML = list.length ? "" : '<div class="empty">No fit matches.</div>';
  const cw = +$("at-size").value, W = cw - 20, H = Math.round(cw * 0.42);
  g.style.setProperty("--cw", cw + "px");
  const ok = AT.data.cards.filter((o) => !o.error);
  const wr = AT.common && ok.length ? [Math.min(...ok.map((o) => o.w0)), Math.max(...ok.map((o) => o.w1))] : null;
  const smax = AT.same && ok.length ? Math.max(...ok.map((o) => o.scale || 0)) : null;
  list.forEach((o) => {
    const card = document.createElement("div");
    card.className = "at-card" + (o.error ? " failed" : o.f.length ? " flagged" : ""); card.dataset.k = o.k;
    const v = o[k], label = ep ? (o.date || "") : (CLASS_LABEL[o.cls] || o.cls || "");
    card.innerHTML = `<div class="hd"><i style="background:${o.error ? "#c9c8c2" : o.color}"></i><span class="nm" title="${esc(o.name)}">${esc(ep ? `#${o.k}` : o.name.replace(/^spec-/, ""))}</span><span class="cl">${esc(label)}</span>
      <span class="val">${o.error || v == null ? "" : k === "mjd" ? "MJD " + Math.round(v) : fmt(v)}</span></div>`;
    if (o.error) { card.insertAdjacentHTML("beforeend", `<div class="why">not fitted: ${esc(o.error)}</div>`); g.appendChild(card); return; }
    card.appendChild(atSVG(o, W, H, wr || [o.w0, o.w1], smax));
    const fl = document.createElement("div"); fl.className = "fl";
    fl.innerHTML = (o.reference ? '<span class="ref">reference</span>' : "") + (o.f.map(([t, why]) => `<span title="${esc(why)}">${esc(t)}</span>`).join("") ||
      `<span class="ok">${ep ? "" : `z ${o.z != null ? o.z.toFixed(3) : "—"} · `}χ²ν ${o.rchi2.toFixed(2)}</span>`);
    card.appendChild(fl);
    card.title = o.f.length ? o.f.map(([t, why]) => `${t}: ${why}`).join("\n") + "\n\nClick to open the fit." : "Click to open the fit.";
    card.onmouseenter = () => atHover(o.k); card.onmouseleave = () => atHover(null);
    card.onclick = () => atOpen(o);
    g.appendChild(card);
  });
}
function atOpen(o) {           // the card's fit in Results › Fit
  if (o.error) return;
  if (AT.data.kind === "sample") sk = o.k; else rk = o.k;
  rv = "fit"; loadResult(true); window.scrollTo({ top: 0, behavior: "smooth" });
}
$("at-dir").querySelectorAll("button").forEach((b) => (b.onclick = () => { AT.dir = +b.dataset.d; if (AT.data) atRender(); }));
$("at-common").onchange = (e) => { AT.common = e.target.checked; if (AT.data) atGrid(); };
$("at-same").onchange = (e) => { AT.same = e.target.checked; if (AT.data) atGrid(); };
$("at-size").oninput = () => { if (AT.data) atGrid(); };
window.addEventListener("resize", () => { if (rv === "atlas" && AT.data) atDist(); });
