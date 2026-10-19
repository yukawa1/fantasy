// FANTASY app: talks to the local server (server.py) and draws the session.
"use strict";
const $ = (id) => document.getElementById(id);
const TOKEN = document.querySelector('meta[name="fantasy-token"]').content;
let state = null;
let clockOffset = 0;          // server time - browser time (s)
let lastFigure = -1;
let pollTimer = null;
const views = [];             // one view per code cell

// ================================================================ icons
const ICONS = {
  upload: '<path d="M7 18a5 5 0 0 1-.6-9.96A7 7 0 0 1 19.6 9.5 4.5 4.5 0 0 1 18 18"/><path d="M12 21v-9m-3.5 3.5L12 12l3.5 3.5"/>',
  arrow: '<path d="M5 12h14m-5-5 5 5-5 5"/>',
  left: '<path d="M19 12H5m5-5-5 5 5 5"/>',
  pulse: '<path d="M3 12h4l2.5-6 4 12 2.5-6H21"/>',
  sliders: '<path d="M4 6h10m4 0h2M4 12h4m4 0h8M4 18h12m4 0h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
  play: '<path d="M8 5v14l11-7z" fill="currentColor" stroke="none"/>',
  download: '<path d="M12 4v11m-4.5-4.5L12 15l4.5-4.5M5 19h14"/>',
  code: '<path d="m9 7-5 5 5 5m6-10 5 5-5 5"/>',
  book: '<path d="M5 4h9a4 4 0 0 1 4 4v12H8a3 3 0 0 1-3-3z"/><path d="M5 17a3 3 0 0 1 3-3h10"/>',
  github: '<path d="M9 19c-4 1.5-4-2-6-2.5m12 5v-3.5a3 3 0 0 0-.9-2.3c3-.3 6.1-1.5 6.1-6.7a5.2 5.2 0 0 0-1.4-3.6 4.8 4.8 0 0 0-.1-3.6s-1.1-.3-3.7 1.4a12.7 12.7 0 0 0-6.6 0C5.8 1.5 4.7 1.8 4.7 1.8a4.8 4.8 0 0 0-.1 3.6 5.2 5.2 0 0 0-1.4 3.6c0 5.2 3.1 6.4 6.1 6.7a3 3 0 0 0-.9 2.3V21"/>',
  refresh: '<path d="M20 12a8 8 0 1 1-2.3-5.6M20 4v5h-5"/>',
  table: '<rect x="4" y="5" width="16" height="14" rx="2"/><path d="M4 10h16M4 14.5h16M10 10v9"/>',
  database: '<ellipse cx="12" cy="6" rx="7" ry="2.8"/><path d="M5 6v12c0 1.5 3.1 2.8 7 2.8s7-1.3 7-2.8V6M5 12c0 1.5 3.1 2.8 7 2.8s7-1.3 7-2.8"/>',
  nodes: '<circle cx="6" cy="6" r="2"/><circle cx="18" cy="6" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="6" cy="18" r="2"/><circle cx="18" cy="18" r="2"/><path d="m7.5 7.5 3 3m3 0 3-3m-3 6 3 3m-6 0-3 3"/>',
  target: '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="1" fill="currentColor"/>',
  zoomin: '<circle cx="11" cy="11" r="6"/><path d="m20 20-4.5-4.5M11 8v6M8 11h6"/>',
  zoomout: '<circle cx="11" cy="11" r="6"/><path d="m20 20-4.5-4.5M8 11h6"/>',
  reset: '<path d="M4 9V4h5M20 15v5h-5M4 4l6 6m10 10-6-6"/>',
  expand: '<path d="M4 9V4h5m6 0h5v5m0 6v5h-5m-6 0H4v-5"/>',
  layers: '<path d="m12 4 8 4-8 4-8-4z"/><path d="m4 12 8 4 8-4M4 16l8 4 8-4"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
};
function icon(name) {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${ICONS[name] || ""}</svg>`;
}
function paintIcons(root = document) {
  for (const el of root.querySelectorAll("i[data-icon]")) if (!el.firstChild) el.innerHTML = icon(el.dataset.icon);
}
paintIcons();

// ================================================================ api
async function api(path, options = {}) {
  options.headers = Object.assign({ "X-Fantasy-Token": TOKEN }, options.headers || {});
  const r = await fetch(path, options);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}
function withToken(url) {
  return url + (url.includes("?") ? "&" : "?") + "t=" + encodeURIComponent(TOKEN);
}
for (const a of document.querySelectorAll("a.tok")) a.href = withToken(a.getAttribute("href"));
function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function fmt(v) {
  if (v === null || v === undefined) return "";
  const a = Math.abs(v);
  return (a !== 0 && (a < 1e-3 || a >= 1e6)) ? v.toExponential(3) : String(+v.toPrecision(6));
}

// ================================================================ views
let currentView = "analysis";
function setView(name, docId) {
  currentView = name;
  for (const v of ["analysis", "model", "multi", "docs"]) $("view-" + v).classList.toggle("hidden", v !== name);
  for (const l of document.querySelectorAll(".link[data-view-link]")) l.classList.toggle("active", l.dataset.viewLink === name);
  $("main").classList.toggle("docs", name === "docs");
  if (name === "docs") openDocs(docId);
  if (name === "model" && state) renderModelEditor(true);
  if (name === "analysis" && plotData) requestAnimationFrame(drawPlot);
  const hash = name === "docs" && docId ? `#docs/${docId}` : `#${name}`;
  if (location.hash !== hash) history.replaceState(null, "", hash);
}
document.addEventListener("click", (e) => {
  const help = e.target.closest(".help");
  if (help) { e.preventDefault(); setView("docs", help.dataset.doc); return; }
  const link = e.target.closest("[data-view-link]");
  if (link) { e.preventDefault(); setView(link.dataset.viewLink); }
});

// panel tabs: code / data
let ptab = "code";
for (const b of document.querySelectorAll(".ptab")) b.onclick = () => {
  ptab = b.dataset.ptab;
  for (const x of document.querySelectorAll(".ptab")) x.classList.toggle("active", x === b);
  $("cells").classList.toggle("hidden", ptab !== "code");
  $("datapane").classList.toggle("hidden", ptab !== "data");
  $("composer").classList.toggle("hidden", ptab !== "code");
  if (ptab === "data") loadData();
};
async function loadData() {
  try {
    const tables = await api("/api/data");
    $("datapane").innerHTML = tables.length ? tables.map((t) =>
      `<div class="dtable"><h3>${esc(t.title)}</h3><table><tr>${t.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>` +
      t.rows.map((r) => `<tr>${r.map((v) => `<td>${esc(v)}</td>`).join("")}</tr>`).join("") + "</table></div>").join("")
      : '<p class="placeholder">Read a spectrum: its properties, the classification and the fitted lines appear here.</p>';
  } catch (e) { $("datapane").textContent = e.message; }
}

// =============================================================== editor
function autosize(area) {
  area.style.height = "auto";
  area.style.height = area.scrollHeight + 2 + "px";
}
function makeView(index) {
  const root = document.createElement("div");
  root.className = "cell";
  root.innerHTML = '<div class="prompt"></div><div><div class="editor"><textarea spellcheck="false" rows="1"></textarea>' +
    `<button class="runbtn" title="Run (Shift+Enter)">▶</button></div><div class="outputs"></div></div>`;
  const v = { root, prompt: root.querySelector(".prompt"), area: root.querySelector("textarea"),
              outputs: root.querySelector(".outputs"), code: null, dirty: false, index };
  v.area.addEventListener("input", () => {
    v.dirty = v.area.value !== v.code;
    root.classList.toggle("dirty", v.dirty);
    autosize(v.area);
  });
  v.area.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.shiftKey) { e.preventDefault(); runView(v); }
    else if (e.key === "Tab") { e.preventDefault(); insertTab(v.area); }
  });
  root.querySelector(".runbtn").onclick = () => runView(v);
  return v;
}
function insertTab(a) {
  const s = a.selectionStart;
  a.value = a.value.slice(0, s) + "    " + a.value.slice(a.selectionEnd);
  a.selectionStart = a.selectionEnd = s + 4;
  a.dispatchEvent(new Event("input"));
}
function runView(v) {
  if (state && state.busy) return;
  v.dirty = false;
  post("/api/exec", { code: v.area.value, index: v.index });
}
const SPIN = "|/-\\";
let lastCellCount = 0;
function renderCells() {
  const now = Date.now() / 1000 + clockOffset;
  const box = $("cells");
  const cells = state.cells;
  while (views.length < cells.length) { const v = makeView(views.length); box.appendChild(v.root); views.push(v); }
  while (views.length > cells.length) views.pop().root.remove();
  cells.forEach((c, i) => {
    const v = views[i];
    v.prompt.textContent = c.running ? "In [*]:" : (c.n ? `In [${c.n}]:` : "In [ ]:");
    v.root.classList.toggle("running", c.running);
    v.root.classList.toggle("last", i === cells.length - 1 && !c.running);
    v.area.readOnly = !!state.busy;
    if (c.code !== v.code && (!v.dirty || document.activeElement !== v.area)) {
      v.code = c.code; v.area.value = c.code; v.dirty = false;
      v.root.classList.remove("dirty");
      autosize(v.area);
    }
    let html = "";
    if (c.running) {
      const el = now - c.start;
      const p = progressText();
      html += `<pre class="run">${SPIN[Math.floor(el * 6) % 4]} running … ${el.toFixed(0)} s${p ? ` <span class="prog">· ${esc(p)}</span>` : ""}</pre>`;
    }
    for (const o of c.out) html += `<pre class="${o.kind === "err" ? "err" : "out"}">${esc(o.text)}</pre>`;
    if (v.outputs.innerHTML !== html) v.outputs.innerHTML = html;
  });
  if (cells.length !== lastCellCount) { box.scrollTop = box.scrollHeight; lastCellCount = cells.length; }
  $("script").classList.toggle("hidden", cells.length < 2);
  $("saveproj").classList.toggle("hidden", cells.length < 2);
  $("runnew").disabled = !!state.busy;
}
// the composer at the bottom: a new cell
const newcode = $("newcode");
newcode.addEventListener("input", () => autosize(newcode));
newcode.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && e.shiftKey) { e.preventDefault(); runNew(); }
  else if (e.key === "Tab") { e.preventDefault(); insertTab(newcode); }
});
$("runnew").onclick = runNew;
function runNew() {
  const code = newcode.value.trim();
  if (!code || (state && state.busy)) return;
  post("/api/exec", { code }).then((ok) => { if (ok) { newcode.value = ""; autosize(newcode); } });
}

// ============================================================ analysis
const STEP_LABEL = { read: "Read", classify: "Classify", model: "Model", fit: "Fit", results: "Results" };
function nextStep() {
  for (const s of state.steps) if (!state.done.includes(s)) return s;
  return "results";
}
function renderStepper() {
  const cur = nextStep();
  $("stepper").innerHTML = state.steps.map((s, i) => {
    const cls = state.done.includes(s) ? "done" : (s === cur ? "cur" : "");
    const dot = state.done.includes(s) ? "✓" : i + 1;
    return `<button class="step ${cls}" data-goto-step="${s}"><span class="dot">${dot}</span>${STEP_LABEL[s]}` +
      (i < state.steps.length - 1 ? '<span class="sep"></span>' : "") + "</button>";
  }).join("");
}
$("stepper").onclick = (e) => {
  const b = e.target.closest("[data-goto-step]");
  if (!b) return;
  setView("analysis");
  const card = document.querySelector(`.card[data-step="${b.dataset.gotoStep}"]`) || $("card-fit");
  card.scrollIntoView({ behavior: "smooth", block: "start" });
};
function renderCards() {
  const busy = !!state.busy;
  const done = state.done;
  $("btn-read").disabled = busy;
  $("btn-classify").disabled = busy || !state.spectrum;
  $("btn-preset").disabled = busy;
  $("btn-fit").disabled = busy || !state.spectrum;
  $("btn-add").disabled = busy;
  const cur = nextStep();
  for (const card of document.querySelectorAll("#view-analysis .card[data-step]")) {
    const step = card.dataset.step;
    card.classList.toggle("done", done.includes(step));
    card.classList.toggle("cur", cur === step);
    card.classList.toggle("disabled", (step === "classify" || step === "fit") && !state.spectrum);
  }
  const badge = $("badge");
  if (state.recommended) {
    badge.textContent = state.recommended.label.replace("_", " ") + (state.recommended.uncertain ? " (uncertain)" : "");
    badge.classList.toggle("ok", !state.recommended.uncertain);
    badge.classList.remove("hidden");
  } else badge.classList.add("hidden");
  const m = state.model;
  $("model-summary").classList.toggle("hidden", !m);
  $("btn-preset").lastChild.textContent = m ? "Replace model" : "Create model";
  if (m) {
    $("model-chips").innerHTML = m.components.map((c) =>
      `<span class="chip" title="${esc(c.type)}${c.lines.length ? ": " + esc(c.lines.join(", ")) : ""}">${esc(c.name)}</span>`).join("") +
      `<span class="hint">${m.n_free} free parameters</span>`;
  }
  $("fit-hint").textContent = m ? "Fits the model of step 3 (edit it in the model editor)." : "Without a model the recommended preset is used.";
  const msg = $("message");
  const err = clientError || state.error;      // a refused action stays visible until the next one
  msg.classList.toggle("error", !!err);
  msg.textContent = err ? err : (busy ? "running …" : state.info);
  $("mmessage").textContent = msg.textContent;
  $("mmessage").classList.toggle("error", !!err);
  renderProgress();
  $("card-results").classList.toggle("hidden", !state.has_result);
  $("tiles").innerHTML = state.results.map((r) => {
    const m2 = /^(.*?)(?: ± (\S+))?(?: ([a-zA-Z/%]+))?$/.exec(r.value);
    const val = m2 ? `${esc(m2[1])}${m2[2] ? ` <small>± ${esc(m2[2])}</small>` : ""}${m2[3] ? ` <small>${esc(m2[3])}</small>` : ""}` : esc(r.value);
    return `<div class="tile"><div class="lab">${esc(r.label)}</div><div class="val">${val}</div></div>`;
  }).join("");
  $("warnings").innerHTML = state.warnings.map((w) => `<div>⚠ ${esc(w)}</div>`).join("");
  // read card side panel
  const side = $("side-read");
  if (state.spectrum) {
    side.querySelector("b").textContent = state.spectrum;
    side.querySelector("span").textContent = "Spectrum loaded — next: classify.";
  } else {
    side.querySelector("b").textContent = "Ready to analyze";
    side.querySelector("span").textContent = "Load a spectrum to get started.";
  }
  if (state.figure_version !== lastFigure) {
    lastFigure = state.figure_version;
    if (state.figure_version > 0) {
      $("figimg").src = withToken(`/api/figure.png?v=${lastFigure}&hd=1`);
      loadPlot();
      if (ptab === "data") loadData();
    } else {
      plotData = null;
      $("plotcard").classList.add("hidden");
      drawSpark();
    }
  }
}

// ============================================================== progress
// fit_many / fit_epochs report {task, done, total, detail}; shown while a cell runs
function progressText() {
  const p = state && state.busy && state.progress;
  if (!p) return "";
  const count = p.total ? `${p.done}/${p.total}` : (p.done ? `${p.done} steps` : "");
  return [p.task, count, p.detail].filter(Boolean).join(" · ");
}
function renderProgress() {
  const box = $("mprogress");
  const p = state.busy && state.progress;
  box.classList.toggle("hidden", !p);
  if (!p) return;
  box.classList.toggle("indeterminate", !p.total);
  box.querySelector(".bar").style.width = p.total ? `${(100 * p.done / p.total).toFixed(1)}%` : "";
  let text = progressText();
  if (p.total && p.done > 0 && p.done < p.total) {
    const el = p.time - p.start;
    if (el > 2) text += ` · about ${Math.ceil(el / p.done * (p.total - p.done))} s left`;
  }
  box.querySelector(".ptext").textContent = text;
}

// ========================================================= interactive plot
const COLORS = {
  data: "#5b6475", model: "#f26b3a", continuum: "#7c8698", host: "#9a9a2e", "Fe II": "#8b5cf6",
  Balmer: "#10b981", "narrow lines": "#d6336c", "broad lines": "#0ea5e9", outflow: "#f2a33a",
  "stars + power law": "#9a9a2e", "broad 1": "#0ea5e9", "broad 2": "#0ea5e9", "broad 3": "#0ea5e9",
};
let plotData = null, view = null, hidden = new Set(), drag = null, pv = "interactive";
async function loadPlot() {
  try {
    const d = await api("/api/plotdata");
    if (!d.x) return;
    // keep the zoom only when the new data cover the same wavelength range
    const lo = d.x[0], hi = d.x[d.x.length - 1];
    const same = plotData && view && Math.abs(plotData.x[0] - lo) < 1 && Math.abs(plotData.x[plotData.x.length - 1] - hi) < 1;
    plotData = d;
    if (!same) view = [lo, hi];
    $("plotcard").classList.remove("hidden");
    $("plottitle").textContent = d.title;
    const stem = state.figure_stem || "";
    const isFigureOnly = stem.endsWith("_classification") || stem === "light_curves";
    setPV(isFigureOnly ? "figure" : "interactive");
    renderLegend();
    drawPlot();
    drawSpark();
  } catch (e) { /* the figure is still available */ }
}
function setPV(v) {
  pv = v;
  for (const b of document.querySelectorAll(".segbtn")) b.classList.toggle("active", b.dataset.pv === v);
  $("plotwrap").classList.toggle("hidden", v !== "interactive");
  $("legend").classList.toggle("hidden", v !== "interactive");
  $("plothelp").classList.toggle("hidden", v !== "interactive");
  $("plottools").classList.toggle("hidden", v !== "interactive");
  $("figure").classList.toggle("hidden", v !== "figure");
  $("lineview").classList.toggle("hidden", v !== "line");
  if (v === "interactive") requestAnimationFrame(drawPlot);
  if (v === "line") showLine(lineName || (state.lines && state.lines.main));
}

// ============================================================ single line
let lineName = null, linesId = null;
function renderLines() {
  const L = state.lines;
  $("pv-line").disabled = !L || !L.lines.length;
  if (!L) { if (pv === "line") setPV("interactive"); linesId = null; return; }
  if (L.id === linesId) return;
  linesId = L.id;
  $("linesel").innerHTML = L.lines.map((d) =>
    `<option value="${esc(d.line)}">${esc(d.label)} · ${d.wave.toFixed(0)} Å${d.n > 1 ? ` · ${d.n} components` : ""}</option>`).join("");
  if (!L.lines.some((d) => d.line === lineName)) lineName = L.main;
  if (pv === "line") showLine(lineName);
}
function showLine(name) {
  const L = state && state.lines;
  if (!L || !name) return;
  lineName = name;
  const k = L.lines.findIndex((d) => d.line === name);
  const d = L.lines[k];
  $("linesel").value = name;
  $("lprev").disabled = k <= 0; $("lnext").disabled = k >= L.lines.length - 1;
  $("lineinfo").textContent = d ? `components: ${d.components.join(", ")}` : "";
  const q = `line=${encodeURIComponent(name)}&v=${L.id}`;
  $("ldl-pdf").href = withToken(`/api/line.pdf?${q}&dl=1`);
  $("ldl-png").href = withToken(`/api/line.png?${q}&dl=1`);
  $("lineimg").parentElement.classList.add("loading");
  $("lineimg").src = withToken(`/api/line.png?${q}&hd=1`);
  for (const j of [k - 1, k + 1]) if (L.lines[j]) new Image().src = withToken(`/api/line.png?line=${encodeURIComponent(L.lines[j].line)}&v=${L.id}&hd=1`);
}
function stepLine(dk) {
  const L = state && state.lines;
  if (!L) return;
  const k = L.lines.findIndex((d) => d.line === lineName) + dk;
  if (k >= 0 && k < L.lines.length) showLine(L.lines[k].line);
}
$("lineimg").addEventListener("load", (e) => {
  e.target.parentElement.classList.remove("loading");
  e.target.style.width = `${e.target.naturalWidth / 2}px`;
});
$("linesel").onchange = () => showLine($("linesel").value);
$("lprev").onclick = () => stepLine(-1);
$("lnext").onclick = () => stepLine(1);
document.addEventListener("keydown", (ev) => {
  if (pv !== "line" || currentView !== "analysis") return;
  if (/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName)) return;
  if (ev.key === "ArrowLeft") { stepLine(-1); ev.preventDefault(); }
  if (ev.key === "ArrowRight") { stepLine(1); ev.preventDefault(); }
});
for (const b of document.querySelectorAll(".segbtn")) b.onclick = () => setPV(b.dataset.pv);
function seriesList() {
  if (!plotData) return [];
  return [{ name: "data", y: plotData.flux }].concat(plotData.series);
}
function renderLegend() {
  $("legend").innerHTML = seriesList().map((s) => {
    const c = COLORS[s.name] || "#555";
    const dash = s.name === "continuum" ? "dashed" : "solid";
    return `<span class="lg ${hidden.has(s.name) ? "off" : ""}" data-s="${esc(s.name)}"><span class="sw" style="border-top-color:${c};border-top-style:${dash}"></span>${esc(s.name)}</span>`;
  }).join("");
}
$("legend").onclick = (e) => {
  const l = e.target.closest("[data-s]");
  if (!l) return;
  hidden.has(l.dataset.s) ? hidden.delete(l.dataset.s) : hidden.add(l.dataset.s);
  renderLegend(); drawPlot();
};
function niceTicks(lo, hi, n) {
  const span = hi - lo || 1, raw = span / n, p = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => span / s <= n) || 10 * p;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * span; v += step) out.push(+v.toPrecision(10));
  return out;
}
function quantile(arr, q) {
  const a = arr.filter((v) => v !== null && isFinite(v)).sort((x, y) => x - y);
  return a.length ? a[Math.min(a.length - 1, Math.max(0, Math.floor(q * (a.length - 1))))] : 0;
}
let layout = null;
function drawPlot() {
  const cv = $("plot");
  if (!plotData || pv !== "interactive" || $("plotcard").classList.contains("hidden") || currentView !== "analysis") return;
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  if (!W || !H) return;
  cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const d = plotData, hasRes = !!d.resid && !hidden.has("residuals");
  const L = 70, R = 14, T = 10, B = 44, gap = 8;
  const mainH = hasRes ? (H - T - B - gap) * 0.76 : H - T - B;
  const resH = hasRes ? (H - T - B - gap) * 0.24 : 0;
  const P = { x: L, y: T, w: W - L - R, h: mainH }, Q = { x: L, y: T + mainH + gap, w: W - L - R, h: resH };
  const [x0, x1] = view;
  const vis = [];
  for (let i = 0; i < d.x.length; i++) if (d.x[i] >= x0 && d.x[i] <= x1) vis.push(i);
  const ys = [];
  for (const s of seriesList()) if (!hidden.has(s.name)) for (const i of vis) ys.push(s.y[i]);
  // show whole lines: the top follows the (smooth) model, or robustly the data
  let ylo = Math.min(0, quantile(ys, 0.002));
  const model = d.series.find((x) => x.name === "model" && !hidden.has("model"));
  const mmax = model ? Math.max(...vis.map((i) => model.y[i] ?? -Infinity)) : -Infinity;
  let yhi = Math.max(mmax, quantile(vis.map((i) => d.flux[i]), 0.999)) * 1.06;
  if (!(yhi > ylo)) yhi = ylo + 1;
  const X = (x) => P.x + (x - x0) / (x1 - x0) * P.w;
  const Y = (y) => P.y + P.h - (y - ylo) / (yhi - ylo) * P.h;
  layout = { P, Q, X, x0, x1, ylo, yhi, vis };
  g.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  g.fillStyle = "#6b7486"; g.strokeStyle = "#eef1f5"; g.lineWidth = 1;
  // grid and ticks
  const xt = niceTicks(x0, x1, Math.max(3, Math.floor(P.w / 90)));
  const yt = niceTicks(ylo, yhi, Math.max(3, Math.floor(P.h / 55)));
  g.textAlign = "center"; g.textBaseline = "top";
  for (const t of xt) { const x = X(t); g.beginPath(); g.moveTo(x, P.y); g.lineTo(x, P.y + P.h); g.stroke(); if (hasRes) { g.beginPath(); g.moveTo(x, Q.y); g.lineTo(x, Q.y + Q.h); g.stroke(); } g.fillText(t.toLocaleString("en-US"), x, (hasRes ? Q.y + Q.h : P.y + P.h) + 6); }
  g.textAlign = "right"; g.textBaseline = "middle";
  for (const t of yt) { const y = Y(t); g.beginPath(); g.moveTo(P.x, y); g.lineTo(P.x + P.w, y); g.stroke(); g.fillText(+t.toPrecision(6), P.x - 6, y); }
  g.strokeStyle = "#c9d1dc"; g.strokeRect(P.x, P.y, P.w, P.h);
  // axis labels
  g.fillStyle = "#3c4556"; g.textAlign = "center"; g.textBaseline = "bottom";
  g.fillText(d.xlabel, P.x + P.w / 2, H - 2);
  g.save(); g.translate(14, P.y + P.h / 2); g.rotate(-Math.PI / 2); g.textBaseline = "middle"; g.fillText(d.ylabel, 0, 0); g.restore();
  // series
  g.save(); g.beginPath(); g.rect(P.x, P.y, P.w, P.h); g.clip();
  const order = seriesList().filter((s) => !hidden.has(s.name));
  order.sort((a, b) => (a.name === "data" ? -1 : a.name === "model" ? 1 : 0) - (b.name === "data" ? -1 : b.name === "model" ? 1 : 0));
  for (const s of order) {
    g.strokeStyle = COLORS[s.name] || "#555";
    g.lineWidth = s.name === "model" ? 1.7 : s.name === "data" ? 0.9 : 1.3;
    g.setLineDash(s.name === "continuum" ? [6, 4] : s.name === "broad 1" ? [7, 4] : s.name.startsWith("broad ") ? [2, 3] : []);
    if (s.name.startsWith("broad ")) g.lineWidth = 1.0;
    g.beginPath();
    let pen = false;
    for (const i of vis) { const y = s.y[i]; if (y === null) { pen = false; continue; } const px = X(d.x[i]), py = Y(y); pen ? g.lineTo(px, py) : g.moveTo(px, py); pen = true; }
    g.stroke();
  }
  g.restore(); g.setLineDash([]);
  // residuals
  if (hasRes) {
    const rr = vis.map((i) => d.resid[i]).filter((v) => v !== null);
    const lim = Math.max(4, Math.min(12, Math.ceil(quantile(rr.map(Math.abs), 0.99))));
    const RY = (v) => Q.y + Q.h / 2 - v / lim * Q.h / 2;
    g.strokeStyle = "#c9d1dc"; g.strokeRect(Q.x, Q.y, Q.w, Q.h);
    g.save(); g.beginPath(); g.rect(Q.x, Q.y, Q.w, Q.h); g.clip();
    g.strokeStyle = "#f6c1a8"; g.setLineDash([3, 3]);
    for (const v of [-3, 3]) { g.beginPath(); g.moveTo(Q.x, RY(v)); g.lineTo(Q.x + Q.w, RY(v)); g.stroke(); }
    g.setLineDash([]); g.strokeStyle = COLORS.model; g.beginPath(); g.moveTo(Q.x, RY(0)); g.lineTo(Q.x + Q.w, RY(0)); g.stroke();
    g.strokeStyle = "#5b6475"; g.lineWidth = 0.8; g.beginPath();
    let pen = false;
    for (const i of vis) { const v = d.resid[i]; if (v === null) { pen = false; continue; } const px = X(d.x[i]), py = RY(Math.max(-lim, Math.min(lim, v))); pen ? g.lineTo(px, py) : g.moveTo(px, py); pen = true; }
    g.stroke(); g.restore();
    g.fillStyle = "#6b7486"; g.textAlign = "right"; g.textBaseline = "middle";
    for (const v of [-lim, 0, lim]) g.fillText(v, Q.x - 6, RY(v));
    g.save(); g.translate(14, Q.y + Q.h / 2); g.rotate(-Math.PI / 2); g.textAlign = "center"; g.fillText("resid. (σ)", 0, 0); g.restore();
  }
  if (hover !== null) drawHover(g);
}
let hover = null;
function drawHover(g) {
  const { P, X } = layout, d = plotData, i = hover;
  const x = X(d.x[i]);
  g.strokeStyle = "rgba(15,27,42,.35)"; g.setLineDash([3, 3]);
  g.beginPath(); g.moveTo(x, P.y); g.lineTo(x, P.y + P.h + (layout.Q.h ? layout.Q.h + 8 : 0)); g.stroke(); g.setLineDash([]);
}
function nearest(px) {
  const { X, vis } = layout;
  let best = null, bd = 1e9;
  for (const i of vis) { const dd = Math.abs(X(plotData.x[i]) - px); if (dd < bd) { bd = dd; best = i; } }
  return best;
}
const cv = $("plot");
cv.addEventListener("wheel", (e) => {
  if (!layout) return;
  e.preventDefault();
  const { P } = layout, f = e.deltaY > 0 ? 1.2 : 1 / 1.2;
  const xm = view[0] + (e.offsetX - P.x) / P.w * (view[1] - view[0]);
  zoomAt(xm, f);
}, { passive: false });
function zoomAt(xm, f) {
  const full = [plotData.x[0], plotData.x[plotData.x.length - 1]];
  let a = xm - (xm - view[0]) * f, b = xm + (view[1] - xm) * f;
  if (b - a < 5) return;
  if (b - a >= full[1] - full[0]) { a = full[0]; b = full[1]; }
  if (a < full[0]) { b += full[0] - a; a = full[0]; }
  if (b > full[1]) { a -= b - full[1]; b = full[1]; }
  view = [Math.max(a, full[0]), Math.min(b, full[1])];
  drawPlot();
}
cv.addEventListener("mousedown", (e) => { drag = { x: e.offsetX, v: view.slice() }; });
window.addEventListener("mouseup", () => { drag = null; });
cv.addEventListener("mousemove", (e) => {
  if (!layout) return;
  const { P } = layout;
  if (drag) {
    const full = [plotData.x[0], plotData.x[plotData.x.length - 1]];
    const dx = (e.offsetX - drag.x) / P.w * (drag.v[1] - drag.v[0]);
    let a = drag.v[0] - dx, b = drag.v[1] - dx;
    if (a < full[0]) { b += full[0] - a; a = full[0]; }
    if (b > full[1]) { a -= b - full[1]; b = full[1]; }
    view = [a, b];
  }
  hover = nearest(e.offsetX);
  drawPlot();
  const tip = $("tip"), d = plotData;
  if (hover === null) return tip.classList.add("hidden");
  const lines = [`λ ${d.x[hover].toFixed(1)} Å`, `data  ${fmt(d.flux[hover])}`];
  for (const s of d.series) if (!hidden.has(s.name) && s.y[hover] !== null) lines.push(`${s.name}  ${fmt(s.y[hover])}`);
  tip.textContent = lines.join("\n");
  tip.classList.remove("hidden");
  const left = e.offsetX + 14 + 180 > cv.clientWidth ? e.offsetX - 190 : e.offsetX + 14;
  tip.style.left = left + "px"; tip.style.top = Math.max(4, e.offsetY - 20) + "px";
});
cv.addEventListener("mouseleave", () => { hover = null; $("tip").classList.add("hidden"); drawPlot(); });
// Alt-click near a fitted line: show that line alone
cv.addEventListener("click", (e) => {
  if (!e.altKey || !layout || !state.lines) return;
  const { P } = layout;
  const lam = view[0] + (e.offsetX - P.x) / P.w * (view[1] - view[0]);
  let best = null;
  for (const d of state.lines.lines) if (!best || Math.abs(d.wave - lam) < Math.abs(best.wave - lam)) best = d;
  if (best && Math.abs(best.wave - lam) < 0.02 * (view[1] - view[0]) + 15) { lineName = best.line; setPV("line"); }
});
cv.addEventListener("dblclick", () => { view = [plotData.x[0], plotData.x[plotData.x.length - 1]]; drawPlot(); });
for (const b of document.querySelectorAll("[data-zoom]")) b.onclick = () => {
  if (!plotData) return;
  if (b.dataset.zoom === "reset") { view = [plotData.x[0], plotData.x[plotData.x.length - 1]]; return drawPlot(); }
  zoomAt((view[0] + view[1]) / 2, b.dataset.zoom === "in" ? 1 / 1.5 : 1.5);
};
$("fullscreen").onclick = () => { $("plotcard").classList.toggle("full"); requestAnimationFrame(drawPlot); };
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && $("plotcard").classList.contains("full")) { $("plotcard").classList.remove("full"); requestAnimationFrame(drawPlot); } });
window.addEventListener("resize", () => requestAnimationFrame(drawPlot));

// small spectrum preview in the read card
function drawSpark() {
  const c = $("spark"), g = c.getContext("2d"), dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth || 200, H = c.clientHeight || 56;
  c.width = W * dpr; c.height = H * dpr; g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  const grad = g.createLinearGradient(0, 0, W, 0);
  grad.addColorStop(0, "#1f78b4"); grad.addColorStop(.45, "#c2185b"); grad.addColorStop(.75, "#f26b3a"); grad.addColorStop(1, "#f6c445");
  g.strokeStyle = grad; g.lineWidth = 1.4; g.beginPath();
  if (plotData) {
    const xs = plotData.x, ys = plotData.flux, hi = quantile(ys, 0.995) || 1, lo = Math.min(0, quantile(ys, 0.005));
    xs.forEach((x, i) => { const px = (x - xs[0]) / (xs[xs.length - 1] - xs[0]) * W, py = H - 2 - ((ys[i] ?? lo) - lo) / (hi - lo) * (H - 6); i ? g.lineTo(px, py) : g.moveTo(px, py); });
  } else {
    for (let i = 0; i <= 100; i++) { const x = i / 100, y = .15 + .5 * Math.exp(-(((x - .78) / .035) ** 2)) + .25 * Math.exp(-(((x - .55) / .05) ** 2)) + .1 * x; i ? g.lineTo(x * W, H - y * H) : g.moveTo(x * W, H - y * H); }
  }
  g.stroke();
}
drawSpark();

// ========================================================= model editor
let modelSig = null;
const pending = {};
function renderModelEditor(force) {
  const m = state.model;
  $("no-model").classList.toggle("hidden", !!m);
  $("components-card").classList.toggle("hidden", !m);
  $("params-card").classList.toggle("hidden", !m);
  if (!m) return;
  const sig = JSON.stringify(m);
  if (!force && sig === modelSig) return;
  modelSig = sig;
  $("model-info").textContent = `${m.components.length} components, ${m.n_free} free parameters` +
    (m.wave_range ? `, fitted ${m.wave_range[0]}–${m.wave_range[1]} Å` : "");
  $("components").innerHTML = "<tr><th>name</th><th>type</th><th>role</th><th>lines</th><th>parameters</th><th></th></tr>" +
    m.components.map((c) => `<tr><td class="name">${esc(c.name)}</td><td>${esc(c.type)}${c.profile && c.profile !== "gaussian" ? " (" + esc(c.profile) + ")" : ""}</td>` +
      `<td>${esc(c.role || "")}</td><td><span class="lines">${esc(c.lines.join(", "))}</span></td><td>${c.n_params}</td>` +
      `<td><button class="del" data-remove="${esc(c.name)}" title="remove ${esc(c.name)}">✕</button></td></tr>`).join("");
  $("paramlist").innerHTML = m.params.map((p) => `<option value="${esc(p.name)}">`).join("");
  renderParams();
}
$("components").onclick = (e) => {
  const b = e.target.closest("[data-remove]");
  if (b) modelAction({ action: "remove", name: b.dataset.remove });
};
function renderParams() {
  const m = state.model;
  if (!m) return;
  const q = $("pfilter").value.trim().toLowerCase();
  const fluxes = $("pfluxes").checked;
  const rows = m.params.filter((p) => (fluxes || !p.linear || q) && (!q || p.name.toLowerCase().includes(q)));
  $("params").innerHTML = "<tr><th>parameter</th><th>value</th><th>min</th><th>max</th><th>fixed</th><th>tied to</th><th>× factor</th><th></th></tr>" +
    rows.map((p) => {
      const e = pending[p.name] || {};
      const val = (k) => (k in e ? e[k] : fmt(p[k]));
      const fixed = "fixed" in e ? e.fixed : p.fixed;
      const tie = "tie" in e ? e.tie : (p.tie || "");
      const factor = "factor" in e ? e.factor : fmt(p.factor);
      return `<tr data-p="${esc(p.name)}" class="${pending[p.name] ? "changed" : ""} ${p.tie ? "tied" : ""}" title="${esc(p.description || "")}">` +
        `<td class="name">${esc(p.name)}</td>` +
        `<td><input type="number" step="any" data-k="value" value="${esc(val("value"))}"></td>` +
        `<td><input type="number" step="any" data-k="min" value="${esc(val("min"))}" placeholder="−∞"></td>` +
        `<td><input type="number" step="any" data-k="max" value="${esc(val("max"))}" placeholder="∞"></td>` +
        `<td><input type="checkbox" data-k="fixed" ${fixed ? "checked" : ""}></td>` +
        `<td><input type="text" class="tie" list="paramlist" data-k="tie" value="${esc(tie)}" placeholder="—"></td>` +
        `<td><input type="number" step="any" data-k="factor" value="${esc(factor)}" ${tie ? "" : "disabled"}></td>` +
        `<td class="unit">${esc(p.unit || "")}</td></tr>`;
    }).join("");
  const n = Object.keys(pending).length;
  $("btn-apply").disabled = !n || !!state.busy;
  $("btn-discard").disabled = !n;
  $("btn-apply").textContent = n ? `Apply ${n} change${n > 1 ? "s" : ""}` : "Apply changes";
}
$("params").addEventListener("change", (e) => {
  const input = e.target.closest("input[data-k]");
  if (!input) return;
  const name = input.closest("tr").dataset.p;
  const p = state.model.params.find((x) => x.name === name);
  const k = input.dataset.k;
  const e2 = pending[name] || {};
  e2[k] = input.type === "checkbox" ? input.checked : input.value;
  const same = (kk) => kk === "fixed" ? e2[kk] === p.fixed : kk === "tie" ? (e2[kk] || null) === (p.tie || null)
    : e2[kk] !== "" && Number(e2[kk]) === p[kk];
  for (const kk of Object.keys(e2)) if (same(kk)) delete e2[kk];
  if (Object.keys(e2).length) pending[name] = e2; else delete pending[name];
  renderParams();
});
$("pfilter").oninput = renderParams;
$("pfluxes").onchange = renderParams;
$("btn-discard").onclick = () => { for (const k of Object.keys(pending)) delete pending[k]; renderParams(); };
$("btn-apply").onclick = () => {
  const changes = Object.entries(pending).map(([name, e]) => {
    const p = state.model.params.find((x) => x.name === name);
    return { name, value: "value" in e ? e.value : null, min: "min" in e ? e.min : null, max: "max" in e ? e.max : null,
             fixed: "fixed" in e ? e.fixed : p.fixed, tie: "tie" in e ? (e.tie || null) : p.tie,
             factor: "factor" in e ? e.factor : p.factor };
  });
  for (const k of Object.keys(pending)) delete pending[k];
  modelAction({ action: "params", changes });
};
const newLines = [];
$("ctype").onchange = () => {
  for (const f of document.querySelectorAll(".ctype-fields"))
    f.classList.toggle("hidden", !f.dataset.for.split(" ").includes($("ctype").value));
};
function renderNewLines() {
  $("clines").innerHTML = newLines.map((l, i) => `<span class="chip">${esc(l)}<button data-i="${i}" title="remove">×</button></span>`).join("");
}
$("clines").onclick = (e) => { const b = e.target.closest("[data-i]"); if (b) { newLines.splice(+b.dataset.i, 1); renderNewLines(); } };
function addLine() {
  const v = $("cline").value.trim();
  if (v && !newLines.includes(v)) newLines.push(v);
  $("cline").value = "";
  renderNewLines();
}
$("cline-add").onclick = addLine;
$("cline").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); addLine(); } });
function triple(prefix) {
  const v = [0, 1, 2].map((i) => $(prefix + i).value);
  if (v.every((x) => x === "")) return null;
  if (v.some((x) => x === "")) throw new Error("give start, min and max (or leave all three empty)");
  return v.map(Number);
}
$("btn-add").onclick = () => {
  try {
    const type = $("ctype").value;
    const comp = { type, name: $("cname").value.trim() };
    if (type === "Lines") {
      if ($("cline").value.trim()) addLine();
      Object.assign(comp, { lines: newLines.slice(), role: $("crole").value, profile: $("cprofile").value,
                            fwhm: triple("fwhm"), shift: triple("shift") });
    } else if (type === "PowerLaw" || type === "BrokenPowerLaw") comp.pivot = $("cpivot").value || null;
    else if (type === "Polynomial") comp.order = $("corder").value;
    else if (type === "FeII") comp.fwhm = triple("ffwhm");
    else if (type === "Extinction") comp.law = $("claw").value;
    else if (type === "StellarPopulation") comp.templates = $("ctpl").value;
    modelAction({ action: "add", component: comp }).then((ok) => {
      if (ok) { newLines.length = 0; renderNewLines(); $("cname").value = ""; }
    });
  } catch (e) { showError(e.message); }
};
async function modelAction(body) { return post("/api/model", body); }

// ========================================================== multi-epoch
let mFiles = [];
function renderMulti() {
  if (!$("mmodel").options.length && state) {
    $("mmodel").innerHTML = '<option value="current">current model (step 3)</option>' +
      state.models.map((m) => `<option>${esc(m)}</option>`).join("");
    $("mmodel").value = "hbeta";
    applyModelDefaults("hbeta", $("moutflow"), null);
    $("mmodel").addEventListener("change", () => applyModelDefaults($("mmodel").value, $("moutflow"), null));
    $("mjobs").value = Math.max(1, (state.cpus || 2) - 1);
    $("mcpus").textContent = `${state.cpus} processor cores on this computer; the independent fits run in parallel.`;
  }
  $("btn-multi").disabled = !!state.busy;
  const m = state.multi;
  $("mresults").classList.toggle("hidden", !m);
  if (!m) return;
  const span = m.mjd ? `${(m.mjd[1] - m.mjd[0]).toFixed(0)} days` : "—";
  $("mtiles").innerHTML = [["epochs", m.n], ["joint χ²/ν", m.rchi2.toFixed(3)], ["time span", span],
    ["fit time", `${m.time.toFixed(0)} s`]].map(([a, b]) =>
    `<div class="tile"><div class="lab">${a}</div><div class="val">${esc(b)}</div></div>`).join("");
  $("minfo").textContent = "Broad-line fluxes and widths and the continuum per epoch (intercalibrated).";
  if (m.id !== mId) {                       // a new result: rebuild table and epoch list
    mId = m.id;
    $("mtable").innerHTML = `<tr>${m.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>` +
      m.rows.map((r, k) => `<tr data-k="${k}">${r.map((v) => `<td>${esc(v)}</td>`).join("")}</tr>`).join("");
    $("esel").innerHTML = m.epochs.map((e, k) =>
      `<option value="${k}">${k}: ${esc(e.name)}${e.mjd ? ` · MJD ${e.mjd}` : ""}${e.reference ? " · reference" : ""}</option>`).join("");
    showEpoch(0);
  }
  const src = withToken(`/api/figure.png?v=${state.figure_version}&hd=1`);
  if (state.figure_stem === "light_curves" && $("mfig").dataset.v !== String(state.figure_version)) {
    $("mfig").dataset.v = String(state.figure_version);
    $("mfig").src = src;
  }
}
let mId = null, epochK = 0;
function showEpoch(k) {
  const m = state && state.multi;
  if (!m) return;
  epochK = Math.max(0, Math.min(m.n - 1, k));
  const e = m.epochs[epochK];
  $("esel").value = String(epochK);
  $("eprev").disabled = epochK === 0;
  $("enext").disabled = epochK === m.n - 1;
  $("einfo").textContent = `Epoch ${epochK + 1} of ${m.n} · χ²/ν = ${e.rchi2.toFixed(3)} · flux scale ${e.scale.toFixed(3)}` +
    (e.reference ? " (reference)" : "");
  $("edl").href = withToken(`/api/epoch.pdf?k=${epochK}&dl=1`);
  for (const tr of $("mtable").querySelectorAll("tr[data-k]")) tr.classList.toggle("sel", Number(tr.dataset.k) === epochK);
  $("efig").parentElement.classList.add("loading");
  $("efig").src = withToken(`/api/epoch.png?k=${epochK}&hd=1&v=${m.id}`);
  // draw the neighbours in the background, so stepping through is quick
  for (const j of [epochK + 1, epochK - 1]) if (j >= 0 && j < m.n) new Image().src = withToken(`/api/epoch.png?k=${j}&hd=1&v=${m.id}`);
}
$("efig").addEventListener("load", (e) => {
  e.target.parentElement.classList.remove("loading");
  e.target.style.width = `${e.target.naturalWidth / 2}px`;
});
$("eprev").onclick = () => showEpoch(epochK - 1);
$("enext").onclick = () => showEpoch(epochK + 1);
$("esel").onchange = () => showEpoch(Number($("esel").value));
$("mtable").onclick = (ev) => {
  const tr = ev.target.closest("tr[data-k]");
  if (!tr) return;
  showEpoch(Number(tr.dataset.k));
  $("efig").scrollIntoView({ behavior: "smooth", block: "center" });
};
document.addEventListener("keydown", (ev) => {
  if ($("view-multi").classList.contains("hidden") || !state || !state.multi) return;
  if (/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName)) return;
  if (ev.key === "ArrowLeft") { showEpoch(epochK - 1); ev.preventDefault(); }
  if (ev.key === "ArrowRight") { showEpoch(epochK + 1); ev.preventDefault(); }
});
function renderMFiles() {
  $("mfiles-info").textContent = mFiles.length ? `${mFiles.length} files: ${mFiles.slice(0, 3).map((f) => f.name).join(", ")}${mFiles.length > 3 ? " …" : ""}`
    : "SDSS, DESI, 1D FITS or text, one file per epoch";
}
const mdrop = $("mdrop");
$("mbrowse").onclick = (e) => { e.preventDefault(); e.stopPropagation(); $("mfiles").click(); };
mdrop.onclick = () => $("mfiles").click();
$("mfiles").onchange = () => { mFiles = [...$("mfiles").files]; renderMFiles(); };
mdrop.ondragover = (e) => { e.preventDefault(); mdrop.classList.add("over"); };
mdrop.ondragleave = () => mdrop.classList.remove("over");
mdrop.ondrop = (e) => { e.preventDefault(); mdrop.classList.remove("over"); mFiles = [...e.dataTransfer.files]; renderMFiles(); };
$("btn-multi").onclick = async () => {
  const opts = {
    model: $("mmodel").value, broad_components: Number($("mncomp").value), outflow: $("moutflow").checked,
    shared: $("mshared").value, intercalibrate: $("minter").checked, n_jobs: Number($("mjobs").value),
    deredden: $("mdered").checked, ebv: $("mdered").checked && $("mebv").value !== "" ? Number($("mebv").value) : null,
    rest_frame: true, z: $("mz").value || null, pattern: $("mpattern").value.trim() || null,
  };
  if (!opts.pattern) {
    if (!mFiles.length) return showError("Drop several spectra or give a pattern such as /data/*.fits");
    const batch = "epochs-" + Date.now();
    opts.paths = [];
    for (const [i, f] of mFiles.entries()) {
      $("mmessage").textContent = `uploading ${i + 1}/${mFiles.length}: ${f.name}`;
      try {
        const r = await api(`/api/upload?${new URLSearchParams({ filename: f.name, batch })}`,
          { method: "POST", body: f, headers: { "Content-Type": "application/octet-stream" } });
        opts.paths.push(r.path);
      } catch (e) { return showError(`${f.name}: ${e.message}`); }
    }
  }
  post("/api/multiepoch", opts);
};

// ================================================================= docs
let docs = null;
async function openDocs(id) {
  if (!docs) {
    $("doc").innerHTML = "<p>loading …</p>";
    docs = await api("/api/docs");
    renderDocList();
  }
  const page = docs.find((p) => p.id === id) || docs[0];
  for (const a of document.querySelectorAll("#doclist a")) a.classList.toggle("active", a.dataset.id === page.id);
  const q = $("docsearch").value.trim();
  let html = `<h1>${esc(page.title)}</h1>` + page.html;
  if (q.length > 1) html = highlight(html, q);
  $("doc").innerHTML = html;
  if (location.hash !== `#docs/${page.id}`) history.replaceState(null, "", `#docs/${page.id}`);
  const first = $("doc").querySelector("mark");
  if (first) first.scrollIntoView({ block: "center" }); else window.scrollTo(0, 0);
}
function textOf(html) { const d = document.createElement("div"); d.innerHTML = html; return d.textContent.toLowerCase(); }
function renderDocList() {
  const q = $("docsearch").value.trim().toLowerCase();
  let sec = null, out = "";
  for (const p of docs) {
    if (q && !(p.title.toLowerCase().includes(q) || textOf(p.html).includes(q))) continue;
    if (p.section !== sec) { sec = p.section; out += `<div class="sec">${esc(sec)}</div>`; }
    out += `<a href="#docs/${p.id}" data-id="${p.id}">${esc(p.title)}</a>`;
  }
  $("doclist").innerHTML = out || '<p class="hint">nothing found</p>';
}
function highlight(html, q) {
  const re = new RegExp("(" + q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "gi");
  return html.replace(/>([^<]+)</g, (m, t) => ">" + t.replace(re, "<mark>$1</mark>") + "<");
}
$("doclist").onclick = (e) => { const a = e.target.closest("a[data-id]"); if (a) { e.preventDefault(); openDocs(a.dataset.id); } };
$("docsearch").oninput = () => {
  renderDocList();
  const first = $("doclist").querySelector("a[data-id]");
  if (first) openDocs(first.dataset.id);
};

// ============================================================== polling
function render() {
  renderStepper();
  renderCells();
  renderCards();
  if (currentView === "model") renderModelEditor(false);
  renderMulti();
  renderLines();
}
async function refresh() {
  try {
    setState(await api("/api/state"));
  } catch (e) {
    showError("The FANTASY server is not running (start it with: fantasy-app).");
    clearTimeout(pollTimer);
    pollTimer = setTimeout(refresh, 3000);
  }
}
function setState(s) {
  const newRecommendation = s.recommended && (!state || !state.recommended || state.recommended.label !== s.recommended.label);
  state = s;
  clockOffset = s.now - Date.now() / 1000;
  if (!$("preset").options.length) $("preset").innerHTML = s.models.map((m) => `<option>${esc(m)}</option>`).join("");
  if (newRecommendation) {
    $("preset").value = s.recommended.model; $("ncomp").value = s.recommended.broad_components;
    presetDefaults();
  }
  renderPresetOptions();
  renderHost();
  $("pyver").innerHTML = `<span class="dotok"></span>Python ${esc(s.python)}`;
  $("pyver").title = `FANTASY ${s.version}, Python ${s.python}`;
  render();
  clearTimeout(pollTimer);
  pollTimer = setTimeout(refresh, s.busy ? 400 : 2000);
}
setInterval(() => { if (state && state.busy) renderCells(); }, 200);

// ============================================================== actions
let clientError = null;
function showError(msg) {
  clientError = msg;
  for (const id of ["message", "mmessage"]) { $(id).textContent = msg; $(id).classList.add("error"); }
}
async function post(path, body, headers) {
  clientError = null;
  try {
    const isJSON = body !== undefined && !(body instanceof Blob);
    const opts = { method: "POST", headers: Object.assign(isJSON ? { "Content-Type": "application/json" } : {}, headers || {}) };
    if (body !== undefined) opts.body = isJSON ? JSON.stringify(body) : body;
    setState(await api(path, opts));
    return true;
  } catch (e) {
    showError(e.message);
    if (currentView === "model") alert(e.message);
    return false;
  }
}
function prepOptions() {
  return { deredden: $("dered").checked, ebv: $("dered").checked && $("ebv").value !== "" ? Number($("ebv").value) : null,
           rest_frame: $("restframe").checked, mask_absorption: $("absorb").checked };
}
function renderPrepHint() {
  const o = prepOptions();
  const parts = [];
  parts.push(o.deredden ? (o.ebv !== null ? `extinction corrected with E(B-V) = ${o.ebv}` :
    "E(B-V) looked up at the position (file, SFD maps or the IRSA dust service online)") : "Galactic extinction not corrected");
  parts.push(o.rest_frame ? "rest frame" : "observed frame (the models need the rest frame)");
  if (o.mask_absorption) parts.push("narrow absorption lines masked");
  $("prep-hint").textContent = parts.join(" · ");
  $("ebv").disabled = !o.deredden;
}
for (const id of ["dered", "ebv", "restframe", "absorb"]) $(id).addEventListener("input", renderPrepHint);
renderPrepHint();
function readJSON() {
  const z = $("z").value;
  const body = Object.assign({ path: $("path").value.trim() || null, targetid: $("targetid").value.trim() || null,
                               z: z || null }, prepOptions());
  if (!body.path && !body.targetid) return showError("Drop a file, give a path or a DESI TARGETID.");
  post("/api/read", body);
}
function readFile(file) {
  const o = prepOptions();
  const q = new URLSearchParams({ filename: file.name, deredden: o.deredden ? "1" : "0", rest_frame: o.rest_frame ? "1" : "0",
                                  mask_absorption: o.mask_absorption ? "1" : "0" });
  if ($("z").value) q.set("z", $("z").value);
  if (o.ebv !== null) q.set("ebv", o.ebv);
  $("message").textContent = `uploading ${file.name} …`;
  post(`/api/read?${q}`, file, { "Content-Type": "application/octet-stream" });
}
$("btn-read").onclick = readJSON;
// projects: save is a download link; open uploads the file and replaces the session
$("openproj").onclick = () => {
  if (state && state.cells.length > 1 && !confirm("Open a project? The current session is replaced (save it first if needed).")) return;
  $("projfile").click();
};
$("projfile").onchange = () => {
  const f = $("projfile").files[0];
  if (!f) return;
  lastFigure = -1; modelSig = null; view = null; mId = null; linesId = null;
  post("/api/project", f, { "Content-Type": "application/octet-stream" });
  $("projfile").value = "";
};
$("path").addEventListener("keydown", (e) => { if (e.key === "Enter") readJSON(); });
$("targetid").addEventListener("keydown", (e) => { if (e.key === "Enter") readJSON(); });
$("btn-classify").onclick = () => post("/api/classify");
function presetOptions() {
  return { model: $("preset").value, broad_components: Number($("ncomp").value), outflow: $("outflow").checked,
           coronal: $("opt-coronal").checked, oi6046: $("opt-oi").checked, red: $("opt-red").checked,
           host: $("host").value };
}
const TPL = { emiles: "emiles", bc03: "galaxev" };
function tplInfo(choice) {
  const name = TPL[choice];
  return name && state && state.templates ? state.templates.find((t) => t.name === name) : null;
}
function renderHost() {
  const choice = $("host").value, t = tplInfo(choice);
  const b = $("btn-tpl");
  if (t && !t.available) {
    b.classList.remove("hidden");
    b.querySelector("span").textContent = `Download now (${t.size_mb} MB)`;
    b.dataset.name = t.name;
    $("host-hint").textContent = `${choice.toUpperCase()} templates are downloaded automatically the first time they are used (${t.size_mb} MB, once)`;
  } else {
    b.classList.add("hidden");
    $("host-hint").textContent = t ? t.cite : (choice === "pca" ? "Yip et al. 2004 eigenspectra (can become negative)" : "");
  }
}
$("host").addEventListener("change", renderHost);
$("btn-tpl").onclick = () => post("/api/templates", { name: $("btn-tpl").dataset.name });
// which options the chosen preset supports (optical has all three, hbeta/halpha only coronal lines)
const PRESET_OPTS = { "optical": ["coronal", "oi6046", "red"], "optical + host": ["coronal", "oi6046", "red"],
                      "optical (physical)": ["coronal", "oi6046", "red"],
                      "hbeta": ["coronal"], "halpha": ["coronal"], "mgii": [], "civ": [], "seyfert2": ["coronal", "red"],
                      "liner": ["red"], "composite": ["red"], "starforming": ["red"], "passive / retired": ["red"],
                      "type2 (simple)": ["coronal"], "galaxy (simple)": [] };
function renderPresetOptions() {
  const ok = PRESET_OPTS[$("preset").value] || [];
  for (const [id, k] of [["opt-coronal", "coronal"], ["opt-oi", "oi6046"], ["opt-red", "red"]]) {
    $(id).disabled = !ok.includes(k);
    $(id).closest("label").style.opacity = ok.includes(k) ? 1 : 0.45;
  }
  $("opts-hint").textContent = ok.length < 3 ? `${$("preset").value}: ` +
    (ok.length ? "only coronal lines can be added to this model" : "no optional line groups") : "";
}
const RED_DEFAULT = ["seyfert2", "liner", "composite", "starforming", "passive / retired"];
// the preset's own defaults (outflow wing, broad components), sent by the server
function applyModelDefaults(model, outflowBox, ncompBox) {
  const d = (state && state.model_defaults && state.model_defaults[model]) || {};
  if (outflowBox) {
    outflowBox.disabled = d.outflow === undefined;
    if (d.outflow !== undefined) outflowBox.checked = !!d.outflow;
  }
  if (ncompBox) {
    ncompBox.disabled = d.broad_components === undefined;
    if (d.broad_components !== undefined) ncompBox.value = d.broad_components;
  }
}
function presetDefaults() {
  $("opt-red").checked = RED_DEFAULT.includes($("preset").value);
  applyModelDefaults($("preset").value, $("outflow"), null);
}
$("preset").addEventListener("change", () => { presetDefaults(); renderPresetOptions(); });
$("btn-preset").onclick = () => modelAction(Object.assign({ action: "preset" }, presetOptions()));
$("btn-fit").onclick = () => post("/api/fit", Object.assign({ errors: $("errors").value }, presetOptions()));
$("reset").onclick = () => { lastFigure = -1; modelSig = null; view = null; post("/api/reset"); };
const drop = $("drop");
$("browse").onclick = (e) => { e.preventDefault(); e.stopPropagation(); $("file").click(); };
drop.onclick = () => $("file").click();
drop.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") $("file").click(); };
$("file").onchange = () => { if ($("file").files[0]) readFile($("file").files[0]); };
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); if (e.dataTransfer.files[0]) readFile(e.dataTransfer.files[0]); };

api("/api/lines").then((ls) => {
  $("linelist").innerHTML = ls.map((l) => `<option value="${esc(l.name)}">${l.wave.toFixed(1)} Å</option>`).join("");
}).catch(() => {});

function route() {
  const h = location.hash.slice(1);
  if (h.startsWith("docs")) setView("docs", h.split("/")[1]);
  else if (h === "model") setView("model");
  else if (h === "multi") setView("multi");
  else setView("analysis");
}
window.addEventListener("hashchange", route);
refresh().then(route);

// figures are drawn at 2x (sharp on high-resolution screens) and shown at their natural size
for (const id of ["figimg", "mfig"]) $(id).addEventListener("load", (e) => { e.target.style.width = `${e.target.naturalWidth / 2}px`; });
