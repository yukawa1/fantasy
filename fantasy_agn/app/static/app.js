// FANTASY app: talks to the local server (server.py) and draws the session.
"use strict";
const $ = (id) => document.getElementById(id);
const TOKEN = document.querySelector('meta[name="fantasy-token"]').content;
let state = null;
let clockOffset = 0;          // server time - browser time (s)
let lastFigure = -1;
let pollTimer = null;
const views = [];             // one view per code cell

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
  for (const t of document.querySelectorAll(".tab")) t.classList.toggle("active", t.dataset.view === name);
  for (const v of ["analysis", "model", "docs"]) $("view-" + v).classList.toggle("hidden", v !== name);
  $("main").classList.toggle("docs", name === "docs");
  if (name === "docs") openDocs(docId);
  if (name === "model" && state) renderModelEditor(true);
  const hash = name === "docs" && docId ? `#docs/${docId}` : `#${name}`;
  if (location.hash !== hash) history.replaceState(null, "", hash);
}
for (const t of document.querySelectorAll(".tab")) t.onclick = () => setView(t.dataset.view);
document.addEventListener("click", (e) => {
  const help = e.target.closest(".help");
  if (help) { e.preventDefault(); setView("docs", help.dataset.doc); }
  const go = e.target.closest("[data-goto]");
  if (go) { e.preventDefault(); setView(go.dataset.goto); }
});
$("open-editor").onclick = (e) => { e.preventDefault(); setView("model"); };

// =============================================================== editor
function autosize(area) {
  area.style.height = "auto";
  area.style.height = area.scrollHeight + 2 + "px";
}
function makeView(index) {
  const root = document.createElement("div");
  root.className = "cell";
  root.innerHTML = '<div class="prompt"></div><div><div class="editor"><textarea spellcheck="false" rows="1"></textarea>' +
    '<button class="runbtn" title="Run (Shift+Enter)">▶</button></div><div class="outputs"></div></div>';
  const v = { root, prompt: root.querySelector(".prompt"), area: root.querySelector("textarea"),
              outputs: root.querySelector(".outputs"), code: null, dirty: false, index };
  v.area.addEventListener("input", () => {
    v.dirty = v.area.value !== v.code;
    root.classList.toggle("dirty", v.dirty);
    autosize(v.area);
  });
  v.area.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.shiftKey) { e.preventDefault(); runView(v); }
    else if (e.key === "Tab") {
      e.preventDefault();
      const a = v.area, s = a.selectionStart;
      a.value = a.value.slice(0, s) + "    " + a.value.slice(a.selectionEnd);
      a.selectionStart = a.selectionEnd = s + 4;
      a.dispatchEvent(new Event("input"));
    }
  });
  root.querySelector(".runbtn").onclick = () => runView(v);
  return v;
}
function runView(v) {
  if (state && state.busy) return;
  const code = v.area.value;
  const body = v.index === null ? { code } : { code, index: v.index };
  if (v.index === null) { v.root.remove(); views.splice(views.indexOf(v), 1); }
  v.dirty = false;
  post("/api/exec", body);
}
const SPIN = "|/-\\";
function renderCells() {
  const now = Date.now() / 1000 + clockOffset;
  const box = $("cells");
  const cells = state.cells;
  const serverViews = views.filter((v) => v.index !== null);
  while (serverViews.length < cells.length) {
    const v = makeView(serverViews.length);
    const firstNew = views.find((x) => x.index === null);
    box.insertBefore(v.root, firstNew ? firstNew.root : null);
    views.splice(serverViews.length, 0, v);
    serverViews.push(v);
  }
  while (serverViews.length > cells.length) {
    const v = serverViews.pop();
    v.root.remove();
    views.splice(views.indexOf(v), 1);
  }
  cells.forEach((c, i) => {
    const v = serverViews[i];
    v.prompt.textContent = c.running ? "In [*]:" : (c.n ? `In [${c.n}]:` : "In [ ]:");
    v.root.classList.toggle("running", c.running);
    v.area.readOnly = !!state.busy;
    if (c.code !== v.code && (!v.dirty || document.activeElement !== v.area)) {
      v.code = c.code;
      v.area.value = c.code;
      v.dirty = false;
      v.root.classList.remove("dirty");
      autosize(v.area);
    }
    let html = "";
    if (c.running) {
      const el = now - c.start;
      html += `<pre class="run">${SPIN[Math.floor(el * 6) % 4]} running … ${el.toFixed(0)} s</pre>`;
    }
    for (const o of c.out) html += `<pre class="${o.kind === "err" ? "err" : "out"}">${esc(o.text)}</pre>`;
    if (v.outputs.innerHTML !== html) v.outputs.innerHTML = html;
  });
  $("script").classList.toggle("hidden", cells.length < 2);
  if (state.busy) {               // keep the running cell in view
    const run = box.querySelector(".cell.running");
    if (run && !run.dataset.scrolled) { run.scrollIntoView({ block: "nearest" }); run.dataset.scrolled = "1"; }
  }
}
$("addcell").onclick = () => {
  const v = makeView(null);
  v.prompt.textContent = "In [ ]:";
  v.code = "";
  $("cells").appendChild(v.root);
  views.push(v);
  v.area.value = state && state.model ? "model.params" : "# your code";
  autosize(v.area);
  v.area.focus();
  v.area.select();
};

// ============================================================ analysis
function nextStep() {
  for (const s of state.steps) if (!state.done.includes(s)) return s;
  return "results";
}
function renderStepper() {
  const cur = nextStep();
  $("stepper").innerHTML = state.steps.map((s) => {
    const cls = state.done.includes(s) ? "done" : (s === cur ? "cur" : "");
    return `<span class="step ${cls}">${state.done.includes(s) ? "✓ " : ""}${s}</span>`;
  }).join('<span class="arrow">→</span>');
}
function renderCards() {
  const busy = !!state.busy;
  const done = state.done;
  $("btn-read").disabled = busy;
  $("btn-classify").disabled = busy || !state.spectrum;
  $("btn-preset").disabled = busy;
  $("btn-fit").disabled = busy || !state.spectrum;
  $("addcell").disabled = busy;
  $("btn-add").disabled = busy;
  const cur = nextStep();
  for (const [card, step] of [["card-read", "read"], ["card-classify", "classify"], ["card-model", "model"], ["card-fit", "fit"]]) {
    const el = $(card);
    el.classList.toggle("done", done.includes(step));
    el.classList.toggle("cur", cur === step);
    el.classList.toggle("disabled", step !== "read" && step !== "model" && !state.spectrum);
  }
  const badge = $("badge");
  if (state.recommended) {
    badge.textContent = state.recommended.label.replace("_", " ") + (state.recommended.uncertain ? " (uncertain)" : "");
    badge.classList.remove("hidden");
  } else badge.classList.add("hidden");
  const m = state.model;
  $("model-summary").classList.toggle("hidden", !m);
  $("btn-preset").textContent = m ? "Replace model" : "Create model";
  if (m) {
    $("model-chips").innerHTML = m.components.map((c) =>
      `<span class="chip" title="${esc(c.type)}${c.lines.length ? ": " + esc(c.lines.join(", ")) : ""}">${esc(c.name)}</span>`).join("") +
      `<span class="hint">${m.n_free} free parameters</span>`;
  }
  $("fit-hint").textContent = m ? "fits the model of step 3" : "without a model: the recommended preset";
  const msg = $("message");
  msg.classList.toggle("error", !!state.error);
  msg.textContent = state.error ? state.error : (busy ? "running …" : state.info);
  $("card-results").classList.toggle("hidden", !state.has_result);
  $("results").innerHTML = state.results.map((r) => `<tr><td>${esc(r.label)}</td><td>${esc(r.value)}</td></tr>`).join("");
  $("warnings").innerHTML = state.warnings.map((w) => `<div>⚠ ${esc(w)}</div>`).join("");
  if (state.figure_version !== lastFigure && state.figure_version > 0) {
    lastFigure = state.figure_version;
    $("figimg").src = withToken(`/api/figure.png?v=${lastFigure}`);
    $("figure").classList.remove("hidden");
  }
  if (state.figure_version === 0) $("figure").classList.add("hidden");
}

// ========================================================= model editor
let modelSig = null;
const pending = {};               // parameter name -> edited row
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
  const row = input.closest("tr");
  const name = row.dataset.p;
  const p = state.model.params.find((x) => x.name === name);
  const k = input.dataset.k;
  const e2 = pending[name] || {};
  e2[k] = input.type === "checkbox" ? input.checked : input.value;
  // drop edits that restore the current value
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
    return {
      name,
      value: "value" in e ? e.value : null, min: "min" in e ? e.min : null, max: "max" in e ? e.max : null,
      fixed: "fixed" in e ? e.fixed : p.fixed,
      tie: "tie" in e ? (e.tie || null) : p.tie,
      factor: "factor" in e ? e.factor : p.factor,
    };
  });
  for (const k of Object.keys(pending)) delete pending[k];
  modelAction({ action: "params", changes });
};

// add-component form
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
    modelAction({ action: "add", component: comp }).then((ok) => {
      if (ok) { newLines.length = 0; renderNewLines(); $("cname").value = ""; }
    });
  } catch (e) { showError(e.message); }
};
async function modelAction(body) { return post("/api/model", body); }

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
  $("doc").scrollTop = 0;
  if (location.hash !== `#docs/${page.id}`) history.replaceState(null, "", `#docs/${page.id}`);
  const first = $("doc").querySelector("mark");
  if (first) first.scrollIntoView({ block: "center" });
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
  const newRecommendation = s.recommended && (!state || !state.recommended ||
    state.recommended.label !== s.recommended.label);
  state = s;
  clockOffset = s.now - Date.now() / 1000;
  if (!$("preset").options.length) $("preset").innerHTML = s.models.map((m) => `<option>${esc(m)}</option>`).join("");
  if (newRecommendation) {
    $("preset").value = s.recommended.model;
    $("ncomp").value = s.recommended.broad_components;
  }
  $("pyver").textContent = `Python ${s.python} · FANTASY ${s.version}`;
  render();
  clearTimeout(pollTimer);
  pollTimer = setTimeout(refresh, s.busy ? 400 : 2000);
}
setInterval(() => { if (state && state.busy) renderCells(); }, 200);

// ============================================================== actions
function showError(msg) {
  $("message").textContent = msg;
  $("message").classList.add("error");
}
async function post(path, body, headers) {
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
function readJSON() {
  const z = $("z").value;
  const body = { path: $("path").value.trim() || null, targetid: $("targetid").value.trim() || null, z: z || null };
  if (!body.path && !body.targetid) return showError("Drop a file, give a path or a DESI TARGETID.");
  post("/api/read", body);
}
function readFile(file) {
  const q = new URLSearchParams({ filename: file.name });
  if ($("z").value) q.set("z", $("z").value);
  $("message").textContent = `uploading ${file.name} …`;
  post(`/api/read?${q}`, file, { "Content-Type": "application/octet-stream" });
}
$("btn-read").onclick = readJSON;
$("path").addEventListener("keydown", (e) => { if (e.key === "Enter") readJSON(); });
$("targetid").addEventListener("keydown", (e) => { if (e.key === "Enter") readJSON(); });
$("btn-classify").onclick = () => post("/api/classify");
$("btn-preset").onclick = () => modelAction({ action: "preset", model: $("preset").value,
  broad_components: Number($("ncomp").value), outflow: $("outflow").checked });
$("btn-fit").onclick = () => post("/api/fit", { errors: $("errors").value, model: $("preset").value,
  broad_components: Number($("ncomp").value), outflow: $("outflow").checked });
$("reset").onclick = () => { lastFigure = -1; modelSig = null; post("/api/reset"); };

const drop = $("drop");
$("browse").onclick = (e) => { e.preventDefault(); $("file").click(); };
drop.onclick = (e) => { if (e.target === drop) $("file").click(); };
drop.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") $("file").click(); };
$("file").onchange = () => { if ($("file").files[0]) readFile($("file").files[0]); };
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => {
  e.preventDefault(); drop.classList.remove("over");
  if (e.dataTransfer.files[0]) readFile(e.dataTransfer.files[0]);
};

// line list for the add-component form
api("/api/lines").then((ls) => {
  $("linelist").innerHTML = ls.map((l) => `<option value="${esc(l.name)}">${l.wave.toFixed(1)} Å</option>`).join("");
}).catch(() => {});

function route() {
  const h = location.hash.slice(1);
  if (h.startsWith("docs")) setView("docs", h.split("/")[1]);
  else if (h === "model") setView("model");
  else if (h === "analysis") setView("analysis");
}
window.addEventListener("hashchange", route);
refresh().then(route);
