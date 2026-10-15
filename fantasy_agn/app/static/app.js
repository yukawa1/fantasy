// FANTASY app: talks to the local server (server.py) and draws the session.
"use strict";
const $ = (id) => document.getElementById(id);
const TOKEN = document.querySelector('meta[name="fantasy-token"]').content;
let state = null;
let clockOffset = 0;          // server time - browser time (s)
let lastFigure = -1;
let pollTimer = null;
const views = [];             // one view per code cell: {root, prompt, area, outputs, code, dirty}

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

// --------------------------------------------------------------- editor
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
  post("/api/exec", JSON.stringify(body), { "Content-Type": "application/json" });
}

const SPIN = "|/-\\";
function renderCells() {
  const now = Date.now() / 1000 + clockOffset;
  const box = $("cells");
  const cells = state.cells;
  // views for server cells (index >= 0); a new, not yet run cell has index null
  const serverViews = views.filter((v) => v.index !== null);
  while (serverViews.length < cells.length) {
    const v = makeView(serverViews.length);
    const firstNew = views.find((x) => x.index === null);
    box.insertBefore(v.root, firstNew ? firstNew.root : null);
    views.splice(serverViews.length, 0, v);
    serverViews.push(v);
  }
  while (serverViews.length > cells.length) {       // after "New analysis"
    const v = serverViews.pop();
    v.root.remove();
    views.splice(views.indexOf(v), 1);
  }
  cells.forEach((c, i) => {
    const v = serverViews[i];
    const running = c.running;
    v.prompt.textContent = running ? "In [*]:" : (c.n ? `In [${c.n}]:` : "In [ ]:");
    v.root.classList.toggle("running", running);
    v.area.readOnly = !!state.busy;
    if (c.code !== v.code && (!v.dirty || document.activeElement !== v.area)) {
      v.code = c.code;
      v.area.value = c.code;
      v.dirty = false;
      v.root.classList.remove("dirty");
      autosize(v.area);
    }
    let html = "";
    if (running) {
      const el = now - c.start;
      html += `<pre class="run">${SPIN[Math.floor(el * 6) % 4]} running … ${el.toFixed(0)} s</pre>`;
    }
    for (const o of c.out) html += `<pre class="${o.kind === "err" ? "err" : "out"}">${esc(o.text)}</pre>`;
    if (v.outputs.innerHTML !== html) v.outputs.innerHTML = html;
  });
  $("script").classList.toggle("hidden", cells.length < 2);
}

$("addcell").onclick = () => {
  const v = makeView(null);
  v.prompt.textContent = "In [ ]:";
  v.code = "";
  $("cells").appendChild(v.root);
  views.push(v);
  v.area.value = "# your code, e.g.\nr.lines";
  autosize(v.area);
  v.area.focus();
  v.area.select();
};

// ------------------------------------------------------------ rendering
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
  $("btn-fit").disabled = busy || !state.spectrum;
  $("addcell").disabled = busy;
  const cur = nextStep();
  for (const [card, step] of [["card-read", "read"], ["card-classify", "classify"], ["card-fit", "fit"]]) {
    const el = $(card);
    el.classList.toggle("done", done.includes(step));
    el.classList.toggle("cur", cur === step);
    el.classList.toggle("disabled", step !== "read" && !state.spectrum);
  }
  const badge = $("badge");
  if (state.recommended) {
    badge.textContent = state.recommended.label.replace("_", " ") + (state.recommended.uncertain ? " (uncertain)" : "");
    badge.classList.remove("hidden");
  } else badge.classList.add("hidden");
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

function render() {
  renderStepper();
  renderCells();
  renderCards();
}

// -------------------------------------------------------------- polling
async function refresh() {
  try {
    setState(await api("/api/state"));
  } catch (e) {
    $("message").textContent = "The FANTASY server is not running (start it with: fantasy-app).";
    $("message").classList.add("error");
    clearTimeout(pollTimer);
    pollTimer = setTimeout(refresh, 3000);
  }
}

function setState(s) {
  const newRecommendation = s.recommended && (!state || !state.recommended ||
    state.recommended.label !== s.recommended.label);
  state = s;
  clockOffset = s.now - Date.now() / 1000;
  if (!$("model").options.length) {
    $("model").innerHTML = s.models.map((m) => `<option>${esc(m)}</option>`).join("");
  }
  if (newRecommendation) {
    $("model").value = s.recommended.model;
    $("ncomp").value = s.recommended.broad_components;
  }
  $("pyver").textContent = `Python ${s.python} · FANTASY ${s.version}`;
  render();
  clearTimeout(pollTimer);
  pollTimer = setTimeout(refresh, s.busy ? 400 : 2000);
}

// The timer moves smoothly between polls.
setInterval(() => { if (state && state.busy) renderCells(); }, 200);

// -------------------------------------------------------------- actions
async function post(path, body, headers) {
  try {
    setState(await api(path, { method: "POST", body, headers }));
  } catch (e) {
    $("message").textContent = e.message;
    $("message").classList.add("error");
  }
}

function readJSON() {
  const z = $("z").value;
  const body = { path: $("path").value.trim() || null, targetid: $("targetid").value.trim() || null, z: z || null };
  if (!body.path && !body.targetid) {
    $("message").textContent = "Drop a file, give a path or a DESI TARGETID.";
    $("message").classList.add("error");
    return;
  }
  post("/api/read", JSON.stringify(body), { "Content-Type": "application/json" });
}

function readFile(file) {
  const z = $("z").value;
  const q = new URLSearchParams({ filename: file.name });
  if (z) q.set("z", z);
  $("message").textContent = `uploading ${file.name} …`;
  post(`/api/read?${q}`, file, { "Content-Type": "application/octet-stream" });
}

$("btn-read").onclick = readJSON;
$("path").addEventListener("keydown", (e) => { if (e.key === "Enter") readJSON(); });
$("targetid").addEventListener("keydown", (e) => { if (e.key === "Enter") readJSON(); });
$("btn-classify").onclick = () => post("/api/classify");
$("btn-fit").onclick = () => post("/api/fit", JSON.stringify({
  model: $("model").value, broad_components: Number($("ncomp").value),
  outflow: $("outflow").checked, errors: $("errors").value,
}), { "Content-Type": "application/json" });
$("reset").onclick = () => { lastFigure = -1; post("/api/reset"); };

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

refresh();
