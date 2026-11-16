/* FANTASY: Model step, the model gallery (candidate models fitted on the spectrum and ranked by BIC)
   and the model builder (a full-screen workspace where every component and parameter can be edited).
   The server turns a builder state into FANTASY code; "Use this model" runs that code as a session cell. */
"use strict";
window.MB = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const svgEl = (t, a, p) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); if (p) p.appendChild(e); return e; };
  const clone = (o) => JSON.parse(JSON.stringify(o));
  const call = (path, body) => api(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const GCOL = { continuum: "#52514e", host: "#008300", feii: "#e34948", narrow: "#1baf7a", outflow: "#eda100", broad: "#2a78d6", absorption: "#4a3aa7", other: "#9aa1ad" };
  const GNAME = { continuum: "continuum", host: "host", feii: "Fe II", narrow: "narrow", outflow: "outflow", broad: "broad", absorption: "absorption, dust", other: "other" };
  const QS = [["broad Hb FWHM", "FWHM Hβ", "km/s", 0], ["broad Ha FWHM", "FWHM Hα", "km/s", 0], ["broad MgII FWHM", "FWHM Mg II", "km/s", 0],
    ["broad CIV FWHM", "FWHM C IV", "km/s", 0], ["log M_BH", "log M_BH", "", 2], ["R_FeII", "R_FeII", "", 2], ["broad Ha/Hb", "Hα/Hβ broad", "", 2]];

  // ================================================================ gallery
  const G = { data: null, sel: null, tab: "comp", used: null, timer: null, specName: null, tableOpen: false, waitFor: null };
  let cands = [], base = null, best = null, rec = null;

  function rank() {
    cands = (G.data ? G.data.cands : []).filter((c) => !c.error && c.bic != null);
    base = cands.find((c) => c.key === "base") || null;
    const ref = base || cands[0];
    cands.forEach((c) => (c.vsBase = ref ? c.bic - ref.bic : 0));
    best = cands.length ? cands.reduce((a, c) => (c.bic < a.bic ? c : a)) : null;
    // recommended: the lowest BIC among the models that put no new parameter at a limit (a limit the preset
    // itself hits is not held against them); a change must beat the preset by ΔBIC < -10 (strong evidence)
    const lim0 = new Set(ref ? ref.limits : []);
    const clean = cands.filter((c) => c.limits.every((p) => lim0.has(p)));
    const top = clean.length ? clean.reduce((a, c) => (c.bic < a.bic ? c : a)) : best;
    rec = base && top && top !== base && top.bic - base.bic > -10 && clean.includes(base) ? base : top;
  }
  const byKey = (k) => cands.find((c) => c.key === k);
  function inUse(c) {                      // the session model is this candidate: its cell is the last model choice
    if (!S || !S.model || !S.cells) return false;
    const last = [...S.cells].reverse().find((x) => /^model = fa\./.test(x.code));
    return !!last && last.code.replace(/\nmodel\s*$/, "").trim() === c.code.trim();
  }
  async function load() {
    try { G.data = await call("/api/wf/gallery"); } catch (e) { G.data = null; }
    rank();
    if (G.waitFor && byKey(G.waitFor)) { G.sel = G.waitFor; G.picked = true; G.tab = "vs"; G.waitFor = null; }
    if (G.waitFor && G.data && !G.data.pending.length) G.waitFor = null;
    if (!G.picked || !byKey(G.sel)) G.sel = rec ? rec.key : null;      // follow the recommendation until you pick one
    renderGallery();
    clearTimeout(G.timer);
    if (G.data && (G.data.running || G.data.pending.length) && page === "model") G.timer = setTimeout(load, 2000);
  }
  function render() {
    if (!$("galcard")) return;
    const on = prepared() && ["single", "multi"].includes(mode());
    $("galcard").classList.toggle("hidden", !on);
    if (!on) return;
    const name = prepared().id ?? prepared().name;
    if (G.specName !== name) { G.specName = name; G.data = null; G.sel = null; G.picked = false; G.used = null; load(); return; }
    renderGallery();
  }
  async function runGallery() {
    try { G.data = await call("/api/mb/gallery", {}); rank(); renderGallery(); clearTimeout(G.timer); G.timer = setTimeout(load, 1500); }
    catch (e) { showError(e.message); }
  }

  function renderGallery() {
    const body = $("gx-body"), D = G.data;
    newButton();
    if (!D || (!D.cands.length && !D.running && !(D.pending || []).length)) {
      body.innerHTML = `<div class="gx-empty"><p>Fit variants of the recommended model on this spectrum: one broad component more and fewer, and each option
        of the preset (host, outflow wing, coronal lines, Fe II, …) switched on its own, then the best combination. They are ranked by BIC = χ² + k ln n,
        which charges every extra parameter: <b>ΔBIC &lt; −10</b> is strong evidence for a change.</p>
        <button class="btn primary" id="gx-run">Compare models</button> <span class="muted small">about a minute, in the background</span></div>`;
      $("gx-run").onclick = runGallery;
      return;
    }
    const pend = D.pending || [], total = D.total || 0, fitted = D.cands.filter((c) => !c.mine).length;
    const sel = byKey(G.sel);
    body.innerHTML = `${D.running ? `<div class="gx-prog"><span class="spin"></span> fitting the candidates · ${fitted} of ${total} done <span class="gx-bar"><i style="width:${total ? (100 * fitted / total).toFixed(0) : 0}%"></i></span></div>` : ""}
      ${D.error ? `<p class="callout">${esc(D.error)}</p>` : ""}
      <div class="gx-picks" id="gx-picks"></div>
      <div class="gx-matter hidden" id="gx-matter"></div>
      <div class="gx-detail" id="gx-detail"></div>
      <details class="gx-all" id="gx-alld" ${G.tableOpen ? "open" : ""}><summary>All ${cands.length} candidates <span class="muted small">· each change on its own against the preset · ΔBIC &lt; −10 is strong evidence for it</span>
        ${!D.running ? `<button class="btn small" id="gx-rerun" title="Fit the candidates again">Fit again</button>` : ""}</summary><table class="tbl clickable" id="gx-all"></table></details>
      ${D.cands.filter((c) => c.error).map((c) => `<p class="note small">${esc(c.title)}: ${esc(c.error)}</p>`).join("")}`;
    picks(pend); if (sel) { matter(); detail(sel); } table();
    $("gx-alld").ontoggle = (e) => (G.tableOpen = e.target.open);
    if ($("gx-rerun")) $("gx-rerun").onclick = (e) => { e.preventDefault(); runGallery(); };
  }
  function newButton() {
    const d = DRAFT && !DRAFT.fitKey ? DRAFT : null;
    $("gx-new").textContent = d ? "Continue your model →" : "+ Your own model";
    $("gx-new").onclick = () => (d ? resume() : open(rec ? fromCand(rec) : S.model ? { model: true } : {}, rec ? rec.title : S.model ? "the current model" : "empty"));
    $("gx-draft").innerHTML = d ? `unsaved edits · <a id="gx-draftnew">start a new one</a>` : "";
    if (d) $("gx-draftnew").onclick = () => { DRAFT = null; newButton(); };
  }
  const fromCand = (c) => (c.state ? { state: c.state } : { preset: c.preset[0], kw: c.preset[1] });

  function mini(svg, c, W, H, big) {
    svg.innerHTML = "";
    const x0 = c.w0, x1 = c.w1, rh = big ? 54 : 18, top = 4, L = big ? 44 : 2, R = big ? 10 : 2, ph = H - rh - top - (big ? 22 : 4);
    let hi = 0; c.flux.forEach((v) => { if (v != null && v > hi) hi = v; }); hi *= 1.04;
    const x = (w) => L + (w - x0) / (x1 - x0) * (W - L - R), y = (v) => top + (1 - v / hi) * ph;
    const path = (a) => { let d = "", pen = false; a.forEach((v, i) => { if (v == null || c.wave[i] == null) { pen = false; return; } d += `${pen ? "L" : "M"}${x(c.wave[i]).toFixed(1)},${y(Math.max(0, v)).toFixed(1)}`; pen = true; }); return d; };
    if (big) for (let w = Math.ceil(x0 / 250) * 250; w <= x1; w += 250) svgEl("text", { x: x(w), y: H - 4, "text-anchor": "middle" }, svg).textContent = w;
    svgEl("path", { d: path(c.flux), fill: "none", stroke: "#b9b8b3", "stroke-width": big ? 1 : 0.8 }, svg);
    if (big) Object.entries(c.groups).forEach(([g, a]) => svgEl("path", { d: path(a), fill: "none", stroke: GCOL[g] || GCOL.other, "stroke-width": 1.2, "stroke-dasharray": g === "continuum" ? "5 4" : "" }, svg));
    else if (c.groups.broad) svgEl("path", { d: path(c.groups.broad), fill: "none", stroke: GCOL.broad, "stroke-width": 1, opacity: 0.8 }, svg);
    svgEl("path", { d: path(c.model), fill: "none", stroke: "#0b0b0b", "stroke-width": big ? 1.1 : 0.9 }, svg);
    const r0 = H - rh - (big ? 18 : 0), ry = (v) => r0 + rh / 2 - Math.max(-5, Math.min(5, v)) / 5 * (rh / 2 - 1);
    svgEl("rect", { x: L, y: r0, width: W - L - R, height: rh, fill: "#f8f9fb" }, svg);
    svgEl("line", { x1: L, x2: W - R, y1: ry(0), y2: ry(0), stroke: "#cfd3d9" }, svg);
    c.resid.forEach((v, i) => { if (v == null || c.wave[i] == null) return; const a = Math.abs(v);
      svgEl("line", { x1: x(c.wave[i]), x2: x(c.wave[i]), y1: ry(0), y2: ry(v), stroke: a > 3 ? "#d4472a" : "#9aa1ad", "stroke-width": big ? 1.4 : 1, opacity: a > 3 ? 0.9 : 0.6 }, svg); });
    if (big) [[0, "0σ"], [4, "+4"], [-4, "−4"]].forEach(([v, t]) => (svgEl("text", { x: L - 4, y: ry(v) + 3, "text-anchor": "end" }, svg).textContent = t));
  }
  const dB = (c) => (c === base ? "—" : `${c.vsBase > 0 ? "+" : ""}${Math.round(c.vsBase)}`);
  function picks(pend) {
    const P = [];
    if (rec) P.push([rec, "recommended", "ok", rec === base ? "no change is strongly favoured (ΔBIC < −10) without new limits" : "lowest BIC without new parameters at a limit"]);
    if (base && base !== rec) P.push([base, "preset", "muted", G.data.preset ? `fa.presets.${G.data.preset}()` : ""]);
    if (best && best !== rec) P.push([best, "lowest BIC", "warn", `${best.limits.length} parameter${best.limits.length > 1 ? "s" : ""} at a limit`]);
    const mine = cands.filter((c) => c.mine && !P.some((p) => p[0] === c));
    if (mine.length) P.push([mine[mine.length - 1], "your model", "mine", mine.length > 1 ? `${mine.length} of yours in the table` : ""]);
    const mp = pend.filter((t) => t.startsWith("#"));
    $("gx-picks").innerHTML = P.slice(0, 4).map(([c, why, cls, note]) => `<div class="gx-mc${c.key === G.sel ? " sel" : ""}" data-k="${esc(c.key)}">
        <div class="gx-why"><span class="badge ${cls === "mine" ? "gx-mine" : cls}">${why}</span>${inUse(c) ? '<span class="badge ok">✓ in use</span>' : ""}</div><h4>${esc(c.title)}</h4>
        <svg data-mini="${esc(c.key)}"></svg>
        <div class="gx-nums"><span>χ²ν <b>${c.rchi2.toFixed(2)}</b></span><span>ΔBIC <b>${dB(c)}</b></span><span><b>${c.nfree}</b> par</span></div>
        ${note ? `<div class="note small">${esc(note)}</div>` : ""}</div>`).join("") +
      mp.map((t) => `<div class="gx-mc busy"><div class="gx-why"><span class="badge gx-mine">your model</span></div><h4>${esc(t)}</h4><div class="note small"><span class="spin"></span> fitting on this spectrum…</div></div>`).join("");
    $("gx-picks").querySelectorAll(".gx-mc[data-k]").forEach((e) => (e.onclick = () => { G.sel = e.dataset.k; G.picked = true; renderGallery(); }));
    $("gx-picks").querySelectorAll("[data-mini]").forEach((s) => { const W = s.clientWidth || 240; s.setAttribute("viewBox", `0 0 ${W} 70`); mini(s, byKey(s.dataset.mini), W, 70, false); });
  }
  function matter() {
    if (!rec || cands.length < 3) return;
    const use = cands.slice().sort((a, b) => a.bic - b.bic).slice(0, 5);
    const rows = QS.map(([k, lab, u, d]) => { const vals = use.map((c) => c.results[k]).filter((v) => v && isFinite(v[0])), r = rec.results[k];
      if (vals.length < 2 || !r || !r[1]) return null;
      const lo = Math.min(...vals.map((v) => v[0])), hi = Math.max(...vals.map((v) => v[0]));
      return { lab, u, d, r, lo, hi, rat: (hi - lo) / 2 / r[1] }; }).filter(Boolean);
    const top = rows.filter((q) => q.rat > 1).sort((a, b) => b.rat - a.rat);
    if (!top.length) return;
    const m = $("gx-matter"); m.classList.remove("hidden");
    m.innerHTML = `<span>The model choice matters here: across the five best models, ${top.slice(0, 2).map((q) => `${q.lab} moves by <b>${q.rat.toFixed(1)}×</b> its fit error`).join(" and ")}.</span>
      <a id="gx-mmore">${G.matterOpen ? "hide" : "see all"}</a>
      ${G.matterOpen ? `<table class="tbl"><thead><tr><th>Quantity</th><th class="num">recommended</th><th class="num">range over the five best</th><th class="num">half range / error</th></tr></thead><tbody>${rows.map((q) => `<tr><td>${q.lab} <span class="muted small">${q.u}</span></td><td class="num">${q.r[0].toFixed(q.d)} ± ${q.r[1].toFixed(q.d)}</td><td class="num">${q.lo.toFixed(q.d)}–${q.hi.toFixed(q.d)}</td><td class="num">${q.rat.toFixed(1)}×</td></tr>`).join("")}</tbody></table>` : ""}`;
    $("gx-mmore").onclick = () => { G.matterOpen = !G.matterOpen; renderGallery(); };
  }
  function detail(c) {
    const why = c === rec ? ["ok", "recommended"] : c === best ? ["warn", `lowest BIC, ${c.limits.length} at a limit`] : c === base ? ["muted", "preset"] : c.mine ? ["gx-mine", "your model"] : null;
    $("gx-detail").innerHTML = `<div class="gx-dhead"><h4>${esc(c.title)}</h4>${why ? `<span class="badge ${why[0]}">${why[1]}</span>` : ""}
        <span class="muted small mono">χ²ν ${c.rchi2.toFixed(3)} · BIC ${Math.round(c.bic)} · ${c.nfree} free parameters · ${c.time} s</span><span class="grow"></span>
        <button class="btn small" id="gx-cust">${c.mine ? "Keep editing…" : "Customize…"}</button>
        ${inUse(c) ? '<span class="badge ok">✓ in use</span>' : '<button class="btn primary small" id="gx-use">Use this model</button>'}</div>
      <div class="gx-leg">${Object.keys(c.groups).map((g) => `<span><i style="background:${GCOL[g] || GCOL.other}"></i>${GNAME[g] || g}</span>`).join("")}<span><i style="background:#0b0b0b"></i>model</span><span><i style="background:#b9b8b3"></i>data</span><span class="muted">· residuals in σ</span></div>
      <svg class="gx-big" id="gx-big"></svg>
      <div class="tabs gx-tabs" id="gx-tabs"><button data-t="comp">Components</button><button data-t="vs">Against the preset</button><button data-t="code">Code</button></div>
      <div id="gx-tab"></div>`;
    const s = $("gx-big"), W = s.clientWidth || 900; s.setAttribute("viewBox", `0 0 ${W} 250`); mini(s, c, W, 250, true);
    $("gx-tabs").querySelectorAll("button").forEach((b) => { b.classList.toggle("on", b.dataset.t === G.tab); b.onclick = () => { G.tab = b.dataset.t; renderGallery(); }; });
    $("gx-cust").onclick = () => (c.mine && DRAFT && DRAFT.fitKey === c.key ? resume() : open(fromCand(c), c.title));
    if ($("gx-use")) $("gx-use").onclick = () => use(c);
    const ref = base || c, T = $("gx-tab");
    if (G.tab === "comp") {
      const removed = ref.components.filter((q) => !c.components.some((p) => p.name === q.name));
      T.innerHTML = `<table class="tbl"><thead><tr><th>Component</th><th>lines</th><th class="num">free parameters</th></tr></thead><tbody>${c.components.map((p) => { const nw = !ref.components.some((q) => q.name === p.name);
        return `<tr><td><span class="gx-comp"><i style="background:${GCOL[p.group] || GCOL.other}"></i>${esc(p.name)}</span> ${nw ? '<span class="badge ok">new</span>' : ""}</td><td class="small muted">${esc(p.lines.slice(0, 6).join(", "))}${p.lines.length > 6 ? ` +${p.lines.length - 6}` : ""}</td><td class="num">${p.nfree}</td></tr>`; }).join("")}
        ${removed.map((q) => `<tr><td colspan="3" class="muted"><s>${esc(q.name)}</s> removed</td></tr>`).join("")}</tbody></table>
        ${c.limits.length ? `<p class="note" style="color:var(--warn)">At a limit (set by the limit, not by the data): ${esc(c.limits.join(", "))}</p>` : ""}`;
    } else if (G.tab === "vs") {
      const row = (lab, a, b) => `<tr><td>${lab}</td><td class="num">${a}</td><td class="num">${b}</td></tr>`;
      T.innerHTML = `<table class="tbl"><thead><tr><th></th><th class="num">${esc(ref.title)}</th><th class="num">${esc(c.title)}</th></tr></thead><tbody>
        ${row("χ²ν", ref.rchi2.toFixed(3), c.rchi2.toFixed(3))}${row("BIC", Math.round(ref.bic), `${Math.round(c.bic)} <span class="muted small">(${dB(c)})</span>`)}
        ${row("free parameters", ref.nfree, c.nfree)}${row("largest residual structure", ref.struct.toFixed(1) + "σ", c.struct.toFixed(1) + "σ")}
        ${QS.map(([k, lab, u, d]) => ref.results[k] && c.results[k] ? row(`${lab} <span class="muted small">${u}</span>`, ref.results[k][0].toFixed(d), `${c.results[k][0].toFixed(d)}${c.results[k][1] ? ` <span class="muted small">± ${c.results[k][1].toFixed(d)}</span>` : ""}`) : "").join("")}
        ${row("parameters at a limit", ref.limits.length, c.limits.length)}</tbody></table>`;
    } else {
      T.innerHTML = `<pre class="gx-code">${esc(c.code)}\nmodel</pre><p class="note"><b>Use this model</b> runs this as a cell of the session; the component editor below stays available for changes by hand.</p>`;
    }
  }
  function table() {
    const rows = cands.slice().sort((a, b) => a.bic - b.bic);
    const lim = Math.max(...rows.map((c) => Math.abs(c.vsBase)), 10), sc = (v) => Math.sign(v) * Math.log10(1 + Math.abs(v)) / Math.log10(1 + lim);
    $("gx-all").innerHTML = `<thead><tr><th>#</th><th>Model</th><th>change</th><th>ΔBIC vs preset</th><th class="num"></th><th class="num">χ²ν</th><th class="num">free</th><th class="num">at a limit</th></tr></thead><tbody>` +
      rows.map((c, k) => `<tr class="${c.key === G.sel ? "sel" : ""}" data-k="${esc(c.key)}"><td class="muted small">${k + 1}</td><td>${esc(c.title)} ${c === rec ? '<span class="badge ok">recommended</span>' : c.mine ? '<span class="badge gx-mine">yours</span>' : ""}</td>
        <td><code class="small muted">${esc(c.change || "")}</code></td><td><svg data-bar="${esc(c.key)}" viewBox="0 0 160 14" preserveAspectRatio="none" class="gx-barsvg"></svg></td>
        <td class="num" style="color:${c.vsBase < -10 ? "#1f7a4d" : c.vsBase > 10 ? "#b5371d" : "var(--muted)"}">${dB(c)}</td>
        <td class="num">${c.rchi2.toFixed(2)}</td><td class="num">${c.nfree}</td><td class="num" style="color:${c.limits.length ? "var(--warn)" : "var(--muted)"}">${c.limits.length || "—"}</td></tr>`).join("") + "</tbody>";
    $("gx-all").querySelectorAll("[data-bar]").forEach((s) => { const c = byKey(s.dataset.bar), v = sc(c.vsBase);
      svgEl("line", { x1: 80, x2: 80, y1: 0, y2: 14, stroke: "#cfd3d9" }, s);
      if (c !== base) svgEl("rect", { x: v < 0 ? 80 + v * 78 : 80, y: 2, width: Math.abs(v) * 78, height: 10, rx: 2, fill: c.vsBase < -10 ? "#1f7a4d" : c.vsBase > 10 ? "#d4472a" : "#b8bec7", opacity: 0.8 }, s); });
    $("gx-all").querySelectorAll("tbody tr").forEach((r) => (r.onclick = () => { G.sel = r.dataset.k; G.picked = true; renderGallery(); $("gx-detail").scrollIntoView({ behavior: "smooth", block: "nearest" }); }));
  }
  async function use(c) {
    const ok = await post("/api/model", c.state ? { action: "recipe", state: c.state } : { action: "presetkw", name: c.preset[0], kw: c.preset[1] });
    if (ok) { G.used = c.key; toast(`Model in use: ${c.title}`); renderGallery(); }
  }

  // ================================================================ workspace (model builder)
  let TYPES = null, LD = { lines: [], sets: {}, range: [0, 1] }, UID = 0, B = null, DRAFT = null, prevTimer = null;
  const GROUPS = [["continuum", "Continuum"], ["host", "Host galaxy"], ["feii", "Fe II and templates"], ["narrow", "Narrow lines"],
    ["outflow", "Outflow / wings"], ["broad", "Broad lines"], ["absorption", "Absorption and dust"], ["other", "Other"]];
  const TGROUP = { PowerLaw: "continuum", BrokenPowerLaw: "continuum", Blackbody: "continuum", Polynomial: "continuum", BalmerContinuum: "continuum",
    StellarPopulation: "host", HostGalaxy: "host", FeII: "feii", HighOrderBalmer: "feii", Absorption: "absorption", Extinction: "absorption" };
  const PROFILE_FIELDS = { h3: "gauss-hermite", h4: "gauss-hermite", fwhm_l: "voigt", span: "stratified", gamma: "stratified" };
  const HIDE = new Set(["name", "lines", "ratios", "line_ratios", "max_ratios"]);
  const FIELD_LABEL = { fwhm: "FWHM (km/s)", shift: "velocity shift (km/s)", tie_kinematics: "one shift and width for all lines", flux_max: "largest flux",
    index: "slope", index_blue: "slope, blue side", index_red: "slope, red side", break_wave: "break wavelength (Å)", ebv: "E(B−V)", r_v: "R_V",
    sigma: "σ (km/s)", velocity: "velocity (km/s)", amp_max: "largest amplitude", max_wavelength: "reddest line (Å)", n_components: "eigenspectra",
    fwhm_l: "Lorentzian FWHM", h3: "h3 (skewness)", h4: "h4 (kurtosis)", span: "span (dex)", gamma: "γ emissivity", fix_shape: "fixed shape", fix_fwhm: "fixed FWHM" };
  const ROLEDEF = { narrow: { role: "narrow", fwhm: [500, 50, 2000], shift: [0, -1000, 1000] }, broad: { role: "broad", fwhm: [3000, 1000, 12000], shift: [0, -3000, 3000] },
    outflow: { role: "outflow", fwhm: [1000, 300, 3000], shift: [-300, -2000, 500] } };
  const fmtN = (v) => (v === "inf" ? "∞" : v === "-inf" ? "−∞" : v == null ? "" : typeof v === "number" ? +v.toPrecision(6) + "" : String(v));
  const parseN = (s) => { s = String(s).trim().replace("−", "-").replace("∞", "inf"); if (s === "") return null; if (s === "inf" || s === "+inf") return "inf"; if (s === "-inf") return "-inf";
    const v = Number(s); if (!isFinite(v)) throw new Error(`not a number: ${s}`); return v; };
  const isTriple = (v) => Array.isArray(v) && v.length === 3 && v.every((x) => typeof x === "number" || x === "inf" || x === "-inf");
  const lineName = (e) => (Array.isArray(e) ? e[0] : e);
  const wl = (e) => (Array.isArray(e) ? e[1] : LD.lines.find((l) => l.name === e)?.wave ?? 0);
  const R = () => $("mb");

  function showWB(on) { R().classList.toggle("hidden", !on); document.body.classList.toggle("mb-open", on); if (on) R().scrollTop = 0; }
  const edited = (b) => b && b.st && (b.undo.length || b.title.trim());
  function resume() { B = DRAFT; DRAFT = null; B.add = false; showWB(true); draw(); newButton(); }
  function close() { if (edited(B) || (B && B.fitKey)) DRAFT = B; B = null; showWB(false); if ($("gx-new")) newButton(); }
  async function open(source, label) {
    if (B && edited(B) && !B.fitKey && !confirm("Start again from this model? Your edits so far are lost.")) return draw();
    B = { label, st: null, sel: null, info: null, prev: null, undo: [], redo: [], error: null, warn: [], busyPrev: false, seq: 0, pseq: 0, add: false, code: "", title: "" };
    showWB(true); DRAFT = null; if ($("gx-new")) newButton();
    R().innerHTML = `<div class="mb-in"><div class="mb-top"><div class="mb-head"><button class="mb-back" id="mb-close">← Back to the Model step</button></div></div>
      <p class="note" style="margin-top:20px"><span class="spin"></span> taking “${esc(label)}” apart into components…</p></div>`;
    $("mb-close").onclick = close;
    try {
      if (!TYPES) { const [t, l] = await Promise.all([call("/api/wf/mbtypes"), call("/api/wf/mblines")]); TYPES = Object.fromEntries(t.map((x) => [x.type, x])); LD = l; }
      const st = source.state ? clone(source.state) : (await call("/api/mb/decompose", { source })).state;
      st.comps.forEach((c) => (c.uid = ++UID));
      B.st = st; B.start = clone(st); B.sel = st.comps.find((c) => c.type === "Lines" && c.kw.role === "broad")?.uid || st.comps[0]?.uid;
      await sync(true);
      B.startNfree = B.info ? B.info.nfree : 0;
    } catch (e) { B.error = e.message; }
    draw();
  }
  const comp = (uid) => B.st.comps.find((c) => c.uid === uid);
  const ready = (c) => c.on !== false && !(c.type === "Lines" && !(c.kw.lines || []).length) && !(c.type === "Absorption" && !(c.kw.troughs || []).length);
  const payload = () => ({ ...B.st, comps: B.st.comps.filter(ready).map(({ type, kw, on }) => ({ type, kw, on })) });
  function edit(fn) {
    B.undo.push(clone(B.st)); if (B.undo.length > 80) B.undo.shift(); B.redo = [];
    try { fn(); } catch (e) { B.st = B.undo.pop(); B.error = e.message; return draw(); }
    B.error = null; sync();
  }
  function undo(dir) {
    const [from, to] = dir < 0 ? [B.undo, B.redo] : [B.redo, B.undo];
    if (!from.length) return; to.push(clone(B.st)); B.st = from.pop(); if (!comp(B.sel)) B.sel = B.st.comps[0]?.uid; sync();
  }
  async function sync(first) {
    const s = ++B.seq, b = B;
    let r; try { r = await call("/api/mb/build", { state: payload() }); } catch (e) { r = { error: e.message }; }
    if (b !== B || s !== B.seq) return;
    if (r.error) B.error = r.error; else { B.info = r; B.code = r.code; B.warn = r.warnings || []; B.error = null; }
    if (!first) draw();
    if (!r.error) schedulePreview();
  }
  function schedulePreview() {
    clearTimeout(prevTimer);
    prevTimer = setTimeout(async () => {
      if (!B) return;
      const s = ++B.pseq, b = B; B.busyPrev = true; plot();
      let r; try { r = await call("/api/mb/preview", { state: payload() }); } catch (e) { r = { error: e.message }; }
      if (b !== B || s !== B.pseq) return;
      B.busyPrev = false; if (!r.error) B.prev = r; plot();
    }, 450);
  }
  const infoOf = (c) => B.info?.components.find((q) => q.name === c.kw.name);
  const groupOf = (c) => infoOf(c)?.group || TGROUP[c.type] || (c.kw.absorption ? "absorption" : c.kw.role || "narrow");
  function uniqueName(b) { const have = new Set(B.st.comps.map((c) => c.kw.name)); if (!have.has(b)) return b; let k = 2; while (have.has(`${b}${k}`)) k++; return `${b}${k}`; }
  function rename(c, nn) {
    nn = nn.trim(); const old = c.kw.name; if (!nn || nn === old) return;
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(nn)) throw new Error("names: letters, digits and _ only");
    if (B.st.comps.some((q) => q !== c && q.kw.name === nn)) throw new Error(`there is already a component called ${nn}`);
    c.kw.name = nn; const P = {};
    for (const [k, o] of Object.entries(B.st.params)) { const k2 = k.startsWith(old + ".") ? nn + k.slice(old.length) : k;
      if (o.tie && o.tie[0].startsWith(old + ".")) o.tie[0] = nn + o.tie[0].slice(old.length); P[k2] = o; }
    B.st.params = P;
  }
  function dropOverrides(c, key) {
    for (const k of Object.keys(B.st.params)) if (k.startsWith(c.kw.name + ".") && k.endsWith("." + key)) {
      const o = B.st.params[k]; delete o.value; delete o.min; delete o.max; if (!Object.keys(o).length) delete B.st.params[k]; }
  }
  function newComp(type, kw = {}) {
    const T = TYPES[type], d = {};
    T.fields.forEach((f) => { if (f.default !== null && f.default !== undefined) d[f.key] = clone(f.default); });
    const b = kw.name || { Lines: kw.role || "lines", PowerLaw: "continuum", BrokenPowerLaw: "continuum", Blackbody: "blackbody" }[type] || d.name || type.toLowerCase();
    const c = { uid: ++UID, type, on: true, kw: { ...d, ...kw } };
    if (type === "Lines") { c.kw.lines = c.kw.lines || []; Object.assign(c.kw, ROLEDEF[c.kw.role || "narrow"], kw); }
    if (type === "Absorption") c.kw.troughs = kw.troughs || [];
    c.kw.name = uniqueName(b);
    return c;
  }
  function lineHit(v) { v = String(v).trim(); return LD.lines.find((l) => l.name === v) || LD.lines.find((l) => l.name.toLowerCase() === v.toLowerCase()) ||
    (isFinite(+v) && v !== "" ? LD.lines.slice().sort((a, b) => Math.abs(a.wave - v) - Math.abs(b.wave - v)).find((l) => Math.abs(l.wave - v) < 4) : null); }

  function draw() {
    if (!B) return showWB(false);
    if (!B.st) { R().innerHTML = `<div class="mb-in"><div class="mb-top"><div class="mb-head"><button class="mb-back" id="mb-close">← Back to the Model step</button></div></div><p class="callout">${esc(B.error || "")}</p></div>`; $("mb-close").onclick = close; return; }
    const c = comp(B.sel), nf = B.info?.nfree ?? 0, dn = nf - (B.startNfree ?? nf);
    const focusId = document.activeElement?.id, keepOpen = $("mb-code")?.open, sy = R().scrollTop;
    const srcs = [["model", "the current model"], ["empty", "empty: one power law"]].filter(([k]) => k !== "model" || S.model).concat(cands.map((q) => [q.key, q.title]));
    R().innerHTML = `<div class="mb-in"><div class="mb-top"><div class="mb-head"><button class="mb-back" id="mb-close" title="Esc · your edits are kept">← Back to the Model step</button><span class="mb-vsep"></span><h3>Your own model</h3>
        <span class="muted small">from</span><select id="mb-from" class="small"><option value="">${esc(B.label)}</option>${srcs.map(([k, t]) => `<option value="${esc(k)}">${esc(t)}</option>`).join("")}</select>
        <button class="btn icon" id="mb-undo" title="Undo (⌘Z)" ${B.undo.length ? "" : "disabled"}>↶</button><button class="btn icon" id="mb-redo" title="Redo (⇧⌘Z)" ${B.redo.length ? "" : "disabled"}>↷</button>
        <span class="grow"></span><span class="muted small">fit range</span><input class="mb-num" id="mb-wr0" value="${fmtN(B.st.wave_range?.[0])}"><span class="muted small">–</span><input class="mb-num" id="mb-wr1" value="${fmtN(B.st.wave_range?.[1])}"><span class="muted small">Å</span></div></div>
      <div class="mb-plot" id="mb-plot"><div class="mb-cap" id="mb-cap"></div><svg id="mb-svg"></svg><div class="mb-pick hidden" id="mb-pick"></div></div>
      <div class="mb-grid"><div class="mb-stack" id="mb-stack">${stackHTML()}</div><div class="mb-insp" id="mb-insp">${c ? inspHTML(c) : '<p class="muted">Select a component, or add one.</p>'}</div></div>
      ${B.warn.length ? `<p class="note" style="color:var(--warn)">${B.warn.map(esc).join("<br>")}</p>` : ""}
      <details class="mb-codebox" id="mb-code" ${keepOpen ? "open" : ""}><summary>Code <span class="muted small">· exactly what is fitted · Use this model runs it as a session cell</span></summary><pre class="gx-code">${esc(B.code)}</pre></details>
      <div class="mb-bot"><div class="mb-foot"><span class="mb-stat"><b>${B.st.comps.filter(ready).length}</b> components · <b>${nf}</b> free parameters ${dn ? `<span class="${dn > 0 ? "up" : "dn"}">${dn > 0 ? "+" : ""}${dn} vs start</span>` : ""}</span>
        <span class="grow"></span><input type="text" id="mb-title" placeholder="name it (optional)" value="${esc(B.title)}">
        <button class="btn" id="mb-use" ${B.error ? "disabled" : ""} title="Make it the session model, without fitting it here">Use this model</button>
        <button class="btn primary" id="mb-fit" ${B.error ? "disabled" : ""} title="Fit it on this spectrum and rank it among the candidates">Fit and rank it</button></div>
        ${B.error ? `<p class="callout" style="margin:0 0 8px">${esc(B.error)}</p>` : ""}</div></div>`;
    R().scrollTop = sy;
    wire(); plot();
    if (focusId && $(focusId)) $(focusId).focus();
  }
  function stackHTML() {
    const by = {}; B.st.comps.forEach((c) => (by[groupOf(c)] = by[groupOf(c)] || []).push(c));
    return `<div class="mb-addw"><button class="btn mb-addb" id="mb-add">${B.add ? "× Close the list" : "+ Add component"}</button>${B.add ? paletteHTML() : ""}</div>` +
      GROUPS.filter(([g]) => by[g]).map(([g, lab]) => `<div class="mb-sg"><div class="mb-sgh"><i style="background:${GCOL[g] || GCOL.other}"></i>${lab}</div>
      ${by[g].map((c) => { const inf = infoOf(c), off = c.on === false, empty = !ready(c) && !off;
        return `<div class="mb-row${c.uid === B.sel ? " sel" : ""}${off ? " off" : ""}" data-u="${c.uid}">
          <input type="checkbox" class="mb-tg" data-on="${c.uid}" ${off ? "" : "checked"} title="${off ? "include it" : "leave it out (keeps the settings)"}">
          <span class="mb-nm">${esc(c.kw.name)}</span><span class="mb-ty">${c.type}${c.type === "Lines" && c.kw.profile && c.kw.profile !== "gaussian" ? " · " + c.kw.profile : ""}</span>
          <span class="mb-sum">${empty ? '<span style="color:var(--warn)">add lines</span>' : c.type === "Lines" ? esc((c.kw.lines || []).map(lineName).slice(0, 4).join(" ")) + ((c.kw.lines || []).length > 4 ? ` +${c.kw.lines.length - 4}` : "") : ""}</span>
          <span class="mb-np" title="free parameters">${inf && !off ? inf.nfree : ""}</span>
          <button class="mb-x" data-dup="${c.uid}" title="duplicate">⧉</button><button class="mb-x" data-del="${c.uid}" title="remove">✕</button></div>`; }).join("")}</div>`).join("");
  }
  function paletteHTML() {
    const grp = [["Lines", [["Lines", "narrow", "Narrow lines"], ["Lines", "broad", "Broad lines"], ["Lines", "outflow", "Outflow / wing"]]],
      ["Continuum", [["PowerLaw"], ["BrokenPowerLaw"], ["Blackbody"], ["Polynomial"], ["BalmerContinuum"]]],
      ["Templates and host", [["FeII"], ["HighOrderBalmer"], ["StellarPopulation"], ["HostGalaxy"]]], ["Absorption and dust", [["Absorption"], ["Extinction"]]]];
    const doc = (t) => esc((TYPES[t]?.doc || "").split(/(?<=\.)\s/)[0].slice(0, 110));
    return `<div class="mb-pal">${grp.map(([h, items]) => `<div class="mb-palh">${h}</div>${items.map(([t, role, lab]) => `<button class="mb-pi" data-new="${t}" data-role="${role || ""}"><b>${lab || t}</b><span>${role ? `role='${role}' · ${ROLEDEF[role].fwhm[1]}–${ROLEDEF[role].fwhm[2]} km/s` : doc(t)}</span></button>`).join("")}`).join("")}
      <div class="mb-palh">A line set as a new component</div><div class="mb-chips">${Object.entries(LD.sets).map(([k, s]) => `<button class="mb-chip" data-set="${k}" title="${esc(s.lines.join(", "))}">${esc(s.label)} <span class="muted">${s.lines.length}</span></button>`).join("")}</div></div>`;
  }
  const seg = (key, opts, cur) => `<div class="mb-seg">${opts.map((o) => `<button class="${o === cur ? "on" : ""}" data-seg="${key}" data-v="${o}">${o}</button>`).join("")}</div>`;
  function inspHTML(c) {
    const T = TYPES[c.type], inf = infoOf(c), kw = c.kw;
    const fields = T.fields.filter((f) => !HIDE.has(f.key) && !(PROFILE_FIELDS[f.key] && PROFILE_FIELDS[f.key] !== (kw.profile || "gaussian")) && !(c.type === "Lines" && ["role", "profile"].includes(f.key)));
    const others = B.st.comps.filter((q) => q !== c && q.on !== false && infoOf(q)?.params.some((p) => p.name === q.kw.name + ".fwhm"));
    const tied = B.st.params[`${kw.name}.shift`]?.tie?.[0]?.split(".")[0] || inf?.params.find((p) => p.name === kw.name + ".shift")?.tie?.[0].split(".")[0];
    let h = `<div class="mb-ih"><input class="mb-iname" id="mb-iname" value="${esc(kw.name)}" title="rename"><span class="mb-tyb">fa.${c.type}</span>${inf ? `<span class="muted small">${inf.nfree} free</span>` : ""}</div>
      <div class="note small">${esc((T.doc || "").split(/(?<=\.)\s/).slice(0, 2).join(" "))}</div>`;
    if (c.type === "Lines") {
      h += `<span class="mb-lbl">Role</span>${seg("role", ["narrow", "broad", "outflow"], kw.role)}
        <span class="mb-lbl">Profile</span>${seg("profile", ["gaussian", "lorentzian", "voigt", "gauss-hermite", "stratified"], kw.profile || "gaussian")}
        <span class="mb-lbl">Lines <span class="mb-lc">· or click the spectrum above</span></span>
        <div class="mb-lns">${(kw.lines || []).map((e, j) => { const n = lineName(e), w = Array.isArray(e) ? e[1] : LD.lines.find((l) => l.name === n)?.wave;
          return `<span class="mb-ln" data-hl="${esc(n)}">${esc(n)}${w ? `<small>${Math.round(w)}</small>` : ""}<button data-rml="${j}" title="remove">×</button></span>`; }).join("")}
          <span class="mb-sugw"><input type="text" id="mb-addln" placeholder="+ line, ion or Å" autocomplete="off" spellcheck="false"></span></div>
        ${ratioHTML(c)}`;
    }
    if (others.length && inf?.params.some((p) => p.name === kw.name + ".fwhm"))
      h += `<span class="mb-lbl">Kinematics</span><div class="mb-frow"><span>same shift and width as</span><select id="mb-tiekin" class="small"><option value="">— its own —</option>${others.map((q) => `<option ${tied === q.kw.name ? "selected" : ""}>${esc(q.kw.name)}</option>`).join("")}</select></div>`;
    h += `<span class="mb-lbl">Settings</span><div>${fields.map((f) => fieldHTML(f, kw[f.key])).join("")}</div>`;
    if (inf) h += paramsHTML(c, inf);
    return h;
  }
  function fieldHTML(f, v) {
    const lab = `<span class="mb-fl" title="${f.key}">${FIELD_LABEL[f.key] || f.key.replace(/_/g, " ")}</span>`;
    const def = f.default, val = v === undefined ? def : v;
    if (f.choices) return `<div class="mb-frow">${lab}<select class="small" data-f="${f.key}">${f.default === null ? `<option value="">auto</option>` : ""}${f.choices.map((o) => `<option ${o === val ? "selected" : ""}>${o}</option>`).join("")}</select></div>`;
    if (typeof def === "boolean" || typeof val === "boolean") return `<div class="mb-frow">${lab}<label class="mb-sw"><input type="checkbox" data-f="${f.key}" data-bool="1" ${val ? "checked" : ""}><span></span></label></div>`;
    if (isTriple(val)) { const t = val, p = [t[0], t[1], t[2]].every((x) => typeof x === "number") && t[2] !== t[1] ? (t[0] - t[1]) / (t[2] - t[1]) : null;
      return `<div class="mb-frow">${lab}<span class="mb-tri"><input class="mb-num" data-f="${f.key}" data-k="0" value="${fmtN(t[0])}" title="start"><span class="muted small">in</span><input class="mb-num" data-f="${f.key}" data-k="1" value="${fmtN(t[1])}" title="min"><span class="muted small">–</span><input class="mb-num" data-f="${f.key}" data-k="2" value="${fmtN(t[2])}" title="max"></span>${p == null ? "" : `<span class="mb-rb"><i style="left:${Math.max(0, Math.min(1, p)) * 100}%"></i></span>`}</div>`; }
    if (typeof val === "number" || val === "inf" || val === "-inf") return `<div class="mb-frow">${lab}<input class="mb-num" data-f="${f.key}" data-n="1" value="${fmtN(val)}"></div>`;
    return `<div class="mb-frow">${lab}<input type="text" class="mb-js" data-f="${f.key}" data-json="1" value="${esc(val == null ? "" : JSON.stringify(val))}" placeholder="${f.key === "troughs" ? "[[1549, 2000], [1240, 1500]]  (Å, FWHM km/s)" : "default"}"></div>`;
  }
  function ratioHTML(c) {
    const kw = c.kw, names = (kw.lines || []).map(lineName);
    if (names.length < 2) return "";
    const fx = Object.entries(kw.ratios || {}), rg = Object.entries(kw.line_ratios || {});
    const opt = (cur) => names.map((n) => `<option ${n === cur ? "selected" : ""}>${esc(n)}</option>`).join("");
    return `<span class="mb-lbl">Flux ratios <span class="mb-lc">· fixed (atomic physics) or bounded</span></span>
      ${fx.map(([a, [b, r]]) => `<div class="mb-frow mb-rat"><select data-rt="fx" data-a="${esc(a)}" data-w="a">${opt(a)}</select><span>=</span><input class="mb-num" data-rt="fx" data-a="${esc(a)}" data-w="r" value="${fmtN(r)}"><span>×</span><select data-rt="fx" data-a="${esc(a)}" data-w="b">${opt(b)}</select><button class="mb-x" data-rmr="fx" data-a="${esc(a)}">✕</button></div>`).join("")}
      ${rg.map(([a, [b, r]]) => { const t = Array.isArray(r) ? r : [r, r, r]; return `<div class="mb-frow mb-rat"><select data-rt="rg" data-a="${esc(a)}" data-w="a">${opt(a)}</select><span>/</span><select data-rt="rg" data-a="${esc(a)}" data-w="b">${opt(b)}</select><span>=</span><input class="mb-num" data-rt="rg" data-a="${esc(a)}" data-w="0" value="${fmtN(t[0])}"><span class="muted small">in</span><input class="mb-num" data-rt="rg" data-a="${esc(a)}" data-w="1" value="${fmtN(t[1])}"><span class="muted small">–</span><input class="mb-num" data-rt="rg" data-a="${esc(a)}" data-w="2" value="${fmtN(t[2])}"><button class="mb-x" data-rmr="rg" data-a="${esc(a)}">✕</button></div>`; }).join("")}
      <div class="mb-frow"><button class="btn small" id="mb-addfx">+ fixed ratio</button><button class="btn small" id="mb-addrg">+ bounded ratio</button></div>`;
  }
  function paramsHTML(c, inf) {
    return `<details class="mb-pt" ${B.ptOpen ? "open" : ""} id="mb-ptd"><summary>Parameters <span class="muted small">· ${inf.params.length} · start, limits, fix, tie to any other parameter</span></summary>
      <table><thead><tr><th>parameter</th><th class="num">start</th><th class="num">min</th><th class="num">max</th><th>fix</th><th>tie to</th><th></th></tr></thead><tbody>
      ${inf.params.map((p) => { const loc = p.name.slice(c.kw.name.length + 1), o = B.st.params[p.name], dis = p.tie ? "disabled" : "";
        return `<tr class="${p.tie ? "tied" : p.fixed ? "fixd" : ""}"><td class="mb-pn" title="${esc(p.description || p.name)}">${esc(loc)} ${o ? '<span class="mb-ed" title="edited here">●</span>' : ""}</td>
          <td><input class="mb-num" data-p="${esc(p.name)}" data-k="value" value="${fmtN(p.value)}" ${dis}></td>
          <td><input class="mb-num" data-p="${esc(p.name)}" data-k="min" value="${fmtN(p.min)}" ${dis}></td>
          <td><input class="mb-num" data-p="${esc(p.name)}" data-k="max" value="${fmtN(p.max)}" ${dis}></td>
          <td><input type="checkbox" data-p="${esc(p.name)}" data-k="fixed" ${p.fixed ? "checked" : ""} ${dis}></td>
          <td><span class="mb-sugw"><input type="text" class="mb-tie" autocomplete="off" spellcheck="false" data-lin="${p.linear ? 1 : 0}" data-p="${esc(p.name)}" data-k="tie" value="${esc(p.tie ? p.tie[0] : "")}" placeholder="—"></span>${p.tie ? `<input class="mb-num f" data-p="${esc(p.name)}" data-k="factor" value="${fmtN(p.tie[1])}" title="factor">` : ""}</td>
          <td>${o ? `<button class="mb-x" data-rst="${esc(p.name)}" title="back to the component's own setting">↺</button>` : ""}</td></tr>`; }).join("")}
      </tbody></table></details>`;
  }

  function wire() {
    const q = (s) => R().querySelectorAll(s), c = comp(B.sel);
    $("mb-close").onclick = close;
    $("mb-from").onchange = (e) => { const k = e.target.value; if (!k) return;
      if (k === "model") open({ model: true }, "the current model"); else if (k === "empty") open({}, "empty"); else { const cd = byKey(k); open(fromCand(cd), cd.title); } };
    $("mb-undo").onclick = () => undo(-1); $("mb-redo").onclick = () => undo(1);
    ["mb-wr0", "mb-wr1"].forEach((id, k) => ($(id).onchange = (e) => edit(() => { const v = parseN(e.target.value); B.st.wave_range = B.st.wave_range || LD.range.slice(); B.st.wave_range[k] = v; })));
    $("mb-title").oninput = (e) => (B.title = e.target.value);
    $("mb-fit").onclick = fitMine;
    $("mb-use").onclick = useMine;
    $("mb-add").onclick = () => { B.add = !B.add; draw(); };
    q("[data-new]").forEach((b) => (b.onclick = () => edit(() => { const n = newComp(b.dataset.new, b.dataset.role ? { role: b.dataset.role, name: b.dataset.role + "_extra" } : {}); B.st.comps.push(n); B.sel = n.uid; B.add = false; })));
    q("[data-set]").forEach((b) => (b.onclick = () => edit(() => { const s = LD.sets[b.dataset.set], br = /broad|feii_IZw1/.test(b.dataset.set);
      const n = newComp("Lines", { role: br ? "broad" : "narrow", name: b.dataset.set.toLowerCase(), lines: s.lines.slice() }); B.st.comps.push(n); B.sel = n.uid; B.add = false; })));
    q(".mb-row[data-u]").forEach((r) => (r.onclick = (e) => { if (e.target.closest("button,input")) return; B.sel = +r.dataset.u; draw(); }));
    q("[data-on]").forEach((i) => (i.onchange = () => edit(() => (comp(+i.dataset.on).on = i.checked))));
    q("[data-del]").forEach((b) => (b.onclick = () => edit(() => { const k = B.st.comps.findIndex((x) => x.uid === +b.dataset.del); B.st.comps.splice(k, 1); if (B.sel === +b.dataset.del) B.sel = B.st.comps[Math.max(0, k - 1)]?.uid; })));
    q("[data-dup]").forEach((b) => (b.onclick = () => edit(() => { const o = comp(+b.dataset.dup), n = { ...clone(o), uid: ++UID }; n.kw.name = uniqueName(o.kw.name.replace(/\d+$/, "") || o.kw.name);
      B.st.comps.splice(B.st.comps.indexOf(o) + 1, 0, n); B.sel = n.uid;
      const pre = o.kw.name + ".";           // the copy keeps the original's parameter edits and its ties inside the component
      for (const [k, v] of Object.entries(clone(B.st.params))) if (k.startsWith(pre)) {
        if (v.tie && v.tie[0].startsWith(pre)) v.tie[0] = n.kw.name + "." + v.tie[0].slice(pre.length);
        B.st.params[n.kw.name + "." + k.slice(pre.length)] = v; } })));
    if (!c) return;
    $("mb-iname").onchange = (e) => edit(() => rename(c, e.target.value));
    q("[data-seg]").forEach((b) => (b.onclick = () => edit(() => { const k = b.dataset.seg, v = b.dataset.v; c.kw[k] = v;
      if (k === "role" && ROLEDEF[v]) { c.kw.fwhm = ROLEDEF[v].fwhm.slice(); c.kw.shift = ROLEDEF[v].shift.slice(); dropOverrides(c, "fwhm"); dropOverrides(c, "shift"); } })));
    q("[data-f]").forEach((i) => (i.onchange = () => edit(() => { const k = i.dataset.f;
      if (i.dataset.bool) c.kw[k] = i.checked;
      else if (i.dataset.k !== undefined) { const t = (c.kw[k] || TYPES[c.type].fields.find((f) => f.key === k).default).slice(); t[+i.dataset.k] = parseN(i.value); c.kw[k] = t; }
      else if (i.dataset.n) c.kw[k] = parseN(i.value);
      else if (i.dataset.json) { if (i.value.trim() === "") delete c.kw[k]; else c.kw[k] = JSON.parse(i.value); }
      else if (i.tagName === "SELECT") { if (i.value === "") delete c.kw[k]; else c.kw[k] = i.value; }
      dropOverrides(c, k); })));
    q("[data-rml]").forEach((b) => (b.onclick = () => edit(() => removeLine(c, +b.dataset.rml))));
    if ($("mb-addln")) suggest($("mb-addln"), (t) => lineItems(c, t), (v) => edit(() => addLine(c, v)));
    q("input.mb-tie").forEach((i) => suggest(i, (t) => tieItems(i.dataset.p, +i.dataset.lin, t), (v) => { i.value = v; i.dispatchEvent(new Event("change")); }, true));
    q("[data-hl]").forEach((s) => { s.onmouseenter = () => plot(s.dataset.hl); s.onmouseleave = () => plot(); });
    if ($("mb-tiekin")) $("mb-tiekin").onchange = (e) => edit(() => { const to = e.target.value, inf = infoOf(c);
      inf.params.filter((p) => /\.(shift|fwhm)$/.test(p.name) && p.name.split(".").length === 2).forEach((p) => {
        const k = p.name.split(".")[1], o = (B.st.params[p.name] = B.st.params[p.name] || {});
        if (to) o.tie = [`${to}.${k}`, 1]; else { if (p.tie) o.tie = null; else delete o.tie; if (!Object.keys(o).length) delete B.st.params[p.name]; } }); });
    q("[data-rt]").forEach((i) => (i.onchange = () => edit(() => { const key = i.dataset.rt === "fx" ? "ratios" : "line_ratios", a = i.dataset.a, d = { ...(c.kw[key] || {}) }, [b, r] = d[a], w = i.dataset.w;
      if (w === "a") { delete d[a]; d[i.value] = [b, r]; } else if (w === "b") d[a] = [i.value, r]; else if (w === "r") d[a] = [b, parseN(i.value)];
      else { const t = Array.isArray(r) ? r.slice() : [r, r, r]; t[+w] = parseN(i.value); d[a] = [b, t]; }
      c.kw[key] = d; })));
    q("[data-rmr]").forEach((b) => (b.onclick = () => edit(() => { const key = b.dataset.rmr === "fx" ? "ratios" : "line_ratios"; const d = { ...c.kw[key] }; delete d[b.dataset.a]; c.kw[key] = Object.keys(d).length ? d : null; })));
    const addRatio = (key, val) => edit(() => { const names = c.kw.lines.map(lineName), used = new Set([...Object.keys(c.kw.ratios || {}), ...Object.keys(c.kw.line_ratios || {})]);
      const a = names.find((n, j) => j > 0 && !used.has(n)) || names.find((n) => !used.has(n)); if (!a) throw new Error("every line already has a ratio");
      const b = names.find((n) => n !== a); c.kw[key] = { ...(c.kw[key] || {}), [a]: [b, val] }; });
    if ($("mb-addfx")) $("mb-addfx").onclick = () => addRatio("ratios", 0.333);
    if ($("mb-addrg")) $("mb-addrg").onclick = () => addRatio("line_ratios", [0.3, 0.1, 1]);
    if ($("mb-ptd")) $("mb-ptd").ontoggle = (e) => (B.ptOpen = e.target.open);
    q("[data-p]").forEach((i) => (i.onchange = () => edit(() => { const n = i.dataset.p, k = i.dataset.k, o = (B.st.params[n] = B.st.params[n] || {});
      if (k === "fixed") o.fixed = i.checked;
      else if (k === "tie") { const v = i.value.trim(); if (v) { if (!B.info.components.some((x) => x.params.some((p) => p.name === v))) throw new Error(`no parameter ${v}`); o.tie = [v, 1]; } else o.tie = null; }
      else if (k === "factor") o.tie = [o.tie?.[0] || infoOf(c).params.find((p) => p.name === n).tie[0], parseN(i.value)];
      else o[k] = parseN(i.value); })));
    q("[data-rst]").forEach((b) => (b.onclick = () => edit(() => delete B.st.params[b.dataset.rst])));
  }
  function addLine(c, v) {
    v = String(v).trim(); if (!v) return;
    const m = v.match(/^([A-Za-z0-9_\[\]\-+]+)\s*[@:]\s*([\d.]+)$/);          // your own line: name@wavelength
    const hit = m ? null : lineHit(v);
    if (!m && !hit) throw new Error(`unknown line ${v}: pick one from the list, or give name@wavelength for your own`);
    const entry = m ? [m[1], +m[2]] : hit.name;
    if (c.kw.lines.some((e) => lineName(e) === lineName(entry))) throw new Error(`${lineName(entry)} is already in ${c.kw.name}`);
    c.kw.lines = [...c.kw.lines, entry].sort((a, b) => wl(a) - wl(b));
  }
  function removeLine(c, j) {
    const n = lineName(c.kw.lines[j]); c.kw.lines = c.kw.lines.filter((_, k) => k !== j);
    for (const key of ["ratios", "line_ratios"]) if (c.kw[key]) { const d = Object.fromEntries(Object.entries(c.kw[key]).filter(([a, [b]]) => a !== n && b !== n)); c.kw[key] = Object.keys(d).length ? d : null; }
  }

  // ---------------------------------------------------------------- the spectrum
  function plot(hl) {
    const s = $("mb-svg"); if (!s || !B) return;
    const p = B.prev, W = s.clientWidth || 1100, H = 230, L = 44, Rm = 10, top = 22, ph = H - top - 22;
    s.setAttribute("viewBox", `0 0 ${W} ${H}`); s.innerHTML = "";
    const cap = $("mb-cap");
    if (!p) { cap.innerHTML = `<span class="spin"></span> initial guess…`; return; }
    cap.innerHTML = `${B.busyPrev ? '<span class="spin"></span> updating · ' : ""}<b>initial guess</b>, not a fit · updates as you edit · <span class="muted">click the spectrum to add a line there</span>`;
    const x0 = p.x0, x1 = p.x1; let hi = 0; p.flux.forEach((v, i) => { if (v != null && p.wave[i] >= p.w0 && p.wave[i] <= p.w1 && v > hi) hi = v; }); hi *= 1.08;
    const x = (w) => L + (w - x0) / (x1 - x0) * (W - L - Rm), y = (v) => top + (1 - v / hi) * ph;
    const path = (a) => { let d = "", pen = false; a.forEach((v, i) => { if (v == null || p.wave[i] == null) { pen = false; return; } d += `${pen ? "L" : "M"}${x(p.wave[i]).toFixed(1)},${y(Math.max(-0.02 * hi, Math.min(v, hi * 1.2))).toFixed(1)}`; pen = true; }); return d; };
    for (let w = Math.ceil(x0 / 500) * 500; w <= x1; w += 500) svgEl("text", { x: x(w), y: H - 6, "text-anchor": "middle" }, s).textContent = w;
    [[x0, p.w0], [p.w1, x1]].forEach(([a, b]) => b > a && svgEl("rect", { x: x(a), y: top, width: x(b) - x(a), height: ph, fill: "#eef0f3", opacity: 0.7 }, s));
    svgEl("path", { d: path(p.flux), fill: "none", stroke: "#b9b8b3", "stroke-width": 1 }, s);
    const sel = comp(B.sel);
    B.st.comps.filter((c) => p.comps[c.kw.name]).forEach((c) => { const on = sel === c;
      svgEl("path", { d: path(p.comps[c.kw.name]), fill: "none", stroke: GCOL[groupOf(c)] || GCOL.other, "stroke-width": on ? 2.2 : 1, opacity: sel && !on ? 0.45 : 0.95, "stroke-dasharray": groupOf(c) === "continuum" && !on ? "5 4" : "" }, s); });
    svgEl("path", { d: path(p.model), fill: "none", stroke: "#0b0b0b", "stroke-width": 1 }, s);
    B.st.comps.filter((c) => c.type === "Lines" && c.on !== false).forEach((c) => { const on = c === sel, col = GCOL[groupOf(c)] || GCOL.other;
      (c.kw.lines || []).forEach((e) => { const w = wl(e); if (!w || w < x0 || w > x1) return; const n = lineName(e), big = on || n === hl;
        svgEl("line", { x1: x(w), x2: x(w), y1: top - (big ? 10 : 4), y2: big ? top + ph : top, stroke: col, "stroke-width": n === hl ? 2 : 1, opacity: big ? (n === hl ? 0.9 : 0.35) : 0.6, "stroke-dasharray": big ? "2 3" : "" }, s);
        if (on && n.length <= 9) { const t = svgEl("text", { x: x(w) + 2, y: top - 12 }, s); t.textContent = n; t.setAttribute("style", `fill:${col};font-size:9px`); } }); });
    const hit = svgEl("rect", { x: L, y: 0, width: W - L - Rm, height: H - 18, fill: "transparent", style: "cursor:crosshair" }, s);
    const guide = svgEl("line", { y1: top, y2: top + ph, stroke: "#4a3aa7", opacity: 0 }, s);
    const at = (e) => { const bx = s.getBoundingClientRect(); return [(e.clientX - bx.left) * W / bx.width, bx]; };
    hit.onmousemove = (e) => { const [X] = at(e); guide.setAttribute("x1", X); guide.setAttribute("x2", X); guide.setAttribute("opacity", 0.5); };
    hit.onmouseleave = () => guide.setAttribute("opacity", 0);
    hit.onclick = (e) => { const [X, bx] = at(e); showPick(x0 + (X - L) / (W - L - Rm) * (x1 - x0), e.clientX - bx.left, e.clientY - bx.top); };
  }
  function showPick(w, px, py) {
    const near = LD.lines.map((l) => ({ ...l, d: Math.abs(l.wave - w) })).filter((l) => l.d < 30).sort((a, b) => a.d - b.d).slice(0, 6);
    const sel = comp(B.sel), sl = sel && sel.type === "Lines" ? sel : null, P = $("mb-pick");
    const targets = B.st.comps.filter((c) => c.type === "Lines");
    P.innerHTML = `<div class="mb-ph"><b>${w.toFixed(0)} Å</b><span class="grow"></span><button class="mb-x" id="mb-pclose">✕</button></div>
      ${near.length ? near.map((l, k) => `<div class="mb-pl"><div><b class="mono">${esc(l.name)}</b> <span class="muted small">${l.wave} · ${esc(l.ion)}</span></div>
        <div class="mb-pa">${sl ? `<button class="mb-chip on" data-pk="${k}" data-to="${sl.uid}">+ ${esc(sl.kw.name)}</button>` : ""}
        <select class="small" data-pk="${k}" data-other="1"><option value="">add to…</option>${targets.filter((t) => t !== sl).map((t) => `<option value="${t.uid}">${esc(t.kw.name)}</option>`).join("")}
        <option value="n:narrow">new narrow component</option><option value="n:broad">new broad component</option><option value="n:outflow">new outflow component</option></select></div></div>`).join("")
        : `<p class="note small">No catalogued line within 30 Å.</p>`}
      <div class="mb-pl"><div class="note small">your own line here</div><div class="mb-pa"><input type="text" id="mb-pown" placeholder="name"><button class="mb-chip" id="mb-pownb">${sl ? "+ " + esc(sl.kw.name) : "+ new narrow"}</button></div></div>`;
    P.classList.remove("hidden");
    const box = $("mb-plot").getBoundingClientRect(); P.style.left = Math.min(px + 8, box.width - 300) + "px"; P.style.top = Math.max(4, Math.min(py - 20, 60)) + "px";
    const add = (entry, to) => edit(() => { let c;
      if (String(to).startsWith("n:")) { const role = to.slice(2); c = newComp("Lines", { role, name: role + "_extra", lines: [] }); B.st.comps.push(c); } else c = comp(+to);
      addLine(c, Array.isArray(entry) ? `${entry[0]}@${entry[1]}` : entry); B.sel = c.uid; P.classList.add("hidden"); });
    $("mb-pclose").onclick = () => P.classList.add("hidden");
    P.querySelectorAll("[data-pk]").forEach((b) => { const l = near[+b.dataset.pk];
      if (b.dataset.other) b.onchange = () => b.value && add(l.name, b.value); else b.onclick = () => add(l.name, b.dataset.to); });
    $("mb-pownb").onclick = () => { const n = $("mb-pown").value.trim() || `line${Math.round(w)}`; add([n.replace(/\s+/g, "_"), +w.toFixed(1)], sl ? sl.uid : "n:narrow"); };
  }

  // ---------------------------------------------------------------- suggestions (in the page, not the browser's list)
  const inFit = (l) => { const r = B.st.wave_range || LD.range; return l.wave >= r[0] && l.wave <= r[1]; };
  function lineItems(c, t) {
    t = t.trim(); const have = new Set((c.kw.lines || []).map(lineName)), out = [];
    const own = t.match(/^([A-Za-z0-9_\[\]\-+]+)\s*[@:]\s*([\d.]+)$/);
    if (own) out.push({ value: t, label: own[1], hint: `your own line at ${own[2]} Å`, own: true });
    let L;
    if (t === "") L = LD.lines.filter((l) => inFit(l) && /^(H[a-e]|\[OIII\]|\[NII\]|\[SII\]|HeI|HeII|\[OI\]|\[FeVI|\[FeVII|MgII|CIV|CIII)/.test(l.name));
    else if (/^\d+(\.\d*)?$/.test(t)) { const w = +t; L = t.length < 4 ? LD.lines.filter((l) => String(Math.round(l.wave)).startsWith(t)) : LD.lines.filter((l) => Math.abs(l.wave - w) < 40).sort((a, b) => Math.abs(a.wave - w) - Math.abs(b.wave - w)); }
    else { const k = t.toLowerCase().replace(/[\[\]\s]/g, ""), key = (l) => l.name.toLowerCase().replace(/[\[\]]/g, "");
      L = LD.lines.filter((l) => key(l).includes(k) || (l.ion || "").toLowerCase().replace(/\s/g, "").includes(k))
        .sort((a, b) => (inFit(b) - inFit(a)) || (key(b).startsWith(k) - key(a).startsWith(k)) || a.wave - b.wave); }
    return out.concat(L.slice(0, 40).map((l) => ({ value: l.name, label: l.name, hint: `${l.wave.toFixed(1)} Å${inFit(l) ? "" : " · outside the fit"}`, sub: l.ion, off: have.has(l.name) })));
  }
  function tieItems(name, lin, t) {
    const k = t.trim().toLowerCase().replace(/[\[\]]/g, "");
    const all = B.info.components.flatMap((q) => q.params).filter((p) => +p.linear === lin && p.name !== name);
    return [{ value: "", label: "— not tied —", hint: "" }].concat(all.filter((p) => !k || p.name.toLowerCase().replace(/[\[\]]/g, "").includes(k)).slice(0, 40)
      .map((p) => ({ value: p.name, label: p.name, hint: fmtN(p.value) + (p.unit ? " " + p.unit : "") })));
  }
  function suggest(inp, items, pick, keepText) {
    const box = document.createElement("div"); box.className = "mb-sug hidden"; inp.parentElement.appendChild(box);
    let list = [], k = -1;
    const mark = () => box.querySelectorAll(".mb-si").forEach((d) => { d.classList.toggle("on", +d.dataset.j === k); if (+d.dataset.j === k) d.scrollIntoView({ block: "nearest" }); });
    const choose = (j) => { const it = list[j]; if (!it || it.off) return; box.classList.add("hidden"); if (!keepText) inp.value = ""; pick(it.value); };
    const show = () => {
      list = items(inp.value);
      if (k >= list.length) k = list.length - 1;
      box.innerHTML = !list.length ? `<div class="mb-sh">No line matches “${esc(inp.value)}”. For your own line, type <b>name@wavelength</b>.</div>`
        : (inp.value.trim() === "" && !keepText ? `<div class="mb-sh">Type a name (Hb, OIII), an ion (Fe VII) or a wavelength · <b>name@λ</b> for your own</div>` : "") +
          list.map((it, j) => `<div class="mb-si${j === k ? " on" : ""}${it.off ? " off" : ""}${it.own ? " own" : ""}" data-j="${j}"><b>${esc(it.label)}</b>${it.sub ? `<span class="ion">${esc(it.sub)}</span>` : ""}<span class="w">${it.off ? "already in" : esc(it.hint)}</span></div>`).join("");
      box.classList.remove("hidden");
      box.querySelectorAll(".mb-si").forEach((d) => { d.onmousedown = (e) => { e.preventDefault(); choose(+d.dataset.j); }; d.onmouseenter = () => { k = +d.dataset.j; mark(); }; });
    };
    inp.addEventListener("focus", () => { k = keepText ? -1 : 0; show(); });
    inp.addEventListener("input", () => { k = 0; show(); });
    inp.addEventListener("blur", () => setTimeout(() => box.classList.add("hidden"), 120));
    inp.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); if (box.classList.contains("hidden")) show(); k = Math.max(0, Math.min(list.length - 1, k + (e.key === "ArrowDown" ? 1 : -1))); mark(); }
      else if (e.key === "Enter") { e.preventDefault(); if (k >= 0 && list[k] && !list[k].off) choose(k); else if (!keepText && inp.value.trim()) { const v = inp.value; inp.value = ""; box.classList.add("hidden"); pick(v); } else if (keepText) inp.dispatchEvent(new Event("change")); }
      else if (e.key === "Escape" && !box.classList.contains("hidden")) { e.stopPropagation(); box.classList.add("hidden"); }
    });
  }

  // ---------------------------------------------------------------- use, fit
  function summary() {
    const a = B.start.comps.map((c) => c.kw.name), b = B.st.comps.filter(ready).map((c) => c.kw.name), out = [];
    b.filter((n) => !a.includes(n)).forEach((n) => out.push("+" + n));
    a.filter((n) => !b.includes(n)).forEach((n) => out.push("−" + n));
    B.st.comps.filter(ready).forEach((c) => { const o = B.start.comps.find((q) => q.kw.name === c.kw.name); if (o && JSON.stringify(o.kw) !== JSON.stringify(c.kw)) out.push("~" + c.kw.name); });
    const np = Object.keys(B.st.params).filter((k) => JSON.stringify(B.st.params[k]) !== JSON.stringify(B.start.params[k])).length;
    if (np) out.push(`${np} parameter edit${np > 1 ? "s" : ""}`);
    return out;
  }
  async function fitMine() {
    const ch = summary(), title = B.title.trim() || (ch.length ? ch.slice(0, 3).join(" ") + (ch.length > 3 ? " …" : "") : "same as " + B.label);
    try {
      G.data = await call("/api/mb/fit", { state: payload(), title, change: `from ${B.label}: ${ch.join(", ") || "no change"}` });
      G.waitFor = G.data.key; const k = G.data.key.replace("mine", "");
      B.title = ""; B.fitKey = G.waitFor; B.start = clone(B.st); B.startNfree = B.info ? B.info.nfree : B.startNfree; B.undo = []; B.redo = []; B.label = `your model #${k}`;
      DRAFT = B; B = null; showWB(false); rank(); renderGallery();
      $("galcard").scrollIntoView({ behavior: "smooth", block: "start" });
      clearTimeout(G.timer); G.timer = setTimeout(load, 1500);
    } catch (e) { B.error = e.message; draw(); }
  }
  async function useMine() {
    const ok = await post("/api/model", { action: "recipe", state: payload() });
    if (!ok) return;
    toast("Your model is now the session model");
    B.start = clone(B.st); B.undo = []; B.redo = []; DRAFT = B; B = null; showWB(false); G.used = null; if ($("gx-new")) newButton();
  }

  document.addEventListener("keydown", (e) => {
    if (!B || !B.st) return;
    if (e.key === "Escape") {
      const P = $("mb-pick"); if (P && !P.classList.contains("hidden")) return P.classList.add("hidden");
      if (B.add) { B.add = false; return draw(); }
      const t = e.target;
      if (t && t.matches && t.matches("input,select")) return t.blur();     // the first Esc leaves the field, the next one the workspace
      return close();
    }
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z" && !(e.target.matches && e.target.matches("input,select"))) { e.preventDefault(); undo(e.shiftKey ? 1 : -1); }
  });
  window.addEventListener("resize", () => { if (B) plot(); else if (page === "model" && G.data) renderGallery(); });

  return { render, open, openCurrent: () => open(S.model ? { model: true } : {}, S.model ? "the current model" : "empty") };
})();
