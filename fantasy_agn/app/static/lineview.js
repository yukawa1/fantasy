// Results › Line profiles: the lines of the shown fit, one at a time or side by side (GET /api/wf/line_list,
// /api/wf/line_view, StepsMixin.wf_line_view). A strip of the fitted lines with their profiles; one line with
// everything else subtracted, its components, the FWHM at half maximum, the centroid and the residuals; or several
// lines scaled to their peaks on one velocity axis with a ladder of their widths. Uses api, esc, unitText and C of wf.js.
(() => {
  const $ = (id) => document.getElementById(id);
  const INK = "#1d2128", INK2 = "#52514e", GRID = "#eceef1", AXIS = "#7b8290", FAINT = "#a3a9b4";
  const MONO = '"IBM Plex Mono", "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace';
  const SANS = '"IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Helvetica, Arial, sans-serif';
  const CMP = ["#2a78d6", "#e34948", "#1baf7a", "#eda100", "#6a4fd8", "#b8459c", "#1f9e9a", "#8a5a2b"];
  const ok = (v) => v != null && isFinite(v);
  const ease = (t) => 1 - Math.pow(1 - Math.min(Math.max(t, 0), 1), 3);
  // a value rounded to two significant digits of its error: 4940 ± 70, (1.34 ± 0.05)×10⁴
  function pm(v, e, unit) {
    if (!ok(v)) return "—";
    let s;
    if (ok(e) && e > 0) {
      const p = Math.floor(Math.log10(e)) - 1, d = Math.max(0, -p), r = (x) => (Math.round(x / 10 ** p) * 10 ** p).toFixed(d);
      s = `${r(v)}<em> ± ${r(e)}</em>`;
    } else s = Math.abs(v) >= 100 ? v.toFixed(0) : v.toPrecision(3);
    return s + (unit ? ` <small>${unit}</small>` : "");
  }
  const kms = (v) => ok(v) ? `${v > 0 ? "+" : ""}${Math.round(v)}` : "—";
  const ROLE_NAME = { broad: "Broad", narrow: "Narrow", outflow: "Outflow", coronal: "Coronal", other: "Line" };
  function compColor(c, all) {
    if (c.bowen) return C.bowen;
    if (c.role === "broad") return C.broad[(c.n - 1) % C.broad.length];
    return { narrow: C.narrow, outflow: C.outflow, coronal: C.coronal }[c.role] || C.other[0];
  }
  const compLabel = (c, all) => `${ROLE_NAME[c.role] || c.name}${all.filter((x) => x.role === c.role).length > 1 ? " " + c.n : ""}`;

  const LV = { k: null, key: null, list: null, focus: null, mode: "one", cmp: [], views: {}, D: null, hidden: new Set(),
    zoom: null, hover: null, drag: null, born: 0, raf: 0, ladder: null, E: {}, norm: false, hotEp: null, pinEp: null, multi: false };
  // epochs coloured by time, early to late
  const RAMP = ["#4a3aa7", "#2a78d6", "#1f9e9a", "#1baf7a", "#d9a21b", "#e34948"];
  const RMS = "#d0247f";                                   // the rms profile: magenta, outside the time ramp
  function tcol(f) {
    const x = Math.min(Math.max(ok(f) ? f : 0, 0), 1) * (RAMP.length - 1), i = Math.min(Math.floor(x), RAMP.length - 2), u = x - i;
    const h = (c) => [1, 3, 5].map((j) => parseInt(c.slice(j, j + 2), 16)), a = h(RAMP[i]), b = h(RAMP[i + 1]);
    return `rgb(${a.map((v, j) => Math.round(v + u * (b[j] - v))).join(",")})`;
  }

  // ---------------------------------------------------------------- data
  async function view(line) {
    const id = `${LV.k}|${line}`;
    if (!LV.views[id]) LV.views[id] = api(`/api/wf/line_view?k=${LV.k}&line=${encodeURIComponent(line)}`).catch((e) => ({ error: String(e.message || e) }));
    return LV.views[id];
  }
  async function epochsView(line) {
    if (!LV.E[line]) LV.E[line] = api(`/api/wf/line_epochs?line=${encodeURIComponent(line)}`).catch((e) => ({ error: String(e.message || e) }));
    return LV.E[line];
  }
  async function show(k, cells) {
    const key = `${k}:${cells}`;
    const F = typeof fitDone === "function" ? fitDone() : null;
    LV.multi = !!(F && ["joint", "fits"].includes(F.kind) && (F.epochs || []).length > 1);
    $("lv-mode").querySelector('[data-m="ep"]').classList.toggle("hidden", !LV.multi);
    if (!LV.multi && LV.mode === "ep") { LV.mode = "one"; $("lv-mode").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.m === "one")); }
    if (LV.key === key) { kick(); return; }
    if (LV.key && LV.key.split(":")[1] !== String(cells)) LV.E = {};      // another fit: the epochs again
    LV.key = key; LV.k = k; LV.views = {}; LV.zoom = null;
    $("lv-strip").innerHTML = '<div class="lv-loading">Measuring the lines…</div>';
    try { LV.list = await api(`/api/wf/line_list?k=${k}`); } catch (e) { LV.list = { lines: [] }; }
    if (LV.key !== key) return;
    const names = LV.list.lines.map((d) => d.line);
    if (!names.includes(LV.focus)) LV.focus = LV.list.main || names[0] || null;
    LV.cmp = LV.cmp.filter((l) => names.includes(l));
    if (!LV.cmp.length) LV.cmp = defaultCmp();
    strip();
    await load();
  }
  // lines to compare first: the main broad lines, one of each width (lines tied to one shape add nothing)
  function defaultCmp() {
    const MAIN = ["Hb", "Ha", "MgII2798", "CIV1549", "HeII4686", "Hg", "Hd", "Lya", "SiIV1400", "CIII]1909", "HeI5876"];
    const L = LV.list.lines.filter((d) => d.fwhm), broad = L.filter((d) => d.roles.includes("broad"));
    const pool = (broad.length >= 2 ? broad : L).slice().sort((a, b) => ((MAIN.indexOf(a.line) + 1) || 99) - ((MAIN.indexOf(b.line) + 1) || 99));
    const out = [];
    for (const d of pool) if (!out.some((o) => Math.abs(o.fwhm - d.fwhm) < 0.01 * d.fwhm)) out.push(d);
    return (out.length ? out : pool).slice(0, 4).map((d) => d.line);
  }
  async function load() {
    if (!LV.focus) { $("lv-side").innerHTML = '<p class="note">This fit has no emission lines to show.</p>'; return; }
    $("lv-wait").classList.remove("hidden");
    if (LV.mode === "one") {
      const D = await view(LV.focus);
      if (LV.mode !== "one") return;
      LV.D = D;
    } else if (LV.mode === "ep") {
      const want = LV.focus;
      $("lv-side").innerHTML = '<div class="dg-skel"><i></i><i></i><i></i></div>';
      const E = await epochsView(want);
      if (LV.mode !== "ep" || LV.focus !== want) return;
      LV.ED = E;
    } else {
      await Promise.all(LV.cmp.map(view));
    }
    $("lv-wait").classList.add("hidden");
    LV.born = performance.now();
    side(); legend(); kick();
  }

  // ---------------------------------------------------------------- the strip of lines
  function sparkPath(sp, w, h) {
    return sp.map((y, i) => `${i ? "L" : "M"}${(i / (sp.length - 1) * w).toFixed(1)},${(h - 2 - y * (h - 4)).toFixed(1)}`).join("");
  }
  function strip() {
    const L = LV.list.lines;
    if (!L.length) { $("lv-strip").innerHTML = '<p class="note small">No fitted lines.</p>'; return; }
    const card = (d) => {
      const col = d.roles.includes("broad") ? C.broad[0] : /^\[Fe(VII|X|XI|XIV)\]/.test(d.line) ? C.coronal : C.narrow;
      const sp = d.spark ? `<svg viewBox="0 0 64 20" preserveAspectRatio="none"><path d="${sparkPath(d.spark, 64, 20)} L64,20 L0,20 Z" fill="${col}" opacity=".14"/><path d="${sparkPath(d.spark, 64, 20)}" fill="none" stroke="${col}" stroke-width="1.4"/></svg>` : "<svg></svg>";
      const inCmp = LV.cmp.indexOf(d.line);
      return `<button type="button" class="lv-card${d.line === LV.focus && LV.mode !== "cmp" ? " on" : ""}${inCmp >= 0 && LV.mode === "cmp" ? " in" : ""}" data-line="${esc(d.line)}"` +
        `${inCmp >= 0 && LV.mode === "cmp" ? ` style="--lc:${CMP[inCmp % CMP.length]}"` : ""} title="${esc(d.label)}: ${esc(d.roles.join(" + "))}">` +
        `<span class="lv-cname">${esc(d.label.replace(" λ", " "))}</span>${sp}<span class="lv-cw mono">${d.fwhm ? Math.round(d.fwhm) : "—"}</span></button>`;
    };
    const broad = L.filter((d) => d.roles.includes("broad")), narrow = L.filter((d) => !d.roles.includes("broad"));
    $("lv-strip").innerHTML = (broad.length ? `<div class="lv-group"><span class="lv-glabel">Broad <small>FWHM of the broad profile, km/s</small></span><div class="lv-cards">${broad.map(card).join("")}</div></div>` : "") +
      (narrow.length ? `<div class="lv-group"><span class="lv-glabel">Narrow <small>FWHM, km/s</small></span><div class="lv-cards">${narrow.map(card).join("")}</div></div>` : "");
    $("lv-strip").querySelectorAll(".lv-card").forEach((b) => (b.onclick = (e) => pick(b.dataset.line, e.shiftKey || e.metaKey || e.ctrlKey)));
    const on = $("lv-strip").querySelector(".lv-card.on");
    if (on) on.parentElement.scrollLeft = Math.max(0, on.offsetLeft - on.parentElement.offsetLeft - 40);
  }
  function pick(line, add) {
    if (add && LV.mode === "one") setMode("cmp");
    if (LV.mode === "one" || LV.mode === "ep") { if (LV.focus === line) return; LV.focus = line; LV.zoom = null; LV.hotEp = LV.pinEp = null; if (LV.mode === "one") LV.hidden.clear(); }
    else {
      const i = LV.cmp.indexOf(line);
      if (i >= 0) { if (LV.cmp.length > 1) LV.cmp.splice(i, 1); } else LV.cmp.push(line);
      LV.focus = line;
    }
    strip(); load();
  }
  function setMode(m) {
    if (m === "ep" && LV.mode !== "ep") LV.hidden = new Set(["data"]);
    else if (m !== "ep" && LV.mode === "ep") LV.hidden.clear();
    LV.mode = m; LV.zoom = null; LV.hotEp = LV.pinEp = null;
    $("lv-mode").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.m === m));
    if (m === "cmp" && !LV.cmp.includes(LV.focus)) LV.cmp.unshift(LV.focus);
    strip(); load();
  }

  // ---------------------------------------------------------------- side panel and legend
  function shapeText(r) {
    if (!ok(r)) return "";
    if (r < 1.6) return "a sharp core on wide wings (Lorentzian-like, or two components of very different width)";
    if (r < 2.1) return "a core with wings wider than a Gaussian";
    if (r <= 2.6) return "close to a Gaussian (2.35)";
    return "flatter than a Gaussian: a flat or double-peaked top";
  }
  function side() {
    if (LV.mode === "cmp") return ladder();
    if (LV.mode === "ep") return epSide();
    const D = LV.D;
    if (!D || D.error) { $("lv-side").innerHTML = `<p class="note">${esc((D && D.error) || "")}</p>`; return; }
    const M = D.meas, unit = D.flux_unit ? unitText(D.flux_unit) : "", nb = D.comps.filter((c) => c.role === "broad").length;
    const of = D.of === "broad" ? (nb > 1 ? `the ${nb} broad components summed` : "the broad component") : D.comps.length > 1 ? "all its components summed" : "its one component";
    const r = M.ratio, gx = (x) => Math.min(Math.max(x / 3.5, 0), 1) * 100;
    const total = D.comps.reduce((a, c) => a + (ok(c.flux) ? c.flux : 0), 0) || 1;
    const rows = D.comps.map((c) => `<tr data-c="${esc(c.name)}"><td><span class="sw" style="background:${compColor(c, D.comps)}"></span>${esc(compLabel(c, D.comps))}${c.profile !== "gaussian" ? ` <small class="muted">${esc(c.profile)}</small>` : ""}</td>` +
      `<td class="num mono">${ok(c.fwhm) ? Math.round(c.fwhm) : "—"}</td><td class="num mono">${kms(c.shift)}</td>` +
      `<td class="lv-share"><span style="width:${Math.max(0, (c.flux || 0) / total * 100).toFixed(1)}%;background:${compColor(c, D.comps)}"></span><b class="mono">${ok(c.flux) ? Math.round(c.flux / total * 100) + "%" : "—"}</b></td></tr>`).join("");
    $("lv-side").innerHTML =
      `<div class="lv-eyebrow">Measured on ${of}</div>` +
      `<div class="lv-big"><div class="lv-stat main"><span>FWHM</span><b>${pm(M.fwhm, M.fwhm_err, "km/s")}</b></div>` +
      `<div class="lv-stat"><span>σ (line dispersion)</span><b>${pm(M.sigma, M.sigma_err, "km/s")}</b></div>` +
      `<div class="lv-stat"><span>Centroid shift</span><b>${pm(M.shift, M.shift_err, "km/s")}</b></div>` +
      `<div class="lv-stat"><span>Flux</span><b>${pm(M.flux, M.flux_err)}</b><small class="muted">${esc(unit)}</small></div>` +
      `<div class="lv-stat"><span>EW</span><b>${pm(M.ew, M.ew_err, "Å")}</b></div></div>` +
      (M.fwhm_fragile ? `<div class="callout warn small">The FWHM is fragile: a shoulder near half maximum (width at 45% / 55% of the peak = ${M.fragile_ratio.toFixed(2)}), so a small change of the profile moves it a lot. σ is the steadier width here.</div>` : "") +
      (ok(r) ? `<div class="lv-shape"><div class="lv-shapehead"><span>Shape</span><b class="mono">FWHM / σ = ${r.toFixed(2)}</b></div>` +
        `<div class="lv-gauge"><span class="z1"></span><span class="z2"></span><span class="z3"></span><i style="left:${gx(2.35)}%" class="g"></i><i style="left:${gx(r)}%" class="me"></i></div>` +
        `<div class="lv-gaugeax"><span>0</span><span style="left:${gx(2.35)}%">Gaussian 2.35</span><span style="right:0">3.5</span></div>` +
        `<p class="small">${esc(shapeText(r))}.</p></div>` : "") +
      `<table class="lv-tbl"><thead><tr><th>Component</th><th class="num">FWHM</th><th class="num">shift</th><th>flux share</th></tr></thead><tbody>${rows}</tbody></table>` +
      `<p class="note small">Point to the plot to read the curves; drag across it to zoom, double-click to zoom out. Shift-click a line above to compare it with this one.</p>`;
    $("lv-side").querySelectorAll("tr[data-c]").forEach((tr) => {
      tr.onmouseenter = () => { LV.hot = tr.dataset.c; kick(); };
      tr.onmouseleave = () => { LV.hot = null; kick(); };
    });
  }
  async function ladder() {
    const Vs = (await Promise.all(LV.cmp.map(view))).filter((v) => v && !v.error);
    const top = Math.max(1000, ...Vs.map((v) => (v.meas.fwhm || 0) + (v.meas.fwhm_err || 0)), ...Vs.map((v) => v.meas.sigma || 0)) * 1.05;
    const step = niceStep(top, 3), max = Math.ceil(top / step) * step;
    const x = (v) => Math.min(100, v / max * 100);
    const rows = Vs.map((v) => {
      const M = v.meas, col = CMP[LV.cmp.indexOf(v.line) % CMP.length];
      const err = ok(M.fwhm_err) ? `<i class="err" style="left:${x(M.fwhm - M.fwhm_err)}%;width:${x(2 * M.fwhm_err)}%"></i>` : "";
      return `<div class="lv-rung"><div class="lv-rname"><span class="sw" style="background:${col}"></span>${esc(v.label)}<small>${v.of === "broad" ? "broad" : "whole line"}</small></div>` +
        `<div class="lv-rbar"><span class="bar" style="width:${x(M.fwhm || 0)}%;background:${col}"></span>${err}${ok(M.sigma) ? `<i class="sig" style="left:${x(M.sigma)}%" title="σ ${Math.round(M.sigma)} km/s"></i>` : ""}</div>` +
        `<div class="lv-rnum mono">${ok(M.fwhm) ? Math.round(M.fwhm) : "—"}<small>σ ${ok(M.sigma) ? Math.round(M.sigma) : "—"} · ${kms(M.shift)}</small></div></div>`;
    }).join("");
    const ticks = [];
    for (let t = 0; t <= max + 1e-9; t += step) ticks.push(`<span style="left:${t / max * 100}%">${t >= 1000 ? +(t / 1000).toFixed(1) + "k" : t}</span>`);
    $("lv-side").innerHTML = `<div class="lv-eyebrow">Widths side by side</div><p class="small muted">Bar: FWHM with its error; tick: σ; then the centroid shift, all in km/s.</p>` +
      `<div class="lv-ladder">${rows}<div class="lv-rax"><span></span><div>${ticks.join("")}</div><span></span></div></div>` +
      (Vs.length >= 2 ? `<div class="lv-ratios">${ratioNote(Vs)}</div>` : "") + tiedNote(Vs) +
      `<p class="note small">Click a line above to add it or take it away.</p>`;
  }
  // the FWHM of each line over that of the first (errors added in quadrature, as if independent)
  function ratioNote(Vs) {
    const a = Vs[0];
    if (!ok(a.meas.fwhm)) return "";
    const rows = Vs.slice(1).filter((b) => ok(b.meas.fwhm)).map((b) => {
      const q = b.meas.fwhm / a.meas.fwhm, e = q * Math.hypot((a.meas.fwhm_err || 0) / a.meas.fwhm, (b.meas.fwhm_err || 0) / b.meas.fwhm);
      return `<div><span>${esc(b.label.replace(" λ", " "))}</span><b class="mono">${q.toFixed(2)} <em>± ${e.toFixed(2)}</em></b></div>`;
    }).join("");
    return rows ? `<div class="lv-rhead">FWHM relative to <b>${esc(a.label.replace(" λ", " "))}</b></div>${rows}` : "";
  }
  // lines whose broad profiles are one shape (tied in the model): their curves lie on each other
  function tiedNote(Vs) {
    const groups = [];
    Vs.forEach((v) => {
      const g = groups.find((x) => ok(x[0].meas.fwhm) && ok(v.meas.fwhm) && Math.abs(x[0].meas.fwhm - v.meas.fwhm) < 0.002 * v.meas.fwhm
        && Math.abs((x[0].meas.shift || 0) - (v.meas.shift || 0)) < 5);
      if (g) g.push(v); else groups.push([v]);
    });
    const tied = groups.filter((g) => g.length > 1);
    return tied.map((g) => `<p class="note small">${g.map((v) => esc(v.label.replace(" λ", " "))).join(", ")}: one shape, tied in the model, so their curves lie on top of each other.</p>`).join("");
  }
  function legend() {
    const el = $("lv-legend");
    if (LV.mode === "ep") {
      const E = LV.ED, eps = E && !E.error ? E.epochs.filter((e) => !e.missing) : [];
      el.innerHTML = eps.length ? `<span class="lv-tbar"><small>${esc(eps[0].date || "#" + eps[0].k)}</small><i style="background:linear-gradient(90deg,${RAMP.join(",")})"></i><small>${esc(eps[eps.length - 1].date || "#" + eps[eps.length - 1].k)}</small></span>` +
        [["data", "data", C.data], ["_mean", "mean", INK], ["_rms", "rms", RMS]].map(([k, t, c]) => `<button type="button" class="lv-chip${LV.hidden.has(k) ? " off" : ""}" data-k="${k}" style="--c:${c}"><i></i>${t}</button>`).join("") : "";
      el.querySelectorAll("button").forEach((b) => (b.onclick = () => { LV.hidden.has(b.dataset.k) ? LV.hidden.delete(b.dataset.k) : LV.hidden.add(b.dataset.k); legend(); kick(); }));
      return;
    }
    if (LV.mode === "cmp") {
      el.innerHTML = LV.cmp.map((l, i) => { const d = LV.list.lines.find((x) => x.line === l); return `<span class="lv-chip" style="--c:${CMP[i % CMP.length]}"><i></i>${esc(d ? d.label : l)}</span>`; }).join("") +
        '<span class="lv-chip muted"><i class="dot"></i>half maximum</span>';
      return;
    }
    const D = LV.D;
    if (!D || D.error) { el.innerHTML = ""; return; }
    const items = [["data", "data, the rest of the model subtracted", C.data]].concat(D.comps.map((c) => [c.name, compLabel(c, D.comps), compColor(c, D.comps)]));
    if (D.comps.filter((c) => c.role === "broad").length > 1) items.push(["_broad", "broad, summed", C.broad[0]]);
    if (D.comps.length > 1) items.push(["_sum", "sum", INK]);
    el.innerHTML = items.map(([k, t, c]) => `<button type="button" class="lv-chip${LV.hidden.has(k) ? " off" : ""}" data-k="${esc(k)}" style="--c:${c}"><i></i>${esc(t)}</button>`).join("");
    el.querySelectorAll("button").forEach((b) => (b.onclick = () => { LV.hidden.has(b.dataset.k) ? LV.hidden.delete(b.dataset.k) : LV.hidden.add(b.dataset.k); legend(); kick(); }));
  }

  // ---------------------------------------------------------------- drawing
  function canvas() {
    const cv = $("lv-plot"), w = cv.clientWidth, h = cv.clientHeight, dpr = window.devicePixelRatio || 1;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    return { g, w, h };
  }
  function niceStep(span, n) {
    const raw = span / n, p = 10 ** Math.floor(Math.log10(raw)), f = raw / p;
    return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
  }
  function axes(g, P, x0, x1, y0, y1, ylab, xlab) {
    g.font = `10px ${MONO}`; g.fillStyle = AXIS; g.strokeStyle = GRID; g.lineWidth = 1;
    const sx = niceStep(x1 - x0, 8);
    g.textAlign = "center"; g.textBaseline = "top";
    for (let v = Math.ceil(x0 / sx) * sx; v <= x1; v += sx) {
      const X = P.X(v);
      g.beginPath(); g.moveTo(X, P.t); g.lineTo(X, P.b); g.stroke();
      if (xlab) g.fillText(Math.abs(sx) >= 1000 ? (v / 1000).toFixed(sx % 1000 ? 1 : 0) : String(Math.round(v)), X, P.rb + 5);
    }
    const sy = niceStep(y1 - y0, 5);
    g.textAlign = "right"; g.textBaseline = "middle";
    for (let v = Math.ceil(y0 / sy) * sy; v <= y1; v += sy) {
      const Y = P.Y(v);
      g.beginPath(); g.moveTo(P.l, Y); g.lineTo(P.r, Y); g.stroke();
      g.fillText(Math.abs(v) < 1e-9 ? "0" : +v.toPrecision(3), P.l - 6, Y);
    }
    if (xlab) { g.textAlign = "center"; g.textBaseline = "top"; g.fillStyle = INK2; g.font = `11px ${SANS}`; g.fillText(xlab, (P.l + P.r) / 2, P.rb + 20); }
    if (ylab) { g.save(); g.translate(14, (P.t + P.b) / 2); g.rotate(-Math.PI / 2); g.textAlign = "center"; g.textBaseline = "middle"; g.fillStyle = INK2; g.font = `11px ${SANS}`; g.fillText(ylab, 0, 0); g.restore(); }
  }
  function curve(g, xs, ys, P, grow) {
    g.beginPath();
    let on = false;
    for (let i = 0; i < xs.length; i++) {
      const y = ys[i];
      if (!ok(y) || xs[i] < P.x0 || xs[i] > P.x1) { on = false; continue; }
      const X = P.X(xs[i]), Y = P.Y(y * grow);
      if (on) g.lineTo(X, Y); else { g.moveTo(X, Y); on = true; }
    }
    g.stroke();
  }
  function frame() {
    LV.raf = 0;
    if (!$("lv-plot") || $("res-lines").classList.contains("hidden")) return;
    const grow = ease((performance.now() - LV.born) / 450);
    if (LV.mode === "cmp") drawCmp(grow); else if (LV.mode === "ep") drawEp(grow); else drawOne(grow);
    if (grow < 1) kick();
  }
  function kick() { if (!LV.raf) LV.raf = requestAnimationFrame(frame); }

  function drawOne(grow) {
    const D = LV.D, { g, w, h } = canvas();
    if (!D || D.error) return;
    const [x0, x1] = LV.zoom || [-D.window, D.window];
    const P = { l: 58, r: w - 14, t: 26, b: Math.round(h * 0.72), rt: Math.round(h * 0.72) + 10, rb: h - 34, x0, x1 };
    const inX = (v) => v >= x0 && v <= x1;
    const sum = D.vf.map((_, i) => D.comps.reduce((a, c) => a + c.y[i], 0));
    const broad = D.comps.filter((c) => c.role === "broad"), bsum = broad.length > 1 ? D.vf.map((_, i) => broad.reduce((a, c) => a + c.y[i], 0)) : null;
    const ys = D.y.filter((y, i) => ok(y) && D.ok[i] && inX(D.v[i])).sort((a, b) => a - b);
    const q = (f) => ys.length ? ys[Math.floor(f * (ys.length - 1))] : 0;
    const mx = Math.max(q(0.995), ...sum.filter((_, i) => inX(D.vf[i])), 1e-9);
    const y0 = Math.min(0, q(0.005)) - 0.04 * mx, y1 = mx * 1.18;
    P.X = (v) => P.l + (v - x0) / (x1 - x0) * (P.r - P.l);
    P.Y = (y) => P.b - (y - y0) / (y1 - y0) * (P.b - P.t);
    axes(g, P, x0, x1, y0, y1, D.flux_unit ? "flux" : "flux", null);
    // rest wavelength and the other lines in the window
    g.setLineDash([3, 3]); g.strokeStyle = FAINT;
    g.beginPath(); g.moveTo(P.X(0), P.t); g.lineTo(P.X(0), P.b); g.stroke(); g.setLineDash([]);
    g.font = `10px ${SANS}`; g.fillStyle = FAINT; g.textAlign = "center"; g.textBaseline = "bottom";
    if (inX(0)) g.fillText("rest", P.X(0), P.t - 4);
    D.others.forEach((o) => { if (!inX(o.v)) return; const X = P.X(o.v); g.strokeStyle = "#d8dbe0"; g.beginPath(); g.moveTo(X, P.t + 2); g.lineTo(X, P.t + 10); g.stroke(); g.fillText(o.label.replace(" λ", " "), X, P.t - 4); });
    g.save(); g.beginPath(); g.rect(P.l, P.t - 2, P.r - P.l, P.b - P.t + 2); g.clip();
    // the data: error band, then the steps
    if (!LV.hidden.has("data")) {
      g.fillStyle = "rgba(155,154,149,.16)";
      g.beginPath();
      let started = false; const back = [];
      for (let i = 0; i < D.v.length; i++) {
        if (!inX(D.v[i]) || !ok(D.y[i]) || !ok(D.e[i])) continue;
        const X = P.X(D.v[i]);
        if (!started) { g.moveTo(X, P.Y(D.y[i] + D.e[i])); started = true; } else g.lineTo(X, P.Y(D.y[i] + D.e[i]));
        back.push([X, P.Y(D.y[i] - D.e[i])]);
      }
      back.reverse().forEach(([X, Y]) => g.lineTo(X, Y)); g.fill();
      g.lineWidth = 1;
      for (const masked of [true, false]) {
        g.strokeStyle = masked ? "#d9d8d3" : C.data;
        g.beginPath(); let on = false;
        for (let i = 0; i < D.v.length; i++) {
          if (!inX(D.v[i]) || !ok(D.y[i]) || (!D.ok[i]) !== masked) { on = false; continue; }
          const X = P.X(D.v[i]), Y = P.Y(D.y[i]);
          if (on) g.lineTo(X, Y); else { g.moveTo(X, Y); on = true; }
        }
        g.stroke();
      }
    }
    // the components, the broad sum and the sum
    D.comps.forEach((c) => {
      if (LV.hidden.has(c.name)) return;
      const col = compColor(c, D.comps), hot = LV.hot === c.name;
      g.strokeStyle = col; g.lineWidth = hot ? 2.6 : 1.5;
      g.setLineDash(c.role === "broad" && c.n > 1 ? [6, 3] : []);
      curve(g, D.vf, c.y, P, grow);
      g.setLineDash([]);
      if (c.role === "broad" || hot) {
        g.fillStyle = col; g.globalAlpha = hot ? 0.16 : 0.07;
        g.beginPath(); g.moveTo(P.X(Math.max(x0, D.vf[0])), P.Y(0));
        D.vf.forEach((v, i) => { if (inX(v)) g.lineTo(P.X(v), P.Y(c.y[i] * grow)); });
        g.lineTo(P.X(Math.min(x1, D.vf[D.vf.length - 1])), P.Y(0)); g.fill(); g.globalAlpha = 1;
      }
    });
    if (bsum && !LV.hidden.has("_broad")) { g.strokeStyle = C.broad[0]; g.globalAlpha = 0.45; g.lineWidth = 3.2; g.lineCap = "round"; curve(g, D.vf, bsum, P, grow); g.globalAlpha = 1; g.lineCap = "butt"; }
    if (D.comps.length > 1 && !LV.hidden.has("_sum")) { g.strokeStyle = INK; g.lineWidth = 1.3; curve(g, D.vf, sum, P, grow); }
    if (D.absorbed && !LV.hidden.has("_sum")) { g.strokeStyle = INK; g.setLineDash([4, 3]); g.lineWidth = 1; curve(g, D.vf, D.absorbed, P, grow); g.setLineDash([]); }
    g.restore();
    // the FWHM at half maximum and the centroid
    if (D.hm && grow > 0.6) {
      const a = Math.min(1, (grow - 0.6) / 0.4), [vl, vr, half] = D.hm, Y = P.Y(half);
      g.globalAlpha = a; g.strokeStyle = INK; g.fillStyle = INK; g.lineWidth = 1.2;
      g.beginPath(); g.moveTo(P.X(vl), Y); g.lineTo(P.X(vr), Y); g.stroke();
      for (const v of [vl, vr]) { g.beginPath(); g.moveTo(P.X(v), Y - 5); g.lineTo(P.X(v), Y + 5); g.stroke(); g.beginPath(); g.arc(P.X(v), Y, 2.6, 0, 7); g.fill(); }
      const M = D.meas, txt = `FWHM ${ok(M.fwhm) ? Math.round(M.fwhm) : "—"}${ok(M.fwhm_err) ? " ± " + Math.round(M.fwhm_err) : ""} km/s`;
      g.font = `600 11px ${MONO}`; g.textAlign = "center"; g.textBaseline = "bottom";
      const tx = Math.min(Math.max((P.X(vl) + P.X(vr)) / 2, P.l + 70), P.r - 70), tw = g.measureText(txt).width + 10;
      g.fillStyle = "rgba(255,255,255,.85)"; g.fillRect(tx - tw / 2, Y - 21, tw, 15);
      g.fillStyle = INK; g.fillText(txt, tx, Y - 7);
      if (ok(M.shift) && inX(M.shift)) {
        const X = P.X(M.shift);
        g.strokeStyle = C.accent; g.setLineDash([2, 3]); g.lineWidth = 1;
        g.beginPath(); g.moveTo(X, Y + 8); g.lineTo(X, P.b); g.stroke(); g.setLineDash([]);
        g.fillStyle = C.accent; g.beginPath(); g.moveTo(X, P.b); g.lineTo(X - 4, P.b + 6); g.lineTo(X + 4, P.b + 6); g.fill();
        g.font = `10px ${MONO}`; g.textBaseline = "top"; g.fillText(`centroid ${kms(M.shift)}`, Math.min(Math.max(X, P.l + 50), P.r - 50), P.b + 8);
      }
      g.globalAlpha = 1;
    }
    // residuals in sigma
    const R = { ...P, t: P.rt + 14, b: P.rb };
    const rm = 5, RY = (s) => R.t + (1 - (Math.max(-rm, Math.min(rm, s)) + rm) / (2 * rm)) * (R.b - R.t);
    g.fillStyle = "rgba(155,154,149,.12)"; g.fillRect(P.l, RY(2), P.r - P.l, RY(-2) - RY(2));
    g.strokeStyle = GRID; g.beginPath(); g.moveTo(P.l, RY(0)); g.lineTo(P.r, RY(0)); g.stroke();
    const bw = Math.max(1, (P.r - P.l) / Math.max(1, D.v.filter(inX).length) * 0.8);
    for (let i = 0; i < D.v.length; i++) {
      const s = D.res[i];
      if (!inX(D.v[i]) || !ok(s) || !D.ok[i]) continue;
      const a = Math.abs(s);
      g.fillStyle = a > 3 ? C.accent : a > 2 ? "#d9a21b" : "#a9a8a3";
      const Y = RY(s * grow), Z = RY(0);
      g.fillRect(P.X(D.v[i]) - bw / 2, Math.min(Y, Z), bw, Math.max(1, Math.abs(Y - Z)));
    }
    g.font = `10px ${MONO}`; g.fillStyle = AXIS; g.textAlign = "right"; g.textBaseline = "middle";
    [-4, 0, 4].forEach((s) => g.fillText(String(s), P.l - 6, RY(s)));
    g.save(); g.translate(14, (R.t + R.b) / 2); g.rotate(-Math.PI / 2); g.textAlign = "center"; g.fillStyle = INK2; g.font = `11px ${SANS}`; g.fillText("resid. σ", 0, 0); g.restore();
    // x axis under the residuals
    g.textAlign = "center"; g.textBaseline = "top"; g.font = `10px ${MONO}`; g.fillStyle = AXIS;
    const sx = niceStep(x1 - x0, 8);
    for (let v = Math.ceil(x0 / sx) * sx; v <= x1; v += sx) g.fillText(sx >= 1000 ? (v / 1000).toFixed(sx % 1000 ? 1 : 0) : String(Math.round(v)), P.X(v), R.b + 5);
    g.fillStyle = INK2; g.font = `11px ${SANS}`; g.fillText(sx >= 1000 ? "velocity (10³ km/s)" : "velocity (km/s)", (P.l + P.r) / 2, R.b + 19);
    LV.P = P;
    hoverOne(g, P, R, sum, bsum);
    dragBox(g, P, R);
  }
  function at(xs, ys, v) {
    let i = 0, j = xs.length - 1;
    if (v <= xs[0]) return ys[0]; if (v >= xs[j]) return ys[j];
    while (j - i > 1) { const m = (i + j) >> 1; if (xs[m] < v) i = m; else j = m; }
    const f = (v - xs[i]) / (xs[j] - xs[i]);
    return ok(ys[i]) && ok(ys[j]) ? ys[i] + f * (ys[j] - ys[i]) : null;
  }
  function hoverOne(g, P, R, sum, bsum) {
    const tip = $("lv-tip"), D = LV.D;
    if (!LV.hover || LV.drag) { tip.classList.add("hidden"); return; }
    const X = LV.hover.x;
    if (X < P.l || X > P.r) { tip.classList.add("hidden"); return; }
    const v = P.x0 + (X - P.l) / (P.r - P.l) * (P.x1 - P.x0);
    g.strokeStyle = "rgba(29,33,40,.35)"; g.lineWidth = 1;
    g.beginPath(); g.moveTo(X, P.t); g.lineTo(X, R.b); g.stroke();
    const rows = [];
    const dv = at(D.v, D.y, v), rv = at(D.v, D.res, v);
    if (!LV.hidden.has("data")) rows.push([C.data, "data", dv]);
    D.comps.forEach((c) => { if (!LV.hidden.has(c.name)) { const y = at(D.vf, c.y, v); rows.push([compColor(c, D.comps), compLabel(c, D.comps), y]); g.fillStyle = compColor(c, D.comps); g.beginPath(); g.arc(X, P.Y(y), 3, 0, 7); g.fill(); } });
    if (bsum && !LV.hidden.has("_broad")) rows.push([C.broad[0], "broad, summed", at(D.vf, bsum, v)]);
    if (D.comps.length > 1) rows.push([INK, "sum", at(D.vf, sum, v)]);
    const f = (y) => !ok(y) ? "—" : Math.abs(y) < 5e-4 ? "0" : Math.abs(y) >= 100 ? y.toFixed(0) : Math.abs(y) >= 1 ? y.toFixed(2) : y.toFixed(3);
    tip.innerHTML = `<div class="lv-tiphead mono">${kms(v)} km/s <span class="muted">· ${(D.lam0 * (1 + v / 299792.458)).toFixed(1)} Å</span></div>` +
      rows.map(([c, t, y]) => `<div><i style="background:${c}"></i>${esc(t)}<b class="mono">${f(y)}</b></div>`).join("") +
      (ok(rv) ? `<div class="lv-tipres">residual <b class="mono">${rv.toFixed(1)} σ</b></div>` : "");
    tip.classList.remove("hidden");
    const wrap = $("lv-plot").clientWidth, tw = tip.offsetWidth;
    tip.style.left = `${X + 14 + tw > wrap ? X - tw - 14 : X + 14}px`;
    tip.style.top = `${P.t + 6}px`;
  }
  function dragBox(g, P, R) {
    if (!LV.drag || Math.abs(LV.drag.x1 - LV.drag.x0) < 3) return;
    const a = Math.max(P.l, Math.min(LV.drag.x0, LV.drag.x1)), b = Math.min(P.r, Math.max(LV.drag.x0, LV.drag.x1));
    g.fillStyle = "rgba(42,120,214,.08)"; g.fillRect(a, P.t, b - a, R.b - P.t);
    g.strokeStyle = "rgba(42,120,214,.5)"; g.strokeRect(a + 0.5, P.t + 0.5, b - a, R.b - P.t);
  }

  async function drawCmp(grow) {
    const Vs = LV.cmp.map((l) => LV.views[`${LV.k}|${l}`]);
    if (Vs.some((p) => !p || !p.then)) return;
    LV.cmpData = LV.cmpData || {};
    const ready = Vs.every((p) => LV.cmpData[LV.cmp[Vs.indexOf(p)]]);
    if (!ready) { const got = await Promise.all(Vs); got.forEach((d, i) => (LV.cmpData[LV.cmp[i]] = d)); kick(); return; }
    const { g, w, h } = canvas();
    const Ds = LV.cmp.map((l) => LV.cmpData[l]).filter((d) => d && !d.error);
    if (!Ds.length) return;
    const W = Math.min(20000, Math.max(...Ds.map((d) => d.window)));
    const [x0, x1] = LV.zoom || [-W, W];
    const P = { l: 58, r: w - 14, t: 22, b: h - 40, rb: h - 40, x0, x1 };
    const y0 = -0.06, y1 = 1.12;
    P.X = (v) => P.l + (v - x0) / (x1 - x0) * (P.r - P.l);
    P.Y = (y) => P.b - (y - y0) / (y1 - y0) * (P.b - P.t);
    axes(g, P, x0, x1, y0, y1, "flux / peak", null);
    g.setLineDash([3, 3]); g.strokeStyle = FAINT; g.beginPath(); g.moveTo(P.X(0), P.t); g.lineTo(P.X(0), P.b); g.stroke(); g.setLineDash([]);
    g.strokeStyle = "#d8dbe0"; g.beginPath(); g.moveTo(P.l, P.Y(0.5)); g.lineTo(P.r, P.Y(0.5)); g.stroke();
    g.font = `10px ${MONO}`; g.fillStyle = FAINT; g.textAlign = "left"; g.textBaseline = "bottom"; g.fillText("½", P.l + 4, P.Y(0.5) - 1);
    const profs = [];
    g.save(); g.beginPath(); g.rect(P.l, P.t, P.r - P.l, P.b - P.t); g.clip();
    Ds.forEach((D) => {
      const col = CMP[LV.cmp.indexOf(D.line) % CMP.length];
      const use = D.of === "broad" ? D.comps.filter((c) => c.role === "broad") : D.comps;
      const ref = D.vf.map((_, i) => use.reduce((a, c) => a + c.y[i], 0)), pk = Math.max(...ref, 1e-12);
      // the data behind it: the line's own data minus its other components, scaled alike
      const other = D.comps.filter((c) => !use.includes(c));
      g.strokeStyle = col; g.globalAlpha = Ds.length > 2 ? 0.1 : 0.18; g.lineWidth = 1;
      g.beginPath(); let on = false;
      D.v.forEach((v, i) => {
        if (v < x0 || v > x1 || !ok(D.y[i]) || !D.ok[i]) { on = false; return; }
        const y = (D.y[i] - other.reduce((a, c) => a + (at(D.vf, c.y, v) || 0), 0)) / pk;
        const X = P.X(v), Y = P.Y(y * grow);
        if (on) g.lineTo(X, Y); else { g.moveTo(X, Y); on = true; }
      });
      g.stroke(); g.globalAlpha = 1;
      const hot = LV.hoverLine === D.line;
      g.strokeStyle = col; g.lineWidth = hot ? 3 : 2;
      curve(g, D.vf, ref.map((y) => y / pk), P, grow);
      if (D.hm && grow > 0.6) {
        g.fillStyle = col;
        for (const v of D.hm.slice(0, 2)) { g.beginPath(); g.arc(P.X(v), P.Y(0.5), 3.5, 0, 7); g.fill(); g.strokeStyle = "#fff"; g.lineWidth = 1; g.stroke(); }
      }
      profs.push({ D, col, n: ref.map((y) => y / pk) });
    });
    g.restore();
    g.textAlign = "center"; g.textBaseline = "top"; g.font = `10px ${MONO}`; g.fillStyle = AXIS;
    const sx = niceStep(x1 - x0, 8);
    for (let v = Math.ceil(x0 / sx) * sx; v <= x1; v += sx) g.fillText(sx >= 1000 ? (v / 1000).toFixed(sx % 1000 ? 1 : 0) : String(Math.round(v)), P.X(v), P.b + 5);
    g.fillStyle = INK2; g.font = `11px ${SANS}`; g.fillText(sx >= 1000 ? "velocity (10³ km/s)" : "velocity (km/s)", (P.l + P.r) / 2, P.b + 19);
    LV.P = P;
    const tip = $("lv-tip");
    if (LV.hover && !LV.drag && LV.hover.x >= P.l && LV.hover.x <= P.r) {
      const X = LV.hover.x, v = x0 + (X - P.l) / (P.r - P.l) * (x1 - x0);
      g.strokeStyle = "rgba(29,33,40,.35)"; g.lineWidth = 1; g.beginPath(); g.moveTo(X, P.t); g.lineTo(X, P.b); g.stroke();
      tip.innerHTML = `<div class="lv-tiphead mono">${kms(v)} km/s</div>` + profs.map((p) => `<div><i style="background:${p.col}"></i>${esc(p.D.label)}<b class="mono">${(at(p.D.vf, p.n, v) || 0).toFixed(2)}</b></div>`).join("");
      tip.classList.remove("hidden");
      const tw = tip.offsetWidth; tip.style.left = `${X + 14 + tw > w ? X - tw - 14 : X + 14}px`; tip.style.top = `${P.t + 6}px`;
    } else tip.classList.add("hidden");
    dragBox(g, P, P);
  }

  // ---------------------------------------------------------------- one line across the epochs
  function epochsOf(E) {
    const eps = E.epochs.filter((e) => !e.missing);
    const ts = eps.map((e, i) => (ok(e.mjd) ? e.mjd : i)), t0 = Math.min(...ts), t1 = Math.max(...ts);
    eps.forEach((e, i) => { e.t = ts[i]; e.f = t1 > t0 ? (ts[i] - t0) / (t1 - t0) : 0; e.col = tcol(e.f); e.pk = Math.max(...e.y, 1e-12); });
    return eps;
  }
  // how much a quantity changes between epochs against its errors: rms scatter, the median error, and the
  // excess variance (Fvar) when the scatter is larger than the errors
  function variability(vals, errs) {
    const v = vals.filter(ok);
    if (v.length < 2) return null;
    const mean = v.reduce((a, b) => a + b, 0) / v.length;
    const rms = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (v.length - 1));
    const e = errs.filter(ok).sort((a, b) => a - b), med = e.length ? e[e.length >> 1] : null;
    const e2 = errs.filter(ok).reduce((a, b) => a + b * b, 0) / Math.max(1, errs.filter(ok).length);
    const ex = rms * rms - e2;
    return { mean, rms, med, fvar: ex > 0 && mean ? Math.sqrt(ex) / Math.abs(mean) : 0 };
  }
  function series(eps, key, title, unit) {
    const P = eps.filter((e) => ok(e[key]));
    if (P.length < 2) return "";
    const W = 300, H = 92, m = { l: 44, r: 8, t: 8, b: 18 };
    const lo = Math.min(...P.map((e) => e[key] - (e[key + "_err"] || 0))), hi = Math.max(...P.map((e) => e[key] + (e[key + "_err"] || 0)));
    const pad = (hi - lo) * 0.12 || Math.abs(hi) * 0.05 || 1, y0 = lo - pad, y1 = hi + pad;
    const t0 = Math.min(...P.map((e) => e.t)), t1 = Math.max(...P.map((e) => e.t));
    const X = (t) => m.l + (t1 > t0 ? (t - t0) / (t1 - t0) : 0.5) * (W - m.l - m.r), Y = (v) => m.t + (y1 - v) / (y1 - y0) * (H - m.t - m.b);
    const st = niceStep(y1 - y0, 3);
    let g = "";
    for (let v = Math.ceil(y0 / st) * st; v <= y1; v += st)
      g += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}" stroke="${GRID}"/><text x="${m.l - 5}" y="${(Y(v) + 3).toFixed(1)}" text-anchor="end">${+v.toPrecision(3)}</text>`;
    const V = variability(P.map((e) => e[key]), P.map((e) => e[key + "_err"]));
    if (V) g += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(V.mean).toFixed(1)}" y2="${Y(V.mean).toFixed(1)}" stroke="${FAINT}" stroke-dasharray="3 3"/>`;
    const hot = LV.plotEp ?? LV.hotEp ?? LV.pinEp;
    P.forEach((e) => {
      const x = X(e.t).toFixed(1), er = e[key + "_err"];
      if (ok(er)) g += `<line x1="${x}" x2="${x}" y1="${Y(e[key] - er).toFixed(1)}" y2="${Y(e[key] + er).toFixed(1)}" stroke="${e.col}" stroke-width="1.2" opacity=".6"/>`;
      g += `<circle data-ep="${e.k}" cx="${x}" cy="${Y(e[key]).toFixed(1)}" r="${hot === e.k ? 5.5 : 3.6}" fill="${e.col}" stroke="${hot === e.k ? INK : "#fff"}" stroke-width="${hot === e.k ? 1.6 : 1}"><title>#${e.k} ${esc(e.date || "")}: ${+e[key].toPrecision(4)}${ok(er) ? " ± " + +er.toPrecision(2) : ""}</title></circle>`;
    });
    const xl = ok(P[0].mjd) ? `MJD ${Math.round(t0)} – ${Math.round(t1)}` : "epoch";
    return `<div class="lv-ser"><div class="lv-serhead"><b>${title}</b>${V ? `<span>${V.fvar > 0 ? `varies by <b>${(V.fvar * 100).toFixed(1)}%</b> beyond the errors` : `<b>no change</b> beyond the errors`}</span>` : ""}</div>` +
      `<svg viewBox="0 0 ${W} ${H}">${g}<text x="${(W + m.l) / 2}" y="${H - 3}" text-anchor="middle">${xl}${unit ? " · " + unit : ""}</text></svg>` +
      (V ? `<div class="lv-sernote mono">mean ${+V.mean.toPrecision(4)} · rms ${+V.rms.toPrecision(2)} (${(V.rms / Math.abs(V.mean) * 100).toFixed(1)}%) · median error ${ok(V.med) ? +V.med.toPrecision(2) + ` (${(V.med / Math.abs(V.mean) * 100).toFixed(1)}%)` : "—"}</div>` : "") + `</div>`;
  }
  function epSide() {
    const E = LV.ED;
    if (!E || E.error) { $("lv-side").innerHTML = `<p class="note">${esc((E && E.error) || "")}</p>`; return; }
    const eps = epochsOf(E), miss = E.epochs.filter((e) => e.missing).length, unit = E.flux_unit ? unitText(E.flux_unit) : "";
    const hot = eps.find((e) => e.k === (LV.plotEp ?? LV.hotEp ?? LV.pinEp));
    $("lv-side").innerHTML = `<div class="lv-eyebrow"><span style="text-transform:none">${esc(E.label)}</span> in ${eps.length} epochs · ${E.of === "broad" ? "the broad profile" : "the whole line"}</div>` +
      `<div class="seg small lv-normseg"><button type="button" data-n="0" class="${LV.norm ? "" : "on"}">as fitted</button><button type="button" data-n="1" class="${LV.norm ? "on" : ""}">scaled to the peak</button></div>` +
      (E.shared ? `<div class="callout info small">${E.joint ? "The joint fit shares this line's shape between the epochs, so its width is the same in every epoch by construction and only its flux changes. To see the width change, fit each epoch on its own (Fit step) or free the shape in the Model step." : "The width is the same in every epoch."}</div>` : "") +
      (hot ? `<div class="lv-epnow"><span class="sw" style="background:${hot.col}"></span><b>#${hot.k}</b> ${esc(hot.date || "")}<span class="grow"></span><span class="mono">FWHM ${pm(hot.fwhm, hot.fwhm_err)} · flux ${pm(hot.flux, hot.flux_err)}</span></div>` : "") +
      stackCard(E.stack) +
      series(eps, "flux", "Flux", unit) + (E.shared ? "" : series(eps, "fwhm", "FWHM", "km/s")) + series(eps, "shift", "Centroid shift", "km/s") +
      (miss ? `<p class="note small">${miss} epoch${miss > 1 ? "s" : ""} without this line.</p>` : "") +
      `<p class="note small">Point at a curve or a dot for its epoch, click to keep it; drag across the plot to zoom.</p>`;
    $("lv-side").querySelectorAll(".lv-normseg button").forEach((b) => (b.onclick = () => { LV.norm = b.dataset.n === "1"; side(); kick(); }));
    $("lv-side").querySelectorAll("circle[data-ep]").forEach((c) => {
      c.onmouseenter = () => { LV.hotEp = +c.dataset.ep; kick(); };
      c.onmouseleave = () => { LV.hotEp = null; kick(); };
      c.onclick = () => { LV.pinEp = LV.pinEp === +c.dataset.ep ? null : +c.dataset.ep; side(); kick(); };
    });
  }
  // a step in velocity wider than 3 typical steps: masked pixels, where a data curve breaks
  const gapOf = (v) => { const d = v.slice(1).map((x, i) => x - v[i]).sort((a, b) => a - b); return d.length ? 3 * d[d.length >> 1] : Infinity; };
  const peakOf = (y) => y.reduce((a, v) => (v != null && v > a ? v : a), 0);
  function stackCard(S) {
    if (!S) return "";
    const w = (x) => (ok(x) ? Math.round(x).toLocaleString("en-US") : "—");
    const row = (k, t, col, dash) => { const d = S[k]; return `<tr><td><i class="lv-sw${dash ? " dash" : ""}" style="--c:${col}"></i>${t}</td><td class="mono">${w(d.fwhm)}</td><td class="mono">${w(d.sigma)}</td><td class="mono">${ok(d.fwhm) && ok(d.sigma) ? (d.fwhm / d.sigma).toFixed(2) : "—"}</td></tr>`; };
    return `<div class="lv-stack"><div class="lv-serhead"><b>Mean and rms profiles</b><span>${S.n} epochs</span></div>` +
      `<table><thead><tr><th></th><th>FWHM</th><th>σ<sub>line</sub></th><th>FWHM/σ</th></tr></thead><tbody>${row("mean", "mean", INK, true)}${row("rms", "rms", RMS, false)}</tbody></table>` +
      `<p class="note small">km/s, of the fitted profiles. The rms profile is the part of the line that varies; its σ<sub>line</sub> is the width reverberation mapping uses for the black hole mass.</p></div>`;
  }
  let epSideHot = null;
  function drawEp(grow) {
    const E = LV.ED, { g, w, h } = canvas();
    if (!E || E.error) return;
    const eps = epochsOf(E), vf = E.vf;
    const [x0, x1] = LV.zoom || [-E.window, E.window];
    const P = { l: 58, r: w - 14, t: 22, b: h - 40, rb: h - 40, x0, x1 };
    const inX = (v) => v >= x0 && v <= x1;
    const sc = (e) => (LV.norm ? 1 / e.pk : 1);
    let top = 0;
    eps.forEach((e) => e.y.forEach((y, i) => { if (inX(vf[i])) top = Math.max(top, y * sc(e)); }));
    if (E.stack && !LV.norm) ["mean", "rms"].forEach((k) => { if (!LV.hidden.has("_" + k)) E.stack[k].y.forEach((y, i) => { if (y != null && inX(vf[i])) top = Math.max(top, y); }); });
    if (!LV.hidden.has("data")) eps.forEach((e) => { const d = e.dy.filter((y, i) => inX(e.dv[i])).map((y) => y * sc(e)).sort((a, b) => a - b); if (d.length) top = Math.max(top, d[Math.floor(0.98 * (d.length - 1))]); });
    const y0 = -0.06 * (top || 1), y1 = (top || 1) * 1.12;
    P.X = (v) => P.l + (v - x0) / (x1 - x0) * (P.r - P.l);
    P.Y = (y) => P.b - (y - y0) / (y1 - y0) * (P.b - P.t);
    axes(g, P, x0, x1, y0, y1, LV.norm ? "flux / peak" : "flux", null);
    g.setLineDash([3, 3]); g.strokeStyle = FAINT; g.beginPath(); g.moveTo(P.X(0), P.t); g.lineTo(P.X(0), P.b); g.stroke(); g.setLineDash([]);
    // the epoch under the pointer: the curve nearest to it
    let near = null;
    if (LV.hover && !LV.drag && LV.hover.x >= P.l && LV.hover.x <= P.r) {
      const v = x0 + (LV.hover.x - P.l) / (P.r - P.l) * (x1 - x0);
      let best = 1e9;
      eps.forEach((e) => { const y = at(vf, e.y, v); if (ok(y)) { const d = Math.abs(P.Y(y * sc(e)) - LV.hover.y); if (d < best) { best = d; near = e; } } });
      if (best > 30) near = null;
    }
    const hotK = near ? near.k : LV.hotEp ?? LV.pinEp;
    LV.plotEp = near ? near.k : null;
    if (epSideHot !== hotK) { epSideHot = hotK; side(); }
    g.save(); g.beginPath(); g.rect(P.l, P.t - 2, P.r - P.l, P.b - P.t + 2); g.clip();
    if (!LV.hidden.has("data")) eps.forEach((e) => {
      g.strokeStyle = e.col; g.globalAlpha = hotK != null ? (e.k === hotK ? 0.45 : 0.04) : Math.max(0.05, 0.5 / eps.length); g.lineWidth = 1;
      g.beginPath(); let on = false;
      const gp = gapOf(e.dv);
      e.dv.forEach((v, i) => { if (!inX(v) || (i && v - e.dv[i - 1] > gp)) on = false; if (!inX(v)) return; const X = P.X(v), Y = P.Y(e.dy[i] * sc(e) * grow); if (on) g.lineTo(X, Y); else { g.moveTo(X, Y); on = true; } });
      g.stroke();
    });
    g.globalAlpha = 1;
    const order = eps.slice().sort((a, b) => (a.k === hotK) - (b.k === hotK));
    order.forEach((e) => {
      const hot = e.k === hotK;
      g.strokeStyle = e.col; g.lineWidth = hot ? 3 : eps.length > 20 ? 1 : 1.6;
      g.globalAlpha = hotK != null && !hot ? 0.22 : eps.length > 20 ? 0.7 : 0.9;
      curve(g, vf, e.y.map((y) => y * sc(e)), P, grow);
    });
    g.globalAlpha = 1;
    // the mean and the rms over the epochs (scaled to their own peaks with "scaled to the peak")
    const S = E.stack;
    if (S) for (const [key, col, dash, lw] of [["mean", INK, [6, 4], 1.4], ["rms", RMS, [], 2.2]]) {
      if (LV.hidden.has("_" + key)) continue;
      const sk = LV.norm ? 1 / (peakOf(S[key].y) || 1) : 1;
      if (!LV.hidden.has("data")) {
        const dy = S[key === "mean" ? "dmean" : "drms"];
        g.strokeStyle = col; g.globalAlpha = hotK != null ? 0.12 : 0.35; g.lineWidth = 1; g.beginPath(); let on = false;
        const gp = gapOf(S.dv);
        S.dv.forEach((v, i) => { const y = dy[i]; if (!inX(v) || y == null || (i && v - S.dv[i - 1] > gp)) { on = false; if (!inX(v) || y == null) return; } const X = P.X(v), Y = P.Y(y * sk * grow); if (on) g.lineTo(X, Y); else { g.moveTo(X, Y); on = true; } });
        g.stroke(); g.globalAlpha = 1;
      }
      g.strokeStyle = col; g.lineWidth = lw; g.setLineDash(dash); g.globalAlpha = hotK != null ? 0.5 : 1;
      curve(g, vf, S[key].y.map((y) => (y == null ? NaN : y * sk)), P, grow); g.setLineDash([]); g.globalAlpha = 1;
    }
    g.restore();
    const he = eps.find((e) => e.k === hotK);
    if (he && he.hm && grow > 0.6) {
      const [vl, vr, half] = he.hm, Y = P.Y(half * sc(he));
      g.strokeStyle = INK; g.fillStyle = he.col; g.lineWidth = 1.2;
      g.beginPath(); g.moveTo(P.X(vl), Y); g.lineTo(P.X(vr), Y); g.stroke();
      for (const v of [vl, vr]) { g.beginPath(); g.arc(P.X(v), Y, 3.5, 0, 7); g.fill(); g.strokeStyle = "#fff"; g.stroke(); g.strokeStyle = INK; }
    }
    g.textAlign = "center"; g.textBaseline = "top"; g.font = `10px ${MONO}`; g.fillStyle = AXIS;
    const sx = niceStep(x1 - x0, 8);
    for (let v = Math.ceil(x0 / sx) * sx; v <= x1; v += sx) g.fillText(sx >= 1000 ? (v / 1000).toFixed(sx % 1000 ? 1 : 0) : String(Math.round(v)), P.X(v), P.b + 5);
    g.fillStyle = INK2; g.font = `11px ${SANS}`; g.fillText(sx >= 1000 ? "velocity (10³ km/s)" : "velocity (km/s)", (P.l + P.r) / 2, P.b + 19);
    LV.P = P;
    const tip = $("lv-tip");
    if (near) {
      const X = LV.hover.x;
      g.strokeStyle = "rgba(29,33,40,.3)"; g.lineWidth = 1; g.beginPath(); g.moveTo(X, P.t); g.lineTo(X, P.b); g.stroke();
      tip.innerHTML = `<div class="lv-tiphead mono"><i style="background:${near.col}"></i> #${near.k} <span class="muted">${esc(near.date || "")}</span></div>` +
        `<div>FWHM<b class="mono">${ok(near.fwhm) ? Math.round(near.fwhm) : "—"}${ok(near.fwhm_err) ? " ± " + Math.round(near.fwhm_err) : ""}</b></div>` +
        `<div>flux<b class="mono">${ok(near.flux) ? +near.flux.toPrecision(4) : "—"}${ok(near.flux_err) ? " ± " + +near.flux_err.toPrecision(2) : ""}</b></div>` +
        `<div>shift<b class="mono">${kms(near.shift)}</b></div>`;
      tip.classList.remove("hidden");
      const tw = tip.offsetWidth; tip.style.left = `${X + 14 + tw > w ? X - tw - 14 : X + 14}px`; tip.style.top = `${P.t + 6}px`;
    } else tip.classList.add("hidden");
    dragBox(g, P, P);
  }

  // ---------------------------------------------------------------- events
  function init() {
    const cv = $("lv-plot");
    if (!cv || cv.dataset.on) return;
    cv.dataset.on = "1";
    cv.addEventListener("mousemove", (e) => { LV.hover = { x: e.offsetX, y: e.offsetY }; if (LV.drag) LV.drag.x1 = e.offsetX; kick(); });
    cv.addEventListener("mouseleave", () => { LV.hover = null; kick(); });
    cv.addEventListener("mousedown", (e) => { LV.drag = { x0: e.offsetX, x1: e.offsetX }; });
    window.addEventListener("mouseup", () => {
      const d = LV.drag; LV.drag = null;
      if (d && LV.P && Math.abs(d.x1 - d.x0) > 8) {
        const P = LV.P, v = (X) => P.x0 + (Math.max(P.l, Math.min(P.r, X)) - P.l) / (P.r - P.l) * (P.x1 - P.x0);
        const a = v(Math.min(d.x0, d.x1)), b = v(Math.max(d.x0, d.x1));
        if (b - a > 60) { LV.zoom = [a, b]; LV.zoomed = true; }
      }
      kick();
    });
    cv.addEventListener("dblclick", () => { LV.zoom = null; kick(); });
    cv.addEventListener("click", () => { if (LV.mode === "ep" && LV.plotEp != null && !LV.zoomed) { LV.pinEp = LV.pinEp === LV.plotEp ? null : LV.plotEp; side(); kick(); } LV.zoomed = false; });
    $("lv-mode").querySelectorAll("button").forEach((b) => (b.onclick = () => setMode(b.dataset.m)));
    new ResizeObserver(() => kick()).observe(cv);
    // the figure for a paper starts from the lines shown above (lineSel, lineKey: wf.js)
    $("lv-paper").addEventListener("toggle", () => {
      if (!$("lv-paper").open || typeof showLines !== "function") return;
      const want = LV.mode === "cmp" ? LV.cmp.slice() : [LV.focus];
      if (want.every(Boolean) && want.join() !== lineSel.join()) {
        const order = ((LV.list && LV.list.lines) || []).map((d) => d.line);
        lineSel = want.sort((a, b) => order.indexOf(a) - order.indexOf(b)); lineKey = null;
        const b = document.querySelector(`#lo-modeseg [data-mode="${LV.mode === "cmp" ? "overlay" : "grid"}"]`);
        if (b) b.click(); else showLines();
      } else showLines();
    });
  }
  const state = () => ({ mode: LV.mode, line: LV.focus, lines: LV.cmp.join(","), norm: LV.norm ? "1" : "0", xmin: LV.zoom ? Math.round(LV.zoom[0]) : "", xmax: LV.zoom ? Math.round(LV.zoom[1]) : "" });
  // from a line label of the Results plot: show that line alone (taken up by the next show)
  const open = (line) => { LV.focus = line; LV.zoom = null; LV.hidden.clear(); if (LV.list) setMode("one"); else LV.mode = "one"; };
  window.LV = { show: (k, cells) => { init(); LV.cmpData = null; return show(k, cells); }, kick, state, open };
})();
