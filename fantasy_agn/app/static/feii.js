// Results › Fe II: the Fe II of the shown fit, group by group (GET /api/wf/feii, StepsMixin.wf_feii_view).
// The spectrum with everything but Fe II subtracted (or the full spectrum with the Fe II stacked on the rest of
// the model), every transition at its fitted strength, the groups, a Grotrian diagram from NIST levels, one Fe II
// line against the broad reference line, and the quasar main sequence. Uses api, esc and unitText of wf.js.
(() => {
  const $ = (id) => document.getElementById(id);
  const C = 299792.458, HC = 12398.42;
  const INK = "#1d2128", GRID = "#eceef1", AXIS = "#7b8290", FAINT = "#a3a9b4";
  const MONO = '"IBM Plex Mono", "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace';
  const SANS = '"IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Helvetica, Arial, sans-serif';
  // the logo's ribbon: violet → blue → teal → green → gold → orange → red
  const STOPS = [[106, 79, 216], [47, 127, 224], [31, 158, 154], [31, 158, 106], [217, 178, 31], [224, 112, 42], [212, 71, 42]];
  const ramp = (u) => { u = Math.min(Math.max(u, 0), 1) * (STOPS.length - 1); const i = Math.min(Math.floor(u), STOPS.length - 2), f = u - i;
    return STOPS[i].map((a, k) => Math.round(a + (STOPS[i + 1][k] - a) * f)); };
  const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  const ease = (t) => 1 - Math.pow(1 - Math.min(Math.max(t, 0), 1), 3);
  const ok = (v) => v != null && isFinite(v);
  const fmt = (v, d) => ok(v) ? v.toFixed(d) : "—";
  const pm = (v, e, d) => ok(v) ? `${v.toFixed(d)}${ok(e) ? ` <em>± ${e.toFixed(d)}</em>` : ""}` : "—";
  const SUPD = { 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  // terms: 'b4F' → b⁴F, 'z4D*' → z⁴D°, '2[6]*' → 2[6]°; html with <sup>, text with superscript digits
  const TERM = /^([a-z]?)(\d)([SPDFGHIKLM])(\d?)(\*?)$/, JK = /^(\d)\[(\d+(?:\/2)?)\](\*?)$/;
  function termParts(t) {
    let m = TERM.exec(t);
    if (m) return { pre: m[1], mult: +m[2], L: m[3], idx: m[4], odd: !!m[5] || /^[u-z]$/.test(m[1]) };
    m = JK.exec(t);
    if (m) return { jk: true, pre: "", mult: +m[1], L: `[${m[2]}]`, idx: "", odd: !!m[3] };
    return null;
  }
  const termText = (t) => { const p = termParts(t); return p ? `${p.pre}${p.jk ? p.mult : SUPD[p.mult]}${p.L}${p.idx}${p.odd ? "°" : ""}` : t; };
  const termHtml = (t) => { const p = termParts(t); return p ? `${p.pre}${p.jk ? p.mult : `<sup>${p.mult}</sup>`}${p.L}${p.idx ? `<small>${p.idx}</small>` : ""}${p.odd ? "°" : ""}` : esc(t); };
  const fluxUnit = (u) => (u ? unitText(u) : "data units");

  const FX = { key: null, D: null, mode: "alone", zoom: true, view: null, from: null, to: null, tView: 0, born: 0, pin: null, hot: null, mouse: null, L: {}, raf: 0, grot: null };

  // ---------------------------------------------------------------- data
  function prepare(D) {
    const N = D.wave.length, X = D.wave, band = D.strength.band;
    const G = D.groups.map((g, i) => {
      const ls = D.lines.filter((l) => l[1] === i), w = ls.reduce((s, l) => s + l[3], 0);
      const centre = w > 0 ? ls.reduce((s, l) => s + l[0] * l[3], 0) / w : (g.wmin + g.wmax) / 2;
      return { ...g, i, lines: ls, centre, sn: ok(g.err) && g.err > 0 && ok(g.amp) ? g.amp / g.err : null, on: !!g.flux && g.total > 0 };
    });
    const live = G.filter((g) => g.on).sort((a, b) => a.centre - b.centre);
    live.forEach((g, k) => (g.color = ramp(live.length > 1 ? k / (live.length - 1) : 0.5)));
    G.filter((g) => !g.on).forEach((g) => (g.color = [178, 184, 194]));
    const stack = live.map(() => new Float32Array(N));
    for (let j = 0; j < N; j++) { let s = 0; live.forEach((g, k) => { s += g.flux[j] || 0; stack[k][j] = s; }); }
    const resid = D.data.map((d, j) => (d == null || D.other[j] == null ? null : d - D.other[j]));
    // where the other emission lines outweigh Fe II against the noise, the subtracted data are uncertain
    const blend = D.emis.map((e, j) => Math.min(1, Math.max(0, (e - 0.6 * Math.max(D.feii[j], 0)) / (2 * (D.err[j] || 1e-9))) / 3));
    const bandTot = live.reduce((s, g) => s + g.band, 0);
    const kind = D.kind;
    const presets = (kind === "optical" ? [["Hβ region", 4150, 5450], ["Hα region", 5900, 6850]]
      : kind === "uv" ? [["Mg II region", 2600, 3100]] : [["UV Fe II", 2200, 3100], ["Mg II region", 2600, 3100]])
      .filter(([, a, b]) => Math.min(b, X[N - 1]) - Math.max(a, X[0]) > 0.5 * (b - a));
    return { ...D, N, X, band, G, live, stack, resid, blend, bandTot, presets };
  }
  const lo = (X, x) => { let a = 0, b = X.length - 1; while (b - a > 1) { const m = (a + b) >> 1; if (X[m] < x) a = m; else b = m; } return a; };

  async function show(k, key) {
    if (FX.key === key && FX.D) { FX.born = performance.now() - 5000; layout(); return; }
    FX.key = key; FX.D = null; FX.pin = FX.hot = null;
    $("fx-empty").classList.add("hidden");
    $("res-feii").querySelectorAll(".fx-kpis, .fx-main, .fx-lower").forEach((el) => el.classList.add("fx-loading"));
    let D;
    try { D = await api(`/api/wf/feii?k=${k}`); } catch (e) { D = { error: String(e.message || e) }; }
    if (FX.key !== key) return;                        // another fit was opened meanwhile
    $("res-feii").querySelectorAll(".fx-loading").forEach((el) => el.classList.remove("fx-loading"));
    if (!D || D.error) {
      $("res-feii").querySelectorAll(".fx-kpis, .fx-main, .fx-lower").forEach((el) => el.classList.add("hidden"));
      $("fx-empty").innerHTML = `<p><b>No Fe II to show.</b></p><p class="muted small">${esc((D && D.error) || "no fit")}</p>`;
      $("fx-empty").classList.remove("hidden");
      return;
    }
    $("res-feii").querySelectorAll(".fx-kpis, .fx-main, .fx-lower").forEach((el) => el.classList.remove("hidden"));
    FX.D = prepare(D);
    const P = FX.D.presets;
    FX.view = P.length ? [P[0][1], P[0][2]] : [FX.D.X[0], FX.D.X[FX.D.N - 1]];
    FX.tView = 0; FX.born = performance.now();
    $("fx-ranges").innerHTML = P.map(([n, a, b], q) => `<button data-r="${a},${b}" class="${q ? "" : "on"}">${n}</button>`).join("") +
      `<button data-r="full" class="${P.length ? "" : "on"}">Whole fit</button>`;
    $("fx-ranges").querySelectorAll("button").forEach((bt) => (bt.onclick = () => {
      $("fx-ranges").querySelectorAll("button").forEach((q) => q.classList.toggle("on", q === bt));
      goTo(bt.dataset.r === "full" ? [FX.D.X[0], FX.D.X[FX.D.N - 1]] : bt.dataset.r.split(",").map(Number));
    }));
    kpis(); layout();
  }

  // ---------------------------------------------------------------- numbers
  const msType = (r, f) => f < 4000 ? (r < 0.5 ? "A1" : r < 1 ? "A2" : r < 1.5 ? "A3" : "A4")
    : f < 8000 ? (r < 0.5 ? "B1" : "B2") : f < 12000 ? "B1+" : "B1++";
  function hasMS() { const D = FX.D, s = D.strength; return s.reference === "Hb" && ok(s.r) && D.ref && ok(D.ref.fwhm); }
  function kpis() {
    const D = FX.D, s = D.strength, k = D.kin, ref = D.ref;
    const rl = s.ratio_label === "R_FeII" ? "R<sub>FeII</sub>" : `Fe II / ${ref ? esc(ref.label) : "Mg II"}`;
    const bandTxt = `${s.band[0].toFixed(0)}–${s.band[1].toFixed(0)} Å`;
    const cov = s.covered ? "" : " (band not covered)";
    const tied = (st) => (st && st !== "free" ? ` · ${esc(st)}` : "");
    const type = hasMS() ? msType(s.r, ref.fwhm) : null;
    const tiles = [
      [rl, pm(s.r, s.r_err, 2), `Fe II ${bandTxt} / broad ${ref ? esc(ref.label) : s.reference}${cov}`, true],
      [`Fe II flux (${bandTxt})`, pm(s.flux, s.flux_err, 0), esc(fluxUnit(D.flux_unit).replace(" Å⁻¹", ""))],
      ["equivalent width", pm(s.ew, s.ew_err, 1), "Å, against the AGN continuum"],
      ["Fe II FWHM", pm(k.fwhm, k.fwhm_err, 0), `km/s${ref && ok(ref.fwhm) ? ` · broad ${esc(ref.label)} ${ref.fwhm.toFixed(0)}` : ""}${tied(k.fwhm_status)}`],
      ["Fe II shift", ok(k.shift) ? (k.shift > 0 ? "+" : "") + pm(k.shift, k.shift_err, 0) : "—", `km/s, + is redward${tied(k.shift_status)}`],
      type ? ["main sequence", type, `Population ${type[0]} · ${D.live.length} of ${D.G.length} groups`]
        : ["groups", `${D.live.length}<em> of ${D.G.length}</em>`, "with Fe II in this fit"],
    ];
    $("fx-kpis").innerHTML = tiles.map(([l, v, u, h]) => `<div class="fx-kpi${h ? " lead" : ""}"><small>${l}</small><b>${v}</b><span>${u}</span></div>`).join("");
    $("fx-sidetxt").innerHTML = `Lines that share a lower term vary together. Share of the Fe II flux in ${bandTxt}, and the S/N of the strength. Point to highlight, click to pin.`;
  }

  // ---------------------------------------------------------------- canvas helpers
  function setup(cv) {
    const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    const w = Math.round(r.width * dpr), h = Math.round(r.height * dpr);
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, r.width, r.height);
    return [g, r.width, r.height];
  }
  function ticks(a, b, n) {
    const s0 = (b - a) / Math.max(n, 1), p = Math.pow(10, Math.floor(Math.log10(s0))), m = s0 / p;
    const st = (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p, out = [];
    for (let v = Math.ceil(a / st) * st; v <= b + 1e-9; v += st) out.push(+v.toFixed(10));
    return out;
  }
  const focus = () => FX.pin ?? FX.hot;
  const path = (g, n, fx, fy) => { for (let q = 0; q < n; q++) { const x = fx(q), y = fy(q); q ? g.lineTo(x, y) : g.moveTo(x, y); } };

  // ---------------------------------------------------------------- the main plot
  function drawPlot(now) {
    const D = FX.D, cv = $("fx-plot");
    if (!D || !cv.offsetParent) return false;
    const [g, W, H] = setup(cv), full = FX.mode === "full", showData = $("fx-data").checked;
    const pad = { l: 50, r: 10, t: 22, b: 40 }, stickH = 64, gap = 16;
    const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b - stickH - gap;
    const [a, b] = FX.view, X = D.X, i0 = Math.max(0, lo(X, a) - 1), i1 = Math.min(D.N - 1, lo(X, b) + 2);
    const age = (now - FX.born) / 1000, f = focus();
    // y range
    let ymax = 0, ymin = 0;
    if (full && FX.zoom) {
      // scaled to the Fe II: from below the continuum to the top of the Fe II and the data away from the
      // strong lines, which run off the top
      const vals = []; let cmin = Infinity;
      for (let j = i0; j <= i1; j++) {
        const base = D.other[j] || 0, cont = base - (D.emis[j] || 0);
        if (cont > 0) cmin = Math.min(cmin, cont);
        if (D.blend[j] < 0.3) { vals.push(base + D.feii[j]); if (showData && D.data[j] != null) vals.push(D.data[j]); }
      }
      vals.sort((p, q) => p - q);
      const top = vals.length ? vals[Math.floor(0.995 * (vals.length - 1))] : 1;
      ymin = isFinite(cmin) ? 0.85 * cmin : 0; ymax = ymin + (top - ymin) * 1.12;
    } else for (let j = i0; j <= i1; j++) {
      const top = full ? (D.other[j] || 0) + D.feii[j] : D.feii[j] * 1.15;
      ymax = Math.max(ymax, top);
      if (showData) {
        if (full && D.data[j] != null) ymax = Math.max(ymax, D.data[j]);
        if (!full && D.resid[j] != null && D.blend[j] < 0.5) { ymax = Math.max(ymax, D.resid[j]); ymin = Math.min(ymin, D.resid[j]); }
      }
    }
    if (!(full && FX.zoom)) { ymax = (ymax || 1) * 1.08; ymin = Math.max(ymin, -0.25 * ymax); }
    const xs = (x) => pad.l + (x - a) / (b - a) * pw, ys = (y) => pad.t + ph - (y - ymin) / (ymax - ymin) * ph;
    FX.L = { pad, pw, ph, xs, ys, a, b, stickTop: pad.t + ph + gap, W };
    // grid and axes
    g.font = `11px ${MONO}`; g.lineWidth = 1;
    ticks(ymin, ymax, 5).forEach((v) => { const y = ys(v); g.strokeStyle = v === 0 ? "#d5d9df" : GRID; g.beginPath(); g.moveTo(pad.l, y); g.lineTo(W - pad.r, y); g.stroke();
      g.fillStyle = AXIS; g.textAlign = "right"; g.textBaseline = "middle"; g.fillText(+v.toPrecision(3), pad.l - 6, y); });
    g.textAlign = "center"; g.textBaseline = "top";
    const xt = ticks(a, b, Math.max(3, Math.floor(pw / 90)));
    xt.forEach((v) => { const x = xs(v); g.strokeStyle = GRID; g.beginPath(); g.moveTo(x, pad.t); g.lineTo(x, pad.t + ph); g.stroke(); g.fillStyle = AXIS; g.fillText(v.toFixed(0), x, H - pad.b + 8); });
    g.textAlign = "right"; g.fillText("rest wavelength (Å)", W - pad.r, H - 14);
    g.save(); g.translate(12, pad.t + ph / 2); g.rotate(-Math.PI / 2); g.textAlign = "center"; g.fillText(`F_λ  ${fluxUnit(D.flux_unit)}`, 0, 0); g.restore();
    // the band of the Fe II strength: its title above the plot
    const bx0 = xs(D.band[0]), bx1 = xs(D.band[1]), bandIn = bx1 > pad.l && bx0 < W - pad.r;
    if (bandIn) {
      g.fillStyle = "#1a7f86"; g.textAlign = "center"; g.textBaseline = "bottom"; g.font = `600 11px ${SANS}`;
      g.fillText(D.strength.ratio_label === "R_FeII" ? "R_FeII band" : "Fe II band", Math.min(Math.max((bx0 + bx1) / 2, pad.l + 40), W - pad.r - 40), pad.t - 5);
    }
    g.save(); g.beginPath(); g.rect(pad.l, pad.t, pw, ph); g.clip();          // the spectrum stays in its frame
    if (bandIn) {
      const gr = g.createLinearGradient(0, pad.t, 0, pad.t + ph); gr.addColorStop(0, "rgba(31,158,154,.10)"); gr.addColorStop(1, "rgba(31,158,154,.02)");
      g.fillStyle = gr; g.fillRect(bx0, pad.t, bx1 - bx0, ph);
      g.strokeStyle = "rgba(31,158,154,.45)"; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(bx0, pad.t); g.lineTo(bx0, pad.t + ph); g.moveTo(bx1, pad.t); g.lineTo(bx1, pad.t + ph); g.stroke(); g.setLineDash([]);
    }
    // where other lines dominate (only meaningful for the subtracted data)
    if (!full && showData) for (let j = i0; j < i1; j++) if (D.blend[j] > 0.05) { g.fillStyle = `rgba(123,130,144,${0.05 * D.blend[j]})`; g.fillRect(xs(X[j]), pad.t, Math.max(1, xs(X[j + 1]) - xs(X[j])) + 0.5, ph); }
    // labels of the other lines, none overlapping
    g.font = `11px ${SANS}`; let lastX = -1e9, lastName = "";
    D.labels.forEach(([n, w]) => {
      if (w < a || w > b) return;
      const x = xs(w), tw = g.measureText(n).width;
      g.strokeStyle = "rgba(123,130,144,.3)"; g.setLineDash([2, 4]); g.beginPath(); g.moveTo(x, pad.t + 14); g.lineTo(x, pad.t + ph); g.stroke(); g.setLineDash([]);
      if (x - tw / 2 < lastX + 6 || (n === lastName && x - lastX < 60)) return;
      g.fillStyle = FAINT; g.textAlign = "center"; g.textBaseline = "top"; g.fillText(n, x, pad.t + 1); lastX = x + tw / 2; lastName = n;
    });
    // the rest of the model under the Fe II (full spectrum)
    const base = (j) => (full ? D.other[j] || 0 : 0);
    if (full) {
      g.beginPath(); path(g, i1 - i0 + 1, (q) => xs(X[i0 + q]), (q) => ys(base(i0 + q))); g.lineTo(xs(X[i1]), ys(ymin)); g.lineTo(xs(X[i0]), ys(ymin)); g.closePath();
      g.fillStyle = "rgba(123,130,144,.10)"; g.fill();
      g.beginPath(); path(g, i1 - i0 + 1, (q) => xs(X[i0 + q]), (q) => ys(base(i0 + q))); g.strokeStyle = "rgba(74,81,96,.55)"; g.lineWidth = 1; g.stroke();
    }
    // the groups, stacked from the bluest up, growing in one after another
    for (let k = D.live.length - 1; k >= 0; k--) {
      const grp = D.live[k], grow = ease((age - 0.15 - k * 0.07) / 0.7);
      if (grow <= 0) continue;
      const top = D.stack[k], bot = k ? D.stack[k - 1] : null, dim = f != null, c = grp.color;
      g.beginPath();
      for (let j = i0; j <= i1; j++) { const x = xs(X[j]), y = ys(base(j) + top[j] * grow); j === i0 ? g.moveTo(x, y) : g.lineTo(x, y); }
      for (let j = i1; j >= i0; j--) g.lineTo(xs(X[j]), ys(base(j) + (bot ? bot[j] * grow : 0)));
      g.closePath();
      const gr = g.createLinearGradient(0, ys(ymax * 0.6), 0, ys(ymin));
      gr.addColorStop(0, rgba(c, dim ? 0.10 : 0.78)); gr.addColorStop(1, rgba(c, dim ? 0.06 : 0.5));
      g.fillStyle = gr; g.fill(); g.strokeStyle = `rgba(255,255,255,${dim ? 0.2 : 0.55})`; g.lineWidth = 0.6; g.stroke();
    }
    // a group in focus: alone, from the base, so its own shape reads
    if (f != null && D.G[f].on) {
      const grp = D.G[f], c = grp.color, line = new Path2D();
      for (let j = i0; j <= i1; j++) { const x = xs(X[j]), y = ys(base(j) + grp.flux[j]); j === i0 ? line.moveTo(x, y) : line.lineTo(x, y); }
      const area = new Path2D(line); for (let j = i1; j >= i0; j--) area.lineTo(xs(X[j]), ys(base(j))); area.closePath();
      const gr = g.createLinearGradient(0, ys(ymax * 0.6), 0, ys(ymin)); gr.addColorStop(0, rgba(c, 0.85)); gr.addColorStop(1, rgba(c, 0.35));
      g.fillStyle = gr; g.fill(area); g.strokeStyle = rgba(c, 1); g.lineWidth = 2; g.stroke(line);
    }
    // the data: subtracted, with the error band, or the observed spectrum
    const dshow = ease((age - 1.2) / 0.6);
    if (showData && dshow > 0) {
      g.globalAlpha = dshow;
      const val = (j) => (full ? D.data[j] : D.resid[j]);
      const up = [], dn = [];
      for (let j = i0; j <= i1; j++) if (val(j) != null) { up.push([xs(X[j]), ys(val(j) + D.err[j])]); dn.push([xs(X[j]), ys(val(j) - D.err[j])]); }
      g.fillStyle = "rgba(29,33,40,.07)"; g.beginPath(); up.forEach(([x, y], q) => (q ? g.lineTo(x, y) : g.moveTo(x, y))); dn.reverse().forEach(([x, y]) => g.lineTo(x, y)); g.closePath(); g.fill();
      g.lineWidth = 0.9; g.lineJoin = "round";
      for (let j = i0; j < i1; j++) {
        if (val(j) == null || val(j + 1) == null) continue;
        g.strokeStyle = full ? "rgba(29,33,40,.6)" : `rgba(74,81,96,${0.5 - 0.38 * Math.max(D.blend[j], D.blend[j + 1])})`;
        g.beginPath(); g.moveTo(xs(X[j]), ys(val(j))); g.lineTo(xs(X[j + 1]), ys(val(j + 1))); g.stroke();
      }
      g.globalAlpha = 1;
    }
    // the total Fe II (or the full model)
    const tot = ease((age - 0.15 - D.live.length * 0.07) / 0.6);
    if (tot > 0) {
      g.beginPath(); path(g, i1 - i0 + 1, (q) => xs(X[i0 + q]), (q) => ys(base(i0 + q) + D.feii[i0 + q]));
      g.globalAlpha = tot; g.strokeStyle = full ? "#d4472a" : INK; g.lineWidth = full ? 1.3 : 1.6; g.stroke(); g.globalAlpha = 1;
    }
    g.restore(); g.save(); g.beginPath(); g.rect(pad.l, 0, pw, H); g.clip();
    // every transition: a stick at its fitted peak height (square root, for the dynamic range)
    const sy0 = FX.L.stickTop + stickH;
    g.strokeStyle = "#d5d9df"; g.lineWidth = 1; g.beginPath(); g.moveTo(pad.l, sy0 + 0.5); g.lineTo(W - pad.r, sy0 + 0.5); g.stroke();
    const pmax = Math.max(...D.lines.map((l) => l[2]), 1e-30);
    let nIn = 0;
    D.lines.forEach((l) => {
      if (l[0] < a || l[0] > b) return;
      nIn++;
      const grp = D.G[l[1]], x = xs(l[0]), fall = ease((age - 0.6 - (l[0] - a) / (b - a) * 0.8) / 0.5);
      if (fall <= 0) return;
      const h = Math.sqrt(l[2] / pmax) * (stickH - 6) * fall, dim = f != null && f !== l[1], hi = f === l[1];
      g.strokeStyle = rgba(grp.color, dim ? 0.15 : 0.9); g.lineWidth = hi ? 2.2 : 1.4;
      g.beginPath(); g.moveTo(x, sy0); g.lineTo(x, sy0 - h); g.stroke();
      g.fillStyle = rgba(grp.color, dim ? 0.2 : 1); g.beginPath(); g.arc(x, sy0 - h, hi ? 2.6 : 1.8, 0, 7); g.fill();
    });
    g.fillStyle = AXIS; g.textAlign = "left"; g.textBaseline = "top"; g.font = `11px ${SANS}`;
    g.fillText(`${nIn} Fe II transitions in view · height √(peak)`, pad.l + 4, FX.L.stickTop - 2);
    if (FX.mouse && FX.mouse.x > pad.l && FX.mouse.x < W - pad.r) { g.strokeStyle = "rgba(29,33,40,.35)"; g.lineWidth = 1; g.beginPath(); g.moveTo(FX.mouse.x + 0.5, pad.t); g.lineTo(FX.mouse.x + 0.5, sy0); g.stroke(); }
    g.restore();
    $("fx-sub").textContent = full
      ? `The spectrum (dark), the rest of the model (grey) and the fitted Fe II stacked on it by group; the red line is the full model.${FX.zoom ? " Scaled to the Fe II: the strong lines run off the top." : ""} Below, every transition of the template at its fitted strength. Point to read; drag on the strip below to zoom.`
      : "Everything but Fe II subtracted from the spectrum (grey; faded where other lines dominate), and the fitted Fe II stacked by group. Below, every transition of the template at its fitted strength. Point to read; drag on the strip below to zoom.";
    return age < 3.2;
  }

  // ---------------------------------------------------------------- overview strip (drag to zoom)
  let drag = null;
  function drawOver() {
    const D = FX.D, cv = $("fx-over");
    if (!D || !cv.offsetParent) return;
    const [g, W, H] = setup(cv), pl = 50, pr = 10, X = D.X, a = X[0], b = X[D.N - 1];
    const xs = (x) => pl + (x - a) / (b - a) * (W - pl - pr);
    let m = 0; for (let j = 0; j < D.N; j++) m = Math.max(m, D.data[j] ?? 0, D.model[j] ?? 0);
    const ys = (y) => H - 4 - y / (m || 1) * (H - 8);
    const baseOf = (j) => (D.other[j] || 0) - (D.emis[j] || 0);
    g.beginPath(); for (let j = 0; j < D.N; j++) { const x = xs(X[j]), y = ys(D.feii[j] + baseOf(j)); j ? g.lineTo(x, y) : g.moveTo(x, y); }
    for (let j = D.N - 1; j >= 0; j--) g.lineTo(xs(X[j]), ys(baseOf(j))); g.closePath();
    const gr = g.createLinearGradient(pl, 0, W - pr, 0); STOPS.forEach((c, q) => gr.addColorStop(q / (STOPS.length - 1), rgba(c, 0.55))); g.fillStyle = gr; g.fill();
    g.beginPath(); let on = false;
    for (let j = 0; j < D.N; j++) { if (D.data[j] == null) { on = false; continue; } const x = xs(X[j]), y = ys(D.data[j]); on ? g.lineTo(x, y) : g.moveTo(x, y); on = true; }
    g.strokeStyle = "rgba(29,33,40,.55)"; g.lineWidth = 0.8; g.stroke();
    const v0 = xs(FX.view[0]), v1 = xs(FX.view[1]);
    g.fillStyle = "rgba(244,245,247,.72)"; g.fillRect(pl, 0, v0 - pl, H); g.fillRect(v1, 0, W - pr - v1, H);
    g.strokeStyle = INK; g.lineWidth = 1.2; g.strokeRect(v0 + 0.5, 0.5, Math.max(1, v1 - v0 - 1), H - 1);
    cv._inv = (px) => a + (px - pl) / (W - pl - pr) * (b - a);
  }

  // ---------------------------------------------------------------- the group list
  function spark(cv, grp) {
    const D = FX.D, [g, W, H] = setup(cv), a = D.X[0], b = D.X[D.N - 1], pk = Math.max(...grp.lines.map((l) => l[2]), 1e-30);
    g.fillStyle = "#f0f2f5"; g.fillRect(0, H - 1, W, 1);
    const x0 = (D.band[0] - a) / (b - a) * W, x1 = (D.band[1] - a) / (b - a) * W; g.fillStyle = "rgba(31,158,154,.10)"; g.fillRect(x0, 0, x1 - x0, H);
    grp.lines.forEach((l) => { const x = (l[0] - a) / (b - a) * W, h = Math.max(2, Math.sqrt(l[2] / pk) * (H - 3)); g.fillStyle = rgba(grp.color, grp.on ? 0.95 : 0.5); g.fillRect(x - 0.6, H - 1 - h, 1.3, h); });
  }
  function buildList() {
    const D = FX.D; if (!D) return;
    const rows = [...D.G].sort((p, q) => (q.on - p.on) || (q.band - p.band) || (q.total - p.total));
    let sep = false;
    $("fx-glist").innerHTML = rows.map((g) => {
      const s = !g.on && !sep ? ((sep = true), '<div class="fx-gsep">At zero in this fit</div>') : "";
      const pct = D.bandTot > 0 ? 100 * g.band / D.bandTot : 0;
      return s + `<div class="fx-g${g.on ? "" : " off"}" data-i="${g.i}"><i style="background:${rgba(g.color, 1)}"></i><b>${termHtml(g.name)}</b><canvas></canvas>` +
        `<span class="pct">${g.on ? (D.bandTot > 0 ? (pct >= 0.5 ? pct.toFixed(0) + "%" : "<1%") : "") : "—"}<small>${g.on && g.sn != null ? "S/N " + g.sn.toFixed(0) : g.n + " lines"}</small></span></div>`;
    }).join("");
    $("fx-glist").querySelectorAll(".fx-g").forEach((el) => {
      const grp = D.G[+el.dataset.i];
      el.title = `${termText(grp.name)}: ${grp.n} lines, ${grp.wmin.toFixed(0)}–${grp.wmax.toFixed(0)} Å; strength ${fmt(grp.amp, 1)}${ok(grp.err) ? " ± " + grp.err.toFixed(1) : ""} (${grp.status})`;
      spark(el.querySelector("canvas"), grp);
      el.onmouseenter = () => { if (grp.on) { FX.hot = grp.i; kick(); } };
      el.onmouseleave = () => { FX.hot = null; kick(); };
      el.onclick = () => {
        if (!grp.on) return;
        FX.pin = FX.pin === grp.i ? null : grp.i; syncList(); kick();
        const best = grp.lines.reduce((p, q) => (q[2] > p[2] ? q : p), grp.lines[0]);   // to where the group is strongest
        if (FX.pin != null && best && (best[0] < FX.view[0] || best[0] > FX.view[1])) goTo([best[0] - 450, best[0] + 450]);
      };
    });
    syncList();
  }
  function syncList() { $("fx-glist").querySelectorAll(".fx-g").forEach((el) => el.classList.toggle("pin", +el.dataset.i === FX.pin)); }

  // ---------------------------------------------------------------- Grotrian diagram
  // Columns by multiplicity, L and parity, energy upwards; every fitted transition a line from its lower term
  // (the group) to its upper term, as wide as its fitted flux. Upper levels of lines with no NIST transition
  // (E_k = E_i + hc/λ) stand in the last column.
  const LORD = "SPDFGHIKLM";
  function grotModel() {
    const D = FX.D;
    if (!D.levels) return null;
    const lines = D.lines.filter((l) => l.length >= 10 && D.G[l[1]].on);
    if (!lines.length) return null;
    const terms = new Map();
    const colKey = (p) => (p ? (p.jk ? `jk${p.odd ? "o" : "e"}` : `${p.mult}${p.L}${p.odd ? "o" : "e"}`) : "?");
    // one bar per term, whether it is the lower term of a group or the upper term of other lines (z⁴D° is both:
    // the cascade e⁴D → z⁴D° → b⁴F runs through it); a label met again at another energy is another term
    const addTerm = (name, E, grp) => {
      const p = termParts(name), canon = p ? termText(name) : "?";
      const list = terms.get(canon) || [];
      let t = canon === "?" ? list[0] : list.find((q) => Math.abs(q.E0 - E) < 0.4);
      if (!t) { t = { key: `${canon}:${list.length}`, name, canon, p, col: colKey(p), E: [], E0: E, grp: null }; list.push(t); terms.set(canon, list); }
      t.E.push(E);
      if (grp) t.grp = grp;
      return t;
    };
    const T = lines.map((l) => {
      const grp = D.G[l[1]], low = addTerm(grp.name, l[7], grp);
      return { l, grp, low, up: l[9] ? addTerm(l[4], l[8], null) : addTerm("?", l[8], null) };
    });
    // a term that is also a group (lower term elsewhere) takes the group's colour
    const byName = new Map(D.G.filter((g) => g.on).map((g) => [termText(g.name), g]));
    for (const list of terms.values()) list.forEach((t) => { if (!t.grp && byName.has(t.canon)) t.grp = byName.get(t.canon); });
    const tt = [...terms.values()].flat();
    tt.forEach((t) => { t.Emin = Math.min(...t.E); t.Emax = Math.max(...t.E); t.Emid = t.E.slice().sort((a, b) => a - b)[t.E.length >> 1]; });
    const cols = [...new Set(tt.map((t) => t.col))].sort((a, b) => {
      const r = (c) => (c === "?" ? [99, 0, 0] : c.startsWith("jk") ? [98, 0, c.endsWith("o") ? 1 : 0] : [+c[0], LORD.indexOf(c[1]), c.endsWith("o") ? 1 : 0]);
      const A = r(a), B = r(b); return A[0] - B[0] || A[1] - B[1] || A[2] - B[2];
    });
    // terms sharing a column side by side when their energies are close
    tt.forEach((t) => { t.slot = 0; });
    cols.forEach((c) => {
      const inCol = tt.filter((t) => t.col === c).sort((a, b) => a.Emid - b.Emid);
      inCol.forEach((t, q) => { const prev = inCol.slice(0, q).filter((o) => Math.abs(o.Emid - t.Emid) < 0.35); t.slot = prev.length ? Math.max(...prev.map((o) => o.slot)) + 1 : 0; });
      const n = Math.max(...inCol.map((t) => t.slot)) + 1; inCol.forEach((t) => (t.nslot = n));
    });
    const fmax = Math.max(...T.map((t) => t.l[3]));
    const E = tt.flatMap((t) => [t.Emin, t.Emax]);
    return { T, terms: tt, cols, fmax, Elo: Math.min(...E), Ehi: Math.max(...E), matched: lines.filter((l) => l[9]).length, n: lines.length };
  }
  function drawGrot(now) {
    const D = FX.D, cv = $("fx-grot");
    if (!D || !cv.offsetParent || !FX.grot) return false;
    const M = FX.grot, [g, W, H] = setup(cv), pad = { l: 46, r: 12, t: 16, b: 42 };
    const ph = H - pad.t - pad.b, cw = Math.min(64, (W - 58) / M.cols.length), pw = cw * M.cols.length;
    pad.l = Math.max(46, (W - pw) / 2 + 12);
    const span = M.Ehi - M.Elo, e0 = M.Elo - 0.06 * span, e1 = M.Ehi + 0.06 * span;
    const ys = (e) => pad.t + ph - (e - e0) / (e1 - e0) * ph;
    const xcol = (t) => { const c = M.cols.indexOf(t.col), w = cw * 0.78 / t.nslot; return pad.l + c * cw + cw * 0.11 + w * (t.slot + 0.5); };
    const half = (t) => Math.min(cw * 0.78 / t.nslot, 46) * 0.42;
    const age = (now - FX.born) / 1000, f = focus();
    // energy axis
    g.font = `11px ${MONO}`; g.lineWidth = 1;
    ticks(e0, e1, 6).forEach((v) => { const y = ys(v); g.strokeStyle = GRID; g.beginPath(); g.moveTo(pad.l, y); g.lineTo(pad.l + pw, y); g.stroke(); g.fillStyle = AXIS; g.textAlign = "right"; g.textBaseline = "middle"; g.fillText(v.toFixed(1), pad.l - 6, y); });
    g.save(); g.translate(pad.l - 36, pad.t + ph / 2); g.rotate(-Math.PI / 2); g.textAlign = "center"; g.fillText("energy (eV)", 0, 0); g.restore();
    // columns: parity shading and labels
    const spans = [];
    M.cols.forEach((c, q) => {
      const x = pad.l + q * cw, odd = c.endsWith("o");
      if (odd) { g.fillStyle = "rgba(29,33,40,.025)"; g.fillRect(x, pad.t, cw, ph); }
      g.fillStyle = AXIS; g.textAlign = "center"; g.textBaseline = "top"; g.font = `600 11px ${SANS}`;
      const lab = c === "?" ? "?" : c.startsWith("jk") ? `jK${odd ? "°" : ""}` : `${SUPD[c[0]]}${c[1]}${odd ? "°" : ""}`;
      g.fillText(lab, x + cw / 2, pad.t + ph + 6);
      const mult = c === "?" ? "?" : c.startsWith("jk") ? "jk" : c[0], last = spans[spans.length - 1];
      if (last && last.m === mult) last.b = q + 1; else spans.push({ m: mult, a: q, b: q + 1 });
    });
    // the multiplicities, each centred under its columns, apart by a rule
    spans.forEach((sp, q) => {
      const x0 = pad.l + sp.a * cw, x1 = pad.l + sp.b * cw;
      if (q) { g.strokeStyle = "#d5d9df"; g.lineWidth = 1; g.beginPath(); g.moveTo(x0, pad.t); g.lineTo(x0, pad.t + ph + 34); g.stroke(); }
      const name = { 2: "doublets", 4: "quartets", 6: "sextets", 8: "octets", jk: "jK", "?": "" }[sp.m] ?? "";
      g.font = `11px ${SANS}`; g.fillStyle = FAINT; g.textAlign = "center";
      g.fillText(name, (x0 + x1) / 2, pad.t + ph + 22);
    });
    // transitions, weakest first
    const grow = ease((age - 0.3) / 0.9);
    FX.grotSeg = [];
    M.T.slice().sort((p, q) => p.l[3] - q.l[3]).forEach((t) => {
      const x0 = xcol(t.low), y0 = ys(t.l[7]), x1 = xcol(t.up), y1 = ys(t.l[8]);
      const s = Math.sqrt(t.l[3] / M.fmax), dim = f != null && f !== t.grp.i, hi = f === t.grp.i;
      const xe = x0 + (x1 - x0) * grow, ye = y0 + (y1 - y0) * grow;
      g.strokeStyle = rgba(t.grp.color, dim ? 0.05 : (hi ? 0.4 + 0.6 * s : 0.12 + 0.75 * s));
      g.lineWidth = 0.6 + (hi ? 3 : 2.4) * s; g.lineCap = "round";
      g.beginPath(); g.moveTo(x0, y0); g.lineTo(xe, ye); g.stroke();
      FX.grotSeg.push([x0, y0, x1, y1, t]);
    });
    // term bars: the J levels as a band, the name beside
    M.terms.forEach((t) => {
      const x = xcol(t), h = half(t), grp = t.grp, lower = !!grp, dim = f != null && (!grp || f !== grp.i);
      const c = grp ? grp.color : [74, 81, 96];
      if (t.name === "?") {                       // upper levels with no NIST transition: one tick each
        g.strokeStyle = "rgba(74,81,96,.55)"; g.lineWidth = 1.2; g.beginPath();
        t.E.forEach((e) => { g.moveTo(x - h * 0.6, ys(e)); g.lineTo(x + h * 0.6, ys(e)); }); g.stroke();
        return;
      }
      if (t.Emax - t.Emin > 0.004) { g.fillStyle = rgba(c, dim ? 0.08 : 0.16); g.fillRect(x - h, ys(t.Emax), 2 * h, Math.max(1, ys(t.Emin) - ys(t.Emax))); }
      g.strokeStyle = rgba(c, dim ? 0.3 : 1); g.lineWidth = lower ? 2.2 : 1.6;
      g.beginPath(); g.moveTo(x - h, ys(t.Emid)); g.lineTo(x + h, ys(t.Emid)); g.stroke();
      g.font = `${lower ? 600 : 500} 11px ${SANS}`; g.fillStyle = dim ? FAINT : lower ? rgba(c, 1) : "#4a5160"; g.textAlign = "left"; g.textBaseline = "bottom";
      g.fillText(termText(t.name), x - h, ys(t.Emax) - 2);
    });
    $("fx-grotsrc").textContent = `${M.n} fitted transitions · ${M.matched} with NIST levels`;
    return age < 1.5;
  }

  // ---------------------------------------------------------------- one Fe II line against the broad reference line
  function drawVel() {
    const D = FX.D, cv = $("fx-vel");
    if (!D || !cv.offsetParent || !D.ref) return;
    const [g, W, H] = setup(cv), pad = { l: 12, r: 12, t: 16, b: 30 }, pw = W - pad.l - pad.r, ph = H - pad.t - pad.b, k = D.kin, R = D.ref;
    const vmax = 11000, xs = (v) => pad.l + (v + vmax) / (2 * vmax) * pw, ys = (y) => pad.t + ph - y * ph * 0.9;
    g.font = `11px ${MONO}`;
    [-10000, -5000, 0, 5000, 10000].forEach((v) => { const x = xs(v); g.strokeStyle = v ? GRID : "#d5d9df"; g.beginPath(); g.moveTo(x, pad.t); g.lineTo(x, pad.t + ph); g.stroke();
      g.fillStyle = AXIS; g.textAlign = "center"; g.textBaseline = "top"; g.fillText((v / 1000).toFixed(0), x, pad.t + ph + 6); });
    g.textAlign = "right"; g.fillText("velocity (10³ km/s)", W - pad.r, H - 10);
    const hm = Math.max(...R.f.filter(ok), 1e-30);
    {                                                   // the half-maximum points of the reference profile: its peak and the middle of its FWHM
      const ip = R.f.indexOf(hm); let a = ip, b = ip;
      while (a > 0 && R.f[a] > hm / 2) a--;
      while (b < R.f.length - 1 && R.f[b] > hm / 2) b++;
      const cross = (i, j) => R.v[i] + (hm / 2 - R.f[i]) / ((R.f[j] - R.f[i]) || 1) * (R.v[j] - R.v[i]);
      R.peak = R.v[ip]; R.lo = cross(a, a + 1); R.hi = cross(b, b - 1);
    }
    const prof = new Path2D(); R.v.forEach((v, q) => { const x = xs(v), y = ys((R.f[q] || 0) / hm); q ? prof.lineTo(x, y) : prof.moveTo(x, y); });
    const area = new Path2D(prof); area.lineTo(xs(R.v[R.v.length - 1]), ys(0)); area.lineTo(xs(R.v[0]), ys(0)); area.closePath();
    g.fillStyle = "rgba(212,71,42,.12)"; g.fill(area); g.strokeStyle = "#d4472a"; g.lineWidth = 1.8; g.stroke(prof);
    if (ok(k.fwhm)) {
      const gauss = (v, fw, sh) => Math.exp(-0.5 * Math.pow((v - sh) / (fw / 2.3548), 2)), sh = k.shift || 0;
      if (ok(k.fwhm_err)) {
        g.beginPath();
        for (let v = -vmax; v <= vmax; v += 60) g.lineTo(xs(v), ys(gauss(v, k.fwhm + k.fwhm_err, sh)));
        for (let v = vmax; v >= -vmax; v -= 60) g.lineTo(xs(v), ys(gauss(v, Math.max(k.fwhm - k.fwhm_err, 50), sh)));
        g.closePath(); g.fillStyle = "rgba(31,158,154,.22)"; g.fill();
      }
      const fg = g.createLinearGradient(xs(-6000), 0, xs(6000), 0); STOPS.forEach((c, q) => fg.addColorStop(q / (STOPS.length - 1), rgba(c, 1)));
      g.strokeStyle = fg; g.lineWidth = 2.4; g.beginPath(); for (let v = -vmax; v <= vmax; v += 40) { const x = xs(v), y = ys(gauss(v, k.fwhm, sh)); v === -vmax ? g.moveTo(x, y) : g.lineTo(x, y); } g.stroke();
      // the FWHM of each curve, drawn exactly at half maximum between the points where that curve crosses it
      const y5 = ys(0.5);
      const bar = (c, v0, v1, dy, lab) => {
        g.strokeStyle = c; g.lineWidth = 1.5; g.beginPath(); g.moveTo(xs(v0), y5); g.lineTo(xs(v1), y5); g.stroke();
        g.fillStyle = c; [v0, v1].forEach((v) => { g.beginPath(); g.arc(xs(v), y5, 3, 0, 2 * Math.PI); g.fill(); });
        g.textAlign = "left"; g.textBaseline = "middle"; g.font = `600 11px ${SANS}`; g.fillText(lab, xs(Math.max(v0, v1)) + 8, y5 + dy); };
      if (ok(R.lo) && ok(R.hi)) bar("#b5371d", R.lo, R.hi, -9, `broad ${R.label}`);
      bar("#1a7f86", sh - k.fwhm / 2, sh + k.fwhm / 2, 9, "Fe II");
    }
    const sgn = (v) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(0);
    let txt = `A single Fe II line at the fitted width (FWHM <b>${pm(k.fwhm, k.fwhm_err, 0)}</b> km/s), against the broad ${esc(R.label)} profile` +
      (ok(R.fwhm) ? ` (FWHM <b>${pm(R.fwhm, R.fwhm_err, 0)}</b> km/s)` : "") + ". Velocities are from each line's laboratory wavelength" +
      (ok(k.shift) ? `: Fe II at <b>${sgn(k.shift)}${ok(k.shift_err) ? " <em>± " + k.shift_err.toFixed(0) + "</em>" : ""}</b> km/s, the ${esc(R.label)} peak at <b>${sgn(R.peak)}</b> km/s, so Fe II is ${Math.abs(k.shift - R.peak) < 50 ? "at the same velocity" : `<b>${Math.abs(k.shift - R.peak).toFixed(0)} km/s</b> ${k.shift > R.peak ? "redder" : "bluer"}`}.` : ".");
    if (ok(k.fwhm) && ok(R.fwhm)) {
      const ratio = k.fwhm / R.fwhm, re = ok(k.fwhm_err) && ok(R.fwhm_err) ? ratio * Math.hypot(k.fwhm_err / k.fwhm, R.fwhm_err / R.fwhm) : null;
      txt += ` FWHM(Fe II)/FWHM(${esc(R.label)}) = <b>${ratio.toFixed(2)}${re != null ? " ± " + re.toFixed(2) : ""}</b>` +
        (k.fwhm_status !== "free" ? ` (the Fe II width is ${esc(k.fwhm_status)} in this model).` : re == null ? "." : Math.abs(ratio - 1) < 2 * re ? ": the same width within the errors." : ratio < 1 ? ": Fe II is narrower." : ": Fe II is broader.");
    }
    $("fx-veltxt").innerHTML = txt;
  }

  // ---------------------------------------------------------------- the quasar main sequence
  function drawMS(now) {
    const D = FX.D, cv = $("fx-ms");
    if (!D || !cv.offsetParent || !hasMS()) return false;
    const [g, W, H] = setup(cv), pad = { l: 46, r: 14, t: 12, b: 32 }, pw = W - pad.l - pad.r, ph = H - pad.t - pad.b, s = D.strength, f = D.ref;
    const xm = Math.max(2.5, Math.ceil((s.r + 0.3) * 2) / 2), ym = Math.max(12000, Math.ceil((f.fwhm + 2000) / 4000) * 4000);
    const xs = (x) => pad.l + x / xm * pw, ys = (y) => pad.t + ph - y / ym * ph, type = msType(s.r, f.fwhm);
    g.font = `11px ${MONO}`;
    ticks(0, xm, 5).forEach((v) => { g.fillStyle = AXIS; g.textAlign = "center"; g.textBaseline = "top"; g.fillText(v.toFixed(1), xs(v), pad.t + ph + 6); });
    ticks(0, ym, 4).forEach((v) => { g.textAlign = "right"; g.textBaseline = "middle"; g.fillText(v ? v / 1000 + "k" : "0", pad.l - 6, ys(v)); });
    g.textAlign = "right"; g.textBaseline = "bottom"; g.fillText("R_FeII", W - pad.r, H - 1);
    g.save(); g.translate(11, pad.t + ph / 2); g.rotate(-Math.PI / 2); g.textAlign = "center"; g.textBaseline = "middle"; g.fillText("FWHM Hβ (km/s)", 0, 0); g.restore();
    g.fillStyle = "rgba(47,127,224,.05)"; g.fillRect(pad.l, ys(4000), pw, ys(0) - ys(4000));
    g.fillStyle = "rgba(212,71,42,.04)"; g.fillRect(pad.l, ys(ym), pw, ys(4000) - ys(ym));
    // the spectral-type grid (fantasy_agn.plotting.main_sequence_grid)
    const grid = [[[0, xm], [4000, 4000]], [[0.5, 0.5], [0, 16000]], [[1, 1], [0, 8000]], [[1.5, 1.5], [0, 4000]], [[2, 2], [0, 4000]], [[0, 1], [8000, 8000]], [[0, 0.5], [12000, 12000]]];
    grid.forEach(([x, y]) => { const main = y[0] === 4000 && y[1] === 4000; g.strokeStyle = main ? "#9aa1ad" : "#c9ced6"; g.setLineDash(main ? [] : [2, 3]); g.beginPath(); g.moveTo(xs(x[0]), ys(Math.min(y[0], ym))); g.lineTo(xs(x[1]), ys(Math.min(y[1], ym))); g.stroke(); });
    g.setLineDash([]);
    g.font = `600 11px ${SANS}`;
    [["A1", 0.25, 2000], ["A2", 0.75, 2000], ["A3", 1.25, 2000], ["A4", 1.75, 2000], ["B1", 0.25, 6000], ["B2", 0.75, 6000], ["B1+", 0.25, 10000], ["B1++", 0.25, 14000]]
      .forEach(([n, x, y]) => { if (y > ym) return; g.fillStyle = n === type ? INK : "#b6bcc6"; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(n, xs(x), ys(y)); });
    g.fillStyle = "#2f7fe0"; g.textAlign = "right"; g.fillText("Population A", W - pad.r - 4, ys(2000));
    g.fillStyle = "#d4472a"; g.fillText("Population B", W - pad.r - 4, ys(6000));
    const t = ease(((now - FX.born) / 1000 - 1.4) / 0.8), x = xs(s.r), y = ys(f.fwhm);
    if (t > 0) {
      g.strokeStyle = INK; g.lineWidth = 1.3; g.beginPath();
      if (ok(s.r_err)) { g.moveTo(xs(s.r - s.r_err), y); g.lineTo(xs(s.r + s.r_err), y); }
      if (ok(f.fwhm_err)) { g.moveTo(x, ys(f.fwhm - f.fwhm_err)); g.lineTo(x, ys(f.fwhm + f.fwhm_err)); }
      g.stroke();
      g.fillStyle = "rgba(31,158,154,.18)"; g.beginPath(); g.arc(x, y, 13 * (1 + 0.6 * (1 - t)), 0, 7); g.fill();
      const fg = g.createLinearGradient(x - 6, y - 6, x + 6, y + 6); fg.addColorStop(0, "#6a4fd8"); fg.addColorStop(0.5, "#1f9e9a"); fg.addColorStop(1, "#e0702a");
      g.fillStyle = fg; g.beginPath(); g.arc(x, y, 6, 0, 7); g.fill(); g.strokeStyle = "#fff"; g.lineWidth = 2; g.stroke();
    }
    const near = (b, v) => ok(s.r_err) && Math.abs(s.r - b) < 2 * s.r_err && v;
    $("fx-mstxt").innerHTML = `R<sub>FeII</sub> = <b>${pm(s.r, s.r_err, 2)}</b>, FWHM(Hβ) = <b>${pm(f.fwhm, f.fwhm_err, 0)}</b> km/s: spectral type <b>${type}</b>, Population ${type[0]} (Sulentic et al. 2000; Marziani et al. 2018)` +
      ([0.5, 1, 1.5].some((b) => near(b, true)) ? ", within 2σ of a boundary in R<sub>FeII</sub>." : ".");
    return t < 1;
  }

  // ---------------------------------------------------------------- tooltip and pointer
  const tip = () => $("fx-tip");
  function place(e) {
    const el = tip(), tw = el.offsetWidth, th = el.offsetHeight;
    el.style.left = Math.min(window.innerWidth - tw - 8, e.clientX + 16) + "px";
    el.style.top = Math.max(8, Math.min(window.innerHeight - th - 8, e.clientY - th / 2)) + "px";
  }
  function plotMove(e) {
    const D = FX.D, L = FX.L; if (!D || !L.xs) return;
    const r = $("fx-plot").getBoundingClientRect(); FX.mouse = { x: e.clientX - r.left, y: e.clientY - r.top };
    if (FX.mouse.x < L.pad.l || FX.mouse.x > r.width - L.pad.r) { tip().classList.remove("on"); kick(); return; }
    const w = L.a + (FX.mouse.x - L.pad.l) / L.pw * (L.b - L.a), j = lo(D.X, w), full = FX.mode === "full", tot = D.feii[j];
    const parts = D.live.map((g) => [g, g.flux[j] || 0]).filter(([, v]) => v > 0.005 * Math.max(tot, 1e-30)).sort((p, q) => q[1] - p[1]);
    let near = null, dmin = 1e9; D.lines.forEach((l) => { const d = Math.abs(l[0] - w); if (d < dmin) { dmin = d; near = l; } });
    const inStrip = FX.mouse.y > L.stickTop - 6;
    if (inStrip) FX.hot = near && Math.abs(L.xs(near[0]) - FX.mouse.x) < 8 ? near[1] : null;
    else {
      const base = full ? D.other[j] || 0 : 0, k = D.live.findIndex((g, q) => L.ys(base + D.stack[q][j]) <= FX.mouse.y);
      FX.hot = FX.mouse.y >= L.ys(base + tot) && FX.mouse.y <= L.ys(base) && k >= 0 ? D.live[k].i : null;
    }
    const row = (c, lab, v) => `<div class="r"><i style="background:${c}"></i><span>${lab}</span>${v}</div>`;
    let h = `<div class="t">λ ${w.toFixed(1)} Å</div>`;
    if (full) {
      if (D.data[j] != null) h += row("#1d2128", "data", `${(+D.data[j]).toPrecision(4)} ± ${(+D.err[j]).toPrecision(2)}`);
      h += row("#d4472a", "model", (+D.model[j]).toPrecision(4)) + row("#9aa1ad", "rest of the model", (+D.other[j]).toPrecision(4));
      h += row(INK, "Fe II", `${(+tot).toPrecision(3)} · ${D.model[j] > 0 ? (100 * tot / D.model[j]).toFixed(0) : "—"}% of the model`);
    } else {
      h += row(INK, "Fe II, all groups", (+tot).toPrecision(3));
      if (D.resid[j] != null) h += row("#c9cdd4", "data − rest of model", `${(+D.resid[j]).toPrecision(3)} ± ${(+D.err[j]).toPrecision(2)}`);
      if (D.blend[j] > 0.3) h += `<div class="m">blended with ${esc(D.labels.filter(([, x]) => Math.abs(x - w) < 60).map(([n]) => n).filter((n, q, a) => a.indexOf(n) === q).join(", ") || "other lines")}: the subtraction is uncertain here</div>`;
    }
    if (parts.length && tot > 0) h += "<hr>" + parts.slice(0, 4).map(([g, v]) => row(rgba(g.color, 1), termHtml(g.name), `${(100 * v / tot).toFixed(0)}%`)).join("");
    if (near && dmin < 6) h += `<hr><div class="m">nearest transition <b>${near[0].toFixed(1)} Å</b>, ${termHtml(D.G[near[1]].name)}${near.length >= 10 && near[9] ? ` → ${termHtml(near[4])}` : ""}</div>`;
    tip().innerHTML = h; tip().classList.add("on"); place(e); syncList(); kick();
  }
  function grotMove(e) {
    if (!FX.grotSeg) return;
    const r = $("fx-grot").getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
    let best = null, dmin = 6;
    FX.grotSeg.forEach(([x0, y0, x1, y1, t]) => {
      const dx = x1 - x0, dy = y1 - y0, u = Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy || 1)));
      const d = Math.hypot(px - x0 - u * dx, py - y0 - u * dy); if (d < dmin || (best && Math.abs(d - dmin) < 0.5 && t.l[3] > best.l[3])) { dmin = Math.min(d, dmin); best = t; }
    });
    if (!best) { FX.hot = null; tip().classList.remove("on"); kick(); return; }
    const l = best.l, D = FX.D;
    FX.hot = best.grp.i;
    const share = best.grp.total > 0 ? l[3] / D.lines.filter((q) => q[1] === l[1]).reduce((s, q) => s + q[3], 0) : 0;
    tip().innerHTML = `<div class="t">λ ${l[0].toFixed(1)} Å</div>` +
      `<div class="r"><i style="background:${rgba(best.grp.color, 1)}"></i><span>${termHtml(best.grp.name)}${l[5] ? ` <small>J=${esc(l[5])}</small>` : ""} → ${l[9] ? termHtml(l[4]) + (l[6] ? ` <small>J=${esc(l[6])}</small>` : "") : "?"}</span></div>` +
      `<div class="m">E<sub>i</sub> = ${l[7].toFixed(3)} eV → E<sub>k</sub> = ${l[8].toFixed(3)} eV</div>` +
      `<div class="m">${(100 * share).toFixed(1)}% of the group's flux · ${l[9] === 1 ? "NIST transition" : l[9] === 2 ? "identified from the NIST levels (no transition probability listed)" : "unidentified: upper level at E<sub>i</sub> + hc/λ"}</div>`;
    tip().classList.add("on"); place(e); syncList(); kick();
  }
  function leave() { FX.mouse = null; FX.hot = null; tip().classList.remove("on"); syncList(); kick(); }

  // ---------------------------------------------------------------- loop and layout
  function goTo(r) { FX.from = FX.view.slice(); FX.to = r; FX.tView = performance.now(); kick(); }
  function frame(now) {
    FX.raf = 0;
    if (!FX.D || $("res-feii").classList.contains("hidden")) return;
    if (FX.tView) { const p = ease((now - FX.tView) / 550); FX.view = [FX.from[0] + (FX.to[0] - FX.from[0]) * p, FX.from[1] + (FX.to[1] - FX.from[1]) * p]; if (p >= 1) FX.tView = 0; }
    const a = drawPlot(now), b = drawMS(now), c = drawGrot(now); drawOver();
    if (a || b || c || FX.tView) kick();
  }
  function kick() { if (!FX.raf) FX.raf = requestAnimationFrame(frame); }
  function layout() {
    const D = FX.D; if (!D) return;
    FX.grot = grotModel();
    $("fx-grotcard").classList.toggle("hidden", !FX.grot);
    $("fx-velcard").classList.toggle("hidden", !D.ref);
    $("fx-mscard").classList.toggle("hidden", !hasMS());
    if (FX.grot) $("fx-grottxt").innerHTML = "Each line runs from the lower term of its group to its upper term, as wide as its fitted flux; the bars are the terms, with their J levels shaded. Levels from the NIST Atomic Spectra Database (Kramida et al.); within each multiplicity the columns run S, P, D, F, G, H, the shaded ones odd parity (°). Lines missing from the NIST transition list are identified from the NIST levels (E<sub>i</sub> + hc/λ = E<sub>k</sub>). Point to a line to read it.";
    if (D.ref) $("fx-velh").textContent = `Is Fe II as broad as ${D.ref.label}?`;
    buildList(); drawVel(); kick();
  }
  function init() {
    const cv = $("fx-plot"); if (!cv) return;
    cv.addEventListener("pointermove", plotMove);
    cv.addEventListener("pointerleave", leave);
    cv.addEventListener("click", () => { FX.pin = FX.hot != null && FX.pin !== FX.hot ? FX.hot : null; syncList(); kick(); });
    $("fx-grot").addEventListener("pointermove", grotMove);
    $("fx-grot").addEventListener("pointerleave", leave);
    $("fx-grot").addEventListener("click", () => { FX.pin = FX.hot != null && FX.pin !== FX.hot ? FX.hot : null; syncList(); kick(); });
    const ov = $("fx-over");
    ov.addEventListener("pointerdown", (e) => { if (!ov._inv) return; drag = { x0: ov._inv(e.offsetX) }; ov.setPointerCapture(e.pointerId); });
    ov.addEventListener("pointermove", (e) => {
      if (!drag || !FX.D) return;
      const X = FX.D.X, w = ov._inv(e.offsetX), a = Math.max(X[0], Math.min(drag.x0, w)), b = Math.min(X[FX.D.N - 1], Math.max(drag.x0, w));
      if (b - a > 40) { FX.view = [a, b]; FX.tView = 0; $("fx-ranges").querySelectorAll("button").forEach((q) => q.classList.remove("on")); kick(); }
    });
    ov.addEventListener("pointerup", () => (drag = null));
    $("fx-mode").querySelectorAll("button").forEach((b) => (b.onclick = () => {
      FX.mode = b.dataset.m; $("fx-mode").querySelectorAll("button").forEach((q) => q.classList.toggle("on", q === b));
      $("fx-zoomwrap").classList.toggle("hidden", FX.mode !== "full"); kick();
    }));
    $("fx-data").onchange = kick;
    $("fx-zoom").onchange = () => { FX.zoom = $("fx-zoom").checked; kick(); };
    let last = 0;
    new ResizeObserver(() => { const w = $("res-feii").clientWidth; if (FX.D && w && w !== last) { last = w; buildList(); drawVel(); kick(); } }).observe($("res-feii"));
  }
  // what the plot shows, for the Python code under the view
  const state = () => FX.D ? { mode: FX.mode, xmin: Math.round(FX.view[0]), xmax: Math.round(FX.view[1]), zoom: FX.zoom ? 1 : 0,
    data: $("fx-data").checked ? 1 : 0, pin: FX.pin != null ? (FX.D.G.find((g) => g.i === FX.pin) || {}).name || "" : "" } : {};
  window.FX = { show, kick, state };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
