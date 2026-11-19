// Fit step: the fit as it runs (GET /api/wf/fitlive, Session.live_state, fitting._Live). The data with the model
// gliding onto them snapshot by snapshot, the residuals settling, chi2 falling on a log scale, and the phases of
// the fit: the initial guess (grid search of the widths and shifts), the trust-region fit, the build-up of
// overlapping components one at a time, the errors. Uses api of wf.js.
(() => {
  const $ = (id) => document.getElementById(id);
  const MONO = '"IBM Plex Mono", "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace';
  const SANS = '"IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Helvetica, Arial, sans-serif';
  const STOPS = [[106, 79, 216], [47, 127, 224], [31, 158, 154], [31, 158, 106], [217, 178, 31], [224, 112, 42], [212, 71, 42]];
  const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  const ok = (v) => v != null && isFinite(v);
  const PHASE = { guess: ["Initial guess", "grid search of the widths and shifts"], fit: ["Fit", "trust-region least squares"],
    "build-up": ["Build-up", "overlapping components added one at a time"], errors: ["Errors", "uncertainties"] };
  const L = { seq: 0, D: null, cur: null, chi: null, live: null, poll: 0, raf: 0, active: false, born: 0, done: 0, ylim: null };

  function setup(cv) {
    const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    const w = Math.round(r.width * dpr), h = Math.round(r.height * dpr);
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, r.width, r.height);
    return [g, r.width, r.height];
  }
  const pct = (a, q) => { const s = a.filter(ok).sort((p, r) => p - r); return s.length ? s[Math.min(s.length - 1, Math.max(0, Math.floor(q * (s.length - 1))))] : 0; };

  async function fetchLive() {
    L.poll = 0;
    if (!L.active) return;
    try {
      const d = await api(`/api/wf/fitlive?seq=${L.seq}`);
      if (d.seq && d.seq !== L.seq && d.x) {                     // a new fit: its data
        L.seq = d.seq; L.D = { x: d.x, y: d.y, err: d.err }; L.cur = null; L.born = performance.now(); L.done = 0;
        const lo = pct(d.y, 0.005), hi = pct(d.y, 0.995), pad = 0.08 * (hi - lo || 1);
        L.ylim = [Math.min(lo - pad, 0), hi + pad];
      }
      if (d.seq === L.seq) {
        L.live = d;
        if (d.phase === "errors" && !L.done) L.done = performance.now();   // converged: the model glows once
      }
    } catch (e) { /* the next poll tries again */ }
    if (L.active) L.poll = setTimeout(fetchLive, 220);
    kick();
  }
  function kick() { if (!L.raf) L.raf = requestAnimationFrame(frame); }
  function frame(now) {
    L.raf = 0;
    if (!L.active) return;
    const D = L.D, V = L.live;
    let moving = false;
    if (D && V && V.model) {
      if (!L.cur) L.cur = V.model.map((v) => v ?? 0);
      else for (let i = 0; i < L.cur.length; i++) {                // glide towards the latest model
        const t = V.model[i] ?? L.cur[i], d = t - L.cur[i];
        if (Math.abs(d) > 1e-6 * (Math.abs(t) + 1e-30)) { L.cur[i] += d * 0.16; moving = true; }
      }
    }
    if (V && ok(V.chi2)) {
      const t = Math.log10(Math.max(V.chi2, 1e-6));
      L.chi = L.chi == null ? t : L.chi + (t - L.chi) * 0.2;
      if (Math.abs(t - L.chi) > 1e-4) moving = true;
    }
    drawPlot(now); drawSpark(now); drawText();
    if (moving || L.active) L.raf = requestAnimationFrame(frame);
  }

  // ------------------------------------------------ the data, the model and the residuals
  function drawPlot(now) {
    const cv = $("fl-plot");
    if (!cv.offsetParent) return;
    const [g, W, H] = setup(cv), D = L.D;
    const pad = { l: 12, r: 12, t: 10, b: 22 }, rh = 46, gap = 8, ph = H - pad.t - pad.b - rh - gap, pw = W - pad.l - pad.r;
    if (!D) {                                                    // setting up: a shimmer where the plot will be
      const x = ((now / 1400) % 1.4 - 0.2) * W, gr = g.createLinearGradient(x - 120, 0, x + 120, 0);
      gr.addColorStop(0, "rgba(238,240,243,0)"); gr.addColorStop(0.5, "rgba(226,229,234,.9)"); gr.addColorStop(1, "rgba(238,240,243,0)");
      g.fillStyle = "#f6f7f9"; g.fillRect(pad.l, pad.t, pw, ph); g.fillStyle = gr; g.fillRect(pad.l, pad.t, pw, ph);
      g.fillStyle = "#7b8290"; g.font = `12px ${SANS}`; g.textAlign = "center"; g.fillText("Setting up the problem…", W / 2, pad.t + ph / 2);
      return;
    }
    const n = D.x.length, x0 = D.x[0], x1 = D.x[n - 1], [y0, y1] = L.ylim;
    const xs = (v) => pad.l + (v - x0) / (x1 - x0) * pw, ys = (v) => pad.t + ph - (v - y0) / (y1 - y0) * ph;
    // wavelength ticks
    g.font = `10.5px ${MONO}`; g.fillStyle = "#7b8290"; g.textAlign = "center"; g.textBaseline = "top";
    const st = [50, 100, 200, 250, 500, 1000, 2000].find((s) => (x1 - x0) / s <= 9) || 5000;
    for (let v = Math.ceil(x0 / st) * st; v <= x1; v += st) {
      g.strokeStyle = "#eef0f3"; g.lineWidth = 1; g.beginPath(); g.moveTo(xs(v), pad.t); g.lineTo(xs(v), pad.t + ph + gap + rh); g.stroke();
      g.fillText(v.toFixed(0), xs(v), H - pad.b + 6);
    }
    g.save(); g.beginPath(); g.rect(pad.l, pad.t, pw, ph); g.clip();
    // the error band and the data
    g.fillStyle = "rgba(29,33,40,.06)"; g.beginPath();
    for (let i = 0; i < n; i++) g.lineTo(xs(D.x[i]), ys((D.y[i] ?? 0) + (D.err[i] ?? 0)));
    for (let i = n - 1; i >= 0; i--) g.lineTo(xs(D.x[i]), ys((D.y[i] ?? 0) - (D.err[i] ?? 0)));
    g.fill();
    g.strokeStyle = "#9aa1ad"; g.lineWidth = 1; g.beginPath();
    for (let i = 0; i < n; i++) { const X = xs(D.x[i]), Y = ys(D.y[i] ?? 0); i ? g.lineTo(X, Y) : g.moveTo(X, Y); }
    g.stroke();
    if (L.cur) {                                                 // the model: the logo's ribbon, with a soft glow
      const gr = g.createLinearGradient(pad.l, 0, pad.l + pw, 0);
      STOPS.forEach((c, k) => gr.addColorStop(k / (STOPS.length - 1), rgba(c, 1)));
      const path = new Path2D();
      for (let i = 0; i < n; i++) { const X = xs(D.x[i]), Y = ys(L.cur[i]); i ? path.lineTo(X, Y) : path.moveTo(X, Y); }
      const fresh = L.done ? Math.max(0, 1 - (now - L.done) / 1200) : 0;
      g.lineJoin = "round"; g.strokeStyle = gr;
      g.globalAlpha = 0.18 + 0.25 * fresh; g.lineWidth = 7; g.stroke(path);
      g.globalAlpha = 1; g.lineWidth = 2; g.stroke(path);
    }
    g.restore();
    // residuals in sigma: grey when within the noise, warming to red beyond 3 sigma
    const top = pad.t + ph + gap, mid = top + rh / 2;
    g.fillStyle = "#f6f7f9"; g.fillRect(pad.l, top, pw, rh);
    g.fillStyle = "rgba(29,33,40,.06)"; g.fillRect(pad.l, mid - rh / 2 * (1 / 5), pw, rh / 5);
    if (L.cur) {
      const bw = Math.max(1, pw / n);
      for (let i = 0; i < n; i++) {
        const e = D.err[i], r = e > 0 ? ((D.y[i] ?? 0) - L.cur[i]) / e : 0, a = Math.min(Math.abs(r), 5);
        const c = a < 2 ? [154, 161, 173] : a < 3 ? [217, 178, 31] : [212, 71, 42];
        g.fillStyle = rgba(c, a < 2 ? 0.5 : 0.9);
        const h = r / 5 * rh / 2;
        g.fillRect(xs(D.x[i]) - bw / 2, Math.min(mid, mid - h), bw, Math.max(0.6, Math.abs(h)));
      }
    }
    g.fillStyle = "#7b8290"; g.font = `10px ${SANS}`; g.textAlign = "left"; g.textBaseline = "top";
    g.fillText("residuals (±5σ) · grey within 2σ, gold 2–3σ, red beyond", pad.l + 4, top + 3);
  }

  // ------------------------------------------------ chi2 against time, the phases marked
  function drawSpark(now) {
    const cv = $("fl-spark"), V = L.live;
    if (!cv.offsetParent) return;
    const [g, W, H] = setup(cv), pad = { l: 34, r: 10, t: 14, b: 16 }, pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
    if (!V || !V.hist || V.hist.length < 1) return;
    const h = V.hist.filter((p) => ok(p[1]) && p[1] > 0), tmax = Math.max(V.t || 0, h.length ? h[h.length - 1][0] : 0, 0.5);
    if (!h.length) return;
    const lg = h.map((p) => Math.log10(p[1])), lo = Math.floor(Math.min(...lg, 0) * 2) / 2, hi = Math.ceil(Math.max(...lg) * 2) / 2 + 0.01;
    const xs = (t) => pad.l + t / tmax * pw, ys = (v) => pad.t + ph - (v - lo) / (hi - lo) * ph;
    g.font = `10px ${MONO}`; g.fillStyle = "#7b8290"; g.textAlign = "right"; g.textBaseline = "middle";
    for (let v = Math.ceil(lo); v <= hi; v++) {
      g.strokeStyle = "#eef0f3"; g.beginPath(); g.moveTo(pad.l, ys(v)); g.lineTo(W - pad.r, ys(v)); g.stroke();
      g.fillText(v === 0 ? "1" : `10${v < 0 ? "⁻" : ""}${"⁰¹²³⁴⁵⁶⁷⁸⁹"[Math.abs(v)] || Math.abs(v)}`, pad.l - 5, ys(v));
    }
    if (lo <= 0 && hi >= 0) { g.strokeStyle = "rgba(31,111,120,.5)"; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(pad.l, ys(0)); g.lineTo(W - pad.r, ys(0)); g.stroke(); g.setLineDash([]); }
    (V.marks || []).forEach(([p, t]) => {                       // where each phase began
      g.strokeStyle = "rgba(123,130,144,.45)"; g.beginPath(); g.moveTo(xs(t), pad.t - 4); g.lineTo(xs(t), pad.t + ph); g.stroke();
      g.fillStyle = "#7b8290"; g.font = `9.5px ${SANS}`; g.textAlign = "left"; g.textBaseline = "top"; g.fillText((PHASE[p] || [p])[0], xs(t) + 3, 1);
    });
    const area = new Path2D(); area.moveTo(xs(h[0][0]), pad.t + ph);
    h.forEach((p, i) => area.lineTo(xs(p[0]), ys(lg[i])));
    area.lineTo(xs(h[h.length - 1][0]), pad.t + ph); area.closePath();
    const fill = g.createLinearGradient(0, pad.t, 0, pad.t + ph); fill.addColorStop(0, "rgba(212,71,42,.22)"); fill.addColorStop(1, "rgba(212,71,42,0)");
    g.fillStyle = fill; g.fill(area);
    g.strokeStyle = "#d4472a"; g.lineWidth = 1.6; g.beginPath(); h.forEach((p, i) => (i ? g.lineTo(xs(p[0]), ys(lg[i])) : g.moveTo(xs(p[0]), ys(lg[i])))); g.stroke();
    const last = h[h.length - 1], pulse = 3 + 2 * (0.5 + 0.5 * Math.sin(now / 180));
    g.fillStyle = "rgba(212,71,42,.25)"; g.beginPath(); g.arc(xs(last[0]), ys(lg[lg.length - 1]), pulse + 3, 0, 2 * Math.PI); g.fill();
    g.fillStyle = "#d4472a"; g.beginPath(); g.arc(xs(last[0]), ys(lg[lg.length - 1]), 3, 0, 2 * Math.PI); g.fill();
  }

  // ------------------------------------------------ the phases, the numbers
  function drawText() {
    const V = L.live, phases = (V && V.phases && V.phases.length ? V.phases : ["guess", "fit", "build-up", "errors"]);
    const cur = V ? V.phase : null, seen = new Set(((V && V.marks) || []).map((m) => m[0])), ci = phases.indexOf(cur);
    const html = phases.map((p, i) => {
      const st = !V ? "next" : p === cur ? "now" : (seen.has(p) || i < ci) ? "done" : "next";
      return `<span class="fl-ph ${st}" title="${(PHASE[p] || ["", ""])[1]}"><i></i>${(PHASE[p] || [p])[0]}</span>`;
    }).join('<span class="fl-sep"></span>');
    if ($("fl-phases").dataset.h !== html) { $("fl-phases").innerHTML = html; $("fl-phases").dataset.h = html; }
    const chi = L.chi != null ? Math.pow(10, L.chi) : null;
    $("fl-chi").textContent = chi == null ? "—" : chi >= 100 ? chi.toFixed(0) : chi >= 10 ? chi.toFixed(1) : chi.toFixed(2);
    $("fl-chi").className = "fl-num " + (chi == null ? "" : chi <= 1.5 ? "good" : chi <= 3 ? "fair" : "poor");
    $("fl-meta").textContent = V && V.n ? `${V.n.toLocaleString()} evaluations · ${V.npar} free parameters · best χ²ᵣ ${ok(V.best) ? V.best.toFixed(2) : "—"}` : "";
    $("fl-what").textContent = V && cur && PHASE[cur] ? `${PHASE[cur][0]}: ${V.detail || PHASE[cur][1]}` : "Preparing the fit";
  }

  // shown while a fit of one spectrum runs; the others keep the list of tasks
  function run(active, elapsed) {
    if (active && !L.active) { L.active = true; L.chi = null; fetchLive(); kick(); }
    if (!active && L.active) { L.active = false; clearTimeout(L.poll); L.poll = 0; }
    $("fl").classList.toggle("hidden", !active);
    if (active) $("fl-time").textContent = `${(elapsed || 0).toFixed(1)} s`;
  }
  window.FL = { run };
})();
