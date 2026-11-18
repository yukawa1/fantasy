// Opening screen of the FANTASY app. The galaxy on the sky (SDSS image); a fiber collects the light of
// its nucleus and a spectrograph spreads it by wavelength; the light rises into the FANTASY logo, whose
// ribbon of spectra collapses into one line: the real SDSS spectrum. It moves to the rest frame and
// separates into the fitted host galaxy, AGN continuum, Fe II, broad and narrow lines. Every curve and
// number comes from static/hero.json (tools/build_app_hero.py), a real FANTASY fit.
(() => {
  "use strict";
  const hero = document.getElementById("hero");
  if (!hero) return;
  const cv = hero.querySelector("canvas"), ctx = cv.getContext("2d");
  const stage = document.getElementById("herostage"), readout = document.getElementById("heroread");
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const COLOR = { host: [196, 132, 22], agn: [52, 105, 210], feii: [28, 146, 108], broad: [212, 71, 42], narrow: [134, 88, 196] };
  const INK = "#1d2128", GRID = "#eceef1", AXIS = "#7b8290", FAINT = "#a3a9b4";
  // seconds of the timeline
  const T = { img: [0.1, 1.1], fiber: [1.4, 2.3], beam: [2.6, 3.4], fan: [3.3, 4.2], logo: [4.0, 5.1],
    word: [4.6, 5.4], imgout: [5.6, 6.3], unword: [6.0, 6.6], morph: [6.3, 7.8], shift: [8.0, 9.0],
    host: [9.2, 10.2], agn: [10.0, 10.8], feii: [10.6, 11.4], broad: [11.2, 12.4], narrow: [12.2, 13.1],
    model: [13.0, 13.8], resid: [13.5, 14.4], end: 14.6 };

  // The logo's front curve (logo.png, 720 px: x right, y down) and its colours along x.
  const LOGO = [[125, 455], [190, 428], [232, 345], [265, 283], [284, 330], [300, 368], [350, 405], [398, 452],
    [415, 540], [432, 352], [448, 470], [476, 465], [498, 412], [515, 4], [528, 330], [548, 470], [600, 562]];
  const LOGO_COLORS = [[0, [126, 190, 204]], [0.14, [47, 127, 184]], [0.26, [150, 60, 130]], [0.3, [214, 40, 90]],
    [0.62, [222, 52, 74]], [0.7, [236, 110, 80]], [0.78, [242, 160, 90]], [0.82, [246, 205, 110]], [0.84, [70, 120, 190]],
    [0.88, [240, 140, 90]], [1, [232, 92, 92]]];
  const RIBBON = 22;                                     // lines in the ribbon

  let D = null, W = 0, H = 0, start = 0, raf = 0, hover = null, running = false, shown = true;
  let lastStage = "", statsShown = false, logoY = null;
  const legend = document.getElementById("herolegend");
  const sky = new Image();

  const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
  const ease = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
  const phase = (t, [a, b]) => ease(clamp((t - a) / (b - a)));
  const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  const mix = (a, b, m) => a.map((v, j) => Math.round(v * (1 - m) + b[j] * m));

  // colour of light of a wavelength (Bruton 1996), darkened to read on white
  function lambdaRGB(lam) {
    const nm = lam / 10;
    let r = 0, g = 0, b = 0;
    if (nm < 440) { r = clamp(-(nm - 440) / 60); b = 1; }
    else if (nm < 490) { g = (nm - 440) / 50; b = 1; }
    else if (nm < 510) { g = 1; b = -(nm - 510) / 20; }
    else if (nm < 580) { r = (nm - 510) / 70; g = 1; }
    else if (nm < 645) { r = 1; g = -(nm - 645) / 65; }
    else r = 1;
    const k = nm > 540 && nm < 600 ? 0.62 : 0.78;
    return [r, g, b].map((v) => Math.round(255 * k * Math.pow(v, 0.8)));
  }
  function logoRGB(u) {
    for (let k = 1; k < LOGO_COLORS.length; k++) {
      const [u1, c1] = LOGO_COLORS[k], [u0, c0] = LOGO_COLORS[k - 1];
      if (u <= u1) return mix(c0, c1, (u - u0) / (u1 - u0));
    }
    return LOGO_COLORS[LOGO_COLORS.length - 1][1];
  }

  // the logo curve sampled at the spectrum's n points (Catmull-Rom through the logo points), 0..1 in y (down)
  function sampleLogo(n) {
    const dense = [];
    for (let k = 0; k < LOGO.length - 1; k++) {
      const p0 = LOGO[Math.max(0, k - 1)], p1 = LOGO[k], p2 = LOGO[k + 1], p3 = LOGO[Math.min(LOGO.length - 1, k + 2)];
      for (let s = 0; s < 24; s++) {
        const t = s / 24, t2 = t * t, t3 = t2 * t;
        const f = (a, b, c, d) => 0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3);
        dense.push([f(p0[0], p1[0], p2[0], p3[0]), f(p0[1], p1[1], p2[1], p3[1])]);
      }
    }
    dense.push(LOGO[LOGO.length - 1]);
    const x0 = LOGO[0][0], x1 = LOGO[LOGO.length - 1][0], out = new Float64Array(n);
    let j = 0;
    for (let i = 0; i < n; i++) {
      const x = x0 + (x1 - x0) * i / (n - 1);
      while (j < dense.length - 2 && dense[j + 1][0] < x) j++;
      const [a, b] = [dense[j], dense[j + 1]], f = b[0] > a[0] ? clamp((x - a[0]) / (b[0] - a[0])) : 0;
      out[i] = (a[1] + (b[1] - a[1]) * f) / 600;
    }
    return out;
  }

  function prepare(d) {
    const n = d.wave.length;
    const cont = d.wave.map((_, i) => d.groups.filter((g) => g.key !== "narrow").reduce((s, g) => s + g.flux[i], 0));
    d.ymax = Math.max(...cont) * 1.2;                   // narrow lines may rise above the scale; their labels say how far
    d.obsDomain = [d.wave[0] * (1 + d.z) - 40, d.wave[n - 1] * (1 + d.z) + 30];
    d.restDomain = [d.wave[0] - 40, d.wave[n - 1] + 30];
    d.resid = d.data.map((v, i) => (v != null && d.err[i] > 0 ? (v - d.model[i]) / d.err[i] : null));
    d.gaps = [];
    for (let i = 0; i < n; i++) if (d.data[i] == null) {
      let j = i; while (j + 1 < n && d.data[j + 1] == null) j++;
      d.gaps.push([d.wave[Math.max(0, i - 1)], d.wave[Math.min(n - 1, j + 1)]]); i = j;
    }
    d.byKey = Object.fromEntries(d.groups.map((g) => [g.key, g]));
    const S = d.stages;
    const im = d.image;
    d.story = [[0, `${im.object} · a Seyfert 1 galaxy at z = ${d.z.toFixed(3)} · ${im.survey} image, ${im.fov}″ across`],
      [T.fiber[0], `an SDSS fiber, ${im.fiber}″ across, collects the light of the nucleus`],
      [T.beam[0], "the spectrograph spreads that light by wavelength"],
      [T.logo[0] + 0.4, "… into a spectrum, and FANTASY takes it from there"],
      [T.morph[0] + 0.4, S.observed], [T.shift[0], S.rest],
      ...d.groups.map((g) => [T[g.key][0], S[g.key]]), [T.model[0], S.model]];
    logoY = sampleLogo(n);
    sky.onload = () => { if (!running) frame(performance.now()); };
    sky.src = im.src;
    return d;
  }

  function size() {
    const r = cv.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    W = Math.max(280, r.width); H = Math.max(200, r.height);
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (!running) frame(performance.now());
  }

  function frame(now) {
    if (!D) return;
    const t = reduce ? T.end + 1 : (now - start) / 1000;
    const phone = W < 560;
    const L = { l: phone ? 40 : 52, r: 12, t: 26, rh: phone ? 22 : 26 };
    const pw = W - L.l - L.r, ph = H - L.t - (L.rh + 58);
    const ybase = L.t + ph, ry0 = ybase + 8 + L.rh / 2, bandY = ybase + 8 + L.rh + 8;
    const n = D.wave.length;
    ctx.clearRect(0, 0, W, H);

    const mo = phase(t, T.morph), sh = phase(t, T.shift);
    const lo = D.obsDomain[0] + (D.restDomain[0] - D.obsDomain[0]) * sh;
    const hi = D.obsDomain[1] + (D.restDomain[1] - D.obsDomain[1]) * sh;
    const X = (lam) => L.l + (lam - lo) / (hi - lo) * pw;
    const Y = (f) => ybase - f / D.ymax * ph;
    const lamAt = (i) => D.wave[i] * (1 + D.z * (1 - sh));

    // the logo's place: centred, as tall as the plot
    const small = phone ? 64 : 104;                     // the sky image once the spectrograph is in
    const lh = Math.min(ph + 20, H * 0.72), lw = lh * (475 / 600), ly = L.t + 4;
    const lx = L.l + Math.max((pw - lw) / 2, small + (phone ? 44 : 64));
    const lbase = ly + lh * (562 / 600);                // the logo stands on its lowest point
    const lg = phase(t, T.logo);                        // the ribbon rises from the dispersed light
    const logoX = (i) => lx + lw * i / (n - 1), logoYp = (i) => lbase + (ly + logoY[i] * lh - lbase) * lg;

    // axes come in with the spectrum
    ctx.globalAlpha = mo;
    ctx.lineWidth = 1; ctx.strokeStyle = GRID; ctx.fillStyle = AXIS;
    ctx.font = "11px 'IBM Plex Mono', ui-monospace, Menlo, monospace";
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let v = Math.ceil(lo / 500) * 500; v <= hi; v += 500) {
      const x = X(v);
      ctx.beginPath(); ctx.moveTo(x, L.t); ctx.lineTo(x, ybase); ctx.stroke();
      if (!phone || v % 1000 === 0) ctx.fillText(v.toFixed(0), x, bandY + 10);
    }
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    const ystep = D.ymax > 200 ? 100 : D.ymax > 80 ? 25 : 10;
    for (let v = 0; v < D.ymax; v += ystep) {
      const y = Y(v);
      ctx.beginPath(); ctx.moveTo(L.l, y); ctx.lineTo(L.l + pw, y); ctx.stroke();
      ctx.fillText(String(v), L.l - 8, y);
    }
    ctx.strokeStyle = "#cfd3d9"; ctx.beginPath(); ctx.moveTo(L.l, ybase + 0.5); ctx.lineTo(L.l + pw, ybase + 0.5); ctx.stroke();
    ctx.textBaseline = "alphabetic"; ctx.textAlign = "left"; ctx.fillStyle = FAINT;
    ctx.fillText(`F_λ  [${D.unit}]`, L.l, L.t - 10);
    ctx.textAlign = "right";
    ctx.fillText(sh < 0.5 ? "observed wavelength (Å)" : "rest-frame wavelength (Å)", L.l + pw, H - 3);
    const band = ctx.createLinearGradient(L.l, 0, L.l + pw, 0);
    for (let k = 0; k <= 24; k++) band.addColorStop(k / 24, rgba(lambdaRGB(lo + (hi - lo) * k / 24), 1));
    ctx.fillStyle = band; ctx.fillRect(L.l, bandY, pw, 3);
    ctx.globalAlpha = 1;

    // the wordmark under the logo, fading out as the spectrum appears
    const wa = phase(t, T.word) * (1 - phase(t, T.unword));
    if (wa > 0) {
      ctx.globalAlpha = wa; ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
      const fs = Math.max(18, Math.min(34, lw * 0.15));
      ctx.font = `500 ${fs}px 'IBM Plex Sans', -apple-system, sans-serif`;
      ctx.fillStyle = "#454a52";
      if ("letterSpacing" in ctx) ctx.letterSpacing = `${(fs * 0.55).toFixed(1)}px`;
      ctx.fillText("FANTASY", W / 2 + fs * 0.27, Math.min(H - 26, ly + lh + fs * 0.9));
      if ("letterSpacing" in ctx) ctx.letterSpacing = "0px";
      ctx.font = `500 ${Math.max(10, fs * 0.36)}px 'IBM Plex Sans', -apple-system, sans-serif`;
      ctx.fillStyle = "#1f9aa8";
      ctx.fillText("Fully Automated pythoN Tool for AGN Spectra analYsis", W / 2, Math.min(H - 8, ly + lh + fs * 1.55));
      ctx.globalAlpha = 1;
    }

    ctx.save();
    ctx.beginPath(); ctx.rect(L.l, mo < 1 ? 0 : L.t, pw, mo < 1 ? H : ph); ctx.clip();

    // stacked fitted components: host galaxy at the bottom
    const g = {};
    const base = new Float64Array(n);
    for (const grp of D.groups) {
      const a = g[grp.key] = phase(t, T[grp.key]);
      if (a <= 0) continue;
      const top = new Float64Array(n);
      for (let i = 0; i < n; i++) top[i] = base[i] + grp.flux[i] * a;
      const c = COLOR[grp.key];
      ctx.beginPath();
      for (let i = 0; i < n; i++) ctx.lineTo(X(lamAt(i)), Y(top[i]));
      for (let i = n - 1; i >= 0; i--) ctx.lineTo(X(lamAt(i)), Y(base[i]));
      ctx.closePath();
      const fill = ctx.createLinearGradient(0, L.t, 0, ybase);
      fill.addColorStop(0, rgba(c, 0.42)); fill.addColorStop(1, rgba(c, grp.key === "host" ? 0.16 : 0.10));
      ctx.fillStyle = fill; ctx.fill();
      ctx.beginPath();
      for (let i = 0; i < n; i++) ctx.lineTo(X(lamAt(i)), Y(top[i]));
      ctx.strokeStyle = rgba(c, 0.95); ctx.lineWidth = 1.2; ctx.stroke();
      base.set(top);
    }

    // the logo ribbon, collapsing into one line that becomes the spectrum
    const target = (i) => [X(lamAt(i)), Y(D.data[i] ?? D.model[i])];
    if (mo < 1 && lg > 0) {
      const reveal = n;
      for (let k = RIBBON - 1; k >= 0; k--) {
        const spread = (1 - mo) * k / (RIBBON - 1);
        const dx = spread * lw * 0.05, dy = spread * lh * 0.16 * lg;
        const grad = ctx.createLinearGradient(lx, 0, lx + lw, 0);
        for (let s = 0; s <= 20; s++) {
          const u = s / 20, wl = lambdaRGB(lo + (hi - lo) * clamp((lx + lw * u - L.l) / pw));
          grad.addColorStop(u, rgba(mix(logoRGB(u), wl, mo), k ? 0.55 * (1 - mo) + 0.05 : 1));
        }
        ctx.beginPath();
        for (let i = 0; i < reveal; i++) {
          const [tx, ty] = target(i);
          const x = logoX(i) * (1 - mo) + tx * mo + dx, y = logoYp(i) * (1 - mo) + ty * mo + dy;
          ctx.lineTo(x, y);
        }
        ctx.strokeStyle = grad; ctx.lineWidth = k ? 0.8 : 1.6; ctx.stroke();
      }
    } else if (mo >= 1) {
      // the data, in the colours of their wavelengths, then grey once the model is in
      const m = phase(t, T.host);
      const ink = ctx.createLinearGradient(L.l, 0, L.l + pw, 0);
      for (let k = 0; k <= 24; k++) {
        const c = lambdaRGB(lo + (hi - lo) * k / 24);
        ink.addColorStop(k / 24, rgba(mix(c, [120, 126, 138], m), 1 - 0.35 * m));
      }
      for (const [a, b] of D.gaps) {
        const f = 1 + D.z * (1 - sh);
        ctx.fillStyle = "rgba(123,130,144,.14)";
        ctx.fillRect(X(a * f), L.t, Math.max(2, X(b * f) - X(a * f)), ph);
      }
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        if (D.data[i] == null) continue;
        if (i && D.data[i - 1] == null) ctx.moveTo(X(lamAt(i)), Y(D.data[i])); else ctx.lineTo(X(lamAt(i)), Y(D.data[i]));
      }
      ctx.strokeStyle = ink; ctx.lineWidth = 1.3; ctx.lineJoin = "round"; ctx.stroke();
    }

    // the full model
    const mp = phase(t, T.model);
    if (mp > 0) {
      ctx.beginPath();
      for (let i = 0; i < Math.round(mp * n); i++) ctx.lineTo(X(lamAt(i)), Y(D.model[i]));
      ctx.strokeStyle = INK; ctx.lineWidth = 1.4; ctx.stroke();
    }
    ctx.restore();

    // prologue: the galaxy, the fiber on its nucleus, the spectrograph and its fan of light
    const ia = phase(t, T.img) * (1 - phase(t, T.imgout));
    if (ia > 0 && sky.complete && sky.naturalWidth) {
      const big = Math.min(ph + 24, pw * 0.55), mv = phase(t, T.beam);
      const S = big + (small - big) * mv;
      const cx = (L.l + pw / 2) * (1 - mv) + (L.l + small / 2) * mv, cy = L.t + ph / 2 - 4;
      const zoom = 1.06 - 0.06 * phase(t, T.img);
      ctx.save();
      ctx.globalAlpha = ia;
      ctx.beginPath(); ctx.roundRect ? ctx.roundRect(cx - S / 2, cy - S / 2, S, S, 8) : ctx.rect(cx - S / 2, cy - S / 2, S, S);
      ctx.clip();
      ctx.drawImage(sky, cx - S * zoom / 2, cy - S * zoom / 2, S * zoom, S * zoom);
      ctx.restore();
      ctx.globalAlpha = ia;
      // the fiber: its true size on the sky
      const fb = phase(t, T.fiber);
      if (fb > 0) {
        const r = S * zoom * (D.image.fiber / 2) / D.image.fov;
        ctx.strokeStyle = `rgba(255,255,255,${0.95 * fb})`; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(cx, cy, r * (1 + 3 * (1 - fb)), 0, 6.283); ctx.stroke();
        if (mv < 0.5) {
          ctx.fillStyle = `rgba(255,255,255,${fb * (1 - 2 * mv)})`; ctx.font = "11px 'IBM Plex Mono', monospace";
          ctx.textAlign = "left"; ctx.textBaseline = "middle";
          ctx.beginPath(); ctx.moveTo(cx + r + 3, cy); ctx.lineTo(cx + r + 16, cy); ctx.strokeStyle = `rgba(255,255,255,${fb * 0.8})`; ctx.stroke();
          const tag = `fiber ${D.image.fiber}″`, tw = ctx.measureText(tag).width;   // legible on the sky and off it
          ctx.fillStyle = `rgba(29,33,40,${0.78 * fb * (1 - 2 * mv)})`;
          ctx.fillRect(cx + r + 16, cy - 8, tw + 11, 16);
          ctx.fillStyle = `rgba(255,255,255,${fb * (1 - 2 * mv)})`;
          ctx.fillText(tag, cx + r + 20, cy);
        }
      }
      ctx.fillStyle = "rgba(255,255,255,.75)"; ctx.font = "10px 'IBM Plex Mono', monospace";
      ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
      if (S > 90) ctx.fillText(`${D.image.survey} · ${D.image.fov}″`, cx - S / 2 + 7, cy + S / 2 - 7);
      // the light leaves the fiber for the spectrograph
      const bm = phase(t, T.beam), fa = phase(t, T.fan);
      if (bm > 0.6) {
        const px = L.l + small + (phone ? 18 : 28), py = cy, sx = cx;
        const k = clamp((bm - 0.6) / 0.4);
        ctx.strokeStyle = "rgba(29,33,40,.55)"; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.moveTo(sx, py); ctx.lineTo(sx + (px - 8 - sx) * k, py); ctx.stroke();
        ctx.fillStyle = "rgba(240,242,245,.95)"; ctx.strokeStyle = "rgba(29,33,40,.7)"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(px - 8, py + 9); ctx.lineTo(px + 8, py + 9); ctx.lineTo(px, py - 9); ctx.closePath();
        ctx.globalAlpha = ia * k; ctx.fill(); ctx.stroke(); ctx.globalAlpha = ia;
        // a fan of rays, each wavelength to its place along the spectrum
        if (fa > 0) {
          const fade = 1 - 0.85 * lg;
          for (let s = 0; s <= 48; s++) {
            const u = s / 48, x = lx + lw * u, c = lambdaRGB(3900 + 3100 * u);    // blue to red, as a prism spreads it
            ctx.strokeStyle = rgba(c, 0.55 * fade); ctx.lineWidth = 1.1;
            ctx.beginPath(); ctx.moveTo(px + 4, py);
            ctx.lineTo(px + 4 + (x - px - 4) * fa, py + (lbase - py) * fa); ctx.stroke();
          }
        }
      }
      ctx.globalAlpha = 1;
    }

    // line identifications: emission above the spectrum, stellar absorption below it
    ctx.font = "11px 'IBM Plex Sans', -apple-system, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
    let lastX = -1e9, lift = 0;
    for (const [name, lam, kind] of D.labels) {
      if (phone && /Hγ|He I|G band|\[S II\]/.test(name)) continue;
      const a = kind === "absorption" ? g.host : Math.max(g.broad || 0, g.narrow || 0);
      if (!a || a <= 0.05) continue;
      const x = X(lam);
      if (x < L.l + 10 || x > L.l + pw - 10) continue;
      const i = Math.max(0, D.wave.findIndex((w) => w >= lam));
      ctx.globalAlpha = a;
      if (kind === "absorption") {
        const y = Math.min(ybase - 6, Y(D.byKey.host.flux[i]) + 22);
        ctx.fillStyle = "rgb(150,98,10)"; ctx.fillText(name, x, y);
      } else {
        lift = x - lastX < 50 ? (lift + 13) % 39 : 0;
        lastX = x;
        let top = 0, from = "";
        for (let j = Math.max(0, i - 6); j < Math.min(n, i + 7); j++) {
          const v = D.data[j] ?? D.model[j];
          if (v > top) { top = v; from = D.data[j] == null ? " model" : ""; }
        }
        const y = Math.max(L.t + 12, Y(Math.max(D.model[i], D.data[i] ?? 0)) - 10 - lift);
        ctx.fillStyle = "rgb(181,55,29)";
        ctx.fillText(top > D.ymax ? `${name} ↑${top.toFixed(0)}${from}` : name, x, y);
      }
    }
    ctx.globalAlpha = 1;

    // normalised residuals (data − model) / error
    const rp = phase(t, T.resid);
    ctx.globalAlpha = mo;
    ctx.strokeStyle = GRID; ctx.beginPath(); ctx.moveTo(L.l, ry0); ctx.lineTo(L.l + pw, ry0); ctx.stroke();
    ctx.fillStyle = FAINT; ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.font = "10px 'IBM Plex Mono', monospace";
    ctx.fillText("res/σ", L.l - 6, ry0);
    ctx.globalAlpha = 1;
    if (rp > 0) {
      ctx.save(); ctx.beginPath(); ctx.rect(L.l, ry0 - L.rh / 2, pw, L.rh); ctx.clip();
      ctx.beginPath();
      for (let i = 0; i < Math.round(rp * n); i++) {
        const y = ry0 - clamp(D.resid[i] ?? 0, -5, 5) / 5 * (L.rh / 2);
        if (D.resid[i] == null || (i && D.resid[i - 1] == null)) ctx.moveTo(X(lamAt(i)), y); else ctx.lineTo(X(lamAt(i)), y);
      }
      ctx.strokeStyle = "#8b929e"; ctx.lineWidth = 0.9; ctx.stroke();
      ctx.restore();
    }
    ctx.textBaseline = "alphabetic";

    // pointer readout: what makes up the light at this wavelength
    if (hover && t > T.shift[1] && hover.x >= L.l && hover.x <= L.l + pw) {
      const lam = lo + (hover.x - L.l) / pw * (hi - lo);
      let i = D.wave.findIndex((w) => w >= lam);
      if (i < 0) i = n - 1;
      const x = X(D.wave[i]);
      ctx.strokeStyle = "rgba(29,33,40,.35)"; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(x, L.t); ctx.lineTo(x, ybase); ctx.stroke(); ctx.setLineDash([]);
      let acc = 0;
      for (const grp of D.groups) {
        if (g[grp.key] < 1) continue;
        acc += grp.flux[i];
        ctx.fillStyle = rgba(COLOR[grp.key], 1); ctx.beginPath(); ctx.arc(x, Y(acc), 3, 0, 6.283); ctx.fill();
      }
      const near = D.labels.reduce((b, l) => (Math.abs(l[1] - D.wave[i]) < Math.abs(b[1] - D.wave[i]) ? l : b), D.labels[0]);
      const id = Math.abs(near[1] - D.wave[i]) < 20 ? ` · ${near[0]}` : "";
      const f = (v) => v.toFixed(1);
      const data = D.data[i] == null ? "pixels flagged by SDSS, not fitted" : `data ${f(D.data[i])} ± ${f(D.err[i])}`;
      readout.innerHTML = `<b>λ<sub>rest</sub> ${D.wave[i].toFixed(1)} Å</b>${id}<br>${data} · model ${f(D.model[i])}<br>` +
        D.groups.map((grp) => `<i style="color:${rgba(COLOR[grp.key], 1)}">${grp.label} ${f(grp.flux[i])}</i>`).join(" · ");
      readout.classList.add("on");
    } else readout.classList.remove("on");

    // the story so far
    let label = "";
    for (const [at, text] of D.story) if (t >= at) label = text;
    if (label !== lastStage) { stage.textContent = label; stage.classList.remove("in"); void stage.offsetWidth; stage.classList.add("in"); lastStage = label; }

    // the legend, entry by entry as the curves come in
    for (const el of legend.children) el.classList.toggle("on", t >= (el.dataset.k ? T[el.dataset.k][0] : el.classList.contains("k-model") ? T.model[0] : T.morph[1] - 0.3));
    if (t >= T.model[0] && !statsShown) showStats(reduce);
    hero.classList.toggle("done", t >= T.end - 0.2);    // FANTASY² drops into place
    running = t < T.end && shown && !document.hidden && hero.offsetParent !== null;    // still once it is told
    if (running) raf = requestAnimationFrame(frame);
  }

  // the measured numbers, counting up
  function showStats(instant) {
    statsShown = true;
    hero.querySelectorAll("[data-count]").forEach((el) => {
      const target = +el.dataset.count, dec = +el.dataset.dec || 0, t0 = performance.now();
      const tick = (now) => {
        const p = instant ? 1 : ease(clamp((now - t0) / 1200));
        el.textContent = (target * p).toFixed(dec);
        if (p < 1) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
    hero.classList.add("measured");
  }

  function fillStats() {
    const label = (s) => s.replace(/_BH/, "<sub>BH</sub>");
    document.getElementById("herostats").innerHTML = D.stats.map((s) =>
      `<div><small>${label(s.label)}</small><b data-count="${s.value}" data-dec="${s.dec}">–</b>` +
      `<span>${s.unit} <em>± ${s.err.toFixed(s.dec)}</em></span></div>`).join("");
    document.getElementById("heroobj").textContent = `${D.name} · z = ${D.z.toFixed(4)}`;
    document.getElementById("herolegend").innerHTML = '<i class="k-data">data</i>' +
      D.groups.map((grp) => `<i data-k="${grp.key}" style="color:${rgba(COLOR[grp.key], 1)}"><span>${grp.label}</span></i>`).join("") + '<i class="k-model">model</i>';
  }

  function replay() {
    start = performance.now(); statsShown = false; lastStage = "";
    hero.classList.remove("measured", "done");
    hero.querySelectorAll("[data-count]").forEach((el) => (el.textContent = "–"));
    wake();
  }
  function wake() {
    if (!running && shown && !document.hidden && hero.offsetParent !== null && D) { running = true; raf = requestAnimationFrame(frame); }
  }

  // the app shows the hero on the Read page until a spectrum is read
  window.fantasyHero = {
    show(on) {
      if (on !== shown) {
        shown = on;
        hero.classList.toggle("hidden", !on);
        if (!on) { cancelAnimationFrame(raf); running = false; }
      }
      if (on) wake();
    },
  };

  const redraw = () => { if (!running) frame(performance.now()); };
  cv.addEventListener("pointermove", (e) => { hover = { x: e.clientX - cv.getBoundingClientRect().left }; redraw(); });
  cv.addEventListener("pointerleave", () => { hover = null; redraw(); });
  document.getElementById("heroreplay").onclick = replay;
  document.getElementById("herotry").onclick = () => window.fantasyTryExample && window.fantasyTryExample(hero.dataset.example);
  document.addEventListener("visibilitychange", wake);
  new ResizeObserver(() => size()).observe(cv);

  fetch("/static/hero.json").then((r) => r.json()).then((d) => {
    D = prepare(d); fillStats(); size(); start = performance.now(); wake();
  }).catch(() => hero.classList.add("hidden"));
})();
