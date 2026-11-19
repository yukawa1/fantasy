// Fit step of a sample: one tile per spectrum, filled in as the worker processes finish them (GET /api/wf/samplelive,
// Session.sample_live_state, the `item` of fantasy_agn.progress.report in fit_sample). A finished tile shows the fitted
// range (data and model), the class and χ²ᵣ; a failed one its error. Once the fit is done a tile opens that spectrum in
// Results. Uses api, esc, CLASS_COLOR and openSampleFit of wf.js.
(() => {
  const $ = (id) => document.getElementById(id);
  const MAX = 600;                       // tiles drawn; the rest are counted
  const SHORT = { type1: "Type 1", type1_host: "Type 1 + host", seyfert2: "Seyfert 2", liner: "LINER", composite: "Composite",
    starforming: "Star-forming", retired: "Retired", passive: "Passive", unknown: "Not classified", uv_mgii: "Mg II (UV)", uv_civ: "C IV (UV)" };
  const T = { run: null, since: 0, total: 0, items: {}, poll: 0, active: false, running: false, built: 0, elapsed: 0 };
  const ok = (v) => v != null && isFinite(v);
  const quality = (q) => (!ok(q) ? "" : q <= 1.5 ? "ok" : q <= 3 ? "warn" : "bad");

  async function fetchTiles() {
    T.poll = 0;
    let again = false;
    try {
      const d = await api(`/api/wf/samplelive?since=${T.since}`);
      if (!d.total) { Object.assign(T, { run: null, since: 0, total: 0, items: {}, built: 0 }); }
      else if (d.run !== T.run) {                       // a new sample fit: start from its first spectrum
        Object.assign(T, { run: d.run, since: 0, total: d.total, items: {}, built: 0 });
        again = true;
      } else {
        d.items.forEach((it) => { T.items[it.k] = Object.assign(it, { fresh: true }); });
        T.since = Math.max(T.since, d.seq); T.running = d.running; T.current = !!d.current;
      }
    } catch (e) { /* the next poll tries again */ }
    draw();
    if (T.active || again) T.poll = setTimeout(fetchTiles, again ? 0 : 700);
  }

  function spark(th, color) {
    if (!th || !th.x || th.x.length < 2) return '<div class="st-noplot"></div>';
    const W = 120, H = 46, x = th.x, all = th.y.concat(th.m).filter(ok).sort((a, b) => a - b);
    const q = (f) => all[Math.min(all.length - 1, Math.max(0, Math.floor(f * (all.length - 1))))];
    let lo = q(0.01), hi = q(0.995); const pad = 0.08 * (hi - lo || 1); lo -= pad; hi += pad;
    const X = (v) => ((v - x[0]) / (x[x.length - 1] - x[0] || 1)) * W, Y = (v) => H - ((v - lo) / (hi - lo || 1)) * H;
    const path = (a) => a.map((v, i) => (ok(v) ? `${i && ok(a[i - 1]) ? "L" : "M"}${X(x[i]).toFixed(1)},${Math.min(H + 2, Math.max(-2, Y(v))).toFixed(1)}` : "")).join("");
    return `<svg class="st-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">` +
      `<path class="st-d" d="${path(th.y)}"/><path class="st-m" style="stroke:${color}" d="${path(th.m)}"/></svg>`;
  }

  function tile(it) {
    if (it.error) {
      return `<div class="st-plot st-fail">failed</div><div class="st-name" title="${esc(it.name)}">${esc(it.name)}</div>` +
        `<div class="st-meta st-err" title="${esc(it.error)}">${esc(it.error.split(":")[0])}</div>`;
    }
    const color = CLASS_COLOR[it.cls] || "#6b7280", what = it.cls ? SHORT[it.cls] || it.cls : (it.preset || "").replace(/\(\)$/, "");
    const lim = it.n_at_limit ? ` <span class="st-lim" title="${it.n_at_limit} parameter${it.n_at_limit > 1 ? "s" : ""} at a limit">${it.n_at_limit} at limit</span>` : "";
    return `<div class="st-plot">${spark(it.thumb, color)}</div><div class="st-name" title="${esc(it.name)} · ${esc(it.preset || "")}">${esc(it.name)}</div>` +
      `<div class="st-meta"><span class="sgdot" style="background:${color}"></span><span class="st-cls">${esc(what)}</span>` +
      `<span class="grow"></span><span class="q ${quality(it.rchi2)}">${ok(it.rchi2) ? it.rchi2.toFixed(2) : "—"}</span></div>` + (lim ? `<div class="st-meta">${lim}</div>` : "");
  }

  function draw() {
    const el = $("st");
    if (!el) return;
    const show = T.active || (T.total > 0 && T.keep && T.current);     // kept only under the sample this run made
    el.classList.toggle("hidden", !show || !T.total);
    if (!show || !T.total) return;
    const n = Math.min(T.total, MAX), grid = $("st-grid");
    if (T.built !== n || grid.children.length !== n) {         // the empty tiles, in the order of the input
      grid.innerHTML = Array.from({ length: n }, (_, k) => `<div class="st-tile wait" data-k="${k}"><div class="st-plot"></div><div class="st-name">#${k}</div></div>`).join("");
      T.built = n;
      Object.values(T.items).forEach((it) => (it.fresh = true));
    }
    Object.values(T.items).forEach((it) => {
      if (!it.fresh || it.k >= n) return;
      it.fresh = false;
      const t = grid.children[it.k];
      t.className = `st-tile ${it.error ? "failed" : "done"} pop`;
      t.innerHTML = tile(it);
      t.title = it.error ? it.error : `#${it.k} ${it.name}: ${it.preset || ""}, χ²ᵣ ${ok(it.rchi2) ? it.rchi2.toFixed(2) : "—"}` + (ok(it.time) ? `, ${it.time.toFixed(1)} s` : "");
    });
    grid.classList.toggle("live", T.active);
    grid.classList.toggle("open", !T.active);
    const items = Object.values(T.items), done = items.length, failed = items.filter((x) => x.error).length;
    const counts = {};
    items.forEach((x) => { if (!x.error) { const c = x.cls || "_model"; counts[c] = (counts[c] || 0) + 1; } });
    $("st-keys").innerHTML = Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([c, v]) =>
      `<span class="cs-key"><i style="background:${CLASS_COLOR[c] || "#6b7280"}"></i>${esc(c === "_model" ? "fitted" : SHORT[c] || c)} <b>${v}</b></span>`).join("");
    const q = items.filter((x) => !x.error && ok(x.rchi2)).map((x) => x.rchi2).sort((a, b) => a - b);
    const med = q.length ? q[Math.floor(q.length / 2)] : null, poor = q.filter((v) => v > 3).length;
    const left = T.total - done, rate = done && T.elapsed > 0 ? T.elapsed / done : null;
    $("st-title").textContent = T.active ? `${done} of ${T.total} spectra fitted` : `The ${T.total} spectra of this fit`;
    const facts = [];
    if (failed) facts.push(`<span class="warnword">${failed} failed</span>`);
    if (med != null) facts.push(`median χ²ᵣ ${med.toFixed(2)}`);
    if (poor) facts.push(`${poor} with χ²ᵣ > 3`);
    if (T.active && rate && left > 0) facts.push(`about ${fmt(rate * left)} left`);
    if (!T.active) facts.push("click a spectrum to open its fit in Results");
    $("st-sub").innerHTML = facts.join(" · ");
    $("st-bar").style.width = `${(100 * done) / T.total}%`;
    $("st-bar").parentElement.classList.toggle("hidden", !T.active);
    $("st-more").classList.toggle("hidden", T.total <= MAX);
    $("st-more").textContent = T.total > MAX ? `The first ${MAX} of ${T.total} spectra are shown; all are in the catalog.` : "";
  }
  const fmt = (s) => (s < 90 ? `${Math.max(1, Math.round(s))} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`);

  // active: a sample fit is running (poll); keep: it is done and its fit is the current one (show, click to open)
  function run(active, keep, elapsed) {
    if (keep && !T.keep && !active) T.asked = false;                 // a fit just ended: ask whether its tiles are current
    T.keep = keep; T.elapsed = elapsed || T.elapsed;
    if (active && !T.active) { T.active = true; if (!T.poll) fetchTiles(); }
    else if (!active && T.active) { T.active = false; clearTimeout(T.poll); T.poll = 0; fetchTiles(); }
    else if (keep && !T.poll && !T.asked) { T.asked = true; fetchTiles(); }   // after a page reload, or once the fit ends
    draw();
  }
  document.addEventListener("click", (e) => {
    const t = e.target.closest && e.target.closest("#st-grid.open .st-tile.done, #st-grid.open .st-tile.failed");
    if (t) openSampleFit(+t.dataset.k);
  });
  window.ST = { run };
})();
