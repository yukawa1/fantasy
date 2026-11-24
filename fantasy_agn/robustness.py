"""
How robust is a fit? chi2 scans of the parameters and the range of the results over all acceptable fits.

The covariance errors of a fit describe the chi2 surface next to the best fit. They miss what lies further away:
a width that can slide up to its limit at a small chi2 cost, a second minimum, or a better fit the optimizer did
not find. :func:`robustness` scans every free non-amplitude parameter: it fixes the parameter on a grid that spans
its limits, refits everything else and records chi2. From these scans::

    rob = fa.robustness(result)
    rob.parameters     # per parameter: 68% and 95% intervals, dchi2 at the limits, flag
    rob.quantities     # per result (broad Hb FWHM, R_FeII, [OIII] ...): range over all acceptable fits
    rob.alternatives   # second minima (other solutions) up to dchi2 = 9
    rob.profiles       # every scan; "result: <name>" are the direct scans of the results
    rob.summary()
    rob.plot()

Each grid point is fitted twice, from the best fit (``warm``) and from a fresh initial guess (``fresh``), and the
lower chi2 is kept, so a scan can find solutions the original fit missed. If a scan finds a fit better than the
original one, it becomes the reference (``rob.best``) and ``rob.improved`` is True.

dchi2 is in units of the scaled errors (divided by ``error_scale**2``), like the covariance errors. The intervals
are dchi2 = 1 (68%) and 4 (95%) for one parameter. Each result (a quantity such as the broad Hb FWHM) is also
scanned directly: it is held at values around its best one by a stiff penalty while every free parameter is
refitted, walking outwards from the best fit, and its range is where dchi2 = 1 and 4, widened to any scan point of
the parameters that is acceptable and lies further out. Scanning only the parameters would miss the direction in
which a result changes most and give ranges that are too narrow: in 80 simulated SDSS spectra they held the truth in
40% of the cases at 68% (70% at 95%), the direct scans in 65% (96%).

Quantities are functions of a FitResult returning (value, error) or a value, as in :func:`fantasy_agn.compare`;
the default is the main broad line (FWHM, sigma, flux, shift), R_FeII and the total [OIII]5007 line (flux, FWHM,
shift), whichever the model has.
"""
import numpy as np
import pandas as pd

__all__ = ("robustness", "Robustness")

LEVELS = (1.0, 4.0)          # dchi2 of the 68% and 95% intervals of one parameter
BETTER = 0.5                 # a scan point this much below the best fit is a better fit
ALTERNATIVE = 9.0            # second minima up to this dchi2 are reported
WEAK = 16.0                  # a closed interval with dchi2 below this at a limit is only weakly bounded


def _value(out):
    """A quantity function may return (value, error) or a value."""
    v = out[0] if isinstance(out, (tuple, list)) else out
    return float(v)


def default_quantities(result):
    """{name: function(FitResult) -> (value, error)}: main broad line, R_FeII and total [OIII]5007."""
    from .compare import default_quantities as broad_quantities
    q = {"broad " + k: v for k, v in broad_quantities(result).items() if not k.startswith(("log M_BH", "log L"))}
    comps = result._line_components()
    broad = [c for c in comps if c.role == "broad"]
    line = result._primary_broad_line()
    if line:
        def shift(r):
            m = r.measure(line, [c.name for c in r._line_components() if c.role == "broad" and line in c.line_names])
            return m["shift"], m["shift_err"]
        q[f"broad {line} shift"] = shift
    for comp in result._feii_components():
        band = result._feii_band(comp)
        if not any(band["reference"] in c.line_names for c in broad):
            continue
        def rfe(r, name=comp.name):
            f = r.feii(name)
            return f["r_feii"], f["r_feii_err"]
        q[band["ratio_label"]] = rfe
    if any("[OIII]5007" in c.line_names for c in comps):
        def oiii(key):
            def f(r):
                m = r.measure("[OIII]5007")
                return m[key], m[key + "_err"]
            return f
        q.update({"[OIII]5007 total flux": oiii("flux"), "[OIII]5007 total FWHM": oiii("fwhm"),
                  "[OIII]5007 total shift": oiii("shift")})
    return q


def _grid(p, best, sigma, n_grid):
    """Points across the limits of p plus points near the best value (finite limits: the ends are included)."""
    lo, hi = p.min, p.max
    s = sigma if np.isfinite(sigma) and sigma > 0 else np.nan
    if not np.isfinite(lo) or not np.isfinite(hi):
        w = 10 * s if np.isfinite(s) else max(abs(best), 1.0)
        lo, hi = max(lo, best - w), min(hi, best + w)
    pts = list(np.linspace(lo, hi, n_grid))
    if not np.isfinite(s):
        s = (hi - lo) / 50
    pts += [best + k * s for k in (-3, -2, -1, -0.5, 0.5, 1, 2, 3)]
    pts = np.unique(np.round(np.clip(pts, lo, hi), 10))
    return pts[np.abs(pts - best) > 1e-9 * max(abs(best), 1.0)]


def _crossing(v, d, k, level, step):
    """Where dchi2 first exceeds level walking from index k; (value, reached the end of the scan)."""
    j = k
    while 0 <= j + step < len(v):
        if d[j + step] > level:
            return float(np.interp(level, [d[j], d[j + step]], [v[j], v[j + step]])), False
        j += step
    return float(v[j]), True


def _minima(v, d, k):
    """Local minima of a scan other than the global one (index k), separated from it by a barrier of dchi2 >= 1."""
    out = []
    for i in range(len(v)):
        if i == k or not np.isfinite(d[i]):
            continue
        left = d[i - 1] if i > 0 else np.inf
        right = d[i + 1] if i < len(v) - 1 else np.inf
        if d[i] < left and d[i] < right:
            a, b = sorted((i, k))
            if np.nanmax(d[a:b + 1]) - d[i] >= 1.0:
                out.append(i)
    return out


# ------------------------------------------------------------- profiles of the results
STEPS = (0.75, 1.5, 2.25, 3, 4, 5.5)         # targets at value + k * error, walked outwards from the best fit
PENALTY = 1e4                                # weight of (q - target)^2 / error^2: the target is met to ~1%


def _at(result, full):
    """A light copy of result with other parameter values (no errors: quantities are then fast)."""
    import copy
    r = copy.copy(result)
    r.full, r.cov, r.samples = np.asarray(full, float), None, None
    return r


def _quantity_profile(task):
    """
    chi2 profile of one result: the result is held at targets around its best value (a stiff penalty) while every
    free parameter is refitted, walking outwards from the best fit. Rows: target, value reached, chi2.
    """
    result, name, func, err = task
    from scipy.optimize import least_squares
    from .fitting import Problem, _interior
    f = func if func is not None else default_quantities(result)[name]
    g = result.mask
    s = result.spectrum
    prob = Problem(result.model, s.wave[g], s.flux[g], s.err[g], s.context)
    th0 = prob.theta_from_model()
    lb, ub = prob.lb, prob.ub
    q = lambda th: _value(f(_at(result, prob.full(th))))
    q0 = q(th0)
    if not np.isfinite(q0):
        return []
    scale = err if np.isfinite(err) and err > 0 else 0.05 * max(abs(q0), 1e-12)
    step = 1e-4 * np.where(np.isfinite(ub - lb), ub - lb, np.abs(th0) + 1.0)

    live = np.ones(th0.size, bool)                 # parameters the result depends on (found at each new target)

    def dq(th, base, which=None):
        out = np.zeros(th.size)
        for i in np.nonzero(live if which is None else which)[0]:
            h = step[i] if th[i] + step[i] <= ub[i] else -step[i]
            t2 = th.copy()
            t2[i] += h
            out[i] = (q(t2) - base) / h
        return out

    rows = [(q0, q0, float(np.sum(prob.residuals(th0) ** 2)))]
    for sign in (-1, 1):
        th = _interior(th0.copy(), lb, ub, prob.linear)
        last = q0
        for k in STEPS:
            target = q0 + sign * k * scale
            w = np.sqrt(PENALTY) / scale
            live[:] = dq(th, q(th), np.ones(th.size, bool)) != 0

            def res(t):
                return np.append(prob.residuals(t), w * (q(t) - target))

            def jac(t):
                return np.vstack([prob.jacobian(t), w * dq(t, q(t))])
            try:
                sol = least_squares(res, th, jac=jac, bounds=(lb, ub), x_scale="jac", max_nfev=100)
            except Exception:
                break
            reached = q(sol.x)
            chi2 = float(np.sum(prob.residuals(sol.x) ** 2))
            rows.append((target, reached, chi2))
            if not np.isfinite(reached) or abs(reached - last) < 0.05 * scale:
                break                                  # the result cannot be pushed further (a limit)
            th, last = _interior(sol.x, lb, ub, prob.linear), reached
            if (chi2 - rows[0][2]) / result.error_scale ** 2 > WEAK:
                break
    return rows


def _run_quantity_profiles(result, quantities, errs, custom, n_jobs):
    """{name: profile rows} for every result, in parallel processes (serially if they cannot be used)."""
    from concurrent.futures import ProcessPoolExecutor
    from ._threads import SINGLE_THREADED, environment
    from .fitting import _n_workers
    from .progress import report
    tasks = [(result, qn, f if custom else None, errs[qn]) for qn, f in quantities.items()]
    workers = _n_workers(n_jobs, len(tasks))
    out = None
    report("robustness: results", 0, len(tasks))
    if workers > 1:
        try:
            with environment(SINGLE_THREADED), ProcessPoolExecutor(workers) as ex:
                out = []
                for k, rows in enumerate(ex.map(_quantity_profile, tasks), 1):
                    out.append(rows)
                    report("robustness: results", k, len(tasks), f"{workers} processes")
        except Exception:                          # e.g. a quantity function that cannot be pickled
            out = None
    if out is None:
        out = []
        for k, task in enumerate(tasks, 1):
            out.append(_quantity_profile(task))
            report("robustness: results", k, len(tasks))
    return dict(zip(quantities, out))


def _profile_interval(rows, chi0, scale):
    """68% and 95% ranges of a result from its profile: where dchi2 = 1 and 4 (interpolated)."""
    if not rows:
        return {}
    v = np.array([r[1] for r in rows])
    d = (np.array([r[2] for r in rows]) - chi0) / scale
    o = np.argsort(v)
    v, d = v[o], d[o]
    k = int(np.nanargmin(d))
    out = {}
    for lev, tag in zip(LEVELS, ("68", "95")):
        (lo, elo), (hi, ehi) = _crossing(v, d, k, lev, -1), _crossing(v, d, k, lev, +1)
        out["lo" + tag], out["hi" + tag], out["open" + tag] = lo, hi, bool(elo or ehi)
    return out


class Robustness:
    """Result of :func:`robustness`; see the module docstring."""

    def __init__(self, best, improved, parameters, quantities, profiles, alternatives, n_fits, best_fixed=None):
        self.best, self.improved, self._best_fixed = best, improved, best_fixed
        self.parameters, self.quantities = parameters, quantities
        self.profiles, self.alternatives = profiles, alternatives
        self.n_fits = n_fits

    def best_model(self):
        """The model of ``best`` with the scanned parameter free again, to fit with errors."""
        m = self.best.model.copy()
        if self._best_fixed:
            m.params[self._best_fixed].fixed = False
        return m

    def summary(self):
        out = [f"Robustness of the fit ({self.n_fits} refits; dchi2 in units of the scaled errors)"]
        if self.improved:
            out.append(f"  a better fit was found (chi2 {self.best.chi2:.1f}); it is the reference below")
        flagged = self.parameters[self.parameters.flag != "measured"]
        if len(flagged):
            out.append("  parameters:")
            for _, r in flagged.iterrows():
                out.append(f"    {r.parameter}: {r.flag}" + (f" ({r.note})" if r.note else ""))
        else:
            out.append("  every scanned parameter is measured (closed 95% interval, no second minimum)")
        if len(self.quantities):
            out.append("  results over all acceptable fits (68% / 95% range):")
            for _, r in self.quantities.iterrows():
                out.append(f"    {r.quantity}: {r.value:.4g}  [{r.lo68:.4g}, {r.hi68:.4g}]  [{r.lo95:.4g}, {r.hi95:.4g}]"
                           f"  error bar {r.error:.2g}  {r.flag}")
        return "\n".join(out)

    def __repr__(self):
        return self.summary()

    def plot(self, parameters=None, ncols=4, fig=None):
        """dchi2 scans, one panel per parameter, with the 68% and 95% levels and the parameter limits."""
        import matplotlib.pyplot as plt
        names = list(parameters or self.profiles)
        nrows = max(1, -(-len(names) // ncols))
        if fig is None:
            fig = plt.figure(figsize=(2.0 * ncols, 1.6 * nrows))
        flags = dict(zip(self.parameters.parameter, self.parameters.flag))
        flags.update(("result: " + q, "measured" if f == "ok" else f) for q, f in zip(self.quantities.quantity, self.quantities.flag))
        for i, n in enumerate(names):
            a = fig.add_subplot(nrows, ncols, i + 1)
            p = self.profiles[n].sort_values("value")
            a.plot(p.value, p.dchi2, "o-", ms=2, lw=0.8, color="0.2")
            for lev in LEVELS:
                a.axhline(lev, color="0.6", ls=":", lw=0.6)
            a.set_ylim(-0.5, 12)
            a.set_title(n + ("" if flags.get(n) == "measured" else f"\n{flags.get(n)}"), fontsize=6)
            a.tick_params(labelsize=5)
        fig.tight_layout()
        return fig


def robustness(result, parameters=None, quantities=None, n_grid=12, starts=("warm", "fresh"), profile_quantities=True,
               n_jobs=-1):
    """
    chi2 scans of the free non-amplitude parameters of a fit and the range of its results over all acceptable fits.

    :param result: a FitResult.
    :param parameters: names to scan (default: every free parameter that is not an amplitude and is measured).
    :param quantities: {name: function(FitResult) -> (value, error) or value}; default :func:`default_quantities`.
    :param n_grid: points spread evenly between the limits of each parameter (8 more are added near the best value).
    :param starts: 'warm' (start from the best fit), 'fresh' (initial guess from the data, as a new fit) or both.
    :param profile_quantities: also scan each result directly (held at values around its best one while all free
        parameters are refitted); its 68% and 95% ranges then include the dchi2 = 1 and 4 points of that scan.
    :param n_jobs: processes (default: all cores but one).
    :return: :class:`Robustness`
    """
    from .predictive import _run
    spec, model = result.spectrum, result.model
    wave = spec.wave[result.mask]
    options = {"wave_range": (float(wave.min()), float(wave.max())), "errors": None, "verbose": False}
    starts = (starts,) if isinstance(starts, str) else tuple(starts)
    start_opts = {"warm": {"guess": False, "build_up": False}, "fresh": {}}
    names = model.params.names
    sig = result._sigma() * 1.0
    if parameters is None:
        parameters = [p.name for p in model.params if p.free and not p.linear and p.name not in result.not_measured]
    custom = quantities is not None
    quantities = default_quantities(result) if quantities is None else dict(quantities)

    jobs, keys = [], []
    for name in parameters:
        p = model.params[name]
        i = names.index(p.name)
        for v in _grid(p, result.full[i], sig[i], n_grid):
            m = model.copy()
            m.fix(p.name, float(v))
            for s in starts:
                ms = m
                if s == "fresh":                   # a fresh guess only: the warm start is the other job
                    ms = m.copy()
                    ms._fitted = False
                jobs.append((spec, ms, {**options, **start_opts[s]}))
                keys.append((p.name, float(v)))
    fits = _run(jobs, n_jobs, "robustness scans")

    # best fit per grid point over the starts
    pts = {}
    for key, r in zip(keys, fits):
        if isinstance(r, Exception) or not np.isfinite(r.chi2):
            continue
        if key not in pts or r.chi2 < pts[key].chi2:
            pts[key] = r
    best, improved, best_fixed = result, False, None
    scale = result.error_scale ** 2
    lowest = min(pts, key=lambda key: pts[key].chi2, default=None)
    if lowest is not None and (result.chi2 - pts[lowest].chi2) / scale > BETTER:
        best, improved, best_fixed = pts[lowest], True, lowest[0]
    chi0 = best.chi2

    def qvals(r):
        out = []
        for f in quantities.values():
            try:
                out.append(_value(f(r)))
            except Exception:
                out.append(np.nan)
        return out

    q0 = qvals(best)
    profiles, prows, alternatives, pooled = {}, [], [], [(0.0, q0)]
    for name in parameters:
        p = model.params[name]
        i = names.index(p.name)
        rows = [(v, r) for (n, v), r in pts.items() if n == p.name]
        rows.append((float(best.full[i]), best))
        rows.sort(key=lambda t: t[0])
        v = np.array([t[0] for t in rows])
        d = np.array([(t[1].chi2 - chi0) / scale for t in rows])
        qs = [qvals(r) if r is not best else q0 for _, r in rows]
        df = pd.DataFrame({"value": v, "chi2": [t[1].chi2 for t in rows], "dchi2": d})
        for j, qn in enumerate(quantities):
            df[qn] = [q[j] for q in qs]
        profiles[p.name] = df
        pooled += [(dd, q) for dd, q in zip(d, qs) if np.isfinite(dd)]

        k = int(np.nanargmin(d))
        (l1, e1l), (h1, e1h) = _crossing(v, d, k, LEVELS[0], -1), _crossing(v, d, k, LEVELS[0], +1)
        (l2, e2l), (h2, e2h) = _crossing(v, d, k, LEVELS[1], -1), _crossing(v, d, k, LEVELS[1], +1)
        at_lo = float(d[0]) if np.isclose(v[0], p.min) else np.nan
        at_hi = float(d[-1]) if np.isclose(v[-1], p.max) else np.nan
        mins = [j for j in _minima(v, d, k) if d[j] <= ALTERNATIVE]
        notes = []
        for j in mins:
            alternatives.append({"parameter": p.name, "value": float(v[j]), "dchi2": float(d[j]), "result": rows[j][1]})
            notes.append(f"second minimum at {v[j]:.4g}, dchi2 = {d[j]:.1f}")
        open_ends = [s for s, e in (("lower", e2l), ("upper", e2h)) if e]
        weak = [(s, x) for s, x in (("lower", at_lo), ("upper", at_hi)) if np.isfinite(x) and LEVELS[1] < x < WEAK]
        if k in (0, len(v) - 1) and (np.isclose(v[k], p.min) or np.isclose(v[k], p.max)):
            flag = "best at its limit"
        elif open_ends:
            flag = "not excluded up to the " + " and ".join(open_ends) + " limit" + ("s" if len(open_ends) > 1 else "")
        elif mins and min(d[j] for j in mins) <= LEVELS[1]:
            flag = "two solutions"
        elif mins:
            flag = "second minimum"
        elif weak:
            flag = "weakly bounded"
            notes += [f"dchi2 only {x:.1f} at the {s} limit" for s, x in weak]
        else:
            flag = "measured"
        cov_err = sig[i]
        if flag == "measured" and np.isfinite(cov_err) and cov_err > 0 and 0.5 * (h1 - l1) > 1.5 * cov_err:
            flag = "error bar too small"
            notes.append(f"68% interval half-width {0.5 * (h1 - l1):.3g}, error bar {cov_err:.3g}")
        prows.append({"parameter": p.name, "value": float(best.full[i]), "error": float(cov_err),
                      "lo68": l1, "hi68": h1, "lo95": l2, "hi95": h2,
                      "dchi2_at_min": at_lo, "dchi2_at_max": at_hi, "flag": flag, "note": "; ".join(notes)})

    # each result scanned directly: held at values around its best one while everything else is refitted
    errs = {}
    for qn, f in quantities.items():
        try:
            out = f(result)                  # the error bar of the original fit
            errs[qn] = float(out[1]) if isinstance(out, (tuple, list)) else np.nan
        except Exception:
            errs[qn] = np.nan
    src = best
    if best_fixed:                       # the better fit came from a scan: profile with that parameter free again
        import copy
        src = copy.copy(best)
        src.model = best.model.copy()
        src.model.params[best_fixed].fixed = False
    qprof = _run_quantity_profiles(src, quantities, errs, custom, n_jobs) if profile_quantities else {}
    qrows = []
    for j, qn in enumerate(quantities):
        err = errs[qn]
        row = {"quantity": qn, "value": q0[j], "error": err}
        iv = _profile_interval(qprof.get(qn) or [], chi0, scale)
        for lev, tag in zip(LEVELS, ("68", "95")):
            vals = [q[j] for dd, q in pooled if dd <= lev]
            if iv:
                vals += [iv["lo" + tag], iv["hi" + tag]]
            vals = np.array(vals, float)
            vals = vals[np.isfinite(vals)]
            row["lo" + tag], row["hi" + tag] = (float(vals.min()), float(vals.max())) if vals.size else (np.nan, np.nan)
        row["open"] = bool(iv.get("open95", False))
        row["fragile"] = False
        if quantities is not None and not custom and qn.startswith("broad ") and qn.endswith(" FWHM"):
            line = qn.split()[1]
            try:
                comps = [c.name for c in best._line_components() if c.role == "broad" and line in c.line_names]
                row["fragile"] = best.measure(line, comps)["fwhm_fragile"]
            except Exception:
                pass
        half = 0.5 * (row["hi68"] - row["lo68"])
        if row["open"]:
            row["flag"] = "not bounded"
        elif not np.isfinite(err) or err <= 0:
            row["flag"] = "no error bar"
        elif half > 1.5 * err:
            row["flag"] = "error bar too small"
        elif row["fragile"]:
            row["flag"] = "fragile FWHM"
        else:
            row["flag"] = "ok"
        qrows.append(row)
    profiles.update({"result: " + qn: pd.DataFrame(rows, columns=["target", "value", "chi2"]).assign(
        dchi2=lambda d: (d.chi2 - chi0) / scale) for qn, rows in qprof.items() if rows})
    pcols = ["parameter", "value", "error", "lo68", "hi68", "lo95", "hi95", "dchi2_at_min", "dchi2_at_max", "flag", "note"]
    qcols = ["quantity", "value", "error", "lo68", "hi68", "lo95", "hi95", "open", "fragile", "flag"]
    rob = Robustness(best, improved, pd.DataFrame(prows, columns=pcols), pd.DataFrame(qrows, columns=qcols), profiles,
                     alternatives, len(jobs), best_fixed)
    rob.result = result
    return rob
