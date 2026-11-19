"""
Fitting engine: data-driven initial guess, trust-region least squares with
exact Jacobians, and uncertainty estimation.
"""
import multiprocessing
import os
import threading
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from itertools import product

import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.optimize import least_squares, lsq_linear, nnls

from ._threads import SINGLE_THREADED, environment, n_threads
from .progress import live, live_enabled, report
from .result import FitResult

__all__ = ("fit", "guess", "fit_many")


class Problem:
    """Weighted least-squares problem in terms of the free parameters."""

    def __init__(self, model, x, y, err, ctx):
        self.model = model
        self.x, self.y, self.err, self.ctx = x, y, err, ctx
        self.T, self.c, self.free = model.params.affine()
        self.lb = np.array([p.min for p in self.free])
        self.ub = np.array([p.max for p in self.free])
        self.linear = np.array([p.linear for p in self.free], bool)
        # the amplitude columns of T, restricted to the rows that feed them
        TL = self.T[:, self.linear]
        self._rows_lin = np.nonzero(np.any(TL != 0, axis=1))[0]
        self._TL = TL[self._rows_lin]
        self._cache = {}

    def full(self, theta):
        return self.T @ theta + self.c

    def theta_from_model(self):
        return np.array([p.value for p in self.free], float)

    def evaluate(self, theta):
        """Model and Jacobian wrt theta, reusing unchanged component results."""
        full = self.full(theta)
        results = []
        for comp, s in zip(self.model.components, self.model._slices()):
            key = full[s].tobytes()
            hit = self._cache.get(comp.name)
            if hit is None or hit[0] != key:
                fc, Jc = comp.evaluate_with_derivatives(self.x, full[s], self.ctx)
                hit = (key, fc, Jc)
                self._cache[comp.name] = hit
            results.append((hit[1], hit[2]))
        f, Jfull = self.model.combine(self.x, results)
        return f, Jfull @ self.T

    def _basis_results(self, full):
        """(model, Jacobian with the amplitude columns) of every component, reusing unchanged ones."""
        results = []
        cache = self.__dict__.setdefault("_bcache", {})
        for comp, s in zip(self.model.components, self.model._slices()):
            key = full[s].tobytes()
            hit = self._cache.get(comp.name)            # a full evaluation is as good
            if hit is None or hit[0] != key:
                hit = cache.get(comp.name)
                if hit is None or hit[0] != key:
                    fc, Jc = comp.evaluate_basis(self.x, full[s], self.ctx)
                    hit = (key, fc, Jc)
                    cache[comp.name] = hit
            results.append((hit[1], hit[2]))
        return results

    def evaluate_basis(self, theta):
        """Model and amplitude columns of the Jacobian (cheaper; for linear solves)."""
        f, Jfull = self.model.combine(self.x, self._basis_results(self.full(theta)))
        return f, Jfull[:, self._rows_lin] @ self._TL       # amplitude columns only

    def scanner(self, theta, ks):
        """
        A function vals -> chi2 of the exact linear solution with the free parameters ``ks`` set to
        ``vals`` and the other nonlinear parameters as in ``theta``: what ``solve_linear`` gives for
        each trial of a grid search, at a fraction of the cost. Only the components that depend on
        ``ks`` are evaluated again; the rest of the design matrix and its normal matrix are computed
        once. A component of windowed Gaussian lines enters through its few non-zero elements and
        its own columns of the normal matrix. Returns None when a multiplicative component
        (extinction, absorption, calibration) depends on ``ks``.
        """
        model, x, L = self.model, self.x, self.linear
        comps, slices = model.components, model._slices()
        mult = model._multipliers()
        touched = np.any(self.T[:, ks] != 0, axis=1)
        changed = [i for i, s in enumerate(slices) if touched[s].any()]
        if not changed or any(i in mult for i in changed):
            return None
        results = self._basis_results(self.full(theta))
        static = [(np.zeros_like(f), np.zeros_like(J)) if i in changed else (f, J)
                  for i, (f, J) in enumerate(results)]
        f_s, J_s = model.combine(x, static)
        JL_s = J_s[:, self._rows_lin] @ self._TL
        inv = 1.0 / self.err
        A_s = JL_s * inv[:, None]
        r_s = (self.y - (f_s - JL_s @ theta[L])) * inv
        n = x.size
        dense, lines = [], None
        for i in changed:
            comp, s = comps[i], slices[i]
            w = inv.copy()                               # weight of the component: transmissions / error
            for m in mult:
                if model._applies(comps[m], comp):
                    w = w * results[m][0]
            lp = np.array([k for k, q in enumerate(comp.params) if q.linear], int)
            rows = s.start + lp
            part = (comp, s, w, lp, self.T[rows][:, L], self.c[rows])
            # one component of Gaussian lines (the usual case) is handled through its non-zero elements
            if lines is None and hasattr(comp, "basis_entries") and np.array_equal(lp, np.arange(lp.size)):
                lines = part
            else:
                dense.append(part)
        static_only = (A_s.T @ A_s, A_s.T @ r_s, float(r_s @ r_s))
        lb, ub = self.lb[L], self.ub[L]
        def chi2(vals):
            t = theta.copy()
            t[ks] = vals
            full = self.full(t)
            todo, hit = list(dense), None
            if lines is not None:
                comp, s, w, lp, TLc, const = lines
                p = full[s]
                M = comp._weights(p) @ TLc                # line -> free amplitude columns
                hit = comp.basis_entries(x, p, self.ctx) if np.count_nonzero(M, axis=1).max() <= 1 else None
                if hit is None:                          # wide lines, or a line that feeds two amplitudes
                    todo.append(lines)
            Cd = d = None
            for other, so, wo, lpo, TLo, _ in todo:
                fc, Jc = other.evaluate_basis(x, full[so], self.ctx)
                JLc = Jc[:, lpo] @ TLo
                Cd = wo[:, None] * JLc if Cd is None else Cd + wo[:, None] * JLc
                off = wo * (fc - JLc @ t[L])
                d = off if d is None else d + off
            if hit is not None:
                g, pix, line = hit
                g = g * w[pix]
                if const.any():                          # fixed amplitudes: part of the data side
                    off = np.bincount(pix, weights=g * (comp._weights(p) @ const)[line], minlength=n)
                    d = off if d is None else d + off
            if Cd is None and d is None:
                A, (G, b, rr) = A_s, static_only
            else:
                A = A_s if Cd is None else A_s + Cd
                rhs = r_s if d is None else r_s - d
                G = static_only[0] if Cd is None else A.T @ A
                b, rr = A.T @ rhs, float(rhs @ rhs)
            if hit is not None:
                col = np.argmax(M != 0, axis=1)
                wt = M[np.arange(M.shape[0]), col]
                cols = np.unique(col[wt != 0])
                if cols.size:
                    rank = np.zeros(M.shape[1], int)
                    rank[cols] = np.arange(cols.size)
                    C = np.bincount(pix * cols.size + rank[col[line]], weights=g * wt[line],
                                    minlength=n * cols.size).reshape(n, cols.size)
                    X = C.T @ A
                    G, b = G.copy(), b.copy()
                    G[cols] += X
                    G[:, cols] += X.T
                    G[np.ix_(cols, cols)] += C.T @ C
                    b[cols] += C.T @ (r_s if d is None else r_s - d)
            norm = np.sqrt(np.clip(np.diag(G), 0, None))
            norm[norm == 0] = 1.0
            Gn = G / np.outer(norm, norm)
            Gn[np.diag_indices_from(Gn)] += 1e-10
            Lc = np.linalg.cholesky(Gn)
            an = _bounded_lsq(Lc.T, solve_triangular(Lc, b / norm, lower=True), lb * norm, ub * norm)
            amps = an / norm
            return float(amps @ (G @ amps) - 2.0 * (amps @ b) + rr)

        return chi2

    def residuals(self, theta):
        f, _ = self.evaluate(theta)
        return (f - self.y) / self.err

    def jacobian(self, theta):
        _, J = self.evaluate(theta)
        return J / self.err[:, None]

    def solve_linear(self, theta):
        """Exact best amplitudes for the current nonlinear parameters."""
        f, JL = self.evaluate_basis(theta)
        L = self.linear
        A = JL / self.err[:, None]
        rhs = (self.y - (f - JL @ theta[L])) / self.err
        norm = np.sqrt((A ** 2).sum(axis=0))
        norm[norm == 0] = 1.0
        lb, ub = self.lb[L] * norm, self.ub[L] * norm
        # ||A a - b||^2 = ||R a - R^-T A^T b||^2 + const with R^T R = A^T A, so
        # the bounded problem is solved on the small k x k factor R instead of A
        # (columns are normalised, so the normal equations are well conditioned).
        An = A / norm
        G = An.T @ An
        G[np.diag_indices_from(G)] += 1e-10
        Lc = np.linalg.cholesky(G)
        amps = _bounded_lsq(Lc.T, solve_triangular(Lc, An.T @ rhs, lower=True), lb, ub) / norm
        chi2 = float(np.sum((A @ amps - rhs) ** 2))
        out = theta.copy()
        out[L] = amps
        return chi2, out


class _Live:
    """
    Snapshots of a running fit for a live display (progress.set_live_hook): the data once, then at most
    about eight times a second the current model (rebuilt from the residuals, model = y + r err), chi2 and
    the phase. Only while a live hook is installed; costs nothing otherwise.
    """
    PHASES = ("guess", "fit", "build-up", "errors")

    def __init__(self, x, y, err, npar):
        self.x, self.y, self.err = x, y, err
        self.idx = slice(None, None, max(1, x.size // 900))
        self.t0 = self.last = time.time()
        self.n, self.best, self.phase = 0, np.inf, "guess"
        self.dof = max(x.size - npar, 1)
        live({"event": "start", "x": x[self.idx], "y": y[self.idx], "err": err[self.idx], "dof": self.dof,
              "phases": self.PHASES, "npar": int(npar)})

    def set_phase(self, phase, detail=""):
        self.phase = phase
        live({"event": "phase", "phase": phase, "detail": detail, "t": time.time() - self.t0, "n": self.n})

    def tick(self, r=None, chi2=None, model=None, force=False):
        """One evaluation: the residual vector r, or chi2 and a function that gives the model (force: send
        it however soon after the last one, e.g. the converged model)."""
        self.n += 1
        if r is not None:
            if r.size != self.y.size:
                return
            chi2 = float(r @ r)
        if not np.isfinite(chi2):
            return
        self.best = min(self.best, chi2)
        now = time.time()
        if now - self.last < 0.12 and not force:
            return
        self.last = now
        f = self.y + r * self.err if r is not None else model() if model is not None else None
        live({"event": "tick", "phase": self.phase, "n": self.n, "t": now - self.t0, "chi2": chi2,
              "best": self.best, "model": None if f is None else f[self.idx]})


_LIVE = threading.local()          # the _Live of the fit running in this thread (fits in other threads have their own)


def _live_now():
    return getattr(_LIVE, "current", None)


def _tick(**kw):
    lv = _live_now()
    if lv is not None:
        lv.tick(**kw)


def _phase(name, detail=""):
    lv = _live_now()
    if lv is not None:
        lv.set_phase(name, detail)


def _bounded_lsq(A, b, lb, ub):
    """
    min ||A x - b|| with lb <= x <= ub. One-sided bounds (the usual case:
    amplitudes >= 0, absorption <= 0) go to the fast Lawson-Hanson NNLS after
    shifting/flipping the variables; two-sided or no bounds use BVLS.
    """
    lo_ok, hi_inf = np.isfinite(lb), ~np.isfinite(ub)
    hi_ok, lo_inf = np.isfinite(ub), ~np.isfinite(lb)
    up = lo_ok & hi_inf                       # x >= lb
    down = hi_ok & lo_inf                     # x <= ub
    if np.all(up | down):
        sign = np.where(up, 1.0, -1.0)
        shift = np.where(up, lb, ub)
        As = A * sign
        try:
            y, _ = nnls(As, b - A @ shift, maxiter=50 * A.shape[1])
            return shift + sign * y
        except RuntimeError:                  # no convergence: fall back
            pass
    return lsq_linear(A, b, bounds=(lb, ub), method="bvls").x


_POOL = {}


def _scan(prob, trials):
    """
    chi2 of the exact linear solution for each trial parameter vector. The
    trials are independent, so they run in threads (the work is in NumPy,
    SciPy and FFT code, which release the GIL).
    """
    return _thread_map(lambda t: prob.solve_linear(t)[0], trials)


def _scan_group(prob, theta, ks, combos):
    """chi2 of each trial ``theta[ks] = vals`` (amplitudes solved exactly), by the fast scanner when it applies."""
    scan = prob.scanner(theta, ks) if os.environ.get("FANTASY_SCANNER", "1") != "0" else None
    if scan is not None:
        return _thread_map(scan, combos)
    trials = []
    for vals in combos:
        t = theta.copy()
        t[ks] = vals
        trials.append(t)
    return _scan(prob, trials)


def _thread_map(fn, items):
    """[fn(i) for i in items], in the shared thread pool when worthwhile."""
    n = n_threads()
    items = list(items)
    if n == 1 or len(items) < 4:
        return [fn(i) for i in items]
    pool = _POOL.get(n)
    if pool is None:
        pool = _POOL[n] = ThreadPoolExecutor(n, thread_name_prefix="fantasy")
    return list(pool.map(fn, items))


# ------------------------------------------------------------- initial guess
def _neutral(p):
    lo, hi = p.min, p.max
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return p.value
    if p.name.endswith("fwhm") and lo > 0:
        return float(np.sqrt(lo * hi))
    if lo <= 0 <= hi:
        return 0.0
    return 0.5 * (lo + hi)


def _grid(p, n):
    if p.name.endswith("fwhm") and p.min > 0:
        return np.geomspace(p.min, p.max, n)
    return np.linspace(p.min, p.max, n)


def _refine_grid(p, best, step, n):
    if p.name.endswith("fwhm") and p.min > 0:
        g = best * np.exp(np.linspace(-1, 1, n) * step)
    else:
        g = best + np.linspace(-1, 1, n) * step
    return np.unique(np.clip(g, p.min, p.max))


_KINEMATIC = ("shift", "fwhm", "velocity", "sigma")


def _groups(prob):
    """
    Nonlinear parameters to grid-search together. Velocity and width of a
    component are searched jointly; further shape parameters (h3, h4, the
    Lorentzian width, E(B-V), ...) one at a time, so the grid size does not
    grow exponentially with the number of parameters.
    """
    groups = {}
    for k, p in enumerate(prob.free):
        if p.linear or not (np.isfinite(p.min) and np.isfinite(p.max)):
            continue
        owner, local = p.name.rsplit(".", 1)
        key = owner if local in _KINEMATIC else p.name
        groups.setdefault(key, []).append(k)
    return groups


def _initial_guess(prob, ngrid=9, nrefine=5, npass=3, verbose=False):
    theta = prob.theta_from_model()
    groups = _groups(prob)
    for ks in groups.values():
        for k in ks:
            theta[k] = _neutral(prob.free[k])

    chi2, theta = prob.solve_linear(theta)
    _tick(chi2=chi2, model=lambda t=theta.copy(): prob.evaluate(t)[0], force=True)   # the starting point
    if verbose:
        print(f"  guess: start chi2 = {chi2:.1f}")

    # Search components carrying most flux first.
    f, J = prob.evaluate(theta)
    weight = {}
    for name, ks in groups.items():
        comp = name.split(".")[0]
        cols = [k for k, p in enumerate(prob.free) if p.linear and p.name.startswith(comp + ".")]
        weight[name] = float(np.abs(J[:, cols] @ theta[cols]).sum()) if cols else 0.0
    order = sorted(groups, key=lambda n: -weight[n])

    steps = {}
    pass_start = chi2
    for ipass in range(npass):
        for name in order:
            ks = groups[name]
            params = [prob.free[k] for k in ks]
            if ipass == 0:
                grids = [_grid(p, ngrid) for p in params]
                steps[name] = [np.log(g[1] / g[0]) if p.name.endswith("fwhm") and p.min > 0
                               else g[1] - g[0] for p, g in zip(params, grids)]
            else:
                grids = [_refine_grid(p, theta[k], s * 0.5 ** (ipass - 1), nrefine)
                         for p, k, s in zip(params, ks, steps[name])]
            best_chi2, best_vals = chi2, theta[ks].copy()
            combos = [np.array(v) for v in product(*grids)]
            for vals, c2 in zip(combos, _scan_group(prob, theta, ks, combos)):
                if c2 < best_chi2:
                    best_chi2, best_vals = c2, vals
            theta[ks] = best_vals
            chi2, theta = prob.solve_linear(theta)
            _tick(chi2=chi2, model=lambda t=theta.copy(): prob.evaluate(t)[0])
        if verbose:
            print(f"  guess: pass {ipass + 1} chi2 = {chi2:.1f}")
        if ipass >= 1 and chi2 > pass_start * (1 - 1e-3):
            break                          # a refinement pass that hardly helps: done
        pass_start = chi2
    return theta, chi2


def _escape_local_minimum(prob, theta, chi2, ngrid=9, only=None):
    """
    After a fit, re-scan each component's nonlinear parameters on a coarse grid
    (amplitudes solved exactly, everything else at the fitted values). Return
    a better starting point if one is found, else None. ``only``: scan just
    the groups of this component.
    """
    best = (chi2, None)
    groups = _groups(prob)
    if only is not None:
        groups = {k: v for k, v in groups.items() if k == only or k.startswith(only + ".")}
    for ks in groups.values():
        combos = [np.array(v) for v in product(*[_grid(prob.free[k], ngrid) for k in ks])]
        chis = _scan_group(prob, theta, ks, combos)
        k = int(np.argmin(chis))
        if chis[k] < best[0]:
            t = theta.copy()
            t[ks] = combos[k]
            best = (chis[k], prob.solve_linear(t)[1])
    if best[1] is not None and best[0] < chi2 * 0.99 - 1.0:
        return best[1]
    return None


# ------------------------------------------------------- staged (build-up) fit
def _extra_components(model):
    """
    Components that overlap others and are therefore hard to place from scratch:
    outflow wings, and broad components beyond the narrowest one of each group
    of broad components fitting the same lines. Returned in the order in which
    they are added (outflows first, then broad components from narrow to wide).
    """
    from .components import Lines
    lines = [c for c in model.components if isinstance(c, Lines)]
    outflows = [c for c in lines if c.role == "outflow"]
    families = {}
    for c in lines:
        if c.role == "broad":
            families.setdefault(tuple(c.line_names), []).append(c)
    broad = []
    for members in families.values():
        members = sorted(members, key=lambda c: model.params[f"{c.name}.fwhm"].min
                         if f"{c.name}.fwhm" in model.params else 0.0)
        broad += members[1:]
    broad.sort(key=lambda c: model.params[f"{c.name}.fwhm"].min
               if f"{c.name}.fwhm" in model.params else 0.0)
    return outflows + broad


def _fit_stage(model, x, y, err, ctx, values, new_component=None, full_escape=True):
    """
    Fit ``model`` (whose inactive components are fixed) starting from
    ``values``. A newly activated component is first placed by a grid search
    over its kinematics with the rest of the model held at ``values``.
    Returns the fitted values of all parameters and chi2.
    """
    prob = Problem(model, x, y, err, ctx)
    if not prob.free:
        r = (model.evaluate(x, ctx) - y) / err
        return {p.name: p.value for p in model.params}, float(r @ r)
    theta = prob.theta_from_model()
    if values is None:
        theta, _ = _initial_guess(prob)
    else:
        chi2, theta = prob.solve_linear(theta)
        if new_component is not None:
            own = {k: ks for k, ks in _groups(prob).items()
                   if k == new_component or k.startswith(new_component + ".")}
            for ks in own.values():
                combos = [np.array(v) for v in product(*[_grid(prob.free[k], 9) for k in ks])]
                best = (chi2, theta[ks].copy())
                for vals, c2 in zip(combos, _scan_group(prob, theta, ks, combos)):
                    if c2 < best[0]:
                        best = (c2, vals)
                theta[ks] = best[1]
                chi2, theta = prob.solve_linear(theta)
    res = _least_squares(prob, theta)
    only = None if full_escape else new_component
    for _ in range(3):
        start = _escape_local_minimum(prob, res.x, float(res.fun @ res.fun), only=only)
        if start is None:
            break
        new = _least_squares(prob, start)
        if new.fun @ new.fun < res.fun @ res.fun:
            res = new
    full = prob.full(res.x)
    return dict(zip(model.params.names, full)), float(res.fun @ res.fun)


def _staged_fit(model, x, y, err, ctx):
    """
    Build the model up one overlapping component at a time: fit without the
    extra components, then add each in turn (placed by a grid search and
    refitted with everything else). Returns fitted values by parameter name,
    or None if the model has no such components.
    """
    extras = _extra_components(model)
    if not extras:
        return None
    original = {p.name: (p.value, p.fixed) for p in model.params}
    stage = model.copy()
    for c in extras:
        for p in stage.params.match(f"{c.name}.*"):
            if p.free:
                p.fixed = True
                p.value = (min(max(0.0, p.min), p.max) if p.linear
                           else _neutral(p))
    values, _ = _fit_stage(stage, x, y, err, ctx, None)
    for c in extras:
        for p in stage.params:
            if p.name in values:
                p.value = values[p.name] if p.free or p.tie else p.value
        for p in stage.params.match(f"{c.name}.*"):
            p.fixed = original[p.name][1]
        values, _ = _fit_stage(stage, x, y, err, ctx, values, new_component=c.name,
                               full_escape=c is extras[-1])   # full re-scan at the end
        for p in stage.params:
            p.value = values[p.name]
    return values


class _Helper:
    """
    One persistent helper process (``python -m fantasy_agn._worker``) that runs the staged search
    of a fit while the fitting process runs the direct one. Used by one fit at a time; a fit that
    finds it busy, a worker process of a parallel run (FANTASY_THREADS=1) and FANTASY_HELPER=0 run
    both searches in turn as before. Any failure ends the helper and the search runs here.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.failures = 0

    def available(self):
        return (os.environ.get("FANTASY_HELPER", "1") != "0" and n_threads() > 1 and self.failures < 3
                and (os.cpu_count() or 1) > 1)

    def _start(self):
        import subprocess
        import sys
        if self.proc is None or self.proc.poll() is not None:
            env = dict(os.environ, **SINGLE_THREADED)
            # the helper imports this copy of the package, also when it was found through sys.path
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            env["PYTHONPATH"] = os.pathsep.join([root] + [p for p in [env.get("PYTHONPATH")] if p])
            self.proc = subprocess.Popen([sys.executable, "-m", "fantasy_agn._worker"], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
        return self.proc

    def stop(self):
        proc, self.proc = self.proc, None
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass

    def submit(self, name, *args):
        """Send the call; returns a function that waits for the result (None: run it yourself)."""
        import pickle
        import struct
        if not self.available() or not self.lock.acquire(blocking=False):
            return None
        try:
            proc = self._start()
            data = pickle.dumps((name, args), protocol=pickle.HIGHEST_PROTOCOL)
            proc.stdin.write(struct.pack("<Q", len(data)) + data)
            proc.stdin.flush()
        except Exception:
            self.failures += 1
            self.stop()
            self.lock.release()
            return None

        def result():
            try:
                head = proc.stdout.read(8)
                body = proc.stdout.read(struct.unpack("<Q", head)[0]) if len(head) == 8 else b""
                ok, value = pickle.loads(body)
                if not ok:
                    raise RuntimeError(value)
                return value
            except Exception:
                self.failures += 1
                self.stop()
                raise
            finally:
                self.lock.release()

        def cancel():                            # the fit ended without the result: the helper is not reused
            self.stop()
            self.lock.release()

        result.cancel = cancel
        return result


_HELPER = _Helper()


# ------------------------------------------------------------------- fitting
def _prepare(spectrum, wave_range):
    good = spectrum.good.copy()
    if wave_range is not None:
        good &= (spectrum.wave >= wave_range[0]) & (spectrum.wave <= wave_range[1])
    if good.sum() < 10:
        raise ValueError("fewer than 10 usable pixels in the fitted range")
    return good


def _interior(theta, lb, ub, linear):
    """
    Move parameters sitting on a bound slightly inside. Starting exactly on a
    bound makes the trust-region steps vanish and the fit stop immediately.
    """
    amps = np.abs(theta[linear])
    amps = amps[amps > 0]
    typical = float(np.median(amps)) if amps.size else 1.0
    span = np.where(np.isfinite(ub - lb), ub - lb,
                    np.where(linear, typical, np.abs(theta) + 1.0))
    eps = 1e-3 * span
    return np.clip(theta, lb + eps, ub - eps)


def _least_squares(prob, theta0, max_nfev=None, tol=1e-10):
    if prob.linear.any() and (~prob.linear).any() and os.environ.get("FANTASY_VARPRO", "1") != "0":
        return _projected_least_squares(prob, theta0, max_nfev, tol)
    def resid(theta):
        r = prob.residuals(theta)
        _tick(r=r)
        return r

    return least_squares(resid if _live_now() is not None else prob.residuals,
                         _interior(theta0, prob.lb, prob.ub, prob.linear),
                         jac=prob.jacobian, bounds=(prob.lb, prob.ub), method="trf",
                         x_scale="jac", ftol=tol, xtol=tol, gtol=tol,
                         max_nfev=max_nfev)


class _Solution:
    """What the fit needs from a least_squares result."""

    def __init__(self, x, fun, jac, nfev, success, message):
        self.x, self.fun, self.jac = x, fun, jac
        self.nfev, self.success, self.message = nfev, success, message


def _column_space(A):
    """
    Orthonormal basis of the columns of A. Plain QR gives repeated columns
    arbitrary directions of their own, and projecting those out can hide the
    downhill direction, so the fit stops early and reports success.
    """
    norm = np.sqrt((A ** 2).sum(axis=0))
    norm[norm == 0] = 1.0
    Q, R, _ = qr(A / norm, mode="economic", pivoting=True)
    d = np.abs(np.diag(R))
    return Q[:, d > d[0] * 1e-12]


def _projected_least_squares(prob, theta0, max_nfev=None, tol=1e-10):
    """
    Variable projection: the trust-region iterations run over the nonlinear
    parameters only (shifts, widths, slopes, E(B-V)); at every step the
    amplitudes are the exact bounded linear least-squares solution. With
    tens of template and line amplitudes this needs far fewer iterations
    than moving all parameters together, and each iteration is cheaper.
    The Jacobian of the projected residuals is Kaufman's approximation:
    the nonlinear columns projected onto the complement of the amplitude
    columns that are not on a bound.
    """
    L = prob.linear
    N = ~L
    base = theta0.copy()
    memo = {}

    def solve(z):
        key = z.tobytes()
        if key not in memo:
            t = base.copy()
            t[N] = z
            _, t = prob.solve_linear(t)
            f, J = prob.evaluate(t)
            r = (f - prob.y) / prob.err
            _tick(r=r)
            J = J / prob.err[:, None]
            span = np.where(np.isfinite(prob.ub - prob.lb), prob.ub - prob.lb, 0.0)
            tiny = 1e-10 * np.maximum(np.abs(t), span)
            free = L & (t - prob.lb > tiny) & (prob.ub - t > tiny)
            Jn = J[:, N]
            if free.any():
                Q = _column_space(J[:, free])
                Jn = Jn - Q @ (Q.T @ Jn)
            memo.clear()
            memo[key] = (t, r, Jn, J)
        return memo[key]

    z0 = _interior(theta0, prob.lb, prob.ub, L)[N]
    res = least_squares(lambda z: solve(z)[1], z0, jac=lambda z: solve(z)[2],
                        bounds=(prob.lb[N], prob.ub[N]), method="trf", x_scale="jac",
                        ftol=tol, xtol=tol, gtol=tol, max_nfev=max_nfev)
    t, r, _, J = solve(res.x)
    return _Solution(t, r, J, res.nfev, res.success, res.message)


def guess(spectrum, model, wave_range=None, verbose=False):
    """Return a copy of ``model`` with data-driven initial parameter values."""
    model = model.copy()
    good = _prepare(spectrum, wave_range or model.wave_range)
    args = (spectrum.wave[good], spectrum.flux[good], spectrum.err[good], spectrum.context)
    work = model.copy()
    # amplitudes without fitted pixels (lines in masked bands or outside the range)
    # are 0, as in fit(); they stay free in the returned model
    weak = _fix_unconstrained(Problem(work, *args), work)
    prob = Problem(work, *args)
    theta, _ = _initial_guess(prob, verbose=verbose)
    model.params.set_full_values(prob.full(theta))
    for name in weak:
        model.params[name].value = work.params[name].value
    return model


def fit(spectrum, model, wave_range=None, guess=True, errors="covariance", n_mc=100,
        scale_errors=True, seed=None, n_jobs=None, mcmc=None, build_up=True, verbose=True):
    """
    Fit a model to a spectrum.

    :param spectrum: a Spectrum (usually in the rest frame).
    :param model: a Model; it is not modified (the result holds a fitted copy).
    :param wave_range: (min, max) wavelengths to fit; default: whole spectrum.
    :param guess: estimate initial parameters from the data first (recommended).
    :param errors: 'covariance' (fast, from the Jacobian), 'mc' (Monte Carlo
        re-fits of the spectrum perturbed by its errors), 'mcmc' (posterior
        samples by Hamiltonian Monte Carlo with flat priors within the
        parameter limits) or None.
    :param mcmc: options for errors='mcmc': n_chains (4), n_warmup (300),
        n_samples per chain (600), trajectory (1.6, whitened units), retry (True:
        chains that have not converged are run again with a 3x longer warm-up and
        a 2x longer trajectory). ``result.mcmc_info`` reports acceptance,
        split-R-hat, the effective sample size and the number of attempts.
    :param n_mc: number of Monte Carlo realizations.
    :param scale_errors: if the reduced chi2 > 1, inflate uncertainties by
        sqrt(reduced chi2) (the error spectrum is then assumed underestimated).
    :param seed: random seed for the Monte Carlo errors.
    :param n_jobs: processes for Monte Carlo and for the MCMC chains (default: all cores, or 1 inside the
        workers of fit_many / fit_sample; 1 = serial). In a script, parallel runs
        need the code under  if __name__ == "__main__":  (else it runs serially).
    :param build_up: for models with overlapping components (several broad
        components of the same lines, outflow wings), also fit by adding them
        one at a time and keep the better of the two solutions
        (``result.start`` says which was used).
    :param verbose: print a short summary.
    :return: FitResult
    """
    t0 = time.time()
    model = model.copy()
    learned = False
    if guess is not True and guess is not False and guess is not None:
        # guess="ml" (or an Initializer): a machine-learned start (fantasy_agn.ml). Only the
        # non-linear parameters are predicted; the amplitudes are solved exactly and the fit
        # itself is the usual one.
        from . import ml
        if guess != "ml" and not isinstance(guess, ml.Initializer):
            raise ValueError("guess must be True, False, 'ml' or a fantasy_agn.ml.Initializer")
        learned = ml.apply(model, spectrum, None if guess == "ml" else guess)
        if not learned:
            warnings.warn("no machine-learned initializer for this model: using the data-driven guess")
        guess = not learned
    if wave_range is None:
        wave_range = model.wave_range
    good = _prepare(spectrum, wave_range)
    x, y, err = spectrum.wave[good], spectrum.flux[good], spectrum.err[good]
    prob = Problem(model, x, y, err, spectrum.context)
    unconstrained = _fix_unconstrained(prob, model)
    if unconstrained:
        prob = Problem(model, x, y, err, spectrum.context)

    theta = prob.theta_from_model()
    given = theta.copy()
    outer = _live_now()
    _LIVE.current = _Live(x, y, err, len(theta)) if live_enabled() else None
    try:
        return _fit_body(spectrum, model, prob, good, x, y, err, theta, given, guess, learned, build_up, unconstrained,
                         errors, n_mc, scale_errors, seed, n_jobs, mcmc, verbose, t0)
    finally:
        _LIVE.current = outer


def _fit_body(spectrum, model, prob, good, x, y, err, theta, given, guess, learned, build_up, unconstrained,
              errors, n_mc, scale_errors, seed, n_jobs, mcmc, verbose, t0):
    """fit() after the problem is set up (see there)."""
    # Overlapping components (extra broad components, outflow wings) are easily trapped in a local
    # minimum when all are placed at once; they are also built up one at a time (the staged search)
    # and the better solution is kept. The staged search is independent of the direct one: it runs in
    # the helper process meanwhile, when there is one to use.
    staged_job = None
    if guess and build_up and _extra_components(model):
        _phase("guess", "starting the build-up in a helper process")
        staged_job = _HELPER.submit("_staged_fit", model.copy(), x, y, err, spectrum.context)
    try:
        res, start_method = _direct_search(prob, model, theta, given, guess, learned)
        if guess and build_up:
            values = None
            if staged_job is not None:
                _phase("build-up", "components added one at a time (in a helper process)")
                job, staged_job = staged_job, None
                try:
                    values = job()
                except Exception:
                    values = _staged_fit(model, x, y, err, spectrum.context)
            else:
                values = _staged_fit(model, x, y, err, spectrum.context)
            if values is not None:
                _phase("build-up", "components added one at a time")
                staged = np.array([values[p.name] for p in prob.free])
                new = _least_squares(prob, staged)
                if new.fun @ new.fun < res.fun @ res.fun * (1 - 1e-9):
                    new.nfev += res.nfev
                    res = new
                    start_method = "staged"
    finally:
        if staged_job is not None:
            staged_job.cancel()
    theta = res.x.copy()
    _tick(r=res.fun, force=True)                     # the converged model
    # An amplitude kept because a wide profile would reach the fitted pixels can end with no effect on them at
    # the fitted width (a narrow line inside a masked band): nothing constrains it, so it is set to 0.
    idle = _idle_amplitudes(prob, theta)
    if idle.size:
        theta[idle] = np.clip(0.0, prob.lb[idle], prob.ub[idle])
        unconstrained = list(unconstrained) + [prob.free[k].name for k in idle]
    at_limit = _at_limit(theta, prob.lb, prob.ub)
    # Amplitudes that end on a bound (undetected components) are set exactly to it.
    snap = at_limit & prob.linear
    theta[snap] = np.where(np.abs(theta - prob.lb) < np.abs(prob.ub - theta), prob.lb, prob.ub)[snap]
    full = prob.full(theta)
    model.params.set_full_values(full)
    model._fitted = True                   # its values are a fit: a later fit of this model also starts from them
    # The widths and shifts of a component without flux have no effect on the
    # model: any value fits equally well, so they are not measurements.
    dead = ~prob.linear & ~prob.evaluate(theta)[1].any(axis=0)
    only_dead = np.abs(prob.T[:, dead]).sum(axis=1) > 0
    only_dead &= np.abs(prob.T[:, ~dead]).sum(axis=1) == 0
    not_measured = [n for n, d in zip(model.params.names, only_dead) if d]
    unused = at_limit | dead

    resid = res.fun
    chi2 = float(resid @ resid)
    dof = max(x.size - len(theta), 1)
    rchi2 = chi2 / dof
    scale = max(rchi2, 1.0) if scale_errors else 1.0

    cov_full = None
    samples = None
    if errors:
        _phase("errors", {"covariance": "from the Jacobian", "mc": f"{n_mc} Monte Carlo fits",
                          "mcmc": "MCMC sampling"}.get(errors, ""))
    if errors in ("covariance", "mc"):
        cov = _covariance(res.jac, unused) * scale
        cov_full = prob.T @ cov @ prob.T.T
    mcmc_info = None
    if errors == "mc":
        samples = _monte_carlo(prob, theta, n_mc, np.sqrt(scale), seed, n_jobs)
    elif errors == "mcmc":
        cov = _covariance(res.jac, unused) * scale
        cov_full = prob.T @ cov @ prob.T.T
        samples, mcmc_info = _mcmc(prob, theta, scale, seed, n_jobs, **(mcmc or {}))
    elif errors not in (None, "covariance"):
        raise ValueError("errors must be 'covariance', 'mc', 'mcmc' or None")

    result = FitResult(
        spectrum=spectrum, model=model, mask=good, full=full, cov=cov_full,
        samples=samples, chi2=chi2, dof=dof, nfev=res.nfev, success=res.success,
        message=res.message,
        at_limit=[p.name for p, a, d in zip(prob.free, at_limit, dead) if a and not d and not p.linear],
        at_zero=[p.name for p, a in zip(prob.free, at_limit) if a and p.linear],
        unconstrained=unconstrained, not_measured=not_measured,
        error_scale=np.sqrt(scale), time=time.time() - t0, error_method=errors)
    result.mcmc_info = mcmc_info
    result.start = start_method
    if verbose:
        print(result.summary(short=True))
    return result



def _direct_search(prob, model, theta, given, guess, learned):
    """The fit from the data-driven (or learned) guess, with the escapes from local minima."""
    if guess:
        _phase("guess", "a grid search for the widths and shifts")
        theta, _ = _initial_guess(prob, verbose=False)
    _phase("fit", "trust-region least squares")
    res = _least_squares(prob, theta)
    if guess and getattr(model, "_fitted", False):
        # the values come from an earlier fit (r.model, a robustness alternative): try them as a start too,
        # so that a refit never ends worse than the solution it was given
        new = _least_squares(prob, _interior(given, prob.lb, prob.ub, prob.linear))
        if new.fun @ new.fun < res.fun @ res.fun:
            new.nfev += res.nfev
            res = new
    start_method = "learned" if learned else "direct"
    if learned:
        # one cheap check for a better nearby solution instead of the full search
        start = _escape_local_minimum(prob, res.x, float(res.fun @ res.fun))
        if start is not None:
            new = _least_squares(prob, start)
            if new.fun @ new.fun < res.fun @ res.fun:
                new.nfev += res.nfev
                res = new
    if guess:
        for _ in range(3):
            _phase("fit", "looking for a better minimum nearby")
            start = _escape_local_minimum(prob, res.x, float(res.fun @ res.fun))
            if start is None:
                break
            new = _least_squares(prob, start)
            if new.fun @ new.fun < res.fun @ res.fun:
                new.nfev += res.nfev
                res = new
    return res, start_method


def _fix_unconstrained(prob, model):
    """
    Fix amplitudes that have (almost) no effect on the fitted pixels, e.g. FeII
    multiplets or lines outside the wavelength window, at 0. They would
    otherwise be unconstrained and make the problem degenerate.
    """
    theta = prob.theta_from_model()
    for k, p in enumerate(prob.free):
        if not p.linear:
            theta[k] = _neutral(p)
    lin = np.nonzero(prob.linear)[0]
    if lin.size == 0:
        return []
    # Effect of each amplitude on chi2 per unit of its own typical size, at its
    # best: a broad line with a masked centre reaches the pixels only when wide.
    power = np.zeros(lin.size)
    for t in (theta, _widest(prob, theta, "min"), _widest(prob, theta, "max")):
        _, J = prob.evaluate(t)
        power = np.maximum(power, np.sqrt(((J[:, lin] / prob.err[:, None]) ** 2).sum(axis=0)))
    ref = np.median(power[power > 0]) if (power > 0).any() else 0.0
    weak = [prob.free[k].name for k, pw in zip(lin, power) if pw <= 1e-6 * ref]
    for name in weak:
        lo, hi = model.params[name].min, model.params[name].max
        model.params.fix(name, min(max(0.0, lo), hi))
    # A component whose amplitudes are all fixed at 0 has no effect, so its
    # widths and shifts are unconstrained too.
    for comp in model.components:
        amps = [p for p in comp.params if p.linear]
        if amps and all(p.fixed and p.value == 0 for p in amps):
            for p in comp.params:
                if p.free:
                    p.fixed = True
                    weak.append(p.name)
    return weak


def _idle_amplitudes(prob, theta):
    """Indices of the free amplitudes that do not change the model on the fitted pixels at ``theta``."""
    lin = np.nonzero(prob.linear)[0]
    if lin.size == 0:
        return lin
    _, J = prob.evaluate(theta)
    power = np.sqrt(((J[:, lin] / prob.err[:, None]) ** 2).sum(axis=0))
    ref = np.median(power[power > 0]) if (power > 0).any() else 0.0
    # numerically no effect only (a profile tens of sigma from every fitted pixel); far wings that still touch
    # the pixels keep their amplitude, which is then reported as not measured
    return lin[power <= 1e-12 * ref]


def _widest(prob, theta, side):
    """``theta`` with the widths at their maximum and the shifts at one limit."""
    out = theta.copy()
    for k, p in enumerate(prob.free):
        last = p.name.rsplit(".", 1)[-1]
        limit = p.max if last.startswith("fwhm") else getattr(p, side) if last == "shift" else np.inf
        if np.isfinite(limit):
            out[k] = limit
    return out


def _at_limit(theta, lb, ub):
    span = np.where(np.isfinite(ub - lb), ub - lb, np.inf)
    tol = 1e-6 * np.where(np.isfinite(span), span, np.abs(theta) + 1)
    return (np.abs(theta - lb) <= tol) | (np.abs(ub - theta) <= tol)


def _covariance(J, exclude):
    """(J^T J)^-1 for the parameters not at a limit, via SVD."""
    n = J.shape[1]
    cov = np.zeros((n, n))
    keep = ~exclude
    if not keep.any():
        return cov
    Jk = J[:, keep]
    norm = np.sqrt((Jk ** 2).sum(axis=0))
    norm[norm == 0] = 1.0
    _, s, Vt = np.linalg.svd(Jk / norm, full_matrices=False)
    good = s > s[0] * 1e-12
    if not good.all():
        warnings.warn("some parameters are degenerate; their errors are unreliable")
    inv = (Vt[good].T / s[good] ** 2) @ Vt[good]
    cov[np.ix_(keep, keep)] = inv / np.outer(norm, norm)
    return cov


# ---------------------------------------------------------------------- MCMC
class _Posterior:
    """log posterior and gradient from one model evaluation; picklable for parallel chains."""

    def __init__(self, prob, scale):
        self.prob, self.scale = prob, scale

    def __call__(self, t):
        f, J = self.prob.evaluate(t)
        r = (f - self.prob.y) / self.prob.err
        return -0.5 * float(r @ r) / self.scale, -((J / self.prob.err[:, None]).T @ r) / self.scale


def _sampler_cov(prob, theta, scale):
    """
    Covariance that whitens every free parameter: the fit covariance, with the
    parameter range for those it says nothing about (the width of a component
    without flux), and no wider than the range.
    """
    J = prob.jacobian(theta)
    cov = _covariance(J, ~J.any(axis=0)) * scale
    span = prob.ub - prob.lb
    col = np.sqrt((J ** 2).sum(axis=0))
    rough = np.where(np.isfinite(span), span / 4, np.where(col > 0, np.sqrt(scale) / np.where(col > 0, col, 1), np.abs(theta) + 1))
    missing = np.diag(cov) <= 0
    cov[np.diag_indices_from(cov)] = np.where(missing, rough ** 2, np.diag(cov))
    sd = np.sqrt(np.diag(cov))
    shrink = np.minimum(1.0, np.where(np.isfinite(span), span / sd, np.inf))
    return cov * np.outer(shrink, shrink)


def _mcmc(prob, theta, scale, seed, n_jobs=None, n_chains=4, n_warmup=300, n_samples=600, trajectory=None,
          retry=True, max_leap_warmup=None):
    """
    Posterior samples by Hamiltonian Monte Carlo (see fantasy_agn.sampling),
    preconditioned with the fit covariance. All free parameters are sampled,
    also those that ended at a limit: their posterior is one-sided, which is
    what an upper limit on an undetected line comes from.

    If the chains have not converged (R-hat >= 1.05 or ESS <= 100) they are run once more with a 3x longer
    warm-up and twice the trajectory length (``retry=False`` to skip): on hard blends of SDSS quasars the
    default settings converged in 6 of 14 runs and the longer ones in 6 of 7, while easy posteriors keep the
    cheaper default. ``info["attempts"]`` says how many runs were needed.
    """
    from .sampling import hmc, DEFAULT_TRAJECTORY
    if n_jobs is None:                     # all cores, but not inside a worker process
        n_jobs = 1 if multiprocessing.parent_process() is not None else -1
    n_jobs = (os.cpu_count() or 1) if n_jobs == -1 else max(int(n_jobs), 1)
    prob._cache = {}
    start = _interior(theta, prob.lb, prob.ub, prob.linear)
    cov = _sampler_cov(prob, start, scale)
    traj = DEFAULT_TRAJECTORY if trajectory is None else float(trajectory)
    plans = [(n_warmup, traj)] + ([(3 * n_warmup, 2 * traj)] if retry and trajectory is None else [])
    for attempt, (warm, tr) in enumerate(plans, 1):
        draws, info = hmc(_Posterior(prob, scale), start, cov, prob.lb, prob.ub, n_chains=n_chains, n_warmup=warm,
                          n_samples=n_samples, seed=seed, n_jobs=n_jobs, trajectory=tr, max_leap_warmup=max_leap_warmup)
        info.update(attempts=attempt, trajectory=tr)
        if info["converged"]:
            break
    if not info["converged"]:
        warnings.warn(f"HMC diagnostics: R-hat {info['r_hat']:.3f}, effective sample size "
                      f"{info['ess']:.0f}; the posterior may have several modes or a long valley: increase "
                      f"mcmc={{'n_samples': ...}} or check the parameters with fa.robustness(r)")
    return np.array([prob.full(t) for t in draws]), info


# --------------------------------------------------------------- Monte Carlo
def _mc_worker(args):
    prob, theta, seeds, noise_scale = args
    y0 = prob.y.copy()
    out, failed = [], 0
    for s in seeds:
        rng = np.random.default_rng(s)
        prob.y = y0 + rng.normal(0.0, prob.err * noise_scale)
        prob._cache = {}
        # 1e-8 instead of 1e-10: the same errors (tested on SDSS-RM stratified fits), 1.3x fewer evaluations
        r = _least_squares(prob, theta, max_nfev=200, tol=1e-8)
        out.append(prob.full(r.x))
        failed += not r.success
    prob.y = y0
    return out, failed


def _monte_carlo(prob, theta, n, noise_scale, seed, n_jobs):
    seeds = np.random.SeedSequence(seed).generate_state(n)
    if n_jobs is None:                     # all cores, but not inside a worker process
        n_jobs = 1 if multiprocessing.parent_process() is not None or n < 8 else -1
    n_jobs = (os.cpu_count() or 1) if n_jobs == -1 else max(int(n_jobs), 1)
    n_jobs = min(n_jobs, n)
    results = None
    if n_jobs > 1:
        chunks = [seeds[i::n_jobs] for i in range(n_jobs)]
        prob._cache = {}
        try:
            with environment(SINGLE_THREADED), ProcessPoolExecutor(n_jobs) as ex:
                results = list(ex.map(_mc_worker, [(prob, theta, c, noise_scale) for c in chunks]))
        except (RuntimeError, BrokenProcessPool, OSError) as exc:
            warnings.warn(
                f"parallel Monte Carlo failed ({type(exc).__name__}); running serially. "
                "In scripts, put your code under  if __name__ == '__main__':  to use n_jobs > 1.")
    if results is None:
        results = [_mc_worker((prob, theta, seeds, noise_scale))]
    failed = sum(f for _, f in results)
    if failed:
        warnings.warn(f"{failed} of {n} Monte Carlo refits did not converge; "
                      "their values are kept, so the errors may be unreliable")
    return np.array([row for rows, _ in results for row in rows])


# ------------------------------------------------------- many spectra in parallel
_WORKER = {}


def _init_fit_worker(model, options):
    """Runs once in each worker process: keep the model (it can be large)."""
    _WORKER["model"], _WORKER["options"] = model, options


def _fit_one(spectrum):
    try:
        return fit(spectrum, _WORKER["model"], verbose=False, **_WORKER["options"])
    except Exception as exc:                      # reported, not raised
        return exc


def _n_workers(n_jobs, n_tasks):
    cpus = os.cpu_count() or 1
    if n_jobs is None or n_jobs == -1:
        n_jobs = max(cpus - 1, 1)
    return max(1, min(int(n_jobs), n_tasks))


def fit_many(spectra, model, n_jobs=-1, verbose=True, **fit_options):
    """
    Fit the same model to many spectra (epochs of one object, or a sample) in
    parallel processes. Returns a list of FitResult in the order of
    ``spectra``; a spectrum that cannot be fitted gives the exception instead
    (with a warning).

    :param n_jobs: processes (-1: all cores but one; 1: serial). In scripts on
        macOS and Windows put the code under ``if __name__ == "__main__":``,
        otherwise the fits run serially.
    :param fit_options: passed to :func:`fit` (e.g. wave_range, errors).
    """
    t0 = time.time()
    spectra = list(spectra)
    n = len(spectra)
    workers = _n_workers(n_jobs, n)
    results = None
    task = f"fitting {n} spectra"
    report(task, 0, n)
    if workers > 1:
        try:
            with environment(SINGLE_THREADED), ProcessPoolExecutor(
                    workers, initializer=_init_fit_worker, initargs=(model, fit_options)) as ex:
                futures = {ex.submit(_fit_one, s): k for k, s in enumerate(spectra)}
                results = [None] * n
                for done, fut in enumerate(as_completed(futures), 1):
                    results[futures[fut]] = fut.result()
                    report(task, done, n, f"{workers} processes")
        except (RuntimeError, BrokenProcessPool, OSError) as exc:
            warnings.warn(f"parallel fitting failed ({type(exc).__name__}); running serially. "
                          "In scripts, put your code under  if __name__ == '__main__':  to use n_jobs > 1.")
            results = None
    if results is None:
        workers = 1
        _init_fit_worker(model, fit_options)
        results = []
        for s in spectra:
            results.append(_fit_one(s))
            report(task, len(results), n, "1 process")
    failed = [(s.name, r) for s, r in zip(spectra, results) if isinstance(r, Exception)]
    for name, exc in failed:
        warnings.warn(f"{name}: {type(exc).__name__}: {exc}")
    if verbose:
        print(f"fitted {n - len(failed)}/{n} spectra in {time.time() - t0:.1f} s "
              f"({workers} process{'es' if workers > 1 else ''})")
    return results
