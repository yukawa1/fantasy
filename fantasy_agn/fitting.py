"""
Fitting engine: data-driven initial guess, trust-region least squares with
exact Jacobians, and uncertainty estimation.
"""
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from itertools import product

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares, lsq_linear, nnls

from .result import FitResult

__all__ = ("fit", "guess")


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

    def evaluate_basis(self, theta):
        """Model and amplitude columns of the Jacobian (cheaper; for linear solves)."""
        full = self.full(theta)
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
        f, Jfull = self.model.combine(self.x, results)
        return f, Jfull[:, self._rows_lin] @ self._TL       # amplitude columns only

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
            trial = theta.copy()
            for vals in product(*grids):
                trial[ks] = vals
                c2, _ = prob.solve_linear(trial)
                if c2 < best_chi2:
                    best_chi2, best_vals = c2, np.array(vals)
            theta[ks] = best_vals
            chi2, theta = prob.solve_linear(theta)
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
        trial = theta.copy()
        for vals in product(*[_grid(prob.free[k], ngrid) for k in ks]):
            trial[ks] = vals
            c2, th = prob.solve_linear(trial)
            if c2 < best[0]:
                best = (c2, th)
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
                grids = [_grid(prob.free[k], 9) for k in ks]
                best = (chi2, theta[ks].copy())
                trial = theta.copy()
                for vals in product(*grids):
                    trial[ks] = vals
                    c2, _ = prob.solve_linear(trial)
                    if c2 < best[0]:
                        best = (c2, np.array(vals))
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


def _least_squares(prob, theta0, max_nfev=None):
    return least_squares(prob.residuals, _interior(theta0, prob.lb, prob.ub, prob.linear),
                         jac=prob.jacobian, bounds=(prob.lb, prob.ub), method="trf",
                         x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-10,
                         max_nfev=max_nfev)


def guess(spectrum, model, wave_range=None, verbose=False):
    """Return a copy of ``model`` with data-driven initial parameter values."""
    model = model.copy()
    good = _prepare(spectrum, wave_range or model.wave_range)
    prob = Problem(model, spectrum.wave[good], spectrum.flux[good], spectrum.err[good],
                   spectrum.context)
    theta, _ = _initial_guess(prob, verbose=verbose)
    model.params.set_full_values(prob.full(theta))
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
        n_samples per chain (600). ``result.mcmc_info`` reports acceptance,
        split-R-hat and the effective sample size.
    :param n_mc: number of Monte Carlo realizations.
    :param scale_errors: if the reduced chi2 > 1, inflate uncertainties by
        sqrt(reduced chi2) (the error spectrum is then assumed underestimated).
    :param seed: random seed for the Monte Carlo errors.
    :param n_jobs: processes for Monte Carlo (default 1; -1 = all cores). In a
        script, parallel runs need the code under  if __name__ == "__main__":
    :param build_up: for models with overlapping components (several broad
        components of the same lines, outflow wings), also fit by adding them
        one at a time and keep the better of the two solutions
        (``result.start`` says which was used).
    :param verbose: print a short summary.
    :return: FitResult
    """
    t0 = time.time()
    model = model.copy()
    if wave_range is None:
        wave_range = model.wave_range
    good = _prepare(spectrum, wave_range)
    x, y, err = spectrum.wave[good], spectrum.flux[good], spectrum.err[good]
    prob = Problem(model, x, y, err, spectrum.context)
    unconstrained = _fix_unconstrained(prob, model)
    if unconstrained:
        prob = Problem(model, x, y, err, spectrum.context)

    theta = prob.theta_from_model()
    if guess:
        theta, _ = _initial_guess(prob, verbose=False)
    res = _least_squares(prob, theta)
    start_method = "direct"
    if guess:
        for _ in range(3):
            start = _escape_local_minimum(prob, res.x, float(res.fun @ res.fun))
            if start is None:
                break
            new = _least_squares(prob, start)
            if new.fun @ new.fun < res.fun @ res.fun:
                new.nfev += res.nfev
                res = new
    if guess and build_up:
        # Overlapping components (extra broad components, outflow wings) are
        # easily trapped in a local minimum when all are placed at once; also
        # build them up one at a time and keep the better solution.
        values = _staged_fit(model, x, y, err, spectrum.context)
        if values is not None:
            staged = np.array([values[p.name] for p in prob.free])
            new = _least_squares(prob, staged)
            if new.fun @ new.fun < res.fun @ res.fun * (1 - 1e-9):
                new.nfev += res.nfev
                res = new
                start_method = "staged"
    theta = res.x.copy()
    at_limit = _at_limit(theta, prob.lb, prob.ub)
    # Amplitudes that end on a bound (undetected components) are set exactly to it.
    snap = at_limit & prob.linear
    theta[snap] = np.where(np.abs(theta - prob.lb) < np.abs(prob.ub - theta), prob.lb, prob.ub)[snap]
    full = prob.full(theta)
    model.params.set_full_values(full)

    resid = res.fun
    chi2 = float(resid @ resid)
    dof = max(x.size - len(theta), 1)
    rchi2 = chi2 / dof
    scale = max(rchi2, 1.0) if scale_errors else 1.0

    cov_full = None
    samples = None
    if errors in ("covariance", "mc"):
        cov = _covariance(res.jac, at_limit) * scale
        cov_full = prob.T @ cov @ prob.T.T
    mcmc_info = None
    if errors == "mc":
        samples = _monte_carlo(prob, theta, n_mc, np.sqrt(scale), seed, n_jobs)
    elif errors == "mcmc":
        cov = _covariance(res.jac, at_limit) * scale
        cov_full = prob.T @ cov @ prob.T.T
        samples, mcmc_info = _mcmc(prob, theta, cov, scale, seed, at_limit, **(mcmc or {}))
    elif errors not in (None, "covariance"):
        raise ValueError("errors must be 'covariance', 'mc', 'mcmc' or None")

    result = FitResult(
        spectrum=spectrum, model=model, mask=good, full=full, cov=cov_full,
        samples=samples, chi2=chi2, dof=dof, nfev=res.nfev, success=res.success,
        message=res.message,
        at_limit=[p.name for p, a in zip(prob.free, at_limit) if a and not p.linear],
        at_zero=[p.name for p, a in zip(prob.free, at_limit) if a and p.linear],
        unconstrained=unconstrained,
        error_scale=np.sqrt(scale), time=time.time() - t0, error_method=errors)
    result.mcmc_info = mcmc_info
    result.start = start_method
    if verbose:
        print(result.summary(short=True))
    return result


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
    _, J = prob.evaluate(theta)
    J = J / prob.err[:, None]
    lin = np.nonzero(prob.linear)[0]
    if lin.size == 0:
        return []
    # Effect of each amplitude on chi2 per unit of its own typical size.
    power = np.sqrt((J[:, lin] ** 2).sum(axis=0))
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
def _mcmc(prob, theta, cov, scale, seed, at_limit, n_chains=4, n_warmup=300, n_samples=600):
    """
    Posterior samples by Hamiltonian Monte Carlo (see fantasy_agn.sampling),
    preconditioned with the fit covariance. Parameters that ended at a limit
    (e.g. undetected components) stay there, as for the covariance errors.
    """
    from .sampling import hmc
    idx = np.nonzero(~at_limit)[0]

    def log_prob_grad(sub):
        t = theta.copy()
        t[idx] = sub
        r = prob.residuals(t)
        J = prob.jacobian(t)[:, idx]
        return -0.5 * float(r @ r) / scale, -(J.T @ r) / scale

    sub0 = _interior(theta, prob.lb, prob.ub, prob.linear)[idx]
    draws, info = hmc(log_prob_grad, sub0, cov[np.ix_(idx, idx)], prob.lb[idx], prob.ub[idx],
                      n_chains=n_chains, n_warmup=n_warmup, n_samples=n_samples, seed=seed)
    if not info["converged"]:
        warnings.warn(f"HMC diagnostics: R-hat {info['r_hat']:.3f}, effective sample size "
                      f"{info['ess']:.0f}; increase mcmc={{'n_samples': ...}}")
    full = np.empty((draws.shape[0], prob.c.size))
    for i, sub in enumerate(draws):
        t = theta.copy()
        t[idx] = sub
        full[i] = prob.full(t)
    return full, info


# --------------------------------------------------------------- Monte Carlo
def _mc_worker(args):
    prob, theta, seeds, noise_scale = args
    y0 = prob.y.copy()
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        prob.y = y0 + rng.normal(0.0, prob.err * noise_scale)
        prob._cache = {}
        r = _least_squares(prob, theta, max_nfev=200)
        out.append(prob.full(r.x))
    prob.y = y0
    return out


def _monte_carlo(prob, theta, n, noise_scale, seed, n_jobs):
    seeds = np.random.SeedSequence(seed).generate_state(n)
    n_jobs = (os.cpu_count() or 1) if n_jobs == -1 else (n_jobs or 1)
    n_jobs = min(n_jobs, n)
    if n_jobs > 1:
        chunks = [seeds[i::n_jobs] for i in range(n_jobs)]
        prob._cache = {}
        try:
            with ProcessPoolExecutor(n_jobs) as ex:
                results = list(ex.map(_mc_worker, [(prob, theta, c, noise_scale) for c in chunks]))
            return np.array([row for chunk in results for row in chunk])
        except (RuntimeError, BrokenProcessPool, OSError) as exc:
            warnings.warn(
                f"parallel Monte Carlo failed ({type(exc).__name__}); running serially. "
                "In scripts, put your code under  if __name__ == '__main__':  to use n_jobs > 1.")
    return np.array(_mc_worker((prob, theta, seeds, noise_scale)))
