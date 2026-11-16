"""
Would this fit recover the truth?

The errors of a fit say how much a result scatters; they do not say whether the
procedure returns the right value on average. A model with many weakly
constrained, bounded parameters can be biased: Fe II multiplets that cannot be
negative come out too strong when they are weak, and take flux from the
continuum and the broad line beside them. ``simulation_check`` measures this
for one fit::

    r = fa.fit(spec, fa.presets.hbeta())
    sim = fa.simulation_check(r, n=60)
    sim.table        # per result: value, mean of the simulations, bias, scatter, error bar, coverage
    sim.summary()

The fitted model is taken as the truth. ``n`` spectra are drawn from it with
the noise of the spectrum and fitted again from scratch, exactly as the
spectrum was (the data-driven first guess, the same model and limits, and the
Fe II pruning if the fit was pruned). For every result the table gives

- ``bias``: mean of the simulations minus the value in the fit, also in units
  of the scatter (``bias_sd``) and of its own uncertainty (``bias_z``);
- ``scatter``: the standard deviation over the simulations, the real
  statistical error of the procedure, next to the ``error`` the fits report;
- ``cover68`` and ``cover95``: how often the truth is within one and two
  reported error bars (0.68 and 0.95 for correct, unbiased errors);
- ``flag``: 'biased' (the bias is significant and larger than 0.3 of the
  scatter), 'error bar too small' or 'error bar too large' (the coverage is
  off by more than its own uncertainty allows), else 'ok'.

What it cannot show: the simulations contain only what the model contains. A
missing component, a wrong template or a wrong line profile is in the data and
not in the simulations, so 'ok' means that the fit procedure is sound for this
model at this signal-to-noise ratio, not that the model is right
(:func:`analyze_residuals`, :func:`compare_fits`). If the errors were scaled
(reduced chi2 > 1), the simulated noise is scaled the same way, as white noise.
"""
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd

from ._threads import SINGLE_THREADED, environment
from .fitting import _n_workers
from .fitting import fit as fit_one
from .progress import report

__all__ = ("simulation_check", "SimulationCheck", "default_quantities")

TASK = "simulation check"


def default_quantities(result):
    """
    {name: function(FitResult) -> (value, error)}: the main broad line (FWHM, sigma, flux, shift),
    R_FeII and the Fe II flux, the total [OIII]5007 line and the AGN continuum at its pivot.
    """
    from .robustness import default_quantities as base
    q = dict(base(result))
    if result._feii_components():
        def feii_flux(r):
            f = r.feii()
            return f["flux"], f["flux_err"]
        q["Fe II flux"] = feii_flux
    pivot = next((getattr(c, "pivot", None) for c in result.model.components if c.kind == "continuum"), None)
    if pivot:
        def continuum(r):
            return r.continuum(pivot)
        q[f"continuum at {pivot:g} A"] = continuum
    return q


def _refit(args):
    """One simulated spectrum fitted as the original was; exceptions are returned."""
    spectrum, model, options, prune = args
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = fit_one(spectrum, model, **options)
            if prune is not None:
                from .refine import prune_feii
                r = prune_feii(r, nsigma=prune)
        return r
    except Exception as exc:                     # noqa: BLE001  (reported as a failed simulation)
        return exc


def _procedure(result):
    """The model to fit the simulations with, and the pruning threshold if the fit was pruned."""
    model = result.model.copy()
    model._fitted = False                        # start from the data, not from the answer
    pruned = list(getattr(result, "feii_pruned", None) or [])
    if not pruned:
        return model, None
    # the fit before pruning had these amplitudes free (and the Fe II kinematics, which the
    # refit fixed if no multiplet was left)
    owners = {name.split(".")[0] for name in pruned}
    for name in pruned + [n for n in result.unconstrained if n.split(".")[0] in owners]:
        model.params[name].fixed = False
    return model, float(getattr(result, "feii_nsigma", 2.0))


def _simulate(result, n, seed):
    s = result.spectrum
    rng = np.random.default_rng(seed)
    mu = result.model_flux(s.wave)
    noise = np.where(np.isfinite(s.err), s.err, 0.0) * result.error_scale
    out = []
    for _ in range(n):
        flux = mu + rng.normal(0.0, 1.0, mu.size) * noise
        flux[~np.isfinite(s.flux)] = np.nan      # the same pixels are used
        out.append(s.copy(flux=flux))
    return out


def _evaluate(quantities, r):
    out = {}
    for name, f in quantities.items():
        try:
            v, e = f(r)
            out[name] = (float(v), float(e))
        except Exception:                        # noqa: BLE001  (e.g. the component vanished in this simulation)
            out[name] = (np.nan, np.nan)
    return out


class SimulationCheck:
    """
    Result of :func:`simulation_check`: ``table`` (one row per result), ``values`` and ``errors``
    (one row per simulation), ``n`` fitted and ``failed`` simulations, ``rchi2`` of the simulated
    fits, ``pruned`` (the Fe II pruning threshold that was repeated, or None) and ``summary()``.
    """

    def __init__(self, result, table, values, errors, n, failed, rchi2, pruned):
        self.result, self.table, self.values, self.errors = result, table, values, errors
        self.n, self.failed, self.rchi2, self.pruned = n, failed, rchi2, pruned

    @property
    def biased(self):
        """Names of the results flagged as biased."""
        return list(self.table.quantity[self.table.flag == "biased"])

    def summary(self):
        out = [f"Simulation check: {self.n} spectra drawn from the fit and fitted again"
               + (f" ({self.failed} failed)" if self.failed else "")
               + (f", Fe II pruned at {self.pruned:g} sigma as in the fit" if self.pruned is not None else "")]
        for _, r in self.table.iterrows():
            # a percentage means something only for a value well away from zero (not for a shift near 0)
            pct = f" ({100 * r.bias / abs(r.value):+.1f}%)" if abs(r.value) > 3 * r.scatter else ""
            cov = f", truth within the error bar in {100 * r.cover68:.0f}%" if np.isfinite(r.cover68) else ""
            out.append(f"  {r.quantity}: {r.value:.4g}, simulations {r['mean']:.4g}{pct}, bias {r.bias_sd:+.2f} of the "
                       f"scatter {r.scatter:.2g} (error bar {r.error:.2g}){cov}  {r.flag}")
        return "\n".join(out)

    def __repr__(self):
        return self.summary()


def _flag(bias_sd, bias_z, cover, n_err):
    if np.isfinite(bias_z) and abs(bias_z) > 3 and abs(bias_sd) > 0.3:
        return "biased"
    if not n_err:
        return "no error bar"
    tol = 3 * np.sqrt(0.68 * 0.32 / n_err)       # what the number of simulations allows
    if cover < 0.68 - tol:
        return "error bar too small"
    if cover > 0.68 + tol:
        return "error bar too large"
    return "ok"


def simulation_check(result, n=60, quantities=None, seed=0, n_jobs=-1, **fit_options):
    """
    Draw ``n`` spectra from a fit, fit them again as the spectrum was fitted, and compare what
    comes back with what went in: the bias of every result, its real scatter and whether the
    reported error bars cover the truth. See the module docstring.

    :param result: a FitResult. If it was pruned (:func:`prune_feii`), every simulation is
        fitted with all Fe II multiplets and pruned at the same threshold.
    :param n: number of simulations; the bias is known to scatter / sqrt(n), so 60 show a bias
        of 0.4 of the scatter at 3 sigma.
    :param quantities: {name: function(FitResult) -> (value, error)} (default:
        :func:`default_quantities`).
    :param seed: random seed of the simulated noise.
    :param n_jobs: processes (-1: all cores but one; 1: serial).
    :param fit_options: passed to :func:`fantasy_agn.fit`. Default: the fitted range of
        ``result`` and ``errors='covariance'`` (also for a fit with Monte Carlo or MCMC errors:
        the coverage then tests the covariance errors).
    :return: :class:`SimulationCheck`

    For a joint fit (a MultiEpochResult) the epochs are drawn together and fitted jointly again
    (default ``n=20``): see :func:`fantasy_agn.jointcheck.joint_simulation_check`; it returns a
    :class:`~fantasy_agn.jointcheck.JointSimulationCheck`.
    """
    from .multiepoch import MultiEpochResult
    if isinstance(result, MultiEpochResult):
        if quantities is not None or fit_options:
            raise ValueError("a joint fit is checked with its own quantities and options")
        from .jointcheck import joint_simulation_check
        return joint_simulation_check(result, n=n, seed=seed, n_jobs=n_jobs)
    n = int(n)
    if n < 2:
        raise ValueError("at least two simulations are needed")
    quantities = default_quantities(result) if quantities is None else dict(quantities)
    if not quantities:
        raise ValueError("no results to check: give quantities={name: function(FitResult) -> (value, error)}")
    model, prune = _procedure(result)
    wave = result.spectrum.wave[result.mask]
    options = {"wave_range": (float(wave.min()), float(wave.max())), "errors": "covariance", "verbose": False,
               "n_jobs": 1, **fit_options}
    truth = _evaluate(quantities, result)
    jobs = [(s, model, options, prune) for s in _simulate(result, n, seed)]

    workers = _n_workers(n_jobs, n)
    fits = None
    report(TASK, 0, n)
    if workers > 1:
        try:
            with environment(SINGLE_THREADED), ProcessPoolExecutor(workers) as ex:
                fits = []
                for k, r in enumerate(ex.map(_refit, jobs), 1):
                    fits.append(r)
                    report(TASK, k, n, f"{workers} processes")
        except (RuntimeError, BrokenProcessPool, OSError):
            fits = None
    if fits is None:
        fits = []
        for k, job in enumerate(jobs, 1):
            fits.append(_refit(job))
            report(TASK, k, n, "1 process")

    good = [r for r in fits if not isinstance(r, Exception)]
    if len(good) < 2:
        first = next((r for r in fits if isinstance(r, Exception)), None)
        raise RuntimeError(f"the simulated spectra could not be fitted ({type(first).__name__}: {first})")
    measured = [_evaluate(quantities, r) for r in good]
    values = pd.DataFrame([{k: m[k][0] for k in quantities} for m in measured])
    errors = pd.DataFrame([{k: m[k][1] for k in quantities} for m in measured])
    rows = []
    for name in quantities:
        t = truth[name][0]
        v, e = values[name].to_numpy(float), errors[name].to_numpy(float)
        ok = np.isfinite(v)
        v, e = v[ok], e[ok]
        if not np.isfinite(t) or v.size < 2:
            rows.append((name, t, truth[name][1], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                         int(v.size), "not measured"))
            continue
        mean, sd = float(v.mean()), float(v.std(ddof=1))
        bias = mean - t
        bias_sd = bias / sd if sd > 0 else np.nan
        bias_z = bias_sd * np.sqrt(v.size) if sd > 0 else np.nan
        has = np.isfinite(e) & (e > 0)
        pull = np.abs(v[has] - t) / e[has]
        c68 = float(np.mean(pull < 1)) if has.any() else np.nan
        c95 = float(np.mean(pull < 2)) if has.any() else np.nan
        rows.append((name, t, truth[name][1], mean, bias, bias_sd, bias_z, sd,
                     float(np.median(e[has])) if has.any() else np.nan, c68, c95, int(v.size),
                     _flag(bias_sd, bias_z, c68, int(has.sum()))))
    table = pd.DataFrame(rows, columns=["quantity", "value", "value_err", "mean", "bias", "bias_sd", "bias_z", "scatter",
                                        "error", "cover68", "cover95", "n", "flag"])
    return SimulationCheck(result, table, values, errors, len(good), len(fits) - len(good),
                           np.array([r.rchi2 for r in good]), prune)
