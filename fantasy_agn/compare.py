"""
Compare several decompositions of the same spectrum and estimate the
systematic uncertainty that comes from the choice of model.

Broad-line widths in particular depend on how a line is decomposed (number of
broad components, outflow wings) much more than on the noise. ``compare`` fits
each model and reports for every quantity the value of the best model (lowest
BIC) with its statistical error, and a systematic error from the spread over
the acceptable models::

    cmp = fa.compare(spec, fa.presets.variants("hbeta"))
    cmp.table          # one row per model: chi2, reduced chi2, BIC, quantities
    cmp.summary()      # best value, statistical, systematic and total error

Acceptable models are those whose reduced chi2 is within ``tolerance``
(default 10%) of the best one. The systematic error is the standard deviation
of the quantity over them. This is a heuristic: with high signal-to-noise data
no simple model fits perfectly (reduced chi2 > 1), so formal model weights such
as exp(-dBIC/2) put all the weight on one model and hide how much the answer
depends on the decomposition. If only one model is acceptable, the systematic
error is reported as NaN: then compare more alternatives or inspect the
residuals.
"""
import numpy as np
import pandas as pd

from .fitting import fit as fit_one

__all__ = ("compare", "Comparison", "default_quantities")


def default_quantities(result):
    """
    Functions of a FitResult to compare: total broad flux, FWHM and line
    dispersion of the main broad line, log lambda L at the continuum pivot and
    the black-hole mass when a recipe exists. Returns {name: function -> (value, error)}.
    """
    comps = [c for c in result._line_components() if c.role == "broad"]
    if not comps:
        return {}
    preferred = ("Hb", "Ha", "MgII2798", "CIV1549")
    lines = [ln for c in comps for ln in c.line_names]
    line = next((ln for ln in preferred if ln in lines), lines[0])

    def broad(key):
        def f(r):
            names = [c.name for c in r._line_components() if c.role == "broad" and line in c.line_names]
            m = r.measure(line, names)
            return m[key], m[key + "_err"]
        return f

    q = {f"{line} FWHM": broad("fwhm"), f"{line} sigma": broad("sigma"),
         f"{line} flux": broad("flux")}
    if line in ("Hb", "MgII2798", "CIV1549") and result.spectrum.z and result.spectrum.flux_unit:
        def mbh(r):
            m = r.bh_mass(line)
            return m["log_mbh"], m["log_mbh_err"]
        q["log M_BH"] = mbh
        at = {"Hb": 5100, "MgII2798": 3000, "CIV1549": 1350}[line]

        def lum(r):
            return r.luminosity(at)
        q[f"log L{at}"] = lum
    return q


class Comparison:
    """Result of :func:`compare`: ``results``, ``table`` and ``summary()``."""

    def __init__(self, results, quantities, tolerance=0.1):
        self.results = results
        self.quantities = quantities
        self.tolerance = tolerance
        rows = []
        for name, r in results.items():
            row = {"model": name, "chi2": r.chi2, "n_free": r.n_free, "reduced_chi2": r.rchi2,
                   "BIC": r.bic, "at_limit": ", ".join(r.at_limit)}
            for qn, f in quantities.items():
                try:
                    row[qn], row[qn + " err"] = (float(v) for v in f(r))
                except Exception:
                    row[qn], row[qn + " err"] = np.nan, np.nan
            rows.append(row)
        t = pd.DataFrame(rows).sort_values("BIC").reset_index(drop=True)
        t.insert(5, "dBIC", t.BIC - t.BIC.min())
        best_rchi2 = t.reduced_chi2.min()
        t.insert(6, "acceptable", t.reduced_chi2 <= (1 + tolerance) * best_rchi2)
        self.table = t
        self.best = t.loc[0, "model"]

    @property
    def result(self):
        """The FitResult of the best model (lowest BIC)."""
        return self.results[self.best]

    def systematic(self):
        """{quantity: systematic error} from :meth:`summary` (NaN if undetermined)."""
        return {row.quantity: row.sys_err for row in self.summary().itertuples()}

    def summary(self):
        """
        Per quantity: value of the best model with its statistical error, the
        systematic error (standard deviation over the acceptable models) and
        the total error, plus the range over all models.
        """
        t = self.table
        ok_models = t.acceptable.to_numpy()
        rows = []
        for qn in self.quantities:
            v = t[qn].to_numpy(float)
            good = np.isfinite(v) & ok_models
            if not np.isfinite(v).any():
                continue
            sys = float(np.std(v[good])) if good.sum() >= 2 else np.nan
            stat = float(t.loc[0, qn + " err"])
            rows.append({"quantity": qn, "value": float(t.loc[0, qn]), "stat_err": stat,
                         "sys_err": sys, "total_err": float(np.hypot(stat, sys)),
                         "n_acceptable": int(good.sum()),
                         "min_all": float(np.nanmin(v)), "max_all": float(np.nanmax(v))})
        return pd.DataFrame(rows)

    def __repr__(self):
        with pd.option_context("display.width", 160, "display.float_format", "{:.4g}".format):
            cols = ["model", "chi2", "n_free", "reduced_chi2", "dBIC", "acceptable"]
            return (f"Best model: {self.best}\n" + self.table[cols].to_string(index=False)
                    + "\n\n" + self.summary().to_string(index=False))


def _fit_named(args):
    spectrum, model, options = args
    try:
        return fit_one(spectrum, model, **options)
    except Exception as exc:                     # one failing model does not stop the comparison
        return exc


def compare(spectrum, models, quantities=None, tolerance=0.1, n_jobs=-1, **fit_options):
    """
    Fit several models to one spectrum and compare them.

    :param models: dict {name: Model} (e.g. ``fa.presets.variants("hbeta")``).
    :param quantities: dict {name: function(FitResult) -> (value, error)};
        default: broad-line FWHM, sigma and flux of the main broad line and the
        black-hole mass (see :func:`default_quantities`).
    :param tolerance: models with reduced chi2 within this fraction of the best
        one are acceptable and define the systematic error.
    :param n_jobs: processes for the fits (-1: all cores but one; 1: one after the other).
    :param fit_options: passed to :func:`fantasy_agn.fit` (e.g. errors='mc').
    :return: Comparison
    """
    import warnings
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    from ._threads import SINGLE_THREADED, environment
    from .fitting import _n_workers
    from .progress import report
    fit_options.setdefault("verbose", False)
    names = list(models)
    jobs = [(spectrum, models[n], fit_options) for n in names]
    workers = _n_workers(n_jobs, len(jobs))
    fits = None
    task = f"comparing {len(jobs)} models"
    report(task, 0, len(jobs))
    if workers > 1:
        try:
            with environment(SINGLE_THREADED), ProcessPoolExecutor(workers) as ex:
                fits = []
                for k, res in enumerate(ex.map(_fit_named, jobs), 1):
                    fits.append(res)
                    report(task, k, len(jobs), f"{workers} processes")
        except (RuntimeError, BrokenProcessPool, OSError):
            fits = None
    if fits is None:
        fits = []
        for k, job in enumerate(jobs, 1):
            fits.append(_fit_named(job))
            report(task, k, len(jobs), "1 process")
    results = {}
    for n, res in zip(names, fits):
        if isinstance(res, Exception):
            warnings.warn(f"model {n!r} failed: {type(res).__name__}: {res}")
        else:
            results[n] = res
    if not results:
        raise RuntimeError("all models failed")
    if quantities is None:
        quantities = default_quantities(next(iter(results.values())))
    return Comparison(results, quantities, tolerance)
