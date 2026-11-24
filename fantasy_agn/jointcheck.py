"""
Checks of a joint (multi-epoch) fit.

The intercalibration of a joint fit rests on one assumption: the shared
components (the narrow lines, the host) are the same in every epoch. Two checks
test it and the procedure:

``calibration_check(me)``
    Every epoch is fitted again on its own, its spectrum divided by the flux
    scale of the joint fit, with every parameter free. If the scales are right
    and the narrow lines really constant, the narrow-line fluxes of these fits
    agree within their errors. A scatter larger than the errors is an error of
    the intercalibration that the joint fit does not know about (extended
    narrow-line emission seen through different apertures, a resolution or
    wavelength scale that differs between epochs, a model that fits some epochs
    worse), and it adds to the noise of the light curves::

        chk = fa.calibration_check(me)
        chk.table        # per shared line flux: mean, chi2/dof, p, excess scatter, flag
        chk.epochs       # per epoch: the pull of each line, the worst one
        chk.summary()

``simulation_check(me, n=20)``
    Spectra are drawn from the joint fit (each epoch's model times its flux
    scale, with its noise), fitted jointly again exactly as the data were, and
    what comes back is compared with what went in: the flux scales, the broad
    line flux of every epoch (the light curve) and the shared parameters. Per
    quantity, pooled over the epochs: the bias in units of the error bar, the
    scatter of (fit - truth) / error (1 for right error bars) and the coverage.
"""
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ._threads import SINGLE_THREADED, environment
from .fitting import _n_workers
from .fitting import fit as fit_one
from .progress import report

__all__ = ("calibration_check", "CalibrationCheck", "JointSimulationCheck")


# ------------------------------------------------------------------ calibration check
def _refit_epoch(args):
    spectrum, model, options = args
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fit_one(spectrum, model, **options)
    except Exception as exc:                     # noqa: BLE001  (reported as a failed epoch)
        return exc


def _shared_fluxes(me):
    """Names of the shared line fluxes (free in the joint fit, one value for all epochs)."""
    names = []
    for p in me._prob.joint:
        if "@" in p.name or not p.linear or p.fixed or p.tie:
            continue
        if p.name.endswith(".flux"):
            names.append(p.name)
    return names


def _excess(values, errors, mean):
    """Fractional scatter s (of the mean) needed in quadrature for chi2/dof = 1 (0 if not needed)."""
    n = values.size
    if n < 3 or mean == 0:
        return 0.0

    def f(s):
        return np.sum((values - mean) ** 2 / (errors ** 2 + (s * mean) ** 2)) - (n - 1)
    if f(0.0) <= 0:
        return 0.0
    hi = 1.0
    while f(hi) > 0 and hi < 1e3:
        hi *= 2
    return float(optimize.brentq(f, 0.0, hi))


class CalibrationCheck:
    """
    Result of :func:`calibration_check`: ``table`` (one row per shared line flux), ``values`` and
    ``errors`` (epochs x lines, input order), ``epochs`` (per epoch: the worst pull and its line),
    ``failed`` (epochs whose own fit failed), ``calibration_error`` (median flux-scale error of the
    joint fit) and ``summary()``.
    """

    def __init__(self, me, table, values, errors, epochs, failed):
        self.result, self.table, self.values, self.errors, self.epochs, self.failed = me, table, values, errors, epochs, failed
        self.calibration_error = float(getattr(me, "calibration_error", np.nan))

    @property
    def ok(self):
        """True when no line scatters significantly more than its errors."""
        return bool((self.table.flag == "ok").all())

    @property
    def excess(self):
        """Excess scatter of the best measured line (fraction): the error of the intercalibration not in the error bars."""
        good = self.table[np.isfinite(self.table.snr)]
        return float(good.sort_values("snr", ascending=False).excess.iloc[0]) if len(good) else np.nan

    def summary(self):
        out = [f"Calibration check: {len(self.values)} epochs fitted on their own with the flux scales of the joint fit"
               + (f" ({len(self.failed)} failed)" if self.failed else "")]
        for _, r in self.table.iterrows():
            out.append(f"  {r.quantity}: chi2/dof {r.rchi2:.2f} (p = {r.p:.2g}), scatter {100 * r.scatter:.1f}% for errors of "
                       f"{100 * r.error:.1f}%, excess {100 * r.excess:.1f}%  {r.flag}")
        bad = self.epochs[np.abs(self.epochs.pull) > 3]
        if len(bad):
            out.append(f"  epochs off by more than 3 sigma: {', '.join(f'{int(e.epoch)} ({e.line} {e.pull:+.1f})' for _, e in bad.head(8).iterrows())}"
                       + (" ..." if len(bad) > 8 else ""))
        if np.isfinite(self.calibration_error):
            out.append(f"  flux-scale error of the joint fit: {100 * self.calibration_error:.1f}% per epoch (median)")
        return "\n".join(out)

    def __repr__(self):
        return self.summary()


def calibration_check(me, n_jobs=-1, **fit_options):
    """
    Fit every epoch of a joint fit again on its own (its spectrum divided by the joint flux scale,
    every parameter free) and test whether the shared narrow-line fluxes agree within their errors.
    See the module docstring.

    :param me: a MultiEpochResult with intercalibration.
    :param n_jobs: processes (-1: all cores but one; 1: serial).
    :param fit_options: passed to :func:`fantasy_agn.fit` (default: the fitted range, covariance errors).
    :return: :class:`CalibrationCheck`
    """
    names = _shared_fluxes(me)
    if not names:
        raise ValueError("the joint fit shares no line flux: there is nothing to check the calibration with")
    base = me._prob.model
    options = {"wave_range": base.wave_range, "errors": "covariance", "verbose": False, "n_jobs": 1, **fit_options}
    jobs = []
    for e in me.epochs:
        model = base.starting_from(e)
        jobs.append((e.spectrum, model, options))
    n = len(jobs)
    task = "calibration check"
    workers = _n_workers(n_jobs, n)
    fits = None
    report(task, 0, n)
    if workers > 1:
        try:
            with environment(SINGLE_THREADED), ProcessPoolExecutor(workers) as ex:
                fits = []
                for k, r in enumerate(ex.map(_refit_epoch, jobs), 1):
                    fits.append(r)
                    report(task, k, n, f"{workers} processes")
        except (RuntimeError, BrokenProcessPool, OSError):
            fits = None
    if fits is None:
        fits = []
        for k, job in enumerate(jobs, 1):
            fits.append(_refit_epoch(job))
            report(task, k, n, "1 process")
    failed = [k for k, r in enumerate(fits) if isinstance(r, Exception)]
    values = pd.DataFrame(np.nan, index=range(n), columns=names)
    errors = pd.DataFrame(np.nan, index=range(n), columns=names)
    for k, r in enumerate(fits):
        if isinstance(r, Exception):
            continue
        p = r.params.set_index("parameter")
        for name in names:
            if name in p.index:
                values.loc[k, name] = float(p.loc[name, "value"])
                errors.loc[k, name] = float(p.loc[name, "error"])
    rows = []
    pulls = pd.DataFrame(np.nan, index=range(n), columns=names)
    for name in names:
        v, e = values[name].to_numpy(float), errors[name].to_numpy(float)
        ok = np.isfinite(v) & np.isfinite(e) & (e > 0)
        if ok.sum() < 3:
            rows.append((name, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, int(ok.sum()), "not measured"))
            continue
        w = 1 / e[ok] ** 2
        mean = float(np.sum(w * v[ok]) / np.sum(w))
        chi2 = float(np.sum((v[ok] - mean) ** 2 * w))
        dof = int(ok.sum() - 1)
        p = float(stats.chi2.sf(chi2, dof))
        pulls.loc[np.flatnonzero(ok), name] = (v[ok] - mean) / e[ok]
        scatter = float(np.std(v[ok], ddof=1) / abs(mean)) if mean else np.nan
        err = float(np.median(e[ok]) / abs(mean)) if mean else np.nan
        excess = _excess(v[ok], e[ok], mean)
        snr = abs(mean) / float(np.median(e[ok]))
        flag = "scatters more than its errors" if p < 0.01 and excess > 0.005 else "ok"
        rows.append((name, mean, snr, chi2 / dof, p, scatter, err, excess, int(ok.sum()), flag))
    table = pd.DataFrame(rows, columns=["quantity", "mean", "snr", "rchi2", "p", "scatter", "error", "excess", "n", "flag"])
    table = table.sort_values("snr", ascending=False, na_position="last").reset_index(drop=True)
    worst = []
    for k in range(n):
        row = pulls.loc[k].to_numpy(float)
        if np.isfinite(row).any():
            j = int(np.nanargmax(np.abs(row)))
            worst.append((k, me.epochs[k].spectrum.name, names[j], float(row[j])))
        else:
            worst.append((k, me.epochs[k].spectrum.name, "", np.nan))
    epochs = pd.DataFrame(worst, columns=["epoch", "name", "line", "pull"])
    return CalibrationCheck(me, table, values, errors, epochs, failed)


# ------------------------------------------------------------------ joint simulation check
def _refit_joint(args):
    spectra, model, options = args
    from .multiepoch import fit_epochs
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fit_epochs(spectra, model, n_jobs=1, verbose=False, **options)
    except Exception as exc:                     # noqa: BLE001
        return exc


def _joint_options(me):
    o = dict(getattr(me, "options", None) or {})
    if not o:                                    # fitted before the options were kept
        prob = me._prob
        o = {"shared": tuple(me.shared_patterns), "intercalibrate": bool(prob.scale_names),
             "calibration_degree": prob.cal_degree, "epoch_scaled": tuple(c for c, _ in prob.scaled),
             "wavelength_offsets": getattr(prob, "offset_limit", 0.0) or False,
             "epoch_broadening": getattr(prob, "broadening_limit", 0.0) or False, "reference": me.reference}
    o.pop("errors", None)
    o["errors"] = "covariance"
    return o


def _truth_spectra(me):
    """Per epoch (input order): the spectrum as it was fitted (file resolution, original flux scale) and the
    noise-free model times its flux scale."""
    prob, order = me._prob, me._order
    out = []
    for orig, e in enumerate(me.epochs):
        k = order.index(orig)
        s = e.spectrum
        sk = prob.factor(k, me.full, s.wave) if prob.scale_names else 1.0
        mu = e.model_flux(s.wave) * sk
        err = s.err * sk                                  # the errors as fitted (after rescale_errors)
        meta = s.meta
        # the simulated spectrum keeps the data's gaps and the errors of the file: a refit with
        # rescale_errors=True rescales them again, as it did the data
        flux = np.where(np.isfinite(s.flux), mu, np.nan)
        spec = s.copy(flux=flux, err=err / meta.get("error_rescale", 1.0))
        if "resolution_file" in meta:
            spec = spec.copy(resolution=meta["resolution_file"], resolution_curve=meta.get("resolution_curve_file"))
        out.append((spec, mu, err))
    return out


def _joint_quantities(me):
    """{name: function(MultiEpochResult) -> (values, errors)} arrays over epochs (input order) or scalars."""
    q = {}
    if me._prob.scale_names:
        def scales(m):
            t = m.scales
            return t.scale.to_numpy(float), t.error.to_numpy(float)
        q["flux scale"] = scales
    lines = [c for c in me._prob.model.components if getattr(c, "role", None) == "broad" and getattr(c, "line_names", None)]
    if lines:
        c = lines[0]
        line = next((n for n in c.line_names if n in ("Hb", "Ha", "MgII2798", "CIV1549")), c.line_names[0])
        pname = f"{c.name}.{line}.flux"

        def broad(m):
            v, e = [], []
            for r in m.epochs:
                p = r.params.set_index("parameter")
                v.append(float(p.loc[pname, "value"]) if pname in p.index else np.nan)
                e.append(float(p.loc[pname, "error"]) if pname in p.index else np.nan)
            return np.array(v), np.array(e)
        q[f"{pname} per epoch"] = broad
    if me.offsets is not None:
        q["wavelength offset"] = lambda m: (m.offsets.offset.to_numpy(float), m.offsets.error.to_numpy(float))
    if me.broadening is not None:
        q["broadening"] = lambda m: (m.broadening.broadening.to_numpy(float), m.broadening.error.to_numpy(float))
    sh = me.shared
    for name in [n for n in sh.parameter if n.endswith(".fwhm") or n.endswith(".shift")][:3]:
        q[name] = (lambda nm: (lambda m: (np.array([float(m.shared.set_index("parameter").loc[nm, "value"])]),
                                          np.array([float(m.shared.set_index("parameter").loc[nm, "error"])]))))(name)
    return q


class JointSimulationCheck:
    """
    Result of ``simulation_check`` on a joint fit: ``table`` (one row per quantity, pooled over epochs and
    simulations), ``n`` fitted and ``failed`` simulations, ``summary()``.
    """

    def __init__(self, me, table, n, failed):
        self.result, self.table, self.n, self.failed = me, table, n, failed

    @property
    def biased(self):
        return list(self.table.quantity[self.table.flag == "biased"])

    def summary(self):
        out = [f"Simulation check of the joint fit: {self.n} sets of {len(self.result.epochs)} epochs drawn from the fit "
               "and fitted jointly again" + (f" ({self.failed} failed)" if self.failed else "")]
        for _, r in self.table.iterrows():
            out.append(f"  {r.quantity}: bias {r.bias_sd:+.2f} of the error bar, scatter {r.spread:.2f} error bars, truth within "
                       f"the error bar in {100 * r.cover68:.0f}%  {r.flag}")
        return "\n".join(out)

    def __repr__(self):
        return self.summary()


def joint_simulation_check(me, n=20, seed=0, n_jobs=-1):
    """
    Draw ``n`` sets of spectra from a joint fit, fit each set jointly as the data were and compare the
    results with the truth (see the module docstring). Called by :func:`fantasy_agn.simulation_check`
    for a MultiEpochResult.
    """
    n = int(n)
    if n < 2:
        raise ValueError("at least two simulations are needed")
    rng = np.random.default_rng(seed)
    truth = _truth_spectra(me)
    options = _joint_options(me)
    model = me._prob.model.copy()
    model._fitted = False
    quantities = _joint_quantities(me)
    true_vals = {k: f(me)[0] for k, f in quantities.items()}
    jobs = []
    for _ in range(n):
        sims = []
        for spec, mu, err in truth:
            noise = np.where(np.isfinite(err), err, 0.0) * me.error_scale
            flux = mu + rng.normal(0.0, 1.0, mu.size) * noise
            flux[~np.isfinite(spec.flux)] = np.nan          # the gaps of the data
            sims.append(spec.copy(flux=flux))
        jobs.append((sims, model, options))
    task = "simulation check"
    workers = _n_workers(n_jobs, n)
    fits = None
    report(task, 0, n)
    if workers > 1:
        try:
            with environment(SINGLE_THREADED), ProcessPoolExecutor(workers) as ex:
                fits = []
                for k, r in enumerate(ex.map(_refit_joint, jobs), 1):
                    fits.append(r)
                    report(task, k, n, f"{workers} processes")
        except (RuntimeError, BrokenProcessPool, OSError):
            fits = None
    if fits is None:
        fits = []
        for k, job in enumerate(jobs, 1):
            fits.append(_refit_joint(job))
            report(task, k, n, "1 process")
    good = [r for r in fits if not isinstance(r, Exception)]
    if len(good) < 2:
        first = next((r for r in fits if isinstance(r, Exception)), None)
        raise RuntimeError(f"the simulated epochs could not be fitted ({type(first).__name__}: {first})")
    rows = []
    for name, f in quantities.items():
        t = np.asarray(true_vals[name], float)
        pulls, signed = [], []
        for r in good:
            try:
                v, e = f(r)
            except Exception:                    # noqa: BLE001
                continue
            v, e = np.asarray(v, float), np.asarray(e, float)
            if v.shape != t.shape:
                continue
            ok = np.isfinite(v) & np.isfinite(e) & (e > 0) & np.isfinite(t)
            pulls.append((v[ok] - t[ok]) / e[ok])
        pull = np.concatenate(pulls) if pulls else np.zeros(0)
        if pull.size < 4:
            rows.append((name, np.nan, np.nan, np.nan, np.nan, int(pull.size), "not measured"))
            continue
        bias = float(np.mean(pull))
        spread = float(np.std(pull, ddof=1))
        c68 = float(np.mean(np.abs(pull) < 1))
        n_sets = len(pulls)
        # the pulls of one set share the reference epoch and the shared parameters: count sets, not epochs
        bias_z = bias / (spread / np.sqrt(n_sets)) if spread > 0 else np.nan
        tol = 3 * np.sqrt(0.68 * 0.32 / n_sets)
        if abs(bias_z) > 3 and abs(bias) > 0.3:
            flag = "biased"
        elif c68 < 0.68 - tol:
            flag = "error bar too small"
        elif c68 > 0.68 + tol:
            flag = "error bar too large"
        else:
            flag = "ok"
        rows.append((name, bias, bias_z, spread, c68, int(pull.size), flag))
    table = pd.DataFrame(rows, columns=["quantity", "bias_sd", "bias_z", "spread", "cover68", "n", "flag"])
    return JointSimulationCheck(me, table, len(good), len(fits) - len(good))
