"""
Joint fitting of several epochs of the same object.

Parameters matching the ``shared`` patterns (by default every narrow-line
parameter) take one value for all epochs; all others are fitted per epoch.
With ``intercalibrate=True`` each epoch after the first also gets a flux scale
factor, so that the constant narrow lines calibrate the relative flux of the
epochs (as in reverberation-mapping campaigns). Variable quantities (broad
lines, continuum) are then reported in the flux scale of the first epoch.
"""
import fnmatch
import time
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve

from .fitting import (Problem, _at_limit, _fix_unconstrained, _interior, _prepare,
                      _thread_map, fit_many)
from .fitting import fit as fit_single
from .parameters import Parameter, ParameterSet
from .progress import report
from .result import FitResult

__all__ = ("fit_epochs", "MultiEpochResult")


def _is_shared(name, patterns):
    glob = [p.replace("[", "[[]") for p in patterns]
    return any(name == p or fnmatch.fnmatchcase(name, g) for p, g in zip(patterns, glob))


def _level_ratios(data):
    """Median flux level of each epoch relative to the first, on common wavelengths."""
    x0, y0 = data[0][0], data[0][1]
    out = [1.0]
    for x, y, *_ in data[1:]:
        common = (x >= max(x0.min(), x.min())) & (x <= min(x0.max(), x.max()))
        ref = np.interp(x[common], x0, y0)
        r = np.median(y[common]) / np.median(ref) if common.sum() > 10 and np.median(ref) > 0 else 1.0
        out.append(float(r) if np.isfinite(r) and r > 0 else 1.0)
    return np.array(out)


class _JointProblem:
    def __init__(self, model, spectra, masks, shared, intercalibrate):
        self.model = model
        self.n = len(spectra)
        self.data = [(s.wave[m], s.flux[m], s.err[m], s.context) for s, m in zip(spectra, masks)]
        base = model.params
        joint = ParameterSet()
        self.index = []  # per epoch: joint index of each base parameter
        names = {}
        for p in base:
            if _is_shared(p.name, shared):
                joint.add(Parameter(**{**vars(p), "tie": None}))
                names[p.name] = [p.name] * self.n
            else:
                names[p.name] = [f"{p.name}@{k}" for k in range(self.n)]
                for k in range(self.n):
                    joint.add(Parameter(**{**vars(p), "name": names[p.name][k], "tie": None}))
        for p in base:
            if p.tie is None:
                continue
            target, factor, offset = p.tie
            if names[p.name][0] == p.name and names[target][0] != target:
                # shared on purpose (e.g. feii.shift, tied to broad.shift by the hbeta
                # preset): one free value for all epochs instead of the tie
                warnings.warn(f"{p.name} is shared, so it is fitted freely (one value for all "
                              f"epochs) instead of being tied to {target}, which varies")
                joint[p.name].tie = None
                continue
            for k in range(self.n):
                joint[names[p.name][k]].tie = (names[target][k], factor, offset)
        self.scale_names = []
        self.scale_guess = _level_ratios(self.data) if intercalibrate else None
        if intercalibrate:
            for k in range(self.n):
                # limits around the flux-level ratio: uncalibrated spectra (e.g. OzDES,
                # in counts) can differ between nights by much more than calibrated ones
                g = self.scale_guess[k]
                joint.add(Parameter(f"scale@{k}", g, g / 3.0, g * 3.0, fixed=(k == 0),
                                    description="flux calibration factor"))
                self.scale_names.append(f"scale@{k}")
        self.joint = joint
        jn = joint.names
        self.index = [np.array([jn.index(names[p.name][k]) for p in base]) for k in range(self.n)]
        self.scale_index = [jn.index(n) for n in self.scale_names]
        T, c, self.free = joint.affine()
        self.T, self.c = sparse.csr_matrix(T), c
        self.lb = np.array([p.min for p in self.free])
        self.ub = np.array([p.max for p in self.free])
        self.linear = np.array([p.linear for p in self.free], bool)
        self.rows = np.cumsum([0] + [d[0].size for d in self.data])

    def full(self, theta):
        return self.T @ theta + self.c

    def _epoch(self, k, full):
        x, y, err, ctx = self.data[k]
        f, Jb = self.model.evaluate_with_jacobian(x, ctx, full[self.index[k]])
        s = full[self.scale_index[k]] if self.scale_names else 1.0
        return x, y, err, f, Jb, s

    def residuals(self, theta):
        return self._evaluate(theta)[0]

    def jacobian(self, theta):
        return self._evaluate(theta)[1]

    def _evaluate(self, theta):
        key = theta.tobytes()
        if getattr(self, "_last", (None,))[0] == key:
            return self._last[1]
        full = self.full(theta)
        resid = []
        blocks_r, blocks_c, blocks_v = [], [], []
        # epochs are independent: evaluated in threads
        for k, (x, y, err, f, Jb, s) in enumerate(_thread_map(lambda k: self._epoch(k, full),
                                                               range(self.n))):
            resid.append((s * f - y) / err)
            rows = np.arange(self.rows[k], self.rows[k + 1])
            vals = s * Jb / err[:, None]
            cols = np.broadcast_to(self.index[k], vals.shape)
            blocks_r.append(np.repeat(rows, vals.shape[1]))
            blocks_c.append(cols.ravel())
            blocks_v.append(vals.ravel())
            if self.scale_names:
                blocks_r.append(rows)
                blocks_c.append(np.full(rows.size, self.scale_index[k]))
                blocks_v.append(f / err)
        J = sparse.csr_matrix((np.concatenate(blocks_v),
                               (np.concatenate(blocks_r), np.concatenate(blocks_c))),
                              shape=(self.rows[-1], len(self.c)))
        out = (np.concatenate(resid), (J @ self.T).tocsr())
        self._last = (key, out)
        return out


def fit_epochs(spectra, model, shared=("narrow.*",), intercalibrate=True, wave_range=None,
               errors="covariance", scale_errors=True, reference="snr", n_jobs=-1, verbose=True):
    """
    Fit several epochs of one object jointly.

    :param spectra: list of Spectrum (rest frame); ``mjd`` is used for light curves.
    :param model: Model to fit to every epoch.
    :param shared: parameter name patterns that are common to all epochs.
        Default: all narrow-line parameters. Add e.g. 'feii.fwhm', 'feii.shift'
        to also share the FeII kinematics.
    :param intercalibrate: fit a flux scale factor per epoch (reference epoch
        = 1). Needs shared narrow lines to be meaningful.
    :param reference: epoch that defines the flux scale: 'snr' (default, the
        epoch with the highest median S/N in the fitted range) or an index into
        ``spectra``. The uncertainty of the reference enters every other
        epoch's calibration, so a high-S/N reference gives smaller errors.
    :param wave_range: fitted window (default: the model's).
    :param errors: 'covariance' or None.
    :param scale_errors: inflate errors by sqrt(reduced chi2) if > 1.
    :param n_jobs: processes for the independent fits that give the starting
        values (-1: all cores but one; 1: serial); see :func:`fantasy_agn.fit_many`.
    :return: MultiEpochResult
    """
    t0 = time.time()
    if errors not in ("covariance", None):
        raise ValueError("multi-epoch errors must be 'covariance' or None")
    model = model.copy()
    wave_range = wave_range or model.wave_range
    usable = []
    for s in spectra:
        try:
            _prepare(s, wave_range)
            usable.append(s)
        except ValueError:
            warnings.warn(f"skipping {s.name}: no usable pixels in the fitted range")
    if len(usable) < 2:
        raise ValueError("need at least two spectra with data in the fitted range")
    if reference == "snr":
        def _snr(sp):
            g = _prepare(sp, wave_range)
            return float(np.median(sp.flux[g] / sp.err[g]))
        ref = int(np.argmax([_snr(sp) for sp in usable]))
    else:
        ref = usable.index(spectra[reference])
    # The reference epoch is fitted first internally; results keep the input order.
    order = [ref] + [i for i in range(len(usable)) if i != ref]
    spectra = [usable[i] for i in order]
    s0, m0 = spectra[0], _prepare(spectra[0], wave_range)
    _fix_unconstrained(Problem(model, s0.wave[m0], s0.flux[m0], s0.err[m0], s0.context), model)

    # 1) independent fits give starting values for every epoch
    singles = fit_many(spectra, model, n_jobs=n_jobs, verbose=verbose, wave_range=wave_range,
                       errors=None)
    bad = [s.name for s, r in zip(spectra, singles) if isinstance(r, Exception)]
    if bad:
        raise RuntimeError(f"the independent fit failed for {', '.join(bad)}")
    masks = [r.mask for r in singles]
    prob = _JointProblem(model, spectra, masks, shared, intercalibrate)

    full0 = np.zeros(len(prob.c))
    per_epoch = np.array([r.full for r in singles])
    ref = per_epoch[0]
    if intercalibrate:
        # Starting scale: median ratio of the shared amplitudes to the first epoch.
        lin = [i for i, p in enumerate(model.params) if p.linear and
               _is_shared(p.name, shared) and ref[i] > 0]
        for k, si in enumerate(prob.scale_index):
            # from the shared (narrow-line) amplitudes; if those are unusable (e.g. not
            # detected in this epoch's own fit), from the flux levels
            s0 = np.median(per_epoch[k, lin] / ref[lin]) if lin else np.nan
            g = prob.scale_guess[k]
            full0[si] = s0 if np.isfinite(s0) and g / 3 < s0 < g * 3 else g
    scales = np.array([full0[si] for si in prob.scale_index]) if intercalibrate else np.ones(len(spectra))
    for i, p in enumerate(model.params):
        vals = per_epoch[:, i] / (scales if p.linear else 1.0)
        for k in range(len(spectra)):
            full0[prob.index[k][i]] = vals[k]
        if _is_shared(p.name, shared):
            full0[prob.index[0][i]] = np.median(vals)
    jn = prob.joint.names
    theta0 = np.array([full0[jn.index(p.name)] for p in prob.free])

    # 2) joint fit with a sparse Jacobian
    report("joint fit", 0, None, f"{len(theta0)} parameters, {len(spectra)} epochs")
    res = _levenberg_marquardt(prob, _interior(theta0, prob.lb, prob.ub, prob.linear))
    theta = res.x
    full = prob.full(theta)
    chi2 = float(res.fun @ res.fun)
    dof = max(res.fun.size - theta.size, 1)
    scale = max(chi2 / dof, 1.0) if scale_errors else 1.0
    limit = _at_limit(theta, prob.lb, prob.ub)

    cov_full = None
    if errors == "covariance":
        J = res.jac if sparse.issparse(res.jac) else sparse.csr_matrix(res.jac)
        JtJ = (J.T @ J).toarray()
        cov = _cov_from_normal(JtJ, limit) * scale
        T = prob.T.toarray()
        cov_full = T @ cov @ T.T

    epochs = []
    for k, s in enumerate(spectra):
        idx = prob.index[k]
        sk = full[prob.scale_index[k]] if intercalibrate else 1.0
        m = model.copy()
        m.params.set_full_values(full[idx])
        spec = s.copy(flux=s.flux / sk, err=s.err / sk)
        r = res.fun[prob.rows[k]:prob.rows[k + 1]]
        free_k = [p.name for p in m.params if p.free]
        epochs.append(FitResult(
            spectrum=spec, model=m, mask=masks[k], full=full[idx],
            cov=None if cov_full is None else cov_full[np.ix_(idx, idx)], samples=None,
            chi2=float(r @ r), dof=max(r.size - len(free_k), 1), nfev=res.nfev,
            success=res.success, message=res.message,
            at_limit=[p.name for p, a in zip(prob.free, limit) if a and not p.linear
                      and (p.name.endswith(f"@{k}") or "@" not in p.name)],
            error_scale=np.sqrt(scale), time=0.0, error_method=errors))
        if intercalibrate:
            si = prob.scale_index[k]
            epochs[-1].scale = float(full[si])
            epochs[-1].scale_err = (float(np.sqrt(max(cov_full[si, si], 0)))
                                    if cov_full is not None else np.nan)
        else:
            epochs[-1].scale, epochs[-1].scale_err = 1.0, 0.0
    in_order = [None] * len(epochs)
    for internal, original in enumerate(order):
        in_order[original] = epochs[internal]
    result = MultiEpochResult(prob, full, cov_full, in_order, chi2, dof, res, shared,
                              np.sqrt(scale), time.time() - t0)
    result.reference = order[0]
    if intercalibrate:
        errs = [e.scale_err for k, e in enumerate(in_order) if k != result.reference and np.isfinite(e.scale_err)]
        result.calibration_error = float(np.median(errs)) if errs else np.nan
        if errs and result.calibration_error > 0.02:
            warnings.warn(
                f"the intercalibration is uncertain: median flux-scale error "
                f"{100 * result.calibration_error:.1f}% per epoch (the shared narrow lines are weak). "
                "This noise can hide real variability; compare with intercalibrate=False (the "
                "pipeline flux calibration), e.g. for SDSS spectra")
    if verbose:
        print(result.summary())
    return result


class _LMResult:
    pass


def _levenberg_marquardt(prob, theta, max_iter=1000, ftol=1e-10, window=20, min_gain=0.1):
    """
    Bounded Levenberg-Marquardt using the normal equations J^T J, which stay
    small (free parameters squared) however many epochs there are. Parameters
    on a bound with the gradient pushing outward are held fixed for the step.
    Converged when a step no longer improves chi2, or when chi2 improved by less
    than ``min_gain`` over the last ``window`` iterations (statistically
    irrelevant; with many epochs the last digits can take hundreds of steps).
    """
    lb, ub = prob.lb, prob.ub
    r, J = prob._evaluate(theta)
    cost = float(r @ r)
    lam, nfev, converged = 1e-3, 1, False
    costs = [cost]
    for _ in range(max_iter):
        g = J.T @ r
        A = (J.T @ J).toarray()
        d = np.diag(A).copy()
        d[d <= 0] = 1.0
        active = ((theta <= lb) & (g > 0)) | ((theta >= ub) & (g < 0))
        free = ~active
        Af, gf, df = A[np.ix_(free, free)], g[free], d[free]
        while True:
            try:
                c = cho_factor(Af + lam * np.diag(df))
                step = np.zeros_like(theta)
                step[free] = -cho_solve(c, gf)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            trial = np.clip(theta + step, lb, ub)
            rt, Jt = prob._evaluate(trial)
            nfev += 1
            ct = float(rt @ rt)
            if ct < cost:
                break
            lam *= 4
            if lam > 1e14:
                break
        if ct >= cost:  # no improving step: at a minimum
            converged = True
            break
        improvement = cost - ct
        theta, r, J, cost = trial, rt, Jt, ct
        report("joint fit", nfev, None, f"chi2 = {cost:.1f}")
        lam = max(lam / 3, 1e-12)
        costs.append(cost)
        if improvement <= ftol * cost or (len(costs) > window and costs[-window - 1] - cost < min_gain):
            converged = True
            break
    res = _LMResult()
    res.x, res.fun, res.jac, res.nfev = theta, r, J, nfev
    res.success, res.message = converged, "converged" if converged else "maximum iterations"
    return res


def _cov_from_normal(JtJ, exclude):
    n = JtJ.shape[0]
    cov = np.zeros((n, n))
    keep = ~exclude
    A = JtJ[np.ix_(keep, keep)]
    d = np.sqrt(np.clip(np.diag(A), 1e-300, None))
    w, V = np.linalg.eigh(A / np.outer(d, d))
    good = w > w.max() * 1e-12
    inv = (V[:, good] / w[good]) @ V[:, good].T
    cov[np.ix_(keep, keep)] = inv / np.outer(d, d)
    return cov


class MultiEpochResult:
    """
    Result of :func:`fit_epochs`:

    - ``epochs``: list of per-epoch FitResult (intercalibrated)
    - ``shared``: table of the shared parameters
    - ``scales``: flux calibration factors per epoch
    - ``light_curve()``: broad-line fluxes/widths and continuum per epoch
    """

    def __init__(self, prob, full, cov, epochs, chi2, dof, res, shared, error_scale, time):
        self._prob, self.full, self.cov = prob, full, cov
        self.epochs, self.chi2, self.dof, self.rchi2 = epochs, chi2, dof, chi2 / dof
        self.success, self.nfev, self.shared_patterns = res.success, res.nfev, shared
        self.error_scale, self.time = error_scale, time

    def _err(self, name):
        if self.cov is None:
            return np.nan
        i = self._prob.joint.names.index(name)
        return float(np.sqrt(max(self.cov[i, i], 0)))

    @property
    def shared(self):
        rows = []
        for p in self._prob.joint:
            if "@" not in p.name:
                i = self._prob.joint.names.index(p.name)
                rows.append((p.name, self.full[i], self._err(p.name), p.unit))
        return pd.DataFrame(rows, columns=["parameter", "value", "error", "unit"])

    @property
    def scales(self):
        """Flux calibration factor of each epoch relative to the reference epoch."""
        rows = [(k, r.spectrum.mjd, r.spectrum.name, r.scale, r.scale_err, k == self.reference)
                for k, r in enumerate(self.epochs)]
        return pd.DataFrame(rows, columns=["epoch", "mjd", "name", "scale", "error", "reference"])

    def _epoch_select(self, mjd_range):
        if mjd_range is None:
            return list(range(len(self.epochs)))
        lo, hi = mjd_range
        sel = [k for k, e in enumerate(self.epochs) if e.spectrum.mjd is not None and lo <= e.spectrum.mjd <= hi]
        if len(sel) < 5:
            raise ValueError(f"only {len(sel)} epochs in MJD {lo}-{hi}")
        return sel

    def rms_spectrum(self, line="Hb", mjd_range=None, **kw):
        """Mean and rms spectrum of a broad line (see :func:`fantasy_agn.rm.rms_spectrum`)."""
        from .rm import rms_spectrum
        return rms_spectrum([self.epochs[k] for k in self._epoch_select(mjd_range)], line, **kw)

    def reverberation(self, line="Hb", continuum=None, lag_range=None, dt=None, n_mc=1000,
                      f_virial=4.47, mjd_range=None, detrend=False, seed=0):
        """
        Lag of the broad ``line`` behind the continuum (ICCF with FR/RSS errors),
        its width in the rms spectrum and the reverberation mass; returns a
        :class:`fantasy_agn.rm.Reverberation` (print it, ``.table()``, ``.plot()``).

        :param continuum: None: the fitted AGN continuum light curve of these
            epochs; or (mjd, flux, error) arrays, e.g. photometry (better sampled).
        :param mjd_range: (min, max): only epochs (and continuum points) in this
            window, e.g. one observing season; lags need dense sampling.
        :param lag_range: (min, max) observed-frame lag to search, days (default:
            +-5x the lag expected from L5100 and the R-L relation, >= 60 d).
        :param detrend: subtract a linear trend from both light curves (Welsh
            1999): slow variations otherwise dominate the cross-correlation.
        """
        from .rm import Reverberation, ccf_lag
        lc = self.light_curve(lines=[line]).iloc[self._epoch_select(mjd_range)]
        if lc["mjd"].isna().any():
            raise ValueError("the epochs need MJDs for a lag")
        if f"{line}_flux" not in lc:
            raise ValueError(f"no broad {line} in the model")
        t2, f2, e2 = lc["mjd"].to_numpy(float), lc[f"{line}_flux"].to_numpy(float), lc[f"{line}_flux_err"].to_numpy(float)
        if continuum is None:
            cc = [c for c in lc.columns if c.startswith("cont") and not c.endswith("_err")]
            if not cc:
                raise ValueError("no continuum light curve in the fit; give continuum=(mjd, flux, error)")
            t1, f1, e1 = t2, lc[cc[0]].to_numpy(float), lc[cc[0] + "_err"].to_numpy(float)
            label = f"continuum {cc[0][4:]} A"
        else:
            t1, f1, e1 = (np.asarray(a, float) for a in continuum)
            label = "continuum (given)"
        if mjd_range is not None and continuum is not None:
            keep = (t1 >= mjd_range[0]) & (t1 <= mjd_range[1])
            t1, f1, e1 = t1[keep], f1[keep], e1[keep]
        if detrend:
            def flat(t, f):
                ok = np.isfinite(f)
                a, b = np.polyfit(t[ok], f[ok], 1)
                return f - (a * t + b) + np.nanmean(f)
            f1, f2 = flat(t1, f1), flat(t2, f2)
            label += ", detrended"
        z = self.epochs[0].spectrum.z
        if lag_range is None:
            # +-5x the lag expected from the R-L relation (Bentz et al. 2013), at least
            # 60 d and at most half the overlap: long baselines with seasonal gaps
            # otherwise produce spurious lags of hundreds of days
            span = min(t1.max(), t2.max()) - max(t1.min(), t2.min())
            lmax = 0.5 * span
            lcol = [c for c in lc.columns if c.startswith("logL") and not c.endswith("_err")]
            if lcol and np.isfinite(lc[lcol[0]]).any():
                tau = 10 ** (1.527 + 0.533 * (np.nanmedian(lc[lcol[0]]) - 44.0)) * (1 + (z or 0))
                lmax = min(lmax, max(5 * tau, 60.0))
            lag_range = (-lmax, lmax)
        lag = ccf_lag(t1, f1, e1, t2, f2, e2, lag_range=lag_range, dt=dt, n_mc=n_mc, seed=seed)
        rms = self.rms_spectrum(line, mjd_range=mjd_range)
        return Reverberation(line, lag, rms, label, z=z, f_virial=f_virial,
                             lc=((t1, f1, e1), (t2, f2, e2)))

    def light_curve(self, lines=None, continuum_at=None):
        """
        Table with, per epoch, the total broad flux and FWHM of each broad line
        (summed over broad components) and the continuum flux density.
        """
        first = self.epochs[0]
        broad = [c for c in first._line_components() if c.role == "broad"]
        if lines is None:
            lines = sorted({ln for c in broad for ln in c.line_names},
                           key=lambda ln: [c.line_names for c in broad if ln in c.line_names][0].index(ln))
        if continuum_at is None:
            pivots = [getattr(c, "pivot", None) for c in first.model.components
                      if c.kind == "continuum"]
            continuum_at = next((p for p in pivots if p), None)
        rows = []
        for k, r in enumerate(self.epochs):
            row = {"epoch": k, "mjd": r.spectrum.mjd, "name": r.spectrum.name}
            for ln in lines:
                comps = [c.name for c in broad if ln in c.line_names]
                m = r.measure(ln, comps)
                row[f"{ln}_flux"], row[f"{ln}_flux_err"] = m["flux"], m["flux_err"]
                row[f"{ln}_fwhm"], row[f"{ln}_fwhm_err"] = m["fwhm"], m["fwhm_err"]
                row[f"{ln}_ew"], row[f"{ln}_ew_err"] = m["ew"], m["ew_err"]
            if continuum_at:
                row[f"cont{continuum_at:g}"], row[f"cont{continuum_at:g}_err"] = r.continuum(continuum_at)
                if r.spectrum.z and r.spectrum.flux_unit:
                    row[f"logL{continuum_at:g}"], row[f"logL{continuum_at:g}_err"] = \
                        r.luminosity(continuum_at)
            rows.append(row)
        return pd.DataFrame(rows)

    def plot_baldwin(self, line="Hb", continuum_at=None, exclude=(), save=None, **kw):
        """
        Intrinsic Baldwin effect: equivalent width of the broad ``line`` against
        the continuum luminosity of each epoch, with a power-law fit (see
        :func:`fantasy_agn.plotting.plot_baldwin`). ``exclude`` lists epoch
        indices to leave out. Returns (figure, fit result).
        """
        from .plotting import plot_baldwin, savefig
        table = self.light_curve(lines=[line], continuum_at=continuum_at)
        table = table[~table.epoch.isin(list(exclude))]
        fig, fit = plot_baldwin(table, line, **kw)
        if save:
            savefig(fig, save)
        return fig, fit

    def plot_light_curves(self, columns=None, fig=None, width="double", save=None):
        """
        Light curves (broad-line fluxes and continuum by default), one panel
        each; ``save="lc.pdf"`` also writes the figure (pdf, eps, svg, png).
        """
        from .plotting import plot_light_curves, savefig
        table = self.light_curve()
        if columns is None:
            columns = [c for c in table.columns if c.endswith(("_flux",)) or c.startswith("cont")]
            columns = [c for c in columns if not c.endswith("_err")]
        out = plot_light_curves(table, columns, fig=fig, width=width,
                                flux_unit=self.epochs[0].spectrum.flux_unit)
        if save:
            savefig(out, save)
        return out

    def plot(self, epoch=0, **kw):
        return self.epochs[epoch].plot(**kw)

    def summary(self):
        lines = [f"Joint fit of {len(self.epochs)} epochs: chi2 = {self.chi2:.1f}, dof = {self.dof}, "
                 f"reduced chi2 = {self.rchi2:.3f}, {self.nfev} evaluations, {self.time:.1f} s",
                 f"  flux reference: epoch {getattr(self, 'reference', 0)} "
                 f"({self.epochs[getattr(self, 'reference', 0)].spectrum.name})",
                 f"  shared: {', '.join(self.shared_patterns)}"]
        if not self.success:
            lines.append("  WARNING: optimizer did not converge")
        if self.error_scale > 1:
            lines.append(f"  errors scaled by sqrt(reduced chi2) = {self.error_scale:.2f}")
        return "\n".join(lines)

    def __repr__(self):
        return self.summary()
