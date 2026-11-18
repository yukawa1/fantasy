"""
Reverberation mapping from a multi-epoch fit: mean and rms spectra of a broad
line, the line width in the rms spectrum, the lag between the continuum and
the line (interpolated cross-correlation function, ICCF) and the
reverberation black-hole mass.

    me = fa.fit_epochs(spectra, fa.presets.hbeta(), ...)
    rm = me.reverberation("Hb")                    # continuum: the fitted 5100 A light curve
    rm = me.reverberation("Hb", continuum=(t, f, e))   # or a photometric light curve
    rm                                             # lag, sigma_line(rms), FWHM(rms), log M_BH
    rm.plot()

Methods:

* rms spectrum: the broad line alone in every epoch (the intercalibrated data
  minus all other fitted emission, including broad blends),
  noise-corrected rms (Park et al. 2012, ApJS 203, 6); widths from the
  bootstrap over epochs.
* ICCF: Gaskell & Peterson (1987), White & Peterson (1994), both light curves
  interpolated; lag = centroid of r >= 0.8 r_max; errors from flux
  randomisation and random subset selection (Peterson et al. 1998, 2004),
  median and 68% interval of the centroid distribution.
* ZDCF: z-transformed discrete correlation function (Alexander 1997) with the
  port of pyZDCF (Jankov et al. 2022, AN 343, e20210090) in fantasy_agn.zdcf:
  no interpolation, equal-population bins; lag = centroid of ZDCF >= 0.8 of
  the peak, errors from flux randomisation and random subset selection.
* M_BH = f c tau sigma_line(rms)^2 / G with f = 4.47 (Woo et al. 2015); the
  ~0.3 dex scatter of f is not in the error.
"""
import numpy as np
import pandas as pd

__all__ = ("iccf", "ccf_lag", "vn_lag", "zdcf", "zdcf_lag", "gp_lag", "von_neumann", "fit_drw", "overlap_weight",
           "primary_peak", "component_reverberation", "ComponentReverberation",
           "rms_spectrum", "Reverberation")

C_KMS = 299792.458
G_MSUN = 1.32712440018e20        # G M_sun, m^3 s^-2
LIGHT_DAY = 2.59020683712e13     # m


# ------------------------------------------------------------------ ICCF
def _corr_columns(a, b, valid):
    """Pearson r per column of a and b, using only the valid entries."""
    n = valid.sum(axis=0)
    w = valid.astype(float)
    ma = (a * w).sum(axis=0) / np.maximum(n, 1)
    mb = (b * w).sum(axis=0) / np.maximum(n, 1)
    da, db = (a - ma) * w, (b - mb) * w
    cov = (da * db).sum(axis=0)
    va, vb = (da * da).sum(axis=0), (db * db).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(va * vb)
    r[(n < 3) | (va <= 0) | (vb <= 0)] = np.nan
    return r


def iccf(t1, f1, t2, f2, lags):
    """
    Interpolated cross-correlation r(lag) of light curve 2 against 1 (positive
    lag: 2 follows 1), averaging the two interpolation directions.
    """
    t1, f1, t2, f2 = (np.asarray(a, float) for a in (t1, f1, t2, f2))
    lags = np.asarray(lags, float)
    # np.interp needs increasing times and gives nonsense without a warning otherwise (epochs need not be in time order)
    o1, o2 = np.argsort(t1, kind="stable"), np.argsort(t2, kind="stable")
    t1, f1, t2, f2 = t1[o1], f1[o1], t2[o2], f2[o2]
    # interpolate 2 at t1 + lag (all lags at once)
    x = t1[:, None] + lags[None, :]
    v1 = (x >= t2.min()) & (x <= t2.max())
    r1 = _corr_columns(np.broadcast_to(f1[:, None], x.shape), np.interp(x, t2, f2), v1)
    # interpolate 1 at t2 - lag
    y = t2[:, None] - lags[None, :]
    v2 = (y >= t1.min()) & (y <= t1.max())
    r2 = _corr_columns(np.interp(y, t1, f1), np.broadcast_to(f2[:, None], y.shape), v2)
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.vstack([r1, r2]), axis=0)


def _centroid(lags, r, frac=0.8):
    if not np.isfinite(r).any():
        return np.nan, np.nan, np.nan
    k = int(np.nanargmax(r))
    rmax = r[k]
    if rmax <= 0:
        return np.nan, lags[k], rmax
    lo = k
    while lo > 0 and np.isfinite(r[lo - 1]) and r[lo - 1] >= frac * rmax:
        lo -= 1
    hi = k
    while hi < len(r) - 1 and np.isfinite(r[hi + 1]) and r[hi + 1] >= frac * rmax:
        hi += 1
    w = r[lo:hi + 1]
    return float(np.sum(w * lags[lo:hi + 1]) / np.sum(w)), float(lags[k]), float(rmax)


def _in_time_order(t1, f1, e1, t2, f2, e2):
    """Both light curves sorted by time, so that a lag and its Monte Carlo errors do not depend on the order of the epochs."""
    t1, f1, e1, t2, f2, e2 = (np.asarray(a, float) for a in (t1, f1, e1, t2, f2, e2))
    o1, o2 = np.argsort(t1, kind="stable"), np.argsort(t2, kind="stable")
    return t1[o1], f1[o1], e1[o1], t2[o2], f2[o2], e2[o2]


def ccf_lag(t1, f1, e1, t2, f2, e2, lag_range=None, dt=None, n_mc=1000, frac=0.8, alias=True, seed=0):
    """
    Lag of light curve 2 behind 1 with the ICCF and FR/RSS errors.

    :param lag_range: (min, max) lag to search, days (default: +-1/2 of the overlap).
    :param dt: lag step (default: a quarter of the median sampling, >= 0.25 d).
    :param alias: keep only the primary peak of the centroid distribution
        weighted for the light-curve overlap (:func:`overlap_weight`).
    :return: dict with tau_cen and its -/+ errors, tau_peak, r_max, the CCF
        (lags, r) and the centroid samples.
    """
    t1, f1, e1, t2, f2, e2 = _in_time_order(t1, f1, e1, t2, f2, e2)
    t1, f1, e1, t2, f2, e2 = (np.asarray(a, float) for a in (t1, f1, e1, t2, f2, e2))
    ok1, ok2 = np.isfinite(f1) & np.isfinite(t1), np.isfinite(f2) & np.isfinite(t2)
    t1, f1, e1, t2, f2, e2 = t1[ok1], f1[ok1], e1[ok1], t2[ok2], f2[ok2], e2[ok2]
    if len(t1) < 5 or len(t2) < 5:
        raise ValueError("the light curves need at least five points each")
    span = min(t1.max(), t2.max()) - max(t1.min(), t2.min())
    if lag_range is None:
        lag_range = (-0.5 * span, 0.5 * span)
    if dt is None:
        dt = max(0.25, 0.25 * float(np.median(np.diff(np.sort(np.concatenate([t1, t2]))))))
    lags = np.arange(lag_range[0], lag_range[1] + dt / 2, dt)
    r = iccf(t1, f1, t2, f2, lags)
    tau_cen, tau_peak, rmax = _centroid(lags, r, frac)
    rng = np.random.default_rng(seed)
    cents = []
    e1 = np.where(np.isfinite(e1), e1, 0.0)
    e2 = np.where(np.isfinite(e2), e2, 0.0)
    for _ in range(int(n_mc)):
        # random subset selection (repeats dropped) and flux randomisation
        i1 = np.unique(rng.integers(0, len(t1), len(t1)))
        i2 = np.unique(rng.integers(0, len(t2), len(t2)))
        if len(i1) < 4 or len(i2) < 4:
            continue
        rr = iccf(t1[i1], f1[i1] + rng.normal(0, e1[i1]), t2[i2], f2[i2] + rng.normal(0, e2[i2]), lags)
        c, _, rm = _centroid(lags, rr, frac)
        if np.isfinite(c) and rm > 0:
            cents.append(c)
    cents = np.array(cents)
    weight = overlap_weight(t1, t2, lags) if alias else np.ones_like(lags)
    s = _summarise(cents, lags, weight)
    return {"method": "ICCF", "tau_cen": s["tau"], "tau_err_lo": s["lo"], "tau_err_hi": s["hi"],
            "tau_cen_ccf": tau_cen, "tau_peak": tau_peak, "r_max": rmax, "lags": lags, "r": r,
            "peak_bounds": s["peak"], "frac_rej": s["frac_rej"],
            "samples": s["samples"], "samples_all": s["samples_all"], "n_mc": int(cents.size)}


# ------------------------------------------------- seasonal aliases
def overlap_weight(t1, t2, lags, gap=None):
    """
    Weight of each lag for the overlap of the light curves, P(tau) =
    [N(tau)/N(0)]^2 (Grier et al. 2017, ApJ 851, 21): N is the number of line
    epochs t whose continuum at t - tau is sampled (a continuum point within
    ``gap`` days; default three times the typical continuum spacing, >= 5 d).
    Lags that move the line epochs into the seasonal gaps get little weight.
    """
    t1 = np.unique(np.asarray(t1, float))
    t2, lags = np.asarray(t2, float), np.atleast_1d(np.asarray(lags, float))
    if gap is None:
        gap = max(5.0, 3.0 * float(np.median(np.diff(np.unique(np.round(t1))))) if t1.size > 2 else 5.0)

    def n(lag):
        q = t2[None, :] - lag[:, None]
        k = np.searchsorted(t1, q)
        near = np.minimum(np.abs(q - t1[np.clip(k - 1, 0, t1.size - 1)]), np.abs(t1[np.clip(k, 0, t1.size - 1)] - q))
        return (near <= gap).sum(axis=1).astype(float)

    return (n(lags) / max(float(n(np.zeros(1))[0]), 1.0)) ** 2


def _smooth(y, sigma_steps):
    if sigma_steps <= 0:
        return y
    h = int(np.ceil(4 * sigma_steps))
    k = np.exp(-0.5 * (np.arange(-h, h + 1) / sigma_steps) ** 2)
    return np.convolve(y, k / k.sum(), mode="same")


def primary_peak(lags, density, smooth=None):
    """
    Bounds (lo, hi) of the primary peak of a lag distribution on the grid
    ``lags``: the highest peak of the smoothed density, out to the minima on
    either side (Grier et al. 2017; Shen et al. 2024).
    """
    lags, d = np.asarray(lags, float), np.asarray(density, float)
    step = lags[1] - lags[0] if lags.size > 1 else 1.0
    d = _smooth(np.nan_to_num(d), (smooth if smooth is not None else 5.0) / step)
    i = int(np.argmax(d))
    lo = i
    while lo > 0 and d[lo - 1] <= d[lo] and d[lo] > 0:
        lo -= 1
    hi = i
    while hi < d.size - 1 and d[hi + 1] <= d[hi] and d[hi] > 0:
        hi += 1
    return float(lags[lo]), float(lags[hi])


def _summarise(samples, lags, weight, smooth=None):
    """Median and 68% interval of the samples in the primary peak of the weighted distribution."""
    samples = np.asarray(samples, float)
    samples = samples[np.isfinite(samples)]
    out = {"samples_all": samples}
    if not samples.size:
        out.update(tau=np.nan, lo=np.nan, hi=np.nan, peak=(np.nan, np.nan), frac_rej=np.nan, samples=samples)
        return out
    step = lags[1] - lags[0]
    edges = np.concatenate([lags - step / 2, [lags[-1] + step / 2]])
    hist = np.histogram(samples, edges)[0].astype(float)
    bounds = primary_peak(lags, hist * weight, smooth)
    keep = samples[(samples >= bounds[0] - step / 2) & (samples <= bounds[1] + step / 2)]
    if not keep.size:
        keep = samples
    lo, med, hi = np.percentile(keep, [15.865, 50, 84.135])
    out.update(tau=float(med), lo=float(med - lo), hi=float(hi - med), peak=bounds,
               frac_rej=float(1 - keep.size / samples.size), samples=keep)
    return out


# ---------------------------------------------------- von Neumann estimator
def von_neumann(t1, f1, t2, f2, lags):
    """
    Von Neumann estimator (Chelouche, Pozo-Nunez & Zucker 2017, ApJ 844, 146):
    the mean squared successive difference of the combined, standardised
    light curves with the line shifted back by each lag; minimal at the lag.
    No interpolation, so it is less affected by gaps than the ICCF.
    """
    t1, t2 = np.asarray(t1, float), np.asarray(t2, float)
    a = (np.asarray(f1, float) - np.mean(f1)) / np.std(f1)
    b = (np.asarray(f2, float) - np.mean(f2)) / np.std(f2)
    v = np.concatenate([a, b])
    out = np.empty(len(lags))
    for i, tau in enumerate(lags):
        o = np.argsort(np.concatenate([t1, t2 - tau]), kind="stable")
        out[i] = np.mean(np.diff(v[o]) ** 2)
    return out


def vn_lag(t1, f1, e1, t2, f2, e2, lag_range, dt=1.0, n_mc=300, alias=True, seed=0):
    """
    Lag of light curve 2 behind 1 with the von Neumann estimator; errors from
    flux randomisation and random subset selection (as for the ICCF), primary
    peak of the lag distribution weighted for the light-curve overlap.
    """
    t1, f1, e1, t2, f2, e2 = _clean(t1, f1, e1, t2, f2, e2)
    lags = np.arange(lag_range[0], lag_range[1] + dt / 2, dt)
    curve = von_neumann(t1, f1, t2, f2, lags)
    rng = np.random.default_rng(seed)
    mins = []
    for _ in range(int(n_mc)):
        i1 = np.unique(rng.integers(0, len(t1), len(t1)))
        i2 = np.unique(rng.integers(0, len(t2), len(t2)))
        if len(i1) < 4 or len(i2) < 4:
            continue
        c = von_neumann(t1[i1], f1[i1] + rng.normal(0, e1[i1]), t2[i2], f2[i2] + rng.normal(0, e2[i2]), lags)
        mins.append(lags[np.argmin(c)])
    weight = overlap_weight(t1, t2, lags) if alias else np.ones_like(lags)
    s = _summarise(mins, lags, weight)
    return {"method": "von Neumann", "tau_cen": s["tau"], "tau_err_lo": s["lo"], "tau_err_hi": s["hi"],
            "tau_peak": float(lags[np.argmin(curve)]), "peak_bounds": s["peak"], "frac_rej": s["frac_rej"],
            "lags": lags, "statistic": curve, "samples": s["samples"], "samples_all": s["samples_all"], "n_mc": len(mins)}


# ---------------------------------------------------------------------- ZDCF
def zdcf(t1, f1, e1, t2, f2, e2, minpts=0, num_mc=100, uniform_sampling=False, omit_zero_lags=True):
    """
    ZDCF of light curve 2 against 1 (Alexander 1997) with the port of pyZDCF
    (Jankov et al. 2022) in :mod:`fantasy_agn.zdcf`: DataFrame with tau
    (positive: 2 lags 1), -sig(tau), +sig(tau), dcf, -err(dcf), +err(dcf), #bin.
    """
    from .zdcf import zdcf as _zdcf
    t1, f1, e1, t2, f2, e2 = _clean(t1, f1, e1, t2, f2, e2)
    return _zdcf((t1, f1, e1), (t2, f2, e2), minpts=minpts, num_mc=num_mc,
                 uniform_sampling=uniform_sampling, omit_zero_lags=omit_zero_lags)


def _zdcf_centroid(z, lag_range, frac=0.8):
    z = z[(z.tau >= lag_range[0]) & (z.tau <= lag_range[1])]
    if len(z) < 3 or not np.isfinite(z.dcf).any():
        return np.nan, np.nan, np.nan
    i = int(np.nanargmax(z.dcf.to_numpy()))
    tau, r = z.tau.to_numpy(), z.dcf.to_numpy()
    lo = i
    while lo > 0 and r[lo - 1] >= frac * r[i]:
        lo -= 1
    hi = i
    while hi < len(r) - 1 and r[hi + 1] >= frac * r[i]:
        hi += 1
    w = r[lo:hi + 1]
    return float(np.sum(w * tau[lo:hi + 1]) / np.sum(w)), float(tau[i]), float(r[i])


def zdcf_lag(t1, f1, e1, t2, f2, e2, lag_range, n_mc=100, minpts=0, frac=0.8, alias=True, seed=0):
    """
    Lag of light curve 2 behind 1 from the ZDCF (:func:`zdcf`):
    centroid of the ZDCF points >= ``frac`` of its peak inside ``lag_range``;
    errors from ``n_mc`` flux-randomisation / random-subset realisations, each
    with its own ZDCF, and the primary peak of the lag distribution weighted
    for the light-curve overlap (as for the ICCF).
    """
    t1, f1, e1, t2, f2, e2 = _clean(t1, f1, e1, t2, f2, e2)
    z = zdcf(t1, f1, e1, t2, f2, e2, minpts=minpts)
    cen, peak, rmax = _zdcf_centroid(z, lag_range, frac)
    rng = np.random.default_rng(seed)
    cents = []
    for _ in range(int(n_mc)):
        i1 = np.unique(rng.integers(0, len(t1), len(t1)))
        i2 = np.unique(rng.integers(0, len(t2), len(t2)))
        if len(i1) < 6 or len(i2) < 6:
            continue
        zz = zdcf(t1[i1], f1[i1] + rng.normal(0, e1[i1]), e1[i1], t2[i2], f2[i2] + rng.normal(0, e2[i2]), e2[i2],
                  minpts=minpts, num_mc=0)
        cents.append(_zdcf_centroid(zz, lag_range, frac)[0])
    step = max(np.median(np.diff(np.sort(z.tau))) / 4 if len(z) > 1 else 1.0, 0.5)
    lags = np.arange(lag_range[0], lag_range[1] + step / 2, step)
    weight = overlap_weight(t1, t2, lags) if alias else np.ones_like(lags)
    s = _summarise(cents, lags, weight)
    return {"method": "ZDCF", "tau_cen": s["tau"], "tau_err_lo": s["lo"], "tau_err_hi": s["hi"],
            "tau_peak": peak, "tau_cen_data": cen, "r_max": rmax, "peak_bounds": s["peak"],
            "frac_rej": s["frac_rej"], "zdcf": z, "samples": s["samples"], "samples_all": s["samples_all"], "n_mc": len(cents)}


def _clean(t1, f1, e1, t2, f2, e2):
    t1, f1, e1, t2, f2, e2 = _in_time_order(t1, f1, e1, t2, f2, e2)
    ok1, ok2 = np.isfinite(f1) & np.isfinite(t1), np.isfinite(f2) & np.isfinite(t2)
    e1, e2 = np.where(np.isfinite(e1), e1, 0.0), np.where(np.isfinite(e2), e2, 0.0)
    return t1[ok1], f1[ok1], e1[ok1], t2[ok2], f2[ok2], e2[ok2]


# ------------------------------------- damped random walk with correlated errors
def _drw(dt, var, tau_d):
    return var * np.exp(-np.abs(dt) / tau_d)


def _gls(y, X, C):
    """Gaussian log-likelihood with the linear mean X beta profiled (restricted likelihood)."""
    from scipy.linalg import cho_factor, cho_solve
    try:
        cf = cho_factor(C, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return -np.inf, None
    Ci_X, Ci_y = cho_solve(cf, X, check_finite=False), cho_solve(cf, y, check_finite=False)
    A = X.T @ Ci_X
    beta = np.linalg.solve(A, X.T @ Ci_y)
    r = y - X @ beta
    logdet = 2 * np.sum(np.log(np.diag(cf[0])))
    return float(-0.5 * (r @ cho_solve(cf, r, check_finite=False) + logdet + np.linalg.slogdet(A)[1])), beta


def fit_drw(t, f, e):
    """Damped random walk (variance, timescale, extra noise) fitted to a light curve; returns a dict."""
    from scipy.optimize import minimize
    t, f, e = (np.asarray(a, float) for a in (t, f, e))
    span = t.max() - t.min()
    X = np.ones((t.size, 1))
    dt = t[:, None] - t[None, :]

    def nll(p):
        C = _drw(dt, np.exp(2 * p[0]), np.exp(p[1])) + np.diag(e ** 2 + np.exp(2 * p[2]))
        return -_gls(f, X, C)[0]

    sd = float(np.std(f))
    x0 = [np.log(sd), np.log(min(100.0, span / 5)), np.log(0.1 * sd)]
    bounds = [(np.log(sd) - 4, np.log(sd) + 3), (np.log(1.0), np.log(10 * span)), (np.log(sd) - 9, np.log(sd))]
    p = minimize(nll, x0, method="L-BFGS-B", bounds=bounds).x
    C = _drw(dt, np.exp(2 * p[0]), np.exp(p[1])) + np.diag(e ** 2 + np.exp(2 * p[2]))
    return {"var": float(np.exp(2 * p[0])), "tau_d": float(np.exp(p[1])), "jitter": float(np.exp(p[2])),
            "mean": float(_gls(f, X, C)[1][0])}


class _Driver:
    """The continuum process conditioned on the driving light curve (or the prior, without one)."""

    def __init__(self, drw, t=None, f=None, e=None):
        from scipy.linalg import cho_factor, cho_solve
        self.var, self.tau_d = drw["var"], drw["tau_d"]
        self.t = None if t is None else np.asarray(t, float)
        if self.t is not None:
            C = _drw(self.t[:, None] - self.t[None, :], self.var, self.tau_d) + \
                np.diag(np.asarray(e, float) ** 2 + drw["jitter"] ** 2)
            self.cf = cho_factor(C, lower=True)
            self.alpha = cho_solve(self.cf, np.asarray(f, float) - drw["mean"])

    def condition(self, q):
        """Mean and covariance of the continuum variation x(q) at times q."""
        from scipy.linalg import solve_triangular
        K = _drw(q[:, None] - q[None, :], self.var, self.tau_d)
        if self.t is None:
            return np.zeros(q.size), K
        Kq = _drw(q[:, None] - self.t[None, :], self.var, self.tau_d)
        V = solve_triangular(self.cf[0], Kq.T, lower=True, check_finite=False)
        return Kq @ self.alpha, K - V.T @ V


def gp_lag(line, continuum=None, spec_continuum=None, corr=None, lag_range=(-100, 100), dt=1.0,
           width=0.0, alias=True, n_samples=4000, seed=0):
    """
    Lag of a broad line from a damped-random-walk model of the continuum
    (in the spirit of JAVELIN, Zu et al. 2011) that also models the errors of
    the spectral decomposition: in each epoch the line flux and the
    spectroscopic continuum come from the same fit and the same flux
    calibration, so their errors are correlated (``corr``, from the fit's
    covariance). Ignoring that biases lags against the spectroscopic
    continuum towards zero.

    The line is the continuum variation x(t - tau) (smoothed by a top-hat of
    ``width`` days) times a response amplitude plus a mean; the spectroscopic
    continuum is x(t) times its own amplitude plus a mean. With a photometric
    ``continuum`` the process is conditioned on it (the DRW is fitted to it)
    and the spectroscopic continuum, if given, adds information with its
    correlated errors; without one the DRW is fitted to the spectroscopic
    continuum. Amplitudes, means and extra noise terms are optimised at each
    lag of the grid (profile likelihood); the lag distribution is the
    likelihood times the overlap weight, restricted to its primary peak.

    :param line, continuum, spec_continuum: (t, flux, error) arrays; the
        spectroscopic continuum on the same epochs as the line.
    :param corr: per-epoch correlation of the line and spectroscopic-continuum errors.
    :return: dict like :func:`ccf_lag` (tau_cen, errors, the lag grid, the log-likelihood).
    """
    from scipy.optimize import minimize
    tL, fL, eL = (np.asarray(a, float) for a in line)
    okL = np.isfinite(tL) & np.isfinite(fL) & np.isfinite(eL)
    have_s = spec_continuum is not None
    if have_s:
        tS, fS, eS = (np.asarray(a, float) for a in spec_continuum)
        rho = np.zeros(tS.size) if corr is None else np.nan_to_num(np.asarray(corr, float))
        okL &= np.isfinite(fS) & np.isfinite(eS)
        tS, fS, eS, rho = tS[okL], fS[okL], eS[okL], rho[okL]
    tL, fL, eL = tL[okL], fL[okL], eL[okL]
    n = tL.size
    if continuum is not None:
        tC, fC, eC = (np.asarray(a, float) for a in continuum)
        ok = np.isfinite(tC) & np.isfinite(fC) & np.isfinite(eC)
        tC, fC, eC = tC[ok], fC[ok], eC[ok]
        # continuum points far outside the line epochs carry no information on the lag
        pad = max(abs(lag_range[0]), abs(lag_range[1]))
        keep = (tC > tL.min() - pad - 200) & (tC < tL.max() + pad + 200)
        tC, fC, eC = tC[keep], fC[keep], eC[keep]
        drw = fit_drw(tC, fC, eC)
        driver = _Driver(drw, tC, fC, eC)
        free_s = True
    elif have_s:
        drw = fit_drw(tS, fS, eS)
        driver = _Driver(drw)
        free_s = False
    else:
        raise ValueError("gp_lag needs a continuum or a spectroscopic continuum")
    u = np.linspace(-width / 2, width / 2, 5) if width > 0 else np.zeros(1)
    m = u.size
    # averaging matrix of the top-hat sub-samples
    M = np.zeros((n, n * m))
    for j in range(m):
        M[np.arange(n), np.arange(n) * m + j] = 1.0 / m
    if have_s:
        y = np.concatenate([fS, fL])
        X = np.zeros((2 * n, 2))
        X[:n, 0] = X[n:, 1] = 1.0
        N = np.diag(np.concatenate([eS ** 2, eL ** 2]))
        N[np.arange(n), n + np.arange(n)] = N[n + np.arange(n), np.arange(n)] = rho * eS * eL
        mu_s, S_ss = driver.condition(tS)
    else:
        y, X, N = fL, np.ones((n, 1)), np.diag(eL ** 2)
    sd_L = float(np.std(fL))

    def build(p, mu_L, S_LL, S_sL):
        aL, jL = np.exp(p[0]), np.exp(p[1])
        if have_s:
            aS = np.exp(p[2]) if free_s else 1.0
            jS = np.exp(p[3]) if free_s else drw["jitter"]
            mean = np.concatenate([aS * mu_s, aL * mu_L])
            C = np.block([[aS ** 2 * S_ss, aS * aL * S_sL], [aS * aL * S_sL.T, aL ** 2 * S_LL]]) + N
            C[np.diag_indices(2 * n)] += np.concatenate([np.full(n, jS ** 2), np.full(n, jL ** 2)])
        else:
            mean = aL * mu_L
            C = aL ** 2 * S_LL + N + np.eye(n) * jL ** 2
        return mean, C

    lags = np.arange(lag_range[0], lag_range[1] + dt / 2, dt)
    ll = np.full(lags.size, -np.inf)
    a0 = np.log(sd_L / np.sqrt(drw["var"]))
    p0 = [a0, np.log(0.1 * sd_L)] + ([np.log(np.std(fS) / np.sqrt(drw["var"])), np.log(0.05 * np.std(fS))]
                                     if have_s and free_s else [])
    bounds = [(a0 - 5, a0 + 3), (np.log(sd_L) - 9, np.log(sd_L))]
    if have_s and free_s:
        bounds += [(p0[2] - 5, p0[2] + 3), (np.log(np.std(fS)) - 9, np.log(np.std(fS)))]
    # pass 1: the profile likelihood on the grid, warm-started from the previous lag
    order = np.argsort(np.abs(lags))
    starts = {}
    for i in order:
        tau = lags[i]
        q = (tL[:, None] - tau - u[None, :]).ravel()
        if have_s:
            mu, S = driver.condition(np.concatenate([tS, q]))
            mu_L, S_LL, S_sL = M @ mu[n:], M @ S[n:, n:] @ M.T, S[:n, n:] @ M.T
        else:
            mu, S = driver.condition(q)
            mu_L, S_LL, S_sL = M @ mu, M @ S @ M.T, None

        def nll(p):
            mean, C = build(p, mu_L, S_LL, S_sL)
            return -_gls(y - mean, X, C)[0]

        near = [starts[k] for k in (i - 1, i + 1) if k in starts]
        x0 = near[0] if near else p0
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200})
        starts[i] = res.x
        ll[i] = -res.fun
    weight = overlap_weight(tC if continuum is not None else tS, tL, lags) if alias else np.ones_like(lags)
    good = np.isfinite(ll)
    post = np.zeros_like(ll)
    post[good] = np.exp(ll[good] - ll[good].max()) * weight[good]
    if post.sum() <= 0:
        post[good] = np.exp(ll[good] - ll[good].max())
    bounds_pk = primary_peak(lags, post, smooth=2 * dt)
    inpk = (lags >= bounds_pk[0]) & (lags <= bounds_pk[1])
    rng = np.random.default_rng(seed)
    pk = post * inpk
    samples = rng.choice(lags, size=n_samples, p=pk / pk.sum()) + rng.uniform(-dt / 2, dt / 2, n_samples)
    lo, med, hi = np.percentile(samples, [15.865, 50, 84.135])
    return {"method": "DRW (correlated errors)" if have_s else "DRW", "tau_cen": float(med),
            "tau_err_lo": float(med - lo), "tau_err_hi": float(hi - med), "tau_peak": float(lags[np.argmax(post)]),
            "peak_bounds": bounds_pk, "frac_rej": float(1 - pk.sum() / post.sum()), "lags": lags,
            "loglike": ll, "posterior": post / post.sum(), "samples": samples, "n_mc": n_samples,
            "drw": drw, "r_max": np.nan}


# ----------------------------------- spectro-temporal (component) reverberation
def _component_quantities(me, line, components, feii_band, continuum_at):
    """Labels and func(epoch, full) of the quantities: continuum, broad components of the line, FeII."""
    first = me.epochs[0]
    model = first.model
    slices = dict(zip([c.name for c in model.components], model._slices()))
    if continuum_at is None:
        continuum_at = next((getattr(c, "pivot", None) for c in model.components
                             if c.kind == "continuum" and getattr(c, "pivot", None)), 5100.0)
    broad = [c for c in first._line_components() if c.role == "broad" and line in c.line_names]
    if components is not None:
        broad = [c for c in broad if c.name in components]
    if not broad:
        raise ValueError(f"no broad component of {line} in the model")
    feii = [c for c in model.components if c.name.startswith("feii")] if feii_band else []
    grid = np.linspace(feii_band[0], feii_band[1], 400) if feii else None
    labels = [f"continuum {continuum_at:g}"] + [f"{c.name} {line}" for c in broad] + \
             ([f"FeII {feii_band[0]:g}-{feii_band[1]:g}"] if feii else [])
    kinds = ["continuum"] + ["line"] * len(broad) + ["feii"] * bool(feii)
    ctx = first.spectrum.context

    def func(r, full):
        out = [r._agn_continuum(continuum_at, full)[0]]
        for c in broad:
            out.append(c.line_fluxes(full[slices[c.name]])[c.line_names.index(line)])
        if feii:
            f = sum(c.evaluate(grid, full[slices[c.name]], ctx)[0] for c in feii)
            out.append(float(np.sum(f) * (grid[1] - grid[0])))
        return np.array(out)

    return labels, kinds, func


def component_reverberation(me, line="Hb", continuum=None, components=None, feii_band=(4434.0, 4684.0),
                            continuum_at=None, lag_range=None, dt=1.0, n_pass=3, alias=True, seed=0):
    """
    Spectro-temporal reverberation: the lag and the response of every spectral
    component at once, from the multi-epoch fit with its full error structure.

    Each epoch gives a vector of component fluxes (the AGN continuum, each
    broad component of ``line``, FeII in ``feii_band``) with their joint
    covariance over all epochs: within an epoch the decomposition and
    calibration errors (the components trade flux with each other), between
    epochs the errors of shared parameters. Every component is modelled as

        F_c(t) = m_c + a_c x(t - tau_c) + intrinsic noise (variance j_c^2)

    with x(t) the continuum variation, a damped random walk conditioned on the
    photometric ``continuum`` (or fitted to the spectroscopic one). The
    response a_c may be zero: the data decide which components reverberate
    (with which lag) and which vary only through the decomposition. The lags
    come from the profile likelihood of each tau_c (the other lags at their
    best values, iterated ``n_pass`` times), weighted for the overlap of the
    light curves. Also returned: the component light curves with the
    decomposition noise removed (the conditional mean given the model).

    :param continuum: (mjd, flux, error) of a photometric continuum, or None.
    :param components: broad components to use (default: all with the line).
    :return: :class:`ComponentReverberation`.
    """
    labels, kinds, func = _component_quantities(me, line, components, feii_band, continuum_at)
    vals, cov_dec = me.quantity_covariance(func)                       # (n, m), epoch-major
    t = np.array([r.spectrum.mjd for r in me.epochs], float)
    n, m = vals.shape
    # component-major ordering
    perm = (np.arange(n)[None, :] * m + np.arange(m)[:, None]).ravel()
    Y = vals.T.ravel()
    Cd = cov_dec[np.ix_(perm, perm)]
    Cd = 0.5 * (Cd + Cd.T)
    ok = np.isfinite(Y) & np.isfinite(np.diag(Cd))
    good_epoch = ok.reshape(m, n).all(axis=0)
    sel = np.tile(good_epoch, m)
    Y, Cd, t = Y[sel], Cd[np.ix_(sel, sel)], t[good_epoch]
    n = t.size
    # scale every component to unit standard deviation (numerics)
    sd = np.array([np.std(Y[c * n:(c + 1) * n]) or 1.0 for c in range(m)])
    Ssc = np.repeat(sd, n)
    Y, Cd = Y / Ssc, Cd / np.outer(Ssc, Ssc)
    z = me.epochs[0].spectrum.z or 0.0
    if continuum is not None:
        tC, fC, eC = (np.asarray(a, float) for a in continuum)
        okc = np.isfinite(tC) & np.isfinite(fC) & np.isfinite(eC)
        tC, fC, eC = tC[okc], fC[okc], eC[okc]
        if lag_range is None:
            lag_range = (-0.25 * (t.max() - t.min()), 0.25 * (t.max() - t.min()))
        pad = max(abs(lag_range[0]), abs(lag_range[1]))
        keep = (tC > t.min() - pad - 200) & (tC < t.max() + pad + 200)
        tC, fC, eC = tC[keep], fC[keep], eC[keep]
        drw = fit_drw(tC, fC, eC)
        driver = _Driver(drw, tC, fC, eC)
        t_ref = tC
    else:
        drw = fit_drw(t, Y[:n], np.sqrt(np.diag(Cd)[:n]))
        driver = _Driver(drw)
        t_ref = t
        if lag_range is None:
            lag_range = (-0.25 * (t.max() - t.min()), 0.25 * (t.max() - t.min()))
    lags = np.arange(lag_range[0], lag_range[1] + dt / 2, dt)
    tau = np.zeros(m)
    resp = [c for c in range(m) if kinds[c] != "continuum"]

    def cond(tau):
        q = np.concatenate([t - tau[c] for c in range(m)])
        return driver.condition(q)

    X = np.zeros((m * n, m))
    for c in range(m):
        X[c * n:(c + 1) * n, c] = 1.0

    def gls(mu, S, a, j):
        """Log-likelihood with the responses a and noise j, the means profiled; returns (ll, beta, (cf, resid))."""
        from scipy.linalg import cho_factor, cho_solve
        A = np.repeat(a, n)
        C = A[:, None] * S * A[None, :] + Cd
        C[np.diag_indices_from(C)] += np.repeat(j ** 2, n) + 1e-10
        y = Y - A * mu
        try:
            cf = cho_factor(C, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return -np.inf, None, None, None
        CiX = cho_solve(cf, X, check_finite=False)
        F = X.T @ CiX
        beta = np.linalg.solve(F, CiX.T @ y)
        r = y - X @ beta
        ll = -0.5 * (r @ cho_solve(cf, r, check_finite=False) + 2 * np.sum(np.log(np.diag(cf[0])))
                     + np.linalg.slogdet(F)[1])
        return float(ll), beta, (cf, r)

    def nll_noise(p, mu, S):
        return -gls(mu, S, p[:m], np.exp(p[m:]))[0]

    # starting responses: regression on the conditional mean at zero lag
    from scipy.optimize import minimize
    mu, S = cond(tau)
    a = np.ones(m)
    j = np.full(m, 0.3)
    for it in range(n_pass):
        # responses and intrinsic noise at the current lags
        x0 = np.concatenate([a, np.log(j)])
        bnd = [(-10, 10)] * m + [(np.log(1e-3), np.log(3.0))] * m
        res = minimize(nll_noise, x0, args=(mu, S), method="L-BFGS-B", bounds=bnd, options={"maxiter": 300})
        a, j = res.x[:m], np.exp(res.x[m:])
        a = np.where(np.abs(a) < 1e-3, 1e-3, a)
        # profile likelihood of each lag, the others fixed
        profiles = {}
        for c in resp:
            ll = np.full(lags.size, -np.inf)
            for i, lag in enumerate(lags):
                tt = tau.copy()
                tt[c] = lag
                mu_i, S_i = cond(tt)
                ll[i] = gls(mu_i, S_i, a, j)[0]
            w = overlap_weight(t_ref, t, lags) if alias else np.ones_like(lags)
            post = np.zeros_like(ll)
            fin = np.isfinite(ll)
            post[fin] = np.exp(ll[fin] - ll[fin].max()) * w[fin]
            profiles[c] = (ll, post)
            tau[c] = lags[np.argmax(post)]
        mu, S = cond(tau)
    # final responses (in units of each component's std per unit continuum variation)
    res = minimize(nll_noise, np.concatenate([a, np.log(j)]), args=(mu, S), method="L-BFGS-B",
                   bounds=[(-10, 10)] * m + [(np.log(1e-3), np.log(3.0))] * m)
    a, j = res.x[:m], np.exp(res.x[m:])
    ll0, beta, (cf, resid) = gls(mu, S, a, j)
    # response significance: likelihood ratio against a_c = 0 (the noise refitted)
    rows = []
    rng = np.random.default_rng(seed)
    var_x = drw["var"]
    for c in range(m):
        if kinds[c] == "continuum":
            lag_c = (0.0, 0.0, 0.0, (np.nan, np.nan))
            dll = np.nan
        else:
            ll, post = profiles[c]
            pk = primary_peak(lags, post, smooth=2 * dt)
            inpk = (lags >= pk[0]) & (lags <= pk[1])
            pp = post * inpk
            smp = rng.choice(lags, size=4000, p=pp / pp.sum()) + rng.uniform(-dt / 2, dt / 2, 4000)
            lo, med, hi = np.percentile(smp, [15.865, 50, 84.135])
            lag_c = (med, med - lo, hi - med, pk)

            def nll0(p, c=c):
                aa = a.copy(); aa[c] = 0.0
                jj = j.copy(); jj[c] = np.exp(p[0])
                return -gls(mu, S, aa, jj)[0]
            r0 = minimize(nll0, [np.log(max(j[c], 0.05))], method="L-BFGS-B", bounds=[(np.log(1e-3), np.log(3.0))])
            dll = ll0 + r0.fun
        frac = a[c] ** 2 * var_x / (a[c] ** 2 * var_x + j[c] ** 2)
        rows.append((labels[c], kinds[c], lag_c[0] / (1 + z), lag_c[1] / (1 + z), lag_c[2] / (1 + z), lag_c[0],
                     a[c] * sd[c] * np.sqrt(var_x), frac, dll, np.sqrt(max(2 * dll, 0)) if np.isfinite(dll) else np.nan))
    table = pd.DataFrame(rows, columns=["component", "kind", "lag_rest", "err_lo", "err_hi", "lag_obs",
                                        "response", "frac_variance_reverberating", "delta_loglike",
                                        "significance"])
    # light curves: observed and with the decomposition noise removed
    from scipy.linalg import cho_solve
    A = np.repeat(a, n)
    mean = X @ beta + A * mu
    signal = A[:, None] * S * A[None, :] + np.diag(np.repeat(j ** 2, n))
    clean = mean + signal @ cho_solve(cf, resid)
    lc = pd.DataFrame({"mjd": t})
    for c in range(m):
        sl = slice(c * n, (c + 1) * n)
        lc[labels[c]] = Y[sl] * sd[c]
        lc[labels[c] + "_err"] = np.sqrt(np.diag(Cd)[sl]) * sd[c]
        lc[labels[c] + "_clean"] = clean[sl] * sd[c]
    corr = Cd / np.sqrt(np.outer(np.diag(Cd), np.diag(Cd)))
    return ComponentReverberation(line, table, lc, {labels[c]: profiles[c] for c in resp}, lags, drw, z, corr,
                                  labels)


class ComponentReverberation:
    """Result of :func:`component_reverberation`: ``table``, ``light_curves``, ``plot()``."""

    def __init__(self, line, table, light_curves, profiles, lags, drw, z, corr, labels):
        self.line, self.table, self.light_curves = line, table, light_curves
        self.profiles, self.lags, self.drw, self.z = profiles, lags, drw, z
        self.error_correlation, self.labels = corr, labels

    def __repr__(self):
        with pd.option_context("display.float_format", "{:.3g}".format, "display.width", 160):
            return (f"Spectro-temporal reverberation of {self.line} (DRW tau_d = {self.drw['tau_d']:.0f} d)\n"
                    + self.table.to_string(index=False))

    def plot(self, fig=None, width="double", style=True):
        """Component light curves (observed and with the decomposition noise removed) and lag profiles."""
        from .plotting import _figsize, new_figure, paper_style
        lc = self.light_curves
        with paper_style(style):
            w, _ = _figsize(width, 1)
            m = len(self.labels)
            fig = new_figure(fig, (w, w * (0.22 * m + 0.35)))
            gs = fig.add_gridspec(m, 2, width_ratios=[2.6, 1], hspace=0.15, wspace=0.25)
            for c, lab in enumerate(self.labels):
                ax = fig.add_subplot(gs[c, 0])
                ax.errorbar(lc.mjd, lc[lab], lc[lab + "_err"], fmt="o", ms=2, lw=0.5, color="0.55")
                ax.plot(lc.mjd, lc[lab + "_clean"], ".-", ms=2, lw=0.7, color="C3")
                ax.set_ylabel(lab, fontsize=7)
                if c < m - 1:
                    ax.set_xticklabels([])
                if lab in self.profiles:
                    ax2 = fig.add_subplot(gs[c, 1])
                    post = self.profiles[lab][1]
                    ax2.fill_between(self.lags / (1 + self.z), post / post.max(), color="C3", alpha=0.4, lw=0)
                    row = self.table[self.table.component == lab].iloc[0]
                    ax2.set_title(f"{row.lag_rest:.1f} d, {row.significance:.1f} sigma", fontsize=7)
                    ax2.set_yticks([])
                    if c == m - 1:
                        ax2.set_xlabel("rest-frame lag (d)")
            ax.set_xlabel("MJD")
        return fig


# ---------------------------------------------------------- rms spectrum
def _broad_only(result, line):
    """Observed broad target plus the data residual, on the epoch's native grid."""
    from .plotting import _line_parts, _transmission
    s = result.spectrum
    broad = [c.name for c in result._line_components() if c.role == "broad" and line in c.line_names]
    if not broad:
        raise ValueError(f"no broad component of {line} in the model")
    parts = _line_parts(result, line, s.wave)
    target = sum(parts[name] for name in broad) * _transmission(result, s.wave)
    return s.flux - result.model_flux() + target


def _rms_resample(grid, wave, flux, err, good):
    """Interpolate between adjacent fitted pixels, leaving masked and inter-segment gaps empty."""
    from .wavelengths import _sampling_gaps
    f = np.interp(grid, wave, np.where(good, flux, np.nan), left=np.nan, right=np.nan)
    e = np.interp(grid, wave, np.where(good, err, np.nan), left=np.nan, right=np.nan)
    gaps, _, _ = _sampling_gaps(np.diff(wave))
    for i in gaps:
        between = (grid > wave[i]) & (grid < wave[i+1])
        f[between] = e[between] = np.nan
    return f, e


def rms_spectrum(epochs, line, window=None, n_boot=200, seed=0):
    """
    Mean and noise-corrected rms spectrum of the broad ``line`` over the epochs
    (list of FitResult), with FWHM and line dispersion of the rms profile and
    bootstrap errors. Other fitted transitions are subtracted on each epoch's
    native pixel grid, including their instrumental and transmission effects.
    Fitted wavelength offsets are removed before stacking. The first covered epoch
    with a measurable width sets the default window and wavelength grid. With an
    explicit window, the first covered epoch sets the grid. Missing pixels remain
    missing, and the rms needs two epochs at a pixel. The wavelengths are in the
    reference epoch's medium and the widths include instrumental broadening.
    Returns a dict.
    """
    from .components import _in_medium
    if window is not None and (not np.isfinite(window) or window <= 0):
        raise ValueError("window must be a finite positive half-width in km/s")
    ref = None
    covered = []
    for r in epochs:
        broad = [c for c in r._line_components() if c.role == "broad" and line in c.line_names]
        if not broad or not np.any(r.mask & r.spectrum.good):
            continue
        names = [c.name for c in broad]
        if not np.isfinite(r._line_flux_func(line, names)(r.full)):
            continue
        covered.append(r)
        if ref is None:
            width = window
            if width is None:
                stats, *_ = r._line_measurement(line, names)
                fwhm = stats(r.full)[1]
                if not np.isfinite(fwhm) or fwhm <= 0:
                    continue
                width = float(np.clip(2.0 * fwhm, 3000, 15000))
            rest = broad[0]._wave[broad[0].line_names.index(line)]
            lam0 = _in_medium(rest, r.spectrum.context)
            wave = r.spectrum.wave / (1 + getattr(r, "offset", 0.0) / C_KMS)
            inside = np.abs(wave / lam0 - 1) * C_KMS <= width
            if np.any(inside & r.mask & r.spectrum.good):
                ref, grid = r, wave[inside]
    if len(covered) < 2:
        raise ValueError(f"rms_spectrum needs at least two fitted epochs covering broad {line}")
    if ref is None:
        raise ValueError(f"no covered epoch supplies a usable window for broad {line}; "
                         "specify a window in km/s containing fitted pixels")
    F, E = [], []
    for r in covered:
        # The joint fit stores calibrated data/errors, its effective resolution,
        # and the offset in the fitted shifts. Subtract on that grid, then align
        # in target-line velocity, preserving integrated flux when resampling F_lambda.
        s = r.spectrum
        factor = (_in_medium(rest, s.context) / lam0) * (1 + getattr(r, "offset", 0.0) / C_KMS)
        w, f, e = s.wave / factor, _broad_only(r, line) * factor, s.err * factor
        good = r.mask & s.good & np.isfinite(f)
        f, e = _rms_resample(grid, w, f, e, good)
        if np.isfinite(f).any():
            F.append(f)
            E.append(e)
    F, E = np.array(F), np.array(E)
    if len(F) < 2 or not np.any(np.isfinite(F).sum(axis=0) >= 2):
        raise ValueError(f"rms_spectrum needs overlapping fitted pixels in at least two epochs for broad {line}")
    v = (grid / lam0 - 1) * C_KMS

    def stats(rows):
        f, e = F[rows], E[rows]
        n = np.isfinite(f).sum(axis=0)
        mean = np.nansum(f, axis=0) / np.where(n > 0, n, np.nan)
        var = (np.nansum((f - mean) ** 2, axis=0) - np.nansum(e ** 2, axis=0)) / np.where(n >= 2, n, np.nan)
        rms = np.sqrt(np.clip(var, 0, None))
        return mean, rms

    def widths(prof):
        p = np.clip(np.nan_to_num(prof), 0, None)
        if p.sum() <= 0:
            return np.nan, np.nan
        c = np.sum(p * v) / p.sum()
        sigma = float(np.sqrt(np.sum(p * (v - c) ** 2) / p.sum()))
        half = p.max() / 2
        above = np.nonzero(p >= half)[0]
        fwhm = float(v[above[-1]] - v[above[0]]) if above.size else np.nan
        return fwhm, sigma

    rows = np.arange(len(F))
    mean, rms = stats(rows)
    fwhm, sigma = widths(rms)
    mfwhm, msigma = widths(mean)
    rng = np.random.default_rng(seed)
    def bootstrap_errors(index, count):
        samples = np.array([widths(stats(rng.integers(0, len(rows), len(rows)))[index])
                            for _ in range(count)]).reshape(-1, 2)
        return [float(np.std(col[np.isfinite(col)])) if np.isfinite(col).any() else np.nan
                for col in samples.T]

    efwhm, esigma = bootstrap_errors(1, n_boot)
    emfwhm, emsigma = bootstrap_errors(0, min(n_boot, 100))
    return {"wave": grid, "velocity": v, "mean": mean, "rms": rms,
            "fwhm_rms": fwhm, "fwhm_rms_err": efwhm,
            "sigma_rms": sigma, "sigma_rms_err": esigma,
            "fwhm_mean": mfwhm, "fwhm_mean_err": emfwhm,
            "sigma_mean": msigma, "sigma_mean_err": emsigma, "line": line}


# ---------------------------------------------------------------- result
def lag_significance(lag):
    """
    How clearly a lag differs from zero: (p, sigma). ``p`` is the fraction of the lag distribution
    (all Monte Carlo or posterior samples, aliases included) on the side of zero where the lag is,
    ``sigma`` the one-sided Gaussian equivalent (2 sigma: 97.7% of the distribution on that side).
    When every sample is on that side, sigma is the lower limit the number of samples allows (its
    value is then marked by ``p`` = 1). (nan, nan) without samples.
    """
    from scipy import stats
    s = np.asarray(lag.get("samples_all", lag.get("samples", [])), float)
    s = s[np.isfinite(s)]
    tau = lag.get("tau_cen", np.nan)
    if s.size < 10 or not np.isfinite(tau):
        return np.nan, np.nan
    p = float(np.mean(s > 0) if tau >= 0 else np.mean(s < 0))
    p_eff = min(p, 1 - 1.0 / s.size)
    return p, float(stats.norm.ppf(p_eff)) if p_eff > 0 else -np.inf


class Reverberation:
    """Lag, rms widths and reverberation mass of one broad line (see the module documentation)."""

    def __init__(self, line, lag, rms, continuum_label, z=None, f_virial=4.47, lc=None, methods=None):
        self.line, self.lag, self.rms, self.continuum_label = line, lag, rms, continuum_label
        self.z, self.f_virial, self.lc = z, f_virial, lc
        self.methods = methods or {"lag": lag}
        tau_obs = lag["tau_cen"]
        self.tau_rest = tau_obs / (1 + z) if z else tau_obs
        sc = 1 / (1 + z) if z else 1.0
        self.tau_rest_err = (lag["tau_err_lo"] * sc, lag["tau_err_hi"] * sc)
        self.warnings = []
        lags = {k: v for k, v in self.methods.items() if np.isfinite(v["tau_cen"])}
        for k, v in lags.items():
            if v is lag:
                continue
            d = v["tau_cen"] - lag["tau_cen"]
            e = np.hypot(lag["tau_err_hi" if d > 0 else "tau_err_lo"], v["tau_err_lo" if d > 0 else "tau_err_hi"])
            if abs(d) > 2 * e:
                self.warnings.append(f"the {v.get('method', k)} lag ({v['tau_cen']:.1f} d) differs from the "
                                     f"{lag.get('method', 'adopted')} lag ({lag['tau_cen']:.1f} d) by more than 2 sigma")
        p, sig = lag_significance(lag)
        if np.isfinite(sig) and sig < 2:
            self.warnings.append(f"the lag is not significant: {100 * p:.0f}% of its distribution is on the "
                                 f"{'positive' if lag['tau_cen'] >= 0 else 'negative'} side of zero (about {sig:.1f} sigma)")
        rmax = lag.get("r_max", np.nan)
        if np.isfinite(rmax) and rmax < 0.5:
            self.warnings.append(f"the light curves are weakly correlated (r_max = {rmax:.2f}): the lag is not reliable")
        for k, v in lags.items():
            if v.get("frac_rej", 0) > 0.3:
                self.warnings.append(f"{v.get('method', k)}: {100 * v['frac_rej']:.0f}% of the lag distribution "
                                     "is outside the primary peak (aliases)")
        if self.warnings:
            import warnings
            for w in self.warnings:
                warnings.warn(w)

    def lag_table(self):
        """Lag of every method (observed and rest frame) with the primary-peak bounds."""
        sc = 1 / (1 + self.z) if self.z else 1.0
        rows = [(v.get("method", k), v["tau_cen"] * sc, v["tau_err_lo"] * sc, v["tau_err_hi"] * sc,
                 v["tau_cen"], v.get("r_max", np.nan), v.get("frac_rej", np.nan), *lag_significance(v), v is self.lag)
                for k, v in self.methods.items()]
        return pd.DataFrame(rows, columns=["method", "lag_rest", "err_lo", "err_hi", "lag_obs", "r_max",
                                           "frac_outside_peak", "p_side", "significance", "adopted"])

    @property
    def log_mbh(self):
        """log M_BH/M_sun = log(f c tau sigma^2 / G) and its error (lag and sigma; not the scatter of f)."""
        tau, s = self.tau_rest, self.rms["sigma_rms"]
        if not (np.isfinite(tau) and tau > 0 and np.isfinite(s)):
            return np.nan, np.nan
        m = self.f_virial * tau * LIGHT_DAY * (s * 1e3) ** 2 / G_MSUN
        e_tau = 0.5 * (self.tau_rest_err[0] + self.tau_rest_err[1]) / tau
        e_s = 2 * self.rms["sigma_rms_err"] / s
        return float(np.log10(m)), float(np.hypot(e_tau, e_s) / np.log(10))

    def table(self):
        lm, le = self.log_mbh
        name = self.lag.get("method", "")
        sc = 1 / (1 + self.z) if self.z else 1.0
        other = [(f"lag (rest, {v.get('method', k)})", v["tau_cen"] * sc,
                  f"-{v['tau_err_lo'] * sc:.2g} +{v['tau_err_hi'] * sc:.2g}", "days")
                 for k, v in self.methods.items() if v is not self.lag]
        rows = [("lag (rest frame)", self.tau_rest, f"-{self.tau_rest_err[0]:.2g} +{self.tau_rest_err[1]:.2g}", "days"),
                ("lag (observed)", self.lag["tau_cen"],
                 f"-{self.lag['tau_err_lo']:.2g} +{self.lag['tau_err_hi']:.2g}", "days"),
                *([("lag method", np.nan, name, "")] if name else []),
                *([("lag significance", lag_significance(self.lag)[1],
                    f"{100 * lag_significance(self.lag)[0]:.0f}% on its side of 0", "sigma")]
                  if np.isfinite(lag_significance(self.lag)[1]) else []),
                *other,
                *([("ICCF r_max", self.methods["iccf"]["r_max"], "", "")] if "iccf" in self.methods else
                  [("r_max", self.lag["r_max"], "", "")] if np.isfinite(self.lag.get("r_max", np.nan)) else []),
                (f"sigma_line (rms {self.line})", self.rms["sigma_rms"], f"{self.rms['sigma_rms_err']:.2g}", "km/s"),
                (f"FWHM (rms {self.line})", self.rms["fwhm_rms"], f"{self.rms['fwhm_rms_err']:.2g}", "km/s"),
                (f"sigma_line (mean {self.line})", self.rms["sigma_mean"], f"{self.rms['sigma_mean_err']:.2g}", "km/s"),
                ("log M_BH (f = %.2f)" % self.f_virial, lm, f"{le:.2g}", "dex")]
        return pd.DataFrame(rows, columns=["quantity", "value", "error", "unit"])

    def __repr__(self):
        with pd.option_context("display.float_format", "{:.4g}".format, "display.width", 120):
            return (f"Reverberation of {self.line} against {self.continuum_label}\n"
                    + self.table().to_string(index=False)
                    + "".join(f"\n  warning: {w}" for w in self.warnings))

    def plot(self, fig=None, width="double", style=True):
        """Light curves, CCF with the lag distribution, and mean/rms profiles."""
        from .plotting import COLORS, _figsize, new_figure, paper_style, pretty_line
        with paper_style(style):
            w, _ = _figsize(width, 1)
            fig = new_figure(fig, (w, w * 0.62))
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.45, wspace=0.28)
            ax_lc = fig.add_subplot(gs[0, :])
            ax_cc = fig.add_subplot(gs[1, 0])
            ax_rms = fig.add_subplot(gs[1, 1])
            if self.lc is not None:
                (t1, f1, e1), (t2, f2, e2) = self.lc
                n1 = (f1 - np.nanmean(f1)) / np.nanstd(f1)
                n2 = (f2 - np.nanmean(f2)) / np.nanstd(f2)
                ax_lc.errorbar(t1, n1, e1 / np.nanstd(f1), fmt="o", ms=2.5, color="0.3", lw=0.6,
                               label=self.continuum_label)
                ax_lc.errorbar(t2, n2 + 3, e2 / np.nanstd(f2), fmt="o", ms=2.5, color=COLORS["broad"], lw=0.6,
                               label=f"broad {pretty_line(self.line)} (+3)")
                ax_lc.set_xlabel("MJD")
                ax_lc.set_ylabel("normalised flux")
                ax_lc.legend(loc="upper left", fontsize=7, frameon=False, ncol=2)
            colors = {"ICCF": "0.25", "von Neumann": "#4c9a2a", "ZDCF": "#7b3fd0", "DRW": COLORS["model"],
                      "DRW (correlated errors)": COLORS["model"]}
            if "r" in self.methods.get("iccf", {}):
                ic = self.methods["iccf"]
                ax_cc.plot(ic["lags"], ic["r"], color="0.2", lw=0.9)
                ax_cc.axhline(0.8 * ic["r_max"], color="0.6", lw=0.6, ls=":")
                ax_cc.set_ylabel("ICCF r")
            if "zdcf" in self.methods and "zdcf" in self.methods["zdcf"]:
                z = self.methods["zdcf"]["zdcf"]
                ax_cc.errorbar(z.tau, z.dcf, yerr=[z["-err(dcf)"], z["+err(dcf)"]], fmt="o", ms=2.2, lw=0.5,
                               color=colors["ZDCF"], label="ZDCF")
                ax_cc.set_ylabel("ICCF r, ZDCF" if "r" in self.methods.get("iccf", {}) else "ZDCF")
            elif "r" not in self.methods.get("iccf", {}):
                ax_cc.set_yticks([])
            ax2 = ax_cc.twinx()
            ax2.set_yticks([])
            for k, v in self.methods.items():
                name = v.get("method", k)
                col = colors.get(name, COLORS["broad"])
                if "posterior" in v:
                    post = v["posterior"]
                    ax2.fill_between(v["lags"], post / post.max(), color=col, alpha=0.25, lw=0, label=name)
                elif len(v.get("samples", [])):
                    h, e = np.histogram(v["samples"], bins=40)
                    ax2.stairs(h / h.max(), e, color=col, lw=1.0, label=name)
                ax_cc.axvline(v["tau_cen"], color=col, lw=1.4 if v is self.lag else 0.8,
                              ls="-" if v is self.lag else "--")
            ax2.set_ylim(0, 1.15)
            ax2.legend(fontsize=6.5, frameon=False, loc="upper right")
            ax_cc.set_xlabel("lag (days, observed)")
            ax_cc.set_title(f"lag = {self.lag['tau_cen']:.1f} $_{{-{self.lag['tau_err_lo']:.1f}}}"
                            f"^{{+{self.lag['tau_err_hi']:.1f}}}$ d ({self.lag.get('method', '')})", fontsize=9)
            v = self.rms["velocity"]
            ax_rms.plot(v, self.rms["mean"], color="0.3", lw=0.9, label="mean")
            ax_rms.plot(v, self.rms["rms"] * np.nanmax(self.rms["mean"]) / max(np.nanmax(self.rms["rms"]), 1e-30),
                        color=COLORS["broad"], lw=1.0, label="rms (scaled)")
            ax_rms.axhline(0, color="0.6", lw=0.5)
            ax_rms.set_xlabel("velocity (km$\\,$s$^{-1}$)")
            ax_rms.set_ylabel("broad " + pretty_line(self.line))
            ax_rms.set_title(f"$\\sigma_{{\\rm line}}$(rms) = {self.rms['sigma_rms']:.0f} $\\pm$ "
                             f"{self.rms['sigma_rms_err']:.0f} km/s", fontsize=9)
            ax_rms.legend(fontsize=7, frameon=False)
        return fig
