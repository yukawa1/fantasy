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
  minus the continuum, FeII, host and narrow lines of that epoch's model),
  noise-corrected rms (Park et al. 2012, ApJS 203, 6); widths from the
  bootstrap over epochs.
* ICCF: Gaskell & Peterson (1987), White & Peterson (1994), both light curves
  interpolated; lag = centroid of r >= 0.8 r_max; errors from flux
  randomisation and random subset selection (Peterson et al. 1998, 2004),
  median and 68% interval of the centroid distribution.
* M_BH = f c tau sigma_line(rms)^2 / G with f = 4.47 (Woo et al. 2015); the
  ~0.3 dex scatter of f is not in the error.
"""
import numpy as np
import pandas as pd

__all__ = ("iccf", "ccf_lag", "rms_spectrum", "Reverberation")

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


def ccf_lag(t1, f1, e1, t2, f2, e2, lag_range=None, dt=None, n_mc=1000, frac=0.8, seed=0):
    """
    Lag of light curve 2 behind 1 with the ICCF and FR/RSS errors.

    :param lag_range: (min, max) lag to search, days (default: +-1/2 of the overlap).
    :param dt: lag step (default: a quarter of the median sampling, >= 0.25 d).
    :return: dict with tau_cen and its -/+ errors, tau_peak, r_max, the CCF
        (lags, r) and the centroid samples.
    """
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
    if cents.size:
        lo, med, hi = np.percentile(cents, [15.865, 50, 84.135])
    else:
        lo = med = hi = np.nan
    return {"tau_cen": float(med), "tau_err_lo": float(med - lo), "tau_err_hi": float(hi - med),
            "tau_cen_ccf": tau_cen, "tau_peak": tau_peak, "r_max": rmax, "lags": lags, "r": r,
            "samples": cents, "n_mc": int(cents.size)}


# ---------------------------------------------------------- rms spectrum
def _broad_only(result, line):
    """Rest wavelength, broad-line-only data and error of one epoch."""
    s = result.spectrum
    parts = result.model_flux(components=True)
    other = sum(v for k, v in parts.items()
                if not any(c.name == k and getattr(c, "role", None) == "broad" for c in result.model.components))
    return s.wave, s.flux - other, s.err


def rms_spectrum(epochs, line, window=None, n_boot=200, seed=0):
    """
    Mean and noise-corrected rms spectrum of the broad ``line`` over the epochs
    (list of FitResult), with FWHM and line dispersion of the rms profile and
    bootstrap errors. Returns a dict.
    """
    from .lines import line_wavelength
    lam0 = line_wavelength(line)
    ref = epochs[0]
    broad = [c.name for c in ref._line_components() if c.role == "broad" and line in c.line_names]
    if not broad:
        raise ValueError(f"no broad component of {line} in the model")
    if window is None:
        fwhm = ref.measure(line, broad)["fwhm"]
        window = float(np.clip(2.0 * fwhm, 3000, 15000))
    w0 = ref.spectrum.wave
    grid = w0[np.abs(w0 / lam0 - 1) * C_KMS <= window]
    F, E = [], []
    for r in epochs:
        w, f, e = _broad_only(r, line)
        good = r.mask & np.isfinite(f) & np.isfinite(e)
        F.append(np.interp(grid, w[good], f[good], left=np.nan, right=np.nan))
        E.append(np.interp(grid, w[good], e[good], left=np.nan, right=np.nan))
    F, E = np.array(F), np.array(E)
    v = (grid / lam0 - 1) * C_KMS

    def stats(rows):
        f, e = F[rows], E[rows]
        mean = np.nanmean(f, axis=0)
        var = np.nanmean((f - mean) ** 2, axis=0) - np.nanmean(e ** 2, axis=0)   # noise-corrected
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

    rows = np.arange(len(epochs))
    mean, rms = stats(rows)
    fwhm, sigma = widths(rms)
    mfwhm, msigma = widths(mean)
    rng = np.random.default_rng(seed)
    boot = np.array([widths(stats(rng.integers(0, len(rows), len(rows)))[1]) for _ in range(n_boot)])
    bm = np.array([widths(stats(rng.integers(0, len(rows), len(rows)))[0]) for _ in range(min(n_boot, 100))])
    return {"wave": grid, "velocity": v, "mean": mean, "rms": rms,
            "fwhm_rms": fwhm, "fwhm_rms_err": float(np.nanstd(boot[:, 0])),
            "sigma_rms": sigma, "sigma_rms_err": float(np.nanstd(boot[:, 1])),
            "fwhm_mean": mfwhm, "fwhm_mean_err": float(np.nanstd(bm[:, 0])),
            "sigma_mean": msigma, "sigma_mean_err": float(np.nanstd(bm[:, 1])), "line": line}


# ---------------------------------------------------------------- result
class Reverberation:
    """Lag, rms widths and reverberation mass of one broad line (see the module documentation)."""

    def __init__(self, line, lag, rms, continuum_label, z=None, f_virial=4.47, lc=None):
        self.line, self.lag, self.rms, self.continuum_label = line, lag, rms, continuum_label
        self.z, self.f_virial, self.lc = z, f_virial, lc
        tau_obs = lag["tau_cen"]
        self.tau_rest = tau_obs / (1 + z) if z else tau_obs
        sc = 1 / (1 + z) if z else 1.0
        self.tau_rest_err = (lag["tau_err_lo"] * sc, lag["tau_err_hi"] * sc)

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
        rows = [("lag (rest frame)", self.tau_rest, f"-{self.tau_rest_err[0]:.2g} +{self.tau_rest_err[1]:.2g}", "days"),
                ("lag (observed)", self.lag["tau_cen"],
                 f"-{self.lag['tau_err_lo']:.2g} +{self.lag['tau_err_hi']:.2g}", "days"),
                ("r_max", self.lag["r_max"], "", ""),
                (f"sigma_line (rms {self.line})", self.rms["sigma_rms"], f"{self.rms['sigma_rms_err']:.2g}", "km/s"),
                (f"FWHM (rms {self.line})", self.rms["fwhm_rms"], f"{self.rms['fwhm_rms_err']:.2g}", "km/s"),
                (f"sigma_line (mean {self.line})", self.rms["sigma_mean"], f"{self.rms['sigma_mean_err']:.2g}", "km/s"),
                ("log M_BH (f = %.2f)" % self.f_virial, lm, f"{le:.2g}", "dex")]
        return pd.DataFrame(rows, columns=["quantity", "value", "error", "unit"])

    def __repr__(self):
        with pd.option_context("display.float_format", "{:.4g}".format, "display.width", 120):
            return (f"Reverberation of {self.line} against {self.continuum_label}: "
                    f"{self.lag['n_mc']} FR/RSS realisations\n" + self.table().to_string(index=False))

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
            lags, r = self.lag["lags"], self.lag["r"]
            ax_cc.plot(lags, r, color="0.2", lw=0.9)
            ax_cc.axvline(self.lag["tau_cen"], color=COLORS["model"], lw=0.9)
            ax_cc.axhline(0.8 * self.lag["r_max"], color="0.6", lw=0.6, ls=":")
            if self.lag["samples"].size:
                ax2 = ax_cc.twinx()
                ax2.hist(self.lag["samples"], bins=40, color=COLORS["broad"], alpha=0.3)
                ax2.set_yticks([])
            ax_cc.set_xlabel("lag (days, observed)")
            ax_cc.set_ylabel("ICCF r")
            ax_cc.set_title(f"lag = {self.lag['tau_cen']:.1f} $_{{-{self.lag['tau_err_lo']:.1f}}}"
                            f"^{{+{self.lag['tau_err_hi']:.1f}}}$ d", fontsize=9)
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
