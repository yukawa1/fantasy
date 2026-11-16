"""
Finding the redshift of a spectrum that comes without one.

For every trial redshift the spectrum is fitted with a linear combination of
template spectra and a smooth polynomial, and chi2 is recorded: the minimum of
chi2(z) is the redshift, as in the SDSS and DESI pipelines (Bolton et al. 2012).
The whole spectrum takes part (breaks, absorption, all the lines at once), not a
few peaks. The templates are the galaxy and quasar eigenspectra of Yip et al.
(2004) that FANTASY ships::

    zf = fa.find_redshift(spec)        # spec in the observed frame
    zf.z, zf.error, zf.kind            # redshift, its error, 'galaxy' or 'quasar'
    zf.secure                          # is the best redshift clearly better than the next one?
    zf.candidates                      # the best few, with the chi2 gap and the lines they would show
    zf.plot()
    spec = spec.to_rest_frame(zf.z)

A wrong redshift is nearly always a confident mistake, one line taken for
another, so the alternatives are kept: ``candidates`` lists the best minima of
both template sets, and ``secure`` is False when the runner-up is close.

What it cannot do: identify a single line in a narrow range (the candidates are
then tied), featureless spectra (BL Lac objects), galaxies above z ~ 1 in an
optical spectrum (the galaxy templates start at 3450 A), and objects the
eigenspectra do not span (strong BALs, stars, supernovae). Without a real error
spectrum the chi2 gap means less.
"""
import functools
import os

import numpy as np
import pandas as pd
from astropy.io import fits

from .lines import DATA
from .wavelengths import air_to_vac

__all__ = ("find_redshift", "RedshiftResult", "MARK_LINES")

C_KMS = 299792.458

# prominent lines (vacuum, A) to mark at a candidate redshift; 'a': usually in absorption
MARK_LINES = (("Lya", 1215.67, "e"), ("NV", 1240.81, "e"), ("SiIV", 1399.8, "e"), ("CIV", 1549.48, "e"),
              ("CIII]", 1908.73, "e"), ("MgII", 2799.12, "e"), ("[OII]", 3728.48, "e"), ("CaII K", 3934.78, "a"),
              ("CaII H", 3969.59, "a"), ("Hd", 4102.89, "e"), ("Hg", 4341.68, "e"), ("Hb", 4862.68, "e"),
              ("[OIII]4959", 4960.30, "e"), ("[OIII]5007", 5008.24, "e"), ("Mg b", 5176.7, "a"), ("Na D", 5895.6, "a"),
              ("Ha", 6564.61, "e"), ("[NII]", 6585.27, "e"), ("[SII]", 6725.0, "e"))


@functools.lru_cache(maxsize=None)
def _eigenspectra(kind):
    """(vacuum wavelength, eigenspectra) of the galaxy or quasar templates, each scaled to unit rms."""
    name = "gal_eigenspec_Yip2004.fits" if kind == "galaxy" else "qso_eigenspec_Yip2004_global.fits"
    with fits.open(os.path.join(DATA, name)) as hdul:
        d = hdul[1].data
        wave = np.asarray(d["wave"], float).ravel()
        pca = np.asarray(d["pca"], float)
        pca = pca.reshape(pca.shape[-2], pca.shape[-1])
    pca = pca / np.sqrt(np.mean(pca ** 2, axis=1, keepdims=True))
    if np.mean(pca[0]) < 0:                     # the first eigenspectrum is the mean spectrum: positive
        pca[0] = -pca[0]
    if kind == "quasar":
        # blueward of the Lyman limit a quasar has (almost) no flux: the templates continue as zero down
        # to 300 A. A trial redshift that puts the blue part of the spectrum there must then explain the
        # flux it has, instead of leaving those pixels out (which lets a fit of half the spectrum win).
        step = wave[1] / wave[0]
        n = int(np.log(wave[0] / 300.0) / np.log(step))
        blue = wave[0] / step ** np.arange(n, 0, -1)
        wave = np.concatenate([blue, wave])
        pca = np.concatenate([np.zeros((pca.shape[0], n)), pca], axis=1)
    return wave, pca


def _rebin(x, y, e, n):
    """Average n neighbouring pixels (inverse-variance weights); for the coarse scan."""
    if n <= 1:
        return x, y, e
    m = (x.size // n) * n
    w = 1.0 / e[:m].reshape(-1, n) ** 2
    sw = w.sum(axis=1)
    return ((x[:m].reshape(-1, n) * w).sum(axis=1) / sw, (y[:m].reshape(-1, n) * w).sum(axis=1) / sw,
            1.0 / np.sqrt(sw))


class _Scan:
    """chi2 of the best linear fit of templates + polynomial to one spectrum, as a function of redshift."""

    def __init__(self, x, y, e, kind, n_templates, n_poly, min_overlap):
        self.x, self.y, self.w = x, y, 1.0 / e
        self.kind, self.min_overlap = kind, min_overlap
        wave, pca = _eigenspectra(kind)
        self.twave, self.T = wave, pca[:n_templates]
        u = 2 * (np.log(x) - np.log(x[0])) / (np.log(x[-1]) - np.log(x[0])) - 1
        self.poly = np.stack([u ** k for k in range(n_poly)], axis=1)       # a smooth additive term
        self.n = x.size

    def z_limits(self):
        """Redshifts at which the templates cover at least ``min_overlap`` of the spectrum."""
        lo, hi = self.x[0], self.x[-1]
        # rest coverage of the data [lo, hi] / (1 + z) against the templates [a, b], in log wavelength
        a, b = self.twave[0], self.twave[-1]
        span = np.log(hi / lo)
        z_hi = hi / a * np.exp(-self.min_overlap * span) - 1      # the blue end of the templates cuts in
        z_lo = lo / b * np.exp(self.min_overlap * span) - 1       # the red end of the templates cuts in
        return max(z_lo, -0.005), min(z_hi, 7.0)

    def chi2(self, z):
        """(chi2 scaled to all pixels, coefficients, pixels used) at redshift z; inf if the overlap is too small."""
        rest = self.x / (1 + z)
        use = (rest >= self.twave[0]) & (rest <= self.twave[-1])
        n = int(use.sum())
        if n < self.min_overlap * self.n or n < 50:
            return np.inf, None, n
        A = np.column_stack([np.interp(rest[use], self.twave, t) for t in self.T] + [self.poly[use]])
        Aw, yw = A * self.w[use, None], self.y[use] * self.w[use]
        G, b = Aw.T @ Aw, Aw.T @ yw
        d = np.sqrt(np.clip(np.diag(G), 1e-300, None))
        try:
            coef = np.linalg.solve(G / np.outer(d, d) + 1e-10 * np.eye(d.size), b / d) / d
        except np.linalg.LinAlgError:
            return np.inf, None, n
        # a physical solution: the mean spectrum enters with a positive weight, and the templates
        # (without the polynomial) are not negative over much of the range. Without this a fit of
        # noise or of half the spectrum can win at a wrong redshift.
        nt = len(self.T)
        if coef[0] <= 0 or np.mean(A[:, :nt] @ coef[:nt] < 0) > 0.2:
            return np.inf, None, n
        chi2 = float(yw @ yw - b @ coef)
        dof = max(n - A.shape[1], 1)
        return chi2 / dof * self.n, coef, n            # comparable between redshifts with different overlap

    def model(self, z, x=None):
        """The best-fit template model at redshift z on wavelengths x (NaN outside the templates)."""
        _, coef, _ = self.chi2(z)
        xs = self.x if x is None else np.asarray(x, float)
        rest = xs / (1 + z)
        ok = (rest >= self.twave[0]) & (rest <= self.twave[-1])
        out = np.full(xs.shape, np.nan)
        if coef is None:
            return out
        u = 2 * (np.log(xs) - np.log(self.x[0])) / (np.log(self.x[-1]) - np.log(self.x[0])) - 1
        A = np.column_stack([np.interp(rest, self.twave, t) for t in self.T] + [u ** k for k in range(self.poly.shape[1])])
        out[ok] = (A @ coef)[ok]
        return out


def _minima(z, c, n, min_sep_kms=1500.0):
    """The n lowest local minima of c(z), at least min_sep apart; [(z, chi2), ...]."""
    ok = np.isfinite(c)
    idx = [i for i in range(1, len(c) - 1) if ok[i] and c[i] <= (c[i - 1] if ok[i - 1] else np.inf)
           and c[i] <= (c[i + 1] if ok[i + 1] else np.inf)]
    out = []
    for i in sorted(idx, key=lambda i: c[i]):
        if all(abs(np.log((1 + z[i]) / (1 + zj))) * C_KMS > min_sep_kms for zj, _ in out):
            out.append((float(z[i]), float(c[i])))
        if len(out) >= n:
            break
    return out


def _refine(scan, z0, half_kms=700.0, step_kms=25.0):
    """Minimum of chi2 near z0 on a fine grid: (z, error, chi2). The error is where chi2 rises by the
    reduced chi2 (or 1), from a parabola through the minimum."""
    v = np.arange(-half_kms, half_kms + step_kms, step_kms)
    zz = (1 + z0) * np.exp(v / C_KMS) - 1
    c = np.array([scan.chi2(z)[0] for z in zz])
    if not np.isfinite(c).any():
        return z0, np.nan, np.inf
    k = int(np.nanargmin(np.where(np.isfinite(c), c, np.inf)))
    z, best = float(zz[k]), float(c[k])
    err = np.nan
    if 0 < k < len(zz) - 1 and np.isfinite(c[k - 1]) and np.isfinite(c[k + 1]):
        a, b, cc = c[k - 1], c[k], c[k + 1]
        curv = (a - 2 * b + cc) / step_kms ** 2
        if curv > 0:
            dv = 0.5 * (a - cc) / (a - 2 * b + cc) * step_kms
            z = float((1 + zz[k]) * np.exp(dv / C_KMS) - 1)
            best = float(b - 0.25 * (a - cc) * dv / step_kms)
            scale = max(best / scan.n, 1.0)                       # errors as if reduced chi2 were 1
            err = float(np.sqrt(2.0 * scale / curv) * (1 + z) / C_KMS)
    return z, err, best


QUASAR_FLOOR_KMS = 75.0


def _emission_peaks(x, y, e, nsigma=6.0):
    """Observed wavelengths of the emission peaks: runs of pixels above a running median by nsigma in total."""
    from scipy.ndimage import median_filter
    size = max(51, int(x.size / 25) | 1)
    dev = (y - median_filter(y, size=size, mode="nearest")) / e
    out, k = [], 0
    while k < x.size:
        if dev[k] < 2.0:
            k += 1
            continue
        j = k
        while j < x.size and dev[j] >= 2.0:
            j += 1
        if j - k >= 2 and np.sqrt(np.sum(dev[k:j] ** 2)) >= nsigma and dev[k:j].max() >= 3.5:
            out.append(float(x[k + int(np.argmax(dev[k:j]))]))
        k = j
    return np.array(out)


def _peaks_matched(peaks, z, tol_kms=1000.0):
    lines = np.array([w for _, w, t in MARK_LINES if t == "e"]) * (1 + z)
    return int(sum(np.min(np.abs(np.log(p / lines))) * C_KMS < tol_kms for p in peaks)) if len(peaks) else 0


def _narrow_line_redshift(spectrum, z):
    """Redshift from the narrow H-beta and [OIII] lines near z: (z, error), or None if [OIII]5007 is not in
    the range or not detected above 5 sigma. A broad H-beta takes the broad line of a quasar."""
    import warnings
    from .components import Lines, PowerLaw
    from .fitting import fit
    from .model import Model
    spec = spectrum.to_rest_frame(z)
    good = spec.good
    if not good.any() or spec.wave[good][0] > 4780 or spec.wave[good][-1] < 5080:
        return None
    model = Model(PowerLaw(pivot=4950.0),
                  Lines("narrow", ["Hb", "[OIII]4959", "[OIII]5007"], fwhm=(300, 50, 1200), shift=(0, -900, 900),
                        ratios={"[OIII]4959": ("[OIII]5007", 1 / 2.98)}),
                  Lines("broad", ["Hb"], fwhm=(4000, 1200, 12000), shift=(0, -2000, 2000)), wave_range=(4700, 5100))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = fit(spec, model, verbose=False)
    except Exception:                              # noqa: BLE001  (too few pixels, a gap: keep the template redshift)
        return None
    t = r.params.set_index("parameter")
    flux, ferr = t.loc["narrow.[OIII]5007.flux", ["value", "error"]]
    shift, serr = t.loc["narrow.shift", ["value", "error"]]
    if not (np.isfinite(ferr) and ferr > 0 and flux > 5 * ferr and np.isfinite(serr) and abs(shift) < 880):
        return None
    return (1 + z) * (1 + shift / C_KMS) - 1, (1 + z) * float(serr) / C_KMS


def _catalog_check(spectrum, table):
    """([(catalog, z)], index of the candidate a catalog redshift agrees with or None)."""
    from .catalogs import _catalog_redshifts, lookup_catalogs
    if spectrum.ra is None or spectrum.dec is None:
        raise ValueError("catalog=True needs the position of the spectrum: fa.read(path, ra=..., dec=...)")
    row = lookup_catalogs((float(spectrum.ra), float(spectrum.dec))).iloc[0].to_dict()
    found = _catalog_redshifts(row)
    match = None
    for k, zc in enumerate(table.z):
        if any(abs(np.log((1 + zk) / (1 + zc))) * C_KMS < 1500.0 for _, zk in found):
            match = k
            break
    return found, match


class RedshiftResult:
    """
    Result of :func:`find_redshift`: ``z``, ``error``, ``kind`` ('galaxy' or 'quasar'), ``secure``,
    ``dchi2`` (chi2 of the runner-up minus that of the best, for all pixels), ``method`` ('templates',
    or 'narrow lines' when the redshift was refined on [OIII]; ``template_z`` then keeps the first),
    ``candidates`` (a table: z, error, kind, chi2, dchi2, rchi2, the emission peaks that sit on a known
    line and the lines in the range), ``curves`` ({kind: (z, chi2)}), ``catalog`` (with ``catalog=True``)
    and :meth:`plot`. ``lines(z)`` lists the prominent lines that fall in the spectrum at a redshift.
    The error of a quasar-template redshift includes 75 km/s for the shifts of the broad lines.
    """

    def __init__(self, spectrum, candidates, curves, scans, n_pixels, secure, min_dchi2):
        self.spectrum, self.candidates, self.curves = spectrum, candidates, curves
        self._scans, self.n_pixels, self.secure, self.min_dchi2 = scans, n_pixels, secure, min_dchi2
        best = candidates.iloc[0]
        self.z, self.error, self.kind = float(best.z), float(best.error), str(best.kind)
        self.dchi2 = float(candidates.dchi2.iloc[1]) if len(candidates) > 1 else np.inf
        self.method, self.template_z, self.template_error = "templates", self.z, self.error
        self.peaks, self.catalog, self.catalog_match = np.zeros(0), None, None

    def lines(self, z=None):
        """[(name, observed wavelength, 'e' or 'a')] of the prominent lines inside the spectrum at redshift z."""
        z = self.z if z is None else z
        lo, hi = self._scans[self.kind].x[0], self._scans[self.kind].x[-1]
        return [(n, w * (1 + z), t) for n, w, t in MARK_LINES if lo <= w * (1 + z) <= hi]

    def model(self, k=0, wave=None):
        """The template model of candidate k on ``wave`` (default: the pixels used)."""
        c = self.candidates.iloc[k]
        return self._scans[c.kind].model(float(c.z), wave)

    def summary(self):
        b = self.candidates.iloc[0]
        out = [f"Redshift {self.z:.5f} +- {self.error:.5f} ({self.kind} templates, reduced chi2 {b.rchi2:.2f}): "
               + ("secure" if self.secure else "NOT secure")]
        if len(self.candidates) > 1:
            r = self.candidates.iloc[1]
            out.append(f"  next best: z = {r.z:.5f} ({r.kind}), chi2 higher by {r.dchi2:.0f} "
                       f"({r.dchi2 / self.n_pixels:.3f} per pixel; secure above {self.min_dchi2:.3f})")
        if self.method == "narrow lines":
            out.append(f"  from the narrow H-beta and [OIII] lines (templates: {self.template_z:.5f})")
        if len(self.peaks):
            out.append(f"  emission peaks on a known line at this redshift: {self.candidates.peaks.iloc[0]}")
        if self.catalog is not None:
            listed = ", ".join(f"{name} {z:.5f}" for name, z in self.catalog) or "no redshift"
            verdict = ("agrees with the best redshift" if self.catalog_match == 0 else
                       f"agrees with candidate {self.catalog_match + 1}, NOT with the best one" if self.catalog_match
                       else "agrees with no candidate" if self.catalog else "")
            out.append(f"  catalogs: {listed}" + (f" ({verdict})" if verdict else ""))
        out.append("  lines in the range: " + ", ".join(n for n, _, _ in self.lines()))
        return "\n".join(out)

    def __repr__(self):
        return self.summary()

    def plot(self, fig=None, n=3):
        """chi2 against redshift for both template sets, and the spectrum with the template model and the
        lines of the best ``n`` candidates marked. Returns the figure."""
        import matplotlib.pyplot as plt
        fig = fig or plt.figure(figsize=(7.1, 5.0))
        ax1, ax2 = fig.subplots(2, 1, gridspec_kw={"height_ratios": [1, 1.4]})
        colors = {"galaxy": "#1b7f5c", "quasar": "#3b6fb6"}
        for kind, (z, c) in self.curves.items():
            ok = np.isfinite(c)
            ax1.plot(np.log10(1 + z[ok]), c[ok] / self.n_pixels, lw=0.9, color=colors[kind], label=f"{kind} templates")
        for k, row in self.candidates.head(n).iterrows():
            ax1.axvline(np.log10(1 + row.z), color="#c0392b" if k == 0 else "#999999", lw=0.8, ls="-" if k == 0 else ":")
        ax1.set_xlabel("log (1 + z)")
        ax1.set_ylabel(r"$\chi^2$ per pixel")
        ax1.legend(frameon=False, fontsize=8)
        s = self._scans[self.kind]
        ax2.step(s.x, s.y, where="mid", lw=0.6, color="#444444")
        ax2.plot(s.x, self.model(0), lw=1.0, color="#c0392b")
        top = np.nanpercentile(s.y, 99.5)
        for name, w, _ in self.lines():
            ax2.axvline(w, color="#c0392b", lw=0.5, alpha=0.5)
            ax2.text(w, top, name, rotation=90, fontsize=6.5, va="top", ha="right", color="#c0392b")
        ax2.set_ylim(min(0, np.nanpercentile(s.y, 0.5)), top * 1.15)
        ax2.set_xlabel("observed wavelength (A)")
        ax2.set_ylabel("flux")
        ax2.set_title(f"z = {self.z:.5f} ({self.kind})" + ("" if self.secure else "   not secure"), fontsize=9, loc="left")
        fig.tight_layout()
        return fig


def find_redshift(spectrum, z_range=None, kinds=("galaxy", "quasar"), n_templates=(5, 4), n_poly=3,
                  n_candidates=5, min_overlap=0.75, min_dchi2=0.2, step_kms=120.0, refine=True, catalog=False):
    """
    Redshift of a spectrum from a chi2 scan with galaxy and quasar templates; see the module docstring.

    :param spectrum: a Spectrum in the observed frame (its ``z``, if any, is not used).
    :param z_range: (min, max) redshift to search (default: all that the templates cover: about
        0 to 1 for galaxies and 0 to 7 for quasars in an optical spectrum).
    :param kinds: template sets to try.
    :param n_templates: number of eigenspectra of the galaxy and of the quasar set.
    :param n_poly: terms of the additive polynomial (3: constant, slope, curvature).
    :param n_candidates: how many minima to keep.
    :param min_overlap: fraction of the spectrum that the templates must cover at a redshift.
    :param min_dchi2: the best redshift is ``secure`` if the next one (more than 1500 km/s away) has a
        chi2 higher by more than this per pixel. On 540 SDSS spectra of nine classes (z = 0.01-2.5,
        median S/N 14) 97% of the best redshifts were right (within 1000 km/s of the pipeline's); with
        0.2, 96% were called secure and 99.8% of those were right, and of the rest 43%. The SDSS pipeline
        warns below 0.01, where here most answers were wrong.
    :param step_kms: step of the search; the minima are then refined on a 25 km/s grid.
    :param refine: for a quasar-template redshift with [OIII]5007 in the range, fit the narrow H-beta and
        [OIII] lines there and take the redshift from them (``method`` = 'narrow lines') if [OIII] is
        detected above 5 sigma. Quasar templates follow the broad lines, which are shifted from the
        systemic velocity by ~100 km/s; the narrow lines much less. The error is then the statistical
        one of the line centre: against SDSS pipeline redshifts such values scatter by ~50 km/s, which is
        the disagreement between methods, not noise. Galaxy-template redshifts follow the stars, agree
        with the pipeline to 4-10 km/s and are not changed.
    :param catalog: also look the position up in SIMBAD, NED and Milliquas (``fa.lookup_catalogs``; sends
        the coordinates, needs the spectrum's ra and dec) and report the redshifts found there in
        ``catalog`` and whether one agrees with a candidate; it does not change the choice.
    :return: :class:`RedshiftResult`
    """
    if getattr(spectrum, "rest_frame", False):
        raise ValueError("the spectrum is in the rest frame: find_redshift needs the observed spectrum "
                         "(read the file again without to_rest_frame)")
    good = spectrum.good
    if good.sum() < 100:
        raise ValueError("fewer than 100 usable pixels")
    x = spectrum.wave[good] if spectrum.medium == "vacuum" else air_to_vac(spectrum.wave[good])
    y, e = spectrum.flux[good], spectrum.err[good]
    # the coarse scan runs on pixels as wide as the step; the minima are refined on all of them
    pix = np.median(np.diff(np.log(x))) * C_KMS
    xb, yb, eb = _rebin(x, y, e, max(1, int(step_kms / max(pix, 1e-6) / 1.5)))
    counts = dict(zip(("galaxy", "quasar"), n_templates))
    curves, coarse, scans, found = {}, {}, {}, []
    for kind in kinds:
        if kind not in counts:
            raise ValueError("kinds: 'galaxy' and/or 'quasar'")
        coarse[kind] = _Scan(xb, yb, eb, kind, counts[kind], n_poly, min_overlap)
        scans[kind] = _Scan(x, y, e, kind, counts[kind], n_poly, min_overlap)
        lo, hi = coarse[kind].z_limits()
        if z_range is not None:
            lo, hi = max(lo, z_range[0]), min(hi, z_range[1])
        if hi <= lo:
            continue
        z = np.exp(np.arange(np.log(1 + lo), np.log(1 + hi), step_kms / C_KMS)) - 1
        c = np.array([coarse[kind].chi2(v)[0] for v in z]) * (x.size / xb.size)      # per pixel of the full spectrum
        curves[kind] = (z, c)
        found += [(kind, zc) for zc, _ in _minima(z, c, n_candidates)]
    if not found:
        raise ValueError("the templates do not cover this spectrum at any redshift in the range")
    rows = []
    for kind, zc in found:
        z, err, c = _refine(scans[kind], zc)
        if np.isfinite(c):
            if kind == "quasar" and np.isfinite(err):
                # the broad lines the quasar templates follow are shifted against each other and against
                # the systemic velocity: on SDSS quasars the scatter against the pipeline is 75 km/s
                err = float(np.hypot(err, QUASAR_FLOOR_KMS * (1 + z) / C_KMS))
            rows.append({"z": z, "error": err, "kind": kind, "chi2": c})
    rows.sort(key=lambda r: r["chi2"])
    kept = []
    for r in rows:                                   # one entry per redshift: the better template set
        if all(abs(np.log((1 + r["z"]) / (1 + k["z"]))) * C_KMS > 1500.0 for k in kept):
            kept.append(r)
    kept = kept[:n_candidates]
    best = kept[0]["chi2"]
    for r in kept:
        r["dchi2"] = r["chi2"] - best
        r["rchi2"] = r["chi2"] / x.size
        r["lines"] = ", ".join(n for n, w, _ in MARK_LINES if x[0] <= w * (1 + r["z"]) <= x[-1])
    # an independent look: how many of the emission peaks of the spectrum sit on a known line
    peaks = _emission_peaks(x, y, e)
    for r in kept:
        r["peaks"] = f"{_peaks_matched(peaks, r['z'])} of {len(peaks)}" if len(peaks) else ""
    table = pd.DataFrame(kept, columns=["z", "error", "kind", "chi2", "dchi2", "rchi2", "peaks", "lines"])
    secure = len(kept) == 1 or kept[1]["dchi2"] / x.size > min_dchi2
    result = RedshiftResult(spectrum, table, curves, scans, int(x.size), bool(secure), float(min_dchi2))
    result.peaks = peaks
    # a galaxy-template redshift follows the stars and is good to a few km/s; [OIII] can sit tens of
    # km/s off them (outflows), so only quasar-template redshifts are refined on the narrow lines
    if refine and result.kind == "quasar":
        fine = _narrow_line_redshift(spectrum, result.z)
        if fine is not None:
            result.template_z, result.template_error = result.z, result.error
            result.z, result.error, result.method = fine[0], fine[1], "narrow lines"
            table.loc[0, ["z", "error"]] = result.z, result.error
    if catalog:
        result.catalog, result.catalog_match = _catalog_check(spectrum, table)
    return result
