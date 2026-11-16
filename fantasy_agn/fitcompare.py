"""
Compare two fits of the same spectrum: why does a result change when the fit looks the same?

Changing the continuum, the line profile or a tie and fitting again can move a result (a broad-line width) far more
than its error bar while the total model hardly changes: the components trade flux where they overlap (broad
Balmer wings, Fe II, the continuum). A residual plot of either fit alone gives no hint. ``compare_fits`` shows it::

    a = fa.fit(spec, model_power_law)
    b = fa.fit(spec, model_broken_power_law)
    cmp = fa.compare_fits(a, b, labels=("power law", "broken power law"))
    cmp.stats          # chi2 of both on the same pixels, free parameters, RMS of (B - A) / error
    cmp.table          # results of both fits, their difference and the difference in units of the errors
    cmp.warnings       # e.g. a result that moved by more than its error while the fits are equally good
    cmp.plot()         # both total fits, (B - A) / error, the change of each kind of component, both broad profiles

The broad profiles are drawn normalised, with their half-maximum crossings: a shoulder near half maximum makes the
FWHM jump between fits that differ little (``fwhm_fragile``); the line dispersion sigma is then more robust.
"""
import numpy as np
import pandas as pd

from .robustness import _value, default_quantities

__all__ = ("compare_fits", "FitComparison", "half_max_crossings", "fwhm_fragile")

PREFERRED = ("Hb", "Ha", "MgII2798", "CIV1549")
FRAGILE = 1.3        # FWHM at 45% of the peak over FWHM at 55%: 1.16 for a Gaussian, 1.22 for a Lorentzian


def _crossings(v, prof, level):
    """Outermost velocities where prof crosses level * peak (interpolated)."""
    peak = np.max(prof)
    if not peak > 0:
        return np.nan, np.nan
    half = level * peak
    above = np.nonzero(prof >= half)[0]
    i0, i1 = above[0], above[-1]
    vl = np.interp(half, [prof[i0 - 1], prof[i0]], [v[i0 - 1], v[i0]]) if i0 > 0 else v[0]
    vr = np.interp(half, [prof[i1 + 1], prof[i1]], [v[i1 + 1], v[i1]]) if i1 < v.size - 1 else v[-1]
    return float(vl), float(vr)


def half_max_crossings(v, prof):
    """(blue, red) velocities of the half-maximum crossings of a line profile on velocity grid v."""
    return _crossings(v, prof, 0.5)


def fwhm_fragile(v, prof):
    """
    (fragile, ratio): the width at 45% of the peak over the width at 55%. A ratio above 1.3 means a shoulder near
    half maximum, where a small change of the profile moves the FWHM a lot (a Gaussian gives 1.16).
    """
    a, b = _crossings(v, prof, 0.45), _crossings(v, prof, 0.55)
    ratio = (a[1] - a[0]) / (b[1] - b[0]) if b[1] > b[0] else np.nan
    return bool(ratio > FRAGILE), float(ratio)


def _main_line(result):
    broad = [c for c in result._line_components() if c.role == "broad"]
    return next((ln for ln in PREFERRED if any(ln in c.line_names for c in broad)), None)


def broad_profile(result, line):
    """(v, profile) of the broad components of ``line`` alone, on a velocity grid (km/s)."""
    names = [c.name for c in result._line_components() if c.role == "broad" and line in c.line_names]
    if not names:
        return None
    _, v, _, profile, _ = result._line_profile_func(line, names)
    return v, profile(result.full)


def _groups(result, x):
    """Model flux on x summed per kind of component (continuum, host, Fe II, broad, narrow, outflow ...)."""
    from .plotting import _group_of
    parts = result.model.evaluate(x, result.spectrum.context, result.full, components=True)
    kinds = {c.name: _group_of(c) for c in result.model.components}
    out = {}
    for name, f in parts.items():
        key, label = kinds.get(name, ("other", name))
        if key not in out:
            out[key] = [label, np.zeros_like(x)]
        out[key][1] = out[key][1] + f
    return out


class FitComparison:
    """Result of :func:`compare_fits`; see the module docstring."""

    def __init__(self, a, b, labels, line, stats, table, warnings, x):
        self.a, self.b, self.labels, self.line = a, b, labels, line
        self.stats, self.table, self.warnings, self._x = stats, table, warnings, x

    def summary(self):
        s = self.stats
        out = [f"{self.labels[0]} vs {self.labels[1]}: chi2 {s['chi2'][0]:.1f} vs {s['chi2'][1]:.1f} on the same "
               f"{s['n_pixels']} pixels (dchi2 = {s['dchi2']:+.1f}), free parameters {s['n_free'][0]} vs {s['n_free'][1]}, "
               f"total models differ by {s['rms_diff']:.3f} error (RMS)"]
        out += ["  " + w for w in self.warnings]
        return "\n".join(out)

    def __repr__(self):
        return self.summary()

    def plot(self, fig=None, window=None, colors=("#1f5f7a", "#c4502a")):
        """Both total fits and (B - A) / error, the change of each kind of component, and both broad profiles."""
        import matplotlib.pyplot as plt
        from .plotting import COLORS, pretty_line
        a, b, (la, lb) = self.a, self.b, self.labels
        x = self._x
        sel = np.ones_like(x, bool) if window is None else (x >= window[0]) & (x <= window[1])
        s = a.spectrum
        idx = np.searchsorted(s.wave, x)
        y, e = s.flux[idx], s.err[idx]
        fa_, fb_ = a.model_flux(x), b.model_flux(x)
        if fig is None:
            fig = plt.figure(figsize=(10, 5.2))
        gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 1, 1.8], width_ratios=[1.35, 1], hspace=0.38, wspace=0.22)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
        ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
        ax3 = fig.add_subplot(gs[:, 1])
        ax0.plot(x[sel], y[sel], color=COLORS["data"], lw=0.7, label="data")
        ax0.plot(x[sel], fa_[sel], color=colors[0], lw=1.1, label=la)
        ax0.plot(x[sel], fb_[sel], color=colors[1], lw=1.1, ls="--", label=lb)
        ax0.legend(frameon=False, fontsize=7, loc="upper right")
        ax0.set_ylabel("flux")
        ax0.set_title("both total fits", fontsize=8, loc="left")
        ax1.plot(x[sel], ((fb_ - fa_) / e)[sel], color="0.2", lw=0.8)
        ax1.axhline(0, color="0.6", lw=0.5)
        ax1.set_ylabel("difference\n/ error", fontsize=7)
        ax1.set_title(f"{lb} minus {la}, in units of the error", fontsize=7, loc="left")
        ga, gb = _groups(a, x), _groups(b, x)
        for key in sorted(set(ga) | set(gb), key=lambda k: list(COLORS).index(k) if k in COLORS else 99):
            lab = (ga.get(key) or gb.get(key))[0]
            d = (gb[key][1] if key in gb else 0) - (ga[key][1] if key in ga else 0)
            if np.max(np.abs(d[sel])) > 0.05 * np.median(e):
                ax2.plot(x[sel], d[sel], color=COLORS.get(key, "0.5"), lw=1.1, label=lab)
        ax2.plot(x[sel], (fb_ - fa_)[sel], color="k", ls="--", lw=0.9, label="total")
        ax2.axhline(0, color="0.6", lw=0.5)
        ax2.legend(frameon=False, fontsize=6, ncol=3)
        ax2.set_ylabel("change of each\ncomponent", fontsize=7)
        ax2.set_title(f"where the flux went: {lb} minus {la}", fontsize=7, loc="left")
        ax2.set_xlabel(r"rest wavelength ($\AA$)")
        for ax in (ax0, ax1):
            plt.setp(ax.get_xticklabels(), visible=False)
        if self.line:
            for r, c, ls, lab in ((a, colors[0], "-", la), (b, colors[1], "--", lb)):
                bp = broad_profile(r, self.line)
                if bp is None:
                    continue
                v, p = bp
                p = p / p.max()
                vl, vr = half_max_crossings(v, p)
                m = r.measure(self.line, [k.name for k in r._line_components() if k.role == "broad" and self.line in k.line_names])
                ax3.plot(v / 1e3, p, color=c, ls=ls, lw=1.3,
                         label=f"{lab}: FWHM {m['fwhm']:,.0f}, $\\sigma$ {m['sigma']:,.0f} km/s")
                ax3.plot([vl / 1e3, vr / 1e3], [0.5, 0.5], "o", color=c, ms=4)
            ax3.axhline(0.5, color="0.7", lw=0.5, ls=":")
            ax3.set_xlim(-20, 20)
            ax3.set_xlabel(r"velocity ($10^3$ km/s)")
            ax3.set_ylabel("fraction of the peak")
            ax3.set_title(f"broad {pretty_line(self.line)} alone (half-maximum crossings marked)", fontsize=8, loc="left")
            ax3.legend(frameon=False, fontsize=7, loc="upper right")
        return fig


def compare_fits(a, b, labels=("A", "B"), line=None, quantities=None):
    """
    Compare two fits of the same spectrum (e.g. before and after changing the continuum or the line profile).

    :param a, b: FitResults of the same spectrum.
    :param labels: names of the two fits.
    :param line: broad line whose profiles are compared (default: Hb, Ha, Mg II or C IV, whichever both have).
    :param quantities: {name: function(FitResult) -> (value, error)}; default: those of :func:`fa.robustness`.
    :return: :class:`FitComparison`
    """
    sa, sb = a.spectrum, b.spectrum
    if sa.wave.size != sb.wave.size or not np.allclose(sa.wave, sb.wave):
        raise ValueError("the two fits are not of the same spectrum (different wavelengths)")
    common = a.mask & b.mask
    if not common.any():
        raise ValueError("the two fits share no fitted pixels")
    x, y, e = sa.wave[common], sa.flux[common], sa.err[common]
    ma, mb = a.model_flux(x), b.model_flux(x)
    chi = [float(np.sum(((m - y) / e) ** 2)) for m in (ma, mb)]
    nfree = [sum(1 for p in r.model.params if p.free) for r in (a, b)]
    stats = {"n_pixels": int(common.sum()), "chi2": chi, "dchi2": chi[1] - chi[0], "n_free": nfree,
             "rchi2": [float(a.rchi2), float(b.rchi2)], "rms_diff": float(np.sqrt(np.mean(((mb - ma) / e) ** 2)))}
    if line is None:
        la, lb = _main_line(a), _main_line(b)
        line = la if la == lb else None

    qa = default_quantities(a) if quantities is None else dict(quantities)
    qb = default_quantities(b) if quantities is None else dict(quantities)
    rows = []
    for name in list(dict.fromkeys(list(qa) + list(qb))):
        vals = []
        for q, r in ((qa, a), (qb, b)):
            try:
                out = q[name](r) if name in q else (np.nan, np.nan)
                vals.append((_value(out), float(out[1]) if isinstance(out, (tuple, list)) else np.nan))
            except Exception:
                vals.append((np.nan, np.nan))
        (va, ea), (vb, eb) = vals
        comb = np.sqrt(np.nansum([ea ** 2, eb ** 2])) if np.isfinite(ea) or np.isfinite(eb) else np.nan
        rows.append({"quantity": name, labels[0]: va, labels[0] + " error": ea, labels[1]: vb, labels[1] + " error": eb,
                     "difference": vb - va, "relative": (vb - va) / va if va else np.nan,
                     "in errors": (vb - va) / comb if comb and np.isfinite(comb) else np.nan})
    table = pd.DataFrame(rows)

    warnings = []
    equal = abs(stats["dchi2"]) < max(4.0, 0.01 * chi[0]) and stats["rms_diff"] < 0.3
    for r in rows:
        if np.isfinite(r["in errors"]) and abs(r["in errors"]) > 1.5:
            warnings.append(f"{r['quantity']} moved by {r['difference']:+.4g} ({100 * r['relative']:+.0f}%), "
                            f"{abs(r['in errors']):.1f} times the combined error"
                            + (", although the two fits are equally good" if equal else ""))
    if line:
        for r, lab in ((a, labels[0]), (b, labels[1])):
            bp = broad_profile(r, line)
            if bp is not None:
                frag, ratio = fwhm_fragile(*bp)
                if frag:
                    warnings.append(f"{lab}: the broad {line} profile has a shoulder near half maximum (width at 45% / "
                                    f"55% of the peak = {ratio:.2f}); its FWHM is fragile, prefer sigma")
    return FitComparison(a, b, tuple(labels), line, stats, table, warnings, x)
