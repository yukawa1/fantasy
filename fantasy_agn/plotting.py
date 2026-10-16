"""
Publication-quality plots of fits, spectra and light curves.

All figures are drawn in a journal style (serif STIX fonts, inward ticks on
all sides, colour-blind-safe colours) that is applied only to FANTASY's own
figures, never to the user's global matplotlib settings. Save them in any
format matplotlib supports, chosen by the file extension::

    result.savefig("fit.pdf")                    # also .eps, .svg, .png
    result.savefig("fit.eps", width="single")    # one journal column (3.5 in)
"""
import re
from contextlib import contextmanager

import matplotlib
import numpy as np
from matplotlib.ticker import MaxNLocator

__all__ = ("plot_result", "plot_light_curves", "plot_spectrum", "plot_baldwin", "fit_line",
           "new_figure", "paper_style", "pretty_line", "flux_label", "savefig")

# Journal look; fonttype 42 embeds TrueType fonts in PDF/EPS, as journals require.
PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}

# Okabe & Ito (2008) colour-blind-safe palette.
COLORS = {
    "data": "0.35",
    "balmer_cont": "#999933",
    "balmer_lines": "#999933",
    "model": "#D55E00",
    "continuum": "#000000",
    "host": "#8C6D31",
    "feii": "#CC79A7",
    "balmer": "#999933",
    "narrow": "#009E73",
    "broad": "#0072B2",
    "outflow": "#E69F00",
    "other": "#56B4E9",
}
WIDTHS = {"single": 3.5, "double": 7.1}

_GREEK = {"Lya": "Ly$\\alpha$", "Ha": "H$\\alpha$", "Hb": "H$\\beta$", "Hg": "H$\\gamma$",
          "Hd": "H$\\delta$", "He": "H$\\epsilon$"}


@contextmanager
def paper_style(enabled=True):
    """Temporarily apply the journal style (used around figure creation and saving)."""
    if not enabled:
        yield
        return
    import matplotlib
    with matplotlib.rc_context(PAPER_RC):
        yield


def pretty_line(name):
    """'[OIII]5007' -> '[O III] λ5007', 'Hb' -> 'Hβ', 'MgII2798' -> 'Mg II λ2798'."""
    if name in _GREEK:
        return _GREEK[name]
    m = re.match(r"^(\[)?([A-Z][a-z]?)([IVX]+)(\])?(\d+)?$", name)
    if not m:
        return name
    left, element, ion, right, wave = m.groups()
    text = f"{left or ''}{element}$\\,${ion}{right or ''}"
    return text + (f" $\\lambda${wave}" if wave else "")


def _unit_scale(unit):
    """'10^{-17}' or '2.5\\times10^{-18}' for a flux unit, '' if unknown."""
    if not unit or unit == 1:
        return ""
    exponent = int(np.floor(np.log10(unit) + 1e-9))
    if np.isclose(unit, 10.0 ** exponent, rtol=1e-6, atol=0):
        return f"10^{{{exponent}}}"
    return f"{unit / 10.0 ** exponent:g}\\times10^{{{exponent}}}"


def flux_label(spectrum, quantity="$F_\\lambda$", per_angstrom=True):
    """Axis label with the physical flux unit when it is known."""
    scale = _unit_scale(getattr(spectrum, "flux_unit", spectrum))
    if not scale:
        return f"{quantity} (arbitrary units)"
    unit = f"${scale}\\,\\mathrm{{erg\\,s^{{-1}}\\,cm^{{-2}}}}"
    unit += "\\,$\u00c5$^{-1}$" if per_angstrom else "$"
    return f"{quantity} ({unit})"


def _wave_label(spectrum):
    frame = "Rest-frame" if spectrum.rest_frame else "Observed"
    return f"{frame} wavelength (Å)"


def new_figure(fig=None, figsize=(7.1, 4.4)):
    """
    A figure to draw into: a pyplot figure by default (shown by notebooks), or
    the given matplotlib Figure (used by the GUI, which keeps its figures out
    of pyplot so that nothing else can close or redraw them meanwhile).
    """
    if fig is None:
        import matplotlib.pyplot as plt
        return plt.figure(figsize=figsize)
    if hasattr(fig, "set_size_inches"):  # subfigures take their parent's size
        fig.set_size_inches(*figsize)
    return fig


def _figsize(width, aspect):
    w = WIDTHS.get(width, width) if isinstance(width, str) else float(width)
    return (w, w * aspect)


# ------------------------------------------------------------------ groups
def _group_of(comp):
    if comp.kind == "host":
        return "host", "Host galaxy"
    if comp.kind == "continuum":
        if type(comp).__name__ == "BalmerContinuum":
            return "balmer_cont", "Balmer continuum"
        return "continuum", "Continuum"
    if comp.kind == "template":
        if type(comp).__name__ == "HighOrderBalmer":
            return "balmer_lines", "High-order Balmer lines"
        return "feii", "Fe$\\,$II"
    role = getattr(comp, "role", None)
    labels = {"narrow": "Narrow lines", "broad": "Broad lines", "outflow": "Outflow"}
    return (role, labels[role]) if role in labels else ("other", comp.name)


_ORDER = ["continuum", "host", "balmer_cont", "balmer_lines", "feii", "narrow", "broad",
          "outflow", "other"]
_LINESTYLES = ["-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1))]


def _curves(result, group):
    """
    List of (kind, label, flux, linestyle) to draw. With group=True components
    of the same kind are summed (e.g. all broad components); otherwise every
    component is drawn, in its kind's colour with its own line style.
    """
    parts = result.model_flux(components=True)
    out = {}
    count = {}
    for comp in result.model.components:
        if comp.name not in parts:      # e.g. an Extinction component
            continue
        kind, label = _group_of(comp)
        if group:
            key = kind
        else:
            key = comp.name
            if kind in ("narrow", "broad", "outflow", "other"):
                label = comp.name
        if key in out:
            out[key][2] = out[key][2] + parts[comp.name]
        else:
            n = count.get(kind, 0)
            count[kind] = n + 1
            ls = {"continuum": "--", "host": ":", "balmer_cont": "-."}.get(
                kind, _LINESTYLES[n % len(_LINESTYLES)])
            out[key] = [kind, label, parts[comp.name].copy(), ls]
    items = sorted(out.values(), key=lambda v: _ORDER.index(v[0]) if v[0] in _ORDER else 99)
    return [tuple(v) for v in items]


def _label_lines(ax, result, view, model, xlim, max_labels=12):
    """Annotate the strongest lines in view with their names."""
    from .components import Lines
    s = result.spectrum
    parts = result.model_flux(components=True)
    peaks = {}
    for comp in result.model.components:
        if not isinstance(comp, Lines):
            continue
        prof = parts[comp.name]
        for ln, lam in zip(comp.line_names, comp._wave):
            lam_obs = lam if s.medium == "vacuum" else lam / 1.00028
            if not (xlim[0] < lam_obs < xlim[1]):
                continue
            near = np.abs(s.wave - lam_obs) < 3
            if near.any():
                peaks[ln] = (lam_obs, max(peaks.get(ln, (0, 0))[1], float(prof[near].max())))
    if not peaks:
        return
    strongest = max(p for _, p in peaks.values())
    chosen = sorted(((p, ln, lam) for ln, (lam, p) in peaks.items() if p > 0.05 * strongest),
                    reverse=True)[:max_labels]
    span = xlim[1] - xlim[0]
    placed = []
    ytop = ax.get_ylim()[1]
    for p, ln, lam in chosen:
        if any(abs(lam - q) < 0.025 * span for q in placed):
            continue
        if not (xlim[0] + 0.03 * span < lam < xlim[1] - 0.03 * span):
            continue  # a label at the very edge would be clipped
        placed.append(lam)
        near = view & (np.abs(s.wave - lam) < 0.01 * span)
        y = np.nanmax(np.maximum(s.flux[near], model[near])) if near.any() else p
        y = min(y + 0.04 * ytop, 0.9 * ytop)
        ax.annotate(pretty_line(ln), (lam, y), xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, rotation=90,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4", shrinkA=0, shrinkB=0))


# -------------------------------------------------------------------- plots
def plot_result(result, ax=None, components=True, residuals=True, xlim=None, fig=None,
                width="double", group=True, label_lines=False, title=None, style=True):
    """
    Data, model, components and residuals.

    :param width: 'single' (3.5 in), 'double' (7.1 in) or a width in inches.
    :param group: sum components by role (one legend entry for all broad
        components, etc.); False draws every component separately.
    :param label_lines: annotate the strongest lines with their names.
    :param title: figure title (default none, as is usual in papers).
    :param style: apply the journal style.
    """
    with paper_style(style):
        s = result.spectrum
        sel = result.mask
        if xlim is None:
            xlim = (s.wave[sel].min(), s.wave[sel].max())
        view = (s.wave >= xlim[0]) & (s.wave <= xlim[1])
        narrow = _figsize(width, 1)[0] < 5
        aspect = (0.95 if narrow else 0.62) if residuals else (0.75 if narrow else 0.48)
        if ax is None:
            fig = new_figure(fig, _figsize(width, aspect))
            if residuals:
                ax, rax = fig.subplots(2, 1, sharex=True,
                                       gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0})
            else:
                ax, rax = fig.subplots(), None
        else:
            fig, rax = ax.figure, None

        wave = s.wave[view]
        model = result.model_flux()
        used = np.where(sel[view], s.flux[view], np.nan)
        ax.plot(wave, used, color=COLORS["data"], lw=0.6, drawstyle="steps-mid",
                label="Data", zorder=2)
        unused = view & ~sel
        if unused.any():
            # NaN where fitted, so that masked stretches are not joined across the gaps
            ax.plot(wave, np.where(unused[view], s.flux[view], np.nan), color="0.75", lw=0.5,
                    drawstyle="steps-mid", zorder=1, label="Not fitted")
        if components:
            for kind, label, flux, ls in _curves(result, group):
                if not np.any(np.abs(flux[view]) > 0):
                    continue  # e.g. Balmer continuum entirely outside the plotted range
                ax.plot(wave, flux[view], color=COLORS.get(kind, COLORS["other"]), lw=0.9,
                        ls=ls, label=label, zorder=3)
        ax.plot(wave, model[view], color=COLORS["model"], lw=1.1, label="Model", zorder=4)

        ax.set_xlim(*xlim)
        good = view & sel & np.isfinite(s.flux)
        lo = min(0.0, float(np.nanpercentile(s.flux[good], 0.5)))
        hi = float(np.nanpercentile(np.maximum(s.flux[good], model[good]), 99.9))
        ax.set_ylim(lo, hi * (1.5 if label_lines else 1.08))
        ylabel = flux_label(s)
        ax.set_ylabel(ylabel.replace(" (", "\n(", 1) if narrow else ylabel)
        handles, labels = ax.get_legend_handles_labels()
        # The legend sits above the axes; a title goes on top of it (as the
        # legend title) so the two cannot overlap.
        ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                  ncol=min(len(labels), 2 if narrow else 4),
                  columnspacing=1.2, borderaxespad=0.3, title=title or None,
                  title_fontsize=matplotlib.rcParams["axes.titlesize"])
        if label_lines:
            _label_lines(ax, result, view, model, xlim)

        if rax is not None:
            r = (s.flux - model) / s.err
            rr = np.where(sel[view], r[view], np.nan)
            rax.plot(wave, rr, color=COLORS["data"], lw=0.5, drawstyle="steps-mid")
            rax.axhline(0, color=COLORS["model"], lw=0.8)
            for v in (-3, 3):
                rax.axhline(v, color="0.5", lw=0.5, ls=":")
            lim = np.nanpercentile(np.abs(rr), 99.5) if np.isfinite(rr).any() else 5
            lim = max(4.0, 1.15 * lim)
            rax.set_ylim(-lim, lim)
            rax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper", symmetric=True))
            rax.set_ylabel("Resid. ($\\sigma$)")
            rax.set_xlabel(_wave_label(s))
            ax.tick_params(labelbottom=False)
            fig.align_ylabels([ax, rax])
        else:
            ax.set_xlabel(_wave_label(s))
    return fig


def plot_spectrum(spectrum, ax=None, fig=None, width="double", style=True, **kw):
    """The spectrum alone, in the same style; returns the axes."""
    with paper_style(style):
        if ax is None:
            ax = new_figure(fig, _figsize(width, 0.4)).subplots()
        ax.plot(spectrum.wave, spectrum.flux, color=COLORS["data"], lw=0.6,
                drawstyle="steps-mid", **kw)
        if spectrum.host is not None:
            ax.plot(spectrum.wave, spectrum.host, color=COLORS["host"], lw=0.9, ls=":",
                    label="Host galaxy")
            ax.legend(loc="upper right")
        ax.set_xlabel(_wave_label(spectrum))
        ax.set_ylabel(flux_label(spectrum))
        ax.set_xlim(spectrum.wave.min(), spectrum.wave.max())
    return ax


def plot_light_curves(table, columns, ax=None, fig=None, width="double", style=True,
                      flux_unit=None):
    """Light curves from MultiEpochResult.light_curve() output, one panel per column."""
    with paper_style(style):
        n = len(columns)
        if ax is None:
            w, _ = _figsize(width, 1)
            fig = new_figure(fig, (w, 1.5 * n + 0.6))
            axes = fig.subplots(n, 1, sharex=True, squeeze=False,
                                gridspec_kw={"hspace": 0})[:, 0]
        else:
            axes = np.atleast_1d(ax)
            fig = axes[0].figure
        t = table["mjd"] if table["mjd"].notna().all() else table["epoch"]
        for a, col in zip(axes, columns):
            a.errorbar(t, table[col], yerr=table.get(col + "_err"), fmt="o", ms=3.5,
                       color=COLORS["broad"], ecolor="0.4", elinewidth=0.7, capsize=0)
            a.set_ylabel(_lc_label(col, flux_unit), fontsize=8)
            a.margins(x=0.03, y=0.15)
            # stacked panels touch: drop the top tick label so it cannot hit the panel above
            a.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="upper"))
        axes[-1].set_xlabel("MJD" if t.name == "mjd" else "Epoch")
        fig.align_ylabels(list(axes))
    return fig


def _lc_label(col, flux_unit=None):
    m = re.match(r"^(.*)_(flux|fwhm)$", col)
    if m and m.group(2) == "fwhm":
        return f"{pretty_line(m.group(1))} FWHM\n(km$\\,$s$^{{-1}}$)"
    if m:
        return flux_label(flux_unit, f"Broad {pretty_line(m.group(1))}",
                          per_angstrom=False).replace(" (", "\n(", 1)
    m = re.match(r"^cont(\d+)$", col)
    if m:
        return flux_label(flux_unit, f"$F_{{{m.group(1)}}}$").replace(" (", "\n(", 1)
    return col


def fit_line(x, y, sx, sy, n_boot=1000, seed=0):
    """
    Fit y = a + b (x - x0) by maximum likelihood with errors in both
    coordinates and intrinsic scatter s: each point has variance
    sy^2 + b^2 sx^2 + s^2. x0 is the median of x (so a and b are nearly
    uncorrelated). Uncertainties come from bootstrap resampling of the points.
    Returns a dict with a, b, scatter, their errors, x0, and the Spearman rank
    correlation coefficient and p-value.
    """
    from scipy.optimize import minimize
    from scipy.stats import spearmanr

    x, y, sx, sy = (np.asarray(v, float) for v in (x, y, sx, sy))
    x0 = float(np.median(x))

    def solve(i):
        xi, yi, sxi, syi = x[i] - x0, y[i], sx[i], sy[i]

        def nll(p):
            a, b, log_s = p
            var = syi ** 2 + (b * sxi) ** 2 + np.exp(2 * log_s)
            return 0.5 * np.sum((yi - a - b * xi) ** 2 / var + np.log(var))

        b0 = np.polyfit(xi, yi, 1)[0] if np.ptp(xi) > 0 else 0.0
        start = [np.median(yi), b0, np.log(max(np.std(yi) * 0.5, 1e-3))]
        return minimize(nll, start, method="Nelder-Mead",
                        options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 4000}).x

    best = solve(np.arange(x.size))
    rng = np.random.default_rng(seed)
    boot = np.array([solve(rng.integers(0, x.size, x.size)) for _ in range(n_boot)])
    rho, pval = spearmanr(x, y)
    err = np.std(boot, axis=0)
    return {"a": best[0], "a_err": err[0], "b": best[1], "b_err": err[1],
            "scatter": float(np.exp(best[2])), "scatter_err": float(np.std(np.exp(boot[:, 2]))),
            "x0": x0, "spearman_rho": float(rho), "spearman_p": float(pval), "n": int(x.size),
            "boot": boot}


def plot_baldwin(table, line="Hb", continuum_at=5100, width="single", ax=None, fig=None,
                 color_by_time=True, fit=True, style=True, n_boot=1000):
    """
    Baldwin-effect plot: log EW of a broad line against log lambda L_lambda.

    :param table: light-curve table (MultiEpochResult.light_curve()) or any
        DataFrame with columns '<line>_ew', '<line>_ew_err', 'logL<continuum_at>'
        and 'logL<continuum_at>_err' (e.g. one row per object for the global
        Baldwin effect).
    :param color_by_time: colour points by MJD when an 'mjd' column exists.
    :param fit: draw the maximum-likelihood power law with its 68% band.
    :return: (figure, fit dict or None)
    """
    lc = f"logL{continuum_at:g}"
    t = table[np.isfinite(table[f"{line}_ew"]) & (table[f"{line}_ew"] > 0)
              & np.isfinite(table[lc])]
    x, sx = t[lc].to_numpy(float), t[f"{lc}_err"].to_numpy(float)
    ew, sew = t[f"{line}_ew"].to_numpy(float), t[f"{line}_ew_err"].to_numpy(float)
    y, sy = np.log10(ew), sew / (ew * np.log(10))
    with paper_style(style):
        if ax is None:
            fig = new_figure(fig, _figsize(width, 0.85))
            ax = fig.subplots()
        else:
            fig = ax.figure
        result = None
        if fit and len(x) >= 5:
            result = fit_line(x, y, sx, sy, n_boot=n_boot)
            xx = np.linspace(x.min() - 0.05 * np.ptp(x), x.max() + 0.05 * np.ptp(x), 200)
            boot_lines = (result["boot"][:, 0][:, None]
                          + result["boot"][:, 1][:, None] * (xx - result["x0"]))
            lo, hi = np.percentile(boot_lines, [15.87, 84.13], axis=0)
            ax.fill_between(xx, lo, hi, color="0.85", lw=0, zorder=0)
            ax.plot(xx, result["a"] + result["b"] * (xx - result["x0"]), color=COLORS["model"],
                    lw=1.2, zorder=1,
                    label=f"$\\beta = {result['b']:.2f} \\pm {result['b_err']:.2f}$")
        ax.errorbar(x, y, xerr=sx, yerr=sy, fmt="none", ecolor="0.6", elinewidth=0.6,
                    capsize=0, zorder=2)
        if color_by_time and "mjd" in t and t["mjd"].notna().all() and t["mjd"].nunique() > 1:
            sc = ax.scatter(x, y, c=t["mjd"], cmap="viridis", s=14, zorder=3, edgecolors="none")
            cb = fig.colorbar(sc, ax=ax, pad=0.02)
            cb.set_label("MJD")
            cb.ax.tick_params(which="both", direction="in")
        else:
            ax.scatter(x, y, color=COLORS["broad"], s=14, zorder=3, edgecolors="none")
        ax.set_xlabel(f"log $\\lambda L_{{{continuum_at:g}}}$ (erg$\\,$s$^{{-1}}$)")
        ax.set_ylabel(f"log EW({pretty_line(line)}) (\u00c5)")
        if result is not None:
            ax.legend(loc="upper right", title=f"$\\rho_s = {result['spearman_rho']:.2f}$, "
                      f"$p = {result['spearman_p']:.1g}$", title_fontsize=7)
    return fig, result


def savefig(fig, path, **kw):
    """Save in the journal style (embedded fonts, 300 dpi for raster formats)."""
    with paper_style(True):
        fig.savefig(path, **kw)
    return path
