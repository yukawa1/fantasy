"""
Publication-quality plots of fits, spectra and light curves.

All figures are drawn in one style (serif STIX fonts, quiet grey axes with
inward ticks on all sides, no grid, colour-blind-safe colours) that is applied only to
FANTASY's own figures, never to the user's global matplotlib settings. Save them in any
format matplotlib supports, chosen by the file extension::

    result.savefig("fit.pdf")                    # also .eps, .svg, .png
    result.savefig("fit.eps", width="single")    # one journal column (3.5 in)
"""
import re
from contextlib import contextmanager

import matplotlib
import matplotlib.colors
import matplotlib.patches
import numpy as np
from matplotlib.ticker import MaxNLocator

C_KMS = 299792.458

__all__ = ("plot_result", "plot_profiles", "plot_correlations", "plot_relation", "plot_light_curves", "plot_spectrum", "plot_baldwin", "plot_main_sequence", "fit_line",
           "new_figure", "paper_style", "pretty_line", "flux_label", "savefig")

# Text and chart inks: primary and secondary text, muted axes, recessive grid, the plot surface.
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e6e5e1", "#ffffff"
MASKED = "#f4f3ef"      # background of the pixels that are not fitted

# Quiet grey axes with inward ticks on all sides and no grid, so that the data carry the figure;
# fonttype 42 embeds TrueType fonts in PDF/EPS, as journals require.
PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "axes.titlelocation": "left",
    "legend.fontsize": 8,
    "legend.labelcolor": INK2,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.7,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "axes.grid": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "lines.solid_capstyle": "round",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",         # SVG text stays text (editable in Inkscape or Illustrator)
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}

# One palette for the figures and the app (app/static/wf.js). The six line colours (broad, narrow,
# outflow, Fe II, host, Bowen) differ from each other for normal vision and the three colour-vision
# deficiencies (OKLab distance x 100 >= 15 and >= 8, every pair); the Balmer components are a lighter
# blue, told apart from the broad lines by their dashed line. The model is drawn in ink over grey data.
COLORS = {
    "data": "#9b9a95",
    "model": INK,
    "continuum": INK2,
    "host": "#008300",
    "feii": "#e34948",
    "balmer": "#86b6ef",
    "balmer_cont": "#86b6ef",
    "balmer_lines": "#86b6ef",
    "narrow": "#1baf7a",
    "broad": "#2a78d6",
    "outflow": "#eda100",
    "bowen": "#4a3aa7",
    "other": MUTED,
}
# several broad components: steps of the broad blue
BROAD_SHADES = ("#2a78d6", "#104281", "#6da7ec", "#1c5cab", "#9ec5f4")
WIDTHS = {"single": 3.5, "double": 7.1,
          # journal column widths (inches): MNRAS 84 / 174 mm, A&A 88 / 180 mm, ApJ 3.5 / 7.1 in
          "mnras": 3.31, "mnras-double": 6.85, "aa": 3.46, "aa-double": 7.09, "apj": 3.5, "apj-double": 7.1}

_GREEK = {"Lya": "Ly$\\alpha$", "Lyb": "Ly$\\beta$", "Ha": "H$\\alpha$", "Hb": "H$\\beta$", "Hg": "H$\\gamma$",
          "Hd": "H$\\delta$", "He": "H$\\epsilon$"}
for _series, _short in (("Pa", "a b g d e"), ("Br", "a b g d e"), ("Pf", "b g d e")):
    for _k in _short.split():
        _GREEK[_series + _k] = _series + "$\\" + {"a": "alpha", "b": "beta", "g": "gamma", "d": "delta",
                                                    "e": "epsilon"}[_k] + "$"


def _shade_masked(ax, wave, unused):
    """Shade the stretches of pixels that are not fitted (contiguous runs, 2 A gaps merged)."""
    edges = np.flatnonzero(np.diff(np.r_[0, unused.astype(int), 0]))
    spans = []
    for a, b in zip(edges[::2], edges[1::2]):
        lo, hi = wave[a], wave[b - 1]
        if spans and lo - spans[-1][1] < 2:
            spans[-1][1] = hi
        else:
            spans.append([lo, hi])
    for lo, hi in spans:
        ax.axvspan(lo, hi, color=MASKED, lw=0, zorder=0)


def _residual_axes(rax, xlim):
    """Residual panel: the +-1 sigma band and the zero line."""
    rax.fill_between(xlim, -1, 1, color=GRID, lw=0, zorder=0.5)
    rax.axhline(0, color=INK, lw=0.6, zorder=1)


SANS_RC = {"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
           "mathtext.fontset": "dejavusans"}


@contextmanager
def paper_style(enabled=True):
    """Temporarily apply the journal style (used around figure creation and saving). ``enabled`` may also
    name the font: 'serif' (the default, STIX / Times) or 'sans-serif' (Helvetica / Arial)."""
    if not enabled:
        yield
        return
    import matplotlib
    rc = dict(PAPER_RC, **(SANS_RC if enabled == "sans-serif" else {}))
    with matplotlib.rc_context(rc):
        yield


def pretty_line(name):
    """'[OIII]5007' -> '[O III] λ5007', 'Hb' -> 'Hβ', 'MgII2798' -> 'Mg II λ2798'."""
    if name in _GREEK:
        return _GREEK[name]
    if name.startswith("H2-"):                       # 'H2-1-0S(1)' -> 'H$_2$ 1-0 S(1)'
        v, branch = name[3:].split("S") if "S" in name[3:] else name[3:].split("Q")
        return f"H$_2$ {v} {'S' if 'S' in name[3:] else 'Q'}{branch}"
    m = re.match(r"^(\[)?([A-Z][a-z]?)([IVX]+)(\])?(\d+)?$", name)
    if not m:
        return name
    left, element, ion, right, wave = m.groups()
    text = f"{left or ''}{element}$\\,${ion}{right or ''}"
    return text + (f" $\\lambda${wave}" if wave else "")


_GREEK_TEXT = {"a": "α", "b": "β", "g": "γ", "d": "δ", "e": "ε"}


def plain_line(name):
    """Plain-text line name: 'Hb' -> 'Hβ', '[OIII]5007' -> '[O III] λ5007', 'HeII4686' -> 'He II λ4686'."""
    m = re.fullmatch(r"(H|Ly|Pa|Br|Pf)([abgde])", name)
    if m:
        return m.group(1) + _GREEK_TEXT[m.group(2)]
    m = re.fullmatch(r"(\[?)([A-Z][a-z]?)([IVX]+)(\]?)(\d+)", name)
    if m:
        return f"{m.group(1)}{m.group(2)} {m.group(3)}{m.group(4)} λ{m.group(5)}"
    m = re.fullmatch(r"FeII(\d+)-(\d+)", name)
    if m:
        return f"Fe II {m.group(1)}–{m.group(2)} Å"
    return name


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
    if role == "broad" and is_bowen(comp):
        return "bowen", "Bowen + He$\\,$II"
    labels = {"narrow": "Narrow lines", "broad": "Broad lines", "outflow": "Outflow"}
    return (role, labels[role]) if role in labels else ("other", comp.name)


_ORDER = ["continuum", "host", "balmer_cont", "balmer_lines", "feii", "narrow", "broad",
          "bowen", "outflow", "other"]
_BOWEN = re.compile(r"(NIII|OIII3|HeII)\d+$")


def is_bowen(comp):
    """A line component of Bowen fluorescence and He II lines only (N III 4640, O III 3760, He II 4686, ...)."""
    names = list(getattr(comp, "line_names", []) or [])
    return bool(names) and all(_BOWEN.match(n) for n in names)
_LINESTYLES = ["-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1))]


def component_index(name):
    """1 for 'broad', 'broad_ha', 'broad_he'; 2 for 'broad2', 'broad_ha2', ..."""
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 1


_ORDINAL = {1: "1st", 2: "2nd", 3: "3rd"}


def _curves(result, group):
    """
    List of (kind, label, flux, linestyle) to draw. With group=True components
    of the same kind are summed (e.g. all broad components); with group=False
    every component is drawn, in its kind's colour with its own line style.
    group='auto' (default) sums by kind but, when a line is fitted with several
    components of one kind (two broad components, ...), also draws each of them:
    the 1st, 2nd, ... broad component (summed over the lines) dashed, dotted.
    """
    parts = result.model_flux(components=True)
    out = {}
    count = {}
    if group == "auto":
        idx = {}
        for comp in result.model.components:
            if comp.name in parts:
                kind, _ = _group_of(comp)
                idx.setdefault(kind, set()).add(component_index(comp.name))
        for comp in result.model.components:
            if comp.name not in parts:
                continue
            kind, label = _group_of(comp)
            if kind in out:
                out[kind][2] = out[kind][2] + parts[comp.name]
            else:
                ls = {"continuum": "--", "host": ":", "balmer_cont": "-."}.get(kind, "-")
                out[kind] = [kind, label, parts[comp.name].copy(), ls]
            if kind in ("broad", "outflow", "narrow") and len(idx[kind]) > 1:
                k = component_index(comp.name)
                key = (kind, k)
                if key in out:
                    out[key][2] = out[key][2] + parts[comp.name]
                else:
                    name = {"broad": "broad", "outflow": "outflow", "narrow": "narrow"}[kind]
                    out[key] = [kind, f"{_ORDINAL.get(k, f'{k}th')} {name} comp.", parts[comp.name].copy(),
                                _LINESTYLES[1 + (k - 1) % (len(_LINESTYLES) - 1)]]
        items = sorted(out.values(), key=lambda v: _ORDER.index(v[0]) if v[0] in _ORDER else 99)
        return [tuple(v) for v in items]
    for comp in result.model.components:
        if comp.name not in parts:      # e.g. an Extinction component
            continue
        kind, label = _group_of(comp)
        if group:
            key = kind
        else:
            key = comp.name
            if kind in ("narrow", "broad", "bowen", "outflow", "other"):
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


def _label_lines(ax, result, view, model, xlim, max_labels=12, which="strongest", fontsize=7, font=None,
                 color="0.15", connectors="leader", rotation=90, lines=None, extra_axes=()):
    """
    Name the lines in view: the labels sit in one row at the top of the panel,
    spread apart so that they do not overlap, each joined to its line by a thin
    grey line.

    :param which: 'strongest' (lines above 5% of the strongest, at most ``max_labels``)
        or 'all' (every fitted line in view); ``lines`` gives the names explicitly.
    :param connectors: 'leader' (grey line from the label down to the spectrum),
        'full' (grey dotted line through the whole panel and the residuals) or 'none'.
    """
    from .components import Lines
    s = result.spectrum
    parts = result.model_flux(components=True)
    peaks = {}
    for comp in result.model.components:
        if not isinstance(comp, Lines) or getattr(comp, "absorption", False):
            continue
        prof = parts[comp.name]
        for ln, lam in zip(comp.line_names, comp._wave):
            lam_obs = lam if s.medium == "vacuum" else lam / 1.00028
            if not (xlim[0] < lam_obs < xlim[1]):
                continue
            near = np.abs(s.wave - lam_obs) < 3
            if near.any():
                peaks[ln] = (lam_obs, max(peaks.get(ln, (0, 0))[1], float(prof[near].max())))
    if lines is not None:
        from .lines import line_wavelength
        peaks = {ln: peaks.get(ln, (_rest_wave(result, ln) / (1.00028 if s.medium != "vacuum" else 1), 1.0))
                 for ln in lines}
        peaks = {ln: v for ln, v in peaks.items() if xlim[0] < v[0] < xlim[1]}
    if not peaks:
        return
    strongest = max(p for _, p in peaks.values()) or 1.0
    if lines is not None or which == "all":
        chosen = [(p, ln, lam) for ln, (lam, p) in peaks.items()]
    else:
        chosen = sorted(((p, ln, lam) for ln, (lam, p) in peaks.items() if p > 0.05 * strongest),
                        reverse=True)[:max_labels]
    # one label per position: lines closer than 0.3% of the range share the strongest one's label
    span = xlim[1] - xlim[0]
    chosen.sort(key=lambda t: -t[0])
    kept = []
    for p, ln, lam in chosen:
        if all(abs(lam - q[2]) > 0.003 * span for q in kept):
            kept.append((p, ln, lam))
    kept.sort(key=lambda t: t[2])
    if not kept:
        return
    # label positions: as close to the line as possible, at least one label height apart
    fig = ax.figure
    width_in = ax.get_position().width * fig.get_size_inches()[0]
    per_in = span / max(width_in, 0.1)              # wavelength per inch
    texts = [pretty_line(k[1]) for k in kept]
    if rotation:
        widths = np.full(len(kept), fontsize * 1.25 / 72 * per_in)
    else:   # approximate text width (mathtext names are shorter than their source)
        widths = np.array([(len(re.sub(r"\$|\\[a-z]+|\\,", "x", t)) * 0.6 + 0.8) * fontsize / 72 * per_in
                           for t in texts])
    lams = np.array([k[2] for k in kept])
    lo_x, hi_x = xlim[0] + 0.015 * span, xlim[1] - 0.015 * span
    # horizontal labels that do not fit in one row alternate between two rows
    rows = np.zeros(len(kept), int)
    if not rotation and widths.sum() > 0.85 * (hi_x - lo_x):
        rows = np.arange(len(kept)) % 2

    def spread(idx):
        pos = lams[idx].copy()
        w = widths[idx]
        for _ in range(300):          # relax: push overlapping neighbours apart
            moved = False
            for i in range(1, len(pos)):
                need = 0.5 * (w[i - 1] + w[i])
                d = pos[i] - pos[i - 1]
                if d < need - 1e-9:
                    pos[i - 1] -= (need - d) / 2
                    pos[i] += (need - d) / 2
                    moved = True
            pos = np.clip(pos, lo_x + w / 2 * (rotation == 0), hi_x - w / 2 * (rotation == 0))
            if not moved:
                break
        return pos

    pos = np.empty(len(kept))
    for r in np.unique(rows):
        idx = np.nonzero(rows == r)[0]
        pos[idx] = spread(idx)
    ytop = ax.get_ylim()[1]
    yrow = 0.97
    style = dict(fontsize=fontsize, color=color, rotation=rotation, ha="center", va="top", clip_on=False,
                 gid="linelabel")
    if font:
        style["family"] = font
    for (p, ln, lam), x, row in zip(kept, pos, rows):
        yrow = 0.97 - row * (fontsize * 1.4 / 72) / (ax.get_position().height * fig.get_size_inches()[1])
        near = view & (np.abs(s.wave - lam) < 0.004 * span)
        y = np.nanmax(np.maximum(s.flux[near], model[near])) if near.any() else p
        y = min(y + 0.03 * ytop, 0.8 * ytop)
        if connectors == "full":
            for a in (ax,) + tuple(extra_axes):
                a.axvline(lam, color="0.6", lw=0.5, ls=":", zorder=0.5)
            ax.annotate(pretty_line(ln), xy=(lam, yrow - 0.02), xycoords=("data", "axes fraction"),
                        xytext=(x, yrow), textcoords=("data", "axes fraction"),
                        arrowprops=dict(arrowstyle="-", lw=0.5, color="0.6", shrinkA=1, shrinkB=0), **style)
        elif connectors == "leader":
            ax.annotate(pretty_line(ln), xy=(lam, y), xycoords="data",
                        xytext=(x, yrow), textcoords=("data", "axes fraction"),
                        arrowprops=dict(arrowstyle="-", lw=0.5, color="0.6", shrinkA=2, shrinkB=1), **style)
        else:
            ax.text(x, yrow, pretty_line(ln), transform=ax.get_xaxis_transform(), **style)


# -------------------------------------------------------------------- plots
def plot_result(result, ax=None, components=True, residuals=True, xlim=None, fig=None,
                width="double", group="auto", label_lines=False, title=None, style=True,
                colors=None, hide=(), data=True, model=True, legend=True, label_options=None, styles=None,
                zoom=None, zoom_subtract=(), height=1.0):
    """
    Data, model, components and residuals.

    :param width: 'single' (3.5 in), 'double' (7.1 in) or a width in inches.
    :param group: 'auto' (default): sum components by role, and when a line
        has several components of one role (two broad components, ...) also
        draw each of them; True: only the sums; False: every component
        separately.
    :param label_lines: annotate the strongest lines with their names.
    :param title: figure title (default none, as is usual in papers).
    :param style: apply the journal style: True or 'serif', 'sans-serif' (Helvetica / Arial), False (none).
    :param height: the figure height relative to the default for its width (e.g. 0.8 compact, 1.25 tall).
    :param colors: colours by kind, overriding ``COLORS``: 'data', 'model', 'continuum',
        'host', 'feii', 'balmer_cont', 'balmer_lines', 'narrow', 'broad', 'outflow',
        'bowen', 'other', e.g. ``{"broad": "navy", "model": "red"}``.
    :param hide: kinds of components not to draw, e.g. ``("host", "feii")``.
    :param styles: line style by kind (the kinds of ``colors``, and 'data', 'model'): a dict of
        ``fill`` (fill under the curve; default only Fe II), ``lw`` (line width, points) and ``ls``
        (matplotlib line style, e.g. '-', '--', ':', '-.'), e.g. ``{"feii": {"fill": False},
        "broad": {"lw": 1.5, "fill": True}, "model": {"ls": "--"}}``.
    :param data, model: draw the data, the total model.
    :param legend: True or 'above' (above the axes), 'inside' (upper right in the panel), 'right' (beside
        the panel) or False.
    :param width: also a journal column: 'mnras', 'mnras-double', 'aa', 'aa-double', 'apj', 'apj-double'.
    :param zoom: wavelength ranges drawn again in a row of panels under the spectrum, e.g.
        ``[(4700, 5100), (6400, 6800)]`` for the H-beta and H-alpha regions (the main panel marks them).
    :param zoom_subtract: kinds of components subtracted from the data and the model in the zoom panels, so that
        the lines stand alone, e.g. ``"continuum"`` or ``("continuum", "host", "feii")``; continuum includes the
        Balmer continuum and the high-order Balmer lines.
    :param label_options: options of the line labels (see ``_label_lines``):
        which ('strongest' or 'all'), lines (names), fontsize, font (family),
        color, connectors ('leader', 'full', 'none'), rotation (90 or 0).
    """
    col = dict(COLORS, **(colors or {}))
    hide = set(hide or ())
    sty = {k: dict(v) for k, v in (styles or {}).items()}
    sty.setdefault("feii", {}).setdefault("fill", True)
    with paper_style(style):
        s = result.spectrum
        sel = result.mask
        if xlim is None:
            xlim = (s.wave[sel].min(), s.wave[sel].max())
        view = (s.wave >= xlim[0]) & (s.wave <= xlim[1])
        narrow = _figsize(width, 1)[0] < 5
        aspect = ((0.95 if narrow else 0.62) if residuals else (0.75 if narrow else 0.48)) * float(height)
        zoom = [tuple(map(float, z)) for z in (zoom or [])]
        zaxes = []
        if ax is None:
            if zoom:
                aspect += 0.62 if narrow else 0.3
                fig = new_figure(fig, _figsize(width, aspect))
                outer = fig.add_gridspec(2, 1, height_ratios=[(4.2 if residuals else 3.2), 2.2], hspace=0.32)
                top = outer[0].subgridspec(2 if residuals else 1, 1, height_ratios=[3.2, 1] if residuals else [1],
                                           hspace=0.06)
                ax = fig.add_subplot(top[0])
                rax = fig.add_subplot(top[1], sharex=ax) if residuals else None
                row = outer[1].subgridspec(1, len(zoom), wspace=0.28 if not narrow else 0.4)
                zaxes = [fig.add_subplot(row[k]) for k in range(len(zoom))]
            else:
                fig = new_figure(fig, _figsize(width, aspect))
                if residuals:
                    ax, rax = fig.subplots(2, 1, sharex=True,
                                           gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.06})
                else:
                    ax, rax = fig.subplots(), None
        else:
            fig, rax = ax.figure, None

        wave = s.wave[view]
        model_flux = result.model_flux()
        used = np.where(sel[view], s.flux[view], np.nan)
        if data:
            ax.plot(wave, used, color=col["data"], lw=sty.get("data", {}).get("lw", 0.6), drawstyle="steps-mid",
                    label="Data", zorder=2)
        unused = view & ~sel
        if data and unused.any():
            for a in (ax, rax) if rax is not None else (ax,):
                _shade_masked(a, wave, unused[view])
            # NaN where fitted, so that masked stretches are not joined across the gaps
            ax.plot(wave, np.where(unused[view], s.flux[view], np.nan), color="#c9c8c2", lw=0.5,
                    drawstyle="steps-mid", zorder=1, label="Not fitted")
        if components:
            seen = set()
            for kind, label, flux, ls in _curves(result, group):
                if kind in hide or not np.any(np.abs(flux[view]) > 0):
                    continue  # e.g. Balmer continuum entirely outside the plotted range
                st = sty.get(kind, {})
                main = kind not in seen          # the sum of a kind; its 1st, 2nd ... components keep their dashes
                seen.add(kind)
                thin = ls not in ("-",) and kind in ("broad", "outflow", "narrow")
                c = col.get(kind, col["other"])
                if main and st.get("fill"):      # e.g. the Fe II pseudo-continuum, filled lightly
                    ax.fill_between(wave, 0, flux[view], color=c, alpha=0.13, lw=0, zorder=2.5)
                lw = st.get("lw", 1.0)
                ax.plot(wave, flux[view], color=c, lw=0.7 * lw if thin else lw,
                        ls=st.get("ls", ls) if main else ls, label=label, zorder=3)
        if model:
            ax.plot(wave, model_flux[view], color=col["model"], lw=sty.get("model", {}).get("lw", 0.9),
                    ls=sty.get("model", {}).get("ls", "-"), label="Model", zorder=4)

        ax.set_xlim(*xlim)
        good = view & sel & np.isfinite(s.flux)
        lo = min(0.0, float(np.nanpercentile(s.flux[good], 0.5)))
        hi = float(np.nanpercentile(np.maximum(s.flux[good], model_flux[good]), 99.9))
        lopt = dict(label_options or {})
        room = 1.08
        if label_lines:
            room = 1.25 + 0.05 * float(lopt.get("fontsize", 7)) if lopt.get("rotation", 90) else 1.25
        ax.set_ylim(lo, hi * room)
        ylabel = flux_label(s)
        ax.set_ylabel(ylabel.replace(" (", "\n(", 1) if narrow else ylabel)
        handles, labels = ax.get_legend_handles_labels()
        # The legend sits above the axes; a title goes on top of it (as the
        # legend title) so the two cannot overlap.
        where = "above" if legend is True else legend
        inside = None
        if where == "above" and labels:
            ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                      ncol=min(len(labels), 2 if narrow else 4),
                      columnspacing=1.2, borderaxespad=0.3, title=title or None,
                      title_fontsize=matplotlib.rcParams["axes.titlesize"])
        elif where == "right" and labels:
            ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1, borderaxespad=0.0,
                      handlelength=1.6)
            if title:
                ax.set_title(title)
        elif where == "inside" and labels:
            inside = (handles, labels, 1 if narrow else 2)      # placed after the line labels, clear of them
            if title:
                ax.set_title(title)
        elif title:
            ax.set_title(title)

        if rax is not None:
            r = (s.flux - model_flux) / s.err
            rr = np.where(sel[view], r[view], np.nan)
            _residual_axes(rax, xlim)
            rax.plot(wave, rr, color=INK2, lw=0.5, drawstyle="steps-mid")
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
        if label_lines:
            if label_lines == "all":
                lopt.setdefault("which", "all")
            _label_lines(ax, result, view, model_flux, xlim, extra_axes=(rax,) if rax is not None else (), **lopt)
        if inside:
            _place_legend(ax, *inside)
        from matplotlib.transforms import blended_transform_factory
        for k, (zax, (z0, z1)) in enumerate(zip(zaxes, zoom)):
            _zoom_panel(zax, result, (z0, z1), col, hide, group, sty, data, model, model_flux, zoom_subtract)
            # each zoomed range: a faint blue band (grey means "not fitted") and its letter
            letter = "abcdefgh"[k % 8]
            for a in (ax, rax) if rax is not None else (ax,):
                a.axvspan(z0, z1, color="#2a78d6", alpha=0.06, lw=0, zorder=0)
            lab_ax = rax if rax is not None else ax
            tr = blended_transform_factory(lab_ax.transData, lab_ax.transAxes)
            lab_ax.text(z0, 0.93, f" ({letter})", transform=tr, ha="left", va="top", fontsize=7, color=INK2, zorder=6)
            zax.text(0.03, 0.95, f"({letter})", transform=zax.transAxes, ha="left", va="top", fontsize=8, color=INK2)
        if zaxes:
            sub = _subtract_kinds(zoom_subtract)
            names = {"continuum": "continuum", "host": "host", "feii": "Fe$\\,$II"}
            told = [names[k] for k in ("continuum", "host", "feii") if k in sub]
            zaxes[0].set_ylabel(flux_label(s).split(" (")[0] + (" $-$ " + " $-$ ".join(told) if told else ""))
    return fig


def _subtract_kinds(subtract):
    """The kinds of components a zoom panel subtracts ('continuum' brings the Balmer continuum and lines along)."""
    if not subtract:
        return set()
    kinds = {subtract} if isinstance(subtract, str) else set(subtract)
    if "continuum" in kinds:
        kinds |= {"balmer_cont", "balmer_lines"}
    return kinds


def _zoom_panel(ax, result, xlim, col, hide, group, sty, data, model, model_flux, subtract=()):
    """One wavelength range of the fit again: data, model and components, in the main panel's colours,
    optionally with some kinds of components (the continuum, Fe II, the host) subtracted."""
    from .fitcompare import _groups
    s = result.spectrum
    view = (s.wave >= xlim[0]) & (s.wave <= xlim[1])
    if view.sum() < 3:
        ax.text(0.5, 0.5, "outside the spectrum", transform=ax.transAxes, ha="center", va="center", color=MUTED)
        return
    wave = s.wave[view]
    sel = result.mask[view]
    kinds = _subtract_kinds(subtract)
    base = np.zeros(wave.size)
    if kinds:
        for key, (_, f) in _groups(result, wave).items():
            if key in kinds:
                base = base + f
    model_flux = np.array(model_flux, float)
    model_flux[view] = model_flux[view] - base
    if data:
        ax.plot(wave, np.where(sel, s.flux[view] - base, np.nan), color=col["data"], lw=0.6, drawstyle="steps-mid",
                zorder=2)
    if kinds:
        ax.axhline(0, color=MUTED, lw=0.5, zorder=1)
    seen = set()
    for kind, label, flux, ls in _curves(result, group):
        if kind in hide or kind in kinds or not np.any(np.abs(flux[view]) > 0):
            continue
        st = sty.get(kind, {})
        main = kind not in seen
        seen.add(kind)
        c = col.get(kind, col["other"])
        if main and st.get("fill"):
            ax.fill_between(wave, 0, flux[view], color=c, alpha=0.13, lw=0, zorder=2.5)
        thin = ls not in ("-",) and kind in ("broad", "outflow", "narrow")
        ax.plot(wave, flux[view], color=c, lw=(0.7 if thin else 1.0) * st.get("lw", 1.0),
                ls=st.get("ls", ls) if main else ls, zorder=3)
    if model:
        ax.plot(wave, model_flux[view], color=col["model"], lw=sty.get("model", {}).get("lw", 0.9), zorder=4)
    good = sel & np.isfinite(s.flux[view])
    shown = s.flux[view] - base
    hi = (max(float(np.nanpercentile(shown[good], 99.5)), float(np.nanmax(model_flux[view][good])))
          if good.any() else 1.0)                    # the model peaks (narrow lines) always fit
    lo = min(0.0, float(np.nanpercentile(shown[good], 0.5))) if good.any() else 0
    lo -= 0.04 * (hi - lo) if kinds else 0.0
    ax.set_xlim(*xlim)
    ax.set_ylim(lo, hi * 1.08)
    ax.set_xlabel(_wave_label(s))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))


def _place_legend(ax, handles, labels, ncol, fontsize=7):
    """A legend inside the axes where it covers the fewest data points and line labels (above, if none is free)."""
    fig = ax.figure
    if not hasattr(fig.canvas, "get_renderer"):       # a bare Figure (the app): give it a canvas to measure with
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        FigureCanvasAgg(fig)
    renderer = fig.canvas.get_renderer()
    pts = []
    for ln in ax.get_lines():
        xy = np.asarray(ln.get_xydata(), float)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy):
            pts.append(ax.transData.transform(xy[:: max(1, len(xy) // 2000)]))
    pts = np.vstack(pts) if pts else np.empty((0, 2))
    texts = [t.get_window_extent(renderer) for t in ax.texts if t.get_visible() and t.get_text()]
    best, score = None, np.inf
    for loc, nc in [(loc, nc) for nc in sorted({ncol, ncol + 1, ncol + 2}) for loc in
                    ("upper right", "upper left", "upper center", "center right", "lower right", "center left",
                     "lower left", "center", "lower center")]:
        leg = ax.legend(handles, labels, loc=loc, ncol=nc, fontsize=fontsize, columnspacing=1.0, handlelength=1.6)
        bb = leg.get_window_extent(renderer)
        inside = ((pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1) & (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1)).sum()
        overlap = sum(max(0, min(bb.x1, b.x1) - max(bb.x0, b.x0)) * max(0, min(bb.y1, b.y1) - max(bb.y0, b.y0))
                      for b in texts)
        s = inside + overlap / 2.0                # a label hidden by the legend is worse than a few data points
        if s < score:
            best, score = (loc, nc), s
        leg.remove()
    if score > 15:                          # nowhere free inside (a crowded figure): above the axes instead
        ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=min(len(labels), ncol + 2),
                  columnspacing=1.2, borderaxespad=0.3)
        return
    leg = ax.legend(handles, labels, loc=best[0], ncol=best[1], fontsize=fontsize, columnspacing=1.0, handlelength=1.6)
    # still touching the data at an upper position: raise the top of the axis a little until it is clear
    if best[0].startswith("upper"):
        lines = [np.asarray(ln.get_xydata(), float) for ln in ax.get_lines()]
        xy = np.vstack([a[np.isfinite(a).all(axis=1)] for a in lines if len(a)]) if lines else np.empty((0, 2))
        for _ in range(8):
            bb = leg.get_window_extent(renderer)
            p = ax.transData.transform(xy)
            if not ((p[:, 0] > bb.x0) & (p[:, 0] < bb.x1) & (p[:, 1] > bb.y0) & (p[:, 1] < bb.y1)).any():
                break
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.05 * (hi - lo))


# --------------------------------------------------------- single lines
_MAIN_LINES = ("Hb", "Ha", "MgII2798", "CIV1549", "[OIII]5007", "Hg", "[NII]6583", "[OII]3729",
               "[SII]6716", "[OI]6300", "HeII4686")
# without broad lines the narrow-line diagnostics come first
_MAIN_NARROW = ("[OIII]5007", "Ha", "Hb", "[NII]6583", "[SII]6716", "[OI]6300", "[OII]3729",
                "[NeIII]3869", "HeII4686", "Hg")


def fitted_lines(result):
    """
    The lines of a fit that can be shown alone, most interesting first: lines
    with several components (broad + narrow, two broad, ...) before single
    ones, main diagnostic lines first, then by flux. Returns a list of dicts
    {line, wavelength, components, roles, flux}.
    """
    from .lines import line_wavelength
    from .components import Lines
    s = result.spectrum
    lo, hi = (result.model.wave_range or (s.wave.min(), s.wave.max()))
    found = {}
    table = result.lines
    for comp in result.model.components:
        if not isinstance(comp, Lines):
            continue
        for ln in comp.line_names:
            lam = _rest_wave(result, ln)
            if not (lo < lam < hi):
                continue
            d = found.setdefault(ln, {"line": ln, "wavelength": float(lam), "components": [], "roles": []})
            d["components"].append(comp.name)
            d["roles"].append(comp.role)
    for ln, d in found.items():
        t = table[table.line == ln]
        d["flux"] = float(t.flux.sum()) if len(t) else 0.0
    has_broad = any("broad" in d["roles"] for d in found.values())
    main = _MAIN_LINES if has_broad else _MAIN_NARROW
    rank = lambda d: (-len(d["components"]), main.index(d["line"]) if d["line"] in main else 99, -d["flux"])
    return sorted((d for d in found.values() if d["flux"] > 0), key=rank)


def _rest_wave(result, line):
    """Rest wavelength of a line: from the line list, or from the model for a line defined by (name, wavelength)."""
    from .lines import line_wavelength
    try:
        return float(line_wavelength(line))
    except KeyError:
        for comp in result._line_components():
            if line in comp.line_names:
                return float(comp._wave[comp.line_names.index(line)])
        raise


def main_line(result):
    """The line a single-line plot shows by default (the main broad line, else [OIII]...)."""
    lines = fitted_lines(result)
    broad = [d for d in lines if "broad" in d["roles"]]
    for name in (_MAIN_LINES if broad else _MAIN_NARROW):
        for d in broad or lines:
            if d["line"] == name:
                return name
    return (broad or lines)[0]["line"] if lines else None


def _line_parts(result, line, wave):
    """{component name: profile of ``line`` in that component} on ``wave``."""
    s = result.spectrum
    slices = dict(zip([c.name for c in result.model.components], result.model._slices()))
    out = {}
    for comp in result._line_components():
        if line in comp.line_names:
            out[comp.name] = comp.line_profile(wave, result.full[slices[comp.name]], s.context, line)
    return out


def _transmission(result, wave):
    """Product of the multiplicative components (absorption, extinction) acting on lines."""
    m = result.model
    slices = m._slices()
    T = np.ones_like(np.asarray(wave, float))
    for i in m._multipliers():
        c = m.components[i]
        T = T * c.value(wave, result.full[slices[i]], result.spectrum.context)
    return T


def _line_info(result, line, sep="\n"):
    """FWHM, sigma and flux of a line (its broad components summed when it has any), rounded to their errors."""
    comps = [c for c in result._line_components() if line in c.line_names]
    broad = [c.name for c in comps if c.role == "broad"]
    m = result.measure(line, broad or None)
    frag = " (fragile)" if m.get("fwhm_fragile") else ""
    return (f"FWHM {_pm(m['fwhm'], m['fwhm_err'], 'km/s')}{frag}{sep}$\\sigma$ {_pm(m['sigma'], m['sigma_err'], 'km/s')}"
            f"{sep}flux {_pm(m['flux'], m['flux_err'])}")


def plot_line(result, line=None, ax=None, fig=None, window=None, velocity=True, residuals=True,
              others=False, info=True, width="single", title=None, style=True, rax=None, mark_fwhm=True):
    """
    One emission line alone: the data with everything else in the model
    subtracted (continuum, host, FeII and all other lines), each component
    of the line (narrow, outflow, 1st, 2nd broad ...) and their sum, and the
    residuals.

    :param line: line name, e.g. 'Hb', 'Ha', '[OIII]5007' (default: the main
        broad line; see :func:`fitted_lines` for what a fit offers).
    :param window: half-width in km/s (default: from the line width).
    :param velocity: velocity axis (km/s from the rest wavelength); False: wavelength.
    :param others: also draw the other lines in the window (dotted), which stay
        subtracted from the data.
    :param info: write the FWHM, line dispersion sigma and flux of the line (summed over its
        broad components, or over all components) in the panel; a FWHM that a shoulder at half
        maximum makes fragile is marked.
    :param mark_fwhm: mark the half-maximum points of the broad profile (its FWHM) with a bar.
    :param rax: with ``ax``, the axes for the residuals (as in a grid of lines).
    """
    from .components import _in_medium
    from .lines import line_wavelength
    with paper_style(style):
        s = result.spectrum
        line = line or main_line(result)
        if line is None:
            raise ValueError("this fit has no emission lines")
        lam0 = float(_in_medium(np.array([_rest_wave(result, line)]), s.context)[0])
        comps = [c for c in result._line_components() if line in c.line_names]
        if not comps:
            raise ValueError(f"{line!r} is not in the model; fitted lines: "
                             f"{', '.join(d['line'] for d in fitted_lines(result))}")
        broad = [c.name for c in comps if c.role == "broad"]
        meas = result.measure(line, broad or None)
        if window is None:
            window = float(np.clip(2.5 * meas["fwhm"], 1500.0, 15000.0)) if broad else 1500.0
        v_data = (s.wave / lam0 - 1) * C_KMS
        view = np.abs(v_data) <= window
        if view.sum() < 5:
            raise ValueError(f"{line} is outside the spectrum")
        # the data with everything but this line subtracted (the line as observed, i.e.
        # after any absorption or extinction in the model)
        own = sum(_line_parts(result, line, s.wave).values()) * _transmission(result, s.wave)
        model = result.model_flux()
        isolated = s.flux - (model - own)
        v_fine = np.linspace(-window, window, 1500)
        lam_fine = lam0 * (1 + v_fine / C_KMS)
        parts = _line_parts(result, line, lam_fine)
        x_data = v_data if velocity else s.wave
        x_fine = v_fine if velocity else lam_fine

        narrow_panel = _figsize(width, 1)[0] < 5
        if ax is None:
            fig = new_figure(fig, _figsize(width, (1.05 if narrow_panel else 0.6) if residuals else
                                           (0.8 if narrow_panel else 0.45)))
            if residuals:
                ax, rax = fig.subplots(2, 1, sharex=True,
                                       gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.06})
            else:
                ax, rax = fig.subplots(), None
        else:
            fig = ax.figure
            rax = rax if residuals else None
        sel = result.mask & view
        ax.plot(x_data[view], np.where(sel[view], isolated[view], np.nan), color=COLORS["data"], lw=0.6,
                drawstyle="steps-mid", label="Data", zorder=2)
        ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
        nth = {}
        table = result.lines.set_index(["component", "line"])
        for comp in comps:
            role = comp.role if comp.role in COLORS else "other"
            k = nth[role] = nth.get(role, 0) + 1
            try:
                fwhm = table.loc[(comp.name, line), "fwhm"]
                extra = f" ({fwhm:.0f} km/s)" if np.isfinite(fwhm) else ""
            except KeyError:
                extra = ""
            n_role = sum(c.role == comp.role for c in comps)
            label = {"broad": "Broad", "narrow": "Narrow", "outflow": "Outflow"}.get(comp.role, comp.name)
            if n_role > 1:
                label = f"{label} {k}"
            ax.plot(x_fine, parts[comp.name], color=(BROAD_SHADES[(k - 1) % len(BROAD_SHADES)] if role == "broad"
                                                     else COLORS[role]), lw=1.0,
                    ls=_LINESTYLES[(k - 1) % len(_LINESTYLES)], label=label + extra, zorder=3)
        if len(broad) > 1:                             # the broad profile that the FWHM and sigma describe
            ax.plot(x_fine, sum(parts[c] for c in broad), color=COLORS["broad"], lw=1.4, alpha=0.45,
                    label="Broad sum", zorder=3.5, solid_capstyle="round")
        if len(comps) > 1:
            ax.plot(x_fine, sum(parts.values()), color=COLORS["model"], lw=0.9, label="Sum", zorder=4)
        T_fine = _transmission(result, lam_fine)
        if np.min(T_fine) < 0.99:                       # absorption in the window: the line as observed
            ax.plot(x_fine, sum(parts.values()) * T_fine, color=COLORS["model"], lw=0.9, ls="--",
                    label="Sum $\\times$ absorption", zorder=4)
        if others:
            seen = set()
            for d in fitted_lines(result):
                if d["line"] == line or d["line"] in seen:
                    continue
                v_other = (_rest_wave(result, d["line"]) / _rest_wave(result, line) - 1) * C_KMS
                if abs(v_other) < window:
                    seen.add(d["line"])
                    prof = sum(_line_parts(result, d["line"], lam_fine).values())
                    ax.plot(x_fine, prof, color=MUTED, lw=0.6, ls=":", zorder=2)
        good = sel & np.isfinite(isolated)
        top = max(float(np.nanpercentile(isolated[good], 99.5)) if good.any() else 1.0,
                  float(np.max(sum(parts.values()))))
        bottom = min(0.0, float(np.nanpercentile(isolated[good], 0.5)) if good.any() else 0.0)
        ax.set_ylim(bottom - 0.05 * top, (1.5 if info else 1.15) * top)   # room for the text above the peak
        ax.set_xlim(x_data[view].min(), x_data[view].max())
        ax.set_ylabel(flux_label(s).replace(" (", "\n(", 1) if narrow_panel else flux_label(s))
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7,
                  ncol=2 if narrow_panel else min(len(labels), 4), handlelength=1.8,
                  columnspacing=1.0, borderaxespad=0.3, frameon=False)
        head = pretty_line(line)
        ax.text(0.03, 0.95, head, transform=ax.transAxes, ha="left", va="top", fontsize=10)
        if info:
            # of the broad profile when there is one (the summed broad components), else of the whole line
            ax.text(0.03, 0.85, _line_info(result, line), transform=ax.transAxes, ha="left", va="top", fontsize=7,
                    color=INK2, linespacing=1.35)
        if mark_fwhm and broad:
            from .fitcompare import half_max_crossings
            bsum = sum(parts[c] for c in broad)
            if np.max(bsum) > 0:
                vl, vr = half_max_crossings(v_fine, bsum)
                half = 0.5 * np.max(bsum)
                xl, xr = (vl, vr) if velocity else (lam0 * (1 + vl / C_KMS), lam0 * (1 + vr / C_KMS))
                ax.plot([xl, xr], [half, half], color=INK2, lw=0.7, zorder=5, solid_capstyle="butt")
                ax.plot([xl, xr], [half, half], ls="none", marker="|", ms=5, mew=0.9, color=INK2, zorder=5)
        if title:
            ax.set_title(title)
        xlabel = _velocity_axis(rax if rax is not None else ax, window) if velocity else _wave_label(s)
        if velocity and rax is not None:
            _velocity_axis(ax, window)
        if rax is not None:
            r = np.where(sel, (s.flux - model) / s.err, np.nan)
            _residual_axes(rax, (x_data[view].min(), x_data[view].max()))
            rax.plot(x_data[view], r[view], color=INK2, lw=0.5, drawstyle="steps-mid")
            m = np.nanpercentile(np.abs(r[view]), 99.5) if np.isfinite(r[view]).any() else 5
            rax.set_ylim(-max(4.0, 1.15 * m), max(4.0, 1.15 * m))
            rax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper", symmetric=True))
            rax.set_ylabel("Resid. ($\\sigma$)")
            rax.set_xlabel(xlabel)
            ax.tick_params(labelbottom=False)
            fig.align_ylabels([ax, rax])
        else:
            ax.set_xlabel(xlabel)
    return fig


def _pm(v, e, unit=""):
    """'9340 $\\pm$ 510' or '(1.46 $\\pm$ 0.09)$\\times10^{4}$': the value rounded to two significant digits of its error."""
    if not np.isfinite(v):
        return "–"
    if not (np.isfinite(e) and e > 0):
        out = _short(v)
    else:
        p = int(np.floor(np.log10(e))) - 1                    # last digit kept
        exp = int(np.floor(np.log10(abs(v)))) if v else 0
        if abs(exp) >= 4:
            d = max(0, exp - p)
            out = f"({v / 10 ** exp:.{d}f} $\\pm$ {e / 10 ** exp:.{d}f})$\\times10^{{{exp}}}$"
        else:
            d = max(0, -p)
            vr, er = round(v, -p), round(e, -p)
            out = f"{vr:.{d}f} $\\pm$ {er:.{d}f}"
    return out + (f" {unit}" if unit else "")


def _velocity_axis(ax, window, which="x"):
    """km/s on the axis, labelled in 10^3 km/s when the range is wide (fewer, shorter tick labels)."""
    from matplotlib.ticker import FuncFormatter
    axis = ax.xaxis if which == "x" else ax.yaxis
    if window >= 4000:
        axis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}".replace("-", "\u2212")))
        return "Velocity (10$^3$ km$\\,$s$^{-1}$)"
    return "Velocity (km$\\,$s$^{-1}$)"


def _short(v):
    """A number in three significant digits, without an exponent where it can do without (1.57e4 -> 15700)."""
    if not np.isfinite(v):
        return "–"
    return f"{v:.3g}" if abs(v) < 1e5 else f"{v:.2e}".replace("e+0", "e").replace("e+", "e")


def line_grid_shape(n):
    """Columns and rows of a grid of n line panels: 1, 2 and 3 in a row, 4 as 2 x 2, then rows of three."""
    ncols = 1 if n <= 1 else 2 if n in (2, 4) else 3
    return ncols, int(np.ceil(n / ncols))


def plot_lines(result, lines=None, fig=None, ncols=None, window=None, velocity=True, width="double",
               style=True, residuals=False, info=True, others=False, mark_fwhm=True, legend=True, sharey=False):
    """
    Several lines in a grid (see :func:`plot_line`), one panel per line, in
    the order given. Default: the lines with several components, main lines
    first (at most four).

    :param ncols: columns of the grid (default: 1, 2 or 3 lines in a row, four as 2 x 2,
        then rows of three).
    :param residuals: a residual panel under each line.
    :param info: write the FWHM and flux in each panel.
    :param others: draw the other lines in each window (dotted).
    :param mark_fwhm: mark the half-maximum points of each broad profile.
    :param legend: one legend above the grid.
    :param sharey: the same flux scale in every panel (to compare the strengths of the lines).
    """
    with paper_style(style):
        if lines is None:
            lines = [d["line"] for d in fitted_lines(result) if len(d["components"]) > 1][:4] \
                or [main_line(result)]
        lines = list(lines)
        n = len(lines)
        if not n:
            raise ValueError("no lines to plot")
        ncols, nrows = (ncols, int(np.ceil(n / ncols))) if ncols else line_grid_shape(n)
        w, _ = _figsize(width, 1)
        if ncols == 1:                         # one panel: the width of a single journal column at most
            w = min(w, WIDTHS["single"] * 1.3)
        panel = w / ncols * (0.82 if residuals else 0.72)
        # FWHM, sigma and flux go above each panel: one line in a single column, two in two, three in three
        info_lines = (1 if ncols == 1 else 2 if ncols == 2 else 3) if info else 0
        title_in = (info_lines * 8.5 + 6) / 72 if info else 0.0
        height = panel * nrows + 0.35 + title_in * nrows
        fig = new_figure(fig, (w, height))
        grid = fig.add_gridspec(nrows, ncols, hspace=(0.32 if residuals else 0.25) + title_in / panel,
                                wspace=0.34 if ncols > 2 else 0.26, top=1 - title_in / height - 0.01, bottom=0.06)
        handles, bottom_axes, main_axes = {}, {}, []
        for k, ln in enumerate(lines):
            cell = grid[k // ncols, k % ncols]
            if residuals:
                sub = cell.subgridspec(2, 1, height_ratios=[3.2, 1], hspace=0.06)
                ax = fig.add_subplot(sub[0])
                rax = fig.add_subplot(sub[1], sharex=ax)
            else:
                ax, rax = fig.add_subplot(cell), None
            plot_line(result, ln, ax=ax, rax=rax, window=window, velocity=velocity, residuals=residuals,
                      info=False, others=others, style=False, mark_fwhm=mark_fwhm)
            if info:                           # above the panel, where it cannot cover the line
                parts = _line_info(result, ln, sep="\n").split("\n")
                txt = ("  ·  ".join(parts) if info_lines == 1 else
                       "  ·  ".join(parts[:2]) + "\n" + parts[2] if info_lines == 2 else "\n".join(parts))
                ax.set_title(txt, fontsize=6.5, color=INK2, loc="left", linespacing=1.3, pad=3)
            # one legend for the figure (below), one flux label (left); room above the line for its text
            for h, lab in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(re.sub(r"\s*\(.*\)$", "", lab), h)
            if ax.get_legend():
                ax.get_legend().remove()
            ax.set_ylabel("")
            if rax is not None:
                rax.set_ylabel("$\\Delta/\\sigma$" if k % ncols == 0 else "")
            bottom_axes[k] = rax if rax is not None else ax
            main_axes.append(ax)
        # x labels only on the bottom row, unless a panel's axis differs from the one below it (km/s, 10^3 km/s)
        for k, a in bottom_axes.items():
            below = bottom_axes.get(k + ncols)
            if below is not None and a.get_xlabel() == below.get_xlabel():
                a.set_xlabel("")
        if sharey:
            mains = main_axes
            lo = min(a.get_ylim()[0] for a in mains)
            hi = max(a.get_ylim()[1] for a in mains)
            for a in mains:
                a.set_ylim(lo, hi)
        fig.supylabel(flux_label(result.spectrum), fontsize=matplotlib.rcParams["axes.labelsize"],
                      color=matplotlib.rcParams["axes.labelcolor"])
        nh = len(handles)
        if legend:
            fig.legend(list(handles.values()), list(handles), loc="lower center", bbox_to_anchor=(0.5, 1.0),
                       ncol=nh if nh <= 6 else int(np.ceil(nh / 2)), frameon=False)
    return fig


PROFILE_COLORS = ("#2a78d6", "#e34948", "#1baf7a", "#eda100", "#4a3aa7", "#52514e")   # fixed order, never cycled


def plot_profiles(results, lines=None, labels=None, which="broad", normalize=True, window=15000.0, data=True,
                  mark_fwhm=True, ax=None, fig=None, width="single", style=True, legend=True):
    """
    Line profiles overlaid on one velocity axis, each scaled to its peak: several lines of one fit
    (e.g. broad Hb against broad Ha, or Mg II) or one line in several fits (epochs, models). The
    model profile of the chosen components is drawn with the data of that line (everything else in
    the model subtracted) behind it; the legend gives FWHM and sigma, and the half-maximum points
    are marked.

    :param results: a FitResult or a list of them.
    :param lines: line names (default: the main broad line); with several results, one line.
    :param labels: legend names of the curves (default: the line names, or 'fit 1', 'fit 2', ...).
    :param which: 'broad' (the broad components of the line, summed) or 'all' (every component).
    :param normalize: scale each profile and its data to the peak of the profile.
    :param window: half-width of the velocity axis (km/s).
    :param data: draw the data of each line behind its profile.

    More than six curves (e.g. many epochs, in time order) are coloured along a sequential ramp.
    """
    from .components import _in_medium
    from .fitcompare import half_max_crossings
    from .lines import line_wavelength
    results = list(results) if isinstance(results, (list, tuple)) else [results]
    if lines is None:
        lines = [main_line(results[0])]
    lines = [lines] if isinstance(lines, str) else list(lines)
    if len(results) > 1 and len(lines) > 1:
        raise ValueError("give several lines of one fit, or one line of several fits")
    pairs = [(r, ln) for r in results for ln in lines]
    if labels is None:
        labels = [pretty_line(ln) for ln in lines] if len(results) == 1 else [f"fit {k + 1}" for k in range(len(results))]
    if len(labels) != len(pairs):
        raise ValueError(f"{len(pairs)} curves but {len(labels)} labels")
    with paper_style(style):
        own_figure = ax is None
        if own_figure:
            ax = new_figure(fig, _figsize(width, 0.78 if _figsize(width, 1)[0] < 5 else 0.5)).subplots()
        v = np.linspace(-window, window, 3001)
        many = len(pairs) > len(PROFILE_COLORS)
        if many:                                   # epochs: a sequential ramp in time order, not cycled hues
            ramp = matplotlib.colormaps["viridis"](np.linspace(0.05, 0.9, len(pairs)))
        for k, ((r, ln), lab) in enumerate(zip(pairs, labels)):
            s = r.spectrum
            lam0 = float(_in_medium(np.array([_rest_wave(r, ln)]), s.context)[0])
            comps = [c for c in r._line_components() if ln in c.line_names]
            use = [c.name for c in comps if c.role == "broad"] if which == "broad" else [c.name for c in comps]
            if not use:
                raise ValueError(f"{ln}: no {'broad ' if which == 'broad' else ''}components in the fit")
            parts = _line_parts(r, ln, lam0 * (1 + v / C_KMS))
            prof = sum(parts[n] for n in use)
            scale = 1.0 / np.max(prof) if normalize and np.max(prof) > 0 else 1.0
            color = ramp[k] if many else PROFILE_COLORS[k]
            m = r.measure(ln, use)
            frag = " (fragile)" if m.get("fwhm_fragile") else ""
            name = (f"{lab}: FWHM {m['fwhm']:.0f} km/s" if many else
                    f"{lab}: FWHM{frag} {_pm(m['fwhm'], m['fwhm_err'])}, $\\sigma$ {_pm(m['sigma'], m['sigma_err'])} km/s")
            if data:
                vd = (s.wave / lam0 - 1) * C_KMS
                sel = r.mask & (np.abs(vd) <= window)
                # the data with everything in the model subtracted except the chosen components of this line
                own = sum(_line_parts(r, ln, s.wave)[n] for n in use) * _transmission(r, s.wave)
                iso = s.flux - (r.model_flux() - own)
                ax.plot(vd[sel], iso[sel] * scale, color=color, lw=0.5, alpha=0.35, drawstyle="steps-mid", zorder=2)
            ax.plot(v, prof * scale, color=color, lw=1.4, ls=_LINESTYLES[0], label=name, zorder=3)
            if mark_fwhm and np.max(prof) > 0:
                vl, vr = half_max_crossings(v, prof)
                ax.plot([vl, vr], [0.5 * np.max(prof) * scale] * 2, ls="none", marker="o", ms=3.5, color=color,
                        zorder=4)
        if normalize:
            ax.axhline(0.5, color=MUTED, lw=0.5, ls=":", zorder=1)
            ax.set_ylabel("Flux / peak")
            ax.set_ylim(-0.08, 1.1 + 0.11 * (int(np.ceil(len(pairs) / 2)) if many else len(pairs)))  # room for the legend
        else:
            ax.set_ylabel(flux_label(results[0].spectrum))
        ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
        ax.axvline(0, color=GRID, lw=0.6, zorder=0)
        ax.set_xlim(-window, window)
        ax.set_xlabel(_velocity_axis(ax, window))
        if legend:
            ax.legend(loc="upper left", fontsize=6 if many else 7, ncol=2 if many else 1, handlelength=1.6,
                      borderaxespad=0.4, columnspacing=1.0)
        if own_figure:
            ax.figure.tight_layout(pad=0.3)
    return ax.figure


def plot_correlations(table, columns=None, width="double", style=True, fig=None, alpha=0.05):
    """
    Corner plot of quantities over the epochs of one object (an epoch table, see :func:`fantasy_agn.epoch_table`):
    on the diagonal each quantity against time, below it every pair with error bars, the points coloured by date
    (dark early, light late) and the Spearman rho and p in the corner; pairs with p < ``alpha`` get a bold label.

    :param columns: quantities to show (default: the luminosity and the flux, FWHM, sigma and EW of the main broad line).
    """
    from .correlations import QUANTITY_LABELS, default_columns, epoch_correlations
    cols = list(columns or default_columns(table))
    n = len(cols)
    if n < 2:
        raise ValueError("choose at least two quantities")
    corr = {(d["x"], d["y"]): d for d in epoch_correlations(table, cols).to_dict("records")}
    mjd = table["mjd"].to_numpy(float) if "mjd" in table else np.arange(len(table), dtype=float)
    t = mjd if np.isfinite(mjd).all() else np.arange(len(table), dtype=float)
    span = np.ptp(t) or 1.0
    colors = matplotlib.colormaps["viridis"](0.05 + 0.85 * (t - t.min()) / span)
    with paper_style(style):
        w = _figsize(width, 1)[0]
        fig = new_figure(fig, (w, w * (0.98 if n > 2 else 0.9)))
        grid = fig.add_gridspec(n, n, hspace=0.08, wspace=0.08)
        axes = {}
        for i, yc in enumerate(cols):
            for j, xc in enumerate(cols):
                if j > i:
                    continue
                ax = fig.add_subplot(grid[i, j])
                axes[(i, j)] = ax
                x, sx = table[xc].to_numpy(float), table.get(xc + "_err", pd_nan(table)).to_numpy(float)
                if i == j:                     # the quantity against time
                    ax.errorbar(t, x, yerr=sx, fmt="none", ecolor=MUTED, elinewidth=0.6, zorder=1)
                    ax.scatter(t, x, c=colors, s=14, zorder=2, edgecolors="white", linewidths=0.4)
                    ax.plot(t[np.argsort(t)], x[np.argsort(t)], color=GRID, lw=0.7, zorder=0)
                    ax.set_title(QUANTITY_LABELS(xc), fontsize=7, color=INK2, pad=3)
                    ax.tick_params(labelleft=False, labelbottom=i == n - 1, labelsize=6)
                    if i == n - 1:
                        ax.set_xlabel("MJD" if np.isfinite(mjd).all() else "epoch", fontsize=7)
                    continue
                y, sy = table[yc].to_numpy(float), table.get(yc + "_err", pd_nan(table)).to_numpy(float)
                ax.errorbar(x, y, xerr=sx, yerr=sy, fmt="none", ecolor=MUTED, elinewidth=0.6, zorder=1)
                ax.scatter(x, y, c=colors, s=14, zorder=2, edgecolors="white", linewidths=0.4)
                d = corr.get((xc, yc)) or corr.get((yc, xc)) or {}
                rho, pv = d.get("rho", np.nan), d.get("p", np.nan)
                if np.isfinite(rho):
                    strong = pv < alpha
                    ptxt = "p < 10$^{-10}$" if pv < 1e-10 else f"p = {pv:.2g}"
                    ax.text(0.04, 0.95, f"$\\rho$ = {rho:+.2f}\n{ptxt}", transform=ax.transAxes, ha="left",
                            va="top", fontsize=6, color=INK if strong else INK2, fontweight="bold" if strong else "normal",
                            zorder=5, bbox=dict(facecolor=SURFACE, alpha=0.8, edgecolor="none", pad=1.2))
                ax.tick_params(labelsize=6, labelleft=j == 0, labelbottom=i == n - 1)
                if j == 0:
                    ax.set_ylabel(QUANTITY_LABELS(yc), fontsize=7)
                if i == n - 1:
                    ax.set_xlabel(QUANTITY_LABELS(xc), fontsize=7)
        # axes follow the points, not their error bars (one poorly measured epoch must not squash the rest)
        def lims(v):
            v = v[np.isfinite(v)]
            if not v.size:
                return None
            lo, hi = float(v.min()), float(v.max())
            pad = 0.08 * (hi - lo) if hi > lo else 0.05 * (abs(hi) or 1.0)
            return lo - pad, hi + pad
        for (i, j), ax in axes.items():
            xc, yc = cols[j], cols[i]
            lx = lims(t) if i == j else lims(table[xc].to_numpy(float))
            ly = lims(table[xc].to_numpy(float)) if i == j else lims(table[yc].to_numpy(float))
            if lx:
                ax.set_xlim(*lx)
            if ly:
                ax.set_ylim(*ly)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    return fig


def plot_relation(table, x, y, log="auto", fit=True, width="single", style=True, fig=None, n_boot=500):
    """
    One pair of quantities over the epochs (see :func:`fantasy_agn.epoch_table`): the points coloured by date with
    their errors and, with ``fit``, the maximum-likelihood line with intrinsic scatter and its 68% band
    (:func:`fantasy_agn.correlations.fit_relation`); the slope, scatter and Spearman rho are written in the panel.
    Returns (figure, fit dict or None).
    """
    from .correlations import QUANTITY_LABELS, fit_relation, relation_data
    vx, ex, vy, ey, lx, ly, ok = relation_data(table, x, y, log)
    mjd = table["mjd"].to_numpy(float)[ok] if "mjd" in table else np.arange(ok.sum(), dtype=float)
    tt = mjd if np.isfinite(mjd).all() else np.arange(vx.size, dtype=float)
    colors = matplotlib.colormaps["viridis"](0.05 + 0.85 * (tt - tt.min()) / (np.ptp(tt) or 1.0))
    res = fit_relation(table, x, y, log, n_boot=n_boot) if fit and vx.size >= 3 else None
    with paper_style(style):
        ax = new_figure(fig, _figsize(width, 0.82 if _figsize(width, 1)[0] < 5 else 0.55)).subplots()
        ax.errorbar(vx, vy, xerr=ex, yerr=ey, fmt="none", ecolor=MUTED, elinewidth=0.6, zorder=1)
        ax.scatter(vx, vy, c=colors, s=18, zorder=2, edgecolors="white", linewidths=0.4)
        if res is not None:
            xs = np.linspace(vx.min(), vx.max(), 200)
            lines = res["boot"][:, 0][:, None] + res["boot"][:, 1][:, None] * (xs - res["x0"])
            lo, hi = np.percentile(lines, [16, 84], axis=0)
            ax.fill_between(xs, lo, hi, color=COLORS["broad"], alpha=0.15, lw=0, zorder=0)
            ax.plot(xs, res["a"] + res["b"] * (xs - res["x0"]), color=COLORS["broad"], lw=1.2, zorder=3)
            ptxt = "p < 10$^{-10}$" if res["spearman_p"] < 1e-10 else f"p = {res['spearman_p']:.2g}"
            scat = "scatter $\\approx$ 0" if res["scatter"] < 1e-4 else f"scatter {_pm(res['scatter'], res['scatter_err'])}"
            # in the corner the relation leaves free: lower left for a falling one, upper left for a rising one
            top = res["b"] >= 0
            ax.text(0.03, 0.96 if top else 0.04, f"slope {_pm(res['b'], res['b_err'])}\n{scat}"
                    f"\n$\\rho$ = {res['spearman_rho']:+.2f}, {ptxt}, N = {res['n']}", transform=ax.transAxes,
                    ha="left", va="top" if top else "bottom", fontsize=7, color=INK2, linespacing=1.35,
                    bbox=dict(facecolor=SURFACE, alpha=0.85, edgecolor="none", pad=1.5), zorder=6)
        name = lambda c, lg: ("log " if lg else "") + QUANTITY_LABELS(c)
        ax.set_xlabel(name(x, lx))
        ax.set_ylabel(name(y, ly))
        if vx.size:                       # the range follows the points: one huge error bar must not flatten the rest
            for v, limit in ((vx, ax.set_xlim), (vy, ax.set_ylim)):
                d = np.ptp(v) or abs(v[0]) or 1.0
                limit(v.min() - 0.08 * d, v.max() + 0.08 * d)
        ax.figure.tight_layout(pad=0.3)
    return ax.figure, res


def pd_nan(table):
    """A column of NaN as long as the table (for quantities without errors)."""
    import pandas as pd
    return pd.Series(np.full(len(table), np.nan), index=table.index)


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
                       color=COLORS["broad"], ecolor=MUTED, elinewidth=0.7, capsize=0)
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


# Quasar main sequence (Sulentic et al. 2000, 2002; Marziani et al. 2018): Population A below
# FWHM(Hb) = 4000 km/s, spectral types in bins of R_FeII = 0.5 (A1-A4) and of FWHM = 4000 km/s (B1, B1+, B1++).
MS_POP_BOUNDARY = 4000.0
MS_TYPES = [("A1", 0.25, 2000), ("A2", 0.75, 2000), ("A3", 1.25, 2000), ("A4", 1.75, 2000),
            ("B1", 0.25, 6000), ("B2", 0.75, 6000), ("B1+", 0.25, 10000), ("B1++", 0.25, 14000)]


def main_sequence_grid():
    """Lines (x, y) of the main-sequence spectral-type grid in the R_FeII - FWHM(Hb) plane."""
    b = MS_POP_BOUNDARY
    return [([0, 3], [b, b]), ([0.5, 0.5], [0, 16000]), ([1.0, 1.0], [0, 8000]), ([1.5, 1.5], [0, b]),
            ([2.0, 2.0], [0, b]), ([0, 1.0], [8000, 8000]), ([0, 0.5], [12000, 12000])]


def _ms_plane(ax):
    """The main-sequence grid, spectral types and populations."""
    for x, y in main_sequence_grid():
        ax.plot(x, y, color="0.55", lw=0.6, ls="-" if y[0] == y[1] == MS_POP_BOUNDARY else ":", zorder=1)
    for name, x, y in MS_TYPES:
        ax.text(x, y, name, ha="center", va="center", fontsize=7, color="0.45", zorder=1, clip_on=True)
    ax.text(2.9, MS_POP_BOUNDARY * 0.5, "Pop. A", ha="right", va="center", fontsize=7, color="0.3", clip_on=True)
    ax.text(2.9, MS_POP_BOUNDARY * 1.5, "Pop. B", ha="right", va="center", fontsize=7, color="0.3", clip_on=True)


def plot_main_sequence(results, labels=None, highlight=None, mjd=None, zoom="auto", width=None, ax=None,
                       fig=None, style=True):
    """
    The quasar main sequence (optical plane of Eigenvector 1; Boroson & Green 1992,
    Sulentic et al. 2000, Marziani et al. 2018): FWHM of broad H-beta against
    R_FeII = F(Fe II 4434-4684) / F(broad H-beta), with the Population A/B boundary
    (4000 km/s) and the spectral types A1-A4, B1, B2, B1+, B1++.

    Several epochs are coloured by date (``mjd``, default: the spectra's MJDs) and
    joined in time order; ``zoom`` adds a panel on the region of the points (``'auto'``:
    when they cover less than a quarter of the plane). Text labels are drawn only for
    up to 10 points.

    :param results: a FitResult, a list of them (e.g. the epochs of a multi-epoch
        fit, ``me.epochs``) or a table with columns r_feii, r_feii_err, fwhm, fwhm_err
        (and mjd).
    :param highlight: index of the point to emphasise.
    """
    import pandas as pd
    if isinstance(results, pd.DataFrame):
        t = results.reset_index(drop=True)
    else:
        rs = results if isinstance(results, (list, tuple)) else [results]
        t = pd.DataFrame([r.main_sequence() for r in rs])
        if mjd is None:
            mjd = [getattr(r.spectrum, "mjd", None) for r in rs]
    if mjd is not None:
        t["mjd"] = pd.to_numeric(pd.Series(list(mjd), dtype=object), errors="coerce").to_numpy(float)
    ok = (np.isfinite(t.r_feii) & np.isfinite(t.fwhm)).to_numpy()
    many = len(t) > 1
    timed = many and "mjd" in t and np.isfinite(t.mjd[ok]).sum() > 1 and t.mjd[ok].nunique() > 1
    xe, ye = t.r_feii_err.fillna(0).to_numpy(), t.fwhm_err.fillna(0).to_numpy()
    x, y = t.r_feii.to_numpy(), t.fwhm.to_numpy()
    full = (0.0, max(3.0, 1.05 * np.nanmax(np.r_[3.0, (x + xe)[ok]])),
            0.0, max(16000.0, 1.05 * np.nanmax(np.r_[16000.0, (y + ye)[ok]])))
    box = None
    if ok.sum() > 1:
        x0, x1 = np.nanmin((x - xe)[ok]), np.nanmax((x + xe)[ok])
        y0, y1 = np.nanmin((y - ye)[ok]), np.nanmax((y + ye)[ok])
        dx, dy = max(x1 - x0, 0.05), max(y1 - y0, 300.0)
        box = (max(x0 - 0.15 * dx, 0), x1 + 0.15 * dx, max(y0 - 0.15 * dy, 0), y1 + 0.15 * dy)
    if zoom == "auto":
        zoom = box is not None and (box[1] - box[0]) / (full[1] - full[0]) * (box[3] - box[2]) / (full[3] - full[2]) < 0.25
    with paper_style(style):
        if ax is None:
            w = width or ("double" if zoom else "single")
            fig = new_figure(fig, _figsize(w, 0.42 if zoom else 0.85))
            axes = fig.subplots(1, 2, gridspec_kw={"wspace": 0.3}) if zoom else [fig.subplots()]
        else:
            fig, axes, zoom = ax.figure, [ax], False
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("ms_time", ["#9ecae1", "#2171b5", "#08306b"])
        norm = matplotlib.colors.Normalize(np.nanmin(t.mjd[ok]), np.nanmax(t.mjd[ok])) if timed else None
        sc = None
        for k, a in enumerate(axes):
            _ms_plane(a)
            if many and ok.sum() > 1:
                order = np.argsort(t.mjd.to_numpy()) if timed else np.arange(len(t))
                order = [i for i in order if ok[i]]
                a.plot(x[order], y[order], color="0.6", lw=0.5, zorder=2)        # the path in time
            a.errorbar(x[ok], y[ok], xerr=xe[ok], yerr=ye[ok], fmt="none", ecolor="0.6", elinewidth=0.5, zorder=2)
            if timed:
                sc = a.scatter(x[ok], y[ok], c=t.mjd[ok], cmap=cmap, norm=norm, s=16 if k else 12,
                               edgecolors="none", zorder=3)
            else:
                a.scatter(x[ok], y[ok], color=COLORS["broad"] if many else COLORS["model"], s=16, zorder=3)
            if highlight is not None and 0 <= highlight < len(t) and ok[highlight]:
                a.scatter([x[highlight]], [y[highlight]], s=46, color=COLORS["model"], edgecolors="k",
                          linewidths=0.7, zorder=4)
            if labels is not None and ok.sum() <= 10:
                for i in np.nonzero(ok)[0]:
                    a.annotate(str(labels[i]), (x[i], y[i]), xytext=(3, 3), textcoords="offset points",
                               fontsize=6, color="0.3")
            a.set_xlabel("$R_{\\rm Fe\\,II}$ = F(Fe$\\,$II 4434$-$4684) / F(H$\\beta$)")
        axes[0].set_xlim(full[0], full[1])
        axes[0].set_ylim(full[2], full[3])
        axes[0].set_ylabel("FWHM(H$\\beta$) (km$\\,$s$^{-1}$)")
        if zoom:
            z = axes[1]
            z.set_xlim(box[0], box[1])
            z.set_ylim(box[2], box[3])
            z.set_title("zoom", fontsize=8)
            axes[0].add_patch(matplotlib.patches.Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2],
                                                           fill=False, ls="--", lw=0.6, ec="0.3", zorder=5))
        if sc is not None:
            cb = fig.colorbar(sc, ax=axes[-1], pad=0.02)
            cb.set_label("MJD")
    return fig


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
