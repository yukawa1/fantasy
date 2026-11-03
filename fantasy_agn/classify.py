"""
Spectral classification before fitting: is there a broad-line region, how
much of the light is starlight, and what ionises the narrow-line gas?

``fa.classify(spec)`` runs three quick fits on a rest-frame spectrum:

1. Stellar continuum: stellar-population templates plus a featureless AGN
   power law (alpha_lambda = -1.7), with the emission lines masked. Gives the
   host fraction at 5100 A and the stellar absorption that must be removed
   before line ratios are measured (Balmer absorption under H-beta).
2. Emission lines on the starlight-subtracted spectrum, twice: narrow lines
   (with an outflow wing) only, and with broad H-beta/H-alpha added. A broad
   line counts when it lowers the BIC by more than ``broad_dbic`` (25), its
   flux has S/N > ``broad_snr`` (5), its equivalent width exceeds
   ``broad_ew`` (30 A) and its FWHM is 1200-12000 km/s. Without a broad-line
   region the "broad" component turns into a shallow, very wide bump that
   absorbs residuals of the stellar subtraction. The thresholds come from 300
   SDSS quasars at z < 0.35 and 650 narrow-line galaxies (MPA-JHU classes):
   79% of the quasars and 1.4% of the galaxies pass (calibration/ in the
   repository). The lines are fitted twice with a broad component: once with
   the outflow wing on the Balmer and forbidden lines, once with the wing on
   the forbidden lines only and a broad core plus a very broad wing; the
   better fit (BIC) is kept and the test uses the whole broad profile, so that
   the outflow wing cannot take the broad core of a Seyfert 1. Spurious bumps are wide (median FWHM 11000 km/s) and weak
   (EW 9 A, 99%: 42 A); real broad lines have FWHM 2300-9500 km/s and EW
   14-210 A (10-90%). The broad flux S/N is a poor criterion on its own: the
   broad line trades flux with the outflow wing and the continuum, so real
   broad lines often have S/N < 20. Candidates that miss only the EW or FWHM
   condition do not change the class but are reported as a possible weak broad
   line (type 1.8/1.9).
3. Narrow-line diagnostics: the BPT diagram [OIII]/Hb vs [NII]/Ha (Kauffmann
   et al. 2003; Kewley et al. 2001), the [SII] and [OI] diagrams with the
   Seyfert/LINER lines of Kewley et al. (2006), and the WHAN diagram
   (Cid Fernandes et al. 2011) for weak-line galaxies.

The stellar fit includes a dust screen on the starlight: without it the
template sum cannot follow the continuum shape and the error at H-beta (a few
per cent of the continuum) is comparable to weak H-beta emission.

Classes: 'type1', 'type1_host' (stellar absorption detected: sigma* S/N >= 5,
or host >= 50% at 5100 A; the recommended model then includes the host),
'seyfert2', 'liner',
'composite', 'starforming', 'retired' (EW(Ha) 0.5-3 A, ionised by old
stars), 'passive' (EW(Ha) < 0.5 A) and 'unknown'. ``result.model()`` returns
the recommended model for :func:`fantasy_agn.fit`.

The classification is only as good as the data: it needs H-alpha for the
BPT/WHAN diagrams (z < ~0.4 for SDSS/DESI) and reports ``uncertain=True``
when a ratio is within its error of a boundary or a line is not detected.
"""
import warnings

import numpy as np
import pandas as pd

from .components import Lines, Polynomial, PowerLaw
from .fitting import fit
from .lines import line_wavelength
from .model import Model

__all__ = ("classify", "Classification", "kauffmann03", "kewley01", "kewley06")

C_KMS = 299792.458
PASSIVE_EW, RETIRED_EW = 0.5, 3.0
BROAD_FWHM_MAX = 15000.0          # fit limit of the broad component
BROAD_FWHM_RANGE = (1200.0, 12000.0)   # accepted as a broad-line region
# The outflow wing must be broader than the narrow core, otherwise the two can
# split into a "Balmer" and a "forbidden" system and bias the line ratios.
NARROW_FWHM_MAX = 800.0
WEAK_BROAD_SNR, WEAK_BROAD_FWHM_MAX = 5.0, 10000.0

# Lines masked for the stellar fit, with half-widths (km/s). The first pass
# uses narrow masks, so that the stellar Balmer absorption troughs (which are
# much wider than the emission lines) are constrained by the data; when broad
# lines are found the stellar fit is repeated with the wide masks.
_LINES_MASKED = ["Hb", "Ha", "Hg", "Hd", "He", "HeII4686", "HeI5876", "[OII]3726", "[OII]3729",
                 "[NeIII]3869", "[NeIII]3968", "[OIII]4363", "[OIII]4959", "[OIII]5007", "[NI]5200",
                 "[OI]6300", "[OI]6364", "[NII]6548", "[NII]6583", "[SII]6716", "[SII]6731"]
_MASK_NARROW = {ln: 700 for ln in _LINES_MASKED}
_MASK_NARROW.update({"[OIII]4959": 1200, "[OIII]5007": 1200})       # outflow wings
_MASK_BROAD = dict(_MASK_NARROW)
_MASK_BROAD.update({"Hb": 4000, "Ha": 4000, "Hg": 3000, "Hd": 2500, "He": 1500, "HeII4686": 2500,
                    "HeI5876": 2500})

DIAG_LINES = ["Hb", "[OIII]4959", "[OIII]5007", "[OI]6300", "[OI]6364", "[NII]6548", "Ha",
              "[NII]6583", "[SII]6716", "[SII]6731"]


# ------------------------------------------------------------ boundaries
def kauffmann03(x):
    """Star-forming upper boundary in the [NII] BPT diagram (x < 0.05)."""
    return 0.61 / (x - 0.05) + 1.30


def kewley01(x, diagram="nii"):
    """Maximum-starburst line (x < 0.47, 0.32, -0.59 for nii, sii, oi)."""
    if diagram == "nii":
        return 0.61 / (x - 0.47) + 1.19
    if diagram == "sii":
        return 0.72 / (x - 0.32) + 1.30
    return 0.73 / (x + 0.59) + 1.33


def kewley06(x, diagram="sii"):
    """Seyfert/LINER separation in the [SII] or [OI] diagram."""
    return 1.89 * x + 0.76 if diagram == "sii" else 1.18 * x + 1.30


_XMAX = {"nii": 0.47, "sii": 0.32, "oi": -0.59}


def _above_kewley(x, y, diagram):
    return x >= _XMAX[diagram] or y > kewley01(x, diagram)


# ----------------------------------------------------------------- fits
def _stellar_fit(spec, wave_range, masks, templates="emiles"):
    from .host import StellarPopulation
    ranges = []
    for ln, dv in masks.items():
        lam = line_wavelength(ln)
        ranges.append((lam * (1 - dv / C_KMS), lam * (1 + dv / C_KMS)))
    model = Model(PowerLaw("continuum", pivot=5100), StellarPopulation(templates=templates, reddening=True))
    model.fix("continuum.index", -1.7)
    smooth = Model(PowerLaw("continuum", pivot=5100), Polynomial("poly", order=3, pivot=5100))
    masked = spec.mask(*ranges)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = fit(masked, model, wave_range=wave_range, verbose=False, build_up=False)
        r_smooth = fit(masked, smooth, wave_range=wave_range, verbose=False, build_up=False)
    return r, r_smooth


def _line_model(lines, broad, balmer_wing=True, broad_wing=False):
    narrow = [ln for ln in DIAG_LINES if ln in lines]
    ratios = {a: b for a, b in {"[OIII]4959": ("[OIII]5007", 1 / 2.98),
                                "[NII]6548": ("[NII]6583", 1 / 3.05),
                                "[OI]6364": ("[OI]6300", 1 / 3.0)}.items()
              if a in narrow and b[0] in narrow}
    m = Model(Polynomial("poly", order=2, pivot=5800, scale=1000),
              Lines("narrow", narrow, role="narrow", ratios=ratios,
                    fwhm=(300, 50, NARROW_FWHM_MAX), shift=(0, -600, 600)))
    wing = [ln for ln in ("Hb", "[OIII]4959", "[OIII]5007", "[NII]6548", "Ha", "[NII]6583")
            if ln in narrow and (balmer_wing or ln not in ("Hb", "Ha"))]
    m.add(Lines("outflow", wing, role="outflow",
                ratios={a: b for a, b in ratios.items() if a in wing and b[0] in wing},
                fwhm=(1100, NARROW_FWHM_MAX, 2500), shift=(-200, -1500, 500)))
    if broad:
        bl = [ln for ln in ("Hb", "Ha") if ln in narrow]
        if broad_wing:                      # broad core + very broad wing; the core is tested
            m.add(Lines("broad", bl, role="broad", tie_kinematics=True,
                        fwhm=(3000, 1200, 7000), shift=(0, -3000, 3000)))
            m.add(Lines("broad_wing", bl, role="broad", tie_kinematics=True,
                        fwhm=(9000, 5000, BROAD_FWHM_MAX), shift=(0, -3000, 3000)))
        else:
            m.add(Lines("broad", bl, role="broad", tie_kinematics=True,
                        fwhm=(3000, 1200, BROAD_FWHM_MAX), shift=(0, -3000, 3000)))
    return m


def _fluxes(r):
    """Narrow (narrow + outflow) flux and error of every diagnostic line."""
    t = r.lines
    t = t[t.component.isin(["narrow", "outflow"])]
    out = {}
    for ln, g in t.groupby("line"):
        err = np.sqrt(np.nansum(np.where(np.isfinite(g.flux_err), g.flux_err, 0) ** 2))
        out[ln] = (float(g.flux.sum()), float(err))
    return out


# --------------------------------------------------------------- result
class Classification:
    """
    Result of :func:`classify`: ``label``, ``uncertain``, ``reason``,
    ``measurements`` (dict), ``model()`` (recommended Model) and ``plot()``.
    """

    RECOMMENDED = {"type1": ("optical", {}), "type1_host": ("optical", {"host": True}),
                   "seyfert2": ("seyfert2", {}), "liner": ("liner", {}),
                   "composite": ("composite", {}), "starforming": ("starforming", {}),
                   "retired": ("passive", {}), "passive": ("passive", {}),
                   "unknown": ("type2", {})}

    def __init__(self, spectrum, label, uncertain, reason, measurements, stellar, lines,
                 narrow_fit=None, broad_fit=None, masks=None):
        self.spectrum = spectrum
        self.label = label
        self.uncertain = uncertain
        self.reason = reason
        self.measurements = measurements
        self.stellar_fit, self.line_fit = stellar, lines
        # both line fits of the broad-line test and the masks of the stellar fit
        self.narrow_fit, self.broad_fit, self.masks = narrow_fit, broad_fit, masks

    def model(self, **options):
        """Recommended model, e.g. ``fa.fit(spec, cls.model())``."""
        from . import presets
        name, kw = self.RECOMMENDED[self.label]
        return getattr(presets, name)(**{**kw, **options})

    @property
    def table(self):
        return pd.Series(self.measurements, name=self.spectrum.name)

    def __repr__(self):
        m = self.measurements
        flag = " (uncertain)" if self.uncertain else ""
        parts = [f"{self.spectrum.name}: {self.label}{flag}", f"  {self.reason}"]
        if np.isfinite(m.get("host_fraction_5100", np.nan)):
            parts.append(f"  host fraction at 5100 A = {m['host_fraction_5100']:.2f}, "
                         f"sigma* = {m['sigma_star']:.0f} km/s")
        if m.get("broad_dbic") is not None:
            parts.append(f"  broad lines: dBIC = {m['broad_dbic']:.0f}, FWHM = {m['broad_fwhm']:.0f} "
                         f"km/s, S/N = {m['broad_snr']:.1f}")
        for k, label in (("log_nii_ha", "log [NII]/Ha"), ("log_oiii_hb", "log [OIII]/Hb"),
                         ("log_sii_ha", "log [SII]/Ha"), ("log_oi_ha", "log [OI]/Ha")):
            if np.isfinite(m.get(k, np.nan)):
                parts.append(f"  {label} = {m[k]:+.2f} +- {m[k + '_err']:.2f}")
        if np.isfinite(m.get("ew_ha", np.nan)):
            parts.append(f"  EW(Ha) = {m['ew_ha']:.1f} A")
        if "ebv" not in self.spectrum.meta:
            parts.append("  note: Galactic extinction not corrected (spec.deredden()); line ratios "
                         "of nearby lines are hardly affected")
        parts.append(f"  recommended: fa.presets.{self.RECOMMENDED[self.label][0]}"
                     f"({', '.join(f'{k}={v}' for k, v in self.RECOMMENDED[self.label][1].items())})")
        return "\n".join(parts)

    def plot_steps(self, fig=None, save=None):
        """
        What :func:`classify` did: the spectrum with the masked lines and the
        stellar fit, H-alpha in the starlight-subtracted spectrum with the
        narrow-only and the narrow+broad fit, and the [NII] BPT diagram.
        """
        from .plotting import COLORS, flux_label, new_figure, paper_style, savefig
        s, m = self.spectrum, self.measurements
        cont = self.stellar_fit.model_flux(s.wave)
        with paper_style():
            fig = new_figure(fig, figsize=(7.1, 5.0))
            gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.38, wspace=0.28)
            ax = fig.add_subplot(gs[0, :])
            lo, hi = self.stellar_fit.spectrum.wave[self.stellar_fit.mask][[0, -1]]
            v = (s.wave >= lo) & (s.wave <= hi)
            ax.plot(s.wave[v], s.flux[v], color="0.55", lw=0.5, label="data")
            for i, (a, b) in enumerate(self.masks or []):
                ax.axvspan(a, b, color="#56B4E9", alpha=0.18, lw=0,
                           label="masked lines" if i == 0 else None)
            ax.plot(s.wave[v], cont[v], color=COLORS.get("host", "#999933"), lw=1.0,
                    label="stars + power law")
            ax.set_xlim(lo, hi)
            ax.set_ylim(0, np.nanpercentile(s.flux[v], 99.7) * 1.1)
            ax.set_xlabel(r"Rest-frame wavelength ($\mathrm{\AA}$)")
            ax.set_ylabel(flux_label(s).replace(" (", "\n(", 1))
            hf = m.get("host_fraction_5100", np.nan)
            ax.set_title(f"1 \u00b7 starlight: host {hf:.0%} at 5100 \u00c5, "
                         f"\u03c3$_\\star$ = {m['sigma_star']:.0f} km/s", fontsize=9)
            ax.legend(loc="upper right", fontsize=7, ncol=3)
            ax2 = fig.add_subplot(gs[1, 0])
            w = (s.wave > 6400) & (s.wave < 6740)
            if w.sum() > 10 and self.narrow_fit is not None:
                res = s.flux - cont
                ax2.plot(s.wave[w], res[w], color="0.35", lw=0.6, drawstyle="steps-mid")
                ax2.plot(s.wave[w], self.narrow_fit.model_flux(s.wave)[w], color="#0072B2", lw=1.0,
                         label="narrow only")
                ax2.plot(s.wave[w], self.broad_fit.model_flux(s.wave)[w], color="#D55E00", lw=1.0,
                         label="+ broad")
                ax2.legend(fontsize=7, loc="upper left")
                ax2.set_xlim(6400, 6740)
            ax2.set_xlabel(r"Rest-frame wavelength ($\mathrm{\AA}$)")
            dec = "yes" if self.label.startswith("type1") else ("weak?" if m.get("weak_broad") else "no")
            ax2.set_title(f"2 \u00b7 broad H\u03b1? {dec}  (\u0394BIC = {m['broad_dbic']:.0f}, "
                          f"S/N = {m['broad_snr']:.0f})", fontsize=9)
            ax3 = fig.add_subplot(gs[1, 1])
            x = np.linspace(-2.0, 0.46, 200)
            ax3.plot(x, kewley01(x), "k", lw=0.8)
            xk = np.linspace(-2.0, 0.04, 200)
            ax3.plot(xk, kauffmann03(xk), "k--", lw=0.8)
            for xx, yy, t in ((-1.3, -0.7, "SF"), (-0.3, -0.85, "comp."), (-0.9, 1.2, "Seyfert"),
                              (0.12, -0.7, "LINER")):
                ax3.text(xx, yy, t, fontsize=7, color="0.4")
            if np.isfinite(m.get("log_nii_ha", np.nan)) and np.isfinite(m.get("log_oiii_hb", np.nan)):
                ax3.errorbar(m["log_nii_ha"], m["log_oiii_hb"], xerr=m["log_nii_ha_err"],
                             yerr=m["log_oiii_hb_err"], fmt="*", ms=9, color="#D55E00")
            ax3.set_xlim(-1.6, 0.6)
            ax3.set_ylim(-1.0, 1.5)
            ax3.set_xlabel(r"log [N$\,$II]/H$\alpha$")
            ax3.set_ylabel(r"log [O$\,$III]/H$\beta$")
            ax3.set_title(f"3 \u00b7 {self.label.replace('_', ' ')}"
                          + (" (uncertain)" if self.uncertain else ""), fontsize=9)
        if save:
            savefig(fig, save)
        return fig

    def plot(self, fig=None, save=None, others=None, highlight=True):
        """
        BPT ([NII], [SII], [OI]) and WHAN diagrams with this object; ``others``
        (e.g. the classifications of all epochs) are drawn too, this one
        highlighted (``highlight=False``: all alike, as for a paper). The axes
        widen to show every point.
        """
        from .plotting import COLORS, new_figure, paper_style, savefig
        m = self.measurements
        group = [c for c in (others or []) if c is not self]
        alike = bool(group) and not highlight
        col_other, col_self = (COLORS["broad"], COLORS["broad"]) if alike else (COLORS["data"], COLORS["feii"])
        ms_self = 3 if alike else 4

        def widen(ax, xlim, ylim, xs, ys):
            xs, ys = np.asarray(xs, float), np.asarray(ys, float)
            ok = np.isfinite(xs) & np.isfinite(ys)
            if ok.any():
                xlim = (min(xlim[0], xs[ok].min() - 0.1), max(xlim[1], xs[ok].max() + 0.1))
                ylim = (min(ylim[0], ys[ok].min() - 0.15), max(ylim[1], ys[ok].max() + 0.15))
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

        def get(c, key):
            v = c.measurements.get(key, np.nan)
            return float(v) if isinstance(v, (int, float, np.floating)) else np.nan
        with paper_style():
            fig = new_figure(fig, figsize=(7.1, 2.1))
            axes = fig.subplots(1, 4)
            panels = [("nii", "log_nii_ha", r"log [N$\,$II]/H$\alpha$"),
                      ("sii", "log_sii_ha", r"log [S$\,$II]/H$\alpha$"),
                      ("oi", "log_oi_ha", r"log [O$\,$I]/H$\alpha$")]
            for ax, (d, key, xl) in zip(axes, panels):
                x = np.linspace(-3.5, _XMAX[d] - 0.01, 300)
                ax.plot(x, kewley01(x, d), color="k", lw=0.8)
                if d == "nii":
                    xk = np.linspace(-3.5, 0.04, 300)
                    ax.plot(xk, kauffmann03(xk), color="k", lw=0.8, ls="--")
                else:
                    # Seyfert/LINER line, drawn where it lies above the Kewley (2001) curve
                    xs = np.linspace(-3.5, 2.0, 600)
                    above = (xs >= _XMAX[d]) | (kewley06(xs, d) > kewley01(np.minimum(xs, _XMAX[d] - 1e-3), d))
                    ax.plot(xs[above], kewley06(xs[above], d), color="k", lw=0.8, ls=":")
                for c in group:
                    ax.errorbar(get(c, key), get(c, "log_oiii_hb"), xerr=get(c, key + "_err"),
                                yerr=get(c, "log_oiii_hb_err"), fmt="o", ms=3, color=col_other, lw=0.6, zorder=2)
                if np.isfinite(m.get(key, np.nan)) and np.isfinite(m.get("log_oiii_hb", np.nan)):
                    ax.errorbar(m[key], m["log_oiii_hb"], xerr=m[key + "_err"],
                                yerr=m["log_oiii_hb_err"], fmt="o", ms=ms_self, color=col_self, zorder=3)
                ax.set_xlabel(xl)
                widen(ax, {"nii": (-1.6, 0.6), "sii": (-1.3, 0.5), "oi": (-2.3, 0.0)}[d], (-1.2, 1.5),
                      [get(c, key) for c in group + [self]], [get(c, "log_oiii_hb") for c in group + [self]])
            axes[0].set_ylabel(r"log [O$\,$III]/H$\beta$")
            ax = axes[3]
            ax.axhline(np.log10(RETIRED_EW), color="k", lw=0.8)
            ax.axhline(np.log10(PASSIVE_EW), color="k", lw=0.8, ls=":")
            ax.plot([-0.4, -0.4], [np.log10(RETIRED_EW), 5], color="k", lw=0.8)
            ax.plot([-0.4, 2.0], [np.log10(6), np.log10(6)], color="k", lw=0.8, ls="--")

            def whan(c):
                ew = get(c, "ew_ha")
                return get(c, "log_nii_ha"), (np.log10(ew) if ew > 0 else np.nan)
            for c in group:
                x, y = whan(c)
                ax.errorbar(x, y, xerr=get(c, "log_nii_ha_err"), fmt="o", ms=3, color=col_other, lw=0.6, zorder=2)
            if np.isfinite(m.get("log_nii_ha", np.nan)) and m.get("ew_ha", 0) > 0:
                ax.errorbar(m["log_nii_ha"], np.log10(m["ew_ha"]), xerr=m["log_nii_ha_err"],
                            fmt="o", ms=ms_self, color=col_self, zorder=3)
            pts = [whan(c) for c in group + [self]]
            widen(ax, (-1.6, 0.8), (-1.0, 3.0), [p[0] for p in pts], [p[1] for p in pts])
            ax.set_xlabel(r"log [N$\,$II]/H$\alpha$")
            ax.set_ylabel(r"log EW(H$\alpha$) [$\mathrm{\AA}$]")
            for x, y, t in ((-1.5, 2.6, "SF"), (0.1, 2.6, "sAGN"), (0.1, 0.62, "wAGN"),
                            (-1.5, 0.05, "retired"), (-1.5, -0.8, "passive")):
                ax.text(x, y, t, fontsize=6)
            label = self.label + (" (uncertain)" if self.uncertain else "")
            if alike:
                labels = sorted({c.label for c in group + [self]})
                fig.suptitle(f"{len(group) + 1} spectra: {', '.join(labels)}", fontsize=9, y=1.04)
            else:
                extra = (f"  (grey: the other {len(group)} spectr{'um' if len(group) == 1 else 'a'})" if group else "")
                fig.suptitle(f"{self.spectrum.name}: {label}{extra}", fontsize=9, y=1.04)
            fig.tight_layout(w_pad=0.6)
        if save:
            savefig(fig, save)
        return fig


# ------------------------------------------------------------- classify
def classify(spectrum, wave_range=(3700, 6900), broad_dbic=25.0, broad_snr=5.0, broad_ew=30.0,
             min_snr=3.0, templates="emiles", verbose=False):
    """
    Classify a rest-frame spectrum before choosing a model (see the module
    documentation for the procedure).

    :param spectrum: Spectrum; moved to the rest frame if it is not.
    :param wave_range: range for the stellar continuum fit (clipped to the data).
    :param broad_dbic: minimum BIC improvement for broad lines.
    :param broad_snr: minimum S/N of the broad H-alpha (or H-beta) flux.
    :param broad_ew: minimum equivalent width (A) of the broad line
        (thresholds: see the module documentation).
    :param min_snr: minimum S/N for a narrow line to enter a diagnostic.
    :param templates: stellar population templates for the starlight ('emiles',
        'bc03', ...; see fantasy_agn.templates).
    :return: Classification
    """
    spec = spectrum if spectrum.rest_frame else spectrum.to_rest_frame()
    good = spec.good
    lo = max(wave_range[0], float(spec.wave[good].min()) + 5)
    hi = min(wave_range[1], float(spec.wave[good].max()) - 5)
    meas = {}

    lines = [ln for ln in DIAG_LINES if lo < line_wavelength(ln) < hi]
    lrange = (max(lo, 4750.0), min(hi, 6800.0))

    def stages(masks):
        stellar, smooth = _stellar_fit(spec, (lo, hi), masks, templates)
        cont = stellar.model_flux(spec.wave)
        resid = spec.copy(flux=spec.flux - cont)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_n = fit(resid, _line_model(lines, False), wave_range=lrange, verbose=False)
            r_b = fit(resid, _line_model(lines, True), wave_range=lrange, verbose=False)
            # In Seyfert 1s with broad lines of ~3000 km/s the Balmer part of the outflow wing (FWHM up to
            # 2500 km/s) can take the broad line, leaving the broad component only the far wings (very wide,
            # no H-beta), which fails the broad-line test. A model with the wing on the forbidden lines only,
            # and a broad core plus a very broad wing, is fitted too; the better of the two (BIC) is kept,
            # and the broad-line test is applied to the core.
            r_f = fit(resid, _line_model(lines, True, balmer_wing=False, broad_wing=True), wave_range=lrange,
                      verbose=False)
            if r_f.bic < r_b.bic:
                r_b = r_f
        return stellar, smooth, cont, r_n, r_b

    def broad_stats(r_n, r_b):
        # the whole broad profile (core, and the very broad wing when it is fitted): flux, its S/N, FWHM
        comps = [c.name for c in r_b.model.components if c.name in ("broad", "broad_wing")]
        bl = r_b.lines[r_b.lines.component == "broad"].set_index("line")
        bline = "Ha" if "Ha" in bl.index else ("Hb" if "Hb" in bl.index else None)
        if bline is None:
            return None, 0.0, 0.0, np.nan, np.nan
        m = r_b.measure(bline, comps)
        f, e = m["flux"], m["flux_err"]
        return (bline, float(r_n.bic - r_b.bic), float(f / e) if e > 0 else 0.0, float(m["fwhm"]), float(f))

    # 1.-2. stellar continuum (narrow masks) and emission lines
    masks = _MASK_NARROW
    stellar, smooth, cont, r_n, r_b = stages(masks)
    bline, dbic, snr, fwhm, bflux = broad_stats(r_n, r_b)
    if dbic > broad_dbic and snr > broad_snr and BROAD_FWHM_RANGE[0] < fwhm < BROAD_FWHM_RANGE[1]:
        # broad lines: repeat with wide masks so that their wings do not
        # leak into the stellar fit
        masks = _MASK_BROAD
        stellar, smooth, cont, r_n, r_b = stages(masks)
        bline, dbic, snr, fwhm, bflux = broad_stats(r_n, r_b)
    hp = stellar.host_properties()
    meas["host_fraction_5100"] = hp["host_fraction_5100"][0] if lo < 5100 < hi else np.nan
    meas["sigma_star"], meas["sigma_star_err"] = hp["sigma_star"]
    sig, sig_err = meas["sigma_star"], meas["sigma_star_err"]
    kin_at_limit = any(n.startswith("host.") and n.endswith((".sigma", ".velocity")) for n in stellar.at_limit)
    sig_snr = sig / sig_err if np.isfinite(sig_err) and sig_err > 0 and not kin_at_limit else 0.0
    meas["sigma_star_snr"] = float(sig_snr)
    host_frac = meas["host_fraction_5100"]
    meas["stellar_detected"] = bool(sig_snr >= 5 or (np.isfinite(host_frac) and host_frac >= 0.5))
    # some starlight but not confirmed: decide by fitting with and without the host
    # (on the calibration quasars this found the host for 95% of those that need it)
    meas["host_borderline"] = bool(not meas["stellar_detected"] and np.isfinite(host_frac)
                                   and host_frac >= 0.1)
    meas["stellar_dbic"] = float(smooth.bic - stellar.bic)
    meas["broad_dbic"], meas["broad_snr"], meas["broad_fwhm"] = dbic, snr, fwhm
    if bline is not None:
        meas["broad_flux_" + bline] = bflux
        c_b = float(np.interp(line_wavelength(bline), spec.wave, cont))
        meas["broad_ew"] = bflux / c_b if c_b > 0 else np.nan
    else:
        meas["broad_ew"] = 0.0
    width_ok = BROAD_FWHM_RANGE[0] < fwhm < BROAD_FWHM_RANGE[1]
    ew_b = meas["broad_ew"]
    broad = (dbic > broad_dbic and snr > broad_snr and width_ok
             and np.isfinite(ew_b) and ew_b > broad_ew)
    weak_broad = (not broad and dbic > broad_dbic and snr > WEAK_BROAD_SNR
                  and BROAD_FWHM_RANGE[0] < fwhm < WEAK_BROAD_FWHM_MAX)
    meas["weak_broad"] = bool(weak_broad)
    r = r_b if broad else r_n

    # 3. narrow-line diagnostics
    fl = _fluxes(r)

    def ratio(num, den):
        if num not in fl or den not in fl:
            return np.nan, np.nan, False
        (a, ea), (b, eb) = fl[num], fl[den]
        det = a > min_snr * ea and b > min_snr * eb
        if a <= 0 or b <= 0:
            return np.nan, np.nan, False
        return (float(np.log10(a / b)),
                float(np.hypot(ea / a, eb / b) / np.log(10)), det)

    def summed(names):
        names = [n for n in names if n in fl]
        if not names:
            return None
        return (sum(fl[n][0] for n in names), np.sqrt(sum(fl[n][1] ** 2 for n in names)))

    if summed(["[SII]6716", "[SII]6731"]):
        fl["[SII]"] = summed(["[SII]6716", "[SII]6731"])
    det = {}
    for key, num, den in (("log_nii_ha", "[NII]6583", "Ha"), ("log_oiii_hb", "[OIII]5007", "Hb"),
                          ("log_sii_ha", "[SII]", "Ha"), ("log_oi_ha", "[OI]6300", "Ha")):
        meas[key], meas[key + "_err"], det[key] = ratio(num, den)
    if "Ha" in fl:
        cont_ha = float(np.interp(line_wavelength("Ha"), spec.wave, cont))
        meas["ew_ha"] = fl["Ha"][0] / cont_ha if cont_ha > 0 else np.nan
        meas["ha_snr"] = fl["Ha"][0] / fl["Ha"][1] if fl["Ha"][1] > 0 else 0.0
    else:
        meas["ew_ha"], meas["ha_snr"] = np.nan, 0.0

    host = meas["host_fraction_5100"]
    uncertain = False
    if broad:
        # The host belongs in the model when the stellar absorption is detected. On 300
        # SDSS quasars at z < 0.35 (calibration/) this rule chose the host for 82% of
        # those whose fit it improves (BIC), and was right in 83% of the cases where it
        # chose it; host >= 50% alone found 65%, the literature's host >= 10% was right
        # in only 59%.
        label = "type1_host" if meas["stellar_detected"] else "type1"
        reason = (f"broad {bline} detected (FWHM {meas['broad_fwhm']:.0f} km/s, "
                  f"dBIC {dbic:.0f}, S/N {meas['broad_snr']:.0f})")
        if label == "type1" and meas["host_borderline"]:
            reason += (f"; possible starlight (host {host_frac:.0%} at 5100 A, sigma* S/N "
                       f"{meas['sigma_star_snr']:.1f}): compare fits with and without the host")
        if label == "type1_host":
            reason += (f"; stellar absorption detected (sigma* = {meas['sigma_star']:.0f} km/s, "
                       f"S/N {meas['sigma_star_snr']:.0f}): fit the host with the AGN")
        uncertain = dbic < 2 * broad_dbic or meas["broad_snr"] < 2 * broad_snr or ew_b < 1.5 * broad_ew
    else:
        ew = meas["ew_ha"]
        x, y = meas["log_nii_ha"], meas["log_oiii_hb"]
        ex, ey = meas["log_nii_ha_err"], meas["log_oiii_hb_err"]
        if not np.isfinite(ew) or ew < PASSIVE_EW or meas["ha_snr"] < min_snr:
            label, reason = "passive", f"no significant H-alpha emission (EW < {PASSIVE_EW} A)"
        elif ew < RETIRED_EW:
            label = "retired"
            reason = f"weak lines, EW(Ha) = {ew:.1f} A: ionised by old stars (WHAN)"
            uncertain = ew > RETIRED_EW - 0.5
        elif det["log_nii_ha"] and det["log_oiii_hb"]:
            if x < 0.05 and y < kauffmann03(x):
                label = "starforming"
                reason = "below the Kauffmann et al. (2003) line"
                near = min(abs(y - kauffmann03(x)), abs(x - 0.05) if x > -0.2 else 9)
            elif not _above_kewley(x, y, "nii"):
                label = "composite"
                reason = "between the Kauffmann (2003) and Kewley (2001) lines"
                near = min(abs(y - kauffmann03(x)) if x < 0.05 else 9,
                           abs(y - kewley01(x)) if x < 0.47 else 9)
            else:
                votes = []
                for d, k in (("sii", "log_sii_ha"), ("oi", "log_oi_ha")):
                    if det[k]:
                        votes.append("seyfert2" if y > kewley06(meas[k], d) else "liner")
                if not votes:
                    votes = ["seyfert2" if y > 0.5 else "liner"]
                label = max(set(votes), key=votes.count)
                if len(set(votes)) > 1:
                    uncertain = True
                reason = ("above the Kewley et al. (2001) line; Seyfert/LINER from the "
                          f"Kewley et al. (2006) {'/'.join(d for d, k in (('[SII]', 'log_sii_ha'), ('[OI]', 'log_oi_ha')) if det[k]) or '[OIII]/Hb'} diagram")
                near = abs(y - kewley01(x)) if x < 0.47 else 9
            uncertain = uncertain or near < np.hypot(ex, ey)
            if 3 <= ew < 6 and label == "seyfert2":
                reason += f"; EW(Ha) = {ew:.1f} A is low (weak AGN in WHAN)"
        elif det["log_nii_ha"]:
            # no H-beta/[OIII]: WHAN only
            if x < -0.4:
                label, reason = "starforming", "WHAN: log [NII]/Ha < -0.4"
            else:
                label = "seyfert2" if ew > 6 else "liner"
                reason = f"WHAN: {'strong' if ew > 6 else 'weak'} AGN (EW(Ha) = {ew:.1f} A)"
            uncertain = True
        else:
            label, reason, uncertain = "unknown", "lines too weak for the diagnostics", True
    if weak_broad:
        reason += (f"; possible weak broad {bline} (FWHM {meas['broad_fwhm']:.0f} km/s, "
                   f"S/N {meas['broad_snr']:.1f}): type 1.8/1.9?")
        uncertain = True
    mask_ranges = [(line_wavelength(ln) * (1 - dv / C_KMS), line_wavelength(ln) * (1 + dv / C_KMS))
                   for ln, dv in masks.items()]
    result = Classification(spec, label, bool(uncertain), reason, meas, stellar, r,
                            narrow_fit=r_n, broad_fit=r_b, masks=mask_ranges)
    if verbose:
        print(result)
    return result
