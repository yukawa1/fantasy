"""
Ready-made models for the most common AGN spectral regions.

Every preset returns an ordinary Model, so it can be inspected and changed::

    model = fa.presets.hbeta(broad_components=2, outflow=True)
    model.set("broad.fwhm", min=2000)

Each model remembers the wavelength window it is meant for
(``model.wave_range``), which :func:`fantasy_agn.fit` uses by default.
"""
from .components import (BalmerContinuum, BrokenPowerLaw, Extinction, FeII, HighOrderBalmer,
                         HostGalaxy, Lines, PowerLaw)
from .model import Model
import numpy as np

__all__ = ("hbeta", "halpha", "optical", "mgii", "civ", "type2", "galaxy", "seyfert2", "liner",
           "composite", "starforming", "passive", "retired", "available", "auto",
           "variants",
           "outflow_kinematics",
           "PRESETS")

NARROW = dict(fwhm=(400, 50, 1200), shift=(0, -600, 600))
# One broad Gaussian covers the whole profile (FWHM 1000-20000 km/s).
BROAD = dict(fwhm=(3500, 1000, 20000), shift=(0, -3000, 3000))
BROAD_FWHM = (1000.0, 30000.0)   # FWHM range shared out among several components
BROAD_OVERLAP = 1.3


def broad_fwhm_ranges(n, fwhm_max=None):
    """
    FWHM limits (start, min, max) for n broad components: 1000-30000 km/s (or
    up to ``fwhm_max``) is split into n logarithmic bands that overlap by a
    factor 1.3, so every component has its own width regime (core,
    intermediate, wings, ...) and two components cannot simply swap roles.
    """
    lo, hi = BROAD_FWHM[0], (fwhm_max or BROAD_FWHM[1])
    edges = lo * (hi / lo) ** (np.arange(n + 1) / n)
    out = []
    for k in range(n):
        a = max(lo, edges[k] / BROAD_OVERLAP)
        b = min(hi, edges[k + 1] * BROAD_OVERLAP)
        out.append((float(np.sqrt(a * b)), float(a), float(b)))
    return out
# Coronal lines: usually broader and blueshifted relative to the narrow lines.
CORONAL = dict(fwhm=(600, 150, 3000), shift=(0, -1000, 500))
CORONAL_HIGH = ("[FeX]6375", "[FeXIV]5303")
CORONAL_LINES = ["[FeVII]3759", "[FeVII]5159", "[FeXIV]5303", "[FeVII]5721", "[FeVII]6087",
                 "[FeX]6375"]
OUTFLOW = dict(fwhm=(900, 300, 2500), shift=(-300, -2000, 200))

OIII_RATIO = 1 / 2.98   # Storey & Zeippen 2000
NII_RATIO = 1 / 3.05
NEIII_RATIO = 1 / 3.32   # [NeIII]3968/3869
OI_RATIO = 1 / 3.0
NII_HA_MAX = 3.0            # upper limit of [NII]6583/H-alpha in an outflow wing
HA_OUTFLOW_FWHM = (900, 300, 2000)
PHYSICAL_SLOPE = -1.7          # alpha_lambda, host-free quasar continuum (Selsing+2016)
PHYSICAL_BROAD_FWHM_MAX = 15000.0
RED_END = 7500.0              # end of the optical window with red=True


def _continuum(kind, pivot, break_wave):
    """'powerlaw' (amplitude at pivot) or 'broken' (slopes change at break_wave)."""
    if kind == "powerlaw":
        return PowerLaw("continuum", pivot=pivot)
    if kind == "broken":
        return BrokenPowerLaw("continuum", pivot=break_wave)
    if kind == "broken_free":
        return BrokenPowerLaw("continuum", break_wave=(break_wave, 4000.0, 6000.0))
    raise ValueError("continuum must be 'powerlaw', 'broken' or 'broken_free'")


def _balmer(model):
    """Balmer continuum and high-order Balmer lines with the broad-line kinematics."""
    model.add(BalmerContinuum())
    model.add(HighOrderBalmer())
    model.tie("balmer_lines.fwhm", "broad.fwhm")
    model.tie("balmer_lines.shift", "broad.shift")


def _host(kind):
    """
    Host galaxy component: True/'stellar' (E-MILES stellar populations),
    a template set ('emiles', 'bc03' = GALAXEV, 'fsps', 'xsl'; see
    fantasy_agn.templates, downloaded once), or 'pca' (Yip+2004 eigenspectra).
    """
    if kind is True or kind == "stellar":
        from .host import StellarPopulation
        return StellarPopulation()
    if kind == "pca":
        return HostGalaxy()
    from .templates import TEMPLATE_SETS, canonical
    if isinstance(kind, str) and canonical(kind) in TEMPLATE_SETS:
        from .host import StellarPopulation
        return StellarPopulation(templates=canonical(kind))
    raise ValueError("host must be True/'stellar' (E-MILES), a template set ('emiles', 'bc03', "
                     "'fsps', 'xsl'), 'pca' or False")


def _coronal(model, wave_range):
    """
    Coronal lines inside the window: [FeVII] (ionisation potential 100 eV) and
    the higher-ionisation [FeX] and [FeXIV] (235 and 361 eV) each with their own
    shared kinematics, since coronal lines are often broader and more
    blueshifted with increasing ionisation potential.
    """
    from .lines import line_wavelength
    inside = [ln for ln in CORONAL_LINES if wave_range[0] < line_wavelength(ln) < wave_range[1]]
    low = [ln for ln in inside if ln not in CORONAL_HIGH]
    high = [ln for ln in inside if ln in CORONAL_HIGH]
    if low:
        model.add(Lines("coronal", low, role="narrow", **CORONAL))
    if high:
        model.add(Lines("coronal_high", high, role="narrow", **CORONAL))


def _broad(model, name, lines, n, profile="gaussian", fwhm_max=None):
    """
    n broad Gaussian components for the same lines, named name, name2, name3, ...
    Each has its own kinematics; the narrowest may shift by +-3000 km/s, the
    broader ones by +-5000 km/s.
    """
    if n < 1:
        raise ValueError("broad_components must be at least 1")
    if n == 1:
        kin = dict(BROAD)
        if fwhm_max:
            kin["fwhm"] = (min(kin["fwhm"][0], fwhm_max), kin["fwhm"][1], min(kin["fwhm"][2], fwhm_max))
        model.add(Lines(name, lines, role="broad", profile=profile, **kin))
        return model
    for k, fwhm in enumerate(broad_fwhm_ranges(n, fwhm_max)):
        shift = (0, -3000, 3000) if k == 0 else (0, -5000, 5000)
        model.add(Lines(name if k == 0 else f"{name}{k + 1}", lines, role="broad",
                        fwhm=fwhm, shift=shift, profile=profile))
    return model


def hbeta(broad_components=2, feii=True, outflow=False, host=False, heii=True,
          continuum="powerlaw", extinction=False, coronal=False,
          broad_profile="gaussian", wave_range=(4150, 5550)):
    """
    H-beta region (4150-5550 A): power law, optical FeII, narrow H-gamma,
    [OIII]4363, H-beta and [OIII]4959,5007 (fixed 1:2.98 ratio), broad
    H-gamma and H-beta with shared kinematics, narrow [NI]5198,5200, broad
    HeI4471 and HeII4686.

    :param broad_components: number of broad components (1, 2, 3, ...).
    :param broad_profile: 'gaussian', 'lorentzian', 'voigt' or 'gauss-hermite'.
        Compare ``result.bic`` between fits to decide how many the data support.
    :param outflow: add a blueshifted [OIII] wing component.
    :param host: fit a host galaxy simultaneously: True/'stellar' (stellar
        population templates with sigma* and v*) or 'pca' (Yip+2004 eigenspectra).
    :param continuum: 'powerlaw' or 'broken' (slope change at 4600 A).
    :param coronal: add the coronal lines [FeVII]5159 and [FeXIV]5303 (own
        kinematics), seen in many Seyferts.
    """
    m = Model(_continuum(continuum, 5100, 4600))
    if host:
        m.add(_host(host))
    if feii:
        m.add(FeII(fwhm=(2500, 1000, 8000), shift=(0, -2000, 2000)))
    m.add(Lines("narrow", ["Hg", "[OIII]4363", "Hb", "[OIII]4959", "[OIII]5007", "[NI]5198",
                           "[NI]5200"], role="narrow",
                ratios={"[OIII]4959": ("[OIII]5007", OIII_RATIO)}, **NARROW))
    if outflow:
        m.add(Lines("outflow", ["[OIII]4959", "[OIII]5007"], role="outflow",
                    ratios={"[OIII]4959": ("[OIII]5007", OIII_RATIO)}, **OUTFLOW))
    _broad(m, "broad", ["Hg", "Hb"], broad_components, broad_profile)
    if heii:
        m.add(Lines("heii", ["HeI4471", "HeII4686"], role="broad", fwhm=(4000, 1000, 15000),
                    shift=(0, -4000, 2000)))
    if coronal:
        _coronal(m, wave_range)
    if extinction:
        m.add(Extinction(law=extinction if isinstance(extinction, str) else "F99"))
    m.wave_range = wave_range
    return m


def halpha(broad_components=2, outflow=False, host=False, continuum="powerlaw",
           extinction=False, hei=True, coronal=False,
          broad_profile="gaussian", wave_range=(6200, 6900)):
    """
    H-alpha region (6200-6900 A): power law, narrow [OI]6300,6364,
    [NII]6548,6583 (1:3.05), H-alpha, HeI6678, [SII]6716,6731, broad H-alpha
    and broad HeI6678.

    :param outflow: add an outflow wing to [NII] and H-alpha:

        - True: free kinematics (FWHM <= 2000 km/s) and [NII]6583/H-alpha <= 3.
          In this window nothing else constrains the wing, which can then
          mimic structure in the broad line.
        - a FitResult of a fit with an [OIII] outflow (e.g. the hbeta preset
          with outflow=True), or a dict {'shift': km/s, 'fwhm': km/s}: the wing
          has the shift and FWHM of the [OIII] outflow, so it can only be the
          gas seen in [OIII]. Recommended.

    :param hei: add HeI6678, narrow (narrow-line kinematics) and broad
        (kinematics of the main broad H-alpha component, so it cannot absorb
        the red wing of H-alpha).
    :param coronal: add the coronal line [FeX]6375 (own kinematics).
    """
    m = Model(_continuum(continuum, 6500, 6500))
    if host:
        m.add(_host(host))
    narrow = ["[OI]6300", "[OI]6364", "[NII]6548", "Ha", "[NII]6583", "[SII]6716", "[SII]6731"]
    if hei:
        narrow.insert(5, "HeI6678")
    m.add(Lines("narrow", narrow, role="narrow", ratios={"[NII]6548": ("[NII]6583", NII_RATIO),
                                          "[OI]6364": ("[OI]6300", OI_RATIO)}, **NARROW))
    if outflow is not False and outflow is not None:
        kin = None if outflow is True else outflow_kinematics(outflow)
        fwhm = HA_OUTFLOW_FWHM if kin is None else (kin["fwhm"], min(300, kin["fwhm"]),
                                                     max(2000, kin["fwhm"]))
        shift = OUTFLOW["shift"] if kin is None else (kin["shift"], min(-2000, kin["shift"]),
                                                      max(200, kin["shift"]))
        # Without a limit the wing can put the broad H-alpha wing into [NII]:
        # keep [NII]6583/H-alpha <= 3 (the AGN/LINER range).
        m.add(Lines("outflow", ["[NII]6548", "Ha", "[NII]6583"], role="outflow",
                    ratios={"[NII]6548": ("[NII]6583", NII_RATIO)},
                    max_ratios={"[NII]6583": ("Ha", NII_HA_MAX)}, fwhm=fwhm, shift=shift))
        if kin is not None:
            m.fix("outflow.shift")
            m.fix("outflow.fwhm")
    _broad(m, "broad", ["Ha"], broad_components, broad_profile)
    if hei:
        m.add(Lines("broad_hei", ["HeI6678"], role="broad", **BROAD))
        m.tie("broad_hei.fwhm", "broad.fwhm")
        m.tie("broad_hei.shift", "broad.shift")
    if coronal:
        _coronal(m, wave_range)
    if extinction:
        m.add(Extinction(law=extinction if isinstance(extinction, str) else "F99"))
    m.wave_range = wave_range
    return m


def outflow_kinematics(source):
    """
    Shift and FWHM (km/s) of an outflow wing: from a FitResult with an
    'outflow' component (e.g. [OIII] in an H-beta fit) or a dict
    {'shift': ..., 'fwhm': ...}. Returns {'shift', 'fwhm', 'shift_err', 'fwhm_err'}.
    """
    if isinstance(source, dict):
        return {"shift": float(source["shift"]), "fwhm": float(source["fwhm"]),
                "shift_err": float(source.get("shift_err", np.nan)),
                "fwhm_err": float(source.get("fwhm_err", np.nan))}
    names = source.model.params.names
    if "outflow.shift" not in names or "outflow.fwhm" not in names:
        raise ValueError("the fit has no 'outflow' component with shared kinematics")
    sig = source._sigma()
    out = {}
    for k in ("shift", "fwhm"):
        i = source._index(f"outflow.{k}")
        out[k], out[k + "_err"] = float(source.full[i]), float(source._param_error(i, sig))
    return out


def optical(broad_components=2, feii=True, outflow=False, host=False, continuum="broken",
            balmer=True, extinction=False, coronal=False, oi6046=False, red=False,
          broad_profile="gaussian", wave_range=None, physical=False, slope=PHYSICAL_SLOPE,
          broad_fwhm_max=None):
    """
    Full optical range (3700-6900 A): broken power law, Balmer continuum and
    high-order Balmer lines, FeII, narrow lines from [OII]3727 to [SII]6731
    (including H8, [NeIII]3968 + H-epsilon, [SII]4069,4076, [NI]5198,5200 and
    HeI6678), broad H-delta/H-gamma/H-beta (shared kinematics), broad H-alpha
    (own kinematics) and broad HeI4471, HeII4686, HeI5876, HeI6678 (shared
    kinematics).

    :param coronal: add the coronal lines [FeVII]3759,5159,5721,6087, [FeXIV]5303
        and [FeX]6375 (shared kinematics of their own).
    :param oi6046: add broad OI6046 (strong in objects with strong FeII).
    :param red: extend the fit to 7500 A (if the spectrum covers it) with narrow
        HeI7065, [ArIII]7136 and [OII]7320,7330 and broad HeI7065.
    :param physical: physically motivated decomposition (after QSFit,
        Calderone et al. 2017, and the references in the documentation):

        - one AGN power law over the whole range with slope alpha_lambda =
          ``slope`` (default -1.7, Selsing et al. 2016; QSFit fixes it for
          z < 0.6, where it is degenerate with the host). ``slope=None`` fits
          it within the accretion-disc range -2.33 ... -1.0. No broken power
          law: the slope change near 5000 A is mostly host-galaxy light;
        - a stellar-population host galaxy (``host=True`` unless given);
        - the velocity-broadened Balmer continuum and high-order Balmer lines,
          with a window starting at 3550 A so that the Balmer edge is covered;
        - broad components no wider than 15000 km/s (wider "lines" trade flux
          with the continuum);
        - with ``outflow=True`` an [NII]/H-alpha wing with the kinematics of the
          [OIII] outflow.

        Check ``result.balmer_decrement()``: broad H-alpha/H-beta near 3
        (Dong et al. 2008) indicates a consistent decomposition.
    :param slope: AGN power-law slope for ``physical=True`` (None: fit it).
    :param broad_fwhm_max: upper FWHM limit of the broad components, km/s
        (default 30000; 15000 with physical=True).

    :param extinction: True (Fitzpatrick 1999) or 'calzetti': fit an intrinsic
        E(B-V) of the AGN; fluxes and luminosities are then dereddened.
    :param continuum: 'broken' (default; slope change at 4600 A), 'broken_free'
        (break wavelength fitted within 4000-6000 A) or 'powerlaw'. A single
        power law cannot follow the curvature of the continuum over this wide
        range. The continuum shape is the main systematic of this preset:
        compare several options (and host=True) with fa.compare.
    :param balmer: add the Balmer continuum and high-order Balmer lines, which
        shape the spectrum below ~4000 A.
    :param host: fit a host galaxy simultaneously: True/'stellar' (stellar
        population templates with sigma* and v*, cannot become negative) or
        'pca' (Yip+2004 eigenspectra; can come out negative, the summary warns).
    """
    if physical:
        continuum = "powerlaw"
        host = host or True
        balmer = True
        broad_fwhm_max = broad_fwhm_max or PHYSICAL_BROAD_FWHM_MAX
        wave_range = wave_range or (3550, RED_END if red else 6900)
    wave_range = wave_range or (3700, RED_END if red else 6900)
    m = Model(_continuum(continuum, 5100, 4600))
    if physical:
        if slope is None:
            m.set("continuum.index", value=PHYSICAL_SLOPE, min=-7 / 3, max=-1.0)
        else:
            m.fix("continuum.index", slope)
    if host:
        m.add(_host(host))
    if feii:
        m.add(FeII(fwhm=(2500, 1000, 8000), shift=(0, -2000, 2000)))
    narrow = ["[OII]3726", "[OII]3729", "[NeIII]3869", "H8", "[NeIII]3968", "He",
              "[SII]4069", "[SII]4076", "Hd", "Hg", "[OIII]4363", "HeII4686", "Hb",
              "[OIII]4959", "[OIII]5007", "[NI]5198", "[NI]5200", "HeI5876", "[OI]6300",
              "[OI]6364", "[NII]6548", "Ha", "[NII]6583", "HeI6678", "[SII]6716", "[SII]6731"]
    if red:
        narrow += ["HeI7065", "[ArIII]7136", "[OII]7320", "[OII]7330"]
    m.add(Lines("narrow", narrow, role="narrow", ratios={"[OIII]4959": ("[OIII]5007", OIII_RATIO),
                                          "[NII]6548": ("[NII]6583", NII_RATIO),
                                          "[OI]6364": ("[OI]6300", OI_RATIO),
                                          "[NeIII]3968": ("[NeIII]3869", NEIII_RATIO)},
                **NARROW))
    if outflow:
        # [NeIII] (ionisation potential 41 eV, like [OIII] 35 eV) shares the wing
        m.add(Lines("outflow", ["[NeIII]3869", "[OIII]4959", "[OIII]5007"], role="outflow",
                    ratios={"[OIII]4959": ("[OIII]5007", OIII_RATIO)}, **OUTFLOW))
    if outflow and physical:
        m.add(Lines("outflow_ha", ["[NII]6548", "Ha", "[NII]6583"], role="outflow",
                    ratios={"[NII]6548": ("[NII]6583", NII_RATIO)},
                    max_ratios={"[NII]6583": ("Ha", NII_HA_MAX)}))
        m.tie("outflow_ha.fwhm", "outflow.fwhm")
        m.tie("outflow_ha.shift", "outflow.shift")
    _broad(m, "broad", ["Hd", "Hg", "Hb"], broad_components, broad_profile, broad_fwhm_max)
    _broad(m, "broad_ha", ["Ha"], broad_components, broad_profile, broad_fwhm_max)
    m.add(Lines("broad_he", ["HeI4471", "HeII4686", "HeI5876", "HeI6678"] + (["HeI7065"] if red else []),
                role="broad",
                fwhm=(4000, 1000, 15000),
                shift=(0, -4000, 2000)))
    if oi6046:
        m.add(Lines("broad_oi", ["OI6046"], role="broad", **BROAD))
        m.tie("broad_oi.fwhm", "broad.fwhm")
        m.tie("broad_oi.shift", "broad.shift")
    if coronal:
        _coronal(m, wave_range)
    if balmer:
        _balmer(m)
    if extinction:
        m.add(Extinction(law=extinction if isinstance(extinction, str) else "F99"))
    m.wave_range = wave_range
    return m


def mgii(broad_components=1, feii=True, balmer_continuum=True, broad_profile="gaussian", wave_range=(2650, 3050)):
    """
    MgII region (2650-3050 A): power law, Balmer continuum, UV FeII and broad
    MgII2798 (doublet blend).
    """
    m = Model(PowerLaw("continuum", pivot=3000))
    if balmer_continuum:
        m.add(BalmerContinuum())
    if feii:
        m.add(FeII("feii_uv", kind="uv", fwhm=(3000, 1000, 10000), shift=(0, -2000, 2000)))
    _broad(m, "broad", ["MgII2798"], broad_components, broad_profile)
    m.add(Lines("narrow", ["MgII2798"], role="narrow", **NARROW))
    m.wave_range = wave_range
    return m


def civ(broad_components=1, broad_profile="gaussian", wave_range=(1450, 1720)):
    """CIV region (1450-1720 A): power law, broad CIV1549, HeII1640 and OIII]1663."""
    m = Model(PowerLaw("continuum", pivot=1450))
    _broad(m, "broad", ["CIV1549"], broad_components, broad_profile)
    m.add(Lines("heii", ["HeII1640", "OIII]1663"], role="broad", fwhm=(3000, 1000, 15000),
                shift=(0, -3000, 3000)))
    m.wave_range = wave_range
    return m


TYPE2_NARROW = ["[OII]3726", "[OII]3729", "[NeIII]3869", "[NeIII]3968", "He", "Hd", "Hg",
                "[OIII]4363", "HeII4686", "Hb", "[OIII]4959", "[OIII]5007", "[NI]5198", "[NI]5200",
                "HeI5876", "[OI]6300", "[OI]6364", "[NII]6548", "Ha", "[NII]6583", "HeI6678",
                "[SII]6716", "[SII]6731"]
_NARROW_RATIOS = {"[OIII]4959": ("[OIII]5007", OIII_RATIO), "[NII]6548": ("[NII]6583", NII_RATIO),
                  "[OI]6364": ("[OI]6300", OI_RATIO), "[NeIII]3968": ("[NeIII]3869", NEIII_RATIO)}


def _narrow_galaxy(m, outflow):
    m.add(Lines("narrow", TYPE2_NARROW, role="narrow", ratios=_NARROW_RATIOS,
                fwhm=(300, 50, 1200), shift=(0, -600, 600)))
    if outflow:
        m.add(Lines("outflow", ["Hb", "[OIII]4959", "[OIII]5007", "[NII]6548", "Ha", "[NII]6583"],
                    role="outflow", ratios={"[OIII]4959": ("[OIII]5007", OIII_RATIO),
                                            "[NII]6548": ("[NII]6583", NII_RATIO)},
                    **OUTFLOW))


def type2(outflow=True, host=True, featureless=True, coronal=False, extinction=False,
          wave_range=(3700, 6900)):
    """
    Type 2 AGN (Seyfert 2, LINER, composite): stellar-population host (with its
    Balmer absorption, which matters for H-beta), an optional weak featureless
    AGN continuum (power law, alpha_lambda = -1.7: scattered light), narrow
    lines from [OII]3727 to [SII]6731 with shared kinematics and an outflow
    wing on H-beta, [OIII], [NII] and H-alpha. No broad lines.

    :param featureless: add the featureless power-law continuum.
    :param coronal: add coronal lines (see :func:`optical`).
    :param extinction: fit an intrinsic E(B-V) of the emission (not the host).
    """
    m = Model(_host(host))
    if featureless:
        m.add(PowerLaw("continuum", pivot=5100))
        m.fix("continuum.index", PHYSICAL_SLOPE)
    _narrow_galaxy(m, outflow)
    if coronal:
        _coronal(m, wave_range)
    if extinction:
        m.add(Extinction(law=extinction if isinstance(extinction, str) else "F99"))
    m.wave_range = wave_range
    return m


def galaxy(host=True, outflow=False, extinction=False, wave_range=(3700, 6900)):
    """
    Star-forming, retired or passive galaxy: stellar populations (sigma*, v*,
    age, metallicity) and narrow emission lines with shared kinematics, no
    AGN continuum. ``outflow=True`` adds a broad wing to the strong lines
    (galactic winds).
    """
    m = Model(_host(host))
    _narrow_galaxy(m, outflow)
    if extinction:
        m.add(Extinction(law=extinction if isinstance(extinction, str) else "F99"))
    m.wave_range = wave_range
    return m


# ---------------------------------------------------------------------------
# Narrow-line galaxies: one preset per class of fantasy_agn.classify. The
# choices follow fits of SDSS examples of every class (see the documentation):
# a dust screen on the starlight (chi2/nu 4.6 -> 1.5 for a composite), young
# populations for star-forming galaxies (BC03 from 1 Myr; dBIC ~ -8000), two
# narrow-line systems and an outflow wing including [SII] for Seyfert 2s
# (dBIC -70 to -370 each), free line fluxes for AGN (Case B ties moved
# [OIII]/Hb by up to 0.1 dex), Case B + gas extinction for HII regions.
# ---------------------------------------------------------------------------
# H8 3889 includes HeI 3889 (0.4 A apart, not separable)
NARROW_FULL = ["[OII]3726", "[OII]3729", "[NeIII]3869", "H8", "[NeIII]3968", "He",
               "[SII]4069", "[SII]4076", "Hd", "Hg", "[OIII]4363", "HeI4471", "HeII4686", "Hb",
               "[OIII]4959", "[OIII]5007", "[NI]5198", "[NI]5200", "[NII]5755", "HeI5876",
               "[OI]6300", "[SIII]6312", "[OI]6364", "[NII]6548", "Ha", "[NII]6583", "HeI6678",
               "[SII]6716", "[SII]6731", "HeI7065", "[ArIII]7136", "[OII]7320", "[OII]7330"]
HIGH_IONISATION = {"[NeIII]3869", "[NeIII]3968", "[OIII]4363", "HeII4686", "[OIII]4959",
                   "[OIII]5007", "[ArIII]7136"}
HIGH_BALMER = ["H12", "H11", "H10", "H9"]
# Case B recombination, 10^4 K, 100 cm^-3 (Storey & Hummer 1995), relative to H-beta
CASE_B = {"Ha": 2.86, "Hg": 0.468, "Hd": 0.259, "He": 0.159, "H8": 0.105, "H9": 0.0731,
          "H10": 0.0530, "H11": 0.0397, "H12": 0.0305}
SF_AGES = (0.001, 0.003, 0.006, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 13.0)
WING_LINES = ["[NeIII]3869", "Hb", "[OIII]4959", "[OIII]5007", "[NII]6548", "Ha", "[NII]6583",
              "[SII]6716", "[SII]6731"]
_ALL_RATIOS = dict(_NARROW_RATIOS)


def _window(red, wave_range):
    return wave_range or (3700, RED_END if red else 6900)


def _inside(names, wave_range):
    from .lines import line_wavelength
    return [n for n in names if wave_range[0] < line_wavelength(n) < wave_range[1]]


def _lines(name, names, case_b=None, **kw):
    ratios = {a: b for a, b in _ALL_RATIOS.items() if a in names and b[0] in names}
    if case_b and "Hb" in names:
        ratios.update({a: ("Hb", v) for a, v in case_b.items() if a in names})
    return Lines(name, names, ratios=ratios, **kw)


def _stars(host, young=False):
    """Stellar populations with a dust screen (needed for the continuum shape)."""
    from .host import StellarPopulation
    if host == "pca":
        return HostGalaxy()
    templates = "emiles" if host in (True, "stellar") else host
    kw = dict(reddening=True)
    if young:
        kw["ages"] = SF_AGES
    return StellarPopulation(templates=templates, **kw)


def _agn_narrow(m, wave_range, two_systems, outflow, narrow_fwhm=(300, 50, 1200)):
    names = _inside(NARROW_FULL, wave_range)
    if two_systems:
        m.add(_lines("narrow", [n for n in names if n not in HIGH_IONISATION], role="narrow",
                     fwhm=narrow_fwhm, shift=(0, -600, 600)))
        m.add(_lines("narrow_hi", [n for n in names if n in HIGH_IONISATION], role="narrow",
                     fwhm=(400, 50, 1500), shift=(0, -600, 600)))
    else:
        m.add(_lines("narrow", names, role="narrow", fwhm=narrow_fwhm, shift=(0, -600, 600)))
    if outflow:
        m.add(_lines("outflow", _inside(WING_LINES, wave_range), role="outflow",
                     fwhm=(1000, 500, 2500), shift=(-200, -1500, 500)))


def seyfert2(host=True, outflow=True, two_systems=True, featureless=True, coronal=False,
             red=True, wave_range=None):
    """
    Seyfert 2 galaxy: stellar populations with dust (E-MILES; ``host='bc03'``
    for BC03), a weak featureless continuum (scattered AGN light, power law
    with alpha_lambda = -1.7), narrow lines from [OII]3727 to [OII]7330 in two
    kinematic systems (high ionisation: [NeIII], [OIII], HeII, [ArIII]; low
    ionisation: Balmer, [NII], [SII], [OI], ...), and an outflow wing on
    [NeIII], H-beta, [OIII], H-alpha, [NII] and [SII]. Line fluxes are free:
    the Balmer decrement measures the reddening afterwards.

    :param two_systems: separate kinematics for high- and low-ionisation lines.
    :param red: extend the fit to 7500 A (HeI 7065, [ArIII] 7136, [OII] 7320, 7330).
    :param coronal: add coronal lines ([FeVII], [FeX], [FeXIV]).
    """
    wave_range = _window(red, wave_range)
    m = Model(_stars(host))
    if featureless:
        m.add(PowerLaw("continuum", pivot=5100))
        m.fix("continuum.index", PHYSICAL_SLOPE)
    _agn_narrow(m, wave_range, two_systems, outflow)
    if coronal:
        _coronal(m, wave_range)
    m.wave_range = wave_range
    return m


def composite(host=True, outflow=True, two_systems=True, featureless=True, red=True,
              wave_range=None):
    """
    Composite galaxy (between the Kauffmann 2003 and Kewley 2001 lines): star
    formation and an AGN. Like :func:`seyfert2` (stellar populations with dust,
    weak featureless continuum, two narrow-line systems, outflow wing).
    """
    return seyfert2(host=host, outflow=outflow, two_systems=two_systems, featureless=featureless,
                    red=red, wave_range=wave_range)


def liner(host=True, broad=False, outflow=False, featureless=True, red=True, wave_range=None):
    """
    LINER: a dominant old stellar population with dust, a weak featureless
    continuum and one narrow-line system that may be broad (FWHM up to 1500
    km/s). The Balmer lines are weak against the stellar absorption, so their
    fluxes stay free. ``broad=True`` adds a broad H-beta/H-alpha (LINER 1.9);
    ``outflow=True`` an outflow wing.
    """
    wave_range = _window(red, wave_range)
    m = Model(_stars(host))
    if featureless:
        m.add(PowerLaw("continuum", pivot=5100))
        m.fix("continuum.index", PHYSICAL_SLOPE)
    _agn_narrow(m, wave_range, False, outflow, narrow_fwhm=(400, 50, 1500))
    if broad:
        m.add(Lines("broad", _inside(["Hb", "Ha"], wave_range), role="broad",
                    fwhm=(2500, 1200, 8000), shift=(0, -2000, 2000)))
    m.wave_range = wave_range
    return m


def starforming(host="bc03", outflow=True, extinction=True, wolf_rayet=False, red=True,
                wave_range=None):
    """
    Star-forming galaxy: young and old stellar populations with dust (BC03
    from 1 Myr; E-MILES starts at 63 Myr and cannot follow young continua),
    narrow lines including the high-order Balmer series, He I, the auroral
    lines [OIII]4363, [NII]5755, [SIII]6312 and [OII]7320,7330 (electron
    temperatures), Balmer lines tied to Case B so that the Balmer decrement
    gives the gas extinction E(B-V) (``extinction``), and a broad wing on the
    strong lines (galactic winds, unresolved velocity structure).

    :param wolf_rayet: add the Wolf-Rayet bumps at 4650 and 5808 A.
    """
    wave_range = _window(red, wave_range)
    m = Model(_stars(host, young=True))
    names = _inside(HIGH_BALMER + NARROW_FULL, wave_range)
    cb = CASE_B if extinction else None
    m.add(_lines("narrow", names, case_b=cb, role="narrow", fwhm=(150, 20, 800), shift=(0, -400, 400)))
    if outflow:
        m.add(_lines("outflow", _inside(WING_LINES, wave_range), case_b=cb, role="outflow",
                     fwhm=(600, 200, 2500), shift=(0, -800, 400)))
    if wolf_rayet:
        # blue bump (NIII 4640, CIII 4650, HeII 4686) and red bump (CIV 5801, 5812)
        bumps = [(n, w) for n, w in (("WR4650", 4652.0), ("WR5808", 5810.0))
                 if wave_range[0] < w < wave_range[1]]
        m.add(Lines("wolf_rayet", bumps, role="broad", tie_kinematics=False,
                    fwhm=(1500, 500, 4000), shift=(0, -1000, 1000)))
    if extinction:
        m.add(Extinction(law="calzetti" if extinction == "calzetti" else "F99"))
    m.wave_range = wave_range
    return m


def passive(host=True, red=True, wave_range=None):
    """
    Passive or retired galaxy (EW(H-alpha) < 3 A): stellar populations with
    dust for sigma*, v*, age and metallicity, plus weak narrow lines with
    limited widths (so that they cannot absorb template mismatch). The
    retired galaxies of the WHAN diagram are ionised by old stars.
    """
    wave_range = _window(red, wave_range)
    m = Model(_stars(host))
    names = _inside(["[OII]3726", "[OII]3729", "Hb", "[OIII]4959", "[OIII]5007", "[OI]6300",
                     "[OI]6364", "[NII]6548", "Ha", "[NII]6583", "[SII]6716", "[SII]6731"], wave_range)
    m.add(_lines("narrow", names, role="narrow", fwhm=(300, 50, 600), shift=(0, -400, 400)))
    m.wave_range = wave_range
    return m


retired = passive

DEFAULT_RANGES = {"optical": (3700, 6900), "seyfert2": (3700, 7500), "liner": (3700, 7500),
                  "composite": (3700, 7500), "starforming": (3700, 7500), "passive": (3700, 7500)}   # presets whose default window depends on options
PRESETS = {"optical": optical, "hbeta": hbeta, "halpha": halpha, "mgii": mgii, "civ": civ,
           "type2": type2, "galaxy": galaxy, "seyfert2": seyfert2, "liner": liner,
           "composite": composite, "starforming": starforming, "passive": passive}


def available(spectrum):
    """Names of the presets whose wavelength window the (rest-frame) spectrum covers."""
    import inspect
    lo, hi = spectrum.wave[spectrum.good].min(), spectrum.wave[spectrum.good].max()
    out = []
    for name, func in PRESETS.items():
        w0, w1 = inspect.signature(func).parameters["wave_range"].default or DEFAULT_RANGES[name]
        if lo <= w0 + 50 and hi >= w1 - 50:
            out.append(name)
    return out


def auto(spectrum, **options):
    """The most suitable preset for the spectrum's rest-frame coverage."""
    names = available(spectrum)
    if not names:
        raise ValueError(
            f"no preset covers {spectrum.wave.min():.0f}-{spectrum.wave.max():.0f} A; "
            "is the spectrum in the rest frame? Otherwise build a Model by hand.")
    name = "hbeta" if "hbeta" in names else names[0]
    return PRESETS[name](**options)


def variants(name="hbeta", broad_components=(1, 2, 3), outflow=(False, True),
             broad_profile=("gaussian",), **options):
    """
    Several decompositions of one preset for fa.compare(): every combination of
    the number of broad components, the broad-line profile and with/without an
    outflow wing (where the preset has one). Other options (e.g. host=True)
    apply to all. For H-alpha, ``outflow=(False, hbeta_result)`` ties the wing
    to the [OIII] outflow.

    :param broad_profile: profile name or several, e.g.
        ('gaussian', 'gauss-hermite'): Gauss-Hermite components can follow
        asymmetric broad lines (bumps, shoulders) with fewer components.
    """
    import inspect
    func = PRESETS[name]
    params = inspect.signature(func).parameters
    profiles = (broad_profile,) if isinstance(broad_profile, str) else tuple(broad_profile)
    short = {"gaussian": "", "gauss-hermite": "GH ", "lorentzian": "Lorentzian ", "voigt": "Voigt "}
    out = {}
    for prof in profiles:
        for n in broad_components:
            for o in (outflow if "outflow" in params else (False,)):
                kw = dict(options, broad_components=n)
                if prof != "gaussian":
                    kw["broad_profile"] = prof
                if "outflow" in params:
                    kw["outflow"] = o
                label = (f"{n} {short.get(prof, prof + ' ')}broad"
                         + ("" if o is False or o is None
                            else " + outflow" if o is True else " + [OIII]-tied outflow"))
                out[label] = func(**kw)
    return out
