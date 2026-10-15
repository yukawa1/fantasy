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

__all__ = ("hbeta", "halpha", "optical", "mgii", "civ", "available", "auto", "variants",
           "outflow_kinematics",
           "PRESETS")

NARROW = dict(fwhm=(400, 50, 1200), shift=(0, -600, 600))
# One broad Gaussian covers the whole profile (FWHM 1000-20000 km/s).
BROAD = dict(fwhm=(3500, 1000, 20000), shift=(0, -3000, 3000))
BROAD_FWHM = (1000.0, 30000.0)   # FWHM range shared out among several components
BROAD_OVERLAP = 1.3


def broad_fwhm_ranges(n):
    """
    FWHM limits (start, min, max) for n broad components: 1000-30000 km/s is
    split into n logarithmic bands that overlap by a factor 1.3, so every
    component has its own width regime (core, intermediate, wings, ...) and
    two components cannot simply swap roles.
    """
    lo, hi = BROAD_FWHM
    edges = lo * (hi / lo) ** (np.arange(n + 1) / n)
    out = []
    for k in range(n):
        a = max(lo, edges[k] / BROAD_OVERLAP)
        b = min(hi, edges[k + 1] * BROAD_OVERLAP)
        out.append((float(np.sqrt(a * b)), float(a), float(b)))
    return out
OUTFLOW = dict(fwhm=(900, 300, 2500), shift=(-300, -2000, 200))

OIII_RATIO = 1 / 2.98   # Storey & Zeippen 2000
NII_RATIO = 1 / 3.05
OI_RATIO = 1 / 3.0
NII_HA_MAX = 3.0            # upper limit of [NII]6583/H-alpha in an outflow wing
HA_OUTFLOW_FWHM = (900, 300, 2000)


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
    host=True or 'stellar': non-negative stellar population templates with
    sigma* and v* (needs fantasy_agn.templates.download('emiles') once);
    host='pca': the Yip et al. (2004) eigenspectra of FANTASY 1.x.
    """
    if kind is True or kind == "stellar":
        from .host import StellarPopulation
        return StellarPopulation()
    if kind == "pca":
        return HostGalaxy()
    raise ValueError("host must be True/'stellar', 'pca' or False")


def _broad(model, name, lines, n, profile="gaussian"):
    """
    n broad Gaussian components for the same lines, named name, name2, name3, ...
    Each has its own kinematics; the narrowest may shift by +-3000 km/s, the
    broader ones by +-5000 km/s.
    """
    if n < 1:
        raise ValueError("broad_components must be at least 1")
    if n == 1:
        model.add(Lines(name, lines, role="broad", profile=profile, **BROAD))
        return model
    for k, fwhm in enumerate(broad_fwhm_ranges(n)):
        shift = (0, -3000, 3000) if k == 0 else (0, -5000, 5000)
        model.add(Lines(name if k == 0 else f"{name}{k + 1}", lines, role="broad",
                        fwhm=fwhm, shift=shift, profile=profile))
    return model


def hbeta(broad_components=2, feii=True, outflow=False, host=False, heii=True,
          continuum="powerlaw", extinction=False,
          broad_profile="gaussian", wave_range=(4150, 5550)):
    """
    H-beta region (4150-5550 A): power law, optical FeII, narrow H-gamma,
    [OIII]4363, H-beta and [OIII]4959,5007 (fixed 1:2.98 ratio), broad
    H-gamma and H-beta with shared kinematics, broad HeII4686.

    :param broad_components: number of broad components (1, 2, 3, ...).
    :param broad_profile: 'gaussian', 'lorentzian', 'voigt' or 'gauss-hermite'.
        Compare ``result.bic`` between fits to decide how many the data support.
    :param outflow: add a blueshifted [OIII] wing component.
    :param host: fit a host galaxy simultaneously: True/'stellar' (stellar
        population templates with sigma* and v*) or 'pca' (Yip+2004 eigenspectra).
    :param continuum: 'powerlaw' or 'broken' (slope change at 4600 A).
    """
    m = Model(_continuum(continuum, 5100, 4600))
    if host:
        m.add(_host(host))
    if feii:
        m.add(FeII(fwhm=(2500, 1000, 8000), shift=(0, -2000, 2000)))
    m.add(Lines("narrow", ["Hg", "[OIII]4363", "Hb", "[OIII]4959", "[OIII]5007"], role="narrow",
                ratios={"[OIII]4959": ("[OIII]5007", OIII_RATIO)}, **NARROW))
    if outflow:
        m.add(Lines("outflow", ["[OIII]4959", "[OIII]5007"], role="outflow",
                    ratios={"[OIII]4959": ("[OIII]5007", OIII_RATIO)}, **OUTFLOW))
    _broad(m, "broad", ["Hg", "Hb"], broad_components, broad_profile)
    if heii:
        m.add(Lines("heii", ["HeII4686"], role="broad", fwhm=(4000, 1000, 15000), shift=(0, -4000, 2000)))
    if extinction:
        m.add(Extinction(law=extinction if isinstance(extinction, str) else "F99"))
    m.wave_range = wave_range
    return m


def halpha(broad_components=2, outflow=False, host=False, continuum="powerlaw",
           extinction=False, hei=True,
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
            balmer=True, extinction=False,
          broad_profile="gaussian", wave_range=(3700, 6900)):
    """
    Full optical range (3700-6900 A): broken power law, Balmer continuum and
    high-order Balmer lines, FeII, narrow lines from [OII]3727 to [SII]6731,
    broad H-delta/H-gamma/H-beta (shared kinematics), broad H-alpha (own
    kinematics) and broad HeI/HeII.

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
    m = Model(_continuum(continuum, 5100, 4600))
    if host:
        m.add(_host(host))
    if feii:
        m.add(FeII(fwhm=(2500, 1000, 8000), shift=(0, -2000, 2000)))
    narrow = ["[OII]3726", "[OII]3729", "[NeIII]3869", "Hd", "Hg", "[OIII]4363", "Hb",
              "[OIII]4959", "[OIII]5007", "HeI5876", "[OI]6300", "[OI]6364",
              "[NII]6548", "Ha", "[NII]6583", "[SII]6716", "[SII]6731"]
    m.add(Lines("narrow", narrow, role="narrow", ratios={"[OIII]4959": ("[OIII]5007", OIII_RATIO),
                                          "[NII]6548": ("[NII]6583", NII_RATIO),
                                          "[OI]6364": ("[OI]6300", OI_RATIO)}, **NARROW))
    if outflow:
        m.add(Lines("outflow", ["[OIII]4959", "[OIII]5007"], role="outflow",
                    ratios={"[OIII]4959": ("[OIII]5007", OIII_RATIO)}, **OUTFLOW))
    _broad(m, "broad", ["Hd", "Hg", "Hb"], broad_components, broad_profile)
    _broad(m, "broad_ha", ["Ha"], broad_components, broad_profile)
    m.add(Lines("broad_he", ["HeII4686", "HeI5876"], role="broad", fwhm=(4000, 1000, 15000),
                shift=(0, -4000, 2000)))
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


PRESETS = {"optical": optical, "hbeta": hbeta, "halpha": halpha, "mgii": mgii, "civ": civ}


def available(spectrum):
    """Names of the presets whose wavelength window the (rest-frame) spectrum covers."""
    import inspect
    lo, hi = spectrum.wave[spectrum.good].min(), spectrum.wave[spectrum.good].max()
    out = []
    for name, func in PRESETS.items():
        w0, w1 = inspect.signature(func).parameters["wave_range"].default
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
