"""
Narrow-line diagnostics from a fit, with uncertainties propagated from the
fit (covariance or Monte Carlo samples): gas extinction from the Balmer
decrement, electron density and temperature, BPT and WHAN positions and
classes, and the gas-phase metallicity of star-forming galaxies.

    d = r.diagnostics()        # a table: quantity, value, error, unit, note

Line fluxes are summed over the narrow-line components ('narrow', and for
two systems 'narrow_hi') and, with ``outflow=True`` (default), the outflow
wing, as for the total narrow-line fluxes of SDSS/MPA-JHU. Ratios of lines far
apart are corrected for the gas extinction from the Balmer decrement.

References: Case B Halpha/Hbeta 2.86 (H II regions, 10^4 K) and 3.1 (AGN
narrow-line regions; Osterbrock & Ferland 2006), Fitzpatrick (1999) law with
R_V = 3.1; [SII] density: Sanders et al. (2016, ApJ 816, 23); [OIII] and [NII]
temperatures: Osterbrock & Ferland (2006); BPT lines: Kauffmann et al. (2003),
Kewley et al. (2001, 2006); WHAN: Cid Fernandes et al. (2011); metallicity:
Marino et al. (2013, A&A 559, A114), O3N2 and N2 calibrations.
"""
import numpy as np
import pandas as pd

from .classify import kauffmann03, kewley01, kewley06, _above_kewley
from .extinction import fitzpatrick99
from .lines import line_wavelength

__all__ = ("diagnostics",)

CASE_B = {"starforming": 2.86, "agn": 3.1}


def _flux_function(result, outflow):
    """{line: list of parameter indices} for the narrow (+ outflow) fluxes."""
    names = result.model.params.names
    roles = ("narrow", "outflow") if outflow else ("narrow",)
    out = {}
    for comp in result._line_components():
        if comp.role not in roles:
            continue
        for ln in comp.line_names:
            key = f"{comp.name}.{ln}.flux"
            if key in names:
                out.setdefault(ln, []).append(names.index(key))
    return out


def _k(line):
    return float(fitzpatrick99(np.array([line_wavelength(line)]))[0])


def _ne_sii(R, T=1e4):
    """Electron density (cm^-3) from [SII]6716/6731 (Sanders et al. 2016, 10^4 K)."""
    a, b, c = 0.4315, 2107.0, 627.1
    if not np.isfinite(R):
        return np.nan
    R = float(np.clip(R, 0.4375 + 1e-4, 1.4484 - 1e-4))      # the low/high-density limits
    return (c * R - a * b) / (a - R)


def _te(ratio, coef, expo, ne_coef, ne=100.0):
    """Electron temperature (K) from an auroral/nebular ratio, solved iteratively."""
    if not np.isfinite(ratio) or ratio <= coef:
        return np.nan
    T = 1e4
    for _ in range(50):
        T_new = expo / np.log(ratio * (1 + ne_coef * ne / np.sqrt(T)) / coef)
        if not np.isfinite(T_new) or T_new <= 0:
            return np.nan
        if abs(T_new - T) < 1:
            return float(T_new)
        T = T_new
    return float(T)


def _te_oiii(R, ne=100.0):
    # (j4959 + j5007) / j4363 = 7.90 exp(32900/T) / (1 + 4.5e-4 ne / T^0.5)
    return _te(R, 7.90, 32900.0, 4.5e-4, ne)


def _te_nii(R, ne=100.0):
    # (j6548 + j6583) / j5755 = 8.23 exp(25000/T) / (1 + 4.4e-3 ne / T^0.5)
    return _te(R, 8.23, 25000.0, 4.4e-3, ne)


def diagnostics(result, outflow=True, kind=None):
    """
    Narrow-line diagnostics of a fit as a DataFrame (quantity, value, error,
    unit, note). Quantities that the model or the data cannot give are left
    out.

    :param outflow: include the outflow wing in the line fluxes.
    :param kind: 'starforming' or 'agn' for the intrinsic Balmer decrement
        (default: from the BPT position, AGN above the Kewley et al. 2001 line).
    """
    idx = _flux_function(result, outflow)
    have = set(idx)
    k_ha, k_hb = _k("Ha"), _k("Hb")

    def flux(full, line):
        return float(np.sum(full[idx[line]])) if line in idx else np.nan

    def ratio(full, a, b):
        fa, fb = flux(full, a), flux(full, b)
        return fa / fb if fb > 0 and fa > 0 else np.nan

    rows = []

    def add(name, func, unit="", note=""):
        try:
            val, err = result.propagate(func)
        except Exception:
            return
        val, err = float(val), float(err)
        if np.isfinite(val):
            rows.append((name, val, err if np.isfinite(err) else np.nan, unit, note))

    log = lambda x: np.log10(x) if x > 0 else np.nan
    # BPT positions (nearby lines: no extinction correction needed)
    pairs = {"log [NII]6583/Ha": ("[NII]6583", "Ha"), "log [OIII]5007/Hb": ("[OIII]5007", "Hb"),
             "log [OI]6300/Ha": ("[OI]6300", "Ha")}
    for name, (a, b) in pairs.items():
        if a in have and b in have:
            add(name, lambda f, a=a, b=b: log(ratio(f, a, b)), "dex")
    if {"[SII]6716", "[SII]6731", "Ha"} <= have:
        add("log [SII]6716,6731/Ha",
            lambda f: log((flux(f, "[SII]6716") + flux(f, "[SII]6731")) / flux(f, "Ha")), "dex")
    vals = {r[0]: r[1] for r in rows}
    x, y = vals.get("log [NII]6583/Ha"), vals.get("log [OIII]5007/Hb")
    bpt = None
    if x is not None and y is not None:
        if x < 0.05 and y < kauffmann03(x):
            bpt = "star-forming"
        elif not _above_kewley(x, y, "nii"):
            bpt = "composite"
        else:
            votes = []
            for name, d in (("log [SII]6716,6731/Ha", "sii"), ("log [OI]6300/Ha", "oi")):
                if name in vals:
                    votes.append("Seyfert" if y > kewley06(vals[name], d) else "LINER")
            bpt = max(set(votes), key=votes.count) if votes else ("Seyfert" if y > 0.5 else "LINER")
    if kind is None:
        kind = "starforming" if bpt in ("star-forming", None) else "agn"

    # Balmer decrement and gas extinction. With an Extinction component in the model
    # (e.g. the starforming preset, Balmer lines tied to Case B) the fitted line
    # fluxes are already dereddened and E(B-V) is a fitted parameter.
    names = result.model.params.names
    fitted_ext = next((n for n in names if n.endswith(".ebv")
                       and any(c.name == n.split(".")[0] and type(c).__name__ == "Extinction"
                               for c in result.model.components)), None)
    intrinsic = CASE_B["starforming" if kind == "starforming" else "agn"]
    ebv_of = lambda f: 2.5 / (k_hb - k_ha) * np.log10(ratio(f, "Ha", "Hb") / intrinsic)
    if fitted_ext is not None:
        i_ext = names.index(fitted_ext)
        add("E(B-V) gas", lambda f: float(f[i_ext]), "mag",
            "fitted (Balmer lines tied to Case B); line fluxes are dereddened")
    elif {"Ha", "Hb"} <= have:
        add("narrow Ha/Hb", lambda f: ratio(f, "Ha", "Hb"), "",
            f"Case B {intrinsic} ({'H II region' if kind == 'starforming' else 'AGN'})")
        add("E(B-V) gas", lambda f: max(ebv_of(f), 0.0), "mag", "Fitzpatrick (1999), R_V = 3.1")

    def dered(f, a, b):
        """Ratio a/b corrected for the gas extinction (none if the decrement is below Case B)."""
        r_ab = ratio(f, a, b)
        if fitted_ext is not None or not ({"Ha", "Hb"} <= have):
            return r_ab
        e = max(ebv_of(f), 0.0)
        return r_ab * 10 ** (0.4 * e * (_k(a) - _k(b)))

    # electron density
    if {"[SII]6716", "[SII]6731"} <= have:
        add("[SII]6716/6731", lambda f: ratio(f, "[SII]6716", "[SII]6731"))
        add("n_e ([SII])", lambda f: _ne_sii(ratio(f, "[SII]6716", "[SII]6731")), "cm^-3",
            "Sanders et al. (2016), 10^4 K; limits 1 and 10^4.5 cm^-3")
    vals = {r[0]: r[1] for r in rows}
    ne = vals.get("n_e ([SII])", 100.0)
    ne = 100.0 if not np.isfinite(ne) else max(ne, 1.0)
    # electron temperatures (the auroral lines must be detected)
    tab = result.lines

    def detected(line, snr=3.0):
        t = tab[tab.line == line]
        return len(t) and t.flux.sum() > snr * np.sqrt(np.nansum(t.flux_err ** 2))

    if {"[OIII]4959", "[OIII]5007", "[OIII]4363"} <= have and detected("[OIII]4363"):
        add("T_e ([OIII])", lambda f: _te_oiii(
            dered(f, "[OIII]5007", "[OIII]4363") * (1 + ratio(f, "[OIII]4959", "[OIII]5007")), ne), "K",
            "Osterbrock & Ferland (2006)")
    if {"[NII]6548", "[NII]6583", "[NII]5755"} <= have and detected("[NII]5755"):
        add("T_e ([NII])", lambda f: _te_nii(
            dered(f, "[NII]6583", "[NII]5755") * (1 + ratio(f, "[NII]6548", "[NII]6583")), ne), "K",
            "Osterbrock & Ferland (2006)")

    # WHAN: equivalent width of narrow H-alpha
    if "Ha" in have:
        def ew_ha(f):
            from .lines import line_wavelength as lw
            cont = result._agn_continuum(lw("Ha"), f)[0] if hasattr(result, "_agn_continuum") else np.nan
            total = result.model.evaluate(np.array([lw("Ha") - 60, lw("Ha") + 60]),
                                          result.spectrum.context, f, components=True)
            line_free = sum(v for k, v in total.items()
                            if not any(c.name == k and hasattr(c, "line_names") for c in result.model.components))
            c_ha = float(np.mean(line_free)) if np.ndim(line_free) else np.nan
            return flux(f, "Ha") / c_ha if c_ha > 0 else np.nan
        add("EW(Ha) narrow", ew_ha, "A", "WHAN (Cid Fernandes et al. 2011)")

    # metallicity of star-forming galaxies
    vals = {r[0]: r[1] for r in rows}
    if bpt == "star-forming" and {"[NII]6583", "Ha", "[OIII]5007", "Hb"} <= have:
        add("12+log(O/H) O3N2", lambda f: 8.533 - 0.214 * (log(ratio(f, "[OIII]5007", "Hb"))
                                                           - log(ratio(f, "[NII]6583", "Ha"))),
            "", "Marino et al. (2013), calibration scatter 0.18 dex; valid for -1.1 < O3N2 < 1.7")
        add("12+log(O/H) N2", lambda f: 8.743 + 0.462 * log(ratio(f, "[NII]6583", "Ha")),
            "", "Marino et al. (2013), calibration scatter 0.16 dex; valid for -1.6 < N2 < -0.2")
    df = pd.DataFrame(rows, columns=["quantity", "value", "error", "unit", "note"])
    if bpt:
        df.attrs["bpt"] = bpt
        df = pd.concat([df, pd.DataFrame([("BPT class", np.nan, np.nan, "", bpt)], columns=df.columns)],
                       ignore_index=True)
    return df
