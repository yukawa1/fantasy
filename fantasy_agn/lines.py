"""
Line library and emission templates.

All wavelengths are stored in vacuum. Following the IAU convention the
reference values below are given in air above 2000 A and in vacuum below, and
are converted once at import. Models convert them to the medium of the
spectrum being fitted, so SDSS-like vacuum spectra are fitted without
resampling.
"""
import os

import numpy as np
import pandas as pd

from .wavelengths import air_to_vac

__all__ = ("LINES", "line_wavelength", "read_line_list", "load_template")

DATA = os.path.join(os.path.dirname(__file__), "data")

# name: wavelength in Angstrom (air above 2000 A, vacuum below)
_REFERENCE = {
    "Lya": 1215.67,
    "NV1240": 1240.14,
    "SiIV1397": 1396.76,
    "CIV1549": 1549.06,
    "CIV1548": 1548.19,
    "CIV1551": 1550.77,
    "HeII1640": 1640.42,
    "OIII]1663": 1663.48,
    "AlIII1857": 1857.40,
    "CIII]1909": 1908.73,
    "[NeIV]2424": 2423.83,
    "MgII2798": 2798.75,
    "MgII2796": 2795.53,
    "MgII2803": 2802.70,
    "[NeV]3426": 3425.87,
    "[OII]3726": 3726.03,
    "[OII]3729": 3728.82,
    "[NeIII]3869": 3868.76,
    "[NeIII]3968": 3967.47,
    "He": 3970.08,
    "Hd": 4101.74,
    "Hg": 4340.47,
    "[OIII]4363": 4363.21,
    "HeI4471": 4471.48,
    "HeII4686": 4685.71,
    "Hb": 4861.33,
    "[OIII]4959": 4958.91,
    "[OIII]5007": 5006.84,
    "[NI]5198": 5197.90,
    "[NI]5200": 5200.26,
    "HeI5876": 5875.62,
    "[OI]6300": 6300.30,
    "[OI]6364": 6363.78,
    "[NII]6548": 6548.05,
    "Ha": 6562.82,
    "[NII]6583": 6583.45,
    "HeI6678": 6678.15,
    "[SII]6716": 6716.44,
    "[SII]6731": 6730.82,
    "HeI7065": 7065.20,
    "[ArIII]7136": 7135.79,
    "OI8446": 8446.36,
    "CaII8498": 8498.02,
    "CaII8542": 8542.09,
    "CaII8662": 8662.14,
    "Pa9": 9229.01,
    "[SIII]9069": 9068.60,
    "[SIII]9531": 9531.10,
    "Pad": 10049.37,
    "HeI10830": 10830.34,
    "Pag": 10938.09,
}

_ALIASES = {"Halpha": "Ha", "Hbeta": "Hb", "Hgamma": "Hg", "Hdelta": "Hd",
            "Hepsilon": "He", "MgII": "MgII2798", "CIV": "CIV1549",
            "CIII]": "CIII]1909", "HeII": "HeII4686", "HeI": "HeI5876"}


def _to_vacuum(wave):
    wave = np.asarray(wave, dtype=float)
    return np.where(wave > 2000, air_to_vac(wave), wave)


LINES = {name: float(_to_vacuum(w)) for name, w in _REFERENCE.items()}


def line_wavelength(name):
    """Vacuum wavelength of a line in the library (accepts common aliases)."""
    name = _ALIASES.get(name, name)
    try:
        return LINES[name]
    except KeyError:
        raise KeyError(
            f"unknown line {name!r}; give (name, wavelength) or use one of "
            f"{sorted(LINES)}") from None


def read_line_list(path, medium="air"):
    """
    Read a CSV line list with columns ``line, position`` (FANTASY 1.x format)
    and return a list of (name, vacuum wavelength).
    """
    df = pd.read_csv(path, skipinitialspace=True)
    wave = df.position.to_numpy(dtype=float)
    if medium == "air":
        wave = _to_vacuum(wave)
    names = [f"{n.strip()}{int(round(w))}" if not n.strip()[-1].isdigit() else n.strip()
             for n, w in zip(df.line, df.position)]
    return list(zip(names, wave))


def load_template(kind):
    """
    Load an emission template as (vacuum wavelength, relative intensity,
    group names, group index of each line).

    kind: 'feii' (optical FeII, Kovacevic et al. 2010 / Shapovalova et al. 2012),
          'feii_uv' (UV FeII around MgII), 'balmer' (high-order Balmer lines).
    """
    if kind == "feii":
        df = pd.read_csv(os.path.join(DATA, "feII_model.csv"))
        names, wave, inten = df.ime, df.wav, df.Int
    elif kind == "feii_uv":
        df = pd.read_csv(os.path.join(DATA, "uvfe.csv"))
        names, wave, inten = df.ime, df.wav, df.Int
    elif kind == "balmer":
        df = pd.read_csv(os.path.join(DATA, "balmer.csv"), encoding="utf-8-sig")
        names = pd.Series(["balmer"] * len(df))
        wave, inten = df.position, df["int"]
    else:
        raise ValueError(f"unknown template {kind!r}")
    groups = list(pd.unique(names))
    index = pd.Index(groups).get_indexer(names)
    return (_to_vacuum(wave.to_numpy(dtype=float)), inten.to_numpy(dtype=float),
            [str(g) for g in groups], index)
