"""
Line library and emission templates.

All wavelengths are stored in vacuum. The library (data/lines.csv) lists each
line with its wavelength, the medium that wavelength refers to (air or vacuum),
ion, type, series and notes; it is converted to vacuum once at import. Models
convert wavelengths to the medium of the spectrum being fitted, so SDSS-like
vacuum spectra are fitted without resampling. The FeII and Balmer templates
follow the IAU convention (air above 2000 A, vacuum below).
"""
import os
import re

import numpy as np
import pandas as pd

from .wavelengths import air_to_vac

__all__ = ("LINES", "LINE_SETS", "line_wavelength", "line_table", "lines_in", "line_info", "line_set",
           "read_line_list", "load_template")

DATA = os.path.join(os.path.dirname(__file__), "data")

_ALIASES = {"Halpha": "Ha", "Hbeta": "Hb", "Hgamma": "Hg", "Hdelta": "Hd",
            "Hepsilon": "He", "H-epsilon": "He", "Lyalpha": "Lya", "Lybeta": "Lyb",
            "MgII": "MgII2798", "CIV": "CIV1549", "CIII]": "CIII]1909", "OVI": "OVI1034",
            "HeII": "HeII4686", "HeI": "HeI5876", "NIII": "NIII4640",
            "Paalpha": "Paa", "Pabeta": "Pab", "Pagamma": "Pag", "Padelta": "Pad", "Paepsilon": "Pae",
            "Bralpha": "Bra", "Brbeta": "Brb", "Brgamma": "Brg", "Brdelta": "Brd", "Brepsilon": "Bre",
            "Pfbeta": "Pfb", "Pfgamma": "Pfg", "Pfdelta": "Pfd", "Pfepsilon": "Pfe"}
for _k in [k for k in _ALIASES if k[:2] in ("Pa", "Br", "Pf")]:
    _ALIASES[_k[:2] + "-" + _k[2:]] = _ALIASES[_k]          # 'Pa-alpha', 'Br-gamma', ...


def _to_vacuum(wave):
    """Air to vacuum above 2000 A (IAU convention), unchanged below; used for the templates."""
    wave = np.asarray(wave, dtype=float)
    return np.where(wave > 2000, air_to_vac(wave), wave)


def _load_library():
    """The line library (data/lines.csv): every wavelength with its medium, converted to vacuum."""
    df = pd.read_csv(os.path.join(DATA, "lines.csv"), comment="#", skipinitialspace=True,
                     keep_default_na=False)
    if df.name.duplicated().any():
        raise ValueError(f"duplicate line names: {sorted(df.name[df.name.duplicated()])}")
    wave = df.wavelength.to_numpy(dtype=float)
    air = (df.medium == "air").to_numpy()
    df.insert(1, "vacuum", np.where(air, air_to_vac(wave), wave))
    return df.sort_values("vacuum", kind="stable").set_index("name")


_LIBRARY = _load_library()
LINES = {name: float(w) for name, w in _LIBRARY.vacuum.items()}


def line_table():
    """
    The line library as a table (one row per line, sorted by wavelength): vacuum
    wavelength, the tabulated wavelength and its medium, ion, type (recombination,
    permitted, semi-forbidden, forbidden, coronal, bowen, molecular, feature), the
    series or multiplet, and notes on blends and caveats.
    """
    return _LIBRARY.copy()


def lines_in(wmin=0.0, wmax=np.inf, type=None, ion=None, series=None):
    """
    Names of the library lines with vacuum wavelengths in [wmin, wmax] (A), optionally
    only of the given type(s), ion(s) or series, e.g.
    ``lines_in(10000, 25000, type="recombination")`` or ``lines_in(series="Brackett")``.
    """
    t = _LIBRARY[(_LIBRARY.vacuum >= wmin) & (_LIBRARY.vacuum <= wmax)]
    for col, val in (("type", type), ("ion", ion), ("series", series)):
        if val is not None:
            t = t[t[col].isin([val] if isinstance(val, str) else list(val))]
    return list(t.index)


def line_info(name):
    """Everything the library knows about one line (accepts aliases)."""
    name = _ALIASES.get(name, name)
    line_wavelength(name)
    return {"name": name, **_LIBRARY.loc[name].to_dict()}


def line_wavelength(name):
    """Vacuum wavelength of a line in the library (accepts common aliases)."""
    name = _ALIASES.get(name, name)
    try:
        return LINES[name]
    except KeyError:
        raise KeyError(
            f"unknown line {name!r}; give (name, wavelength) or use one of "
            f"{sorted(LINES)}") from None


# The line lists of FANTASY 1 (data/<file>.csv, columns line, position) as named sets of library lines.
LINE_SETS = {
    "narrow_basic": ("Narrow lines, basic", "narrow_basic"),
    "narrow_plus": ("Narrow lines, extended", "narrow_plus"),
    "oiii_nii": ("[O III] and [N II]", "oiii_nii"),
    "hydrogen": ("Hydrogen", "hydrogen"),
    "helium": ("Helium", "helium"),
    "broad": ("Other broad lines", "broad"),
    "coronal": ("Coronal and high-ionisation", "coronal"),
    "feII_forbidden": ("[Fe II] forbidden", "feII_forbidden"),
    "feii_IZw1": ("Fe II permitted (I Zw 1)", "feii_IZw1"),
}


def _ion(label):
    """'[FeII]21F' -> 'Fe II', 'Hb' -> 'H I', 'feii' -> 'Fe II' (labels of the FANTASY 1 line lists)."""
    label = label.strip()
    if label in ("Ha", "Hb", "Hg", "Hd", "Heps") or label[:2] in ("Pa", "Ly", "Br", "Pf"):
        return "H I"
    if label.lower() == "feii":
        return "Fe II"
    m = re.match(r"\[?([A-Z][a-z]?)([IVX]+)", label)
    return f"{m.group(1)} {m.group(2)}" if m else label


def line_set(name, wmin=0.0, wmax=np.inf):
    """
    Library names of a FANTASY 1 line list (see ``LINE_SETS``: 'narrow_basic',
    'narrow_plus', 'oiii_nii', 'hydrogen', 'helium', 'broad', 'coronal',
    'feII_forbidden', 'feii_IZw1'), optionally only those with vacuum
    wavelengths in [wmin, wmax], e.g. ``Lines("coronal", fa.line_set("coronal", 4000, 7000))``.
    """
    if name not in LINE_SETS:
        raise KeyError(f"unknown line set {name!r}; one of {sorted(LINE_SETS)}")
    df = pd.read_csv(os.path.join(DATA, LINE_SETS[name][1] + ".csv"), skipinitialspace=True)
    out = []
    for label, w in zip(df.line, df.position.astype(float)):
        c = _LIBRARY[_LIBRARY.ion == _ion(label)]
        d = np.minimum((c.wavelength - w).abs(), (c.vacuum - w).abs()).drop(out, errors="ignore")
        if len(d) and d.min() < 3.0:
            out.append(d.idxmin())
    return [n for n in out if wmin <= LINES[n] <= wmax]


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
          'feii_uv' (the UV model around MgII of Kovacevic-Dojcinovic & Popovic 2015, 2690-3005 A),
          'feii_uv_extended' (UV FeII 2200-3500 A, grouped by lower term as the optical model),
          'balmer' (high-order Balmer lines).
    """
    if kind == "feii":
        df = pd.read_csv(os.path.join(DATA, "feII_model.csv"))
        names, wave, inten = df.ime, df.wav, df.Int
    elif kind == "feii_uv_extended":
        df = pd.read_csv(os.path.join(DATA, "uvfe2.csv"), comment="#")
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
