"""
Stellar population templates for host-galaxy fitting.

The templates are not bundled with FANTASY. They are the ready-to-use files
that M. Cappellari distributes for pPXF (https://github.com/micappe/ppxf_data,
provided there with permission of the model authors), downloaded once on
request and kept in a cache folder (``$FANTASY_DATA`` or ``~/.fantasy_agn``)::

    import fantasy_agn as fa
    fa.templates.download("emiles")    # 13 MB, once

Please cite the model papers when you use them (see ``TEMPLATE_SETS``).
"""
import os
import urllib.request

import numpy as np

__all__ = ("TEMPLATE_SETS", "download", "load", "cache_dir", "available")

_BASE = "https://raw.githubusercontent.com/micappe/ppxf_data/main/"

TEMPLATE_SETS = {
    "emiles": dict(file="spectra_emiles_9.0.npz", size_mb=13.4, medium="air",
                   cite="Vazdekis et al. 2016, MNRAS 463, 3409 (E-MILES); Padova "
                        "isochrones, Salpeter IMF"),
    "galaxev": dict(file="spectra_galaxev_9.0.npz", size_mb=20.1, medium=None,
                    cite="Bruzual & Charlot 2003, MNRAS 344, 1000 (GALAXEV v2020)"),
    "fsps": dict(file="spectra_fsps_9.0.npz", size_mb=15.0, medium=None,
                 cite="Conroy et al. 2009, ApJ 699, 486; Conroy et al. 2010 (FSPS v3.2)"),
    "xsl": dict(file="spectra_xsl_9.0.npz", size_mb=93.8, medium=None,
                cite="Verro et al. 2022, A&A 661, A50 (X-Shooter library)"),
}


def cache_dir():
    path = os.environ.get("FANTASY_DATA") or os.path.join(os.path.expanduser("~"), ".fantasy_agn")
    return os.path.join(path, "templates")


def _path(name):
    if name not in TEMPLATE_SETS:
        raise ValueError(f"unknown template set {name!r}; choose from {sorted(TEMPLATE_SETS)}")
    return os.path.join(cache_dir(), TEMPLATE_SETS[name]["file"])


def available():
    """Template sets already downloaded."""
    return [n for n in TEMPLATE_SETS if os.path.exists(_path(n))]


def download(name="emiles", force=False):
    """Download a template set into the cache folder (once). Returns its path."""
    path = _path(name)
    if os.path.exists(path) and not force:
        return path
    info = TEMPLATE_SETS[name]
    os.makedirs(cache_dir(), exist_ok=True)
    url = _BASE + info["file"]
    print(f"Downloading {info['file']} ({info['size_mb']} MB) from {url}")
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    print(f"Saved to {path}. Please cite: {info['cite']}")
    return path


def load(name="emiles"):
    """
    Load a template set as a dict with ``lam`` (A), ``templates`` (n_lam, n_age,
    n_metal), ``ages`` (Gyr), ``metals`` ([M/H]), ``fwhm`` (instrumental FWHM in A
    at each wavelength), ``masses`` and ``medium``.
    """
    path = _path(name)
    if not os.path.exists(path):
        info = TEMPLATE_SETS[name]
        raise FileNotFoundError(
            f"stellar templates {name!r} not found. Download them once with\n"
            f"    fantasy_agn.templates.download({name!r})   # {info['size_mb']} MB from "
            f"github.com/micappe/ppxf_data")
    with np.load(path) as d:
        out = {k: d[k] for k in d.files}
    medium = TEMPLATE_SETS[name]["medium"]
    if medium is None:
        import warnings
        warnings.warn(f"the wavelength medium of {name!r} has not been verified; assuming air")
        medium = "air"
    out["medium"] = medium
    out["name"] = name
    return out
