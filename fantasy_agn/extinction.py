"""
Galactic extinction: the Fitzpatrick (1999) law, and optional E(B-V) lookup
in the Schlegel, Finkbeiner & Davis (1998) dust maps.
"""
import os

import numpy as np
from scipy.interpolate import CubicSpline

__all__ = ("fitzpatrick99", "deredden", "ebv_sfd", "ebv_irsa", "galactic_ebv", "SF11")


def fitzpatrick99(wave, r_v=3.1):
    """
    A(lambda)/E(B-V) of the Fitzpatrick (1999) law for wavelengths in Angstrom
    (port of the IDL routine fm_unred).
    """
    x = 1e4 / np.asarray(wave, dtype=float)  # inverse microns
    curve = np.zeros_like(x)

    x0, gamma, c3, c4 = 4.596, 0.99, 3.23, 0.41
    c2 = -0.824 + 4.717 / r_v
    c1 = 2.030 - 3.007 * c2

    def uv(xx):
        xx = np.asarray(xx, float)
        y = c1 + c2 * xx + c3 * xx ** 2 / ((xx ** 2 - x0 ** 2) ** 2 + (xx * gamma) ** 2)
        d = np.maximum(xx, 5.9) - 5.9
        return y + c4 * (0.5392 * d ** 2 + 0.05644 * d ** 3) + r_v

    xcut_uv = 1e4 / 2700.0
    x_spl_uv = 1e4 / np.array([2700.0, 2600.0])
    is_uv = x >= xcut_uv
    curve[is_uv] = uv(x[is_uv])

    x_spl_opir = np.concatenate(([0.0], 1e4 / np.array([26500.0, 12200.0, 6000.0, 5470.0, 4670.0, 4110.0])))
    y_ir = np.array([0.0, 0.26469, 0.82925]) * r_v / 3.1
    y_op = [np.polyval(c[::-1], r_v) for c in (
        [-4.22809e-01, 1.00270, 2.13572e-04],
        [-5.13540e-02, 1.00216, -7.35778e-05],
        [7.00127e-01, 1.00184, -3.32598e-05],
        [1.19456, 1.01707, -5.46959e-03, 7.97809e-04, -4.45636e-05])]
    spline = CubicSpline(np.concatenate((x_spl_opir, x_spl_uv)),
                         np.concatenate((y_ir, y_op, uv(x_spl_uv))), bc_type="natural")
    curve[~is_uv] = spline(x[~is_uv])
    return curve


def deredden(wave, flux, ebv, r_v=3.1):
    """Correct flux for Galactic extinction E(B-V) (wave in observed frame, A)."""
    return flux * 10 ** (0.4 * ebv * fitzpatrick99(wave, r_v))


SF11 = 0.86          # Schlafly & Finkbeiner (2011): E(B-V) = 0.86 x E(B-V)_SFD
IRSA_DUST = "https://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust"


def ebv_sfd(ra, dec, map_dir=None):
    """
    E(B-V) of the SFD98 maps (not recalibrated) at (ra, dec) in degrees.
    Requires the optional ``sfdmap2`` package and the map files; ``map_dir``
    defaults to the environment variable SFD_DIR.
    """
    try:
        from sfdmap2 import sfdmap
    except ImportError:
        raise ImportError(
            "E(B-V) lookup needs the optional dependency: pip install sfdmap2, "
            "plus the SFD map files (set SFD_DIR). Alternatively pass ebv=... "
            "explicitly, e.g. from https://irsa.ipac.caltech.edu/applications/DUST/") from None
    map_dir = map_dir or os.environ.get("SFD_DIR")
    if not map_dir:
        raise ValueError("set SFD_DIR to the folder with the SFD .fits maps or pass map_dir")
    try:
        m = sfdmap.SFDMap(map_dir, scaling=1.0)
    except TypeError:
        m = sfdmap.SFDMap(map_dir)
    return float(m.ebv(ra, dec))


_IRSA_CACHE = {}


def ebv_irsa(ra, dec, timeout=20):
    """
    E(B-V) of the SFD98 maps at (ra, dec) from the IRSA dust service
    (NASA/IPAC, needs internet). Returns (SFD, Schlafly & Finkbeiner 2011).
    """
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET
    key = (round(float(ra), 4), round(float(dec), 4))     # ~0.4 arcsec; the map pixels are 2.4'
    if key in _IRSA_CACHE:
        return _IRSA_CACHE[key]
    query = urllib.parse.urlencode({"locstr": f"{float(ra):.6f} {float(dec):.6f} equ j2000",
                                    "regSize": "2.0"})
    with urllib.request.urlopen(f"{IRSA_DUST}?{query}", timeout=timeout) as resp:
        root = ET.fromstring(resp.read())
    if root.attrib.get("status", "ok") != "ok":
        raise RuntimeError(f"IRSA dust service: {root.findtext('.//message') or 'error'}")
    stats = root.find(".//statistics")           # the first block is E(B-V)
    if stats is None:
        raise RuntimeError("IRSA dust service: unexpected answer")

    def value(tag):
        return float(stats.findtext(tag).split()[0])
    _IRSA_CACHE[key] = (value("refPixelValueSFD"), value("refPixelValueSandF"))
    return _IRSA_CACHE[key]


def galactic_ebv(ra=None, dec=None, source="auto", sf11=True, file_value=None):
    """
    Galactic E(B-V) for a position, and a description of where it came from.

    :param source: 'auto' (the value in the file, then local SFD maps with
        sfdmap2, then the IRSA dust service online), 'file', 'sfd' or 'irsa'.
    :param sf11: apply the Schlafly & Finkbeiner (2011) recalibration of the
        SFD98 maps (factor 0.86), the usual choice today.
    :param file_value: an SFD98 E(B-V) stored in the data file (DESI: EBV).
    :return: (ebv, source description)
    """
    order = ["file", "sfd", "irsa"] if source == "auto" else [source]
    problems = []
    for s in order:
        try:
            if s == "file":
                if file_value is None:
                    raise ValueError("no E(B-V) in the file")
                sfd, where = float(file_value), "from the file (SFD98)"
            elif s == "sfd":
                if ra is None or dec is None:
                    raise ValueError("no coordinates")
                sfd, where = ebv_sfd(ra, dec), "SFD98 maps (sfdmap2)"
            elif s == "irsa":
                if ra is None or dec is None:
                    raise ValueError("no coordinates")
                sfd, sandf = ebv_irsa(ra, dec)
                if sf11:
                    return sandf, "IRSA dust service, Schlafly & Finkbeiner 2011"
                return sfd, "IRSA dust service, SFD98"
            else:
                raise ValueError(f"unknown source {s!r}")
            if sf11:
                return SF11 * sfd, where + " x 0.86 (Schlafly & Finkbeiner 2011)"
            return sfd, where
        except Exception as exc:                 # try the next source
            problems.append(f"{s}: {exc}")
    raise RuntimeError("no Galactic E(B-V) found (" + "; ".join(problems) + "); pass ebv=... "
                       "explicitly, e.g. from https://irsa.ipac.caltech.edu/applications/DUST/")
