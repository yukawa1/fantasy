"""
Galactic extinction: the Fitzpatrick (1999) law, and optional E(B-V) lookup
in the Schlegel, Finkbeiner & Davis (1998) dust maps.
"""
import os

import numpy as np
from scipy.interpolate import CubicSpline

__all__ = ("fitzpatrick99", "deredden", "ebv_sfd")


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


def ebv_sfd(ra, dec, map_dir=None):
    """
    E(B-V) from the SFD98 maps at (ra, dec) in degrees. Requires the optional
    ``sfdmap2`` package and the map files; ``map_dir`` defaults to the
    environment variable SFD_DIR.
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
    return float(sfdmap.SFDMap(map_dir).ebv(ra, dec))
