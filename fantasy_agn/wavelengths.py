"""Air/vacuum wavelength conversion (Ciddor 1996, as used by SDSS)."""
import numpy as np

__all__ = ("vac_to_air", "air_to_vac")


def _sampling_gaps(steps):
    """Indices of detector gaps and the local pixel spacing on either side."""
    if len(steps) < 2:
        return np.array([], dtype=int), steps[:0], steps[:0]
    left = np.r_[steps[1], steps[:-1]]
    right = np.r_[steps[1:], steps[-2]]
    gaps = np.flatnonzero(steps > 3 * np.maximum(left, right))
    return gaps, left[gaps], right[gaps]


def _refraction_factor(lam_vac):
    """n_air (Ciddor 1996) above 2000 A; 1 below, the IAU convention: wavelengths below 2000 A are vacuum
    wavelengths in either medium (the formula has a pole at 1320 A and is not meant for the UV)."""
    lam = np.asarray(lam_vac, dtype=float)
    return np.where(lam >= 2000.0, _ciddor(lam), 1.0)


def _ciddor(lam_vac):
    sigma2 = (1e4 / np.maximum(lam_vac, 2000.0)) ** 2
    return 1 + 5.792105e-2 / (238.0185 - sigma2) + 1.67917e-3 / (57.362 - sigma2)


def vac_to_air(lam_vac):
    """Convert vacuum wavelengths (Angstrom) to air."""
    return np.asarray(lam_vac, dtype=float) / _refraction_factor(lam_vac)


def air_to_vac(lam_air):
    """Convert air wavelengths (Angstrom) to vacuum (iterative inverse)."""
    lam_air = np.asarray(lam_air, dtype=float)
    lam_vac = lam_air.copy()
    for _ in range(4):
        lam_vac = lam_air * _ciddor(lam_vac)
    edge = 2000.0 / _ciddor(2000.0)           # vac_to_air(2000): below it air and vacuum are the same
    return np.where(lam_air >= edge, lam_vac, lam_air)
