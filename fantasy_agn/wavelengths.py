"""Air/vacuum wavelength conversion (Ciddor 1996, as used by SDSS)."""
import numpy as np

__all__ = ("vac_to_air", "air_to_vac")


def _refraction_factor(lam_vac):
    sigma2 = (1e4 / np.asarray(lam_vac, dtype=float)) ** 2
    return 1 + 5.792105e-2 / (238.0185 - sigma2) + 1.67917e-3 / (57.362 - sigma2)


def vac_to_air(lam_vac):
    """Convert vacuum wavelengths (Angstrom) to air."""
    return np.asarray(lam_vac, dtype=float) / _refraction_factor(lam_vac)


def air_to_vac(lam_air):
    """Convert air wavelengths (Angstrom) to vacuum (iterative inverse)."""
    lam_air = np.asarray(lam_air, dtype=float)
    lam_vac = lam_air.copy()
    for _ in range(4):
        lam_vac = lam_air * _refraction_factor(lam_vac)
    return lam_vac
