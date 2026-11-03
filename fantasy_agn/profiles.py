"""
Unit-flux line profiles with exact derivatives.

Every function returns ``(P, derivs)``: ``P`` has shape (len(x), n_lines) and
integrates to one over wavelength, and ``derivs`` holds dP/d(parameter) for each
kinematic parameter in the order given by ``PROFILES[name]``. Velocities and
widths are in km/s; the line centre is ``lam0 * (1 + shift / c)`` and widths
scale with the centre wavelength. The instrumental resolution ``sig_inst``
(Gaussian sigma, km/s) is included: exactly for Gaussian, Lorentzian and Voigt
profiles (whose convolution with a Gaussian is again Voigt), and by adding it
in quadrature to the Gaussian core for Gauss-Hermite profiles.
"""
import numpy as np
from scipy.special import wofz

__all__ = ("PROFILES", "evaluate_profile")

C_KMS = 299792.458
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
SQRT_2PI = np.sqrt(2.0 * np.pi)
SQRT_PI = np.sqrt(np.pi)

# kinematic parameters of each profile, in order
PROFILES = {
    "gaussian": ("shift", "fwhm"),
    "lorentzian": ("shift", "fwhm"),
    "voigt": ("shift", "fwhm", "fwhm_l"),
    "gauss-hermite": ("shift", "fwhm", "h3", "h4"),
    "stratified": ("shift", "fwhm", "span", "gamma"),
}

# Gauss-Hermite polynomials normalised as in van der Marel & Franx (1993)
_H3 = lambda y: (2 * np.sqrt(2) * y ** 3 - 3 * np.sqrt(2) * y) / np.sqrt(6)
_dH3 = lambda y: (6 * np.sqrt(2) * y ** 2 - 3 * np.sqrt(2)) / np.sqrt(6)
_H4 = lambda y: (4 * y ** 4 - 12 * y ** 2 + 3) / np.sqrt(24)
_dH4 = lambda y: (16 * y ** 3 - 24 * y) / np.sqrt(24)
_H4_FLUX = np.sqrt(6) / 4  # integral of the Gaussian times H4


def _centre(x, lam0, shift):
    centre = lam0 * (1 + shift / C_KMS)
    return centre, x[:, None] - centre


def _gaussian(x, lam0, shift, fwhm, sig_inst):
    sig_v = fwhm / FWHM_PER_SIGMA
    sig_eff = np.sqrt(sig_v ** 2 + sig_inst ** 2)
    c, dx = _centre(x, lam0, shift)
    sig_lam = c * sig_eff / C_KMS
    u = dx / sig_lam
    G = np.exp(-0.5 * u * u) / (SQRT_2PI * sig_lam)
    d_shift = G * (lam0 / (C_KMS * sig_lam)) * (u + (u * u - 1) * sig_eff / C_KMS)
    d_fwhm = G * (u * u - 1) * sig_v / (sig_eff ** 2 * FWHM_PER_SIGMA)
    return G, [d_shift, d_fwhm]


def _gauss_hermite(x, lam0, kin, sig_inst):
    shift, fwhm = kin[0], kin[1]
    h3 = kin[2] if len(kin) > 2 else np.zeros_like(shift)
    h4 = kin[3] if len(kin) > 3 else np.zeros_like(shift)
    sig_v = fwhm / FWHM_PER_SIGMA
    sig_eff = np.sqrt(sig_v ** 2 + sig_inst ** 2)
    c, dx = _centre(x, lam0, shift)
    sig_lam = c * sig_eff / C_KMS
    y = dx / sig_lam
    # The h4 term integrates to sqrt(6)/4 * h4 (the polynomials are orthogonal
    # with respect to exp(-y^2), not exp(-y^2/2)); dividing by that factor makes
    # the profile integrate to one, so the amplitude is the line flux.
    norm = 1 + _H4_FLUX * h4
    G0 = np.exp(-0.5 * y * y) / (SQRT_2PI * sig_lam) / norm
    shape = 1 + h3 * _H3(y) + h4 * _H4(y)
    P = G0 * shape
    dP_dy = G0 * (-y * shape + h3 * _dH3(y) + h4 * _dH4(y))
    dP_dc = dP_dy * (-1 / sig_lam - y / c) - P / c
    dP_dsig = (dP_dy * (-y) - P) / sig_lam
    derivs = [dP_dc * lam0 / C_KMS,
              dP_dsig * (c / C_KMS) * sig_v / (sig_eff * FWHM_PER_SIGMA)]
    if len(kin) > 2:
        derivs += [G0 * _H3(y), G0 * _H4(y) - P * _H4_FLUX / norm]
    return P, derivs


def _voigt(x, lam0, shift, fwhm_g, fwhm_l, sig_inst):
    """Voigt profile and derivatives wrt shift, Gaussian FWHM and Lorentzian FWHM."""
    sig_v = fwhm_g / FWHM_PER_SIGMA
    gam_v = fwhm_l / 2.0
    # A pure Lorentzian with no instrumental broadening is the limit sigma -> 0;
    # keep sigma tiny but positive so the Faddeeva function stays defined.
    sig_eff = np.sqrt(sig_v ** 2 + sig_inst ** 2 + (1e-4 * np.maximum(gam_v, 1.0)) ** 2)
    c, dx = _centre(x, lam0, shift)
    sig_lam = c * sig_eff / C_KMS
    gam_lam = c * gam_v / C_KMS
    root2s = np.sqrt(2.0) * sig_lam
    z = (dx + 1j * gam_lam) / root2s
    w = wofz(z)
    norm = sig_lam * SQRT_2PI
    V = w.real / norm
    dw = -2 * z * w + 2j / SQRT_PI
    dV_dc_part = (dw * (-1 / root2s)).real / norm
    dV_dgam = (dw * (1j / root2s)).real / norm
    dV_dsig = (dw * (-z / sig_lam)).real / norm - V / sig_lam
    dV_dc = dV_dc_part + dV_dsig * sig_lam / c + dV_dgam * gam_lam / c
    d_shift = dV_dc * lam0 / C_KMS
    d_fwhm_g = dV_dsig * (c / C_KMS) * sig_v / (sig_eff * FWHM_PER_SIGMA)
    d_fwhm_l = dV_dgam * c / (2 * C_KMS)
    return V, d_shift, d_fwhm_g, d_fwhm_l


# Radially stratified BLR: a continuous sum of Gaussians over radius. Gas at radius
# r has a Gaussian velocity distribution of FWHM v(r) = fwhm (r/r_out)^(-1/2)
# (virial), and emits with a volume emissivity j ~ r^(-gamma): dL = 4 pi r^2 j dr,
# so dL/d ln r ~ r^(3 - gamma) between r_in and r_out = 10^span r_in. The line
# wings fall roughly as a power law (steeper than the v^(2 gamma - 7) of a single
# velocity per radius): gamma ~ 3 gives Lorentzian-like v^-2 wings; span = 0 is a
# single Gaussian. Gauss-Legendre quadrature in u = ln(r/r_out).
_STRAT_T, _STRAT_Q = np.polynomial.legendre.leggauss(12)


def _stratified(x, lam0, kin, sig_inst):
    shift, fwhm, span, gamma = (np.asarray(k, float) for k in kin)
    s = span * np.log(10.0)                         # ln(r_out / r_in), per line
    a = 3.0 - gamma
    P = 0.0
    d_shift = d_fwhm = d_s = d_g = 0.0
    # nodes: u_k = -s (1 - t_k) / 2 in [-s, 0]; du_k/ds = -(1 - t_k) / 2
    terms = []
    logw = []
    for t, q in zip(_STRAT_T, _STRAT_Q):
        dudss = -(1 - t) / 2
        u = s * dudss
        logw.append(np.log(q) + a * u)
        terms.append((u, dudss))
    logw = np.array(logw)                           # (K, n_lines)
    w = np.exp(logw - logw.max(axis=0))
    w /= w.sum(axis=0)
    # d ln w~ / ds = a du/ds (the 1/s Jacobian cancels on normalisation); d ln w~ / dgamma = -u
    dls = np.array([a * dudss for _, dudss in terms])
    dlg = np.array([-u for u, _ in terms])
    dws = w * (dls - (w * dls).sum(axis=0))
    dwg = w * (dlg - (w * dlg).sum(axis=0))
    for k, (u, dudss) in enumerate(terms):
        fk = fwhm * np.exp(-u / 2)
        G, (gs, gf) = _gaussian(x, lam0, shift, fk, sig_inst)
        P = P + w[k] * G
        d_shift = d_shift + w[k] * gs
        d_fwhm = d_fwhm + w[k] * gf * np.exp(-u / 2)
        # fwhm_k changes with s through u_k: d fwhm_k / ds = -fwhm_k / 2 * du_k/ds
        d_s = d_s + dws[k] * G + w[k] * gf * (-fk / 2 * dudss)
        d_g = d_g + dwg[k] * G
    return P, [d_shift, d_fwhm, d_s * np.log(10.0), d_g]


def evaluate_profile(name, x, lam0, kin, sig_inst):
    """
    Profiles of all lines. ``kin`` is a list of per-line arrays in the order of
    ``PROFILES[name]``; returns (P, [dP/dkin_k ...]).
    """
    if name == "gaussian":
        return _gaussian(x, lam0, kin[0], kin[1], sig_inst)
    if name == "gauss-hermite":
        return _gauss_hermite(x, lam0, kin, sig_inst)
    if name == "stratified":
        return _stratified(x, lam0, kin, sig_inst)
    if name == "lorentzian":
        shift, fwhm = kin
        V, d_shift, _, d_fwhm_l = _voigt(x, lam0, shift, np.zeros_like(fwhm), fwhm, sig_inst)
        return V, [d_shift, d_fwhm_l]
    if name == "voigt":
        shift, fwhm_g, fwhm_l = kin
        V, d_shift, d_g, d_l = _voigt(x, lam0, shift, fwhm_g, fwhm_l, sig_inst)
        return V, [d_shift, d_g, d_l]
    raise ValueError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")
