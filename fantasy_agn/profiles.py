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

With ``pix`` (the pixel width at each ``x``, in Angstrom) a profile is the mean over the pixel
[x - pix/2, x + pix/2] instead of its value at x, which is what a spectrograph records: exactly
for Gaussians (through the error function), by Simpson's rule for the other profiles. This
matters for lines not much wider than a pixel: the narrow lines of SDSS spectra (69 km/s pixels),
whose widths point-sampled profiles overestimate by a few percent.
"""
import numpy as np
from scipy.special import ndtr, wofz

__all__ = ("PROFILES", "evaluate_profile", "gaussian_windowed")

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


def _gaussian(x, lam0, shift, fwhm, sig_inst, pix=None):
    sig_v = fwhm / FWHM_PER_SIGMA
    sig_eff = np.sqrt(sig_v ** 2 + sig_inst ** 2)
    c, dx = _centre(x, lam0, shift)
    sig_lam = c * sig_eff / C_KMS
    if pix is not None:
        return _gaussian_binned(dx, np.asarray(pix, float)[:, None], lam0, c, sig_lam, sig_v, sig_eff)
    u = dx / sig_lam
    G = np.exp(-0.5 * u * u) / (SQRT_2PI * sig_lam)
    d_shift = G * (lam0 / (C_KMS * sig_lam)) * (u + (u * u - 1) * sig_eff / C_KMS)
    d_fwhm = G * (u * u - 1) * sig_v / (sig_eff ** 2 * FWHM_PER_SIGMA)
    return G, [d_shift, d_fwhm]


def _bin_mean(dx, pix, c, sig_lam):
    """
    Mean of the unit Gaussian (centre c, sigma sig_lam) over pixels of width pix centred at
    c + dx, with its derivatives wrt c and sig_lam: (P, dP/dc, dP/dsig_lam).

    Where the Gaussian is wider than WIDE pixels, the mean is the Gaussian with the variance of
    the pixel (pix^2 / 12) added: they differ by the fourth cumulant of the pixel, below 1e-5 of
    the peak there, and it costs one exponential instead of two and two normal integrals.
    """
    ua, ub = (dx - 0.5 * pix) / sig_lam, (dx + 0.5 * pix) / sig_lam
    s2 = np.sqrt(sig_lam * sig_lam + pix * pix / 12.0)
    u = dx / s2
    G = np.exp(-0.5 * u * u) / (SQRT_2PI * s2)
    ds_dsig = sig_lam / s2
    P, dc, ds = G, G * u / s2, G * (u * u - 1) / s2 * ds_dsig
    narrow = sig_lam < WIDE * pix
    if np.any(narrow):
        a, b, sl, w = (np.broadcast_to(v, narrow.shape)[narrow] for v in (ua, ub, sig_lam, pix))
        ga = np.exp(-0.5 * a * a) / (SQRT_2PI * sl)          # the Gaussian at the pixel edges
        gb = np.exp(-0.5 * b * b) / (SQRT_2PI * sl)
        P[narrow] = (ndtr(b) - ndtr(a)) / w
        dc[narrow] = (ga - gb) / w
        ds[narrow] = (ga * a - gb * b) / w
    return P, dc, ds


WIDE = 4.0       # Gaussians wider than this many pixels (sigma) are averaged through their variance


def _gaussian_binned(dx, pix, lam0, c, sig_lam, sig_v, sig_eff):
    P, dc, ds = _bin_mean(dx, pix, c, sig_lam)
    # c = lam0 (1 + shift / C), sig_lam = c sig_eff / C, sig_eff^2 = (fwhm / 2.355)^2 + sig_inst^2
    d_shift = (dc + ds * sig_eff / C_KMS) * lam0 / C_KMS
    d_fwhm = ds * (c / C_KMS) * sig_v / (sig_eff * FWHM_PER_SIGMA)
    return P, [d_shift, d_fwhm]


# A Gaussian is evaluated only within WINDOW sigma of its centre: beyond 8 sigma it is below
# 1e-14 of its peak, so the result is the same to rounding, and templates with hundreds of lines
# (FeII, the high-order Balmer lines) fill a few percent of the pixels x lines matrix. 0: never.
WINDOW = 8.0


def gaussian_windowed(x, lam0, shift, fwhm, sig_inst, derivatives=True, entries=False, pix=None):
    """
    Like the Gaussian of :func:`evaluate_profile`, as sparse (len(x), n_lines) matrices
    (P, dP/dshift, dP/dfwhm; the last two None without ``derivatives``) holding each line only
    within WINDOW sigma of its centre. Returns None when that saves nothing (lines that cover
    most of the pixels) or ``x`` is not increasing: the caller then uses the dense profiles.
    With ``entries`` the non-zero elements of P themselves: (values, pixel index, line index).
    ``pix``: pixel widths, for profiles averaged over the pixels (see the module documentation).
    """
    if not WINDOW or x.ndim != 1 or x.size < 2 or np.any(x[1:] < x[:-1]):
        return None
    sig_v = fwhm / FWHM_PER_SIGMA
    sig_eff = np.sqrt(sig_v ** 2 + sig_inst ** 2)
    c = lam0 * (1 + shift / C_KMS)
    sig_lam = c * sig_eff / C_KMS
    if not np.all(sig_lam > 0):
        return None
    reach = WINDOW * sig_lam + (0.0 if pix is None else 0.5 * float(np.max(pix)))
    i0 = np.searchsorted(x, c - reach)
    n = np.searchsorted(x, c + reach) - i0
    total = int(n.sum())
    if total > 0.5 * x.size * lam0.size:
        return None
    from scipy.sparse import csc_matrix
    indptr = np.concatenate([[0], np.cumsum(n)])
    line = np.repeat(np.arange(lam0.size), n)
    width = None if pix is None else np.asarray(pix, float)
    pix = np.arange(total) - indptr[line] + i0[line]          # ordered by line, then pixel: CSC as it is
    s = sig_lam[line]
    if width is not None:
        G, (d_shift, d_fwhm) = _gaussian_binned(x[pix] - c[line], width[pix], lam0[line], c[line], s,
                                                sig_v[line], sig_eff[line])
        if entries:
            return G, pix, line
    else:
        u = (x[pix] - c[line]) / s
        G = np.exp(-0.5 * u * u) / (SQRT_2PI * s)
        if entries:
            return G, pix, line
    shape = (x.size, lam0.size)
    P = csc_matrix((G, pix, indptr), shape=shape)
    if not derivatives:
        return P, None, None
    if width is None:
        w = u * u - 1
        d_shift = G * (lam0[line] / (C_KMS * s)) * (u + w * sig_eff[line] / C_KMS)
        d_fwhm = G * w * (sig_v / (sig_eff ** 2 * FWHM_PER_SIGMA))[line]
    return P, csc_matrix((d_shift, pix, indptr), shape=shape), csc_matrix((d_fwhm, pix, indptr), shape=shape)


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


def _stratified(x, lam0, kin, sig_inst, pix=None):
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
        G, (gs, gf) = _gaussian(x, lam0, shift, fk, sig_inst, pix)
        P = P + w[k] * G
        d_shift = d_shift + w[k] * gs
        d_fwhm = d_fwhm + w[k] * gf * np.exp(-u / 2)
        # fwhm_k changes with s through u_k: d fwhm_k / ds = -fwhm_k / 2 * du_k/ds
        d_s = d_s + dws[k] * G + w[k] * gf * (-fk / 2 * dudss)
        d_g = d_g + dwg[k] * G
    return P, [d_shift, d_fwhm, d_s * np.log(10.0), d_g]


def evaluate_profile(name, x, lam0, kin, sig_inst, pix=None):
    """
    Profiles of all lines. ``kin`` is a list of per-line arrays in the order of
    ``PROFILES[name]``; returns (P, [dP/dkin_k ...]). With ``pix`` (pixel widths at ``x``) the
    profiles are means over the pixels.
    """
    if name == "gaussian":
        return _gaussian(x, lam0, kin[0], kin[1], sig_inst, pix)
    if name == "stratified":
        return _stratified(x, lam0, kin, sig_inst, pix)
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")
    if pix is None:
        return _point_profile(name, x, lam0, kin, sig_inst)
    # Simpson's rule over the pixel: (edge + 4 centre + edge) / 6, derivatives alike
    h = 0.5 * np.asarray(pix, float)
    parts = [_point_profile(name, x + d, lam0, kin, sig_inst) for d in (-h, 0.0, h)]
    mean = lambda a, b, c: (a + 4.0 * b + c) / 6.0
    return mean(*(q[0] for q in parts)), [mean(*(q[1][k] for q in parts)) for k in range(len(parts[0][1]))]


def _point_profile(name, x, lam0, kin, sig_inst):
    if name == "gauss-hermite":
        return _gauss_hermite(x, lam0, kin, sig_inst)
    if name == "lorentzian":
        shift, fwhm = kin
        V, d_shift, _, d_fwhm_l = _voigt(x, lam0, shift, np.zeros_like(fwhm), fwhm, sig_inst)
        return V, [d_shift, d_fwhm_l]
    shift, fwhm_g, fwhm_l = kin
    V, d_shift, d_g, d_l = _voigt(x, lam0, shift, fwhm_g, fwhm_l, sig_inst)
    return V, [d_shift, d_g, d_l]
