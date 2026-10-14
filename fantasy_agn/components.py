"""
Model components with analytic derivatives.

Each component declares its parameters and implements
``evaluate(x, p, ctx) -> (f, J)`` where ``p`` are the component's parameter
values (in declaration order), ``f`` the model on wavelengths ``x`` and ``J``
the derivatives df/dp with shape (len(x), len(p)). Components that cannot
provide a derivative for some parameter list it in ``numeric_derivatives``;
``evaluate_with_derivatives`` then fills that column by finite differences.

``ctx`` carries the spectrum's wavelength medium ('air' or 'vacuum') and its
instrumental resolution (FWHM in km/s), which is added in quadrature to every
line width so fitted widths are intrinsic.
"""
import os

import numpy as np
from astropy.io import fits

from .lines import DATA, line_wavelength, load_template
from .parameters import Parameter
from .profiles import PROFILES, evaluate_profile
from .wavelengths import vac_to_air

__all__ = ("Component", "Lines", "FeII", "HighOrderBalmer", "PowerLaw",
           "BrokenPowerLaw", "Polynomial", "BalmerContinuum", "HostGalaxy", "Extinction",
           "calzetti00")

C_KMS = 299792.458
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
SQRT_2PI = np.sqrt(2.0 * np.pi)


def instrumental_fwhm(wave, ctx):
    """
    Instrumental FWHM (km/s) at the given wavelengths: from the spectrum's
    resolution curve when it has one (e.g. DESI), else its single value.
    """
    curve = ctx.get("resolution_curve")
    if curve is not None:
        return np.interp(wave, curve[0], curve[1])
    return np.full(np.shape(wave), ctx.get("resolution", 0.0), dtype=float)


def _in_medium(wave_vac, ctx):
    return vac_to_air(wave_vac) if ctx.get("medium", "vacuum") == "air" else wave_vac


def _limits(spec, default):
    """Accept a number (initial value) or a tuple (value, min, max)."""
    if spec is None:
        return default
    if np.isscalar(spec):
        return (float(spec), default[1], default[2])
    if len(spec) == 2:
        lo, hi = spec
        return (np.sqrt(lo * hi) if lo > 0 else 0.5 * (lo + hi), lo, hi)
    return tuple(float(v) for v in spec)


class Component:
    """Base class. Subclasses fill ``self.params`` and implement ``evaluate``."""
    kind = "other"           # 'line', 'continuum', 'host' or 'template'
    numeric_derivatives = ()  # local parameter names without analytic derivative

    def __init__(self, name):
        if "." in name:
            raise ValueError("component names cannot contain '.'")
        self.name = name
        self.params = []

    def _par(self, local, value, min=-np.inf, max=np.inf, linear=False,
             fixed=False, unit="", description=""):
        p = Parameter(f"{self.name}.{local}", float(value), float(min), float(max),
                      fixed=fixed, linear=linear, unit=unit, description=description)
        self.params.append(p)
        return p

    def local_names(self):
        n = len(self.name) + 1
        return [p.name[n:] for p in self.params]

    def evaluate(self, x, p, ctx):
        raise NotImplementedError

    def value(self, x, p, ctx):
        """The model alone, without derivatives (overridden where that is cheaper)."""
        return self.evaluate(x, p, ctx)[0]

    def evaluate_with_derivatives(self, x, p, ctx):
        """Like evaluate, with finite differences for ``numeric_derivatives``."""
        f, J = self.evaluate(x, p, ctx)
        if self.numeric_derivatives:
            J = np.array(J, dtype=float)
            for k, local in enumerate(self.local_names()):
                if local in self.numeric_derivatives:
                    q = np.array(p, dtype=float)
                    h = 1e-6 * max(abs(q[k]), 1.0)
                    q[k] += h
                    J[:, k] = (self.evaluate(x, q, ctx)[0] - f) / h
        return f, J

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r}, {len(self.params)} parameters)"


# --------------------------------------------------------------------- lines
class _GaussianSet(Component):
    """
    Sum of Gaussian lines. Line i has vacuum wavelength wave[i], relative weight
    weight[i], amplitude parameter amp_index[i] and kinematic group kin_index[i]
    (each kinematic group has a velocity shift and FWHM in km/s). Amplitudes are
    integrated fluxes, so a line's flux is amplitude * weight.
    """

    profile = "gaussian"

    def _setup(self, wave, weight, amp_index, kin_index, n_amp, n_kin):
        self._wave = np.asarray(wave, float)
        self._weight = np.asarray(weight, float)
        self._amp_index = np.asarray(amp_index, int)
        self._kin_index = np.asarray(kin_index, int)
        self._n_amp, self._n_kin = n_amp, n_kin
        self._W = np.zeros((len(self._wave), n_amp))
        self._W[np.arange(len(self._wave)), self._amp_index] = self._weight

    def _split(self, p):
        """Amplitudes and one array per kinematic parameter (shift, fwhm, ...)."""
        n_par = len(PROFILES[self.profile])
        amp = p[:self._n_amp]
        kin = [p[self._n_amp + k * self._n_kin:self._n_amp + (k + 1) * self._n_kin]
               for k in range(n_par)]
        return amp, kin

    def _profiles(self, x, kin, ctx):
        """Unit-flux profiles and their derivatives wrt each kinematic parameter."""
        lam0 = _in_medium(self._wave, ctx)
        per_line = [k[self._kin_index] for k in kin]
        sig_inst = instrumental_fwhm(lam0, ctx) / FWHM_PER_SIGMA
        return evaluate_profile(self.profile, x, lam0, per_line, sig_inst)

    def evaluate(self, x, p, ctx):
        amp, kin = self._split(p)
        P, derivs = self._profiles(x, kin, ctx)
        line_amp = amp[self._amp_index] * self._weight
        f = P @ line_amp
        K = np.zeros((len(self._wave), self._n_kin))
        K[np.arange(len(self._wave)), self._kin_index] = line_amp
        J = np.hstack([P @ self._W] + [d @ K for d in derivs])
        return f, J

    def value(self, x, p, ctx):
        amp, kin = self._split(p)
        lam0 = _in_medium(self._wave, ctx)
        per_line = [k[self._kin_index] for k in kin]
        sig_inst = instrumental_fwhm(lam0, ctx) / FWHM_PER_SIGMA
        if self.profile == "gaussian":
            centre = lam0 * (1 + per_line[0] / C_KMS)
            sig_lam = centre * np.sqrt((per_line[1] / FWHM_PER_SIGMA) ** 2 + sig_inst ** 2) / C_KMS
            u = (x[:, None] - centre) / sig_lam
            P = np.exp(-0.5 * u * u) / (SQRT_2PI * sig_lam)
        else:
            P, _ = evaluate_profile(self.profile, x, lam0, per_line, sig_inst)
        return P @ (amp[self._amp_index] * self._weight)

    def basis(self, x, p, ctx):
        """Profile of each amplitude parameter at unit amplitude."""
        _, kin = self._split(p)
        P, _ = self._profiles(x, kin, ctx)
        return P @ self._W


class Lines(_GaussianSet):
    """
    A group of Gaussian emission lines, e.g. all narrow lines or all broad lines.

    Parameters are the integrated flux of each line (``<name>.<line>.flux``) and,
    when ``tie_kinematics`` is True (default), one shared velocity shift
    (``<name>.shift``, km/s) and FWHM (``<name>.fwhm``, km/s). With
    ``tie_kinematics=False`` every line has its own shift and FWHM.

    :param name: component name, e.g. 'narrow', 'broad'.
    :param lines: line names from the library (e.g. 'Hb', '[OIII]5007') or
        (name, vacuum wavelength) pairs.
    :param fwhm: initial FWHM or (min, max) or (initial, min, max), km/s.
    :param shift: initial shift or (min, max) or (initial, min, max), km/s.
    :param ratios: dict {line: (reference line, flux ratio)} fixing flux ratios,
        e.g. {'[OIII]4959': ('[OIII]5007', 1/2.98)}.
    :param absorption: if True, fluxes are negative (absorption lines).
    :param role: 'narrow', 'broad', 'outflow' or None (guessed from the name,
        else from the initial FWHM). Used to pick components for black-hole
        masses and plots.
    :param profile: line shape:

        - 'gaussian' (default): ``fwhm`` is the intrinsic FWHM.
        - 'lorentzian': ``fwhm`` is the Lorentzian FWHM.
        - 'voigt': ``fwhm`` is the Gaussian FWHM and ``fwhm_l`` the Lorentzian FWHM.
        - 'gauss-hermite': Gaussian core ``fwhm`` with skewness ``h3`` and
          kurtosis ``h4`` (van der Marel & Franx 1993); the flux does not
          depend on h3 and h4.

        For non-Gaussian profiles, ``result.measure(line)`` gives the actual
        FWHM, line dispersion and centroid of the profile.
    :param h3, h4, fwhm_l: limits of the extra shape parameters, as for fwhm.
    """
    kind = "line"

    def __init__(self, name, lines, fwhm=(500, 50, 2000), shift=(0, -1000, 1000),
                 ratios=None, tie_kinematics=True, absorption=False, flux_max=np.inf,
                 role=None, profile="gaussian", h3=(0, -0.3, 0.3), h4=(0, -0.3, 0.3),
                 fwhm_l=None):
        super().__init__(name)
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {sorted(PROFILES)}")
        self.profile = profile
        entries = []
        for item in lines:
            if isinstance(item, str):
                entries.append((item, line_wavelength(item)))
            else:
                entries.append((str(item[0]), float(item[1])))
        self.line_names = [n for n, _ in entries]
        if len(set(self.line_names)) != len(self.line_names):
            raise ValueError(f"duplicate line names in {name!r}")
        n = len(entries)
        fwhm = _limits(fwhm, (500, 50, 2000))
        shift = _limits(shift, (0, -1000, 1000))
        if role is None:
            role = next((r for r in ("narrow", "broad", "outflow") if r in name.lower()),
                        "broad" if fwhm[0] > 1500 else "narrow")
        self.role = role
        lo, hi = (-flux_max, 0.0) if absorption else (0.0, flux_max)
        for ln in self.line_names:
            self._par(f"{ln}.flux", -1.0 if absorption else 1.0, lo, hi, linear=True,
                      unit="flux*A", description="integrated line flux")
        kin_names = self.line_names if not tie_kinematics else [None]
        fwhm_l = _limits(fwhm_l, (fwhm[0], max(fwhm[1], 1.0), fwhm[2]))
        specs = {"shift": (shift, "km/s", "velocity shift"),
                 "fwhm": (fwhm, "km/s", "Lorentzian FWHM" if profile == "lorentzian"
                          else "Gaussian FWHM" if profile == "voigt" else "intrinsic FWHM"),
                 "fwhm_l": (fwhm_l, "km/s", "Lorentzian FWHM"),
                 "h3": (_limits(h3, (0, -0.3, 0.3)), "", "Gauss-Hermite skewness"),
                 "h4": (_limits(h4, (0, -0.3, 0.3)), "", "Gauss-Hermite kurtosis")}
        for par in PROFILES[profile]:
            limits, unit, description = specs[par]
            for kn in kin_names:
                self._par(par if kn is None else f"{kn}.{par}", *limits, unit=unit,
                          description=description)
        self._setup([w for _, w in entries], np.ones(n), np.arange(n),
                    np.arange(n) if not tie_kinematics else np.zeros(n, int),
                    n, n if not tie_kinematics else 1)
        self.ratios = dict(ratios or {})

    def default_ties(self):
        return [(f"{self.name}.{a}.flux", f"{self.name}.{b}.flux", r)
                for a, (b, r) in self.ratios.items()]


class _Template(_GaussianSet):
    kind = "template"

    def __init__(self, name, template, fwhm, shift, amp_max, single_amplitude=False,
                 max_wavelength=None):
        super().__init__(name)
        wave, inten, groups, index = load_template(template)
        if max_wavelength is not None:
            keep = wave < max_wavelength
            wave, inten, index = wave[keep], inten[keep], index[keep]
        if single_amplitude:
            groups, index = ["amp"], np.zeros(len(wave), int)
        self.groups = groups
        for g in groups:
            self._par(g, 1.0, 0.0, amp_max, linear=True,
                      description="multiplet strength" if not single_amplitude else "strength")
        self._par("shift", *_limits(shift, (0, -1000, 1000)), unit="km/s")
        self._par("fwhm", *_limits(fwhm, (2000, 1000, 6000)), unit="km/s")
        self._setup(wave, inten, index, np.zeros(len(wave), int), len(groups), 1)


class FeII(_Template):
    """
    FeII emission template: lines grouped in multiplets with relative
    intensities fixed within each multiplet and a common shift and FWHM.
    ``kind='optical'`` uses the Kovacevic et al. (2010) / Shapovalova et al.
    (2012) model (3700-10900 A); ``kind='uv'`` the UV template around MgII.
    """

    def __init__(self, name="feii", kind="optical", fwhm=(2000, 1000, 6000),
                 shift=(0, -1500, 1500), amp_max=np.inf):
        super().__init__(name, "feii" if kind == "optical" else "feii_uv",
                         fwhm, shift, amp_max)


class HighOrderBalmer(_Template):
    """
    High-order Balmer lines blending towards the Balmer edge (3646 A), with
    fixed relative intensities and one common amplitude, shift and FWHM.

    Only lines blueward of ``max_wavelength`` are included. The default (4000 A)
    starts at H-epsilon (3970 A), so that H-gamma and H-delta, which are usually
    fitted as separate lines, are not counted twice. Do not also fit H-epsilon
    separately unless you lower ``max_wavelength``.
    """

    def __init__(self, name="balmer_lines", fwhm=(3000, 1000, 10000),
                 shift=(0, -1500, 1500), max_wavelength=4000.0):
        super().__init__(name, "balmer", fwhm, shift, np.inf, single_amplitude=True,
                         max_wavelength=max_wavelength)


# ----------------------------------------------------------------- continua
class PowerLaw(Component):
    """f = amp * (x / pivot) ** index; amp is the flux density at ``pivot``."""
    kind = "continuum"

    def __init__(self, name="continuum", pivot=5100.0, index=(-1.5, -5, 3)):
        super().__init__(name)
        self.pivot = float(pivot)
        self._par("amp", 1.0, 0.0, np.inf, linear=True, unit="flux/A",
                  description=f"flux density at {pivot:g} A")
        self._par("index", *_limits(index, (-1.5, -5, 3)))

    def evaluate(self, x, p, ctx):
        amp, index = p
        r = x / self.pivot
        shape = r ** index
        f = amp * shape
        return f, np.column_stack([shape, f * np.log(r)])


class BrokenPowerLaw(Component):
    """Power law with different indices blueward and redward of ``pivot``."""
    kind = "continuum"

    def __init__(self, name="continuum", pivot=5100.0, index_blue=(-1.5, -5, 3),
                 index_red=(-1.5, -5, 3)):
        super().__init__(name)
        self.pivot = float(pivot)
        self._par("amp", 1.0, 0.0, np.inf, linear=True, unit="flux/A",
                  description=f"flux density at {pivot:g} A")
        self._par("index_blue", *_limits(index_blue, (-1.5, -5, 3)))
        self._par("index_red", *_limits(index_red, (-1.5, -5, 3)))

    def evaluate(self, x, p, ctx):
        amp, ib, ir = p
        r = x / self.pivot
        blue = r < 1
        shape = np.where(blue, r ** ib, r ** ir)
        f = amp * shape
        logr = np.log(r)
        return f, np.column_stack([shape, np.where(blue, f * logr, 0), np.where(blue, 0, f * logr)])


class Polynomial(Component):
    """Polynomial in (x - pivot) / scale; all coefficients are linear."""
    kind = "continuum"

    def __init__(self, name="poly", order=2, pivot=5100.0, scale=1000.0):
        super().__init__(name)
        self.pivot, self.scale = float(pivot), float(scale)
        for k in range(order + 1):
            self._par(f"c{k}", 1.0 if k == 0 else 0.0, linear=True)

    def evaluate(self, x, p, ctx):
        t = (x - self.pivot) / self.scale
        J = np.vander(t, len(p), increasing=True)
        return J @ p, J


class BalmerContinuum(Component):
    """
    Balmer continuum (Grandi 1982; Dietrich et al. 2002): optically thick
    Planck function below the Balmer edge (3646 A). ``amp`` is the flux density
    just blueward of the edge. Temperature and optical depth are fixed by
    default (15000 K, tau=1), as is common practice.
    """
    kind = "continuum"
    numeric_derivatives = ("temperature", "tau")
    EDGE = 3646.0

    def __init__(self, name="balmer_cont", temperature=15000.0, tau=1.0,
                 fix_shape=True):
        super().__init__(name)
        self._par("amp", 1.0, 0.0, np.inf, linear=True, unit="flux/A",
                  description="flux density at the Balmer edge")
        self._par("temperature", temperature, 5000.0, 50000.0, fixed=fix_shape, unit="K")
        self._par("tau", tau, 0.1, 10.0, fixed=fix_shape)

    @staticmethod
    def _planck(lam, T):
        hc_k = 1.4387769e8  # h c / k in A K
        return lam ** -5.0 / np.expm1(hc_k / (lam * T))

    def shape(self, x, T, tau):
        edge = self.EDGE
        s = (self._planck(x, T) / self._planck(edge, T)
             * -np.expm1(-tau * (x / edge) ** 3) / -np.expm1(-tau))
        return np.where(x <= edge, s, 0.0)

    def evaluate(self, x, p, ctx):
        amp, T, tau = p
        s = self.shape(x, T, tau)
        return amp * s, np.column_stack([s, np.full_like(x, np.nan), np.full_like(x, np.nan)])


# Emission-line windows (vacuum, A) removed from the galaxy eigenspectra, which
# contain emission lines that would otherwise absorb AGN narrow-line flux.
EMISSION_WINDOWS = [(3718, 3740), (3860, 3880), (4092, 4112), (4331, 4351), (4853, 4873),
                    (4952, 4972), (4999, 5019), (6290, 6312), (6355, 6375), (6540, 6595),
                    (6710, 6745)]


def remove_emission_lines(wave, spectra):
    """Interpolate spectra (rows) linearly across EMISSION_WINDOWS."""
    bad = np.zeros(wave.shape, bool)
    for lo, hi in EMISSION_WINDOWS:
        bad |= (wave > lo) & (wave < hi)
    out = np.array(spectra, dtype=float, copy=True)
    for row in np.atleast_2d(out):
        row[bad] = np.interp(wave[bad], wave[~bad], row[~bad])
    return out


class HostGalaxy(Component):
    """
    Host galaxy as a linear combination of the SDSS galaxy eigenspectra of
    Yip et al. (2004), with their emission lines interpolated over. Fitting it
    together with the AGN components propagates the host uncertainty into all
    other parameters. Coefficients are unbounded
    (eigenspectra can enter with either sign); check that the resulting host
    spectrum is positive.
    """
    kind = "host"

    def __init__(self, name="host", n_components=5):
        super().__init__(name)
        with fits.open(os.path.join(DATA, "gal_eigenspec_Yip2004.fits")) as hdul:
            d = hdul[1].data
            self._eig_wave = np.asarray(d["wave"], float).ravel()
            pca = np.asarray(d["pca"], float)
            pca = pca.reshape(pca.shape[-2], pca.shape[-1])
        if n_components > pca.shape[0]:
            raise ValueError(f"at most {pca.shape[0]} galaxy eigenspectra available")
        # Normalise each eigenspectrum to unit rms so coefficients are comparable.
        eig = remove_emission_lines(self._eig_wave, pca[:n_components])
        self._eig = eig / np.sqrt(np.mean(eig ** 2, axis=1, keepdims=True))
        for k in range(n_components):
            self._par(f"c{k}", 1.0 if k == 0 else 0.0, linear=True)
        self._cache = {}

    @property
    def coverage(self):
        return self._eig_wave[0], self._eig_wave[-1]

    def _basis(self, x, ctx):
        key = (x.size, x[0], x[-1], ctx.get("medium"))
        if key not in self._cache:
            wave = _in_medium(self._eig_wave, ctx)
            self._cache = {key: np.column_stack(
                [np.interp(x, wave, e, left=0.0, right=0.0) for e in self._eig])}
        return self._cache[key]

    def evaluate(self, x, p, ctx):
        B = self._basis(x, ctx)
        return B @ p, B


# ---------------------------------------------------------------- extinction
def calzetti00(wave, r_v=4.05):
    """A(lambda)/E(B-V) of the Calzetti et al. (2000) starburst law (wave in A)."""
    lam = np.asarray(wave, float) / 1e4
    blue = 2.659 * (-2.156 + 1.509 / lam - 0.198 / lam ** 2 + 0.011 / lam ** 3) + r_v
    red = 2.659 * (-1.857 + 1.040 / lam) + r_v
    return np.where(lam < 0.63, blue, red)


class Extinction(Component):
    """
    Intrinsic dust screen on the AGN: every component except the host galaxy
    is multiplied by 10**(-0.4 E(B-V) k(lambda)). Fitted line fluxes,
    continuum fluxes and luminosities are then the dereddened (intrinsic)
    values, with the extinction uncertainty included in their errors; plots
    show the observed (reddened) components.

    :param law: 'F99' (Fitzpatrick 1999, Milky Way, R_V = 3.1) or 'calzetti'
        (Calzetti et al. 2000, R_V = 4.05).
    :param ebv: initial value and limits of E(B-V), mag.

    E(B-V) is degenerate with the slope of the power law over narrow
    wavelength ranges; it is best constrained over a wide range or with
    fixed line ratios (e.g. a tied narrow Balmer decrement).
    """
    kind = "attenuation"

    def __init__(self, name="extinction", law="F99", ebv=(0.1, 0.0, 2.0), r_v=None):
        super().__init__(name)
        if law not in ("F99", "calzetti"):
            raise ValueError("law must be 'F99' or 'calzetti'")
        self.law = law
        self.r_v = r_v if r_v is not None else (3.1 if law == "F99" else 4.05)
        self._par("ebv", *_limits(ebv, (0.1, 0.0, 2.0)), unit="mag",
                  description="intrinsic E(B-V) of the AGN")

    def curve(self, x):
        from .extinction import fitzpatrick99
        return fitzpatrick99(x, self.r_v) if self.law == "F99" else calzetti00(x, self.r_v)

    def evaluate(self, x, p, ctx):
        """Transmission and its derivative with respect to E(B-V)."""
        k = self.curve(x)
        T = 10 ** (-0.4 * p[0] * k)
        return T, (-0.4 * np.log(10) * k * T)[:, None]
