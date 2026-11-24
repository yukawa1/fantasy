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
import functools
import os

import numpy as np
from astropy.io import fits

from .lines import DATA, line_wavelength, load_template
from .parameters import Parameter
from .profiles import PROFILES, evaluate_profile, gaussian_windowed
from .wavelengths import vac_to_air

__all__ = ("Component", "Lines", "FeII", "HighOrderBalmer", "PowerLaw",
           "BrokenPowerLaw", "Blackbody", "Polynomial", "Calibration", "BalmerContinuum", "HostGalaxy", "Extinction",
           "Absorption", "calzetti00")

C_KMS = 299792.458
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
SQRT_2PI = np.sqrt(2.0 * np.pi)


def _grid_key(x):
    """Cache key of a wavelength grid: size, ends and a hash of every value (grids with the same ends and
    size can differ inside, e.g. masked differently)."""
    x = np.ascontiguousarray(x, dtype=float)
    return (x.size, float(x[0]), float(x[-1]), hash(x.tobytes())) if x.size else (0,)


def instrumental_fwhm(wave, ctx):
    """
    Instrumental FWHM (km/s) at the given wavelengths: from the spectrum's
    resolution curve when it has one (e.g. DESI), else its single value.
    """
    curve = ctx.get("resolution_curve")
    if curve is not None:
        return np.interp(wave, curve[0], curve[1])
    return np.full(np.shape(wave), ctx.get("resolution", 0.0), dtype=float)


def pixel_width(x, ctx):
    """
    Width (A) of the spectrum's pixel at each wavelength ``x``, for line profiles averaged over the
    pixels, or None when the context has no pixel grid (models evaluated outside a spectrum, the
    intrinsic line measurements, FANTASY_PIXEL=0).
    """
    grid = ctx.get("pixel")
    if grid is None:
        return None
    return np.interp(x, grid[0], grid[1])


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

    def __init_subclass__(cls, **kw):
        # remember how each component was made (class and arguments of the outermost constructor),
        # so that a model can be saved as a recipe and rebuilt (Model.to_dict / Model.from_dict)
        super().__init_subclass__(**kw)
        if "__init__" not in cls.__dict__:
            return
        init = cls.__init__

        @functools.wraps(init)
        def recorded(self, *args, **kwargs):
            if "_recipe" not in self.__dict__:
                self._recipe = (type(self).__name__, args, kwargs)
            init(self, *args, **kwargs)
        cls.__init__ = recorded

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

    def evaluate_basis(self, x, p, ctx):
        """
        The model and the columns of the Jacobian for the linear (amplitude)
        parameters; the other columns may be zero. Used by the linear solves of
        the initial guess, which do not need the nonlinear derivatives.
        Subclasses override this where it saves work.
        """
        return self.evaluate_with_derivatives(x, p, ctx)

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
        self._ratio_rows = np.zeros(0, int)      # lines whose flux is ratio * flux of another line
        self._ratio_cols = np.zeros(0, int)

    def _weights(self, p):
        """Line weights per amplitude; the free flux ratios (if any) are the last parameters."""
        rows = getattr(self, "_ratio_rows", ())         # absent in components saved before 2.0.0 had ratios
        if not len(rows):
            return self._W
        W = self._W.copy()
        W[rows, self._ratio_cols] = np.asarray(p, float)[len(p) - len(rows):]
        return W

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
        return evaluate_profile(self.profile, x, lam0, per_line, sig_inst, pixel_width(x, ctx))

    def _windowed(self, x, kin, ctx, derivatives=True):
        """Sparse Gaussian profiles (see profiles.gaussian_windowed), or None: use the dense ones."""
        if self.profile != "gaussian":
            return None
        lam0 = _in_medium(self._wave, ctx)
        sig_inst = instrumental_fwhm(lam0, ctx) / FWHM_PER_SIGMA
        return gaussian_windowed(x, lam0, kin[0][self._kin_index], kin[1][self._kin_index], sig_inst, derivatives,
                                 pix=pixel_width(x, ctx))

    def evaluate(self, x, p, ctx):
        amp, kin = self._split(p)
        sparse = self._windowed(x, kin, ctx)
        if sparse is not None:
            P, derivs = sparse[0], sparse[1:]
        else:
            P, derivs = self._profiles(x, kin, ctx)
        W = self._weights(p)
        line_amp = W @ amp
        f = P @ line_amp
        K = np.zeros((len(self._wave), self._n_kin))
        K[np.arange(len(self._wave)), self._kin_index] = line_amp
        cols = [P @ W] + [d @ K for d in derivs]
        if len(getattr(self, "_ratio_rows", ())):   # d f / d ratio = profile of the line * reference flux
            Pr = P[:, self._ratio_rows]
            cols.append((Pr.toarray() if sparse is not None else Pr) * amp[self._ratio_cols])
        return f, np.hstack(cols)

    def _unit_profiles(self, x, p, ctx):
        """Unit-flux profiles of the lines, values only (sparse where that is cheaper)."""
        _, kin = self._split(p)
        sparse = self._windowed(x, kin, ctx, derivatives=False)
        if sparse is not None:
            return sparse[0]
        lam0 = _in_medium(self._wave, ctx)
        per_line = [k[self._kin_index] for k in kin]
        sig_inst = instrumental_fwhm(lam0, ctx) / FWHM_PER_SIGMA
        pix = pixel_width(x, ctx)
        if pix is not None:
            return evaluate_profile(self.profile, x, lam0, per_line, sig_inst, pix)[0]
        if self.profile == "gaussian":            # values only: no derivative arrays
            centre = lam0 * (1 + per_line[0] / C_KMS)
            sig_lam = centre * np.sqrt((per_line[1] / FWHM_PER_SIGMA) ** 2 + sig_inst ** 2) / C_KMS
            u = (x[:, None] - centre) / sig_lam
            return np.exp(-0.5 * u * u) / (SQRT_2PI * sig_lam)
        return evaluate_profile(self.profile, x, lam0, per_line, sig_inst)[0]

    def value(self, x, p, ctx):
        return self._unit_profiles(x, p, ctx) @ (self._weights(p) @ p[:self._n_amp])

    def line_fluxes(self, p):
        """Integrated flux of every line (amplitude parameters mixed by W)."""
        return self._weights(p) @ np.asarray(p, float)[:self._n_amp]

    def line_profile(self, x, p, ctx, line):
        """Profile of one line only."""
        amp, kin = self._split(p)
        P, _ = self._profiles(x, kin, ctx)
        k = self.line_names.index(line)
        return P[:, k] * (self._weights(p)[k] @ amp)

    def evaluate_basis(self, x, p, ctx):
        A = self._unit_profiles(x, p, ctx) @ self._weights(p)
        J = np.zeros((x.size, len(p)))
        J[:, :self._n_amp] = A
        return A @ p[:self._n_amp], J

    def basis_entries(self, x, p, ctx):
        """
        The non-zero elements of the unit-flux line profiles, (values, pixel index, line index), or
        None without windowed profiles. With ``_weights(p)`` they give the amplitude columns; the
        grid searches of the fit build their normal matrix from these directly.
        """
        if self.profile != "gaussian":
            return None
        _, kin = self._split(p)
        lam0 = _in_medium(self._wave, ctx)
        sig_inst = instrumental_fwhm(lam0, ctx) / FWHM_PER_SIGMA
        return gaussian_windowed(x, lam0, kin[0][self._kin_index], kin[1][self._kin_index], sig_inst,
                                 derivatives=False, entries=True, pix=pixel_width(x, ctx))

    def basis(self, x, p, ctx):
        """Profile of each amplitude parameter at unit amplitude."""
        _, kin = self._split(p)
        P, _ = self._profiles(x, kin, ctx)
        return P @ self._weights(p)


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
    :param max_ratios: dict {line: (reference line, maximum ratio)} keeping
        flux(line) <= ratio * flux(reference), e.g. {'[NII]6583': ('Ha', 3)}.
        The reference line is then parametrised by its flux in excess of
        flux(line) / ratio (``<name>.<reference>.excess``, >= 0), which keeps
        the problem linear in the fluxes with simple bounds.
    :param line_ratios: dict {line: (reference line, (start, min, max))} making the flux ratio
        line/reference a free parameter within physical limits (``<name>.<line>.ratio``) instead of a
        free flux, e.g. {'Hg': ('Hb', (0.35, 0.1, 0.5))} keeps the Balmer decrement H-gamma/H-beta
        between 0.1 and 0.5 (case B: 0.47). The reference keeps its flux parameter.
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
        - 'stratified': a radially stratified, virialised BLR: Gaussians of FWHM
          ``fwhm`` (r/r_out)^(-1/2) summed from r_in to r_out = 10^``span`` r_in with
          dL/d ln r ~ r^(3 - ``gamma``). ``fwhm`` is the width at the outer edge (the
          core), fwhm 10^(span/2) the width at the inner edge; the wings fall
          roughly as a power law, v^-2 (Lorentzian-like) for gamma ~ 3. span = 0 is a single Gaussian.

        For non-Gaussian profiles, ``result.measure(line)`` gives the actual
        FWHM, line dispersion and centroid of the profile.
    :param h3, h4, fwhm_l: limits of the extra shape parameters, as for fwhm.
    """
    kind = "line"

    def __init__(self, name, lines, fwhm=(500, 50, 2000), shift=(0, -1000, 1000),
                 ratios=None, tie_kinematics=True, absorption=False, flux_max=np.inf,
                 role=None, profile="gaussian", h3=(0, -0.3, 0.3), h4=(0, -0.3, 0.3),
                 fwhm_l=None, max_ratios=None, span=(0.6, 0.0, 2.5), gamma=(2.5, 0.0, 5.0),
                 line_ratios=None):
        super().__init__(name)
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {sorted(PROFILES)}")
        self.profile = profile
        # the arguments, to rebuild the component with another profile (Model.set_profile)
        self._args = dict(lines=list(lines), fwhm=fwhm, shift=shift, ratios=ratios, tie_kinematics=tie_kinematics,
                          absorption=absorption, flux_max=flux_max, role=role, h3=h3, h4=h4, fwhm_l=fwhm_l,
                          max_ratios=max_ratios, span=span, gamma=gamma, line_ratios=line_ratios)
        self.absorption = bool(absorption)
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
        self.max_ratios = dict(max_ratios or {})
        self.line_ratios = {a: (b, _limits(r, (r, r, r)) if np.ndim(r) else (float(r),) * 3)
                            for a, (b, r) in (line_ratios or {}).items()}
        for a, (b, _) in self.line_ratios.items():
            if a not in self.line_names or b not in self.line_names or a == b or b in self.line_ratios:
                raise ValueError(f"line_ratios: bad entry {a!r}: reference {b!r}")
            if a in self.max_ratios or any(a == ref for ref, _ in self.max_ratios.values()):
                raise ValueError(f"line_ratios: {a!r} is also in max_ratios")
            if b in self.max_ratios:          # its parameter is the excess over the bound, not its flux
                raise ValueError(f"line_ratios: the reference {b!r} is bounded in max_ratios")
        refs = {b: a for a, (b, _) in self.max_ratios.items()}
        if len(refs) < len(self.max_ratios):
            raise ValueError("max_ratios: two lines cannot be bounded by the same reference")
        for a, (b, r) in self.max_ratios.items():
            if a not in self.line_names or b not in self.line_names or r <= 0 or a == b:
                raise ValueError(f"max_ratios: bad entry {a!r}: ({b!r}, {r!r})")
            if a in refs:
                raise ValueError("max_ratios: a line cannot be both bounded and a reference")
        for ln in self.line_names:
            if ln in self.line_ratios:
                continue
            if ln in refs:
                self._par(f"{ln}.excess", -1.0 if absorption else 1.0, lo, hi, linear=True,
                          unit="flux*A",
                          description=f"flux above {refs[ln]} / {self.max_ratios[refs[ln]][1]:g}")
            else:
                self._par(f"{ln}.flux", -1.0 if absorption else 1.0, lo, hi, linear=True,
                          unit="flux*A", description="integrated line flux")
        kin_names = self.line_names if not tie_kinematics else [None]
        fwhm_l = _limits(fwhm_l, (fwhm[0], max(fwhm[1], 1.0), fwhm[2]))
        specs = {"shift": (shift, "km/s", "velocity shift"),
                 "fwhm": (fwhm, "km/s", "Lorentzian FWHM" if profile == "lorentzian"
                          else "Gaussian FWHM" if profile == "voigt"
                          else "FWHM at the outer BLR edge" if profile == "stratified" else "intrinsic FWHM"),
                 "fwhm_l": (fwhm_l, "km/s", "Lorentzian FWHM"),
                 "h3": (_limits(h3, (0, -0.3, 0.3)), "", "Gauss-Hermite skewness"),
                 "h4": (_limits(h4, (0, -0.3, 0.3)), "", "Gauss-Hermite kurtosis"),
                 "span": (_limits(span, (0.6, 0.0, 2.5)), "dex", "log10 r_out/r_in of the stratified BLR"),
                 "gamma": (_limits(gamma, (2.5, 0.0, 5.0)), "", "emissivity index, dL/dlnr ~ r^(3-gamma)")}
        for par in PROFILES[profile]:
            limits, unit, description = specs[par]
            for kn in kin_names:
                self._par(par if kn is None else f"{kn}.{par}", *limits, unit=unit,
                          description=description)
        for a, (b, lim) in self.line_ratios.items():
            self._par(f"{a}.ratio", *lim, fixed=lim[1] == lim[2], unit="", description=f"flux ratio {a}/{b}")
        amp_lines = [ln for ln in self.line_names if ln not in self.line_ratios]
        col = {ln: k for k, ln in enumerate(amp_lines)}
        amp_index = [col[self.line_ratios[ln][0]] if ln in self.line_ratios else col[ln] for ln in self.line_names]
        weight = [self.line_ratios[ln][1][0] if ln in self.line_ratios else 1.0 for ln in self.line_names]
        self._setup([w for _, w in entries], weight, amp_index,
                    np.arange(n) if not tie_kinematics else np.zeros(n, int),
                    len(amp_lines), n if not tie_kinematics else 1)
        self._ratio_rows = np.array([self.line_names.index(a) for a in self.line_ratios], int)
        self._ratio_cols = np.array([col[b] for b, _ in self.line_ratios.values()], int)
        for a, (b, r) in self.max_ratios.items():
            self._W[self.line_names.index(b), col[a]] = 1.0 / r
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
    (2012) model (3700-10900 A); ``kind='uv'`` the UV model around Mg II of
    Kovacevic-Dojcinovic & Popovic (2015, 2690-3005 A: multiplets UV 60-63, 78 and
    the I Zw 1 lines); ``kind='uv_extended'`` a UV model built as the optical one
    (2200-3500 A: lines grouped by lower term, a4D split into UV 60-63 as in 2015,
    intensities at 10^4 K from NIST atomic data, and the I Zw 1 lines).
    """

    def __init__(self, name="feii", kind="optical", fwhm=(2000, 1000, 6000),
                 shift=(0, -1500, 1500), amp_max=np.inf):
        templates = {"optical": "feii", "uv": "feii_uv", "uv_extended": "feii_uv_extended"}
        if kind not in templates:
            raise ValueError(f"FeII kind must be one of {sorted(templates)}")
        self.feii_kind = kind
        super().__init__(name, templates[kind], fwhm, shift, amp_max)


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


class Blackbody(Component):
    """
    Blackbody continuum f = amp * B_lambda(x, T) / B_lambda(pivot, T); amp is the
    flux density at ``pivot`` and ``temperature`` the blackbody temperature (K).
    The hot (1-5e4 K) continuum of tidal disruption events; in the optical it is
    close to the Rayleigh-Jeans tail, so T is constrained only loosely without UV
    data (fix it if it is known).
    """
    kind = "continuum"
    C2 = 1.438776877e8          # h c / k in A K

    def __init__(self, name="continuum", pivot=5100.0, temperature=(20000.0, 5000.0, 80000.0)):
        super().__init__(name)
        self.pivot = float(pivot)
        self._par("amp", 1.0, 0.0, np.inf, linear=True, unit="flux/A",
                  description=f"flux density at {pivot:g} A")
        self._par("temperature", *_limits(temperature, (20000.0, 5000.0, 80000.0)), unit="K",
                  description="blackbody temperature")

    @classmethod
    def _log_b(cls, x, t):
        """ln B_lambda (up to a constant) and its derivative with respect to T."""
        y = cls.C2 / (x * t)
        em1 = np.expm1(y)
        return -5 * np.log(x) - np.log(em1), (y / t) * (em1 + 1) / em1

    def evaluate(self, x, p, ctx):
        amp, t = p
        lb, dlb = self._log_b(np.asarray(x, float), t)
        lp, dlp = self._log_b(self.pivot, t)
        shape = np.exp(lb - lp)
        f = amp * shape
        return f, np.column_stack([shape, f * (dlb - dlp)])


class BrokenPowerLaw(Component):
    """
    Power law with different indices blueward and redward of a break.

    :param pivot: break wavelength (A), fixed; ``amp`` is the flux density there.
    :param break_wave: (initial, min, max) to fit the break wavelength instead
        (parameter ``break``); then ``pivot`` is ignored and ``amp`` is the flux
        density at the fitted break.
    """
    kind = "continuum"

    def __init__(self, name="continuum", pivot=5100.0, index_blue=(-1.5, -5, 3),
                 index_red=(-1.5, -5, 3), break_wave=None):
        super().__init__(name)
        self.free_break = break_wave is not None
        self.pivot = float(pivot if break_wave is None else _limits(break_wave, (pivot, pivot, pivot))[0])
        self._par("amp", 1.0, 0.0, np.inf, linear=True, unit="flux/A",
                  description="flux density at the break" if self.free_break
                  else f"flux density at {pivot:g} A")
        self._par("index_blue", *_limits(index_blue, (-1.5, -5, 3)))
        self._par("index_red", *_limits(index_red, (-1.5, -5, 3)))
        if self.free_break:
            self._par("break", *_limits(break_wave, (pivot, pivot, pivot)), unit="A",
                      description="break wavelength")

    def evaluate(self, x, p, ctx):
        amp, ib, ir = p[:3]
        pivot = p[3] if self.free_break else self.pivot
        r = x / pivot
        blue = r < 1
        shape = np.where(blue, r ** ib, r ** ir)
        f = amp * shape
        logr = np.log(r)
        cols = [shape, np.where(blue, f * logr, 0), np.where(blue, 0, f * logr)]
        if self.free_break:
            cols.append(-np.where(blue, ib, ir) * f / pivot)
        return f, np.column_stack(cols)


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
    just blueward of the edge (before broadening). Temperature and optical
    depth are fixed by default (15000 K, tau=1), as is common practice. The
    edge is smeared by the velocities of the emitting gas: the shape is
    convolved with a Gaussian of FWHM ``fwhm`` (km/s; default 5000, fixed, as
    in QSFit, Calderone et al. 2017); fwhm=0 gives a sharp edge.
    """
    kind = "continuum"
    numeric_derivatives = ("temperature", "tau", "fwhm")
    EDGE = 3646.0

    def __init__(self, name="balmer_cont", temperature=15000.0, tau=1.0,
                 fix_shape=True, fwhm=5000.0, fix_fwhm=True):
        super().__init__(name)
        self._par("amp", 1.0, 0.0, np.inf, linear=True, unit="flux/A",
                  description="flux density at the Balmer edge")
        self._par("temperature", temperature, 5000.0, 50000.0, fixed=fix_shape, unit="K")
        self._par("tau", tau, 0.1, 10.0, fixed=fix_shape)
        self._par("fwhm", fwhm, 0.0, 20000.0, fixed=fix_fwhm, unit="km/s",
                  description="velocity broadening of the edge")
        self._cache = {}

    @staticmethod
    def _planck(lam, T):
        hc_k = 1.4387769e8  # h c / k in A K
        return lam ** -5.0 / np.expm1(hc_k / (lam * T))

    def shape(self, x, T, tau, fwhm=0.0):
        edge = self.EDGE
        if fwhm <= 0:
            s = (self._planck(x, T) / self._planck(edge, T)
                 * -np.expm1(-tau * (x / edge) ** 3) / -np.expm1(-tau))
            return np.where(x <= edge, s, 0.0)
        key = (float(T), float(tau), float(fwhm), _grid_key(x))
        hit = self._cache.get(key)          # (threads share the cache: no re-lookups)
        if hit is None:
            sigma = fwhm / C_KMS / FWHM_PER_SIGMA            # in ln(lambda)
            step = 2e-5                                      # fixed: smooth in fwhm
            lo, hi = np.log(min(x.min(), edge) / (1 + 6 * sigma)), np.log(max(x.max(), edge) * (1 + 6 * sigma))
            grid = np.arange(lo, hi + step, step)
            sharp = self.shape(np.exp(grid), T, tau, 0.0)
            half = int(np.ceil(5 * sigma / step))
            u = np.arange(-half, half + 1) * step
            kern = np.exp(-0.5 * (u / sigma) ** 2)
            from scipy.signal import fftconvolve          # the kernel is long: FFT
            smooth = fftconvolve(sharp, kern / kern.sum(), mode="same")
            hit = np.interp(np.log(x), grid, smooth)
            if len(self._cache) > 8:
                self._cache = {}
            self._cache[key] = hit
        return hit

    def evaluate(self, x, p, ctx):
        amp, T, tau, fwhm = p
        s = self.shape(x, T, tau, fwhm)
        nan = np.full_like(x, np.nan)
        return amp * s, np.column_stack([s, nan, nan, nan])

    def __deepcopy__(self, memo):
        import copy
        new = copy.copy(self)
        new.params = copy.deepcopy(self.params, memo)
        new._cache = {}
        return new


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
        key = (_grid_key(x), ctx.get("medium"))
        hit = self._cache.get(key)
        if hit is None:
            wave = _in_medium(self._eig_wave, ctx)
            hit = np.column_stack([np.interp(x, wave, e, left=0.0, right=0.0) for e in self._eig])
            self._cache = {key: hit}
        return hit

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


class Calibration(Component):
    """
    Smooth multiplicative correction of the whole model, 1 + sum_k c_k P_k(u)
    (Legendre polynomials P_1..P_degree of u in [-1, 1] over ``wave_range``),
    as the multiplicative polynomial of full-spectrum fitting (pPXF ``mdegree``,
    Cappellari 2017). It absorbs the errors of the spectrophotometric
    calibration and of the dust law, at the 1-3% level over hundreds of A,
    which in spectra of S/N > ~40 are as large as the noise. The constant term
    is fixed (1): fluxes keep the calibration of the data on average.

    :param wave_range: (min, max) wavelength over which u runs from -1 to 1.
    :param degree: number of Legendre terms; keep it low (3-8 over the optical)
        so that it cannot follow absorption or emission features.
    :param limit: bound on every coefficient.
    :param start: first Legendre term; 2 leaves out the linear tilt, which is
        degenerate with a fitted dust screen (E(B-V) of the stars or the gas).
    """
    kind = "calibration"

    def __init__(self, wave_range, degree=4, name="calibration", limit=0.3, start=1):
        super().__init__(name)
        self.wave_range = (float(wave_range[0]), float(wave_range[1]))
        self.degree = int(degree)
        self.start = int(start)
        for k in range(self.start, self.degree + 1):
            self._par(f"c{k}", 0.0, -limit, limit, description=f"Legendre coefficient {k}")

    def _basis(self, x):
        from numpy.polynomial import legendre
        a, b = self.wave_range
        u = np.clip(2 * (np.asarray(x, float) - a) / (b - a) - 1, -1, 1)
        return np.stack([legendre.legval(u, np.eye(self.degree + 1)[k]) for k in range(self.start, self.degree + 1)],
                        axis=1)

    def evaluate(self, x, p, ctx):
        B = self._basis(x)
        return 1.0 + B @ np.asarray(p, float), B


class Absorption(Component):
    """
    Absorption troughs in front of the AGN (intervening or outflowing gas: CIV,
    SiIV, MgII absorbers, mini-BAL and BAL troughs): every component is
    multiplied by exp(-tau(lambda)), with tau a sum of Gaussians in velocity.
    Each trough k has the central optical depth ``a<k>.tau``, the wavelength
    ``a<k>.wave`` (A) and the width ``a<k>.fwhm`` (km/s). The emission
    components then describe the unabsorbed spectrum.

        model.add(fa.Absorption([(1510.5, 1200), (1547.0, 300)]))   # (wavelength, FWHM)

    :func:`fantasy_agn.absorbers.fit_absorbed` finds the troughs in the residuals
    of a fit and adds them automatically.

    :param troughs: list of (wavelength, FWHM km/s) or (wavelength, FWHM, tau) starting values.
    :param shift_max: how far (km/s) each centre may move from its start.
    :param fwhm_range: (min, max) FWHM in km/s.
    """
    kind = "absorption"

    def __init__(self, troughs, name="absorption", shift_max=500.0, fwhm_range=(30.0, 20000.0),
                 tau_max=10.0):
        super().__init__(name)
        self.n = len(troughs)
        for k, t in enumerate(troughs):
            wave, fwhm = float(t[0]), float(t[1])
            tau = float(t[2]) if len(t) > 2 else 0.5
            dw = wave * shift_max / C_KMS
            self._par(f"a{k}.tau", min(max(tau, 1e-3), tau_max), 0.0, tau_max,
                      description="central optical depth of the trough")
            self._par(f"a{k}.wave", wave, wave - dw, wave + dw, unit="A", description="trough wavelength")
            self._par(f"a{k}.fwhm", float(np.clip(fwhm, *fwhm_range)), *fwhm_range, unit="km/s",
                      description="trough FWHM")

    def evaluate(self, x, p, ctx):
        p = np.asarray(p, float).reshape(self.n, 3)
        tau = np.zeros_like(x)
        cols = []
        for t0, lam, fwhm in p:
            sig = lam * fwhm / C_KMS / FWHM_PER_SIGMA
            u = (x - lam) / sig
            g = np.exp(-0.5 * u * u)
            tau += t0 * g
            # d tau / d (t0, lam, fwhm)
            cols.append((g, t0 * g * (u / sig + u * u / lam), t0 * g * u * u / fwhm))
        T = np.exp(-tau)
        J = np.empty((x.size, 3 * self.n))
        for k, (dt0, dlam, dfwhm) in enumerate(cols):
            J[:, 3 * k] = -T * dt0
            J[:, 3 * k + 1] = -T * dlam
            J[:, 3 * k + 2] = -T * dfwhm
        return T, J
