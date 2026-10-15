"""
Host galaxy from stellar population templates, with stellar kinematics.

The host is a non-negative combination of simple-stellar-population (SSP)
templates, shifted by the stellar velocity v* and broadened by the stellar
velocity dispersion sigma*. Unlike the PCA eigenspectra of HostGalaxy, the host
cannot become negative, and sigma* and v* are measured.

How it is computed:

1. The templates are converted to the spectrum's wavelength medium and placed
   on a logarithmic wavelength grid, where a velocity shift and a Gaussian
   broadening are the same at every wavelength.
2. They are broadened once to a common resolution: that of the data, or, if
   the templates are coarser somewhere in the fitted range, the coarsest
   template resolution there. sigma* is then unbiased at all wavelengths as
   long as it is larger than ``sigma_floor`` = sqrt(common^2 - data^2), the
   smallest dispersion these templates can measure in this range (about
   40 km/s for E-MILES and SDSS in the H-beta region).
3. For every trial (v*, sigma*) they are shifted and broadened in Fourier
   space with the analytic kernel exp(-2 pi i f v - 2 pi^2 f^2 sigma^2); the
   derivatives with respect to v* and sigma* follow exactly from the same
   kernel.
4. The template weights are linear, non-negative parameters, solved together
   with all other amplitudes.
"""
import numpy as np
import pandas as pd
from scipy import fft

from .components import Component, C_KMS, FWHM_PER_SIGMA, _limits, instrumental_fwhm
from .extinction import fitzpatrick99
from .templates import TEMPLATE_SETS, load, canonical
from .wavelengths import air_to_vac

__all__ = ("StellarPopulation",)

DEFAULT_METALS = (-0.71, 0.0, 0.22)


class StellarPopulation(Component):
    """
    Host galaxy as a non-negative combination of stellar population templates
    with stellar velocity ``velocity`` and dispersion ``sigma`` (km/s).

    :param templates: template set ('emiles' by default; see fantasy_agn.templates).
    :param ages: ages in Gyr to use (nearest available), default every second
        age of the grid.
    :param metals: metallicities [M/H] to use, default (-0.71, 0, +0.22).
    :param velocity, sigma: initial value and limits, (value, min, max) in km/s.
    :param reddening: fit a dust screen E(B-V) (Fitzpatrick 1999) on the host.
    :param dv: velocity step of the internal log-wavelength grid (km/s).

    Template weights are named ``<name>.w00``, ``<name>.w01``, ...; the
    ``grid`` attribute lists the age and metallicity of each. After a fit,
    ``result.host_properties()`` reports sigma*, v*, the host fraction and the
    light-weighted age and metallicity with uncertainties.
    """
    kind = "host"

    def __init__(self, name="host", templates="emiles", ages=None, metals=DEFAULT_METALS,
                 velocity=(0, -600, 600), sigma=(150, 30, 450), reddening=False,
                 ebv=(0.1, 0.0, 1.5), dv=20.0):
        super().__init__(name)
        templates = canonical(templates)            # 'bc03' -> 'galaxev'
        data = load(templates)
        all_ages, all_metals = data["ages"], data["metals"]
        age_idx = (np.arange(0, all_ages.size, 2) if ages is None else
                   np.unique([np.argmin(abs(all_ages - a)) for a in np.atleast_1d(ages)]))
        met_idx = np.unique([np.argmin(abs(all_metals - m)) for m in np.atleast_1d(metals)])
        cube = data["templates"][:, age_idx][:, :, met_idx]
        self._lam_t = np.asarray(data["lam"], float)
        self._T = cube.reshape(cube.shape[0], -1).T.copy()          # (n_tpl, n_lam)
        self._fwhm_t = np.asarray(data["fwhm"], float)
        self._medium_t = data["medium"]
        aa, mm = np.meshgrid(all_ages[age_idx], all_metals[met_idx], indexing="ij")
        self.grid = pd.DataFrame({"age_gyr": aa.ravel(), "metal": mm.ravel()})
        self.grid.index = [f"w{k:02d}" for k in range(len(self.grid))]
        self.template_set = templates
        self.citation = TEMPLATE_SETS[templates]["cite"]
        self.dv = float(dv)
        self.reddening = reddening
        for k in self.grid.index:
            self._par(k, 1.0 if k == "w00" else 0.0, 0.0, np.inf, linear=True,
                      description="template weight (flux density at the normalisation)")
        self._par("velocity", *_limits(velocity, (0, -600, 600)), unit="km/s",
                  description="stellar velocity")
        self._par("sigma", *_limits(sigma, (150, 30, 450)), unit="km/s",
                  description="stellar velocity dispersion")
        if reddening:
            self._par("ebv", *_limits(ebv, (0.1, 0.0, 1.5)), unit="mag",
                      description="host E(B-V)")
        self._setup_key = None
        self._eval_cache = (None, None)
        self._reference = None
        self.sigma_floor = 0.0

    @property
    def n_templates(self):
        return len(self.grid)

    # Arrays that are never modified after creation; copies of the component
    # (models are deep-copied for every fit and every epoch) share them.
    _SHARED = ("_lam_t", "_T", "_fwhm_t", "_reference") + (
        "_T_hat", "_freq", "_i0", "_frac", "_k_ext", "_norms")

    def __deepcopy__(self, memo):
        import copy
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k in self._SHARED:
                new.__dict__[k] = v
            elif k == "_setups":
                new.__dict__[k] = dict(v)       # prepared grids are read-only too
            elif k == "_eval_cache":
                new.__dict__[k] = (None, None)
            else:
                new.__dict__[k] = copy.deepcopy(v, memo)
        return new

    # ----------------------------------------------------------- setup
    _STATE = ("_T_hat", "_freq", "_n_fft", "_i0", "_frac", "_k_ext", "_floor2", "sigma_floor",
              "_norms")

    def _setup(self, x, ctx):
        curve = ctx.get("resolution_curve")
        key = (x.size, float(x[0]), float(x[-1]), ctx.get("medium"), ctx.get("resolution"),
               None if curve is None else float(np.median(curve[1])))
        if key == self._setup_key:
            return
        cache = self.__dict__.setdefault("_setups", {})
        if key in cache:
            for name, value in zip(self._STATE, cache[key]):
                setattr(self, name, value)
            self._setup_key = key
            return
        lam_t = self._lam_t
        if ctx.get("medium", "vacuum") == "vacuum" and self._medium_t == "air":
            lam_t = np.where(lam_t > 2000, air_to_vac(lam_t), lam_t)
        vel = [p for p in self.params if p.name.endswith(".velocity")][0]
        sig = [p for p in self.params if p.name.endswith(".sigma")][0]
        reach = max(abs(vel.min), abs(vel.max)) + 6 * sig.max + 200
        lo, hi = x.min() * np.exp(-reach / C_KMS), x.max() * np.exp(reach / C_KMS)
        if lo < lam_t[0] or hi > lam_t[-1]:
            raise ValueError(f"templates cover {lam_t[0]:.0f}-{lam_t[-1]:.0f} A; the fit "
                             f"needs {lo:.0f}-{hi:.0f} A")
        step = self.dv / C_KMS
        ln_grid = np.arange(np.log(lo), np.log(hi), step)
        lam_g = np.exp(ln_grid)
        T = np.array([np.interp(lam_g, lam_t, t) for t in self._T])

        # Resolution matching. Templates are broadened to a common resolution
        # sig_c (the data's, or the coarsest template resolution in the fitted
        # range if that is larger). A galaxy observed with resolution sig_d
        # then needs an extra kernel sig_k^2 = sigma*^2 + sig_d^2 - sig_c^2, so
        # sigma* is unbiased at all wavelengths as long as it exceeds
        # sigma_floor = sqrt(sig_c^2 - sig_d^2).
        # A resolution that varies with wavelength (e.g. DESI) is represented by
        # its median over the fitted range.
        sig_d = float(np.median(instrumental_fwhm(x, ctx))) / FWHM_PER_SIGMA
        sig_t = np.interp(lam_g, lam_t, self._fwhm_t) / FWHM_PER_SIGMA / lam_g * C_KMS
        if self.__dict__.get("_reference") is None:
            # The common resolution and the template normalisation are fixed at
            # the first evaluation (the fitted range), so the host is the same
            # function of wavelength wherever it is evaluated later (plots,
            # host fractions, other epochs).
            inside = (lam_g >= x.min()) & (lam_g <= x.max())
            sig_c = max(sig_d, float(sig_t[inside].max()) if inside.any() else sig_d)
            ref = (self._lam_t >= x.min()) & (self._lam_t <= x.max())
            norms = np.median(self._T[:, ref], axis=1) if ref.any() else np.ones(len(self._T))
            norms[norms <= 0] = 1.0
            self._reference = (sig_c, norms, (float(x.min()), float(x.max())))
        sig_c, norms, _ = self._reference
        T = _variable_gaussian(T, np.sqrt(np.clip(sig_c ** 2 - sig_t ** 2, 0, None)) / self.dv)
        self._floor2 = max(sig_c ** 2 - sig_d ** 2, 0.0)
        self.sigma_floor = float(np.sqrt(self._floor2))
        T /= norms[:, None]
        self._norms = norms

        pad = int(np.ceil(reach / self.dv))
        n_fft = fft.next_fast_len(T.shape[1] + 2 * pad)
        Tp = np.zeros((T.shape[0], n_fft))
        Tp[:, pad:pad + T.shape[1]] = T
        self._T_hat = fft.rfft(Tp, axis=1)
        self._freq = fft.rfftfreq(n_fft, d=self.dv)              # cycles per km/s
        self._n_fft = n_fft
        pos = (np.log(x) - ln_grid[0]) / step + pad
        self._i0 = np.floor(pos).astype(int)
        self._frac = pos - self._i0
        self._k_ext = fitzpatrick99(x)
        self._setup_key = key
        if len(cache) >= 8:
            cache.pop(next(iter(cache)))
        cache[key] = tuple(getattr(self, name) for name in self._STATE)

    def _interp(self, Y):
        return (Y[:, self._i0] * (1 - self._frac) + Y[:, self._i0 + 1] * self._frac).T

    def _broadened(self, v, s):
        key = (self._setup_key, float(v), float(s))
        if self._eval_cache[0] == key:
            return self._eval_cache[1]
        f = self._freq
        # extra broadening sig_k^2 = sigma*^2 - floor^2 (zero below the floor)
        sk2 = max(s * s - self._floor2, 0.0)
        kernel = np.exp(-2j * np.pi * f * v - 2 * np.pi ** 2 * f ** 2 * sk2)
        base = self._T_hat * kernel
        B = self._interp(fft.irfft(base, n=self._n_fft, axis=1))
        dBv = self._interp(fft.irfft(base * (-2j * np.pi * f), n=self._n_fft, axis=1))
        ds = (-4 * np.pi ** 2 * f ** 2 * s) if sk2 > 0 else np.zeros_like(f)
        dBs = self._interp(fft.irfft(base * ds, n=self._n_fft, axis=1))
        out = (B, dBv, dBs)
        self._eval_cache = (key, out)
        return out

    # ------------------------------------------------------- evaluation
    def value(self, x, p, ctx):
        x = np.asarray(x, float)
        self._setup(x, ctx)
        n = self.n_templates
        v, s = p[n], p[n + 1]
        sk2 = max(s * s - self._floor2, 0.0)
        f = self._freq
        Y = fft.irfft((p[:n] @ self._T_hat)[None, :]
                      * np.exp(-2j * np.pi * f * v - 2 * np.pi ** 2 * f ** 2 * sk2),
                      n=self._n_fft, axis=1)
        stars = self._interp(Y)[:, 0]
        return stars * 10 ** (-0.4 * p[n + 2] * self._k_ext) if self.reddening else stars

    def evaluate(self, x, p, ctx):
        x = np.asarray(x, float)
        self._setup(x, ctx)
        n = self.n_templates
        w, v, s = p[:n], p[n], p[n + 1]
        B, dBv, dBs = self._broadened(v, s)
        att = 10 ** (-0.4 * p[n + 2] * self._k_ext) if self.reddening else 1.0
        stars = B @ w
        f = att * stars
        cols = [B * np.atleast_1d(att)[:, None] if self.reddening else B,
                (att * (dBv @ w))[:, None], (att * (dBs @ w))[:, None]]
        if self.reddening:
            cols.append((-0.4 * np.log(10) * self._k_ext * f)[:, None])
        return f, np.hstack(cols)


def _variable_gaussian(T, sigma_pix):
    """Convolve rows of T with a Gaussian whose width (pixels) varies along the row."""
    if not np.any(sigma_pix > 0.05):
        return T
    half = int(np.ceil(4 * sigma_pix.max())) + 1
    s = np.maximum(sigma_pix, 1e-3)
    offsets = np.arange(-half, half + 1)
    weights = np.exp(-0.5 * (offsets[:, None] / s[None, :]) ** 2)
    weights /= weights.sum(axis=0, keepdims=True)
    padded = np.pad(T, ((0, 0), (half, half)), mode="edge")
    out = np.zeros_like(T)
    n = T.shape[1]
    for i, o in enumerate(offsets):
        out += weights[i][None, :] * padded[:, half + o:half + o + n]
    return out
