"""Fit results: parameter and line tables with uncertainties, plots, export."""
import json

import numpy as np
import pandas as pd

from .components import C_KMS, Lines

__all__ = ("FitResult",)

# Single-epoch virial black-hole mass recipes:
# log M = a + 2 log(FWHM / 1000 km/s) + b log(L / 1e44 erg/s)
BH_RECIPES = {
    "Hb": dict(a=6.91, b=0.50, lum=5100, ref="Vestergaard & Peterson 2006"),
    "MgII2798": dict(a=6.86, b=0.50, lum=3000, ref="Vestergaard & Osmer 2009"),
    "CIV1549": dict(a=6.66, b=0.53, lum=1350, ref="Vestergaard & Peterson 2006"),
}


class FitResult:
    """
    Outcome of :func:`fantasy_agn.fit`. Main attributes and methods:

    - ``params``: table of parameters with uncertainties
    - ``lines``: every line's flux, FWHM, shift and EW, with errors
    - ``measure(line)``: a line summed over components (e.g. total broad
      H-beta): flux, FWHM, sigma, centroid shift and EW, with errors
    - ``continuum(5100)``: AGN continuum flux density, with error
    - ``luminosity(5100)``: log lambda L_lambda (needs z and flux units)
    - ``bh_mass()``: single-epoch virial black-hole mass
    - ``plot()``, ``summary()``, ``save(prefix)``, ``to_dataframe()``
    """

    def __init__(self, spectrum, model, mask, full, cov, samples, chi2, dof, nfev,
                 success, message, at_limit, error_scale, time, error_method, at_zero=(),
                 unconstrained=()):
        self.spectrum, self.model, self.mask = spectrum, model, mask
        self.full, self.cov, self.samples = full, cov, samples
        self.chi2, self.dof, self.rchi2 = chi2, dof, chi2 / dof
        self.nfev, self.success, self.message = nfev, success, message
        self.at_limit, self.error_scale, self.time = at_limit, error_scale, time
        # amplitudes that went to a bound (usually 0: component not detected)
        self.at_zero = list(at_zero)
        # amplitudes fixed at 0 because they do not affect the fitted pixels
        self.unconstrained = list(unconstrained)
        self.error_method = error_method

    # ------------------------------------------------------- propagation
    def _sigma(self):
        if self.cov is None:
            return np.full(self.full.size, np.nan)
        return np.sqrt(np.clip(np.diag(self.cov), 0, None))

    def propagate(self, func):
        """
        Value and 1-sigma error of any function of the full parameter vector,
        from Monte Carlo samples if available, else by linear propagation of the
        covariance (finite-difference gradient on the scale of each error).
        Returns (value, error) with arrays if func returns an array.
        """
        value = np.asarray(func(self.full), float)
        if self.samples is not None:
            vals = np.array([func(s) for s in self.samples], float)
            lo, hi = np.nanpercentile(vals, [15.865, 84.135], axis=0)
            return value, 0.5 * (hi - lo)
        if self.cov is None:
            return value, np.full_like(value, np.nan)
        sig = self._sigma()
        idx = np.nonzero(sig > 0)[0]
        grad = np.zeros(value.shape + (self.full.size,))
        for i in idx:
            h = 0.1 * sig[i]
            a, b = self.full.copy(), self.full.copy()
            a[i] += h
            b[i] -= h
            grad[..., i] = (np.asarray(func(a)) - np.asarray(func(b))) / (2 * h)
        g = grad[..., idx]
        C = self.cov[np.ix_(idx, idx)]
        var = np.einsum("...i,ij,...j->...", g, C, g)
        return value, np.sqrt(np.clip(var, 0, None))

    # -------------------------------------------------------------- tables
    @property
    def params(self):
        sig = self._sigma()
        rows = []
        for i, p in enumerate(self.model.params):
            status = ("at limit" if p.name in self.at_limit else
                      "at bound" if p.name in self.at_zero else
                      "outside range" if p.name in self.unconstrained else
                      "free" if p.free else "tied" if p.tie else "fixed")
            err = sig[i] if p.free or p.tie else 0.0
            if self.samples is not None and (p.free or p.tie):
                lo, hi = np.percentile(self.samples[:, i], [15.865, 84.135])
                err = 0.5 * (hi - lo)
            if status in ("at limit", "at bound") and self.samples is None:
                err = np.nan
            rows.append((p.name, p.value, err, p.min, p.max, p.unit, status))
        return pd.DataFrame(rows, columns=["parameter", "value", "error", "min", "max",
                                           "unit", "status"])

    def _line_components(self):
        return [c for c in self.model.components if isinstance(c, Lines)]

    def _index(self, name):
        return self.model.params.index(name)

    def _agn_continuum(self, x, full):
        x = np.atleast_1d(np.asarray(x, float))
        total = np.zeros_like(x)
        for c, s in zip(self.model.components, self.model._slices()):
            if c.kind == "continuum":
                total += c.evaluate(x, full[s], self.spectrum.context)[0]
        return total

    @property
    def lines(self):
        """Every fitted line with flux, FWHM, shift and equivalent width."""
        sig = self._sigma()
        rows = []
        for comp in self._line_components():
            sl = dict(zip([c.name for c in self.model.components], self.model._slices()))[comp.name]
            for k, ln in enumerate(comp.line_names):
                kin = "" if f"{comp.name}.fwhm" in self.model.params else f"{ln}."
                wi = self._index(f"{comp.name}.{kin}fwhm")
                si = self._index(f"{comp.name}.{kin}shift")
                lam = comp._wave[k]

                def flux(full, k=k, comp=comp, sl=sl):
                    return comp.line_fluxes(full[sl])[k]

                def ew(full, flux=flux, si=si, lam=lam):
                    centre = lam * (1 + full[si] / C_KMS)
                    cont = self._agn_continuum(centre, full)[0]
                    return flux(full) / cont if cont > 0 else np.nan

                name = f"{comp.name}.{ln}.flux"
                if name in self.model.params:     # a flux parameter: its own error/status
                    fi = self._index(name)
                    f_val, f_err = self.full[fi], self._param_error(fi, sig)
                else:                             # mixed (max_ratios): propagate
                    f_val, f_err = (float(v) for v in self.propagate(flux))
                ew_val, ew_err = self.propagate(ew)
                errs = [self._param_error(i, sig) for i in (wi, si)]
                rows.append((comp.name, ln, lam, f_val, f_err, self.full[wi],
                             errs[0], self.full[si], errs[1], float(ew_val), float(ew_err)))
        return pd.DataFrame(rows, columns=[
            "component", "line", "wavelength", "flux", "flux_err", "fwhm", "fwhm_err",
            "shift", "shift_err", "ew", "ew_err"])

    def _param_error(self, i, sig):
        if self.samples is not None:
            lo, hi = np.percentile(self.samples[:, i], [15.865, 84.135])
            return 0.5 * (hi - lo)
        name = self.model.params.names[i]
        return np.nan if name in self.at_limit or name in self.at_zero else sig[i]

    # ---------------------------------------------------------- measuring
    def _line_profile_func(self, line, components):
        comps = self._line_components()
        if components is None:
            components = [c.name for c in comps if line in c.line_names]
        if not components:
            raise ValueError(f"no Lines component contains {line!r}")
        sel = [c for c in comps if c.name in components]
        lam0 = sel[0]._wave[sel[0].line_names.index(line)]
        v = np.linspace(-40000, 40000, 16001)
        lam = lam0 * (1 + v / C_KMS)
        ctx = {"medium": "vacuum", "resolution": 0.0}
        slices = dict(zip([c.name for c in self.model.components], self.model._slices()))

        def profile(full):
            total = np.zeros_like(lam)
            for c in sel:
                total += c.line_profile(lam, full[slices[c.name]], ctx, line)
            return total

        return lam0, v, lam, profile, components

    def _profile_stats(self, prof, lam0, v, lam, full):
        """flux, FWHM, sigma, centroid shift and EW of a profile on velocity grid v."""
        flux = prof.sum() * (lam[1] - lam[0])
        peak = np.argmax(prof)
        half = prof[peak] / 2
        above = np.nonzero(prof >= half)[0]
        i0, i1 = above[0], above[-1]
        # Interpolate the outermost half-maximum crossings.
        vl = np.interp(half, [prof[i0 - 1], prof[i0]], [v[i0 - 1], v[i0]]) if i0 > 0 else v[0]
        vr = (np.interp(half, [prof[i1 + 1], prof[i1]], [v[i1 + 1], v[i1]])
              if i1 < v.size - 1 else v[-1])
        total = prof.sum()
        w = prof / total if total > 0 else prof
        centroid = float(np.sum(w * v))
        sigma = float(np.sqrt(np.sum(w * (v - centroid) ** 2)))
        # Widths are velocities relative to the line centroid (as for the
        # fitted FWHM parameters), not to the rest wavelength.
        stretch = 1 + centroid / C_KMS
        cont = self._agn_continuum(lam0 * stretch, full)[0]
        ew = flux / cont if cont > 0 else np.nan
        return np.array([flux, (vr - vl) / stretch, sigma / stretch, centroid, ew])

    def measure(self, line, components=None):
        """
        Measure a line summed over components (default: every Lines component
        that contains it, e.g. broad + very broad), directly from the model
        profile. Returns a dict with flux, FWHM and line dispersion sigma
        (km/s), centroid shift (km/s), EW (A) and their errors.
        """
        lam0, v, lam, profile, comps = self._line_profile_func(line, components)

        def stats(full):
            return self._profile_stats(profile(full), lam0, v, lam, full)

        val, err = self.propagate(stats)
        keys = ["flux", "fwhm", "sigma", "shift", "ew"]
        out = {"line": line, "components": comps}
        for k, a, e in zip(keys, val, err):
            out[k], out[k + "_err"] = float(a), float(e)
        return out

    def balmer_decrement(self):
        """
        Broad H-alpha / H-beta flux ratio (all broad components of each line)
        with its error, or None if the model lacks either line. Unreddened
        broad-line AGN have about 3.06 with 0.03 dex scatter (Dong et al. 2008;
        2.7-2.9 according to Gaskell 2017); values far outside 2.5-4 point to
        a decomposition problem (continuum vs. broad wings) or to reddening.
        """
        comps = {ln: [c.name for c in self._line_components()
                      if c.role == "broad" and ln in c.line_names] for ln in ("Ha", "Hb")}
        if not comps["Ha"] or not comps["Hb"]:
            return None
        fa = self._line_profile_func("Ha", comps["Ha"])
        fb = self._line_profile_func("Hb", comps["Hb"])

        def ratio(full):
            pa, pb = fa[3](full).sum(), fb[3](full).sum()
            dla, dlb = fa[2][1] - fa[2][0], fb[2][1] - fb[2][0]
            return pa * dla / (pb * dlb) if pb > 0 else np.nan

        val, err = self.propagate(ratio)
        return float(val), float(err)

    def continuum(self, at=5100.0):
        """AGN continuum flux density at a rest wavelength, with error."""
        val, err = self.propagate(lambda full: self._agn_continuum(at, full)[0])
        return float(val), float(err)

    def _lum_factor(self):
        s = self.spectrum
        if s.z is None or not s.flux_unit:
            raise ValueError("luminosity needs spectrum.z and spectrum.flux_unit")
        from astropy.cosmology import Planck18
        d_l = Planck18.luminosity_distance(s.z).to("cm").value
        return 4 * np.pi * d_l ** 2 * s.flux_unit

    def luminosity(self, at=5100.0):
        """log10 of lambda L_lambda (erg/s) of the AGN continuum, and its error (dex)."""
        f = self._lum_factor()
        val, err = self.propagate(lambda full: np.log10(at * f * self._agn_continuum(at, full)[0]))
        return float(val), float(err)

    def bh_mass(self, line="Hb", components=None):
        """
        Single-epoch virial black-hole mass log10(M/Msun) from the FWHM of the
        (summed) broad line and the continuum luminosity. Recipes: Hb
        (Vestergaard & Peterson 2006, L5100), MgII2798 (Vestergaard & Osmer
        2009, L3000), CIV1549 (Vestergaard & Peterson 2006, L1350). The error
        is statistical only; the recipes have ~0.3-0.4 dex intrinsic scatter.
        """
        rec = BH_RECIPES[{"MgII": "MgII2798", "CIV": "CIV1549"}.get(line, line)]
        if components is None:
            components = [c.name for c in self._line_components()
                          if line in c.line_names and c.role == "broad"]
        lam0, v, lam, profile, comps = self._line_profile_func(line, components)
        f = self._lum_factor()
        at = rec["lum"]

        def logm(full):
            fwhm = self._profile_stats(profile(full), lam0, v, lam, full)[1]
            lum = at * f * self._agn_continuum(at, full)[0]
            return rec["a"] + 2 * np.log10(fwhm / 1000) + rec["b"] * np.log10(lum / 1e44)

        val, err = self.propagate(logm)
        return {"log_mbh": float(val), "log_mbh_err": float(err), "line": line,
                "components": comps, "recipe": rec["ref"], "systematic_dex": 0.4}

    # ------------------------------------------------------------ outputs
    def model_flux(self, wave=None, components=False):
        wave = self.spectrum.wave if wave is None else wave
        return self.model.evaluate(wave, self.spectrum.context, self.full, components)

    def to_dataframe(self):
        s = self.spectrum
        df = pd.DataFrame({"wave": s.wave, "flux": s.flux, "err": s.err, "used": self.mask,
                           "model": self.model_flux()})
        for name, comp in self.model_flux(components=True).items():
            df[name] = comp
        return df

    def host_properties(self, at=5100.0):
        """
        Stellar-population host: sigma* and v* (km/s), host fraction of the
        total model at wavelength ``at`` and light-weighted age (log Gyr) and
        metallicity [M/H] at the normalisation, with uncertainties.
        """
        from .host import StellarPopulation
        hosts = [c for c in self.model.components if isinstance(c, StellarPopulation)]
        if not hosts:
            raise ValueError("the model has no StellarPopulation component")
        h = hosts[0]
        x = self.spectrum.wave[self.mask]
        h._setup(x, self.spectrum.context)       # the fitted grid: its resolution floor
        idx = {n: self._index(f"{h.name}.{n}") for n in ("velocity", "sigma")}
        sig = self._sigma()
        out = {"sigma_star": (self.full[idx["sigma"]], self._param_error(idx["sigma"], sig)),
               "v_star": (self.full[idx["velocity"]], self._param_error(idx["velocity"], sig)),
               "sigma_floor": h.sigma_floor, "templates": h.template_set,
               "cite": h.citation}
        w_idx = [self._index(f"{h.name}.{k}") for k in h.grid.index]
        log_age = np.log10(h.grid.age_gyr.to_numpy())
        metal = h.grid.metal.to_numpy()

        def weighted(values):
            def f(full):
                w = full[w_idx]
                return float(np.sum(w * values) / np.sum(w)) if np.sum(w) > 0 else np.nan
            return f

        out["log_age"] = tuple(map(float, self.propagate(weighted(log_age))))
        out["metal"] = tuple(map(float, self.propagate(weighted(metal))))
        if x.min() <= at <= x.max():
            def frac(full):
                parts = self.model.evaluate(x, self.spectrum.context, full, components=True)
                total = sum(parts.values())
                return float(np.interp(at, x, parts[h.name] / total))
            out[f"host_fraction_{at:g}"] = tuple(map(float, self.propagate(frac)))
        warnings = []
        kin_limit = [n for n in ("velocity", "sigma") if f"{h.name}.{n}" in self.at_limit]
        if kin_limit:
            warnings.append(f"stellar {' and '.join(kin_limit)} at the fit limit: the stellar "
                            "absorption features are not detected, so the 'host' only "
                            "describes a continuum shape; consider fitting without a host")
        elif out["sigma_star"][0] < 1.1 * h.sigma_floor:
            warnings.append(f"sigma* is close to the template resolution limit "
                            f"({h.sigma_floor:.0f} km/s) and is not reliable")
        if not self.spectrum.resolution:
            warnings.append("instrumental resolution unknown (spectrum.resolution = 0): "
                            "sigma* includes the instrumental broadening")
        if warnings:
            out["warning"] = "; ".join(warnings)
        return out

    @property
    def n_free(self):
        """Number of free parameters (excluding fixed, tied and auto-fixed ones)."""
        return sum(1 for p in self.model.params if p.free)

    @property
    def bic(self):
        """Bayesian information criterion chi2 + k ln(N); lower is better."""
        return self.chi2 + self.n_free * np.log(self.mask.sum())

    @property
    def aic(self):
        """Akaike information criterion chi2 + 2k; lower is better."""
        return self.chi2 + 2 * self.n_free

    def negative_host_fraction(self):
        """Fraction of fitted pixels where a HostGalaxy component is negative."""
        hosts = [c.name for c in self.model.components if c.kind == "host"]
        if not hosts:
            return 0.0
        parts = self.model_flux(components=True)
        host = sum(parts[n] for n in hosts)[self.mask]
        return float(np.mean(host < 0))

    def summary(self, short=False):
        msg = [f"Fit of {self.spectrum.name}: chi2 = {self.chi2:.1f}, dof = {self.dof}, "
               f"reduced chi2 = {self.rchi2:.3f}, {self.nfev} evaluations, {self.time:.2f} s"]
        if not self.success:
            msg.append(f"  WARNING: optimizer did not converge: {self.message}")
        if self.error_scale > 1:
            msg.append(f"  errors scaled by sqrt(reduced chi2) = {self.error_scale:.2f}")
        info = getattr(self, "mcmc_info", None)
        if info:
            msg.append(f"  HMC: {info['n_chains']} chains x {info['n_samples']} samples, "
                       f"acceptance {info['acceptance']:.2f}, R-hat {info['r_hat']:.3f}, "
                       f"effective samples {info['ess']:.0f}, "
                       f"{'converged' if info['converged'] else 'NOT converged'}")
        if self.at_limit:
            msg.append(f"  WARNING: at their limits (consider widening): {', '.join(self.at_limit)}")
        negative = self.negative_host_fraction()
        if negative > 0.1:
            msg.append(f"  WARNING: the host galaxy is negative over {negative:.0%} of the fitted "
                       "range, which is unphysical; the host is probably negligible, so fit "
                       "without it")
        try:
            dec = self.balmer_decrement()
        except Exception:
            dec = None
        if dec is not None and np.isfinite(dec[0]) and not 2.5 <= dec[0] <= 4.0:
            msg.append(f"  WARNING: broad Halpha/Hbeta = {dec[0]:.2f} +- {dec[1]:.2f}, far from "
                       "the typical ~3 (Dong et al. 2008): the continuum probably trades flux "
                       "with the broad-line wings (try physical=True or another continuum), "
                       "or the broad-line region is reddened")
        if self.unconstrained and not short:
            msg.append(f"  outside the fitted range, fixed at 0: {', '.join(self.unconstrained)}")
        if self.at_zero and not short:
            msg.append(f"  not detected (amplitude at its bound): {', '.join(self.at_zero)}")
        if short:
            return "\n".join(msg)
        with pd.option_context("display.width", 140, "display.max_rows", 200,
                               "display.float_format", "{:.4g}".format):
            msg += ["", "Lines:", self.lines.to_string(index=False)]
        return "\n".join(msg)

    def __repr__(self):
        return self.summary(short=True)

    def save(self, prefix):
        """Write <prefix>_params.csv, _lines.csv, _model.csv and _summary.json."""
        self.params.to_csv(f"{prefix}_params.csv", index=False)
        self.lines.to_csv(f"{prefix}_lines.csv", index=False)
        self.to_dataframe().to_csv(f"{prefix}_model.csv", index=False)
        with open(f"{prefix}_summary.json", "w") as fh:
            json.dump({"name": self.spectrum.name, "z": self.spectrum.z, "chi2": self.chi2,
                       "dof": self.dof, "reduced_chi2": self.rchi2,
                       "error_method": self.error_method, "error_scale": self.error_scale,
                       "parameters_at_limit": self.at_limit,
                       "amplitudes_at_bound": self.at_zero}, fh, indent=2)

    def plot(self, ax=None, components=True, residuals=True, xlim=None, fig=None,
             width="double", group=True, label_lines=False, title=None, style=True):
        """
        Publication-quality plot of data, model, components and residuals;
        returns the figure. See :func:`fantasy_agn.plotting.plot_result`.

        :param width: 'single' (3.5 in) or 'double' (7.1 in) journal column, or inches.
        :param group: one legend entry per kind of component (all broad
            components summed, ...); False shows every component.
        :param label_lines: annotate the strongest lines.
        :param xlim: wavelength range to show, e.g. (4700, 5100) for a zoom.
        """
        from .plotting import plot_result
        return plot_result(self, ax=ax, components=components, residuals=residuals, xlim=xlim,
                           fig=fig, width=width, group=group, label_lines=label_lines,
                           title=title, style=style)

    def savefig(self, path, **plot_options):
        """
        Plot and save in one step; the format follows the extension (.pdf,
        .eps, .svg, .png). Fonts are embedded; raster formats use 300 dpi.
        Keyword arguments go to :meth:`plot`, e.g. width="single".
        """
        import matplotlib.pyplot as plt
        from .plotting import savefig
        fig = self.plot(**plot_options)
        savefig(fig, path)
        plt.close(fig)
        return path
