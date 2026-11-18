"""Fit results: parameter and line tables with uncertainties, plots, export."""
import json

import numpy as np
import pandas as pd

from .components import C_KMS, Lines, instrumental_fwhm
from .wavelengths import _sampling_gaps, air_to_vac

__all__ = ("FitResult",)

# Single-epoch virial black-hole mass recipes:
# log M = a + c log(W / 1000 km/s) + b log(L / 10^lnorm erg/s), with W the FWHM (or the line dispersion sigma for
# width="sigma") of the broad line and L the continuum luminosity lambda L_lambda at ``lum`` A or, for lum="line",
# the luminosity of the broad line itself. The first recipe of each line is the default.
# Dalla Bonta et al. (2020), eqs. (38) and (40), with log f = 0.683 (Batiste et al. 2017) included in a; their
# L5100 forms use log L(Hb) = 41.746 + [log L_AGN(5100) - 43.396] / 1.003 (their eq. 24).
BH_RECIPES = {
    "Hb": {
        "vp06": dict(a=6.91, b=0.50, c=2.0, lum=5100, lnorm=44, ref="Vestergaard & Peterson 2006", scatter=0.43),
        "db20": dict(a=7.0045, b=0.784, c=1.387, lum="line", lnorm=42, ref="Dalla Bonta et al. 2020 (FWHM, L(Hb))",
                     scatter=0.371),
        "db20_sigma": dict(a=7.1215, b=0.703, c=2.183, lum="line", lnorm=42, width="sigma",
                           ref="Dalla Bonta et al. 2020 (sigma, L(Hb))", scatter=0.309),
        "db20_l5100": dict(a=7.2775, b=0.7817, c=1.387, lum=5100, lnorm=44,
                           ref="Dalla Bonta et al. 2020 (FWHM, L5100)", scatter=0.371),
        "db20_sigma_l5100": dict(a=7.3663, b=0.7009, c=2.183, lum=5100, lnorm=44, width="sigma",
                                 ref="Dalla Bonta et al. 2020 (sigma, L5100)", scatter=0.309),
        "shen24": dict(a=6.85, b=0.50, c=2.0, lum=5100, lnorm=44, ref="Shen et al. 2024 (SDSS-RM)", scatter=0.45),
    },
    "Ha": {
        # M = 2.0e6 (L_Ha / 1e42)^0.55 (FWHM_Ha / 1000)^2.06 Msun (Greene & Ho 2005, eq. 6)
        "gh05": dict(a=6.30, b=0.55, c=2.06, lum="line", lnorm=42, ref="Greene & Ho 2005", scatter=0.3),
        # the same with the Bentz et al. (2013) R-L relation and <f> = 4.3 (Reines & Volonteri 2015, eq. 5)
        "rv15": dict(a=6.57, b=0.47, c=2.06, lum="line", lnorm=42, ref="Reines & Volonteri 2015", scatter=0.3),
    },
    "MgII2798": {
        "vo09": dict(a=6.86, b=0.50, c=2.0, lum=3000, lnorm=44, ref="Vestergaard & Osmer 2009", scatter=0.55),
        "shen11": dict(a=6.74, b=0.62, c=2.0, lum=3000, lnorm=44, ref="Shen et al. 2011", scatter=0.4),
        "shen24": dict(a=6.95, b=0.60, c=3.0, lum=3000, lnorm=45, ref="Shen et al. 2024 (SDSS-RM)", scatter=0.47),
    },
    "CIV1549": {
        "vp06": dict(a=6.66, b=0.53, c=2.0, lum=1350, lnorm=44, ref="Vestergaard & Peterson 2006", scatter=0.36),
        "shen24": dict(a=7.40, b=0.50, c=2.0, lum=1350, lnorm=45, ref="Shen et al. 2024 (SDSS-RM)", scatter=0.58),
    },
}
# Fe II bands (rest, vacuum): optical 4434-4684 A air (Boroson & Green 1992; R_FeII = F(FeII)/F(Hb broad)),
# Local Mg II preset window and the full UV comparison band (Tsuzuki et al. 2006).
FEII_BANDS = {"optical": (4435.2, 4685.3), "uv": (2650.0, 3050.0), "uv_extended": (2200.0, 3090.0)}
FEII_TEMPLATE_RANGES = {"optical": (3700.0, 10900.0), "uv": (2650.0, 3050.0), "uv_extended": (2200.0, 3500.0)}


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

    not_measured = ()                     # results saved before this was recorded

    def __init__(self, spectrum, model, mask, full, cov, samples, chi2, dof, nfev,
                 success, message, at_limit, error_scale, time, error_method, at_zero=(),
                 unconstrained=(), not_measured=()):
        self.spectrum, self.model, self.mask = spectrum, model, mask
        self.full, self.cov, self.samples = full, cov, samples
        self.chi2, self.dof, self.rchi2 = chi2, dof, chi2 / dof
        self.nfev, self.success, self.message = nfev, success, message
        self.at_limit, self.error_scale, self.time = at_limit, error_scale, time
        # amplitudes that went to a bound (usually 0: component not detected)
        self.at_zero = list(at_zero)
        # amplitudes fixed at 0 because they do not affect the fitted pixels
        self.unconstrained = list(unconstrained)
        # widths and shifts of components without flux: their values mean nothing
        self.not_measured = list(not_measured)
        self.error_method = error_method

    # ------------------------------------------------------- propagation
    def _sigma(self):
        if self.cov is None:
            return np.full(self.full.size, np.nan)
        return np.sqrt(np.clip(np.diag(self.cov), 0, None))

    def propagate(self, func, covariance=False, depends=None):
        """
        Value and 1-sigma error of any function of the full parameter vector,
        from Monte Carlo samples if available, else by linear propagation of the
        covariance (finite-difference gradient on the scale of each error).
        Returns (value, error) with arrays if func returns an array; with
        ``covariance=True`` (value, covariance matrix) for a 1-d func.

        :param depends: indices of the parameters ``func`` depends on (default: all);
            the gradient is only computed for these (much faster for one line of a large model).
        """
        value = np.asarray(func(self.full), float)
        if self.samples is not None:
            vals = np.array([func(s) for s in self.samples], float)
            if covariance:
                ok = np.all(np.isfinite(vals.reshape(len(vals), -1)), axis=1)
                return value, np.atleast_2d(np.cov(vals[ok].reshape(ok.sum(), -1), rowvar=False))
            lo, hi = np.nanpercentile(vals, [15.865, 84.135], axis=0)
            return value, 0.5 * (hi - lo)
        if self.cov is None:
            return value, (np.full((value.size, value.size), np.nan) if covariance
                           else np.full_like(value, np.nan))
        sig = self._sigma()
        idx = np.nonzero(sig > 0)[0]
        if depends is not None:
            idx = np.intersect1d(idx, np.asarray(depends, int))
        grad = np.zeros(value.shape + (self.full.size,))
        lo = np.array([p.min for p in self.model.params], float)
        hi = np.array([p.max for p in self.model.params], float)
        for i in idx:
            # steps stay within the parameter limits (a width must not become negative)
            h = 0.1 * sig[i]
            a, b = self.full.copy(), self.full.copy()
            a[i] = min(a[i] + h, hi[i])
            b[i] = max(b[i] - h, lo[i])
            if a[i] > b[i]:
                grad[..., i] = (np.asarray(func(a)) - np.asarray(func(b))) / (a[i] - b[i])
        g = grad[..., idx]
        C = self.cov[np.ix_(idx, idx)]
        if covariance:
            g = g.reshape(-1, idx.size)
            return value, g @ C @ g.T
        var = np.einsum("...i,ij,...j->...", g, C, g)
        return value, np.sqrt(np.clip(var, 0, None))

    # -------------------------------------------------------------- tables
    @property
    def params(self):
        sig = self._sigma()
        rows = []
        weak = self.unmeasured_fluxes
        for i, p in enumerate(self.model.params):
            status = ("not measured" if p.name in self.not_measured or p.name in weak else
                      "lower limit (inner edge not constrained)"
                      if p.name in self.at_limit and p.name.endswith(".span") and p.value >= p.max - 1e-6 else
                      "at limit" if p.name in self.at_limit else
                      "at bound" if p.name in self.at_zero else
                      "outside range" if p.name in self.unconstrained else
                      "free" if p.free else "tied" if p.tie else "fixed")
            err = sig[i] if p.free or p.tie else 0.0
            if self.samples is not None and (p.free or p.tie):
                lo, hi = np.percentile(self.samples[:, i], [15.865, 84.135])
                err = 0.5 * (hi - lo)
            if (status in ("at limit", "at bound") or status.startswith("lower limit")) and self.samples is None:
                err = np.nan
            if p.name in self.not_measured:
                err = np.nan
            rows.append((p.name, p.value, err, p.min, p.max, p.unit, status))
        return pd.DataFrame(rows, columns=["parameter", "value", "error", "min", "max",
                                           "unit", "status"])

    @property
    def unmeasured_fluxes(self):
        """Line fluxes with an error at least as large as the flux: the data do not measure them."""
        sig = self._sigma()
        return [p.name for i, p in enumerate(self.model.params)
                if p.name.endswith(".flux") and (p.free or p.tie) and p.value > 0
                and self._param_error(i, sig) >= p.value]

    def upper_limit(self, name, level=0.95):
        """
        One-sided limit on a parameter from the posterior samples (``errors='mcmc'``),
        e.g. the flux of an undetected line. It assumes the flat priors within the
        parameter limits, so it depends on those limits.
        """
        if self.error_method != "mcmc":
            raise ValueError("upper_limit needs posterior samples: fit with errors='mcmc'")
        return float(np.percentile(self.samples[:, self._index(name)], 100 * level))

    def line_limit(self, line, component=None, nsigma=3.0, level=0.95):
        """
        Upper limit on the flux of a line that is not detected.

        With posterior samples (``errors='mcmc'``) it is the one-sided ``level`` limit of the
        samples (:meth:`upper_limit`). Otherwise it is ``max(flux, 0) + nsigma * sigma_F``, where
        sigma_F is the error of the flux of a line with the fitted width and shift, measured from
        the noise under it alone (the other parameters held at their values; errors scaled as in
        the fit): the flux a line of that shape would need to be seen at ``nsigma``.

        :param line: line name, e.g. '[OIII]5007'.
        :param component: the component (default: the first that has the line).
        :return: dict with ``limit``, ``method`` ('mcmc' or 'noise'), ``nsigma`` or ``level``,
            ``sigma`` (sigma_F, for the noise method) and the fitted ``flux``.
        """
        comps = [c for c in self._line_components() if line in c.line_names
                 and (component is None or c.name == component)]
        if not comps:
            raise ValueError(f"no line {line!r}" + (f" in {component!r}" if component else ""))
        comp = comps[0]
        sl = dict(zip([c.name for c in self.model.components], self.model._slices()))[comp.name]
        k = comp.line_names.index(line)
        name = f"{comp.name}.{line}.flux"
        flux = float(comp.line_fluxes(self.full[sl])[k])
        if self.error_method == "mcmc" and self.samples is not None and name in self.model.params:
            return {"limit": self.upper_limit(name, level), "method": "mcmc", "level": level, "flux": flux,
                    "sigma": np.nan}
        if name not in self.model.params:
            raise ValueError(f"{name} is not a parameter of the model")
        s, m = self.spectrum, self.mask
        x, err = s.wave[m], s.err[m] * self.error_scale
        i = self._index(name)
        step = abs(self.full[i]) or 1.0           # the model is linear in a line flux: any step is exact
        hi = self.full.copy()
        hi[i] += step
        dflux = float(comp.line_fluxes(hi[sl])[k]) - flux
        if dflux <= 0:
            raise ValueError(f"the flux of {line} does not depend on {name}")
        g = (self.model.evaluate(x, s.context, hi) - self.model.evaluate(x, s.context, self.full)) / dflux
        info = float(np.sum((g / err) ** 2))
        sigma = 1.0 / np.sqrt(info) if info > 0 else np.inf
        return {"limit": float(max(flux, 0.0) + nsigma * sigma), "method": "noise", "nsigma": float(nsigma), "flux": flux,
                "sigma": float(sigma)}

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
            local = self._line_parameters(comp, sl)
            for k, ln in enumerate(comp.line_names):
                kin = "" if f"{comp.name}.fwhm" in self.model.params else f"{ln}."
                wi = self._index(f"{comp.name}.{kin}fwhm")
                si = self._index(f"{comp.name}.{kin}shift")
                lam = comp._wave[k]

                def flux(full, k=k, comp=comp, local=local):
                    return comp.line_fluxes(local(full))[k]

                def ew(full, flux=flux, si=si, lam=lam):
                    centre = lam * (1 + full[si] / C_KMS)
                    cont = self._agn_continuum(centre, full)[0]
                    return flux(full) / cont if cont > 0 else np.nan

                name = f"{comp.name}.{ln}.flux"
                if name in self.model.params:     # a flux parameter: its own error/status
                    fi = self._index(name)
                    f_val, f_err = flux(self.full), self._param_error(fi, sig)
                else:                             # mixed (max_ratios): propagate
                    f_val, f_err = (float(v) for v in self.propagate(flux))
                ew_val, ew_err = self.propagate(ew)
                covered = self._line_covered(comp, k, sl)
                if not covered:
                    f_val = f_err = ew_val = ew_err = np.nan
                kin = ([self._measured(wi), self._param_error(wi, sig), self._measured(si), self._param_error(si, sig)]
                       if covered and f_val != 0 else [np.nan] * 4)
                rows.append((comp.name, ln, lam, f_val, f_err, *kin, float(ew_val), float(ew_err)))
        for comp in self._feii_components():
            m = self.feii(comp.name)
            wi, si = self._index(f"{comp.name}.fwhm"), self._index(f"{comp.name}.shift")
            errs = [self._param_error(i, sig) for i in (wi, si)]
            rows.append((comp.name, m["line"], float(np.mean(m["band"])), m["flux"], m["flux_err"], self._measured(wi),
                         errs[0], self._measured(si), errs[1], m["ew"], m["ew_err"]))
        return pd.DataFrame(rows, columns=[
            "component", "line", "wavelength", "flux", "flux_err", "fwhm", "fwhm_err",
            "shift", "shift_err", "ew", "ew_err"])

    def _feii_components(self):
        from .components import FeII
        return [c for c in self.model.components if isinstance(c, FeII)]

    def _band_coverage(self, lo, hi):
        """Fraction of a vacuum band occupied by fitted pixels, excluding gaps and masks."""
        w = self.spectrum.wave
        if self.spectrum.medium == "air":
            w = air_to_vac(w)
        if len(w) < 2:
            return 0.0
        d = np.diff(w)
        left, right = np.r_[d[0], d] / 2, np.r_[d, d[-1]] / 2
        if d.size > 1:
            # Extrapolate each segment's terminal pixel using its neighbouring spacing.
            # For an isolated endpoint, the next sampled segment supplies that spacing.
            gaps, before, after = _sampling_gaps(d)
            right[gaps] = before / 2
            left[gaps + 1] = after / 2
            left[0], right[-1] = right[0], left[-1]
        overlap = np.minimum(hi, w + right) - np.maximum(lo, w - left)
        return float(np.clip(np.maximum(overlap[self.mask], 0).sum() / (hi - lo), 0, 1))

    def _feii_band(self, comp, band=None):
        kind = getattr(comp, "feii_kind", None)
        if kind is None:  # fitted components saved before feii_kind was recorded
            kind = "uv_extended" if min(comp._wave) < 2500 else "uv" if min(comp._wave) < 3500 else "optical"
        default = FEII_BANDS[kind]
        lo, hi = default if band is None else map(float, band)
        if not np.isfinite([lo, hi]).all() or not 0 < lo < hi:
            raise ValueError("an Fe II band needs two increasing positive wavelengths")
        optical = kind == "optical" and (lo, hi) == default
        name = "FeII4434-4684" if optical else f"FeII{lo:g}-{hi:g}"
        ref = "Hb" if kind == "optical" else "MgII2798"
        start, end = FEII_TEMPLATE_RANGES[kind]
        supported = start <= lo and hi <= end
        coverage = self._band_coverage(lo, hi)
        return {"line": name, "band": (lo, hi), "reference": ref, "ratio_label": "R_FeII" if optical else f"{name}/{ref}",
                "coverage": coverage, "covered": coverage >= 0.95, "template_supported": supported}

    def feii(self, component=None, band=None):
        """
        Fe II band flux, equivalent width and flux ratio to the total broad reference line.
        The optical default is 4434–4684 Å (air) divided by H-beta. The local UV template
        uses 2650–3050 Å and the extended UV template uses 2200–3090 Å, both divided by Mg II.
        Supply a rest-vacuum wavelength pair as ``band`` for a custom integral.

        ``coverage`` is the fraction of the band covered by fitted pixels. Flux, EW and
        ratio are NaN below 95% coverage or outside the template's supported interval
        (3700–10900 Å optical, 2650–3050 Å local UV, 2200–3500 Å extended UV).
        A missing or undetected broad reference
        leaves the ratio undefined. ``ratio_label`` identifies the quantity in exports.
        """
        comps = self._feii_components()
        if component is not None:
            comps = [c for c in comps if c.name == component]
        if not comps:
            raise ValueError("the model has no FeII component" + (f" named {component!r}" if component else ""))
        comp = comps[0]
        meta = self._feii_band(comp, band)
        lo, hi = meta["band"]
        ref = meta["reference"]
        broad = [c.name for c in self._line_components() if c.role == "broad" and ref in c.line_names]
        ref_flux = self._line_flux_func(ref, broad) if broad else None
        if meta["covered"] and meta["template_supported"]:
            from scipy.integrate import trapezoid
            x = np.linspace(lo, hi, 2001)
            ctx = {"medium": "vacuum", "resolution": 0.0}
            sl = dict(zip([c.name for c in self.model.components], self.model._slices()))[comp.name]

            def stats(full):
                f = comp.value(x, full[sl], ctx)
                cont = self._agn_continuum(x, full)
                flux = trapezoid(f, x)
                ew = trapezoid(f / cont, x) if (cont > 0).all() else np.nan
                fb = ref_flux(full) if ref_flux else np.nan
                return np.array([flux, ew, flux / fb if fb > 0 else np.nan])

            val, err = self.propagate(stats, depends=self._depends([comp.name] + broad))
        else:
            val = err = np.full(3, np.nan)
        return {**meta, "component": comp.name, "flux": float(val[0]), "flux_err": float(err[0]),
                "ew": float(val[1]), "ew_err": float(err[1]), "r_feii": float(val[2]),
                "r_feii_err": float(err[2]) if np.isfinite(val[2]) else np.nan}

    def _measured(self, i):
        return np.nan if self.model.params.names[i] in self.not_measured else self.full[i]

    def _param_error(self, i, sig):
        if self.model.params.names[i] in self.not_measured:
            return np.nan
        if self.samples is not None:
            lo, hi = np.percentile(self.samples[:, i], [15.865, 84.135])
            return 0.5 * (hi - lo)
        name = self.model.params.names[i]
        return np.nan if name in self.at_limit or name in self.at_zero else sig[i]

    # ---------------------------------------------------------- measuring
    def _line_covered(self, comp, k, sl):
        """At least one fitted pixel samples the shifted line within one observed FWHM."""
        local = self.full[sl]
        if self.unconstrained:
            # An excess and a ratio reference may each supply part of the flux.
            # A dead input at zero leaves the other, measured inputs usable.
            measured_input = False
            for j in np.flatnonzero(comp._weights(local)[k]):
                p = comp.params[j]
                while p.tie is not None and p.name not in self.unconstrained:
                    p = self.model.params[p.tie[0]]
                if p.name not in self.unconstrained:
                    measured_input = True
                elif local[j] != 0:
                    return False
            if not measured_input:
                return False
        _, kin = comp._split(local)
        group = comp._kin_index[k]
        centre = comp._wave[k] * (1 + kin[0][group] / C_KMS)
        width = kin[1][group]
        if comp.profile == "voigt":
            width = 0.5346 * kin[2][group] + np.hypot(width, np.sqrt(0.2166) * kin[2][group])
        inst = instrumental_fwhm(comp._wave[k:k+1], self.spectrum.context)[0]
        span = centre * np.hypot(width, inst) / C_KMS
        return span > 0 and self._band_coverage(centre - span, centre + span) > 0

    def _line_parameters(self, comp, sl):
        """Evaluate a line with zero-bound amplitudes snapped, including affine ties.

        Joint fits can leave round-off above the active bound. Keep the position of
        the fit untouched, and preserve amplitudes changed in MC draws or propagation.
        """
        from .fitting import _at_limit
        indices, offsets = [], []
        flagged = set(self.at_zero)
        if flagged:
            for j, p in enumerate(comp.params[:comp._n_amp]):
                root, factor, offset = p, 1., 0.
                while root.tie is not None:
                    target, scale, shift = root.tie
                    offset += factor * shift
                    factor *= scale
                    root = self.model.params[target]
                if root.name not in flagged:
                    continue
                value = self.full[self._index(root.name)]
                lower, upper = root.min, root.max
                bound = lower if abs(value - lower) < abs(upper - value) else upper
                if bound == 0 and _at_limit(value, lower, upper):
                    indices.append(j)
                    offsets.append(offset)
        if not indices:
            return lambda full: full[sl]
        indices, offsets = np.asarray(indices), np.asarray(offsets)
        fitted = self.full[sl][indices]

        def parameters(full):
            local = full[sl].copy()
            unchanged = local[indices] == fitted
            local[indices[unchanged]] = offsets[unchanged]
            return local
        return parameters

    def _line_flux_func(self, line, components):
        """Physical integrated amplitudes of a line, including all ratio and excess parameterizations."""
        covered = []
        for c, sl in zip(self.model.components, self.model._slices()):
            if not isinstance(c, Lines) or line not in c.line_names or (components is not None and c.name not in components):
                continue
            k = c.line_names.index(line)
            params = self._line_parameters(c, sl)
            if self._line_covered(c, k, sl):
                covered.append((c, params, k))
            elif c.line_fluxes(params(self.full))[k] != 0:
                # A positive contribution extrapolated from another line cannot be reported as measured.
                return lambda full: np.nan
        if not covered:
            return lambda full: np.nan

        def flux(full):
            return sum(c.line_fluxes(params(full))[k] for c, params, k in covered)
        return flux

    def _line_profile_func(self, line, components):
        comps = self._line_components()
        if components is None:
            components = [c.name for c in comps if line in c.line_names]
        sel = [c for c in comps if c.name in components and line in c.line_names]
        if not sel:
            raise ValueError(f"no selected Lines component contains {line!r}")
        components = [c.name for c in sel]
        lam0 = sel[0]._wave[sel[0].line_names.index(line)]
        v = np.linspace(-40000, 40000, 16001)
        lam = lam0 * (1 + v / C_KMS)
        ctx = {"medium": "vacuum", "resolution": 0.0}
        slices = dict(zip([c.name for c in self.model.components], self.model._slices()))

        params = [(c, self._line_parameters(c, slices[c.name])) for c in sel]

        def profile(full):
            total = np.zeros_like(lam)
            for c, local in params:
                total += c.line_profile(lam, local(full), ctx, line)
            return total

        return lam0, v, lam, profile, components

    def _depends(self, components):
        """Indices of the parameters of ``components`` and of the AGN continuum (used by EWs)."""
        out = []
        for c, sl in zip(self.model.components, self.model._slices()):
            if c.name in components or c.kind == "continuum":
                out.extend(range(sl.start, sl.stop))
        return out

    def _profile_stats(self, prof, lam0, v, full, flux, damped, moment_window):
        """Total flux and EW, intrinsic FWHM, windowed dispersion and centroid shift."""
        if not np.isfinite(flux):
            return np.full(5, np.nan)
        if flux == 0 or not np.isfinite(prof).all():
            cont = self._agn_continuum(lam0, full)[0]
            return np.array([flux, np.nan, np.nan, np.nan, 0.0 if flux == 0 and cont > 0 else np.nan])
        # A pure absorption line has a negative integrated flux and a positive width.
        shape = prof * np.sign(flux)
        half = shape.max() / 2
        above = np.flatnonzero(shape >= half)
        if half <= 0 or not above.size:
            return np.array([flux, np.nan, np.nan, np.nan, np.nan])
        i0, i1 = above[0], above[-1]
        width = np.nan
        if i0 > 0 and i1 < len(v) - 1:
            vl = np.interp(half, shape[i0-1:i0+1], v[i0-1:i0+1])
            vr = np.interp(half, shape[i1:i1+2][::-1], v[i1:i1+2][::-1])
            width = vr - vl

        def moments(u, y):
            from scipy.integrate import trapezoid
            area = trapezoid(y, u)
            if area <= 0:
                return np.nan, np.nan
            mean = trapezoid(u * y, u) / area
            var = trapezoid((u - mean) ** 2 * y, u) / area
            return mean, np.sqrt(var) if var >= 0 else np.nan

        # Preserve the same FWHM and EW when a different moment aperture is requested.
        centre, sigma = moments(v, shape)
        stretch = 1 + centre / C_KMS
        cont = self._agn_continuum(lam0 * stretch, full)[0]
        if moment_window is not None:
            lo, hi = moment_window
            inside = (v > lo) & (v < hi)
            u = np.r_[lo, v[inside], hi]
            centre, sigma = moments(u, np.interp(u, v, shape))
        elif damped:
            sigma = np.nan
        return np.array([flux, width / stretch, sigma / (1 + centre / C_KMS), centre,
                         flux / cont if cont > 0 else np.nan])

    def _line_measurement(self, line, components, moment_window=None):
        lam0, v, lam, profile, comps = self._line_profile_func(line, components)
        flux = self._line_flux_func(line, comps)
        damped = False
        for c, sl in zip(self.model.components, self.model._slices()):
            if c.name in comps and c.profile in ("lorentzian", "voigt"):
                k = c.line_names.index(line)
                local = self._line_parameters(c, sl)(self.full)
                _, kin = c._split(local)
                if c.line_fluxes(local)[k] != 0 and kin[-1][c._kin_index[k]] > 0:
                    damped = True

        def stats(full):
            return self._profile_stats(profile(full), lam0, v, full, flux(full), damped, moment_window)
        return stats, comps, damped, v, profile

    def measure(self, line, components=None, moment_window=None):
        """
        Measure the intrinsic summed profile of the selected Lines components.
        Flux is the sum of normalized, integrated model line amplitudes. FWHM, shift
        and dispersion are in km/s; EW is relative to the AGN continuum in Å.
        A covered zero-flux line has undefined kinematics; a line without fitted
        pixels within one observed FWHM of its centre also has undefined flux and EW.

        The centroid shift is the first moment within ``moment_window`` in km/s
        relative to the rest wavelength (default −40000 to +40000). Dispersion
        uses the same aperture. A contributing Lorentzian or Voigt damping wing
        has no finite intrinsic dispersion, so sigma is NaN unless you explicitly
        give a finite ``(vmin, vmax)`` within that range. Flux and FWHM are independent
        of that choice. ``fwhm_fragile`` marks a shoulder at half-maximum.
        """
        if moment_window is not None:
            lo, hi = map(float, moment_window)
            if not -40000 <= lo < hi <= 40000:
                raise ValueError("moment_window must have increasing limits between -40000 and 40000 km/s")
            moment_window = (lo, hi)
        stats, comps, damped, v, profile = self._line_measurement(line, components, moment_window)
        val, err = self.propagate(stats, depends=self._depends(comps))
        keys = ["flux", "fwhm", "sigma", "shift", "ew"]
        out = {"line": line, "components": comps, "moment_window": moment_window or (float(v[0]), float(v[-1])),
               "damping_wings": damped}
        for k, a, e in zip(keys, val, err):
            out[k], out[k + "_err"] = float(a), float(e) if np.isfinite(a) else np.nan
        from .fitcompare import fwhm_fragile
        out["fwhm_fragile"] = bool(np.isfinite(out["fwhm"]) and fwhm_fragile(v, profile(self.full) * np.sign(val[0]))[0])
        return out

    def _primary_broad_line(self, preferred=("Hb", "Ha", "MgII2798", "CIV1549")):
        """First covered broad emission line with positive flux and a measurable profile FWHM."""
        broad = [c for c in self._line_components() if c.role == "broad"]
        for line in preferred:
            names = [c.name for c in broad if line in c.line_names]
            if names:
                stats, _, _, _, _ = self._line_measurement(line, names)
                flux, fwhm = stats(self.full)[:2]
                if flux > 0 and np.isfinite(fwhm) and fwhm > 0:
                    return line
        return None

    def key_results(self):
        """
        The main numbers of the fit as a table (quantity, value, error, unit):
        for broad-line objects the FWHM and flux of the main broad line, the
        black-hole mass and the Balmer decrement; for narrow-line objects
        [OIII]5007 and the BPT ratios; the stellar velocity dispersion and
        host fraction when a stellar host is fitted.
        """
        rows = []
        broad = [c for c in self._line_components() if c.role == "broad"]
        line = self._primary_broad_line()
        if line:
            m = self.measure(line, [c.name for c in broad if line in c.line_names])
            rows += [(f"broad {line} FWHM", m["fwhm"], m["fwhm_err"], "km/s"),
                     (f"broad {line} flux", m["flux"], m["flux_err"], "")]
            for ln in [line] + (["Ha"] if line == "Hb" else []):
                if not any(ln in c.line_names for c in broad):
                    continue
                try:
                    bh = self.bh_mass(ln)
                    if not np.isfinite(bh["log_mbh"]):
                        continue
                    rows.append(("log M_BH" if ln == line else f"log M_BH ({ln})", bh["log_mbh"], bh["log_mbh_err"], "dex"))
                except Exception:
                    pass
            dec = self.balmer_decrement()
            if dec and np.isfinite(dec[0]):
                rows.append(("broad Ha/Hb", dec[0], dec[1], ""))
            for comp in self._feii_components():
                meta = self._feii_band(comp)
                if any(meta["reference"] in c.line_names for c in broad):
                    fe = self.feii(comp.name)
                    rows.append((fe["ratio_label"], fe["r_feii"], fe["r_feii_err"], ""))
        else:
            from .diagnostics import _flux_function
            oiii = _flux_function(self, outflow=True).get("[OIII]5007")
            if oiii is not None:
                f, e = self.propagate(oiii)
                if np.isfinite(f):
                    rows.append(("[OIII]5007 flux", float(f), float(e), ""))
                core = [c.name for c in self._line_components() if c.role == "narrow" and "[OIII]5007" in c.line_names]
                if core:
                    m = self.measure("[OIII]5007", core)
                    if np.isfinite(m["fwhm"]):
                        rows.append(("[OIII]5007 FWHM (narrow)", m["fwhm"], m["fwhm_err"], "km/s"))
            try:
                d = self.diagnostics().set_index("quantity")
            except Exception:
                d = None
            if d is not None:
                for q, lab, unit in (("log [NII]6583/Ha", "log [NII]/Ha", "dex"),
                                     ("log [OIII]5007/Hb", "log [OIII]/Hb", "dex"),
                                     ("E(B-V) gas", "E(B-V) gas", "mag"),
                                     ("n_e ([SII])", "n_e ([SII])", "cm^-3"),
                                     ("T_e ([OIII])", "T_e ([OIII])", "K"),
                                     ("12+log(O/H) O3N2", "12+log(O/H) (O3N2)", "")):
                    if q in d.index:
                        rows.append((lab, float(d.loc[q, "value"]), float(d.loc[q, "error"]), unit))
        from .host import StellarPopulation
        if any(isinstance(c, StellarPopulation) for c in self.model.components):
            hp = self.host_properties()
            rows.append(("sigma* (host)", hp["sigma_star"][0], hp["sigma_star"][1], "km/s"))
            k = next((k for k in hp if k.startswith("host_fraction")), None)
            if k:
                rows.append((f"host fraction at {k.split('_')[-1]} A", hp[k][0], hp[k][1], ""))
        return pd.DataFrame(rows, columns=["quantity", "value", "error", "unit"])

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
        fa = self._line_flux_func("Ha", comps["Ha"])
        fb = self._line_flux_func("Hb", comps["Hb"])

        def ratio(full):
            pa, pb = fa(full), fb(full)
            return pa / pb if pa > 0 and pb > 0 else np.nan

        val, err = self.propagate(ratio, depends=self._depends(comps["Ha"] + comps["Hb"]))
        return float(val), float(err) if np.isfinite(val) else np.nan

    def continuum(self, at=5100.0):
        """AGN continuum flux density at a rest wavelength, with error."""
        val, err = self.propagate(lambda full: self._agn_continuum(at, full)[0], depends=self._depends([]))
        return float(val), float(err)

    def _lum_factor(self):
        s = self.spectrum
        if s.z is None or not s.flux_unit:
            raise ValueError("luminosity needs spectrum.z and spectrum.flux_unit (the flux unit is not known: "
                             "read the spectrum with flux_unit=..., e.g. 1e-17 for values in 1e-17 erg/s/cm2/A)")
        level = s.flux_unit * np.nanmedian(np.abs(s.flux[np.isfinite(s.flux)]))
        if level > 1e-10:
            # no AGN is brighter than ~1e-12 erg/s/cm2/A in the optical: the values are not in these units
            raise ValueError(f"flux_unit = {s.flux_unit:g} gives a flux density of {level:.2g} erg/s/cm2/A, far "
                             "too bright: the flux unit is wrong (e.g. values in units of 1e-16 need flux_unit=1e-16)")
        from astropy.cosmology import Planck18
        d_l = Planck18.luminosity_distance(s.z).to("cm").value
        return 4 * np.pi * d_l ** 2 * s.flux_unit

    def luminosity(self, at=5100.0):
        """log10 of lambda L_lambda (erg/s) of the AGN continuum, and its error (dex)."""
        f = self._lum_factor()
        val, err = self.propagate(lambda full: np.log10(at * f * self._agn_continuum(at, full)[0]),
                                  depends=self._depends([]))
        return float(val), float(err)

    def bh_mass(self, line="Hb", components=None, recipe=None):
        """
        Single-epoch virial black-hole mass log10(M/Msun) from the FWHM of the
        (summed) broad line and a luminosity:

        - Hb: ``'vp06'`` (default; Vestergaard & Peterson 2006, FWHM and L5100);
          Dalla Bonta et al. 2020 (calibrated on SDSS-RM and the RM database, corrected for
          the Eddington-ratio bias) with the broad H-beta luminosity, ``'db20'`` (FWHM) or
          ``'db20_sigma'`` (line dispersion), or with L5100, ``'db20_l5100'`` and
          ``'db20_sigma_l5100'``; or ``'shen24'`` (Shen et al. 2024, SDSS-RM, L5100);
        - Ha: ``'gh05'`` (default; Greene & Ho 2005) or ``'rv15'`` (Reines & Volonteri
          2015), both from the broad H-alpha luminosity, so no continuum (or host
          subtraction) is needed;
        - MgII2798: ``'vo09'`` (default; Vestergaard & Osmer 2009), ``'shen11'``
          (Shen et al. 2011) or ``'shen24'``, all with L3000;
        - CIV1549: ``'vp06'`` (default) or ``'shen24'``, with L1350.

        The error is statistical only; ``systematic_dex`` is the intrinsic
        scatter of the recipe (0.3-0.6 dex). A zero or uncovered broad line returns NaN.
        Dispersion-based recipes also return NaN for a line with damping wings; an
        aperture-dependent dispersion is not calibrated for these recipes.
        """
        line = {"MgII": "MgII2798", "CIV": "CIV1549", "Halpha": "Ha", "Hbeta": "Hb"}.get(line, line)
        if line not in BH_RECIPES:
            raise ValueError(f"no black-hole mass recipe for {line!r}; one of {sorted(BH_RECIPES)}")
        recipes = BH_RECIPES[line]
        key = recipe or next(iter(recipes))
        if key not in recipes:
            raise ValueError(f"unknown recipe {key!r} for {line}; one of {sorted(recipes)}")
        rec = recipes[key]
        if components is None:
            components = [c.name for c in self._line_components()
                          if line in c.line_names and c.role == "broad"]
        stats, comps, damped, _, _ = self._line_measurement(line, components)
        f = self._lum_factor()
        at = rec["lum"]

        def logm(full):
            if damped and rec.get("width") == "sigma":
                return np.nan
            m = stats(full)
            lum = f * m[0] if at == "line" else at * f * self._agn_continuum(at, full)[0]
            width = m[2] if rec.get("width") == "sigma" else m[1]
            if not (m[0] > 0 and width > 0 and lum > 0):
                return np.nan
            return rec["a"] + rec["c"] * np.log10(width / 1000) + rec["b"] * (np.log10(lum) - rec["lnorm"])

        val, err = self.propagate(logm, depends=self._depends(comps))
        return {"log_mbh": float(val), "log_mbh_err": float(err) if np.isfinite(val) else np.nan, "line": line,
                "components": comps, "recipe": rec["ref"], "recipe_key": key,
                "luminosity": "broad line" if at == "line" else f"L{at}",
                "width": rec.get("width", "fwhm"), "systematic_dex": rec["scatter"]}

    def main_sequence(self):
        """
        Position on the quasar main sequence: R_FeII = F(Fe II 4434-4684) / F(broad
        H-beta) and FWHM of the (summed) broad H-beta, with errors, and the
        population (A: FWHM < 4000 km/s) and spectral type (Sulentic et al. 2002;
        Marziani et al. 2018). NaN where the model lacks Fe II or broad H-beta.
        """
        out = {"r_feii": np.nan, "r_feii_err": np.nan, "fwhm": np.nan, "fwhm_err": np.nan,
               "population": "", "spectral_type": ""}
        broad = [c.name for c in self._line_components() if c.role == "broad" and "Hb" in c.line_names]
        if broad:
            m = self.measure("Hb", broad)
            out.update(fwhm=m["fwhm"], fwhm_err=m["fwhm_err"])
        optical = next((c for c in self._feii_components() if self._feii_band(c)["ratio_label"] == "R_FeII"), None)
        if broad and optical:
            fe = self.feii(optical.name)
            out.update(r_feii=fe["r_feii"], r_feii_err=fe["r_feii_err"])
        r, w = out["r_feii"], out["fwhm"]
        if np.isfinite(w):
            out["population"] = "A" if w < 4000 else "B"
        if np.isfinite(w) and np.isfinite(r):
            if w < 4000:
                out["spectral_type"] = f"A{min(int(r / 0.5) + 1, 4)}"
            elif w < 8000:
                out["spectral_type"] = "B1" if r < 0.5 else "B2" if r < 1.0 else "B3"
            else:
                out["spectral_type"] = "B1+" if w < 12000 else "B1++"
        return out

    def bh_masses(self):
        """
        The single-epoch black-hole mass from every recipe of :data:`BH_RECIPES` that
        the model allows (a broad H-beta, H-alpha, Mg II or C IV line), as a table:
        line, recipe key, reference, width (FWHM or sigma), luminosity used,
        log M_BH, statistical error and the intrinsic scatter of the recipe (dex).
        """
        rows = []
        broad = [c for c in self._line_components() if c.role == "broad"]
        for line, recipes in BH_RECIPES.items():
            if not any(line in c.line_names for c in broad):
                continue
            for key in recipes:
                try:
                    m = self.bh_mass(line, recipe=key)
                except Exception:
                    continue
                if not np.isfinite(m["log_mbh"]):
                    continue
                rows.append((line, key, m["recipe"], m["width"], m["luminosity"], m["log_mbh"], m["log_mbh_err"],
                             m["systematic_dex"]))
        return pd.DataFrame(rows, columns=["line", "recipe", "reference", "width", "luminosity", "log_mbh",
                                           "log_mbh_err", "scatter_dex"])

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

    def undersampled(self, min_pixels=1.2):
        """
        Line components whose lines are narrower than the data can sample: the observed FWHM (the fitted
        width and the instrumental resolution in quadrature) is below ``min_pixels`` pixels. Profiles are
        averaged over the pixels, so the flux of such a line is kept, but a width below about a pixel
        barely changes the data (it is poorly measured) and the line can fit single noisy pixels.
        This happens when the instrumental resolution is unknown (``spectrum.resolution`` = 0) and a narrow
        width ends at its lower limit. Returns a list of dicts: parameter, fwhm (the observed one, km/s),
        pixel (km/s), pixels (their ratio), resolution_known.
        """
        from .components import instrumental_fwhm
        s = self.spectrum
        x = s.wave[self.mask]
        out = []
        for comp in self._line_components():
            sl = dict(zip([c.name for c in self.model.components], self.model._slices()))[comp.name]
            fluxes = comp.line_fluxes(self.full[sl])
            groups = {}
            local = self._line_parameters(comp, sl)
            for k, ln in enumerate(comp.line_names):
                kin = "" if f"{comp.name}.fwhm" in self.model.params else f"{ln}."
                name = f"{comp.name}.{kin}fwhm"
                lam = comp._wave[k]
                if fluxes[k] == 0 or not (x[0] <= lam <= x[-1]) or name in self.not_measured:
                    continue
                inst = float(instrumental_fwhm(np.array([lam]), s.context)[0])
                total = float(np.hypot(self.full[self._index(name)], inst))
                pixel = float(s.pixel_kms(lam))
                if total < min_pixels * pixel and (name not in groups or total / pixel < groups[name]["pixels"]):
                    groups[name] = {"parameter": name, "fwhm": total, "pixel": pixel, "pixels": total / pixel,
                                    "resolution_known": bool(inst > 0)}
            out.extend(groups.values())
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
        weak = self.unmeasured_fluxes
        if weak:
            msg.append(f"  WARNING: not measured (error larger than the flux): {', '.join(weak)}")
        for u in self.undersampled():
            msg.append(f"  WARNING: {u['parameter']}: the lines are {u['pixels']:.1f} pixels wide (FWHM {u['fwhm']:.0f} km/s, "
                       f"pixel {u['pixel']:.0f} km/s), too narrow to be sampled: their fluxes are unreliable"
                       + ("" if u["resolution_known"] else "; the instrumental resolution is unknown: set it with "
                          "spec.with_resolution(...) or raise the lower limit of the width"))
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
        if not self.spectrum.meta.get("ebv") and "ebv" not in self.spectrum.meta:
            msg.append("  note: Galactic extinction not corrected; fluxes and luminosities are "
                       "too low (use spec.deredden() before fitting)")
        if self.unconstrained and not short:
            msg.append(f"  outside the fitted range, fixed at 0: {', '.join(self.unconstrained)}")
        if self.at_zero and not short:
            msg.append(f"  not detected (amplitude at its bound): {', '.join(self.at_zero)}")
        if self.not_measured and not short:
            msg.append(f"  not measured (no flux in the component): {', '.join(self.not_measured)}")
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
             width="double", group="auto", label_lines=False, title=None, style=True, **options):
        """
        Publication-quality plot of data, model, components and residuals;
        returns the figure. See :func:`fantasy_agn.plotting.plot_result` for the
        options (``colors``, ``hide``, ``data``, ``model``, ``legend``, ``label_options``).

        :param width: 'single' (3.5 in) or 'double' (7.1 in) journal column, or inches.
        :param group: 'auto' (default): components summed by kind, and the 1st,
            2nd, ... broad component drawn too when a line has several; True:
            only the sums; False: every component separately.
        :param label_lines: name the strongest lines (True) or all fitted lines ('all').
        :param xlim: wavelength range to show, e.g. (4700, 5100) for a zoom.
        """
        from .plotting import plot_result
        return plot_result(self, ax=ax, components=components, residuals=residuals, xlim=xlim,
                           fig=fig, width=width, group=group, label_lines=label_lines,
                           title=title, style=style, **options)

    def diagnostics(self, outflow=True, kind=None):
        """
        Narrow-line diagnostics with errors: BPT ratios and class, narrow
        Halpha/Hbeta and gas E(B-V), [SII] electron density, [OIII] and [NII]
        temperatures (when the auroral lines are detected), EW(Halpha) for the
        WHAN diagram and, for star-forming galaxies, 12+log(O/H). See
        :func:`fantasy_agn.diagnostics.diagnostics`.
        """
        from .diagnostics import diagnostics
        return diagnostics(self, outflow=outflow, kind=kind)

    def plot_line(self, line=None, **kw):
        """
        One line alone, e.g. ``r.plot_line("Hb")``: the data with everything
        else subtracted, each component of the line (narrow, outflow, 1st and
        2nd broad, ...) and their sum, on a velocity axis, with residuals.
        Default: the main broad line. ``r.fitted_lines()`` lists the choices.
        See :func:`fantasy_agn.plotting.plot_line` for the options.
        """
        from .plotting import plot_line
        return plot_line(self, line, **kw)

    def plot_lines(self, lines=None, **kw):
        """Several lines side by side (default: those with several components)."""
        from .plotting import plot_lines
        return plot_lines(self, lines, **kw)

    def plot_profiles(self, lines=None, **kw):
        """
        Broad profiles of several lines overlaid, each scaled to its peak, e.g.
        ``r.plot_profiles(["Hb", "Ha"])``: are the lines equally wide, is one more asymmetric?
        See :func:`fantasy_agn.plotting.plot_profiles` (also for one line in several fits).
        """
        from .plotting import plot_profiles
        return plot_profiles(self, lines, **kw)

    def fitted_lines(self):
        """The lines that ``plot_line`` can show, most interesting first (a DataFrame)."""
        from .plotting import fitted_lines
        return pd.DataFrame(fitted_lines(self))

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
