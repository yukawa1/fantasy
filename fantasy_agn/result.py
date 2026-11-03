"""Fit results: parameter and line tables with uncertainties, plots, export."""
import json

import numpy as np
import pandas as pd

from .components import C_KMS, Lines

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
# UV 2200-3090 A (Tsuzuki et al. 2006).
FEII_BANDS = {"optical": (4435.2, 4685.3), "uv": (2200.0, 3090.0)}


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
        for i, p in enumerate(self.model.params):
            status = ("lower limit (inner edge not constrained)"
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
        for comp in self._feii_components():
            m = self.feii(comp.name)
            wi, si = self._index(f"{comp.name}.fwhm"), self._index(f"{comp.name}.shift")
            errs = [self._param_error(i, sig) for i in (wi, si)]
            rows.append((comp.name, m["line"], float(np.mean(m["band"])), m["flux"], m["flux_err"], self.full[wi],
                         errs[0], self.full[si], errs[1], m["ew"], m["ew_err"]))
        return pd.DataFrame(rows, columns=[
            "component", "line", "wavelength", "flux", "flux_err", "fwhm", "fwhm_err",
            "shift", "shift_err", "ew", "ew_err"])

    def _feii_components(self):
        from .components import FeII
        return [c for c in self.model.components if isinstance(c, FeII)]

    def feii(self, component=None, band=None):
        """
        Fe II emission of an FeII template component (default: the first one) in a
        band: flux, EW (against the AGN continuum) and, when the model has broad
        H-beta (optical) or broad Mg II (UV), R_FeII = F(FeII) / F(broad line),
        with errors. The default band is 4434-4684 A for the optical template
        (Boroson & Green 1992) and 2200-3090 A for the UV one (Tsuzuki et al. 2006).
        """
        comps = self._feii_components()
        if component is not None:
            comps = [c for c in comps if c.name == component]
        if not comps:
            raise ValueError("the model has no FeII component" + (f" named {component!r}" if component else ""))
        comp = comps[0]
        kind = "uv" if np.min(comp._wave) < 3500 else "optical"
        lo, hi = band if band is not None else FEII_BANDS[kind]
        x = np.linspace(lo, hi, 2001)
        dx = x[1] - x[0]
        ctx = {"medium": "vacuum", "resolution": 0.0}
        sl = dict(zip([c.name for c in self.model.components], self.model._slices()))[comp.name]
        ref = "Hb" if kind == "optical" else "MgII2798"
        broad = [c.name for c in self._line_components() if c.role == "broad" and ref in c.line_names]
        prof = self._line_profile_func(ref, broad) if broad else None

        def stats(full):
            f = comp.evaluate(x, full[sl], ctx)[0]
            cont = self._agn_continuum(x, full)
            flux = f.sum() * dx
            ew = np.sum(np.where(cont > 0, f / np.where(cont > 0, cont, 1), np.nan)) * dx
            r = np.nan
            if prof is not None:
                fb = prof[3](full).sum() * (prof[2][1] - prof[2][0])
                r = flux / fb if fb > 0 else np.nan
            return np.array([flux, ew, r])

        val, err = self.propagate(stats, depends=self._depends([comp.name] + broad))
        name = f"FeII{lo:.0f}-{hi:.0f}" if band is not None else ("FeII4434-4684" if kind == "optical" else "FeII2200-3090")
        return {"component": comp.name, "line": name, "band": (float(lo), float(hi)),
                "flux": float(val[0]), "flux_err": float(err[0]), "ew": float(val[1]), "ew_err": float(err[1]),
                "r_feii": float(val[2]), "r_feii_err": float(err[2]), "reference": ref if broad else None}

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

    def _depends(self, components):
        """Indices of the parameters of ``components`` and of the AGN continuum (used by EWs)."""
        out = []
        for c, sl in zip(self.model.components, self.model._slices()):
            if c.name in components or c.kind == "continuum":
                out.extend(range(sl.start, sl.stop))
        return out

    def _profile_stats(self, prof, lam0, v, lam, full):
        """flux, FWHM, sigma, centroid shift and EW of a profile on velocity grid v."""
        if not np.all(np.isfinite(prof)):        # e.g. a width stepped past its limit
            return np.full(5, np.nan)
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

        val, err = self.propagate(stats, depends=self._depends(comps))
        keys = ["flux", "fwhm", "sigma", "shift", "ew"]
        out = {"line": line, "components": comps}
        for k, a, e in zip(keys, val, err):
            out[k], out[k + "_err"] = float(a), float(e)
        return out

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
        line = next((ln for ln in ("Hb", "Ha", "MgII2798", "CIV1549")
                     if any(ln in c.line_names for c in broad)), None)
        if line:
            m = self.measure(line, [c.name for c in broad if line in c.line_names])
            rows += [(f"broad {line} FWHM", m["fwhm"], m["fwhm_err"], "km/s"),
                     (f"broad {line} flux", m["flux"], m["flux_err"], "")]
            for ln in [line] + (["Ha"] if line == "Hb" else []):
                if not any(ln in c.line_names for c in broad):
                    continue
                try:
                    bh = self.bh_mass(ln)
                    rows.append(("log M_BH" if ln == line else f"log M_BH ({ln})", bh["log_mbh"], bh["log_mbh_err"], "dex"))
                except Exception:
                    pass
            dec = self.balmer_decrement()
            if dec:
                rows.append(("broad Ha/Hb", dec[0], dec[1], ""))
            if self._feii_components() and line in ("Hb", "MgII2798"):
                try:
                    fe = self.feii()
                    if np.isfinite(fe["r_feii"]):
                        rows.append(("R_FeII", fe["r_feii"], fe["r_feii_err"], ""))
                except Exception:
                    pass
        else:
            t = self.lines[self.lines.component.isin(["narrow", "narrow_hi", "outflow"])]
            if "[OIII]5007" in set(t.line):
                from .diagnostics import _flux_function
                idx = _flux_function(self, outflow=True).get("[OIII]5007", [])
                f, e = self.propagate(lambda full: float(np.sum(full[idx])))
                core = t[t.component.isin(["narrow", "narrow_hi"]) & (t.line == "[OIII]5007")]
                n = core.iloc[0]
                rows += [("[OIII]5007 flux", float(f), float(e), ""),
                         ("[OIII]5007 FWHM (narrow)", n.fwhm, n.fwhm_err, "km/s")]
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
        fa = self._line_profile_func("Ha", comps["Ha"])
        fb = self._line_profile_func("Hb", comps["Hb"])

        def ratio(full):
            pa, pb = fa[3](full).sum(), fb[3](full).sum()
            dla, dlb = fa[2][1] - fa[2][0], fb[2][1] - fb[2][0]
            return pa * dla / (pb * dlb) if pb > 0 else np.nan

        val, err = self.propagate(ratio, depends=self._depends(comps["Ha"] + comps["Hb"]))
        return float(val), float(err)

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
        scatter of the recipe (0.3-0.6 dex).
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
        lam0, v, lam, profile, comps = self._line_profile_func(line, components)
        f = self._lum_factor()
        at = rec["lum"]

        def logm(full):
            stats = self._profile_stats(profile(full), lam0, v, lam, full)
            lum = f * stats[0] if at == "line" else at * f * self._agn_continuum(at, full)[0]
            width = stats[2] if rec.get("width") == "sigma" else stats[1]
            return rec["a"] + rec["c"] * np.log10(width / 1000) + rec["b"] * (np.log10(lum) - rec["lnorm"])

        val, err = self.propagate(logm, depends=self._depends(comps))
        return {"log_mbh": float(val), "log_mbh_err": float(err), "line": line,
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
        if broad and self._feii_components():
            fe = self.feii()
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
        if not self.spectrum.meta.get("ebv") and "ebv" not in self.spectrum.meta:
            msg.append("  note: Galactic extinction not corrected; fluxes and luminosities are "
                       "too low (use spec.deredden() before fitting)")
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
