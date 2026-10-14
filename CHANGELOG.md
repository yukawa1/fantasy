# Changelog

## 2.0.0

A complete rewrite. Sherpa is gone; fitting uses NumPy/SciPy with exact analytic
derivatives.

### New
- `Lines(max_ratios={line: (reference, r)})`: upper limits on flux ratios
  (kept linear: the reference line is fitted as an excess above flux/r). The
  `halpha` preset uses it for the outflow wing: [NII]6583/Hα ≤ 3 and FWHM ≤ 2000 km/s,
  so the wing can no longer absorb the broad Hα flux as [NII].
- Fits are 10–20× faster (Hβ region with FeII: ~1.5 s instead of ~15–35 s).
- Data-driven initial guess: amplitudes are solved exactly, widths and shifts are
  grid-searched within the limits, and a post-fit check escapes local minima.
- Uncertainties: covariance (default) or Monte Carlo (`errors="mc"`), errors
  inflated by sqrt(reduced χ²) when it is above 1, parameters at limits flagged,
  and errors for derived quantities (`measure`, `continuum`, `luminosity`,
  `bh_mass`).
- Ready-made models: `fa.presets.hbeta/halpha/optical/mgii/civ`.
- Joint multi-epoch fitting with shared parameters and flux intercalibration
  (`fa.fit_epochs`).
- Jupyter GUI (`fa.gui()`, needs `pip install "fantasy_agn[gui]"`).
- Publication-quality figures: journal style, physical units, grouped legend,
  optional line labels, single/double column widths; `result.savefig("fit.pdf")`
  writes PDF, EPS, SVG or PNG with embedded fonts.
- Any number of broad components (`broad_components=3`, ...) with `result.bic`
  and `result.aic` to decide how many the data support.
- The optical preset uses a broken power law and high-order Balmer lines.
- Stellar-population host galaxy (`fa.StellarPopulation`, `host=True` in presets):
  non-negative E-MILES templates with stellar velocity and dispersion fitted in
  Fourier space with exact derivatives, matched to the data's resolution;
  `result.host_properties()`. Templates are downloaded on request
  (`fa.templates.download("emiles")`), not bundled.
- Line profiles: Lorentzian, Voigt and Gauss-Hermite (h3, h4) besides Gaussian
  (`fa.Lines(..., profile=...)`, `broad_profile=` in presets).
- Intrinsic AGN reddening (`fa.Extinction`, `extinction=True`): Fitzpatrick (1999)
  or Calzetti et al. (2000); fitted fluxes and luminosities are dereddened.
- Bayesian uncertainties by Hamiltonian Monte Carlo (`errors="mcmc"`): exact
  gradients, preconditioned with the fit covariance, reflection at parameter
  limits, step size tuned by dual averaging; reports split-R-hat and effective
  sample size. No extra dependency.
- Intrinsic Baldwin effect plot for multi-epoch fits (`res.plot_baldwin()`).
- Model comparison with systematic errors (`fa.compare`, `fa.presets.variants`).
- Overlapping components (extra broad components, outflow wings) are also fitted by
  adding them one at a time; the better solution is kept. This fixes local minima
  in fits with three or more broad components.
- Direct download of public DESI spectra from NOIRLab's SPARCL service
  (`fa.desi.fetch`, `fa.desi.find`), no account or extra package needed.
- DESI spectra (`fa.desi`): coadd and spectra files of the DESI data releases (target
  selection by TARGETID, redshifts from redrock/zcatalog files, B/R/Z arms merged,
  masks applied) and SPARCL records; wavelength-dependent instrumental resolution from
  the resolution matrix, applied to each line at its own wavelength.
- Linear parameter ties (`model.tie(a, b, factor, offset)`), fixed line ratios,
  and wildcard parameter names (`model.set("broad.*.flux", min=0)`).
- Host galaxy eigenspectra can be fitted simultaneously (`fa.HostGalaxy`).
- Instrumental resolution is removed from line widths (read from SDSS files).
- Air/vacuum handled automatically; SDSS spectra stay in vacuum.

### Fixed (compared to 1.x)
- SDSS pixels with `ivar = 0` were given the median error and used in the fit;
  they are now excluded.
- Host-galaxy decomposition weighted pixels by 1/flux² instead of 1/error².
- The Monte Carlo error loop started each refit from the previous perturbed
  solution and did not reset; it now starts every refit from the best fit.
- FeII and Balmer templates were re-read from disk at every model evaluation.

### Removed or changed
- Dependencies dropped: sherpa, PyAstronomy, spectres, natsort, scikit-learn,
  RegscorePy, sfdmap2 (now optional, `pip install "fantasy_agn[dust]"`).
- The 160 MB SFD dust maps are no longer bundled. Pass `ebv=...` to
  `deredden` (e.g. from the IRSA dust service), or install sfdmap2, download
  the maps and set `SFD_DIR`.
- Line amplitudes are now **integrated fluxes** (1.x used peak heights), so
  flux tables need no conversion.
- `rebin()` is removed; models are evaluated on the native wavelength grid.

### Moving a 1.x script

| 1.x | 2.0 |
|---|---|
| `read_sdss(f)`, `read_gama_fits(f)`, `read_text(f)` | `fa.read(f)` (format detected) |
| `make_spec(w, f, e, z=...)` | `fa.Spectrum(w, f, e, z=...)` |
| `s.DeRedden(ebv)` | `s = s.deredden(ebv=...)` |
| `s.CorRed()` | `s = s.to_rest_frame()` |
| `s.crop(a, b)` | `s = s.crop(a, b)` (returns a new spectrum) |
| `s.fit_host_sdss()` / `s.fit_host()` | `s = s.subtract_host()`, or `fa.HostGalaxy()` in the model |
| `s.vac_to_air()` | not needed (`s.to_air()` exists) |
| `create_input_folder(...)`, `automatic_path(s)` | not needed |
| `continuum(s, refer=5100)` | `fa.PowerLaw(pivot=5100)` or `fa.BrokenPowerLaw(...)` |
| `create_fixed_model(["hydrogen.csv"], name="br")` | `fa.Lines("br", ["Hd", "Hg", "Hb", "Ha"])` |
| `create_model(["hydrogen.csv"], prefix="br")` | `fa.Lines("br", [...], tie_kinematics=False)` |
| `create_tied_model(name="OIII5007", files=[...], fix_oiii_ratio=True)` | `fa.Lines("narrow", [...], ratios={"[OIII]4959": ("[OIII]5007", 1/2.98)})` |
| `create_feii_model(fwhm=..., ...)` | `fa.FeII(fwhm=(value, min, max))` |
| line lists from CSV | `fa.Lines("br", fa.read_line_list("hydrogen.csv"))` |
| `cont + broad + narrow + fe` | `fa.Model(cont, broad, narrow, fe)` (or `+`) |
| `s.fit(model, ntrial=2)` | `result = fa.fit(s, model)` |
| `s.monte_carlo(nsample=100)` | `fa.fit(s, model, errors="mc", n_mc=100)` |
| `s.gres.format()`, `s.save_json()` | `result.summary()`, `result.save(prefix)` |
