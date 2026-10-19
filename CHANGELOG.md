# Changelog

## 2.0.0

A complete rewrite. Sherpa is gone; fitting uses NumPy/SciPy with exact analytic
derivatives.

### New
- Model uncertainty: `fa.compare` fits the models in parallel and also compares log L5100;
  `fa.presets.variants(..., host=(False, True))`; `cmp.result`, `cmp.systematic()`. App: **Model
  uncertainty** in the Fit step fits 1-3 broad components x outflow x host and shows value +-
  statistical +- model error.
- Saving work: `fa.save(obj, "x.fantasy")` / `fa.load(...)` for fit results, multi-epoch results,
  classifications, models and spectra (stellar templates and caches are not stored: a fit with
  a host shrinks from 13 MB to 0.2 MB). App: save the whole project (cells, outputs, spectra,
  fits) and open it again.
- Plots of single lines: `r.plot_line("Hb")` shows one line with the rest of the model subtracted,
  every component (narrow, outflow, 1st/2nd broad with their FWHM) and their sum on a velocity
  axis; `r.plot_lines([...])`, `r.fitted_lines()`. The main figure draws the individual broad
  components when a line has several (`group="auto"`, the new default). App: **Single line**
  view with a list of the lines, arrows / arrow keys and Alt-click on a line; broad components
  in the interactive plot.
- Borderline hosts (host >= 10% at 5100 A but sigma* not confirmed): the classification says so
  (`measurements["host_borderline"]`) and the app's Fit fits with and without the host and keeps
  the host if the BIC prefers it (dBIC < -10). With this, the host is used for 95% of the
  calibration quasars whose fit it improves.
- `classify` recommends a stellar host for a type 1 AGN when the stellar absorption is detected
  (sigma* S/N >= 5) or the host dominates (>= 50% at 5100 A). On 300 SDSS quasars this found
  82% of those whose fit the host improves (was 65%) and was right in 83% of its choices; the
  common "host >= 10%" rule was right in only 59%.
- Presets calibrated on 1500 SDSS spectra (calibration/, tested on a held-out 30%): [OIII]
  outflow wing on by default in `optical` and `hbeta`; FeII shift tied to broad Hb in `hbeta`;
  `mgii` without narrow MgII (`narrow=True` to add it); `civ` with two broad components and the
  HeII shift tied to CIV; `composite` and `starforming` without outflow wing by default;
  `classify` recommends `optical(host=True)` for host-dominated quasars. App: the outflow box
  follows each preset's default; MgII, CIV and "optical + host" in the model menu.
- `spec.mask_absorption()` masks narrow absorption lines (CIV, MgII absorbers); app option
  "Mask narrow absorption" in step 1.
- OzDES DR2 spectra: `fa.read` reads the original multi-exposure files (stack, or
  `epoch=k`) and Data Central SSA files (noise estimated from the data);
  `fa.ozdes.read_epochs(path, skip_qc=True, combine="night")`, `fa.ozdes.search`,
  `fa.ozdes.download` (original files from VizieR). Not flux calibrated: `flux_unit` is None.
- Installation for testers: `sh install/make_bundle.sh` builds `dist/FANTASY-<version>.zip`
  (wheel, tested package versions, installers for macOS, Linux and Windows). The installer
  sets everything up with uv and runs the self-test.
- `fantasy` command: `fantasy test [--full]` (end-to-end self-test), `fantasy app`,
  `fantasy info`, `fantasy examples DIR`, `fantasy python`.
- Example spectra in the package: `fa.examples.path("type1")`, `fa.examples.path("seyfert2")`.
- Variable projection: the least-squares iterations move only the nonlinear parameters
  (shifts, widths, E(B-V)); the amplitudes are solved exactly at every step. Same results,
  2-5x fewer iterations; a Seyfert 2 + BC03 host fit of a GAMA spectrum that crawled through
  5000 evaluations (624 s) now converges in 13 (4.5 s). `FANTASY_VARPRO=0` restores the
  joint iterations.
- Single fits use several cores: the start-value and local-minimum grid searches run in
  threads (`FANTASY_THREADS`, default min(8, cores)) and the stellar-template FFTs use all
  cores.
- `fa.fit_many(spectra, model, n_jobs=-1)`: one model, many spectra, in parallel processes.
  `fit_epochs(..., n_jobs=-1)` runs its independent starting fits in parallel (24 SDSS-RM
  epochs: 69 s -> 22 s on 10 cores).
- Progress of long computations: `fit_many` and `fit_epochs` report through
  `fantasy_agn.progress` (`set_hook`); the app shows a progress bar with the fitted
  spectra, the time left and the joint-fit iterations, and printed output appears while
  a cell runs.
- App, multi-epoch results: **Fit of each epoch** (data, model, components, residuals) with
  arrows / arrow keys, a list, or a click on a table row; one PDF with all epoch fits.
- App: **Multi-epoch** view (drop several spectra or give a pattern; model, shared
  parameters, intercalibration, processes; light curves as figure, table and CSV).
- Faster single fits (about 2-2.5x): NNLS for the amplitudes, value-only evaluation in the
  start-value search, early stop of the search, targeted local-minimum scans.
- IRSA E(B-V) lookups are cached per position.
- Presets for every class of `classify()`: `seyfert2`, `composite`, `liner`, `starforming`,
  `passive` (= `retired`), chosen by fitting SDSS examples of each class one ingredient at a
  time (host dust; BC03 young populations for star-forming galaxies; two narrow-line systems
  and an outflow wing incl. [SII] for Seyfert 2s; Case B + gas extinction for H II regions;
  free AGN line fluxes). χ²/ν: star-forming 6.9 → 2.8 and 6.2 → 2.2, composite 4.6 → 1.5,
  Seyfert 2 2.1 → 1.3, 2.0 → 1.1, 4.1 → 2.1. `classify()` and the app recommend them.
- Lines: H9–H12, [NII] 5755, [SIII] 6312.

### Fixed
- App: after reading a new spectrum, Fit still used the model created for the previous one
  (e.g. a Seyfert 2 model, without FeII, for a quasar). Reading a spectrum now clears the
  model (`model = None` in the session), so Fit uses the model recommended for the new one.
- Figures: pixels excluded from the fit were joined by straight lines across the masked gaps.
- `classify` missed the broad lines of about half of the low-redshift quasars: its broad-flux
  S/N > 20 condition failed when the broad line trades flux with the outflow wing. Recalibrated
  on 300 SDSS quasars and 650 narrow-line galaxies (calibration/): a broad line now needs
  dBIC > 25, S/N > 5, EW > 30 A and FWHM 1200-12000 km/s. Quasars found: 47% -> 76% (the rest
  show no broad line); galaxies wrongly called type 1: 0.9%.
- `fit_epochs`: an epoch whose own fit found no narrow lines gave an infinite starting
  flux scale (NaN in the joint fit). Starting scales and their limits now come from the
  flux levels when needed (uncalibrated spectra can differ by more than 0.3-3x). The joint
  fit stops when chi2 improves by < 0.1 over 20 iterations instead of crawling to 300.
- Error propagation keeps its finite-difference steps within the parameter limits (a width
  stepped below zero gave NaN and an IndexError in `measure`).
- Data Central spectra (header keyword WMIN) were mistaken for GAMA files.
- App: a wide light-curve table stretched the page, so the figures were drawn several
  times too large; figures are now shown at their natural size (rendered at 2x for sharp
  screens).
- `fit_many` worker processes run single-threaded (no processes x threads
  oversubscription; 20 epochs 17.6 -> 12.4 s); the joint multi-epoch evaluation runs the
  epochs in threads.
- `host_properties()` / `key_results()` could fail with "operands could not be broadcast"
  when the app drew a plot while a cell was running: the stellar component kept its
  prepared grid in attributes that another thread could replace. Grids are now immutable
  objects; component caches are thread-safe.
- SDSS spectra: the instrumental resolution now follows the wavelength (from `wdisp`;
  ~170 km/s at 3800 Å, ~130 km/s at 8000 Å) instead of one median value.
- BC03 (Bruzual & Charlot 2003, GALAXEV) stellar populations for the host: `host="bc03"`
  in the presets, `StellarPopulation(templates="bc03")`, `classify(templates=...)`; app:
  host-galaxy menu (E-MILES, BC03, PCA, none) and template choice in the model editor.
  Template sets download automatically on first use (`FANTASY_NO_DOWNLOAD=1` to switch
  off); BC03 wavelengths verified to be in air.
- `optical`: narrow HeII 4686 by default; [NeIII] 3869 shares the [OIII] outflow wing;
  `red=True` extends the fit to 7500 Å (HeI 7065, [ArIII] 7136, [OII] 7320, 7330). App:
  "Also include" options for coronal lines, OI 6046 and the red side (found missing in
  IC 1198, a coronal-line Seyfert: χ²/ν 1.21 → 0.89).
- Galactic extinction: `spec.deredden()` looks up E(B-V) automatically (value in the file
  for DESI, local SFD98 maps, or the IRSA dust service), with the Schlafly & Finkbeiner
  (2011) recalibration; never applied twice; `spec.dereddened`. The fit summary and the
  classification note uncorrected spectra. The app and `fa.gui()` have
  "Correct Galactic extinction" and "Move to rest frame" options when reading.
- App redesign with the FANTASY logo: header with logo, wordmark and step indicator;
  navy code panel with Code/Data tabs, Python version and a code input at the bottom;
  step cards with side panels; results as large numbers; an interactive spectrum plot
  (zoom, pan, hover read-out, legend toggles, full screen) next to the publication
  figure (PDF/EPS/PNG).
- App: **Model** tab (start from a preset, add/remove components with a form, edit
  parameter values, limits, fixed and ties in a table; every change is written as code)
  and **Docs** tab (guides, line list and an API reference generated from the docstrings,
  with search). Steps are now read → classify → model → fit → results.
- **FANTASY app** (`fantasy-app`, `python -m fantasy_agn.app`, `fa.app()`): a local web
  application (standard library server, no extra dependencies) with the steps
  read → classify → fit → results and an editable code panel: buttons generate code into
  cells, every cell can be edited and re-run, and the session can be downloaded as a
  script. API protected by a per-start token and a Host check.
- Launchers in `launch/` (macOS, Linux, Windows) run the app with uv, Python 3.12 and the
  locked dependencies (`uv.lock`, `.python-version`) in `~/.fantasy_agn/env-py3.12`.
- `FitResult.key_results()`: the main numbers of a fit as a table.
- `fa.gui()` is now the guided workflow; `fa.gui(classic=True)` the tabbed form.
  `Classification.plot_steps()` shows what `classify()` did.
- `fa.classify(spec)`: classifies a spectrum before fitting: stellar continuum
  (E-MILES + featureless power law, dust screen), broad-line test (ΔBIC, S/N, FWHM),
  BPT [NII]/[SII]/[OI] diagrams (Kauffmann 2003; Kewley 2001, 2006) and WHAN
  (Cid Fernandes 2011). Classes: type1, type1_host, seyfert2, liner, composite,
  starforming, retired, passive; `.model()` gives the recommended preset, `.plot()`
  the diagnostic diagrams. Validated against MPA-JHU on 11 SDSS galaxies.
- New presets `type2` (host + featureless continuum + narrow lines + outflow) and
  `galaxy` (stellar populations + narrow lines).
- `presets.optical(physical=True)`: physically motivated decomposition after QSFit
  (Calderone et al. 2017): one AGN power law (α_λ = −1.7, Selsing et al. 2016;
  `slope=None` fits it within −2.33…−1.0), stellar host, velocity-broadened Balmer
  continuum from 3550 Å, broad components ≤ 15 000 km/s, and an [OIII]-tied
  [NII]/Hα outflow wing.
- `FitResult.balmer_decrement()`; the summary warns when broad Hα/Hβ is outside
  2.5–4 (typical ~3, Dong et al. 2008).
- Coronal lines are split by ionisation: [FeVII] and [FeX]/[FeXIV] (`coronal_high`)
  have separate kinematics (FWHM up to 3000 km/s).
- More lines: narrow H8, [NeIII]3968 (tied to 3869) + Hε, [SII]4069,4076,
  [NI]5198,5200 and HeI6678 in `optical`; [NI]5198,5200 in `hbeta`; broad HeI4471
  (`hbeta`, `optical`) and HeI6678 (`optical`). Optional `coronal=True` ([FeVII]3759,
  5159, 5721, 6087, [FeXIV]5303, [FeX]6375 with their own kinematics) and
  `oi6046=True` (broad OI 6046, `optical`).
- `Lines(max_ratios={line: (reference, r)})`: upper limits on flux ratios
  (kept linear: the reference line is fitted as an excess above flux/r). The
  `halpha` preset uses it for the outflow wing: [NII]6583/Hα ≤ 3 and FWHM ≤ 2000 km/s,
  so the wing can no longer absorb the broad Hα flux as [NII].
- `halpha(outflow=hbeta_result)` (or `{'shift':, 'fwhm':}`) fixes the [NII]/Hα wing to the
  kinematics of the [OIII] outflow (`presets.outflow_kinematics`). The preset now has
  narrow and broad HeI 6678 (broad tied to the main broad Hα; `hei=False` to drop).
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
