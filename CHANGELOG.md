# Changelog

## 2.0.0

A complete rewrite. Sherpa is gone; fitting uses NumPy/SciPy with exact analytic
derivatives.

### New
- License: FANTASY is released under the BSD 3-Clause License (`LICENSE`, also in the package metadata);
  the sources of the bundled data are listed there. `CITATION.cff` gives the papers to cite (GitHub shows
  **Cite this repository**).
- New versions: the app shows **New: FANTASY x.y.z** in its header when PyPI has a newer release, with
  what changed, the update command for this installation (installer/uv tool, pip, or a source checkout)
  and an **Install** button; ⋯ › **Check for updates** asks now. In a terminal: `fantasy update`
  (`--check`: only check), and `fantasy info` shows the latest version. PyPI is asked at most once a
  day and gets nothing but the package name; `FANTASY_NO_UPDATE_CHECK=1` turns it off. In Python:
  `fantasy_agn.update.check()` and `install()`.
- Releases: `.github/workflows/release.yml` publishes to PyPI (Trusted Publishing) and makes a GitHub
  release when a `v*` tag is pushed, after the tests pass; see `RELEASING.md`. The version is set only
  in `fantasy_agn/__init__.py` (`pyproject.toml` reads it).
- App, Read › Path / glob: a folder reads the spectra in it, and pasted paths work with quotes, terminal escapes
  (`My\ Data`), `file://` URLs and `[ ]` in file names. When nothing matches, the message appears under the box
  and says why (the folder does not exist, a relative path). Before, the error showed at the top of the page and
  went away at the next state update a few seconds later, so Read seemed to do nothing.
- App, Read: typing works in the Path / glob, DESI TARGETID and SDSS plate-mjd-fiber boxes. Their Enter-key
  handler returned `false` for every other key, which cancels the key in the browser, so only pasted text got in.
- Line profiles are averaged over each pixel, as the spectrograph records them, instead of taken at the pixel
  centres: exactly for Gaussians (through the error function, with exact derivatives), by Simpson's rule for the
  other profiles. The pixel widths come from the spectrum's own grid (`Spectrum.pixel_grid`, passed to the
  components in `spectrum.context`). Point values overestimate the width of lines only a few pixels wide: on an
  SDSS-like grid (69 km/s pixels, 150 km/s resolution), FWHM 120 km/s came out at 129 (+7%) and 200 at 205 (+3%);
  now both within 0.3%. Broad lines and Fe II do not change. A fit costs about 5% more. Models evaluated
  without a spectrum, and the intrinsic line measurements (FWHM, σ), use the profile itself. Results saved
  before this change evaluate their model with pixel means when loaded, which can differ slightly from the fit.
  `FANTASY_PIXEL=0` returns to point values.
- Faster fits: when a model has outflow or extra broad components, `fa.fit` runs its two searches (the direct fit
  with its escapes, and the staged build-up) at the same time, the second in one persistent helper process
  (`python -m fantasy_agn._worker`). The result is identical; a typical SDSS Hβ–[O III] fit takes about 5 s instead
  of 8 s. Worker processes of parallel sample and multi-epoch fits, and `FANTASY_HELPER=0`, run both in turn as
  before; any failure of the helper falls back to that.
- `fa.suggest_lines(r)`: emission lines the fit is missing, in milliseconds. Every library line the model lacks is
  tested at the fitted width and shift of the model's own line components (forbidden and coronal lines with narrow
  and outflow, permitted also with broad), with the flux and Δχ² computed on the residuals without refitting, scaled
  by the residual scatter; lines that lower the BIC by more than 10 are picked greedily (each subtracted before the
  next). `attrs["code"]` adds them as tied `fa.Lines` components. In the app, Results › Checks › **Lines to add**
  lists them for a single fit; **Add & refit** adds a line and fits again in one cell with the options of the
  last fit, without leaving Results (after the trial fit of a multi-epoch analysis: that epoch only) (new
  `POST /api/wf/addlines`).
- `Model.split_lines(name, lines, new_name)`: moves lines (with their doublet partners) of a Lines component to a
  new component with its own shift and width, keeping fluxes, ratios, limits and ties.
- `fa.suggest_lines` also finds lines of the model whose velocity differs from the other lines of their component
  (e.g. [O III] blueshifted against the other narrow lines), from the residuals projected on the line's velocity
  derivative; rows with `action == "split"` give the offset `dv` and `model.split_lines(...)`. The residual analysis
  suggests the same for a feature on such a line, instead of "check ties and limits". In the app the list is
  **Lines to add or separate**, with Separate & refit and Apply all & refit.
- `fa.analyze_residuals` no longer suggests broad components for forbidden lines, for excesses wider than
  12 000 km/s, or, in a model without broad lines, for lines other than the strong broad ones (Hα, Hβ, Mg II, C IV,
  ...): such excesses are reported as continuum or host mismatches.
- Residual analysis and `fa.suggest_lines` now search the whole high-ionisation list: the [Fe V], [Fe VI], [Fe IV],
  [Ar IV] and [Ar V] lines of the coronal list are typed forbidden and were left out ([Fe V] 4181, 4227 went
  undetected).
- App, Model › **Compare models on this spectrum**: variants of the recommended preset (one broad component more and
  fewer, each on/off option switched, then the best combination) fitted in the background and ranked by BIC, with a
  recommendation (lowest BIC without new parameters at a limit; a change must win by ΔBIC < −10), the spread of the
  main measurements across the best models, and **Use this model**, which runs the candidate's code as a session cell.
- App, Model › **model builder**: a full-screen workspace where any model (a preset, a candidate, the current model or
  an empty one) is taken apart into components and every constructor argument and parameter can be edited: add,
  duplicate, switch off or rename components, lines by name, ion, wavelength or your own `name@λ`, flux ratios, ties of
  any parameter, limits; a live initial guess per component, click the spectrum to add a line, undo/redo. The model is
  shown as plain FANTASY code (`fa.Model(...)` plus `model.tie/set/fix`), used as the session model or fitted and
  ranked with the candidates. New endpoints `/api/wf/gallery`, `/api/mb/*`.
- App, Results › **Light curves & lags** redesigned: choose a driver and a responder (any continuum, Fe II, or the
  flux, FWHM or EW of a broad line), see both with the responder shifted back by a lag slider, and the lag of the pair
  by ICCF, von Neumann, ZDCF and DRW, each with its own curve; the adopted method sets the curve under the slider and
  the `primary` method of **Measure lag** (`me.reverberation(..., primary=...)`). New `GET /api/wf/lagpair`.
- `MultiEpochResult.window_continuum(windows=None)`: continuum light curves measured on the data in line-free
  rest-frame windows (default: Hβ 4435–4700 and 5100–5535 Å, Hα 6000–6250 and 6800–7000 Å, after Shen et al. 2011,
  where the fit covers them), the mean of the intercalibrated flux minus the fitted lines and Fe II, with errors.
  In the app they are drivers in Light curves & lags (the 5100–5535 Å window is the default), and Measure lag
  then passes them as `continuum=`.
- App, Results › **Atlas** (samples and multi-epoch fits): every fit as a small card coloured by class (or by date),
  ordered by χ²ν, residual structure, FWHM, R_FeII, M_BH, Eddington ratio or, for epochs, date, luminosity and
  broad-line flux; the fits worth a look are flagged with the reason, and a card opens its fit.
- App, Results › **Degeneracies**: the correlations of the free parameters as a wheel, grouped by component, with
  ribbons for the pairs above a threshold (red together, blue against), the strongest trades between components
  with a reason for each, and the joint 68%/95% region of any pair.
- App, Results › **Anatomy**: the fit as stacked components with the data over them, the share of each component in
  the flux of the range shown, the composition at any wavelength on hover, **Peel layers** (the data minus the top
  components: what is left for the rest to explain) and the residuals in σ with a running mean, which shows structure
  the model misses.
- `fa.classify_fit(r)`: the final class of a fitted spectrum, from the values of the fit (broad lines summed over
  their components; narrow-line ratios and EW(Hα) of `r.diagnostics()`), with the rules of `fa.classify`, which stays
  the preliminary classification that chooses the model. App: the Results page shows the final class after every fit
  (every epoch of a multi-epoch fit), with the preliminary one beside it when it was made; the report lists both.
- App: the session is saved after every step (to `~/.fantasy_agn/sessions`, or `$FANTASY_AUTOSAVE`; `0` switches it
  off), and the Read step offers **Restore last session** (and the one before) when the app starts again or after New
  analysis: a server restart no longer loses the work. **■ Stop** in the top bar stops a running step: the worker
  processes of a parallel run are ended and the step is interrupted, what ran before stays.
- App, Prepare: with several spectra the redshift can be **one for all (median)** of the files, the default for epochs
  of one object; the note says by how much the redshifts of the files differ. Hovering over a candidate of **Find the
  redshift** marks its lines on the spectrum.
- Multi-epoch: `fa.fit_epochs(..., epoch_broadening=True)` fits an extra broadening per epoch, a FWHM added in
  quadrature to every line width and the stellar dispersion (up to 300 km/s, or the limit given; an epoch may be
  sharper than the reference), measured on the shared narrow lines. Exact for Gaussian lines; each epoch's result keeps
  the intrinsic widths and its spectrum the broadened resolution. In simulations with epochs up to 250 km/s apart in
  resolution the flux scales were off by up to 7% without it and below 1% with it, and the broadening was recovered to
  about 10 km/s. `me.broadening` lists it. App: Fit step, *Resolution of the epochs*; a column in the table of all
  epochs and a Results tile.
- `fa.calibration_check(me)`: every epoch of a joint fit fitted again on its own with the joint flux scale; whether the
  shared line fluxes agree within their errors (chi2/dof, p, the excess scatter that is calibration noise not in the
  error bars, the epochs more than 3 sigma off). `fa.simulation_check(me, n=20)` checks a joint fit: sets of epochs
  drawn from it and fitted jointly again; bias, scatter and coverage of the flux scales, the light curve, offsets,
  broadening and shared widths. App: Results, Checks, *Is the calibration between the epochs right?*
- App, Model step: the host's flux in the sharing table can be *amplitude per epoch* (`epoch_scaled`), which replaces
  the Advanced option *Host amplitude free in each epoch*. After a fit of one spectrum, the *Shift and width* choice
  says for every possible tie whether it would change the broad-line FWHM and flux by more than 5%
  (`fa.model_confidence` on that fit, no new fit).
- Reverberation: the significance of every lag (`rm.lag_table()` columns `p_side` and `significance`,
  `fa.rm.lag_significance`): the share of the lag distribution on its side of zero as a one-sided Gaussian equivalent.
  Warnings when the adopted lag is below 2 sigma and when r_max < 0.5. App: a Significance column in the lag table.
- `r.line_limit(line)`: the upper limit on the flux of an undetected line, the fitted flux (if positive) plus
  3 sigma_F from the noise under a line of the fitted shape, or the 95% posterior limit with MCMC errors. The app's lines
  table shows it for undetected and weak lines instead of 0.
- App, Results, Download: **Report**, one HTML file of the current fit with the figure, the statistics and warnings, the
  key results and lines, every check run on this fit with its outcome, the settings and the Python code.
- The website's first-fit guide includes a narrated, captioned walkthrough of a single-spectrum Hβ fit in the app, with chapters for preparation, the fit, checks and export.
- The project website includes searchable, offline documentation generated from the current app guides and Python API, including redshift search, weak Fe II, fit checks and multi-epoch measurements.
- A project landing page in `website/` lets visitors reconstruct the Hβ blend in NGC 2885, watch the scored animation, and find the v2 installation and first-fit instructions.
- `fa.find_redshift(spec)`: the redshift of a spectrum that comes without one. The observed spectrum is fitted with
  galaxy and quasar eigenspectra (Yip et al. 2004, shipped with FANTASY) and a polynomial at every redshift; the minimum
  of chi2 is the redshift, as in the SDSS and DESI pipelines. The result has the redshift and its error, the template
  set, `secure` (is the next candidate clearly worse), the candidates with the lines each would show, and a figure
  (`zf.plot()`). A quasar redshift is refined on the narrow H-beta and [OIII] lines when they are detected;
  `catalog=True` also reports what SIMBAD, NED and Milliquas list. On 540 SDSS spectra of nine classes (z = 0.01-2.5)
  97% of the best redshifts were right, and 99.8% of the 96% called secure; quasars above z ~ 1.7 are the weak spot
  (83-90% right, the true value among the candidates for 95%). About 3 s per spectrum. App: Prepare, **Find the
  redshift**, with the candidates to click and the figure.
- Multi-epoch: `fa.fit_epochs(..., wavelength_offsets=True)` fits a wavelength offset per epoch, a velocity within
  +-150 km/s (or the limit given) added to every shift of the model in that epoch, with the reference epoch at 0. The
  shared narrow lines measure it, as they measure the flux scale. Epochs are never on exactly the same wavelength
  scale, and files with slightly different redshifts (SDSS pipeline: about 20 km/s rms, up to 70) are each moved to the
  rest frame with their own. In simulations of 8 epochs, unmodelled offsets of 30 km/s rms raised the error of the flux
  scales from 4% to 17% (some scales ran to their limit); with the offsets fitted it was 3%, the offsets were recovered
  to 3 km/s with correct errors, and with no offsets in the data nothing changed. `me.offsets` lists them and
  `me.redshift()` gives the redshift of the object from the shared narrow lines, with its error. App: Fit step,
  *Wavelength scale of the epochs* (fit a small offset per epoch, the default there, or as prepared), with a note when
  the redshifts of the files differ; the table of all epochs has an Offset column and Results a tile *Redshift, narrow
  lines*. The Prepare summary no longer shows z = 0 when the first file is unusable.
- App: the run button of the Fit step is in the bar at the bottom of the page, so it is always in view, and only one
  button there is highlighted. Each check of the Results step is one line with a status that opens to show the rest; a
  check that needs attention opens by itself, and the Checks tab shows how many do. A joint fit shows the median error
  of the flux scale of its epochs as a tile, marked above 2%. The page loads the changed stylesheet and script after an
  update without a hard reload.
- App, Results: a black-hole mass is shown with both its errors, that of the fit and the scatter of the recipe
  (8.64 ± 0.01 fit · ± 0.43 recipe, dex), instead of the fit error alone; line fluxes carry their unit (tiles, the lines
  table, the light curves); the table of all epochs has readable headings, the date and the MJD in full and the flux
  scale of each epoch; errors are written out (± 280, not ± 2.8e+02); the light curves show a typical error bar inside
  the frame and their axes never repeat a tick label.
- App, multi-epoch: **The same in all epochs, or one per epoch?** A table in the Model step sets, for every component,
  whether its fluxes and its shape (shift, width, profile) are shared by the epochs of the joint fit or fitted in each.
  The defaults are the old behaviour (narrow lines and host shared; shape of the broad lines and Fe II shared); the table
  replaces the "same broad-line shape" checkbox, the component list shows what is shared, the Fit step repeats it in one
  line, and the name patterns of the advanced options still override it. A fit with nothing shared is refused.
- Galactic extinction: when the IRSA dust service cannot be reached, the error says so (the service is down or the
  computer is offline) instead of only quoting the connection error, and the Prepare step of the app offers the ways on
  (enter E(B-V) by hand, go on without the correction, install the maps for work offline).
- Multi-epoch: agree on the model with one epoch first. `model.starting_from(r)` gives a copy of a model whose free
  parameters start at the values of a fit (limits, fixed values and ties stay those of the model); a fit of it also tries
  that start and keeps the better solution: `me = fa.fit_epochs(spectra, model.starting_from(r))`. App, Fit step of a
  multi-epoch analysis: **First, try the model on one epoch** fits one epoch alone (highest S/N by default); Results shows
  it with every check and leads back to the model or on to **Fit all epochs**, which then start from the trial fit
  (*Start every epoch from the trial fit*).
- Instrumental resolution: `spec.with_resolution(150)` (FWHM in km/s) or `spec.with_resolution(R=2000)` sets it for
  files that do not carry it (text, plain FITS), so that fitted widths are intrinsic. `r.undersampled()` and a warning
  in the fit summary name the line widths below 1.2 pixels (the fitted width and the resolution in quadrature): the
  model is evaluated at the pixel centres, so the flux of such a line depends on where its peak falls between two pixels
  (1% at 1.2 pixels, 6% at 1.0, 30% at 0.7). This happens when the resolution is unknown and a narrow width ends at its
  lower limit. App: Prepare, **Instrumental resolution** (as in the file, FWHM in km/s, or R) with the pixel size; Results
  shows *A line narrower than the data can sample* with a button to where it is fixed.
- `fa.simulation_check(r)`: would this fit recover the truth? The fitted model is taken as the truth, spectra are drawn
  from it with the noise of the data and fitted again from scratch as the spectrum was (60 by default, about a minute for
  the H-beta region). For every result (broad-line FWHM, sigma, flux and shift, R_FeII and the Fe II flux, [OIII], the
  continuum, or your own functions) the table gives the bias, the real scatter, the reported error bar and how often the
  truth is within it, with a flag: biased, error bar too small or too large, ok. A fit pruned with `fa.prune_feii` is
  simulated with the pruning. On the example quasar it flags the Fe II flux (+14%), R_FeII, the continuum and the
  broad H-beta flux as biased, and none of them after pruning. App: Results, **Checks**, 5. *Would this fit recover
  the truth?*
- `fa.prune_feii(r)`: fits again without the Fe II multiplets that the data do not detect (amplitude below 2 sigma
  or at its bound), and lists them in `r.feii_pruning`. The free, non-negative multiplet amplitudes make weak Fe II come
  out too high, with the continuum too low and the broad line beside it too strong: in simulations of an SDSS quasar at
  S/N 30 the Fe II flux was 9% too high at R_FeII = 0.25, 66% at 0.08, and with no Fe II the fit gave R_FeII = 0.12 and
  a broad H-beta flux 12% too high. Pruning removes most of this (broad H-beta with no Fe II: +15% -> +1%) but at
  moderate Fe II it drops real weak multiplets (Fe II flux 13% low instead of 6% high at R_FeII = 0.25), so it is
  opt-in. Nothing changes in fits that do not call it.
  App: Results, **Checks**, 4. *Is weak Fe II inflating the fit?* lists the groups (detected or not), drops the undetected
  ones at 2 or 3 sigma, shows what changed and can go back to the fit with every group.
  Epochs and samples: `fa.prune_feii(fits)` prunes a list of fits (the epochs of `fa.fit_many`) in parallel, and
  `fa.fit_sample(files, prune_feii=True)` prunes every spectrum with Fe II; the catalog gets `feii_groups`, `feii_dropped`
  and the position before pruning (`R_FeII all groups`, `broad Hb FWHM all groups`). App: the Fe II check also works for
  epochs fitted one by one (a line per epoch; light curves and the main sequence then use the pruned fits); the Fit step
  of a sample has *Drop undetected Fe II groups and fit again*, the list a *Fe II dropped* column, and the main sequence
  diagram *before pruning Fe II*. A joint fit has no pruning step.
- Faster fits of models with many lines: Gaussian lines are evaluated only within 8 sigma of their centre (beyond that
  they are below 1e-14 of the peak), so the Fe II and high-order Balmer templates no longer cost pixels x lines. The
  results do not change (the same chi2 and parameters to rounding); a full optical fit of an SDSS quasar takes
  about half the time on one core (15 -> 8 s, 19 -> 10 s), the H-beta region 1.2-1.5x less.
  `fantasy_agn.profiles.WINDOW = 0` switches it off.
- Faster grid searches (the initial guess, the search for a better solution after a fit, placing a component): for each
  trial only the component that changes is evaluated again and only its part of the normal matrix is rebuilt; the chi2
  values are the same to 1e-12, and so are the fits. With the windowed lines a full optical fit of an SDSS quasar now takes
  5-6 s on one core instead of 15-19 s. `FANTASY_SCANNER=0` uses the general solve for every trial.
- `fa.lookup_catalogs`: what SIMBAD, NED, Milliquas, AllWISE, FIRST, NVSS, XMM, Chandra and 2RXS list around each object, for one
  object or a whole sample, with warnings where they disagree with the fit (type, redshift, WISE colour). Sends only coordinates;
  no key and no new dependency. App: **Look up** in step 1 (Read), with a sky image (SDSS, Pan-STARRS, Legacy Surveys or DSS, `fa.catalogs.sky_image`); after the fit
  the warnings are worked out again with the classes found, and a sample gets new columns and a "Catalogs disagree" filter.
- `fa.classify` (and so automatic mode of a sample and the app): a spectrum that ends below 5200 A in the rest frame is labelled
  `uv_mgii` or `uv_civ` and gets that line model, instead of failing or being called passive.
- App, Results of a sample: a broad-line width at the limit of the model now puts the spectrum on the "To check" list.
- `fa.EpochSet(fits)`: epochs fitted one by one (`fa.fit_many`) seen together, with the light curve, rms spectrum,
  reverberation (lags, rms widths, mass) and Baldwin plot of a joint fit (no intercalibration; failed epochs left
  out). App: fitting epochs one by one now also gives **Light curves & lags** (before, only the joint fit did).
- App, Results of a sample: tiles that filter the list (fitted, to check, not fitted), class filter, search, a list that sorts
  by any column with readable headings, why spectra failed and what to do, previous / next (or the arrow keys), and
  a Diagrams tab (BPT, quasar main sequence) coloured by class. Opens on the first fitted spectrum.
- Fixed: lags depended on the order of the epochs. `rm.iccf` interpolated light curves that were not in time order
  (a joint fit keeps the order of its input), which gives wrong values without a warning: for RM 17, 2 of 76 epochs out
  of order turned the ICCF centroid from +44 d into -92 d. All lag methods now sort by time first. App: light curves are
  joined in time order; the Main sequence tab follows the fit, not the model being edited.
- Correlations over epochs: `ep = fa.epoch_table(me)` (or a list of fits of the epochs) gives per epoch the broad
  flux, FWHM, sigma, shift and EW of each broad line, the continuum and log L, R_FeII and the total [OIII]5007 flux,
  with errors; `fa.epoch_correlations(ep)` the Spearman rho, p and Pearson r of every pair (pairs that are the same
  number twice, such as widths tied in the model, are marked trivial and listed last);
  `fa.plotting.plot_correlations(ep, cols)` a corner plot with each quantity against time on the diagonal, points
  coloured by date and rho, p in every panel. On 76 SDSS-RM epochs of RM 17 it shows the intrinsic Baldwin effect
  (log L against H-beta EW, rho = -0.91) and the breathing of the broad-line region (against FWHM, rho = -0.65).
  `fa.epoch_table` measures the epochs on all cores (76 epochs: 20 s instead of 3 minutes).
  App, Results of several epochs: **Correlations** measures the epochs when opened, then works in the browser:
  a colour map of rho (click a square), one large relation with its fit, numbers above it (slope, scatter, rho, p,
  epochs), a time-lag slider that updates the map and the fit as it moves, hover for an epoch, click to leave it out.
  Matplotlib only draws what is saved (PDF / PNG / SVG, Python code, with the same lag and epochs).
  `fantasy_agn.multiepoch.light_curve_table(epochs)` builds the light curve of epochs fitted one by one.
  The relation of a pair: `fa.correlations.fit_relation(ep, x, y)` / `fa.plotting.plot_relation(ep, x, y)`,
  maximum likelihood with errors in both quantities and intrinsic scatter, bootstrap errors, fluxes, widths and EWs
  in log10 (power-law slopes). RM 17, 76 epochs: Baldwin slope -0.52 ± 0.05, breathing slope -0.115 ± 0.017.
  Time lag: `fa.correlations.shift_continuum(ep, lag)` (app: Continuum earlier by, observed-frame days) sets each line epoch
  against the continuum interpolated one lag earlier; for RM 17 with its H-beta lag (172 d observed) the Baldwin slope
  is -0.11 ± 0.06 instead of -0.52 ± 0.05 at zero lag.
- Figures: `r.plot(zoom=[(4650, 5150), (6350, 6800)])` adds zoom panels under the spectrum
  (`zoom_subtract=("continuum", "feii")` shows the lines alone in them); `legend="inside"`
  places the legend where it covers the fewest data and line labels ("right" beside the plot); journal widths
  (`width="mnras"`, `"mnras-double"`, `"aa"`, `"aa-double"`, `"apj"`, `"apj-double"`); SVG keeps its text editable.
  Line panels give FWHM, sigma and flux rounded to their errors ("(1.46 ± 0.09)×10⁴"), mark the half-maximum
  points, draw the summed broad profile, flag a fragile FWHM and label wide velocity axes in 10³ km/s.
  `r.plot_profiles(["Hb", "Ha"])` / `fa.plotting.plot_profiles([r1, r2], "Hb")` overlay profiles scaled to their peaks.
  Many curves (epochs, in time order) follow a sequential colour ramp. App, Line profiles, Overlaid: for several
  epochs, **Compare: Every epoch** overlays the chosen line of every epoch, labelled by date.
  `r.plot(style="sans-serif", height=0.8)`: sans-serif font, compact or tall figures; `plot_lines(..., sharey=True,
  legend=False)`: one flux scale for all panels, no legend.
- App, figure side panels: collapsible sections (Size and style, Range and zoom, Components, Legend and labels; Lines,
  Display, Axes, Size and style) each with a one-line summary when closed, segmented buttons and switches, a save
  bar always in view (PDF, PNG at 300 or 600 dpi, EPS, SVG; Reset), sections remembered; new options: font, height,
  legend and one flux scale for the line panels. Downloaded PNGs are 300 dpi (they were 150). The figure style
  (width, height, font, legend, colours, line options) is remembered between sessions; Reset restores the defaults.
  A range zoomed in the interactive fit plot goes to the publication figure with **Use this range in the figure**.
- App, Results: a **Checks** tab (residuals, robustness, compare, numbered in the order to use them); Publication
  figure with journal widths, legend position, zoom panels (quick Hβ / Hα / Mg II buttons), SVG and the Python code
  of the figure to copy; Line profiles with broad lines listed first, **Each in a panel** or **Overlaid**, a velocity
  range and the code; the figure preview stays in view while the options scroll; results tiles rounded to their
  errors. The step bar no longer loses clicks while a computation updates the page.
- Compare two fits: `cmp = fa.compare_fits(a, b, labels=("power law", "broken power law"))` puts two fits of the
  same spectrum side by side: chi2 on the same pixels, the RMS difference of the total models in units of the
  error, the results of both with their difference in errors, and warnings when a result moves while the fits
  are equally good or when a broad profile has a shoulder at half maximum (its FWHM is then fragile,
  `fwhm_fragile`). `cmp.plot()`: both fits and their difference, where the flux went between the components,
  and both broad-line profiles with their half-maximum crossings. App, Results: "Compare with another fit" (Pin
  this fit, change the model, fit again, Compare with current; PDF).
- Fragile FWHM: `r.measure(line)["fwhm_fragile"]` is True when the profile has a shoulder near half maximum
  (width at 45% of the peak over the width at 55% above 1.3; a Gaussian gives 1.16), so that a small change of
  the fit moves the FWHM a lot; `fa.robustness` and `fa.compare_fits` flag it. Use sigma then.
- MCMC: chains that have not converged (R-hat >= 1.05 or ESS <= 100) are run once more with a 3x longer warm-up
  and twice the trajectory (`mcmc={"retry": False}` to skip); `r.mcmc_info["attempts"]`. On hard SDSS blends the
  default converged in 6 of 14 runs, the longer settings in 6 of 7. The warm-up uses at most 256 leapfrog steps per
  trajectory: a tiny early step size had cost up to 98% of all gradients (hours on one spectrum).
- App, Model step: a Continuum choice (power law or broken power law) for the presets that have one.
- Robustness of a fit: `rob = fa.robustness(r)` fixes each free width, shift and ratio on a grid across its
  limits, refits everything else (from the best fit and from a fresh guess) and records chi2.
  `rob.parameters` gives the 68% and 95% intervals and flags parameters that are only weakly bounded, can
  slide to a limit, have a second minimum or a too small error bar; `rob.quantities` gives the range of the
  results (broad line FWHM, sigma, flux and shift, R_FeII, total [OIII]5007) over all acceptable fits;
  `rob.alternatives` the other solutions; `rob.plot()` the scans. A better fit found by the scans becomes
  `rob.best` (`rob.best_model()` to fit it again). App, Results: "How robust is this fit?".
- Simpler models: `fa.model_confidence(r, ties, targets)` estimates, from one fit of a flexible model, the
  probability that ties move chosen results by less than a tolerance (safe, model-dependent or changes the
  result); `fa.candidate_ties(r)` lists the width and shift pairs to test, and `conf.refit()` confirms with a
  real fit.
- Saved models: `model.save("my_model.json")` and `fa.load_model(path)` keep a model (components, line
  profiles, starting values, limits, fixed and tied parameters) as a plain JSON recipe; loading only makes
  FANTASY components, so a model file from someone else cannot run code. App, Model step: Save model (name
  and folder; `~/.fantasy_agn/models` by default), Download, and "Start from: a saved model" with the saved
  models and models from a file.
- Line profiles: `r.plot_lines(["Hg", "Hb", "Ha"], residuals=True)` draws the chosen lines in a grid (one,
  two or three in a row, four as 2 x 2), each with its components, residuals, FWHM and flux, one legend and one
  flux label. App, Results: Line profiles view to tick lines and save the grid (PDF, PNG, EPS).
- Figure style: quiet grey axes with inward ticks on all sides, no grid, the model in ink over grey data, Fe II
  filled lightly, a +-1 sigma band in the residuals, and a palette whose six line colours stay distinct for
  colour-vision deficiencies; the app's plots match. `r.plot(styles={"feii": {"fill": False}, "broad":
  {"lw": 1.5, "fill": True}, "model": {"ls": "--"}})` sets fill, line width and line style per kind; the app's
  publication figure has these controls for every kind.
- App, step-by-step interface (feedback from users). Model step: the classification is a card at the top
  (Classify spectrum; the class, the reason and the recommended model; Use recommended model; View
  classification with the BPT diagrams, PNG and PDF) and stays available after the fit (Results: Fit /
  Classification switch). Start from a preset or from your own model. One "Add component" dialog for all
  components: emission and absorption lines picked from the line list (grouped by type, limited to the fitted
  range, searchable), kind (broad, narrow, outflow), profile (Gaussian, Lorentzian, Voigt, Gauss-Hermite,
  stratified), width and shift limits, and shift and width fitted, fitted per line or tied to another component;
  continua, Fe II, high-order Balmer lines, host galaxy, absorption troughs and dust extinction. The selected
  component's panel changes the line profile and ties or frees its shift and width (e.g. Fe II to the narrow
  lines). Fit step: MCMC errors. Results step: a summary of the fit quality and the main numbers with errors,
  a warning for parameters at a limit that opens the component in the Model step, views for the fit, the
  publication figure (ranges, title, components, width, font size, legend, line labels, redrawn as they
  change; PDF, PNG, EPS), the posterior (corner plot of the Monte Carlo or MCMC samples, parameters at a limit
  left out), the classification and, for several epochs, light curves and lags; the lines table grouped by
  component; one Download menu with the fitted lines, the model and its components (CSV of the shown fit:
  data, model, every component, host, AGN, spectrum minus host and, with extinction, the transmission and the
  intrinsic components), the fit figures, the script, the notebook and the project.
- Black-hole masses: `r.bh_mass(line, recipe=...)` with H-alpha (Greene & Ho 2005, default; Reines & Volonteri
  2015: FWHM and broad H-alpha luminosity), H-beta from Dalla Bonta et al. (2020) with the broad H-beta
  luminosity (`'db20'` FWHM, `'db20_sigma'` line dispersion) or L5100 (`'db20_l5100'`, `'db20_sigma_l5100'`) and
  the SDSS-RM recalibration of Shen et al. (2024) for H-beta, Mg II and C IV (`'shen24'`), besides the defaults
  (Vestergaard & Peterson 2006, Vestergaard & Osmer 2009) and Shen et al. (2011) for Mg II; `r.bh_masses()` gives
  every recipe that applies; `key_results` adds the H-alpha mass when the model has broad H-alpha and H-beta.
  App: the masses of every recipe in the Results step.
- Fe II in the line table: `r.lines` has a row per Fe II template with its flux and EW in 4434-4684 A (UV:
  2200-3090 A), width and shift; `r.feii()` also gives R_FeII = F(Fe II 4434-4684) / F(broad H-beta) (Mg II for
  the UV template), which is among the key results.
- Line library: the lines of the FANTASY 1 line lists that were missing (159; coronal and high-ionisation [Fe V],
  [Fe VI], [Fe VII], [Ar IV], [Ar V], [Ar XIV] ... lines, the [Fe II] forbidden multiplets 4F-34F, the narrow
  permitted Cr II, Ti II, N II, C III, Si II lines, the I Zw 1 Fe II lines, He I, O I, N I, Al III), and the old lists
  as sets: `fa.line_set("coronal")`, `fa.LINE_SETS`. App, Add lines: one click adds a whole list (lines in the
  fitted range).
- Reverberation: ZDCF lags (z-transformed discrete correlation function, Alexander 1997). `fantasy_agn.zdcf`
  is a port of pyZDCF (Jankov et al. 2022, MIT licence) that works on arrays and runs with NumPy 2 and pandas 3
  (pyZDCF requires numpy < 2 and pandas < 3); it reproduces pyZDCF to 1e-14 for the same order of equal lags
  (pyZDCF's unstable heapsort orders them differently in NumPy 1 and 2; the port sorts stably). `fa.rm.zdcf`,
  `fa.rm.zdcf_lag` (lag from the ZDCF centroid with flux-randomisation / random-subset errors) and
  `me.reverberation(..., methods=("iccf", "zdcf"))`; ZDCF option in the app.
- Publication figure: `r.plot(colors=..., hide=..., data=, model=, legend=, label_options=...)` and
  `label_lines="all"`; line labels sit in a row at the top, spread so they do not overlap and joined to their
  lines by grey lines (or dotted lines across the plot); font, size, direction and colour can be set. App:
  what is drawn and its colours, line-label options, and no title.
- UV Fe II model built as the optical one: `fa.FeII(kind="uv_extended")`, 2200-3500 A, 325 lines from the NIST
  Atomic Spectra Database grouped by the lower term of the transition (a6D, a4F, a4D, a4P, b4P, a4H, b4F, a4G,
  b2H), the a4D lines split into the multiplets UV 60-63 as in Kovacevic-Dojcinovic & Popovic (2015), relative
  intensities within a group from eq. (1) of Kovacevic et al. (2010) at 10^4 K, and the empirical I Zw 1 lines:
  14 group amplitudes, a common width and shift (data/uvfe2.csv, with the NIST query). It covers the UV Fe II band
  2200-3090 A, which the 2015 model (`kind="uv"`, 2690-3005 A, unchanged and still the default of
  `presets.mgii`) does not; `presets.mgii(feii="uv_extended", wave_range=(2200, 3300))` uses it. Not yet
  calibrated on real spectra.
- Quasar main sequence: `r.main_sequence()` (R_FeII, FWHM of broad H-beta, population A/B and spectral type,
  Sulentic et al. 2002, Marziani et al. 2018) and `fantasy_agn.plotting.plot_main_sequence(fits)` (every epoch,
  one highlighted). App, Results: a Main sequence view (all epochs, click to highlight; PNG and PDF).
- Classification of several epochs: 'Suggest from classification' classifies every epoch (the reference one
  recommends the model) and the BPT and WHAN diagrams show all of them (coloured by class, the chosen one
  outlined; `c.plot(others=...)` in Python). Where a classification has no narrow-line ratio (broad-line
  spectra), the narrow lines of the fit give the point. The diagrams widen to show every point and its error bars
  (they were cut at fixed limits).
- Colours: one palette for the figures and the app (broad blue, narrow green, outflow amber, Fe II magenta,
  host brown, Balmer sky blue, model vermilion, continuum near-black), checked for colour-vision deficiency;
  Fe II has its own colour in the app.
- GAMA spectra: the flux unit is read from the header (GAMA/AAOmega: 'Flux-calibrated spectrum in 10^-17
  erg/s/cm^2/A', was taken as 1, so luminosities and black-hole masses were wrong), the date from UTMJD (or MJD,
  DATE-OBS), the spectrum name and GAMA IDs (SPECID, CATAID, GAMANAME, NQ) into `meta`; GAMA files are also
  recognised by ORIGIN = 'GAMA', CATAID or GAMANAME. App, Read step: a File format choice (automatic, SDSS, DESI,
  OzDES, GAMA, 1D FITS, text) for files that are read wrongly, and GAMA and OzDES listed among the formats.
- Flux units of text and 1-D FITS files: the unit is read from a header or comment line ("flux in 1e-16
  erg/s/cm2/A", "F_lambda [10^-17 erg s^-1 cm^-2 A^-1]", "x10⁻¹⁶") and from the BUNIT keyword; values in erg/s/cm2/A
  are stored in units of 1e-17 as before. When the file gives no unit and the values are not physical, `flux_unit`
  is now None (was 1): luminosities and black-hole masses are refused instead of computed from a wrong unit, and
  `luminosity` also refuses a unit that makes the source implausibly bright (> 1e-10 erg/s/cm2/A).
  `Spectrum.with_flux_unit(unit)` sets it (`fa.read(path, flux_unit=1e-16)` too); `fa.spectrum.flux_unit_from_text`
  parses one. App, Prepare step: a Flux unit field shows the unit from the file and sets it when it is missing.
- App, Fit step: what will be fitted (with links to change it), choices in plain words for fitting the epochs
  together or on their own and for the uncertainties (with how long each takes), the rarely needed options under
  Advanced options, a run button that says what it does, and when the fit is done its quality, the parameters at a
  limit (once each, with the epochs) and links to the results and the Model step.
- `Model.set_profile(name, profile)` changes the line profile of a `Lines` component in place, keeping its lines,
  flux ratios, role, and the values, limits and ties of the parameters both profiles have; `Model.untie(pattern)`
  frees tied parameters.
- Broad Balmer decrement in the `hbeta` and `optical` presets (`balmer_decrement=True`, default): in every
  broad component H-gamma is a ratio to H-beta, `broad.Hg.ratio`, free between 0.1 and 0.5 (case B 0.47), and
  broad H-delta is tied to it by the case-B ratio H-delta/H-gamma = 0.553 (`broad.Hd.ratio`); `hbeta` adds broad
  H-delta when the window starts below 4250 A (its wing enters the default 4150-5550 A window). Before, the
  broad H-gamma flux was free: in the multi-epoch fits of the nuclear transient PS1-10adi the second, very
  broad component made H-gamma/H-beta 0.5-1.6 (up to 4.7 in one component), a broad hump at 4250-4450 A that
  took the H-delta wing, FeII and continuum. `balmer_decrement=False` restores the free fluxes. H-beta keeps
  its `broad.Hb.flux` parameter; `broad.Hg.flux` is replaced by `broad.Hg.ratio` (use
  `result.measure("Hg")` for the flux). New `Lines(..., line_ratios={line: (reference, (start, min, max))})`
  makes any flux ratio a free parameter within limits.
- Line library reorganised and extended from 74 to 170 lines: one table, `fantasy_agn/data/lines.csv`, gives
  for every line its wavelength with the medium it refers to (air or vacuum), ion, type (recombination,
  permitted, semi-forbidden, forbidden, coronal, Bowen, molecular, dust feature), series or multiplet, and
  notes on blends. New: the far-UV lines (Ly-beta, O VI, Si II, O I 1304, C II, O IV], N IV], N III],
  Si III], C II], [O II] 2471), He I/He II and Bowen O III lines in the near-UV and optical, [Ar IV], [Fe III],
  [Fe VI], [Ca V], [Cl III], [Ca II], [Ni II], forbidden [Fe II], Na I D, O I 7774 and 11287, the complete
  Paschen series (Pa-alpha to Pa14), Brackett (Br-alpha to Br13) and Pfund (Pf-beta to Pf-epsilon) lines,
  He I 1.083 and 2.059 um, near-infrared [Fe II], [S II], [C I], [P II] and coronal lines, H2 rovibrational
  lines and the 3.3 um PAH band, covering the SPHEREx range (to 5 um). `fa.line_table()` returns the table,
  `fa.lines_in(wmin, wmax, type=..., ion=..., series=...)` selects lines, `fa.line_info(name)` describes one;
  names such as 'Pa-alpha', 'Brgamma' are accepted. Existing line names are unchanged. The Fe II templates are
  unchanged.
- Faster Monte Carlo errors: `errors="mc"` uses all cores by default (one inside the workers of `fit_many`
  and `fit_sample`), and the re-fits stop at a tolerance of 1e-8 instead of 1e-10 (1.3x fewer model
  evaluations, the same errors on SDSS-RM stratified fits). 100 realizations of a stratified H-beta fit:
  73 s -> 24 s on a busy 10-core machine. Covariance errors stay the default; for line widths, sigma and the
  continuum of degenerate models they differed from Monte Carlo by up to 3x in either direction.
- App, Fit step (joint fit): "Shared broad-line shape (only fluxes vary)", on by default. The shape parameters
  of every broad component (shift, widths, stratified span and gamma, Gauss-Hermite h3/h4, ...) and the Fe II width
  are shared between the epochs on top of the narrow lines and the host. With free shapes per epoch the line flux
  trades with the continuum and Fe II at SDSS S/N: in SDSS-RM quasars the broad-H-beta light curves then recovered
  the published reverberation lag in only 24 of 81 objects, with shared shapes in all 8 objects tested.
- App: a new step-by-step interface (Read, Prepare, Model, Fit, Results) for one spectrum, a sample,
  several epochs of one object or classification only. Read: files, path/glob, DESI TARGETIDs or SDSS
  plate-mjd-fiber (`fa.sdss.fetch`), epochs ordered by the date in the file names
  (`fa.spectrum.mjd_from_name`) or the MJD; Prepare: redshift, Galactic extinction from the position or by
  hand, range, telluric/sky/absorption masks and ranges dragged on the plot, with a raw/prepared preview;
  Model: preset and options, component list, initial guess on the reference epoch, parameter table
  (initial, limits, fixed, tie); Fit: independent or joint epochs (reference, shared parameters, calibration
  degree, host amplitude per epoch, error rescaling), progress, chi2 and parameters at limits; Results:
  interactive fit with components, residuals and line labels per epoch, line table with flags, continuum,
  all epochs, parameters, downloads (CSV, FITS, PDF, script, notebook). The Python session panel mirrors
  every step and is hidden by default. The earlier interface is at `/classic`.
- Residual analysis: `fa.analyze_residuals(r)` (also a joint fit or a list of fits) finds coherent residual
  features on several scales (relative to the scatter of the residuals), measures them (centre, FWHM, flux,
  significance), identifies them with the line library and suggests a change with its code: a missing line
  (doublets together), a broad component or wing, an outflow for forbidden-line wings, the host for stellar
  absorption, Fe II or the Balmer continuum for broad excesses, masks for telluric/sky residuals (not on a fitted
  line) and absorption troughs, continuum-shape mismatches. For several epochs the features are merged and
  counted. App: "Residual analysis" on the Results step, features marked on the plot, Apply buttons and refit.
- App, multi-epoch results: light curves (broad flux, FWHM, EW, continuum per epoch; click a point to open the
  epoch) and reverberation mapping (line, lag methods, Monte Carlo, MJD range, detrending: lags of every method,
  rms width, mass, aliasing warnings, figure). The result view loads the plot and line table first and the slow
  tables (main numbers with errors, all epochs) afterwards; the light curve is computed once, in the background
  (it blocked the page for many minutes with 12 epochs and stratified profiles). Projects keep the workflow state.
- Faster error propagation: `measure`, `bh_mass`, `balmer_decrement`, `continuum`, `luminosity` and the
  line-continuum correlation of the light curves only differentiate with respect to the parameters they depend on
  (`propagate(..., depends=)`); same errors, about 2x faster on a 141-parameter model, and the light curve is cached.
- Plots: a broad component of Bowen fluorescence and He II lines only (N III 4640, O III 3760, He II 4686/5412,
  as in the TDE and BFF presets) is drawn in its own colour as "Bowen + He II" instead of being summed with the
  broad Balmer lines (also in the app).
- `fa.guess`: amplitudes of lines without fitted pixels (in masked bands or outside the range) are 0, as
  in `fit` (they gave huge values in the initial guess).
- Nuclear transients: `fa.Blackbody` continuum (amplitude at a pivot, temperature); Bowen lines OIII3760,
  NIII4100, NIII4640 and HeII5412 in the line library; presets `fa.presets.tde()` (blackbody + host, broad Balmer,
  He II + Bowen with shared kinematics, He I, narrow host lines; options for Fe II, coronal lines and broad
  profiles) and `fa.presets.bff()` (Bowen fluorescence flares / extreme coronal line emitters, Baldini et al.
  2025). App: both in the model menu. Text files with a constant error column (a placeholder) get errors from
  the local scatter.
- Text spectra: columns separated by several spaces are read correctly (they produced an empty flux column);
  fluxes in physical units (|F| < 1e-10) are stored in units of 1e-17 (`flux_unit=1e-17`, switch off with an
  explicit `flux_unit=`); without an error column the error is estimated from the local scatter (running window)
  instead of one value for the whole spectrum; header lines are detected with any pandas version.
- Stratified BLR line profile (`profile="stratified"`, presets: `broad_profile="stratified"`): a virialised,
  radially stratified broad-line region (Gaussians of FWHM ∝ r^-1/2 summed over r_in..r_out, dL/dlnr ∝ r^(3-γ));
  parameters outer FWHM, span = log(r_out/r_in), γ; analytic derivatives. A span at its upper limit is reported
  as a lower limit (inner edge not constrained). App: "stratified BLR profile" tick box in the Model step and
  the Multi-epoch view (one broad component).
- Coordinates when reading: `fa.read(path, ra=..., dec=...)` (degrees or sexagesimal, or `coords="12:34:56.7
  +12:34:56"` / a SkyCoord) for every format, and `spec.with_coordinates(...)`, so that the Galactic E(B-V) can be
  looked up for files without a position (text files). App: RA and Dec fields in the Read step.
- Wavelength-dependent intercalibration: `fit_epochs(..., calibration_degree=1|2)` fits a smooth
  Legendre correction per epoch (scale x (1 + sum c_j P_j(lambda))) anchored by the shared narrow lines
  and host, beyond the single scale per epoch of PrepSpec. Simulated 6% tilts per epoch: recovered to
  1.5%, broad-line flux errors 3.9% -> 1.3%.
- Broad lines have physical limits: no component wider than 15000 km/s (a 30000 km/s Gaussian spans the
  whole Hb-[OIII] window and trades flux with the continuum). With two components the core is
  1000-9000 km/s and the very broad component 3000-15000 km/s, shifts within +-3000 km/s (Sulentic et
  al. 2000; Marziani et al. 2009). On 120 calibration spectra the fits were unchanged (median dBIC 0,
  chi2/nu +0.001).
- High-S/N narrow-line galaxies: `fa.refine(spec, preset)` adds the extensions of the preset that the BIC
  prefers (flux-calibration polynomial, outflow wing, Gauss-Hermite profiles, broad Balmer lines in a
  LINER), rejecting any whose parameters end at a limit; `spec.mask_sky()` masks telluric bands and sky
  lines; new `fa.Calibration` (multiplicative Legendre polynomial, as pPXF mdegree) and the preset options
  `calibration=` and `narrow_profile=`. `fit_sample` and the app's Fit step use them for the
  narrow-line classes. Eight SDSS examples: chi2/nu 3.4 -> 2.0, 2.2 -> 1.8, 2.1 -> 1.6, 1.6 -> 1.1, 2.1 -> 1.3.
  `starforming`: H8 (blended with HeI 3889) is no longer tied to Case B.
- Lag methods: von Neumann estimator (`fa.rm.vn_lag`), damped-random-walk lags with the correlated line
  and continuum errors of each fit (`fa.rm.gp_lag`), seasonal-alias weighting for all methods; 
  `me.reverberation(methods=, primary=)`, `rm.lag_table()`, warnings when the methods disagree.
- Spectro-temporal reverberation: `fa.rm.component_reverberation(me, ...)` gives the lag and response of
  every spectral component (continuum, each broad component, FeII) from their fluxes and full covariance
  over all epochs (`me.quantity_covariance`), and decomposition-cleaned light curves.
- Multi-epoch joint fits are much faster for large models: the normal equations are assembled per
  epoch and solved block by block (Schur complement on the shared parameters), the covariance is
  kept per epoch, host template grids are shared between epochs and the Balmer-continuum
  broadening uses an FFT. 76 SDSS-RM epochs with `optical(host=True)`: joint fit 205 s (it ran
  for more than 30 minutes before). The stellar host is shared between epochs by default (its
  starlight does not vary); `shared=` overrides.
- `fit_epochs` warns when the intercalibration is uncertain (median flux-scale error > 2% per epoch;
  `me.calibration_error`). On SDSS-RM RMID 766 (weak [OIII]) the narrow-line intercalibration added
  ~3% noise and damped the Hb variability; without it the light curve matched the SDSS-RM one best
  (r = 0.77). The ICCF is vectorised (6x faster).
- Absorption: `fa.Absorption` (Gaussian optical-depth troughs multiplying the model) and
  `fa.fit_absorbed(spec, model)`, which finds troughs in the residuals and adds them while the BIC
  improves (`r.absorbers`, `r.no_absorbers`). The worst CIV calibration fits went from reduced chi2
  8-24 to 1.0-1.9. Models now accept several multiplicative components (absorption and extinction).
  App: **Fit absorption troughs** in the Fit step; single-line figures show the absorbed profile.
- Reverberation mapping: `me.reverberation("Hb")` gives the lag (ICCF with FR/RSS errors, search
  window from the R-L relation), sigma_line and FWHM of the rms spectrum and the reverberation
  mass (f = 4.47); `continuum=(mjd, flux, err)` for photometry, `mjd_range`, `detrend`;
  `me.rms_spectrum()`, `fa.rm.ccf_lag()`. App: **Lag, rms spectrum and mass** in the Multi-epoch view.
- Samples: `fa.fit_sample(files, out=...)` reads, dereddens, classifies and fits many spectra in
  parallel (model from the classification, host check included) and returns one catalog
  (`.table`), the saved fits (`.load(i)`) and a gallery. App: **Sample** view with a sortable
  catalog, the fit and main line of every spectrum, and the catalog as CSV.
- Narrow-line diagnostics, `r.diagnostics()`, with errors: BPT ratios and class, narrow Ha/Hb and
  gas E(B-V), [SII] electron density, [OIII]/[NII] temperatures, EW(Ha) for WHAN and the O3N2/N2
  metallicity of star-forming galaxies. `key_results` of narrow-line fits now has errors for the
  [OIII] flux and the BPT ratios (they were missing) and includes E(B-V), n_e and T_e; the app's
  Data tab lists all diagnostics.
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
- The macOS and Linux installer now accepts bundle folders with spaces in their names, including numbered copies created by Finder.
- `errors="mcmc"` froze parameters at a limit, so undetected lines had no uncertainty. All free parameters are now
  sampled, `r.upper_limit(name, level=0.95)` gives one-sided limits, the whitening is relearned during the warm-up,
  and the chains run in parallel (`n_jobs`). Samples do not depend on `n_jobs`, but differ from earlier versions.
- Variable projection stopped early and reported success when amplitude columns were repeated: the projection
  is now rank-aware.
- A broad line with a masked centre could be removed as unconstrained: amplitudes are now tested at the widest
  widths and at the shift limits as well as at the neutral values.
- `errors="mc"` warns when refits did not converge.
- Widths and shifts of components without flux no longer look like measurements (they came with errors of
  ~1e-14): they are listed in `r.not_measured`, have status "not measured", no error, and are NaN in `r.lines`.
- Line fluxes with an error at least as large as the flux are listed in `r.unmeasured_fluxes`, have status
  "not measured", and are warned about in the summary and flagged in the app.
- Classification: Seyfert 1s with broad lines of ~2500-3500 km/s could be classified as 'unknown' with
  the type-2 model recommended. The Balmer part of the outflow wing (FWHM up to 2500 km/s) took the broad
  core, the broad component kept only the far wings (FWHM > 12000 km/s, no H-beta) and failed the broad-line
  test. `classify` now also fits the lines with the wing on the forbidden lines only and a broad core plus
  a very broad wing, keeps the better fit (BIC), and tests the whole broad profile. On the calibration
  sample (SDSS): type 1 quasars recognised 228 -> 238 of 300 (none lost), narrow-line galaxies called
  type 1 6 -> 9 of 650 (the three new ones are MPA-JHU Seyfert 2s with broad H-alpha of 2000-3000 km/s).
- Line wavelengths: the Mg II 2798 blend and [Ne IV] 2424 are vacuum values (Vanden Berk et al. 2001) but
  were converted from air to vacuum a second time, which put them 88 and 91 km/s too far to the red: Mg II
  velocity shifts measured with `MgII2798` (the `mgii` preset) came out 88 km/s too blue. Widths, fluxes and
  black-hole masses are unaffected. [S III] 9531 corrected from 9531.1 to 9530.6 A (air).
- Multi-epoch: per-epoch results did not report parameters at a limit (the names carried the
  epoch suffix), so e.g. a broad width at its maximum showed as "free" with error 0 and no warning;
  amplitudes at zero were not reported either.
- Multi-epoch: a shared parameter tied to a per-epoch one (e.g. feii.shift, tied to broad.shift
  by the hbeta preset) stopped the fit; it is now fitted freely, with a warning.
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
- The Jupyter widget interface (`fa.gui()`, the ipywidgets extra) and the app's classic page (`/classic`) are
  removed; the app (`fa.app()`, `fantasy-app`) does both: reverberation mapping is in Results, Light curves &
  lags, single lines in Results, Line profiles; the comparison of decompositions is the Fit step's Model
  uncertainty option and `fa.compare`.

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
