*************
Uncertainties
*************

``fa.fit(..., errors=...)`` offers two methods.

``"covariance"`` (default)
    The covariance matrix (J\ :sup:`T` J)\ :sup:`-1` from the exact Jacobian at the
    best fit. It is fast and accurate for well-measured quantities.

``"mc"``
    The spectrum is perturbed with its error spectrum and refitted ``n_mc`` times,
    starting from the best fit. Errors are half the 16-84 percentile range. It is more
    reliable for weak or blended features and for strongly non-linear quantities.

``"mcmc"``
    Posterior samples by Hamiltonian Monte Carlo (``fantasy_agn.sampling``): flat priors
    within the parameter limits and likelihood exp(−χ²/2). The exact gradient of χ² comes
    from the analytic Jacobian, and the fit covariance whitens the parameters, so short
    trajectories give nearly independent samples. Trajectories that reach a parameter
    limit are reflected there. The step size is tuned during warm-up (dual averaging)
    for an acceptance rate of 0.8; four chains are run and ``result.mcmc_info`` reports
    split-R̂ and the effective sample size. It makes no assumption of a Gaussian
    posterior and takes about a minute for a 30-parameter model. Options:
    ``mcmc={"n_chains": 4, "n_warmup": 300, "n_samples": 600, "trajectory": 1.6, "retry": True}``.
    Chains that have not converged (R̂ ≥ 1.05 or an effective sample size ≤ 100) are run
    once more with a 3× longer warm-up and twice the trajectory; ``result.mcmc_info["attempts"]``
    says how many runs were needed. If they still do not converge, the posterior usually has
    several modes or a long valley: ``fa.robustness`` shows which parameters.

In all cases:

* if the reduced χ² exceeds 1, errors are multiplied by its square root
  (``scale_errors=True``), assuming the error spectrum is underestimated;
* errors of derived quantities (``measure``, ``continuum``, ``luminosity``,
  ``bh_mass``, EWs) are propagated from the covariance matrix, or computed from the
  Monte Carlo samples;
* parameters that end at a limit get no error and are listed in the summary. Widths or
  shifts at a limit usually mean the limits are too tight; an amplitude at zero means
  the component is not detected.

Systematic errors from the choice of model
==========================================

The statistical errors assume the model is right. Broad-line widths in particular depend
much more on how a line is decomposed (number of broad components, outflow wings) than on
the noise; on a DESI quasar, the broad Hα FWHM ranged from 3600 to 6900 km/s over six
reasonable decompositions while each fit quoted ±30 km/s. ``fa.compare`` fits several
models and adds a systematic error::

    cmp = fa.compare(spec, fa.presets.variants("hbeta"))
    cmp.summary()

The quoted value is that of the model with the lowest BIC. The systematic error is the
standard deviation over the *acceptable* models, those whose reduced χ² is within 10% of
the best (``tolerance=``). This is a heuristic: with high signal-to-noise data no simple
model fits perfectly, and formal BIC weights then give all the weight to one model and
hide the model dependence. If only one model is acceptable the systematic error is NaN.

Is the fit robust?
==================

Covariance errors describe the χ² surface next to the best fit. They miss what lies
further away: a width that can slide to its limit at a small χ² cost, a second minimum,
or a better fit the optimizer did not find. ``fa.robustness`` looks there::

    rob = fa.robustness(r)
    rob.summary()
    rob.parameters      # 68% and 95% intervals and a flag for every width, shift and ratio
    rob.quantities      # broad-line FWHM, sigma, flux and shift, R_FeII, total [OIII]: ranges
    rob.alternatives    # other solutions (second minima up to Δχ² = 9)
    rob.plot()          # the Δχ² scans

Every free parameter that is not an amplitude is fixed on a grid across its limits and
everything else is refitted, from the best fit and from a fresh initial guess (the lower
χ² is kept); the intervals are where Δχ² = 1 and 4 in units of the scaled errors. Flags:
*weakly bounded* (Δχ² below 16 at a limit), *not excluded up to the limit*, *best at its
limit*, *second minimum* or *two solutions*, *error bar too small*. If a scan finds a
better fit than the original one, it becomes ``rob.best`` (``rob.best_model()`` to fit it
again with errors).

Each result is also scanned directly: it is held at values around its best one (a stiff
penalty) while every free parameter is refitted, and its range is where Δχ² = 1 and 4.
Scanning only the parameters is not enough: it rarely moves a result such as R\ :sub:`FeII`
in its own most sensitive direction. In 80 spectra simulated from two fits of an SDSS quasar
(narrow and broad [O III] outflow, Hβ preset; 8 results each), ranges from the parameter
scans alone held the truth in 40% of the cases at 68% and 70% at 95%; with the direct scans
of the results, 65% and 96% (the covariance errors: 70% and 95%). The parameter intervals
from the scans held it in 57–94% at 68% and 88–100% at 95%; for the weakly bounded [O III]
outflow width, the ±2σ covariance error held it in 75% of the cases and the 95% scan
interval in 94%. A broad FWHM is flagged *fragile* when the profile has a
shoulder near half maximum (``r.measure(...)["fwhm_fragile"]``): a small change of the fit
then moves the FWHM a lot, and σ is the safer width.

It takes a few hundred refits, 1–5 minutes on 8 cores for the Hβ preset.

Comparing two fits
==================

Changing the continuum, the line profile or a tie can move a result far more than its
error while the total model hardly changes, because the components trade flux where they
overlap. ``fa.compare_fits`` shows it::

    a = fa.fit(spec, fa.presets.hbeta())
    b = fa.fit(spec, fa.presets.hbeta(continuum="broken"))
    cmp = fa.compare_fits(a, b, labels=("power law", "broken power law"))
    cmp.summary()       # chi2 on the same pixels, RMS of (b - a) / error
    cmp.table           # the results of both and the change in units of the errors
    cmp.plot()          # both fits, where the flux went, both broad profiles

On the SDSS quasar spec-1744-53055-0630 the two total models differ by 0.04 of the error
(RMS) while the broad Hβ FWHM changes by 12–17%: the data cannot tell these fits apart.
In the app: Results, *Compare with another fit* (Pin this fit, change the model, fit
again, Compare with current).

Validation
==========

The test suite and the development simulations refit spectra simulated from a known
model (an SDSS quasar fitted with ``presets.hbeta()``) and compare the pulls
(fit − truth)/error with a unit Gaussian.

* Covariance errors: pull standard deviations of 0.8–1.1 for continuum, broad and
  narrow line fluxes and widths, FeII, and the total broad Hβ FWHM and σ from
  ``measure``. The exception is the weak HeII 4686 line, blended with FeII and the Hβ
  wing, whose covariance errors were 2× too small.
* Monte Carlo errors: pull standard deviations of 0.7–1.07, including HeII 4686
  (0.94). For weak or blended lines, use ``errors="mc"``.
* Parameters whose true value lies on a physical bound (e.g. FeII multiplets with zero
  flux) bias nearby quantities slightly, because noise can only push them one way. In
  the simulations the total broad Hβ flux was biased by about +1.5%, or 0.5σ.

Is the result biased?
=====================

Error bars say how much a result scatters, not whether the fit returns the
right value on average. ``fa.simulation_check`` takes the fitted model as the
truth, draws spectra from it with the noise of the data and fits them again as
the spectrum was fitted::

    sim = fa.simulation_check(r, n=60)
    sim.table      # value, mean of the simulations, bias, scatter, error bar, coverage, flag
    sim.biased     # the results flagged as biased

A model with many weakly constrained, bounded amplitudes can be biased: weak
Fe II multiplets cannot go negative, so the summed Fe II comes out too high and
takes flux from the continuum and the neighbouring broad line.
``fa.prune_feii(r)`` fits again without the multiplets the data do not detect;
run the check on both fits to see which is better for your object. The check
tests the fitting procedure for this model at this signal-to-noise ratio; it
cannot see what the model leaves out.
