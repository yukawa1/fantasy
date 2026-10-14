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
    ``mcmc={"n_chains": 4, "n_warmup": 300, "n_samples": 600}``.

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
