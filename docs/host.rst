**********************************
Host galaxy and stellar kinematics
**********************************

``fa.StellarPopulation`` (or ``host=True`` in the presets) models the host galaxy as a
non-negative combination of simple-stellar-population templates, shifted by the stellar
velocity v\ :sub:`*` and broadened by the stellar velocity dispersion σ\ :sub:`*`.

Templates
=========

By default it uses the E-MILES models (Vazdekis et al. 2016; Padova isochrones, Salpeter
IMF) as distributed for pPXF by M. Cappellari. They are not bundled with FANTASY::

    fa.templates.download("emiles")   # 13 MB, once, into ~/.fantasy_agn/templates

The default grid has 13 ages (0.06–15.8 Gyr) and three metallicities ([M/H] = −0.71, 0,
+0.22). Choose others with ``fa.StellarPopulation(ages=[...], metals=[...])``.

How it works
============

* The templates (air wavelengths) are converted to the medium of the spectrum and put on a
  logarithmic wavelength grid, where a velocity shift and a Gaussian broadening are the
  same at every wavelength.
* They are broadened once to a common resolution: that of the data, or the coarsest
  template resolution in the fitted range if that is larger. σ\ :sub:`*` is then unbiased
  at all wavelengths as long as it exceeds ``sigma_floor`` (about 40–50 km/s for E-MILES
  on SDSS spectra), which ``result.host_properties()`` reports.
* Each trial (v\ :sub:`*`, σ\ :sub:`*`) is applied in Fourier space with the analytic kernel
  exp(−2πifv − 2π²f²σ²); the derivatives with respect to v\ :sub:`*` and σ\ :sub:`*` follow
  exactly from the same kernel.
* The template weights are non-negative and solved together with all other amplitudes, so
  the host cannot become negative.

Validation
==========

* Synthetic host + AGN spectra (independent truth made by direct convolution, S/N ≈ 30,
  σ\ :sub:`*` = 180 km/s): σ\ :sub:`*`, v\ :sub:`*` and the host fraction were recovered
  without bias, with error-normalised scatter of 0.7–0.9.
* SDSS spec-0461-51910-0361 (host-dominated AGN): σ\ :sub:`*` = 114 ± 5 km/s from the
  Hβ window and 109 ± 15 km/s from the Hα window.
* SDSS spec-2770-54510-0433 (luminous quasar): host fraction 2.4 ± 1.8 % at 5100 Å, where
  the PCA eigenspectra gave a host that was negative at every pixel.

Warnings
========

``host_properties()`` warns when

* σ\ :sub:`*` or v\ :sub:`*` end at their limits: no stellar absorption features were
  detected and the "host" only describes a continuum shape (common for quasars and in the
  narrow Hα window), so fit without a host;
* σ\ :sub:`*` is close to the resolution floor;
* the instrumental resolution of the spectrum is unknown (``spectrum.resolution = 0``), so
  σ\ :sub:`*` includes the instrumental broadening. Set ``spectrum.resolution`` (FWHM in
  km/s) for such data.
