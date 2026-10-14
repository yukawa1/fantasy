API reference
=============

Everything below is available directly as ``fantasy_agn.<name>``.

Spectra
-------

.. autofunction:: fantasy_agn.read

.. automodule:: fantasy_agn.desi
   :members: targets, read, read_all, from_sparcl, find, fetch

.. autoclass:: fantasy_agn.Spectrum
   :members: to_rest_frame, deredden, crop, mask, to_air, subtract_host, plot, good, copy

Fitting
-------

.. autofunction:: fantasy_agn.fit

.. autofunction:: fantasy_agn.guess

.. autofunction:: fantasy_agn.fit_epochs

.. autofunction:: fantasy_agn.compare

.. autoclass:: fantasy_agn.Comparison
   :members: summary

.. autoclass:: fantasy_agn.FitResult
   :members: params, lines, measure, continuum, luminosity, bh_mass, host_properties, propagate, plot, savefig, summary, save, to_dataframe, model_flux, bic, aic

.. autoclass:: fantasy_agn.MultiEpochResult
   :members: shared, scales, light_curve, plot_light_curves, plot_baldwin, plot, summary

Ready-made models
-----------------

.. automodule:: fantasy_agn.presets
   :members: hbeta, halpha, optical, mgii, civ, available, auto, variants

Building models
---------------

.. autoclass:: fantasy_agn.Model
   :members: add, remove, set, fix, free, tie, evaluate, table, copy

.. autoclass:: fantasy_agn.Lines
.. autoclass:: fantasy_agn.FeII
.. autoclass:: fantasy_agn.HighOrderBalmer
.. autoclass:: fantasy_agn.PowerLaw
.. autoclass:: fantasy_agn.BrokenPowerLaw
.. autoclass:: fantasy_agn.Polynomial
.. autoclass:: fantasy_agn.BalmerContinuum
.. autoclass:: fantasy_agn.HostGalaxy
.. autoclass:: fantasy_agn.StellarPopulation
.. autoclass:: fantasy_agn.Extinction
.. autoclass:: fantasy_agn.Component
   :members: evaluate, evaluate_with_derivatives

.. automodule:: fantasy_agn.templates
   :members: download, load, available, TEMPLATE_SETS

.. autofunction:: fantasy_agn.line_wavelength
.. autofunction:: fantasy_agn.read_line_list

GUI
---

.. autofunction:: fantasy_agn.gui
