.. hide the title (based on astropy)
.. raw:: html

   <style media="screen" type="text/css">h1 { display: none; }</style>

FANTASY - Fully Automated pythoN tool for AGN Spectra analYsis
==============================================================

.. image:: fantasy.png
   :width: 500px

*Nemanja Rakić, Dragana Ilić and Luka Č. Popović*

With the growing number of spectral surveys and monitoring campaigns (e.g. reverberation mapping) of active galactic nuclei (AGN), there is an indispensable need for increasingly innovative and independent tools for AGN spectra analysis and measurement of spectral parameters. Especially in the case of AGN, whose UV, optical, and NIR spectra can be very complex with numerous blended and satellite emission lines (e.g., contamination of strong Fe II emission to the Hβ line), there is a strong desire for complex and flexible spectral fitting tools that would enable the reliable extraction of spectral parameters, such as the flux of pure AGN continuum or emission line fluxes and widths of heavily blended lines.

We present here the open-source code FANTASY - Fully Automated pythoN tool for AGN Spectra analYsis, a Python-based code for simultaneous multi-component fitting of AGN spectra, optimized for the optical rest-frame band (3600-8000 Å), but also applicable to the UV range (2000-3600 Å) and NIR range (8000-11000 Å). The code fits simultaneously the underlying broken power-law continuum and sets of defined emission line lists. There is an option to import and fit together the Fe II model, to fully account for the rich emission of complex Fe II ion which can produce thousands of line transitions, both in the range of Hg and Hb, but also Ha line (Ilic et al., 2022). The model is based on the atomic parameters of Fe II and builds upon the models presented in Kovacevic et al. (2010), Shapovalova et al. (2012), and assumes that different Fe II lines are grouped according to the same lower energy level in the transition with line ratios connected through the line transition oscillatory strengths.

Before performing the fittings, the FANTASY code provides several preprocessing options to prepare the spectrum, namely:

- Galactic extinction correction
- Redshift correction
- Host-galaxy fitting and subtraction

The code is flexible in the selection of different groups of lines, either from predefined lists (e.g., standard narrow lines, Hydrogen lines, Helium lines, other broad lines, forbidden Fe II lines, coronal lines, etc.), but it also gives full flexibility to the user to merge predefined line lists, create a customized line list, etc. The code can be used for fitting a single spectrum as well as for a sample of spectra, which is illustrated in several given examples.

It is an open-source, easy-to-use code, available for installation through the pip install option. It is based on the sherpa Python package (Burke et al., 2019).

If you are using this code in your research, please cite the following papers:

- `Ilic et al. (2020) <https://ui.adsabs.harvard.edu/abs/2020A%26A...638A..13I/abstract>`_
- `Rakic (2022) <https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.1624R/abstract>`
- Ilic, D., Rakic, N., Popovic, L. C., 2023, ApJS, accepted

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   Tutorial
   Tutorial_mcmc
   fantasy_agn

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
