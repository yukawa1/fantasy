"""
FANTASY - Fully Automated pythoN tool for AGN Spectra analYsis.

Quick start::

    import fantasy_agn as fa

    spec = fa.read("spec-1059-52618-0453.fits").deredden(ebv=0.03).to_rest_frame()
    result = fa.fit(spec, fa.presets.hbeta())
    result.plot()
    result.lines             # fluxes, widths, shifts, EWs with errors
    result.measure("Hb")     # total broad H-beta: flux, FWHM, sigma, ...
    result.bh_mass()

In a Jupyter notebook, ``fa.gui()`` opens a point-and-click interface.
"""
__version__ = "2.0.0"

from . import desi, presets, templates
from .host import StellarPopulation
from .components import (BalmerContinuum, BrokenPowerLaw, Component, Extinction, FeII,
                         HighOrderBalmer, HostGalaxy, Lines, Polynomial, PowerLaw)
from .extinction import fitzpatrick99
from .compare import Comparison, compare
from .fitting import fit, guess
from .gui import gui
from .lines import LINES, line_wavelength, read_line_list
from .model import Model
from .multiepoch import MultiEpochResult, fit_epochs
from .parameters import Parameter, ParameterSet
from .result import FitResult
from .spectrum import Spectrum, read

__all__ = [
    "read", "Spectrum", "Model", "fit", "guess", "fit_epochs", "compare", "Comparison",
    "presets", "gui",
    "FitResult", "MultiEpochResult", "Lines", "FeII", "HighOrderBalmer", "PowerLaw",
    "BrokenPowerLaw", "Polynomial", "BalmerContinuum", "HostGalaxy", "StellarPopulation",
    "Extinction",
    "templates", "desi", "Component",
    "Parameter", "ParameterSet", "LINES", "line_wavelength", "read_line_list",
    "fitzpatrick99",
]
