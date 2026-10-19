"""
FANTASY - Fully Automated pythoN tool for AGN Spectra analYsis.

Quick start::

    import fantasy_agn as fa

    spec = fa.read("spec-1059-52618-0453.fits").deredden(ebv=0.03).to_rest_frame()
    result = fa.fit(spec, fa.presets.hbeta())
    result.plot()
    result.lines             # fluxes, widths, shifts, EWs with errors
    result.measure("Hb", ["broad", "broad2"])   # broad H-beta: flux, FWHM, sigma, ...
    result.bh_mass()

The application ``fantasy-app`` (or ``fa.app()``) opens a point-and-click
interface in the browser; in a Jupyter notebook ``fa.gui()`` shows it as widgets.
"""
__version__ = "2.0.0"

from . import desi, examples, ozdes, presets, templates
from .host import StellarPopulation
from .components import (BalmerContinuum, BrokenPowerLaw, Component, Extinction, FeII,
                         HighOrderBalmer, HostGalaxy, Lines, Polynomial, PowerLaw)
from .extinction import fitzpatrick99
from .classify import Classification, classify
from .compare import Comparison, compare
from .fitting import fit, fit_many, guess
from .gui import gui


def app(port=8780, open_browser=True):
    """Start the FANTASY application in the browser (see :mod:`fantasy_agn.app`)."""
    from .app import run
    run(port=port, open_browser=open_browser)
from .lines import LINES, line_wavelength, read_line_list
from .model import Model
from .multiepoch import MultiEpochResult, fit_epochs
from .parameters import Parameter, ParameterSet
from .result import FitResult
from .spectrum import Spectrum, read
from .io import save, load

__all__ = [
    "read", "Spectrum", "Model", "save", "load", "fit", "fit_many", "guess", "fit_epochs", "compare", "Comparison",
    "classify", "Classification",
    "presets", "gui", "app",
    "FitResult", "MultiEpochResult", "Lines", "FeII", "HighOrderBalmer", "PowerLaw",
    "BrokenPowerLaw", "Polynomial", "BalmerContinuum", "HostGalaxy", "StellarPopulation",
    "Extinction",
    "templates", "desi", "ozdes", "examples", "Component",
    "Parameter", "ParameterSet", "LINES", "line_wavelength", "read_line_list",
    "fitzpatrick99",
]
