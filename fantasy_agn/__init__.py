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
interface in the browser.
"""
__version__ = "2.0.0"

from . import desi, examples, ml, ozdes, presets, rm, sdss, templates
from .host import StellarPopulation
from .components import (Absorption, Blackbody, Calibration, BalmerContinuum, BrokenPowerLaw, Component, Extinction, FeII,
                         HighOrderBalmer, HostGalaxy, Lines, Polynomial, PowerLaw)
from .extinction import fitzpatrick99
from .classify import Classification, classify, classify_fit
from .compare import Comparison, compare
from .confidence import ModelConfidence, candidate_ties, model_confidence
from .fitting import fit, fit_many, guess


def app(port=8780, open_browser=True):
    """Start the FANTASY application in the browser (see :mod:`fantasy_agn.app`)."""
    from .app import run
    run(port=port, open_browser=open_browser)
from .lines import LINE_SETS, LINES, line_info, line_set, line_table, line_wavelength, lines_in, read_line_list
from .model import Model, load_model
from .multiepoch import MultiEpochResult, fit_epochs
from .parameters import Parameter, ParameterSet
from .result import FitResult
from .spectrum import Spectrum, read
from .io import save, load
from .absorbers import fit_absorbed
from .refine import prune_feii, refine
from .simcheck import SimulationCheck, simulation_check
from .jointcheck import CalibrationCheck, JointSimulationCheck, calibration_check
from .redshift import RedshiftResult, find_redshift
from .predictive import PredictiveCheck, calibrate, link_test, predictive_check, profile_likelihood
from .robustness import Robustness, robustness
from .fitcompare import FitComparison, compare_fits
from .correlations import epoch_correlations, epoch_table
from .catalogs import lookup_catalogs
from .multiepoch import EpochSet
from .residuals import analyze_residuals
from .linescan import suggest_lines
from .sample import fit_sample, SampleResult

__all__ = [
    "read", "Spectrum", "Model", "save", "load", "fit_absorbed", "refine", "prune_feii", "simulation_check", "SimulationCheck", "calibration_check", "CalibrationCheck", "JointSimulationCheck", "find_redshift", "RedshiftResult", "analyze_residuals", "suggest_lines", "lookup_catalogs", "fit_sample", "SampleResult", "fit", "fit_many", "guess", "fit_epochs", "compare", "Comparison", "model_confidence", "ModelConfidence", "candidate_ties",
    "predictive_check", "PredictiveCheck", "link_test", "profile_likelihood", "calibrate", "robustness", "Robustness", "compare_fits", "FitComparison", "epoch_table", "epoch_correlations", "EpochSet",
    "classify", "classify_fit", "Classification",
    "presets", "app",
    "FitResult", "MultiEpochResult", "Lines", "FeII", "HighOrderBalmer", "PowerLaw",
    "BrokenPowerLaw", "Polynomial", "BalmerContinuum", "HostGalaxy", "StellarPopulation",
    "Extinction", "Absorption", "Blackbody", "Calibration",
    "templates", "desi", "sdss", "ozdes", "rm", "ml", "examples", "Component",
    "Parameter", "ParameterSet", "LINES", "LINE_SETS", "line_set", "line_wavelength", "line_table", "lines_in", "line_info",
    "read_line_list",
    "fitzpatrick99",
]
