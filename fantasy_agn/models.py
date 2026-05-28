import glob
import os
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.modeling.models import BlackBody
from sherpa.models import model
from sherpa.models.parameter import Parameter, tinyval
from sherpa.utils.numeric_types import SherpaFloat
from sherpa.utils import sao_fcmp

__all__ = (
    # Original exports (maintained)
    "create_input_folder",
    "automatic_path",
    "create_line",
    "create_tied_model",
    "create_model",
    "create_fixed_model",
    "create_feii_model",
    # New exports for v1.0 (Lorentz and Voigt profiles)
    "create_lorentz_line",
    "create_voigt_line",
    "Lorentz",
    "Voigt",
    # Expose Gaussian line classes for direct instantiation
    "Emission_Line",
    "Absorption_Line",
    
    )
script_dir = os.path.dirname(__file__)
input_path = os.path.join(script_dir, "input")


def create_input_folder(xmin=4000, xmax=7000, path_to_folder='', overwrite= True):
    """
    The create_input_folder function creates a folder in the specified path.
    The function takes three arguments:
        1) xmin - the minimum value of the range to be used for creating input files, default is 4000
        2) xmax - the maximum value of the range to be used for creating input files, default is 7000
        3) path_to_folder - The path where you want your new folder created. Default is current directory
    
    :param xmin=4000: Used to Set the minimum value of the x-axis (position or wavelength).
    :param xmax=7000: Used to Limit the maximum value of the x-axis.
    :param path_to_folder='': Used to Specify the path to the folder where you want to save your file.
    :param overwrite=True: Used to Overwrite the files in the input folder.
    :return: The path to the folder where the input files are stored.
        """
    
    try:
        os.mkdir(path_to_folder)
        print("Directory ", path_to_folder, " Created ")
    except FileExistsError:
        print("Directory ", path_to_folder, " already exists")
    global path1
    path1 = path_to_folder
    if overwrite:
        for files in glob.glob(input_path + "/*.csv"):
            df = pd.read_csv(files)
            try:
                df = df[df.position > xmin]
                df = df[df.position < xmax]
            except:
                df = df[df.wav > xmin]
                df = df[df.wav < xmax]

            name = path_to_folder  + Path(files).name
            df.to_csv(name, index=False)

def automatic_path(s, path_to_models=input_path, overwrite=True):
    """
    The automatic_path function takes a spectrum object and creates a directory with the name of the spectrum.
    It then saves all models within the input path that fall within the wavelength range of your data to this directory.
    
    :param s: Used to Define the spectrum object.
    :param path_to_models=input_path: Used to Specify the path to the models.
    :param overwrite=True: Used to Overwrite a model if it already exists.
    :return: The path to the directory where the.
    """
    dirName = s.name
    xmin = s.wave[0]
    xmax = s.wave[-1]
    try:
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")
    global path1
    path1 = dirName + "/"
    print("global", path1)
    
    if overwrite:

        for files in glob.glob(path_to_models + "/*.csv"):
            df = pd.read_csv(files)
            try:
                df = df[df.position > xmin]
                df = df[df.position < xmax]
            except:
                df = df[df.wav > xmin]
                df = df[df.wav < xmax]

            name = dirName + "/" + Path(files).name
            df.to_csv(name, index=False)
def continuum(spec,name='brokenpowerlaw', refer=5500, min_refer=5400, max_refer=7000, index1=-1.7, min_index1=-3, max_index1=0, index2=0, min_index2=-1, max_index2=1):
    """
    The continuum function is a broken power law.
    
    The function is defined as:
    
        f(x) = ampl * (x / refer)^index1 for x > refer
             = ampl * (refer / refer)^(index1+index2) for x < refer
    
        where the amplitude, index, and reference wavelength are parameters of the model.  The default values are:
    
            ampl     1e-16 erg/cm**2/s/Angstrom
            index    -0.7   dimensionless constant that defines the slope of the power law above 5000 Angstroms; it is equal to -0.7 in our fit to UVES spectra from Vestergaard & Peterson 1991 ApJ 374, 344; this value was changed from -0.5 to account for higher resolution data from Keck Observatories since that paper was published (see Section 2 of Smith et al 2006 ApJS 164, 508).  This parameter can be set between 0 and 3 inclusive with a limit placed on its value by requiring that it must be greater than or equal to min_index when evaluated at max_refer and less than or equal to max_index when evaluated at min_refer; see below for more details on these parameters which control
    
    :param spec: Used to Pass in the spectrum to fit.
    :param name='brokenpowerlaw': Used to Name the model in a fit.
    :param refer=5500: Used to Set the wavelength of the continuum.
    :param min_refer=5400: Used to Set the minimum wavelength of the continuum.
    :param max_refer=5600: Used to Limit the range of wavelengths to which the continuum is calculated.
    :param index1=-1.7: Used to Fit the continuum.
    :param min_index1=-3: Used to Set the minimum value of the index.
    :param max_index1=0: Used to Fix the index of the power law at zero.
    :param index2=0: Used to Make the continuum flat.
    :param min_index2=-1: Used to Set the minimum value of the index2 parameter.
    :param max_index2=1: Used to Fix the index of the continuum at 1.
    :return: The continuum of the spectrum given as arguments.
    
    """
    mask=(spec.wave>min_refer)&(spec.wave<max_refer)
    ampl=np.median(spec.flux[mask])
    

    class BrokenPowerlaw(model.RegriddableModel1D):
        def __init__(self, name="brokenpowerlaw"):

            self.refer = model.Parameter(
                name,
                "refer",
                refer,
                min=min_refer,
                max=max_refer,
                hard_min=tinyval,
                frozen=False,
                units="angstroms",
            )

            self.ampl = model.Parameter(
                name, "ampl", ampl, max=1.05 * ampl, min=0.95*ampl, hard_min=tinyval, units="angstroms"
            )

            self.index1 = model.Parameter(
                name, "index1", index1, min=min_index1, max=max_index1, frozen=False)

            self.index2 = model.Parameter(name, "index2", index2, min=min_index2, max=max_index2)

            model.ArithmeticModel.__init__(
                self, name, (self.refer, self.ampl, self.index1, self.index2)
            )

        # @modelCacher1d
        def calc(self, p, x, xhi=None, **kwargs):

            if 0.0 == p[0]:
                raise ValueError(
                    "model evaluation failed, " + "%s refer cannot be zero" % self.name
                )

            x = np.asarray(x, dtype=SherpaFloat)
            arg = x / p[0]
            arg = p[1] * (
                np.where(arg > 1.0, np.power(
                    arg, p[2] + p[3]), np.power(arg, p[2]))
            )
            return arg
    return BrokenPowerlaw()


def _feii(pars, x):
    """
    The _feii function is a function that takes in the parameter vector, pars,
    and x-values. It then calculates the model fluxes for each FeII line and 
    combines them into one array. The parameters are:
    
        - pars[0] = offset velocity (km/s)
        - pars[len(uniq)] = offset velocity (km/s)
        - fwhm = full width half max of lines (km/s)
    
            where len(uniq) is the number of unique FeII lines in our data set.
    
    :param pars: Used to Pass the parameters of the function.
    :param x: Used to Define the wavelength range.
    :return: The f value for each of the unique values in the feii_model.
    
    :doc-author: Trelent
    """
    c = 299792.458

    dft = pd.read_csv(path1 + "feII_model.csv")

    uniq = pd.unique(dft.ime)
    offs_kms = pars[len(uniq)]
    fwhm = pars[len(uniq) + 1]

    f2 = 0
    index = 1  # starting index in pars
    for i in range(len(uniq)):
        df = dft[dft["ime"] == uniq[i]]

        rxc = df.wav.to_numpy()
        Int = df.Int.to_numpy()

        par = pars[i]
        for i in range(len(rxc)):

            shift = rxc[i] * offs_kms / c
            sigma = (rxc[i] + shift) * fwhm / (c * 2.354)
            f1 = (x - rxc[i] - shift) ** 2.0 / (2.0 * sigma ** 2)
            f = Int[i] * np.exp(-f1)
            f = par * f
            f2 += f
    return f2


class FeII(model.RegriddableModel1D):
    def __init__(self, name="feii"):
        dft = pd.read_csv(path1 + "feII_model.csv")

        uniq = pd.unique(dft.ime)
        pars = []

        for i in range(len(uniq)):
            pars.append(
                Parameter(name, "amp_" + uniq[i],
                          2, min=0, max=1000, frozen=False)
            )
        for p in pars:
            setattr(self, p.name, p)
        self.offs_kms = model.Parameter(
            name, "offs_kms", 0, min=-1000, hard_min=-3000, max=1000
        )
        self.fwhm = model.Parameter(
            name, "fwhm", 2000, min=tinyval, hard_min=tinyval, max=10000
        )

        pars.append(self.offs_kms)
        pars.append(self.fwhm)

        model.RegriddableModel1D.__init__(self, name, pars)

    def calc(self, pars, x, *args, **kwargs):
        return _feii(pars, x)


def _uv_feii(pars, x):
    c = 299792.458

    dft = pd.read_csv(path1 + "uvfe.csv")

    uniq = pd.unique(dft.ime)
    offs_kms = pars[len(uniq)]
    fwhm = pars[len(uniq) + 1]

    f2 = 0
    index = 1  # starting index in pars
    for i in range(len(uniq)):
        df = dft[dft["ime"] == uniq[i]]

        rxc = df.wav.to_numpy()
        Int=df.Int.to_numpy()

        par = pars[i]
        for i in range(len(rxc)):

            shift = rxc[i] * offs_kms / c
            sigma = (rxc[i] + shift) * fwhm / (c * 2.354)
            f1 = (x - rxc[i] - shift) ** 2.0 / (2.0 * sigma ** 2)
            f = Int[i] * np.exp(-f1)
            f = par * f
            f2 += f
    return f2


class UV_FeII(model.RegriddableModel1D):
    def __init__(self, name="uv_feii"):
        dft = pd.read_csv(path1 + "uvfe.csv")

        uniq = pd.unique(dft.ime)
        pars = []

        for i in range(len(uniq)):
            pars.append(
                Parameter(name, "amp_" + uniq[i],
                          2, min=0, max=20, frozen=False)
            )
        for p in pars:
            setattr(self, p.name, p)
        self.offs_kms = model.Parameter(
            name, "offs_kms", 0, min=-1000, hard_min=-3000, max=1000
        )
        self.fwhm = model.Parameter(
            name, "fwhm", 2000, min=tinyval, hard_min=tinyval, max=10000
        )

        pars.append(self.offs_kms)
        pars.append(self.fwhm)

        model.RegriddableModel1D.__init__(self, name, pars)

    def calc(self, pars, x, *args, **kwargs):
        return _uv_feii(pars, x)


def _resolve_legacy_line_kwargs(kwargs, amplitude, min_amplitude, max_amplitude, offset):
    """Map legacy line keyword names onto the refactored API."""
    if "ampl" in kwargs:
        amplitude = kwargs.pop("ampl")
    if "min_ampl" in kwargs:
        min_amplitude = kwargs.pop("min_ampl")
    if "max_ampl" in kwargs:
        max_amplitude = kwargs.pop("max_ampl")
    if "offs_kms" in kwargs:
        offset = kwargs.pop("offs_kms")
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")
    return amplitude, min_amplitude, max_amplitude, offset



def create_line(name='line', pos=4861, amplitude=5, min_amplitude=0, max_amplitude=500, 
                fwhm=1000, min_fwhm=5, max_fwhm=10000, offset=0, min_offset=-3000, max_offset=3000,
                **kwargs):
    """
    Create a Gaussian emission line with the specified parameters.
    
    Parameters
    ----------
    name : str
        The name of the emission line (default: 'line')
    pos : float
        The central wavelength of the emission line in Angstroms (default: 4861)
    amplitude : float
        The amplitude of the emission line (default: 5)
    min_amplitude : float
        Lower limit for amplitude (default: 0)
    max_amplitude : float
        Upper limit for amplitude (default: 500)
    fwhm : float
        Full width at half maximum in km/s (default: 1000)
    min_fwhm : float
        Lower limit for FWHM (default: 5)
    max_fwhm : float
        Upper limit for FWHM (default: 10000)
    offset : float
        Velocity offset in km/s (default: 0)
    min_offset : float
        Lower limit for velocity offset (default: -3000)
    max_offset : float
        Upper limit for velocity offset (default: 3000)
    
    Returns
    -------
    line : Emission_Line
        An instance of the Emission_Line class with specified parameters
        
    Notes
    -----
    This function provides backward compatibility with the original API.
    You can also directly instantiate Emission_Line, Lorentz, or Voigt classes.
    """
    amplitude, min_amplitude, max_amplitude, offset = _resolve_legacy_line_kwargs(
        kwargs, amplitude, min_amplitude, max_amplitude, offset
    )
    line = Emission_Line(name=name, amplitude=amplitude, pos=pos, offset=offset, fwhm=fwhm,
                         min_amplitude=min_amplitude, max_amplitude=max_amplitude,
                         min_offset=min_offset, max_offset=max_offset,
                         min_fwhm=min_fwhm, max_fwhm=max_fwhm)
    return line


def create_lorentz_line(name='lorentz', pos=4861, amplitude=5, min_amplitude=0, max_amplitude=500,
                        fwhm=100, min_fwhm=5, max_fwhm=10000, offset=0, min_offset=-3000, max_offset=3000,
                        **kwargs):
    """
    Create a Lorentzian line with the specified parameters.
    
    Parameters
    ----------
    name : str
        The name of the line (default: 'lorentz')
    pos : float
        The central wavelength in Angstroms (default: 4861)
    amplitude : float
        The amplitude of the line (default: 5)
    min_amplitude : float
        Lower limit for amplitude (default: 0)
    max_amplitude : float
        Upper limit for amplitude (default: 500)
    fwhm : float
        Full width at half maximum in km/s (default: 100)
    min_fwhm : float
        Lower limit for FWHM (default: 5)
    max_fwhm : float
        Upper limit for FWHM (default: 10000)
    offset : float
        Velocity offset in km/s (default: 0)
    min_offset : float
        Lower limit for velocity offset (default: -3000)
    max_offset : float
        Upper limit for velocity offset (default: 3000)
    
    Returns
    -------
    line : Lorentz
        An instance of the Lorentz class with specified parameters
    """
    amplitude, min_amplitude, max_amplitude, offset = _resolve_legacy_line_kwargs(
        kwargs, amplitude, min_amplitude, max_amplitude, offset
    )
    line = Lorentz(name=name, amplitude=amplitude, pos=pos, offset=offset, fwhm=fwhm,
                   min_amplitude=min_amplitude, max_amplitude=max_amplitude,
                   min_offset=min_offset, max_offset=max_offset,
                   min_fwhm=min_fwhm, max_fwhm=max_fwhm)
    return line


def create_voigt_line(name='voigt', pos=4861, amplitude=5, min_amplitude=0, max_amplitude=500,
                      fwhm_g=100, fwhm_l=100, min_fwhm_g=5, max_fwhm_g=10000,
                      min_fwhm_l=5, max_fwhm_l=10000, offset=0, min_offset=-3000, max_offset=3000,
                      normalize_peak=False, **kwargs):
    """
    Create a Voigt line profile with the specified parameters.
    
    Parameters
    ----------
    name : str
        The name of the line (default: 'voigt')
    pos : float
        The central wavelength in Angstroms (default: 4861)
    amplitude : float
        The amplitude of the line (default: 5)
    min_amplitude : float
        Lower limit for amplitude (default: 0)
    max_amplitude : float
        Upper limit for amplitude (default: 500)
    fwhm_g : float
        Gaussian FWHM in km/s (default: 100)
    fwhm_l : float
        Lorentzian FWHM in km/s (default: 100)
    min_fwhm_g : float
        Lower limit for Gaussian FWHM (default: 5)
    max_fwhm_g : float
        Upper limit for Gaussian FWHM (default: 10000)
    min_fwhm_l : float
        Lower limit for Lorentzian FWHM (default: 5)
    max_fwhm_l : float
        Upper limit for Lorentzian FWHM (default: 10000)
    offset : float
        Velocity offset in km/s (default: 0)
    min_offset : float
        Lower limit for velocity offset (default: -3000)
    max_offset : float
        Upper limit for velocity offset (default: 3000)
    normalize_peak : bool
        If True, normalize the profile so `amplitude` matches the peak value.
        Defaults to False for backward compatibility.
    
    Returns
    -------
    line : Voigt
        An instance of the Voigt class with specified parameters
    """
    amplitude, min_amplitude, max_amplitude, offset = _resolve_legacy_line_kwargs(
        kwargs, amplitude, min_amplitude, max_amplitude, offset
    )
    line = Voigt(name=name, amplitude=amplitude, pos=pos, offset=offset,
                 fwhm_g=fwhm_g, fwhm_l=fwhm_l,
                 min_amplitude=min_amplitude, max_amplitude=max_amplitude,
                 min_offset=min_offset, max_offset=max_offset,
                 min_fwhm_g=min_fwhm_g, max_fwhm_g=max_fwhm_g,
                 min_fwhm_l=min_fwhm_l, max_fwhm_l=max_fwhm_l)
    line.normalize_peak = normalize_peak
    return line
    
class Emission_Line(model.RegriddableModel1D):
    """Gaussian emission line profile with unified parameter names.
    
    Parameters
    ----------
    name : str
        Name of the line component
    amplitude : float
        Line amplitude (default: 10)
    pos : float
        Line position in Angstroms (default: 4861)
    offset : float
        Velocity offset in km/s (default: 0)
    fwhm : float
        Full width at half maximum in km/s (default: 1000)
    min_amplitude, max_amplitude : float
        Bounds for amplitude
    min_offset, max_offset : float
        Bounds for velocity offset in km/s
    min_fwhm, max_fwhm : float
        Bounds for FWHM in km/s
    """
    def __init__(self, name='line', amplitude=10, pos=4861, offset=0, fwhm=1000,
                 min_amplitude=0, max_amplitude=10000,
                 min_offset=-10000, max_offset=10000,
                 min_fwhm=0, max_fwhm=10000, **kwargs):
        amplitude, min_amplitude, max_amplitude, offset = _resolve_legacy_line_kwargs(
            kwargs, amplitude, min_amplitude, max_amplitude, offset
        )
        self.amplitude = model.Parameter(name, "amplitude", amplitude, 
                                         min=min_amplitude, hard_min=0, max=max_amplitude)
        self.pos = model.Parameter(
            name, "pos", pos, min=0, frozen=True, units="angstroms"
        )
        self.offset = model.Parameter(
            name, "offset", offset, min=min_offset, hard_min=-30000, max=max_offset, units="km/s"
        )
        self.fwhm = model.Parameter(
            name, "fwhm", fwhm, min=min_fwhm, hard_min=0, max=max_fwhm, units="km/s"
        )

        # Backward compatibility aliases
        self.ampl = self.amplitude
        self.offs_kms = self.offset

        model.RegriddableModel1D.__init__(
            self, name, (self.amplitude, self.pos, self.offset, self.fwhm)
        )
    
    def line(self, pars, x):
        """Calculate Gaussian line profile."""
        (amplitude, pos, offset, fwhm) = pars
        c = 299792.458
        shift = pos * offset / c
        sigma = (pos + shift) * fwhm / (c * 2.354)

        f1 = amplitude
        f2 = -((x - pos - shift) ** 2.0) / (2 * sigma ** 2.0)

        return f1 * np.exp(f2)

    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the model"""
        return self.line(pars, x)
    
class Absorption_Line(model.RegriddableModel1D):
    """Gaussian absorption line profile with unified parameter names.
    
    Parameters
    ----------
    name : str
        Name of the line component
    amplitude : float
        Line amplitude (negative for absorption, default: -10)
    pos : float
        Line position in Angstroms (default: 4861)
    offset : float
        Velocity offset in km/s (default: 0)
    fwhm : float
        Full width at half maximum in km/s (default: 1000)
    """
    def __init__(self, name='line', amplitude=-10, pos=4861, offset=0, fwhm=1000,
                 min_amplitude=-10000, max_amplitude=0,
                 min_offset=-10000, max_offset=10000,
                 min_fwhm=0, max_fwhm=10000, **kwargs):
        amplitude, min_amplitude, max_amplitude, offset = _resolve_legacy_line_kwargs(
            kwargs, amplitude, min_amplitude, max_amplitude, offset
        )
        self.amplitude = model.Parameter(name, "amplitude", amplitude, 
                                         min=min_amplitude, hard_min=-10000, max=max_amplitude)
        self.pos = model.Parameter(
            name, "pos", pos, min=0, frozen=True, units="angstroms"
        )
        self.offset = model.Parameter(
            name, "offset", offset, min=min_offset, hard_min=-30000, max=max_offset, units="km/s"
        )
        self.fwhm = model.Parameter(
            name, "fwhm", fwhm, min=min_fwhm, hard_min=0, max=max_fwhm, units="km/s"
        )

        # Backward compatibility aliases
        self.ampl = self.amplitude
        self.offs_kms = self.offset

        model.RegriddableModel1D.__init__(
            self, name, (self.amplitude, self.pos, self.offset, self.fwhm)
        )
    
    def line(self, pars, x):
        """Calculate Gaussian absorption line profile."""
        (amplitude, pos, offset, fwhm) = pars
        c = 299792.458
        shift = pos * offset / c
        sigma = (pos + shift) * fwhm / (c * 2.354)

        f1 = amplitude
        f2 = -((x - pos - shift) ** 2.0) / (2 * sigma ** 2.0)

        return f1 * np.exp(f2)

    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the model"""
        return self.line(pars, x)

class Lorentz(model.RegriddableModel1D):
    """Lorentzian line profile with unified parameter names.
    
    Parameters
    ----------
    name : str
        Name of the line component
    amplitude : float
        Line amplitude (default: 5)
    pos : float
        Line position in Angstroms (default: 4861)
    offset : float
        Velocity offset in km/s (default: 0)
    fwhm : float
        Full width at half maximum in km/s (default: 100)
    min_amplitude, max_amplitude : float
        Bounds for amplitude
    min_offset, max_offset : float
        Bounds for velocity offset in km/s
    min_fwhm, max_fwhm : float
        Bounds for FWHM in km/s
    """
    def __init__(self, name="lorentz", amplitude=5, pos=4861, offset=0, fwhm=100,
                 min_amplitude=0, max_amplitude=10000,
                 min_offset=-3000, max_offset=3000,
                 min_fwhm=0, max_fwhm=10000, **kwargs):
        amplitude, min_amplitude, max_amplitude, offset = _resolve_legacy_line_kwargs(
            kwargs, amplitude, min_amplitude, max_amplitude, offset
        )
        self.amplitude = model.Parameter(name, "amplitude", amplitude, 
                                         min=min_amplitude, hard_min=0, max=max_amplitude)
        self.pos = model.Parameter(
            name, "pos", pos, min=0, frozen=True, units="angstroms"
        )
        self.offset = model.Parameter(
            name, "offset", offset, min=min_offset, hard_min=-30000, max=max_offset, units="km/s"
        )
        self.fwhm = model.Parameter(
            name, "fwhm", fwhm, min=min_fwhm, hard_min=0, max=max_fwhm, units="km/s"
        )

        # Backward compatibility aliases
        self.ampl = self.amplitude
        self.offs_kms = self.offset

        model.RegriddableModel1D.__init__(
            self, name, (self.amplitude, self.pos, self.offset, self.fwhm)
        )

    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the Lorentzian model"""
        (amplitude, pos, offset, fwhm) = pars
        c = 299792.458

        shift = pos * offset / c
        gamma = (pos + shift) * fwhm / (c * 2)
        f = (amplitude / np.pi) * (gamma / ((x - pos - shift) ** 2.0 + gamma ** 2))
        return f


class Voigt(model.RegriddableModel1D):
    """Voigt line profile (convolution of Gaussian and Lorentzian) with unified parameter names.
    
    The Voigt profile is a convolution of Gaussian and Lorentzian profiles, useful for
    modeling lines with both thermal and natural broadening.
    
    Parameters
    ----------
    name : str
        Name of the line component
    amplitude : float
        Line amplitude (default: 5)
    pos : float
        Line position in Angstroms (default: 4861)
    offset : float
        Velocity offset in km/s (default: 0)
    fwhm_g : float
        Gaussian FWHM in km/s (default: 100)
    fwhm_l : float
        Lorentzian FWHM in km/s (default: 100)
    min_amplitude, max_amplitude : float
        Bounds for amplitude
    min_offset, max_offset : float
        Bounds for velocity offset in km/s
    min_fwhm_g, max_fwhm_g : float
        Bounds for Gaussian FWHM in km/s
    min_fwhm_l, max_fwhm_l : float
        Bounds for Lorentzian FWHM in km/s
    
    Notes
    -----
    By default, `amplitude` acts as a multiplicative scaling factor on the
    Voigt profile, so it is not equal to the peak value. Set
    `normalize_peak = True` to force `amplitude` to represent the peak height.
    """
    def __init__(self, name="voigt", amplitude=5, pos=4861, offset=0,
                 fwhm_g=100, fwhm_l=100,
                 min_amplitude=0, max_amplitude=10000,
                 min_offset=-3000, max_offset=3000,
                 min_fwhm_g=0, max_fwhm_g=10000,
                 min_fwhm_l=0, max_fwhm_l=10000, **kwargs):
        amplitude, min_amplitude, max_amplitude, offset = _resolve_legacy_line_kwargs(
            kwargs, amplitude, min_amplitude, max_amplitude, offset
        )
        self.amplitude = model.Parameter(name, "amplitude", amplitude, 
                                         min=min_amplitude, hard_min=0, max=max_amplitude)
        self.pos = model.Parameter(
            name, "pos", pos, min=0, frozen=True, units="angstroms"
        )
        self.offset = model.Parameter(
            name, "offset", offset, min=min_offset, hard_min=-30000, max=max_offset, units="km/s"
        )
        self.fwhm_g = model.Parameter(
            name, "fwhm_g", fwhm_g, min=min_fwhm_g, hard_min=0, max=max_fwhm_g, units="km/s"
        )
        self.fwhm_l = model.Parameter(
            name, "fwhm_l", fwhm_l, min=min_fwhm_l, hard_min=0, max=max_fwhm_l, units="km/s"
        )

        # Backward compatibility aliases
        self.ampl = self.amplitude
        self.offs_kms = self.offset

        # Option to interpret `amplitude` as the peak value of the Voigt
        # profile (True) instead of a multiplicative scaling factor (False).
        # Default is False to preserve the historical multiplicative scaling
        # behavior.
        self.normalize_peak = False

        # Import wofz once per instance to avoid repeated imports inside
        # `calc()` during tight fitting loops. If scipy isn't available at
        # construction time we allow `calc()` to perform the import lazily.
        try:
            from scipy.special import wofz as _wofz
        except Exception:
            _wofz = None
        self._wofz = _wofz

        model.RegriddableModel1D.__init__(
            self, name, (self.amplitude, self.pos, self.offset, self.fwhm_g, self.fwhm_l)
        )

    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the Voigt model using Faddeeva function"""
        # reuse per-instance import if available, otherwise import lazily
        _wofz = getattr(self, "_wofz", None)
        if _wofz is None:
            from scipy.special import wofz as _wofz
            self._wofz = _wofz

        (amplitude, pos, offset, fwhm_g, fwhm_l) = pars
        c = 299792.458

        shift = pos * offset / c
        center = pos + shift
        
        # Convert FWHM to standard parameters
        sigma_g = (center * fwhm_g) / (c * 2.354)  # Gaussian sigma
        gamma_l = (center * fwhm_l) / (c * 2.0)    # Lorentzian gamma (HWHM)
        
        # Voigt profile using Faddeeva function
        z = ((x - center) + 1j * gamma_l) / (sigma_g * np.sqrt(2))
        voigt = np.real(_wofz(z)) / (sigma_g * np.sqrt(2 * np.pi))
        # If requested, scale the profile so that `amplitude` equals the
        # peak (maximum) value of the Voigt profile. This keeps the
        # historical behavior (amplitude as multiplicative factor) by default.
        if getattr(self, "normalize_peak", False):
            # compute analytic peak at x == center
            z0 = (1j * gamma_l) / (sigma_g * np.sqrt(2))
            peak = np.real(_wofz(z0)) / (sigma_g * np.sqrt(2 * np.pi))
            if peak == 0 or not np.isfinite(peak):
                return amplitude * voigt
            return amplitude * (voigt / peak)

        return amplitude * voigt


def _narrow_gaussian(pars, x):

    (ampl, pos, offset, fwhm) = pars
    sigma = fwhm / 2.355
    f1 = ampl / (sigma * np.sqrt(2 * np.pi))
    f2 = -((x - pos - offset) ** 2.0) / (2 * sigma ** 2.0)

    return f1 * np.exp(f2)


class Narrow_Line(model.RegriddableModel1D):
    def __init__(self, name="Narrow_Line"):
        self.ampl = model.Parameter(name, "ampl", 100, min=0, hard_min=0)
        self.pos = model.Parameter(
            name, "pos", 5008.240, min=0, frozen=True, units="angstroms"
        )
        self.offset = model.Parameter(
            name, "offset", 1, min=-5, hard_min=-5, max=5.0, units="angstroms"
        )
        self.fwhm = model.Parameter(
            name, "fwhm", 5, min=0, hard_min=0, max=100)

        model.RegriddableModel1D.__init__(
            self, name, (self.ampl, self.pos, self.offset, self.fwhm)
        )

    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the model"""

        return _narrow_gaussian(pars, x)


def _host(pars, x):

    # (a1, a2, a3, a4, a5) = pars
    gal = np.load("gal.npy")

    f = 0
    for i in range(gal.shape[0]):
        f += pars[i] * gal[i]
    # normpars = []
    # print('ovo je norma', np.linalg.norm(f))

    # for i in range(len(pars)-1):
    #     pars[i]=pars[i]/np.linalg.norm(f)
    #     normpars.append(pars[i])

    # for i in range(gal.shape[0]):
    #     f += normpars[ i ] * gal[ i ]
    # #f=pars[-1]*f

    return f
    # return a1*gal[0]+a2*gal[1]+a3*gal[2]+a4*gal[3]+a5*gal[4]


class Host(model.RegriddableModel1D):
    def __init__(self, name="Host"):
        gal = np.load("gal.npy")
        pars = []

        for i in range(gal.shape[0]):
            pars.append(Parameter(name, "a%d" % i, 1, min=0))
        for p in pars:
            setattr(self, p.name, p)

        model.RegriddableModel1D.__init__(self, name, pars)

    def calc(self, pars, x, *args, **kwargs):
        return _host(pars, x)



def _balmer_conti(pars, x):
    """Fit the Balmer continuum from the model of Dietrich+02"""
    lambda_BE = 3646.0  # A
    bbflux = BlackBody(temperature=pars[1] * u.K, scale=10000)
    tau = pars[2] * (x / lambda_BE) ** 3
    bb = bbflux(x * u.AA)
    result = pars[0] * bb * (1.0 - np.exp(-tau))
    ind = np.where(x > lambda_BE, True, False)
    if ind.any() == True:
        result[ind] = 0.0
    return result.value


class BalCon(model.RegriddableModel1D):
    def __init__(self, name="BalCon"):
        self.A = model.Parameter(name, "A", 1, min=tinyval, hard_min=0)
        self.T = model.Parameter(
            name, "T", 10000, min=5000, frozen=False, units="kelvins"
        )
        self.tau = model.Parameter(
            name, "tau", 1, min=0.01, hard_min=tinyval, max=2)

        model.RegriddableModel1D.__init__(
            self, name, (self.A, self.T, self.tau))

    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the model"""
        return _balmer_conti(pars, x)


class Polynomial(model.RegriddableModel1D):
    def __init__(self, name="polynomial"):
        pars = []

        for i in range(6):
            pars.append(Parameter(name, "c%d" % i, 0, frozen=True))
        pars[0].val = 1
        pars[0].frozen = False
        for p in pars:
            setattr(self, p.name, p)

        self.offset = Parameter(name, "offset", 0, frozen=True)
        pars.append(self.offset)
        model.ArithmeticModel.__init__(self, name, pars)

    # @modelCacher1d
    def calc(self, p, x, xhi=None, **kwargs):
        x = np.asarray(x, dtype=SherpaFloat)
        y = np.zeros_like(x)
        xtemp = x - p[6]
        y += p[5]
        for i in [4, 3, 2, 1, 0]:
            y = y * xtemp + p[i]

        return y


# This model computes continuum emission using a power-law.
class Powerlaw(model.RegriddableModel1D):
    """Power-law model.
    It is for use when the independent axis is in Angstroms.
    Attributes
    ----------
    refer
        The reference point at which the amplitude is defined, with
        units of Angstroms.
    ampl
        The amplitude at the reference point.
    index
        The index for the power law.
    See Also
    --------
    BrokenPowerlaw, Polynomial
    Notes
    -----
    The functional form of the model for points is::
        f(x) = ampl * (x / refer)^index
    and for integrated data sets the low-edge of the grid is used.

    """

    def __init__(self, name="powerlaw"):
        self.refer = Parameter(
            name,
            "refer",
            5600.0,
            tinyval,
            hard_min=tinyval,
            frozen=True,
            units="angstroms",
        )
        self.ampl = Parameter(
            name, "ampl", 1.0, tinyval, hard_min=tinyval, units="angstroms"
        )
        self.index = Parameter(name, "index", -1.7, -3.0, 0)

        model.ArithmeticModel.__init__(
            self, name, (self.refer, self.ampl, self.index))

    # @modelCacher1d
    def calc(self, p, x, xhi=None, **kwargs):
        """
        The calc function evaluates the model for given parameter values and data.
        
        Parameters
        ----------
        pars : list of numbers or a number or array-like object (list, tuple, array) 
        
            The parameter value(s) to be evaluated. If pars is a single number then it is used as the amplitude value for all bins; if pars has more than one element then they are used as the values for each bin in order. 
        
            .. note: If an array-like object is provided that does not match the size of x then an error will be raised by numpy when calc() is called. This can occur if you do something like p[0] = [2,3] instead of p[0][:] = [2,3].
        
            .. warning: The parameters must already have been set before calling this function - there isn't a mechanism to set them at evaluation time (i.e., you can't do p[0].val=[2.,3.]). You can use `set_par` to change parameter settings after defining them with `create_parameter`.  
        x : number or array-like object (list, tuple, NumPy ndarray), optional 
        
            The independent variable where calc
        
        :param self: Used to Access the attributes and methods of the class in which the function is defined.
        :param p: Used to Calculate the model.
        :param x: Used to Evaluate the model.
        :param xhi=None: Used to Indicate that the calculation is to be done without.
        :param **kwargs: Used to Handle named parameters that are not defined in the model class.
        :return: The model evaluated at the values of x for a given set of parameters.
        
        :doc-author: Trelent
        """
        if 0.0 == p[0]:
            raise ValueError(
                "model evaluation failed, " + "%s refer cannot be zero" % self.name
            )

        x = np.asarray(x, dtype=SherpaFloat)
        arg = x / p[0]
        arg = p[1] * np.power(arg, p[2])

        return arg


def _gaus(A, xc, s, w, x):
    g = A * np.exp(-((x - xc - s) ** 2.0) / (2.0 * w ** 2.0))
    return g


# with open('/home/yukawa/fantasy/fantasy/uvfe/uv_temp.txt') as f:
#     uv=f.readlines()
# uv = [x.strip() for x in uv]

# def _uv_fe(pars,x):
#     F=0
#     index=0
#     f=0
#     shift=pars[6]
#     sigma=pars[7]
#     for files in uv:
#         wave, rat=np.genfromtxt(files, unpack=True)
#         for i in range(len(wave)):
#             f += rat[i] * _line((pars[index],wave[i],shift, sigma), x)

#         F +=f
#         #plt.plot(x,F)
#         index += 1
#     return np.asarray(F)

# class UV_FeII(model.RegriddableModel1D):

#     def __init__(self, name='uv_feii'):
#         self.amp60 = model.Parameter(name, 'amp60', 1, min = 0, hard_min = 0, max = 100)
#         self.amp61 = model.Parameter(name, 'amp61', 1, min = 0, hard_min = 0, max = 100)
#         self.amp62 = model.Parameter(name, 'amp62', 1, min = 0, hard_min = 0, max = 100)
#         self.amp63 = model.Parameter(name, 'amp63', 1, min = 0, hard_min = 0, max = 100)
#         self.amp78 = model.Parameter(name, 'amp78', 1, min = 0, hard_min = 0, max = 100)
#         self.ampzw1 = model.Parameter(name, 'ampzw1', 1, min = 0, hard_min = 0, max = 100)
#         self.shift = model.Parameter(name, 'shift', 0, min=-3000, max=3000)
#         self.sigma = model.Parameter(name, 'sigma', 2000, min=500, hard_min=0, max = 7000)

#         model.RegriddableModel1D.__init__(self, name,
#                                           (self.amp60, self.amp61, self.amp62,
#                                            self.amp63, self.amp78, self.ampzw1, self.shift, self.sigma))

#     def calc(self, pars, x, *args, **kwargs):
#         """Evaluate the model"""
#         return _uv_fe(pars, x)

def _line(pars, x):
    

    (ampl, pos, offs_kms, fwhm) = pars
    c = 299792.458 
    offset = pos * offs_kms / c
    sigma = (pos + offset) * fwhm / (c * 2.354)
    
    f1 = ampl#/(sigma*np.sqrt(2*np.pi))
    f2 =- (x - pos - offset)**2./(2*sigma**2.)
    #fwhm = sigma * 2.355
    #flux = ampl * (sigma*np.sqrt(2*np.pi))
    
    return f1*np.exp(f2)

def _balmer(pars, x):
    lambda_BE = 3646.0
    f = 0
    df=pd.read_csv(path1+'balmer.csv')
    wave=df.position.to_numpy()
    rat=df.int.to_numpy()
    for i in range(len(wave)):
        f += rat[i] * _line((pars[0], wave[i], pars[1], pars[2]), x)
    ind = np.where(x <= lambda_BE, True, False)
    if ind.any() == True:
        f[ind] = 0.0
    return f


class Bal_Lines(model.RegriddableModel1D):
    def __init__(self, name="Bal_lines"):
        self.ampl = model.Parameter(name, "ampl", 50, min=1, hard_min=0)
        # self.pos = model.Parameter(name, 'pos', 4861, min = 0,frozen = True, units = 'angstroms')
        self.offs_kms = model.Parameter(
            name, "offs_kms", 1, min=-3000, hard_min=-3000, max=3000, units="km/s"
        )
        # self.sigma = model.Parameter(name, 'sigma', 1, min = 0, hard_min = 0, max = 200, units = 'angstroms')
        # self.flux = model.Parameter(name, 'flux', 10,  min = 0)

        self.fwhm = model.Parameter(
            name, "fwhm", 3000, min=1000, hard_min=0, max=4000, units="km/s"
        )

        model.RegriddableModel1D.__init__(
            self, name, (self.ampl, self.offs_kms, self.fwhm)
        )

    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the model"""
        return _balmer(pars, x)


def _ssp_host(pars, x):

    gal = np.load("ica.npy")

    f = 0
    for i in range(gal.shape[0]):
        f += pars[i] * gal[i]

    return f
    # return a1*gal[0]+a2*gal[1]+a3*gal[2]+a4*gal[3]+a5*gal[4]


class SSP(model.RegriddableModel1D):
    def __init__(self, name="Host"):
        gal = np.load("ica.npy")
        pars = []

        for i in range(gal.shape[0]):
            pars.append(Parameter(name, "a%d" % i, 1, min=0))
        for p in pars:
            setattr(self, p.name, p)

        model.RegriddableModel1D.__init__(self, name, pars)

    def calc(self, pars, x, *args, **kwargs):
        return _ssp_host(pars, x)


# def Line(
#     name="line",
#     pos=4861,
#     amplitude=2,
#     fwhm=3000,
#     offset=0,
#     min_offset=-3000,
#     max_offset=3000,
#     min_amplitude=0,
#     max_amplitude=1000,
#     min_fwhm=0,
#     max_fwhm=7000
# ):
#     """
#         Function to define lines in the model.

#         Parameters:
#         ___________
#         name: Name of the line; should be string.
#         position: Air wavelength of the line; it is fixed.
#         amplitude: Amplitude of the line.
#         fwhm: Full widhth of the half maximum; given in Angstroms.
#         offset: Ofsset of the line from the starting position.
#         Additionally all the parameters constraints can be given here. Example:
#         min_offset=0

#         Returns: Gaussian  with given parameters.


#         """
#     gaussian = Broad_Line(name)
#     gaussian.pos = pos
#     gaussian.ampl = amplitude
#     gaussian.fwhm = fwhm
#     gaussian.offs_kms = offset
#     gaussian.offs_kms.max = max_offset
#     gaussian.offs_kms.min = min_offset
#     gaussian.ampl.min = min_amplitude
#     gaussian.ampl.max = max_amplitude
#     gaussian.fwhm.min = min_fwhm
#     gaussian.fwhm.max = max_fwhm

#     return gaussian


def create_model(
    files=[],
    prefix="",
    default_limits=True,
    amplitude=2,
    fwhm=3000,
    offset=0,
    min_offset=-3000,
    max_offset=3000,
    min_amplitude=0,
    max_amplitude=600,
    min_fwhm=100,
    max_fwhm=7000,
):
    """
    Create a composite model from multiple emission lines defined in CSV files.
    
    Parameters
    ----------
    files : list
        List of CSV filenames containing line information. Each CSV should have columns:
        'line' (line name), 'position' (wavelength in Angstroms), and optionally
        'amplitude', 'fwhm', 'offset', and their min/max values
    prefix : str
        Prefix to add to each line name (default: '')
    default_limits : bool
        If True, use the default parameter values and limits provided.
        If False, use values from the CSV files (default: True)
    amplitude : float
        Default amplitude for all lines (default: 2)
    fwhm : float
        Default FWHM in km/s for all lines (default: 3000)
    offset : float
        Default velocity offset in km/s (default: 0)
    min_offset : float
        Lower limit for velocity offset (default: -3000)
    max_offset : float
        Upper limit for velocity offset (default: 3000)
    min_amplitude : float
        Lower limit for amplitude (default: 0)
    max_amplitude : float
        Upper limit for amplitude (default: 600)
    min_fwhm : float
        Lower limit for FWHM (default: 100)
    max_fwhm : float
        Upper limit for FWHM (default: 7000)
    
    Returns
    -------
    model : composite model
        Sum of all emission lines from the specified files
        
    Examples
    --------
    >>> model = create_model(files=['narrow_basic.csv', 'broad.csv'], 
    ...                      prefix='agn', amplitude=10, fwhm=2000)
    """

    if len(files) > 0:
        F = []
        for file in files:
            F.append(pd.read_csv(path1 + file))
        df = pd.concat(F)
        df.reset_index(drop=True, inplace=True)
    else:
        print("List of csv files should be given to create model")
    model = 0
    if default_limits:
        for i in range(len(df.line)):
            model += create_line(
                prefix + "_" + df.line[i]+'_'+df.position[i].round(0).astype(int).astype(str),
                pos=df.position[i],
                amplitude=amplitude,
                fwhm=fwhm,
                offset=offset,
                min_offset=min_offset,
                max_offset=max_offset,
                min_amplitude=min_amplitude,
                max_amplitude=max_amplitude,
                min_fwhm=min_fwhm,
                max_fwhm=max_fwhm,
            )
    else:
        # backward-compatible handling: some CSVs use 'ampl' instead of 'amplitude'
        if "amplitude" in df.columns:
            amp_col = "amplitude"
        elif "ampl" in df.columns:
            amp_col = "ampl"
        else:
            amp_col = None

        for i in range(len(df.line)):
            amp_val = df[amp_col][i] if amp_col is not None else amplitude
            model += create_line(
                name=prefix + "_" + df.line[i]+'_'+df.position[i].round(0).astype(int).astype(str),
                pos=df.position[i],
                amplitude=amp_val,
                fwhm=df.fwhm[i],
                offset=df.offset[i],
                max_offset=df.max_offset[i],
                min_offset=df.min_offset[i],
                min_amplitude=df.min_amplitude[i],
                max_amplitude=df.max_amplitude[i],
                min_fwhm=df.min_fwhm[i],
                max_fwhm=df.max_fwhm[i],
            )
    return model


def create_absorption_model(
    files=[],
    prefix="",
    amplitude=-2,
    fwhm=3000,
    offset=0,
    min_offset=-3000,
    max_offset=3000,
    min_amplitude=-1000,
    max_amplitude=0,
    min_fwhm=100,
    max_fwhm=7000,
):
    """
    Create a composite absorption model from multiple lines defined in CSV files.
    
    Parameters
    ----------
    files : list
        List of CSV filenames containing line information
    prefix : str
        Prefix to add to each line name (default: '')
    amplitude : float
        Default amplitude for all lines (negative for absorption, default: -2)
    fwhm : float
        Default FWHM in km/s for all lines (default: 3000)
    offset : float
        Default velocity offset in km/s (default: 0)
    min_offset : float
        Lower limit for velocity offset (default: -3000)
    max_offset : float
        Upper limit for velocity offset (default: 3000)
    min_amplitude : float
        Lower limit for amplitude (default: -1000)
    max_amplitude : float
        Upper limit for amplitude (default: 0)
    min_fwhm : float
        Lower limit for FWHM (default: 100)
    max_fwhm : float
        Upper limit for FWHM (default: 7000)
    
    Returns
    -------
    model : composite model
        Sum of all absorption lines from the specified files
    """

    if len(files) > 0:
        F = []
        for file in files:
            F.append(pd.read_csv(path1 + file))
        df = pd.concat(F)
        df.reset_index(drop=True, inplace=True)
    else:
        print("List of csv files should be given to create model")
    model = 0
    for i in range(len(df.line)):
        line = Absorption_Line(
            name=prefix + "_" + df.line[i]+'_'+df.position[i].round(0).astype(int).astype(str),
            amplitude=amplitude,
            pos=df.position[i],
            offset=offset,
            fwhm=fwhm,
            min_amplitude=min_amplitude,
            max_amplitude=max_amplitude,
            min_offset=min_offset,
            max_offset=max_offset,
            min_fwhm=min_fwhm,
            max_fwhm=max_fwhm
        )
        model += line
    
    return model


OIII5007 = create_line(
    "OIII5007",
    pos=5006.803341,
    fwhm=100,
    amplitude=10,
    min_fwhm=0,
    max_fwhm=10000,
    offset=0,
    min_offset=-3000,
    max_offset=3000,
)


def OIII_NII(ref_line=OIII5007, prefix=""):
    """
    Create a tied model for [OIII] and [NII] lines with fixed flux ratios.
    
    The function creates the following lines tied to the reference line:
    - [OIII]4959: amplitude = ref_line.amplitude / 3.0
    - [NII]6584: tied FWHM and offset
    - [NII]6548: amplitude = [NII]6584.amplitude / 3.0
    
    Parameters
    ----------
    ref_line : Emission_Line
        Reference emission line (typically [OIII]5007) to which other lines are tied
    prefix : str
        Prefix to add to line names (default: '')
    
    Returns
    -------
    model : composite model
        Sum of all tied lines including the reference line
    
    Examples
    --------
    >>> ref = create_line('OIII5007', pos=5006.803341, amplitude=10, fwhm=100)
    >>> model = OIII_NII(ref_line=ref, prefix='narrow')
    """

    OIII4958 = create_line(
        name=prefix + "_" + "OIII4958",
        pos=4958.896072,
        fwhm=ref_line.fwhm.val,
        amplitude=ref_line.amplitude.val / 3.0,
        offset=ref_line.offset.val,
    )
    # Tie parameters
    OIII4958.fwhm = ref_line.fwhm
    OIII4958.offset = ref_line.offset
    OIII4958.amplitude = ref_line.amplitude / 3.0
    
    NII6584 = create_line(
        name=prefix + "_" + "NII6584", 
        pos=6583.46, 
        fwhm=ref_line.fwhm.val, 
        offset=ref_line.offset.val
    )
    # Tie parameters
    NII6584.fwhm = ref_line.fwhm
    NII6584.offset = ref_line.offset
    
    NII6548 = create_line(
        name=prefix + "_" + "NII6548",
        pos=6548.05,
        fwhm=ref_line.fwhm.val,
        offset=ref_line.offset.val,
        amplitude=NII6584.amplitude.val / 3.0,
    )
    # Tie parameters
    NII6548.fwhm = ref_line.fwhm
    NII6548.offset = ref_line.offset
    NII6548.amplitude = NII6584.amplitude / 3.0
    
    return ref_line + OIII4958 + NII6584 + NII6548


def create_tied_model(
    name='OIII5007',
    files=[],
    prefix="",
    fix_oiii_ratio=True,
    position=5006.803341,
    amplitude=2,
    fwhm=10,
    offset=0,
    min_offset=-3000,
    max_offset=1000,
    min_amplitude=0,
    max_amplitude=1000,
    min_fwhm=0,
    max_fwhm=10000,
    included=False
    ):
    """
    The create_tied_model function creates a model that is tied to another parameter.
    It takes the following arguments:
        name - The name of the line (e.g., OIII5007)
        files - A list of csv files containing data for this line (optional)
        prefix - The prefix used in defining parameters in other functions (default = "") 
    
         If included == True and fix_oiii_ratio==True, then it creates an instance of 
         the OIII_NII class with a reference emission line defined by ref_line.  
    
         If included == True and fix_oiii_ratio==False, then it creates an instance 
         of EmissionLine with a reference emission line defined by ref_line.  
    
         If included == False and fix_oiii_ratio==True, then it creates an instance 
         of OIII5007 only.
    
    :param name='OIII5007': Used to Name the emission line.
    :param files=[]: Used to Pass a list of csv files to the model.
    :param prefix="": Used to Give a name to the model.
    :param fix_oiii_ratio=True: Used to Fix the ratio of oiii5007 to nii6583.
    :param position=0: Used to Define the position of the reference line.
    :param amplitude=2: Used to Set the amplitude of the emission line to 2.
    :param fwhm=10: Used to Set the fwhm of the reference line.
    :param offset=0: Used to Indicate that the line should be fixed to 0.
    :param min_offset=-3000: Used to Set the minimum offset of the line.
    :param max_offset=1000: Used to Set the maximum offset of a line to 1000 km/s.
    :param min_amplitude=0: Used to Avoid the model to go below 0.
    :param max_amplitude=1000: Used to Set a maximum value for the amplitude of the emission line.
    :param min_fwhm=0: Used to Avoid the fwhm of lines to be set to 0.
    :param max_fwhm=10000: Used to Avoid errors when the fwhm is very high.
    :param included=False: Used to Create a model without the emission line of interest.
    :return: A model that is tied to the parameters of a reference line.
    """
    
    ref_line = Emission_Line(prefix+'_'+name)
    ref_line.pos=position
    ref_line.fwhm=fwhm
    ref_line.ampl=amplitude
    ref_line.ampl.min = min_amplitude
    ref_line.ampl.max= max_amplitude
    ref_line.fwhm.min=min_fwhm
    ref_line.fwhm.max=max_fwhm
    ref_line.offs_kms=offset
    ref_line.offs_kms.min=min_offset
    ref_line.offs_kms.max=max_offset

    if included == True and fix_oiii_ratio==True:
        model = OIII_NII(prefix=prefix, ref_line=ref_line)
    if included == True and fix_oiii_ratio==False:
        model=ref_line
    if included == False and fix_oiii_ratio==True:
        model = OIII_NII(prefix=prefix, ref_line=ref_line)

    if fix_oiii_ratio==False and included==False:
        model = 0

    if len(files) > 0:
        F = []
        for file in files:
            F.append(pd.read_csv(path1 + file))
        df = pd.concat(F)
        df.reset_index(drop=True, inplace=True)
    else:
        print("List of csv files should be given to create model")
    for i in range(len(df.line)):
        model += create_line(name =
            prefix + "_" + df.line[i] + "_" + str(df.position.astype(int)[i]),
            pos=df.position[i],
            fwhm=ref_line.fwhm,
            offset=ref_line.offs_kms,
            ampl=amplitude,
            min_offset=ref_line.offs_kms.min,
            max_offset=ref_line.offs_kms.max,
            min_fwhm=ref_line.fwhm.min,
            max_fwhm=ref_line.fwhm.max,
        )

    return model


def create_fixed_model(files=[], 
                       name='',     
                       amplitude=2,
                       fwhm=3000,
                       offset=0,
                       min_offset=-3000,
                       max_offset=3000,
                       min_amplitude=0,
                       max_amplitude=600,
                       min_fwhm=100,
                       max_fwhm=7000,):
    """
    The create_fixed_model function creates a model that is fixed fwhm for all lines.
    The function takes as an argument a list of csv files, which contain the information (name and position) of all lines included in the model.
    It also takes as arguments: name, amplitude, fwhm (km/s), offset (km/s) and min_offset and max_offset (km/s). 
    
    :param files=[]: Used to Pass a list of csv files to the model.
    :param name='': Used to Give the model a name.
    :param amplitude=2: Used to Set the initial value of the amplitude parameter.
    :param fwhm=3000: Used to Set the fwhm of the gaussian profile.
    :param offset=0: Used to Shift the lines to the center of each pixel.
    :param min_offset=-3000: Used to Set the minimum value of the offset.
    :param max_offset=3000: Used to Set the maximum offset of the line from its rest position.
    :param min_amplitude=0: Used to Set the minimum value of the amplitude parameter.
    :param max_amplitude=600: Used to Limit the maximum amplitude of the lines.
    :param min_fwhm=100: Used to Set a lower limit to the fwhm parameter.
    :param max_fwhm=7000: Used to Set the maximum value of the fwhm parameter.
    :param : Used to Set the initial value of the amplitude parameter.
    :return: A fixed_lines class object.
    """

    if len(files) > 0:
        F = []
        for file in files:
            F.append(pd.read_csv(path1 + file))
        df = pd.concat(F)
        df.reset_index(drop=True, inplace=True)
    else:
        print("List of csv files should be given to create model")
        
    class Fixed_Lines(model.RegriddableModel1D):
        def __init__(self, name=name):
            dft = df
            dft['name']=dft.line+'_'+dft.position.round(0).astype(int).astype(str)

            uniq = dft.name.tolist()
            pars = []

            for i in range(len(uniq)):
                pars.append(
                    Parameter(name, "amp_" + uniq[i],
                            amplitude, min=min_amplitude, max=max_amplitude, frozen=False)
                )
            for p in pars:
                setattr(self, p.name, p)
            self.offs_kms = model.Parameter(
                name, "offs_kms", offset, min=min_offset, hard_min=min_offset, max=max_offset, units='km/s'
            )
            self.fwhm = model.Parameter(
                name, "fwhm", fwhm, min=min_fwhm, hard_min=min_fwhm, max=max_fwhm, units='km/s'
            )

            pars.append(self.offs_kms)
            pars.append(self.fwhm)
            self.dft=dft

            model.RegriddableModel1D.__init__(self, name, pars)

        def fixed(self,pars, x):
            f=0
            dft=self.dft
            fwhm=pars[-1]
            offs_kms=pars[-2]
            pos = dft.position.to_numpy()
            for i in range(len(pos)):
                
                
                c = 299792.458
                offset = pos[i] * offs_kms / c
                sigma = (pos[i] + offset) * fwhm / (c * 2.354)

                f1 = pars[i]  # /(sigma*np.sqrt(2*np.pi))
                f2 = -((x - pos[i] - offset) ** 2.0) / (2 * sigma ** 2.0)
                # fwhm = sigma * 2.355
                # flux = ampl * (sigma*np.sqrt(2*np.pi))

                f+= f1 * np.exp(f2)
            return f
        def calc(self, pars, x, *args, **kwargs):
            return self.fixed(pars, x)
    return Fixed_Lines()

def create_feii_model(name='feii', fwhm=2000, min_fwhm=1000, max_fwhm=8000, offset=0, min_offset=-3000, max_offset=3000):
    """
    The create_feii_model function creates a FeII model with the specified parameters.
    
    :param name='feii': Used to Name the component.
    :param fwhm=2000: Used to Set the fwhm of the feii emission line.
    :param min_fwhm=1000: Used to Set the minimum value of the fwhm parameter.
    :param max_fwhm=8000: Used to Set the upper limit of the fwhm range.
    :param offset=0: Used to Set the central wavelength of the feii emission line.
    :param min_offset=-3000: Used to Set the minimum value of the offset parameter.
    :param max_offset=3000: Used to Set the maximum offset velocity of the feii emission line.
    :return: A feii object.
    """
    fe=FeII(name)
    fe.fwhm=fwhm
    fe.fwhm.max=max_fwhm
    fe.fwhm.min=min_fwhm
    fe.offs_kms=offset
    fe.offs_kms.min= min_offset
    fe.offs_kms.max=max_offset
    return fe
