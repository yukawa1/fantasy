import glob
import os
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.modeling.models import BlackBody
from sherpa.models import model
from sherpa.models.parameter import Parameter, tinyval
from sherpa.utils import SherpaFloat, sao_fcmp

__all__ = (
    "create_input_folder",
    "automatic_path",
    "create_line",
    "create_tied_model",
    "create_model",
    "create_fixed_model",
    "create_feii_model"
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



def create_line(name='line', pos=4861, ampl=5, min_ampl=0, max_ampl=500, fwhm= 1000, min_fwhm=5, max_fwhm=10000, offset=0, min_offset=-3000, max_offset=3000):
    """
    The create_line function creates a line with the specified parameters.
    
    Parameters:
    
        name (str): The name of the emission line.
    
        pos (float): The central wavelength of the emission line in Angstroms.
    
        ampl (float): The amplitude of the emission line in units of flux density at 
            position x=0, i.e., F(x=0) = ampl * continuum_level + offset .  Note that this is not an absolute value but depends on how you normalize your spectrum!  
            Default is 5, which means that if your spectrum has a continuum level equal to 1 then F(x=0)=5 and if it's 0 then F(x=0)=5+offset .  
            If you want to set an absolute flux density rather than relative values, use hard_min and hard_max instead!
    
        min_ampl (float): A lower limit for amplitude above which no lines will be created by create_line().  This can be useful when creating multiple lines from one input parameter because sometimes there are "bumps" or other features in a single spectrum where it makes sense to have multiple lines with different centroids but similar amplitudes so they don't overlap each other
    
    :param name='line': Used to Name the line in the model.
    :param pos=4861: Used to Specify the central wavelength of the line.
    :param ampl=5: Used to Set the amplitude of the emission line.
    :param min_ampl=0: Used to Set the lower limit of the amplitude parameter.
    :param max_ampl=500: Used to Set the maximum value that ampl can take.
    :param fwhm=1000: Used to Set the width of the line.
    :param min_fwhm=5: Used to Set the minimum value of the fwhm.
    :param max_fwhm=10000: Used to Set a hard limit on the fwhm.
    :param offset=0: Used to Shift the line center to a different position.
    :param min_offset=-3000: Used to Set the minimum value of the offset.
    :param max_offset=3000: Used to Set the maximum offset of the line.
    :return: An instance of the emission_line class.
    """
    line=Emission_Line(name=name)
    line.pos =pos
    
    line.ampl=ampl
    line.ampl.min=min_ampl
    line.ampl.max=max_ampl
    line.fwhm=fwhm
    line.fwhm.min=min_fwhm
    line.fwhm.max=max_fwhm
    line.offs_kms=offset
    line.offs_kms.min=min_offset
    line.offs_kms.max=max_offset
    
    return line
    
class Emission_Line(model.RegriddableModel1D):
    def __init__(self, name='line'):
        self.ampl = model.Parameter(name, "ampl", 10, min=0, hard_min=0, max=10000)
        self.pos = model.Parameter(
            name, "pos", 4861, min=0, frozen=True, units="angstroms"
        )
        self.offs_kms = model.Parameter(
            name, "offs_kms", 0, min=-10000, hard_min=-10000, max=10000, units="km/s"
        )
        

        self.fwhm = model.Parameter(
            name, "fwhm", 1000, min=0, hard_min=0, max=10000, units="km/s"
        )

        model.RegriddableModel1D.__init__(
            self, name, (self.ampl, self.pos, self.offs_kms, self.fwhm)
        )
    def line(self, pars, x):


        (ampl, pos, offs_kms, fwhm) = pars
        c = 299792.458
        offset = pos * offs_kms / c
        sigma = (pos + offset) * fwhm / (c * 2.354)

        f1 = ampl  # /(sigma*np.sqrt(2*np.pi))
        f2 = -((x - pos - offset) ** 2.0) / (2 * sigma ** 2.0)
        # fwhm = sigma * 2.355
        # flux = ampl * (sigma*np.sqrt(2*np.pi))

        return f1 * np.exp(f2)

    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the model"""
        return self.line(pars, x)
    
class Absorption_Line(model.RegriddableModel1D):
    def __init__(self, name='line'):
        self.ampl = model.Parameter(name, "ampl", -10, min=-10000, hard_min=-10000, max=0)
        self.pos = model.Parameter(
            name, "pos", 4861, min=0, frozen=True, units="angstroms"
        )
        self.offs_kms = model.Parameter(
            name, "offs_kms", 0, min=-10000, hard_min=-10000, max=10000, units="km/s"
        )
        

        self.fwhm = model.Parameter(
            name, "fwhm", 1000, min=0, hard_min=0, max=10000, units="km/s"
        )

        model.RegriddableModel1D.__init__(
            self, name, (self.ampl, self.pos, self.offs_kms, self.fwhm)
        )
    def line(self, pars, x):


        (ampl, pos, offs_kms, fwhm) = pars
        c = 299792.458
        offset = pos * offs_kms / c
        sigma = (pos + offset) * fwhm / (c * 2.354)

        f1 = ampl  # /(sigma*np.sqrt(2*np.pi))
        f2 = -((x - pos - offset) ** 2.0) / (2 * sigma ** 2.0)
        # fwhm = sigma * 2.355
        # flux = ampl * (sigma*np.sqrt(2*np.pi))

        return f1 * np.exp(f2)

    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the model"""
        return self.line(pars, x)

def _lorentz(pars, x):
    (ampl, pos, offs_kms, fwhm) = pars
    c = 299792.458

    offset = pos * offs_kms / c
    sigma = (pos + offset) * fwhm / (c * 2)
    f = (ampl / np.pi) * (sigma / ((x - pos - offset) ** 2.0 + sigma ** 2))
    return f


class Lorentz(model.RegriddableModel1D):
    def __init__(self, name="lorentz"):
        self.ampl = model.Parameter(name, "ampl", 5, min=0, hard_min=0)
        self.pos = model.Parameter(
            name, "pos", 4861, min=0, frozen=True, units="angstroms"
        )
        self.offs_kms = model.Parameter(
            name, "offs_kms", 1, min=-3000, hard_min=-3000, max=3000, units="km/s"
        )
        # self.sigma = model.Parameter(name, 'sigma', 1, min = 0, hard_min = 0, max = 200, units = 'angstroms')
        # self.flux = model.Parameter(name, 'flux', 10,  min = 0)

        self.fwhm = model.Parameter(
            name, "fwhm", 100, min=0, hard_min=0, max=1000, units="km/s"
        )

        model.RegriddableModel1D.__init__(
            self, name, (self.ampl, self.pos, self.offs_kms, self.fwhm)
        )

    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the model"""
        return _lorentz(pars, x)


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

        model.RegriddableModel1D.__init__(self, name, pars)

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
    The create_broad_model function creates a model for the broad lines in the data.
    It takes as input:
        filename - The name of the file containing all of your line names and positions. 
                This should be a csv file with columns named 'line', 'position'. 
                The first row should contain column headers.

        prefix - A string that will be added to each component name in your model, e.g., if you give it "broad_", all components will have names like "broad_[LINE NAME]".

        default_limits - If True, limits on amplitudes, fwhms, offsets are set automatically based on what is reasonable for this dataset (see below). If False, no limits are set except such that min(amplitude) > 0 and max(amplitude) < 100 (this is because some models may not have any amplitude parameters at all).

        amplitude - Initial value for amplitudes; see above documentation about how this value might change depending on whether default_limits=True or False.. Default = 2.0 .

        fwhm - Initial value for FWHMs; see above documentation about how this value might change depending on whether default_limits=

    :param filename='': Used to Specify the name of the file that contains all of the lines.
    :param prefix='': Used to Give each line a unique name.
    :param default_limits=True: Used to Set the limits of the parameters to a default value.
    :param amplitude=2: Used to Set the default amplitude of the lines to 2.
    :param fwhm=3000: Used to Set the default value for the fwhm of each line.
    :param offset=0: Used to Indicate that the offset is not fixed.
    :param min_offset=-3000: Used to Set the minimum value for the offset.
    :param max_offset=3000: Used to Set the upper limit of the offset parameter.
    :param min_amplitude=0: Used to Remove the baseline from the fit.
    :param max_amplitude=100: Used to Set the upper limit of the amplitude to 100.
    :param min_fwhm=100: Used to Remove the noise lines in the spectra.
    :param max_fwhm=7000: Used to Avoid the model to go out of the data range.
    :return: A model, which is a list of line objects.

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
                ampl=amplitude,
                fwhm=fwhm,
                offset=offset,
                min_offset=min_offset,
                max_offset=max_offset,
                min_ampl=min_amplitude,
                max_ampl=max_amplitude,
                min_fwhm=min_fwhm,
                max_fwhm=max_fwhm,
            )
    else:
        for i in range(len(df.line)):
            model += create_line(
                name=prefix + "_" + df.line[i]+'_'+df.position[i].round(0).astype(int).astype(str),
                pos=df.position[i],
                ampl=df.ampl[i],
                fwhm=df.fwhm[i],
                offset=df.offset[i],
                max_offset=df.max_offset[i],
                min_offset=df.min_offset[i],
                min_ampl=df.min_amplitude[i],
                max_ampl=df.max_amplitude[i],
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
        line = Absorption_Line(prefix + "_" + df.line[i]+'_'+df.position[i].round(0).astype(int).astype(str),)
            
        line.pos=df.position[i],
        line.ampl=amplitude,
        line.fwhm=fwhm,
        line.offs_kms=offset,
        line.offs_kms.min=min_offset,
        line.offs_kms.min=max_offset,
        line.ampl.min=min_amplitude,
        line.ampl.max=max_amplitude,
        line.fwhm.min=min_fwhm,
        line.fwhm.max=max_fwhm,
        model+=line
    
    return model


OIII5007 = create_line(
    "OIII5007",
    pos=5006.803341,
    fwhm=100,
    ampl=10,
    min_fwhm=0,
    max_fwhm=10000,
    offset=0,
    min_offset=-3000,
    max_offset=3000,
)


def OIII_NII(ref_line=OIII5007, prefix =""):
    """
    The narrow_basic function creates a narrow line profile for the OIII5007, NII6584, and NII6548 lines.
    It takes in the position of each line as well as its FWHM and amplitude. It also takes in an offset value which is 
    the same for all three lines. The function returns a dictionary with each of these parameters.

    :return: A list of the lines that are created.
    """

    OIII4958 = create_line(
        name=prefix + "_" + "OIII4958",
        pos=4958.896072,
        fwhm=ref_line.fwhm,
        ampl=ref_line.ampl / 3.0,
        offset=ref_line.offs_kms,
    )
    NII6584 = create_line(
        name=prefix + "_" + "NII6584", pos=6583.46, fwhm=ref_line.fwhm, offset=ref_line.offs_kms
    )
    NII6548 = create_line(
        name=prefix + "_" + "NIII6548",
        pos=6548.05,
        fwhm=ref_line.fwhm,
        offset=ref_line.offs_kms,
        ampl=NII6584.ampl / 3.0,
    )
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
