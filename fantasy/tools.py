from astropy.io import fits
from itertools import product
from numpy.testing._private.utils import KnownFailureException
import sfdmap
import numpy as np
from PyAstronomy import pyasl
from copy import deepcopy
import pylab as plt
from spectres import spectres
from scipy import interpolate, linalg
from scipy.integrate import simps
from sherpa.data import Data1D
from sherpa.fit import Fit
from sherpa.optmethods import LevMar, NelderMead, MonCar, GridSearch
from sherpa.stats import LeastSq, Cash, Chi2, Likelihood, Stat
from sherpa.estmethods import Confidence
import pandas as pd
import glob
import json
import concurrent.futures
from sherpa.models.parameter import Parameter, tinyval
import natsort
from sklearn import preprocessing
from RegscorePy import aic, bic
import multiprocessing.pool
import os

script_dir = os.path.dirname(__file__)
sfdpath = os.path.join(script_dir, "sfddata")
pathEigen = os.path.join(script_dir, "eigen/")
import warnings

warnings.filterwarnings("ignore")


class spectrum(object):
    def __init__(self):

        self._wave = None
        self._flux = None
        self._err = None
        self._z = None
        self._ra = None
        self._dec = None
        self._mjd = None
        self._plate = None
        self._fiber = None
        self._host = None
        self._agn = None
        self._style = None

    def DeRedden(self, ebv=0):
        """
        Function for dereddening  a flux vector  using the parametrization given by
        [Fitzpatrick (1999)](https://iopscience.iop.org/article/10.1086/316293).

        Parameters
        ----------
        ebv : float, optional
            Color excess E(B-V). If not given it will be automatically derived from
             Dust map data from [Schlegel, Finkbeiner and Davis (1998)](http://adsabs.harvard.edu/abs/1998ApJ...500..525S),
             by default 0
        """

        if ebv != 0:
            self.flux = pyasl.unred(self.wave, self.flux, ebv)
        else:
            m = sfdmap.SFDMap(sfdpath)
            self.flux = pyasl.unred(self.wave, self.flux, m.ebv(self.ra, self.dec))

    def CorRed(self, redshift=0):
        """
        The CorRed function corrects the flux for redshift.
        It takes in a redshift and corrects the wavelength, flux, and error arrays by that given redshift.

        :param self: Used to Reference the class object.
        :param redshift=0: Used to Specify the redshift of the object.
        :return: The wavelength, flux and error arrays for the object at a redshift of z=0.

        
        """

        if redshift != 0:
            self.z = redshift

        self.wave = self.wave / (1 + self.z)
        self.flux = self.flux * (1 + self.z)
        self.err = self.err * (1 + self.z)
        self.fwhm = self.fwhm / (1 + self.z)

    def rebin(self, minw=3900, maxw=6800):
        """
        The rebin function rebins the spectrum to a smaller wavelength range.
        The default is 3900-6800 angstroms, but you can specify your own min and max values.
        
        :param self: Used to Reference the object itself.
        :param minw=3900: Used to Set the minimum wavelength of the new array.
        :param maxw=6800: Used to Define the upper limit of the wavelength range to be used for fitting.
        :return: The new wavelength and flux values for the spectrum.
        
        
        """
        new = np.arange(minw, maxw, 1)
        flux, err = spectres(new, self.wave, self.flux, spec_errs=self.err)
        self.flux = flux
        self.err = err
        self.wave = new

    def crop(self, xmin=4050, xmax=7300):
        """
        The crop function crops the spectrum to a specified wavelength range.
        
        Parameters: 
        xmin (float): The minimum wavelength of the crop region. Default is 4050 Angstroms. 
        xmax (float): The maximum wavelength of the crop region. Default is 7300 Angstroms. 
        
            Returns: None, but modifies self in place by cropping it to only include wavelengths between xmin and xmax.
        
        :param self: Used to Reference the object itself.
        :param xmin=4050: Used to Set the lower wavelength limit of the cropped spectrum.
        :param xmax=7300: Used to Select the wavelength range of interest.
        :return: A new spectrum object with the cropped wavelength range.
        
    
        """

        mask = (self.wave > xmin) & (self.wave < xmax)
        self.wave = self.wave[mask]
        self.flux = self.flux[mask]
        self.err = self.err[mask]
    def plot_spec(self):
        """
        The plot_spec function plots the spectra.
        
        :param self: Used to Access the class attributes.
        :return: A tuple containing the x and y values for a plot of the spectrum.  
        """
        plt.plot(self.wave, self.flux)
        plt.show()

    def show_range(self):
        print("Minimum of wavelength is ", self.wave[0])
        print("Maximum of wavelength is ", self.wave[-1])

    def fit_host(self):
        """
        The fit_host function fits the host galaxy and AGN components to the spectrum.
        It takes a list of tuples as an input, which are combinations of number of 
        galaxy components and number of QSO components. The function fits each combination 
        of parameters with a linear model, then calculates reduced chi square for each fit. 
        The function returns the best fit combination based on reduced chi square value.
        
        :param self: Used to Access variables that belongs to the class.
        :return: The number of components for the host galaxy and the agn.
        
        """

        self.wave_old = self.wave
        self.flux_old = self.flux
        self.err_old = self.err
        lis = list(product(np.linspace(3, 10, 8), np.linspace(3, 15, 13)))
        ch = []
        HOST = []
        AGN = []
        FIT = []
        a = _prepare_host(self.wave, self.flux, self.err, 10, 15)
        for i in range(len(lis)):
            gal = int(lis[i][0])
            qso = int(lis[i][1])
            hs = a[3][:gal]
            ags = a[4][:qso]
            k = _fith(hs, ags, a[1])
            host = 0
            for i in range(gal):
                host += k[i] * a[3][i]
            ag_r = 0
            for i in range(qso):
                ag_r += k[gal + i] * a[4][i]
            if _check_host(host) == True and _check_host(ag_r) == True:
                HOST.append(host)
                AGN.append(ag_r)
                FIT.append(host + ag_r)
                ch.append(_calculate_chi(a[1], a[2], host + ag_r, k))
        if not ch:
            print(
                "Uuups... All of the combinations returned negative host.Better luck with other AGN"
            )
        else:
            idx = _find_nearest(ch)
            print("Number of galaxy components ", int(lis[idx][0]))
            print("Number of QSO components ", int(lis[idx][1]))
            print("Reduced chi square ", ch[idx])

            plt.figure(figsize=(18, 6))
            plt.plot(a[0], a[1])
            plt.plot(a[0], HOST[idx], label="host")
            plt.plot(a[0], AGN[idx], label="AGN")
            plt.plot(a[0], FIT[idx], label="FIT")

            plt.legend()
            plt.show()
        self.wave = a[0]
        self.flux = a[1] - HOST[idx]
        self.err = a[2]
        self.host = HOST[idx]

    def restore(self):
        self.wave = self.wave_old
        self.flux = self.flux_old
        self.err = self.err_old

    def plot(self):
        fig = px.line(x=self.wave, y=self.flux)
        fig.show()

    def fit(self, model, stat=Chi2(), method=LevMar(), ntrial=1):
        """
       The fit function fits a model to the data. 
       The user can choose between several different statistics and methods.
       The default is least squares with the LevMar method.
       
       :param self: Used to Reference the class itself.
       :param model: Used to Define the model that is being fit to the data.
       :param stat=LeastSq(): Used to Specify the statistic that is used to fit the data.
       :param method=LevMar(): Used to Specify the type of fit to be used.
       :return: The model and data objects.
       
       """
        stat = stat
        method = method
        d = Data1D("AGN", self.wave, self.flux, self.err)
        gfit = Fit(d, model, stat=stat, method=method)
        gfit = Fit(d, model, stat=stat, method=method)

        gres = gfit.fit()
        statistic=gres.dstatval
        print('stati', statistic)
        if ntrial > 1:
            
            i=0
            while i < ntrial:
                gfit = Fit(d, model, stat=stat, method=method)
                gres = gfit.fit()
                print(i+1, "iter stat: ", gres.dstatval)
                if gres.dstatval==statistic:
                    break
                i+=1
            
        self.gres = gres
        self.d = d
        self.model = model
        mod = model

    def save_json(self, suffix="pars"):
        """
        The save_json function saves the parameter values in a JSON file.
        The filename is constructed from the name of the model and either 'pars' or 'samples'.
        
        :param self: Used to Refer to the object itself.
        :param suffix='pars': Used to Specify the name of the file that is saved.
        :return: A dictionary of the parameter names and values.
        
        """
        dicte = zip(self.gres.parnames, self.gres.parvals)
        res = dict(dicte)
        filename = self.name + "_" + suffix + ".json"
        with open(filename, "w") as fp:
            json.dump(res, fp)

    @staticmethod
    def __fitting(d):
        model = mod
        stat = LeastSq()
        method = LevMar()
        gfit = Fit(d, model, stat=stat, method=method)
        gres = gfit.fit()
        return list(gres.parvals)

    def bootstrap(self, ntrails):
        d = self.d
        rande = []
        for i in range(ntrails):
            rande.append(randomize(d))
        results = []
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     #results=executor.map(self.__fitting,rande)
        #      for da in rande:
        #          f= executor.submit(self.__fitting, da)
        #          print((f.result)

        # #dicte=zip(name, reuslts)
        # #res=dict(dicte)
        # print(results)
        pool = multiprocessing.pool.ThreadPool(processes=7)
        warnings.filterwarnings("ignore")

        return_list = pool.map(self.__fitting, rande, chunksize=1)
        pool.close()
        # dicte=zip(self.gres.parnames, return_list)
        # res=dict(dicte)
        # print(res)
        df = pd.DataFrame(return_list, columns=self.gres.parnames)
        print(df)
        self.df = df


class read_sdss(spectrum):
    def __init__(self, filename):

        super(read_sdss, self).__init__()
        hdulist = fits.open(filename)
        hdu = hdulist[1]
        data = hdu.data
        self.z = hdulist[2].data["z"][0]
        self.ra = hdulist[0].header["plug_ra"]
        self.dec = hdulist[0].header["plug_dec"]
        self.mjd = hdulist[0].header["mjd"]
        self.plate = hdulist[0].header["plateid"]
        self.fiber = hdulist[0].header["fiberid"]
        self.name = filename.split(".")[0]
        try:
            y = data.flux
            x = 10 ** (data.loglam)
            iv = data.ivar
        except AttributeError:
            y = data.FLUX
            x = 10 ** (data.LOGLAM)
            iv = data.IVAR
        super_threshold_indices = iv == 0
        iv[super_threshold_indices] = np.median(iv)
        self.err = 1.0 / np.sqrt(iv)
        self.flux = y
        self.wave = x
        c = 299792.458
        wdisp = data["wdisp"]
        frac = self.wave[1] / self.wave[0]
        dlam = (frac - 1) * self.wave
        fwhm = 2.355 * wdisp * dlam
        velscale = np.log(frac) * c
        self.fwhm = fwhm
        self.velscale = velscale


class read_gama_fits(spectrum):
    def __init__(self, filename):
        """Class for reading [GAMA](http://www.gama-survey.org/) spectrum file as ```spectrum``` object.
        Extracting the
        - Wavelength
        - Flux
        - Flux error
        - Redshift
        - Ra and Dec
        - Plate, Fiber and MJD

        Parameters
        ----------
        filename : .fits
            Name of the SDSS fits file to read
        """

        super(read_gama_fits, self).__init__()
        hdu = fits.open(filename)
        if hdu[0].header["TELESCOP"] == "SDSS 2.5-M":
            h1 = hdu[0].header
            lam_temp = h1["CRVAL1"] + h1["CDELT1"] * np.arange(h1["NAXIS1"])
            wave = 10 ** lam_temp
            self.wave = wave
        else:
            wmin = hdu[0].header["WMIN"]
            wmax = hdu[0].header["WMAX"]
            NAXIS1 = hdu[0].header["NAXIS1"]
            self.wave = np.linspace(wmin, wmax, NAXIS1)
        data = hdu[0].data
        flux = data[0].tolist()
        c = 299792.458
        self.ra = hdu[0].header["RA"]
        self.dec = hdu[0].header["DEC"]
        self.z = hdu[0].header["Z"]
        self.name = filename.split(".")[-2]
        dataset = pd.DataFrame(
            {"wave": self.wave, "flux": flux, "err": data[1].tolist()}, index=None
        )
        dataset = dataset.interpolate()
        dataset = dataset.dropna()

        # flux=np.asarray(flux)
        self.wave = dataset.wave.to_numpy()
        # flux[np.isnan(flux)]=0
        self.flux = dataset.flux.to_numpy()
        self.err = dataset.err.to_numpy()
        # wdisp = data['wdisp']
        frac = self.wave[1] / self.wave[0]
        dlam = (frac - 1) * self.wave
        fwhm = 2.355 * dlam
        velscale = np.log(frac) * c
        self.fwhm = fwhm
        self.velscale = velscale


class read_text(spectrum):
    def __init__(self, filename):
        """Class to read ASCI like file as as ```spectrum``` object.

        Parameters
        ----------
        filename : ASCII
            File name should be three or two columns where first column is 
            wavelength, second flux and third (optinal) flux error. 
        """
        super(read_text, self).__init__()
        try:
            self.wave, self.flux, self.err = np.genfromtxt(filename, unpack=True)
        except ValueError:
            self.wave, self.flux = np.genfromtxt(filename, unpack=True)
            self.err = np.ones_like(self.wave)

        c = 299792.458
        # wdisp = data['wdisp']
        frac = self.wave[1] / self.wave[0]
        dlam = (frac - 1) * self.wave
        fwhm = 2.355 * dlam
        velscale = np.log(frac) * c
        self.fwhm = fwhm
        self.velscale = velscale
        self.name = filename.split(".")[-2]


def _maskingEigen(x):
    """Preparing mask for eigen spectra host preparation

    Parameters
    ----------
    x : array
        Spectrum wavelength to mask in order to rebin  

    Returns
    -------
    boolean array
        
    """

    if min(x) > 3450:
        minx = min(x)
    else:
        minx = 3450
    if max(x) < 7000:
        maxx = max(x) - 10
    else:
        maxx = 7200 - 10

    mask = (x > minx) & (x < maxx)
    return mask


def _prepare_host(wave, flux, err, galaxy=5, agns=10):
    """Preparing qso and galaxy eigen spectra

    Parameters
    ----------
    wave : array
        Spectrum wavelength
    flux : array
        Spectrum flux
    err : array
        Spectrum error flux
    galaxy : int, optional
        number of galaxy eigen spectra, by default 5
    agns : int, optional
        number of qso eigen spectra, by default 10

    Returns
    -------
    data cube
        rebbined spectra and derived host galaxy
    """
    if galaxy > 10 or agns > 50:
        print(
            "number of galaxy eigenspectra has to be less than 10 and QSO eigenspectra less than 50"
        )

    else:

        glx = fits.open(pathEigen + "gal_eigenspec_Yip2004.fits")
        glx = glx[1].data
        gl_wave = glx["wave"].flatten()
        gl_flux = glx["pca"].reshape(glx["pca"].shape[1], glx["pca"].shape[2])

        qso = fits.open(pathEigen + "qso_eigenspec_Yip2004_global.fits")
        qso = qso[1].data
        qso_wave = qso["wave"].flatten()
        qso_flux = qso["pca"].reshape(qso["pca"].shape[1], qso["pca"].shape[2])

        mask = _maskingEigen(wave)
        wave1 = wave[mask]

        flux, err = spectres(wave1, wave, flux, spec_errs=err)
        qso_prime = []
        g_prime = []
        for i in range(galaxy):
            gix = _resam(wave, gl_wave, gl_flux[i])
            g_prime.append(gix)

        for i in range(agns):
            yi = _resam(wave, qso_wave, qso_flux[i])
            qso_prime.append(yi)
        # self.wave = wave1
        # np.save('gal.npy', g_prime)
        return wave1, flux, err, g_prime, qso_prime
    return


def _resam(x, wv, flx):
    """[summary]

    Args:
        x ([type]): [description]
        wv ([type]): [description]
        flx ([type]): [description]

    Returns:
        [type]: [description]
    """
    mask = _maskingEigen(x)
    y = spectres(x[mask], wv, flx)
    return y


def _fitHost(wave, flux, err, galax, qsos):
    a = _prepare_host(wave, flux, err, galax, qsos)
    A = np.vstack(a[3], a[4]).T
    k = np.linalg.lstsq(A, a[1], rcond=None)[0]


def _calculate_chi(data, er, func, coef):
    diff = pow(data - func, 2.0)
    test_statistic = (diff / pow(er, 2.0)).sum()
    NDF = len(data) - len(coef)

    return test_statistic / NDF


def _fith(hs, ags, data):
    A = np.vstack([hs, ags]).T
    k = np.linalg.lstsq(A, data, rcond=None)[0]
    return k


def _find_nearest(array):
    array = np.asarray(array)
    idx = (np.abs(array - 1)).argmin()
    return idx


def _check_host(hst):
    if np.sum(np.where(hst < 0.0, True, False)) > 100:
        return False
    else:
        return True


def randomize(d):
    flux = d.y + np.random.randn(len(d.y)) * d.staterror
    d = Data1D("AGN", d.x, flux, d.staterror)
    return d

