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
    def vac_to_air(self):
        """
        Convert vacuum to air wavelengths
        :param lam_vac - Wavelength in Angstroms
        :return: lam_air - Wavelength in Angstroms
        """
        self.wave=self.wave/_wave_convert(self.wave)
        
    def fit_host_sdss(self, mask_host=False, custom=False, mask_list=[]):
        """
        The fit_host_sdss function fits a 5 host galaxy eigenspectra and an 10 AGNs eigenspectra to the observed spectrum.
        It takes as input:
        mask_host (bool): If True, it masks the host emission lines using _host_mask function. Default is False.
        custom (bool): If True, it masks user defined emission lines using _custom_mask function. Default is False. 
        mask_list ([float]): A list of wavelengths in angstroms that will be masked if custom=True .Default is empty list [].
        
             Returns: 
        
                self.(wave,flux,err) : The wavelength array , flux array and error array after fitting for host galaxy and AGN components respectively.
        
        :param self: Used to Access variables that belongs to the class.
        :param mask_host=False: Used to Mask the host.
        :param custom=False: Used to Mask the host.
        :param mask_list=[]: Used to Mask the data points that are not used in the fit.
        :return: The host and the agn component.
        """
        

        self.wave_old = self.wave
        self.flux_old = self.flux
        self.err_old = self.err
        a = _prepare_host(self.wave, self.flux, self.err, 5, 10)
        k = _fith(a[3], a[4], a[1])
        host = 0
        for i in range(5):
            host += k[i] * a[3][i]
        if mask_host and custom==False:
            host= _host_mask(a[0], host)
        if mask_host and custom==True:
            host= _custom_mask(a[0], host, mask_list)
        ag_r = 0
        for i in range(10):
            ag_r += k[5 + i] * a[4][i]
        if _check_host(host) == True and _check_host(ag_r) == True:
            self.host = host
            self.flux = a[1] - host
            self.agn=ag_r
            self.wave = a[0]
            self.err = a[2]
            plt.figure(figsize=(18, 6))
            plt.plot(a[0], a[1])
            plt.plot(a[0], host, label="host")
            plt.plot(a[0], ag_r, label="AGN")
            plt.plot(a[0], host+ag_r, label="FIT")
        else:
            print('Host contribution is negliglable')
            

    def fit_host(self, mask_host=False, custom=False, mask_list=[]):
        """
        The fit_host function fits a host galaxy and an AGN to the observed spectrum.
        It takes as input:
        mask_host (bool): If True, it masks the host emission lines using _host_mask function. Default is False.
        custom (bool): If True, it masks user defined emission lines using _custom_mask function. Default is False. 
        mask_list ([float]): A list of wavelengths in angstroms that will be masked if custom=True .Default is empty list [].
        
             Returns: 
        
                self.(wave,flux,err) : The wavelength array , flux array and error array after fitting for host galaxy and AGN components respectively.
        
        :param self: Used to Access variables that belongs to the class.
        :param mask_host=False: Used to Mask the host.
        :param custom=False: Used to Mask the host.
        :param mask_list=[]: Used to Mask the data points that are not used in the fit.
        :return: The host and the agn component.
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
            if mask_host and custom==False:
                host= _host_mask(a[0], host)
            if mask_host and custom==True:
                host= _custom_mask(a[0], host, mask_list)
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
            self.host=np.zeros(len(a[0]))
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
            self.host = HOST[idx]
            self.flux = a[1] - HOST[idx]
            self.agn=AGN[idx]
            self.wave = a[0]
            self.err = a[2]


            plt.legend()
            plt.savefig(self.name+'_host.pdf') 
        
        

    def restore(self):
        """
        Restore the spectra (wave, flux and error) to the one before host removal
        """
        
        self.wave = self.wave_old
        self.flux = self.flux_old
        self.err = self.err_old

    def plot(self):
        fig = px.line(x=self.wave, y=self.flux)
        fig.show()

    def fit(self, model, ntrial=1):
        """
        The fit function fits a model to the data. 
        It returns a tuple of (model, fit results).
        
        :param self: Used to Reference the class object.
        :param model: Used to Define the model that is used in the fit.
        :param ntrial=1: Used to Specify the number of times we want to repeat the fit.
        :return: The results of the fit.
        """
        stat=Chi2()
        method=LevMar()
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
        """
        The bootstrap function takes a data set and returns the best fit parameters for that dataset.
        It does this by generating random datasets from the original dataset using a poisson distribution with mean equal to 1.
        The function then fits these datasets and stores the resulting parameters in an array of dictionaries, which is returned.
        
        :param self: Used to Access variables that belongs to the class.
        :param ntrails: Used to Specify the number of bootstrap trials.
        :return: A pandas dataframe with the fitted parameters.
        """
        d = self.d
        rande = []
        for i in range(ntrails):
            rande.append(_randomize(d))
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
        x=_vac_to_air(x)
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

class make_spec(spectrum):
    
    def __init__(self, wav, fl, er, ra = None, dec = None, z = None):
        super(make_spec,self).__init__()
        self.wave = wav
        self.flux = fl
        self.err = er
        self.ra = ra
        self.dec = dec
        self.z = z


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
        gl_wave=_vac_to_air(gl_wave)
        gl_flux = glx["pca"].reshape(glx["pca"].shape[1], glx["pca"].shape[2])

        qso = fits.open(pathEigen + "qso_eigenspec_Yip2004_global.fits")
        qso = qso[1].data
        qso_wave = qso["wave"].flatten()
        qso_wave=_vac_to_air(qso_wave)
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
    W=1/data**2
    Aw = A * np.sqrt(W[:,np.newaxis])
    dataw=data * np.sqrt(W)
    k = np.linalg.lstsq(Aw, dataw, rcond=None)[0]
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


def _randomize(d):
    flux = d.y + np.random.randn(len(d.y)) * d.staterror
    d = Data1D("AGN", d.x, flux, d.staterror)
    return d


def _host_mask(wave, host):
    line_mask = np.where( (wave < 4970.) & (wave > 4950.) |
    (wave < 5020.) & (wave > 5000.) | 
    (wave < 6590.) & (wave > 6540.) |
    (wave < 3737.) & (wave > 3717.) |
    (wave < 4872.) & (wave > 4852.) |
    (wave < 4350.) & (wave > 4330.) |
    (wave < 6750.) & (wave > 6600.) |
    (wave < 4111.) & (wave > 4091.), True, False)
                     
    f = interpolate.interp1d(wave[~line_mask],host[~line_mask], bounds_error = False, fill_value = 0)
    return f(wave)
def _custom_mask(wave, host, mask_list):
    line_mask=np.where(wave, False, True)
    for m in mask_list:
        mask = np.where( (wave < m[1]) & (wave > m[0]), True, False)
        line_mask += mask
    f = interpolate.interp1d(wave[~line_mask],host[~line_mask], bounds_error = False, fill_value = 0)
    return f(wave)

def _wave_convert(lam):
    """
    Convert between vacuum and air wavelengths using
    equation (1) of Ciddor 1996, Applied Optics 35, 1566
        http://doi.org/10.1364/AO.35.001566

    :param lam - Wavelength in Angstroms
    :return: conversion factor

    """
    lam = np.asarray(lam)
    sigma2 = (1e4/lam)**2
    fact = 1 + 5.792105e-2/(238.0185 - sigma2) + 1.67917e-3/(57.362 - sigma2)
    return fact
def _vac_to_air(lam_vac):
    """
    Convert vacuum to air wavelengths

    :param lam_vac - Wavelength in Angstroms
    :return: lam_air - Wavelength in Angstroms

    """
    return lam_vac/_wave_convert(lam_vac)