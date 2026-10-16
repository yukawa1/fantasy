"""
Spectrum container, file readers and preprocessing.

Preprocessing methods return a new Spectrum, so steps can be chained::

    spec = fa.read("spec-0461-51910-0361.fits").deredden(ebv=0.03).to_rest_frame()
"""
import os

import warnings

import numpy as np
import pandas as pd
from astropy.io import fits

from .components import remove_emission_lines
from .extinction import deredden as _deredden, galactic_ebv
from .lines import DATA
from .wavelengths import vac_to_air

__all__ = ("Spectrum", "read")

C_KMS = 299792.458


class Spectrum:
    """
    A 1D spectrum.

    :param wave: wavelengths in Angstrom.
    :param flux: flux density (any units; ``flux_unit`` records the scale,
        e.g. 1e-17 erg/s/cm2/A for SDSS, used only for luminosities).
    :param err: 1-sigma flux uncertainty. Pixels with non-finite or
        non-positive errors are ignored in fits.
    :param z: redshift.
    :param medium: 'vacuum' or 'air' wavelengths.
    :param resolution: instrumental FWHM in km/s (0 if unknown). It is
        removed in quadrature from fitted line widths.
    :param resolution_curve: optional (wavelength, FWHM in km/s) arrays for a
        resolution that changes with wavelength (e.g. DESI); each line then
        uses the resolution at its own wavelength.
    """

    def __init__(self, wave, flux, err=None, z=None, name="spectrum", medium="vacuum",
                 resolution=0.0, flux_unit=1.0, rest_frame=False, ra=None, dec=None,
                 mjd=None, meta=None, resolution_curve=None):
        self.wave = np.asarray(wave, dtype=float)
        self.flux = np.asarray(flux, dtype=float)
        estimate_err = err is None
        if estimate_err:
            err = np.full_like(self.flux, np.nan)
        self.err = np.asarray(err, dtype=float)
        if not (self.wave.shape == self.flux.shape == self.err.shape):
            raise ValueError("wave, flux and err must have the same shape")
        if np.any(np.diff(self.wave) <= 0):
            order = np.argsort(self.wave)
            self.wave, self.flux, self.err = self.wave[order], self.flux[order], self.err[order]
        if medium not in ("air", "vacuum"):
            raise ValueError("medium must be 'air' or 'vacuum'")
        self.z = z
        self.name = name
        self.medium = medium
        self.resolution = float(resolution)
        self.resolution_curve = None if resolution_curve is None else (
            np.asarray(resolution_curve[0], float), np.asarray(resolution_curve[1], float))
        self.flux_unit = flux_unit
        self.rest_frame = rest_frame
        self.ra, self.dec, self.mjd = ra, dec, mjd
        self.meta = dict(meta or {})
        self.user_mask = np.zeros(self.wave.shape, bool)
        self.host = None
        if estimate_err:
            # No errors given: estimate a constant error from the scatter.
            noise = 1.4826 * np.nanmedian(np.abs(np.diff(self.flux))) / np.sqrt(2)
            self.err = np.full_like(self.flux, noise if noise > 0 else 1.0)
            self.meta["err_estimated"] = True

    # ----------------------------------------------------------- basics
    @property
    def good(self):
        """Pixels used in fits: finite flux and error, error > 0, not masked."""
        return (np.isfinite(self.flux) & np.isfinite(self.err) & (self.err > 0)
                & ~self.user_mask)

    @property
    def context(self):
        ctx = {"medium": self.medium, "resolution": self.resolution}
        if self.resolution_curve is not None:
            ctx["resolution_curve"] = self.resolution_curve
        return ctx

    def copy(self, **changes):
        new = Spectrum.__new__(Spectrum)
        new.__dict__ = {k: (v.copy() if isinstance(v, (np.ndarray, dict)) else v)
                        for k, v in self.__dict__.items()}
        new.__dict__.setdefault("resolution_curve", None)
        new.__dict__.update(changes)
        return new

    def __repr__(self):
        frame = "rest" if self.rest_frame else "observed"
        ext = (f", E(B-V)={self.meta['ebv']:.4f} corrected" if "ebv" in self.meta
               else ", Galactic extinction not corrected")
        return (f"Spectrum({self.name!r}, {self.wave[0]:.0f}-{self.wave[-1]:.0f} A {frame} "
                f"({self.medium}), {self.wave.size} pixels, z={self.z}, "
                f"R~{self.resolution:.0f} km/s{ext})")

    # ---------------------------------------------------- preprocessing
    def to_rest_frame(self, z=None):
        """Shift to the rest frame: wave/(1+z), flux and err * (1+z)."""
        if self.rest_frame:
            return self.copy()
        z = self.z if z is None else z
        if z is None:
            raise ValueError("redshift unknown: pass z=...")
        curve = self.resolution_curve
        if curve is not None:                      # widths in km/s do not change
            curve = (curve[0] / (1 + z), curve[1])
        return self.copy(wave=self.wave / (1 + z), flux=self.flux * (1 + z),
                         err=self.err * (1 + z), z=z, rest_frame=True,
                         resolution_curve=curve)

    def deredden(self, ebv=None, r_v=3.1, source="auto", sf11=True):
        """
        Correct for Galactic (Milky Way) extinction with the Fitzpatrick (1999)
        law. Do this once, before measuring fluxes and luminosities.

        :param ebv: E(B-V) in magnitudes. If not given it is looked up at the
            position of the spectrum (see :func:`fantasy_agn.extinction.galactic_ebv`):
            the value in the file (DESI), local SFD98 maps (sfdmap2) or the IRSA
            dust service online, with the Schlafly & Finkbeiner (2011)
            recalibration when ``sf11`` is True.
        :param source: 'auto', 'file', 'sfd' or 'irsa' for the lookup.
        """
        if "ebv" in self.meta:
            warnings.warn(f"{self.name}: already corrected for Galactic extinction "
                          f"(E(B-V) = {self.meta['ebv']:.4f}); not applied again")
            return self.copy()
        if ebv is None:
            ebv, where = galactic_ebv(self.ra, self.dec, source=source, sf11=sf11,
                                      file_value=self.meta.get("ebv_file"))
        else:
            ebv, where = float(ebv), "given"
        obs_wave = self.wave * (1 + self.z) if self.rest_frame else self.wave
        corr = _deredden(obs_wave, np.ones_like(obs_wave), ebv, r_v)
        new = self.copy(flux=self.flux * corr, err=self.err * corr)
        new.meta["ebv"] = ebv
        new.meta["ebv_source"] = where
        return new

    @property
    def dereddened(self):
        """True if Galactic extinction was corrected (``deredden``)."""
        return "ebv" in self.meta

    def crop(self, wmin=None, wmax=None):
        """Keep wavelengths between wmin and wmax."""
        keep = np.ones(self.wave.shape, bool)
        if wmin is not None:
            keep &= self.wave >= wmin
        if wmax is not None:
            keep &= self.wave <= wmax
        new = self.copy()
        for attr in ("wave", "flux", "err", "user_mask"):
            setattr(new, attr, getattr(self, attr)[keep])
        if self.host is not None:
            new.host = self.host[keep]
        return new

    def mask(self, *ranges):
        """Exclude wavelength ranges from fits, e.g. spec.mask((5570, 5585))."""
        new = self.copy()
        for lo, hi in ranges:
            new.user_mask |= (self.wave >= lo) & (self.wave <= hi)
        return new

    def mask_absorption(self, nsigma=3.5, window_kms=1500.0, max_width_kms=1000.0, grow=1):
        """
        Exclude narrow absorption lines from fits: intervening and associated
        absorbers (CIV, SiIV, MgII doublets, ...) that cut into broad emission
        lines and otherwise dominate chi2 (common in CIV and MgII spectra).

        A pixel is part of an absorber when it lies in a run of at least two
        pixels more than 2 sigma below the local continuum (a running median
        over ``window_kms``, recomputed without the absorbers), the deepest
        more than ``nsigma`` below it, and the run is narrower than
        ``max_width_kms``; runs are widened by ``grow`` pixels. In pure noise
        this masks about one pixel in 10^5. Broad absorption (BAL troughs) is
        wider and is not masked. Returns a new spectrum;
        ``spec.meta['absorbers']`` lists the masked ranges.
        """
        from scipy.ndimage import generic_filter, median_filter
        new = self.copy()
        good = self.good
        if good.sum() < 20:
            return new
        idx = np.nonzero(good)[0]
        w, f, e = self.wave[idx], self.flux[idx], self.err[idx]
        dv = np.median(np.diff(np.log(w))) * C_KMS                    # km/s per pixel
        size = max(5, int(round(window_kms / dv)) | 1)
        ref = median_filter(f, size=size, mode="nearest")
        for _ in range(2):                                            # leave the troughs out
            clipped = np.where((f - ref) / e < -2.0, np.nan, f)
            ref = generic_filter(clipped, np.nanmedian, size=size, mode="nearest")
            ref = np.where(np.isfinite(ref), ref, median_filter(f, size=size, mode="nearest"))
        dev = (f - ref) / e
        low = dev < -2.0
        ranges, k = [], 0
        while k < idx.size:
            if not low[k]:
                k += 1
                continue
            j = k
            while j < idx.size and low[j]:
                j += 1
            if j - k >= 2 and dev[k:j].min() < -nsigma and (j - k) * dv <= max_width_kms:
                a, b = max(k - grow, 0), min(j - 1 + grow, idx.size - 1)
                ranges.append((float(w[a]), float(w[b])))
            k = j
        for lo, hi in ranges:
            new.user_mask |= (self.wave >= lo) & (self.wave <= hi)
        new.meta = dict(self.meta, absorbers=ranges)
        return new

    def to_air(self):
        if self.medium == "air":
            return self.copy()
        return self.copy(wave=vac_to_air(self.wave), medium="air")

    def subtract_host(self, n_galaxy=5, n_qso=10):
        """
        Decompose the spectrum into host-galaxy and quasar eigenspectra (Yip et
        al. 2004) with an error-weighted linear fit, and return a new spectrum
        with the host removed (stored in ``.host``). For propagating the host
        uncertainty into line measurements, include ``HostGalaxy`` in the model
        instead.
        """
        if not self.rest_frame:
            raise ValueError("call to_rest_frame() first")
        gal_wave, gal_flux = _eigen("gal_eigenspec_Yip2004.fits", n_galaxy)
        gal = (gal_wave, remove_emission_lines(gal_wave, gal_flux))
        qso = _eigen("qso_eigenspec_Yip2004_global.fits", n_qso)
        wave = self.wave if self.medium == "vacuum" else None
        cols = []
        for ew, ef in (gal, qso):
            ew = ew if wave is not None else vac_to_air(ew)
            cols.extend(np.interp(self.wave, ew, e, left=np.nan, right=np.nan) for e in ef)
        A = np.column_stack(cols)
        use = self.good & np.all(np.isfinite(A), axis=1)
        if use.sum() < 100:
            raise ValueError("spectrum does not overlap the eigenspectra (~3450-8000 A)")
        w = 1 / self.err[use]
        coef = np.linalg.lstsq(A[use] * w[:, None], self.flux[use] * w, rcond=None)[0]
        host = np.where(np.isfinite(A[:, 0]), A[:, :n_galaxy] @ coef[:n_galaxy], 0.0)
        if np.mean(host[use] < 0) > 0.1:
            print("warning: host decomposition gives a negative host over >10% "
                  "of pixels; the host is probably negligible")
        new = self.copy(flux=self.flux - host)
        new.host = host
        new.meta["host_fraction_5100"] = _fraction_at(self.wave, host, self.flux, 5100)
        return new

    # --------------------------------------------------------------- plot
    def plot(self, ax=None, fig=None, width="double", **kw):
        """Plot the spectrum in the journal style; returns the axes."""
        from .plotting import plot_spectrum
        return plot_spectrum(self, ax=ax, fig=fig, width=width, **kw)

    # ------------------------------------------------------------ readers
    @classmethod
    def read(cls, path, format="auto", **kw):
        """
        Read a spectrum file. format: 'auto', 'sdss', 'desi', 'ozdes', 'gama',
        'fits1d' (linear or log-linear WCS in the primary HDU) or 'text' (2-3
        columns: wave, flux, [err]). Extra keywords override what is read (e.g.
        z=0.1). DESI coadd files hold many targets: pass targetid=... or
        index=..., and optionally redrock=... for the redshift (see
        fantasy_agn.desi). OzDES DR2 files hold the stack and every epoch: pass
        epoch=... for one epoch (see fantasy_agn.ozdes).
        """
        if format == "auto":
            format = _detect_format(path)
        if format == "desi":
            from .desi import read as read_desi
            return read_desi(path, **kw)
        if format in ("ozdes", "ozdes-ssa"):
            from . import ozdes
            if format == "ozdes":
                args = ozdes.read_original(path, kw.pop("epoch", None))
            else:
                if kw.pop("epoch", None) is not None:
                    raise ValueError("a public OzDES file holds a single spectrum (no epochs)")
                args = ozdes.read_public(path)
            args.update(kw)
            return cls(**args)
        reader = {"sdss": _read_sdss, "gama": _read_gama, "fits1d": _read_fits1d,
                  "text": _read_text}[format]
        args = reader(path)
        args.update(kw)
        return cls(**args)


read = Spectrum.read


# ------------------------------------------------------------------ helpers
def _eigen(filename, n):
    with fits.open(os.path.join(DATA, filename)) as hdul:
        d = hdul[1].data
        wave = np.asarray(d["wave"], float).ravel()
        pca = np.asarray(d["pca"], float)
        pca = pca.reshape(pca.shape[-2], pca.shape[-1])
    return wave, pca[:n]


def _fraction_at(wave, part, total, at):
    sel = np.abs(wave - at) < 20
    if not sel.any():
        return np.nan
    return float(np.nanmedian(part[sel]) / np.nanmedian(total[sel]))


def _name(path):
    return os.path.splitext(os.path.basename(path))[0]


def _detect_format(path):
    ext = path.lower().rsplit(".", 1)[-1]
    if ext not in ("fits", "fit", "fts"):
        return "text"
    with fits.open(path) as hdul:
        h0 = hdul[0].header
        names = {h.name for h in hdul}
        if "FIBERMAP" in names and ({"B_FLUX", "R_FLUX", "Z_FLUX"} & names):
            return "desi"
        if len(hdul) > 1 and isinstance(hdul[1], fits.BinTableHDU):
            names = [c.lower() for c in hdul[1].columns.names]
            if "loglam" in names and "flux" in names:
                return "sdss"
        from .ozdes import is_original, is_public
        if is_public(hdul):
            return "ozdes-ssa"
        if is_original(hdul):
            return "ozdes"
        data = hdul[0].data
        if ("GAMA" in str(h0.get("SURVEY", "")).upper()
                or ("WMIN" in h0 and data is not None and data.ndim == 2)):
            return "gama"
    return "fits1d"


def _read_sdss(path):
    with fits.open(path) as hdul:
        d = hdul[1].data
        h0 = hdul[0].header
        names = [c.lower() for c in hdul[1].columns.names]
        col = lambda n: np.asarray(d[[c for c in hdul[1].columns.names if c.lower() == n][0]], float)
        wave = 10 ** col("loglam")
        flux = col("flux")
        ivar = col("ivar")
        z = None
        if len(hdul) > 2 and "Z" in [c.upper() for c in hdul[2].columns.names]:
            z = float(hdul[2].data["Z"][0])
        with np.errstate(divide="ignore"):
            # Bad pixels have ivar = 0; they get an infinite error and are skipped.
            err = np.where(ivar > 0, 1 / np.sqrt(np.where(ivar > 0, ivar, 1)), np.inf)
        resolution, curve = 0.0, None
        if "wdisp" in names:
            # wdisp: Gaussian sigma of the line-spread function in pixels; the
            # resolution changes along the spectrum (~170 km/s FWHM at 3800 A,
            # ~130 km/s at 8000 A), so keep the whole curve
            wdisp = col("wdisp")
            dloglam = np.median(np.diff(col("loglam")))
            fwhm_kms = 2.3548 * wdisp * dloglam * np.log(10) * C_KMS
            ok = (wdisp > 0) & np.isfinite(err)
            if ok.any():
                resolution = float(np.median(fwhm_kms[ok]))
                if ok.sum() > 20:
                    from scipy.ndimage import median_filter
                    smooth = median_filter(np.interp(wave, wave[ok], fwhm_kms[ok]), 25, mode="nearest")
                    curve = (wave, smooth)
        return dict(wave=wave, flux=flux, err=err, z=z, name=_name(path), medium="vacuum",
                    resolution=resolution, flux_unit=1e-17, ra=h0.get("PLUG_RA"),
                    dec=h0.get("PLUG_DEC"), mjd=h0.get("MJD"), resolution_curve=curve,
                    meta={"plate": h0.get("PLATEID"), "fiber": h0.get("FIBERID")})


def _read_gama(path):
    with fits.open(path) as hdul:
        h = hdul[0].header
        data = np.asarray(hdul[0].data, float)
        if str(h.get("TELESCOP", "")).startswith("SDSS"):
            wave = 10 ** (h["CRVAL1"] + h["CDELT1"] * np.arange(h["NAXIS1"]))
        else:
            wave = np.linspace(h["WMIN"], h["WMAX"], h["NAXIS1"])
        flux, err = data[0], data[1]
        return dict(wave=wave, flux=flux, err=np.where(np.isfinite(err), err, np.inf),
                    z=h.get("Z"), name=_name(path), medium="air", ra=h.get("RA"),
                    dec=h.get("DEC"))


def _read_fits1d(path):
    with fits.open(path) as hdul:
        h = hdul[0].header
        data = np.asarray(hdul[0].data, float)
    flux = data if data.ndim == 1 else data[0]
    err = None if data.ndim == 1 else data[1]
    pix = np.arange(flux.size) + 1 - h.get("CRPIX1", 1)
    lam = h["CRVAL1"] + h.get("CDELT1", h.get("CD1_1", 1)) * pix
    wave = 10 ** lam if h.get("DC-FLAG", 0) == 1 or h["CRVAL1"] < 5 else lam
    return dict(wave=wave, flux=flux, err=err, z=h.get("Z"), name=_name(path),
                medium="air" if "AIR" in str(h.get("CTYPE1", "")).upper() else "vacuum")


def _read_text(path):
    df = pd.read_csv(path, sep=None, engine="python", comment="#", header=None)
    if not np.issubdtype(df.dtypes.iloc[0], np.number):
        df = pd.read_csv(path, sep=None, engine="python", comment="#")
    arr = df.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    err = arr[:, 2] if arr.shape[1] > 2 else None
    return dict(wave=arr[:, 0], flux=arr[:, 1], err=err, name=_name(path), medium="air")
