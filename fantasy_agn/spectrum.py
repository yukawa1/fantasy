"""
Spectrum container, file readers and preprocessing.

Preprocessing methods return a new Spectrum, so steps can be chained::

    spec = fa.read("spec-0461-51910-0361.fits").deredden(ebv=0.03).to_rest_frame()
"""
import datetime
import os
import re
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
        grid = self.pixel_grid
        if grid is not None:
            ctx["pixel"] = grid
        return ctx

    @property
    def pixel_grid(self):
        """
        (wavelength, pixel width) of every pixel, for line profiles averaged over the pixels (as
        the spectrograph records them) instead of taken at the pixel centres; None with
        FANTASY_PIXEL=0 or fewer than two pixels. The width of a pixel is the smaller distance to
        its neighbours, so gaps in the grid do not widen the pixels next to them.
        """
        import os
        w = np.asarray(self.wave, float)
        if os.environ.get("FANTASY_PIXEL", "1") == "0" or w.size < 2 or np.any(np.diff(w) <= 0):
            return None
        d = np.diff(w)
        return w, np.minimum(np.concatenate([d[:1], d]), np.concatenate([d, d[-1:]]))

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

    # observed-frame telluric bands (O2 B band, H2O, O2 A band) and strong sky lines, A
    TELLURIC = ((6860.0, 6960.0), (7160.0, 7340.0), (7590.0, 7700.0))
    SKY_LINES = (5577.3, 5889.9, 5895.9, 6300.3, 6363.8)

    def mask_sky(self, telluric=True, sky_lines=True, width=5.0):
        """
        Mask the telluric absorption bands and the strongest night-sky emission
        lines (observed frame; their residuals after the standard corrections
        dominate chi2 of high-S/N galaxy spectra). Works before and after
        ``to_rest_frame``. Returns a new spectrum (masked pixels get an infinite error).
        """
        z = (self.z or 0.0) if self.rest_frame else 0.0
        w = self.wave * (1 + z)
        bad = np.zeros(w.size, bool)
        if telluric:
            for a, b in self.TELLURIC:
                bad |= (w > a) & (w < b)
        if sky_lines:
            for line in self.SKY_LINES:
                bad |= np.abs(w - line) < width
        return self.copy(err=np.where(bad, np.inf, self.err))

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
    def read(cls, path, format="auto", ra=None, dec=None, coords=None, **kw):
        """
        Read a spectrum file. format: 'auto', 'sdss', 'desi', 'ozdes', 'gama',
        'fits1d' (linear or log-linear WCS in the primary HDU) or 'text' (2-3
        columns: wave, flux, [err]). Extra keywords override what is read (e.g.
        z=0.1). DESI coadd files hold many targets: pass targetid=... or
        index=..., and optionally redrock=... for the redshift (see
        fantasy_agn.desi). OzDES DR2 files hold the stack and every epoch: pass
        epoch=... for one epoch (see fantasy_agn.ozdes).

        :param ra, dec: sky position, overriding the file (needed for the Galactic
            extinction of files without coordinates, e.g. text files): degrees or
            sexagesimal strings ('12:34:56.7', '+12:34:56'); see :func:`parse_coordinates`.
        :param coords: the position as one string ('12:34:56.7 +12:34:56') or a SkyCoord.

            spec = fa.read("spectrum.txt", z=0.123, ra="12:34:56.7", dec="+12:34:56").deredden()
        """
        position = None
        if coords is not None:
            position = parse_coordinates(coords)
        elif ra is not None or dec is not None:
            if ra is None or dec is None:
                raise ValueError("give both ra and dec (or coords=...)")
            position = parse_coordinates(ra, dec)
        spec = cls._read(path, format, **kw)
        if position is not None:
            spec.ra, spec.dec = position
            spec.meta["coordinates_source"] = "given"
        return spec

    def with_flux_unit(self, unit):
        """
        A copy whose flux values are declared to be in units of ``unit`` erg/s/cm2/A (the values are not
        changed), e.g. ``spec.with_flux_unit(1e-16)`` for a text file in units of 1e-16 erg/s/cm2/A.
        Luminosities and black-hole masses need the flux unit.
        """
        unit = float(unit)
        if not unit > 0:
            raise ValueError("the flux unit must be positive, e.g. 1e-17")
        new = self.copy()
        new.flux_unit = unit
        new.meta["flux_unit_source"] = "given"
        new.meta.pop("flux_unit_unknown", None)
        return new

    def with_resolution(self, fwhm=None, R=None):
        """
        A copy with the instrumental resolution set: ``fwhm`` in km/s (FWHM of the line-spread function),
        or the resolving power ``R`` = lambda / FWHM, e.g. ``spec.with_resolution(R=2000)`` (150 km/s).
        It replaces the value (and the resolution curve) of the file. Fitted line widths are then
        intrinsic: the resolution is removed in quadrature. Text and plain FITS files carry no
        resolution (0): the widths then include the instrumental broadening.
        """
        if (fwhm is None) == (R is None):
            raise ValueError("give the resolution as fwhm=... (km/s) or R=... (lambda / FWHM)")
        value = float(fwhm) if R is None else C_KMS / float(R)
        if not (np.isfinite(value) and 0 <= value < 5000):
            raise ValueError("the instrumental FWHM must be between 0 and 5000 km/s (R above 60)")
        new = self.copy(resolution=value, resolution_curve=None)
        new.meta["resolution_source"] = "given"
        return new

    def pixel_kms(self, wave=None):
        """Pixel size in km/s at ``wave`` (default: the median over the spectrum)."""
        step = np.gradient(self.wave) / self.wave * C_KMS
        if wave is None:
            return float(np.median(step))
        return np.interp(wave, self.wave, step)

    def with_coordinates(self, ra, dec=None):
        """A copy at the given sky position (degrees, sexagesimal or SkyCoord; see :func:`parse_coordinates`)."""
        new = self.copy()
        new.ra, new.dec = parse_coordinates(ra, dec)
        new.meta["coordinates_source"] = "given"
        return new

    @classmethod
    def _read(cls, path, format="auto", **kw):
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
        if format == "text":
            args = _read_text(path, autoscale="flux_unit" not in kw)
        else:
            args = {"sdss": _read_sdss, "gama": _read_gama, "fits1d": _read_fits1d}[format](path)
        args.update(kw)
        return cls(**args)


read = Spectrum.read


_DATE = re.compile(r"(?<!\d)((?:19|20)\d\d)[-_]?(0[1-9]|1[0-2])[-_]?(0[1-9]|[12]\d|3[01])(?!\d)")
_MJD0 = datetime.date(1858, 11, 17)


def date_from_name(path):
    """'YYYY-MM-DD' from a date (YYYYMMDD, YYYY-MM-DD or YYYY_MM_DD) in a file name, else None."""
    m = _DATE.search(os.path.basename(str(path)))
    if not m:
        return None
    try:
        return datetime.date(*map(int, m.groups())).isoformat()
    except ValueError:
        return None


def mjd_from_name(path):
    """MJD (at 0h) of the date in a file name, e.g. 'tAT2024zkr_20241031_0.txt' -> 60614.0; None without one."""
    date = date_from_name(path)
    return None if date is None else float((datetime.date.fromisoformat(date) - _MJD0).days)


def mjd_to_date(mjd):
    """'YYYY-MM-DD' of an MJD (None for None)."""
    if mjd is None or not np.isfinite(mjd):
        return None
    return (_MJD0 + datetime.timedelta(days=float(mjd))).isoformat()


def parse_coordinates(ra, dec=None):
    """
    (RA, Dec) in degrees from: numbers (degrees); sexagesimal strings
    ('12:34:56.7', '+12:34:56' or '12h34m56.7s', '+12d34m56s'); one string with
    both ('12:34:56.7 +12:34:56', '188.7362 12.5822'); or an astropy SkyCoord.
    A sexagesimal RA is in hours, a sexagesimal Dec in degrees.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    if isinstance(ra, SkyCoord):
        c = ra.icrs
        return float(c.ra.deg), float(c.dec.deg)
    if dec is None:
        if not isinstance(ra, str):
            ra, dec = ra
        else:
            parts = ra.replace(",", " ").split()
            if len(parts) == 2:
                ra, dec = parts
            elif len(parts) == 6:
                ra, dec = " ".join(parts[:3]), " ".join(parts[3:])
            else:
                raise ValueError(f"cannot read coordinates from {ra!r}: give RA and Dec")

    def number(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    r, d = number(ra), number(dec)
    if r is not None and d is not None:
        c = SkyCoord(r * u.deg, d * u.deg)
    else:
        c = SkyCoord(str(ra), str(dec), unit=(u.hourangle, u.deg))
    if not -90 <= c.dec.deg <= 90:
        raise ValueError(f"declination {c.dec.deg} is outside -90..90 degrees")
    return float(c.ra.deg), float(c.dec.deg)


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
        if (any("GAMA" in str(h0.get(k, "")).upper() for k in ("SURVEY", "ORIGIN"))
                or (data is not None and data.ndim == 2 and ("WMIN" in h0 or "CATAID" in h0 or "GAMANAME" in h0))):
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
    """
    A GAMA spectrum (GAMA/AAOmega and the other surveys of the GAMA data releases): rows of flux, error, ...
    in the primary HDU. The flux unit is taken from the description of the first row (GAMA/AAOmega:
    'Flux-calibrated spectrum in 10^-17 erg/s/cm^2/A') or BUNIT; spectra that are not flux-calibrated get
    flux_unit None. The date is UTMJD (or MJD, DATE-OBS).
    """
    with fits.open(path) as hdul:
        h = hdul[0].header
        data = np.asarray(hdul[0].data, float)
    n = data.shape[-1]
    if str(h.get("TELESCOP", "")).startswith("SDSS"):
        wave = 10 ** (h["CRVAL1"] + h["CDELT1"] * np.arange(n))
    elif "WMIN" in h and "WMAX" in h:
        wave = np.linspace(h["WMIN"], h["WMAX"], n)
    else:
        step = h.get("CD1_1", h.get("CDELT1", 1.0))
        wave = h["CRVAL1"] + step * (np.arange(n) + 1 - h.get("CRPIX1", 1))
    flux = data[0] if data.ndim == 2 else data
    err = data[1] if data.ndim == 2 and data.shape[0] > 1 else None
    meta = {k.lower(): h[k] for k in ("SPECID", "CATAID", "GAMANAME", "SURVEY", "NQ", "PROB", "SN") if k in h}
    row1 = f"{h.get('ROW1', '')} {h.comments['ROW1'] if 'ROW1' in h else ''}"
    unit, source = None, None
    for text, where in ((row1, "ROW1"), (str(h.get("BUNIT", "")), "BUNIT")):
        unit = flux_unit_from_text(text)
        if unit:
            source = f"{where}: {text.strip()[:80]}"
            break
    if unit:
        meta["flux_unit_source"] = source
    else:
        meta["flux_unit_unknown"] = ("no flux unit in the header (spectra of some surveys in the GAMA releases are "
                                     "not flux-calibrated); give it with flux_unit=... if it is known")
    mjd = h.get("UTMJD", h.get("MJD"))
    if mjd is None and h.get("DATE-OBS"):
        try:
            mjd = (datetime.date.fromisoformat(str(h["DATE-OBS"])[:10]) - _MJD0).days
        except ValueError:
            mjd = None
    return dict(wave=wave, flux=flux, err=None if err is None else np.where(np.isfinite(err), err, np.inf),
                z=h.get("Z"), name=str(h.get("SPECID", "")).strip() or _name(path), medium="air",
                ra=h.get("RA"), dec=h.get("DEC"), flux_unit=unit, mjd=float(mjd) if mjd is not None else None,
                meta=meta)


def _read_fits1d(path):
    with fits.open(path) as hdul:
        h = hdul[0].header
        data = np.asarray(hdul[0].data, float)
    flux = data if data.ndim == 1 else data[0]
    err = None if data.ndim == 1 else data[1]
    pix = np.arange(flux.size) + 1 - h.get("CRPIX1", 1)
    lam = h["CRVAL1"] + h.get("CDELT1", h.get("CD1_1", 1)) * pix
    wave = 10 ** lam if h.get("DC-FLAG", 0) == 1 or h["CRVAL1"] < 5 else lam
    bunit = str(h.get("BUNIT", ""))
    unit = flux_unit_from_text(bunit)
    meta = {}
    if unit is None and "erg" in bunit.lower() and np.nanmedian(np.abs(flux)) < 1e-10:
        unit = 1.0                                             # BUNIT in erg/s/cm2/A, physical values
    if unit is not None:
        meta["flux_unit_source"] = f"BUNIT = {bunit}"
    else:
        meta["flux_unit_unknown"] = "no flux unit (BUNIT) in the header; give it with flux_unit=..."
    return dict(wave=wave, flux=flux, err=err, z=h.get("Z"), name=_name(path), flux_unit=unit, meta=meta,
                medium="air" if "AIR" in str(h.get("CTYPE1", "")).upper() else "vacuum")


def _text_separator(path):
    """',', tab or ';' if the data lines use one, else any run of whitespace."""
    with open(path, errors="replace") as fh:
        lines = [ln for ln in (next(fh, "") for _ in range(50)) if ln.strip() and not ln.lstrip().startswith("#")]
    for sep in (",", "\t", ";"):
        if lines and all(sep in ln for ln in lines[1:] or lines):
            return sep
    return r"\s+"


def _local_noise(flux, half=25):
    """Noise per pixel from the scatter of neighbouring pixels in a running window (no error column)."""
    d = np.abs(np.diff(flux)) / np.sqrt(2)
    d = np.concatenate([d[:1], d])
    out = np.empty_like(flux)
    for i in range(flux.size):
        w = d[max(0, i - half):i + half + 1]
        out[i] = 1.4826 * np.nanmedian(w)
    good = np.isfinite(out) & (out > 0)
    if good.any():
        out[~good] = np.nanmedian(out[good])
    return out


_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
_UNIT_SCALE = [re.compile(p, re.I) for p in (
    r"(?<![\w.])(\d+(?:\.\d+)?)?\s*(?:x|×|\*|\\times)?\s*10\s*(?:\^|\*\*)\s*[({]?\s*(-\d+)",   # 1e-16 as 10^-16, 2.5x10**-17
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*[eE]\s*(-\d+)(?![\d.])",                                          # 1e-16, 1.0E-17
    r"(?<![\w.])()10(-\d+)(?![\d.])",                                                                   # 10-16 (after removing ^)
    r"[x×]\s*()10\s*(-\d+)(?![\d.])")                                                                   # x10⁻¹⁶
]


def flux_unit_from_text(text):
    """
    The flux unit (erg/s/cm2/A per unit of the flux values) written in a header line or keyword, e.g.
    '1e-16 erg/s/cm2/A', 'flux [10^-17 erg s-1 cm-2 A-1]', 'F_lambda (x10⁻¹⁶)'; None if there is none.
    Only scales from 1e-25 to 1e-10 are taken (flux densities of astronomical sources).
    """
    t = str(text).translate(_SUPERSCRIPT).replace("−", "-")
    for rx in _UNIT_SCALE:
        for m in rx.finditer(t):
            mant, exp = m.group(1), int(m.group(2))
            if -25 <= exp <= -10:
                return float(mant or 1.0) * 10.0 ** exp
    return None


def _header_lines(path, n=40):
    """The comment and header lines at the top of a text file."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for _ in range(n):
            line = fh.readline()
            if not line:
                break
            body = line.strip()
            if not body:
                continue
            first = re.split(r"[\s,;]+", body.lstrip("#"))[0]
            try:
                float(first)
                if not body.startswith("#"):
                    break                                   # the data begin
            except ValueError:
                pass
            out.append(body)
    return out


def _read_text(path, autoscale=True):
    sep = _text_separator(path)
    df = pd.read_csv(path, sep=sep, engine="python", comment="#", header=None)
    if not pd.api.types.is_numeric_dtype(df.dtypes.iloc[0]):          # a header line
        df = pd.read_csv(path, sep=sep, engine="python", comment="#")
    arr = df.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    arr = arr[:, np.isfinite(arr).any(axis=0)]        # columns that are empty (e.g. double separators)
    arr = arr[np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])]
    if arr.shape[1] < 2:
        raise ValueError(f"{os.path.basename(path)}: need at least two numeric columns (wavelength, flux)")
    wave, flux = arr[:, 0], arr[:, 1]
    err = arr[:, 2] if arr.shape[1] > 2 else None
    meta = {}
    if err is not None and np.nanmax(err) - np.nanmin(err) <= 1e-9 * abs(np.nanmax(err)):
        # one value in every pixel: a placeholder, not a measured error
        err = None
        meta["err_column_constant"] = True
    if err is None:
        order = np.argsort(wave)
        e = np.empty_like(flux)
        e[order] = _local_noise(flux[order])
        err = e
        meta["err_estimated"] = "local scatter (no error column)"
    flux_unit = None
    scale = np.nanmedian(np.abs(flux))
    header = [ln for ln in _header_lines(path) if re.search(r"flux|f_?lam|erg|unit|w/m|f\(", ln, re.I)]
    stated = next(((u, ln) for ln in header for u in [flux_unit_from_text(ln)] if u), None)
    if not autoscale:
        pass                                                   # the caller gives flux_unit
    elif stated:
        # the values are in units of the scale written in the header (e.g. '1e-16 erg/s/cm2/A')
        flux_unit = stated[0]
        meta["flux_unit_source"] = f"file header: {stated[1][:120]}"
    elif np.isfinite(scale) and 0 < scale < 1e-10:
        # physical units (erg/s/cm2/A): store in units of 1e-17 as for SDSS (numerically safer)
        flux, err, flux_unit = flux * 1e17, err * 1e17, 1e-17
        meta["flux_scaled"] = "values multiplied by 1e17; flux_unit = 1e-17"
        meta["flux_unit_source"] = "values in erg/s/cm2/A"
    else:
        # neither physical values nor a unit in the header: luminosities need flux_unit=...
        meta["flux_unit_unknown"] = ("the flux unit is not in the file; give it with flux_unit=..., e.g. "
                                     "fa.read(path, flux_unit=1e-16) for values in units of 1e-16 erg/s/cm2/A")
    return dict(wave=wave, flux=flux, err=err, name=_name(path), medium="air", flux_unit=flux_unit, meta=meta)
