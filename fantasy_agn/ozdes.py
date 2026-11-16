"""
OzDES spectra (2dF/AAOmega on the AAT, Lidman et al. 2020, DR2).

Two kinds of files exist:

* **Original DR2 files** (public at VizieR/CDS, J/MNRAS/496/19, or the Data
  Central bulk download): one multi-extension FITS file per object with the
  stacked spectrum, its variance and bad-pixel mask (HDU 0-2) and the same
  three for every exposure (HDU 3i+3, 3i+4, 3i+5, by date; MJD in ``UTMJD``).

      path = fa.ozdes.download("SVA1_COADD-2937976373", "ozdes/")[0]
      spec = fa.read(path)                                      # the stack
      spec = fa.read(path, epoch=4)                             # one exposure
      nights = fa.ozdes.read_epochs(path, skip_qc=True, combine="night")
      me = fa.fit_epochs([n.deredden().to_rest_frame() for n in nights], fa.presets.hbeta(),
                         shared=("narrow.*", "outflow.*", "feii.fwhm", "feii.shift"))

* **Spectra from the Data Central SSA service**: one file per spectrum (the
  stack or one exposure), flux only. FANTASY estimates the noise from the
  spectrum itself (``spec.meta["err_estimated"]``; good to ~20% for stacks, an
  underestimate by ~25% for single exposures). Use them only if the original
  file is not available: ``fa.ozdes.download(name, folder, source="ssa")``.

      table = fa.ozdes.search("SVA1_COADD-2940863213")           # or ra=, dec=

Notes. OzDES spectra are **not flux calibrated** ("normalised counts"), not
even relatively between epochs: ``flux_unit`` is None, so luminosities and
black-hole masses are not computed; widths, shifts, EWs and line ratios are
fine. For several epochs use ``fa.fit_epochs`` with intercalibration, which
puts the epochs on a common scale with the narrow lines. Wavelengths are
heliocentric and treated as vacuum, consistent with the catalogue redshifts.
The resolution is R ~ 1400 (about 210 km/s FWHM).
"""
import io
import os
import urllib.parse
import urllib.request

import numpy as np
from astropy.io import fits

from . import _net

__all__ = ("read_epochs", "search", "download", "SSA_URL")

SSA_URL = "https://datacentral.org.au/vo/ssa/query"
VIZIER_URL = "https://cdsarc.cds.unistra.fr/VizieR/ftp/J/MNRAS/496/19"
COLLECTION = "ozdes_dr2"
RESOLUTION_KMS = 210.0          # 2dF/AAOmega with the 580V/385R gratings, R ~ 1400


# ----------------------------------------------------------------- detection
def is_original(hdul):
    """An original DR2 file: stack + variance + mask, then three HDUs per epoch."""
    h0 = hdul[0].header
    return (len(hdul) >= 3 and (len(hdul) - 3) % 3 == 0
            and ("VERCOADD" in h0 or (len(hdul) >= 6 and "UTMJD" in hdul[3].header)))


def is_public(hdul):
    """A spectrum from the Data Central SSA service."""
    return str(hdul[0].header.get("SURVEY", "")).lower().startswith("ozdes")


def _get(url, timeout, tries=4):
    """GET with retries: the Data Central services answer 500 now and then."""
    import time
    import urllib.error
    for k in range(tries):
        try:
            with _net.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            code = getattr(exc, "code", None)
            if k == tries - 1 or (code is not None and code < 500):
                raise
            time.sleep(2.0 * (k + 1))


# ------------------------------------------------------------------- reading
def _wave(h, n):
    # FITS convention: pixel i (0-based) is at CRVAL1 + (i + 1 - CRPIX1) CDELT1
    return h["CRVAL1"] + (np.arange(n) + 1 - h.get("CRPIX1", 1)) * h["CDELT1"]


def _redshift(h):
    z = h.get("Z")
    return None if z is None or z <= -9 or not np.isfinite(z) else float(z)


def _err_from_variance(var, mask=None):
    var = np.asarray(var, float)
    bad = ~np.isfinite(var) | (var <= 0)
    if mask is not None:
        bad |= np.asarray(mask) != 0
    return np.where(bad, np.inf, np.sqrt(np.where(bad, 1.0, var)))


def estimate_noise(flux, window=25):
    """
    Noise per pixel estimated from the spectrum itself (DER_SNR, Stoehr et al.
    2008: 1.4826/sqrt(6) median|2f_i - f_(i-2) - f_(i+2)|), in a running window,
    for spectra delivered without errors. Lines broader than a few pixels do not
    bias it; unresolved features and sky residuals raise it locally.
    """
    f = np.asarray(flux, float)
    good = np.isfinite(f) & (f != 0)
    d = np.full(f.size, np.nan)
    d[2:-2] = np.abs(2 * f[2:-2] - f[:-4] - f[4:])
    d[~good] = np.nan
    d[2:-2][~(good[:-4] & good[4:])] = np.nan
    from scipy.ndimage import generic_filter
    sigma = 1.482602 / np.sqrt(6.0) * generic_filter(d, np.nanmedian, size=window, mode="nearest")
    ok = np.isfinite(sigma) & (sigma > 0)
    if ok.any():
        sigma = np.interp(np.arange(f.size), np.nonzero(ok)[0], sigma[ok])
    return np.where(good & np.isfinite(sigma) & (sigma > 0), sigma, np.inf)


def read_original(path, epoch=None):
    """
    Arguments for Spectrum from an original DR2 file: the stack (epoch None) or
    epoch ``epoch`` (0 = the first, in date order).
    """
    name = os.path.splitext(os.path.basename(path))[0]
    with fits.open(path) as hdul:
        h0 = hdul[0].header
        n_epochs = (len(hdul) - 3) // 3
        if epoch is None:
            i, mjd, meta = 0, None, {"n_epochs": n_epochs}
        else:
            epoch = int(epoch)
            if not -n_epochs <= epoch < n_epochs:
                raise IndexError(f"{name} has {n_epochs} epochs (0-{n_epochs - 1})")
            epoch %= n_epochs
            i = 3 * epoch + 3
            he = hdul[i].header
            mjd = he.get("UTMJD")
            meta = {k.lower(): he.get(k) for k in ("QC", "INCOADD", "EXPOSED", "FLXSCALE", "SOURCEF",
                                                   "DICHROIC") if k in he}
            meta["epoch"] = epoch
            name = f"{name}_e{epoch:02d}"
        flux = np.asarray(hdul[i].data, float)
        err = _err_from_variance(hdul[i + 1].data, hdul[i + 2].data)
        wave = _wave(hdul[i].header if "CRVAL1" in hdul[i].header else h0, flux.size)
        meta.update(survey="OzDES DR2", qop=h0.get("QOP"), source=h0.get("SOURCE"))
    err[~np.isfinite(flux)] = np.inf
    flux = np.where(np.isfinite(flux), flux, 0.0)
    return dict(wave=wave, flux=flux, err=err, z=_redshift(h0), name=name, medium="vacuum",
                resolution=RESOLUTION_KMS, flux_unit=None, ra=h0.get("RA"), dec=h0.get("DEC"),
                mjd=mjd, meta=meta)


def read_public(path):
    """Arguments for Spectrum from a Data Central SSA file (flux only)."""
    with fits.open(path) as hdul:
        h = hdul[0].header
        flux = np.asarray(hdul[0].data, float)
    stacked = (h.get("TXEL") or 1) > 1
    obj = str(h.get("OBJECT") or os.path.splitext(os.path.basename(path))[0])
    mjd = None if stacked else h.get("TMID")
    err = estimate_noise(flux)
    err[~np.isfinite(flux)] = np.inf
    flux = np.where(np.isfinite(flux), flux, 0.0)
    name = obj if stacked else f"{obj}_{mjd:.3f}" if mjd else obj
    meta = {"survey": "OzDES DR2 (Data Central SSA)", "err_estimated": True,
            "stacked": stacked, "n_epochs": h.get("TXEL"), "exptime": h.get("EXPTIME")}
    return dict(wave=_wave(h, flux.size), flux=flux, err=err, z=_redshift(h), name=name,
                medium="vacuum", resolution=RESOLUTION_KMS, flux_unit=None, ra=h.get("RA"),
                dec=h.get("DEC"), mjd=mjd, meta=meta)


def read_epochs(path, coadded_only=False, skip_qc=False, combine=None):
    """
    All epochs of an original DR2 file as a list of Spectrum (observed frame,
    in date order), ready for ``fa.fit_epochs`` after ``to_rest_frame()``.

    :param coadded_only: only the epochs the OzDES pipeline put into the stack.
    :param skip_qc: leave out epochs with a quality-control flag (QC keyword).
    :param combine: 'night' combines the exposures of each night (usually
        three of 40 min; MJDs within 0.5 d) into one spectrum, weighted by
        inverse variance after the pipeline's scale factors (FLXSCALE). The
        line cannot change within a night, and single exposures are often too
        noisy for a multi-component fit; OzDES reverberation mapping uses
        nightly spectra too.
    """
    from .spectrum import Spectrum
    with fits.open(path) as hdul:
        if not is_original(hdul):
            raise ValueError(f"{path} is not an original OzDES DR2 file (multi-epoch); public "
                             "SSA files hold one spectrum each: read them with fa.read")
        n = (len(hdul) - 3) // 3
    out = []
    for k in range(n):
        s = Spectrum(**read_original(path, k))
        incoadd = str(s.meta.get("incoadd", True)).strip().upper() not in ("F", "FALSE", "0")
        qc = str(s.meta.get("qc") or "").strip()
        if (coadded_only and not incoadd) or (skip_qc and qc and qc.lower() not in ("ok", "none")):
            continue
        out.append(s)
    if combine == "night":
        out = _combine_nights(out)
    elif combine not in (None, "none"):
        raise ValueError("combine must be None or 'night'")
    return out


def _combine_nights(spectra):
    """Inverse-variance mean of the exposures taken within 0.5 d of each other."""
    from .spectrum import Spectrum
    groups, current = [], []
    for s in sorted(spectra, key=lambda s: s.mjd):
        if current and s.mjd - current[0].mjd > 0.5:
            groups.append(current)
            current = []
        current.append(s)
    if current:
        groups.append(current)
    out = []
    for g in groups:
        if len(g) == 1:
            out.append(g[0])
            continue
        num = np.zeros(g[0].wave.size)
        den = np.zeros(g[0].wave.size)
        for s in g:
            c = float(s.meta.get("flxscale") or 1.0)          # puts the exposures on one scale
            w = np.where(np.isfinite(s.err), 1.0 / (c * s.err) ** 2, 0.0)
            num += w * c * s.flux
            den += w
        flux = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
        err = np.where(den > 0, 1 / np.sqrt(np.where(den > 0, den, 1)), np.inf)
        first = g[0]
        mjd = float(np.mean([s.mjd for s in g]))
        meta = dict(first.meta, n_exposures=len(g), exposures=[s.meta.get("epoch") for s in g],
                    exposed=sum(float(s.meta.get("exposed") or 0) for s in g))
        meta.pop("flxscale", None)
        out.append(Spectrum(first.wave, flux, err, z=first.z, name=f"{first.name.rsplit('_e', 1)[0]}_{mjd:.2f}",
                            medium=first.medium, resolution=first.resolution, flux_unit=None,
                            ra=first.ra, dec=first.dec, mjd=mjd, meta=meta))
    return out


# -------------------------------------------------------------------- archive
def search(name=None, ra=None, dec=None, radius=2.0, maxrec=2000, timeout=120):
    """
    OzDES DR2 spectra in the Data Central archive (SSA service), by target name
    and/or position (degrees; radius in arcsec). Returns a pandas DataFrame
    with one row per spectrum: target, kind ('combined' = stack, 'science' =
    one epoch), MJD, redshift and the download URL, sorted by target and MJD.
    """
    from astropy.io.votable import parse_single_table
    params = {"REQUEST": "queryData", "COLLECTION": COLLECTION, "MAXREC": int(maxrec)}
    if name:
        params["TARGETNAME"] = name
    if ra is not None and dec is not None:
        params["POS"] = f"{float(ra)},{float(dec)}"
        params["SIZE"] = float(radius) / 3600.0
    elif not name:
        raise ValueError("give a target name or ra and dec")
    url = SSA_URL + "?" + urllib.parse.urlencode(params)
    data = _get(url, timeout)
    if not data.lstrip().startswith(b"<"):
        raise RuntimeError("unexpected answer from the Data Central SSA service")
    try:
        df = parse_single_table(io.BytesIO(data)).to_table().to_pandas()
    except Exception as exc:
        raise RuntimeError(f"the Data Central SSA service did not answer properly ({exc})") from exc
    if df.empty:
        return df
    out = df.rename(columns={"target_name": "target", "dataproduct_subtype": "kind",
                             "t_midpoint": "mjd", "s_ra": "ra", "s_dec": "dec"})
    cols = [c for c in ("target", "kind", "mjd", "redshift", "t_exptime", "ra", "dec", "access_url")
            if c in out]
    out = out[cols].copy()
    out["kind"] = out["kind"].astype(str)
    out["_epoch"] = out["kind"] != "combined"            # the stack first, then by date
    return out.sort_values(["target", "_epoch", "mjd"]).drop(columns="_epoch").reset_index(drop=True)


def download(name=None, folder=".", source="vizier", ra=None, dec=None, radius=2.0, overwrite=False,
             timeout=120, verbose=True):
    """
    Download the OzDES DR2 spectra of one target into ``folder``; returns the
    file paths.

    :param source: 'vizier' (default): the original file with all exposures,
        variances and masks, from CDS (one file; a second one, ``*_both``, if
        the target was also observed with the x6700 dichroic). 'ssa': one
        flux-only file per spectrum from the Data Central SSA service (the stack
        first, then the exposures by date); 'auto': VizieR, else SSA.
    """
    if source not in ("vizier", "ssa", "auto"):
        raise ValueError("source must be 'vizier', 'ssa' or 'auto'")
    if source in ("vizier", "auto"):
        if not name:
            table = search(ra=ra, dec=dec, radius=radius, timeout=timeout)
            names = table.target.unique() if not table.empty else []
            if len(names) != 1:
                raise LookupError(f"{len(names)} targets at this position; give the name")
            name = names[0]
        try:
            return _download_vizier(name, folder, overwrite, timeout, verbose)
        except LookupError:
            if source == "vizier":
                raise
    table = search(name, ra=ra, dec=dec, radius=radius, timeout=timeout)
    if table.empty:
        raise LookupError(f"no OzDES DR2 spectra found for {name or (ra, dec)}")
    targets = table.target.unique()
    if len(targets) > 1:
        raise LookupError(f"{len(targets)} targets match ({', '.join(map(str, targets[:5]))}"
                          f"{' …' if len(targets) > 5 else ''}); give the name or a smaller radius")
    os.makedirs(folder, exist_ok=True)
    paths = []
    for k, row in enumerate(table.itertuples()):
        kind = "stack" if row.kind == "combined" else f"{row.mjd:.4f}"
        path = os.path.join(folder, f"{row.target}_{kind}.fits")
        if overwrite or not os.path.exists(path):
            url = row.access_url + ("&" if "?" in row.access_url else "?") + "RESPONSEFORMAT=fits"
            data = _get(url, timeout)
            if not data.startswith(b"SIMPLE"):
                raise RuntimeError(f"{url} did not return a FITS file")
            tmp = path + ".part"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
        paths.append(path)
        if verbose:
            print(f"\r{row.target}: {k + 1}/{len(table)} spectra", end="", flush=True)
    if verbose:
        print(f"  -> {os.path.abspath(folder)}")
    return paths


def _download_vizier(name, folder, overwrite, timeout, verbose):
    import urllib.error
    os.makedirs(folder, exist_ok=True)
    paths = []
    for sub, suffix in (("sp", ""), ("sp_both", "_both")):
        path = os.path.join(folder, f"{name}{suffix}.fits")
        if overwrite or not os.path.exists(path):
            try:
                data = _get(f"{VIZIER_URL}/{sub}/{urllib.parse.quote(name)}.fits", timeout)
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    continue
                raise
            if not data.startswith(b"SIMPLE"):
                continue
            with open(path + ".part", "wb") as fh:
                fh.write(data)
            os.replace(path + ".part", path)
        paths.append(path)
    if not paths:
        raise LookupError(f"{name}: no OzDES DR2 file at VizieR (J/MNRAS/496/19); the name must be "
                          "the OzDES ID, e.g. SVA1_COADD-2937976373")
    if verbose:
        print(f"{name}: {', '.join(os.path.basename(p) for p in paths)} -> {os.path.abspath(folder)}")
    return paths
