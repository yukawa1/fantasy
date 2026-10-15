"""
DESI spectra.

Reads DESI coadd (``coadd-*.fits``) and per-exposure (``spectra-*.fits``) files of
the DESI data releases (EDR, DR1, ...), which hold many targets each, and
spectra retrieved with NOIRLab's SPARCL client::

    import fantasy_agn as fa
    fa.desi.targets("coadd-main-dark-10032.fits", redrock="redrock-main-dark-10032.fits")
    spec = fa.desi.read("coadd-main-dark-10032.fits", targetid=39627..., redrock=...)
    spec = fa.read("coadd-main-dark-10032.fits", targetid=39627...)    # same thing

    spec = fa.desi.from_sparcl(record)       # a record from sparcl.client.retrieve()

The three spectrograph arms (B, R, Z) are merged into one spectrum: they share a
common 0.8 A wavelength grid, and in the overlaps the pixels are averaged with
inverse-variance weights. Pixels with a non-zero mask or zero inverse variance
are excluded from fits. The instrumental resolution, which changes by about a
factor of two from blue to red, is computed from the resolution matrix of each
target and kept as a curve, so every emission line is corrected with the
resolution at its own wavelength. Wavelengths are in vacuum and fluxes in
1e-17 erg/s/cm2/A.
"""
import os

import numpy as np
import pandas as pd
from astropy.io import fits

from .spectrum import Spectrum

__all__ = ("read", "read_all", "targets", "from_sparcl", "fetch", "find", "is_desi")

SPARCL_API = "https://astrosparcl.datalab.noirlab.edu/api"

C_KMS = 299792.458
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
ARMS = ("B", "R", "Z")


def is_desi(path):
    """True for DESI coadd/spectra files (they have per-arm flux extensions)."""
    try:
        with fits.open(path) as hdul:
            names = {h.name for h in hdul}
    except OSError:
        return False
    return "FIBERMAP" in names and any(f"{a}_FLUX" in names for a in ARMS)


def _redshifts(redrock):
    """Redshift table (TARGETID, Z, ZERR, ZWARN, SPECTYPE, ...) of a redrock or zcatalog file."""
    if redrock is None:
        return None
    from astropy.table import Table
    with fits.open(redrock) as hdul:
        ext = "REDSHIFTS" if "REDSHIFTS" in hdul else "ZCATALOG" if "ZCATALOG" in hdul else 1
        table = Table(hdul[ext].data)
    cols = [c for c in ("TARGETID", "Z", "ZERR", "ZWARN", "SPECTYPE", "SUBTYPE", "DELTACHI2")
            if c in table.colnames]
    df = table[cols].to_pandas()
    for c in ("SPECTYPE", "SUBTYPE"):
        if c in df:
            df[c] = df[c].astype(str).str.strip()
    return df.drop_duplicates("TARGETID").set_index("TARGETID")


def targets(path, redrock=None):
    """
    Table of the targets in a DESI coadd/spectra file: index, TARGETID, RA, Dec
    and, if a redrock (or zcatalog) file is given, redshift and spectral type.
    """
    with fits.open(path) as hdul:
        fm = hdul["FIBERMAP"].data
        df = pd.DataFrame({"index": np.arange(len(fm)),
                           "TARGETID": np.asarray(fm["TARGETID"], dtype=np.int64)})
        for src, dst in (("TARGET_RA", "RA"), ("TARGET_DEC", "DEC"), ("MEAN_MJD", "MJD")):
            if src in fm.columns.names:
                df[dst] = np.asarray(fm[src], float)
    z = _redshifts(redrock)
    if z is not None:
        df = df.join(z, on="TARGETID")
    return df


def _lsf_fwhm_kms(resolution, wave):
    """
    Instrumental FWHM (km/s) at each pixel from a DESI resolution matrix stored
    as diagonals (n_diag, n_wave): the second moment of the line-spread function.
    """
    nd = resolution.shape[0]
    offsets = np.arange(nd // 2, -(nd // 2) - 1, -1)[:, None].astype(float)
    weights = np.clip(resolution, 0, None)
    norm = weights.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (offsets * weights).sum(axis=0) / norm
        var = ((offsets - mean) ** 2 * weights).sum(axis=0) / norm
    dlam = np.gradient(wave)
    fwhm = FWHM_PER_SIGMA * np.sqrt(np.clip(var, 0, None)) * dlam / wave * C_KMS
    return np.where(norm > 0, fwhm, np.nan)


def _merge_arms(arms):
    """Merge per-arm (wave, flux, ivar, fwhm) arrays on their common grid."""
    keys = np.round(np.concatenate([a[0] for a in arms]), 2)
    grid = np.unique(keys)
    flux_w = np.zeros(grid.size)
    ivar = np.zeros(grid.size)
    fwhm_w = np.zeros(grid.size)
    fwhm_n = np.zeros(grid.size)
    for wave, flux, iv, fwhm in arms:
        idx = np.searchsorted(grid, np.round(wave, 2))
        good = np.isfinite(flux) & (iv > 0)
        np.add.at(flux_w, idx[good], (flux * iv)[good])
        np.add.at(ivar, idx[good], iv[good])
        ok = np.isfinite(fwhm)
        w = np.where(good, iv, 1e-30)[ok]          # resolution even where data are bad
        np.add.at(fwhm_w, idx[ok], fwhm[ok] * w)
        np.add.at(fwhm_n, idx[ok], w)
    with np.errstate(invalid="ignore", divide="ignore"):
        flux = np.where(ivar > 0, flux_w / ivar, 0.0)
        err = np.where(ivar > 0, 1 / np.sqrt(np.where(ivar > 0, ivar, 1)), np.inf)
        fwhm = np.where(fwhm_n > 0, fwhm_w / fwhm_n, np.nan)
    good_fwhm = np.isfinite(fwhm)
    if good_fwhm.any() and not good_fwhm.all():
        fwhm = np.interp(grid, grid[good_fwhm], fwhm[good_fwhm])
    return grid, flux, err, fwhm


def read(path, targetid=None, index=None, redrock=None, z=None):
    """
    One target of a DESI coadd or spectra file as a Spectrum.

    :param targetid: DESI TARGETID (or give ``index``, the row in the file).
    :param redrock: redrock or zcatalog file with the redshifts (optional).
    :param z: redshift, overriding the redrock value.
    """
    with fits.open(path) as hdul:
        fm = hdul["FIBERMAP"].data
        tids = np.asarray(fm["TARGETID"], dtype=np.int64)
        if targetid is not None:
            rows = np.nonzero(tids == int(targetid))[0]
            if rows.size == 0:
                raise ValueError(f"TARGETID {targetid} is not in {os.path.basename(path)}")
            row = int(rows[0])
        elif index is not None:
            row = int(index)
        elif len(tids) == 1:
            row = 0
        else:
            raise ValueError(f"{os.path.basename(path)} holds {len(tids)} targets: pass "
                             "targetid=... or index=... (see fantasy_agn.desi.targets)")
        tid = int(tids[row])
        arms = []
        for arm in ARMS:
            if f"{arm}_FLUX" not in hdul:
                continue
            wave = np.asarray(hdul[f"{arm}_WAVELENGTH"].data, float)
            flux = np.asarray(hdul[f"{arm}_FLUX"].data[row], float)
            ivar = np.asarray(hdul[f"{arm}_IVAR"].data[row], float)
            if f"{arm}_MASK" in hdul:
                ivar = np.where(np.asarray(hdul[f"{arm}_MASK"].data[row]) != 0, 0.0, ivar)
            if f"{arm}_RESOLUTION" in hdul:
                fwhm = _lsf_fwhm_kms(np.asarray(hdul[f"{arm}_RESOLUTION"].data[row], float), wave)
            else:
                fwhm = np.full(wave.size, np.nan)
            arms.append((wave, flux, ivar, fwhm))
        if not arms:
            raise ValueError(f"no B/R/Z spectra in {os.path.basename(path)}")
        meta = {"targetid": tid, "file": os.path.basename(path)}
        for col in ("SURVEY", "PROGRAM", "HEALPIX", "TILEID", "COADD_NUMEXP", "DESI_TARGET"):
            if col in fm.columns.names:
                v = fm[col][row]
                meta[col.lower()] = v.strip() if isinstance(v, str) else v.item()
        if "EBV" in fm.columns.names:                 # SFD98 E(B-V) of the target
            meta["ebv_file"] = float(fm["EBV"][row])
        ra = float(fm["TARGET_RA"][row]) if "TARGET_RA" in fm.columns.names else None
        dec = float(fm["TARGET_DEC"][row]) if "TARGET_DEC" in fm.columns.names else None
        mjd = None
        for col in ("MEAN_MJD", "MJD"):
            if col in fm.columns.names:
                mjd = float(fm[col][row])
                break
    zt = _redshifts(redrock)
    if z is None and zt is not None and tid in zt.index:
        z = float(zt.loc[tid, "Z"])
        for c in ("ZERR", "ZWARN", "SPECTYPE", "SUBTYPE", "DELTACHI2"):
            if c in zt:
                v = zt.loc[tid, c]
                meta[c.lower()] = v.item() if hasattr(v, "item") else v
    wave, flux, err, fwhm = _merge_arms(arms)
    ok = np.isfinite(fwhm)
    resolution = float(np.median(fwhm[ok])) if ok.any() else 0.0
    return Spectrum(wave, flux, err, z=z, name=f"desi-{tid}", medium="vacuum",
                    resolution=resolution, flux_unit=1e-17, ra=ra, dec=dec, mjd=mjd,
                    meta=meta, resolution_curve=(wave, fwhm) if ok.any() else None)


def read_all(path, redrock=None, spectype=None, zmin=None, zmax=None):
    """
    All targets of a DESI file as a list of Spectrum, optionally only those of
    a redrock spectral type (e.g. 'QSO') and redshift range (needs ``redrock``).
    """
    table = targets(path, redrock)
    if spectype is not None:
        table = table[table["SPECTYPE"] == spectype]
    if zmin is not None:
        table = table[table["Z"] >= zmin]
    if zmax is not None:
        table = table[table["Z"] <= zmax]
    return [read(path, index=int(i), redrock=redrock) for i in table["index"]]


def from_sparcl(record, z=None):
    """
    Spectrum from a SPARCL record (sparcl.client.SparclClient().retrieve(...)
    .records[i]), which needs the fields wavelength, flux and ivar; mask,
    redshift, targetid, ra, dec and wave_sigma (line-spread sigma in A per
    pixel) are used when present.
    """
    get = (lambda k, d=None: record.get(k, d)) if isinstance(record, dict) else \
        (lambda k, d=None: getattr(record, k, d))
    wave = np.asarray(get("wavelength"), float)
    flux = np.asarray(get("flux"), float)
    ivar = np.asarray(get("ivar"), float)
    mask = get("mask")
    if mask is not None:
        ivar = np.where(np.asarray(mask) != 0, 0.0, ivar)
    with np.errstate(divide="ignore"):
        err = np.where(ivar > 0, 1 / np.sqrt(np.where(ivar > 0, ivar, 1)), np.inf)
    sigma = get("wave_sigma")
    curve, resolution = None, 0.0
    if sigma is not None:
        fwhm = FWHM_PER_SIGMA * np.asarray(sigma, float) / wave * C_KMS
        ok = np.isfinite(fwhm) & (fwhm > 0)
        if ok.any():
            curve = (wave[ok], fwhm[ok])
            resolution = float(np.median(fwhm[ok]))
    tid = get("targetid", get("specid"))
    dr = str(get("data_release", "") or "")
    return Spectrum(wave, flux, err, z=get("redshift") if z is None else z,
                    name=f"desi-{tid}" if tid is not None else "sparcl",
                    medium="vacuum", resolution=resolution, flux_unit=1e-17,
                    ra=get("ra"), dec=get("dec"), mjd=get("mean_mjd"), resolution_curve=curve,
                    meta={"targetid": tid, "data_release": dr, "sparcl_id": get("sparcl_id"),
                          "spectype": get("spectype"), "survey": get("survey"),
                          "program": get("program")})


# ------------------------------------------------------------------ SPARCL
def _post(endpoint, payload, timeout=300):
    import json
    import urllib.request
    req = urllib.request.Request(f"{SPARCL_API}/{endpoint}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def find(targetid=None, ra=None, dec=None, radius=2.0, data_release="DESI-DR1", limit=50):
    """
    DESI spectra available in NOIRLab's SPARCL service, by TARGETID or by
    position (ra, dec in degrees, radius in arcsec). Returns a table with
    sparcl_id, targetid, redshift, spectype, survey, program and whether it is
    the primary spectrum of the target.
    """
    search = [["data_release", data_release]]
    if targetid is not None:
        search.append(["specid", int(targetid)])
    elif ra is not None and dec is not None:
        r = radius / 3600
        cosd = max(np.cos(np.radians(dec)), 1e-3)
        search += [["ra", ra - r / cosd, ra + r / cosd], ["dec", dec - r, dec + r]]
    else:
        raise ValueError("give targetid or ra and dec")
    out = _post(f"find/?limit={int(limit)}",
                {"outfields": ["sparcl_id", "specid", "redshift", "spectype", "survey",
                               "program", "specprimary", "ra", "dec"], "search": search})
    rows = [r for r in out[1:]]
    df = pd.DataFrame(rows).rename(columns={"specid": "targetid"})
    return df.drop(columns=[c for c in ("_dr",) if c in df])


def fetch(targetid=None, sparcl_id=None, data_release="DESI-DR1", save=None):
    """
    Download a public DESI spectrum from NOIRLab's SPARCL service and return
    it as a Spectrum (the primary spectrum of the target unless ``sparcl_id``
    is given). ``save`` writes the retrieved record as JSON, readable later
    with ``from_sparcl(json.load(open(path))[1])``.

    No account or extra package is needed. Please acknowledge DESI and the
    Astro Data Lab when you use these data.
    """
    import json
    if sparcl_id is None:
        found = find(targetid=targetid, data_release=data_release)
        if found.empty:
            raise ValueError(f"no {data_release} spectrum for TARGETID {targetid} in SPARCL")
        primary = found[found.get("specprimary", True) == True]  # noqa: E712
        sparcl_id = (primary if len(primary) else found).iloc[0]["sparcl_id"]
    fields = ["sparcl_id", "specid", "targetid", "data_release", "redshift", "redshift_err",
              "redshift_warning", "spectype", "ra", "dec", "mean_mjd", "survey", "program",
              "healpix", "wavelength", "flux", "ivar", "mask", "wave_sigma"]
    out = _post(f"spectras/?include={','.join(fields)}&format=json"
                f"&dataset_list={data_release}", [sparcl_id])
    records = [r for r in out if isinstance(r, dict) and "flux" in r]
    if not records:
        raise ValueError(f"SPARCL returned no spectrum for {sparcl_id}")
    if save:
        with open(save, "w") as fh:
            json.dump(out, fh)
    return from_sparcl(records[0])
