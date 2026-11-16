"""
Download public SDSS spectra by plate, MJD and fibre from the Science Archive
Server (DR17: SDSS-I/II and BOSS/eBOSS "lite" spectra).

    spec = fa.sdss.fetch(1059, 52618, 453)

Files are kept in ``~/.fantasy_agn/sdss`` and read from there the next time.
"""
import os
import urllib.error
import urllib.request

from . import _net

__all__ = ("fetch", "url")

SAS = "https://data.sdss.org/sas/dr17"
FOLDER = os.path.join(os.path.expanduser("~"), ".fantasy_agn", "sdss")
# reductions tried in turn: SDSS-I/II (run2d 26, 103, 104), then BOSS/eBOSS (v5_13_2)
RUN2D = ("26", "103", "104", "v5_13_2")


def url(plate, mjd, fiber, run2d="26"):
    """SAS address of a spec-PLATE-MJD-FIBER.fits file for one reduction."""
    tree = "sdss" if str(run2d) in ("26", "103", "104") else "eboss"
    return (f"{SAS}/{tree}/spectro/redux/{run2d}/spectra/lite/{int(plate):04d}/"
            f"spec-{int(plate):04d}-{int(mjd)}-{int(fiber):04d}.fits")


def fetch(plate, mjd, fiber, folder=None, timeout=120):
    """
    Download (or reuse) the SDSS spectrum ``plate``-``mjd``-``fiber`` and return it
    as a Spectrum. ``folder``: where the FITS file is kept (default ~/.fantasy_agn/sdss).
    """
    from .spectrum import read
    folder = os.path.expanduser(folder or FOLDER)
    name = f"spec-{int(plate):04d}-{int(mjd)}-{int(fiber):04d}.fits"
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        os.makedirs(folder, exist_ok=True)
        for run2d in RUN2D:
            try:
                with _net.urlopen(url(plate, mjd, fiber, run2d), timeout=timeout) as resp:
                    data = resp.read()
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    continue
                raise
            if not data.startswith(b"SIMPLE"):
                continue
            with open(path + ".part", "wb") as fh:
                fh.write(data)
            os.replace(path + ".part", path)
            break
        else:
            raise ValueError(f"no SDSS DR17 spectrum {name[5:-5]} on the Science Archive Server")
    return read(path)
