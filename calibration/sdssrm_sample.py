"""
Download SDSS-RM quasars with published H-beta lags (Shen et al. 2024): all their SDSS
spectra (by position, SkyServer DR17) and the published light curves, for testing
multi-epoch models on several objects. Usage: python calibration/sdssrm_sample.py 776 790 ...
"""
import os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from astropy.table import Table
import select_sample as ss
import download

ROOT = os.path.expanduser("~/.fantasy_agn/samples/sdssrm")
SUMMARY = sys.argv[1] if sys.argv[1].endswith(".fits") else None
RM = "https://ariel.astro.illinois.edu/sdssrm/final_result"

if __name__ == "__main__":
    summary = Table.read(sys.argv[1])
    for rmid in map(int, sys.argv[2:]):
        r = summary[summary["RMID"] == rmid][0]
        ra, dec = float(r["RA"]), float(r["DEC"])
        d = 1.5 / 3600
        sql = (f"SELECT {ss.SPEC} FROM SpecObjAll s WHERE s.ra BETWEEN {ra - d / np.cos(np.radians(dec))} AND "
               f"{ra + d / np.cos(np.radians(dec))} AND s.dec BETWEEN {dec - d} AND {dec + d}")
        for attempt in range(4):
            try:
                t = ss.query(sql)
                break
            except Exception as exc:
                print(f"  rm{rmid}: query failed ({type(exc).__name__}), retrying", flush=True)
                __import__("time").sleep(10 * (attempt + 1))
        else:
            continue
        folder = os.path.join(ROOT, f"rm{rmid:03d}")
        os.makedirs(folder, exist_ok=True)
        jobs = [(download.url(row), os.path.join(folder, os.path.basename(download.url(row)))) for row in t.itertuples()]
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(download.fetch, jobs))
        for f in ("hb_lc.txt", "cont_lc.txt", "ha_lc.txt"):
            try:
                with urllib.request.urlopen(f"{RM}/rm{rmid:03d}/{f}", timeout=60) as resp:
                    open(os.path.join(folder, f), "wb").write(resp.read())
            except Exception as exc:
                print(f"  rm{rmid}: no {f} ({exc})")
        bad = [p for p, e in res if e]
        print(f"rm{rmid:03d}: z = {float(r['ZSYS']):.3f}, {len(t)} spectra ({len(bad)} failed), "
              f"published Hb lag {float(r['HB_LAG_DET']):.1f} d", flush=True)
