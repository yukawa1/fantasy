"""
Download the SDSS DR17 'lite' spectra of sample.csv (about 0.2 MB each).

    python calibration/download.py [--out ~/.fantasy_agn/samples]
"""
import argparse
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

SAS = "https://data.sdss.org/sas/dr17"


def url(row):
    tree = "sdss" if str(row.run2d) in ("26", "103", "104") else "eboss"
    return (f"{SAS}/{tree}/spectro/redux/{row.run2d}/spectra/lite/{int(row.plate):04d}/"
            f"spec-{int(row.plate):04d}-{int(row.mjd)}-{int(row.fiberID):04d}.fits")


def fetch(args):
    u, path = args
    if os.path.exists(path):
        return path, None
    for k in range(4):
        try:
            with urllib.request.urlopen(u, timeout=120) as r:
                data = r.read()
            if not data.startswith(b"SIMPLE"):
                return path, "not a FITS file"
            with open(path + ".part", "wb") as fh:
                fh.write(data)
            os.replace(path + ".part", path)
            return path, None
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return path, "404"
            time.sleep(2 * (k + 1))
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 * (k + 1))
    return path, "failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/.fantasy_agn/samples")
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    sample = pd.read_csv(os.path.join(out, "sample.csv"))
    folder = os.path.join(out, "spectra")
    os.makedirs(folder, exist_ok=True)
    jobs = [(url(r), os.path.join(folder, os.path.basename(url(r)))) for r in sample.itertuples()]
    sample["file"] = [os.path.basename(p) for _, p in jobs]
    bad = {}
    with ThreadPoolExecutor(8) as ex:
        for k, (path, err) in enumerate(ex.map(fetch, jobs), 1):
            if err:
                bad[os.path.basename(path)] = err
            if k % 50 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} ({len(bad)} failed)", flush=True)
    sample["downloaded"] = ~sample.file.isin(bad)
    sample.to_csv(os.path.join(out, "sample.csv"), index=False)
    size = sum(os.path.getsize(os.path.join(folder, f)) for f in os.listdir(folder)) / 1e6
    print(f"{sample.downloaded.sum()} spectra, {size:.0f} MB in {folder}")
    if bad:
        print("failed:", dict(list(bad.items())[:10]))


if __name__ == "__main__":
    main()
