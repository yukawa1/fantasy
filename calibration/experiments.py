"""
Candidate preset changes, fitted to the validation half of the sample (the
part not used to derive the parameter distributions) and compared with the
baseline: chi2, BIC, parameters at a limit, evaluations.

    python calibration/experiments.py [--groups t1_low t1_hb] [--jobs 9]
"""
import argparse
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from analyze import half

ROOT = os.path.expanduser("~/.fantasy_agn/samples")


def _tie_if(m, a, b):
    if a in m.params.names and b in m.params.names:
        m.tie(a, b)
    return m


def experiments(fa):
    P = fa.presets

    def he_tied(m):
        _tie_if(m, "broad_he.fwhm", "broad.fwhm")
        return _tie_if(m, "broad_he.shift", "broad.shift")

    def feii_tied(m):
        return _tie_if(m, "feii.shift", "broad.shift")

    def feii_narrow(m):
        m.set("feii.shift", min=-1000, max=1000)
        return m

    def mg_narrow600(m):
        m.set("narrow.fwhm", value=400, min=50, max=600)
        return m

    def mg_nonarrow(m):
        m.remove("narrow")
        return m

    def mg_index_fixed(m):
        m.fix("continuum.index", -1.5)
        return m

    def civ_he_tied(m):
        return _tie_if(m, "heii.shift", "broad.shift")

    def civ_he_limits(m):
        m.set("heii.shift", min=-1500, max=1500)
        m.set("heii.fwhm", min=1000, max=8000)
        return m

    def sy2_cap(m):
        m.set("outflow.fwhm", value=900, min=500, max=1500)
        return m

    return {
        "t1_low": [("optical+outflow", lambda: P.optical(outflow=True)),
                   ("+He tied", lambda: he_tied(P.optical(outflow=True))),
                   ("+He tied +FeII tied", lambda: feii_tied(he_tied(P.optical(outflow=True)))),
                   ("+He tied +FeII +-1000", lambda: feii_narrow(he_tied(P.optical(outflow=True)))),
                   ("optical+outflow+host", lambda: he_tied(P.optical(outflow=True, host=True)))],
        "t1_hb": [("hbeta+outflow", lambda: P.hbeta(outflow=True)),
                  ("+FeII tied", lambda: feii_tied(P.hbeta(outflow=True))),
                  ("+FeII +-1000", lambda: feii_narrow(P.hbeta(outflow=True)))],
        "t1_mgii": [("mgii", lambda: P.mgii()),
                    ("narrow <= 600", lambda: mg_narrow600(P.mgii())),
                    ("no narrow", lambda: mg_nonarrow(P.mgii())),
                    ("narrow <= 600, index -1.5", lambda: mg_index_fixed(mg_narrow600(P.mgii())))],
        "t1_civ": [("civ 2 broad", lambda: P.civ(broad_components=2)),
                   ("+HeII shift tied", lambda: civ_he_tied(P.civ(broad_components=2))),
                   ("+HeII limits", lambda: civ_he_limits(P.civ(broad_components=2)))],
        "sy2": [("seyfert2", lambda: P.seyfert2()),
                ("outflow <= 1500", lambda: sy2_cap(P.seyfert2()))],
    }


def run(job):
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")
    import fantasy_agn as fa
    file, group, ebv = job
    s = fa.read(os.path.join(ROOT, "spectra", file))
    if ebv == ebv:
        s = s.deredden(ebv=0.86 * ebv)
    s = s.to_rest_frame()
    out = []
    for label, make in experiments(fa)[group]:
        t0 = time.time()
        try:
            r = fa.fit(s, make(), verbose=False)
            out.append(dict(file=file, group=group, variant=label, rchi2=r.rchi2, bic=r.bic, n_free=r.n_free,
                            nfev=r.nfev, time=time.time() - t0, n_limit=len(r.at_limit),
                            at_limit=";".join(r.at_limit)))
        except Exception as exc:
            out.append(dict(file=file, group=group, variant=label, error=f"{type(exc).__name__}: {exc}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=["t1_low", "t1_hb", "t1_mgii", "t1_civ", "sy2"])
    ap.add_argument("--jobs", type=int, default=9)
    ap.add_argument("--tag", default="exp1")
    a = ap.parse_args()
    ebv = pd.read_csv(os.path.join(ROOT, "ebv.csv"))
    s = pd.read_csv(os.path.join(ROOT, "sample.csv")).merge(ebv, on=["plate", "mjd", "fiberID"], how="left")
    s = s[s.group.isin(a.groups) & (s.file.map(half) == "val")]
    print(f"{len(s)} validation spectra", flush=True)
    os.environ.update(FANTASY_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1")
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(run, j) for j in zip(s.file, s.group, s.ebv_sfd)]
        for k, f in enumerate(as_completed(futs), 1):
            rows += f.result()
            if k % 25 == 0:
                print(f"{k}/{len(futs)} {(time.time() - t0) / 60:.1f} min", flush=True)
    df = pd.DataFrame(rows)
    path = os.path.join(ROOT, "analysis", f"{a.tag}.csv")
    df.to_csv(path, index=False)
    summarise(df)


def summarise(df):
    ok = df[df.get("error").isna()] if "error" in df else df
    for group, d in ok.groupby("group", sort=False):
        base = d.variant.iloc[0]
        b = d[d.variant == base].set_index("file")
        print(f"\n{group} (baseline: {base}, n = {len(b)})")
        print(f"  {'variant':28s} {'rchi2':>6s} {'p90':>6s} {'limits':>6s} {'nfev':>5s} {'time':>5s} "
              f"{'BIC wins vs base':>16s} {'median dBIC':>11s}")
        for v, x in d.groupby("variant", sort=False):
            x = x.set_index("file")
            dbic = (x.bic - b.bic.reindex(x.index)).dropna()
            print(f"  {v:28s} {x.rchi2.median():6.3f} {x.rchi2.quantile(0.9):6.2f} {x.n_limit.mean():6.2f} "
                  f"{x.nfev.median():5.0f} {x.time.median():5.1f} {np.mean(dbic < -2):16.2f} {dbic.median():11.1f}")


if __name__ == "__main__":
    main()
