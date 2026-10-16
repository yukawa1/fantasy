"""
Fit the calibration sample with each group's preset and a few variants, and
classify every spectrum. One JSON file per spectrum in <out>/fits/, so the run
can be interrupted and resumed; analyse with calibration/analyze.py.

    python calibration/fit_sample.py [--out ~/.fantasy_agn/samples] [--jobs 9] [--limit N]

Galactic E(B-V) comes from the SDSS photometry (extinction_r / 2.751, SFD98),
recalibrated by 0.86 (Schlafly & Finkbeiner 2011), queried once for the sample.
"""
import argparse
import io
import json
import os
import time
import traceback
import urllib.parse
import urllib.request
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

URL = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"

# group -> list of (label, preset, kwargs); the first is the preset as it is now
VARIANTS = {
    "t1_low": [("optical", "optical", {}),
               ("optical+outflow", "optical", {"outflow": True}),
               ("hbeta", "hbeta", {}),
               ("hbeta 1 broad", "hbeta", {"broad_components": 1}),
               ("hbeta 3 broad", "hbeta", {"broad_components": 3}),
               ("hbeta+outflow", "hbeta", {"outflow": True}),
               ("halpha", "halpha", {}),
               ("halpha 1 broad", "halpha", {"broad_components": 1}),
               ("halpha+outflow", "halpha", {"outflow": True})],
    "t1_hb": [("hbeta", "hbeta", {}),
              ("hbeta 1 broad", "hbeta", {"broad_components": 1}),
              ("hbeta 3 broad", "hbeta", {"broad_components": 3}),
              ("hbeta+outflow", "hbeta", {"outflow": True})],
    "t1_mgii": [("mgii", "mgii", {}),
                ("mgii 2 broad", "mgii", {"broad_components": 2})],
    "t1_civ": [("civ", "civ", {}),
               ("civ 2 broad", "civ", {"broad_components": 2})],
    "sy2": [("seyfert2", "seyfert2", {}),
            ("seyfert2 no outflow", "seyfert2", {"outflow": False}),
            ("seyfert2 one system", "seyfert2", {"two_systems": False})],
    "liner": [("liner", "liner", {}),
              ("liner+broad", "liner", {"broad": True}),
              ("liner+outflow", "liner", {"outflow": True})],
    "comp": [("composite", "composite", {}),
             ("composite no outflow", "composite", {"outflow": False})],
    "sf": [("starforming", "starforming", {}),
           ("starforming no outflow", "starforming", {"outflow": False})],
    "passive": [("passive", "passive", {})],
}


def galactic_ebv(sample, chunk=80):
    """SFD98 E(B-V) from the SDSS photometry of each object."""
    rows = []
    keys = list(sample[["plate", "mjd", "fiberID"]].itertuples(index=False))
    for k in range(0, len(keys), chunk):
        cond = " OR ".join(f"(s.plate={p} AND s.mjd={m} AND s.fiberID={f})" for p, m, f in keys[k:k + chunk])
        sql = ("SELECT s.plate, s.mjd, s.fiberID, p.extinction_r FROM SpecObj s "
               f"JOIN PhotoObj p ON p.objID = s.bestObjID WHERE {cond}")
        url = URL + "?" + urllib.parse.urlencode({"cmd": sql, "format": "csv"})
        with urllib.request.urlopen(url, timeout=300) as r:
            text = "\n".join(ln for ln in r.read().decode().splitlines() if not ln.startswith("#"))
        rows.append(pd.read_csv(io.StringIO(text)))
    df = pd.concat(rows)
    df["ebv_sfd"] = df.extinction_r / 2.751
    return df[["plate", "mjd", "fiberID", "ebv_sfd"]]


def _params(r):
    """Fitted value and error of every free parameter (names as in the model)."""
    out = {}
    names = r.model.params.names
    for p in r.model.params:
        if p.free:
            i = names.index(p.name)
            err = float(np.sqrt(r.cov[i, i])) if r.cov is not None and r.cov[i, i] >= 0 else None
            out[p.name] = [float(r.full[i]), err, float(p.min), float(p.max), bool(p.linear)]
    return out


def fit_object(row):
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")
    import fantasy_agn as fa
    path = os.path.join(row["folder"], row["file"])
    out = {"file": row["file"], "group": row["group"], "z": row["z"], "sn": row["snMedian"],
           "class": row["class"], "subclass": row.get("subclass"), "fits": []}
    try:
        spec = fa.read(path)
        ebv = row.get("ebv_sfd")
        if ebv is not None and np.isfinite(ebv):
            spec = spec.deredden(ebv=0.86 * float(ebv))
        spec = spec.to_rest_frame()
    except Exception as exc:
        out["error"] = f"read: {type(exc).__name__}: {exc}"
        return out
    if row["group"] in ("t1_low", "sy2", "liner", "comp", "sf", "passive"):
        try:
            t0 = time.time()
            c = fa.classify(spec)
            out["classify"] = {"label": c.label, "uncertain": bool(c.uncertain), "time": time.time() - t0}
        except Exception as exc:
            out["classify"] = {"error": f"{type(exc).__name__}: {exc}"}
    for label, preset, kw in VARIANTS[row["group"]]:
        rec = {"variant": label, "preset": preset, "kwargs": kw}
        try:
            model = getattr(fa.presets, preset)(**kw)
            t0 = time.time()
            r = fa.fit(spec, model, verbose=False)
            rec.update(time=time.time() - t0, chi2=r.chi2, dof=r.dof, rchi2=r.rchi2, bic=r.bic,
                       n_free=r.n_free, nfev=r.nfev, success=bool(r.success), start=r.start,
                       at_limit=list(r.at_limit), at_zero=list(r.at_zero), params=_params(r),
                       range=[float(r.spectrum.wave[r.mask].min()), float(r.spectrum.wave[r.mask].max())])
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["trace"] = traceback.format_exc()[-1500:]
        out["fits"].append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/.fantasy_agn/samples")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=None, help="only the first N per group (testing)")
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    sample = pd.read_csv(os.path.join(out, "sample.csv"))
    import download
    sample["file"] = [os.path.basename(download.url(r)) for r in sample.itertuples()]
    present = set(os.listdir(os.path.join(out, "spectra")))
    sample = sample[sample.file.isin(present)]                # what is downloaded (rerun for the rest)
    ebv_path = os.path.join(out, "ebv.csv")
    if not os.path.exists(ebv_path):
        galactic_ebv(sample).to_csv(ebv_path, index=False)
    sample = sample.merge(pd.read_csv(ebv_path), on=["plate", "mjd", "fiberID"], how="left")
    if a.limit:
        sample = sample.groupby("group").head(a.limit)
    folder = os.path.join(out, "fits")
    os.makedirs(folder, exist_ok=True)
    todo = [dict(r, folder=os.path.join(out, "spectra")) for r in sample.to_dict("records")
            if not os.path.exists(os.path.join(folder, r["file"] + ".json"))]
    print(f"{len(todo)} spectra to fit ({len(sample) - len(todo)} done), {a.jobs} processes", flush=True)
    os.environ.update({"FANTASY_THREADS": "1", "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1"})
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        futures = {ex.submit(fit_object, r): r["file"] for r in todo}
        for k, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            with open(os.path.join(folder, res["file"] + ".json"), "w") as fh:
                json.dump(res, fh, default=float)
            if k % 20 == 0 or k == len(todo):
                el = time.time() - t0
                print(f"{k}/{len(todo)}  {el / 60:.1f} min, about {el / k * (len(todo) - k) / 60:.0f} min left",
                      flush=True)


if __name__ == "__main__":
    main()
