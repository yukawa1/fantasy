"""
Validation of the updated presets on the held-out 30% of the sample: the new
default of each group (and, for low-z quasars, the classifier's
recommendation) against the old preset fitted in fit_sample.py.

    python calibration/validate.py [--jobs 9]
"""
import argparse
import glob
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from analyze import half

ROOT = os.path.expanduser("~/.fantasy_agn/samples")
OLD = {"t1_low": "optical", "t1_hb": "hbeta", "t1_mgii": "mgii", "t1_civ": "civ", "sy2": "seyfert2",
       "liner": "liner", "comp": "composite", "sf": "starforming", "passive": "passive"}


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

    def one(label, model):
        t0 = time.time()
        r = fa.fit(s, model, verbose=False)
        out.append(dict(file=file, group=group, variant=label, rchi2=r.rchi2, bic=r.bic, nfev=r.nfev,
                        time=time.time() - t0, n_limit=len(r.at_limit)))

    one("new default", getattr(fa.presets, OLD[group])())
    if group == "t1_low":
        c = fa.classify(s)
        if c.label.startswith("type1"):
            one("classify -> recommended", c.model())
        if group == "t1_low":
            out[-1]["label"] = c.label
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=9)
    a = ap.parse_args()
    ebv = pd.read_csv(os.path.join(ROOT, "ebv.csv"))
    s = pd.read_csv(os.path.join(ROOT, "sample.csv")).merge(ebv, on=["plate", "mjd", "fiberID"], how="left")
    s = s[s.file.map(half) == "val"]
    old = []
    for p in glob.glob(os.path.join(ROOT, "fits", "*.json")):
        o = json.load(open(p))
        if half(o["file"]) != "val":
            continue
        for f in o["fits"]:
            if f["variant"] == OLD[o["group"]] and "rchi2" in f:
                old.append(dict(file=o["file"], group=o["group"], variant="old default", rchi2=f["rchi2"],
                                bic=f["bic"], nfev=f["nfev"], time=f["time"], n_limit=len(f["at_limit"])))
    os.environ.update(FANTASY_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1")
    rows = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for res in ex.map(run, zip(s.file, s.group, s.ebv_sfd)):
            rows += res
    df = pd.concat([pd.DataFrame(old), pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(os.path.join(ROOT, "analysis", "validation.csv"), index=False)
    lines = ["| group | model | n | median rchi2 | 90% rchi2 | limits per fit | evaluations | time (s) | "
             "BIC better than old | median dBIC |", "|---|---|---|---|---|---|---|---|---|---|"]
    for group in OLD:
        d = df[df.group == group]
        b = d[d.variant == "old default"].set_index("file")
        for v in ("old default", "new default", "classify -> recommended"):
            x = d[d.variant == v].set_index("file")
            if x.empty:
                continue
            dbic = (x.bic - b.bic.reindex(x.index)).dropna()
            lines.append(f"| {group} | {v} | {len(x)} | {x.rchi2.median():.3f} | {x.rchi2.quantile(0.9):.2f} | "
                         f"{x.n_limit.mean():.2f} | {x.nfev.median():.0f} | {x.time.median():.1f} | "
                         f"{np.mean(dbic < -2):.0%} | {dbic.median():.1f} |")
    text = "\n".join(lines)
    with open(os.path.join(ROOT, "analysis", "validation.md"), "w") as fh:
        fh.write("# Validation of the updated presets (held-out 30%)\n\n" + text + "\n")
    print(text)


if __name__ == "__main__":
    main()
