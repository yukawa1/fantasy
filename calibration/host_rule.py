"""
When should a type-1 fit include the stellar host? For the low-z quasars:
classifier measurements (host fraction, sigma* and its error) and optical fits
with and without the host. Writes analysis/host_rule.csv.
"""
import os, time, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from analyze import half

ROOT = os.path.expanduser("~/.fantasy_agn/samples")


def run(job):
    warnings.filterwarnings("ignore")
    import matplotlib; matplotlib.use("Agg")
    import fantasy_agn as fa
    file, ebv = job
    s = fa.read(os.path.join(ROOT, "spectra", file))
    if ebv == ebv:
        s = s.deredden(ebv=0.86 * ebv)
    s = s.to_rest_frame()
    c = fa.classify(s)
    m = c.measurements
    out = dict(file=file, split=half(file), label=c.label, host_cls=m["host_fraction_5100"],
               sig_cls=m["sigma_star"], sig_cls_err=m["sigma_star_err"], broad_ew=m.get("broad_ew"))
    try:
        r0 = fa.fit(s, fa.presets.optical(), verbose=False)
        r1 = fa.fit(s, fa.presets.optical(host=True), verbose=False)
        hp = r1.host_properties()
        out.update(bic0=r0.bic, bic1=r1.bic, rchi0=r0.rchi2, rchi1=r1.rchi2, lim0=len(r0.at_limit),
                   lim1=len(r1.at_limit), host_fit=hp.get("host_fraction_5100", (np.nan,))[0],
                   sig_fit=hp["sigma_star"][0], sig_fit_err=hp["sigma_star"][1],
                   sig_at_limit=any(n.endswith(("sigma", "velocity")) and n.startswith("host") for n in r1.at_limit))
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


if __name__ == "__main__":
    ebv = pd.read_csv(os.path.join(ROOT, "ebv.csv"))
    s = pd.read_csv(os.path.join(ROOT, "sample.csv")).merge(ebv, on=["plate", "mjd", "fiberID"], how="left")
    s = s[s.group == "t1_low"]
    os.environ.update(FANTASY_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1")
    t0 = time.time()
    with ProcessPoolExecutor(9) as ex:
        rows = list(ex.map(run, zip(s.file, s.ebv_sfd)))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "analysis", "host_rule.csv"), index=False)
    print(f"{len(rows)} quasars, {(time.time() - t0) / 60:.0f} min")
