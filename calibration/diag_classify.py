"""Re-run classify on type-1 objects and record the broad-line test quantities."""
import glob, json, os, sys, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

ROOT = os.path.expanduser("~/.fantasy_agn/samples")


def one(args):
    warnings.filterwarnings("ignore")
    import matplotlib; matplotlib.use("Agg")
    import fantasy_agn as fa
    file, ebv = args
    s = fa.read(os.path.join(ROOT, "spectra", file))
    if ebv == ebv:
        s = s.deredden(ebv=0.86 * ebv)
    c = fa.classify(s.to_rest_frame())
    m = c.measurements
    keys = ("broad_dbic", "broad_snr", "broad_fwhm", "broad_ew", "host_fraction_5100", "ew_ha", "weak_broad")
    return dict(file=file, label=c.label, **{k: m.get(k) for k in keys})


if __name__ == "__main__":
    groups = sys.argv[1:] or ["t1_low"]
    ebv = pd.read_csv(os.path.join(ROOT, "ebv.csv"))
    s = pd.read_csv(os.path.join(ROOT, "sample.csv")).merge(ebv, on=["plate", "mjd", "fiberID"], how="left")
    s = s[s.group.isin(groups)]
    os.environ.update(FANTASY_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1")
    with ProcessPoolExecutor(9) as ex:
        rows = list(ex.map(one, zip(s.file, s.ebv_sfd)))
    df = pd.DataFrame(rows).merge(s[["file", "group", "z", "snMedian"]], on="file")
    df.to_csv(os.path.join(ROOT, "analysis", "classify_" + "_".join(groups) + ".csv"), index=False)
    print(df.label.value_counts().to_dict())
