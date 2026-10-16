"""
Analyse the calibration fits (calibration/fit_sample.py): per group and model
variant the fit quality, parameters at their limits, BIC preferences, the
distributions of the nonlinear parameters with suggested start values and
limits (from the calibration half of the sample), and the classification.

    python calibration/analyze.py [--out ~/.fantasy_agn/samples]

Writes report.md and params.csv to <out>/analysis/.
"""
import argparse
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd

EXPECTED = {"t1_low": "type1", "sy2": "seyfert2", "liner": "liner", "comp": "composite",
            "sf": "starforming", "passive": "passive"}


def md(df, index=True):
    """A pandas table as Markdown (without the tabulate package)."""
    d = df.reset_index() if index and not isinstance(df.index, pd.RangeIndex) else df
    fmt = lambda v: f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v)
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    rule = "|" + "|".join("---" for _ in d.columns) + "|"
    return "\n".join([head, rule] + ["| " + " | ".join(fmt(v) for v in row) + " |"
                                      for row in d.itertuples(index=False)])


def half(file):
    """Stable 70/30 split: 'cal' for calibration, 'val' for validation."""
    return "cal" if int(hashlib.md5(file.encode()).hexdigest(), 16) % 10 < 7 else "val"


def load(folder):
    objs, fits, params = [], [], []
    for path in glob.glob(os.path.join(folder, "*.json")):
        with open(path) as fh:
            o = json.load(fh)
        split = half(o["file"])
        cl = o.get("classify") or {}
        objs.append({"file": o["file"], "group": o["group"], "z": o["z"], "sn": o["sn"], "split": split,
                     "label": cl.get("label"), "uncertain": cl.get("uncertain"), "error": o.get("error")})
        for f in o.get("fits", []):
            rec = {k: f.get(k) for k in ("variant", "preset", "time", "chi2", "dof", "rchi2", "bic",
                                         "n_free", "nfev", "success", "start", "error")}
            rec.update(file=o["file"], group=o["group"], z=o["z"], sn=o["sn"], split=split,
                       range=tuple(np.round(f.get("range", [np.nan, np.nan]), -1)),
                       n_limit=len(f.get("at_limit", [])), at_limit=f.get("at_limit", []))
            fits.append(rec)
            for name, (val, err, lo, hi, lin) in (f.get("params") or {}).items():
                params.append({"file": o["file"], "group": o["group"], "variant": f["variant"],
                               "split": split, "param": name, "value": val, "err": err, "min": lo,
                               "max": hi, "linear": lin, "at_limit": name in f.get("at_limit", [])})
    return pd.DataFrame(objs), pd.DataFrame(fits), pd.DataFrame(params)


def nice(x, up):
    """Round a limit outwards to 2 significant digits."""
    if not np.isfinite(x) or x == 0:
        return x
    mag = 10 ** (np.floor(np.log10(abs(x))) - 1)
    return float((np.ceil if up else np.floor)(x / mag) * mag)


def param_table(params, fits):
    import fantasy_agn as fa
    rows = []
    defaults = fits.drop_duplicates(["group", "variant"]).set_index(["group", "variant"])
    for (group, variant), d in params[~params.linear].groupby(["group", "variant"]):
        preset = defaults.loc[(group, variant), "preset"]
        cal = d[d.split == "cal"]
        for name, p in cal.groupby("param"):
            v = p.value.to_numpy()
            q = np.percentile(v, [0.5, 5, 50, 95, 99.5])
            lo, hi = p["min"].iloc[0], p["max"].iloc[0]
            at_lo = np.mean(np.isclose(v, lo, rtol=1e-3, atol=1e-6))
            at_hi = np.mean(np.isclose(v, hi, rtol=1e-3, atol=1e-6))
            span = q[4] - q[0]
            # suggested limits: the 0.5-99.5% range plus 20% of it, but never narrower
            # than now where many fits end at a limit (that limit is too tight)
            new_lo = nice(q[0] - 0.2 * span, up=False) if at_lo < 0.02 else lo
            new_hi = nice(q[4] + 0.2 * span, up=True) if at_hi < 0.02 else hi
            if at_lo >= 0.02:
                new_lo = nice(lo - 0.5 * (hi - lo), up=False) if np.isfinite(lo) else lo
            if at_hi >= 0.02:
                new_hi = nice(hi + 0.5 * (hi - lo), up=True) if np.isfinite(hi) else hi
            rows.append({"group": group, "variant": variant, "preset": preset, "param": name, "n": len(v),
                         "p0.5": q[0], "p5": q[1], "median": q[2], "p95": q[3], "p99.5": q[4],
                         "min": lo, "max": hi, "at_min": at_lo, "at_max": at_hi,
                         "suggest_value": float(np.round(q[2], 3)), "suggest_min": new_lo,
                         "suggest_max": new_hi})
    return pd.DataFrame(rows)


def bic_preference(fits):
    """Within a group, among variants fitted over the same range: how often each wins."""
    rows = []
    ok = fits[fits.error.isna() & fits.bic.notna()]
    for (group, rng), d in ok.groupby(["group", "range"]):
        if d.variant.nunique() < 2:
            continue
        for file, f in d.groupby("file"):
            best = f.loc[f.bic.idxmin()]
            for _, r in f.iterrows():
                rows.append({"group": group, "variant": r.variant, "win": r.variant == best.variant,
                             "dbic": r.bic - best.bic})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (df.groupby(["group", "variant"])
              .agg(n=("win", "size"), wins=("win", "mean"), median_dbic=("dbic", "median"))
              .reset_index())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/.fantasy_agn/samples")
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    objs, fits, params = load(os.path.join(out, "fits"))
    dest = os.path.join(out, "analysis")
    os.makedirs(dest, exist_ok=True)
    fits.to_csv(os.path.join(dest, "fits.csv"), index=False)
    params.to_csv(os.path.join(dest, "params_all.csv"), index=False)
    lines = [f"# FANTASY calibration: {len(objs)} spectra, {len(fits)} fits\n"]

    ok = fits[fits.error.isna()]
    q = (fits.groupby(["group", "variant"])
             .agg(n=("file", "size"), failed=("error", lambda e: e.notna().sum()))
             .join(ok.groupby(["group", "variant"]).agg(
                 rchi2=("rchi2", "median"), rchi2_90=("rchi2", lambda x: x.quantile(0.9)),
                 nfev=("nfev", "median"), time=("time", "median"), at_limit=("n_limit", "mean"),
                 free=("n_free", "median"))))
    lines += ["## Fit quality per group and variant\n", md(q.round(2)), ""]

    errs = fits[fits.error.notna()]
    if len(errs):
        failed = errs.groupby(["group", "variant", "error"]).size().rename("n").reset_index().head(40)
        lines += ["## Failed fits\n", md(failed, index=False), ""]

    lim = (params.groupby(["group", "variant", "param"]).at_limit.mean().rename("fraction")
                 .reset_index().query("fraction >= 0.1").sort_values("fraction", ascending=False))
    lines += ["## Parameters at a limit in at least 10% of the fits\n", md(lim.round(2), index=False), ""]

    pref = bic_preference(fits)
    if len(pref):
        lines += ["## BIC preference (same group and wavelength range)\n",
                  md(pref.round(2), index=False), ""]

    pt = param_table(params, fits)
    pt.to_csv(os.path.join(dest, "params.csv"), index=False)
    cols = ["group", "variant", "param", "n", "p0.5", "median", "p99.5", "min", "max", "at_min", "at_max",
            "suggest_value", "suggest_min", "suggest_max"]
    lines += ["## Nonlinear parameters (calibration half): distribution, current and suggested limits\n",
              md(pt[cols].round(3), index=False), ""]

    cl = objs[objs.label.notna()]
    if len(cl):
        conf = pd.crosstab(cl.group, cl.label)
        agree = np.mean([EXPECTED.get(g, "").split("_")[0] == str(lab).split("_")[0]
                         for g, lab in zip(cl.group, cl.label) if g in EXPECTED])
        lines += [f"## Classification against the SDSS/MPA-JHU groups (agreement {agree:.0%})\n",
                  md(conf), ""]
    path = os.path.join(dest, "report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
