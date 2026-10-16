"""
Figures of calibration fits for inspection: per group the worst fits (highest
reduced chi2), those with most parameters at a limit and a random set,
refitted with the group's preset (fits are deterministic) and collected in one
HTML page per group: <out>/gallery/<group>.html.

    python calibration/gallery.py [--groups t1_hb sy2] [--n 8] [--variant NAME] [--jobs 3]
"""
import argparse
import glob
import html
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from fit_sample import VARIANTS


def pick(df, n, seed=1):
    """Worst, most limited and random fits (without repeats)."""
    out, seen = [], set()
    for kind, order in (("worst", df.sort_values("rchi2", ascending=False)),
                        ("limits", df.sort_values("n_limit", ascending=False)),
                        ("random", df.sample(frac=1, random_state=seed))):
        k = 0
        for r in order.itertuples():
            if k >= n:
                break
            if r.file not in seen:
                seen.add(r.file)
                out.append((kind, r))
                k += 1
    return out


def draw(job):
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")
    import fantasy_agn as fa
    file, group, variant, ebv, spectra, dest = job
    png = os.path.join(dest, f"{group}_{variant.replace(' ', '_').replace('+', 'p')}_{file[:-5]}.png")
    if os.path.exists(png):
        return png, None
    try:
        label, preset, kw = next(v for v in VARIANTS[group] if v[0] == variant)
        spec = fa.read(os.path.join(spectra, file))
        if ebv is not None and np.isfinite(ebv):
            spec = spec.deredden(ebv=0.86 * float(ebv))
        r = fa.fit(spec.to_rest_frame(), getattr(fa.presets, preset)(**kw), verbose=False)
        fig = r.plot(title=f"{file[:-5]}  z = {spec.z:.3f}  ({variant}, reduced chi2 = {r.rchi2:.2f})")
        fig.savefig(png, dpi=80, bbox_inches="tight")
        return png, None
    except Exception as exc:
        return png, f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/.fantasy_agn/samples")
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--variant", default=None, help="default: the group's preset as it is now")
    ap.add_argument("--n", type=int, default=8, help="figures per kind (worst, limits, random)")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    dest = os.path.join(out, "gallery")
    os.makedirs(dest, exist_ok=True)
    ebv = pd.read_csv(os.path.join(out, "ebv.csv"))
    sample = pd.read_csv(os.path.join(out, "sample.csv")).merge(ebv, on=["plate", "mjd", "fiberID"], how="left")
    ebv_of = dict(zip(sample.file, sample.ebv_sfd))
    rows = []
    for p in glob.glob(os.path.join(out, "fits", "*.json")):
        o = json.load(open(p))
        for f in o.get("fits", []):
            if "rchi2" in f:
                rows.append({"file": o["file"], "group": o["group"], "variant": f["variant"], "rchi2": f["rchi2"],
                             "n_limit": len(f["at_limit"]), "at_limit": ", ".join(f["at_limit"]),
                             "label": (o.get("classify") or {}).get("label"), "z": o["z"], "sn": o["sn"]})
    fits = pd.DataFrame(rows)
    for group in (a.groups or sorted(fits.group.unique())):
        variant = a.variant or VARIANTS[group][0][0]
        d = fits[(fits.group == group) & (fits.variant == variant)]
        if d.empty:
            print(f"{group}: no fits yet")
            continue
        chosen = pick(d, a.n)
        jobs = [(r.file, group, variant, ebv_of.get(r.file), os.path.join(out, "spectra"), dest) for _, r in chosen]
        with ProcessPoolExecutor(a.jobs) as ex:
            results = list(ex.map(draw, jobs))
        parts = [f"<h1>{html.escape(group)}: {html.escape(variant)}</h1>",
                 f"<p>{len(d)} fits; median reduced chi2 {d.rchi2.median():.2f}; "
                 f"parameters at a limit per fit {d.n_limit.mean():.1f}</p>"]
        for (kind, r), (png, err) in zip(chosen, results):
            info = (f"<b>{kind}</b> · {html.escape(r.file)} · z = {r.z:.3f} · S/N {r.sn:.0f} · "
                    f"reduced chi2 {r.rchi2:.2f}" + (f" · classify: {html.escape(str(r.label))}" if r.label else "")
                    + (f"<br>at limits: {html.escape(r.at_limit)}" if r.at_limit else ""))
            img = (f'<img src="{os.path.basename(png)}">' if err is None
                   else f"<pre>{html.escape(err)}</pre>")
            parts.append(f'<div class="fig"><p>{info}</p>{img}</div>')
        page = os.path.join(dest, f"{group}.html")
        with open(page, "w") as fh:
            fh.write("<!doctype html><meta charset=utf-8><title>FANTASY calibration: " + html.escape(group)
                     + "</title><style>body{font:14px system-ui;margin:20px;max-width:1100px}"
                       ".fig{margin:18px 0;border-top:1px solid #ddd}img{max-width:100%}</style>"
                     + "\n".join(parts))
        print(f"{group}: {len(chosen)} figures -> {page}", flush=True)


if __name__ == "__main__":
    main()
