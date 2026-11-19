"""
Fitting a sample of spectra: each spectrum is read, corrected for Galactic
extinction, moved to the rest frame, classified and fitted with the model
the classification recommends (with the host check for borderline
starlight), in parallel processes. The result is one catalog table; with an
output folder every fit is also saved (``.fantasy``) with its figures.

    s = fa.fit_sample(glob.glob("data/*.fits"), out="fits/")
    s.table                       # one row per spectrum
    s.table.to_csv("catalog.csv")
    r = s.load(3)                 # the full fit of row 3
    s.gallery()                   # HTML page with every fit (for checking)

In scripts on macOS and Windows put the code under
``if __name__ == "__main__":`` so that the processes can start.
"""
import os
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

__all__ = ("fit_sample", "SampleResult")


def _row_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def _thumb(r, n=160):
    """The fitted range of a fit in ``n`` bins: wavelength, data and model (a small picture of the fit)."""
    m = r.mask & np.isfinite(r.spectrum.flux)
    x, y, f = r.spectrum.wave[m], r.spectrum.flux[m], r.model_flux()[m]
    if x.size < 2:
        return None
    parts = np.array_split(np.arange(x.size), min(n, x.size))
    pick = lambda a: [float(f"{np.mean(a[p]):.4g}") for p in parts]      # noqa: E731
    return {"x": pick(x), "y": pick(y), "m": pick(f)}


def _live_item(k, row):
    """What the app shows of one finished spectrum (and the thumbnail leaves the row)."""
    num = lambda v: float(v) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else None  # noqa: E731
    return {"k": k, "name": str(row.get("name") or row.get("file") or k), "cls": row.get("cls"),
            "preset": row.get("preset"), "rchi2": num(row.get("rchi2")), "n_at_limit": row.get("n_at_limit"),
            "error": row.get("error"), "time": num(row.get("time")), "thumb": row.pop("_thumb", None)}


def _process(job):
    """One spectrum: read, classify, fit, measure; returns a catalog row (dict)."""
    source, prepared, index, opts = job
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")
    import fantasy_agn as fa
    from .plotting import main_line
    from .refine import EXTENSIONS
    t0 = time.time()
    path = None if isinstance(source, fa.Spectrum) else os.fspath(source)
    row = {"file": os.path.basename(path) if path is not None else source.name, "path": path}
    try:
        spec = prepared
        steps = opts.get("prepare")
        default_preparation = prepared is None and steps is None
        if spec is None:
            spec = source if path is None else fa.read(path, format=opts["format"])
            if default_preparation:
                if opts.get("z") is not None:
                    spec = spec.copy(z=float(opts["z"]))
                if opts.get("deredden", True):
                    spec = spec.deredden(ebv=opts.get("ebv"))
                spec = spec.to_rest_frame()
                if opts.get("mask_absorption"):
                    spec = spec.mask_absorption()
            else:
                for step in steps:
                    spec = step(spec)
        if spec.dereddened:
            row["ebv_gal"] = spec.meta.get("ebv")
        row.update(name=spec.name, z=spec.z, ra=spec.ra, dec=spec.dec)
        if not spec.rest_frame and spec.z != 0:
            raise ValueError("sample presets require rest wavelengths: move this spectrum to the rest frame "
                             "in Prepare, or set z=0 if its wavelengths are already rest wavelengths")
        preset, kw = opts.get("model") or (None, {})
        c = None
        if preset is None:
            c = fa.classify(spec)
            row.update(cls=c.label, cls_uncertain=bool(c.uncertain))
            preset, kw = c.RECOMMENDED[c.label]
        mask_sky = opts.get("mask_sky", "auto")
        if default_preparation and (mask_sky is True or (mask_sky == "auto" and preset in
                ("starforming", "seyfert2", "composite", "liner", "passive", "type2", "galaxy"))):
            spec = spec.mask_sky()
        row["preset"] = preset + (f"({', '.join(f'{k}={v!r}' for k, v in kw.items())})" if kw else "()")
        r = fa.fit(spec, getattr(fa.presets, preset)(**kw), verbose=False)
        # borderline starlight: keep the host if the BIC prefers it
        if c is not None and c.measurements.get("host_borderline") and preset in ("optical", "hbeta", "halpha") \
                and "host" not in kw:
            r_host = fa.fit(spec, getattr(fa.presets, preset)(**dict(kw, host=True)), verbose=False)
            row["host_dbic"] = r_host.bic - r.bic
            if r_host.bic < r.bic - 10:
                r = r_host
                row["preset"] = row["preset"][:-1] + (", " if kw else "") + "host=True)"
        # high S/N narrow-line galaxies: extensions of the preset where the BIC wants them
        if opts.get("refine", True) and preset in EXTENSIONS and r.rchi2 > 1.2:
            r = fa.refine(spec, preset, kw, result=r)
            kept = r.refinement[r.refinement.kept].extension.tolist()
            if kept:
                row["preset"] = preset + "(" + ", ".join(f"{k}={v!r}" for k, v in r.preset_options.items()) + ")"
                row["refined"] = "; ".join(kept)
        # weak Fe II: fit again without the multiplets the data do not detect (fa.prune_feii); the
        # position before pruning is kept, to see what it changed
        prune = opts.get("prune_feii")
        if prune and r._feii_components():
            try:
                for comp in r._feii_components():
                    fe = r.feii(comp.name)
                    row[fe["ratio_label"] + " all groups"] = fe["r_feii"]
                hb = [c.name for c in r._line_components() if c.role == "broad" and "Hb" in c.line_names]
                if hb:
                    row["broad Hb FWHM all groups"] = r.measure("Hb", hb)["fwhm"]
                r = fa.prune_feii(r, nsigma=2.0 if prune is True else float(prune))
                row["feii_groups"], row["feii_dropped"] = len(r.feii_pruning), len(r.feii_pruned)
            except Exception as exc:                 # noqa: BLE001  (the fit itself stands)
                row["feii_note"] = f"not pruned: {type(exc).__name__}: {exc}"
        row.update(rchi2=r.rchi2, n_free=r.n_free, n_at_limit=len(r.at_limit),
                   at_limit=", ".join(r.at_limit))
        if opts.get("thumb"):
            try:
                row["_thumb"] = _thumb(r)
            except Exception:                        # noqa: BLE001  (only a picture)
                pass
        for q in r.key_results().itertuples():
            row[q.quantity] = q.value
            row[q.quantity + " err"] = q.error
        if opts.get("diagnostics", True) and not any(getattr(cp, "role", None) == "broad"
                                                     for cp in r.model.components):
            try:
                for q in r.diagnostics().itertuples():
                    if q.quantity == "BPT class":
                        row["BPT class"] = q.note
                    elif q.quantity not in row:
                        row[q.quantity] = q.value
                        row[q.quantity + " err"] = q.error
            except Exception:
                pass
        out = opts.get("out")
        if out:
            filename = os.path.basename(row["name"])
            if prepared is not None or steps is not None or path is None:
                filename = f"{index:06d}_{filename}"
            base = os.path.join(out, filename)
            fa.save(r, base + ".fantasy")
            redshift = f"  z = {spec.z:.4f}" if spec.z is not None else ""
            fig = r.plot(title=f"{row['name']}{redshift}  ({row['preset']}, "
                               f"reduced chi2 = {r.rchi2:.2f})")
            fig.savefig(base + "_fit.png", dpi=80, bbox_inches="tight")
            line = main_line(r)
            if line:
                r.plot_line(line, others=True).savefig(base + "_line.png", dpi=90, bbox_inches="tight")
            row["saved"] = base + ".fantasy"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["trace"] = traceback.format_exc()[-1200:]
    row["time"] = time.time() - t0
    return row


class SampleResult:
    """Result of :func:`fit_sample`: ``table`` (one row per spectrum), ``load(i)``, ``gallery()``."""

    def __init__(self, table, out=None):
        self.table = table
        self.out = out

    def load(self, i):
        """The full FitResult of row ``i`` (needs an output folder)."""
        from .io import load
        path = self.table.iloc[i].get("saved") if "saved" in self.table else None
        if not isinstance(path, str):
            raise ValueError("fits are only kept with fit_sample(..., out=folder)")
        return load(path)

    @property
    def failed(self):
        return self.table[self.table["error"].notna()] if "error" in self.table else self.table.iloc[0:0]

    def gallery(self, path=None, sort="rchi2"):
        """An HTML page with the figure of every fit, worst first (needs an output folder)."""
        import html
        if not self.out:
            raise ValueError("figures are only made with fit_sample(..., out=folder)")
        path = path or os.path.join(self.out, "gallery.html")
        t = self.table.sort_values(sort, ascending=False) if sort in self.table else self.table
        parts = [f"<h1>FANTASY sample: {len(t)} spectra</h1>"]
        for row in t.itertuples():
            d = row._asdict()
            if not isinstance(d.get("saved"), str):
                parts.append(f"<div class=fig><p><b>{html.escape(str(d.get('file')))}</b>: "
                             f"{html.escape(str(d.get('error')))}</p></div>")
                continue
            base = os.path.basename(d["saved"])[:-8]
            redshift = f" · z = {d['z']:.4f}" if pd.notna(d.get("z")) else ""
            info = (f"<b>{html.escape(str(d.get('name')))}</b>{redshift} · "
                    f"{html.escape(str(d.get('cls', '')))} · {html.escape(str(d.get('preset')))} · "
                    f"reduced chi2 {d.get('rchi2', np.nan):.2f}"
                    + (f" · at limits: {html.escape(str(d.get('at_limit')))}" if d.get("at_limit") else ""))
            parts.append(f'<div class=fig><p>{info}</p><img src="{base}_fit.png">'
                         f'<img class=line src="{base}_line.png"></div>')
        with open(path, "w") as fh:
            fh.write("<!doctype html><meta charset=utf-8><title>FANTASY sample</title><style>"
                     "body{font:14px system-ui;margin:20px;max-width:1300px}.fig{border-top:1px solid #ddd;"
                     "margin:14px 0}img{max-width:62%;vertical-align:top}img.line{max-width:34%}</style>"
                     + "\n".join(parts))
        return path

    def __repr__(self):
        n = len(self.table)
        bad = len(self.failed)
        cls = self.table["cls"].value_counts().to_dict() if "cls" in self.table else {}
        return (f"SampleResult: {n} spectra, {n - bad} fitted, {bad} failed"
                + (f"; classes {cls}" if cls else "") + (f"; saved in {self.out}" if self.out else ""))


def fit_sample(files, out=None, model=None, deredden=True, ebv=None, z=None, mask_absorption=False,
               mask_sky="auto", refine=True, diagnostics=True, prune_feii=False, n_jobs=-1, verbose=True,
               *, prepared=None, prepare=None, format="auto"):
    """
    Classify and fit many spectra in parallel; returns a :class:`SampleResult`.

    :param files: ordered paths (any format ``fa.read`` knows) or raw Spectrum objects.
    :param prepared: spectra already prepared for the first rows. They are fitted exactly as supplied.
    :param prepare: ordered callables taking and returning a Spectrum for the remaining rows.
        Each worker applies them once, replacing the default extinction, rest-frame and masking
        steps. Use picklable callables such as ``operator.methodcaller`` for parallel fits.
    :param format: file format used to read paths.
    :param out: folder for the fits (``.fantasy``) and figures (made if needed); None: catalog only.
    :param model: None (default): the model the classification recommends for each spectrum;
        or a preset name ('hbeta', 'seyfert2', ...) or (name, {options}) for all spectra.
    :param deredden: correct the Galactic extinction (looked up, or ``ebv``).
    :param z: redshift for all spectra (when the files have none).
    :param mask_absorption: mask narrow absorption lines (CIV, MgII quasars).
    :param mask_sky: mask telluric bands and strong sky lines ('auto': for the
        narrow-line galaxy presets, whose high-S/N fits they would dominate).
    :param refine: for narrow-line galaxies with reduced chi2 > 1.2, add the preset
        extensions the BIC prefers (:func:`fantasy_agn.refine`).
    :param prune_feii: fit again without the Fe II multiplets the data do not detect
        (:func:`fantasy_agn.prune_feii`; True: below 2 sigma, or the number of sigma). Weak Fe II comes
        out too high otherwise, which moves the weak-Fe II end of a main sequence diagram to the right;
        with moderate Fe II pruning removes real weak multiplets too, so it is a choice. The catalog gets
        ``feii_groups``, ``feii_dropped`` and the band-specific ratios before pruning (for example,
        ``R_FeII all groups`` for optical Fe II), plus ``broad Hb FWHM all groups`` when applicable.
    :param n_jobs: processes (-1: all cores but one).
    """
    from ._threads import SINGLE_THREADED, environment
    from .fitting import _n_workers
    from .progress import enabled, report
    files = list(files)
    prepared = list(prepared) if prepared is not None else []
    if len(prepared) > len(files):
        raise ValueError("more prepared spectra than input rows")
    prepare = tuple(prepare) if prepare is not None else None
    if isinstance(model, str):
        model = (model, {})
    if out:
        os.makedirs(out, exist_ok=True)
    opts = dict(out=out, model=model, deredden=deredden, ebv=ebv, z=z, mask_absorption=mask_absorption,
                mask_sky=mask_sky, refine=refine, diagnostics=diagnostics, prune_feii=prune_feii,
                prepare=prepare, format=format, thumb=enabled())
    jobs = [(f, prepared[k] if k < len(prepared) else None, k, opts) for k, f in enumerate(files)]
    n = len(jobs)
    workers = _n_workers(n_jobs, n)
    rows, t0, task = [None] * n, time.time(), f"fitting a sample of {n} spectra"
    report(task, 0, n, item={})
    done = 0
    if workers > 1:
        try:
            with environment(SINGLE_THREADED), ProcessPoolExecutor(workers) as ex:
                futures = {ex.submit(_process, job): k for k, job in enumerate(jobs)}
                for fut in as_completed(futures):
                    k = futures[fut]
                    rows[k] = fut.result()
                    done += 1
                    report(task, done, n, f"{workers} processes", item=_live_item(k, rows[k]))
        except Exception as exc:                                   # e.g. no __main__ guard
            warnings.warn(f"parallel fitting failed ({type(exc).__name__}); running serially")
            rows = [None] * n
            done = 0
    for k, job in enumerate(jobs):
        if rows[k] is None:
            rows[k] = _process(job)
            done += 1
            report(task, done, n, "1 process", item=_live_item(k, rows[k]))
    for row in rows:
        row.pop("_thumb", None)
    table = pd.DataFrame(rows)
    first = [c for c in ("name", "file", "z", "cls", "preset", "rchi2", "n_at_limit") if c in table]
    table = table[first + [c for c in table.columns if c not in first]]
    result = SampleResult(table, out)
    if out:
        table.drop(columns=[c for c in ("trace",) if c in table]).to_csv(os.path.join(out, "catalog.csv"),
                                                                          index=False)
    if verbose:
        print(f"{result} ({time.time() - t0:.0f} s)")
    return result
