"""
The step-by-step workflow of the app: Read -> Prepare -> Model -> Fit -> Results,
for one spectrum, a sample, several epochs of one object, or classification only.

As everywhere in the app, every action writes code that runs in the session
namespace, so the session panel (and the downloaded script) repeats it:

    files     paths of the spectra
    raw       the spectra as read
    spectra   the prepared spectra (dereddened, rest frame, cropped, masked)
    spec      the prepared spectrum (one spectrum, classification)
    model     the model (step 3)
    r         fit of one spectrum        fits     epochs fitted on their own
    me        joint fit of the epochs    sample   fa.fit_sample result
    classes   classifications (classification only)
"""
import datetime
import glob
import io
import os
import re

import numpy as np

from ..spectrum import date_from_name, mjd_to_date

MODES = ("single", "sample", "multi", "classify")
MAX_POINTS = 4000
TELLURIC_NAMES = ("B", "H₂O", "A")         # Spectrum.TELLURIC, in order


def _num(v):
    return None if v in (None, "") else float(v)


def _g(v):
    """Short repr of a number for generated code."""
    return f"{float(v):g}"


def _stride(n, max_points=MAX_POINTS):
    return max(1, int(np.ceil(n / max_points)))


def _arr(a, idx):
    out = np.asarray(a, float)[idx]
    return [round(float(v), 6) if np.isfinite(v) else None for v in out]


def _runs(mask, x):
    """Contiguous runs of True in mask as [x_lo, x_hi] pairs."""
    out = []
    if not mask.any():
        return out
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
        lo, hi = x[a], x[min(b, x.size) - 1]
        out.append([float(lo), float(hi)])
    return out


def _fmt_err(v, e, digits=4):
    if v is None or not np.isfinite(v):
        return "—"
    s = f"{v:.{digits}g}"
    if e is not None and np.isfinite(e) and e > 0:
        s += f" ± {e:.2g}"
    return s


class StepsMixin:
    """Workflow methods of :class:`fantasy_agn.app.session.Session`."""

    def _wf_reset(self):
        self.wf = {"mode": "single", "group": "auto", "fit_kind": None}
        self._wf_cache = {}

    # ----------------------------------------------------------- objects
    def _list(self, name):
        from ..spectrum import Spectrum
        v = self.ns.get(name)
        if isinstance(v, (list, tuple)) and v and all(isinstance(s, Spectrum) for s in v):
            return list(v)
        return None

    @property
    def raw(self):
        return self._list("raw")

    @property
    def spectra(self):
        return self._list("spectra")

    @property
    def fits(self):
        from ..result import FitResult
        v = self.ns.get("fits")
        if isinstance(v, list) and v and any(isinstance(r, FitResult) for r in v):
            return v
        return None

    @property
    def classes(self):
        from ..classify import Classification
        v = self.ns.get("classes")
        if isinstance(v, list) and v and all(isinstance(c, Classification) for c in v):
            return v
        return None

    def _classifications(self):
        """The classifications to show: those of the classification mode, else the one made by
        'Suggest from classification' (``c``), which stays available after modelling and fitting."""
        from ..classify import Classification
        if self.classes is not None:
            return self.classes
        c, sp = self.ns.get("c"), self.spectra or []
        names = {s.name for s in sp}
        # every epoch (multi-epoch 'Suggest from classification'), while they are the spectra read
        ep = self.ns.get("epoch_classes")
        if (isinstance(ep, list) and len(ep) > 1 and all(isinstance(x, Classification) for x in ep)
                and {x.spectrum.name for x in ep} == names):
            return ep
        # only while it belongs to one of the spectra read (not one from an earlier object)
        ok = isinstance(c, Classification) and c.spectrum.name in names
        return [c] if ok else None

    def _replace_last(self, marker):
        """Index of the last cell if it was made by the same workflow action (rerun it in place)."""
        if self.cells and self.cells[-1].code.startswith(marker) and len(self.cells) > 1:
            return len(self.cells) - 1
        return None

    def run_step(self, code, marker=None):
        """Run workflow code; a repeated action replaces its own previous cell."""
        index = self._replace_last(marker) if marker else None
        return self.run_cell(code, index)

    # -------------------------------------------------------------- 1 Read
    def wf_read_code(self, mode="single", paths=None, pattern=None, targetid=None, sdss=None,
                     group="auto", format="auto"):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        if group not in ("auto", "date", "mjd", "order"):
            raise ValueError("group epochs by auto, date, mjd or order")
        format = format or "auto"
        if format not in ("auto", "sdss", "desi", "ozdes", "ozdes-ssa", "gama", "fits1d", "text"):
            raise ValueError(f"unknown file format {format!r}")
        self.wf.update(mode=mode, group=group, fit_kind=None, format=format)
        read = "fa.read(f)" if format == "auto" else f"fa.read(f, format={format!r})"
        for key in ("prepare", "reference"):          # settings of the previous spectra
            self.wf.pop(key, None)
        lines = []
        if targetid:
            ids = [int(t) for t in re.split(r"[\s,;]+", str(targetid).strip()) if t]
            if mode == "single" and len(ids) > 1:
                raise ValueError("one TARGETID for a single spectrum (or choose Sample / Multi-epoch)")
            lines.append(f"raw = [fa.desi.fetch(t) for t in {ids!r}]   # DESI DR1 via SPARCL")
        elif sdss:
            ids = []
            for item in re.split(r"[\s,;]+", str(sdss).strip()):
                parts = [p for p in re.split(r"[-/]", item) if p]
                if len(parts) != 3 or not all(p.isdigit() for p in parts):
                    raise ValueError(f"SDSS spectra as plate-mjd-fiber, e.g. 1059-52618-453 (got {item!r})")
                ids.append(tuple(int(p) for p in parts))
            if mode == "single" and len(ids) > 1:
                raise ValueError("one plate-mjd-fiber for a single spectrum (or choose Sample / Multi-epoch)")
            lines.append(f"raw = [fa.sdss.fetch(*pmf) for pmf in {ids!r}]   # SDSS DR17 (plate, mjd, fiber)")
        else:
            if pattern:
                files = sorted(glob.glob(os.path.expanduser(pattern.strip())))
                if not files:
                    raise ValueError(f"no files match {pattern!r}")
            elif paths:
                files = [os.path.expanduser(str(p).strip()) for p in paths if str(p).strip()]
            else:
                raise ValueError("choose files, give a path or pattern, a DESI TARGETID or an SDSS plate-mjd-fiber")
            if mode == "single" and len(files) > 1:
                raise ValueError(f"{len(files)} files for a single spectrum: choose Sample or Multi-epoch")
            if mode == "multi" and len(files) < 2:
                raise ValueError("a multi-epoch fit needs at least two spectra")
            if mode == "multi" and group in ("auto", "date"):
                dated = [date_from_name(f) for f in files]
                if all(dated):
                    group = "date"
                    files = [f for _, f in sorted(zip(dated, files))]
                elif group == "date":
                    raise ValueError("no date (YYYYMMDD) in the name of " +
                                     ", ".join(os.path.basename(f) for f, d in zip(files, dated) if not d)
                                     + ": choose 'automatic' (MJD from the headers) or file order")
                else:
                    group = "mjd"                     # e.g. SDSS spec-PLATE-MJD-FIBER files
            if pattern and len(files) > 40:
                lines += ["import glob", f"files = sorted(glob.glob({os.path.expanduser(pattern.strip())!r}))"]
            elif len(files) == 1:
                lines.append(f"files = [{files[0]!r}]")
            else:
                lines.append("files = [" + ",\n         ".join(repr(f) for f in files) + "]")
            if mode == "sample" and len(files) > 20:
                lines.append(f"raw = [{read} for f in files[:20]]   # a few to look at; the fit reads all")
            else:
                lines.append(f"raw = [{read} for f in files]")
            if mode == "multi" and group == "date":
                lines += ["for s, f in zip(raw, files):          # observation dates from the file names",
                          "    s.mjd = fa.spectrum.mjd_from_name(f)"]
        if mode == "multi" and group in ("auto", "mjd"):
            lines += ["# epochs in time order (MJD from the file headers)",
                      "order = sorted(range(len(raw)), key=lambda k: raw[k].mjd if raw[k].mjd is not None else float('inf'))",
                      "raw = [raw[k] for k in order]"]
            if not (targetid or sdss):
                lines.append("files = [files[k] for k in order]")
            lines.append('missing = [s.name for s in raw if s.mjd is None]\n'
                         'if missing:\n    print("no MJD in the header of", ", ".join(missing), "(kept at the end)")')
        # the model is kept (e.g. after removing an epoch); preparation and fits start again
        lines.append("spectra = r = me = fits = classes = None")
        lines.append('print(len(raw), "spectra:", ", ".join(s.name for s in raw[:8]))')
        return "# read\n" + "\n".join(ln for ln in lines if ln)

    def loaded_state(self):
        raw = self.raw
        if raw is None:
            return None
        key = ("loaded", id(self.ns.get("raw")), len(raw))
        if key in self._wf_cache:
            return self._wf_cache[key]
        files = self.ns.get("files") if isinstance(self.ns.get("files"), list) else None
        rows = []
        for k, s in enumerate(raw):
            path = files[k] if files and k < len(files) else s.meta.get("path") or s.name
            fname = os.path.basename(str(path))
            date = mjd_to_date(s.mjd) or date_from_name(fname)
            ext = fname.lower().rsplit(".", 1)[-1] if "." in fname else ""
            if ext in ("txt", "dat", "csv", "ascii", "asc", "tsv"):
                fmt = f"{ext} · λ, flux" + (", err" if not s.meta.get("err_estimated") else " (err estimated)")
            elif ext in ("fits", "fit", "fts"):
                try:
                    from ..spectrum import _detect_format
                    kind = self.wf.get("format") if self.wf.get("format") not in (None, "auto") else _detect_format(str(path))
                except Exception:
                    kind = "fits"
                fmt = {"sdss": "SDSS spec FITS", "desi": "DESI coadd", "fits1d": "1D FITS", "gama": "GAMA FITS",
                       "ozdes": "OzDES FITS", "ozdes-ssa": "OzDES FITS"}.get(kind, "FITS")
            else:
                fmt = "DESI" if "targetid" in {k.lower() for k in s.meta} else (ext or "—")
            good = s.good
            rows.append({"k": k, "date": date, "mjd": s.mjd, "file": fname, "name": s.name, "format": fmt,
                         "path": str(path) if files else None,
                         "ra": s.ra, "dec": s.dec, "z": s.z,
                         "range": [float(s.wave[good].min()), float(s.wave[good].max())] if good.any() else None,
                         "npix": int(good.sum()), "flux_unit": s.flux_unit,
                         "flux_unit_source": s.meta.get("flux_unit_source")
                         or (f"{fmt} header" if s.flux_unit else None)})
        n_all = len(files) if files else len(raw)
        out = {"rows": rows, "n": n_all, "shown": len(raw), "id": id(self.ns.get("raw"))}
        self._wf_cache[key] = out
        return out

    # ----------------------------------------------------------- 2 Prepare
    def wf_prepare_code(self, z=None, z_source="file", rest_frame=True, extinction="coords", ebv=None,
                        ra=None, dec=None, wmin=None, wmax=None, telluric=True, sky_lines=False,
                        absorption=False, masks=(), apply_to="all", flux_unit="file"):
        raw = self.raw
        if raw is None:
            raise ValueError("read the spectra first (step 1)")
        chain = []
        if flux_unit not in (None, "", "file"):
            try:
                unit = _num(flux_unit)
            except ValueError:
                unit = None
            if unit is None or not 0 < unit < 1:
                raise ValueError("flux unit: give the value of one flux unit in erg/s/cm2/A, e.g. 1e-16")
            chain.append(f".with_flux_unit({unit:g})")
        if ra not in (None, "") or dec not in (None, ""):
            from ..spectrum import parse_coordinates
            parse_coordinates(ra, dec)
            pos = lambda v: repr(float(v)) if _is_number(v) else repr(str(v).strip())
            chain.append(f".with_coordinates({pos(ra)}, {pos(dec)})")
        if extinction == "value":
            if ebv in (None, ""):
                raise ValueError("give E(B-V), or choose 'From coordinates'")
            chain.append(f".deredden(ebv={_g(ebv)})")
        elif extinction == "coords":
            chain.append(".deredden()")
        elif extinction != "none":
            raise ValueError("unknown extinction choice")
        if telluric or sky_lines:
            opts = [] if telluric else ["telluric=False"]
            opts += [] if sky_lines else ["sky_lines=False"]
            chain.append(f".mask_sky({', '.join(opts)})")
        zval = _num(z) if z_source == "manual" else None
        if z_source == "manual" and zval is None:
            raise ValueError("give the redshift z, or take it from the file")
        if rest_frame:
            chain.append(f".to_rest_frame({'z=' + _g(zval) if zval is not None else ''})")
            if zval is None and any(s.z is None for s in raw):
                raise ValueError("these files have no redshift: enter z")
        lo, hi = _num(wmin), _num(wmax)
        if lo is not None or hi is not None:
            if lo is not None and hi is not None and lo >= hi:
                raise ValueError("λ min must be below λ max")
            chain.append(f".crop({_g(lo) if lo is not None else 'None'}, {_g(hi) if hi is not None else 'None'})")
        ranges = []
        for m in masks or ():
            a, b = sorted(float(v) for v in m)
            ranges.append(f"({a:.0f}, {b:.0f})")
        if ranges:
            chain.append(f".mask({', '.join(ranges)})")
        if absorption:
            chain.append(".mask_absorption()")
        expr = "s" + "".join(chain)
        if len(expr) > 60:
            expr = "s" + "\n            ".join(chain)
        lines = ["# prepare"]
        if apply_to not in ("all", None, ""):
            k = int(apply_to)
            if not 0 <= k < len(raw):
                raise ValueError(f"epoch {k} does not exist")
            if self.spectra is None or len(self.spectra) != len(raw):
                raise ValueError("prepare all spectra once before changing one epoch")
            lines.append(f"spectra[{k}] = {expr.replace('s.', f'raw[{k}].', 1) if expr.startswith('s.') else f'raw[{k}]' + expr[1:]}")
        else:
            lines.append(f"spectra = [{expr}\n           for s in raw]" if "\n" in expr else f"spectra = [{expr} for s in raw]")
        if self.wf.get("mode") == "single" or (self.wf.get("mode") == "classify" and len(raw) == 1):
            lines.append("spec = spectra[0]")
        lines.append('print(len(spectra), "spectra · z =", spectra[0].z, "· E(B-V) =", '
                     'spectra[0].meta.get("ebv"), spectra[0].meta.get("ebv_source", ""))')
        self.wf["prepare"] = {"z": zval, "z_source": z_source, "rest_frame": bool(rest_frame),
                              "extinction": extinction, "ebv": _num(ebv), "ra": ra, "dec": dec,
                              "wmin": lo, "wmax": hi, "telluric": bool(telluric), "sky_lines": bool(sky_lines),
                              "absorption": bool(absorption), "masks": [list(map(float, m)) for m in masks or ()],
                              "flux_unit": flux_unit if flux_unit not in (None, "") else "file"}
        return "\n".join(lines)

    def prepared_state(self):
        sp = self.spectra
        if sp is None:
            return None
        s = sp[0]
        good = s.good
        return {"n": len(sp), "z": s.z, "ebv": s.meta.get("ebv"), "ebv_source": s.meta.get("ebv_source"),
                "flux_unit": s.flux_unit,
                "rest_frame": bool(s.rest_frame),
                "range": [float(s.wave[good].min()), float(s.wave[good].max())] if good.any() else None,
                "id": id(self.ns.get("spectra"))}

    def wf_preview(self, k=0, kind="prepared"):
        """Arrays for the Prepare plot: one spectrum, raw (observed) or prepared, with masked ranges."""
        from ..spectrum import Spectrum
        src = self.spectra if kind == "prepared" else self.raw
        if src is None:
            src = self.raw
            kind = "raw"
        if src is None:
            return None
        k = min(max(int(k), 0), len(src) - 1)
        s = src[k]
        fin = np.isfinite(s.flux)
        idx = np.nonzero(fin)[0]
        idx = idx[::_stride(idx.size)]
        z = (s.z or 0.0) if s.rest_frame else 0.0
        bands = [{"lo": a / (1 + z), "hi": b / (1 + z), "label": TELLURIC_NAMES[i]}
                 for i, (a, b) in enumerate(Spectrum.TELLURIC)]
        masked = _runs(~s.good & fin, s.wave) if kind == "prepared" else []
        return {"k": k, "kind": kind, "name": s.name, "x": _arr(s.wave, idx), "flux": _arr(s.flux, idx),
                "masked": masked, "telluric": bands, "rest_frame": bool(s.rest_frame),
                "date": mjd_to_date(s.mjd)}

    # ------------------------------------------------------------- 3 Model
    def wf_reference(self):
        """Index of the spectrum shown in the Model step (the flux reference of a joint fit)."""
        sp = self.spectra or self.raw or []
        ref = self.wf.get("reference")
        if isinstance(ref, int) and 0 <= ref < len(sp):
            return ref
        if len(sp) > 1:                                  # highest median S/N, as fit_epochs does
            snr = [np.nanmedian(np.abs(s.flux[s.good] / s.err[s.good])) if s.good.any() else 0 for s in sp]
            return int(np.argmax(snr))
        return 0

    def wf_guess(self, k=None):
        """Initial guess of the model (data-driven starting values) on one prepared spectrum."""
        from ..fitting import guess
        m, sp = self.model, self.spectra
        if m is None or sp is None:
            return None
        k = self.wf_reference() if k in (None, "") else min(max(int(k), 0), len(sp) - 1)
        s = sp[k]
        sig = (id(self.ns.get("spectra")), k,
               tuple((p.name, p.value, p.min, p.max, p.fixed, p.tie) for p in m.params),
               tuple(c.name for c in m.components))
        key = ("guess", sig)
        if key in self._wf_cache:
            return self._wf_cache[key]
        lo, hi = m.wave_range or (s.wave[s.good].min(), s.wave[s.good].max())
        sel = (s.wave >= lo) & (s.wave <= hi) & np.isfinite(s.flux)
        try:
            g = guess(s, m)
            parts = g.evaluate(s.wave[sel], s.context, components=True)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}", "k": k}
        idx = np.arange(sel.sum())
        idx = idx[::_stride(idx.size)]
        x = s.wave[sel]
        total = sum(parts.values(), np.zeros(x.size))
        out = {"k": k, "name": s.name, "date": mjd_to_date(s.mjd), "x": _arr(x, idx),
               "flux": _arr(s.flux[sel], idx), "model": _arr(total, idx),
               "components": {n: _arr(v, idx) for n, v in parts.items()}}
        if len(self._wf_cache) > 50:
            self._wf_cache.clear()
        self._wf_cache[key] = out
        return out

    # --------------------------------------------------------------- 4 Fit
    def wf_fit_code(self, epochs="joint", reference="auto", n_jobs=-1, shared="", errors="covariance",
                    scale_errors=True, rescale_errors=False, calibration_degree=0, host_per_epoch=False,
                    uncertainty=False, absorbers=False, sample_model="auto", out=None, shared_shape=False):
        mode = self.wf.get("mode")
        self.wf["fit_opts"] = {"epochs": epochs, "reference": reference, "n_jobs": n_jobs, "shared": shared,
                               "shared_shape": bool(shared_shape),
                               "errors": errors, "scale_errors": bool(scale_errors),
                               "rescale_errors": bool(rescale_errors), "calibration_degree": int(calibration_degree or 0),
                               "host_per_epoch": bool(host_per_epoch), "uncertainty": bool(uncertainty),
                               "absorbers": bool(absorbers), "sample_model": sample_model}
        sp = self.spectra
        if mode != "sample" and sp is None:
            raise ValueError("prepare the spectra first (step 2)")
        if errors not in ("covariance", "mc", "mcmc", "none"):
            raise ValueError("errors: covariance, mc, mcmc or none")
        err_arg = None if errors == "none" else errors
        n_jobs = int(n_jobs or -1)
        if mode == "classify":
            self.wf["fit_kind"] = "classify"
            return ("# fit\nclasses = [fa.classify(s) for s in spectra]\nc = classes[0]\n"
                    "[(s.name, c.label) for s, c in zip(spectra, classes)]")
        if mode == "sample":
            return self._wf_sample_code(sample_model, n_jobs, out)
        if self.model is None:
            raise ValueError("create the model first (step 3)")
        if mode == "single":
            if self.spec is None:
                raise ValueError("prepare the spectrum first (step 2)")
            self.wf["fit_kind"] = "single"
            if uncertainty:
                code = self.uncertainty_code()
            else:
                code = self.fit_model_code("covariance" if errors == "none" else errors, absorbers=absorbers)
                if errors == "none":
                    code = code.replace("r = fa.fit(spec, model", "r = fa.fit(spec, model, errors=None", 1)
            if not scale_errors:
                code = re.sub(r"(r(?:_host)? = fa\.fit\(spec, model(?:_host)?)", r"\1, scale_errors=False", code)
            return "# fit\n" + code
        if epochs == "own":
            self.wf["fit_kind"] = "fits"
            opts = [f"n_jobs={n_jobs}"]
            if errors != "covariance":
                opts.append(f"errors={err_arg!r}")
            if not scale_errors:
                opts.append("scale_errors=False")
            return ("# fit\nfits = fa.fit_many(spectra, model, " + ", ".join(opts) + ")   # each epoch on its own\n"
                    "[(f.spectrum.name, round(f.rchi2, 3)) for f in fits if hasattr(f, 'rchi2')]")
        if errors in ("mc", "mcmc"):
            raise ValueError("the joint fit has covariance errors (or none); Monte Carlo and MCMC are for independent fits")
        self.wf["fit_kind"] = "joint"
        patterns = tuple(p.strip() for p in str(shared or "").split(",") if p.strip())
        if shared_shape:
            # the shape of the broad lines (and the Fe II width) is the same in all epochs: only the
            # fluxes vary. Free shapes per epoch trade flux with the continuum and Fe II at SDSS S/N
            # and can wash out the reverberation signal; the explicit list replaces the default
            # (narrow lines and host), so that is added too.
            if not patterns:
                patterns = ("narrow.*",) + tuple(f"{c.name}.*" for c in self.model.components if c.kind == "host")
            patterns = patterns + tuple(self.broad_shape_params())
        opts = []
        if patterns:
            opts.append(f"shared={patterns!r}")
        if reference not in ("auto", None, ""):
            ref = int(reference)
            if not 0 <= ref < len(sp):
                raise ValueError(f"epoch {ref} does not exist")
            opts.append(f"reference={ref}")
            self.wf["reference"] = ref
        else:
            self.wf.pop("reference", None)
        if int(calibration_degree or 0):
            opts.append(f"calibration_degree={int(calibration_degree)}")
        if host_per_epoch:
            hosts = [c.name for c in self.model.components if c.kind == "host"]
            if not hosts:
                raise ValueError("the model has no host galaxy to scale per epoch")
            opts.append(f"epoch_scaled={tuple(hosts)!r}")
        if rescale_errors:
            opts.append("rescale_errors=True")
        if not scale_errors:
            opts.append("scale_errors=False")
        if err_arg is None:
            opts.append("errors=None")
        opts.append(f"n_jobs={n_jobs}")
        return ("# fit\nme = fa.fit_epochs(spectra, model,\n                   " +
                ",\n                   ".join(opts) + ")")

    def broad_shape_params(self):
        """Non-linear parameters that set the shape of the broad lines (shift, widths and the profile
        parameters: span and gamma, h3 and h4, ...) and the width of Fe II: everything except fluxes."""
        m = self.model
        if m is None:
            return []
        names = []
        for c in m.components:
            broad = getattr(c, "role", None) == "broad"
            feii = type(c).__name__ == "FeII"
            if not (broad or feii):
                continue
            for p in c.params:
                if p.linear or p.fixed or p.tie:        # a tied parameter follows its (shared) target
                    continue
                if feii and not p.name.endswith(("fwhm", "shift")):
                    continue
                names.append(p.name)
        return names

    def _wf_sample_code(self, sample_model, n_jobs, out):
        from .session import MODELS
        files = self.ns.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("a sample is fitted from files: read them with Files or Path / glob")
        p = self.wf.get("prepare") or {}
        args = []
        if out is None:
            out = os.path.join(os.path.expanduser("~"), ".fantasy_agn", "sample_fits",
                               f"sample_{datetime.datetime.now():%Y%m%d_%H%M%S}")
        args.append(f"out={out!r}")
        if sample_model not in ("auto", None, ""):
            name, kw = MODELS[sample_model]
            args.append(f"model={(name, kw)!r}" if kw else f"model={name!r}")
        ext = p.get("extinction", "coords")
        if ext == "none":
            args.append("deredden=False")
        elif ext == "value" and p.get("ebv") is not None:
            args.append(f"ebv={p['ebv']:g}")
        if p.get("z_source") == "manual" and p.get("z") is not None:
            args.append(f"z={p['z']:g}")
        if p.get("absorption"):
            args.append("mask_absorption=True")
        if p and not (p.get("telluric") or p.get("sky_lines")):
            args.append("mask_sky=False")
        args.append(f"n_jobs={int(n_jobs)}")
        self.wf["fit_kind"] = "sample"
        return ("# fit\nsample = fa.fit_sample(files, " + ", ".join(args) + ")\n"
                'sample.table[[c for c in ("name", "cls", "preset", "rchi2") if c in sample.table]]')

    # ------------------------------------------------------------ results
    def wf_results(self):
        """[(label, date, FitResult, reference)] of the last fit of the workflow, or None."""
        kind = self.wf.get("fit_kind")
        me, fits, r = self.multi, self.fits, self.result
        if kind == "joint" and me is not None or kind is None and me is not None:
            ref = getattr(me, "reference", 0)
            return [(e.spectrum.name, mjd_to_date(e.spectrum.mjd), e, k == ref) for k, e in enumerate(me.epochs)]
        if kind == "fits" and fits is not None or kind is None and fits is not None:
            return [(f.spectrum.name if hasattr(f, "spectrum") else f"epoch {k}",
                     mjd_to_date(f.spectrum.mjd) if hasattr(f, "spectrum") else None,
                     f if hasattr(f, "spectrum") else None, False) for k, f in enumerate(fits)]
        if r is not None and kind in ("single", None):
            return [(r.spectrum.name, mjd_to_date(r.spectrum.mjd), r, False)]
        return None

    def fit_state(self):
        """Numbers for the Fit step (cached per result object)."""
        kind = self.wf.get("fit_kind")
        if kind == "classify":
            cl = self.classes
            if cl is None:
                return None
            return {"kind": kind, "n": len(cl), "labels": [c.label for c in cl]}
        if kind == "sample":
            sm = self.sample
            if sm is None:
                return None
            t = sm.table
            return {"kind": kind, "n": len(t),
                    "failed": int(t["error"].notna().sum()) if "error" in t else 0,
                    "classes": t["cls"].value_counts().to_dict() if "cls" in t else {}}
        res = self.wf_results()
        if not res:
            return None
        obj = self.multi if kind == "joint" or (kind is None and self.multi is not None) else (
            self.fits if kind == "fits" else self.result)
        key = ("fit", id(obj))
        if key in self._wf_cache:
            return self._wf_cache[key]
        epochs = []
        for k, (name, date, fr, ref) in enumerate(res):
            if fr is None:
                epochs.append({"k": k, "name": name, "date": date, "failed": True})
                continue
            epochs.append({"k": k, "name": name, "date": date, "reference": ref, "rchi2": float(fr.rchi2),
                           "limits": list(fr.at_limit), "scale": float(getattr(fr, "scale", 1.0) or 1.0)})
        if kind == "joint" or (kind is None and self.multi is not None):
            me = self.multi
            out = {"kind": "joint", "chi2": float(me.chi2), "dof": int(me.dof), "rchi2": float(me.rchi2),
                   "error_scale": float(me.error_scale), "time": float(me.time), "nfev": int(me.nfev),
                   "success": bool(me.success), "summary": me.summary(), "epochs": epochs,
                   "reference": int(getattr(me, "reference", 0))}
        elif kind == "fits":
            ok = [f for _, _, f, _ in res if f is not None]
            chi2 = float(sum(f.chi2 for f in ok))
            dof = int(sum(f.dof for f in ok))
            out = {"kind": "fits", "chi2": chi2, "dof": dof, "rchi2": chi2 / dof if dof else None,
                   "error_scale": float(np.median([f.error_scale for f in ok])) if ok else None,
                   "time": float(sum(f.time for f in ok)), "epochs": epochs, "failed": len(res) - len(ok)}
        else:
            r = res[0][2]
            out = {"kind": "single", "chi2": float(r.chi2), "dof": int(r.dof), "rchi2": float(r.rchi2),
                   "error_scale": float(r.error_scale), "time": float(r.time), "nfev": int(r.nfev),
                   "success": bool(r.success), "summary": r.summary(short=True), "epochs": epochs,
                   "error_method": r.error_method}
        notes = []
        for e in epochs:
            if e.get("limits"):
                notes.append(f"epoch {e['k']}: at a limit: {', '.join(e['limits'][:6])}"
                             + (" …" if len(e["limits"]) > 6 else "") if len(epochs) > 1 else
                             f"at a limit: {', '.join(e['limits'][:8])}")
        out["notes"] = notes
        self._wf_cache[key] = out
        return out

    def _series(self, r):
        """Model, component curves and residuals of one FitResult for the browser plot."""
        from ..plotting import is_bowen, plain_line
        s = r.spectrum
        lo, hi = r.model.wave_range or (s.wave.min(), s.wave.max())
        sel = (s.wave >= lo) & (s.wave <= hi) & np.isfinite(s.flux)
        idx = np.nonzero(sel)[0]
        idx = idx[::_stride(idx.size)]
        parts = r.model_flux(components=True)
        model = r.model_flux()
        series, narrow, cont = [], [], []
        for comp in r.model.components:
            if comp.name not in parts:
                continue
            role, kind = getattr(comp, "role", None), getattr(comp, "kind", "")
            y = parts[comp.name]
            if kind == "continuum" and not comp.name.startswith("balmer"):
                cont.append(y)
            elif kind == "host":
                series.append({"name": comp.name, "role": "host", "y": y})
            elif role == "narrow":
                narrow.append(y)
            elif role in ("broad", "outflow"):
                series.append({"name": comp.name, "role": role, "y": y, "bowen": is_bowen(comp),
                               "lines": [plain_line(n) for n in getattr(comp, "line_names", [])][:6]})
            else:
                t = type(comp).__name__
                series.append({"name": comp.name, "y": y, "role": "feii" if t == "FeII" else
                               "balmer" if t in ("BalmerContinuum", "HighOrderBalmer") else "other"})
        if narrow:
            series.append({"name": "narrow", "role": "narrow", "y": np.sum(narrow, axis=0)})
        if cont:
            series.insert(0, {"name": "continuum", "role": "continuum", "y": np.sum(cont, axis=0)})
        for d in series:
            d["y"] = _arr(d["y"], idx)
        good = s.good
        resid = np.where(good, (s.flux - model) / np.where(good, s.err, 1.0), np.nan)
        labels = []
        t = r.lines
        roles = {c.name: getattr(c, "role", None) for c in r.model.components}
        for row in t.itertuples():
            if row.wavelength < lo or row.wavelength > hi or not np.isfinite(row.flux) or row.flux <= 0:
                continue
            if "-" in row.line:           # an Fe II band (r.lines row of the template), not a line
                continue
            if np.isfinite(row.flux_err) and row.flux_err > 0 and row.flux < 3 * row.flux_err:
                continue
            labels.append({"x": float(row.wavelength), "text": plain_line(row.line).replace(" λ", " "),
                           "flux": float(row.flux), "role": roles.get(row.component)})
        # broad lines first (as in the figures), then the strongest narrow ones; one label per line
        labels.sort(key=lambda d: (d["role"] != "broad", -d["flux"]))
        seen = set()
        labels = [d for d in labels if not (d["text"] in seen or seen.add(d["text"]))]
        broad_x = [d["x"] for d in labels if d["role"] == "broad"]
        labels = [d for d in labels if d["role"] == "broad" or all(abs(d["x"] - b) > 60 for b in broad_x)]
        return {"x": _arr(s.wave, idx), "flux": _arr(s.flux, idx), "model": _arr(model, idx),
                "resid": _arr(resid, idx), "series": series, "labels": labels[:24],
                "masked": _runs(~good[sel] & np.isfinite(s.flux[sel]), s.wave[sel]),
                "range": [float(lo), float(hi)]}

    @staticmethod
    def _line_rows(r):
        from ..plotting import plain_line
        params = r.model.params
        t = r.lines
        roles = {c.name: getattr(c, "role", None) for c in r.model.components}
        rows = []
        order = {"broad": 0, "outflow": 1, "narrow": 2}
        t = t.assign(_o=[order.get(roles.get(c), 3) for c in t.component]).sort_values(["_o", "wavelength"], kind="stable")
        for row in t.itertuples():
            comp, ln = row.component, row.line
            kin = f"{comp}.{ln}." if f"{comp}.{ln}.fwhm" in params.names else f"{comp}."
            fwhm_p, shift_p, flux_p = kin + "fwhm", kin + "shift", f"{comp}.{ln}.flux"
            flags = []
            if flux_p in r.unconstrained:
                flags.append(("outside range", "muted"))
            elif flux_p in r.at_zero or not (row.flux > 0):
                flags.append(("not detected", "muted"))
            elif np.isfinite(row.flux_err) and row.flux_err > 0 and row.flux < 2 * row.flux_err:
                flags.append(("weak (<2σ)", "muted"))
            if fwhm_p in params.names and params[fwhm_p].tie:
                flags.append((f"FWHM tied to {params[fwhm_p].tie[0]}", "info"))
            if fwhm_p in r.at_limit:
                flags.append(("FWHM at bound", "warn"))
            if shift_p in r.at_limit:
                flags.append(("shift at bound", "warn"))
            rows.append({"component": comp, "role": roles.get(comp), "line": ln, "label": plain_line(ln),
                         "wave": float(row.wavelength),
                         "flux": float(row.flux), "flux_err": float(row.flux_err),
                         "fwhm": float(row.fwhm), "fwhm_err": float(row.fwhm_err),
                         "shift": float(row.shift), "shift_err": float(row.shift_err),
                         "ew": float(row.ew), "ew_err": float(row.ew_err),
                         "detected": not any(f[0] in ("not detected", "outside range") for f in flags),
                         "flags": [{"text": a, "kind": b} for a, b in flags]})
        return rows

    @staticmethod
    def _param_rows(r, kinds=None):
        p = r.params
        rows = []
        for q in p.itertuples():
            comp = q.parameter.split(".")[0]
            ckind = next((getattr(c, "kind", "") for c in r.model.components if c.name == comp), "")
            if kinds is not None and ckind not in kinds:
                continue
            rows.append({"name": q.parameter, "value": _fmt_err(q.value, q.error), "min": q.min, "max": q.max,
                         "unit": q.unit or "", "status": q.status, "component": comp})
        return rows

    def wf_result(self, k=0):
        res = self.wf_results()
        if not res:
            return None
        k = min(max(int(k), 0), len(res) - 1)
        name, date, r, ref = res[k]
        key = ("result", id(r), k)
        if key in self._wf_cache:
            return self._wf_cache[key]
        epochs = [{"k": i, "name": n, "date": d, "reference": rf, "failed": fr is None}
                  for i, (n, d, fr, rf) in enumerate(res)]
        out = {"k": k, "name": name, "date": date, "epochs": epochs}
        if r is None:
            out["error"] = "this spectrum could not be fitted"
            return out
        out["plot"] = self._series(r)
        out["lines"] = self._line_rows(r)
        out["params"] = self._param_rows(r)
        out["continuum"] = self._param_rows(r, kinds=("continuum", "host", "template", "calibration",
                                                    "attenuation", "absorption"))
        out["fit"] = {"chi2": float(r.chi2), "dof": int(r.dof), "rchi2": float(r.rchi2),
                      "scale": float(getattr(r, "scale", 1.0) or 1.0)}
        out["corner_params"] = self.sampled_params(r)
        if len(self._wf_cache) > 80:
            self._wf_cache.clear()
        self._wf_cache[key] = out
        return out

    def wf_main_sequence(self):
        """Every epoch's position on the quasar main sequence (R_FeII, FWHM of broad H-beta) and the grid."""
        from ..plotting import MS_TYPES, main_sequence_grid
        res = self.wf_results()
        if not res:
            return None
        key = ("ms", tuple(id(r) for _, _, r, _ in res))
        if key not in self._wf_cache:
            pts = []
            for k, (name, date, r, ref) in enumerate(res):
                if r is None:
                    continue
                try:
                    m = r.main_sequence()
                except Exception:                    # noqa: BLE001
                    continue
                mjd = getattr(r.spectrum, "mjd", None)
                pts.append({"k": k, "name": name, "date": date, "reference": bool(ref),
                            "mjd": float(mjd) if mjd is not None and np.isfinite(mjd) else None,
                            **{q: (float(v) if np.isfinite(v) else None) for q, v in m.items()
                               if q in ("r_feii", "r_feii_err", "fwhm", "fwhm_err")},
                            "population": m["population"], "type": m["spectral_type"]})
            self._wf_cache[key] = {"points": pts,
                                   "grid": [{"x": list(x), "y": list(y)} for x, y in main_sequence_grid()],
                                   "types": [{"t": t, "x": x, "y": y} for t, x, y in MS_TYPES]}
        return self._wf_cache[key]

    def wf_main_sequence_figure(self, k=0, fmt="png"):
        """Publication figure of the main sequence: every epoch alike (shaded by date), none highlighted."""
        from matplotlib.figure import Figure
        from ..plotting import plot_main_sequence
        res = [x for x in (self.wf_results() or []) if x[2] is not None]
        if not res:
            return None
        many = len(res) > 1
        with self._render_lock:
            fig = plot_main_sequence([r for _, _, r, _ in res], fig=Figure(),
                                     labels=[f"#{i}" for i in range(len(res))] if many else None)   # up to 10
            return self._fig_bytes(fig, fmt, 150 if fmt == "png" else 300)

    def wf_result_more(self, k=0, part="key"):
        """The slow parts of the result view, loaded after the plot: 'key' (main numbers with
        propagated errors) of fit ``k``, or 'all' (the table of all epochs)."""
        res = self.wf_results()
        if not res:
            return None
        if part == "all":
            key = ("all", id(res[0][2]))
            if key not in self._wf_cache:
                self._wf_cache[key] = {"all": self._all_epochs()}
            return self._wf_cache[key]
        k = min(max(int(k), 0), len(res) - 1)
        r = res[k][2]
        if r is None:
            return {"key": []}
        key = ("key", id(r))
        if key not in self._wf_cache:
            rows = []
            try:
                for q in r.key_results().itertuples():
                    rows.append({"name": q.quantity, "value": _fmt_err(q.value, q.error), "unit": q.unit or ""})
            except Exception:
                pass
            bh = []
            try:
                for q in r.bh_masses().itertuples():
                    bh.append({"line": q.line, "recipe": q.recipe, "reference": q.reference, "width": q.width,
                               "luminosity": q.luminosity, "value": _fmt_err(q.log_mbh, q.log_mbh_err),
                               "scatter": q.scatter_dex})
            except Exception:
                pass
            self._wf_cache[key] = {"key": rows, "bh": bh}
        return self._wf_cache[key]

    # ------------------------------------------------------------- residuals
    def wf_residuals_code(self, scope="all", nsigma=4.0):
        """Code that analyses the residuals of the last fit (one epoch ``scope`` = k, or all)."""
        kind = self.wf.get("fit_kind")
        obj = {"joint": "me", "fits": "fits", "single": "r"}.get(kind)
        if obj is None or self.wf_results() is None:
            raise ValueError("fit first: the residuals of a fit are analysed")
        target = obj
        if scope not in ("all", None, "") and kind in ("joint", "fits"):
            target = f"{obj}.epochs[{int(scope)}]" if kind == "joint" else f"{obj}[{int(scope)}]"
            self.wf["resid_scope"] = int(scope)
        else:
            self.wf["resid_scope"] = "all"
        opt = f", nsigma={float(nsigma):g}" if float(nsigma) != 4.0 else ""
        return ("# residuals\nresid = fa.analyze_residuals(" + target + opt + ")   # features the model misses, with suggestions\n"
                'resid[["center", "kind", "sigma", "fwhm", "line", "suggestion"]].round(1)')

    def wf_residuals(self):
        import pandas as pd
        t = self.ns.get("resid")
        if not isinstance(t, pd.DataFrame) or "suggestion" not in t:
            return None
        key = ("resid", id(t))
        if key in self._wf_cache:
            return self._wf_cache[key]
        num = lambda v: float(v) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else None
        rows = []
        for i, d in enumerate(t.to_dict("records")):
            rows.append({"i": i, "center": num(d["center"]), "lo": num(d["lo"]), "hi": num(d["hi"]),
                         "kind": d["kind"], "sigma": num(d["sigma"]), "fwhm": num(d["fwhm"]),
                         "line": d.get("line") or "", "in_model": bool(d.get("in_model")),
                         "suggestion": d["suggestion"], "code": d.get("code") or "",
                         "action": d.get("action") if isinstance(d.get("action"), dict) else None,
                         "epochs": [int(k) for k in (d.get("epochs") or [])]})
        res = self.wf_results() or []
        out = {"rows": rows, "scope": self.wf.get("resid_scope", "all"), "n_epochs": len(res), "id": id(t)}
        self._wf_cache[key] = out
        return out

    # -------------------------------------------- light curves, reverberation
    def wf_lightcurve(self):
        """Light curves of the joint fit (broad-line flux, FWHM, EW and continuum per epoch)."""
        me = self.multi
        if me is None or self.wf.get("fit_kind") not in ("joint", None):
            return None
        if not me.has_light_curve():
            self.multi_state()                       # starts the background computation
            return {"computing": True}
        key = ("lc", id(me))
        if key in self._wf_cache:
            return self._wf_cache[key]
        from ..plotting import plain_line
        t = me.light_curve()
        mjd = t["mjd"].to_numpy(float)
        t0 = np.nanmin(mjd) if np.isfinite(mjd).any() else 0.0
        val = lambda c: [float(v) if np.isfinite(v) else None for v in t[c].to_numpy(float)]
        series = []
        for c in t.columns:
            for kind, unit in (("_flux", ""), ("_fwhm", "km/s"), ("_ew", "\u00c5")):
                if c.endswith(kind) and c + "_err" in t:
                    line = c[: -len(kind)]
                    series.append({"name": c, "line": line, "label": plain_line(line), "kind": kind[1:],
                                   "unit": unit, "y": val(c), "err": val(c + "_err")})
            if c.startswith("cont") and not c.endswith("_err") and c + "_err" in t:
                series.append({"name": c, "line": c, "label": f"continuum {c[4:]} \u00c5", "kind": "flux",
                               "unit": "", "y": val(c), "err": val(c + "_err")})
        lines = sorted({d["line"] for d in series if not d["line"].startswith("cont")},
                       key=lambda ln: [d["line"] for d in series].index(ln))
        out = {"mjd": [float(v) if np.isfinite(v) else None for v in mjd], "t0": float(t0),
               "dates": [mjd_to_date(v) for v in mjd], "series": series, "lines": lines,
               "names": [e.spectrum.name for e in me.epochs]}
        self._wf_cache[key] = out
        return out

    def wf_reverb_code(self, line="Hb", n_mc=1000, methods=("iccf", "vn", "drw"), detrend=False,
                       mjd_min=None, mjd_max=None):
        if self.multi is None:
            raise ValueError("fit the epochs jointly first")
        if not str(line).replace("[", "").replace("]", "").isalnum():
            raise ValueError("unknown line")
        methods = tuple(m for m in methods if m in ("iccf", "vn", "zdcf", "drw"))
        if not methods:
            raise ValueError("choose at least one lag method")
        opts = [repr(str(line)), f"n_mc={int(n_mc)}"]
        if methods != ("iccf", "vn", "drw"):
            opts.append(f"methods={methods!r}")
            if "iccf" not in methods:
                opts.append(f"primary={methods[0]!r}")
        if detrend:
            opts.append("detrend=True")
        if mjd_min not in (None, "") and mjd_max not in (None, ""):
            opts.append(f"mjd_range=({float(mjd_min):g}, {float(mjd_max):g})")
        return ("# reverberation\nrm = me.reverberation(" + ", ".join(opts) + ")   # lag, rms spectrum, mass\n"
                "print(rm.lag_table().round(2).to_string(index=False))\nrm")

    def wf_reverb(self):
        rv = self.reverb
        if rv is None:
            return None
        key = ("rv", id(rv))
        if key in self._wf_cache:
            return self._wf_cache[key]
        fmt = lambda v: f"{v:.4g}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else "\u2014"
        lags = []
        try:
            for q in rv.lag_table().itertuples():
                lags.append({"method": str(q.method), "lag": f"{q.lag_rest:.1f} (\u2212{q.err_lo:.1f} +{q.err_hi:.1f})",
                             "obs": fmt(q.lag_obs), "r_max": fmt(q.r_max),
                             "outside": f"{100 * q.frac_outside_peak:.0f}%" if np.isfinite(q.frac_outside_peak) else "\u2014",
                             "adopted": bool(q.adopted)})
        except Exception:
            pass
        out = {"line": rv.line, "rows": [[q.quantity, fmt(q.value), str(q.error), q.unit] for q in rv.table().itertuples()],
               "lags": lags, "warnings": list(rv.warnings), "id": id(rv)}
        self._wf_cache[key] = out
        return out

    def wf_reverb_figure(self):
        from matplotlib.figure import Figure
        rv = self.reverb
        if rv is None:
            return None
        key = ("rvfig", id(rv))
        if key not in self._wf_cache:
            with self._render_lock:
                fig = rv.plot(fig=Figure())
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            self._wf_cache[key] = buf.getvalue()
        return self._wf_cache[key]

    def _all_epochs(self):
        res = self.wf_results()
        if not res or len(res) < 2:
            return None
        if self.multi is not None and self.wf.get("fit_kind") in ("joint", None):
            t = self.multi.light_curve()
            cols = [c for c in t.columns if not c.endswith("_err") and not c.endswith("_corr")]
            rows = []
            for d in t.to_dict("records"):
                rows.append([(_fmt_err(d[c], d.get(c + "_err")) if isinstance(d[c], (float, np.floating))
                              else str(d[c])) for c in cols])
            return {"columns": cols, "rows": rows}
        rows, cols = [], ["epoch", "date", "name", "rchi2"]
        for k, (name, date, r, _) in enumerate(res):
            vals = {"epoch": str(k), "date": date or "", "name": name,
                    "rchi2": "failed" if r is None else f"{r.rchi2:.3f}"}
            if r is not None:
                try:
                    for q in r.key_results().itertuples():
                        if q.quantity not in cols:
                            cols.append(q.quantity)
                        vals[q.quantity] = _fmt_err(q.value, q.error)
                except Exception:
                    pass
            rows.append(vals)
        return {"columns": cols, "rows": [[v.get(c, "") for c in cols] for v in rows]}

    @staticmethod
    def diagram_curves():
        """Demarcation lines of the BPT ([NII], [SII], [OI]) and WHAN diagrams as polylines."""
        from ..classify import _XMAX, PASSIVE_EW, RETIRED_EW, kauffmann03, kewley01, kewley06

        def line(x, y, dash=None):
            return {"x": [round(float(v), 4) for v in x], "y": [round(float(v), 4) for v in y], "dash": dash}
        out = {}
        # the curves extend beyond the default range: the plot widens to show every point and clips them
        for d, xlim in (("nii", (-1.6, 0.6)), ("sii", (-1.3, 0.5)), ("oi", (-2.3, 0.0))):
            x = np.linspace(-3.5, _XMAX[d] - 0.01, 200)
            curves = [line(x, kewley01(x, d))]
            if d == "nii":
                xk = np.linspace(-3.5, 0.04, 200)
                curves.append(line(xk, kauffmann03(xk), "5 4"))
                texts = [(-1.3, -0.7, "SF"), (-0.3, -0.95, "comp."), (-0.9, 1.2, "Seyfert"), (0.15, -0.7, "LINER")]
            else:
                xs = np.linspace(-3.5, 2.0, 400)
                above = (xs >= _XMAX[d]) | (kewley06(xs, d) > kewley01(np.minimum(xs, _XMAX[d] - 1e-3), d))
                curves.append(line(xs[above], kewley06(xs[above], d), "2 3"))
                texts = ([(-1.1, -0.8, "SF"), (-0.4, 1.2, "Seyfert"), (0.1, -0.2, "LINER")] if d == "sii" else
                         [(-2.1, -0.8, "SF"), (-1.2, 1.2, "Seyfert"), (-0.4, -0.2, "LINER")])
            out[d] = {"xlim": xlim, "ylim": (-1.2, 1.5), "curves": curves,
                      "texts": [{"x": a, "y": b, "t": t} for a, b, t in texts]}
        r, p_ = np.log10(RETIRED_EW), np.log10(PASSIVE_EW)
        out["whan"] = {"xlim": (-1.6, 0.8), "ylim": (-1.0, 3.0),
                       "curves": [line([-3.5, 2.0], [r, r]), line([-3.5, 2.0], [p_, p_], "2 3"),
                                  line([-0.4, -0.4], [r, 5.0]), line([-0.4, 2.0], [np.log10(6)] * 2, "5 4")],
                       "texts": [{"x": a, "y": b, "t": t} for a, b, t in
                                 ((-1.5, 2.6, "SF"), (0.1, 2.6, "sAGN"), (0.1, 0.62, "wAGN"),
                                  (-1.5, 0.05, "retired"), (-1.5, -0.8, "passive"))]}
        return out

    def _fit_diagram_points(self):
        """
        (key, {spectrum name: BPT points}) from the narrow lines of the fits (r.diagnostics()): the
        classification of a broad-line spectrum has no narrow [N II]/H-alpha, the decomposition does.
        """
        res = self.wf_results() or []
        key = tuple(id(r) for _, _, r, _ in res)
        cached = self._wf_cache.get(("fitbpt", key))
        if cached is not None:
            return key, cached
        out = {}
        for _, _, r, _ in res:
            if r is None:
                continue
            try:
                d = r.diagnostics().set_index("quantity")
            except Exception:                        # noqa: BLE001
                continue

            def get(q):
                if q not in d.index or not np.isfinite(d.loc[q, "value"]):
                    return None, None
                e = d.loc[q, "error"]
                return float(d.loc[q, "value"]), (float(e) if np.isfinite(e) else None)
            y, ye = get("log [OIII]5007/Hb")
            pts = {}
            for dk, q in (("nii", "log [NII]6583/Ha"), ("sii", "log [SII]6716,6731/Ha"), ("oi", "log [OI]6300/Ha")):
                x, xe = get(q)
                pts[dk] = None if x is None or y is None else {"x": x, "y": y, "xerr": xe, "yerr": ye, "fit": True}
            out[r.spectrum.name] = pts
        self._wf_cache[("fitbpt", key)] = out
        return key, out

    @staticmethod
    def _merge_points(own, fit):
        """The classification's diagram points, completed with those of the fit where it has none."""
        if not fit:
            return own
        return {d: own.get(d) or fit.get(d) for d in set(own) | set(fit)}

    @staticmethod
    def _diagram_point(m):
        """The object's position in each diagram (None where a ratio is missing)."""
        def val(key):
            v = m.get(key, np.nan)
            return float(v) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else None
        y, ye = val("log_oiii_hb"), val("log_oiii_hb_err")
        out = {}
        for d in ("nii", "sii", "oi"):
            x = val(f"log_{d}_ha")
            out[d] = None if x is None or y is None else {"x": x, "y": y, "xerr": val(f"log_{d}_ha_err"), "yerr": ye}
        x, ew = val("log_nii_ha"), val("ew_ha")
        out["whan"] = None if x is None or not ew or ew <= 0 else {"x": x, "y": float(np.log10(ew)),
                                                                  "xerr": val("log_nii_ha_err"), "yerr": None}
        return out

    def classify_state(self, k=0):
        cl = self._classifications()
        if cl is None:
            return None
        k = min(max(int(k), 0), len(cl) - 1)
        c = cl[k]
        fits = self._fit_diagram_points()
        key = ("cstate", id(c), len(cl), fits[0])
        if key in self._wf_cache:
            return self._wf_cache[key]
        rows = []
        for name, v in c.measurements.items():
            if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) and np.isfinite(v):
                rows.append({"name": name, "value": f"{v:.4g}"})
        s = c.spectrum
        out = {"k": k, "label": c.label, "uncertain": bool(c.uncertain), "reason": str(c.reason or ""),
               "name": s.name,
               "labels": [{"k": i, "name": x.spectrum.name, "label": x.label, "uncertain": bool(x.uncertain),
                           "date": mjd_to_date(x.spectrum.mjd),
                           "points": self._merge_points(self._diagram_point(x.measurements), fits[1].get(x.spectrum.name))}
                          for i, x in enumerate(cl)],
               "rows": rows, "diagrams": self.diagram_curves(),
               "points": self._merge_points(self._diagram_point(c.measurements), fits[1].get(s.name))}
        # 1: the spectrum with the stellar (+ power-law) fit and the masked lines
        try:
            st = c.stellar_fit
            lo, hi = st.spectrum.wave[st.mask][[0, -1]]
            sel = (s.wave >= lo) & (s.wave <= hi) & np.isfinite(s.flux)
            idx = np.nonzero(sel)[0][::_stride(int(sel.sum()))]
            m = c.measurements
            out["starlight"] = {"x": _arr(s.wave, idx), "flux": _arr(s.flux, idx),
                                "model": _arr(st.model_flux(s.wave), idx),
                                "masks": [[float(a), float(b)] for a, b in (c.masks or [])],
                                "title": f"host {m.get('host_fraction_5100', np.nan):.0%} at 5100 Å, "
                                         f"σ★ = {m.get('sigma_star', np.nan):.0f} km/s"}
        except Exception:
            out["starlight"] = None
        # 2: H-alpha in the starlight-subtracted spectrum, narrow only and narrow + broad
        try:
            w = (s.wave > 6400) & (s.wave < 6740) & np.isfinite(s.flux)
            if w.sum() > 10 and c.narrow_fit is not None:
                cont = c.stellar_fit.model_flux(s.wave)
                idx = np.nonzero(w)[0]
                m = c.measurements
                dec = "yes" if c.label.startswith("type1") else ("weak?" if m.get("weak_broad") else "no")
                out["halpha"] = {"x": _arr(s.wave, idx), "flux": _arr(s.flux - cont, idx),
                                 "narrow": _arr(c.narrow_fit.model_flux(s.wave), idx),
                                 "broad": _arr(c.broad_fit.model_flux(s.wave), idx) if c.broad_fit is not None else None,
                                 "title": f"broad Hα? {dec} (ΔBIC = {m.get('broad_dbic', np.nan):.0f}, "
                                          f"S/N = {m.get('broad_snr', np.nan):.0f})"}
            else:
                out["halpha"] = None
        except Exception:
            out["halpha"] = None
        # 3: the emission-line fit used for the line ratios
        from ..result import FitResult
        lf = c.line_fit
        out["lines_fit"] = None
        if isinstance(lf, FitResult):
            try:
                out["lines_fit"] = {"plot": self._series(lf), "lines": self._line_rows(lf)}
            except Exception:
                pass
        if len(self._wf_cache) > 80:
            self._wf_cache.clear()
        self._wf_cache[key] = out
        return out

    def classify_figure(self, k=0, kind="steps", fmt="png"):
        """Publication figure of classification ``k``: 'steps' (plot_steps) or 'diagrams' (plot)."""
        from matplotlib.figure import Figure
        cl = self._classifications()
        if cl is None:
            return None
        k = min(max(int(k), 0), len(cl) - 1)
        key = ("cfig", id(cl[k]), kind, fmt)
        if key in self._wf_cache:
            return self._wf_cache[key]
        with self._render_lock:
            fig = (cl[k].plot(fig=Figure(), others=cl if len(cl) > 1 else None, highlight=False) if kind == "diagrams"
                   else cl[k].plot_steps(fig=Figure()))
            data = self._fig_bytes(fig, fmt, 150)
        self._wf_cache[key] = data
        return data

    def wf_result_figure(self, k=0):
        """Publication figure (PNG) of fit ``k`` of the workflow (cached)."""
        from matplotlib.figure import Figure
        res = self.wf_results()
        if not res:
            return None
        k = min(max(int(k), 0), len(res) - 1)
        name, date, r, _ = res[k]
        if r is None:
            raise ValueError("this spectrum could not be fitted")
        key = ("rfig", id(r))
        if key not in self._wf_cache:
            with self._render_lock:
                title = f"epoch {k}: {name}" + (f"  ({date})" if date else "") if len(res) > 1 else name
                fig = r.plot(fig=Figure(), title=title)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            self._wf_cache[key] = buf.getvalue()
        return self._wf_cache[key]

    # ------------------------------------------------------------- samples
    def sample_view(self, k=0):
        """The fit of row ``k`` of a fitted sample (loaded from its .fantasy file) for the interactive view."""
        sm = self.sample
        if sm is None:
            return None
        k = min(max(int(k), 0), len(sm.table) - 1)
        row = sm.table.iloc[k]
        key = ("sample", id(sm), k)
        if key in self._wf_cache:
            return self._wf_cache[key]
        out = {"k": k, "name": str(row.get("name", f"row {k}")), "cls": str(row.get("cls", "") or ""),
               "preset": str(row.get("preset", "") or "")}
        err = row.get("error") if "error" in sm.table else None
        if isinstance(err, str) and err:
            out["error"] = err
        else:
            try:
                r = sm.load(k)
                out["plot"] = self._series(r)
                out["lines"] = self._line_rows(r)
                out["params"] = self._param_rows(r)
                out["continuum"] = self._param_rows(r, kinds=("continuum", "host", "template", "calibration",
                                                            "attenuation", "absorption"))
                try:
                    out["key"] = [{"name": q.quantity, "value": _fmt_err(q.value, q.error), "unit": q.unit or ""}
                                  for q in r.key_results().itertuples()]
                except Exception:
                    out["key"] = []
                out["fit"] = {"chi2": float(r.chi2), "dof": int(r.dof), "rchi2": float(r.rchi2)}
            except Exception as exc:
                out["error"] = f"{type(exc).__name__}: {exc}"
        if len(self._wf_cache) > 80:
            self._wf_cache.clear()
        self._wf_cache[key] = out
        return out

    def sample_points(self):
        """[NII] BPT and FWHM-luminosity data of a fitted sample (one point per spectrum)."""
        sm = self.sample
        if sm is None:
            return None
        t = sm.table
        col = lambda c: [float(v) if np.isfinite(v) else None for v in t[c].astype(float)] if c in t else None
        return {"names": [str(v) for v in t["name"]] if "name" in t else None,
                "cls": [str(v) for v in t["cls"]] if "cls" in t else None,
                "nii": col("log [NII]/Ha"), "oiii": col("log [OIII]/Hb"),
                "fwhm": col("broad Hb FWHM") or col("broad Ha FWHM"), "rchi2": col("rchi2"),
                "nii_curves": self.diagram_curves()["nii"]}

    # ---------------------------------------------------------- downloads
    def wf_lines_table(self):
        import pandas as pd
        res = self.wf_results()
        if not res:
            return None
        frames = []
        for k, (name, date, r, _) in enumerate(res):
            if r is None:
                continue
            t = r.lines.copy()
            t.insert(0, "date", date or "")
            t.insert(0, "mjd", r.spectrum.mjd if r.spectrum.mjd is not None else np.nan)
            t.insert(0, "spectrum", name)
            t.insert(0, "epoch", k)
            frames.append(t)
        return pd.concat(frames, ignore_index=True) if frames else None

    def _wf_fit(self, k=0):
        res = self.wf_results()
        if not res:
            raise ValueError("fit first")
        k = min(max(int(k), 0), len(res) - 1)
        name, date, r, _ = res[k]
        if r is None:
            raise ValueError("this spectrum could not be fitted")
        return name, date, r, len(res) > 1

    # ---------------------------------------------------- saved models
    MODELS = os.path.join(os.path.expanduser("~"), ".fantasy_agn", "models")

    def _model_path(self, name):
        clean = re.sub(r"[^\w\- ]", "", str(name or "")).strip().replace(" ", "_")[:60]
        if not clean:
            raise ValueError("give the model a name (letters, digits, - and _)")
        return clean, os.path.join(self.MODELS, clean + ".json")

    PLACES = os.path.join(MODELS, "places.json")     # models saved in other folders, listed with the library

    def _places(self):
        import json
        try:
            with open(self.PLACES) as fh:
                return [p for p in json.load(fh) if isinstance(p, str)]
        except (OSError, ValueError):
            return []

    def _remember(self, path):
        """Remember a model saved outside the library (newest first, at most 200)."""
        import json
        places = [path] + [p for p in self._places() if p != path]
        os.makedirs(self.MODELS, exist_ok=True)
        with open(self.PLACES, "w") as fh:
            json.dump(places[:200], fh, indent=1)

    def saved_models(self):
        """
        The saved models, newest first: those in ~/.fantasy_agn/models and those saved in other
        folders (remembered; files that are gone are left out). Each: name, path, folder, components,
        free parameters, whether it is in the library.
        """
        import json
        paths = []
        if os.path.isdir(self.MODELS):
            paths += [os.path.join(self.MODELS, fn) for fn in os.listdir(self.MODELS)
                      if fn.endswith(".json") and fn != "places.json"]
        paths += [p for p in self._places() if p not in paths]
        out = []
        for path in paths:
            try:
                with open(path) as fh:
                    d = json.load(fh)
                comps = [c.get("type") for c in d["components"]]
                n_free = sum(1 for p in d.get("params", []) if not p.get("fixed") and not p.get("tie"))
            except (OSError, ValueError, AttributeError, KeyError, TypeError):
                continue
            folder = os.path.dirname(path)
            out.append({"name": os.path.basename(path)[:-5], "path": path, "folder": folder,
                        "library": folder == self.MODELS, "components": comps, "n_components": len(comps),
                        "n_free": n_free, "modified": os.path.getmtime(path)})
        return sorted(out, key=lambda m: -m["modified"])

    def model_folders(self):
        """Folders to suggest when saving: the library, then the folders used before."""
        seen = [self.MODELS]
        for p in self._places():
            f = os.path.dirname(p)
            if f not in seen:
                seen.append(f)
        home = os.path.expanduser("~")
        return [{"path": f, "label": "~" + f[len(home):] if f.startswith(home) else f, "library": f == self.MODELS}
                for f in seen]

    def _target(self, name, folder=None):
        """The file a model is saved to: ``name``.json in ``folder`` (default: the library)."""
        clean, path = self._model_path(name)
        if folder and str(folder).strip():
            folder = os.path.abspath(os.path.expanduser(str(folder).strip()))
            if os.path.exists(folder) and not os.path.isdir(folder):
                raise ValueError(f"{folder} is a file, not a folder")
            path = os.path.join(folder, clean + ".json")
        return path

    def save_model_code(self, name, folder=None, overwrite=False):
        """Code that saves ``model`` as ``name``.json in ``folder`` (default: the library of models)."""
        if self.model is None:
            raise ValueError("create a model first")
        self.model.to_dict()                       # raises here, before the cell, if it cannot be saved
        path = self._target(name, folder)
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.dirname(path) != self.MODELS:
            self._remember(path)
        return f"model.save({path!r})"

    def load_model_code(self, name=None, path=None):
        """Code that makes ``model`` from a saved model: by name in the library, or by its file."""
        if path:
            path = os.path.abspath(os.path.expanduser(str(path)))
            if not path.endswith(".json"):
                raise ValueError("a model file ends in .json")
        else:
            _, path = self._model_path(name)
        if not os.path.exists(path):
            raise ValueError(f"no saved model at {path}")
        return f"model = fa.load_model({path!r})\nmodel"

    def store_model(self, name, text):
        """Check a model file (JSON from Model.save) and put it in the library; returns its name."""
        import json
        from ..model import Model
        try:
            d = json.loads(text)
        except ValueError:
            raise ValueError("not a model file (JSON from Model.save or the Save model button)")
        Model.from_dict(d)                         # validates: known components, parameters of the model
        clean, path = self._model_path(name)
        os.makedirs(self.MODELS, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(d, fh, indent=1)
        return clean

    def model_json(self):
        """The current model as JSON (Model.to_dict)."""
        import json
        if self.model is None:
            raise ValueError("create a model first")
        return json.dumps(self.model.to_dict(), indent=1).encode()

    def wf_line_list(self, k=0):
        """The lines of fit ``k`` that can be drawn alone (see plotting.fitted_lines), by wavelength."""
        from ..plotting import fitted_lines, main_line, plain_line
        r = self._wf_fit(k)[2]
        lines = [{"line": d["line"], "label": plain_line(d["line"]), "wave": d["wavelength"],
                  "n": len(d["components"]), "roles": sorted(set(d["roles"]))} for d in fitted_lines(r)]
        return {"lines": sorted(lines, key=lambda d: d["wave"]), "main": main_line(r)}

    def wf_lines_figure(self, k=0, lines=(), opts=None, fmt="png"):
        """
        A grid of the chosen lines of fit ``k`` (plotting.plot_lines), one panel each; options:
        velocity (0: wavelength axis), residuals, info (FWHM and flux), others (the other lines in
        each window), ncols (default: from the number of lines), width ('single', 'double').
        """
        from matplotlib.figure import Figure
        from ..plotting import fitted_lines, plot_lines
        r = self._wf_fit(k)[2]
        opts = dict(opts or {})
        known = {d["line"] for d in fitted_lines(r)}
        lines = [ln for ln in lines if ln in known]
        if not lines:
            raise ValueError("choose at least one fitted line")
        on = lambda key, default="1": str(opts.get(key, default)) != "0"
        nc = str(opts.get("ncols", ""))
        ncols = min(int(nc), len(lines)) if nc.isdigit() and int(nc) > 0 else None
        width = opts.get("width") if opts.get("width") in ("single", "double") else "double"
        with self._render_lock:
            fig = plot_lines(r, lines, fig=Figure(), ncols=ncols, velocity=on("velocity"), width=width,
                             residuals=on("residuals"), info=on("info"), others=on("others", "0"))
            return self._fig_bytes(fig, fmt, 150 if fmt == "png" else 300)

    def wf_components_csv(self, k=0):
        """The fitted model of fit ``k`` and its components as CSV (see Session.components_csv)."""
        return self.components_csv(r=self._wf_fit(k)[2])

    def wf_corner(self, k=0, params=None, fmt="png"):
        r = self._wf_fit(k)[2]
        with self._render_lock:
            return self.corner_figure(params=params, fmt=fmt, dpi=110 if fmt == "png" else 300, r=r)

    def wf_custom_figure(self, k=0, opts=None, fmt="png"):
        """The publication figure of fit ``k`` with the appearance options of the Results step."""
        name, date, r, many = self._wf_fit(k)
        opts = dict(opts or {})
        if "title" not in opts:
            opts["title"] = f"epoch {k}: {name}" + (f"  ({date})" if date else "") if many else name
        with self._render_lock:
            return self.custom_figure(opts, fmt=fmt, dpi=150 if fmt == "png" else 300, r=r)

    def wf_lines_csv(self):
        t = self.wf_lines_table()
        return None if t is None else t.to_csv(index=False)

    def wf_lines_fits(self):
        from astropy.table import Table
        t = self.wf_lines_table()
        if t is None:
            return None
        tab = Table.from_pandas(t)
        tab.meta["ORIGIN"] = "FANTASY " + __import__("fantasy_agn").__version__
        buf = io.BytesIO()
        tab.write(buf, format="fits")
        return buf.getvalue()

    def wf_plots_pdf(self):
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.figure import Figure
        from ..plotting import paper_style
        res = self.wf_results()
        if not res:
            return None
        buf = io.BytesIO()
        with self._render_lock, paper_style(), PdfPages(buf) as pdf:
            for k, (name, date, r, _) in enumerate(res):
                if r is None:
                    continue
                title = f"epoch {k}: {name}" + (f"  ({date})" if date else "") if len(res) > 1 else name
                pdf.savefig(r.plot(fig=Figure(), title=title), bbox_inches="tight")
        return buf.getvalue()

    def notebook(self):
        """The session as a Jupyter notebook (.ipynb JSON)."""
        import json
        cells = [{"cell_type": "markdown", "metadata": {},
                  "source": ["# FANTASY session\n", "Generated by the FANTASY app."]}]
        for c in self.cells:
            if not c.n:
                continue
            src = c.code.split("\n")
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": [ln + "\n" for ln in src[:-1]] + [src[-1]]})
        nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
              "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}}
        return json.dumps(nb, indent=1)

    def wf_state(self):
        return {"mode": self.wf.get("mode"), "group": self.wf.get("group"),
                "fit_kind": self.wf.get("fit_kind"), "settings": {k: v for k, v in self.wf.items()},
                "loaded": self.loaded_state(), "prepared": self.prepared_state(),
                "reference": self.wf_reference() if (self.spectra or self.raw) else None,
                "fit": self.fit_state(), "reverb": self.wf_reverb(), "residuals": self.wf_residuals(),
                "classified": self._classifications() is not None}


def _is_number(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False
