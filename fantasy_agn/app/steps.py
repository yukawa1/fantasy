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
import hashlib
import io
import os
import re

import numpy as np
import pandas as pd

from ..spectrum import date_from_name, mjd_to_date

# the papers to cite, in the headers of the downloaded script and notebook (the app's How to cite)
CITE_REFS = ("Ilić, Rakić & Popović 2023, ApJS 267, 19 (the FANTASY paper); Rakić 2022, MNRAS 516, 1624; "
             "Ilić et al. 2020, A&A 638, A13")
CITE_COMMENT = f"# If you publish these results, please cite FANTASY: {CITE_REFS}.\n"
CITE_MARKDOWN = f"If you publish these results, please cite FANTASY: {CITE_REFS}."

MODES = ("single", "sample", "multi", "classify")
MAX_POINTS = 4000
TELLURIC_NAMES = ("B", "H₂O", "A")         # Spectrum.TELLURIC, in order


def _target_key(obj):
    """Catalog identity from the input spectrum, including the position queried."""
    return (str(obj.name), *(None if pd.isna(v) else float(v)
                             for v in (getattr(obj, "ra", None), getattr(obj, "dec", None))))


def _epoch_key(spec):
    return _target_key(spec), (None if pd.isna(spec.mjd) else spec.mjd)


def _num(v):
    return None if v in (None, "") else float(v)


def _g(v):
    """Short repr of a number for generated code."""
    return repr(float(v)).removesuffix(".0")


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
        s += f" ± {float(f'{e:.2g}'):g}"         # two significant digits, written out (280, not 2.8e+02)
    return s


SPECTRUM_EXTENSIONS = (".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".txt", ".dat", ".csv", ".tsv", ".ascii",
                       ".asc", ".spec", ".flm")


def resolve_pattern(text):
    """
    The files a path typed in the Read step stands for, and the glob pattern for the generated code
    (None when it was not a pattern). Forgiving about how paths get pasted: surrounding quotes,
    shell escapes ("My\\ Data"), file:// URLs, ~; a folder means the spectra in it; an existing
    file is taken as it is, also with [ ] in its name.
    """
    from urllib.parse import unquote
    raw = str(text).strip()
    if len(raw) > 1 and raw[0] == raw[-1] and raw[0] in "'\"":
        raw = raw[1:-1].strip()
    if raw.startswith("file://"):
        raw = unquote(raw[len("file://"):])
    path = os.path.expanduser(raw)
    if not os.path.exists(path) and "\\" in path:
        unescaped = re.sub(r"\\(.)", r"\1", path)
        if os.path.exists(unescaped) or glob.glob(unescaped):
            path = unescaped
    if os.path.isfile(path):
        return [path], None
    if os.path.isdir(path):
        files = sorted(os.path.join(path, f) for f in os.listdir(path)
                       if not f.startswith(".") and f.lower().endswith(SPECTRUM_EXTENSIONS)
                       and os.path.isfile(os.path.join(path, f)))
        if not files:
            raise ValueError(f"no spectra (FITS or text files) in the folder {path}")
        return files, None
    files = sorted(f for f in glob.glob(path) if os.path.isfile(f))
    if files:
        return files, path
    where = os.path.dirname(path) or "."
    if not os.path.isabs(path):
        hint = f" (a relative path starts from {os.getcwd()}; give the full path)"
    elif not os.path.isdir(where) and not glob.has_magic(where):
        hint = f" (the folder {where} does not exist)"
    else:
        hint = ""
    raise ValueError(f"no files match {raw!r}{hint}")


_FEII_LEVELS = {}


def _feii_levels(kind, n):
    """
    [term_k, J_i, J_k, E_i, E_k, nist] of every line of a Fe II template, in the template's order, for the
    Grotrian diagram of Results › Fe II; None for a template without atomic data (the local UV model).
    """
    if kind not in _FEII_LEVELS:
        from ..lines import DATA
        src = {"optical": "feii_levels.csv", "uv_extended": "uvfe2.csv"}.get(kind)
        rows = None
        if src:
            t = pd.read_csv(os.path.join(DATA, src), comment="#", keep_default_na=False)
            nist = t["nist"] if "nist" in t else pd.Series(1, index=t.index)
            rows = [[str(tk).replace(" ", ""), str(ji), str(jk), round(float(ei), 4), round(float(ek), 4), int(m)]
                    for tk, ji, jk, ei, ek, m in zip(t.term_k, t.J_i, t.J_k, t.Ei_eV, t.Ek_eV, nist)]
        _FEII_LEVELS[kind] = rows
    rows = _FEII_LEVELS[kind]
    return rows if rows is not None and len(rows) == n else None


class StepsMixin:
    """Workflow methods of :class:`fantasy_agn.app.session.Session`."""

    def _wf_reset(self):
        self.wf = {"mode": "single", "group": "auto", "fit_kind": None}
        self._read_revision = getattr(self, "_read_revision", 0) + 1
        self._wf_cache = {}
        self._final_kept = self._catalog_kept = self._sample_cat = None

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
        sp = self.spectra or self.raw or ([self.spec] if self.spec is not None else [])
        current = {_epoch_key(s) for s in sp}
        classes = self.classes or self.ns.get("epoch_classes")
        if isinstance(classes, list):
            matched = [c for c in classes if isinstance(c, Classification) and _epoch_key(c.spectrum) in current]
            if matched:
                return matched
        c = self.ns.get("c")
        return [c] if isinstance(c, Classification) and _epoch_key(c.spectrum) in current else None

    def _final_classifications(self, cached_only=False):
        """Current final classifications. Polling a catalog uses only an already computed result."""
        if self.wf.get("fit_kind") not in ("single", "joint", "fits"):
            return None
        res = [x for x in (self.wf_results() or []) if x[2] is not None]
        if not res:
            return None
        prelim = {_epoch_key(c.spectrum): c for c in (self._classifications() or [])}
        key = (tuple(r for _, _, r, _ in res), tuple(prelim.values()))
        kept = self._final_kept
        if kept is not None and kept[0] == key:
            return kept[1]
        if cached_only:
            return None
        from ..classify import classify_fit
        out = []
        for _, _, r, _ in res:
            try:
                out.append(classify_fit(r, preliminary=prelim.get(_epoch_key(r.spectrum))))
            except Exception:                        # noqa: BLE001  (a fit that cannot be classified)
                pass
        out = out or None
        self._final_kept = (key, out)
        return out

    def _replace_last(self, marker):
        """Index of the last cell if it was made by the same workflow action (rerun it in place)."""
        if self.cells and self.cells[-1].code.startswith(marker) and len(self.cells) > 1:
            return len(self.cells) - 1
        return None

    def run_step(self, code, marker=None):
        """Run workflow code; a repeated action replaces its own previous cell."""
        if marker == "# prepare" and code.startswith("# prepare row "):
            marker = code.partition("\n")[0] + "\n"
        index = self._replace_last(marker) if marker else None
        return self.run_cell(code, index)

    # -------------------------------------------------------------- 1 Read
    def wf_input_files(self):
        """Complete ordered local input paths, or None when the current spectra were fetched."""
        files = self.ns.get("files")
        if not isinstance(files, list) or not files:
            return None
        source = self.ns.get("read_source")
        if source is not None:
            return files if source == "files" else None
        # Older saved sessions could keep a previous file list after an archive fetch.
        # Use the last completed Read cell, independent of display names and input count.
        for cell in reversed(self.cells):
            if cell.code.startswith("# read\n") and not any(kind == "err" for _, kind in cell.out):
                if "fa.sdss.fetch(" in cell.code or "fa.desi.fetch(" in cell.code:
                    return None
                if "fa.read(" in cell.code:
                    return files
        # For user-written code without a Read cell, the loaded objects are authoritative.
        return None

    def wf_read_code(self, mode="single", paths=None, pattern=None, targetid=None, sdss=None,
                     group="auto", format="auto", _files_expr=None):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        if group not in ("auto", "date", "mjd", "order"):
            raise ValueError("group epochs by auto, date, mjd or order")
        format = format or "auto"
        if format not in ("auto", "sdss", "desi", "ozdes", "ozdes-ssa", "gama", "fits1d", "text"):
            raise ValueError(f"unknown file format {format!r}")
        if sum(bool(v) for v in (paths, pattern, targetid, sdss)) != 1:
            raise ValueError("choose exactly one input source: files, path, DESI TARGETID or SDSS plate-mjd-fiber")
        read = "fa.read(f)" if format == "auto" else f"fa.read(f, format={format!r})"
        lines = []
        if targetid or sdss:
            source = "desi" if targetid else "sdss"
            items = [t for t in re.split(r"[\s,;]+", str(targetid or sdss).strip()) if t]
            if targetid:
                ids = [int(t) for t in items]
                fetch = f"[fa.desi.fetch(t) for t in {ids!r}]   # DESI DR1 via SPARCL"
            else:
                ids = []
                for item in items:
                    parts = re.split(r"[-/]", item)
                    if len(parts) != 3 or not all(p.isdigit() for p in parts):
                        raise ValueError(f"SDSS spectra as plate-mjd-fiber, e.g. 1059-52618-453 (got {item!r})")
                    ids.append(tuple(int(p) for p in parts))
                fetch = f"[fa.sdss.fetch(*pmf) for pmf in {ids!r}]   # SDSS DR17"
            if mode == "multi" and group == "date":
                raise ValueError("archive spectra use MJD or input order; filenames are unavailable")
            lines += [f"_read_raw = {fetch}", "_read_files = None"]
            count = len(ids)
        else:
            source = "files"
            glob_expr = None
            if pattern:
                files, glob_expr = resolve_pattern(pattern)
            else:
                files = [os.path.expanduser(str(p).strip()) for p in paths if str(p).strip()]
            count = len(files)
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
                    group = "mjd"
            if _files_expr:
                lines.append(f"_read_files = {_files_expr}")
            elif glob_expr and count > 40:
                lines += ["import glob, hashlib, os",
                          f"_read_files = sorted(f for f in glob.glob({glob_expr!r}) if os.path.isfile(f))"]
            else:
                lines.append("_read_files = [" + ",\n               ".join(repr(f) for f in files) + "]")
            if mode == "multi" and group == "date" and (_files_expr or glob_expr and count > 40):
                lines.append("_read_files.sort(key=fa.spectrum.date_from_name)")
            if glob_expr and count > 40:
                stamp = hashlib.sha256("\0".join(files).encode()).hexdigest()
                lines += [f"if hashlib.sha256(chr(0).join(_read_files).encode()).hexdigest() != {stamp!r}:",
                          "    raise ValueError('the files matching the original input pattern changed; read the sample again')"]
            if mode == "sample" and count > 20:
                lines.append(f"_read_raw = [{read} for f in _read_files[:20]]   # preview; the fit reads every file")
            else:
                lines.append(f"_read_raw = [{read} for f in _read_files]")
            if mode == "multi" and group == "date":
                lines += ["for s, f in zip(_read_raw, _read_files):   # dates from the file names",
                          "    s.mjd = fa.spectrum.mjd_from_name(f)"]
        if not count:
            raise ValueError("choose at least one spectrum")
        if mode == "single" and count > 1:
            raise ValueError(f"{count} spectra for a single spectrum: choose Sample or Multi-epoch")
        if mode == "multi" and count < 2:
            raise ValueError("a multi-epoch fit needs at least two spectra")
        if mode == "multi" and group in ("auto", "mjd"):
            lines += ["# epochs in time order (missing dates last)",
                      "order = sorted(range(len(_read_raw)), key=lambda k: _read_raw[k].mjd if _read_raw[k].mjd is not None else float('inf'))",
                      "_read_raw = [_read_raw[k] for k in order]"]
            if source == "files":
                lines.append("_read_files = [_read_files[k] for k in order]")
            lines += ['missing = [s.name for s in _read_raw if s.mjd is None]',
                      'if missing:\n    print("no MJD in the header of", ", ".join(missing), "(kept at the end)")']
        # Only publish the new input once every read and ordering operation has succeeded.
        options = {"mode": mode, "group": group, "format": format}
        lines += [f"raw, files, read_source, read_options = _read_raw, _read_files, {source!r}, {options!r}",
                  "del _read_raw, _read_files", self._clear_read_results(),
                  'print(len(raw), "spectra:", ", ".join(s.name for s in raw[:8]))']
        return "# read\n" + "\n".join(lines)

    @staticmethod
    def _clear_read_results():
        return ("spec = spectra = sample_prepare = r = me = fits = sample = rm = None\n"
                "c = epoch_classes = classes = catalogs = None")

    def _loaded_id(self):
        # A distinct revision survives allocator reuse after reads and session resets.
        return f"{self._read_revision}/{id(self.ns.get('raw'))}"

    def wf_remove_code(self, indices, loaded_id):
        """Drop selected preview rows from the current complete input; keep the other rows in order."""
        raw = self.raw
        if not raw or loaded_id != self._loaded_id():
            raise ValueError("the loaded spectra changed; select the rows again")
        if (not isinstance(indices, list) or not indices or
                any(type(k) is not int or not 0 <= k < len(raw) for k in indices)):
            raise ValueError("select rows in the current preview")
        drop = set(indices)
        files = self.wf_input_files()
        count = len(files) if files is not None else len(raw)
        n = count - len(drop)
        if not n:
            return ("# remove spectra\nraw = files = read_source = None\n" + self._clear_read_results())
        mode = self.wf["mode"]
        if mode == "multi" and n < 2:
            raise ValueError("a multi-epoch fit needs at least two spectra")
        if mode == "single" and n != 1:
            raise ValueError("choose Sample to keep multiple spectra")
        if files is not None:
            keep = [f for k, f in enumerate(files) if k not in drop]
            return self.wf_read_code(mode, paths=keep, group=self.wf.get("group", "auto"),
                                     format=self.wf.get("format", "auto"),
                                     _files_expr=f"[f for k, f in enumerate(files) if k not in {sorted(drop)!r}]")
        # Archive data are already in memory. Append this cell after the fetch so it can be replayed.
        return ("# remove spectra\n" +
                f"raw = [s for k, s in enumerate(raw) if k not in {sorted(drop)!r}]\n"
                "files = None\n" + self._clear_read_results())

    def wf_reselect_code(self, mode, loaded_id, group="auto"):
        """Reuse all current inputs with another read mode or ordering."""
        raw = self.raw
        if not raw or loaded_id != self._loaded_id():
            raise ValueError("the loaded spectra changed; select them again")
        files = self.wf_input_files()
        if files is not None:
            return self.wf_read_code(mode, paths=files, group=group, format=self.wf.get("format", "auto"),
                                     _files_expr="list(files)")
        if mode not in MODES or group not in ("auto", "mjd", "order"):
            raise ValueError("choose an analysis mode and MJD or input order for archive spectra")
        if mode == "single" and len(raw) != 1 or mode == "multi" and len(raw) < 2:
            raise ValueError("read the required number of spectra for this mode")
        expr = "sorted(raw, key=lambda s: s.mjd if s.mjd is not None else float('inf'))" if (
            mode == "multi" and group != "order") else "list(raw)"
        options = {"mode": mode, "group": group, "format": "auto"}
        return ("# select spectra\n" +
                f"raw, files, read_options = {expr}, None, {options!r}\n" + self._clear_read_results())

    def loaded_state(self):
        raw = self.raw
        if raw is None:
            return None
        key = ("loaded", self._loaded_id(), len(raw))
        if key in self._wf_cache:
            return self._wf_cache[key]
        files = self.wf_input_files()
        rows = []
        for k, s in enumerate(raw):
            path = files[k] if files and k < len(files) else s.meta.get("path") or s.name
            fname = os.path.basename(str(path))
            date = mjd_to_date(s.mjd) or date_from_name(fname)
            ext = fname.lower().rsplit(".", 1)[-1] if "." in fname else ""
            if self.ns.get("read_source") in ("sdss", "desi"):
                fmt = self.ns["read_source"].upper()
            elif ext in ("txt", "dat", "csv", "ascii", "asc", "tsv"):
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
                         "resolution": float(s.resolution) if s.resolution else None,
                         "resolution_varies": s.resolution_curve is not None,
                         "pixel_kms": float(s.pixel_kms()) if s.wave.size > 1 else None,
                         "flux_unit_source": s.meta.get("flux_unit_source")
                         or (f"{fmt} header" if s.flux_unit else None)})
        n_all = len(files) if files else len(raw)
        out = {"rows": rows, "n": n_all, "shown": len(raw), "id": self._loaded_id()}
        self._wf_cache[key] = out
        return out

    # ----------------------------------------------------------- 2 Prepare
    def wf_prepare_code(self, z=None, z_source="file", rest_frame=True, extinction="coords", ebv=None,
                        ra=None, dec=None, wmin=None, wmax=None, telluric=True, sky_lines=False,
                        absorption=False, masks=(), apply_to="all", flux_unit="file", resolution="file",
                        resolution_unit="kms"):
        raw = self.raw
        if raw is None:
            raise ValueError("read the spectra first (step 1)")
        steps = []
        if flux_unit not in (None, "", "file"):
            try:
                unit = _num(flux_unit)
            except ValueError:
                unit = None
            if unit is None or not 0 < unit < 1:
                raise ValueError("flux unit: give the value of one flux unit in erg/s/cm2/A, e.g. 1e-16")
            steps.append(("with_flux_unit", _g(unit)))
        res_val = None
        if resolution not in (None, "", "file"):
            try:
                res_val = _num(resolution)
            except ValueError:
                res_val = None
            if resolution_unit not in ("kms", "R"):
                raise ValueError("resolution: FWHM in km/s or resolving power R")
            if res_val is None or res_val <= 0:
                raise ValueError("instrumental resolution: give the FWHM in km/s (e.g. 150) or the resolving power "
                                 "R (e.g. 2000), or take it from the file")
            if resolution_unit == "R" and res_val < 60 or resolution_unit == "kms" and res_val >= 5000:
                raise ValueError("instrumental resolution: the FWHM must be below 5000 km/s (R above 60)")
            steps.append(("with_resolution", ("R=" if resolution_unit == "R" else "") + _g(res_val)))
        if ra not in (None, "") or dec not in (None, ""):
            from ..spectrum import parse_coordinates
            parse_coordinates(ra, dec)
            pos = lambda v: repr(float(v)) if _is_number(v) else repr(str(v).strip())
            steps.append(("with_coordinates", f"{pos(ra)}, {pos(dec)}"))
        if extinction == "value":
            if ebv in (None, ""):
                raise ValueError("give E(B-V), or choose 'From coordinates'")
            steps.append(("deredden", f"ebv={_g(ebv)}"))
        elif extinction == "coords":
            steps.append(("deredden", ""))
        elif extinction != "none":
            raise ValueError("unknown extinction choice")
        if telluric or sky_lines:
            opts = [] if telluric else ["telluric=False"]
            opts += [] if sky_lines else ["sky_lines=False"]
            steps.append(("mask_sky", ", ".join(opts)))
        if z_source not in ("file", "manual", "median"):
            raise ValueError("unknown redshift choice")
        zval = _num(z) if z_source == "manual" else None
        if z_source == "manual" and zval is None:
            raise ValueError("give the redshift z, or take it from the file")
        if z_source == "median" and not any(s.z is not None and np.isfinite(s.z) for s in raw):
            raise ValueError("these files have no redshift: enter z")
        redshift = "z=z" if z_source == "median" else f"z={_g(zval)}" if zval is not None else ""
        if rest_frame:
            steps.append(("to_rest_frame", redshift))
            check = raw if apply_to in ("all", None, "") else raw[int(apply_to):int(apply_to)+1]
            if z_source == "file" and any(s.z is None and not s.rest_frame for s in check):
                raise ValueError("these files have no redshift: enter z")
        elif redshift:
            steps.append(("copy", redshift))
        lo, hi = _num(wmin), _num(wmax)
        if lo is not None or hi is not None:
            if lo is not None and hi is not None and lo >= hi:
                raise ValueError("λ min must be below λ max")
            steps.append(("crop", f"{_g(lo) if lo is not None else 'None'}, {_g(hi) if hi is not None else 'None'}"))
        ranges = []
        for m in masks or ():
            a, b = sorted(float(v) for v in m)
            ranges.append(f"({_g(a)}, {_g(b)})")
        if ranges:
            steps.append(("mask", ", ".join(ranges)))
        if absorption:
            steps.append(("mask_absorption", ""))
        calls = [f".{method}({args})" for method, args in steps]
        chain = ("\n            " if sum(map(len, calls)) > 60 else "").join(calls)
        lines = ["# prepare"]
        if z_source == "median":
            lines += ["import numpy as np",
                      "z = float(np.median([s.z for s in raw if s.z is not None and np.isfinite(s.z)]))"]
        if apply_to not in ("all", None, ""):
            k = int(apply_to)
            if not 0 <= k < len(raw):
                raise ValueError(f"epoch {k} does not exist")
            if self.spectra is None or len(self.spectra) != len(raw):
                raise ValueError("prepare all spectra once before changing one epoch")
            lines[0] = f"# prepare row {k}"
            lines.append(f"spectra = [*spectra[:{k}], (raw[{k}]{chain}), *spectra[{k + 1}:]]")
        else:
            lines.append(f"spectra = [s{chain}\n           for s in raw]" if "\n" in chain else f"spectra = [s{chain} for s in raw]")
            if self.wf.get("mode") == "sample":
                lines.append("from operator import methodcaller")
                recipe = [f"methodcaller({method!r}{', ' + args if args else ''})" for method, args in steps]
                lines.append("sample_prepare = [" + ",\n                  ".join(recipe) + "]")
        if self.wf.get("mode") == "single" or (self.wf.get("mode") == "classify" and len(raw) == 1):
            lines.append("spec = spectra[0]")
        lines.append('print(len(spectra), "spectra · z =", spectra[0].z, "· E(B-V) =", '
                     'spectra[0].meta.get("ebv"), spectra[0].meta.get("ebv_source", ""))')
        options = {"z": zval, "z_source": z_source, "rest_frame": bool(rest_frame),
                   "extinction": extinction, "ebv": _num(ebv), "ra": ra, "dec": dec,
                   "wmin": lo, "wmax": hi, "telluric": bool(telluric), "sky_lines": bool(sky_lines),
                   "absorption": bool(absorption), "masks": [list(map(float, m)) for m in masks or ()],
                   "flux_unit": flux_unit if flux_unit not in (None, "") else "file",
                   "resolution": res_val if res_val is not None else "file",
                   "resolution_unit": resolution_unit if res_val is not None else "kms"}
        lines.append(f"_prepare_form = {options!r}")
        return "\n".join(lines)

    # ------------------------------------------------- finding the redshift
    def wf_find_redshift_code(self, k=0):
        """Code that finds the redshift of one spectrum as read (fa.find_redshift)."""
        raw = self.raw
        if raw is None:
            raise ValueError("read the spectra first (step 1)")
        k = int(k)
        if not 0 <= k < len(raw):
            raise ValueError(f"spectrum {k} does not exist")
        if raw[k].rest_frame:
            raise ValueError("this spectrum is already in the rest frame: the search needs the observed one")
        return ("# find redshift\n"
                f"zf = fa.find_redshift(raw[{k}])   # chi2 against galaxy and quasar templates at every redshift\n"
                "print(zf.summary())")

    def _redshift_result(self):
        from ..redshift import RedshiftResult
        zf, raw = self.ns.get("zf"), self.raw
        if not isinstance(zf, RedshiftResult) or raw is None:
            return None, None
        k = next((i for i, s in enumerate(raw) if s is zf.spectrum), None)
        return (zf, k) if k is not None else (None, None)

    def wf_redshift(self):
        """The last redshift search, while it belongs to one of the spectra read (for the Prepare step)."""
        from ..redshift import MARK_LINES
        zf, k = self._redshift_result()
        if zf is None:
            return None
        num = lambda v: float(v) if v is not None and np.isfinite(v) else None
        rows = [{"z": num(d["z"]), "error": num(d["error"]), "kind": d["kind"],
                 "dchi2": num(d["dchi2"] / zf.n_pixels), "rchi2": num(d["rchi2"]), "peaks": d["peaks"],
                 "lines": d["lines"]} for d in zf.candidates.to_dict("records")]
        return {"id": id(zf), "k": k, "z": num(zf.z), "error": num(zf.error), "kind": zf.kind,
                "secure": bool(zf.secure), "method": zf.method, "file_z": num(zf.spectrum.z),
                "threshold": float(zf.min_dchi2), "candidates": rows,
                "marks": [[n, float(w), t] for n, w, t in MARK_LINES]}

    def wf_redshift_curve(self, n=700):
        """chi2 per pixel against log(1+z) for each template set of the last redshift search, for the
        interactive plot of the Prepare step. Thinned to about ``n`` points by keeping the lowest value of
        each bin, so that no minimum is lost."""
        zf, _ = self._redshift_result()
        if zf is None:
            return None
        key = ("redshift_curve", id(zf))
        if key in self._wf_cache:
            return self._wf_cache[key]
        curves = {}
        for kind, (z, c) in zf.curves.items():
            z, c = np.asarray(z, float), np.asarray(c, float) / zf.n_pixels
            ok = np.isfinite(c) & (z > -1)
            if ok.sum() < 2:
                continue
            x, c = np.log10(1 + z[ok]), c[ok]
            if len(x) > n:
                edges = np.linspace(0, len(x), n + 1).astype(int)
                pick = [a + int(np.argmin(c[a:b])) for a, b in zip(edges[:-1], edges[1:]) if b > a]
                x, c = x[pick], c[pick]
            curves[kind] = {"x": [round(float(v), 6) for v in x], "y": [round(float(v), 5) for v in c]}
        out = {"id": id(zf), "curves": curves}
        self._wf_cache[key] = out
        return out

    def wf_redshift_figure(self, fmt="png"):
        from matplotlib.figure import Figure
        zf, _ = self._redshift_result()
        if zf is None:
            return None
        with self._render_lock:
            return self._fig_bytes(zf.plot(fig=Figure(figsize=(7.1, 5.0))), fmt, 130 if fmt == "png" else 300)

    def prepared_state(self):
        sp = self.spectra
        if sp is None:
            return None
        s = next((x for x in sp if x.good.any()), sp[0])       # the first usable one: a broken file has no z, no pixels
        good = s.good
        raw = self.raw if self.raw is not None and len(self.raw) == len(sp) else sp
        zs = [float(x.z) for x, y in zip(raw, sp) if y.good.any() and x.z]      # the redshifts of the files
        return {"n": len(sp), "z": s.z, "ebv": s.meta.get("ebv"), "ebv_source": s.meta.get("ebv_source"),
                "z_spread_kms": (float(299792.458 * (max(zs) - min(zs)) / (1 + float(np.median(zs)))) if len(zs) > 1 else 0.0),
                "z_rms_kms": (float(299792.458 * np.std(zs) / (1 + float(np.median(zs)))) if len(zs) > 1 else 0.0),
                "z_one": len({x.z for x in sp if x.good.any()}) == 1,
                "unusable": sum(not x.good.any() for x in sp),
                "flux_unit": s.flux_unit,
                "resolution": float(s.resolution) if s.resolution else None,
                "resolution_given": s.meta.get("resolution_source") == "given",
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
                "z": float(s.z) if s.z else None, "medium": s.medium,
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
        sig = (id(s), id(m), k, m.wave_range and tuple(m.wave_range),
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
                    uncertainty=False, absorbers=False, sample_model="auto", out=None, shared_shape=False,
                    prune_feii=False, trial=None, from_trial=True, wavelength_offsets=False, epoch_broadening=False):
        mode = self.wf.get("mode")
        self.wf["fit_opts"] = {"epochs": epochs, "reference": reference, "n_jobs": n_jobs, "shared": shared,
                               "shared_shape": bool(shared_shape),
                               "errors": errors, "scale_errors": bool(scale_errors),
                               "rescale_errors": bool(rescale_errors), "calibration_degree": int(calibration_degree or 0),
                               "host_per_epoch": bool(host_per_epoch), "uncertainty": bool(uncertainty),
                               "absorbers": bool(absorbers), "sample_model": sample_model,
                               "prune_feii": bool(prune_feii), "from_trial": bool(from_trial),
                               "wavelength_offsets": bool(wavelength_offsets),
                               "epoch_broadening": bool(epoch_broadening)}
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
            return self._wf_sample_code(sample_model, n_jobs, out, prune_feii)
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
        if not (trial is None or trial is False or (isinstance(trial, str) and trial == "")):   # epoch 0 is a choice
            # one epoch first: the model is worked out on it (with all the checks of a single fit)
            # before every epoch is fitted
            k = self.wf_reference() if trial is True or (isinstance(trial, str) and trial == "auto") else int(trial)
            if not 0 <= k < len(sp):
                raise ValueError(f"epoch {k} does not exist")
            if not sp[k].rest_frame:
                raise ValueError("the models are defined in the rest frame: tick 'Move to rest frame' in Prepare")
            self.wf["fit_kind"] = "single"
            args = "" if errors == "covariance" else f", errors={err_arg!r}"
            if not scale_errors:
                args += ", scale_errors=False"
            date = mjd_to_date(sp[k].mjd)
            return ("# trial fit\n"
                    f"spec = spectra[{k}]   # one epoch first{f' ({date})' if date else ''}: agree on the model, then fit them all\n"
                    f"r = fa.fit(spec, model{args})\nr.key_results()")
        # the epochs start from the trial fit of one of them, if there is one
        start = "model.starting_from(r)" if from_trial and self._trial() is not None else "model"
        if epochs == "own":
            self.wf["fit_kind"] = "fits"
            opts = [f"n_jobs={n_jobs}"]
            if errors != "covariance":
                opts.append(f"errors={err_arg!r}")
            if not scale_errors:
                opts.append("scale_errors=False")
            return ("# fit\nfits = fa.fit_many(spectra, " + start + ", " + ", ".join(opts) + ")   # each epoch on its own\n"
                    "me = fa.EpochSet(fits) if sum(hasattr(f, 'rchi2') for f in fits) > 1 else None"
                    "   # together: light curves, lags (not intercalibrated)\n"
                    "[(f.spectrum.name, round(f.rchi2, 3)) for f in fits if hasattr(f, 'rchi2')]")
        if errors in ("mc", "mcmc"):
            raise ValueError("the joint fit has covariance errors (or none); Monte Carlo and MCMC are for independent fits")
        self.wf["fit_kind"] = "joint"
        patterns = tuple(p.strip() for p in str(shared or "").split(",") if p.strip())
        if patterns:
            # patterns typed by hand replace the table (the expert override of the Fit step)
            if shared_shape:
                patterns = patterns + tuple(self.broad_shape_params())
        else:
            # the sharing table of the Model step. By default the narrow lines and the host are shared
            # and, with shared_shape, the shape of the broad lines and the Fe II width: free shapes per
            # epoch trade flux with the continuum and Fe II at SDSS S/N and can wash out the
            # reverberation signal.
            patterns = self.sharing_patterns(shape_default=shared_shape) or ()
            if not patterns and self.wf.get("sharing"):
                raise ValueError("nothing is shared between the epochs: share at least the narrow lines (Model step), "
                                 "or fit each epoch on its own")
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
        if wavelength_offsets:
            opts.append("wavelength_offsets=True")
        if epoch_broadening:
            opts.append("epoch_broadening=True")
        scaled = [] if patterns and str(shared or "").strip() else self.sharing_scaled()
        if host_per_epoch:
            scaled = [c.name for c in self.model.components if c.kind == "host"]
            if not scaled:
                raise ValueError("the model has no host galaxy to scale per epoch")
        if scaled:
            opts.append(f"epoch_scaled={tuple(scaled)!r}")
        if rescale_errors:
            opts.append("rescale_errors=True")
        if not scale_errors:
            opts.append("scale_errors=False")
        if err_arg is None:
            opts.append("errors=None")
        opts.append(f"n_jobs={n_jobs}")
        return ("# fit\nme = fa.fit_epochs(spectra, " + start + ",\n                   " +
                ",\n                   ".join(opts) + ")")

    def _trial(self):
        """(epoch index, FitResult) of the trial fit of one epoch, while it belongs to the prepared
        spectra and shares parameters with the current model; else None."""
        r, sp, m = self.result, self.spectra, self.model
        if r is None or sp is None or m is None or self.wf.get("mode") != "multi" or len(sp) < 2:
            return None
        k = next((i for i, s in enumerate(sp) if s is r.spectrum), None)
        if k is None or not any(p.free and p.name in r.model.params for p in m.params):
            return None
        return k, r

    def wf_trial(self):
        """The trial fit of one epoch, for the Fit and Results steps of a multi-epoch analysis."""
        t = self._trial()
        if t is None:
            return None
        k, r = t
        m = self.model
        same = [p.name for p in m.params] == [p.name for p in r.model.params]
        return {"k": k, "name": r.spectrum.name, "date": mjd_to_date(r.spectrum.mjd), "rchi2": float(r.rchi2),
                "n": len(self.spectra), "active": self.wf.get("fit_kind") == "single",
                "model_changed": not same}

    # ---------------------------------------------- what is the same in all epochs
    def sharing_table(self, shape_default=True):
        """
        For every component of the model, whether its fluxes and its shape (shift, width, profile
        parameters, slope: everything that is not an amplitude) take one value for all epochs of a joint
        fit or one per epoch. ``flux``/``shape`` are True (shared), False (per epoch) or None (the
        component has no free parameter of that kind). The defaults are those of ``fa.fit_epochs`` (the
        component ``narrow`` and a stellar host: everything shared) plus, with ``shape_default``, the
        shape of the broad lines and of Fe II; choices made in the Model step (``wf['sharing']``)
        replace them. For a stellar host, ``flux`` False means its spectrum (the template weights) is
        shared but its amplitude is free in each epoch (``epoch_scaled``): the host seen through
        different apertures or seeing; ``host`` marks those rows.
        """
        m = self.model
        if m is None:
            return []
        chosen = self.wf.get("sharing") or {}
        rows = []
        for c in m.components:
            free = [p for p in c.params if not p.fixed and not p.tie]
            has_flux, has_shape = any(p.linear for p in free), any(not p.linear for p in free)
            whole = c.name == "narrow" or c.kind == "host"
            broad = getattr(c, "role", None) == "broad" or type(c).__name__ == "FeII"
            d_flux, d_shape = whole, whole or (bool(shape_default) and broad)
            pick = chosen.get(c.name) or {}
            rows.append({"name": c.name,
                         "flux": bool(pick.get("flux", d_flux)) if has_flux else None,
                         "shape": bool(pick.get("shape", d_shape)) if has_shape else None,
                         "default_flux": d_flux if has_flux else None,
                         "default_shape": d_shape if has_shape else None,
                         "lines": list(getattr(c, "line_names", []) or [])[:6],
                         "host": c.kind == "host"})
        return rows

    def sharing_scaled(self):
        """Host components whose amplitude is free per epoch in the sharing table (``epoch_scaled``)."""
        return [r["name"] for r in self.sharing_table() if r.get("host") and r["flux"] is False]

    def set_sharing(self, component=None, part=None, shared=None, reset=False):
        """Choose whether the fluxes or the shape of a component are shared by the epochs (a setting, not code)."""
        if reset:
            self.wf.pop("sharing", None)
            return
        if self.model is None or component not in [c.name for c in self.model.components]:
            raise ValueError(f"no component {component!r} in the model")
        if part not in ("flux", "shape"):
            raise ValueError("part: flux or shape")
        table = self.wf.setdefault("sharing", {})
        table.setdefault(component, {})[part] = bool(shared)

    def sharing_patterns(self, shape_default=True):
        """The ``shared=`` argument of fa.fit_epochs for the sharing table: whole components as
        'name.*', then the single parameters of those shared in part. None: the default of fit_epochs."""
        m = self.model
        rows = self.sharing_table(shape_default)
        whole, single = [], []
        for c, row in zip(m.components, rows):
            flux, shape = row["flux"], row["shape"]
            if row.get("host") and flux is False:
                flux = True                             # the host's spectrum is shared, its amplitude is not
            if flux in (True, None) and shape in (True, None) and (flux or shape):
                whole.append(f"{c.name}.*")
                continue
            for p in c.params:
                if p.fixed or p.tie:                    # a tied parameter follows its target
                    continue
                if (flux and p.linear) or (shape and not p.linear):
                    single.append(p.name)
        default = ["narrow.*"] if any(c.name == "narrow" for c in m.components) else []
        default += [f"{c.name}.*" for c in m.components if c.kind == "host"]
        if not single and whole == default and default:
            return None
        return tuple(whole + single)

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

    def _legacy_sample_preparation(self):
        """Recover the all-row chain from a pre-recipe session, accepting only generated calls."""
        import ast
        methods = {"with_flux_unit", "with_resolution", "with_coordinates", "deredden", "mask_sky",
                   "to_rest_frame", "crop", "mask", "mask_absorption"}
        for cell in reversed(self.cells):
            code = cell.code
            if code.startswith("# read"):
                break
            if not code.startswith("# prepare\n") or "spectra = [" not in code:
                continue
            try:
                tree = ast.parse(code)
                comp = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                            and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "spectra")
                if not isinstance(comp, ast.ListComp) or len(comp.generators) != 1:
                    break
                source = comp.generators[0]
                if source.ifs or source.is_async or ast.unparse(source.iter) != "raw" or ast.unparse(source.target) != "s":
                    break
                node, calls = comp.elt, []
                while isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in methods:
                    args = [repr(ast.literal_eval(arg)) for arg in node.args]
                    for kw in node.keywords:
                        if kw.arg == "z" and ast.unparse(kw.value) == "z" and "z = float(np.median([s.z for s in raw if s.z]))" in code:
                            args.append("z=float(np.median([s.z for s in raw if s.z]))")
                        else:
                            args.append(f"{kw.arg}={ast.literal_eval(kw.value)!r}")
                    calls.append(f"methodcaller({', '.join([repr(node.func.attr), *args])})")
                    node = node.func.value
                if isinstance(node, ast.Name) and node.id == "s":
                    return ["from operator import methodcaller", "import numpy as np",
                            "sample_prepare = [" + ", ".join(reversed(calls)) + "]"]
            except (SyntaxError, ValueError, IndexError, StopIteration, TypeError):
                pass
            break
        raise ValueError("repeat Prepare for all spectra to record how to prepare the files beyond the preview")

    def wf_sample_plan(self):
        """The Model step of a sample: which model each class gets (Classification.RECOMMENDED, the presets
        in words and code), and per spectrum its route (ends below 5200 Å in the rest frame: the UV model of
        Mg II or C IV; else the optical classifier at fit time), with the class and model it got once fitted."""
        import inspect
        from .. import presets
        from ..classify import Classification, UV_RED_LIMIT, _uv_choice
        from .session import MODELS
        src = self.spectra if self.spectra is not None else self.raw
        sm = self.sample
        key = ("sample_plan", id(src), id(sm), str((self.wf.get("fit_opts") or {}).get("sample_model")))
        if key in self._wf_cache:
            return self._wf_cache[key]
        num = lambda v: float(v) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else None

        def summary(name):
            doc = (inspect.getdoc(getattr(presets, name)) or "").split("\n\n")[0].replace("\n", " ")
            doc = re.sub(r"``([^`]+)``", r"\1", doc)
            first = re.split(r"(?<=[a-z0-9)])\. ", doc)
            text = first[0] + ("." if len(first) > 1 else "")
            return text if len(text) <= 260 else text[:257].rsplit(" ", 1)[0] + "…"
        order = ["type1", "type1_host", "seyfert2", "liner", "composite", "starforming", "retired", "passive",
                 "unknown", "uv_mgii", "uv_civ"]
        routes = []
        for cls in order:
            name, kw = Classification.RECOMMENDED[cls]
            args = ", ".join(f"{k}={v!r}" for k, v in kw.items())
            routes.append({"cls": cls, "preset": name, "code": f"fa.presets.{name}({args})", "about": summary(name)})
        table = sm.table if sm is not None else None
        rows = []
        for k, sp in enumerate(src or []):
            row = {"k": k, "name": getattr(sp, "name", f"spectrum {k}"), "z": num(getattr(sp, "z", None)),
                   "blue": None, "red": None, "route": "optical", "cls": None, "preset": None}
            try:
                rest = sp if sp.rest_frame else sp.to_rest_frame()
                good = rest.good
                if good.any():
                    row["blue"], row["red"] = float(rest.wave[good].min()), float(rest.wave[good].max())
                uv = _uv_choice(rest)
                row["route"] = uv.label if uv is not None else "optical"
            except ValueError as exc:       # no redshift, or no line model for this range
                row["route"] = "none"
                row["why"] = str(exc)
            if table is not None and k < len(table):
                t = table.iloc[k]
                row["cls"] = str(t["cls"]) if "cls" in t and isinstance(t["cls"], str) else None
                row["preset"] = str(t["preset"]) if "preset" in t and isinstance(t["preset"], str) else None
            rows.append(row)
        choice = (self.wf.get("fit_opts") or {}).get("sample_model") or "auto"
        files = self.wf_input_files()
        counts = ({str(c): int(n) for c, n in table["cls"].value_counts().items()}
                  if table is not None and "cls" in table else {})
        out = {"routes": routes, "spectra": rows, "uv_limit": UV_RED_LIMIT, "fitted": sm is not None,
               # the rows are the spectra read for the preview (the first 20 files); the fit reads them all
               "n_input": len(table) if table is not None else max(len(files or []), len(rows)),
               "class_counts": counts,
               "one_model": None if choice in ("auto", "") else {"label": choice, "preset": MODELS.get(choice, (choice, {}))[0]}}
        self._wf_cache[key] = out
        return out

    def _wf_sample_code(self, sample_model, n_jobs, out, prune_feii=False):
        from .session import MODELS
        if self.raw is None:
            raise ValueError("read the spectra first (step 1)")
        files = self.wf_input_files()
        source = "files" if files else "raw"
        args = []
        lines = ["# fit"]
        if out is None:
            out = os.path.join(os.path.expanduser("~"), ".fantasy_agn", "sample_fits",
                               f"sample_{datetime.datetime.now():%Y%m%d_%H%M%S}")
        args.append(f"out={out!r}")
        if sample_model not in ("auto", None, ""):
            name, kw = MODELS[sample_model]
            args.append(f"model={(name, kw)!r}" if kw else f"model={name!r}")
        prepared = self.spectra
        if prepared is not None:
            args.append("prepared=spectra")
            if files and len(files) > len(prepared):
                if self.ns.get("sample_prepare") is None:
                    lines.extend(self._legacy_sample_preparation())
                args.append("prepare=sample_prepare")
        fmt = self.wf.get("format", "auto")
        if files and fmt != "auto":
            args.append(f"format={fmt!r}")
        if prune_feii:
            args.append("prune_feii=True")
        args.append(f"n_jobs={int(n_jobs)}")
        self.wf["fit_kind"] = "sample"
        lines += [f"sample = fa.fit_sample({source}, " + ", ".join(args) + ")",
                  'sample.table[[c for c in ("name", "cls", "preset", "rchi2") if c in sample.table]]']
        return "\n".join(lines)

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
                           "limits": list(fr.at_limit), "scale": float(getattr(fr, "scale", 1.0) or 1.0),
                           "undersampled": [{"parameter": u["parameter"], "pixels": round(u["pixels"], 2),
                                             "fwhm": round(u["fwhm"], 1), "pixel": round(u["pixel"], 1),
                                             "resolution_known": u["resolution_known"]}
                                            for u in (fr.undersampled() if hasattr(fr, "undersampled") else [])]})
        if kind == "joint" or (kind is None and self.multi is not None):
            me = self.multi
            out = {"kind": "joint", "chi2": float(me.chi2), "dof": int(me.dof), "rchi2": float(me.rchi2),
                   "error_scale": float(me.error_scale), "time": float(me.time), "nfev": int(me.nfev),
                   "success": bool(me.success), "summary": me.summary(), "epochs": epochs,
                   "reference": int(getattr(me, "reference", 0))}
            # median flux-scale error of the epochs; NaN without intercalibration
            off = getattr(me, "offsets", None)
            if off is not None:
                v, e = off.offset[~off.reference].to_numpy(float), off.error[~off.reference].to_numpy(float)
                out["offsets"] = {"rms": float(np.sqrt(np.mean(v ** 2))), "min": float(v.min()), "max": float(v.max()),
                                  "error": float(np.nanmedian(e)) if np.isfinite(e).any() else None}
            br = getattr(me, "broadening", None)
            if br is not None:
                v = br.broadening[~br.reference].to_numpy(float)
                out["broadening"] = {"min": float(v.min()), "max": float(v.max()), "median": float(np.median(np.abs(v)))}
            try:
                z, ze = me.redshift()
                out["z_best"] = [float(z), float(ze) if np.isfinite(ze) else None]
            except Exception:                              # no redshift, or the narrow shift is not shared
                out["z_best"] = None
            cal = getattr(me, "calibration_error", None)
            try:
                cal = float(cal)
            except (TypeError, ValueError):
                cal = None
            out["calibration_error"] = cal if cal is not None and np.isfinite(cal) else None
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
        from ..plotting import is_bowen, is_coronal, plain_line
        s = r.spectrum
        lo, hi = r.model.wave_range or (s.wave.min(), s.wave.max())
        sel = (s.wave >= lo) & (s.wave <= hi) & np.isfinite(s.flux)
        idx = np.nonzero(sel)[0]
        idx = idx[::_stride(idx.size)]
        parts = r.model_flux(components=True)
        model = r.model_flux()
        series, narrow, coronal, cont = [], [], [], []
        for comp in r.model.components:
            if comp.name not in parts:
                continue
            role, kind = getattr(comp, "role", None), getattr(comp, "kind", "")
            y = parts[comp.name]
            if kind == "continuum" and not comp.name.startswith("balmer"):
                cont.append(y)
            elif kind == "host":
                series.append({"name": comp.name, "role": "host", "y": y})
            elif role == "narrow" and is_coronal(comp):
                coronal.append(y)
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
        if coronal:
            series.append({"name": "coronal", "role": "coronal", "y": np.sum(coronal, axis=0)})
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
            labels.append({"x": float(row.wavelength), "text": plain_line(row.line).replace(" λ", " "), "line": row.line,
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
            elif not np.isfinite(row.flux) or flux_p in r.unmeasured_fluxes:
                flags.append(("not measured", "warn"))
            elif flux_p in r.at_zero or row.flux <= 0:
                flags.append(("not detected", "muted"))
            elif np.isfinite(row.flux_err) and row.flux_err > 0 and row.flux < 2 * row.flux_err:
                flags.append(("weak (<2σ)", "muted"))
            if fwhm_p in params.names and params[fwhm_p].tie:
                flags.append((f"FWHM tied to {params[fwhm_p].tie[0]}", "info"))
            if fwhm_p in r.at_limit:
                flags.append(("FWHM at bound", "warn"))
            if shift_p in r.at_limit:
                flags.append(("shift at bound", "warn"))
            limit = None
            if flags and flags[0][0] in ("not detected", "weak (<2σ)") and "-" not in ln:
                # what the spectrum allows: a 3 sigma limit from the noise under a line of this shape
                # (the 95% limit of the posterior with MCMC errors)
                try:
                    lim = r.line_limit(ln, comp)
                    if np.isfinite(lim["limit"]):
                        limit = {"value": float(lim["limit"]),
                                 "how": "95% of the posterior" if lim["method"] == "mcmc" else "3σ from the noise under the line"}
                except Exception:                  # noqa: BLE001  (a line without its own flux parameter)
                    limit = None
            rows.append({"component": comp, "role": roles.get(comp), "line": ln, "label": plain_line(ln), "limit": limit,
                         "wave": float(row.wavelength),
                         "flux": float(row.flux), "flux_err": float(row.flux_err),
                         "fwhm": float(row.fwhm), "fwhm_err": float(row.fwhm_err),
                         "shift": float(row.shift), "shift_err": float(row.shift_err),
                         "ew": float(row.ew), "ew_err": float(row.ew_err),
                         "detected": not any(f[0] in ("not detected", "outside range", "not measured", "weak (<2σ)") for f in flags),
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

    @staticmethod
    def _atlas_card(r, nbins=200, kv=None):
        """
        One card of Results › Atlas: the fit binned to ``nbins`` points over its fitted range (data, model,
        broad lines, host, residuals in sigma) and the numbers the cards are ordered and flagged by. ``kv``:
        those numbers already at hand (a sample's catalog row), else from ``r.key_results()``.
        """
        good = np.asarray(r.mask, bool)
        if not good.any():
            raise ValueError("no pixel was fitted")
        i0, i1 = int(np.argmax(good)), len(good) - int(np.argmax(good[::-1]))
        s = r.spectrum
        w, f, e, good = s.wave[i0:i1], s.flux[i0:i1], s.err[i0:i1], good[i0:i1]
        parts = {k: v[i0:i1] for k, v in r.model_flux(components=True).items()}
        model = np.sum(list(parts.values()), axis=0)
        comps = r.model.components
        broad = sum((parts[c.name] for c in comps if getattr(c, "role", None) == "broad"), np.zeros_like(w))
        host = sum((parts[c.name] for c in comps if getattr(c, "kind", "") == "host"), np.zeros_like(w))
        edges = np.linspace(w[0], w[-1], nbins + 1)
        k = np.clip(np.digitize(w, edges) - 1, 0, nbins - 1)
        z = (f - model) / e

        def binned(y, use):
            out = []
            for b in range(nbins):
                v = y[(k == b) & use]
                v = v[np.isfinite(v)]
                out.append(round(float(np.median(v)), 4) if v.size else None)
            return out

        allpix = np.ones_like(good)
        zz = z[good & np.isfinite(z)]
        run = np.convolve(zz, np.ones(40) / 40, mode="valid") * np.sqrt(40) if zz.size >= 40 else np.zeros(1)
        num = lambda v: None if v is None or not np.isfinite(v) else float(v)
        if kv is None:
            kv = {}
            try:
                kr = r.key_results()
                kv = dict(zip(kr["quantity"], kr["value"]))
            except Exception:
                pass
        fw = next((q for q in kv if q.startswith("broad ") and q.endswith(" FWHM") and num(kv[q]) is not None), None)
        bline = fw.split()[1] if fw else None
        mbh, lum, edd = kv.get("log M_BH"), None, None
        try:                 # Eddington ratio with bolometric corrections 9.26 (5100 A) and 5.15 (3000 A)
            at, bc = (3000.0, 5.15) if bline in ("MgII2798", "CIV1549") else (5100.0, 9.26)
            lum = r.luminosity(at)[0]
            if mbh is not None and np.isfinite(mbh):
                edd = float(np.log10(bc * 10 ** lum / (1.26e38 * 10 ** mbh)))
        except Exception:
            pass
        flux_b = binned(f, good)
        return {"z": num(getattr(s, "z", None)), "mjd": num(s.mjd) if s.mjd is not None else None,
                "rchi2": num(r.rchi2), "limits": list(r.at_limit),
                "fwhm": num(kv.get(fw)) if fw else None, "bline": bline, "bflux": num(kv.get(f"broad {bline} flux")) if bline else None,
                "rfeii": num(kv.get("R_FeII")) if bline == "Hb" else None,
                "width_limits": [p for p in r.at_limit if p.endswith(".fwhm")
                                 and any(c.role == "broad" and bline in c.line_names and
                                         p in (f"{c.name}.fwhm", f"{c.name}.{bline}.fwhm")
                                         for c in r._line_components())],
                "mbh": num(mbh), "lum": num(lum), "edd": num(edd),
                "o3hb": num(kv.get("log [OIII]/Hb", kv.get("log [OIII]5007/Hb"))),
                "struct": float(np.max(np.abs(run))), "w0": float(w[0]), "w1": float(w[-1]),
                "scale": float(np.nanpercentile([v for v in flux_b if v is not None] or [1.0], 99)) or 1.0,
                "flux": flux_b, "model": binned(model, allpix), "broad": binned(broad, allpix) if broad.any() else None,
                "host": binned(host, allpix) if host.any() else None,
                "resid": [None if v is None else round(v, 2) for v in binned(z, good)]}

    def wf_atlas(self):
        """
        Results › Atlas: every fit of a sample (or every epoch of a multi-epoch fit) as a small card, with
        the numbers the cards can be ordered by. Cached per sample or per set of epoch fits.
        """
        kind = self.wf.get("fit_kind")
        if kind == "sample":
            sm = self.sample
            if sm is None:
                return None
            key = ("atlas", id(sm))
            if key in self._wf_cache:
                return self._wf_cache[key]
            t = sm.table
            text = lambda v: v if isinstance(v, str) else ""
            cards = []
            for k in range(len(t)):
                row = t.iloc[k]
                c = {"k": k, "name": str(row.get("name", f"row {k}")), "cls": text(row.get("cls")), "preset": text(row.get("preset"))}
                err = row.get("error") if "error" in t else None
                if isinstance(err, str) and err:
                    c["error"] = err
                else:
                    try:
                        kv = {q: float(v) for q, v in row.items() if isinstance(v, (int, float, np.floating)) and not q.endswith(" err")}
                        c.update(self._atlas_card(sm.load(k), kv=kv))
                    except Exception as exc:
                        c["error"] = f"{type(exc).__name__}: {exc}"
                cards.append(c)
            out = {"kind": "sample", "cards": cards}
        else:
            res = self.wf_results()
            if not res or len(res) < 2:
                return None
            key = ("atlas", tuple(id(r) for _, _, r, _ in res))
            if key in self._wf_cache:
                return self._wf_cache[key]
            cards = []
            for k, (name, date, r, ref) in enumerate(res):
                c = {"k": k, "name": name, "date": date, "reference": bool(ref), "cls": ""}
                if r is None:
                    c["error"] = "this spectrum could not be fitted"
                else:
                    try:
                        c.update(self._atlas_card(r))
                    except Exception as exc:
                        c["error"] = f"{type(exc).__name__}: {exc}"
                cards.append(c)
            out = {"kind": "epochs", "cards": cards}
        if len(self._wf_cache) > 80:
            self._wf_cache.clear()
        self._wf_cache[key] = out
        return out

    def wf_degeneracy(self, k=0):
        """
        Results › Degeneracies: the free parameters of fit ``k`` and their correlation matrix (from
        the MCMC samples when there are, else from the covariance), for the wheel of the browser.
        """
        from ..plotting import plain_line
        res = self.wf_results()
        if not res:
            return None
        k = min(max(int(k), 0), len(res) - 1)
        name, _, r, _ = res[k]
        if r is None:
            return {"k": k, "name": name, "error": "this spectrum could not be fitted"}
        key = ("degeneracy", id(r))
        if key in self._wf_cache:
            return self._wf_cache[key]
        plist = list(r.model.params)
        if r.samples is not None and len(r.samples) > 10:
            sam = np.asarray(r.samples, float)
            sig = sam.std(axis=0)
            src = "MCMC samples"
        elif r.cov is not None:
            sig = np.sqrt(np.clip(np.diag(r.cov), 0, None))
            src = "covariance"
        else:
            out = {"k": k, "name": name, "error": "this fit has no errors (errors='none'), so no correlations"}
            self._wf_cache[key] = out
            return out
        free = [i for i, q in enumerate(plist) if q.free and np.isfinite(sig[i]) and sig[i] > 0]
        if len(free) < 2:
            return {"k": k, "name": name, "error": "fewer than two free parameters with an error"}
        if src == "MCMC samples":
            C = np.corrcoef(sam[:, free], rowvar=False)
        else:
            C = r.cov[np.ix_(free, free)] / np.outer(sig[free], sig[free])
        C = np.clip(np.nan_to_num(C), -1, 1)
        status = dict(zip(r.params["parameter"], r.params["status"]))
        comps = {c.name: c for c in r.model.components}
        broad_n = {}
        groups = []
        for c in r.model.components:
            role, kind, tname = getattr(c, "role", None), getattr(c, "kind", ""), type(c).__name__
            fam = ("broad" if role == "broad" else "narrow" if role == "narrow" else "outflow" if role == "outflow"
                   else "host" if kind == "host" else "feii" if tname == "FeII"
                   else "balmer" if tname in ("BalmerContinuum", "HighOrderBalmer") else "continuum" if kind == "continuum" else "other")
            if fam == "broad":
                broad_n[c.name] = len(broad_n)
            groups.append({"name": c.name, "family": fam, "n": broad_n.get(c.name, 0),
                           "lines": [plain_line(n).replace(" λ", " ") for n in getattr(c, "line_names", [])]})

        def short(pname):
            rest = pname.split(".", 1)[1] if "." in pname else pname
            parts = rest.rsplit(".", 1)
            return f"{plain_line(parts[0]).replace(' λ', ' ')} {parts[1]}" if len(parts) == 2 else rest

        fin = lambda v: float(v) if v is not None and np.isfinite(v) else None
        params = [{"name": plist[i].name, "comp": plist[i].name.split(".")[0], "short": short(plist[i].name),
                   "value": fin(r.full[i]), "err": fin(sig[i]), "unit": plist[i].unit or "",
                   "lim": status.get(plist[i].name, "") in ("at limit", "at bound") or str(status.get(plist[i].name, "")).startswith("lower limit")}
                  for i in free]
        out = {"k": k, "name": name, "source": src, "rchi2": float(r.rchi2), "groups": [g for g in groups if any(p["comp"] == g["name"] for p in params)],
               "params": params, "corr": [[round(float(v), 3) for v in row] for row in C],
               # free parameters at a limit have no error from the covariance, so they are not in the wheel
               "limits": [q.name for i, q in enumerate(plist) if q.free and i not in free
                          and str(status.get(q.name, "")).startswith(("at ", "lower limit"))]}
        if len(self._wf_cache) > 80:
            self._wf_cache.clear()
        self._wf_cache[key] = out
        return out

    def wf_feii_view(self, k=0):
        """
        Results › Fe II: the Fe II of fit ``k`` group by group, for the browser. The spectrum with
        everything but Fe II subtracted, the fitted Fe II of every group (lines sharing a lower term),
        every transition at its fitted strength, the band flux, EW and ratio of ``r.feii()``, the
        kinematics against the broad reference line (H-beta, or Mg II for the UV templates) and, for
        the optical template, the position on the quasar main sequence.
        """
        from ..lines import line_table
        from ..plotting import plain_line
        from ..wavelengths import vac_to_air
        res = self.wf_results()
        if not res:
            return None
        k = min(max(int(k), 0), len(res) - 1)
        name, _, r, _ = res[k]
        if r is None:
            return {"k": k, "name": name, "error": "this spectrum could not be fitted"}
        key = ("feii", id(r))
        if key in self._wf_cache:
            return self._wf_cache[key]
        fes = r._feii_components() if hasattr(r, "_feii_components") else []
        if not fes:
            return {"k": k, "name": name, "error": "this model has no Fe II component"}
        fe = fes[0]
        spec, ctx = r.spectrum, r.spectrum.context
        comps = {c.name: c for c in r.model.components}
        slices = dict(zip(comps, r.model._slices()))
        p = np.asarray(r.full[slices[fe.name]], float)
        fit = r.mask & spec.good
        lo, hi = float(spec.wave[fit].min()), float(spec.wave[fit].max())
        keep = (spec.wave >= lo) & (spec.wave <= hi)
        x = spec.wave[keep]
        parts = {n: v[keep] for n, v in r.model_flux(components=True).items()}
        model = sum(parts.values(), np.zeros_like(x))
        fe_obs = parts[fe.name]                                  # with any extinction applied
        fe_raw = fe.value(x, p, ctx)
        scale = np.divide(fe_obs, fe_raw, out=np.ones_like(x), where=np.abs(fe_raw) > 0)
        line_kinds = ("continuum", "host", "template")
        emis = sum((v for n, v in parts.items() if n != fe.name and getattr(comps[n], "kind", "") not in line_kinds
                    and type(comps[n]).__name__ not in ("BalmerContinuum",)), np.zeros_like(x))
        P = r.params.set_index("parameter")
        fin = lambda v: float(v) if v is not None and np.isfinite(v) else None   # noqa: E731
        num = lambda a: [None if not np.isfinite(v) else float(f"{v:.5g}") for v in np.asarray(a, float)]  # noqa: E731
        meta = r.feii(component=fe.name)
        blo, bhi = meta["band"]
        inband = (x > blo) & (x < bhi)
        groups = []
        for i, g in enumerate(fe.groups):
            q = p.copy()
            q[:fe._n_amp] = 0.0
            q[i] = p[i]
            f = fe.value(x, q, ctx) * scale
            w = fe._wave[fe._amp_index == i]
            pname = f"{fe.name}.{g}"
            groups.append({"name": g, "amp": fin(P.loc[pname, "value"]), "err": fin(P.loc[pname, "error"]),
                           "status": str(P.loc[pname, "status"]), "flux": num(f) if np.any(f > 0) else None, "n": int(w.size),
                           "wmin": float(w.min()), "wmax": float(w.max()),
                           "total": float(np.trapezoid(f, x)) if x.size > 1 else 0.0,
                           "band": float(np.trapezoid(f[inband], x[inband])) if inband.sum() > 1 else 0.0})
        C = 299792.458
        par = lambda n: (fin(P.loc[f"{fe.name}.{n}", "value"]), fin(P.loc[f"{fe.name}.{n}", "error"]))  # noqa: E731
        (shift, shift_err), (fwhm, fwhm_err) = par("shift"), par("fwhm")
        air = getattr(spec, "medium", "vacuum") == "air"
        atoms = _feii_levels(getattr(fe, "feii_kind", "optical"), len(fe._wave))
        lines = []
        for q, (w0, gi, fl) in enumerate(zip(fe._wave, fe._amp_index, fe.line_fluxes(p))):
            wc = w0 * (1 + (shift or 0) / C)
            wc = float(vac_to_air(wc)) if air else float(wc)
            sig = (fwhm or 2000) / 2.3548 / C * wc
            if lo - 20 < wc < hi + 20 and fl > 0:
                lines.append([round(wc, 2), int(gi), float(f"{fl / (np.sqrt(2 * np.pi) * sig):.4g}"), float(f"{fl:.4g}")]
                             + (atoms[q] if atoms else []))
        # the broad reference line in velocity, and one Fe II line of the fitted width beside it
        ref = meta["reference"]
        broad = [c.name for c in r._line_components() if c.role == "broad" and ref in c.line_names]
        refd = None
        if broad:
            w_ref = float(line_table().loc[ref].vacuum)
            if air:
                w_ref = float(vac_to_air(w_ref))
            # the intrinsic profile (no instrumental broadening, as its FWHM and the Fe II FWHM), on a fine grid
            v = np.arange(-12000.0, 12000.0 + 1, 20.0)
            ictx = {"medium": ctx.get("medium", "vacuum"), "resolution": 0.0}
            prof = sum(comps[n].line_profile(w_ref * (1 + v / C), r.full[slices[n]], ictx, ref) for n in broad)
            win = np.isfinite(prof)
            try:
                m = r.measure(ref, broad)
                rf, rfe = fin(m["fwhm"]), fin(m["fwhm_err"])
            except Exception:                         # noqa: BLE001 - a profile without a measurable FWHM
                rf = rfe = None
            if win.sum() > 3:
                refd = {"line": ref, "label": plain_line(ref).split(" λ")[0], "v": num(v[win]), "f": num(prof[win]),
                        "fwhm": rf, "fwhm_err": rfe}
        # the other emission lines of the model, to label where they sit
        seen, labels = set(), []
        for c in r._line_components():
            for ln, w0 in zip(c.line_names, np.atleast_1d(getattr(c, "_wave", []))):
                wl = float(vac_to_air(w0)) if air else float(w0)
                if ln not in seen and lo < wl < hi:
                    seen.add(ln)
                    labels.append([plain_line(ln).split(" λ")[0], round(wl, 1)])
        labels.sort(key=lambda t: t[1])
        out = {"k": k, "name": name, "z": fin(getattr(spec, "z", None)), "kind": getattr(fe, "feii_kind", "optical"),
               "component": fe.name, "flux_unit": spec.flux_unit, "rchi2": float(r.rchi2),
               "wave": num(x), "data": num(np.where(fit[keep], spec.flux[keep], np.nan)), "err": num(spec.err[keep]),
               "model": num(model), "other": num(model - fe_obs), "feii": num(fe_obs), "emis": num(emis),
               "groups": groups, "lines": lines, "labels": labels, "levels": bool(atoms),
               "kin": {"shift": shift, "shift_err": shift_err, "fwhm": fwhm, "fwhm_err": fwhm_err,
                       "shift_status": str(P.loc[f"{fe.name}.shift", "status"]),
                       "fwhm_status": str(P.loc[f"{fe.name}.fwhm", "status"])},
               "ref": refd,
               "strength": {"line": meta["line"], "ratio_label": meta["ratio_label"], "reference": ref,
                            "band": [blo, bhi], "covered": bool(meta["covered"]),
                            "r": fin(meta["r_feii"]), "r_err": fin(meta["r_feii_err"]),
                            "flux": fin(meta["flux"]), "flux_err": fin(meta["flux_err"]),
                            "ew": fin(meta["ew"]), "ew_err": fin(meta["ew_err"])}}
        if len(self._wf_cache) > 80:
            self._wf_cache.clear()
        self._wf_cache[key] = out
        return out

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
        out["flux_unit"] = r.spectrum.flux_unit
        out["plot"] = self._series(r)
        out["lines"] = self._line_rows(r)
        out["params"] = self._param_rows(r)
        out["continuum"] = self._param_rows(r, kinds=("continuum", "host", "template", "calibration",
                                                    "attenuation", "absorption"))
        out["fit"] = {"chi2": float(r.chi2), "dof": int(r.dof), "rchi2": float(r.rchi2),
                      "scale": float(getattr(r, "scale", 1.0) or 1.0)}
        out["corner_params"] = self.sampled_params(r)
        out["has_feii"] = bool(hasattr(r, "_feii_components") and r._feii_components())
        if len(self._wf_cache) > 80:
            self._wf_cache.clear()
        self._wf_cache[key] = out
        return out

    def _has_main_sequence(self):
        """Whether the fit itself (not the model now being edited) has Fe II and a broad H-beta."""
        r = next((e[2] for e in self.wf_results() or [] if e[2] is not None), None)
        if r is None or not hasattr(r, "_feii_components"):
            return False
        return any(r._feii_band(c)["reference"] == "Hb" for c in r._feii_components()) and any(
            c.role == "broad" and "Hb" in c.line_names for c in r._line_components())

    def wf_main_sequence(self):
        """Every epoch's position on the quasar main sequence (R_FeII, FWHM of broad H-beta) and the grid."""
        from ..plotting import MS_TYPES, main_sequence_grid
        res = self.wf_results()
        if not res:
            return None
        key = ("ms", tuple(id(r) for _, _, r, _ in res))
        if key not in self._wf_cache:
            from ..correlations import main_sequences
            pts = []
            usable = [(k, e) for k, e in enumerate(res) if e[2] is not None]
            for (k, (name, date, r, ref)), m in zip(usable, main_sequences([e[2] for _, e in usable])):
                if m is None:
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
            self._wf_cache[key] = self._result_quantities(r)
        return self._wf_cache[key]

    @staticmethod
    def _result_quantities(r):
        """Main results and mass calibrations for a fitted spectrum in either Results view."""
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
        main = bh[0]["line"] if bh else None
        first = {}
        for b in bh:
            first.setdefault(b["line"], b["scatter"])
        for row in rows:
            if row["name"].startswith("log M_BH"):
                line = row["name"][row["name"].index("(") + 1:-1] if "(" in row["name"] else main
                if line in first:
                    row["scatter"] = float(first[line])
        return {"key": rows, "bh": bh}

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

    # ------------------------------------------------------------ missing lines
    def _linescan(self):
        """fa.suggest_lines of the single fit, or of every epoch of a multi-epoch fit (milliseconds each;
        cached per fit). For epochs, one row per suggestion: the epoch where it lowers the BIC most, with
        ``epochs`` (where it is found, by index) and ``by_epoch`` ({index: flux}), the most frequent first."""
        from ..linescan import suggest_lines

        def scan(r):
            try:
                return suggest_lines(r)
            except Exception:       # a model without line components, an odd mask: nothing to suggest
                return None
        kind = self.wf.get("fit_kind")
        if kind in ("joint", "fits"):
            eps = [(k, e) for k, (_, _, e, _) in enumerate(self.wf_results() or []) if e is not None]
            if len(eps) < 2:
                return None
            key = ("linescan", tuple(id(e) for _, e in eps))
            if key not in self._wf_cache:
                self._wf_cache[key] = self._merge_scans({k: scan(e) for k, e in eps})
            return self._wf_cache[key]
        r = self.result
        if r is None or kind != "single":
            return None
        key = ("linescan", id(r))
        if key not in self._wf_cache:
            self._wf_cache[key] = scan(r)
        return self._wf_cache[key]

    @staticmethod
    def _merge_scans(tabs):
        """One table from fa.suggest_lines of each epoch ({index: table}): a row per line to add (with the component
        it is found with in most epochs, then where it helps most) and per line to separate from its component."""
        frames = [t.assign(epoch=k) for k, t in tabs.items() if t is not None and len(t)]
        if not frames:
            return next((t for t in tabs.values() if t is not None), None)
        allr = pd.concat(frames, ignore_index=True)
        rows = []
        allr["_key"] = [c if a == "split" else "" for a, c in zip(allr["action"], allr["component"])]
        for _, g in allr.groupby(["action", "line", "_key"], sort=False):
            count = g["component"].map(g["component"].value_counts())
            g = g[count == count.max()]                                  # the component of most epochs
            best = g.loc[g["dbic"].astype(float).idxmin()].copy()     # dbic < 0: the epoch where it helps most
            g = g[g["component"] == best["component"]]
            best["epochs"] = sorted(int(k) for k in g["epoch"])
            best["by_epoch"] = {int(k): float(f) for k, f in zip(g["epoch"], g["flux"])}
            rows.append(best.drop("_key"))
        out = pd.DataFrame(rows).reset_index(drop=True)
        out["_n"] = out["epochs"].map(len)
        out = out.sort_values(["_n", "dbic"], ascending=[False, True]).drop(columns="_n").reset_index(drop=True)
        return out

    def _line_homes(self):
        """{line: names of the components of the current model that have it}."""
        out = {}
        for c in (self.model.components if self.model is not None else []):
            for ln in getattr(c, "line_names", []) or []:
                out.setdefault(ln, set()).add(c.name)
        return out

    @staticmethod
    def _ls_done(d, homes):
        """A suggestion the current model has already taken: the line added, or moved out of its component."""
        if d["action"] == "split":
            return d["component"] not in homes.get(d["line"], set())
        return d["line"] in homes

    def wf_linescan(self):
        """Lines the single fit is missing, and lines to separate from their component, for the Results
        page (those the model has since taken are marked)."""
        t = self._linescan()
        if t is None:
            return None
        homes = self._line_homes()
        num = lambda v: None if v is None or not np.isfinite(v) else round(float(v), 1)
        rows = [{"action": d["action"], "line": d["line"], "lines": list(d["lines"]), "wave": round(float(d["wave"]), 1),
                 "ion": d["ion"], "type": d["type"], "component": d["component"], "role": d["role"],
                 "snr": num(d["snr"]), "dv": num(d["dv"]), "dbic": num(d["dbic"]), "added": self._ls_done(d, homes),
                 **({"epochs": list(d["epochs"]), "ep": int(d["epoch"])} if "epochs" in d else {})}
                for d in t.to_dict("records")]
        if self.wf.get("fit_kind") in ("joint", "fits"):
            return {"id": id(t), "rows": rows, "trial": None, "n_epochs": len(self.wf_results() or [])}
        trial = self._trial()
        return {"id": id(self.result), "rows": rows, "trial": trial[0] if trial else None}

    @staticmethod
    def _leftover(r, lo, hi, n=150):
        """data - model of fit ``r`` between ``lo`` and ``hi`` (Å, the frame of its spectrum), binned to at most
        about ``n`` points (mean, with its error); masked pixels are left out."""
        s = r.spectrum
        sel = (s.wave >= lo) & (s.wave <= hi) & np.asarray(r.mask, bool) & np.isfinite(s.flux) & np.isfinite(s.err) & (s.err > 0)
        if sel.sum() < 3:
            return None
        x, d, e = s.wave[sel], (s.flux - r.model_flux(s.wave))[sel], s.err[sel]
        b = max(1, int(np.ceil(x.size / n)))
        m = x.size // b * b or x.size
        if b > 1:
            x, d = x[:m].reshape(-1, b).mean(1), d[:m].reshape(-1, b).mean(1)
            e = np.sqrt((e[:m].reshape(-1, b) ** 2).sum(1)) / b
        rd = lambda v: [float(f"{t:.4g}") for t in v]
        return {"x": [round(float(t), 2) for t in x], "d": rd(d), "e": rd(e)}

    def _line_snip(self, r, d, kin):
        """The leftover flux around suggestion ``d`` of fa.suggest_lines in fit ``r`` (``kin``: {component: (fwhm,
        shift)}), with the line it would add, or where a line to separate sits now and where it would go."""
        from ..plotting import C_KMS
        from ..components import _in_medium
        fwhm, shift = kin.get(d["component"], (500.0, 0.0))
        lam = float(_in_medium(np.array([float(d["wave"])]), r.spectrum.context)[0])
        half = lam * max(3.0 * fwhm, 1500.0) / C_KMS
        snip = self._leftover(r, lam - half, lam + half)
        if snip is None:
            return None
        now = lam * (1 + shift / C_KMS)
        snip.update({"lam": round(lam, 2), "now": round(now, 2)})
        if d["action"] == "split" and d.get("dv") is not None and np.isfinite(d["dv"]):
            snip["to"] = round(lam * (1 + (shift + float(d["dv"])) / C_KMS), 2)
        elif d["action"] != "split" and np.isfinite(d.get("flux", np.nan)):
            sig = now * fwhm / C_KMS / 2.3548
            xf = np.linspace(lam - half, lam + half, 160)
            g = float(d["flux"]) / (sig * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((xf - now) / sig) ** 2)
            snip["add"] = {"x": [round(float(v), 2) for v in xf], "y": [float(f"{v:.4g}") for v in g]}
        return snip

    def wf_sample_checks(self, k=0):
        """Results › Checks of spectrum ``k`` of a sample: fa.analyze_residuals and fa.suggest_lines of its fit,
        each row with the leftover flux around it and the code that applies it (computed on demand, cached)."""
        from ..linescan import suggest_lines
        from ..residuals import analyze_residuals
        r = self._sample_fit(k)
        k = min(max(int(k), 0), len(self.sample.table) - 1)
        key = ("sample_checks", id(r))
        if key in self._wf_cache:
            return self._wf_cache[key]
        num = lambda v: float(v) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else None
        out = {"k": k, "name": r.spectrum.name, "lines": [], "resid": [], "rchi2": num(r.rchi2)}
        kin = {}
        for d in self._lines_table(r).to_dict("records"):
            if d.get("component") not in kin and np.isfinite(d.get("fwhm", np.nan)):
                kin[d["component"]] = (float(d["fwhm"]), float(d.get("shift", 0.0) or 0.0))
        try:
            t = suggest_lines(r)
        except Exception:       # a model without line components: nothing to suggest
            t = None
        for d in (t.to_dict("records") if t is not None else []):
            out["lines"].append({"action": d["action"], "line": d["line"], "lines": list(d["lines"]),
                                 "wave": round(float(d["wave"]), 1), "ion": d["ion"], "type": d["type"],
                                 "component": d["component"], "role": d["role"], "snr": num(d["snr"]),
                                 "dv": num(d["dv"]), "dbic": num(d["dbic"]), "code": d.get("code") or "",
                                 "snip": self._line_snip(r, d, kin)})
        try:
            res = analyze_residuals(r)
        except Exception as exc:                # noqa: BLE001
            res, out["resid_error"] = None, f"{type(exc).__name__}: {exc}"
        for i, d in enumerate(res.to_dict("records") if res is not None else []):
            lo, hi = num(d["lo"]), num(d["hi"])
            snip = None
            if lo is not None and hi is not None:
                pad = max(1.5 * (hi - lo), 15.0)
                snip = self._leftover(r, lo - pad, hi + pad)
                if snip:
                    snip.update({"lo": lo, "hi": hi})
            out["resid"].append({"i": i, "center": num(d["center"]), "lo": lo, "hi": hi, "kind": d["kind"],
                                 "sigma": num(d["sigma"]), "fwhm": num(d["fwhm"]), "line": d.get("line") or "",
                                 "in_model": bool(d.get("in_model")), "suggestion": d["suggestion"],
                                 "code": d.get("code") or "", "snip": snip})
        self._wf_cache[key] = out
        return out

    def wf_sample_checks_code(self, k=0, lines=(), resid=()):
        """Code that refits spectrum ``k`` of a sample with the chosen rows of its checks (indices into the lines and
        the residual features of wf_sample_checks): the lines to add in one tied component per kinematic source
        (as fa.suggest_lines combines them), the lines to separate, then the residual suggestions not covered."""
        from ..linescan import _code
        D = self.wf_sample_checks(k)
        r = self._sample_fit(k)
        rows = [D["lines"][int(i)] for i in lines if 0 <= int(i) < len(D["lines"])]
        adds = [d for d in rows if d["action"] == "add"]
        parts = ([_code(adds, r.model)] if adds else []) + [d["code"] for d in rows if d["action"] != "add" and d["code"]]
        covered = {ln for d in rows for ln in d["lines"]}
        for i in resid:
            if 0 <= int(i) < len(D["resid"]):
                d = D["resid"][int(i)]
                if d["code"] and d["line"] not in covered and d["code"] not in parts:
                    parts.append(d["code"])
                    covered.add(d["line"])
        k = D["k"]
        return "\n".join([f"r = sample.load({k})                     # {D['name']}", "model = r.model.copy()"]
                         + (parts or ["# choose changes above"])
                         + [f"r{k} = fa.fit(r.spectrum, model)",
                            f"print(r{k}.rchi2, r{k}.bic - r.bic)   # ΔBIC < -10: keep the change"])

    def wf_check_snips(self, k=0):
        """For Results › Checks: the leftover flux (data - model) around each suggestion of fa.suggest_lines
        (with the line it would add: a Gaussian of the flux found, at the width and shift of the component
        it takes them from; for a line to separate, where it sits now and where it would go) and around each
        feature of the residual analysis of the shown fit."""
        out = {"lines": [], "resid": []}
        t, r = self._linescan(), self.result
        multi = t is not None and "epochs" in t.columns
        if t is not None and (r is not None or multi):
            key = ("check_snips", id(r), id(t), int(k) if multi else 0)
            if key in self._wf_cache:
                out["lines"] = self._wf_cache[key]
            else:
                kins = {}

                def kinematics(r):
                    if id(r) not in kins:
                        kin = kins[id(r)] = {}
                        for d in self._lines_table(r).to_dict("records"):
                            if d.get("component") not in kin and np.isfinite(d.get("fwhm", np.nan)):
                                kin[d["component"]] = (float(d["fwhm"]), float(d.get("shift", 0.0) or 0.0))
                    return kins[id(r)]
                rows = []
                for d in t.to_dict("records"):
                    ep = None
                    if multi:             # the shown epoch, or the one where the line helps most when it is not found there
                        ep = int(k) if int(k) in d["epochs"] else int(d["epoch"])
                        r = self._wf_fit(ep)[2]
                        d = dict(d, flux=d["by_epoch"].get(ep, d["flux"]))
                    snip = self._line_snip(r, d, kinematics(r))
                    if snip is not None and ep is not None:
                        snip["ep"] = ep
                    rows.append(snip)
                self._wf_cache[key] = out["lines"] = rows
        Rz = self.wf_residuals()
        if Rz and Rz["rows"]:
            k = int(k)
            key = ("resid_snips", k, Rz["id"], id(self.wf_results()))
            if key in self._wf_cache:
                out["resid"] = self._wf_cache[key]
            else:
                rows = []
                for d in Rz["rows"]:
                    # the shown epoch, or the first epoch the feature was found in when it is not there
                    ep = k if not d["epochs"] or k in d["epochs"] else d["epochs"][0]
                    try:
                        _, _, rk, _ = self._wf_fit(ep)
                    except ValueError:
                        rk = None
                    if rk is None or d["lo"] is None or d["hi"] is None:
                        rows.append(None)
                        continue
                    pad = max(1.5 * (d["hi"] - d["lo"]), 15.0)
                    snip = self._leftover(rk, d["lo"] - pad, d["hi"] + pad)
                    if snip:
                        snip.update({"lo": d["lo"], "hi": d["hi"], "ep": ep})
                    rows.append(snip)
                self._wf_cache[key] = out["resid"] = rows
        out["id"] = [id(t) if multi else id(r) if r is not None else None, Rz["id"] if Rz else None]
        return out

    def wf_addlines_code(self, lines, refit=False):
        """Code that applies suggestions of fa.suggest_lines: adds the missing lines, tied to the kinematics
        they were found with, and separates offset lines from their component (Model.split_lines);
        ``refit``: and fits again in the same cell, with the options of the last fit."""
        from ..linescan import _code, _split_name
        t = self._linescan()
        if t is None:
            raise ValueError("fit one spectrum first: the lines are searched in its residuals")
        if self.model is None:
            raise ValueError("there is no model to add the lines to")
        homes = self._line_homes()
        comps = {c.name for c in self.model.components}
        want = set(str(x) for x in lines or [])
        rows = [d for d in t.to_dict("records") if d["line"] in want and not self._ls_done(d, homes)]
        if not rows:
            raise ValueError("these suggestions are already in the model")
        lost = sorted({d["component"] for d in rows} - comps)
        if lost:
            raise ValueError(f"the model no longer has {', '.join(lost)}: fit again and search again")
        adds = [d for d in rows if d["action"] == "add"]
        parts = [_code(adds, self.model)] if adds else []
        for d in rows:
            if d["action"] == "split":
                group = [ln for ln in d["lines"] if d["component"] in homes.get(ln, set())]
                parts.append(f"model.split_lines({d['component']!r}, {group!r}, "
                             f"{_split_name(self.model, d['component'], d['line'])!r})")
        change = "\n".join(parts)
        multi = "epochs" in t.columns
        found = "fa.suggest_lines on each epoch" if multi else "fa.suggest_lines(r)"
        if not refit:
            return f"# add lines found by {found}\n" + change + "\nmodel"
        # the model is edited by the time the fit runs: no preset-specific host check
        self.__dict__["_preset_made"] = None
        opts = dict(self.wf.get("fit_opts") or {})
        trial = None if multi else self._trial()   # a trial fit of one epoch is refitted, not all epochs
        if trial is not None:
            opts["trial"] = trial[0]
        fit = self.wf_fit_code(**opts)
        fit = fit.split("\n", 1)[1] if fit.startswith(("# fit\n", "# trial fit\n")) else fit
        stale = "resid = None\n" if self.ns.get("resid") is not None else ""
        return f"# add lines and refit (found by {found})\n" + change + "\n" + stale + fit

    # ------------------------------------------------------------ robustness
    @staticmethod
    def _plain(name):
        """A parameter in words: 'feii.fwhm' -> 'Fe II width'."""
        comp, _, rest = name.partition(".")
        comp = {"feii": "Fe II", "heii": "He", "continuum": "continuum"}.get(comp, re.sub(r"(\D)(\d)", r"\1 \2", comp))
        rest = {"fwhm": "width", "shift": "velocity shift", "index": "slope", "amp": "level"}.get(rest, rest.replace(".flux", " flux"))
        return f"{comp} {rest}"

    def wf_robust_code(self):
        """Code that scans the parameters of the single fit (fa.robustness)."""
        if self.result is None or self.wf.get("fit_kind") != "single":
            raise ValueError("this check needs the fit of one spectrum")
        return ("# robustness\nrob = fa.robustness(r)   # refit with each width, shift and ratio fixed across its limits\n"
                "print(rob.summary())")

    def wf_robust_apply_code(self):
        """Code that starts the model from the better fit the scans found."""
        from ..robustness import Robustness
        rob = self.ns.get("rob")
        if not isinstance(rob, Robustness) or not rob.improved:
            raise ValueError("the scans did not find a better fit")
        return "model = rob.best_model()   # the better solution, all scanned parameters free\nmodel"

    @staticmethod
    def _pretty(text):
        if text.startswith("FeII") and text.endswith("/MgII2798"):
            from ..correlations import plain_label
            return plain_label(text)
        for a, b in (("[OIII]5007", "[O III] \u03bb5007"), ("R_FeII", "R(Fe II)"), ("Hb", "H\u03b2"), ("Ha", "H\u03b1"), ("MgII2798", "Mg II"), ("CIV1549", "C IV")):
            text = text.replace(a, b)
        return text

    def wf_robust(self):
        """The last robustness check of the single fit, for the Results page."""
        from ..robustness import Robustness
        r, rob = self.result, self.ns.get("rob")
        if r is None or self.wf.get("fit_kind") != "single":
            return None
        if not isinstance(rob, Robustness) or getattr(rob, "result", None) is not r:
            return {"id": 0}
        key = ("robust", id(rob))
        if key in self._wf_cache:
            return self._wf_cache[key]
        num = lambda v: float(v) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else None
        unit = {p.name: p.unit for p in r.model.params}
        qunit = lambda q: "km/s" if q.split()[-1] in ("FWHM", "sigma", "shift") else ""
        def qscan(name):
            s = rob.profiles.get("result: " + name)
            if s is None:
                return []
            s = s.sort_values("value")
            return [[num(v), num(c)] for v, c in zip(s.value, s.dchi2)]
        quants = [{"label": self._pretty(d["quantity"]), "unit": qunit(d["quantity"]),
                   **{k: num(d[k]) for k in ("value", "error", "lo68", "hi68", "lo95", "hi95")}, "flag": d["flag"],
                   "scan": qscan(d["quantity"])}
                  for d in rob.quantities.to_dict("records")]
        params = []
        for d in rob.parameters.to_dict("records"):
            p, s = r.model.params[d["parameter"]], rob.profiles[d["parameter"]]
            params.append({"name": d["parameter"], "label": self._plain(d["parameter"]), "unit": unit.get(d["parameter"], ""),
                           **{k: num(d[k]) for k in ("value", "error", "lo68", "hi68", "lo95", "hi95")},
                           "flag": d["flag"], "note": d["note"], "min": num(p.min), "max": num(p.max),
                           "scan": [[num(v), num(c)] for v, c in zip(s.value, s.dchi2)]})
        alts = [{"i": i, "label": self._plain(a["parameter"]), "name": a["parameter"], "value": num(a["value"]), "dchi2": num(a["dchi2"])}
                for i, a in enumerate(rob.alternatives)]
        out = {"id": id(rob), "n_fits": rob.n_fits, "improved": bool(rob.improved), "alternatives": alts,
               "chi2": [num(r.chi2), num(rob.best.chi2)], "quantities": quants, "params": params}
        if len(self._wf_cache) > 50:
            self._wf_cache.clear()
        self._wf_cache[key] = out
        return out

    # ------------------------------------------------------ simulation check
    def wf_simcheck_code(self, n=60):
        """Code that draws spectra from the single fit and fits them again (fa.simulation_check)."""
        if self.wf.get("fit_kind") == "joint" and self.multi is not None:
            n = int(n)
            if not 4 <= n <= 200:
                raise ValueError("the number of simulations is between 4 and 200")
            return ("# simulation check\n"
                    f"sim = fa.simulation_check(me, n={n})   # sets of epochs drawn from the joint fit, fitted jointly again\n"
                    "print(sim.summary())")
        if self.result is None or self.wf.get("fit_kind") != "single":
            raise ValueError("this check needs the fit of one spectrum")
        n = int(n)
        if not 10 <= n <= 500:
            raise ValueError("the number of simulations is between 10 and 500")
        return ("# simulation check\n"
                f"sim = fa.simulation_check(r, n={n})   # spectra drawn from the fit, fitted again as the spectrum was\n"
                "print(sim.summary())")

    def wf_simcheck(self):
        """The last simulation check of the single fit, for the Results page."""
        from ..simcheck import SimulationCheck
        r, sim = self.result, self.ns.get("sim")
        if r is None or self.wf.get("fit_kind") != "single":
            return None
        if not isinstance(sim, SimulationCheck) or sim.result is not r:
            return {"id": 0}
        key = ("simcheck", id(sim))
        if key in self._wf_cache:
            return self._wf_cache[key]
        num = lambda v: float(v) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else None
        unit = lambda q: "km/s" if q.split()[-1] in ("FWHM", "sigma", "shift") else ""
        rows = [{"label": self._pretty(d["quantity"]), "unit": unit(d["quantity"]), "flag": d["flag"],
                 **{k: num(d[k]) for k in ("value", "mean", "bias", "bias_sd", "bias_z", "scatter", "error",
                                           "cover68", "cover95")}}
                for d in sim.table.to_dict("records")]
        out = {"id": id(sim), "n": int(sim.n), "failed": int(sim.failed), "pruned": num(sim.pruned), "rows": rows,
               "rchi2": num(np.median(sim.rchi2))}
        if len(self._wf_cache) > 50:
            self._wf_cache.clear()
        self._wf_cache[key] = out
        return out

    def wf_tie_check(self, component):
        """
        Would tying the shift and width of ``component`` to another component change the results? From
        the last single fit, where both are free (fa.model_confidence, no new fit): per possible target,
        the probability that the main broad line (FWHM and flux) moves by less than 5%.
        """
        from ..confidence import model_confidence
        r = self.result
        if r is None or self.wf.get("fit_kind") != "single" or r.cov is None:
            return {"available": False, "why": "needs a fit of one spectrum with covariance errors"}
        params = r.model.params
        comps = [c.name for c in r.model.components]
        if component not in comps:
            return {"available": False, "why": f"{component} is not in the last fit"}
        key = ("tiecheck", id(r), component)
        if key in self._wf_cache:
            return self._wf_cache[key]
        free = lambda n: n in params.names and params[n].free and not params[n].tie
        broad = next((c for c in r.model.components if getattr(c, "role", None) == "broad"
                      and free(f"{c.name}.fwhm")), None)
        if broad is None:
            return {"available": False, "why": "the fit has no broad line with a free width to judge by"}
        line = next((n for n in broad.line_names if n in ("Hb", "Ha", "MgII2798", "CIV1549")), broad.line_names[0])
        targets = {}
        for name in (f"{broad.name}.fwhm", f"{broad.name}.{line}.flux"):
            if name in params.names:
                v = float(r.full[params.index(name)])
                if v > 0:
                    targets[name] = 0.05 * v
        rows = []
        for other in comps:
            if other == component:
                continue
            ties = [(f"{component}.{k}", f"{other}.{k}") for k in ("fwhm", "shift")
                    if free(f"{component}.{k}") and free(f"{other}.{k}")]
            if not ties:
                continue
            try:
                conf = model_confidence(r, ties, targets, n_draws=4000)
                p = float(conf.probability)
                rows.append({"to": other, "verdict": conf.verdict, "probability": p if np.isfinite(p) else None,
                             "test_p": float(conf.test_p) if np.isfinite(conf.test_p) else None})
            except Exception as exc:               # noqa: BLE001  (no error for a width at its limit, ...)
                rows.append({"to": other, "verdict": "cannot assess", "probability": None, "test_p": None,
                             "why": str(exc)})
        out = {"available": True, "component": component, "fit": id(r), "rows": rows,
               "targets": f"{self._pretty(broad.name)} {self._pretty(line)} FWHM and flux within 5%"}
        if len(self._wf_cache) > 50:
            self._wf_cache.clear()
        self._wf_cache[key] = out
        return out

    def wf_calcheck_code(self):
        """Code that fits every epoch of the joint fit on its own with the joint flux scales (fa.calibration_check)."""
        if self.wf.get("fit_kind") != "joint" or self.multi is None:
            raise ValueError("this check needs a joint fit of the epochs")
        if not self.multi._prob.scale_names:
            raise ValueError("the epochs were not intercalibrated: there is no flux scale to check")
        return ("# calibration check\n"
                "chk = fa.calibration_check(me)   # each epoch on its own, with the flux scale of the joint fit\n"
                "print(chk.summary())")

    def wf_joint_checks(self):
        """The calibration and simulation checks of the joint fit, for the Results page (None: no joint fit)."""
        from ..jointcheck import CalibrationCheck, JointSimulationCheck
        me = self.multi
        if me is None or self.wf.get("fit_kind") != "joint":
            return None
        num = lambda v: float(v) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else None
        out = {"intercalibrated": bool(me._prob.scale_names), "cal": {"id": 0}, "sim": {"id": 0},
               "n_epochs": len(me.epochs)}
        chk = self.ns.get("chk")
        if isinstance(chk, CalibrationCheck) and chk.result is me:
            bad = chk.epochs[np.abs(chk.epochs.pull) > 3]
            out["cal"] = {"id": id(chk), "ok": chk.ok, "excess": num(chk.excess), "failed": len(chk.failed),
                          "calibration_error": num(chk.calibration_error),
                          "rows": [{"label": self._pretty(d["quantity"]).replace(".", " "), **{k: num(d[k]) for k in
                                    ("mean", "snr", "rchi2", "p", "scatter", "error", "excess")}, "n": int(d["n"]),
                                    "flag": d["flag"]} for d in chk.table.to_dict("records")],
                          "outliers": [{"epoch": int(e.epoch), "name": str(e.name), "line": self._pretty(e.line).replace(".", " "), "pull": num(e.pull)}
                                       for _, e in bad.sort_values("pull", key=np.abs, ascending=False).head(12).iterrows()],
                          "n_outliers": int(len(bad))}
        sim = self.ns.get("sim")
        if isinstance(sim, JointSimulationCheck) and sim.result is me:
            out["sim"] = {"id": id(sim), "n": int(sim.n), "failed": int(sim.failed),
                          "rows": [{"label": self._pretty(d["quantity"]).replace(".", " "),
                                    "flag": d["flag"], **{k: num(d[k]) for k in ("bias_sd", "bias_z", "spread", "cover68")},
                                    "n": int(d["n"])} for d in sim.table.to_dict("records")]}
        return out

    # ------------------------------------------------------------ weak Fe II
    def _feii_fit(self):
        """The single fit if its model has Fe II, else None."""
        r = self.result
        if r is None or self.wf.get("fit_kind") != "single" or not hasattr(r, "_feii_components"):
            return None
        return r if r._feii_components() else None

    def _feii_before(self, r):
        """The fit that ``r`` was pruned from (kept as r_feii_all), if it is still that one."""
        from ..result import FitResult
        prev = self.ns.get("r_feii_all")
        ok = (getattr(r, "feii_pruned", None) and isinstance(prev, FitResult) and prev is not r
              and prev.spectrum is r.spectrum)
        return prev if ok else None

    def _feii_epochs(self):
        """The epochs fitted one by one, if their model has Fe II, else None."""
        fits = self.fits
        if fits is None or self.wf.get("fit_kind") != "fits":
            return None
        ok = [f for f in fits if hasattr(f, "_feii_components")]
        return fits if ok and ok[0]._feii_components() else None

    def _feii_epochs_before(self, fits):
        """The fits the epochs were pruned from (kept as fits_feii_all), if they are still those."""
        prev = self.ns.get("fits_feii_all")
        if not isinstance(prev, list) or prev is fits or len(prev) != len(fits):
            return None
        if not any(getattr(f, "feii_pruned", None) for f in fits):
            return None
        same = all(getattr(a, "spectrum", None) is getattr(b, "spectrum", None) for a, b in zip(prev, fits))
        return prev if same else None

    def wf_prune_feii_code(self, nsigma=2.0):
        """Code that fits again without the Fe II groups the data do not detect (fa.prune_feii)."""
        nsigma = float(nsigma)
        if not 0 < nsigma <= 10:
            raise ValueError("the threshold is a number of sigma between 0 and 10")
        arg = "" if nsigma == 2.0 else f", nsigma={nsigma:g}"
        fits = self._feii_epochs()
        if fits is not None:
            if not any(getattr(f, "cov", None) is not None or getattr(f, "samples", None) is not None for f in fits):
                raise ValueError("the fits have no errors: fit with errors first (Fit step)")
            # always from the fits with every group; the epochs together again for the light curves
            return ("# prune Fe II\n"
                    "fits_feii_all = fits_feii_all if any(getattr(f, 'feii_pruned', None) for f in fits) else fits"
                    "   # the fits with every Fe II group\n"
                    f"fits = fa.prune_feii(fits_feii_all{arg})   # every epoch: groups below {nsigma:g} sigma fixed at 0, fitted again\n"
                    "me = fa.EpochSet(fits) if sum(hasattr(f, 'rchi2') for f in fits) > 1 else None\n"
                    "[(f.spectrum.name, len(f.feii_pruned)) for f in fits if hasattr(f, 'feii_pruned')]")
        r = self._feii_fit()
        if r is None:
            raise ValueError("this step needs the fit of one spectrum, or of epochs fitted one by one, with Fe II in the model")
        if r.cov is None and r.samples is None:
            raise ValueError("the fit has no errors: fit with errors first (Fit step)")
        if getattr(r, "feii_pruned", None) and self._feii_before(r) is None:
            # a pruned fit whose full fit is not in the session (e.g. opened from a file): prune it further
            return ("# prune Fe II\n"
                    f"r = fa.prune_feii(r{arg})   # groups below {nsigma:g} sigma fixed at 0, fitted again\n"
                    "r.feii_pruning")
        # always from the fit with every group, also when the threshold is changed after pruning
        return ("# prune Fe II\n"
                "r_feii_all = r_feii_all if getattr(r, 'feii_pruned', None) else r   # the fit with every Fe II group\n"
                f"r = fa.prune_feii(r_feii_all{arg})   # groups below {nsigma:g} sigma fixed at 0, fitted again\n"
                "r.feii_pruning")

    def wf_prune_feii_undo_code(self):
        fits = self._feii_epochs()
        if fits is not None:
            if self._feii_epochs_before(fits) is None:
                raise ValueError("there are no fits with every Fe II group to go back to")
            return ("fits = fits_feii_all   # back to the fits with every Fe II group\n"
                    "me = fa.EpochSet(fits) if sum(hasattr(f, 'rchi2') for f in fits) > 1 else None\n"
                    "len(fits)")
        r = self._feii_fit()
        if r is None or self._feii_before(r) is None:
            raise ValueError("there is no fit with every Fe II group to go back to")
        return "r = r_feii_all   # back to the fit with every Fe II group\nr"

    def wf_feii(self):
        """Fe II groups of the single fit: detected or not, and what pruning changed (for the Results page)."""
        from ..refine import feii_detections
        fits = self._feii_epochs()
        if fits is not None:
            return self._wf_feii_epochs(fits)
        r = self._feii_fit()
        if r is None:
            return None
        before = self._feii_before(r)
        key = ("feii", id(r), id(before))
        if key in self._wf_cache:
            return self._wf_cache[key]
        num = lambda v: float(v) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else None
        errors = r.cov is not None or r.samples is not None
        pruned = bool(getattr(r, "feii_pruned", None))
        table = r.feii_pruning if pruned else feii_detections(r) if errors else None
        rows = [] if table is None else [{"name": d["multiplet"], "label": d["multiplet"].split(".", 1)[1],
                                           "amplitude": num(d["amplitude"]), "error": num(d["error"]),
                                           "kept": bool(d["kept"])} for d in table.to_dict("records")]

        def numbers(x):
            out = {"bic": num(x.bic), "rchi2": num(x.rchi2)}
            try:
                f = x.feii()
                out.update(flux=num(f["flux"]), flux_err=num(f["flux_err"]), r_feii=num(f["r_feii"]),
                           r_feii_err=num(f["r_feii_err"]), reference=f["reference"],
                           ratio_label=self._pretty(f["ratio_label"]), band_label=f["line"])
                broad = [c.name for c in x._line_components() if c.role == "broad" and f["reference"] in c.line_names]
                if broad:
                    m = x.measure(f["reference"], broad)
                    out.update(line_flux=num(m["flux"]), line_flux_err=num(m["flux_err"]),
                               line_fwhm=num(m["fwhm"]), line_fwhm_err=num(m["fwhm_err"]))
                out["continuum"] = num(x.continuum(5100.0 if f["band"][0] > 3500 else 3000.0)[0])
            except Exception:                                             # noqa: BLE001
                pass
            return out

        now = numbers(r)
        out = {"id": id(r), "errors": bool(errors), "pruned": pruned, "rows": rows,
               "weak": sum(not d["kept"] for d in rows), "now": now,
               "before": numbers(before) if before is not None else None,
               "line": self._pretty(now.get("reference") or "")}
        if len(self._wf_cache) > 50:
            self._wf_cache.clear()
        self._wf_cache[key] = out
        return out

    def _wf_feii_epochs(self, fits):
        """Fe II groups of the epochs fitted one by one: how many each has, and how many are not detected."""
        from ..refine import feii_detections
        before = self._feii_epochs_before(fits)
        key = ("feii-epochs", id(fits), id(before))
        if key in self._wf_cache:
            return self._wf_cache[key]
        pruned = before is not None or any(getattr(f, "feii_pruned", None) for f in fits)
        rows, errors = [], False
        for k, f in enumerate(fits):
            if not hasattr(f, "_feii_components"):
                continue
            has = f.cov is not None or f.samples is not None
            errors = errors or has
            table = getattr(f, "feii_pruning", None) if pruned else None
            if table is None and has:
                table = feii_detections(f)
            rows.append({"k": k, "name": f.spectrum.name, "date": mjd_to_date(f.spectrum.mjd),
                         "groups": None if table is None else int(len(table)),
                         "weak": None if table is None else int((~table.kept).sum())})
        out = {"id": id(fits), "kind": "fits", "errors": bool(errors), "pruned": bool(pruned),
               "can_undo": before is not None, "rows": rows,
               "weak": sum(r["weak"] or 0 for r in rows), "groups": sum(r["groups"] or 0 for r in rows)}
        if len(self._wf_cache) > 50:
            self._wf_cache.clear()
        self._wf_cache[key] = out
        return out

    # ------------------------------------------------------- pin and compare
    def wf_pin_code(self, label):
        """Code that keeps the current single fit under ``label`` to compare later fits with."""
        if self.result is None or self.wf.get("fit_kind") != "single":
            raise ValueError("pin needs the fit of one spectrum")
        label = str(label or "").strip()[:60] or f"fit {len(self.ns.get('pins') or {}) + 1}"
        return f"pins = {{**globals().get('pins', {{}}), {label!r}: r}}   # keep this fit to compare later fits with\nlist(pins)"

    def wf_pin_alternative_code(self, i):
        """Code that pins an alternative solution found by the robustness scans."""
        from ..robustness import Robustness
        rob = self.ns.get("rob")
        if not isinstance(rob, Robustness) or not 0 <= int(i) < len(rob.alternatives):
            raise ValueError("no such alternative solution")
        a = rob.alternatives[int(i)]
        label = f"alternative: {a['parameter']} = {a['value']:.4g}"
        return (f"pins = {{**globals().get('pins', {{}}), {label!r}: rob.alternatives[{int(i)}]['result']}}"
                "   # another solution found by the scans\nlist(pins)")

    def wf_unpin_code(self, label):
        if label not in (self.ns.get("pins") or {}):
            raise ValueError(f"no pinned fit {label!r}")
        return f"pins = {{k: v for k, v in pins.items() if k != {label!r}}}\nlist(pins)"

    def wf_compare_code(self, label):
        """Code that compares a pinned fit with the current one."""
        if label not in (self.ns.get("pins") or {}):
            raise ValueError(f"no pinned fit {label!r}")
        if self.result is None or self.wf.get("fit_kind") != "single":
            raise ValueError("compare needs a current fit of one spectrum")
        return (f"# compare fits\ncmp = fa.compare_fits(pins[{label!r}], r, labels=({label!r}, \"current\"))\n"
                "print(cmp.summary())\ncmp.table.round(4)")

    def wf_compare(self):
        """Pinned fits and the last comparison with the current fit, for the Results page."""
        from ..fitcompare import FitComparison
        from ..result import FitResult
        r = self.result
        pins = {k: v for k, v in (self.ns.get("pins") or {}).items() if isinstance(v, FitResult)}
        if r is None or self.wf.get("fit_kind") != "single":
            return None
        cmp = self.ns.get("cmp")
        cmp = cmp if isinstance(cmp, FitComparison) and cmp.b is r else None
        key = ("compare", id(r), tuple((k, id(v)) for k, v in pins.items()), id(cmp))
        if key in self._wf_cache:
            return self._wf_cache[key]
        num = lambda v: float(v) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else None
        same = lambda p: p.spectrum.wave.size == r.spectrum.wave.size and np.allclose(p.spectrum.wave, r.spectrum.wave)
        out = {"pins": [{"label": k, "chi2": num(v.chi2), "rchi2": num(v.rchi2), "current": v is r, "same": bool(same(v)),
                         "n_free": sum(1 for p in v.model.params if p.free),
                         "components": ", ".join(c.name for c in v.model.components)} for k, v in pins.items()],
               "cmp": None}
        if cmp is not None:
            la, lb = cmp.labels
            rows = []
            for d in cmp.table.to_dict("records"):
                rows.append({"label": self._pretty(d["quantity"]), "a": num(d[la]), "ea": num(d[la + " error"]),
                             "b": num(d[lb]), "eb": num(d[lb + " error"]), "diff": num(d["difference"]),
                             "rel": num(d["relative"]), "sig": num(d["in errors"]),
                             "unit": "km/s" if d["quantity"].split()[-1] in ("FWHM", "sigma", "shift") else ""})
            s = cmp.stats
            out["cmp"] = {"id": id(cmp), "labels": [la, lb], "line": cmp.line, "rows": rows, "warnings": list(cmp.warnings),
                          "stats": {"chi2": [num(v) for v in s["chi2"]], "dchi2": num(s["dchi2"]), "n_pixels": s["n_pixels"],
                                    "n_free": s["n_free"], "rms": num(s["rms_diff"])}}
        if len(self._wf_cache) > 50:
            self._wf_cache.clear()
        self._wf_cache[key] = out
        return out

    def wf_compare_figure(self, fmt="png"):
        """The comparison figure (both fits, the change of each component, both broad profiles)."""
        from ..fitcompare import FitComparison
        cmp = self.ns.get("cmp")
        if not isinstance(cmp, FitComparison):
            return None
        import io
        from matplotlib.figure import Figure
        from ..plotting import paper_style
        with paper_style():
            fig = cmp.plot(fig=Figure(figsize=(10, 5.2)))
            buf = io.BytesIO()
            fig.savefig(buf, format=fmt, dpi=150, bbox_inches="tight")
        return buf.getvalue()

    # ------------------------------------------------- correlations over epochs
    def _epochs_name(self):
        kind = self.wf.get("fit_kind")
        if kind == "joint" and self.multi is not None:
            return "me"
        if kind == "fits" and self.ns.get("fits") is not None:
            return "fits"
        raise ValueError("correlations need a fit of several epochs")

    def wf_correlations_code(self):
        """Code that measures every epoch (fa.epoch_table)."""
        name = self._epochs_name()
        self._ep_source = id(self.multi if name == "me" else self.ns.get("fits"))
        return f"# correlations\nep = fa.epoch_table({name})   # broad lines, continuum, R_FeII, [OIII] per epoch\nep.shape"

    def _epoch_table(self):
        """The measured epochs of the current fit (None if they are not measured or the fit has changed since)."""
        import pandas as pd
        kind = self.wf.get("fit_kind")
        source = self.multi if kind == "joint" else self.ns.get("fits") if kind == "fits" else None
        ep = self.ns.get("ep")
        if source is None or getattr(self, "_ep_source", None) != id(source):
            return None
        return ep if isinstance(ep, pd.DataFrame) and "mjd" in ep else None

    def _epoch_view(self, lag=0.0, off=()):
        """The epoch table with the continuum taken ``lag`` days earlier and the epochs ``off`` left out."""
        from ..correlations import shift_continuum
        ep = self._epoch_table()
        if ep is None:
            return None
        lag = float(lag or 0.0)
        out = shift_continuum(ep, lag) if lag else ep
        return out.drop(index=[k for k in off if k in out.index]).reset_index(drop=True) if off else out

    def wf_correlations(self):
        """For the state: whether the epochs are measured (None: not a fit of at least three epochs)."""
        res = self.wf_results() or []
        if self.wf.get("fit_kind") not in ("joint", "fits") or len(res) < 3:
            return None
        ep = self._epoch_table()
        return {"id": id(ep) if ep is not None else 0, "n_epochs": len(res)}

    def wf_correlations_data(self):
        """The epoch table for the browser, which computes the correlations and fits itself (sent once)."""
        from ..correlations import _is_feii_ratio, _use_log, default_columns, plain_label, quantity_columns
        ep = self._epoch_table()
        if ep is None:
            return None
        held = self._wf_cache.get("corrdata")   # holds the table, so its id cannot be reused by a newer one
        if held is not None and held[0] is ep:
            return held[1]
        arr = lambda c: [float(v) if np.isfinite(v) else None for v in ep[c].to_numpy(float)] if c in ep else [None] * len(ep)
        cols = quantity_columns(ep)
        lines = default_columns(ep)
        extra = [c for c in cols if _is_feii_ratio(c) or c.startswith("[")]
        shift = [c for c in cols if c.endswith("_shift") and c.replace("_shift", "_flux") in lines]
        out = {"id": id(ep), "t": arr("mjd"), "names": [str(v) for v in ep["name"]],
               "dates": [mjd_to_date(v) for v in ep["mjd"].to_numpy(float)],
               "columns": [{"name": c, "label": plain_label(c), "log": _use_log(ep, c), "v": arr(c), "e": arr(c + "_err")}
                           for c in cols],
               "map": [c for c in cols if c in lines or c in extra or c in shift]}
        self._wf_cache["corrdata"] = (ep, out)
        return out

    @staticmethod
    def _view_code(name, lag, off):
        lines = [f"ep = fa.epoch_table({name})"]
        if lag:
            lines.append(f"ep = fa.correlations.shift_continuum(ep, {lag:g})   # the continuum {lag:g} days earlier")
        if off:
            lines.append(f"ep = ep.drop(index={sorted(off)!r}).reset_index(drop=True)   # epochs left out")
        return lines

    def wf_correlations_figure(self, cols, fmt="png", dpi=None, lag=0.0, off=()):
        from matplotlib.figure import Figure
        from ..plotting import plot_correlations
        ep = self._epoch_view(lag, off)
        if ep is None:
            return None
        cols = [c for c in cols if c in ep.columns] or None
        with self._render_lock:
            fig = plot_correlations(ep, cols, fig=Figure())
            return self._fig_bytes(fig, fmt, dpi or (150 if fmt == "png" else 300))

    def wf_relation_figure(self, x, y, fmt="png", dpi=None, lag=0.0, off=()):
        from matplotlib.figure import Figure
        from ..plotting import plot_relation
        ep = self._epoch_view(lag, off)
        if ep is None or x not in ep.columns or y not in ep.columns:
            return None
        with self._render_lock:
            fig, _ = plot_relation(ep, x, y, fig=Figure())
            return self._fig_bytes(fig, fmt, dpi or (150 if fmt == "png" else 300))

    def wf_relation_code(self, x, y, lag=0.0, off=()):
        return "\n".join(["from fantasy_agn.plotting import paper_style, plot_relation", ""] +
                         self._view_code(self._epochs_name(), lag, off) +
                         ["with paper_style():", f"    fig, fit = plot_relation(ep, {x!r}, {y!r})   # fit: slope b, scatter, errors",
                          '    fig.savefig("relation.pdf")'])

    def wf_correlations_figure_code(self, cols, lag=0.0, off=()):
        return "\n".join(["from fantasy_agn.plotting import paper_style, plot_correlations", ""] +
                         self._view_code(self._epochs_name(), lag, off) +
                         ["with paper_style():", f"    fig = plot_correlations(ep, {list(cols)!r})" if cols else "    fig = plot_correlations(ep)",
                          '    fig.savefig("correlations.pdf")'])

    # -------------------------------------------- light curves, reverberation
    def wf_lightcurve(self):
        """Light curves of the joint fit (broad-line flux, FWHM, EW and continuum per epoch)."""
        me = self.multi
        if me is None or self.wf.get("fit_kind") not in ("joint", "fits", None):
            return None
        if not me.has_light_curve():
            self.multi_state()                       # starts the background computation
            return {"computing": True}
        fe = me.__dict__.get("_feii_lc")
        if fe is None:
            self._feii_job(me)                       # computed in the background
        key = ("lc", id(me), fe is not None)
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
                               "unit": "", "y": val(c), "err": val(c + "_err"), "source": "fit"})
        try:                                         # measured on the data in line-free windows (Shen et al. 2011)
            w = me.window_continuum()
            for c in w.columns:
                if c.startswith("win") and not c.endswith("_err") and np.isfinite(w[c].to_numpy(float)).sum() >= 3:
                    lo, hi = c[3:].split("_")
                    wv = lambda col: [float(v) if np.isfinite(v) else None for v in w[col].to_numpy(float)]
                    series.append({"name": c, "line": "cont" + c, "label": f"continuum {lo}\u2013{hi} \u00c5", "kind": "flux",
                                   "unit": "", "y": wv(c), "err": wv(c + "_err"), "source": "window", "window": [float(lo), float(hi)]})
        except Exception:
            pass
        lines = sorted({d["line"] for d in series if not d["line"].startswith("cont")},
                       key=lambda ln: [d["line"] for d in series].index(ln))
        try:                                         # the Fe II flux, when the model has Fe II
            y = np.array([d["flux"] for d in fe or []], float)
            if len(y) == len(mjd) and np.isfinite(y).any() and np.nanmax(np.abs(y)) > 0:
                fix = lambda a: [float(v) if np.isfinite(v) else None for v in a]
                series.append({"name": "feii", "line": "feii", "label": "Fe II", "kind": "flux", "unit": "",
                               "y": fix(y), "err": fix(np.array([d.get("flux_err", np.nan) for d in fe], float))})
        except Exception:
            pass
        out = {"mjd": [float(v) if np.isfinite(v) else None for v in mjd], "t0": float(t0),
               "dates": [mjd_to_date(v) for v in mjd], "series": series, "lines": lines,
               "names": [e.spectrum.name for e in me.epochs], "z": float(me.epochs[0].spectrum.z or 0.0),
               "lag_range": self._lag_range(t, mjd), "feii_pending": fe is None}
        self._wf_cache[key] = out
        return out

    @staticmethod
    def _lag_range(t, mjd):
        """The lags to search, as me.reverberation() does by default: +-5x the R-L lag, 60 d to half the baseline."""
        ok = np.isfinite(mjd)
        lmax = 0.5 * (mjd[ok].max() - mjd[ok].min()) if ok.sum() > 1 else 100.0
        lcol = [c for c in t.columns if c.startswith("logL") and not c.endswith("_err")]
        if lcol and np.isfinite(t[lcol[0]].to_numpy(float)).any():
            tau = 10 ** (1.527 + 0.533 * (np.nanmedian(t[lcol[0]].to_numpy(float)) - 44.0))
            lmax = min(lmax, max(5 * tau, 60.0))
        lmax = float(np.ceil(max(lmax, 10.0)))
        return [-lmax, lmax]

    def wf_lagpair(self, a, b):
        """The lag of light curve b behind a by every method, with each method's curve (Results > Light curves)."""
        L = self.wf_lightcurve()
        if not L or "series" not in L:
            return None
        S = {d["name"]: d for d in L["series"]}
        if a not in S or b not in S or a == b:
            raise ValueError("unknown light curve")
        key = ("lagpair", id(self.multi), a, b)
        if key in self._wf_cache:
            return self._wf_cache[key]
        from .. import rm
        arr = lambda v: np.array([np.nan if x is None else x for x in v], float)
        t, fa, ea, fb, eb = arr(L["mjd"]), arr(S[a]["y"]), arr(S[a]["err"]), arr(S[b]["y"]), arr(S[b]["err"])
        ok = np.isfinite(t) & np.isfinite(fa) & np.isfinite(fb) & np.isfinite(ea) & np.isfinite(eb) & (ea > 0) & (eb > 0)
        t, fa, ea, fb, eb = t[ok], fa[ok], ea[ok], fb[ok], eb[ok]
        o = np.argsort(t)
        t, fa, ea, fb, eb = t[o], fa[o], ea[o], fb[o], eb[o]
        lr = tuple(L["lag_range"])
        out = {"a": a, "b": b, "lag_range": list(lr), "n": int(t.size), "methods": [], "zdcf": None}
        if t.size < 5:
            out["error"] = "fewer than 5 epochs with both light curves"
            return out
        num = lambda v: round(float(v), 4) if np.isfinite(v) else None
        rng = np.random.default_rng(0)
        runs = [("ICCF", "iccf", "r", lambda: rm.ccf_lag(t, fa, ea, t, fb, eb, lag_range=lr, n_mc=300)),
                ("von Neumann", "vn", "statistic", lambda: rm.vn_lag(t, fa, ea, t, fb, eb, lr, n_mc=200)),
                ("ZDCF", "zdcf", None, lambda: rm.zdcf_lag(t, fa, ea, t, fb, eb, lr, n_mc=50)),
                ("DRW", "drw", "posterior", lambda: rm.gp_lag((t, fb, eb), spec_continuum=(t, fa, ea), lag_range=lr))]
        for name, k, cy, fn in runs:
            try:
                r = fn()
            except Exception:
                continue
            if not np.isfinite(r.get("tau_cen", np.nan)):
                continue
            smp = np.asarray(r.get("samples", []), float)
            smp = smp[np.isfinite(smp)]
            if smp.size > 300:
                smp = rng.choice(smp, 300, replace=False)
            m = {"key": k, "method": name, "tau": num(r["tau_cen"]), "lo": num(r["tau_err_lo"]), "hi": num(r["tau_err_hi"]),
                 "samples": [round(float(v), 1) for v in smp]}
            if cy and cy in r and "lags" in r:
                lg, y = np.asarray(r["lags"], float), np.asarray(r[cy], float)
                fin = y[np.isfinite(y)]
                if k == "vn" and fin.size:
                    y = y / np.median(fin)               # the statistic relative to its median
                st = max(1, len(lg) // 200)
                m["curve"] = {"lags": [round(float(v), 2) for v in lg[::st]], "y": [num(v) for v in y[::st]]}
            out["methods"].append(m)
        try:
            z = rm.zdcf(t, fa, ea, t, fb, eb)
            z = z[(z.tau >= lr[0]) & (z.tau <= lr[1])]
            col = lambda c: [num(v) for v in z[c].to_numpy(float)]
            out["zdcf"] = {"tau": col("tau"), "sl": col("-sig(tau)"), "sh": col("+sig(tau)"), "dcf": col("dcf"),
                           "el": col("-err(dcf)"), "eh": col("+err(dcf)"), "n": [int(v) for v in z["#bin"]]}
        except Exception:
            pass
        if len(self._wf_cache) > 80:
            self._wf_cache.clear()
        self._wf_cache[key] = out
        return out

    def wf_reverb_code(self, line="Hb", n_mc=1000, methods=("iccf", "vn", "drw"), detrend=False,
                       mjd_min=None, mjd_max=None, primary=None, window=None):
        if self.multi is None:
            raise ValueError("fit several epochs first")
        if not str(line).replace("[", "").replace("]", "").isalnum():
            raise ValueError("unknown line")
        methods = tuple(m for m in methods if m in ("iccf", "vn", "zdcf", "drw"))
        if not methods:
            raise ValueError("choose at least one lag method")
        if primary not in methods:
            primary = "iccf" if "iccf" in methods else methods[0]
        opts = [repr(str(line)), f"n_mc={int(n_mc)}"]
        if methods != ("iccf", "vn", "drw"):
            opts.append(f"methods={methods!r}")
        if primary != "iccf":
            opts.append(f"primary={primary!r}")
        head = ""
        if window:                                   # the continuum measured in a line-free window
            lo, hi = (float(v) for v in window)
            col = f"win{lo:g}_{hi:g}"
            head = (f"w = me.window_continuum([({lo:g}, {hi:g})])   # mean of data minus lines and Fe II in the window\n")
            opts.insert(1, f"continuum=(w.mjd, w.{col}, w.{col}_err)")
        if detrend:
            opts.append("detrend=True")
        if mjd_min not in (None, "") and mjd_max not in (None, ""):
            opts.append(f"mjd_range=({float(mjd_min):g}, {float(mjd_max):g})")
        return ("# reverberation\n" + head + "rm = me.reverberation(" + ", ".join(opts) + ")   # lag, rms spectrum, mass\n"
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
                             "sig": (None if not np.isfinite(q.significance) else
                                     (f"> {q.significance:.1f}\u03c3" if q.p_side >= 1 else f"{q.significance:.1f}\u03c3")),
                             "sig_value": float(q.significance) if np.isfinite(q.significance) else None,
                             "p_side": float(q.p_side) if np.isfinite(q.p_side) else None,
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
        if self.multi is not None and self.wf.get("fit_kind") in ("joint", "fits", None):
            from ..correlations import plain_label
            me = self.multi
            t = me.light_curve()
            cols = [c for c in t.columns if not c.endswith("_err") and not c.endswith("_corr") and c != "mjd"]
            # the flux scale of each epoch (the product of the intercalibration), when it was fitted
            scales = [(getattr(e, "scale", None), getattr(e, "scale_err", None)) for e in me.epochs]
            has_scale = len(scales) == len(t) and any(v is not None and abs(v - 1.0) > 1e-12 for v, _ in scales)
            labels = {"epoch": "Epoch", "name": "Spectrum"}
            head = [labels.get(c) or plain_label(c).replace("F_λ", "Fλ") for c in cols]
            at = cols.index("name") + 1 if "name" in cols else 1
            offsets = [(getattr(e, "offset", None), getattr(e, "offset_err", None)) for e in me.epochs]
            has_offset = len(offsets) == len(t) and any(v is not None for v, _ in offsets)
            broad = [(getattr(e, "broadening", None), getattr(e, "broadening_err", None)) for e in me.epochs]
            has_broad = len(broad) == len(t) and any(v is not None for v, _ in broad)
            head[at:at] = (["Date", "MJD"] + (["Flux scale"] if has_scale else []) + (["Offset (km/s)"] if has_offset else [])
                           + (["Broadening (km/s)"] if has_broad else []))
            rows = []
            for k, d in enumerate(t.to_dict("records")):
                row = [(_fmt_err(d[c], d.get(c + "_err")) if isinstance(d[c], (float, np.floating))
                        else str(d[c])) for c in cols]
                mjd = d.get("mjd")
                known = mjd is not None and np.isfinite(mjd)
                extra = [mjd_to_date(mjd) or "" if known else "", f"{mjd:.2f}" if known else ""]
                if has_scale:
                    v, e = scales[k]
                    extra.append("" if v is None else f"{v:.3f}" + (f" ± {e:.3f}" if e and np.isfinite(e) and e > 0 else " (reference)" if abs(v - 1) < 1e-12 else ""))
                if has_offset:
                    v, e = offsets[k]
                    extra.append("" if v is None else f"{round(v, 1) + 0.0:+.1f}" + (f" ± {e:.1f}" if e and np.isfinite(e) and e > 0 else " (reference)" if v == 0 else ""))
                if has_broad:
                    v, e = broad[k]
                    extra.append("" if v is None else f"{round(v) + 0.0:+.0f}" + (f" ± {e:.0f}" if e and np.isfinite(e) and e > 0 else " (reference)" if v == 0 else ""))
                row[at:at] = extra
                rows.append(row)
            return {"columns": head, "rows": rows}
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
            out[_epoch_key(r.spectrum)] = pts
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

    def wf_catalogs_code(self):
        """Code of the catalog lookup of what was read: every file of a sample, or the one object."""
        raw, files = self.raw, self.wf_input_files()
        if not raw:
            raise ValueError("read a spectrum first")
        if self.wf.get("mode") == "sample" and isinstance(files, list) and len(files) > len(raw):
            return ('# catalogs\ncatalogs = fa.lookup_catalogs(files)\n'
                    'catalogs.attrs["source_files"] = tuple(files)\ncatalogs')
        if self.wf.get("mode") in ("sample", "classify") and len(raw) > 1:
            return "# catalogs\ncatalogs = fa.lookup_catalogs(raw)\ncatalogs"
        if raw[0].ra is None or raw[0].dec is None:
            raise ValueError("this spectrum has no sky position, so the catalogs cannot be searched")
        return "# catalogs\ncatalogs = fa.lookup_catalogs(raw[0])\ncatalogs"

    def _catalog_view(self):
        """The lookup of what is loaded, with the disagreements for the classes known so far; None if there is none."""
        cat, raw = self.ns.get("catalogs"), self.raw
        if not raw or not isinstance(cat, pd.DataFrame) or cat.empty or "name" not in cat:
            return None
        targets = tuple(_target_key(row) for row in cat.itertuples(index=False))
        loaded = {_target_key(s) for s in raw}
        files = self.wf_input_files()
        large_sample = self.wf.get("mode") == "sample" and isinstance(files, list) and len(files) > len(raw)
        sm = self.sample if self.wf.get("fit_kind") == "sample" else None
        if not set(targets) <= loaded:
            # A large sample displays a preview, while its lookup reads all files.
            if not large_sample:
                return None
            if cat.attrs.get("source_files") != tuple(files):
                # Legacy projects have no file provenance; use the fitted sample's file paths.
                if sm is None or "path" not in sm.table or set(sm.table.path) != set(files):
                    return None
                return self._sample_catalogs(sm, catalog_order=True)
        if sm is not None:
            sample_view = self._sample_catalogs(sm, catalog_order=True)
            if sample_view is not None:
                return sample_view

        labels = {}
        for classes in (self._classifications(), self._final_classifications(cached_only=True)):
            stage = {}
            for c in classes or ():
                stage.setdefault(_target_key(c.spectrum), set()).add(c.label)
            # Repeated spectra of one target can have conflicting classifications.
            labels.update({k: next(iter(v)) if len(v) == 1 else None for k, v in stage.items()})
        aligned = tuple(labels.get(key) for key in targets)
        kept = self._catalog_kept
        if kept is None or kept[0] is not cat or kept[1] != (targets, aligned):
            from ..catalogs import with_fit_classes
            view = with_fit_classes(cat, cls=aligned) if any(aligned) else cat.copy(deep=False)
            kept = self._catalog_kept = (cat, (targets, aligned), view)
        return kept[2]

    def wf_catalog(self):
        """For the state: what the lookup of the loaded spectra found, in short."""
        raw = self.raw
        if not raw:
            return None
        files = self.wf_input_files()
        n = len(files) if self.wf.get("mode") == "sample" and isinstance(files, list) else (
            len(raw) if self.wf.get("mode") in ("sample", "classify") else 1)
        out = {"n": n, "has_coords": any(s.ra is not None and s.dec is not None for s in raw), "done": False}
        cat = self._catalog_view()
        if cat is None:
            return out
        bad = np.nonzero(cat["n_disagreements"].to_numpy() > 0)[0]
        found = int(cat["catalog_name"].notna().sum()) if "catalog_name" in cat else 0
        out.update(done=True, stamp=id(cat), n=len(cat), found=found, flagged=[{"k": int(k), "name": str(cat["name"].iloc[k])}
                                                                               for k in bad[:60]], n_flagged=len(bad))
        positions = {}
        for k, row in enumerate(cat.itertuples(index=False)):
            positions.setdefault(_target_key(row), k)
        out["loaded_rows"] = [positions.get(_target_key(s)) for s in raw]
        if len(cat) == 1 and "disagreements" in cat and isinstance(cat["disagreements"].iloc[0], str):
            out["text"] = cat["disagreements"].iloc[0]
        return out

    def wf_catalog_row(self, k=0):
        """Everything the lookup found for object ``k``."""
        cat = self._catalog_view()
        if cat is None:
            return None
        k = min(max(int(k), 0), len(cat) - 1)
        return {"k": k, **{c: (v.item() if hasattr(v, "item") else v) for c, v in cat.iloc[k].items()
                           if v is not None and not (isinstance(v, (float, np.floating)) and not np.isfinite(v))}}

    def wf_sky(self, k=0, fov=60):
        """Image of the sky around object ``k`` of the lookup, as a data URL."""
        row = self.wf_catalog_row(k)
        if not row or "ra" not in row:
            return {"image": None, "survey": None}
        fov = min(max(float(fov), 10.0), 600.0)
        key = ("sky", round(row["ra"], 5), round(row["dec"], 5), fov)
        if key not in self._wf_cache:
            import base64
            from ..catalogs import sky_image
            body, survey = sky_image(row["ra"], row["dec"], fov)
            self._wf_cache[key] = {"image": "data:image/jpeg;base64," + base64.b64encode(body).decode() if body else None,
                                   "survey": survey, "fov": fov}
        return self._wf_cache[key]

    def _class_sets(self, stage="final"):
        """(classifications shown, {name: preliminary}, stage): the final classes (from the fit)
        when there is a fit and ``stage`` is 'final', else the preliminary ones."""
        prelim = self._classifications()
        final = self._final_classifications() if stage == "final" else None
        pmap = {_epoch_key(c.spectrum): c for c in (prelim or [])}
        if final:
            return final, pmap, "final"
        return prelim, pmap, "preliminary"

    @staticmethod
    def _diagram_region(diagram, p):
        """Where a point lies in a diagnostic diagram, with the boundaries of fa.classify: [N II] BPT star-forming
        (Kauffmann 2003), composite or AGN (Kewley 2001); [S II] and [O I] star-forming, Seyfert or LINER (Kewley 2001,
        2006); WHAN (Cid Fernandes 2011) SF, sAGN, wAGN, retired or passive."""
        from ..classify import kauffmann03, kewley01, kewley06, _above_kewley, PASSIVE_EW, RETIRED_EW
        if not p or p.get("x") is None or p.get("y") is None or not (np.isfinite(p["x"]) and np.isfinite(p["y"])):
            return None
        x, y = float(p["x"]), float(p["y"])
        if diagram == "nii":
            if x < 0.05 and y < kauffmann03(x):
                return "star-forming"
            return "AGN" if _above_kewley(x, y, "nii") else "composite"
        if diagram in ("sii", "oi"):
            if not _above_kewley(x, y, diagram):
                return "star-forming"
            return "Seyfert" if y > kewley06(x, diagram) else "LINER"
        ew = 10 ** y
        if ew < PASSIVE_EW:
            return "passive"
        if ew < RETIRED_EW:
            return "retired"
        if x < -0.4:
            return "SF"
        return "sAGN" if ew > 6 else "wAGN"

    def classify_state(self, k=0, stage="final"):
        cl, pmap, stage = self._class_sets(stage)
        if cl is None:
            return None
        k = min(max(int(k), 0), len(cl) - 1)
        c = cl[k]
        fits = self._fit_diagram_points() if stage == "preliminary" else ((), {})
        key = ("cstate", id(c), len(cl), fits[0], stage)
        if key in self._wf_cache:
            return self._wf_cache[key]
        rows = []
        for name, v in c.measurements.items():
            if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) and np.isfinite(v):
                rows.append({"name": name, "value": f"{v:.4g}"})
        s = c.spectrum
        pre = pmap.get(_epoch_key(s)) if stage == "final" else None

        def points(x):
            own = self._diagram_point(x.measurements)
            if stage == "final":                     # every point is a value of the fit
                return {d: (dict(p, fit=True) if p else None) for d, p in own.items()}
            return self._merge_points(own, fits[1].get(_epoch_key(x.spectrum)))
        out = {"k": k, "label": c.label, "uncertain": bool(c.uncertain), "reason": str(c.reason or ""),
               "name": s.name, "stage": stage,
               "narrow_class": c.measurements.get("narrow_class") if stage == "final" else None,
               "preliminary": ({"label": pre.label, "uncertain": bool(pre.uncertain), "reason": str(pre.reason or "")}
                               if pre is not None else None),
               "labels": [self._class_row(i, x, points(x), pmap, stage) for i, x in enumerate(cl)],
               "rows": rows, "diagrams": self.diagram_curves(), "points": points(c)}
        out["regions"] = {d: self._diagram_region(d, p) for d, p in out["points"].items()}
        # the steps of the preliminary classification (its own fits), when it was made
        c = pre if stage == "final" else c
        if c is None:
            out.update(starlight=None, halpha=None, lines_fit=None)
            if len(self._wf_cache) > 80:
                self._wf_cache.clear()
            self._wf_cache[key] = out
            return out
        s = c.spectrum
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

    def _class_row(self, i, x, pts, pmap, stage):
        """One spectrum of a classification set, for the table and the summary across epochs: its class, the
        region of each diagram it falls in, and the numbers the class rests on (broad-line S/N and FWHM, host)."""
        num = lambda v: float(v) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) and np.isfinite(v) else None
        m = x.measurements
        mjd = getattr(x.spectrum, "mjd", None)
        return {"k": i, "name": x.spectrum.name, "label": x.label, "uncertain": bool(x.uncertain),
                "date": mjd_to_date(mjd), "mjd": num(mjd), "points": pts,
                "regions": {d: self._diagram_region(d, p) for d, p in pts.items()},
                "m": {key: num(m.get(key)) for key in ("broad_snr", "broad_fwhm", "host_fraction_5100", "ew_ha")},
                "broad_line": m.get("broad_line"), "narrow_class": m.get("narrow_class"),
                "preliminary": pmap[_epoch_key(x.spectrum)].label if stage == "final" and _epoch_key(x.spectrum) in pmap else None}

    def classify_figure(self, k=0, kind="steps", fmt="png", stage="final"):
        """Publication figure of classification ``k``: 'steps' (plot_steps) or 'diagrams' (plot);
        the final classes (from the fit) when there is a fit, else the preliminary ones."""
        from matplotlib.figure import Figure
        cl, pmap, stage = self._class_sets(stage)
        if cl is None:
            return None
        k = min(max(int(k), 0), len(cl) - 1)
        c = cl[k]
        if kind != "diagrams" and stage == "final" and _epoch_key(c.spectrum) in pmap:
            c = pmap[_epoch_key(c.spectrum)]               # the steps: those of the preliminary classification
        key = ("cfig", id(c), kind, fmt, stage)
        if key in self._wf_cache:
            return self._wf_cache[key]
        with self._render_lock:
            fig = (c.plot(fig=Figure(), others=cl if len(cl) > 1 else None, highlight=False) if kind == "diagrams"
                   else c.plot_steps(fig=Figure()))
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
        text = lambda v: v if isinstance(v, str) else ""          # a missing value in the catalog is NaN
        out = {"k": k, "name": str(row.get("name", f"row {k}")), "cls": text(row.get("cls")),
               "preset": text(row.get("preset"))}
        err = row.get("error") if "error" in sm.table else None
        if isinstance(err, str) and err:
            out["error"] = err
        else:
            try:
                r = sm.load(k)
                out["flux_unit"] = r.spectrum.flux_unit
                out["plot"] = self._series(r)
                out["lines"] = self._line_rows(r)
                out["params"] = self._param_rows(r)
                out["continuum"] = self._param_rows(r, kinds=("continuum", "host", "template", "calibration",
                                                            "attenuation", "absorption"))
                out.update(self._result_quantities(r))
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
        from ..plotting import MS_TYPES, main_sequence_grid
        return {"names": [str(v) for v in t["name"]] if "name" in t else None,
                "cls": [v if isinstance(v, str) else "" for v in t["cls"]] if "cls" in t else None,
                "r_feii": col("R_FeII"), "hb_fwhm": col("broad Hb FWHM"),
                "r_feii_all": col("R_FeII all groups"), "hb_fwhm_all": col("broad Hb FWHM all groups"),
                "ms_grid": [{"x": list(x), "y": list(y)} for x, y in main_sequence_grid()],
                "ms_types": [{"t": name, "x": x, "y": y} for name, x, y in MS_TYPES],
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

    def _sample_fit(self, k):
        """Fit ``k`` of a fitted sample, loaded from its .fantasy file (the last few kept, so the views of one
        spectrum share one object and its caches)."""
        sm = self.sample
        if sm is None:
            raise ValueError("fit first")
        k = min(max(int(k), 0), len(sm.table) - 1)
        keep = self.__dict__.setdefault("_sample_fits", {})
        if any(s is not sm for s, _ in keep.values()):           # another sample: forget the fits of the old one
            for s, r in keep.values():
                self._forget(r)
            keep.clear()
        if k not in keep:
            err = sm.table.iloc[k].get("error") if "error" in sm.table else None
            if isinstance(err, str) and err:
                raise ValueError("this spectrum could not be fitted")
            if len(keep) >= 6:
                self._forget(keep.pop(next(iter(keep)))[1])
            keep[k] = (sm, sm.load(k))
        return keep[k][1]

    def _forget(self, obj):
        """Drop the cached views of an object about to be released: its id() can be reused by a new object."""
        i = id(obj)
        for key in [key for key in self._wf_cache if isinstance(key, tuple) and i in key[1:]]:
            del self._wf_cache[key]

    def _wf_fit(self, k=0):
        if self.wf.get("fit_kind") == "sample":
            r = self._sample_fit(k)
            return r.spectrum.name, None, r, False
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
        self._preset_made = None
        return self._model_choice_code(f"model = fa.load_model({path!r})")

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

    def _lines_table(self, r):
        """r.lines of a fit, kept while the fit is the same object (it takes a moment)."""
        key = ("lines_table", id(r))
        if key not in self._wf_cache:
            self._wf_cache[key] = (r, r.lines)
        return self._wf_cache[key][1]

    def wf_line_list(self, k=0):
        """The lines of fit ``k`` that can be drawn alone (see plotting.fitted_lines), by wavelength, each with a
        small profile (the summed broad components, or all components, on 48 points) and its FWHM."""
        from ..plotting import fitted_lines, main_line, plain_line, _line_parts, _rest_wave, C_KMS
        from ..components import _in_medium
        from ..fitcompare import half_max_crossings
        r = self._wf_fit(k)[2]
        key = ("line_list", id(r))
        if key in self._wf_cache:
            return self._wf_cache[key]
        lines = []
        for d in fitted_lines(r, self._lines_table(r)):
            row = {"line": d["line"], "label": plain_line(d["line"]), "wave": d["wavelength"],
                   "n": len(d["components"]), "roles": sorted(set(d["roles"]))}
            try:
                lam0 = float(_in_medium(np.array([_rest_wave(r, d["line"])]), r.spectrum.context)[0])
                broad = "broad" in d["roles"]
                win = 12000.0 if broad else 2000.0
                v = np.linspace(-win, win, 600)
                parts = _line_parts(r, d["line"], lam0 * (1 + v / C_KMS))
                roles = dict(zip(d["components"], d["roles"]))
                prof = sum(y for n, y in parts.items() if not broad or roles.get(n) == "broad")
                if np.max(prof) > 0:
                    a, b = half_max_crossings(v, prof)
                    row["fwhm"] = float(b - a) if np.isfinite(a) and np.isfinite(b) else None
                    row["spark"] = [round(float(x), 3) for x in prof[::12] / np.max(prof)]
                    row["win"] = win
            except Exception:                                    # a line that cannot be drawn keeps its name
                pass
            lines.append(row)
        out = {"lines": sorted(lines, key=lambda d: d["wave"]), "main": main_line(r, self._lines_table(r))}
        self._wf_cache[key] = out
        return out

    def wf_line_view(self, k=0, line=None, window=None):
        """
        Results › Line profiles, one line of fit ``k`` for the browser: the data with everything else in the model
        subtracted, its errors and residuals (in sigma) around the line, each component of the line and their sum
        on a fine velocity grid, the half-maximum crossings of the broad profile (or of the whole line), the
        measurement of r.measure (with errors) and the other fitted lines in the window.
        """
        from ..plotting import (fitted_lines, main_line, plain_line, _line_parts, _rest_wave, _transmission,
                                C_KMS, is_coronal, is_bowen)
        from ..components import _in_medium
        from ..fitcompare import half_max_crossings, fwhm_fragile
        name, _, r, _ = self._wf_fit(k)
        line = line or main_line(r, self._lines_table(r))
        fl = fitted_lines(r, self._lines_table(r))
        known = {d["line"]: d for d in fl}
        if line not in known:
            raise ValueError(f"{line} is not a fitted line")
        win = float(window) if window not in (None, "") else None
        key = ("line_view", id(r), line, win)
        if key in self._wf_cache:
            return self._wf_cache[key]
        s = r.spectrum
        lam0 = float(_in_medium(np.array([_rest_wave(r, line)]), s.context)[0])
        comps = [c for c in r._line_components() if line in c.line_names]
        broad = [c.name for c in comps if c.role == "broad"]
        meas = r.measure(line, broad or None)
        if win is None:
            win = float(np.clip(2.5 * meas["fwhm"], 1500.0, 15000.0)) if broad and np.isfinite(meas["fwhm"]) else 1500.0
        win = min(max(win, 300.0), 40000.0)
        v_data = (s.wave / lam0 - 1) * C_KMS
        view = np.abs(v_data) <= win
        if view.sum() < 5:
            raise ValueError(f"{line} is outside the spectrum")
        own = sum(_line_parts(r, line, s.wave).values()) * _transmission(r, s.wave)
        model = r.model_flux()
        iso = s.flux - (model - own)
        idx = np.nonzero(view)[0]
        idx = idx[::max(1, idx.size // 1500)]
        sel = r.mask
        res = np.where(sel, (s.flux - model) / s.err, np.nan)
        vf = np.linspace(-win, win, 700)
        lamf = lam0 * (1 + vf / C_KMS)
        parts = _line_parts(r, line, lamf)
        T = _transmission(r, lamf)
        table = self._lines_table(r)
        nth = {}
        out_comps = []
        for c in comps:
            role = "coronal" if c.role == "narrow" and is_coronal(c) else c.role or "other"
            nth[role] = nth.get(role, 0) + 1
            row = table[(table.component == c.name) & (table.line == line)]
            g = lambda col: (float(row[col].iloc[0]) if len(row) and np.isfinite(row[col].iloc[0]) else None)
            out_comps.append({"name": c.name, "role": role, "n": nth[role], "bowen": bool(is_bowen(c)),
                              "profile": getattr(c, "profile", "gaussian"),
                              "y": [round(float(x), 5) for x in parts[c.name]],
                              "flux": g("flux"), "flux_err": g("flux_err"), "fwhm": g("fwhm"), "fwhm_err": g("fwhm_err"),
                              "shift": g("shift"), "shift_err": g("shift_err"), "ew": g("ew")})
        ref = sum(parts[n] for n in broad) if broad else sum(parts.values())
        hm = None
        if np.max(ref) > 0:
            a, b = half_max_crossings(vf, ref)
            if np.isfinite(a) and np.isfinite(b):
                hm = [float(a), float(b), float(0.5 * np.max(ref))]
        others = []
        for d in fitted_lines(r, self._lines_table(r)):
            if d["line"] == line:
                continue
            vo = (_rest_wave(r, d["line"]) / _rest_wave(r, line) - 1) * C_KMS
            if abs(vo) < win and d["line"] not in [o["line"] for o in others]:
                others.append({"line": d["line"], "label": plain_line(d["line"]), "v": float(vo)})
        clean = lambda a: [None if not np.isfinite(x) else round(float(x), 5) for x in a]
        m = {key_: (None if isinstance(val, float) and not np.isfinite(val) else val)
             for key_, val in meas.items() if key_ not in ("components", "moment_window")}
        m["ratio"] = (meas["fwhm"] / meas["sigma"]) if meas["sigma"] and np.isfinite(meas["sigma"]) and np.isfinite(meas["fwhm"]) else None
        m["fragile_ratio"] = fwhm_fragile(vf, ref)[1] if np.max(ref) > 0 else None
        if m["fragile_ratio"] is not None and not np.isfinite(m["fragile_ratio"]):
            m["fragile_ratio"] = None
        out = {"k": int(k), "name": name, "line": line, "label": plain_line(line), "lam0": lam0, "window": win,
               "of": "broad" if broad else "all", "flux_unit": s.flux_unit,
               "v": [round(float(x), 2) for x in v_data[idx]], "y": clean(iso[idx]), "e": clean(s.err[idx]),
               "ok": [int(b) for b in sel[idx]], "res": clean(res[idx]),
               "vf": [round(float(x), 2) for x in vf], "comps": out_comps,
               "absorbed": [round(float(x), 5) for x in sum(parts.values()) * T] if np.min(T) < 0.99 else None,
               "hm": hm, "meas": m, "others": others}
        self._wf_cache[key] = out
        return out

    def wf_line_epochs(self, line="", window=""):
        """Results › Line profiles › Across epochs: one line in every epoch of a multi-epoch fit, on the velocity
        grid of the first epoch: the profile that is measured (the broad components summed, or the whole line),
        the data with the rest of each epoch's model subtracted (thinned), and FWHM, sigma, shift and flux with
        errors per epoch. ``shared``: the width does not change between epochs (a shape shared in the joint fit)."""
        res = self.wf_results() or []
        eps = [(k, name, date, e) for k, (name, date, e, _) in enumerate(res) if e is not None]
        if len(eps) < 2:
            raise ValueError("fit more than one epoch to compare a line between them")
        first = self.wf_line_view(eps[0][0], line, window)
        line, win = first["line"], first["window"]
        key = ("line_epochs", tuple(id(e) for *_, e in eps), line, win)
        if key in self._wf_cache:
            return self._wf_cache[key]
        num = lambda v: float(v) if v is not None and np.isfinite(v) else None
        rows, of, stack, dstack, estack = [], set(), [], [], []
        grid = np.array(first["vf"], float)
        d0 = np.array(first["v"], float)
        dgrid = d0[np.array(first["ok"], bool)] if np.sum(first["ok"]) > 1 else d0
        gap = 1.5 * np.median(np.diff(d0)) if len(d0) > 1 else np.inf    # no data across masked stretches
        views = {}
        for k, name, date, e in eps:
            try:
                views[k] = first if k == eps[0][0] else self.wf_line_view(k, line, win)
            except ValueError:
                views[k] = None
        # one profile for every epoch: the broad components when any epoch has them (zero where an epoch has
        # none: a broad line that disappeared), else the whole line; a mix would put the narrow line in the rms
        broad_any = any(D is not None and D["of"] == "broad" for D in views.values())
        of = {"broad" if broad_any else "all"}
        for k, name, date, e in eps:
            D = views[k]
            if D is None:
                rows.append({"k": k, "name": name, "date": date, "missing": True})
                continue
            use = [c for c in D["comps"] if c["role"] == "broad"] if broad_any else D["comps"]
            prof = np.sum([c["y"] for c in use], axis=0) if use else np.zeros(len(D["vf"]))
            other = [c for c in D["comps"] if c not in use]
            # the data of this epoch: the line alone, without its own components that are not measured (narrow)
            vd = np.array(D["v"], float)
            yd = np.array([np.nan if y is None else y for y in D["y"]], float)
            if other:
                yd = yd - np.interp(vd, D["vf"], np.sum([c["y"] for c in other], axis=0))
            ok = np.array(D["ok"], bool) & np.isfinite(yd)
            st = max(1, int(ok.sum() // 400))
            M = D["meas"]
            vfk = np.array(D["vf"], float)
            stack.append(np.interp(grid, vfk, prof, left=np.nan, right=np.nan))
            ed = np.array([np.nan if x is None else x for x in D["e"]], float)
            dstack.append(self._near(dgrid, vd[ok], yd[ok], gap))
            estack.append(self._near(dgrid, vd[ok], ed[ok], gap))
            rows.append({"k": k, "name": name, "date": date, "mjd": num(getattr(e.spectrum, "mjd", None)),
                         "y": [round(float(x), 5) for x in prof],
                         "dv": [round(float(x), 1) for x in vd[ok][::st]], "dy": [round(float(x), 4) for x in yd[ok][::st]],
                         "hm": D["hm"], "fwhm": M.get("fwhm"), "fwhm_err": M.get("fwhm_err"), "sigma": M.get("sigma"),
                         "sigma_err": M.get("sigma_err"), "shift": M.get("shift"), "shift_err": M.get("shift_err"),
                         "flux": M.get("flux"), "flux_err": M.get("flux_err")})
        good = [r_ for r_ in rows if not r_.get("missing")]
        w = [r_["fwhm"] for r_ in good if r_["fwhm"] is not None]
        shared = len(w) > 1 and (max(w) - min(w)) <= 1e-3 * np.mean(w)
        out = {"line": line, "label": first["label"], "window": win, "of": "broad" if of == {"broad"} else "all",
               "flux_unit": first["flux_unit"], "vf": first["vf"], "epochs": rows, "shared": bool(shared),
               "joint": self.wf.get("fit_kind") == "joint", "stack": self._mean_rms(grid, stack, dgrid, dstack, estack)}
        self._wf_cache[key] = out
        return out

    @staticmethod
    def _near(grid, x, y, gap):
        """y(x) interpolated onto ``grid``, NaN farther than ``gap`` from the nearest x (masked stretches, the
        ends); x ascending."""
        if x.size < 2:
            return np.full(len(grid), np.nan)
        i = np.clip(np.searchsorted(x, grid), 1, x.size - 1)
        dist = np.minimum(np.abs(grid - x[i - 1]), np.abs(x[i] - grid))
        return np.where(dist < gap, np.interp(grid, x, y), np.nan)

    @staticmethod
    def _mean_rms(grid, stack, dgrid, dstack, estack):
        """The mean and rms (sample standard deviation, the variable part) of a line's profile over the epochs,
        of the fitted profiles on ``grid`` and of the data on ``dgrid``, with FWHM and line dispersion (second
        moment) of each fitted one. The rms of the data is the excess over the noise (the mean squared error
        subtracted in quadrature). None for fewer than two epochs."""
        from ..rm import _half_max_walk, _profile_widths
        if len(stack) < 2:
            return None
        import warnings
        with warnings.catch_warnings():                                # all-NaN columns: outside some epoch's window
            warnings.simplefilter("ignore", RuntimeWarning)
            Y, D = np.array(stack, float), np.array(dstack, float)
            mean, rms = np.nanmean(Y, axis=0), np.nanstd(Y, axis=0, ddof=1)
            dm, ds = np.nanmean(D, axis=0), np.nanstd(D, axis=0, ddof=1)
            ds = np.sqrt(np.clip(ds ** 2 - np.nanmean(np.array(estack, float) ** 2, axis=0), 0, None))

        def width(y):                   # as rm.rms_spectrum: FWHM walked out from the peak, sigma over the line
            y = np.where(np.isfinite(y), y, 0.0)
            if not np.max(y) > 0:
                return None, None, None
            fwhm, sigma = _profile_widths(grid, y, 0.0, np.inf)
            a, b = _half_max_walk(grid, np.clip(y, 0, None))
            fwhm = float(fwhm) if np.isfinite(fwhm) else None
            sigma = float(sigma) if np.isfinite(sigma) else None
            return fwhm, sigma, ([a, b, float(0.5 * np.max(y))] if fwhm else None)

        clean = lambda a: [None if not np.isfinite(x) else round(float(x), 5) for x in a]
        out = {"n": len(stack)}
        for key, y in (("mean", mean), ("rms", rms)):
            f, sg, hm = width(y)
            out[key] = {"y": clean(y), "fwhm": f, "sigma": sg, "hm": hm}
        st = max(1, len(dgrid) // 400)
        out["dv"] = [round(float(x), 1) for x in dgrid[::st]]               # thinned; a None breaks the curve
        out["dmean"], out["drms"] = clean(dm[::st]), clean(ds[::st])
        return out

    def _lines_kwargs(self, r, lines, opts):
        """(function name, keyword arguments) of the line figure: a grid of panels or an overlay of profiles."""
        from ..plotting import WIDTHS, fitted_lines
        opts = dict(opts or {})
        known = {d["line"] for d in fitted_lines(r)}
        lines = [ln for ln in lines if ln in known]
        if not lines:
            raise ValueError("choose at least one fitted line")
        on = lambda key, default="1": str(opts.get(key, default)) != "0"
        width = opts.get("width") if opts.get("width") in WIDTHS else None
        style = "sans-serif" if opts.get("font") == "sans-serif" else True
        win = opts.get("window")
        window = min(max(float(win), 500.0), 40000.0) if win not in (None, "") else None
        if opts.get("mode") == "overlay" and opts.get("across") == "epochs" and len(self.wf_results() or []) > 1:
            eps = [(name, date, e) for name, date, e, _ in self.wf_results() if e is not None]
            ln = lines[0]
            broad_in_all = all(any(c.role == "broad" and ln in c.line_names for c in e._line_components()) for _, _, e in eps)
            kw = dict(lines=ln, labels=[date or name for name, date, _ in eps],
                      which="broad" if opts.get("which", "broad") == "broad" and broad_in_all else "all",
                      normalize=on("normalize"), data=on("data"), mark_fwhm=on("mark_fwhm"), width=width or "single",
                      style=style, legend=on("legend"))
            if window:
                kw["window"] = window
            kw["results"] = [e for _, _, e in eps]
            return "plot_profiles_epochs", kw
        if opts.get("mode") == "overlay":
            broad = {d["line"] for d in fitted_lines(r) if "broad" in d["roles"]}
            which = "broad" if opts.get("which", "broad") == "broad" and all(ln in broad for ln in lines) else "all"
            kw = dict(lines=lines, which=which, normalize=on("normalize"), data=on("data"),
                      mark_fwhm=on("mark_fwhm"), width=width or "single", style=style, legend=on("legend"))
            if window:
                kw["window"] = window
            return "plot_profiles", kw
        nc = str(opts.get("ncols", ""))
        kw = dict(lines=lines, ncols=min(int(nc), len(lines)) if nc.isdigit() and int(nc) > 0 else None,
                  velocity=on("velocity"), width=width or "double", residuals=on("residuals"), info=on("info"),
                  others=on("others", "0"), mark_fwhm=on("mark_fwhm"), window=window, style=style,
                  legend=on("legend"), sharey=on("sharey", "0"))
        return "plot_lines", kw

    def wf_lines_figure(self, k=0, lines=(), opts=None, fmt="png", dpi=None):
        """
        The chosen lines of fit ``k``: a grid of panels (plotting.plot_lines) or, with mode='overlay', the profiles
        overlaid and scaled to their peaks (plotting.plot_profiles). Options: velocity (0: wavelength axis),
        residuals, info (FWHM, sigma and flux), others (the other lines in each window), mark_fwhm, window (km/s),
        ncols, width ('single', 'double' or a journal column), and for the overlay which ('broad', 'all'),
        normalize, data.
        """
        from matplotlib.figure import Figure
        from .. import plotting
        r = self._wf_fit(k)[2]
        func, kw = self._lines_kwargs(r, lines, opts)
        with self._render_lock:
            if func == "plot_profiles_epochs":
                fig = plotting.plot_profiles(kw.pop("results"), fig=Figure(), **kw)
            elif func == "plot_profiles":
                fig = plotting.plot_profiles(r, fig=Figure(), **kw)
            else:
                fig = plotting.plot_lines(r, fig=Figure(), **kw)
            return self._fig_bytes(fig, fmt, dpi or (150 if fmt == "png" else 300))

    def _fit_name(self, k=0):
        """How the notebook refers to fit ``k``: r, me.epochs[k] or fits[k]."""
        kind = self.wf.get("fit_kind")
        return {"joint": f"me.epochs[{int(k)}]", "fits": f"fits[{int(k)}]",
                "sample": f"sample.load({int(k)})"}.get(kind, "r")

    def wf_figure_code(self, k=0, opts=None):
        """Python code of the publication figure of fit ``k``, with the title the preview has (wf_custom_figure)."""
        name, date, _, many = self._wf_fit(k)
        opts = dict(opts or {})
        if "title" not in opts:
            opts["title"] = f"epoch {k}: {name}" + (f"  ({date})" if date else "") if many else name
        return self.figure_code(opts, name=self._fit_name(k))

    def wf_lines_code(self, k=0, lines=(), opts=None):
        """Python code that draws the line figure with these options."""
        import inspect
        from .. import plotting
        r = self._wf_fit(k)[2]
        func, kw = self._lines_kwargs(r, lines, opts)
        if func == "plot_profiles_epochs":
            kw.pop("results")
            defaults = {n: p.default for n, p in inspect.signature(plotting.plot_profiles).parameters.items()}
            obj = "me.epochs" if self.wf.get("fit_kind") == "joint" else "fits"
            line = kw.pop("lines")
            args = [obj, repr(line)] + [f"{n}={v!r}" for n, v in kw.items() if not (n in defaults and v == defaults[n])]
            head = "    fig = plot_profiles("
            return "\n".join(["from fantasy_agn.plotting import paper_style, plot_profiles", "", "with paper_style():",
                              head + (",\n" + " " * len(head)).join(args) + ")", '    fig.savefig("profiles.pdf")'])
        defaults = {n: p.default for n, p in inspect.signature(getattr(plotting, func)).parameters.items()}
        lines_arg = kw.pop("lines")
        args = [repr(lines_arg)] + [f"{n}={v!r}" for n, v in kw.items() if not (n in defaults and v == defaults[n])]
        head = f"    fig = {self._fit_name(k)}.{func}("
        return "\n".join(["from fantasy_agn.plotting import paper_style", "", "with paper_style():",
                          head + (",\n" + " " * len(head)).join(args) + ")", '    fig.savefig("lines.pdf")'])

    def wf_view_code(self, view, k=0, opts=None):
        """Python code for the plot of a Results view (see viewcode.py), following what the view shows."""
        from .viewcode import view_code
        return view_code(self, view, k, opts)

    def wf_components_csv(self, k=0):
        """The fitted model of fit ``k`` and its components as CSV (see Session.components_csv)."""
        return self.components_csv(r=self._wf_fit(k)[2])

    def wf_corner(self, k=0, params=None, fmt="png"):
        r = self._wf_fit(k)[2]
        with self._render_lock:
            return self.corner_figure(params=params, fmt=fmt, dpi=110 if fmt == "png" else 300, r=r)

    def wf_custom_figure(self, k=0, opts=None, fmt="png", dpi=None):
        """The publication figure of fit ``k`` with the appearance options of the Results step; ``dpi`` of a
        downloaded PNG (the preview is drawn at 150)."""
        name, date, r, many = self._wf_fit(k)
        opts = dict(opts or {})
        if "title" not in opts:
            opts["title"] = f"epoch {k}: {name}" + (f"  ({date})" if date else "") if many else name
        with self._render_lock:
            return self.custom_figure(opts, fmt=fmt, dpi=dpi or (150 if fmt == "png" else 300), r=r)

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
                  "source": ["# FANTASY session\n", "Generated by the FANTASY app.\n\n", CITE_MARKDOWN]}]
        for c in self.cells:
            if not c.replayable():
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
                "fit": self.fit_state(), "reverb": self.wf_reverb(), "residuals": self.wf_residuals(), "linescan": self.wf_linescan(),
                "redshift": self.wf_redshift(), "trial": self.wf_trial(), "sharing": self.sharing_table() if self.wf.get("mode") == "multi" else None,
                "sharing_custom": bool(self.wf.get("sharing")), "robust": self.wf_robust(), "feii": self.wf_feii(), "simcheck": self.wf_simcheck(), "jointcheck": self.wf_joint_checks(), "compare": self.wf_compare(), "corr": self.wf_correlations(), "has_ms": self._has_main_sequence(), "catalog": self.wf_catalog(), "classified": self._classifications() is not None, "final_class": self.wf.get("fit_kind") in ("single", "joint", "fits") and bool(self.wf_results())}


def _is_number(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False
