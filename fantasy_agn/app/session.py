"""
The state of one analysis, independent of the user interface.

Everything is code: the buttons (read, classify, fit) generate Python code
into a new cell and run it, and the user can edit any cell and run it again
or add cells. All cells run in one namespace (``fa``, ``spec``, ``c``,
``model``, ``r``, ...), so editing a cell has the same effect as changing the
form. After every run the session looks at ``spec``, ``c`` and ``r`` and
updates the steps, the figure and the results.
"""
import ast
import contextlib
import inspect
import io
import os
import platform
import threading
import time
import traceback

import numpy as np

from .. import presets

STEPS = ("read", "classify", "model", "fit", "results")
UPLOADS = os.path.join(os.path.expanduser("~"), ".fantasy_agn", "uploads")

# Models offered in the interface: label -> (preset, options)
MODELS = {
    "optical": ("optical", {}),
    "optical (physical)": ("optical", {"physical": True}),
    "hbeta": ("hbeta", {}),
    "halpha": ("halpha", {}),
    "seyfert2": ("seyfert2", {}),
    "liner": ("liner", {}),
    "composite": ("composite", {}),
    "starforming": ("starforming", {}),
    "passive / retired": ("passive", {}),
    "type2 (simple)": ("type2", {}),
    "galaxy (simple)": ("galaxy", {}),
}
CLASS_MODEL = {"type1": "optical", "type1_host": "optical (physical)", "seyfert2": "seyfert2",
               "liner": "liner", "composite": "composite", "unknown": "type2 (simple)",
               "starforming": "starforming", "retired": "passive / retired", "passive": "passive / retired"}
MAX_OUTPUT = 6000


_SUP = str.maketrans("-0123456789", "\u207b\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")


def _plain_label(text):
    """A matplotlib mathtext label as plain Unicode text for the browser."""
    import re
    text = re.sub(r"\^\{?(-?\d+)\}?", lambda m: m.group(1).translate(_SUP), text)
    for a, b in (("\\lambda", "\u03bb"), ("\\,", " "), ("\\mathrm", ""), ("\\times", "\u00d7"), ("_", "")):
        text = text.replace(a, b)
    return re.sub(r"[${}]", "", text)


def _templates_state():
    from .. import templates
    have = set(templates.available())
    return [{"name": n, "available": n in have, "size_mb": info["size_mb"], "cite": info["cite"]}
            for n, info in templates.TEMPLATE_SETS.items()]


class Cell:
    def __init__(self, code):
        self.code = code
        self.n = None
        self.start = None
        self.end = None
        self.out = []          # (text, kind): kind 'out' or 'err'

    def as_dict(self):
        return {"code": self.code, "n": self.n, "start": self.start, "end": self.end,
                "running": self.start is not None and self.end is None,
                "out": [{"text": t, "kind": k} for t, k in self.out]}


def _short_repr(value):
    import pandas as pd
    if isinstance(value, pd.DataFrame):
        with pd.option_context("display.width", 110, "display.max_columns", 12,
                               "display.float_format", "{:.4g}".format):
            text = value.to_string(max_rows=40)
    else:
        text = repr(value)
    return text if len(text) < MAX_OUTPUT else text[:MAX_OUTPUT] + "\n…"


class Session:
    """One analysis: a list of code cells run in a shared namespace."""

    def __init__(self):
        self.lock = threading.RLock()
        self.reset()

    def reset(self):
        import fantasy_agn
        with self.lock:
            self.ns = {"fa": fantasy_agn, "np": np, "__name__": "__fantasy__"}
            self.cells = [Cell("import fantasy_agn as fa")]
            self.cells[0].n, self.cells[0].start, self.cells[0].end = 1, time.time(), time.time()
            self.count = 1
            self.busy = None
            self.error = None
            self.figure = None
            self.figure_version = 0
            self.figure_stem = None
            self.results = []
            self.warnings = []
            self.recommended = None
            self.info = "Read a spectrum to start."
            self._seen = {}               # id() of spec, c, r at the last update

    # ----------------------------------------------------------- objects
    @property
    def spec(self):
        from ..spectrum import Spectrum
        s = self.ns.get("spec")
        return s if isinstance(s, Spectrum) else None

    @property
    def classification(self):
        from ..classify import Classification
        c = self.ns.get("c")
        return c if isinstance(c, Classification) else None

    @property
    def model(self):
        from ..model import Model
        m = self.ns.get("model")
        return m if isinstance(m, Model) else None

    @property
    def result(self):
        from ..result import FitResult
        r = self.ns.get("r")
        return r if isinstance(r, FitResult) else None

    def done(self):
        steps = []
        if self.spec is not None:
            steps.append("read")
        if self.classification is not None:
            steps.append("classify")
        if self.model is not None:
            steps.append("model")
        if self.result is not None:
            steps += ["fit", "results"]
        return steps

    def state(self):
        with self.lock:
            spec = self.spec
            return {"steps": list(STEPS), "done": self.done(), "busy": self.busy,
                    "now": time.time(), "cells": [c.as_dict() for c in self.cells],
                    "error": self.error, "figure_version": self.figure_version,
                    "figure_stem": self.figure_stem,
                    "results": [{"label": a, "value": b} for a, b in self.results],
                    "warnings": self.warnings, "recommended": self.recommended,
                    "models": list(MODELS), "info": self.info,
                    "spectrum": spec.name if spec is not None else None,
                    "has_result": self.result is not None,
                    "model": self.model_state(),
                    "templates": _templates_state(),
                    "python": platform.python_version(),
                    "version": __import__("fantasy_agn").__version__}

    def script(self):
        body = [c.code for c in self.cells[1:] if c.n]
        return ("# FANTASY session (generated by the FANTASY app)\n"
                "import fantasy_agn as fa\n\n" + "\n\n".join(body) + "\n")

    def figure_bytes(self, fmt="png", dpi=110):
        from ..plotting import paper_style
        with self.lock:
            fig = self.figure
        if fig is None:
            return None
        buf = io.BytesIO()
        if fmt == "png" and dpi <= 150:
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        else:
            with paper_style():
                fig.savefig(buf, format=fmt, dpi=300 if fmt == "png" else None,
                            bbox_inches="tight")
        return buf.getvalue()

    def lines_csv(self):
        r = self.result
        return r.lines.to_csv(index=False) if r is not None else None

    # ------------------------------------------------------------- running
    def run_cell(self, code, index=None):
        """Run code in a new cell, or replace cell ``index`` and run it again."""
        with self.lock:
            if self.busy:
                raise RuntimeError("a cell is still running")
            if index is None:
                cell = Cell(code)
                self.cells.append(cell)
            else:
                cell = self.cells[int(index)]
                cell.code, cell.out, cell.n, cell.end = code, [], None, None
            cell.start = time.time()
            self.busy, self.error = "running", None
        threading.Thread(target=self._execute, args=(cell,), daemon=True).start()
        return cell

    def _execute(self, cell):
        printed = io.StringIO()
        value = None
        try:
            tree = ast.parse(cell.code, mode="exec")
            last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            if last is not None:
                tree.body = tree.body[:-1]
            with contextlib.redirect_stdout(printed):
                exec(compile(tree, "<cell>", "exec"), self.ns)
                if last is not None:
                    value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), self.ns)
            out = printed.getvalue().rstrip()
            with self.lock:
                if out:
                    cell.out.append((out[:MAX_OUTPUT], "out"))
                if value is not None:
                    cell.out.append((_short_repr(value), "out"))
            self._update(value)
        except Exception as exc:
            tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
            # drop the frames of the app itself
            tb = [ln for ln in tb if "fantasy_agn/app/" not in ln]
            with self.lock:
                if printed.getvalue():
                    cell.out.append((printed.getvalue().rstrip(), "out"))
                cell.out.append(("".join(tb[-6:]).rstrip(), "err"))
                self.error = f"{type(exc).__name__}: {exc}"
        finally:
            with self.lock:
                self.count += 1
                cell.n, cell.end = self.count, time.time()
                self.busy = None

    def _show(self, fig, stem):
        with self.lock:
            self.figure, self.figure_stem = fig, stem
            self.figure_version += 1

    def _update(self, value):
        """Follow changes of spec, c and r (from buttons or edited cells)."""
        from matplotlib.axes import Axes
        from matplotlib.figure import Figure
        spec, c, r = self.spec, self.classification, self.result
        seen = {"spec": id(spec) if spec is not None else None,
                "c": id(c) if c is not None else None,
                "r": id(r) if r is not None else None}
        changed = {k for k in seen if seen[k] != self._seen.get(k) and seen[k] is not None}
        self._seen = seen
        if isinstance(value, (Figure, Axes)):
            fig = value if isinstance(value, Figure) else value.figure
            self._show(fig, "figure")
        elif "r" in changed:
            self._show(r.plot(fig=Figure()), f"{spec.name if spec else 'fit'}_fit")
            self._results(r)
            self.info = "Done. Download the figure, the line table or the session as a script."
        elif "c" in changed:
            self._show(c.plot_steps(fig=Figure()), f"{c.spectrum.name}_classification")
            self.recommended = {"model": CLASS_MODEL.get(c.label, "type2"),
                                "broad_components": 2 if c.label.startswith("type1") else 1,
                                "label": c.label, "uncertain": c.uncertain}
            self.info = (f"Recommended model: {self.recommended['model']}. Change it if you like, "
                         "then press Fit.")
        elif "spec" in changed:
            self._show(spec.plot(fig=Figure()).figure, f"{spec.name}_spectrum")
            self.recommended, self.results, self.warnings = None, [], []
            self.info = "Next: classify the spectrum."

    def _results(self, r):
        t = r.key_results()
        rows = []
        for q in t.itertuples():
            unit = f" {q.unit}" if q.unit else ""
            if q.unit == "" and "fraction" in q.quantity:
                val = f"{q.value:.0%}"
            elif np.isfinite(q.error):
                val = f"{q.value:.4g} ± {q.error:.2g}{unit}"
            else:
                val = f"{q.value:.4g}{unit}"
            rows.append((q.quantity, val))
        self.results = rows
        self.warnings = [ln.strip().replace("WARNING: ", "").replace("note: ", "")
                         for ln in r.summary(short=True).split("\n") if "WARNING" in ln or "note:" in ln]

    def model_state(self):
        """Components and parameters of ``model`` for the model editor."""
        m = self.model
        if m is None:
            return None
        comps = [{"name": c.name, "type": type(c).__name__, "kind": getattr(c, "kind", ""),
                  "role": getattr(c, "role", None), "profile": getattr(c, "profile", None),
                  "lines": list(getattr(c, "line_names", []) or [])[:40],
                  "n_params": len(c.params)} for c in m.components]
        params = []
        for p in m.params:
            params.append({"name": p.name, "value": p.value, "min": p.min, "max": p.max,
                           "fixed": bool(p.fixed), "unit": p.unit, "linear": bool(p.linear),
                           "description": p.description,
                           "tie": p.tie[0] if p.tie else None,
                           "factor": p.tie[1] if p.tie else 1.0})
        return {"components": comps, "params": params,
                "n_free": len(m.params.free_params()),
                "wave_range": list(m.wave_range) if m.wave_range else None}

    # ------------------------------------------------- code for the buttons
    def read_code(self, path=None, data=None, filename=None, targetid=None, z=None,
                  deredden=True, ebv=None, rest_frame=True):
        """
        Code that reads a spectrum, optionally corrects Galactic extinction
        (E(B-V) looked up automatically, or ``ebv``) and moves it to the rest frame.
        """
        z = float(z) if z not in (None, "", 0, "0") else None
        if targetid:
            src = f"fa.desi.fetch({int(targetid)})"
        else:
            if data is not None:
                # uploaded files are kept, so that the code (and the script) can read them again
                os.makedirs(UPLOADS, exist_ok=True)
                name = os.path.basename(filename or "spectrum.fits")
                path = os.path.join(UPLOADS, name)
                with open(path, "wb") as fh:
                    fh.write(data)
            if not path:
                raise ValueError("choose a file, give a path or a DESI TARGETID")
            src = f"fa.read({os.path.expanduser(path.strip())!r})"
        code = "spec = " + src
        if deredden:
            if ebv not in (None, ""):
                code += f".deredden(ebv={float(ebv):g})"
            else:
                code += ".deredden()"            # E(B-V) looked up at the position
        if rest_frame:
            code += f".to_rest_frame({'z=' + repr(z) if z else ''})"
        elif z:
            code += f"\nspec.z = {z!r}"
        lines = [code]
        if deredden:
            lines.append('print("Galactic E(B-V) =", round(spec.meta["ebv"], 4), "from", spec.meta["ebv_source"])')
        lines.append("spec")
        return "\n".join(lines)

    @staticmethod
    def classify_code():
        return "c = fa.classify(spec)\nc"

    def fit_code(self, model="optical", broad_components=2, outflow=True, errors="covariance", **extra):
        self._need_rest_frame()
        name, kw = self._preset_args(model, broad_components, outflow, **extra)
        args = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        err = f', errors="{errors}"' if errors != "covariance" else ""
        return f"model = fa.presets.{name}({args})\nr = fa.fit(spec, model{err})\nr.key_results()"

    # ------------------------------------------------- code for the model editor
    @staticmethod
    def _preset_args(model, broad_components=2, outflow=True, **extra):
        """Preset name and keyword arguments; options the preset lacks are left out."""
        name, kw = MODELS[model]
        kw = dict(kw)
        params = inspect.signature(getattr(presets, name)).parameters
        host = extra.pop("host", None)
        opts = dict(broad_components=int(broad_components), outflow=bool(outflow))
        opts.update({k: bool(v) for k, v in extra.items() if v is not None})
        for k, v in opts.items():
            if k not in params:
                continue
            # write an option when it differs from the preset's default (e.g. red=False
            # for the class presets, which fit to 7500 A by default)
            if k in ("broad_components", "outflow") or v != params[k].default:
                kw[k] = v
        if host not in (None, "", "default") and "host" in params:
            if host == "none":
                if name not in ("type2", "galaxy") and not kw.get("physical"):
                    kw["host"] = False
            else:
                from ..templates import TEMPLATE_SETS, canonical
                if host != "pca" and canonical(host) not in TEMPLATE_SETS:
                    raise ValueError(f"unknown host templates {host!r}")
                kw["host"] = host
        if kw.get("host") is False:
            del kw["host"]                         # the default of the type 1 presets
        return name, kw

    def preset_code(self, model="optical", broad_components=2, outflow=True, **extra):
        name, kw = self._preset_args(model, broad_components, outflow, **extra)
        args = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        return f"model = fa.presets.{name}({args})\nmodel"

    def _need_rest_frame(self):
        if self.spec is not None and not self.spec.rest_frame:
            raise ValueError("the models are defined in the rest frame: read the spectrum again with "
                             "'Move to rest frame' ticked (or run spec = spec.to_rest_frame())")

    def fit_model_code(self, errors="covariance"):
        """Fit the current model (from the model step or an edited cell)."""
        if self.model is None:
            raise ValueError("create a model first (step 3)")
        self._need_rest_frame()
        err = f', errors="{errors}"' if errors != "covariance" else ""
        return f"r = fa.fit(spec, model{err})\nr.key_results()"

    @staticmethod
    def remove_code(name):
        return f"model.remove({str(name)!r})\nmodel"

    def add_code(self, spec):
        """Code that adds a component described by the editor form (validated)."""
        from ..lines import LINES
        kind = spec.get("type")
        name = str(spec.get("name") or "").strip()
        if name and not all(ch.isalnum() or ch == "_" for ch in name):
            raise ValueError("component names may contain letters, digits and _ only")
        if self.model is not None and name in [c.name for c in self.model.components]:
            raise ValueError(f"there is already a component called {name!r}")

        def limits(key):
            v = spec.get(key)
            if not v:
                return None
            vals = tuple(float(x) for x in v)
            if len(vals) != 3 or not vals[1] <= vals[0] <= vals[2]:
                raise ValueError(f"{key}: give start, min and max with min <= start <= max")
            return vals

        args = []
        if kind == "Lines":
            lines = [str(x) for x in spec.get("lines") or []]
            bad = [x for x in lines if x not in LINES]
            if not lines or bad:
                raise ValueError(f"choose lines from the line list{': unknown ' + ', '.join(bad) if bad else ''}")
            args += [repr(name or "lines"), repr(lines)]
            role = spec.get("role")
            if role in ("narrow", "broad", "outflow"):
                args.append(f"role={role!r}")
            prof = spec.get("profile") or "gaussian"
            if prof != "gaussian":
                if prof not in ("lorentzian", "voigt", "gauss-hermite"):
                    raise ValueError("unknown profile")
                args.append(f"profile={prof!r}")
            for key in ("fwhm", "shift"):
                lim = limits(key)
                if lim:
                    args.append(f"{key}={lim!r}")
        elif kind in ("PowerLaw", "BrokenPowerLaw"):
            args.append(repr(name or "continuum"))
            if spec.get("pivot"):
                args.append(f"pivot={float(spec['pivot']):g}")
        elif kind == "Polynomial":
            args += [repr(name or "poly"), f"order={int(spec.get('order') or 2)}"]
        elif kind in ("FeII", "BalmerContinuum", "HighOrderBalmer", "StellarPopulation", "HostGalaxy"):
            if name:
                args.append(f"name={name!r}")
            if kind == "StellarPopulation" and spec.get("templates"):
                from ..templates import TEMPLATE_SETS, canonical
                tpl = canonical(spec["templates"])
                if tpl not in TEMPLATE_SETS:
                    raise ValueError(f"unknown templates {spec['templates']!r}")
                if tpl != "emiles":
                    args.append(f"templates={tpl!r}")
            if kind == "FeII":
                lim = limits("fwhm")
                if lim:
                    args.append(f"fwhm={lim!r}")
        elif kind == "Extinction":
            law = spec.get("law") or "F99"
            if law not in ("F99", "calzetti"):
                raise ValueError("law must be F99 or calzetti")
            args.append(f"law={law!r}")
        else:
            raise ValueError(f"unknown component type {kind!r}")
        prefix = "" if self.model is not None else "model = fa.Model()\n"
        return f"{prefix}model.add(fa.{kind}({', '.join(args)}))\nmodel"

    def params_code(self, changes):
        """
        Code for parameter edits: changes is a list of {name, value, min, max,
        fixed, tie, factor}; only what differs from the current model is written.
        """
        m = self.model
        if m is None:
            raise ValueError("create a model first")
        lines = []
        for ch in changes:
            p = m.params[ch["name"]]
            name = repr(p.name)
            tie = ch.get("tie") or None
            factor = float(ch.get("factor") or 1.0)
            new = {k: float(ch[k]) for k in ("value", "min", "max")
                   if ch.get(k) not in (None, "") and float(ch[k]) != getattr(p, k)}
            fixed = bool(ch.get("fixed"))
            if tie:
                if tie not in m.params.names:
                    raise ValueError(f"{p.name}: unknown parameter {tie!r} to tie to")
                if p.tie != (tie, factor, 0.0) and not (p.tie and p.tie[0] == tie and p.tie[1] == factor):
                    lines.append(f"model.tie({name}, {tie!r}" + (f", factor={factor:g}" if factor != 1 else "") + ")")
                continue
            if p.tie or (p.fixed and not fixed):
                lines.append(f"model.free({name})")
            lim = {k: v for k, v in new.items() if k in ("min", "max")}
            if fixed:
                if lim:
                    lines.append(f"model.set({name}, " + ", ".join(f"{k}={v:g}" for k, v in lim.items()) + ")")
                if not p.fixed or "value" in new:
                    lines.append(f"model.fix({name}" + (f", {new['value']:g}" if "value" in new else "") + ")")
            elif new:
                lines.append(f"model.set({name}, " + ", ".join(f"{k}={v:g}" for k, v in new.items()) + ")")
        if not lines:
            raise ValueError("nothing changed")
        return "\n".join(lines) + "\nmodel"

    # ------------------------------------------------------ interactive plot
    def plot_data(self, max_points=6000):
        """
        Arrays for the interactive plot in the browser: the spectrum, and when
        available the fitted model with its components grouped by kind (or the
        stellar continuum of the classification), and the residuals.
        """
        spec, c, r = self.spec, self.classification, self.result
        if spec is None:
            return None
        series = []
        if r is not None and r.spectrum.name == spec.name:
            s = r.spectrum
            lo, hi = r.model.wave_range or (s.wave.min(), s.wave.max())
            sel = (s.wave >= lo) & (s.wave <= hi)
            parts = r.model_flux(components=True)
            groups = {"continuum": [], "host": [], "Fe II": [], "Balmer": [], "narrow lines": [],
                      "broad lines": [], "outflow": []}
            for comp in r.model.components:
                name, role, kind = comp.name, getattr(comp, "role", None), getattr(comp, "kind", "")
                if name not in parts:
                    continue
                if kind == "continuum" and name.startswith("balmer"):
                    key = "Balmer"
                elif kind == "continuum":
                    key = "continuum"
                elif kind == "host":
                    key = "host"
                elif name.startswith("feii"):
                    key = "Fe II"
                elif name.startswith("balmer"):
                    key = "Balmer"
                elif role in ("broad", "outflow"):
                    key = "broad lines" if role == "broad" else "outflow"
                else:
                    key = "narrow lines"
                groups[key].append(parts[name])
            model = r.model_flux()
            series.append(("model", model))
            for key, arrs in groups.items():
                if arrs:
                    series.append((key, np.sum(arrs, axis=0)))
            resid = (s.flux - model) / s.err
            title = f"{s.name}: fit, \u03c7\u00b2/\u03bd = {r.rchi2:.2f}"
        else:
            s = spec
            good = s.good
            lo, hi = s.wave[good].min(), s.wave[good].max()
            sel = (s.wave >= lo) & (s.wave <= hi)
            resid = None
            title = f"{s.name}: spectrum"
            if c is not None and c.spectrum.name == spec.name:
                series.append(("stars + power law", c.stellar_fit.model_flux(s.wave)))
                title = f"{s.name}: starlight from the classification"
        idx = np.nonzero(sel)[0]
        step = max(1, int(np.ceil(idx.size / max_points)))
        idx = idx[::step]

        def arr(a):
            out = np.asarray(a, float)[idx]
            return [round(float(v), 6) if np.isfinite(v) else None for v in out]

        from ..plotting import flux_label
        ylab = _plain_label(flux_label(s))
        return {"title": title, "x": arr(s.wave), "flux": arr(s.flux),
                "series": [{"name": n, "y": arr(v)} for n, v in series],
                "resid": arr(resid) if resid is not None else None,
                "xlabel": "Rest-frame wavelength (\u00c5)", "ylabel": ylab,
                "version": self.figure_version}

    def data_tables(self):
        """Tables for the Data tab: spectrum, classification and fitted lines."""
        out = []
        spec, c, r = self.spec, self.classification, self.result
        if spec is not None:
            good = spec.good
            out.append({"title": "Spectrum", "columns": ["quantity", "value"], "rows": [
                ["name", spec.name], ["redshift", f"{spec.z:.5f}" if spec.z is not None else "\u2014"],
                ["pixels", int(good.sum())],
                ["rest-frame range", f"{spec.wave[good].min():.0f}\u2013{spec.wave[good].max():.0f} \u00c5"],
                ["resolution", f"{spec.resolution:.0f} km/s" if spec.resolution else "unknown"],
                ["medium", spec.medium]]})
        if c is not None:
            rows = [["class", c.label + (" (uncertain)" if c.uncertain else "")]]
            for k, v in c.measurements.items():
                if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) and np.isfinite(v):
                    rows.append([k, f"{v:.4g}"])
            out.append({"title": "Classification", "columns": ["quantity", "value"], "rows": rows})
        if r is not None:
            t = r.lines
            rows = [[row.component, row.line, f"{row.flux:.4g}", f"{row.flux_err:.2g}",
                     f"{row.fwhm:.0f}", f"{row.shift:.0f}", f"{row.ew:.3g}"] for row in t.itertuples()]
            out.append({"title": "Fitted lines", "columns": ["component", "line", "flux", "\u00b1",
                                                             "FWHM km/s", "shift km/s", "EW \u00c5"],
                        "rows": rows})
        return out

    @staticmethod
    def download_templates_code(name):
        from ..templates import TEMPLATE_SETS, canonical
        name = canonical(name)
        if name not in TEMPLATE_SETS:
            raise ValueError(f"unknown template set {name!r}")
        return f"fa.templates.download({name!r})"
