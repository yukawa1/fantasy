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
import pandas as pd

from .. import presets
from .builder import BuilderMixin
from .steps import StepsMixin

STEPS = ("read", "classify", "model", "fit", "results")
UPLOADS = os.path.join(os.path.expanduser("~"), ".fantasy_agn", "uploads")


def autosave_folder():
    """
    Where the session is saved after every run: ``$FANTASY_AUTOSAVE`` (``0`` switches saving
    off), else ``$FANTASY_DATA/sessions`` or ``~/.fantasy_agn/sessions``.
    """
    env = os.environ.get("FANTASY_AUTOSAVE")
    if env is not None and env.strip().lower() in ("0", "", "no", "off", "false"):
        return None
    if env:
        return os.path.expanduser(env)
    base = os.environ.get("FANTASY_DATA") or os.path.join(os.path.expanduser("~"), ".fantasy_agn")
    return os.path.join(base, "sessions")

# Models offered in the interface: label -> (preset, options)
MODELS = {
    "optical": ("optical", {}),
    "optical + host": ("optical", {"host": True}),
    "optical (physical)": ("optical", {"physical": True}),
    "hbeta": ("hbeta", {}),
    "halpha": ("halpha", {}),
    "mgii": ("mgii", {}),
    "civ": ("civ", {}),
    "TDE (blackbody + Bowen)": ("tde", {}),
    "Bowen flare / coronal (BFF)": ("bff", {}),
    "seyfert2": ("seyfert2", {}),
    "liner": ("liner", {}),
    "composite": ("composite", {}),
    "starforming": ("starforming", {}),
    "passive / retired": ("passive", {}),
    "type2 (simple)": ("type2", {}),
    "galaxy (simple)": ("galaxy", {}),
}
CLASS_MODEL = {"type1": "optical", "type1_host": "optical + host", "seyfert2": "seyfert2",
               "liner": "liner", "composite": "composite", "unknown": "type2 (simple)",
               "starforming": "starforming", "retired": "passive / retired", "passive": "passive / retired",
               "uv_mgii": "mgii", "uv_civ": "civ"}
MAX_OUTPUT = 6000


def model_defaults():
    """Per model in the menu: the preset's default outflow and number of broad components."""
    out = {}
    for label, (name, kw) in MODELS.items():
        params = inspect.signature(getattr(presets, name)).parameters
        d = {}
        for k in ("outflow", "broad_components"):
            if k in params:
                d[k] = kw.get(k, params[k].default)
        d["stratified"] = "broad_profile" in params and not kw.get("physical")
        # options the interface can switch for this preset, and their defaults
        d["options"] = [k for k in ("broad_components", "outflow", "host", "continuum", "coronal", "oi6046", "red")
                        if k in params and not (k == "host" and kw.get("physical"))]
        d["values"] = {k: kw.get(k, params[k].default) for k in ("coronal", "oi6046", "red", "continuum") if k in params}
        d["preset"], d["kw"] = name, kw
        out[label] = d
    return out


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


class Stopped(KeyboardInterrupt):
    """Raised in a running cell when the run is stopped from the app."""


class Cell:
    def __init__(self, code):
        self.code = code
        self.n = None
        self.start = None
        self.end = None
        self.out = []          # (text, kind): kind 'out' or 'err'
        self.live = None       # printed output so far, while running

    def as_dict(self):
        running = self.start is not None and self.end is None
        out = [{"text": t, "kind": k} for t, k in self.out]
        live = self.live
        if running and live is not None and live.getvalue():
            out = [{"text": live.getvalue().rstrip()[-MAX_OUTPUT:], "kind": "out"}]
        return {"code": self.code, "n": self.n, "start": self.start, "end": self.end,
                "running": running, "out": out}


def _short_repr(value):
    import pandas as pd
    if isinstance(value, pd.DataFrame):
        with pd.option_context("display.width", 110, "display.max_columns", 12,
                               "display.float_format", "{:.4g}".format):
            text = value.to_string(max_rows=40)
    else:
        text = repr(value)
    return text if len(text) < MAX_OUTPUT else text[:MAX_OUTPUT] + "\n…"


def _is_number(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


class Session(BuilderMixin, StepsMixin):
    """One analysis: a list of code cells run in a shared namespace."""

    def __init__(self):
        from ..progress import set_hook
        self.lock = threading.RLock()
        self._render_lock = threading.Lock()     # matplotlib drawing (requests run in threads)
        self.progress = None
        self._saving = threading.Lock()          # one auto-save at a time
        self.reset()
        set_hook(self._on_progress)

    def _on_progress(self, p):
        """Progress reports of fit_many / fit_epochs while a cell runs (and where a stop takes effect)."""
        with self.lock:
            if self.busy:
                self.progress = p
            stop = self.stopping and threading.current_thread() is self._thread
        if stop:
            raise Stopped()

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
            self.class_figure = None             # the classification figure, kept when a fit replaces the plot
            self.results = []
            self.warnings = []
            self.recommended = None
            self.info = "Read a spectrum to start."
            self._seen = {}               # id() of spec, c, r at the last update
            self._thread, self._in_body = None, False
            self._rotated = False         # the first auto-save of this session keeps the last one as 'previous'
            self.stopping = False
            self._wf_reset()

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
    def multi(self):
        from ..multiepoch import MultiEpochResult
        m = self.ns.get("me")
        return m if isinstance(m, MultiEpochResult) else None

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
                    "progress": self.progress,
                    "figure_stem": self.figure_stem,
                    "has_classification_figure": self.class_figure is not None,
                    "has_samples": self.result is not None and self.result.samples is not None,
                    "results": [{"label": a, "value": b} for a, b in self.results],
                    "warnings": self.warnings, "recommended": self.recommended,
                    "models": list(MODELS), "model_defaults": model_defaults(), "info": self.info,
                    "spectrum": spec.name if spec is not None else None,
                    "has_result": self.result is not None,
                    "model": self.model_state(),
                    "templates": _templates_state(),
                    "multi": self.multi_state(),
                    "lines": self.line_state(),
                    "sample": self.sample_state(),
                    "wf": self.wf_state(),
                    "stopping": self.stopping,
                    "autosaved": self.autosaved() if len(self.cells) < 2 else None,
                    "cpus": os.cpu_count() or 1,
                    "python": platform.python_version(),
                    "version": __import__("fantasy_agn").__version__}

    # --------------------------------------------------------------- projects
    def save_project(self):
        """The whole session (cells with outputs, spectra, models, fits) as .fantasy bytes."""
        import pickle
        import tempfile
        from ..io import save
        keep, skipped = {}, []
        with self.lock:
            for k, v in list(self.ns.items()):
                if k.startswith("_") or k in ("fa", "np", "__name__", "__builtins__"):
                    continue
                try:
                    pickle.dumps(v)
                    keep[k] = v
                except Exception:
                    skipped.append(k)
            cells = [(c.code, c.out, c.n) for c in self.cells]
            project = {"cells": cells, "ns": keep, "count": self.count, "skipped": skipped, "wf": dict(self.wf)}
        with tempfile.NamedTemporaryFile(suffix=".fantasy", delete=False) as tmp:
            path = tmp.name
        try:
            save(project, path)
            with open(path, "rb") as fh:
                return fh.read()
        finally:
            os.remove(path)

    def load_project(self, data):
        """Restore a session saved with :meth:`save_project` (replaces the current one)."""
        import tempfile
        from ..io import load
        with tempfile.NamedTemporaryFile(suffix=".fantasy", delete=False) as tmp:
            tmp.write(data)
            path = tmp.name
        try:
            project = load(path)
        finally:
            os.remove(path)
        if not isinstance(project, dict) or "cells" not in project:
            raise ValueError("this .fantasy file holds a single object, not an app project; "
                             "open it in Python with fa.load()")
        self.reset()
        with self.lock:
            self.ns.update(project["ns"])
            self.cells = []
            for code, out, n in project["cells"]:
                cell = Cell(code)
                cell.out, cell.n = list(out), n
                cell.start = cell.end = time.time()
                self.cells.append(cell)
            self.count = project.get("count", len(self.cells))
            self.wf.update(project.get("wf") or {})
            if not project.get("wf") and "me" in project["ns"]:      # projects saved before the workflow
                self.wf.update(mode="multi", fit_kind="joint")
        # figures, results and recommendations follow the restored objects
        self._seen = {}
        for name in ("spec", "c", "r", "me"):
            if name in self.ns:
                self._update(None)
        if self.result is not None:
            from matplotlib.figure import Figure
            self._show(self.result.plot(fig=Figure()), f"{self.spec.name if self.spec else 'fit'}_fit")
            self._results(self.result)
        self.info = "Project opened." + (f" Not restored: {', '.join(project['skipped'])}." if project.get("skipped") else "")

    # ------------------------------------------------------------ auto-save
    def autosave(self):
        """
        Save the session to :func:`autosave_folder` (``last.fantasy`` with ``last.json``); the
        first save of a new session keeps the one before as ``previous``. Errors are ignored:
        auto-saving must never break a run. Returns the path, or None.
        """
        folder = autosave_folder()
        if folder is None or len(self.cells) < 2 or not self._saving.acquire(blocking=False):
            return None
        import json
        try:
            os.makedirs(folder, exist_ok=True)
            data = self.save_project()
            path = os.path.join(folder, "last.fantasy")
            if not self._rotated:
                for ext in (".fantasy", ".json"):
                    src = os.path.join(folder, "last" + ext)
                    if os.path.exists(src):
                        os.replace(src, os.path.join(folder, "previous" + ext))
                self._rotated = True
            with self.lock:
                spec = self.spec
                meta = {"saved": time.time(), "cells": sum(1 for c in self.cells[1:] if c.n),
                        "spectrum": spec.name if spec is not None else None,
                        "epochs": len(self.ns.get("spectra") or []) if isinstance(self.ns.get("spectra"), list) else 0,
                        "fitted": self.result is not None or self.multi is not None,
                        "size": len(data), "version": __import__("fantasy_agn").__version__}
            for name, body in (("last.fantasy", data), ("last.json", json.dumps(meta).encode())):
                tmp = os.path.join(folder, name + ".tmp")
                with open(tmp, "wb") as fh:
                    fh.write(body)
                os.replace(tmp, os.path.join(folder, name))
            return path
        except Exception:                        # noqa: BLE001  (disk full, an object that cannot be pickled, ...)
            return None
        finally:
            self._saving.release()

    @staticmethod
    def autosaved():
        """The auto-saved sessions: {'last': meta, 'previous': meta} (only those that exist)."""
        import json
        folder = autosave_folder()
        out = {}
        if folder is None:
            return out
        for which in ("last", "previous"):
            meta, data = os.path.join(folder, which + ".json"), os.path.join(folder, which + ".fantasy")
            if os.path.exists(meta) and os.path.exists(data):
                try:
                    with open(meta) as fh:
                        out[which] = json.load(fh)
                except Exception:                # noqa: BLE001
                    continue
        return out

    def restore(self, which="last"):
        """Open an auto-saved session (``'last'`` or ``'previous'``)."""
        if which not in ("last", "previous"):
            raise ValueError("which must be 'last' or 'previous'")
        folder = autosave_folder()
        path = os.path.join(folder or "", which + ".fantasy")
        if folder is None or not os.path.exists(path):
            raise ValueError("there is no auto-saved session")
        with open(path, "rb") as fh:
            self.load_project(fh.read())
        self._rotated = False                    # the next save keeps this one as 'previous'
        self.info = "Last session restored." if which == "last" else "Previous session restored."

    def script(self):
        body =[c.code for c in self.cells[1:] if c.n]
        return ("# FANTASY session (generated by the FANTASY app)\n"
                "import fantasy_agn as fa\n\n" + "\n\n".join(body) + "\n")

    @staticmethod
    def _fig_bytes(fig, fmt="png", dpi=110):
        from ..plotting import paper_style
        buf = io.BytesIO()
        if fmt == "png" and dpi <= 220:
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        else:
            with paper_style():
                fig.savefig(buf, format=fmt, dpi=max(dpi, 300) if fmt == "png" else None, bbox_inches="tight")
        return buf.getvalue()

    def components_csv(self, r=None):
        """
        The fitted model as a table: wavelength, observed flux, error, total model, residual, every
        component (as observed), the host and the AGN (everything but the host), the observed spectrum
        minus the host, and, with an Extinction component, its transmission and the intrinsic components.
        """
        import pandas as pd
        r = self.result if r is None else r
        if r is None:
            return None
        s, m = r.spectrum, r.model
        keep = r.mask if r.mask is not None else np.ones(s.wave.shape, bool)
        x = s.wave[keep]
        comps = m.evaluate(x, s.context, components=True)
        total = sum(comps.values(), np.zeros_like(x))
        host = [c.name for c in m.components if c.kind == "host"]
        cols = {"wavelength": x, "flux": s.flux[keep], "error": s.err[keep], "model": total,
                "residual": s.flux[keep] - total}
        cols.update({f"comp_{k}": v for k, v in comps.items()})
        h = sum((comps[k] for k in host), np.zeros_like(x))
        cols.update({"host": h, "agn": total - h, "flux_minus_host": s.flux[keep] - h})
        att = [c for c in m.components if c.kind == "attenuation"]
        if att:
            full = m.params.full_values().copy()
            for c in att:
                full[m.params.names.index(f"{c.name}.ebv")] = 0.0
            intr = m.evaluate(x, s.context, full=full, components=True)
            now, slices = m.params.full_values(), m._slices()
            for c in att:
                k = m.components.index(c)
                cols[f"transmission_{c.name}"] = c.value(x, now[slices[k]], s.context)
            cols.update({f"intrinsic_{k}": v for k, v in intr.items() if k not in [c.name for c in att]})
        for c in (c for c in m.components if c.kind == "absorption"):   # multiplicative troughs
            now, slices = m.params.full_values(), m._slices()
            cols[f"transmission_{c.name}"] = c.value(x, now[slices[m.components.index(c)]], s.context)
        head = (f"# FANTASY fit of {s.name}: rest-frame wavelength (A, {s.medium}); fluxes in units of "
                f"{getattr(s, 'flux_unit', 1)} erg/s/cm2/A; comp_*: components as observed"
                + ("; intrinsic_*: without the extinction" if att else "") + "\n")
        return head + pd.DataFrame(cols).to_csv(index=False, float_format="%.6g")

    def corner_figure(self, params=None, fmt="png", dpi=110, r=None):
        """Corner plot of the posterior samples (MCMC or Monte Carlo) of the fit."""
        from matplotlib.figure import Figure
        r = self.result if r is None else r
        if r is None or r.samples is None:
            raise ValueError("no samples: fit with errors='mcmc' (or 'mc')")
        names = r.model.params.names
        S = np.asarray(r.samples)
        sel = [names.index(p) for p in (params or self.sampled_params(r)[:8]) if p in names]
        if not sel:
            raise ValueError("no free parameters with samples")
        k = len(sel)
        fig = Figure(figsize=(1.6 * k + 0.6, 1.6 * k + 0.6))
        axs = fig.subplots(k, k, squeeze=False)
        fig.subplots_adjust(hspace=0.06, wspace=0.06, left=0.12, bottom=0.12, right=0.98, top=0.98)
        for i, a in enumerate(sel):
            for j, b in enumerate(sel):
                ax = axs[i, j]
                if j > i:
                    ax.set_visible(False)
                    continue
                if i == j:
                    ax.hist(S[:, a], bins=30, color="#4c72b0", histtype="stepfilled", alpha=0.6)
                    q = np.percentile(S[:, a], [15.87, 50, 84.13])
                    for v in q:
                        ax.axvline(v, color="k", lw=0.8, ls="--" if v != q[1] else "-")
                    ax.set_title(f"{names[a]}\n{q[1]:.4g} +{q[2] - q[1]:.2g} -{q[1] - q[0]:.2g}", fontsize=7)
                    ax.set_yticks([])
                else:
                    ax.hist2d(S[:, b], S[:, a], bins=30, cmap="Blues")
                if i < k - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel(names[b], fontsize=7)
                if j > 0 or i == 0:
                    ax.set_yticklabels([])
                else:
                    ax.set_ylabel(names[a], fontsize=7)
                ax.tick_params(labelsize=6)
        return self._fig_bytes(fig, fmt, dpi)

    @staticmethod
    def sampled_params(r):
        """Free non-linear parameters that the samples constrain (not at a limit, with a spread)."""
        if r.samples is None:
            return []
        S, at = np.asarray(r.samples), set(r.at_limit) | set(r.not_measured)
        out = []
        for i, n in enumerate(r.model.params.names):
            p = r.model.params[n]
            if p.linear or p.tie is not None or p.fixed or n in at:
                continue
            if np.std(S[:, i]) > 1e-6 * max(1.0, abs(np.median(S[:, i]))):
                out.append(n)
        return out

    FIG_KINDS = ("continuum", "host", "balmer_cont", "balmer_lines", "feii", "narrow", "broad", "outflow",
                 "bowen", "other")

    def _figure_kwargs(self, opts):
        """Keyword arguments of r.plot for the appearance options of the Results step (see custom_figure),
        and the y range and font size applied afterwards."""
        import re
        from ..plotting import WIDTHS

        def num(k):
            v = opts.get(k)
            return float(v) if v not in (None, "") else None

        def on(k, default="1"):
            return str(opts.get(k, default)) != "0"

        xmin, xmax = num("xmin"), num("xmax")
        width = opts.get("width") or "double"
        width = width if width in WIDTHS else float(width)
        group = {"auto": "auto", "sum": True, "all": False}.get(opts.get("group") or "auto", "auto")
        colors = {}
        for k in ("data", "model") + self.FIG_KINDS:
            c = str(opts.get(f"color_{k}") or "").strip()
            if re.fullmatch(r"#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3}|[a-zA-Z]{3,20}|0?\.\d+", c):
                colors[k] = c
        hide = [k for k in self.FIG_KINDS if not on(f"show_{k}")]
        styles = {}
        dashes = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}
        for k in ("data", "model") + self.FIG_KINDS:
            st = {}
            if str(opts.get(f"fill_{k}", "")) in ("0", "1") and k not in ("data", "model"):
                st["fill"] = opts[f"fill_{k}"] == "1"
            if num(f"lw_{k}"):
                st["lw"] = min(max(num(f"lw_{k}"), 0.1), 6.0)
            if opts.get(f"ls_{k}") in dashes and k != "data":
                st["ls"] = dashes[opts[f"ls_{k}"]]
            if st:
                styles[k] = st
        labels = str(opts.get("label_lines", "0"))
        lopt = {}
        if num("label_fs"):
            lopt["fontsize"] = min(max(num("label_fs"), 4.0), 20.0)
        if opts.get("label_font") in ("serif", "sans-serif", "monospace"):
            lopt["font"] = opts["label_font"]
        lc = str(opts.get("label_color") or "").strip()
        if re.fullmatch(r"#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3}|[a-zA-Z]{3,20}", lc):
            lopt["color"] = lc
        if opts.get("connectors") in ("leader", "full", "none"):
            lopt["connectors"] = opts["connectors"]
        if str(opts.get("label_rot", "")) in ("0", "90"):
            lopt["rotation"] = int(opts["label_rot"])
        legend = {"0": False, "1": True, "above": True, "inside": "inside", "right": "right"}.get(
            str(opts.get("legend", "1")), True)
        title = None if on("notitle", "0") else (opts.get("title") or None)
        zoom = []
        for part in re.split(r"[,;]", str(opts.get("zoom") or "")):
            m = re.fullmatch(r"\s*(\d+(?:\.\d*)?)\s*[-–:]\s*(\d+(?:\.\d*)?)\s*", part)
            if m and float(m.group(1)) < float(m.group(2)):
                zoom.append((float(m.group(1)), float(m.group(2))))
        zsub = {"continuum": ("continuum",), "continuum_feii": ("continuum", "feii"),
                "lines": ("continuum", "host", "feii")}.get(str(opts.get("zoom_subtract") or ""), ())
        kw = dict(components=on("components"), residuals=on("residuals"), zoom=zoom[:4] or None,
                  zoom_subtract=zsub if zoom else (),
                  xlim=(xmin, xmax) if xmin is not None and xmax is not None else None,
                  width=width, group=group, label_lines={"0": False, "1": True, "all": "all"}.get(labels, False),
                  title=title, colors=colors, hide=hide, data=on("data"), model=on("model"),
                  legend=legend, label_options=lopt, styles=styles,
                  style="sans-serif" if opts.get("font") == "sans-serif" else True,
                  height={"compact": 0.8, "tall": 1.25}.get(str(opts.get("height") or ""), 1.0))
        return kw, (num("ymin"), num("ymax")), num("fontsize")

    def custom_figure(self, opts, fmt="png", dpi=110, r=None):
        """
        The fit plot with options: xmin, xmax, ymin, ymax, title (notitle=1: none),
        residuals, components, data, model, legend (0, 1 = above, 'inside', 'right'), group ('auto', 'sum', 'all'),
        show_<kind> (0 hides that kind of component), color_<kind> / color_data / color_model,
        fill_<kind> (0/1: fill under the curve), lw_<kind> (line width, points), ls_<kind>
        ('solid', 'dashed', 'dotted', 'dashdot'),
        label_lines (0, 1 = strongest, 'all'), label_fs, label_font, label_color, connectors
        ('leader', 'full', 'none'), label_rot (90/0), width ('single', 'double', a journal column such as
        'mnras' or 'aa-double', or inches), fontsize (points), zoom ('4650-5150, 6350-6800': panels of these
        ranges under the spectrum, at most four), zoom_subtract ('continuum', 'continuum_feii', 'lines' = continuum,
        host and Fe II: subtracted in the zoom panels).
        """
        from matplotlib.figure import Figure
        r = self.result if r is None else r
        if r is None:
            raise ValueError("fit a model first")
        kw, (ymin, ymax), fs = self._figure_kwargs(opts)
        fig = r.plot(fig=Figure(), **kw)
        ax = fig.axes[0]
        if ymin is not None or ymax is not None:
            lo, hi = ax.get_ylim()
            ax.set_ylim(ymin if ymin is not None else lo, ymax if ymax is not None else hi)
        if fs:
            for t in fig.findobj(lambda o: hasattr(o, "set_fontsize") and o.get_gid() != "linelabel"):
                try:
                    t.set_fontsize(fs)
                except Exception:                        # noqa: BLE001
                    pass
        return self._fig_bytes(fig, fmt, dpi)

    def figure_code(self, opts, name="r"):
        """Python code that draws the publication figure with these options (only what differs from the defaults)."""
        import inspect
        from ..plotting import plot_result
        from ..plotting import COLORS
        kw, (ymin, ymax), fs = self._figure_kwargs(opts)
        # leave out what equals the defaults (the app sends every colour and style it shows)
        kw["colors"] = {k: v for k, v in kw["colors"].items() if str(v).lower() != str(COLORS.get(k, "")).lower()}
        lo_def = {"color": "#262626", "connectors": "leader", "rotation": 90}
        kw["label_options"] = {k: v for k, v in kw["label_options"].items() if lo_def.get(k) != v}
        if not kw["label_lines"]:
            kw["label_options"] = {}
        styles = {}
        for k, st in kw["styles"].items():
            st = {n: v for n, v in st.items() if not (n == "fill" and v == (k == "feii"))}
            if st:
                styles[k] = st
        kw["styles"] = styles
        defaults = {k: p.default for k, p in inspect.signature(plot_result).parameters.items()}
        empty = lambda v: v in ((), {}, None, [])
        args = [f"{k}={v!r}" for k, v in kw.items()
                if not (k in defaults and (v == defaults[k] or (empty(v) and empty(defaults[k]))))]
        head = f"    fig = {name}.plot("
        lines = ["from fantasy_agn.plotting import paper_style", "", "with paper_style():",
                 head + (",\n" + " " * len(head)).join(args) + ")"]
        if ymin is not None or ymax is not None:
            lines.append(f"    fig.axes[0].set_ylim({ymin!r}, {ymax!r})")
        if fs:
            lines.append(f"    # font size {fs:g} pt: set matplotlib.rcParams['font.size'] before plotting")
        lines.append('    fig.savefig("fit.pdf")')
        return "\n".join(lines)

    def tie_kinematics_code(self, comp, to):
        """Code that ties the velocity shift and width of component comp to those of component to."""
        m = self.model
        if m is None:
            raise ValueError("create a model first")
        names = [c.name for c in m.components]
        for c in (comp, to):
            if c not in names:
                raise ValueError(f"no component {c!r}")
        if comp == to:
            raise ValueError("choose two different components")
        lines = [f"model.tie({comp + '.' + k!r}, {to + '.' + k!r})" for k in ("shift", "fwhm")
                 if f"{comp}.{k}" in m.params.names and f"{to}.{k}" in m.params.names]
        if not lines:
            raise ValueError(f"{comp} and {to} have no common shift/fwhm parameters")
        return "\n".join(lines) + "\nmodel"

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
            self.busy, self.error, self.progress, self.stopping = "running", None, None, False
            self._in_body = True
            self._thread = threading.Thread(target=self._run, args=(cell,), daemon=True)
            self._thread.start()
        return cell

    def stop(self):
        """
        Stop the running cell: the worker processes of a parallel run are ended and the cell
        is interrupted (at the next progress report or Python step; a single long call into
        compiled code finishes first). Returns False when nothing runs.
        """
        import ctypes
        import multiprocessing
        with self.lock:
            if not self.busy or self.stopping:
                return False
            self.stopping, thread = True, self._thread
        for p in multiprocessing.active_children():
            p.terminate()
        with self.lock:
            if self._in_body and thread is not None and thread.is_alive():
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread.ident), ctypes.py_object(Stopped))
        return True

    def _run(self, cell):
        try:
            self._execute(cell)
        except KeyboardInterrupt:                # a stop that arrived after the cell had finished
            with self.lock:
                if cell.n is None:
                    self.count += 1
                    cell.n, cell.end = self.count, time.time()
                cell.live, self.busy, self.progress, self.stopping = None, None, None, False

    def _execute(self, cell):
        printed = io.StringIO()
        cell.live = printed
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
        except Stopped:
            with self.lock:
                if printed.getvalue():
                    cell.out.append((printed.getvalue().rstrip()[:MAX_OUTPUT], "out"))
                cell.out.append(("Stopped.", "err"))
                self.error = "Stopped: the run was interrupted, its results are incomplete."
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
                self._in_body = False
                self.count += 1
                cell.n, cell.end = self.count, time.time()
                cell.live = None
                self.busy, self.progress, self.stopping = None, None, False
            threading.Thread(target=self.autosave, daemon=True).start()

    def _show(self, fig, stem):
        with self.lock:
            self.figure, self.figure_stem = fig, stem
            self.figure_version += 1

    def _update(self, value):
        """Follow changes of spec, c and r (from buttons or edited cells)."""
        from matplotlib.axes import Axes
        from matplotlib.figure import Figure
        spec, c, r, me = self.spec, self.classification, self.result, self.multi
        seen = {"spec": id(spec) if spec is not None else None,
                "c": id(c) if c is not None else None,
                "r": id(r) if r is not None else None,
                "me": id(me) if me is not None else None,
                "rm": id(self.reverb) if self.reverb is not None else None}
        changed = {k for k in seen if seen[k] != self._seen.get(k) and seen[k] is not None}
        self._seen = seen
        if isinstance(value, (Figure, Axes)):
            fig = value if isinstance(value, Figure) else value.figure
            self._show(fig, "figure")
        elif "rm" in changed:
            self._show(self.reverb.plot(fig=Figure()), "reverberation")
            self.info = "Reverberation done: lag, rms widths and mass below the light curves."
        elif "me" in changed:
            self._light_curve_job(me)
            self.info = (f"Multi-epoch fit of {len(me.epochs)} spectra done. Light curves below; "
                         "download the table or the session.")
        elif "r" in changed:
            self._show(r.plot(fig=Figure()), f"{spec.name if spec else 'fit'}_fit")
            self._results(r)
            self.info = "Done. Download the figure, the line table or the session as a script."
        elif "c" in changed:
            self._show(c.plot_steps(fig=Figure()), f"{c.spectrum.name}_classification")
            self.class_figure = self.figure
            self.recommended = {"model": CLASS_MODEL.get(c.label, "type2"),
                                "broad_components": 2 if c.label.startswith("type1") or c.label == "uv_civ" else 1,
                                "label": c.label, "uncertain": c.uncertain}
            self.info = (f"Recommended model: {self.recommended['model']}. Change it if you like, "
                         "then press Fit.")
        elif "spec" in changed:
            self._show(spec.plot(fig=Figure()).figure, f"{spec.name}_spectrum")
            self.recommended, self.results, self.warnings = None, [], []
            self.class_figure = None
            self.info = "Next: classify the spectrum."

    def _results(self, r):
        from ..compare import Comparison
        t = r.key_results()
        cmp = self.ns.get("cmp")
        sys = {}
        if isinstance(cmp, Comparison) and any(v is r for v in cmp.results.values()):
            # quantity names of compare -> key_results
            sys = {("broad " + k if k.split()[0] in ("Hb", "Ha", "MgII2798", "CIV1549") else k): v
                   for k, v in cmp.systematic().items()}
        rows = []
        for q in t.itertuples():
            unit = f" {q.unit}" if q.unit else ""
            s_err = sys.get(q.quantity, np.nan)
            if q.unit == "" and "fraction" in q.quantity:
                val = f"{q.value:.0%}"
            elif np.isfinite(q.error) and np.isfinite(s_err):
                val = f"{q.value:.4g} ± {q.error:.2g} ± {s_err:.2g} (model){unit}"
            elif np.isfinite(q.error):
                val = f"{q.value:.4g} ± {q.error:.2g}{unit}"
            else:
                val = f"{q.value:.4g}{unit}"
            rows.append((q.quantity, val))
        if sys:
            acc = int(cmp.table.acceptable.sum())
            rows.append(("decompositions compared", f"{len(cmp.results)} ({acc} acceptable), best: {cmp.best}"))
        self.results = rows
        self.warnings = [ln.strip().replace("WARNING: ", "").replace("note: ", "")
                         for ln in r.summary(short=True).split("\n") if "WARNING" in ln or "note:" in ln]

    def model_state(self):
        """Components and parameters of ``model`` for the model editor."""
        m = self.model
        if m is None:
            return None
        def tied_to(c):
            t = [m.params[f"{c.name}.{k}"].tie for k in ("shift", "fwhm") if f"{c.name}.{k}" in m.params.names]
            t = [x[0].split(".")[0] for x in t if x]
            return t[0] if t else None

        comps = [{"name": c.name, "type": type(c).__name__, "kind": getattr(c, "kind", ""),
                  "role": getattr(c, "role", None), "profile": getattr(c, "profile", None),
                  "absorption": bool(getattr(c, "absorption", False)) or bool(
                      getattr(c, "line_names", None) and all(p.max <= 0 for p in c.params if p.linear)),
                  "tied_to": tied_to(c), "rebuildable": getattr(c, "_args", None) is not None,
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
                  deredden=True, ebv=None, rest_frame=True, mask_absorption=False, ra=None, dec=None):
        """
        Code that reads a spectrum, optionally corrects Galactic extinction
        (E(B-V) looked up automatically at the position, or ``ebv``) and moves it
        to the rest frame. ``ra``/``dec`` (degrees or sexagesimal) set the position
        when the file has none (needed for the extinction look-up).
        """
        z = float(z) if z not in (None, "", 0, "0") else None
        ra = str(ra).strip() if ra not in (None, "") else None
        dec = str(dec).strip() if dec not in (None, "") else None
        if (ra is None) != (dec is None):
            raise ValueError("give both RA and Dec (or neither)")
        if ra is not None:
            from ..spectrum import parse_coordinates
            parse_coordinates(ra, dec)                     # fails early on unreadable input
            fmt = lambda v: repr(float(v)) if _is_number(v) else repr(v)
            position = f"ra={fmt(ra)}, dec={fmt(dec)}"
        if targetid:
            src = f"fa.desi.fetch({int(targetid)})"
            if ra is not None:
                src += f".with_coordinates({position.replace('ra=', '').replace('dec=', '')})"
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
            src = f"fa.read({os.path.expanduser(path.strip())!r}" + (f", {position}" if ra is not None else "") + ")"
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
        if mask_absorption:
            code += ".mask_absorption()" if rest_frame or not z else ""
            if not rest_frame and z:
                code += "\nspec = spec.mask_absorption()"
        lines = [code]
        if self.model is not None:
            # the model of the previous spectrum would otherwise be fitted to this one
            lines.append("model = None   # new spectrum: create its model in step 3 (or Fit uses the recommended one)")
        if deredden:
            lines.append('print("Galactic E(B-V) =", round(spec.meta["ebv"], 4), "from", spec.meta["ebv_source"])')
        if mask_absorption:
            lines.append('print(len(spec.meta["absorbers"]), "narrow absorption lines masked")')
        lines.append("spec")
        return "\n".join(lines)

    @staticmethod
    def _absorbed(code, err):
        """Use fa.fit_absorbed instead of fa.fit (troughs found in the residuals)."""
        return code.replace(f"r = fa.fit(spec, model{err})",
                            f"r = fa.fit_absorbed(spec, model{err})   # adds absorption troughs while the BIC improves\n"
                            "print(r.absorbers.round(2))")

    def fit_code(self, model="optical", broad_components=2, outflow=True, errors="covariance",
                 absorbers=False, **extra):
        self._need_rest_frame()
        host_choice = extra.get("host")
        name, kw = self._preset_args(model, broad_components, outflow, **extra)
        args = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        err = f', errors="{errors}"' if errors != "covariance" else ""
        code = f"model = fa.presets.{name}({args})\nr = fa.fit(spec, model{err})"
        from ..refine import EXTENSIONS
        if absorbers:
            code = self._absorbed(code, err)
        elif name in EXTENSIONS and extra.get("refine", True):
            # narrow-line galaxies: sky residuals masked; at high S/N (chi2/nu > 1.2) the
            # extensions of the preset (calibration polynomial, outflow, Gauss-Hermite
            # profiles, ...) that the BIC prefers
            code = (f"r = fa.refine(spec.mask_sky(), {name!r}, {kw!r}{err})   "
                    "# the preset, plus extensions the BIC prefers if chi2/nu > 1.2\n"
                    "model = r.model\n"
                    "print(r.refinement.to_string(index=False) if len(r.refinement) else 'no refinement needed')")
        elif host_choice in (None, "", "default"):
            code += self._host_check_code(name, kw, err)
        return code + "\nr.key_results()"

    def uncertainty_code(self, model="optical", broad_components=2, outflow=True, **extra):
        """
        Code that fits several decompositions of the preset (1-3 broad components,
        with/without outflow and host) and reports the model (systematic) error.
        With a model from step 3 that is still the preset, its options are used.
        """
        self._need_rest_frame()
        if self.model is not None:
            made = self._unedited_preset()
            if made is None:
                raise ValueError("the model uncertainty compares decompositions of a preset; this model "
                                 "was edited. Fit it without the option, or use fa.compare in a cell")
            name, kw = made
        else:
            name, kw = self._preset_args(model, broad_components, outflow, **extra)
        params = inspect.signature(getattr(presets, name)).parameters
        if "broad_components" not in params:
            raise ValueError(f"the model uncertainty compares broad-line decompositions; {name} has no "
                             "broad lines")
        kw = {k: v for k, v in kw.items() if k not in ("broad_components", "outflow", "host")}
        if "host" in params and not kw.get("physical"):
            kw["host"] = (False, True)
        args = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        return (f"models = fa.presets.variants({name!r}{', ' + args if args else ''})   "
                f"# 1-3 broad components, with/without outflow{' and host' if 'host' in kw else ''}\n"
                "cmp = fa.compare(spec, models)\n"
                "r = cmp.result                        # the best model (lowest BIC)\n"
                "model = r.model\n"
                "cmp")

    def _host_check_code(self, name, kw, err):
        """
        For a borderline host (classification: some starlight, sigma* not confirmed)
        and a type-1 preset without host: also fit with the host and keep it if the
        BIC prefers it. Empty otherwise.
        """
        c = self.classification
        if not (c is not None and c.spectrum is self.spec and c.measurements.get("host_borderline")
                and name in ("optical", "hbeta", "halpha") and "host" not in kw and not kw.get("physical")):
            return ""
        m = c.measurements
        args_h = ", ".join(f"{k}={v!r}" for k, v in dict(kw, host=True).items())
        return (f"\n# possible starlight (host {m['host_fraction_5100']:.0%} at 5100 A, sigma* S/N "
                f"{m['sigma_star_snr']:.1f}): keep the host if the BIC prefers it"
                f"\nmodel_host = fa.presets.{name}({args_h})"
                f"\nr_host = fa.fit(spec, model_host{err})"
                f"\nprint(f'with host: dBIC = {{r_host.bic - r.bic:.0f}}')"
                f"\nif r_host.bic < r.bic - 10:"
                f"\n    model, r = model_host, r_host")

    def _unedited_preset(self):
        """(name, kwargs) if the current model is still the preset made in step 3, else None."""
        made = self.__dict__.get("_preset_made")
        m = self.model
        if made is None or m is None:
            return None
        name, kw = made
        fresh = getattr(presets, name)(**kw)
        sig = lambda mod: [(p.name, p.value, p.min, p.max, p.fixed, p.tie) for p in mod.params]
        return made if sig(fresh) == sig(m) else None

    # ------------------------------------------------- code for the model editor
    @staticmethod
    def _preset_args(model, broad_components=2, outflow=True, **extra):
        """Preset name and keyword arguments; options the preset lacks are left out."""
        name, kw = MODELS[model]
        kw = dict(kw)
        params = inspect.signature(getattr(presets, name)).parameters
        host = extra.pop("host", None)
        continuum = extra.pop("continuum", None)
        if continuum not in (None, "") and "continuum" in params:
            if continuum not in ("powerlaw", "broken"):
                raise ValueError(f"unknown continuum {continuum!r}: 'powerlaw' or 'broken'")
            if continuum != kw.get("continuum", params["continuum"].default):
                kw["continuum"] = continuum
        profile = extra.pop("broad_profile", None)
        if profile:
            if "broad_profile" not in params:
                raise ValueError(f"{model} has no broad lines: the stratified BLR profile does not apply")
            kw["broad_profile"] = profile
            if profile == "stratified":
                broad_components = 1           # one stratified profile replaces core + very broad
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
        # remembered for the host check at fit time; an explicit host choice turns it off
        host_choice = extra.get("host")
        self._preset_made = (name, kw) if host_choice in (None, "", "default") else None
        return f"model = fa.presets.{name}({args})\nmodel"

    def _need_rest_frame(self):
        if self.spec is not None and not self.spec.rest_frame:
            raise ValueError("the models are defined in the rest frame: read the spectrum again with "
                             "'Move to rest frame' ticked (or run spec = spec.to_rest_frame())")

    def fit_model_code(self, errors="covariance", absorbers=False):
        """Fit the current model (from the model step or an edited cell)."""
        if self.model is None:
            raise ValueError("create a model first (step 3)")
        self._need_rest_frame()
        err = f', errors="{errors}"' if errors != "covariance" else ""
        code = f"r = fa.fit(spec, model{err})"
        if absorbers:
            return self._absorbed(code, err) + "\nr.key_results()"
        made = self._unedited_preset()               # an edited model is fitted as it is
        if made is not None:
            code += self._host_check_code(made[0], made[1], err)
        return code + "\nr.key_results()"

    def split_code(self, component, lines, name):
        """Code that moves lines of a Lines component into their own component (own shift and width)."""
        if self.model is None:
            raise ValueError("create a model first (step 3)")
        name = str(name or "").strip()
        if not name or not all(ch.isalnum() or ch == "_" for ch in name):
            raise ValueError("component names may contain letters, digits and _ only")
        lines = [str(x) for x in lines]
        self.model.copy().split_lines(str(component), lines, name)        # checks it can be done
        return f"model.split_lines({str(component)!r}, {lines!r}, {name!r})\nmodel"

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
            if spec.get("absorption"):
                args.append("absorption=True")
            role = spec.get("role")
            if role in ("narrow", "broad", "outflow"):
                args.append(f"role={role!r}")
            prof = spec.get("profile") or "gaussian"
            if prof != "gaussian":
                if prof not in ("lorentzian", "voigt", "gauss-hermite", "stratified"):
                    raise ValueError("unknown profile")
                args.append(f"profile={prof!r}")
            for key in ("fwhm", "shift"):
                lim = limits(key)
                if lim:
                    args.append(f"{key}={lim!r}")
            if spec.get("own_kinematics"):
                args.append("tie_kinematics=False")
        elif kind in ("PowerLaw", "BrokenPowerLaw", "Blackbody"):
            args.append(repr(name or ("blackbody" if kind == "Blackbody" else "continuum")))
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
                fk = spec.get("feii_kind") or "optical"
                if fk not in ("optical", "uv", "uv_extended"):
                    raise ValueError(f"unknown Fe II model {fk!r}")
                if fk != "optical":
                    args.append(f"kind={fk!r}")
                lim = limits("fwhm")
                if lim:
                    args.append(f"fwhm={lim!r}")
        elif kind == "Absorption":
            troughs = []
            for item in str(spec.get("troughs") or "").replace(";", ",").split(","):
                if not item.strip():
                    continue
                parts = [float(v) for v in item.replace(":", " ").split()]
                if len(parts) not in (2, 3):
                    raise ValueError("troughs: give wavelength:FWHM (km/s), e.g. 4861:300, 5890:150")
                troughs.append(tuple(parts))
            if not troughs:
                raise ValueError("troughs: give at least one wavelength:FWHM")
            args.append(repr(troughs))
            if name:
                args.append(f"name={name!r}")
        elif kind == "Extinction":
            law = spec.get("law") or "F99"
            if law not in ("F99", "calzetti"):
                raise ValueError("law must be F99 or calzetti")
            args.append(f"law={law!r}")
        else:
            raise ValueError(f"unknown component type {kind!r}")
        prefix = "" if self.model is not None else "model = fa.Model()\n"
        code = f"{prefix}model.add(fa.{kind}({', '.join(args)}))"
        to = spec.get("tie_to")
        if to:                                   # the same velocity shift and width as another component
            if self.model is None or to not in [c.name for c in self.model.components]:
                raise ValueError(f"no component {to!r} to tie to")
            new = name or {"Lines": "lines", "FeII": "feii"}.get(kind, "")
            have = {p.split(".", 1)[1] for p in self.model.params.names if p.startswith(to + ".")}
            ks = [k for k in ("shift", "fwhm") if k in have]
            if not ks or kind not in ("Lines", "FeII"):
                raise ValueError(f"{to} has no velocity shift and width to tie to")
            code += "".join(f"\nmodel.tie({new + '.' + k!r}, {to + '.' + k!r})" for k in ks)
        return code + "\nmodel"

    def empty_model_code(self):
        """A model to build by hand: a power-law continuum to start with."""
        return "model = fa.Model(fa.PowerLaw())   # add components to it\nmodel"

    def profile_code(self, name, profile):
        """Code that changes the line profile of a Lines component."""
        if profile not in ("gaussian", "lorentzian", "voigt", "gauss-hermite", "stratified"):
            raise ValueError("unknown profile")
        from ..components import Lines
        if self.model is None or not isinstance(self.model.component(name), Lines):
            raise ValueError(f"{name!r} is not a line component")
        return f"model.set_profile({name!r}, {profile!r})\nmodel"

    def untie_code(self, name):
        """Code that frees the velocity shift and width of component ``name``."""
        m = self.model
        if m is None:
            raise ValueError("create a model first")
        ks = [k for k in ("shift", "fwhm") if f"{name}.{k}" in m.params.names and m.params[f"{name}.{k}"].tie]
        if not ks:
            raise ValueError(f"{name}: shift and width are not tied")
        return "\n".join(f"model.untie({name + '.' + k!r})" for k in ks) + "\nmodel"

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
    # ------------------------------------------------------------ multi-epoch
    @property
    def reverb(self):
        from ..rm import Reverberation
        v = self.ns.get("rm")
        return v if isinstance(v, Reverberation) else None

    def multi_state(self):
        me = self.multi
        if me is None:
            return None
        cached = self.__dict__.get("_multi_cache")
        if cached is not None and cached[0] is me:        # polled often: computed once per result
            return cached[1]
        if not me.has_light_curve():
            self._light_curve_job(me)
            return {"n": len(me.epochs), "rchi2": float(me.chi2 / me.dof), "time": float(me.time), "id": id(me),
                    "mjd": None, "columns": [], "rows": [], "computing": True,
                    "epochs": [{"name": e.spectrum.name, "mjd": e.spectrum.mjd, "rchi2": float(e.chi2 / e.dof),
                                "scale": float(getattr(e, "scale", 1.0)), "reference": k == me.reference,
                                "limits": list(e.at_limit)} for k, e in enumerate(me.epochs)]}
        lc = me.light_curve()
        mjd = [e.spectrum.mjd for e in me.epochs if e.spectrum.mjd is not None]
        out = {"n": len(me.epochs), "rchi2": float(me.chi2 / me.dof), "time": float(me.time),
               "id": id(me),
               "mjd": [min(mjd), max(mjd)] if mjd else None,
               "columns": [str(c) for c in lc.columns],
               "rows": [[(f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)) for v in row]
                        for row in lc.itertuples(index=False)],
               "epochs": [{"name": e.spectrum.name, "mjd": e.spectrum.mjd,
                           "rchi2": float(e.chi2 / e.dof), "scale": float(getattr(e, "scale", 1.0)),
                           "reference": k == me.reference, "limits": list(e.at_limit)}
                          for k, e in enumerate(me.epochs)]}
        self._multi_cache = (me, out)
        return out

    def _light_curve_job(self, me):
        """
        Compute the light curve of a multi-epoch fit (errors of every line in every epoch: slow
        for large models) once, in the background, then show its figure; requests stay fast.
        """
        from matplotlib.figure import Figure
        if self.__dict__.get("_lc_job") is me:
            return
        self._lc_job = me

        def work():
            try:
                me.light_curve()
                with self._render_lock:
                    fig = me.plot_light_curves(fig=Figure())
                if self.multi is me:
                    self._show(fig, "light_curves")
            except Exception:
                pass
        threading.Thread(target=work, daemon=True).start()

    def _feii_job(self, me):
        """The Fe II flux of every epoch (with errors: seconds for many epochs), once, in the background."""
        if self.__dict__.get("_feii_job_of") is me:
            return
        self._feii_job_of = me

        def work():
            try:
                fe = [e.feii() for e in me.epochs]
            except Exception:
                fe = []
            me.__dict__["_feii_lc"] = fe
        threading.Thread(target=work, daemon=True).start()

    def line_state(self):
        """Lines of the current fit for the single-line view (most interesting first)."""
        r, spec = self.result, self.spec
        if r is None or spec is None or r.spectrum.name != spec.name:
            return None
        cached = self.__dict__.get("_line_cache")
        if cached is not None and cached[0] is r:
            return cached[1]
        from ..plotting import fitted_lines, main_line, plain_line
        lines = [{"line": d["line"], "wave": d["wavelength"], "n": len(d["components"]),
                  "label": plain_line(d["line"]),
                  "components": d["components"]} for d in fitted_lines(r)]
        out = {"lines": lines, "main": main_line(r), "id": id(r)}
        self._line_cache = (r, out)
        return out

    def light_curve_csv(self):
        me = self.multi
        return me.light_curve().to_csv(index=False) if me is not None else None

    def save_upload(self, data, filename, batch):
        """Keep an uploaded epoch so that the code (and the script) can read it."""
        name = os.path.basename(filename or "spectrum.fits")
        folder = os.path.join(UPLOADS, "".join(ch for ch in str(batch) if ch.isalnum() or ch in "-_")[:40] or "batch")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, name)
        with open(path, "wb") as fh:
            fh.write(data)
        return path

    # ------------------------------------------------------------------ samples
    @property
    def sample(self):
        from ..sample import SampleResult
        s = self.ns.get("sample")
        return s if isinstance(s, SampleResult) else None

    # (column of the catalog, heading, decimals; None: text), in the order of the table
    _SAMPLE_COLS = (("name", "Spectrum", None), ("z", "z", 3), ("cls", "Class", None), ("preset", "Model", None),
                    ("rchi2", "\u03c7\u00b2\u1d63", 2), ("n_at_limit", "At limit", 0),
                    ("broad Hb FWHM", "H\u03b2 FWHM", 0), ("broad Ha FWHM", "H\u03b1 FWHM", 0),
                    ("broad MgII2798 FWHM", "Mg II FWHM", 0), ("broad CIV1549 FWHM", "C IV FWHM", 0),
                    ("R_FeII", "R(Fe II)", 2), ("feii_dropped", "Fe II dropped", 0), ("log M_BH", "log M_BH", 2), ("log [NII]/Ha", "log [N II]/H\u03b1", 2),
                    ("log [OIII]/Hb", "log [O III]/H\u03b2", 2), ("BPT class", "BPT", None),
                    ("E(B-V) gas", "E(B\u2212V) gas", 2), ("n_e ([SII])", "n_e", 0),
                    ("12+log(O/H) (O3N2)", "12+log(O/H)", 2), ("sigma* (host)", "\u03c3\u2605", 0))

    @staticmethod
    def _sample_failure(message, z):
        """What a failed row means, in words: (reason, whether choosing one model for all can help)."""
        too_blue = "templates cover" in message or ("usable pixels" in message and z is not None and z > 0.8)
        if too_blue:
            return "The spectrum stops before H\u03b2, and the automatic classifier needs the optical range.", True
        return message, False

    def _sample_catalogs(self, sm):
        """The catalog lookup of this sample, or None (none yet, or made for other spectra)."""
        cat = self.ns.get("catalogs")
        if not (isinstance(cat, pd.DataFrame) and "name" in cat and len(cat) == len(sm.table)):
            return None
        if any(isinstance(b, str) and str(a) != b for a, b in zip(cat["name"], sm.table["name"])):
            return None
        kept = self.__dict__.get("_sample_cat")
        if kept is None or kept[0] is not sm or kept[1] is not cat:     # once per lookup and per fit
            from ..catalogs import with_fit_classes
            z = sm.table["z"] if "z" in sm.table else None
            cls = sm.table["cls"] if "cls" in sm.table else None
            kept = self._sample_cat = (sm, cat, with_fit_classes(cat, z, cls))
        return kept[2]

    def sample_state(self):
        sm = self.sample
        if sm is None:
            return None
        cat = self._sample_catalogs(sm)
        cached = self.__dict__.get("_sample_cache")
        if cached is not None and cached[0] is sm and cached[2] is cat:
            return cached[1]
        t = sm.table

        def plain(v):
            if v is None or isinstance(v, (float, np.floating)) and not np.isfinite(v):
                return None
            return float(v) if isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_)) else str(v)
        spec_cols = list(self._SAMPLE_COLS)
        catalog = None
        if cat is not None:
            t = t.copy()
            for c in ("catalog_class", "catalog_z", "wise_w1_w2", "radio_mjy"):
                if c in cat:
                    t[c] = cat[c].values
            if "xray" in cat:
                t["xray"] = ["yes" if isinstance(v, str) else None for v in cat["xray"]]
            at = next(i for i, c in enumerate(spec_cols) if c[0] == "cls") + 1
            spec_cols[at:at] = [("catalog_class", "Catalogs say", None)]
            spec_cols += [("catalog_z", "Catalog z", 3), ("wise_w1_w2", "W1\u2212W2", 2), ("radio_mjy", "Radio (mJy)", 1),
                          ("xray", "X-ray", None)]
            catalog = {"flags": [int(v) for v in cat["n_disagreements"]],
                       "flagged": int((cat["n_disagreements"] > 0).sum()), "stamp": id(cat)}
        columns = [{"key": c, "label": label, "digits": digits} for c, label, digits in spec_cols
                   if c in t and t[c].notna().any()]
        rows = [[plain(v) for v in row] for row in t[[c["key"] for c in columns]].itertuples(index=False)]
        status, notes, failures = [], [], {}
        for k in range(len(t)):
            row = t.iloc[k]
            error = plain(row.get("error"))
            if error is not None:
                reason, fixable = self._sample_failure(str(error), plain(row.get("z")))
                failures.setdefault(reason, {"reason": reason, "n": 0, "fixable": fixable})["n"] += 1
                status.append("failed")
                notes.append(reason)
                continue
            rchi2, why = plain(row.get("rchi2")), []
            if rchi2 is not None and rchi2 > 2:
                why.append(f"poor fit: reduced \u03c7\u00b2 = {rchi2:.1f}")
            wide = [c for c in t.columns if c.startswith("broad ") and c.endswith(" FWHM") and (plain(row[c]) or 0) > 20000]
            if wide:
                why.append(f"implausible width: {wide[0]} = {plain(row[wide[0]]):.0f} km/s")
            elif any("broad" in p and "fwhm" in p.lower() for p in str(row.get("at_limit") or "").split(", ")):
                why.append("a broad-line width sits at the limit of the model")
            if bool(row.get("cls_uncertain")) and plain(row.get("cls_uncertain")) is not None:
                why.append("uncertain class")
            status.append("check" if why else "ok")
            if cat is not None and isinstance(cat["disagreements"].iloc[k], str):
                why.append(cat["disagreements"].iloc[k])
            notes.append("; ".join(why))
        has_fig = [isinstance(v, str) for v in (t["saved"] if "saved" in t else [None] * len(t))]
        chi2 = t["rchi2"].to_numpy(float) if "rchi2" in t else np.array([np.nan])
        out = {"n": len(t), "failed": status.count("failed"), "check": status.count("check"), "fitted": len(t) - status.count("failed"),
               "classes": t["cls"].value_counts().to_dict() if "cls" in t else {},
               "median_rchi2": float(np.median(chi2[np.isfinite(chi2)])) if np.isfinite(chi2).any() else None,
               "columns": columns, "rows": rows, "status": status, "notes": notes,
               "failures": sorted(failures.values(), key=lambda f: -f["n"]), "figs": has_fig, "out": sm.out, "id": id(sm),
               "catalog": catalog}
        self._sample_cache = (sm, out, cat)
        return out

    def sample_figure(self, k, kind="fit"):
        """PNG of the fit (or main line) of row ``k`` of the sample, from its output folder."""
        sm = self.sample
        if sm is None:
            return None
        k = int(k)
        if not 0 <= k < len(sm.table):
            raise ValueError(f"row {k} does not exist")
        saved = sm.table.iloc[k].get("saved") if "saved" in sm.table else None
        if not isinstance(saved, str):
            raise ValueError("no figure for this spectrum (it failed or no output folder)")
        path = saved[:-len(".fantasy")] + ("_line.png" if kind == "line" else "_fit.png")
        if not os.path.realpath(path).startswith(os.path.realpath(sm.out) + os.sep) or not os.path.exists(path):
            raise ValueError("figure not found")
        with open(path, "rb") as fh:
            return fh.read()

    def sample_csv(self):
        sm = self.sample
        if sm is None:
            return None
        t = sm.table.drop(columns=[c for c in ("trace", "path") if c in sm.table])
        cat = self._sample_catalogs(sm)
        if cat is not None:
            extra = cat.drop(columns=[c for c in cat.columns if c in t or c == "catalog_family"])
            t = pd.concat([t.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
        return t.to_csv(index=False)
