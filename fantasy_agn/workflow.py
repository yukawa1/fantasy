"""
Guided interface for Jupyter: read -> classify -> fit -> results.

    import fantasy_agn as fa
    app = fa.gui()          # the guided workflow (fa.gui(classic=True): the tabbed form)

Every button runs ordinary FANTASY code, and the panel on the left shows that
code, Jupyter-style, with its output: ``In [*]`` and a timer while it runs,
``In [n]`` when it is done. At the end the whole session can be downloaded as
a Python script, so what was done by clicking can be repeated (or changed) in
code. Afterwards the objects are available as ``app.spec``, ``app.classification``,
``app.model`` and ``app.result``.
"""
import base64
import html
import os
import threading
import time

import numpy as np

from . import presets
from .classify import classify
from .fitting import fit
from .gui import _figure, _figure_links, _save_upload
from .spectrum import read

__all__ = ("WorkflowApp",)

STEPS = ("read", "classify", "fit", "results")
ACCENT = "#D55E00"
_SPIN = "|/-\\"

# Recommended model for every class: (preset, options)
MODELS = {
    "optical": ("optical", {}),
    "optical (physical)": ("optical", {"physical": True}),
    "hbeta": ("hbeta", {}),
    "halpha": ("halpha", {}),
    "type2": ("type2", {}),
    "galaxy": ("galaxy", {}),
}
_CLASS_MODEL = {"type1": "optical", "type1_host": "optical (physical)", "seyfert2": "type2",
                "liner": "type2", "composite": "type2", "unknown": "type2",
                "starforming": "galaxy", "retired": "galaxy", "passive": "galaxy"}


def _row(w, children):
    """A row of widgets that wraps in a narrow window."""
    return w.HBox(children, layout=w.Layout(flex_flow="row wrap", align_items="center"))


class _Cell:
    def __init__(self, code):
        self.code = code
        self.n = None              # execution count when done
        self.start = None
        self.elapsed = None
        self.out = []              # (text, kind) kind: 'out' or 'err'


class WorkflowApp:
    """
    The guided widget application (see the module documentation).

    :param background: run long steps in a thread so that the timer moves
        (False runs them directly, e.g. in scripts and tests).
    """

    def _ipython_display_(self):
        from IPython.display import display
        display(self.widget)

    def __init__(self, background=True):
        import ipywidgets as w
        self.w = w
        self.background = background
        self.spec = self.classification = self.model = self.result = None
        self.cells = []
        self._count = 0
        self._lock = threading.Lock()
        self.step = 0
        self.done = set()

        self.stepper = w.HTML()
        self.code = w.HTML(layout=w.Layout(width="100%"))
        self.script = w.HTML()
        self.plot = w.Output(layout=w.Layout(min_height="300px"))
        self.info = w.HTML()

        # 1. read
        self.path = w.Text(placeholder="path to spec-*.fits, a DESI file or a text file",
                           description="File", layout=w.Layout(width="100%"))
        self.upload = w.FileUpload(accept=".fits,.fit,.txt,.dat,.csv", multiple=False,
                                   description="Upload")
        self.targetid = w.Text(placeholder="DESI TARGETID (downloads from Data Lab)",
                               description="DESI ID", layout=w.Layout(flex="1 1 220px"))
        self.z = w.FloatText(value=0.0, description="z (0 = from file)", step=0.001,
                             style={"description_width": "initial"},
                             layout=w.Layout(width="200px"))
        self.dered = w.Checkbox(value=True, description="Correct Galactic extinction", indent=False,
                                layout=w.Layout(width="auto"))
        self.ebv = w.Text(placeholder="auto", description="E(B-V)", layout=w.Layout(width="150px"))
        self.rest = w.Checkbox(value=True, description="Move to rest frame", indent=False,
                               layout=w.Layout(width="auto"))
        self.read_btn = w.Button(description="Read", button_style="primary", icon="folder-open")
        self.read_btn.on_click(lambda _: self._run(self._do_read))
        read_box = w.VBox([w.HTML("<b>1 · Read a spectrum</b> — a file path, an upload, or a DESI "
                                  "target"),
                           self.path, _row(w, [self.upload, self.targetid]),
                           _row(w, [self.dered, self.ebv, self.rest]),
                           _row(w, [self.z, self.read_btn])])

        # 2. classify
        self.classify_btn = w.Button(description="Classify", button_style="primary", icon="search",
                                     disabled=True)
        self.classify_btn.on_click(lambda _: self._run(self._do_classify))
        classify_box = w.VBox([w.HTML("<b>2 · Classify</b> — starlight, broad lines, narrow-line "
                                      "diagnostics (BPT, WHAN)"), self.classify_btn])

        # 3. fit
        self.model_dd = w.Dropdown(options=list(MODELS), description="Model")
        self.ncomp = w.IntSlider(value=2, min=1, max=4, description="Broad comp.",
                                 style={"description_width": "initial"})
        self.outflow = w.Checkbox(value=True, description="Outflow wing")
        self.errors = w.Dropdown(options=[("fast (covariance)", "covariance"),
                                          ("Monte Carlo", "mc")], description="Errors")
        self.fit_btn = w.Button(description="Fit", button_style="primary", icon="play",
                                disabled=True)
        self.fit_btn.on_click(lambda _: self._run(self._do_fit))
        fit_box = w.VBox([w.HTML("<b>3 · Fit</b> — the model recommended by the classification "
                                 "is preselected"),
                          _row(w, [self.model_dd, self.ncomp]),
                          _row(w, [self.outflow, self.errors, self.fit_btn])])

        self.controls = w.VBox([read_box, classify_box, fit_box],
                               layout=w.Layout(border="1px solid #ddd", padding="6px"))
        # side by side when there is room, stacked in a narrow window
        left = w.VBox([self.code, self.script],
                      layout=w.Layout(flex="1 1 400px", min_width="300px", margin="0 10px 10px 0"))
        right = w.VBox([self.controls, self.info, self.plot],
                       layout=w.Layout(flex="1 1 480px", min_width="300px"))
        self.widget = w.VBox([self.stepper,
                              w.HBox([left, right], layout=w.Layout(flex_flow="row wrap",
                                                                    width="100%"))])
        self._render()

    # ------------------------------------------------------------ display
    def _render(self):
        parts = []
        for i, name in enumerate(STEPS):
            if i in self.done:
                style, mark = "color:#009E73;font-weight:bold", "✓ "
            elif i == self.step:
                style, mark = f"color:{ACCENT};font-weight:bold", ""
            else:
                style, mark = "color:#aaa", ""
            parts.append(f"<span style='{style};font-size:15px'>{mark}{name}</span>")
        self.stepper.value = ("<div style='margin:4px 0 8px 0'>"
                              + " <span style='color:#ccc'>&nbsp;→&nbsp;</span> ".join(parts)
                              + "</div>")
        rows = []
        for c in self.cells:
            running = c.n is None and c.start is not None
            label = "In [*]:" if running else (f"In [{c.n}]:" if c.n else "In [ ]:")
            lab_col = ACCENT if running else "#6a9955"
            for k, line in enumerate(c.code.split("\n")):
                lab = label if k == 0 else ""
                rows.append(f"<tr><td style='color:{lab_col};padding-right:8px;white-space:pre;"
                            f"vertical-align:top'>{lab}</td><td style='color:#d4d4d4;"
                            f"white-space:pre'>{html.escape(line)}</td></tr>")
            if running:
                el = time.time() - c.start
                spin = _SPIN[int(el * 6) % 4]
                rows.append(f"<tr><td></td><td style='color:{ACCENT}'>{spin} running … "
                            f"{el:4.0f} s</td></tr>")
            for text, kind in c.out:
                col = "#f48771" if kind == "err" else "#9cdcfe"
                rows.append(f"<tr><td></td><td style='color:{col};white-space:pre-wrap'>"
                            f"{html.escape(text)}</td></tr>")
            rows.append("<tr><td style='height:6px'></td><td></td></tr>")
        if not rows:
            rows.append("<tr><td></td><td style='color:#777'># the code of every step appears "
                        "here</td></tr>")
        self.code.value = ("<div style='background:#1e1e24;border-radius:6px;padding:10px;"
                           "font-family:Menlo,Consolas,monospace;font-size:12px;min-height:420px;"
                           "overflow:auto'><div style='color:#8a8a99;margin-bottom:6px'>"
                           "Python · FANTASY 2.0</div><table style='border-collapse:collapse'>"
                           + "".join(rows) + "</table></div>")
        if any(c.n for c in self.cells):
            text = "import fantasy_agn as fa\n\n" + "\n\n".join(
                c.code for c in self.cells if c.n and not c.code.startswith("import"))
            b64 = base64.b64encode(text.encode()).decode()
            self.script.value = (f"<a download='fantasy_session.py' href='data:text/x-python;"
                                 f"base64,{b64}'>⬇ download this session as a Python "
                                 f"script</a>")

    def _show_figure(self, fig, stem=None):
        import io
        from IPython.display import HTML, Image
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
        self.plot.clear_output()
        self.plot.append_display_data(Image(data=buf.getvalue(), format="png"))
        if stem:
            self.plot.append_display_data(HTML(_figure_links(fig, stem)))

    # -------------------------------------------------------------- runner
    def _run(self, func):
        """Run a step (in a thread when background=True), keeping the panel alive."""
        if self.background:
            t = threading.Thread(target=self._guarded, args=(func,), daemon=True)
            t.start()
            threading.Thread(target=self._tick, args=(t,), daemon=True).start()
        else:
            self._guarded(func)

    def _tick(self, thread):
        while thread.is_alive():
            self._render()
            time.sleep(0.25)
        self._render()

    def _guarded(self, func):
        for b in (self.read_btn, self.classify_btn, self.fit_btn):
            b.disabled = True
        try:
            func()
        except Exception as exc:                       # shown in the code panel
            if self.cells and self.cells[-1].n is None:
                c = self.cells[-1]
                c.out.append((f"{type(exc).__name__}: {exc}", "err"))
                c.start = None
            self.info.value = f"<span style='color:#c62828'>{html.escape(str(exc))}</span>"
        finally:
            self.read_btn.disabled = False
            self.classify_btn.disabled = self.spec is None
            self.fit_btn.disabled = self.spec is None
            self._render()

    def _cell(self, code):
        c = _Cell(code)
        c.start = time.time()
        with self._lock:
            self.cells.append(c)
        self._render()
        return c

    def _finish(self, c, *outputs):
        c.elapsed = time.time() - c.start
        self._count += 1
        c.n = self._count
        c.out += [(o, "out") for o in outputs if o]
        self._render()

    # --------------------------------------------------------------- steps
    def _do_read(self):
        self.info.value = ""
        if not any(c.code.startswith("import") for c in self.cells):
            c = self._cell("import fantasy_agn as fa")
            self._finish(c)
        z = self.z.value or None
        if self.targetid.value.strip():
            tid = int(self.targetid.value.strip())
            src = f"fa.desi.fetch({tid})"
            from . import desi
            spec = desi.fetch(tid)
        elif self.upload.value:
            item = self.upload.value[0] if isinstance(self.upload.value, (list, tuple)) \
                else list(self.upload.value.values())[0]
            name = item["name"]
            src = f'fa.read("{name}")'
            path = _save_upload(item)
            try:
                spec = read(path)
            finally:
                os.unlink(path)
            if not spec.name.startswith("desi-"):
                spec.name = os.path.splitext(name)[0]
        elif self.path.value.strip():
            path = os.path.expanduser(self.path.value.strip())
            src = f'fa.read("{path}")'
            spec = read(path)
        else:
            raise ValueError("give a file path, upload a file or enter a DESI TARGETID")
        code = "spec = " + src
        ebv_text = self.ebv.value.strip()
        if self.dered.value:
            code += f".deredden(ebv={float(ebv_text):g})" if ebv_text else ".deredden()"
        if self.rest.value:
            code += f".to_rest_frame({'z=' + repr(z) if z else ''})"
        c = self._cell(code)
        if self.dered.value:
            spec = spec.deredden(ebv=float(ebv_text)) if ebv_text else spec.deredden()
        if self.rest.value:
            spec = spec.to_rest_frame(z=z) if z else spec.to_rest_frame()
        elif z:
            spec.z = z
        self.spec = spec
        good = spec.good
        ext = (f"E(B-V) = {spec.meta['ebv']:.4f} ({spec.meta['ebv_source']})" if spec.dereddened
               else "Galactic extinction not corrected")
        self._finish(c, f"<Spectrum {spec.name}: {good.sum()} px, "
                        f"{'rest' if spec.rest_frame else 'observed'} frame "
                        f"{spec.wave[good].min():.0f}-{spec.wave[good].max():.0f} \u00c5, "
                        f"z = {spec.z if spec.z is None else round(spec.z, 4)}>", ext)
        self._show_figure(spec.plot(fig=_figure()).figure)
        self.done = {0}
        self.step = 1
        self.classification = self.result = None

    def _do_classify(self):
        c = self._cell("c = fa.classify(spec)\nc")
        cl = classify(self.spec)
        self.classification = cl
        m = cl.measurements
        out = []
        if np.isfinite(m.get("host_fraction_5100", np.nan)):
            out.append(f"host fraction at 5100 Å = {m['host_fraction_5100']:.2f}, "
                       f"σ* = {m['sigma_star']:.0f} km/s")
        out.append(f"broad lines: ΔBIC = {m['broad_dbic']:.0f}, S/N = {m['broad_snr']:.0f}"
                   + (f", FWHM = {m['broad_fwhm']:.0f} km/s" if np.isfinite(m['broad_fwhm']) else ""))
        if np.isfinite(m.get("log_nii_ha", np.nan)):
            out.append(f"log [NII]/Hα = {m['log_nii_ha']:+.2f}, "
                       f"log [OIII]/Hβ = {m['log_oiii_hb']:+.2f}, EW(Hα) = {m['ew_ha']:.1f} Å")
        out.append(f"→ {cl.label}{' (uncertain)' if cl.uncertain else ''}: {cl.reason}")
        self._finish(c, *out)
        self._show_figure(cl.plot_steps(fig=_figure()), f"{self.spec.name}_classification")
        self.model_dd.value = _CLASS_MODEL.get(cl.label, "type2")
        self.ncomp.value = 2 if cl.label.startswith("type1") else 1
        self.info.value = (f"<b>Recommended model:</b> {self.model_dd.value} "
                           "— change it below if you like, then press <b>Fit</b>.")
        self.done = {0, 1}
        self.step = 2

    def _model_code(self):
        name, kw = MODELS[self.model_dd.value]
        kw = dict(kw)
        import inspect
        params = inspect.signature(getattr(presets, name)).parameters
        if "broad_components" in params:
            kw["broad_components"] = self.ncomp.value
        if "outflow" in params:
            kw["outflow"] = bool(self.outflow.value)
        args = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        return name, kw, f"model = fa.presets.{name}({args})"

    def _do_fit(self):
        if not self.spec.rest_frame:
            raise ValueError("the models are defined in the rest frame: read the spectrum with "
                             "'Move to rest frame' ticked")
        name, kw, code = self._model_code()
        err = self.errors.value
        fit_code = "r = fa.fit(spec, model" + (f', errors="{err}"' if err != "covariance" else "") + ")"
        c = self._cell(code + "\n" + fit_code)
        self.model = getattr(presets, name)(**kw)
        self.result = r = fit(self.spec, self.model, errors=err, verbose=False)
        lines = [f"χ²/ν = {r.rchi2:.3f}, {r.n_free} free parameters, {r.time:.1f} s"]
        if r.at_limit:
            lines.append(f"at their limits: {', '.join(r.at_limit[:6])}"
                         + (" …" if len(r.at_limit) > 6 else ""))
        self._finish(c, *lines)
        fig = r.plot(fig=_figure())
        self._show_figure(fig, f"{self.spec.name}_fit")
        self.done = {0, 1, 2}
        self.step = 3
        self._results()

    def _results(self):
        r = self.result
        rows, code = [], []
        broad = [cm for cm in r._line_components() if cm.role == "broad"]
        line = next((ln for ln in ("Hb", "Ha", "MgII2798", "CIV1549")
                     if any(ln in cm.line_names for cm in broad)), None)
        if line:
            comps = [cm.name for cm in broad if line in cm.line_names]
            mm = r.measure(line, comps)
            code.append(f'r.measure("{line}")')
            rows.append((f"broad {line} FWHM", f"{mm['fwhm']:.0f} ± {mm['fwhm_err']:.0f} km/s"))
            rows.append((f"broad {line} flux", f"{mm['flux']:.4g} ± {mm['flux_err']:.2g}"))
            try:
                bh = r.bh_mass(line)
                code.append(f'r.bh_mass("{line}")')
                rows.append(("log M_BH", f"{bh['log_mbh']:.2f} ± {bh['log_mbh_err']:.2f} "
                                         f"(±{bh['systematic_dex']} syst.)"))
            except Exception:
                pass
            dec = r.balmer_decrement()
            if dec:
                code.append("r.balmer_decrement()")
                rows.append(("broad Hα/Hβ", f"{dec[0]:.2f} ± {dec[1]:.2f}"))
        if not broad:
            # narrow-line object: the main narrow lines and the BPT ratios from the full fit
            t = r.lines[r.lines.component.isin(["narrow", "outflow"])]
            tot = t.groupby("line").flux.sum()
            code.append('r.lines')
            if "[OIII]5007" in tot:
                n = t[(t.component == "narrow") & (t.line == "[OIII]5007")].iloc[0]
                rows.append(("[OIII]5007 flux", f"{tot['[OIII]5007']:.4g} (narrow FWHM {n.fwhm:.0f} km/s)"))
            for a, b, lab in (("[NII]6583", "Ha", "log [NII]/H\u03b1"),
                              ("[OIII]5007", "Hb", "log [OIII]/H\u03b2")):
                if a in tot and b in tot and tot[a] > 0 and tot[b] > 0:
                    rows.append((lab, f"{np.log10(tot[a] / tot[b]):+.2f}"))
        if any(cm.kind == "host" and hasattr(cm, "grid") for cm in r.model.components):
            hp = r.host_properties()
            code.append("r.host_properties()")
            rows.append(("σ* (host)", f"{hp['sigma_star'][0]:.0f} ± {hp['sigma_star'][1]:.0f} km/s"))
            k = next((k for k in hp if k.startswith("host_fraction")), None)
            if k:
                rows.append((f"host at {k.split('_')[-1]} Å", f"{hp[k][0]:.0%}"))
        if code:
            c = self._cell("\n".join(code))
            self._finish(c, *[f"{a:<18s} {b}" for a, b in rows])
        table = "".join(f"<tr><td style='padding-right:14px'>{html.escape(a)}</td>"
                        f"<td><b>{html.escape(b)}</b></td></tr>" for a, b in rows)
        csv = base64.b64encode(r.lines.to_csv(index=False).encode()).decode()
        warn = [ln.strip() for ln in r.summary(short=True).split("\n") if "WARNING" in ln]
        self.info.value = ("<div style='border:1px solid " + ACCENT + ";border-radius:6px;"
                           "padding:8px;margin:6px 0'><b>Results</b><table>" + table + "</table>"
                           + "".join(f"<div style='color:#c62828;font-size:12px'>{html.escape(x)}</div>"
                                     for x in warn)
                           + f"<a download='{self.spec.name}_lines.csv' href='data:text/csv;base64,"
                           f"{csv}'>⬇ line table (.csv)</a></div>")
        self.done = {0, 1, 2, 3}
        self.step = 3
