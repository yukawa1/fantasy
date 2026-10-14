"""
Point-and-click interface for Jupyter notebooks.

    import fantasy_agn as fa
    app = fa.gui()      # shows the interface
    app.result.lines    # results are available in code afterwards

Needs the optional dependency ipywidgets (pip install "fantasy_agn[gui]").
"""
import base64
import io
import os
import tempfile

import numpy as np

from . import presets
from .fitting import fit
from .multiepoch import fit_epochs
from .spectrum import Spectrum

__all__ = ("gui", "FantasyApp")

_HELP = {
    "spectrum": "Upload an SDSS spec-*.fits file, a DESI coadd file, a 1D FITS spectrum or "
                "a text file with columns wavelength, flux, [error]. Check the redshift, then "
                "press <b>Prepare</b>.",
    "model": "Pick a ready-made model. The list shows the models your spectrum covers. "
             "Defaults work for most quasars; change limits under <i>Advanced</i>.",
    "fit": "Press <b>Fit</b>. <i>Fast</i> errors come from the covariance matrix; "
           "<i>Monte Carlo</i> refits the spectrum many times with added noise and is "
           "more reliable for weak or blended lines.",
    "multi": "Upload several spectra of the same object. Narrow lines are shared by all "
             "epochs and used to intercalibrate the flux scale; broad lines and continuum "
             "vary. Redshift and E(B-V) are taken from the Spectrum tab.",
}


def _download_link(text, filename, label):
    b64 = base64.b64encode(text.encode()).decode()
    return (f'<a download="{filename}" href="data:text/csv;base64,{b64}" '
            f'style="margin-right:14px">⬇ {label}</a>')


def _figure():
    """
    A matplotlib Figure that is not registered with pyplot. Widget callbacks
    can run in a kernel subshell thread; the notebook's inline backend draws
    and closes all *pyplot* figures after each execution, which blanks a pyplot
    figure that a callback is drawing at the same moment.
    """
    from matplotlib.figure import Figure
    return Figure()


def _show(fig):
    """Render a figure to PNG and display it inside an Output widget."""
    from IPython.display import Image, display
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    display(Image(data=buf.getvalue(), format="png"))


def _figure_links(fig, stem):
    """Download links for a figure as PDF and EPS (fonts embedded) and PNG (300 dpi)."""
    from .plotting import paper_style
    links = []
    for ext, mime in (("pdf", "application/pdf"), ("eps", "application/postscript"),
                      ("png", "image/png")):
        buf = io.BytesIO()
        with paper_style():
            fig.savefig(buf, format=ext)
        b64 = base64.b64encode(buf.getvalue()).decode()
        links.append(f'<a download="{stem}.{ext}" href="data:{mime};base64,{b64}" '
                     f'style="margin-right:14px">⬇ figure .{ext}</a>')
    return "".join(links)


def _save_upload(item):
    """Write an uploaded file to a temporary file and return its path."""
    suffix = os.path.splitext(item["name"])[1] or ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
        fh.write(bytes(item["content"]))
        return fh.name


def _read_upload(item, fmt, **kw):
    path = _save_upload(item)
    try:
        spec = Spectrum.read(path, format=fmt, **kw)
    finally:
        os.unlink(path)
    if not spec.name.startswith("desi-"):
        spec.name = os.path.splitext(item["name"])[0]
    return spec


class FantasyApp:
    """
    The widget application. It displays itself in a notebook; afterwards the
    results are available as ``app.spec``, ``app.model``, ``app.result`` and
    ``app.multi``.
    """

    def _ipython_display_(self):
        from IPython.display import display
        display(self.widget)

    def __init__(self):
        import ipywidgets as w
        self.w = w
        self.raw = None
        self.spec = None
        self.model = None
        self.result = None
        self.multi = None
        lay = w.Layout(width="320px")
        style = {"description_width": "130px"}

        # ---------------------------------------------------------- spectrum tab
        self.upload = w.FileUpload(accept=".fits,.fit,.fts,.txt,.dat,.csv,.ascii",
                                   multiple=False, description="Spectrum file")
        self.target = w.Dropdown(description="DESI target", layout=w.Layout(width="520px",
                                 display="none"), style=style)
        self.target.observe(self._on_target, names="value")
        self.redrock = w.FileUpload(accept=".fits", multiple=False,
                                    description="redrock file (optional)",
                                    layout=w.Layout(display="none"))
        self.redrock.observe(self._on_redrock, names="value")
        self._desi_path = None
        self._redrock_path = None
        self.fmt = w.Dropdown(options=[("automatic", "auto"), ("SDSS", "sdss"), ("DESI", "desi"),
                                       ("GAMA", "gama"), ("1D FITS", "fits1d"),
                                       ("text columns", "text")],
                              value="auto", description="Format", layout=lay, style=style)
        self.z = w.FloatText(value=0.0, description="Redshift z", layout=lay, style=style,
                             step=0.001)
        self.ebv = w.FloatText(value=0.0, description="Galactic E(B-V)", layout=lay,
                               style=style, step=0.01)
        self.medium = w.Dropdown(options=["vacuum", "air"], value="vacuum",
                                 description="Wavelengths in", layout=lay, style=style)
        self.host = w.Checkbox(value=False, description="Subtract host galaxy (PCA)",
                               indent=False)
        self.prepare_btn = w.Button(description="Prepare", button_style="primary",
                                    icon="check")
        self.spec_out = w.Output()
        self.upload.observe(self._on_upload, names="value")
        self.prepare_btn.on_click(self._on_prepare)
        spec_tab = w.VBox([
            w.HTML(_HELP["spectrum"]), w.HBox([self.upload, self.fmt]),
            w.HBox([self.target, self.redrock]),
            w.HBox([self.z, self.ebv]), w.HBox([self.medium, self.host]),
            self.prepare_btn, self.spec_out])

        # ------------------------------------------------------------- model tab
        self.preset = w.Dropdown(options=list(presets.PRESETS), description="Model",
                                 layout=lay, style=style)
        self.ncomp = w.IntSlider(value=2, min=1, max=4, description="Broad components",
                                 layout=lay, style=style)
        self.feii = w.Checkbox(value=True, description="FeII template", indent=False)
        self.outflow = w.Checkbox(value=False, description="[OIII]/[NII] outflow wing",
                                  indent=False)
        self.host_model = w.Dropdown(options=[("none", False),
                                              ("stellar templates (σ*, v*)", "stellar"),
                                              ("PCA eigenspectra (FANTASY 1.x)", "pca")],
                                     value=False, description="Host galaxy", layout=lay,
                                     style=style)
        self.tpl_btn = w.Button(description="Download stellar templates (E-MILES, 13 MB)",
                                icon="download", layout=w.Layout(width="330px", display="none"))
        self.tpl_btn.on_click(self._on_download_templates)
        self.profile = w.Dropdown(options=[("Gaussian", "gaussian"), ("Lorentzian", "lorentzian"),
                                           ("Voigt", "voigt"), ("Gauss-Hermite (h3, h4)",
                                                                "gauss-hermite")],
                                  value="gaussian", description="Broad-line profile",
                                  layout=lay, style=style)
        self.extinction = w.Checkbox(value=False, description="Intrinsic AGN reddening E(B-V)",
                                     indent=False)
        self.cont = w.Dropdown(options=[("model default", None), ("power law", "powerlaw"),
                                        ("broken power law", "broken")],
                               value=None, description="Continuum", layout=lay, style=style)
        self.balmer = w.Checkbox(value=True, description="Balmer continuum + high-order lines",
                                 indent=False)
        self.range = w.FloatRangeSlider(value=(4150, 5550), min=1000, max=11000, step=10,
                                        description="Fitted range [A]",
                                        layout=w.Layout(width="520px"), style=style)
        self.par_name = w.Dropdown(description="Parameter", layout=lay, style=style)
        self.par_value = w.FloatText(description="value", layout=w.Layout(width="200px"))
        self.par_min = w.FloatText(description="min", layout=w.Layout(width="200px"))
        self.par_max = w.FloatText(description="max", layout=w.Layout(width="200px"))
        self.par_fixed = w.Checkbox(description="fixed", indent=False)
        self.par_apply = w.Button(description="Apply", icon="pencil")
        self.par_name.observe(self._on_par_select, names="value")
        self.par_apply.on_click(self._on_par_apply)
        self.model_out = w.Output()
        for ctl in (self.preset, self.ncomp, self.feii, self.outflow, self.host_model,
                    self.cont, self.balmer, self.profile, self.extinction):
            ctl.observe(self._on_model_change, names="value")
        advanced = w.Accordion([w.VBox([
            w.HTML("Change a parameter's start value, limits, or fix it."),
            self.par_name, w.HBox([self.par_value, self.par_min, self.par_max]),
            w.HBox([self.par_fixed, self.par_apply])])])
        advanced.set_title(0, "Advanced: edit parameters")
        advanced.selected_index = None
        model_tab = w.VBox([w.HTML(_HELP["model"]), self.preset,
                            w.HBox([self.ncomp, self.profile]),
                            w.HBox([self.cont, self.host_model]), self.tpl_btn,
                            w.HBox([self.feii, self.outflow, self.balmer, self.extinction]),
                            self.range,
                            advanced, self.model_out])

        # --------------------------------------------------------------- fit tab
        self.errors = w.ToggleButtons(options=[("Fast", "covariance"), ("Monte Carlo", "mc"),
                                               ("MCMC", "mcmc")],
                                      description="Errors", style=style)
        self.n_mc = w.IntSlider(value=50, min=20, max=500, step=10, description="MC samples",
                                layout=lay, style=style)
        self.fit_btn = w.Button(description="Fit", button_style="success", icon="play")
        self.fit_btn.on_click(self._on_fit)
        self.status = w.HTML()
        self.fit_out = w.Output()
        fit_tab = w.VBox([w.HTML(_HELP["fit"]), w.HBox([self.errors, self.n_mc]),
                          w.HBox([self.fit_btn, self.status]), self.fit_out])

        # ------------------------------------------------------- multi-epoch tab
        self.uploads = w.FileUpload(accept=".fits,.fit,.fts,.txt,.dat,.csv", multiple=True,
                                    description="Epoch files")
        self.shared = w.Text(value="narrow.*", description="Shared parameters",
                             layout=w.Layout(width="520px"), style=style)
        self.intercal = w.Checkbox(value=True, description="Intercalibrate flux with narrow lines",
                                   indent=False)
        self.multi_btn = w.Button(description="Fit all epochs", button_style="success",
                                  icon="play")
        self.multi_btn.on_click(self._on_multi)
        self.multi_status = w.HTML()
        self.multi_out = w.Output()
        multi_tab = w.VBox([w.HTML(_HELP["multi"]), self.uploads, self.shared, self.intercal,
                            w.HBox([self.multi_btn, self.multi_status]), self.multi_out])

        self.tabs = w.Tab([spec_tab, model_tab, fit_tab, multi_tab])
        for i, t in enumerate(["1 · Spectrum", "2 · Model", "3 · Fit & results",
                               "Multi-epoch"]):
            self.tabs.set_title(i, t)
        self.widget = w.VBox([
            w.HTML("<h3 style='margin:4px 0'>FANTASY · AGN spectral fitting</h3>"), self.tabs])
        self._on_model_change()

    # ----------------------------------------------------------- callbacks
    def _show_error(self, out, exc):
        with out:
            print(f"Problem: {exc}")

    def _on_upload(self, change):
        if not self.upload.value:
            return
        self.spec_out.clear_output()
        from . import desi
        item = self.upload.value[0]
        path = _save_upload(item)
        if self.fmt.value == "desi" or (self.fmt.value == "auto" and desi.is_desi(path)):
            # DESI files hold many targets: keep the file and let the user choose
            self._desi_path = path
            self.target.layout.display = None
            self.redrock.layout.display = None
            self._fill_targets()
            return
        os.unlink(path)
        self._desi_path = None
        self.target.layout.display = "none"
        self.redrock.layout.display = "none"
        try:
            self.raw = _read_upload(item, self.fmt.value)
        except Exception as exc:  # show any reading problem to the user
            self._show_error(self.spec_out, exc)
            return
        self._show_raw()

    def _fill_targets(self):
        from . import desi
        table = desi.targets(self._desi_path, redrock=self._redrock_path)
        options = []
        for row in table.itertuples():
            label = f"{row.TARGETID}"
            if hasattr(row, "Z") and np.isfinite(row.Z):
                label += f"   z = {row.Z:.4f}   {row.SPECTYPE}"
            options.append((label, int(row.TARGETID)))
        self.target.options = options
        if hasattr(table, "SPECTYPE") and "SPECTYPE" in table:
            qso = table[table.SPECTYPE == "QSO"]
            if len(qso):
                self.target.value = int(qso.TARGETID.iloc[0])
        with self.spec_out:
            print(f"DESI file with {len(table)} targets. Choose one above"
                  + ("" if self._redrock_path else "; upload the matching redrock file to "
                     "get redshifts and spectral types."))

    def _on_redrock(self, change):
        if not self.redrock.value or self._desi_path is None:
            return
        self._redrock_path = _save_upload(self.redrock.value[0])
        self.spec_out.clear_output()
        self._fill_targets()
        self._on_target({"new": self.target.value})

    def _on_target(self, change):
        if change["new"] is None or self._desi_path is None:
            return
        from . import desi
        try:
            self.raw = desi.read(self._desi_path, targetid=change["new"],
                                 redrock=self._redrock_path)
        except Exception as exc:
            self._show_error(self.spec_out, exc)
            return
        self._show_raw()

    def _show_raw(self):
        self.z.value = float(self.raw.z or 0.0)
        self.medium.value = self.raw.medium
        with self.spec_out:
            print(self.raw)
            if self.raw.z is None:
                print("No redshift found in the file: please enter it.")
            if self.raw.resolution_curve is not None:
                print("Instrumental resolution varies with wavelength; each line is "
                      "corrected with the resolution at its own wavelength.")

    def _on_prepare(self, _):
        self.spec_out.clear_output()
        if self.raw is None:
            self._show_error(self.spec_out, "upload a spectrum first")
            return
        try:
            s = self.raw.copy(medium=self.medium.value)
            if self.ebv.value:
                s = s.deredden(ebv=self.ebv.value)
            s = s.to_rest_frame(z=self.z.value)
            if self.host.value:
                s = s.subtract_host()
            self.spec = s
        except Exception as exc:
            self._show_error(self.spec_out, exc)
            return
        names = presets.available(self.spec)
        self.preset.options = names or list(presets.PRESETS)
        if "hbeta" in names:
            self.preset.value = "hbeta"
        with self.spec_out:
            print(self.spec)
            print("Models covering this spectrum:", ", ".join(names) or "none (build one in code)")
            _show(self.spec.plot(fig=_figure()).figure)
        self._on_model_change()

    def _build_model(self):
        name = self.preset.value
        func = presets.PRESETS[name]
        kw = {}
        import inspect
        params = inspect.signature(func).parameters
        if "broad_components" in params:
            kw["broad_components"] = self.ncomp.value
        if "feii" in params:
            kw["feii"] = self.feii.value
        if "outflow" in params:
            kw["outflow"] = self.outflow.value
        from . import templates
        stellar_missing = self.host_model.value == "stellar" and \
            "emiles" not in templates.available()
        self.tpl_btn.layout.display = None if stellar_missing else "none"
        if "host" in params:
            kw["host"] = False if stellar_missing else self.host_model.value
        if "broad_profile" in params:
            kw["broad_profile"] = self.profile.value
        if "extinction" in params:
            kw["extinction"] = self.extinction.value
        if "continuum" in params and self.cont.value:
            kw["continuum"] = self.cont.value
        if "balmer" in params:
            kw["balmer"] = self.balmer.value
        return func(**kw)

    def _on_download_templates(self, _):
        from . import templates
        self.tpl_btn.disabled = True
        self.tpl_btn.description = "downloading…"
        try:
            templates.download("emiles")
        except Exception as exc:
            self.tpl_btn.description = f"download failed: {exc}"
            self.tpl_btn.disabled = False
            return
        self._on_model_change()

    def _on_model_change(self, *_):
        self.model = self._build_model()
        lo, hi = self.model.wave_range
        if self.spec is not None:
            lo = max(lo, float(self.spec.wave.min()))
            hi = min(hi, float(self.spec.wave.max()))
        self.range.value = (lo, hi)
        self.par_name.options = [p.name for p in self.model.params if not p.tie]
        self.model_out.clear_output()
        with self.model_out:
            print(self.model)

    def _on_par_select(self, change):
        if not change["new"]:
            return
        p = self.model.params[change["new"]]
        self.par_value.value, self.par_fixed.value = p.value, p.fixed
        self.par_min.value = p.min if np.isfinite(p.min) else -1e30
        self.par_max.value = p.max if np.isfinite(p.max) else 1e30

    def _on_par_apply(self, _):
        p = self.model.params[self.par_name.value]
        try:
            p.set(value=self.par_value.value, min=self.par_min.value, max=self.par_max.value,
                  fixed=self.par_fixed.value)
        except ValueError as exc:
            self._show_error(self.model_out, exc)
            return
        with self.model_out:
            print(f"updated {p.name}: value={p.value:g} min={p.min:g} max={p.max:g} "
                  f"fixed={p.fixed}")

    def _on_fit(self, _):
        from IPython.display import HTML, display
        self.fit_out.clear_output()
        if self.spec is None:
            self.status.value = "<span style='color:#c62828'>Prepare a spectrum first (tab 1).</span>"
            return
        self.status.value = "fitting…"
        self.fit_btn.disabled = True
        try:
            self.result = fit(self.spec, self.model, wave_range=self.range.value,
                              errors=self.errors.value, n_mc=self.n_mc.value, verbose=False)
        except Exception as exc:
            self.status.value = f"<span style='color:#c62828'>{exc}</span>"
            return
        finally:
            self.fit_btn.disabled = False
        r = self.result
        self.status.value = f"done in {r.time:.1f} s"
        with self.fit_out:
            print(r.summary(short=True))
            fig = r.plot(fig=_figure(), label_lines=True)
            _show(fig)
            display(HTML(_figure_links(fig, f"{self.spec.name}_fit")))
            display(HTML("<b>Lines</b>"))
            display(r.lines.round(4))
            display(HTML("<b>Broad lines (all broad components together)</b>"))
            rows = []
            broad = {ln for c in r._line_components() if c.role == "broad" for ln in c.line_names}
            for ln in sorted(broad):
                comps = [c.name for c in r._line_components()
                         if c.role == "broad" and ln in c.line_names]
                rows.append(r.measure(ln, comps))
            if rows:
                import pandas as pd
                display(pd.DataFrame(rows).drop(columns="components").round(3))
            extra = self._physical(r)
            if extra:
                display(HTML(extra))
            if any(c.kind == "host" and hasattr(c, "grid") for c in r.model.components):
                hp = r.host_properties()
                rows = [f"σ<sub>*</sub> = {hp['sigma_star'][0]:.0f} ± {hp['sigma_star'][1]:.0f} km/s",
                        f"v<sub>*</sub> = {hp['v_star'][0]:.0f} ± {hp['v_star'][1]:.0f} km/s",
                        f"light-weighted age = {10 ** hp['log_age'][0]:.2f} Gyr, "
                        f"[M/H] = {hp['metal'][0]:+.2f} ± {hp['metal'][1]:.2f}"]
                frac = [k for k in hp if k.startswith("host_fraction")]
                if frac:
                    rows.append(f"host fraction at {frac[0].split('_')[-1]} Å = "
                                f"{hp[frac[0]][0]:.2f} ± {hp[frac[0]][1]:.2f}")
                if hp.get("warning"):
                    rows.append(f"<span style='color:#c62828'>{hp['warning']}</span>")
                rows.append(f"<small>templates: {hp['templates']} ({hp['cite']})</small>")
                display(HTML("<b>Stellar host</b><br>" + "<br>".join(rows)))
            display(HTML(
                _download_link(r.lines.to_csv(index=False), f"{self.spec.name}_lines.csv", "lines")
                + _download_link(r.params.to_csv(index=False), f"{self.spec.name}_params.csv", "parameters")
                + _download_link(r.to_dataframe().to_csv(index=False), f"{self.spec.name}_model.csv",
                                 "model spectrum")))

    @staticmethod
    def _physical(r):
        """Luminosity and BH mass when the preset supports them."""
        out = []
        for line in ("Hb", "MgII2798", "CIV1549"):
            if not any(line in c.line_names and c.role == "broad" for c in r._line_components()):
                continue
            try:
                rec = r.bh_mass(line)
            except Exception:
                continue
            out.append(f"log M<sub>BH</sub> ({line}) = {rec['log_mbh']:.2f} ± "
                       f"{rec['log_mbh_err']:.2f} (stat) ± {rec['systematic_dex']} (syst) "
                       f"[{rec['recipe']}]")
        return "<br>".join(out)

    def _on_multi(self, _):
        from IPython.display import HTML, display
        self.multi_out.clear_output()
        files = self.uploads.value
        if len(files) < 2:
            self.multi_status.value = "<span style='color:#c62828'>Upload at least two spectra.</span>"
            return
        self.multi_status.value = "reading and fitting…"
        self.multi_btn.disabled = True
        try:
            spectra = []
            for item in files:
                s = _read_upload(item, self.fmt.value)
                z = s.z if s.z is not None else self.z.value
                if self.ebv.value:
                    s = s.deredden(ebv=self.ebv.value)
                spectra.append(s.to_rest_frame(z=z))
            spectra.sort(key=lambda s: s.mjd if s.mjd is not None else 0)
            shared = tuple(p.strip() for p in self.shared.value.split(",") if p.strip())
            self.multi = fit_epochs(spectra, self.model, shared=shared,
                                    intercalibrate=self.intercal.value,
                                    wave_range=self.range.value, verbose=False)
        except Exception as exc:
            self.multi_status.value = f"<span style='color:#c62828'>{exc}</span>"
            return
        finally:
            self.multi_btn.disabled = False
        m = self.multi
        self.multi_status.value = f"done in {m.time:.1f} s"
        lc = m.light_curve()
        with self.multi_out:
            print(m.summary())
            fig = m.plot_light_curves(fig=_figure())
            _show(fig)
            display(HTML(_figure_links(fig, "light_curves")))
            display(lc.round(4))
            display(HTML("<b>Shared parameters</b>"))
            display(m.shared.round(4))
            display(HTML(_download_link(lc.to_csv(index=False), "light_curve.csv", "light curve")))


def gui():
    """Show the FANTASY point-and-click interface in a Jupyter notebook."""
    try:
        import ipywidgets  # noqa: F401
    except ImportError:
        raise ImportError('the GUI needs ipywidgets: pip install "fantasy_agn[gui]"') from None
    return FantasyApp()
