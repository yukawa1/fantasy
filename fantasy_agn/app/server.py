"""
A small local web server for the FANTASY app (standard library only). It
listens on 127.0.0.1, serves the page in ``static/`` and a JSON API:

    GET  /api/state                  session state (cells, steps, results)
    GET  /api/update                 a newer FANTASY on PyPI? (?force=1 asks again); POST {version} installs it
    POST /api/exec                   JSON {code, index}: run a code cell (index: rerun that cell)
    POST /api/model                  JSON {action: preset|add|remove|split|params|save|load|upload, ...}: code for the model editor
                                      (save {name, folder, overwrite}, load {name | path}, upload {name, text};
                                      the library is ~/.fantasy_agn/models, models saved elsewhere are remembered)
    GET  /api/models                 the saved models; GET /api/model.json the current model as JSON
    GET  /api/docs                   documentation pages (guides and API reference)
    GET  /api/lines                  the line list
    GET  /api/linesets               the FANTASY 1 line lists (coronal, [Fe II], ...)
    POST /api/upload                 body = a file (?filename=&batch=): keep it, return its path
    GET  /api/lightcurve.csv
    GET  /api/sample.csv, /api/sample_fig.png?k=N&kind=fit|line
    GET  /api/session.py             the session as a Python script
    POST /api/reset
    POST /api/stop                   stop the running cell (worker processes are ended)
    POST /api/restore                JSON {which: last|previous}: open the auto-saved session
                                      (saved after every run to ~/.fantasy_agn/sessions, see FANTASY_AUTOSAVE)
    GET  /api/project.fantasy        the whole session (cells, spectra, models, fits) in one file
    POST /api/project                body = a .fantasy project file: open it
    GET  /api/session.ipynb          the session as a Jupyter notebook
    GET  /api/report.html            one-file report of the current fit (figure, numbers, checks, settings)

The step-by-step workflow (page ``/``):

    POST /api/wf/read                JSON {mode, paths | pattern | targetid | sdss, group}
    POST /api/wf/prepare             JSON {z, z_source, extinction, ebv, ra, dec, wmin, wmax, masks, ...}
    POST /api/wf/suggest             classify the reference spectrum (recommended model)
    POST /api/wf/fit                 JSON {epochs, reference, shared, errors, n_jobs, ...}
    POST /api/wf/addlines            JSON {lines, refit}: add lines found by fa.suggest_lines and refit (state: S.wf.linescan)
    GET  /api/wf/preview?k=&kind=raw|prepared, /api/wf/guess?k=, /api/wf/result?k=
    GET  /api/wf/degeneracy?k=          correlations of the free parameters of fit k (Results › Degeneracies)
    GET  /api/wf/gallery                candidate models fitted on the spectrum (Model › gallery)
    GET  /api/wf/mbtypes, /api/wf/mblines   component types and the line list for the model builder
    POST /api/mb/decompose|build|preview    JSON {source} | {state}: the model builder (answers with data)
    POST /api/mb/gallery, /api/mb/fit       fit the candidates, or a model of the builder, in the background
    GET  /api/wf/atlas                  every fit of a sample, or every epoch, as a small card (Results › Atlas)
    GET  /api/wf/lagpair?a=&b=          lag of light curve b behind a by every method, with its curve (Results › Light curves)
    GET  /api/wf/classify?k=&stage=, /api/wf/classify.png?k=   (stage: final = from the fit, or preliminary)
    GET  /api/wf/lines.csv, /api/wf/lines.fits, /api/wf/plots.pdf, /api/wf/components.csv?k=
    GET  /api/wf/classify.<png|pdf>?k=&kind=, /api/wf/corner.<png|pdf>?k=&params=,
         /api/wf/line_list?k=, /api/wf/linegrid.<png|pdf|eps|svg>?k=&lines=Hb,Ha&mode=grid|overlay&velocity=&residuals=&info=&others=&mark_fwhm=&window=&ncols=&width=
         /api/wf/figure_code?<figure options>, /api/wf/lines_code?lines=&<line options>: the Python code of the figure
         /api/wf/figure.<png|pdf|eps>?k=&xmin=&xmax=&ymin=&ymax=&title=&residuals=&components=&group=
                                      &label_lines=&width=&fontsize=&legend=  (dl=1: as a download)

The app runs Python code, so the API only answers requests that carry the
token created when the server starts (it is put into the page) and that are
addressed to 127.0.0.1/localhost (against DNS rebinding); other web pages open
in the browser cannot use it.
"""
import html
import json
import math
import mimetypes
import os
import secrets
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .session import Session

STATIC = os.path.join(os.path.dirname(__file__), "static")
# the bundled Seyfert 1 galaxy of the opening screen, read by its "Fit this galaxy yourself" button
EXAMPLE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "examples", "spec-0354-51792-0617.fits")


def _versioned(html):
    """Add the modification time to the stylesheet and script links, so a changed file is never served from the cache."""
    for name in ("wf.css", "wf.js", "mb.js", "hero.js"):
        v = int(os.path.getmtime(os.path.join(STATIC, name)))
        html = html.replace(f'"/static/{name}"', f'"/static/{name}?v={v}"')
    return html


def _options(opts):
    """Optional parts of the presets that the interface can switch on."""
    out = {k: bool(opts.get(k)) for k in ("coronal", "oi6046", "red") if k in opts}
    if opts.get("host"):
        out["host"] = str(opts["host"])
    if opts.get("continuum"):
        out["continuum"] = str(opts["continuum"])
    if opts.get("stratified"):
        out["broad_profile"] = "stratified"         # radially stratified BLR for the broad lines
    return out


def _dpi(value, download):
    """Resolution of a PNG: 150 for the preview, 300 (or 600) when it is downloaded."""
    if not download:
        return None
    return 600 if str(value) == "600" else 300


def _finite(obj):
    """JSON has no inf/nan: replace them with None (null)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):     # numpy scalars
        return _finite(obj.item())
    return obj
MAX_UPLOAD = 200 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    session = None              # set by make_server
    token = None
    _docs = None
    server_version = "FANTASY"

    def log_message(self, fmt, *args):          # quiet: only errors
        if len(args) > 1 and str(args[1])[:1] in "45":
            super().log_message(fmt, *args)

    # ------------------------------------------------------------ helpers
    def _send(self, body, ctype="application/json", status=200, filename=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(_finite(body), default=float, allow_nan=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def _error(self, msg, status=400):
        self._send({"error": msg}, status=status)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            raise ValueError("file too large")
        return self.rfile.read(n) if n else b""

    def _host_ok(self):
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        return host in ("127.0.0.1", "localhost", "::1")

    def _authorized(self, query):
        given = self.headers.get("X-Fantasy-Token") or query.get("t", [None])[0]
        return given is not None and secrets.compare_digest(given, self.token)

    # -------------------------------------------------------------- routes
    def do_GET(self):
        if not self._host_ok():
            return self._error("forbidden", 403)
        url = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url.query)
        s = self.session
        if url.path in ("/", "/index.html"):
            return self._index()
        if url.path.startswith("/static/"):
            return self._static(url.path[len("/static/"):])
        if not url.path.startswith("/api/"):
            return self._error("not found", 404)
        if not self._authorized(q):
            return self._error("forbidden", 403)
        if url.path == "/api/state":
            return self._send(s.state())
        if url.path == "/api/update":                 # a newer release on PyPI? (cached for a day)
            from .. import update
            return self._send(update.check(force=q.get("force", ["0"])[0] == "1"))
        if url.path == "/api/models":
            return self._send({"models": s.saved_models(), "folders": s.model_folders()})
        if url.path == "/api/model.json":
            try:
                return self._send(s.model_json(), "application/json", filename="model.json")
            except ValueError as exc:
                return self._error(str(exc), 404)
        if url.path == "/api/docs":
            if Handler._docs is None:
                from .docs import all_pages
                Handler._docs = all_pages()
            return self._send(Handler._docs)
        if url.path == "/api/linesets":
            from ..lines import LINE_SETS, line_set
            return self._send([{"key": k, "label": lab, "lines": line_set(k)} for k, (lab, _) in LINE_SETS.items()])
        if url.path == "/api/lines":
            from ..lines import line_table
            t = line_table()
            return self._send([{"name": n, "wave": float(r.vacuum), "ion": r.ion, "type": r.type,
                                "series": r.series} for n, r in t.iterrows()])
        if url.path == "/api/sample.csv":
            csv = s.sample_csv()
            if csv is None:
                return self._error("no sample fitted yet", 404)
            return self._send(csv, "text/csv", filename="fantasy_catalog.csv")
        if url.path == "/api/sample_fig.png":
            try:
                data = s.sample_figure(q.get("k", ["0"])[0], q.get("kind", ["fit"])[0])
            except ValueError as exc:
                return self._error(str(exc), 404)
            if data is None:
                return self._error("no sample fitted yet", 404)
            return self._send(data, "image/png")
        if url.path == "/api/lightcurve.csv":
            csv = s.light_curve_csv()
            if csv is None:
                return self._error("no multi-epoch fit yet", 404)
            return self._send(csv, "text/csv", filename="light_curve.csv")
        if url.path == "/api/project.fantasy":
            name = (s.spec.name if s.spec is not None else "session") + ".fantasy"
            return self._send(s.save_project(), "application/octet-stream", filename=name)
        if url.path.startswith("/api/wf/"):
            return self._wf_get(url.path[len("/api/wf/"):], q)
        if url.path == "/api/session.ipynb":
            return self._send(s.notebook(), "application/x-ipynb+json", filename="fantasy_session.ipynb")
        if url.path == "/api/report.html":
            from .report import build
            try:
                page = build(s)
            except ValueError as exc:
                return self._error(str(exc), 404)
            name = (s.spec.name if s.spec is not None else "fit").replace('"', "") + "_report.html"
            return self._send(page, "text/html; charset=utf-8", filename=name)
        if url.path == "/api/session.py":
            return self._send(s.script(), "text/x-python", filename="fantasy_session.py")
        return self._error("not found", 404)

    def do_POST(self):
        if not self._host_ok():
            return self._error("forbidden", 403)
        url = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(url.query)
        if not self._authorized(query):
            return self._error("forbidden", 403)
        q = {k: v[0] for k, v in query.items()}
        s = self.session
        try:
            if url.path == "/api/update":             # install the new release; used after a restart
                from .. import update
                opts = json.loads(self._body() or b"{}")
                ok, output = update.install(opts.get("version"))
                return self._send({"ok": ok, "output": output})
            if url.path == "/api/exec":
                opts = json.loads(self._body() or b"{}")
                s.run_cell(opts.get("code", ""), opts.get("index"))
            elif url.path == "/api/model":
                opts = json.loads(self._body() or b"{}")
                action = opts.get("action")
                if action == "preset":
                    code = s.preset_code(opts.get("model", "optical"), opts.get("broad_components", 2),
                                         opts.get("outflow", True), **_options(opts))
                    s.run_step(code, "model = fa.presets.")      # a new choice replaces the last one
                    return self._send(s.state())
                elif action == "add":
                    code = s.add_code(opts.get("component") or {})
                elif action == "split":
                    code = s.split_code(opts.get("component", ""), opts.get("lines") or [], opts.get("name", ""))
                elif action == "remove":
                    code = s.remove_code(opts.get("name", ""))
                elif action == "params":
                    code = s.params_code(opts.get("changes") or [])
                elif action == "tie":
                    code = s.tie_kinematics_code(opts.get("component", ""), opts.get("to", ""))
                elif action == "untie":
                    code = s.untie_code(opts.get("component", ""))
                elif action == "profile":
                    code = s.profile_code(opts.get("component", ""), opts.get("profile", ""))
                elif action == "save":
                    try:
                        code = s.save_model_code(opts.get("name"), opts.get("folder"), bool(opts.get("overwrite")))
                    except FileExistsError as exc:
                        return self._send({"exists": str(exc)})
                    s.run_cell(code)
                    return self._send(s.state())
                elif action in ("load", "upload"):
                    if action == "upload":
                        code = s.load_model_code(s.store_model(opts.get("name"), opts.get("text", "")))
                    else:
                        code = s.load_model_code(opts.get("name"), opts.get("path"))
                    s.run_step(code, "model = fa.")       # replaces the last model choice
                    return self._send(s.state())
                elif action in ("recipe", "presetkw"):        # Use this model, from the gallery or the builder
                    with s.lock:
                        if s.busy:
                            raise RuntimeError("a cell is still running")
                        code = (s.recipe_code(opts.get("state") or {}) if action == "recipe"
                                else s.preset_kw_code(str(opts.get("name", "")), opts.get("kw") or {}))
                        s.run_step(code, "model = fa.")
                    return self._send(s.state())
                elif action == "empty":
                    s.run_step(s.empty_model_code(), "model = fa.")       # replaces a preset chosen just before
                    return self._send(s.state())
                else:
                    raise ValueError("unknown model action")
                s.run_cell(code)
            elif url.path == "/api/upload":
                path = s.save_upload(self._body(), q.get("filename"), q.get("batch", "batch"))
                return self._send({"path": path})
            elif url.path.startswith("/api/mb/"):
                return self._send(self._mb_post(url.path[len("/api/mb/"):], json.loads(self._body() or b"{}")))
            elif url.path.startswith("/api/wf/"):
                self._wf_post(url.path[len("/api/wf/"):], json.loads(self._body() or b"{}"))
            elif url.path == "/api/project":
                s.load_project(self._body())
            elif url.path == "/api/reset":
                s.reset()
            elif url.path == "/api/stop":
                s.stop()
            elif url.path == "/api/restore":
                opts = json.loads(self._body() or b"{}")
                s.restore(opts.get("which", "last"))
            else:
                return self._error("not found", 404)
        except Exception as exc:
            return self._error(str(exc), 409)
        return self._send(s.state())

    # ------------------------------------------------------------- workflow
    def _wf_get(self, name, q):
        s = self.session
        one = lambda key, default="0": q.get(key, [default])[0]
        try:
            if name == "preview":
                return self._send(s.wf_preview(one("k"), one("kind", "prepared")) or {})
            if name == "gallery":
                return self._send(s.wf_gallery())
            if name == "mbtypes":
                return self._send(s.mb_types())
            if name == "mblines":
                return self._send(s.mb_lines())
            if name == "guess":
                return self._send(s.wf_guess(one("k", "")) or {})
            if name == "degeneracy":
                return self._send(s.wf_degeneracy(one("k")) or {})
            if name == "atlas":
                return self._send(s.wf_atlas() or {})
            if name == "result":
                return self._send(s.wf_result(one("k")) or {})
            if name == "classify":
                return self._send(s.classify_state(one("k"), one("stage", "final")) or {})
            if name == "tie_check":
                return self._send(s.wf_tie_check(one("component", "")))
            if name == "result_more":
                return self._send(s.wf_result_more(one("k"), one("part", "key")) or {})
            if name == "lightcurve":
                return self._send(s.wf_lightcurve() or {})
            if name == "lagpair":
                return self._send(s.wf_lagpair(one("a", ""), one("b", "")) or {})
            if name == "reverb.png":
                data = s.wf_reverb_figure()
                return self._send(data, "image/png") if data else self._error("no reverberation yet", 404)
            if name in ("compare.png", "compare.pdf"):
                fmt = name.rsplit(".", 1)[1]
                data = s.wf_compare_figure(fmt)
                if not data:
                    return self._error("no comparison yet", 404)
                return self._send(data, "image/png" if fmt == "png" else "application/pdf")
            if name == "result.png":
                data = s.wf_result_figure(one("k"))
                return self._send(data, "image/png") if data else self._error("no fit yet", 404)
            if name == "sample":
                return self._send(s.sample_view(one("k")) or {})
            if name == "catalog_row":
                return self._send(s.wf_catalog_row(one("k")) or {})
            if name == "sky":
                return self._send(s.wf_sky(one("k"), one("fov", "60")))
            if name == "sample_points":
                return self._send(s.sample_points() or {})
            if name == "mainseq":
                return self._send(s.wf_main_sequence() or {})
            if name == "line_list":
                return self._send(s.wf_line_list(one("k")) or {})
            if name == "correlations_data":
                return self._send(s.wf_correlations_data() or {})
            if name.split(".")[0] in ("correlations", "relation", "correlations_code", "relation_code"):
                lag = float(one("lag", "0") or 0)
                off = [int(k) for k in one("off", "").split(",") if k]
                x, y, cols = one("x", ""), one("y", ""), [c for c in one("cols", "").split(",") if c]
                if name == "correlations_code":
                    return self._send({"code": s.wf_correlations_figure_code(cols, lag, off)})
                if name == "relation_code":
                    return self._send({"code": s.wf_relation_code(x, y, lag, off)})
                base, fmt = name.rsplit(".", 1)
                types = {"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml"}
                dl = one("dl", "") == "1"
                dpi = _dpi(one("dpi", ""), dl)
                data = (s.wf_correlations_figure(cols, fmt, dpi, lag, off) if base == "correlations"
                        else s.wf_relation_figure(x, y, fmt, dpi, lag, off))
                if not data:
                    return self._error("no epoch table yet", 404)
                return self._send(data, types[fmt], filename=f"{base}.{fmt}" if dl else None)
            if name == "figure_code":
                opts = {key: v[0] for key, v in q.items() if key not in ("k", "t", "dl", "v")}
                return self._send({"code": s.wf_figure_code(one("k"), opts)})
            if name == "lines_code":
                opts = {key: v[0] for key, v in q.items() if key not in ("k", "t", "dl", "v", "lines")}
                lines = [ln for ln in one("lines", "").split(",") if ln]
                return self._send({"code": s.wf_lines_code(one("k"), lines, opts)})
            ctypes = {"png": "image/png", "pdf": "application/pdf", "eps": "application/postscript",
                      "svg": "image/svg+xml"}
            stem_fmt = name.rsplit(".", 1)
            if len(stem_fmt) == 2 and stem_fmt[1] in ctypes and stem_fmt[0] in ("classify", "corner", "figure", "mainseq", "linegrid", "redshift"):
                base, fmt = stem_fmt
                dl = one("dl", "") == "1"
                if base == "classify":
                    data = s.classify_figure(one("k"), one("kind", "steps"), fmt, one("stage", "final"))
                    if not data:
                        return self._error("not classified yet", 404)
                    fn = f"classification_{one('kind', 'steps')}.{fmt}"
                elif base == "corner":
                    params = [p for p in one("params", "").split(",") if p]
                    data, fn = s.wf_corner(one("k"), params or None, fmt), f"corner_{one('k')}.{fmt}"
                elif base == "mainseq":
                    data, fn = s.wf_main_sequence_figure(one("k"), fmt), f"main_sequence.{fmt}"
                elif base == "redshift":
                    data, fn = s.wf_redshift_figure(fmt), f"redshift.{fmt}"
                    if not data:
                        return self._error("no redshift search yet", 404)
                elif base == "linegrid":
                    opts = {key: v[0] for key, v in q.items() if key not in ("k", "t", "dl", "v", "lines", "dpi")}
                    lines = [ln for ln in one("lines", "").split(",") if ln]
                    data, fn = (s.wf_lines_figure(one("k"), lines, opts, fmt, dpi=_dpi(one("dpi", ""), dl)),
                                f"lines_{one('k')}.{fmt}")
                elif base == "figure":
                    opts = {key: v[0] for key, v in q.items() if key not in ("k", "t", "dl", "v", "dpi")}
                    data, fn = (s.wf_custom_figure(one("k"), opts, fmt, dpi=_dpi(one("dpi", ""), dl)),
                                f"fit_{one('k')}.{fmt}")
                return self._send(data, ctypes[fmt], filename=fn if dl else None)
            stem = (s.wf_results() or [("fantasy",)])[0][0]
            if name == "lines.csv":
                data = s.wf_lines_csv()
                return self._send(data, "text/csv", filename=f"{stem}_lines.csv") if data else self._error("no fit yet", 404)
            if name == "components.csv":
                k = int(one("k"))
                return self._send(s.wf_components_csv(k), "text/csv", filename=f"{stem}_model_{k}.csv")
            if name == "lines.fits":
                data = s.wf_lines_fits()
                return (self._send(data, "application/fits", filename=f"{stem}_lines.fits") if data
                        else self._error("no fit yet", 404))
            if name == "plots.pdf":
                data = s.wf_plots_pdf()
                return (self._send(data, "application/pdf", filename=f"{stem}_fits.pdf") if data
                        else self._error("no fit yet", 404))
        except (ValueError, KeyError) as exc:
            return self._error(str(exc), 404)
        return self._error("not found", 404)

    def _mb_post(self, name, opts):
        """Model builder and gallery: queries that answer with data, not with the session state."""
        s = self.session
        if name == "decompose":
            return s.mb_decompose(opts.get("source"))
        if name == "build":
            return s.mb_build(opts.get("state") or {})
        if name == "preview":
            return s.mb_preview(opts.get("state") or {})
        if name == "gallery":
            s.gallery_run(opts.get("preset"), opts.get("kw"))
            return s.wf_gallery()
        if name == "fit":
            key = s.gallery_fit(opts.get("state") or {}, str(opts.get("title", ""))[:80], str(opts.get("change", ""))[:300])
            return {**s.wf_gallery(), "key": key}
        raise ValueError("unknown model builder action")

    def _wf_post(self, name, opts):
        s = self.session
        if name == "mode":
            mode = opts.get("mode")
            from .steps import MODES
            if mode not in MODES:
                raise ValueError("unknown mode")
            with s.lock:
                if s.busy:
                    raise RuntimeError("a cell is still running")
                if s.raw:
                    s.run_step(s.wf_reselect_code(mode, opts.get("loaded_id", s._loaded_id()), opts.get("group", "auto")))
                else:
                    s.wf["mode"] = mode
        elif name == "read":
            s.run_step(s.wf_read_code(mode=opts.get("mode", "single"), paths=opts.get("paths"),
                                      pattern=opts.get("pattern"), targetid=opts.get("targetid"),
                                      sdss=opts.get("sdss"), group=opts.get("group", "auto"),
                                      format=opts.get("format", "auto")))
        elif name == "remove":
            with s.lock:
                s.run_step(s.wf_remove_code(opts.get("indices"), opts.get("loaded_id")))
        elif name == "prepare":
            keys = ("z", "z_source", "rest_frame", "extinction", "ebv", "ra", "dec", "wmin", "wmax",
                    "telluric", "sky_lines", "absorption", "masks", "apply_to", "flux_unit", "resolution",
                    "resolution_unit")
            s.run_step(s.wf_prepare_code(**{k: opts[k] for k in keys if k in opts}), "# prepare")
        elif name == "suggest":
            sp = s.spectra
            if sp is None:
                raise ValueError("prepare the spectra first")
            k = s.wf_reference()
            if len(sp) > 1:     # every epoch, for the diagrams; the reference one recommends the model
                s.run_cell(f"epoch_classes = [fa.classify(s) for s in spectra]\nc = epoch_classes[{k}]\n"
                           "[(s.name, x.label) for s, x in zip(spectra, epoch_classes)]")
            else:
                s.run_cell("c = fa.classify(spectra[0])\nc")
        elif name == "residuals":
            s.run_step(s.wf_residuals_code(scope=opts.get("scope", "all"), nsigma=opts.get("nsigma", 4.0)),
                       "# residuals")
        elif name == "addlines":
            s.run_cell(s.wf_addlines_code(opts.get("lines") or [], refit=bool(opts.get("refit", True))))
        elif name == "robust":
            s.run_step(s.wf_robust_code(), "# robustness")
        elif name == "find_redshift":
            s.run_step(s.wf_find_redshift_code(opts.get("k", 0)), "# find redshift")
        elif name == "sharing":
            s.set_sharing(opts.get("component"), opts.get("part"), opts.get("shared"), reset=bool(opts.get("reset")))
        elif name == "calcheck":
            s.run_step(s.wf_calcheck_code(), "# calibration check")
        elif name == "simcheck":
            s.run_step(s.wf_simcheck_code(n=opts.get("n", 60)), "# simulation check")
        elif name == "prune_feii":
            s.run_step(s.wf_prune_feii_code(nsigma=opts.get("nsigma", 2.0)), "# prune Fe II")
        elif name == "prune_feii_undo":
            s.run_cell(s.wf_prune_feii_undo_code())
        elif name == "robust_apply":
            s.run_cell(s.wf_robust_apply_code())
        elif name == "pin":
            s.run_cell(s.wf_pin_code(opts.get("label", "")))
        elif name == "correlations":
            s.run_step(s.wf_correlations_code(), "# correlations")
        elif name == "catalogs":
            s.run_step(s.wf_catalogs_code(), "# catalogs")
        elif name == "pin_alternative":
            s.run_cell(s.wf_pin_alternative_code(opts.get("i", -1)))
        elif name == "unpin":
            s.run_cell(s.wf_unpin_code(opts.get("label", "")))
        elif name == "compare":
            s.run_step(s.wf_compare_code(opts.get("label", "")), "# compare fits")
        elif name == "reverb":
            s.run_step(s.wf_reverb_code(line=opts.get("line", "Hb"), n_mc=opts.get("n_mc", 1000),
                                        methods=tuple(opts.get("methods") or ("iccf", "vn", "drw")),
                                        detrend=bool(opts.get("detrend")), mjd_min=opts.get("mjd_min"),
                                        mjd_max=opts.get("mjd_max"), primary=opts.get("primary"), window=opts.get("window")), "# reverberation")
        elif name == "fit":
            keys = ("epochs", "reference", "n_jobs", "shared", "errors", "scale_errors", "rescale_errors",
                    "calibration_degree", "host_per_epoch", "uncertainty", "absorbers", "sample_model",
                    "shared_shape", "prune_feii", "trial", "from_trial", "wavelength_offsets",
                    "epoch_broadening")
            code = s.wf_fit_code(**{k: opts[k] for k in keys if k in opts})
            # the trial fit of one epoch keeps its own cell: the fit of all epochs starts from it
            marker = "# trial fit" if code.startswith("# trial fit") else "# fit"
            if s.ns.get("resid") is not None:            # residuals of the previous fit
                code = code.replace(marker + "\n", marker + "\nresid = None\n", 1)
            s.run_step(code, marker)
        else:
            raise ValueError("unknown workflow action")

    def _index(self, page="index.html"):
        with open(os.path.join(STATIC, page), encoding="utf-8") as fh:
            page = _versioned(fh.read().replace("__FANTASY_TOKEN__", self.token)
                              .replace("__FANTASY_EXAMPLE__", html.escape(EXAMPLE)))
        self._send(page, "text/html; charset=utf-8")

    def _static(self, name):
        path = os.path.normpath(os.path.join(STATIC, name))
        if not path.startswith(STATIC + os.sep) or not os.path.isfile(path):
            return self._error("not found", 404)
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as fh:
            self._send(fh.read(), ctype)


def make_server(port=8780, host="127.0.0.1"):
    """A server bound to the local machine only, with a fresh session and token."""
    Handler.session = Session()
    Handler.token = secrets.token_urlsafe(24)
    return ThreadingHTTPServer((host, port), Handler)
