"""
A small local web server for the FANTASY app (standard library only). It
listens on 127.0.0.1, serves the page in ``static/`` and a JSON API:

    GET  /api/state                  session state (cells, steps, results)
    POST /api/exec                   JSON {code, index}: run a code cell (index: rerun that cell)
    POST /api/read                   JSON {path | targetid, z} or the file itself
                                     (body = file, ?filename=...&z=...)
    POST /api/classify
    POST /api/fit                    JSON {errors} (fits `model`; without one: {model, broad_components, outflow})
    POST /api/model                  JSON {action: preset|add|remove|params, ...}: code for the model editor
    GET  /api/docs                   documentation pages (guides and API reference)
    GET  /api/lines                  the line list
    GET  /api/plotdata               arrays for the interactive plot
    GET  /api/data                   tables for the Data tab
    POST /api/upload                 body = a file (?filename=&batch=): keep it, return its path
    POST /api/multiepoch             JSON: joint fit of several epochs (code cell)
    GET  /api/lightcurve.csv
    GET  /api/epoch.<png|pdf>?k=N     the fit of epoch N of the multi-epoch result
    GET  /api/epochs.pdf             all epoch fits, one page each
    GET  /api/line.<png|pdf>?line=Hb one line of the current fit with its components
    POST /api/templates              JSON {name}: download a stellar template set (as a code cell)
    POST /api/sample                 JSON: classify and fit many spectra (fa.fit_sample, code cell)
    POST /api/reverberation          JSON {line}: lag, rms spectrum and mass of the multi-epoch fit
    GET  /api/sample.csv, /api/sample_fig.png?k=N&kind=fit|line
    POST /api/reset
    GET  /api/figure.<png|pdf|eps>   the current figure (?dl=1 for publication quality)
    GET  /api/lines.csv, /api/session.py
    GET  /api/project.fantasy        the whole session (cells, spectra, models, fits) in one file
    POST /api/project                body = a .fantasy project file: open it

The app runs Python code, so the API only answers requests that carry the
token created when the server starts (it is put into the page) and that are
addressed to 127.0.0.1/localhost (against DNS rebinding); other web pages open
in the browser cannot use it.
"""
import json
import math
import mimetypes
import os
import secrets
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .session import Session

STATIC = os.path.join(os.path.dirname(__file__), "static")


def _options(opts):
    """Optional parts of the presets that the interface can switch on."""
    out = {k: bool(opts.get(k)) for k in ("coronal", "oi6046", "red") if k in opts}
    if opts.get("host"):
        out["host"] = str(opts["host"])
    return out


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
        if url.path == "/api/plotdata":
            return self._send(s.plot_data() or {})
        if url.path == "/api/data":
            return self._send(s.data_tables())
        if url.path == "/api/docs":
            if Handler._docs is None:
                from .docs import all_pages
                Handler._docs = all_pages()
            return self._send(Handler._docs)
        if url.path == "/api/lines":
            from ..lines import LINES
            return self._send([{"name": n, "wave": w} for n, w in sorted(LINES.items(), key=lambda kv: kv[1])])
        if url.path.startswith("/api/figure."):
            fmt = url.path.rsplit(".", 1)[1]
            if fmt not in ("png", "pdf", "eps"):
                return self._error("unknown format", 404)
            dl = "dl" in q
            data = s.figure_bytes(fmt, dpi=300 if dl else (220 if "hd" in q else 110))
            if data is None:
                return self._error("no figure yet", 404)
            ctype = {"png": "image/png", "pdf": "application/pdf",
                     "eps": "application/postscript"}[fmt]
            return self._send(data, ctype, filename=f"{s.figure_stem}.{fmt}" if dl else None)
        if url.path.startswith("/api/epoch."):
            fmt = url.path.rsplit(".", 1)[1]
            if fmt not in ("png", "pdf"):
                return self._error("unknown format", 404)
            try:
                data = s.epoch_figure(q.get("k", ["0"])[0], fmt,
                                      dpi=300 if "dl" in q else (220 if "hd" in q else 110))
            except ValueError as exc:
                return self._error(str(exc), 404)
            if data is None:
                return self._error("no multi-epoch fit yet", 404)
            return self._send(data, "image/png" if fmt == "png" else "application/pdf",
                              filename=f"epoch_{q.get('k', ['0'])[0]}.{fmt}" if "dl" in q else None)
        if url.path.startswith("/api/line."):
            fmt = url.path.rsplit(".", 1)[1]
            if fmt not in ("png", "pdf"):
                return self._error("unknown format", 404)
            line = q.get("line", [None])[0]
            try:
                data = s.line_figure(line, fmt, dpi=300 if "dl" in q else (220 if "hd" in q else 110))
            except ValueError as exc:
                return self._error(str(exc), 404)
            if data is None:
                return self._error("no fit yet", 404)
            safe = "".join(ch for ch in str(line) if ch.isalnum()) or "line"
            return self._send(data, "image/png" if fmt == "png" else "application/pdf",
                              filename=f"{s.figure_stem}_{safe}.{fmt}" if "dl" in q else None)
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
        if url.path == "/api/epochs.pdf":
            data = s.epochs_pdf()
            if data is None:
                return self._error("no multi-epoch fit yet", 404)
            return self._send(data, "application/pdf", filename="epoch_fits.pdf")
        if url.path == "/api/lines.csv":
            csv = s.lines_csv()
            if csv is None:
                return self._error("no fit yet", 404)
            return self._send(csv, "text/csv", filename=f"{s.spec.name}_lines.csv")
        if url.path == "/api/lightcurve.csv":
            csv = s.light_curve_csv()
            if csv is None:
                return self._error("no multi-epoch fit yet", 404)
            return self._send(csv, "text/csv", filename="light_curve.csv")
        if url.path == "/api/project.fantasy":
            name = (s.spec.name if s.spec is not None else "session") + ".fantasy"
            return self._send(s.save_project(), "application/octet-stream", filename=name)
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
            if url.path == "/api/exec":
                opts = json.loads(self._body() or b"{}")
                s.run_cell(opts.get("code", ""), opts.get("index"))
            elif url.path == "/api/read":
                if self.headers.get("Content-Type", "").startswith("application/json"):
                    opts = json.loads(self._body() or b"{}")
                    code = s.read_code(path=opts.get("path"), targetid=opts.get("targetid"),
                                       z=opts.get("z"), deredden=opts.get("deredden", True),
                                       ebv=opts.get("ebv"), rest_frame=opts.get("rest_frame", True),
                                       mask_absorption=bool(opts.get("mask_absorption")))
                else:
                    code = s.read_code(data=self._body(), filename=q.get("filename"), z=q.get("z"),
                                       deredden=q.get("deredden", "1") == "1", ebv=q.get("ebv"),
                                       rest_frame=q.get("rest_frame", "1") == "1",
                                       mask_absorption=q.get("mask_absorption", "0") == "1")
                s.run_cell(code)
            elif url.path == "/api/classify":
                s.run_cell(s.classify_code())
            elif url.path == "/api/fit":
                opts = json.loads(self._body() or b"{}")
                errors = opts.get("errors", "covariance")
                if errors not in ("covariance", "mc"):
                    raise ValueError("errors must be covariance or mc")
                if opts.get("uncertainty"):
                    s.run_cell(s.uncertainty_code(model=opts.get("model", "optical"),
                                                  broad_components=opts.get("broad_components", 2),
                                                  outflow=opts.get("outflow", True), **_options(opts)))
                elif s.model is not None:
                    s.run_cell(s.fit_model_code(errors, absorbers=bool(opts.get("absorbers"))))
                else:
                    s.run_cell(s.fit_code(model=opts.get("model", "optical"),
                                          broad_components=opts.get("broad_components", 2),
                                          outflow=opts.get("outflow", True), errors=errors,
                                          absorbers=bool(opts.get("absorbers")), **_options(opts)))
            elif url.path == "/api/model":
                opts = json.loads(self._body() or b"{}")
                action = opts.get("action")
                if action == "preset":
                    code = s.preset_code(opts.get("model", "optical"), opts.get("broad_components", 2),
                                         opts.get("outflow", True), **_options(opts))
                elif action == "add":
                    code = s.add_code(opts.get("component") or {})
                elif action == "remove":
                    code = s.remove_code(opts.get("name", ""))
                elif action == "params":
                    code = s.params_code(opts.get("changes") or [])
                else:
                    raise ValueError("unknown model action")
                s.run_cell(code)
            elif url.path == "/api/upload":
                path = s.save_upload(self._body(), q.get("filename"), q.get("batch", "batch"))
                return self._send({"path": path})
            elif url.path == "/api/multiepoch":
                opts = json.loads(self._body() or b"{}")
                extra = _options(opts)
                extra.pop("host", None) if opts.get("model") == "current" else None
                s.run_cell(s.multiepoch_code(
                    paths=opts.get("paths"), pattern=opts.get("pattern"), model=opts.get("model", "current"),
                    broad_components=opts.get("broad_components", 2), outflow=opts.get("outflow", True),
                    shared=opts.get("shared", "narrow.*"), intercalibrate=opts.get("intercalibrate", True),
                    n_jobs=opts.get("n_jobs", -1), deredden=opts.get("deredden", True), ebv=opts.get("ebv"),
                    rest_frame=opts.get("rest_frame", True), z=opts.get("z"), **extra))
            elif url.path == "/api/reverberation":
                opts = json.loads(self._body() or b"{}")
                s.run_cell(s.reverberation_code(opts.get("line", "Hb"), opts.get("n_mc", 1000)))
            elif url.path == "/api/sample":
                opts = json.loads(self._body() or b"{}")
                s.run_cell(s.sample_code(paths=opts.get("paths"), pattern=opts.get("pattern"),
                                         model=opts.get("model", "auto"), deredden=opts.get("deredden", True),
                                         ebv=opts.get("ebv"), mask_absorption=bool(opts.get("mask_absorption")),
                                         n_jobs=opts.get("n_jobs", -1), batch=opts.get("batch", "sample")))
            elif url.path == "/api/templates":
                opts = json.loads(self._body() or b"{}")
                s.run_cell(s.download_templates_code(opts.get("name", "")))
            elif url.path == "/api/project":
                s.load_project(self._body())
            elif url.path == "/api/reset":
                s.reset()
            else:
                return self._error("not found", 404)
        except Exception as exc:
            return self._error(str(exc), 409)
        return self._send(s.state())

    def _index(self):
        with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as fh:
            page = fh.read().replace("__FANTASY_TOKEN__", self.token)
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
