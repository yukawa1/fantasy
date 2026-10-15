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
    POST /api/reset
    GET  /api/figure.<png|pdf|eps>   the current figure (?dl=1 for publication quality)
    GET  /api/lines.csv, /api/session.py

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
            data = s.figure_bytes(fmt, dpi=300 if dl else 110)
            if data is None:
                return self._error("no figure yet", 404)
            ctype = {"png": "image/png", "pdf": "application/pdf",
                     "eps": "application/postscript"}[fmt]
            return self._send(data, ctype, filename=f"{s.figure_stem}.{fmt}" if dl else None)
        if url.path == "/api/lines.csv":
            csv = s.lines_csv()
            if csv is None:
                return self._error("no fit yet", 404)
            return self._send(csv, "text/csv", filename=f"{s.spec.name}_lines.csv")
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
                                       z=opts.get("z"))
                else:
                    code = s.read_code(data=self._body(), filename=q.get("filename"), z=q.get("z"))
                s.run_cell(code)
            elif url.path == "/api/classify":
                s.run_cell(s.classify_code())
            elif url.path == "/api/fit":
                opts = json.loads(self._body() or b"{}")
                errors = opts.get("errors", "covariance")
                if errors not in ("covariance", "mc"):
                    raise ValueError("errors must be covariance or mc")
                if s.model is not None:
                    s.run_cell(s.fit_model_code(errors))
                else:
                    s.run_cell(s.fit_code(model=opts.get("model", "optical"),
                                          broad_components=opts.get("broad_components", 2),
                                          outflow=opts.get("outflow", True), errors=errors))
            elif url.path == "/api/model":
                opts = json.loads(self._body() or b"{}")
                action = opts.get("action")
                if action == "preset":
                    code = s.preset_code(opts.get("model", "optical"), opts.get("broad_components", 2),
                                         opts.get("outflow", True))
                elif action == "add":
                    code = s.add_code(opts.get("component") or {})
                elif action == "remove":
                    code = s.remove_code(opts.get("name", ""))
                elif action == "params":
                    code = s.params_code(opts.get("changes") or [])
                else:
                    raise ValueError("unknown model action")
                s.run_cell(code)
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
