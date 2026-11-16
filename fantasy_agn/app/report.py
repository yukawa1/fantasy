"""
A one-file report of the current fit: the figure, the key numbers, every check that was run
with its outcome, and the settings (with the Python code that repeats the analysis). One HTML
file with the figure embedded, to keep with the results or send to a colleague; it prints to PDF
from the browser.
"""
import base64
import html
import time

import numpy as np

CSS = """
body { font: 14px/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #1d2128; background: #fff;
  max-width: 980px; margin: 24px auto; padding: 0 16px; }
h1 { font-size: 22px; margin: 0 0 2px; } h2 { font-size: 16px; margin: 28px 0 8px; border-bottom: 1px solid #e6e5e1; padding-bottom: 4px; }
.meta { color: #6b6a66; font-size: 12.5px; margin: 0 0 16px; }
img { max-width: 100%; border: 1px solid #e6e5e1; }
table { border-collapse: collapse; font-size: 12.5px; margin: 6px 0; }
th, td { border-bottom: 1px solid #eeede9; padding: 3px 10px 3px 0; text-align: left; vertical-align: top; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
pre { background: #f6f5f2; padding: 10px 12px; font: 12px/1.45 ui-monospace, Menlo, Consolas, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
.ok { color: #2e7d4f; } .warn { color: #9a6200; } .muted { color: #6b6a66; }
.check { margin: 10px 0 14px; } .check b { font-size: 13.5px; }
@media print { body { margin: 0; } h2 { break-after: avoid; } img, table, pre { break-inside: avoid; } }
"""


def _e(text):
    return html.escape(str(text))


def _table(head, rows, num=()):
    out = ["<table><thead><tr>" + "".join(f'<th class="{"num" if i in num else ""}">{_e(h)}</th>' for i, h in enumerate(head))
           + "</tr></thead><tbody>"]
    for r in rows:
        out.append("<tr>" + "".join(f'<td class="{"num" if i in num else ""}">{_e(v)}</td>' for i, v in enumerate(r)) + "</tr>")
    return "".join(out) + "</tbody></table>"


def _num(v, err=None):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(v):
        return "—"
    if err is not None:
        try:
            err = float(err)
        except (TypeError, ValueError):
            err = np.nan
        if np.isfinite(err) and err > 0:
            return f"{v:.4g} ± {err:.2g}"
    return f"{v:.4g}"


def _check(title, status, text):
    cls = "ok" if status == "ok" else "warn" if status == "attention" else "muted"
    return (f'<div class="check"><b>{_e(title)}</b> — <span class="{cls}">{_e(status)}</span>'
            f"<pre>{_e(text.strip())}</pre></div>")


def _summary(obj):
    s = getattr(obj, "summary", None)
    try:
        return s() if callable(s) else repr(obj)
    except Exception:                            # noqa: BLE001  (a report must not fail for one check)
        return repr(obj)


def build(session):
    """The report of the session's current fit as HTML text (ValueError if nothing is fitted)."""
    from matplotlib.figure import Figure

    s = session
    kind = s.wf.get("fit_kind")
    r, me = s.result, s.multi
    if kind == "joint" and me is not None:
        shown = me.epochs[getattr(me, "reference", 0)]
    elif r is not None:
        shown = r
    else:
        raise ValueError("there is no fit to report yet")
    name = shown.spectrum.name or "spectrum"
    parts = [f"<h1>FANTASY report: {_e(name)}</h1>",
             f'<p class="meta">{_e(time.strftime("%Y-%m-%d %H:%M"))} · FANTASY {_e(__import__("fantasy_agn").__version__)} · '
             + ("joint fit of %d epochs" % len(me.epochs) if kind == "joint" and me is not None else
                "%d epochs fitted one by one" % len(s.fits or []) if kind == "fits" else "fit of one spectrum")
             + "</p>"]

    # ---------------------------------------------------------------- figure
    title = name + (" (reference epoch)" if kind == "joint" else "")
    with s._render_lock:
        png = s._fig_bytes(shown.plot(fig=Figure(), title=title), "png", 130)
    parts.append(f'<img alt="the fit" src="data:image/png;base64,{base64.b64encode(png).decode()}">')

    # ---------------------------------------------------------------- numbers
    parts.append("<h2>Fit</h2>")
    if kind == "joint" and me is not None:
        parts.append(f"<pre>{_e(me.summary())}</pre>")
        sc = me.scales
        if len(sc) <= 200:
            rows = [(int(q.epoch), q.name, _num(q.mjd), "reference" if q.reference else _num(q.scale, q.error))
                    for q in sc.itertuples()]
            parts.append(_table(["Epoch", "Spectrum", "MJD", "Flux scale"], rows, num=(0, 2, 3)))
    else:
        parts.append(f"<pre>{_e(shown.summary())}</pre>")
        if s.results:
            parts.append("<h2>Key results</h2>" + _table(["Quantity", "Value"], s.results))
        lines = s._line_rows(shown)
        rows = []
        for d in lines:
            if d.get("limit") and (not d["detected"] or (np.isfinite(d["flux_err"]) and d["flux"] < 2 * d["flux_err"])):
                flux = f"< {d['limit']['value']:.3g} ({d['limit']['how']})"
            else:
                flux = _num(d["flux"], d["flux_err"]) if d["detected"] else "0"
            rows.append((d["component"], d["label"], flux, _num(d["fwhm"], d["fwhm_err"]), _num(d["shift"], d["shift_err"]),
                         _num(d["ew"], d["ew_err"]) if d["detected"] else "0", ", ".join(f["text"] for f in d["flags"])))
        if rows:
            parts.append("<h2>Lines</h2>" + _table(["Component", "Line", "Flux", "FWHM (km/s)", "Shift (km/s)", "EW (Å)", "Flags"],
                                                   rows, num=(2, 3, 4, 5)))

    # ---------------------------------------------------------------- class
    try:
        final = s._final_classifications()
    except Exception:                                # noqa: BLE001
        final = None
    if final:
        parts.append("<h2>Classification</h2>")
        rows = [(c.spectrum.name, c.label + (" (uncertain)" if c.uncertain else ""),
                 c.measurements.get("preliminary", "—"), c.reason) for c in final]
        parts.append(_table(["Spectrum", "Final class (fit)", "Preliminary", "Reason"], rows))
        parts.append('<p class="muted">Final class: fa.classify_fit, from the fitted values. Preliminary: fa.classify, '
                     "its own quick fits before the model was chosen.</p>")

    # ---------------------------------------------------------------- checks
    checks = []
    if kind == "single" and r is not None:
        resid = s.wf_residuals()
        if resid is not None:
            n = len(resid.get("rows", []))
            text = "\n".join(f"{d['kind']} at {_num(d['center'])} A ({_num(d['sigma'])} sigma)"
                             + (f", {d['line']}" if d["line"] else "") + (f": {d['suggestion']}" if d["suggestion"] else "")
                             for d in resid.get("rows", [])) or "no coherent feature in the residuals"
            checks.append(("1. Does the model miss anything? (fa.analyze_residuals)", "attention" if n else "ok", text))
        rob = s.wf_robust()
        if rob and rob.get("id"):
            obj = s.ns.get("rob")
            flagged = rob.get("improved") or any(q.get("flag") != "ok" for q in rob.get("quantities", []))
            checks.append(("2. Are the results stable? (fa.robustness)", "attention" if flagged else "ok", _summary(obj)))
        cmp = s.wf_compare()
        if cmp and cmp.get("cmp"):
            checks.append(("3. Comparison with another fit (fa.compare_fits)", "done", _summary(s.ns.get("cmp"))))
        if getattr(r, "feii_pruned", None) is not None:
            dropped = list(r.feii_pruned or [])
            checks.append(("4. Weak Fe II (fa.prune_feii)", "done",
                           f"Fe II groups below {getattr(r, 'feii_nsigma', 2):g} sigma fixed at zero and the fit repeated: "
                           + (", ".join(dropped) if dropped else "none dropped")))
        sim = s.wf_simcheck()
        if sim and sim.get("id"):
            bad = [d for d in sim["rows"] if d["flag"] != "ok"]
            checks.append(("5. Would the fit recover the truth? (fa.simulation_check)", "attention" if bad else "ok",
                           _summary(s.ns.get("sim"))))
    if kind == "joint" and me is not None:
        J = s.wf_joint_checks() or {}
        if J.get("cal", {}).get("id"):
            checks.append(("Calibration between the epochs (fa.calibration_check)", "ok" if J["cal"]["ok"] else "attention",
                           _summary(s.ns.get("chk"))))
        if J.get("sim", {}).get("id"):
            bad = [d for d in J["sim"]["rows"] if d["flag"] not in ("ok", "not measured")]
            checks.append(("Simulations of the joint fit (fa.simulation_check)", "attention" if bad else "ok",
                           _summary(s.ns.get("sim"))))
    if kind in ("joint", "fits") and s.reverb is not None:
        rv = s.reverb
        checks.append((f"Reverberation of {rv.line}", "attention" if rv.warnings else "ok", repr(rv)))
    parts.append("<h2>Checks</h2>")
    if checks:
        parts += [_check(*c) for c in checks]
    else:
        parts.append('<p class="muted">No check was run on this fit (Results › Checks in the app).</p>')
    if s.warnings:
        parts.append("<h2>Warnings of the fit</h2><ul>" + "".join(f"<li>{_e(w)}</li>" for w in s.warnings) + "</ul>")

    # ---------------------------------------------------------------- settings
    parts.append("<h2>Settings</h2>")
    rows = []
    for k, v in (s.wf.get("prepare") or {}).items():
        if v not in (None, "", [], False):
            rows.append(("prepare", k, v))
    for k, v in (s.wf.get("fit_opts") or {}).items():
        if v not in (None, "", [], False):
            rows.append(("fit", k, v))
    if rows:
        parts.append(_table(["Step", "Setting", "Value"], rows))
    parts.append("<p>The Python code of the whole analysis (it repeats every step):</p>"
                 f"<pre>{_e(s.script())}</pre>")
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>FANTASY report: {_e(name)}</title><style>{CSS}</style></head><body>"
            + "\n".join(parts) + "</body></html>")
