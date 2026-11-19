"""
Documentation shown in the app: hand-written guides and an API reference
generated from the docstrings (so it always matches the installed version).
"""
import html
import inspect
import re

from .guides import GUIDES      # (id, section, title, HTML)


# --------------------------------------------------------------- API reference
def _api_objects():
    import fantasy_agn as fa
    from fantasy_agn import presets
    from fantasy_agn.classify import Classification
    from fantasy_agn.result import FitResult
    from fantasy_agn.spectrum import Spectrum
    groups = [
        ("Reading data", [("fa.read", fa.read), ("Spectrum", Spectrum),
                          ("Spectrum.to_rest_frame", Spectrum.to_rest_frame),
                          ("Spectrum.deredden", Spectrum.deredden), ("Spectrum.crop", Spectrum.crop),
                          ("Spectrum.mask", Spectrum.mask), ("Spectrum.with_resolution", Spectrum.with_resolution), ("fa.find_redshift", fa.find_redshift), ("fa.desi.fetch", fa.desi.fetch),
                          ("fa.desi.find", fa.desi.find)]),
        ("Classification", [("fa.classify", fa.classify), ("fa.classify_fit", fa.classify_fit), ("Classification", Classification),
                            ("Classification.plot_steps", Classification.plot_steps),
                            ("fa.lookup_catalogs", fa.lookup_catalogs)]),
        ("Models", [("fa.Model", fa.Model), ("Model.starting_from", fa.Model.starting_from), ("Model.split_lines", fa.Model.split_lines)]
         + [(f"fa.presets.{n}", getattr(presets, n)) for n in
            ("hbeta", "halpha", "optical", "mgii", "civ", "type2", "galaxy", "seyfert2",
             "composite", "liner", "starforming", "passive", "tde", "bff", "auto", "variants")]
         + [(f"fa.{n}", getattr(fa, n)) for n in
            ("Lines", "PowerLaw", "BrokenPowerLaw", "Polynomial", "FeII", "BalmerContinuum",
             "HighOrderBalmer", "StellarPopulation", "HostGalaxy", "Extinction")]),
        ("Fitting", [("fa.fit", fa.fit), ("fa.guess", fa.guess), ("fa.compare", fa.compare),
                     ("fa.fit_many", fa.fit_many), ("fa.fit_sample", fa.fit_sample), ("fa.fit_absorbed", fa.fit_absorbed),
                     ("fa.refine", fa.refine), ("fa.prune_feii", fa.prune_feii), ("fa.fit_epochs", fa.fit_epochs),
                     ("fantasy_agn.progress.set_hook", _progress().set_hook),
                     ("fantasy_agn.progress.report", _progress().report),
                     ("fantasy_agn.progress.enabled", _progress().enabled),
                     ("fantasy_agn.progress.set_live_hook", _progress().set_live_hook),
                     ("fantasy_agn.progress.live", _progress().live),
                     ("fantasy_agn.progress.live_enabled", _progress().live_enabled)]),
        ("Results", [(f"FitResult.{n}", getattr(FitResult, n)) for n in
                     ("key_results", "lines", "line_limit", "upper_limit", "measure", "bh_mass", "bh_masses", "feii", "feii_groups", "balmer_decrement", "undersampled",
                      "host_properties", "luminosity", "continuum", "plot", "savefig", "save",
                      "summary")]),
    ]
    groups.extend([
        ("Fit checks", [(f"fa.{name}", getattr(fa, name)) for name in (
            "analyze_residuals", "suggest_lines", "simulation_check", "calibration_check", "robustness", "compare_fits", "model_confidence",
            "candidate_ties", "predictive_check", "link_test", "profile_likelihood", "calibrate")]),
        ("Epochs and reverberation", [
            ("fa.epoch_table", fa.epoch_table), ("fa.epoch_correlations", fa.epoch_correlations),
            ("fa.EpochSet", fa.EpochSet), ("fa.rm.component_reverberation", fa.rm.component_reverberation),
            ("fa.rm.lag_significance", fa.rm.lag_significance),
        ] + [(f"MultiEpochResult.{name}", getattr(fa.MultiEpochResult, name)) for name in (
            "shared", "scales", "offsets", "broadening", "redshift", "light_curve", "window_continuum", "reverberation", "rms_spectrum")]),
        ("New versions", [("fantasy_agn.update.check", _update().check), ("fantasy_agn.update.install", _update().install)]),
    ])
    return groups


def _progress():
    from .. import progress
    return progress


def _update():
    from .. import update
    return update


def _doc_html(text):
    """Docstring (reST-like) -> simple HTML."""
    text = inspect.cleandoc(text or "")
    if not text:
        return "<p><i>no description</i></p>"
    out, para, in_code, params = [], [], False, []

    def flush():
        if para:
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
            para.clear()

    lines = text.split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i]
        if in_code:
            if ln.strip() and not ln.startswith(" "):
                in_code = False
                out.append("</pre>")
                continue
            out.append(html.escape(ln[4:] if ln.startswith("    ") else ln))
            i += 1
            continue
        m = re.match(r"\s*:(param|return|returns)\s*([^:]*):\s*(.*)", ln)
        if m:
            flush()
            body = [m.group(3)]
            while i + 1 < len(lines) and lines[i + 1].startswith(" ") and not re.match(r"\s*:", lines[i + 1]):
                i += 1
                body.append(lines[i].strip())
            label = m.group(2).strip() or "returns"
            params.append((label, " ".join(body)))
            i += 1
            continue
        if params and not ln.strip():
            i += 1
            continue
        if ln.rstrip().endswith("::"):
            para.append(ln.rstrip()[:-1])
            flush()
            out.append('<pre>')
            in_code = True
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            continue
        if re.match(r"\s*[-*] ", ln):
            flush()
            out.append("<li>" + _inline(ln.strip()[2:]) + "</li>")
        elif not ln.strip():
            flush()
        else:
            para.append(ln.strip())
        i += 1
    if in_code:
        out.append("</pre>")
    flush()
    if params:
        out.append('<table class="doc params">' + "".join(
            f"<tr><td><code>{html.escape(a)}</code></td><td>{_inline(b)}</td></tr>" for a, b in params)
            + "</table>")
    body = "\n".join(out)
    return re.sub(r"(<li>.*?</li>\n?)+", lambda m_: "<ul>" + m_.group(0) + "</ul>", body, flags=re.S)


def _inline(text):
    t = html.escape(text)
    t = re.sub(r"``(.+?)``", r"<code>\1</code>", t)
    t = re.sub(r":(?:func|class|meth|mod):`~?([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"`([^`]+)`", r"<i>\1</i>", t)
    return t


def _signature(name, obj):
    target = obj.fget if isinstance(obj, property) else obj
    if isinstance(obj, property):
        return f"{name}"
    try:
        sig = str(inspect.signature(target))
    except (TypeError, ValueError):
        sig = "(...)"
    sig = sig.replace("(self, ", "(").replace("(self)", "()")
    return f"{name}{sig}"


def api_pages():
    pages = []
    for group, objs in _api_objects():
        items = []
        for name, obj in objs:
            doc = (obj.fget.__doc__ if isinstance(obj, property) else obj.__doc__) or ""
            if inspect.isclass(obj) and obj.__init__.__doc__ and len(doc) < 40:
                doc += "\n\n" + obj.__init__.__doc__
            sig = _signature(name, obj.__init__ if inspect.isclass(obj) and name != "Classification" else obj)
            if inspect.isclass(obj):
                sig = name + sig[sig.index("("):] if "(" in sig else name
            anchor = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
            items.append(f'<div class="api" id="{anchor}"><h3><code>{html.escape(sig)}</code></h3>'
                         f"{_doc_html(doc)}</div>")
        pid = "api-" + re.sub(r"[^a-z]+", "-", group.lower()).strip("-")
        pages.append({"id": pid, "title": group, "section": "API reference", "html": "\n".join(items)})
    return pages


def line_list_page():
    from ..lines import line_table
    t = line_table()
    e = html.escape
    rows = "".join(f"<tr><td><code>{e(n)}</code></td><td>{r.vacuum:.2f}</td><td>{r.wavelength:.2f} {e(r.medium)}</td>"
                   f"<td>{e(r.ion)}</td><td>{e(r.type)}</td><td>{e(r.series)}</td><td>{e(r.note)}</td></tr>"
                   for n, r in t.iterrows())
    return {"id": "lines", "title": "Line list", "section": "Reference",
            "html": f"<p>The {len(t)} lines of the library, with the names to use in "
                    "<code>fa.Lines(name, [lines])</code>. Wavelengths are used in vacuum; the tabulated value "
                    "and its medium (air or vacuum) are given next to it. In Python: <code>fa.line_table()</code>, "
                    "<code>fa.lines_in(wmin, wmax, type=..., ion=..., series=...)</code>, "
                    "<code>fa.line_info(name)</code>.</p>"
                    '<table class="doc"><tr><th>name</th><th>λ vacuum (Å)</th><th>tabulated</th><th>ion</th>'
                    f"<th>type</th><th>series</th><th>note</th></tr>{rows}</table>"}


def all_pages():
    pages = [{"id": i, "title": t, "section": s, "html": h.strip()} for i, s, t, h in GUIDES]
    pages.append(line_list_page())
    pages += api_pages()
    return pages
