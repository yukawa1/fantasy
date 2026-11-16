"""
The ``fantasy`` command.

    fantasy test            check the installation (offline, about a minute)
    fantasy test --full     also download the stellar templates (13 MB) and use online services
    fantasy app             start the FANTASY app in the browser
    fantasy info            versions, processor cores, templates, data folder
    fantasy examples DIR    copy the example spectra and a script to DIR, to try FANTASY
    fantasy python [...]    the Python of the FANTASY installation (scripts, interactive use)
"""
import argparse
import os
import platform
import shutil
import sys
import time
import traceback

__all__ = ("main", "selftest")


def _symbols():
    """Check marks if the terminal can print them (old Windows consoles cannot)."""
    try:
        "✓✗–".encode(sys.stdout.encoding or "ascii")
        return "✓", "✗", "–"
    except (UnicodeEncodeError, LookupError):
        return "ok", "FAIL", "skip"


# --------------------------------------------------------------------- test
class _Checks:
    def __init__(self, verbose=False):
        self.ok, self.bad, self.skip = _symbols()
        self.failed, self.skipped, self.verbose = [], [], verbose

    def run(self, title, fn, *args):
        """Run one check; fn returns a short result text, raises on failure, or returns
        ('skip', reason)."""
        t0 = time.time()
        live = sys.stdout.isatty()           # "…" while running, replaced by the result
        if live:
            print(f"  {title} …", end="", flush=True)
        try:
            out = fn(*args)
        except Exception as exc:                       # a failed check, not a crash
            print(("\r" if live else "") + f"  {self.bad} {title}: {type(exc).__name__}: {exc}")
            if self.verbose:
                traceback.print_exc()
            self.failed.append(title)
            return None
        if isinstance(out, tuple) and out and out[0] == "skip":
            print(("\r" if live else "") + f"  {self.skip} {title}: skipped ({out[1]})")
            self.skipped.append(title)
            return None
        text = f"  {self.ok} {title}" + (f": {out}" if out else "")
        print(("\r" if live else "") + text + f"  ({time.time() - t0:.1f} s)")
        return out if out is not None else True


def _templates_ready(name="emiles"):
    from . import templates
    return os.path.exists(templates._path(name))


def selftest(full=False, verbose=False):
    """
    End-to-end check of the installation: reading, fitting, parallel and
    multi-epoch fits, the app server; with ``full`` also the stellar templates
    (downloaded once), classification, host-galaxy fits and the online dust
    maps. Returns True if nothing failed.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    import fantasy_agn as fa

    c = _Checks(verbose)
    print(f"FANTASY {fa.__version__} self-test (Python {platform.python_version()}, "
          f"{platform.system()} {platform.machine()}, {os.cpu_count()} cores)")
    st = {}

    def read():
        s = fa.read(fa.examples.path("type1"))
        st["raw"] = s
        st["spec"] = s.deredden(ebv=0.03).to_rest_frame()
        return f"{s.name}, z = {s.z:.3f}, {s.wave.size} pixels"
    c.run("Read an SDSS spectrum", read)

    def fit_hbeta():
        r = fa.fit(st["spec"], fa.presets.hbeta(), verbose=False)
        hb = r.measure("Hb", ["broad", "broad2"])
        if not (r.rchi2 < 3 and hb["flux"] > 0 and 1000 < hb["fwhm"] < 20000):
            raise AssertionError(f"unexpected fit: reduced chi2 {r.rchi2:.2f}, Hb FWHM {hb['fwhm']:.0f}")
        st["r"] = r
        return f"reduced chi2 = {r.rchi2:.2f}, broad Hb FWHM = {hb['fwhm']:.0f} ± {hb['fwhm_err']:.0f} km/s"
    if "spec" in st:
        c.run("Fit the Hb region (Fe II, two broad components, outflow)", fit_hbeta)

    def figure():
        import io
        buf = io.BytesIO()
        st["r"].plot().savefig(buf, format="pdf")
        return f"PDF figure, {len(buf.getvalue()) // 1024} kB"
    if "r" in st:
        c.run("Draw a publication figure", figure)

    def epochs():
        # three "epochs" of the example with a varying broad line and calibration
        rng = np.random.default_rng(1)
        s, r = st["spec"], st["r"]
        good = (s.wave > 4500) & (s.wave < 5200)
        x = s.wave[good]
        base = r.model.evaluate(x, s.context)
        broad = sum(v for k, v in r.model.evaluate(x, s.context, components=True).items()
                    if k.startswith("broad"))
        spectra, scales = [], [1.0, 0.92, 1.07]
        for k, sc in enumerate(scales):
            flux = sc * (base + 0.25 * np.sin(k + 1) * broad)
            err = s.err[good]
            spectra.append(fa.Spectrum(x, flux + rng.normal(0, err), err, rest_frame=True,
                                       z=s.z, mjd=55000 + 30 * k, name=f"epoch{k}"))
        me = fa.fit_epochs(spectra, fa.presets.hbeta(broad_components=1), reference=0, n_jobs=1,
                           verbose=False)
        got, err = me.scales.scale.to_numpy(), me.scales.error.to_numpy()
        if np.any(np.abs(got - scales) > 4 * err + 1e-9):
            raise AssertionError(f"flux scales {np.round(got, 3)} ± {np.round(err, 3)} instead of {scales}")
        return ("3 epochs, flux scales " + ", ".join(f"{v:.3f}±{e:.3f}" for v, e in zip(got, err))
                + f" (true {', '.join(map(str, scales))})")
    if "r" in st:
        c.run("Joint multi-epoch fit with intercalibration", epochs)

    def parallel():
        spectra = [st["spec"], st["spec"].copy(name="copy")]
        t0 = time.time()
        out = fa.fit_many(spectra, fa.presets.hbeta(broad_components=1), n_jobs=2, verbose=False)
        bad = [o for o in out if isinstance(o, Exception)]
        if bad:
            raise bad[0]
        return f"2 spectra in 2 processes, {time.time() - t0:.1f} s"
    if "spec" in st:
        c.run("Fit several spectra in parallel processes", parallel)

    def app():
        import json
        import threading
        import urllib.request
        from .app.server import make_server
        server = None
        for port in range(8890, 8910):
            try:
                server = make_server(port)
                break
            except OSError:
                continue
        if server is None:
            raise OSError("no free port")
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            from .app import server as srv
            url = f"http://127.0.0.1:{server.server_address[1]}"
            page = urllib.request.urlopen(url + "/", timeout=10).read().decode()
            req = urllib.request.Request(url + "/api/state", headers={"X-Fantasy-Token": srv.Handler.token})
            state = json.loads(urllib.request.urlopen(req, timeout=10).read())
            if "FANTASY" not in page or "cells" not in state:
                raise AssertionError("unexpected answer from the app server")
        finally:
            server.shutdown()
            server.server_close()
        return "web interface and API answer"
    c.run("Start the app server", app)

    # ---- needs the stellar templates (downloaded once) and the internet
    have = _templates_ready("emiles")
    if full and not have:
        def download():
            path = fa.templates.download("emiles")
            return "E-MILES, 13 MB, kept in " + os.path.dirname(path)
        have = bool(c.run("Download the stellar templates (once)", download))

    def classify(name, expect):
        s = fa.read(fa.examples.path(name)).deredden(ebv=0.03).to_rest_frame()
        cl = fa.classify(s)
        if not cl.label.startswith(expect):
            raise AssertionError(f"classified as {cl.label}, expected {expect}")
        st[name] = s
        return f"{cl.label} (preset: fa.presets.{cl.RECOMMENDED[cl.label][0]})"

    def seyfert2():
        r = fa.fit(st["seyfert2"], fa.presets.seyfert2(), verbose=False)
        hp = r.host_properties()
        if not (r.rchi2 < 3 and 50 < hp["sigma_star"][0] < 300):
            raise AssertionError(f"reduced chi2 {r.rchi2:.2f}, sigma* {hp['sigma_star'][0]:.0f}")
        return (f"reduced chi2 = {r.rchi2:.2f}, stellar sigma = {hp['sigma_star'][0]:.0f} km/s, "
                f"host fraction at 5100 A = {hp.get('host_fraction_5100', (np.nan,))[0]:.0%}")

    reason = "stellar templates not downloaded: run  fantasy test --full"
    if have:
        c.run("Classify the quasar", classify, "type1", "type1")
        c.run("Classify the Seyfert 2", classify, "seyfert2", "seyfert2")
        if "seyfert2" in st:
            c.run("Fit a Seyfert 2 with a stellar host", seyfert2)
    else:
        for title in ("Classify the quasar", "Classify the Seyfert 2", "Fit a Seyfert 2 with a stellar host"):
            c.run(title, lambda: ("skip", reason))

    def dust():
        s = st["raw"].deredden()
        return f"E(B-V) = {s.meta['ebv']:.4f} ({s.meta['ebv_source']})"
    if full and "raw" in st:
        c.run("Look up Galactic extinction (SFD dust maps / IRSA)", dust)

    print()
    if c.failed:
        print(f"{len(c.failed)} check(s) failed: {', '.join(c.failed)}.")
        print("Run  fantasy test --verbose  for details, and send the output to the FANTASY authors.")
        return False
    extra = f" ({len(c.skipped)} skipped)" if c.skipped else ""
    print(f"All checks passed{extra}. Start the app with:  fantasy app")
    return True


# --------------------------------------------------------------------- info
def info():
    import fantasy_agn as fa
    from . import templates
    rows = [("FANTASY", fa.__version__), ("Python", f"{platform.python_version()} ({sys.executable})"),
            ("system", f"{platform.system()} {platform.release()} {platform.machine()}"),
            ("processor cores", str(os.cpu_count()))]
    for mod in ("numpy", "scipy", "astropy", "pandas", "matplotlib", "sfdmap2"):
        try:
            rows.append((mod, __import__(mod).__version__))
        except Exception:
            rows.append((mod, "not installed" + (" (optional: online dust lookup is used)"
                                                 if mod == "sfdmap2" else "")))
    data = os.environ.get("FANTASY_DATA") or os.path.join(os.path.expanduser("~"), ".fantasy_agn")
    rows.append(("data folder", data))
    from . import update
    u = update.check()
    rows.append(("latest on PyPI", "not checked" if not u["checked"] else "unknown (offline)" if u["latest"] is None
                 else f"{u['latest']}" + ("  (newer: fantasy update)" if u["newer"] else "")))
    for name, meta in templates.TEMPLATE_SETS.items():
        rows.append((f"templates {name}", "installed" if _templates_ready(name)
                     else f"not downloaded ({meta['size_mb']} MB, on first use)"))
    w = max(len(a) for a, _ in rows)
    for a, b in rows:
        print(f"{a:<{w}}  {b}")


# ------------------------------------------------------------------- update
def update_cmd(check_only=False, yes=False):
    from . import update
    u = update.check(force=True)
    if not u["checked"]:
        print("The update check is off (FANTASY_NO_UPDATE_CHECK).")
        return 0
    if u["latest"] is None:
        print(f"FANTASY {u['current']}: PyPI could not be reached ({update.PYPI_URL}).")
        return 1
    if not u["newer"]:
        print(f"FANTASY {u['current']} is the latest version.")
        return 0
    print(f"FANTASY {u['latest']} is published (you have {u['current']}). What's new: {u['changelog']}")
    if check_only or not u["can_install"]:
        print(f"Update with:  {u['command']}")
        return 0
    if not yes:
        try:
            if input(f"Install it now with  {u['command']}  ? [y/N] ").strip().lower() not in ("y", "yes"):
                return 0
        except EOFError:
            return 0
    ok, out = update.install(u["latest"])
    print(out)
    print(f"FANTASY {u['latest']} is installed; it is used from the next start." if ok
          else "The update did not install: run the command above yourself.")
    return 0 if ok else 1


# ----------------------------------------------------------------- examples
_EXAMPLE_SCRIPT = '''"""A first FANTASY analysis: run with  python first_fit.py"""
import fantasy_agn as fa

if __name__ == "__main__":
    spec = fa.read("{type1}").deredden().to_rest_frame()
    print(spec)
    print(fa.classify(spec))                     # what kind of object is it?
    result = fa.fit(spec, fa.presets.hbeta())    # broad Hb, Fe II, [OIII]
    print(result.measure("Hb", ["broad", "broad2"]))   # broad Hb: flux, FWHM, sigma, EW
    print(result.bh_mass())                      # single-epoch black-hole mass
    result.savefig("first_fit.pdf")
    print("figure saved: first_fit.pdf")
'''


def examples(folder):
    import fantasy_agn as fa
    os.makedirs(folder, exist_ok=True)
    for name in fa.examples.EXAMPLES:
        shutil.copy(fa.examples.path(name), folder)
    script = os.path.join(folder, "first_fit.py")
    with open(script, "w", encoding="utf-8") as fh:
        fh.write(_EXAMPLE_SCRIPT.format(type1=fa.examples.EXAMPLES["type1"][0]))
    print(f"Copied to {os.path.abspath(folder)}:")
    for name, (f, desc) in fa.examples.EXAMPLES.items():
        print(f"  {f}   {desc}")
    print("  first_fit.py   a first analysis:  cd", folder, "&& python first_fit.py")


# --------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(prog="fantasy", description="FANTASY: AGN spectral fitting")
    sub = ap.add_subparsers(dest="command")
    t = sub.add_parser("test", help="check the installation")
    t.add_argument("--full", action="store_true",
                   help="also download the stellar templates (13 MB) and use online services")
    t.add_argument("--verbose", action="store_true", help="show tracebacks of failed checks")
    a = sub.add_parser("app", help="start the FANTASY app in the browser")
    a.add_argument("--port", type=int, default=8780)
    a.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    sub.add_parser("info", help="versions, processor cores, templates, data folder")
    u = sub.add_parser("update", help="install the latest FANTASY from PyPI (or only check with --check)")
    u.add_argument("--check", action="store_true", help="only say whether a new version is published")
    u.add_argument("--yes", "-y", action="store_true", help="install without asking")
    e = sub.add_parser("examples", help="copy the example spectra and a first script to a folder")
    e.add_argument("folder", nargs="?", default="fantasy-examples")
    sub.add_parser("python", add_help=False,
                   help="run the Python of this installation: fantasy python script.py")
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["python"]:
        # FANTASY is installed in an environment of its own (uv tool): this is its Python
        os.execv(sys.executable, [sys.executable] + argv[1:])
    args = ap.parse_args(argv)
    if args.command == "test":
        return 0 if selftest(full=args.full, verbose=args.verbose) else 1
    if args.command == "app":
        from .app import run
        run(port=args.port, open_browser=not args.no_browser)
        return 0
    if args.command == "info":
        info()
        return 0
    if args.command == "examples":
        examples(args.folder)
        return 0
    if args.command == "update":
        return update_cmd(check_only=args.check, yes=args.yes)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
