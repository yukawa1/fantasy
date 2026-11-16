"""
The FANTASY application: a local web interface (read -> classify -> fit ->
results) that opens in the browser and runs only on this computer.

    fantasy-app                 # from a terminal
    python -m fantasy_agn.app   # the same
    fa.app()                    # from Python

Every step shows the FANTASY code it runs and its output; the session can be
downloaded as a Python script.
"""
import argparse
import sys
import threading
import webbrowser

__all__ = ("run", "main", "RECOMMENDED_PYTHON")

# The Python version the app is tested with (also in .python-version and the launchers).
RECOMMENDED_PYTHON = (3, 12)


def run(port=8780, open_browser=True):
    """Start the application (blocks until Ctrl+C)."""
    import matplotlib
    matplotlib.use("Agg")                    # figures are rendered to images
    from .server import make_server
    for p in range(port, port + 20):         # next free port if this one is taken
        try:
            server = make_server(p)
            break
        except OSError:
            continue
    else:
        raise OSError(f"no free port in {port}-{port + 19}")
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"FANTASY app running at {url}  (Ctrl+C to stop)", flush=True)
    if sys.version_info[:2] != RECOMMENDED_PYTHON:
        print(f"note: the app is tested with Python {RECOMMENDED_PYTHON[0]}.{RECOMMENDED_PYTHON[1]}, "
              f"this is {sys.version.split()[0]}; uv tool install --python 3.12 fantasy_agn sets up "
              "the recommended environment.", flush=True)
    if open_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fantasy-app", description="FANTASY: AGN spectral fitting app")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    a = ap.parse_args(argv)
    run(port=a.port, open_browser=not a.no_browser)
