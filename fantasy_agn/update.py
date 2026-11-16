"""
New versions of FANTASY: is one published on PyPI, and how to install it.

``check()`` asks PyPI (https://pypi.org/pypi/fantasy-agn/json) for the latest release, at most once a
day (the answer is kept in the data folder), and says how this installation is updated:

- ``uv-tool``: installed by the FANTASY installer (``uv tool install``),
- ``pip``: installed with pip into an environment (venv, conda),
- ``source``: run from a source checkout (``pip install -e`` or ``python -m``): update with git.

``install()`` runs that update for the first two; the running program keeps the old version until it
is started again. The request to PyPI sends nothing but the package name. ``FANTASY_NO_UPDATE_CHECK=1``
turns the check off.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

from . import _net
from . import __version__

__all__ = ("check", "install", "newer")

PYPI_URL = "https://pypi.org/pypi/fantasy-agn/json"
PROJECT_URL = "https://pypi.org/project/fantasy-agn/"
CHANGELOG_URL = "https://github.com/yukawa1/fantasy/blob/main/CHANGELOG.md"
CACHE_HOURS = 24


def _data_dir():
    return os.environ.get("FANTASY_DATA") or os.path.join(os.path.expanduser("~"), ".fantasy_agn")


def _parse(version):
    """(major, minor, patch, ...) of a final release, or None for pre-releases and odd strings."""
    if not re.fullmatch(r"\d+(\.\d+)*", str(version).strip()):
        return None
    return tuple(int(p) for p in str(version).strip().split("."))


def newer(latest, current=__version__):
    """Whether release ``latest`` is newer than ``current`` (pre-releases never are)."""
    a, b = _parse(latest), _parse(current)
    if a is None or b is None:
        return False
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)) > b + (0,) * (n - len(b))


def latest_version(timeout=4.0, force=False):
    """The latest release on PyPI (cached for a day), or None when PyPI cannot be reached."""
    cache = os.path.join(_data_dir(), "update.json")
    if not force:
        try:
            with open(cache) as f:
                c = json.load(f)
            if time.time() - c["time"] < CACHE_HOURS * 3600:
                return c["latest"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
    try:
        req = urllib.request.Request(PYPI_URL, headers={"User-Agent": "fantasy-agn update check",
                                                        "Accept": "application/json",
                                                        "Cache-Control": "no-cache"})   # fresh after a release
        with _net.urlopen(req, timeout=timeout) as r:
            latest = json.load(r)["info"]["version"]
    except Exception:
        return None
    try:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w") as f:
            json.dump({"time": time.time(), "latest": latest}, f)
    except OSError:
        pass
    return latest


def _uv():
    found = shutil.which("uv")
    if found:
        return found
    local = os.path.join(os.path.expanduser("~"), ".local", "bin", "uv")
    return local if os.path.exists(local) else None


def install_kind():
    """'uv-tool', 'pip' or 'source' (see the module documentation)."""
    here = os.path.dirname(os.path.abspath(__file__))
    if not any(part in ("site-packages", "dist-packages") for part in here.split(os.sep)):
        return "source"
    prefix = os.path.abspath(sys.prefix).replace("\\", "/")
    if "/uv/tools/" in prefix + "/":
        return "uv-tool"
    return "pip"


def update_command(version=None, kind=None):
    """The command (a list) that installs ``version`` (the latest when None), or None from source."""
    kind = kind or install_kind()
    spec = "fantasy_agn" if version is None else f"fantasy_agn=={version}"
    if kind == "uv-tool":
        uv = _uv()
        if uv is None:
            return None
        py = f"{sys.version_info.major}.{sys.version_info.minor}"
        return [uv, "tool", "install", "--force", "--refresh-package", "fantasy_agn", "--python", py, spec]
    if kind == "pip":
        try:
            import pip  # noqa: F401
            # without the cache pip asks PyPI afresh: its cached index can miss a release for 10 minutes
            return [sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", spec]
        except ImportError:                      # an environment made by uv has no pip
            uv = _uv()
            return None if uv is None else [uv, "pip", "install", "--python", sys.executable, "--upgrade",
                                            "--refresh-package", "fantasy_agn", spec]
    return None


def _shell(cmd):
    return " ".join(f'"{c}"' if " " in c else c for c in cmd)


def check(force=False):
    """
    The current and the latest version and how to update, as a dict: current, latest (None when
    PyPI could not be reached or the check is off), newer, kind, command (to show; None from
    source, where the advice is to update the checkout), can_install, url, changelog.
    """
    off = os.environ.get("FANTASY_NO_UPDATE_CHECK", "") not in ("", "0")
    latest = None if off else latest_version(force=force)
    kind = install_kind()
    cmd = update_command(latest, kind) if latest else update_command(None, kind)
    advice = (_shell(cmd) if cmd else
              "git pull  (then reinstall if you installed with pip install -e)" if kind == "source"
              else "uv tool install --force fantasy_agn")
    return {"current": __version__, "latest": latest, "newer": bool(latest and newer(latest)),
            "kind": kind, "command": advice, "can_install": cmd is not None, "checked": not off,
            "url": PROJECT_URL + (f"{latest}/" if latest else ""), "changelog": CHANGELOG_URL}


def install(version=None, timeout=600):
    """
    Install ``version`` (the latest when None) the way this copy was installed. Returns (ok, output);
    the new version is used once FANTASY is started again.
    """
    version = version or latest_version(force=True)
    if version and not newer(version):
        return True, f"FANTASY {__version__} is already the latest version."
    cmd = update_command(version)
    if cmd is None:
        return False, ("This copy runs from a source checkout: update it with git pull."
                       if install_kind() == "source" else "uv was not found: run the FANTASY installer again.")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           env=dict(os.environ, NO_COLOR="1", PIP_NO_COLOR="1"))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"{_shell(cmd)}\n{exc}"
    out = re.sub(r"\x1b\[[0-9;]*m", "", p.stdout + p.stderr).strip()      # no colour codes
    if p.returncode != 0 and "No matching distribution" in out:
        out += ("\n\nPyPI can take a few minutes to list a new release everywhere: "
                "try again in a few minutes.")
    return p.returncode == 0, f"$ {_shell(cmd)}\n{out[-3000:]}"
