"""
Progress of long computations (independent fits of many spectra, the joint
multi-epoch fit). Code reports with :func:`report`; an application (the
FANTASY app) can install a hook with :func:`set_hook` to show it.
"""
import time

__all__ = ("report", "set_hook", "enabled", "live", "set_live_hook", "live_enabled")

_HOOK = None
_LIVE_HOOK = None
_START = {}


def set_hook(fn):
    """Call ``fn(dict)`` on every report (None: stop); returns the previous hook."""
    global _HOOK
    old, _HOOK = _HOOK, fn
    return old


def enabled():
    """True when a hook is installed (worth preparing extra detail for it)."""
    return _HOOK is not None


def report(task, done=None, total=None, detail="", item=None):
    """Report ``done`` of ``total`` steps of ``task`` (total None: unknown). ``item``: optional detail of
    the step just finished (for example one spectrum of a sample: its class, chi2 and a thumbnail)."""
    hook = _HOOK
    if hook is None:
        return
    now = time.time()
    if not done or task not in _START:
        _START[task] = now
    try:
        p = {"task": task, "done": done, "total": total, "detail": detail, "time": now, "start": _START[task]}
        if item is not None:
            p["item"] = item
        hook(p)
    except Exception:            # a display problem must never stop a fit
        pass


def set_live_hook(fn):
    """Call ``fn(dict)`` with snapshots of a running fit (the data, the current model, chi2; see
    ``fitting._Live``); None: stop. Returns the previous hook. The snapshots are larger than the
    reports of :func:`report`, so they have their own hook."""
    global _LIVE_HOOK
    old, _LIVE_HOOK = _LIVE_HOOK, fn
    return old


def live_enabled():
    """True when a live hook is installed (worth sending snapshots of a running fit)."""
    return _LIVE_HOOK is not None


def live(data):
    """Send a snapshot of a running fit to the live hook, if any."""
    hook = _LIVE_HOOK
    if hook is None:
        return
    try:
        hook(data)
    except Exception:            # a display problem must never stop a fit (a KeyboardInterrupt does)
        pass
