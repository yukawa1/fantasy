"""
Progress of long computations (independent fits of many spectra, the joint
multi-epoch fit). Code reports with :func:`report`; an application (the
FANTASY app) can install a hook with :func:`set_hook` to show it.
"""
import time

__all__ = ("report", "set_hook")

_HOOK = None
_START = {}


def set_hook(fn):
    """Call ``fn(dict)`` on every report (None: stop); returns the previous hook."""
    global _HOOK
    old, _HOOK = _HOOK, fn
    return old


def report(task, done=None, total=None, detail=""):
    """Report ``done`` of ``total`` steps of ``task`` (total None: unknown)."""
    hook = _HOOK
    if hook is None:
        return
    now = time.time()
    if not done or task not in _START:
        _START[task] = now
    try:
        hook({"task": task, "done": done, "total": total, "detail": detail, "time": now,
              "start": _START[task]})
    except Exception:            # a display problem must never stop a fit
        pass
