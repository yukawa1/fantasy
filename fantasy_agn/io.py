"""
Saving and loading FANTASY objects: fit results, multi-epoch results,
classifications, models and spectra, in one compressed file.

    fa.save(r, "quasar.fantasy")
    r = fa.load("quasar.fantasy")       # plot, measure, refit ... as before

The file is a compressed Python pickle with a small header (FANTASY version,
type, date). Like any pickle it can run code when loaded: open only files you
or your collaborators made. Stellar templates are not stored; they are read
from the template folder (downloaded once) when the file is loaded.
"""
import datetime
import gzip
import pickle

__all__ = ("save", "load")

MAGIC = "FANTASY-SAVE"


def save(obj, path):
    """Save a FitResult, MultiEpochResult, Classification, Model, Spectrum (or a dict of them)."""
    from . import __version__
    header = {"magic": MAGIC, "version": __version__, "type": type(obj).__name__,
              "saved": datetime.datetime.now().isoformat(timespec="seconds")}
    with gzip.open(path, "wb", compresslevel=6) as fh:
        pickle.dump(header, fh, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load(path):
    """Load what :func:`save` wrote. Only open files from sources you trust."""
    with gzip.open(path, "rb") as fh:
        header = pickle.load(fh)
        if not isinstance(header, dict) or header.get("magic") != MAGIC:
            raise ValueError(f"{path} is not a FANTASY file")
        obj = pickle.load(fh)
    try:
        obj.saved_with = header
    except AttributeError:
        pass
    return obj
