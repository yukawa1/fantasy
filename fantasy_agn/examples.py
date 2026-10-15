"""
Example spectra shipped with the package (for first tries and the self-test).

    fa.examples.list()                     # names
    spec = fa.read(fa.examples.path("type1"))
"""
import os

__all__ = ("path", "list")

_DIR = os.path.join(os.path.dirname(__file__), "data", "examples")

# name: (file, description)
EXAMPLES = {
    "type1": ("spec-1059-52618-0453.fits",
              "SDSS quasar, z = 0.344: broad Balmer lines and Fe II (preset: hbeta / optical)"),
    "seyfert2": ("spec-2139-53878-0330.fits",
                 "SDSS Seyfert 2 galaxy (Z 133-99): narrow lines and a stellar host (preset: seyfert2)"),
}


def path(name="type1"):
    """Full path of an example spectrum."""
    if name not in EXAMPLES:
        raise KeyError(f"unknown example {name!r}; choose from {', '.join(EXAMPLES)}")
    return os.path.join(_DIR, EXAMPLES[name][0])


def list():                                       # noqa: A001 (fa.examples.list())
    """{name: description} of the example spectra."""
    return {k: v[1] for k, v in EXAMPLES.items()}
