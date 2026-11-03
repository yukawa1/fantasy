"""
Refining a fit of high S/N: the presets are parsimonious and fit most SDSS
spectra to chi2/nu ~ 1, but at S/N > ~45 the data show what a simple model
leaves out (line wings, non-Gaussian profiles, a broad H-alpha in a LINER).
``refine`` tries the physically meaningful extensions of the preset one at a
time and keeps an extension only if the BIC improves by more than
``min_dbic`` and its new parameters do not end at a limit (a component at its
limit absorbs template mismatch rather than describing gas).

    r = fa.fit(spec, fa.presets.starforming())
    r = fa.refine(spec, "starforming", result=r)     # or refine(spec, "starforming")
    r.refinement                                      # what was tried, dBIC, kept or not

In the calibration on 160 SDSS narrow-line galaxies the chi2/nu of the presets
is ~1.0 below S/N 45 and 1.5 (median) above it; extensions are only tried
when chi2/nu exceeds ``min_rchi2``.
"""
import numpy as np
import pandas as pd

__all__ = ("refine", "EXTENSIONS")

# preset -> [(label, options added to the preset, components that the option adds)]
# The calibration polynomial (degree 10 over 3700-7500 A, ~one term per 400 A)
# corrects 1-3% flux-calibration and dust-law errors; on 390 SDSS galaxies it
# changed sigma* by +1-7 km/s and the BPT ratios by < 0.03 dex (median).
_CAL = ("flux-calibration polynomial", {"calibration": 10}, ("calibration",))
_GH = ("Gauss-Hermite profiles", {"narrow_profile": "gauss-hermite"}, ("narrow.", "narrow_hi."))
EXTENSIONS = {
    "starforming": [_CAL, ("outflow wing", {"outflow": True}, ("outflow",)), _GH],
    "seyfert2": [_CAL, _GH],
    "composite": [_CAL, _GH, ("outflow wing", {"outflow": True}, ("outflow",))],
    "liner": [_CAL, ("broad Balmer lines", {"broad": True}, ("broad",)),
              ("outflow wing", {"outflow": True}, ("outflow",)), _GH],
    "passive": [_CAL],
}


def _new_at_limit(result, prefixes):
    """Parameters of the added components (or the new h3/h4) at a limit."""
    out = []
    for name in result.at_limit:
        for p in prefixes:
            if p.endswith(".") and name.startswith(p) and name.split(".")[-1] in ("h3", "h4"):
                out.append(name)
            elif not p.endswith(".") and name.startswith(p + "."):
                out.append(name)
    return out


def refine(spectrum, preset, options=None, result=None, min_rchi2=1.2, min_dbic=10.0, max_steps=3,
           verbose=False, **fit_options):
    """
    Fit ``spectrum`` with ``preset`` (a name in :data:`EXTENSIONS`, with
    ``options``) and add extensions while the BIC prefers them (forward
    selection). Returns the best FitResult with ``result.refinement`` (a table
    of the trials) and ``result.preset_options`` (the options of the kept model).

    :param result: an existing fit with the same preset and options (saves one fit).
    :param min_rchi2: only refine if the fit has a larger reduced chi2.
    """
    from . import presets
    from .fitting import fit
    fit_options.setdefault("verbose", False)
    options = dict(options or {})
    best = result if result is not None else fit(spectrum, getattr(presets, preset)(**options), **fit_options)
    rows = []
    candidates = list(EXTENSIONS.get(preset, []))
    if best.rchi2 <= min_rchi2 or not candidates:
        best.refinement = _table(rows)
        best.preset_options = options
        return best
    for step in range(1, max_steps + 1):
        trials = []
        for label, opt, prefixes in candidates:
            if all(options.get(k) == v for k, v in opt.items()):
                continue
            kw = dict(options, **opt)
            try:
                r = fit(spectrum, getattr(presets, preset)(**kw), **fit_options)
            except Exception as exc:
                rows.append((step, label, np.nan, np.nan, False, f"failed: {type(exc).__name__}"))
                continue
            lim = _new_at_limit(r, prefixes)
            d = r.bic - best.bic
            ok = d < -min_dbic and not lim
            note = ("at limit: " + ", ".join(lim)) if lim else ""
            rows.append((step, label, d, r.rchi2, False, note))
            if ok:
                trials.append((d, label, kw, r, len(rows) - 1))
            if verbose:
                print(f"step {step}: {label}: dBIC = {d:.1f}, chi2/nu = {r.rchi2:.3f} {note}")
        if not trials:
            break
        d, label, kw, r, row = min(trials, key=lambda t: t[0])
        rows[row] = rows[row][:4] + (True,) + rows[row][5:]
        best, options = r, kw
        candidates = [c for c in candidates if c[0] != label]
        if best.rchi2 <= min_rchi2:
            break
    best.refinement = _table(rows)
    best.preset_options = options
    return best


def _table(rows):
    t = pd.DataFrame(rows, columns=["step", "extension", "dBIC", "rchi2", "kept", "note"])
    return t.astype({"step": int, "dBIC": float, "rchi2": float, "kept": bool})
