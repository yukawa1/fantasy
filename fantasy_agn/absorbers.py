"""
Fits with absorption troughs: find troughs in the residuals of a fit, add
them as an :class:`~fantasy_agn.components.Absorption` component and refit,
until the BIC no longer improves. The emission components then describe the
unabsorbed spectrum (e.g. the full CIV profile under a mini-BAL trough).

    r = fa.fit_absorbed(spec, fa.presets.civ())
    r.absorbers          # the troughs found: wavelength, FWHM, optical depth
    r.plot()

Narrow absorbers (a few pixels) can also simply be masked:
``spec.mask_absorption()``.
"""
import numpy as np
import pandas as pd

from .components import Absorption

__all__ = ("find_troughs", "fit_absorbed")

C_KMS = 299792.458


def find_troughs(result, nsigma=4.0, min_pixels=2, max_width_kms=10000.0, avoid=()):
    """
    Troughs in the residuals of a fit: runs of pixels below the model by more
    than 2 sigma (smoothed over three pixels), the deepest beyond ``nsigma``.
    Returns a list of (wavelength, FWHM km/s, tau) starting values, deepest first.
    """
    s = result.spectrum
    sel = result.mask & np.isfinite(s.flux) & np.isfinite(s.err)
    idx = np.nonzero(sel)[0]
    w, f, e = s.wave[idx], s.flux[idx], s.err[idx]
    model = result.model_flux()[idx]
    dev = (f - model) / e
    smooth = np.convolve(dev, np.ones(3) / 3, mode="same")
    low = smooth < -2.0
    dv = np.median(np.diff(np.log(w))) * C_KMS
    out, k = [], 0
    while k < idx.size:
        if not low[k]:
            k += 1
            continue
        j = k
        while j < idx.size and low[j]:
            j += 1
        run = slice(k, j)
        width = (j - k) * dv
        if j - k >= min_pixels and dev[run].min() < -nsigma and width <= max_width_kms:
            depth = np.clip(model[run] - f[run], 0, None)
            centre = float(np.sum(w[run] * depth) / depth.sum()) if depth.sum() > 0 else float(w[run].mean())
            ratio = np.clip(f[run] / np.where(model[run] > 0, model[run], np.nan), 0.02, 1.0)
            tau = float(-np.log(np.nanmin(ratio))) if np.isfinite(ratio).any() else 0.5
            if not any(abs(centre - a) < 0.5 * a * max(width, fw) / C_KMS for a, fw in avoid):
                out.append((centre, max(width * 0.6, 2 * dv), max(tau, 0.05)))
        k = j
    return sorted(out, key=lambda t: -t[2])


def fit_absorbed(spectrum, model, max_troughs=10, nsigma=4.0, iterations=3, min_dbic=10.0,
                 max_fwhm=6000.0, verbose=False, **fit_options):
    """
    Fit ``model``, then add absorption troughs found in the residuals and refit
    while the BIC improves by more than ``min_dbic``. Returns the best FitResult
    with ``result.absorbers`` (a table of the troughs) and ``result.no_absorbers``
    (the fit without them).

    :param max_fwhm: widest trough, km/s; each trough may also grow to at most three
        times the width at which it was found. Wider troughs would trade flux with
        the emission (a broad shallow "absorber" under a too-broad line).
    """
    from .fitting import fit
    fit_options.setdefault("verbose", False)
    base = fit(spectrum, model, **fit_options)
    best, troughs = base, []
    for _ in range(iterations):
        known = []
        if troughs:
            p = {q.name: q.value for q in best.model.params}
            troughs = [(p[f"absorption.a{k}.wave"], p[f"absorption.a{k}.fwhm"], p[f"absorption.a{k}.tau"])
                       for k in range(len(troughs))]
            known = [(t[0], t[1]) for t in troughs]
        new = find_troughs(best, nsigma=nsigma, avoid=known)
        new = new[:max(0, max_troughs - len(troughs))]
        if not new:
            break
        m = model.copy()
        start = troughs + new
        m.add(Absorption(start, fwhm_range=(30.0, max_fwhm)))
        for k, t in enumerate(start):
            m.set(f"absorption.a{k}.fwhm", max=min(max_fwhm, max(3.0 * t[1], 300.0)))
        trial = fit(spectrum, m, **fit_options)
        if verbose:
            print(f"{len(troughs + new)} troughs: dBIC = {trial.bic - best.bic:.1f}")
        if trial.bic < best.bic - min_dbic:
            best, troughs = trial, troughs + new
        else:
            break
    rows = []
    if troughs:
        names = best.model.params.names
        for k in range(len(troughs)):
            row = {"trough": k}
            for q in ("wave", "fwhm", "tau"):
                i = names.index(f"absorption.a{k}.{q}")
                row[q] = best.full[i]
                row[q + "_err"] = float(np.sqrt(best.cov[i, i])) if best.cov is not None else np.nan
            rows.append(row)
    best.absorbers = pd.DataFrame(rows)
    best.no_absorbers = base
    return best
