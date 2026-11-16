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

__all__ = ("refine", "prune_feii", "feii_detections", "EXTENSIONS")

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


def feii_detections(result, nsigma=2.0):
    """
    The free Fe II multiplet amplitudes of a fit with their errors, and whether each is detected (at
    least ``nsigma`` times its error): the table that :func:`prune_feii` acts on (multiplet, amplitude,
    error, kept). Multiplets outside the fitted range, fixed or tied are not listed.
    """
    table = result.params.set_index("parameter")
    rows = []
    for comp in result._feii_components():
        for p in comp.params:
            q = result.model.params[p.name]
            if not q.linear or not q.free:              # out of the fitted range, fixed or tied by the user
                continue
            value, error = float(table.loc[p.name, "value"]), float(table.loc[p.name, "error"])
            rows.append((p.name, value, error, bool(np.isfinite(error) and error > 0 and value >= nsigma * error)))
    return pd.DataFrame(rows, columns=["multiplet", "amplitude", "error", "kept"])


def prune_feii(result, nsigma=2.0, n_jobs=-1, **fit_options):
    """
    Fit again without the Fe II multiplets that the data do not detect.

    The amplitude of every multiplet group of an :class:`FeII` component is free and cannot be negative.
    Noise then moves a weak or absent group only upwards, so the summed Fe II comes out too high, the
    continuum too low and the broad line next to it too strong. In simulations of an SDSS quasar at
    S/N 30 per pixel fitted with ``presets.hbeta()`` the Fe II flux was 9% too high at R_FeII = 0.25 and
    66% at R_FeII = 0.08, and with no Fe II at all the fit returned R_FeII = 0.12 and a broad H-beta flux
    12% too high; with Fe II three times stronger, or noise three times lower, the bias was below 2%.

    This step fixes at zero the multiplets whose amplitude is below ``nsigma`` times its error (or at its
    bound) and fits the model again from the solution it was given::

        r = fa.fit(spec, fa.presets.hbeta())
        r = fa.prune_feii(r)
        r.feii_pruning        # multiplet, amplitude, error, kept or dropped

    In the same simulations it removed most of the bias where it is large (broad H-beta flux with no
    Fe II: +15% -> +1%; at R_FeII = 0.08: +6% -> 0%, with a 3 times smaller total error). It has a cost
    where Fe II is moderate: real but weak multiplets are dropped too, and at R_FeII = 0.25 the Fe II flux
    came out 13% too low instead of 6% too high (broad H-beta -2% instead of +1%), with the same total
    error. It is therefore a choice, not a default: use it for weak Fe II and low S/N, and say so with the
    results. The numbers are from one spectrum and one model; check your own case with simulations.

    :param result: a FitResult with errors (``errors='covariance'``, 'mc' or 'mcmc'), or a list of them
        (the epochs of ``fa.fit_many``): each is pruned on its own, in parallel processes, and a list
        comes back in the same order; entries that are not fits with Fe II (a failed epoch) are returned
        as they are.
    :param nsigma: a multiplet is kept if its amplitude is at least this many times its error.
    :param n_jobs: processes for a list of fits (-1: all cores but one; 1: serial).
    :param fit_options: passed to :func:`fantasy_agn.fit` (default: the fitted range and the error
        method of ``result``).
    :return: the new FitResult (``result`` itself if nothing is dropped), with ``feii_pruning`` (the
        table) and ``feii_pruned`` (names of the dropped multiplets). The other attributes describe the
        new fit; compare ``bic`` with that of ``result``.
    """
    from .fitting import fit
    if isinstance(result, (list, tuple)):
        return _prune_many(list(result), nsigma, n_jobs, fit_options)
    comps = result._feii_components()
    if not comps:
        raise ValueError("the model has no FeII component")
    if result.cov is None and result.samples is None:
        raise ValueError("prune_feii needs the errors of the fit: fit with errors='covariance', 'mc' or 'mcmc'")
    pruning = feii_detections(result, nsigma)
    drop = list(pruning.multiplet[~pruning.kept])
    if not drop:
        result.feii_pruning, result.feii_pruned = pruning, []
        result.feii_nsigma = float(nsigma)
        return result
    model = result.model.copy()
    for name in drop:
        model.fix(name, 0.0)
    wave = result.spectrum.wave[result.mask]
    fit_options.setdefault("wave_range", (float(wave.min()), float(wave.max())))
    fit_options.setdefault("errors", result.error_method or "covariance")
    fit_options.setdefault("verbose", False)
    new = fit(result.spectrum, model, **fit_options)
    new.feii_pruning, new.feii_pruned = pruning, drop
    new.feii_nsigma = float(nsigma)
    return new


def _prunable(r):
    return hasattr(r, "_feii_components") and bool(r._feii_components()) and (r.cov is not None or r.samples is not None)


def _prune_one(args):
    r, nsigma, options = args
    try:
        return prune_feii(r, nsigma, **options)
    except Exception as exc:                     # noqa: BLE001  (reported, not raised: the other epochs go on)
        return exc


def _prune_many(results, nsigma, n_jobs, fit_options):
    """prune_feii of every fit of a list that has Fe II and errors; the others are returned as they are."""
    import warnings
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    from ._threads import SINGLE_THREADED, environment
    from .fitting import _n_workers
    from .progress import report
    todo = [k for k, r in enumerate(results) if _prunable(r)]
    if not todo:
        raise ValueError("none of the fits has Fe II and errors: fit with errors='covariance' and an FeII component")
    jobs = [(results[k], nsigma, dict(fit_options)) for k in todo]
    task, n = "pruning Fe II", len(jobs)
    workers = _n_workers(n_jobs, n)
    done = None
    report(task, 0, n)
    if workers > 1:
        try:
            with environment(SINGLE_THREADED), ProcessPoolExecutor(workers) as ex:
                done = []
                for k, r in enumerate(ex.map(_prune_one, jobs), 1):
                    done.append(r)
                    report(task, k, n, f"{workers} processes")
        except (RuntimeError, BrokenProcessPool, OSError):
            done = None
    if done is None:
        done = []
        for k, job in enumerate(jobs, 1):
            done.append(_prune_one(job))
            report(task, k, n, "1 process")
    out = list(results)
    for k, r in zip(todo, done):
        if isinstance(r, Exception):
            warnings.warn(f"{results[k].spectrum.name}: Fe II not pruned ({type(r).__name__}: {r})")
        else:
            r.spectrum = results[k].spectrum     # the same object as before (a worker returns a copy)
            out[k] = r
    return out
