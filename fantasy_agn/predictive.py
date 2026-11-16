"""
Is an added component physically supported, or does it only improve the fit?

A component that is nested in a larger model always lowers chi2, and a physical name does not change its
likelihood. What a physical interpretation adds is a set of *linked predictions*: one parameter that changes
several features together. This module tests those links:

``predictive_check``
    Fit every model with one block of the spectrum withheld (a line group, a wavelength region), predict
    that block with its uncertainty and score the prediction by its log predictive density. The gain of a model
    over the baseline, and its advantage over the strongest competitor (``specificity``), measure how much
    information the model transfers from the fitted features to the withheld ones. A block whose prediction
    needs a parameter that only the block itself constrains is flagged: such a model cannot predict it::

        chk = fa.predictive_check(spec, {"base": m0, "physical": mp, "flexible": mf},
                                  blocks={"UV 60": [(2580, 2650)], "UV 62-63": [(2730, 2760)]},
                                  reference="base")
        chk.summary()

``link_test``
    The chi2 cost of the ties that make a component physical (linked model against the same model with
    the links freed); a parametric bootstrap calibrates it where the chi2 distribution does not apply.

``profile_likelihood``
    chi2 against one parameter with all others refitted: is the parameter measured, or only allowed?

``calibrate``
    The distribution of the specificity of a model when the baseline is true, from simulated spectra, and the
    threshold for a chosen false-positive rate. It covers only the procedure and alternatives simulated.

The scores are operational criteria, not probabilities that a component is real; their meaning rests on the
choice of blocks (they must test a link, e.g. a line group predicted from another group) and of competitors.
"""
import numpy as np
import pandas as pd

from .fitting import fit as fit_one

__all__ = ("predictive_check", "PredictiveCheck", "link_test", "profile_likelihood", "calibrate")


# ------------------------------------------------------------------ helpers
def _blocks(blocks):
    """{name: [(lo, hi), ...]} from a dict or a list of (lo, hi) ranges."""
    if isinstance(blocks, dict):
        return {str(k): [tuple(map(float, r)) for r in (v if isinstance(v[0], (list, tuple)) else [v])]
                for k, v in blocks.items()}
    return {f"{lo:g}-{hi:g}": [(float(lo), float(hi))] for lo, hi in blocks}


def _in(wave, ranges):
    sel = np.zeros(wave.shape, bool)
    for lo, hi in ranges:
        sel |= (wave >= lo) & (wave <= hi)
    return sel


def _run(jobs, n_jobs, task):
    """fit(spectrum, model, **options) for every (spectrum, model, options); exceptions are returned."""
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    from ._threads import SINGLE_THREADED, environment
    from .compare import _fit_named
    from .fitting import _n_workers
    from .progress import report
    workers = _n_workers(n_jobs, len(jobs))
    out = None
    report(task, 0, len(jobs))
    if workers > 1:
        try:
            with environment(SINGLE_THREADED), ProcessPoolExecutor(workers) as ex:
                out = []
                for k, res in enumerate(ex.map(_fit_named, jobs), 1):
                    out.append(res)
                    report(task, k, len(jobs), f"{workers} processes")
        except (RuntimeError, BrokenProcessPool, OSError):
            out = None
    if out is None:
        out = []
        for k, job in enumerate(jobs, 1):
            out.append(_fit_named(job))
            report(task, k, len(jobs))
    return out


def _fit_range(spectrum, model, wave_range):
    lo, hi = wave_range or getattr(model, "wave_range", None) or (-np.inf, np.inf)
    return (spectrum.wave >= lo) & (spectrum.wave <= hi)


def _unconstrained(result, model, train, test, ratio=1e-3):
    """
    Free parameters of the model as given (``model``; the fit itself fixes an amplitude that no training pixel
    constrains) that matter for the test pixels but are not constrained by the training pixels: the Fisher
    information sum((d model / d p / err)^2) on the training pixels is below ``ratio`` times that on the test
    pixels (scale-free: the information per unit depends on the units of the parameter). Each free parameter is
    perturbed together with everything tied to it, so a group amplitude constrained through a tie to fitted
    features counts as constrained.
    """
    s, m = result.spectrum, result.model
    T, _, free = model.params.affine()
    x = s.wave[train | test]
    err = s.err[train | test]
    is_test = test[train | test]
    f0 = m.evaluate(x, s.context, result.full)
    names = []
    for k, p in enumerate(free):
        i = model.params.index(p.name)
        h = 1e-3 * max(abs(result.full[i]), 1e-3 * (abs(p.max - p.min) if np.isfinite(p.max - p.min) else 1.0), 1e-6)
        sign = 1.0 if result.full[i] + h <= p.max else -1.0
        d = (m.evaluate(x, s.context, result.full + sign * h * T[:, k]) - f0) / (sign * h) / err
        i_test, i_train = np.sum(d[is_test] ** 2), np.sum(d[~is_test] ** 2)
        if i_test > 0 and i_train < ratio * i_test:
            names.append(p.name)
    return names


def _log_predictive(result, test, scale_noise=True):
    """Gaussian log predictive density of the test pixels: noise (scaled as in the fit) + parameter covariance."""
    s = result.spectrum
    x, y = s.wave[test], s.flux[test]
    mu = result.model.evaluate(x, s.context, result.full)
    noise = (s.err[test] * (result.error_scale if scale_noise else 1.0)) ** 2
    _, cp = result.propagate(lambda f: result.model.evaluate(x, s.context, f), covariance=True)
    cp = np.nan_to_num(np.atleast_2d(cp)) if cp.size == x.size ** 2 else np.zeros((x.size, x.size))
    C = np.diag(noise) + cp
    r = y - mu
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(C + np.diag(1e-9 * noise))
    z = np.linalg.solve(L, r)
    logp = -0.5 * (z @ z) - np.sum(np.log(np.diag(L))) - 0.5 * x.size * np.log(2 * np.pi)
    return float(logp), float(np.sum(r ** 2 / noise)), float(np.trace(cp) / np.sum(noise))


# ------------------------------------------------------------------ predictive check
class PredictiveCheck:
    """
    Result of :func:`predictive_check`. ``table`` has one row per model and block: number of test pixels,
    log predictive density, chi2 of the prediction, the share of the predictive variance that comes from the
    parameters, and the parameters that only the withheld block constrains (then the block is not
    predicted by that model and is left out of its score).
    """

    def __init__(self, table, reference):
        self.table = table
        self.reference = reference

    @property
    def models(self):
        return list(dict.fromkeys(self.table.model))

    def _scores(self):
        """log predictive density per block (rows) and model (columns); NaN where the model cannot predict."""
        t = self.table.copy()
        t.loc[~t.predictable, "logp"] = np.nan
        return t.pivot(index="block", columns="model", values="logp")[self.models]

    @property
    def lpred(self):
        """Summed log predictive density per model over the blocks it predicts (compare models with gain)."""
        return self._scores().sum(min_count=1)

    def gain(self, model, reference=None):
        """
        2 [lpred(model) - lpred(reference)] over the blocks that both models predict, its standard error from
        the scatter over those blocks (NaN for one block) and the number of blocks used.
        """
        ref = reference or self.reference
        sc = self._scores()
        d = (2 * (sc[model] - sc[ref])).dropna()
        se = float(np.sqrt(len(d)) * d.std(ddof=1)) if len(d) > 1 else np.nan
        return (float(d.sum()) if len(d) else np.nan), se, int(len(d))

    def specificity(self, model):
        """
        Advantage of ``model`` over its strongest competitor: the smallest gain against any other model that
        predicts at least one block in common. Returns (gain, standard error, competitor, number of blocks);
        competitors that predict no block are not candidates (see ``table``).
        """
        best = None
        for m in self.models:
            if m == model:
                continue
            g = self.gain(model, m)
            if g[2] and (best is None or g[0] < best[0]):
                best = (g[0], g[1], m, g[2])
        return best or (np.nan, np.nan, "", 0)

    def summary(self):
        rows = []
        for m in self.models:
            sub = self.table[self.table.model == m]
            g = self.gain(m) if self.reference and m != self.reference else (np.nan, np.nan, 0)
            s = self.specificity(m) if len(self.models) > 1 else (np.nan, np.nan, "", 0)
            rows.append({"model": m, "lpred": float(self.lpred[m]), "gain": g[0], "gain_se": g[1],
                         "specificity": s[0], "specificity_se": s[1], "strongest_competitor": s[2],
                         "chi2_test": float(sub.chi2_test.sum()), "unpredicted_blocks": int((~sub.predictable).sum())})
        return pd.DataFrame(rows).set_index("model")

    def __repr__(self):
        n = self.table.block.nunique()
        return f"PredictiveCheck({len(self.models)} models, {n} blocks, reference={self.reference!r})"


def predictive_check(spectrum, models, blocks, reference=None, wave_range=None, scale_noise=True,
                     n_jobs=-1, **fit_options):
    """
    Predict withheld blocks of a spectrum with several models and score the predictions.

    Every model is fitted once per block with that block masked; the block is then predicted with the
    Gaussian predictive density N(model, noise + J C J^T), where C is the covariance of the training fit,
    so a confident but wrong prediction is penalized. A model whose prediction of a block needs a parameter
    constrained only by the block (e.g. a free amplitude of a group that lies inside it) cannot predict it:
    the row is flagged and the block is left out of that model's scores; models are compared on the blocks
    both predict. The gain over the baseline shows that a model transfers information between blocks; that
    the transfer follows the proposed link is tested by :func:`link_test` (a model can beat an empty baseline
    even when its link is violated).

    :param models: dict {name: Model}; include the baseline and plausible alternatives (a flexible
        empirical component, a continuum correction, another mechanism).
    :param blocks: dict {name: [(lo, hi), ...]} or a list of (lo, hi): the withheld regions. They should
        test a physical link (features that the proposed physics connects), not interpolation inside a line.
    :param reference: name of the baseline model for :meth:`PredictiveCheck.gain`.
    :param wave_range: fitted range (default: each model's own range, else the whole spectrum).
    :param scale_noise: scale the noise of the test pixels by the error scale of the training fit
        (sqrt of its reduced chi2 when > 1), as the fit scales its own errors.
    :param fit_options: passed to :func:`fantasy_agn.fit`.
    :return: PredictiveCheck
    """
    fit_options = {"verbose": False, "errors": "covariance", **fit_options}
    if fit_options["errors"] not in ("covariance", "mc", "mcmc"):
        raise ValueError("predictive_check needs parameter uncertainties (errors='covariance', 'mc' or 'mcmc')")
    if reference is not None and reference not in models:
        raise ValueError(f"reference {reference!r} is not one of the models")
    blocks = _blocks(blocks)
    jobs, keys = _check_jobs(spectrum, models, blocks, wave_range, fit_options)
    fits = _run(jobs, n_jobs, f"predicting {len(blocks)} blocks with {len(models)} models")
    return PredictiveCheck(_check_table(spectrum, models, blocks, wave_range, keys, fits, scale_noise), reference)


def _check_jobs(spectrum, models, blocks, wave_range, fit_options):
    """One fit per model and withheld block."""
    jobs, keys = [], []
    for m in models:
        for b, ranges in blocks.items():
            jobs.append((spectrum.mask(*ranges), models[m], {**fit_options, "wave_range": wave_range}))
            keys.append((m, b))
    return jobs, keys


def _check_table(spectrum, models, blocks, wave_range, keys, fits, scale_noise):
    rows = []
    for (m, b), r in zip(keys, fits):
        rng = _fit_range(spectrum, models[m], wave_range)
        test = spectrum.good & rng & _in(spectrum.wave, blocks[b])
        row = {"model": m, "block": b, "n_test": int(test.sum())}
        if isinstance(r, Exception) or test.sum() == 0:
            row.update(logp=np.nan, chi2_test=np.nan, param_var_frac=np.nan, predictable=False,
                       unconstrained=repr(r) if isinstance(r, Exception) else "no test pixels")
        else:
            train = r.mask & rng
            un = _unconstrained(r, models[m], train, test)
            lp, c2, pv = _log_predictive(r, test, scale_noise)
            row.update(logp=lp, chi2_test=c2, param_var_frac=pv, predictable=not un, unconstrained=", ".join(un))
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ link test
def link_test(spectrum, linked, free, n_sim=0, seed=None, n_jobs=-1, **fit_options):
    """
    Cost of the links that give a component its physical meaning: T = chi2(linked) - chi2(free), where
    ``free`` is the same model with the links released (e.g. untied group amplitudes or ratios).

    The asymptotic p-value uses chi2 with dof = difference in free parameters; it is unreliable when a
    released parameter sits at a bound (an amplitude at zero) or is undefined there. With ``n_sim`` > 0 the
    p-value is calibrated by a parametric bootstrap: spectra simulated from the linked fit with its noise,
    refitted with both models.

    :return: dict with T, dof, p_chi2, p_boot (NaN without simulations), the simulated T and the two fits.
    """
    from scipy.stats import chi2 as chi2_dist
    fit_options = {"verbose": False, **fit_options}
    r_link, r_free = _run([(spectrum, linked, fit_options), (spectrum, free, fit_options)], n_jobs, "link test")
    for r in (r_link, r_free):
        if isinstance(r, Exception):
            raise r
    T = r_link.chi2 - r_free.chi2
    dof = r_free.n_free - r_link.n_free
    out = {"T": float(T), "dof": int(dof), "p_chi2": float(chi2_dist.sf(T, dof)) if dof > 0 else np.nan,
           "p_boot": np.nan, "T_sim": np.array([]), "linked": r_link, "free": r_free}
    if n_sim > 0:
        sims = _simulate(spectrum, r_link, n_sim, seed)
        fits = _run([(s, m, {**fit_options, "errors": None}) for s in sims for m in (linked, free)], n_jobs,
                    f"link test: {n_sim} simulations")
        Ts = np.array([a.chi2 - b.chi2 for a, b in zip(fits[::2], fits[1::2])
                       if not isinstance(a, Exception) and not isinstance(b, Exception)])
        out.update(T_sim=Ts, p_boot=float((1 + np.sum(Ts >= T)) / (1 + Ts.size)))
    return out


def _simulate(spectrum, result, n, seed):
    """Spectra drawn from a fitted model: model + Gaussian noise of the error spectrum (scaled as in the fit)."""
    rng = np.random.default_rng(seed)
    mu = result.model_flux(spectrum.wave)
    sig = spectrum.err * result.error_scale
    return [spectrum.copy(flux=mu + rng.normal(0.0, 1.0, mu.shape) * sig) for _ in range(n)]


# ------------------------------------------------------------------ profile likelihood
def profile_likelihood(spectrum, model, parameter, values, n_jobs=-1, **fit_options):
    """
    chi2 profile of one parameter: the model is refitted with ``parameter`` fixed at each value (all other
    parameters free; ``values`` sorted). Returns a DataFrame (value, chi2, dchi2); ``.attrs`` holds the best grid
    value and the 1-sigma interval (dchi2 = 1, interpolated between grid points). ``at_grid_edge`` is True when
    dchi2 stays below 1 up to an end of the grid: the parameter is not measured on that side.
    """
    fit_options = {"verbose": False, "errors": None, **fit_options}
    jobs = []
    for v in values:
        m = model.copy()
        p = m.params[parameter]
        p.min, p.max = min(p.min, v), max(p.max, v)
        m.fix(parameter, float(v))
        jobs.append((spectrum, m, fit_options))
    fits = _run(jobs, n_jobs, f"profile of {parameter}")
    chi2 = np.array([np.nan if isinstance(r, Exception) else r.chi2 for r in fits])
    df = pd.DataFrame({"value": np.asarray(values, float), "chi2": chi2})
    df["dchi2"] = df.chi2 - np.nanmin(chi2)
    v, d = df.value.to_numpy(), df.dchi2.to_numpy()
    ok = np.isfinite(d)
    v, d = v[ok], d[ok]
    k = int(np.argmin(d))

    def crossing(step):                        # dchi2 = 1, interpolated between grid points, walking from the minimum
        j = k
        while 0 <= j + step < len(v):
            if d[j + step] > 1.0:
                return float(np.interp(1.0, [d[j], d[j + step]], [v[j], v[j + step]])), False
            j += step
        return float(v[j]), True

    (lo, lo_edge), (hi, hi_edge) = crossing(-1), crossing(+1)
    df.attrs["best"] = float(v[k])
    df.attrs["interval"] = (min(lo, hi), max(lo, hi))
    df.attrs["at_grid_edge"] = bool(lo_edge or hi_edge)
    return df


# ------------------------------------------------------------------ calibration
def calibrate(spectrum, null, models, blocks, target, n_sim=50, alpha=0.01, seed=None, reference=None,
              n_jobs=-1, **fit_options):
    """
    False-positive calibration of the specificity of ``target``: spectra are simulated from the fit of the
    baseline model ``null`` (models[null]) with the noise of the spectrum, and the whole predictive check is
    repeated on each. Returns the simulated specificities, the threshold exceeded by a fraction ``alpha`` of
    them and the fraction of simulations with a positive specificity. The calibration covers only the noise
    and the models simulated; add simulations with continuum or template mismatch for those cases.
    """
    fit_options = {"verbose": False, "errors": "covariance", **fit_options}
    blocks = _blocks(blocks)
    base = fit_one(spectrum, models[null], **{**fit_options, "errors": None})
    sims = _simulate(spectrum, base, n_sim, seed)
    jobs, keys = [], []                          # every fit of every simulation in one pool
    for s in sims:
        j, keys = _check_jobs(s, models, blocks, None, fit_options)
        jobs += j
    fits = _run(jobs, n_jobs, f"calibration: {n_sim} simulations")
    n = len(keys)
    scores = []
    for k, s in enumerate(sims):
        table = _check_table(s, models, blocks, None, keys, fits[k * n:(k + 1) * n], True)
        scores.append(PredictiveCheck(table, reference or null).specificity(target)[0])
    scores = np.array(scores, float)
    ok = scores[np.isfinite(scores)]
    return {"specificity_null": scores, "threshold": float(np.quantile(ok, 1 - alpha)) if ok.size else np.nan,
            "alpha": alpha, "fraction_positive": float(np.mean(ok > 0)) if ok.size else np.nan, "n_ok": int(ok.size)}
