"""
Can a simpler model be trusted for a given result?

No fit can show that a model is true: every model of a spectrum is an
approximation, and a tie such as "the narrow Hb and [OIII] widths are equal" is
never exact. What can be estimated is the probability that a simplification
does not change the result beyond a tolerance the user accepts::

    r = fa.fit(spec, flexible_model)
    conf = fa.model_confidence(r, ties=[("narrow.fwhm", "oiii.fwhm")],
                               targets={"broad.fwhm": 60})
    conf.probability   # P(every target moves by less than its tolerance)
    conf.verdict       # 'safe', 'model-dependent', 'changes the result' or 'rejected by the data'
    conf.targets       # predicted shift of each target
    conf.ties          # the same for each tie alone
    conf.refit()       # confirm with a real fit of the simplified model

It needs one fit of the flexible model. To first order, imposing A theta = o
moves the parameters by -C A^T (A C A^T)^-1 (A theta - o), with C the covariance
of the flexible fit, so the shift of each target is linear in the violation of
the ties. Drawing that violation from its distribution gives the probability.

Unlike a model weight exp(-dBIC/2), the probability stays low when the data
cannot tell whether a tie holds but the tie would matter: noisy data do not make
a tie safe. In simulations of a broad + narrow Hb blend the BIC weight said 0.99
when the simplified model was good enough in 54-84% of the cases, whereas this
probability stayed below 0.95 at that signal-to-noise ratio. It tends to be
cautious at low values, since the linear shift can overestimate the real one.

The probability is conditional on the flexible model: if that model is wrong in
the same way as the simplified one (a Gaussian for a non-Gaussian line), nothing
here shows it. ``reference_rchi2`` and :func:`analyze_residuals` are the check.

The shift is first order, so it can be far off for a tie the data strongly reject
(test probability below 0.001). Such a tie is never called safe, whatever its
predicted shift: its verdict is 'rejected by the data', and only :meth:`refit`
can show that it does no harm.

A target or tied parameter without an error (at a limit, or the width of a
component with no flux) has no answer: the verdict is then 'cannot assess'.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2

from .fitting import fit as fit_one

__all__ = ("candidate_ties", "model_confidence", "ModelConfidence")


REJECTED = 1e-3      # below this test probability a shift needs a real fit


def _tie(tie):
    """(param, other or value, factor, offset) from the forms a tie can be given in."""
    name, other, *rest = tie
    return name, other, (float(rest[0]) if rest else 1.0), (float(rest[1]) if len(rest) > 1 else 0.0)


def _label(tie):
    name, other, factor, offset = _tie(tie)
    if not isinstance(other, str):
        return f"{name} = {other:g}"
    return f"{name} = {factor:g} * {other}" + (f" + {offset:g}" if offset else "") if factor != 1 or offset \
        else f"{name} = {other}"


def candidate_ties(result):
    """
    The ties worth testing in a fit: every pair of free widths and of free velocity
    shifts that have errors, without narrow with broad lines (different regions).
    """
    sig = result._sigma()
    params = result.model.params
    role = {c.name: getattr(c, "role", None) for c in result.model.components}
    ties = []
    for kind in (".fwhm", ".shift"):
        names = [p.name for p in params if p.free and p.name.endswith(kind) and sig[params.index(p.name)] > 0]
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if {role.get(a.split(".")[0]), role.get(b.split(".")[0])} != {"narrow", "broad"}:
                    ties.append((a, b))
    return ties


def _gradient(result, target):
    """Gradient of a target (parameter name or function of the full vector) and its value."""
    sig = result._sigma()
    if isinstance(target, str):
        g = np.zeros(result.full.size)
        i = result.model.params.index(target)
        g[i] = 1.0
        return g, float(result.full[i])
    g = np.zeros(result.full.size)
    for i in np.nonzero(sig > 0)[0]:
        a, b = result.full.copy(), result.full.copy()
        a[i] += 0.1 * sig[i]
        b[i] -= 0.1 * sig[i]
        g[i] = (target(a) - target(b)) / (a[i] - b[i])
    return g, float(target(result.full))


def _redundant(result, tie):
    """True if the model already satisfies a tie exactly, e.g. it holds the same tie."""
    name, other, factor, offset = _tie(tie)
    params = result.model.params
    if not isinstance(other, str):
        return params[name].fixed and result.full[params.index(name)] == float(other)
    p, q = params[name], params[other]
    return p.tie is not None and p.tie[0] == other and p.tie[1:] == (factor, offset) or \
        q.tie is not None and p.tie is not None and q.tie[0] == p.tie[0] and factor == 1 and offset == 0 \
        and q.tie[1:] == p.tie[1:]


def _assess(result, ties, grads, tol, rng, n_draws):
    """Predicted shifts and the probability that they all stay within tol."""
    if np.any(np.einsum("ti,ij,tj->t", grads, result.cov, grads) <= 0):
        return None                              # a target without an error cannot be assessed
    names = result.model.params
    sig = result._sigma()
    A = np.zeros((len(ties), result.full.size))
    o = np.zeros(len(ties))
    for k, tie in enumerate(ties):
        name, other, factor, offset = _tie(tie)
        i = names.index(name)
        A[k, i] = 1.0
        if isinstance(other, str):
            A[k, names.index(other)] -= factor
            o[k] = offset
            if sig[names.index(other)] <= 0:
                return None
        else:
            o[k] = float(other)
        if sig[i] <= 0:
            return None
    C = result.cov
    d = A @ result.full - o
    V = A @ C @ A.T
    Vi = np.linalg.pinv(V, hermitian=True)
    M = -grads @ C @ A.T @ Vi                    # target shift = M @ violation
    rank = np.linalg.matrix_rank(V, hermitian=True)
    draws = rng.multivariate_normal(d, V, size=n_draws, method="svd", check_valid="ignore")
    inside = np.abs(draws @ M.T) < tol
    return {"violation": d, "violation_err": np.sqrt(np.diag(V)), "shift": M @ d,
            "shift_err": np.sqrt(np.clip(np.diag(M @ V @ M.T), 0, None)),
            "p_target": inside.mean(axis=0), "probability": float(inside.all(axis=1).mean()),
            "test_p": float(chi2.sf(d @ Vi @ d, rank))}


def _verdict(p, test_p, safe):
    if not np.isfinite(p):
        return "cannot assess"
    if p <= 1 - safe:
        return "changes the result"
    if test_p < REJECTED:            # far from the data the linear shift cannot be trusted
        return "rejected by the data"
    return "safe" if p >= safe else "model-dependent"


class ModelConfidence:
    """
    Result of :func:`model_confidence`: ``probability``, ``verdict``, ``test_p``
    (test of the ties themselves, which says if they hold, not if they matter),
    ``targets`` and ``ties`` tables, ``reference_rchi2`` and :meth:`refit`.
    """

    def __init__(self, result, ties, target_names, targets_in, probability, verdict, test_p, targets, table):
        self.result, self.tie_list = result, list(ties)
        self._names, self._targets = target_names, targets_in
        self.probability, self.verdict, self.test_p = probability, verdict, test_p
        self.targets, self.ties = targets, table
        self.reference_rchi2 = result.rchi2
        self.real = None

    def __repr__(self):
        return (f"ModelConfidence(probability={self.probability:.3f}, verdict={self.verdict!r}, "
                f"ties={len(self.tie_list)}, reference_rchi2={self.reference_rchi2:.2f})")

    def refit(self, **fit_options):
        """Fit the simplified model and compare the real shifts with the predicted ones."""
        model = self.result.model.copy()
        for tie in self.tie_list:
            name, other, factor, offset = _tie(tie)
            if isinstance(other, str):
                model.params.tie(name, other, factor, offset)
            else:
                model.params[name].set(float(other), fixed=True)
        fit_options.setdefault("verbose", False)
        new = fit_one(self.result.spectrum, model, **fit_options)
        value = lambda r, t: float(t(r.full)) if callable(t) else float(r.full[r.model.params.index(t)])
        out = self.targets[["target", "value", "shift"]].rename(columns={"shift": "predicted_shift"})
        out["real_shift"] = [value(new, self._targets[n]) - v for n, v in zip(self._names, out["value"])]
        out.attrs["result"] = new
        self.real = out
        return out


def model_confidence(result, ties, targets, safe=0.95, n_draws=20000, seed=0):
    """
    Probability that simplifying a model leaves the results within tolerance.

    :param result: FitResult of the flexible model, with covariance errors.
    :param ties: list of simplifications of that model: (param, other),
        (param, other, factor), (param, other, factor, offset), for
        param = factor * other + offset, or (param, value) to fix a parameter.
    :param targets: {target: tolerance}. A target is a parameter name, or a name
        mapped to (function of the full parameter vector, tolerance).
    :param safe: probability above which a simplification is called safe, and
        below 1 - safe, that it changes the result.
    :return: :class:`ModelConfidence`.
    """
    if result.cov is None:
        raise ValueError("model_confidence needs a fit with covariance errors")
    ties = list(ties)
    if not ties or not targets:
        raise ValueError("give at least one tie and one target")
    names, funcs, tol = [], {}, []
    for name, spec in targets.items():
        funcs[name], t = spec if isinstance(spec, tuple) else (name, spec)
        names.append(name)
        tol.append(float(t))
    tol = np.array(tol)
    if np.any(tol <= 0):
        raise ValueError("tolerances must be positive")
    pairs = [_gradient(result, funcs[n]) for n in names]
    grads = np.array([g for g, _ in pairs])
    values = [v for _, v in pairs]
    rng = np.random.default_rng(seed)

    rows = []
    done = [t for t in ties if _redundant(result, t)]
    ties = [t for t in ties if t not in done]
    for tie in done:
        rows.append({"tie": _label(tie), "probability": np.nan, "verdict": "already in the model"})
    for tie in ties:
        one = _assess(result, [tie], grads, tol, rng, n_draws)
        p = one["probability"] if one else np.nan
        row = {"tie": _label(tie), "violation": one["violation"][0] if one else np.nan,
               "violation_err": one["violation_err"][0] if one else np.nan,
               "test_p": one["test_p"] if one else np.nan, "probability": p,
               "verdict": _verdict(p, one["test_p"] if one else np.nan, safe)}
        row.update({f"shift {n}": one["shift"][k] if one else np.nan for k, n in enumerate(names)})
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("probability", ascending=False, na_position="last") \
        .reset_index(drop=True)

    joint = _assess(result, ties, grads, tol, rng, n_draws) if ties else None
    nan = np.full(len(names), np.nan)
    p = joint["probability"] if joint else np.nan
    target_table = pd.DataFrame({
        "target": names, "value": values, "tolerance": tol,
        "shift": joint["shift"] if joint else nan, "shift_err": joint["shift_err"] if joint else nan,
        "probability": joint["p_target"] if joint else nan})
    test_p = joint["test_p"] if joint else np.nan
    return ModelConfidence(result, ties, names, funcs, p, _verdict(p, test_p, safe), test_p, target_table, table)
