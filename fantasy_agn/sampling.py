"""
Hamiltonian Monte Carlo for the posterior of the free parameters.

The posterior is exp(-chi2 / 2) with flat priors within the parameter limits.
Hamiltonian Monte Carlo uses the exact gradient of chi2 (from the analytic
Jacobian) and the covariance matrix of the fit as a preconditioner: in the
whitened coordinates q = L^-1 (theta - theta_best), with L L^T the covariance,
a nearly Gaussian posterior is close to isotropic and HMC produces nearly
independent samples with short trajectories. Trajectories reaching the
parameter limits are reflected at the limit (position and momentum), which
keeps the sampler exact without shrinking the step size. The step size is
tuned during warm-up by dual averaging (Hoffman & Gelman 2014) for an
acceptance rate of about 0.8. The whitening is also relearned from the warm-up
draws, since parameters at a limit have one-sided posteriors that the fit
covariance does not describe. Several chains are run; split-R-hat and the
effective sample size are reported.
"""
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pickle import PicklingError

import numpy as np
from scipy.linalg import solve_triangular

from ._threads import SINGLE_THREADED, environment

__all__ = ("hmc",)

ADAPT_AT = (0.25, 0.5, 0.8)        # fractions of the warm-up where the whitening is updated
DEFAULT_TRAJECTORY = 1.6           # whitened units: about pi/2, independent draws for a Gaussian
JITTER = (0.8, 1.2)                # trajectory length drawn from trajectory * uniform(JITTER)
MAX_LEAP_WARMUP = 256              # leapfrog steps per trajectory during the warm-up (the step size can be tiny early;
                                   # 64 cut the cost but broke convergence on hard blends, no limit cost hours)


def hmc(log_prob_grad, theta0, cov, lb, ub, n_chains=4, n_warmup=300, n_samples=600,
        target_accept=0.8, trajectory=DEFAULT_TRAJECTORY, seed=None, n_jobs=1, max_leap_warmup=None):
    """
    Sample with HMC.

    :param log_prob_grad: function theta -> (log posterior, gradient); it has
        to be picklable for ``n_jobs`` > 1.
    :param theta0: starting point (the best fit).
    :param cov: covariance matrix used to precondition (whiten) the parameters.
    :param lb, ub: parameter limits.
    :param max_leap_warmup: most leapfrog steps per trajectory during the warm-up (default MAX_LEAP_WARMUP;
        0: no limit).
    :param n_jobs: processes running the chains. Every chain has its own random
        stream, so the samples do not depend on it.
    :return: (samples of shape (n_chains * n_samples, n), diagnostics dict)
    """
    n = theta0.size
    cov = np.array(cov, float)
    scale = np.sqrt(np.clip(np.diag(cov), 0, None))
    scale[scale == 0] = 1e-6 * (np.abs(theta0[scale == 0]) + 1)
    corr = cov / np.outer(scale, scale)
    corr[np.diag_indices(n)] = 1.0
    w, V = np.linalg.eigh(corr)
    L = (V * np.sqrt(np.clip(w, 1e-10, None))) * scale[:, None]       # cov ~ L L^T
    cap = MAX_LEAP_WARMUP if max_leap_warmup is None else max_leap_warmup
    tasks = [(log_prob_grad, theta0, L, lb, ub, n_warmup, n_samples, target_accept, trajectory, cap, s)
             for s in np.random.SeedSequence(seed).spawn(n_chains)]
    results = None
    if min(n_jobs, n_chains) > 1:
        try:
            with environment(SINGLE_THREADED), ProcessPoolExecutor(min(n_jobs, n_chains)) as ex:
                results = list(ex.map(_chain, tasks))
        except (RuntimeError, BrokenProcessPool, OSError, PicklingError, AttributeError) as exc:
            warnings.warn(
                f"parallel chains failed ({type(exc).__name__}); running serially. "
                "In scripts, put your code under  if __name__ == '__main__':  to use n_jobs > 1.")
    if results is None:
        results = [_chain(t) for t in tasks]
    chains = np.array([r[0] for r in results])         # (n_chains, n_samples, n)
    info = {"n_chains": n_chains, "n_warmup": n_warmup, "n_samples": n_samples,
            "acceptance": float(np.mean([r[1] for r in results])),
            "step_size": float(np.mean([r[2] for r in results])),
            "r_hat": float(np.nanmax(_split_rhat(chains))),
            "ess": float(np.nanmin(_ess(chains)))}
    info["converged"] = bool(info["r_hat"] < 1.05 and info["ess"] > 100)
    return chains.reshape(-1, n), info


def _chain(task):
    """One chain: (draws, acceptance rate, step size)."""
    log_prob_grad, theta0, L, lb, ub, n_warmup, n_samples, target_accept, trajectory, cap, seed = task
    rng = np.random.default_rng(seed)
    n = theta0.size

    def to_theta(q):
        return theta0 + L @ q

    # The whitening L is updated during the warm-up (only there: a sampler that
    # keeps adapting would no longer sample the posterior).
    # Parameter limits are planes in the whitened coordinates:
    # theta_i = theta0_i + L[i] . q must stay within [lb_i, ub_i].
    norms2 = np.einsum("ij,ij->i", L, L)

    def reflect(q, p):
        """Bounce position and momentum off the parameter limits (Neal 2011)."""
        for _ in range(50):
            th = to_theta(q)
            low, high = th < lb, th > ub
            if not (low.any() or high.any()):
                return q, p
            i = int(np.argmax(np.where(low, lb - th, 0) + np.where(high, th - ub, 0)))
            target = lb[i] if low[i] else ub[i]
            over = th[i] - target                     # signed distance beyond the wall
            q = q - 2 * over * L[i] / norms2[i]
            p = p - 2 * (p @ L[i]) * L[i] / norms2[i]
        return q, None

    def potential(q):
        th = to_theta(q)
        if np.any(th < lb) or np.any(th > ub):
            return np.inf, None
        lp, g = log_prob_grad(th)
        return -lp, -(L.T @ g)

    q = 0.3 * rng.standard_normal(n)
    for _ in range(60):
        if np.isfinite(potential(q)[0]):
            break
        q *= 0.5
    else:
        raise ValueError("HMC: the posterior is zero at the starting point (outside the limits?)")
    U, gU = potential(q)
    eps = 0.5 / n ** 0.25
    mu, h_bar, log_eps_bar, t0, gamma, kappa = np.log(10 * eps), 0.0, 0.0, 10, 0.05, 0.75
    shift = 0
    draws, accepted, window = [], 0, []
    updates = {int(f * n_warmup) for f in ADAPT_AT}
    for it in range(n_warmup + n_samples):
        n_leap = max(1, int(np.ceil(trajectory / eps * rng.uniform(*JITTER))))
        if it < n_warmup and cap:    # a tiny early step size must not cost thousands of gradients per draw
            n_leap = min(n_leap, cap)
        p = rng.standard_normal(n)
        H0 = U + 0.5 * p @ p
        q_new, p_new, g_new = q.copy(), p - 0.5 * eps * gU, gU
        U_new = U
        for step in range(n_leap):
            q_new = q_new + eps * p_new
            q_new, p_new = reflect(q_new, p_new)
            if p_new is None:
                U_new = np.inf
                break
            U_new, g_new = potential(q_new)
            if not np.isfinite(U_new):
                break
            if step < n_leap - 1:
                p_new = p_new - eps * g_new
        if np.isfinite(U_new):
            p_new = p_new - 0.5 * eps * g_new
            H1 = U_new + 0.5 * p_new @ p_new
            accept_prob = float(min(1.0, np.exp(H0 - H1)))
        else:
            accept_prob = 0.0
        if rng.uniform() < accept_prob:
            q, U, gU = q_new, U_new, g_new
            if it >= n_warmup:
                accepted += 1
        if it < n_warmup:            # dual averaging of the step size
            m = it + 1
            k = m - shift
            h_bar = (1 - 1 / (k + t0)) * h_bar + (target_accept - accept_prob) / (k + t0)
            log_eps = mu - np.sqrt(k) / gamma * h_bar
            eps = float(np.exp(log_eps))
            log_eps_bar = k ** -kappa * log_eps + (1 - k ** -kappa) * log_eps_bar
            if it == n_warmup - 1:
                eps = float(np.exp(log_eps_bar))
            window.append(q)
            if m in updates:         # learn the true shape, then tune the step size again
                C = _regularized_cov(np.array(window))
                A = np.linalg.cholesky(C)
                L = L @ A
                norms2 = np.einsum("ij,ij->i", L, L)
                q = solve_triangular(A, q, lower=True)
                U, gU = potential(q)
                if gU is None:           # rounding put the same point just outside a limit
                    q = reflect(q, np.zeros(n))[0]
                    U, gU = potential(q)
                window = []
                mu, h_bar, log_eps_bar, shift = np.log(10 * eps), 0.0, 0.0, m
        else:
            draws.append(to_theta(q))
    return np.array(draws), accepted / n_samples, eps


def _regularized_cov(q):
    """Covariance of warm-up draws, pulled towards the identity when they are few."""
    n = len(q)
    return n / (n + 5.0) * np.cov(q.T).reshape(q.shape[1], -1) + 5.0 / (n + 5.0) * np.eye(q.shape[1])


def _split_rhat(chains):
    """Split-R-hat (Gelman et al. 2013) for each parameter."""
    m, n, _ = chains.shape
    half = n // 2
    split = np.concatenate([chains[:, :half], chains[:, half:2 * half]], axis=0)
    means = split.mean(axis=1)
    within = split.var(axis=1, ddof=1).mean(axis=0)
    between = half * means.var(axis=0, ddof=1)
    var = (half - 1) / half * within + between / half
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(var / within)


def _ess(chains):
    """Effective sample size per parameter from the chain autocorrelation (Geyer)."""
    m, n, d = chains.shape
    out = np.empty(d)
    for k in range(d):
        x = chains[:, :, k] - chains[:, :, k].mean()
        f = np.fft.rfft(x, n=2 * n, axis=1)
        acov = np.fft.irfft(f * np.conj(f), axis=1)[:, :n].mean(axis=0)
        if acov[0] <= 0:
            out[k] = np.nan
            continue
        rho = acov / acov[0]
        tau = 1.0
        for t in range(1, n - 1, 2):
            pair = rho[t] + rho[t + 1]
            if pair < 0:
                break
            tau += 2 * pair
        out[k] = m * n / tau
    return out
