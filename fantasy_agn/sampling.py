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
acceptance rate of about 0.8. Several chains are run; split-R-hat and the
effective sample size are reported.
"""
import numpy as np

__all__ = ("hmc",)


def hmc(log_prob_grad, theta0, cov, lb, ub, n_chains=4, n_warmup=300, n_samples=600,
        target_accept=0.8, trajectory=1.6, seed=None):
    """
    Sample with HMC.

    :param log_prob_grad: function theta -> (log posterior, gradient).
    :param theta0: starting point (the best fit).
    :param cov: covariance matrix used to precondition (whiten) the parameters.
    :param lb, ub: parameter limits.
    :return: (samples of shape (n_chains * n_samples, n), diagnostics dict)
    """
    rng = np.random.default_rng(seed)
    n = theta0.size
    cov = np.array(cov, float)
    scale = np.sqrt(np.clip(np.diag(cov), 0, None))
    scale[scale == 0] = 1e-6 * (np.abs(theta0[scale == 0]) + 1)
    corr = cov / np.outer(scale, scale)
    corr[np.diag_indices(n)] = 1.0
    w, V = np.linalg.eigh(corr)
    L = (V * np.sqrt(np.clip(w, 1e-10, None))) * scale[:, None]       # cov ~ L L^T

    def to_theta(q):
        return theta0 + L @ q

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

    chains, accept_rates, steps = [], [], []
    for _ in range(n_chains):
        q = 0.3 * rng.standard_normal(n)
        while not np.isfinite(potential(q)[0]):
            q *= 0.5
        U, gU = potential(q)
        eps = 0.5 / n ** 0.25
        mu, h_bar, log_eps_bar, t0, gamma, kappa = np.log(10 * eps), 0.0, 0.0, 10, 0.05, 0.75
        draws, accepted = [], 0
        for it in range(n_warmup + n_samples):
            n_leap = max(1, int(np.ceil(trajectory / eps * rng.uniform(0.8, 1.2))))
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
                h_bar = (1 - 1 / (m + t0)) * h_bar + (target_accept - accept_prob) / (m + t0)
                log_eps = mu - np.sqrt(m) / gamma * h_bar
                eps = float(np.exp(log_eps))
                log_eps_bar = m ** -kappa * log_eps + (1 - m ** -kappa) * log_eps_bar
                if it == n_warmup - 1:
                    eps = float(np.exp(log_eps_bar))
            else:
                draws.append(to_theta(q))
        chains.append(np.array(draws))
        accept_rates.append(accepted / n_samples)
        steps.append(eps)
    chains = np.array(chains)                          # (n_chains, n_samples, n)
    info = {"n_chains": n_chains, "n_warmup": n_warmup, "n_samples": n_samples,
            "acceptance": float(np.mean(accept_rates)), "step_size": float(np.mean(steps)),
            "r_hat": float(np.nanmax(_split_rhat(chains))),
            "ess": float(np.nanmin(_ess(chains)))}
    info["converged"] = bool(info["r_hat"] < 1.05 and info["ess"] > 100)
    return chains.reshape(-1, n), info


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
