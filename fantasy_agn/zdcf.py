"""
Z-transformed discrete correlation function (ZDCF; Alexander 1997, 2013).

A port of pyZDCF (Jankov, Kovacevic, Ilic, Sanchez-Saez & Nikutta 2022,
Astronomische Nachrichten 343, e20210090; https://github.com/LSST-sersag/pyzdcf,
MIT licence, Copyright (c) 2022 Isidora Jankov), itself a translation of the
Fortran ZDCF 2.2 code of T. Alexander. The algorithm is unchanged (binning,
correlation coefficients, Fisher's z errors, Monte Carlo averaging, and the
same random numbers for the same seed); it works on arrays instead of files,
has no interactive input or global state, and runs with NumPy 2 and pandas 3.

    t = zdcf((t1, f1, e1), (t2, f2, e2))          # CCF: positive lag = 2 lags 1
    t = zdcf((t1, f1, e1))                        # auto-correlation
"""
import numpy as np
import pandas as pd

__all__ = ("zdcf",)

ENOUGH = 11          # default minimum number of pairs per bin
ISEED = 123456       # seed of the Monte Carlo runs (as in ZDCF and pyZDCF)
RSLUTN = 0.001       # lags closer than RSLUTN x (lag range) are the same lag
EPS = 1e-7
COLUMNS = ["tau", "-sig(tau)", "+sig(tau)", "dcf", "-err(dcf)", "+err(dcf)", "#bin"]


def _light_curve(lc):
    """(t, flux, err) sorted in time, observations at identical times averaged (read_obs)."""
    t, f, e = (np.asarray(a, float) for a in lc)
    df = pd.DataFrame({"t": t, "flux": f, "err": e}).sort_values("t", kind="stable")
    df = df.groupby("t", sort=True).mean().reset_index()
    return df.t.to_numpy(), df.flux.to_numpy(), df.err.to_numpy()


def _rndnrm(rng, n):
    """n standard Gaussian deviates by the Box-Muller method (as pyZDCF, same stream)."""
    n2 = int((n + 1) / 2)
    x1 = rng.uniform(size=n2)
    x2 = rng.uniform(size=n2) * 2 * np.pi
    c = np.sqrt(-2 * np.log(x1))
    out = np.empty(n)
    out[:n2] = c * np.cos(x2)
    out[n2:] = (c * np.sin(x2))[: n - n2]
    return out


def _fishs(r, n):
    """Fisher's small-sample s(z) (Kendall & Stuart, vol. 1, p. 391)."""
    s = 1.0 / (n - 1) * (1 + (4 - r ** 2) / 2.0 / (n - 1) + (22 - 6 * r ** 2 - 3 * r ** 4) / 6 / (n - 1) ** 2)
    return np.sqrt(np.maximum(s, 0.0))


def _fishe(r, n):
    """Fisher's small-sample E(z) (Kendall & Stuart, vol. 1, p. 391)."""
    return np.log((1 + r) / (1 - r)) / 2.0 + r / 2 / (n - 1) * (
        1 + (5 + r ** 2) / 4 / (n - 1) + (11 + 2 * r ** 2 + 3 * r ** 4) / 8 / (n - 1) ** 2)


def _pairs(ta, tb, autocf, no_zero_lag):
    """All pairs (a_i, b_j) with their lags tb_j - ta_i, and the order of increasing lag (tlag_pts)."""
    taus, ia, ib = [], [], []
    for i in range(len(ta)):
        j = np.arange(i, len(tb)) if autocf else np.arange(len(tb))
        tij = tb[j] - ta[i]
        if no_zero_lag:
            j, tij = j[tij != 0], tij[tij != 0]
        taus.append(tij)
        ia.append(np.full(len(j), i))
        ib.append(j)
    # float32 lags as in pyZDCF (and ZDCF), so that equal lags compare as there
    wtau = np.concatenate(taus).astype(np.float32)
    waidx, wbidx = np.concatenate(ia), np.concatenate(ib)
    # pyZDCF sorts with NumPy's heapsort, which is not stable: the order of pairs with equal lags (and
    # so which pairs a bin discards) depends on the NumPy version. A stable sort gives the same bins
    # with every version.
    return wtau.astype(float), waidx, wbidx, np.argsort(wtau, kind="stable")


def _bins(ta, tb, minpts, autocf, uniform, no_zero_lag):
    """
    Allocate the pairs to bins (alcbin): the lag-ordered pairs are divided into
    bins of at least ``minpts`` pairs, a pair is discarded when its a or b point
    is already in the bin, and pairs with the same lag (within the tolerance)
    are not split. For the CCF the binning goes from the median lag up and from
    the median down. Returns the bin lags, their -/+ 1 sigma, the number of
    pairs and the a and b indices of each bin, in increasing lag.
    """
    wtau, waidx, wbidx, idx = _pairs(ta, tb, autocf, no_zero_lag)
    npp = len(wtau)
    if npp < 2:
        raise ValueError("ZDCF: not enough pairs of points")
    tolrnc = float(wtau[idx[npp - 1]] - wtau[idx[0]]) * RSLUTN
    a_n, b_n = len(ta), len(tb)
    out = []                                  # (t, sigm, sigp, n, a indices, b indices)
    if autocf or uniform:
        pfr, pmax, incr = 0, npp - 1, 1
    else:
        pfr, pmax, incr = int(npp / 2) - 1, 0, -1
    nnegtv = None
    while True:
        inb, tij, tsum = 0, wtau[idx[pfr]], 0.0
        a_used, b_used = np.zeros(a_n, bool), np.zeros(b_n, bool)
        members_a, members_b = [], []
        next_bin = False
        i = pfr
        while (i <= pmax) if incr == 1 else (i >= pmax):
            p = idx[i]
            if ((abs(wtau[p] - tij) > tolrnc) and (inb >= minpts or uniform)) or i == pmax:
                # the bin is full: its mean lag and the lags 0.3413 below and above
                t = tsum / inb if inb else 0.0
                if uniform:
                    pfr = i
                    if inb >= minpts:
                        out.append((t, 0.0, 0.0, inb, members_a, members_b))
                    if pfr != pmax:
                        next_bin = True
                    break
                if inb <= 1:                  # a last point alone in its bin is ignored
                    break
                plo, phi = (pfr, i - 1) if incr == 1 else (i + 1, pfr)
                j = phi
                for q in range(plo, phi + 1):
                    if wtau[idx[q]] >= t:
                        j = q                 # the pair closest to the bin's mean lag
                        break
                q = max(j - round(float(j - plo) * 0.3413 * 2), plo)
                sigm = max(t - wtau[idx[q]], 0.0)
                q = min(j + round(float(phi - j) * 0.3413 * 2), phi)
                sigp = max(wtau[idx[q]] - t, 0.0)
                out.append((t, sigm, sigp, inb, members_a, members_b))
                pfr = i
                if pfr != pmax:
                    next_bin = True
                break
            if not a_used[waidx[p]] and not b_used[wbidx[p]]:      # no correlated pairs
                inb += 1
                a_used[waidx[p]] = b_used[wbidx[p]] = True
                tij = wtau[p]
                tsum += float(tij)
                members_a.append(waidx[p])
                members_b.append(wbidx[p])
            i += incr
        if next_bin:
            continue
        if not (autocf or uniform or incr == 1):
            # now the other half of the lag axis
            nnegtv = len(out)
            pfr, pmax, incr = int(npp / 2), npp - 1, 1
            continue
        break
    if nnegtv is not None:                    # the negative-lag bins were found in reverse order
        out = out[:nnegtv][::-1] + out[nnegtv:]
    if not out:
        raise ValueError("ZDCF: no bin with enough pairs")
    return out


def _dcf(fa, fb, bins):
    """Correlation coefficient of every bin and its -/+ 1 sigma from Fisher's z (clcdcf)."""
    r = np.zeros(len(bins))
    for k, (_, _, _, n, ia, ib) in enumerate(bins):
        x, y = fa[ia], fb[ib]
        ea, eb = x.sum() / n, y.sum() / n
        va = (np.sum(x ** 2) - n * ea ** 2) / (n - 1)
        vb = (np.sum(y ** 2) - n * eb ** 2) / (n - 1)
        if va * vb <= 0:
            continue                          # pathological case: r = 0
        c = np.sum((x - ea) * (y - eb)) / np.sqrt(va * vb) / (n - 1)
        r[k] = min(max(c, -1 + EPS), 1 - EPS)
    return r


def zdcf(lc1, lc2=None, minpts=0, num_mc=100, uniform_sampling=False, omit_zero_lags=True, seed=ISEED):
    """
    ZDCF of two light curves (cross-correlation; positive lags: ``lc2`` lags
    ``lc1``) or of one (auto-correlation), following Alexander (1997) as
    implemented in ZDCF 2.2 and pyZDCF (Jankov et al. 2022).

    :param lc1, lc2: (time, flux, error) arrays; observations at the same time are averaged.
    :param minpts: minimum number of pairs per bin (0: the default 11).
    :param num_mc: Monte Carlo runs with the fluxes perturbed by their errors
        (the mean Fisher z over the runs is used); <= 1 for none.
    :param uniform_sampling: the light curves are uniformly sampled (bins of equal lag).
    :param omit_zero_lags: leave out pairs with zero lag.
    :param seed: random seed (123456 gives the numbers of ZDCF and pyZDCF).
    :returns: DataFrame with tau, -sig(tau), +sig(tau), dcf, -err(dcf), +err(dcf), #bin.
    """
    autocf = lc2 is None
    ta, fa, ea = _light_curve(lc1)
    tb, fb, eb = (ta, fa, ea) if autocf else _light_curve(lc2)
    minpts = int(minpts) if int(minpts) > 0 else ENOUGH
    num_mc = int(num_mc) if int(num_mc) > 1 else 0
    bins = _bins(ta, tb, minpts, autocf, bool(uniform_sampling), bool(omit_zero_lags))
    n = np.array([b[3] for b in bins], float)
    if num_mc:
        rng = np.random.RandomState(seed)
        avz = np.zeros(len(bins))
        for _ in range(num_mc):
            ma = fa - ea * _rndnrm(rng, len(fa))
            mb = fb - eb * _rndnrm(rng, len(fb))       # independent also for the auto-correlation (as pyZDCF)
            r = np.clip(_dcf(ma, mb, bins), -1 + EPS, 1 - EPS)
            avz += np.log((1 + r) / (1 - r)) / 2
        r = np.clip(np.tanh(avz / num_mc), -1 + EPS, 1 - EPS)
    else:
        r = _dcf(fa, fb, bins)
    expz, sigz = _fishe(r, n), _fishs(r, n)
    rows = np.column_stack([[b[0] for b in bins], [b[1] for b in bins], [b[2] for b in bins],
                            r, r - np.tanh(expz - sigz), np.tanh(expz + sigz) - r, n])
    return pd.DataFrame(rows, columns=COLUMNS)
