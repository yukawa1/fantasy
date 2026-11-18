"""
Correlations between spectral quantities over the epochs of one object.

How the broad line responds to the continuum (the intrinsic Baldwin effect, the "breathing" of the broad-line region,
in which the line narrows as the source brightens) and whether a narrow line stays constant (a check of the flux
calibration) are read from how the quantities of the epochs vary together::

    t = fa.epoch_table(me)                       # or a list of FitResults of the same model
    fa.epoch_correlations(t)                     # Spearman rho and p (and Pearson r) for every pair
    fa.plotting.plot_correlations(t, ["cont5100", "Hb_flux", "Hb_fwhm", "Hb_sigma"])

The table adds to the light curve (broad flux, FWHM and EW of each broad line, the continuum) the line dispersion
and centroid shift of the broad lines, R_FeII and the total [OIII]5007 flux, with errors.

With few epochs a correlation is weak evidence: the p-values are those of the rank test with no correction for the
number of pairs tried, so with many quantities some pairs reach p < 0.05 by chance.
"""
import numpy as np
import pandas as pd

__all__ = ("epoch_table", "epoch_correlations", "default_columns", "fit_relation", "shift_continuum",
           "QUANTITY_LABELS")


def _epochs(results):
    from .multiepoch import MultiEpochResult
    if isinstance(results, MultiEpochResult):
        return list(results.epochs), results
    return [r for r in results if r is not None], None


def _epoch_row(task):
    """One row of the epoch table: every quantity of one epoch, each line measured once."""
    k, r, lines, continuum_at = task
    row = {"epoch": k, "mjd": r.spectrum.mjd, "name": r.spectrum.name}
    comps = r._line_components()
    for ln in lines:
        m = r.measure(ln, [c.name for c in comps if c.role == "broad" and ln in c.line_names])
        for key in ("flux", "fwhm", "ew", "sigma", "shift"):
            row[f"{ln}_{key}"], row[f"{ln}_{key}_err"] = m[key], m[key + "_err"]
    if continuum_at:
        row[f"cont{continuum_at:g}"], row[f"cont{continuum_at:g}_err"] = r.continuum(continuum_at)
        if r.spectrum.z and r.spectrum.flux_unit:
            row[f"logL{continuum_at:g}"], row[f"logL{continuum_at:g}_err"] = r.luminosity(continuum_at)
    for comp in r._feii_components():
        band = r._feii_band(comp)
        if band["reference"] in lines:
            f = r.feii(comp.name)
            label = band["ratio_label"]
            row[label], row[label + "_err"] = f["r_feii"], f["r_feii_err"]
    if any("[OIII]5007" in c.line_names for c in comps):
        m = r.measure("[OIII]5007")
        row["[OIII]5007_flux"], row["[OIII]5007_flux_err"] = m["flux"], m["flux_err"]
    return row


def _main_sequence_task(result):
    try:
        return result.main_sequence()
    except Exception:                                             # noqa: BLE001
        return None


def main_sequences(epochs, n_jobs=-1):
    """``r.main_sequence()`` of each result on all cores but one (None where it failed), in the order given."""
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    from ._threads import SINGLE_THREADED, environment
    from .fitting import _n_workers
    epochs = list(epochs)
    workers = _n_workers(n_jobs, len(epochs))
    if workers > 1 and len(epochs) > 3:
        try:
            with environment(SINGLE_THREADED), ProcessPoolExecutor(workers) as ex:
                return list(ex.map(_main_sequence_task, epochs))
        except (RuntimeError, BrokenProcessPool, OSError):
            pass
    return [_main_sequence_task(r) for r in epochs]


def epoch_table(results, lines=None, continuum_at=None, n_jobs=-1):
    """
    One row per epoch: the flux, FWHM, EW, line dispersion and shift of each broad line, the continuum and log L, R_FeII
    and the total [OIII]5007 flux, each with its error.

    :param results: a MultiEpochResult or a list of FitResults of the same model (epochs fitted one by one).
    :param lines: broad lines to measure (default: all of them).
    :param continuum_at: wavelength of the continuum (default: the pivot of the continuum component).
    :param n_jobs: processes (default: all cores but one; 1 = serial).
    """
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    from ._threads import SINGLE_THREADED, environment
    from .fitting import _n_workers
    from .progress import report
    epochs, _ = _epochs(results)
    if len(epochs) < 2:
        raise ValueError("correlations need at least two epochs")
    first = epochs[0]
    broad = [c for c in first._line_components() if c.role == "broad"]
    if lines is None:
        lines = list(dict.fromkeys(ln for c in broad for ln in c.line_names))
    if continuum_at is None:
        continuum_at = next((p for p in (getattr(c, "pivot", None) for c in first.model.components
                                         if c.kind == "continuum") if p), None)
    tasks = [(k, r, lines, continuum_at) for k, r in enumerate(epochs)]
    workers = _n_workers(n_jobs, len(tasks))
    rows = None
    report("measuring the epochs", 0, len(tasks))
    if workers > 1:
        try:
            with environment(SINGLE_THREADED), ProcessPoolExecutor(workers) as ex:
                rows = []
                for k, row in enumerate(ex.map(_epoch_row, tasks), 1):
                    rows.append(row)
                    report("measuring the epochs", k, len(tasks), f"{workers} processes")
        except (RuntimeError, BrokenProcessPool, OSError):
            rows = None
    if rows is None:
        rows = []
        for k, task in enumerate(tasks, 1):
            rows.append(_epoch_row(task))
            report("measuring the epochs", k, len(tasks))
    return pd.DataFrame(rows)


def _is_feii_ratio(col):
    return col == "R_FeII" or (col.startswith("FeII") and col.endswith("/MgII2798"))


def quantity_columns(table):
    """The quantities of an epoch table that have errors (the columns worth correlating), grouped: the continuum,
    the main broad line (flux, FWHM, sigma, shift, EW), R_FeII, the other broad lines, [OIII]."""
    cols = [c for c in table.columns if c + "_err" in table.columns and not c.endswith("_err")]
    lines = [c[:-5] for c in cols if c.endswith("_flux") and not c.startswith("[")]
    main = next((ln for ln in ("Hb", "Ha", "MgII2798", "CIV1549") if ln in lines), lines[0] if lines else None)
    kinds = ("flux", "fwhm", "sigma", "shift", "ew")

    def key(c):
        if c.startswith(("cont", "logL")):
            return (0, 0, c)
        if _is_feii_ratio(c):
            return (2, 0, c)
        line, _, kind = c.rpartition("_")
        k = kinds.index(kind) if kind in kinds else 9
        if line == main:
            return (1, k, c)
        return (4 if line.startswith("[") else 3, lines.index(line) if line in lines else 99, k)
    return sorted(cols, key=key)


def default_columns(table, n=5):
    """The usual first look: the luminosity (or continuum) and the flux, FWHM, sigma and EW of the main broad line."""
    cols = quantity_columns(table)
    main = next((ln for ln in ("Hb", "Ha", "MgII2798", "CIV1549") if f"{ln}_flux" in cols),
                next((c[:-5] for c in cols if c.endswith("_flux") and not c.startswith("[")), None))
    lum = next((c for c in cols if c.startswith("logL")), next((c for c in cols if c.startswith("cont")), None))
    out = [c for c in [lum] + ([f"{main}_{k}" for k in ("flux", "fwhm", "sigma", "ew")] if main else []) if c in cols]
    return out[:n]


def _label(col):
    from .plotting import pretty_line
    if col.startswith("cont"):
        return f"$F_\\lambda$({col[4:]} Å)"
    if col.startswith("logL"):
        return f"log $\\lambda L_{{{col[4:]}}}$"
    if col == "R_FeII":
        return "$R_{\\rm Fe\\,II}$"
    if _is_feii_ratio(col):
        return plain_label(col)
    line, _, kind = col.rpartition("_")
    name = pretty_line(line)
    return {"flux": f"{name} flux", "fwhm": f"{name} FWHM", "sigma": f"{name} $\\sigma$", "shift": f"{name} shift",
            "ew": f"{name} EW"}.get(kind, col)


QUANTITY_LABELS = _label


def plain_label(col):
    """The name of a quantity in plain text (for the app): 'Hβ FWHM', 'log λL5100', 'R(Fe II)'."""
    from .plotting import plain_line
    if col.startswith("cont"):
        return f"F_λ({col[4:]} Å)"
    if col.startswith("logL"):
        return f"log λL{col[4:]}"
    if col == "R_FeII":
        return "R(Fe II)"
    if _is_feii_ratio(col):
        band = col.removeprefix("FeII").split("/")[0].replace("-", "–")
        return f"Fe II {band} Å / Mg II"
    line, _, kind = col.rpartition("_")
    return f"{plain_line(line)} {'σ' if kind == 'sigma' else kind.upper() if kind in ('fwhm', 'ew') else kind}"


def epoch_correlations(table, columns=None):
    """
    Spearman rank correlation (rho, p) and Pearson r for every pair of quantities of an epoch table (epochs with
    a missing value are left out of that pair), strongest first; ``trivial`` marks pairs that are the same number
    twice (the continuum and its luminosity, widths tied in the model), listed last.
    """
    from scipy.stats import pearsonr, spearmanr
    cols = list(columns or quantity_columns(table))
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            x, y = table[a].to_numpy(float), table[b].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            n = int(ok.sum())
            if n < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
                rho = p = r = np.nan
            else:
                rho, p = spearmanr(x[ok], y[ok])
                r = pearsonr(x[ok], y[ok])[0]
            # the same number twice (continuum and its luminosity) or tied in the model (H-gamma and H-beta widths)
            same = {a[:4], b[:4]} <= {"cont", "logL"} or (np.isfinite(r) and abs(r) > 1 - 1e-9)
            rows.append({"x": a, "y": b, "n": n, "rho": float(rho), "p": float(p), "pearson_r": float(r),
                         "trivial": bool(same)})
    out = pd.DataFrame(rows, columns=["x", "y", "n", "rho", "p", "pearson_r", "trivial"])
    return out.sort_values(["trivial", "p"], na_position="last").reset_index(drop=True)


def _use_log(table, col):
    """Logarithmic by default for positive quantities (fluxes, widths, EWs); not for log L, shifts or R_FeII."""
    if col.startswith("logL") or col.endswith("_shift") or _is_feii_ratio(col):
        return False
    v = table[col].to_numpy(float)
    return bool(np.all(v[np.isfinite(v)] > 0))


def relation_data(table, x, y, log="auto"):
    """(x, sx, y, sy, logx, logy) of the epochs where both are measured, in log10 where chosen."""
    lx = _use_log(table, x) if log == "auto" else bool(log)
    ly = _use_log(table, y) if log == "auto" else bool(log)
    vx, vy = table[x].to_numpy(float), table[y].to_numpy(float)
    ex = table[x + "_err"].to_numpy(float) if x + "_err" in table else np.zeros(len(table))
    ey = table[y + "_err"].to_numpy(float) if y + "_err" in table else np.zeros(len(table))
    ok = np.isfinite(vx) & np.isfinite(vy)
    if lx:
        ok &= vx > 0
    if ly:
        ok &= vy > 0
    vx, vy, ex, ey = vx[ok], vy[ok], np.nan_to_num(ex[ok]), np.nan_to_num(ey[ok])
    if lx:
        vx, ex = np.log10(vx), ex / (vx * np.log(10))
    if ly:
        vy, ey = np.log10(vy), ey / (vy * np.log(10))
    return vx, ex, vy, ey, lx, ly, ok


def fit_relation(table, x, y, log="auto", n_boot=500, seed=0):
    """
    The relation y = a + b (x - x0) between two quantities over the epochs, by maximum likelihood with errors in
    both coordinates and an intrinsic scatter (``fantasy_agn.plotting.fit_line``; bootstrap errors). With
    log='auto' fluxes, widths and EWs are taken in log10 (so b is a power-law slope: the Baldwin effect is
    log EW against log L, breathing log FWHM against log L). Returns a dict: a, b, scatter and their errors, x0,
    spearman_rho, spearman_p, n, log_x, log_y.
    """
    from .plotting import fit_line
    vx, ex, vy, ey, lx, ly, _ = relation_data(table, x, y, log)
    if vx.size < 3:
        raise ValueError("a relation needs at least three epochs with both quantities")
    out = fit_line(vx, vy, ex, ey, n_boot=n_boot, seed=seed)
    out.update(log_x=lx, log_y=ly, x=x, y=y)
    return out


def shift_continuum(table, lag):
    """
    The epoch table with the continuum (and log L) taken ``lag`` days earlier, interpolated linearly in time between
    the epochs: the broad line responds to the continuum one lag earlier, so the line of each epoch is set against the
    continuum it saw. ``lag`` is in observed-frame days (a rest-frame lag times 1 + z). Epochs whose earlier time falls
    before the first epoch get NaN. The errors are interpolated the same way (a lower limit: interpolation between
    sparse epochs adds an uncertainty that is not included).
    """
    t = table.copy()
    lag = float(lag or 0.0)
    if lag == 0.0:
        return t
    mjd = t["mjd"].to_numpy(float)
    order = np.argsort(mjd)
    when = mjd - lag
    inside = (when >= mjd[order][0]) & (when <= mjd[order][-1])
    for c in [c for c in t.columns if c.startswith(("cont", "logL"))]:
        v = t[c].to_numpy(float)[order]
        ok = np.isfinite(v)
        if ok.sum() < 2:
            continue
        t[c] = np.where(inside, np.interp(when, mjd[order][ok], v[ok]), np.nan)
    return t
