"""
Fast search for emission lines a fit is missing.

    t = fa.suggest_lines(r)              # lines worth adding, strongest first (milliseconds)
    print(t[["line", "component", "snr", "dbic"]])
    exec(t.attrs["code"])                # adds them, tied to the kinematics they were found with

Every library line inside the fitted range that the model does not have is tested
as a Gaussian with the fitted width and velocity shift of a line component the
model already has: forbidden and coronal lines with the narrow and outflow
components, permitted lines also with the broad ones. With the shape fixed, the
best flux and the chi-square it removes are linear (no refit), so the whole library
takes milliseconds. The chi-square gain is divided by the scatter of the
normalised residuals (errors that are too small do not make noise look like
lines) and turned into a BIC change, one free parameter per line: a line is
suggested when it lowers the BIC by more than ``threshold`` (10, strong evidence).
Lines are picked one at a time and each picked line is subtracted before the
next is judged, so blends are not counted twice. Lines on top of a line the
model already has, on night-sky lines, or covered by a template the model has
(Fe II, high-order Balmer) are not tested.
"""
import re

import numpy as np
import pandas as pd

from .residuals import BOWEN, DOUBLETS, _library, _sky

__all__ = ("suggest_lines",)

C_KMS = 299792.458
FORBIDDEN = {"forbidden", "coronal", "semi-forbidden"}
COLUMNS = ["action", "line", "lines", "wave", "ion", "type", "component", "role", "flux", "flux_err", "snr",
           "dv", "dbic", "code"]


def _sources(model, values):
    """(component, role, fwhm, shift) of the line components with one shared width and shift."""
    out = []
    for c in model.components:
        if type(c).__name__ != "Lines" or getattr(c, "absorption", False):
            continue
        fw, sh = values.get(c.name + ".fwhm"), values.get(c.name + ".shift", 0.0)
        if fw and np.isfinite(fw) and fw > 0:
            out.append((c.name, getattr(c, "role", None) or "narrow", float(fw), float(sh or 0.0)))
    return out


def _present(model, values, medium):
    """Names and observed-frame (rest) centres of the lines the model has."""
    from .lines import LINES
    from .wavelengths import vac_to_air
    names, centres = set(), []
    for c in model.components:
        lines = list(getattr(c, "line_names", []) or [])
        names |= set(lines)
        if getattr(c, "absorption", False):
            continue
        sh = values.get(c.name + ".shift", 0.0) or 0.0
        for ln in lines:
            if ln in LINES:
                w = LINES[ln] if medium != "air" else float(vac_to_air(np.array([LINES[ln]]))[0])
                centres.append(w * (1 + sh / C_KMS))
    return names, np.array(centres)


def _gauss_columns(x, lams, fluxes, shift, fwhm):
    """Derivatives of a sum of Gaussian lines (shared shift and FWHM) at x: one column per line flux
    (per unit flux), then d/d shift and d/d FWHM (km/s)."""
    cols, dv, dw = [], np.zeros_like(x), np.zeros_like(x)
    for lam, f in zip(lams, fluxes):
        c0 = lam * (1 + shift / C_KMS)
        sig = c0 * fwhm / C_KMS / 2.3548
        u = (x - c0) / sig
        g = np.where(np.abs(u) < 5, np.exp(-0.5 * u * u), 0.0) / (sig * np.sqrt(2 * np.pi))
        cols.append(g)
        dv += f * g * u / sig * c0 / C_KMS
        dw += f * g * (u * u - 1) / fwhm
    return cols, dv, dw


def _gain(cols, w, res):
    """Chi-square removed by the weighted least-squares step along the columns."""
    if not cols:
        return 0.0, None
    D = np.array(cols)
    A = (D * w) @ D.T
    b = (D * w) @ res
    step = np.linalg.lstsq(A, b, rcond=None)[0]
    return float(b @ step), step


def _offsets(result, x, res, w, scale, roles=("narrow", "outflow")):
    """Lines (or doublets) whose velocity or width differs from that of the other lines of their
    component: [(component, role, group, line, rest wavelength, dv km/s, significance)].

    A linearised refit of the lines: the residuals are projected on the derivatives of every line
    component (each line flux, the shift and the width), with and without a shift and width of the
    group's own; the difference of the chi-square they remove (divided by the residual scatter) is
    the evidence for separating the group."""
    from .lines import LINES
    from .wavelengths import vac_to_air
    m, s = result.model, result.spectrum
    P = {p.name: p.value for p in m.params}
    air = getattr(s, "medium", "vacuum") == "air"
    wave = lambda n: float(vac_to_air(np.array([LINES[n]]))[0]) if air else LINES[n]
    comps = []
    for c in m.components:
        fw = P.get(c.name + ".fwhm")
        if type(c).__name__ != "Lines" or getattr(c, "absorption", False) or not fw or fw <= 0:
            continue
        lines = [n for n in (c.line_names or []) if n in LINES]
        comps.append((c, lines, [wave(n) for n in lines], [max(P.get(f"{c.name}.{n}.flux") or 0.0, 0.0) for n in lines],
                      P.get(c.name + ".shift", 0.0) or 0.0, fw))
    if not comps:
        return []
    # every line component re-adjusts (fluxes, shift, width), to first order
    base = []
    for c2, n2, l2, f2, sh2, fw2 in comps:
        cols, dv, dw = _gauss_columns(x, l2, f2, sh2, fw2)
        base += [col for col, f in zip(cols, f2)] + [dv, dw]
    g0, _ = _gain(base, w, res)
    out = []
    for c, names, lams, fluxes, shift, fwhm in comps:
        if (len(names) < 2 or not getattr(c, "_args", None) or (getattr(c, "role", None) or "narrow") not in roles):
            continue
        done = set()
        for ln in names:
            if ln in done:
                continue
            group = next((list(d) for d in DOUBLETS if ln in d and all(n in names for n in d)), [ln])
            done |= set(group)
            if len(group) == len(names):
                continue
            gi = [names.index(n) for n in group]
            if not any(fluxes[i] > 0 for i in gi):
                continue
            _, dv, dw = _gauss_columns(x, [lams[i] for i in gi], [fluxes[i] for i in gi], shift, fwhm)
            g1, step = _gain(base + [dv, dw], w, res)
            gain = g1 - g0
            if not np.isfinite(gain) or gain <= 0:
                continue
            main = max(gi, key=lambda i: fluxes[i])
            out.append((c.name, getattr(c, "role", None) or "narrow", group, names[main], lams[main],
                        float(step[-2]), np.sqrt(gain / scale)))
    return out


def _split_name(model, comp, line):
    from .lines import line_info
    ion = line_info(line).get("ion") or ""
    # hydrogen by its line (narrow_hb), others by ion (narrow_oiii; not "hi", the high-ionisation component)
    ion = "".join(ch for ch in (line if ion in ("", "H I") else ion) if ch.isalnum()).lower()
    taken = {c.name for c in model.components}
    name, k = f"{comp}_{ion}", 2
    while name in taken:
        name, k = f"{comp}_{ion}{k}", k + 1
    return name


def _skip(model):
    """Lines a template of the model already describes."""
    kinds = {type(c).__name__ for c in model.components}
    out = []
    if "FeII" in kinds:
        out.append(re.compile(r"^\[?FeII"))
    if "HighOrderBalmer" in kinds:
        out.append(re.compile(r"^H(\d+|8|9)$"))
    return out


def _code(rows, model):
    """One Lines component per kinematic source, tied to its width and shift."""
    taken = {c.name for c in model.components}
    out = []
    for comp in dict.fromkeys(r["component"] for r in rows):
        lines = [r["line"] for r in rows if r["component"] == comp]
        role = next(r["role"] for r in rows if r["component"] == comp)
        name, k = f"extra_{comp}", 2
        while name in taken:
            name, k = f"extra_{comp}{k}", k + 1
        taken.add(name)
        out.append(f"model.add(fa.Lines({name!r}, {lines!r}, role={role!r}))\n"
                   f"model.tie({name + '.shift'!r}, {comp + '.shift'!r}).tie({name + '.fwhm'!r}, {comp + '.fwhm'!r})")
    return "\n".join(out)


def suggest_lines(result, threshold=10.0, max_lines=12, snr_min=3.0):
    """
    Emission lines the fit is missing, found by testing every library line at the
    kinematics of the model's own line components (no refitting: milliseconds).

    :param result: a FitResult (for a joint multi-epoch fit pass one epoch, ``me.epochs[k]``).
    :param threshold: a line must lower the BIC by more than this (10: strong evidence).
    :param max_lines: at most this many lines.
    :param snr_min: and have at least this signal-to-noise in flux.
    :return: DataFrame (strongest first) with line, wave (Å, rest), ion, type, component (whose
        width and shift it takes), role, flux, flux_err, snr, dbic and code (one line);
        ``attrs["code"]`` adds all of them, one tied Lines component per kinematic source.
    """
    from .lines import line_info
    from .result import FitResult
    if not isinstance(result, FitResult):
        raise TypeError("suggest_lines takes one FitResult (for a joint fit: me.epochs[k])")
    s, m = result.spectrum, result.model
    msk = np.asarray(result.mask, bool)
    x, y, e = s.wave[msk], s.flux[msk], s.err[msk]
    good = np.isfinite(y) & np.isfinite(e) & (e > 0)
    x, y, e = x[good], y[good], e[good]
    res = y - np.asarray(result.model_flux(s.wave))[msk][good]
    w = 1.0 / e ** 2
    z = res * np.sqrt(w)
    n = x.size
    out = pd.DataFrame(columns=COLUMNS)
    out.attrs["code"] = ""
    if n < 20:
        return out
    scale = max(1.0, (1.4826 * np.median(np.abs(z - np.median(z)))) ** 2)
    values = {p.name: p.value for p in m.params}
    sources = _sources(m, values)
    medium = getattr(s, "medium", "vacuum")
    present, centres = _present(m, values, medium)
    skip = _skip(m)
    bowen = any(ln in BOWEN for ln in present)
    names, waves = _library(medium)
    lo, hi = x.min(), x.max()
    cands = []
    for nm, lam in zip(names, waves):
        if nm in present or (nm in BOWEN and not bowen) or any(p.match(nm) for p in skip):
            continue
        kind = line_info(nm).get("type", "")
        if kind in ("molecular", "feature"):
            continue
        for comp, role, fwhm, shift in sources:
            if role == "broad" and (kind in FORBIDDEN or nm.startswith("[")):
                continue
            c0 = lam * (1 + shift / C_KMS)
            sig = c0 * fwhm / C_KMS / 2.3548
            if not lo + 3 * sig < c0 < hi - 3 * sig:
                continue
            # on top of a line the model has: a mismatch there is that line's shape, not a new line
            if centres.size and np.min(np.abs(centres - c0)) < c0 * fwhm / C_KMS:
                continue
            if _sky(s, c0, c0 - 2 * sig, c0 + 2 * sig):
                continue
            i0, i1 = np.searchsorted(x, [c0 - 5 * sig, c0 + 5 * sig])
            if i1 - i0 < 3:
                continue
            g = np.exp(-0.5 * ((x[i0:i1] - c0) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
            cands.append((nm, float(lam), comp, role, kind, i0, i1, g))
    res0 = res.copy()
    rows = []
    while len(rows) < max_lines and cands:
        best = None
        for cd in cands:
            i0, i1, g = cd[5], cd[6], cd[7]
            wg = w[i0:i1] * g
            sgg = float(wg @ g)
            if sgg <= 0:
                continue
            a = float(wg @ res[i0:i1]) / sgg
            if a <= 0:
                continue
            d = a * a * sgg / scale
            if best is None or d > best[0]:
                best = (d, a, np.sqrt(scale / sgg), cd)
        if best is None:
            break
        d, a, ea, cd = best
        dbic = -d + np.log(n)
        if dbic > -threshold or a / ea < snr_min:
            break
        nm, lam, comp, role, kind, i0, i1, g = cd
        res[i0:i1] -= a * g
        info = line_info(nm)
        rows.append({"action": "add", "line": nm, "lines": [nm], "wave": lam, "ion": info.get("ion", ""), "type": kind,
                     "component": comp, "role": role, "flux": a, "flux_err": float(ea), "snr": a / ea, "dv": np.nan,
                     "dbic": float(dbic)})
        cands = [c for c in cands if c[0] != nm]
    for r in rows:
        r["code"] = _code([r], m)
    # lines the model has whose velocity differs from that of their component ([O III] blueshifted
    # against the other narrow lines): their own shift and width (two parameters)
    splits = []
    for comp, role, group, main, lam, dv, sig in _offsets(result, x, res0, w, scale):
        dbic = -sig * sig + 2 * np.log(n)
        if dbic < -threshold and abs(dv) >= 10:
            info = line_info(main)
            new = _split_name(m, comp, main)
            splits.append({"action": "split", "line": main, "lines": group, "wave": lam, "ion": info.get("ion", ""),
                           "type": info.get("type", ""), "component": comp, "role": role, "flux": np.nan,
                           "flux_err": np.nan, "snr": sig, "dv": dv, "dbic": float(dbic),
                           "code": f"model.split_lines({comp!r}, {group!r}, {new!r})"})
    # one separation per component and round (in a component of two groups, separating either is the
    # same model; the others are judged again after the refit)
    splits.sort(key=lambda r: r["dbic"])
    splits = [r for i, r in enumerate(splits) if r["component"] not in {q["component"] for q in splits[:i]}]
    rows = sorted(rows + splits, key=lambda r: r["dbic"])
    out = pd.DataFrame(rows, columns=COLUMNS)
    adds = [r for r in rows if r["action"] == "add"]
    out.attrs["code"] = "\n".join(c for c in [_code(adds, m)] + [r["code"] for r in splits] if c)
    return out
