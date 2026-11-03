"""
Residual analysis: find what a fit misses and suggest how to change the model.

    t = fa.analyze_residuals(r)          # one fit
    t = fa.analyze_residuals(me)         # a joint multi-epoch fit: features found in several epochs
    print(t[["center", "kind", "sigma", "fwhm", "line", "suggestion"]])

Coherent residual features are searched on several scales (a few pixels for a
missed narrow line up to ~60 pixels for broad wings): the summed normalised
residual over a window is significant when it exceeds ``nsigma`` times the
noise of the residuals themselves (so a fit with underestimated errors does not
report noise as features). Each feature is measured (centre, FWHM, flux,
significance), matched with the line library and turned into a suggestion with
the code that applies it: add a missing line, a broad component or wing, the
stellar host, Fe II or the Balmer continuum, or mask sky residuals and
absorption troughs.
"""
import numpy as np
import pandas as pd

from .lines import LINES
from .wavelengths import vac_to_air

__all__ = ("analyze_residuals",)

C_KMS = 299792.458
SCALES = (3, 7, 15, 31, 61)
# stellar absorption features (vacuum, A)
STELLAR = {"Ca II K": 3934.8, "Ca II H": 3969.6, "G band": 4305.6, "Mg b": 5176.7, "Na D": 5895.9,
           "Ca II 8542": 8544.4}
# lines worth suggesting (the strongest of the library); others are only named
COMMON = {"[OII]3726", "[OII]3729", "[NeIII]3869", "[NeIII]3968", "H8", "He", "Hd", "Hg", "[OIII]4363",
          "HeI4471", "HeII4686", "NIII4640", "Hb", "[OIII]4959", "[OIII]5007", "[NI]5198", "[NI]5200",
          "HeI5876", "[OI]6300", "[OI]6364", "[NII]6548", "Ha", "[NII]6583", "HeI6678", "[SII]6716",
          "[SII]6731", "[NeV]3426", "MgII2798", "CIV1549", "CIII]1909", "Lya", "NV1240", "HeII1640",
          "[FeVII]6087", "[FeX]6375", "[FeVII]5721", "[FeXIV]5303", "OI8446", "[ArIII]7136",
          "[SIII]9069", "[SIII]9531", "HeI7065", "[NeIV]2424", "H9", "H10", "H11", "H12"}
# doublets are added together (an outflow in [OIII]5007 is also in 4959)
DOUBLETS = [("[OIII]4959", "[OIII]5007"), ("[NII]6548", "[NII]6583"), ("[OI]6300", "[OI]6364"),
            ("[SII]6716", "[SII]6731"), ("[OII]3726", "[OII]3729"), ("[NeIII]3869", "[NeIII]3968")]
BOWEN = {"OIII3760", "NIII4100", "NIII4640", "HeII5412"}      # only for models that have Bowen lines (TDEs)
BROAD_CONT = 3000.0       # km/s: wider deficits are continuum mismatches, not absorption lines


def _robust_std(x):
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if x.size else 1.0


def _windows(r, w):
    """Sum of r over a centred window of w pixels, divided by sqrt(w) (significance of a coherent excess)."""
    k = np.ones(w)
    return np.convolve(r, k, mode="same") / np.sqrt(w)


def _library(medium):
    """Lines that residual features are identified with: the library without the weak Fe II, [Fe II],
    Cr II, Ti II ... lines of the FANTASY 1 lists (blends that the Fe II template describes)."""
    from .lines import line_table
    t = line_table()
    weak = t.note.str.contains("FANTASY 1 list") & (t.type != "coronal")
    names = [n for n in LINES if not weak.get(n, False)]
    waves = np.array([LINES[n] for n in names])
    if medium == "air":
        waves = vac_to_air(waves)
    return names, waves


def _model_lines(model):
    """{line name: set of roles} for the lines of the model."""
    out = {}
    for c in model.components:
        for ln in getattr(c, "line_names", []) or []:
            out.setdefault(ln, set()).add(getattr(c, "role", None) or "other")
    return out


def _features(result, nsigma, scales):
    s = result.spectrum
    sel = result.mask & np.isfinite(s.flux)
    x = s.wave[sel]
    model = result.model_flux()[sel]
    excess = s.flux[sel] - model
    r = excess / s.err[sel]
    noise = max(_robust_std(r), 1e-12)
    rn = r / noise
    best = np.zeros(x.size)
    best_w = np.zeros(x.size, int)
    for w in scales:
        if w >= x.size:
            continue
        S = _windows(rn, w)
        # a window of w pixels of independent noise has sigma 1 after the sqrt(w) normalisation;
        # larger windows are more affected by correlated noise: demand a little more
        S = S / (1.0 + 0.1 * np.log2(w / scales[0]))
        better = np.abs(S) > np.abs(best)
        best[better], best_w[better] = S[better], w
    feats = []
    for sign in (1, -1):
        hit = sign * best >= nsigma
        if not hit.any():
            continue
        d = np.diff(np.concatenate([[0], hit.astype(int), [0]]))
        for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
            half = int(best_w[a:b].max()) // 2
            lo, hi = max(0, a - half), min(x.size, b + half)
            feats.append([sign, lo, hi])
    # merge overlapping regions of the same sign
    feats.sort(key=lambda f: (f[0], f[1]))
    merged = []
    for f in feats:
        if merged and merged[-1][0] == f[0] and f[1] <= merged[-1][2]:
            merged[-1][2] = max(merged[-1][2], f[2])
        else:
            merged.append(f)
    out = []
    for sign, lo, hi in merged:
        xs, ex, rs = x[lo:hi], excess[lo:hi], rn[lo:hi]
        w = np.clip(sign * ex, 0, None)
        if w.sum() <= 0 or xs.size < 2:
            continue
        center = float(np.sum(xs * w) / w.sum())
        sigma_l = float(np.sqrt(np.sum(w * (xs - center) ** 2) / w.sum()))
        dl = np.gradient(xs)
        out.append({"center": center, "lo": float(xs[0]), "hi": float(xs[-1]),
                    "kind": "emission" if sign > 0 else "absorption",
                    "sigma": float(abs(rs.sum()) / np.sqrt(rs.size)),
                    "fwhm": float(2.3548 * sigma_l / center * C_KMS),
                    "flux": float(np.sum(ex * dl)), "npix": int(xs.size),
                    "chi2_share": float(np.sum(rs ** 2) / max(np.sum(rn ** 2), 1e-12))})
    return [f for f in out if f["sigma"] >= nsigma], noise


def _sky(spec, center, lo, hi):
    """A residual of a telluric band or sky line: mostly inside it and not much wider."""
    from .spectrum import Spectrum
    z = (spec.z or 0.0) if spec.rest_frame else 0.0
    a, b = lo * (1 + z), hi * (1 + z)
    for t0, t1 in Spectrum.TELLURIC:
        overlap = min(b, t1) - max(a, t0)
        if overlap > 0.5 * (b - a) and (b - a) < 1.5 * (t1 - t0):
            return "telluric band"
    for line in Spectrum.SKY_LINES:
        if abs(center * (1 + z) - line) < 8 and (b - a) < 40:
            return f"night-sky line {line:.0f} Å"
    return None


def _suggest(f, result, lib, model_lines):
    """Identify one feature and suggest a model change (text, action for the app, code)."""
    names, waves = lib
    m = result.model
    kinds = {getattr(c, "kind", "") for c in m.components}
    has_feii = any(type(c).__name__ == "FeII" for c in m.components)
    has_balmer = any(type(c).__name__ == "BalmerContinuum" for c in m.components)
    c0, fwhm = f["center"], f["fwhm"]
    rng = (round(f["lo"] - 2), round(f["hi"] + 2))
    mask = {"type": "mask", "range": list(rng)}
    sky = _sky(result.spectrum, c0, f["lo"], f["hi"])
    f.update(line="", in_model=False, action=None, code="")
    if sky:
        f["kind"] = "sky"
        on = [n for n, w in zip(names, waves) if n in model_lines and f["lo"] - 5 <= w <= f["hi"] + 5]
        if on:
            f["suggestion"] = (f"residual of a {sky} (observed frame) on {', '.join(on[:3])}: masking would remove "
                               "the line there; better keep it and treat this part of the line with care")
        else:
            f["suggestion"] = f"residual of a {sky} (observed frame): mask {rng[0]}\u2013{rng[1]} \u00c5"
            f["action"], f["code"] = mask, f"spectra = [s.mask({rng}) for s in spectra]"
        return f
    tol = min(max(600.0, 0.6 * fwhm), 3000.0)
    dv = (waves - c0) / c0 * C_KMS
    bowen = any(ln in BOWEN for ln in model_lines)
    near = [i for i in np.argsort(np.abs(dv)) if abs(dv[i]) < tol
            and (names[i] not in BOWEN or bowen)
            # forbidden lines are never that broad, unless the feature is centred on one (a wing merged
            # with neighbouring residuals)
            and not (names[i].startswith("[") and fwhm > 2500 and abs(dv[i]) > 1500)]
    common = [i for i in near if names[i] in COMMON or names[i] in BOWEN] or near
    if f["kind"] == "absorption" and fwhm > BROAD_CONT and f["hi"] - f["lo"] > 60:
        f["kind"] = "continuum"
        f["suggestion"] = (f"the model is too high over {f['lo']:.0f}\u2013{f['hi']:.0f} \u00c5 (continuum shape): "
                           "check the continuum, host and Fe II; between epochs a wavelength-dependent "
                           "calibration (calibration_degree=1)")
        return f
    if f["kind"] == "absorption":
        star = min(STELLAR.items(), key=lambda kv: abs(kv[1] - c0))
        if abs(star[1] - c0) / c0 * C_KMS < max(800.0, 0.6 * fwhm):
            f["line"] = star[0]
            if "host" in kinds:
                f["suggestion"] = (f"stellar {star[0]} deeper than the host model: check the host templates "
                                   "and σ⋆ (or a younger/older population)")
            else:
                f["suggestion"] = f"stellar {star[0]} absorption: the model has no host galaxy; add one"
                f["action"] = {"type": "component", "component": "StellarPopulation"}
                f["code"] = "model.add(fa.StellarPopulation())"
            return f
        if common and names[common[0]] in model_lines:
            ln = names[common[0]]
            f["line"], f["in_model"] = ln, True
            f["suggestion"] = (f"the model over-predicts {ln} here (too strong or too broad): check its width, "
                               "ties and flux ratios")
            return f
        f["suggestion"] = (f"absorption trough at {c0:.0f} Å (FWHM {fwhm:.0f} km/s): fit absorbers "
                           "(fa.fit_absorbed) or mask it")
        f["action"], f["code"] = mask, f"spectra = [s.mask({rng}) for s in spectra]"
        return f
    # emission
    if fwhm > 6000 and f["hi"] - f["lo"] > 150 and not common:
        f["kind"] = "continuum"
        if c0 < 3800 and not has_balmer:
            f["suggestion"] = "broad excess blueward of 3800 Å: add the Balmer continuum"
            f["action"] = {"type": "component", "component": "BalmerContinuum"}
            f["code"] = "model.add(fa.BalmerContinuum())"
        elif (4400 < c0 < 5600 or 2200 < c0 < 3100) and not has_feii:
            f["suggestion"] = f"broad excess around {c0:.0f} Å: add Fe II emission"
            f["action"] = {"type": "component", "component": "FeII"}
            f["code"] = "model.add(fa.FeII())"
        else:
            f["suggestion"] = (f"broad continuum mismatch around {c0:.0f} Å: a smooth calibration "
                               "polynomial or a different continuum shape")
        return f
    if not common:
        f["suggestion"] = (f"unidentified emission at {c0:.0f} Å (FWHM {fwhm:.0f} km/s, "
                           f"{f['sigma']:.0f}σ): no library line within {tol:.0f} km/s")
        return f
    ln = names[common[0]]
    f["line"] = ln
    roles = model_lines.get(ln, set())
    if ln in ("H8", "H9", "H10", "H11", "H12") and any(type(c).__name__ == "HighOrderBalmer" for c in m.components):
        f["in_model"] = True
        f["suggestion"] = (f"high-order Balmer lines (near {ln}) are under-fitted: free the width of the "
                           "high-order Balmer template or fit them as lines")
        return f
    forbidden = ln.startswith("[")
    group = next((list(d) for d in DOUBLETS if ln in d), [ln])
    f["in_model"] = bool(roles)
    safe = "".join(ch for ch in ln if ch.isalnum()).lower()
    if not roles:
        role = ("outflow" if fwhm > 1000 else "narrow") if forbidden else ("broad" if fwhm > 1200 else "narrow")
        f["suggestion"] = f"{ln} is missing from the model: add it as a {role} line (FWHM about {fwhm:.0f} km/s)"
        f["action"] = {"type": "lines", "name": f"{role}_{safe}", "lines": group, "role": role}
        f["code"] = f"model.add(fa.Lines({role + '_' + safe!r}, {group!r}, role={role!r}))"
    elif forbidden and "outflow" not in roles:
        shift = (c0 - LINES.get(ln, c0)) / c0 * C_KMS
        f["suggestion"] = (f"{ln} has an unfitted {'blue' if shift < 0 else 'red'} wing (FWHM {fwhm:.0f} km/s, "
                           f"{shift:+.0f} km/s): add an outflow component")
        f["action"] = {"type": "lines", "name": f"outflow_{safe}", "lines": group, "role": "outflow"}
        f["code"] = f"model.add(fa.Lines('outflow_{safe}', {group!r}, role='outflow'))"
    elif fwhm > 2000 and "broad" in roles:
        f["suggestion"] = (f"broad {ln} is under-fitted (wings or asymmetry, FWHM {fwhm:.0f} km/s, "
                           f"shift {(c0 - LINES.get(ln, c0)) / c0 * C_KMS:+.0f} km/s): add another broad component")
        f["action"] = {"type": "lines", "name": f"broad_{safe}_wing", "lines": [ln], "role": "broad"}
        f["code"] = f"model.add(fa.Lines('broad_{safe}_wing', [{ln!r}], role='broad'))"
    elif fwhm > 1200 and "broad" not in roles:
        f["suggestion"] = f"a broad {ln} is missing (FWHM {fwhm:.0f} km/s): add a broad component"
        f["action"] = {"type": "lines", "name": f"broad_{safe}", "lines": [ln], "role": "broad"}
        f["code"] = f"model.add(fa.Lines('broad_{safe}', [{ln!r}], role='broad'))"
    elif "outflow" not in roles:
        f["suggestion"] = (f"narrow {ln} is not matched (FWHM {fwhm:.0f} km/s): add an outflow/second "
                           "narrow component, or free its flux ratio")
        f["action"] = {"type": "lines", "name": f"outflow_{safe}", "lines": group, "role": "outflow"}
        f["code"] = f"model.add(fa.Lines('outflow_{safe}', {group!r}, role='outflow'))"
    else:
        f["suggestion"] = f"{ln} is not matched although narrow and outflow components exist: check ties and limits"
    return f


def _one(result, nsigma, scales, max_features):
    feats, noise = _features(result, nsigma, scales)
    lib = _library(result.spectrum.medium)
    ml = _model_lines(result.model)
    feats = sorted(feats, key=lambda f: -f["sigma"])[:max_features]
    return [_suggest(f, result, lib, ml) for f in feats], noise


COLUMNS = ["center", "lo", "hi", "kind", "sigma", "fwhm", "flux", "chi2_share", "line", "in_model",
           "suggestion", "code", "action", "epochs"]


def analyze_residuals(result, nsigma=4.0, scales=SCALES, max_features=12):
    """
    Significant residual features of a fit, with suggestions (strongest first).

    :param result: a FitResult, or a MultiEpochResult (or list of FitResults): features are then
        found in each epoch and merged; ``epochs`` lists where each was found.
    :param nsigma: detection threshold, relative to the scatter of the residuals.
    :param scales: window widths (pixels) for coherent features.
    :return: DataFrame with center, lo, hi (A), kind (emission, absorption, sky), sigma
        (significance), fwhm (km/s), flux, chi2_share, line (identified), in_model, suggestion,
        code (to apply it), action (for the app) and epochs.
    """
    from .result import FitResult
    epochs = getattr(result, "epochs", None)
    if isinstance(result, FitResult):
        feats, _ = _one(result, nsigma, scales, max_features)
        for f in feats:
            f["epochs"] = [0]
        return pd.DataFrame(feats, columns=COLUMNS)
    results = epochs if epochs is not None else list(result)
    groups = []
    for k, r in enumerate(results):
        if not isinstance(r, FitResult):
            continue
        feats, _ = _one(r, nsigma, scales, max_features)
        for f in feats:
            for g in groups:
                tol = max(15.0, 0.5 * (g["hi"] - g["lo"]))
                if g["kind"] == f["kind"] and abs(g["center"] - f["center"]) < tol:
                    g["members"].append((k, f))
                    break
            else:
                groups.append({"center": f["center"], "lo": f["lo"], "hi": f["hi"], "kind": f["kind"],
                               "members": [(k, f)]})
    rows = []
    for g in groups:
        ks = sorted({k for k, _ in g["members"]})
        best = max(g["members"], key=lambda kf: kf[1]["sigma"])[1].copy()
        best["center"] = float(np.median([f["center"] for _, f in g["members"]]))
        best["lo"] = float(min(f["lo"] for _, f in g["members"]))
        best["hi"] = float(max(f["hi"] for _, f in g["members"]))
        if best.get("action") and best["action"].get("type") == "mask":
            best["action"] = dict(best["action"], range=[round(best["lo"] - 2), round(best["hi"] + 2)])
        best["epochs"] = ks
        rows.append(best)
    # features found in many epochs first, then by significance
    rows.sort(key=lambda f: (-len(f["epochs"]), -f["sigma"]))
    return pd.DataFrame(rows[:max_features * 2], columns=COLUMNS)
