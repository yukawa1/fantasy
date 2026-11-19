"""
Python code for the plots of the Results views (Fit, Atlas, Anatomy, Fe II, Degeneracies, Posterior,
Classification, Main sequence, Light curves, Checks), following what the view shows: its range, the
components hidden or picked out, its mode. Used by the "Python code" box under each view.

Every snippet runs on its own in the session namespace (``r``, ``me``, ``fits``, ``sample``, ...) or in a
script that has these objects, and draws with matplotlib.
"""
import re

KIND_FUNC = '''def kind(c):
    """The kind of a component, as the app colours it."""
    t, role = type(c).__name__, getattr(c, "role", None)
    if getattr(c, "kind", "") == "continuum" and not c.name.startswith("balmer"):
        return "continuum"
    if getattr(c, "kind", "") == "host":
        return "host"
    lines = getattr(c, "line_names", None) or ["-"]
    if role == "narrow" and (c.name.startswith("coronal")
                             or all(n.startswith(("[FeVII]", "[FeX]", "[FeXI]", "[FeXIV]")) for n in lines)):
        return "coronal"
    if role in ("narrow", "broad", "outflow"):
        return role
    if t == "FeII":
        return "feii"
    return "balmer" if t in ("BalmerContinuum", "HighOrderBalmer") else "other"
'''
COLOR = ('COLOR = {"continuum": "#52514e", "host": "#008300", "balmer": "#86b6ef", "feii": "#e34948", '
         '"broad": "#2a78d6", "narrow": "#1baf7a", "coronal": "#b8459c", "outflow": "#eda100", "other": "#898781"}')


def _num(opts, k):
    v = opts.get(k)
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def _names(opts, k):
    return [n for n in str(opts.get(k) or "").split(",") if n]


def _range(opts, var="w"):
    """The mask of the plotted range, and a comment."""
    a, b = _num(opts, "xmin"), _num(opts, "xmax")
    if a is None and b is None:
        return f"view = np.isfinite({var})                          # the whole range"
    cond = ([f"({var} >= {a:.6g})"] if a is not None else []) + ([f"({var} <= {b:.6g})"] if b is not None else [])
    return f"view = {' & '.join(cond)}   # the range in view"


def _head(s, k):
    """How the snippet refers to the fit: 'r', or res = me.epochs[k] / fits[k]."""
    name = s._fit_name(k)
    return ("r", []) if name == "r" else ("res", [f"res = {name}"])


# ------------------------------------------------------------------ Fit
def fit(s, k, opts):
    r, pre = _head(s, k)
    hidden = _names(opts, "hide")
    return "\n".join([
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "", KIND_FUNC, COLOR, "", *pre,
        f"sp = {r}.spectrum",
        "w = sp.wave",
        f"parts = {r}.model_flux(components=True)            # every component, by name",
        f"model = {r}.model_flux()",
        "# the curves of the plot: the continuum summed, the narrow lines summed, every other component alone",
        "curves = {}",
        f"for c in {r}.model.components:",
        "    if c.name not in parts:                        # e.g. an absorber, which multiplies the model",
        "        continue",
        "    kd = kind(c)",
        "    key = kd if kd in (\"continuum\", \"narrow\") else c.name",
        "    curves[key] = (kd, curves.get(key, (kd, 0))[1] + parts[c.name])",
        f"hidden = {set(hidden)!r}" if hidden else "hidden = set()                                  # names clicked off in the legend",
        _range(opts),
        "",
        "fig, (ax, rax) = plt.subplots(2, 1, sharex=True, figsize=(10, 5.5),",
        "                              gridspec_kw={\"height_ratios\": [3.2, 1], \"hspace\": 0.05})",
        "if \"data\" not in hidden:",
        "    ax.plot(w[view], np.where(sp.good, sp.flux, np.nan)[view], color=\"#9b9a95\", lw=0.8, drawstyle=\"steps-mid\", label=\"data\")",
        "for name, (kd, y) in curves.items():",
        "    if name not in hidden and np.any(y[view] != 0):",
        "        ax.plot(w[view], y[view], color=COLOR[kd], lw=1.2, ls=\"--\" if kd in (\"continuum\", \"balmer\") else \"-\", label=name)",
        "        if kd == \"feii\":",
        "            ax.fill_between(w[view], 0, y[view], color=COLOR[kd], alpha=0.15, lw=0)",
        "if \"model\" not in hidden:",
        "    ax.plot(w[view], model[view], color=\"k\", lw=1.1, label=\"model\")",
        "ax.set_ylabel(\"flux\")",
        "ax.legend(fontsize=8, ncol=4)",
        "resid = np.where(sp.good, (sp.flux - model) / np.where(sp.good, sp.err, 1.0), np.nan)   # in sigma",
        "rax.plot(w[view], resid[view], color=\"#52514e\", lw=0.7, drawstyle=\"steps-mid\")",
        "rax.axhline(0, color=\"k\", lw=0.6)",
        "rax.set_ylabel(\"(data - model) / err\")",
        "rax.set_xlabel(\"rest wavelength (Å)\")",
        "plt.show()"])


# ------------------------------------------------------------------ Anatomy
AN_GROUPS = [("continuum", "Continuum"), ("host", "Host galaxy"), ("balmer", "Balmer continuum"), ("feii", "Fe II"),
             ("broad", "Broad lines"), ("narrow", "Narrow lines"), ("coronal", "Coronal lines"),
             ("outflow", "Outflow"), ("other", "Other")]


def anatomy(s, k, opts):
    r, pre = _head(s, k)
    off, solo, peel = _names(opts, "off"), opts.get("solo") or None, int(_num(opts, "peel") or 0)
    show = f"name == {solo!r}" if solo else f"name not in {set(off)!r}" if off else "True"
    return "\n".join([
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "", KIND_FUNC, COLOR, "", *pre,
        f"sp = {r}.spectrum",
        "w = sp.wave",
        f"parts = {r}.model_flux(components=True)",
        f"model = {r}.model_flux()",
        "# the layers: the components summed by kind, stacked from the continuum up",
        f"GROUPS = {AN_GROUPS!r}",
        "layers = {}",
        "for kd, name in GROUPS:",
        f"    y = sum((parts[c.name] for c in {r}.model.components if c.name in parts and kind(c) == kd), np.zeros_like(w))",
        "    if np.any(y > 0):",
        "        layers[name] = (kd, y)",
        f"peel = {peel}                                       # layers peeled off the top (taken from the data)",
        "names = list(layers)",
        "active, peeled = names[:len(names) - peel], names[len(names) - peel:]",
        _range(opts),
        "",
        "# the flux budget: the share of each layer in the model flux in view",
        "tot = {n: np.trapezoid(np.clip(layers[n][1][view], 0, None), w[view]) for n in names}",
        "for n in names:",
        "    print(f\"{n:18s} {100 * tot[n] / sum(tot.values()):5.1f} %\")",
        "",
        "fig, (ax, rax) = plt.subplots(2, 1, sharex=True, figsize=(10, 6),",
        "                              gridspec_kw={\"height_ratios\": [3.2, 1.2], \"hspace\": 0.05})",
        "base = np.zeros(view.sum())",
        "for name in active:",
        "    kd, y = layers[name]",
        f"    if {show}:                                  # the layers switched off or picked out in the view",
        "        top = base + np.clip(y[view], 0, None)",
        "        ax.fill_between(w[view], base, top, color=COLOR[kd], alpha=0.5, lw=0, label=name)",
        "        base = top",
        "data = sp.flux - sum((layers[n][1] for n in peeled), np.zeros_like(w))   # the data minus the peeled layers",
        "ax.plot(w[view], np.where(sp.good, data, np.nan)[view], color=\"k\", lw=0.7, drawstyle=\"steps-mid\", label=\"data\")",
        "ax.set_ylabel(\"F_λ\")",
        "ax.legend(fontsize=8, ncol=3)",
        "# residuals in sigma, with the ±1σ and ±3σ bands and a running mean over 15 pixels",
        "res = np.where(sp.good, (sp.flux - model) / np.where(sp.good, sp.err, 1.0), np.nan)[view]",
        "rax.axhspan(-3, 3, color=\"0.92\", lw=0)",
        "rax.axhspan(-1, 1, color=\"0.82\", lw=0)",
        "rax.plot(w[view], res, color=\"0.45\", lw=0.6, drawstyle=\"steps-mid\")",
        "smooth = np.convolve(np.nan_to_num(res), np.ones(15) / 15, mode=\"same\")",
        "rax.plot(w[view], smooth, color=\"#c8432f\", lw=1.2)",
        "far = np.abs(res) > 3",
        "rax.plot(w[view][far], res[far], \"o\", ms=2.5, color=\"#c8432f\")",
        "rax.set_ylabel(\"σ\")",
        "rax.set_xlabel(\"rest wavelength (Å)\")",
        "plt.show()"])


# ------------------------------------------------------------------ Fe II
def feii(s, k, opts):
    r, pre = _head(s, k)
    full = opts.get("mode") == "full"
    data = str(opts.get("data", "1")) != "0"
    zoom = str(opts.get("zoom", "1")) != "0"
    pin = opts.get("pin") or None
    lines = [
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "", *pre,
        f"sp = {r}.spectrum",
        "w = sp.wave",
        f"groups = {r}.feii_groups()                       # the fitted Fe II by group (lower term)",
        "fe = sum(groups.values())",
        f"model = {r}.model_flux()",
        "other = model - fe                                # everything but the Fe II",
        f"parts = {r}.model_flux(components=True)",
        f"emis = sum((parts[c.name] for c in {r}.model.components if c.name in parts and type(c).__name__ == \"Lines\"",
        "            and not getattr(c, \"absorption\", False)), np.zeros_like(w))   # the emission lines",
        _range(opts),
        "# the groups with Fe II, ordered by their mean wavelength, coloured along it",
        "live = sorted((g for g, y in groups.items() if y.max() > 0), key=lambda g: np.average(w, weights=groups[g]))",
        "colors = dict(zip(live, plt.cm.viridis(np.linspace(0.05, 0.95, len(live)))))",
        f"pin = {pin!r}                                    # a group picked out (clicked), or None",
        "",
        "fig, ax = plt.subplots(figsize=(10, 4.5))",
        f"base = {'other' if full else 'np.zeros_like(w)'}",
    ]
    if full:
        lines += ["ax.plot(w[view], other[view], color=\"0.6\", lw=1, label=\"rest of the model\")"]
    lines += [
        "for g in live:",
        "    if pin is None:",
        "        top = base + groups[g]",
        "        ax.fill_between(w[view], base[view], top[view], color=colors[g], alpha=0.75, lw=0, label=g)",
        "        base = top",
        "if pin is not None:                              # the stack dimmed, the picked group alone from the base",
        "    ax.fill_between(w[view], base[view], (base + fe)[view], color=\"0.85\", lw=0, label=\"all Fe II\")",
        "    ax.fill_between(w[view], base[view], (base + groups[pin])[view], color=colors[pin], alpha=0.9, lw=0, label=pin)",
    ]
    if data:
        lines += [("ax.plot(w[view], np.where(sp.good, sp.flux, np.nan)[view], color=\"k\", lw=0.7, drawstyle=\"steps-mid\", label=\"data\")"
                   if full else
                   "ax.plot(w[view], np.where(sp.good, sp.flux - other, np.nan)[view], color=\"k\", lw=0.7, drawstyle=\"steps-mid\",\n"
                   "        label=\"data − everything but Fe II\")")]
    if full:
        lines += ["ax.plot(w[view], model[view], color=\"#d4472a\", lw=1, label=\"model\")"]
        if zoom:
            lines += [
                "# scaled to the Fe II: from below the continuum to the top of the Fe II away from the strong lines",
                "blend = emis / np.maximum(fe + emis, 1e-30)",
                "cont = (other - emis)[view]",
                "away = view & (blend < 0.3)",
                f"vals = np.concatenate([(other + fe)[away]{', sp.flux[away & sp.good]' if data else ''}])",
                "ymin = 0.85 * cont[cont > 0].min()",
                "ax.set_ylim(ymin, ymin + (np.percentile(vals, 99.5) - ymin) * 1.12)"]
    lines += [
        "ax.set_xlim(w[view].min(), w[view].max())",
        "ax.set_xlabel(\"rest wavelength (Å)\")",
        "ax.set_ylabel(\"flux\")",
        "ax.legend(fontsize=7, ncol=6)",
        "plt.show()",
        "",
        f"print({r}.feii())                                # band flux, EW and R_FeII"]
    return "\n".join(lines)


# ------------------------------------------------------------------ Degeneracies
def degen(s, k, opts):
    r, pre = _head(s, k)
    th = _num(opts, "th") or 0.3
    between = opts.get("scope", "between") != "all"
    return "\n".join([
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "", *pre,
        f"names = list({r}.model.params.names)",
        f"if {r}.samples is not None and len({r}.samples) > 10:     # MCMC samples, else the covariance",
        f"    S = np.asarray({r}.samples, float)",
        "    sig = S.std(axis=0)",
        "else:",
        f"    sig = np.sqrt(np.clip(np.diag({r}.cov), 0, None))",
        f"free = [i for i, q in enumerate({r}.model.params) if q.free and np.isfinite(sig[i]) and sig[i] > 0]",
        f"C = (np.corrcoef(S[:, free], rowvar=False) if {r}.samples is not None and len({r}.samples) > 10",
        f"     else {r}.cov[np.ix_(free, free)] / np.outer(sig[free], sig[free]))",
        "C = np.clip(np.nan_to_num(C), -1, 1)",
        "labels = [names[i] for i in free]",
        "",
        f"threshold = {th:g}",
        f"between = {between}                                # only pairs from different components",
        "pairs = [(abs(C[a, b]), C[a, b], labels[a], labels[b]) for a in range(len(free)) for b in range(a)",
        "         if abs(C[a, b]) >= threshold and (not between or labels[a].split(\".\")[0] != labels[b].split(\".\")[0])]",
        "for _, rho, a, b in sorted(pairs, reverse=True):",
        "    print(f\"{rho:+.2f}  {a}  ~  {b}\")",
        "",
        "show = np.where(np.abs(C) >= threshold, C, np.nan)",
        "if between:",
        "    comp = np.array([n.split(\".\")[0] for n in labels])",
        "    show[comp[:, None] == comp[None, :]] = np.nan",
        "np.fill_diagonal(show, np.nan)",
        "fig, ax = plt.subplots(figsize=(8, 7))",
        "im = ax.imshow(show, cmap=\"RdBu_r\", vmin=-1, vmax=1)",
        "ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)",
        "ax.set_yticks(range(len(labels)), labels, fontsize=6)",
        "fig.colorbar(im, label=\"correlation ρ\")",
        "plt.show()"])


# ------------------------------------------------------------------ Atlas
def atlas(s, k, opts):
    order = [int(v) for v in _names(opts, "order") if v.lstrip("-").isdigit()]
    common, same = str(opts.get("common", "0")) == "1", str(opts.get("same", "0")) == "1"
    kind = s.wf.get("fit_kind")
    if kind == "sample":
        get = ["import fantasy_agn as fa", "", "fit_of = lambda k: fa.load(sample.table.saved.iloc[k])   # the saved fit of row k"]
    else:
        obj = "me.epochs" if kind == "joint" else "fits"
        get = [f"fit_of = lambda k: {obj}[k]"]
    return "\n".join([
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        *get, "",
        f"order = {order!r}                                # the cards shown, in their order",
        "ncols = 4",
        "nrows = -(-len(order) // ncols)",
        f"fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows), squeeze=False,",
        f"                         sharex={common}, sharey={same})",
        "for ax, k in zip(axes.flat, order):",
        "    f = fit_of(k)",
        "    f.plot(ax=ax, residuals=False, legend=False)",
        "    ax.set_title(f\"#{k} {f.spectrum.name}  χ²ᵣ {f.rchi2:.2f}\", fontsize=7)",
        "for ax in axes.flat[len(order):]:",
        "    ax.set_visible(False)",
        "fig.tight_layout()",
        "plt.show()"])


# ------------------------------------------------------------------ Posterior
def posterior(s, k, opts):
    r, pre = _head(s, k)
    params = _names(opts, "params")
    return "\n".join([
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "", *pre,
        f"names = list({r}.model.params.names)",
        f"S = np.asarray({r}.samples)",
        f"params = {params!r}" if params else "params = [n for n in names if np.std(S[:, names.index(n)]) > 0][:8]",
        "sel = [names.index(p) for p in params]",
        "n = len(sel)",
        "fig, axs = plt.subplots(n, n, figsize=(1.6 * n + 0.6, 1.6 * n + 0.6), squeeze=False)",
        "fig.subplots_adjust(hspace=0.06, wspace=0.06)",
        "for i, a in enumerate(sel):",
        "    for j, b in enumerate(sel):",
        "        ax = axs[i, j]",
        "        if j > i:",
        "            ax.set_visible(False)",
        "            continue",
        "        if i == j:",
        "            ax.hist(S[:, a], bins=30, color=\"#4c72b0\", histtype=\"stepfilled\", alpha=0.6)",
        "            q = np.percentile(S[:, a], [15.87, 50, 84.13])",
        "            for v in q:",
        "                ax.axvline(v, color=\"k\", lw=0.8, ls=\"-\" if v == q[1] else \"--\")",
        "            ax.set_title(f\"{names[a]}\\n{q[1]:.4g} +{q[2] - q[1]:.2g} -{q[1] - q[0]:.2g}\", fontsize=7)",
        "            ax.set_yticks([])",
        "        else:",
        "            ax.hist2d(S[:, b], S[:, a], bins=30, cmap=\"Blues\")",
        "        if i < n - 1:",
        "            ax.set_xticklabels([])",
        "        else:",
        "            ax.set_xlabel(names[b], fontsize=7)",
        "        if j > 0 or i == 0:",
        "            ax.set_yticklabels([])",
        "        else:",
        "            ax.set_ylabel(names[a], fontsize=7)",
        "        ax.tick_params(labelsize=6)",
        "plt.show()"])


# ------------------------------------------------------------------ Classification
def classification(s, k, opts):
    kind = opts.get("kind", "steps")
    have = s.ns.get("classes") is not None
    fit = s.wf_results() is not None and s.wf.get("fit_kind") not in ("classify", "sample")
    if fit:
        r, pre = _head(s, k)
        get = ["import fantasy_agn as fa", "", *pre,
               f"c = fa.classify_fit({r})                         # the final class, from the values of the fit"]
        if kind != "diagrams":
            get.append(f"steps = fa.classify({r}.spectrum)              # the preliminary classification: its steps")
    elif have:
        get = [f"c = classes[{int(k)}]"]
    else:
        get = ["import fantasy_agn as fa", "", "c = fa.classify(spec)"]
    draw = ("fig = c.plot()                                 # the BPT and WHAN diagrams" if kind == "diagrams"
            else f"fig = {'steps' if fit else 'c'}.plot_steps()                          # how the class was reached")
    return "\n".join([*get, "print(c.label)", draw, 'fig.savefig("classification.pdf")'])


# ------------------------------------------------------------------ Main sequence
def main_sequence(s, k, opts):
    kind = s.wf.get("fit_kind")
    zoom = "\"auto\"" if str(opts.get("zoom", "1")) != "0" else "None"
    if kind == "joint":
        obj, many = "me.epochs", True
    elif kind == "fits":
        obj, many = "[f for f in fits if f is not None]", True
    else:
        obj, many = "[r]", False
    return "\n".join([
        "from fantasy_agn.plotting import plot_main_sequence", "",
        f"results = {obj}",
        "for m in (f.main_sequence() for f in results):   # FWHM(Hβ), R_FeII, population and spectral type",
        "    print(m)",
        "fig = plot_main_sequence(results" + (", labels=[f\"#{i}\" for i in range(len(results))]" if many else "") +
        (f", zoom={zoom}" if many else "") + ")",
        'fig.savefig("main_sequence.pdf")'])


# ------------------------------------------------------------------ Light curves and lags
def time(s, k, opts):
    a, b = opts.get("a") or "", opts.get("b") or ""
    ok = re.fullmatch(r"[A-Za-z0-9_\[\]\-.]+", a or "-") and re.fullmatch(r"[A-Za-z0-9_\[\]\-.]+", b or "-")
    if not (a and b and ok):
        return "\n".join(["t = me.light_curve()        # flux, FWHM and EW of the broad lines and the continuum per epoch",
                          "fig = me.plot_light_curves()", "t"])
    src = lambda c: ("w" if c.startswith("win") else "t")   # noqa: E731
    lines = ["import numpy as np", "import matplotlib.pyplot as plt", "from fantasy_agn import rm", "",
             "t = me.light_curve()                             # per epoch: the fitted line fluxes, widths and continuum"]
    if a.startswith("win") or b.startswith("win"):
        lines.append("w = me.window_continuum()                      # the continuum measured in line-free windows")
    if "feii" in (a, b):
        return "\n".join(lines + ["# the Fe II light curve is measured by the app from each epoch's r.feii():",
                                       "fe = [e.feii() for e in me.epochs]",
                                       "t[\"feii\"], t[\"feii_err\"] = [d[\"flux\"] for d in fe], [d[\"flux_err\"] for d in fe]"]
                         + _lag_lines(a, b, "t", "t"))
    return "\n".join(lines + _lag_lines(a, b, src(a), src(b)))


def _lag_lines(a, b, sa, sb):
    return [
        f"mjd = t[\"mjd\"].to_numpy(float)",
        f"fa_, ea = {sa}[{a!r}].to_numpy(float), {sa}[{a + '_err'!r}].to_numpy(float)   # driver",
        f"fb, eb = {sb}[{b!r}].to_numpy(float), {sb}[{b + '_err'!r}].to_numpy(float)   # responder",
        "ok = np.isfinite(mjd + fa_ + ea + fb + eb) & (ea > 0) & (eb > 0)",
        "o = np.argsort(mjd[ok])",
        "tt, fa_, ea, fb, eb = (v[ok][o] for v in (mjd, fa_, ea, fb, eb))",
        "half = 0.5 * (tt.max() - tt.min())",
        "lag = rm.ccf_lag(tt, fa_, ea, tt, fb, eb, lag_range=(-half, half), n_mc=300)   # ICCF; also rm.vn_lag, rm.zdcf_lag, rm.gp_lag",
        "print(f\"lag {lag['tau_cen']:.1f} +{lag['tau_err_hi']:.1f} -{lag['tau_err_lo']:.1f} d\")",
        "",
        "fig, (ax, cx) = plt.subplots(2, 1, figsize=(9, 6))",
        "pct = lambda f: 100 * (f / f.mean() - 1)",
        f"ax.errorbar(tt, pct(fa_), 100 * ea / fa_.mean(), fmt=\"o-\", ms=3, label={a!r})",
        f"ax.errorbar(tt, pct(fb), 100 * eb / fb.mean(), fmt=\"o-\", ms=3, label={b!r})",
        f"ax.plot(tt - lag[\"tau_cen\"], pct(fb), \"--\", color=\"C1\", alpha=0.6, label={b!r} + \" shifted back by τ\")",
        "ax.set_xlabel(\"MJD\")",
        "ax.set_ylabel(\"% change from the mean\")",
        "ax.legend(fontsize=8)",
        "cx.plot(lag[\"lags\"], lag[\"r\"], color=\"k\")",
        "cx.axvline(lag[\"tau_cen\"], color=\"C3\")",
        "cx.set_xlabel(\"τ (days)\")",
        "cx.set_ylabel(\"ICCF r\")",
        "fig.tight_layout()",
        "plt.show()"]


# ------------------------------------------------------------------ Checks
def checks(s, k, opts):
    from ..fitcompare import FitComparison
    from ..robustness import Robustness
    out = []
    if s.wf_results() is not None:
        kind = s.wf.get("fit_kind")
        obj = {"joint": "me", "fits": "fits"}.get(kind, "r")
        out += ["import fantasy_agn as fa", "", f"res = fa.analyze_residuals({obj})        # features the model misses",
                "print(res)"]
    elif s.wf.get("fit_kind") == "sample" and s.sample is not None:
        out += ["import fantasy_agn as fa", "", f"r = sample.load({int(k or 0)})",
                "res = fa.analyze_residuals(r)        # features the model misses",
                "lines = fa.suggest_lines(r)          # lines to add, or to separate from their component",
                "print(res)", "print(lines)"]
    if isinstance(s.ns.get("rob"), Robustness):
        out += ["", "fig = rob.plot()                                # the scans of fa.robustness(r)"]
    if isinstance(s.ns.get("cmp"), FitComparison):
        out += ["", "fig = cmp.plot()                                # the two fits compared: fa.compare_fits"]
    return "\n".join(out) or "# fit first"


# ------------------------------------------------------------------ Line profiles
def lines(s, k, opts):
    r, pre = _head(s, k)
    safe = lambda n: re.sub(r"[^\w\[\]\.\-]", "", str(n))
    xmin, xmax = _num(opts, "xmin"), _num(opts, "xmax")
    if opts.get("mode") == "cmp":
        names = [safe(n) for n in _names(opts, "lines")][:8] or ["Hb"]
        out = ["import matplotlib.pyplot as plt", "", *pre,
               f"lines = {names!r}",
               "for ln in lines:                                 # the broad profile of each line (or the whole line)",
               f"    broad = [c.name for c in {r}._line_components() if c.role == 'broad' and ln in c.line_names]",
               f"    m = {r}.measure(ln, broad or None)",
               "    print(f\"{ln:10s} FWHM {m['fwhm']:7.0f} ± {m['fwhm_err']:4.0f}   sigma {m['sigma']:6.0f} ± {m['sigma_err']:4.0f}"
               "   shift {m['shift']:+6.0f} km/s\")",
               "# scaled to their peaks on one velocity axis, the half-maximum points marked: the broad profiles",
               "# when every line has broad components, else the whole lines",
               f"which = 'broad' if all(any(c.role == 'broad' and ln in c.line_names for c in {r}._line_components()) "
               "for ln in lines) else 'all'",
               f"fig = {r}.plot_profiles(lines, which=which, normalize=True)"]
        if xmin is not None and xmax is not None:
            out.append(f"fig.axes[0].set_xlim({xmin:.0f}, {xmax:.0f})")
        return "\n".join(out + ["plt.show()"])
    if opts.get("mode") == "ep" and len(s.wf_results() or []) > 1:
        obj = "me.epochs" if s.wf.get("fit_kind") == "joint" else "fits"
        line = safe(opts.get("line") or "Hb")
        out = ["import matplotlib.pyplot as plt", "from fantasy_agn.plotting import plot_profiles", "",
               f"line = {line!r}",
               f"for k, e in enumerate({obj}):                  # the line in every epoch: broad profile, else the whole line",
               "    broad = [c.name for c in e._line_components() if c.role == 'broad' and line in c.line_names]",
               "    m = e.measure(line, broad or None)",
               "    print(f\"#{k} MJD {e.spectrum.mjd:.0f}  FWHM {m['fwhm']:6.0f} ± {m['fwhm_err']:4.0f}  flux {m['flux']:.4g} ± {m['flux_err']:.2g}"
               "  shift {m['shift']:+5.0f} km/s\")",
               "# the profiles of all epochs on one velocity axis",
               f"which = 'broad' if all(any(c.role == 'broad' and line in c.line_names for c in e._line_components()) for e in {obj}) else 'all'",
               f"fig = plot_profiles({obj}, line, which=which, normalize={opts.get('norm') == '1'})",
               "",
               "# the mean and rms profiles over the epochs, on one velocity grid, with FWHM and line dispersion",
               "import numpy as np",
               "from fantasy_agn.lines import line_wavelength",
               "lam0, v = line_wavelength(line), np.linspace(-15000, 15000, 1201)",
               "prof = []",
               f"for e in {obj}:",
               "    parts = e.model_flux(components=True)",
               "    names = [c.name for c in e._line_components() if line in c.line_names and (which == 'all' or c.role == 'broad')]",
               "    vel = (e.spectrum.wave / lam0 - 1) * 299792.458",
               "    prof.append(np.interp(v, vel, sum(parts[n] for n in names), left=np.nan, right=np.nan))",
               "mean, rms = np.nanmean(prof, axis=0), np.nanstd(prof, axis=0, ddof=1)",
               "for name, y in (('mean', mean), ('rms', rms)):",
               "    y = np.nan_to_num(y).clip(0)",
               "    up = v[y >= y.max() / 2]",
               "    c = np.trapezoid(y * v, v) / np.trapezoid(y, v)",
               "    sigma = np.sqrt(np.trapezoid(y * (v - c) ** 2, v) / np.trapezoid(y, v))",
               "    print(f'{name:4s}  FWHM {up[-1] - up[0]:6.0f}  sigma_line {sigma:6.0f} km/s')",
               "fig2, ax = plt.subplots()",
               "ax.plot(v, mean, 'k--', label='mean')",
               "ax.plot(v, rms, color='#d0247f', lw=2, label='rms')",
               "ax.set_xlabel('velocity (km/s)'); ax.set_ylabel('flux'); ax.legend()"]
        if xmin is not None and xmax is not None:
            out.append(f"for f in (fig, fig2):\n    f.axes[0].set_xlim({xmin:.0f}, {xmax:.0f})")
        return "\n".join(out + ["plt.show()"])
    line = safe(opts.get("line") or "Hb")
    out = ["import matplotlib.pyplot as plt", "", *pre,
           f"line = {line!r}",
           f"broad = [c.name for c in {r}._line_components() if c.role == 'broad' and line in c.line_names]",
           f"m = {r}.measure(line, broad or None)             # of the summed broad components, else of the whole line",
           "for key in ('fwhm', 'sigma', 'shift', 'flux', 'ew'):",
           "    print(f\"{key:6s} {m[key]:.4g} ± {m[key + '_err']:.2g}\")",
           f"print({r}.lines[{r}.lines.line == line])          # each component of the line",
           "# the line with everything else subtracted, its components, the FWHM and the residuals",
           f"fig = {r}.plot_line(line, residuals=True, info=True, mark_fwhm=True)"]
    if xmin is not None and xmax is not None:
        out.append(f"for ax in fig.axes:\n    ax.set_xlim({xmin:.0f}, {xmax:.0f})")
    return "\n".join(out + ["plt.show()"])


VIEWS = {"lines": lines, "fit": fit, "anatomy": anatomy, "feii": feii, "degen": degen, "atlas": atlas, "posterior": posterior,
         "class": classification, "ms": main_sequence, "time": time, "checks": checks}


def view_code(s, view, k=0, opts=None):
    if view not in VIEWS:
        raise ValueError(f"no code for the view {view!r}")
    return VIEWS[view](s, int(k or 0), dict(opts or {}))
